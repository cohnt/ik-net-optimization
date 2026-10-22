"""`src/svgd/batched_analytic.py` against the pydrake closed forms it transcribes.

THE PYDRAKE `Solve` IS THE ORACLE, composed exactly as the programs compose it
(`AnalyticProgramMixin._Solve`: `RigidTransform(RollPitchYaw(rpy), xyz) @ pose_offset`, then
the robot's `_AnalyticSolve` on the program's branch). No scene and no network: the map is
pure kinematics, so this file builds a stand-in program (`_Fake`) carrying exactly the
attributes the mixin's `_Solve` and `BatchedAnalyticMap.from_program` read, with a random
pose offset so the composition is exercised. Pinned, for both robots and EVERY branch (the
iiwa's 8 global configurations, the Panda's 8 including the mirrored elbow A = -1, and the
Panda's 4-branch 2-element gc):

  1. `rpy_to_R` against `RollPitchYaw(rpy).ToRotationMatrix()`;
  2. values: q (wrapped difference -- atan2's branch cut at +-pi is a discontinuity of the map
     itself, so a last-ulp flip there is counted, not failed) and the four pre-clip arguments
     u, float64, <= 1e-12, at in-reach poses (the FK of a random configuration, on every
     branch) and at OUT-of-reach poses (the clipped regime);
  3. the autograd Jacobian `d(q, u)/dX` against the AutoDiffXd chain, <= 1e-9 relative, in
     both regimes -- and where pydrake's clip is active, the clipped joint's derivative is
     zero in both;
  4. the Panda's q6 wrap: the seam moved to either side of a sample's q6 (pydrake reads the
     module constant at call time, the twin its tensor), values agree and differ by 2 pi
     between the two sides, derivative unchanged;
  5. float32 values against the float64 oracle;
  6. `iiwa_psi` against `Analytic_IK_7DoF.psi`.

Script-style like the other tests: `.venv/bin/python tests/test_batched_analytic.py`, alone,
under the memory scope; prints ok / FAIL per check and exits nonzero on any failure.
"""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.all import AutoDiffXd, RigidTransform, RollPitchYaw, RotationMatrix   # noqa: E402
from pydrake.autodiffutils import InitializeAutoDiff                                # noqa: E402

import src.panda_analytic_ik as panda_ik                                           # noqa: E402
from src.analytic_program import AnalyticProgramMixin                              # noqa: E402
from src.iiwa_analytic_ik import (Analytic_IK_7DoF, iiwa_limits_lower,             # noqa: E402
                                  iiwa_limits_upper)
from src.panda_analytic_ik import (Analytic_IK_Panda, panda_limits_lower,          # noqa: E402
                                   panda_limits_upper)
from src.svgd.batched_analytic import BatchedAnalyticMap, iiwa_psi, rpy_to_R       # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_STEPBACK = 1e-4
FAILURES = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"\n          {detail}"))
    if not cond:
        FAILURES.append(name)


class _Fake(AnalyticProgramMixin):
    """What `_Solve` and `BatchedAnalyticMap.from_program` read off a program, and nothing
    else: the robot's map, its branch, the frame offset and the clip option."""

    def __init__(self, robot, gc, pose_offset):
        self.robot = robot
        self.analytic_ik = Analytic_IK_Panda() if robot == "panda" else Analytic_IK_7DoF()
        self.gc = np.asarray(gc)
        self.pose_offset = pose_offset
        self.options = SimpleNamespace(analytic_clip_stepback=CLIP_STEPBACK)
        ## `_InitAnalytic`'s Python-reference backend: the oracle is the reference, never the C++.
        self._analytic_cpp = None

    def _AnalyticSolve(self, R, p, psi, gc, clip_stepback):
        ## The two robots' `_AnalyticSolve`, verbatim (src/panda_program.py, src/iiwa_program.py).
        if self.robot == "panda":
            return self.analytic_ik.Solve(R, p, psi, gc, clip_stepback=clip_stepback)
        return self.analytic_ik.Solve(R, p, gc, psi, clip_stepback=clip_stepback)

    def psi_of(self, q):
        return float(q[6]) if self.robot == "panda" else self.analytic_ik.psi(q)


def _branches(robot, n):
    if robot == "iiwa":
        return [tuple(b) for b in Analytic_IK_7DoF().BranchSet()]
    return [tuple(b) for b in Analytic_IK_Panda().BranchSet(n)]


def _limits(robot):
    if robot == "panda":
        return panda_limits_lower, panda_limits_upper
    return iiwa_limits_lower, iiwa_limits_upper


def _random_offset(rng):
    rpy = rng.uniform(-math.pi, math.pi, 3)
    return RigidTransform(RollPitchYaw(rpy), rng.uniform(-0.2, 0.2, 3))


def _x_of(fake, q):
    """The decision vector whose map pose is the analytic FK of q: `X_WF = FK(q) @ X_FA^-1`."""
    X_WA = RigidTransform(np.asarray(fake.analytic_ik.FK(q), dtype=float))
    X_WF = X_WA @ fake.pose_offset.inverse()
    return np.concatenate([X_WF.translation(), X_WF.rotation().ToRollPitchYaw().vector(),
                           [fake.psi_of(q)]])


def _oracle(fake, x):
    q, u = fake._Solve(np.asarray(x, dtype=float), CLIP_STEPBACK)
    return np.asarray(q, dtype=float), np.asarray(u, dtype=float)


def _oracle_jac(fake, x):
    q, u = fake._Solve(InitializeAutoDiff(np.asarray(x, dtype=float)).reshape(-1), CLIP_STEPBACK)
    rows = list(q) + list(u)
    return np.stack([r.derivatives() if isinstance(r, AutoDiffXd) and r.derivatives().size == 7
                     else np.zeros(7) for r in rows])


def _signs(amap, gcs, dtype):
    kw = dict(dtype=dtype, device=DEVICE)
    per = [amap.branch_signs(g, kw) for g in gcs]
    return tuple(torch.stack([p[k] for p in per]) for k in range(3))


def _wrapdiff(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def _cases(robot, n_branches, rng, n_q, far=False):
    """`(fakes, X, gcs)`: every branch at each of `n_q` random configurations' poses, one
    shared random offset. `far`: the pose's translation pushed out of reach (clipped)."""
    lo, hi = _limits(robot)
    offset = _random_offset(rng)
    branches = _branches(robot, n_branches)
    base = _Fake(robot, branches[0], offset)
    xs, gcs = [], []
    for _ in range(n_q):
        x = _x_of(base, rng.uniform(lo, hi))
        if far:
            x[:3] = x[:3] * 2.5 + np.array([0.0, 0.0, 0.5])
        for b in branches:
            xs.append(x)
            gcs.append(b)
    return base, np.stack(xs), gcs


def test_rpy_to_R():
    rng = np.random.default_rng(0)
    rpy = rng.uniform(-math.pi, math.pi, size=(256, 3))
    got = rpy_to_R(torch.tensor(rpy, dtype=torch.float64, device=DEVICE)).cpu().numpy()
    ref = np.stack([RollPitchYaw(r).ToRotationMatrix().matrix() for r in rpy])
    err = float(np.abs(got - ref).max())
    check(f"rpy_to_R vs RollPitchYaw.ToRotationMatrix ({err:.1e})", err <= 1e-14, err)


def _value_and_jacobian(robot, n_branches, far, rng, n_q, n_jac_per_branch):
    base, X_np, gcs = _cases(robot, n_branches, rng, n_q, far=far)
    amap = BatchedAnalyticMap.from_program(base, dtype=torch.float64, device=DEVICE)
    X = torch.tensor(X_np, dtype=torch.float64, device=DEVICE)
    signs = _signs(amap, gcs, torch.float64)
    q, u = amap(X, signs)
    q, u = q.cpu().numpy(), u.cpu().numpy()
    ref = [_oracle(_Fake(robot, g, base.pose_offset), x) for x, g in zip(X_np, gcs)]
    q_ref = np.stack([r[0] for r in ref])
    u_ref = np.stack([r[1] for r in ref])
    dq = _wrapdiff(q, q_ref)
    flips = int((np.abs(q - q_ref) > 1.0).sum())
    du = np.abs(u - u_ref)
    clip = 1 - CLIP_STEPBACK
    n_clipped = int((np.abs(u_ref) > clip).any(axis=1).sum())
    label = f"{robot} {len(_branches(robot, n_branches))}-branch{' far' if far else ''}"
    check(f"{label}: values q {dq.max():.1e} (wrapped; {flips} atan2 cut flips), u {du.max():.1e} "
          f"on {len(gcs)} cells ({n_clipped} clipped)",
          dq.max() <= 1e-12 and du.max() <= 1e-12 * max(1.0, np.abs(u_ref).max()),
          (dq.max(), du.max()))
    if far:
        check(f"{label}: the far poses are in the clipped regime ({n_clipped}/{len(gcs)})",
              n_clipped >= 0.9 * len(gcs), n_clipped)

    ## Jacobians: n_jac_per_branch cells of every branch.
    worst, worst_zero, n_zero = 0.0, 0.0, 0
    branches = _branches(robot, n_branches)
    for b in branches:
        idx = [i for i, g in enumerate(gcs) if g == b][:n_jac_per_branch]
        for i in idx:
            x = torch.tensor(X_np[i], dtype=torch.float64, device=DEVICE)
            s = tuple(t[i:i + 1] for t in signs)
            J = torch.func.jacrev(lambda v: torch.cat(amap(v[None], s), dim=1)[0])(x).cpu().numpy()
            J_ref = _oracle_jac(_Fake(robot, b, base.pose_offset), X_np[i])
            worst = max(worst, float(np.abs(J - J_ref).max() / max(1.0, np.abs(J_ref).max())))
            if not np.isfinite(J).all():
                worst = math.inf
            ## Where the clip is active, the joints computed from that argument carry no
            ## derivative through it -- check the iiwa's q4 (EQ 4) and the Panda's q2 (eq 28).
            j_clip = {"iiwa": (0, 3), "panda": (3, 1)}[robot]
            if abs(u_ref[i][j_clip[0]]) > clip:
                n_zero += 1
                if robot == "iiwa":
                    worst_zero = max(worst_zero, float(np.abs(J[j_clip[1]]).max()))
    check(f"{label}: d(q, u)/dX vs AutoDiffXd, {n_jac_per_branch} cells x {len(branches)} "
          f"branches, rel {worst:.1e}", worst <= 1e-9, worst)
    if far and robot == "iiwa":
        check(f"{label}: clipped elbow q4 carries zero derivative ({n_zero} cells, max "
              f"{worst_zero:.1e})", n_zero > 0 and worst_zero == 0.0, (n_zero, worst_zero))


def test_values_and_jacobians_every_branch():
    rng = np.random.default_rng(1)
    for robot, nb in (("iiwa", 8), ("panda", 8), ("panda", 4)):
        _value_and_jacobian(robot, nb, False, rng, 48, 6)
        _value_and_jacobian(robot, nb, True, rng, 16, 3)


def test_from_program_uses_the_programs_branch_and_offset():
    """The map built from a program, called WITHOUT explicit signs, is that program's map."""
    rng = np.random.default_rng(2)
    for robot, nb in (("iiwa", 8), ("panda", 8), ("panda", 4)):
        lo, hi = _limits(robot)
        offset = _random_offset(rng)
        worst = 0.0
        for b in _branches(robot, nb):
            fake = _Fake(robot, b, offset)
            amap = BatchedAnalyticMap.from_program(fake, dtype=torch.float64, device=DEVICE)
            X_np = np.stack([_x_of(fake, rng.uniform(lo, hi)) for _ in range(8)])
            q, u = amap(torch.tensor(X_np, dtype=torch.float64, device=DEVICE))
            for i, x in enumerate(X_np):
                q_ref, u_ref = _oracle(fake, x)
                worst = max(worst, float(_wrapdiff(q[i].cpu().numpy(), q_ref).max()),
                            float(np.abs(u[i].cpu().numpy() - u_ref).max()))
        check(f"{robot} {nb}-branch from_program: the program's own gc and offset ({worst:.1e})",
              worst <= 1e-12, worst)


def test_panda_q6_seam():
    rng = np.random.default_rng(3)
    lo, hi = _limits("panda")
    offset = _random_offset(rng)
    saved = panda_ik.Q6_WRAP_SEAM
    worst, worst_j, jumps = 0.0, 0.0, []
    try:
        for b in _branches("panda", 8):
            fake = _Fake("panda", b, offset)
            amap = BatchedAnalyticMap.from_program(fake, dtype=torch.float64, device=DEVICE)
            x = _x_of(fake, rng.uniform(lo, hi))
            q6 = _oracle(fake, x)[0][5]
            vals = []
            for side in (-1e-7, 1e-7):
                seam = q6 + side - 2 * math.pi          # q6 sits just above / below a seam
                panda_ik.Q6_WRAP_SEAM = seam
                amap.seam = torch.tensor(seam, dtype=torch.float64, device=DEVICE)
                q_ref, _ = _oracle(fake, x)
                X = torch.tensor(x[None], dtype=torch.float64, device=DEVICE)
                q_t = amap(X)[0][0].cpu().numpy()
                worst = max(worst, float(np.abs(q_t - q_ref).max()))
                J = torch.func.jacrev(lambda v: amap(v[None])[0][0])(X[0]).cpu().numpy()
                J_ref = _oracle_jac(fake, x)[:7]
                worst_j = max(worst_j, float(np.abs(J - J_ref).max() / max(1.0, np.abs(J_ref).max())))
                vals.append((q_t[5], J[5]))
            jumps.append((vals[0][0] - vals[1][0], float(np.abs(vals[0][1] - vals[1][1]).max())))
    finally:
        panda_ik.Q6_WRAP_SEAM = saved
    jump_ok = all(abs(abs(d) - 2 * math.pi) < 1e-9 and dj == 0.0 for d, dj in jumps)
    check(f"panda q6 seam: values {worst:.1e} (unwrapped: the seam is pinned to the same side), "
          f"jacobian rel {worst_j:.1e}, q6 jumps by 2pi across the seam with its derivative "
          f"unchanged on all 8 branches", worst <= 1e-12 and worst_j <= 1e-9 and jump_ok,
          (worst, worst_j, jumps))


def test_float32():
    rng = np.random.default_rng(4)
    for robot, nb in (("iiwa", 8), ("panda", 8)):
        base, X_np, gcs = _cases(robot, nb, rng, 32)
        amap = BatchedAnalyticMap.from_program(base, dtype=torch.float32, device=DEVICE)
        X = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
        q, u = amap(X, _signs(amap, gcs, torch.float32))
        ref = [_oracle(_Fake(robot, g, base.pose_offset), x) for x, g in zip(X_np, gcs)]
        dq = _wrapdiff(q.double().cpu().numpy(), np.stack([r[0] for r in ref]))
        du = np.abs(u.double().cpu().numpy() - np.stack([r[1] for r in ref]))
        med, p99 = float(np.median(dq.max(axis=1))), float(np.quantile(dq.max(axis=1), 0.99))
        check(f"{robot} float32: q median {med:.1e}, p99 {p99:.1e}, max {dq.max():.1e}; u max "
              f"{du.max():.1e}", med <= 1e-5 and du.max() <= 1e-4, (med, p99, dq.max(), du.max()))


def test_iiwa_psi():
    rng = np.random.default_rng(5)
    ik = Analytic_IK_7DoF()
    Q = rng.uniform(iiwa_limits_lower, iiwa_limits_upper, size=(256, 7))
    got = iiwa_psi(torch.tensor(Q, dtype=torch.float64, device=DEVICE), ik.alpha, ik.d).cpu().numpy()
    ref = np.array([ik.psi(q) for q in Q])
    err = float(_wrapdiff(got, ref).max())
    check(f"iiwa_psi vs Analytic_IK_7DoF.psi on 256 configurations ({err:.1e})", err <= 1e-10, err)
    ## And psi recovered from q reproduces q through the map on q's own branch, wherever the
    ## clip is inactive (inside the clip band the clipped map returns a nearby configuration
    ## by design -- `UnclippedStartError`).
    rt, n_used = 0.0, 0
    for q in Q[:64]:
        fake = _Fake("iiwa", ik.GC(q), RigidTransform())
        amap = BatchedAnalyticMap.from_program(fake, dtype=torch.float64, device=DEVICE)
        x = _x_of(fake, q)
        x[6] = float(amap.psi_of(torch.tensor(q[None], dtype=torch.float64, device=DEVICE))[0])
        q_back, u_back = amap(torch.tensor(x[None], dtype=torch.float64, device=DEVICE))
        if np.abs(u_back.cpu().numpy()).max() >= 1 - CLIP_STEPBACK:
            continue
        n_used += 1
        rt = max(rt, float(_wrapdiff(q_back[0].cpu().numpy(), q).max()))
    check(f"iiwa: map(FK(q), psi_of(q), GC(q)) == q off the clip band ({n_used}/64, {rt:.1e})",
          n_used >= 32 and rt <= 1e-8, (n_used, rt))


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    test_rpy_to_R()
    test_values_and_jacobians_every_branch()
    test_from_program_uses_the_programs_branch_and_offset()
    test_panda_q6_seam()
    test_float32()
    test_iiwa_psi()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
