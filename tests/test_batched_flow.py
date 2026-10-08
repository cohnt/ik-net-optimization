"""`src/svgd/batched_flow.py` against the program's own unbatched flow path.

The particle solvers evaluate the network for N particles at once through
`BatchedFlow`; the Drake programs evaluate it one iterate at a time through `VarsToQ`,
`CToPose7`, `InvertFlow` and `jacobian_gen`. Every claim the particle side makes about
a configuration rests on the two agreeing, so this file pins them against each other on
REAL learned programs (the adopted `n6` charts of both rigid arms), not on a toy network:

  * `q_from_vars` against `VarsToQ` on the plain float path: the network path is
    bit-identical at B = 1 given the same conditioning row, agrees at the cuBLAS
    reduction-order level at B = 64, and end to end the only difference is the last ulp of
    the rpy -> quaternion conversion -- which the iiwa chart's runaway population amplifies
    to 1e-9..1e-7, so that figure is RECORDED on every particle and asserted on the
    ordinary ones;
  * `conditioning` against `CToPose7` + the softflow zero, including the `w >= 0`
    canonicalisation Drake's `ToQuaternion` applies, exercised by rpy draws over the whole
    circle and by pitch next to +-pi/2;
  * `invert` against `InvertFlow`, and the round trip `q(c, invert(q, c), 0) == q` that
    the paired start relies on -- which has a ~1e-7 floor belonging to the shared network
    (a float32-computed `M_inv` in its first fixed linear layer), so it is pinned against
    the program's own round trip rather than against zero;
  * autograd's `dq/d[c6, z, q_c]` against the program's `jacobian_gen` (whose input is the
    7-vector conditioning, so its `dq/dc7` block is composed with `d quat / d rpy` from
    this module's own helper) and, independently of that helper, against central
    differences of `VarsToQ`;
  * the float32 path uses a PRIVATE copy: the shared network stays float64, the copy is
    cached, and float32 agrees with float64 to ~1e-4 on ordinary configurations -- and by
    RADIANS on the iiwa's runaway particles, recorded because a float32 swarm is the plan's
    default;
  * a timing table (printed, nothing asserted) for forward and forward + backward
    (`autograd.grad` w.r.t. the particles, which is what a solver step does).

No pytest config in this repo, so these are plain `test_*` functions with a `__main__`
driver, as `tests/test_screw_arm_kinematics.py` does. Loading a program takes seconds, so
one program per robot is built once and reused by every test.

    .venv/bin/python tests/test_batched_flow.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.generic_program import ProgramOptions                       # noqa: E402
from src.iiwa_program import Iiwa14IKProgram                          # noqa: E402
from src.panda_program import PandaIKProgram                          # noqa: E402
from src.svgd.batched_flow import (BatchedFlow, _PRIVATE_COPIES,      # noqa: E402
                                   quat_from_rpy_batched)
from src.target_screening import SceneFile                            # noqa: E402
from src.utils import BuildEnv, HiddenPrints, RepoDir                 # noqa: E402

REPO = RepoDir()
CHECKPOINTS = {
    "panda": os.path.join(REPO, "models/panda/panda__n6__step620000.pkl"),
    "iiwa": os.path.join(REPO, "models/iiwa14/iiwa14__n6__step620000.pkl"),
}
ROBOTS = ("panda", "iiwa")
_PROGRAMS = {}


def program(robot):
    """One learned pose program per robot, built once, at a reachable target."""
    if robot not in _PROGRAMS:
        options = ProgramOptions()
        with HiddenPrints():
            diagram = BuildEnv(meshcat=None, directives_file=SceneFile(robot, "pose"))
            cls = PandaIKProgram if robot == "panda" else Iiwa14IKProgram
            p = cls(diagram, options=options, checkpoint=CHECKPOINTS[robot])
            # The target is the flow-frame pose of a real configuration, the way the
            # benchmark makes them, so the initial guess for `c` is an ordinary pose.
            rng = np.random.default_rng(7)
            q = rng.uniform(*p.ConfigLimits())
            translation, wxyz = p.fk(q)
            p.create_prog(np.concatenate([translation, wxyz]))
        _PROGRAMS[robot] = p
    return _PROGRAMS[robot]


def device_of(p):
    return next(p.ik_solver.nn_model.parameters()).device


def lumped_batch(p, rng, B):
    """`[c6 | z | q_c]` rows: c near the program's initial guess, z ~ N(0, I), q_c ~ U(+-0.1)."""
    width, ndof = p.ik_solver.network_width, p.num_arm_dof
    c0 = p.prog.GetInitialGuess(p.c)
    c = c0[None, :] + rng.uniform(-0.3, 0.3, size=(B, 6))
    z = rng.standard_normal((B, width))
    qc = rng.uniform(-0.1, 0.1, size=(B, ndof))
    return np.concatenate([c, z, qc], axis=1)


def flow_frame_c6(p, q):
    """The conditioning pose of configuration `q`, as `SetStartFromQ` forms it."""
    p.plant.SetPositions(p.plant_context, p.ConfigToPlantQ(q))
    pose = p.FlowPoseInWorld()
    return np.concatenate([pose.translation(), pose.rotation().ToRollPitchYaw().vector()])


def test_q_from_vars_matches_VarsToQ():
    """Three agreements, kept apart because they fail for different reasons.

    (a) The NETWORK path, handed Drake's own conditioning row: bit-identical at B = 1
        (the same kernels on the same inputs), and at the cuBLAS reduction-order level at
        B = 64 -- which the runaway particles amplify too, so that figure is asserted on
        the ordinary ones. This isolates the batched call from the rpy -> quaternion
        conversion.
    (b) End to end (`q_from_vars` against `VarsToQ`) on ORDINARY particles, |q|_inf < 100.
    (c) End to end on every particle, recorded rather than asserted: the conversion agrees
        with Drake to ~3e-16, and on the iiwa's runaway population (CLAUDE.md, the gain
        ceiling: |q| ~ 1e5 with |dq/dc| ~ 1e6 on some N(0, I) latents) the chart multiplies
        that last ulp into 1e-9..1e-7. That is the chart's sensitivity, not a disagreement
        between the two implementations -- (a) shows the network path exact there too.
    """
    rng = np.random.default_rng(0)
    for robot in ROBOTS:
        p = program(robot)
        bf = BatchedFlow.from_program(p)
        assert bf.model is p.ik_solver.nn_model, "float64 must use the shared network"
        assert bf.dtype == torch.float64
        width, ndof, dev = p.ik_solver.network_width, p.num_arm_dof, device_of(p)
        X_np = lumped_batch(p, rng, 64)
        expected = np.stack([p.VarsToQ(x)[:ndof] for x in X_np])
        X = torch.tensor(X_np, dtype=torch.float64, device=dev)

        # (a) the network alone, conditioned on Drake's conversion of the same c6.
        cond_drake = torch.tensor(
            np.stack([np.concatenate([p.CToPose7(x[:6]), [0.0]]) for x in X_np]),
            dtype=torch.float64, device=dev)
        z, qc = X[:, 6:6 + width], X[:, 6 + width:6 + width + ndof]
        with torch.no_grad():
            net1 = torch.cat([bf.model(z[i:i + 1], c=cond_drake[i:i + 1], rev=True)[0]
                              for i in range(64)])[:, :ndof] + qc
            net64 = bf.model(z, c=cond_drake, rev=True)[0][:, :ndof] + qc
        ordinary = np.abs(expected).max(axis=1) < 100.0
        net_err1 = np.abs(net1.cpu().numpy() - expected).max()
        net_diff64 = np.abs(net64.cpu().numpy() - expected)
        net_err64, net_all64 = net_diff64[ordinary].max(), net_diff64.max()

        # (b), (c) end to end.
        single = torch.cat([bf.q_from_vars(X[i:i + 1])
                            for i in range(64)]).detach().cpu().numpy()
        batched = bf.q_from_vars(X).detach().cpu().numpy()
        err1 = np.abs(single - expected)[ordinary].max()
        err64 = np.abs(batched - expected)[ordinary].max()
        all1 = np.abs(single - expected).max()
        all64 = np.abs(batched - expected).max()
        n_runaway = int((~ordinary).sum())
        worst_q = np.abs(expected).max()
        print(f"  [{robot}] network path with Drake's conditioning vs VarsToQ: "
              f"max |diff| B=1 {net_err1:.3e}, B=64 {net_err64:.3e} on ordinary particles "
              f"({net_all64:.3e} over all 64)")
        print(f"  [{robot}] q_from_vars vs VarsToQ on {int(ordinary.sum())}/64 ordinary "
              f"particles: max |diff| B=1 {err1:.3e}, B=64 {err64:.3e}; over all 64: "
              f"B=1 {all1:.3e}, B=64 {all64:.3e} ({n_runaway} particles at |q|_inf >= 100, "
              f"max |q|_inf {worst_q:.3g})")
        assert net_err1 == 0.0, (robot, net_err1)
        assert net_err64 <= 1e-8, (robot, net_err64)
        assert err1 <= 1e-10, (robot, err1)
        assert err64 <= 1e-8, (robot, err64)

        # Extra trailing columns (the `lift_q` block) are ignored, as VarsToQ ignores them.
        X_lifted = torch.cat([X, torch.ones((64, ndof), dtype=X.dtype, device=X.device)], 1)
        assert torch.equal(bf.q_from_vars(X_lifted), bf.q_from_vars(X))


def test_conditioning_matches_CToPose7():
    rng = np.random.default_rng(1)
    for robot in ROBOTS:
        p = program(robot)
        bf = BatchedFlow.from_program(p)
        # rpy over the whole circle (w < 0 on roughly half the draws before
        # canonicalisation), plus pitch next to +-pi/2 with random yaw sign.
        c = np.concatenate([
            np.concatenate([rng.uniform(-1, 1, (200, 3)),
                            rng.uniform(-np.pi, np.pi, (200, 3))], axis=1),
            np.concatenate([rng.uniform(-1, 1, (40, 3)),
                            rng.uniform(-np.pi, np.pi, (40, 1)),
                            rng.choice([-1, 1], (40, 1)) * (np.pi / 2 - 1e-3),
                            rng.uniform(-np.pi, np.pi, (40, 1))], axis=1),
        ])
        expected = np.stack([np.concatenate([p.CToPose7(ci), [0.0]]) for ci in c])
        got = bf.conditioning(torch.tensor(c, dtype=torch.float64,
                                           device=device_of(p))).cpu().numpy()
        err = np.abs(got - expected).max()
        n_flipped = int((expected[:, 3] < 0).sum())
        assert n_flipped == 0, "Drake's ToQuaternion returned w < 0; the premise is wrong"
        assert (got[:, 3] >= 0).all()
        print(f"  [{robot}] conditioning vs CToPose7: max |diff| {err:.3e} over {len(c)} "
              f"poses (all w >= 0)")
        assert err <= 1e-14, (robot, err)
        # The direct formula and Drake agree, so the canonicalisation really was exercised:
        # the uncanonicalised w must be negative on a good fraction of the full-circle draws.
        half = 0.5 * c[:200, 3:6]
        w_raw = (np.cos(half[:, 0]) * np.cos(half[:, 1]) * np.cos(half[:, 2])
                 + np.sin(half[:, 0]) * np.sin(half[:, 1]) * np.sin(half[:, 2]))
        assert (w_raw < 0).mean() > 0.2, "the draw did not exercise canonicalisation"


def test_invert_matches_InvertFlow_and_round_trips():
    """`invert` against `InvertFlow`, and the round trip the paired start relies on.

    The round trip `q(c, invert(q, c), 0) == q` is NOT exact to float64 on these charts,
    and the reason is the shared network, not this module: the flow's first
    `FixedLinearTransform` stores an `M_inv` that was computed in float32 at construction,
    so `M @ M_inv - I` is ~3e-8 even after `ConfigureNetworkDtype` casts both to float64.
    The program's own `InvertFlow` -> `ik_inference` round trip therefore sits at a
    ~1e-7 floor (measured 8e-8 Panda, 4e-8 iiwa, flat across |z| and across the local
    gain -- a constant, not a sensitivity), and that is the `start_q_error` the paired
    start records on the learned arm. So the round trip is asserted to EQUAL the program's
    own, and the floor itself is printed and bounded loosely.
    """
    rng = np.random.default_rng(2)
    for robot in ROBOTS:
        p = program(robot)
        bf = BatchedFlow.from_program(p)
        width, ndof, dev = p.ik_solver.network_width, p.num_arm_dof, device_of(p)
        B = 32
        lower, upper = p.ConfigLimits()
        q = rng.uniform(lower[:ndof], upper[:ndof], size=(B, ndof))
        c6 = np.stack([flow_frame_c6(p, qi) for qi in q])
        expected_z = np.stack([p.InvertFlow(qi, ci) for qi, ci in zip(q, c6)])
        # The program's own round trip, through the same path `SetStartFromQ` measures.
        rt_program = np.stack([
            p.ik_inference(np.concatenate([p.CToPose7(ci), zi, np.zeros(ndof)])
                           ).detach().cpu().numpy() - qi
            for qi, ci, zi in zip(q, c6, expected_z)])

        q_t = torch.tensor(q, dtype=torch.float64, device=dev)
        c_t = torch.tensor(c6, dtype=torch.float64, device=dev)
        z = bf.invert(q_t, c_t)
        assert not z.requires_grad
        err_z = np.abs(z.cpu().numpy() - expected_z).max()
        q_back = bf.q(c_t, z, torch.zeros((B, ndof), dtype=torch.float64, device=dev))
        rt_mine = (q_back - q_t).detach().cpu().numpy()
        err_rt = np.abs(rt_mine).max()
        err_vs_program = np.abs(rt_mine - rt_program).max()
        print(f"  [{robot}] invert vs InvertFlow: max |diff| {err_z:.3e}; round trip "
              f"max |q(c, invert(q, c), 0) - q| {err_rt:.3e} (the program's own: "
              f"{np.abs(rt_program).max():.3e}; |mine - program's| {err_vs_program:.3e}); "
              f"median |z| {np.median(np.linalg.norm(expected_z, axis=1)):.2f}")
        assert err_z <= 1e-9, (robot, err_z)
        assert err_vs_program <= 1e-12, (robot, err_vs_program)
        assert err_rt <= 1e-6, (robot, err_rt)


def test_gradients_match_jacobian_gen():
    rng = np.random.default_rng(3)
    for robot in ROBOTS:
        p = program(robot)
        bf = BatchedFlow.from_program(p)
        width, ndof, dev = p.ik_solver.network_width, p.num_arm_dof, device_of(p)
        worst_rel, worst_fd, n_used, n_skipped = 0.0, 0.0, 0, 0
        for x_np in lumped_batch(p, rng, 16):
            # Ordinary particles only: on a runaway one (|q| ~ 1e5, |dq/dc| ~ 1e6) a
            # central difference at h = 1e-6 measures the chart's curvature, not the
            # derivative, and says nothing about either implementation.
            if np.abs(p.VarsToQ(x_np)[:ndof]).max() >= 100.0 or n_used >= 8:
                n_skipped += 1
                continue
            n_used += 1
            x = torch.tensor(x_np, dtype=torch.float64, device=dev)
            J6 = torch.func.jacrev(lambda v: bf.q_from_vars(v[None])[0])(x)   # [ndof, 6+w+n]

            # The program's Jacobian is against the 7-vector conditioning. Compose its
            # dq/dc7 block with d quat / d rpy to land on the program's variables.
            c7 = torch.cat([x[:3], quat_from_rpy_batched(x[None, 3:6])[0]])
            vars7 = torch.cat([c7, x[6:]])
            J7, q7 = p.jacobian_gen(vars7)
            dquat_drpy = torch.func.jacrev(lambda r: quat_from_rpy_batched(r[None])[0])(x[3:6])
            expected = torch.cat([J7[:, :3], J7[:, 3:7] @ dquat_drpy, J7[:, 7:]], dim=1)
            rel = ((J6 - expected).abs().max() / expected.abs().max()).item()
            worst_rel = max(worst_rel, rel)
            assert J7[:, 7 + width:].allclose(
                torch.eye(ndof, dtype=torch.float64, device=dev)), "correction block"

            # And dq/dc6 against central differences of VarsToQ itself, so the rpy chain is
            # checked against Drake's templated conversion rather than against this
            # module's own helper. The chart is strongly curved (relative FD error 3e-2 at
            # h = 1e-4 on an ordinary Panda particle, 2e-10 at 1e-6), and the convergence
            # window shifts with the local gain, so the best of three step sizes is taken
            # per particle -- the usual remedy; the analytic-vs-analytic check above is the
            # primary assertion and this one is the independent sanity check.
            J_c6 = J6[:, :6].detach().cpu().numpy()
            fd_rel = np.inf
            for h in (1e-5, 1e-6, 1e-7):
                fd = np.zeros((ndof, 6))
                for k in range(6):
                    e = np.zeros_like(x_np)
                    e[k] = h
                    fd[:, k] = (p.VarsToQ(x_np + e)[:ndof]
                                - p.VarsToQ(x_np - e)[:ndof]) / (2 * h)
                fd_rel = min(fd_rel, np.abs(J_c6 - fd).max() / np.abs(fd).max())
            worst_fd = max(worst_fd, fd_rel)
        print(f"  [{robot}] autograd vs jacobian_gen (composed): worst relative "
              f"{worst_rel:.3e}; dq/dc6 vs central differences of VarsToQ: worst relative "
              f"{worst_fd:.3e} ({n_used} ordinary particles, {n_skipped} skipped)")
        assert n_used == 8, (robot, n_used)
        assert worst_rel <= 1e-7, (robot, worst_rel)
        assert worst_fd <= 1e-6, (robot, worst_fd)


def bf64_jac(bf, x):
    """|dq/dvars|_max at one lumped particle, the chart's local gain."""
    J = torch.func.jacrev(lambda v: bf.q_from_vars(v[None])[0])(x)
    return J.detach().abs().max()


def test_float32_uses_a_cached_private_copy():
    rng = np.random.default_rng(4)
    for robot in ROBOTS:
        p = program(robot)
        shared = p.ik_solver.nn_model
        before = next(shared.parameters()).dtype
        assert before == torch.float64
        n_entries = len(_PRIVATE_COPIES)

        bf32 = BatchedFlow.from_program(p, dtype=torch.float32)
        assert next(shared.parameters()).dtype == torch.float64, "the shared network moved"
        assert bf32.model is not shared and id(bf32.model) != id(shared)
        assert next(bf32.model.parameters()).dtype == torch.float32
        assert len(_PRIVATE_COPIES) == n_entries + 1
        again = BatchedFlow.from_program(p, dtype=torch.float32)
        assert again.model is bf32.model, "the copy must be cached, not rebuilt"
        assert len(_PRIVATE_COPIES) == n_entries + 1
        assert BatchedFlow.from_program(p).model is shared

        bf64 = BatchedFlow.from_program(p)
        X_np = lumped_batch(p, rng, 256)
        dev = device_of(p)
        q64 = bf64.q_from_vars(torch.tensor(X_np, dtype=torch.float64, device=dev))
        q32 = bf32.q_from_vars(torch.tensor(X_np, dtype=torch.float32, device=dev))
        assert q32.dtype == torch.float32
        diff = (q32.double() - q64).abs().max(dim=1).values
        # float32's error is the chart's local gain times float32 rounding, so it is
        # stated two ways. Raw, on particles inside or near the joint limits (|q|_inf < 10,
        # the ones a solver would keep): 7e-5 Panda, 1.2e-3 iiwa. Normalised by the
        # particle's own |dq/dvars|_max, on EVERY particle: 4e-7 Panda, 4e-6 iiwa -- a few
        # tens of float32 ulps, which is the invariant. On the iiwa's runaway
        # population (|q| ~ 1e5, |dq/dc| ~ 1e6; CLAUDE.md, the gain ceiling) the raw error
        # is RADIANS -- measured 25 rad on one of 256 particles -- which is a fact about
        # running a float32 swarm on that chart, recorded because the plan's default is a
        # float32 swarm with a float64 polish: those particles are the ones the swarm
        # re-draws (|q|_inf > 1000), and the polish runs in float64.
        X64 = torch.tensor(X_np, dtype=torch.float64, device=dev)
        gain = torch.stack([bf64_jac(bf64, X64[i]) for i in range(256)])
        normalised = diff / (1.0 + gain)
        near = q64.abs().max(dim=1).values < 10.0
        n_runaway = int((q64.abs().max(dim=1).values >= 100.0).sum().item())
        print(f"  [{robot}] float32 vs float64 q over 256 particles: raw max |diff| "
              f"{diff[near].max().item():.3e} on {int(near.sum().item())} with |q|_inf < 10 "
              f"(median {diff.median().item():.3e}); gain-normalised max "
              f"{normalised.max().item():.3e}; raw over all 256: {diff.max().item():.3e} "
              f"({n_runaway} particles at |q|_inf >= 100, max gain {gain.max().item():.3g})")
        # The raw bound is a gross-dtype-bug catch only (a particle with |q| < 10 can still
        # sit at gain 1e4, measured 1.2e-3 on the iiwa); the gain-normalised bound is the
        # invariant, at a few tens of float32 ulps.
        assert diff[near].max().item() <= 1e-2, (robot, diff[near].max().item())
        assert normalised.max().item() <= 1e-5, (robot, normalised.max().item())


def test_timing():
    """Printed only. Median of 20 after 3 warm-ups, `torch.cuda.synchronize()` around each."""
    if not torch.cuda.is_available():
        print("  (no CUDA; timing skipped)")
        return
    rng = np.random.default_rng(5)
    for robot in ROBOTS:
        p = program(robot)
        dev = device_of(p)
        print(f"  [{robot}] q_from_vars timing, ms (median of 20):")
        print(f"    {'B':>6} {'dtype':>8} {'forward':>10} {'fwd+bwd':>10}")
        for dtype in (torch.float64, torch.float32):
            bf = BatchedFlow.from_program(p, dtype=dtype)
            for B in (1, 64, 256, 1024):
                X0 = torch.tensor(lumped_batch(p, rng, B), dtype=dtype, device=dev)

                def forward():
                    with torch.no_grad():
                        bf.q_from_vars(X0)

                def forward_backward():
                    # `autograd.grad` rather than `.backward()`: the shared network's
                    # parameters require grad, and `.backward()` would accumulate into
                    # THEIR `.grad` buffers -- a side effect on the shared model and work
                    # no solver needs. Same graph, same reverse pass, gradient w.r.t. X only.
                    X = X0.clone().requires_grad_(True)
                    torch.autograd.grad(bf.q_from_vars(X).sum(), X)

                row = []
                for fn in (forward, forward_backward):
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(20):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        samples.append(1e3 * (time.perf_counter() - t0))
                    row.append(float(np.median(samples)))
                print(f"    {B:>6} {str(dtype).replace('torch.', ''):>8} "
                      f"{row[0]:>10.2f} {row[1]:>10.2f}")


if __name__ == "__main__":
    test_q_from_vars_matches_VarsToQ()
    test_conditioning_matches_CToPose7()
    test_invert_matches_InvertFlow_and_round_trips()
    test_gradients_match_jacobian_gen()
    test_float32_uses_a_cached_private_copy()
    test_timing()
    print("ALL PASS")
