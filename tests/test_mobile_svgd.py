"""AL-SVGD on a planar mobile base (`src/svgd/` on `src/mobile_base/` programs).

THE PROGRAM IS THE ORACLE, as in `tests/test_batched_program.py` (whose `_check_rows` this file
reuses). On a mobile program the batched CONFIGURATION is the extended `[q_arm | b]` (arm FIRST,
so every `cfg[:, :ndof]` read is the arm; `BatchedProgram.ncfg = ndof + 3`) and the decision
vector is `[b | c | z | q_c]` (learned) or `[b | q]` (joint space). Pinned here, on mobile Panda
pose and grasp, mobile soft PCS pose, and mobile iiwa and screw pose, learned and joint space:

  1. every row (`drake_rows`, the extra bindings), `F`, `F_reported`, `h`, `g` against the Drake
     program at random `X` (float64, ~1e-9; `_check_rows`), and `to_drake_x` / `from_drake_x`;
  2. the arm-frame conditioning pose of the learned arm (`BatchedProgram._local_pose7`: the
     torch composition `X_WA(b)^-1 X(c)`) against the program's own `TaskVarsToPose7`;
  3. the autograd Jacobian of the rows against the program's AutoDiffXd chain, `jacobian_q` /
     `vjp_q` against autograd of the configuration, and the closed-form path -- `evaluate(...,
     row_jacobians=True, detach_kinematics=True)`, `generic_rows_jacobian_cfg`, `J_q`, the cost
     gradient parts -- against autograd of the rows and of `F` w.r.t. `X`;
  4. the product kernels `bcz` (learned) and `bq` (joint space): the repulsion pulled back to
     the decision variables (`bq`'s arm factor padded in the base columns) against autograd of
     the frozen-bandwidth kernel sum; the kernel validation table (`bcz` / `bq` only on a base);
  5. the native draws (`SvgdSolver._draw_native`): finite, inside the bounds, reproducible, the
     base from the cell's base-start rule; `region_scale`; a few eager steps through all three
     stages; the compiled-structure signature telling a mobile target from a fixed-base one.

Needs the `n6` charts under `models/`. Run as a script, alone, under the memory scope:
`CUDA_VISIBLE_DEVICES= taskset -c 0 .venv/bin/python tests/test_mobile_svgd.py`.
Tiers (tests/_tiers.py): `fast` ~3-4 min on one CPU core; `slow` is the compiled step on CPU.
"""

import os
import sys
from dataclasses import replace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import _tiers                                                             # noqa: E402

from pydrake.all import AutoDiffXd, RigidTransform, RollPitchYaw, RigidTransform_  # noqa: E402,F401
from pydrake.autodiffutils import InitializeAutoDiff                       # noqa: E402

from src.generic_program import ProgramOptions                              # noqa: E402
from src.mobile_base import generate_scenes as G                            # noqa: E402
from src.mobile_base.robots import MOBILE_CLASSES                           # noqa: E402
from src.mobile_base.starts import MobileGrid                               # noqa: E402
from src.svgd import fused                                                  # noqa: E402
from src.svgd.batched_program import BatchedProgram, formulation_of         # noqa: E402
from src.svgd.solver import SvgdSolver, _Target, region_scale               # noqa: E402
from src.target_screening import SCENES                                     # noqa: E402
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints, RepoDir  # noqa: E402
from test_batched_program import _check_rows, drake_q                       # noqa: E402

CKPT = {"panda": "models/panda/panda__n6__step620000.pkl",
        "iiwa": "models/iiwa14/iiwa14__n6__step620000.pkl",
        "screw": "models/screw7_p050/screw7_p050__n6__step620000.pkl",
        "soft": "models/soft12/soft12__n6__step620000.pkl"}
SCENE_KEY = {"panda": "panda", "iiwa": "iiwa", "screw": "screw7_p050", "soft": "soft12"}
SEEDS = {"panda": 5, "iiwa": 6, "screw": 7, "soft": 8}

_PROGRAMS, _SCENES, _SOLVERS = {}, {}, {}


## ------------------------------------------------------------------------------------ ##
##                                 program construction                                  ##
## ------------------------------------------------------------------------------------ ##

def _options(arm, **kw):
    return ProgramOptions(visualize=False, use_float64=True,
                          joint_centering_cost=1e-4 if arm == "learned" else 1.0,
                          latent_trust_region=4.0, correction_cost_weight=10.0, mug_height=0.04,
                          ik_constraint_tol=(1e-4, 0.01), **kw)


def _ctor_kwargs(robot):
    return {"rung": "soft12"} if robot == "soft" else {}


def _yaml(robot):
    return os.path.join(RepoDir(), G.ROBOTS[SCENE_KEY[robot]][1])


def _scene(robot, task):
    """`(diagram, target, helper)`: the mobile scene and a pose target / a welded-mug target at
    the pose of a collision-free configuration drawn by the mobile grid."""
    key = (robot, task)
    if key in _SCENES:
        return _SCENES[key]
    cls = MOBILE_CLASSES[(robot, "pose", "learned")]
    with HiddenPrints():
        d = BuildEnv(None, _yaml(robot))
        kw = dict(_ctor_kwargs(robot))
        if robot in _SOLVERS:
            kw["model"] = _SOLVERS[robot]
        else:
            kw["checkpoint"] = CKPT[robot]
        helper = cls(d, options=_options("learned"), **kw)
        _SOLVERS[robot] = helper.ik_solver
        helper.create_prog()
    grid = MobileGrid(helper, lambda v: True, SCENES[(SCENE_KEY[robot], "pose")].wrist_frame)
    rng = np.random.default_rng(SEEDS[robot])
    if task == "pose":
        v = grid.Draw(rng)
        helper.plant.SetPositions(helper.plant_context, helper.GridToPlantQ(v))
        X = helper.frame.CalcPoseInWorld(helper.plant_context)
        _SCENES[key] = (d, np.array([*X.translation(), *X.rotation().ToQuaternion().wxyz()]),
                        helper)
    else:
        while True:
            v = grid.Draw(rng)
            q = helper.GridToPlantQ(v)
            if np.ravel(helper.collision_free_constraint_eval.Eval(q))[0] < 1.0:
                break
        with HiddenPrints():
            dm, mug = GenerateDiagramWithMug(q, helper, _yaml(robot), None)
        _SCENES[key] = (dm, mug, helper)
    return _SCENES[key]


def program(robot, task, arm):
    key = (robot, task, arm)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    diagram, target, helper = _scene(robot, task)
    cls = MOBILE_CLASSES[(robot, task, arm)]
    with HiddenPrints():
        p = cls(diagram, options=_options(arm), model=helper.ik_solver, **_ctor_kwargs(robot))
        if task == "pose":
            p.create_prog(target)
        else:
            p.create_prog(target_mug=target)
    attach(p, robot)
    _PROGRAMS[key] = p
    return p


def attach(p, robot, rule="uniform"):
    """What `MobileSetup.Attach` gives a per-cell program (the swarm's native base draw)."""
    p.base_start_rule = rule
    p.base_start_grid = MobileGrid(p, lambda v: True, SCENES[(SCENE_KEY[robot], "pose")].wrist_frame)
    return p


def batched(robot, task, arm, dtype=torch.float64):
    return BatchedProgram.from_program(program(robot, task, arm), dtype=dtype, device="cpu")


def _draw_batch(bp, rng, B):
    p = bp.program
    blo, bhi = p.BaseLimits()
    b = rng.uniform(blo, bhi, size=(B, 3))
    b[:, 2] = rng.uniform(-np.pi, np.pi, size=B)
    if bp.is_learned:
        c = bp.native_c().cpu().numpy()[None, :] + rng.uniform(-0.3, 0.3, size=(B, 6))
        z = rng.standard_normal((B, bp.width))
        qc = rng.uniform(-0.1, 0.1, size=(B, bp.ndof))
        return np.concatenate([b, c, z, qc], axis=1)
    lower, upper = p.ConfigLimits()
    q = rng.uniform(np.asarray(lower)[:bp.ndof], np.asarray(upper)[:bp.ndof], size=(B, bp.ndof))
    return np.concatenate([b, q], axis=1)


def lumped_batch(bp, rng, B, tame=5.0):
    """Random decision vectors: the base uniform in its box (yaw in [-pi, pi]); learned `c =
    native_c +- U(0.3)`, `z ~ N(0, I)`, `q_c ~ U(+-0.1)`; joint space `q ~ U(limits)`.

    Learned particles are kept only where the arm's `|q|_inf < tame`: the flow's runaway
    population (CLAUDE.md, the gain ceiling; 85% of the soft PCS chart's draws here) amplifies
    a last-ulp difference in the pose conversion by 1e8 (measured 2026-11-25: ordinary
    particles agree to 1e-15, |q| ~ 1e7 particles to 1e-6), which is chaos, not a parity
    failure. The runaway population is exercised by the finite-step checks instead."""
    X = _draw_batch(bp, rng, B)
    if not bp.is_learned or tame is None:
        return X
    out = []
    for _ in range(200):
        X = _draw_batch(bp, rng, 64)
        with torch.no_grad():
            arm = bp._config(torch.tensor(X, dtype=torch.float64))[:, :bp.ndof].abs().amax(dim=1)
        out.extend(X[(arm < tame).numpy()])
        if len(out) >= B:
            return np.stack(out[:B])
    raise RuntimeError("no tame learned particles drawn")


def _tensor(X_np):
    return torch.tensor(X_np, dtype=torch.float64)


## (robot, task) pairs the fast tier covers. The iiwa and screw poses are the cheap extras.
CASES = [("panda", "pose"), ("panda", "mug"), ("soft", "pose"), ("iiwa", "pose"),
         ("screw", "pose")]
KEYS = [(r, t, a) for r, t in CASES for a in ("learned", "numerical")]


## ------------------------------------------------------------------------------------ ##
##                                       the tests                                       ##
## ------------------------------------------------------------------------------------ ##

def test_layout_and_formulation():
    """Layout, formulation detection (a mobile joint program's lumped vars are `[b | q]`),
    the arm-first configuration and the closed-form (cfg-Jacobian) derivative path."""
    for robot, task, arm in KEYS:
        p = program(robot, task, arm)
        bp = BatchedProgram.from_program(p, dtype=torch.float64, device="cpu")
        assert p.MOBILE and bp.is_mobile and bp.ncfg == bp.ndof + 3
        assert formulation_of(p) == ("learned" if arm == "learned" else "joint")
        expect = 3 + 6 + bp.width + bp.ndof if arm == "learned" else 3 + bp.ndof
        assert bp.nvars == expect, (robot, task, arm, bp.nvars, expect)
        assert not bp.plant_q_is_padded_cfg and bp._cfg_jacobians
        assert bp.has_analytic_row_jacobians
        X = _tensor(lumped_batch(bp, np.random.default_rng(0), 3))
        cfg = bp._config(X)
        assert cfg.shape == (3, bp.ncfg)
        assert torch.equal(cfg[:, bp.ndof:], X[:, :3]), "the configuration's tail is the base"
        q_drake = [np.asarray(p.Config(x), dtype=float) for x in X.numpy()]
        q_batched = cfg[:, :bp.ndof].detach().numpy()
        if arm == "numerical":
            assert np.array_equal(q_batched, np.stack(q_drake))
        ## The decision vectors round-trip through the Drake layout.
        for x in X.numpy():
            assert np.array_equal(bp.from_drake_x(bp.to_drake_x(x)), x)
        try:
            bp.init_from_q(np.zeros((1, bp.ndof)))
        except NotImplementedError:
            pass
        else:
            raise AssertionError("init_from_q must refuse on a mobile base")
        ## The soft arm's configuration is not the plant's; the batched map is the program's.
        qp = bp.config_to_plant_q(cfg.detach()).numpy()
        ref = np.stack([p.SvgdConfigToPlantQ(c) for c in cfg.detach().numpy()])
        assert np.abs(qp - ref).max() <= 1e-12, (robot, task, arm)
    print("PASS mobile layout: nvars, arm-first [q | b] configuration, map == SvgdConfigToPlantQ, "
          "[b | q] detected as joint space, init_from_q refused, drake x round trip "
          f"({len(KEYS)} programs)")


def test_row_parity():
    rng = np.random.default_rng(0)
    print("  row parity (max |batched - Drake|, ordinary particles):")
    for robot, task, arm in KEYS:
        bp = batched(robot, task, arm)
        label = f"mobile {robot}/{task}/{arm}"
        fig = _check_rows(bp, lumped_batch(bp, rng, 8), label)
        print(f"    {label:30s} non-collision rows B=1 {fig['noncoll_B1']:.2e}, B=8 "
              f"{fig['noncoll_BN']:.2e} ({fig['n_ordinary']}/8 ordinary); extra rows "
              f"{fig['extra_rows']:.1e}; F rel {fig['F_rel']:.1e}, F_reported rel "
              f"{fig['F_reported_rel']:.1e}")
    print(f"PASS mobile row parity on {len(KEYS)} programs")


def test_local_pose_matches_the_program():
    """`_local_pose7` (torch) against `TaskVarsToPose7` (Drake) for bases anywhere, yaw beyond
    +-pi, and rotations of every sign of w."""
    p = program("panda", "pose", "learned")
    bp = batched("panda", "pose", "learned")
    rng = np.random.default_rng(1)
    B = 200
    b = np.column_stack([rng.uniform(-2, 2, B), rng.uniform(-2, 2, B),
                         rng.uniform(-2 * np.pi, 2 * np.pi, B)])
    c = np.column_stack([rng.uniform(-1, 1, (B, 3)), rng.uniform(-np.pi, np.pi, (B, 3))])
    mine = bp._local_pose7(_tensor(b), _tensor(c)).numpy()
    err_p = err_q = 0.0
    n_neg = 0
    for i in range(B):
        pos, quat = p.TaskVarsToPose7(np.concatenate([b[i], c[i]]), float)
        err_p = max(err_p, float(np.abs(np.asarray(pos) - mine[i, :3]).max()))
        err_q = max(err_q, float(np.abs(np.asarray(quat) - mine[i, 3:]).max()))
        n_neg += int(mine[i, 3] < 0)
    assert n_neg == 0 and (mine[:, 3] >= 0).all()
    assert err_p <= 1e-12 and err_q <= 1e-12, (err_p, err_q)
    print(f"PASS arm-frame conditioning pose == TaskVarsToPose7 ({B} draws): position "
          f"{err_p:.1e}, quaternion {err_q:.1e}, w >= 0")


def test_gradients_match_autodiff():
    """(a) autograd Jacobian of `drake_rows` against the program's AutoDiffXd chain;
    (b) `jacobian_q` / `vjp_q` against autograd of the configuration; (c) the closed-form path
    against autograd of the rows and of `F`."""
    rng = np.random.default_rng(2)
    print("  gradients (relative to max(1, |J_Drake|_max)):")
    for robot, task, arm in KEYS:
        bp = batched(robot, task, arm)
        p = bp.program
        coll_row = next(b.start for b in bp._blocks if b.kind == "collision")
        noncoll = [r for r in range(bp.n_rows) if r != coll_row]
        worst_ad = worst_jq = worst_cf = worst_F = worst_coll = 0.0
        used = 0
        for x_np in lumped_batch(bp, rng, 12):
            if used >= 3 or np.abs(drake_q(p, x_np)).max() >= 100.0:
                continue
            used += 1
            x = torch.tensor(x_np, dtype=torch.float64)
            J = torch.autograd.functional.jacobian(
                lambda v: bp.evaluate(v[None]).drake_rows[0], x).numpy()
            rows_ad = p.EvalAllConstraints(InitializeAutoDiff(x_np).reshape(-1))
            J_ref = np.stack([
                r.derivatives() if isinstance(r, AutoDiffXd) and r.derivatives().size == bp.nvars
                else np.zeros(bp.nvars) for r in rows_ad])
            denom = max(1.0, np.abs(J_ref[noncoll]).max())
            worst_ad = max(worst_ad, float(np.abs(J[noncoll] - J_ref[noncoll]).max() / denom))
            ## jacobian_q / vjp_q
            Jq = bp.jacobian_q(x[None])[0]
            assert Jq.shape == (bp.ncfg, bp.nvars)
            Jq_ref = torch.autograd.functional.jacobian(
                lambda v: bp.evaluate(v[None], need_collision=False).q[0], x)
            worst_jq = max(worst_jq, float((Jq - Jq_ref).abs().max()))
            R = torch.tensor(rng.standard_normal((1, bp.ncfg)), dtype=torch.float64)
            vjp = bp.vjp_q(x[None], R)
            scale = max(1.0, float((R[0].abs() @ Jq.abs()).max()))
            assert torch.allclose(vjp, (R[0] @ Jq)[None], rtol=0, atol=1e-13 * scale)
            ## closed form: rows through (generic_rows_jacobian_cfg, J_q) vs autograd, same X
            ev = bp.evaluate(x[None], detach_kinematics=True, row_jacobians=True)
            Dq = bp.generic_rows_jacobian_cfg(ev)
            assert Dq.shape == (1, bp.n_rows, bp.ncfg)
            Jcf = (Dq[0] @ Jq).numpy()
            dn = max(1.0, np.abs(J).max())
            worst_cf = max(worst_cf, float(np.abs(Jcf[noncoll] - J[noncoll]).max() / dn))
            worst_coll = max(worst_coll, float(np.abs(Jcf[coll_row] - J[coll_row]).max()
                                               / max(1.0, np.abs(J[coll_row]).max())))
            ## F: closed-form cost gradient parts vs autograd of F
            gF_ref = torch.autograd.functional.jacobian(
                lambda v: bp.evaluate(v[None], need_collision=False).F[0], x)
            dFc, dFx = bp.cost_gradient_parts(x[None], ev.q)
            gF = (Jq.T @ dFc[0]) + dFx[0]
            worst_F = max(worst_F, float((gF - gF_ref).abs().max() / max(1.0, float(gF_ref.abs().max()))))
        assert used == 3, (robot, task, arm, used)
        tol = 1e-8 if bp.is_learned else 1e-10
        print(f"    mobile {robot}/{task}/{arm:9s}: rows vs Drake AD {worst_ad:.1e}, jacobian_q "
              f"{worst_jq:.1e}, closed form vs autograd {worst_cf:.1e} (collision {worst_coll:.1e}), "
              f"dF/dX {worst_F:.1e}")
        assert worst_ad <= tol and worst_jq <= 1e-12, (robot, task, arm)
        assert worst_cf <= 1e-9 and worst_coll <= 1e-9 and worst_F <= 1e-10, (robot, task, arm)
    print("PASS mobile gradients: autograd vs Drake AD, jacobian_q / vjp_q, closed-form row and "
          "cost Jacobians")


def _target(bp):
    return _Target(bp, 1e-4)


def test_region_scale_and_validation():
    for arm in ("learned", "numerical"):
        bp = batched("panda", "pose", arm)
        s = region_scale(bp)
        assert s.shape == (bp.nvars,) and np.all(s > 0)
        lo, hi = (b.numpy() for b in bp.bounds)
        assert np.allclose(s[:3], 0.5 * (hi[:3] - lo[:3])), "base half-widths lead the scales"
        if arm == "numerical":
            assert np.allclose(s, 0.5 * (hi - lo))
        else:
            c, w = bp.program.options.c_position_slack, bp.width
            assert np.allclose(s[3:6], c) and np.allclose(s[6:9], np.pi) and s.size == 9 + w + 7
    ## The kernel table.
    for space in fused.ALL_KERNEL_SPACES:
        ProgramOptions(which_solver="svgd", svgd_kernel=space)
    ok = {"learned": ("bcz", "x", "none"), "joint": ("bq", "x", "none")}
    for kind, learned in (("learned", True), ("joint", False)):
        for space in fused.ALL_KERNEL_SPACES:
            call = lambda: fused.check_kernel_for_arm(space, learned, arm="X", mobile=True)
            if space in ok[kind]:
                call()
            else:
                try:
                    call()
                except ValueError as exc:
                    assert space in str(exc) and "mobile" in str(exc), str(exc)
                else:
                    raise AssertionError(f"{space} accepted on the mobile {kind} arm")
    for space in fused.MOBILE_KERNEL_SPACES:
        for learned in (True, False):
            try:
                fused.check_kernel_for_arm(space, learned, arm="X")
            except ValueError as exc:
                assert space in str(exc) and "fixed base" in str(exc)
            else:
                raise AssertionError(f"{space} accepted on a fixed base")
    for arm, bad in (("learned", ("cz", "q", "qz", "z", "c", "pose")),
                     ("numerical", ("cz", "q", "pose", "bcz"))):
        p = program("panda", "pose", arm)
        saved = p.options
        try:
            for space in bad:
                p.options = replace(saved, which_solver="svgd", svgd_kernel=space)
                try:
                    SvgdSolver(p)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"SvgdSolver accepted {space} on mobile {arm}")
            good = "bcz" if arm == "learned" else "bq"
            p.options = replace(saved, which_solver="svgd", svgd_kernel=good)
            SvgdSolver(p)
        finally:
            p.options = saved
    print("PASS region_scale (base half-widths first); kernel table: bcz / bq only on a base, "
          "refused on a fixed base, fixed-base spaces refused on a base")


def _autograd_repulsion(tg, X, kernel, cfg_fn):
    """`-1/2 d (sum_ab K_ab) / d y` at frozen bandwidths, in the normalised coordinates y."""
    sc = fused.StepConfig.from_options(replace(tg.bp.options, svgd_kernel=kernel))
    N = X.shape[0]
    finite = torch.ones(N, dtype=torch.bool)
    cfg0 = cfg_fn(X)
    feats = fused.kernel_features(kernel, tg, X, cfg0)
    _, _, _, bw = fused.kernel_terms(sc, cfg0, finite, X, tg=tg)
    assert bw.shape == (len(feats),)
    y = (X / tg.s.unsqueeze(0)).detach().requires_grad_(True)
    Xg = y * tg.s.unsqueeze(0)
    cfg = cfg_fn(Xg)
    blk = fused.kernel_blocks(tg)
    if kernel == "bcz":
        us = [y[:, blk["b"]], y[:, blk["c"]], y[:, blk["z"]]]
    else:
        us = [y[:, blk["b"]], cfg[:, :tg.bp.ndof]]          # the arm columns, with a graph
    expo = 0.0
    for u0, h in zip(us, bw):
        expo = expo + torch.cdist(u0, u0) ** 2 / h
    (g,) = torch.autograd.grad(torch.exp(-expo).sum(), y)
    return -0.5 * g, sc, bw


def test_kernels_bcz_and_bq():
    for arm, kernel in (("learned", "bcz"), ("numerical", "bq")):
        bp = batched("panda", "pose", arm)
        p = bp.program
        p.options = replace(p.options, which_solver="svgd", svgd_kernel=kernel)
        try:
            tg = _target(bp)
            blk = fused.kernel_blocks(tg)
            assert blk["b"] == slice(0, 3)
            if arm == "learned":
                assert blk["c"] == slice(3, 9) and blk["z"] == slice(9, 9 + bp.width)
            rng = np.random.default_rng(3)
            X = _tensor(lumped_batch(bp, rng, 12))
            R_ref, sc, bw = _autograd_repulsion(tg, X, kernel, bp._config)
            ## the production path: kernel_terms -> repulsion_y through the Jacobian of cfg
            cfg = bp._config(X).detach()
            finite = torch.ones(X.shape[0], dtype=torch.bool)
            K, R, on, bw2 = fused.kernel_terms(sc, cfg, finite, X, tg=tg)
            assert on and torch.equal(bw2, bw)
            assert torch.allclose(K, K.T) and torch.allclose(K.diag(), torch.ones(12, dtype=K.dtype))
            if arm == "numerical":
                R_cfg, R_y = R
                assert R_cfg.shape == (12, bp.ncfg) and torch.all(R_cfg[:, bp.ndof:] == 0)
            Jq = bp.jacobian_q(X)
            R_y = fused.repulsion_y(tg, X, Jq, R, on, finite)
            err = float((R_y - R_ref).abs().max() / max(1e-12, float(R_ref.abs().max())))
            assert err <= 1e-10, (kernel, err)
            ## kernel_features' spread reads the right columns
            feats = fused.kernel_features(kernel, tg, X, cfg)
            assert [f[0].shape[1] for f in feats] == (
                [3, 6, bp.width] if arm == "learned" else [3, bp.ndof])
            print(f"PASS kernel {kernel}: repulsion pulled back to y == -1/2 grad of the kernel sum "
                  f"(rel {err:.1e}); bandwidths {[round(float(h), 3) for h in bw]}")
        finally:
            p.options = replace(p.options, svgd_kernel="cz")


def _solver_for(robot, task, arm, kernel, **kw):
    p = program(robot, task, arm)
    p.options = replace(p.options, which_solver="svgd", svgd_kernel=kernel, svgd_n=16,
                        svgd_dtype="float64", svgd_paired_init="native", max_wall_time=60.0,
                        svgd_compile=False, **kw)
    return p, SvgdSolver(p)


def test_native_draws():
    for robot, task, arm, kernel in (("panda", "pose", "learned", "bcz"),
                                     ("panda", "pose", "numerical", "bq"),
                                     ("panda", "mug", "numerical", "bq"),
                                     ("soft", "pose", "learned", "bcz")):
        p, solver = _solver_for(robot, task, arm, kernel)
        solver._build()
        bp = solver._tg.bp
        gen = solver._generator(123)
        X, stats = solver._draw_native(16, gen)
        assert X.shape == (16, bp.nvars) and torch.isfinite(X).all()
        lo, hi = bp.bounds
        ## drawn inside the bounds (the learned c / z start from the program's own native)
        sl = slice(0, 3) if arm == "learned" else slice(0, bp.nvars)
        assert (X[:, sl] >= lo[sl]).all() and (X[:, sl] <= hi[sl]).all(), (robot, task, arm)
        assert (X[:, 2].abs() <= np.pi).all(), "native yaw is drawn in [-pi, pi]"
        X2, _ = solver._draw_native(16, solver._generator(123))
        assert torch.equal(X, X2), "the native swarm is reproducible from the generator"
        X3, _ = solver._draw_native(16, solver._generator(124))
        assert not torch.equal(X[:, :3], X3[:, :3]), "a different generator gives different bases"
        if arm == "numerical":
            ## joint space: every drawn particle is collision free on the full plant vector
            cfg = bp._config(X)
            qp = bp.config_to_plant_q(cfg).numpy()
            free = np.array([np.ravel(p.collision_free_constraint_eval.Eval(q))[0] < 1.0 for q in qp])
            assert free.all() or stats.get("native_fallback", 0) > 0, (robot, task)
        x0 = np.nan_to_num(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        Xi, clip, _ = solver._init_particles(x0)
        assert torch.isfinite(Xi).all() and torch.equal(Xi[0], torch.tensor(x0)) and clip[0] == 0
        p.options = replace(p.options, svgd_kernel="cz")
    ## the draw refuses, by name, a program with no base-start rule attached
    p = program("panda", "pose", "learned")
    saved = p.base_start_rule
    p.base_start_rule = None
    try:
        p.SvgdNativeBase(2, np.random.default_rng(0))
    except RuntimeError as exc:
        assert "base-start rule" in str(exc)
    else:
        raise AssertionError("SvgdNativeBase without a rule")
    finally:
        p.base_start_rule = saved
    print("PASS native draws: finite, inside the bounds, reproducible, collision-filtered "
          "[b | q] on joint space, base from the cell's rule, particle 0 at the guess")


def test_eager_steps_are_finite_and_descend():
    """A few real steps through stage 1/2/3 (`warm_up`, eager) on the learned and joint-space
    arms; every quantity finite and the swarm's median constraint violation lower than at the
    start. The learned arm starts from the `facing` base rule: from `uniform` / `home` the
    target is far outside the chart's training distribution in the arm's frame, the flow's
    gain there is the runaway population's, and 20 steps WITHOUT the outer loop's resampling
    send |q| to 1e30 (measured 2026-11-25; a fixed-base program does not do that)."""
    for robot, task, arm, kernel, rule in (("panda", "pose", "learned", "bcz", "facing"),
                                           ("panda", "pose", "numerical", "bq", "uniform"),
                                           ("soft", "pose", "numerical", "bq", "uniform")):
        p, solver = _solver_for(robot, task, arm, kernel)
        attach(p, robot, rule)
        solver.warm_up()
        assert solver._runner is not None and solver._runner.mode == "eager"
        x0 = np.nan_to_num(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        X, _, _ = solver._init_particles(x0)
        S = solver._init_state(X)
        tg = solver._tg
        v0 = float(torch.nan_to_num(tg.evaluate(X, need_grad=False).infeas, posinf=1e9).median())
        for k in range(20):
            X, clip, n_clip, lam, fro, ratio, bw = solver._step_split(X, S, want_lam=(k == 19))
        ev = tg.evaluate(X, need_grad=False)
        assert torch.isfinite(X).all() and torch.isfinite(bw).all(), (robot, arm)
        v1 = float(torch.nan_to_num(ev.infeas, posinf=1e9).median())
        assert v1 < v0, (robot, arm, v0, v1)
        lo, hi = tg.bp.bounds
        assert (X >= lo).all() and (X <= hi).all()
        print(f"PASS 20 eager svgd steps, mobile {robot}/{task}/{arm}/{kernel} ({rule}): median "
              f"violation {v0:.3g} -> {v1:.3g}, bandwidths {[round(float(h), 3) for h in bw]}")
        p.options = replace(p.options, svgd_kernel="cz")
        attach(p, robot, "uniform")


def test_structure_signature_separates_mobile_from_fixed():
    from src.panda_program import PandaIKProgram
    L = program("panda", "pose", "learned")
    bp_m = BatchedProgram.from_program(L, dtype=torch.float64, device="cpu")
    with HiddenPrints():
        d = BuildEnv(None, os.path.join(RepoDir(), "models/panda/panda_finray_collision_hardened.yaml"))
        F = PandaIKProgram(d, options=_options("learned"), model=L.ik_solver)
        F.create_prog(program("panda", "pose", "learned").target_pose)
    bp_f = BatchedProgram.from_program(F, dtype=torch.float64, device="cpu")
    sig_m, _ = fused.structure_signature(_target(bp_m))
    sig_f, _ = fused.structure_signature(_target(bp_f))
    assert sig_m != sig_f
    flat = repr(sig_m)
    assert "is_mobile" in flat and "ncfg" in flat and "_config_map_key" in flat
    assert bp_m._config_map_key[0] == "mobile" and bp_f._config_map_key is None
    ## two mobile programs of different robots never share a map key
    k_soft = batched("soft", "pose", "learned")._config_map_key
    assert k_soft != bp_m._config_map_key
    print("PASS the structure signature records is_mobile, ncfg and a map key that differs by "
          "robot; a mobile target never shares a template with a fixed-base one")


def test_compiled_step_matches_eager():
    """slow tier: `svgd_compile` (CPU inductor) reproduces the eager step on a mobile program."""
    for arm, kernel in (("learned", "bcz"), ("numerical", "bq")):
        p, eager = _solver_for("panda", "pose", arm, kernel)
        attach(p, "panda", "facing")
        eager.warm_up()
        x0 = np.nan_to_num(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        X, _, _ = eager._init_particles(x0)
        S = eager._init_state(X)
        Xe = eager._step_split(X, S)[0]
        p2, comp = _solver_for("panda", "pose", arm, kernel)
        p2.options = replace(p2.options, svgd_compile=True)
        comp = SvgdSolver(p2)
        comp.warm_up()
        Xc = comp._step_split(X, S)[0]
        ## Tame particles only (|q|_inf < 5 before AND after the step): the flow's runaway
        ## population amplifies the compiled kernels' reassociation by its gain.
        with torch.no_grad():
            tame = ((eager._tg.bp._config(X)[:, :7].abs().amax(dim=1) < 5)
                    & (eager._tg.bp._config(Xe)[:, :7].abs().amax(dim=1) < 5))
        assert int(tame.sum()) >= 4, "too few tame particles to compare"
        err = float((Xe - Xc)[tame].abs().max())
        assert err <= 1e-8, (arm, err)
        print(f"PASS compiled step == eager step, mobile panda {arm}: {err:.1e} "
              f"({int(tame.sum())} tame particles)")
        p.options = replace(p.options, svgd_kernel="cz")
        attach(p, "panda", "uniform")


def test_clearances_backend_covers_the_platform():
    """The record's AL-SVGD collision backend (`svgd_collision_backend=clearances`, one
    `CalcRobotClearances` call) equals the program's own `MinimumDistanceLowerBoundConstraint`
    on the mobile scenes -- the platform's pairs included (its construction refuses a pair-set
    mismatch; here the values and gradients are compared, and the base gradient must be live)."""
    from pydrake.common import Parallelism
    from src.svgd.collision_backend import (ClearanceEvaluator, CollisionEvaluator,
                                            clearances_unavailable_reason)
    reason = clearances_unavailable_reason()
    if reason is not None:
        print("SKIP clearances backend:", reason)
        return
    for robot in ("panda", "soft"):
        for task in ("pose", "mug"):
            p = program(robot, task, "numerical")
            rng = np.random.default_rng(7)
            Q = np.stack([p.GridToPlantQ(p.base_start_grid.Draw(rng)) for _ in range(16)])
            va, ga = CollisionEvaluator.from_program(p).eval(Q)
            vc, gc = ClearanceEvaluator.from_program(p, parallelism=Parallelism(1)).eval(Q)
            assert np.all(np.isfinite(va)) and np.all(np.isfinite(vc)), (robot, task)
            assert np.max(np.abs(va - vc)) < 1e-12, (robot, task, np.max(np.abs(va - vc)))
            assert np.max(np.abs(ga - gc)) < 1e-10, (robot, task, np.max(np.abs(ga - gc)))
            assert np.max(np.abs(gc[:, p.BaseSlots()])) > 0, (robot, task)
    print("PASS clearances backend == autodiff constraint on the mobile scenes (platform in)")


GROUPS = {"fast": [test_layout_and_formulation, test_row_parity, test_local_pose_matches_the_program,
                   test_gradients_match_autodiff, test_region_scale_and_validation,
                   test_kernels_bcz_and_bq, test_native_draws,
                   test_eager_steps_are_finite_and_descend,
                   test_structure_signature_separates_mobile_from_fixed,
                   test_clearances_backend_covers_the_platform],
          "slow": [test_compiled_step_matches_eager]}


def main():
    _tiers.run(GROUPS)
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
