"""The soft PCS arm's svgd fast path: configuration-space Jacobians by forward mode.

Since 2026-10-12 the soft PCS arm (`soft12`, chart `n6`) takes the split step
(`src/svgd/fused.py`) instead of the eager autograd step. Its plant vector (231 floating-body
positions) is not its configuration (12 normalised strains), so the row Jacobians are taken
w.r.t. the CONFIGURATION: `BatchedProgram._kinematics_cfg_jacobians` pushes the composite
`cfg -> TorchConfigToPlantQ -> FrameChain poses` forward along the 12 coordinate directions
(`torch.func.jvp`, vmapped), giving `J_plant = d q_plant / d cfg`, the frame origin's `Jp` and
the world angular velocity's `Jw` (from the quaternion's tangent); the collision row's plant
gradient is pulled back through `J_plant`. This file pins:

  1. the convention: on a RIGID arm (screw, Panda) the forward-mode `Jp` / `Jw` / `J_plant` of
     the padded map equal `FrameChain`'s closed form to 1e-12 (float64) -- the `Jw` from the
     quaternion's tangent is the same world angular-velocity Jacobian the rigid arms use;
  2. on the soft PCS pose and grasp programs (joint-space arm: `X` IS the configuration) at
     B = 1 and B = 64, float64: the rows from the new path equal the old path's, the rows'
     configuration Jacobian equals autograd of the same rows (the eager path, through the
     map and the collision row's autograd node) to 1e-10, the collision pullback separately to
     1e-10, the cost gradient to 1e-10; and on the learned arm the solver's `J_h`, `J_g` (the
     rows w.r.t. the decision variables, through the flow) against the autograd target. Also
     central differences on the frame Jacobians. float32 is REPORTED (not asserted tightly);
  3. a solve's step: from the same swarm and AL state, split steps in eager mode against the
     old eager autograd step (`_Target.analytic = False`, `_step_eager`), learned arm, both
     tasks, float64 and float32, at the fielded pin from a native start: the difference after
     one step is bounded, and after 30 steps it is bounded by the trajectory's own sensitivity
     (the old path against itself from a start one ulp away). (From a PAIRED start with the swarm jittered around
     the inverted latent, |z| ~ 5-6, both paths send particles into |q| ~ 1e5-1e8 within one
     step and, with no check in between to resample them, the whole swarm within two -- through
     the kernel average of their objective gradients; a step-only loop there compares two
     divergent trajectories, so this test does not use it.)
  4. the compiled and CUDA-graphed split step on the soft arm: captured, replayed bitwise
     against the compiled stages, and agreeing with eager before the metric solve;
  5. the GVS arm still has no torch map (refused by name), so it never reaches this path.

Plain `test_*` functions with a `__main__` driver, like the other svgd tests. Needs the
`soft12` / `screw7_p050` / `panda` `n6` charts under `models/`. Run alone, under the memory
scope: `.venv/bin/python tests/test_svgd_soft_fastpath.py` (`... quick` skips 4).
Tiers (tests/_tiers.py): the default `fast` runs 3 at 5 steps and skips 4, in ~22 s on one
CPU core; `slow` (~85 s, CPU ok; 4 needs CUDA) holds 3 at 30 steps and 4.
"""

import os
import sys
import time
from dataclasses import replace
from functools import partial

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)

from src.svgd import fused                                              # noqa: E402
from src.svgd.batched_program import BatchedProgram                     # noqa: E402
from src.svgd.solver import SvgdSolver, _Target                         # noqa: E402
from src.utils import HiddenPrints                                      # noqa: E402
import test_batched_program as T                                        # noqa: E402
import _tiers                                                           # noqa: E402
_tiers.jax_compilation_cache()                 # the GVS arm's model is XLA programs

DEVICE = T.DEVICE
FAILURES = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"\n          {detail}"))
    if not cond:
        FAILURES.append(name)


def _maxabs(a, b=None):
    d = a if b is None else a - b
    return float(d.detach().abs().max()) if d.numel() else 0.0


def _rel(a, b):
    return _maxabs(a, b) / max(1.0, _maxabs(b))


def _batched_jacobian(out, inp, N):
    """`d out / d inp` per particle, `[N, m, n]`, by `m` batched backwards."""
    m = out.shape[1]
    E = torch.eye(m, dtype=out.dtype, device=out.device).unsqueeze(1).expand(m, N, m)
    (J,) = torch.autograd.grad(out, inp, grad_outputs=E, is_grads_batched=True, retain_graph=True)
    return J.transpose(0, 1)


## ------------------------------------------------------------------------------------ ##
##              1. the forward-mode convention against FrameChain on rigid arms           ##
## ------------------------------------------------------------------------------------ ##

def test_forward_mode_matches_framechain_on_rigid_arms():
    """`_kinematics_cfg_jacobians` is defined for any plant replay; on a rigid arm (padded
    map, one-coordinate joints) its `Jp`, `Jw` must equal `FrameChain`'s closed form and its
    `J_plant` the padding's identity block. This pins the `Jw` convention (world angular
    velocity, `omega = 2 [-v, w I + [v]x] quat_dot`) against the one the rigid arms' step uses."""
    print("\n--- forward-mode Jp / Jw / J_plant vs FrameChain's closed form (rigid arms, float64) ---")
    rng = np.random.default_rng(2)
    for robot, task in (("screw", "pose"), ("screw", "mug"), ("panda", "mug")):
        bp = T.batched(robot, task, "numerical")
        assert not bp._cfg_jacobians and bp.plant_q_is_padded_cfg
        lo, hi = (b.detach().cpu().numpy() for b in bp.bounds)
        Q = torch.tensor(rng.uniform(lo, hi, size=(32, bp.ndof)), dtype=torch.float64, device=bp.device)
        qp_c, (qa_c, pa_c), (qb_c, pb_c), (Jp_c, Jw_c) = bp.kinematics(Q, row_jacobians=True)
        qp_f, (qa_f, pa_f), (qb_f, pb_f), (Jp_f, Jw_f, J_plant) = bp._kinematics_cfg_jacobians(Q)
        nd = bp.ndof
        e_val = max(_maxabs(qp_c, qp_f), _maxabs(qa_c, qa_f), _maxabs(pa_c, pa_f),
                    _maxabs(qb_c, qb_f), _maxabs(pb_c, pb_f))
        e_jp = _maxabs(Jp_c[:, :, :nd], Jp_f)
        e_jw = _maxabs(Jw_c[:, :, :nd], Jw_f)
        eye = torch.zeros(bp.num_pos, nd, dtype=Q.dtype, device=Q.device)
        eye[:nd] = torch.eye(nd, dtype=Q.dtype, device=Q.device)
        e_jpl = _maxabs(J_plant, eye.unsqueeze(0).expand_as(J_plant))
        print(f"    {robot}/{task}: values {e_val:.1e}  Jp {e_jp:.1e}  Jw {e_jw:.1e}  J_plant {e_jpl:.1e}")
        check(f"{robot}/{task}: forward-mode frame Jacobians equal FrameChain's to 1e-12",
              max(e_val, e_jp, e_jw, e_jpl) <= 1e-12, f"{e_val} {e_jp} {e_jw} {e_jpl}")
    T.close_all()


## ------------------------------------------------------------------------------------ ##
##                   2. the soft PCS arm's row Jacobians against autograd                 ##
## ------------------------------------------------------------------------------------ ##

def _soft_numerical_parity(task, B, dtype, rng):
    """Joint-space arm (`X` is the configuration): returns a dict of errors."""
    bp64 = T.batched("soft", task, "numerical")
    bp = bp64 if dtype == torch.float64 else BatchedProgram.from_program(
        bp64.program, dtype=dtype, device=bp64.device, collision=bp64.collision)
    assert bp.has_analytic_row_jacobians and bp._cfg_jacobians
    Q = torch.tensor(T.collision_free_q(bp.program, rng, B) if B == 1 else
                     rng.uniform(-1.0, 1.0, size=(B, bp.ndof)), dtype=torch.float64, device=bp.device)
    ## reference: autograd through the OLD eager path, in float64
    Qg = Q.clone().requires_grad_(True)
    ev_a = bp64.evaluate(Qg)
    D = ev_a.drake_rows
    Dq = _batched_jacobian(D, Qg, B)
    (gF,) = torch.autograd.grad(ev_a.F.sum(), Qg, retain_graph=True)
    coll = next(b.start for b in bp64._blocks if b.kind == "collision")
    ## new: the closed-form path (detached kinematics, configuration-space Jacobians)
    Qd = Q.to(dtype)
    ev_n = bp.evaluate(Qd, detach_kinematics=True, row_jacobians=True)
    Ja = bp.generic_rows_jacobian_cfg(ev_n).double()
    dFc, dFx = bp.cost_gradient_parts(Qd, ev_n.q)
    gF_n = (dFc + dFx).double()
    non = [r for r in range(D.shape[1]) if r != coll]
    scale_rows = max(1.0, _maxabs(Dq[:, non]))
    scale_coll = max(1.0, _maxabs(Dq[:, coll]))
    return dict(
        rows=_maxabs(ev_n.drake_rows.double(), D),
        jac=_maxabs(Ja[:, non], Dq[:, non]) / scale_rows,
        coll=_maxabs(Ja[:, coll], Dq[:, coll]) / scale_coll,
        coll_abs=_maxabs(Ja[:, coll], Dq[:, coll]),
        cost=_rel(gF_n, gF),
        has_cg=ev_n.extras.get("collision_grad") is not None)


def _soft_learned_parity(task, B, rng):
    """Learned arm: the solver's `J_h`, `J_g` w.r.t. the decision variables, analytic target
    against the autograd target (the old path), float64; and the cost gradient."""
    bp = T.batched("soft", task, "learned")
    X = torch.tensor(T.lumped_batch(bp, rng, B), dtype=torch.float64, device=bp.device)
    tg_n = _Target(bp, 1e-4)
    assert tg_n.analytic
    tg_a = _Target(bp, 1e-4)
    tg_a.analytic = False                       # the pre-2026-10-12 autograd path
    ev_n, ev_a = tg_n.evaluate(X), tg_a.evaluate(X)
    J_q = tg_a.jacobian_q(ev_a).detach()
    Jh_n, Jg_n = tg_n.row_jacobians_ev(ev_n, J_q)
    Jh_a, Jg_a = tg_a.row_jacobians_ev(ev_a, J_q)
    dFc_n, dFx_n = tg_n.cost_parts(ev_n)
    dFc_a, dFx_a = tg_a.cost_parts(ev_a)
    gF_n = (J_q.transpose(1, 2) @ dFc_n.unsqueeze(2)).squeeze(2) + dFx_n
    gF_a = (J_q.transpose(1, 2) @ dFc_a.unsqueeze(2)).squeeze(2) + dFx_a
    return dict(rows=max(_maxabs(ev_n.h, ev_a.h), _maxabs(ev_n.g, ev_a.g)),
                jac=max(_rel(Jh_n, Jh_a), _rel(Jg_n, Jg_a)),
                cost=_rel(gF_n, gF_a),
                q_inf=float(ev_a.cfg.detach().abs().max()))


def test_soft_row_jacobians_match_autograd():
    print("\n--- soft PCS: configuration-space row Jacobians vs autograd of the same rows ---")
    rng = np.random.default_rng(12)
    for task in ("pose", "mug"):
        for B in (1, 64):
            e = _soft_numerical_parity(task, B, torch.float64, rng)
            print(f"    soft/{task}/numerical B={B:2d} float64: rows {e['rows']:.1e}  row Jacobian "
                  f"{e['jac']:.1e}  collision pullback {e['coll']:.1e} (abs {e['coll_abs']:.1e})  "
                  f"cost gradient {e['cost']:.1e}")
            check(f"soft/{task}/numerical B={B}: rows unchanged and Jacobians (collision included) "
                  f"agree with autograd to 1e-10",
                  e["has_cg"] and e["rows"] <= 1e-12 and max(e["jac"], e["coll"], e["cost"]) <= 1e-10,
                  str(e))
            e32 = _soft_numerical_parity(task, B, torch.float32, rng)
            print(f"    soft/{task}/numerical B={B:2d} float32 (reported): rows {e32['rows']:.1e}  "
                  f"row Jacobian {e32['jac']:.1e}  collision pullback {e32['coll']:.1e}  cost "
                  f"gradient {e32['cost']:.1e}")
            ## The float32 collision pullback is REPORTED, not asserted: the float32 plant
            ## vector differs from the float64 one by rounding, and at a witness-feature switch
            ## Drake's own `d row / d q_plant` jumps across that gap (measured 2026-10-12: 4.4e-3
            ## relative on one grasp batch of 64, against 1e-9..2e-6 elsewhere) -- a difference of
            ## the two plant vectors, not of the pullback, which float64 pins to 1e-15 above.
            check(f"soft/{task}/numerical B={B} float32: task rows and cost within float32 rounding "
                  f"(1e-4 relative); collision pullback finite",
                  max(e32["jac"], e32["cost"]) <= 1e-4 and np.isfinite(e32["coll"]), str(e32))
            el = _soft_learned_parity(task, B, rng)
            print(f"    soft/{task}/learned   B={B:2d} float64: rows {el['rows']:.1e}  J_h/J_g "
                  f"{el['jac']:.1e}  cost gradient {el['cost']:.1e}  (|q|_inf max {el['q_inf']:.2f})")
            check(f"soft/{task}/learned B={B}: the solver's row Jacobians agree with the autograd "
                  f"target to 1e-10", el["rows"] <= 1e-12 and max(el["jac"], el["cost"]) <= 1e-10, str(el))
        T.close_all()


def test_soft_frame_jacobians_match_central_differences():
    """`Jp` and the quaternion's derivative through `Jw` (`fused.pose_and_jacobian`) against
    central differences of the batched kinematics on the soft grasp frame, float64."""
    print("\n--- soft PCS: frame Jacobians vs central differences ---")
    rng = np.random.default_rng(4)
    for task in ("pose", "mug"):
        bp = T.batched("soft", task, "numerical")
        Q = torch.tensor(rng.uniform(-0.9, 0.9, size=(8, bp.ndof)), dtype=torch.float64, device=bp.device)
        _, (quat, pos), _, (Jp, Jw, _) = bp.kinematics(Q, row_jacobians=True)
        pose, J = fused.pose_and_jacobian(quat, pos, Jp, Jw, bp.ndof)
        eps = 1e-6
        Jfd = torch.zeros_like(J)
        for k in range(bp.ndof):
            d = torch.zeros_like(Q)
            d[:, k] = eps
            _, (qa, pa), _, _ = bp.kinematics(Q + d)
            _, (qb, pb), _, _ = bp.kinematics(Q - d)
            Jfd[:, :, k] = (torch.cat([pa, fused.canonical_quat(qa)], 1)
                            - torch.cat([pb, fused.canonical_quat(qb)], 1)) / (2 * eps)
        err = _maxabs(J, Jfd)
        print(f"    soft/{task}: [position; quaternion] Jacobian vs central differences {err:.1e}")
        check(f"soft/{task}: frame Jacobians match central differences to 1e-7", err <= 1e-7, f"{err}")
    T.close_all()


## ------------------------------------------------------------------------------------ ##
##                  3. the split step (eager) against the old eager autograd step          ##
## ------------------------------------------------------------------------------------ ##

def _solver(task, dtype, N=16, kernel="cz", seed=9, mode="eager", start="native"):
    """A learned soft PCS solver at the fielded pin (c x z, rho 1e5, the native-draw swarm),
    its initial swarm and AL state."""
    p = T.program("soft", task, "learned")
    p.options = replace(p.options, which_solver="svgd", svgd_n=N, svgd_kernel=kernel,
                        svgd_dtype=dtype, svgd_rho=1e5,
                        svgd_paired_init="native",
                        svgd_compile=mode != "eager", svgd_cuda_graph=mode == "graphed",
                        acceptable_constr_viol_tol=1e-4)
    rng = np.random.default_rng(seed)
    q_init = T.collision_free_q(p, rng, 1)[0]
    with HiddenPrints():
        if start == "paired":
            p.SetStartFromQ(q_init)
        else:
            p.SetNativeStart(q_init, rng)
    s = SvgdSolver(p)
    s._build()
    x0 = np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float)
    X, _, _ = s._init_particles(x0)
    return s, X, s._init_state(X)


def _trajectory(s, X0, S, path, steps=30):
    """`[steps + 1]` swarms from `X0` under the split step (`path="split"`, eager) or the
    pre-2026-10-12 eager autograd step (`"autograd"`: `_Target.analytic = False`, no runner),
    with the AL state held fixed. Returns the list and the mean seconds per step."""
    runner = s._runner
    if path == "autograd":
        s._runner = None
        s._tg.analytic = False
    try:
        X, out = X0.clone(), [X0.clone()]
        t0 = time.perf_counter()
        for k in range(steps):
            fn = s._step_split if path == "split" else s._step_eager
            X = fn(X, S, want_lam=(k % 10 == 9))[0]
            out.append(X)
        if s.device.type == "cuda":
            torch.cuda.synchronize()
        return out, (time.perf_counter() - t0) / steps
    finally:
        s._runner = runner
        s._tg.analytic = True


## Each difference is bounded by FLOOR_FACTOR x the trajectory's own sensitivity at the same
## step (the old path against itself from a start perturbed by one ulp), or by ABS_TOL where
## that floor is smaller. Measured 2026-10-12 (RTX 3080 Ti, N = 16, c x z, rho 1e5, H / T,
## native start), max |dX| split vs autograd / 1-ulp floor:
##   float64  1 step 1.6e-11 / 1.2e-10, 9.2e-12 / 1.9e-10;  30 steps 2.3e-10 / 3.2e-10, 8.7e-07 / 5.1e-07
##   float32  1 step 5.1e-04 / 2.7e-02, 1.5e-02 / 1.1e-01;  30 steps 3.2e-01 / 1.0e-01, 1.1e-01 / 1.9e-01
## (pose, grasp). Every difference is at or below the floor within 3x. In float32 one GN solve
## at rho = 1e5 already turns rounding into 1e-4..1e-1: the old path does the same to a
## one-ulp perturbation of its own start.
ABS_TOL = {"float64": 1e-8, "float32": 1e-4}
FLOOR_FACTOR = 10.0


def test_split_step_matches_the_eager_autograd_step(steps=30):
    """The derivative arithmetic differs (forward mode through the map vs a backward), so the
    two steps agree to rounding, not bitwise, and the GN solve at rho = 1e5 amplifies rounding
    by the metric's conditioning; over 30 coupled steps the swarm amplifies it further. So the
    claim is two numbers: the difference after ONE step (rounding times conditioning), and the
    difference after 30 steps against the floor the trajectory itself sets -- the old path
    against itself from a start perturbed by one ulp. The fast tier runs `steps = 5` (the floor
    is taken at the same step, so the claim is the same one, over a shorter trajectory)."""
    print(f"\n--- soft PCS: split steps (eager) vs eager autograd steps, same start, {steps} steps ---")
    print(f"    {'row':>18}  {'1 step':>9}  {'floor (1)':>9}  {f'{steps} steps':>9}  {f'floor ({steps})':>10}  "
          f"{'moved':>6}  {'ms/step split / autograd':>24}")
    for dtype in ("float64", "float32"):
        for task in ("pose", "mug"):
            s, X0, S = _solver(task, dtype)
            r = s._make_runner()
            check(f"soft/{task}/{dtype}: the solver takes the split step", r is not None and r.mode == "eager")
            tn, t_new = _trajectory(s, X0, S, "split", steps)
            ta, t_old = _trajectory(s, X0, S, "autograd", steps)
            g = torch.Generator(device=X0.device).manual_seed(5)
            sign = torch.where(torch.rand(X0.shape, generator=g, device=X0.device, dtype=X0.dtype) < 0.5, -1.0, 1.0)
            Xp = X0 + sign * torch.finfo(X0.dtype).eps * X0.abs()
            Xp[0] = X0[0]
            tp, _ = _trajectory(s, Xp, S, "autograd", steps)
            d1, d30 = _maxabs(tn[1], ta[1]), _maxabs(tn[-1], ta[-1])
            f1, f30 = _maxabs(tp[1], ta[1]), _maxabs(tp[-1], ta[-1])
            moved = _maxabs(ta[-1], X0)
            print(f"    {'soft/' + task + '/' + dtype:>18}  {d1:9.1e}  {f1:9.1e}  {d30:9.1e}  {f30:10.1e}  "
                  f"{moved:6.2f}  {1e3 * t_new:11.0f} / {1e3 * t_old:.0f}")
            for k, d, f in (("one step", d1, f1), (f"{steps} steps", d30, f30)):
                bound = max(ABS_TOL[dtype], FLOOR_FACTOR * f)
                check(f"soft/{task}/{dtype}: {k}: split vs autograd within max({ABS_TOL[dtype]:g}, "
                      f"{FLOOR_FACTOR:g} x the 1-ulp floor)", np.isfinite(d) and d <= bound and moved > 1e-3,
                      f"{d} floor {f}")
    T.close_all()


## ------------------------------------------------------------------------------------ ##
##                        4. compiled and CUDA-graphed on the soft arm                    ##
## ------------------------------------------------------------------------------------ ##

def test_compiled_and_graphed_split_step():
    print("\n--- soft PCS: compiled / CUDA-graphed split step ---")
    if not torch.cuda.is_available():
        print("    (no CUDA: skipped)")
        return
    for task in ("pose", "mug"):
        s, X, S = _solver(task, "float32", mode="graphed")
        t0 = time.perf_counter()
        s.warm_up()
        r = s._make_runner()
        check(f"soft/{task}: three stages captured", r.mode == "graphed" and r.graphs_captured() >= 3,
              f"{r.mode} {r.graphs_captured()}")
        print(f"    soft/{task}: compile + capture {time.perf_counter() - t0:.1f} s")
        g1, g2, g3 = r.fns["s1"], r.fns["s2"], r.fns["s3"]
        cfg, qp = g1(X)
        J_q, kin = g2(X, cfg)
        v, gr = s._col.eval(qp.to("cpu", torch.float64).numpy())
        col = torch.as_tensor(v).to(X), torch.as_tensor(gr).to(X)
        args3 = (X, cfg, J_q, kin, col[0], col[1], fused.state_dict(S))
        out_g = g3(*args3)
        with torch.no_grad():
            out_c = g3.fn(*args3)
            cfg_c, _ = g1.fn(X)
        with torch.enable_grad():
            J_c, kin_c = g2.fn(X, cfg)
        flat_g, _ = torch.utils._pytree.tree_flatten(out_g)
        flat_c, _ = torch.utils._pytree.tree_flatten(out_c)
        bit3 = all(torch.equal(a, b) if a.dtype in (torch.bool, torch.long) else
                   bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
                   for a, b in zip(flat_g, flat_c) if isinstance(a, torch.Tensor))
        kin_flat, _ = torch.utils._pytree.tree_flatten(kin)
        kinc_flat, _ = torch.utils._pytree.tree_flatten(kin_c)
        bit12 = (torch.equal(cfg, cfg_c) and torch.equal(J_q, J_c)
                 and all(torch.equal(a, b) for a, b in zip(kin_flat, kinc_flat)))
        with torch.no_grad():
            out_e = fused.stage3(r.tg, s._sc, *args3)
            kin_e = r.tg.bp.kinematics(cfg, row_jacobians=True)
        dkin = max(_rel(a, b) for a, b in zip(kinc_flat, torch.utils._pytree.tree_flatten(kin_e)[0]))
        dpre = max(_rel(out_e[k], out_g[k]) for k in ("h", "g", "F", "H"))
        print(f"    soft/{task}: replay == compiled bitwise: stages 1-2 {bit12}, stage 3 {bit3}; "
              f"compiled vs eager kinematics + Jacobians (relative) {dkin:.1e}, rows / F / H {dpre:.1e}")
        check(f"soft/{task}: replay is bit-identical to the compiled stages", bit12 and bit3)
        check(f"soft/{task}: compiled agrees with eager to float32 rounding", max(dkin, dpre) <= 1e-3,
              f"{dkin} {dpre}")
        ## a few graphed steps run and stay finite
        Xs = X
        for _ in range(10):
            Xs = s._step_split(Xs, S)[0]
        check(f"soft/{task}: 10 graphed steps stay finite", bool(torch.isfinite(Xs).all()))
    T.close_all()


## ------------------------------------------------------------------------------------ ##
##                                        5. GVS                                          ##
## ------------------------------------------------------------------------------------ ##

def test_gvs_still_refused():
    from src.gvs_arm_program import GvsArmIKProgram
    try:
        GvsArmIKProgram.TorchConfigToPlantQ(None)
    except NotImplementedError as e:
        check("gvs: TorchConfigToPlantQ refuses by name (no torch forward map)", "JAX" in str(e), str(e))
    else:
        check("gvs: TorchConfigToPlantQ refuses by name (no torch forward map)", False, "no raise")


GROUPS = {
    "fast": [test_forward_mode_matches_framechain_on_rigid_arms, test_soft_row_jacobians_match_autograd,
             test_soft_frame_jacobians_match_central_differences,
             partial(test_split_step_matches_the_eager_autograd_step, steps=5), test_gvs_still_refused],
    ## The 30-step trajectories, and the compiled / CUDA-graphed step (it skips without CUDA).
    "slow": [test_split_step_matches_the_eager_autograd_step, test_compiled_and_graphed_split_step],
}
## The old `quick` run: everything but the compiled step, the trajectories at 30 steps.
GROUPS["quick"] = [*GROUPS["fast"][:3], test_split_step_matches_the_eager_autograd_step,
                   test_gvs_still_refused]


def main():
    try:
        _tiers.run(GROUPS)
    finally:
        T.close_all()
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
