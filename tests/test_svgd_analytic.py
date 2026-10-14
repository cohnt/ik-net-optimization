"""End-to-end checks of the svgd solver on the ANALYTIC formulation (Panda and iiwa).

The analytic arm under svgd (2026-10-12): the swarm moves `[xyz_rpy | psi]`, the configuration
is the robot's closed form on the cell's branch (`src/svgd/batched_analytic.py`), and the
`ReachabilityConstraint` rows are the map's aux output. Thomas ruled both kernel spaces are
run -- `x` (the arm's own normalised decision vector) and `q` (the configuration, pulled back
through the map's VJP) -- and the learned-only kernels refused. This file pins:

  1. the SPLIT STEP's closed-form derivatives (stages 1-2 of `fused`, then `assemble` and
     `_Target.row_jacobians_out` -- the path every svgd step takes) against full autograd of
     the very same rows in the AL's units, float64, on all four programs, the reachability
     rows' `J_u` included, and the stage path's rows equal to `evaluate`'s;
  2. one FEASIBLE-BY-CONSTRUCTION cell per robot x task (the pose target at the frame of a
     collision-free configuration of the hardened scene; the grasp mug welded at the grasp
     frame of one, `test_batched_program._scene`), paired start at another collision-free
     configuration, solved under svgd with kernels `x` and `q`, eager: no exception, the
     solver's Drake verdict equals `benchmark.verify`'s, the trace and counts are sane;
  3. the same under `svgd_compile` (and `svgd_cuda_graph` on CUDA) on two structures, warmed
     by `WarmUpSvgdStep` first exactly as the benchmark does;
  4. the native start protocol on one cell per robot;
  5. `svgd_kernel=cz` (the fielded default, learned-only) refused on the analytic arm.

Successes are PRINTED (the cap is 20 s; a single cell's verdict is not a measurement);
what is asserted is that the verdicts agree and nothing breaks. Models: the `n6` charts
(loaded only because the analytic programs inherit the learned base's construction; no
network is evaluated). `.venv/bin/python tests/test_svgd_analytic.py`, alone, under the
memory scope.
"""

import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from src import benchmark as bm                                             # noqa: E402
from src.generic_program import ProgramOptions                              # noqa: E402
from src.svgd import fused                                                  # noqa: E402
from src.svgd.batched_program import BatchedProgram                         # noqa: E402
from src.svgd.result import SvgdResult                                      # noqa: E402
from src.svgd.solver import SvgdSolver, _Target                             # noqa: E402
from src.target_screening import SceneFile                                  # noqa: E402
from src.utils import BuildEnv, HiddenPrints, RepoDir                       # noqa: E402
import test_batched_program as T                                            # noqa: E402
from test_svgd_solver import grasp_gate, pose_gate                          # noqa: E402

WALL = 20.0
CUDA = torch.cuda.is_available()
FAILURES = []
_PROGRAMS = {}
_POSE_SCENES = {}


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"\n          {detail}"))
    if not cond:
        FAILURES.append(name)


def _in_analytic_reach(robot, q, margin=0.01):
    """Whether the analytic formulation can represent the pose of `q` at all: the four
    pre-clip arguments at `(FK(q), psi(q), branch(q))` inside the reachability band with a
    margin. Not a property the benchmark screens for -- a target whose elbow is nearly
    straight (the iiwa's u0 depends on the wrist position alone) is infeasible for this arm
    for every psi -- but a FEASIBLE-BY-CONSTRUCTION cell for this arm needs it."""
    cls = T.ANALYTIC_CLASSES[(robot, "pose")]
    ik = cls.ANALYTIC_IK_CLASS()
    X = np.asarray(ik.FK(q), dtype=float)
    if robot == "panda":
        _, u = ik.Solve(X[:3, :3], X[:3, 3], q[6], ik.gc(q, branches=3))
    else:
        _, u = ik.Solve(X[:3, :3], X[:3, 3], ik.GC(q), ik.psi(q))
    return np.abs(np.asarray(u, dtype=float)).max() < 1 - ProgramOptions().analytic_reach_stepback - margin


def _pose_scene(robot, seed=5):
    """`(diagram, target)`: the hardened pose scene, the target at the frame of a
    COLLISION-FREE configuration (the benchmark's rule; `test_batched_program._scene`'s pose
    target is an unscreened draw, right for row parity and wrong for a solve) that the
    analytic map can represent (`_in_analytic_reach`)."""
    if robot not in _POSE_SCENES:
        oracle = T.program(robot, "pose", "numerical")
        rng = np.random.default_rng(seed)
        while True:
            q_t = T.collision_free_q(oracle, rng, 1)[0]
            if _in_analytic_reach(robot, q_t):
                break
        translation, wxyz = oracle.fk(oracle.ConfigToPlantQ(q_t))
        with HiddenPrints():
            diagram = BuildEnv(meshcat=None, directives_file=SceneFile(T.SCENE_NAMES[robot], "pose", "hardened"))
        _POSE_SCENES[robot] = (diagram, np.concatenate([translation, wxyz]))
    return _POSE_SCENES[robot]


def analytic_program(robot, task):
    """A fresh analytic program per call (the benchmark builds one per cell): the 8-branch
    chart on both robots, the benchmark's analytic options."""
    diagram, target = _pose_scene(robot) if task == "pose" else T._scene(robot, "mug")
    opts = ProgramOptions(visualize=False, joint_centering_cost=1.0, mug_height=0.04,
                          ik_constraint_tol=(1e-4, 0.01), analytic_branches=8)
    with HiddenPrints():
        p = T.ANALYTIC_CLASSES[(robot, task)](diagram, options=opts, model=T._solver(robot))
        if task == "pose":
            p.create_prog(target)
        else:
            p.create_prog(target_mug=target)
    return p


def svgd_options(p, task, **ov):
    path = os.path.join(RepoDir(), "results", f"_test_svgd_analytic_{task}.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return replace(p.options, which_solver="svgd", max_wall_time=WALL,
                   acceptable_constr_viol_tol=1e-4, file_print_name=path, **ov)


def run_cell(robot, task, kernel, start="paired", seed=7, warm=False, **ov):
    p = analytic_program(robot, task)
    p.options = svgd_options(p, task, svgd_kernel=kernel, **ov)
    rng = np.random.default_rng(seed)
    q_init = T.collision_free_q(p, rng, 1)[0]
    with HiddenPrints():
        if start == "paired":
            p.SetStartFromQ(q_init)
        else:
            p.SetNativeStart(q_init, rng)
    if warm:
        p.WarmUpSvgdStep()
    p.PrepareSvgdSolve()
    p.ResetEvalCounts()
    t0 = time.time()
    with HiddenPrints():
        result = p.Solve()
    wall = time.time() - t0
    d = result.get_solver_details()
    gate = pose_gate() if task == "pose" else grasp_gate()
    verdict = bm.verify(p, result, gate, 1e-4, relaxed_tol=1e-3)
    cost = bm.reported_cost(p, result, p.options.joint_centering_cost)
    mode = ov.get("svgd_cuda_graph") and "graphed" or (ov.get("svgd_compile") and "compiled") or "eager"
    tag = f"{robot}/{task} analytic8 kernel={kernel} {start} {mode}"
    print(f"    {tag}: feasible={verdict.feasible} fail={verdict.fail_reason!r} "
          f"max_viol={verdict.detail.get('max_violation'):.2g} cost={cost:.4g} outer={d.iterations} "
          f"steps={d.inner_steps} n_feasible={d.n_feasible} resampled={d.n_resampled} "
          f"stop={d.stop_reason!r} wall={wall:.1f}s step_mode={d.extras.get('step_mode')!r} "
          f"ms/step={1e3 * d.phase_times.get('swarm', 0.0) / max(1, d.inner_steps):.1f}")
    check(f"{tag}: an SvgdResult", isinstance(result, SvgdResult))
    check(f"{tag}: drake_feasible == verify feasible", bool(d.drake_feasible) == bool(verdict.feasible),
          f"{d.drake_feasible} vs {verdict.feasible} ({verdict.fail_reason})")
    check(f"{tag}: wall <= cap + 1", wall <= WALL + 1.0, wall)
    check(f"{tag}: the closed-form split step ran ({d.extras.get('derivative_mode')}, "
          f"{d.extras.get('step_mode')})", d.extras.get("derivative_mode") == "analytic"
          and d.extras.get("step_mode") == mode, d.extras.get("step_mode"))
    check(f"{tag}: map_jacobian counted once per step", d.extras["eval_counts_delta"].get(
        "map_jacobian", 0) >= d.inner_steps, d.extras["eval_counts_delta"])
    return verdict, d


## ------------------------------------------------------------------------------------ ##
##                                       the tests                                      ##
## ------------------------------------------------------------------------------------ ##

def test_split_step_derivatives_match_autograd():
    print("\n--- the split step's closed-form derivatives vs autograd (float64) ---")
    rng = np.random.default_rng(3)
    for robot in ("panda", "iiwa"):
        for task in ("pose", "mug"):
            p = analytic_program(robot, task)
            p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
            p.options = svgd_options(p, task, svgd_kernel="x", svgd_dtype="float64")
            bp = BatchedProgram.from_program(p, dtype=torch.float64)
            tg = _Target(bp, 1e-4)
            X_np = T.analytic_batch(p, rng, 12)
            X = torch.tensor(X_np, dtype=torch.float64, device=bp.device)
            cfg, qp = fused.stage1(tg, X)
            with torch.enable_grad():
                J_q, kin = fused.stage2(tg, X, cfg)
            row, grad = bp.collision_eval(kin[0], need_grad=True)
            out = bp.assemble(X, cfg, cfg, kin, row, grad)
            h, g, finite, _ = tg.scale(out)
            J_h, J_g = tg.row_jacobians_out(out, X, J_q)
            ev = tg.evaluate(X, need_grad=False)
            d_val = max(float((h - ev.h).abs().max()), float((g - ev.g).abs().max()))

            def hg(v):
                o = bp.evaluate(v[None])
                hh, gg, _, _ = tg.scale(o)
                return torch.cat([hh[0], gg[0]])
            err, err_q, err_F = 0.0, 0.0, 0.0
            dFc, dFx = bp.cost_gradient_parts(X, cfg)
            gF = (J_q.transpose(1, 2) @ dFc.unsqueeze(2)).squeeze(2) + dFx
            for i in range(X.shape[0]):
                J_ref = torch.autograd.functional.jacobian(hg, X[i])
                J_cf = torch.cat([J_h[i], J_g[i]], dim=0)
                err = max(err, float((J_cf - J_ref).abs().max() / max(1.0, float(J_ref.abs().max()))))
                Jq_ref = torch.autograd.functional.jacobian(lambda v: bp.evaluate(v[None], need_collision=False).q[0], X[i])
                err_q = max(err_q, float((J_q[i] - Jq_ref).abs().max()))
                gF_ref = torch.autograd.functional.jacobian(lambda v: bp.evaluate(v[None], need_collision=False).F[0], X[i])
                err_F = max(err_F, float((gF[i] - gF_ref).abs().max()))
            print(f"    {robot}/{task}: rows (h, g; reachability incl.) {err:.1e} rel, J_q {err_q:.1e}, "
                  f"objective gradient {err_F:.1e}, stage values vs evaluate {d_val:.1e}")
            check(f"{robot}/{task}: split-step Jacobians == autograd", err <= 1e-10 and err_q <= 1e-12
                  and err_F <= 1e-12 and d_val <= 1e-12, (err, err_q, err_F, d_val))


def test_cells_eager_both_kernels():
    print("\n--- one feasible-by-construction cell per robot x task, paired, eager, kernels x and q ---")
    n_ok, n = 0, 0
    for robot in ("panda", "iiwa"):
        for task in ("pose", "mug"):
            for kernel in ("x", "q"):
                verdict, _ = run_cell(robot, task, kernel)
                n += 1
                n_ok += bool(verdict.feasible)
    print(f"    solved {n_ok}/{n}")


def test_native_start():
    print("\n--- the native start protocol, kernel x ---")
    for robot, task in (("panda", "pose"), ("iiwa", "mug")):
        run_cell(robot, task, "x", start="native")


def test_compiled_and_graphed():
    print("\n--- svgd_compile (and svgd_cuda_graph on CUDA), warmed by WarmUpSvgdStep ---")
    modes = [dict(svgd_compile=True)]
    if CUDA:
        modes.append(dict(svgd_compile=True, svgd_cuda_graph=True))
    for robot, task, kernel in (("panda", "mug", "q"), ("iiwa", "pose", "x")):
        for ov in modes:
            t0 = time.time()
            run_cell(robot, task, kernel, warm=True, **ov)
            print(f"      (with the warm-up compile / capture: {time.time() - t0:.0f} s)")
            ## And a second program of the same structure REUSES the template: rebinding its
            ## branch / offset / target tensors, nothing compiled or captured inside the solve.
            fused.FreezeSvgdSteps()
            try:
                _, d = run_cell(robot, task, kernel, seed=8, **ov)
                check(f"{robot}/{task} {ov}: a second cell reuses the frozen template",
                      d.extras.get("step_reused_template") is True, d.extras.get("step_reused_template"))
            finally:
                fused.FreezeSvgdSteps(False)


def test_latent_kernels_refused():
    p = analytic_program("panda", "pose")
    p.options = replace(p.options, which_solver="svgd")          # the default kernel: cz
    try:
        SvgdSolver(p)
        ok, msg = False, "accepted"
    except ValueError as exc:
        ok, msg = "analytic" in str(exc) and "cz" in str(exc), str(exc)
    check("svgd_kernel=cz (the learned-only default) refused on the analytic arm", ok, msg)


if __name__ == "__main__":
    print(f"device: {'cuda' if CUDA else 'cpu'}")
    try:
        test_latent_kernels_refused()
        test_split_step_derivatives_match_autograd()
        test_cells_eager_both_kernels()
        test_native_start()
        if "--no-compile" not in sys.argv:
            test_compiled_and_graphed()
    finally:
        T.close_all()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
