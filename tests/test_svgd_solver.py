"""End-to-end checks of `src/svgd/solver.py` on the Panda (n6 chart), both arms, both tasks.

Every cell here is FEASIBLE BY CONSTRUCTION: the grasp cells are `test_batched_program`'s
mug scene (the mug welded at the grasp frame of a collision-free configuration), and the
pose cells put the target at the scene frame of a collision-free configuration drawn here.
`test_batched_program`'s own pose target is the frame of a UNIFORM random q with no
collision screen, which is right for row parity and wrong for a solve: the first wave's
"neither arm feasible" signal was measured on a target that can sit inside a shelf, where
both formulations stall at the SAME scaled violation (~269 = 0.027 raw) with the pose rows
at 1e-5 -- a floor of the cell, not of the solver.

What is asserted on every cell: no exception; an `SvgdResult`; `last_iterate` set;
`wall_time <= cap + 1`; `details.map_evals` agrees with `program.eval_counts`;
`drake_feasible == verify's feasible` (same Drake functional, so a disagreement is a bug);
the per-outer trace is present and the right length. The numbers are PRINTED either way.

The default run (`.venv/bin/python tests/test_svgd_solver.py`) is the short set: the
closed-form derivative parity, the row scaling against Drake, the step with the kernel off
against projected gradient descent on the program's own L_rho (full autograd), the grasp
cells (learned and joint space, `native` and `jitter` paired inits), determinism, the
wall-clock stop, NaN injection, the CEM warm-up toggle, the stop reasons, the CUDA-graph
replay, and one cell of `panda_benchmark.py --solver svgd` in a subprocess. The SWEEP over
the pose configurations (`CONFIGS` x arms) is `... sweep`, run separately because it is
20 s cells on the GPU. `... profile` times one step at N in {64, 256}.

Spawns collision workers, so it needs the `__main__` guard. Models: `panda__n6__step620000`.

MEMORY: every pool this file opens has `WORKERS` = 4 processes (the PI's rule after a laptop
OOM on 2026-10-09: 84 pool processes, 48 GB, from 20-worker pools cached across tests and a
benchmark subprocess). The shared pools and the test programs' own pools are CLOSED after
every test (`release`), so no pool outlives the test that opened it; the most alive at once
is the closed-form / row-scaling tests' four BatchedProgram pools (16 workers) plus the
solver's shared pool for one scene (4).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from pydrake.math import RollPitchYaw, RotationMatrix                    # noqa: E402
from pydrake.common.eigen_geometry import Quaternion                    # noqa: E402

from src import benchmark as bm                                         # noqa: E402
from src.generic_program import ProgramOptions, orientation_error_rpy   # noqa: E402
from src.panda_program import PandaIKProgram, PandaIKProgramNumerical  # noqa: E402
from src.svgd import al                                                 # noqa: E402
from src.svgd.result import SvgdResult                                  # noqa: E402
from src.svgd.solver import SvgdSolver, _Target                         # noqa: E402
from src.target_screening import SceneFile                              # noqa: E402
from src.utils import BuildEnv, HiddenPrints, RepoDir                   # noqa: E402
import test_batched_program as T                                        # noqa: E402

WALL = 20.0
N_SWEEP = 64
WORKERS = 4                   # collision workers per pool, everywhere in this file
CONFIGS = [
    ("al_svgd N=1 kernel=q", dict(svgd_n=1, svgd_kernel="q")),
    ("al_svgd N=64 kernel=none", dict(svgd_n=64, svgd_kernel="none")),
    ("al_svgd N=64 kernel=q", dict(svgd_n=64, svgd_kernel="q")),
    ("al_svgd N=64 kernel=q literal", dict(svgd_n=64, svgd_kernel="q", svgd_constraint_inside_kernel=True)),
]
FAILURES = []
_POSE = {}


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"\n          {detail}"))
    if not cond:
        FAILURES.append(name)


## ------------------------------------------------------------------------------------ ##
##                                   cells and gates                                    ##
## ------------------------------------------------------------------------------------ ##

def pose_gate(task_tol=1e-3, ori_tol=0.01):
    def task_gate(program, q):
        translation, wxyz = program.fk(q)
        target = program.target_pose
        axis_max = float(np.max(np.abs(np.asarray(translation) - target[:3])))
        target_rpy = RollPitchYaw(RotationMatrix(Quaternion(target[3:]))).vector()
        rpy_max = float(np.max(np.abs(np.asarray(orientation_error_rpy(wxyz, target_rpy), dtype=float))))
        return axis_max <= task_tol and rpy_max <= ori_tol, dict(pos_error=axis_max, rpy_error=rpy_max)
    return task_gate


def grasp_gate(task_tol=1e-3):
    """`scripts/panda/panda_benchmark.py`'s mug gate: `between_fingers` by name."""
    def task_gate(program, q):
        program.plant.SetPositions(program.plant_context, q)
        grasp = program.plant.GetFrameByName("between_fingers")
        p_W = grasp.CalcPoseInWorld(program.plant_context).translation()
        p_M = program.target_mug.middle.inverse() @ p_W
        axis_error = float(np.linalg.norm(p_M[:2]))
        height = float(abs(p_M[2]))
        ok = axis_error <= task_tol and height <= program.options.mug_height + task_tol
        return ok, dict(axis_error=axis_error, height=height)
    return task_gate


def pose_program(arm, seed=5):
    """A Panda pose program whose target is the scene frame of a COLLISION-FREE q (the
    benchmark's rule), on the hardened pose scene; one diagram shared by both arms."""
    if arm in _POSE:
        return _POSE[arm]
    if "diagram" not in _POSE:
        rng = np.random.default_rng(seed)
        oracle = T.program("panda", "pose", "numerical")       # same scene: a collision oracle
        q_t = T.collision_free_q(oracle, rng, 1)[0]
        translation, wxyz = oracle.fk(oracle.ConfigToPlantQ(q_t))
        _POSE["target"] = np.concatenate([translation, wxyz])
        _POSE["q_target"] = q_t
        with HiddenPrints():
            _POSE["diagram"] = BuildEnv(meshcat=None, directives_file=SceneFile("panda", "pose", "hardened"))
    cls = PandaIKProgram if arm == "learned" else PandaIKProgramNumerical
    with HiddenPrints():
        p = cls(_POSE["diagram"], options=T._options("panda", "pose", arm), model=T._solver("panda"))
        p.create_prog(_POSE["target"])
    _POSE[arm] = p
    return p


def program_for(task, arm):
    return pose_program(arm) if task == "pose" else T.program("panda", "mug", arm)


def run_cell(p, task, label, overrides, wall=WALL, seed=7, start="paired", quiet=False, q_init=None):
    """One solve on `p` under `svgd` with `overrides`; returns `(verdict, details, result)`.
    `q_init` overrides the drawn collision-free start."""
    arm = "learned" if hasattr(p, "z") else "numerical"
    opts = replace(p.options, which_solver="svgd", max_wall_time=wall, acceptable_constr_viol_tol=1e-4,
                   svgd_collision_workers=WORKERS,
                   file_print_name=os.path.join(RepoDir(), "results", f"_test_svgd_{task}_{arm}.log"),
                   **overrides)
    os.makedirs(os.path.dirname(opts.file_print_name), exist_ok=True)
    p.options = opts
    rng = np.random.default_rng(seed)
    if q_init is None:
        q_init = T.collision_free_q(p, rng, 1)[0]
    with HiddenPrints():
        if start == "paired":
            p.SetStartFromQ(q_init)
        else:
            p.SetNativeStart(q_init, rng)
    p.PrepareSvgdSolve()          # the pool and its workers' scenes, outside the clock (as run_grid)
    p.ResetEvalCounts()
    t0 = time.time()
    with HiddenPrints():
        result = p.Solve()
    wall_time = time.time() - t0
    d = result.get_solver_details()
    gate = pose_gate() if task == "pose" else grasp_gate()
    verdict = bm.verify(p, result, gate, 1e-4, relaxed_tol=1e-3)
    cost = bm.reported_cost(p, result, p.options.joint_centering_cost)
    tr = d.extras.get("trace", {})
    if not quiet:
        print(f"  [{task}/{arm}] {label}: feasible={verdict.feasible} fail={verdict.fail_reason!r} "
              f"max_viol={verdict.detail.get('max_violation'):.3g} cost={cost:.4g} "
              f"outer={d.iterations} steps={d.inner_steps} map_evals={d.map_evals} "
              f"n_feasible={d.n_feasible} resampled={d.n_resampled} status={d.status_name!r} "
              f"wall={wall_time:.1f}s collision={d.collision_seconds:.1f}s "
              f"({1e3 * d.collision_seconds / max(1, d.extras.get('pool_calls', 1)):.1f} ms/call) "
              f"phases={ {k: round(v, 2) for k, v in d.phase_times.items()} }")
        if d.inner_steps:
            print(f"      ms/step = {1e3 * d.phase_times.get('swarm', 0.0) / d.inner_steps:.1f}; "
                  f"trace tail: min_infeas {tr.get('min_infeas', [float('nan')])[-1]:.3g} "
                  f"n_feasible {tr.get('n_feasible', [float('nan')])[-1]:.0f} "
                  f"med |lam| {tr.get('med_lam_inf', [float('nan')])[-1]:.3g} "
                  f"med |mu| {tr.get('med_mu_inf', [float('nan')])[-1]:.3g} "
                  f"bound_clip {d.bound_clip:.3g} spread {d.feasible_q_spread}")
    tag = f"[{task}/{arm}] {label}"
    check(f"{tag}: returns an SvgdResult", isinstance(result, SvgdResult))
    check(f"{tag}: last_iterate set", getattr(p, "last_iterate", None) is not None)
    check(f"{tag}: wall_time <= cap + 1", wall_time <= wall + 1.0, f"{wall_time}")
    check(f"{tag}: map_evals agrees with eval_counts",
          ## `verify` and the Drake re-check evaluate single points through QAndPose, so the
          ## program's counter is >= the batched count; the delta recorded by the solver is exact.
          d.map_evals == d.extras["eval_counts_delta"]["map_forward"]
          and d.map_evals <= p.eval_counts["map_forward"], str(p.eval_counts))
    check(f"{tag}: drake_feasible == verify feasible",
          bool(d.drake_feasible) == bool(verdict.feasible), f"{d.drake_feasible} vs {verdict}")
    check(f"{tag}: the trace has one row per outer step",
          all(len(v) == d.iterations for v in tr.values()) and set(tr) == set(_Target.TRACE_COLUMNS),
          f"{ {k: len(v) for k, v in tr.items()} } vs {d.iterations}")
    return verdict, d, result


## ------------------------------------------------------------------------------------ ##
##                                       the tests                                      ##
## ------------------------------------------------------------------------------------ ##

def test_closed_form_derivatives_match_autograd():
    """The solver's derivative path on the fielded robots is CLOSED FORM downstream of the
    configuration (`FrameChain` geometric Jacobians, the pool's collision gradient, identity
    joint limits, closed-form region rows and costs). Pin each piece against autograd of the
    very same batched rows, in float64, on all four Panda programs."""
    print("\n--- closed-form row / cost Jacobians and the FrameChain vs autograd (float64) ---")
    rng = np.random.default_rng(3)
    for task in ("pose", "mug"):
        for arm in ("learned", "numerical"):
            bp = T.batched("panda", task, arm)
            X = torch.tensor(T.lumped_batch(bp, rng, 16), dtype=torch.float64, device=bp.device)
            Xg = X.clone().requires_grad_(True)
            ev = bp.evaluate(Xg, row_jacobians=True)
            cfg, D = ev.q, ev.drake_rows
            N, nr = D.shape
            E = torch.eye(nr, dtype=D.dtype, device=D.device).unsqueeze(1).expand(nr, N, nr)
            (Dq,) = torch.autograd.grad(D, cfg, grad_outputs=E, is_grads_batched=True, retain_graph=True)
            Dq = Dq.transpose(0, 1)
            ok = torch.isfinite(Dq).all(dim=(1, 2)) & (cfg.detach().abs().amax(dim=1) < 100)
            Ja = bp.generic_rows_jacobian_cfg(ev)
            err_rows = (Ja - Dq)[ok].abs().amax().item()
            (gx,) = torch.autograd.grad(ev.F.sum(), Xg, retain_graph=True)
            dFc, dFx = bp.cost_gradient_parts(Xg, cfg)
            if bp.is_learned:
                Jq = bp.jacobian_q(X)
                gx_a = (Jq.transpose(1, 2) @ dFc.unsqueeze(2)).squeeze(2) + dFx
            else:
                gx_a = dFc + dFx
            err_cost = (gx_a - gx)[ok].abs().amax().item()
            err_extra = 0.0
            if bp.extra_blocks:
                ex = torch.cat([ev.extra_rows[k] for k, _ in bp.extra_blocks], dim=1)
                m = ex.shape[1]
                E2 = torch.eye(m, dtype=D.dtype, device=D.device).unsqueeze(1).expand(m, N, m)
                (Jx,) = torch.autograd.grad(ex, Xg, grad_outputs=E2, is_grads_batched=True, retain_graph=True)
                err_extra = (bp.extra_row_jacobian(X) - Jx.transpose(0, 1)).abs().amax().item()
            qp = bp.config_to_plant_q(cfg.detach())
            quat_all, pos_all = bp.fk.body_poses(qp)
            full = [bp.fk.frame_pose(quat_all, pos_all, bp._frame_body, bp._frame_X_BF),
                    bp.fk.frame_pose(quat_all, pos_all, bp._flow_body, bp._flow_X_BF)]
            fast = bp._frame_chain.poses(qp)
            err_chain = max((a[i] - b[i]).abs().max().item() for a, b in zip(fast, full) for i in (0, 1))
            ev2 = bp.evaluate(X, detach_kinematics=True, row_jacobians=True)
            err_det = max((ev2.drake_rows - D).abs().max().item(), (ev2.F - ev.F).abs().max().item())
            ## The SOLVER's path (`_Target.evaluate`: detached kinematics on the learned
            ## arm, no graph at all on joint space) must carry the collision row's gradient
            ## too -- until 2026-10-08 it was read off an autograd node that path never
            ## builds, so the solver's collision Jacobian was silently zero on both arms.
            tg = _Target(bp, 1e-4)
            evs = tg.evaluate(X)
            has_cg = evs.out.extras.get("collision_grad") is not None
            Ds = tg.generic_jacobian_cfg(evs)
            err_solver = (Ds - Dq)[ok].abs().amax().item()
            check(f"panda/{task}/{arm}: the solver path's row Jacobian (collision row included) "
                  f"matches autograd", has_cg and err_solver < 1e-12,
                  f"collision_grad present {has_cg}, err {err_solver}")
            print(f"    panda/{task}/{arm}: rows {err_rows:.2e}  cost {err_cost:.2e}  extra rows "
                  f"{err_extra:.2e}  frame chain {err_chain:.2e}  detached evaluate {err_det:.2e} "
                  f"({int(ok.sum())}/{N} ordinary particles)")
            check(f"panda/{task}/{arm}: closed-form derivatives agree with autograd to 1e-12",
                  max(err_rows, err_cost, err_extra, err_chain, err_det) < 1e-12 and int(ok.sum()) >= 8,
                  f"rows {err_rows} cost {err_cost} extra {err_extra} chain {err_chain} det {err_det}")


def test_grasp_cells():
    """Panda mug, both arms, the two paired-start swarm inits (`jitter`, `native`)."""
    print("\n--- Panda grasp cells (al_svgd N=64 kernel=q), paired start ---")
    for arm in ("learned", "numerical"):
        for init in ("jitter", "native"):
            run_cell(T.program("panda", "mug", arm), "mug", f"init={init}",
                     dict(svgd_n=N_SWEEP, svgd_kernel="q", svgd_paired_init=init))


def _run_fixed_steps(p, N, outer, seed=11, **ov):
    """A solve with the clock off (60 s cap, `outer` outer steps) on a fixed start."""
    _, d, result = run_cell(p, "pose", f"N={N} outer={outer}",
                            dict(svgd_n=N, svgd_kernel="q", svgd_outer_iters=outer,
                                 svgd_stop_patience=10 ** 6, **ov), wall=60.0, seed=seed, quiet=True)
    return result.GetSolution(p.lumped_vars), d


def test_determinism():
    """Same program, same start, N=16, 50 steps (5 outer x 10 inner), kernel q: two solves
    must agree. The swarm is float32 on the GPU; every op in the step is deterministic
    (`index_add_` on disjoint indices, no atomics in the reductions used), the pool is Drake
    in float64, and the seed is derived from the start -- so the contract is BITWISE."""
    print("\n--- determinism: N=16, 50 steps, kernel q, two solves ---")
    p = pose_program("learned")
    xa, da = _run_fixed_steps(p, 16, 5)
    xb, db = _run_fixed_steps(p, 16, 5)
    diff = float(np.max(np.abs(xa - xb)))
    hist_a, hist_b = da.extras["best_history"], db.extras["best_history"]
    print(f"    |x_a - x_b|_inf = {diff:.3g}; best_history equal: {hist_a == hist_b}; "
          f"steps {da.inner_steps} / {db.inner_steps}")
    check("determinism: the returned point is bitwise identical across two solves", diff == 0.0,
          f"max |diff| {diff} (contract: bitwise; <= 1e-10 would be the float32 fallback)")
    check("determinism: the best-merit history is identical", hist_a == hist_b)
    check("determinism: 50 inner steps were taken", da.inner_steps == 50 and db.inner_steps == 50,
          f"{da.inner_steps} {db.inner_steps}")


def test_wall_clock_stop():
    """`max_wall_time = 3` at N=64 stops at the clock with the iterate kept. The pool for
    this scene is spawned and warmed first (`warm_up`), as the benchmark's `WarmUpSvgdStep`
    guarantees before any timed cell."""
    print("\n--- wall-clock stop: max_wall_time=3, N=64 ---")
    p = pose_program("learned")
    p.options = replace(p.options, which_solver="svgd", svgd_n=64, svgd_kernel="q",
                        svgd_collision_workers=WORKERS)
    SvgdSolver(p).warm_up()
    t0 = time.time()
    ## Both budgets lifted explicitly: `run_cell` builds on `p.options`, which an earlier
    ## test may have left with a small `svgd_outer_iters` (the determinism test's 5).
    _, d, _ = run_cell(p, "pose", "wall=3", dict(svgd_n=64, svgd_kernel="q",
                                                 svgd_stop_patience=10 ** 6, svgd_outer_iters=10 ** 6),
                       wall=3.0, quiet=True)
    wall = time.time() - t0
    print(f"    wall {wall:.2f} s, status {d.status_name!r}, outer {d.iterations}, timed_out {d.timed_out}")
    ## `timed_out` records that the swarm was stopped by the clock; the status may still be
    ## `converged` when the stopped swarm's best particle passes the Drake re-check (the
    ## harness rule: a timeout that lands on a valid point is a success).
    check("wall-clock: timed_out is True", d.timed_out, f"{d.status_name} timed_out={d.timed_out}")
    check("wall-clock: wall <= 3.5 s", wall <= 3.5, f"{wall}")
    check("wall-clock: last_iterate set", getattr(p, "last_iterate", None) is not None)


def test_nan_injection():
    """A non-finite particle handed in through `particles_override` is resampled and
    counted, and the solve returns normally."""
    print("\n--- NaN injection via particles_override ---")
    p = pose_program("learned")
    p.options = replace(p.options, which_solver="svgd", svgd_n=16, svgd_kernel="q",
                        svgd_collision_workers=WORKERS, svgd_outer_iters=3, max_wall_time=20.0, acceptable_constr_viol_tol=1e-4,
                        file_print_name="")
    rng = np.random.default_rng(3)
    with HiddenPrints():
        p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
    x0 = np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float)
    X = np.repeat(x0[None, :], 16, axis=0) + 0.01 * rng.standard_normal((16, x0.size))
    X[3, :] = np.nan
    X[7, 8] = np.inf
    p.ResetEvalCounts()
    solver = SvgdSolver(p, particles_override=X)
    result = solver.solve()
    d = result.get_solver_details()
    print(f"    n_resampled {d.n_resampled}, status {d.status_name!r}, returned finite "
          f"{bool(np.all(np.isfinite(result.get_x_val())))}")
    check("nan injection: returns an SvgdResult", isinstance(result, SvgdResult))
    check("nan injection: n_resampled >= 1", d.n_resampled >= 1, str(d.n_resampled))
    check("nan injection: the returned vector is finite", bool(np.all(np.isfinite(result.get_x_val()))))


def test_cem_warmup_toggle():
    print("\n--- CEM warm-up toggle ---")
    p = pose_program("learned")
    _, d, _ = run_cell(p, "pose", "cem warm-up", dict(svgd_n=32, svgd_kernel="q",
                                                      svgd_warmup="cem", svgd_warmup_iters=3,
                                                      svgd_outer_iters=3), wall=20.0, quiet=True)
    w = d.extras.get("warmup")
    print(f"    warmup: {w}; phase {d.phase_times.get('warmup')}")
    check("cem: extras['warmup'] recorded with iters == 3",
          isinstance(w, dict) and w.get("iters") == 3 and np.isfinite(w.get("merit_after", np.nan)), str(w))
    check("cem: warmup phase time recorded", "warmup" in d.phase_times)


def test_benchmark_subprocess():
    """One cell of `panda_benchmark.py --solver svgd`: `summary.json` carries both arms with
    `record["svgd"]` populated. The tag's directory is deleted afterwards."""
    print("\n--- panda_benchmark.py --solver svgd, one cell, subprocess ---")
    tag = "smoke_svgd_test"
    out_dir = os.path.join(RepoDir(), "results", "panda", "benchmark", tag)
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, os.path.join(RepoDir(), "scripts", "panda", "panda_benchmark.py"),
           "--task", "pose", "--targets", "1", "--guesses", "1", "--wall-time", "20",
           "--solver", "svgd", "--arms", "learned,numerical", "--config", "latent", "--tag", tag,
           "--set", f"svgd_collision_workers={WORKERS}"]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(f"    exit {proc.returncode} in {time.time() - t0:.0f} s")
    if proc.returncode != 0:
        print("    stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-25:]))
    check("subprocess: panda_benchmark.py --solver svgd exits 0", proc.returncode == 0)
    path = os.path.join(out_dir, "summary.json")
    check("subprocess: summary.json written", os.path.exists(path), path)
    if os.path.exists(path):
        with open(path) as f:
            payload = json.load(f)
        recs = payload.get("records", {})
        for arm in ("learned", "numerical"):
            rs = recs.get(arm, [])
            ok = len(rs) == 1 and isinstance(rs[0].get("svgd"), dict) and rs[0]["svgd"].get("n_particles")
            print(f"    {arm}: feasible={rs[0].get('feasible') if rs else None} "
                  f"svgd={ {k: rs[0]['svgd'].get(k) for k in ('method', 'n_particles', 'iterations', 'n_feasible', 'drake_feasible')} if ok else None }")
            check(f"subprocess: {arm} record carries a populated record['svgd']", bool(ok), str(rs[:1]))
        check("subprocess: no 'error' records",
              all(r.get("fail_reason") != "error" for rs in recs.values() for r in rs))
    shutil.rmtree(out_dir, ignore_errors=True)


def test_scaled_rows_map_back_to_drake_rows():
    """The row-scaling pin. The solver scales every row by `1 / tol` so that
    `||[h~; g~+]||_inf <= 1` is "feasible at the gate"; `unscale` must recover, for EVERY
    entry, exactly the signed violation of the Drake row it came from -- `value - lb` (eq),
    `lb - value` (lo), `value - ub` (hi) of `prog.EvalBinding(binding, x)` -- and the solver's
    violation times `tol` is the worst Drake violation over those rows. All four Panda
    programs, float64."""
    print("\n--- scaled rows map back to the Drake rows exactly ---")
    rng = np.random.default_rng(11)
    tol = 1e-4
    for task in ("pose", "mug"):
        for arm in ("learned", "numerical"):
            bp = T.batched("panda", task, arm)
            p = bp.program
            by_desc = T.bindings_by_description(p)
            X_np = T.lumped_batch(bp, rng, 8)
            X = torch.tensor(X_np, dtype=torch.float64, device=bp.device)
            worst_map, worst_inf = 0.0, 0.0
            tg = _Target(bp, tol)
            with torch.no_grad():
                ev = tg.evaluate(X, need_grad=False)
            h_u, g_u = tg.unscale(ev.h, ev.g)
            for i in range(X.shape[0]):
                if not (bool(ev.finite[i]) and float(ev.cfg[i].abs().max()) < 100):
                    continue
                xf = bp.to_drake_x(X_np[i])
                vals = {d: np.asarray(p.prog.EvalBinding(b[0], xf), dtype=float).ravel()
                        for d, b in by_desc.items()}
                drake_viol = []
                for col, specs in ((h_u, bp.h_spec), (g_u, bp.g_spec)):
                    for j, r in enumerate(specs):
                        v = vals[r.drake_binding][r.drake_row]
                        ref = {"eq": v - r.lb, "lo": r.lb - v, "hi": v - r.ub}[r.kind]
                        worst_map = max(worst_map, abs(float(col[i, j]) - ref) / max(1.0, abs(ref)))
                        drake_viol.append(abs(ref) if r.kind == "eq" else max(ref, 0.0))
                worst_inf = max(worst_inf, abs(float(ev.infeas[i]) * tol - max(drake_viol))
                                / max(1e-12, max(drake_viol)))
            print(f"    panda/{task}/{arm}: max |unscaled - Drake| (rel) {worst_map:.2e}; "
                  f"|infeas * tol - Drake max violation| (rel) {worst_inf:.2e}")
            check(f"panda/{task}/{arm}: unscaled rows equal the Drake rows' signed violations",
                  worst_map < 1e-9, f"{worst_map}")
            check(f"panda/{task}/{arm}: infeasibility * tol is the Drake max violation",
                  worst_inf < 1e-9, f"{worst_inf}")


def test_stop_reasons_and_log():
    """The swarm's reason for stopping is recorded on the details and in the log --
    `converged` (some particle feasible at the gate, the best feasible objective improved
    by less than `svgd_stop_rel` over `svgd_stop_patience` checks), `step_cap`,
    `wall_clock` -- and the log carries the per-check multiplier trajectories. The
    converged case is the JOINT-SPACE arm started AT the target configuration, so particle 0
    is feasible (v = 0) from the first check: this is a test of the stop rule, not of the
    method's reach. One dual update is taken per check, and the details carry the
    multiplier magnitudes at stop."""
    print("\n--- stop reasons ---")
    p = pose_program("numerical")
    _, d, _ = run_cell(p, "pose", "patience=2", dict(svgd_n=32, svgd_kernel="q", svgd_stop_patience=2,
                                                     svgd_outer_iters=300),
                       wall=20.0, quiet=True, q_init=_POSE["q_target"])
    check("stop: a feasible swarm that stalls stops with stop_reason 'converged'",
          d.stop_reason == "converged" and d.status_name == "converged",
          f"{d.stop_reason} {d.status_name}")
    with open(p.options.file_print_name) as f:
        log = f.read()
    check("stop: the log records the stop reason and the multiplier trajectories",
          "SVGD stop reason: converged" in log and "SVGD trace med_lam_inf = " in log
          and "SVGD trace max_mu_inf = " in log and "SVGD dual updates: " in log, log[-400:])
    check("stop: one dual update per check; multiplier magnitudes on the details",
          d.n_dual_updates == d.iterations and d.lam_inf_max is not None
          and d.lam_inf_median <= d.lam_inf_max and d.mu_inf_median <= d.mu_inf_max,
          f"{d.n_dual_updates} {d.iterations} {d.lam_inf_median} {d.lam_inf_max}")
    p = pose_program("learned")
    _, d, _ = run_cell(p, "pose", "outer=2", dict(svgd_n=16, svgd_kernel="q",
                                                  svgd_stop_patience=10 ** 6, svgd_outer_iters=2),
                       wall=20.0, quiet=True)
    check("stop: the outer-step cap stops with stop_reason 'step_cap'", d.stop_reason == "step_cap",
          d.stop_reason)
    check("stop: multipliers stay inside +-svgd_multiplier_max",
          max(d.extras["trace"]["max_lam_inf"] + d.extras["trace"]["max_mu_inf"])
          <= float(p.options.svgd_multiplier_max), str(d.extras["trace"]["max_lam_inf"]))


def test_step_with_kernel_off_is_pgd_on_the_programs_L():
    """On the real learned pose program (float64 swarm, collision row included), one step
    with `svgd_kernel = none` is projected gradient descent on the program's own L_rho / T in
    the normalised coordinates, `x <- clamp_B(x - (svgd_lr / rho) s^2 grad_x L_i / T)`, with
    the reference gradient from FULL AUTOGRAD through the batched program (flow, kinematics
    and the collision row's autograd node) -- not from the closed forms the step uses. Each
    particle carries its own random multipliers; rho (here 37) is the one shared scalar."""
    print("\n--- one step with the kernel off == PGD on L_rho / T (full autograd reference) ---")
    from dataclasses import replace as dc_replace
    for arm in ("learned", "numerical"):
        p = pose_program(arm)
        p.options = replace(p.options, which_solver="svgd", svgd_n=8, svgd_kernel="none",
                            svgd_collision_workers=WORKERS,
                            svgd_dtype="float64", svgd_compile=False, svgd_cuda_graph=False,
                            svgd_lr=1e-9, svgd_temperature=0.5, svgd_rho=37.0,
                            acceptable_constr_viol_tol=1e-4)
        rng = np.random.default_rng(4)
        with HiddenPrints():
            p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
        s = SvgdSolver(p)
        s._build()
        s._make_runner()
        X, _, _ = s._init_particles(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        S = s._init_state(X)
        g = torch.Generator(device=s.device).manual_seed(0)
        kw = dict(dtype=s.dtype, device=s.device)
        S = dc_replace(S, lam=torch.randn(S.lam.shape, generator=g, **kw),
                       mu=torch.randn(S.mu.shape, generator=g, **kw).abs())
        Xn, _, _ = s._step_split(X, S)
        tg, bp = s._tg, s._tg.bp
        Xg = X.clone().requires_grad_(True)
        out = bp.evaluate(Xg)
        L = al.al_value(out.F, out.h * tg.sh, out.g * tg.sg, S, 37.0)
        (gx,) = torch.autograd.grad(L.sum(), Xg)
        ref, _ = bp.project(X - tg.s ** 2 * (p.options.svgd_lr / 37.0) * gx
                            / p.options.svgd_temperature)
        rel = float((Xn - ref).abs().max() / max(1e-30, float((ref - X).abs().max())))
        print(f"    {arm}: |x_step - x_PGD|_inf / |x_PGD - x|_inf = {rel:.2e}")
        check(f"pgd/{arm}: the kernel-off step is PGD on the program's L_rho / T", rel < 1e-8, f"{rel}")
        p.options = replace(p.options, svgd_kernel="q", svgd_dtype="float32", svgd_n=64,
                            svgd_lr=ProgramOptions().svgd_lr, svgd_temperature=1.0,
                            svgd_rho=ProgramOptions().svgd_rho)


def test_graph_replay_is_bitwise():
    """A captured stage replays BIT-IDENTICALLY to the compiled function it captured, on a
    fixed input -- all three stages, learned and joint-space pose, N=16, float32 -- and the
    compiled stages agree with eager to float32 rounding. Also: a second program of the same
    structure REUSES the cached graphs (constants rebound, no capture), and after
    `FreezeSvgdSteps()` a new structure raises instead of capturing."""
    print("\n--- CUDA-graph replay vs the compiled step, bitwise ---")
    if not torch.cuda.is_available():
        print("    (no CUDA: skipped)")
        return
    from src.svgd import fused
    for arm in ("learned", "numerical"):
        p = pose_program(arm)
        p.options = replace(p.options, which_solver="svgd", svgd_n=16, svgd_kernel="q",
                            svgd_compile=True, svgd_cuda_graph=True, acceptable_constr_viol_tol=1e-4,
                            svgd_collision_workers=WORKERS)
        rng = np.random.default_rng(9)
        with HiddenPrints():
            p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
        s = SvgdSolver(p)
        s.warm_up()
        r = s._make_runner()
        check(f"graph/{arm}: three stages captured", r.mode == "graphed" and r.graphs_captured() >= 3,
              f"{r.mode} {r.graphs_captured()}")
        X, _, _ = s._init_particles(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        tg = r.tg
        S = s._init_state(X)
        g1, g2, g3 = r.fns["s1"], r.fns["s2"], r.fns["s3"]
        cfg, qp = g1(X)
        J_q, kin = g2(X, cfg)
        v, gr = s._pool.eval(qp.to("cpu", torch.float64).numpy())
        col = torch.as_tensor(v).to(X), torch.as_tensor(gr).to(X)
        args3 = (X, cfg, J_q, kin, col[0], col[1], fused.state_dict(S))
        out_g = g3(*args3)
        with torch.no_grad():
            out_c = g3.fn(*args3)
            cfg_c, _ = g1.fn(X)
        with torch.enable_grad():
            J_c, _ = g2.fn(X, cfg)
        flat_g, _ = torch.utils._pytree.tree_flatten(out_g)
        flat_c, _ = torch.utils._pytree.tree_flatten(out_c)
        bit = all(torch.equal(a, b) if a.dtype in (torch.bool, torch.long) else
                  bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())
                  for a, b in zip(flat_g, flat_c) if isinstance(a, torch.Tensor))
        bit12 = torch.equal(cfg, cfg_c) and torch.equal(J_q, J_c)
        with torch.no_grad():
            out_e = fused.stage3(tg, s._sc, *args3)
        dX = float((out_e["X"] - out_g["X"]).abs().max())
        print(f"    {arm}: replay == compiled bitwise: stages 1-2 {bit12}, stage 3 {bit}; "
              f"|X_eager - X_graph| = {dX:.2e}")
        check(f"graph/{arm}: replay is bit-identical to the compiled stages", bit and bit12)
        check(f"graph/{arm}: compiled agrees with eager to float32 rounding", dX < 1e-3, f"{dX}")
        ## a second program of the same structure reuses the graphs
        p2 = TS_second_pose_program(arm)
        p2.options = p.options
        s2 = SvgdSolver(p2)
        s2._build()
        r2 = s2._make_runner()
        check(f"graph/{arm}: a second program of one structure reuses the cached graphs",
              r2.reused and r2.fns is r.fns, f"reused {r2.reused}")
        p.options = replace(p.options, svgd_compile=False, svgd_cuda_graph=False)
    fused.FreezeSvgdSteps()
    try:
        p = pose_program("learned")
        p.options = replace(p.options, svgd_compile=True, svgd_cuda_graph=True, svgd_n=8)
        s = SvgdSolver(p)
        s._build()
        try:
            s._make_runner()
            check("graph: a new structure after the freeze raises", False, "no raise")
        except RuntimeError as exc:
            check("graph: a new structure after the freeze raises", "frozen" in str(exc), str(exc))
    finally:
        fused.FreezeSvgdSteps(False)
        p.options = replace(p.options, svgd_compile=False, svgd_cuda_graph=False)


_POSE2 = {}


def TS_second_pose_program(arm):
    """A second pose program on the same scene with a DIFFERENT target: same structure,
    different constants -- the template-rebinding case."""
    if arm in _POSE2:
        return _POSE2[arm]
    pose_program(arm)
    rng = np.random.default_rng(77)
    oracle = T.program("panda", "pose", "numerical")
    q_t = T.collision_free_q(oracle, rng, 1)[0]
    translation, wxyz = oracle.fk(oracle.ConfigToPlantQ(q_t))
    cls = PandaIKProgram if arm == "learned" else PandaIKProgramNumerical
    with HiddenPrints():
        p = cls(_POSE["diagram"], options=T._options("panda", "pose", arm), model=T._solver("panda"))
        p.create_prog(np.concatenate([translation, wxyz]))
        p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
    _POSE2[arm] = p
    return p


## ------------------------------------------------------------------------------------ ##
##                          the go/no-go sweep and the profile                          ##
## ------------------------------------------------------------------------------------ ##

def sweep():
    """The six pose configurations (four learned, two joint space), 20 s each, paired start,
    on the feasible-by-construction pose cell. The go/no-go: at least one configuration
    per arm is verify-feasible."""
    print("\n--- SWEEP: Panda pose, paired start, 20 s ---")
    for arm, configs in (("learned", CONFIGS), ("numerical", CONFIGS[1:3])):
        any_ok = False
        for label, ov in configs:
            v, _, _ = run_cell(pose_program(arm), "pose", label, ov)
            any_ok |= bool(v.feasible)
        check(f"sweep: {arm} pose -- at least one configuration is verify-feasible (go/no-go)", any_ok)


def profile(Ns=(64, 256)):
    """ms per step of `al_svgd` at N in `Ns`, float32, learned pose, eager split step."""
    print("\n--- PROFILE: al_svgd step time, learned pose, float32 ---")
    p = pose_program("learned")
    for N in Ns:
        p.options = replace(p.options, which_solver="svgd", svgd_n=N, svgd_kernel="q",
                            max_wall_time=60.0, acceptable_constr_viol_tol=1e-4, file_print_name="",
                            svgd_collision_workers=WORKERS)
        rng = np.random.default_rng(7)
        with HiddenPrints():
            p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
        s = SvgdSolver(p)
        s._build()
        s._make_runner()
        X, _, _ = s._init_particles(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        S = s._init_state(X)

        def sync():
            if s.device.type == "cuda":
                torch.cuda.synchronize(s.device)
        for _ in range(3):
            X, _, _ = s._step_split(X, S)
        s._pool.seconds, s._pool.calls = 0.0, 0
        sync()
        t0 = time.perf_counter()
        for _ in range(20):
            X, _, _ = s._step_split(X, S)
        sync()
        dt = (time.perf_counter() - t0) / 20
        print(f"    N={N:4d}: {1e3 * dt:6.1f} ms/step  "
              f"(pool {1e3 * s._pool.seconds / max(1, s._pool.calls):.1f} ms/call, {s.workers} workers)")


def release():
    """Close every collision pool this process holds -- the solver's shared ones and the test
    programs' own BatchedProgram pools -- so none outlives the test that opened it. A later
    `T.batched` rebuilds its program's batched replay (and pool) on demand."""
    from src.svgd.solver import close_shared_pools
    close_shared_pools()
    T.close_all()


def main():
    which = sys.argv[1:] or ["short"]
    tests = []
    if "short" in which:
        tests += [test_closed_form_derivatives_match_autograd, test_scaled_rows_map_back_to_drake_rows,
                  test_grasp_cells, test_determinism, test_wall_clock_stop, test_nan_injection,
                  test_cem_warmup_toggle, test_stop_reasons_and_log,
                  test_step_with_kernel_off_is_pgd_on_the_programs_L, test_graph_replay_is_bitwise,
                  test_benchmark_subprocess]
    if "sweep" in which:
        tests.append(sweep)
    if "profile" in which:
        tests.append(profile)
    for t in tests:
        try:
            t()
        finally:
            release()
    from src.svgd.collision_backend import live_pools, live_workers
    from src.svgd.collision_backend import peak_live_workers
    print(f"    peak live collision workers in this process: {peak_live_workers()}")
    check("no collision pool is live at the end of the file", live_workers() == 0, str(live_pools()))
    print(f"\n{len(FAILURES)} failed")
    for f in FAILURES:
        print("  FAILED:", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
