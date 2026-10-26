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

Models: `panda__n6__step620000`. The collision row runs in this process (no worker processes).
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
from src.svgd import al, fused                                          # noqa: E402
from src.svgd.result import SvgdResult                                  # noqa: E402
from src.svgd.solver import SvgdSolver, _Target                         # noqa: E402
from src.target_screening import SceneFile                              # noqa: E402
from src.utils import BuildEnv, HiddenPrints, RepoDir                   # noqa: E402
import test_batched_program as T                                        # noqa: E402

WALL = 20.0
N_SWEEP = 64
#: The pin these tests were written against -- every svgd round through R19 ran it (kernel q,
#: metric off and init jitter through the then-defaults, rho 1000 named by the manifests; the
#: metric form is no longer an option, the temperature options having been removed 2026-10-14,
#: and `off` at T = 1 is the only step). The
#: defaults moved to c x z + rho 1e5 + H / T + the native-draw paired swarm on 2026-10-12, and
#: `cz` is learned-only, so every solve here names the old pin explicitly: what each test pins
#: is unchanged, and tests/test_svgd_pin_and_gate.py holds the old pin to the pre-change solver
#: bit for bit and exercises the new one.
LEGACY_PIN = dict(svgd_kernel="q", svgd_rho=1000.0, svgd_paired_init="jitter")
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
                   file_print_name=os.path.join(RepoDir(), "results", f"_test_svgd_{task}_{arm}.log"),
                   **{**LEGACY_PIN, **overrides})
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
    p.PrepareSvgdSolve()          # the batched program and a first collision row, outside the clock
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
              f"({1e3 * d.collision_seconds / max(1, d.extras.get('collision_calls', 1)):.1f} ms/call) "
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
    configuration (`FrameChain` geometric Jacobians, the in-process row's collision gradient, identity
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
    (`index_add_` on disjoint indices, no atomics in the reductions used), the collision row is Drake
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
    """`max_wall_time = 3` at N=64 stops at the clock with the iterate kept. The solver is
    warmed first (`warm_up`), as the benchmark's `WarmUpSvgdStep` guarantees before any timed
    cell."""
    print("\n--- wall-clock stop: max_wall_time=3, N=64 ---")
    p = pose_program("learned")
    p.options = replace(p.options, which_solver="svgd", **dict(LEGACY_PIN, svgd_n=64))
    SvgdSolver(p).warm_up()
    t0 = time.time()
    ## Both budgets lifted explicitly: `run_cell` builds on `p.options`, which an earlier
    ## test may have left with a small `svgd_outer_iters` (the determinism test's 5).
    _, d, _ = run_cell(p, "pose", "wall=3", dict(svgd_n=64, svgd_kernel="q",
                                                 svgd_stop_patience=10 ** 6, svgd_outer_iters=10 ** 6),
                       wall=3.0, quiet=True)
    wall = time.time() - t0
    ## The solver's own clock (`solve_seconds`, around `Solve()` only): `wall` above also
    ## holds `run_cell`'s start draw, `SetStartFromQ`, `PrepareSvgdSolve` and the verify,
    ## which the cap does not govern (the harness times the solve the same way).
    solve = float(d.solve_seconds)
    swarm = float(d.phase_times.get("swarm", solve))
    step = swarm / max(int(d.iterations), 1)
    print(f"    wall {wall:.2f} s (solve {solve:.2f}, swarm {swarm:.2f}, {step * 1e3:.0f} ms/outer "
          f"step), status {d.status_name!r}, outer {d.iterations}, timed_out {d.timed_out}")
    ## `timed_out` records that the swarm was stopped by the clock; the status may still be
    ## `converged` when the stopped swarm's best particle passes the Drake re-check (the
    ## harness rule: a timeout that lands on a valid point is a success).
    check("wall-clock: timed_out is True", d.timed_out, f"{d.status_name} timed_out={d.timed_out}")
    ## The clock is checked between outer steps, so the swarm may overrun its deadline
    ## (`max_wall_time` less the reserve) by one outer step, and the selection and Drake
    ## re-check follow. The bound is the cap plus that granularity, not a fixed margin.
    reserve = float(p.options.svgd_time_reserve)
    check("wall-clock: swarm overruns its deadline by < 1.5 outer steps",
          swarm <= 3.0 * (1.0 - reserve) + 1.5 * step, f"swarm {swarm:.2f} step {step:.3f}")
    check("wall-clock: solve <= cap + 1.5 outer steps + 0.5 s",
          solve <= 3.0 + 1.5 * step + 0.5, f"solve {solve:.2f} step {step:.3f}")
    check("wall-clock: last_iterate set", getattr(p, "last_iterate", None) is not None)


def test_nan_injection():
    """A non-finite particle handed in through `particles_override` is resampled and
    counted, and the solve returns normally."""
    print("\n--- NaN injection via particles_override ---")
    p = pose_program("learned")
    p.options = replace(p.options, which_solver="svgd", **dict(LEGACY_PIN, svgd_n=16),
                        svgd_outer_iters=3, max_wall_time=20.0, acceptable_constr_viol_tol=1e-4,
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
           "--solver", "svgd", "--arms", "learned,numerical", "--config", "latent", "--tag", tag]
    for k, v in LEGACY_PIN.items():             # cz, the new default kernel, is learned-only
        cmd += ["--set", f"{k}={v}"]
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
            ## The AL's row units: natural by default (1, or svgd_row_length_scale on the
            ## orientation-type rows), and the tolerance control bit-identical to the old path
            ## (every row times 1/tol); the feasibility verdict the same under both.
            from src.svgd.solver import radian_rows
            rad_h = radian_rows(bp, bp.h_spec)
            tg_n = _Target(bp, tol, units="natural", length_scale=2.5)
            tg_t = _Target(bp, tol, units="tolerance")
            with torch.no_grad():
                ev_n, ev_t = tg_n.evaluate(X, need_grad=False), tg_t.evaluate(X, need_grad=False)
            exp_sh = torch.tensor([2.5 if r else 1.0 for r in rad_h], dtype=X.dtype, device=X.device)
            check(f"panda/{task}/{arm}: natural units -- 1 per row, the length scale on the "
                  f"{sum(rad_h)} orientation-type rows, never on the joint limits",
                  torch.equal(tg_n.sh, exp_sh)
                  and not any(r and spec.group == "joint_limit" for r, spec in zip(rad_h, bp.h_spec))
                  and (task != "pose" or sum(rad_h) >= 3), f"{tg_n.sh}")
            old_h = ev_t.out.h * torch.full((len(bp.h_spec),), 1.0 / tol, dtype=X.dtype, device=X.device)
            old_g = ev_t.out.g * torch.full((len(bp.g_spec),), 1.0 / tol, dtype=X.dtype, device=X.device)
            check(f"panda/{task}/{arm}: the tolerance control is bit-identical to the old path, "
                  f"and the feasibility verdict is the same in both units",
                  torch.equal(ev_t.h, old_h) and torch.equal(ev_t.g, old_g)
                  and torch.equal(ev_n.infeas, ev_t.infeas))
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
    ## A tiny identity-metric step keeps particle 0 at the target: this isolates the stop
    ## RULE from the dynamics (which are not under test here).
    _, d, _ = run_cell(p, "pose", "patience=2", dict(svgd_n=32, svgd_kernel="q", svgd_stop_patience=2,
                                                     svgd_outer_iters=300, svgd_metric="identity",
                                                     svgd_lr=1e-6),
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
    ## Multiplier clips by row group, on the cluster's step (compiled + CUDA graphs) with a
    ## clip bound small enough to bind: the groups are the program's own and sum to the total.
    ov = dict(svgd_n=16, svgd_kernel="q", svgd_stop_patience=10 ** 6, svgd_outer_iters=3,
              svgd_multiplier_max=1e-3)
    if torch.cuda.is_available():
        ov.update(svgd_compile=True, svgd_cuda_graph=True)
    _, d, _ = run_cell(p, "pose", "clips by group", ov, wall=60.0, quiet=True)
    groups = d.multiplier_clips_by_group
    print(f"    multiplier clips by group ({'graphed' if ov.get('svgd_cuda_graph') else 'eager'}): "
          f"{groups}; total {d.n_multiplier_clipped}; median |.|_inf by group "
          f"{ {k: round(v, 4) for k, v in d.multiplier_inf_median_by_group.items()} }")
    check("clips by group: keyed by the program's row groups, summing to the total (> 0)",
          isinstance(groups, dict) and set(groups) == set(d.multiplier_inf_median_by_group)
          and {"pose_pos", "pose_rpy", "collision", "joint_limit_lo", "joint_limit_hi"} <= set(groups)
          and sum(groups.values()) == d.n_multiplier_clipped > 0, f"{groups} {d.n_multiplier_clipped}")
    check("clips by group: per-group median |.|_inf at stop inside the clip bound",
          all(0.0 <= v <= 1e-3 * (1 + 1e-6) for v in d.multiplier_inf_median_by_group.values()),  # float32
          str(d.multiplier_inf_median_by_group))
    p.options = replace(p.options, svgd_compile=False, svgd_cuda_graph=False,
                        svgd_multiplier_max=ProgramOptions().svgd_multiplier_max)


def _latent_pose_program(weight):
    """A Panda pose LEARNED program on `pose_program`'s scene and target, built with
    `latent_cost_weight = weight`: the costs are fixed at `create_prog`, so the regulariser
    needs its own program (cached by weight)."""
    key = ("learned", "latent", float(weight))
    if key not in _POSE:
        pose_program("learned")                                # the shared diagram and target
        with HiddenPrints():
            p = PandaIKProgram(_POSE["diagram"], model=T._solver("panda"),
                               options=replace(T._options("panda", "pose", "learned"),
                                               latent_cost_weight=float(weight)))
            p.create_prog(_POSE["target"])
        _POSE[key] = p
    return _POSE[key]


def test_stop_select_objective():
    """`svgd_stop_select_objective` (stage SVGD_R29): which objective the stall rule and the
    selection read. On a Panda pose learned cell with `latent_cost_weight = 1` (so the full
    objective and the shared one differ by a material regulariser), the final selection's
    candidates are re-derived from an INDEPENDENT batched evaluation of the selection's own
    swarm: under 'shared' the selected particle is the argmin of `F_reported` among those
    feasible at the solve's `feas_level`, under 'full' the argmin of `F`. The selection is
    captured by wrapping `SvgdSolver._select` (its last call is the one returned). An unknown
    value is refused at options construction."""
    print("\n--- stop / select objective: 'shared' vs 'full' (latent_cost_weight=1) ---")
    try:
        ProgramOptions(svgd_stop_select_objective="bogus")
        refused = False
    except ValueError:
        refused = True
    check("stop/select objective: an unknown value raises at ProgramOptions construction", refused)
    p = _latent_pose_program(1.0)
    names = {b.evaluator().get_description() for b in p.prog.GetAllCosts()}
    check("stop/select objective: the program carries the latent regulariser",
          "LatentRegularizerCost" in names, str(names))
    orig = SvgdSolver._select
    for mode in ("shared", "full"):
        captured = {}

        def spy(self, X, S, stop, feas_level):
            sel = orig(self, X, S, stop, feas_level)
            captured.update(X=X.detach().clone(), sel=sel, bp=self._tg.bp, feas_level=feas_level)
            return sel
        SvgdSolver._select = spy
        try:
            _, d, _ = run_cell(p, "pose", f"objective={mode}",
                               dict(svgd_n=32, svgd_kernel="q", svgd_outer_iters=300,
                                    svgd_stop_select_objective=mode), wall=30.0, quiet=True)
        finally:
            SvgdSolver._select = orig
        tag = f"stop/select objective={mode}"
        check(f"{tag}: the solve stops on the stall rule (converged)",
              d.stop_reason == "converged" and d.status_name == "converged",
              f"{d.stop_reason} {d.status_name}")
        sel, bp = captured["sel"], captured["bp"]
        with torch.no_grad():
            ev = bp.evaluate(captured["X"])
        F_full = torch.nan_to_num(ev.F, nan=float("inf")).cpu().double().numpy()
        F_shared = torch.nan_to_num(ev.F_reported, nan=float("inf")).cpu().double().numpy()
        feas = np.asarray(sel.feas, dtype=bool)
        chosen, other = (F_shared, F_full) if mode == "shared" else (F_full, F_shared)
        amin = int(np.argmin(np.where(feas, chosen, np.inf)))
        amin_other = int(np.argmin(np.where(feas, other, np.inf)))
        print(f"    [{mode}] n_feasible {sel.n_feasible} at feas_level {captured['feas_level']:.3g}; "
              f"selected {sel.selected}, candidates {sel.cands}; argmin of the chosen objective "
              f"{amin}, of the other {amin_other}; F {F_full[sel.selected]:.6g} "
              f"F_reported {F_shared[sel.selected]:.6g}; outer {d.iterations}")
        check(f"{tag}: F_reported is finite wherever F is",
              bool(np.all(np.isfinite(F_shared[np.isfinite(F_full)]))))
        check(f"{tag}: the candidates are the feasible particles in order of the chosen objective",
              sel.n_feasible > 0 and sel.cands == [int(i) for i in np.argsort(
                  np.where(feas, chosen, np.inf), kind="stable")[:min(len(sel.cands), sel.n_feasible)]],
              f"{sel.cands}")
        ## The first candidate is selected unless the exact Drake re-check rejected it.
        check(f"{tag}: the selected particle is the argmin of the chosen objective among the "
              f"feasible", sel.selected == amin or (sel.cands[0] == amin and sel.selected in sel.cands[1:]),
              f"selected {sel.selected} argmin {amin} candidates {sel.cands}")


def test_polish():
    """`svgd_polish` (a smoke, 2026-10-18): the float64 minimum-norm Gauss-Newton polish of
    the returned particle. On the Panda pose learned cell under 'newton': the solve converges,
    a polish ran, the worst signed Drake violation did not rise and stays within the gate, and
    `verify` (the harness's own re-measurement) agrees the point is feasible. Under 'none' no
    polish ran and the per-cell svgd record carries no polish_* key. An unknown value is
    refused at options construction."""
    print("\n--- polish: 'newton' vs 'none' (Panda pose, learned) ---")
    try:
        ProgramOptions(svgd_polish="bogus")
        refused = False
    except ValueError:
        refused = True
    check("polish: an unknown value raises at ProgramOptions construction", refused)
    p = pose_program("learned")
    tol = 1e-4
    for mode in ("newton", "none"):
        v, d, result = run_cell(p, "pose", f"polish={mode}",
                                dict(svgd_n=32, svgd_kernel="q", svgd_outer_iters=300,
                                     svgd_polish=mode, svgd_polish_steps=3), wall=30.0, quiet=True)
        rec = bm.solver_diagnostics(result, "svgd").get("svgd", {})
        pol = d.polish
        tag = f"polish={mode}"
        print(f"    [{mode}] stop {d.stop_reason}, drake {d.drake_feasible}, verify max_violation "
              f"{v.detail.get('max_violation'):.3e}, drake_max_violation "
              f"{d.extras.get('drake_max_violation'):.3e}; polish {pol}")
        check(f"{tag}: the solve converges and is Drake-feasible",
              d.stop_reason == "converged" and d.drake_feasible and v.feasible,
              f"{d.stop_reason} {d.drake_feasible} {v.feasible}")
        if mode == "newton":
            check(f"{tag}: a polish ran and is recorded per cell",
                  pol is not None and all(k in rec for k in pol), f"{pol} {sorted(rec)}")
            if pol is not None:
                check(f"{tag}: polish_viol_after <= polish_viol_before",
                      pol["polish_viol_after"] <= pol["polish_viol_before"], str(pol))
                check(f"{tag}: polish_viol_after <= tol", pol["polish_viol_after"] <= tol, str(pol))
                check(f"{tag}: drake_max_violation is the polished one",
                      d.extras.get("drake_max_violation") == pol["polish_viol_after"])
        else:
            check(f"{tag}: no polish ran and no polish_* key is recorded",
                  pol is None and not any(k.startswith("polish") for k in rec), f"{pol} {sorted(rec)}")


def test_polish_holds_satisfied_active_rows():
    """`svgd_polish = newton`'s targeting, at unit level (seconds, no network): the solver's own
    `_polish_newton` / `_drake_rows` / `_drake_max_violation` on a three-variable Drake program
    with a nonlinear equality (residual 3e-5), an inequality SATISFIED within the gate of its
    bound (slack 5e-5: held at its value, not spent) and one VIOLATED within the gate (by 2e-5:
    driven to its bound). Every variable is lumped; there are no variable bounds."""
    print("\n--- polish: a satisfied active inequality is held, a violated one driven (unit) ---")
    from types import SimpleNamespace
    from pydrake.solvers import MathematicalProgram
    tol = 1e-4
    x1 = 0.2
    x_start = np.array([x1 + 5e-5, x1, 0.5 + 2e-5 - x1])
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(3, "x")
    c = x_start[0] + x_start[1] + x_start[2] + 0.1 * x_start[1] ** 2 - 3e-5
    prog.AddConstraint(x[0] + x[1] + x[2] + 0.1 * x[1] ** 2 - c == 0)          # equality
    held_b = prog.AddLinearConstraint(x[0] - x[1] >= 0)                       # satisfied, slack 5e-5
    driven_b = prog.AddLinearConstraint(x[1] + x[2] <= 0.5)                  # violated by 2e-5
    ident = lambda v: np.array(v, dtype=float).copy()
    s = SimpleNamespace(program=SimpleNamespace(prog=prog), tol=tol,
                        options=SimpleNamespace(svgd_polish_steps=3),
                        _tg=SimpleNamespace(bp=SimpleNamespace(to_drake_x=ident, from_drake_x=ident,
                                                               _lumped_idx=np.arange(3))))
    s._drake_rows = lambda xf, with_gradient=False: SvgdSolver._drake_rows(s, xf, with_gradient)
    s._drake_max_violation = lambda xf: SvgdSolver._drake_max_violation(s, xf)
    viol0 = s._drake_max_violation(x_start)
    x_out, viol, pol = SvgdSolver._polish_newton(s, x_start, viol0)
    ## Slack to the nearer finite bound, read from the binding (Drake may store a row negated).
    def slack(b, xf):
        v = float(np.asarray(prog.EvalBinding(b, xf)).reshape(-1)[0])
        lb, ub = float(b.evaluator().lower_bound()[0]), float(b.evaluator().upper_bound()[0])
        return min(v - lb, ub - v)
    held0, held1 = slack(held_b, x_start), slack(held_b, x_out)
    driven0, driven1 = slack(driven_b, x_start), slack(driven_b, x_out)
    print(f"    viol {viol0:.3e} -> {viol:.3e}; held slack {held0:.6e} -> {held1:.6e}; "
          f"driven slack {driven0:+.3e} -> {driven1:+.3e}; stats {pol}")
    check("polish unit: the starting point is within the gate with a violated row", 0 < viol0 <= tol, f"{viol0}")
    check("polish unit: at least one step was accepted", pol["polish_steps"] >= 1, str(pol))
    check("polish unit: the active-inequality counts are 2 = 1 held + 1 driven",
          (pol["polish_active_ineq"], pol["polish_active_ineq_held"], pol["polish_active_ineq_driven"])
          == (2, 1, 1), str(pol))
    check("polish unit: the held row did not move toward its bound (beyond 1e-9)",
          held1 >= held0 - 1e-9, f"{held0} -> {held1}")
    check("polish unit: the driven row now satisfies its bound (to 1e-12)",
          driven0 < 0 and driven1 >= -1e-12, f"{driven0} -> {driven1}")
    check("polish unit: the worst violation fell and is the returned one",
          viol <= viol0 and viol == pol["polish_viol_after"]
          and viol == s._drake_max_violation(x_out), f"{viol0} -> {viol} {pol}")


def test_step_with_kernel_off_is_pgd_on_the_programs_L():
    """On the real pose programs (float64 swarm, collision row included), one step with
    `svgd_kernel = none` and `svgd_metric = identity` is projected gradient descent on the
    program's own L_rho in the normalised coordinates,
    `x <- clamp_B(x - (svgd_lr / ||H_i||_F) s^2 grad_x L_i)`, with the reference gradient
    from FULL AUTOGRAD through the batched program (flow, kinematics and the collision row's
    autograd node) -- not from the closed forms the step uses -- on the rows in NATURAL units.
    Each particle carries its own random multipliers; rho (here 37) is the one shared scalar."""
    print("\n--- one step with the kernel off == PGD on L_rho (full autograd reference) ---")
    from dataclasses import replace as dc_replace
    for arm in ("learned", "numerical"):
        p = pose_program(arm)
        p.options = replace(p.options, which_solver="svgd",
                            **dict(LEGACY_PIN, svgd_kernel="none", svgd_rho=37.0),
                            svgd_n=8, svgd_dtype="float64", svgd_compile=False, svgd_cuda_graph=False,
                            svgd_lr=0.7,
                            svgd_metric="identity", acceptable_constr_viol_tol=1e-4)
        rng = np.random.default_rng(4)
        with HiddenPrints():
            p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
        s = SvgdSolver(p)
        s._build()
        r = s._make_runner()
        X, _, _ = s._init_particles(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        S = s._init_state(X)
        g = torch.Generator(device=s.device).manual_seed(0)
        kw = dict(dtype=s.dtype, device=s.device)
        S = dc_replace(S, lam=torch.randn(S.lam.shape, generator=g, **kw),
                       mu=torch.randn(S.mu.shape, generator=g, **kw).abs())
        tg, bp = s._tg, s._tg.bp
        cfg, qp = r.s1(X)
        J_q, kin = r.s2(X, cfg)
        v, gr = s._col.eval(qp.to("cpu", torch.float64).numpy())
        o = r.s3(X, cfg, J_q, kin, torch.as_tensor(v).to(X), torch.as_tensor(gr).to(X),
                 fused.state_dict(S))
        Xg = X.clone().requires_grad_(True)
        out = bp.evaluate(Xg)
        L = al.al_value(out.F, out.h * tg.sh, out.g * tg.sg, S, 37.0)
        (gx,) = torch.autograd.grad(L.sum(), Xg)
        fro = o["H"].flatten(1).norm(dim=1).unsqueeze(1)
        ref, _ = bp.project(X - tg.s ** 2 * (p.options.svgd_lr / fro) * gx)
        rel = float((o["X"] - ref).abs().max() / max(1e-30, float((ref - X).abs().max())))
        print(f"    {arm}: |x_step - x_PGD|_inf / |x_PGD - x|_inf = {rel:.2e}")
        check(f"pgd/{arm}: the kernel-off identity step is PGD on the program's L_rho",
              rel < 1e-8, f"{rel}")
        p.options = replace(p.options, **LEGACY_PIN, svgd_dtype="float32", svgd_n=64,
                            svgd_lr=ProgramOptions().svgd_lr,
                            svgd_metric=ProgramOptions().svgd_metric)


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
        p.options = replace(p.options, which_solver="svgd", **dict(LEGACY_PIN, svgd_n=16),
                            svgd_compile=True, svgd_cuda_graph=True, acceptable_constr_viol_tol=1e-4)
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
        v, gr = s._col.eval(qp.to("cpu", torch.float64).numpy())
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
        ## Eager vs compiled is compared on the quantities BEFORE the metric solve (the rows,
        ## the objective, the GN Hessian): the Newton solve amplifies float32 reordering by the
        ## Hessian's conditioning, so the stepped X is not a rounding-level check.
        rel = lambda a, b: float((a - b).abs().max() / max(1.0, float(b.abs().max())))
        dpre = max(rel(out_e[k], out_g[k]) for k in ("h", "g", "F", "H"))
        dX = float((out_e["X"] - out_g["X"]).abs().max())
        print(f"    {arm}: replay == compiled bitwise: stages 1-2 {bit12}, stage 3 {bit}; "
              f"eager vs compiled (rows, F, H, relative) {dpre:.2e}; |X_eager - X_graph| = {dX:.2e}")
        check(f"graph/{arm}: replay is bit-identical to the compiled stages", bit and bit12)
        check(f"graph/{arm}: compiled agrees with eager to float32 rounding before the solve",
              dpre < 1e-3, f"{dpre}")
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
        p.options = replace(p.options, which_solver="svgd", **dict(LEGACY_PIN, svgd_n=N),
                            max_wall_time=60.0, acceptable_constr_viol_tol=1e-4, file_print_name="")
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
            X = s._step_split(X, S)[0]
        s._col.reset()
        sync()
        t0 = time.perf_counter()
        for _ in range(20):
            X = s._step_split(X, S)[0]
        sync()
        dt = (time.perf_counter() - t0) / 20
        print(f"    N={N:4d}: {1e3 * dt:6.1f} ms/step  "
              f"(collision row {1e3 * s._col.seconds / max(1, s._col.calls):.1f} ms/call, in process)")


def release():
    """Forget the test programs' cached batched replays between tests (a later `T.batched`
    rebuilds one on demand)."""
    T.close_all()


def main():
    which = sys.argv[1:] or ["short"]
    tests = []
    if "short" in which:
        tests += [test_closed_form_derivatives_match_autograd, test_scaled_rows_map_back_to_drake_rows,
                  test_grasp_cells, test_determinism, test_wall_clock_stop, test_nan_injection,
                  test_cem_warmup_toggle, test_stop_reasons_and_log, test_stop_select_objective,
                  test_polish, test_polish_holds_satisfied_active_rows,
                  test_step_with_kernel_off_is_pgd_on_the_programs_L, test_graph_replay_is_bitwise,
                  test_benchmark_subprocess]
    if "polish" in which:
        tests += [test_polish_holds_satisfied_active_rows, test_polish]
    if "sweep" in which:
        tests.append(sweep)
    if "profile" in which:
        tests.append(profile)
    for t in tests:
        try:
            t()
        finally:
            release()
    print(f"\n{len(FAILURES)} failed")
    for f in FAILURES:
        print("  FAILED:", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
