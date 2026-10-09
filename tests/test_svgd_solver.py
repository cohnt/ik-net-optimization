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
closed-form derivative parity, the grasp cells (learned and joint space, `native` and
`jitter` paired inits), determinism, the wall-clock stop, NaN injection, the CEM warm-up
toggle, and one cell of `panda_benchmark.py --solver svgd` in a subprocess. The go/no-go
SWEEP over the six pose configurations (`CONFIGS` x arms) is `... sweep`, run separately
because it is six 20 s cells on the GPU. `... profile` times one step at N in {64, 256}.

Spawns collision workers, so it needs the `__main__` guard. Models: `panda__n6__step620000`.
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
from src.generic_program import orientation_error_rpy                   # noqa: E402
from src.panda_program import PandaIKProgram, PandaIKProgramNumerical  # noqa: E402
from src.svgd import al                                                 # noqa: E402
from src.svgd.result import SvgdResult                                  # noqa: E402
from src.svgd.solver import SvgdSolver, _Target                         # noqa: E402
from src.target_screening import SceneFile                              # noqa: E402
from src.utils import BuildEnv, HiddenPrints, RepoDir                   # noqa: E402
import test_batched_program as T                                        # noqa: E402

WALL = 20.0
N_SWEEP = 64
CONFIGS = [
    ("al_svgd N=1 kernel=none", dict(svgd_method="al_svgd", svgd_n=1, svgd_kernel="none")),
    ("al_svgd N=64 kernel=none", dict(svgd_method="al_svgd", svgd_n=64, svgd_kernel="none")),
    ("al_svgd N=64 kernel=q", dict(svgd_method="al_svgd", svgd_n=64, svgd_kernel="q")),
    ("tsvgd N=64 kernel=q", dict(svgd_method="tsvgd", svgd_n=64, svgd_kernel="q")),
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


def run_cell(p, task, label, overrides, wall=WALL, seed=7, start="paired", quiet=False):
    """One solve on `p` under `svgd` with `overrides`; returns `(verdict, details, result)`."""
    arm = "learned" if hasattr(p, "z") else "numerical"
    opts = replace(p.options, which_solver="svgd", max_wall_time=wall, acceptable_constr_viol_tol=1e-4,
                   file_print_name=os.path.join(RepoDir(), "results", f"_test_svgd_{task}_{arm}.log"),
                   **overrides)
    os.makedirs(os.path.dirname(opts.file_print_name), exist_ok=True)
    p.options = opts
    rng = np.random.default_rng(seed)
    q_init = T.collision_free_q(p, rng, 1)[0]
    with HiddenPrints():
        if start == "paired":
            p.SetStartFromQ(q_init)
        else:
            p.SetNativeStart(q_init, rng)
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
                  f"med_rho {tr.get('med_rho', [float('nan')])[-1]:.3g} "
                  f"med_lr {tr.get('med_lr', [float('nan')])[-1]:.3g} "
                  f"frac_gn_clamped {tr.get('frac_gn_clamped', [float('nan')])[-1]:.2f}")
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
                     dict(svgd_method="al_svgd", svgd_n=N_SWEEP, svgd_kernel="q", svgd_paired_init=init))


def _run_fixed_steps(p, N, outer, seed=11, **ov):
    """A solve with the clock off (60 s cap, `outer` outer steps) on a fixed start."""
    _, d, result = run_cell(p, "pose", f"N={N} outer={outer}",
                            dict(svgd_method="al_svgd", svgd_n=N, svgd_kernel="q", svgd_outer_iters=outer,
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
    this scene is already spawned and warm (the preceding cells), as the benchmark's
    `WarmUpSvgdStep` guarantees before any timed cell."""
    print("\n--- wall-clock stop: max_wall_time=3, N=64 ---")
    p = pose_program("learned")
    p.options = replace(p.options, which_solver="svgd", svgd_n=64, svgd_kernel="q", svgd_method="al_svgd")
    SvgdSolver(p).warm_up()
    t0 = time.time()
    ## Both budgets lifted explicitly: `run_cell` builds on `p.options`, which an earlier
    ## test may have left with a small `svgd_outer_iters` (the determinism test's 5).
    _, d, _ = run_cell(p, "pose", "wall=3", dict(svgd_method="al_svgd", svgd_n=64, svgd_kernel="q",
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
    p.options = replace(p.options, which_solver="svgd", svgd_n=16, svgd_kernel="q", svgd_method="al_svgd",
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
    _, d, _ = run_cell(p, "pose", "cem warm-up", dict(svgd_method="al_svgd", svgd_n=32, svgd_kernel="q",
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
    """ms per step of `al_svgd` at N in `Ns`, GN every step and never, float32, learned pose."""
    print("\n--- PROFILE: al_svgd step time, learned pose, float32 ---")
    p = pose_program("learned")
    for N in Ns:
        p.options = replace(p.options, which_solver="svgd", svgd_n=N, svgd_kernel="q", svgd_method="al_svgd",
                            max_wall_time=60.0, acceptable_constr_viol_tol=1e-4, file_print_name="")
        rng = np.random.default_rng(7)
        with HiddenPrints():
            p.SetStartFromQ(T.collision_free_q(p, rng, 1)[0])
        s = SvgdSolver(p)
        s._build()
        tg = s._tg
        X, _, _ = s._init_particles(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
        S = al.ALState.init(N, s.n, tg.m_e, tg.m_i, rho0=10.0, eta0=1.0, lr0=0.05, dtype=s.dtype, device=s.device)
        s._eta0 = 1.0

        def sync():
            if s.device.type == "cuda":
                torch.cuda.synchronize(s.device)
        for t in range(3):
            X, S, _ = s._step_al(X, S, 0, t, 100, True)
        sync()
        for do_gn in (True, False):
            s._pool.seconds, s._pool.calls = 0.0, 0
            sync()
            t0 = time.perf_counter()
            for t in range(20):
                X, S, _ = s._step_al(X, S, 0, t, 100, do_gn)
            sync()
            dt = (time.perf_counter() - t0) / 20
            print(f"    N={N:4d} do_gn={do_gn!s:5s}: {1e3 * dt:6.1f} ms/step  "
                  f"(pool {1e3 * s._pool.seconds / max(1, s._pool.calls):.1f} ms/call, {s.workers} workers)")


def main():
    which = sys.argv[1:] or ["short"]
    if "short" in which:
        test_closed_form_derivatives_match_autograd()
        test_grasp_cells()
        test_determinism()
        test_wall_clock_stop()
        test_nan_injection()
        test_cem_warmup_toggle()
        test_benchmark_subprocess()
    if "sweep" in which:
        sweep()
    if "profile" in which:
        profile()
    T.close_all()
    print(f"\n{len(FAILURES)} failed")
    for f in FAILURES:
        print("  FAILED:", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
