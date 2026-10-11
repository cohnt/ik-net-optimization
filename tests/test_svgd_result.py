"""Pin the svgd solver's duck-typed result against what the harness actually reads.

There is no test suite in this repo; run this by hand:

    python tests/test_svgd_result.py

`SvgdResult` (src/svgd/result.py) stands in for Drake's `MathematicalProgramResult` on the
`svgd` column, because a hand-built Drake result ABORTS THE INTERPRETER (`set_x_val`'s C++
assert). The harness reads a result through exactly seven calls, and this file drives a real
`svgd` solve on a Panda joint-space program through every one of them the way `run_grid`
does -- `verify`, `reported_cost`, `solver_diagnostics`, `parse_log`, `summarise` -- so a
contract break shows up here rather than as a column of `fail_reason="error"`.

This is a CONTRACT test, not a quality one: the solve is a 5 s Panda joint-space pose cell
and nothing here asserts that it converges. What is asserted of the returned point is that
it is a real point of the program -- finite, inside the variable bounds, not necessarily the
start -- that `is_success()` agrees with the solver's own Drake re-check, and that the status
is one of the real statuses with its name on the log's exit line.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
from pydrake.solvers import SolutionResult
from pydrake.symbolic import Variable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)

import src.benchmark as bm                                              # noqa: E402
from src.generic_program import ProgramOptions                          # noqa: E402
from src.panda_program import PandaIKProgram, PandaIKProgramNumerical  # noqa: E402
from src.svgd.result import (SvgdResult, STATUS_NAMES,                   # noqa: E402
                             STATUS_NOT_IMPLEMENTED, status_name)
from src.utils import BuildEnv, HiddenPrints                            # noqa: E402

SCENE = os.path.join(REPO, "models/panda/panda_finray_collision_hardened.yaml")
FAILURES = []
CHECKS = [0]


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          {detail}")
        FAILURES.append(name)


def a_reachable_target():
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=SCENE)
        sampler = PandaIKProgram(diagram, options=ProgramOptions())
        sampler.create_prog()
    rng = np.random.default_rng(0)
    q = rng.uniform(sampler.plant.GetPositionLowerLimits(),
                    sampler.plant.GetPositionUpperLimits())
    translation, wxyz = sampler.fk(q)
    return np.concatenate([translation, wxyz]), sampler


def test_svgd_result_duck_type():
    print("\n--- svgd result: the seven calls the harness makes ---")
    log = os.path.join(REPO, "results/_test_svgd_result.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    target, sampler = a_reachable_target()
    opts = ProgramOptions(which_solver="svgd", max_wall_time=5.0, file_print_name=log,
                          collision_avoidance=True)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=SCENE)
        p = PandaIKProgramNumerical(diagram, options=opts)
        p.create_prog(target)
    rng = np.random.default_rng(1)
    q_init = rng.uniform(sampler.plant.GetPositionLowerLimits(),
                         sampler.plant.GetPositionUpperLimits())
    p.SetStartFromQ(q_init)
    x0 = np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float)

    with HiddenPrints():
        result = p.Solve()

    check("Solve() under svgd returns an SvgdResult", isinstance(result, SvgdResult),
          type(result).__name__)
    check("is_success() is a bool", isinstance(result.is_success(), bool))
    check("get_solution_result() is a real pydrake SolutionResult",
          isinstance(result.get_solution_result(), SolutionResult),
          repr(result.get_solution_result()))
    x_full = result.get_x_val()
    check("get_x_val() has prog.num_vars() entries",
          x_full.shape == (p.prog.num_vars(),), str(x_full.shape))
    lumped = result.GetSolution(p.lumped_vars)
    check("GetSolution(lumped_vars) is an ndarray the length of lumped_vars",
          isinstance(lumped, np.ndarray) and lumped.shape == (len(p.lumped_vars),),
          str(getattr(lumped, "shape", None)))
    check("GetSolution(lumped_vars) agrees with get_x_val() scattered by Drake index",
          np.array_equal(lumped, x_full[p.prog.FindDecisionVariableIndices(p.lumped_vars)]))
    single = result.GetSolution(p.lumped_vars[2])
    check("GetSolution(single Variable) returns a float",
          isinstance(single, float), type(single).__name__)
    check("... equal to that entry of GetSolution(lumped_vars)", single == lumped[2])
    check("GetSolution accepts a plain list of Variables",
          np.array_equal(result.GetSolution(list(p.lumped_vars[:3])), lumped[:3]))
    ## The returned point is a point of the program: finite and inside the program's own
    ## BOUNDING BOXES (on this arm a +-10 box on q -- the joint limits are generic rows, which
    ## an unconverged return may violate). It is NOT required to be the start (the skeleton
    ## returned the start; the solver moves).
    lo = np.full(len(lumped), -np.inf)
    hi = np.full(len(lumped), np.inf)
    idx_of = {int(k): i for i, k in enumerate(p.prog.FindDecisionVariableIndices(p.lumped_vars))}
    for b in p.prog.bounding_box_constraints():
        for row, k in enumerate(p.prog.FindDecisionVariableIndices(b.variables())):
            i = idx_of.get(int(k))
            if i is not None:
                lo[i] = max(lo[i], float(b.evaluator().lower_bound()[row]))
                hi[i] = min(hi[i], float(b.evaluator().upper_bound()[row]))
    check("GetSolution(lumped_vars) is finite and inside the program's variable bounds",
          np.all(np.isfinite(lumped)) and np.all(lumped >= lo - 1e-12) and np.all(lumped <= hi + 1e-12),
          f"lumped {lumped} bounds {lo} {hi}")
    details = result.get_solver_details()
    print(f"        (status {details.status_name!r}, moved |x - x0|_inf = "
          f"{np.max(np.abs(lumped - x0)):.3g}, drake_feasible {details.drake_feasible})")
    check("status is one of the real statuses, not the skeleton's",
          details.status in STATUS_NAMES and details.status != STATUS_NOT_IMPLEMENTED
          and details.status_name == status_name(details.status), str(details.status))
    check("is_success() == the solver's own exact Drake re-check (drake_feasible)",
          result.is_success() == bool(details.drake_feasible),
          f"{result.is_success()} vs {details.drake_feasible}")

    ## The harness side, exactly as run_grid drives it.
    check("last_iterate was recorded", getattr(p, "last_iterate", None) is not None)
    counts = p.eval_counts
    check("eval_counts: callback and map_forward were bumped",
          counts["callback"] > 0 and counts["map_forward"] > 0, str(counts))
    try:
        ## `verify` calls `task_gate(program, q)` (its docstring's `task_gate(q)` is stale
        ## shorthand); a permissive gate keeps this test about the result contract.
        verdict = bm.verify(p, result, lambda program, q: (True, {}), 1e-4, relaxed_tol=1e-3)
        check("verify() accepts the result", verdict is not None
              and verdict.fail_reason != "nan", f"{verdict}")
        check("verify() measured the point (max_violation is a number)",
              verdict.detail.get("max_violation") is not None, str(verdict.detail.keys()))
    except Exception as exc:
        check("verify() accepts the result", False, f"{type(exc).__name__}: {exc}")
    try:
        cost = bm.reported_cost(p, result, 1.0)
        check("reported_cost() returns a finite float", np.isfinite(cost), str(cost))
        check("... and matches EvalBinding summed over the program's costs",
              abs(cost - result.get_optimal_cost()) < 1e-12,
              f"{cost} vs {result.get_optimal_cost()} (no regularizer cost on this arm)")
    except Exception as exc:
        check("reported_cost() returns a finite float", False, f"{type(exc).__name__}: {exc}")

    parsed = bm.parse_log(log, "svgd")
    check("parse_log recovers the step count from the svgd log",
          parsed["iterations"] is not None and parsed["iterations"] >= 1, str(parsed))
    check("parse_log recovers the exit line (== the details' status_name)",
          parsed["exit"] == result.get_solver_details().status_name, str(parsed))
    check("parse_log recovers solver_seconds", parsed["solver_seconds"] is not None, str(parsed))
    check("parse_log leaves IPOPT-only keys None",
          parsed["objective_evals"] is None and parsed["jacobian_evals"] is None, str(parsed))

    diag = bm.solver_diagnostics(result, "svgd")
    check("solver_diagnostics names the solver", diag["solver"] == "svgd", str(diag))
    ## Decoded from the details' status, not from log text: each flag is a bool and agrees
    ## with the status code (a 5 s cell may or may not reach the clock; either way is fine).
    check("solver_diagnostics decodes the budget flags numerically",
          isinstance(diag["timed_out_status"], bool) and isinstance(diag["hit_iteration_cap_status"], bool)
          and diag["timed_out_status"] == (details.status_name == "wall-clock limit")
          and diag["hit_iteration_cap_status"] == (details.status_name == "step cap"),
          str(diag))
    block = diag.get("svgd") or {}
    check("the svgd block carries the particle count and the method",
          block.get("n_particles") == opts.svgd_n and block.get("method") == opts.svgd_method,
          str(block))
    check("the svgd block carries phase_times as a dict",
          isinstance(block.get("phase_times"), dict), str(block))

    emitted = getattr(p, "emitted_solver_options", None)
    check("emitted_solver_options records every svgd_* field under 'svgd'",
          emitted is not None and set(emitted) == {"svgd"}
          and all(k.startswith("svgd_") for k in emitted["svgd"])
          and emitted["svgd"]["svgd_n"] == opts.svgd_n,
          str(emitted))

    ## summarise must carry the svgd columns on every arm: numbers where svgd ran, nan
    ## where it did not.
    base = dict(target=0, guess=0, feasible=False, fail_reason="x", eval_counts={})
    records = {"svgd_arm": [dict(base, svgd=block)], "drake_arm": [dict(base, svgd=None)]}
    arms = [SimpleNamespace(name="svgd_arm"), SimpleNamespace(name="drake_arm")]
    try:
        summary = bm.summarise(records, arms, 1, 1)
        s, d = summary["svgd_arm"], summary["drake_arm"]
        check("summarise: svgd columns are numbers on the svgd arm",
              s["mean_svgd_steps"] == block["iterations"]
              and s["median_n_feasible_particles"] == block["n_feasible"]
              and np.isfinite(s["mean_collision_seconds"]), str(s))
        check("summarise: svgd columns are nan on a Drake-solver arm",
              all(np.isnan(d[k]) for k in ("mean_svgd_steps", "median_n_feasible_particles",
                                           "mean_collision_seconds")), str(d))
    except Exception as exc:
        check("summarise() runs with the svgd key present", False,
              f"{type(exc).__name__}: {exc}")


def test_bad_svgd_set_is_refused_at_options():
    print("\n--- a bad --set svgd_method dies at ProgramOptions, before any cell ---")
    try:
        ProgramOptions(which_solver="svgd", svgd_method="foo")
        check("ProgramOptions(svgd_method='foo') raises", False, "constructed without raising")
    except ValueError as exc:
        check("ProgramOptions(svgd_method='foo') raises ValueError naming the field",
              "svgd_method" in str(exc) and "foo" in str(exc), str(exc))


def test_option_guard():
    """The svgd option checks at ProgramOptions: `svgd_method` has one value; the removed
    methods and kernel spaces are refused; a non-positive temperature, step size or initial
    penalty is refused; the CUDA-graph switch without the compile switch is refused."""
    print("\n--- svgd option checks ---")
    cases = [(dict(svgd_method="al_svgd"), False),
             (dict(svgd_method="tsvgd"), True),
             (dict(svgd_method="admm_svgd"), True),
             (dict(svgd_kernel="x"), True),
             (dict(svgd_kernel="none"), False),
             (dict(svgd_temperature=0.0), True),
             (dict(svgd_lr=-1.0), True),
             (dict(svgd_rho=0.0), True),
             (dict(svgd_dual_lr=0.0), False),
             (dict(svgd_dual_lr=-1.0), True),
             (dict(svgd_constraint_inside_kernel=True), False),
             (dict(svgd_cuda_graph=True), True),
             (dict(svgd_compile=True, svgd_cuda_graph=True), False)]
    for kw, should_raise in cases:
        try:
            ProgramOptions(which_solver="svgd", **kw)
            raised, msg = False, ""
        except ValueError as exc:
            raised, msg = True, str(exc)
        check(f"ProgramOptions({kw}) {'raises' if should_raise else 'is accepted'}",
              raised == should_raise, msg or "no raise")
    removed = ("svgd_collision_workers", "svgd_pool_overlap", "svgd_polish_iters", "svgd_gn_every", "svgd_eta_rel", "svgd_q_step_max",
               "svgd_repulsion_T0", "svgd_anneal_frac", "svgd_gamma_t", "svgd_admm_rho",
               "svgd_tsvgd_switch_infeas", "svgd_lr_decay_t", "svgd_bandwidth",
               "svgd_row_scale_rot", "svgd_resample_every", "svgd_jitter_z", "svgd_rho0",
               "svgd_rho_growth", "svgd_rho_gamma", "svgd_rho_max")
    present = [f for f in removed if hasattr(ProgramOptions(), f)]
    check("no field of a removed piece survives on ProgramOptions", not present, str(present))


def test_svgd_kill_keeps_the_iterate():
    """The kill test owed by the smoke's go/no-go (4): a solve that dies mid-swarm must leave
    `program.last_iterate` at the swarm's latest best particle, and the harness's recovery path
    (`_recover_from_last_iterate`, what `run_grid`'s per-cell `except` calls) must score it into
    the `recovered_*` keys. The kill is an exception raised from the solver's own
    `RecordIterate` hook on its 4th outer check -- the same path an OOM, a CUDA error or a
    SIGTERM-handler raise would take -- so the solve is lost exactly as a real kill loses it."""
    print("\n--- svgd kill test: the iterate survives an abnormal exit ---")
    log = os.path.join(REPO, "results/_test_svgd_kill.log")
    target, sampler = a_reachable_target()
    opts = ProgramOptions(which_solver="svgd", max_wall_time=60.0, file_print_name=log,
                          collision_avoidance=True)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=SCENE)
        p = PandaIKProgramNumerical(diagram, options=opts)
        p.create_prog(target)
    rng = np.random.default_rng(2)
    q_init = rng.uniform(sampler.plant.GetPositionLowerLimits(),
                         sampler.plant.GetPositionUpperLimits())
    p.SetStartFromQ(q_init)

    calls = []
    real_record = p.RecordIterate

    def killing_record(vars):
        real_record(vars)
        calls.append(np.array(vars, dtype=float))
        if len(calls) == 4:
            raise RuntimeError("kill test: the solve dies here")
    p.RecordIterate = killing_record

    record = {"feasible": None}
    raised = None
    with HiddenPrints():
        try:
            p.Solve()
        except Exception as exc:                     # what run_grid's per-cell except sees
            raised = exc
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["feasible"] = False
            record["fail_reason"] = "error"
            bm._recover_from_last_iterate(p, record, lambda program, q: (True, {}), 1e-4, 1e-3)
    check("the kill propagated out of Solve() as an exception",
          isinstance(raised, RuntimeError) and "kill test" in str(raised), repr(raised))
    check("RecordIterate was called on every outer check up to the kill (4 calls)",
          len(calls) == 4, str(len(calls)))
    last = getattr(p, "last_iterate", None)
    check("program.last_iterate is the iterate recorded at the kill",
          last is not None and np.array_equal(last, calls[-1]))
    check("... and it moved from the start (the swarm had run)",
          last is not None and np.max(np.abs(last - calls[0])) > 0 or len(calls) < 2,
          f"|x_kill - x_0|_inf = {np.max(np.abs(last - calls[0])) if last is not None else None}")
    check("recovered_feasible is scored (a bool)", isinstance(record.get("recovered_feasible"), bool),
          str(record.get("recovered_feasible")))
    check("recovered_max_violation is a finite number",
          isinstance(record.get("recovered_max_violation"), float)
          and np.isfinite(record["recovered_max_violation"]),
          str(record.get("recovered_max_violation")))
    check("recovered_q has the robot's joint count",
          record.get("recovered_q") is not None and len(record["recovered_q"]) == 7,
          str(record.get("recovered_q")))
    check("no recovered_error", "recovered_error" not in record, str(record.get("recovered_error")))
    check("the verdict keys are untouched by recovery (the cell still FAILED)",
          record["feasible"] is False and record["fail_reason"] == "error")
    print(f"        (recovered_feasible {record.get('recovered_feasible')}, recovered_max_violation "
          f"{record.get('recovered_max_violation')!r}, recovered_fail_reason "
          f"{record.get('recovered_fail_reason')!r})")


def main():
    test_svgd_result_duck_type()
    test_svgd_kill_keeps_the_iterate()
    test_bad_svgd_set_is_refused_at_options()
    test_option_guard()
    print(f"\n{CHECKS[0]} checks, {len(FAILURES)} failed")
    for name in FAILURES:
        print(f"  FAILED: {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
