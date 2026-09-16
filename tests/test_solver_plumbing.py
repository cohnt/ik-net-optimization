"""Assert the solver axis reports what it claims to, under all three solvers.

There is no test suite in this repo; run this by hand:

    python tests/test_solver_plumbing.py

**The axis is three method classes, not three vendors.** IPOPT is an interior-point
method, SNOPT is SQP, and NLopt is here as an augmented Lagrangian. That is the whole
reason NLopt is in the comparison, and it is why `LD_SLSQP` would be the wrong default --
it is an SQP method, so it would make the NLopt column a duplicate of SNOPT's rather than
a third data point.

What this file protects, each learned by the thing being possible:

  * `parse_log` used to match only IPOPT's format, so a SNOPT run reported `None` for
    iterations, the evaluation counts, `solver_seconds` and `exit`. Iterations is the
    hardware-independent number every result is told in, so the axis meant nothing. The
    SNOPT parse is checked against a print file this test actually produces.
  * `Solve()` used to be two bare `if`s, so any `which_solver` outside {ipopt, snopt}
    left `solver` unbound and died with `UnboundLocalError` several lines later --
    reachable from `--set which_solver=`, which bypasses argparse's `choices`.
  * **The NLopt option names are pinned to Drake 1.56.0's six.** The cluster runs the
    official 1.56.0 tarball; a workstation source build additionally offers five
    `local_optimizer_*` options, and Drake validates NLopt names strictly and RAISES on
    one it does not know. So code written against a newer local Drake passes here and
    fails on every cell of a cluster run. This is the check that catches that, and it
    must keep failing on the newer API rather than being relaxed to whatever is present.
  * No solver reports an iteration count through Drake -- `SnoptSolverDetails` has `info`
    and `solve_time` but no count, and `NloptSolverDetails` has a single `status` field --
    so the program counts map evaluations itself. If that stops working, the NLopt column
    has no cost measure at all.
"""
import os
import sys
import tempfile
from dataclasses import replace

import numpy as np
from pydrake.solvers import NloptSolver

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.benchmark as bm                                              # noqa: E402
from src.generic_program import IKFlowProgram, ProgramOptions           # noqa: E402
from src.utils import BuildEnv, HiddenPrints                            # noqa: E402
from src.panda_program import PandaIKProgram                            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(REPO, "models/panda/panda_finray_collision_hardened.yaml")
FAILURES = []
CHECKS = [0]

## The exact option surface of Drake 1.56.0's NloptSolver, which is what the cluster runs.
## Read off ~/learned-ik/drake/include/drake/solvers/nlopt_solver.h there. Do NOT extend
## this from a local build's `dir(NloptSolver)` -- that is the failure this pins.
NLOPT_KEYS_1_56 = {"algorithm", "constraint_tol", "xtol_rel", "xtol_abs",
                   "max_eval", "max_time"}


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          {detail}")
        FAILURES.append(name)


def build(options, target):
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=SCENE)
        p = PandaIKProgram(diagram, options=options)
        p.create_prog(target)
    return p


def a_reachable_target(options):
    """A target the way the benchmark makes them: the gripper pose of a real configuration.

    A fabricated pose could be unreachable, and a solver failing for that reason would
    tell us nothing about the plumbing.
    """
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=SCENE)
        sampler = PandaIKProgram(diagram, options=options)
        sampler.create_prog()
    rng = np.random.default_rng(0)
    q = rng.uniform(sampler.plant.GetPositionLowerLimits(),
                    sampler.plant.GetPositionUpperLimits())
    translation, wxyz = sampler.fk(q)
    return np.concatenate([translation, wxyz])


def test_option_surface_is_the_cluster_s():
    """Every NLopt key the code emits must exist in Drake 1.56.0."""
    print("\n--- NLopt option surface (pinned to the cluster's Drake 1.56.0) ---")
    opts = ProgramOptions(which_solver="nlopt", nlopt_constraint_tol=1e-6,
                          nlopt_xtol_rel=1e-6, nlopt_xtol_abs=1e-6, max_iter=500)
    target = a_reachable_target(ProgramOptions())
    p = build(opts, target)
    _, solver_options = p._NloptOptions()
    emitted = set(solver_options.options.get(NloptSolver.id().name(), {}))
    check("every emitted NLopt key exists in Drake 1.56.0",
          emitted <= NLOPT_KEYS_1_56,
          f"emitted {sorted(emitted)}; 1.56.0 has {sorted(NLOPT_KEYS_1_56)}. Keys outside "
          f"that set raise on the cluster while passing on a newer local build.")
    check("no local_optimizer_* key is emitted",
          not any(k.startswith("local_optimizer") for k in emitted),
          f"emitted {sorted(emitted)} -- these do not exist in 1.56.0 and Drake raises on "
          f"an unknown NLopt option name, so this would fail every cell of a cluster run")
    check("the wall-clock cap is passed to NLopt as max_time",
          "max_time" in emitted,
          "without it a cell runs past the cap to the harness's per-item timeout")
    check("max_eval is set explicitly, not left to Drake's default of 1000",
          "max_eval" in emitted,
          "Drake defaults max_eval to 1000; that is a cap, not 'unset', and it binds here")


def test_nlopt_is_an_augmented_lagrangian():
    """The method class, asserted by name so the axis cannot silently become two SQPs."""
    print("\n--- NLopt is the augmented-Lagrangian arm ---")
    algorithm = ProgramOptions().nlopt_algorithm
    check("default nlopt_algorithm is an AUGLAG variant",
          "AUGLAG" in algorithm.upper(),
          f"got {algorithm!r}. LD_SLSQP is an SQP method and would duplicate SNOPT's "
          f"method class, leaving the comparison with two SQP columns and no AL column.")
    check("default is LD_AUGLAG rather than LD_AUGLAG_EQ",
          algorithm == "LD_AUGLAG",
          f"got {algorithm!r}. _EQ absorbs only EQUALITY constraints into the augmented "
          f"Lagrangian and leaves inequalities to the inner solver; this program carries "
          f"both kinds. (Comparing the two is recorded future work, not a default.)")


def test_unknown_solver_raises_clearly():
    print("\n--- an unknown solver fails loudly ---")
    target = a_reachable_target(ProgramOptions())
    p = build(ProgramOptions(which_solver="cplex"), target)
    try:
        p.Solve()
        check("unknown which_solver raises", False, "Solve() returned instead of raising")
    except ValueError as exc:
        check("unknown which_solver raises ValueError naming the valid set",
              "cplex" in str(exc) and "ipopt" in str(exc), str(exc))
    except Exception as exc:
        check("unknown which_solver raises ValueError, not something else", False,
              f"{type(exc).__name__}: {exc}  <- UnboundLocalError is the old bug")


def test_each_solver_solves_and_reports(solver):
    print(f"\n--- {solver}: solve, report, and keep the iterate ---")
    log = os.path.join(REPO, f"results/_test_solver_{solver}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    opts = ProgramOptions(which_solver=solver, max_wall_time=20.0,
                          file_print_name=log, collision_avoidance=True)
    target = a_reachable_target(ProgramOptions())
    p = build(opts, target)
    with HiddenPrints():
        result = p.Solve()

    check(f"{solver}: Solve() returned a result", result is not None)
    ## Every solver calls EvalVisualizationCallbacks inside its objective evaluation, so
    ## the last-iterate recovery that every abnormal exit depends on works unchanged under
    ## all three. If a solver ever stops doing that, this is where it shows up.
    check(f"{solver}: last_iterate was recorded",
          getattr(p, "last_iterate", None) is not None,
          "an abnormal exit could not be verified from the point the solver had")

    counts = getattr(p, "eval_counts", {})
    check(f"{solver}: map_jacobian was counted",
          counts.get("map_jacobian", 0) > 0,
          f"eval_counts={counts}. This is the ONLY cost measure NLopt has, since it "
          f"reports no iteration count and writes no log.")
    check(f"{solver}: the visualization callback fired",
          counts.get("callback", 0) > 0, f"eval_counts={counts}")

    parsed = bm.parse_log(log, solver)
    diag = bm.solver_diagnostics(result, solver)
    if solver == "nlopt":
        check("nlopt: parse_log returns all None without raising",
              all(v is None for v in parsed.values()),
              f"{parsed} -- NLopt writes no log; a number here means we are reading "
              f"another solver's file")
        check("nlopt: a numeric status is recovered from the details object",
              diag["solver_status"] is not None,
              f"{diag} -- status is the ONLY field NloptSolverDetails carries")
    else:
        check(f"{solver}: parse_log recovered an iteration count",
              parsed["iterations"] is not None and parsed["iterations"] > 0,
              f"{parsed} -- this is the defect the solver axis was blocked on")
        check(f"{solver}: parse_log recovered the exit line",
              parsed["exit"] is not None, str(parsed))
    if solver == "snopt":
        check("snopt: the exit line is the SOLVE's, not the SNMEMA memory pass's",
              parsed["exit"] != "finished successfully",
              f"got {parsed['exit']!r}; the print file opens with "
              f"'SNMEMA EXIT 100 -- finished successfully', so matching a bare EXIT "
              f"reports the memory estimation and calls a failed solve a success")
        check("snopt: minor iterations are recorded separately from majors",
              parsed["minor_iterations"] is not None
              and parsed["minor_iterations"] >= parsed["iterations"],
              f"{parsed} -- `iterations` must be MAJORS, to match what IPOPT counts")
        check("snopt: a numeric INFO is recovered from the details object",
              diag["solver_status"] is not None, str(diag))
        check("snopt: solve_time comes from the details object",
              diag["solver_detail_seconds"] is not None, str(diag))
    check(f"{solver}: the record carries which solver produced it",
          diag["solver"] == solver, str(diag))
    check(f"{solver}: Drake's solver-neutral solution_result is recorded",
          diag["solution_result"] is not None, str(diag))


## Every knob the settings sweep moves, with the string the SOLVER ITSELF prints when the
## value has landed. The `expect` column is not decoration: `Timing Level` was set for the
## life of this repo and silently did nothing, and while adding these, `Hessian updates`
## turned out to be inert in the full-memory mode this problem size selects and
## `linear_solver=mumps` turned out not to exist in Drake's IPOPT at all. An option that
## SetOption accepts is NOT an option that took effect, and only the echo tells them apart.
IPOPT_ECHO_KNOBS = {
    "ipopt_limited_memory_max_history": (25, "limited_memory_max_history"),
    "ipopt_limited_memory_update_type": ("sr1", "limited_memory_update_type"),
    "ipopt_alpha_for_y": ("bound-mult", "alpha_for_y"),
    "ipopt_recalc_y": ("yes", "recalc_y"),
    "ipopt_bound_relax_factor": (0.0, "bound_relax_factor"),
    ## Read only under mu_strategy=monotone, so the sweep entry has to pass both. Set alone
    ## it is echoed `used = no`, which is exactly the silent-no-op this test exists to catch.
    "ipopt_mu_init": (1.0, "mu_init"),
}
SNOPT_ECHO_KNOBS = {
    "snopt_hessian_frequency": (50, "Hessian frequency......        50"),
    "snopt_elastic_weight": (100.0, "Elastic weight.........  1.00E+02"),
    "snopt_crash_option": (0, "Crash option...........         0"),
    ## SNOPT capitalises the second word here and nowhere else nearby.
    "snopt_proximal_point_method": (2, "Proximal Point method..         2"),
}


def _solve_once_capturing_log(opts, target, path):
    """One real solve, kept to a couple of iterations -- the echo is written at startup."""
    if os.path.exists(path):
        os.remove(path)
    opts = replace(opts, file_print_name=path, max_iter=3, max_wall_time=20.0)
    p = build(opts, target)
    with HiddenPrints():
        p.Solve()
    return open(path).read() if os.path.exists(path) else ""


def test_new_knobs_reach_the_solver():
    """Assert from the solver's own parameter echo, never from SetOption not raising."""
    print("\n--- every swept knob actually lands (read back from the solver's echo) ---")
    target = a_reachable_target(ProgramOptions())
    log_dir = tempfile.mkdtemp(prefix="solver_echo_")

    ipopt_sets = {name: value for name, (value, _) in IPOPT_ECHO_KNOBS.items()}
    ipopt_sets["ipopt_mu_strategy"] = "monotone"
    text = _solve_once_capturing_log(
        ProgramOptions(which_solver="ipopt", **ipopt_sets), target,
        os.path.join(log_dir, "ipopt.txt"))
    for field_name, (value, option) in IPOPT_ECHO_KNOBS.items():
        ## IPOPT's user-options block has a `used` column, and `no` means it was accepted
        ## and then ignored -- which is a failure for our purposes, not a pass.
        line = next((ln.strip() for ln in text.splitlines()
                     if ln.strip().startswith(option + " =")), None)
        check(f"ipopt: {field_name} reaches the solver and is used",
              line is not None and line.endswith("yes"),
              f"echo line was {line!r}")

    snopt_sets = {name: value for name, (value, _) in SNOPT_ECHO_KNOBS.items()}
    text = _solve_once_capturing_log(
        ProgramOptions(which_solver="snopt", **snopt_sets), target,
        os.path.join(log_dir, "snopt.txt"))
    for field_name, (value, expect) in SNOPT_ECHO_KNOBS.items():
        check(f"snopt: {field_name} reaches the solver",
              expect in text, f"expected {expect!r} in the parameter echo")

    ## The valueless keyword gets its own solve, because its whole point is that it cannot
    ## be turned off by passing a value -- so it must not ride along with the others.
    off = _solve_once_capturing_log(ProgramOptions(which_solver="snopt"), target,
                                    os.path.join(log_dir, "ls_off.txt"))
    on = _solve_once_capturing_log(
        ProgramOptions(which_solver="snopt", snopt_nonderivative_linesearch=True), target,
        os.path.join(log_dir, "ls_on.txt"))
    check("snopt: the line search is derivative-based by default",
          "Derivative linesearch" in off, "default echo did not name the line search")
    check("snopt: snopt_nonderivative_linesearch switches the line search",
          "Nonderiv." in on and "Derivative linesearch" not in on,
          "the echo still shows a derivative line search")


def test_no_step_rejection_knob_is_set_by_default():
    """The step-rejection family is a separate question and must stay at solver defaults."""
    print("\n--- step rejection stays out of this branch ---")
    opts = ProgramOptions()
    for field_name in ("ipopt_theta_max_fact", "ipopt_watchdog_trigger", "ipopt_max_soc",
                       "snopt_violation_limit", "snopt_major_step_limit"):
        check(f"{field_name} is unset by default",
              getattr(opts, field_name) is None, f"was {getattr(opts, field_name)!r}")


def main():
    print("solver plumbing: three method classes -- interior point, SQP, augmented Lagrangian")
    test_option_surface_is_the_cluster_s()
    test_nlopt_is_an_augmented_lagrangian()
    test_unknown_solver_raises_clearly()
    test_no_step_rejection_knob_is_set_by_default()
    test_new_knobs_reach_the_solver()
    for solver in ("ipopt", "snopt", "nlopt"):
        test_each_solver_solves_and_reports(solver)

    print(f"\n{CHECKS[0]} checks, {len(FAILURES)} failed")
    for name in FAILURES:
        print(f"  FAILED: {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
