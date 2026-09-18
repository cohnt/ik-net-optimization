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
  * **A default NLopt configuration emits Drake 1.56.0's option names and nothing else.**
    The cluster runs the official 1.56.0 tarball (six NLopt options); this workstation's
    source build has eleven and a 2026-09-18 nightly has sixteen, and Drake RAISES on a
    name it does not know. Worse, that raise lands in `run_grid`'s per-cell
    `except Exception` and is recorded as `fail_reason="error"`, so a wrong name does not
    stop a run -- it returns a full COLUMN of instant failures. The post-1.56 fields
    therefore default to unset and are refused at configuration time on a Drake lacking
    them; the checks here pin both halves, and must not be relaxed to whatever is present.
  * **An inner-solver option set without an inner algorithm is refused.** Drake applies
    `local_optimizer_*` only when `local_optimizer_algorithm` is non-empty
    (`nlopt_solver.cc:546-564`), so the combination is accepted and INERT -- a swept column
    that measures the default inner solver at every setting and reports a real-looking
    null. That is the `Timing Level` failure, caught at the door.
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
from src.generic_program import (IKFlowProgram, ProgramOptions,         # noqa: E402
                                 NLOPT_POST_1_56_OPTIONS, NloptOptionSurface)
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

## The ten option names that arrived AFTER 1.56.0, written out by hand for the same reason
## NLOPT_KEYS_1_56 is: this file is the independent pin, and deriving either set from
## dir(NloptSolver) would make it agree with whatever Drake happens to be installed. The two
## sets must stay disjoint -- a field wired to the WRONG accessor would emit a real, valid
## option under a plausible name, and nothing else here would notice.
NLOPT_KEYS_POST_1_56 = {
    "ftol_rel", "ftol_abs", "stopval",
    "local_optimizer_algorithm", "local_optimizer_xtol_rel", "local_optimizer_xtol_abs",
    "local_optimizer_ftol_rel", "local_optimizer_ftol_abs",
    "local_optimizer_max_eval", "local_optimizer_max_time",
}
## A value per post-1.56 field, used only to prove the field LANDS. Note max_eval's 50 and
## stopval's finite value: 0 and 0.0 are MEANINGFUL values for these options ("no cap",
## "never fires"), so the unset sentinel has to be None and no probe may be falsy by accident.
NLOPT_POST_1_56_PROBES = {
    "nlopt_ftol_rel": 1e-8,
    "nlopt_ftol_abs": 1e-10,
    "nlopt_stopval": -1e30,
    "nlopt_local_optimizer_algorithm": "LD_LBFGS",
    "nlopt_local_optimizer_xtol_rel": 1e-8,
    "nlopt_local_optimizer_xtol_abs": 1e-10,
    "nlopt_local_optimizer_ftol_rel": 1e-8,
    "nlopt_local_optimizer_ftol_abs": 1e-10,
    "nlopt_local_optimizer_max_eval": 50,
    "nlopt_local_optimizer_max_time": 5.0,
}


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
    ## IPOPT's REAL convergence test, as opposed to the `acceptable_*` early stop. Never
    ## set before, so IPOPT converged at 1e-4 constraint violation while the SQP column it
    ## is compared against was held to 1e-6.
    "ipopt_tol": (1e-6, "tol"),
    "ipopt_constr_viol_tol": (1e-6, "constr_viol_tol"),
    "ipopt_dual_inf_tol": (1e-6, "dual_inf_tol"),
    "ipopt_compl_inf_tol": (1e-6, "compl_inf_tol"),
    ## Read only under mu_strategy=monotone, so the sweep entry has to pass both. Set alone
    ## it is echoed `used = no`, which is exactly the silent-no-op this test exists to catch.
    "ipopt_mu_init": (1.0, "mu_init"),
    ## The step-rejection family, which stage STEP sweeps. These were plumbed and left
    ## unverified for the life of the branch: `test_no_step_rejection_knob_is_set_by_default`
    ## below proves only that they are UNSET by default, which says nothing about whether
    ## setting one reaches IPOPT. All three are read unconditionally by the filter
    ## line-search acceptor (IpFilterLSAcceptor.cpp:212-223) and the backtracking line search
    ## (IpBacktrackingLineSearch.cpp:218), so unlike `mu_init` none needs a companion option
    ## -- but that is an argument from the source, and this is the measurement.
    ##
    ## Usedness is set on RETRIEVAL, not per value, so proving the knob lands at one value
    ## proves it for all of them. The values here are the ones stage STEP actually fields.
    "ipopt_theta_max_fact": (1.0, "theta_max_fact"),
    "ipopt_watchdog_trigger": (0, "watchdog_shortened_iter_trigger"),
    "ipopt_max_soc": (8, "max_soc"),
}
SNOPT_ECHO_KNOBS = {
    "snopt_hessian_frequency": (50, "Hessian frequency......        50"),
    "snopt_elastic_weight": (100.0, "Elastic weight.........  1.00E+02"),
    "snopt_crash_option": (0, "Crash option...........         0"),
    ## SNOPT capitalises the second word here and nowhere else nearby.
    "snopt_proximal_point_method": (2, "Proximal Point method..         2"),
    ## Step rejection, SNOPT's side. The dot counts differ between the two -- seven after
    ## `limit` for the step limit and eight for the violation limit -- because SNOPT pads the
    ## label to a fixed width, so these strings cannot be written by analogy from each other.
    "snopt_major_step_limit": (0.5, "Major step limit.......  5.00E-01"),
    "snopt_violation_limit": (1.0, "Violation limit........  1.00E+00"),
    ## Fielded by stage SNOPTTUNE, which chooses SNOPT's own configuration, so these have to
    ## be proven to land for the same reason the step-rejection five did.
    "snopt_linesearch_tolerance": (0.99, "Linesearch tolerance...   0.99000"),
    "snopt_major_optimality_tol": (1e-8, "Major optimality tol...  1.00E-08"),
    ## A VALUELESS keyword: passing 0 turns it on exactly as 1 does, so it is emitted only
    ## when True and the echo prints the label with an EMPTY value field. Verification is
    ## therefore the presence of the line at all -- it is absent from a default run, which is
    ## what makes presence sufficient. Note the two spaces after the abbreviating period.
    "snopt_nonderivative_linesearch": (True, "Nonderiv.  linesearch.."),
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


def _clear_post_1_56(options):
    for field_name in NLOPT_POST_1_56_PROBES:
        setattr(options, field_name, None)


def test_defaults_emit_exactly_what_they_always_have():
    """The byte-identity pin: a default NLopt cell must be the archived one.

    `test_option_surface_is_the_cluster_s` only bounds the key SET. This bounds the keys AND
    the values, because the whole point of the post-1.56 fields being opt-in is that adding
    them changed nothing for a run that does not ask for one -- and "nothing" has to include
    `max_time`'s 60.0 and `max_eval`'s 0, either of which would silently redefine what the
    NLopt column measures.
    """
    print("\n--- a default NLopt configuration is unchanged ---")
    target = a_reachable_target(ProgramOptions())
    p = build(ProgramOptions(which_solver="nlopt"), target)
    _, solver_options = p._NloptOptions()
    emitted = dict(solver_options.options.get(NloptSolver.id().name(), {}))
    check("a default NLopt configuration emits exactly three keys",
          set(emitted) == {"algorithm", "max_time", "max_eval"},
          f"emitted {sorted(emitted)}; the three tolerances are None by default and the "
          f"post-1.56 fields must add nothing")
    check("and exactly the values every archived NLopt cell ran with",
          emitted == {"algorithm": "LD_AUGLAG", "max_time": 60.0, "max_eval": 0},
          f"emitted {emitted}")
    check("no post-1.56 key appears in a default configuration",
          not (set(emitted) & NLOPT_KEYS_POST_1_56), f"emitted {sorted(emitted)}")


def test_new_nlopt_fields_are_unset_by_default():
    """Unset, and unset as None -- 0 is a real value for two of them."""
    print("\n--- the post-1.56 NLopt fields are opt-in ---")
    opts = ProgramOptions()
    for field_name in NLOPT_POST_1_56_PROBES:
        check(f"{field_name} is None by default",
              getattr(opts, field_name) is None,
              f"was {getattr(opts, field_name)!r}. A default other than None would be "
              f"emitted, and on the cluster's Drake that fails every cell.")


def test_the_accessor_table_names_the_options_it_claims_to():
    """A field wired to the wrong accessor emits a REAL option under a plausible name.

    NLOPT_POST_1_56_OPTIONS maps each field to a static accessor on NloptSolver rather than
    to a string, so Drake decides the spelling. The cost of that is that swapping
    FRelativeToleranceName for XRelativeToleranceName would be invisible: the run would set a
    valid option, Drake would accept it, and the sweep would report the wrong knob's effect
    under the right knob's name. This reads the mapping back.
    """
    print("\n--- the post-1.56 accessor table ---")
    surface = NloptOptionSurface()
    check("the code exposes exactly the ten post-1.56 fields this file knows about",
          set(NLOPT_POST_1_56_OPTIONS) == set(NLOPT_POST_1_56_PROBES),
          f"code has {sorted(NLOPT_POST_1_56_OPTIONS)}")
    check("the 1.56.0 six and the post-1.56 ten are disjoint",
          not (NLOPT_KEYS_1_56 & NLOPT_KEYS_POST_1_56), "a name cannot be in both eras")
    for field_name, spec in NLOPT_POST_1_56_OPTIONS.items():
        if spec.accessor not in surface:
            continue
        expected = field_name[len("nlopt_"):]
        check(f"{field_name} -> NloptSolver.{spec.accessor}() == {expected!r}",
              surface[spec.accessor] == expected,
              f"got {surface[spec.accessor]!r}; the field name and the option name must "
              f"agree or --set is lying about which knob it moves")
    for name in surface.values():
        check(f"the option {name!r} Drake reports is one this file knows",
              name in (NLOPT_KEYS_1_56 | NLOPT_KEYS_POST_1_56),
              f"this Drake has an NLopt option this file has never heard of -- a surface "
              f"newer than the sixteen; extend NLOPT_ACCESSORS and this set together")


def test_inner_solver_options_are_refused_without_an_algorithm():
    """Drake would accept these and throw them away. That is not an option this repo takes.

    nlopt_solver.cc:546-564 applies every local_optimizer_* setting inside
    `if (!parsed_options.local_optimizer_algorithm.empty())`. With the algorithm unnamed Drake
    never calls set_local_optimizer and the numbers vanish -- a swept column measuring the
    default inner solver at every setting, reporting no difference, which reads exactly like a
    real null result.

    Checked BEFORE availability on purpose, so this refusal is reproducible on the cluster's
    1.56.0 too: there every one of these options is also missing, and reporting only that
    would hide a modelling error behind a packaging one.
    """
    print("\n--- an inner-solver option without an inner algorithm is refused ---")
    for field_name, spec in NLOPT_POST_1_56_OPTIONS.items():
        if not spec.inner:
            continue
        try:
            ProgramOptions(which_solver="nlopt",
                           **{field_name: NLOPT_POST_1_56_PROBES[field_name]})
            check(f"{field_name} alone is refused", False,
                  "accepted -- Drake would read it and silently discard it")
        except ValueError as exc:
            check(f"{field_name} alone is refused, naming the gate",
                  field_name in str(exc) and "nlopt_local_optimizer_algorithm" in str(exc),
                  str(exc))
    ## An empty string is Drake's own spelling of "no inner optimizer", so it must not count
    ## as naming one.
    try:
        ProgramOptions(which_solver="nlopt", nlopt_local_optimizer_algorithm="",
                       nlopt_local_optimizer_max_eval=50)
        check("an EMPTY inner algorithm does not satisfy the gate", False,
              "'' is what Drake's `local_optimizer_algorithm.empty()` tests for")
    except ValueError as exc:
        check("an EMPTY inner algorithm does not satisfy the gate", True, str(exc))
    ## And the mutation path, which __post_init__ never sees: benchmark.py assigns onto a live
    ## options object, so _NloptOptions has to re-check rather than trust it.
    target = a_reachable_target(ProgramOptions())
    p = build(ProgramOptions(which_solver="nlopt"), target)
    p.options.nlopt_local_optimizer_max_eval = 50
    try:
        p._NloptOptions()
        check("_NloptOptions re-checks a mutated options object", False,
              "the inner option was emitted or dropped silently after an attribute "
              "assignment bypassed __post_init__")
    except ValueError as exc:
        check("_NloptOptions re-checks a mutated options object",
              "nlopt_local_optimizer_algorithm" in str(exc), str(exc))
    _clear_post_1_56(p.options)


def test_post_1_56_options_are_checked_against_the_running_drake():
    """Each new field either emits its option or refuses, per what THIS Drake carries.

    Written to pass on all three surfaces without being relaxed on any: the assertion is not
    "stopval works", it is "stopval works iff NloptSolver declares StopValName, and says so by
    name when it does not". On the cluster's 1.56.0 all ten take the refusal branch; on a
    nightly all ten take the emit branch; this workstation's build splits five and five.
    """
    print("\n--- post-1.56 NLopt options are gated on the running Drake ---")
    surface = NloptOptionSurface()
    target = a_reachable_target(ProgramOptions())
    p = build(ProgramOptions(which_solver="nlopt"), target)
    for field_name, probe in NLOPT_POST_1_56_PROBES.items():
        spec = NLOPT_POST_1_56_OPTIONS[field_name]
        _clear_post_1_56(p.options)
        setattr(p.options, field_name, probe)
        ## Inner options need the gate satisfied, or they are refused for the OTHER reason --
        ## which is the previous test's business, not this one's.
        if spec.inner:
            p.options.nlopt_local_optimizer_algorithm = "LD_LBFGS"
        have = (spec.accessor in surface
                and (not spec.inner or "LocalOptimizerAlgorithmName" in surface))
        try:
            _, solver_options = p._NloptOptions()
            emitted = set(solver_options.options.get(NloptSolver.id().name(), {}))
            check(f"{field_name} emits its option on a Drake that has it",
                  have and surface[spec.accessor] in emitted,
                  (f"emitted {sorted(emitted)}; this Drake's NLopt surface is "
                   f"{sorted(surface.values())}") if have else
                  (f"NO RAISE, but NloptSolver has no {spec.accessor} -- this is the key "
                   f"that would reach Drake and fail every cell of a cluster run"))
        except ValueError as exc:
            check(f"{field_name} is refused by name on a Drake that lacks it",
                  not have and field_name in str(exc) and spec.accessor in str(exc),
                  str(exc) if not have else
                  f"refused although NloptSolver.{spec.accessor} exists here: {exc}")
    _clear_post_1_56(p.options)
    ## The same refusal from the EARLY side. __post_init__ is what makes this a configuration
    ## error rather than a per-cell one, and the difference is a whole column of compute.
    missing = [n for n, spec in NLOPT_POST_1_56_OPTIONS.items()
               if spec.accessor not in surface]
    if missing:
        try:
            ProgramOptions(which_solver="nlopt",
                           **{missing[0]: NLOPT_POST_1_56_PROBES[missing[0]]})
            check("constructing ProgramOptions with an unavailable NLopt option raises",
                  False, "it was accepted, so the failure would arrive per-cell instead")
        except ValueError as exc:
            check("constructing ProgramOptions with an unavailable NLopt option raises",
                  missing[0] in str(exc), str(exc))
    else:
        print("  skip  every post-1.56 option exists on this Drake; the refusal path is "
              "exercised on the cluster's 1.56.0 and on this workstation's source build")


def main():
    print("solver plumbing: three method classes -- interior point, SQP, augmented Lagrangian")
    test_option_surface_is_the_cluster_s()
    test_defaults_emit_exactly_what_they_always_have()
    test_new_nlopt_fields_are_unset_by_default()
    test_the_accessor_table_names_the_options_it_claims_to()
    test_inner_solver_options_are_refused_without_an_algorithm()
    test_post_1_56_options_are_checked_against_the_running_drake()
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
