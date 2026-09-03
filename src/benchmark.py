"""Shared benchmark driver for the three IK formulations.

Everything in here is formulation-agnostic: an experiment supplies a scene, a way to
build a program for a target, and a task-space gate, and this module runs the paired
grid, scores it, and writes the summary.

Three things distinguish it from the older per-script harnesses:

  * **Paired grid.** `num_targets x num_guesses` cells, one solve per cell, no
    retry-on-failure. Every formulation sees the identical `(target_i, guess_j)`, so
    success can be compared with a paired test rather than two independent proportions.

  * **A shared starting configuration.** A guess is a collision-free arm configuration
    `q_init`. Each formulation starts from its own exact representation of that same
    configuration -- the joint-space arm from `q_init` itself, the analytic arm from
    `FK(q_init)` with `psi`/`GC` recovered by inversion, and the learned arm from
    `c = FK(q_init)` with `z` recovered by running the normalizing flow *forwards*
    (`rev=False`), which is exact. Where a formulation's variable bounds do not contain
    that point the start is clipped and the clip distance is recorded, because a start
    outside the box is not the same start.

  * **Success verified independently of the solver.** Every binding of the program is
    re-evaluated at the returned point and the task is re-measured from `q`. The solver's
    own status is recorded alongside but is not the criterion: the learned formulation's
    failures were all wall-clock timeouts, and a timeout that landed on a valid grasp is
    a success by any definition that matters.
"""
import faulthandler
import glob
import json
import math
import os
import re
import shutil
import tarfile
import time
from dataclasses import dataclass, field, replace

import numpy as np


## ----------------------------- solver log parsing ----------------------------- ##

_IPOPT_PATTERNS = {
    "iterations": r"Number of Iterations\.*:\s*(\d+)",
    "objective_evals": r"Number of objective function evaluations\s*=\s*(\d+)",
    "objective_grad_evals": r"Number of objective gradient evaluations\s*=\s*(\d+)",
    "constraint_evals": r"Number of inequality constraint evaluations\s*=\s*(\d+)",
    "jacobian_evals": r"Number of inequality constraint Jacobian evaluations\s*=\s*(\d+)",
    "solver_seconds": r"Total seconds in IPOPT\s*=\s*([\d.]+)",
}


## Cost bindings that regularise the *learned* arm's own decision variables rather than
## expressing the task. They are part of the solve -- they change where the solver goes,
## deliberately -- but they must not be part of the number a cost *table* reports, because
## the baselines have no such variables and so can never carry the corresponding term. A
## column that included them would compare the learned arm's objective-plus-penalties
## against the baselines' bare objective and call the difference a result.
##
## `LatentCost`'s docstring has always claimed this exclusion; nothing implemented it, and
## `result.get_optimal_cost()` returned the total. That was immaterial while both weights
## were zero and material as soon as they were swept: at `latent_cost_weight = 0.1` with
## `||z|| ~ 1.6` the term is ~0.26 against a reported cost of ~3, i.e. 8% of the column.
## At the approved `correction_cost_weight = 10` the correction term is negligible by
## comparison (median 2e-08, because the penalty drives `|q_c|` to ~2e-05), so this
## changes no published learned-arm cost materially -- but it makes the column exactly
## what it says it is instead of nearly so.
_REGULARIZER_COSTS = ("LatentRegularizerCost", "CorrectionCost", "JointLimitPenaltyCost")


def reported_cost(program, result, weight):
    """The objective every formulation shares, at the returned point.

    Sums the program's cost bindings except the learned-only regularizers, so the number
    is comparable across arms. Falls back to `get_optimal_cost()` if the bindings cannot
    be walked, which keeps a cell scoring rather than erroring.
    """
    try:
        total = 0.0
        for binding in program.prog.GetAllCosts():
            if binding.evaluator().get_description() in _REGULARIZER_COSTS:
                continue
            total += float(np.asarray(result.EvalBinding(binding)).sum())
        return total / weight
    except Exception:
        return float(result.get_optimal_cost()) / weight


def parse_log(path):
    """Iteration and evaluation counts from a solver log.

    These are the hardware-independent cost measure. For the learned formulation the
    constraint-Jacobian count is what the flow actually pays for -- one `jacrev` through
    the network each -- so it is the number to quote when comparing formulations rather
    than this laptop's GPU.
    """
    out = {k: None for k in _IPOPT_PATTERNS}
    out["exit"] = None
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return out
    for key, pattern in _IPOPT_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            out[key] = float(m.group(1)) if key == "solver_seconds" else int(m.group(1))
    m = re.search(r"EXIT: (.*)", text)
    if m:
        out["exit"] = m.group(1).strip()
    return out


def is_timeout(exit_string):
    if not exit_string:
        return False
    lowered = exit_string.lower()
    return "wallclock" in lowered or "time" in lowered and "exceeded" in lowered


def is_iteration_cap(exit_string):
    """Did the solve stop because it ran out of iterations rather than seconds?

    `is_timeout` deliberately does not match IPOPT's "Maximum Number of Iterations
    Exceeded" (no "time" in it), so without this a run under `--set max_iter=N` reports
    `timeouts: 0` and looks as if nothing hit a cap at all.
    """
    if not exit_string:
        return False
    lowered = exit_string.lower()
    return ("iteration" in lowered and "exceeded" in lowered) or "iteration limit" in lowered


## --------------------------------- sharding ---------------------------------- ##

def parse_shard(spec):
    """`"K/N"` -> `(K, N)`, validated."""
    if spec is None:
        return None
    try:
        index, _, count = spec.partition("/")
        index, count = int(index), int(count)
    except ValueError:
        raise SystemExit(f"--shard expects K/N, got {spec!r}")
    if count < 1 or not 0 <= index < count:
        raise SystemExit(f"--shard: need 0 <= K < N and N >= 1, got {spec!r}")
    return index, count


def shard_cells(index, count, n_targets, n_guesses):
    """The `(target, guess)` cells of shard `index` of `count`, split **target-major**.

    Whole targets go to a shard; a target's guesses are never split across shards. Two
    things depend on that. The bootstrap CI resamples whole targets (guesses within a
    target are correlated) and `solved_within_k` counts restarts within a target, so a
    shard holding half of a target's guesses would make both meaningless on the merged
    run. And a shard that dies drops a legible set of whole targets rather than silently
    truncating every one of them.

    The grid itself is drawn identically in every shard -- the seeded draws and
    `grid_hash` happen before any filtering -- so a shard is bit-identical to those cells
    of the unsharded run.
    """
    return [(ti, gi) for ti in range(n_targets) if ti % count == index
            for gi in range(n_guesses)]


def provenance():
    """Where and with what this run was measured.

    Recorded in every summary because performance numbers are not comparable across
    machines: without `host` nothing stops a cluster run being paired cell-for-cell
    against a laptop one, and without `device` there is no record of whether a table was
    measured on a GPU or on CPU (which also differ at the ulp level, so a single table
    must be one device).
    """
    import platform
    out = {"host": platform.node(), "platform": platform.platform()}
    try:
        import torch
        out["torch_version"] = torch.__version__
    except Exception:
        pass
    try:
        from ikflow.config import DEVICE
        out["device"] = str(DEVICE)
    except Exception:
        pass
    try:
        # A source build reports "unknown"/0.0.0 here while a release tarball reports its
        # tag, which is exactly the distinction worth recording: the laptop runs a local
        # build and the cluster runs the official noble tarball.
        from pydrake.common import GetDrakePath
        import pydrake
        out["drake_version"] = getattr(pydrake, "__version__", None) or "unknown"
        out["drake_path"] = GetDrakePath()
    except Exception:
        pass
    return out


## ------------------------------- verification -------------------------------- ##

def binding_worst(prog, x):
    """Worst *signed* violation of every binding of `prog` at `x`, keyed by description.

    Signed and unconditional: a negative value is slack (the binding is satisfied with
    room to spare), a positive value is the amount by which it is missed. Nothing is
    filtered by a tolerance here, which is the point -- the tolerance at which a solve
    counts as feasible is a contested choice (`ik_constraint_tol = 1e-4` for the
    program's own rows against `task_tol = 1e-3` for the task gate, and the two are not
    obviously commensurate once a solver applies its own scaling), so the *continuous*
    quantity is what gets recorded and any threshold is applied to it afterwards.
    Recording only "violations above 1e-4" would make that a re-run rather than a
    re-analysis.
    """
    worst = {}
    for binding in prog.GetAllConstraints():
        evaluator = binding.evaluator()
        value = np.asarray(prog.EvalBinding(binding, x), dtype=float).flatten()
        lb = np.asarray(evaluator.lower_bound(), dtype=float).flatten()
        ub = np.asarray(evaluator.upper_bound(), dtype=float).flatten()
        below = np.where(np.isfinite(lb), lb - value, -np.inf)
        above = np.where(np.isfinite(ub), value - ub, -np.inf)
        w = float(np.max(np.maximum(below, above)))
        name = evaluator.get_description() or type(evaluator).__name__
        worst[name] = max(w, worst.get(name, -np.inf))
    return worst


def binding_violations(prog, x, tol):
    """Bindings of `prog` violated by more than `tol` at `x`, keyed by description.

    Mirrors ../codebase's `CheckConstraints`: the point is not only whether the returned
    point is feasible but *which* constraint it misses, since "timed out" and "converged
    to something infeasible" call for different fixes.
    """
    return {k: v for k, v in binding_worst(prog, x).items() if v > tol}


@dataclass
class Verdict:
    feasible: bool
    fail_reason: str = ""          # "" | nan | constraint | task_error | collision
    ## `detail` stays third: existing call sites construct a Verdict positionally as
    ## Verdict(False, "nan", detail), so any new field has to go AFTER it.
    detail: dict = field(default_factory=dict)
    # The same point scored again with the program's constraint tolerance relaxed to the
    # task gate's. Both are recorded because which one the paper's success criterion
    # should be is an open question: a cell can satisfy the task (axis error < 1 mm,
    # collision-free) while missing the program's own rows by 2e-4 to 6e-4, and it is not
    # obvious that those two tolerances are commensurate once a solver scales the problem.
    feasible_relaxed: bool = False
    fail_reason_relaxed: str = ""


def verify(program, result, task_gate, tol, x_lumped=None, relaxed_tol=None):
    """Score a solve from the returned point, independently of the solver's own status.

    `task_gate(q)` returns `(ok, detail_dict)` and is where the experiment states what
    "the arm is actually where it was asked to be" means -- distance off the mug axis for
    a grasp, per-axis position and rpy residual for a pose target.

    `x_lumped`, when given instead of `result`, is a raw value of
    `program.lumped_vars` -- the last iterate of a solve that ended abnormally. The point
    the solver actually had is scored the same way a returned solution is, so an aborted
    solve that was sitting on a feasible point is still visible as one.
    """
    if result is not None:
        x = result.get_x_val()
        x_lumped = result.GetSolution(program.lumped_vars)
    else:
        x_lumped = np.asarray(x_lumped, dtype=float)
        x = np.empty(program.prog.num_vars())
        x[program.prog.FindDecisionVariableIndices(program.lumped_vars)] = x_lumped
    try:
        q = program.VarsToQ(x_lumped)
        q = np.asarray([float(v) for v in q])
    except Exception as exc:                      # a diverged solve can produce garbage
        return Verdict(False, "nan", {"exception": f"{type(exc).__name__}: {exc}"})
    if not np.all(np.isfinite(q)):
        return Verdict(False, "nan", {})

    detail = {"q": [float(v) for v in q]}

    # Every binding's worst signed violation, unconditionally: this is the continuous
    # quantity from which any feasibility threshold can be recomputed later, so a change
    # of mind about the success criterion is a re-analysis and not a re-run of the grid.
    worst = binding_worst(program.prog, x)
    detail["violations_all"] = {k: float(v) for k, v in worst.items()}
    detail["max_violation"] = _finite(max(worst.values())) if worst else None

    violations = {k: v for k, v in worst.items() if v > tol}
    if violations:
        detail["violations"] = {k: float(v) for k, v in violations.items()}
    relaxed_tol = tol if relaxed_tol is None else float(relaxed_tol)
    relaxed_violations = {k: v for k, v in worst.items() if v > relaxed_tol}

    collision = float(np.asarray(program.collision_free_constraint_eval.Eval(q)).flatten()[0])
    detail["collision_value"] = collision

    # `collision_value` is the RAW value of Drake's MinimumDistanceLowerBoundConstraint: a
    # smooth penalty aggregated over every geometry pair inside the influence distance, and
    # a pure number, not a length. It is the right thing to gate on (it is what the binding
    # sees) but it cannot be quoted as clearance, and "1.26 against a limit of 1.0" says
    # nothing about how deep the penetration is. So also record the actual minimum signed
    # distance in metres, negative when geometry overlaps, with the pair that attains it.
    try:
        scene_graph = program.diagram.GetSubsystemByName("scene_graph")
        sg_context = scene_graph.GetMyContextFromRoot(program.diagram_context)
        program.plant.SetPositions(program.plant_context, q)
        pairs = scene_graph.get_query_output_port().Eval(
            sg_context).ComputeSignedDistancePairwiseClosestPoints()
        if pairs:
            worst = min(pairs, key=lambda p: p.distance)
            inspector = scene_graph.model_inspector()
            detail["min_distance"] = float(worst.distance)
            detail["min_distance_pair"] = [
                inspector.GetName(inspector.GetFrameId(worst.id_A)),
                inspector.GetName(inspector.GetFrameId(worst.id_B))]
    except Exception as exc:                      # never let instrumentation kill a cell
        detail["min_distance_error"] = f"{type(exc).__name__}: {exc}"

    # How much of its own budget the learned formulation used: |q_c| against
    # correction_bound says whether the correction box is binding, and ||z|| says whether
    # the latent left the flow's typical set (sqrt(latent_dim), so ~2.65 and ~2.83).
    if hasattr(program, "correction"):
        detail["correction_inf"] = float(np.max(np.abs(
            x[program.prog.FindDecisionVariableIndices(program.correction)])))
    if hasattr(program, "z"):
        detail["z_norm"] = float(np.linalg.norm(
            x[program.prog.FindDecisionVariableIndices(program.z)]))

    ok, task_detail = task_gate(program, q)
    detail.update(task_detail)

    # An interior-point method parks *on* an active constraint, so a converged solve
    # routinely reports a collision value of 1 + 1e-7. The gate has to carry the same
    # slack the program's own binding does: the collision row is scaled by
    # `collision_row_scale`, so `tol` in constraint units is `tol / scale` in Drake's
    # smoothed-distance units (10 * tol at the default scale of 0.1).
    scale = getattr(program.options, "collision_row_scale", 0.1)

    def _score(viol, t):
        """The same three gates at a given constraint tolerance, in the same order."""
        if viol:
            return False, "constraint"
        if collision > 1.0 + t / scale:
            return False, "collision"
        if not ok:
            return False, "task_error"
        return True, ""

    feasible, reason = _score(violations, tol)
    feasible_relaxed, reason_relaxed = _score(relaxed_violations, relaxed_tol)
    return Verdict(feasible, reason, detail, feasible_relaxed, reason_relaxed)


def _finite(value):
    """None rather than inf/nan, so a diverged cell does not poison an aggregate."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def start_diagnostics(program, q_init):
    """How far this arm's *actual* starting configuration is from the shared q_init.

    The protocol rests on every formulation beginning at the same configuration, and only
    the joint-space arm can hold that exactly: each of the others starts at its own
    projection of `q_init` onto its variables and their bounds. The analytic arm cannot
    represent a configuration its chart's branches miss, and the task-parameterised learned
    arm cannot represent one that is not a grasp of this mug at all -- its `c` is projected
    onto the mug axis, which moves the configuration by radians. That is a property of the
    formulations, not a defect, but it has to be *reported* rather than assumed away, so
    every cell carries the number.
    """
    out = {"clip_distance": _finite(getattr(program, "clip_distance", None))}
    n = getattr(program, "num_arm_dof", 7)
    try:
        x0 = program.prog.GetInitialGuess(program.lumped_vars)
        q0 = np.asarray([float(v) for v in program.VarsToQ(x0)], dtype=float)
        out["start_q_error"] = _finite(np.max(np.abs(q0[:n] - np.asarray(q_init, dtype=float)[:n])))
    except Exception:
        out["start_q_error"] = None
    if hasattr(program, "z"):
        try:
            out["start_z_norm"] = _finite(
                np.linalg.norm(program.prog.GetInitialGuess(program.z)))
        except Exception:
            pass
    return out


## -------------------------------- statistics --------------------------------- ##

def mcnemar_exact(a_success, b_success):
    """Two-sided exact McNemar on paired boolean outcomes.

    Paired, because both formulations were run on the identical (target, guess) cells;
    comparing two independent proportions would throw that pairing away and be far less
    sensitive.
    """
    a_only = sum(1 for a, b in zip(a_success, b_success) if a and not b)
    b_only = sum(1 for a, b in zip(a_success, b_success) if b and not a)
    n = a_only + b_only
    if n == 0:
        return dict(a_only=0, b_only=0, p=1.0)
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return dict(a_only=a_only, b_only=b_only, p=float(min(1.0, 2.0 * tail)))


def bootstrap_success_ci(per_target_success, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI on the success rate, resampling whole *targets*.

    Guesses within one target are correlated -- a target that admits few grasps is hard
    from every start -- so resampling cells would understate the interval.
    """
    rng = np.random.default_rng(seed)
    targets = [np.asarray(s, dtype=float) for s in per_target_success if len(s)]
    if not targets:
        return (float("nan"), float("nan"))
    n = len(targets)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = np.mean(np.concatenate([targets[i] for i in idx]))
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def solved_within_k(per_target_success):
    """Fraction of targets solved by at least one of the first k guesses.

    A secondary lens only. It never picks a winner -- the primary comparison is
    single-start, because "succeeds eventually given enough restarts" is a statement
    about the restart budget, not about the formulation.
    """
    if not per_target_success:
        return {}
    k_max = max(len(s) for s in per_target_success)
    out = {}
    for k in range(1, k_max + 1):
        out[k] = float(np.mean([any(s[:k]) for s in per_target_success if len(s) >= k]))
    return out


## ------------------------------- the grid runner ------------------------------- ##

@dataclass
class Arm:
    """One formulation, as the benchmark sees it."""
    name: str
    make_program: object      # (target, q_init, (ti, gi)) -> program, already `create_prog`ed
    weight: float = 1.0       # joint-centering weight, to normalise the reported cost


def run_grid(arms, targets, guesses, task_gate, log_dir, out_path, tol,
             progress=None, metadata=None, cell_timeout=None, cells=None,
             unrepresentable_tol=None, relaxed_tol=None):
    """Run every (arm, target, guess) cell once and write a checkpointed summary.

    Checkpointing after each target matters: these runs are hours long and the learned
    arm's cells can each take the full wall-clock cap.

    `cells`, when given, is a collection of (target, guess) index pairs and restricts the
    run to exactly those cells. It exists to make a single cell of a past run addressable
    (the grid is seeded and hashed, so cell (ti, gi) of an archived summary is
    reproducible bit for bit) -- a debugging aid, not a sampling mechanism: the summary of
    a filtered run is partial and says so in its metadata.
    """
    os.makedirs(log_dir, exist_ok=True)
    # `guesses` is per-target -- guesses[ti][gi] -- by decision: shared guesses made the
    # grid's effective sample size for start-dependent effects equal to the number of
    # guesses (branch membership of ONE shared guess decided entire columns), and guesses
    # being comparable across targets is not a property any analysis here uses.
    records = {arm.name: [] for arm in arms}
    # A dedicated file rather than stderr: program construction runs under HiddenPrints,
    # which redirects fd 2 to /dev/null, and a dump that lands in that window is simply
    # lost -- which is what happened to the one stall this was meant to catch.
    # Per-cell solver logs go to node-local scratch and are rolled into one archive
    # per run at the end; see `_log_scratch_dir`.
    scratch_dir = _log_scratch_dir(log_dir)
    stalls = open(os.path.join(scratch_dir, "stalls.txt"), "a") if cell_timeout else None
    n_targets, n_guesses = len(targets), len(guesses[0])
    assert len(guesses) == n_targets and all(len(row) == n_guesses for row in guesses)

    if cells is not None:
        cells = {(int(t), int(g)) for t, g in cells}
        if metadata is not None:
            metadata = dict(metadata, cells=sorted(cells))

    for ti in range(n_targets):
        for gi in range(n_guesses):
            if cells is not None and (ti, gi) not in cells:
                continue
            for arm in arms:
                log_path = os.path.join(scratch_dir, f"{arm.name}_{ti}_{gi}.txt")
                if os.path.exists(log_path):
                    os.remove(log_path)
                record = dict(target=ti, guess=gi)
                t0 = time.time()
                # If a cell wedges somewhere the solver's own caps cannot reach -- program
                # construction, verification, a solve that overruns even the hard cap --
                # dump every thread's stack to the log rather than leaving a silent stall.
                if cell_timeout:
                    faulthandler.dump_traceback_later(cell_timeout, repeat=True,
                                                      file=stalls)
                program = None
                try:
                    program = arm.make_program(targets[ti], guesses[ti][gi], (ti, gi))
                    record["setup_time"] = time.time() - t0
                    record.update(start_diagnostics(program, guesses[ti][gi]))
                    record["correction_bound"] = getattr(program.options, "correction_bound", None)
                    # The paired protocol means starting AT q_init. A formulation whose
                    # variables cannot represent q_init (an analytic chart that does not
                    # cover it) has no paired start to be given, and quietly starting it
                    # from a projection instead is a different experiment -- so the cell
                    # is an immediate failure, by decision, with no solve attempted.
                    # start_q_error is the round-trip residual of expressing q_init in the
                    # arm's own variables; representable arms sit at <=1e-6, the analytic
                    # chart's uncharted region at >=0.4 rad, so the threshold is not
                    # delicate.
                    if (unrepresentable_tol is not None
                            and record.get("start_q_error") is not None
                            and record["start_q_error"] > unrepresentable_tol):
                        record["feasible"] = False
                        record["fail_reason"] = "unrepresentable_start"
                        record["solver_success"] = False
                        record["wall_time"] = 0.0
                        records[arm.name].append(record)
                        if progress is not None:
                            progress(arm.name, ti, gi, record)
                        if cell_timeout:
                            faulthandler.cancel_dump_traceback_later()
                        continue
                    program.options.file_print_name = log_path
                    start = time.time()
                    result = program.Solve()
                    record["wall_time"] = time.time() - start
                    record["solver_success"] = bool(result.is_success())
                    record.update(parse_log(log_path))
                    record["timed_out"] = is_timeout(record.get("exit"))
                    record["hit_iteration_cap"] = is_iteration_cap(record.get("exit"))
                    verdict = verify(program, result, task_gate, tol,
                                     relaxed_tol=relaxed_tol)
                    record["feasible"] = verdict.feasible
                    record["fail_reason"] = verdict.fail_reason
                    record["feasible_relaxed"] = verdict.feasible_relaxed
                    record["fail_reason_relaxed"] = verdict.fail_reason_relaxed
                    detail = dict(verdict.detail)
                    record["q"] = detail.pop("q", None)
                    ## Stage F: under `lift_q` the returned configuration and the flow's
                    ## own output are different quantities, and their difference is the
                    ## equality residual -- the thing the intervention is measured by. It
                    ## cannot be recomputed later without re-solving, so persist it.
                    if getattr(program, "_LiftingQ", None) is not None and program._LiftingQ():
                        try:
                            x = result.GetSolution(program.lumped_vars)
                            record["q_lift"] = [float(v) for v in x[-program.num_arm_dof:]]
                            record["q_flow"] = [float(v) for v in
                                                np.asarray(program.VarsToQ(x), dtype=float)]
                        except Exception:
                            record["q_lift"] = record["q_flow"] = None
                    ## Promoted out of `detail` because it is the quantity the success
                    ## criterion is argued over, and it should be one key away in every
                    ## record rather than nested.
                    record["max_violation"] = detail.pop("max_violation", None)
                    for key in ("correction_inf", "z_norm"):
                        record[key] = _finite(detail.pop(key, None))
                    record["detail"] = detail
                    if verdict.feasible:
                        record["cost"] = reported_cost(program, result, arm.weight)
                except Exception as exc:            # never let one cell kill a sweep
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["feasible"] = False
                    record["fail_reason"] = "error"
                    record["wall_time"] = time.time() - t0
                    # The solve is lost, but the point it had need not be: score the last
                    # iterate the solver reached exactly as a returned solution would be.
                    # A cell that died on a feasible point is then still visible as one.
                    last = getattr(program, "last_iterate", None) if program is not None else None
                    if last is not None:
                        try:
                            verdict = verify(program, None, task_gate, tol,
                                             x_lumped=last, relaxed_tol=relaxed_tol)
                            record["recovered_feasible"] = verdict.feasible
                            record["recovered_fail_reason"] = verdict.fail_reason
                            record["recovered_feasible_relaxed"] = verdict.feasible_relaxed
                            detail = dict(verdict.detail)
                            record["recovered_q"] = detail.pop("q", None)
                            record["recovered_detail"] = detail
                        except Exception as exc2:
                            record["recovered_error"] = f"{type(exc2).__name__}: {exc2}"
                finally:
                    if cell_timeout:
                        faulthandler.cancel_dump_traceback_later()
                records[arm.name].append(record)
                _abort_on_dead_arm(arm.name, records[arm.name])
                if progress is not None:
                    progress(arm.name, ti, gi, record)

        _write_summary(records, arms, n_targets, n_guesses, out_path, metadata,
                       partial=(ti < n_targets - 1))

    if stalls is not None:
        stalls.close()
    _roll_up_logs(scratch_dir, log_dir)
    return records


## How many identical, instantaneous errors at the head of an arm's records are taken as
## proof that the arm is misconfigured rather than merely failing.
_DEAD_ARM_STREAK = 3


def _abort_on_dead_arm(name, recs):
    """Fail loudly when an arm cannot be constructed at all, instead of scoring it zero.

    This has now cost two whole columns of a cluster campaign. `--set
    correction_cost_weight=10` raised `AttributeError: no attribute 'correction'` inside
    every baseline program's construction, and `--set
    ipopt_nlp_scaling_method=equilibration-based` raised `RuntimeError: Error setting
    IPOPT string option` (Drake's IPOPT is built without the HSL MC19 routine it needs).
    Both scored 0 of every cell at about 10 ms each and were only caught during analysis,
    after the compute had been spent.

    The signature is unmistakable and worth trapping: a *configuration* error is
    deterministic, so it is byte-identical on every cell and returns instantly, whereas a
    genuine numerical failure varies between cells and costs real time. Requiring the
    streak to start at the arm's very first record keeps a sporadic mid-run exception from
    tripping it.
    """
    if len(recs) != _DEAD_ARM_STREAK:
        return
    errs = [r.get("error") for r in recs]
    if not all(errs) or len(set(errs)) != 1:
        return
    if any((r.get("wall_time") or 0.0) > 1.0 for r in recs):
        return
    raise SystemExit(
        f"benchmark: arm {name!r} failed identically on its first {_DEAD_ARM_STREAK} "
        f"cells in under a second each -- this is a misconfiguration, not a hard problem, "
        f"and the run would score the whole column zero. Fix it and re-run.\n"
        f"    {errs[0]}")


def summarise(records, arms, n_targets, n_guesses):
    summary = {}
    for arm in arms:
        recs = records[arm.name]
        ok = [r for r in recs if r.get("feasible")]
        per_target = [[bool(r.get("feasible")) for r in recs if r["target"] == t]
                      for t in range(n_targets)]
        modes = {}
        for r in recs:
            if not r.get("feasible"):
                modes[r.get("fail_reason", "?")] = modes.get(r.get("fail_reason", "?"), 0) + 1
        summary[arm.name] = dict(
            n=len(recs),
            successes=len(ok),
            success_rate=len(ok) / len(recs) if recs else float("nan"),
            success_ci=bootstrap_success_ci(per_target),
            solver_successes=sum(1 for r in recs if r.get("solver_success")),
            timeouts=sum(1 for r in recs if r.get("timed_out")),
            iteration_capped=sum(1 for r in recs if r.get("hit_iteration_cap")),
            fail_reasons=modes,
            mean_wall_time=_mean(recs, "wall_time", ok_only=False),
            mean_wall_time_success=_mean(ok, "wall_time", ok_only=False),
            mean_setup_time=_mean(recs, "setup_time", ok_only=False),
            mean_iterations=_mean(ok, "iterations", ok_only=False),
            mean_jacobian_evals=_mean(ok, "jacobian_evals", ok_only=False),
            mean_cost=_mean(ok, "cost", ok_only=False),
            median_cost=_median(ok, "cost"),
            solved_within_k=solved_within_k(per_target),
            ## The same grid scored with the program's constraint tolerance relaxed to
            ## the task gate's, reported alongside rather than instead of the strict
            ## count. Which of the two the paper's success criterion should be is open;
            ## `max_violation` per record lets any other threshold be applied later.
            successes_relaxed=sum(1 for r in recs if r.get("feasible_relaxed")),
            success_rate_relaxed=(sum(1 for r in recs if r.get("feasible_relaxed"))
                                  / len(recs) if recs else float("nan")),
            success_ci_relaxed=bootstrap_success_ci(
                [[bool(r.get("feasible_relaxed")) for r in recs if r["target"] == t]
                 for t in range(n_targets)]),
            solved_within_k_relaxed=solved_within_k(
                [[bool(r.get("feasible_relaxed")) for r in recs if r["target"] == t]
                 for t in range(n_targets)]),
            ## Cells the relaxation turns from failure into success -- the size of the
            ## definitional question, per run.
            relaxation_gain=sum(1 for r in recs
                                if r.get("feasible_relaxed") and not r.get("feasible")),
            median_max_violation=_median(recs, "max_violation"),
            median_clip_distance=_median(recs, "clip_distance"),
            median_start_q_error=_median(recs, "start_q_error"),
            max_start_q_error=_max(recs, "start_q_error"),
            median_start_z_norm=_median(recs, "start_z_norm"),
            median_correction_inf=_median(ok, "correction_inf"),
            correction_binding=_binding_fraction(ok),
            median_z_norm=_median(ok, "z_norm"),
        )

    # Cost is only comparable on cells every formulation solved: each arm's success set
    # is a different, self-selected subset of the problems.
    common = [i for i in range(len(records[arms[0].name]))
              if all(records[a.name][i].get("feasible") for a in arms)]
    summary["_common_cells"] = common
    summary["_common"] = {}
    for arm in arms:
        recs = [records[arm.name][i] for i in common]
        summary["_common"][arm.name] = dict(
            n=len(recs),
            mean_cost=_mean(recs, "cost", ok_only=False),
            median_cost=_median(recs, "cost"),
            mean_wall_time=_mean(recs, "wall_time", ok_only=False),
            mean_iterations=_mean(recs, "iterations", ok_only=False),
            mean_jacobian_evals=_mean(recs, "jacobian_evals", ok_only=False),
        )

    summary["_mcnemar"] = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            sa = [bool(r.get("feasible")) for r in records[a.name]]
            sb = [bool(r.get("feasible")) for r in records[b.name]]
            summary["_mcnemar"][f"{a.name} vs {b.name}"] = mcnemar_exact(sa, sb)
    return summary


def _mean(recs, key, ok_only=True):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _median(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return float(np.median(vals)) if vals else float("nan")


def _max(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return float(np.max(vals)) if vals else float("nan")


def _binding_fraction(recs, tol=1e-6):
    """Fraction of solutions sitting on the correction box.

    The number that decides whether widening `correction_bound` is worth measuring: a
    correction pinned at its bound is the chart error the box refused to absorb."""
    vals = [(r["correction_inf"], r["correction_bound"]) for r in recs
            if r.get("correction_inf") is not None and r.get("correction_bound")]
    if not vals:
        return float("nan")
    return float(np.mean([c >= b - tol for c, b in vals]))


def _log_scratch_dir(log_dir):
    """Where the per-cell solver logs are written while a run is in flight.

    One 20 KB file per (cell x arm) is the shared-filesystem anti-pattern SuperCloud's
    own guidance names: a campaign accumulated 35,596 of them, 87% of every archive's
    file count, and made a routine collection metadata-bound at thirty minutes instead
    of three. They are written to node-local $TMPDIR instead and rolled into a single
    archive per run by `_roll_up_logs`.

    The subdirectory is keyed on the run's tag AND the pid because $TMPDIR is per-node,
    not per-process, and a node runs several workers at once (PROCS=8 in this campaign).
    Without a scratch directory -- a laptop run, say -- the logs stay in `log_dir` and
    are rolled up in place, so behaviour is identical either way apart from where the
    intermediate files live.
    """
    tmp = os.environ.get("TMPDIR")
    if not tmp or not os.path.isdir(tmp):
        return log_dir
    scratch = os.path.join(tmp, "solver_logs",
                           f"{os.path.basename(log_dir)}_{os.getpid()}")
    os.makedirs(scratch, exist_ok=True)
    return scratch


def _roll_up_logs(scratch, log_dir):
    """Collapse a run's per-cell solver logs into one `solver_logs.tar.gz`.

    Called once at the end of a run. Measured at 3.7x compression on real IPOPT logs,
    so the archive is both a single file and a quarter of the bytes; the per-cell logs
    remain individually recoverable with `tar xf`, which is what keeps this a storage
    change rather than a loss of diagnostic detail.

    Failure here must never lose a run: the summary is already written by the time this
    is called, and the logs are a diagnostic. So any error leaves the loose files where
    they are and reports, rather than raising into the caller.
    """
    files = sorted(glob.glob(os.path.join(scratch, "*.txt")))
    if not files:
        if scratch != log_dir:
            shutil.rmtree(scratch, ignore_errors=True)
        return None
    archive = os.path.join(log_dir, "solver_logs.tar.gz")
    try:
        os.makedirs(log_dir, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=os.path.basename(f))
    except OSError as exc:                     # out of space, unwritable dir, ...
        print(f"warning: could not roll up solver logs into {archive}: {exc}")
        return None
    for f in files:
        os.remove(f)
    if scratch != log_dir:
        shutil.rmtree(scratch, ignore_errors=True)
    return archive


def _write_summary(records, arms, n_targets, n_guesses, out_path, metadata, partial):
    payload = dict(metadata=metadata or {}, n_targets=n_targets, n_guesses=n_guesses,
                   summary=summarise(records, arms, n_targets, n_guesses),
                   records=records)
    path = out_path + ".partial" if partial else out_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    if not partial and os.path.exists(out_path + ".partial"):
        os.remove(out_path + ".partial")
    return path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def print_table(summary, arm_names):
    header = (f"{'formulation':<14} {'success':>10} {'95% CI':>16} {'solver ok':>10} "
              f"{'t/out':>6} {'i/cap':>6} {'iters':>8} {'jac':>8} {'wall(s)':>9} {'cost':>8}")
    print(header)
    print("-" * len(header))
    for name in arm_names:
        s = summary[name]
        lo, hi = s["success_ci"]
        print(f"{name:<14} {s['successes']:>4}/{s['n']:<5} [{lo:.2f}, {hi:.2f}]".ljust(42)
              + f"{s['solver_successes']:>10} {s['timeouts']:>6} "
              f"{s.get('iteration_capped', 0):>6} "
              f"{s['mean_iterations']:>8.0f} {s['mean_jacobian_evals']:>8.0f} "
              f"{s['mean_wall_time']:>9.2f} {s['median_cost']:>8.2f}")
    print(f"\ncost on the {len(summary['_common_cells'])} cells every formulation solved:")
    for name in arm_names:
        c = summary["_common"][name]
        print(f"  {name:<14} mean {c['mean_cost']:>8.3f}  median {c['median_cost']:>8.3f}  "
              f"wall {c['mean_wall_time']:>7.2f}  jac {c['mean_jacobian_evals']:>7.0f}")
    print("\npaired McNemar (exact, two-sided):")
    for pair, m in summary["_mcnemar"].items():
        print(f"  {pair:<34} {m['a_only']:>4} / {m['b_only']:<4}  p = {m['p']:.3g}")
    print("\nstart fidelity (|q(start) - q_init|, and how far the start was clipped):")
    for name in arm_names:
        s = summary[name]
        print(f"  {name:<14} median {s['median_start_q_error']:>8.4f}  "
              f"max {s['max_start_q_error']:>8.4f}  clip {s['median_clip_distance']:>8.4f}  "
              f"|z| at start {s['median_start_z_norm']:>7.3f}")
    print("\nat the solution:  |q_c| against its box, and ||z||:")
    for name in arm_names:
        s = summary[name]
        print(f"  {name:<14} median |q_c| {s['median_correction_inf']:>8.4f}  "
              f"on the box {s['correction_binding']:>6.2f}  ||z|| {s['median_z_norm']:>7.3f}")
    print("\nfailure modes:")
    for name in arm_names:
        print(f"  {name:<14} {summary[name]['fail_reasons']}")
