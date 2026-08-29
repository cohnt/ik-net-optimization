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
import json
import math
import os
import re
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


## ------------------------------- verification -------------------------------- ##

def binding_violations(prog, x, tol):
    """Largest violation of each binding of `prog` at `x`, keyed by description.

    Mirrors ../codebase's `CheckConstraints`: the point is not only whether the returned
    point is feasible but *which* constraint it misses, since "timed out" and "converged
    to something infeasible" call for different fixes.
    """
    violations = {}
    for binding in prog.GetAllConstraints():
        evaluator = binding.evaluator()
        value = np.asarray(prog.EvalBinding(binding, x), dtype=float).flatten()
        lb = np.asarray(evaluator.lower_bound(), dtype=float).flatten()
        ub = np.asarray(evaluator.upper_bound(), dtype=float).flatten()
        below = np.where(np.isfinite(lb), lb - value, -np.inf)
        above = np.where(np.isfinite(ub), value - ub, -np.inf)
        worst = float(np.max(np.maximum(below, above)))
        if worst > tol:
            name = evaluator.get_description() or type(evaluator).__name__
            violations[name] = max(worst, violations.get(name, -np.inf))
    return violations


@dataclass
class Verdict:
    feasible: bool
    fail_reason: str = ""          # "" | nan | constraint | task_error | collision
    detail: dict = field(default_factory=dict)


def verify(program, result, task_gate, tol):
    """Score a solve from the returned point, independently of the solver's own status.

    `task_gate(q)` returns `(ok, detail_dict)` and is where the experiment states what
    "the arm is actually where it was asked to be" means -- distance off the mug axis for
    a grasp, per-axis position and rpy residual for a pose target.
    """
    x = result.get_x_val()
    try:
        q = program.VarsToQ(result.GetSolution(program.lumped_vars))
        q = np.asarray([float(v) for v in q])
    except Exception as exc:                      # a diverged solve can produce garbage
        return Verdict(False, "nan", {"exception": f"{type(exc).__name__}: {exc}"})
    if not np.all(np.isfinite(q)):
        return Verdict(False, "nan", {})

    detail = {"q": [float(v) for v in q]}
    violations = binding_violations(program.prog, x, tol)
    if violations:
        detail["violations"] = {k: float(v) for k, v in violations.items()}

    collision = float(np.asarray(program.collision_free_constraint_eval.Eval(q)).flatten()[0])
    detail["collision_value"] = collision

    # How much of its own budget the learned formulation used: |q_c| against
    # correction_bound says whether the correction box is binding, and ||z|| says whether
    # the latent left the flow's typical set (sqrt(latent_dim), so ~2.65 and ~2.83).
    if hasattr(program, "correction"):
        detail["correction_inf"] = float(np.max(np.abs(result.GetSolution(program.correction))))
    if hasattr(program, "z"):
        detail["z_norm"] = float(np.linalg.norm(result.GetSolution(program.z)))

    ok, task_detail = task_gate(program, q)
    detail.update(task_detail)

    if violations:
        return Verdict(False, "constraint", detail)
    # An interior-point method parks *on* an active constraint, so a converged solve
    # routinely reports a collision value of 1 + 1e-7. The gate has to carry the same
    # slack the program's own binding does: the collision row is scaled by 0.1, so `tol`
    # in constraint units is `10 * tol` in Drake's smoothed-distance units.
    if collision > 1.0 + 10.0 * tol:
        return Verdict(False, "collision", detail)
    if not ok:
        return Verdict(False, "task_error", detail)
    return Verdict(True, "", detail)


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
    make_program: object      # (target, q_init) -> program, already `create_prog`ed
    weight: float = 1.0       # joint-centering weight, to normalise the reported cost


def run_grid(arms, targets, guesses, task_gate, log_dir, out_path, tol,
             progress=None, metadata=None):
    """Run every (arm, target, guess) cell once and write a checkpointed summary.

    Checkpointing after each target matters: these runs are hours long and the learned
    arm's cells can each take the full wall-clock cap.
    """
    os.makedirs(log_dir, exist_ok=True)
    records = {arm.name: [] for arm in arms}
    n_targets, n_guesses = len(targets), len(guesses)

    for ti in range(n_targets):
        for gi in range(n_guesses):
            for arm in arms:
                log_path = os.path.join(log_dir, f"{arm.name}_{ti}_{gi}.txt")
                if os.path.exists(log_path):
                    os.remove(log_path)
                record = dict(target=ti, guess=gi)
                t0 = time.time()
                try:
                    program = arm.make_program(targets[ti], guesses[gi])
                    record["setup_time"] = time.time() - t0
                    record.update(start_diagnostics(program, guesses[gi]))
                    record["correction_bound"] = getattr(program.options, "correction_bound", None)
                    program.options.file_print_name = log_path
                    start = time.time()
                    result = program.Solve()
                    record["wall_time"] = time.time() - start
                    record["solver_success"] = bool(result.is_success())
                    record.update(parse_log(log_path))
                    record["timed_out"] = is_timeout(record.get("exit"))
                    verdict = verify(program, result, task_gate, tol)
                    record["feasible"] = verdict.feasible
                    record["fail_reason"] = verdict.fail_reason
                    detail = dict(verdict.detail)
                    detail.pop("q", None)
                    for key in ("correction_inf", "z_norm"):
                        record[key] = _finite(detail.pop(key, None))
                    record["detail"] = detail
                    if verdict.feasible:
                        record["cost"] = float(result.get_optimal_cost()) / arm.weight
                except Exception as exc:            # never let one cell kill a sweep
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["feasible"] = False
                    record["fail_reason"] = "error"
                records[arm.name].append(record)
                if progress is not None:
                    progress(arm.name, ti, gi, record)

        _write_summary(records, arms, n_targets, n_guesses, out_path, metadata,
                       partial=(ti < n_targets - 1))

    return records


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
            fail_reasons=modes,
            mean_wall_time=_mean(recs, "wall_time", ok_only=False),
            mean_wall_time_success=_mean(ok, "wall_time", ok_only=False),
            mean_setup_time=_mean(recs, "setup_time", ok_only=False),
            mean_iterations=_mean(ok, "iterations", ok_only=False),
            mean_jacobian_evals=_mean(ok, "jacobian_evals", ok_only=False),
            mean_cost=_mean(ok, "cost", ok_only=False),
            median_cost=_median(ok, "cost"),
            solved_within_k=solved_within_k(per_target),
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
              f"{'t/out':>6} {'iters':>8} {'jac':>8} {'wall(s)':>9} {'cost':>8}")
    print(header)
    print("-" * len(header))
    for name in arm_names:
        s = summary[name]
        lo, hi = s["success_ci"]
        print(f"{name:<14} {s['successes']:>4}/{s['n']:<5} [{lo:.2f}, {hi:.2f}]".ljust(42)
              + f"{s['solver_successes']:>10} {s['timeouts']:>6} "
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
