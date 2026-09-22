#!/usr/bin/env python3
"""Read stage NLOPTTUNE: every NLopt option Drake now exposes, plus the cap arm.

Why this is its own reader rather than `collate.py --pair` or `report_snopttune.py`:

  * **Success is not the informative column here.** The archived NLopt baseline is at the
    FLOOR on most rows -- 0, 1, 0, 0, 39, 8 (iiwa) and 27, 1, 12, 0, 38, 8 (Panda) of 60 --
    and it times out on 60 of 60 cells in four of those rows. A setting can move the column
    a long way toward feasibility without converting a single cell, so this prints
    `median_max_violation` (the baseline sits 4.5e-02 to 9.7e-02 from feasible) beside
    success, and treats a fall in that number as a real result.
  * **NLopt reports no iteration count.** `NloptSolverDetails` carries a single `status`
    field: no majors, no evaluation count, no solve time. So the cost column is
    `mean_map_jacobians`, the program's own counter, which is the only cost measure this
    column has -- and it cross-validates, having equalled SNOPT's `User function calls
    (total)` exactly on every cell of the smoke run.
  * **One arm is a wall-clock arm, not a setting.** `cap180` re-runs the DEFAULT options at
    180 s. It is in the table because CLAUDE.md's cap rule requires it: a losing arm with
    significant timeouts is measuring throughput rather than method class, and this one
    times out on nearly everything. It is compared against the 45 s default, and its result
    is a statement about the BUDGET, never about a setting.

PRE-REGISTERED PROMOTION GATE, fixed before any result was read. A setting advances to 480
cells only if, on the LEARNED arm, it beats `default` by >= +8 cells of 60 on >= 6 of the 12
rows, OR takes any single row from <= 2/60 to >= 20/60. At most three settings advance.
Implemented inline here so it cannot drift. A NULL IS A COMPLETE RESULT: "every option Drake
now exposes, measured, and the augmented Lagrangian is still not competitive" is what closes
this axis, and CLAUDE.md records that as the thing the unswept column was missing.

Settings are named by their OPTIONS, never by the manifest token (Thomas: an internal token
"is an abbreviation. Don't do that.").

Usage:
    scripts/report_nlopttune.py                  # every candidate found
    scripts/report_nlopttune.py innertnewton ... # only these
"""
import glob
import json
import os
import sys
from math import comb

OPTION = {
    "default": "NLopt/Drake defaults (baseline): LD_AUGLAG, inner optimizer unset",
    "innermma": "local_optimizer_algorithm = LD_MMA",
    "innerccsaq": "local_optimizer_algorithm = LD_CCSAQ",
    "innerslsqp": "local_optimizer_algorithm = LD_SLSQP",
    "mma50": "local_optimizer_algorithm = LD_MMA, local_optimizer_max_eval = 50",
    "mma200": "local_optimizer_algorithm = LD_MMA, local_optimizer_max_eval = 200",
    "slsqp50": "local_optimizer_algorithm = LD_SLSQP, local_optimizer_max_eval = 50",
    "mmaloose": ("local_optimizer_algorithm = LD_MMA, "
                 "local_optimizer_xtol_rel = 1e-3, local_optimizer_ftol_rel = 1e-3"),
    "ctol1em04": "constraint_tol = 1e-4",
    "cap180": "NLopt/Drake defaults at a 180 s wall-clock cap (not a setting: a BUDGET arm)",
}
## Rows that need a caveat printed with them, so the caveat travels with the number.
CAVEAT = {
    "innerslsqp": ("puts an SQP method INSIDE the augmented Lagrangian. The outer method is "
                   "still AL and the inner subproblem is bound-constrained only, so this "
                   "reads as an AL column rather than a duplicate of SNOPT's -- but the "
                   "three-method-classes rule is Thomas's, so this row is excluded from any "
                   "adoption recommendation unless he agrees it is still an AL column."),
    "default": ("the inner optimizer is UNSET, which is what every archived NLopt cell ran "
                "with. There is no `name what NLopt already picks` control available: Drake's "
                "NLopt omits the LGPL Luksan sources, so LD_LBFGS and the whole variable-metric "
                "and truncated-Newton family are compiled out and refused, which also means "
                "whatever NLopt picks when unset is NOT LD_LBFGS. So `default` vs any named "
                "inner algorithm mixes `Drake called set_local_optimizer at all` with `which "
                "algorithm` -- unavoidably, and worth saying when reading these rows."),
    "cap180": ("a BUDGET result, not a setting result. Read it against CLAUDE.md's cap rule: "
               "the 45 s column times out on 60 of 60 learned cells in four rows, so at that "
               "cap the NLopt column measures throughput rather than method class."),
}
ROW_ORDER = {"mugfree": 0, "mugshelf": 1, "posetip": 2}
ROW_NAME = {"mugfree": "grasp free", "mugshelf": "grasp contained", "posetip": "pose tip"}
TAG = "sc_NLOPTTUNE_"
CELLS = 60
## The gate, as pre-registered.
GATE_DELTA, GATE_ROWS, GATE_FLOOR, GATE_LIFT = 8, 6, 2, 20


def mcnemar(b, w):
    """Exact two-sided McNemar on the discordant pairs alone."""
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load():
    """(robot, label, task, start) -> {setting: summary}, merged runs only.

    Parsed field by field rather than by stripping a suffix, because the cap arm's tag carries
    a DIFFERENT wall time (180 against 45) -- so deriving a candidate's tag from the
    baseline's by substitution silently misses it.
    """
    rows = {}
    for pat in (f"results/_cluster_staging/*/results/*/benchmark/{TAG}*/summary.json",
                f"results/*/benchmark/{TAG}*/summary.json"):
        for f in sorted(glob.glob(pat)):
            tag = os.path.basename(os.path.dirname(f))
            if "_shard" in tag:
                continue
            p = tag.split("_")
            if len(p) != 10:
                print(f"  warning: {tag} does not parse as a NLOPTTUNE tag; skipped")
                continue
            _, _, robot, label, solver, task, cells, wall, start, setting = p
            with open(f) as fh:
                s = json.load(fh)
            if len(s["records"].get("learned", [])) != CELLS:
                continue
            rows.setdefault((robot, label, task, start), {})[setting] = s
    return rows


def stats(s, arm="learned"):
    recs = s["records"][arm]
    solved = sum(r["feasible"] for r in recs)
    return dict(
        solved=solved,
        to=s["summary"][arm].get("timeouts"),
        viol=s["summary"][arm].get("median_max_violation"),
        jac=s["summary"][arm].get("mean_map_jacobians"),
        wall=s["summary"][arm].get("mean_wall_time"),
        cost=median([r.get("cost") for r in recs if r["feasible"]]))


def _num(x, fmt="%.3f"):
    """`--`, never `nan`: a column with no shared solved cell has no cost to report."""
    return "--" if x is None or x != x else fmt % x


def both_solved_cost(A, B, shared):
    """Median cost over cells BOTH columns solved.

    The standing rule: a median over each column's own successes compares different cell
    sets, and the easy cells are the ones a weaker column also solves, so that form
    flatters whichever column fails more.  `record["cost"]` is already the reported cost,
    with learned-only regularizers excluded.
    """
    both = [k for k in shared if A[k]["feasible"] and B[k]["feasible"]]
    return (len(both),
            median([A[k].get("cost") for k in both]),
            median([B[k].get("cost") for k in both]))


def status_hist(s, arm="learned"):
    h = {}
    for r in s["records"][arm]:
        h[r.get("solver_status")] = h.get(r.get("solver_status"), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: -kv[1]))


def main(only):
    rows = load()
    if not rows:
        print(f"no merged {CELLS}-cell {TAG}* run found")
        return
    settings = sorted({k for v in rows.values() for k in v} - {"default"})
    if only:
        settings = [t for t in settings if t in only]
    print(f"stage NLOPTTUNE: {len(rows)} row(s) of 12, {len(settings)} candidate column(s), "
          f"{CELLS} cells each, on the nightly Drake")
    print("NLopt reports no iteration count, so `jac` is mean map Jacobians -- this column's "
          "only cost measure. `viol` is median max_violation: at this floor a setting can "
          "move toward feasibility without converting a cell, and that is the signal to read.")
    print("The joint-space (numerical) arm is shown too: it is also near zero under NLopt on "
          "every archived row, so NLopt fails on the PROGRAM rather than on the chart.")

    advanced = []
    for tok in settings:
        name = OPTION.get(tok, f"{tok} (UNNAMED -- add it to OPTION)")
        print(f"\n=== {name}")
        if tok in CAVEAT:
            print(f"    NOTE: {CAVEAT[tok]}")
        table, deltas, lifted = [], [], False
        for key in sorted(rows, key=lambda k: (k[0], ROW_ORDER[k[2]], k[3])):
            have = rows[key]
            if "default" not in have or tok not in have:
                continue
            base, cand = have["default"], have[tok]
            A = {(r["target"], r["guess"]): r for r in base["records"]["learned"]}
            B = {(r["target"], r["guess"]): r for r in cand["records"]["learned"]}
            shared = [k for k in A if k in B]
            b = sum(1 for k in shared if B[k]["feasible"] and not A[k]["feasible"])
            w = sum(1 for k in shared if A[k]["feasible"] and not B[k]["feasible"])
            sb, sc = stats(base), stats(cand)
            jb, jc = stats(base, "numerical"), stats(cand, "numerical")
            deltas.append(sc["solved"] - sb["solved"])
            if sb["solved"] <= GATE_FLOOR and sc["solved"] >= GATE_LIFT:
                lifted = True
            nboth, cb, cc = both_solved_cost(A, B, shared)
            table.append((f"{key[0]} {ROW_NAME[key[2]]} {key[3]}", sb, sc, b, w,
                          mcnemar(b, w), jb["solved"], jc["solved"], nboth, cb, cc))
        if not table:
            continue
        print(f"    {'row':<32}{'base':>5}{'cand':>6}{'+':>4}{'-':>4}{'p':>9}"
              f"{'TOb':>5}{'TOc':>5}{'violb':>10}{'violc':>10}{'jacb':>8}{'jacc':>8}"
              f"{'sb':>7}{'sc':>7}{'nboth':>6}{'costb':>8}{'costc':>8}{'JSb':>5}{'JSc':>5}")
        for label, sb, sc, b, w, pv, jsb, jsc, nboth, cb, cc in table:
            flag = ""
            if pv < 0.05:
                flag = "  BETTER" if sc["solved"] > sb["solved"] else "  WORSE"
            print(f"    {label:<32}{sb['solved']:>5}{sc['solved']:>6}{b:>4}{w:>4}{pv:>9.3g}"
                  f"{sb['to']:>5}{sc['to']:>5}{sb['viol']:>10.2e}{sc['viol']:>10.2e}"
                  f"{_num(sb['jac'], '%.0f'):>8}{_num(sc['jac'], '%.0f'):>8}"
                  f"{sb['wall']:>7.1f}{sc['wall']:>7.1f}{nboth:>6}"
                  f"{_num(cb):>8}{_num(cc):>8}{jsb:>5}{jsc:>5}{flag}")
        n_big = sum(1 for d in deltas if d >= GATE_DELTA)
        passes = len(table) == 12 and (n_big >= GATE_ROWS or lifted)
        why = []
        if n_big >= GATE_ROWS:
            why.append(f">= +{GATE_DELTA} cells on {n_big} rows")
        if lifted:
            why.append(f"a row lifted from <= {GATE_FLOOR} to >= {GATE_LIFT}")
        print(f"    rows with >= +{GATE_DELTA} cells: {n_big} of {len(table)}; "
              f"a floor row lifted: {'yes' if lifted else 'no'}  ->  "
              f"{'ADVANCES to 480' if passes else 'does not advance'}"
              + (f"  ({', '.join(why)})" if why else "")
              + ("" if len(table) == 12 else f"  [INCOMPLETE, {len(table)}/12 rows]"))
        ## The cap arm is never a promotion candidate: it is a budget statement, and promoting
        ## it would mean running the whole campaign at a cap no other column uses.
        if passes and tok != "cap180":
            advanced.append(tok)

    print(f"\ngate: >= +{GATE_DELTA} cells of {CELLS} on >= {GATE_ROWS} of 12 rows, or any row "
          f"<= {GATE_FLOOR} -> >= {GATE_LIFT}; at most three settings advance")
    if advanced:
        print(f"advancing: {', '.join(OPTION.get(t, t) for t in advanced[:3])}")
        if len(advanced) > 3:
            print(f"  NOTE: {len(advanced)} settings met the gate and only three may advance; "
                  f"pick by largest number of qualifying rows and say which were dropped -- a "
                  f"silent truncation reads as 'covered everything'. Dropped: "
                  f"{', '.join(OPTION.get(t, t) for t in advanced[3:])}")
    else:
        print("advancing: NONE. That is a complete result: every option Drake now exposes was "
              "measured on all twelve rows and the augmented Lagrangian remains "
              "uncompetitive on this problem.")
    print("\nStatus histograms (learned arm; NLopt status 5 is MAXEVAL_REACHED, which should "
          "not appear because max_eval is disabled):")
    for key in sorted(rows, key=lambda k: (k[0], ROW_ORDER[k[2]], k[3])):
        for setting, s in sorted(rows[key].items()):
            if only and setting not in only and setting != "default":
                continue
            print(f"  {key[0]} {ROW_NAME[key[2]]:<16} {key[3]:<7} {setting:<14} "
                  f"{status_hist(s)}")
    print("\nNothing is adopted by this script. Fielding an NLopt configuration, or moving the "
          "project's Drake pin off 1.56.0, is Thomas's call.")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
