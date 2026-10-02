#!/usr/bin/env python3
"""Read stages SCREW, SCREWCHART and SCREWPITCH: the screw-joint arm `screw7`.

Same reporting rules as `scripts/report_statusquo.py`, whose helpers this reuses so the two
cannot drift: success with exact McNemar, iterations, cost on cells BOTH arms solved, wall
clock -- and the cap check read on BOTH `timed_out` and `hit_iteration_cap`, never timeouts
alone, because IPOPT's default 3000-iteration limit is reachable inside the 180 s cap.

Two columns are specific to this robot:

  - `at budget`: of the learned arm's FAILURES, how many stopped at a budget (wall clock or
    iteration cap). Where that number could close the gap, the row's verdict is not
    established -- the pre-registered cap rule, applied per row.
  - `|q|>1e3`: of the learned arm's failures, how many returned a configuration above 1000 rad,
    the record's runaway separator. This is what tells a gain-ceiling runaway (iiwa-style,
    1e7-1e16 rad) from an ordinary non-converged solve, and on this robot the two occur on
    different charts.

Rungs pair cell-for-cell only WITHIN a pitch; each pitch draws its own grid because the screw
changes the reachable set, so McNemar is never applied across pitches.

Usage:
    python scripts/report_screw.py            # all three stages
    python scripts/report_screw.py SCREWPITCH # one
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from report_statusquo import arm_stats, by_cell, load, mcnemar, verdict  # noqa: E402

STAGES = ("SCREW", "SCREWCHART", "SCREWPITCH")


def qinf(record):
    q = record.get("q")
    return float(np.max(np.abs(q))) if q is not None else float("nan")


def row(tag, s, prefix):
    L = arm_stats(s, "learned", "numerical")
    J = arm_stats(s, "numerical", "learned")
    A, B = by_cell(s, "learned"), by_cell(s, "numerical")
    lo = sum(1 for k in A if k in B and A[k]["feasible"] and not B[k]["feasible"])
    jo = sum(1 for k in A if k in B and B[k]["feasible"] and not A[k]["feasible"])
    p = mcnemar(lo, jo)
    fails = [r for r in A.values() if not r["feasible"]]
    budget = sum(1 for r in fails if r.get("timed_out") or r.get("hit_iteration_cap"))
    runaway = sum(1 for r in fails if qinf(r) > 1e3)
    j_fails = sum(1 for r in B.values() if not r["feasible"])
    f = lambda x, pr=0: "--" if x is None else "%.*f" % (pr, x)
    name = tag.replace(prefix, "").replace("screw7_", "").replace("_480_180", "")
    return ("| %s | %d | %d | %s, p = %.2g | %s / %s | %s / %s | %s / %s | %d of %d | %d | %d of %d |"
            % (name, L["succ"], J["succ"], verdict(L["succ"], J["succ"], p), p,
               f(L["iters"]), f(J["iters"]), f(L["cost"], 2), f(J["cost"], 2),
               f(L["wall"], 2), f(J["wall"], 2), budget, len(fails), runaway, lo, j_fails))


def main(stages):
    for stage in stages:
        prefix = f"sc_{stage}_"
        runs = load(prefix, cells=480)
        print(f"\n### Stage {stage}: {len(runs)} logical runs of 480 cells\n")
        print("| run | learned | joint space | verdict | iterations L / J | cost L / J (shared) "
              "| wall s L / J | L fails at budget | L fails with |q|>1e3 | rescued |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for tag, s in sorted(runs.items()):
            print(row(tag, s, prefix))


if __name__ == "__main__":
    main(sys.argv[1:] or STAGES)
