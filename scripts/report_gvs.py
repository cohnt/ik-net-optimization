#!/usr/bin/env python3
"""Read the GVS push-rod arm's stages: the go/no-go pre-check, and stage GVS itself.

Same reporting rules as `scripts/report_statusquo.py`, whose helpers this reuses so the two
cannot drift: success with exact McNemar (ties stated as ties), iterations, cost on cells
BOTH arms solved (`record["cost"]` is already `reported_cost`, learned-only regularizers
excluded), wall clock, the rescue rate, and the cap check read on BOTH `timed_out` and
`hit_iteration_cap`, never timeouts alone.

Two columns are PRE-REGISTERED for this robot (docs/gvs-arm.md, "Pre-registered columns",
written before any trained-chart cell existed), because its argument rests on them:

  - TIME-MATCHED joint space: for every learned cell the budget is the wall time the learned
    arm spent on that cell; joint space runs that target's 8 recorded solves in a random
    order, stopping at the first success, and succeeds if that lands within the budget;
    averaged over 2,000 orders. Printed with the STARVED share (budget exceeds all 8 solves,
    so the true multi-start figure is higher) and the any-of-8 ceiling. Computed by
    `scripts/report_time_matched.py`'s own `row_stats`, so the record's numbers and these
    come from one function.
  - ITERATIONS ON MUTUAL SUCCESSES: the median over cells BOTH arms solved of the per-cell
    learned / joint-space iteration ratio. Per-arm medians over each arm's own successes
    compare different cells and bias against the arm that solves harder ones.

And the per-EVALUATION premium beside them: milliseconds per network-and-map evaluation
(`solver_seconds / eval_counts["map_jacobian"]`) for each arm and their ratio. On this robot
both arms pay the equilibrium solve every evaluation, which is why the premium is the
quantity the time-matched column turns on.

The two rungs draw DIFFERENT targets (the forward model differs by order), so they are
compared by target-level success rate with a bootstrap CI over targets, never by McNemar.

Usage:
    python scripts/report_gvs.py precheck  [--root <checkout holding results/>]
    python scripts/report_gvs.py stage     [--root ...] [--prefix sc_GVS_]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from report_statusquo import arm_stats, by_cell, mcnemar, median, parse_tag, verdict  # noqa: E402
from report_time_matched import row_stats  # noqa: E402

CELLS = 480
ROW_NAME = {"mugshelf": "grasp", "posetip": "pose"}
ORDER = {"ipopt": 0, "snopt": 1, "nlopt": 2}


def load_any(prefix):
    """tag -> summary for every MERGED run with the prefix, whatever arms it carries."""
    out = {}
    for root in ("results/_cluster_staging/*/results", "results"):
        for f in sorted(glob.glob(f"{root}/*/benchmark/{prefix}*/summary.json")):
            tag = os.path.basename(os.path.dirname(f))
            if "_shard" in tag:
                continue
            with open(f) as fh:
                out[tag] = json.load(fh)
    return out


def ms_per_eval(records):
    """Per-cell ms per network-and-map evaluation, over the solver's own time."""
    return [1e3 * r["solver_seconds"] / r["eval_counts"]["map_jacobian"] for r in records
            if r.get("solver_seconds") is not None
            and (r.get("eval_counts") or {}).get("map_jacobian")]


def ms_per_eval_wall(records):
    """The same over the cell's whole wall time (setup included), as the brief asks."""
    return [1e3 * r["wall_time"] / r["eval_counts"]["map_jacobian"] for r in records
            if (r.get("eval_counts") or {}).get("map_jacobian")]


def ms_per_iter(records):
    return [1e3 * r["solver_seconds"] / r["iterations"] for r in records
            if r.get("solver_seconds") is not None and r.get("iterations")]


def cap_counts(records):
    return (sum(1 for r in records if r.get("timed_out")),
            sum(1 for r in records if r.get("hit_iteration_cap")))


def f(x, prec=0):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else "%.*f" % (prec, x)


# ------------------------------------------------------------------------------ pre-check

def precheck():
    js = {t: s for t, s in load_any("sc_GVSJS_").items()
          if len(s["records"].get("numerical", [])) == CELLS}
    print("=== PRE-CHECK 1: room to win -- single-start joint-space success (stage GVS's own "
          f"cells, {CELLS} per row)")
    print("Proceed criterion proposed: below ~90% on the rows.")
    print(f"{'row':<34}{'JS ok':>7}{'%':>7}{'timed out':>10}{'iter cap':>9}"
          f"{'JS it':>7}{'JS s':>7}{'ms/eval':>8}{'grid_hash':>22}")
    for tag in sorted(js, key=lambda t: (ORDER[parse_tag(t)["solver"]], t)):
        s, t = js[tag], parse_tag(tag)
        recs = s["records"]["numerical"]
        ok = [r for r in recs if r["feasible"]]
        to, ic = cap_counts(recs)
        print(f"{t['robot'] + ' ' + ROW_NAME[t['row']] + ' ' + t['solver']:<34}"
              f"{len(ok):>7}{100 * len(ok) / len(recs):>6.1f}%{to:>10}{ic:>9}"
              f"{f(median([r['iterations'] for r in ok])):>7}"
              f"{f(median([r['wall_time'] for r in ok]), 2):>7}"
              f"{f(median(ms_per_eval(recs)), 1):>8}{s['metadata']['grid_hash']:>22}")
    if not js:
        print("  (no merged 480-cell sc_GVSJS_ runs yet)")

    prem = load_any("sc_GVSPREM_")
    print("\n=== PRE-CHECK 2: the per-evaluation premium (untrained n6 chart: cost is set by the")
    print("architecture, not the weights; its solve quality is meaningless here)")
    print("Proceed criterion proposed: below ~2x. Predicted ~1.3-1.5x.")
    print(f"{'row':<34}{'L ms/ev':>8}{'JS ms/ev':>9}{'ratio':>7}{'(wall)':>8}"
          f"{'L ms/it':>8}{'JS ms/it':>9}{'ratio':>7}")
    for tag in sorted(prem, key=lambda t: (ORDER[parse_tag(t)["solver"]], t)):
        s, t = prem[tag], parse_tag(tag)
        L, J = s["records"].get("learned", []), s["records"].get("numerical", [])
        le, je = median(ms_per_eval(L)), median(ms_per_eval(J))
        lw, jw = median(ms_per_eval_wall(L)), median(ms_per_eval_wall(J))
        li, ji = median(ms_per_iter(L)), median(ms_per_iter(J))
        ratio = lambda a, b: a / b if a and b else None
        print(f"{t['robot'] + ' ' + ROW_NAME[t['row']] + ' ' + t['solver']:<34}"
              f"{f(le, 1):>8}{f(je, 1):>9}{f(ratio(le, je), 2):>7}{f(ratio(lw, jw), 2):>8}"
              f"{f(li, 1):>8}{f(ji, 1):>9}{f(ratio(li, ji), 2):>7}")
    if not prem:
        print("  (no sc_GVSPREM_ runs yet)")

    ## The split check: the same joint-space solves, in a mixed-arm job (GVSPREM) and in a
    ## joint-space-only job (GVSJS). Agreement within ~10% licenses running the learned cells
    ## alone (GVSL) and joining them to GVSJS; the time-matched column compares exactly these
    ## wall times, so they must come from equivalent conditions.
    print("\n=== PRE-CHECK 3: is a joint-space solve's wall time the same in either job?")
    print(f"{'row':<34}{'cells':>6}{'outcome =':>10}{'median wall ratio':>18}{'iters =':>8}")
    for tag in sorted(prem, key=lambda t: (ORDER[parse_tag(t)["solver"]], t)):
        t = parse_tag(tag)
        js_tag = f"sc_GVSJS_{t['robot']}_js_{t['solver']}_{t['row']}_{t['cells']}_{t['cap']}_paired"
        if js_tag not in js:
            print(f"{t['robot'] + ' ' + ROW_NAME[t['row']] + ' ' + t['solver']:<34}  (GVSJS row not merged yet)")
            continue
        if prem[tag]["metadata"]["grid_hash"] != js[js_tag]["metadata"]["grid_hash"]:
            print(f"{tag}: GRID MISMATCH against {js_tag} -- not the same cells")
            continue
        P, Q = by_cell(prem[tag], "numerical"), by_cell(js[js_tag], "numerical")
        keys = [k for k in P if k in Q]
        same = sum(1 for k in keys if bool(P[k]["feasible"]) == bool(Q[k]["feasible"]))
        same_it = sum(1 for k in keys if P[k]["iterations"] == Q[k]["iterations"])
        ratios = [P[k]["wall_time"] / Q[k]["wall_time"] for k in keys if Q[k]["wall_time"]]
        print(f"{t['robot'] + ' ' + ROW_NAME[t['row']] + ' ' + t['solver']:<34}{len(keys):>6}"
              f"{same:>10}{f(median(ratios), 3):>18}{same_it:>8}")
    return 0


# ------------------------------------------------------------------------------ stage GVS

def mutual_iteration_ratio(s):
    A, B = by_cell(s, "learned"), by_cell(s, "numerical")
    ratios = [A[k]["iterations"] / B[k]["iterations"] for k in A
              if k in B and A[k]["feasible"] and B[k]["feasible"]
              and A[k].get("iterations") and B[k].get("iterations")]
    return median(ratios), len(ratios)


def rescue_rate(s):
    A, B = by_cell(s, "learned"), by_cell(s, "numerical")
    js_fail = [k for k in B if not B[k]["feasible"] and k in A]
    return (sum(1 for k in js_fail if A[k]["feasible"]), len(js_fail))


def target_rate_diff_ci(sa, sb, arm="learned", n_boot=2000, seed=0):
    """Difference in target-level success rate between two UNPAIRED grids, with a bootstrap
    CI that resamples whole targets independently in each (guesses within a target are
    correlated, and the two grids share no targets)."""
    def per_target(s):
        by_t = {}
        for r in s["records"][arm]:
            by_t.setdefault(r["target"], []).append(bool(r["feasible"]))
        return np.array([np.mean(v) for v in by_t.values()])
    a, b = per_target(sa), per_target(sb)
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return a.mean() - b.mean(), lo, hi


def stage(prefix):
    runs = {t: s for t, s in load_any(prefix).items()
            if len(s["records"].get("learned", [])) == CELLS
            and len(s["records"].get("numerical", [])) == CELLS}
    if not runs:
        print(f"no merged {CELLS}-cell two-arm {prefix} runs (join GVSL to GVSJS first: "
              "cluster/join_arm_runs.py)")
        return 1
    rng = np.random.default_rng(0)
    print(f"=== STAGE GVS -- {CELLS} cells, 180 s, seed 1, exact forward model")
    print("Verdicts by exact McNemar; cost on cells both arms solved (reported_cost); "
          "'TM' = time-matched joint space (pre-registered).")
    hdr = (f"{'row':<36}{'L':>5}{'JS':>5}{'p':>9}{'verdict':>12}{'L it':>6}{'JS it':>6}"
           f"{'it L/JS':>8}{'L cost':>8}{'JS cost':>8}{'L s':>7}{'JS s':>7}"
           f"{'rescue':>9}{'TM JS':>7}{'starv':>6}{'any8':>6}"
           f"{'ms/ev L':>8}{'JS':>6}{'x':>6}{'L to/ic':>8}{'JS to/ic':>9}")
    print(hdr)
    print("-" * len(hdr))
    for tag in sorted(runs, key=lambda t: (ORDER[parse_tag(t)["solver"]], parse_tag(t)["robot"],
                                           parse_tag(t)["row"], parse_tag(t)["start"])):
        s, t = runs[tag], parse_tag(tag)
        L, J = arm_stats(s, "learned", "numerical"), arm_stats(s, "numerical", "learned")
        A, B = by_cell(s, "learned"), by_cell(s, "numerical")
        lo = sum(1 for k in A if k in B and A[k]["feasible"] and not B[k]["feasible"])
        jo = sum(1 for k in A if k in B and B[k]["feasible"] and not A[k]["feasible"])
        p = mcnemar(lo, jo)
        ratio, _ = mutual_iteration_ratio(s)
        res, js_fail = rescue_rate(s)
        tm = row_stats(s, 2000, rng)
        le, je = median(ms_per_eval(s["records"]["learned"])), median(ms_per_eval(s["records"]["numerical"]))
        lto, lic = cap_counts(s["records"]["learned"])
        jto, jic = cap_counts(s["records"]["numerical"])
        label = f"{t['robot'].replace('gvs_pushrod9_', '')} {t['solver']} {ROW_NAME[t['row']]} {t['start']}"
        print(f"{label:<36}{L['succ']:>5}{J['succ']:>5}{p:>9.2g}{verdict(L['succ'], J['succ'], p):>12}"
              f"{f(L['iters']):>6}{f(J['iters']):>6}{f(ratio, 2):>8}"
              f"{f(L['cost'], 2):>8}{f(J['cost'], 2):>8}{f(L['wall'], 1):>7}{f(J['wall'], 1):>7}"
              f"{f'{res}/{js_fail}':>9}{100 * tm['J_match']:>6.1f}%{100 * tm['starved']:>5.0f}%"
              f"{100 * tm['J_any']:>5.0f}%{f(le, 1):>8}{f(je, 1):>6}"
              f"{f(le / je if le and je else None, 2):>6}{f'{lto}/{lic}':>8}{f'{jto}/{jic}':>9}")
    print("\nCap check: 'to/ic' = timed_out / hit_iteration_cap. A loss or tie where the losing "
          "arm has a budget-bound population that could close the gap carries NO verdict.")

    print("\n=== o1 against o2: target-level success rate, bootstrap CI over targets (unpaired grids)")
    keyed = {}
    for tag, s in runs.items():
        t = parse_tag(tag)
        keyed[(t["solver"], t["row"], t["start"], t["robot"])] = s
    for (solver, row, start, robot), s in sorted(keyed.items()):
        if robot != "gvs_pushrod9_o1":
            continue
        other = keyed.get((solver, row, start, "gvs_pushrod9_o2"))
        if other is None:
            continue
        for arm in ("learned", "numerical"):
            d, lo_, hi_ = target_rate_diff_ci(s, other, arm)
            print(f"  {solver} {ROW_NAME[row]} {start} {arm:<10} o1 - o2 = {100 * d:+.1f} pts "
                  f"[{100 * lo_:+.1f}, {100 * hi_:+.1f}]")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=("precheck", "stage"))
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--prefix", default="sc_GVS_")
    args = p.parse_args()
    os.chdir(args.root)
    return precheck() if args.what == "precheck" else stage(args.prefix)


if __name__ == "__main__":
    sys.exit(main())
