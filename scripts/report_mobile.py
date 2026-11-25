"""The mobile-base pilot / scale-up tables: success, cost on shared cells, wall clock and
time-matched success, together, per robot x task x base-start rule (docs/notes-mobile-base.md).

The four combinations are {learned, joint space} x {AL-SVGD, IPOPT}, read from the runs whose tag
starts with `--prefix`. Pairing is only ever WITHIN the mobile grid: the four runs of one
(robot, task, rule) must carry one `grid_hash` and one `scene_fingerprint`, or the row is
refused. The fixed-base record is never paired with these rows (a different scene); compare it
marginally, by hand, from docs/svgd-2x2-tables.md.

    python scripts/report_mobile.py --prefix mb_PILOT_

Columns per row:
  success     feasible cells / cells for each combination, `P` the proposed (learned + AL-SVGD);
  McNemar     P against each other combination on the same cells (P-only, other-only, exact p);
  cost        median reported cost of each combination on the cells it and P both solved,
              as a ratio to P's median there (N/A below 10 shared cells);
  wall        mean wall seconds per cell, clamped at the clock (`--cap`);
  matched     time-matched success of joint space under the same solver: restarts from the
              target's recorded starts inside the learned cell's wall, the running solve
              finishing (scripts/report_time_matched.py's rule);
  stops       cells at a budget (clock or step cap) -- read before calling any loss;
  reach       median start conditioning reach of the learned arm (metres from the arm base).
"""

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_time_matched import portfolio_stats                            # noqa: E402

COMBOS = (("learned", "svgd"), ("learned", "ipopt"), ("numerical", "svgd"),
          ("numerical", "ipopt"))
NAMES = {("learned", "svgd"): "P  L+SVGD", ("learned", "ipopt"): "A  L+IPOPT",
         ("numerical", "svgd"): "B  J+SVGD", ("numerical", "ipopt"): "J  J+IPOPT"}


def load(root, prefix):
    rows = {}
    for f in sorted(glob.glob(os.path.join(root, "results", "*", "benchmark", f"{prefix}*",
                                           "summary.json"))):
        if "_shard" in f:
            continue
        with open(f) as fh:
            s = json.load(fh)
        m = s["metadata"]
        if not m.get("mobile_base"):
            continue
        key = (m.get("robot"), m["task"], m["base_start"])
        for arm, recs in s["records"].items():
            for r in recs:
                r.setdefault("target", r.get("target_index"))
                r.setdefault("guess", r.get("guess_index"))
            combo = (arm, m["solver"])
            if combo in rows.setdefault(key, {}):
                raise SystemExit(f"refused: two runs for {key} {combo}")
            rows[key][combo] = (recs, m.get("grid_hash"), m.get("scene_fingerprint"))
    return rows


def _k(r):
    return (r["target"], r["guess"])


def mcnemar(a, b):
    bk = {_k(r): bool(r["feasible"]) for r in b}
    ap = sum(bool(r["feasible"]) and not bk[_k(r)] for r in a)
    bp = sum(bk[_k(r)] and not bool(r["feasible"]) for r in a)
    n, k = ap + bp, min(ap, bp)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return ap, bp, p


def shared_cost_ratio(P, X):
    xk = {_k(r): r for r in X}
    pairs = [(r["cost"], xk[_k(r)]["cost"]) for r in P
             if r["feasible"] and xk[_k(r)]["feasible"]
             and r.get("cost") is not None and xk[_k(r)].get("cost") is not None]
    if len(pairs) < 10:
        return None, len(pairs)
    return st.median(c for _, c in pairs) / st.median(c for c, _ in pairs), len(pairs)


def stops(recs):
    return sum(bool(r.get("timed_out")) or bool(r.get("hit_iteration_cap"))
               or (r.get("svgd") or {}).get("stop_reason") in ("wall_clock", "step_cap")
               for r in recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--cap", type=float, default=180.0)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    rows = load(args.root, args.prefix)
    rng = np.random.default_rng(0)
    for key in sorted(rows, key=lambda k: tuple(str(x) for x in k)):
        runs = rows[key]
        missing = [c for c in COMBOS if c not in runs]
        hashes = {v[1] for v in runs.values()}
        fps = {v[2] for v in runs.values()}
        print(f"\n== {key[0]} {key[1]} base={key[2]}"
              + (f"  (missing {missing})" if missing else ""))
        if len(hashes) != 1 or len(fps) != 1:
            print(f"   REFUSED: grid hashes {hashes}, scene fingerprints {fps}")
            continue
        P = runs.get(("learned", "svgd"), (None,))[0]
        for combo in COMBOS:
            if combo not in runs:
                continue
            recs = runs[combo][0]
            n_ok = sum(bool(r["feasible"]) for r in recs)
            wall = np.mean([min(r["wall_time"], args.cap) for r in recs])
            line = f"   {NAMES[combo]:<11} {n_ok:>4}/{len(recs):<4} wall {wall:7.2f}s  stops {stops(recs):>3}"
            if P is not None and combo != ("learned", "svgd"):
                a, b, p = mcnemar(P, recs)
                ratio, n_sh = shared_cost_ratio(P, recs)
                line += (f"  McNemar P-only {a:>3} other-only {b:>3} p={p:.3g}"
                         f"  cost/P {('%.2f' % ratio) if ratio else 'N/A'} (n={n_sh})")
            if combo[0] == "learned" and ("numerical", combo[1]) in runs:
                tm = portfolio_stats(recs, runs[("numerical", combo[1])][0], args.n_perm, rng,
                                     cap=args.cap)
                line += f"  matched J {tm['J_match']:.3f} (starved {tm['starved']:.2f})"
            if combo[0] == "learned":
                reach = [r.get("start_conditioning_reach") for r in recs
                         if r.get("start_conditioning_reach") is not None]
                if reach:
                    line += f"  reach {st.median(reach):.2f} m"
            print(line)


if __name__ == "__main__":
    main()
