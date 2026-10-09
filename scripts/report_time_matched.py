"""Does the learned arm's success survive a TIME-MATCHED, restart-enabled joint space?

The record's success verdicts compare one learned solve against one joint-space solve per cell.
A joint-space solve is often far cheaper, so a practitioner could restart it several times in
the time of one learned solve. This reader asks what that baseline would score, from the
per-cell records already on disk -- no new runs.

For every learned cell (target t, guess g) the budget is the wall time the learned arm spent on
that cell, success or failure. The joint-space portfolio runs target t's recorded joint-space
solves in a random order, stopping at the first success, and succeeds if that first success
lands within the budget; averaged over --n-perm random orders per target. Where the budget
exceeds ALL of a target's joint-space solves the portfolio is STARVED (more starts would have
fit), so its figure is a lower bound there, and the starved share is printed. `JS any` is the
share of targets joint space solves from at least one start, with no time limit: the
portfolio's ceiling at this many starts.

This is an ANALYSIS of recorded guesses, not multi-start machinery; multi-start as a method
stays out of scope. Each joint-space wall time includes its own program setup, which a real
multi-start would reuse, so the baseline is if anything slightly pessimistic for joint space.

    python scripts/report_time_matched.py --prefix sc_STATUSQUO_ --root <checkout with results/>
    python scripts/report_time_matched.py --prefix sc_SOFT12_    --root <checkout with results/>
    python scripts/report_time_matched.py --prefix sc_REMEASURE_ \
        --exclude sc_REMEASURE_LEGACY_ --exclude sc_REMEASURE_RULE_     # the record since 2026-10-09

Selection matches scripts/report_statusquo.py's `load`: merged runs with --cells learned cells,
staged and promoted trees both (promoted wins on a duplicate tag), per-shard directories
skipped, rows `mugshelf` (contained grasp) and `posetip` (contained pose). Success is
`feasible`. Results on the record: CLAUDE.md, "What the tables say".
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load(prefix, cells, tag_re, exclude=()):
    out = {}
    for root in ("results/_cluster_staging/*/results", "results"):
        for f in sorted(glob.glob(f"{root}/*/benchmark/{prefix}*/summary.json")):
            tag = os.path.basename(os.path.dirname(f))
            if "_shard" in tag or not tag_re.match(tag) or tag.startswith(tuple(exclude)):
                continue
            with open(f) as fh:
                s = json.load(fh)
            if len(s["records"].get("learned", [])) != cells:
                continue
            out[tag] = s
    return out


def row_stats(s, n_perm, rng):
    L, J = s["records"]["learned"], s["records"]["numerical"]
    js_by_t = {}
    for r in J:
        js_by_t.setdefault(r["target"], []).append(r)
    first, total = {}, {}
    for t, rs in js_by_t.items():
        w = np.array([r["wall_time"] for r in rs])
        ok = np.array([bool(r["feasible"]) for r in rs])
        perms = np.argsort(rng.random((n_perm, len(rs))), axis=1)
        cw = np.cumsum(w[perms], axis=1)
        okp = ok[perms]
        idx = okp.argmax(axis=1)
        ## Cumulative time at the FIRST success under each order; inf if no start succeeds.
        first[t] = np.where(okp.any(axis=1), cw[np.arange(n_perm), idx], np.inf)
        total[t] = w.sum()
    return dict(
        L=np.mean([bool(r["feasible"]) for r in L]),
        J1=np.mean([bool(r["feasible"]) for r in J]),
        J_match=np.mean([(first[r["target"]] <= r["wall_time"]).mean() for r in L]),
        J_any=np.mean([any(bool(x["feasible"]) for x in rs) for rs in js_by_t.values()]),
        starved=np.mean([total[r["target"]] < r["wall_time"] for r in L]),
        L_wall=np.median([r["wall_time"] for r in L]),
        J_wall=np.median([r["wall_time"] for r in J]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", default="sc_STATUSQUO_")
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="the checkout whose results/ holds the runs (default: this one)")
    p.add_argument("--exclude", action="append", default=[],
                   help="skip tags with this prefix; REMEASURE's control sub-stages (LEGACY, RULE) "
                        "share its prefix and would otherwise parse as robots")
    p.add_argument("--cells", type=int, default=480)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.chdir(args.root)
    tag_re = re.compile(rf"^{re.escape(args.prefix)}(\w+?)_(n\d+)_(ipopt|snopt|nlopt)_"
                        rf"(mugshelf|posetip)_{args.cells}_\d+_(native|paired)$")
    runs = load(args.prefix, args.cells, tag_re, args.exclude)
    if not runs:
        raise SystemExit(f"no merged {args.cells}-cell {args.prefix} runs under {args.root}/results")
    rng = np.random.default_rng(args.seed)
    rows = []
    for tag, s in sorted(runs.items()):
        robot, _, solver, token, start = tag_re.match(tag).groups()
        rows.append(dict(robot=robot, solver=solver, start=start,
                         task="grasp" if token == "mugshelf" else "pose",
                         **row_stats(s, args.n_perm, rng)))
    order = {"ipopt": 0, "snopt": 1, "nlopt": 2}
    rows.sort(key=lambda r: (order[r["solver"]], r["task"], r["robot"], r["start"]))
    hdr = (f"{'solver':<6} {'row':<30} {'learned':>8} {'JS x1':>7} {'JS matched':>10} "
           f"{'JS any':>7} {'starved':>8} {'L wall':>7} {'JS wall':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['solver']:<6} {r['robot'] + ' ' + r['task'] + ' ' + r['start']:<30} "
              f"{100 * r['L']:>7.1f}% {100 * r['J1']:>6.1f}% {100 * r['J_match']:>9.1f}% "
              f"{100 * r['J_any']:>6.1f}% {100 * r['starved']:>7.0f}% "
              f"{r['L_wall']:>6.2f}s {r['J_wall']:>7.2f}s")


if __name__ == "__main__":
    main()
