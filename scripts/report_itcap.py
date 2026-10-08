#!/usr/bin/env python3
"""Read stage ITCAP: every iteration-cap-bound row outside the screw arm, budgets lifted.

Each sc_ITCAP1e6_<STAGE>_<run> (IPOPT, max_iter 1e6) or sc_ITCAP1e5_<STAGE>_<run> (SNOPT, 1e5
majors and 1e8 minors) re-measures sc_<STAGE>_<run> on the same seed, grid, chart and 180 s
clock (cluster/gen_manifest.py, `stage_ITCAP`). The rows pair cell for cell on grid_hash, so
per row this prints, as SCREWCAP's reader does:

  - the verdict before and after, by exact McNemar between the arms;
  - per arm, cells gained / lost between the two runs on IDENTICAL cells, with exact McNemar;
  - the reporting quartet of the NEW run: success, median iterations, cost on cells both arms
    solved, median wall clock;
  - `still capped`: cells at the NEW iteration budget, which must be ~0 for the row to count as
    re-measured. On SNOPT rows a capped cell that ran past the 180 s clock is CYCLING (SNOPT does
    not check its time limit on a major with no minors), counted separately as `cycling`: such a
    cell made no progress and its verdict cannot depend on where it was stopped.

    python scripts/report_itcap.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from report_statusquo import arm_stats, by_cell, load, mcnemar, verdict  # noqa: E402

WALL = 180.0


def between_arms(s):
    A, B = by_cell(s, "learned"), by_cell(s, "numerical")
    L = sum(r["feasible"] for r in A.values())
    J = sum(r["feasible"] for r in B.values())
    lo = sum(1 for k in A if k in B and A[k]["feasible"] and not B[k]["feasible"])
    jo = sum(1 for k in A if k in B and B[k]["feasible"] and not A[k]["feasible"])
    p = mcnemar(lo, jo)
    return L, J, verdict(L, J, p), p


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    new = {**load("sc_ITCAP1e6_", cells=480), **load("sc_ITCAP1e5_", cells=480)}
    if not new:
        print("no merged sc_ITCAP runs found")
        return 1
    print(f"### Stage ITCAP against the originals: {len(new)} runs, iteration budgets lifted, "
          f"180 s clock\n")
    print("| run | was L / J (verdict) | now L / J (verdict) | learned gained / lost (p) "
          "| joint gained / lost (p) | iterations L / J | cost L / J (shared) | wall s L / J "
          "| still capped L / J (cycling) | timeouts L / J |")
    print("| " + " | ".join(["---"] * 10) + " |")
    moved_verdicts = 0
    for tag, s in sorted(new.items(), key=lambda kv: re.sub(r"^sc_ITCAP1e[56]_", "", kv[0])):
        orig_tag = re.sub(r"^sc_ITCAP1e[56]_", "sc_", tag)
        olds = load(orig_tag, cells=480)
        o = olds.get(orig_tag)
        name = re.sub(r"^sc_ITCAP1e[56]_", "", tag).replace("_480_180", "")
        if o is None:
            print(f"| {name} | ORIGINAL {orig_tag} NOT FOUND |" + " |" * 8)
            continue
        if s["metadata"].get("grid_hash") != o["metadata"].get("grid_hash"):
            print(f"| {name} | GRID MISMATCH |" + " |" * 8)
            continue
        was, now = between_arms(o), between_arms(s)
        moved_verdicts += was[2] != now[2]
        moved = []
        for arm in ("learned", "numerical"):
            O, N = by_cell(o, arm), by_cell(s, arm)
            g = sum(1 for k in N if N[k]["feasible"] and not O[k]["feasible"])
            lost = sum(1 for k in N if O[k]["feasible"] and not N[k]["feasible"])
            moved.append("%d / %d (p = %.2g)" % (g, lost, mcnemar(g, lost)))
        Ls, Js = arm_stats(s, "learned", "numerical"), arm_stats(s, "numerical", "learned")
        f = lambda x, pr=0: "--" if x is None else "%.*f" % (pr, x)
        capped = []
        for arm in ("learned", "numerical"):
            cap = [r for r in s["records"][arm] if r.get("hit_iteration_cap")]
            cyc = sum(1 for r in cap if (r.get("wall_time") or 0) > WALL)
            capped.append(f"{len(cap)} ({cyc})")
        tos = " / ".join(str(sum(bool(r.get("timed_out")) for r in s["records"][a]))
                         for a in ("learned", "numerical"))
        print(f"| {name} | {was[0]} / {was[1]} ({was[2]}, p = {was[3]:.2g}) "
              f"| {now[0]} / {now[1]} ({now[2]}, p = {now[3]:.2g}) | {moved[0]} | {moved[1]} "
              f"| {f(Ls['iters'])} / {f(Js['iters'])} | {f(Ls['cost'], 2)} / {f(Js['cost'], 2)} "
              f"| {f(Ls['wall'], 2)} / {f(Js['wall'], 2)} | {' / '.join(capped)} | {tos} |")
    print(f"\nVerdicts that moved: {moved_verdicts} of {len(new)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
