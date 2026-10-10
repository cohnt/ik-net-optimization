#!/usr/bin/env python3
"""Stage PAPER against stage REMEASURE: does one solve per GPU move anything but the seconds?

Stage REMEASURE, the record, ran PROCS=8 under MPS=1, a development-throughput condition. Stage
PAPER (28 runs) and stage SVGD_R2 (the four Panda IPOPT runs) re-ran its 32 IPOPT and SNOPT
logical runs on 2026-10-10 with the same args, grids and scenes at PROCS=2 with no MPS -- one
solve per V100, the paper's condition (CLAUDE.md, "Paper numbers run at one solve per GPU").
NLopt and the GVS arm were not re-run. This pairs every run with its REMEASURE twin, cell for
cell, and prints per arm:

  - success in each run, and the discordant cells (+gained/-lost at PAPER), with how many of
    those were cap-bound (timed_out / at an iteration budget) in neither run;
  - iteration identity on the cells that arm solved in BOTH runs, and how many of the cells
    that differ were cap-bound in neither run (a cell stopped by the 180 s clock stops on a
    different iterate when the machine is faster, so only an UNcapped difference is drift);
  - mean wall over all cells, each clamped at the 180 s clock (Table 3's quantity), in each
    run, and REMEASURE / PAPER -- the MPS-and-contention premium the record's seconds carried --
    beside the MEDIAN PER-CELL ratio, which a few slow cells cannot move. The two disagree on
    soft PCS IPOPT native: grasp 1.96x by means against 1.15x per cell, because 8 REMEASURE cells
    (one in each of targets 0-7) ran ~60 s longer than their PAPER twins, 473 of the run's 557 s
    of excess; pose 1.28x against 1.09x, one such cell carrying 58 of 82 s. A few slow cells in
    the development run, not a premium.

ACCEPTANCE, checked before any number prints (the script refuses otherwise): every pair shares
`scene_fingerprint` and `grid_hash`, so pairing cells is legitimate.

Usage:
    scripts/report_paper.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from report_statusquo import (ROW_ORDER, by_cell, load_record, mean, median, parse_tag)  # noqa: E402

ROW = {"mugshelf": "grasp", "posetip": "pose"}
ROBOT = {"panda": "Panda", "iiwa": "iiwa", "soft12": "soft PCS", "screw7_p050": "screw"}


def capped(r):
    return bool(r.get("timed_out") or r.get("hit_iteration_cap") or r.get("hit_eval_cap"))


def clamped_wall(s, arm):
    cap = s["metadata"].get("wall_time") or float("inf")
    return mean([min(r["wall_time"], cap) for r in s["records"][arm]])


def compare(rm, pp, arm):
    A, B = by_cell(rm, arm), by_cell(pp, arm)
    if set(A) != set(B):
        raise SystemExit(f"cell sets differ for {arm}")
    disc = [c for c in A if A[c]["feasible"] != B[c]["feasible"]]
    gained = sum(1 for c in disc if B[c]["feasible"])
    uncapped = sum(1 for c in disc if not (capped(A[c]) or capped(B[c])))
    both = [c for c in A if A[c]["feasible"] and B[c]["feasible"]]
    same_it = sum(1 for c in both if A[c]["iterations"] == B[c]["iterations"])
    it_unc = sum(1 for c in both if A[c]["iterations"] != B[c]["iterations"]
                 and not (capped(A[c]) or capped(B[c])))
    wr, wp = clamped_wall(rm, arm), clamped_wall(pp, arm)
    cell = median([A[c]["wall_time"] / B[c]["wall_time"] for c in A if B[c]["wall_time"]])
    return dict(sr=sum(r["feasible"] for r in A.values()), sp=sum(r["feasible"] for r in B.values()),
                g=gained, l=len(disc) - gained, unc=uncapped, both=len(both), same_it=same_it, it_unc=it_unc,
                wr=wr, wp=wp, ratio=wr / wp, cell=cell)


def main():
    remeasure, _ = load_record(which="remeasure")
    paper, provenance = load_record(which="paper")
    pairs = []
    for key, pp in paper.items():
        t = parse_tag(key)
        if t["solver"] == "nlopt":
            continue
        twin = "sc_REMEASURE_" + key[len("sc_PAPER_"):]
        rm = remeasure.get(twin)
        if rm is None:
            raise SystemExit(f"{key}: no REMEASURE twin {twin}")
        for field in ("scene_fingerprint", "grid_hash"):
            a, b = rm["metadata"].get(field), pp["metadata"].get(field)
            if not a or a != b:
                raise SystemExit(f"REFUSING to pair {key}: {field} {a} vs {b}")
        pairs.append((t, provenance.get(key, key), rm, pp))
    pairs.sort(key=lambda x: (x[0]["solver"], x[0]["robot"], ROW_ORDER[x[0]["row"]], x[0]["start"]))

    n_r2 = sum(1 for _, src, _, _ in pairs if src.startswith("sc_SVGD_R2_"))
    print("STAGE PAPER (+ SVGD_R2) vs STAGE REMEASURE -- the record's 32 IPOPT/SNOPT runs re-run at")
    print("PROCS=2, no MPS (paper conditions) against PROCS=8 under MPS=1, same args, grids, scenes.")
    print(f"  {len(pairs)} of 32 pairs found ({len(pairs) - n_r2} sc_PAPER_, {n_r2} sc_SVGD_R2_ Panda IPOPT).")
    print(f"  ACCEPTANCE: scene_fingerprint and grid_hash identical on all {len(pairs)} pairs.")
    print("  succ = REMEASURE -> PAPER; disc = +gained/-lost at PAPER (u = discordant cells cap-bound in")
    print("  neither run); it= = cells solved in both runs with identical iterations / cells solved in")
    print("  both; wall = mean over ALL cells clamped at 180 s (Table 3), REMEASURE / PAPER, and x = their")
    print("  ratio (> 1: PAPER faster); xc = median over cells of the per-cell wall ratio.")
    head = (f"{'succ':>10}{'disc':>9}{'it=':>10}{'wall R':>8}{'P':>7}{'x':>6}{'xc':>6}")
    print(f"\n  {'run':<34}{'learned':^56}{'joint space':^56}")
    print(f"  {'':<34}{head}{'':>0}  {head}")
    summary = {}
    for t, src, rm, pp in pairs:
        label = f"{t['solver']} {ROBOT.get(t['robot'], t['robot'])} {ROW[t['row']]} {t['start']}"
        cols = []
        for arm in ("learned", "numerical"):
            c = compare(rm, pp, arm)
            summary.setdefault((t["solver"], arm), []).append((label, c))
            disc = f"+{c['g']}/-{c['l']}" + (f" u{c['unc']}" if c["unc"] else "")
            cols.append(f"{c['sr']:>4}->{c['sp']:<4}{disc:>9}{c['same_it']:>5}/{c['both']:<4}"
                        f"{c['wr']:>8.2f}{c['wp']:>7.2f}{c['ratio']:>6.2f}{c['cell']:>6.2f}")
        print(f"  {label:<34}{cols[0]}  {cols[1]}")

    print("\n  SUMMARY per solver x arm: runs with identical success, total discordant cells (of which")
    print("  cap-bound in neither run), iteration identity on cells solved in both runs, wall ratio range.")
    for (solver, arm), rows in sorted(summary.items()):
        cs = [c for _, c in rows]
        same = sum(c["sr"] == c["sp"] for c in cs)
        disc = sum(c["g"] + c["l"] for c in cs)
        unc = sum(c["unc"] for c in cs)
        it_bad = [(lbl, c["both"] - c["same_it"]) for lbl, c in rows if c["same_it"] != c["both"]]
        it_unc = sum(c["it_unc"] for c in cs)
        rat = sorted(rows, key=lambda x: x[1]["ratio"])
        it_txt = ("identical on every run" if not it_bad else
                  f"differ on {min(b for _, b in it_bad)}-{max(b for _, b in it_bad)} cells in "
                  f"{len(it_bad)} of {len(rows)} runs, {it_unc} of them uncapped")
        print(f"    {solver:<6}{'joint space' if arm == 'numerical' else arm:<12} success identical on "
              f"{same}/{len(rows)}, {disc} discordant ({unc} uncapped); iterations {it_txt};")
        print(f"    {'':<18} wall x {rat[0][1]['ratio']:.2f} ({rat[0][0]}) .. {rat[-1][1]['ratio']:.2f} "
              f"({rat[-1][0]}), median {median([c['ratio'] for c in cs]):.2f}; per-cell xc "
              f"{min(c['cell'] for c in cs):.2f}..{max(c['cell'] for c in cs):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
