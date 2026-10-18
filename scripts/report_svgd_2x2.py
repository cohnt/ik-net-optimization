"""The AL-SVGD result as a double contribution: the formulation x the optimizer, on the record's rows.

Thomas's framing (2026-10-18): the proposed method is the learned-IK formulation under AL-SVGD.
Two ablations, one per contribution -- (A) the learned formulation under IPOPT / SNOPT / NLopt
(the optimizer removed) and (B) joint space under AL-SVGD (the formulation removed; kernel q,
otherwise the fielded pin, never tuned) -- and the baselines are joint space under the three
Drake solver classes. Every corner is read on the SAME cells (grid hash and scene fingerprint
asserted), 480 cells x 16 rows (four robots x grasp / pose x native / paired), 180 s, seed 1,
one solve per GPU.

Metrics (no iterations: svgd counts outer steps, not majors): SUCCESS (count, exact McNemar
against the proposed method); COST (median on cells both solved, the shared objective --
`reported_cost` excludes the learned-only regularisers -- as the ratio corner / proposed; the
seed-to-seed band on a row's median is 0.94-1.10, stage SVGD_R30, so a ratio inside it is a
tie); WALL (mean over all cells, each clamped at the cap); and TIME-MATCHED (the corner restarts
from the target's recorded starts inside the proposed method's wall on that cell, the running
solve allowed to finish: `report_time_matched.portfolio_stats`). AL-SVGD rows also print clock
stops and the median number of feasible particles at stop.

Runs: proposed = stage SVGD_R29 `pin4` (learned arm); B = SVGD_R30 `pin4kq` (numerical arm);
IPOPT twins = SVGD_R2 (Panda) / PAPER (the rest), SNOPT = PAPER, NLopt = REMEASURE_NLOPT (its
seconds are PROCS=8 under MPS, every other corner's PROCS=2). Prints the per-row tables as
Markdown; `--write docs/svgd-2x2-tables.md` writes them. Exit 2 on a refused pairing.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_time_matched import portfolio_stats  # noqa: E402

CAP = 180.0
BAND = (0.94, 1.10)
ROBOTS = [  # robot, results dir, rung, IPOPT twin tag stem, SNOPT stem, NLopt stem
    ("panda", "panda", "n6", "sc_SVGD_R2_panda_n6_ipopt", "sc_PAPER_panda_n6_snopt", "sc_REMEASURE_panda_n6_nlopt"),
    ("iiwa", "iiwa", "n4", "sc_PAPER_iiwa_n4_ipopt", "sc_PAPER_iiwa_n4_snopt", "sc_REMEASURE_iiwa_n4_nlopt"),
    ("screw7_p050", "screw_arm", "n6", "sc_PAPER_screw7_p050_n6_ipopt", "sc_PAPER_screw7_p050_n6_snopt",
     "sc_REMEASURE_screw7_p050_n6_nlopt"),
    ("soft12", "soft12", "n6", "sc_PAPER_soft12_n6_ipopt", "sc_PAPER_soft12_n6_snopt", "sc_REMEASURE_soft12_n6_nlopt"),
]
ROBOT_NAME = {"panda": "Panda", "iiwa": "iiwa", "screw7_p050": "screw", "soft12": "soft PCS"}
PROPOSED = "sc_SVGD_R29_{robot}_{rung}_svgd_{task}_480_180_{start}_pin4"
ABL_B = "sc_SVGD_R30_{robot}_{rung}_svgd_{task}_480_180_{start}_pin4kq"
CORNERS = [  # label, kind, arm
    ("B: JS + AL-SVGD", "B", "numerical"),
    ("A: learned + IPOPT", "ipopt", "learned"),
    ("A: learned + SNOPT", "snopt", "learned"),
    ("A: learned + NLopt", "nlopt", "learned"),
    ("base: JS + IPOPT", "ipopt", "numerical"),
    ("base: JS + SNOPT", "snopt", "numerical"),
    ("base: JS + NLopt", "nlopt", "numerical"),
]
MUST_MATCH = ("grid_hash", "scene_fingerprint")


def load(root, tag, arm):
    path = os.path.join(root, "results", "*", "benchmark", tag, "summary.json")
    import glob
    hits = [p for p in glob.glob(path) if "_shard" not in p]
    if len(hits) != 1:
        raise SystemExit(f"refused: {len(hits)} runs match {tag}")
    with open(hits[0]) as fh:
        s = json.load(fh)
    recs = s["records"][arm]
    for r in recs:
        if "target" not in r:
            r["target"], r["guess"] = r["target_index"], r["guess_index"]
    return recs, {k: s["metadata"].get(k) for k in MUST_MATCH}


def key(r):
    return (r["target"], r["guess"])


def mcnemar(a, b):
    """(a-only successes, b-only successes, exact two-sided p) on the cells of a."""
    bk = {key(r): bool(r["feasible"]) for r in b}
    ap = sum(bool(r["feasible"]) and not bk[key(r)] for r in a)
    bp = sum(bk[key(r)] and not bool(r["feasible"]) for r in a)
    n, k = ap + bp, min(ap, bp)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return ap, bp, p


def cost_ratio(P, X):
    xk = {key(r): r for r in X}
    shared = [(r["cost"], xk[key(r)]["cost"]) for r in P
              if bool(r["feasible"]) and bool(xk[key(r)]["feasible"])
              and r.get("cost") is not None and xk[key(r)].get("cost") is not None]
    if len(shared) < 10:
        return float("nan"), float("nan"), len(shared)
    cp, cx = st.median(c for c, _ in shared), st.median(c for _, c in shared)
    return cp, cx / cp, len(shared)


def wall(recs):
    return float(np.mean([min(r["wall_time"], CAP) for r in recs]))


def clock_stops(recs):
    return sum(bool(r.get("timed_out")) or (r.get("svgd") or {}).get("stop_reason") in ("wall_clock", "step_cap")
               for r in recs)


def n_feasible(recs):
    v = [(r.get("svgd") or {}).get("n_feasible") for r in recs]
    return st.median(x for x in v if x is not None) if any(x is not None for x in v) else None


def verdict(ap, bp, p):
    if p >= 0.05:
        return "tie"
    return "proposed" if ap > bp else "corner"


def band(ratio):
    if ratio != ratio:
        return "N/A"
    if BAND[0] <= ratio <= BAND[1]:
        return "tie"
    return "proposed cheaper" if ratio > 1 else "corner cheaper"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", default=None, help="write the Markdown tables to this file")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for robot, rdir, rung, ip, sn, nl in ROBOTS:
        stems = {"ipopt": ip, "snopt": sn, "nlopt": nl}
        for task in ("mugshelf", "posetip"):
            for start in ("native", "paired"):
                P, mp = load(args.root, PROPOSED.format(robot=robot, rung=rung, task=task, start=start), "learned")
                row = dict(robot=robot, task="grasp" if task == "mugshelf" else "pose", start=start,
                           P=dict(succ=sum(bool(r["feasible"]) for r in P), wall=wall(P), clock=clock_stops(P),
                                  nfeas=n_feasible(P)), corners={})
                for label, kind, arm in CORNERS:
                    tag = ABL_B.format(robot=robot, rung=rung, task=task, start=start) if kind == "B" \
                        else f"{stems[kind]}_{task}_480_180_{start}"
                    X, mx = load(args.root, tag, arm)
                    if mx != mp:
                        raise SystemExit(f"refused: {tag} does not pair with the proposed run: {mx} vs {mp}")
                    a, b, p = mcnemar(P, X)
                    cp, ratio, n = cost_ratio(P, X)
                    c = dict(succ=sum(bool(r["feasible"]) for r in X), ap=a, bp=b, p=p, cost_p=cp,
                             ratio=ratio, n=n, wall=wall(X), clock=clock_stops(X), nfeas=n_feasible(X))
                    if kind != "B":
                        c.update(portfolio_stats(P, X, args.n_perm, rng, cap=CAP))
                    row["corners"][label] = c
                rows.append(row)
    out = []
    pr = out.append
    pr("# AL-SVGD as a double contribution: the 2x2 on the record's rows\n")
    pr("Generated by `python scripts/report_svgd_2x2.py --write docs/svgd-2x2-tables.md`; numbers and verdicts "
       "only. Proposed = learned formulation under AL-SVGD (stage SVGD_R29 `pin4`, the V4 pin). Ablation A = the "
       "same formulation under IPOPT / SNOPT / NLopt (stages SVGD_R2 / PAPER / REMEASURE_NLOPT). Ablation B = "
       "joint space under AL-SVGD (SVGD_R30 `pin4kq`, kernel q, never tuned). Baselines = joint space under the "
       "three Drake solvers (the same stages). Same cells throughout (grid hash and scene fingerprint asserted); "
       "480 cells, 180 s, seed 1, one solve per GPU (NLopt's seconds PROCS=8 under MPS). Verdicts: exact McNemar "
       f"at p < 0.05; a cost ratio inside [{BAND[0]}, {BAND[1]}] (the seed-to-seed band, SVGD_R30) is a tie.\n")
    def rowname(r):
        return f"{ROBOT_NAME[r['robot']]} {r['task']} {r['start']}"

    labels = [c[0] for c in CORNERS]
    tm_labels = [l for l in labels if not l.startswith("B:")]
    CLASSES = (("rigid grasp", lambda r: r["robot"] != "soft12" and r["task"] == "grasp"),
               ("rigid pose", lambda r: r["robot"] != "soft12" and r["task"] == "pose"),
               ("soft PCS grasp", lambda r: r["robot"] == "soft12" and r["task"] == "grasp"),
               ("soft PCS pose", lambda r: r["robot"] == "soft12" and r["task"] == "pose"))

    def rng_str(vals, fmt="{:.2f}"):
        vals = [v for v in vals if v == v]
        if not vals:
            return "N/A"
        lo, hi = min(vals), max(vals)
        return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}-{fmt.format(hi)}"

    pr("## Summary (the condensed read; every number below is in the per-row tables that follow)\n")
    pr("Proposed: " + ", ".join(f"{r['P']['succ']}" for r in rows[:1]) + "/480 on all 16 rows, wall "
       f"{rng_str([r['P']['wall'] for r in rows])} s, {sum(r['P']['clock'] for r in rows)} clock stops, "
       f"{rng_str([r['P']['nfeas'] for r in rows], '{:.0f}')} feasible particles of 64 at stop.\n")
    pr("### Success: range over the 16 rows and the McNemar tally against the proposed method\n")
    pr("| corner | range | proposed wins / ties / losses | the wins |")
    pr("| --- | --- | --- | --- |")
    for l in labels:
        s = [(r, r["corners"][l]) for r in rows]
        v = [verdict(c["ap"], c["bp"], c["p"]) for _, c in s]
        wins = [rowname(r) for (r, c), vv in zip(s, v) if vv == "proposed"]
        pr(f"| {l} | {min(c['succ'] for _, c in s)}-{max(c['succ'] for _, c in s)} | "
           f"{v.count('proposed')} / {v.count('tie')} / {v.count('corner')} | "
           f"{'all 16' if len(wins) == 16 else ', '.join(wins) if wins else 'none'} |")
    pr("")
    pr("### Cost: ratio corner / proposed on cells both solved (range; cheaper-proposed / tie / cheaper-corner rows)\n")
    pr("| corner | " + " | ".join(c for c, _ in CLASSES) + " |")
    pr("| --- | " + " | ".join("---" for _ in CLASSES) + " |")
    for l in labels:
        cells = []
        for _, pred in CLASSES:
            rs = [r["corners"][l]["ratio"] for r in rows if pred(r)]
            b = [band(x) for x in rs]
            cells.append(f"{rng_str(rs)} ({b.count('proposed cheaper')} / {b.count('tie')} / {b.count('corner cheaper')})")
        pr(f"| {l} | " + " | ".join(cells) + " |")
    pr("")
    pr("### Wall: mean per cell (range over the 16 rows); proposed faster / level / slower (10% either way)\n")
    pr("| corner | wall range | faster / level / slower | the slower rows |")
    pr("| --- | --- | --- | --- |")
    for l in labels:
        s = [(r, r["P"]["wall"], r["corners"][l]["wall"]) for r in rows]
        f = sum(a < 0.9 * b for _, a, b in s)
        slow = [rowname(r) for r, a, b in s if a > 1.1 * b]
        pr(f"| {l} | {rng_str([b for _, _, b in s])} s | {f} / {16 - f - len(slow)} / {len(slow)} | "
           f"{', '.join(slow) if slow else 'none'} |")
    pr("")
    pr("### Time-matched: the corner restarting inside the proposed method's wall on each cell\n")
    pr("| corner | range over the 16 rows | rows at or above 99% | lowest rows |")
    pr("| --- | --- | --- | --- |")
    for l in tm_labels:
        s = sorted(((r["corners"][l]["J_match"], rowname(r)) for r in rows))
        pr(f"| {l} | {100 * s[0][0]:.1f}-{100 * s[-1][0]:.1f}% | {sum(x >= 0.99 for x, _ in s)} | "
           + ", ".join(f"{n} {100 * x:.1f}%" for x, n in s[:3]) + " |")
    pr("")

    pr("## Success (feasible cells of 480; proposed-only / corner-only cells, p)\n")
    pr("| row | proposed | " + " | ".join(labels) + " |")
    pr("| --- | --- | " + " | ".join("---" for _ in labels) + " |")
    for r in rows:
        cells = []
        for l in labels:
            c = r["corners"][l]
            cells.append(f"{c['succ']} (+{c['ap']} / -{c['bp']}, p={c['p']:.1e})")
        pr(f"| {rowname(r)} | {r['P']['succ']} | " + " | ".join(cells) + " |")
    pr("")
    pr("| corner | range over the 16 rows | proposed wins / ties / losses |")
    pr("| --- | --- | --- |")
    for l in labels:
        s = [r["corners"][l] for r in rows]
        v = [verdict(c["ap"], c["bp"], c["p"]) for c in s]
        pr(f"| {l} | {min(c['succ'] for c in s)}-{max(c['succ'] for c in s)} | "
           f"{v.count('proposed')} / {v.count('tie')} / {v.count('corner')} |")
    pr("")
    pr("## Cost (median on cells both solved, shared objective; ratio = corner / proposed, n shared)\n")
    pr("| row | proposed | " + " | ".join(labels) + " |")
    pr("| --- | --- | " + " | ".join("---" for _ in labels) + " |")
    for r in rows:
        cp = next((c["cost_p"] for c in r["corners"].values() if c["cost_p"] == c["cost_p"]), float("nan"))
        cells = [f"{c['ratio']:.2f} ({c['n']}) {band(c['ratio'])}" if c["ratio"] == c["ratio"] else f"N/A ({c['n']})"
                 for c in (r["corners"][l] for l in labels)]
        pr(f"| {rowname(r)} | {cp:.2f} | " + " | ".join(cells) + " |")
    pr("")
    pr("## Wall (mean over all cells, clamped at 180 s; AL-SVGD corners also clock stops and median feasible "
       "particles of 64)\n")
    pr("| row | proposed | " + " | ".join(labels) + " |")
    pr("| --- | --- | " + " | ".join("---" for _ in labels) + " |")
    for r in rows:
        p = r["P"]
        cells = []
        for l in labels:
            c = r["corners"][l]
            s = f"{c['wall']:.2f} s"
            if c["nfeas"] is not None:
                s += f", {c['clock']} clock, {c['nfeas']:.0f} feas"
            cells.append(s)
        pr(f"| {rowname(r)} | {p['wall']:.2f} s, {p['clock']} clock, {p['nfeas']:.0f} feas | " + " | ".join(cells) + " |")
    pr("")
    pr("| corner | wall range | proposed faster (ratio < 0.9) / level / slower |")
    pr("| --- | --- | --- |")
    for l in labels:
        s = [(r["P"]["wall"], r["corners"][l]["wall"]) for r in rows]
        f = sum(a < 0.9 * b for a, b in s)
        sl = sum(a > 1.1 * b for a, b in s)
        pr(f"| {l} | {min(b for _, b in s):.2f}-{max(b for _, b in s):.2f} s | {f} / {16 - f - sl} / {sl} |")
    pr("")
    pr("## Time-matched (the corner restarts inside the proposed method's wall on each cell, the running solve "
       "finishing; success share, starved share, median starts taken). The proposed method is 100% on every row.\n")
    pr("| row | " + " | ".join(tm_labels) + " |")
    pr("| --- | " + " | ".join("---" for _ in tm_labels) + " |")
    for r in rows:
        cells = [f"{100 * c['J_match']:.1f}% (starved {100 * c['starved']:.1f}%, {c['k_median']:.0f} starts)"
                 for c in (r["corners"][l] for l in tm_labels)]
        pr(f"| {rowname(r)} | " + " | ".join(cells) + " |")
    pr("")
    pr("| corner | time-matched range over the 16 rows | rows at or above 99% |")
    pr("| --- | --- | --- |")
    for l in tm_labels:
        s = [r["corners"][l]["J_match"] for r in rows]
        pr(f"| {l} | {100 * min(s):.1f}-{100 * max(s):.1f}% | {sum(x >= 0.99 for x in s)} |")
    text = "\n".join(out) + "\n"
    print(text)
    if args.write:
        with open(args.write, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
