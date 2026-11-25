"""The obstacle campaign's tables (docs/notes-obstacles.md): stages OBST and PILLAR.

Per family x rung x row (four robots x grasp / pose, NATIVE start, 480 cells, 180 s), the four
combinations of the double contribution, read on the SAME cells:

    proposed  learned + AL-SVGD      sc_<RUNG>_<robot>_<rung>_svgd_<task>_480_180_native_pin4     (learned)
    B         JS + AL-SVGD           ..._native_pin4kq                                            (numerical)
    A         learned + IPOPT        sc_<RUNG>_<robot>_<rung>_ipopt_<task>_480_180_native         (learned)
    base      JS + IPOPT             the same IPOPT run                                           (numerical)

Metrics: success (exact McNemar of proposed against each corner, p < 0.05; also learned + IPOPT
against JS + IPOPT, and learned + SVGD against JS + SVGD), cost (median on cells both solved, the
ratio corner / proposed, tie inside [0.94, 1.10]), wall (mean per cell clamped at 180 s),
time-matched success (the corner restarting inside the proposed method's wall, the running solve
finishing: `report_time_matched.portfolio_stats`, exactly as `report_svgd_2x2`; the AL-SVGD
ablation B is not time-matched, as there), and the cap check (a run with >= 24 cells at a budget,
`timed_out`, `hit_iteration_cap` or an AL-SVGD clock / step-cap stop, carries no verdict).

Pairing: every run of one rung x row must share `grid_hash`, `scene_fingerprint` and cell set, or
the report REFUSES (exit 2). Cells are never paired across rungs. Within OBST the grid is
rung-independent, so every rung of a row must share `grid_hash` and the fingerprints must differ
between rungs. A run that is not on disk prints as `missing`.

    python scripts/report_obstacles.py [--root DIR] [--family OBST|PILLAR] [--write PATH]
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_svgd_2x2 import BAND, CAP, band, cost_ratio, key, mcnemar, verdict, wall  # noqa: E402
from report_time_matched import portfolio_stats  # noqa: E402

NO_VERDICT_CELLS = 24
ROBOTS = [("panda", "n6"), ("iiwa", "n4"), ("screw7_p050", "n6"), ("soft12", "n6")]
ROBOT_NAME = {"panda": "Panda", "iiwa": "iiwa", "screw7_p050": "screw", "soft12": "soft PCS"}
TASKS = [("mugshelf", "grasp"), ("posetip", "pose")]
FAMILIES = {
    "OBST": [(f"{n}", f"sc_OBST{n}_") for n in (0, 8, 16, 32, 64)],
    "PILLAR": [(label, f"sc_PILLAR{label}_") for label in ("000", "040", "080", "120")],
}
FAMILY_UNIT = {"OBST": "obstacles per target", "PILLAR": "pillar radius x 1000 (m)"}
#: combination -> (run kind, arm). The four corners of the 2x2, learned and JS adjacent per solver.
COMBOS = {"L+SVGD": ("svgd", "learned"), "JS+SVGD": ("svgdkq", "numerical"),
          "L+IPOPT": ("ipopt", "learned"), "JS+IPOPT": ("ipopt", "numerical")}
ORDER = list(COMBOS)
#: (label, first, second, the word when first wins, the word when second wins)
PAIRS = [("proposed v B", "L+SVGD", "JS+SVGD", "proposed", "corner"),
         ("proposed v A", "L+SVGD", "L+IPOPT", "proposed", "corner"),
         ("proposed v base", "L+SVGD", "JS+IPOPT", "proposed", "corner"),
         ("L+IPOPT v JS+IPOPT", "L+IPOPT", "JS+IPOPT", "learned", "JS")]
COST_CORNERS = ["JS+SVGD", "L+IPOPT", "JS+IPOPT"]
TM_CORNERS = ["L+IPOPT", "JS+IPOPT"]


class Refused(Exception):
    pass


def tag_of(prefix, robot, rung, task, kind):
    stem = f"{prefix}{robot}_{rung}_"
    if kind == "ipopt":
        return f"{stem}ipopt_{task}_480_180_native"
    return f"{stem}svgd_{task}_480_180_native_" + ("pin4" if kind == "svgd" else "pin4kq")


def load_run(root, tag):
    """The run's {metadata, records} or None when it is not on disk (shards excluded)."""
    hits = [p for p in glob.glob(os.path.join(root, "results", "*", "benchmark", tag, "summary.json"))
            if "_shard" not in p]
    if not hits:
        return None
    if len(hits) > 1:
        raise Refused(f"{len(hits)} runs match {tag}")
    with open(hits[0]) as fh:
        s = json.load(fh)
    for recs in s["records"].values():
        for r in recs:
            if "target" not in r:
                r["target"], r["guess"] = r["target_index"], r["guess_index"]
    return dict(tag=tag, metadata=s.get("metadata", {}), records=s["records"])


def cells_at_budget(recs):
    """Cells stopped by a budget: the clock (`timed_out`), the iteration cap, or AL-SVGD's own
    clock / step cap (`svgd.stop_reason`), as `report_svgd_2x2.clock_stops` counts them."""
    return sum(bool(r.get("timed_out")) or bool(r.get("hit_iteration_cap"))
               or (r.get("svgd") or {}).get("stop_reason") in ("wall_clock", "step_cap")
               for r in recs)


def start_collision(recs):
    v = [r.get("start_collision_value") for r in recs]
    v = [x for x in v if x is not None]
    return sum(x > 1 for x in v), len(v)


def check_row(prefix, robot, rung, task, runs):
    """One rung x row: every present run on one grid, one scene, one cell set."""
    present = {k: r for k, r in runs.items() if r is not None}
    for field in ("grid_hash", "scene_fingerprint"):
        vals = {k: r["metadata"].get(field) for k, r in present.items()}
        if present and (None in vals.values() or len(set(vals.values())) != 1):
            raise Refused(f"{prefix}{robot}_{rung}_{task}: runs do not share {field}: {vals}")
    cellsets = {(k, arm): frozenset(key(r) for r in recs)
                for k, run in present.items() for arm, recs in run["records"].items()}
    if len(set(cellsets.values())) > 1:
        raise Refused(f"{prefix}{robot}_{rung}_{task}: runs cover different cells "
                      f"({ {k: len(v) for k, v in cellsets.items()} })")


def check_across_rungs(family, row_runs):
    """OBST: one grid across the rungs of a row, a distinct fingerprint per rung."""
    if family != "OBST":
        return
    for (robot, task), by_rung in row_runs.items():
        grids, fps = {}, {}
        for label, runs in by_rung.items():
            for r in runs.values():
                if r is not None:
                    grids[label] = r["metadata"].get("grid_hash")
                    fps[label] = r["metadata"].get("scene_fingerprint")
        if len(set(grids.values())) > 1:
            raise Refused(f"OBST {robot} {task}: the grid is rung-independent but grid_hash differs "
                          f"across rungs: {grids}")
        if len(set(fps.values())) != len(fps):
            raise Refused(f"OBST {robot} {task}: scene_fingerprint must differ between rungs: {fps}")


def analyse_row(runs, n_perm, rng):
    """The row's numbers from {kind: run}; every entry None when its run is missing."""
    recs = {}
    for name, (kind, arm) in COMBOS.items():
        run = runs.get(kind)
        recs[name] = run["records"].get(arm) if run is not None else None
    out = dict(recs=recs, succ={}, wall={}, cap={}, flag={}, startcol={}, tm={}, pairs={}, cost={})
    for n, r in recs.items():
        if r is None:
            continue
        out["succ"][n] = sum(bool(x["feasible"]) for x in r)
        out["wall"][n] = wall(r)
        c = cells_at_budget(r)
        out["cap"][n] = (sum(bool(x.get("timed_out")) for x in r),
                         sum(bool(x.get("hit_iteration_cap")) for x in r), c)
        out["flag"][n] = c >= NO_VERDICT_CELLS
        out["startcol"][n] = start_collision(r)
    for label, a, b, wa, wb in PAIRS:
        if recs[a] is None or recs[b] is None:
            continue
        ap, bp, p = mcnemar(recs[a], recs[b])
        out["pairs"][label] = dict(ap=ap, bp=bp, p=p, verdict=verdict_word(ap, bp, p, wa, wb),
                                   no_verdict=out["flag"][a] or out["flag"][b])
    P = recs["L+SVGD"]
    if P is not None:
        for n in COST_CORNERS:
            if recs[n] is not None:
                out["cost"][n] = cost_ratio(P, recs[n])
        for n in TM_CORNERS:
            if recs[n] is not None:
                out["tm"][n] = portfolio_stats(P, recs[n], n_perm, rng, cap=CAP)
    return out


def verdict_word(ap, bp, p, first, second):
    v = verdict(ap, bp, p)
    return {"proposed": first, "corner": second}.get(v, "tie")


def obstacle_diag(runs):
    """Achieved / short targets / certificate clearance from the first present run's metadata."""
    for kind in ("svgd", "svgdkq", "ipopt"):
        run = runs.get(kind)
        if run is None:
            continue
        ob = run["metadata"].get("obstacles") or {}
        if ob.get("family", "none") == "none":
            return dict(family="none")
        ach = ob.get("achieved") or []
        cl = [c["witness_obstacle_clearance"] for c in (ob.get("certificates") or [])
              if c and c.get("witness_obstacle_clearance") is not None]
        return dict(family=ob.get("family"), n=ob.get("n"),
                    median=st.median(ach) if ach else None, min=min(ach) if ach else None,
                    short=len(ob.get("short_targets") or []), cap_hits=len(ob.get("cap_hits") or []),
                    cert_min=min(cl) if cl else None, n_cert=len(cl))
    return None


def rowname(robot, task):
    return f"{ROBOT_NAME[robot]} {dict(TASKS)[task]} native"


def fmt(x, spec):
    return "missing" if x is None else format(x, spec)


def build(root, families, n_perm, seed):
    """{family: {(robot, task): {rung label: dict(runs, an, diag)}}}; refuses on any bad pairing."""
    rng = np.random.default_rng(seed)
    data = {}
    for family in families:
        rows = {}
        for robot, rung in ROBOTS:
            for task, _ in TASKS:
                by_rung = {}
                for label, prefix in FAMILIES[family]:
                    runs = {k: load_run(root, tag_of(prefix, robot, rung, task, k))
                            for k in ("svgd", "svgdkq", "ipopt")}
                    check_row(prefix, robot, rung, task, runs)
                    by_rung[label] = dict(runs=runs)
                rows[(robot, task)] = by_rung
        check_across_rungs(family, {k: {l: v["runs"] for l, v in br.items()} for k, br in rows.items()})
        for by_rung in rows.values():
            for v in by_rung.values():
                v["an"] = analyse_row(v["runs"], n_perm, rng)
                v["diag"] = obstacle_diag(v["runs"])
        data[family] = rows
    return data


def render(data):
    out = []
    pr = out.append
    pr("# Obstacle campaign tables\n")
    pr("Generated by `python scripts/report_obstacles.py --write docs/obstacle-tables.md`; numbers and "
       "verdicts only. Proposed = L+SVGD (learned + AL-SVGD, `pin4`); B = JS+SVGD (`pin4kq`); A = L+IPOPT; "
       "base = JS+IPOPT. Native start, 480 cells, 180 s, seed 1. Exact McNemar at p < 0.05 within a rung; "
       f"cost ratio inside [{BAND[0]}, {BAND[1]}] is a tie; a run with >= {NO_VERDICT_CELLS} cells at a "
       "budget (`timed_out`, `hit_iteration_cap`, or AL-SVGD's `stop_reason` wall_clock / step_cap) gives `no verdict` for every comparison it enters. "
       "Cells are never paired across rungs.\n")
    for family, rows in data.items():
        unit = FAMILY_UNIT[family]
        pr(f"# {family} ({unit})\n")

        def table(title, header, cellfn):
            pr(f"## {family}: {title}\n")
            pr("| row | rung | " + " | ".join(header) + " |")
            pr("| --- | --- | " + " | ".join("---" for _ in header) + " |")
            for (robot, task), by_rung in rows.items():
                for label, v in by_rung.items():
                    pr(f"| {rowname(robot, task)} | {label} | " + " | ".join(cellfn(v)) + " |")
            pr("")

        def succ_cells(v):
            a = v["an"]
            return [fmt(a["succ"].get(n), "d") for n in ORDER]

        def mcn_cells(v):
            a = v["an"]
            cells = []
            for label, *_ in PAIRS:
                c = a["pairs"].get(label)
                if c is None:
                    cells.append("missing")
                else:
                    cells.append(f"+{c['ap']} / -{c['bp']}, p={c['p']:.1e}: "
                                 + ("no verdict" if c["no_verdict"] else c["verdict"]))
            return cells

        table("success (feasible cells of 480) and exact McNemar (first-only / second-only cells)",
              ORDER + [p[0] for p in PAIRS], lambda v: succ_cells(v) + mcn_cells(v))

        def cost_cells(v):
            a = v["an"]
            P = a["recs"]["L+SVGD"]
            cp = next((c[0] for c in a["cost"].values() if c[0] == c[0]), None)
            cells = [fmt(cp, ".2f") if P is not None else "missing"]
            for n in COST_CORNERS:
                c = a["cost"].get(n)
                cells.append("missing" if c is None else
                             (f"N/A ({c[2]})" if c[1] != c[1] else f"{c[1]:.2f} ({c[2]}) {band(c[1])}"))
            return cells

        table("cost (median of the proposed method on its shared cells; ratio = corner / proposed, "
              "n shared)", ["proposed median"] + COST_CORNERS, cost_cells)
        table("wall (mean per cell, clamped at 180 s)", ORDER,
              lambda v: [fmt(v["an"]["wall"].get(n), ".2f") for n in ORDER])

        def tm_cells(v):
            cells = []
            for n in TM_CORNERS:
                c = v["an"]["tm"].get(n)
                cells.append("missing" if c is None else
                             f"{100 * c['J_match']:.1f}% (starved {100 * c['starved']:.1f}%, "
                             f"{c['k_median']:.0f} starts)")
            return cells

        table("time-matched success of each corner inside the proposed method's wall", TM_CORNERS, tm_cells)

        def cap_cells(v):
            a = v["an"]
            cells = []
            for n in ORDER:
                c = a["cap"].get(n)
                cells.append("missing" if c is None else
                             f"{c[0]} t/o, {c[1]} it-cap" + (" **NO VERDICT**" if a["flag"][n] else ""))
            return cells

        table(f"cap check (cells timed out / at the iteration cap; no verdict at >= {NO_VERDICT_CELLS} "
              "cells at either)", ORDER, cap_cells)

        def sc_cells(v):
            a = v["an"]
            return [("missing" if n not in a["startcol"] else
                     f"{a['startcol'][n][0]} / {a['startcol'][n][1]}") for n in ORDER]

        table("starts in collision (cells with start_collision_value > 1 / cells with the value)",
              ORDER, sc_cells)

        if family == "OBST":
            pr("## OBST: obstacle sets (from the runs' metadata; achieved per target, certificate = "
               "witness-to-obstacle clearance in metres)\n")
            pr("| row | rung | achieved median | achieved min | short targets | cap hits | "
               "certificate min clearance (n certified) |")
            pr("| --- | --- | --- | --- | --- | --- | --- |")
            for (robot, task), by_rung in rows.items():
                for label, v in by_rung.items():
                    d = v["diag"]
                    if d is None:
                        cells = ["missing"] * 5
                    elif d.get("family") == "none":
                        cells = ["none (rung 0)"] + [""] * 4
                    else:
                        cells = [fmt(d["median"], "g"), fmt(d["min"], "g"), str(d["short"]),
                                 str(d["cap_hits"]), f"{fmt(d['cert_min'], '.4f')} ({d['n_cert']})"]
                    pr(f"| {rowname(robot, task)} | {label} | " + " | ".join(cells) + " |")
            pr("")

        pr(f"## {family}: across rungs (success of proposed / B / A / base, feasible cells of 480)\n")
        labels = [lab for lab, _ in FAMILIES[family]]
        pr("| row | " + " | ".join(labels) + " |")
        pr("| --- | " + " | ".join("---" for _ in labels) + " |")
        for (robot, task), by_rung in rows.items():
            cells = []
            for label in labels:
                s = by_rung[label]["an"]["succ"]
                cells.append(" / ".join(fmt(s.get(n), "d") for n in ORDER_ACROSS))
            pr(f"| {rowname(robot, task)} | " + " | ".join(cells) + " |")
        pr("")
    return "\n".join(out) + "\n"


#: proposed / B / A / base
ORDER_ACROSS = ["L+SVGD", "JS+SVGD", "L+IPOPT", "JS+IPOPT"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--family", choices=sorted(FAMILIES), action="append", default=None)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", default=None, help="write the Markdown tables to this file")
    args = ap.parse_args(argv)
    try:
        data = build(args.root, args.family or ["OBST", "PILLAR"], args.n_perm, args.seed)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    text = render(data)
    print(text)
    if args.write:
        with open(args.write, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
