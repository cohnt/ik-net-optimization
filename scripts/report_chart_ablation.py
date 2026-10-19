#!/usr/bin/env python3
"""The chart ablation: the record's learned rows with the IKFlow chart replaced by a cVAE decoder.

Stage CVAE (cluster/gen_manifest.py) re-runs the record's learned arm -- IPOPT at paper conditions
and AL-SVGD at the V4 pin -- with ONLY the chart swapped, on two pre-registered latent rungs per
robot: K = the flow's width (the program's decision-variable count is the record's) and K = the
redundancy, DOF - 6 (whose trust region is the rule at its own width). Joint space is not re-run:
every cVAE cell pairs with the record's flow and joint-space cells on the same (target, guess),
and this script REFUSES the pairing unless each cVAE run carries the record run's `grid_hash` and
`scene_fingerprint`, the same 180 s cap and the same 480 cells.

Rows are robot x task x start; one block per solver (IPOPT, AL-SVGD); six columns per block:
  flow   -- the record's learned arm: IPOPT stage PAPER (Panda: stage SVGD_R2, the same items at the
            same condition), AL-SVGD stage SVGD_R29 `pin4`, both via report_statusquo's loaders;
            on the SCREW rows, stage SCREWREDO's `n6r` run (the same rung retrained on the rebuilt
            dataset the cVAE trains on) replaces it wherever that run exists, else the record's
            `n6`; the source of every screw row is printed to stderr;
  cVAE-K -- stage CVAE at K = the flow's width (learned decoder variance);
  cVAE-k -- stage CVAE at K = the redundancy (learned decoder variance);
  cVAEf-K, cVAEf-k -- the same two rungs with the FIXED decoder variance (the textbook objective;
            stage CVAEF / CVAEF_SVGD, checkpoints `cvaef<K>`);
  JS     -- the record's joint-space arm: the IPOPT runs' numerical arm, AL-SVGD stage SVGD_R30
            `pin4kq` (kernel q).
Per column: success of 480; median iterations over successes (IPOPT; AL-SVGD counts outer steps,
not majors, so N/A by construction); median wall over successes; and against flow and joint space,
exact McNemar on the discordant cells and the median cost on cells BOTH arms of that pair solved
(`record["cost"]` is the reported cost, the learned-only regularisers excluded; `--` under
MIN_COST_CELLS shared cells). Diagnostics: `unrepresentable_start` (a paired start the chart cannot
represent within the correction box), the cap check (clock stops and iteration-budget stops read
separately; a row with >= 24 cells at a BUDGET carries no verdict), and runaways (failures returning
|q|_inf > 1e3, the record's separator; scripts/report_screw.py).

Plus the ACCEPTANCE run (`sc_CVAEACC_`): joint space re-run on the Panda's pose paired IPOPT row
after the chart refactor, compared cell for cell with the record's joint-space arm (q, iterations,
cost and verdict identical).

A cVAE run not yet collected prints `not yet measured`; the record's columns print regardless.

Usage:
    python scripts/report_chart_ablation.py            # print the Markdown
    python scripts/report_chart_ablation.py --write    # ... and write docs/chart-ablation-tables.md
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "cluster"))
from report_statusquo import (BUDGET_CELLS, CELLS, MIN_COST_CELLS, STATUS_QUO_ROWS,  # noqa: E402
                              arm_stats, budget_bound, by_cell, clock_bound, load,
                              load_record, load_svgd_record, mcnemar, median, parse_tag)
from gen_manifest import CVAE_ACCEPT, CVAE_WIDTHS, SCREWREDO_RUNG  # noqa: E402

DEFAULT_OUT = "docs/chart-ablation-tables.md"
ROBOTS = ("panda", "iiwa", "screw7_p050", "soft12")
ROBOT_NAME = {"panda": "Panda", "iiwa": "iiwa", "screw7_p050": "screw", "soft12": "soft PCS"}
TASK_NAME = {"mugshelf": "grasp", "posetip": "pose"}
SOLVERS = (("ipopt", "IPOPT (interior point)"), ("svgd", "AL-SVGD (V4 pin)"))
#: The cVAE columns: (column, rung-label prefix, which rung of CVAE_WIDTHS, tag stem). `cvae` is the
#: learned-decoder-variance variant (stage CVAE), `cvaef` the fixed one (stage CVAEF).
CVAE_COLUMNS = (("cVAE-K", "cvae", 0, "sc_CVAE_"), ("cVAE-k", "cvae", 1, "sc_CVAE_"),
                ("cVAEf-K", "cvaef", 0, "sc_CVAEF_"), ("cVAEf-k", "cvaef", 1, "sc_CVAEF_"))
CVAE_COLS = tuple(c for c, *_ in CVAE_COLUMNS)
COLUMNS = ("flow",) + CVAE_COLS + ("JS",)
#: The record's runaway separator (scripts/report_screw.py): a failure returning |q|_inf above it.
RUNAWAY_Q = 1e3
MUST_MATCH = ("grid_hash", "scene_fingerprint", "wall_time")
NYM = "not yet measured"


def qinf(r):
    q = r.get("q")
    return float(np.max(np.abs(q))) if q is not None else float("nan")


def cvae_label(robot, which, prefix="cvae"):
    """`<prefix><K>` (`cvae` or `cvaef`) for the flow-width (which=0) or redundancy (which=1) rung."""
    return f"{prefix}{CVAE_WIDTHS[robot][which]}"


def load_columns():
    """{(solver, robot, row, start): {column: records or None}}, refusing a mismatched pairing.

    The record's flow and joint-space columns come from report_statusquo's own loaders (paper
    conditions for IPOPT; R29 pin4 / R30 pin4kq for AL-SVGD, already checked there against the
    IPOPT record run). Each cVAE run is checked here against the record run of its row."""
    record, _ = load_record(which="paper")
    ipopt = {(t["robot"], t["row"], t["start"]): s for tag, s in record.items()
             for t in [parse_tag(tag)] if t["solver"] == "ipopt"}
    svgd, _ = load_svgd_record(record)
    svgd = {(t["robot"], t["row"], t["start"]): s for tag, s in svgd.items() for t in [parse_tag(tag)]}
    cvae = {}
    for stem in sorted({st for *_, st in CVAE_COLUMNS}):
        for tag, s in load(stem, cells=CELLS).items():
            t = parse_tag(tag)
            cvae[(t["solver"], t["robot"], t["rung"], t["row"], t["start"])] = (tag, s)

    ## The screw flow column: stage SCREWREDO's retrained chart (`n6r`, on the rebuilt dataset the
    ## cVAE trains on) where its run exists, else the record's n6 run. Joint space stays the record's.
    redo, redo_robots = {}, {"screw7_p050"}
    for tag, s in load("sc_SCREWREDO_", cells=CELLS).items():
        t = parse_tag(tag)
        if t["rung"] == SCREWREDO_RUNG:
            redo[(t["solver"], t["robot"], t["row"], t["start"])] = (tag, s)

    out, refused = {}, []
    for solver, _ in SOLVERS:
        rec = ipopt if solver == "ipopt" else svgd
        for robot in ROBOTS:
            for row in STATUS_QUO_ROWS:
                for start in ("native", "paired"):
                    r = rec.get((robot, row, start))
                    cols = {c: None for c in COLUMNS}
                    if r is not None:
                        cols["flow"] = r["records"]["learned"]
                        cols["JS"] = r["records"]["numerical"]
                    hit = redo.get((solver, robot, row, start))
                    if hit is not None and r is not None:
                        tag, s = hit
                        ref = ipopt[(robot, row, start)]["metadata"]
                        bad = [f for f in MUST_MATCH
                               if s["metadata"].get(f) is None or s["metadata"].get(f) != ref.get(f)]
                        if bad or {(x["target"], x["guess"]) for x in s["records"]["learned"]} \
                                != set(by_cell(r, "learned")):
                            refused.append(f"{tag}: {bad or 'cells'} do not match the record run of its row")
                        else:
                            cols["flow"] = s["records"]["learned"]
                    if robot in redo_robots:
                        src = (hit[0] if hit is not None and r is not None else
                               "the record (n6)" if r is not None else NYM)
                        print(f"flow column, {solver} {robot} {row} {start}: {src}", file=sys.stderr)
                    for c, prefix, which, _ in CVAE_COLUMNS:
                        label = cvae_label(robot, which, prefix)
                        hit = cvae.get((solver, robot, label, row, start))
                        if hit is None:
                            continue
                        tag, s = hit
                        if r is None:
                            refused.append(f"{tag}: no record run for its row to pair with")
                            continue
                        ## The IPOPT record run's metadata for both solvers: the AL-SVGD record was
                        ## itself checked against it at load, so one reference serves both blocks.
                        ref = ipopt[(robot, row, start)]["metadata"]
                        for f in MUST_MATCH:
                            if s["metadata"].get(f) is None or s["metadata"].get(f) != ref.get(f):
                                refused.append(f"{tag}: {f} {s['metadata'].get(f)!r} != the record's "
                                               f"{ref.get(f)!r}")
                        ck = s["metadata"].get("checkpoint") or ""
                        if f"__{label}__" not in ck:
                            refused.append(f"{tag}: checkpoint {ck!r} is not the {label} chart")
                        recs = s["records"]["learned"]
                        if {(x["target"], x["guess"]) for x in recs} != set(by_cell(r, "learned")):
                            refused.append(f"{tag}: does not cover the record row's cells")
                        cols[c] = recs
                    out[(solver, robot, row, start)] = cols
    if refused:
        for msg in refused:
            print(f"REFUSING: {msg}", file=sys.stderr)
        raise SystemExit(2)
    return out


def pair(x, y):
    """x against y on the same cells: x-only and y-only successes, exact p, and each one's median
    cost on the cells both solved (None below MIN_COST_CELLS), with that count."""
    s = {"records": {"x": x, "y": y}, "metadata": {}}
    X, Y = by_cell(s, "x"), by_cell(s, "y")
    xo = sum(1 for k in X if k in Y and X[k]["feasible"] and not Y[k]["feasible"])
    yo = sum(1 for k in X if k in Y and Y[k]["feasible"] and not X[k]["feasible"])
    a, b = arm_stats(s, "x", "y"), arm_stats(s, "y", "x")
    return dict(xo=xo, yo=yo, p=mcnemar(xo, yo), cx=a["cost"], cy=b["cost"], n=a["n_both"])


def col_stats(recs):
    ok = [r for r in recs if r["feasible"]]
    fails = [r for r in recs if not r["feasible"]]
    return dict(succ=len(ok),
                iters=median([r["iterations"] for r in ok]),
                wall=median([r["wall_time"] for r in ok]),
                unrep=sum(1 for r in recs if r.get("fail_reason") == "unrepresentable_start"),
                clock=sum(map(clock_bound, recs)),
                budget=sum(map(budget_bound, recs)),
                runaway=sum(1 for r in fails if qinf(r) > RUNAWAY_Q))


def f(x, prec=0):
    return "--" if x is None or x != x else f"{x:.{prec}f}"


def tables(data):
    out = []
    pr = out.append
    pr("# Chart ablation: the cVAE decoder in place of the IKFlow chart\n")
    pr("Generated by `python scripts/report_chart_ablation.py --write`; numbers only. Rows are the record's "
       "experiments (robot x task x start), 480 cells each, 180 s, seed 1, one solve per GPU. Columns: **flow** = "
       "the record's learned arm (IPOPT: stage PAPER, Panda stage SVGD_R2; AL-SVGD: SVGD_R29 `pin4`); **cVAE-K** = "
       "stage CVAE at K = the flow's width (Panda 7, iiwa 8, screw 7, soft PCS 12; the record's trust region); "
       "**cVAE-k** = K = the redundancy (Panda 1, iiwa 1, screw 1, soft PCS 6; trust region round(sqrt(K) + 1.5, 2)); "
       "both with the LEARNED decoder variance; **cVAEf-K** / **cVAEf-k** = the same two rungs with the FIXED "
       "decoder variance (the textbook objective; stage CVAEF); "
       "**JS** = the record's joint-space arm (IPOPT: the same runs; AL-SVGD: SVGD_R30 `pin4kq`). Every cVAE run "
       "is refused unless its grid hash, scene fingerprint, cap and cells are the record row's. "
       f"`{NYM}` = no merged 480-cell cVAE run collected yet.\n")
    for solver, title in SOLVERS:
        rows = [(robot, row, start, data[(solver, robot, row, start)])
                for robot in ROBOTS for row in STATUS_QUO_ROWS for start in ("native", "paired")]
        stats = [(robot, row, start, {c: (col_stats(v) if v is not None else None) for c, v in cols.items()}, cols)
                 for robot, row, start, cols in rows]

        def name(robot, row, start):
            return f"{ROBOT_NAME[robot]} {TASK_NAME[row]} {start}"

        def cell(st, key, prec=0):
            return NYM if st is None else f(st[key], prec)

        pr(f"## {title}\n")
        pr("### Success (of 480), and McNemar of each cVAE rung against flow and joint space\n")
        pr("Pair cells: `cVAE-only / other-only, p` on the discordant cells (exact two-sided).\n")
        short = {"cVAE-K": "K", "cVAE-k": "k", "cVAEf-K": "fK", "cVAEf-k": "fk"}
        vs = [f"{short[c]} v {o}" for c in CVAE_COLS for o in ("flow", "JS")]
        pr("| row | " + " | ".join(COLUMNS) + " | " + " | ".join(vs) + " |")
        pr("| --- |" + " ---: |" * len(COLUMNS) + " --- |" * len(vs))
        for robot, row, start, st, cols in stats:
            pairs = []
            for c in CVAE_COLS:
                for o in ("flow", "JS"):
                    if cols[c] is None or cols[o] is None:
                        pairs.append(NYM if cols[c] is None else "--")
                        continue
                    p = pair(cols[c], cols[o])
                    pairs.append(f"{p['xo']} / {p['yo']}, p = {p['p']:.2g}")
            pr(f"| {name(robot, row, start)} | " + " | ".join(cell(st[c], "succ") for c in COLUMNS)
               + " | " + " | ".join(pairs) + " |")
        pr("")

        pr("### Iterations (median over successes)\n")
        if solver == "svgd":
            pr("N/A by construction: AL-SVGD counts outer steps (K = 10 inner steps each), not majors.\n")
        else:
            pr("| row | " + " | ".join(COLUMNS) + " |")
            pr("| --- |" + " ---: |" * len(COLUMNS))
            for robot, row, start, st, _ in stats:
                pr(f"| {name(robot, row, start)} | " + " | ".join(cell(st[c], "iters") for c in COLUMNS) + " |")
            pr("")

        pr("### Wall clock, s (median over successes)\n")
        pr("| row | " + " | ".join(COLUMNS) + " |")
        pr("| --- |" + " ---: |" * len(COLUMNS))
        for robot, row, start, st, _ in stats:
            pr(f"| {name(robot, row, start)} | " + " | ".join(cell(st[c], "wall", 2) for c in COLUMNS) + " |")
        pr("")

        pr("### Cost on cells both arms of the pair solved\n")
        pr(f"`median cVAE / median other (n shared)`; `--` under {MIN_COST_CELLS} shared cells. The reported "
           "cost: learned-only regularisers excluded.\n")
        pr("| row | " + " | ".join(vs) + " |")
        pr("| --- |" + " --- |" * len(vs))
        for robot, row, start, _, cols in stats:
            cells = []
            for c in CVAE_COLS:
                for o in ("flow", "JS"):
                    if cols[c] is None or cols[o] is None:
                        cells.append(NYM if cols[c] is None else "--")
                        continue
                    p = pair(cols[c], cols[o])
                    cells.append(f"{f(p['cx'], 3)} / {f(p['cy'], 3)} ({p['n']})")
            pr(f"| {name(robot, row, start)} | " + " | ".join(cells) + " |")
        pr("")

        pr("### Diagnostics: unrepresentable starts, cap check, runaways\n")
        pr("`unrep` = paired starts scored `unrepresentable_start`; `clock` / `budget` = cells stopped by the wall "
           f"clock / an iteration budget (a row with >= {BUDGET_CELLS} at a BUDGET carries no verdict, flagged "
           f"**VOID**); `run` = failures returning |q|_inf > {RUNAWAY_Q:g}.\n")
        pr("| row | flow unrep / clock / budget / run | " + " | ".join(CVAE_COLS) + " | JS clock / budget |")
        pr("| --- |" + " --- |" * (len(CVAE_COLS) + 2))
        for robot, row, start, st, _ in stats:
            def diag(s):
                if s is None:
                    return NYM
                b = f"**{s['budget']} VOID**" if s["budget"] >= BUDGET_CELLS else str(s["budget"])
                return f"{s['unrep']} / {s['clock']} / {b} / {s['runaway']}"
            js = st["JS"]
            jsd = NYM if js is None else f"{js['clock']} / " + (
                f"**{js['budget']} VOID**" if js["budget"] >= BUDGET_CELLS else str(js["budget"]))
            pr(f"| {name(robot, row, start)} | {diag(st['flow'])} | "
               + " | ".join(diag(st[c]) for c in CVAE_COLS) + f" | {jsd} |")
        pr("")
    return out


def acceptance():
    """The joint-space acceptance run against the record's joint-space arm on its row."""
    robot, rung, seed, task, start = CVAE_ACCEPT
    row = {"mug": "mugshelf", "pose": "posetip"}[task]
    tag = f"sc_CVAEACC_{robot}_{rung}_ipopt_{row}_{CELLS}_180_{start}"
    out = ["## Acceptance: joint space after the chart refactor\n",
           f"`{tag}` (joint space only, the record's flags) against the record's joint-space arm on "
           f"{ROBOT_NAME[robot]} {TASK_NAME[row]} {start} under IPOPT: per cell, q, iterations, cost and "
           "verdict must be identical.\n"]
    got = load("sc_CVAEACC_").get(tag)
    if got is None:
        return out + [f"{NYM}.\n"]
    record, prov = load_record(which="paper")
    key = f"sc_PAPER_{robot}_{rung}_ipopt_{row}_{CELLS}_180_{start}"
    ref = record.get(key)
    if ref is None:
        return out + ["the record run is missing locally; nothing to compare.\n"]
    for fld in ("grid_hash", "scene_fingerprint"):
        if got["metadata"].get(fld) != ref["metadata"].get(fld):
            return out + [f"**REFUSED**: {fld} differs ({got['metadata'].get(fld)!r} v "
                          f"{ref['metadata'].get(fld)!r}).\n"]
    A, B = by_cell(got, "numerical"), by_cell(ref, "numerical")
    if set(A) != set(B):
        return out + [f"**REFUSED**: {len(A)} cells against the record's {len(B)}; not the same cells.\n"]
    same = {k: 0 for k in ("q", "iterations", "cost", "feasible")}
    for k in A:
        a, b = A[k], B[k]
        same["q"] += a.get("q") is not None and b.get("q") is not None and \
            np.array_equal(np.asarray(a["q"]), np.asarray(b["q"]))
        same["iterations"] += a["iterations"] == b["iterations"]
        same["cost"] += a["cost"] == b["cost"]
        same["feasible"] += a["feasible"] == b["feasible"]
    verdict = "IDENTICAL" if all(v == len(A) for v in same.values()) else "**NOT IDENTICAL**"
    return out + [f"{verdict} -- of {len(A)} cells: " + ", ".join(f"{k} {v}" for k, v in same.items())
                  + f" (record run `{prov.get(key, key)}`).\n"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", nargs="?", const=DEFAULT_OUT, default=None,
                    help=f"also write the Markdown (default path {DEFAULT_OUT})")
    args = ap.parse_args()
    os.chdir(REPO)              # the record loaders read results/ relative to the repo
    data = load_columns()
    text = "\n".join(tables(data) + acceptance())
    print(text)
    if args.write:
        with open(args.write, "w") as fh:
            fh.write(text)
        print(f"wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
