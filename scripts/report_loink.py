#!/usr/bin/env python3
"""The LOInK chart campaign: the record's learned rows with the IKFlow chart replaced by a LOInK chart.

Stages LOINKPROBE / LOINKEXT (cluster/gen_manifest.py, runbook cluster/LOINK_RUNBOOK.md) re-run the
record's learned arm -- IPOPT at paper conditions and AL-SVGD at the V4 pin -- with ONLY the chart
swapped (square bi-Lipschitz G, K = DOF - 6, trust region round(sqrt(K) + 1.5, 2)), twelve charts:
`loink<cost><act><recipe>`, cost m / x / u, activation r (relu) / t (tanh), recipe P (batch 4096, 620000
steps) / A (batch 500, 150000 steps). Joint space is not re-run: every LOInK cell pairs with the
record's flow and joint-space cells on the same (target, guess), and this script REFUSES the pairing
unless each LOInK run carries the record run's `grid_hash`, `scene_fingerprint` and 180 s cap, the
chart named by its tag in its `checkpoint`, and covers the record row's cells (or a probe's 60).

Rows are the 16 record rows (robot x task x start); one block per solver (IPOPT, AL-SVGD) and, inside
it, one block per recipe (P, A) of six charts between the record's columns:
  flow -- the record's learned arm (same loaders as scripts/report_chart_ablation.py: IPOPT stage PAPER,
          AL-SVGD stage SVGD_R29 `pin4`; on the screw rows SCREWREDO's `n6r` where it exists);
  JS   -- the record's joint-space arm (IPOPT: the same runs; AL-SVGD: SVGD_R30 `pin4kq`).
Tables per block: success of 480; exact McNemar of each chart against flow and against JS; iterations
(median over successes; IPOPT only); wall (median over successes); cost on cells BOTH arms of the pair
solved (median per-cell ratio chart / other, 95% bootstrap CI over whole targets; `--` under
MIN_COST_CELLS shared cells; the reported cost, learned-only regularisers excluded). Per solver: a tally
(chart v JS and chart v flow over the measured rows, McNemar p < 0.05) and the LOInK diagnostics.

Diagnostics, per chart x row (success-of-480 cells = all cells of the run):
  unrep    -- cells scored `unrepresentable_start` / cells;
  q err    -- median `start_q_error`; c err med / max -- `start_c_error` (cells with a paired learned start);
  runaway  -- failed cells returning |q|_inf > 1e3;
  trust    -- fraction of SUCCESSFUL cells with ||z|| within 1e-6 of the run's latent trust radius
              (`metadata.latent_trust_region_effective`; per-cell `z_norm`);
  clock / budget -- cells stopped by the wall clock / by an iteration budget (a row with >= 24 at a
              budget carries no verdict, flagged VOID).

A run not yet collected prints `not yet measured`, a cancelled rung (LOINK_EXCLUDED_RUNGS) its reason
(`chart does not fit (K = 1)` on the rigid arms, `recipe A: LR collapsed` on soft12 A); the record's columns print regardless. `x/60` = the
60-cell probe (guess 0 of every target; stage LOINKPROBE), shown where the row has no full run; every
pairing, cost and median against it reads the other column on those 60 cells. A full run (LOINKEXT)
replaces its probe. Tags: `sc_LOINK_<robot>_<label>_<solver>_<row>_<cells>_180_<start>`; smokes
(`sc_LOINKSMOKE_`) are never read. The acceptance run `sc_LOINKACC_` (joint space after the chart
refactor) is compared cell for cell with the record's joint-space arm.

Usage:
    python scripts/report_loink.py                      # print the Markdown
    python scripts/report_loink.py --write              # ... and write docs/loink-tables.md
    python scripts/report_loink.py --recipe P           # only the P-recipe blocks
    python scripts/report_loink.py --extra-root DIR     # also read DIR/*/benchmark/<tag>/summary.json
                                                        # (rehearsal with scratch copies; never results/)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "cluster"))
from report_statusquo import (BUDGET_CELLS, CELLS, MIN_COST_CELLS, STATUS_QUO_ROWS,  # noqa: E402
                              budget_bound, by_cell, clock_bound, load,
                              load_record, load_svgd_record, mcnemar, median, parse_tag)
from gen_manifest import (CVAE_ACCEPT, CVAE_MODEL_DIR, LOINK_EXCLUDED_RUNGS, LOINK_LABELS,  # noqa: E402
                          LOINK_RECIPES, SCREWREDO_RUNG)

DEFAULT_OUT = "docs/loink-tables.md"
ROBOTS = ("panda", "iiwa", "screw7_p050", "soft12")
ROBOT_NAME = {"panda": "Panda", "iiwa": "iiwa", "screw7_p050": "screw", "soft12": "soft PCS"}
TASK_NAME = {"mugshelf": "grasp", "posetip": "pose"}
SOLVERS = (("ipopt", "IPOPT (interior point)"), ("svgd", "AL-SVGD (V4 pin)"))
STEM = "sc_LOINK_"
PROBE_TARGETS = 60
PROBE_CELLS = {(t, 0) for t in range(PROBE_TARGETS)}
RUNAWAY_Q = 1e3
TRUST_TOL = 1e-6
BOOT = 1000
MUST_MATCH = ("grid_hash", "scene_fingerprint", "wall_time")
NYM = "not yet measured"
#: (robot, label) -> why a cancelled rung will never be measured. LOINK_EXCLUDED_RUNGS names model dirs
#: (`iiwa14:loinkmrP`), the script keys robots (`iiwa`), hence CVAE_MODEL_DIR; the rigid arms' reason wins.
_DIR_ROBOT = {d: r for r, d in CVAE_MODEL_DIR.items()}
REASON = {}
for _e in LOINK_EXCLUDED_RUNGS:
    _dir, _label = _e.split(":")
    REASON[(_DIR_ROBOT[_dir], _label)] = ("recipe A: LR collapsed" if _dir == "soft12"
                                          else "chart does not fit (K = 1)")


def nym(robot, label):
    """What a missing run prints: the cancellation reason if the rung was excluded, else NYM."""
    return REASON.get((robot, label), NYM)
RECIPES = tuple(LOINK_RECIPES)                      # ("P", "A")
LABELS = {r: tuple(l for l in LOINK_LABELS if l[-1] == r) for r in RECIPES}
ROWS = [(robot, row, start) for robot in ROBOTS for row in STATUS_QUO_ROWS for start in ("native", "paired")]


def name(robot, row, start):
    return f"{ROBOT_NAME[robot]} {TASK_NAME[row]} {start}"


def qinf(r):
    q = r.get("q")
    return float(np.max(np.abs(q))) if q is not None else float("nan")


def load_runs(prefix, cells=None, extra_roots=()):
    """tag -> summary for every MERGED run with the prefix, report_statusquo.load's roots plus the
    extra ones (later roots win)."""
    out = load(prefix, cells=cells)
    for root in extra_roots:
        for f in sorted(glob.glob(f"{root}/*/benchmark/{prefix}*/summary.json")):
            tag = os.path.basename(os.path.dirname(f))
            if "_shard" in tag:
                continue
            with open(f) as fh:
                s = json.load(fh)
            if cells is not None and len(s["records"].get("learned", [])) != cells:
                continue
            out[tag] = s
    return out


def load_columns(extra_roots=()):
    """({(solver, robot, row, start): {"flow": recs, "JS": recs, label: (recs, metadata) | None}}),
    refusing a mismatched pairing. Record columns exactly as report_chart_ablation builds them."""
    record, _ = load_record(which="paper")
    ipopt = {(t["robot"], t["row"], t["start"]): s for tag, s in record.items()
             for t in [parse_tag(tag)] if t["solver"] == "ipopt"}
    svgd, _ = load_svgd_record(record)
    svgd = {(t["robot"], t["row"], t["start"]): s for tag, s in svgd.items() for t in [parse_tag(tag)]}
    chart = {}
    for full in (False, True):          # probes first, so a full run of the row replaces its probe
        for tag, s in load_runs(STEM, cells=CELLS if full else PROBE_TARGETS, extra_roots=extra_roots).items():
            t = parse_tag(tag)
            if t["stage"] == "LOINK":
                chart[(t["solver"], t["robot"], t["rung"], t["row"], t["start"])] = (tag, s)
    redo = {}
    for tag, s in load("sc_SCREWREDO_", cells=CELLS).items():
        t = parse_tag(tag)
        if t["rung"] == SCREWREDO_RUNG:
            redo[(t["solver"], t["robot"], t["row"], t["start"])] = (tag, s)

    out, refused = {}, []
    for solver, _ in SOLVERS:
        rec = ipopt if solver == "ipopt" else svgd
        for robot, row, start in ROWS:
            r = rec.get((robot, row, start))
            cols = {"flow": None, "JS": None}
            if r is not None:
                cols["flow"] = r["records"]["learned"]
                cols["JS"] = r["records"]["numerical"]
            hit = redo.get((solver, robot, row, start))
            if hit is not None and r is not None:
                tag, s = hit
                ref = ipopt[(robot, row, start)]["metadata"]
                bad = [f for f in MUST_MATCH if s["metadata"].get(f) is None or s["metadata"].get(f) != ref.get(f)]
                if bad or {(x["target"], x["guess"]) for x in s["records"]["learned"]} != set(by_cell(r, "learned")):
                    refused.append(f"{tag}: {bad or 'cells'} do not match the record run of its row")
                else:
                    cols["flow"] = s["records"]["learned"]
            for label in LOINK_LABELS:
                cols[label] = None
                hit = chart.get((solver, robot, label, row, start))
                if hit is None:
                    continue
                tag, s = hit
                if r is None:
                    refused.append(f"{tag}: no record run for its row to pair with")
                    continue
                ref = ipopt[(robot, row, start)]["metadata"]
                for f in MUST_MATCH:
                    if s["metadata"].get(f) is None or s["metadata"].get(f) != ref.get(f):
                        refused.append(f"{tag}: {f} {s['metadata'].get(f)!r} != the record's {ref.get(f)!r}")
                ck = s["metadata"].get("checkpoint") or ""
                if f"__{label}__" not in ck:
                    refused.append(f"{tag}: checkpoint {ck!r} is not the {label} chart")
                recs = s["records"]["learned"]
                want = PROBE_CELLS if len(recs) == PROBE_TARGETS else set(by_cell(r, "learned"))
                if {(x["target"], x["guess"]) for x in recs} != want or not want <= set(by_cell(r, "learned")):
                    refused.append(f"{tag}: does not cover the record row's cells")
                cols[label] = (recs, s["metadata"])
            out[(solver, robot, row, start)] = cols
    if refused:
        for msg in refused:
            print(f"REFUSING: {msg}", file=sys.stderr)
        raise SystemExit(2)
    return out


def recs_of(v):
    """Records of a column entry: a chart entry is (recs, metadata), a record column the bare list."""
    return v[0] if isinstance(v, tuple) else v


def pair(x, y):
    """x against y on x's cells: x-only / y-only successes, exact p, median cost ratio x/y on the cells
    both solved with a 95% bootstrap CI over whole targets (None under MIN_COST_CELLS), and that count."""
    s = {"records": {"x": x, "y": y}, "metadata": {}}
    X, Y = by_cell(s, "x"), by_cell(s, "y")
    xo = sum(1 for k in X if k in Y and X[k]["feasible"] and not Y[k]["feasible"])
    yo = sum(1 for k in X if k in Y and Y[k]["feasible"] and not X[k]["feasible"])
    both = [k for k in X if k in Y and X[k]["feasible"] and Y[k]["feasible"]
            and X[k].get("cost") is not None and Y[k].get("cost") is not None and Y[k]["cost"] > 0]
    out = dict(xo=xo, yo=yo, p=mcnemar(xo, yo), n=len(both), ratio=None, lo=None, hi=None)
    if len(both) >= MIN_COST_CELLS:
        ratios = {}
        for k in both:
            ratios.setdefault(k[0], []).append(X[k]["cost"] / Y[k]["cost"])
        groups = [np.asarray(v) for v in ratios.values()]
        out["ratio"] = float(np.median(np.concatenate(groups)))
        rng = np.random.default_rng(0)
        meds = []
        for _ in range(BOOT):
            idx = rng.integers(0, len(groups), len(groups))
            meds.append(np.median(np.concatenate([groups[i] for i in idx])))
        out["lo"], out["hi"] = (float(v) for v in np.percentile(meds, [2.5, 97.5]))
    return out


def col_stats(recs):
    ok = [r for r in recs if r["feasible"]]
    return dict(n=len(recs), succ=len(ok),
                iters=median([r["iterations"] for r in ok]),
                wall=median([r["wall_time"] for r in ok]),
                clock=sum(map(clock_bound, recs)), budget=sum(map(budget_bound, recs)))


def f(x, prec=0):
    return "--" if x is None or x != x else f"{x:.{prec}f}"


def cell_str(recs, key, prec=0, miss=NYM):
    if recs is None:
        return miss
    st = col_stats(recs)
    return f(st[key], prec) + (f"/{st['n']}" if key == "succ" and st["n"] != CELLS else "")


def table(pr, header, rows):
    pr("| " + " | ".join(header) + " |")
    pr("| --- |" + " ---: |" * (len(header) - 1))
    for r in rows:
        pr("| " + " | ".join(r) + " |")
    pr("")


def mc_cell(x, y, miss=NYM):
    if x is None or y is None:
        return miss if x is None else "--"
    p = pair(x, y)
    return f"{p['xo']} / {p['yo']}, p = {p['p']:.2g}"


def cost_cell(x, y, miss=NYM):
    if x is None or y is None:
        return miss if x is None else "--"
    p = pair(x, y)
    if p["ratio"] is None:
        return f"-- ({p['n']})"
    return f"{p['ratio']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}] ({p['n']})"


def tally(data, solver):
    """Per chart: rows where it beats / ties / loses to JS and to flow (McNemar p < 0.05)."""
    out = []
    for label in LOINK_LABELS:
        cells = []
        n_full = n_probe = 0
        for other in ("JS", "flow"):
            w = t = l = 0
            for robot, row, start in ROWS:
                cols = data[(solver, robot, row, start)]
                if cols[label] is None or cols[other] is None:
                    continue
                p = pair(recs_of(cols[label]), cols[other])
                if p["p"] >= 0.05:
                    t += 1
                elif p["xo"] > p["yo"]:
                    w += 1
                else:
                    l += 1
            cells.append(f"{w} / {t} / {l}")
        for robot, row, start in ROWS:
            c = data[(solver, robot, row, start)][label]
            if c is not None:
                if len(c[0]) == CELLS:
                    n_full += 1
                else:
                    n_probe += 1
        # every robot's rung cancelled: the reasons (rigid first), else the plain NYM
        why = list(dict.fromkeys(REASON[(r, label)] for r in ROBOTS if (r, label) in REASON))
        miss = "; ".join(why) if all((r, label) in REASON for r in ROBOTS) else NYM
        out.append([label] + cells + [f"{n_full} full + {n_probe} probe" if n_full + n_probe else miss])
    return out


def trust_fraction(recs, meta):
    """Fraction of successful cells with z_norm within TRUST_TOL of the run's trust radius, or the
    reason it cannot be computed."""
    radius = meta.get("latent_trust_region_effective")
    if radius is None:
        return "n/a (metadata.latent_trust_region_effective missing)"
    ok = [r for r in recs if r["feasible"]]
    zs = [r["z_norm"] for r in ok if r.get("z_norm") is not None]
    if not ok:
        return "--"
    if not zs:
        return "n/a (per-cell z_norm missing)"
    return f"{np.mean([z >= radius - TRUST_TOL for z in zs]):.2f}"


def diag_row(label, entry, miss=NYM):
    if entry is None:
        return [miss] * 8
    recs, meta = entry if isinstance(entry, tuple) else (entry, None)
    n = len(recs)
    unrep = sum(1 for r in recs if r.get("fail_reason") == "unrepresentable_start")
    qe = [r["start_q_error"] for r in recs if r.get("start_q_error") is not None]
    ce = [r["start_c_error"] for r in recs if r.get("start_c_error") is not None]
    run = sum(1 for r in recs if not r["feasible"] and qinf(r) > RUNAWAY_Q)
    clock, budget = sum(map(clock_bound, recs)), sum(map(budget_bound, recs))
    b = f"**{budget} VOID**" if budget >= BUDGET_CELLS else str(budget)
    return [f"{unrep}/{n} ({unrep / n:.2f})", f(median(qe), 3), f(median(ce), 3),
            f(max(ce), 3) if ce else "--", str(run),
            trust_fraction(recs, meta) if meta is not None else "n/a (record column)",
            str(clock), b]


def tables(data, recipes):
    out = []
    pr = out.append
    pr("# Chart campaign: LOInK in place of the IKFlow chart\n")
    pr("Generated by `python scripts/report_loink.py --write`; numbers only. Rows are the record's experiments "
       "(robot x task x start), 480 cells each, 180 s, seed 1, one solve per GPU. **flow** = the record's learned "
       "arm (IPOPT: stage PAPER, Panda stage SVGD_R2; AL-SVGD: SVGD_R29 `pin4`; screw: SCREWREDO `n6r` where it "
       "exists); **JS** = the record's joint-space arm (IPOPT: the same runs; AL-SVGD: SVGD_R30 `pin4kq`). Charts "
       "are `loink<cost><act><recipe>`: cost m / x / u, activation r (relu) / t (tanh), recipe P (batch "
       f"{LOINK_RECIPES['P'][0]}, {LOINK_RECIPES['P'][1]} steps) / A (batch {LOINK_RECIPES['A'][0]}, "
       f"{LOINK_RECIPES['A'][1]} steps); K = DOF - 6 (Panda 1, iiwa 1, screw 1, soft PCS 6), trust region "
       "round(sqrt(K) + 1.5, 2). Every chart run is refused unless its grid hash, scene fingerprint, cap and cells "
       "are the record row's and its checkpoint is the chart its tag names. "
       f"`{NYM}` = no merged chart run collected. `chart does not fit (K = 1)` (every Panda, iiwa and screw "
       "chart) and `recipe A: LR collapsed` (soft PCS recipe A) mark the 42 rungs cancelled after their training "
       "wave, never to be measured; mechanism in `docs/notes-chart.md`, \"Training outcome (2026-10-23)\". `x/60` = the 60-cell PROBE (guess 0 of every target), shown where "
       "the row has no full run; its pairings, costs and medians are over those 60 cells only. McNemar cells: "
       "`chart-only / other-only, p` on the discordant cells (exact two-sided). Cost cells: `median per-cell ratio "
       f"chart / other [95% bootstrap CI over targets, {BOOT} resamples] (n shared cells)`, on cells both arms of "
       f"the pair solved, `--` under {MIN_COST_CELLS}; the reported cost, learned-only regularisers excluded. "
       f"Cap check: `clock` / `budget` = cells stopped by the wall clock / an iteration budget (>= {BUDGET_CELLS} "
       "at a budget is flagged **VOID**).\n")
    for solver, title in SOLVERS:
        pr(f"## {title}\n")
        pr("### Tally: rows won / tied / lost by each chart (exact McNemar, p < 0.05)\n")
        table(pr, ["chart", "v JS (chart / tie / JS)", "v flow (chart / tie / flow)", "rows measured"],
              tally(data, solver))
        for rcp in recipes:
            labels = LABELS[rcp]
            cols = ("flow",) + labels + ("JS",)
            pr(f"### {title.split(' (')[0]}, recipe {rcp} (batch {LOINK_RECIPES[rcp][0]}, "
               f"{LOINK_RECIPES[rcp][1]} steps)\n")
            rows = [(r, data[(solver,) + r]) for r in ROWS]

            def cells(key, prec=0):
                return [[name(*r)] + [cell_str(recs_of(c[x]), key, prec, nym(r[0], x)) for x in cols] for r, c in rows]
            pr("#### Success (of 480)\n")
            table(pr, ["row"] + list(cols), cells("succ"))
            for other in ("flow", "JS"):
                pr(f"#### Success: McNemar of each chart against {other}\n")
                table(pr, ["row"] + [f"{l} v {other}" for l in labels],
                      [[name(*r)] + [mc_cell(recs_of(c[l]), c[other], nym(r[0], l)) for l in labels] for r, c in rows])
            pr("#### Iterations (median over successes)\n")
            if solver == "svgd":
                pr("N/A by construction: AL-SVGD counts outer steps (K = 10 inner steps each), not majors.\n")
            else:
                table(pr, ["row"] + list(cols), cells("iters"))
            pr("#### Wall clock, s (median over successes)\n")
            table(pr, ["row"] + list(cols), cells("wall", 2))
            for other in ("flow", "JS"):
                pr(f"#### Cost on shared cells: chart / {other}\n")
                table(pr, ["row"] + [f"{l} / {other}" for l in labels],
                      [[name(*r)] + [cost_cell(recs_of(c[l]), c[other], nym(r[0], l)) for l in labels] for r, c in rows])
        pr(f"### LOInK diagnostics, {title.split(' (')[0]}\n")
        pr(f"`unrep` = cells scored `unrepresentable_start`; `q err` = median `start_q_error`; `c err` = median / "
           f"max `start_c_error`; `runaway` = failures returning |q|_inf > {RUNAWAY_Q:g}; `trust` = fraction of "
           f"successes with ||z|| within {TRUST_TOL:g} of the trust radius; `clock` / `budget` = the cap check.\n")
        rows = []
        for r in ROWS:
            c = data[(solver,) + r]
            for label in ("flow",) + LOINK_LABELS:
                d = diag_row(label, c[label], nym(r[0], label))
                rows.append([name(*r), label] + d)
        table(pr, ["row", "chart", "unrep", "q err", "c err med", "c err max", "runaway", "trust", "clock",
                   "budget"], rows)
    return out


def same_value(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a == b or (a != a and b != b)


def acceptance(extra_roots=()):
    """The joint-space acceptance run against the record's joint-space arm on its row."""
    robot, rung, seed, task, start = CVAE_ACCEPT
    row = {"mug": "mugshelf", "pose": "posetip"}[task]
    tag = f"sc_LOINKACC_{robot}_{rung}_ipopt_{row}_{CELLS}_180_{start}"
    out = ["## Acceptance: joint space after the chart refactor\n",
           f"`{tag}` (joint space only, the record's flags) against the record's joint-space arm on "
           f"{ROBOT_NAME[robot]} {TASK_NAME[row]} {start} under IPOPT: per cell, q, iterations, cost and "
           "verdict must be identical (a cell the clock stopped in both runs need only agree on the verdict).\n"]
    got = load_runs("sc_LOINKACC_", extra_roots=extra_roots).get(tag)
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
    clocked = [k for k in A if clock_bound(A[k]) and clock_bound(B[k])]
    clock_ok = sum(A[k]["feasible"] == B[k]["feasible"] for k in clocked)
    same = {k: 0 for k in ("q", "iterations", "cost", "feasible")}
    for k in A:
        if k in clocked:
            continue
        a, b = A[k], B[k]
        same["q"] += a.get("q") is not None and b.get("q") is not None and \
            np.array_equal(np.asarray(a["q"]), np.asarray(b["q"]))
        same["iterations"] += same_value(a.get("iterations"), b.get("iterations"))
        same["cost"] += same_value(a.get("cost"), b.get("cost"))
        same["feasible"] += a["feasible"] == b["feasible"]
    n = len(A) - len(clocked)
    verdict = ("IDENTICAL" if all(v == n for v in same.values()) and clock_ok == len(clocked)
               else "**NOT IDENTICAL**")
    return out + [f"{verdict} -- of the {n} cells not stopped by the clock in both runs: "
                  + ", ".join(f"{k} {v}" for k, v in same.items())
                  + f"; of the {len(clocked)} clock-stopped in both, verdict identical on {clock_ok}"
                  + f" (record run `{prov.get(key, key)}`).\n"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", nargs="?", const=DEFAULT_OUT, default=None,
                    help=f"also write the Markdown (default path {DEFAULT_OUT})")
    ap.add_argument("--recipe", choices=RECIPES, default=None, help="only this recipe's blocks")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="extra directory read like results/ (rehearsal with scratch copies)")
    args = ap.parse_args()
    os.chdir(REPO)              # the record loaders read results/ relative to the repo
    extra = [os.path.abspath(p) for p in args.extra_root]
    data = load_columns(extra)
    text = "\n".join(tables(data, (args.recipe,) if args.recipe else RECIPES) + acceptance(extra))
    print(text)
    if args.write:
        with open(args.write, "w") as fh:
            fh.write(text)
        print(f"wrote {args.write}", file=sys.stderr)


if __name__ == "__main__":
    main()
