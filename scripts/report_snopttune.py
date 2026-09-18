#!/usr/bin/env python3
"""Read stage SNOPTTUNE: every candidate SNOPT configuration against its own baseline.

Why this exists rather than `collate.py --pair` or `report_step.py`:

  - the stage's unit of judgement is a ROW, never a pool. Thomas, 2026-09-17: "Do not pool
    experiments, that is useless." So this prints one line per row and decides by COUNTING
    rows, and it never sums success across experiments.
  - the pre-registered rule needs three separate counts -- rows better, rows significantly
    worse, rows significantly better -- and applies them together. It is implemented here so
    the verdict cannot drift when read by a later session.
  - settings are named by their OPTION, not by the manifest token (Thomas: an internal token
    "is an abbreviation. Don't do that.").

Reads merged 480-cell summaries out of results/_cluster_staging/*/ as well as the promoted
tree, because a run in flight is only ever staged -- a glob over the promoted location alone
silently matches nothing and any check written against it passes vacuously.

Usage:
    scripts/report_snopttune.py                 # every candidate found
    scripts/report_snopttune.py hessfreq20 ...  # only these
"""
import glob
import json
import os
import sys
from math import comb

## Manifest token -> what the option actually is. Anything not here prints its token and a
## warning, so a new table entry cannot quietly appear as an abbreviation.
OPTION = {
    "default": "SNOPT defaults (baseline)",
    "nonderivls": "Nonderivative linesearch",
    "hessfreq20": "Hessian frequency = 20",
    "hessfreq100": "Hessian frequency = 100",
    "lstol0p99": "Linesearch tolerance = 0.99",
    "lstol0p1": "Linesearch tolerance = 0.1",
    "majopt1em08": "Major optimality tolerance = 1e-8",
    "elastic1e2": "Elastic weight = 100",
    "crash0": "Crash option = 0",
    "mstep0p5": "Major step limit = 0.5",
    "ndlsmstep": "Nonderivative linesearch + Major step limit = 0.5",
    "ndlshess20": "Nonderivative linesearch + Hessian frequency = 20",
    "ndlshess20mstep": ("Nonderivative linesearch + Hessian frequency = 20 "
                        "+ Major step limit = 0.5"),
}
ROW_ORDER = {"mugfree": 0, "mugshelf": 1, "posetip": 2}
ROW_NAME = {"mugfree": "grasp free", "mugshelf": "grasp contained", "posetip": "pose tip"}


def mcnemar(b, w):
    """Exact two-sided McNemar on the discordant pairs alone."""
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load_complete():
    """tag -> summary, for every MERGED 480-cell run. Per-shard dirs are skipped."""
    out = {}
    for pat in ("results/_cluster_staging/*/results/*/benchmark/sc_SNOPTTUNE_*/summary.json",
                "results/*/benchmark/sc_SNOPTTUNE_*/summary.json"):
        for f in sorted(glob.glob(pat)):
            tag = os.path.basename(os.path.dirname(f))
            if "_shard" in tag:
                continue
            with open(f) as fh:
                s = json.load(fh)
            if len(s["records"].get("learned", [])) != 480:
                continue
            out[tag] = s
    return out


def compare(base, cand, arm="learned"):
    A = {(r["target"], r["guess"]): r for r in base["records"][arm]}
    B = {(r["target"], r["guess"]): r for r in cand["records"][arm]}
    shared = [k for k in A if k in B]
    b = sum(1 for k in shared if B[k]["feasible"] and not A[k]["feasible"])
    w = sum(1 for k in shared if A[k]["feasible"] and not B[k]["feasible"])
    return (sum(A[k]["feasible"] for k in shared), sum(B[k]["feasible"] for k in shared),
            b, w, mcnemar(b, w), A, B)


def main(only):
    runs = load_complete()
    baselines = {t: s for t, s in runs.items() if t.endswith("_default")}
    tokens = sorted({t.rsplit("_", 1)[-1] for t in runs} - {"default"})
    if only:
        tokens = [t for t in tokens if t in only]
    for tok in tokens:
        rows = []
        for btag, bs in baselines.items():
            ctag = btag[:-len("default")] + tok
            if ctag not in runs:
                continue
            cs = runs[ctag]
            p = btag.split("_")
            nb, nc, b, w, pv, A, B = compare(bs, cs)
            _, jc, _, _, _, JA, JB = compare(bs, cs, "numerical")
            rows.append(dict(
                robot=p[2], task=p[5], start=p[8], base=nb, cand=nc, better=b, worse=w, p=pv,
                js_base=sum(r["feasible"] for r in JA.values()),
                js_cand=sum(r["feasible"] for r in JB.values()),
                it=median([r["iterations"] for r in B.values() if r["feasible"]]),
                wall=median([r["wall_time"] for r in B.values() if r["feasible"]]),
                to=cs["summary"]["learned"]["timeouts"]))
        if not rows:
            continue
        rows.sort(key=lambda r: (r["robot"], ROW_ORDER[r["task"]], r["start"]))
        name = OPTION.get(tok)
        if name is None:
            name = f"{tok} (UNNAMED -- add it to OPTION)"
        print(f"\n=== {name}   [{len(rows)} of 12 rows]")
        print(f"{'row':<34}{'base':>5}{'cand':>6}{'+':>5}{'-':>5}{'p':>8}"
              f"{'JSb':>6}{'JSc':>5}{'iters':>7}{'wall':>7}{'TO':>5}")
        for r in rows:
            flag = ""
            if r["p"] < 0.05:
                flag = "  BETTER" if r["cand"] > r["base"] else "  WORSE"
            label = f"{r['robot']} {ROW_NAME[r['task']]} {r['start']}"
            print(f"{label:<34}{r['base']:>5}{r['cand']:>6}{r['better']:>5}{r['worse']:>5}"
                  f"{r['p']:>8.3g}{r['js_base']:>6}{r['js_cand']:>5}{r['it']:>7.0f}"
                  f"{r['wall']:>7.2f}{r['to']:>5}{flag}")
        nb = sum(1 for r in rows if r["cand"] > r["base"])
        nw = sum(1 for r in rows if r["cand"] < r["base"])
        sb = sum(1 for r in rows if r["p"] < 0.05 and r["cand"] > r["base"])
        sw = sum(1 for r in rows if r["p"] < 0.05 and r["cand"] < r["base"])
        ## The rule, exactly as pre-registered in cluster/RUNSTATE_SNOPTTUNE.md before any
        ## result was visible: >= 9 of 12 rows better, none significantly worse, >= 1
        ## significantly better. Partial coverage can only ever FAIL or be undecided.
        if len(rows) < 12:
            verdict = f"INCOMPLETE ({len(rows)}/12 rows)"
        else:
            verdict = "PASSES" if (nb >= 9 and sw == 0 and sb >= 1) else "FAILS"
        print(f"  rows better {nb}, worse {nw};  significantly better {sb}, "
              f"significantly worse {sw}   ->  {verdict}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
