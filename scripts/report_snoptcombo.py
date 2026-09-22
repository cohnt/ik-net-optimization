#!/usr/bin/env python3
"""Read stage SNOPTCOMBO: SNOPTTUNE's survivor crossed with the other positive factors.

Why this exists rather than reusing `report_snopttune.py`: this stage has TWO pre-registered
bars, not one, and they are asked against different references in the same run.

  Bar A, against Drake's SNOPT defaults -- SNOPTTUNE's bar unchanged: on the LEARNED arm,
    better on >= 9 of the 12 rows, significantly worse (p < 0.05) on none, significantly
    better on >= 1. This answers "is this a valid SNOPT configuration at all".
  Bar B, against `Major step limit = 0.5` alone, in the same run: better on >= 8 of the 12
    rows and significantly worse on none. This answers the question the stage exists for --
    "does adding the second factor buy anything the survivor did not already".

A combination is recommended over the survivor alone only if it clears BOTH. Clearing A alone
means "valid, but no better than the step limit", which is not a reason to field a more
complicated configuration.

Both bars are counted by ROW and never pooled (Thomas, 2026-09-17: "Do not pool experiments,
that is useless"), because a pooled win can be a trade between robots and a setting that helps
one robot and hurts the other cannot be adopted by picking the robot it helps. The rule is
implemented inline here rather than imported, so it cannot drift when read by a later session.

Settings are named by their OPTIONS, never by the manifest token (Thomas: an internal token
"is an abbreviation. Don't do that.").

Reads merged 480-cell summaries out of results/_cluster_staging/*/ as well as the promoted
tree, because a run in flight is only ever staged -- a glob over the promoted location alone
silently matches nothing and any check written against it passes vacuously.

Usage:
    scripts/report_snoptcombo.py                    # every candidate found
    scripts/report_snoptcombo.py mstepelastic ...   # only these
"""
import glob
import json
import os
import sys
from math import comb

## Manifest token -> what the options actually are. Anything not here prints its token and a
## warning, so a new table entry cannot quietly appear as an abbreviation.
OPTION = {
    "default": "SNOPT defaults (baseline)",
    "mstep0p5": "Major step limit = 0.5",
    "elastic1e2": "Elastic weight = 100",
    "mstepelastic": "Major step limit = 0.5 + Elastic weight = 100",
    "mstephess20": "Major step limit = 0.5 + Hessian frequency = 20",
    "mstepmajopt": "Major step limit = 0.5 + Major optimality tolerance = 1e-8",
    "mstepelastichess": ("Major step limit = 0.5 + Elastic weight = 100 "
                         "+ Hessian frequency = 20"),
    "mstepelastichessmajopt": ("Major step limit = 0.5 + Elastic weight = 100 "
                               "+ Hessian frequency = 20 "
                               "+ Major optimality tolerance = 1e-8"),
}
## The reference for Bar B: the one setting that passed SNOPTTUNE's bar.
SURVIVOR = "mstep0p5"
ROW_ORDER = {"mugfree": 0, "mugshelf": 1, "posetip": 2}
ROW_NAME = {"mugfree": "grasp free", "mugshelf": "grasp contained", "posetip": "pose tip"}
TAG = "sc_SNOPTCOMBO_"


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


def load_complete(cells=480):
    """tag -> summary, for every MERGED run of the right size. Per-shard dirs are skipped."""
    out = {}
    for pat in (f"results/_cluster_staging/*/results/*/benchmark/{TAG}*/summary.json",
                f"results/*/benchmark/{TAG}*/summary.json"):
        for f in sorted(glob.glob(pat)):
            tag = os.path.basename(os.path.dirname(f))
            if "_shard" in tag:
                continue
            with open(f) as fh:
                s = json.load(fh)
            if len(s["records"].get("learned", [])) != cells:
                continue
            out[tag] = s
    return out


def compare(base, cand, arm="learned"):
    """McNemar of `cand` against `base` on the cells they share, for one arm."""
    A = {(r["target"], r["guess"]): r for r in base["records"][arm]}
    B = {(r["target"], r["guess"]): r for r in cand["records"][arm]}
    shared = [k for k in A if k in B]
    b = sum(1 for k in shared if B[k]["feasible"] and not A[k]["feasible"])
    w = sum(1 for k in shared if A[k]["feasible"] and not B[k]["feasible"])
    return (sum(A[k]["feasible"] for k in shared), sum(B[k]["feasible"] for k in shared),
            b, w, mcnemar(b, w), A, B)


def both_solved_cost(A, B):
    """Median cost on cells BOTH columns solved.

    Not `summary["_common"]`, which is cells both ARMS solved within ONE run: comparing one
    column's median over its own successes to another's is the banned form, because the easy
    cells are exactly the ones a weaker column also solves.
    """
    shared = [k for k in A if k in B and A[k]["feasible"] and B[k]["feasible"]]
    return (median([A[k].get("cost") for k in shared]),
            median([B[k].get("cost") for k in shared]), len(shared))


def rows_for(runs, baselines, tok, ref_suffix):
    """One row per (robot, task, protocol), comparing `tok` against `ref_suffix`."""
    rows = []
    for btag, bs in baselines.items():
        ref_tag = btag[: -len("default")] + ref_suffix
        cand_tag = btag[: -len("default")] + tok
        if ref_tag not in runs or cand_tag not in runs:
            continue
        p = btag.split("_")
        nb, nc, b, w, pv, A, B = compare(runs[ref_tag], runs[cand_tag])
        _, jc, jb, jw, jp, JA, JB = compare(runs[ref_tag], runs[cand_tag], "numerical")
        ca, cb, n_both = both_solved_cost(A, B)
        rows.append(dict(
            robot=p[2], task=p[5], start=p[8], base=nb, cand=nc, better=b, worse=w, p=pv,
            js_base=sum(r["feasible"] for r in JA.values()),
            js_cand=sum(r["feasible"] for r in JB.values()), js_p=jp,
            cost_base=ca, cost_cand=cb, n_both=n_both,
            it=median([r["iterations"] for r in B.values() if r["feasible"]]),
            wall=median([r["wall_time"] for r in B.values() if r["feasible"]]),
            to=runs[cand_tag]["summary"]["learned"]["timeouts"]))
    rows.sort(key=lambda r: (r["robot"], ROW_ORDER[r["task"]], r["start"]))
    return rows


def print_table(title, rows):
    print(f"\n  vs {title}")
    print(f"    {'row':<32}{'ref':>5}{'cand':>6}{'+':>5}{'-':>5}{'p':>9}"
          f"{'JSref':>7}{'JScand':>7}{'cost ref':>10}{'cost cand':>10}{'n':>5}"
          f"{'iters':>7}{'wall':>7}{'TO':>5}")
    for r in rows:
        flag = ""
        if r["p"] < 0.05:
            flag = "  BETTER" if r["cand"] > r["base"] else "  WORSE"
        label = f"{r['robot']} {ROW_NAME[r['task']]} {r['start']}"
        print(f"    {label:<32}{r['base']:>5}{r['cand']:>6}{r['better']:>5}{r['worse']:>5}"
              f"{r['p']:>9.3g}{r['js_base']:>7}{r['js_cand']:>7}{r['cost_base']:>10.3f}"
              f"{r['cost_cand']:>10.3f}{r['n_both']:>5}{r['it']:>7.0f}{r['wall']:>7.2f}"
              f"{r['to']:>5}{flag}")


def counts(rows):
    return (sum(1 for r in rows if r["cand"] > r["base"]),
            sum(1 for r in rows if r["cand"] < r["base"]),
            sum(1 for r in rows if r["p"] < 0.05 and r["cand"] > r["base"]),
            sum(1 for r in rows if r["p"] < 0.05 and r["cand"] < r["base"]))


def main(only):
    runs = load_complete()
    baselines = {t: s for t, s in runs.items() if t.endswith("_default")}
    if not baselines:
        print(f"no merged 480-cell {TAG}*_default run found -- nothing to pair against")
        return
    tokens = sorted({t.rsplit("_", 1)[-1] for t in runs} - {"default"})
    if only:
        tokens = [t for t in tokens if t in only]
    print(f"stage SNOPTCOMBO: {len(baselines)} baseline row(s) of 12, "
          f"{len(tokens)} candidate column(s)")
    print("the joint-space (numerical) arm is shown beside the learned one on every row: "
          "these are SOLVER options, so it moves too, and a setting that helps both arms is a "
          "property of the problem rather than of the chart's conditioning")
    for tok in tokens:
        name = OPTION.get(tok)
        if name is None:
            name = f"{tok} (UNNAMED -- add it to OPTION)"
        a_rows = rows_for(runs, baselines, tok, "default")
        b_rows = rows_for(runs, baselines, tok, SURVIVOR) if tok != SURVIVOR else []
        if not a_rows:
            continue
        print(f"\n=== {name}   [{len(a_rows)} of 12 rows]")
        print_table(f"SNOPT defaults  (Bar A: >= 9 of 12 better, 0 significantly worse, "
                    f">= 1 significantly better)", a_rows)
        nb, nw, sb, sw = counts(a_rows)
        a_pass = len(a_rows) == 12 and nb >= 9 and sw == 0 and sb >= 1
        print(f"    Bar A: rows better {nb}, worse {nw}; significantly better {sb}, worse "
              f"{sw}  ->  {'PASSES' if a_pass else 'FAILS'}"
              + ("" if len(a_rows) == 12 else f"  (INCOMPLETE, {len(a_rows)}/12 rows)"))
        if tok == SURVIVOR:
            print("    Bar B: not applicable -- this IS the reference for Bar B")
            continue
        if not b_rows:
            print(f"    Bar B: no {SURVIVOR!r} column found in the same run(s)")
            continue
        print_table(f"{OPTION[SURVIVOR]}  (Bar B: >= 8 of 12 better, 0 significantly worse)",
                    b_rows)
        nb2, nw2, sb2, sw2 = counts(b_rows)
        b_pass = len(b_rows) == 12 and nb2 >= 8 and sw2 == 0
        print(f"    Bar B: rows better {nb2}, worse {nw2}; significantly better {sb2}, worse "
              f"{sw2}  ->  {'PASSES' if b_pass else 'FAILS'}"
              + ("" if len(b_rows) == 12 else f"  (INCOMPLETE, {len(b_rows)}/12 rows)"))
        if a_pass and b_pass:
            verdict = f"RECOMMENDED over {OPTION[SURVIVOR]} (both bars)"
        elif a_pass:
            verdict = (f"valid SNOPT configuration, but NO BETTER than "
                       f"{OPTION[SURVIVOR]} alone -- do not field the extra factor")
        else:
            verdict = "REJECTED"
        print(f"    => {verdict}")
    print("\nNothing is adopted by this script. Fielding a SNOPT configuration is Thomas's "
          "call: changing it breaks comparability with every archived SNOPT column.")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
