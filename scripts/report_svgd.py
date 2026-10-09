#!/usr/bin/env python3
"""Read the svgd smoke (and later stage SVGD): the fourth method class against the record.

    .venv/bin/python scripts/report_svgd.py                                  # the smoke
    .venv/bin/python scripts/report_svgd.py --prefix sc_SVGD --cells 480 --cap 180
    .venv/bin/python scripts/report_svgd.py --rows panda,posetip --profile results/profiling/svgd_step_<host>_<t>.json

The matrix (robots x rows x starts x columns) and the tag scheme are IMPORTED from
`scripts/svgd/smoke.py`, so the reporter and the driver cannot disagree about what a tag means.
The rules are the project's (CLAUDE.md, "Every result is told in success, iterations, cost and
wall clock") and the pre-registration's (`docs/svgd-solver.md`):

  - **every row prints**, every variant as a block with learned and joint space ADJACENT, and a
    missing run prints as MISSING with its tag -- never silently dropped. `*x*` marks the better
    arm of each pair (both on a McNemar tie).
  - success with the exact McNemar **learned vs joint space within the variant**, and per arm
    **under svgd vs under IPOPT on the same cells** (the same-machine IPOPT twin), and against the
    record where that pairing is legitimate. The record ran at 180 s on SuperCloud V100s: its
    SUCCESSES pair, its seconds never do.
  - every pairing checks `grid_hash` (plus task / start / scene / placement / inset, which the
    hash does not cover -- the Panda's grasp and pose grids hash identically) and REFUSES on a
    mismatch, loudly, with a nonzero exit. A learned pairing also needs the same chart; the iiwa
    smoke runs n6 against the record's n4, so that pairing is skipped by design and says so.
  - cost only on cells BOTH arms solved (`record["cost"]` is already the reported cost, the
    learned-only regularizers excluded), `N/A` under 10 shared cells.
  - svgd steps are OUTER steps of a population method and are NOT comparable to IPOPT majors;
    the column header says which one a block carries.
  - the cap check reads BOTH `timed_out` AND `hit_iteration_cap`; the ">= 24 of 480 cells at a
    budget carries no verdict" rule is scaled to the cell count (>= 5%, at least one cell).
  - from `record["svgd"]`: feasible particles, resampled particles (as a fraction of N),
    `selected_index`, the collision pool's share of the wall, `stop_reason` counts, warm-up and
    compile seconds -- and `solver_feasible` vs `drake_feasible`, and `drake_feasible` vs
    `verify()`'s verdict. Any disagreement is a BUG and prints as one.

Then a CEM A/B section, an N-ladder section, and the go/no-go rules (1)-(7) evaluated
mechanically where the summaries can decide them. Exit status: 0 clean (missing runs are not
fatal), 2 if any pairing was refused, 3 if any bug line printed.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts", "svgd")):
    if _p not in sys.path:
        sys.path.append(_p)

import smoke as M                                                         # noqa: E402
from report_statusquo import by_cell, mean, median, verdict               # noqa: E402
from src.benchmark import mcnemar_exact                                   # noqa: E402

MIN_COST_CELLS = 10
BUDGET_FRACTION = 24 / 480
ARM_NAME = {"learned": "learned", "numerical": "joint space"}
ARMS = ("learned", "numerical")
## Metadata a matching grid_hash does NOT guarantee (see `scripts/collate.py --pair`).
MUST_MATCH = ("task", "start", "scene", "target_placement", "shelf_inset")


class Report:
    """What the run found, beyond the printed tables: drives the exit status and the go/no-go."""

    def __init__(self):
        self.missing, self.refused, self.bugs = [], [], []
        self.blocks = {}          # (robot, row, start, column) -> {"L": stats, "J": stats, ...}

    def bug(self, msg):
        self.bugs.append(msg)
        print(f"  !!!!!! BUG: {msg}")

    def refuse(self, msg):
        self.refused.append(msg)
        print(f"  !!!!!! REFUSED: {msg}")


## ------------------------------------------------------------------- loading --
def find_summary(root, tag):
    """`<root>/<robot>/benchmark/<tag>/summary.json`, staged trees too (a run in flight is only
    ever staged). Per-shard directories are never read: only merged runs are."""
    for pattern in (f"{root}/*/benchmark/{tag}/summary.json",
                    f"{root}/_cluster_staging/*/results/*/benchmark/{tag}/summary.json"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def load(root, tag):
    path = find_summary(root, tag)
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def load_record(root, robot, row, start, stage):
    for tag in M.record_tags(robot, row, start, stage):
        s = load(root, tag)
        if s is not None:
            return tag, s
    return M.record_tags(robot, row, start, stage)[-1], None


## ------------------------------------------------------------------ pairing --
def incomparable(a, b):
    """Why runs `a` and `b` must not be paired, or None. (A learned pairing ALSO needs the same
    chart -- `same_chart` -- because two charts are two formulations of the arm.)"""
    ma, mb = a.get("metadata", {}), b.get("metadata", {})
    ha, hb = ma.get("grid_hash"), mb.get("grid_hash")
    if ha is None or hb is None:
        return "no grid_hash on one side -- provenance unknown"
    if ha != hb:
        return f"grid_hash {ha} != {hb}"
    differs = [k for k in MUST_MATCH if ma.get(k) != mb.get(k)]
    if differs:
        return "metadata differs: " + ", ".join(f"{k} {ma.get(k)!r} != {mb.get(k)!r}"
                                                for k in differs)
    return None


def same_chart(a, b):
    ca = os.path.basename(a.get("metadata", {}).get("checkpoint") or "")
    cb = os.path.basename(b.get("metadata", {}).get("checkpoint") or "")
    return ca == cb


def mcnemar_pair(sa, arm_a, sb, arm_b):
    """Exact McNemar of `arm_a` in run `sa` against `arm_b` in run `sb`, on the cells BOTH ran.
    `a_only` counts cells only A solved: the direction is always (first, second)."""
    A, B = by_cell(sa, arm_a), by_cell(sb, arm_b)
    shared = sorted(k for k in A if k in B)
    a = [bool(A[k].get("feasible")) for k in shared]
    b = [bool(B[k].get("feasible")) for k in shared]
    m = mcnemar_exact(a, b)
    return dict(n=len(shared), a_succ=sum(a), b_succ=sum(b), **m)


## -------------------------------------------------------------------- stats --
def budget_threshold(n):
    return max(1, math.ceil(BUDGET_FRACTION * n))


def ms_per_step(r):
    """Per OUTER step for svgd (the swarm phase over outer steps -- what `profile_step` times),
    per major for a Drake solver (wall over iterations)."""
    it = r.get("iterations")
    if not it:
        return None
    sv = r.get("svgd") or {}
    swarm = (sv.get("phase_times") or {}).get("swarm")
    seconds = swarm if swarm is not None else r.get("wall_time")
    return None if seconds is None else 1e3 * seconds / it


def ms_per_inner(r):
    """Per INNER step for svgd: the swarm phase over `inner_steps` (one fused step -- the
    quantity `profile_step` times as `ms_per_inner`). An outer step holds `svgd_inner_iters`
    of them, so comparing the per-outer figure against the profiler is a units error."""
    sv = r.get("svgd") or {}
    inner = sv.get("inner_steps")
    swarm = (sv.get("phase_times") or {}).get("swarm")
    if not inner or swarm is None:
        return None
    return 1e3 * swarm / inner


def arm_stats(summary, arm, other):
    A, B = by_cell(summary, arm), by_cell(summary, other)
    recs = list(A.values())
    ok = [r for r in recs if r.get("feasible")]
    both = [k for k in A if k in B and A[k].get("feasible") and B[k].get("feasible")]
    cap = summary.get("metadata", {}).get("wall_time")
    svgd = [r.get("svgd") for r in recs if r.get("svgd")]
    out = dict(
        n=len(recs), succ=len(ok),
        to=sum(1 for r in recs if r.get("timed_out")),
        icap=sum(1 for r in recs if r.get("hit_iteration_cap")),
        errors=sum(1 for r in recs if r.get("fail_reason") == "error"),
        recovered=sum(1 for r in recs if "recovered_feasible" in r),
        viol=median([r.get("max_violation") for r in recs]),
        cost=median([A[k].get("cost") for k in both]) if len(both) >= MIN_COST_CELLS else None,
        n_both=len(both),
        steps=median([r.get("iterations") for r in ok]),
        ms_step=median([ms_per_step(r) for r in recs]),
        ms_inner=median([ms_per_inner(r) for r in recs]),
        wall=mean([r.get("wall_time") for r in recs]),
        wall_max=max([r.get("wall_time") or 0.0 for r in recs], default=None),
        over_cap=(sum(1 for r in recs if (r.get("wall_time") or 0.0) > cap + 1.0)
                  if cap is not None else 0),
        is_svgd=bool(svgd),
    )
    if svgd:
        frac = [sv["n_resampled"] / sv["n_particles"] for sv in svgd
                if sv.get("n_resampled") is not None and sv.get("n_particles")]
        coll = [(r["svgd"].get("collision_seconds") or 0.0) / r["wall_time"]
                for r in recs if r.get("svgd") and r.get("wall_time")]
        out.update(
            n_particles=median([sv.get("n_particles") for sv in svgd]),
            n_feasible=median([sv.get("n_feasible") for sv in svgd]),
            resampled_median=median(frac), resampled_max=max(frac, default=None),
            selected_median=median([sv.get("selected_index") for sv in svgd]),
            selected_none=sum(1 for sv in svgd if sv.get("selected_index") == -1),
            coll_share=median(coll),
            stop=Counter(sv.get("stop_reason") or "?" for sv in svgd),
            solver_vs_drake=[(r["target"], r["guess"]) for r in recs if r.get("svgd")
                             and bool(r["svgd"].get("solver_feasible"))
                             != bool(r["svgd"].get("drake_feasible"))],
            drake_vs_verify=[(r["target"], r["guess"]) for r in recs if r.get("svgd")
                             and r.get("fail_reason") != "error"
                             and bool(r["svgd"].get("drake_feasible"))
                             != bool(r.get("feasible"))],
            compile_record=median([sv.get("compile_seconds") for sv in svgd]),
            warmup=(summary.get("metadata", {}).get("svgd_warmup_seconds") or {}).get(arm),
        )
    return out


def star(pair, lower_better=True, tie=False):
    """Format a (learned, joint) pair, `*`-marking the better. `tie` marks both."""
    l, j = pair
    if l is None or j is None or (isinstance(l, float) and l != l) or (isinstance(j, float) and j != j):
        return pair
    if tie or l == j:
        return ("*", "*")
    return ("*", "") if (l < j) == lower_better else ("", "*")


def fmt(x, w, prec=0, mark=""):
    if x is None or (isinstance(x, float) and x != x):
        s = "--"
    else:
        s = f"{x:.{prec}f}" if isinstance(x, (int, float)) else str(x)
    s = f"{mark}{s}{mark}" if mark and s != "--" else s
    return f"{s:>{w}}"


def fmt_e(x, w, mark=""):
    s = "--" if x is None or (isinstance(x, float) and x != x) else f"{x:.1e}"
    s = f"{mark}{s}{mark}" if mark and s != "--" else s
    return f"{s:>{w}}"


def pair_str(m):
    if m is None:
        return "--"
    return f"+{m['a_only']}/-{m['b_only']} p={m['p']:.2g}"


## ------------------------------------------------------------------ the rows --
def print_row(rep, root, robot, row, start, columns, n_cells, cap, prefix, variant_kw, stage,
              use_void_record=False):
    title = f"{robot} {M.ROW_NAME[row]}, {start}"
    print(f"\n{'=' * 100}\n{title}   ({n_cells} cells, cap {cap:g} s, rung {M.ROBOTS[robot]['rung']})")
    twin_tag = M.make_tag(robot, row, start, M.IPOPT, n_cells, cap, prefix, **variant_kw)
    twin = load(root, twin_tag)
    rec_tag, rec = load_record(root, robot, row, start, stage)
    void = (rec is not None and row == "mugshelf"
            and M.ROBOTS[robot].get("record_grasp_void") and not use_void_record)
    if void:
        print(f"  record:     {rec_tag} -- NOT PAIRED: the record's wsg-gripper grasp rows predate the "
              f"mug-handle fix (452d784) and are void; --use-void-record overrides")
        rec = None
    print(f"  IPOPT twin: {twin_tag}{'' if twin else '  (MISSING)'}")
    if not void:
        print(f"  record:     {rec_tag}{'' if rec else '  (MISSING)'}"
              f"{'' if rec is None else ' -- 180 s on SuperCloud V100s: its successes pair, its seconds never do'}")
    thr = budget_threshold(n_cells)
    print(f"  cap rule: >= {thr} of {n_cells} cells at the ITERATION budget -> no verdict; "
          f">= {thr} at the clock -> a result at the fielded clock, flagged")
    hdr = (f"    {'arm':<12}{'succ':>8}{'vs IPOPT twin':>20}{'vs record':>20}{'viol':>10}"
           f"{'cost':>9}{'n':>4}{'steps':>8}{'ms/step':>9}{'wall':>8}{'max':>7}{'t/out':>6}{'i/cap':>6}")
    for column in columns:
        tag = M.make_tag(robot, row, start, column, n_cells, cap, prefix, **variant_kw)
        token = M.variant_token(column, **variant_kw)
        print(f"\n  [{column}] {token}")
        s = twin if column == M.IPOPT else load(root, tag)
        if s is None:
            rep.missing.append(tag)
            print(f"    MISSING: {tag}")
            continue
        L, J = arm_stats(s, "learned", "numerical"), arm_stats(s, "numerical", "learned")
        lj = mcnemar_pair(s, "learned", s, "numerical")
        budget_bound = max(L["icap"], J["icap"]) >= thr
        clock_bound = max(L["to"], J["to"]) >= thr
        v = "NO VERDICT (iteration-budget-bound)" if budget_bound else verdict(L["succ"], J["succ"], lj["p"])
        print(f"    learned vs joint space: {L['succ']} v {J['succ']} of {L['n']}, "
              f"L-only {lj['a_only']} / JS-only {lj['b_only']}, p = {lj['p']:.3g} -> {v}"
              + ("   [clock-bound: a result at the fielded clock]" if clock_bound else ""))
        ## Pairings, per arm: against the IPOPT twin (same machine, cap and chart) and the record.
        vs_twin, vs_rec = {}, {}
        why_twin = incomparable(s, twin) if column != M.IPOPT and twin is not None else None
        why_rec = incomparable(s, rec) if rec is not None else None
        if why_twin:
            rep.refuse(f"{tag} vs IPOPT twin {twin_tag}: {why_twin}")
        if why_rec:
            rep.refuse(f"{tag} vs record {rec_tag}: {why_rec}")
        for arm in ARMS:
            if column != M.IPOPT and twin is not None and not why_twin:
                vs_twin[arm] = mcnemar_pair(s, arm, twin, arm)
            if rec is not None and not why_rec:
                ## A learned pairing needs the same chart: two charts are two formulations.
                vs_rec[arm] = ("chart" if arm == "learned" and not same_chart(s, rec)
                               else mcnemar_pair(s, arm, rec, arm))
        steps_label = "steps = svgd OUTER steps, NOT comparable to IPOPT majors" if L["is_svgd"] or J["is_svgd"] \
            else "steps = IPOPT majors"
        print(f"    ({steps_label}; viol = median max_violation over all cells; cost = median "
              f"reported cost on the n cells both arms solved; wall = mean over all cells)")
        print(hdr)
        tie = lj["p"] >= 0.05
        ms = dict(succ=star((L["succ"], J["succ"]), lower_better=False, tie=tie),
                  viol=star((L["viol"], J["viol"])), cost=star((L["cost"], J["cost"])),
                  steps=star((L["steps"], J["steps"])), wall=star((L["wall"], J["wall"])))
        for i, (arm, st) in enumerate((("learned", L), ("numerical", J))):
            mk = {k: (m[i] if isinstance(m[0], str) else "") for k, m in ms.items()}
            rv = vs_rec.get(arm)
            rec_s = ("chart differs" if rv == "chart" else pair_str(rv)) if rec is not None else "--"
            succ = f"{mk['succ']}{st['succ']}{mk['succ']}/{st['n']}"
            print(f"    {ARM_NAME[arm]:<12}{succ:>8}{pair_str(vs_twin.get(arm)):>20}{rec_s:>20}"
                  f"{fmt_e(st['viol'], 10, mk['viol'])}{fmt(st['cost'], 9, 3, mk['cost']) if st['n_both'] >= MIN_COST_CELLS else f"{'N/A':>9}"}"
                  f"{st['n_both']:>4}{fmt(st['steps'], 8, 0, mk['steps'])}{fmt(st['ms_step'], 9, 1)}"
                  f"{fmt(st['wall'], 8, 2, mk['wall'])}{fmt(st['wall_max'], 7, 1)}{st['to']:>6}{st['icap']:>6}")
        for arm, st in (("learned", L), ("numerical", J)):
            if st["errors"]:
                rep.bug(f"{tag} {ARM_NAME[arm]}: {st['errors']} cell(s) with fail_reason 'error'")
            if st["over_cap"]:
                rep.bug(f"{tag} {ARM_NAME[arm]}: {st['over_cap']} cell(s) over cap + 1 s "
                        f"(max {st['wall_max']:.1f} s)")
            if not st["is_svgd"]:
                continue
            print(f"    svgd {ARM_NAME[arm]:<12} N {fmt(st['n_particles'], 0)}  feasible particles "
                  f"(median) {fmt(st['n_feasible'], 0, 1)}  resampled/N median "
                  f"{fmt(st['resampled_median'], 0, 3)} max {fmt(st['resampled_max'], 0, 3)}  "
                  f"selected idx median {fmt(st['selected_median'], 0, 0)} (none: {st['selected_none']})  "
                  f"collision share of wall {fmt(st['coll_share'], 0, 2)}")
            print(f"    {'':<17} stop {dict(st['stop'])}  warm-up {fmt(st['warmup'], 0, 1)} s  "
                  f"compile (per record) {fmt(st['compile_record'], 0, 1)} s  "
                  f"recovered_* on {st['recovered']} cell(s)")
        for arm, st in (("learned", L), ("numerical", J)):
            ## Any disagreement between the solver's own verdict, its exact Drake re-check and
            ## the harness's verify() is a bug in the solver or the harness, never a result.
            if st.get("solver_vs_drake"):
                rep.bug(f"{tag} {ARM_NAME[arm]}: solver_feasible != drake_feasible on "
                        f"{len(st['solver_vs_drake'])} cell(s) {st['solver_vs_drake']}")
            if st.get("drake_vs_verify"):
                rep.bug(f"{tag} {ARM_NAME[arm]}: drake_feasible != verify() verdict on "
                        f"{len(st['drake_vs_verify'])} cell(s) {st['drake_vs_verify']}")
        meta = s.get("metadata", {})
        if meta.get("compile_seconds") is not None and column == M.IPOPT:
            print(f"    flow compile (script) {meta['compile_seconds']:.1f} s")
        rep.blocks[(robot, row, start, column)] = dict(
            L=L, J=J, lj=lj, run=s, vs_twin=vs_twin, vs_rec=vs_rec, rec=rec, twin=twin)


## ---------------------------------------------------------- CEM A/B, N ladder --
def cem_ab(rep, matrix, columns):
    print(f"\n{'=' * 100}\nCEM warm-up A/B: svgd_warmup 'cem' against 'none', same method / N / "
          f"kernel / cells; exact McNemar per arm (cem+ = cells only the CEM run solved)")
    print(f"  {'row':<34}{'column':<10}{'arm':<13}{'none':>6}{'cem':>6}{'cem+':>6}{'none+':>6}"
          f"{'p':>9}{'steps none':>11}{'steps cem':>10}{'wall none':>10}{'wall cem':>9}")
    for base in M.SVGD_BASES:
        if f"{base}-none" not in columns and f"{base}-cem" not in columns:
            continue
        for robot, row, start in matrix:
            label = f"{robot} {M.ROW_NAME[row]} {start}"
            a = rep.blocks.get((robot, row, start, f"{base}-none"))
            b = rep.blocks.get((robot, row, start, f"{base}-cem"))
            for arm, key in (("learned", "L"), ("numerical", "J")):
                if a is None or b is None:
                    print(f"  {label:<34}{base:<10}{ARM_NAME[arm]:<13}  missing "
                          f"({'none' if a is None else ''}{' ' if a is None and b is None else ''}"
                          f"{'cem' if b is None else ''})")
                    continue
                why = incomparable(a["run"], b["run"])
                if why:
                    rep.refuse(f"CEM A/B {label} {base}: {why}")
                    continue
                m = mcnemar_pair(b["run"], arm, a["run"], arm)
                print(f"  {label:<34}{base:<10}{ARM_NAME[arm]:<13}{a[key]['succ']:>6}{b[key]['succ']:>6}"
                      f"{m['a_only']:>6}{m['b_only']:>6}{m['p']:>9.3g}{fmt(a[key]['steps'], 11)}"
                      f"{fmt(b[key]['steps'], 10)}{fmt(a[key]['wall'], 10, 2)}{fmt(b[key]['wall'], 9, 2)}")


def n_ladder(rep, matrix, columns):
    print(f"\n{'=' * 100}\nN ladder: al_svgd at N = 1 (kernel q, the single-particle control), "
          f"N = 64 kernel none (no interaction), N = 64 kernel q; McNemar N=64 q vs N=1")
    print(f"  {'row':<34}{'warmup':<8}{'arm':<13}{'N=1':>6}{'64 none':>9}{'64 q':>6}"
          f"{'64q+':>6}{'1+':>5}{'p':>9}  rule (7)")
    for w in M.WARMUPS:
        for robot, row, start in matrix:
            label = f"{robot} {M.ROW_NAME[row]} {start}"
            b1 = rep.blocks.get((robot, row, start, f"al1-{w}"))
            bn = rep.blocks.get((robot, row, start, f"al64none-{w}"))
            bq = rep.blocks.get((robot, row, start, f"al64-{w}"))
            for arm, key in (("learned", "L"), ("numerical", "J")):
                def succ(b):
                    return f"{b[key]['succ']}" if b else "--"
                m, flag = None, ""
                if b1 and bq:
                    why = incomparable(bq["run"], b1["run"])
                    if why:
                        rep.refuse(f"N ladder {label} {w}: {why}")
                    else:
                        m = mcnemar_pair(bq["run"], arm, b1["run"], arm)
                        flag = "FAILS: N=1 beats N=64" if b1[key]["succ"] > bq[key]["succ"] else "ok"
                else:
                    flag = "missing"
                print(f"  {label:<34}{w:<8}{ARM_NAME[arm]:<13}{succ(b1):>6}{succ(bn):>9}{succ(bq):>6}"
                      f"{(m['a_only'] if m else '--'):>6}{(m['b_only'] if m else '--'):>5}"
                      f"{(format(m['p'], '.3g') if m else '--'):>9}  {flag}")


## ---------------------------------------------------------------- go / no-go --
def load_profile(path):
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def profile_ms(profile, arm, method, n, dtype, mode):
    if profile is None:
        return None
    for r in profile.get("rows", []):
        if (r.get("arm") == arm and r.get("method") == method and r.get("N") == n
                and r.get("dtype") == dtype and r.get("overlap", True)
                and r.get("mode") == (mode if method != "admm_svgd" else "eager")
                and "ms_per_step" in r):
            ## Per INNER step: the profiler's own `ms_per_inner`, else its per-step figure
            ## over the inner steps each of its steps held (`inner_per_step`).
            if r.get("ms_per_inner") is not None:
                return r["ms_per_inner"]
            return r["ms_per_step"] / float(r.get("inner_per_step") or 1)
    return None


def go_no_go(rep, matrix, columns, variant_kw, profile):
    print(f"\n{'=' * 100}\nGo / no-go (docs/svgd-solver.md), evaluated where the summaries decide it")
    svgd_blocks = {k: b for k, b in rep.blocks.items() if k[3] != M.IPOPT}
    errors = sum(b["L"]["errors"] + b["J"]["errors"] for b in svgd_blocks.values())
    print(f"  (1) error records across svgd runs: {errors}; runs missing (a dead-arm abort writes no "
          f"summary): {len(rep.missing)}; tests: run them -- not decidable here")
    hits = []
    for (robot, row, start, column), b in svgd_blocks.items():
        if b["J"]["succ"] == 0:
            continue
        rv, tv = b["vs_rec"].get("learned"), b["vs_twin"].get("learned")
        if isinstance(rv, dict):
            ref, src = rv["b_succ"], "the record's IPOPT learned count"
        elif isinstance(tv, dict):
            why = "record chart differs" if rv == "chart" else "record not paired"
            ref, src = tv["b_succ"], f"the IPOPT twin's learned count; {why}"
        else:
            continue
        if b["L"]["succ"] >= ref:
            hits.append(f"{robot} {row} {start} {column}: learned {b['L']['succ']} >= {ref} ({src})")
    print(f"  (2) rows where a variant's learned successes >= IPOPT's learned count on those cells, "
          f"joint space not dead: {len(hits)} -> {'PASS' if hits else 'FAIL'}")
    for h in hits:
        print(f"        {h}")
    dis = sum(len(b[k].get("solver_vs_drake", [])) + len(b[k].get("drake_vs_verify", []))
              for b in svgd_blocks.values() for k in ("L", "J"))
    print(f"  (3) feasibility disagreements (solver vs Drake re-check vs verify()): {dis} -> "
          f"{'PASS' if dis == 0 and svgd_blocks else 'FAIL' if dis else '--'}")
    over = sum(b[k]["over_cap"] for b in svgd_blocks.values() for k in ("L", "J"))
    rec = sum(b[k]["recovered"] for b in svgd_blocks.values() for k in ("L", "J"))
    print(f"  (4) cells over cap + 1 s: {over} -> {'PASS' if over == 0 and svgd_blocks else 'FAIL' if over else '--'}; "
          f"cells with recovered_*: {rec} (the kill test is a separate run)")
    print("  (5) ms per INNER step against profile_step's ms_per_inner (same arm, method, N, dtype, "
          "mode; Panda only; the table's ms/step is per OUTER step = svgd_inner_iters inner steps):")
    if profile is None:
        print("        no --profile given")
    for (robot, row, start, column), b in sorted(svgd_blocks.items()):
        if profile is None or robot != "panda":
            continue
        s = M.column_settings(column)
        task = M.TASK_ROWS[row][0]
        if profile.get("metadata", {}).get("task") != task:
            continue
        for arm, k in (("learned", "L"), ("numerical", "J")):
            ref = profile_ms(profile, arm, s["svgd_method"], s["svgd_n"], variant_kw["dtype"],
                             variant_kw["mode"])
            got = b[k].get("ms_inner")
            if ref is None or got is None:
                continue
            ratio = got / ref
            print(f"        {robot} {row} {start} {column} {ARM_NAME[arm]}: {got:.2f} against "
                  f"{ref:.2f} ms/inner step = {ratio:.2f}x -> {'PASS' if ratio <= 2.0 else 'FAIL'}")
    worst = max((b[k].get("resampled_max") or 0.0 for b in svgd_blocks.values() for k in ("L", "J")),
                default=None)
    print(f"  (6) worst per-cell resampled / N: {fmt(worst, 0, 3)} -> "
          f"{'--' if worst is None or not svgd_blocks else 'PASS' if worst < 0.10 else 'FAIL'}")
    bad = []
    for w in M.WARMUPS:
        for robot, row, start in matrix:
            b1 = rep.blocks.get((robot, row, start, f"al1-{w}"))
            bq = rep.blocks.get((robot, row, start, f"al64-{w}"))
            if b1 and bq:
                for arm, k in (("learned", "L"), ("numerical", "J")):
                    if b1[k]["succ"] > bq[k]["succ"]:
                        bad.append(f"{robot} {row} {start} warmup {w} {ARM_NAME[arm]}")
    print(f"  (7) N=1 beating N=64 (al_svgd, kernel q): {len(bad)} -> {'PASS' if not bad else 'FAIL'}"
          + (f" {bad}" if bad else ""))
    print("  (8) the selected variant(s) and N are written into docs/svgd-solver.md BEFORE any cluster "
          "manifest is generated -- a manual step")


## ---------------------------------------------------------------------- main --
def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.join(REPO, "results"))
    p.add_argument("--prefix", default=M.PREFIX, help="smoke_SVGD, or sc_SVGD for the stage")
    p.add_argument("--cells", type=int, default=len(M.RECORD_CELLS.split(",")),
                   help="the cell-count token in the tags (10 for the smoke, 480 for the stage)")
    p.add_argument("--cap", type=float, default=20.0, help="the cap token in the tags")
    p.add_argument("--rows", default=None, help="as smoke.py --rows")
    p.add_argument("--columns", default=None, help="as smoke.py --columns")
    p.add_argument("--paired-init", default="jitter")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--mode", default="graphed")
    p.add_argument("--record-stage", default="STATUSQUO")
    p.add_argument("--use-void-record", action="store_true",
                   help="pair wsg-gripper grasp rows against the record anyway (it predates the "
                        "mug-handle fix 452d784; valid only while both runs share that scene)")
    p.add_argument("--profile", default=None, help="a profile_step JSON, for go/no-go rule (5)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    robots, rows, starts = M.parse_row_filter(args.rows)
    columns = M.parse_columns(args.columns)
    if M.IPOPT not in columns:
        columns = [M.IPOPT] + columns            # the twin is every row's reference
    matrix = M.rows_matrix(robots, rows, starts)
    variant_kw = dict(paired_init=args.paired_init, dtype=args.dtype, mode=args.mode)
    rep = Report()
    print(f"svgd report: prefix {args.prefix}, {args.cells} cells, cap {args.cap:g} s, "
          f"{len(matrix)} rows x {len(columns)} columns, root {args.root}")
    for robot, row, start in matrix:
        print_row(rep, args.root, robot, row, start, columns, args.cells, args.cap, args.prefix,
                  variant_kw, args.record_stage, args.use_void_record)
    cem_ab(rep, matrix, columns)
    n_ladder(rep, matrix, columns)
    go_no_go(rep, matrix, columns, variant_kw, load_profile(args.profile))
    print(f"\n{'=' * 100}\nmissing runs: {len(rep.missing)}")
    for t in rep.missing:
        print(f"  {t}")
    if rep.refused:
        print(f"\nREFUSED pairings: {len(rep.refused)}")
        for t in rep.refused:
            print(f"  {t}")
    if rep.bugs:
        print(f"\n!!!!!! {len(rep.bugs)} BUG line(s):")
        for t in rep.bugs:
            print(f"  {t}")
    rep.status = 3 if rep.bugs else 2 if rep.refused else 0
    return rep


if __name__ == "__main__":
    sys.exit(main().status)
