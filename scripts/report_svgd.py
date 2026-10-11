#!/usr/bin/env python3
"""Read the svgd smoke (and later stage SVGD): the fourth method class against the record.

    .venv/bin/python scripts/report_svgd.py                                  # the smoke
    .venv/bin/python scripts/report_svgd.py --stage SVGD                     # the cluster stage
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
  - from `record["svgd"]`: the two POPULATION metrics -- feasible particles at stop and the
    median pairwise distance in q among them (`feasible_q_spread`) -- resampled particles (as a
    fraction of N), `selected_index`, the dual updates and the multiplier magnitudes at stop
    (median and max |lam_i|_inf, |mu_i|_inf) with the clip count, the collision row's share of the wall, `stop_reason` counts, warm-up and compile seconds --
    and `solver_feasible` vs `drake_feasible`, and `drake_feasible` vs `verify()`'s verdict.
    Any disagreement is a BUG and prints as one.

Then an A/B section -- every svgd column against the primary `al64` it differs from in one
setting, per arm, exact McNemar on the same cells -- and the go/no-go rules (1)-(8) evaluated
mechanically where the summaries can decide them. Exit status: 0 clean (missing runs are not
fatal), 2 if any pairing was refused, 3 if any bug line printed.

`--stage SVGD` reads the cluster family `sc_SVGD_*` instead. Its tag scheme is NOT restated here:
every logical run is enumerated from `cluster/gen_manifest.py`'s own builder (`stage_SVGD` over
`SVGD_STAGES`), each tag parsed and the parse checked against the args that builder gave it, and
each variant's settings read off those args -- so a round added there appears here (MISSING until
it lands) without an edit. A run is used only if its `metadata["overrides"]`, solver, start, task,
cap and cell count are what its manifest item ran. The rule is docs/svgd-solver.md's "per-row
analysis rule": the LEAD of every row is learned under svgd against learned under IPOPT (the
stage SVGD_R2 twin, same cells, exact McNemar); joint space under svgd is an ABLATION, printed
beside it with its own McNemar against the joint-space IPOPT twin; learned-vs-joint-space under
svgd is printed but is no headline and no tally is built on it. Every stage pairing also checks
`scene_fingerprint` and the chart. Mean wall is over all cells, each clamped at the cap. Then
every variant against the primary `kq` per arm, and a variant-by-row table across the rounds.

A run carries the arms its manifest item's `--arms` names, and that is checked like the rest. The
rounds R3-R5 run the LEARNED arm only (gen_manifest's SVGD_ROUND_ARMS): their ablation lines and
joint-space cells read `not run` / `--`, their block has no learned-vs-joint-space line, and
their cost column (`cost/tw`) is taken on the cells the run and the IPOPT twin's learned arm both
solved, the twin standing in for the absent arm.
"""
import argparse
import ast
import glob
import json
import math
import os
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts", "svgd"),
           os.path.join(REPO, "cluster")):
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
def incomparable(a, b, must=MUST_MATCH):
    """Why runs `a` and `b` must not be paired, or None. (A learned pairing ALSO needs the same
    chart -- `same_chart` -- because two charts are two formulations of the arm.)"""
    ma, mb = a.get("metadata", {}), b.get("metadata", {})
    ha, hb = ma.get("grid_hash"), mb.get("grid_hash")
    if ha is None or hb is None:
        return "no grid_hash on one side -- provenance unknown"
    if ha != hb:
        return f"grid_hash {ha} != {hb}"
    differs = [k for k in must if ma.get(k) != mb.get(k)]
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


def arm_stats(summary, arm, other, clamp_wall=False):
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
        wall=mean([min(r["wall_time"], cap) if clamp_wall and cap is not None
                   and r.get("wall_time") is not None else r.get("wall_time") for r in recs]),
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
            q_spread=median([sv.get("feasible_q_spread") for sv in svgd]),
            n_dual=median([sv.get("n_dual_updates") for sv in svgd]),
            lam_med=median([sv.get("lam_inf_median") for sv in svgd]),
            lam_max=max([sv.get("lam_inf_max") or 0.0 for sv in svgd], default=None),
            mu_med=median([sv.get("mu_inf_median") for sv in svgd]),
            mu_max=max([sv.get("mu_inf_max") or 0.0 for sv in svgd], default=None),
            mclip=sum(sv.get("n_multiplier_clipped") or 0 for sv in svgd),
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
def print_svgd_detail(rep, tag, L, J):
    """Per arm: error / over-cap BUG lines, the svgd population lines, and the feasibility
    agreement checks (solver vs its Drake re-check vs verify()). An arm the run did not carry
    (stage mode: a learned-only round) is passed as None and skipped."""
    for arm, st in (("learned", L), ("numerical", J)):
        if st is None:
            continue
        if st["errors"]:
            rep.bug(f"{tag} {ARM_NAME[arm]}: {st['errors']} cell(s) with fail_reason 'error'")
        if st["over_cap"]:
            rep.bug(f"{tag} {ARM_NAME[arm]}: {st['over_cap']} cell(s) over cap + 1 s "
                    f"(max {st['wall_max']:.1f} s)")
        if not st["is_svgd"]:
            continue
        print(f"    svgd {ARM_NAME[arm]:<12} N {fmt(st['n_particles'], 0)}  feasible particles "
              f"(median) {fmt(st['n_feasible'], 0, 1)}  q-spread among them (median) "
              f"{fmt(st['q_spread'], 0, 3)}  dual updates (median) {fmt(st['n_dual'], 0, 0)}  "
              f"|lam|_inf at stop median {fmt_e(st['lam_med'], 0)} max {fmt_e(st['lam_max'], 0)}  "
              f"|mu|_inf median {fmt_e(st['mu_med'], 0)} max {fmt_e(st['mu_max'], 0)}  "
              f"multiplier clips {st['mclip']}  resampled/N median "
              f"{fmt(st['resampled_median'], 0, 3)} max {fmt(st['resampled_max'], 0, 3)}  "
              f"selected idx median {fmt(st['selected_median'], 0, 0)} (none: {st['selected_none']})  "
              f"collision share of wall {fmt(st['coll_share'], 0, 2)}")
        print(f"    {'':<17} stop {dict(st['stop'])}  warm-up {fmt(st['warmup'], 0, 1)} s  "
              f"compile (per record) {fmt(st['compile_record'], 0, 1)} s  "
              f"recovered_* on {st['recovered']} cell(s)")
    for arm, st in (("learned", L), ("numerical", J)):
        if st is None:
            continue
        ## Any disagreement between the solver's own verdict, its exact Drake re-check and
        ## the harness's verify() is a bug in the solver or the harness, never a result.
        if st.get("solver_vs_drake"):
            rep.bug(f"{tag} {ARM_NAME[arm]}: solver_feasible != drake_feasible on "
                    f"{len(st['solver_vs_drake'])} cell(s) {st['solver_vs_drake']}")
        if st.get("drake_vs_verify"):
            rep.bug(f"{tag} {ARM_NAME[arm]}: drake_feasible != verify() verdict on "
                    f"{len(st['drake_vs_verify'])} cell(s) {st['drake_vs_verify']}")


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
        print_svgd_detail(rep, tag, L, J)
        meta = s.get("metadata", {})
        if meta.get("compile_seconds") is not None and column == M.IPOPT:
            print(f"    flow compile (script) {meta['compile_seconds']:.1f} s")
        rep.blocks[(robot, row, start, column)] = dict(
            L=L, J=J, lj=lj, run=s, vs_twin=vs_twin, vs_rec=vs_rec, rec=rec, twin=twin)


## ------------------------------------------------------------- A/B vs primary --
def ab_vs_primary(rep, matrix, columns, primary=M.PRIMARY, twin=M.IPOPT, legend=None,
                  must=MUST_MATCH, arms_of=None):
    """`arms_of(column)` -> the arms that column's runs carry (stage mode reads it off the
    manifest); an arm either side did not run prints as `not run`, never as missing."""
    legend = M.variant_token(M.PRIMARY) if legend is None else legend
    print(f"\n{'=' * 100}\nA/B against the primary {primary} ({legend}): each "
          f"column differs from it in ONE setting; exact McNemar per arm on the same cells "
          f"(col+ = cells only that column solved)")
    print(f"  {'row':<34}{'column':<12}{'arm':<13}{primary:>6}{'col':>6}{'col+':>6}{'prim+':>6}"
          f"{'p':>9}{'steps prim':>11}{'steps col':>10}{'wall prim':>10}{'wall col':>9}"
          f"{'feas prim':>10}{'feas col':>9}{'spread prim':>12}{'spread col':>11}")
    for column in columns:
        if column in (twin, primary):
            continue
        for robot, row, start in matrix:
            label = f"{robot} {M.ROW_NAME[row]} {start}"
            a = rep.blocks.get((robot, row, start, primary))
            b = rep.blocks.get((robot, row, start, column))
            for arm, key in (("learned", "L"), ("numerical", "J")):
                if arms_of is not None and arm not in arms_of(column):
                    print(f"  {label:<34}{column:<12}{ARM_NAME[arm]:<13}  not run "
                          f"({arms_only(arms_of(column))})")
                    continue
                if arms_of is not None and arm not in arms_of(primary):
                    print(f"  {label:<34}{column:<12}{ARM_NAME[arm]:<13}  not run by the primary "
                          f"({arms_only(arms_of(primary))})")
                    continue
                if a is None or b is None:
                    print(f"  {label:<34}{column:<12}{ARM_NAME[arm]:<13}  missing "
                          f"({'primary' if a is None else column})")
                    continue
                why = incomparable(a["run"], b["run"], must)
                if why:
                    rep.refuse(f"A/B {label} {column}: {why}")
                    continue
                m = mcnemar_pair(b["run"], arm, a["run"], arm)
                A, B = a[key], b[key]
                print(f"  {label:<34}{column:<12}{ARM_NAME[arm]:<13}{A['succ']:>6}{B['succ']:>6}"
                      f"{m['a_only']:>6}{m['b_only']:>6}{m['p']:>9.3g}{fmt(A['steps'], 11)}"
                      f"{fmt(B['steps'], 10)}{fmt(A['wall'], 10, 2)}{fmt(B['wall'], 9, 2)}"
                      f"{fmt(A.get('n_feasible'), 10, 1)}{fmt(B.get('n_feasible'), 9, 1)}"
                      f"{fmt(A.get('q_spread'), 12, 3)}{fmt(B.get('q_spread'), 11, 3)}")


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
                and r.get("dtype") == dtype
                and r.get("mode") == mode and "ms_per_step" in r):
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
    for robot, row, start in matrix:
        b1 = rep.blocks.get((robot, row, start, "al1"))
        bq = rep.blocks.get((robot, row, start, M.PRIMARY))
        if b1 and bq:
            for arm, k in (("learned", "L"), ("numerical", "J")):
                if b1[k]["succ"] > bq[k]["succ"]:
                    bad.append(f"{robot} {row} {start} {ARM_NAME[arm]}")
    print(f"  (7) N=1 beating N=64 (al_svgd, kernel q): {len(bad)} -> {'PASS' if not bad else 'FAIL'}"
          + (f" {bad}" if bad else ""))
    print("  (8) the selected variant(s) and N are written into docs/svgd-solver.md BEFORE any cluster "
          "manifest is generated -- a manual step")


## ================================================================== stage mode --
## `--stage SVGD`: the cluster family `sc_SVGD_*`. Nothing about its tags is restated here: the
## runs are enumerated from cluster/gen_manifest.py's own builder, each tag parsed and the parse
## checked against the args the builder gave that item, and a variant's settings are read off
## those args. A round added to gen_manifest's SVGD_ROUNDS therefore shows up here -- MISSING
## until it lands -- with no edit.
STAGE_FAMILIES = {"SVGD": "sc_SVGD"}
STAGE_TWIN = M.IPOPT              # the IPOPT twin's variant name; its tags carry no variant token
STAGE_TWIN_STAGE = "SVGD_R2"      # the gen_manifest stage that builds the twin
STAGE_SKIP = ("SVGD_SMOKE",)      # a 4-cell grid at 60 s: it pairs with nothing
STAGE_SOLVERS = ("ipopt", "svgd")
## A matching grid_hash covers none of these, nor a scene edit: never pair across a scene change.
STAGE_MUST_MATCH = MUST_MATCH + ("scene_fingerprint",)
STAGE_VERDICT_MARK = {"svgd": "W", "tie": "T", "IPOPT": "L"}


def arms_only(arms):
    """`learned arm only`: what a run that does not carry every arm ran."""
    names = [ARM_NAME[a] for a in arms]
    return f"{' and '.join(names)} arm{'s' if len(names) > 1 else ''} only"


def _gen_manifest():
    import gen_manifest                                    # cluster/, torch-free and fast
    return gen_manifest


def _flag(args, flag):
    return args[args.index(flag) + 1]


def manifest_sets(args):
    """An item's `--set NAME=VALUE`s, typed as the benchmark scripts' `apply_overrides` types
    them (`literal_eval`, else the string) -- exactly what lands in `metadata["overrides"]`."""
    out = {}
    for i, a in enumerate(args):
        if a == "--set":
            name, _, value = args[i + 1].partition("=")
            try:
                out[name.strip()] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                out[name.strip()] = value
    return out


def parse_stage_tag(tag):
    """`sc_SVGD_<round>_<robot>_<rung>_<solver>_<row>_<cells>_<cap>_<start>[_<variant>]` -> its
    fields. The IPOPT twin carries no variant token (variant `ipopt`); every svgd run ends in
    one. Parsed from the right, as report_statusquo.parse_tag is, so a robot name containing
    underscores cannot shift the fields. Raises ValueError on anything else."""
    p = tag.split("_")
    if len(p) < 10 or p[:2] != ["sc", "SVGD"]:
        raise ValueError(f"not an sc_SVGD stage tag: {tag}")
    variant = None
    if p[-1] not in M.STARTS:
        variant, p = p[-1], p[:-1]
    rung, solver, row, cells, cap, start = p[-6:]
    robot = "_".join(p[3:-6])
    if (not robot or solver not in STAGE_SOLVERS or row not in M.TASK_ROWS
            or (solver == "ipopt") != (variant is None) or not cells.isdigit()):
        raise ValueError(f"not an sc_SVGD stage tag: {tag}")
    return dict(round=p[2], robot=robot, rung=rung, solver=solver, row=row, cells=int(cells),
                cap=float(cap), start=start, variant=variant or STAGE_TWIN)


def stage_catalogue(family="SVGD"):
    """Every logical run of the family as gen_manifest builds it.

    Returns (runs, variants): `runs` maps tag -> its parsed fields plus `manifest` (the
    gen_manifest stage that builds it), `sets` (its --set overrides) and `arms` (its --arms, in
    order); `variants` maps each variant, in manifest order (the twin, then the primary, then the
    rounds), to its round, manifest, sets and arms. Asserts that every tag parses to what its item actually runs, and that a
    variant runs the same settings on every row -- a broken scheme fails here, not in a table."""
    assert family in STAGE_FAMILIES, family
    G = _gen_manifest()
    runs, variants = {}, {}
    for stage in G.SVGD_STAGES:
        if stage in STAGE_SKIP:
            continue
        for it in G.stage_SVGD(stage):
            a = it["args"]
            tag = _flag(a, "--tag")
            t = parse_stage_tag(tag)
            ran = dict(solver=_flag(a, "--solver"), start=_flag(a, "--start"),
                       cap=float(_flag(a, "--wall-time")),
                       cells=int(_flag(a, "--targets")) * int(_flag(a, "--guesses")))
            parsed = {k: t[k] for k in ran}
            assert parsed == ran, f"{tag} parses to {parsed}; its {stage} item runs {ran}"
            assert M.TASK_ROWS[t["row"]][0] == _flag(a, "--task"), f"{tag}: task {_flag(a, '--task')}"
            if stage == STAGE_TWIN_STAGE:
                assert t["variant"] == STAGE_TWIN, f"{tag}: {stage} must be the IPOPT twin"
            elif stage in G.SVGD_R1_SPLIT:
                assert t["variant"] == G.SVGD_R1_SPLIT[stage], f"{tag}: {stage} carries {G.SVGD_R1_SPLIT[stage]}"
            elif stage in G.SVGD_ROUNDS:
                assert t["variant"] in G.SVGD_ROUNDS[stage], f"{tag}: not a variant of {stage}"
            sets = manifest_sets(a)
            arms = tuple(_flag(a, "--arms").split(","))
            assert arms and set(arms) <= set(ARMS), f"{tag}: --arms {_flag(a, '--arms')}"
            info = dict(t, manifest=stage, sets=sets, arms=arms)
            prev = runs.get(tag)                  # shards of one run share its base tag
            assert prev is None or prev == info, f"{tag}: two items disagree about the run"
            runs[tag] = info
            v = variants.setdefault(t["variant"], dict(round=t["round"], manifest=stage, sets=sets,
                                                       arms=arms))
            assert (v["manifest"], v["sets"], v["arms"]) == (stage, sets, arms), \
                f"variant {t['variant']} runs different settings or arms in {v['manifest']} and {stage}"
    return runs, variants


def stage_primary():
    """The primary svgd variant: gen_manifest's SVGD_VARIANTS is in claim order, primary first."""
    return next(iter(_gen_manifest().SVGD_VARIANTS))


def setting_diff(variants, variant, primary):
    """What `variant` sets differently from `primary`, named in full."""
    if variant == primary:
        return "the primary"
    p, s = variants[primary]["sets"], variants[variant]["sets"]
    out = [f"{k}={s[k]!r}" + (f" (primary {p[k]!r})" if k in p else "")
           for k in s if k not in p or p[k] != s[k]]
    out += [f"{k} unset (primary {p[k]!r})" for k in p if k not in s]
    return ", ".join(out) or "nothing (identical to the primary)"


def stage_run_problems(info, s):
    """Why the run on disk is not the run its tag names, or []: its overrides, solver, start,
    task, cap, arms and cell count must be exactly what its manifest item ran. The arms are the
    item's `--arms` (a round runs the learned arm only), read off the records -- the metadata
    does not carry them -- and checked against `metadata["arms"]` too should a run ever record it."""
    m = s.get("metadata", {})
    want = dict(overrides=info["sets"], solver=info["solver"], start=info["start"],
                task=M.TASK_ROWS[info["row"]][0], wall_time=info["cap"])
    bad = [f"{k} {m.get(k)!r} != the manifest's {v!r}" for k, v in want.items() if m.get(k) != v]
    records = s.get("records", {})
    extra = [a for a in records if a not in info["arms"]]
    if extra:
        bad.append(f"carries arm(s) {extra} its manifest item did not run (--arms "
                   f"{','.join(info['arms'])})")
    if "arms" in m and list(m["arms"] if not isinstance(m["arms"], str) else m["arms"].split(",")) \
            != list(info["arms"]):
        bad.append(f"arms {m['arms']!r} != the manifest's {','.join(info['arms'])!r}")
    for arm in info["arms"]:
        n = len(records.get(arm, []))
        if n != info["cells"]:
            bad.append(f"{ARM_NAME[arm]} has {n} of {info['cells']} cells (an incomplete merge?)")
    return bad


def solver_verdict(m):
    """svgd vs IPOPT on the same cells: `a_only` is svgd's, `b_only` IPOPT's."""
    if m["p"] >= 0.05:
        return "tie"
    return "svgd" if m["a_only"] > m["b_only"] else "IPOPT"


def stage_rows(runs, robots=None, rows=None, starts=None):
    """(robot, row, start) in report order -- every row the manifests build, after the filter."""
    keys = {(t["robot"], t["row"], t["start"]) for t in runs.values()}
    order = lambda k: (k[0], list(M.TASK_ROWS).index(k[1]), M.STARTS.index(k[2]))
    return [k for k in sorted(keys, key=order)
            if (not robots or k[0] in robots) and (not rows or k[1] in rows)
            and (not starts or k[2] in starts)]


def cost_on_shared(sa, arm_a, sb, arm_b):
    """Median reported cost of `arm_a` in run `sa` on the cells it AND `arm_b` in run `sb` both
    solved (None under MIN_COST_CELLS such cells), and that cell count."""
    A, B = by_cell(sa, arm_a), by_cell(sb, arm_b)
    both = [k for k in A if k in B and A[k].get("feasible") and B[k].get("feasible")]
    cost = median([A[k].get("cost") for k in both]) if len(both) >= MIN_COST_CELLS else None
    return cost, len(both)


def stage_print_row(rep, root, key, tags, runs, order, primary):
    robot, row, start = key
    infos = [runs[t] for t in tags.values()]
    cells, cap = infos[0]["cells"], infos[0]["cap"]
    assert all((i["cells"], i["cap"]) == (cells, cap) for i in infos), key
    thr = budget_threshold(cells)
    print(f"\n{'=' * 100}\n{robot} {M.ROW_NAME[row]}, {start}   ({cells} cells, cap {cap:g} s, "
          f"rung {infos[0]['rung']})")
    twin_tag = tags.get(STAGE_TWIN)
    print(f"  IPOPT twin ({STAGE_TWIN_STAGE}): {twin_tag}")
    print(f"  cap rule: >= {thr} of {cells} cells at the ITERATION budget (svgd: the step cap) -> "
          f"no verdict; >= {thr} at the clock -> a result at the fielded clock, flagged")
    ## Load, and use a run only if it is the run its tag names.
    loaded, status = {}, {}
    for v in order:
        tag = tags.get(v)
        if tag is None:
            continue
        s = load(root, tag)
        if s is None:
            rep.missing.append(tag)
            loaded[v], status[v] = None, "MISSING"
            continue
        bad = stage_run_problems(runs[tag], s)
        if bad:
            rep.refuse(f"{tag} is not the run its tag names: " + "; ".join(bad))
            loaded[v], status[v] = None, "REFUSED"
            continue
        loaded[v], status[v] = s, "ok"
    twin = loaded.get(STAGE_TWIN)
    ## An arm the manifest item did not run (a round: learned only) has no stats -- None.
    arms_of = lambda v: runs[tags[v]]["arms"]
    stats = {v: tuple(arm_stats(s, arm, other, clamp_wall=True) if arm in arms_of(v) else None
                      for arm, other in (("learned", "numerical"), ("numerical", "learned")))
             for v, s in loaded.items() if s is not None}
    vs_twin = {}
    for v, s in loaded.items():
        if v == STAGE_TWIN or s is None or twin is None:
            continue
        why = incomparable(s, twin, STAGE_MUST_MATCH)
        if why is None and not same_chart(s, twin):
            why = "checkpoint differs -- two charts are two formulations"
        if why:
            rep.refuse(f"{tags[v]} vs IPOPT twin {twin_tag}: {why}")
            continue
        vs_twin[v] = {arm: mcnemar_pair(s, arm, twin, arm) for arm in ARMS
                      if arm in arms_of(v) and arm in arms_of(STAGE_TWIN)}
    ## The LEAD (learned) and the ABLATION (joint space): each arm under svgd against the same
    ## arm under IPOPT, on the same cells.
    for arm, k, title in (
            ("learned", 0, "LEAD -- learned under svgd vs learned under IPOPT"),
            ("numerical", 1, "ABLATION -- joint space under svgd vs joint space under IPOPT "
                             "(the swarm without the network; not a baseline)")):
        print(f"\n  {title}, same cells, exact McNemar (svgd+ / IPOPT+ = cells only that solver solved):")
        print(f"    {'variant':<10}{'round':<7}{'svgd':>6}{'IPOPT':>7}{'svgd+':>7}{'IPOPT+':>8}"
              f"{'p':>10}  verdict")
        for v in order:
            if v == STAGE_TWIN or v not in tags:
                continue
            head = f"    {v:<10}{runs[tags[v]]['round']:<7}"
            if arm not in arms_of(v):
                print(f"{head}not run ({arms_only(arms_of(v))})")
                continue
            if status[v] != "ok":
                print(f"{head}{status[v]}: {tags[v]}")
                continue
            m = vs_twin.get(v, {}).get(arm)
            if m is None:
                why = "the twin is MISSING" if twin is None else "refused, see above"
                print(f"{head}{stats[v][k]['succ']:>6}{'--':>7}   not paired: {why}")
                continue
            sv, tw = stats[v][k], stats[STAGE_TWIN][k]
            if max(sv["icap"], tw["icap"]) >= thr:
                verdict_s = "NO VERDICT (iteration-budget-bound)"
            else:
                verdict_s = solver_verdict(m)
                rep.verdicts[(key, v, arm)] = verdict_s
            if max(sv["to"], tw["to"]) >= thr:
                verdict_s += "   [clock-bound: a result at the fielded clock]"
            print(f"{head}{m['a_succ']:>6}{m['b_succ']:>7}{m['a_only']:>7}{m['b_only']:>8}"
                  f"{m['p']:>10.3g}  {verdict_s}")
    ## Per variant: the reporting quartet, learned and joint space adjacent, and record["svgd"].
    hdr = (f"    {'arm':<12}{'succ':>8}{'vs IPOPT twin':>20}{'viol':>10}{'cost':>9}{'n':>4}"
           f"{'steps':>8}{'ms/step':>9}{'wall':>8}{'max':>7}{'t/out':>6}{'i/cap':>6}")
    for v in order:
        if v not in tags:
            continue
        tag, info = tags[v], runs[tags[v]]
        print(f"\n  [{v}] round {info['round']} (manifest {info['manifest']}): {tag}")
        if status[v] != "ok":
            print(f"    {status[v]}: {tag}")
            continue
        s = loaded[v]
        L, J = stats[v]
        if L is None or J is None:
            stage_print_one_arm(rep, tag, info, s, L, J, twin, vs_twin.get(v, {}), cap)
            rep.blocks[key + (v,)] = dict(L=L, J=J, lj=None, run=s, vs_twin=vs_twin.get(v, {}),
                                          twin=twin, arms=info["arms"])
            continue
        lj = mcnemar_pair(s, "learned", s, "numerical")
        budget_bound = max(L["icap"], J["icap"]) >= thr
        clock_bound = max(L["to"], J["to"]) >= thr
        vj = "NO VERDICT (iteration-budget-bound)" if budget_bound else verdict(L["succ"], J["succ"], lj["p"])
        print(f"    learned vs joint space under {info['solver']} (printed, NOT a headline): "
              f"{L['succ']} v {J['succ']} of {L['n']}, L-only {lj['a_only']} / JS-only "
              f"{lj['b_only']}, p = {lj['p']:.3g} -> {vj}"
              + ("   [clock-bound: a result at the fielded clock]" if clock_bound else ""))
        steps_label = ("steps = svgd OUTER steps, NOT comparable to IPOPT majors"
                       if L["is_svgd"] or J["is_svgd"] else "steps = IPOPT majors")
        print(f"    ({steps_label}; viol = median max_violation over all cells; cost = median "
              f"reported cost on the n cells both arms solved; wall = mean over all cells, each "
              f"clamped at the {cap:g} s cap)")
        print(hdr)
        tie = lj["p"] >= 0.05
        ms = dict(succ=star((L["succ"], J["succ"]), lower_better=False, tie=tie),
                  viol=star((L["viol"], J["viol"])), cost=star((L["cost"], J["cost"])),
                  steps=star((L["steps"], J["steps"])), wall=star((L["wall"], J["wall"])))
        for i, (arm, st) in enumerate((("learned", L), ("numerical", J))):
            mk = {k: (m[i] if isinstance(m[0], str) else "") for k, m in ms.items()}
            succ = f"{mk['succ']}{st['succ']}{mk['succ']}/{st['n']}"
            cost = (fmt(st["cost"], 9, 3, mk["cost"]) if st["n_both"] >= MIN_COST_CELLS
                    else f"{'N/A':>9}")
            print(f"    {ARM_NAME[arm]:<12}{succ:>8}{pair_str(vs_twin.get(v, {}).get(arm)):>20}"
                  f"{fmt_e(st['viol'], 10, mk['viol'])}{cost}{st['n_both']:>4}"
                  f"{fmt(st['steps'], 8, 0, mk['steps'])}{fmt(st['ms_step'], 9, 1)}"
                  f"{fmt(st['wall'], 8, 2, mk['wall'])}{fmt(st['wall_max'], 7, 1)}{st['to']:>6}{st['icap']:>6}")
        print_svgd_detail(rep, tag, L, J)
        rep.blocks[key + (v,)] = dict(L=L, J=J, lj=lj, run=s, vs_twin=vs_twin.get(v, {}), twin=twin,
                                      arms=info["arms"])


def stage_print_one_arm(rep, tag, info, s, L, J, twin, vs_twin, cap):
    """A run that does not carry both arms (the rounds run the learned arm only): its arm's row,
    no learned-vs-joint-space line, and cost on the cells it and the SAME arm of the IPOPT twin
    both solved -- the both-arms-solved rule, with the twin standing in for the absent arm."""
    not_run = " and ".join(ARM_NAME[a] for a in ARMS if a not in info["arms"])
    print(f"    {not_run} arm not run ({arms_only(info['arms'])}: its manifest item ran --arms "
          f"{','.join(info['arms'])}) -- no learned-vs-joint-space line")
    steps_label = ("steps = svgd OUTER steps, NOT comparable to IPOPT majors"
                   if (L or J)["is_svgd"] else "steps = IPOPT majors")
    print(f"    ({steps_label}; viol = median max_violation over all cells; cost/tw = median "
          f"reported cost on the n cells this arm AND the same arm of the IPOPT twin both solved "
          f"(this run has no second arm); wall = mean over all cells, each clamped at the "
          f"{cap:g} s cap)")
    print(f"    {'arm':<12}{'succ':>8}{'vs IPOPT twin':>20}{'viol':>10}{'cost/tw':>9}{'n':>4}"
          f"{'steps':>8}{'ms/step':>9}{'wall':>8}{'max':>7}{'t/out':>6}{'i/cap':>6}")
    for arm, st in (("learned", L), ("numerical", J)):
        if st is None:
            continue
        if arm in vs_twin:
            c, n = cost_on_shared(s, arm, twin, arm)
            st["cost_vs_twin"], st["n_vs_twin"] = c, n
            cost, n_s = (fmt(c, 9, 3) if n >= MIN_COST_CELLS else f"{'N/A':>9}"), f"{n:>4}"
        else:                                     # the twin is missing or refused
            cost, n_s = f"{'--':>9}", f"{'--':>4}"
        succ = f"{st['succ']}/{st['n']}"
        print(f"    {ARM_NAME[arm]:<12}{succ:>8}{pair_str(vs_twin.get(arm)):>20}"
              f"{fmt_e(st['viol'], 10)}{cost}{n_s}"
              f"{fmt(st['steps'], 8, 0)}{fmt(st['ms_step'], 9, 1)}"
              f"{fmt(st['wall'], 8, 2)}{fmt(st['wall_max'], 7, 1)}{st['to']:>6}{st['icap']:>6}")
    print_svgd_detail(rep, tag, L, J)


def stage_overview(rep, matrix, order, runs_by_row, runs=None):
    """Every variant across the rounds, one table per arm: successes per row and the verdict
    against the IPOPT twin on the same cells. The learned table's W/T/L is the LEAD tally; the
    joint-space table is the ablation's and carries no headline. A variant whose manifest items
    do not run an arm (`runs[tag]["arms"]`; the rounds are learned only) prints `--` there."""
    arms_of = (lambda tag: runs[tag]["arms"]) if runs is not None else (lambda tag: ARMS)
    partial = any(set(arms_of(t)) != set(ARMS) for key in matrix for t in runs_by_row[key].values())
    print(f"\n{'=' * 100}\nVARIANTS ACROSS ROUNDS -- successes per row; the letter is the verdict "
          f"against the IPOPT twin on the same cells\n  (W svgd wins, T tie, L IPOPT wins, NV no "
          f"verdict: iteration-budget-bound; '-' not paired"
          + ("; '--' arm not run by that variant's manifest items" if partial else "") + ")")
    cols = [f"{M.ROW_NAME[r].split(' ')[0]} {s}" for _, r, s in matrix]
    for arm, k, title in (("learned", "L", "learned under svgd -- the question"),
                          ("numerical", "J", "joint space under svgd -- the ablation")):
        print(f"\n  {title}")
        print(f"    {'variant':<10}{'round':<7}" + "".join(f"{c:>16}" for c in cols)
              + ("     W/T/L vs IPOPT" if arm == "learned" else ""))
        for v in order:
            cells, tally, rnd = [], Counter(), None
            for key in matrix:
                tag = runs_by_row[key].get(v)
                if tag is None:
                    cells.append("")
                    continue
                rnd = rnd or tag.split("_")[2]
                if arm not in arms_of(tag):
                    cells.append("--")
                    continue
                b = rep.blocks.get(key + (v,))
                if b is None:
                    cells.append("MISSING" if tag in rep.missing else "REFUSED")
                    continue
                if v == STAGE_TWIN:
                    cells.append(str(b[k]["succ"]))
                    continue
                vd = rep.verdicts.get((key, v, arm))
                mark = (STAGE_VERDICT_MARK[vd] if vd else
                        "NV" if arm in b["vs_twin"] else "-")
                if vd:
                    tally[mark] += 1
                cells.append(f"{b[k]['succ']} {mark}")
            line = f"    {v:<10}{rnd or '':<7}" + "".join(f"{c:>16}" for c in cells)
            if arm == "learned" and v != STAGE_TWIN:
                line += f"     {tally['W']}/{tally['T']}/{tally['L']}"
            print(line)


def stage_pairwise(rep, matrix, order):
    """Variant against variant, every pair that has landed, per arm and row: exact McNemar on the
    same cells. Cell (row variant A, column variant B) reads `+a/-b p`, a = cells only A solved."""
    print(f"\n{'=' * 100}\nVARIANT vs VARIANT -- every pair of landed svgd variants, exact McNemar on "
          f"the same cells; row variant A against column variant B, +A-only/-B-only p")
    for arm, title in (("learned", "learned under svgd"), ("numerical", "joint space under svgd (ablation)")):
        for key in matrix:
            landed = [v for v in order if v != STAGE_TWIN and key + (v,) in rep.blocks
                      and arm in rep.blocks[key + (v,)].get("arms", ARMS)]
            label = f"{key[0]} {M.ROW_NAME[key[1]]} {key[2]}"
            if len(landed) < 2:
                print(f"\n  {title}, {label}: {len(landed)} variant(s) landed -- nothing to pair")
                continue
            print(f"\n  {title}, {label}")
            print(f"    {'A vs B':<10}" + "".join(f"{v:>18}" for v in landed[1:]))
            for i, a in enumerate(landed[:-1]):
                ra = rep.blocks[key + (a,)]["run"]
                cells = []
                for j, b in enumerate(landed[1:], start=1):
                    if j <= i:
                        cells.append("")
                        continue
                    rb = rep.blocks[key + (b,)]["run"]
                    why = incomparable(ra, rb, STAGE_MUST_MATCH)
                    if why:
                        rep.refuse(f"{label} {a} vs {b}: {why}")
                        cells.append("REFUSED")
                        continue
                    m = mcnemar_pair(ra, arm, rb, arm)
                    cells.append(f"+{m['a_only']}/-{m['b_only']} p={m['p']:.2g}")
                print(f"    {a:<10}" + "".join(f"{c:>18}" for c in cells))


def stage_main(args):
    prefix = STAGE_FAMILIES[args.stage]
    runs, variants = stage_catalogue(args.stage)
    primary = stage_primary()
    order = list(variants)
    if args.variants:
        want = [t.strip() for t in args.variants.split(",") if t.strip()]
        unknown = [t for t in want if t not in variants]
        if unknown:
            raise SystemExit(f"--variants: {unknown} not built by gen_manifest; it builds {order}")
        order = [v for v in order if v in want or v in (STAGE_TWIN, primary)]
    robots, rows, starts = M.parse_row_filter(args.rows)
    matrix = stage_rows(runs, robots, rows, starts)
    runs_by_row = {key: {} for key in matrix}
    for tag, t in runs.items():
        key = (t["robot"], t["row"], t["start"])
        if key in runs_by_row and t["variant"] in order:
            runs_by_row[key][t["variant"]] = tag
    rep = Report()
    rep.verdicts = {}
    print(f"svgd report, stage {args.stage}: the {prefix}_* family from cluster/gen_manifest.py, "
          f"{len(matrix)} rows x {len(order)} variants, root {args.root}")
    print("  LEAD per row: learned under svgd vs learned under IPOPT (the twin) on the same cells; "
          "joint space under svgd is an ABLATION (docs/svgd-solver.md, the per-row analysis rule).")
    print(f"  variants (manifest order), each with every --set it runs, then what it changes from "
          f"the primary {primary}:")
    for v in order:
        info = variants[v]
        sets = " ".join(f"{k}={val!r}" for k, val in info["sets"].items())
        what = "the IPOPT twin" if v == STAGE_TWIN else setting_diff(variants, v, primary)
        if set(info["arms"]) != set(ARMS):
            what += f"; {arms_only(info['arms'])} (--arms {','.join(info['arms'])})"
        print(f"    {v:<10}{info['round']:<4}{info['manifest']:<10}  {what}\n"
              f"    {'':<24}--set {sets}")
    for key in matrix:
        stage_print_row(rep, args.root, key, runs_by_row[key], runs, order, primary)
    ab_vs_primary(rep, matrix, order, primary=primary, twin=STAGE_TWIN,
                  legend=f"round {variants[primary]['round']}, manifest {variants[primary]['manifest']}",
                  must=STAGE_MUST_MATCH, arms_of=lambda v: variants[v]["arms"])
    stage_overview(rep, matrix, order, runs_by_row, runs)
    stage_pairwise(rep, matrix, order)
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
    p.add_argument("--stage", default=None, choices=sorted(STAGE_FAMILIES),
                   help="read the cluster stage family instead of the smoke (SVGD: sc_SVGD_*, "
                        "its runs enumerated from cluster/gen_manifest.py); --rows filters it, "
                        "--root and --variants apply, the smoke's tag options do not")
    p.add_argument("--variants", default=None,
                   help="stage mode: only these variants (the twin and the primary always print)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.stage:
        return stage_main(args)
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
    ab_vs_primary(rep, matrix, columns)
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
