#!/usr/bin/env python3
"""Read stage REMEASURE: the record re-measured on the fixed wsg scene.

Three things changed under the record at once (cluster/gen_manifest.py, the REMEASURE block):
the wsg finray's `between_fingers` yaw (every grasp target on the iiwa, soft PCS, screw and GVS
arms started with the mug handle 18 mm inside a finger; the Panda's gripper is a different SDF),
the five cross-robot settings, and CUDA graphs with the iteration budgets lifted. This prints,
per row of the record:

  1. RECORD -> REMEASURE. Both arms' successes before and after, the learned-vs-joint-space
     verdict (exact McNemar WITHIN each run) before and after and whether it flipped, target-level
     success-rate CIs, and the after-run's quartet: median iterations and cost on cells BOTH arms
     solved, mean wall over all cells clamped at the clock, and the cap check.
  2. ATTRIBUTION, IPOPT rows: record -> REMEASURE_LEGACY (fixed scene, OLD settings) ->
     REMEASURE (fixed scene, new settings). Scene effect = LEGACY - record; settings effect =
     REMEASURE - LEGACY.
  3. THE TRUST-REGION A/B, IPOPT, Panda and iiwa: REMEASURE_RULE against REMEASURE.
  4. ACCEPTANCE: scene fingerprints, joint space bit-identical between protocols, and the Panda
     (whose scene did not change) reproducing the record under the legacy settings.

THE ONE RULE THIS SCRIPT EXISTS TO KEEP: cells are never paired across the scene change.
`grid_hash` hashes only the sampled q's, so a grasp run on the defective SDF and one on the fixed
SDF share a grid hash while their targets sit in different scenes -- a McNemar across them would
count "the same cell" twice over different problems. So record -> REMEASURE compares VERDICTS
(each computed within one run) and target-level rates with an UNPAIRED bootstrap. Cells are
paired only between runs that share a scene fingerprint (the three REMEASURE sub-stages), and
in one place between the record and a re-measurement: the Panda, whose gripper SDF the fix did
not touch, in the acceptance check -- gated on `SCENE_UNCHANGED` and an identical grid_hash.

Layout follows CLAUDE.md "Every result is told in success, iterations, cost and wall clock":
learned and joint space adjacent, the better of each pair *starred* (a success tie is decided by
McNemar, not numeric equality), every expected row printed whether or not it has run.

The cap check: `timed_out` and `hit_iteration_cap` are both printed. A row with >= 24 of 480
cells on either arm at an ITERATION budget (`hit_iteration_cap`, or NLopt's `hit_eval_cap`)
carries NO VERDICT (CLAUDE.md, the cap check). Timeouts are not a budget in that sense -- the
180 s clock is the fielded usability limit and is never lifted -- so they are printed beside
the verdict, not used to void it.

Usage:
    scripts/report_remeasure.py                  # everything found
    scripts/report_remeasure.py ipopt            # section 1 for these solvers only
    scripts/report_remeasure.py --dry-record     # exercise every table on the record itself
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cluster"))

from report_statusquo import (CELLS, SOLVERS, SOLVER_NAME, by_cell, load,  # noqa: E402
                              load_record, mcnemar, median, num, parse_tag, verdict, arm_stats)
from report_gvs import target_rate_diff_ci  # noqa: E402
from gen_manifest import REMEASURE_VARIANTS, stage_REMEASURE  # noqa: E402

#: The robots whose scene the yaw fix did NOT touch. The Panda carries `panda_finray.sdf`, whose
#: `between_fingers` always had yaw 1.57; the fix edited `wsg50_110_finray_fingers_box_collision.sdf`
#: only. This is the one robot on which a record cell may be paired with a re-measured one.
SCENE_UNCHANGED = {"panda"}
#: CLAUDE.md, the cap check: this many cells of 480 at an iteration budget void a verdict.
BUDGET_CELLS = 24
ROW_NAME = {"mugshelf": "grasp", "posetip": "pose"}
ROW_ORDER = {"mugshelf": 0, "posetip": 1}
ROBOT_ORDER = {"iiwa": 0, "soft12": 1, "screw7_p050": 2, "gvs_pushrod9_o1": 3, "panda": 4}
GVS = "gvs_pushrod9_o1"


# ------------------------------------------------------------------------------- loading

def key_of(tag, prefix="sc_"):
    """(robot, rung, row, start, solver) of a tag, with its stage prefix collapsed to one token.

    `sc_REMEASURE_LEGACY_iiwa_...` read by parse_tag directly would put `LEGACY_iiwa` in the
    robot field, so every tag is re-spelt `sc_X_<rest>` first.
    """
    t = parse_tag("sc_X_" + tag[len(prefix):])
    return (t["robot"], t["rung"], t["row"], t["start"], t["solver"])


def expected_rows(variant):
    """Every logical run the manifest generates, keyed as above -- so zero rows still print."""
    prefix = f"sc_{REMEASURE_VARIANTS[variant][0]}_"
    tags = {it["id"].rsplit("_shard", 1)[0] for it in stage_REMEASURE(variant)}
    return {key_of(t, prefix): t for t in tags}


def load_remeasure():
    """{variant: {key: summary}} for every merged 480-cell REMEASURE run, staged or promoted.

    REMEASURE_NLOPT shares the primary's tag family (they ARE its NLopt rows), so it is told
    apart by solver, not by prefix."""
    out = {v: {} for v in REMEASURE_VARIANTS}
    for tag, s in load("sc_REMEASURE_", cells=CELLS).items():
        for v in ("REMEASURE_LEGACY", "REMEASURE_RULE", "REMEASURE"):   # longest prefix first
            prefix = f"sc_{REMEASURE_VARIANTS[v][0]}_"
            if tag.startswith(prefix):
                k = key_of(tag, prefix)
                if v == "REMEASURE" and k[4] == "nlopt":
                    v = "REMEASURE_NLOPT"
                out[v][k] = s
                break
    return out


def load_before():
    """The record as the 'before' column: STATUSQUO + SOFT12 + SCREW (lifted rows substituted,
    exactly as report_statusquo reads them) and stage GVS's primary rung."""
    runs, _ = load_record(which="legacy")
    runs.update(load(f"sc_GVS_{GVS}_", cells=CELLS))
    out = {}
    for tag, s in runs.items():
        k = key_of(tag, "sc_" + tag.split("_")[1] + "_")
        if k[2] in ROW_NAME:
            out[k] = s
    return out


def check_fingerprints(rm):
    """Every REMEASURE run must carry `metadata["scene_fingerprint"]`, and every run on one
    (robot, task) scene -- all three sub-stages, both protocols, every solver -- the SAME one.
    Different robots and tasks are different scenes and legitimately differ. Raises on failure:
    a run without a fingerprint, or two fingerprints for one scene, means the stage was staged
    from two trees or from a tree before 8e7323d, and nothing below may be read."""
    groups, missing = {}, []
    for v, runs in rm.items():
        for k, s in runs.items():
            fp = s["metadata"].get("scene_fingerprint")
            if not fp:
                missing.append(f"{v} {k}")
                continue
            groups.setdefault((k[0], k[2]), {}).setdefault(fp, []).append(f"{v} {k[3]} {k[4]}")
    print("\n=== SCENE FINGERPRINTS (metadata['scene_fingerprint'], per robot x task scene)")
    for (robot, row), fps in sorted(groups.items(), key=lambda kv: (ROBOT_ORDER.get(kv[0][0], 9),
                                                                     ROW_ORDER[kv[0][1]])):
        n = sum(len(v) for v in fps.values())
        state = "identical" if len(fps) == 1 else f"{len(fps)} DIFFERENT"
        print(f"  {robot:<16}{ROW_NAME[row]:<7}{n:>3} runs  {state}: "
              + ", ".join(fp[:12] for fp in sorted(fps)))
    if missing:
        raise SystemExit(f"REFUSING: {len(missing)} REMEASURE run(s) carry no scene_fingerprint, "
                         f"e.g. {missing[:3]} -- staged from a tree before 8e7323d?")
    split = {k: fps for k, fps in groups.items() if len(fps) > 1}
    if split:
        raise SystemExit(f"REFUSING: runs on one scene disagree on scene_fingerprint: {split}")


# ------------------------------------------------------------------------------ per run

def within(s):
    """Everything one run says about learned vs joint space, computed inside that run only."""
    L, J = arm_stats(s, "learned", "numerical"), arm_stats(s, "numerical", "learned")
    A, B = by_cell(s, "learned"), by_cell(s, "numerical")
    shared = [k for k in A if k in B]
    b = sum(1 for k in shared if A[k]["feasible"] and not B[k]["feasible"])
    w = sum(1 for k in shared if B[k]["feasible"] and not A[k]["feasible"])
    both = [k for k in shared if A[k]["feasible"] and B[k]["feasible"]]
    p = mcnemar(b, w)
    budget = {arm: sum(1 for r in recs.values()
                       if r.get("hit_iteration_cap") or r.get("hit_eval_cap"))
              for arm, recs in (("learned", A), ("numerical", B))}
    void = max(budget.values()) >= BUDGET_CELLS
    return dict(L=L, J=J, p=p, lonly=b, jonly=w,
                verdict="NO VERDICT" if void else verdict(L["succ"], J["succ"], p),
                it_L=median([A[k]["iterations"] for k in both]),
                it_J=median([B[k]["iterations"] for k in both]),
                to_L=L["to"], to_J=J["to"], ic_L=budget["learned"], ic_J=budget["numerical"],
                ci_L=s["summary"]["learned"].get("success_ci"),
                ci_J=s["summary"]["numerical"].get("success_ci"))


def pair(a, b, w, prec, lower, tied=False):
    """Two adjacent cells, the better *starred*. `None` prints as `--` and is starred never."""
    sa, sb = num(a, 0, prec).strip(), num(b, 0, prec).strip()
    if a is not None and b is not None:
        if tied or a == b:
            sa, sb = f"*{sa}*", f"*{sb}*"
        elif (a < b) == lower:
            sa = f"*{sa}*"
        else:
            sb = f"*{sb}*"
    return f"{sa:>{w}}{sb:>{w}}"


def ci(c):
    return "[  --  ,  --  ]" if not c else f"[{c[0]:.3f},{c[1]:.3f}]"


def dci(sa, sb, arm):
    """Unpaired, target-level difference in success rate, after - before, as percentage points."""
    if sa is None or sb is None:
        return f"{'--':>20}"
    d, lo, hi = target_rate_diff_ci(sa, sb, arm=arm)
    return f"{100 * d:+6.1f} [{100 * lo:+5.1f},{100 * hi:+5.1f}]"


def label(k):
    robot, _, row, start, _ = k
    return f"{robot} {ROW_NAME[row]} {start}"


def order(k):
    return (ROBOT_ORDER.get(k[0], 9), ROW_ORDER[k[2]], k[3])


# ------------------------------------------------------------------------------ sections

def section_record(before, after, want):
    print("\n=== 1. RECORD -> REMEASURE  (verdicts within each run; NEVER cells across the scene change)")
    print("  L = learned, JS = joint space. Successes of 480; verdict by exact McNemar within the run;")
    print(f"  NO VERDICT = >= {BUDGET_CELLS} cells on an arm at an iteration budget. dL/dJS = after -"
          " before, target-level")
    print("  success rate in points with an UNPAIRED target bootstrap 95% CI. CIs on a rate are the")
    print("  run's own target-level bootstrap (summary success_ci).")
    expected = {**expected_rows("REMEASURE"), **expected_rows("REMEASURE_NLOPT")}
    tally = {s: {"before": {}, "after": {}, "flips": 0, "rows": 0} for s in SOLVERS}
    for solver in want:
        keys = sorted((k for k in expected if k[4] == solver), key=order)
        if not keys:
            continue
        print(f"\n--- {SOLVER_NAME[solver]}")
        print(f"  {'row':<30}{'L bef':>6}{'L aft':>6}{'L CI aft':>16}{'JS bef':>7}{'JS aft':>7}"
              f"{'JS CI aft':>16}{'verdict before':>15}{'verdict after':>15}{'flip':>6}"
              f"{'dL pts [CI]':>21}{'dJS pts [CI]':>21}")
        for k in keys:
            b, a = before.get(k), after.get(k)
            wb, wa = (within(b) if b else None), (within(a) if a else None)
            vb, va = (wb["verdict"] if wb else "--"), (wa["verdict"] if wa else "not run")
            flip = "--" if not (wb and wa) else ("FLIP" if vb != va else "")
            tally[solver]["rows"] += 1
            if wb:
                tally[solver]["before"][vb] = tally[solver]["before"].get(vb, 0) + 1
            if wa:
                tally[solver]["after"][va] = tally[solver]["after"].get(va, 0) + 1
                tally[solver]["flips"] += flip == "FLIP"
            g = lambda w, arm: w[arm]["succ"] if w else None  # noqa: E731
            print(f"  {label(k):<30}{num(g(wb, 'L'), 6)}{num(g(wa, 'L'), 6)}"
                  f"{ci(wa['ci_L'] if wa else None):>16}{num(g(wb, 'J'), 7)}{num(g(wa, 'J'), 7)}"
                  f"{ci(wa['ci_J'] if wa else None):>16}{vb:>15}{va:>15}{flip:>6}"
                  f"{dci(a, b, 'learned'):>21}{dci(a, b, 'numerical'):>21}")
        print(f"\n  after-run quartet ({solver}): iterations and cost are medians on cells BOTH arms"
              " solved; wall")
        print("  is the mean over all cells, each clamped at the 180 s clock; TO = timed_out, IC = at")
        print("  an iteration budget (hit_iteration_cap / hit_eval_cap). *starred* = better of the pair.")
        print(f"  {'row':<30}{'success L':>11}{'JS':>9}{'iters L':>10}{'JS':>8}{'cost L':>10}"
              f"{'JS':>9}{'n both':>7}{'wall L s':>10}{'JS':>8}{'TO L':>6}{'JS':>5}{'IC L':>6}"
              f"{'JS':>5}")
        for k in keys:
            a = after.get(k)
            if not a:
                print(f"  {label(k):<30}{'not run':>11}")
                continue
            w = within(a)
            L, J = w["L"], w["J"]
            print(f"  {label(k):<30}"
                  f"{pair(L['succ'], J['succ'], 10, 0, False, tied=w['p'] >= 0.05)}"
                  f"{pair(w['it_L'], w['it_J'], 9, 0, True)}"
                  f"{pair(L['cost'], J['cost'], 9, 3, True)}{L['n_both']:>7}"
                  f"{pair(L['wall_all_clock'], J['wall_all_clock'], 9, 2, True)}"
                  f"{w['to_L']:>6}{w['to_J']:>5}{w['ic_L']:>6}{w['ic_J']:>5}")
    print("\n  TALLY (learned vs joint space, per solver)")
    for solver in want:
        t = tally[solver]
        if not t["rows"]:
            continue
        fmt = lambda d: ", ".join(f"{v} {d.get(v, 0)}" for v in  # noqa: E731
                                  ("learned", "tie", "joint space", "NO VERDICT"))
        print(f"    {solver:<6} before: {fmt(t['before'])}")
        print(f"    {'':<6} after:  {fmt(t['after'])}   ({t['flips']} verdict(s) flipped, "
              f"{sum(t['after'].values())} of {t['rows']} rows run)")


def paired_arm(sa, sb, arm):
    """Cell-paired comparison of ONE arm between two runs on the SAME scene and grid."""
    A, B = by_cell(sa, arm), by_cell(sb, arm)
    shared = [k for k in A if k in B]
    gained = sum(1 for k in shared if B[k]["feasible"] and not A[k]["feasible"])
    lost = sum(1 for k in shared if A[k]["feasible"] and not B[k]["feasible"])
    return gained, lost, mcnemar(gained, lost)


def same_scene(sa, sb, what):
    fa, fb = sa["metadata"].get("scene_fingerprint"), sb["metadata"].get("scene_fingerprint")
    ga, gb = sa["metadata"].get("grid_hash"), sb["metadata"].get("grid_hash")
    if not fa or fa != fb or ga != gb:
        raise SystemExit(f"REFUSING to pair {what}: scene_fingerprint {fa} vs {fb}, "
                         f"grid_hash {ga} vs {gb}")


def section_attribution(before, rm):
    print("\n=== 2. ATTRIBUTION, IPOPT: record -> LEGACY (fixed scene, OLD settings) -> REMEASURE"
          " (new settings)")
    print("  scene = LEGACY - record: unpaired target bootstrap, points [95% CI]. It also carries CUDA")
    print("  graphs and (on rows the record never lifted) the lifted budget, so on the Panda, whose")
    print("  scene did not change, it measures exactly those -- the control for the other four.")
    print("  settings = REMEASURE vs LEGACY, same scene and grid, so CELL-PAIRED: +gained/-lost, McNemar p.")
    exp = sorted(expected_rows("REMEASURE_LEGACY"), key=order)
    print(f"  {'row':<30}{'L rec':>6}{'LEG':>5}{'NEW':>5}{'JS rec':>7}{'LEG':>5}{'NEW':>5}"
          f"{'scene L':>21}{'scene JS':>21}{'settings L':>16}{'settings JS':>16}")
    for k in exp:
        b, leg, new = before.get(k), rm["REMEASURE_LEGACY"].get(k), rm["REMEASURE"].get(k)
        s = lambda run, arm: (len([r for r in run["records"][arm] if r["feasible"]])  # noqa: E731
                              if run else None)
        sett = {}
        for arm in ("learned", "numerical"):
            if leg and new:
                same_scene(leg, new, f"{label(k)} LEGACY vs REMEASURE")
                g, lo, p = paired_arm(leg, new, arm)
                sett[arm] = f"+{g}/-{lo} p={p:.2g}"
            else:
                sett[arm] = "--"
        print(f"  {label(k):<30}{num(s(b, 'learned'), 6)}{num(s(leg, 'learned'), 5)}"
              f"{num(s(new, 'learned'), 5)}{num(s(b, 'numerical'), 7)}{num(s(leg, 'numerical'), 5)}"
              f"{num(s(new, 'numerical'), 5)}{dci(leg, b, 'learned'):>21}"
              f"{dci(leg, b, 'numerical'):>21}{sett['learned']:>16}{sett['numerical']:>16}")


def section_rule(rm):
    print("\n=== 3. TRUST-REGION A/B, IPOPT: REMEASURE_RULE (radius sqrt(dim)+1.5) vs REMEASURE")
    print("  (Panda 4.0 -> 4.15, iiwa 4.3 -> 4.33). Same scene and grid, so cell-paired. The region is")
    print("  learned-only, so joint space must not move: its discordance is printed as a check.")
    exp = sorted(expected_rows("REMEASURE_RULE"), key=order)
    print(f"  {'row':<30}{'L rule':>7}{'L new':>7}{'+/-':>9}{'p':>9}{'verdict':>12}"
          f"{'iters rule':>11}{'new':>6}{'JS disc':>9}{'radius':>14}")
    for k in exp:
        rule, new = rm["REMEASURE_RULE"].get(k), rm["REMEASURE"].get(k)
        if not (rule and new):
            print(f"  {label(k):<30}{'not run' if not rule else '--':>7}"
                  f"{'not run' if not new else '--':>7}")
            continue
        same_scene(new, rule, f"{label(k)} RULE vs REMEASURE")
        g, lo, p = paired_arm(new, rule, "learned")
        jg, jl, _ = paired_arm(new, rule, "numerical")
        A, B = by_cell(rule, "learned"), by_cell(new, "learned")
        both = [c for c in A if c in B and A[c]["feasible"] and B[c]["feasible"]]
        lr, ln = (sum(r["feasible"] for r in X.values()) for X in (A, B))
        v = "tie" if p >= 0.05 else ("rule" if lr > ln else "constant")
        rad = lambda s: ((s["metadata"].get("robot_settings") or {}).get("learned")  # noqa: E731
                         or {}).get("latent_trust_region")
        radius = f"{num(rad(rule), 0, 2).strip()}/{num(rad(new), 0, 2).strip()}"
        print(f"  {label(k):<30}{lr:>7}{ln:>7}{f'+{g}/-{lo}':>9}{p:>9.3g}{v:>12}"
              f"{num(median([A[c]['iterations'] for c in both]), 11)}"
              f"{num(median([B[c]['iterations'] for c in both]), 6)}{jg + jl:>9}{radius:>14}")


def cap_bound(r):
    return bool(r.get("timed_out") or r.get("hit_iteration_cap") or r.get("hit_eval_cap"))


def section_acceptance(before, rm):
    print("\n=== 4. ACCEPTANCE")
    print("--- 4a. scene fingerprints: checked above, before any table (the script refuses otherwise).")
    print("\n--- 4b. joint space is bit-identical between the two start protocols (its native start IS")
    print("  a random configuration). Per cell: verdict, iterations, cost and q. Cells cap-bound in")
    print("  either run are excluded and counted -- a clock stop lands on a different iterate.")
    bad = n = 0
    for v, runs in rm.items():
        for k, nat in sorted(runs.items(), key=lambda kv: order(kv[0]) + (kv[0][4],)):
            if k[3] != "native":
                continue
            par = runs.get((k[0], k[1], k[2], "paired", k[4]))
            if par is None:
                continue
            A, B = by_cell(nat, "numerical"), by_cell(par, "numerical")
            shared = [c for c in A if c in B]
            capped = [c for c in shared if cap_bound(A[c]) or cap_bound(B[c])]
            diff = [c for c in shared if c not in capped and any(
                A[c].get(f) != B[c].get(f) for f in ("feasible", "iterations", "cost", "q"))]
            n += 1
            bad += bool(diff)
            print(f"  {v:<17}{label(k).rsplit(' ', 1)[0]:<22}{k[4]:<6} {len(shared) - len(capped):>4}"
                  f" compared, {len(capped):>3} cap-bound excluded, "
                  + ("identical" if not diff else f"{len(diff)} DIFFER, e.g. {diff[:3]}"))
    print(f"  => {n - bad} of {n} protocol pairs identical" if n else "  (no protocol pairs found)")

    print("\n--- 4c. the Panda reproduces the record under the legacy settings (its scene did not")
    print("  change). Cell-paired against the record -- the only record pairing this script makes,")
    print("  gated on SCENE_UNCHANGED and an identical grid_hash. REPRODUCES = every discordant cell")
    print("  was cap-bound in one of the two runs (the cap-bound reproducibility band); CUDA graphs")
    print("  may add learned cells (+0-11 per IPOPT row in stage CUDAGRAPH), never remove them.")
    for k, leg in sorted(rm["REMEASURE_LEGACY"].items(), key=lambda kv: order(kv[0])):
        if k[0] not in SCENE_UNCHANGED:
            continue
        rec = before.get(k)
        if rec is None:
            print(f"  {label(k):<30} no record run")
            continue
        if rec["metadata"].get("grid_hash") != leg["metadata"].get("grid_hash"):
            print(f"  {label(k):<30} grid_hash differs ({rec['metadata'].get('grid_hash')} vs "
                  f"{leg['metadata'].get('grid_hash')}): NOT PAIRED")
            continue
        out = []
        for arm, short in (("learned", "L"), ("numerical", "JS")):
            A, B = by_cell(rec, arm), by_cell(leg, arm)
            disc = [c for c in A if c in B and A[c]["feasible"] != B[c]["feasible"]]
            uncapped = [c for c in disc if not (cap_bound(A[c]) or cap_bound(B[c]))]
            gained = sum(1 for c in disc if B[c]["feasible"])
            out.append(f"{short} {sum(r['feasible'] for r in A.values())}->"
                       f"{sum(r['feasible'] for r in B.values())} (+{gained}/-{len(disc) - gained}, "
                       f"{len(uncapped)} uncapped) "
                       + ("REPRODUCES" if not uncapped else "DIFFERS"))
        print(f"  {label(k):<30} " + "   ".join(out))


def dry_record(before):
    """Hand the record to every section as if it were the three sub-stages, with synthetic
    per-scene fingerprints, so each table's code runs before the stage has produced a cell."""
    import copy
    rm = {v: {} for v in REMEASURE_VARIANTS}
    for variant in REMEASURE_VARIANTS:
        for k in expected_rows(variant):
            if k in before:
                s = copy.copy(before[k])
                s["metadata"] = dict(s["metadata"], scene_fingerprint=f"dry-{k[0]}-{k[2]}")
                rm[variant][k] = s
    return rm


def main(argv):
    dry = "--dry-record" in argv
    want = [s for s in SOLVERS if s in argv] or list(SOLVERS)
    before = load_before()
    rm = dry_record(before) if dry else load_remeasure()
    n_after = sum(len(v) for v in rm.values())
    if not n_after:
        print("no merged 480-cell sc_REMEASURE_ runs found (staged or promoted). Collect and merge "
              "first:\n  cluster/collect_results.sh   (cluster/REMEASURE_RUNBOOK.md)\n"
              f"The record is readable: {len(before)} runs found to compare against.")
        return 1
    print("STAGE REMEASURE -- the record re-measured on the fixed wsg scene, unified settings, CUDA")
    print("graphs, lifted iteration budgets; 480 cells, 180 s, seed 1, PROCS=8 under MPS.")
    if dry:
        print("*** DRY RUN: the RECORD is standing in for every sub-stage (fingerprints synthetic). ***")
    for v in REMEASURE_VARIANTS:
        print(f"  {v:<17} {len(rm[v]):>3} of {len(expected_rows(v)):>3} logical runs found")
    print(f"  record (before)   {len(before):>3} runs: STATUSQUO + SOFT12 + SCREW (lifted rows "
          "substituted) + GVS o1")
    print("  The record ran at PROCS=8 without MPS and stage GVS at PROCS=2, so wall clock moves for")
    print("  reasons other than the scene; read iterations for the formulation.")
    check_fingerprints(rm)
    section_record(before, {**rm["REMEASURE"], **rm["REMEASURE_NLOPT"]}, want)
    section_attribution(before, rm)
    section_rule(rm)
    section_acceptance(before, rm)
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() if not a.startswith("--") else a for a in sys.argv[1:]]))
