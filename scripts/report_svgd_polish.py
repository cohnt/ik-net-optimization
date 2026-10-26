"""Stage SVGD_R31 read against its twins: the float64 polish, cell-paired, by the pre-registered rule.

`pol` (learned) against R29 `pin4`, `polkq` (joint space) against R30 `pin4kq`, on the record's 16
rows. Per row: success (exact McNemar on the discordant cells), median `max_violation` of both runs,
median cost ratio on cells both solved (`record["cost"]` is the reported cost), mean wall (each
clamped at the cap) and its ratio, the cap check (`timed_out` / `hit_iteration_cap`), the per-cell
`min_distance` change on cells both solved (median, min; cells losing > 0.1 mm named), and the
polish's own counters. The rule (`docs/svgd-solver.md`, "Stage SVGD_R31"): success identical,
median residual <= 1e-12, cost ratio in [0.94, 1.10], wall up < 5%, every row, both arms.

    .venv/bin/python scripts/report_svgd_polish.py            # Markdown to stdout; exit 2 on a refused pairing
    .venv/bin/python scripts/report_svgd_polish.py --same-swarm   # only cells whose pre-polish point is the twin's

`--same-swarm` keeps a cell only if the polish's `polish_viol_before` equals the twin's `max_violation`
(the same particle reached the selection), so the pair differs by the polish alone. R31 found the
swarm itself is not reproducible shard for shard on the rows whose maps run compiled torch (every
learned row but the Panda's, and joint space on the screw and soft PCS arms): a whole shard either
reproduces bit for bit or diverges from its first cell, independent of the host. The SAME column
counts the kept cells; on the full grid the divergent cells are a same-config replicate, not the polish.
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from collections import Counter

CAP = 180.0
BAND = (0.94, 1.10)
RESIDUAL_MAX = 1e-12
WALL_MAX = 1.05
CLEARANCE_LOSS = 1e-4  # metres
ROBOTS = [("panda", "panda", "n6"), ("iiwa", "iiwa", "n4"), ("screw7_p050", "screw_arm", "n6"),
          ("soft12", "soft12", "n6")]
ARMS = [  # label, arm key, twin stage + token, polish stage + token
    ("learned", "learned", ("R29", "pin4"), ("R31", "pol")),
    ("joint space", "numerical", ("R30", "pin4kq"), ("R31", "polkq")),
]
TAG = "sc_SVGD_{stage}_{robot}_{rung}_svgd_{task}_480_180_{start}_{token}"
MUST_MATCH = ("grid_hash", "scene_fingerprint")


def load(results_dir, tag, arm):
    path = os.path.join("results", results_dir, "benchmark", tag, "summary.json")
    if not os.path.exists(path):
        return None, None
    s = json.load(open(path))
    return s["metadata"], s["records"][arm]


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def min_distance(r):
    return (r.get("detail") or {}).get("min_distance")


def row_name(row):
    return row.get("row") if isinstance(row, dict) else row


def same_swarm(a, b):
    v = (b.get("svgd") or {}).get("polish_viol_before")
    return v is not None and abs(a["max_violation"] - v) <= 1e-15


def read_row(results_dir, robot, rung, task, start, arm, twin, pol, only_same=False):
    t_tag = TAG.format(stage=twin[0], robot=robot, rung=rung, task=task, start=start, token=twin[1])
    p_tag = TAG.format(stage=pol[0], robot=robot, rung=rung, task=task, start=start, token=pol[1])
    tm, tr = load(results_dir, t_tag, arm)
    pm, pr = load(results_dir, p_tag, arm)
    if tr is None or pr is None:
        return {"missing": t_tag if tr is None else p_tag}
    for k in MUST_MATCH:
        if tm.get(k) != pm.get(k):
            print(f"REFUSED: {k} differs between {t_tag} and {p_tag}", file=sys.stderr)
            sys.exit(2)
    if len(tr) != len(pr):
        print(f"REFUSED: {len(tr)} v {len(pr)} cells ({t_tag})", file=sys.stderr)
        sys.exit(2)
    for a, b in zip(tr, pr):  # same cell: same target and start
        if a["target"] != b["target"] or a["guess"] != b["guess"]:
            print(f"REFUSED: cell order differs ({t_tag})", file=sys.stderr)
            sys.exit(2)
    same = [same_swarm(a, b) for a, b in zip(tr, pr)]
    n_same = sum(same)
    if only_same:
        tr = [r for r, k in zip(tr, same) if k]
        pr = [r for r, k in zip(pr, same) if k]
        if not tr:
            return {"missing": f"no same-swarm cell ({p_tag})"}
    ok_t = [bool(r["feasible"]) for r in tr]
    ok_p = [bool(r["feasible"]) for r in pr]
    b = sum(x and not y for x, y in zip(ok_t, ok_p))
    c = sum(y and not x for x, y in zip(ok_t, ok_p))
    both = [i for i in range(len(tr)) if ok_t[i] and ok_p[i]]
    cost_t = st.median(tr[i]["cost"] for i in both) if both else float("nan")
    cost_p = st.median(pr[i]["cost"] for i in both) if both else float("nan")
    wall_t = st.mean(min(r["wall_time"], CAP) for r in tr)
    wall_p = st.mean(min(r["wall_time"], CAP) for r in pr)
    dd = []
    for i in both:
        d0, d1 = min_distance(tr[i]), min_distance(pr[i])
        if d0 is not None and d1 is not None:
            dd.append((d1 - d0, i))
    losers = sorted((d, i) for d, i in dd if d < -CLEARANCE_LOSS)
    sv = [r.get("svgd") or {} for r in pr]
    ran = [s for s in sv if s.get("polish_steps") is not None]
    kinds = Counter(s.get("polish_reject_kind") for s in ran if s.get("polish_rejected"))
    rows = Counter(row_name(s.get("polish_reject_row")) for s in ran if s.get("polish_rejected"))
    return dict(
        n=len(tr), n_same=n_same, succ_t=sum(ok_t), succ_p=sum(ok_p), b=b, c=c, p=mcnemar_p(b, c),
        viol_t=st.median(r["max_violation"] for r in tr), viol_p=st.median(r["max_violation"] for r in pr),
        viol_p_max_ok=max((r["max_violation"] for r, o in zip(pr, ok_p) if o), default=float("nan")),
        cost_ratio=cost_p / cost_t if both else float("nan"), n_both=len(both),
        wall_t=wall_t, wall_p=wall_p, wall_ratio=wall_p / wall_t,
        cap_t=sum(bool(r.get("timed_out")) or bool(r.get("hit_iteration_cap")) for r in tr),
        cap_p=sum(bool(r.get("timed_out")) or bool(r.get("hit_iteration_cap")) for r in pr),
        dmin_med=st.median(d for d, _ in dd) if dd else float("nan"),
        dmin_min=min((d for d, _ in dd), default=float("nan")), losers=losers,
        polished=len(ran), steps=Counter(s["polish_steps"] for s in ran),
        rejected=sum(bool(s.get("polish_rejected")) for s in ran), kinds=kinds, reject_rows=rows,
        held=sum(s.get("polish_active_ineq_held") or 0 for s in ran),
        driven=sum(s.get("polish_active_ineq_driven") or 0 for s in ran),
        slack_drop_max=max((s.get("polish_inactive_slack_drop") or 0.0 for s in ran), default=float("nan")),
        polish_ms=1e3 * st.median(s.get("polish_seconds") or 0.0 for s in ran) if ran else float("nan"),
    )


def verdict(r):
    fails = []
    if r["b"] or r["c"]:
        fails.append("1")
    if not r["viol_p"] <= RESIDUAL_MAX:
        fails.append("2")
    if not BAND[0] <= r["cost_ratio"] <= BAND[1]:
        fails.append("3")
    if not r["wall_ratio"] < WALL_MAX:
        fails.append("4")
    return "pass" if not fails else "FAIL " + ",".join(fails)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--same-swarm", action="store_true", help="only cells whose pre-polish point is the twin's")
    args = ap.parse_args()
    scope = "same-swarm cells only" if args.same_swarm else "all 480 cells"
    out = [f"# Stage SVGD_R31: the float64 polish against its twins (cell-paired, {scope})", ""]
    all_fails = Counter()
    for label, arm, twin, pol in ARMS:
        out += [f"## {label}: `{pol[1]}` v `{twin[1]}` ({twin[0]})", "",
                "| row | same | success twin / pol (b/c, p) | median resid twin -> pol | max resid pol (solved) "
                "| cost ratio (n) | mean wall twin -> pol (ratio) | cap twin / pol | d min_dist med / min (mm) "
                "| polish steps | rejected (kind; row) | held / driven | polish ms | rule |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        notes = []
        for robot, results_dir, rung in ROBOTS:
            for task in ("mugshelf", "posetip"):
                for start in ("native", "paired"):
                    name = f"{robot} {'grasp' if task == 'mugshelf' else 'pose'} {start}"
                    r = read_row(results_dir, robot, rung, task, start, arm, twin, pol, args.same_swarm)
                    if "missing" in r:
                        out.append(f"| {name} | MISSING `{r['missing']}` |" + " |" * 12)
                        all_fails["missing"] += 1
                        continue
                    v = verdict(r)
                    for f in v.split()[1:2]:
                        for k in f.split(","):
                            all_fails[f"{label}:{k}"] += 1
                    steps = " ".join(f"{k}:{n}" for k, n in sorted(r["steps"].items()))
                    kinds = ", ".join(f"{k} {n}" for k, n in r["kinds"].most_common())
                    rrows = ", ".join(f"{k} {n}" for k, n in r["reject_rows"].most_common(2))
                    out.append(
                        f"| {name} | {r['n_same']} | {r['succ_t']} / {r['succ_p']} ({r['b']}/{r['c']}, {r['p']:.2g}) "
                        f"| {r['viol_t']:.1e} -> {r['viol_p']:.1e} | {r['viol_p_max_ok']:.1e} "
                        f"| {r['cost_ratio']:.4f} ({r['n_both']}) "
                        f"| {r['wall_t']:.2f} -> {r['wall_p']:.2f} ({r['wall_ratio']:.3f}) "
                        f"| {r['cap_t']} / {r['cap_p']} "
                        f"| {1e3 * r['dmin_med']:+.4f} / {1e3 * r['dmin_min']:+.4f} "
                        f"| {steps} | {r['rejected']} ({kinds}; {rrows}) | {r['held']} / {r['driven']} "
                        f"| {r['polish_ms']:.0f} | {v} |")
                    for d, i in r["losers"]:
                        notes.append(f"- {name} cell {i}: min_distance {1e3 * d:+.3f} mm")
        out += [""] + (notes or ["No cell lost more than 0.1 mm of clearance."]) + [""]
    out.append("Rule clause failures: " + (", ".join(f"{k} x{n}" for k, n in sorted(all_fails.items()))
                                          or "none"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
