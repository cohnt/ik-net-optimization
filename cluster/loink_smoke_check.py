# Run from the laptop: ssh tcohn@txe1-login.mit.edu 'python3 -' < cluster/loink_smoke_check.py
"""Smoke-pass check over published LOINKSMOKE run dirs (stdlib only; runs on the login node).
Pass = every item has summary.json, family loink, no dead arm (mean wall >> 1 s), paired
start_q_error ~0 with clip_distance recorded, and reports the success/timeout tallies."""
import json, glob, os, sys, math
root = os.path.expanduser("~/learned-ik/results/soft12/benchmark")
dirs = sorted(glob.glob(f"{root}/sc_LOINKSMOKE_*"))
bad = 0
for d in dirs:
    p = f"{d}/summary.json"
    if not os.path.exists(p):
        print("NO SUMMARY", os.path.basename(d)); bad += 1; continue
    s = json.load(open(p)); m = s["metadata"]; fam = (m.get("chart_arch") or {}).get("family")
    recs = [c for v in s["records"].values() for c in (v if isinstance(v, list) else [v])]
    cells = [c for c in recs if isinstance(c, dict)]
    sq = max((c.get("start_q_error") or 0.0) for c in cells) if cells else float("nan")
    cd = [c.get("clip_distance") for c in cells]
    a = s["summary"].get("learned", {})
    wall = a.get("mean_wall_time", float("nan"))
    tag = os.path.basename(d).replace("sc_LOINKSMOKE_soft12_", "")
    flags = []
    if fam != "loink": flags.append(f"family={fam}")
    if not (wall > 1.0): flags.append(f"dead arm wall={wall}")
    if "paired" in tag and not (sq < 1e-6): flags.append(f"start_q_error={sq:.2e}")
    if any(x is None for x in cd): flags.append("clip_distance missing")
    bad += bool(flags)
    print(f"{tag:48s} n={a.get('n')} succ={a.get('successes')} timeouts={a.get('timeouts')} "
          f"wall={wall:6.1f} sqe={sq:.1e} clip={max(x or 0 for x in cd):.2f} fails={a.get('fail_reasons')} "
          f"{'FLAGS: ' + '; '.join(flags) if flags else 'ok'}")
print(f"{len(dirs)} dirs, {bad} flagged (48 expected)")
