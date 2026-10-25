# Run from the laptop: ssh tcohn@txe1-login.mit.edu 'python3 -' < cluster/loink_smoke_check.py
# The single-branch (phase 2) smokes, LOINKSMOKE_branch + LOINKSMOKE_SVGD_branch (96 + 96 items):
#   ssh tcohn@txe1-login.mit.edu 'LOINK_SMOKE_ROBOTS=panda,iiwa LOINK_SMOKE_MATCH=_loinkB LOINK_SMOKE_EXPECTED=192 LOINK_SMOKE_OFF_BRANCH_OK=1 python3 -' < cluster/loink_smoke_check.py
"""Smoke-pass check over published LOINKSMOKE run dirs (stdlib only; runs on the login node).
Pass = every item has summary.json, family loink, no dead arm (mean wall >> 1 s), paired
start_q_error ~0 with clip_distance recorded, and reports the success/timeout tallies.

Environment (defaults reproduce the soft12 `_soft12P` smokes exactly):
  LOINK_SMOKE_ROBOTS    comma-separated results/<robot> dirs to scan (default soft12)
  LOINK_SMOKE_MATCH     only run dirs whose name contains this substring (default: all)
  LOINK_SMOKE_EXPECTED  the item count the last line quotes (default 48 = 24 IPOPT + 24 AL-SVGD);
                        a positional argument (`python3 - 192`) overrides it
  LOINK_SMOKE_OFF_BRANCH_OK=1  single-branch charts: a paired cell scored `unrepresentable_start`
                        (start off the chart's branch, wall 0 by design) is the stated limitation,
                        not a bug -- it is left out of the start_q_error and dead-arm checks and
                        counted (`unrep=`). Default off: every paired cell must start exactly.
The last line is always `<N> dirs, <M> flagged (<E> expected)` (a watcher greps it)."""
import json, glob, os, sys, math
robots = [r for r in os.environ.get("LOINK_SMOKE_ROBOTS", "soft12").split(",") if r]
match = os.environ.get("LOINK_SMOKE_MATCH", "")
off_branch_ok = os.environ.get("LOINK_SMOKE_OFF_BRANCH_OK", "") == "1"
expected = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else int(os.environ.get("LOINK_SMOKE_EXPECTED", "48"))
dirs = sorted(d for robot in robots
              for d in glob.glob(os.path.expanduser(f"~/learned-ik/results/{robot}/benchmark/sc_LOINKSMOKE_*"))
              if match in os.path.basename(d))
bad = 0
for d in dirs:
    p = f"{d}/summary.json"
    if not os.path.exists(p):
        print("NO SUMMARY", os.path.basename(d)); bad += 1; continue
    s = json.load(open(p)); m = s["metadata"]; fam = (m.get("chart_arch") or {}).get("family")
    recs = [c for v in s["records"].values() for c in (v if isinstance(v, list) else [v])]
    cells = [c for c in recs if isinstance(c, dict)]
    unrep = [c for c in cells if c.get("fail_reason") == "unrepresentable_start"]
    solved = [c for c in cells if not (off_branch_ok and c in unrep)]
    sq = max((c.get("start_q_error") or 0.0) for c in solved) if solved else float("nan")
    cd = [c.get("clip_distance") for c in cells]
    a = s["summary"].get("learned", {})
    wall = a.get("mean_wall_time", float("nan"))
    if off_branch_ok and unrep and solved:      # the dead-arm test reads the cells that were solved
        ws = [c["wall_time"] for c in solved if c.get("wall_time") is not None]
        wall = sum(ws) / len(ws) if ws else float("nan")
    tag = os.path.basename(d).replace("sc_LOINKSMOKE_", "")
    if len(robots) == 1:
        tag = tag.replace(f"{robots[0]}_", "", 1)
    flags = []
    if fam != "loink": flags.append(f"family={fam}")
    all_unrep = off_branch_ok and cells and not solved
    if not (wall > 1.0) and not all_unrep: flags.append(f"dead arm wall={wall}")
    if "paired" in tag and not all_unrep and not (sq < 1e-6): flags.append(f"start_q_error={sq:.2e}")
    if any(x is None for x in cd): flags.append("clip_distance missing")
    bad += bool(flags)
    print(f"{tag:48s} n={a.get('n')} succ={a.get('successes')} timeouts={a.get('timeouts')} "
          f"wall={wall:6.1f} sqe={sq:.1e} unrep={len(unrep)} clip={max(x or 0 for x in cd):.2f} fails={a.get('fail_reasons')} "
          f"{'FLAGS: ' + '; '.join(flags) if flags else 'ok'}")
print(f"{len(dirs)} dirs, {bad} flagged ({expected} expected)")
