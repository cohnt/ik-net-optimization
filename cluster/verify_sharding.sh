#!/bin/bash
# Prove that `--shard K/N` + cluster/merge_shard_summaries.py is a no-op.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Runs entirely on the local machine -- no cluster involved -- and takes a couple
# of minutes. Run it after ANY change to the sharding path, the merger, or the
# grid construction, and before submitting a sharded campaign.
#
# The solves are bounded by `--set max_iter`, not by the wall clock, ON PURPOSE.
# A wall-clock-capped solve stops wherever the clock runs out, so its iterate
# count (and returned point) legitimately vary with machine load -- the repo's
# documented +-1-cell reproducibility. That is a property of the cap, not of
# sharding, and mixing the two makes this test unable to fail informatively.
# Under an iteration cap the computation is deterministic, so anything that
# differs here is a real defect in the shard/merge path.
#
# Usage: bash cluster/verify_sharding.sh [shards]
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
N="${1:-2}"
export TQDM_DISABLE=1 MPLBACKEND=Agg
COMMON=(--task pose --targets 2 --guesses 2 --wall-time 600 --set max_iter=40
        --arms learned,numerical,analytic --config latent --seed 0)

rm -rf results/panda/benchmark/_shardcheck_ref results/panda/benchmark/_shardcheck_sh*
echo "== reference run (unsharded)"
"$PY" -u scripts/panda/panda_benchmark.py "${COMMON[@]}" --tag _shardcheck_ref >/dev/null || exit 1
for ((k = 0; k < N; k++)); do
    echo "== shard $k/$N"
    "$PY" -u scripts/panda/panda_benchmark.py "${COMMON[@]}" --tag _shardcheck_sh \
        --shard "$k/$N" >/dev/null || exit 1
done
rm -rf results/panda/benchmark/_shardcheck_sh
"$PY" cluster/merge_shard_summaries.py results/panda/benchmark --only _shardcheck_sh \
    | grep -E "_shardcheck_sh:|wrote" || exit 1

"$PY" - <<'PYEOF'
import json, math, sys
ref = json.load(open("results/panda/benchmark/_shardcheck_ref/summary.json"))
mrg = json.load(open("results/panda/benchmark/_shardcheck_sh/summary.json"))
# Per-process timings cannot match across processes and are not what sharding
# must preserve. Everything else -- including the returned configuration q and
# every paired statistic -- must be identical.
TIMING = {"wall_time", "setup_time", "solver_seconds",
          "mean_wall_time", "mean_wall_time_success", "mean_setup_time"}

def eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True                       # nan != nan, but "both absent" is agreement
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        return all(eq(a[k], b[k]) for k in a)
    return a == b

bad = 0
arms = [k for k in ref["summary"] if not k.startswith("_")]
if arms != [k for k in mrg["summary"] if not k.startswith("_")]:
    print("  ARM ORDER DIFFERS -- _mcnemar pairs would read the other way round")
    bad += 1
for arm in arms:
    a = {(r["target"], r["guess"]): r for r in ref["records"][arm]}
    b = {(r["target"], r["guess"]): r for r in mrg["records"][arm]}
    if set(a) != set(b):
        print(f"  {arm}: cell sets differ"); bad += 1; continue
    for cell in sorted(a):
        for k in set(a[cell]) | set(b[cell]):
            if k not in TIMING and not eq(a[cell].get(k), b[cell].get(k)):
                print(f"  DIFF {arm} {cell}.{k}: {a[cell].get(k)!r} vs {b[cell].get(k)!r}")
                bad += 1
    for k in ref["summary"][arm]:
        if k not in TIMING and not eq(ref["summary"][arm][k], mrg["summary"][arm][k]):
            print(f"  SUMMARY DIFF {arm}.{k}"); bad += 1
    print(f"  {arm:<11} {len(a)} cells: every record field (incl. q) and every statistic")
for k in ("_mcnemar", "_common_cells", "_common"):
    if not eq(ref["summary"][k], mrg["summary"][k]):
        print(f"  {k} DIFFERENT"); bad += 1
if ref["metadata"]["grid_hash"] != mrg["metadata"]["grid_hash"]:
    print("  grid_hash differs"); bad += 1
print()
print("SHARDING OK -- identical apart from per-process timings" if not bad
      else f"SHARDING BROKEN -- {bad} difference(s)")
sys.exit(1 if bad else 0)
PYEOF
