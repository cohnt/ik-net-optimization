#!/bin/bash
# Collect results from Engaging into the laptop's results/ (DATA ONLY; storage there is not backed up).
#
#   bash cluster/engaging/collect.sh                 collect the staged commit's results (and kill/hardware logs)
#   bash cluster/engaging/collect.sh --reclaim <manifest-basename>
#                                                    clear claims that have no .done marker, once NO lik_bench job
#                                                    of that manifest is queued or running (a job that died WITHOUT
#                                                    a requeue: walltime, node failure, scancel; a requeued job
#                                                    releases its own, job_lib.sh)
#
# Security: only data crosses back. The rsync is an ALLOWLIST (*.json *.jsonl *.npz *.csv *.txt *.log and
# directories); code, notebooks, instruction files and pickles never match, and nothing copied back is run or
# sourced. Arrays loaded from it use np.load(..., allow_pickle=False). Like SuperCloud's collection it ends in
# merge_shard_summaries.py and condense_staging.py --apply (staging is temporary; readers read results/ only).
set -uo pipefail
source "$(dirname "$0")/eg_common.sh" || exit 1
eg_require_connection || exit 255
SDIR="$EG_REPO/logs/staged"
COMMIT="${COMMIT:-$(cat "$SDIR/engaging-last-commit" 2>/dev/null)}"
[ -n "$COMMIT" ] || { echo "no staged commit"; exit 2; }
CROOT=$(eg_commit_root "$COMMIT")
PY="${LIK_PY:-$EG_REPO/.venv/bin/python}"; [ -x "$PY" ] || PY=python3

if [ "${1:-}" = "--reclaim" ]; then
    M="${2:?usage: --reclaim <manifest basename, no .txt>}"
    [[ "$M" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad manifest name" >&2; exit 2; }
    eg_run "cd ~/$CROOT/state/$M 2>/dev/null || { echo 'no such manifest state'; exit 1; }
BUSY=\$(squeue --me -h -t RUNNING,PENDING -o '%j' | grep -c '^lik_bench_')
[ \"\$BUSY\" -eq 0 ] || { echo \"REFUSING: \$BUSY lik_bench task(s) queued or running -- a live item must not be stolen\"; exit 1; }
N=0; for c in *.claim; do [ -d \"\$c\" ] || continue; id=\${c%.claim}
  if [ ! -f \"\$id.done\" ]; then rm -rf \"\$c\"; N=\$((N+1)); echo \"reclaimed \$id\"; fi; done
echo \"\$N stale claim(s) cleared; re-submit the same manifest to mop up.\""
    exit $?
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
STAGING="$EG_REPO/results/_cluster_staging/$STAMP"
LOGDEST="$EG_REPO/results/_cluster_logs/engaging/${CROOT##*/}"
mkdir -p "$STAGING" "$LOGDEST"
ALLOW=(--include='*/' --include='*.json' --include='*.jsonl' --include='*.npz' --include='*.csv' --include='*.txt' --include='*.log' --exclude='*')
eg_rsync -a --prune-empty-dirs "${ALLOW[@]}" "$EG_DEST:$CROOT/results/" "$STAGING/" || { echo "rsync of results failed" >&2; exit 3; }
## state/<manifest>/{kills,hardware,submissions}.log and the jobs' own logs, whole (append-only, tiny).
eg_rsync -a --prune-empty-dirs --include='*/' --include='kills.log' --include='hardware.log' --include='submissions.log' \
    --include='*.SENTINEL' --exclude='*' "$EG_DEST:$CROOT/state/" "$LOGDEST/state/" || echo "WARNING: state logs not fetched" >&2
eg_rsync -a --include='*.out' --exclude='*' "$EG_DEST:$CROOT/logs/" "$LOGDEST/jobs/" || echo "WARNING: job logs not fetched" >&2
echo "collected to $STAGING; logs to $LOGDEST"
echo "hardware recorded:"; cat "$LOGDEST"/state/*/hardware.log 2>/dev/null | sed 's/ job=[0-9]*//; s/ [0-9T:-]*$//' | sort | uniq -c

ALSO=()
while IFS= read -r d; do [ -n "$d" ] && ALSO+=(--also "$d"); done < <(ls -1d "$EG_REPO"/results/_cluster_staging/[0-9]*-[0-9]*/ 2>/dev/null | grep -v "^$STAGING/\?$")
"$PY" "$EG_REPO/cluster/merge_shard_summaries.py" "$STAGING" "${ALSO[@]}"
"$PY" "$EG_REPO/cluster/condense_staging.py" --apply
