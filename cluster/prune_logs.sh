#!/bin/bash
# Delete solver logs whose findings have already been written up.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Thomas's retention rule: "once obsolete or mined, they can be deleted, since we can
# always recreate them and you have saved their knowledge." Two properties make that
# safe, and both hold here:
#
#   - the artifacts are REPRODUCIBLE -- the grid is seeded and hashed, so any cell can
#     be re-solved bit for bit from its tag and `--cells`;
#   - their findings are RECORDED -- every table in CLAUDE.md is derived and written up
#     before the logs behind it are dropped.
#
# What is never deleted is `summary.json`: it carries the per-cell records, the returned
# configuration `q`, every binding's violation and all the statistics, and a
# wall-clock-capped run would NOT reproduce bit for bit. It is the primary record; the
# logs are traces of the same solves. A `summary.json.partial` with no completed sibling
# is the only record of a crashed run and is kept for the same reason.
#
#   cluster/prune_logs.sh --dry-run   report what would go, change nothing
#   cluster/prune_logs.sh             prune locally
#   cluster/prune_logs.sh --cluster   also prune the cluster's copy (read the note below)
#
# --cluster is deliberately separate and not the default: the cluster copy is usually the
# only remaining one once the local prune has run, and cluster storage is not backed up.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DRY=0; CLUSTER=0
for a in "$@"; do
    case "$a" in
    --dry-run) DRY=1 ;;
    --cluster) CLUSTER=1 ;;
    *) echo "unknown flag: $a"; exit 1 ;;
    esac
done

count() { find results -name "$1" -type f 2>/dev/null | wc -l; }
bytes() { find results -name "$1" -type f -printf '%s\n' 2>/dev/null \
              | awk '{s+=$1} END {printf "%.0f", s/1048576}'; }

echo "local results/: $(du -sh results 2>/dev/null | cut -f1)"
printf "  loose .txt logs      %6s files  %5s MB\n" "$(count '*.txt')" "$(bytes '*.txt')"
printf "  solver_logs.tar.gz   %6s files  %5s MB\n" "$(count 'solver_logs.tar.gz')" "$(bytes 'solver_logs.tar.gz')"
printf "  summary.json (KEPT)  %6s files  %5s MB\n" "$(count 'summary.json')" "$(bytes 'summary.json')"
STAGING=$(du -sh results/_cluster_staging 2>/dev/null | cut -f1)
[ -n "$STAGING" ] && echo "  _cluster_staging     $STAGING (duplicate of promoted runs)"

if [ "$DRY" = "1" ]; then echo "DRY RUN: nothing removed"; exit 0; fi

rm -rf results/_cluster_staging
find results -name '*.txt' -type f -delete
find results -name 'solver_logs.tar.gz' -type f -delete
find results -type d -empty -delete 2>/dev/null
echo "pruned. local results/ now $(du -sh results | cut -f1), $(count 'summary.json') summaries kept"

if [ "$CLUSTER" = "1" ]; then
    source "$(dirname "$0")/ssh_common.sh"
    # Same worker-scoped guard as compact_logs.sh: the account is shared with Thomas's
    # other projects, so never ask whether ANY job is running.
    sc_run "cd ~/$SC_ROOT || exit 1
BUSY=\$(squeue -u \$USER -h -n run_items.sh -t RUNNING,PENDING 2>/dev/null | wc -l)
if [ \"\$BUSY\" -gt 0 ]; then echo 'REFUSING: run_items.sh queued or running'; exit 1; fi
N=\$(find results -name 'solver_logs.tar.gz' | wc -l)
find results -name 'solver_logs.tar.gz' -delete
find results -name '*.txt' -delete
echo \"cluster: removed \$N log archives; results/ now \$(du -sh results | cut -f1), \$(find results -name summary.json | wc -l) summaries kept\""
fi
