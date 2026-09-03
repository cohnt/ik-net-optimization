#!/bin/bash
# One-time (and safely repeatable) compaction of the per-cell solver logs already on
# the cluster, into one `solver_logs.tar.gz` per run directory.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Why: src/benchmark.py used to write one ~20 KB IPOPT log per (cell x arm) straight
# onto the shared filesystem. By Stage G that was 35,596 files -- 87% of every
# collection's file count and 78% of its bytes -- and it is exactly the many-small-files
# pattern SuperCloud's guidance warns about, which is metadata-bound rather than
# bandwidth-bound on Lustre. The benchmark now writes them to node-local $TMPDIR and
# rolls them up itself; this script fixes the runs that predate that change.
#
#   cluster/compact_logs.sh --dry-run   report what would be compacted, change nothing
#   cluster/compact_logs.sh             compact, removing the loose logs it archived
#
# Safety properties:
#   - It refuses to run while any job is active, so it cannot race a run that is still
#     writing logs into a directory it is archiving.
#   - It archives first and removes only the files it successfully added, per directory.
#   - A directory that already has solver_logs.tar.gz is folded into that archive rather
#     than overwriting it, so re-running is safe and idempotent.
#   - Nothing outside a benchmark run directory is touched, and no summary.json is read
#     or written.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

sc_run "cd ~/$SC_ROOT || exit 1
RUNNING=\$(LLstat 2>/dev/null | grep -c RUNNI || true)
if [ \"\$RUNNING\" -gt 0 ] && [ $DRY -eq 0 ]; then
    echo \"REFUSING: \$RUNNING job(s) running -- a live run may still be writing logs.\"
    exit 1
fi

TOTAL_FILES=0; TOTAL_DIRS=0
for d in results/*/benchmark/*/; do
    [ -d \"\$d\" ] || continue
    N=\$(find \"\$d\" -maxdepth 1 -name '*.txt' | wc -l)
    [ \"\$N\" -eq 0 ] && continue
    TOTAL_DIRS=\$((TOTAL_DIRS+1)); TOTAL_FILES=\$((TOTAL_FILES+N))
    if [ $DRY -eq 1 ]; then continue; fi
    # Fold into any existing archive rather than clobbering it: -r appends, and an
    # absent archive is created by -c. Both need the plain .tar, so gzip separately.
    A=\"\$d/solver_logs.tar\"
    [ -f \"\$A.gz\" ] && gunzip -f \"\$A.gz\"
    if [ -f \"\$A\" ]; then
        tar rf \"\$A\" -C \"\$d\" \$(cd \"\$d\" && ls *.txt) 2>/dev/null
    else
        tar cf \"\$A\" -C \"\$d\" \$(cd \"\$d\" && ls *.txt) 2>/dev/null
    fi
    if [ \$? -eq 0 ] && [ -f \"\$A\" ]; then
        gzip -f \"\$A\" && find \"\$d\" -maxdepth 1 -name '*.txt' -delete
    else
        echo \"WARNING: archiving failed for \$d -- loose logs left in place\"
    fi
done
if [ $DRY -eq 1 ]; then
    echo \"DRY RUN: \$TOTAL_FILES loose logs across \$TOTAL_DIRS run directories would be compacted\"
else
    echo \"compacted \$TOTAL_FILES loose logs across \$TOTAL_DIRS run directories\"
    echo \"remaining loose .txt under results/: \$(find results -name '*.txt' | wc -l)\"
    echo \"total files under results/ now:     \$(find results -type f | wc -l)\"
fi"
