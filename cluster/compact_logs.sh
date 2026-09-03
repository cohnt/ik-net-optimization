#!/bin/bash
# Compact the per-cell solver logs already on the cluster into one
# `solver_logs.tar.gz` per run directory. Repeatable and idempotent.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Why: src/benchmark.py used to write one ~20 KB IPOPT log per (cell x arm) straight
# onto the shared filesystem. By Stage G that was 35,596 files -- 87% of every
# collection's file count and 78% of its bytes -- which is the many-small-files pattern
# SuperCloud's guidance warns about and is metadata-bound rather than bandwidth-bound on
# Lustre. The benchmark now writes them to node-local $TMPDIR and rolls them up itself;
# this fixes the runs that predate that change.
#
#   cluster/compact_logs.sh --dry-run   report what would be compacted, change nothing
#   cluster/compact_logs.sh             submit the compaction as a debug-cpu job
#   cluster/compact_logs.sh --status    has the job finished?
#
# The compaction itself runs INSIDE A JOB (cluster/compact_logs_job.sh), never on the
# login node: compressing several hundred megabytes is exactly the work the login nodes
# are not for. Only the dry run and the status check are done over ssh, and both are
# read-only.
#
# Safety: it refuses to submit while a benchmark job is active (a live run may still be
# writing into a directory it would archive); it archives before removing, and removes
# only the names it archived; and it folds into an existing archive rather than
# overwriting, so a re-run after a partial pass cannot destroy earlier work.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"

case "${1:-}" in
--dry-run)
    sc_run "cd ~/$SC_ROOT || exit 1
N=\$(find results -name '*.txt' | wc -l)
D=\$(find results -name '*.txt' -printf '%h\n' | sort -u | wc -l)
B=\$(find results -name '*.txt' -printf '%s\n' | awk '{s+=\$1} END {printf \"%.0f\", s/1048576}')
echo \"DRY RUN: \$N loose logs (\$B MB) across \$D run directories would be compacted\"
echo \"total files under results/ now: \$(find results -type f | wc -l)\""
    exit 0
    ;;
--status)
    sc_run "cd ~/$SC_ROOT || exit 1
if [ -f compact_logs.DONE ]; then echo \"DONE at \$(stat -c %y compact_logs.DONE)\"; else echo 'not finished'; fi
echo \"loose .txt remaining: \$(find results -name '*.txt' | wc -l)\"
echo \"total files under results/: \$(find results -type f | wc -l)\"
ls -t compact_logs_job.sh.log-* 2>/dev/null | head -1 | xargs -r tail -6"
    exit 0
    ;;
esac

# Only this project's workers can be writing into ~/learned-ik/results, and the
# SuperCloud account is shared with Thomas's other projects -- a broad any-job-running
# check refuses whenever an unrelated campaign is on the cluster, which is most of the
# time (it fired on a run_matrix.sh job belonging to another project). Filter by the
# worker script's job name, and count PENDING too: a queued run_items.sh could start
# part way through and write into a directory already archived.
sc_run "cd ~/$SC_ROOT/repo || exit 1
BUSY=\$(squeue -u \$USER -h -n run_items.sh -t RUNNING,PENDING 2>/dev/null | wc -l)
if [ \"\$BUSY\" -gt 0 ]; then
    echo \"REFUSING: \$BUSY run_items.sh job(s) queued or running -- a live run may still be writing logs.\"
    exit 1
fi
rm -f ~/$SC_ROOT/compact_logs.DONE
cd ~/$SC_ROOT && LLsub ~/$SC_ROOT/repo/cluster/compact_logs_job.sh -s 8 -q debug-cpu -T 2:00:00"
