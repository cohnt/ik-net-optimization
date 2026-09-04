#!/bin/bash
# Submit a manifest to xeon-g6-volta, after checking it can actually be run.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
#   cluster/submit.sh <manifest-basename> [n_jobs]
#   cluster/submit.sh manifest_stageEQ4base.txt 4
#
# Why this exists: a manifest was generated, committed and submitted without being
# staged to the cluster first. The jobs started, found no manifest, and exited silently
# -- the stage looked "submitted" and produced nothing, and it was only noticed when the
# poller reported 0 of 16 done with no jobs left. So this verifies the manifest exists
# REMOTELY before submitting, and refuses rather than launching jobs that cannot work.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"

MANIFEST="${1:?usage: submit.sh <manifest-basename> [n_jobs]}"
NJOBS="${2:-4}"
LOCAL="$(_sc_repo_root)/cluster/$MANIFEST"
[ -f "$LOCAL" ] || { echo "no such local manifest: $LOCAL"; exit 1; }
echo "local:  $MANIFEST ($(grep -vc '^#' "$LOCAL") items)"

sc_run "cd ~/$SC_ROOT/repo || exit 1
if [ ! -f cluster/$MANIFEST ]; then
    echo 'REFUSING: cluster/$MANIFEST is not on the cluster -- run cluster/stage_code.sh first.'
    exit 1
fi
echo \"remote: \$(grep -vc '^#' cluster/$MANIFEST) items\"
for i in \$(seq 1 $NJOBS); do
    MANIFEST=cluster/$MANIFEST PROCS=\${PROCS:-8} LLsub ./cluster/run_items.sh \\
        -g volta:2 -s 40 -q xeon-g6-volta -T 12:00:00 2>&1 | grep Submitted
done"
