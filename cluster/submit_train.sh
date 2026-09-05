#!/bin/bash
# Submit (or resume) the iiwa14 retraining job -- one command, idempotent.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Usage:
#   bash cluster/submit_train.sh <run_name> [nnodes] [walltime] [-- extra train args]
# e.g.
#   bash cluster/submit_train.sh iiwa14_ddp_r1 4 96:00:00
#   bash cluster/submit_train.sh smoke2n 2 00:20:00 -- --max_steps=200 --eval_every=100 \
#       --val_set_size=20 --checkpoint_every=100 --pole_eval_n=500 --disable_wandb
#
# Refuses if a lik_train job for ANY run is already RUNNING or PENDING: a second
# job on the same RUN_DIR would race its checkpoints (with <4 nodes in use it
# would START, not queue), and the volta group cap is shared with Thomas's other
# work. Scoped to this project's own job names -- the account is shared across
# projects (cluster/README, guard-scoping lesson).
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"

RUN_NAME="${1:?usage: submit_train.sh <run_name> [nnodes] [walltime] [-- extra args]}"
NNODES="${2:-4}"
WALL="${3:-96:00:00}"
shift $(( $# > 3 ? 3 : $# ))
[ "${1:-}" = "--" ] && shift
EXTRA_ARGS="$*"

LIVE=$(sc_run 'LLstat 2>/dev/null | grep -c "lik_train"' 2>/dev/null | tr -dc '0-9')
if [ -n "${LIVE:-}" ] && [ "${LIVE:-0}" -gt 0 ]; then
    echo "REFUSING: $LIVE lik_train job(s) already RUNNING/PENDING." >&2
    echo "Two jobs on one RUN_DIR race the checkpoints. LLkill the old one or wait." >&2
    exit 3
fi

echo "Submitting: RUN_NAME=$RUN_NAME NNODES=$NNODES WALL=$WALL EXTRA='$EXTRA_ARGS'"
sc_run "cd ~/$SC_ROOT/repo && \
  RUN_NAME='$RUN_NAME' NNODES=$NNODES TRAIN_EXTRA_ARGS='$EXTRA_ARGS' \
  LLsub ./cluster/train_iiwa.sh \"[$NNODES,1,40]\" -g volta:2 -q xeon-g6-volta \
    -T $WALL -J 'lik_train_$RUN_NAME'"
