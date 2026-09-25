#!/bin/bash
# Queue a block of dataset builds as a Slurm dependency chain: exactly ONE runs at a time.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Usage, from the repo root:
#   LEARNED_IK_ROOT=$HOME/learned-ik-helix bash cluster/chain_datasets.sh helix7_p050 helix7_p000
#
# WHY A CHAIN AND NOT A BATCH. ikflow's end-of-run summary walks the WHOLE dataset
# directory and torch.load()s every tensor it finds, so a sibling build's half-written
# file makes a FINISHED job exit 1 with its own data correct on disk. The data is fine;
# the missing `.DONE` sentinel is not, and train_flow.sh hard-fails without one. The soft
# arm measured this. Serialising is the fix, and a dependency chain serialises without a
# session attached.
#
# `afterok`, not `afterany` -- the opposite of chain_ladder.sh, deliberately. A training
# rung that dies should not stall the rungs behind it, because each is an independent
# measurement. A dataset build that dies is almost always a code or environment fault that
# every later rung would hit too, so burning three more nodes on it helps nobody.
#
# ONE NODE. --cpus-per-task is a whole xeon-p8 node's 48 cores, and the chain means one
# job at a time, so this occupies exactly one node of the account's 8-node xeon-p8 group
# cap however many rungs are listed.
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${LEARNED_IK_ROOT:?set LEARNED_IK_ROOT to the cluster tree for this campaign}"
SIZE="${DATASET_SIZE:-25000000}"
SEED="${DATASET_SEED:-0}"
PART="${PARTITION:-xeon-p8}"
WALL="${WALL:-04:00:00}"
CPUS="${CPUS:-48}"
[ $# -gt 0 ] || { echo "usage: chain_datasets.sh <robot> [robot...]" >&2; exit 2; }

PREV=""
for robot in "$@"; do
    DEP=()
    [ -n "$PREV" ] && DEP=(--dependency="afterok:$PREV")
    out=$(sbatch --partition="$PART" --time="$WALL" --cpus-per-task="$CPUS" \
                 --job-name="lik_ds_${robot}" \
                 --output="./lik_ds_${robot}.log-%j" \
                 "${DEP[@]}" \
                 --export="ALL,LEARNED_IK_ROOT=$ROOT,DATASET_ROBOT=$robot,DATASET_SIZE=$SIZE,DATASET_SEED=$SEED" \
                 ./cluster/build_dataset_job.sh 2>&1)
    echo "$out"
    jobid=$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | grep -oE '[0-9]+$')
    [ -n "$jobid" ] || { echo "REFUSING to continue: no job id parsed for $robot" >&2; exit 4; }
    echo "  queued $robot as $jobid${PREV:+ (after $PREV)}"
    PREV="$jobid"
done

echo
echo "chain queued into $ROOT; last job: $PREV"
echo "Sentinels to poll: \$ROOT/home/.cache/ikflow/datasets/<robot>/.DONE"
