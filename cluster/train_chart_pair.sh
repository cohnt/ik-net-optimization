#!/bin/bash
# The LLsub payload for TWO chart training runs in ONE job: one run per V100 of one volta node.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit through `cluster/submit_chart_train.sh --pair` (or `--all` with PAIRS=1) ONLY.
#
# WHY THIS EXISTS. The design is one run per GPU, all eight at once. With one single-GPU job per
# run, that needs Slurm to co-schedule two jobs on a node AND the account's GrpTRES node=4 cap to
# count such a node once. If the cap counts each job's node separately -- Slurm's documented
# behaviour for a node limit, to be confirmed on the cluster (cluster/CVAE_RUNBOOK.md) -- eight
# single-GPU jobs run four at a time and half the GPUs idle. A pair job takes one node and both
# its GPUs, counts as one node however the cap counts, and runs two runs.
#
# Each run is cluster/train_chart.sh, unchanged, as a background process with
# CUDA_VISIBLE_DEVICES narrowed to ONE of the job's two GPUs and its own RUN_NAME / ROBOT /
# TRAIN_EXTRA_ARGS. Both runs share this job's FAMILY / BATCH / MAX_STEPS environment; a row
# carrying its own --max_steps= / --batch_size= (the LOInK recipes) overrides them inside
# train_chart.sh, so the two runs of a pair may train different budgets. Narrowing the variable in-process rather than asking srun for per-step
# GPUs keeps this independent of the cluster's step-gres semantics. Each run's output goes to
# its own $RUN_DIR/train.log-<jobid>; this job's own log (the pair's launch.log) carries the two
# exit codes. The job exits with the larger, so a failure in either is the job's.
#
# NOTE: source /etc/profile BEFORE enabling -u -- Z97-byobu.sh reads unset LC_BYOBU.
source /etc/profile
set -uo pipefail

echo "train_chart_pair: alive on $(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-?}"

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
: "${RUN_A:?}" "${ROBOT_A:?}" "${RUN_B:?}" "${ROBOT_B:?}"

## Slurm sets the job's GPUs as a comma-separated list (UUIDs or indices). Exactly two, or the
## pair cannot be one-per-GPU and must not start.
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [ "${#GPUS[@]}" -ne 2 ]; then
    echo "FATAL: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' is not two GPUs" >&2
    exit 2
fi
nvidia-smi -L || true

launch() {   # <gpu> <run> <robot> <extra args>
    local dir="$ROOT/results/train/$2"
    mkdir -p "$dir"
    echo "train_chart_pair: $2 (robot $3) on GPU $1, log $dir/train.log-${SLURM_JOB_ID:-local}"
    CUDA_VISIBLE_DEVICES="$1" RUN_NAME="$2" ROBOT="$3" TRAIN_EXTRA_ARGS="$4" \
        "$REPO/cluster/train_chart.sh" > "$dir/train.log-${SLURM_JOB_ID:-local}" 2>&1
}

launch "${GPUS[0]}" "$RUN_A" "$ROBOT_A" "${ARGS_A:-}" &
PID_A=$!
launch "${GPUS[1]}" "$RUN_B" "$ROBOT_B" "${ARGS_B:-}" &
PID_B=$!
wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "train_chart_pair: $RUN_A rc=$RC_A, $RUN_B rc=$RC_B $(date -Is)"
exit $(( RC_A > RC_B ? RC_A : RC_B ))
