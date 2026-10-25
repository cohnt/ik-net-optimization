#!/bin/bash
# LLsub payload: the LOInK branch reach ceiling (scripts/loink_branch_ceiling.py) for ONE robot x
# task, on a CPU node. How many of the probe targets (guess 0 of targets 0..19 of the record grid)
# any configuration on the nominal IK branch can solve at all -- the ceiling a single-branch
# LOInK chart (phase 2) is read against -- and the same count over all eight analytic branches.
# Analytic only: it never touches the GPU, so the CPU partition (xeon-p8) is the right one.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node), one job per robot x task:
#   ROBOT=panda TASK=pose LLsub ./cluster/ceiling_job.sh -s 8 -q xeon-p8 -T 02:00:00 -J lik_ceiling
#   ROBOT=panda TASK=mug  LLsub ./cluster/ceiling_job.sh -s 8 -q xeon-p8 -T 02:00:00 -J lik_ceiling
#   ROBOT=iiwa  TASK=pose LLsub ./cluster/ceiling_job.sh -s 8 -q xeon-p8 -T 02:00:00 -J lik_ceiling
#   ROBOT=iiwa  TASK=mug  LLsub ./cluster/ceiling_job.sh -s 8 -q xeon-p8 -T 02:00:00 -J lik_ceiling
# Optional: TARGETS (default 20, LOINK_PROBE_TARGETS), EXTRA (more script flags, e.g.
# "--psi-steps 720 --yaw-steps 72").
# Output: $ROOT/results/loink_branch_ceiling/<robot>_<task>.json and <robot>_<task>_<job id>.log.
# The script exits 3 if the grid it drew does not hash to the record's grid_hash.
#
# NOTE: source /etc/profile BEFORE -u -- Z97-byobu.sh reads an unset LC_BYOBU.
source /etc/profile
set -uo pipefail

echo "ceiling_job: alive on $(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-?}"
ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
ROBOT="${ROBOT:?set ROBOT=panda|iiwa}"
TASK="${TASK:?set TASK=pose|mug}"
TARGETS="${TARGETS:-20}"
EXTRA="${EXTRA:-}"
case "$ROBOT" in panda|iiwa) ;; *) echo "FATAL: ROBOT=$ROBOT" >&2; exit 4 ;; esac
case "$TASK" in pose|mug) ;; *) echo "FATAL: TASK=$TASK" >&2; exit 4 ;; esac
LOG_DIR="$ROOT/results/loink_branch_ceiling"
mkdir -p "$LOG_DIR"

export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/mpl"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
## The nightly Drake pin, as run_items.sh's default (non-crc) env and cluster/test_job.sh. HOME is
## $ROOT/home, so the warmed drake_models / ikflow caches are in place (compute nodes cannot
## download them).
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/venv/bin/python"

cd "$REPO" || exit 4
SCRIPT=scripts/loink_branch_ceiling.py
[ -f "$SCRIPT" ] || { echo "FATAL: no $SCRIPT under $REPO" >&2; exit 4; }
log="$LOG_DIR/${ROBOT}_${TASK}_${SLURM_JOB_ID:-local}.log"
echo "running $SCRIPT --robot $ROBOT --task $TASK --targets $TARGETS $EXTRA at staged commit" \
     "$(cat .staged-commit 2>/dev/null || echo UNKNOWN) -> $log"
# shellcheck disable=SC2086  # EXTRA is deliberately word-split into flags
LEARNED_IK_ROOT="$ROOT" "$PY" -u "$SCRIPT" --robot "$ROBOT" --task "$TASK" --targets "$TARGETS" \
    --json_out "$LOG_DIR/${ROBOT}_${TASK}.json" $EXTRA 2>&1 | tee "$log"
RC=${PIPESTATUS[0]}
echo "ceiling_job: $ROBOT $TASK rc=$RC date=$(date -Is)"
exit "$RC"
