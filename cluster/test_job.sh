#!/bin/bash
# LLsub payload: run ONE test file of the staged tree on a CPU node, so a long invariant file
# (minutes to half an hour) never runs on the laptop (Thomas, 2026-10-23: a 10-30 minute test is
# too expensive locally). The file's own output is the verdict (its last line, ALL PASS).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node):
#   TEST=tests/test_loink_chart.py LLsub ./cluster/test_job.sh -s 8 -q xeon-p8 -T 01:00:00 -J lik_test
# Log: $ROOT/results/tests/<file stem>_<job id>.log.
#
# NOTE: source /etc/profile BEFORE -u -- Z97-byobu.sh reads an unset LC_BYOBU.
source /etc/profile
set -uo pipefail

echo "test_job: alive on $(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-?}"
ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
TEST="${TEST:?set TEST, a path relative to the staged repo, e.g. tests/test_loink_chart.py}"
LOG_DIR="$ROOT/results/tests"
mkdir -p "$LOG_DIR"

export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/mpl"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/venv/bin/python"

cd "$REPO" || exit 4
[ -f "$TEST" ] || { echo "FATAL: no $TEST under $REPO" >&2; exit 4; }
log="$LOG_DIR/$(basename "$TEST" .py)_${SLURM_JOB_ID:-local}.log"
echo "running $TEST at staged commit $(cat .staged-commit 2>/dev/null || echo UNKNOWN) -> $log"
LEARNED_IK_ROOT="$ROOT" "$PY" -u "$TEST" 2>&1 | tee "$log"
RC=${PIPESTATUS[0]}
echo "test_job: $TEST rc=$RC date=$(date -Is)"
exit "$RC"
