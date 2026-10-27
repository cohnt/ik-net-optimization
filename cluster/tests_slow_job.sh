#!/bin/bash
# The tests' SLOW tier (tests/_tiers.py), on one V100: every file's `all` (fast + slow),
# one process at a time, a TSV of seconds and exit codes. ~1 h; not a measurement, so its
# seconds are for budgeting the tier, never compared with anything.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit from ~/learned-ik/repo on the login node (the GPU partition: the slow tier holds the
# CUDA-graph and compiled paths and test_resume_e2e's record flags; it is not debug-gpu work):
#     LLsub ./cluster/tests_slow_job.sh -g volta:2 -s 40 -q xeon-g6-volta -T 2:00:00 -J lik_tests_slow
#
# Read: $ROOT/tests_slow/<stamp>/times.tsv (file, rc, seconds) and one log per file.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/tests_slow/$STAMP"
mkdir -p "$OUT"
exec > >(tee "$OUT/job.log") 2>&1

export HOME="${TMPDIR:-/tmp}/home.tests_slow"
mkdir -p "$HOME/.cache"
ln -sfn "$ROOT/home/.cache/ikflow" "$HOME/.cache/ikflow" 2>/dev/null || true
ln -sfn "$ROOT/home/.cache/drake" "$HOME/.cache/drake" 2>/dev/null || true
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 GVS_ARM_XLA_THREADS=1
# One device, as every job script pins it (jrl otherwise polls nvml for the "least used" GPU).
export CUDA_VISIBLE_DEVICES="$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl.tests_slow"
mkdir -p "$MPLCONFIGDIR"
PY="$ROOT/venv/bin/python"
cd "$REPO" || { echo "no repo at $REPO"; exit 1; }

hostname; date -Is; nvidia-smi -L
echo "staged commit: $(cat .staged-commit 2>/dev/null || echo UNKNOWN)"

printf "file\trc\tseconds\n" > "$OUT/times.tsv"
FAILED=0
for f in tests/test_*.py; do
    name="$(basename "$f" .py)"
    t0=$(date +%s)
    "$PY" -u "$f" all > "$OUT/$name.log" 2>&1
    rc=$?
    printf "%s\t%s\t%s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" | tee -a "$OUT/times.tsv"
    [ "$rc" -eq 0 ] || FAILED=$((FAILED + 1))
done
echo "===== tests_slow complete at $(date -Is): $FAILED file(s) failed; $OUT/times.tsv ====="
