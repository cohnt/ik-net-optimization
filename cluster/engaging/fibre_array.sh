#!/bin/bash
# Engaging Slurm array payload for the IK-fibre investigation (scripts/fibre/): one
# single-threaded CPU process per array task, inside the CPU image (learned_ik_cpu.def).
# Array task i runs line i+1 of MANIFEST: "<robot> <seed> <targets> [extra args...]".
#
#   MANIFEST=$HOME/learned-ik/fibre/conn.txt TAG=gvs_conn sbatch -p mit_normal -c 1 \
#       --mem=3900M -t 06:00:00 -a 0-95 -J lik_fibre_conn cluster/engaging/fibre_array.sh
#
# Output: $ROOT/scratch-fibre/$TAG/<robot>.s<seed>.{json,log}, read by summarise_fibre.py.
# Safe under --requeue: a restarted task simply reruns its line and overwrites its files.
#SBATCH -o /home/tcohn/learned-ik/logs/%x-%A_%a.out
set -uo pipefail
ROOT="${LIK_ROOT:-$HOME/learned-ik}"
MANIFEST="${MANIFEST:?set MANIFEST}"; TAG="${TAG:?set TAG}"
OUT_DIR="$ROOT/scratch-fibre/$TAG"; mkdir -p "$OUT_DIR"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[ -n "$LINE" ] || { echo "no manifest line $SLURM_ARRAY_TASK_ID"; exit 2; }
read -r ROBOT SEED TARGETS EXTRA <<< "$LINE"
JT="/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p -m 700 "$JT/home"; trap 'rm -rf "$JT"' EXIT
# Every JAX-importing process owns ~100 idle XLA threads; many tasks per node can exhaust
# the default `ulimit -u` (seen on SuperCloud, 2026-11-09).
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
module load apptainer/1.4.2 >/dev/null 2>&1 || true
echo "$(date -Is) $(hostname) task $SLURM_ARRAY_TASK_ID: $ROBOT seed $SEED targets $TARGETS $EXTRA (staged $(cat "$ROOT/repo/.staged-commit" 2>/dev/null))"
# shellcheck disable=SC2086
apptainer exec --cleanenv --home "$JT/home" -B "$ROOT" \
    --env TMPDIR="$JT" --env LIK_REPO="$ROOT/repo" --env PYTHONPATH="$ROOT/repo/third_party/ikflow" \
    --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --env GVS_ARM_XLA_THREADS=1 --env CUDA_VISIBLE_DEVICES= --env PYTHONUNBUFFERED=1 \
    --env MPLBACKEND=Agg --env MPLCONFIGDIR="$JT/mpl" \
    --pwd "$ROOT/repo" "$ROOT/lik-cpu.sif" \
    python -u "$ROOT/repo/scripts/fibre/fibre_probe.py" --robot "$ROBOT" --seed "$SEED" \
        --targets "$TARGETS" --out "$OUT_DIR/$ROBOT.s$SEED.json" $EXTRA \
        > "$OUT_DIR/$ROBOT.s$SEED.log" 2>&1
rc=$?
echo "$(date -Is) done rc=$rc"; exit $rc
