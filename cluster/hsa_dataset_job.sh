#!/bin/bash
# LLsub payload: the LOInK Sec. VII planar-HSA dataset (2e6 inputs -> equilibrium -> EE pose).
#
# Submit (from ~/learned-ik/repo on the login node) -- CPU partition, one node:
#   LLsub ./cluster/hsa_dataset_job.sh -s 48 -q xeon-p8 -T 00:30:00 -J lik_hsa_dataset
# Env overrides: HSA_N (2000000) HSA_SEED (0) HSA_CHUNK (1000) HSA_WORKERS (nproc) HSA_ARGS (extra flags,
# e.g. "--mounting upright --variant epu_control").
#
# One single-threaded JAX process per CPU slice (cluster/cpu_slice.py), exactly as the GVS
# builder: worker k solves chunks k, k+W, k+2W, ...; each chunk is its own file under the
# output tree, so a killed job resumes by resubmitting. Worker 0..W-1 run with --no-merge; one
# --merge-only pass writes the final npz and the .DONE sentinel.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
N="${HSA_N:-2000000}"; SEED="${HSA_SEED:-0}"; CHUNK="${HSA_CHUNK:-1000}"
W="${HSA_WORKERS:-$(nproc)}"
OUTDIR="$ROOT/datasets/loink_hsa"
mkdir -p "$OUTDIR/chunks"

export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GVS_ARM_XLA_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

PY="$ROOT/venv/bin/python"
cd "$REPO" || exit 2
echo "node $(hostname), $W workers, N=$N seed=$SEED chunk=$CHUNK -> $OUTDIR"
OUT="$OUTDIR/loink_hsa_n${N}_seed${SEED}.npz"

PIDS=()
for ((k = 0; k < W; k++)); do
    taskset -c "$("$PY" cluster/cpu_slice.py "$W" "$k")" \
        "$PY" -u scripts/loink_replication/hsa_dataset.py --n "$N" --seed "$SEED" --chunk "$CHUNK" \
        --out "$OUT" --chunk-dir "$OUTDIR/chunks" --start "$k" --stride "$W" --no-merge ${HSA_ARGS:-} \
        > "$OUTDIR/worker_$k.log" 2>&1 &
    PIDS+=($!)
done
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
echo "workers done rc=$RC"
if [ $RC -eq 0 ]; then
    "$PY" -u scripts/loink_replication/hsa_dataset.py --n "$N" --seed "$SEED" --chunk "$CHUNK" \
        --out "$OUT" --chunk-dir "$OUTDIR/chunks" --merge-only ${HSA_ARGS:-} && \
        echo "n=$N seed=$SEED $(date -Is)" > "$OUTDIR/.DONE" || RC=1
fi
ls -la "$OUTDIR" | head
exit $RC
