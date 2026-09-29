#!/bin/bash
# LLsub payload: build a robot's IKFlow training dataset into the persistent fake HOME.
#
# DATASET_ROBOT selects the arm (iiwa14 / panda). The Panda ladder needs its own dataset;
# the iiwa14 one was built for the ddp-r1 campaign and is already on the cluster.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node) -- CPU partition, one node:
#   DATASET_ROBOT=panda LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-p8 -T 02:00:00 -J lik_dataset
#
# Offline-safe (jrl URDFs ship in the wheel; no downloads). Single-threaded Klampt
# sampling, measured ~14us/config on the laptop -- 25M is minutes-to-an-hour here.
# Output: 4 float32 tensors (~1.4 GB total) + info.txt in
#   $ROOT/home/.cache/ikflow/datasets/$DATASET_ROBOT/
# which is where training jobs (HOME=$ROOT/home) will find them. Writes a .DONE
# sentinel next to the dataset for polling.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
ROBOT="${DATASET_ROBOT:-iiwa14}"
SIZE="${DATASET_SIZE:-25000000}"
SEED="${DATASET_SEED:-0}"

export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PY="$ROOT/venv/bin/python"
OUT="$ROOT/home/.cache/ikflow/datasets/$ROBOT"
echo "building $ROBOT dataset: size=$SIZE seed=$SEED -> $OUT"

case "$ROBOT" in
    gvs_*)
        ## The GVS push-rod arm: every sample is a Newton solve on SoRoMoX's rod, and the
        ## batched JAX solve does not spread across a node's cores (14.9 ms/sample measured
        ## on 96 cores with XLA unpinned). scripts/gvs_arm/build_dataset_parallel.py runs
        ## one single-threaded JAX per CPU the job owns and writes ikflow's exact files.
        ## One thread per worker process: the builder is process-parallel, and a per-core
        ## BLAS/OpenMP pool in each of ~96 workers exhausts RLIMIT_NPROC (measured).
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GVS_ARM_XLA_THREADS=1
        ## ... and even so, every JAX process still creates XLA's own Eigen pool (one thread
        ## per core, idle) and its compiler threads, which no flag in jax 0.11 turns off:
        ## ~96 workers x ~100 threads against a SOFT process limit of 4096 killed the
        ## second attempt at `GetPjRtCpuClient`. The hard limit is ~770k, so raise the
        ## soft one (and the open-file one, which the symbolizer also complained about).
        ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
        ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
        echo "process limit $(ulimit -u), open files $(ulimit -n), CPUs $(nproc)"
        "$PY" -u "$REPO/scripts/gvs_arm/build_dataset_parallel.py" \
            --robot_name="$ROBOT" --training_set_size="$SIZE" --only_non_self_colliding --seed="$SEED"
        RC=$? ;;
    *)
        "$PY" -u "$REPO/scripts/training/ikflow_entry.py" build_dataset \
            --robot_name="$ROBOT" --training_set_size="$SIZE" --only_non_self_colliding --seed="$SEED"
        RC=$? ;;
esac

if [ $RC -eq 0 ] && [ -d "$OUT" ]; then
    du -sh "$OUT"
    ls -la "$OUT"
    echo "size=$SIZE seed=$SEED $(date -Is)" > "$OUT/.DONE"
fi
exit $RC
