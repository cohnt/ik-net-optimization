#!/bin/bash
# LLsub payload: does the CUDA-graph gain survive several solver processes sharing a V100?
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node):
#   LLsub ./cluster/cuda_graphs_contention_job.sh -g volta:2 -s 40 -q xeon-g6-volta \
#       -T 02:00:00 -J lik_cal_cgcontend
#
# WHY. probe_cuda_graphs.py times ONE process alone on a node. Graph replay removes the CPU
# dispatch that used to leave the GPU idle most of each call, so a graphed process is close
# to GPU-bound -- and the record runs PROCS=8, four processes per V100. If four GPU-bound
# processes time-slice one card, the single-process gain does not transfer. So: K copies of
# the probe per GPU on both GPUs (2K per node), K = 1, 2, 4, all timing the same window
# (--start-at). K = 4 is the record's PROCS=8. Named *_cal_*: no campaign records.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
SLURM_GPUS="${CUDA_VISIBLE_DEVICES:-}"
NGPU=$(echo "$SLURM_GPUS" | tr ',' '\n' | grep -c .)
OUT="$ROOT/results/profiling/contention_$(hostname)"
mkdir -p "$OUT"

cd "$REPO" || exit 2
echo "node $(hostname), $(nproc) cores, $NGPU GPUs, staged commit $(cat .staged-commit 2>/dev/null)"
RC=0
for K in ${KS:-1 2 4}; do
    ## Build, compile and check take ~1-2 min per process; start the clock well after.
    START=$(( $(date +%s) + ${LEAD:-240} ))
    pids=()
    for g in $(seq 1 "$NGPU"); do
        for k in $(seq 1 "$K"); do
            CUDA_VISIBLE_DEVICES="$(echo "$SLURM_GPUS" | cut -d, -f$g)" \
            "$ROOT/venv/bin/python" -u scripts/profiling/probe_cuda_graphs.py \
                --chart panda:models/panda/panda__n6__step620000.pkl \
                --variants E,C,CG --calls 2000 --warmup 100 --check 50 --skip-profile \
                --start-at "$START" --out "$OUT/K${K}_gpu${g}_p${k}.json" \
                > "$OUT/K${K}_gpu${g}_p${k}.log" 2>&1 &
            pids+=($!)
        done
    done
    for pid in "${pids[@]}"; do wait "$pid" || RC=1; done
    echo "K=$K per GPU done (rc so far $RC)"
done
echo "cuda graphs contention rc=$RC"
exit $RC
