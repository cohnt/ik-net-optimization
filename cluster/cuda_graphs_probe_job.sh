#!/bin/bash
# LLsub payload: does CUDA-graph replay remove the flow's CPU dispatch cost on a V100?
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node):
#   LLsub ./cluster/cuda_graphs_probe_job.sh -g volta:2 -s 40 -q xeon-g6-volta \
#       -T 01:30:00 -J lik_cal_cudagraph
#
# A MEASUREMENT, so a real node, and ALONE on it: CPU contention is exactly what the
# probe measures. One process, one GPU. Named *_cal_* so the staging guard ignores it --
# it produces no campaign records, only results/profiling/cuda_graphs_<host>.json and
# the printed table. See scripts/profiling/probe_cuda_graphs.py for what it measures.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
## Slurm hands out GPU UUIDs, not indices; keep the first card only.
export CUDA_VISIBLE_DEVICES="$(echo "${CUDA_VISIBLE_DEVICES:-}" | cut -d, -f1)"

cd "$REPO" || exit 2
echo "node $(hostname), $(nproc) cores, staged commit $(cat .staged-commit 2>/dev/null)"
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
"$ROOT/venv/bin/python" -u scripts/profiling/probe_cuda_graphs.py \
    --chart panda:models/panda/panda__n6__step620000.pkl \
    --chart iiwa14:models/iiwa14/iiwa14__n4__step620000.pkl \
    --chart panda:models/panda/panda__n12__step620000.pkl \
    --chart iiwa14:models/iiwa14/iiwa14__ddp-r1__step620000.pkl \
    --out "$ROOT/results/profiling/cuda_graphs_$(hostname).json"
RC=$?
echo "cuda graphs probe rc=$RC"
exit $RC
