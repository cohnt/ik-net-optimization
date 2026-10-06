#!/bin/bash
# LLsub payload: measure the GVS push-rod arm's dataset-sampler rate on a REAL CPU node.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node) -- the same partition and core
# count the dataset build will use, so the number transfers:
#   LLsub ./cluster/datagen_rate_job.sh -s 48 -q xeon-p8 -T 00:40:00 -J gvs_cal_rate
#
# WHY A JOB. The rate is a reportable number that sizes DATASET_SIZE and the build's wall
# time, so it is measured where the build runs and not on a login or debug node (standing
# rule) and not on the laptop, whose figure was taken under another session's load. Named
# *_cal_* so the staging guard ignores it: it produces no records.
#
# Same environment as build_dataset_job.sh: HOME rebound into the tree, no GPU, XLA's
# thread pool deliberately UNPINNED so JAX takes the whole node, as the build will.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset GVS_ARM_XLA_THREADS

cd "$REPO" || exit 2
echo "node $(hostname), $(nproc) cores, staged commit $(cat .staged-commit 2>/dev/null)"
"$ROOT/venv/bin/python" -u scripts/gvs_arm/probe_datagen_rate.py \
    --rung "${RUNG:-gvs_pushrod9_o1}" --batches "${BATCHES:-2000,20000,100000}" --repeats 2
