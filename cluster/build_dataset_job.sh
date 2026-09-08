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

"$PY" -u "$REPO/third_party/ikflow/scripts/build_dataset.py" \
    --robot_name="$ROBOT" --training_set_size="$SIZE" --only_non_self_colliding --seed="$SEED"
RC=$?

if [ $RC -eq 0 ] && [ -d "$OUT" ]; then
    du -sh "$OUT"
    ls -la "$OUT"
    echo "size=$SIZE seed=$SEED $(date -Is)" > "$OUT/.DONE"
fi
exit $RC
