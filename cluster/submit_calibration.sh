#!/bin/bash
# Submit the full calibration set for the iiwa14 DDP retraining (plan rung 7).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Allotment (Thomas-approved envelope ~8 node-hours on xeon-g6-volta, reshaped
# from the serial A/B/C jobs into independent fire-and-forget jobs so nothing
# needs babysitting): 8 throughput jobs (~10 min each actual) + 2 lr-stability
# probes (~25 min each actual) ~= 6-7 node-hours actual. All jobs are submitted
# at once; the volta GrpTRES cap (node=4) meters them, so at most 4 nodes run at
# any moment and Thomas's other volta work shares the same queue.
#
# Throughput jobs: FRESH runs (a resumed leg dilutes steps/s), 1000 optimizer
# steps so the ~30-40 s startup amortizes, eval_every > max_steps so no
# validation or pole eval lands inside the timed window, no wandb. Each prints
# one THROUGHPUT line (steps/s, samples/s) from rank 0 at the end of fit.
#
# LR probes: the two candidate operating points at full scale (4 nodes, world
# size 8), 3000 steps with real validation cadence, watching for NaN --
#   r106: per-GPU 256 -> global 2048 @ lr 1.06e-4 (the committed default)
#   r150: per-GPU 512 -> global 4096 @ lr 1.50e-4 (the possible swap)
set -uo pipefail
cd "$(dirname "$0")/.."

THRU_ARGS="--max_steps=1000 --eval_every=100000 --val_set_size=20 \
--checkpoint_every=100000 --pole_eval_n=100 --disable_wandb"
PROBE_ARGS="--max_steps=3000 --eval_every=1000 --val_set_size=500 \
--checkpoint_every=100000 --pole_eval_n=500 --disable_wandb"

submit() {  # name nnodes gpus batch wall extra...
    local name=$1 nnodes=$2 gpus=$3 batch=$4 wall=$5; shift 5
    ALLOW_CONCURRENT=1 GPUS_PER_NODE=$gpus BATCH=$batch \
        bash cluster/submit_train.sh "$name" "$nnodes" "$wall" -- "$@" \
        || { echo "SUBMIT FAILED: $name" >&2; exit 1; }
}

# Throughput grid: (world_size, per-GPU batch)
submit cal_1n_1g_b512  1 1 512  00:40:00 $THRU_ARGS
submit cal_1n_2g_b256  1 2 256  00:30:00 $THRU_ARGS
submit cal_1n_2g_b512  1 2 512  00:40:00 $THRU_ARGS
submit cal_1n_2g_b1024 1 2 1024 00:50:00 $THRU_ARGS
submit cal_2n_b256     2 2 256  00:30:00 $THRU_ARGS
submit cal_2n_b512     2 2 512  00:40:00 $THRU_ARGS
submit cal_4n_b256     4 2 256  00:30:00 $THRU_ARGS
submit cal_4n_b512     4 2 512  00:40:00 $THRU_ARGS

# LR-stability probes at full scale
submit lrprobe_4n_r106 4 2 256 01:00:00 $PROBE_ARGS --learning_rate=1.06e-4
submit lrprobe_4n_r150 4 2 512 01:15:00 $PROBE_ARGS --learning_rate=1.5e-4

echo "All 10 calibration jobs submitted. Slurm meters them under the volta cap."
