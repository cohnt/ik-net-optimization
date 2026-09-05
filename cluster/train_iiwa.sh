#!/bin/bash
# The LLsub payload for iiwa14 IKFlow retraining: one shell per node, torchrun
# forks one rank per GPU, all nodes rendezvous inside ONE Slurm job.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit through cluster/submit_train.sh ONLY. It generates a launcher with
# #SBATCH directives (--nodes=N --ntasks-per-node=1 --gres=gpu:volta:2) into
# $RUN_DIR/launch.sh and LLsubs that; the launcher's `srun --ntasks-per-node=1
# --kill-on-bad-exit=1` runs THIS script once per node inside one shared
# allocation. LLsub triple mode must NOT be used: "[N,1,40]" is a job ARRAY of
# N independent single-node jobs (no shared nodelist -> no c10d rendezvous, and
# its bare-`wait` wrapper swallows payload exit codes). See submit_train.sh.
#
# Extra train_ddp.py args travel via TRAIN_EXTRA_ARGS (sbatch exports the
# submission env by default and srun propagates it; trailing args do not pass).
#
# The job is IDEMPOTENT: --ckpt_path=auto resumes from
# $RUN_DIR/checkpoints/last.ckpt, so resubmitting the same RUN_NAME after a
# walltime kill continues the run; a new RUN_NAME starts fresh.
#
# NOTE: source /etc/profile BEFORE enabling -u -- Z97-byobu.sh reads unset LC_BYOBU.
source /etc/profile
set -uo pipefail

## First line of output BEFORE anything that can fail: the triple-mode postmortem
## showed launcher plumbing can lose last-second stderr entirely.
echo "train_iiwa: alive on $(hostname) date=$(date -Is) nodeid=${SLURM_NODEID:-?} job=${SLURM_JOB_ID:-?}"

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. iiwa14_ddp_r1}"
RUN_DIR="$ROOT/results/train/$RUN_NAME"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
BATCH="${BATCH:-256}"          # PER-RANK batch size (global = NNODES*GPUS_PER_NODE*BATCH)

## Training HOME must be the persistent tree: ikflow computes DATASET_DIR and
## TRAINING_LOGS_DIR from expanduser("~") at import, and the dataset lives under
## $ROOT/home/.cache/ikflow/datasets. (The benchmark workers do the opposite --
## per-rank $TMPDIR HOMEs -- because they only READ caches. Training must not.)
export HOME="$ROOT/home"
## Node-local caches stay off Lustre.
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton"
export MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/mpl"
export IS_SLURM=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
## wandb: offline on compute nodes (no internet, no credentials here). Sync happens
## from the laptop. Entity/project are asserted by ikflow when wandb is enabled.
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR"
export WANDB_ENTITY="${WANDB_ENTITY:-cohnt-massachusetts-institute-of-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-ikflow}"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
## Two ranks x OMP threads must stay well under 40 cores; the dataloader is in-RAM.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

## CUDA_VISIBLE_DEVICES: leave EXACTLY as Slurm set it (two UUIDs on a volta node).
## torchrun's LOCAL_RANK 0/1 index into the visible set via torch.cuda.set_device.

PY="$ROOT/venv/bin/python"
NODE_RANK="${SLURM_NODEID:-0}"   # srun sets it per task; 1 task per node => node rank
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
mkdir -p "$RUN_DIR/checkpoints"

DATASET_DIR="$ROOT/home/.cache/ikflow/datasets/iiwa14"
if [ ! -d "$DATASET_DIR" ]; then
    echo "FATAL: no dataset at $DATASET_DIR -- run cluster/build_dataset job first" >&2
    exit 4
fi

echo "train_iiwa: node_rank=$NODE_RANK/$NNODES master=$MASTER_ADDR run=$RUN_NAME batch=$BATCH gpus=$GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

"$PY" -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$GPUS_PER_NODE" --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" --rdzv_id="${SLURM_JOB_ID:-local}" \
    "$REPO/third_party/ikflow/scripts/train_ddp.py" \
    --robot_name=iiwa14 --run_dir="$RUN_DIR" --ckpt_path=auto \
    --num_nodes="$NNODES" --gpus_per_node="$GPUS_PER_NODE" --batch_size="$BATCH" \
    ${TRAIN_EXTRA_ARGS:-}
RC=$?
echo "train_iiwa node $NODE_RANK rc=$RC $(date -Is)" > "$RUN_DIR/node${NODE_RANK}.SENTINEL"
exit $RC
