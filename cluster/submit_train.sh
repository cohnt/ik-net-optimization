#!/bin/bash
# Submit (or resume) an IKFlow training job -- one command, idempotent.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Usage:
#   ROBOT=<robot> bash cluster/submit_train.sh <run_name> [nnodes] [walltime] [-- extra train args]
# e.g.
#   ROBOT=iiwa14 bash cluster/submit_train.sh iiwa14_ddp_r1 4 96:00:00
#   ROBOT=iiwa14 bash cluster/submit_train.sh iiwa14_n6 4 96:00:00 -- --max_steps=620000 --nb_nodes=6
#   ROBOT=panda  bash cluster/submit_train.sh panda_n12_w256 4 96:00:00 -- \
#       --max_steps=620000 --coeff_fn_internal_size=256 --dim_latent_space=7
#   ROBOT=panda  bash cluster/submit_train.sh smoke2n 2 00:20:00 -- --max_steps=200 --eval_every=100 \
#       --val_set_size=20 --checkpoint_every=100 --pole_eval_n=500 --disable_wandb
#
# The ARCHITECTURE travels in the extra args (--nb_nodes, --coeff_fn_internal_size,
# --dim_latent_space, --rnvp_clamp). It is saved into the checkpoint's own
# hyper_parameters, so scripts/training/export_ckpt_to_pkl.py can write the `.arch.json`
# sidecar without being told separately what was trained. Each robot's ladder should hold
# --dim_latent_space at its own baseline (iiwa14: 8, panda: 7) so the downstream program
# keeps the same decision-variable count across the ladder.
#
# WHY NOT LLsub TRIPLE MODE: "[N,1,40]" generates a Slurm job ARRAY of N
# independent single-node jobs (llsub_batch.py: --array=1-N), so the nodes never
# share an allocation and the c10d rendezvous cannot form; its wrapper also execs
# the payload directly (needs the exec bit) and ends in a bare `wait`, which
# swallows every payload exit code (our first smoke "COMPLETED 0:0" in 1 s with
# empty logs on a chmod-less script). Instead we generate a launcher with
# #SBATCH directives -- LLsub's supported directives path
# (_submit_sbatch_with_directives) -- giving ONE job spanning N nodes, with
# `srun` fanning train_flow.sh out once per node inside the allocation, so
# SLURM_JOB_NODELIST covers all nodes (MASTER_ADDR is right), SLURM_NODEID is
# the node rank, and srun --kill-on-bad-exit propagates real exit codes.
# NOTE: in directives mode LLsub DROPS -g/-q/-T/-J, so every resource lives in
# the directives below. The launcher is written to $RUN_DIR/launch.sh and kept.
#
# Refuses if a lik_train job for ANY run is already RUNNING or PENDING: a second
# job on the same RUN_DIR would race its checkpoints (with <4 nodes in use it
# would START, not queue), and the volta group cap is shared with Thomas's other
# work. Scoped to this project's own job names -- the account is shared across
# projects (cluster/README, guard-scoping lesson).
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"

RUN_NAME="${1:?usage: ROBOT=<robot> submit_train.sh <run_name> [nnodes] [walltime] [-- extra args]}"
ROBOT="${ROBOT:?set ROBOT, e.g. ROBOT=iiwa14 or ROBOT=panda}"
NNODES="${2:-4}"
WALL="${3:-96:00:00}"
PARTITION="${PARTITION:-xeon-g6-volta}"   # PARTITION=debug-gpu for smoke-sized runs ONLY
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
BATCH="${BATCH:-256}"
shift $(( $# > 3 ? 3 : $# ))
[ "${1:-}" = "--" ] && shift
EXTRA_ARGS="$*"

## NOTE for anyone extending these greps: LLstat TRUNCATES the NAME column to 15
## characters, so job name `lik_train_iiwa14_n6` displays as `lik_train_iiwa1` and an
## exact-name match silently never fires. The patterns below match a PREFIX for that
## reason. To test one specific job, use `sacct -j <jobid> -X --format=State`, which is
## not truncated.
##
## ALLOW_CONCURRENT=1 skips the one-at-a-time guard. Calibration-only: distinct
## RUN_NAMEs cannot race each other's checkpoints, and the volta GrpTRES cap
## meters however many jobs are queued. NEVER set it when resubmitting a run
## that might still have a live job -- that is exactly the race the guard stops.
if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    LIVE=$(sc_run 'LLstat 2>/dev/null | grep -c "lik_train"' 2>/dev/null | tr -dc '0-9')
    if [ -n "${LIVE:-}" ] && [ "${LIVE:-0}" -gt 0 ]; then
        echo "REFUSING: $LIVE lik_train job(s) already RUNNING/PENDING." >&2
        echo "Two jobs on one RUN_DIR race the checkpoints. LLkill the old one or wait." >&2
        exit 3
    fi
fi

## The payload is exec'd by srun, so the exec bit is load-bearing (rsync -a
## preserves it from the local checkout; this catches a bad restage).
sc_run "test -x ~/$SC_ROOT/repo/cluster/train_flow.sh" || {
    echo "REFUSING: train_flow.sh on the cluster is not executable -- restage first." >&2
    exit 4
}

RUN_DIR_R="\$HOME/$SC_ROOT/results/train/$RUN_NAME"    # expanded remotely

echo "Submitting: RUN_NAME=$RUN_NAME ROBOT=$ROBOT NNODES=$NNODES WALL=$WALL PARTITION=$PARTITION EXTRA='$EXTRA_ARGS'"
sc_run "mkdir -p $RUN_DIR_R && cat > $RUN_DIR_R/launch.sh && chmod +x $RUN_DIR_R/launch.sh" <<LAUNCH
#!/bin/bash
#SBATCH --nodes=$NNODES
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:$GPUS_PER_NODE
#SBATCH --partition=$PARTITION
#SBATCH --exclusive
#SBATCH --time=$WALL
#SBATCH --job-name=lik_train_$RUN_NAME
#SBATCH --output=../results/train/$RUN_NAME/launch.log-%j
# Generated by cluster/submit_train.sh $(date -Is) -- do not edit; resubmit instead.
srun --ntasks-per-node=1 --kill-on-bad-exit=1 "\$HOME/$SC_ROOT/repo/cluster/train_flow.sh"
LAUNCH

sc_run "cd ~/$SC_ROOT/repo && \
  RUN_NAME='$RUN_NAME' ROBOT='$ROBOT' NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE BATCH=$BATCH \
  TRAIN_EXTRA_ARGS='$EXTRA_ARGS' \
  LLsub $RUN_DIR_R/launch.sh"
