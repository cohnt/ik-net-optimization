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
## A REAL partition, always. Even a 200-step validation run trains, and anything that
## trains is a job for a real partition however short it is -- debug-gpu is a small
## non-ExclusiveUser pool shared with other people's quick checks (Thomas, 2026-09-08:
## "No big jobs on debug nodes"). submit_ladder.sh --smoke sizes that check instead:
## one node, 20 min, on this default.
PARTITION="${PARTITION:-xeon-g6-volta}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
BATCH="${BATCH:-256}"
## DEPENDENCY (e.g. "afterany:12345") chains rungs so the ladder advances with no session
## attached. afterany rather than afterok on purpose: a rung that dies must not stall every
## rung behind it, and --ckpt_path=auto means resubmitting the dead one resumes it.
DEPENDENCY="${DEPENDENCY:-}"
DEP_DIRECTIVE="${DEPENDENCY:+1}"
## EXCLUDE_NODES drops known-bad nodes from consideration. 2026-09-08: d-8-8-1 took down
## three jobs in a row -- a single-node run hung to TIMEOUT on it, and two 4-node runs
## FAILED because the rank landing there could not reach the c10d master. The same run had
## trained happily for 75 minutes on a node set without it. With ~108 volta nodes free,
## excluding one costs nothing and a bad node otherwise drains an afterany chain in minutes.
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXC_DIRECTIVE="${EXCLUDE_NODES:+1}"
shift $(( $# > 3 ? 3 : $# ))
[ "${1:-}" = "--" ] && shift
EXTRA_ARGS="$*"

## NEVER USE LLstat FOR A PROGRAMMATIC CHECK. It truncates the NAME column to 15
## characters, so `lik_train_soft12_n4`, `lik_train_soft12_n8` and `lik_train_soft16_n6`
## all render as `lik_train_soft1`, and every rung of a ladder collapses to one string. A
## guard built on that cannot distinguish the run it is protecting from its siblings --
## measured, not assumed. `squeue -h -n <name>` matches the FULL name exactly and prints
## nothing when there is no match; `sacct -j <id> -X --format=State` is exact for one job.
##
## The hazard here is two jobs sharing one RUN_DIR and racing its checkpoints, which is a
## property of the RUN, not of the account: an account-wide `lik_train` match meant a
## sibling campaign's jobs, in a different tree with no shared RUN_DIR, refused this
## submission.
##
## TWO DISTINCT HAZARDS, and they were conflated in one over-broad check. Separating them
## is what lets each be scoped correctly.
##
## (a) THE CHECKPOINT RACE is a property of the RUN: two jobs sharing one RUN_DIR overwrite
## each other's checkpoints. Always refused, and NOT bypassable -- a legitimate reason to
## run rungs concurrently is never a reason to run the same rung twice.
WANT="${SC_JOB_PREFIX}_train_$RUN_NAME"
LIVE=$(sc_run "squeue -u \$USER -h -n '$WANT' -o '%i' 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
if [ -n "${LIVE:-}" ] && [ "${LIVE:-0}" -gt 0 ]; then
    echo "REFUSING: $LIVE job(s) named exactly $WANT are queued, running or completing." >&2
    echo "Two jobs on one RUN_DIR race the checkpoints. LLkill the old one, or wait." >&2
    echo "NOTE: a job just killed lingers in CG for up to a minute and squeue still" >&2
    echo "lists it, so an immediate resubmit of the same name is refused. That is" >&2
    echo "this guard working, not a bug -- wait for the old job to clear." >&2
    exit 3
fi

## (b) ONE RUNG AT A TIME is a property of the TREE and of the volta cap: each rung takes
## all 4 nodes, so two concurrent rungs halve each other's throughput inside a fixed step
## budget. cluster/README.md documents this guard as what enforces it, and `--next` relies
## on it rather than duplicating it -- so scoping the check above to RUN_NAME alone would
## have silently dropped the property for anyone submitting rungs by hand.
##
## Scoped by submission directory (`%Z`), for the reason in stage_code.sh: a sibling
## campaign in another tree competes for the cap but is not ours to refuse, and matching
## job names cannot see a job submitted without `-J`.
##
## ALLOW_CONCURRENT=1 bypasses THIS check only. chain_ladder.sh sets it because its rungs
## are queued as a dependency chain: they are submitted together but run strictly one at a
## time, so the property is enforced by Slurm rather than by this guard.
if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    BUSY=$(sc_run "squeue -u \$USER -h -o '%Z %j' 2>/dev/null | awk -v d=\"\$HOME/$SC_ROOT/repo\" '\$1==d && \$2 ~ /_train_/' | wc -l" 2>/dev/null | tr -dc '0-9')
    if [ -n "${BUSY:-}" ] && [ "${BUSY:-0}" -gt 0 ]; then
        echo "REFUSING: $BUSY training job(s) of this tree (~/$SC_ROOT) already queued." >&2
        echo "One rung at a time: each takes the whole volta cap. Wait, or chain them" >&2
        echo "with cluster/chain_ladder.sh, which sets ALLOW_CONCURRENT=1 deliberately." >&2
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

echo "Submitting: RUN_NAME=$RUN_NAME ROBOT=$ROBOT NNODES=$NNODES WALL=$WALL PARTITION=$PARTITION EXCLUDE='${EXCLUDE_NODES:-none}' EXTRA='$EXTRA_ARGS'"
sc_run "mkdir -p $RUN_DIR_R && cat > $RUN_DIR_R/launch.sh && chmod +x $RUN_DIR_R/launch.sh" <<LAUNCH
#!/bin/bash
#SBATCH --nodes=$NNODES
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:$GPUS_PER_NODE
#SBATCH --partition=$PARTITION
#SBATCH --exclusive
#SBATCH --time=$WALL
#SBATCH --job-name=${SC_JOB_PREFIX}_train_$RUN_NAME${DEP_DIRECTIVE:+
#SBATCH --dependency=$DEPENDENCY}${EXC_DIRECTIVE:+
#SBATCH --exclude=$EXCLUDE_NODES}
#SBATCH --output=../results/train/$RUN_NAME/launch.log-%j
# Generated by cluster/submit_train.sh $(date -Is) -- do not edit; resubmit instead.
srun --ntasks-per-node=1 --kill-on-bad-exit=1 "\$HOME/$SC_ROOT/repo/cluster/train_flow.sh"
LAUNCH

## RUN_EXPORT reaches train_flow.sh's inline export step; validation runs set it to 0
## because a 200-step checkpoint is not worth exporting or screening.
## LEARNED_IK_ROOT must be forwarded EXPLICITLY. This script resolves its own paths from
## SC_ROOT, but train_flow.sh reads `${LEARNED_IK_ROOT:-$HOME/learned-ik}` and would
## otherwise fall back to the DEFAULT tree while running THIS one's code -- reading another
## campaign's dataset cache and writing into its results. Silent, and worse from an
## isolated root than from the default one, because the two trees then disagree.
sc_run "cd ~/$SC_ROOT/repo && \
  LEARNED_IK_ROOT=\$HOME/$SC_ROOT \
  RUN_NAME='$RUN_NAME' ROBOT='$ROBOT' NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE BATCH=$BATCH \
  RUN_EXPORT='${RUN_EXPORT:-1}' \
  TRAIN_EXTRA_ARGS='$EXTRA_ARGS' \
  LLsub $RUN_DIR_R/launch.sh"
