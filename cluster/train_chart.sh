#!/bin/bash
# The LLsub payload for training a NON-FLOW chart (the cVAE, branch chart-cvae; LOInK, branch
# chart-loink): ONE process on ONE GPU, then the final checkpoint exported and screened inside
# the same job.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit through cluster/submit_chart_train.sh ONLY. It writes a launcher with #SBATCH
# directives (--nodes=1 --ntasks=1 --cpus-per-task=20 --gres=gpu:volta:1, NOT --exclusive)
# into $RUN_DIR/launch.sh and LLsubs that; the launcher's `srun --ntasks=1
# --kill-on-bad-exit=1` runs THIS script inside the allocation so its exit code is the job's.
# Unlike train_flow.sh there is no torchrun and no rendezvous: the cVAE trains on one GPU,
# and two runs share a volta node, one GPU each.
#
# FAMILY selects the trainer / exporter / screen (`cvae`, `loink`). The latent width and
# any other architecture travel in TRAIN_EXTRA_ARGS from cluster/chart_runs.txt; the trainer
# records them in the checkpoint, and the exporter writes them into the `.arch.json` sidecar
# with `family`, so nothing downstream has to be told what was trained.
#
# THE STEP BUDGET MAY TRAVEL ON THE ROW. A row whose args carry `--max_steps=N` / `--batch_size=B`
# (the LOInK paper recipe: 150000 steps at batch 500) sets MAX_STEPS / BATCH -- and FINAL_STEP, the
# checkpoint exported -- from them, overriding the environment. Parsed HERE rather than exported by
# submit_chart_train.sh because a pair job (train_chart_pair.sh) runs two rows under ONE environment
# and the two may carry different recipes. A row without them (every cVAE row) keeps the
# environment's values, exactly as before. The explicit --batch_size/--max_steps below then carry
# the row's own values, so the trainer sees one value whichever occurrence its argparse takes.
#
# The job is IDEMPOTENT: --ckpt_path=auto resumes from $RUN_DIR/checkpoints/last.ckpt, so
# resubmitting the same RUN_NAME after a walltime kill continues the run.
#
# NOTE: source /etc/profile BEFORE enabling -u -- Z97-byobu.sh reads unset LC_BYOBU.
source /etc/profile
set -uo pipefail

## First line of output BEFORE anything that can fail (train_flow.sh: launcher plumbing can
## lose last-second stderr entirely).
echo "train_chart: alive on $(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-?}"

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
FAMILY="${FAMILY:-cvae}"
RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. panda_cvae7}"
ROBOT="${ROBOT:?set ROBOT, e.g. panda or iiwa14}"
RUN_DIR="$ROOT/results/train/$RUN_NAME"
BATCH="${BATCH:-4096}"           # ONE GPU, so this is the global batch (= the flows' 8 x 512)
MAX_STEPS="${MAX_STEPS:-620000}"
FINAL_STEP="${FINAL_STEP:-$MAX_STEPS}"
## The row's own budget (see the header): the LAST `--<name>=V` or `--<name> V` in TRAIN_EXTRA_ARGS.
row_arg() {
    local name="$1" prev="" w val=""
    for w in ${TRAIN_EXTRA_ARGS:-}; do
        case "$w" in "--$name="*) val="${w#--$name=}" ;; esac
        [ "$prev" = "--$name" ] && val="$w"
        prev="$w"
    done
    echo "$val"
}
ROW_STEPS="$(row_arg max_steps)"
ROW_BATCH="$(row_arg batch_size)"
ROW_CKPT_EVERY="$(row_arg checkpoint_every)"
if [ -n "$ROW_STEPS" ]; then MAX_STEPS="$ROW_STEPS"; FINAL_STEP="$ROW_STEPS"; fi
if [ -n "$ROW_BATCH" ]; then BATCH="$ROW_BATCH"; fi
## The export reads the STEP-NAMED checkpoint, so a checkpoint interval that does not divide the
## final step trains the whole budget and then exits 5. Refused up front when the row names one
## (the trainer's default interval is not visible from here: see the runbook).
if [ -n "$ROW_CKPT_EVERY" ] && [ $(( FINAL_STEP % ROW_CKPT_EVERY )) -ne 0 ]; then
    echo "FATAL: --checkpoint_every=$ROW_CKPT_EVERY does not divide the final step $FINAL_STEP;" >&2
    echo "no step-named checkpoint would be written to export. Fix the row." >&2
    exit 2
fi
## Where the exported final checkpoint lands: the name stage CVAE's items read,
## models/<robot>/<robot>__<label>__step<N>.pkl -- or, for a smoke (EXPORT_TO_RUN_DIR=1), the
## run's own directory, so a 200-step network can never sit under a name a benchmark would load.
if [ "${EXPORT_TO_RUN_DIR:-0}" = "1" ]; then
    EXPORT_DIR="$RUN_DIR/pkl"
else
    EXPORT_DIR="$REPO/models/$ROBOT"
fi

case "$FAMILY" in
    cvae)
        TRAINER="$REPO/scripts/training/train_cvae.py"
        EXPORTER="$REPO/scripts/training/export_cvae.py"
        SCREEN="$REPO/scripts/training/screen_cvae.py"
        CKPT_PREFIX="cvae-checkpoint-step=" ;;
    loink)
        TRAINER="$REPO/scripts/training/train_loink.py"
        EXPORTER="$REPO/scripts/training/export_loink.py"
        SCREEN="$REPO/scripts/training/screen_loink.py"
        CKPT_PREFIX="loink-checkpoint-step=" ;;
    *)
        echo "FATAL: unknown FAMILY='$FAMILY' (cvae, loink)" >&2; exit 2 ;;
esac

## Training HOME must be the persistent tree (train_flow.sh): ikflow resolves DATASET_DIR
## from expanduser("~") at import, and the trainer reads the same IkfLitDataset files the
## flows trained on, under $ROOT/home/.cache/ikflow/datasets.
export HOME="$ROOT/home"
## Node-local caches stay off Lustre.
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton"
export MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/mpl"
export IS_SLURM=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
## wandb: offline on compute nodes (no internet, no credentials). Synced from the laptop.
export WANDB_MODE=offline WANDB_DIR="$RUN_DIR"
export WANDB_ENTITY="${WANDB_ENTITY:-cohnt-massachusetts-institute-of-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-ikflow}"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
## One process on the job's 20 cores, sharing the node with a second run on the other 20.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

## CUDA_VISIBLE_DEVICES: leave EXACTLY as Slurm set it -- ONE UUID, this job's V100. The
## other run on the node sees only its own.
PY="$ROOT/venv/bin/python"
mkdir -p "$RUN_DIR/checkpoints"

## The .DONE sentinel, NOT just the directory (train_flow.sh): build_dataset_job.sh writes it
## only on success, and a killed build leaves half-written tensors a -d test accepts.
DATASET_DIR="$ROOT/home/.cache/ikflow/datasets/$ROBOT"
if [ ! -f "$DATASET_DIR/.DONE" ]; then
    echo "FATAL: no COMPLETE dataset at $DATASET_DIR (no .DONE sentinel)." >&2
    echo "Run cluster/build_dataset_job.sh with DATASET_ROBOT=$ROBOT; if a partial build" >&2
    echo "is present, delete the directory first -- a partial one is worse than none." >&2
    exit 4
fi

BUDGET_FROM="the environment"; [ -n "$ROW_STEPS$ROW_BATCH" ] && BUDGET_FROM="the row's args"
echo "train_chart: family=$FAMILY run=$RUN_NAME robot=$ROBOT batch=$BATCH max_steps=$MAX_STEPS" \
     "final_step=$FINAL_STEP (budget from $BUDGET_FROM)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  dataset: $(cat "$DATASET_DIR/.DONE")"
nvidia-smi -L || true

"$PY" -u "$TRAINER" \
    --robot_name="$ROBOT" --run_dir="$RUN_DIR" --ckpt_path=auto \
    --batch_size="$BATCH" --max_steps="$MAX_STEPS" \
    ${TRAIN_EXTRA_ARGS:-}
RC=$?

## Export + screen INSIDE the training job, only if training succeeded (train_flow.sh's
## reasoning: no second dependency link per run, so a paused session costs nothing). ONLY the
## final checkpoint: there is no chart to select among a family's -- every rung (cVAE, LOInK) is
## pre-registered at its final step -- and the flows' per-step screening answered a question
## (when does pole mass appear) that these training curves do not need re-answering here.
## RUN_EXPORT=0 disables it.
if [ "$RC" -eq 0 ] && [ "${RUN_EXPORT:-1}" = "1" ]; then
    echo "train_chart: training ok, exporting and screening $RUN_NAME $(date -Is)"
    CKPT="$RUN_DIR/checkpoints/${CKPT_PREFIX}${FINAL_STEP}.ckpt"
    ## The STEP-NAMED checkpoint, never last.ckpt: last.ckpt is whatever the run reached, and a
    ## run cut short would be exported under the step-$FINAL_STEP name the benchmark loads.
    if [ ! -f "$CKPT" ]; then
        echo "FATAL: no checkpoint $CKPT -- training exited 0 short of step $FINAL_STEP," >&2
        echo "or the trainer's checkpoint interval does not divide it. Nothing exported." >&2
        RC=5
    else
        LABEL="${RUN_NAME#${ROBOT}_}"
        mkdir -p "$EXPORT_DIR"
        PKL="$EXPORT_DIR/${ROBOT}__${LABEL}__step${FINAL_STEP}.pkl"
        ## The exporter (export_cvae.py / export_loink.py) reads the architecture from the
        ## checkpoint's own hyper_parameters and round-trips the export through the family's
        ## loader before returning 0. LEARNED_IK_ROOT stays
        ## explicit for every child: HOME was reassigned above, and a script deriving ROOT from
        ## $HOME would resolve one level deep (train_flow.sh, the iiwa14_n6 export of 2026-09-09).
        LEARNED_IK_ROOT="$ROOT" "$PY" -u "$EXPORTER" "$CKPT" --out "$PKL" || RC=6
        if [ "$RC" -eq 0 ]; then
            ## The screen's record goes to $ROOT/results/screen_<family>/<name>.json, which
            ## collect_results.sh ships (its default, results/ under the REPO, is never collected).
            NAME="$(basename "${PKL%.pkl}")"
            SCREEN_DIR="$ROOT/results/screen_$FAMILY"
            mkdir -p "$SCREEN_DIR"
            LEARNED_IK_ROOT="$ROOT" "$PY" -u "$SCREEN" --robot "$ROBOT" --checkpoint "$PKL" \
                --json_out "$SCREEN_DIR/$NAME.json" || RC=7
        fi
    fi
    echo "train_chart: export/screen rc=$RC $(date -Is)"
fi

echo "train_chart $RUN_NAME rc=$RC $(date -Is)" > "$RUN_DIR/node0.SENTINEL"
exit $RC
