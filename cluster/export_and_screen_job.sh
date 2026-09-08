#!/bin/bash
# LLsub payload: export a finished ladder rung's checkpoints to .pkl (+ .arch.json) and
# screen each one, so a rung is measurable the moment its training job ends.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node), CPU partition, one node:
#   RUN_NAME=iiwa14_n6 ROBOT=iiwa14 LLsub ./cluster/export_and_screen_job.sh \
#       -s 48 -q xeon-p8 -T 02:00:00 -J lik_export
#
# CPU rather than GPU on purpose: this is minutes of work, and the whole
# xeon-g6-volta group cap (4 nodes) is held by the training job at the time -- a GPU
# request here would simply queue behind the very run it is meant to follow.
#
# EVERY kept checkpoint is screened, not only the last one. That is the point of
# save_top_k=-1: ddp_r1 developed pole mass somewhere near step 360000, and knowing WHEN
# a chart goes bad is a result about training, not a diagnostic afterthought. The final
# step is additionally exported to models/<robot>/ under the name the benchmark manifest
# expects (LADDER_RUNGS in cluster/gen_manifest.py).
#
# NOTE: source /etc/profile BEFORE -u -- Z97-byobu.sh reads an unset LC_BYOBU.
source /etc/profile
set -uo pipefail

echo "export_and_screen: alive on $(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-?}"

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
RUN_NAME="${RUN_NAME:?set RUN_NAME, e.g. iiwa14_n6}"
ROBOT="${ROBOT:?set ROBOT, e.g. iiwa14 or panda}"
FINAL_STEP="${FINAL_STEP:-620000}"
SCREEN_N="${SCREEN_N:-20000}"

RUN_DIR="$ROOT/results/train/$RUN_NAME"
OUT_DIR="$REPO/models/$ROBOT"
SCREEN_DIR="$ROOT/results/pole/$RUN_NAME"
mkdir -p "$OUT_DIR" "$SCREEN_DIR"

export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/mpl"
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/venv/bin/python"

shopt -s nullglob
CKPTS=("$RUN_DIR"/checkpoints/ikflow-checkpoint-step=*.ckpt)
if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "FATAL: no checkpoints under $RUN_DIR/checkpoints" >&2
    exit 4
fi
echo "found ${#CKPTS[@]} checkpoints for $RUN_NAME"

RC=0
for ckpt in "${CKPTS[@]}"; do
    step="${ckpt##*step=}"; step="${step%.ckpt}"
    # The name the benchmark manifest expects, with the run's own label. RUN_NAME carries
    # the robot prefix already (iiwa14_n6), so strip it to get the rung label (n6).
    label="${RUN_NAME#${ROBOT}_}"
    pkl="$OUT_DIR/${ROBOT}__${label}__step${step}.pkl"

    echo "=== step $step -> $pkl"
    "$PY" -u "$REPO/scripts/training/export_ckpt_to_pkl.py" "$ckpt" "$pkl" \
        --robot_name="$ROBOT" || { RC=1; continue; }

    # Both pole domains AND the accuracy, because the ladder's whole question is the
    # trade between them: the box domain is what every recorded baseline was measured on,
    # the task-pose domain is what the benchmark actually presents to the flow, and
    # chart accuracy is what a smaller chart is expected to give up.
    "$PY" -u "$REPO/scripts/training/pole_metric.py" --robot "$ROBOT" --checkpoint "$pkl" \
        --n "$SCREEN_N" --json_out "$SCREEN_DIR/${label}_step${step}_box.json" || RC=1
    "$PY" -u "$REPO/scripts/training/pole_at_task_poses.py" --robot "$ROBOT" --checkpoint "$pkl" \
        --n "$SCREEN_N" --json_out "$SCREEN_DIR/${label}_step${step}_taskpose.json" || RC=1
    "$PY" -u "$REPO/scripts/training/chart_accuracy.py" --robot "$ROBOT" --checkpoint "$pkl" \
        --n 5000 --json_out "$SCREEN_DIR/${label}_step${step}_accuracy.json" || RC=1

    # Only the final step keeps a copy under the manifest's expected name; the
    # intermediates stay for the training-time curve but are not benchmarked.
    # SidecarPath() replaces the .pkl extension rather than appending, so the sidecar is
    # <name>.arch.json, NOT <name>.pkl.arch.json. Getting this wrong would leave the
    # sidecar behind and the moved checkpoint would silently load at legacy defaults.
    sidecar="${pkl%.pkl}.arch.json"
    if [ "$step" != "$FINAL_STEP" ]; then
        mkdir -p "$RUN_DIR/pkl"
        mv "$pkl" "$sidecar" "$RUN_DIR/pkl/" || RC=1
    fi
done

echo "export_and_screen $RUN_NAME rc=$RC $(date -Is)" > "$RUN_DIR/EXPORT.SENTINEL"
exit $RC
