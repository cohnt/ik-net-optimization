#!/bin/bash
# LLsub payload: fit the soft arm's learned forward-kinematics surrogate.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node), ONE GPU node, minutes not hours:
#   FK_RUNG=soft12 LLsub ./cluster/fk_surrogate_job.sh -s 20 -q xeon-g6-volta -T 01:00:00 -J lik_fksurr
#
# Needs NO dataset: the surrogate is fitted against the exact constant-strain map, which is
# the same torch code the solver differentiates. It is therefore independent of the IKFlow
# chart ladder and can be slotted into any gap in the queue.
#
# Every model fit in this project is a cluster job (Thomas, 2026-09-24: "we shouldn't train
# models locally"). A laptop attempt reached 11 mm median tip error against a 1 mm task gate
# and was deleted; float64 on a consumer GPU was most of why it never converged.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
RUNG="${FK_RUNG:-soft12}"
STEPS="${FK_STEPS:-60000}"
WIDTH="${FK_WIDTH:-1024}"
DEPTH="${FK_DEPTH:-6}"

## ikflow resolves its cache dirs from expanduser("~") AT IMPORT, so HOME is reassigned
## before anything imports it -- the same reason train_flow.sh does, and a path bug of
## exactly that shape once cost a rung its whole 620k-step run.
export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PY="$ROOT/venv/bin/python"
OUT="$REPO/models/$RUNG/${RUNG}__fk_surrogate.pt"

echo "fitting $RUNG FK surrogate: steps=$STEPS width=$WIDTH depth=$DEPTH -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"

cd "$REPO" || exit 2
"$PY" -u scripts/soft_arm/train_fk_surrogate.py \
    --rung "$RUNG" --steps "$STEPS" --width "$WIDTH" --depth "$DEPTH" --out "$OUT"
RC=$?

if [ $RC -eq 0 ] && [ -f "$OUT" ]; then
    echo "OK -- wrote $OUT"
    ## The accuracy report sits beside the weights, so a later reader can see what this
    ## surrogate is worth without reloading it. The number that matters is tip_mm/p99
    ## against the 1 mm task gate.
    cat "${OUT%.pt}.json"
    touch "$REPO/models/$RUNG/.FK_SURROGATE.DONE"
else
    echo "FAILED rc=$RC -- no sentinel written" >&2
fi
exit $RC
