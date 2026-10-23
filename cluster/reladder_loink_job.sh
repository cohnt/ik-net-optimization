#!/bin/bash
# LLsub payload: re-choose the decode budget (`n_dys`, `n_newton`, `newton_damping`) of ONE
# exported LOInK chart with the exporter's ladder (`scripts/training/export_loink.py
# --reladder`), rewriting only the inverse fields of its `.arch.json` sidecar. The `.pkl` is
# never written. Why this exists: the ladder's round-trip gate once drew latents from N(0, I)
# only, and the budgets it chose failed to invert 15-50% of REAL configurations' latents
# (|z| 13-25 on the soft12 cost-head charts), scored `unrepresentable_start` on paired cells
# (2026-10-23). The gate now covers three latent draws (prior, the +-5 box, real); see the
# exporter's module docstring for the best-effort rule when no triple is exact on real latents.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node), one job per chart, CPU only:
#   PKL=models/soft12/soft12__loinkmrP__step620000.pkl LLsub ./cluster/reladder_loink_job.sh \
#       -s 8 -q xeon-p8 -T 08:00:00 -J lik_reladder_mrP
# Optional: N_DYS_MAX (default 400), N_NEWTON_MAX (default 50) cap the ladders; a relu chart
# cannot pass the real draw at any budget, so without caps it would walk to 3200 x 50.
# Logs: $ROOT/results/reladder/<pkl stem>.log (the ladder rows and the CHOSEN line).
# Afterwards `stage_code.sh` is NOT the way back: the sidecar lives in the staged tree
# ($REPO/models/<robot>/), which the benchmarks read; copy it to the laptop's models/ by hand.
#
# NOTE: source /etc/profile BEFORE -u -- Z97-byobu.sh reads an unset LC_BYOBU.
source /etc/profile
set -uo pipefail

echo "reladder_loink: alive on $(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-?}"

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
PKL="${PKL:?set PKL, a path relative to the staged repo, e.g. models/soft12/soft12__loinkmrP__step620000.pkl}"
N_DYS_MAX="${N_DYS_MAX:-400}"
N_NEWTON_MAX="${N_NEWTON_MAX:-50}"
LOG_DIR="$ROOT/results/reladder"
mkdir -p "$LOG_DIR"

export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/mpl"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-8}}"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/venv/bin/python"

cd "$REPO" || exit 4
[ -f "$PKL" ] || { echo "FATAL: no $PKL under $REPO" >&2; exit 4; }
stem="$(basename "$PKL" .pkl)"
log="$LOG_DIR/$stem.log"
echo "reladder $PKL (n_dys <= $N_DYS_MAX, n_newton <= $N_NEWTON_MAX, $OMP_NUM_THREADS threads) -> $log"
cp "${PKL%.pkl}.arch.json" "$LOG_DIR/$stem.arch.json.before"
LEARNED_IK_ROOT="$ROOT" "$PY" -u scripts/training/export_loink.py --reladder "$PKL" \
    --n_dys_max "$N_DYS_MAX" --n_newton_max "$N_NEWTON_MAX" 2>&1 | tee "$log"
RC=${PIPESTATUS[0]}
cp "${PKL%.pkl}.arch.json" "$LOG_DIR/$stem.arch.json.after"
echo "reladder_loink: done rc=$RC date=$(date -Is)"
exit "$RC"
