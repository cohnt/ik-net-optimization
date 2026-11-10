#!/bin/bash
# LLsub payload: the IK-fibre investigation (scripts/fibre/, docs/notes-fibre.md), one node.
# ROBOTS x SHARDS processes, one single-threaded CPU process each, seeds SEED_BASE + k.
# Produces no benchmark records: per-seed JSONs under OUT_DIR, read by summarise_fibre.py.
#
#   SCRIPT=fibre_probe ROBOTS=soft12 SHARDS=48 TARGETS=2 SEED_BASE=1000 TAG=probe \
#       LLsub ./cluster/fibre_job.sh -s 48 -q xeon-p8 -T 02:00:00 -J lik_cal_fibre_soft12_1000
#
# CPU only (CUDA hidden): it runs on xeon-p8, or on xeon-g6-volta when the CPU partition is
# wanted for other work. PROBE_DIR defaults to the staged repo's scripts/fibre; point it at a
# copy to run code that is not staged. Named *_cal_* so the staging guard ignores it.
#
# Learned the hard way (2026-11-09):
#   * every JAX-importing worker owns ~100 idle XLA threads, so 48+ workers exhaust the
#     default `ulimit -u` (env.cc "Thread tf_XLAEigen creation via pthread_create() failed");
#   * a private HOME per worker (jrl rewrites ~/.cache/jrl/urdfs at every Robot()) also means
#     a private Matplotlib font cache unless MPLCONFIGDIR is shared: ~9 min of startup with
#     48 building it at once. It is built ONCE per node below.
source /etc/profile
set -uo pipefail
ROOT="$HOME/learned-ik"
export LIK_REPO="${LIK_REPO:-$ROOT/repo}"
PROBE_DIR="${PROBE_DIR:-$LIK_REPO/scripts/fibre}"
SCRIPT="${SCRIPT:-fibre_probe}"
TAG="${TAG:-$SCRIPT}"
OUT_DIR="${OUT_DIR:-$ROOT/scratch-fibre/$TAG}"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export GVS_ARM_XLA_THREADS=1
export CUDA_VISIBLE_DEVICES=""
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
ROBOTS="${ROBOTS:?set ROBOTS}"; SHARDS="${SHARDS:-48}"; TARGETS="${TARGETS:-2}"
SEED_BASE="${SEED_BASE:-1000}"; EXTRA="${EXTRA:-}"
mkdir -p "$OUT_DIR"
cd "$LIK_REPO" || exit 2
echo "$(date -Is) node $(hostname), $(nproc --all) cores, ulimit -u $(ulimit -u), staged commit $(cat .staged-commit 2>/dev/null)"
echo "SCRIPT=$SCRIPT PROBE_DIR=$PROBE_DIR OUT_DIR=$OUT_DIR ROBOTS=$ROBOTS SHARDS=$SHARDS TARGETS=$TARGETS SEED_BASE=$SEED_BASE EXTRA=$EXTRA"
export MPLCONFIGDIR="$TMPDIR/mpl"; mkdir -p "$MPLCONFIGDIR"
"$ROOT/venv/bin/python" -c "import matplotlib.font_manager" 2>/dev/null
echo "$(date -Is) font cache warm"
pids=()
for r in $ROBOTS; do
  for k in $(seq 0 $((SHARDS - 1))); do
    seed=$((SEED_BASE + k))
    H="$TMPDIR/home.$r.$k"; mkdir -p "$H/.cache"
    ln -sfn "$ROOT/home/.cache/ikflow" "$H/.cache/ikflow"
    ln -sfn "$ROOT/home/.cache/drake" "$H/.cache/drake"
    # shellcheck disable=SC2086
    HOME="$H" "$ROOT/venv/bin/python" -u "$PROBE_DIR/$SCRIPT.py" --robot "$r" --seed "$seed" \
        --targets "$TARGETS" --out "$OUT_DIR/$r.s$seed.json" $EXTRA \
        > "$OUT_DIR/$r.s$seed.log" 2>&1 &
    pids+=($!)
  done
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date -Is) done rc=$rc"; exit $rc
