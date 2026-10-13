#!/bin/bash
# LLsub payload: build the analytic arm's C++ backend (cpp/build.sh) in the staged tree, then
# load it with the project's own venv and Drake pin and run its parity test.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node), after cluster/stage_code.sh -- a compile,
# so debug-cpu (the supercloud skill's compile carve-out), minutes only:
#   LLsub ./cluster/build_cpp_job.sh -s 4 -q debug-cpu -T 00:20:00 -J lik_build_cpp
#
# stage_code.sh never pushes cpp/build/ (a laptop .so links the laptop's libstdc++), so this
# runs after every restage that changes cpp/analytic_ik/; the loader refuses a library whose
# compiled-in source hash differs from the staged sources anyway. Writes
# $ROOT/build_cpp.DONE reading OK <hash> only if the build AND the parity test pass;
# submit_bench.sh refuses an analytic_backend=cpp manifest without it.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
SENTINEL="$ROOT/build_cpp.DONE"
rm -f "$SENTINEL"
cd "$REPO" || { echo "no repo at $REPO" >&2; exit 2; }

echo "=== build on $(hostname) at $(date -Is); g++: $(command -v g++ || echo MISSING)"
g++ --version | head -1
bash cpp/build.sh || { echo "FAIL: cpp/build.sh" >&2; exit 3; }

## The run_items.sh environment for the pinned Drake, so the test sees what a worker sees.
export HOME="$TMPDIR/home.build_cpp"; mkdir -p "$HOME/.cache"
ln -sfn "$ROOT/home/.cache/drake" "$HOME/.cache/drake" 2>/dev/null || true
ln -sfn "$ROOT/home/.cache/ikflow" "$HOME/.cache/ikflow" 2>/dev/null || true
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" TQDM_DISABLE=1 MPLBACKEND=Agg
"$ROOT/venv/bin/python" tests/test_analytic_cpp.py > "$TMPDIR/test_analytic_cpp.log" 2>&1
STATUS=$?
tail -5 "$TMPDIR/test_analytic_cpp.log"
[ $STATUS -eq 0 ] || { echo "FAIL: tests/test_analytic_cpp.py (exit $STATUS)" >&2;
                       grep FAIL "$TMPDIR/test_analytic_cpp.log" | head -20 >&2; exit 4; }
HASH=$("$ROOT/venv/bin/python" -c "import sys; sys.path.insert(0, '.'); from src.analytic_cpp import SourceHash; print(SourceHash())")
echo "OK $HASH" > "$SENTINEL"
echo "=== done: $(cat "$SENTINEL")"
