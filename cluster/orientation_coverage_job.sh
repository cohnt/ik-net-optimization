#!/bin/bash
# LLsub payload: how much of SO(3) does the arm reach at a fixed tip position?
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node):
#   LLsub ./cluster/orientation_coverage_job.sh \
#       -s 48 -q xeon-p8 -T 02:00:00 -J gvs_cal_cover
#
# WHY A JOB. Answering it needs tens of millions of equilibrium solves: the tip must land
# in a small ball about a fixed position before its orientation counts, and that is a
# ~0.03-0.3% acceptance per draw. The laptop's own run took 5 h 20 min for one position
# and 6004 orientations, under a peer session's load. Named *_cal_* so the staging guard
# ignores it: it produces no records, only a printed table.
#
# Process-parallel like the dataset build, and for the same reason -- the vmapped JAX
# solve does not spread across a node's cores -- so each worker is pinned to its own CPU
# slice with a single-threaded JAX.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
export HOME="$ROOT/home"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
## Each JAX worker still owns ~100 threads it never runs; the default soft limit cannot
## hold 48 of them (the dataset build's second attempt died exactly there).
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

cd "$REPO" || exit 2
echo "node $(hostname), $(nproc) cores, staged commit $(cat .staged-commit 2>/dev/null)"
grep MemTotal /proc/meminfo
"$ROOT/venv/bin/python" -u scripts/probe_orientation_coverage.py \
    --robot "${ROBOT:-gvs_pushrod9_o1}" \
    --draws "${DRAWS:-20000000}" \
    --pilot "${PILOT:-400000}" \
    --num_centres "${CENTRES:-8}" \
    --probes "${PROBES:-20000}" \
    --out "$ROOT/results/orientation_coverage_${ROBOT:-gvs_pushrod9_o1}.npz"
RC=$?
echo "coverage probe rc=$RC"
exit $RC
