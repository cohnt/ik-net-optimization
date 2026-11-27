#!/bin/bash
# SUBMIT WITH: bash cluster/submit_job.sh cluster/verify_drake_imports.sh -s 8 -q debug-cpu   (a bare LLsub refuses at job start, README "Code integrity")
# Verify the Drake at ~/learned-ik/drake (the pin, release 1.58.0) on a compute node with NO internet:
# IMPORTS AND AVAILABILITY ONLY. No solve, no CalcRobotClearances call -- a debug partition takes pure
# yes/no plumbing (cluster/verify_drake_solves.sh is the solve half, on xeon-p8).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Checks (any failure writes FAIL to the sentinel ~/learned-ik/verify_drake_imports.DONE; OK on success):
#   1. share/doc/drake/VERSION.TXT reads "1.58.0 <commit>" and .drake-ok holds the same pin stamp.
#   2. `import pydrake` resolves into ~/learned-ik/drake and reports its version.
#   3. SnoptSolver available+enabled, IpoptSolver available, NloptSolver available.
#   4. Every NLopt option accessor of PR 25002 (NLOPT_POST_1_56_OPTIONS, src/generic_program.py) exists on
#      NloptSolver (NloptOptionSurface reads class attributes only; nothing is constructed or solved).
#   5. CollisionChecker has the attribute CalcRobotClearances (not called).
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
source "$REPO/cluster/verify_staged_tree.sh" || exit 1; lik_job_check "$REPO"   # code integrity: tree == laptop manifest (README)
DONE="$ROOT/verify_drake_imports.DONE"
LOG="$ROOT/verify_drake_imports.log"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
echo "===== drake import check on $(hostname) at $(date -Is) ====="
Fail() { echo "VERIFY FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

PIN="drake 1.58.0 09bc913cf1af0cdde31073d5f1f7f66ca35ee0d8"
[ "$(cat "$ROOT/.drake-ok" 2>/dev/null)" = "$PIN" ] || Fail ".drake-ok is '$(cat "$ROOT/.drake-ok" 2>/dev/null)', not '$PIN' (setup_supercloud.sh not run, or the old drake not moved aside)"

export HOME="${TMPDIR:-/tmp}/home.vdi.$$"; mkdir -p "$HOME/.cache"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="$ROOT/venv/bin/python"

LEARNED_IK_REPO="$REPO" LEARNED_IK_ROOT="$ROOT" "$PY" - <<'PYCHK' || Fail "import / availability check"
import os, sys
root = os.environ["LEARNED_IK_ROOT"]
vt = open(os.path.join(root, "drake/share/doc/drake/VERSION.TXT")).read().split()
print("VERSION.TXT", vt)
assert vt[:2] == ["1.58.0", "09bc913cf1af0cdde31073d5f1f7f66ca35ee0d8"], vt
import pydrake
p = os.path.realpath(pydrake.__file__)
assert p.startswith(os.path.realpath(os.path.join(root, "drake"))), f"pydrake from {p}, not $ROOT/drake"
print("pydrake", getattr(pydrake, "__version__", "?"), "from", p)
from pydrake.solvers import SnoptSolver, IpoptSolver, NloptSolver
s = SnoptSolver(); assert s.available() and s.enabled(), "SNOPT unavailable"
assert IpoptSolver().available(), "IPOPT unavailable"
assert NloptSolver().available(), "NLopt unavailable"
print("SNOPT available+enabled | IPOPT available | NLopt available")
sys.path.insert(0, os.environ["LEARNED_IK_REPO"])
from src.generic_program import NloptOptionSurface, NLOPT_POST_1_56_OPTIONS
surface = NloptOptionSurface()
missing = sorted(o.accessor for o in NLOPT_POST_1_56_OPTIONS.values() if o.accessor not in surface)
assert not missing, f"NLopt options missing (PR 25002): {missing}"
print(f"NLopt: all {len(NLOPT_POST_1_56_OPTIONS)} post-1.56 option accessors present; surface {sorted(surface.values())}")
from pydrake.planning import CollisionChecker
assert hasattr(CollisionChecker, "CalcRobotClearances"), "no CollisionChecker.CalcRobotClearances"
print("CollisionChecker.CalcRobotClearances: attribute present (not called)")
PYCHK

echo "OK" > "$DONE"
echo "===== drake imports verified at $(date -Is); sentinel $DONE ====="
