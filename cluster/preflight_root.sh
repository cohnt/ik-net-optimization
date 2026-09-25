#!/bin/bash
# Pre-flight for a cluster tree: does a benchmark worker's environment actually resolve?
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit as a job (never on a login node):
#   LEARNED_IK_ROOT=$HOME/learned-ik-helix ROBOT=helix7_p050 \
#     LLsub ./cluster/preflight_root.sh -s 8 -q debug-cpu -T 00:20:00 -J helix_cal_preflight
#
# WHY IT EXISTS. An isolated tree can be complete enough to build datasets and still be
# missing what a BENCHMARK worker needs, because run_items.sh -- not the payload -- is what
# puts Drake on PYTHONPATH, the extracted system libraries on LD_LIBRARY_PATH, and the
# drake_models cache where ProcessModelDirectives can find it (compute nodes cannot
# download it). A tree with only a venv resolves none of that, and the failure lands per
# cell inside a queued campaign rather than here.
#
# Named *_cal_* so stage_code.sh's live-campaign guard ignores it: it produces no records.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
ROBOT="${ROBOT:-helix7_p050}"

## Exactly what run_items.sh:125-142 sets up for a worker, minus the per-worker TMPDIR.
export HOME="$ROOT/home"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

echo "ROOT=$ROOT  ROBOT=$ROBOT"
echo "staged commit: $(cat "$REPO/.staged-commit" 2>/dev/null || echo '(none)')"
cd "$REPO" || exit 2

"$ROOT/venv/bin/python" - "$ROBOT" <<'PY'
import os
import sys

robot = sys.argv[1]
print("--- Drake")
import pydrake.all as drake
print("   pydrake from", os.path.dirname(drake.__file__))

print("--- this project's robots")
import src.register_robots as rr
print("   registered:", rr.ProjectRobotNames())

print("--- the scene builds, with collision geometry")
from src.utils import BuildEnv, HiddenPrints
with HiddenPrints():
    diagram = BuildEnv(
        meshcat=None,
        directives_file=f"models/{robot}/{robot}_collision_hardened.yaml")
plant = diagram.GetSubsystemByName("plant")
print("   positions:", plant.num_positions(), " bodies:", plant.num_bodies())

print("--- the screw joint's limits, which no parser preserves")
from src.helix_arm.limits import ApplyScrewJointLimits
from src.helix_arm.params import GetSpec
spec = GetSpec(robot)
print("   before repair:", plant.GetPositionLowerLimits()[:3], "...")
ApplyScrewJointLimits(plant, spec)
lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
import numpy as np
assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), "limits still non-finite"
print("   after repair: all finite, screw row +-%.4f" % hi[
    plant.GetJointByName(spec.screw_joint_names[0]).position_start()])

print("--- the dataset this tree will train on")
from ikflow.config import DATASET_DIR
d = os.path.join(DATASET_DIR, robot)
print("   DATASET_DIR:", DATASET_DIR)
print("   %s: %s" % (robot, "present with .DONE" if os.path.exists(os.path.join(d, ".DONE"))
                     else "MISSING -- train_flow.sh will hard-fail"))

print("\nPREFLIGHT OK")
PY
