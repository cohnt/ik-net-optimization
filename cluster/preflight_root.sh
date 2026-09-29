#!/bin/bash
# Pre-flight for a cluster tree: does a benchmark worker's environment actually resolve?
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit as a job (never on a login node) -- a smoke test, so debug-cpu is the right pool:
#   LEARNED_IK_ROOT=$HOME/learned-ik-gvs ROBOT=gvs_pushrod9_o1 \
#     LLsub ./cluster/preflight_root.sh -s 8 -q debug-cpu -T 00:20:00 -J gvs_cal_preflight
#
# WHY IT EXISTS. An isolated tree can be complete enough to build datasets and still be
# missing what a BENCHMARK worker needs, because run_items.sh -- not the payload -- is what
# puts Drake on PYTHONPATH, the extracted system libraries on LD_LIBRARY_PATH, and the
# drake_models cache where ProcessModelDirectives can find it (compute nodes cannot
# download it). A tree with only a venv resolves none of that, and the failure lands per
# cell inside a queued campaign rather than here. For the GVS push-rod arm there is one
# more thing a fresh venv can lack: SoRoMoX and a CPU jaxlib, which the forward model is.
#
# Named *_cal_* so stage_code.sh's live-campaign guard ignores it: it produces no records.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
ROBOT="${ROBOT:-gvs_pushrod9_o1}"

## Exactly what run_items.sh sets up for a worker, minus the per-worker TMPDIR.
export HOME="$ROOT/home"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GVS_ARM_XLA_THREADS=1
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

echo "ROOT=$ROOT  ROBOT=$ROBOT"
echo "staged commit: $(cat "$REPO/.staged-commit" 2>/dev/null || echo '(none)')"
cd "$REPO" || exit 2

"$ROOT/venv/bin/python" - "$ROBOT" <<'PY'
import os
import sys
import time

robot = sys.argv[1]
print("--- Drake")
import pydrake.all as drake
print("   pydrake from", os.path.dirname(drake.__file__))

print("--- the forward model's runtime: SoRoMoX on a CPU jaxlib, float64")
import jax, soromox, optimistix
print("   jax", jax.__version__, "backend", jax.default_backend(), "soromox", soromox.__version__,
      "optimistix", optimistix.__version__)
assert jax.default_backend() == "cpu", "the forward model must run on the CPU beside the flow"

print("--- this project's robots")
import src.register_robots as rr
print("   registered:", rr.ProjectRobotNames())
assert robot in rr.ProjectRobotNames(), f"{robot} is not a registered robot"

print("--- the scene builds, with collision geometry")
from src.utils import BuildEnv, HiddenPrints
with HiddenPrints():
    diagram = BuildEnv(
        meshcat=None,
        directives_file=f"models/{robot}/{robot}_collision_hardened.yaml")
plant = diagram.GetSubsystemByName("plant")
print("   positions:", plant.num_positions(), " bodies:", plant.num_bodies())

if robot.startswith("gvs_"):
    print("--- one equilibrium solve and one implicit Jacobian, jitted")
    import numpy as np
    from src.gvs_arm.params import GetSpec
    from src.gvs_arm.model import GetModel, PlantSlotMap
    spec = GetSpec(robot)
    model = GetModel(spec)
    start = time.time(); model.WarmUp(); print(f"   warm-up (build + jit): {time.time() - start:.1f} s")
    cfg = np.zeros(spec.ninputs); cfg[0] = 0.5
    start = time.time(); q = model.PlantQ(cfg); print(f"   PlantQ: {1e3 * (time.time() - start):.1f} ms")
    start = time.time(); J, _ = model.PlantJacobian(cfg); print(f"   PlantJacobian {J.shape}: {1e3 * (time.time() - start):.1f} ms")
    picks = PlantSlotMap(plant, spec)
    assert plant.num_positions() == spec.num_plant_positions == len(picks)
    print("   plant slot map read from the plant:", len(picks), "positions")
    print("--- the dataset sampler, one small batch")
    from jrl.robots import get_robot
    r = get_robot(robot)
    start = time.time(); s, p = r.sample_joint_angles_and_poses(500); print(f"   500 samples: {time.time() - start:.1f} s, shapes {s.shape} {p.shape}")

print("--- the dataset this tree will train on")
from ikflow.config import DATASET_DIR
d = os.path.join(DATASET_DIR, robot)
print("   DATASET_DIR:", DATASET_DIR)
print("   %s: %s" % (robot, "present with .DONE" if os.path.exists(os.path.join(d, ".DONE"))
                     else "MISSING -- train_flow.sh will hard-fail (expected before datagen)"))

print("\nPREFLIGHT OK")
PY
