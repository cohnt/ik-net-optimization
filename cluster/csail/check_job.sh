#!/bin/bash
# JOB PAYLOAD: the GPU plumbing check, on a real GPU of the pinned kind (submitted by submit_check.sh).
# Verifies, in order: the staged tree and the image (sha256) are the ones the laptop anchored; where this runs
# (machine record); nvidia-smi; torch.cuda available (exit non-zero otherwise: the job shows FAILED and no OK
# sentinel is written); one real kernel launch; the device matches EXPECT_GPU; pydrake comes from the Drake pin
# (release 1.58.0, VERSION.TXT) with IPOPT and CollisionChecker.CalcRobotClearances; the project's imports resolve
# (ikflow is the vendored fork); the hardened Panda scene builds with the image's warmed drake_models cache.
# The sentinel $LIK_ROOT/check.<jobid>.DONE says "OK" only if everything passed.
set -uo pipefail
# Slurm runs a COPY of this script from its spool directory, so BASH_SOURCE says nothing about the tree: use the root.
HERE="${LIK_ROOT:?LIK_ROOT unset}/repo/cluster/csail"
. "$HERE/job_lib.sh"
lik_start
lik_end_epoch
lik_machine_record
SENT="$LIK_ROOT/check.$JOBTAG.DONE"; rm -f "$SENT"
fail() { echo "CHECK FAILED: $1"; echo "FAIL $1" > "$SENT"; exit "${2:-1}"; }

lik_gpu_check --drake || fail "gpu_check" 3

## HOME as run_items.sh sets it per worker: the image's warmed ikflow / drake_models caches linked into a fresh HOME.
CHK_HOME="$TMPDIR/home.chk"; mkdir -p "$CHK_HOME/.cache"
ln -sfn /opt/lik/home/.cache/ikflow "$CHK_HOME/.cache/ikflow"; ln -sfn /opt/lik/home/.cache/drake "$CHK_HOME/.cache/drake"
lik_container env "HOME=$CHK_HOME" /opt/lik/venv/bin/python - <<'PYEOF' || fail "project imports / scene" 4
import os, sys
sys.path.insert(0, os.environ["LEARNED_IK_ROOT"] + "/repo")
import ikflow, ikflow.model_loading, jrl, FrEIA, klampt, jax
assert "/opt/lik/ikflow" in ikflow.model_loading.__file__, ikflow.model_loading.__file__
print("ikflow (vendored fork):", ikflow.model_loading.__file__)
from src.utils import BuildEnv, RepoDir
BuildEnv(meshcat=None, directives_file=os.path.join(RepoDir(), "models/panda/panda_finray_collision_hardened.yaml"))
print("hardened Panda scene builds")
PYEOF

echo "OK" > "$SENT"
echo "===== check OK on $(hostname) at $(date -Is) ====="
