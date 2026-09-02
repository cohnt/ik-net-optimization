#!/bin/bash
# The ONLY thing this project runs on debug-gpu: a few-minute smoke test.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit from ~/learned-ik/repo on the login node:
#     LLsub ./cluster/smoke.sh -g volta:2 -s 40 -q debug-gpu -T 0:20:00
#
# Standing rule 6: debug partitions take quick smoke tests only. Nothing that
# produces a measurement runs here -- the calibration is a real experiment and
# goes to xeon-g6-volta. debug-gpu is also not ExclusiveUser, so a long job here
# takes capacity from other people's quick checks.
#
# The decisive check is phase 1's KERNEL LAUNCH. PyTorch 2.11's cu128 wheels
# dropped sm_70, so a wrong wheel imports cleanly, reports a CUDA device, and
# only fails when it first tries to run something -- "no kernel image is
# available for execution on the device". `get_arch_list()` is necessary but not
# sufficient; actually multiplying two matrices is.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
LOG="$ROOT/smoke.log"
DONE="$ROOT/smoke.DONE"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
set -x

export HOME="${TMPDIR:-/tmp}/home.smoke"
mkdir -p "$HOME/.cache"
ln -sfn "$ROOT/home/.cache/ikflow" "$HOME/.cache/ikflow" 2>/dev/null || true
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl.smoke"
mkdir -p "$MPLCONFIGDIR"
PY="$ROOT/venv/bin/python"
cd "$REPO" || { echo "no repo at $REPO"; echo FAIL > "$DONE"; exit 1; }

hostname; date -Is; nvidia-smi -L; echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES:-unset}]"
echo "staged commit: $(cat .staged-commit 2>/dev/null || echo UNKNOWN)"

Fail() { set +x; echo "SMOKE FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

## --- 1. the GPU is real and this torch can run on it -----------------------
"$PY" - <<'PYEOF' || Fail "phase 1: GPU / torch"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
assert torch.cuda.is_available(), "no CUDA device visible"
cap = torch.cuda.get_device_capability()
print("device", torch.cuda.get_device_name(0), "capability", cap)
print("arch list", torch.cuda.get_arch_list())
assert "sm_70" in torch.cuda.get_arch_list(), "this torch has no sm_70; V100 cannot run it"
# The decisive test: actually launch a kernel. A cu128 wheel gets this far and
# then raises "no kernel image is available for execution on the device".
x = torch.randn(512, 512, device="cuda", dtype=torch.float64)
print("kernel launch OK, checksum", float((x @ x).sum()))
PYEOF

## --- 2. Drake, the flows, and the robots, all with no network --------------
"$PY" - <<'PYEOF' || Fail "phase 2: drake / ikflow / jrl"
from pydrake.solvers import IpoptSolver, SnoptSolver
assert IpoptSolver().available(), "IPOPT unavailable"
print("IPOPT", IpoptSolver().available(), "| SNOPT", SnoptSolver().available())
from ikflow.model_loading import get_ik_solver
get_ik_solver("panda__full__lp191_5.25m")
print("panda flow loaded from cache")
import jrl.robots
for name in ("panda", "iiwa14"):
    jrl.robots.get_robot(name)
    print("jrl robot OK:", name)
PYEOF

## --- 3. one cell per (robot, task), every arm ------------------------------
# --compile also proves torch.compile/inductor/triton work offline against the
# system gcc, which the campaign depends on and nothing else here exercises.
"$PY" -u scripts/panda/panda_benchmark.py --task pose --targets 1 --guesses 1 \
    --wall-time 20 --arms learned,numerical,analytic,analytic8 --config latent \
    --compile --tag smoke_panda_pose || Fail "panda pose"
"$PY" -u scripts/panda/panda_benchmark.py --task mug --targets 1 --guesses 1 \
    --wall-time 20 --arms learned,numerical,analytic,analytic8 --config latent \
    --compile --tag smoke_panda_mug || Fail "panda mug"
"$PY" -u scripts/iiwa/iiwa_benchmark.py --task pose --targets 1 --guesses 1 \
    --wall-time 20 --arms learned,numerical --config latent \
    --compile --tag smoke_iiwa_pose || Fail "iiwa pose"
"$PY" -u scripts/iiwa/iiwa_benchmark.py --task mug --targets 1 --guesses 1 \
    --wall-time 20 --arms learned,numerical --config latent \
    --compile --tag smoke_iiwa_mug || Fail "iiwa mug"

## --- 4. the harness's own invariant ---------------------------------------
# Under the paired protocol the learned and joint-space arms must start EXACTLY
# at q_init. A nonzero start_q_error there means the flow inversion or the
# conditioning-frame calibration is wrong on this machine, which would silently
# void every learned column.
set +x
"$PY" - <<'PYEOF' || Fail "phase 4: paired-start invariant"
import glob, json, sys
bad = 0
for path in sorted(glob.glob("results/*/benchmark/smoke_*/summary.json")):
    s = json.load(open(path))["summary"]
    for arm in ("learned", "numerical"):
        if arm not in s: continue
        err = s[arm]["median_start_q_error"]
        flag = "" if err is not None and err < 1e-4 else "   <-- NOT EXACT"
        if flag: bad += 1
        print(f"{path.split('/')[-2]:<20} {arm:<10} start_q_error {err}{flag}")
sys.exit(1 if bad else 0)
PYEOF

echo OK > "$DONE"
echo "===== smoke complete at $(date -Is) ====="
