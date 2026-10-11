#!/bin/bash
# Verify ~/learned-ik/venv-crc on a compute node with NO internet, the way a DRAKE_ENV=crc worker
# will run it. A smoke test (does it import, does the binding exist, does the parity test pass):
# it produces no number that is reported.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit from ~/learned-ik/scratch-crc on the login node, after make_venv_crc.sh wrote OK:
#     LLsub ./cluster/verify_venv_crc.sh -s 8 -q debug-cpu
#
# Checks, in order (any failure writes FAIL to the sentinel):
#   1. The worker environment run_items.sh builds under DRAKE_ENV=crc: PYTHONPATH without the
#      tarball, no sysdeps on LD_LIBRARY_PATH, OMP_NUM_THREADS=1, DRAKE_NUM_THREADS set, a private
#      HOME linking ~/learned-ik/home/.cache/ikflow and ~/learned-ik/home-crc/.cache/drake.
#   2. pydrake loads from venv-crc; CollisionChecker.CalcRobotClearances exists; one small solve
#      each with SNOPT, IPOPT and NLopt; a MultibodyPlant from an inline SDF steps; Parallelism
#      honours DRAKE_NUM_THREADS over OMP_NUM_THREADS=1.
#   3. torch imports and runs on the CPU (no GPU here), jax is CPU-only, ikflow is the staged fork.
#   4. Every svgd-context scene (Panda, iiwa x grasp, pose, hardened) builds OFFLINE from the
#      home-crc model cache.
#   5. THE REAL TEST: tests/test_collision_clearances.py from the svgd tip in $TREE passes, and
#      does not take its SKIP branch (the SKIP branch is what a pydrake without the binding runs).
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
VENV="$ROOT/venv-crc"
TREE="${CRC_TEST_TREE:-$ROOT/scratch-crc}"
LOG="$ROOT/venv-crc-build/verify_venv_crc.log"
DONE="$ROOT/verify_venv_crc.DONE"
mkdir -p "$(dirname "$LOG")"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
echo "===== venv-crc verification on $(hostname) at $(date -Is) ====="
Fail() { echo "VERIFY FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }
[ "$(cat "$ROOT/make_venv_crc.DONE" 2>/dev/null)" = "OK" ] || Fail "make_venv_crc.DONE is not OK"
[ -d "$TREE/tests" ] || Fail "no test tree at $TREE"

## ------------------------------------------------ 1. the DRAKE_ENV=crc worker env --
unset PYTHONPATH
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "^$ROOT/" | paste -sd: -)"
[ -n "$LD_LIBRARY_PATH" ] || unset LD_LIBRARY_PATH
export HOME="${TMPDIR:-/tmp}/home.crc.$$"
mkdir -p "$HOME/.cache"
ln -sfn "$ROOT/home/.cache/ikflow" "$HOME/.cache/ikflow"
ln -sfn "$ROOT/home-crc/.cache/drake" "$HOME/.cache/drake"
## Before OMP_NUM_THREADS is set: GNU nproc reports that variable when it is set (run_items.sh
## uses `env -u` for the same reason).
NCPU="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GVS_ARM_XLA_THREADS=1
export DRAKE_NUM_THREADS="$NCPU"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
PY="$VENV/bin/python"
echo "PYTHONPATH=[${PYTHONPATH:-unset}] LD_LIBRARY_PATH=[${LD_LIBRARY_PATH:-unset}]"
echo "HOME=$HOME OMP_NUM_THREADS=$OMP_NUM_THREADS DRAKE_NUM_THREADS=$DRAKE_NUM_THREADS nproc=$NCPU"
echo "test tree: $TREE at $(cat "$TREE/.staged-commit" 2>/dev/null || echo UNKNOWN)"

## ---------------------------------------------------------- 2-3. imports and solves --
echo "===== [2-3] pydrake, solvers, plant, torch, jax, ikflow ====="
"$PY" - "$VENV" "$ROOT" <<'PYCHK' || Fail "import / solve check"
import os, sys
venv, root = (os.path.realpath(a) for a in sys.argv[1:3])
assert not any("/drake/lib/python" in p for p in sys.path), sys.path
import numpy as np
import pydrake
from pydrake.planning import CollisionChecker
p = os.path.realpath(pydrake.__file__)
assert p.startswith(venv), f"pydrake from {p}, not {venv}"
import importlib.metadata as md
print("pydrake", md.version("drake"), "from", os.path.dirname(p))
assert hasattr(CollisionChecker, "CalcRobotClearances"), "no CalcRobotClearances"
print("CollisionChecker.CalcRobotClearances: present")

from pydrake.common import Parallelism
n = Parallelism.Max().num_threads()
assert n == int(os.environ["DRAKE_NUM_THREADS"]), (n, os.environ["DRAKE_NUM_THREADS"])
print(f"Parallelism.Max() = {n} threads (OMP_NUM_THREADS=1, DRAKE_NUM_THREADS={n})")

from pydrake.solvers import MathematicalProgram, SnoptSolver, IpoptSolver, NloptSolver
for S in (SnoptSolver, IpoptSolver, NloptSolver):
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(2, "x")
    prog.AddCost((x[0] - 1) ** 2 + (x[1] - 2) ** 2)
    prog.AddConstraint(x[0] + x[1] <= 2.5)
    s = S()
    assert s.available() and s.enabled(), S.__name__
    r = s.Solve(prog, np.zeros(2))
    assert r.is_success(), (S.__name__, r.get_solution_result())
    xs = r.GetSolution(x)
    assert np.allclose(xs, [0.75, 1.75], atol=1e-4), (S.__name__, xs)
    print(f"{S.__name__}: solved, x = {np.round(xs, 6)}")

from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
SDF = """<?xml version="1.0"?><sdf version="1.7"><model name="ball"><link name="b">
<inertial><mass>1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
</link></model></sdf>"""
builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
Parser(plant).AddModelsFromString(SDF, "sdf")
plant.Finalize()
diagram = builder.Build()
sim = Simulator(diagram)
z0 = plant.GetPositions(plant.GetMyContextFromRoot(sim.get_mutable_context()))[6]
sim.AdvanceTo(0.1)
z1 = plant.GetPositions(plant.GetMyContextFromRoot(sim.get_context()))[6]
assert z1 < z0 - 0.04, (z0, z1)
print(f"MultibodyPlant stepped: free body fell {z0 - z1:.4f} m in 0.1 s")

import torch, jax, ikflow
assert not torch.cuda.is_available()
a = torch.randn(64, 64, dtype=torch.float64) + 16 * torch.eye(64, dtype=torch.float64)
assert torch.allclose(a @ torch.linalg.inv(a), torch.eye(64, dtype=torch.float64), atol=1e-8)
assert "sm_70" in torch.cuda.get_arch_list()
print("torch", torch.__version__, "on the CPU path: ok (cuda available:", torch.cuda.is_available(), ")")
assert jax.default_backend() == "cpu"
assert "repo/third_party/ikflow" in os.path.realpath(ikflow.__file__)
print("jax", jax.__version__, jax.default_backend(), "| ikflow", os.path.dirname(os.path.realpath(ikflow.__file__)))
PYCHK

## ------------------------------------------- provenance: what a record will carry --
LEARNED_IK_REPO="$TREE" "$PY" - <<'PYMETA' || Fail "benchmark metadata"
import os, sys
sys.path.insert(0, os.environ["LEARNED_IK_REPO"])
from src import benchmark
m = {k: v for k, v in benchmark.provenance().items() if k.startswith("drake")}
print("benchmark provenance:", m)
assert "crc.15a2c97" in m.get("drake_version_txt", ""), m
PYMETA

## ------------------------------------------------------------- 4. offline scenes --
echo "===== [4] the svgd-context scenes build offline from home-crc ====="
LEARNED_IK_REPO="$TREE" "$PY" - <<'PYSCENE' || Fail "offline scene build"
import os, sys
sys.path.insert(0, os.environ["LEARNED_IK_REPO"])
from src.utils import BuildEnv
from src.target_screening import SceneFile
for robot in ("panda", "iiwa"):
    for task in ("mug", "pose"):
        BuildEnv(meshcat=None, directives_file=SceneFile(robot, task, "hardened"))
        print("offline scene builds:", robot, task)
PYSCENE

## ------------------------------------------------- 5. the clearance parity test --
echo "===== [5] tests/test_collision_clearances.py ====="
cd "$TREE" || Fail "cd $TREE"
OUT="$ROOT/venv-crc-build/test_collision_clearances.out"
"$PY" -u tests/test_collision_clearances.py > "$OUT" 2>&1
RC=$?
cat "$OUT"
[ $RC -eq 0 ] || Fail "test_collision_clearances.py exited $RC"
grep -q '^SKIP' "$OUT" && Fail "the test took its SKIP branch: this pydrake has no binding"
grep -q '^ALL PASS' "$OUT" || Fail "no ALL PASS line"
[ "$DRAKE_NUM_THREADS" -gt 1 ] || Fail "only one Drake thread: the thread-independence check was vacuous"

echo "OK" > "$DONE"
echo "===== venv-crc verified at $(date -Is); sentinel $DONE ====="
