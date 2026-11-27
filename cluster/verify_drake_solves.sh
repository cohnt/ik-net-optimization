#!/bin/bash
# SUBMIT WITH: bash cluster/submit_job.sh cluster/verify_drake_solves.sh -s 8 -q xeon-p8   (a bare LLsub refuses at job start)
#   with the optional second step in the same job (paths relative to the repo, comma- or space-separated):
#   JOB_ENV="NPZ_CONVERT=models/cusp7/cusp7__n6__step620000.pkl,models/cusp7/cusp7__n4__step620000.pkl" \
#       bash cluster/submit_job.sh cluster/verify_drake_solves.sh -s 8 -q xeon-p8
# Verify the Drake pin (release 1.58.0) by SOLVING, on a small CPU job: NO network, NO GPU. The import half is
# cluster/verify_drake_imports.sh (debug-cpu). A smoke test; it produces no number that is reported.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Step A (always), sentinel ~/learned-ik/verify_drake_solves.DONE:
#   one tiny solve each with SNOPT, IPOPT and NLopt, a MultibodyPlant stepping from an inline SDF, and ONE
#   CollisionChecker.CalcRobotClearances call on a two-body plant (ball v wall; the distance is known: 0.85 m).
# Step B (only if NPZ_CONVERT is set), sentinel ~/learned-ik/verify_npz_convert.DONE:
#   scripts/training/convert_pkl_to_npz.py on each path. It reads the .pkl through a restricted unpickler and
#   verifies the .npz bit for bit; it refuses to overwrite. An absolute path or one with `..` is refused here.
#   Thomas must have OK'd loading any cluster pickle (CLAUDE.md).
# The steps are independent: B runs even if A failed, and each writes its own pass/fail sentinel.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
source "$REPO/cluster/verify_staged_tree.sh" || exit 1; lik_job_check "$REPO"   # code integrity (README)
DONE_A="$ROOT/verify_drake_solves.DONE"; DONE_B="$ROOT/verify_npz_convert.DONE"
LOG="$ROOT/verify_drake_solves.log"
rm -f "$DONE_A" "$DONE_B"
exec > >(tee "$LOG") 2>&1
echo "===== drake solve check on $(hostname) at $(date -Is) ====="

export HOME="${TMPDIR:-/tmp}/home.vds.$$"; mkdir -p "$HOME/.cache"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY="$ROOT/venv/bin/python"
RC=0

## ------------------------------------------------------------------ step A --
LEARNED_IK_ROOT="$ROOT" "$PY" - <<'PYSOLVE'
import os
import numpy as np
root = os.environ["LEARNED_IK_ROOT"]
vt = open(os.path.join(root, "drake/share/doc/drake/VERSION.TXT")).read().split()
assert vt[:2] == ["1.58.0", "09bc913cf1af0cdde31073d5f1f7f66ca35ee0d8"], vt
print("Drake", vt)

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
BALL = """<?xml version="1.0"?><sdf version="1.7"><model name="ball"><link name="b">
<inertial><mass>1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
<collision name="c"><geometry><sphere><radius>0.1</radius></sphere></geometry></collision>
</link></model></sdf>"""
WALL = """<?xml version="1.0"?><sdf version="1.7"><model name="wall"><static>true</static>
<link name="w"><pose>1 0 0 0 0 0</pose>
<collision name="c"><geometry><box><size>0.1 1 1</size></box></geometry></collision></link></model></sdf>"""
builder = DiagramBuilder()
plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
Parser(plant).AddModelsFromString(BALL, "sdf")
plant.Finalize()
diagram = builder.Build()
sim = Simulator(diagram)
z0 = plant.GetPositions(plant.GetMyContextFromRoot(sim.get_mutable_context()))[6]
sim.AdvanceTo(0.1)
z1 = plant.GetPositions(plant.GetMyContextFromRoot(sim.get_context()))[6]
assert z1 < z0 - 0.04, (z0, z1)
print(f"MultibodyPlant stepped: free body fell {z0 - z1:.4f} m in 0.1 s")

## CollisionChecker.CalcRobotClearances: the ball (free, 7 positions: quaternion then xyz) is the robot,
## the wall is the environment; the ball's surface is 1 - 0.05 - 0.1 = 0.85 m from the wall's face.
from pydrake.common import Parallelism
from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
rb = RobotDiagramBuilder()
rb.parser().AddModelsFromString(BALL, "sdf")
rb.parser().AddModelsFromString(WALL, "sdf")
model = rb.Build()
ball = model.plant().GetModelInstanceByName("ball")
checker = SceneGraphCollisionChecker(model=model, robot_model_instances=[ball], edge_step_size=0.05)
q = np.array([1.0, 0, 0, 0, 0, 0, 0])
res = checker.CalcRobotClearances([q], 5.0, Parallelism(1))
d = np.asarray(res[0].distances())
J = np.asarray(res[0].jacobians())
assert d.size == 1 and abs(d[0] - 0.85) < 1e-6, d
assert J.shape == (1, 7), J.shape
print(f"CalcRobotClearances: distance {d[0]:.6f} m (expected 0.85), jacobian shape {J.shape}")
PYSOLVE
if [ $? -eq 0 ]; then echo "OK" > "$DONE_A"; else echo "FAIL solve / CalcRobotClearances check" > "$DONE_A"; RC=1; fi
echo "step A sentinel: $(cat "$DONE_A")"

## ------------------------------------------------------------------ step B --
if [ -n "${NPZ_CONVERT:-}" ]; then
    echo "===== step B: pkl -> npz for: ${NPZ_CONVERT} ====="
    cd "$REPO" || { echo "FAIL no repo" > "$DONE_B"; exit 1; }
    OKB=1
    for f in ${NPZ_CONVERT//,/ }; do
        case "$f" in /*|*..*) echo "REFUSING path '$f' (must be relative, inside the repo)"; OKB=0; continue ;; esac
        [ -f "$f" ] || { echo "no such file: $f"; OKB=0; continue; }
        "$PY" scripts/training/convert_pkl_to_npz.py "$f" || { echo "convert failed: $f"; OKB=0; }
    done
    if [ "$OKB" = 1 ]; then echo "OK" > "$DONE_B"; else echo "FAIL see $LOG" > "$DONE_B"; RC=1; fi
    echo "step B sentinel: $(cat "$DONE_B")"
else
    echo "step B skipped (NPZ_CONVERT unset)"
fi
echo "===== done at $(date -Is); rc $RC ====="
exit $RC
