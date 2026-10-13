#!/bin/bash
# Build ~/learned-ik/venv-crc: the project venv's exact environment, with Drake from the
# `calc-robot-clearances` WHEEL in place of the nightly tarball on PYTHONPATH. Not idempotent:
# it refuses if venv-crc exists (rm -rf it to rebuild; nothing else depends on it).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit from ~/learned-ik/scratch-crc (or any tree holding this file) on the login node; the
# `download` partition is the only one with internet and runs ONE job at a time:
#     LLsub ./cluster/make_venv_crc.sh -s 8 -q download
# Then verify on a node with NO internet: cluster/verify_venv_crc.sh (debug-cpu).
#
# WHY A SECOND VENV (Thomas, 2026-10-11): the svgd solver's batched clearance collision row
# (`svgd_collision_backend=clearances`, src/svgd/collision_backend.py) needs
# `CollisionChecker.CalcRobotClearances`, which only his Drake branch carries. The svgd rounds
# from R11 on run under it (`DRAKE_ENV=crc`, cluster/run_items.sh); every other stage -- the
# record, the IPOPT twins -- keeps the tarball pin through ~/learned-ik/venv, untouched. A wheel
# must not share a venv with a tarball Drake on PYTHONPATH (two pydrakes, different bundled libs),
# hence a separate venv rather than a pip install into the old one.
#
# WHAT "SAME ENVIRONMENT" MEANS, and how it is proved rather than assumed:
#   1. `pip freeze` of ~/learned-ik/venv is taken at job time and installed into the new venv with
#      --no-deps, every version pinned exactly (torch's `+cu126` from the cu126 index). No resolve
#      happens, so nothing can drift.
#   2. ikflow is the editable install of the vendored fork in ~/learned-ik/repo/third_party/ikflow.
#      Its three site-packages entries (the .pth, the PEP 660 finder and the dist-info) are COPIED
#      from the old venv, so it resolves to the same staged source and a restage updates both
#      venvs alike. `pip install -e` is not used because it writes build metadata into the staged
#      tree, which running jobs read.
#   3. The freeze is diffed against the reference BEFORE Drake goes in (must be identical) and
#      AFTER (only `drake` and the wheel's own new dependencies may appear; nothing may change or
#      disappear). The job fails otherwise.
# The wheel brings numpy, matplotlib, PyYAML, pydot and Mosek as requirements; the first three are
# already pinned by the freeze and are satisfied, so pip leaves them alone.
#
# THE DRAKE MODEL CACHE goes to ~/learned-ik/home-crc, not ~/learned-ik/home: the iiwa scenes
# fetch `package://drake_models/...` lazily, compute nodes have no internet, and the cache key
# belongs to the Drake build. Warming into the shared home would write next to what running jobs
# read; a separate home costs one ~MB-scale download and isolates the two Drakes completely.
# run_items.sh links it under DRAKE_ENV=crc. The ikflow weight cache stays shared (read-only).
#
# NOTE: source /etc/profile BEFORE `set -u` (Z97-byobu.sh reads an unset LC_BYOBU).
source /etc/profile
set -uo pipefail
## The wheel's pydrake must be the ONLY pydrake this venv sees (the skill's rule for wheels).
unset PYTHONPATH

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
VENV="$ROOT/venv-crc"
REF="$ROOT/venv"
## The tree whose src/ warms the scenes: the svgd tip (SceneFile, the hardened scenes). Reading it
## only; the campaign's ~/learned-ik/repo is never written.
SRC_TREE="${CRC_SRC_TREE:-$ROOT/scratch-crc}"
WHEEL_NAME="drake-0.0.20261011+crc.15a2c97-cp312-abi3-manylinux_2_34_x86_64.whl"
## sha256 of the exact bytes verified on the workstation (branch calc-robot-clearances, 15a2c97228).
WHEEL_SHA256="aae70be87d8c981588c8034aeffd2b1ee3f5e6de66568bf53ba8b47f7fd44ecc"
WHEEL="$ROOT/wheels/$WHEEL_NAME"
WORK="$ROOT/venv-crc-build"
LOG="$WORK/make_venv_crc.log"
DONE="$ROOT/make_venv_crc.DONE"

mkdir -p "$WORK"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
echo "===== venv-crc build on $(hostname) at $(date -Is) ====="
Fail() { echo "BUILD FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

[ -x "$REF/bin/python" ] || Fail "no reference venv at $REF"
[ -e "$VENV" ] && Fail "$VENV already exists; rm -rf it to rebuild (nothing else depends on it)"
[ -f "$WHEEL" ] || Fail "no wheel at $WHEEL (sc_rsync it to $ROOT/wheels/ first)"
GOT=$(sha256sum "$WHEEL" | cut -d' ' -f1)
[ "$GOT" = "$WHEEL_SHA256" ] || Fail "wheel checksum $GOT, expected $WHEEL_SHA256"
[ -d "$SRC_TREE/src" ] || Fail "no source tree at $SRC_TREE (rsync the svgd tip there first)"

## ---------------------------------------------------------- 1. the reference --
echo "===== [1/6] reference freeze of $REF ====="
"$REF/bin/python" -m pip freeze --all 2>/dev/null | grep -viE '^(pip|wheel)==' | sort > "$WORK/freeze.reference.txt" \
    || Fail "pip freeze of the reference venv"
## The editable ikflow line is reproduced by copying (step 3), not by pip.
grep -vE '^(-e |#)' "$WORK/freeze.reference.txt" > "$WORK/requirements.txt"
echo "$(wc -l < "$WORK/freeze.reference.txt") distributions in the reference, $(wc -l < "$WORK/requirements.txt") to pip"
grep -E '^torch==' "$WORK/requirements.txt" | grep -q '+cu126' || Fail "reference torch is not a +cu126 build"

## ----------------------------------------------------------------- 2. venv --
echo "===== [2/6] venv + pinned requirements (--no-deps) ====="
python3 -m venv --without-pip "$VENV" || Fail "venv creation"
wget -q -O "$WORK/get-pip.py" https://bootstrap.pypa.io/get-pip.py || Fail "get-pip download"
"$VENV/bin/python" "$WORK/get-pip.py" --quiet || Fail "pip bootstrap"
## PyPI first, the cu126 index for torch's local version. Every line is an exact pin and nothing is
## resolved, so which index serves an identical nvidia-* wheel does not matter.
"$VENV/bin/pip" install --quiet --no-deps --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu126 -r "$WORK/requirements.txt" \
    || Fail "pinned requirements"

## ------------------------------------------------------- 3. editable ikflow --
echo "===== [3/6] editable ikflow (copied from the reference venv) ====="
SP_REF="$REF/lib/python3.12/site-packages"
SP="$VENV/lib/python3.12/site-packages"
shopt -s nullglob
IK=("$SP_REF"/__editable__.ikflow-*.pth "$SP_REF"/__editable___ikflow_*_finder.py "$SP_REF"/ikflow-*.dist-info)
shopt -u nullglob
[ "${#IK[@]}" -eq 3 ] || Fail "expected 3 ikflow editable entries in $SP_REF, found ${#IK[@]}: ${IK[*]}"
cp -a "${IK[@]}" "$SP/" || Fail "copying the ikflow editable entries"

"$VENV/bin/python" -m pip freeze --all 2>/dev/null | grep -viE '^(pip|wheel)==' | sort > "$WORK/freeze.before_drake.txt"
if ! diff "$WORK/freeze.reference.txt" "$WORK/freeze.before_drake.txt"; then
    Fail "venv-crc before Drake is not the reference venv (diff above)"
fi
echo "before Drake: identical to the reference ($(wc -l < "$WORK/freeze.before_drake.txt") lines)"

## ---------------------------------------------------------------- 4. Drake --
echo "===== [4/6] the Drake wheel (with its own requirements) ====="
"$VENV/bin/pip" install --quiet "$WHEEL" || Fail "wheel install"
"$VENV/bin/python" -m pip freeze --all 2>/dev/null | grep -viE '^(pip|wheel)==' | sort > "$WORK/freeze.after_drake.txt"
REMOVED=$(comm -23 "$WORK/freeze.reference.txt" "$WORK/freeze.after_drake.txt")
ADDED=$(comm -13 "$WORK/freeze.reference.txt" "$WORK/freeze.after_drake.txt")
echo "--- added by the wheel:"; echo "$ADDED"
[ -z "$REMOVED" ] || Fail "the wheel changed or removed reference distributions: $REMOVED"
## Allowed: drake itself and requirements the reference does not have (pydot, Mosek). Anything
## else means pip moved something it should have left alone.
BAD=$(echo "$ADDED" | grep -viE '^(drake|mosek|pydot)( @|==)' || true)
[ -z "$BAD" ] || Fail "unexpected additions: $BAD"
echo "$ADDED" | grep -qiE '^drake( @|==)' || Fail "drake is not in the freeze"
"$VENV/bin/python" -m pip check || echo "NOTE: pip check reports the above (the reference venv's jrl/ikflow pins torch==2.0.1 by metadata; --no-deps is deliberate)"

## ----------------------------------------------------------- 5. import check --
echo "===== [5/6] imports (PYTHONPATH unset, no sysdeps) ====="
env -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" - "$VENV" <<'PYCHK' || Fail "import check"
import os, sys
venv = os.path.realpath(sys.argv[1])
import pydrake
from pydrake.planning import CollisionChecker
from pydrake.solvers import IpoptSolver, NloptSolver, SnoptSolver
p = os.path.realpath(pydrake.__file__)
assert p.startswith(venv), f"pydrake from {p}, not {venv}"
assert hasattr(CollisionChecker, "CalcRobotClearances"), "no CalcRobotClearances"
import importlib.metadata as md
print("pydrake", md.version("drake"), "from", os.path.dirname(p))
## benchmark.py's provenance stamp: a wheel keeps VERSION.TXT at pydrake/doc/drake.
print("VERSION.TXT:", open(os.path.join(os.path.dirname(p), "doc", "drake", "VERSION.TXT")).read().strip())
for S in (SnoptSolver, IpoptSolver, NloptSolver):
    assert S().available(), S.__name__
print("SNOPT / IPOPT / NLopt available")
import torch, jax, soromox, optimistix, ikflow, jrl, FrEIA, klampt, numpy
assert "sm_70" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
assert jax.default_backend() == "cpu", jax.default_backend()
assert "repo/third_party/ikflow" in os.path.realpath(ikflow.__file__), ikflow.__file__
print("torch", torch.__version__, "| jax", jax.__version__, "| numpy", numpy.__version__,
      "| ikflow", os.path.dirname(os.path.realpath(ikflow.__file__)))
PYCHK

## --------------------------------------------- 6. the Drake model cache (home-crc) --
echo "===== [6/6] warming drake_models into $ROOT/home-crc with the wheel's Drake ====="
mkdir -p "$ROOT/home-crc/.cache"
## jrl/ikflow are imported by src/; give them the shared, read-only ikflow cache and a private jrl one.
ln -sfn "$ROOT/home/.cache/ikflow" "$ROOT/home-crc/.cache/ikflow"
env -u PYTHONPATH -u LD_LIBRARY_PATH HOME="$ROOT/home-crc" CUDA_VISIBLE_DEVICES="" \
    LEARNED_IK_REPO="$SRC_TREE" "$VENV/bin/python" - <<'PYWARM' || Fail "drake model warm-up"
import os, sys
sys.path.insert(0, os.environ["LEARNED_IK_REPO"])
from src.utils import BuildEnv
from src.target_screening import SceneFile
## The svgd contexts' scenes (Panda and iiwa, both tasks, hardened). The iiwa's tables are
## package://drake_models; the Panda scenes use none, but are built anyway as an offline check.
for robot in ("panda", "iiwa"):
    for task in ("mug", "pose"):
        path = SceneFile(robot, task, "hardened")
        BuildEnv(meshcat=None, directives_file=path)
        print("scene builds:", robot, task, os.path.relpath(path, os.environ["LEARNED_IK_REPO"]))
PYWARM
ls "$ROOT/home-crc/.cache/drake/package_map" 2>/dev/null
du -sh "$VENV" "$ROOT/home-crc/.cache/drake" 2>/dev/null

echo "OK" > "$DONE"
echo "===== venv-crc built at $(date -Is); sentinel $DONE; freezes in $WORK ====="
