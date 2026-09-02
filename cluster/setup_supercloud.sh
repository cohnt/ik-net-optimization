#!/bin/bash
# Build this project's ISOLATED environment on SuperCloud. Idempotent.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit from ~/learned-ik/repo on the login node:
#     LLsub ./cluster/setup_supercloud.sh -s 8 -q download
#
# It runs as a JOB on the `download` partition rather than on a login node,
# because it pulls ~7 GB (a Drake tarball plus a CUDA torch wheel set) and
# standing rule 6 keeps work of that size off the login and debug nodes.
# `download` is the only non-login partition with internet, and it allows one
# job at a time (MaxJobs=1), so do not queue two of these.
#
# Everything lands under $ROOT and nothing outside it is touched. In
# particular ~/ik-tune is a DIFFERENT project's tree -- its own venv, its own
# Drake, its own sysdeps prefix, and it has jobs running. It is not reused,
# not modified, not symlinked into. `rm -rf ~/learned-ik` undoes all of this.
#
# NOTE: source /etc/profile BEFORE `set -u` (Z97-byobu.sh reads an unset LC_BYOBU).
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
LOG="$ROOT/setup.log"
DONE="$ROOT/setup.DONE"
mkdir -p "$ROOT"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
echo "===== learned-ik setup on $(hostname) at $(date -Is) ====="
cd "$ROOT"

DRAKE_URL="https://github.com/RobotLocomotion/drake/releases/download/v1.56.0/drake-1.56.0-noble.tar.gz"

Fail() { echo "SETUP FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

## ---------------------------------------------------------------- 1. Drake --
echo "===== [1/6] Drake tarball ====="
if [ -d "$ROOT/drake" ] && [ -f "$ROOT/.drake-ok" ]; then
    echo "drake/ already present and verified -- skipping"
else
    wget -q -O drake.tar.gz "$DRAKE_URL" || Fail "drake download"
    wget -q -O drake.tar.gz.sha256 "$DRAKE_URL.sha256" || Fail "drake sha download"
    EXPECTED=$(cut -d' ' -f1 drake.tar.gz.sha256)
    ACTUAL=$(sha256sum drake.tar.gz | cut -d' ' -f1)
    [ "$EXPECTED" = "$ACTUAL" ] || Fail "drake checksum: $EXPECTED vs $ACTUAL"
    tar xzf drake.tar.gz && rm -f drake.tar.gz drake.tar.gz.sha256
    touch "$ROOT/.drake-ok"
    echo "drake extracted to $ROOT/drake"
fi

## ------------------------------------------------------------- 2. sysdeps --
# gridos ships neither libfmt.so.9 nor libspdlog.so.1.12, which the Drake noble
# tarball links, and there is no sudo. Extract the official Ubuntu noble .debs
# (sha512-verified) into a user prefix. RUNTIME libs only: unlike the sibling
# project this repo builds no C++, so the -dev packages and Eigen are not needed.
echo "===== [2/6] vendored runtime libs ====="
MIRROR="http://archive.ubuntu.com/ubuntu/pool/universe"
DEBS="\
fmt9|$MIRROR/f/fmtlib/libfmt9_9.1.0%2bds1-2_amd64.deb|9f55c9dc109cd9ef0dc2191fc677a7e992e2d63bfb7f117458c872bcdb30b469ee64da9ac8d3f0aca0b4bc811c9428bee6f337d30e0681f81c20087d06347989
spd12|$MIRROR/s/spdlog/libspdlog1.12_1.12.0%2bds-2build1_amd64.deb|dddc57863f3d64b4807349866f3f575e809e8c58f5f02f07f322f06840e828e057def7cc21e1b451738048580bfb59fac2946129d0f994abc10a8783bad8e80b"
if [ -f "$ROOT/sysdeps/.deb-ok" ]; then
    echo "sysdeps/ already extracted -- skipping"
else
    rm -rf "$ROOT/sysdeps" && mkdir -p "$ROOT/sysdeps/.work" && cd "$ROOT/sysdeps/.work"
    echo "$DEBS" | while IFS='|' read -r name url sha; do
        [ -z "$name" ] && continue
        wget -q -O "$name.deb" "$url" || { echo "DEB DOWNLOAD FAILED: $url"; exit 1; }
        echo "$sha  $name.deb" | sha512sum -c - >/dev/null \
            || { echo "DEB CHECKSUM MISMATCH: $name"; exit 1; }
        dpkg-deb -x "$name.deb" "$ROOT/sysdeps"
    done || Fail "sysdeps"
    cd "$ROOT" && rm -rf "$ROOT/sysdeps/.work" && touch "$ROOT/sysdeps/.deb-ok"
    echo "sysdeps prefix ready: $(du -sh "$ROOT/sysdeps" | cut -f1)"
fi
cd "$ROOT"

## ----------------------------------------------------------------- 3. venv --
# System python3 (3.12.3) is the only cp312 interpreter here, but ships without
# ensurepip (plain venv fails) and is PEP-668 externally-managed (--user refused).
echo "===== [3/6] venv ====="
if [ -x "$ROOT/venv/bin/pip" ]; then
    echo "venv already present: $("$ROOT/venv/bin/python" --version)"
else
    python3 -m venv --without-pip "$ROOT/venv" || Fail "venv creation"
    wget -q -O "$ROOT/get-pip.py" https://bootstrap.pypa.io/get-pip.py || Fail "get-pip download"
    "$ROOT/venv/bin/python" "$ROOT/get-pip.py" --quiet || Fail "pip bootstrap"
    rm -f "$ROOT/get-pip.py"
fi
PY="$ROOT/venv/bin/python"
PIP="$ROOT/venv/bin/pip"

## ---------------------------------------------------------------- 4. torch --
# THE constraint on this machine: PyTorch 2.11's cu128/cu129 wheels dropped
# Volta (sm_70) support to take CuDNN 9.15.1, and these nodes are V100s. The
# laptop's torch is 2.11.0+cu128 and would import fine here and then fail at the
# first kernel launch with "no kernel image is available for execution on the
# device". cu126 still carries sm_70; try the newest and walk down if not.
echo "===== [4/6] torch (cu126 -- NOT cu128, these are V100s) ====="
HasSm70() {
    "$PY" - <<'PYCHK' 2>/dev/null
import sys, torch
sys.exit(0 if "sm_70" in torch.cuda.get_arch_list() else 1)
PYCHK
}
if "$PY" -c "import torch" 2>/dev/null && HasSm70; then
    echo "torch already installed with sm_70: $("$PY" -c 'import torch;print(torch.__version__)')"
else
    IDX="https://download.pytorch.org/whl/cu126"
    for SPEC in torch torch==2.8.0 torch==2.7.1 torch==2.6.0; do
        echo "--- trying $SPEC from cu126"
        "$PIP" install --quiet --index-url "$IDX" "$SPEC" || continue
        if HasSm70; then echo "installed $SPEC with sm_70"; break; fi
        echo "    $SPEC has no sm_70: $("$PY" -c 'import torch;print(torch.cuda.get_arch_list())')"
    done
    HasSm70 || Fail "no cu126 torch build with sm_70 -- a V100 cannot run any of them"
fi

## ------------------------------------------------------- 5. the rest of it --
# Versions pinned to the laptop's where it matters. --no-deps on jrl and ikflow
# is mandatory: their metadata pins torch==2.0.1, which would silently replace
# the cu126 build installed above. Install from GitHub, not PyPI -- the PyPI
# ikflow (0.0.8) predates the jkinpylib -> jrl rename this repo depends on.
echo "===== [5/6] python deps ====="
# Pinned to this workstation's versions. The unpinned resolve happened to land on
# exactly these, but that is luck, not reproducibility -- and numpy in particular has
# broken this project's sibling before (2.5 removed 2-D np.cross). torch is the one
# deliberate exception: the laptop runs 2.11.0+cu128, which cannot execute on a V100
# at all, so the cluster necessarily runs a cu126 build (measured: 2.14.0+cu126).
"$PIP" install --quiet numpy==2.5.2 klampt==0.10.1.post1 roma==1.6.1 \
    more-itertools==11.1.0 FrEIA==0.2 tqdm==4.70.0 pyyaml==6.0.3 \
    matplotlib==3.11.1 meshcat==0.3.2 PyOpenGL==3.1.10 || Fail "pip deps"
# nvidia-ml-py is NOT in the laptop's venv and is needed here. jrl.config._get_device()
# picks the "least used" GPU by polling nvml through torch.cuda.memory_usage(), and only
# short-circuits when a single device is visible -- which is always true on the laptop's
# one GPU and false on a node with two, where the import then dies with "nvidia-ml-py
# does not seem to be installed". Every job script also pins CUDA_VISIBLE_DEVICES to one
# device, which avoids the poll entirely; this is the belt to that pair of braces, so an
# interactive session or a stray script cannot fail on an import.
"$PIP" install --quiet nvidia-ml-py || Fail "nvidia-ml-py"
"$PIP" install --quiet --no-deps "jrl @ git+https://github.com/jstmn/Jrl.git" \
    || Fail "jrl"
# --ignore-requires-python: ikflow 0.2.0 declares Requires-Python "<3.12,>=3.10", and the
# only cp312 interpreter here is 3.12.3, so pip refuses it. The pin is stale, not a real
# incompatibility: this laptop runs that exact version on 3.12.3 and imports it fine --
# uv (which built the local venv) does not enforce Requires-Python, which is why the
# constraint never surfaced before. Overriding it reproduces the local environment rather
# than departing from it. Drop the flag if ikflow ever republishes with a corrected pin.
"$PIP" install --quiet --no-deps --ignore-requires-python \
    "ikflow @ git+https://github.com/jstmn/ikflow.git" || Fail "ikflow"

# Warm the model caches. Compute nodes have no internet, and ikflow computes its
# MODELS_DIR from expanduser("~") at import, so HOME must point at the tree the
# jobs will symlink from -- not at the real $HOME, which is shared with other
# projects.
echo "--- warming the ikflow weight cache into $ROOT/home"
mkdir -p "$ROOT/home/.cache"
export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOME="$ROOT/home" CUDA_VISIBLE_DEVICES="" "$PY" - <<'PYWARM' || Fail "weight warm-up"
from ikflow.model_loading import get_ik_solver
get_ik_solver("panda__full__lp191_5.25m")
print("panda weights cached")
import jrl.robots
for name in ("panda", "iiwa14"):
    r = jrl.robots.get_robot(name)
    print("jrl urdf cache warmed:", name)
PYWARM
du -sh "$ROOT/home/.cache" 2>/dev/null

## --------------------------------------------------------- 6. verification --
echo "===== [6/6] verification ====="
CUDA_VISIBLE_DEVICES="" "$PY" - <<'PYVERIFY' || Fail "verification"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
arch = torch.cuda.get_arch_list()
print("arch list", arch)
assert "sm_70" in arch, "this torch cannot run on a V100"
import pydrake
from pydrake.solvers import SnoptSolver, IpoptSolver
print("pydrake OK from", pydrake.__file__)
assert IpoptSolver().available(), "IPOPT unavailable"
print("SNOPT", SnoptSolver().available(), "| IPOPT", IpoptSolver().available())
import ikflow, jrl, FrEIA, klampt, numpy
print("ikflow/jrl/FrEIA/klampt import OK; numpy", numpy.__version__)
PYVERIFY

# The iiwa checkpoint is gitignored and 204 MB, so it is rsynced by
# cluster/stage_code.sh rather than downloaded here. Report whether it arrived.
IIWA="$ROOT/repo/models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl"
if [ -f "$IIWA" ]; then
    echo "iiwa checkpoint present: $(du -h "$IIWA" | cut -f1)"
else
    echo "WARNING: iiwa checkpoint missing at $IIWA -- run cluster/stage_code.sh"
fi

echo "OK" > "$DONE"
echo "===== setup complete at $(date -Is); sentinel $DONE ====="
