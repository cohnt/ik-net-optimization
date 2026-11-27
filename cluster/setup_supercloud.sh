#!/bin/bash
# SUBMIT WITH: bash cluster/submit_job.sh cluster/setup_supercloud.sh [LLsub args] (env vars via JOB_ENV="A=b"); a bare LLsub refuses at job start (README "Code integrity").
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
source "$ROOT/repo/cluster/verify_staged_tree.sh" || exit 1; lik_job_check "$ROOT/repo"   # code integrity: the repo it installs from == laptop manifest (README)

## THE PROJECT'S DRAKE PIN: release 1.58.0 (Thomas, 2026-11-25). It carries PR 25002 (the sixteen
## NLopt options the adopted NLopt configuration needs) and #25073 (CollisionChecker.CalcRobotClearances,
## which AL-SVGD's `svgd_collision_backend=clearances` needs), so this is the ONE Drake: the old
## nightly 0.0.20260918 (expired upstream, unrecoverable) and the DRAKE_ENV=crc wheel are retired.
## The CSAIL and Engaging images pin the same tarball (cluster/csail/learned-ik.def).
## A release publishes a `.sha256`; this checks the download against the hash recorded here AND
## against the published one, so a changed upstream asset is caught as well as a corrupt download.
## share/doc/drake/VERSION.TXT of the release reads "<version> <commit>" (the form the CSAIL def and
## cluster/engaging/plumb_check.py check).
DRAKE_VERSION="1.58.0"
DRAKE_SHA256="91f5e2807b10b1ec117cf2037419d3dc50b8ba8a341f509f2d4696a9e27c63b9"
DRAKE_URL="https://github.com/RobotLocomotion/drake/releases/download/v${DRAKE_VERSION}/drake-${DRAKE_VERSION}-noble.tar.gz"
DRAKE_COMMIT="09bc913cf1af0cdde31073d5f1f7f66ca35ee0d8"
DRAKE_STAMP_TXT="drake ${DRAKE_VERSION} ${DRAKE_COMMIT}"   # what $ROOT/.drake-ok must contain

## DRAKE_ONLY=1 (JOB_ENV="DRAKE_ONLY=1"): redo only the Drake tarball, the sysdeps check and the
## model-cache warm-up. The venv holds NO Drake (Drake is on PYTHONPATH), so a Drake move needs no
## venv rebuild; this skips every pip step so nothing in the venv (jrl is unpinned) can move.
DRAKE_ONLY="${DRAKE_ONLY:-0}"

Fail() { echo "SETUP FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

## ---------------------------------------------------------------- 1. Drake --
echo "===== [1/6] Drake tarball ($DRAKE_VERSION) ====="
if [ -d "$ROOT/drake" ] && [ "$(cat "$ROOT/.drake-ok" 2>/dev/null)" = "$DRAKE_STAMP_TXT" ]; then
    echo "drake/ already present and verified as $DRAKE_VERSION -- skipping"
elif [ -e "$ROOT/drake" ] || [ -e "$ROOT/.drake-ok" ]; then
    ## NEVER delete or overwrite an installed Drake: the previous nightly (0.0.20260918) is
    ## unrecoverable upstream. The operator moves it aside; this script only adds.
    Fail "$ROOT/drake (or .drake-ok) exists but is not '$DRAKE_STAMP_TXT' (.drake-ok reads: '$(cat "$ROOT/.drake-ok" 2>/dev/null)').
   Nothing was touched. Move the old install aside (nothing may be running on it), then resubmit:
     cd $ROOT && mv drake drake-0.0.20260918 && mv .drake-ok .drake-ok-0.0.20260918"
else
    wget -q -O drake.tar.gz "$DRAKE_URL" || Fail "drake download failed: $DRAKE_URL"
    ACTUAL=$(sha256sum drake.tar.gz | cut -d' ' -f1)
    [ "$DRAKE_SHA256" = "$ACTUAL" ] || Fail "drake checksum vs the recorded pin: $DRAKE_SHA256 vs $ACTUAL"
    PUBLISHED=$(wget -q -O - "${DRAKE_URL}.sha256" | cut -d' ' -f1)
    [ "$PUBLISHED" = "$ACTUAL" ] || Fail "drake checksum vs the release's published .sha256: '$PUBLISHED' vs $ACTUAL"
    tar xzf drake.tar.gz && rm -f drake.tar.gz
    GOT=$(cat "$ROOT/drake/share/doc/drake/VERSION.TXT" 2>/dev/null)
    [ "$GOT" = "$DRAKE_VERSION $DRAKE_COMMIT" ] || Fail "drake is not the pin: VERSION.TXT reads '$GOT'"
    echo "$DRAKE_STAMP_TXT" > "$ROOT/.drake-ok"
    echo "drake $DRAKE_VERSION ($DRAKE_COMMIT) extracted to $ROOT/drake"
fi

## ------------------------------------------------------------- 2. sysdeps --
# gridos ships neither libfmt.so.9 nor libspdlog.so.1.12, which the Drake noble
# tarball links, and there is no sudo. Extract the official Ubuntu noble .debs
# (sha512-verified) into a user prefix. RUNTIME libs only: unlike the sibling
# project this repo builds no C++, so the -dev packages and Eigen are not needed.
# The CSAIL/Engaging images install the same two libs for the 1.58.0 noble tarball (plus the
# tarball's own package list, which gridos may lack more of): step 6 imports pydrake, so a
# missing lib fails there rather than silently.
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
if [ "$DRAKE_ONLY" = 1 ]; then
    [ -x "$ROOT/venv/bin/pip" ] || Fail "DRAKE_ONLY=1 but no venv at $ROOT/venv"
    echo "DRAKE_ONLY=1: venv left alone"
elif [ -x "$ROOT/venv/bin/pip" ]; then
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
if [ "$DRAKE_ONLY" = 1 ]; then
    echo "DRAKE_ONLY=1: torch left alone"
elif "$PY" -c "import torch" 2>/dev/null && HasSm70; then
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
if [ "$DRAKE_ONLY" = 1 ]; then echo "DRAKE_ONLY=1: pip steps skipped"; else
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
# The GVS push-rod arm's forward model IS SoRoMoX (JAX), solved to equilibrium with
# optimistix. CPU jaxlib ONLY: the flow owns the GPU, and a CUDA jax beside torch's CUDA
# would fight it for the device and the driver. Pinned to the laptop's versions, where the
# model tests were run; `jax[cpu]` pins jaxlib to the same version.
"$PIP" install --quiet "jax[cpu]==0.11.2" soromox==0.5.0 optimistix==0.1.0 \
    equinox==0.13.8 diffrax==0.7.2 lineax==0.1.1 || Fail "soromox/jax"
"$PY" - <<'PYJAX' || Fail "jax is not CPU-only, or soromox does not import"
import jax, soromox, optimistix
assert jax.default_backend() == "cpu", jax.default_backend()
PYJAX
# Training deps (iiwa retraining campaign). Pinned to the laptop's versions. wandb runs
# WANDB_MODE=offline on compute nodes (no internet, no credentials there); syncing to
# the dashboard happens from the laptop.
"$PIP" install --quiet pytorch-lightning==2.6.5 wandb==0.29.0 || Fail "lightning/wandb"
"$PIP" install --quiet --no-deps "jrl @ git+https://github.com/jstmn/Jrl.git" \
    || Fail "jrl"
# ikflow comes from the VENDORED FORK (third_party/ikflow, cohnt/ikflow), editable so a
# restage of the repo tree updates the training code with no reinstall. The upstream
# wheel is inference-only (no ikflow/training/), and the fork carries the DDP training
# support and the pole-fraction metric. --no-deps remains mandatory (metadata would
# replace the cu126 torch). Requires stage_code.sh to have run first.
[ -d "$ROOT/repo/third_party/ikflow/ikflow" ] || Fail "vendored fork missing -- run cluster/stage_code.sh first"
"$PIP" install --quiet --no-deps -e "$ROOT/repo/third_party/ikflow" || Fail "ikflow (vendored fork)"
"$PY" - <<'PYFORK' || Fail "ikflow does not resolve to the vendored fork"
import ikflow, os, sys
path = os.path.realpath(ikflow.__file__)
sys.exit(0 if "repo/third_party/ikflow" in path else 1)
PYFORK
fi   # DRAKE_ONLY

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
# Drake fetches its `drake_models` remote package (the meshes the scenes reference as
# package://drake_models/...) LAZILY, at scene-load time, into ~/.cache/drake/package_map.
# Compute nodes have no internet, so a job would die inside ProcessModelDirectives with
# "Network is unreachable". Warm it here by actually building every scene. It has to be
# warmed with the CLUSTER's Drake, not copied from the laptop: the cache key includes the
# models commit that Drake version pins, and the two builds differ. It runs on EVERY invocation
# (never skipped because a cache exists) and only ADDS: a new Drake pins a new models commit, so
# the old nightly's cache entry stays in place beside the new one.
echo "--- warming the Drake model cache (drake_models) into $ROOT/home"
# $PWD is $ROOT here (this script cd'd there), not the repo, so pass the repo explicitly.
HOME="$ROOT/home" CUDA_VISIBLE_DEVICES="" LEARNED_IK_REPO="$ROOT/repo" \
    "$PY" - <<'PYDRAKE' || Fail "drake model warm-up"
import os, sys
sys.path.insert(0, os.environ["LEARNED_IK_REPO"])
from src.utils import BuildEnv, RepoDir
for scene in ("models/panda/panda_finray_collision.yaml",
              "models/iiwa14/iiwa14_collision.yaml",
              # The hardened scenes the campaign actually runs. They pull no package the
              # legacy ones do not (dropping the bin only DROPS bin2.sdf), but a compute
              # node has no internet, so warm them rather than reason about it.
              "models/panda/panda_finray_collision_hardened.yaml",
              "models/iiwa14/iiwa14_collision_nobin.yaml",
              "models/iiwa14/iiwa14_collision_hardened.yaml"):
    BuildEnv(meshcat=None, directives_file=os.path.join(RepoDir(), scene))
    print("scene builds offline:", scene)
PYDRAKE
du -sh "$ROOT/home/.cache" "$ROOT/home/.cache/drake" 2>/dev/null

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
import os
vt = open(os.path.join(os.environ["PYTHONPATH"].split(":")[0], "../../../share/doc/drake/VERSION.TXT")).read().split()
assert vt[:2] == ["1.58.0", "09bc913cf1af0cdde31073d5f1f7f66ca35ee0d8"], vt   # the pin
print("Drake VERSION.TXT", vt)
import ikflow, jrl, FrEIA, klampt, numpy
print("ikflow/jrl/FrEIA/klampt import OK; numpy", numpy.__version__)
PYVERIFY

# Plumbing beyond imports (NLopt option names, CalcRobotClearances, tiny solves) is verified on
# compute nodes with no internet: cluster/verify_drake_imports.sh (debug-cpu) and
# cluster/verify_drake_solves.sh (xeon-p8); cluster/README.md, "Moving SuperCloud's Drake".

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
