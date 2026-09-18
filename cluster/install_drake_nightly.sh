#!/bin/bash
# Install a SECOND Drake -- a nightly -- ALONGSIDE this project's pinned 1.56.0. Idempotent.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit from ~/learned-ik/repo on the login node:
#     LLsub ./cluster/install_drake_nightly.sh -s 4 -q download
#
# WHY A SECOND INSTALL AND NOT A BUMP.
#
# Every archived IPOPT and SNOPT column in this project was produced against
# $ROOT/drake, the pinned v1.56.0 release. Replacing it would silently put the
# whole campaign on a different solver build and make every archived comparison
# a cross-version one. So the pinned install is left exactly as it is and the
# nightly goes beside it at $ROOT/drake-nightly, selected PER MANIFEST ITEM by
# the `DRAKE=nightly` sentinel that cluster/run_items.sh translates into this
# install's PYTHONPATH. A nightly item and a pinned item can then run in the
# same job, which is what stage DRAKEBUMP needs in order to compare them.
#
# Both installs live under $ROOT, i.e. inside THIS project's tree. Cluster Drake
# versions are per-project: ~/ik-tune has its own Drake and its own jobs, and
# nothing here touches it, nor introduces a shared ~/drake that two projects
# would both resolve. `rm -rf $ROOT/drake-nightly` undoes all of this.
#
# WHY A NIGHTLY RATHER THAN A RELEASE.
#
# The augmented-Lagrangian column needs the NLopt local-optimizer options and
# those from Drake PR 25002 (ftol_rel, ftol_abs, stopval and the two inner ftol
# options), merged 2026-09-17 as 5a73436cd6941519684786d409c67ce25ce16305. Drake
# 1.56.0 declares six NLopt option names; v1.57.0 was cut before BOTH sets and
# still declares six; this nightly declares sixteen. So there is no release that
# carries them yet. 1.58.0 (~mid-October) should, and when it does this whole
# script should be replaced by a pin bump rather than kept -- a nightly is not a
# thing to depend on for long, see the expiry note below.
#
# NOTE: source /etc/profile BEFORE `set -u` (Z97-byobu.sh reads an unset LC_BYOBU).
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
LOG="$ROOT/drake_nightly.log"
DONE="$ROOT/drake_nightly.DONE"
mkdir -p "$ROOT"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
echo "===== learned-ik Drake nightly install on $(hostname) at $(date -Is) ====="
cd "$ROOT"

# The exact artifact this project is pinned to. Nightlies publish NO .sha256 file (unlike
# releases -- `<url>.sha256` returns Drake's "Resource not Available" page), so this hash is
# OURS: it is the sha256 of the tarball that was downloaded, smoke-tested and used to develop
# and test the NLopt plumbing on the workstation on 2026-09-18. Verifying against it is
# strictly stronger than trusting the URL, because it pins the exact bytes the local tests ran
# against rather than "whatever that URL serves today".
NIGHTLY_STAMP="0.0.20260918"
NIGHTLY_SHA256="a6ce34cdaeb3dd9b9d0e8c9d37eb35407d967541482455fc2a9ad31b98c746a8"
NIGHTLY_URL="https://drake-packages.csail.mit.edu/drake/nightly/drake-${NIGHTLY_STAMP}-noble.tar.gz"
# The commit this nightly's VERSION.TXT should carry: PR 25002's merge commit.
NIGHTLY_COMMIT="5a73436cd6941519684786d409c67ce25ce16305"
DEST="$ROOT/drake-nightly"
OK="$ROOT/.drake-nightly-ok"

Fail() { echo "DRAKE NIGHTLY INSTALL FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

## ------------------------------------------------------- 1. the tarball --
echo "===== [1/3] nightly tarball $NIGHTLY_STAMP ====="
if [ -d "$DEST" ] && [ -f "$OK" ]; then
    echo "drake-nightly/ already present and verified -- skipping"
else
    rm -rf "$DEST" drake-nightly.tar.gz
    if ! wget -q -O drake-nightly.tar.gz "$NIGHTLY_URL"; then
        Fail "download. Nightly artifacts EXPIRE 45 days after they are built, and
   $NIGHTLY_STAMP may simply be gone. It is unrecoverable if so: pick a newer nightly,
   re-verify the NLopt option surface against it locally, update NIGHTLY_STAMP,
   NIGHTLY_SHA256 and NIGHTLY_COMMIT here -- or, better, move to the first RELEASE that
   carries Drake PR 25002 (1.58.0 or later) and delete this script."
    fi
    # A 403 or an expiry page is served as an HTML body with a 200-ish shape, so the size
    # check catches "downloaded the error page" before the hash does and says so plainly.
    SIZE=$(stat -c %s drake-nightly.tar.gz)
    [ "$SIZE" -gt 10000000 ] || Fail "download is only $SIZE bytes -- almost certainly Drake's
   'Resource not Available' HTML page rather than a tarball, i.e. the nightly has expired"
    ACTUAL=$(sha256sum drake-nightly.tar.gz | cut -d' ' -f1)
    [ "$NIGHTLY_SHA256" = "$ACTUAL" ] || Fail "sha256 mismatch:
   expected $NIGHTLY_SHA256
   got      $ACTUAL
   The URL served different bytes than the ones this project's tests were written against.
   Do NOT relax this: re-run the local plumbing tests against whatever it is now serving
   first (tests/test_solver_plumbing.py, with PYTHONPATH pointed at it)."
    mkdir -p "$DEST"
    # --strip-components: the tarball's root is `drake/`, and nesting it as
    # drake-nightly/drake/ would put site-packages one level off every PYTHONPATH that
    # references it, including run_items.sh's.
    tar xzf drake-nightly.tar.gz -C "$DEST" --strip-components=1 || Fail "extract"
    rm -f drake-nightly.tar.gz
    echo "extracted to $DEST"
fi

STAMPED=$(cat "$DEST/share/doc/drake/VERSION.TXT" 2>/dev/null || echo "")
echo "VERSION.TXT: $STAMPED"
case "$STAMPED" in
    *"$NIGHTLY_COMMIT"*) echo "carries PR 25002's merge commit" ;;
    *) Fail "this install does not carry $NIGHTLY_COMMIT (PR 25002), so it does not have the
   NLopt options the augmented-Lagrangian stage needs: VERSION.TXT says '$STAMPED'" ;;
esac

## ------------------------------------------- 2. the model cache, warmed --
# Drake fetches its `drake_models` remote package LAZILY, at scene-load time, into
# ~/.cache/drake/package_map, and compute nodes have no internet. The cache key includes the
# models commit THAT DRAKE VERSION PINS, so the pinned install's warmed cache cannot be
# borrowed: this install needs its own, warmed here on the one partition with internet.
# Warmed into the same $ROOT/home that run_items.sh symlinks as $HOME, so a nightly item
# finds it exactly where a pinned item finds its own.
echo "===== [2/3] warming the Drake model cache with the NIGHTLY ====="
PY="${TEST_PYTHON:-$ROOT/venv/bin/python}"
[ -x "$PY" ] || Fail "no venv at $PY -- run cluster/setup_supercloud.sh first"
HOME="$ROOT/home" CUDA_VISIBLE_DEVICES="" LEARNED_IK_REPO="$ROOT/repo" \
PYTHONPATH="$DEST/lib/python3.12/site-packages" \
    "$PY" - <<'PYDRAKE' || Fail "drake model warm-up under the nightly"
import os, sys
sys.path.insert(0, os.environ["LEARNED_IK_REPO"])
import pydrake
print("pydrake from", pydrake.__file__)
from src.utils import BuildEnv, RepoDir
# The hardened scenes are the only ones the NLopt stage runs, but warm the legacy pair too:
# they cost seconds here and a missing package on a compute node costs a whole item.
for scene in ("models/panda/panda_finray_collision.yaml",
              "models/iiwa14/iiwa14_collision.yaml",
              "models/panda/panda_finray_collision_hardened.yaml",
              "models/iiwa14/iiwa14_collision_nobin.yaml",
              "models/iiwa14/iiwa14_collision_hardened.yaml"):
    BuildEnv(meshcat=None, directives_file=os.path.join(RepoDir(), scene))
    print("scene builds offline under the nightly:", scene)
PYDRAKE

## ------------------------------------------------------ 3. verification --
# A wrong or half-extracted Drake IMPORTS CLEANLY and fails only at first real use, which is
# why this launches actual work rather than checking that the import succeeded. The NLopt
# surface is asserted by NAME: sixteen accessors is what makes this install worth having, and
# a nightly that regressed them should fail here rather than on the cluster's first cell.
echo "===== [3/3] verification ====="
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$DEST/lib/python3.12/site-packages" \
    "$PY" - <<'PYVERIFY' || Fail "verification"
import pydrake
from pydrake.solvers import IpoptSolver, SnoptSolver, NloptSolver
print("pydrake OK from", pydrake.__file__)
for S in (IpoptSolver, SnoptSolver, NloptSolver):
    assert S().available(), f"{S.__name__} is not available in this build"
    print(f"{S.__name__} available")
need = ["FRelativeToleranceName", "FAbsoluteToleranceName", "StopValName",
        "LocalOptimizerAlgorithmName", "LocalOptimizerXRelativeToleranceName",
        "LocalOptimizerXAbsoluteToleranceName", "LocalOptimizerFRelativeToleranceName",
        "LocalOptimizerFAbsoluteToleranceName", "LocalOptimizerMaxEvalName",
        "LocalOptimizerMaxTimeName"]
missing = [n for n in need if not hasattr(NloptSolver, n)]
assert not missing, f"this Drake lacks {missing} -- it is not new enough for stage NLOPTTUNE"
print("all ten post-1.56.0 NLopt options present:",
      sorted(getattr(NloptSolver, n)() for n in need))
PYVERIFY

# And the repo's own plumbing guards, against THIS Drake. They are what prove the ten fields
# emit here and are refused where absent, and they are cheap.
echo "--- tests/test_solver_plumbing.py against the nightly"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$DEST/lib/python3.12/site-packages" \
    "$PY" "$ROOT/repo/tests/test_solver_plumbing.py" | tail -5 \
    || Fail "tests/test_solver_plumbing.py against the nightly"

touch "$OK"
echo "OK $NIGHTLY_STAMP $(date -Is)" > "$DONE"
echo "===== drake nightly install complete ====="
echo "pinned : $ROOT/drake            (v1.56.0, unchanged, still the default)"
echo "nightly: $DEST  ($NIGHTLY_STAMP, selected by DRAKE=nightly in a manifest)"
