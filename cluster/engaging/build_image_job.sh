#!/bin/bash
# sbatch payload: build the Engaging image FROM THE COMMITTED lik.def, on a compute node (never a login node).
#   LIK_CROOT=... EXPECT_MANIFEST_SHA=... [DRAKE_TARBALL=<abs path>] sbatch -p mit_normal ... build_image_job.sh
#
# Output: ~/lik-engaging/image/lik.sif, its sha256 (image/lik.sif.sha256), the commit it was built from
# (image/lik.sif.commit), pip-freeze and the Drake pin it carries; sentinel ~/lik-engaging/build.DONE
# (OK <sha256> / FAIL). The submitter reads the sha256 and anchors it for every later job (EXPECT_SIF_SHA).
# The image is IMMUTABLE once built: a changed .def is a new build, written beside and moved into place
# only on success, so no running job sees the file change.
#
# The Drake pin is the release tarball (1.58.0) verified against DRAKE_SHA256 and its published .sha256 in lik.def. If the URL no longer
# serves it, the build FAILS rather than substituting
# another Drake; DRAKE_TARBALL names a tarball already on the cluster, bind-mounted into the build and
# verified against the same hash.
LIK_NEED_IMAGE=0
# submit.sh passes LIK_CROOT relative to the cluster home.
case "${LIK_CROOT:?LIK_CROOT unset}" in /*) ;; *) LIK_CROOT="$HOME/$LIK_CROOT" ;; esac; export LIK_CROOT
source "$LIK_CROOT/repo/cluster/engaging/job_lib.sh" || exit 2
eg_job_init
DONE="$EB/build.DONE"; rm -f "$DONE"
Fail() { echo "BUILD FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

mkdir -p "$EB/image"
export APPTAINER_TMPDIR="$TMPDIR/atmp" APPTAINER_CACHEDIR="$TMPDIR/acache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
cd "$TMPDIR" || exit 2
BIND=()
if [ -n "${DRAKE_TARBALL:-}" ]; then
    [ -f "$DRAKE_TARBALL" ] || Fail "DRAKE_TARBALL $DRAKE_TARBALL not found"
    BIND=(--bind "$DRAKE_TARBALL:/drake.tar.gz:ro")
fi
echo "===== apptainer build $(date -Is) ====="
apptainer build --fakeroot "${BIND[@]}" lik.sif "$REPO/cluster/engaging/lik.def" || Fail "apptainer build"

echo "===== in-image check ====="
apptainer exec --env PYTHONPATH=/opt/lik/drake/lib/python3.12/site-packages lik.sif /opt/lik/venv/bin/python - <<'PY' || Fail "in-image check"
import torch, pydrake
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
print("pydrake", pydrake.__file__)
print(open("/opt/lik/drake/share/doc/drake/VERSION.TXT").read().strip())
PY
SHA=$(sha256sum lik.sif | cut -d' ' -f1)
apptainer exec lik.sif cat /opt/lik/pip-freeze.txt > "$EB/image/lik.sif.pip-freeze.txt"
cp lik.sif "$EB/image/lik.sif.new" && mv -f "$EB/image/lik.sif.new" "$EB/image/lik.sif" || Fail "install image"
echo "$SHA  lik.sif" > "$EB/image/lik.sif.sha256"
cat "$REPO/.staged-commit" > "$EB/image/lik.sif.commit"
echo "OK $SHA" > "$DONE"
echo "===== built lik.sif sha256 $SHA ($(du -h "$EB/image/lik.sif" | cut -f1)) at $(date -Is) ====="
