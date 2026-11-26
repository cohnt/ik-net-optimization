#!/bin/bash
# JOB PAYLOAD (tig-cpu, not preemptible): build the learned-ik Apptainer image from cluster/csail/learned-ik.def.
# Submitted by cluster/csail/build_image.sh, which passes (in front of sbatch, never --export):
#   LIK_ROOT=<$CS_ROOT/roots/c12>  LIK_CS_ROOT=<$CS_ROOT>  EXPECT_MANIFEST_SHA=<laptop's manifest hash>
#   optional: FAKEROOT=auto|yes|no (default auto), BUILD_NAME
#
# Outputs under $LIK_CS_ROOT/images/: <name>.sif, <name>.sif.sha256 (checked by EVERY job), <name>.freeze.txt,
# <name>.DONE ("OK <sha256>" or "FAIL ..."), <name>.build.log. Nothing outside $LIK_CS_ROOT is written; caches go
# to $LIK_CS_ROOT/cache (always: TIG tig-cpu /tmp is tiny; df is logged).
# when it has room).
#
# THE DRAKE PIN: release 1.58.0's noble tarball (Thomas, 2026-11-25). URL, sha256 and commit are the .def's
# %arguments (the one place); the sha256 must also equal the release's published .sha256. NOT the PyPI wheel,
# NOT a nightly. A copy of the tarball may instead be placed at $LIK_CS_ROOT/cache/drake-<version>-noble.tar.gz;
# it is used only if its sha256 is the pin's.
set -uo pipefail
ROOT="${LIK_ROOT:?LIK_ROOT unset}"; CSR="${LIK_CS_ROOT:?LIK_CS_ROOT unset}"; REPO="$ROOT/repo"
. "$REPO/cluster/verify_staged_tree.sh" || exit 1
lik_job_check "$REPO"                       # code integrity: the tree it builds from == the laptop's manifest

echo "===== learned-ik image build on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none} ====="
echo "commit $(cat "$REPO/.staged-commit" 2>/dev/null)"
apptainer --version || { echo "no apptainer on $(hostname)"; exit 21; }
echo "uid $(id -u) user $(id -un); subuid line: $(grep "^$(id -un):" /etc/subuid 2>/dev/null || echo none)"
echo "unprivileged userns: $(unshare -U true 2>&1 && echo yes || echo no)"
df -h /tmp "$CSR" 2>/dev/null

DEF="$REPO/cluster/csail/learned-ik.def"
pin() { sed -n "s/^    $1=\(.*\)\$/\1/p" "$DEF" | head -1; }
DRAKE_VERSION=$(pin DRAKE_VERSION); DRAKE_URL=$(pin DRAKE_URL); DRAKE_SHA256=$(pin DRAKE_SHA256); DRAKE_COMMIT=$(pin DRAKE_COMMIT)
if [ -z "$DRAKE_VERSION" ] || [ -z "$DRAKE_URL" ] || [ ${#DRAKE_SHA256} -ne 64 ] || [ ${#DRAKE_COMMIT} -ne 40 ]; then
    echo "could not read the Drake pin from $DEF"; exit 22
fi
echo "pin: drake $DRAKE_VERSION $DRAKE_URL sha256 $DRAKE_SHA256 commit $DRAKE_COMMIT"

NAME="${BUILD_NAME:-lik-noble-$(cut -c1-12 "$REPO/.staged-commit")-${SLURM_JOB_ID:-0}}"
IMG="$CSR/images"; CACHE="$CSR/cache"; mkdir -p "$IMG" "$CACHE"
LOG="$IMG/$NAME.build.log"; exec > >(tee -a "$LOG") 2>&1
Fail() { echo "BUILD FAILED: $1"; echo "FAIL $1" > "$IMG/$NAME.DONE"; exit "${2:-1}"; }

TAR="$CACHE/drake-${DRAKE_VERSION}-noble.tar.gz"
have_pin() { [ -f "$TAR" ] && [ "$(sha256sum "$TAR" | cut -d' ' -f1)" = "$DRAKE_SHA256" ]; }
if ! have_pin; then
    ## The release's PUBLISHED checksum must agree with the one recorded in the .def, before anything is trusted.
    PUB=$(curl -fsSL --retry 2 "$DRAKE_URL.sha256" | cut -d' ' -f1) || Fail "cannot fetch $DRAKE_URL.sha256" 20
    [ "$PUB" = "$DRAKE_SHA256" ] || Fail "the published sha256 ($PUB) is not the one in learned-ik.def ($DRAKE_SHA256)" 20
    rm -f "$TAR.part"
    echo "fetching $DRAKE_URL"
    curl -fsSL --retry 2 -o "$TAR.part" "$DRAKE_URL" || { rm -f "$TAR.part"; Fail "download of $DRAKE_URL failed" 20; }
    mv -f "$TAR.part" "$TAR"
fi
have_pin || Fail "tarball sha256 is not the pin's ($DRAKE_SHA256)" 20
echo "drake tarball verified: $TAR"

if [ -z "${APPTAINER_TMPDIR:-}" ]; then
    ## Always under /data (Thomas, 2026-11-25): tig-cpu's /tmp is tiny (2.1 GB on groenig-4).
    export APPTAINER_TMPDIR="$CACHE/apptainer-tmp-${SLURM_JOB_ID:-$$}"
fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$CACHE/apptainer}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"; lik_cleanup' EXIT
echo "APPTAINER_TMPDIR=$APPTAINER_TMPDIR APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR"

FR=()
case "${FAKEROOT:-auto}" in
    yes) FR=(--fakeroot) ;;
    no)  ;;
    auto) grep -q "^$(id -un):" /etc/subuid 2>/dev/null && FR=(--fakeroot) ;;
    *) Fail "FAKEROOT must be auto, yes or no" ;;
esac
echo "build mode: ${FR[*]:-unprivileged (no --fakeroot)}"

## Build to a temp name, publish by rename: a stale or half-written image is never the named one.
OUT="$IMG/$NAME.sif"; rm -f "$OUT.tmp"
SECONDS=0
apptainer build "${FR[@]}" \
    --build-arg "DRAKE_TARBALL=$TAR" \
    --build-arg "IKFLOW_SRC=$REPO/third_party/ikflow" \
    --build-arg "CONSTRAINTS=$REPO/cluster/csail/constraints.txt" \
    "$OUT.tmp" "$DEF" || Fail "apptainer build failed (mode: ${FR[*]:-unprivileged})" 30
mv -f "$OUT.tmp" "$OUT"
echo "built in $SECONDS s: $(du -h "$OUT" | cut -f1)"
SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "$SHA  $NAME.sif" > "$OUT.sha256"
apptainer exec "$OUT" cat /opt/lik/freeze.txt > "$IMG/$NAME.freeze.txt" || Fail "cannot read the image's freeze"
GOT=$(apptainer exec "$OUT" cat /opt/lik/drake/share/doc/drake/VERSION.TXT)
[ "$GOT" = "$DRAKE_VERSION $DRAKE_COMMIT" ] || Fail "the built image's Drake is '$GOT', not the pin"
echo "OK $SHA" > "$IMG/$NAME.DONE"
echo "===== image $NAME ready; sha256 $SHA; drake $GOT ====="
