#!/bin/bash
# Stage the CURRENT COMMIT of this checkout onto CSAIL's group scratch, with the integrity checks.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a CSAIL action, STOP and ask Thomas. Cluster writes need his clearance.
# ===========================================================================
#
#   CS_DEST=<user>@slurm-login.csail.mit.edu bash cluster/csail/stage.sh
#
# What it does (the SuperCloud flow of cluster/stage_code.sh, with CSAIL's paths and the skill's recipe):
#   1. refuses unless `git status --porcelain` is empty (no override), the submodules are initialised, and models/
#      holds no symlink;
#   2. builds a sha256 MANIFEST of exactly what rsync will ship (every file tracked, or an allowlisted checkpoint)
#      BEFORE anything is sent, keeps it on the laptop (logs/staged/csail-<c12>.sha256, .commit, .large);
#   3. creates $CS_ROOT/roots/<c12>/{repo,state,results,logs,manifests,home/.cache}   <- the ONLY remote write
#      besides the rsync into repo/;
#   4. rsync --delete, then a --checksum content pass (size+mtime can miss a same-size change);
#   5. writes .staged-manifest.sha256 and .staged-commit (full hash) into the tree;
#   6. proves it: the cluster's own `sha256sum -c` against the laptop's manifest (csail_laptop_check).
# Prints the root and EXPECT_MANIFEST_SHA. Group scratch can be modified by others: the SAME check runs again
# inside every job (lik_job_check) and before every submit.
#
# One staged commit = one root, so restaging after a commit never changes code under a running or requeued job.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/csail_common.sh" || exit 1
REPO_ROOT="$(csail_repo_root)"
. "$REPO_ROOT/cluster/verify_staged_tree.sh"
. "$HERE/csail_verify.sh"
csail_connected || exit $?

COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
C12=${COMMIT:0:12}
DIRTY=$(git -C "$REPO_ROOT" status --porcelain)
if [ -n "$DIRTY" ]; then
    echo "REFUSING: the working tree is not clean; commit (or discard) first, then stage that commit." >&2
    printf '%s\n' "$DIRTY" | head -20 >&2
    exit 6
fi
if [ -f "$REPO_ROOT/.gitmodules" ]; then
    UNINIT=$(git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | grep -c "^-" || true)
    if [ "${UNINIT:-0}" -gt 0 ]; then
        echo "REFUSING: $UNINIT uninitialised submodule path(s) locally (the vendored ikflow fork is one)." >&2
        echo "Fix: git -C $REPO_ROOT submodule update --init --recursive" >&2
        exit 4
    fi
fi
LINKS=$(find "$REPO_ROOT/models" -type l 2>/dev/null | wc -l | tr -d " ")
if [ "${LINKS:-0}" -gt 0 ]; then
    echo "REFUSING: $LINKS symlink(s) under models/ (rsync would ship links, not checkpoints)." >&2
    exit 5
fi

EXCLUDES=("${LIK_STAGE_EXCLUDES[@]}")
EXCLUDE_ARGS=(); for _e in "${EXCLUDES[@]}"; do [ -n "$_e" ] && EXCLUDE_ARGS+=("--exclude=$_e"); done
FILTER_ARGS=(); for _p in "${LIK_STAGE_PROTECT[@]}"; do FILTER_ARGS+=("--filter=P $_p"); done

ROOT="$(csail_tree_root "$C12")"
SDIR="$REPO_ROOT/logs/staged"; NEWDIR="$SDIR/.new-csail"
rm -rf "$NEWDIR"; mkdir -p "$NEWDIR"
lik_build_manifest "$REPO_ROOT" "$NEWDIR/csail-$C12.sha256" "${EXCLUDE_ARGS[@]}" || exit 7
echo "$COMMIT" > "$NEWDIR/csail-$C12.commit"
echo "commit $COMMIT; manifest: $(wc -l < "$NEWDIR/csail-$C12.sha256") files"

cs_run "mkdir -p $ROOT/repo $ROOT/state $ROOT/results $ROOT/logs $ROOT/manifests $ROOT/home/.cache" || exit 8

cs_rsync -az --delete "${FILTER_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
    "$REPO_ROOT/" "$CS_DEST:$ROOT/repo/" || { echo "ERROR: rsync failed; the cluster tree is a mix: do NOT submit from it" >&2; exit 11; }
cs_rsync -az --checksum "--max-size=$LIK_LARGE_BYTES" "${FILTER_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
    "$REPO_ROOT/" "$CS_DEST:$ROOT/repo/" || { echo "ERROR: the --checksum rsync pass failed; do NOT submit from this tree" >&2; exit 11; }

cs_run "cat > $ROOT/repo/.staged-manifest.sha256 && echo $COMMIT > $ROOT/repo/.staged-commit" < "$NEWDIR/csail-$C12.sha256" || exit 8
for _x in sha256 commit large; do mv -f "$NEWDIR/csail-$C12.$_x" "$SDIR/csail-$C12.$_x"; done
rmdir "$NEWDIR" 2>/dev/null || true

csail_laptop_check "$C12" || exit $?
echo "ROOT=$ROOT"
echo "C12=$C12"
echo "EXPECT_MANIFEST_SHA=$EXPECT_MANIFEST_SHA"
