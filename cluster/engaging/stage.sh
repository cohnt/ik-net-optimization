#!/bin/bash
# Stage the CURRENT COMMIT onto Engaging, with the code-integrity checks (cluster/README.md, "Code integrity").
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about an Engaging action, STOP and ask Thomas.
# ===========================================================================
#
#   bash cluster/engaging/stage.sh
#
# Mirrors cluster/stage_code.sh, with one structural difference: the tree is IMMUTABLE and keyed by the
# commit, ~/lik-engaging/c-<12 hex>/repo, beside that commit's state/, results/ and home. A requeued
# (preempted) job therefore re-reads exactly the code it started with, and restaging a new commit never
# disturbs a running campaign, so there is no live-campaign guard to forget. Restaging the SAME commit
# is idempotent (same bytes).
#
# What is checked, in order: the working tree is clean (no override); submodules are initialised
# (rsync reads the working tree, and an empty third_party/ikflow would ship an empty fork); no symlink
# under models/; every shipped file is git-tracked or allowlisted payload (lik_build_manifest) and is
# sha256'd into a manifest BEFORE anything is sent; after the transfer (size pass + a --checksum content
# pass) the cluster's own sha256sum is run against the laptop's manifest, inside the tree
# (lik_laptop_check). Anything but a match is STOP: read the listed files, never bypass, fix, restage.
# The manifest's hash is what every job is handed (EXPECT_MANIFEST_SHA) and re-checks.
set -uo pipefail
source "$(dirname "$0")/eg_common.sh" || exit 1
eg_require_connection || exit 255
REPO_ROOT="$EG_REPO"

COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
DIRTY=$(git -C "$REPO_ROOT" status --porcelain)
if [ -n "$DIRTY" ]; then
    echo "REFUSING: the working tree is not clean; commit (or discard) first, then stage that commit." >&2
    printf '%s\n' "$DIRTY" | head -20 >&2
    exit 6
fi
if [ -f "$REPO_ROOT/.gitmodules" ]; then
    UNINIT=$(git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | grep -c "^-" || true)
    if [ "${UNINIT:-0}" -gt 0 ]; then
        echo "REFUSING: $UNINIT uninitialised submodule path(s) locally (git submodule update --init --recursive)." >&2
        exit 4
    fi
fi
LINKS=$(find "$REPO_ROOT/models" -type l 2>/dev/null | wc -l | tr -d " ")
if [ "${LINKS:-0}" -gt 0 ]; then
    echo "REFUSING: $LINKS symlink(s) under models/ (rsync would ship links, not checkpoints)." >&2
    exit 5
fi

CROOT=$(eg_commit_root "$COMMIT"); SUB="c-${COMMIT:0:12}/repo"
EXCLUDE_ARGS=(); for _e in "${LIK_STAGE_EXCLUDES[@]}"; do [ -n "$_e" ] && EXCLUDE_ARGS+=("--exclude=$_e"); done
FILTER_ARGS=();  for _p in "${LIK_STAGE_PROTECT[@]}";  do FILTER_ARGS+=("--filter=P $_p"); done

SDIR="$REPO_ROOT/logs/staged"; NEWDIR="$SDIR/.new-eg"
rm -rf "$NEWDIR"; mkdir -p "$NEWDIR"
lik_build_manifest "$REPO_ROOT" "$NEWDIR/m.sha256" "${EXCLUDE_ARGS[@]}" || exit 7
echo "$COMMIT" > "$NEWDIR/m.commit"
echo "commit $COMMIT: manifest of $(wc -l < "$NEWDIR/m.sha256") files"

eg_run "mkdir -p ~/$CROOT/repo ~/$CROOT/state ~/$CROOT/results ~/$EG_BASE/image ~/$EG_BASE/home-cache/.cache" || exit 9
eg_rsync -az --delete "${FILTER_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
    "$REPO_ROOT/" "$EG_DEST:$CROOT/repo/" || { echo "ERROR: rsync failed; the cluster tree is a mix: do NOT submit from it" >&2; exit 11; }
eg_rsync -az --checksum "--max-size=$LIK_LARGE_BYTES" "${FILTER_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
    "$REPO_ROOT/" "$EG_DEST:$CROOT/repo/" || { echo "ERROR: the --checksum pass failed; do NOT submit from this tree" >&2; exit 11; }

eg_run "cat > ~/$CROOT/repo/.staged-manifest.sha256" < "$NEWDIR/m.sha256" || exit 8
_MKEY=$(sha256sum < "$NEWDIR/m.sha256" | cut -d' ' -f1)
mkdir -p "$SDIR/by-manifest"
for _x in sha256 commit large; do cp "$NEWDIR/m.$_x" "$SDIR/by-manifest/$_MKEY.$_x"; done
echo "$_MKEY" > "$SDIR/engaging-$COMMIT.manifest-sha"
echo "$COMMIT" > "$SDIR/engaging-last-commit"
rm -rf "$NEWDIR"
eg_run "echo $COMMIT > ~/$CROOT/repo/.staged-commit; echo staged commit: \$(cat ~/$CROOT/repo/.staged-commit); du -sh ~/$CROOT/repo"

## Prove it: the cluster's own sha256sum against the laptop's manifest.
eg_check_commit "$COMMIT" || exit $?
echo "STAGED $COMMIT at ~/$CROOT (manifest sha $_MKEY)"
