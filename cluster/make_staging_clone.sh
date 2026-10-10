#!/bin/bash
# Build (or refresh) a throwaway STAGING CLONE of a branch tip, ready for cluster/stage_code.sh.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
#   bash cluster/make_staging_clone.sh [DEST]          # BRANCH=svgd by default
#   bash $DEST/cluster/stage_code.sh                   # then stage FROM THE CLONE
#
# Touches nothing on the cluster and nothing in the checkout it is run from: it only reads that
# checkout's repository. WHY IT EXISTS: a git worktree cannot be staged as it stands. Its
# submodule (third_party/ikflow) is not checked out and its gitignored checkpoints are symlinks
# into the main checkout, and stage_code.sh refuses both (exits 4 and 5) because its rsync runs
# with --delete. So stage from a clean clone pinned at the tip instead:
#   - a local clone of the repository's branch tip, detached at that commit;
#   - the ikflow submodule initialised FROM THE MAIN CHECKOUT'S OWN MODULE STORE, with no network
#     (falls back to the .gitmodules URL, GitHub over ssh, if that store lacks the commit);
#   - each checkpoint in CKPTS made a real file: a hardlink from the main checkout where both are
#     on one filesystem, else a copy (a session scratchpad on tmpfs cannot hold a hardlink to
#     /home). A tracked file (e.g. the .arch.json sidecar) already comes with the clone and is
#     only checked against the main checkout's copy.
# Re-running refreshes an existing clone to the CURRENT tip, so after committing more on the
# branch, re-run this before staging. stage_code.sh stamps .staged-commit from the clone's HEAD.
set -euo pipefail

WT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MAIN="$(dirname "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)")"
BRANCH="${BRANCH:-svgd}"
DEST="${1:-/tmp/claude-1000/-home-tommy-Documents-programming-work-rlg-analytic-and-optimization-ik-learned-ik/27aa4506-d9d8-4485-b48d-8f0ef51ffca8/scratchpad/stage-svgd}"
CKPTS="${CKPTS:-models/panda/panda__n6__step620000.pkl models/panda/panda__n6__step620000.arch.json}"

TIP="$(git -C "$WT" rev-parse "$BRANCH")"
echo "branch $BRANCH at $(git -C "$WT" log -1 --oneline "$TIP")"
## The tip is a COMMIT: uncommitted edits in the worktree are not in it. Say so loudly.
if [ "$(git -C "$WT" rev-parse HEAD)" = "$TIP" ] && \
        [ -n "$(git -C "$WT" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: $WT has uncommitted tracked changes; they are NOT in the clone." >&2
fi
if [ -n "$(git -C "$WT" status --porcelain -- cluster/manifest_stage*.txt cluster/SVGD_RUNBOOK.md 2>/dev/null)" ]; then
    echo "WARNING: manifests / runbook are uncommitted or modified; commit them and re-run." >&2
fi

if [ ! -d "$DEST/.git" ]; then
    git clone --quiet --no-checkout "$MAIN" "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$BRANCH"
git -C "$DEST" checkout --quiet --detach "$TIP"
[ -z "$(git -C "$DEST" status --porcelain --untracked-files=no)" ] || {
    echo "REFUSING: $DEST has local changes; it is a throwaway, delete it and re-run." >&2; exit 2; }

## Submodules from the main checkout's module store (local, no network), else the real URL.
git -C "$DEST" submodule init --quiet
while read -r key path; do
    name="${key#submodule.}"; name="${name%.path}"
    store="$MAIN/.git/modules/$name"
    if [ -d "$store" ]; then
        git -C "$DEST" config "submodule.$name.url" "$store"
    fi
done < <(git -C "$DEST" config -f .gitmodules --get-regexp '^submodule\..*\.path$')
## protocol.file.allow: git refuses file-transport submodule clones by default (CVE-2022-39253).
if ! git -C "$DEST" -c protocol.file.allow=always submodule update --quiet --recursive; then
    echo "local module store failed; falling back to the .gitmodules URLs" >&2
    git -C "$DEST" submodule sync --quiet --recursive
    git -C "$DEST" submodule update --quiet --init --recursive
fi
UNINIT=$(git -C "$DEST" submodule status --recursive | grep -c '^-' || true)
[ "${UNINIT:-0}" -eq 0 ] || { echo "FAILED: $UNINIT submodule(s) still uninitialised" >&2; exit 3; }

## Checkpoints: real files, never symlinks (stage_code.sh refuses a symlink under models/).
for rel in $CKPTS; do
    src="$MAIN/$rel"; dst="$DEST/$rel"
    src_real="$(readlink -f "$src")"
    [ -f "$src_real" ] || { echo "FAILED: no $src in the main checkout" >&2; exit 4; }
    if git -C "$DEST" ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
        cmp -s "$src_real" "$dst" || { echo "FAILED: tracked $rel differs from $src" >&2; exit 4; }
        echo "tracked:   $rel (matches the main checkout)"
        continue
    fi
    mkdir -p "$(dirname "$dst")"
    rm -f "$dst"
    if ln "$src_real" "$dst" 2>/dev/null; then
        echo "hardlink:  $rel"
    else
        cp --preserve=timestamps "$src_real" "$dst"
        echo "copied:    $rel (different filesystem, no hardlink possible)"
    fi
    cmp -s "$src_real" "$dst" || { echo "FAILED: $rel did not arrive intact" >&2; exit 4; }
done
LINKS=$(find "$DEST/models" -type l | wc -l)
[ "$LINKS" -eq 0 ] || { echo "FAILED: $LINKS symlink(s) under $DEST/models" >&2; exit 5; }

echo "ready: $DEST at $(git -C "$DEST" rev-parse --short HEAD)"
echo "next (cluster, NOT run here):  bash $DEST/cluster/stage_code.sh"
