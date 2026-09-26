#!/bin/bash
# Stage the repo and the gitignored model checkpoint onto SuperCloud.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Everything lands under ~/learned-ik/, this project's own tree. ~/ik-tune is a
# different project's and is never touched.
#
# Two things need explicit care:
#   - models/ is NOT excluded. The iiwa checkpoint
#     (iiwa14__lemon-haze-7__global_step_4.25M.pkl, 204 MB) is gitignored and
#     exists only on this workstation, and src/iiwa_program.py loads it by a path
#     relative to RepoDir(), so it has to travel with the tree. The panda weights
#     are different -- ikflow downloads those, which cluster/setup_supercloud.sh
#     does on the download partition, since compute nodes have no internet.
#   - results/ IS excluded. It is large, it is the output, and pushing a local
#     copy up would make it ambiguous which machine produced a summary.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"
REPO_ROOT="$(_sc_repo_root)"

COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: tracked files are dirty; staging the working tree anyway." >&2
fi

## Refuse to restage under a live campaign unless explicitly forced. rsync has no
## --inplace here, so it renames over the tree and a RUNNING script keeps its own
## inode -- but any .py an item imports when it STARTS is re-read from disk, so a
## mid-stage restage silently produces one result set built from two code
## versions. That is the same hazard as changing a result schema mid-campaign,
## and it is invisible afterwards. Calibration and smoke jobs are exempt: they
## produce no campaign records.
## Match on the payload script name, which LLstat shows when a job is submitted
## without -J, AND on the lik_<stage>_n<i> convention used when it is. These are
## PREFIXES on purpose: LLstat truncates NAME to 15 characters, so a full job name
## like `lik_train_iiwa14_n6` shows as `lik_train_iiwa1` and would never match. Calibration
## and smoke are named lik_cal_* / smoke.sh and deliberately do not match.
RUNNING=$(sc_run 'LLstat 2>/dev/null | grep -c "run_items\|train_flow\|lik_train\|lik_[A-Za-z]*_n[0-9]"' 2>/dev/null | tr -dc '0-9')
if [ -n "${RUNNING:-}" ] && [ "${RUNNING:-0}" -gt 0 ] && [ "${FORCE_STAGE:-0}" != "1" ]; then
    echo "REFUSING: $RUNNING campaign job(s) are on the cluster right now." >&2
    echo "Restaging would change the code later items import mid-stage." >&2
    echo "Wait for the stage to drain, or re-run with FORCE_STAGE=1 if you are sure." >&2
    exit 3
fi

## An INCOMPLETE local tree is worse than a stale one, because the rsync below runs with
## --delete: anything missing here is deleted THERE. Two ways to arrive with one, both hit
## on 2026-09-16 while staging from a git worktree:
##
##   - a worktree does not check out submodules, so third_party/ikflow was empty locally and
##     rsync deleted the cluster's vendored fork down to 18 files;
##   - the gitignored checkpoints had been symlinked into the worktree, and rsync copies a
##     symlink AS a symlink, so three cluster-side .pkl files were replaced by links into a
##     path that does not exist there. The --filter=P rules protect them from deletion, not
##     from being overwritten with a link.
##
## Both are cheap to check and expensive to undo, so check.
if [ -f "$REPO_ROOT/.gitmodules" ]; then
    UNINIT=$(git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | grep -c "^-" || true)
    if [ "${UNINIT:-0}" -gt 0 ]; then
        echo "REFUSING: $UNINIT uninitialised submodule path(s) locally." >&2
        echo "rsync runs with --delete, so staging now would DELETE them on the cluster." >&2
        echo "Fix: git -C $REPO_ROOT submodule update --init --recursive" >&2
        exit 4
    fi
fi
LINKS=$(find "$REPO_ROOT/models" -type l 2>/dev/null | wc -l | tr -d " ")
if [ "${LINKS:-0}" -gt 0 ]; then
    echo "REFUSING: $LINKS symlink(s) under models/." >&2
    echo "rsync copies a symlink as a symlink, so the cluster would get links into a path" >&2
    echo "that does not exist there, replacing real checkpoints. Use hardlinks instead." >&2
    find "$REPO_ROOT/models" -type l >&2
    exit 5
fi

sc_run "mkdir -p ~/$SC_ROOT/repo ~/$SC_ROOT/state ~/$SC_ROOT/results ~/$SC_ROOT/home/.cache"

## PROTECT filters, not excludes: cluster-side exported checkpoints must survive
## --delete, but locally-held ones (lemon-haze-7, ddp-r1) must still be PUSHED.
## An --exclude would do both; rsync's "P" filter only suppresses deletion.
## This bit on 2026-09-10: iiwa14_n4's step-620000 export lived only on the cluster,
## a routine stage_code run deleted it, and the queued benchmark would have hit a
## missing checkpoint. models/ is NOT excluded (see the header), so it needs this.
## What is never staged. rsync reads the WORKING TREE, not git, so .gitignore has NO
## effect here -- anything present locally is pushed unless it is named below. That is
## how `.venv-soromox` reached the cluster and stayed there: the JAX oracle venv was
## committed by accident, then removed from history and gitignored, and none of that
## touched the cluster, because none of it is something rsync reads. It cost 621 MB of
## Lustre and about eleven minutes of every stage until it was named here (2026-09-26).
##
## An --exclude ALSO protects the path from --delete, so adding one stops future pushes
## but does NOT remove a copy already on the cluster. That needs an explicit rm.
EXCLUDES=(
    '.git/' '.git' '.claude/' '.venv/' '.venv-soromox/'
    'results/' 'logs/' 'notebooks/artifacts/'
    '__pycache__/' '*.pyc' '.pytest_cache/'
    'workshop-paper-draft.pdf'
)

## STAGE_EXTRA_EXCLUDES adds to the list for one run, space-separated:
##   STAGE_EXTRA_EXCLUDES='models/panda/ scratch.npz' bash cluster/stage_code.sh
## Same warning as above: an exclude added here suppresses the PUSH and the DELETE, so
## it holds whatever is on the cluster frozen rather than removing it.
if [ -n "${STAGE_EXTRA_EXCLUDES:-}" ]; then
    read -r -a _EXTRA_EXCLUDES <<< "$STAGE_EXTRA_EXCLUDES"
    EXCLUDES+=("${_EXTRA_EXCLUDES[@]}")
fi

EXCLUDE_ARGS=()
for _e in "${EXCLUDES[@]}"; do
    [ -n "$_e" ] && EXCLUDE_ARGS+=("--exclude=$_e")
done
echo "excluding: ${EXCLUDES[*]}"

sc_rsync -az --delete \
    --filter='P models/*/*__step*.pkl' --filter='P models/*/*__step*.arch.json' \
    --filter='P models/*/*__global_step*.pkl' --filter='P models/*/*__global_step*.arch.json' \
    "${EXCLUDE_ARGS[@]}" \
    "$REPO_ROOT/" "$SC_DEST:$SC_ROOT/repo/"

sc_run "echo $COMMIT > ~/$SC_ROOT/repo/.staged-commit
echo 'staged commit:' \$(cat ~/$SC_ROOT/repo/.staged-commit)
du -sh ~/$SC_ROOT/repo
IIWA=~/$SC_ROOT/repo/models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl
if [ -f \"\$IIWA\" ]; then
    echo \"iiwa checkpoint: \$(du -h \"\$IIWA\" | cut -f1)  sha256 \$(sha256sum \"\$IIWA\" | cut -c1-16)\"
else
    echo 'ERROR: iiwa checkpoint did not arrive -- the iiwa arms cannot run' >&2
fi"

echo
echo "local iiwa checkpoint sha256: $(sha256sum "$REPO_ROOT/models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl" 2>/dev/null | cut -c1-16)"
