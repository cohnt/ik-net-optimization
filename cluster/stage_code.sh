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

sc_run "mkdir -p ~/$SC_ROOT/repo ~/$SC_ROOT/state ~/$SC_ROOT/results ~/$SC_ROOT/home/.cache"

sc_rsync -az --delete \
    --exclude='.git/' --exclude='.claude/' --exclude='.venv/' \
    --exclude='results/' --exclude='logs/' --exclude='notebooks/artifacts/' \
    --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
    --exclude='workshop-paper-draft.pdf' \
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
