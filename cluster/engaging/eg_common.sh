#!/bin/bash
# Laptop-side plumbing for MIT ORCD Engaging (sourced by cluster/engaging/*.sh; see ENGAGING_RUNBOOK.md).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about an Engaging action, STOP and ask Thomas. Logging in is HIS act
# (password + Duo on the workstation): nothing here ever calls eg_open, and when the shared
# connection is down every script stops and says so. Never retry a login.
# ===========================================================================
#
# All traffic goes through the engaging-cluster skill's eg_run / eg_rsync over its ONE existing
# connection (it ignores ~/.ssh/config and refuses any other host). Provides:
#   $EG_DEST   tcohn@orcd-login.mit.edu (env, else ~/.claude/engaging-cluster.local.md)
#   $EG_BASE   this project's isolated tree on the cluster, ~/lik-engaging (NOT ~/learned-ik, the
#              pre-2026-11-11 CPU attempt, and nobody else's)
#   eg_commit_root <commit>   -> lik-engaging/c-<12 hex>   (one tree per commit: a requeued job can
#                                never see code change under it)
# and the SuperCloud names verify_staged_tree.sh expects (sc_run, SC_ROOT, _sc_repo_root), so the
# integrity check is the SAME code as SuperCloud's, not a fork.

EG_SKILL="${EG_SKILL:-$HOME/.claude/skills/engaging-cluster/scripts/engaging_ssh.sh}"
if [ -z "${EG_DEST:-}" ]; then
    EG_DEST=$(sed -n 's/.*EG_DEST=\([A-Za-z0-9._@-]*\).*/\1/p' "$HOME/.claude/engaging-cluster.local.md" 2>/dev/null | head -1)
fi
# Security (engaging-cluster skill): only <kerberos>@orcd-login.mit.edu is ever used.
if ! [[ "${EG_DEST:-}" =~ ^[A-Za-z0-9._-]+@orcd-login\.mit\.edu$ ]]; then
    echo "eg_common.sh: EG_DEST='${EG_DEST:-}' is not <user>@orcd-login.mit.edu; refusing." >&2
    return 1 2>/dev/null || exit 1
fi
export EG_DEST
# shellcheck disable=SC1090
. "$EG_SKILL" || { echo "eg_common.sh: cannot source $EG_SKILL" >&2; return 1 2>/dev/null || exit 1; }

EG_BASE="${EG_BASE:-lik-engaging}"
_eg_repo_root() { git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel; }
EG_REPO="$(_eg_repo_root)"

eg_commit_root() { echo "$EG_BASE/c-${1:0:12}"; }

## The connection must be up before anything else. Down => STOP (rule 2): this never logs in.
eg_require_connection() {
    eg_status || { echo "STOP: the Engaging connection is down. Tell Thomas; do NOT log in or retry." >&2; return 255; }
}

## --- the names verify_staged_tree.sh uses on SuperCloud ------------------------------------
sc_run() { eg_run "$@"; }
_sc_repo_root() { _eg_repo_root; }
SC_ROOT="$EG_BASE"
# shellcheck disable=SC1091
. "$EG_REPO/cluster/verify_staged_tree.sh"

## eg_check_commit <commit>: the laptop's reference vs the cluster's own sha256sum, inside that commit's
## tree. Exports EXPECT_MANIFEST_SHA. Exit 0 ok, 1 STOP (tree differs), 75 cannot verify, 76 no reference.
eg_check_commit() {
    STAGE_SUBDIR="c-${1:0:12}/repo" lik_laptop_check
}
