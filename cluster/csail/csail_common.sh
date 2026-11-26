#!/bin/bash
# Shared plumbing for the CSAIL scripts (cluster/CSAIL_RUNBOOK.md). Sourced by the laptop-side scripts in this
# directory; never executed, never sent to the cluster.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a CSAIL cluster action, STOP and ask Thomas. All traffic goes through the
# csail-cluster skill's cs_run / cs_rsync over the EXISTING connection: nothing here opens a login, nothing calls
# plain ssh/scp/rsync, nothing touches a compute node, and the lab node (tedrake-h200-1, locomotion-h200) is never
# named as a target. If cs_status fails: STOP and tell Thomas (a login is his act, with Duo).
# ===========================================================================
#
# Provides (after `source`):
#   CS_DEST CS_USER CS_ROOT        the login, the user, the project's scratch tree
#   cs_run / cs_rsync / cs_status  from the skill's csail_ssh.sh (CSAIL_SSH_SH overrides the path: tests only)
#   csail_connected                exit 75 (cannot verify) unless the shared connection is up
#   csail_repo_root                the git checkout this script lives in
#   csail_tree_root C12            $CS_ROOT/roots/<12 hex>: one staged commit = one root (repo/, state/, results/...)
#
# The skill's local file (~/.claude/csail-cluster.local.md) holds the username; CS_DEST must be exported or set
# in $CSAIL_ENV_FILE (default none). It is never guessed.

[ -n "${BASH_VERSION:-}" ] || { echo "csail_common.sh: bash only" >&2; return 1; }

if [ -z "${CS_DEST:-}" ]; then
    echo "csail_common.sh: CS_DEST is not set (<csail username>@slurm-login.csail.mit.edu; see ~/.claude/csail-cluster.local.md)" >&2
    return 1 2>/dev/null || exit 1
fi
CS_USER="${CS_DEST%@*}"
CS_ROOT="${CS_ROOT:-/data/locomotion/scratch/$CS_USER/learned-ik}"
## CSAIL_ROOT_PREFIX is for the stub test only (tests/test_csail_scripts.py); on the cluster it is /data.
case "$CS_ROOT" in
    "${CSAIL_ROOT_PREFIX:-/data}"/*) ;;
    *) echo "csail_common.sh: CS_ROOT=$CS_ROOT is not under /data (jobs cannot see AFS)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
case "$CS_ROOT" in
    *[Cc][Ll][Aa][Uu][Dd][Ee]*) echo "csail_common.sh: keep the string 'claude' out of remote paths (skill, rule 6)" >&2
                                return 1 2>/dev/null || exit 1 ;;
esac

CSAIL_SSH_SH="${CSAIL_SSH_SH:-$HOME/.claude/skills/csail-cluster/scripts/csail_ssh.sh}"
# shellcheck disable=SC1090
. "$CSAIL_SSH_SH" || { echo "csail_common.sh: cannot source $CSAIL_SSH_SH" >&2; return 1 2>/dev/null || exit 1; }

csail_repo_root() { git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel; }

csail_connected() {
    cs_status || { echo "STOP: the shared CSAIL connection is down. Do not retry or log in: tell Thomas (a login is his act, with Duo)." >&2; return 75; }
}

csail_tree_root() { echo "$CS_ROOT/roots/${1:?csail_tree_root <12 hex>}"; }

## Partition table. NEVER the lab node: locomotion-h200 does not appear here, and vision-shared-h200 always
## carries --exclude=tedrake-h200-1. -A/-p/-q follow the skill's table. Constraint names read from sinfo %f.
##   csail_gpu_args <kind>  prints the sbatch option words for a preemptible GPU job of that GPU kind.
csail_gpu_args() {
    local common="-q shared-if-available --requeue --open-mode=append"
    case "${1:?gpu kind}" in
        rtx3090) echo "-A locomotion -p vision-shared-rtx3090 $common --constraint=nvidia_geforce_rtx_3090" ;;
        rtx4090) echo "-A locomotion -p vision-shared-rtx4090 $common --constraint=nvidia_geforce_rtx_4090" ;;
        l40s)    echo "-A locomotion -p vision-shared-l40s $common --constraint=nvidia_l40s" ;;
        a100)    echo "-A locomotion -p vision-shared-a100 $common --constraint=nvidia_a100-sxm4-80gb" ;;
        h200)    echo "-A locomotion -p vision-shared-h200 $common --constraint=nvidia_h200 --exclude=tedrake-h200-1" ;;
        csail-h200) echo "-A csail -p csail-shared-h200 -q shared-if-available --requeue --open-mode=append --constraint=nvidia_h200" ;;
        *) echo "unknown GPU kind '$1' (rtx3090 rtx4090 l40s a100 h200 csail-h200)" >&2; return 2 ;;
    esac
}

## Refuse anything that names the lab node, in any option string about to be submitted.
csail_refuse_lab_node() {
    case "$*" in
        *locomotion-h200*|*locomotion-low*|*locomotion-main*|*locomotion-high*)
            echo "REFUSING: the lab node (locomotion-h200 / locomotion-* QoS) is never used by these scripts" >&2; return 1 ;;
    esac
    case "$*" in
        *vision-shared-h200*) case "$*" in *--exclude=tedrake-h200-1*) ;; *)
            echo "REFUSING: vision-shared-h200 without --exclude=tedrake-h200-1" >&2; return 1 ;; esac ;;
    esac
    return 0
}
