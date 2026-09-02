#!/bin/bash
# Shared SSH plumbing for MIT SuperCloud interaction.
#
# ============================ STANDING REMINDER ============================
# If you (human or agent) are even REMOTELY unsure about a SuperCloud action
# -- a command's side effects, a submission's size, a policy question -- STOP
# and ask Thomas before running it.  Cluster interaction goes through the
# committed scripts in cluster/, not ad-hoc ssh command strings.
# Be cautious, be polite, run as few commands as possible.
# ===========================================================================
#
# Provides:
#   $SC_DEST          -- user@host for the SuperCloud login node
#   $SC_ROOT          -- this project's isolated tree on the cluster
#   sc_run "<cmd>"    -- run a remote command over one multiplexed connection,
#                        logged (with timestamp) to logs/cluster_audit.log
#   sc_rsync <args>   -- rsync through the same multiplexed connection, logged
#
# The ControlMaster socket keeps ONE ssh session alive for 10 minutes of reuse
# (finite ControlPersist -- never "yes"), so repeated calls do not re-handshake.
#
# $SC_ROOT is deliberately NOT ~/ik-tune: that is a different project's tree,
# with its own venv, its own Drake and its own extracted-deb prefix, and it has
# jobs running.  Every project on this account gets an independent tree that
# `rm -rf` removes without disturbing anything else.

SC_DEST="tcohn@txe1-login.mit.edu"
SC_ROOT="learned-ik"
SC_CTL_DIR="${XDG_RUNTIME_DIR:-/tmp}/supercloud-ctl"
mkdir -p "$SC_CTL_DIR"
SC_SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=$SC_CTL_DIR/%r@%h:%p" -o ControlPersist=10m -o BatchMode=yes)

_sc_repo_root() { git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel; }
SC_AUDIT_LOG="$(_sc_repo_root)/logs/cluster_audit.log"
mkdir -p "$(dirname "$SC_AUDIT_LOG")"

sc_log() {
    printf '%s  %s\n' "$(date -Is)" "$*" >> "$SC_AUDIT_LOG"
}

sc_run() {
    sc_log "ssh $SC_DEST :: $*"
    ssh "${SC_SSH_OPTS[@]}" "$SC_DEST" "$@"
}

sc_rsync() {
    sc_log "rsync :: $*"
    rsync -e "ssh ${SC_SSH_OPTS[*]}" "$@"
}
