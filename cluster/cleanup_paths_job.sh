#!/bin/bash
# SUBMIT WITH: JOB_ENV="CLEANUP_PATHS='name1 name2 ...'" bash cluster/submit_job.sh cluster/cleanup_paths_job.sh -s 4 -q xeon-p8
# Delete named top-level entries of ~/learned-ik as a JOB: a bulk rm -rf of a venv or a staged tree is
# many thousands of Lustre metadata operations, which is a job's work, not the login node's.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# Deleting cluster files needs Thomas's OK EACH TIME (SuperCloud has no backups); record what he
# approved in the commit or session that submits this.
# ===========================================================================
#
# CLEANUP_PATHS: space-separated names DIRECTLY under $ROOT (no '/', no '..', no globs). The live
# environment (repo, venv, drake, sysdeps, home, state, results) is refused outright. Every name must
# exist, or nothing is deleted. Sizes are recorded before deletion. Sentinel: $ROOT/cleanup_paths.DONE.
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
source "$ROOT/repo/cluster/verify_staged_tree.sh" || exit 1; lik_job_check "$ROOT/repo"   # code integrity (README)
DONE="$ROOT/cleanup_paths.DONE"; LOG="$ROOT/cleanup_paths.log"
rm -f "$DONE"
exec > >(tee -a "$LOG") 2>&1
echo "===== cleanup on $(hostname) at $(date -Is): [${CLEANUP_PATHS:-}] ====="
Fail() { echo "CLEANUP REFUSED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }

[ -n "${CLEANUP_PATHS:-}" ] || Fail "CLEANUP_PATHS is empty"
cd "$ROOT" || Fail "no $ROOT"
for n in $CLEANUP_PATHS; do
    case "$n" in */*|*..*|*'*'*|*'?'*|*'['*|.|'') Fail "bad name '$n' (top-level names only)" ;; esac
    case "$n" in repo|venv|drake|sysdeps|home|state|results|.drake-ok|setup.DONE) Fail "'$n' is the live environment" ;; esac
    [ -e "$n" ] || [ -L "$n" ] || Fail "'$n' does not exist under $ROOT (nothing deleted)"
done
for n in $CLEANUP_PATHS; do
    echo "$(du -sh "$n" 2>/dev/null | cut -f1)	$n"
done
for n in $CLEANUP_PATHS; do
    rm -rf -- "$ROOT/$n" || Fail "rm -rf $n failed (earlier names may be gone)"
    echo "deleted $n"
done
echo "OK" > "$DONE"
echo "===== cleanup done at $(date -Is) ====="
