#!/bin/bash
# Submit ONE tree job script by hand, with the laptop code check (README "Code integrity"):
#
#   bash cluster/submit_job.sh cluster/smoke.sh -g volta:2 -s 40 -q debug-gpu -T 0:20:00
#   JOB_ENV="FK_RUNG=soft12" bash cluster/submit_job.sh cluster/fk_surrogate_job.sh -s 20 -q xeon-g6-volta
#   STAGE_SUBDIR=repo-next SUBMIT_CMD="sbatch --dependency=afterany:<job>" \
#       bash cluster/submit_job.sh cluster/swap_staged_tree.sh
#
# Checks the staged tree (STAGE_SUBDIR, default repo) against the laptop manifest, then runs
# `[JOB_ENV] EXPECT_MANIFEST_SHA=<sha> ${SUBMIT_CMD:-LLsub} ./<script> <args>` inside it. The job script
# refuses to run without EXPECT_MANIFEST_SHA, so a bare LLsub of a tree script stops at job start.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"
source "$(dirname "$0")/verify_staged_tree.sh"
SCRIPT="${1:?usage: submit_job.sh cluster/<script>.sh [submit args...]}"; shift
case "$SCRIPT" in cluster/*.sh) ;; *) echo "REFUSING: $SCRIPT is not a cluster/*.sh path" >&2; exit 2 ;; esac
lik_laptop_check || exit $?
ARGS=""; [ $# -gt 0 ] && ARGS=$(printf '%q ' "$@")
sc_run "cd ~/$SC_ROOT/${STAGE_SUBDIR:-repo} && ${JOB_ENV:-} EXPECT_MANIFEST_SHA=$EXPECT_MANIFEST_SHA ${SUBMIT_CMD:-LLsub} ./$SCRIPT $ARGS"
