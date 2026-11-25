#!/bin/bash
# sbatch payload: ONE worker (one GPU, one solve at a time, no MPS) draining a manifest of benchmark items.
# Submitted as an array by cluster/engaging/submit.sh (`--array=0-N%N`, one task per GPU); every task
# claims items atomically (mkdir <id>.claim, run_items.sh) so any number of tasks, started at any time,
# drain one manifest, and re-submitting the same manifest is the mop-up.
#
# This reuses cluster/run_items.sh UNCHANGED apart from two env-gated lines (LIK_CLAIM_JOB, LIK_TAG_SUFFIX):
# the payload runs on the host (claims, overrun watcher, per-cell resume, result publishing) and only the
# interpreter is in the image (TEST_PYTHON = py_in_image.sh).
#
# PREEMPTION. On mit_preemptable (`--requeue`) a killed task is requeued with the SAME job id and restarts
# here from the top. Its dead attempt's claims are released first (eg_release_own_claims), and the item
# resumes from its per-cell JSONL (LEARNED_IK_RESUME=1, set by run_items.sh): only the interrupted cell
# is redone.
#
# Env (set by submit.sh): LIK_CROOT MANIFEST EXPECT_MANIFEST_SHA EXPECT_SIF_SHA ITEM_TIMEOUT
#                         LIK_EXPECT_GPU LIK_EXPECT_CPU
source "${LIK_CROOT:?LIK_CROOT unset}/repo/cluster/engaging/job_lib.sh" || exit 2
eg_job_init

[ -n "${MANIFEST:-}" ] && [ -f "$REPO/$MANIFEST" ] || { echo "bench_job.sh: no manifest $REPO/${MANIFEST:-?}" >&2; exit 2; }
STATE="$LIK_CROOT/state/$(basename "$MANIFEST" .txt)"
mkdir -p "$STATE"
echo "$LIK_HARDWARE_LINE" >> "$STATE/hardware.log"
eg_release_own_claims "$STATE"

## Cache dir layout run_items.sh expects ($ROOT/home/.cache/{ikflow,drake}); the caches are warmed once.
ln -sfn "$EB/home-cache" "$LIK_CROOT/home"
[ -d "$EB/home-cache/.cache/ikflow" ] || { echo "no warmed caches: run the plumbing/warm job first" >&2; exit 2; }

export LEARNED_IK_ROOT="$LIK_CROOT" TEST_PYTHON="$REPO/cluster/engaging/py_in_image.sh" PROCS=1 MPS=0
export LIK_CLAIM_JOB="${SLURM_JOB_ID:-}" LIK_TAG_SUFFIX="_j${SLURM_JOB_ID:-0}"
export ITEM_TIMEOUT="${ITEM_TIMEOUT:-28800}"
cd "$REPO" || exit 2
bash "$REPO/cluster/run_items.sh"
RC=$?
echo "bench_job: run_items rc=$RC at $(date -Is)"
exit $RC
