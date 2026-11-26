#!/bin/bash
# sbatch payload: ONE worker (one GPU, one solve at a time, no MPS) draining a manifest of benchmark items.
# Submitted as an array by cluster/engaging/submit.sh (`--array=0-N%N`, one task per GPU); every task
# claims items atomically (mkdir <id>.claim, run_items.sh) so any number of tasks, started at any time,
# drain one manifest, and re-submitting the same manifest is the mop-up.
#
# This reuses cluster/run_items.sh UNCHANGED apart from one env-gated suffix (LIK_TAG_SUFFIX; the claim owner already carries the Slurm job id):
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
# submit.sh passes LIK_CROOT relative to the cluster home.
case "${LIK_CROOT:?LIK_CROOT unset}" in /*) ;; *) LIK_CROOT="$HOME/$LIK_CROOT" ;; esac; export LIK_CROOT
source "$LIK_CROOT/repo/cluster/engaging/job_lib.sh" || exit 2
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
export LIK_TAG_SUFFIX="_j${SLURM_JOB_ID:-0}"
## run_items.sh refuses to claim an item the job's remaining wall cannot cover, and reads that wall with
## `squeue -j`. For an array task the spec must be <array id>_<task> (the first task's SLURM_JOB_ID is the array
## id, and `squeue -j <array id>` lists every task, concatenating their %L into garbage); and an unreadable
## remaining wall must FAIL LOUDLY here, never silently mean "unlimited".
[ -z "${SLURM_ARRAY_JOB_ID:-}" ] || export LIK_SQUEUE_JOB="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
WALL_LEFT="$(squeue -h -j "${LIK_SQUEUE_JOB:-${SLURM_JOB_ID:-}}" -o %L 2>/dev/null | tr -d '[:space:]')"
if ! [[ "$WALL_LEFT" =~ ^([0-9]+-)?([0-9]+:)?[0-9]+:[0-9]+$ ]]; then
    echo "FATAL: cannot read this job's remaining wall (squeue -j ${LIK_SQUEUE_JOB:-${SLURM_JOB_ID:-}} -o %L gave '${WALL_LEFT}'); run_items.sh would treat it as unlimited" >&2
    exit 3
fi
echo "job wall left at start: $WALL_LEFT (squeue -j ${LIK_SQUEUE_JOB:-$SLURM_JOB_ID})"
export ITEM_TIMEOUT="${ITEM_TIMEOUT:-28800}"
cd "$REPO" || exit 2
bash "$REPO/cluster/run_items.sh"
RC=$?
echo "bench_job: run_items rc=$RC at $(date -Is)"
exit $RC
