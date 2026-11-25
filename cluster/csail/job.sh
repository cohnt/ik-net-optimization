#!/bin/bash
# JOB PAYLOAD of a benchmark campaign on CSAIL's shared PREEMPTIBLE GPUs: one array task = one GPU = one worker
# process (PROCS=1 always: one solve per GPU, CLAUDE.md "Processes per GPU"). Submitted by cluster/csail/submit.sh,
# which passes the variables listed in job_lib.sh plus MANIFEST (repo-relative path) in front of sbatch.
#
# Runs the UNCHANGED cluster/run_items.sh inside the image. What this wrapper adds:
#
#   PREEMPTION. The queue's GraceTime is 0: a preempted job gets SIGTERM and is killed, then (--requeue) goes back
#   in the queue under the SAME job id and starts again from the top of this script. That is the normal case here.
#   Item progress survives because the benchmark appends every finished cell to its JSONL on /data and resumes
#   from it (LEARNED_IK_RESUME=1, set by run_items.sh).
#
#   THE STALE-CLAIM TRAP. run_items.sh claims an item with `mkdir state/<manifest>/<id>.claim` and only a
#   `.done` marker proves it finished. A killed attempt leaves its claims behind, and nothing in run_items.sh
#   ever clears one: the items would be orphaned. Each claim's owner file now ends with the Slurm job id (the one
#   shared-file change: run_items.sh line "echo \"$TAGID ...\" > $CLAIM/owner"). So, before every round:
#     - a claim with no .done owned by THIS job id is stale (the attempt that made it is dead: we are the new one)
#       and is released;
#     - a claim owned by another job is released iff that job is not RUNNING (gone from squeue, or PENDING after a
#       requeue). A RUNNING owner keeps its claim. A claim with no owner job id (a hand run) is never touched.
#   run_items.sh walks the manifest once, so after a round the wrapper re-reads the state and runs it again while
#   unclaimed items remain, which also picks up items a preempted sibling task released meanwhile.
#
#   The SIGTERM trap copies this attempt's worker logs (node-local) next to the results before exiting; a loop
#   copies them every 60 s as well, since with GraceTime 0 the trap may never get to run.
#
# Exit: 0 when no item of the manifest is left unclaimed-and-unfinished by this task; 1 if run_items reported a
# failed item; 3 GPU fault; 1 on an integrity STOP.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/job_lib.sh"
lik_start
lik_need MANIFEST
echo "===== csail job $JOBTAG attempt $ATTEMPT on $(hostname) at $(date -Is) ====="
lik_end_epoch
lik_machine_record
lik_gpu_check
# One solve per GPU, whatever the caller asked.
export PROCS=1

NAME="$(basename "$MANIFEST" .txt)"
STATE="$LIK_ROOT/state/$NAME"; mkdir -p "$STATE"
mkdir -p "$LIK_ROOT/home/.cache"
ln -sfn /opt/lik/home/.cache/ikflow "$LIK_ROOT/home/.cache/ikflow"   # dangling on the host, valid in the image
ln -sfn /opt/lik/home/.cache/drake  "$LIK_ROOT/home/.cache/drake"

mapfile -t IDS < <(grep -vE '^[[:space:]]*(#|$)' "$REPO/$MANIFEST" | cut -d'|' -f1)
[ "${#IDS[@]}" -gt 0 ] || { echo "no items in $MANIFEST"; exit 2; }

## ---- the stale-claim release (the trap above) -------------------------------------------------------------
lik_job_running() {   # is job $1 RUNNING right now? (a requeued job is PENDING; a finished one is gone)
    local st; st=$(squeue -h -j "$1" -o %T 2>/dev/null | head -1 | tr -d '[:space:]')
    case "$st" in RUNNING|COMPLETING) return 0 ;; esac
    return 1
}
lik_release_stale() {   # $1 = "self" (start of an attempt: own claims are stale) or "others"
    local id c owner job n=0
    for id in "${IDS[@]}"; do
        c="$STATE/$id.claim"
        [ -d "$c" ] && [ ! -f "$STATE/$id.done" ] || continue
        job=$(awk '{print $3}' "$c/owner" 2>/dev/null)
        [ -n "$job" ] || continue
        if [ "$job" = "${SLURM_JOB_ID:-}" ]; then
            [ "$1" = self ] || continue
        elif lik_job_running "$job"; then
            continue
        fi
        rm -rf "$c" && n=$((n + 1)) && echo "released stale claim $id (owner job $job, ${SLURM_JOB_ID:-none} is the releaser)"
    done
    echo "released $n stale claim(s)"
}
lik_unclaimed() {   # items that are neither done nor claimed
    local id n=0
    for id in "${IDS[@]}"; do [ -f "$STATE/$id.done" ] || [ -d "$STATE/$id.claim" ] || n=$((n + 1)); done
    echo "$n"
}
lik_done_count() { local id n=0; for id in "${IDS[@]}"; do [ -f "$STATE/$id.done" ] && n=$((n + 1)); done; echo "$n"; }

## ---- logs and the TERM trap -----------------------------------------------------------------------------------
lik_sync_logs() { local f; for f in "$TMPDIR"/worker_*.log; do [ -f "$f" ] && cp -f "$f" "$LOGD/a$ATTEMPT.$(basename "$f")"; done; return 0; }
lik_tree() { local c; echo "$1"; for c in $(pgrep -P "$1"); do lik_tree "$c"; done; }
CHILD=""
on_term() {
    echo "SIGTERM at $(date -Is): preempted or cancelled; saving logs, leaving the claims for the next attempt to release"
    [ -z "$CHILD" ] || kill -TERM $(lik_tree "$CHILD") 2>/dev/null   # the whole subtree at once: workers must not start the next item
    lik_sync_logs
    exit 143
}
trap on_term TERM
( while sleep 60; do lik_sync_logs; done ) & SYNCER=$!
trap 'pkill -P $SYNCER 2>/dev/null; kill $SYNCER 2>/dev/null; lik_sync_logs; lik_cleanup' EXIT

lik_release_stale self
RC=0
for ROUND in $(seq 1 "${MAX_ROUNDS:-50}"); do
    [ "$ROUND" -eq 1 ] || lik_release_stale others
    LEFT=$(lik_unclaimed)
    echo "--- round $ROUND: $(lik_done_count)/${#IDS[@]} done, $LEFT unclaimed ($(date -Is))"
    [ "$LEFT" -gt 0 ] || break
    lik_container bash "$REPO/cluster/run_items.sh" & CHILD=$!
    wait "$CHILD"; R=$?; CHILD=""
    lik_sync_logs
    if [ "$R" -ne 0 ]; then RC=1; echo "run_items.sh exited $R (an item failed or the job is out of time)"; break; fi
    ## Not enough job time left for even one more item: stop; the requeue or a sibling finishes the rest.
    NOW=$(date +%s); [ -z "${JOB_END_EPOCH:-}" ] || [ $(( JOB_END_EPOCH - NOW )) -gt $(( ${ITEM_TIMEOUT:-28800} + 300 )) ] || break
done
echo "===== job $JOBTAG attempt $ATTEMPT leaving at $(date -Is): $(lik_done_count)/${#IDS[@]} done, $(lik_unclaimed) unclaimed, rc $RC ====="
exit $RC
