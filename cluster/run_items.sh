#!/bin/bash
# The LLsub payload that drains a manifest of benchmark runs on one GPU node.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node), one job per node:
#   MANIFEST=cluster/manifest_stageA.txt PROCS=1 \
#     LLsub ./cluster/run_items.sh -g volta:2 -s 40 -q xeon-g6-volta -T 48:00:00
#
# Keep the job wall ABOVE ITEM_TIMEOUT (below), or Slurm kills a long item and every other
# worker on the node with it.  `xeon-g6-volta` allows 4-04:00:00 (100 h), so there is room.
#
# Submit as many of these as there are items worth running: the account's
# `xeon-g6-volta` cap is a Slurm GrpTRES *group* limit (node=4), not a per-job
# limit, so surplus jobs are ACCEPTED AND QUEUED (`PENDING`, reason
# `AssocGrpNodeLimit`) and start as earlier ones finish.  There is no need to
# hand-throttle submissions or poll in order to submit the next one.
#
# WORK CLAIMING -- why this is a lock and not a rank stride.  The sibling
# project's runner deals item k to rank (k*STRIDE)%SIZE, which is right when a
# fixed set of ranks all start at once.  Here they do not: jobs start whenever
# the group limit frees a node, so there is no stable rank space to deal into.
# Instead every process claims work atomically:
#
#     mkdir "$STATE/<id>.claim"      # atomic on POSIX/Lustre; fails if taken
#
# A failed mkdir means someone else owns the item.  Any number of processes, in
# any number of jobs, started at any time, therefore cooperate to drain one
# manifest with no coordination and no duplicated work -- and re-submitting the
# same manifest is still exactly the mop-up mechanism.
#
# A finished item also touches "<id>.done".  A claim WITHOUT a done marker is a
# dead item (its process was killed); `cluster/collect_results.sh --reclaim`
# clears those so a resubmission picks them up.  Nothing here ever clears a
# claim automatically -- an item that is genuinely still running elsewhere must
# not be stolen.
#
# PROCS is how many worker processes this job forks on its node.  It comes from
# the calibration (cluster/calibrate.sh), NOT from a guess: the benchmark is
# wall-clock capped, so concurrent processes that contend for CPU change what is
# measured.  PROCS=1 is always safe.
#
# NOTE: source /etc/profile BEFORE enabling -u -- the cluster's
# /etc/profile.d/Z97-byobu.sh reads an unset LC_BYOBU and would kill the job.
source /etc/profile
set -uo pipefail

## --- GPU selection --------------------------------------------------------
## Slurm exports CUDA_VISIBLE_DEVICES as a comma-separated list, and on this cluster
## the entries are GPU **UUIDs**, not indices -- so exporting "0" or "1" is not a
## valid selector for the allocated cards. Capture the list once, before any worker
## narrows it, and hand each worker one entry from it.
SLURM_GPUS="${CUDA_VISIBLE_DEVICES:-}"
PinGpu() {   ## $1 = worker index; empty $SLURM_GPUS means "no GPU was allocated"
    if [ -z "$SLURM_GPUS" ]; then export CUDA_VISIBLE_DEVICES=""; return; fi
    local n; n=$(echo "$SLURM_GPUS" | tr ',' '\n' | grep -c .)
    export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_GPUS" | cut -d, -f$(( $1 % n + 1 )))"
}

## Seconds of wall clock this Slurm job has left, or "" when that cannot be known (no
## SLURM_JOB_ID, squeue absent, unparseable output) -- in which case the caller treats the
## budget as unlimited, so a laptop or login-node run behaves exactly as it did before.
## squeue's %L is D-HH:MM:SS / HH:MM:SS / MM:SS, and "UNLIMITED" for a job without a limit.
JobSecondsLeft() {
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    command -v squeue >/dev/null 2>&1 || return 0
    local L; L="$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | tr -d '[:space:]')"
    case "$L" in
        ""|UNLIMITED|NOT_SET|INVALID) return 0 ;;
    esac
    local D=0
    case "$L" in *-*) D="${L%%-*}"; L="${L#*-}" ;; esac
    local A B C
    IFS=: read -r A B C <<< "$L"
    case "$L" in
        *:*:*) : ;;
        *:*)   C="$B"; B="$A"; A=0 ;;
        *)     C="$A"; B=0; A=0 ;;
    esac
    ## Strip leading zeros so 08 is not read as octal, then emit seconds.
    printf '%d' $(( 10#${D:-0} * 86400 + 10#${A:-0} * 3600 + 10#${B:-0} * 60 + 10#${C:-0} )) \
        2>/dev/null || return 0
}

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="${TEST_REPO:-$ROOT/repo}"
STATE_ROOT="${TEST_STATE_DIR:-$ROOT/state}"
RESULTS_ROOT="${LEARNED_IK_RESULTS:-$ROOT/results}"
PROCS="${PROCS:-1}"
## A BACKSTOP FOR A WEDGED SOLVE, NOT A BUDGET.  It was 4 h, exactly equal to what
## submit_bench.sh requested as the Slurm wall, which meant one long item could consume the
## whole job and Slurm's kill would land on all PROCS workers at once, leaving PROCS stale
## claims.  The `xeon-g6-volta` ceiling is 4-04:00:00 (100 h), so there is no reason to sit
## anywhere near the job wall; submit_bench.sh now asks for 48 h and the invariant to keep is
## JOB WALL > ITEM_TIMEOUT, with the ClaimBudget guard below refusing an item the remaining
## job time cannot cover.  A real wedge does exist: generic_program.py records a 102-minute
## stall inside a single IPOPT iteration during which no Python ran, so a hard cap is needed.
ITEM_TIMEOUT="${ITEM_TIMEOUT:-28800}"
export TMPDIR="${TMPDIR:-/tmp}"

if [ -z "${MANIFEST:-}" ]; then
    echo "run_items.sh: set MANIFEST to a manifest path (cluster/gen_manifest.py)" >&2
    exit 2
fi
cd "$REPO" || { echo "run_items.sh: no repo at $REPO" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "run_items.sh: no manifest at $MANIFEST (cwd $PWD)" >&2; exit 2; }

MANIFEST_NAME="$(basename "$MANIFEST" .txt)"
STATE_DIR="$STATE_ROOT/$MANIFEST_NAME"
mkdir -p "$STATE_DIR/logs"

## --- one worker ------------------------------------------------------------
Worker() {
    local LOCAL=$1
    local TAGID; TAGID="$(hostname)_p$LOCAL"
    local LOG="$TMPDIR/worker_$LOCAL.log"

    ## Per-worker HOME.  jrl/urdf_utils.py unconditionally truncates and rewrites
    ## ~/.cache/jrl/urdfs/<robot>_link_filepaths_absolute.urdf at every Robot()
    ## construction and klampt then reads it back -- with several workers starting
    ## at once on a shared $HOME that is a truncate-while-reading race.  This also
    ## moves the torch/inductor/triton/matplotlib caches off Lustre.
    local REAL_HOME="$HOME"
    export HOME="$TMPDIR/home.$LOCAL"
    mkdir -p "$HOME/.cache"
    ## The ikflow weight cache is warmed on the download partition (compute nodes have
    ## no internet) and used read-only from here. jrl's cache is deliberately NOT shared:
    ## it is regenerated from package data with no network, and sharing it is exactly the
    ## race described above.
    ln -sfn "$ROOT/home/.cache/ikflow" "$HOME/.cache/ikflow" 2>/dev/null || true
    ## Drake's drake_models package, fetched during setup: compute nodes cannot download
    ## it, and ProcessModelDirectives would fail with "Network is unreachable".
    ln -sfn "$ROOT/home/.cache/drake" "$HOME/.cache/drake" 2>/dev/null || true

    export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    ## One thread per process: the node is filled with processes, not threads.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg MPLCONFIGDIR="$TMPDIR/mpl.$LOCAL"
    export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor.$LOCAL"
    export TRITON_CACHE_DIR="$TMPDIR/triton.$LOCAL"
    mkdir -p "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    ## Pin to one of the node's two V100s.  Besides balancing them, a single
    ## visible device takes jrl.config._get_device() down its fast path instead
    ## of an nvml "least used GPU" poll that would stampede every worker onto the
    ## same card.  DEVICE=cpu (set per item in the manifest) clears it instead.
    if [ "${DEVICE:-gpu}" = "cpu" ]; then
        export CUDA_VISIBLE_DEVICES=""
    else
        PinGpu "$LOCAL"
    fi
    local PY="${TEST_PYTHON:-$ROOT/venv/bin/python}"

    ## Stagger, so PROCS workers do not import torch and Drake in lockstep.
    sleep $(( LOCAL * 3 ))

    {
      echo "=== worker $LOCAL on $(hostname) at $(date -Is) ==="
      echo "manifest $MANIFEST, state $STATE_DIR, TMPDIR=$TMPDIR"
      echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES:-unset}] HOME=$HOME"
      echo "staged commit: $(cat "$REPO/.staged-commit" 2>/dev/null || echo UNKNOWN)"
    } >> "$LOG" 2>&1

    local OK=0 FAIL=0 SKIP=0
    ## Comments and blanks dropped first, so every worker numbers items alike.
    mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST")
    local k LINE ID ENVS ITEM_ENVS SCRIPT ARGS REST MARKER CLAIM T0 STATUS LEFT
    for ((k = 0; k < ${#LINES[@]}; k++)); do
        LINE="${LINES[$k]}"
        ID="${LINE%%|*}";      REST="${LINE#*|}"
        ENVS="${REST%%|*}";    REST="${REST#*|}"
        SCRIPT="${REST%%|*}";  ARGS="${REST#*|}"
        [ "$ENVS" = "-" ] && ENVS=""
        MARKER="$STATE_DIR/$ID.done"
        CLAIM="$STATE_DIR/$ID.claim"

        ## Cheap check first (one `test -f`, never a glob), then the atomic claim.
        [ -f "$MARKER" ] && { SKIP=$((SKIP + 1)); continue; }

        ## NEVER CLAIM AN ITEM THIS JOB CANNOT FINISH.  Without this a worker will happily
        ## claim an item 20 minutes before its job ends; Slurm then kills it mid-solve and the
        ## claim survives with no `.done`, so the work is lost AND the item is invisible to a
        ## resubmission until `collect_results.sh --reclaim` runs.  Stopping instead leaves the
        ## item UNCLAIMED for the next job, which is the idempotent behaviour the claim design
        ## already wants.  Raising ITEM_TIMEOUT alone does not fix this -- it widens the window.
        ## Budget is the item cap plus a margin for startup (torch/Drake/ikflow imports and, on
        ## a compiled run, ~35 s of dynamo) and for publishing results afterwards.
        LEFT="$(JobSecondsLeft)"
        if [ -n "$LEFT" ] && [ "$LEFT" -lt $(( ITEM_TIMEOUT + 300 )) ]; then
            echo "--- [$k] stopping: $LEFT s of job wall left, need $(( ITEM_TIMEOUT + 300 ));" \
                 "$((${#LINES[@]} - k)) item(s) left unclaimed for the next job" >> "$LOG" 2>&1
            break
        fi

        mkdir "$CLAIM" 2>/dev/null || { SKIP=$((SKIP + 1)); continue; }
        echo "$TAGID $(date -Is)" > "$CLAIM/owner"

        ## There is ONE Drake here and it is this project's pin, exported as PYTHONPATH
        ## above.  A per-item `DRAKE=nightly` sentinel used to select a second install so a
        ## stage could pair two Drake versions; that is gone by decision (2026-09-19, Thomas:
        ## every arm runs on the current pin, and a version-induced regression is reported and
        ## fixed upstream rather than pinned around).  Do not reintroduce a version switch.
        ITEM_ENVS="$ENVS"

        { echo "--- [$k] $ID START $(date -Is)"; echo "    env $ITEM_ENVS -- $SCRIPT $ARGS"; } \
            >> "$LOG" 2>&1
        T0=$SECONDS
        ## $ENVS and $ARGS are intentionally word-split; gen_manifest.py asserts
        ## that no token in either contains whitespace.
        # shellcheck disable=SC2086
        ## -k: SIGTERM first so Python can unwind, then SIGKILL 60 s later.  Without it a
        ## solve wedged inside a C++ solver ignores the TERM and holds the worker until
        ## Slurm arrives, which is the failure the item cap exists to prevent.
        env $ITEM_ENVS timeout -k 60 "$ITEM_TIMEOUT" "$PY" -u $SCRIPT $ARGS >> "$LOG" 2>&1
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            touch "$MARKER"; OK=$((OK + 1))
            ## Publish the item's output to $RESULTS_ROOT, which is what
            ## collect_results.sh tars. The benchmark writes under RepoDir(),
            ## i.e. $REPO/results, because that is how the scripts locate
            ## themselves; without this copy the collection point stays empty and
            ## a stage looks like it produced nothing. The item id IS the run's
            ## tag (gen_manifest.py passes the base tag and the shard suffix the
            ## script appends reproduces it), so one glob finds the directory
            ## whichever robot it belongs to.
            for RD in "$REPO"/results/*/benchmark/"$ID"; do
                [ -d "$RD" ] || continue
                REL="${RD#"$REPO"/results/}"
                mkdir -p "$RESULTS_ROOT/$(dirname "$REL")"
                cp -r "$RD" "$RESULTS_ROOT/$(dirname "$REL")/" 2>>"$LOG"
            done
        else
            FAIL=$((FAIL + 1))
        fi
        echo "--- [$k] $ID END status $STATUS after $((SECONDS - T0)) s" >> "$LOG" 2>&1
    done

    echo "=== worker $LOCAL finished $(date -Is): OK=$OK FAIL=$FAIL SKIP=$SKIP ===" >> "$LOG" 2>&1
    cp "$LOG" "$STATE_DIR/logs/${TAGID}.log"
    printf 'OK=%d FAIL=%d SKIP=%d worker=%d procs=%d host=%s gpu=%s finished=%s\n' \
        "$OK" "$FAIL" "$SKIP" "$LOCAL" "$PROCS" "$(hostname)" \
        "${CUDA_VISIBLE_DEVICES:-none}" "$(date -Is)" \
        > "$STATE_DIR/${TAGID}.SENTINEL"
    HOME="$REAL_HOME"
    [ $FAIL -eq 0 ]
}

echo "run_items: $PROCS worker(s) on $(hostname), manifest $MANIFEST_NAME, TMPDIR=$TMPDIR"
RC=0
## Wait on PIDs we collected ourselves, never on a bare `wait` and never on
## `jobs -p`. Both of those also pick up the `tee` of an `exec > >(tee ...)`
## redirection, which can never exit while the script holds its stdout -- see
## the WaitPids comment in calibrate.sh, where exactly that hung three GPU nodes.
## This script has no such redirection today; collecting PIDs keeps it correct
## if one is ever added.
WORKER_PIDS=()
for ((i = 0; i < PROCS; i++)); do Worker "$i" & WORKER_PIDS+=($!); done
for job in "${WORKER_PIDS[@]}"; do wait "$job" || RC=1; done
echo "run_items: done on $(hostname) at $(date -Is), rc=$RC"
exit $RC
