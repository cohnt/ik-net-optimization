#!/bin/bash
# The LLsub payload that drains a manifest of benchmark runs on one GPU node.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit (from ~/learned-ik/repo on the login node), one job per node:
#   MANIFEST=cluster/manifest_stageA.txt PROCS=1 \
#     LLsub ./cluster/run_items.sh -g volta:2 -s 40 -q xeon-g6-volta -T 12:00:00
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

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="${TEST_REPO:-$ROOT/repo}"
STATE_ROOT="${TEST_STATE_DIR:-$ROOT/state}"
RESULTS_ROOT="${LEARNED_IK_RESULTS:-$ROOT/results}"
PROCS="${PROCS:-1}"
ITEM_TIMEOUT="${ITEM_TIMEOUT:-14400}"
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

    export LD_LIBRARY_PATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages"
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
        export CUDA_VISIBLE_DEVICES=$(( LOCAL % 2 ))
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
    local k LINE ID ENVS SCRIPT ARGS REST MARKER CLAIM T0 STATUS
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
        mkdir "$CLAIM" 2>/dev/null || { SKIP=$((SKIP + 1)); continue; }
        echo "$TAGID $(date -Is)" > "$CLAIM/owner"

        { echo "--- [$k] $ID START $(date -Is)"; echo "    env $ENVS -- $SCRIPT $ARGS"; } \
            >> "$LOG" 2>&1
        T0=$SECONDS
        ## $ENVS and $ARGS are intentionally word-split; gen_manifest.py asserts
        ## that no token in either contains whitespace.
        # shellcheck disable=SC2086
        env $ENVS timeout "$ITEM_TIMEOUT" "$PY" -u $SCRIPT $ARGS >> "$LOG" 2>&1
        STATUS=$?
        if [ $STATUS -eq 0 ]; then
            touch "$MARKER"; OK=$((OK + 1))
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
for ((i = 0; i < PROCS; i++)); do Worker "$i" & done
for job in $(jobs -p); do wait "$job" || RC=1; done
echo "run_items: done on $(hostname) at $(date -Is), rc=$RC"
exit $RC
