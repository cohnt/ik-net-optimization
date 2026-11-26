#!/bin/bash
# Common prologue of every Engaging job (sourced by the *_job.sh payloads from the STAGED tree).
#
#   LIK_CROOT=~/lik-engaging/c-<12hex>  EXPECT_MANIFEST_SHA=...  [EXPECT_SIF_SHA=...]  sbatch .../X_job.sh
#   (cluster/engaging/submit.sh sets all of these; a hand-submitted job refuses.)
#
# eg_job_init does, in order: the Slurm-node environment (apptainer module, host loader/compiler
# variables unset), the code-integrity check of the staged tree (verify_staged_tree.sh, the SAME code as
# SuperCloud's: refuses unless the tree's manifest has the hash the workstation anchored and sha256sum -c
# passes; PYTHONPYCACHEPREFIX fresh), the image's sha256 against the value recorded at build (when the
# job uses the image), a private /tmp dir, and the HARDWARE RECORD + pin check.
#
# HARDWARE PIN. A campaign must run every arm on identical hardware. The submitter pins the GPU type
# (`-G l40s:1`) and the CPU model (a node-name constraint); this prologue records host, GPU and CPU model
# of every job to state/<manifest>/hardware.log and the job log, and REFUSES to run if the node's GPU name
# lacks LIK_EXPECT_GPU or its CPU model lacks LIK_EXPECT_CPU (when set). Never mix hardware in a campaign.

[ -n "${LIK_CROOT:-}" ] || { echo "job_lib.sh: LIK_CROOT unset (submit through cluster/engaging/submit.sh)" >&2; exit 2; }
REPO="$LIK_CROOT/repo"
EB="$(dirname "$LIK_CROOT")"

eg_job_init() {
    ## /etc/profile BEFORE `set -u` (the cluster's profile.d scripts read unset variables), then strict mode.
    set +u; source /etc/profile 2>/dev/null; set -uo pipefail
    module load apptainer/1.4.2 2>/dev/null || module load apptainer 2>/dev/null || true
    command -v apptainer >/dev/null || { echo "no apptainer on $(hostname)" >&2; exit 2; }
    ## The login shell's gcc module exports CC/CXX etc.; Apptainer passes the host environment in.
    unset CC CXX CPATH LIBRARY_PATH LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
    ## Per-JOB private /tmp (the node's /tmp is shared by every job: skill trap 3).
    export TMPDIR="/tmp/$USER-${SLURM_JOB_ID:-nojob}"
    mkdir -p -m 700 "$TMPDIR"
    trap 'declare -F lik_cleanup >/dev/null && lik_cleanup; rm -rf "$TMPDIR"' EXIT

    # shellcheck disable=SC1091
    source "$REPO/cluster/verify_staged_tree.sh" || exit 1
    lik_job_check "$REPO"
    echo "job: ${SLURM_JOB_NAME:-?} id ${SLURM_JOB_ID:-?} restart ${SLURM_RESTART_COUNT:-0} on $(hostname), commit $(cat "$REPO/.staged-commit" 2>/dev/null || echo UNKNOWN)"

    LIK_SIF="$EB/image/lik.sif"
    if [ "${LIK_NEED_IMAGE:-1}" = 1 ]; then
        [ -n "${EXPECT_SIF_SHA:-}" ] || { echo "EXPECT_SIF_SHA unset: the image's sha256 is anchored by the submitter" >&2; exit 1; }
        have=$(sha256sum "$LIK_SIF" | cut -d' ' -f1)
        [ "$have" = "$EXPECT_SIF_SHA" ] || { echo "STOP: image $LIK_SIF is $have, expected $EXPECT_SIF_SHA -- tell Thomas" >&2; exit 1; }
        echo "image verified: $have"
    fi
    ## LIK_BIND is the LOGICAL path (/home/tcohn/...), never `readlink -f`: on a compute node that resolves to
    ## /orcd/home/002/tcohn/..., and since py_in_image.sh overrides --home (so apptainer no longer auto-binds
    ## /home/tcohn), binding the real path leaves every logical path we use invisible (2026-11-26 diagnostic).
    export LIK_SIF LIK_BIND="$EB" LIK_REPO="$REPO"

    eg_hardware_check
}

## Record and pin the hardware. $1 = a state dir to append hardware.log to (optional).
eg_hardware_check() {
    local gpu cpu line
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | paste -sd, -)
    cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
    line="HARDWARE host=$(hostname) gpu=[${gpu:-none}] cpu=[$cpu] job=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-} expect_gpu=[${LIK_EXPECT_GPU:-}] expect_cpu=[${LIK_EXPECT_CPU:-}] $(date -Is)"
    echo "$line"
    LIK_HARDWARE_LINE="$line"; export LIK_HARDWARE_LINE
    if [ -n "${LIK_EXPECT_GPU:-}" ]; then
        case "$gpu" in *"$LIK_EXPECT_GPU"*) ;; *) echo "STOP: GPU [$gpu] is not the campaign's pinned [$LIK_EXPECT_GPU]; a campaign never mixes hardware" >&2; exit 1 ;; esac
    fi
    if [ -n "${LIK_EXPECT_CPU:-}" ]; then
        case "$cpu" in *"$LIK_EXPECT_CPU"*) ;; *) echo "STOP: CPU [$cpu] is not the campaign's pinned [$LIK_EXPECT_CPU]; a campaign never mixes hardware" >&2; exit 1 ;; esac
    fi
}

## Release the claims left by THIS JOB'S PREVIOUS ATTEMPT. THE REQUEUE TRAP: run_items.sh claims an item
## with `mkdir <id>.claim`; a preempted job is requeued under the SAME job id, restarts from the top of
## the script, and its dead attempt's claims (no .done) would otherwise orphan those items: every
## worker, this one included, sees "claimed" and skips. run_items.sh writes the Slurm job id as the 3rd field of each
## claim's owner file (shared with cluster/csail/job.sh), so the claims of this job id are exactly the dead
## attempt's (a requeued job's old processes are gone before it restarts). Other jobs' claims are never
## touched; a claim from a job that died WITHOUT a requeue is cluster/engaging/collect.sh --reclaim's.
## $1 = the manifest's state dir.
eg_release_own_claims() {
    local state="$1" c id n=0
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    for c in "$state"/*.claim; do
        [ -d "$c" ] || continue
        id=$(basename "$c" .claim)
        [ -f "$state/$id.done" ] && continue
        if [ "$(awk '{print $3}' "$c/owner" 2>/dev/null)" = "$SLURM_JOB_ID" ]; then
            rm -rf "$c"; n=$((n + 1)); echo "released stale claim of this job's previous attempt: $id"
        fi
    done
    echo "claims released at (re)start of job ${SLURM_JOB_ID} (restart ${SLURM_RESTART_COUNT:-0}): $n"
}
