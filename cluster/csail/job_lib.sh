#!/bin/bash
# Host-side helpers sourced by cluster/csail/job.sh and check_job.sh (they run ON A COMPUTE NODE, from the
# verified staged tree). Never sourced on the laptop.
#
# Expects, in front of sbatch (never --export): LIK_ROOT LIK_CS_ROOT EXPECT_MANIFEST_SHA IMAGE IMAGE_SHA256
# and optionally EXPECT_GPU (regex on the device name).

lik_need() { local v; for v in "$@"; do [ -n "${!v:-}" ] || { echo "job_lib: $v is not set (submit through cluster/csail/submit*.sh)"; exit 2; }; done; }

## The code check, the image check (sha256 recorded at build, anchored on the laptop), a per-attempt TMPDIR.
lik_start() {
    lik_need LIK_ROOT LIK_CS_ROOT EXPECT_MANIFEST_SHA IMAGE IMAGE_SHA256
    REPO="$LIK_ROOT/repo"
    ATTEMPT="${SLURM_RESTART_COUNT:-0}"
    JOBTAG="${SLURM_JOB_ID:-nojob}"
    LOGD="$LIK_ROOT/logs/$JOBTAG"; mkdir -p "$LOGD"
    export TMPDIR="${LIK_TMP_BASE:-/tmp}/lik_${JOBTAG}_a${ATTEMPT}"; mkdir -p "$TMPDIR"   # node-local: caches, worker logs
    . "$REPO/cluster/verify_staged_tree.sh" || exit 1
    lik_job_check "$REPO"                          # the tree == the laptop's manifest, fresh PYTHONPYCACHEPREFIX
    local have; have=$(sha256sum "$IMAGE" | cut -d' ' -f1)
    if [ "$have" != "$IMAGE_SHA256" ]; then
        echo "STOP: the image $IMAGE hashes to $have, the laptop anchored $IMAGE_SHA256 -- tell Thomas"; exit 1
    fi
    echo "image verified: $(basename "$IMAGE") $have"
}

## One file per attempt: where this ran and on what, so a campaign's hardware is a recorded fact.
lik_machine_record() {
    {
        echo "job $JOBTAG array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} attempt $ATTEMPT at $(date -Is)"
        echo "host $(hostname) partition ${SLURM_JOB_PARTITION:-?} nodelist ${SLURM_JOB_NODELIST:-?}"
        echo "commit $(cat "$REPO/.staged-commit" 2>/dev/null) image $(basename "$IMAGE") sha256 $IMAGE_SHA256"
        echo "cpu: $(lscpu | sed -n 's/^Model name: *//p' | head -1), $(nproc) usable cpus, $(lscpu | sed -n 's/^Socket(s): *//p') socket(s)"
        echo "kernel $(uname -r); apptainer $(apptainer --version 2>&1)"
        nvidia-smi --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader 2>&1 | sed 's/^/gpu: /'
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} EXPECT_GPU=${EXPECT_GPU:-unset}"
    } | tee "$LOGD/machine.a$ATTEMPT.txt"
}

## nvidia-smi on the host, then the in-image check; a fault exits non-zero and names the node for --exclude.
lik_gpu_check() {
    nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv || { lik_gpu_fault "nvidia-smi failed"; }
    lik_container /opt/lik/venv/bin/python "$REPO/cluster/csail/gpu_check.py" "$@" || lik_gpu_fault "in-image gpu_check failed"
}
lik_gpu_fault() {
    echo "GPU FAULT on $(hostname): $1"
    echo "$(date -Is) $(hostname -s) job $JOBTAG: $1" >> "$LIK_CS_ROOT/bad_nodes.log"   # -s: the Slurm node name
    exit 3
}

## apptainer exec with a clean environment: only what the payload needs crosses in. --nv for the GPU, no HOME
## (AFS is invisible to jobs). The shim directory first on PATH answers run_items.sh's squeue time-left query.
lik_container() {
    local envs=(--env "PATH=$REPO/cluster/csail/shim:/opt/lik/venv/bin:/usr/local/bin:/usr/bin:/bin"
                --env "TMPDIR=$TMPDIR" --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
                --env "EXPECT_MANIFEST_SHA=$EXPECT_MANIFEST_SHA" --env "LEARNED_IK_ROOT=$LIK_ROOT"
                --env "SLURM_JOB_ID=${SLURM_JOB_ID:-}" --env "SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
                --env "JOB_END_EPOCH=${JOB_END_EPOCH:-}" --env "TEST_PYTHON=/opt/lik/venv/bin/python"
                --env "EXPECT_GPU=${EXPECT_GPU:-}")
    local v
    for v in MANIFEST PROCS MPS ITEM_TIMEOUT OVERRUN_FACTOR MAX_OVERRUNS DEVICE; do
        [ -z "${!v:-}" ] || envs+=(--env "$v=${!v}")
    done
    apptainer exec --nv --cleanenv --no-home --bind "$LIK_CS_ROOT" --bind /tmp "${envs[@]}" "$IMAGE" "$@"
}

## Seconds this attempt has left, read from the host's squeue (the container has none).
lik_end_epoch() {
    local L; L=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | tr -d '[:space:]')
    JOB_END_EPOCH=""
    case "$L" in ""|UNLIMITED|NOT_SET|INVALID) return 0 ;; esac
    local D=0 A B C; case "$L" in *-*) D="${L%%-*}"; L="${L#*-}" ;; esac
    IFS=: read -r A B C <<< "$L"
    case "$L" in *:*:*) : ;; *:*) C="$B"; B="$A"; A=0 ;; *) C="$A"; B=0; A=0 ;; esac
    JOB_END_EPOCH=$(( $(date +%s) + 10#${D:-0} * 86400 + 10#${A:-0} * 3600 + 10#${B:-0} * 60 + 10#${C:-0} ))
    export JOB_END_EPOCH
}
