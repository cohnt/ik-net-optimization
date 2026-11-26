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
    lik_exclude_from_array "$(hostname -s)"
    exit 3
}
## THE BLACK HOLE: a node whose GPU fails in seconds frees its slot at once, Slurm puts the array's next pending
## task there, and a throttled array drains onto it (OBST 3212436: 46 of 48 tasks in a minute on torralba-3090-3).
## So the failing task adds its node to its OWN array's ExcNodeList, which every still-pending task inherits.
## Fail-safe: any trouble here is logged and the task exits 3 as before (bad_nodes.log still has the node).
lik_exclude_from_array() {
    local node="$1" arr="${SLURM_ARRAY_JOB_ID:-}" cur new
    [ -n "$arr" ] || { echo "not an array job: no ExcNodeList update"; return 0; }
    cur=$(scontrol show job -o "$arr" 2>/dev/null | grep -o 'ExcNodeList=[^ ]*' | head -1 | cut -d= -f2)
    case "$cur" in "(null)") cur="" ;; esac
    if [ -n "$cur" ] && scontrol show hostnames "$cur" 2>/dev/null | grep -qx "$node"; then
        echo "array $arr already excludes $node"; return 0
    fi
    new="${cur:+$cur,}$node"
    if scontrol update JobId="$arr" ExcNodeList="$new" 2>&1 | sed 's/^/scontrol: /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
        echo "excluded $node from array $arr's pending tasks (ExcNodeList=$new)"
    else
        echo "scontrol update of array $arr's ExcNodeList to $new returned nonzero (logged only: running tasks" \
             "refuse it, pending ones may still have taken it; bad_nodes.log has $node either way)"
    fi
    return 0
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

## The epoch this attempt's walltime ends, read from the host's squeue (the container has none; its shim answers
## from JOB_END_EPOCH). An array task is asked for as <array>_<task>: the highest task carries the array MASTER id
## as its SLURM_JOB_ID, and `squeue -j <master>` lists EVERY task of the array, one line each; glued together they
## parsed to nothing and that task claimed items with no limit (mobile-base 3211402_1, 2026-11-26). Anything but
## exactly one well-formed time is fatal: a job never runs without knowing its end.
lik_end_epoch() {
    local id="${SLURM_JOB_ID:-}" out="" n=0 L try
    [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    [ -n "$id" ] || lik_no_end "no SLURM_JOB_ID"
    for try in 1 2 3; do   # a busy slurmctld can fail one query
        out=$(squeue -h -j "$id" -o %L 2>&1); n=$(printf '%s\n' "$out" | grep -c .)
        L=$(printf '%s' "$out" | tr -d '[:space:]')
        [ "$n" -eq 1 ] && [[ "$L" =~ ^([0-9]+-)?[0-9]+(:[0-9]+){0,2}$ ]] && break
        L=""; [ "$try" -eq 3 ] || sleep "${LIK_END_RETRY_S:-10}"
    done
    [ -n "$L" ] || lik_no_end "squeue -h -j $id -o %L answered $n line(s): '$(printf '%s' "$out" | tr '\n' ' ' | cut -c1-200)'"
    JOB_END_EPOCH=""
    local D=0 A B C; case "$L" in *-*) D="${L%%-*}"; L="${L#*-}" ;; esac
    IFS=: read -r A B C <<< "$L"
    case "$L" in *:*:*) : ;; *:*) C="$B"; B="$A"; A=0 ;; *) C="$A"; B=0; A=0 ;; esac
    JOB_END_EPOCH=$(( $(date +%s) + 10#${D:-0} * 86400 + 10#${A:-0} * 3600 + 10#${B:-0} * 60 + 10#${C:-0} ))
    export JOB_END_EPOCH
    echo "job end: $(date -Is -d "@$JOB_END_EPOCH") (squeue -j $id: $(printf '%s' "$out" | tr -d '[:space:]') left)"
}
lik_no_end() {
    echo "CANNOT DETERMINE JOB END TIME: $1. Refusing to claim items without knowing the walltime (exit 5)."
    exit 5
}
