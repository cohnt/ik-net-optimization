#!/bin/bash
# Submit jobs to MIT ORCD Engaging (laptop side). One entry point, one subcommand per job kind:
#
#   bash cluster/engaging/submit.sh build                 the image, on mit_normal (CPU), from the committed lik.def
#   bash cluster/engaging/submit.sh record-sif            read the built image's sha256 into the laptop's anchor file
#   bash cluster/engaging/submit.sh plumb                 GPU plumbing check + cache warm-up (1 GPU)
#   bash cluster/engaging/submit.sh plumbcpu              the same without the GPU block, on mit_quicktest (CPU, 15 min)
#   bash cluster/engaging/submit.sh bench <manifest>      a manifest of benchmark items, one array task per GPU
#   bash cluster/engaging/submit.sh survey                read-only: GPU nodes per type, what is queued (mine)
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about an Engaging action, STOP and ask Thomas. Uses the skill's one existing
# connection only; never logs in. Before EVERY submit the workstation re-verifies the staged tree
# (eg_check_commit: the cluster's own sha256sum against the laptop manifest) and hands the manifest's hash
# to the job, which re-checks (job_lib.sh). Never `sbatch` in a loop: arrays only. Jobs are named lik_*.
# ===========================================================================
#
# HARDWARE PIN (a campaign never mixes hardware; the runbook says why):
#   GPU=l40s       the GPU type, `-G <GPU>:1` per task (default l40s; H200 only if needed; NO B200: the image's
#                  torch is cu126, which has no sm_100)
#   NODES=...      the CPU pin: nodes of ONE CPU model, comma-separated `nodeNNNN` or `nodeAAAA-BBBB`, turned into
#                  Slurm's node-name OR-constraint (skill job-patterns.md: node features are only the node name)
#   CPU_MODEL=...  a substring of lscpu's "Model name" (e.g. "EPYC 9474F"); every job REFUSES to run elsewhere
#   ANY_CPU=1      explicit opt-out of the CPU pin (smoke / plumbing only; never a campaign)
# Both are recorded in the job's log, in state/<manifest>/hardware.log (every job), and in
# state/<manifest>/submissions.log (every submit, with the commit, partition and allotment).
#
# ALLOTMENT (every value printed before the submit; override with env):
#   PARTITION  mit_normal_gpu (default; 2 GPUs at once, 6 h) | mit_preemptable (4 GPUs, 48 h, --requeue)
#   NTASKS     array tasks = GPUs (default 2 / 4, capped at the partition's per-user GPU limit); ONE SOLVE PER GPU
#   WALL       default 06:00:00 (normal) / 48:00:00 (preemptable);  CPUS 4 per task;  MEM 48G per task
#   COMMIT     the staged commit (default: the last one stage.sh staged)
set -uo pipefail
source "$(dirname "$0")/eg_common.sh" || exit 1

KIND="${1:?usage: submit.sh build|record-sif|plumb|bench <manifest>|survey}"
SDIR="$EG_REPO/logs/staged"
COMMIT="${COMMIT:-$(cat "$SDIR/engaging-last-commit" 2>/dev/null)}"
GPU="${GPU:-l40s}"
PARTITION="${PARTITION:-mit_normal_gpu}"
CPUS="${CPUS:-4}"; MEM="${MEM:-48G}"
NODES="${NODES:-}"; CPU_MODEL="${CPU_MODEL:-}"; ANY_CPU="${ANY_CPU:-0}"

die() { echo "REFUSING: $*" >&2; exit 4; }
eg_require_connection || exit 255

## expand "node4100-4103,node4200" -> "node4100|node4101|node4102|node4103|node4200"
expand_nodes() {
    local t out=() lo hi pre i; local IFS=,
    for t in $1; do
        if [[ $t =~ ^([A-Za-z]+)([0-9]+)-([0-9]+)$ ]]; then
            pre=${BASH_REMATCH[1]}; lo=${BASH_REMATCH[2]}; hi=${BASH_REMATCH[3]}
            for ((i = lo; i <= hi; i++)); do out+=("$pre$i"); done
        elif [[ $t =~ ^[A-Za-z]+[0-9]+(-c[0-9]+)?$ ]]; then out+=("$t")
        else echo "bad node token '$t'" >&2; return 1; fi
    done
    local IFS='|'; echo "${out[*]}"
}

case "$GPU" in
    l40s) GPU_NAME="L40S" ;; h200) GPU_NAME="H200" ;; a100) GPU_NAME="A100" ;; h100) GPU_NAME="H100" ;;
    b200) die "GPU=b200: the image's torch is cu126 (no sm_100)" ;;
    *) die "GPU='$GPU' is not one of l40s h200 a100 h100" ;;
esac

case "$KIND" in
survey)
    eg_run "date; echo '--- my queue ---'; squeue --me -o '%.12i %.9P %.16j %.2t %.10M %R'
echo '--- GPU nodes by type (partition $PARTITION; CPU model is learned from a plumb job, see hardware.log) ---'
sinfo -h -p $PARTITION -N -o '%N %G %c %t' | grep -i '$GPU' | sort | awk '{print}' | head -60
echo '--- QOS limits ---'; sacctmgr -n show qos $PARTITION format=Name%18,MaxTRESPU%50,MaxSubmitPU%8,MaxWall
echo '--- quota ---'; sed -n 3,6p ~/orcd/.quota"
    exit $? ;;
record-sif)
    OUT=$(eg_run "cat ~/$EG_BASE/build.DONE; cat ~/$EG_BASE/image/lik.sif.commit 2>/dev/null") || exit 1
    SHA=$(printf '%s\n' "$OUT" | sed -n 's/^OK \([0-9a-f]\{64\}\)$/\1/p' | head -1)
    [ -n "$SHA" ] || die "build.DONE does not read 'OK <sha256>' (got: $(printf '%s' "$OUT" | head -c 200))"
    mkdir -p "$SDIR"; echo "$SHA" > "$SDIR/engaging-sif.sha256"
    echo "anchored image sha256 $SHA in $SDIR/engaging-sif.sha256"; exit 0 ;;
build|plumb|plumbcpu|bench) ;;
*) die "unknown subcommand '$KIND'" ;;
esac

[ -n "$COMMIT" ] || die "no staged commit: run cluster/engaging/stage.sh first"
[ -s "$SDIR/engaging-$COMMIT.manifest-sha" ] || die "commit $COMMIT was not staged from this workstation"
CROOT=$(eg_commit_root "$COMMIT")

## The staged tree must match the laptop's manifest RIGHT NOW (exports EXPECT_MANIFEST_SHA).
eg_check_commit "$COMMIT" || exit $?

## ---- per-kind allotment ----------------------------------------------------------------------
case "$KIND" in
build)
    ## A CPU job: no GPU pin (the in-job hardware check would refuse a node with no nvidia-smi).
    ## BUILD_PARTITION=mit_preemptable (idempotent: a preemption only costs a rerun; the job builds to a temp
    ## name and renames, and build.DONE is written last) adds --requeue.
    BUILD_PARTITION="${BUILD_PARTITION:-mit_normal}"
    case "$BUILD_PARTITION" in mit_normal|mit_preemptable) ;; *) die "BUILD_PARTITION='$BUILD_PARTITION' is not mit_normal or mit_preemptable" ;; esac
    WALL=01:00:00; GPU_NAME=""; JOB=lik_build; SB=(-p "$BUILD_PARTITION" -c 8 --mem=32G -t "$WALL"); NEEDIMG=0; SCRIPT=build_image_job.sh; ARRAY=()
    [ "$BUILD_PARTITION" != mit_preemptable ] || SB+=(--requeue) ;;
plumbcpu)
    ## No GPU: validates the image, the bind/home mounts, the staged tree inside the container and warms the caches.
    WALL=00:15:00; GPU_NAME=""; JOB=lik_plumbcpu; SB=(-p mit_quicktest -c 4 --mem=16G -t "$WALL"); NEEDIMG=1
    SCRIPT=plumb_job.sh; ARRAY=() ;;
plumb|bench)
    case "$PARTITION" in
        mit_normal_gpu)  CAP=2; DEFWALL=06:00:00; REQ=() ;;
        mit_preemptable) CAP=4; DEFWALL=48:00:00; REQ=(--requeue) ;;
        *) die "PARTITION='$PARTITION' is not mit_normal_gpu or mit_preemptable" ;;
    esac
    WALL="${WALL:-$DEFWALL}"; NTASKS="${NTASKS:-$CAP}"
    [ "$KIND" = plumb ] && NTASKS=1
    [ "$NTASKS" -le "$CAP" ] || die "NTASKS=$NTASKS exceeds $PARTITION's per-user GPU limit $CAP (one solve per GPU)"
    if [ "$ANY_CPU" != 1 ]; then
        [ -n "$NODES" ] && [ -n "$CPU_MODEL" ] || die "pin the CPU: NODES=... and CPU_MODEL='...' (or ANY_CPU=1 for a smoke test, never a campaign)"
    fi
    CONSTRAINT=""; [ -z "$NODES" ] || CONSTRAINT=$(expand_nodes "$NODES") || exit 4
    SB=(-p "$PARTITION" -G "$GPU:1" -c "$CPUS" --mem="$MEM" -t "$WALL" "${REQ[@]}")
    [ -z "$CONSTRAINT" ] || SB+=(--constraint="$CONSTRAINT")
    NEEDIMG=1
    if [ "$KIND" = plumb ]; then JOB=lik_plumb; SCRIPT=plumb_job.sh; ARRAY=()
    else
        MANIFEST="${2:?usage: submit.sh bench <manifest path relative to the repo, e.g. cluster/engaging/manifest_engaging_smoke.txt>}"
        [ -f "$EG_REPO/$MANIFEST" ] || die "no such local manifest: $MANIFEST"
        eg_run "test -f ~/$CROOT/repo/$MANIFEST" || die "$MANIFEST is not in the staged tree of $COMMIT"
        for c in $(grep -vE '^[[:space:]]*(#|$)' "$EG_REPO/$MANIFEST" | grep -o -- '--checkpoint[= ][^ ]*' | sed 's/^--checkpoint[= ]//' | sort -u); do
            eg_run "test -f ~/$CROOT/repo/$c" || die "$MANIFEST references a checkpoint that is not on the cluster: $c"
        done
        JOB="lik_bench_$(basename "$MANIFEST" .txt | sed 's/^manifest_//' | cut -c1-40)"; SCRIPT=bench_job.sh
        ARRAY=(--array="0-$((NTASKS - 1))%$NTASKS")
    fi ;;
esac

SECS=$(awk -F'[-:]' -v w="$WALL" 'BEGIN{n=split(w,a,/[-:]/); if(n==4)s=a[1]*86400+a[2]*3600+a[3]*60+a[4]; else s=a[1]*3600+a[2]*60+a[3]; print s}' 2>/dev/null)
ITEM_TIMEOUT=28800
if [ "$KIND" != build ] && [ -n "${SECS:-}" ] && [ "$SECS" -lt $((ITEM_TIMEOUT + 1500)) ]; then ITEM_TIMEOUT=$((SECS - 1500)); fi

ENV=(LIK_CROOT="$CROOT" EXPECT_MANIFEST_SHA="$EXPECT_MANIFEST_SHA" ITEM_TIMEOUT="$ITEM_TIMEOUT"
     LIK_EXPECT_GPU="${GPU_NAME:-}" LIK_EXPECT_CPU="$CPU_MODEL")
[ "${NEEDIMG:-0}" = 0 ] || {
    [ -s "$SDIR/engaging-sif.sha256" ] || die "no anchored image hash: build, then submit.sh record-sif"
    ENV+=(EXPECT_SIF_SHA="$(cat "$SDIR/engaging-sif.sha256")"); }
[ "$KIND" != bench ] || ENV+=(MANIFEST="$MANIFEST")
[ "$KIND" != plumbcpu ] || ENV+=(PLUMB_CPU=1)
[ "${PLUMB_DIAG:-0}" != 1 ] || ENV+=(PLUMB_DIAG=1)   # plumbcpu only: probe the apptainer mount spellings, then exit
[ -z "${DRAKE_TARBALL:-}" ] || ENV+=(DRAKE_TARBALL="$DRAKE_TARBALL")
ENVSTR=$(printf '%q ' "${ENV[@]}")

SBATCH=(sbatch --parsable -J "$JOB" --open-mode=append -o "logs/%x-%A${ARRAY:+_%a}.out" "${SB[@]}" ${ARRAY[@]+"${ARRAY[@]}"} "repo/cluster/engaging/$SCRIPT")
echo "== $KIND | job $JOB | commit $COMMIT | ${SB[*]} ${ARRAY[*]:-} | ITEM_TIMEOUT $ITEM_TIMEOUT"
echo "== hardware pin: gpu=${GPU_NAME:-n/a} cpu_model='${CPU_MODEL:-ANY}' nodes='${NODES:-any}'"
REC="$(date -Is) $KIND $JOB commit=$COMMIT partition=${PARTITION:-mit_normal} gpu=${GPU:-} cpu_model='$CPU_MODEL' nodes='$NODES' ntasks=${NTASKS:-1} wall=${WALL:-} manifest=${MANIFEST:-}"
JID=$(eg_run "cd ~/$CROOT && mkdir -p logs state && echo $(printf '%q' "$REC") >> state/submissions.log && $ENVSTR $(printf '%q ' "${SBATCH[@]}")") || die "sbatch failed: $JID"
echo "submitted job $JID ($JOB)"
echo "$JID" > "$SDIR/engaging-last-job-$KIND"
