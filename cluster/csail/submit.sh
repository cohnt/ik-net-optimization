#!/bin/bash
# Submit a manifest of benchmark items (cluster/gen_manifest.py format) as ONE preemptible Slurm ARRAY on CSAIL's
# shared GPU partitions. Each array task is one GPU running one worker process (PROCS=1: one solve per GPU) and
# claims items dynamically (cluster/csail/job.sh), so any number of tasks, started whenever the queue frees a GPU,
# drain the manifest together, and re-submitting the same manifest is the mop-up.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a CSAIL action, STOP and ask Thomas. Submissions need his clearance (state the
# plan: account, partition, QoS, CPUs, memory, GPUs, walltime, count, name, preemptible). Never the lab node.
# ===========================================================================
#
#   GPU=l40s bash cluster/csail/submit.sh <manifest path in the repo> <image name> [tasks]
#
#   GPU            REQUIRED, the pinned GPU kind: rtx3090 rtx4090 l40s a100 h200 csail-h200. It sets the partition
#                  (vision-shared-<gpu>, -A locomotion; csail-h200 = -A csail -p csail-shared-h200), the
#                  --constraint (nvidia_l40s ...), and EXPECT_GPU, which each job checks against nvidia-smi. A
#                  CAMPAIGN NEVER MIXES GPU TYPES: the first submission writes state/<manifest>/HARDWARE and a
#                  later one with another kind is refused.
#   tasks          array size (default min(items, 8)).
#   MAX_RUNNING    how many array tasks run at once (default: tasks). tasks > MAX_RUNNING queues more task-days than
#                  run concurrently, so a long campaign drains without a resubmit (tasks that find nothing left exit
#                  0; a task that claims nothing while items remain exits 4, "NOTHING CLAIMED", flagged by poll.sh).
#   DEPENDENCY     optional sbatch --dependency, e.g. afterany:3212436 (a follow-on array that starts only when the
#                  earlier one has ended, so the footprint never exceeds one array's MAX_RUNNING).
#   TIME           default 1-00:00:00 (the partitions' limit). CPUS (default 8), MEM (default 48G).
#   ITEM_TIMEOUT   per-item backstop in s (default 28800); a task only claims an item that fits in its remaining time,
#                  so a submission with ITEM_TIMEOUT + 300 >= TIME is refused (a short smoke must lower it).
#   EXCLUDE_NODES  comma list added to the exclusions (the lab node and nodes with a recorded GPU fault are always in).
#   CPU model      Slurm has NO CPU constraint on these partitions (features are the GPU, the datacenter and the OS
#                  only), so CPU models can differ between nodes of one partition. Every attempt writes
#                  logs/<jobid>/machine.a<N>.txt (node, GPU, CPU model); read them before comparing wall clocks, and
#                  pin a node family with EXCLUDE_NODES if they differ.
#
# The HEAD commit must be staged (stage.sh), clean, and the image anchored (build_image.sh record). Both the
# staged tree and the image hash are checked from the laptop here, and again inside every job.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/csail_common.sh" || exit 1
REPO_ROOT="$(csail_repo_root)"
. "$REPO_ROOT/cluster/verify_staged_tree.sh"
. "$HERE/csail_verify.sh"; . "$HERE/csail_submit.sh"; . "$HERE/submit_lib.sh"
csail_connected || exit $?

MANIFEST="${1:?usage: GPU=<kind> submit.sh <manifest> <image name> [tasks]}"; IMGNAME="${2:?image name}"
GPU="${GPU:?set GPU=<rtx3090|rtx4090|l40s|a100|h200|csail-h200> (a campaign pins ONE GPU type)}"
[[ "$MANIFEST" =~ ^cluster/[A-Za-z0-9_./-]+\.txt$ ]] || { echo "manifest must be a repo-relative cluster/*.txt path" >&2; exit 2; }
git -C "$REPO_ROOT" ls-files --error-unmatch "$MANIFEST" >/dev/null 2>&1 || { echo "REFUSING: $MANIFEST is not tracked (stage only committed files)" >&2; exit 2; }
ITEMS=$(grep -vcE '^[[:space:]]*(#|$)' "$REPO_ROOT/$MANIFEST")
[ "$ITEMS" -gt 0 ] || { echo "no items in $MANIFEST" >&2; exit 2; }
TASKS="${3:-$(( ITEMS < 8 ? ITEMS : 8 ))}"
[[ "$TASKS" =~ ^[0-9]+$ ]] && [ "$TASKS" -ge 1 ] && [ "$TASKS" -le 64 ] || { echo "tasks must be 1..64" >&2; exit 2; }
MAX_RUNNING="${MAX_RUNNING:-$TASKS}"
[[ "$MAX_RUNNING" =~ ^[0-9]+$ ]] && [ "$MAX_RUNNING" -ge 1 ] && [ "$MAX_RUNNING" -le "$TASKS" ] || { echo "MAX_RUNNING must be 1..tasks ($TASKS)" >&2; exit 2; }
TIME="${TIME:-1-00:00:00}"; CPUS="${CPUS:-8}"; MEM="${MEM:-48G}"; ITEM_TIMEOUT="${ITEM_TIMEOUT:-28800}"
[[ "$TIME" =~ ^[0-9:-]+$ && "$CPUS" =~ ^[0-9]+$ && "$MEM" =~ ^[0-9]+[GM]$ && "$ITEM_TIMEOUT" =~ ^[0-9]+$ ]] || { echo "bad TIME/CPUS/MEM/ITEM_TIMEOUT" >&2; exit 2; }
## Slurm's --time forms: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S.
slurm_time_s() {
    local t="$1" d=0 h=0 m=0 s=0
    if   [[ "$t" =~ ^([0-9]+)-([0-9]+)(:([0-9]+)(:([0-9]+))?)?$ ]]; then d=${BASH_REMATCH[1]}; h=${BASH_REMATCH[2]}; m=${BASH_REMATCH[4]:-0}; s=${BASH_REMATCH[6]:-0}
    elif [[ "$t" =~ ^([0-9]+)$ ]];                                  then m=${BASH_REMATCH[1]}
    elif [[ "$t" =~ ^([0-9]+):([0-9]+)$ ]];                         then m=${BASH_REMATCH[1]}; s=${BASH_REMATCH[2]}
    elif [[ "$t" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]];                then h=${BASH_REMATCH[1]}; m=${BASH_REMATCH[2]}; s=${BASH_REMATCH[3]}
    else return 1; fi
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}
## Optional chaining: DEPENDENCY=afterany:<jobid>[:<jobid>...] (or after/afterok/afternotok/aftercorr, or singleton).
DEPENDENCY="${DEPENDENCY:-}"
[[ -z "$DEPENDENCY" || "$DEPENDENCY" =~ ^((after|afterany|afterok|afternotok|aftercorr)(:[0-9]+(_[0-9]+)?)+|singleton)$ ]] || {
    echo "bad DEPENDENCY '$DEPENDENCY' (e.g. afterany:3212436)" >&2; exit 2; }
DEP_OPT="${DEPENDENCY:+ --dependency=$DEPENDENCY}"
TIME_S=$(slurm_time_s "$TIME") || { echo "bad TIME '$TIME' (Slurm forms: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S)" >&2; exit 2; }
## A task claims an item only with ITEM_TIMEOUT + 300 s of walltime left (run_items.sh); below that it claims
## NOTHING and exits 0 (smoke 3211413). Refuse the submission instead.
if [ $(( ITEM_TIMEOUT + 300 )) -ge "$TIME_S" ]; then
    echo "REFUSING: ITEM_TIMEOUT + 300 = $(( ITEM_TIMEOUT + 300 )) s >= TIME $TIME = $TIME_S s: no task could claim an item." \
         "Lower ITEM_TIMEOUT or raise TIME." >&2
    exit 2
fi
## The image's Drake 1.58.0 has CalcRobotClearances (DRAKE_ENV=crc is retired everywhere).
if grep -vE '^[[:space:]]*(#|$)' "$REPO_ROOT/$MANIFEST" | grep -qE '(^|[ |])DRAKE=nightly'; then echo "REFUSING: a per-item DRAKE= selector (one Drake per job)" >&2; exit 2; fi

COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD); C12=${COMMIT:0:12}; ROOT="$(csail_tree_root "$C12")"
[ "$(git -C "$REPO_ROOT" status --porcelain | wc -l)" -eq 0 ] || { echo "REFUSING: the working tree is not clean" >&2; exit 6; }
IMGSHA=$(csail_image_sha "$IMGNAME") || exit 1
csail_laptop_check "$C12" || exit $?
csail_resolve_gpu "$GPU" || exit 2

CKPTS=$(grep -vE '^[[:space:]]*(#|$)' "$REPO_ROOT/$MANIFEST" | grep -o -- '--checkpoint[= ][^ ]*' | sed 's/^--checkpoint[= ]//' | sort -u | tr '\n' ' ')
for c in $CKPTS; do
    cs_run "test -f $ROOT/repo/$c" || { echo "REFUSING: checkpoint $c is not in the staged tree" >&2; exit 3; }
done

## One campaign, one hardware: the first submission records it, a different kind is refused.
STNAME="$(basename "$MANIFEST" .txt)"
HW=$(cs_run "cat $ROOT/state/$STNAME/HARDWARE 2>/dev/null" || true)
if [ -n "$HW" ] && [ "$HW" != "$GPU" ]; then
    echo "REFUSING: $STNAME already runs on GPU kind '$HW'; a campaign never mixes GPU types (asked for '$GPU')." >&2; exit 5
fi

NAME="lik_${STNAME:0:24}_$(date +%H%M%S)"
echo "PLAN: array 0-$((TASKS - 1))%$MAX_RUNNING of $NAME, $ITEMS item(s); $GPU_OPTS --gpus=1 --cpus-per-task=$CPUS --mem=$MEM --time=$TIME$DEP_OPT; one solve per GPU; preemptible"
cs_run "mkdir -p $ROOT/state/$STNAME && [ -s $ROOT/state/$STNAME/HARDWARE ] || echo $GPU > $ROOT/state/$STNAME/HARDWARE" || exit 8
JOB=$(csail_sbatch "$ROOT" "$NAME" '%x-%A_%a' \
    "LIK_ROOT=$ROOT LIK_CS_ROOT=$CS_ROOT EXPECT_MANIFEST_SHA=$EXPECT_MANIFEST_SHA IMAGE=$CS_ROOT/images/$IMGNAME.sif IMAGE_SHA256=$IMGSHA EXPECT_GPU=$EXPECT_GPU MANIFEST=$MANIFEST PROCS=1 ITEM_TIMEOUT=$ITEM_TIMEOUT" \
    "$GPU_OPTS --gpus=1 --cpus-per-task=$CPUS --mem=$MEM --time=$TIME --array=0-$((TASKS - 1))%$MAX_RUNNING$DEP_OPT" \
    "$ROOT/repo/cluster/csail/job.sh") || exit 1
echo "submitted $NAME as array job $JOB (state $ROOT/state/$STNAME, results $ROOT/results, logs $ROOT/logs/)"
echo "JOB=$JOB ROOT=$ROOT"
