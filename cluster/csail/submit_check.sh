#!/bin/bash
# Submit the GPU plumbing check (check_job.sh): ONE preemptible job on a pinned GPU kind.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a CSAIL action, STOP and ask Thomas. Submissions need his clearance.
# ===========================================================================
#
#   GPU=rtx3090 bash cluster/csail/submit_check.sh <image name>
#
# Plan it prints: account locomotion, partition vision-shared-<gpu>, QoS shared-if-available, 1 GPU, 4 CPUs, 16G,
# 20 min, --requeue --open-mode=append (preemptible), array of 1. The HEAD commit must be staged (stage.sh), and
# the image anchored (build_image.sh record). Both are re-verified here, from the laptop, before the submit.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/csail_common.sh" || exit 1
REPO_ROOT="$(csail_repo_root)"
. "$REPO_ROOT/cluster/verify_staged_tree.sh"
. "$HERE/csail_verify.sh"; . "$HERE/csail_submit.sh"; . "$HERE/submit_lib.sh"
csail_connected || exit $?

IMGNAME="${1:?usage: GPU=<kind> submit_check.sh <image name>}"; GPU="${GPU:?set GPU=<rtx3090|rtx4090|l40s|a100|h200|csail-h200>}"
COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD); C12=${COMMIT:0:12}; ROOT="$(csail_tree_root "$C12")"
[ "$(git -C "$REPO_ROOT" status --porcelain | wc -l)" -eq 0 ] || { echo "REFUSING: the working tree is not clean" >&2; exit 6; }
IMGSHA=$(csail_image_sha "$IMGNAME") || exit 1
csail_laptop_check "$C12" || exit $?
csail_resolve_gpu "$GPU" || exit 2
NAME="lik_check_${C12:0:8}_$(date +%H%M%S)"
JOB=$(csail_sbatch "$ROOT" "$NAME" '%x-%j' \
    "LIK_ROOT=$ROOT LIK_CS_ROOT=$CS_ROOT EXPECT_MANIFEST_SHA=$EXPECT_MANIFEST_SHA IMAGE=$CS_ROOT/images/$IMGNAME.sif IMAGE_SHA256=$IMGSHA EXPECT_GPU=$EXPECT_GPU" \
    "$GPU_OPTS --gpus=1 --cpus-per-task=4 --mem=16G --time=00:20:00" \
    "$ROOT/repo/cluster/csail/check_job.sh") || exit 1
echo "submitted $NAME as job $JOB (log $ROOT/logs/$NAME-$JOB.out; sentinel $ROOT/check.$JOB.DONE)"
echo "JOB=$JOB"
