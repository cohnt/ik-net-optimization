#!/bin/bash
# Build the learned-ik Apptainer image ON CSAIL, as a CPU job (cluster/csail/build_image_job.sh).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a CSAIL action, STOP and ask Thomas. Submissions need his clearance.
# ===========================================================================
#
#   bash cluster/csail/build_image.sh submit            # needs the HEAD commit staged (stage.sh); submits ONE job
#   bash cluster/csail/build_image.sh status <jobid>    # read-only
#   bash cluster/csail/build_image.sh record <name>     # after the job: anchor the image's sha256 ON THE LAPTOP
#
# The job: -A csail -p tig-cpu -q tig-main (not preemptible; nothing preempts tig-main), 8 CPUs, 32G, 3 h.
# `record` reads the build's .DONE ("OK <sha256>") and sidecar, recomputes nothing on the laptop, and keeps
# logs/staged/csail-image-<name>.sha256: every GPU job is submitted with that hash and refuses an image that
# hashes otherwise (group scratch may be modified by others).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/csail_common.sh" || exit 1
REPO_ROOT="$(csail_repo_root)"
. "$REPO_ROOT/cluster/verify_staged_tree.sh"
. "$HERE/csail_verify.sh"
. "$HERE/csail_submit.sh"
csail_connected || exit $?

CMD="${1:-}"
case "$CMD" in
submit)
    COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD); C12=${COMMIT:0:12}; ROOT="$(csail_tree_root "$C12")"
    [ "$(git -C "$REPO_ROOT" status --porcelain | wc -l)" -eq 0 ] || { echo "REFUSING: the working tree is not clean" >&2; exit 6; }
    csail_laptop_check "$C12" || exit $?                       # tree on the cluster == this commit's manifest
    NAME="lik_build_${C12:0:8}_$(date +%H%M%S)"
    JOB=$(csail_sbatch "$ROOT" "$NAME" '%x-%j' \
        "LIK_ROOT=$ROOT LIK_CS_ROOT=$CS_ROOT EXPECT_MANIFEST_SHA=$EXPECT_MANIFEST_SHA FAKEROOT=${FAKEROOT:-auto}" \
        "-A csail -p tig-cpu -q tig-main --cpus-per-task=8 --mem=32G --time=03:00:00" \
        "$ROOT/repo/cluster/csail/build_image_job.sh") || exit 1
    echo "submitted $NAME as job $JOB (tig-cpu; log $ROOT/logs/$NAME-$JOB.out)"
    echo "JOB=$JOB"
    ;;
status)
    JOB="${2:?usage: build_image.sh status <jobid>}"
    cs_run "squeue -h -j $JOB -o '%i %T %M %R' ; sacct -j $JOB -X -o JobID,JobName,State,Elapsed,NodeList,ExitCode; ls -l $CS_ROOT/images 2>/dev/null"
    ;;
record)
    NAME="${2:?usage: build_image.sh record <image name, e.g. lik-noble-<c12>-<jobid>>}"
    [[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad image name" >&2; exit 2; }
    OUT=$(cs_run "cat $CS_ROOT/images/$NAME.DONE $CS_ROOT/images/$NAME.sif.sha256") || exit 1
    echo "$OUT"
    case "$(printf '%s\n' "$OUT" | head -1)" in OK\ *) ;; *) echo "the build did not finish OK" >&2; exit 1 ;; esac
    SHA=$(printf '%s\n' "$OUT" | head -1 | cut -d' ' -f2)
    [ "$SHA" = "$(printf '%s\n' "$OUT" | sed -n 2p | cut -d' ' -f1)" ] || { echo "STOP: .DONE and the sidecar disagree" >&2; exit 1; }
    [[ "$SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "bad sha" >&2; exit 1; }
    mkdir -p "$REPO_ROOT/logs/staged"
    echo "$SHA  $NAME.sif" > "$REPO_ROOT/logs/staged/csail-image-$NAME.sha256"
    echo "anchored image $NAME sha256 $SHA on the laptop"
    ;;
*) echo "usage: build_image.sh submit | status <jobid> | record <name>" >&2; exit 2 ;;
esac
