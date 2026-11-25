#!/bin/bash
# sbatch payload: the GPU plumbing check + one-time cache warm-up, on ONE GPU of the campaign's pinned type.
#   nvidia-smi, torch.cuda + one kernel, pydrake from the image at the pinned release (VERSION.TXT, CalcRobotClearances), ikflow = the vendored
#   fork, Panda weights and the scenes' Drake models cached under ~/lik-engaging/home-cache (internet is
#   used HERE, once; benchmark jobs run from the warm cache).
# Sentinel ~/lik-engaging/plumb.DONE (OK / FAIL). Env as bench_job.sh, no MANIFEST.
source "${LIK_CROOT:?LIK_CROOT unset}/repo/cluster/engaging/job_lib.sh" || exit 2
eg_job_init
DONE="$EB/plumb.DONE"; rm -f "$DONE"
echo "===== nvidia-smi ====="; nvidia-smi
## The Drake the image must carry: the DRAKE_* variables of the committed lik.def (the pin lives only there).
DEF="$REPO/cluster/engaging/lik.def"
PIN_VER=$(sed -n 's/^ *DRAKE_VERSION="\([0-9.]*\)".*/\1/p' "$DEF" | head -1)
PIN_COMMIT=$(sed -n 's/^ *DRAKE_VERSION_TXT_COMMIT="\([0-9a-f]*\)".*/\1/p' "$DEF" | head -1)
[ -n "$PIN_VER" ] && [ ${#PIN_COMMIT} -eq 40 ] || { echo "FAIL no Drake pin in lik.def" | tee "$DONE"; exit 1; }
mkdir -p "$EB/home-cache/.cache"
export LIK_REPO="$REPO"
## HOME = the warm cache's home, so ikflow/Drake caches land where benchmark jobs symlink from.
HOME="$EB/home-cache" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$REPO/cluster/engaging/py_in_image.sh" "$REPO/cluster/engaging/plumb_check.py" "$PIN_VER" "$PIN_COMMIT" --warm
RC=$?
if [ $RC -eq 0 ]; then echo "OK $(date -Is) $(hostname)" > "$DONE"; else echo "FAIL rc=$RC" > "$DONE"; fi
exit $RC
