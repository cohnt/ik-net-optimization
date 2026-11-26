#!/bin/bash
# sbatch payload: the GPU plumbing check + one-time cache warm-up, on ONE GPU of the campaign's pinned type.
#   nvidia-smi, torch.cuda + one kernel, pydrake from the image at the pinned release (VERSION.TXT, CalcRobotClearances), ikflow = the vendored
#   fork, Panda weights and the scenes' Drake models cached under ~/lik-engaging/home-cache (internet is
#   used HERE, once; benchmark jobs run from the warm cache).
# Sentinel ~/lik-engaging/plumb.DONE (OK / FAIL). Env as bench_job.sh, no MANIFEST.
# submit.sh passes LIK_CROOT relative to the cluster home.
case "${LIK_CROOT:?LIK_CROOT unset}" in /*) ;; *) LIK_CROOT="$HOME/$LIK_CROOT" ;; esac; export LIK_CROOT
source "$LIK_CROOT/repo/cluster/engaging/job_lib.sh" || exit 2
eg_job_init
CPU_ARGS=(); DONE="$EB/plumb.DONE"
## PLUMB_CPU=1 (submit.sh plumbcpu, mit_quicktest): the same check without the GPU block; also warms the caches.
if [ "${PLUMB_CPU:-0}" = 1 ]; then CPU_ARGS=(--cpu); DONE="$EB/plumb-cpu.DONE"; CUDA_VISIBLE_DEVICES=""; export CUDA_VISIBLE_DEVICES; fi
rm -f "$DONE"
if [ "${PLUMB_CPU:-0}" != 1 ]; then echo "===== nvidia-smi ====="; nvidia-smi; fi
## The Drake the image must carry: the DRAKE_* variables of the committed lik.def (the pin lives only there).
DEF="$REPO/cluster/engaging/lik.def"
PIN_VER=$(sed -n 's/^ *DRAKE_VERSION="\([0-9.]*\)".*/\1/p' "$DEF" | head -1)
PIN_COMMIT=$(sed -n 's/^ *DRAKE_VERSION_TXT_COMMIT="\([0-9a-f]*\)".*/\1/p' "$DEF" | head -1)
[ -n "$PIN_VER" ] && [ ${#PIN_COMMIT} -eq 40 ] || { echo "FAIL no Drake pin in lik.def" | tee "$DONE"; exit 1; }
mkdir -p "$EB/home-cache/.cache"
export LIK_REPO="$REPO"
## PLUMB_DIAG=1: which apptainer mount spellings let the container see the staged tree? Prints, then exits.
if [ "${PLUMB_DIAG:-0}" = 1 ]; then
    T="$REPO/cluster/engaging/plumb_check.py"
    echo "host sees $T: $(test -f "$T" && echo yes || echo NO); apptainer $(apptainer --version); LIK_BIND=$LIK_BIND PWD=$PWD HOME=$HOME"
    PROBE='import os,sys; t=sys.argv[1]; print("  exists:", os.path.exists(t), "| /home:", os.path.isdir("/home") and sorted(os.listdir("/home")), "| HOME", os.environ.get("HOME"), "| cwd", os.getcwd())'
    run() { echo "--- $1"; shift; "$@" 2>&1 | head -12; }
    A=(apptainer exec --env PATH=/opt/lik/venv/bin:/usr/bin:/bin)
    run "A: --home HOME:/lik-home -B LIK_BIND (current)" "${A[@]}" --home "$EB/home-cache:/lik-home" -B "$LIK_BIND" "$LIK_SIF" /opt/lik/venv/bin/python -c "$PROBE" "$T"
    run "B: -B LIK_BIND, default home" "${A[@]}" -B "$LIK_BIND" "$LIK_SIF" /opt/lik/venv/bin/python -c "$PROBE" "$T"
    run "C: --no-home -B LIK_BIND" "${A[@]}" --no-home -B "$LIK_BIND" "$LIK_SIF" /opt/lik/venv/bin/python -c "$PROBE" "$T"
    run "D: -B LIK_BIND:LIK_BIND --pwd tree" "${A[@]}" -B "$LIK_BIND:$LIK_BIND" --pwd "$REPO" "$LIK_SIF" /opt/lik/venv/bin/python -c "$PROBE" "$T"
    run "E: no bind at all, default home" "${A[@]}" "$LIK_SIF" /opt/lik/venv/bin/python -c "$PROBE" "$T"
    run "F: --contain -B LIK_BIND" "${A[@]}" --contain -B "$LIK_BIND" "$LIK_SIF" /opt/lik/venv/bin/python -c "$PROBE" "$T"
    echo "OK diag $(date -Is)" > "$EB/plumb-cpu.DONE"; exit 0
fi
## HOME = the warm cache's home, so ikflow/Drake caches land where benchmark jobs symlink from.
HOME="$EB/home-cache" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" \
    "$REPO/cluster/engaging/py_in_image.sh" "$REPO/cluster/engaging/plumb_check.py" "$PIN_VER" "$PIN_COMMIT" --warm "${CPU_ARGS[@]}"
RC=$?
if [ $RC -eq 0 ]; then echo "OK $(date -Is) $(hostname)" > "$DONE"; else echo "FAIL rc=$RC" > "$DONE"; fi
exit $RC
