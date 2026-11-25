#!/bin/bash
# Run python INSIDE the Engaging image: `py_in_image.sh [python args...]`. This is run_items.sh's
# TEST_PYTHON on Engaging, so the payload (claims, watcher, resume, publishing) runs on the host
# unchanged and only the interpreter lives in the image.
#
# Needs (exported by the job script): LIK_SIF (the image), LIK_BIND (the project tree on the cluster,
# which holds the staged repo, state/, results/ and the warmed caches), LIK_REPO (the staged repo).
# The host's environment is passed through by Apptainer, so the job script has already unset the host
# compiler/loader variables (CC CXX CPATH LIBRARY_PATH LD_LIBRARY_PATH); PATH and PYTHONPATH are set
# here, explicitly, to the image's Drake pin and the staged repo's vendored ikflow fork.
: "${LIK_SIF:?py_in_image.sh: LIK_SIF unset}" "${LIK_BIND:?py_in_image.sh: LIK_BIND unset}" "${LIK_REPO:?py_in_image.sh: LIK_REPO unset}"
NV=(--nv)
# An empty CUDA_VISIBLE_DEVICES (DEVICE=cpu items) means no GPU: leave --nv off.
[ "${CUDA_VISIBLE_DEVICES-x}" != "" ] || NV=()
exec apptainer exec "${NV[@]}" --home "$HOME" -B "$LIK_BIND" --pwd "$PWD" \
    --env PATH=/opt/lik/venv/bin:/usr/local/bin:/usr/bin:/bin \
    --env PYTHONPATH="/opt/lik/drake/lib/python3.12/site-packages:$LIK_REPO/third_party/ikflow" \
    "$LIK_SIF" /opt/lik/venv/bin/python "$@"
