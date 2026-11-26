#!/bin/bash
# Shared by submit.sh and submit_check.sh (laptop side). Needs csail_common.sh, csail_verify.sh, csail_submit.sh.
#
# csail_resolve_gpu <kind>  sets GPU_OPTS (partition/account/QoS/requeue/constraint words), EXPECT_GPU, and the
# exclusion list: tedrake-h200-1 whenever the partition is vision-shared-h200 (the lab node, never ours), every
# node the cluster has recorded as having a faulty GPU ($CS_ROOT/bad_nodes.log, written by jobs that hit one),
# plus EXCLUDE_NODES from the caller. One --exclude option, so none of them is lost.
csail_resolve_gpu() {
    local kind="${1:?GPU kind}" words ex bad
    words=$(csail_gpu_args "$kind") || return 2
    EXPECT_GPU=$(csail_gpu_regex "$kind") || return 2
    ex=$(printf '%s\n' "$words" | tr ' ' '\n' | sed -n 's/^--exclude=//p' | tr '\n' ',')
    GPU_OPTS=$(printf '%s\n' "$words" | tr ' ' '\n' | grep -v '^--exclude=' | tr '\n' ' ')
    # hostname may be the FQDN (torralba-3090-1.csail.mit.edu); Slurm's node name is the part before the first dot
    bad=$(cs_run "awk '{print \$2}' $CS_ROOT/bad_nodes.log 2>/dev/null" 2>/dev/null | sed 's/\..*//' | grep -E '^[a-z0-9-]+$' | sort -u | tr '\n' ',')
    ex="$ex$bad${EXCLUDE_NODES:+$EXCLUDE_NODES,}"
    ex="${ex%,}"
    [ -z "$ex" ] || GPU_OPTS="$GPU_OPTS --exclude=$ex"
    case "$GPU_OPTS" in *vision-shared-h200*) case "$GPU_OPTS" in *tedrake-h200-1*) ;; *) echo "BUG: lab node not excluded" >&2; return 2 ;; esac ;; esac
    csail_refuse_lab_node "$GPU_OPTS" || return 2
    echo "GPU kind $kind: $GPU_OPTS (EXPECT_GPU=$EXPECT_GPU)" >&2
}
