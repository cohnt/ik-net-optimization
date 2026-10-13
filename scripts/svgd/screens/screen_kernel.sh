#!/bin/bash
# Kernel-space x temperature x metric-form screen, LEARNED ARM ONLY, 12 Panda cells (3 targets x 2
# guesses x {mug, pose}), paired, 20 s, N = 64, fielded config otherwise. NOT TO BE RUN until Thomas's
# go-ahead (2026-10-11 evening). Needs svgd tip with svgd_kernel {none,q,x,z,c,cz,qz,pose} and
# svgd_metric_temperature {off,on}. One process at a time under the 20G memory scope (run_e2e.sh).
SP=$(cd "$(dirname "$0")" && pwd)
run() { name=$1; shift; for t in "${TASKS[@]}"; do TMO=1800 $SP/run_e2e.sh $t e2e_ker_${t}_$name --arms learned "$@" 2>&1 | grep "load\|exit" | tr '\n' ' '; echo; done; }
# Stage A: the kernel SPACE at the fielded T = 1 and the fielded metric H.
TASKS=(mug pose)
run q      --set svgd_kernel=q
run x      --set svgd_kernel=x
run z      --set svgd_kernel=z
run c      --set svgd_kernel=c
run cz     --set svgd_kernel=cz
run qz     --set svgd_kernel=qz
run none   --set svgd_kernel=none
TASKS=(mug)                      # task-space kernel on the grasp task only (pose IK has no task freedom)
run pose   --set svgd_kernel=pose
# Stage B: temperature under both metric forms, for the spaces that separate in stage A (edit SPACES).
SPACES=(q z x cz qz)
TASKS=(mug pose)
for sp in "${SPACES[@]}"; do
  for T in 1e-1 1e-2 1e-4; do
    run ${sp}_T${T}_Hoff --set svgd_kernel=$sp --set svgd_temperature=$T --set svgd_metric_temperature=off
    run ${sp}_T${T}_Hon  --set svgd_kernel=$sp --set svgd_temperature=$T --set svgd_metric_temperature=on
  done
done
echo SCREEN_KERNEL DONE
