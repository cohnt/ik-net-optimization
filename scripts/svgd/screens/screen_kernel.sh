#!/bin/bash
# Kernel-space screen, LEARNED ARM ONLY, 12 Panda cells (3 targets x 2 guesses x {mug, pose}),
# paired, 20 s, N = 64, fielded config otherwise. Needs svgd tip with svgd_kernel
# {none,q,x,z,c,cz,qz,pose}. One process at a time under the 20G memory scope (run_e2e.sh).
# Its stage B (temperature x metric form) ran 2026-10-11 and is gone: svgd_temperature and
# svgd_metric_temperature were removed 2026-10-14 (stage SVGD_R22 measured T inert); git history
# holds it.
SP=$(cd "$(dirname "$0")" && pwd)
run() { name=$1; shift; for t in "${TASKS[@]}"; do TMO=1800 $SP/run_e2e.sh $t e2e_ker_${t}_$name --arms learned "$@" 2>&1 | grep "load\|exit" | tr '\n' ' '; echo; done; }
# Stage A: the kernel SPACE under the fielded metric H.
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
echo SCREEN_KERNEL DONE
