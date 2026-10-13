#!/bin/bash
# The 12-cell local screen runner (2026-10-10): 3 targets x 2 guesses, paired, 20 s, learned + joint space,
# fielded svgd config, one process under a 20G memory scope. Logs to $SP (default results/screens/).
# usage: run_e2e.sh <task> <tag> [--set k=v ...]   (env ROOT overrides the code tree)
ROOT=${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd $ROOT
SP=${SP:-$ROOT/results/screens}; mkdir -p $SP
task=$1; tag=$2; shift 2
free -g | sed -n 2p
echo "load before $(cut -d' ' -f1 /proc/loadavg)"
timeout ${TMO:-590} systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 --quiet .venv/bin/python -u scripts/panda/panda_benchmark.py --task $task --start paired --targets 3 --guesses 2 --seed 1 --wall-time 20 --arms learned,numerical --compile --config latent --set correction_cost_weight=10.0 --scene hardened --shelf-inset 0.1 --target-placement shelf --solver svgd --set svgd_n=64 --set svgd_kernel=q --set svgd_compile=True --set svgd_cuda_graph=True --set flow_cuda_graph=True "$@" --tag $tag > $SP/run_$tag.log 2>&1
rc=$?
echo "load after $(cut -d' ' -f1 /proc/loadavg)"; echo "$tag exit $rc"
free -g | sed -n 2p
