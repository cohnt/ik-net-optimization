#!/usr/bin/env bash
# The measurement queue, in priority order. One process per run, so each pays its own
# ~10 s compile and nothing shares state; every run writes its own summary.json and is
# skipped if that file already exists, so the queue is resumable after an interruption.
#
#   nohup scripts/run_queue.sh > results/queue.log 2>&1 &
#
# Every run is --compile and seed 0 on a 15 x 2 grid, so the cells are identical across
# rungs and sweep points and the runs can be compared with scripts/collate.py --pair.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=15
G=2

run () {                       # run <robot> <tag> <args...>
    local robot=$1 tag=$2; shift 2
    local out="results/$robot/benchmark/$tag/summary.json"
    if [ -f "$out" ]; then echo "== skip $tag (already done)"; return; fi
    echo "== $(date +%H:%M:%S) $tag"
    timeout 7200 $PY "scripts/$robot/${robot}_benchmark.py" --targets $T --guesses $G \
        --seed 0 --compile --tag "$tag" "$@" || echo "!! $tag failed with $?"
}

## 1. The ablation ladder: which change bought the grasp-success gain.
for CFG in baseline frame eval task latent latent-free-c; do
    run panda "ladder3_$CFG" --task mug --arms learned --config $CFG --wall-time 20
done

## 2. Panda finals, both wall-clock caps.
for W in 20 45; do
    run panda "final3_panda_mug_$W"  --task mug  --config latent --wall-time $W \
        --arms learned,numerical,analytic
    run panda "final3_panda_pose_$W" --task pose --config latent --wall-time $W \
        --arms learned,numerical,analytic
done

## 3. iiwa finals -- the first valid iiwa numbers; every archived table used --start native.
for W in 20 45; do
    run iiwa "final3_iiwa_mug_$W"  --task mug  --config latent --wall-time $W --arms learned,numerical
    run iiwa "final3_iiwa_pose_$W" --task pose --config latent --wall-time $W --arms learned,numerical
done

## 4. Knob sweeps, learned only, one factor at a time. The default point of each sweep
## (correction_bound 0.1, latent_trust_region 4.0 / 4.3) is the finals' learned column on
## the same grid, so it is not re-run.
for B in 0.2 0.4 0.8; do
    run iiwa  "sweep3_iiwa_corr_$B"  --task mug --config latent --wall-time 20 \
        --arms learned --set correction_bound=$B
done
for B in 0.2 0.4 0.8; do
    run panda "sweep3_panda_corr_$B" --task mug --config latent --wall-time 20 \
        --arms learned --set correction_bound=$B
done
for R in 3.0 6.0 None; do
    run iiwa  "sweep3_iiwa_latent_$R"  --task mug --config latent --wall-time 20 \
        --arms learned --set latent_trust_region=$R
done
for R in 3.0 6.0 None; do
    run panda "sweep3_panda_latent_$R" --task mug --config latent --wall-time 20 \
        --arms learned --set latent_trust_region=$R
done

echo "== $(date +%H:%M:%S) queue finished"
