#!/usr/bin/env bash
# The measurement queue, in priority order. One process per run, so each pays its own
# ~10 s compile and nothing shares state; every run writes its own summary.json and is
# skipped if that file already exists, so the queue is resumable after an interruption.
#
#   nohup scripts/run_queue.sh > results/queue.log 2>&1 &
#
# Every run is --compile and seed 0 on a 15 x 2 grid, so the cells are identical across
# rungs, start protocols and sweep points, and runs can be compared with
# scripts/collate.py --pair.
#
# Both start protocols are measured for every experiment. --start paired puts every arm at
# the same q_init in its own variables; --start native gives each formulation its own
# initialisation -- the flow's latent from its prior, the analytic map's redundancy
# parameter and branch from theirs, the joint-space arm from a random configuration.
# Neither searches: no candidate is ever scored against the problem before the solve.
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

finals () {                    # finals <wall-time>
    local W=$1
    for S in paired native; do
        run panda "final3_panda_mug_${W}_$S"  --task mug  --config latent --wall-time $W \
            --start $S --arms learned,numerical,analytic
        run panda "final3_panda_pose_${W}_$S" --task pose --config latent --wall-time $W \
            --start $S --arms learned,numerical,analytic
        run iiwa  "final3_iiwa_mug_${W}_$S"   --task mug  --config latent --wall-time $W \
            --start $S --arms learned,numerical
        run iiwa  "final3_iiwa_pose_${W}_$S"  --task pose --config latent --wall-time $W \
            --start $S --arms learned,numerical
    done
}

## 1. The ablation ladder: which change bought the grasp-success gain.
for CFG in baseline frame eval task latent latent-free-c; do
    run panda "ladder3_$CFG" --task mug --arms learned --config $CFG --wall-time 20
done

## 2. Every experiment, both start protocols, at the cap the archived tables used.
finals 20

## 3. The same again at 45 s. The GPU-profiling note argues 20 s is too tight for the
## learned grasp arm and that 45 s is the realistic budget; running both says how much of
## each result is the formulation and how much is the cap.
finals 45

## 4. Knob sweeps, learned only, paired start, one factor at a time. The default point of
## each sweep (correction_bound 0.1, latent_trust_region 4.0 / 4.3) is the 20 s paired
## finals' learned column on the same grid, so it is not re-run.
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
