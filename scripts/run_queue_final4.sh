#!/bin/bash
# The final4 measurement queue: exact paired starts, the analytic8 column, the re-run
# ladder, the charted-bundle grid, and the chart-degradation dose-response.
# Resumable: a run whose summary.json exists is skipped. One run at a time -- the
# wall-clock caps are only meaningful on an uncontended machine.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=15
G=2

run() {
    local robot=$1 tag=$2; shift 2
    local out="results/$robot/benchmark/$tag/summary.json"
    if [ -f "$out" ]; then echo "== skip $tag (already done)"; return; fi
    echo "== $(date +%H:%M:%S) $tag"
    # 7500 s: generous. A run is ~5-15 min; the margin exists for the (rare, measured
    # once in 1740 cells) C++-level wedge, which releases on its own and now costs only
    # wall time -- IPOPT's own max_wall_time ends the solve at the next iteration and the
    # returned point is verified normally.
    timeout 7500 $PY "scripts/$robot/${robot}_benchmark.py" --targets $T --guesses $G \
        --seed 0 --compile --tag "$tag" "$@" || echo "!! $tag failed with $?"
}

finals() {
    local W=$1
    for S in paired native; do
        run panda "final4_panda_mug_${W}_${S}"  --task mug  --wall-time "$W" --start "$S" \
            --config latent --arms learned,numerical,analytic,analytic8
        run panda "final4_panda_pose_${W}_${S}" --task pose --wall-time "$W" --start "$S" \
            --config latent --arms learned,numerical,analytic,analytic8
        run iiwa  "final4_iiwa_mug_${W}_${S}"   --task mug  --wall-time "$W" --start "$S" \
            --config latent
        run iiwa  "final4_iiwa_pose_${W}_${S}"  --task pose --wall-time "$W" --start "$S" \
            --config latent
    done
}

echo "== queue started $(date)"

# A. Finals at the 20 s cap (the headline tables).
finals 20

# C. The ablation ladder, re-run under the exact paired start (the start repair changes
# what every rung measures, so the attribution table must be regenerated, not reused).
for CFG in baseline frame eval task latent latent-free-c; do
    run panda "ladder4_${CFG}" --task mug --wall-time 20 --arms learned --config "$CFG"
done

# B. Finals at the 45 s cap (the cap-sensitivity story needs both).
finals 45

# D. The charted-bundle grid: guesses rejection-sampled into the four wide branches,
# identically for every arm, before any solve. A separate table with its caveat.
run panda final4_panda_mug_20_paired_charted  --task mug  --wall-time 20 --start paired \
    --config latent --arms learned,numerical,analytic,analytic8 --guess-filter charted
run panda final4_panda_pose_20_paired_charted --task pose --wall-time 20 --start paired \
    --config latent --arms learned,numerical,analytic,analytic8 --guess-filter charted

# E(ii). Chart-degradation dose-response: the Panda flow output perturbed to a target
# median chart error, learned arm only, paired, mug. eps values calibrated by
# scripts/../smoke (see CLAUDE.md); 0 is the control point and is the final4 run above.
for EPS in 0.008 0.016 0.032 0.064; do
    run panda "dose4_eps${EPS}" --task mug --wall-time 20 --start paired --config latent \
        --arms learned --set chart_error_scale=${EPS}
done

echo "== queue finished $(date)"
