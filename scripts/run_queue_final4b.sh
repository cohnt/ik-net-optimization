#!/bin/bash
# Second pass of the final4 queue, reordered after two multi-hour GPU wedges -- both of
# them iiwa mug runs. Load-bearing stages first; the wedge-prone iiwa mug runs are
# quarantined at the tail so a wedge there cannot eat the ladder or the panda finals.
# Skip logic unchanged: any run with a summary.json is not re-run.
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
    timeout 7500 $PY "scripts/$robot/${robot}_benchmark.py" --targets $T --guesses $G \
        --seed 0 --compile --tag "$tag" "$@" || echo "!! $tag failed with $?"
}

echo "== queue (pass 2, reordered) started $(date)"

# C. The ablation ladder (load-bearing).
for CFG in baseline frame eval latent; do
    run panda "ladder4_${CFG}" --task mug --wall-time 20 --arms learned --config "$CFG"
done

# B'. Panda finals at 45 s.
for S in paired native; do
    run panda "final4_panda_mug_45_${S}"  --task mug  --wall-time 45 --start "$S" \
        --config latent --arms learned,numerical,analytic,analytic8
    run panda "final4_panda_pose_45_${S}" --task pose --wall-time 45 --start "$S" \
        --config latent --arms learned,numerical,analytic,analytic8
done

# D. Charted-bundle grid.
run panda final4_panda_mug_20_paired_charted  --task mug  --wall-time 20 --start paired \
    --config latent --arms learned,numerical,analytic,analytic8 --guess-filter charted
run panda final4_panda_pose_20_paired_charted --task pose --wall-time 20 --start paired \
    --config latent --arms learned,numerical,analytic,analytic8 --guess-filter charted

# E(ii). Dose-response.
for EPS in 0.016 0.032 0.064 0.128; do
    run panda "dose4_eps${EPS}" --task mug --wall-time 20 --start paired --config latent \
        --arms learned --set chart_error_scale=${EPS}
done

# iiwa 45 s pose (never wedged).
for S in paired native; do
    run iiwa "final4_iiwa_pose_45_${S}" --task pose --wall-time 45 --start "$S" --config latent
done

# Quarantine: the iiwa mug runs, every one of which is wedge-prone tonight.
for S in paired native; do
    run iiwa "final4_iiwa_mug_20_${S}" --task mug --wall-time 20 --start "$S" --config latent
done
for S in paired native; do
    run iiwa "final4_iiwa_mug_45_${S}" --task mug --wall-time 45 --start "$S" --config latent
done

echo "== queue (pass 2) finished $(date)"
