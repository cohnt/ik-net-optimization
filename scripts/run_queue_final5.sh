#!/bin/bash
# The final5 queue: the first measurement of the corrected protocol. Everything before it
# is superseded for the grasp task and for every paired column:
#   - the learned arm is the draft's eq. (6) formulation only (task-param removed);
#   - the conditioning-pose and pose-target boxes are linear constraints, so the exact
#     paired start genuinely reaches the solver;
#   - paired cells whose q_init an arm cannot represent are immediate failures;
#   - guesses are drawn per target;
#   - runs go under a sleep inhibitor (launch via: systemd-inhibit --what=sleep:idle
#     --mode=block bash scripts/run_queue_final5.sh, or start the inhibitor separately).
# Resumable: a run whose summary.json exists is skipped.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=15
G=4

run() {
    local robot=$1 tag=$2; shift 2
    local out="results/$robot/benchmark/$tag/summary.json"
    if [ -f "$out" ]; then echo "== skip $tag (already done)"; return; fi
    echo "== $(date +%H:%M:%S) $tag"
    timeout 7500 $PY "scripts/$robot/${robot}_benchmark.py" --targets $T --guesses $G \
        --seed 0 --compile --tag "$tag" "$@" || echo "!! $tag failed with $?"
}

finals() {
    local W=$1
    for S in paired native; do
        run panda "final5_panda_mug_${W}_${S}"  --task mug  --wall-time "$W" --start "$S" \
            --config latent --arms learned,numerical,analytic,analytic8
        run panda "final5_panda_pose_${W}_${S}" --task pose --wall-time "$W" --start "$S" \
            --config latent --arms learned,numerical,analytic,analytic8
        run iiwa  "final5_iiwa_mug_${W}_${S}"   --task mug  --wall-time "$W" --start "$S" \
            --config latent
        run iiwa  "final5_iiwa_pose_${W}_${S}"  --task pose --wall-time "$W" --start "$S" \
            --config latent
    done
}

echo "== final5 queue started $(date)"
systemd-inhibit --list 2>/dev/null | grep -q "sleep" || \
    echo "WARNING: no sleep inhibitor detected -- idle suspend will corrupt wall clocks"

# A. Finals at 20 s (headline tables).
finals 20

# B. The ladder, now four rungs (task-param rungs are gone with the formulation).
for CFG in baseline frame eval latent; do
    run panda "ladder5_${CFG}" --task mug --wall-time 20 --arms learned --config "$CFG"
done

# C. Finals at 45 s (cap sensitivity).
finals 45

# D. Charted-bundle grid (separate table, stated caveat).
run panda final5_panda_mug_20_paired_charted  --task mug  --wall-time 20 --start paired \
    --config latent --arms learned,numerical,analytic,analytic8 --guess-filter charted
run panda final5_panda_pose_20_paired_charted --task pose --wall-time 20 --start paired \
    --config latent --arms learned,numerical,analytic,analytic8 --guess-filter charted

# E. Chart-degradation dose-response (learned only; eps -> {12, 20, 43, 83} mm median).
for EPS in 0.016 0.032 0.064 0.128; do
    run panda "dose5_eps${EPS}" --task mug --wall-time 20 --start paired --config latent \
        --arms learned --set chart_error_scale=${EPS}
done

echo "== final5 queue finished $(date)"
