#!/bin/bash
# Phase 0 of the reduced-chart campaign: does a 6-block iiwa chart actually win cells?
#
# elated-firefly-11 is a 6-block iiwa14 chart of unknown provenance that has been sitting
# in results/train/controls/. Screened, it has ZERO pole mass on both domains (pole/max
# 181, against ddp-r1's 7.8e9) at a cost of 29.6 mm median chart error against 10.1 mm.
# That is exactly the trade the whole ladder is built to measure, available for free.
#
# So: the grasp task -- the row with the live deficit -- at 60 cells, both start
# protocols, firefly against the adopted ddp-r1 chart on IDENTICAL cells.
#
# This is a PILOT, not a reportable arm: firefly's provenance is unknown, so it cannot be
# attributed to an architecture choice. The trained iiwa14_n6 rung is the reportable one.
# What this buys is an early read, before a week of GPU time, on whether trading an order
# of magnitude of chart accuracy for a lower gain ceiling pays.
#
# Laptop run: both arms of a comparison are on one machine, so the comparison is valid;
# the wall-clock numbers are NOT comparable with any cluster run.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
FIREFLY=results/train/controls/iiwa14__elated-firefly-11.pkl
CAP="${CAP:-20}"
TARGETS="${TARGETS:-15}"
GUESSES="${GUESSES:-4}"

for start in paired native; do
    for chart in ddpr1 firefly; do
        ckpt=""
        [ "$chart" = firefly ] && ckpt="--checkpoint $FIREFLY"
        tag="pilot_${chart}_mug_${CAP}_${start}"
        echo "=========================================================="
        echo "=== $tag   $(date -Is)"
        echo "=========================================================="
        # --compile on both, identically: it changes how many iterations fit inside a
        # fixed cap, so it must not differ between things being compared.
        $PY -u scripts/iiwa/iiwa_benchmark.py \
            --task mug --targets "$TARGETS" --guesses "$GUESSES" --wall-time "$CAP" \
            --arms learned,numerical --config latent --start "$start" --seed 1 \
            --compile --set correction_cost_weight=10 --tag "$tag" $ckpt \
            || echo "FAILED: $tag"
    done
done
echo "PILOT COMPLETE $(date -Is)"
