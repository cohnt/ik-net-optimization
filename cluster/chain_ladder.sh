#!/bin/bash
# Queue a block of ladder rungs as a Slurm dependency chain, so the campaign advances with
# no Claude session and no human attached.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Usage:
#   bash cluster/chain_ladder.sh <after_jobid> <run_name> [run_name...]
#   bash cluster/chain_ladder.sh 5559773 iiwa14_n4 iiwa14_n8 iiwa14_n12_w256
#
# Each rung is submitted with --dependency=afterany:<previous>, so exactly one runs at a
# time and it starts the moment its predecessor leaves the queue. Training is sequential
# at full 4-node parallelism, which is the instruction; the chain just removes the human
# from the hand-off.
#
# WHY afterany, NOT afterok. A rung that dies must not stall every rung behind it -- that
# would turn one node failure into days of idle allocation discovered hours later. The
# cost is that a dead rung's successor starts anyway, so ALWAYS check
# `submit_ladder.sh --status` after a chain: a rung short of MAX_STEPS was not trained,
# whatever the queue says. Resubmitting it resumes from last.ckpt (--ckpt_path=auto).
#
# Export and screening happen INSIDE each training job (train_flow.sh, node 0, on rc=0),
# so a rung leaves the queue already exported. No second dependency link per rung.
#
# ALLOW_CONCURRENT=1 is set deliberately and is safe HERE, unlike in normal use: every
# rung has its own RUN_DIR, so the checkpoint race the guard exists to stop cannot happen,
# and the dependency chain -- not the guard -- is what serialises them. Without it the
# guard would refuse every rung after the first, because chained jobs sit PENDING.
set -uo pipefail
cd "$(dirname "$0")/.."

PREV="${1:?usage: chain_ladder.sh <after_jobid> <run_name> [run_name...]}"
FIRST_PREV="$PREV"    # so the first link can differ in dependency TYPE
shift
[ $# -gt 0 ] || { echo "no rungs given" >&2; exit 2; }

MANIFEST=cluster/ladder_runs.txt
MAX_STEPS="${MAX_STEPS:-620000}"
NNODES="${NNODES:-4}"
WALL="${WALL:-96:00:00}"
BATCH="${BATCH:-512}"
## Known-bad nodes, passed through to every rung in the chain. See submit_train.sh.
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
COMMON_ARGS="--max_steps=$MAX_STEPS --learning_rate=1.5e-4 --step_lr_every=2441"

for name in "$@"; do
    row=$(grep -vE '^\s*(#|$)' "$MANIFEST" | awk -v t="$name" '$2==t {print; exit}')
    [ -n "$row" ] || { echo "REFUSING: '$name' is not a rung in $MANIFEST" >&2; exit 3; }
    robot=$(awk '{print $1}' <<<"$row")
    args=$(cut -d' ' -f3- <<<"$(tr -s ' ' <<<"$row")")

    ## GATE_TYPE (e.g. "afterok") gates EVERY rung on the job this chain was queued
    ## behind -- `$1`, whatever it is -- in addition to the rung's own predecessor link.
    ## Slurm reads a comma-separated dependency list as an AND.
    ##
    ## ONLY SET GATE_TYPE WHEN `$1` IS ACTUALLY A GATE, such as a 200-step validation run
    ## whose failure should stop the ladder. Chains are also queued behind an ordinary
    ## predecessor rung purely for sequencing, and gating on one of those would make every
    ## rung conditional on an unrelated run's exit code. Unset (the default) is right
    ## there.
    ##
    ## Gating only the first rung is not enough, and whether it is enough depends on
    ## cluster configuration, which is the worst kind of dependence:
    ##   * with `kill_invalid_depend` UNSET (MIT SuperCloud: DependencyParameters is null),
    ##     a failed smoke leaves rung 1 PENDING for ever with DependencyNeverSatisfied, and
    ##     later rungs pend behind it only because rung 1 never TERMINATES. The ladder is
    ##     blocked by the accident of the chain's shape.
    ##   * with `kill_invalid_depend` SET, rung 1 is CANCELLED, its termination satisfies
    ##     rung 2's `afterany`, and the whole ladder trains on a robot whose smoke failed.
    ## Gating every rung gives the same, correct answer under both.
    ##
    ## Rung-to-rung links stay `afterany` on purpose: an independent measurement that dies
    ## must not stall the measurements behind it.
    ##
    ## NOTE either way: an unsatisfiable dependency here does not cancel anything and
    ## notifies nobody. The jobs simply sit PENDING looking queued, so the smoke's exit has
    ## to be watched rather than assumed.
    if [ -n "${GATE_TYPE:-}" ]; then
        if [ "$PREV" = "$FIRST_PREV" ]; then
            DEP="$GATE_TYPE:$FIRST_PREV"
        else
            DEP="$GATE_TYPE:$FIRST_PREV,afterany:$PREV"
        fi
    else
        DEP="afterany:$PREV"
    fi

    echo "=== chaining $name (robot=$robot, arch: $args) with dependency $DEP"
    out=$(ALLOW_CONCURRENT=1 DEPENDENCY="$DEP" ROBOT="$robot" BATCH="$BATCH" \
            EXCLUDE_NODES="${EXCLUDE_NODES:-}" \
            bash cluster/submit_train.sh "$name" "$NNODES" "$WALL" -- $COMMON_ARGS $args 2>&1)
    echo "$out" | tail -3
    jobid=$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | grep -oE '[0-9]+$')
    [ -n "$jobid" ] || { echo "REFUSING to continue: no job id parsed for $name" >&2; exit 4; }
    PREV="$jobid"
done

echo
echo "chain queued. Last job in the chain: $PREV"
echo "Check with: bash cluster/submit_ladder.sh --status"
