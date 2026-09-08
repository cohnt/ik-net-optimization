#!/bin/bash
# Drive the reduced-capacity chart ladder: one training run at a time, in the order given
# by cluster/ladder_runs.txt.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Usage:
#   bash cluster/submit_ladder.sh --status        # one batched query: where every rung is
#   bash cluster/submit_ladder.sh --next          # submit the first incomplete rung
#   bash cluster/submit_ladder.sh --smoke iiwa14  # 200-step debug-gpu smoke at nb_nodes=4
#
# WHY SEQUENTIAL. Thomas's instruction is one chart at a time at full parallelism: each run
# takes all 4 nodes of the account's xeon-g6-volta GrpTRES cap. submit_train.sh's
# one-at-a-time lik_train guard is what enforces it, and --next relies on that guard rather
# than duplicating it. Never set ALLOW_CONCURRENT=1 here: two runs would not merely share
# the cap, they would halve each other's throughput inside a fixed step budget and make the
# wall-clock column meaningless.
#
# WHY A FIXED STEP BUDGET. Every rung gets --max_steps=620000 at ddp_r1's actual optimiser
# settings, so the ONLY variable across the ladder is the architecture. Smaller charts step
# faster, so this costs far less wall clock than 9 x 27 h; it does not cost fewer steps.
#
# Checkpoints: the fork sets save_top_k=-1, so every checkpoint_every=20000 step is kept.
# That is deliberate -- ddp_r1 developed pole mass somewhere near step 360000 and rotation
# had already deleted the evidence.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"

MANIFEST="$(dirname "$0")/ladder_runs.txt"
MAX_STEPS="${MAX_STEPS:-620000}"
NNODES="${NNODES:-4}"
WALL="${WALL:-96:00:00}"
BATCH="${BATCH:-512}"

## ddp_r1's ACTUAL optimiser settings, which differ from train_ddp.py's defaults
## (1.06e-4 / 4883). Matching them is what makes the ladder comparable to the existing
## 12x1024 iiwa rung.
COMMON_ARGS="--max_steps=$MAX_STEPS --learning_rate=1.5e-4 --step_lr_every=2441"

_rows() { grep -vE '^\s*(#|$)' "$MANIFEST"; }

cmd_status() {
    # ONE remote call for the whole ladder: a per-run status line plus the live queue.
    local names
    names=$(_rows | awk '{print $2}' | tr '\n' ' ')
    sc_run "cd ~/$SC_ROOT/results/train 2>/dev/null || exit 0
        for r in $names; do
          if [ -f \"\$r/status.json\" ]; then
            printf '%-18s %s\n' \"\$r\" \"\$(python3 -c \"import json,sys;d=json.load(open('\$r/status.json'));print('step %d  val_l2 %.5f  pole_gt1000 %.5f'%(d['global_step'],d.get('val_l2_error') or -1,(d.get('pole') or {}).get('pole/frac_gt_1000',-1)))\" 2>/dev/null || echo 'status.json unreadable')\"
          elif [ -d \"\$r\" ]; then printf '%-18s %s\n' \"\$r\" 'submitted, no status yet'
          else printf '%-18s %s\n' \"\$r\" '-'; fi
        done
        echo '--- queue ---'; LLstat 2>/dev/null | grep -E 'lik_train|JOBID' || echo '(no lik_train jobs)'"
}

# Prints the first rung whose status.json has not reached MAX_STEPS.
cmd_next_name() {
    local done_list
    done_list=$(sc_run "cd ~/$SC_ROOT/results/train 2>/dev/null || exit 0
        for d in */status.json; do
          [ -f \"\$d\" ] || continue
          s=\$(python3 -c \"import json;print(json.load(open('\$d'))['global_step'])\" 2>/dev/null || echo 0)
          [ \"\$s\" -ge $MAX_STEPS ] && dirname \"\$d\"
        done" 2>/dev/null)
    _rows | while read -r robot name _args; do
        grep -qx "$name" <<<"$done_list" || { echo "$name"; break; }
    done | head -1
}

cmd_next() {
    local target robot name args
    target=$(cmd_next_name)
    [ -z "$target" ] && { echo "ladder complete -- every rung has reached $MAX_STEPS steps."; return 0; }

    read -r robot name args <<<"$(_rows | awk -v t="$target" '$2==t {print; exit}')"
    echo "Next rung: $name (robot=$robot, arch: $args)"
    # submit_train.sh refuses if any lik_train job is RUNNING or PENDING, which is the
    # sequential guarantee. Resubmitting a partially-trained rung resumes it
    # (--ckpt_path=auto), so this is safe to re-run after a walltime kill.
    ROBOT="$robot" BATCH="$BATCH" \
        bash "$(dirname "$0")/submit_train.sh" "$name" "$NNODES" "$WALL" -- $COMMON_ARGS $args
}

cmd_smoke() {
    local robot="${1:?usage: --smoke <robot>}"
    echo "200-step smoke for $robot at nb_nodes=4 on debug-gpu (2 nodes, 20 min cap)."
    # Deliberately the SMALLEST architecture in the ladder: it is the one whose tensor
    # shapes differ most from everything already proven on this cluster, so it is the one
    # most likely to expose a plumbing error, and it is the cheapest to run.
    PARTITION=debug-gpu ROBOT="$robot" BATCH=64 \
        bash "$(dirname "$0")/submit_train.sh" "smoke_${robot}_n4" 2 00:20:00 -- \
        --max_steps=200 --nb_nodes=4 --eval_every=100 --val_set_size=20 \
        --checkpoint_every=100 --pole_eval_n=500 --disable_wandb \
        $( [ "$robot" = panda ] && echo --dim_latent_space=7 )
}

case "${1:---status}" in
    --status) cmd_status ;;
    --next)   cmd_next ;;
    --smoke)  shift; cmd_smoke "$@" ;;
    --next-name) cmd_next_name ;;
    *) echo "usage: $0 [--status|--next|--smoke <robot>]" >&2; exit 2 ;;
esac
