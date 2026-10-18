#!/bin/bash
# Submit (or resume) a NON-FLOW chart training run -- the cVAE -- as ONE single-GPU job.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Usage:
#   ROBOT=<robot> bash cluster/submit_chart_train.sh <run_name> [walltime] [-- extra trainer args]
#   bash cluster/submit_chart_train.sh --all [walltime]      # every row of cluster/chart_runs.txt
#   bash cluster/submit_chart_train.sh --status              # every row's step, plus this tree's queue
#   bash cluster/submit_chart_train.sh --smoke <robot> <K>   # 200 steps, one GPU, export+screen in RUN_DIR
# e.g.
#   ROBOT=panda bash cluster/submit_chart_train.sh panda_cvae1 24:00:00 -- --dim_latent_space=1
#
# ONE GPU PER RUN, TWO RUNS PER NODE. Thomas's design for this family (2026-10-18): all eight
# cVAE runs train concurrently, one per GPU. So the launcher asks for one task, 20 of a volta
# node's 40 cores and ONE of its two V100s, and is deliberately NOT --exclusive, so Slurm can
# place a second run on the same node. How the account's GrpTRES node=4 cap counts two jobs
# sharing one node is a cluster fact this script cannot know; cluster/CVAE_RUNBOOK.md has the
# check that measures it before the eight are submitted.
#
# LLsub DIRECTIVES MODE, exactly as submit_train.sh does it and for the same reasons: triple
# mode generates a job array and swallows the payload's exit code, so we write a launcher with
# #SBATCH directives into $RUN_DIR/launch.sh (kept) and LLsub that. In directives mode LLsub
# DROPS -g/-q/-T/-J, so every resource lives in the directives below.
#
# THE SAME-RUN GUARD IS NEVER BYPASSABLE: two jobs on one RUN_DIR race its checkpoints. It
# matches the run's FULL job name with `squeue -h -n`, never LLstat (which truncates NAME to 15
# characters, so sibling runs collapse into one string -- cluster/README.md). It checks the
# flow family's name too: results/train/<run>/ is one namespace for both trainers.
#
# NO one-rung-at-a-time guard here, deliberately: that guard (submit_train.sh) exists because a
# flow rung takes all 4 nodes, and these runs are independent single-GPU jobs meant to run
# together. The two families do not trip each other's guards -- this family's jobs are named
# <prefix>_trainc_<run>, and the flow guard matches /_train_/, which `_trainc_` does not
# contain -- so a flow ladder submitted while cVAE runs hold GPUs would START and contend for
# the cap rather than be refused. Do not run the two families at once (the runbook says so).
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"

SPEC="$(dirname "$0")/chart_runs.txt"
PARTITION="${PARTITION:-xeon-g6-volta}"
BATCH="${BATCH:-4096}"
MAX_STEPS="${MAX_STEPS:-620000}"
## DEPENDENCY (e.g. "afterok:12345") so the eight runs can wait on the screw dataset build or
## the smoke, unattended. EXCLUDE_NODES drops known-bad nodes (submit_train.sh, d-8-8-1).
DEPENDENCY="${DEPENDENCY:-}"
DEP_DIRECTIVE="${DEPENDENCY:+1}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXC_DIRECTIVE="${EXCLUDE_NODES:+1}"

_rows() { grep -vE '^\s*(#|$)' "$SPEC"; }

## ------------------------------------------------------------------ one run --
submit_one() {
    local RUN_NAME="$1" WALL="$2" EXTRA_ARGS="$3"
    local FAMILY="${FAMILY:-cvae}"
    local ROBOT="${ROBOT:?set ROBOT, e.g. ROBOT=panda or ROBOT=iiwa14}"

    ## (a) THE CHECKPOINT RACE (submit_train.sh): a property of the RUN, always refused.
    local WANT="${SC_JOB_PREFIX}_trainc_$RUN_NAME" FLOW="${SC_JOB_PREFIX}_train_$RUN_NAME"
    local LIVE
    LIVE=$(sc_run "squeue -u \$USER -h -n '$WANT,$FLOW' -o '%i' 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
    if [ -n "${LIVE:-}" ] && [ "${LIVE:-0}" -gt 0 ]; then
        echo "REFUSING: $LIVE job(s) named exactly $WANT (or $FLOW) are queued, running or completing." >&2
        echo "Two jobs on one RUN_DIR race the checkpoints. LLkill the old one, or wait." >&2
        echo "NOTE: a job just killed lingers in CG for up to a minute and squeue still lists it," >&2
        echo "so an immediate resubmit of the same name is refused. That is this guard working." >&2
        return 3
    fi

    ## The payload is exec'd by srun, so the exec bit is load-bearing (rsync -a preserves it
    ## from the local checkout; this catches a bad restage).
    sc_run "test -x ~/$SC_ROOT/repo/cluster/train_chart.sh" || {
        echo "REFUSING: train_chart.sh on the cluster is not executable -- restage first." >&2
        return 4
    }

    local RUN_DIR_R="\$HOME/$SC_ROOT/results/train/$RUN_NAME"    # expanded remotely
    echo "Submitting: RUN_NAME=$RUN_NAME FAMILY=$FAMILY ROBOT=$ROBOT WALL=$WALL PARTITION=$PARTITION DEP='${DEPENDENCY:-none}' EXCLUDE='${EXCLUDE_NODES:-none}' EXTRA='$EXTRA_ARGS'"
    sc_run "mkdir -p $RUN_DIR_R && cat > $RUN_DIR_R/launch.sh && chmod +x $RUN_DIR_R/launch.sh" <<LAUNCH
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --partition=$PARTITION
#SBATCH --time=$WALL
#SBATCH --job-name=${SC_JOB_PREFIX}_trainc_$RUN_NAME${DEP_DIRECTIVE:+
#SBATCH --dependency=$DEPENDENCY}${EXC_DIRECTIVE:+
#SBATCH --exclude=$EXCLUDE_NODES}
#SBATCH --output=../results/train/$RUN_NAME/launch.log-%j
# Generated by cluster/submit_chart_train.sh $(date -Is) -- do not edit; resubmit instead.
# NOT --exclusive: two single-GPU runs share a volta node.
srun --ntasks=1 --kill-on-bad-exit=1 "\$HOME/$SC_ROOT/repo/cluster/train_chart.sh"
LAUNCH

    ## LEARNED_IK_ROOT forwarded EXPLICITLY (submit_train.sh): the payload reads
    ## ${LEARNED_IK_ROOT:-$HOME/learned-ik} and would otherwise fall back to the default tree.
    ## sbatch exports the submission environment and srun propagates it; trailing args do not.
    sc_run "cd ~/$SC_ROOT/repo && \
      LEARNED_IK_ROOT=\$HOME/$SC_ROOT \
      FAMILY='$FAMILY' RUN_NAME='$RUN_NAME' ROBOT='$ROBOT' BATCH=$BATCH MAX_STEPS=$MAX_STEPS \
      FINAL_STEP='${FINAL_STEP:-$MAX_STEPS}' RUN_EXPORT='${RUN_EXPORT:-1}' \
      EXPORT_TO_RUN_DIR='${EXPORT_TO_RUN_DIR:-0}' \
      TRAIN_EXTRA_ARGS='$EXTRA_ARGS' \
      LLsub $RUN_DIR_R/launch.sh"
}

## ------------------------------------------------------------------ status --
## ONE remote call for every row: step and validation numbers from status.json, then this
## tree's queue (scoped by submission directory, %Z, as stage_code.sh does).
cmd_status() {
    local names
    names=$(_rows | awk '{print $3}' | tr '\n' ' ')
    sc_run "cd ~/$SC_ROOT/results/train 2>/dev/null || exit 0
        for r in $names; do
          if [ -f \"\$r/status.json\" ]; then
            printf '%-20s %s\n' \"\$r\" \"\$(python3 -c \"import json;d=json.load(open('\$r/status.json'));print('step %s  ' % d.get('global_step') + '  '.join('%s %.5g' % (k, v) for k, v in d.items() if k != 'global_step' and isinstance(v, (int, float))))\" 2>/dev/null || echo 'status.json unreadable')\"
          elif [ -d \"\$r\" ]; then printf '%-20s %s\n' \"\$r\" 'submitted, no status yet'
          else printf '%-20s %s\n' \"\$r\" '-'; fi
        done
        echo '--- exported (models/*/*__cvae*__step*.pkl) ---'
        ls ~/$SC_ROOT/repo/models/*/*__cvae*__step*.pkl 2>/dev/null || echo '(none)'
        echo '--- queue ---'; squeue -u \$USER -h -o '%Z|%.10i %.9T %j %N %R' 2>/dev/null | awk -F'|' -v d=\"\$HOME/$SC_ROOT/repo\" '\$1==d {print \$2; n++} END {if (!n) print \"(no jobs for this tree)\"}'"
}

## --------------------------------------------------------------------- all --
## Every row of chart_runs.txt, each through submit_one and so through the same-run guard. A
## row whose status.json already reached MAX_STEPS is skipped (resubmitting it would only
## resume, find the step budget spent and re-export). One remote call decides that.
cmd_all() {
    local WALL="$1" done_list family robot name args rc fails=0
    done_list=$(sc_run "cd ~/$SC_ROOT/results/train 2>/dev/null || exit 0
        for d in */status.json; do
          [ -f \"\$d\" ] || continue
          s=\$(python3 -c \"import json;print(json.load(open('\$d'))['global_step'])\" 2>/dev/null || echo 0)
          [ \"\$s\" -ge $MAX_STEPS ] && dirname \"\$d\"
        done" 2>/dev/null)
    while read -r family robot name args; do
        if grep -qx "$name" <<<"$done_list"; then
            echo "skip $name: already at $MAX_STEPS steps"
            continue
        fi
        FAMILY="$family" ROBOT="$robot" submit_one "$name" "$WALL" "$args" </dev/null
        rc=$?
        [ "$rc" -eq 0 ] || { echo "  -> $name NOT submitted (rc=$rc)"; fails=$((fails + 1)); }
    done < <(_rows)
    [ "$fails" -eq 0 ] || { echo "$fails run(s) not submitted" >&2; return 3; }
}

## ------------------------------------------------------------------- smoke --
## 200 steps on one GPU, then the in-job export and screen into the run's OWN directory
## (EXPORT_TO_RUN_DIR=1 -> $RUN_DIR/pkl), never models/: the plumbing check covers the export and the screen too, and a
## 200-step network must never sit under a name a benchmark would load. On a real partition
## (cluster/README.md rule 3: anything that trains is a job, however short).
cmd_smoke() {
    local robot="${1:?usage: --smoke <robot> <K>}" k="${2:?usage: --smoke <robot> <K>}"
    echo "200-step cVAE smoke for $robot at K=$k on $PARTITION (one GPU), export+screen into RUN_DIR."
    FAMILY=cvae ROBOT="$robot" MAX_STEPS=200 FINAL_STEP=200 RUN_EXPORT=1 EXPORT_TO_RUN_DIR=1 \
        submit_one "smoke_${robot}_cvae${k}" 00:30:00 \
        "--dim_latent_space=$k --checkpoint_every=100 --eval_every=100 --val_set_size=20 --disable_wandb"
}

case "${1:-}" in
    --status) cmd_status ;;
    --all)    cmd_all "${2:-24:00:00}" ;;
    --smoke)  shift; cmd_smoke "$@" ;;
    ""|-h|--help)
        sed -n '2,15p' "$0"; exit 2 ;;
    *)
        RUN_NAME="$1"
        WALL="${2:-24:00:00}"
        shift $(( $# > 2 ? 2 : $# ))
        [ "${1:-}" = "--" ] && shift
        submit_one "$RUN_NAME" "$WALL" "$*" ;;
esac
