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
#   bash cluster/submit_chart_train.sh --pair <run_a> <run_b> [walltime]   # two rows, ONE 2-GPU job
#   PAIRS=1 bash cluster/submit_chart_train.sh --all [walltime]           # every row, two per job
#   ONLY='<regex>' [PAIRS=1] bash cluster/submit_chart_train.sh --all [walltime]  # rows whose run name matches
#   bash cluster/submit_chart_train.sh --smoke-pair <robot> <K_a> <K_b>     # the pair mode's smoke
# e.g.
#   ROBOT=panda bash cluster/submit_chart_train.sh panda_cvae1 24:00:00 -- --dim_latent_space=1
#
# ONE GPU PER RUN, TWO RUNS PER NODE. Thomas's design for this family (2026-10-18): all eight
# cVAE runs train concurrently, one per GPU. So the launcher asks for one task, 20 of a volta
# node's 40 cores and ONE of its two V100s, and is deliberately NOT --exclusive, so Slurm can
# place a second run on the same node. How the account's GrpTRES node=4 cap counts two jobs
# sharing one node is a cluster fact this script cannot know; cluster/CVAE_RUNBOOK.md has the
# check that measures it before the eight are submitted. If the cap counts each job's node
# separately (Slurm's documented behaviour for a node limit), eight single-GPU jobs run four at a
# time; `--pair` / `PAIRS=1` is the answer then: one job per node, both V100s, two runs
# (cluster/train_chart_pair.sh), named <prefix>_trainc_<run_a>+<run_b>.
#
# LLsub DIRECTIVES MODE, exactly as submit_train.sh does it and for the same reasons: triple
# mode generates a job array and swallows the payload's exit code, so we write a launcher with
# #SBATCH directives into $RUN_DIR/launch.sh (kept) and LLsub that. In directives mode LLsub
# DROPS -g/-q/-T/-J, so every resource lives in the directives below.
#
# THE SAME-RUN GUARD IS NEVER BYPASSABLE: two jobs on one RUN_DIR race its checkpoints. It
# matches the run's FULL job name with `squeue -h -n`, never LLstat (which truncates NAME to 15
# characters, so sibling runs collapse into one string -- cluster/README.md); a pair job counts
# for each of its two runs. It checks the flow family's name too: results/train/<run>/ is one
# namespace for both trainers.
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

## ---------------------------------------------------------------- the guard --
## (a) THE CHECKPOINT RACE (submit_train.sh): a property of the RUN, always refused, never
## bypassable. Counts every queued, running or completing job of this account that would write
## into any of the given runs' RUN_DIRs: a single cVAE job `<prefix>_trainc_<run>`, a pair job
## naming the run on either side of its `+`, or a flow job `<prefix>_train_<run>`. Full names
## from `squeue -o '%j'` (never LLstat). A failed query REFUSES: an empty answer is not "none".
refuse_if_live() {
    local names n run p hits=0
    names=$(sc_run "squeue -u \$USER -h -o '%j'") || {
        echo "REFUSING: could not read the queue to check for a live job on $*." >&2
        return 3
    }
    for run in "$@"; do
        while read -r n; do
            case "$n" in
                "${SC_JOB_PREFIX}_train_$run"|"${SC_JOB_PREFIX}_trainc_$run")
                    hits=$((hits + 1)) ;;
                "${SC_JOB_PREFIX}_trainc_"*+*)
                    IFS='+' read -r -a parts <<< "${n#${SC_JOB_PREFIX}_trainc_}"
                    for p in "${parts[@]}"; do [ "$p" = "$run" ] && hits=$((hits + 1)); done ;;
            esac
        done <<< "$names"
    done
    if [ "$hits" -gt 0 ]; then
        echo "REFUSING: $hits queued, running or completing job(s) already train $* (single, pair" >&2
        echo "or flow job of that run name). Two jobs on one RUN_DIR race the checkpoints." >&2
        echo "LLkill the old one, or wait. NOTE: a job just killed lingers in CG for up to a" >&2
        echo "minute and squeue still lists it, so an immediate resubmit is refused. That is this" >&2
        echo "guard working, not a bug." >&2
        return 3
    fi
}

## The payload is exec'd by srun, so the exec bit is load-bearing (rsync -a preserves it from
## the local checkout; this catches a bad restage).
require_exec() {
    sc_run "test -x ~/$SC_ROOT/repo/cluster/$1" || {
        echo "REFUSING: $1 on the cluster is not executable -- restage first." >&2
        return 4
    }
}

## The environment both payloads read. LEARNED_IK_ROOT forwarded EXPLICITLY (submit_train.sh):
## the payload reads ${LEARNED_IK_ROOT:-$HOME/learned-ik} and would otherwise fall back to the
## default tree. sbatch exports the submission environment and srun propagates it; trailing
## arguments do not.
common_env() {
    echo "LEARNED_IK_ROOT=\$HOME/$SC_ROOT FAMILY='${FAMILY:-cvae}' BATCH=$BATCH MAX_STEPS=$MAX_STEPS \
      FINAL_STEP='${FINAL_STEP:-$MAX_STEPS}' RUN_EXPORT='${RUN_EXPORT:-1}' \
      EXPORT_TO_RUN_DIR='${EXPORT_TO_RUN_DIR:-0}'"
}

## ------------------------------------------------------------------ one run --
submit_one() {
    local RUN_NAME="$1" WALL="$2" EXTRA_ARGS="$3"
    local ROBOT="${ROBOT:?set ROBOT, e.g. ROBOT=panda or ROBOT=iiwa14}"
    refuse_if_live "$RUN_NAME" || return $?
    require_exec train_chart.sh || return $?

    local RUN_DIR_R="\$HOME/$SC_ROOT/results/train/$RUN_NAME"    # expanded remotely
    echo "Submitting: RUN_NAME=$RUN_NAME FAMILY=${FAMILY:-cvae} ROBOT=$ROBOT WALL=$WALL PARTITION=$PARTITION DEP='${DEPENDENCY:-none}' EXCLUDE='${EXCLUDE_NODES:-none}' EXTRA='$EXTRA_ARGS'"
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

    sc_run "cd ~/$SC_ROOT/repo && $(common_env) \
      RUN_NAME='$RUN_NAME' ROBOT='$ROBOT' TRAIN_EXTRA_ARGS='$EXTRA_ARGS' \
      LLsub $RUN_DIR_R/launch.sh"
}

## ----------------------------------------------------------------- one pair --
## Two rows of chart_runs.txt in ONE job: one node, both V100s, all 40 cores, one run per GPU
## (cluster/train_chart_pair.sh). The launcher and the job's own log live in
## results/train/pair_<a>+<b>/; each run's log in its own RUN_DIR (train.log-<jobid>).
spec_row() {    # <run> -> "<family> <robot> <args...>" from chart_runs.txt, or fail
    _rows | awk -v r="$1" '$3==r {f=$1; b=$2; $1=$2=$3=""; sub(/^ +/, ""); print f, b, $0; n++}
                           END {exit n == 1 ? 0 : 1}'
}

submit_pair() {     # <run_a> <run_b> <walltime>: two rows of chart_runs.txt
    local A="$1" B="$2" WALL="$3" rowa rowb fa ra aa fb rb ab
    rowa=$(spec_row "$A") || { echo "REFUSING: $A is not exactly one row of $SPEC" >&2; return 2; }
    rowb=$(spec_row "$B") || { echo "REFUSING: $B is not exactly one row of $SPEC" >&2; return 2; }
    read -r fa ra aa <<< "$rowa"
    read -r fb rb ab <<< "$rowb"
    [ "$fa" = "$fb" ] || { echo "REFUSING: $A ($fa) and $B ($fb) are different families" >&2; return 2; }
    submit_pair_runs "$fa" "$A" "$ra" "$aa" "$B" "$rb" "$ab" "$WALL"
}

submit_pair_runs() {   # <family> <run_a> <robot_a> <args_a> <run_b> <robot_b> <args_b> <walltime>
    local fa="$1" A="$2" ra="$3" aa="$4" B="$5" rb="$6" ab="$7" WALL="$8"
    [ "$A" != "$B" ] || { echo "REFUSING: a pair of one run would race itself" >&2; return 2; }
    refuse_if_live "$A" "$B" || return $?
    require_exec train_chart.sh || return $?
    require_exec train_chart_pair.sh || return $?

    local PAIR="$A+$B"
    local PAIR_DIR_R="\$HOME/$SC_ROOT/results/train/pair_$PAIR"    # expanded remotely
    echo "Submitting pair: $A (robot $ra, '$aa') + $B (robot $rb, '$ab') FAMILY=$fa WALL=$WALL PARTITION=$PARTITION DEP='${DEPENDENCY:-none}' EXCLUDE='${EXCLUDE_NODES:-none}'"
    sc_run "mkdir -p $PAIR_DIR_R && cat > $PAIR_DIR_R/launch.sh && chmod +x $PAIR_DIR_R/launch.sh" <<LAUNCH
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:2
#SBATCH --partition=$PARTITION
#SBATCH --time=$WALL
#SBATCH --job-name=${SC_JOB_PREFIX}_trainc_$PAIR${DEP_DIRECTIVE:+
#SBATCH --dependency=$DEPENDENCY}${EXC_DIRECTIVE:+
#SBATCH --exclude=$EXCLUDE_NODES}
#SBATCH --output=../results/train/pair_$PAIR/launch.log-%j
# Generated by cluster/submit_chart_train.sh $(date -Is) -- do not edit; resubmit instead.
srun --ntasks=1 --kill-on-bad-exit=1 "\$HOME/$SC_ROOT/repo/cluster/train_chart_pair.sh"
LAUNCH

    sc_run "cd ~/$SC_ROOT/repo && $(FAMILY="$fa" common_env) \
      RUN_A='$A' ROBOT_A='$ra' ARGS_A='$aa' RUN_B='$B' ROBOT_B='$rb' ARGS_B='$ab' \
      LLsub $PAIR_DIR_R/launch.sh"
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
## Every row of chart_runs.txt, each through the same-run guard: one job per row, or with
## PAIRS=1 one job per two consecutive rows (the file lists each robot's two rungs together, so
## a pair is one robot). A row whose status.json already reached MAX_STEPS is skipped
## (resubmitting it would only resume, find the step budget spent and re-export); under PAIRS=1
## a pair with one finished row submits the other alone. One remote call decides that.
## ONLY=<regex> restricts --all to the rows whose RUN NAME matches (awk ERE, on the third
## column): e.g. ONLY='_cvaef[0-9]+$' submits exactly the eight fixed-variance cVAE rows and
## never the learned-variance ones. A regex that matches no row refuses. Pairing (PAIRS=1) is
## over consecutive MATCHING rows, so keep each robot's two rungs adjacent in chart_runs.txt.
_all_rows() {
    if [ -n "${ONLY:-}" ]; then _rows | awk -v re="$ONLY" '$3 ~ re'; else _rows; fi
}

cmd_all() {
    local WALL="$1" done_list family robot name args rc fails=0 pending=()
    if [ -z "$(_all_rows)" ]; then
        echo "REFUSING: ONLY='${ONLY:-}' matches no run name in $SPEC" >&2
        return 2
    fi
    echo "--all over: $(_all_rows | awk '{print $3}' | tr '\n' ' ')"
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
        if [ "${PAIRS:-0}" = "1" ]; then
            pending+=("$name")
            [ "${#pending[@]}" -lt 2 ] && continue
            submit_pair "${pending[0]}" "${pending[1]}" "$WALL" </dev/null
            rc=$?
            [ "$rc" -eq 0 ] || { echo "  -> pair ${pending[*]} NOT submitted (rc=$rc)"; fails=$((fails + 1)); }
            pending=()
        else
            FAMILY="$family" ROBOT="$robot" submit_one "$name" "$WALL" "$args" </dev/null
            rc=$?
            [ "$rc" -eq 0 ] || { echo "  -> $name NOT submitted (rc=$rc)"; fails=$((fails + 1)); }
        fi
    done < <(_all_rows)
    if [ "${#pending[@]}" -eq 1 ]; then         # an odd row out, alone
        read -r family robot _ args <<< "$(_rows | awk -v r="${pending[0]}" '$3==r')"
        FAMILY="$family" ROBOT="$robot" submit_one "${pending[0]}" "$WALL" "$args" </dev/null \
            || { echo "  -> ${pending[0]} NOT submitted"; fails=$((fails + 1)); }
    fi
    [ "$fails" -eq 0 ] || { echo "$fails submission(s) refused" >&2; return 3; }
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

## The pair mode's own plumbing check: two smoke runs in one 2-GPU job, same settings as --smoke.
## Run it before PAIRS=1 --all, never the real run names with a small MAX_STEPS -- a real run
## resumes from whatever last.ckpt its RUN_DIR holds, so a 200-step smoke there would BECOME
## the start of the 620k-step run.
cmd_smoke_pair() {
    local robot="${1:?usage: --smoke-pair <robot> <K_a> <K_b>}" ka="${2:?}" kb="${3:?}"
    local extra="--checkpoint_every=100 --eval_every=100 --val_set_size=20 --disable_wandb"
    echo "200-step cVAE PAIR smoke for $robot at K=$ka and K=$kb (one job, both GPUs)."
    MAX_STEPS=200 FINAL_STEP=200 RUN_EXPORT=1 EXPORT_TO_RUN_DIR=1 \
        submit_pair_runs cvae "smoke_${robot}_cvae${ka}" "$robot" "--dim_latent_space=$ka $extra" \
                              "smoke_${robot}_cvae${kb}" "$robot" "--dim_latent_space=$kb $extra" 00:30:00
}

case "${1:-}" in
    --status) cmd_status ;;
    --all)    cmd_all "${2:-24:00:00}" ;;
    --smoke)  shift; cmd_smoke "$@" ;;
    --smoke-pair) shift; cmd_smoke_pair "$@" ;;
    --pair)   submit_pair "${2:?usage: --pair <run_a> <run_b> [walltime]}" \
                          "${3:?usage: --pair <run_a> <run_b> [walltime]}" "${4:-24:00:00}" ;;
    ""|-h|--help)
        sed -n '2,19p' "$0"; exit 2 ;;
    *)
        RUN_NAME="$1"
        WALL="${2:-24:00:00}"
        shift $(( $# > 2 ? 2 : $# ))
        [ "${1:-}" = "--" ] && shift
        submit_one "$RUN_NAME" "$WALL" "$*" ;;
esac
