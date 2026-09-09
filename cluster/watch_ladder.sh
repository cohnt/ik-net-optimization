#!/bin/bash
# Watch the chart ladder and keep wandb current. Emits one line per notable event on stdout.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Intended to be run under a Monitor, which turns each stdout line into a notification:
#   bash cluster/watch_ladder.sh
#
# Two jobs, both of which otherwise need a human:
#   1. WANDB SYNC. Compute nodes are offline (WANDB_MODE=offline), so the dashboard only
#      advances when someone rsyncs the run dir down and `wandb sync`s it. Nothing is lost
#      while nobody syncs -- the cluster-side file keeps accumulating with correct
#      timestamps and one later sync backfills the gap -- but the dashboard is stale until
#      then. This loop syncs the running rung every pass.
#   2. EVENTS. Rung transitions, step milestones, and terminal failures.
#
# Emission is deliberately sparse -- every line becomes a notification. Rung changes,
# every STEP_EVERY steps, and anything that ends badly.
#
# Poll interval is long (default 30 min) because this holds an ssh connection to a shared
# login node, and because nothing here changes on a minute timescale: a rung runs for
# 15-25 hours.
set -uo pipefail
cd "$(dirname "$0")/.."
source cluster/ssh_common.sh

INTERVAL="${INTERVAL:-1800}"
STEP_EVERY="${STEP_EVERY:-100000}"
PY=.venv/bin/python

last_rung=""
last_milestone=0

sync_wandb() {
    local rung="$1"
    mkdir -p "results/train/$rung"
    sc_rsync -az "$SC_DEST:learned-ik/results/train/$rung/wandb" "results/train/$rung/" >/dev/null 2>&1 || return 0
    for d in results/train/"$rung"/wandb/offline-run-*; do
        [ -d "$d" ] || continue
        # --legacy is REQUIRED while the run is live: the newest record in an open .wandb
        # file is half-written and the current engine aborts on it. Sync is incremental and
        # idempotent (position kept in .syncstate), so repeated calls are cheap and safe.
        ( cd results && "../$PY" -m wandb sync --legacy "${d#results/}" ) >/dev/null 2>&1
    done
}

while true; do
    # The running rung, by job name. LLstat truncates NAME to 15 chars, so ask squeue.
    rung=$(sc_run 'squeue -u tcohn -h -t RUNNING -o "%j" 2>/dev/null | grep "^lik_train_" | head -1' 2>/dev/null \
           | tr -d '\r\n' | sed 's/^lik_train_//')
    pending=$(sc_run 'squeue -u tcohn -h -t PENDING -o "%j" 2>/dev/null | grep -c "^lik_train_"' 2>/dev/null | tr -dc '0-9')

    if [ -z "$rung" ]; then
        if [ "${pending:-0}" -eq 0 ]; then
            echo "ladder: nothing running and nothing pending -- chain finished or drained. Check submit_ladder.sh --status"
            break
        fi
        # Held or waiting on a dependency; not worth a notification every pass.
        sleep "$INTERVAL"; continue
    fi

    if [ "$rung" != "$last_rung" ]; then
        echo "ladder: now training '$rung' ($pending rung(s) still queued)"
        last_rung="$rung"; last_milestone=0
    fi

    sync_wandb "$rung"

    read -r step polemax polefrac vall2 <<<"$(sc_run "cat ~/$SC_ROOT/results/train/$rung/status.json 2>/dev/null" 2>/dev/null \
        | $PY -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print("0 0 0 0"); raise SystemExit
p=d.get("pole") or {}
print(d.get("global_step",0), p.get("pole/max",-1), p.get("pole/frac_gt_1000",-1), d.get("val_l2_error",-1))
' 2>/dev/null)"

    if [ -n "${step:-}" ] && [ "${step:-0}" -ge $((last_milestone + STEP_EVERY)) ]; then
        echo "ladder: $rung at step $step | pole/max=$polemax frac_gt_1000=$polefrac val_l2=$vall2 (wandb synced)"
        last_milestone=$step
    fi

    sleep "$INTERVAL"
done
