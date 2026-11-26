#!/bin/bash
# Poll Engaging jobs and item progress (laptop side, read-only).
#
#   bash cluster/engaging/poll.sh                 one small call: my queue, sentinels, per-manifest done/claimed
#   bash cluster/engaging/poll.sh wait <jobid>    loop (every 5 min, a single small call each) until the job
#                                                 (all array tasks) has left the queue, or the connection is gone
#
# The shared connection is never reopened: when it is down this stops and says so (ask Thomas).
set -uo pipefail
source "$(dirname "$0")/eg_common.sh" || exit 1
SDIR="$EG_REPO/logs/staged"
COMMIT="${COMMIT:-$(cat "$SDIR/engaging-last-commit" 2>/dev/null)}"
CROOT=$(eg_commit_root "${COMMIT:-none0000000000}")

status() {
    eg_run "date; echo '--- queue (mine) ---'; squeue --me -o '%.12i %.9P %.18j %.2t %.10M %.6D %R'
echo '--- sentinels ---'; for f in ~/$EG_BASE/build.DONE ~/$EG_BASE/plumb.DONE; do [ -f \$f ] && printf '%s: %s\n' \$(basename \$f) \"\$(cat \$f)\"; done
cd ~/$CROOT 2>/dev/null || { echo 'no tree for $COMMIT'; exit 0; }
echo '--- manifests (items done / claimed) ---'
for d in state/*/; do [ -d \$d ] || continue
  printf '%-46s %4d done %4d claimed\n' \$(basename \$d) \$(find \$d -maxdepth 1 -name '*.done' | wc -l) \$(find \$d -maxdepth 1 -name '*.claim' -type d | wc -l)
done
echo '--- hardware seen ---'; cat state/*/hardware.log 2>/dev/null | sed 's/ [0-9T:-]*-0[45]:00\$//' | sort | uniq -c | tail -8
echo '--- kills ---'; cat state/*/kills.log 2>/dev/null | grep -vc WITHDRAWN"
}

if [ "${1:-}" = "wait" ]; then
    J="${2:?usage: poll.sh wait <jobid>}"
    while eg_status; do
        ## A failed squeue (slurmctld socket timeouts happen) is NOT "left the queue": only an OK, empty answer is.
        out=$(eg_run "squeue -h -j $J -o '%i %t %M %R' 2>&1; test \${PIPESTATUS[0]} -eq 0") || { echo "$(date -Is) squeue failed ($out); retrying"; sleep 60; continue; }
        [ -n "$out" ] || { echo "job $J has left the queue"; break; }
        printf '%s\n%s\n' "$(date -Is)" "$out"
        sleep 300
    done
    eg_status || { echo "SESSION DOWN: tell Thomas; do NOT log in" >&2; exit 255; }
    eg_run "sacct -j $J -X -o JobID,JobName%20,State,Elapsed,NodeList,ExitCode"
    exit 0
fi
eg_require_connection || exit 255
status
