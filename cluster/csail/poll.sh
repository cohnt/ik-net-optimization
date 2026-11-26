#!/bin/bash
# Read-only status of a staged commit's campaign on CSAIL. One cs_run; poll every 5-10 minutes at most.
#
#   bash cluster/csail/poll.sh [<12-hex commit> [<manifest basename>]]      (default: HEAD, all manifests)
#
# Prints: our lik_* jobs in the queue (with reason, so a PENDING task says why), sacct attempts (REQUEUED rows are
# preemptions), "!!" lines for loud exits (NOTHING CLAIMED, CANNOT DETERMINE JOB END TIME),
# per-manifest items done / claimed, HARDWARE, the last lines of the newest job logs, and the machine
# records (node / GPU / CPU model) of every attempt. Only OUR jobs are listed (--me, name prefix lik_).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/csail_common.sh" || exit 1
csail_connected || exit $?
REPO_ROOT="$(csail_repo_root)"
C12="${1:-$(git -C "$REPO_ROOT" rev-parse HEAD | cut -c1-12)}"; ONLY="${2:-}"
[[ "$C12" =~ ^[0-9a-f]{12}$ ]] || { echo "bad commit id" >&2; exit 2; }
[[ -z "$ONLY" || "$ONLY" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad manifest name" >&2; exit 2; }
ROOT="$(csail_tree_root "$C12")"
cs_run "cd $ROOT 2>/dev/null || { echo no tree $ROOT; exit 0; }
echo '--- queue (ours, lik_*) ---'
squeue --me -h -o '%.12i %.9T %.10M %.22j %R' | grep -E ' lik_' | head -60
echo '--- sacct, last 2 days (REQUEUED = preempted) ---'
sacct -X -S now-2days -o JobID%16,JobName%28,Partition%20,State%12,Elapsed,NodeList%18,ExitCode --state=R,PD,CD,F,RQ,CA,TO,NF,PR 2>/dev/null | grep -E 'lik_|JobID|----' | tail -60
echo '--- loud exits: a task that claimed nothing while items remain (rc 4), or ran blind of its walltime (rc 5) ---'
grep -H -E 'NOTHING CLAIMED|CANNOT DETERMINE JOB END TIME' logs/*.out 2>/dev/null | sed 's/^/!! /' | tail -20
echo '--- sentinels ---'
ls check.*.DONE 2>/dev/null | while read f; do echo \"\$f: \$(cat \$f)\"; done
echo '--- manifests: items done / claimed / HARDWARE ---'
for d in state/*${ONLY}*/; do [ -d \"\$d\" ] || continue
  printf '%-44s %4d done %4d claimed  hw=%s\n' \"\$(basename \$d)\" \"\$(find \$d -maxdepth 1 -name '*.done' | wc -l)\" \"\$(find \$d -maxdepth 1 -name '*.claim' -type d | wc -l)\" \"\$(cat \$d/HARDWARE 2>/dev/null)\"; done
echo '--- machine records (node, GPU, CPU model), newest 12 ---'
ls -t logs/*/machine.a*.txt 2>/dev/null | head -12 | while read f; do echo \"== \$f\"; grep -E '^(host|cpu:|gpu:)' \$f; done
echo '--- newest job logs (tail) ---'
for f in \$(ls -t logs/*.out 2>/dev/null | head -3); do echo \"== \$f\"; tail -4 \$f; done"
