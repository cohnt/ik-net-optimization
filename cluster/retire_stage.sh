#!/bin/bash
# Stop a stage's jobs early, when a decision has been taken to abandon the rest of it.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# Retiring a stage THROWS AWAY queued measurement. It is Thomas's call, never
# an agent's, and this script is the record of that call being carried out.
# ===========================================================================
#
#   cluster/retire_stage.sh <manifest> [--groups a,b,c]         DRY RUN: report only
#   cluster/retire_stage.sh <manifest> [--groups a,b,c] --yes   actually scancel
#
# <manifest> is a name under cluster/, with or without the .txt (e.g.
# manifest_stageSOFTCAP). --groups takes substrings of item ids and reports
# done/claimed/pending per group, which is how you confirm that the PART of the
# stage you mean to keep has finished before abandoning the rest. For a cap
# ladder those are the caps: --groups 90,180,360.
#
# WHY THIS EXISTS RATHER THAN AN ssh scancel. Two reasons, both learned here.
#
# The first is scoping. This account is shared between projects and agents, and
# the record's own post-mortem is that a cluster-wide action must be scoped to
# THIS project's jobs BY JOB NAME -- the one time that was got wrong, the guard
# was unconditionally 0 and never refused for its whole life. So this script
# never takes a job id. It derives the job NAME from the manifest exactly as
# cluster/submit_bench.sh constructs it (`lik_bench_<stem>`, carrying the `lik_`
# prefix that distinguishes this tree from a sibling like the screw-joint arm's),
# and hands that name to `scancel --name --user`, which filters server-side. A
# mistyped id cannot reach another project's job because no id is ever typed.
#
# The second is that the dry run is the point. It prints the jobs it would kill,
# the jobs it is deliberately NOT touching, and the per-group state counts, so
# the scoping is OBSERVED working before anything is cancelled rather than
# asserted in a comment.
#
# WHAT RETIRING COSTS, stated because it is easy to forget. Items in flight when
# the jobs die keep their `<id>.claim` and never get an `<id>.done`. That is the
# documented dead-item state, and it is honest: `--status` will show the stage
# short of its item count, which is exactly what happened. Nothing is corrupted
# and nothing already finished is lost -- every completed item's summary is
# already on cluster disk awaiting `collect_results.sh`. To run the abandoned
# items later, `collect_results.sh --reclaim <manifest>` clears the dead claims
# and resubmitting the same manifest picks them up.
#
# DO NOT instead pre-create `.done` markers for the items you want skipped. It
# is tempting -- workers skip a marked item cleanly and the jobs exit by
# themselves -- but it makes the state lie: the stage would read 192/192 done
# with half its summaries missing, and a later reader could not tell a retired
# item from a finished one. A `.done` that means "never ran" is precisely the
# check that silently stops running.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster/ssh_common.sh
source "$HERE/ssh_common.sh"

MANIFEST=""
RS_GROUPS=""
CONFIRM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --groups) RS_GROUPS="${2:?--groups needs a comma-separated list}"; shift 2 ;;
        --yes)    CONFIRM=1; shift ;;
        -*)       echo "retire_stage.sh: unknown flag $1" >&2; exit 2 ;;
        *)        [ -z "$MANIFEST" ] || { echo "retire_stage.sh: one manifest only" >&2; exit 2; }
                  MANIFEST="$1"; shift ;;
    esac
done
[ -n "$MANIFEST" ] || { echo "usage: retire_stage.sh <manifest> [--groups a,b,c] [--yes]" >&2; exit 2; }

STEM="${MANIFEST%.txt}"
LOCAL_MANIFEST="$HERE/$STEM.txt"
[ -f "$LOCAL_MANIFEST" ] || { echo "retire_stage.sh: no manifest at $LOCAL_MANIFEST" >&2; exit 2; }

## The job name, derived the same way submit_bench.sh builds it. This is the
## scoping handle: it carries the project prefix AND the stage.
JOB_NAME="lik_bench_$STEM"

## One ssh for everything the operator needs to see. Read-only.
REMOTE=$(cat <<REMOTE_EOF
set -u
cd "\$HOME/$SC_ROOT" 2>/dev/null || { echo "NOROOT"; exit 0; }
echo "---MINE---"
squeue -h -u "\$USER" -n "$JOB_NAME" -o '%i %j %T %M' 2>/dev/null
echo "---OTHERS---"
squeue -h -u "\$USER" -o '%i %j %T' 2>/dev/null | grep -v " $JOB_NAME " || true
echo "---STATE---"
if [ -d "state/$STEM" ]; then
    ls "state/$STEM" 2>/dev/null | grep '\.done\$'  | sed 's/\.done\$//'  > /tmp/.rs_done.\$\$ || true
    ls "state/$STEM" 2>/dev/null | grep '\.claim\$' | sed 's/\.claim\$//' > /tmp/.rs_claim.\$\$ || true
    echo "DONE=\$(wc -l < /tmp/.rs_done.\$\$) CLAIM=\$(wc -l < /tmp/.rs_claim.\$\$)"
    echo "---DONEIDS---"; cat /tmp/.rs_done.\$\$
    echo "---CLAIMIDS---"; cat /tmp/.rs_claim.\$\$
    rm -f /tmp/.rs_done.\$\$ /tmp/.rs_claim.\$\$
else
    echo "DONE=0 CLAIM=0"; echo "---DONEIDS---"; echo "---CLAIMIDS---"
fi
REMOTE_EOF
)

OUT="$(sc_run "$REMOTE")" || { echo "retire_stage.sh: status ssh failed" >&2; exit 1; }
case "$OUT" in NOROOT*) echo "retire_stage.sh: no ~/$SC_ROOT on the cluster" >&2; exit 1 ;; esac

Section() { printf '%s\n' "$OUT" | sed -n "/^---$1---\$/,/^---/p" | grep -v '^---'; }

MINE="$(Section MINE)"
OTHERS="$(Section OTHERS)"
DONE_IDS="$(Section DONEIDS)"
CLAIM_IDS="$(Section CLAIMIDS)"
NDONE=$(printf '%s\n' "$DONE_IDS" | grep -c . || true)
NCLAIM=$(printf '%s\n' "$CLAIM_IDS" | grep -c . || true)
NITEMS=$(grep -cvE '^[[:space:]]*(#|$)' "$LOCAL_MANIFEST")

echo "stage        : $STEM"
echo "job name     : $JOB_NAME   (the ONLY name this script can cancel)"
echo "manifest     : $NITEMS items"
echo "state        : $NDONE done, $NCLAIM claimed  ($((NCLAIM - NDONE)) in flight)"
echo

if [ -n "$RS_GROUPS" ]; then
    echo "per-group state (a group is a substring of an item id):"
    printf '  %-10s %7s %7s %7s\n' group items done "in flight"
    ## NOT named GROUPS. `GROUPS` is a bash SPECIAL variable -- an array of the
    ## invoking user's group ids -- and bash ignores assignments to it while
    ## keeping its own value, so `--groups 90,180,360` silently became the
    ## primary gid (1001) and every group counted zero items. `set -x` showed the
    ## assignment succeeding and the expansion returning something else, which is
    ## the signature of a special-variable collision rather than a parsing bug.
    unset _RS_SPLIT
    IFS=',' read -r -a _RS_SPLIT <<< "$RS_GROUPS"
    for g in "${_RS_SPLIT[@]}"; do
        gi=$(grep -vE '^[[:space:]]*(#|$)' "$LOCAL_MANIFEST" | cut -d'|' -f1 | grep -c "_${g}_" || true)
        gd=$(printf '%s\n' "$DONE_IDS"  | grep -c "_${g}_" || true)
        gc=$(printf '%s\n' "$CLAIM_IDS" | grep -c "_${g}_" || true)
        printf '  %-10s %7s %7s %7s\n' "$g" "$gi" "$gd" "$((gc - gd))"
    done
    echo
fi

if [ -z "$(printf '%s' "$MINE" | tr -d '[:space:]')" ]; then
    echo "REFUSING: no $JOB_NAME job is queued or running, so there is nothing to retire."
    echo "(If the stage is already finished, this is the expected answer.)"
    exit 3
fi

echo "WOULD CANCEL these jobs (name == $JOB_NAME):"
printf '%s\n' "$MINE" | sed 's/^/  /'
echo
echo "NOT TOUCHING these jobs of yours (different name):"
if [ -z "$(printf '%s' "$OTHERS" | tr -d '[:space:]')" ]; then
    echo "  (none)"
else
    printf '%s\n' "$OTHERS" | sed 's/^/  /'
fi
echo
echo "$((NCLAIM - NDONE)) item(s) in flight will die with a .claim and no .done."
echo "That is the documented dead-item state: honest, recoverable with"
echo "  cluster/collect_results.sh --reclaim $STEM"
echo "Completed items are unaffected and still need a normal collect_results.sh run."
echo

if [ "$CONFIRM" != 1 ]; then
    echo "DRY RUN. Nothing was cancelled. Re-run with --yes to act."
    exit 0
fi

echo "cancelling by name, scoped server-side to this user and this stage..."
sc_run "scancel --user=\"\$USER\" --name='$JOB_NAME' && echo SCANCEL_OK" || {
    echo "retire_stage.sh: scancel failed" >&2; exit 1; }
echo "done. Next: cluster/collect_results.sh to pull what DID finish."
