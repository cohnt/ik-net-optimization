#!/bin/bash
# Poll status, and pull results off the cluster (storage there is NOT backed up).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
#   cluster/collect_results.sh --status            one read-only ssh: counts and sentinels
#   cluster/collect_results.sh --reclaim <manifest> clear claims with no done marker
#   cluster/collect_results.sh                     tar + rsync + extract + merge shards
#
# --status counts done markers with `find -name '*.done'` inside ONE state
# subdirectory per manifest. It never walks the results tree: filesystem-scan
# storms over a shared Lustre mount are the documented anti-pattern, and the
# done markers carry the same information at a fraction of the cost.
#
# --reclaim is the mop-up for items whose worker died mid-solve. run_items.sh
# claims an item by atomically creating <id>.claim and touches <id>.done only on
# success, so a claim with no done marker is an item nobody is finishing. This
# removes exactly those, after which re-submitting the same manifest picks them
# up. It deliberately will NOT run while any job is active: an item that is still
# genuinely running elsewhere must not be stolen.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"
REPO_ROOT="$(_sc_repo_root)"

if [ "${1:-}" = "--status" ]; then
    sc_run "cd ~/$SC_ROOT 2>/dev/null || { echo 'no ~/$SC_ROOT yet'; exit 0; }
echo '--- sentinels (setup / smoke / calibrate) ---'
for f in setup.DONE smoke.DONE calibrate.*.DONE; do
    [ -f \"\$f\" ] && printf '%-26s %s\n' \"\$f\" \"\$(cat \"\$f\")\"
done
echo '--- manifests: items done / claimed ---'
for d in state/*/; do
    [ -d \"\$d\" ] || continue
    printf '%-34s %4d done  %4d claimed\n' \"\$(basename \"\$d\")\" \\
        \"\$(find \"\$d\" -maxdepth 1 -name '*.done' | wc -l)\" \\
        \"\$(find \"\$d\" -maxdepth 1 -name '*.claim' -type d | wc -l)\"
done
echo '--- worker sentinels (most recent 20) ---'
find state -maxdepth 2 -name '*.SENTINEL' -printf '%T@ %p ' -exec head -1 {} \; \\
    | sort -n | tail -20 | cut -d' ' -f2-
echo '--- queue ---'
LLstat 2>/dev/null | head -20"
    exit 0
fi

if [ "${1:-}" = "--reclaim" ]; then
    MANIFEST_NAME="${2:?usage: --reclaim <manifest-basename>}"
    sc_run "cd ~/$SC_ROOT/state/$MANIFEST_NAME 2>/dev/null || { echo 'no such manifest state'; exit 1; }
RUNNING=\$(LLstat 2>/dev/null | grep -c RUNNI || true)
if [ \"\$RUNNING\" -gt 0 ]; then
    echo \"REFUSING: \$RUNNING job(s) still running -- a live item must not be stolen.\"
    echo 'Wait for the queue to drain (collect_results.sh --status), then retry.'
    exit 1
fi
N=0
for c in *.claim; do
    [ -d \"\$c\" ] || continue
    id=\${c%.claim}
    if [ ! -f \"\$id.done\" ]; then rm -rf \"\$c\"; N=\$((N+1)); echo \"reclaimed \$id\"; fi
done
echo \"\$N stale claim(s) cleared; re-submit the same manifest to mop up.\""
    exit 0
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="$SC_ROOT/collect_$STAMP.tar"
STAGING="$REPO_ROOT/results/_cluster_staging/$STAMP"
mkdir -p "$STAGING"

# No gzip: the payload is JSON summaries plus already-compressed logs, and the
# login node should not spend CPU on a transfer that is I/O bound anyway.
sc_run "cd ~/$SC_ROOT && tar cf 'collect_$STAMP.tar' results state calib 2>/dev/null; ls -lh 'collect_$STAMP.tar'"
sc_rsync -a --info=progress2 "$SC_DEST:$ARCHIVE" "$STAGING/"
tar xf "$STAGING/collect_$STAMP.tar" -C "$STAGING"
echo "extracted to $STAGING"

# The remote archive is deliberately left in place; nothing here deletes cluster
# data. Merge shards in the staging tree, then review before promoting anything
# into results/ proper -- staging does not match collate.py's glob, so a
# half-collected campaign cannot silently enter a table.
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/cluster/merge_shard_summaries.py" "$STAGING"
echo
echo "review, then promote with:  cp -r $STAGING/results/<robot>/benchmark/<tag> results/<robot>/benchmark/"
