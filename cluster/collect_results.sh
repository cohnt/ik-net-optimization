#!/bin/bash
# Poll status, and pull results off the cluster (storage there is NOT backed up).
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
#   cluster/collect_results.sh --status            one read-only ssh: counts and sentinels
#   cluster/collect_results.sh --reclaim <manifest> clear claims with no done marker
#   cluster/collect_results.sh                     tar + rsync + extract + merge shards,
#                                                  INCREMENTAL since the last success
#   cluster/collect_results.sh --full              the same, but the whole results tree
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

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="$SC_ROOT/collect_$STAMP.tar"
STAGING="$REPO_ROOT/results/_cluster_staging/$STAMP"
mkdir -p "$STAGING"

# Incremental by default. The archive used to be `tar cf ... results state calib`
# every time, so each collection re-shipped the whole cumulative campaign to pick up
# one stage: by Stage G that was 40,733 files and 950 MB for ~150 MB of new data, and
# it took thirty minutes because a shared Lustre mount is metadata-bound on files that
# size. `.last_collect` on the cluster records the previous successful collection, and
# tar's --newer-mtime ships only what has been written since.
#
# The stamp is advanced ONLY after the local extract and merge succeed (see below), so
# a transfer that dies half way is retried in full rather than silently skipping the
# data it did not fetch. `--full` forces the complete archive.
#
# `state/` is deliberately not shipped: its done-markers and claim directories are
# load-bearing for resume ON THE CLUSTER and are never read locally, and they are
# several thousand near-empty inodes.
NEWER=""
if [ "$FULL" = "0" ]; then
    LAST="$(sc_run "cat ~/$SC_ROOT/.last_collect 2>/dev/null" || true)"
    if [ -n "$LAST" ]; then
        NEWER="--newer-mtime=@$LAST"
        echo "incremental collection: everything written since $(date -d "@$LAST" 2>/dev/null || echo "@$LAST")"
    else
        echo "no .last_collect on the cluster -- collecting in full this once"
    fi
else
    echo "--full: collecting the entire results tree"
fi

# No gzip: the payload is JSON summaries plus already-compressed logs, and the
# login node should not spend CPU on a transfer that is I/O bound anyway.
sc_run "cd ~/$SC_ROOT && tar cf 'collect_$STAMP.tar' $NEWER results calib 2>/dev/null; ls -lh 'collect_$STAMP.tar'"
sc_rsync -a --info=progress2 "$SC_DEST:$ARCHIVE" "$STAGING/"
tar xf "$STAGING/collect_$STAMP.tar" -C "$STAGING"
echo "extracted to $STAGING"

# The remote archive is deliberately left in place; nothing here deletes cluster
# data. Merge shards in the staging tree, then review before promoting anything
# into results/ proper -- staging does not match collate.py's glob, so a
# half-collected campaign cannot silently enter a table.
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/cluster/merge_shard_summaries.py" "$STAGING"

# Only now is it safe to advance the incremental watermark: everything above has to
# have succeeded, or the next run must re-fetch what this one failed to bring back.
sc_run "date +%s > ~/$SC_ROOT/.last_collect"
echo
echo "review, then promote with:  cp -r $STAGING/results/<robot>/benchmark/<tag> results/<robot>/benchmark/"
