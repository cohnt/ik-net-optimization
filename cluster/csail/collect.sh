#!/bin/bash
# Collect a staged commit's results from CSAIL into results/ -- DATA ONLY, never code, never executed.
#
#   bash cluster/csail/collect.sh [<12-hex commit>]       (default: HEAD)
#
# What comes back: the published run directories (`results/*/benchmark/<tag>/`, which run_items.sh copies there when
# an item succeeds), the job logs and machine records (logs/), and the claim/done state of each manifest (done
# markers, kills.log; claims are listed, not needed). It is one tar of `results logs state`, streamed over the
# existing connection, written to a file, and LISTED before anything is extracted: any absolute path, `..`
# component, symlink, hardlink or device entry refuses the whole collection (a cluster-side file is data, and group
# scratch is writable by others). Extraction uses --no-same-owner --no-same-permissions. Nothing from the cluster is
# ever run, sourced or unpickled here; the merge step only reads JSON summaries.
#
# Then, as cluster/collect_results.sh does: merge shards (cluster/merge_shard_summaries.py) and promote into
# results/ (cluster/condense_staging.py --apply). logs/ and state/ go to results/_csail/<c12>/ (not a run tree).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/csail_common.sh" || exit 1
csail_connected || exit $?
REPO_ROOT="$(csail_repo_root)"
C12="${1:-$(git -C "$REPO_ROOT" rev-parse HEAD | cut -c1-12)}"
[[ "$C12" =~ ^[0-9a-f]{12}$ ]] || { echo "bad commit id" >&2; exit 2; }
ROOT="$(csail_tree_root "$C12")"
PY="${COLLECT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
## A git worktree has no .venv of its own: fall back to the main checkout's (the git common dir's parent).
if [ -z "${COLLECT_PYTHON:-}" ] && [ ! -x "$PY" ]; then
    PY="$(cd "$REPO_ROOT" && cd "$(git rev-parse --git-common-dir)/.." && pwd)/.venv/bin/python"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
STAGING="$REPO_ROOT/results/_cluster_staging/$STAMP"
KEEP="$REPO_ROOT/results/_csail/$C12"
mkdir -p "$STAGING" "$KEEP"
TAR="$STAGING/csail_collect.tar"

## The remote side lists names only inside ROOT, skipping the staged tree (repo/) and its per-run scratch.
cs_run "cd $ROOT && tar cf - results logs state 2>/dev/null" > "$TAR" || { echo "collection failed" >&2; rmdir "$STAGING" 2>/dev/null; exit 3; }
[ -s "$TAR" ] || { echo "empty archive (nothing under results/ logs/ state/ yet)" >&2; rm -f "$TAR"; rmdir "$STAGING" 2>/dev/null; exit 3; }

## Refuse unsafe entries BEFORE extracting.
BAD=$(tar tvf "$TAR" --quoting-style=escape 2>/dev/null | awk '
    { t = substr($1, 1, 1); name = $6; for (i = 7; i <= NF; i++) name = name " " $i }
    t != "-" && t != "d" { print "type " t ": " name; next }
    name ~ /^\// || name ~ /(^|\/)\.\.(\/|$)/ { print "path: " name }')
if [ -n "$BAD" ]; then
    echo "STOP: the archive holds entries that are not plain files or directories, or unsafe paths:" >&2
    printf '%s\n' "$BAD" | head -20 >&2
    echo "Do not extract it; tell Thomas (security: cluster-side files are untrusted data)." >&2
    exit 4
fi
tar xf "$TAR" -C "$STAGING" --no-same-owner --no-same-permissions || { echo "extract FAILED; kept $TAR" >&2; exit 4; }
rm -f "$TAR"
## logs/ and state/ are for reading, not a run tree.
for d in logs state; do [ -d "$STAGING/$d" ] && { mkdir -p "$KEEP/$STAMP"; mv "$STAGING/$d" "$KEEP/$STAMP/$d"; }; done
echo "extracted to $STAGING; logs/state kept in $KEEP/$STAMP"

ALSO=()
while IFS= read -r d; do [ -n "$d" ] && ALSO+=(--also "$d"); done < <(ls -1d "$REPO_ROOT"/results/_cluster_staging/[0-9]*-[0-9]*/ 2>/dev/null | grep -v "^$STAGING/\?$")
if [ -d "$STAGING/results" ]; then
    "$PY" "$REPO_ROOT/cluster/merge_shard_summaries.py" "$STAGING" "${ALSO[@]}"
    "$PY" "$REPO_ROOT/cluster/condense_staging.py" --apply
else
    echo "no results/ in the archive yet (no item has finished)"
fi
