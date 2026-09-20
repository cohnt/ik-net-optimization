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
# Only this project's workers can be writing into ~/learned-ik/results, and the
# SuperCloud account is shared with Thomas's other projects -- a broad any-job-running
# check refuses whenever an unrelated campaign is on the cluster, which is most of the
# time (it fired on a run_matrix.sh job belonging to another project). So filter by job
# name, and count PENDING too: a queued worker could start part way through and write
# into a directory already archived.
#
# THE NAME MUST BE THE ONE SLURM ACTUALLY SEES, which is submit_bench.sh's
# `--job-name=lik_bench_${MANIFEST%.txt}` -- NOT the payload script's filename. This
# guard spent its whole life matching `run_items.sh`, a name no job has ever carried, so
# BUSY was unconditionally 0 and the guard never refused anything: --reclaim would have
# happily stolen items from 32 live workers. Found 2026-09-20 mid-campaign, by running the
# same filter by hand against a queue known to hold four running jobs and getting nothing
# back. Scoping to THIS manifest is also tighter than the original intent -- a worker only
# ever touches the manifest it was handed, so an unrelated learned-ik campaign is no
# reason to refuse. Two lessons: a guard that cannot be observed refusing has not been
# tested, and a job's name is a property of the submitter, not of the script it runs.
    sc_run "cd ~/$SC_ROOT/state/$MANIFEST_NAME 2>/dev/null || { echo 'no such manifest state'; exit 1; }
JOB_NAME=lik_bench_\${MANIFEST_NAME%.txt}
BUSY=\$(squeue -u \$USER -h -n \"\$JOB_NAME\" -t RUNNING,PENDING 2>/dev/null | wc -l)
if [ \"\$BUSY\" -gt 0 ]; then
    echo \"REFUSING: \$BUSY \$JOB_NAME job(s) queued or running -- a live item must not be stolen.\"
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

## The watermark for the NEXT run is taken NOW, before tar reads anything -- not at the
## end of this one. Stamping it at the end opens a window: tar snapshots the tree, the
## transfer and merge take minutes, and anything a worker writes in the meantime is both
## too late for this archive and older than the watermark, so no later run ever fetches it.
## It bit on 2026-09-14: two shards of stage 2 were written at 20:54:30 while the 20:54
## collection was in flight, and the next collection reported their runs INCOMPLETE with
## the files sitting on the cluster the whole time.
## Taking it from the CLUSTER's clock, because the mtimes tar compares it against are the
## cluster's. Erring early only re-ships a few files that were already collected, which
## costs a little bandwidth and is idempotent; erring late loses them silently.
COLLECT_WATERMARK="$(sc_run "date +%s")"

# No gzip: the payload is JSON summaries plus already-compressed logs, and the
# login node should not spend CPU on a transfer that is I/O bound anyway.
# Training checkpoints are NOT collected.  `results/` held nothing but benchmark output
# when this script was written; the chart ladder added results/train/<run>/{checkpoints,pkl},
# which is 81 GB of .ckpt plus exported .pkl and grows with every rung.  A routine
# incremental collection on 2026-09-12 therefore built a 57 GB archive and timed out twice
# before it transferred anything.  Everything worth having from a training run is small --
# status.json, metrics/, the launch log that carries the screening output, wandb/ -- the
# weights are regenerable on the cluster from the .ckpt, and the one exported .pkl a
# benchmark actually loads travels in repo/models/.  Excluded outright rather than gated
# behind --full: there is no reason to want 81 GB of checkpoints on a laptop.
#
# The patterns are single-quoted so the REMOTE shell hands them to tar verbatim.  Unquoted
# they glob against the directory we just cd'd into, expand to the real subdirectory list,
# and the --exclude silently covers only the first of them.
TRAIN_EXCLUDES="--exclude='results/train/*/checkpoints' --exclude='results/train/*/pkl'"
sc_run "cd ~/$SC_ROOT && tar cf 'collect_$STAMP.tar' $TRAIN_EXCLUDES $NEWER results calib 2>/dev/null; ls -lh 'collect_$STAMP.tar'"
sc_rsync -a --info=progress2 "$SC_DEST:$ARCHIVE" "$STAGING/"
# The local archive is redundant the moment it is extracted: `tar xf` unpacks it directly
# beside itself, so keeping it stores every collection's payload twice. That went unnoticed
# until 2026-09-13, when `results/_cluster_staging` had reached 48 GB on the laptop -- a
# 24 GB archive from the 09-09 collection sitting next to its own 24 GB extraction, both of
# them almost entirely the training checkpoints the exclusion above now keeps out anyway.
#
# Guarded on the extract rather than run unconditionally: this script sets `-uo pipefail`
# but NOT `-e`, so a failed `tar xf` otherwise falls straight through and deletes the only
# copy of an archive that would have to be re-transferred to diagnose.
if tar xf "$STAGING/collect_$STAMP.tar" -C "$STAGING"; then
    rm -f "$STAGING/collect_$STAMP.tar"
    echo "extracted to $STAGING"
else
    echo "extract FAILED -- keeping $STAGING/collect_$STAMP.tar for diagnosis" >&2
    exit 4
fi

# The archive this run just built is removed once it has been safely extracted locally --
# it is this script's own scratch file rather than cluster data, and leaving every one of
# them behind had accumulated 177 GB of superseded tars by 2026-09-12.  Nothing else here
# deletes anything on the cluster.  Merge shards in the staging tree, then review before
# promoting anything into results/ proper -- staging does not match collate.py's glob, so
# a half-collected campaign cannot silently enter a table.
sc_run "rm -f ~/$ARCHIVE"
## Search EVERY earlier staging directory, not just the most recent one. Collection is
## incremental, so a collection that runs while a stage is in flight splits that stage, and
## any run whose shards straddle a split is unmergeable from either directory alone -- it
## looks exactly like data loss while the shards sit on disk. `--also` adds a directory to
## the SEARCH and is `action="append"`, so it takes as many as we give it; the merged run is
## written beside the shard that anchors it, so read the merger's own path when promoting
## rather than assuming this collection's staging directory.
##
## It used to pass only the single most recent prior directory, which covers a two-way split
## and nothing more. Stage STATUSQUO broke that on 2026-09-20: NLopt items ran 27-144 min
## while collections ran hourly, so one row's 24 shards landed across THREE collections and
## the merger reported 22 of 24 "missing" -- from shard directories that existed, because
## incremental rsync creates the directory and skips a `summary.json` it already shipped.
## Passing every prior directory removes the failure mode rather than widening it by one.
## Verified when the fix landed: re-merging the seven rows that had already merged normally
## reproduced all seven exactly, so a wider search changes nothing but what it can find.
##
## Restricted to TIMESTAMP-shaped directory names, which also keeps a hand-made directory in
## the staging tree from being treated as a collection. That bit once, on 2026-09-17: a
## manual merge directory named `manualmerge-row12` sorted after every `20260917-*` and
## became "the previous collection", so two straddled rows of stage SNOPTTUNE went unmerged
## with all sixteen shards on disk.
ALSO=()
while IFS= read -r d; do
    [ -n "$d" ] || continue
    ALSO+=(--also "$d")
done < <(ls -1d "$REPO_ROOT"/results/_cluster_staging/[0-9]*-[0-9]*/ 2>/dev/null \
         | grep -v "^$STAGING/\?$")
## ALSO holds two elements per directory (the flag and the path), hence the halving.
echo "merging: searching $STAGING plus $(( ${#ALSO[@]} / 2 )) prior staging directory/ies"
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/cluster/merge_shard_summaries.py" "$STAGING" "${ALSO[@]}"

# Only now is it safe to advance the incremental watermark: everything above has to
# have succeeded, or the next run must re-fetch what this one failed to bring back.
sc_run "echo $COLLECT_WATERMARK > ~/$SC_ROOT/.last_collect"
echo
echo "review, then promote with:  cp -r $STAGING/results/<robot>/benchmark/<tag> results/<robot>/benchmark/"
