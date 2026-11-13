#!/bin/bash
# Stage the repo and the gitignored model checkpoint onto SuperCloud.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Everything lands under ~/learned-ik/, this project's own tree. ~/ik-tune is a
# different project's and is never touched.
#
# Two things need explicit care:
#   - models/ is NOT excluded. The iiwa checkpoint
#     (iiwa14__lemon-haze-7__global_step_4.25M.pkl, 204 MB) is gitignored and
#     exists only on this workstation, and src/iiwa_program.py loads it by a path
#     relative to RepoDir(), so it has to travel with the tree. The panda weights
#     are different -- ikflow downloads those, which cluster/setup_supercloud.sh
#     does on the download partition, since compute nodes have no internet.
#   - results/ IS excluded. It is large, it is the output, and pushing a local
#     copy up would make it ambiguous which machine produced a summary.
set -uo pipefail
source "$(dirname "$0")/ssh_common.sh"
source "$(dirname "$0")/verify_staged_tree.sh"
REPO_ROOT="$(_sc_repo_root)"
## STAGE_SUBDIR (default `repo`, the live tree): `STAGE_SUBDIR=repo-next` stages the same tree
## BESIDE the live one, to be swapped in between two campaigns by `cluster/swap_staged_tree.sh`
## (a cluster job chained after the running campaign) -- the laptop-independent way to restage
## between rounds that are queued back to back (2026-10-16). The live-campaign guard below applies
## to the live tree only; nothing runs out of a side tree.
STAGE_SUBDIR="${STAGE_SUBDIR:-repo}"

COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
## Code integrity (README "Code integrity"): only a clean, committed tree is staged, so the cluster's
## code is exactly COMMIT. Commit first; there is no override.
DIRTY=$(git -C "$REPO_ROOT" status --porcelain)
if [ -n "$DIRTY" ]; then
    echo "REFUSING: the working tree is not clean; commit (or discard) first, then stage that commit." >&2
    printf '%s\n' "$DIRTY" | head -20 >&2
    exit 6
fi

## Refuse to restage under a live campaign unless explicitly forced. rsync has no
## --inplace here, so it renames over the tree and a RUNNING script keeps its own
## inode -- but any .py an item imports when it STARTS is re-read from disk, so a
## mid-stage restage silently produces one result set built from two code
## versions. That is the same hazard as changing a result schema mid-campaign,
## and it is invisible afterwards. Calibration and smoke jobs are exempt: they
## produce no campaign records.
## Scoped to THIS TREE, BY SUBMISSION DIRECTORY rather than by job name. The guard
## protects the tree whose code queued items re-read when they start, so a sibling
## campaign's jobs are none of its business -- an account-wide match had two campaigns
## refusing each other's staging for as long as either ran.
##
## `%Z` is each job's submission directory, and every submitter here invokes through
## `sc_run "cd ~/$SC_ROOT/repo && ..."`, so it identifies the tree BY CONSTRUCTION. Name
## matching cannot: `submit.sh` submits `LLsub ./cluster/run_items.sh` with no
## `--job-name`, so a whole benchmark campaign displays as its payload script name, and
## `submit_export.sh` names jobs `<prefix>_export_*`. A pattern over `(train|bench)` misses
## both, and would miss whatever submitter is added next -- enumerating the instances
## cannot cover the class. Checked on the live queue: this account's three trees
## (learned-ik, learned-ik-screw and another project's ik-tune) are cleanly separated by
## `%Z`, including for a job submitted with no name at all.
##
## Calibration and smoke stay exempt (`*_cal_*`, `smoke.sh`): they produce no campaign
## records. Everything else running out of this tree counts.
##
## A DELIBERATE WIDENING, so nobody is surprised by it: scoping by directory also catches
## DATASET jobs, which run out of this tree on the CPU partition and matched no previous
## pattern. That is correct -- datagen reads repo code through `ikflow_entry.py`, so
## restaging under one is the same hazard as restaging under a training run -- but it does
## refuse in a case the old guard let through.
##
## NEVER use LLstat for a programmatic check: it truncates NAME to 15 characters, so
## `lik_train_soft12_n4`, `lik_train_soft12_n8` and `lik_train_soft16_n6` all render as
## `lik_train_soft1` and every rung of a ladder collapses into one string. Measured.
RUNNING=$(sc_run "squeue -u \$USER -h -o '%Z %j' 2>/dev/null | awk -v d=\"\$HOME/$SC_ROOT/repo\" '\$1==d && \$2 !~ /_cal_/ && \$2 != \"smoke.sh\"' | wc -l" 2>/dev/null | tr -dc '0-9')
if [ "$STAGE_SUBDIR" = "repo" ] && [ -n "${RUNNING:-}" ] && [ "${RUNNING:-0}" -gt 0 ] && [ "${FORCE_STAGE:-0}" != "1" ]; then
    echo "REFUSING: $RUNNING campaign job(s) of THIS tree (~/$SC_ROOT) are running." >&2
    echo "Restaging would change the code later items import mid-stage." >&2
    echo "Wait for the stage to drain, or re-run with FORCE_STAGE=1 if you are sure." >&2
    exit 3
fi
[ "$SC_ROOT" = "learned-ik" ] || echo "NOTE: staging the ISOLATED tree ~/$SC_ROOT (job prefix '$SC_JOB_PREFIX')."

## An INCOMPLETE local tree is worse than a stale one, because the rsync below runs with
## --delete: anything missing here is deleted THERE. Two ways to arrive with one, both hit
## on 2026-09-16 while staging from a git worktree:
##
##   - a worktree does not check out submodules, so third_party/ikflow was empty locally and
##     rsync deleted the cluster's vendored fork down to 18 files;
##   - the gitignored checkpoints had been symlinked into the worktree, and rsync copies a
##     symlink AS a symlink, so three cluster-side .pkl files were replaced by links into a
##     path that does not exist there. The --filter=P rules protect them from deletion, not
##     from being overwritten with a link.
##
## Both are cheap to check and expensive to undo, so check.
if [ -f "$REPO_ROOT/.gitmodules" ]; then
    UNINIT=$(git -C "$REPO_ROOT" submodule status --recursive 2>/dev/null | grep -c "^-" || true)
    if [ "${UNINIT:-0}" -gt 0 ]; then
        echo "REFUSING: $UNINIT uninitialised submodule path(s) locally." >&2
        echo "rsync runs with --delete, so staging now would DELETE them on the cluster." >&2
        echo "Fix: git -C $REPO_ROOT submodule update --init --recursive" >&2
        exit 4
    fi
fi
LINKS=$(find "$REPO_ROOT/models" -type l 2>/dev/null | wc -l | tr -d " ")
if [ "${LINKS:-0}" -gt 0 ]; then
    echo "REFUSING: $LINKS symlink(s) under models/." >&2
    echo "rsync copies a symlink as a symlink, so the cluster would get links into a path" >&2
    echo "that does not exist there, replacing real checkpoints. Use hardlinks instead." >&2
    find "$REPO_ROOT/models" -type l >&2
    exit 5
fi

## PROTECT filters, not excludes: cluster-side exported checkpoints must survive
## --delete, but locally-held ones (lemon-haze-7, ddp-r1) must still be PUSHED.
## An --exclude would do both; rsync's "P" filter only suppresses deletion.
## This bit on 2026-09-10: iiwa14_n4's step-620000 export lived only on the cluster,
## a routine stage_code run deleted it, and the queued benchmark would have hit a
## missing checkpoint. models/ is NOT excluded (see the header), so it needs this.
## What is never staged. rsync reads the WORKING TREE, not git, so .gitignore has NO
## effect here -- anything present locally is pushed unless it is named below. That is
## how `.venv-soromox` reached the cluster and stayed there: the JAX oracle venv was
## committed by accident, then removed from history and gitignored, and none of that
## touched the cluster, because none of it is something rsync reads. It cost 621 MB of
## Lustre and about eleven minutes of every stage until it was named here (2026-09-26).
##
## An --exclude ALSO protects the path from --delete, so adding one stops future pushes
## but does NOT remove a copy already on the cluster. That needs an explicit rm.
EXCLUDES=("${LIK_STAGE_EXCLUDES[@]}")    ## the list lives in verify_staged_tree.sh (shared with the swap)

## STAGE_EXTRA_EXCLUDES adds to the list for one run, space-separated:
##   STAGE_EXTRA_EXCLUDES='models/panda/ scratch.npz' bash cluster/stage_code.sh
## Same warning as above: an exclude added here suppresses the PUSH and the DELETE, so
## it holds whatever is on the cluster frozen rather than removing it.
if [ -n "${STAGE_EXTRA_EXCLUDES:-}" ]; then
    read -r -a _EXTRA_EXCLUDES <<< "$STAGE_EXTRA_EXCLUDES"
    EXCLUDES+=("${_EXTRA_EXCLUDES[@]}")
fi

EXCLUDE_ARGS=()
for _e in "${EXCLUDES[@]}"; do
    [ -n "$_e" ] && EXCLUDE_ARGS+=("--exclude=$_e")
done
echo "excluding: ${EXCLUDES[*]}"

FILTER_ARGS=()
for _p in "${LIK_STAGE_PROTECT[@]}"; do FILTER_ARGS+=("--filter=P $_p"); done

## The MANIFEST of what is shipped: a local rsync dry run with the same excludes; every file must be
## tracked or allowlisted payload, then is sha256'd. Built BEFORE anything is sent.
SDIR="$REPO_ROOT/logs/staged"; NAME="$SC_ROOT-$STAGE_SUBDIR"; NEWDIR="$SDIR/.new"
rm -rf "$NEWDIR"; mkdir -p "$NEWDIR"
lik_build_manifest "$REPO_ROOT" "$NEWDIR/$NAME.sha256" "${EXCLUDE_ARGS[@]}" || exit 7
echo "$COMMIT" > "$NEWDIR/$NAME.commit"
echo "manifest: $(wc -l < "$NEWDIR/$NAME.sha256") files"

sc_run "mkdir -p ~/$SC_ROOT/$STAGE_SUBDIR ~/$SC_ROOT/state ~/$SC_ROOT/results ~/$SC_ROOT/home/.cache"

sc_rsync -az --delete "${FILTER_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
    "$REPO_ROOT/" "$SC_DEST:$SC_ROOT/$STAGE_SUBDIR/" || { echo "ERROR: rsync failed; the cluster tree is a mix: do NOT submit from it" >&2; exit 11; }
## Content pass: rsync's size+mtime quick check skips a same-size same-second change; --checksum (files up
## to the large threshold, no --delete) sends exactly what differs in content.
sc_rsync -az --checksum "--max-size=$LIK_LARGE_BYTES" "${FILTER_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
    "$REPO_ROOT/" "$SC_DEST:$SC_ROOT/$STAGE_SUBDIR/" || { echo "ERROR: the --checksum rsync pass failed; do NOT submit from this tree" >&2; exit 11; }

sc_run "cat > ~/$SC_ROOT/$STAGE_SUBDIR/.staged-manifest.sha256" < "$NEWDIR/$NAME.sha256" || exit 8
## The laptop's reference: per-subdir, and keyed by the manifest's own hash (a repo-next -> repo swap
## then needs no laptop step: the check finds its reference from the live tree's manifest).
_MKEY=$(sha256sum < "$NEWDIR/$NAME.sha256" | cut -d' ' -f1)
mkdir -p "$SDIR/by-manifest"
for _x in sha256 commit large; do cp "$NEWDIR/$NAME.$_x" "$SDIR/by-manifest/$_MKEY.$_x"; mv -f "$NEWDIR/$NAME.$_x" "$SDIR/$NAME.$_x"; done
rmdir "$NEWDIR" 2>/dev/null || true
sc_run "echo $COMMIT > ~/$SC_ROOT/$STAGE_SUBDIR/.staged-commit
echo 'staged commit:' \$(cat ~/$SC_ROOT/$STAGE_SUBDIR/.staged-commit)
du -sh ~/$SC_ROOT/$STAGE_SUBDIR
IIWA=~/$SC_ROOT/repo/models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl
if [ -f \"\$IIWA\" ]; then
    echo \"iiwa checkpoint: \$(du -h \"\$IIWA\" | cut -f1)  sha256 \$(sha256sum \"\$IIWA\" | cut -c1-16)\"
else
    echo 'ERROR: iiwa checkpoint did not arrive -- the iiwa arms cannot run' >&2
fi"

echo
echo "local iiwa checkpoint sha256: $(sha256sum "$REPO_ROOT/models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl" 2>/dev/null | cut -c1-16)"

## Prove it: the cluster's own sha256sum against the laptop's manifest (also what the submit scripts do).
lik_laptop_check || exit $?
