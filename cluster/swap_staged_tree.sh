#!/bin/bash
# Swap a side-staged tree (~/learned-ik/repo-next, written by `STAGE_SUBDIR=repo-next
# bash cluster/stage_code.sh`) into the live tree (~/learned-ik/repo) BETWEEN two campaigns,
# as a cluster job chained after the running one:
#
#   sc_run "cd ~/learned-ik && sbatch --dependency=afterany:<last job of the running round> \
#           repo-next/cluster/swap_staged_tree.sh"
#   DEPENDENCY=afterok:<this job> ... bash cluster/submit_bench.sh <next manifest> 4
#
# Submit it with `STAGE_SUBDIR=repo-next SUBMIT_CMD="sbatch --dependency=afterany:<job>" bash cluster/submit_job.sh
# cluster/swap_staged_tree.sh` (laptop check of repo-next, EXPECT_MANIFEST_SHA into the job).
#
# Why a job and not the laptop: a round queued behind another starts the instant the first
# drains, and `stage_code.sh` (an rsync from the laptop) cannot run under a live campaign --
# every item re-reads the tree's .py files when it starts. A chained job runs exactly in the
# gap, with no laptop awake, and `afterok` on it means the next round never starts on a tree
# that was not swapped. It refuses if anything is still running out of the live tree, and it
# mirrors with stage_code.sh's excludes and protects: the cluster-only files under the live tree (the
# cpp build, exported checkpoints, LLSUB output) survive, a source file the new commit removed does not. Submitted from
# ~/learned-ik (NOT from repo/), so its own submission directory never trips the guard.
#SBATCH --job-name=lik_swap_tree
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/home/gridsan/tcohn/learned-ik/results/swap_tree.log-%j
set -euo pipefail
ROOT="$HOME/learned-ik"
echo "swap_staged_tree: $(date -Is) on $(hostname)"
test -f "$ROOT/repo-next/.staged-commit" || { echo "REFUSING: no $ROOT/repo-next/.staged-commit"; exit 2; }
N=$(squeue -u "$USER" -h -o '%Z %j %T' | awk -v d="$ROOT/repo" '$1==d && $3=="RUNNING" && $2 !~ /_cal_/ && $2 != "smoke.sh"' | wc -l)
[ "$N" -eq 0 ] || { echo "REFUSING: $N job(s) still running out of $ROOT/repo"; squeue -u "$USER" -o '%i %Z %j %T'; exit 3; }
source "$ROOT/repo-next/cluster/verify_staged_tree.sh" || exit 1
lik_job_check "$ROOT/repo-next"   # repo-next is exactly what the laptop recorded, else STOP
echo "live tree was: $(cat "$ROOT/repo/.staged-commit" 2>/dev/null || echo unknown)"
## --delete (with stage_code.sh's excludes and protects), so a source file the new commit removed does not
## linger as an unlisted code file; cluster-made files (cpp/build, exported checkpoints, LLSUB output) stay.
SWAP=(--exclude='.staged-commit' --exclude='.staged-manifest.sha256')
for e in "${LIK_STAGE_EXCLUDES[@]}"; do SWAP+=("--exclude=$e"); done
for e in "${LIK_STAGE_PROTECT[@]}"; do SWAP+=("--filter=P $e"); done
rsync -a --delete "${SWAP[@]}" "$ROOT/repo-next/" "$ROOT/repo/"
cp "$ROOT/repo-next/.staged-manifest.sha256" "$ROOT/repo/.staged-manifest.sha256"
cp "$ROOT/repo-next/.staged-commit" "$ROOT/repo/.staged-commit"
lik_job_check "$ROOT/repo"   # the live tree now carries the same manifest and matches it
echo "live tree is now: $(cat "$ROOT/repo/.staged-commit")"
for f in src/generic_program.py src/svgd/solver.py cluster/gen_manifest.py; do
    cmp -s "$ROOT/repo-next/$f" "$ROOT/repo/$f" || { echo "MISMATCH after swap: $f"; exit 4; }
done
echo "swap_staged_tree: OK $(date -Is)"
