#!/bin/bash
# Laptop-side submission helpers (sourced by build_image.sh, submit_check.sh, submit.sh; needs csail_common.sh).
#
# csail_sbatch <root> <job-name> <output-pattern> "<VAR=value ...>" "<sbatch option words>" <script-in-tree> [extra sbatch words]
#   Prints the job id. Never uses `sbatch --export` (root cancels such jobs: skill trap 10): variables go in front
#   of sbatch. --chdir and absolute --output/--error on /data (jobs cannot see AFS). The job name is unique by
#   construction (callers add a stamp). After ANY error the caller must NOT resubmit: `squeue --me -n <name>`
#   first (an sbatch that errors can still have submitted).
csail_sbatch() {
    local root="$1" name="$2" outpat="$3" envs="$4" opts="$5" script="$6" cmd out rc
    case "$name" in lik_*) ;; *) echo "job names must start with lik_ (got $name)" >&2; return 2 ;; esac
    case "$name$root$envs$opts$script" in *[Cc][Ll][Aa][Uu][Dd][Ee]*) echo "no 'claude' in names/paths" >&2; return 2 ;; esac
    case "$opts $envs" in *--export*) echo "REFUSING: --export is never used (skill trap 10)" >&2; return 2 ;; esac
    csail_refuse_lab_node "$opts" || return 2
    [[ "$envs" =~ ^([A-Za-z_][A-Za-z0-9_]*=[^[:space:]\'\"\;\&\|\$\`\\]*[[:space:]]*)*$ ]] \
        || { echo "REFUSING: an environment value has whitespace or shell characters: $envs" >&2; return 2; }
    cmd="cd $root && $envs sbatch --parsable --job-name=$name --chdir=$root --output=$root/logs/$outpat.out --error=$root/logs/$outpat.err $opts $script"
    ## stdout only is the job id: sbatch prints informational lines ("sbatch: partition ...") on stderr.
    out=$(cs_run "$cmd" 2>"${TMPDIR:-/tmp}/csail_sbatch.err.$$") && rc=0 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "SBATCH ERROR (rc $rc): $out $(cat "${TMPDIR:-/tmp}/csail_sbatch.err.$$")" >&2
        rm -f "${TMPDIR:-/tmp}/csail_sbatch.err.$$"
        echo "DO NOT RESUBMIT. Wait 1-2 minutes, then: squeue --me -n $name   (sbatch can error and still submit)" >&2
        return 1
    fi
    rm -f "${TMPDIR:-/tmp}/csail_sbatch.err.$$"
    printf '%s\n' "$out" | tail -1
}

## GPU-kind -> regex on nvidia-smi's device name, so a job on the wrong hardware fails closed.
csail_gpu_regex() {
    case "$1" in rtx3090) echo "3090" ;; rtx4090) echo "4090" ;; l40s) echo "L40S" ;; a100) echo "A100" ;; h200|csail-h200) echo "H200" ;; *) return 2 ;; esac
}

## The image's sha256 is ANCHORED ON THE LAPTOP (logs/staged/csail-image-<name>.sha256, written by
## `build_image.sh record <name>` from the build job's own record). Jobs refuse an image that hashes otherwise.
csail_image_sha() {
    local f; f="$(csail_repo_root)/logs/staged/csail-image-$1.sha256"
    [ -s "$f" ] || { echo "no laptop record of image '$1' ($f): build it, then 'build_image.sh record $1'" >&2; return 1; }
    cut -d' ' -f1 "$f"
}
