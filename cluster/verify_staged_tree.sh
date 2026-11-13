#!/bin/bash
# Code integrity: the code that runs on SuperCloud is exactly the committed code (cluster/README.md,
# "Code integrity"). stage_code.sh ships only a clean, committed tree and records WHAT it shipped as a
# sha256 manifest (laptop copy under logs/staged/, cluster copy `.staged-manifest.sha256`).
#
#   LAPTOP (bash cluster/verify_staged_tree.sh, or `source` it and call `lik_laptop_check || exit $?`):
#     the laptop's manifest goes to the cluster's own `sha256sum -c -` inside the tree. Exports
#     EXPECT_MANIFEST_SHA for the submitter to pass into the job. Exit 0 ok, 1 STOP (the tree differs),
#     75 CANNOT VERIFY (ssh/infrastructure; not evidence of anything), 76 no laptop reference.
#   JOB (every job script that runs repo code):  source "$REPO/cluster/verify_staged_tree.sh" || exit 1
#                                                lik_job_check "$REPO"
#     refuses unless EXPECT_MANIFEST_SHA is set and is the hash of the tree's manifest, then hashes the
#     whole tree (checkpoints too) against it; exports PYTHONPYCACHEPREFIX (fresh dir, removed at exit).
# Both refuse an unlisted code file (*.py *.sh *.pth) and any symlink. Not hashed, never flagged:
# cpp/build/, __pycache__ contents other than code names, results/, logs/, LLSUB.*/, *.egg-info/, the
# cluster-exported checkpoints. Large files (> LIK_LARGE_BYTES) are only checked for existence from the
# laptop, to keep login-node work light; the job hashes them.

LIK_STOP_MSG="STOP: staged code differs from its recorded commit -- tell Thomas"
LIK_CANNOT_MSG="CANNOT VERIFY (connection/infrastructure failure) -- not evidence of tampering; tell Thomas"
LIK_LARGE_BYTES="${LIK_LARGE_BYTES:-52428800}"

## What stage_code.sh and swap_staged_tree.sh send / protect. An --exclude also protects from --delete;
## a P filter only suppresses the deletion (cluster-made files survive, local ones are still pushed).
LIK_STAGE_EXCLUDES=(
    '.git/' '.git' '.claude/' '.venv/' '.venv-soromox/'
    'results/' 'logs/' 'notebooks/artifacts/'
    '__pycache__/' '*.pyc' '.pytest_cache/'
    'workshop-paper-draft.pdf'
    'cpp/build/'          ## built ON the cluster (build_cpp_job.sh), never pushed
    '*.egg-info/'         ## pip's metadata, made on the cluster
    '/ikflow_solver_log.txt' '/ipopt_output.txt'
)
LIK_STAGE_PROTECT=(
    'models/*/*__step*.pkl' 'models/*/*__step*.arch.json'
    'models/*/*__global_step*.pkl' 'models/*/*__global_step*.arch.json'
    'models/*/*__fk_surrogate.pt' 'models/*/*__fk_surrogate.json' 'models/*/.FK_SURROGATE.DONE'
    '/LLSUB.*/***' '/*.log-[0-9]*' '/slurm-*.out'     ## LLsub/sbatch output lands in the tree root
)

## The check, as script text: `bash -c "$LIK_CHECK_SNIPPET" lik EXPECT_SHA < payload`, run inside the tree.
## payload = `sha256sum` lines, optionally then `%%LARGE%%` and paths checked for existence only.
## Prints `LIK-VERIFY-OK $LIK_NONCE` last on success. Exit 0 ok, 1 mismatch, 70 infrastructure.
read -r -d '' LIK_CHECK_SNIPPET <<'SNIP' || true
expect=$1
export LC_ALL=C
w=$(mktemp -d "${TMPDIR:-/tmp}/lik-verify.XXXXXX") || { echo "mktemp failed"; exit 70; }
trap 'rm -rf "$w"' EXIT
cat > "$w/in" || exit 70
sed '/^%%LARGE%%$/,$d' "$w/in" > "$w/sums"
sed '1,/^%%LARGE%%$/d' "$w/in" > "$w/large"
rc=0; bad() { rc=1; echo "$*"; }
[ -s "$w/sums" ] || bad "empty manifest"
have=$(sha256sum < .staged-manifest.sha256 2>/dev/null | cut -d' ' -f1)
[ "$have" = "$expect" ] || bad ".staged-manifest.sha256 hashes to '${have:-nothing}', expected $expect"
sha256sum --quiet --strict -c - < "$w/sums" > "$w/out" 2>&1 || { rc=1; head -20 "$w/out"; }
while IFS= read -r p; do [ -f "$p" ] && [ ! -L "$p" ] || bad "large file missing or not regular: $p"; done < "$w/large"
PR=( \( -path ./cpp/build -o -path ./results -o -path ./logs -o -path './LLSUB.*' -o -name '*.egg-info' \) -prune -o )
find . "${PR[@]}" -type l -print | sed 's|^\./||' | sort > "$w/links"
[ ! -s "$w/links" ] || { bad "SYMLINK(s) in the tree:"; head -20 "$w/links"; }
{ sed 's/^[0-9a-f]\{64\}  //' "$w/sums"; cat "$w/large"; } | sort -u > "$w/listed"
find . "${PR[@]}" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.pth' \) -print | sed 's|^\./||' | sort > "$w/code"
comm -13 "$w/listed" "$w/code" > "$w/extra"
[ ! -s "$w/extra" ] || { bad "unlisted CODE file(s) in the tree (not in the manifest):"; head -20 "$w/extra"; }
[ "$rc" -eq 0 ] && echo "LIK-VERIFY-OK ${LIK_NONCE:-}"
exit $rc
SNIP

## ------------------------------------------------------------------- stage_code.sh ----

## lik_build_manifest <repo_root> <out.sha256> <rsync --exclude args...>: a local rsync dry run lists what
## the real rsync ships; each file must be git-tracked (incl. submodule) or allowlisted gitignored payload
## (checkpoints, sidecars, FK surrogates), then is sha256'd. Also writes <out minus .sha256>.large.
lik_payload_ok() {
    case "$1" in models/*/*.pkl|models/*/*.arch.json|models/*/*__fk_surrogate.*) return 0 ;; esac
    return 1
}
lik_build_manifest() {
    local root="$1" out="$2" d list tracked nul f refused=0; shift 2
    d=$(mktemp -d "${TMPDIR:-/tmp}/lik-manifest.XXXXXX") || return 1
    mkdir "$d/dest"
    rsync -an --out-format='%n' "$@" "$root/" "$d/dest/" | LC_ALL=C sort > "$d/list" || { rm -rf "$d"; echo "manifest: rsync dry run failed" >&2; return 1; }
    git -C "$root" ls-files --recurse-submodules | LC_ALL=C sort > "$d/tracked" || { rm -rf "$d"; echo "manifest: git ls-files failed" >&2; return 1; }
    : > "${out%.sha256}.large"; : > "$d/nul"
    while IFS= read -r f; do
        case "$f" in */) continue ;; esac
        if [ -L "$root/$f" ]; then echo "REFUSING: a symlink would be shipped: $f" >&2; refused=1; continue; fi
        [ -f "$root/$f" ] || continue
        case "$f" in *$'\n'*|*\\*) echo "REFUSING: unsupported file name: $f" >&2; refused=1; continue ;; esac
        if ! grep -qxF -- "$f" "$d/tracked" && ! lik_payload_ok "$f"; then
            echo "REFUSING: shipped but neither git-tracked nor allowlisted payload: $f" >&2; refused=1; continue
        fi
        [ "$(stat -c %s "$root/$f")" -le "$LIK_LARGE_BYTES" ] || echo "$f" >> "${out%.sha256}.large"
        printf '%s\0' "$f" >> "$d/nul"
    done < "$d/list"
    (cd "$root" && xargs -0 -r sha256sum -- < "$d/nul") > "$out" || refused=1
    rm -rf "$d"
    [ "$refused" -eq 0 ] || { echo "manifest: refused (see above); nothing staged" >&2; return 1; }
    [ -s "$out" ] || { echo "manifest: empty" >&2; return 1; }
}

## ------------------------------------------------------------------- laptop side ----

## lik_verify_remote <reference manifest>: the cluster's own sha256sum against it, inside the tree.
lik_verify_remote() {
    local ref="${1:?}" payload expect b64 nonce out rc
    [ -s "$ref" ] || { echo "no laptop manifest $ref -- run cluster/stage_code.sh first"; return 76; }
    expect=$(sha256sum < "$ref" | cut -d' ' -f1)
    nonce=$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n')
    payload=$(mktemp "${TMPDIR:-/tmp}/lik-payload.XXXXXX") || { echo "$LIK_CANNOT_MSG"; return 75; }
    {
        awk -v lf="${ref%.sha256}.large" 'BEGIN{while((getline l < lf)>0) big[l]=1} {p=$0; sub(/^[0-9a-f]+  /,"",p); if(!(p in big)) print}' "$ref"
        echo '%%LARGE%%'; cat "${ref%.sha256}.large" 2>/dev/null
    } > "$payload"
    b64=$(printf '%s' "$LIK_CHECK_SNIPPET" | base64 -w0)
    ## An empty decode would run `bash -c ""` and exit 0, a vacuous pass: refuse it (and demand the token).
    out=$(sc_run "cd ~/$SC_ROOT/${STAGE_SUBDIR:-repo} || exit 70; S=\$(echo $b64 | base64 -d) && [ -n \"\$S\" ] || exit 70; LIK_NONCE=$nonce nice -n19 bash -c \"\$S\" lik $expect" < "$payload") && rc=0 || rc=$?
    rm -f "$payload"
    printf '%s\n' "$out" | grep -v '^LIK-VERIFY-OK' || true
    if [ "$rc" -eq 0 ]; then
        printf '%s\n' "$out" | grep -qx "LIK-VERIFY-OK $nonce" && { echo "staged tree matches the laptop manifest ($(wc -l < "$ref") files)"; return 0; }
        echo "the remote check printed no success token"; echo "$LIK_CANNOT_MSG"; return 75
    fi
    if [ "$rc" -eq 1 ]; then echo "$LIK_STOP_MSG"; return 1; fi
    echo "remote check exited $rc"; echo "$LIK_CANNOT_MSG"; return 75
}

## The laptop's reference is found by the hash of the tree's own manifest (an untrusted hint that only
## SELECTS one of the laptop's genuine manifests, which the tree must then match file by file), so a
## repo-next -> repo swap needs no laptop step; else the per-subdir copy. Exports EXPECT_MANIFEST_SHA.
lik_laptop_check() {
    local sub="${STAGE_SUBDIR:-repo}" sdir ref key rc
    sdir="$(_sc_repo_root)/logs/staged"; ref="$sdir/$SC_ROOT-$sub.sha256"
    key=$(sc_run "cd ~/$SC_ROOT/$sub && sha256sum < .staged-manifest.sha256" </dev/null) && rc=0 || rc=$?
    [ "$rc" -eq 0 ] || { echo "could not read the tree's manifest (rc $rc)"; echo "$LIK_CANNOT_MSG"; return 75; }
    key=${key%% *}
    [ ! -s "$sdir/by-manifest/$key.sha256" ] || ref="$sdir/by-manifest/$key.sha256"
    lik_verify_remote "$ref" || return $?
    EXPECT_MANIFEST_SHA=$(sha256sum < "$ref" | cut -d' ' -f1); export EXPECT_MANIFEST_SHA
}

## ---------------------------------------------------------------------- job side ----

[ -n "${_lik_pc_dirs+x}" ] || _lik_pc_dirs=()
_lik_trap_set="${_lik_trap_set:-}"
## Removes the per-job pycache dirs. The one EXIT trap calls it; a script that sets its own EXIT trap
## afterwards (run_items.sh) must call lik_cleanup from it.
lik_cleanup() { local d; for d in ${_lik_pc_dirs[@]+"${_lik_pc_dirs[@]}"}; do rm -rf "$d"; done; return 0; }
_lik_exit() { lik_cleanup; [ -z "${_lik_prev_exit:-}" ] || eval "$_lik_prev_exit"; }

## lik_job_check <tree>: exits the job (1) on any problem. Safe under `set -euo pipefail`.
lik_job_check() {
    local tree="${1:?lik_job_check <tree>}" have out rc prefix
    if [ -z "${EXPECT_MANIFEST_SHA:-}" ]; then
        echo "EXPECT_MANIFEST_SHA is not set: submit through the cluster/submit*.sh scripts or cluster/submit_job.sh"
        echo "$LIK_STOP_MSG"; exit 1
    fi
    have=$(sha256sum < "$tree/.staged-manifest.sha256" 2>/dev/null | cut -d' ' -f1) || have=""
    if [ "$have" != "$EXPECT_MANIFEST_SHA" ]; then
        echo "tree manifest hashes to '${have:-nothing}', the laptop anchored $EXPECT_MANIFEST_SHA"
        echo "$LIK_STOP_MSG"; exit 1
    fi
    out=$(cd "$tree" && LIK_NONCE=job bash -c "$LIK_CHECK_SNIPPET" lik "$EXPECT_MANIFEST_SHA" < "$tree/.staged-manifest.sha256") && rc=0 || rc=$?
    printf '%s\n' "$out" | grep -v '^LIK-VERIFY-OK' || true
    if [ "$rc" -ne 0 ] || ! printf '%s\n' "$out" | grep -qx 'LIK-VERIFY-OK job'; then echo "$LIK_STOP_MSG"; exit 1; fi
    prefix=$(mktemp -d "${TMPDIR:-/tmp}/lik-pycache.XXXXXX") || { echo "$LIK_CANNOT_MSG"; exit 1; }
    export PYTHONPYCACHEPREFIX="$prefix"
    _lik_pc_dirs+=("$prefix")
    if [ -z "$_lik_trap_set" ]; then
        eval "set -- $(trap -p EXIT)"; _lik_prev_exit="${3:-}"
        trap '_lik_exit' EXIT; _lik_trap_set=1
    fi
    echo "staged tree verified against the manifest ($(wc -l < "$tree/.staged-manifest.sha256") files)"
}

## Executed, not sourced: the laptop check, printing what a hand-submitted job needs.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -uo pipefail
    source "$(dirname "$0")/ssh_common.sh" || exit 1
    lik_laptop_check || exit $?
    echo "EXPECT_MANIFEST_SHA=$EXPECT_MANIFEST_SHA"
fi
