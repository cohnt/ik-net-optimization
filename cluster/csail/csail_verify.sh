#!/bin/bash
# Laptop-side code-integrity check for a tree staged on CSAIL (README "Code integrity"; the SuperCloud twin is
# lik_verify_remote / lik_laptop_check in cluster/verify_staged_tree.sh, which assume `cd ~/$SC_ROOT/...` and so
# cannot be reused here: ~ on CSAIL is AFS, which jobs cannot see). The CHECK ITSELF is shared: it is the
# verifier's own LIK_CHECK_SNIPPET, sent to the cluster's `sha256sum -c` inside the tree. The job-side check is
# the unchanged lik_job_check.
#
# Source csail_common.sh and ../verify_staged_tree.sh first.
#
#   csail_verify_remote <reference manifest> <remote tree>   0 ok, 1 STOP (tree differs), 75 CANNOT VERIFY, 76 no reference
#   csail_laptop_check <12 hex>                              the same for the staged commit; exports EXPECT_MANIFEST_SHA

csail_verify_remote() {
    local ref="${1:?}" tree="${2:?}" payload expect b64 nonce out rc
    [ -s "$ref" ] || { echo "no laptop manifest $ref -- run cluster/csail/stage.sh first"; return 76; }
    expect=$(sha256sum < "$ref" | cut -d' ' -f1)
    nonce=$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n')
    payload=$(mktemp "${TMPDIR:-/tmp}/lik-payload.XXXXXX") || { echo "$LIK_CANNOT_MSG"; return 75; }
    {
        awk -v lf="${ref%.sha256}.large" 'BEGIN{while((getline l < lf)>0) big[l]=1} {p=$0; sub(/^[0-9a-f]+  /,"",p); if(!(p in big)) print}' "$ref"
        echo '%%LARGE%%'; cat "${ref%.sha256}.large" 2>/dev/null
    } > "$payload"
    b64=$(printf '%s' "$LIK_CHECK_SNIPPET" | base64 -w0)
    ## An empty decode would run `bash -c ""` and exit 0, a vacuous pass: refuse it (and demand the token).
    out=$(cs_run "cd $tree || exit 70; S=\$(echo $b64 | base64 -d) && [ -n \"\$S\" ] || exit 70; LIK_NONCE=$nonce nice -n19 bash -c \"\$S\" lik $expect" < "$payload") && rc=0 || rc=$?
    rm -f "$payload"
    printf '%s\n' "$out" | grep -v '^LIK-VERIFY-OK' || true
    if [ "$rc" -eq 0 ]; then
        printf '%s\n' "$out" | grep -qx "LIK-VERIFY-OK $nonce" && { echo "staged tree matches the laptop manifest ($(wc -l < "$ref") files)"; return 0; }
        echo "the remote check printed no success token"; echo "$LIK_CANNOT_MSG"; return 75
    fi
    if [ "$rc" -eq 1 ]; then echo "$LIK_STOP_MSG"; return 1; fi
    echo "remote check exited $rc"; echo "$LIK_CANNOT_MSG"; return 75
}

## The laptop's reference manifest for commit C12 is the one stage.sh kept (logs/staged/csail-<c12>.sha256); the
## tree's own manifest must hash to it, else STOP (the cluster's copy never selects a reference by itself).
csail_laptop_check() {
    local c12="${1:?}" sdir ref have rc
    sdir="$(csail_repo_root)/logs/staged"; ref="$sdir/csail-$c12.sha256"
    [ -s "$ref" ] || { echo "no laptop manifest $ref -- run cluster/csail/stage.sh first"; return 76; }
    have=$(cs_run "sha256sum < $(csail_tree_root "$c12")/repo/.staged-manifest.sha256" </dev/null) && rc=0 || rc=$?
    [ "$rc" -eq 0 ] || { echo "could not read the tree's manifest (rc $rc)"; echo "$LIK_CANNOT_MSG"; return 75; }
    have=${have%% *}
    [ "$have" = "$(sha256sum < "$ref" | cut -d' ' -f1)" ] \
        || { echo "the cluster tree's manifest hashes to $have, not the laptop's"; echo "$LIK_STOP_MSG"; return 1; }
    csail_verify_remote "$ref" "$(csail_tree_root "$c12")/repo" || return $?
    EXPECT_MANIFEST_SHA=$(sha256sum < "$ref" | cut -d' ' -f1); export EXPECT_MANIFEST_SHA
}
