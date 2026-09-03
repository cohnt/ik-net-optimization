#!/bin/bash
# The compaction body, run INSIDE a job -- never on the login node, which is for
# editing, staging and submitting only (compression is explicitly a job's work).
# Submitted by cluster/compact_logs.sh; see that script for the rationale.
#
# CRITICAL: source /etc/profile BEFORE set -u. /etc/profile.d/Z97-byobu.sh reads an
# unset LC_BYOBU, and `set -u` kills the job instantly with a one-line log.
source /etc/profile
set -uo pipefail

cd ~/learned-ik || exit 1
echo "compact_logs job starting on $(hostname) at $(date)"

TOTAL_FILES=0
TOTAL_DIRS=0
FAILED=0
for d in results/*/benchmark/*/; do
    [ -d "$d" ] || continue
    # ls rather than find: one directory, no recursion, and we need the names anyway.
    mapfile -t LOGS < <(cd "$d" && ls -1 *.txt 2>/dev/null)
    [ "${#LOGS[@]}" -eq 0 ] && continue

    A="$d/solver_logs.tar"
    # Fold into an existing archive rather than clobbering it, so a re-run after a
    # partial pass cannot destroy what the first pass already stored.
    [ -f "$A.gz" ] && gunzip -f "$A.gz"
    if [ -f "$A" ]; then
        tar rf "$A" -C "$d" "${LOGS[@]}"
    else
        tar cf "$A" -C "$d" "${LOGS[@]}"
    fi
    RC=$?

    if [ "$RC" -eq 0 ] && [ -f "$A" ] && gzip -f "$A"; then
        # Remove only the names we archived -- never a blanket delete, so a log written
        # after the mapfile above survives to the next run.
        (cd "$d" && rm -f "${LOGS[@]}")
        TOTAL_DIRS=$((TOTAL_DIRS + 1))
        TOTAL_FILES=$((TOTAL_FILES + ${#LOGS[@]}))
    else
        echo "WARNING: archiving failed for $d (rc=$RC) -- loose logs left in place"
        FAILED=$((FAILED + 1))
    fi
done

echo "compacted $TOTAL_FILES loose logs across $TOTAL_DIRS run directories ($FAILED failed)"
echo "remaining loose .txt under results/: $(find results -name '*.txt' | wc -l)"
echo "total files under results/ now:      $(find results -type f | wc -l)"
echo "compact_logs job done at $(date)"
touch ~/learned-ik/compact_logs.DONE
