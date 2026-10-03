#!/bin/bash
# Create an ISOLATED cluster tree beside the default one, in the layout every isolated tree
# uses. Run from the laptop; it acts over SSH.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Usage:
#   bash cluster/make_tree.sh learned-ik-screw
#   SC_ROOT=learned-ik-screw bash cluster/stage_code.sh      # then stage code into it
#
# The layout, as cluster/README.md and each campaign's runbook describe it:
#   OWN:      repo/ (filled by stage_code.sh), home/ (the tree's own ikflow dataset cache),
#             state/, results/
#   SHARED, READ-ONLY, by symlink into the default tree ~/learned-ik:
#             venv/, drake/, sysdeps/, home/.cache/drake
# Read-only means no `pip install` through these links: the default tree's campaign runs out
# of that venv, and a package added here would land inside its run invisibly.
#
# The links are RELATIVE, so the tree survives being moved beside ~/learned-ik. It REFUSES if
# the tree already exists -- it never repairs or overwrites one -- and if the default tree is
# missing any of the four things it links to, since a dangling link fails only at the first
# job that reaches through it.
set -euo pipefail
source "$(dirname "$0")/ssh_common.sh"

NAME="${1:?usage: make_tree.sh <tree-name, e.g. learned-ik-screw>}"
case "$NAME" in
    learned-ik)    echo "REFUSING: that is the default tree" >&2; exit 2 ;;
    learned-ik-*)  ;;
    *)             echo "REFUSING: an isolated tree is named learned-ik-<campaign>" >&2; exit 2 ;;
esac

sc_run "set -e
cd ~
if [ -e '$NAME' ]; then echo 'REFUSING: ~/$NAME already exists' >&2; exit 3; fi
for p in learned-ik/venv learned-ik/drake learned-ik/sysdeps learned-ik/home/.cache/drake; do
    [ -e \"\$p\" ] || { echo \"REFUSING: ~/\$p is missing; the link would dangle\" >&2; exit 4; }
done
mkdir -p '$NAME'/home/.cache '$NAME'/state '$NAME'/results
ln -s ../learned-ik/venv    '$NAME'/venv
ln -s ../learned-ik/drake   '$NAME'/drake
ln -s ../learned-ik/sysdeps '$NAME'/sysdeps
ln -s ../../../learned-ik/home/.cache/drake '$NAME'/home/.cache/drake
for p in venv drake sysdeps home/.cache/drake; do
    [ -e '$NAME'/\$p ] || { echo \"ERROR: ~/$NAME/\$p does not resolve\" >&2; exit 5; }
done
echo 'created ~/$NAME:'
find '$NAME' -maxdepth 3 \\( -type l -printf '  %p -> %l\n' \\) -o \\( -type d -printf '  %p/\n' \\)"
