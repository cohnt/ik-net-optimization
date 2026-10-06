#!/bin/bash
# Add the GVS push-rod arm's JAX stack to an EXISTING learned-ik venv, changing nothing else.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit from ~/learned-ik/repo on the login node (download is the only partition with internet):
#     LLsub ./cluster/add_jax_stack_job.sh -s 8 -q download
#
# Written for the 2026-10-06 merge of ~/learned-ik-gvs into ~/learned-ik. The main tree's venv was
# built before setup_supercloud.sh learned the JAX step, and re-running that whole script would
# re-resolve its UNPINNED python deps and move patch versions under the campaign of record. So
# this installs exactly the eleven distributions the GVS tree's venv carried -- the versions stage
# GVS was measured with -- with --no-deps, and then proves that no pre-existing distribution
# changed version (`pip freeze` before and after, minus the eleven).
source /etc/profile
set -uo pipefail

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
PY="$ROOT/venv/bin/python"
LOG="$ROOT/add_jax_stack.log"
DONE="$ROOT/add_jax_stack.DONE"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
echo "===== add JAX stack to $ROOT/venv on $(hostname) at $(date -Is) ====="
Fail() { echo "FAILED: $*"; echo "FAIL $*" > "$DONE"; exit 1; }
[ -x "$PY" ] || Fail "no venv at $ROOT/venv"

PKGS=(jax==0.11.2 jaxlib==0.11.2 soromox==0.5.0 optimistix==0.1.0 equinox==0.13.8
      diffrax==0.7.2 lineax==0.1.1 jaxtyping==0.3.11 wadler_lindig==0.1.7
      ml_dtypes==0.6.0 opt_einsum==3.4.0)
NEW_RE='^(jax|jaxlib|soromox|optimistix|equinox|diffrax|lineax|jaxtyping|wadler[-_]lindig|ml[-_]dtypes|opt[-_]einsum)=='

"$PY" -m pip freeze | grep -viE "$NEW_RE" | sort > "$ROOT/.freeze_before"
"$PY" -m pip install --quiet --no-deps "${PKGS[@]}" || Fail "pip install"
"$PY" -m pip freeze | grep -viE "$NEW_RE" | sort > "$ROOT/.freeze_after"
if ! diff "$ROOT/.freeze_before" "$ROOT/.freeze_after"; then
    Fail "a pre-existing distribution changed version (diff above)"
fi
echo "pre-existing distributions unchanged ($(wc -l < "$ROOT/.freeze_after") of them)"
rm -f "$ROOT/.freeze_before" "$ROOT/.freeze_after"

## CPU jaxlib only: the flow owns the GPU (see setup_supercloud.sh step 5).
"$PY" - <<'PYCHECK' || Fail "import check"
import importlib.metadata as md
import jax, soromox, optimistix, torch
assert jax.default_backend() == "cpu", jax.default_backend()
print("jax", jax.__version__, jax.default_backend(), "| soromox", md.version("soromox"),
      "| torch", torch.__version__)
PYCHECK
"$PY" -m pip check || echo "NOTE: pip check reports the above (compare with the GVS venv's)"
echo "OK" > "$DONE"
echo "===== done $(date -Is) ====="
