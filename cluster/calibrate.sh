#!/bin/bash
# Placement / contention / cap calibration -- a REAL job on a REAL node.
#
# ============================ STANDING REMINDER ============================
# If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
# ===========================================================================
#
# Submit four of these from ~/learned-ik/repo, one per node (they queue behind
# the account's 4-node GrpTRES cap if anything else is running):
#     CALIB_ARM=gpu-procs LLsub ./cluster/calibrate.sh -g volta:2 -s 40 -q xeon-g6-volta -T 3:00:00 -J lik_cal_gpu
#     CALIB_ARM=cpu-procs LLsub ./cluster/calibrate.sh -g volta:2 -s 40 -q xeon-g6-volta -T 3:00:00 -J lik_cal_cpu
#     CALIB_ARM=caps      LLsub ./cluster/calibrate.sh -g volta:2 -s 40 -q xeon-g6-volta -T 3:00:00 -J lik_cal_cap
#     CALIB_ARM=parity    LLsub ./cluster/calibrate.sh -g volta:2 -s 40 -q xeon-g6-volta -T 3:00:00 -J lik_cal_par
#
# This is a real experiment with real runtime, so per standing rule 6 it does NOT
# go on debug-gpu. Only cluster/smoke.sh goes there.
#
# WHAT IT DECIDES, and why it has to be measured rather than assumed. The
# benchmark is WALL-CLOCK capped, so a cell's result depends on how much work the
# process got done inside its cap. Two processes contending for a core therefore
# change what is measured, not merely how long it takes. Cell-level sharding
# across many workers per node is legitimate only up to the largest worker count
# at which per-cell ITERATION counts are unchanged -- and that number is a
# property of this hardware, this scene and this network, not something to guess.
#
#   gpu-procs  workers/node = 1,2,4,8,20,40 on the V100s: the largest count whose
#              median iterations still match the single-worker run.
#   cpu-procs  the same with CUDA_VISIBLE_DEVICES="" (jrl then reports "cpu" with
#              no code change). CLAUDE.md's profiling says the flow is ENTIRELY
#              CPU-bound at batch 1 -- the GPU is never behind, and float64 and
#              float32 cost the same wall time despite 3.4x different kernel
#              time -- so CPU-only may cost little, and if so it removes GPU
#              time-slicing as a contention source altogether.
#   caps       iterations achieved at 10/20/45/90/180 s with ONE worker, so the
#              campaign's cap is chosen from data on this machine. The laptop's
#              20/45 s carry no meaning here and are not to be inherited: timing
#              is never compared across machines.
#   parity     a small grid at the archived seed (does the cluster reproduce the
#              laptop's qualitative findings?), plus how long it takes to stage
#              $ROOT to node-local $TMPDIR and to import at 40-way concurrency.
source /etc/profile
set -uo pipefail

## --- GPU selection --------------------------------------------------------
## Slurm exports CUDA_VISIBLE_DEVICES as a comma-separated list, and on this cluster
## the entries are GPU **UUIDs**, not indices -- so exporting "0" or "1" is not a
## valid selector for the allocated cards. Capture the list once, before any worker
## narrows it, and hand each worker one entry from it.
SLURM_GPUS="${CUDA_VISIBLE_DEVICES:-}"
PinGpu() {   ## $1 = worker index; empty $SLURM_GPUS means "no GPU was allocated"
    if [ -z "$SLURM_GPUS" ]; then export CUDA_VISIBLE_DEVICES=""; return; fi
    local n; n=$(echo "$SLURM_GPUS" | tr ',' '\n' | grep -c .)
    export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_GPUS" | cut -d, -f$(( $1 % n + 1 )))"
}

## --- waiting on workers ---------------------------------------------------
## `wait` with NO arguments does not do what it looks like it does here. This
## script sends its own stdout through a process substitution
## (`exec > >(tee "$LOG")`), and bash counts that `tee` as one of its background
## children -- but `tee` cannot exit until the script does, because it holds the
## script's stdout open. So a bare `wait` blocks forever, having already reaped
## every worker it was actually meant to wait for. Three of the four arms of the
## first calibration attempt hung on exactly this, holding a GPU node each and
## doing nothing; `caps` survived only because it never forks. Always wait on the
## PIDs we collected ourselves.
WaitPids() { local p rc=0; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }

ROOT="${LEARNED_IK_ROOT:-$HOME/learned-ik}"
REPO="$ROOT/repo"
ARM="${CALIB_ARM:-gpu-procs}"
OUT="$ROOT/calib/$ARM"
LOG="$ROOT/calibrate.$ARM.log"
DONE="$ROOT/calibrate.$ARM.DONE"
mkdir -p "$OUT"
rm -f "$DONE"
exec > >(tee "$LOG") 2>&1
echo "===== calibrate [$ARM] on $(hostname) at $(date -Is) ====="
nvidia-smi -L || true
cd "$REPO" || { echo "no repo at $REPO"; echo FAIL > "$DONE"; exit 1; }

PY="$ROOT/venv/bin/python"
BASE_PYTHONPATH="$ROOT/drake/lib/python3.12/site-packages"
BASE_LDPATH="$ROOT/sysdeps/usr/lib/x86_64-linux-gnu"
REAL_HOME="$HOME"

# The fixed workload every arm measures. Learned-arm-only (the only arm whose
# cost depends on the machine), and -- critically -- the GRASP task, because the
# measurement only works on a workload that BINDS against the wall-clock cap.
#
# The first attempt used --task pose and measured nothing: a pose cell converges
# in ~74 iterations and ~6 s, so its iteration count is identical at 10, 20, 45,
# 90 and 180 s. A converged solve takes the same iterations however contended the
# machine is -- only its wall time moves -- so a contention probe run on it
# reports "no contention" no matter how bad the contention is. On the grasp task
# roughly 40% of cells exit at the cap, so iterations achieved inside a fixed cap
# is a direct throughput measurement and contention is visible in it.
WORKLOAD=(--task mug --targets 4 --guesses 2 --arms learned --config latent --compile --seed 0)

RunOne() {   # RunOne <local-index> <tag> <device> <wall-time>
    local i=$1 tag=$2 device=$3 wall=$4
    export HOME="${TMPDIR:-/tmp}/home.$i"
    mkdir -p "$HOME/.cache"
    ln -sfn "$ROOT/home/.cache/ikflow" "$HOME/.cache/ikflow" 2>/dev/null || true
    ln -sfn "$ROOT/home/.cache/drake" "$HOME/.cache/drake" 2>/dev/null || true
    export PYTHONPATH="$BASE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$BASE_LDPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl.$i"
    export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/inductor.$i"
    export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton.$i"
    mkdir -p "$MPLCONFIGDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
    if [ "$device" = "cpu" ]; then export CUDA_VISIBLE_DEVICES=""
    else PinGpu "$i"; fi
    "$PY" -u scripts/panda/panda_benchmark.py "${WORKLOAD[@]}" \
        --wall-time "$wall" --tag "$tag" > "$OUT/$tag.log" 2>&1
    HOME="$REAL_HOME"
}

Sweep() {    # Sweep <device>
    local device=$1 P
    for P in 1 2 4 8 20 40; do
        echo "--- $device, $P worker(s), $(date -Is)"
        local T0=$SECONDS
        local pids=()
        for ((i = 0; i < P; i++)); do
            RunOne "$i" "calib_${device}_p${P}_w${i}" "$device" 20 & pids+=($!)
        done
        WaitPids "${pids[@]}"
        echo "    $P worker(s) took $((SECONDS - T0)) s wall"
    done
}

case "$ARM" in
  gpu-procs) Sweep gpu ;;
  cpu-procs) Sweep cpu ;;
  caps)
    for W in 10 20 45 90 180; do
        echo "--- cap ${W}s, single worker, $(date -Is)"
        RunOne 0 "calib_cap_${W}" gpu "$W"
    done ;;
  parity)
    echo "--- node-local staging cost (informs whether items should run off Lustre)"
    T0=$SECONDS; cp -r "$ROOT/venv" "${TMPDIR:-/tmp}/venv" >/dev/null 2>&1
    echo "    venv -> TMPDIR: $((SECONDS - T0)) s"
    T0=$SECONDS; cp -r "$ROOT/drake" "${TMPDIR:-/tmp}/drake" >/dev/null 2>&1
    echo "    drake -> TMPDIR: $((SECONDS - T0)) s"
    T0=$SECONDS; cp -r "$REPO" "${TMPDIR:-/tmp}/repo" >/dev/null 2>&1
    echo "    repo -> TMPDIR: $((SECONDS - T0)) s"
    echo "--- import cost at 40-way concurrency (Lustre read amplification)"
    T0=$SECONDS; imp_pids=()
    for ((i = 0; i < 40; i++)); do
        ( export HOME="${TMPDIR:-/tmp}/home.imp.$i"; mkdir -p "$HOME/.cache"
          ln -sfn "$ROOT/home/.cache/ikflow" "$HOME/.cache/ikflow" 2>/dev/null
          ln -sfn "$ROOT/home/.cache/drake" "$HOME/.cache/drake" 2>/dev/null
          PYTHONPATH="$BASE_PYTHONPATH" LD_LIBRARY_PATH="$BASE_LDPATH" \
          CUDA_VISIBLE_DEVICES=$((i % 2)) "$PY" -c \
            "import torch, pydrake.all, ikflow, jrl" ) & imp_pids+=($!)
    done
    WaitPids "${imp_pids[@]}"
    echo "    40 concurrent imports: $((SECONDS - T0)) s"
    echo "--- a small grid at the archived seed, all arms, for qualitative parity"
    RunOne 0 "calib_parity" gpu 20
    export HOME="${TMPDIR:-/tmp}/home.0"
    PYTHONPATH="$BASE_PYTHONPATH" LD_LIBRARY_PATH="$BASE_LDPATH" \
    CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/panda/panda_benchmark.py \
        --task mug --targets 4 --guesses 2 --wall-time 20 --config latent \
        --arms learned,numerical,analytic,analytic8 --compile --seed 0 \
        --tag calib_parity_mug > "$OUT/calib_parity_mug.log" 2>&1
    HOME="$REAL_HOME" ;;
  *) echo "unknown CALIB_ARM=$ARM"; echo FAIL > "$DONE"; exit 2 ;;
esac

## --- the verdict ----------------------------------------------------------
echo; echo "===== summary [$ARM] ====="
HOME="$REAL_HOME" PYTHONPATH="$BASE_PYTHONPATH" LD_LIBRARY_PATH="$BASE_LDPATH" \
CUDA_VISIBLE_DEVICES="" "$PY" - "$ARM" <<'PYEOF'
import glob, json, os, re, statistics, sys
arm = sys.argv[1]
rows = {}
for path in sorted(glob.glob("results/panda/benchmark/calib_*/summary.json")):
    tag = os.path.basename(os.path.dirname(path))
    payload = json.load(open(path))
    recs = payload["records"].get("learned", [])
    iters = [r["iterations"] for r in recs if r.get("iterations") is not None]
    walls = [r["wall_time"] for r in recs if r.get("wall_time") is not None]
    if not iters:
        continue
    rows.setdefault(tag, {})
    rows[tag] = dict(n=len(iters), med_iters=statistics.median(iters),
                     med_wall=statistics.median(walls),
                     ok=sum(1 for r in recs if r.get("feasible")),
                     device=payload["metadata"].get("device"),
                     cap=payload["metadata"].get("wall_time"))

if arm in ("gpu-procs", "cpu-procs"):
    dev = "gpu" if arm == "gpu-procs" else "cpu"
    # Pool the workers that ran together: the question is what a cell measures
    # when P of them share a node, not what any single worker did.
    by_p = {}
    for tag, r in rows.items():
        m = re.match(rf"calib_{dev}_p(\d+)_w\d+$", tag)
        if m:
            by_p.setdefault(int(m.group(1)), []).append(r)
    if by_p:
        base = statistics.median([r["med_iters"] for r in by_p[min(by_p)]])
        print(f"{'workers':>8} {'runs':>5} {'median iters':>13} {'vs P=1':>8} {'median wall':>12}")
        for p in sorted(by_p):
            med = statistics.median([r["med_iters"] for r in by_p[p]])
            wall = statistics.median([r["med_wall"] for r in by_p[p]])
            print(f"{p:>8} {len(by_p[p]):>5} {med:>13.0f} {med / base:>7.2f}x {wall:>12.2f}")
        print()
        print("The usable worker count is the largest P whose median iterations are")
        print("within ~2% of P=1. Below that, concurrent workers are changing what the")
        print("wall-clock-capped benchmark measures, not just how long it takes.")
elif arm == "caps":
    print(f"{'cap (s)':>8} {'median iters':>13} {'median wall':>12} {'feasible':>9}")
    for tag in sorted(rows, key=lambda t: rows[t]["cap"] or 0):
        if not tag.startswith("calib_cap_"): continue
        r = rows[tag]
        print(f"{r['cap']:>8.0f} {r['med_iters']:>13.0f} {r['med_wall']:>12.2f} {r['ok']:>9}")
    print()
    print("Pick the campaign's cap from this curve on THIS hardware. Do not inherit")
    print("the laptop's 20/45 s: cross-machine timing is never compared.")
else:
    for tag, r in sorted(rows.items()):
        print(f"{tag:<28} n={r['n']:<3} median iters {r['med_iters']:>6.0f}  "
              f"feasible {r['ok']}  device {r['device']}")
PYEOF

echo OK > "$DONE"
echo "===== calibrate [$ARM] complete at $(date -Is) ====="
