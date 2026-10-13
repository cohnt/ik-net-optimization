"""Time the exact collision row: the autodiff loop against the batched clearances call.

    PYTHONPATH=/home/tommy/opt/rlg/drake-build-crc/lib/python3.12/site-packages:$PYTHONPATH \\
        .venv/bin/python scripts/svgd/profile_clearances.py [--Ns 64,256,1024] [--repeats 5]

Needs a pydrake carrying `CollisionChecker.CalcRobotClearances` (branch
`calc-robot-clearances`; the default environment's lacks it and this script then says so and
exits). On both Panda scenes -- hardened pose, and grasp with the target mug welded at a
sampled collision-free configuration as `tests/test_collision_backend.py` places it (its
helpers build the scene) -- and for each N, the row and gradient over N configurations drawn
uniformly in the plant's limits, timed as the median of `--repeats` calls after one warm-up:

  autodiff          `CollisionEvaluator` (the program's constraint, one AutoDiffXd Eval each)
  clearances        `ClearanceEvaluator`, `Parallelism.Max()`
  clearances_1t     `ClearanceEvaluator`, one thread

plus `drake_ms`, the share of the clearances row spent inside the `CalcRobotClearances` call
itself (the rest is the pybind result conversion and the numpy aggregation).

Output: `results/profiling/svgd_clearances_<host>_<UTC time>.json`, host-tagged because timing
is never compared across machines.
"""

import argparse
import datetime
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.svgd.collision_backend import clearances_unavailable_reason       # noqa: E402
from src.utils import RepoDir                                               # noqa: E402


def _median_seconds(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--Ns", default="64,256,1024")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    reason = clearances_unavailable_reason()
    if reason is not None:
        sys.exit("profile_clearances: " + reason)
    import pydrake
    from pydrake.common import Parallelism
    from src.svgd.collision_backend import ClearanceEvaluator, CollisionEvaluator
    from tests.test_collision_backend import _program, _spec

    Ns = [int(n) for n in args.Ns.split(",")]
    host = socket.gethostname()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or os.path.join(RepoDir(), "results", "profiling",
                                   f"svgd_clearances_{host}_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"{'scene':5s} {'N':>5s} {'backend':14s} {'threads':>7s} {'ms/batch':>9s} "
          f"{'us/config':>9s} {'drake_ms':>9s} {'speedup':>8s}")
    for task in ("pose", "mug"):
        program = _program("panda", task)
        spec = _spec(program)
        evs = {"autodiff": CollisionEvaluator.from_program(program),
               "clearances": ClearanceEvaluator.from_program(program, spec=spec),
               "clearances_1t": ClearanceEvaluator.from_program(program, spec=spec,
                                                                parallelism=Parallelism(1))}
        lo, hi = program.plant.GetPositionLowerLimits(), program.plant.GetPositionUpperLimits()
        for N in Ns:
            Q = rng.uniform(lo, hi, size=(N, lo.size))
            base = None
            for name, ev in evs.items():
                s = _median_seconds(lambda: ev.eval(Q, need_grad=True), args.repeats)
                drake_ms = None
                if name != "autodiff":
                    configs = [np.ascontiguousarray(q) for q in Q]
                    drake_ms = 1e3 * _median_seconds(
                        lambda: ev.checker.CalcRobotClearances(configs, ev.influence,
                                                               ev.parallelism), args.repeats)
                base = s if base is None else base
                r = dict(task=task, N=N, backend=name, threads=int(ev.num_threads),
                         ms_per_batch=1e3 * s, us_per_config=1e6 * s / N,
                         drake_ms=drake_ms, speedup_vs_autodiff=base / s)
                rows.append(r)
                print(f"{task:5s} {N:5d} {name:14s} {r['threads']:7d} {r['ms_per_batch']:9.2f} "
                      f"{r['us_per_config']:9.1f} "
                      f"{'' if drake_ms is None else format(drake_ms, '9.2f'):>9s} "
                      f"{r['speedup_vs_autodiff']:8.1f}", flush=True)
    meta = dict(host=host, cpu_count=os.cpu_count(), utc=stamp, argv=sys.argv[1:],
                pydrake=os.path.dirname(pydrake.__file__), repeats=args.repeats, seed=args.seed,
                configs="uniform in the plant's position limits")
    with open(out, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
