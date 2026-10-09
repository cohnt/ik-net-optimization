"""Time one svgd step: method x N x dtype x step mode, with the collision pool's share.

    .venv/bin/python scripts/svgd/profile_step.py                       # the full table
    .venv/bin/python scripts/svgd/profile_step.py --Ns 64 --modes eager,graphed --overlap both

For every (arm, method, N, dtype, mode) the solver is built on a Panda program of the task
asked for (hardened scene, the benchmark's option shape; the pose target at the scene frame of
a collision-free configuration, the grasp target a mug welded at one), WARMED UP outside the
timing (`SvgdSolver.warm_up`: the pool's worker scenes, and under `compiled` / `graphed` the
compile and the capture, whose seconds are recorded as `warmup_seconds`), and then `--steps`
steps of the method are timed between two CUDA synchronisations, on a swarm drawn by the
solver's own paired init around a collision-free start. The step is the method's real step:
the split step of `src/svgd/fused.py` (`_step_split`) of `al_svgd`, the only method.

Columns: `ms_per_step`; `pool_ms_per_step`, the host time BLOCKED in the pool per step;
`pool_span_ms_per_step`, the pool's latency (submit to collect; equal to the blocked time
without the overlap); `pool_us_per_config` (blocked time over configurations sent) and
`pool_share` (blocked / total). `--overlap both` runs every row with the pool dispatched
before stage 2 and collected after (`svgd_pool_overlap=True`) and with it collected at once,
which is the overlap's measurement. `graphed` needs CUDA. Every pool has `--workers` processes
(default 4: each is a whole Drake scene in memory).

Output: `results/profiling/svgd_step_<host>_<UTC time>.json`, host-tagged because timing is
never compared across machines. Spawns pool workers, so it needs the `__main__` guard.
"""

import argparse
import datetime
import json
import os
import socket
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.flow_loading import LoadFlowSolver                                   # noqa: E402
from src.generic_program import ProgramOptions                                # noqa: E402
from src.panda_program import (PandaIKProgram, PandaIKProgramNumerical,       # noqa: E402
                               PandaMugProgram, PandaMugProgramNumerical)
from src.svgd.solver import SvgdSolver, close_shared_pools                    # noqa: E402
from src.target_screening import SceneFile                                    # noqa: E402
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints, RepoDir  # noqa: E402

CHECKPOINT = "models/panda/panda__n6__step620000.pkl"
CLASSES = {("pose", "learned"): PandaIKProgram, ("pose", "numerical"): PandaIKProgramNumerical,
           ("mug", "learned"): PandaMugProgram, ("mug", "numerical"): PandaMugProgramNumerical}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("pose", "mug"), default="pose")
    p.add_argument("--arms", default="learned,numerical")
    p.add_argument("--methods", default="al_svgd")
    p.add_argument("--Ns", default="1,16,64,256,1024")
    p.add_argument("--dtypes", default="float32,float64")
    p.add_argument("--modes", default="eager,compiled,graphed")
    p.add_argument("--overlap", choices=("on", "off", "both"), default="on")
    p.add_argument("--steps", type=int, default=20, help="timed steps (ADMM: rounds) per row")
    p.add_argument("--workers", type=int, default=4, help="svgd_collision_workers (default 4)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default=None)
    return p.parse_args()


def options(arm, **kw):
    """The benchmark's shape (`--config latent`): centering 1e-4 learned / 1.0 joint space,
    the trust region, the approved correction penalty."""
    return ProgramOptions(visualize=False, joint_centering_cost=1e-4 if arm == "learned" else 1.0,
                          latent_trust_region=4.0, correction_cost_weight=10.0, mug_height=0.04,
                          which_solver="svgd", acceptable_constr_viol_tol=1e-4, max_wall_time=1e6,
                          file_print_name="", **kw)


def collision_free_q(p, rng):
    lower, upper = p.ConfigLimits()
    while True:
        q = rng.uniform(np.asarray(lower)[:p.num_arm_dof], np.asarray(upper)[:p.num_arm_dof])
        if np.ravel(p.collision_free_constraint_eval.Eval(p.ConfigToPlantQ(q)))[0] < 1.0:
            return q


def build_programs(task, arms, seed):
    """One program per arm on one scene and target."""
    rng = np.random.default_rng(seed)
    with HiddenPrints():
        solver = LoadFlowSolver("panda", os.path.join(RepoDir(), CHECKPOINT))
        yaml = SceneFile("panda", task, "hardened")
        diagram = BuildEnv(meshcat=None, directives_file=yaml)
        helper = CLASSES[(task, "numerical")](diagram, options=options("numerical"), model=solver)
        if task == "pose":
            helper.create_prog(np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]))   # for its collision row
            q = collision_free_q(helper, rng)
            translation, wxyz = helper.fk(helper.ConfigToPlantQ(q))
            target = np.concatenate([translation, wxyz])
        else:
            helper.create_prog()
            q = collision_free_q(helper, rng)
            diagram, target = GenerateDiagramWithMug(helper.ConfigToPlantQ(q), helper, yaml, None)
        out = {}
        for arm in arms:
            p = CLASSES[(task, arm)](diagram, options=options(arm), model=solver)
            if task == "pose":
                p.create_prog(target)
            else:
                p.create_prog(target_mug=target)
            p.SetStartFromQ(collision_free_q(p, rng))
            out[arm] = p
    return out


def time_row(p, method, N, dtype, mode, overlap, steps, workers):
    p.options = replace(p.options, svgd_method=method, svgd_n=N, svgd_dtype=dtype, svgd_kernel="q",
                        svgd_compile=(mode != "eager"), svgd_cuda_graph=(mode == "graphed"),
                        svgd_pool_overlap=overlap, svgd_collision_workers=workers,
                        svgd_stop_patience=10 ** 6)
    s = SvgdSolver(p)
    warm = s.warm_up()
    x0 = np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float)
    X, _, _ = s._init_particles(x0)
    S = s._init_state(X)
    s._pool.drain()
    s._make_runner()
    sync = (lambda: torch.cuda.synchronize(s.device)) if s.device.type == "cuda" else (lambda: None)
    inner = 1
    for _ in range(2):                                                   # untimed
        X, _, _ = s._step_split(X, S)
    s._pool.reset()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        X, _, _ = s._step_split(X, S)
    sync()
    dt = time.perf_counter() - t0
    pool = s._pool
    return dict(
        method=method, N=N, dtype=dtype, mode=mode,
        overlap=bool(overlap), steps=steps, inner_per_step=inner, workers=s.workers,
        ms_per_step=1e3 * dt / steps, ms_per_inner=1e3 * dt / (steps * inner),
        pool_ms_per_step=1e3 * pool.seconds / steps, pool_span_ms_per_step=1e3 * pool.span_seconds / steps,
        pool_calls=int(pool.calls), pool_configs=int(pool.configs),
        pool_us_per_config=1e6 * pool.seconds / max(1, pool.configs),
        pool_share=pool.seconds / dt if dt > 0 else float("nan"),
        warmup_seconds=warm, step_reused_template=bool(s.warmup_info.get("reused_template")),
        graphs_captured=int(s.warmup_info.get("graphs_captured", 0)))


def main():
    args = parse_args()
    arms = args.arms.split(",")
    methods = args.methods.split(",")
    Ns = [int(n) for n in args.Ns.split(",")]
    dtypes = args.dtypes.split(",")
    modes = args.modes.split(",")
    if not torch.cuda.is_available():
        modes = [m for m in modes if m != "graphed"]
    overlaps = {"on": (True,), "off": (False,), "both": (True, False)}[args.overlap]
    host = socket.gethostname()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or os.path.join(RepoDir(), "results", "profiling", f"svgd_step_{host}_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = dict(host=host, device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                torch=torch.__version__, cpu_count=os.cpu_count(), task=args.task, steps=args.steps,
                checkpoint=CHECKPOINT, utc=stamp, argv=sys.argv[1:])
    programs = build_programs(args.task, arms, args.seed)
    rows = []
    try:
        for arm in arms:
            for method in methods:
                for dtype in dtypes:
                    for N in Ns:
                        for mode in modes:
                            for ov in overlaps:
                                try:
                                    r = time_row(programs[arm], method, N, dtype, mode, ov,
                                                 args.steps, args.workers)
                                except Exception as e:                     # recorded, not fatal
                                    r = dict(method=method, N=N, dtype=dtype, mode=mode, overlap=ov,
                                             error=f"{type(e).__name__}: {e}"[:400])
                                r["arm"] = arm
                                rows.append(r)
                                if "error" in r:
                                    print(f"{arm:9s} {method:9s} N={N:5d} {dtype:7s} {mode:8s} "
                                          f"overlap={ov!s:5s}  ERROR {r['error'][:120]}", flush=True)
                                else:
                                    print(f"{arm:9s} {method:9s} N={N:5d} {dtype:7s} {r['mode']:8s} "
                                          f"overlap={ov!s:5s} {r['ms_per_step']:8.2f} ms/step "
                                          f"(pool blocked {r['pool_ms_per_step']:6.2f}, span "
                                          f"{r['pool_span_ms_per_step']:6.2f} ms; "
                                          f"{r['pool_us_per_config']:6.1f} us/config; share "
                                          f"{r['pool_share']:.2f}; warm-up {r['warmup_seconds']:.1f} s)",
                                          flush=True)
                                with open(out, "w") as f:
                                    json.dump(dict(metadata=meta, rows=rows), f, indent=1)
    finally:
        close_shared_pools()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
