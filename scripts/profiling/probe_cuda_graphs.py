"""Does CUDA-graph replay remove the flow evaluation's CPU dispatch cost?

At batch size 1 the flow is CPU-bound: most of a `jacrev` is Python and PyTorch describing
~2800 tiny kernels to the GPU, one launch at a time. A CUDA graph records that sequence once
and replays it in a single launch, so it removes exactly the cost the profiler attributed.
This probe measures how much of it comes back, on the code the solver actually runs
(`MakeFlowInference` / `FlowJacobianGen`), for both paths a solve uses:

  jac    `jacrev(has_aux)` -> (dq/dvars, q), the AutoDiffXd path of `VarsToQ`
  value  the forward pass -> q, the float path (`ik_inference`)

and five ways of running each:

  E   eager, today's default
  E_vjp  (jac only) the same Jacobian as a vmapped VJP; what G captures, see make_vjp_jacobian
  C   torch.compile default mode, today's `--compile`
  G   torch.cuda.CUDAGraph capture of E (static input buffer, copy in, replay)
  CG  the same capture of C: keeps inductor's fusion and drops the dispatch
  RO  torch.compile(mode="reduce-overhead"): inductor + cudagraph trees, off the shelf

The value path additionally gets `E_ng` -- eager under `torch.no_grad()` -- because the
solver's float path runs with autograd recording the parameters' graph for nothing.

Compile and capture are OFFLINE compute, as in the benchmark (`WarmUpJacobian` pays them
before the grid). They are reported in their own column and never enter a per-call time;
every variant is warmed up before it is timed.

Timing is per call with the GPU synchronised, which is what the solver sees, two ways:
`call` (device tensor in, device tensors out) and `np` (numpy in, numpy out, the H2D + call
+ D2H that `VarsToQ` actually pays). Correctness against E runs FIRST, over distinct
in-distribution inputs (configurations sampled within joint limits, their FK as the
conditioning pose, latents in the trust-region ball), so a graph returning a stale buffer
fails the run before any number is printed.

    .venv/bin/python scripts/profiling/probe_cuda_graphs.py \
        --chart panda:models/panda/panda__n6__step620000.pkl \
        --chart iiwa14:models/iiwa14/iiwa14__n6__step620000.pkl \
        --chart panda:upstream

Results go to results/profiling/cuda_graphs_<host>.json. Timing is a property of the
machine it ran on and is never compared across machines -- report ratios per machine.
"""

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch

from ikflow.config import DEVICE
from ikflow.model_loading import get_ik_solver

from src.flow_loading import LoadFlowSolver
from src.generic_program import MakeFlowInference

NUM_ARM_DOF = 7
UPSTREAM = {"panda": "panda__full__lp191_5.25m"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chart", action="append", required=True,
                   help="robot:checkpoint.pkl, or robot:upstream for ikflow's own chart")
    p.add_argument("--calls", type=int, default=500, help="timed calls per variant")
    p.add_argument("--warmup", type=int, default=50, help="untimed calls per variant")
    p.add_argument("--check", type=int, default=300, help="distinct inputs for the correctness check")
    p.add_argument("--profile-calls", type=int, default=20)
    p.add_argument("--variants", default="E,C,G,CG,RO")
    p.add_argument("--out", default=None)
    return p.parse_args()


def sync():
    torch.cuda.synchronize()


## ------------------------------- the callables --------------------------------- ##

class Graphed:
    """Capture `fn(static_in)` once; each call copies into the static input and replays.

    Returns the static outputs themselves -- the next call overwrites them, so a caller
    must copy before calling again (`VarsToQ` does: it converts to numpy immediately).
    """

    def __init__(self, fn, example, no_grad=False):
        self.static_in = example.clone()
        ctx = torch.no_grad if no_grad else torch.enable_grad
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), ctx():
            for _ in range(3):
                fn(self.static_in)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), ctx():
            self.static_out = fn(self.static_in)
        sync()

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out


def no_grad_fn(fn):
    def wrapped(x):
        with torch.no_grad():
            return fn(x)
    return wrapped


def make_vjp_jacobian(infer, device):
    """`jacrev(infer, has_aux=True)` written as a vmapped VJP against a fixed basis.

    Same ops, and CLAUDE.md measured it bit-identical to jacrev. It exists because eager
    jacrev cannot be captured: it builds its basis offsets with a bare `torch.tensor(list)`
    every call, which ikflow's global default-device override turns into an unpinned
    host-to-device copy (illegal under capture, and a host sync in ordinary eager use).
    Scoping the default device to CPU does not help, since FrEIA's GraphINN does
    `torch.zeros(n).to(x)` and would then copy the other way.
    """
    basis = torch.eye(NUM_ARM_DOF, dtype=torch.float64, device=device)
    value = lambda x: infer(x)[0]

    def jac(x):
        q, vjp_fn = torch.func.vjp(value, x)
        (J,) = torch.func.vmap(vjp_fn)(basis)
        return J, q
    return jac


def build_variants(nn_model, width, names, example):
    """{(path, tag): (callable, offline_seconds)}. Every callable takes a device tensor."""
    infer = MakeFlowInference(nn_model, width, NUM_ARM_DOF, DEVICE)
    jac = torch.func.jacrev(infer, has_aux=True)
    jac_vjp = make_vjp_jacobian(infer, DEVICE)
    value = lambda x: infer(x)[0]
    value_ng = no_grad_fn(value)

    out = {}

    def add(key, make):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn = make()
        fn(example)                       # first call pays compile / capture
        sync()
        out[key] = (fn, time.perf_counter() - t0)

    # `graphable` is what G captures: for the Jacobian, the VJP form (see above).
    for path, base, base_ng, graphable in (("jac", jac, jac, jac_vjp),
                                           ("value", value, value_ng, value)):
        if "E" in names:
            add((path, "E"), lambda: base)
            if path == "value":
                add((path, "E_ng"), lambda: value_ng)
            else:
                add((path, "E_vjp"), lambda: jac_vjp)
        if "C" in names:
            compiled = torch.compile(base_ng)
            add((path, "C"), lambda: compiled)
        if "G" in names:
            add((path, "G"), lambda: Graphed(graphable, example, no_grad=(path == "value")))
        if "CG" in names:
            compiled_cg = torch.compile(base_ng)
            add((path, "CG"), lambda: Graphed(compiled_cg, example, no_grad=(path == "value")))
        if "RO" in names:
            ro = torch.compile(base_ng, mode="reduce-overhead")
            add((path, "RO"), lambda: ro)
    return out


## ------------------------------- inputs ---------------------------------------- ##

def sample_vars(solver, n, rng, radius):
    """In-distribution lumped vars: FK of a configuration within limits as the conditioning
    pose (the frame the flow was trained on), a latent in the trust-region ball."""
    width = solver.network_width
    robot = solver.robot
    q = robot.sample_joint_angles(n)
    # jrl's own dtype: the poses only need to be in distribution, not float64-exact.
    pose = robot.forward_kinematics(torch.tensor(q, dtype=torch.float32, device=DEVICE),
                                    out_device=DEVICE)
    pose = pose.detach().cpu().numpy().astype(np.float64)
    z = rng.standard_normal((n, width))
    scale = radius * rng.uniform(size=(n, 1)) ** (1.0 / width) / np.linalg.norm(z, axis=1, keepdims=True)
    v = np.zeros((n, 7 + width + NUM_ARM_DOF))
    v[:, :7] = pose[:, :7]
    v[:, 7:7 + width] = z * scale
    v[:, 7 + width:] = rng.uniform(-0.1, 0.1, size=(n, NUM_ARM_DOF))
    return v


def as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


## ------------------------------- measurements ---------------------------------- ##

def check(variants, inputs):
    """Max abs / relative difference against E on distinct consecutive inputs, and how many
    calls were bit-identical. Consecutive inputs differ, so a stale buffer shows here."""
    report = {}
    for path in ("jac", "value"):
        ref_fn = variants[(path, "E")][0]
        refs = []
        for v in inputs:
            x = torch.tensor(v, dtype=torch.float64, device=DEVICE)
            refs.append([t.detach().cpu().numpy().copy() for t in as_tuple(ref_fn(x))])
        for (p, tag), (fn, _) in variants.items():
            if p != path or tag == "E":
                continue
            max_abs = max_rel = 0.0
            identical = 0
            for v, ref in zip(inputs, refs):
                x = torch.tensor(v, dtype=torch.float64, device=DEVICE)
                got = [t.detach().cpu().numpy().copy() for t in as_tuple(fn(x))]
                same = True
                for a, b in zip(got, ref):
                    d = np.abs(a - b)
                    finite = np.isfinite(b)
                    if not np.array_equal(np.isfinite(a), finite):
                        max_abs = max_rel = float("inf")
                    if finite.any():
                        max_abs = max(max_abs, float(d[finite].max()))
                        max_rel = max(max_rel, float((d[finite] / np.maximum(1.0, np.abs(b[finite]))).max()))
                    same &= np.array_equal(a, b, equal_nan=True)
                identical += same
            report[f"{path}/{tag}"] = dict(max_abs=max_abs, max_rel=max_rel,
                                           bit_identical=identical, n=len(inputs))
    return report


def time_variant(fn, inputs, calls, warmup, numpy_io):
    def one(v):
        if numpy_io:
            x = torch.tensor(v, dtype=torch.float64, device=DEVICE)
            return [t.detach().cpu().numpy() for t in as_tuple(fn(x))]
        x = v
        fn(x)
        sync()

    if not numpy_io:
        inputs = [torch.tensor(v, dtype=torch.float64, device=DEVICE) for v in inputs]
    for i in range(warmup):
        one(inputs[i % len(inputs)])
    sync()
    times = np.empty(calls)
    for i in range(calls):
        v = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        one(v)
        times[i] = time.perf_counter() - t0
    ms = 1e3 * times
    return dict(mean=float(ms.mean()), median=float(np.median(ms)),
                p95=float(np.percentile(ms, 95)), min=float(ms.min()))


def profile_variant(fn, inputs, n):
    """Kernel launches and GPU kernel time per call, from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    xs = [torch.tensor(v, dtype=torch.float64, device=DEVICE) for v in inputs[:n]]
    for x in xs[:3]:
        fn(x)
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for x in xs:
            fn(x)
        sync()
    launches = graph_launches = 0
    device_us = 0.0
    for evt in prof.events():
        name = evt.name
        if name in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx"):
            launches += 1
        elif name in ("cudaGraphLaunch", "cuGraphLaunch"):
            graph_launches += 1
        dtype = getattr(evt, "device_type", None)
        if dtype is not None and "CUDA" in str(dtype):
            device_us += getattr(evt, "device_time", None) or getattr(evt, "cuda_time", 0.0) or 0.0
    return dict(kernel_launches_per_call=launches / n, graph_launches_per_call=graph_launches / n,
                gpu_kernel_ms_per_call=device_us / n / 1e3)


## ------------------------------- driver ---------------------------------------- ##

def load_chart(spec):
    robot, path = spec.split(":", 1)
    if path == "upstream":
        solver, _ = get_ik_solver(UPSTREAM[robot])
    else:
        solver = LoadFlowSolver(robot, path)
    solver.nn_model.to(torch.float64)
    solver.nn_model.eval()
    return solver


def main():
    args = parse_args()
    assert torch.device(DEVICE).type == "cuda", "CUDA graphs need a CUDA device"
    names = set(args.variants.split(","))
    names.add("E")
    rng = np.random.default_rng(0)
    host = socket.gethostname()
    results = dict(host=host, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                   cuda=torch.version.cuda, calls=args.calls, warmup=args.warmup, charts={})

    for spec in args.chart:
        torch._dynamo.reset()
        solver = load_chart(spec)
        width = solver.network_width
        nb_nodes = getattr(solver, "arch", {}).get("nb_nodes", "upstream")
        radius = np.sqrt(width) + 1.5
        print(f"\n=== {spec}  (width {width}, nb_nodes {nb_nodes}) on {results['device']}", flush=True)

        inputs = sample_vars(solver, max(args.check, 64), rng, radius)
        example = torch.tensor(inputs[0], dtype=torch.float64, device=DEVICE)
        variants = build_variants(solver.nn_model, width, names, example)

        correctness = check(variants, inputs[:args.check])
        for key, r in correctness.items():
            print(f"  check {key:9s} max_abs {r['max_abs']:.2e}  max_rel {r['max_rel']:.2e}  "
                  f"bit-identical {r['bit_identical']}/{r['n']}", flush=True)
            if not r["max_rel"] < 1e-8:
                raise SystemExit(f"{key} disagrees with eager (max_rel {r['max_rel']}); no timing reported")

        rows = {}
        for (path, tag), (fn, offline) in variants.items():
            row = dict(offline_s=offline)
            row["call"] = time_variant(fn, inputs, args.calls, args.warmup, numpy_io=False)
            row["np"] = time_variant(fn, inputs, args.calls, args.warmup, numpy_io=True)
            row.update(profile_variant(fn, inputs, args.profile_calls))
            rows[f"{path}/{tag}"] = row

        print(f"  {'variant':10s} {'call med':>9s} {'np med':>9s} {'np mean':>9s} {'np p95':>9s} "
              f"{'x vs C':>7s} {'x vs E':>7s} {'launch':>7s} {'graph':>6s} {'gpu ms':>7s} {'offline s':>9s}")
        for path in ("jac", "value"):
            base_c = rows.get(f"{path}/C", {}).get("np", {}).get("median")
            base_e = rows[f"{path}/E"]["np"]["median"]
            for key, row in rows.items():
                if not key.startswith(path + "/"):
                    continue
                med = row["np"]["median"]
                print(f"  {key:10s} {row['call']['median']:9.3f} {med:9.3f} {row['np']['mean']:9.3f} "
                      f"{row['np']['p95']:9.3f} {(base_c / med) if base_c else float('nan'):7.2f} "
                      f"{base_e / med:7.2f} {row['kernel_launches_per_call']:7.0f} "
                      f"{row['graph_launches_per_call']:6.0f} {row['gpu_kernel_ms_per_call']:7.3f} "
                      f"{row['offline_s']:9.2f}", flush=True)

        results["charts"][spec] = dict(width=width, nb_nodes=nb_nodes, correctness=correctness,
                                       rows=rows)
        del variants
        torch.cuda.empty_cache()

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__)))), "results", "profiling", f"cuda_graphs_{host}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
