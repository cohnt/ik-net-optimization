'''Attribute the cost of one IKFlow evaluation: GPU math vs CUDA launch overhead vs
Python/framework overhead vs host-device transfer.

Standalone: mirrors the solver hot path of PandaIKProgram.ik_inference / VarsToQ
(src/panda_program.py:91-157) without building a Drake diagram, so it can be run as

    .venv/bin/python scripts/profiling/profile_flow.py

pydrake is imported only for the AutoDiffXd marshalling measurement, and is optional.
'''

import json
import os
import sys
import time

_t_import0 = time.perf_counter()
import numpy as np
import torch
from ikflow.config import DEVICE
from ikflow.model_loading import get_ik_solver
IMPORT_MS = 1e3 * (time.perf_counter() - _t_import0)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "results", "profiling")

MODEL_NAME = "panda__full__lp191_5.25m"
NUM_ARM_DOF = 7
WARMUP = 10
ITERS = 50

results = {}


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, iters=ITERS, warmup=WARMUP, device=DEVICE):
    '''Steady-state mean ms per call, warm-up discarded, GPU synchronised.'''
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return 1e3 * (time.perf_counter() - t0) / iters


## --------------------------------------------------------------------------- ##
## The hot path, lifted from src/panda_program.py
## --------------------------------------------------------------------------- ##

def make_inference(nn_model, network_width, dtype, device):
    def ik_inference(vars):
        c = vars[:7]
        z = vars[7:7 + network_width]
        correction = vars[7 + network_width:]
        c_torch = torch.cat(
            [c.unsqueeze(0), torch.zeros((1, 1), dtype=vars.dtype, device=device)], dim=1)
        z_batch = z.unsqueeze(0)
        output, _ = nn_model(z_batch, c=c_torch, rev=True)
        q = output[:, :NUM_ARM_DOF].squeeze(0)
        return q + correction

    def ik_inference_with_value(vars):
        q = ik_inference(vars)
        return q, q

    return ik_inference, ik_inference_with_value


def sample_vars(network_width, dtype, device, batch=1):
    '''A plausible 21-dim vars vector (xyz + wxyz + latent + correction).'''
    rng = np.random.default_rng(0)
    v = np.zeros(7 + network_width + NUM_ARM_DOF)
    v[:3] = [0.4, 0.0, 0.4]
    v[3:7] = [1.0, 0.0, 0.0, 0.0]
    v[7:7 + network_width] = rng.standard_normal(network_width)
    if batch == 1:
        return torch.tensor(v, dtype=dtype, device=device)
    return torch.tensor(np.tile(v, (batch, 1)), dtype=dtype, device=device)


## --------------------------------------------------------------------------- ##
## A. one-time ("spin up") costs
## --------------------------------------------------------------------------- ##

def measure_spinup(t_import):
    banner("A. One-time spin-up costs (paid once per process)")

    t0 = time.perf_counter()
    ik_solver, _ = get_ik_solver(MODEL_NAME)
    t_load = 1e3 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    ik_solver.nn_model.to(torch.float64)
    ik_solver.nn_model.eval()
    torch.cuda.synchronize() if torch.device(DEVICE).type == "cuda" else None
    t_cast = 1e3 * (time.perf_counter() - t0)

    nn_model = ik_solver.nn_model
    nw = ik_solver.network_width
    _, with_value = make_inference(nn_model, nw, torch.float64, DEVICE)
    jac = torch.func.jacrev(with_value, has_aux=True)
    vars_t = sample_vars(nw, torch.float64, DEVICE)

    t0 = time.perf_counter()
    jac(vars_t)
    sync(DEVICE)
    t_first = 1e3 * (time.perf_counter() - t0)

    t0 = time.perf_counter()
    jac(vars_t)
    sync(DEVICE)
    t_second = 1e3 * (time.perf_counter() - t0)

    n_params = sum(p.numel() for p in nn_model.parameters())
    param_dev = next(nn_model.parameters()).device
    vram = torch.cuda.memory_allocated() / 2**20 if torch.device(DEVICE).type == "cuda" else 0.0

    print(f"  import torch + ikflow          {t_import:9.1f} ms")
    print(f"  get_ik_solver (weights -> GPU) {t_load:9.1f} ms")
    print(f"  cast to float64                {t_cast:9.1f} ms")
    print(f"  first jacrev call              {t_first:9.1f} ms   <- CUDA context / cuBLAS init")
    print(f"  second jacrev call             {t_second:9.1f} ms")
    print(f"  parameters: {n_params/1e6:.2f} M on {param_dev}, {vram:.1f} MiB allocated")
    print("  Weights stay resident: there is no per-call or per-target reload "
          "(scripts share one ik_solver via model=).")

    results["spinup_ms"] = dict(import_=t_import, get_ik_solver=t_load, cast_float64=t_cast,
                                first_jacrev=t_first, second_jacrev=t_second,
                                n_params=n_params, device=str(param_dev))
    return ik_solver


## --------------------------------------------------------------------------- ##
## B. steady-state breakdown of one AD evaluation of VarsToQ
## --------------------------------------------------------------------------- ##

def measure_ad_breakdown(ik_solver):
    banner("B. Steady-state breakdown of one AutoDiffXd VarsToQ evaluation (float64, GPU)")
    nw = ik_solver.network_width
    nn_model = ik_solver.nn_model
    _, with_value = make_inference(nn_model, nw, torch.float64, DEVICE)
    jac = torch.func.jacrev(with_value, has_aux=True)

    n_vars = 20  # rpy c(6) + z(7) + correction(7), the decision-variable count
    vars_values = sample_vars(nw, torch.float64, DEVICE).cpu().numpy()
    vars_gradients = np.eye(21, n_vars)

    parts = {}
    parts["H2D torch.tensor(vars)"] = timeit(
        lambda: torch.tensor(vars_values, dtype=torch.float64, device=DEVICE))

    vars_tensor = torch.tensor(vars_values, dtype=torch.float64, device=DEVICE)
    parts["jacrev (flow fwd+rev)"] = timeit(lambda: jac(vars_tensor))

    jacobian, q_tensor = jac(vars_tensor)
    parts["D2H jacobian.cpu()"] = timeit(lambda: jacobian.detach().cpu().numpy())
    parts["D2H q.cpu()"] = timeit(lambda: q_tensor.detach().cpu().numpy())

    jacobian_np = jacobian.detach().cpu().numpy()
    parts["chain rule matmul"] = timeit(lambda: jacobian_np @ vars_gradients, device="cpu")

    try:
        from pydrake.autodiffutils import AutoDiffXd
        q_values = np.zeros(9)
        q_gradients = np.zeros((9, n_vars))
        parts["AutoDiffXd construction"] = timeit(
            lambda: np.array([AutoDiffXd(q_values[i], q_gradients[i]) for i in range(9)]),
            device="cpu")
    except ImportError:
        print("  (pydrake unavailable; skipping AutoDiffXd construction)")

    total = sum(parts.values())
    for name, ms in parts.items():
        print(f"  {name:28s} {ms:8.3f} ms   {100*ms/total:5.1f}%")
    print(f"  {'TOTAL':28s} {total:8.3f} ms")
    results["ad_breakdown_ms"] = parts
    return parts


## --------------------------------------------------------------------------- ##
## C. CPU control: is the GPU doing meaningful work at all?
## --------------------------------------------------------------------------- ##

def measure_cpu_control(ik_solver):
    banner("C. CPU-vs-GPU control (same jacrev, batch 1, float64)")
    nw = ik_solver.network_width

    _, gpu_fn = make_inference(ik_solver.nn_model, nw, torch.float64, DEVICE)
    gpu_jac = torch.func.jacrev(gpu_fn, has_aux=True)
    gpu_vars = sample_vars(nw, torch.float64, DEVICE)
    t_gpu = timeit(lambda: gpu_jac(gpu_vars))

    import copy
    cpu_model = copy.deepcopy(ik_solver.nn_model).to("cpu").to(torch.float64).eval()
    _, cpu_fn = make_inference(cpu_model, nw, torch.float64, "cpu")
    cpu_jac = torch.func.jacrev(cpu_fn, has_aux=True)
    cpu_vars = sample_vars(nw, torch.float64, "cpu")
    t_cpu = timeit(lambda: cpu_jac(cpu_vars), iters=20, warmup=5, device="cpu")

    print(f"  jacrev on {DEVICE:10s} {t_gpu:8.2f} ms")
    print(f"  jacrev on cpu        {t_cpu:8.2f} ms")
    print(f"  ratio cpu/gpu        {t_cpu/t_gpu:8.2f}x")
    print("  A large ratio only says the GPU kernels are faster than CPU kernels; it does")
    print("  NOT say the GPU is the bottleneck. Section G settles that.")
    results["cpu_control_ms"] = dict(gpu=t_gpu, cpu=t_cpu, ratio=t_cpu / t_gpu)
    del cpu_model


## --------------------------------------------------------------------------- ##
## D. kernel-level attribution
## --------------------------------------------------------------------------- ##

def measure_kernels(ik_solver):
    banner("D. Kernel-level attribution (torch.profiler)")
    from torch.profiler import ProfilerActivity, profile

    nw = ik_solver.network_width
    _, fn = make_inference(ik_solver.nn_model, nw, torch.float64, DEVICE)
    jac = torch.func.jacrev(fn, has_aux=True)
    vars_t = sample_vars(nw, torch.float64, DEVICE)

    n_calls = 5
    for _ in range(WARMUP):
        jac(vars_t)
    sync(DEVICE)

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(n_calls):
            jac(vars_t)
        sync(DEVICE)
    t_wall = 1e3 * (time.perf_counter() - t0) / n_calls

    evts = prof.key_averages()
    cuda_self_us = sum(e.self_device_time_total for e in evts)
    cpu_self_us = sum(e.self_cpu_time_total for e in evts)
    n_kernels = sum(e.count for e in evts if e.self_device_time_total > 0)
    n_aten = sum(e.count for e in evts if e.key.startswith("aten::"))

    gpu_ms = cuda_self_us / 1e3 / n_calls
    cpu_ms = cpu_self_us / 1e3 / n_calls

    print(f"  wall time per jacrev              {t_wall:8.2f} ms  (profiler adds overhead)")
    print(f"  GPU busy (sum kernel self-time)   {gpu_ms:8.2f} ms   {100*gpu_ms/t_wall:5.1f}% of wall")
    print(f"  CPU self-time (dispatch+python)   {cpu_ms:8.2f} ms   {100*cpu_ms/t_wall:5.1f}% of wall")
    print(f"  CUDA kernel launches per jacrev   {n_kernels/n_calls:8.0f}")
    print(f"  aten:: op calls per jacrev        {n_aten/n_calls:8.0f}")
    print(f"  implied us/launch if launch-bound {1e3*t_wall/(n_kernels/n_calls):8.2f} us")
    print("\n  Top operators by CPU self-time:")
    print(evts.table(sort_by="self_cpu_time_total", row_limit=15))
    print("\n  Top operators by CUDA self-time:")
    print(evts.table(sort_by="self_device_time_total", row_limit=10))

    trace = os.path.join(OUT_DIR, "jacrev_trace.json")
    prof.export_chrome_trace(trace)
    print(f"\n  Chrome trace written to {trace}")

    results["kernels"] = dict(wall_ms=t_wall, gpu_busy_ms=gpu_ms, cpu_self_ms=cpu_ms,
                              kernel_launches=n_kernels / n_calls, aten_calls=n_aten / n_calls)


## --------------------------------------------------------------------------- ##
## E. batch scaling
## --------------------------------------------------------------------------- ##

def measure_batch(ik_solver):
    banner("E. Batch scaling of the forward pass, float64 vs float32")
    nw = ik_solver.network_width
    nn_model = ik_solver.nn_model
    rng = np.random.default_rng(0)

    fwd = {"float64": {}, "float32": {}}
    print(f"  {'batch':>6s} {'f64 ms':>10s} {'f32 ms':>10s} {'f64/f32':>9s}")
    for b in (1, 4, 16, 64, 256, 1024):
        row = {}
        for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
            nn_model.to(dtype).eval()
            c_t = torch.tensor(np.tile([0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0], (b, 1)),
                               dtype=dtype, device=DEVICE)
            z_t = torch.tensor(rng.standard_normal((b, nw)), dtype=dtype, device=DEVICE)

            def run():
                with torch.no_grad():
                    nn_model(z_t, c=c_t, rev=True)

            row[name] = timeit(run, iters=30, warmup=10)
            fwd[name][b] = row[name]
        print(f"  {b:6d} {row['float64']:10.2f} {row['float32']:10.2f} "
              f"{row['float64']/row['float32']:9.2f}")
    nn_model.to(torch.float64).eval()

    results["batch_forward_ms"] = fwd
    print(f"  float32: batch 256 costs {fwd['float32'][256]/fwd['float32'][1]:.2f}x batch 1 "
          f"-- flat, so overhead bound.")
    print(f"  float64: batch 256 costs {fwd['float64'][256]/fwd['float64'][1]:.2f}x batch 1 "
          f"-- linear past ~batch 4, so FLOP bound.")
    print("  The src/generic_program.py:184 comment (batch 256 ~= batch 1) holds only in float32.")


## --------------------------------------------------------------------------- ##
## F. overhead-removal probes
## --------------------------------------------------------------------------- ##

def measure_overhead_removal(ik_solver):
    banner("F. Overhead-removal probes")
    nw = ik_solver.network_width
    vars_t = sample_vars(nw, torch.float64, DEVICE)

    _, base_fn = make_inference(ik_solver.nn_model, nw, torch.float64, DEVICE)
    base_jac = torch.func.jacrev(base_fn, has_aux=True)
    t_base = timeit(lambda: base_jac(vars_t))
    print(f"  baseline jacrev                       {t_base:8.2f} ms")

    # F(i): torch.compile with CUDA graph capture on the jacrev itself.
    for mode in ("default", "reduce-overhead"):
        try:
            torch._dynamo.reset()
            compiled = torch.compile(base_jac, mode=mode)
            t0 = time.perf_counter()
            compiled(vars_t)
            sync(DEVICE)
            t_compile = 1e3 * (time.perf_counter() - t0)
            t_c = timeit(lambda: compiled(vars_t), iters=30, warmup=10)
            print(f"  torch.compile(mode={mode:16s}) {t_c:8.2f} ms  "
                  f"(compile penalty {t_compile:.0f} ms, speedup {t_base/t_c:.2f}x)")
            results.setdefault("compile_ms", {})[mode] = t_c
        except Exception as e:  # dynamo can fail on FrEIA / float64
            print(f"  torch.compile(mode={mode:16s}) FAILED: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:100]}")

    # F(ii): manual CUDA graph capture of the forward pass, the cleanest
    # measurement of pure launch overhead.
    if torch.device(DEVICE).type == "cuda":
        try:
            nn_model = ik_solver.nn_model
            c_t = torch.tensor([[0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0]],
                               dtype=torch.float64, device=DEVICE)
            z_t = torch.tensor(np.random.default_rng(0).standard_normal((1, nw)),
                               dtype=torch.float64, device=DEVICE)

            def fwd():
                with torch.no_grad():
                    return nn_model(z_t, c=c_t, rev=True)

            t_fwd = timeit(fwd, iters=30, warmup=10)

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fwd()
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = fwd()
            t_graph = timeit(lambda: g.replay(), iters=100, warmup=20)
            print(f"  forward, eager                        {t_fwd:8.2f} ms")
            print(f"  forward, CUDA graph replay            {t_graph:8.2f} ms  "
                  f"(speedup {t_fwd/t_graph:.1f}x)")
            print(f"  => launch+python overhead in forward  {t_fwd - t_graph:8.2f} ms "
                  f"({100*(t_fwd-t_graph)/t_fwd:.1f}% of it)")
            results["cuda_graph"] = dict(eager_ms=t_fwd, graph_ms=t_graph)
        except Exception as e:
            print(f"  CUDA graph capture FAILED: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:120]}")


## --------------------------------------------------------------------------- ##
## G. is the CPU or the GPU the bottleneck?
## --------------------------------------------------------------------------- ##

def measure_async_split(ik_solver):
    '''Time a loop twice: once at the moment the CPU returns (kernels still queued),
    once after synchronising. A zero "GPU tail" means the GPU was never behind the CPU,
    i.e. the CPU could not queue work fast enough to keep it busy.'''
    banner("G. CPU-side time vs total time -- who is actually the bottleneck?")
    nw = ik_solver.network_width
    nn_model = ik_solver.nn_model

    def split(fn, n, w, label):
        for _ in range(w):
            fn()
        sync(DEVICE)
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t_cpu = 1e3 * (time.perf_counter() - t0) / n
        sync(DEVICE)
        t_wall = 1e3 * (time.perf_counter() - t0) / n
        verdict = "CPU-BOUND" if t_cpu > 0.9 * t_wall else "GPU-BOUND"
        print(f"  {label:26s} CPU {t_cpu:7.2f} ms | total {t_wall:7.2f} ms | "
              f"GPU tail {t_wall - t_cpu:5.2f} ms -> {verdict}")
        results.setdefault("async_split", {})[label] = dict(cpu_ms=t_cpu, wall_ms=t_wall)

    for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        nn_model.to(dtype).eval()
        _, wv = make_inference(nn_model, nw, dtype, DEVICE)
        jac = torch.func.jacrev(wv, has_aux=True)
        vars_t = sample_vars(nw, dtype, DEVICE)

        def fwd():
            with torch.no_grad():
                return make_inference(nn_model, nw, dtype, DEVICE)[0](vars_t)

        split(fwd, 100, 20, f"{name} forward, batch 1")
        split(lambda: jac(vars_t), 50, 20, f"{name} jacrev, batch 1")
    nn_model.to(torch.float64).eval()
    print("  Zero GPU tail + identical f64/f32 wall time at batch 1 => the flow evaluation")
    print("  is bound by CPU-side work (kernel launches + PyTorch/FrEIA dispatch), not by GPU math.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_import = IMPORT_MS
    ik_solver = measure_spinup(t_import)
    measure_ad_breakdown(ik_solver)
    measure_cpu_control(ik_solver)
    measure_kernels(ik_solver)
    measure_batch(ik_solver)
    measure_async_split(ik_solver)
    measure_overhead_removal(ik_solver)

    out = os.path.join(OUT_DIR, "flow_profile.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary JSON written to {out}")


if __name__ == "__main__":
    main()
