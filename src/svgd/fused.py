"""The svgd step split at the collision pool: three pure stages, eager / compiled / graphed.

One `al_svgd` step on the fielded robots (`BatchedProgram.has_analytic_row_jacobians`)
is, with the only host round trip -- Drake's exact collision row, in a process pool -- between
the stages:

    stage 1 (GPU)   X -> cfg = q(X), q_plant                     the flow forward, no graph
    host            q_plant -> pool.submit                       Drake starts in the workers
    stage 2 (GPU)   (X, cfg) -> J_q = dq/dX, kinematics + frame Jacobians
    host            pool.collect -> (row, d row / d q_plant)     waits for the workers
    stage 3 (GPU)   everything else: rows, scaling, AL coefficients, the objective and
                    constraint gradients, kernel, the Stein direction, the gradient step
                    and the clamp onto the true bounds (`update_step`).

THE OVERLAP (`svgd_pool_overlap`). Stage 2 is the expensive one (the flow Jacobian is ndof
reverse passes) and does not depend on the collision row, so the pool is dispatched BEFORE
it is launched and collected after: the workers' IPC and Drake compute run while the host
dispatches (eager) or the GPU replays (graphed) stage 2. Strictly across steps nothing can
overlap -- step k+1's configurations are step k's output -- so this is the whole of what
the dependency allows. `_TimedPool` reports both the pool's latency (`span_seconds`) and
the host time actually blocked in it (`seconds`); their difference is the time hidden.

THREE MODES, ONE CODE PATH. The stage functions below are what runs in every mode: called
directly (`eager`), wrapped in `torch.compile` (`compiled`), or the compiled function
captured once into a `torch.cuda.CUDAGraph` and replayed (`graphed`) -- the pattern of
`GraphedFlowCall` on main: static input buffers copied into, one replay, outputs cloned out
of the static output buffers (the next replay overwrites them). The stages are written for
capture: static shapes, no `.item()` / host sync / data-dependent shape, every branch a
`torch.where`, no random numbers (the swarm draws them OUTSIDE, at resampling), no
Python side effect (the evaluation counters are bumped by the driver, never inside).

ONE COMPILE / GRAPH PER STRUCTURE, NOT PER PROGRAM. The benchmark builds a new program per
cell; dynamo guards on the objects a compiled function closes over, and a graph bakes in
the ADDRESSES of every constant it reads (targets, bounds, row scales). So compiled and
graphed stages close over a TEMPLATE `_Target` held in this module's cache, keyed by the
structure signature (`structure_signature`: every tensor's path, shape and dtype and every
Python scalar's path and value in the target, minus a short list of program-specific values
the stages never read), the particle count, the dtype and the step options. A
solve whose target has the template's structure COPIES its constant tensors into the
template's (`copy_constants`) and runs the template's stages; anything that changes a
stage's Python-level behaviour changes the signature and so gets its own entry. After
`FreezeSvgdSteps()` (run_grid calls it once every arm is warmed up) a new entry -- a compile
or a capture inside a timed solve -- RAISES instead of quietly eating the cap.
"""

import time
from dataclasses import dataclass, fields as dataclass_fields

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from src.svgd import al, kernels
from src.svgd.al import ALState

## ------------------------------------------------------------------------------------ ##
##                                   freeze and cache                                    ##
## ------------------------------------------------------------------------------------ ##

_FROZEN = False
_CACHE = {}          # key -> _Entry


def FreezeSvgdSteps(frozen=True):
    """After this, a compile or a CUDA-graph capture of a new step structure raises."""
    global _FROZEN
    _FROZEN = bool(frozen)


def svgd_steps_frozen():
    return _FROZEN


def clear_cache():
    """Drop every template, compiled function and graph (tests only)."""
    _CACHE.clear()


@dataclass(frozen=True)
class StepConfig:
    """The option values the stages read as Python constants (part of the cache key)."""
    kernel: str                 # svgd_kernel: "q" | "none"
    bandwidth_floor: float      # svgd_bandwidth_floor
    inside: bool                # svgd_constraint_inside_kernel
    lr: float                   # svgd_lr (the step is svgd_lr / rho_i)
    T: float                    # svgd_temperature

    @staticmethod
    def from_options(opts):
        return StepConfig(kernel=str(opts.svgd_kernel),
                          bandwidth_floor=float(opts.svgd_bandwidth_floor),
                          inside=bool(opts.svgd_constraint_inside_kernel),
                          lr=float(opts.svgd_lr), T=float(opts.svgd_temperature))


## ------------------------------------------------------------------------------------ ##
##                      structure signature and constant rebinding                      ##
## ------------------------------------------------------------------------------------ ##

## Attributes never walked: the program and its options (read outside the stages), the
## pool (called by the driver), the networks (shared by identity: recorded as an id), and
## values that are program-specific but never read by a stage (the native conditioning
## pose, the decision-variable objects, the calibrated flow-frame transform, RowSpec bounds
## -- the bounds the stages read are the `_eq` / `_lo` / `_hi` TENSORS, which are copied).
_SKIP_ATTRS = {"program", "options", "pool", "model", "shared_model", "_native_c",
               "lumped_vars", "X_ee_flow", "profile"}
_SKIP_CLASS_ATTRS = {("RowSpec", "lb"), ("RowSpec", "ub")}


## The Python FLOATS a stage reads (by attribute name): their values are baked into a
## compiled / captured stage, so they belong to the signature. Every other float in the
## target is construction-time data -- the weld pose of a grasp scene's mug in the
## kinematic tree, the pre-composed frame offsets, the calibrated flow frame -- whose tensor
## form is what the stages read and what `copy_constants` rebinds; recording those values
## made every mug target a new structure. Ints, bools, strings and None (sizes, slots,
## kinds, flags) are always recorded. Adding a float read to a stage means adding its name.
_READ_FLOATS = {"_w_centering", "collision_scale", "c", "correction_bound", "mug_height"}


def _walk(obj, path, tensors, scalars, seen):
    if isinstance(obj, torch.Tensor):
        tensors.append((path, obj))
        return
    if isinstance(obj, (float, np.floating)) and not isinstance(obj, bool):
        name = path.rsplit(".", 1)[-1]
        scalars.append((path, float(obj) if name in _READ_FLOATS else "float"))
        return
    if obj is None or isinstance(obj, (bool, int, str)):
        scalars.append((path, obj))
        return
    if isinstance(obj, (np.integer, np.bool_)):
        scalars.append((path, obj.item()))
        return
    if isinstance(obj, np.ndarray):
        scalars.append((path, ("ndarray", obj.shape, str(obj.dtype))))
        return
    if isinstance(obj, (torch.dtype, torch.device)):
        scalars.append((path, str(obj)))
        return
    if isinstance(obj, torch.nn.Module):
        scalars.append((path, ("module", id(obj))))
        return
    if id(obj) in seen:
        scalars.append((path, ("seen", type(obj).__name__)))
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            _walk(obj[k], f"{path}[{k!r}]", tensors, scalars, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", tensors, scalars, seen)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("src."):
        cls = type(obj).__name__
        for k, v in vars(obj).items():
            if k in _SKIP_ATTRS or (cls, k) in _SKIP_CLASS_ATTRS:
                continue
            _walk(v, f"{path}.{k}", tensors, scalars, seen)
    elif callable(obj):
        scalars.append((path, ("callable", getattr(obj, "__qualname__", type(obj).__name__))))
    else:
        scalars.append((path, ("opaque", type(obj).__name__)))


def structure_signature(tg):
    """`(signature, tensors)` of a `_Target`: a hashable description of everything a stage
    reads at Python level (tensor paths / shapes / dtypes / devices, scalar paths / values,
    the shared network's identity) and the ordered `(path, tensor)` list `copy_constants`
    rebinds."""
    tensors, scalars = [], []
    _walk(tg, "tg", tensors, scalars, set())
    bp = tg.bp
    net = id(bp.flow.shared_model) if getattr(bp, "flow", None) is not None else None
    sig = (type(bp.program).__name__, net,
           tuple((p, tuple(t.shape), str(t.dtype), str(t.device)) for p, t in tensors),
           tuple(scalars))
    return sig, tensors


def copy_constants(dst_tensors, src_tensors):
    """Copy every constant tensor of a solve's target into the template's, in place, so the
    template's compiled / graphed stages compute on this program. An expanded tensor (a
    stride-0 view, which cannot be written) is instead CHECKED equal -- it is a structural
    constant, and a difference would be a signature hole, so it raises."""
    for (pd, d), (ps, s) in zip(dst_tensors, src_tensors):
        if pd != ps:                                            # pragma: no cover
            raise RuntimeError(f"svgd fused step: tensor path mismatch {pd!r} vs {ps!r}")
        if d is s:
            continue
        if d.numel() > 1 and any(st == 0 for st in d.stride()):
            if not torch.equal(d, s):
                raise RuntimeError(f"svgd fused step: expanded constant {pd} differs between "
                                   "programs of one structure")
            continue
        with torch.no_grad():
            d.copy_(s)


## ------------------------------------------------------------------------------------ ##
##                                    the stage maths                                    ##
## ------------------------------------------------------------------------------------ ##

def _bmv(A, v):
    """Batched matrix-vector: `A [N, a, b] @ v [N, b] -> [N, a]`."""
    return (A @ v.unsqueeze(2)).squeeze(2)


def kernel_terms(sc, cfg, finite, X):
    """`(K, R, kernel_on)`: the RBF kernel on the configuration and its repulsion in
    configuration space (`kernels.rbf_terms`, median bandwidth). Non-finite particles are
    isolated (their K row and column are the identity's, their R zero) so a NaN cannot
    spread through the kernel average. `svgd_kernel = none` (or N = 1): `K = I`, `R = 0`."""
    N = X.shape[0]
    dtype, device = X.dtype, X.device
    if sc.kernel == "none" or N < 2:
        return (torch.eye(N, dtype=dtype, device=device),
                torch.zeros_like(cfg), sc.kernel != "none")
    y = cfg.detach()
    fin = finite & torch.isfinite(y).all(dim=1)
    y0 = torch.where(fin.unsqueeze(1), y, torch.zeros_like(y))
    hbw = kernels.median_bandwidth(kernels.pairwise_sqdist(y0), N, floor=sc.bandwidth_floor)
    K, _ = kernels.rbf_terms(y0, hbw)
    mask = fin.unsqueeze(1) & fin.unsqueeze(0)
    K = torch.where(mask, K, torch.eye(N, dtype=dtype, device=device))
    R = (2.0 / hbw) * (K.sum(dim=1, keepdim=True) * y0 - K @ y0)
    R = torch.where(fin.unsqueeze(1), R, torch.zeros_like(R))
    return K, R, True


def update_step(tg, sc, X, gF, gC, K, R_x, finite, rho):
    """The update every mode shares (the split step's stage 3 and the autograd path):

        y = X / s,   phi = kernels.stein_direction(K, s R_x, s gF, s gC, T, inside, kernel),
        y <- y + (svgd_lr / rho_i) phi_i,   then the clamp onto the true bounds B.

    `gF = dF/dX`, `gC = d(L - f)/dX` and the pulled-back repulsion `R_x = J_q^T R` are in the
    decision variables; multiplying by `s = tg.s` (the region half-widths) is the chain rule
    into `y = X / s`. Returns `(X_new, clip [N], n_clip [N])`: the clamp's distance per
    particle in the normalised coordinates and how many coordinates it moved."""
    s = tg.s.unsqueeze(0)
    fin = finite.unsqueeze(1)
    zero = torch.zeros_like(X)
    gF_y = torch.where(fin, s * gF, zero)
    gC_y = torch.where(fin, s * gC, zero)
    R_y = torch.where(fin, s * R_x, zero)
    phi = kernels.stein_direction(K, R_y, gF_y, gC_y, sc.T, inside=sc.inside,
                                  kernel=sc.kernel != "none")
    eps = (sc.lr / rho).unsqueeze(1)
    Xn = X + s * (eps * phi)
    Xp, _ = tg.bp.project(Xn)
    moved = (Xp != Xn) & torch.isfinite(Xn)
    clip = torch.where(moved, (Xp - Xn) / s, zero).norm(dim=1)
    return Xp.detach(), clip, moved.sum(dim=1)


def state_dict(S):
    return {f.name: getattr(S, f.name) for f in dataclass_fields(ALState)}


def stage1(tg, X):
    """`(cfg, q_plant)`: the configuration and the plant vector the pool needs."""
    bp = tg.bp
    with torch.no_grad():
        cfg = bp._config(X)
        return cfg, bp.config_to_plant_q(cfg)


def stage2(tg, X, cfg):
    """`(J_q [N, ndof, n], kin)`: the flow Jacobian (one `jacrev`, `ndof` reverse passes,
    vmapped -- the particles are independent through the flow, so the Jacobian of the
    batch-sum is the per-particle one) and the kinematics with the frame Jacobians at the
    stage-1 configuration. Identity on joint space."""
    bp = tg.bp
    N = X.shape[0]
    if bp.is_learned:
        J = torch.func.jacrev(lambda Xv: bp._config(Xv).sum(dim=0))(X)     # [ndof, N, n]
        J_q = J.permute(1, 0, 2).contiguous()
    else:
        eye = torch.eye(bp.ndof, bp.nvars, dtype=X.dtype, device=X.device)
        J_q = eye.unsqueeze(0).expand(N, -1, -1).contiguous()
    with torch.no_grad():
        kin = bp.kinematics(cfg.detach(), row_jacobians=True)
    return J_q, kin


@torch.compiler.disable
def _scale(tg, out):
    """`tg.scale(out)`, kept OUT of inductor: compiled, these few row reductions fuse into
    one reduction kernel that Triton 3.6 cannot compile ("operand #0 does not dominate this
    use" in `make_ttgir`, on the joint-space grasp program; a fused persistent-reduction
    variant crashed on the learned pose program, 2026-10-09). Running them eager costs a
    handful of launches in `compiled` mode and nothing in `graphed` mode, where the CUDA
    graph captures them like any other kernels."""
    return tg.scale(out)


def stage3(tg, sc, X, cfg, J_q, kin, col_row, col_grad, S):
    """The rest of one step: the rows and costs, their scaling, the AL coefficients, the
    gradients `dF/dX` and `d(L - f)/dX` (closed form downstream of the configuration, the
    flow's `J_q` from stage 2), the kernel, and `update_step`."""
    bp = tg.bp
    S = ALState(**S)
    out = bp.assemble(X, cfg, cfg, kin, col_row, col_grad)
    h, g, finite, infeas = _scale(tg, out)
    ch, cg = al.al_constraint_grad_coefficients(h, g, S)
    dFc, dFx = bp.cost_gradient_parts(X, cfg)
    gF = _bmv(J_q.transpose(1, 2), dFc) + dFx
    gC = tg.constraint_gradient(out, X, J_q, ch, cg)
    K, R, kernel_on = kernel_terms(sc, cfg, finite, X)
    R_x = _bmv(J_q.transpose(1, 2), R) if kernel_on else torch.zeros_like(X)
    Xn, clip, n_clip = update_step(tg, sc, X, gF, gC, K, R_x, finite, S.rho)
    return dict(X=Xn, h=h.detach(), g=g.detach(), F=out.F.detach(), infeas=infeas,
                finite=finite, clip=clip, n_clip=n_clip)


## ------------------------------------------------------------------------------------ ##
##                                    CUDA-graph replay                                  ##
## ------------------------------------------------------------------------------------ ##

class GraphedStage:
    """`fn(*args)` replayed as a CUDA graph; same call, returns the same pytree of FRESH
    tensors (cloned out of the static output buffers).

    Captured on the first call, at that call's shapes (`args` is any pytree of tensors and
    Python constants; the constants must not change between calls -- they were baked in).
    A later call with a different shape or constant raises: one graph per key, by design.
    Capture follows `GraphedFlowCall`: three warm-up calls on a side stream (which also
    trigger the compile), then the capture under the same grad mode."""

    def __init__(self, fn, name, grad_enabled=False):
        self.fn = fn
        self.name = name
        self.grad_enabled = grad_enabled
        self.graph = None
        self.captures = 0
        self.capture_seconds = 0.0

    def _mode(self):
        return torch.enable_grad() if self.grad_enabled else torch.no_grad()

    def _capture(self, flat, spec):
        if _FROZEN:
            raise RuntimeError(f"svgd fused step: a CUDA graph ({self.name}) would be captured "
                               "inside a timed solve; WarmUpSvgdStep must capture every graph "
                               "the run uses (a new N, dtype, step option or program structure?)")
        t0 = time.perf_counter()
        self.spec = spec
        self.static_in = [a.detach().clone() if isinstance(a, torch.Tensor) else a for a in flat]
        self.consts = [None if isinstance(a, torch.Tensor) else a for a in flat]
        args = tree_unflatten(self.static_in, spec)
        device = next(a for a in flat if isinstance(a, torch.Tensor)).device
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        with torch.cuda.stream(side), self._mode():
            for _ in range(3):
                self.fn(*args)
        current.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), self._mode():
            out = self.fn(*args)
        self.out_flat, self.out_spec = tree_flatten(out)
        torch.cuda.synchronize(device)
        self.graph = graph
        self.captures += 1
        self.capture_seconds += time.perf_counter() - t0

    def __call__(self, *args):
        flat, spec = tree_flatten(args)
        if self.graph is None:
            self._capture(flat, spec)
        elif spec != self.spec:
            raise RuntimeError(f"svgd fused step: {self.name} called with a different input "
                               "structure than it was captured with")
        for i, a in enumerate(flat):
            dst = self.static_in[i]
            if isinstance(a, torch.Tensor):
                if a.shape != dst.shape or a.dtype != dst.dtype:
                    raise RuntimeError(f"svgd fused step: {self.name} input {i} is "
                                       f"{tuple(a.shape)} {a.dtype}, captured at "
                                       f"{tuple(dst.shape)} {dst.dtype}")
                if a is not dst:
                    dst.copy_(a)
            elif a != self.consts[i]:
                raise RuntimeError(f"svgd fused step: {self.name} constant input {i} changed")
        self.graph.replay()
        return tree_unflatten([t.clone() if isinstance(t, torch.Tensor) else t
                               for t in self.out_flat], self.out_spec)


## ------------------------------------------------------------------------------------ ##
##                                       the runner                                      ##
## ------------------------------------------------------------------------------------ ##

_INDUCTOR_OPTIONS = {"triton.persistent_reductions": False, "max_fusion_size": 16}


class _Entry:
    def __init__(self, template, tensors, fns):
        self.template = template
        self.tensors = tensors
        self.fns = fns
        self.compile_seconds = 0.0


def _make_fns(tg, sc, mode):
    """The stage callables over `tg` for `mode`: `{"s1", "s2", "s3"}`."""

    def f1(X):
        return stage1(tg, X)

    def f2(X, cfg):
        return stage2(tg, X, cfg)

    def f3(X, cfg, J_q, kin, col_row, col_grad, S):
        return stage3(tg, sc, X, cfg, J_q, kin, col_row, col_grad, S)

    fns = {"s1": f1, "s2": f2, "s3": f3}
    if mode == "eager":
        return fns
    ## `triton.persistent_reductions=False`, scoped to these compiles: with persistent
    ## reductions on, stage 3's largest fused reduction kernel crashes Triton 3.6's
    ## `make_ttgir` pass ("PassManager::run failed") on an sm_86 GPU under torch 2.11
    ## (2026-10-08); `max_fusion_size=16` avoids it too, at a similar compile time.
    ## Every entry's stage closures share ONE code object per stage, and dynamo caches per
    ## code object, so the default `recompile_limit` of 8 is reached after eight (structure,
    ## N, dtype, step options) entries in one process -- after which dynamo SILENTLY runs the stage
    ## eager (and a graph capture of it can fail): the profiling sweep hit it at its ninth
    ## compiled row (2026-10-09). Raised, process-wide, to the accumulated limit.
    import torch._dynamo.config as dynamo_config
    dynamo_config.recompile_limit = max(int(dynamo_config.recompile_limit),
                                        int(dynamo_config.accumulated_recompile_limit))
    compiled = {k: torch.compile(f, dynamic=False, options=_INDUCTOR_OPTIONS)
                for k, f in fns.items()}
    if mode == "compiled":
        return compiled
    return {k: GraphedStage(f, str(k), grad_enabled=(k == "s2")) for k, f in compiled.items()}


class StepRunner:
    """The three stages for one solve, in `mode` (`eager` | `compiled` | `graphed`), over
    the solve's own target (eager) or the cached template it was rebound to."""

    def __init__(self, tg, sc, N, mode):
        self.mode = mode
        self.sc = sc
        self.N = int(N)
        self.reused = False
        if mode == "eager":
            self.tg = tg
            self.fns = _make_fns(tg, sc, "eager")
            self.key = None
            return
        sig, tensors = structure_signature(tg)
        key = (sig, self.N, str(tg.dtype), sc, mode)
        entry = _CACHE.get(key)
        if entry is None:
            if _FROZEN:
                raise RuntimeError(
                    f"svgd fused step: no {mode} step for this structure (N={N}, "
                    f"{tg.dtype}, {sc}) and steps are frozen -- WarmUpSvgdStep must run "
                    f"on a program of every arm and task before the first timed cell")
            entry = _Entry(tg, tensors, _make_fns(tg, sc, mode))
            _CACHE[key] = entry
        else:
            copy_constants(entry.tensors, tensors)
            self.reused = True
        self.key = key
        self.entry = entry
        self.tg = entry.template
        self.fns = entry.fns

    ## Every stage runs under a FIXED grad mode in every mode -- dynamo guards on it, so a
    ## caller's ambient mode must not leak in and force a recompile: stage 2 (the jacrev)
    ## under enable_grad, stages 1 and 3 under no_grad.
    def s1(self, X):
        with torch.no_grad():
            return self.fns["s1"](X)

    def s2(self, X, cfg):
        with torch.enable_grad():
            return self.fns["s2"](X, cfg)

    def s3(self, *args):
        with torch.no_grad():
            return self.fns["s3"](*args)

    def graphs_captured(self):
        if self.mode != "graphed":
            return 0
        return sum(1 for f in self.fns.values() if getattr(f, "graph", None) is not None)
