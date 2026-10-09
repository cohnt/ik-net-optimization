"""The svgd step split at the collision pool: three pure stages, eager / compiled / graphed.

One `al_svgd` or `tsvgd` step on the fielded robots (`BatchedProgram.has_analytic_row_jacobians`)
is, with the only host round trip -- Drake's exact collision row, in a process pool -- between
the stages:

    stage 1 (GPU)   X -> cfg = q(X), q_plant                     the flow forward, no graph
    host            q_plant -> pool.submit                       Drake starts in the workers
    stage 2 (GPU)   (X, cfg) -> J_q = dq/dX, kinematics + frame Jacobians
    host            pool.collect -> (row, d row / d q_plant)     waits for the workers
    stage 3 (GPU)   everything else: rows, scaling, AL coefficients, kernel, Adam, the
                    Gauss-Newton correction (Levenberg-Marquardt damped, q-step clamped),
                    bound projection, the LM gain ratio, best tracking, the lr schedule.

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
`torch.where`, linear solves through `solve_ex(check_errors=False)` with `info` folded in
(`al._solve_gram`), no random numbers (the swarm draws them OUTSIDE, at resampling), no
Python side effect (the evaluation counters are bumped by the driver, never inside).

ONE COMPILE / GRAPH PER STRUCTURE, NOT PER PROGRAM. The benchmark builds a new program per
cell; dynamo guards on the objects a compiled function closes over, and a graph bakes in
the ADDRESSES of every constant it reads (targets, bounds, row scales). So compiled and
graphed stages close over a TEMPLATE `_Target` held in this module's cache, keyed by the
structure signature (`structure_signature`: every tensor's path, shape and dtype and every
Python scalar's path and value in the target, minus a short list of program-specific values
the stages never read), the particle count, the dtype, the method and the step options. A
solve whose target has the template's structure COPIES its constant tensors into the
template's (`copy_constants`) and runs the template's stages; anything that changes a
stage's Python-level behaviour changes the signature and so gets its own entry. After
`FreezeSvgdSteps()` (run_grid calls it once every arm is warmed up) a new entry -- a compile
or a capture inside a timed solve -- RAISES instead of quietly eating the cap.
"""

import math
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
    method: str
    kernel: str
    bandwidth: str
    bandwidth_floor: float
    q_step_max: float
    gn_delta: float
    lm: bool
    gn_lm_min: float
    gn_lm_max: float
    gn_lm_growth: float
    tangent_delta: float
    switch_infeas: float
    lr0: float
    lr_min: float
    lr_decay_t: float

    @staticmethod
    def from_options(opts):
        return StepConfig(
            method=str(opts.svgd_method), kernel=str(opts.svgd_kernel),
            bandwidth=str(opts.svgd_bandwidth), bandwidth_floor=float(opts.svgd_bandwidth_floor),
            q_step_max=float(opts.svgd_q_step_max), gn_delta=float(opts.svgd_gn_delta),
            lm=float(opts.svgd_gn_lm) > 0.0, gn_lm_min=float(opts.svgd_gn_lm_min),
            gn_lm_max=float(opts.svgd_gn_lm_max), gn_lm_growth=float(opts.svgd_gn_lm_growth),
            tangent_delta=float(opts.svgd_tangent_delta),
            switch_infeas=float(opts.svgd_tsvgd_switch_infeas),
            lr0=float(opts.svgd_lr), lr_min=float(opts.svgd_lr_min),
            lr_decay_t=float(opts.svgd_lr_decay_t))


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
    """`(K, R, needs_pullback)`: the kernel and the repulsion in the kernel's space (q for
    `svgd_kernel = "q"`, x otherwise). Non-finite particles are isolated (their K row and
    column are the identity's) so a NaN cannot spread through the kernel average."""
    N = X.shape[0]
    dtype, device = X.dtype, X.device
    if sc.kernel == "none" or N < 2:
        K, R = kernels.identity_terms(N, X.shape[1], dtype, device)
        return K, R, False
    y = cfg.detach() if sc.kernel == "q" else X
    fin = finite & torch.isfinite(y).all(dim=1)
    y0 = torch.where(fin.unsqueeze(1), y, torch.zeros_like(y))
    D = kernels.pairwise_sqdist(y0)
    if sc.bandwidth == "median":
        hbw = kernels.median_bandwidth(D, N, floor=sc.bandwidth_floor)
    else:
        hbw = torch.full((), float(sc.bandwidth) ** 2, dtype=dtype, device=device)
    K, _ = kernels.rbf_terms(y0, hbw)
    mask = fin.unsqueeze(1) & fin.unsqueeze(0)
    K = torch.where(mask, K, torch.eye(N, dtype=dtype, device=device))
    R = (2.0 / hbw) * (K.sum(dim=1, keepdim=True) * y0 - K @ y0)
    R = torch.where(fin.unsqueeze(1), R, torch.zeros_like(R))
    return K, R, sc.kernel == "q"


def gn_rows(h, g, J_h, J_g, mask):
    """`(J, r)` of the Gauss-Newton row set `mask [N, m_e + m_i]` (0/1): rows outside the
    set ZEROED in both (static shape), non-finite entries zeroed."""
    J = torch.cat([J_h, J_g], dim=1) * mask.unsqueeze(2)
    r = torch.cat([h, g], dim=1).detach() * mask
    J = torch.nan_to_num(J.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return J, r


def clamp_step(sc, learned, dx, J_q):
    """`clamp_q_step` plus the per-particle flag that the bound was hit."""
    Jq = J_q if learned else None
    dq = dx if Jq is None else _bmv(Jq, dx)
    clamped = dq.abs().amax(dim=1) > sc.q_step_max
    return al.clamp_q_step(dx, Jq, sc.q_step_max), clamped


def q_motion(learned, J_q, d):
    """`|J_q d|_inf` (learned) or `|d|_inf` (joint space): a step's configuration motion."""
    return (_bmv(J_q, d) if learned else d).abs().amax(dim=1)


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
def _scale_and_merit(tg, out):
    """`tg.scale(out)` and the merit, kept OUT of inductor: compiled, these few row
    reductions fuse into one reduction kernel that Triton 3.6 cannot compile ("operand #0
    does not dominate this use" in `make_ttgir`, on the joint-space grasp program; a fused
    persistent-reduction variant crashed on the learned pose program, 2026-10-09). Running
    them eager costs a handful of launches in `compiled` mode and nothing in `graphed`
    mode, where the CUDA graph captures them like any other kernels."""
    h, g, finite, infeas = tg.scale(out)
    return h, g, finite, infeas, tg.merit_of(out.F, infeas)


def _common(tg, sc, X, cfg, J_q, kin, col_row, col_grad, S, gamma, T):
    """What both methods' stage 3 start with: rows, scaling, merit, the LM gain update,
    AL coefficients, kernel, row Jacobians and the three gradients."""
    bp = tg.bp
    out = bp.assemble(X, cfg, cfg, kin, col_row, col_grad)
    h, g, finite, infeas, merit = _scale_and_merit(tg, out)
    ratio = torch.full_like(infeas, float("nan"))
    if sc.lm:
        S, ratio = al.lm_gain_update(S, h, g, sc.gn_lm_growth, sc.gn_lm_min, sc.gn_lm_max)
    ch, cg = al.al_constraint_grad_coefficients(h, g, S)
    K, R, pull = kernel_terms(sc, cfg, finite, X)
    J_h, J_g = tg.row_jacobians_out(out, X, J_q)
    dFc, dFx = bp.cost_gradient_parts(X, cfg)
    gF = _bmv(J_q.transpose(1, 2), dFc) + dFx
    R_pulled = _bmv(J_q.transpose(1, 2), R) if pull else R
    fin = finite.unsqueeze(1)
    R_pulled = torch.where(fin, R_pulled, torch.zeros_like(R_pulled))
    return dict(out=out, h=h, g=g, finite=finite, infeas=infeas, merit=merit, ratio=ratio,
                S=S, ch=ch, cg=cg, K=K, R_pulled=R_pulled, J_h=J_h, J_g=J_g, gF=gF, fin=fin)


def _finish(tg, sc, X, Xn, S, c, dq_adam, clamped, lm_set, t_outer, extra):
    """Projection, the LM prediction over the step taken, best tracking, lr schedule; the
    stage-3 output dict."""
    bp = tg.bp
    Xn, _ = bp.project(Xn)
    if sc.lm and lm_set is not None:
        J, r, mask = lm_set
        S = al.lm_record_prediction(S, J, r, mask, Xn - X)
    S = al.track_best(S, X, c["merit"])
    S = al.lr_schedule(S, torch.nan_to_num(dq_adam, nan=float("inf")), sc.q_step_max, sc.lr0,
                       sc.lr_min, t_outer, sc.lr_decay_t)
    out = dict(X=Xn.detach(), S=state_dict(S), h=c["h"].detach(), g=c["g"].detach(),
               F=c["out"].F.detach(), infeas=c["infeas"], finite=c["finite"], merit=c["merit"],
               dq_adam=dq_adam, clamped=clamped, ratio=c["ratio"])
    out.update(extra)
    return out


def stage3_al(tg, sc, do_gn, X, cfg, J_q, kin, col_row, col_grad, S, gamma, T, t_outer):
    """The rest of one `al_svgd` step (the GN path of `SvgdSolver._step_al`, plus LM)."""
    learned = tg.bp.is_learned
    S = ALState(**S)
    c = _common(tg, sc, X, cfg, J_q, kin, col_row, col_grad, S, gamma, T)
    S, fin = c["S"], c["fin"]
    gC = _bmv(c["J_h"].transpose(1, 2), c["ch"]) + _bmv(c["J_g"].transpose(1, 2), c["cg"])
    gF = torch.where(fin, c["gF"], torch.zeros_like(c["gF"]))
    gAL = torch.where(fin, gF + gC, torch.zeros_like(gF))
    phi = kernels.svgd_direction(c["K"], -gF, c["R_pulled"], gamma, T)
    Xn, S = al.adam_step(X, -(phi - gAL), S)
    ## The lr schedule sees ADAM'S OWN configuration step, not the GN correction's.
    dq_adam = q_motion(learned, J_q, Xn - X)
    lm_set = None
    if do_gn:
        m_e = c["h"].shape[1]
        mask = torch.cat([torch.ones_like(c["h"]), al.active_mask(c["g"], S.mu, S.rho)], dim=1)
        J, r = gn_rows(c["h"], c["g"], c["J_h"], c["J_g"], mask)
        dx = al.gn_correction(J, r, sc.gn_delta, S.gn_lam if sc.lm else None)
        dx, clamped = clamp_step(sc, learned, dx, J_q)
        Xn = Xn + dx
        lm_set = (J, r, mask)
    else:
        clamped = torch.zeros_like(c["finite"])
    return _finish(tg, sc, X, Xn, S, c, dq_adam, clamped, lm_set, t_outer, {})


def stage3_t(tg, sc, do_gn, X, cfg, J_q, kin, col_row, col_grad, S, gamma, T, t_outer):
    """The rest of one `tsvgd` step: BOTH an al_svgd update and a tangent-space update from
    one evaluation, each particle taking the tangent one iff its scaled infeasibility is at
    or below `svgd_tsvgd_switch_infeas` (`SvgdSolver._step_t`, plus LM). `do_gn` is unused:
    tsvgd corrects onto the manifold every step."""
    learned = tg.bp.is_learned
    S = ALState(**S)
    c = _common(tg, sc, X, cfg, J_q, kin, col_row, col_grad, S, gamma, T)
    S, fin = c["S"], c["fin"]
    h, g = c["h"], c["g"]
    J_h = torch.nan_to_num(c["J_h"].detach(), nan=0.0, posinf=0.0, neginf=0.0)
    J_g = torch.nan_to_num(c["J_g"].detach(), nan=0.0, posinf=0.0, neginf=0.0)
    gI = _bmv(J_g.transpose(1, 2), c["cg"])
    gC = _bmv(J_h.transpose(1, 2), c["ch"]) + gI
    gF = torch.where(fin, c["gF"], torch.zeros_like(c["gF"]))
    gAL = torch.where(fin, gF + gC, torch.zeros_like(gF))
    K, R_pulled = c["K"], c["R_pulled"]

    ## -- the al_svgd update --
    phi_al = kernels.svgd_direction(K, -gF, R_pulled, gamma, T)
    Xn_al, S = al.adam_step(X, -(phi_al - gAL), S)
    dq_al = q_motion(learned, J_q, Xn_al - X)
    act = al.active_mask(g, S.mu, S.rho)
    ones_h = torch.ones_like(h)
    mask_al = torch.cat([ones_h, act], dim=1)
    J_al, r_al = gn_rows(h, g, J_h, J_g, mask_al)
    lam = S.gn_lam if sc.lm else None
    dx_gn, clamped_al = clamp_step(sc, learned, al.gn_correction(J_al, r_al, sc.gn_delta, lam), J_q)
    Xn_al = Xn_al + dx_gn

    ## -- the tangent-space update --
    J_A = torch.cat([J_h, J_g * act.unsqueeze(2)], dim=1)
    P = al.tangent_projector(J_A, sc.tangent_delta)
    driving = torch.where(fin, -(gF + gI), torch.zeros_like(gF))
    driving = _bmv(P, driving)
    phi = _bmv(P, kernels.svgd_direction(K, driving, R_pulled, gamma, T))
    step_lr, _ = clamp_step(sc, learned, S.lr.unsqueeze(1) * phi, J_q)
    mask_t = torch.cat([ones_h, torch.zeros_like(g)], dim=1)
    J_t, r_t = gn_rows(h, g, J_h, J_g, mask_t)
    dx_c, clamped_c = clamp_step(sc, learned, al.gn_correction(J_t, r_t, sc.gn_delta, lam), J_q)
    Xn_t = X + step_lr + dx_c
    dq_t = q_motion(learned, J_q, step_lr)

    ## -- per-particle mode --
    tangent = torch.nan_to_num(c["infeas"], nan=float("inf")) <= sc.switch_infeas
    t1 = tangent.unsqueeze(1)
    Xn = torch.where(t1, Xn_t, Xn_al)
    dq_adam = torch.where(tangent, dq_t, dq_al)
    clamped = torch.where(tangent, clamped_c, clamped_al)
    lm_set = (torch.where(tangent.view(-1, 1, 1), J_t, J_al), torch.where(t1, r_t, r_al),
              torch.where(t1, mask_t, mask_al))
    return _finish(tg, sc, X, Xn, S, c, dq_adam, clamped, lm_set, t_outer,
                   dict(tangent=tangent, P=P.detach()))


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
                               "the run uses (a new N, dtype, method or program structure?)")
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
    """The stage callables over `tg` for `mode`: `{"s1", "s2", ("s3", do_gn)}`."""
    s3 = stage3_al if sc.method == "al_svgd" else stage3_t

    def f1(X):
        return stage1(tg, X)

    def f2(X, cfg):
        return stage2(tg, X, cfg)

    def make3(do_gn):
        def f3(X, cfg, J_q, kin, col_row, col_grad, S, gamma, T, t_outer):
            return s3(tg, sc, do_gn, X, cfg, J_q, kin, col_row, col_grad, S, gamma, T, t_outer)
        return f3

    fns = {"s1": f1, "s2": f2, ("s3", True): make3(True), ("s3", False): make3(False)}
    if mode == "eager":
        return fns
    ## `triton.persistent_reductions=False`, scoped to these compiles: with persistent
    ## reductions on, stage 3's largest fused reduction kernel crashes Triton 3.6's
    ## `make_ttgir` pass ("PassManager::run failed") on an sm_86 GPU under torch 2.11
    ## (2026-10-08); `max_fusion_size=16` avoids it too, at a similar compile time.
    ## Every entry's stage closures share ONE code object per stage, and dynamo caches per
    ## code object, so the default `recompile_limit` of 8 is reached after eight (structure,
    ## N, dtype, method) entries in one process -- after which dynamo SILENTLY runs the stage
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
                    f"{tg.dtype}, {sc.method}) and steps are frozen -- WarmUpSvgdStep must run "
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

    def s3(self, do_gn, *args):
        with torch.no_grad():
            return self.fns[("s3", bool(do_gn))](*args)

    def graphs_captured(self):
        if self.mode != "graphed":
            return 0
        return sum(1 for f in self.fns.values() if getattr(f, "graph", None) is not None)
