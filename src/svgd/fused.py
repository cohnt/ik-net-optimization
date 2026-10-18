"""The svgd step split around the collision row: three pure stages, eager / compiled / graphed.

One `al_svgd` step on the fielded robots (`BatchedProgram.has_analytic_row_jacobians`) is, with
the only host work -- Drake's exact collision row, in this process -- between the stages:

    stage 1 (GPU)   X -> cfg = q(X), q_plant                     the flow forward, no graph
    host            q_plant -> CPU
    stage 2 (GPU)   (X, cfg) -> J_q = dq/dX, kinematics + frame Jacobians    (launched, async)
    host            the collision row (row, d row / d q_plant), a per-particle Drake loop
    stage 3 (GPU)   rows, scaling, the objective and constraint gradients, the kernel, the SVGD
                    direction phi and the GN Hessian H (`direction_and_hessian`), the step --
                    the GN solve `solve_ex` (capturable under the cuSOLVER pin) or
                    svgd_lr / ||H||_F -- and the clamp onto the true bounds (`step_from`)

On a robot whose plant vector is not the padded configuration (the soft PCS arm, since
2026-10-12) stage 2's kinematics are `BatchedProgram._kinematics_cfg_jacobians`: the frame
Jacobians and `J_plant = d q_plant / d cfg` by forward mode through the robot's exact torch
map, and stage 3 pulls the collision row's plant gradient back through `J_plant`.

Stage 2 does not depend on the collision row, so it is launched before the Drake loop runs;
on the GPU it executes while the host is in Drake (asynchronously in graphed mode). Nothing
eigen-related is in the step: the exact lambda_max(H) is a diagnostic the solver takes eagerly
at the dual-update checks (`eigvalsh` synchronises with the host to check for failure, which a
CUDA graph cannot capture -- probed 2026-10-09 at every batch size, with either linalg backend).

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
from src.svgd.kinematics_from_plant import canonical_quat

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


## The kernel spaces (`svgd_kernel`). `q` is fielded; the rest are the kernel-space screen's
## (2026-10-11). The ones naming a latent or conditioning block exist only on the learned arm.
KERNEL_SPACES = ("none", "q", "x", "z", "c", "cz", "qz", "pose")
LATENT_KERNELS = ("z", "c", "cz", "qz")


def check_kernel_for_arm(kernel, is_learned, arm="", formulation=None):
    """Refuse, at construction, a kernel space the arm has no block for: `z`, `c`, `cz` and
    `qz` name the learned arm's latent / conditioning blocks, which neither joint space nor the
    analytic arm (`formulation="analytic"`: decision variables `[xyz_rpy | psi]`) has."""
    if kernel not in KERNEL_SPACES:
        raise ValueError(f"svgd_kernel={kernel!r} is not one of {KERNEL_SPACES}")
    if kernel in LATENT_KERNELS and not is_learned and formulation == "analytic":
        raise ValueError(f"svgd_kernel={kernel!r} needs the learned arm's conditioning-pose / "
                         f"latent blocks; the analytic arm {arm} has neither (its decision "
                         f"variables are the pose xyz_rpy and the redundancy parameter psi). "
                         f"Use x (the normalised [xyz_rpy | psi]) or q (the configuration, "
                         f"pulled back through the map), pose or none")
    if kernel in LATENT_KERNELS and not is_learned:
        raise ValueError(f"svgd_kernel={kernel!r} needs the learned arm's conditioning-pose / "
                         f"latent blocks; the joint-space arm {arm} has neither (its decision "
                         f"variables are q alone). Use q, x (== q normalised by the half-range), "
                         f"pose or none")


@dataclass(frozen=True)
class StepConfig:
    """The option values the stages read as Python constants (part of the cache key)."""
    kernel: str                 # svgd_kernel: KERNEL_SPACES
    bandwidth_floor: float      # svgd_bandwidth_floor
    inside: bool                # svgd_constraint_inside_kernel
    lr: float                   # svgd_lr (a fraction of the Newton / Lipschitz step)
    rho: float                  # svgd_rho, the penalty: fixed, shared by every particle
    metric: str = "gn"          # svgd_metric: "gn" (Stein variational Newton) | "identity"
    gn_lm: float = 0.0          # svgd_gn_lm, the Levenberg damping delta of the GN metric
    ## svgd_resample != "native": the step reads the AL state's live weight (`ALState.live`)
    ## and leaves dropped particles out. False under `native`, whose stages are then exactly
    ## the pre-2026-10-11 code (the live tensor is an input they never read).
    live_weights: bool = False

    @staticmethod
    def from_options(opts):
        return StepConfig(kernel=str(opts.svgd_kernel),
                          bandwidth_floor=float(opts.svgd_bandwidth_floor),
                          inside=bool(opts.svgd_constraint_inside_kernel),
                          lr=float(opts.svgd_lr), rho=float(opts.svgd_rho),
                          metric=str(opts.svgd_metric),
                          gn_lm=float(opts.svgd_gn_lm),
                          live_weights=str(getattr(opts, "svgd_resample", "native")) != "native")


## ------------------------------------------------------------------------------------ ##
##                      structure signature and constant rebinding                      ##
## ------------------------------------------------------------------------------------ ##

## Attributes never walked: the program and its options (read outside the stages), the
## collision evaluator (called by the driver), the networks (shared by identity: recorded as an id), and
## values that are program-specific but never read by a stage (the native conditioning
## pose, the decision-variable objects, the calibrated flow-frame transform, RowSpec bounds
## -- the bounds the stages read are the `_eq` / `_lo` / `_hi` TENSORS, which are copied).
_SKIP_ATTRS = {"program", "options", "collision", "model", "shared_model", "_native_c",
               "lumped_vars", "X_ee_flow", "profile"}
_SKIP_CLASS_ATTRS = {("RowSpec", "lb"), ("RowSpec", "ub")}


## The Python FLOATS a stage reads (by attribute name): their values are baked into a
## compiled / captured stage, so they belong to the signature. Every other float in the
## target is construction-time data -- the weld pose of a grasp scene's mug in the
## kinematic tree, the pre-composed frame offsets, the calibrated flow frame -- whose tensor
## form is what the stages read and what `copy_constants` rebinds; recording those values
## made every mug target a new structure. Ints, bools, strings and None (sizes, slots,
## kinds, flags) are always recorded. Adding a float read to a stage means adding its name.
_READ_FLOATS = {"_w_centering", "w_cfg", "collision_scale", "c", "correction_bound", "mug_height",
                "_w_logdet"}


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


def kernel_terms(sc, cfg, finite, X, live=None, tg=None, pose=None, pose_jac=None):
    """`(K, R, kernel_on, bw)`: the kernel matrix, its repulsion, whether a kernel is on, and the
    bandwidth(s) `bw [n_features]` it used (`[0]` under `none`).

    `svgd_kernel = q` (fielded): the RBF on the configuration, median bandwidth, and `R` its
    repulsion IN CONFIGURATION SPACE `[N, ndof]` (`kernels.rbf_terms`), pulled back to y by
    `repulsion_y` through the flow's J_q -- the pre-2026-10-11 code, unchanged. Every other
    space (`kernel_features`) returns `R` as a pair `(R_cfg [N, ndof] or None, R_y [N, n] or
    None)`: the configuration-space part (the `q` factor of `qz`, and `pose`'s repulsion pulled
    back through `pose_jac = d pose / d cfg`) and the part acting DIRECTLY on the normalised
    decision variables y (`x`, `z`, `c`, `cz`, the `z` factor of `qz`). A product kernel
    (`cz`, `qz`) is `prod_k exp(-|u_k,i - u_k,j|^2 / h_k)`, each factor with its own median
    bandwidth, and its repulsion the sum of the factors' terms (grad log K = sum_k grad log K_k).

    Non-finite particles are isolated (their K row and column are the identity's, their R zero)
    so a NaN cannot spread through the kernel average. `svgd_kernel = none` (or N = 1): `K = I`,
    `R = 0`. `live [N]` bool (`svgd_resample = drop` / `keep`): dropped particles are isolated
    the same way -- `K_ij = 0` for a dropped j, so they enter no other particle's average or
    repulsion -- and left out of the median bandwidth(s). `tg` (the target: region scale `s`,
    the block layout) is needed by every space but `q` and `none`; `pose` / `pose_jac` by `pose`."""
    N = X.shape[0]
    dtype, device = X.dtype, X.device
    n_feat = {"none": 0, "cz": 2, "qz": 2}.get(sc.kernel, 1)
    if sc.kernel == "none" or N < 2:
        bw = torch.full((n_feat,), float(sc.bandwidth_floor) ** 2, dtype=dtype, device=device)
        return (torch.eye(N, dtype=dtype, device=device),
                torch.zeros_like(cfg), sc.kernel != "none", bw)
    if sc.kernel == "q":
        y = cfg.detach()
        fin = finite & torch.isfinite(y).all(dim=1)
        if live is not None:
            fin = fin & live
        y0 = torch.where(fin.unsqueeze(1), y, torch.zeros_like(y))
        hbw = kernels.median_bandwidth(kernels.pairwise_sqdist(y0), N, floor=sc.bandwidth_floor,
                                       live=live)
        K, _ = kernels.rbf_terms(y0, hbw)
        mask = fin.unsqueeze(1) & fin.unsqueeze(0)
        K = torch.where(mask, K, torch.eye(N, dtype=dtype, device=device))
        R = (2.0 / hbw) * (K.sum(dim=1, keepdim=True) * y0 - K @ y0)
        R = torch.where(fin.unsqueeze(1), R, torch.zeros_like(R))
        return K, R, True, hbw.reshape(1)
    feats = kernel_features(sc.kernel, tg, X, cfg, pose)
    fin = finite
    for u, _, _ in feats:
        fin = fin & torch.isfinite(u).all(dim=1)
    if live is not None:
        fin = fin & live
    f1 = fin.unsqueeze(1)
    us, hs = [], []
    expo = torch.zeros(N, N, dtype=dtype, device=device)
    for u, _, _ in feats:
        u0 = torch.where(f1, u, torch.zeros_like(u))
        sq = kernels.pairwise_sqdist(u0)
        h = kernels.median_bandwidth(sq, N, floor=sc.bandwidth_floor, live=live)
        expo = expo + sq / h
        us.append(u0)
        hs.append(h)
    K = torch.exp(-expo)
    mask = fin.unsqueeze(1) & fin.unsqueeze(0)
    K = torch.where(mask, K, torch.eye(N, dtype=dtype, device=device))
    Ksum = K.sum(dim=1, keepdim=True)
    n = X.shape[1]
    R_cfg, R_y = None, None
    for u0, h, (_, route, sl) in zip(us, hs, feats):
        R = (2.0 / h) * (Ksum * u0 - K @ u0)
        R = torch.where(f1, R, torch.zeros_like(R))
        if route == "cfg":
            R_cfg = R if R_cfg is None else R_cfg + R
        elif route == "pose":
            Jp = torch.where(f1.unsqueeze(2), torch.nan_to_num(pose_jac.detach(), nan=0.0,
                             posinf=0.0, neginf=0.0), torch.zeros_like(pose_jac))
            Rc = _bmv(Jp.transpose(1, 2), R)
            R_cfg = Rc if R_cfg is None else R_cfg + Rc
        else:                                       # "y": a slice of the decision variables
            parts = [R]
            if sl.start > 0:
                parts.insert(0, torch.zeros(N, sl.start, dtype=dtype, device=device))
            if sl.stop < n:
                parts.append(torch.zeros(N, n - sl.stop, dtype=dtype, device=device))
            Ry = torch.cat(parts, dim=1) if len(parts) > 1 else R
            R_y = Ry if R_y is None else R_y + Ry
    return K, (R_cfg, R_y), True, torch.stack(hs)


def kernel_blocks(tg):
    """`{"c": slice, "z": slice}`: the conditioning-pose and latent blocks of the learned arm's
    lumped decision vector `[c6 | z | q_c]` (`BatchedProgram._split`); {} on joint space."""
    bp = tg.bp
    if not bp.is_learned:
        return {}
    return {"c": slice(0, 6), "z": slice(6, 6 + int(bp.width))}


def kernel_features(space, tg, X, cfg, pose=None):
    """`[(u [N, d], route, slice)]`, the factor(s) of kernel `space` (every space but `none`):
    `u` the coordinates the RBF factor measures distance in, `route` how its repulsion reaches
    y -- "cfg" (through the flow's J_q), "pose" (through d pose / d cfg, then J_q) or "y"
    (directly on the decision-variable `slice`). All y-coordinates are the normalised
    `y = x / s` the step moves in. Also what `feasible_spread_kernel` measures."""
    if space == "q":
        return [(cfg.detach(), "cfg", None)]
    if space == "pose":
        return [(pose.detach(), "pose", None)]
    y = X.detach() / tg.s.unsqueeze(0)
    if space == "x":
        return [(y, "y", slice(0, y.shape[1]))]
    blk = kernel_blocks(tg)
    if not blk:
        raise ValueError(f"svgd_kernel={space!r} needs the learned arm's blocks")
    zs, cs = blk["z"], blk["c"]
    if space == "z":
        return [(y[:, zs], "y", zs)]
    if space == "c":
        return [(y[:, cs], "y", cs)]
    if space == "cz":
        return [(y[:, cs], "y", cs), (y[:, zs], "y", zs)]
    if space == "qz":
        return [(cfg.detach(), "cfg", None), (y[:, zs], "y", zs)]
    raise ValueError(f"unknown kernel space {space!r}")


def pose_and_jacobian(quat, pos, Jp, Jw, ndof):
    """`(pose [N, 7], d pose / d cfg [N, 7, ndof])` of the task frame: the 7-vector
    `[position; quaternion wxyz canonical w >= 0]` (`pose7`) and its derivative from the frame's
    geometric Jacobian -- `Jp` for the position, and for the quaternion the world angular
    velocity's `Jw` through `d quat = 1/2 (0, omega) (x) quat`, i.e. `1/2 [-v^T; w I - [v]x] Jw`,
    evaluated at the CANONICAL quaternion (the sign flip is piecewise constant, so this is the
    canonical quaternion's own derivative). Static shapes; the plant columns beyond the
    configuration (the gripper pad) are dropped."""
    qc = canonical_quat(quat)
    w, v = qc[:, 0], qc[:, 1:]
    zero = torch.zeros_like(w)
    vx = torch.stack([torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
                      torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
                      torch.stack([-v[:, 1], v[:, 0], zero], dim=1)], dim=1)       # [v]x
    eye = torch.eye(3, dtype=qc.dtype, device=qc.device).unsqueeze(0)
    E = 0.5 * torch.cat([-v.unsqueeze(1), w.view(-1, 1, 1) * eye - vx], dim=1)  # [N, 4, 3]
    J = torch.cat([Jp[:, :, :ndof], E @ Jw[:, :, :ndof]], dim=1)
    return torch.cat([pos, qc], dim=1), J


def gn_hessian(s, w_cfg, H_fx, J_q, J_h, J_g, g, mu, rho):
    """`H_y [N, n, n]`, the Gauss-Newton Hessian of the augmented Lagrangian in the
    normalised coordinates `y = X / s`:

        H_x = w_cfg J_q^T J_q + H_fx + rho J_h^T J_h + rho J_g,act^T J_g,act,   H_y = S H_x S,

    `w_cfg J_q^T J_q` the joint-centering cost through dq/dx, `H_fx` the constant Hessian of
    the costs quadratic in x (the correction penalty, ...), and the active set the standard PHR
    one, `mu + rho g > 0`. `J_*` are the rows' Jacobians in the AL's units w.r.t. X."""
    sv = s.view(1, 1, -1)
    Jq, Jh, Jg = J_q * sv, J_h * sv, J_g * sv
    act = ((mu + rho * g) > 0).to(Jg.dtype).unsqueeze(2)
    H = (w_cfg * (Jq.transpose(1, 2) @ Jq) + (s.view(-1, 1) * H_fx * s.view(1, -1)).unsqueeze(0)
         + rho * (Jh.transpose(1, 2) @ Jh) + rho * (Jg.transpose(1, 2) @ (act * Jg)))
    return 0.5 * (H + H.transpose(1, 2))


def repulsion_y(tg, X, J_q, R, kernel_on, finite):
    """The kernel's repulsion in the normalised coordinates y, `[N, n]`, zero at non-finite
    particles: `s * J_q^T R` for a configuration-space `R` (`svgd_kernel = q`, the fielded
    code), and for the other spaces' pair `(R_cfg, R_y)` the configuration part pulled back
    the same way plus the part acting directly on y (`kernel_terms`)."""
    sv = tg.s.unsqueeze(0)
    fin = finite.unsqueeze(1)
    zero = torch.zeros_like(X)
    if isinstance(R, tuple):
        R_cfg, R_dir = R
        R_out = zero
        if R_cfg is not None:
            R_out = sv * _bmv(J_q.transpose(1, 2), R_cfg)
        if R_dir is not None:
            R_out = R_out + R_dir
        return torch.where(fin, R_out, zero)
    R_x = _bmv(J_q.transpose(1, 2), R) if kernel_on else torch.zeros_like(X)
    return torch.where(fin, sv * R_x, zero)


def direction_and_hessian(tg, sc, X, gF, J_q, J_h, J_g, h, g, finite, K, R, kernel_on, S,
                          live=None):
    """The SVGD direction `phi` in `y = X / s` (`kernels.stein_direction`, the Tabor-Hermans
    form) and the GN Hessian `H_y` (`gn_hessian`) at every particle, and the median over
    particles of `|repulsion| / |drive|` (the kernel's actual share of phi). Non-finite
    particles get `phi = 0`, `H = I`. `live [N]` bool: dropped particles are treated as
    non-finite, and the kernel average is over the `n_live` live particles (`avg_scale =
    N / n_live`, exactly 1.0 with every particle live)."""
    N, n = X.shape
    avg_scale = None
    if live is not None:
        finite = finite & live
        n_live = live.sum()
        avg_scale = torch.where(n_live == N, torch.ones((), dtype=X.dtype, device=X.device),
                                N / torch.clamp(n_live, min=1).to(X.dtype))
    ch, cg = al.al_constraint_grad_coefficients(h, g, S, sc.rho)
    gC = _bmv(J_h.transpose(1, 2), ch) + _bmv(J_g.transpose(1, 2), cg)
    sv = tg.s.unsqueeze(0)
    fin = finite.unsqueeze(1)
    zero = torch.zeros_like(X)
    gF_y = torch.where(fin, sv * gF, zero)
    gC_y = torch.where(fin, sv * gC, zero)
    R_y = repulsion_y(tg, X, J_q, R, kernel_on, finite)
    use_kernel = sc.kernel != "none"
    phi = kernels.stein_direction(K, R_y, gF_y, gC_y, inside=sc.inside, kernel=use_kernel,
                                  avg_scale=avg_scale)
    rep = R_y / N if use_kernel else zero
    if use_kernel and avg_scale is not None:
        rep = rep * avg_scale
    ratio = rep.norm(dim=1) / torch.clamp((phi - rep).norm(dim=1), min=torch.finfo(X.dtype).tiny)
    ratio = torch.nanmedian(torch.where(finite, ratio, torch.full_like(ratio, float("nan"))))
    H = gn_hessian(tg.s, tg.w_cfg, tg.H_fx, J_q, J_h, J_g, g, S.mu, sc.rho)
    eye = torch.eye(n, dtype=X.dtype, device=X.device).unsqueeze(0).expand(N, -1, -1)
    ok = finite & torch.isfinite(H).flatten(1).all(dim=1)
    H = torch.where(ok.view(-1, 1, 1), H, eye)
    return phi, H, ratio


def step_from(tg, sc, X, phi, H, live=None):
    """The update in `y = X / s`, then the clamp onto the true bounds B:

        svgd_metric = "gn":       dy_i = svgd_lr (H_i + delta I)^-1 phi_i     (Stein variational
                                  Newton, block-diagonal; delta = svgd_gn_lm)
        svgd_metric = "identity": dy_i = (svgd_lr / ||H_i||_F) phi_i         (the per-particle
                                  step; ||H||_F >= lambda_max(H), a safe Lipschitz bound and a
                                  pure reduction, so the step stays graph-capturable)

    Returns `(X_new, clip [N], n_clip [N])`: the clamp's distance per particle in y and how
    many coordinates it moved. A failed solve (or a non-finite step) is a zero step. `live
    [N]` bool: a dropped particle is FROZEN -- `X_new = X` exactly, no clamp, nothing counted."""
    n = X.shape[1]
    if sc.metric == "gn":
        eye = torch.eye(n, dtype=X.dtype, device=X.device).unsqueeze(0)
        Y, info = torch.linalg.solve_ex(H + sc.gn_lm * eye, phi.unsqueeze(2), check_errors=False)
        dy = sc.lr * Y.squeeze(2)
        good = (info == 0) & torch.isfinite(dy).all(dim=1)
    else:
        fro = torch.clamp(H.flatten(1).norm(dim=1), min=torch.finfo(X.dtype).tiny)
        dy = (sc.lr / fro).unsqueeze(1) * phi
        good = torch.isfinite(dy).all(dim=1)
    dy = torch.where(good.unsqueeze(1), dy, torch.zeros_like(dy))
    sv = tg.s.unsqueeze(0)
    Xn = X + sv * dy
    Xp, _ = tg.bp.project(Xn)
    if live is not None:
        l1 = live.unsqueeze(1)
        Xp = torch.where(l1, Xp, X)
        Xn = torch.where(l1, Xn, X)
    moved = (Xp != Xn) & torch.isfinite(Xn)
    clip = torch.where(moved, (Xp - Xn) / sv, torch.zeros_like(X)).norm(dim=1)
    return Xp.detach(), clip, moved.sum(dim=1)


def lambda_max(H):
    """`lambda_max(H_i)` exactly (`eigvalsh`, batched): a DIAGNOSTIC, never in the step.
    EAGER ONLY: eigvalsh synchronises with the host to check for failure, which a CUDA graph
    cannot capture. Taken in float64; a batch the eigensolver fails on (an ill-conditioned or
    degenerate float32 H, seen 2026-10-10) records NaN rather than ending the solve."""
    try:
        return torch.linalg.eigvalsh(H.double())[:, -1].to(H.dtype)
    except torch.linalg.LinAlgError:
        return torch.full(H.shape[:1], float("nan"), dtype=H.dtype, device=H.device)


def live_mask(sc, S):
    """The live set the step reads, `[N]` bool, or None -- always None under `native`
    (`sc.live_weights` False), so that mode's stages never touch `S.live`."""
    if not sc.live_weights or S.live is None:
        return None
    return S.live > 0


def state_dict(S):
    return {f.name: getattr(S, f.name) for f in dataclass_fields(ALState)}


def stage1(tg, X):
    """`(cfg, q_plant)`: the configuration and the plant vector the collision row needs."""
    bp = tg.bp
    with torch.no_grad():
        cfg = bp._config(X)
        return cfg, bp.config_to_plant_q(cfg)


def stage2(tg, X, cfg):
    """`(J_q [N, ndof, n], kin)`: the flow Jacobian (one `jacrev`, `ndof` reverse passes,
    vmapped -- the particles are independent through the flow, so the Jacobian of the
    batch-sum is the per-particle one) and the kinematics with the frame Jacobians at the
    stage-1 configuration. Identity on joint space. On the analytic arm the closed form's
    Jacobian the same way, and `kin` carries a fifth entry `(u, J_u)`, the map's aux output.

    With the flow log-det term (`bp.has_flow_logdet`) the jacrev takes ONE extra row, the
    log-det's, from the same network call, and a third entry `(ld [N], J_ld [N, n])` is
    returned for stage 3. Only then: an extra row is not bit-identical in the q rows."""
    bp = tg.bp
    N = X.shape[0]
    aux = None
    flow_ld = None
    if bp.is_learned and bp.has_flow_logdet:
        def f(Xv):
            cfg_v, ld_v = bp._config_and_logdet(Xv)
            return torch.cat([cfg_v, ld_v.unsqueeze(1)], dim=1).sum(dim=0), ld_v
        J, ld = torch.func.jacrev(f, has_aux=True)(X)                       # [ndof + 1, N, n]
        J_q = J[:bp.ndof].permute(1, 0, 2).contiguous()
        flow_ld = (ld.detach(), J[bp.ndof].contiguous())
    elif bp.is_learned:
        J = torch.func.jacrev(lambda Xv: bp._config(Xv).sum(dim=0))(X)     # [ndof, N, n]
        J_q = J.permute(1, 0, 2).contiguous()
    elif bp.is_analytic:
        ## The closed-form map and its aux output (the reachability rows' pre-clip arguments)
        ## from one vmapped jacrev; `(u, J_u)` rides on `kin` (`BatchedProgram.assemble`).
        J_q, J_u, u = bp.map_jacobians(X)
        aux = (u.detach(), J_u.detach())
    else:
        eye = torch.eye(bp.ndof, bp.nvars, dtype=X.dtype, device=X.device)
        J_q = eye.unsqueeze(0).expand(N, -1, -1).contiguous()
    with torch.no_grad():
        kin = bp.kinematics(cfg.detach(), row_jacobians=True)
    if aux is not None:
        kin = kin + (aux,)
    if flow_ld is not None:
        return J_q, kin, flow_ld
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


def stage3(tg, sc, X, cfg, J_q, kin, col_row, col_grad, S, flow_ld=None):
    """The rest of one step: the rows and costs, their scaling, the gradients (closed form
    downstream of the configuration, the flow's `J_q` from stage 2), the kernel, the SVGD
    direction and the GN Hessian (`direction_and_hessian`), and the step (`step_from`). `H`
    is returned for the solver's eager lambda_max diagnostic at the checks. `flow_ld = (ld,
    J_ld)` is stage 2's third entry, present exactly when the program carries the flow
    log-det term."""
    bp = tg.bp
    S = ALState(**S)
    ld, J_ld = flow_ld if flow_ld is not None else (None, None)
    out = bp.assemble(X, cfg, cfg, kin, col_row, col_grad, ld=ld)
    h, g, finite, infeas = _scale(tg, out)
    dFc, dFx = bp.cost_gradient_parts(X, cfg)
    gF = _bmv(J_q.transpose(1, 2), dFc) + dFx
    if J_ld is not None:
        ## The log-det enters the DRIVE only, with NO block in the SVN metric (option (a)):
        ## its Hessian would need second derivatives through the flow, and the `|z|^2/2`
        ## half of the density (`latent_cost_weight`, when set) already enters `H_fx`
        ## through `LatentCost`'s quadratic.
        gF = gF + bp._w_logdet * J_ld
    J_h, J_g = tg.row_jacobians_out(out, X, J_q)
    live = live_mask(sc, S)
    pose = pose_jac = None
    if sc.kernel == "pose":
        (quat, pos), (Jp, Jw) = kin[1], kin[3][:2]
        pose, pose_jac = pose_and_jacobian(quat, pos, Jp, Jw, bp.ndof)
    K, R, kernel_on, bw = kernel_terms(sc, cfg, finite, X, live, tg=tg, pose=pose,
                                       pose_jac=pose_jac)
    phi, H, ratio = direction_and_hessian(tg, sc, X, gF, J_q, J_h, J_g, h, g, finite, K, R,
                                          kernel_on, S, live)
    Xn, clip, n_clip = step_from(tg, sc, X, phi, H, live)
    return dict(X=Xn, clip=clip, n_clip=n_clip, H=H, ratio=ratio, h=h.detach(), g=g.detach(),
                F=out.F.detach(), infeas=infeas, finite=finite, bw=bw)


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

    def f3(X, cfg, J_q, kin, col_row, col_grad, S, *flow_ld):
        return stage3(tg, sc, X, cfg, J_q, kin, col_row, col_grad, S, *flow_ld)

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
