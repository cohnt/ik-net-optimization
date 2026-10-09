"""The `svgd` solver: `SvgdSolver(program).solve() -> SvgdResult`.

PLAIN AUGMENTED-LAGRANGIAN SVGD (`svgd_method = "al_svgd"`, the only method), on the program
as written: replayed for N particles by `BatchedProgram` (`src/svgd/batched_program.py`) and
turned into a per-particle PHR augmented Lagrangian by `src/svgd/al.py`. Nothing here knows
what a decision variable means beyond two formulation-agnostic facts: every arm has a
configuration `q` (the kernel's space), and every block of variables has a region whose
half-width normalises it (the coordinates below).

    init -> (optional CEM warm-up) -> swarm -> select -> Drake re-check

THE METHOD, every part named (`docs/svgd-solver.md` carries the option table):

 1. Merit, per particle: `L_rho(x; lam, mu) = f + lam.h~ + (rho/2)|h~|^2
    + (1/(2 rho)) sum [max(0, mu + rho g~)^2 - mu^2]`; target density `exp(-L_rho / T)`,
    `T = svgd_temperature` and `rho = svgd_rho` FIXED and shared by every particle. The
    augmented Lagrangian is the FORMULATION and SVGD the optimizer: one dynamics.
 2. Rows: every equality `h` and inequality `g` of the batched program divided by
    `tol = acceptable_constr_viol_tol`, so `v = ||[h~ ; max(g~, 0)]||_inf <= 1` is "feasible at
    the harness gate" (the c box and the latent ball are inequality ROWS; the true variable
    bounds B -- the correction box on the learned arm, the joint limits on joint space -- are
    not rows but a clamp).
 3. Coordinates: `y = x / s`, `s` the per-block region half-widths (`region_scale`).
 4. Step (Tabor-Hermans "Q method", `kernels.stein_direction`, `fused.update_step`):
    `phi_i = (1/N) sum_j [K(q_j, q_i)(-grad_y f_j / T) + grad_{y_j} K(q_j, q_i)]
             - (1/T) grad_y (L - f)_i`,
    `y_i <- clamp_B(y_i + (svgd_lr / rho) phi_i)`. RBF kernel on q, median bandwidth; the
    repulsion pulled back to y through the flow's VJP. `svgd_kernel = none` drops both kernel
    terms; `svgd_constraint_inside_kernel` puts the whole `-grad L / T` under the average.
 5. Every K = `svgd_inner_iters` steps a CHECK: the dual-ascent step on every particle,
    unconditionally (`al.dual_update`: `lam_i += rho h~_i`, `mu_i = max(0, mu_i + rho g~_i)`,
    clipped to `+-svgd_multiplier_max`).
 6. At every check, particles with `|q|_inf > svgd_resample_q_max` or a non-finite row are
    redrawn from the arm's NATIVE start distribution with zero multipliers (`n_resampled`).
 7. Stop: the clock (`max_wall_time` less `svgd_time_reserve`), the outer-step cap
    (`svgd_outer_iters`, or `max_iter`), or -- once a particle is feasible at the gate --
    `svgd_stop_patience` consecutive checks at which the best feasible particle's objective
    improved by less than `svgd_stop_rel` (relative): `stop_reason` converged / wall_clock /
    step_cap.
 8. Select: the particles feasible on the batched rows, by objective; the top
    `svgd_recheck_topk` re-checked exactly in Drake (`prog.EvalBinding`, float64, same x);
    the first passer is returned, else the smallest Drake violation, scored infeasible.

DERIVATIVES: THE ONLY BACKWARD IS THROUGH THE FLOW. On the fielded robots
(`BatchedProgram.has_analytic_row_jacobians`) every derivative downstream of the
configuration is closed form -- the task rows from the frame's geometric Jacobian, the
collision row from the pool's own stored gradient, the joint-limit rows the identity, the
region rows and the costs in `x` directly -- and the flow's `J_q` is one vmapped `jacrev`.
A robot behind its own `BodyPoseProvider` falls back to autograd through its kinematics
(`_Target.analytic = False`, `_step_eager`).

THE INIT PROTOCOL cannot be read off the program (the harness called `SetStartFromQ` or
`SetNativeStart` before `Solve()`), so particle 0 is ALWAYS the program's initial guess,
exactly, never clipped (the +-5 latent-clip trap). The other N-1 are drawn around it
(`svgd_paired_init = "jitter"`: `y0 + svgd_jitter * xi` in the normalised coordinates) or from
the arm's native distribution (`"native"`: learned `c = native_c, z ~ N(0, I), q_c = 0`; joint
space uniform in `ConfigLimits`, kept collision-free by `ParallelCollisionChecker` with
bounded redraws and a documented fallback), then clamped onto the true bounds with the clip
distance recorded. The seed is `svgd_seed` mixed with a CRC of the initial guess.

THE COLLISION POOL is cached per `(SceneSpec, workers)` in this process (`_POOLS`, ONE pool:
`POOL_CACHE_SIZE = 1`, the old one closed before a new scene's is spawned): the benchmark builds
a new program per cell and spawning workers costs seconds, while the scene is the same for every
cell of a pose grid and for both arms of a grasp cell. `svgd_collision_workers = None` resolves
to `cpu_count // PROCS` in a Slurm job and `min(cpu_count // PROCS, 8)` elsewhere
(`resolve_workers`); every pool is admitted by the memory guard in `collision_backend`.

THE STEP IS SPLIT AT THE POOL (`src/svgd/fused.py`, `_step_split`) on the fielded robots:
stage 1 (the configuration) -> the pool dispatched -> stage 2 (the flow Jacobian, the
kinematics) while Drake runs -> the pool collected -> stage 3 (the rest). `svgd_compile`
compiles the three stages and `svgd_cuda_graph` replays them as CUDA graphs, captured by
`warm_up` (`WarmUpSvgdStep`) before any timed cell and frozen by `run_grid`;
`svgd_pool_overlap=False` collects the pool before stage 2. `extras["step_mode"]` records
which mode ran.
"""

import atexit
import math
import os
import time
import warnings
import zlib
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from src.svgd import al, fused
from src.svgd.al import ALState
from src.svgd.batched_program import EXTRA_ROW_KEYS, GENERIC_BINDING, BatchedProgram
from src.svgd.collision_backend import DrakeCollisionPool, ParallelCollisionChecker, SceneSpec
from src.svgd.result import (STATUS_CONVERGED, STATUS_INFEASIBLE, STATUS_NAN, STATUS_STEP_CAP,
                             STATUS_WALL_CLOCK, SvgdResult, SvgdSolverDetails, status_name,
                             write_log)

POOL_CACHE_SIZE = 1           # ONE pool per process: a new scene closes the old pool first
LOCAL_WORKERS_MAX = 8         # the default worker count's cap off a Slurm allocation
NATIVE_DRAW_ROUNDS = 20       # bounded rejection sampling for the joint-space native start
LATENT_BOX = 5.0              # the latent's +-5 box (`LatentBoxConstraint`), where no trust radius


## ------------------------------------------------------------------------------------ ##
##                         shared collision pools and checkers                          ##
## ------------------------------------------------------------------------------------ ##

_POOLS = OrderedDict()        # (SceneSpec, workers) -> DrakeCollisionPool
_CHECKERS = OrderedDict()     # SceneSpec -> ParallelCollisionChecker


def resolve_workers(option):
    """`svgd_collision_workers`, or, when None, `max(1, cpu_count // PROCS)` inside a Slurm
    job (`SLURM_JOB_ID` set: a dedicated node) and `min(that, LOCAL_WORKERS_MAX)` anywhere
    else (a shared workstation, where each worker is a whole Drake scene in memory)."""
    if option is not None:
        return max(1, int(option))
    try:
        procs = max(1, int(os.environ.get("PROCS", "1") or 1))
    except ValueError:
        procs = 1
    n = max(1, (os.cpu_count() or 2) // procs)
    return n if os.environ.get("SLURM_JOB_ID") else min(n, LOCAL_WORKERS_MAX)


def _spawn_pool(spec, workers):
    """Spawn the pool with the GPU HIDDEN from its workers.

    Under the `spawn` start method every worker re-imports the main script as
    `__mp_main__`, and a benchmark script's (or a test's) module-level imports pull in
    `src.generic_program` -> ikflow -> `jrl.config`, whose import initialises CUDA. So
    twenty workers that only ever run Drake on the CPU each took a ~0.5 GB CUDA context,
    and in a parent already holding the GPU they died at import with `CUDA error: out of
    memory` -- surfacing as "worker k died (ConnectionResetError)" with the worker's own
    traceback lost to `HiddenPrints` (the plumbing test, 2026-10-08). The environment is
    what the child receives at exec, so it is masked for the spawn and restored after;
    the parent's own CUDA state is untouched (the variable is read at CUDA init only).
    """
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        return DrakeCollisionPool(spec, workers=int(workers))
    finally:
        if old is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def shared_pool(spec, workers):
    """The process's pool for `(spec, workers)`, spawned on first use. At most
    `POOL_CACHE_SIZE` pools are alive: the least recently used are CLOSED BEFORE a new one is
    spawned, so two scenes' workers never coexist (the memory guard's registry would refuse
    the second anyway on a tight host)."""
    key = (spec, int(workers))
    pool = _POOLS.get(key)
    if pool is not None and not pool.closed and all(pool.alive()):
        _POOLS.move_to_end(key)
        return pool
    if pool is not None:
        pool.close()
        del _POOLS[key]
    while len(_POOLS) >= POOL_CACHE_SIZE:
        _, old = _POOLS.popitem(last=False)
        old.close()
    pool = _spawn_pool(spec, workers)
    _POOLS[key] = pool
    return pool


def shared_checker(spec):
    checker = _CHECKERS.get(spec)
    if checker is None:
        checker = ParallelCollisionChecker(spec)
        _CHECKERS[spec] = checker
        while len(_CHECKERS) > POOL_CACHE_SIZE:
            _CHECKERS.popitem(last=False)
    return checker


def close_shared_pools():
    for pool in list(_POOLS.values()):
        try:
            pool.close()
        except Exception:
            pass
    _POOLS.clear()
    _CHECKERS.clear()


atexit.register(close_shared_pools)


class _TimedPool:
    """A shared pool, with the seconds spent inside it accumulated per solve. `close()` is a
    no-op: the pool belongs to the module cache, not to any BatchedProgram.

    `seconds` is HOST time blocked in pool calls (an `eval`, or a `submit` plus its
    `collect`); `span_seconds` is the pool's latency, submit to collect. On the split step
    the two differ by exactly the IPC and Drake time hidden behind GPU work (`seconds <=
    span_seconds`); on a plain `eval` they are equal. `configs` counts configurations."""

    def __init__(self, pool):
        self.pool = pool
        self.reset()

    def reset(self):
        self.seconds = 0.0
        self.span_seconds = 0.0
        self.calls = 0
        self.configs = 0
        self._t_submit = None

    def eval(self, Q, need_grad=True):
        t = time.perf_counter()
        out = self.pool.eval(Q, need_grad=need_grad)
        dt = time.perf_counter() - t
        self.seconds += dt
        self.span_seconds += dt
        self.calls += 1
        self.configs += int(np.shape(Q)[0])
        return out

    def submit(self, Q, need_grad=True):
        t = time.perf_counter()
        ticket = self.pool.submit(Q, need_grad=need_grad)
        self._t_submit = t
        self.seconds += time.perf_counter() - t
        self.configs += int(np.shape(Q)[0])
        return ticket

    def collect(self, ticket):
        t = time.perf_counter()
        out = self.pool.collect(ticket)
        t1 = time.perf_counter()
        self.seconds += t1 - t
        if self._t_submit is not None:
            self.span_seconds += t1 - self._t_submit
        self._t_submit = None
        self.calls += 1
        return out

    def drain(self):
        self.pool.drain()

    @property
    def closed(self):
        return self.pool.closed

    def alive(self):
        return self.pool.alive()

    def close(self):
        pass


def _scene_spec(program):
    """`SceneSpec.from_program`, falling back to the bare scene (no welded mug) for a
    program whose plant carries no target mug -- the benchmark's SAMPLER program on the grasp
    task, which `warm_up` is called on."""
    try:
        return SceneSpec.from_program(program)
    except Exception:
        yaml = getattr(program, "scene_yaml", None) or getattr(program.diagram, "scene_yaml", None)
        if yaml is None:
            raise
        opts = program.options
        return SceneSpec(yaml_path=yaml, collision_bound=float(opts.collision_bound),
                         influence_distance_offset=float(opts.collision_influence_offset),
                         row_scale=float(opts.collision_row_scale))


## ------------------------------------------------------------------------------------ ##
##                      the scaled target: rows, costs, Jacobians                        ##
## ------------------------------------------------------------------------------------ ##

def region_scale(bp):
    """`s [nvars]`, the normalising half-width of each decision variable's region: on the
    learned arm the conditioning position's box (`c_position_slack`), pi on its orientation,
    the latent trust radius (the +-5 box where the program sets none) and the correction box
    (`correction_bound`); on joint space half the range of each coordinate's true bound (the
    joint limits). Read from the program's own options and bounds, never assumed."""
    p, opts = bp.program, bp.options
    if bp.is_learned:
        radius = p.LatentTrustRadius() if hasattr(p, "LatentTrustRadius") else None
        radius = LATENT_BOX if radius is None else float(radius)
        s = ([float(opts.c_position_slack)] * 3 + [math.pi] * 3 + [radius] * bp.width
             + [float(opts.correction_bound)] * bp.ndof)
        s = np.asarray(s, dtype=float)
    else:
        lo, hi = (b.detach().cpu().double().numpy() for b in bp.bounds)
        s = 0.5 * (hi - lo)
    if s.shape != (bp.nvars,) or not np.all(np.isfinite(s)) or not np.all(s > 0):
        raise ValueError(f"svgd: no finite positive region half-width for every variable of "
                         f"{type(p).__name__}: {s}")
    return s


class _Target:
    """A `BatchedProgram` plus the row scaling, the region scale `s` and the bookkeeping that
    assembles per-row Jacobians from the `RowSpec`s."""

    def __init__(self, bp, tol):
        self.bp = bp
        self.dtype, self.device = bp.dtype, bp.device
        self.n = bp.nvars
        self.ndof = bp.ndof
        kw = dict(dtype=self.dtype, device=self.device)
        self.sh = torch.full((len(bp.h_spec),), 1.0 / float(tol), **kw)
        self.sg = torch.full((len(bp.g_spec),), 1.0 / float(tol), **kw)
        self.m_e, self.m_i = len(bp.h_spec), len(bp.g_spec)
        self.s = torch.tensor(region_scale(bp), **kw)

        ## Row index bookkeeping: each h/g entry maps to a row of the stacked
        ## `[drake_rows | extra rows...]` Jacobian, with a sign (eq/hi: +1, lo: -1).
        self._extra_offsets = {}
        off = 0
        for key, size in bp.extra_blocks:
            self._extra_offsets[key] = off
            off += int(size)
        self._n_extra = off

        def locate(spec):
            if spec.drake_binding == GENERIC_BINDING:
                return int(spec.drake_row)
            key = EXTRA_ROW_KEYS.get(spec.drake_binding, spec.drake_binding)
            return bp.n_rows + self._extra_offsets[key] + int(spec.drake_row)
        sign = {"eq": 1.0, "hi": 1.0, "lo": -1.0}
        li = lambda v: torch.tensor(v, dtype=torch.long, device=self.device)
        self.h_idx = li([locate(r) for r in bp.h_spec])
        self.g_idx = li([locate(r) for r in bp.g_spec])
        self.h_coef = (torch.tensor([sign[r.kind] for r in bp.h_spec], **kw) * self.sh)
        self.g_coef = (torch.tensor([sign[r.kind] for r in bp.g_spec], **kw) * self.sg)
        ## Row groups of `h`, for the trace: which equality rows are position / rotation.
        self.h_is_rot = torch.tensor([r.group == "rotation" for r in bp.h_spec],
                                     dtype=torch.bool, device=self.device)
        self._n_rot = sum(1 for r in bp.h_spec if r.group == "rotation")

        ## Routing of the GENERIC rows' derivative w.r.t. the configuration (`generic_blocks`),
        ## for the autograd path: task rows by a vmapped backward through the frame chain,
        ## the collision row from the pool's stored gradient, the joint-limit rows the identity.
        task, jl, col = [], [], []
        for kind, start, size in bp.generic_blocks:
            rows = list(range(start, start + size))
            if kind in ("pose_pos", "pose_rpy", "mug"):
                task += rows
            elif kind == "joint_limit":
                jl += rows
            elif kind == "collision":
                col += rows
            else:                                   # pragma: no cover
                raise RuntimeError(f"unknown generic row kind {kind!r}")
        self._task_rows = li(task)
        self._jl_rows = li(jl)
        self._col_rows = li(col)
        self._E_task = torch.eye(len(task), **kw).unsqueeze(1)           # [nt, 1, nt]
        ## Derivative mode (see `evaluate`): closed-form rows where the batched program
        ## offers them, autograd through the kinematics otherwise.
        self.analytic = bool(bp.has_analytic_row_jacobians)

    ## -- batched autograd, with the vmap-fallback chatter silenced ----------------------
    @staticmethod
    def _batched_grad(outputs, inputs, E, N):
        """`autograd.grad` with `is_grads_batched` over the rows of `E [m, 1, m]`, returning
        `[N, m, ...]`; torch's per-call "no batching rule" warning is filtered here only."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*batching rule.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch._functorch")
            (J,) = torch.autograd.grad(outputs, inputs, grad_outputs=E.expand(-1, N, -1),
                                       is_grads_batched=True, retain_graph=True)
        return J.transpose(0, 1)

    def _extra_jacobian(self, X, N):
        if self._n_extra == 0:
            return torch.zeros(N, 0, self.n, dtype=self.dtype, device=self.device)
        return self.bp.extra_row_jacobian(X)                     # closed form, no autograd

    ## -- evaluation -------------------------------------------------------------------
    def evaluate(self, X, need_grad=True):
        """Rows and costs at `X [N, n]`; the scaled `h`, `g`, the violation and a per-particle
        finiteness flag. With `need_grad`, ANALYTIC mode keeps a graph through the flow only
        (none at all on joint space); AUTOGRAD mode keeps the full graph."""
        need_grad = bool(need_grad)
        graph = need_grad and (self.bp.is_learned or not self.analytic)
        Xg = X.detach().requires_grad_(graph)
        with torch.set_grad_enabled(graph):
            out = self.bp.evaluate(Xg, detach_kinematics=self.analytic and graph,
                                   row_jacobians=self.analytic and need_grad)
        h, g, finite, infeas = self.scale(out)
        return SimpleNamespace(Xg=Xg, out=out, h=h, g=g, F=out.F, cfg=out.q, finite=finite,
                               infeas=infeas)

    def scale(self, out):
        """`(h~, g~, finite [N], v [N])` from a batched `Evaluation`: the rows divided by `tol`,
        the per-particle finiteness flag and the violation `||[h~; max(g~, 0)]||_inf` (+inf
        where not finite)."""
        h = out.h * self.sh
        g = out.g * self.sg
        finite = (torch.isfinite(h).all(dim=1) & torch.isfinite(g).all(dim=1)
                  & torch.isfinite(out.F) & torch.isfinite(out.q).all(dim=1))
        v = al.violation(h, g)
        v = torch.where(finite, v, torch.full_like(v, float("inf"))).detach()
        return h, g, finite, v

    def unscale(self, h, g):
        """The inverse of `scale`'s row scaling: the `Evaluation`'s own `h` / `g` (Drake's
        signed violations)."""
        return h / self.sh, g / self.sg

    ## -- derivatives ------------------------------------------------------------------
    def pullback(self, ev, cots):
        """`[J_q^T v for v in cots]`, each `v [N, ndof]` -> `[N, n]`: ONE vmapped backward
        through the flow with all of them stacked; the identity on joint space."""
        if not self.bp.is_learned:
            return [v.detach().clone() for v in cots]
        E = torch.stack([v.detach() for v in cots])                   # [k, N, ndof]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*batching rule.*")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch._functorch")
            (G,) = torch.autograd.grad(ev.cfg, ev.Xg, grad_outputs=E, is_grads_batched=True,
                                       retain_graph=True)
        self._count("map_jacobian")
        return [G[i] for i in range(len(cots))]

    def cost_parts(self, ev):
        """`(dF/dcfg [N, ndof], dF/dx|direct [N, n])`: closed form in analytic mode; in
        autograd mode the whole gradient lands in the direct part."""
        if self.analytic:
            return self.bp.cost_gradient_parts(ev.Xg, ev.cfg)
        (g,) = torch.autograd.grad(ev.F.sum(), ev.Xg, retain_graph=True)
        return torch.zeros_like(ev.cfg), g

    def row_cotangents(self, ev, ch, cg):
        """The cotangent on the stacked UNSCALED rows `[drake_rows | extra rows]` that the
        scaled `h`, `g` cotangents induce: `[N, nr + n_extra]`."""
        N = ev.Xg.shape[0]
        c = torch.zeros(N, self.bp.n_rows + self._n_extra, dtype=self.dtype, device=self.device)
        c.index_add_(1, self.h_idx, ch * self.h_coef)
        c.index_add_(1, self.g_idx, cg * self.g_coef)
        return c

    def al_gradients(self, ev, ch, cg, R):
        """`(dF/dx, J_h~^T ch + J_g~^T cg, J_q^T R)` with ONE backward through the flow (`R`
        may be None): the autograd path's three gradients."""
        nr = self.bp.n_rows
        dFc, dFx = self.cost_parts(ev)
        if self.analytic:
            N = ev.Xg.shape[0]
            c = self.row_cotangents(ev, ch, cg)
            Dq = self.generic_jacobian_cfg(ev)                            # [N, nr, ndof]
            v_cfg = (Dq.transpose(1, 2) @ c[:, :nr].unsqueeze(2)).squeeze(2)
            J_ex = self._extra_jacobian(ev.Xg, N)
            v_x = (J_ex.transpose(1, 2) @ c[:, nr:].unsqueeze(2)).squeeze(2) if self._n_extra else 0.0
            G = self.pullback(ev, [dFc, v_cfg] + ([R] if R is not None else []))
            return G[0] + dFx, G[1] + v_x, (G[2] if R is not None else None)
        s = (ev.h * ch).sum() + (ev.g * cg).sum()
        (gC,) = torch.autograd.grad(s, ev.Xg, retain_graph=True)
        R_pulled = self.pullback(ev, [R])[0] if R is not None else None
        return dFx, gC, R_pulled

    def jacobian_q(self, ev):
        """`dq/dx [N, ndof, n]` from the live graph (identity on joint space)."""
        bp = self.bp
        N = ev.Xg.shape[0]
        if not bp.is_learned:
            return bp.jacobian_q(ev.Xg.detach())
        E = torch.eye(self.ndof, dtype=self.dtype, device=self.device).unsqueeze(1)
        J = self._batched_grad(ev.cfg, ev.Xg, E, N)
        self._count("map_jacobian")
        return J

    def generic_jacobian_cfg(self, ev):
        """`d drake_rows / d cfg`, `[N, nr, ndof]`. Analytic mode: the batched program's
        closed form. Autograd mode: routed by row kind -- the task rows by one vmapped
        backward through the kinematics, the collision row from the pool's stored
        `d row / d q_plant` through `config_to_plant_q`, the joint-limit rows the identity."""
        out = ev.out
        if self.analytic:
            return torch.nan_to_num(self.bp.generic_rows_jacobian_cfg(out), nan=0.0,
                                    posinf=0.0, neginf=0.0)
        D = out.drake_rows
        N, nr = D.shape
        Dq = torch.zeros(N, nr, self.ndof, dtype=D.dtype, device=D.device)
        if self._task_rows.numel():
            Dq[:, self._task_rows] = self._batched_grad(D[:, self._task_rows], ev.cfg, self._E_task, N)
        if self._jl_rows.numel():
            Dq[:, self._jl_rows] = torch.eye(self.ndof, dtype=D.dtype, device=D.device)[
                :self._jl_rows.numel()].unsqueeze(0).expand(N, -1, -1)
        if self._col_rows.numel():
            cg = out.extras.get("collision_grad")
            if cg is None:                          # evaluated without a graph: no row grad
                cg = torch.zeros(N, out.q_plant.shape[1], dtype=D.dtype, device=D.device)
            if self.bp.plant_q_is_padded_cfg:
                gcol = cg[:, :self.ndof]
            else:
                (gcol,) = torch.autograd.grad((out.q_plant * cg).sum(), ev.cfg, retain_graph=True)
            Dq[:, self._col_rows[0]] = gcol
        return torch.nan_to_num(Dq, nan=0.0, posinf=0.0, neginf=0.0)

    def constraint_gradient(self, out, X, J_q, ch, cg):
        """`grad_X (L - f) = J_h~^T ch + J_g~^T cg` from a bare `Evaluation` (analytic mode
        only): what the split step's stage 3 calls, having no `ev` namespace and no graph."""
        Dq = torch.nan_to_num(self.bp.generic_rows_jacobian_cfg(out), nan=0.0, posinf=0.0,
                              neginf=0.0)
        J_stack = torch.cat([Dq @ J_q, self._extra_jacobian(X, X.shape[0])], dim=1)
        J_h = J_stack[:, self.h_idx] * self.h_coef.view(1, -1, 1)
        J_g = J_stack[:, self.g_idx] * self.g_coef.view(1, -1, 1)
        return ((J_h.transpose(1, 2) @ ch.unsqueeze(2)).squeeze(2)
                + (J_g.transpose(1, 2) @ cg.unsqueeze(2)).squeeze(2))

    ## -- the per-outer-check trace -------------------------------------------------------
    TRACE_COLUMNS = ("min_infeas", "med_infeas", "n_feasible", "med_lam_inf", "max_lam_inf",
                     "med_mu_inf", "max_mu_inf", "min_F_feasible", "med_h_pos", "med_h_rot",
                     "med_g_plus", "n_multiplier_clipped", "n_resampled", "bound_clip")

    def trace_row(self, ev, S, n_clipped, n_resampled, bound_clip):
        """One `[len(TRACE_COLUMNS)]` tensor of swarm statistics at a check, on the device
        (stacked and moved to the host ONCE at the end): the best and median scaled
        violation, the feasible count, the median and largest `|lam_i|_inf` and `|mu_i|_inf`
        after the dual step, the best objective among feasible particles, the median over
        particles of the worst position / rotation equality and of the worst violated
        inequality, how many multiplier entries the dual step clipped, the particles
        redrawn, and the clamp distance onto the bounds since the last check."""
        kw = dict(dtype=self.dtype, device=self.device)
        nan = torch.full((), float("nan"), **kw)
        v = torch.nan_to_num(ev.infeas, nan=float("inf"))
        feas = v <= 1.0
        habs = torch.nan_to_num(ev.h.detach().abs(), nan=float("inf"))
        gpos = torch.nan_to_num(torch.clamp(ev.g.detach(), min=0.0), nan=float("inf"))

        def med_group(mask, count):
            return nan if count == 0 else habs[:, mask].amax(dim=1).median()
        F = torch.nan_to_num(ev.F.detach(), nan=float("inf"))
        minF = torch.where(feas, F, torch.full_like(F, float("inf"))).min()
        minF = torch.where(torch.isfinite(minF), minF, nan)
        ## A particle with a non-finite row has NaN multipliers until it is redrawn at this
        ## same check: the multiplier columns ignore it.
        lam = S.lam.abs().amax(dim=1) if S.lam.shape[1] else torch.zeros_like(v)
        mu = S.mu.abs().amax(dim=1) if S.mu.shape[1] else torch.zeros_like(v)
        nanmax = lambda t: torch.nan_to_num(t, nan=float("-inf")).max()
        return torch.stack([
            v.min(), v.median(), feas.sum().to(self.dtype), torch.nanmedian(lam), nanmax(lam),
            torch.nanmedian(mu), nanmax(mu), minF, med_group(~self.h_is_rot, self.m_e - self._n_rot),
            med_group(self.h_is_rot, self._n_rot),
            gpos.amax(dim=1).median() if gpos.shape[1] else nan,
            n_clipped.to(self.dtype), torch.full((), float(n_resampled), **kw),
            bound_clip.to(self.dtype)])

    def _count(self, bucket):
        counts = getattr(self.bp.program, "eval_counts", None)
        if counts is not None:
            counts[bucket] = counts.get(bucket, 0) + 1

    def project(self, X):
        return self.bp.project(X)


## ------------------------------------------------------------------------------------ ##
##                                      the solver                                       ##
## ------------------------------------------------------------------------------------ ##

def _median_pairwise_distance(Q):
    """The median over pairs of `||q_a - q_b||` for `Q [k, ndof]` (numpy), None for k < 2."""
    k = Q.shape[0]
    if k < 2:
        return None
    d = np.sqrt(np.maximum(((Q[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2), 0.0))
    return float(np.median(d[np.triu_indices(k, 1)]))


class SvgdSolver:
    """See the module docstring.

    `particles_override` (tests only): a `[N, n]` array used as the initial swarm instead of
    the drawn one -- the way to hand the solver a non-finite particle and watch it resample.
    """

    def __init__(self, program, particles_override=None):
        self.program = program
        self.options = program.options
        opts = self.options
        self.method = opts.svgd_method
        self._sc = fused.StepConfig.from_options(opts)
        self._overlap = bool(opts.svgd_pool_overlap)
        self._runner = None
        self.N = int(opts.svgd_n)
        self.dtype = torch.float64 if opts.svgd_dtype == "float64" else torch.float32
        self.workers = resolve_workers(opts.svgd_collision_workers)
        self.tol = float(opts.acceptable_constr_viol_tol)
        self._override = None if particles_override is None else np.asarray(particles_override, dtype=float)
        self._tg = None
        self._pool = None
        self._rho = float(opts.svgd_rho)

    ## ----------------------------------- setup ----------------------------------------
    def _build(self):
        if self._tg is not None:
            return
        program = self.program
        spec = _scene_spec(program)
        self._spec = spec
        self._pool = _TimedPool(shared_pool(spec, self.workers))
        bp = BatchedProgram.from_program(program, dtype=self.dtype, pool=self._pool)
        self._tg = _Target(bp, self.tol)
        self.device = bp.device
        self.n = bp.nvars

    def _seed(self, x0):
        crc = zlib.crc32(np.ascontiguousarray(x0, dtype=np.float64).tobytes())
        return int((int(self.options.svgd_seed) * 1000003 + crc) % (2 ** 31 - 1))

    def _generator(self, seed):
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return g

    def _draw_native(self, N, gen):
        """N particles from the arm's NATIVE start distribution (unprojected); `(X, stats)`."""
        bp = self._tg.bp
        stats = {}
        if bp.is_learned:
            z = torch.randn(N, bp.width, generator=gen, dtype=self.dtype, device=self.device)
            c = bp.native_c().unsqueeze(0).expand(N, 6)
            qc = torch.zeros(N, bp.ndof, dtype=self.dtype, device=self.device)
            return torch.cat([c, z, qc], dim=1), stats
        lower, upper = bp.program.ConfigLimits()
        lo = torch.tensor(np.asarray(lower, dtype=float)[:bp.ndof], dtype=self.dtype, device=self.device)
        hi = torch.tensor(np.asarray(upper, dtype=float)[:bp.ndof], dtype=self.dtype, device=self.device)
        checker = shared_checker(self._spec)
        out = torch.empty(0, self.n, dtype=self.dtype, device=self.device)
        rounds = 0
        while out.shape[0] < N and rounds < NATIVE_DRAW_ROUNDS:
            u = torch.rand(N, self.n, generator=gen, dtype=self.dtype, device=self.device)
            q = lo + u * (hi - lo)
            with torch.no_grad():
                qp = bp.config_to_plant_q(q).to(device="cpu", dtype=torch.float64).numpy()
            ok = torch.as_tensor(checker.collision_free(qp), device=self.device)
            out = torch.cat([out, q[ok]], dim=0)
            rounds += 1
        if out.shape[0] < N:
            ## Documented fallback: fill with uniform draws regardless of collision.
            u = torch.rand(N - out.shape[0], self.n, generator=gen, dtype=self.dtype, device=self.device)
            stats["native_fallback"] = int(N - out.shape[0])
            out = torch.cat([out, lo + u * (hi - lo)], dim=0)
        stats["native_rounds"] = rounds
        return out[:N], stats

    def _draw(self, N, x0, gen):
        """N fresh particles from the init distribution (unprojected); `(X, stats)`."""
        if self.options.svgd_paired_init == "jitter":
            xi = torch.randn(N, self.n, generator=gen, dtype=self.dtype, device=self.device)
            return x0.unsqueeze(0) + float(self.options.svgd_jitter) * xi * self._tg.s.unsqueeze(0), {}
        return self._draw_native(N, gen)

    def _init_particles(self, x0_np):
        x0 = torch.tensor(x0_np, dtype=self.dtype, device=self.device)
        gen = self._generator(self._seed(x0_np))
        self._gen = gen
        stats = {}
        if self._override is not None:
            X = torch.tensor(self._override, dtype=self.dtype, device=self.device)
            if X.shape != (self.N, self.n):
                raise ValueError(f"particles_override must be [{self.N}, {self.n}]")
            return X, torch.zeros(self.N, dtype=self.dtype, device=self.device), stats
        X, stats = self._draw(self.N, x0, gen)
        Xp, clip = self._tg.project(X)
        keep0 = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        keep0[0] = True
        X = torch.where(keep0.unsqueeze(1), x0.unsqueeze(0).expand_as(Xp), Xp)
        clip = torch.where(keep0, torch.zeros_like(clip), clip)
        return X, clip, stats

    def _init_state(self, X):
        """The AL state at the start: zero multipliers on every particle."""
        tg = self._tg
        return ALState.init(X.shape[0], tg.m_e, tg.m_i, self.dtype, self.device)

    ## ------------------------------- the step -----------------------------------------
    def _step_mode(self):
        """`eager` | `compiled` | `graphed` from `svgd_compile` / `svgd_cuda_graph` (graphs
        need CUDA; on a CPU device the request degrades to `compiled`)."""
        opts = self.options
        if not opts.svgd_compile:
            return "eager"
        if opts.svgd_cuda_graph and self.device.type == "cuda":
            return "graphed"
        return "compiled"

    def _make_runner(self):
        """The split step's `fused.StepRunner` on a target with closed-form row Jacobians;
        None for a robot behind its own pose provider (`_step_eager`, autograd)."""
        self._runner = None
        if self._tg.analytic:
            self._runner = fused.StepRunner(self._tg, self._sc, self.N, self._step_mode())
        return self._runner

    def _step_split(self, X, S):
        """One step as `fused`'s three stages with the collision pool between them; returns
        `(X_new, clip [N], n_clip [N])`."""
        r = self._runner
        bp = self._tg.bp
        cfg, q_plant = r.s1(X)
        self._tg._count("map_forward")
        ticket, col = None, (None, None)
        if bp._has_collision:
            qp = q_plant.to("cpu", torch.float64).numpy()
            ticket = self._pool.submit(qp, need_grad=True)
            if not self._overlap:
                col, ticket = self._pool.collect(ticket), None
        try:
            J_q, kin = r.s2(X, cfg)
        finally:
            if ticket is not None:
                col = self._pool.collect(ticket)
        if bp.is_learned:
            self._tg._count("map_jacobian")
        kw = dict(device=self.device, dtype=self.dtype)
        col_row = None if col[0] is None else torch.as_tensor(col[0]).to(**kw)
        col_grad = None if col[1] is None else torch.as_tensor(col[1]).to(**kw)
        o = r.s3(X, cfg, J_q, kin, col_row, col_grad, fused.state_dict(S))
        return o["X"], o["clip"], o["n_clip"]

    def _step_eager(self, X, S):
        """The same step through autograd (a robot behind its own pose provider)."""
        tg = self._tg
        ev = tg.evaluate(X)
        ch, cg = al.al_constraint_grad_coefficients(ev.h, ev.g, S, self._rho)
        K, R, kernel_on = fused.kernel_terms(self._sc, ev.cfg, ev.finite, X)
        gF, gC, R_x = tg.al_gradients(ev, ch, cg, R if kernel_on else None)
        if R_x is None:
            R_x = torch.zeros_like(X)
        return fused.update_step(tg, self._sc, X, gF.detach(), gC.detach(), K, R_x.detach(),
                                 ev.finite)

    ## ---------------------------------- swarm -----------------------------------------
    def _swarm(self, X, S, deadline, outer_iters, inner_iters, record):
        """The outer/inner loop; returns `(X, S, stop_status, stats)`."""
        opts = self.options
        tg = self._tg
        N = X.shape[0]
        step_fn = self._step_split if self._runner is not None else self._step_eager
        stats = dict(outer=0, inner=0, n_resampled=0, n_dual_updates=0,
                     n_multiplier_clipped=0, bound_clip=0.0, bound_clip_count=0,
                     best_history=[], trace=[], stop_reason="step_cap")
        clip_tot = torch.zeros((), dtype=self.dtype, device=self.device)
        nclip_tot = torch.zeros((), dtype=torch.long, device=self.device)
        best_seen = float("inf")
        stale = 0
        status = STATUS_STEP_CAP
        for t_outer in range(outer_iters):
            clip_check = torch.zeros((), dtype=self.dtype, device=self.device)
            for _ in range(inner_iters):
                X, clip, n_clip = step_fn(X, S)
                clip_check = clip_check + torch.nan_to_num(clip, nan=0.0).sum()
                nclip_tot = nclip_tot + n_clip.sum()
                stats["inner"] += 1
            clip_tot = clip_tot + clip_check
            ## -- the check, at the current swarm: the dual step, unconditionally --
            with torch.no_grad():
                ev = tg.evaluate(X, need_grad=False)
            S, n_clipped = al.dual_update(ev.h, ev.g, S, self._rho, float(opts.svgd_multiplier_max))
            stats["n_dual_updates"] += 1
            mask = al.resample_mask(ev.cfg, ev.finite, float(opts.svgd_resample_q_max))
            n_re = int(mask.sum().item())
            v = torch.nan_to_num(ev.infeas, nan=float("inf"))
            feas = (v <= 1.0) & ~mask
            F = torch.where(feas, torch.nan_to_num(ev.F, nan=float("inf")),
                            torch.full_like(ev.F, float("inf")))
            i_best = int(torch.argmin(F).item()) if bool(feas.any()) else int(torch.argmin(v).item())
            record(X[i_best])
            stats["trace"].append(tg.trace_row(ev, S, n_clipped, n_re, clip_check))
            stats["n_multiplier_clipped"] += int(n_clipped.item())
            if n_re:
                fresh, _ = self._draw_native(N, self._gen)
                fresh, _ = tg.project(fresh)
                X = torch.where(mask.unsqueeze(1), fresh, X)
                S = al.reset(S, mask)
                stats["n_resampled"] += n_re
            stats["outer"] = t_outer + 1
            ## -- the stop rule: the best feasible particle's objective --
            f_best = float(F.min().item())
            stats["best_history"].append(f_best)
            if math.isfinite(f_best):
                if not math.isfinite(best_seen) or f_best < best_seen - float(opts.svgd_stop_rel) * abs(best_seen):
                    best_seen = f_best
                    stale = 0
                else:
                    stale += 1
                if stale >= int(opts.svgd_stop_patience):
                    status = STATUS_CONVERGED
                    stats["stop_reason"] = "converged"
                    break
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if time.perf_counter() >= deadline:
                status = STATUS_WALL_CLOCK
                stats["stop_reason"] = "wall_clock"
                break
        stats["bound_clip"] = float(clip_tot.item())
        stats["bound_clip_count"] = int(nclip_tot.item())
        return X, S, status, stats

    ## -------------------------------- Drake re-check ----------------------------------
    def _drake_max_violation(self, x_full):
        """`max` over every binding of the worst signed violation (mirrors
        `benchmark.binding_worst`, kept local so the solver does not import the harness)."""
        prog = self.program.prog
        worst = -np.inf
        for binding in prog.GetAllConstraints():
            ev = binding.evaluator()
            value = np.asarray(prog.EvalBinding(binding, x_full), dtype=float).flatten()
            lb = np.asarray(ev.lower_bound(), dtype=float).flatten()
            ub = np.asarray(ev.upper_bound(), dtype=float).flatten()
            below = np.where(np.isfinite(lb), lb - value, -np.inf)
            above = np.where(np.isfinite(ub), value - ub, -np.inf)
            w = float(np.max(np.maximum(below, above))) if value.size else -np.inf
            if not np.isfinite(w) and np.isnan(w):
                w = np.inf
            worst = max(worst, w)
        return worst

    ## ----------------------------------- solve ----------------------------------------
    def solve(self):
        program = self.program
        opts = self.options
        t0 = time.perf_counter()
        phase = {}
        wall = float(opts.max_wall_time)
        swarm_deadline = t0 + wall * (1.0 - float(opts.svgd_time_reserve))

        self._build()
        tg = self._tg
        bp = tg.bp
        self._pool.drain()
        self._pool.reset()
        self._make_runner()
        counts0 = dict(getattr(program, "eval_counts", {}) or {})

        x0_np = np.asarray(program.prog.GetInitialGuess(program.lumped_vars), dtype=float)
        X, clip, init_stats = self._init_particles(x0_np)
        N = X.shape[0]
        extras = dict(init_protocol=opts.svgd_paired_init,
                      clip_distance=clip.detach().cpu().numpy().tolist(),
                      init_stats=init_stats, workers=self.workers,
                      region_scale=tg.s.detach().cpu().double().numpy().tolist(),
                      step_mode=(self._runner.mode if self._runner is not None else "eager (autograd path)"),
                      step_reused_template=bool(self._runner is not None and self._runner.reused),
                      pool_overlap=self._overlap)

        ## -- optional CEM warm-up --
        warmup_steps = 0
        if opts.svgd_warmup == "cem" and N > 1:
            from src.svgd.warmup import cem_warmup
            t = time.perf_counter()
            X, wstats = cem_warmup(tg, X, self._rho, int(opts.svgd_warmup_iters),
                                   float(opts.svgd_warmup_elite), self._gen,
                                   deadline=swarm_deadline)
            warmup_steps = wstats.get("iters", 0)
            extras["warmup"] = wstats
            phase["warmup"] = time.perf_counter() - t
        S = self._init_state(X)
        phase["init"] = time.perf_counter() - t0 - phase.get("warmup", 0.0)

        def record(x_best):
            program.RecordIterate(x_best.detach().to(device="cpu", dtype=torch.float64).numpy())

        ## -- swarm --
        t = time.perf_counter()
        outer_iters = int(opts.max_iter) if opts.max_iter is not None else int(opts.svgd_outer_iters)
        inner_iters = max(1, int(opts.svgd_inner_iters))
        X, S, stop, sstats = self._swarm(X, S, swarm_deadline, outer_iters, inner_iters, record)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        phase["swarm"] = time.perf_counter() - t

        ## -- select: feasible on the batched rows, by objective; exact Drake re-check --
        t = time.perf_counter()
        with torch.no_grad():
            evf = tg.evaluate(X, need_grad=False)
        v = torch.nan_to_num(evf.infeas, nan=float("inf")).cpu().double().numpy()
        Fv = torch.nan_to_num(evf.F, nan=float("inf")).cpu().double().numpy()
        feas = v <= 1.0
        n_feasible = int(feas.sum())
        cfg_np = evf.cfg.detach().cpu().double().numpy()
        spread = _median_pairwise_distance(cfg_np[feas]) if n_feasible >= 2 else None
        ## The swarm's clamp onto B ran in the swarm's dtype, whose rounding of a bound can sit
        ## an ulp outside the float64 bound Drake holds; the same clamp in float64 removes that
        ## (it moves no coordinate by more than the swarm dtype's rounding, recorded).
        X32 = X.detach().cpu().double().numpy()
        lo64, hi64 = bp._variable_bounds()
        X_np = np.minimum(np.maximum(X32, lo64), hi64)
        extras["bound_rounding"] = float(np.nan_to_num(np.abs(X_np - X32), nan=0.0).max(initial=0.0))
        k = max(1, int(opts.svgd_recheck_topk))
        if not np.isfinite(v).any():
            x_ret, status, drake_ok, solver_ok, selected, drake_viol, cands = (
                x0_np.copy(), STATUS_NAN, False, False, -1, None, [])
        else:
            if n_feasible:
                order = [int(i) for i in np.argsort(np.where(feas, Fv, np.inf), kind="stable")[:min(k, n_feasible)]]
            else:
                order = [int(i) for i in np.argsort(v, kind="stable")[:k]]
            cands = order
            selected, drake_viol, drake_ok = None, np.inf, False
            for j in order:
                vj = self._drake_max_violation(bp.to_drake_x(X_np[j]))
                if n_feasible and vj <= self.tol:
                    selected, drake_viol, drake_ok = j, vj, True
                    break
                if vj < drake_viol:
                    selected, drake_viol = j, vj
            if not drake_ok:
                drake_ok = bool(drake_viol <= self.tol)
            x_ret = X_np[selected].copy()
            solver_ok = bool(feas[selected])
            status = STATUS_CONVERGED if drake_ok else (
                stop if stop in (STATUS_WALL_CLOCK, STATUS_STEP_CAP) else STATUS_INFEASIBLE)
        phase["select"] = time.perf_counter() - t

        counts = dict(getattr(program, "eval_counts", {}) or {})
        map_evals = int(counts.get("map_forward", 0) - counts0.get("map_forward", 0))
        total = time.perf_counter() - t0
        rows = sstats.get("trace", [])
        if rows:
            T_np = torch.stack(rows).detach().to(device="cpu", dtype=torch.float64).numpy()
            trace = {name: [float(x) for x in T_np[:, j]] for j, name in enumerate(_Target.TRACE_COLUMNS)}
        else:
            trace = {name: [] for name in _Target.TRACE_COLUMNS}
        def inf_norms(M):
            if M.shape[1] == 0:
                return np.zeros(M.shape[0])
            return M.detach().abs().amax(dim=1).cpu().double().numpy()
        lam_inf, mu_inf = inf_norms(S.lam), inf_norms(S.mu)
        extras.update(dict(
            trace=trace, derivative_mode="analytic" if tg.analytic else "autograd",
            violation_at_stop=[float(x) for x in v],
            drake_max_violation=None if drake_viol is None else float(drake_viol),
            best_history=sstats.get("best_history", []), stop_status=status_name(stop),
            total_steps=int(sstats.get("inner", 0)), eval_counts_delta={
                key: int(counts.get(key, 0) - counts0.get(key, 0)) for key in counts},
            pool_calls=int(self._pool.calls), pool_configs=int(self._pool.configs),
            pool_span_seconds=float(self._pool.span_seconds),
            candidates=cands,
            bound_clip_count=int(sstats.get("bound_clip_count", 0)),
            warmup_steps=int(warmup_steps)))
        details = SvgdSolverDetails(
            status=int(status), status_name=status_name(status), method=self.method,
            n_particles=N, dtype=opts.svgd_dtype,
            iterations=int(sstats.get("outer", 0)), inner_steps=int(sstats.get("inner", 0)),
            map_evals=map_evals, n_feasible=n_feasible, n_resampled=int(sstats.get("n_resampled", 0)),
            selected_index=int(selected), phase_times=phase,
            timed_out=(stop == STATUS_WALL_CLOCK), hit_iteration_cap=(stop == STATUS_STEP_CAP),
            solver_feasible=bool(solver_ok), drake_feasible=bool(drake_ok),
            solve_seconds=total, collision_seconds=float(self._pool.seconds),
            stop_reason=str(sstats.get("stop_reason", "")),
            n_dual_updates=int(sstats.get("n_dual_updates", 0)),
            lam_inf_median=float(np.median(lam_inf)), lam_inf_max=float(np.max(lam_inf)),
            mu_inf_median=float(np.median(mu_inf)), mu_inf_max=float(np.max(mu_inf)),
            bound_clip=float(sstats.get("bound_clip", 0.0)),
            n_multiplier_clipped=int(sstats.get("n_multiplier_clipped", 0)),
            feasible_q_spread=spread, extras=extras)
        write_log(opts.file_print_name, details)
        return SvgdResult(program, bp.to_drake_x(x_ret), details, success=bool(drake_ok))

    ## ---------------------------------- prepare ---------------------------------------
    def prepare(self):
        """Infrastructure only, for a harness to call BEFORE it starts a cell's clock: build
        the batched program and make sure this scene's collision pool exists with every
        worker's scene built (one configuration per worker). A grasp grid welds a new mug
        per target, so without this the first arm solved on each target paid the pool spawn
        and the workers' scene builds (seconds) inside its own wall-clock cap, and the
        second arm did not. Returns the seconds spent."""
        t0 = time.perf_counter()
        self._build()
        bp = self._tg.bp
        if bp._has_collision:
            x0 = np.nan_to_num(np.asarray(self.program.prog.GetInitialGuess(self.program.lumped_vars),
                                          dtype=float))
            with torch.no_grad():
                X = torch.tensor(x0, dtype=self.dtype, device=self.device).unsqueeze(0)
                qp = bp.config_to_plant_q(bp._config(X)).to("cpu", torch.float64).numpy()
            self._pool.drain()
            self._pool.eval(np.repeat(np.nan_to_num(qp), self.workers, axis=0), need_grad=True)
        return time.perf_counter() - t0

    ## ---------------------------------- warm-up ---------------------------------------
    def warm_up(self):
        """Pay the pool spawn, the first flow / FK calls and -- under `svgd_compile` /
        `svgd_cuda_graph` -- the compile of every stage and the capture of every graph this
        solver's (N, dtype, structure, step options) uses, outside any timed cell: a few
        steps on particles drawn around the program's current guess. Returns the seconds
        spent; `self.warmup_info` says what was compiled / captured."""
        t0 = time.perf_counter()
        self._build()
        x0 = np.nan_to_num(np.asarray(self.program.prog.GetInitialGuess(self.program.lumped_vars),
                                      dtype=float))
        X, _, _ = self._init_particles(x0)
        S = self._init_state(X)
        self._pool.drain()
        self._make_runner()
        step_fn = self._step_split if self._runner is not None else self._step_eager
        for _ in range(3):
            X, _, _ = step_fn(X, S)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if hasattr(self.program, "ResetEvalCounts"):
            self.program.ResetEvalCounts()
        r = self._runner
        self.warmup_info = dict(step_mode=r.mode if r is not None else "eager",
                                reused_template=bool(r is not None and r.reused),
                                graphs_captured=r.graphs_captured() if r is not None else 0,
                                seconds=time.perf_counter() - t0)
        return self.warmup_info["seconds"]
