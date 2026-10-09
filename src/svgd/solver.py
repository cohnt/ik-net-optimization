"""The `svgd` solver: `SvgdSolver(program).solve() -> SvgdResult`.

A common loop over three SVGD-style methods (`options.svgd_method`), all on the SAME target:
the program as written, replayed for N particles by `BatchedProgram` (`src/svgd/batched_program.py`)
and turned into a per-particle PHR augmented Lagrangian by `src/svgd/al.py`. Nothing here
knows what a decision variable means: the only formulation-agnostic structure used is that
every arm has a configuration `q` (the kernel's default space and the step clamp's unit).

    init -> (optional CEM warm-up) -> swarm(method) -> float64 polish -> select -> Drake re-check

ROW SCALING. Every equality `h` and inequality `g` row of the batched program is divided by
`tol * s`, `tol = options.acceptable_constr_viol_tol` (the harness's verify tolerance) and `s`
the row's group scale (`svgd_row_scale_rot` on orientation rows, 1 elsewhere), so that
`||[h~ ; max(g~, 0)]||_inf <= 1` means "feasible at the harness gate" for every row, the z/c
boxes and the trust region included. `BatchedProgram.RowScaling` cannot scale the region rows
(they carry no group), so the scaling is applied HERE from the public `h_spec` / `g_spec`.

THE STEP, as implemented (sign conventions settled against `tests/test_svgd_kernels_al.py`,
whose plain-SVGD loop is `y <- y + lr * phi` with `driving = -grad(neg log target)`):

  al_svgd (Tabor-Hermans "Q-method"):
      grad_F   = dF/dx                                   (one VJP through the graph)
      grad_AL  = grad_F + J_h^T (lam + rho h~) + J_g^T max(0, mu + rho g~)
      y        = q(x) | x | -      (svgd_kernel = q | x | none),  K, R = rbf_terms(y, h_med)
      R_pulled = J_q^T R  (q-space)  |  R  (x-space)  |  0  (none, K = I)
      phi      = (1/Z) [ gamma(t) K (-grad_F) + T(t) R_pulled ]          (svgd_direction)
      x        <- project_bounds( adam_step(x, grad = -(phi - grad_AL)) )
                 i.e. ascent along phi, descent along each particle's OWN AL gradient
      every svgd_gn_every steps, OUTSIDE Adam:
      J = [J_h~ ; active * J_g~],  r = [h~ ; active * g~],  dx = -J^T (J J^T + delta I)^-1 r,
      dx <- clamp_q_step(dx, J_q, svgd_q_step_max),  x <- project_bounds(x + dx)
  tsvgd (CSVTO / O-SVGD):
      J_A = [J_h~ ; active * J_g~],  P = I - J_A^T (J_A J_A^T + delta I)^-1 J_A
      driving  = P (-grad_F - J_g^T max(0, mu + rho g~))
      phi      = P * svgd_direction(K, driving, R_pulled, gamma, T)
      x        <- project_bounds( x + lr * phi + gn_correction(J_h~, h~) )     (alpha_C = 1)
      lr is the per-particle `lr_schedule` rate (no Adam); resampling is in the tangent
      space of the best particle: x_best + P_best xi, xi ~ N(0, svgd_jitter_z^2 I).
  admm_svgd: Stein-projected consensus ADMM on q_bar = f(x) (`src/svgd/admm.py`), its own
      outer loop on the same init, clock, trace, stall rule, polish and re-check; on the
      joint-space arm (f = I) it degenerates to a projection split and says so in
      `extras["admm"]`.

  LEVENBERG-MARQUARDT (`svgd_gn_lm > 0`, default on): the GN Gram carries Marquardt's
  `lam_i diag(J J^T)`, `lam_i` adapted per particle on the gain ratio of the step it took
  (actual over linearly predicted reduction of the GN rows' `|r|^2`; `al.lm_gain_update`),
  and the q-step clamp is applied on top.

Inequality multipliers and penalties follow Nocedal-Wright 17.4 (`update_multipliers`) at every
OUTER boundary, after `svgd_inner_iters` steps. The per-particle best is tracked by the merit
`F` where the particle is feasible (`infeas <= 1`) and `F + FEASIBLE_WEIGHT * infeas` otherwise,
so any feasible particle outranks any infeasible one and feasible particles rank by objective.

DERIVATIVES: THE ONLY BACKWARD IS THROUGH THE FLOW. On the fielded robots
(`BatchedProgram.has_analytic_row_jacobians`) every derivative downstream of the
configuration is closed form -- the task rows from the frame's geometric Jacobian
(`FrameChain.poses(jacobians=True)`, `generic_rows_jacobian_cfg`), the collision row from the
pool's own stored gradient, the joint-limit rows the identity, the region rows and the costs
in `x` directly (`extra_row_jacobian`, `cost_gradient_parts`) -- so `evaluate` detaches the
kinematics and the per-step autograd is ONE vmapped backward through the flow
(`_Target.pullback`: `dF/dcfg`, the rows' cotangent and the q-space repulsion stacked as
three cotangents), or `J_q` itself (ndof cotangents) on a Gauss-Newton step; the
joint-space arm, whose map is the identity, builds no graph at all. A robot behind its own
`BodyPoseProvider` falls back to autograd through its kinematics (`_Target.analytic = False`).
Measured at N = 64 float32 on the laptop (the IPOPT smoke sharing the GPU): a step fell from
181 / 106 ms (GN / no GN) to 40 / 34 ms, with the closed forms agreeing with autograd to
1e-15 on all four Panda programs (`tests/test_svgd_solver.py`). What remains is launch-bound
(evaluate at N = 1 is 14 ms) plus the pool's ~8 ms round trip.

THE INIT PROTOCOL cannot be read off the program (the harness called `SetStartFromQ` or
`SetNativeStart` before `Solve()`), so particle 0 is ALWAYS the program's initial guess,
exactly, never clipped (the +-5 latent-clip trap). The other N-1 are drawn around it
(`svgd_paired_init = "jitter"`, sigma per variable block) or from the arm's native
distribution (`"native"`: learned `c = native_c, z ~ N(0, I), q_c = 0`; joint space uniform in
`ConfigLimits`, kept collision-free by `ParallelCollisionChecker` with bounded redraws and a
documented fallback), then projected onto the TRUE variable bounds with the clip distance
recorded. The seed is `svgd_seed` mixed with a CRC of the initial guess so a cell reproduces.

BUDGETS. `svgd_outer_iters` (or `max_iter`) outer steps of `svgd_inner_iters`; the wall clock
is `max_wall_time * (1 - svgd_time_reserve)` for the swarm, checked after a CUDA sync at every
outer boundary, the reserve kept for polish and re-check. `program.RecordIterate(best)` runs
at every outer boundary so `last_iterate` is always the best particle so far.

THE COLLISION POOL is cached per `(SceneSpec, workers)` in this process (`_POOLS`, LRU of
`POOL_CACHE_SIZE`): the benchmark builds a new program per cell and spawning workers costs
seconds, while the scene is the same for every cell of a pose grid and for both arms of a
grasp cell. `svgd_collision_workers = None` resolves to `max(1, cpu_count // PROCS)`
(`PROCS` the environment's processes-per-node, 1 when unset). Pools close at exit.

THE STEP IS SPLIT AT THE POOL (`src/svgd/fused.py`, `_step_split`) on the fielded robots
for `al_svgd` and `tsvgd`: stage 1 (the configuration) -> the pool dispatched -> stage 2 (the
flow Jacobian, the kinematics) while Drake runs -> the pool collected -> stage 3 (the rest).
`svgd_compile` compiles the three stages and `svgd_cuda_graph` replays them as CUDA graphs,
captured by `warm_up` (`WarmUpSvgdStep`) before any timed cell and frozen by `run_grid`;
`svgd_pool_overlap=False` collects the pool before stage 2 (the overlap's control). A robot
behind its own pose provider runs `_step_al` / `_step_t` (autograd, eager); `admm_svgd` runs
eager always. `extras["step_mode"]` records which.

STOPPING. The swarm stops at the clock, at the outer-step cap, or -- `stop_reason =
"feasible_stall"`, status converged -- once some particle has been feasible at the gate and
the best feasible objective has improved by less than `svgd_stop_rel` (relative) over
`svgd_stop_patience` outer steps; `SvgdSolverDetails.stop_reason` records which.
"""

import atexit
import math
import os
import time
import warnings
import zlib
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from src.svgd import al, fused, kernels
from src.svgd.al import ALState
from src.svgd.batched_program import EXTRA_ROW_KEYS, GENERIC_BINDING, BatchedProgram
from src.svgd.collision_backend import DrakeCollisionPool, ParallelCollisionChecker, SceneSpec
from src.svgd.result import (STATUS_CONVERGED, STATUS_INFEASIBLE, STATUS_NAN, STATUS_STEP_CAP,
                             STATUS_WALL_CLOCK, SvgdResult, SvgdSolverDetails, status_name,
                             write_log)

## Any feasible particle outranks any infeasible one (merit = F + this * infeasibility where
## the scaled infeasibility exceeds 1). The objectives here are O(1e-4 .. 1e1).
FEASIBLE_WEIGHT = 1e6
POOL_CACHE_SIZE = 2
NATIVE_DRAW_ROUNDS = 20       # bounded rejection sampling for the joint-space native start
PATIENCE_REL = 1e-6
POLISH_ALPHA_MIN = 1.0 / 64   # shortest backtracked Newton step in the float64 polish


## ------------------------------------------------------------------------------------ ##
##                         shared collision pools and checkers                          ##
## ------------------------------------------------------------------------------------ ##

_POOLS = OrderedDict()        # (SceneSpec, workers) -> DrakeCollisionPool
_CHECKERS = OrderedDict()     # SceneSpec -> ParallelCollisionChecker


def resolve_workers(option):
    """`svgd_collision_workers`, or `max(1, cpu_count // PROCS)` when None."""
    if option is not None:
        return max(1, int(option))
    try:
        procs = max(1, int(os.environ.get("PROCS", "1") or 1))
    except ValueError:
        procs = 1
    return max(1, (os.cpu_count() or 2) // procs)


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
    key = (spec, int(workers))
    pool = _POOLS.get(key)
    if pool is not None and not pool.closed and all(pool.alive()):
        _POOLS.move_to_end(key)
        return pool
    if pool is not None:
        pool.close()
        del _POOLS[key]
    pool = _spawn_pool(spec, workers)
    _POOLS[key] = pool
    while len(_POOLS) > POOL_CACHE_SIZE:
        _, old = _POOLS.popitem(last=False)
        old.close()
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

class _Target:
    """A `BatchedProgram` plus the row scaling and the bookkeeping that assembles per-row
    Jacobians from the `RowSpec`s. One per dtype (the swarm's and the float64 polish's)."""

    def __init__(self, bp, tol, rot_scale):
        self.bp = bp
        self.dtype, self.device = bp.dtype, bp.device
        self.n = bp.nvars
        self.ndof = bp.ndof
        kw = dict(dtype=self.dtype, device=self.device)

        def scale(spec):
            return 1.0 / (float(tol) * (float(rot_scale) if spec.group == "rotation" else 1.0))
        self.sh = torch.tensor([scale(r) for r in bp.h_spec], **kw)
        self.sg = torch.tensor([scale(r) for r in bp.g_spec], **kw)
        self.m_e, self.m_i = len(bp.h_spec), len(bp.g_spec)

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

        ## Routing of the GENERIC rows' derivative w.r.t. the configuration (`generic_blocks`):
        ## task rows (pose / mug) by a vmapped backward through the frame chain, the
        ## collision row from the pool's stored gradient, the joint-limit rows the identity.
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
        self._E_q = torch.eye(self.ndof, **kw).unsqueeze(1)              # [ndof, 1, ndof]
        ## Derivative mode (see `evaluate`): closed-form rows where the batched program
        ## offers them, autograd through the kinematics otherwise.
        self.analytic = bool(bp.has_analytic_row_jacobians)

    ## -- batched autograd, with the vmap-fallback chatter silenced ----------------------
    @staticmethod
    def _batched_grad(outputs, inputs, E, N):
        """`autograd.grad` with `is_grads_batched` over the rows of `E [m, 1, m]`
        (expanded to `[m, N, m]`), returning `[N, m, ...]`. torch warns, per call, about
        every op without a batching rule ("There is a performance drop because we have not
        yet implemented the batching rule for ..."); the fallback is correct and the warning
        is noise at thousands of steps, so it is filtered HERE, around this call only."""
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
        """Rows and costs at `X [N, n]`; the scaled `h`, `g`, infeasibility and a per-particle
        finiteness flag. With `need_grad`, the derivative machinery is set up in the mode
        the target supports: ANALYTIC (`bp.has_analytic_row_jacobians`) keeps a graph through
        the flow only (`q <- x`), everything downstream of `q` -- kinematics, rows, costs --
        is detached and differentiated in closed form (`generic_rows_jacobian_cfg`,
        `extra_row_jacobian`, `cost_gradient_parts`); on the joint-space arm, whose map is
        the identity, no graph is built at all. AUTOGRAD (a robot behind its own pose
        provider) keeps the full graph and differentiates the rows with `autograd`."""
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
        """`(h~, g~, finite [N], infeas [N])` from a batched `Evaluation`: the rows divided by
        `tol * s` (see the module docstring), the per-particle finiteness flag and the
        scaled infeasibility `||[h~; max(g~, 0)]||_inf` (+inf where not finite)."""
        h = out.h * self.sh
        g = out.g * self.sg
        finite = (torch.isfinite(h).all(dim=1) & torch.isfinite(g).all(dim=1)
                  & torch.isfinite(out.F) & torch.isfinite(out.q).all(dim=1))
        gplus = torch.clamp(g, min=0.0)
        stacked = torch.cat([h.abs(), gplus], dim=1)
        infeas = stacked.amax(dim=1) if stacked.shape[1] else torch.zeros_like(out.F)
        infeas = torch.where(finite, infeas, torch.full_like(infeas, float("inf"))).detach()
        return h, g, finite, infeas

    def merit_of(self, F, infeas):
        """`F` where feasible at the gate, else `F + FEASIBLE_WEIGHT * infeas`."""
        excess = torch.where(infeas <= 1.0, torch.zeros_like(infeas), infeas)
        return al.best_merit(F.detach(), excess, FEASIBLE_WEIGHT)

    def merit(self, ev):
        return self.merit_of(ev.F, ev.infeas)

    def unscale(self, h, g):
        """The inverse of `scale`'s row scaling: `(h~ * tol * s, g~ * tol * s)`, i.e. the
        `Evaluation`'s own `h` / `g` (Drake's signed violations under `RowScaling`)."""
        return h / self.sh, g / self.sg

    ## -- derivatives ------------------------------------------------------------------
    def pullback(self, ev, cots):
        """`[J_q^T v for v in cots]`, each `v [N, ndof]` -> `[N, n]`: the configuration
        cotangents pulled back to the decision variables. Learned arm: ONE vmapped backward
        through the flow with all of them stacked; joint space: the identity."""
        if not self.bp.is_learned:
            return [v.detach().clone() for v in cots]
        N = ev.Xg.shape[0]
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
        autograd mode the whole gradient lands in the direct part (`dF/dcfg = 0`)."""
        if self.analytic:
            return self.bp.cost_gradient_parts(ev.Xg, ev.cfg)
        (g,) = torch.autograd.grad(ev.F.sum(), ev.Xg, retain_graph=True)
        return torch.zeros_like(ev.cfg), g

    def row_cotangents(self, ev, ch, cg):
        """The cotangent on the stacked UNSCALED rows `[drake_rows | extra rows]` that the
        scaled `h`, `g` cotangents `ch`, `cg` induce: `[N, nr + n_extra]`."""
        N = ev.Xg.shape[0]
        c = torch.zeros(N, self.bp.n_rows + self._n_extra, dtype=self.dtype, device=self.device)
        c.index_add_(1, self.h_idx, ch * self.h_coef)
        c.index_add_(1, self.g_idx, cg * self.g_coef)
        return c

    def al_gradients(self, ev, ch, cg, R):
        """`(dF/dx, J_h^T ch + J_g^T cg, J_q^T R)` -- the three gradients a non-GN
        `al_svgd` step needs -- with ONE backward through the flow (`R` may be None).
        Analytic mode: the rows' cotangent is carried to the configuration through the
        closed-form row Jacobians, the extras' directly to `x`. Autograd mode: the row
        term is one VJP through the full graph."""
        N = ev.Xg.shape[0]
        nr = self.bp.n_rows
        dFc, dFx = self.cost_parts(ev)
        if self.analytic:
            c = self.row_cotangents(ev, ch, cg)
            Dq = self.generic_jacobian_cfg(ev)                            # [N, nr, ndof]
            v_cfg = (Dq.transpose(1, 2) @ c[:, :nr].unsqueeze(2)).squeeze(2)
            J_ex = self._extra_jacobian(ev.Xg, N)
            v_x = (J_ex.transpose(1, 2) @ c[:, nr:].unsqueeze(2)).squeeze(2) if self._n_extra else 0.0
            cots = [dFc, v_cfg] + ([R] if R is not None else [])
            G = self.pullback(ev, cots)
            gF = G[0] + dFx
            gC = G[1] + v_x
            R_pulled = G[2] if R is not None else None
            return gF, gC, R_pulled
        s = (ev.h * ch).sum() + (ev.g * cg).sum()
        (gC,) = torch.autograd.grad(s, ev.Xg, retain_graph=True)
        R_pulled = self.pullback(ev, [R])[0] if R is not None else None
        return dFx, gC, R_pulled

    def grad_F(self, ev):
        dFc, dFx = self.cost_parts(ev)
        return self.pullback(ev, [dFc])[0] + dFx if self.analytic else dFx

    def jacobian_q(self, ev):
        """`dq/dx [N, ndof, n]` from the live graph (identity on joint space)."""
        bp = self.bp
        N = ev.Xg.shape[0]
        if not bp.is_learned:
            return bp.jacobian_q(ev.Xg.detach())
        J = self._batched_grad(ev.cfg, ev.Xg, self._E_q, N)
        self._count("map_jacobian")
        return J

    def generic_jacobian_cfg(self, ev):
        """`d drake_rows / d cfg`, `[N, nr, ndof]`. Analytic mode: the batched program's
        closed form (geometric Jacobian of the task frame, the pool's collision gradient,
        identity joint limits). Autograd mode: routed by row kind -- the task rows by one
        vmapped backward through the kinematics, the collision row from the pool's stored
        `d row / d q_plant` through `config_to_plant_q`, the joint-limit rows the identity."""
        out = ev.out
        if self.analytic:
            return torch.nan_to_num(self.bp.generic_rows_jacobian_cfg(out), nan=0.0,
                                    posinf=0.0, neginf=0.0)
        D = out.drake_rows
        N, nr = D.shape
        Dq = torch.zeros(N, nr, self.ndof, dtype=D.dtype, device=D.device)
        if self._task_rows.numel():
            Jt = self._batched_grad(D[:, self._task_rows], ev.cfg, self._E_task, N)
            Dq[:, self._task_rows] = Jt
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

    def row_jacobians(self, ev, J_q):
        """`(J_h [N, m_e, n], J_g [N, m_i, n])` of the SCALED rows."""
        return self._stack_jacobians(self.generic_jacobian_cfg(ev) @ J_q, ev.Xg)

    def row_jacobians_out(self, out, X, J_q):
        """`row_jacobians` from a bare `Evaluation` (analytic mode only): what the split
        step calls, having no `ev` namespace and no graph."""
        Dq = torch.nan_to_num(self.bp.generic_rows_jacobian_cfg(out), nan=0.0, posinf=0.0,
                              neginf=0.0)
        return self._stack_jacobians(Dq @ J_q, X)

    def _stack_jacobians(self, J_gen, X):
        J_ex = self._extra_jacobian(X, X.shape[0])
        J_stack = torch.cat([J_gen, J_ex], dim=1)
        J_h = J_stack[:, self.h_idx] * self.h_coef.view(1, -1, 1)
        J_g = J_stack[:, self.g_idx] * self.g_coef.view(1, -1, 1)
        return J_h, J_g

    ## -- the per-outer-iteration trace ---------------------------------------------------
    TRACE_COLUMNS = ("min_infeas", "med_infeas", "n_feasible", "med_rho", "med_lr",
                     "frac_gn_clamped", "med_h_pos", "med_h_rot", "med_g_plus",
                     "min_F_feasible", "med_dq", "med_nw_infeas", "med_eta", "med_gn_lam",
                     "n_tangent", "frac_nw_ok", "med_gn_ratio")

    def trace_row(self, ev, S, dq, clamped, nw_ok=None):
        """One `[len(TRACE_COLUMNS)]` tensor of swarm statistics at an outer boundary, all
        on the device (the caller stacks them and moves them to the host ONCE at the end):
        the best and median scaled violation, the feasible count, median rho and lr, the
        fraction of particles whose last GN step hit the q-step clamp (`nan` when none ran),
        the median over particles of the worst position / rotation equality row and of the
        worst violated inequality, the best objective among feasible particles, the median
        configuration step, the median Nocedal-Wright infeasibility measure, the median AL
        tolerance `eta` and LM damping, the number of particles in tangent mode (tsvgd;
        `nan` elsewhere), the fraction whose last NW test passed (`nw_ok`) and the median LM
        gain ratio (actual / predicted reduction of the GN rows; `nan` with LM off).""" 
        kw = dict(dtype=self.dtype, device=self.device)
        nan = torch.tensor(float("nan"), **kw)
        infeas = torch.nan_to_num(ev.infeas, nan=float("inf"))
        feas = infeas <= 1.0
        habs = torch.nan_to_num(ev.h.detach().abs(), nan=float("inf"))
        gpos = torch.nan_to_num(torch.clamp(ev.g.detach(), min=0.0), nan=float("inf"))

        def med_group(mask, count):           # `count` is a Python int fixed at construction
            if count == 0:
                return nan
            return habs[:, mask].amax(dim=1).median()
        F = torch.nan_to_num(ev.F.detach(), nan=float("inf"))
        minF = torch.where(feas, F, torch.full_like(F, float("inf"))).min()
        minF = torch.where(torch.isfinite(minF), minF, nan)
        frac = clamped.to(self.dtype).mean() if clamped is not None else nan
        dqm = torch.nan_to_num(dq, nan=float("inf")).median() if dq is not None else nan
        return torch.stack([
            infeas.min(), infeas.median(), feas.sum().to(self.dtype),
            S.rho.median(), S.lr.median(), frac,
            med_group(~self.h_is_rot, self.m_e - self._n_rot), med_group(self.h_is_rot, self._n_rot),
            gpos.amax(dim=1).median() if gpos.shape[1] else nan,
            minF, dqm, al.infeasibility(ev.h.detach(), ev.g.detach(), S).median(),
            S.eta.median(), S.gn_lam.median(),
            ev.n_tangent.to(self.dtype).sum() if getattr(ev, "n_tangent", None) is not None else nan,
            nw_ok.to(self.dtype).mean() if nw_ok is not None else nan,
            torch.nanmedian(ev.gn_ratio) if getattr(ev, "gn_ratio", None) is not None else nan])

    def _count(self, bucket):
        counts = getattr(self.bp.program, "eval_counts", None)
        if counts is not None:
            counts[bucket] = counts.get(bucket, 0) + 1

    def project(self, X):
        return self.bp.project(X)


## ------------------------------------------------------------------------------------ ##
##                                      the solver                                       ##
## ------------------------------------------------------------------------------------ ##

class SvgdSolver:
    """See the module docstring.

    `particles_override` (tests only): a `[N, n]` array used as the initial swarm instead of
    the drawn one -- the way to hand the solver a non-finite particle and watch it resample.

    `admm_svgd` runs its own outer loop (`src/svgd/admm.py`, `AdmmSwarm.run`) in place of
    `_swarm`, on the same init, clock, trace, stall rule, polish and selection.
    """

    def __init__(self, program, particles_override=None):
        self.program = program
        self.options = program.options
        opts = self.options
        self.method = opts.svgd_method
        self._lm = float(opts.svgd_gn_lm) > 0.0
        self._sc = fused.StepConfig.from_options(opts)
        self._overlap = bool(opts.svgd_pool_overlap)
        self._runner = None
        self.N = int(opts.svgd_n)
        self.dtype = torch.float64 if opts.svgd_dtype == "float64" else torch.float32
        self.workers = resolve_workers(opts.svgd_collision_workers)
        self.tol = float(opts.acceptable_constr_viol_tol)
        self._override = None if particles_override is None else np.asarray(particles_override, dtype=float)
        self._tg = None
        self._tg64 = None
        self._pool = None

    ## ----------------------------------- setup ----------------------------------------
    def _build(self):
        if self._tg is not None:
            return
        program = self.program
        opts = self.options
        spec = _scene_spec(program)
        self._spec = spec
        raw = shared_pool(spec, self.workers)
        self._pool = _TimedPool(raw)
        bp = BatchedProgram.from_program(program, dtype=self.dtype, pool=self._pool)
        if bp.device.type == "cuda":
            ## cuSOLVER for every batched solve (`solve_ex` in the GN correction, the
            ## tangent projector, the polish). PROCESS-WIDE, and deliberate: torch's default
            ## heuristic routes larger batches (measured: [256, 27, 27], and the [64, 47, 47]
            ## Gram of a learned pose step) to MAGMA, whose batched getrf/potrs allocate
            ## inside the call and so cannot be captured into a CUDA graph; cuSOLVER captured
            ## at every size and dtype probed. Set for eager and compiled runs too, so the
            ## three step modes share one backend's numerics.
            torch.backends.cuda.preferred_linalg_library("cusolver")
        self._tg = _Target(bp, self.tol, opts.svgd_row_scale_rot)
        if self.dtype == torch.float64:
            self._tg64 = self._tg
        else:
            bp64 = BatchedProgram.from_program(program, dtype=torch.float64, pool=self._pool,
                                               device=bp.device)
            self._tg64 = _Target(bp64, self.tol, opts.svgd_row_scale_rot)
        self.device = bp.device
        self.n = bp.nvars

    def _seed(self, x0):
        crc = zlib.crc32(np.ascontiguousarray(x0, dtype=np.float64).tobytes())
        return int((int(self.options.svgd_seed) * 1000003 + crc) % (2 ** 31 - 1))

    def _generator(self, seed):
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        return g

    def _jitter_sigma(self):
        opts = self.options
        bp = self._tg.bp
        if bp.is_learned:
            s = ([opts.svgd_jitter_c_pos] * 3 + [opts.svgd_jitter_c_rot] * 3
                 + [opts.svgd_jitter_z] * bp.width + [opts.svgd_jitter_qc] * bp.ndof)
        else:
            s = [opts.svgd_jitter_q] * bp.ndof
        return torch.tensor(s, dtype=self.dtype, device=self.device)

    def _draw(self, N, x0, gen):
        """N fresh particles from the init distribution (unprojected); `(X, stats)`."""
        bp = self._tg.bp
        stats = {}
        if self.options.svgd_paired_init == "jitter":
            xi = torch.randn(N, self.n, generator=gen, dtype=self.dtype, device=self.device)
            return x0.unsqueeze(0) + xi * self._jitter_sigma().unsqueeze(0), stats
        ## native
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

    def _init_particles(self, x0_np):
        x0 = torch.tensor(x0_np, dtype=self.dtype, device=self.device)
        gen = self._generator(self._seed(x0_np))
        self._gen = gen
        self._x0 = x0
        stats = {}
        if self._override is not None:
            X = torch.tensor(self._override, dtype=self.dtype, device=self.device)
            if X.shape != (self.N, self.n):
                raise ValueError(f"particles_override must be [{self.N}, {self.n}]")
            clip = torch.zeros(self.N, dtype=self.dtype, device=self.device)
            return X, clip, stats
        X, stats = self._draw(self.N, x0, gen)
        Xp, clip = self._tg.project(X)
        keep0 = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        keep0[0] = True
        X = torch.where(keep0.unsqueeze(1), x0.unsqueeze(0).expand_as(Xp), Xp)
        clip = torch.where(keep0, torch.zeros_like(clip), clip)
        return X, clip, stats

    def _fresh(self, N):
        X, _ = self._draw(N, self._x0, self._gen)
        Xp, _ = self._tg.project(X)
        return Xp

    ## ------------------------------- the steps ----------------------------------------
    def _kernel_terms(self, ev, X):
        """`(K, R, needs_pullback)`: `fused.kernel_terms` on an evaluation (the kernel and
        the repulsion in the kernel's space; non-finite particles isolated)."""
        return fused.kernel_terms(self._sc, ev.cfg, ev.finite, X)

    def _gn_step(self, ev, S, J_h, J_g, rows_active):
        """The (unclamped) Gauss-Newton correction onto the equality rows and, when
        `rows_active`, the PHR-active inequality rows, LM-damped by `S.gn_lam` where LM is
        on; returns `(dx, (J, r, mask))`, the row set for `al.lm_record_prediction`."""
        opts = self.options
        ones = torch.ones_like(ev.h)
        if rows_active:
            mask = torch.cat([ones, al.active_mask(ev.g, S.mu, S.rho)], dim=1)
        else:
            mask = torch.cat([ones, torch.zeros_like(ev.g)], dim=1)
        J, r = fused.gn_rows(ev.h, ev.g, J_h, J_g, mask)
        lam = S.gn_lam if self._lm else None
        return al.gn_correction(J, r, opts.svgd_gn_delta, lam), (J, r, mask)

    def _lm_gain(self, S, ev):
        """`al.lm_gain_update` where LM is on (`svgd_gn_lm > 0`): `(S, ratio)`."""
        if not self._lm:
            return S, None
        opts = self.options
        return al.lm_gain_update(S, ev.h.detach(), ev.g.detach(), opts.svgd_gn_lm_growth,
                                 opts.svgd_gn_lm_min, opts.svgd_gn_lm_max)

    def _clamp(self, dx, J_q):
        """`clamp_q_step` plus the per-particle flag that the bound was hit (for the trace:
        a clamp that binds every step says `svgd_q_step_max` is the knob)."""
        Jq = J_q if self._tg.bp.is_learned else None
        dq = dx if Jq is None else (Jq @ dx.unsqueeze(2)).squeeze(2)
        clamped = dq.abs().amax(dim=1) > float(self.options.svgd_q_step_max)
        return al.clamp_q_step(dx, Jq, self.options.svgd_q_step_max), clamped

    def _step_al(self, X, S, t_outer, t_step, total_steps, do_gn):
        """One `al_svgd` step. Without a GN correction (`do_gn=False`) the whole derivative
        work is ONE vmapped backward (`vjp_three`: dF/dx, the AL row cotangents, the q-space
        repulsion pullback); with it, `J_q` and the row Jacobians are added."""
        tg = self._tg
        opts = self.options
        ev = tg.evaluate(X)
        S, ev.gn_ratio = self._lm_gain(S, ev)
        ch, cg = al.al_constraint_grad_coefficients(ev.h, ev.g, S)
        K, R, pull = self._kernel_terms(ev, X)
        J_q = None
        lm_set = None
        if do_gn:
            J_q = tg.jacobian_q(ev)
            J_h, J_g = tg.row_jacobians(ev, J_q)
            dFc, dFx = tg.cost_parts(ev)
            gF = (J_q.transpose(1, 2) @ dFc.unsqueeze(2)).squeeze(2) + dFx
            gC = (J_h.transpose(1, 2) @ ch.unsqueeze(2)).squeeze(2) \
                + (J_g.transpose(1, 2) @ cg.unsqueeze(2)).squeeze(2)
            R_pulled = (J_q.transpose(1, 2) @ R.unsqueeze(2)).squeeze(2) if pull else R
        else:
            gF, gC, R_pulled = tg.al_gradients(ev, ch, cg, R if pull else None)
            if not pull:
                R_pulled = R
        fin = ev.finite.unsqueeze(1)
        gF = torch.where(fin, gF, torch.zeros_like(gF))
        gAL = torch.where(fin, gF + gC, torch.zeros_like(gF))
        R_pulled = torch.where(fin, R_pulled, torch.zeros_like(R_pulled))
        gamma = kernels.anneal_gamma(t_outer, opts.svgd_gamma_t)
        T = kernels.anneal_T(t_step, opts.svgd_repulsion_T0, opts.svgd_anneal_frac, total_steps)
        phi = kernels.svgd_direction(K, -gF, R_pulled, gamma, T)
        Xn, S = al.adam_step(X, -(phi - gAL), S)
        ev.gn_clamped = None
        ev.dq_adam = None
        if do_gn:
            ## The lr schedule must see ADAM'S OWN configuration step, not the total
            ## movement: the clamped GN correction alone sits at ~q_step_max, so measuring
            ## the sum halved the learning rate on every GN step and froze the Stein/AL
            ## part at `svgd_lr_min` (read off the trace: med_lr at the floor from outer 1).
            ## On a GN step it is estimated linearly with the J_q already in hand; on a
            ## non-GN step the observed step IS Adam's and the caller measures it.
            if tg.bp.is_learned:
                ev.dq_adam = (J_q @ (Xn - X).unsqueeze(2)).squeeze(2).abs().amax(dim=1)
            else:
                ev.dq_adam = (Xn - X).abs().amax(dim=1)
            dx, lm_set = self._gn_step(ev, S, J_h, J_g, rows_active=True)
            dx, clamped = self._clamp(dx, J_q)
            Xn = Xn + dx
            ev.gn_clamped = clamped
        Xn, _ = tg.project(Xn)
        if self._lm and lm_set is not None:
            S = al.lm_record_prediction(S, *lm_set, Xn.detach() - X)
        S = al.track_best(S, X, tg.merit(ev))
        return Xn.detach(), S, ev

    def _step_t(self, X, S, t_outer, t_step, total_steps, do_gn):
        """One `tsvgd` step: ONE evaluation and one set of Jacobians, from which BOTH an
        `al_svgd` update (the GN path) and a tangent-space update are formed, and each
        particle takes the tangent one iff its scaled infeasibility is at or below
        `svgd_tsvgd_switch_infeas` (`torch.where`; the extra cost of the second update is
        the projector solve). Tangent mode from a far start was the first wave's failure:
        the manifold correction alone had to do the equality work, 6+ rad a step, clamped
        to `q_step_max` and fighting the lr schedule -- so a particle is carried near the
        manifold by the AL first. In tangent mode the Stein step and the manifold
        correction are CLAMPED SEPARATELY (each to `q_step_max`), so a long correction no
        longer scales the Stein step to nothing and the lr schedule sees only its own step."""
        tg = self._tg
        opts = self.options
        ev = tg.evaluate(X)
        S, ev.gn_ratio = self._lm_gain(S, ev)
        J_q = tg.jacobian_q(ev)
        J_h, J_g = tg.row_jacobians(ev, J_q)
        dFc, dFx = tg.cost_parts(ev)
        gF = (J_q.transpose(1, 2) @ dFc.unsqueeze(2)).squeeze(2) + dFx
        J_h = torch.nan_to_num(J_h.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        J_g = torch.nan_to_num(J_g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        ch, cg = al.al_constraint_grad_coefficients(ev.h, ev.g, S)
        gI = (J_g.transpose(1, 2) @ cg.unsqueeze(2)).squeeze(2)
        gC = (J_h.transpose(1, 2) @ ch.unsqueeze(2)).squeeze(2) + gI
        fin = ev.finite.unsqueeze(1)
        gF = torch.where(fin, gF, torch.zeros_like(gF))
        gAL = torch.where(fin, gF + gC, torch.zeros_like(gF))
        K, R, pull = self._kernel_terms(ev, X)
        R_pulled = (J_q.transpose(1, 2) @ R.unsqueeze(2)).squeeze(2) if pull else R
        R_pulled = torch.where(fin, R_pulled, torch.zeros_like(R_pulled))
        gamma = kernels.anneal_gamma(t_outer, opts.svgd_gamma_t)
        T = kernels.anneal_T(t_step, opts.svgd_repulsion_T0, opts.svgd_anneal_frac, total_steps)

        ## -- the al_svgd update (the GN path of `_step_al`, verbatim) --
        phi_al = kernels.svgd_direction(K, -gF, R_pulled, gamma, T)
        Xn_al, S = al.adam_step(X, -(phi_al - gAL), S)
        if tg.bp.is_learned:
            dq_al = (J_q @ (Xn_al - X).unsqueeze(2)).squeeze(2).abs().amax(dim=1)
        else:
            dq_al = (Xn_al - X).abs().amax(dim=1)
        dx_gn, set_al = self._gn_step(ev, S, J_h, J_g, rows_active=True)
        dx_gn, clamped_al = self._clamp(dx_gn, J_q)
        Xn_al = Xn_al + dx_gn

        ## -- the tangent-space update --
        act = al.active_mask(ev.g, S.mu, S.rho)
        J_A = torch.cat([J_h, J_g * act.unsqueeze(2)], dim=1)
        P = al.tangent_projector(J_A, opts.svgd_tangent_delta)
        driving = torch.where(fin, -(gF + gI), torch.zeros_like(gF))
        driving = (P @ driving.unsqueeze(2)).squeeze(2)
        phi = kernels.svgd_direction(K, driving, R_pulled, gamma, T)
        phi = (P @ phi.unsqueeze(2)).squeeze(2)
        step_lr, clamped_t = self._clamp(S.lr.unsqueeze(1) * phi, J_q)
        dx_c, set_t = self._gn_step(ev, S, J_h, J_g, rows_active=False)
        dx_c, clamped_c = self._clamp(dx_c, J_q)
        Xn_t = X + step_lr + dx_c
        if tg.bp.is_learned:
            dq_t = (J_q @ step_lr.unsqueeze(2)).squeeze(2).abs().amax(dim=1)
        else:
            dq_t = step_lr.abs().amax(dim=1)

        ## -- per-particle mode --
        tangent = (torch.nan_to_num(ev.infeas, nan=float("inf"))
                   <= float(opts.svgd_tsvgd_switch_infeas))
        Xn = torch.where(tangent.unsqueeze(1), Xn_t, Xn_al)
        ev.dq_adam = torch.where(tangent, dq_t, dq_al)
        clamped = torch.where(tangent, clamped_c, clamped_al)
        ev.gn_clamped = clamped
        ev.n_tangent = tangent
        Xn, _ = tg.project(Xn)
        if self._lm:
            t1 = tangent.unsqueeze(1)
            S = al.lm_record_prediction(
                S, torch.where(tangent.view(-1, 1, 1), set_t[0], set_al[0]),
                torch.where(t1, set_t[1], set_al[1]), torch.where(t1, set_t[2], set_al[2]),
                Xn.detach() - X)
        S = al.track_best(S, X, tg.merit(ev))
        self._last_P = P.detach()
        return Xn.detach(), S, ev

    ## --------------------------- the split (fused) step -------------------------------
    def _step_mode(self):
        """`eager` | `compiled` | `graphed` from `svgd_compile` / `svgd_cuda_graph` (graphs
        need CUDA; on a CPU device the request degrades to `compiled`, recorded as such)."""
        opts = self.options
        if not opts.svgd_compile:
            return "eager"
        if opts.svgd_cuda_graph and self.device.type == "cuda":
            return "graphed"
        return "compiled"

    def _make_runner(self):
        """The split step's `fused.StepRunner` where it applies (al_svgd / tsvgd on a target
        with closed-form row Jacobians); None elsewhere (admm_svgd's own loop; a robot behind
        its own pose provider, which runs `_step_al` / `_step_t` through autograd)."""
        self._runner = None
        if self.method in ("al_svgd", "tsvgd") and self._tg.analytic:
            self._runner = fused.StepRunner(self._tg, self._sc, self.N, self._step_mode())
        kw = dict(dtype=self.dtype, device=self.device)
        self._gamma_t = torch.zeros((), **kw)
        self._T_t = torch.zeros((), **kw)
        self._tout_t = torch.zeros((), **kw)
        return self._runner

    def _step_split(self, X, S, t_outer, t_step, total_steps, do_gn):
        """One `al_svgd` / `tsvgd` step as `fused`'s three stages with the collision pool
        between them: stage 1 (the configuration), the pool DISPATCHED, stage 2 (the flow
        Jacobian and the kinematics) launched while Drake runs, the pool collected, stage 3.
        The same maths as `_step_al` / `_step_t` plus the lr schedule (`ev.lr_done`)."""
        r = self._runner
        opts = self.options
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
        self._gamma_t.fill_(float(kernels.anneal_gamma(t_outer, opts.svgd_gamma_t)))
        self._T_t.fill_(float(kernels.anneal_T(t_step, opts.svgd_repulsion_T0,
                                               opts.svgd_anneal_frac, total_steps)))
        self._tout_t.fill_(float(t_outer))
        o = r.s3(do_gn, X, cfg, J_q, kin, col_row, col_grad, fused.state_dict(S),
                 self._gamma_t, self._T_t, self._tout_t)
        S = ALState(**o["S"])
        ev = SimpleNamespace(h=o["h"], g=o["g"], F=o["F"], infeas=o["infeas"], finite=o["finite"],
                             cfg=cfg, gn_clamped=o["clamped"], dq_adam=o["dq_adam"],
                             gn_ratio=o["ratio"], lr_done=True, n_tangent=o.get("tangent"))
        if "P" in o:
            self._last_P = o["P"]
        return o["X"], S, ev

    ## ------------------------------- resampling ---------------------------------------
    def _reset_state(self, S, mask):
        """Fresh AL / Adam state on the masked particles; the best is kept."""
        opts = self.options
        m1 = mask.unsqueeze(1)
        z = lambda a: torch.where(m1, torch.zeros_like(a), a)
        return replace(S, lam=z(S.lam), mu=z(S.mu),
                       rho=torch.where(mask, torch.full_like(S.rho, float(opts.svgd_rho0)), S.rho),
                       eta=torch.where(mask, torch.full_like(S.eta, self._eta0), S.eta),
                       adam_m=z(S.adam_m), adam_v=z(S.adam_v),
                       step=torch.where(mask, torch.zeros_like(S.step), S.step),
                       lr=torch.where(mask, torch.full_like(S.lr, float(opts.svgd_lr)), S.lr),
                       gn_lam=torch.where(mask, torch.full_like(S.gn_lam, float(opts.svgd_gn_lm)), S.gn_lam),
                       gn_r2=torch.where(mask, torch.full_like(S.gn_r2, float("nan")), S.gn_r2))

    def _resample(self, X, S, ev_q, finite_rows, merit):
        """Redraw runaway / non-finite particles (both methods) and, under tsvgd, the worst
        decile in the tangent space of the best particle. Returns `(X, S, n_resampled)`."""
        opts = self.options
        N = X.shape[0]
        mask = al.resample_mask(ev_q, finite_rows, opts.svgd_resample_q_max)
        if self.method == "tsvgd" and N >= 10:
            k = max(1, N // 10)
            thresh = torch.kthvalue(torch.nan_to_num(merit, nan=float("inf")), N - k + 1).values
            mask = mask | (merit >= thresh) | ~torch.isfinite(merit)
        best_i = torch.argmin(torch.nan_to_num(S.best_merit, nan=float("inf")))
        best_ok = torch.isfinite(S.best_merit[best_i]) & torch.isfinite(S.best_x[best_i]).all()
        fresh = self._fresh(N)
        if self.method == "tsvgd" and getattr(self, "_last_P", None) is not None and bool(best_ok):
            xi = torch.randn(N, self.n, generator=self._gen, dtype=self.dtype, device=self.device)
            xi = xi * float(opts.svgd_jitter_z)
            P_best = self._last_P[best_i]
            tangent = S.best_x[best_i].unsqueeze(0) + xi @ P_best.T
            tangent, _ = self._tg.project(tangent)
            fresh = torch.where(torch.isfinite(tangent).all(dim=1, keepdim=True), tangent, fresh)
        X = torch.where(mask.unsqueeze(1), fresh, X)
        S = self._reset_state(S, mask)
        return X, S, int(mask.sum().item())

    ## ---------------------------------- swarm -----------------------------------------
    def _swarm(self, X, S, deadline, outer_iters, inner_iters, record):
        """The outer/inner loop; returns `(X, S, stop_status, stats)`."""
        opts = self.options
        tg = self._tg
        total_steps = max(1, outer_iters * inner_iters)
        if self._runner is not None:
            step_fn = self._step_split
        else:
            step_fn = self._step_al if self.method == "al_svgd" else self._step_t
        stats = dict(outer=0, inner=0, n_resampled=0, best_history=[], trace=[],
                     stop_reason="step_cap")
        prev_q = None
        best_seen = float("inf")
        stale = 0
        status = STATUS_STEP_CAP
        t_step = 0
        last_ev = None
        dq = None
        for t_outer in range(outer_iters):
            ## -- resample check (runaway / nan), at the outer boundary --
            if t_outer % max(1, int(opts.svgd_resample_every)) == 0:
                with torch.no_grad():
                    ev0 = tg.evaluate(X, need_grad=False)
                if t_outer == 0:
                    ## The AL tolerance seeded from each particle's OWN initial
                    ## infeasibility (`svgd_eta_rel`); the resample below resets the
                    ## redrawn particles to the absolute `eta0`, which their first failed
                    ## test then replaces by the relative one.
                    S = al.seed_eta(ev0.h, ev0.g, S, opts.svgd_eta_rel)
                X, S, n_re = self._resample(X, S, ev0.cfg, ev0.finite, tg.merit(ev0))
                stats["n_resampled"] += n_re
            for k in range(inner_iters):
                do_gn = (t_step % max(1, int(opts.svgd_gn_every))) == 0
                X, S, ev = step_fn(X, S, t_outer, t_step, total_steps, do_gn)
                q_now = ev.cfg.detach()
                if prev_q is not None:
                    dq = torch.nan_to_num((q_now - prev_q).abs().amax(dim=1), nan=float("inf"))
                else:
                    dq = torch.zeros(X.shape[0], dtype=self.dtype, device=self.device)
                ## `dq` (the observed step, for the trace) includes the previous step's GN
                ## correction; the lr schedule gets Adam's own step where the step reports it.
                if not getattr(ev, "lr_done", False):          # the split step does its own
                    dq_lr = getattr(ev, "dq_adam", None)
                    if dq_lr is None:
                        dq_lr = dq
                    S = al.lr_schedule(S, torch.nan_to_num(dq_lr, nan=float("inf")),
                                       opts.svgd_q_step_max, opts.svgd_lr, opts.svgd_lr_min,
                                       float(t_outer), opts.svgd_lr_decay_t)
                prev_q = q_now
                last_ev = ev
                t_step += 1
                stats["inner"] += 1
            nw_ok = al.infeasibility(ev.h.detach(), ev.g.detach(), S) <= S.eta
            S = al.update_multipliers(ev.h.detach(), ev.g.detach(), S, opts.svgd_rho_growth,
                                      opts.svgd_rho_max, opts.svgd_multiplier_max,
                                      eta_rel=opts.svgd_eta_rel)
            stats["outer"] = t_outer + 1
            stats["trace"].append(tg.trace_row(ev, S, dq, getattr(ev, "gn_clamped", None), nw_ok))
            ## -- outer boundary: sync, record, clock, patience --
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            best_i = int(torch.argmin(torch.nan_to_num(S.best_merit, nan=float("inf"))).item())
            best_m = float(S.best_merit[best_i].item())
            record(S.best_x[best_i])
            stats["best_history"].append(best_m)
            ## The stall test. Before any particle is feasible every improvement of the
            ## merit counts (PATIENCE_REL); once one is, the merit IS the best feasible
            ## particle's objective and an outer step counts as progress only if it
            ## improved by `svgd_stop_rel` relative -- the first wave's 1e-6 never fired
            ## and every cell polished its cost for the whole cap. It is a stall on the
            ## objective the solver minimises, not on the harness's reported subset,
            ## which the solver does not know about.
            feasible_seen = best_seen < FEASIBLE_WEIGHT
            rel = float(opts.svgd_stop_rel) if feasible_seen else PATIENCE_REL
            ## (`best_seen` starts at +inf, and `inf - inf` is nan, which compares False:
            ## that is why the first wave's patience never fired, not the 1e-6 threshold.)
            thresh = best_seen - rel * max(1.0, abs(best_seen)) if math.isfinite(best_seen) else float("inf")
            if math.isfinite(best_m) and best_m < thresh:
                best_seen = best_m
                stale = 0
            else:
                stale += 1
            if best_seen < FEASIBLE_WEIGHT and stale >= int(opts.svgd_stop_patience):
                status = STATUS_CONVERGED
                stats["stop_reason"] = "feasible_stall"
                break
            if time.perf_counter() >= deadline:
                status = STATUS_WALL_CLOCK
                stats["stop_reason"] = "wall_clock"
                break
        return X, S, status, stats, last_ev

    ## ---------------------------------- polish ----------------------------------------
    def _polish(self, Xc, deadline):
        """Pure per-particle Gauss-Newton on `[h~ ; violated g~]` in float64, on the `k`
        candidates `Xc [k, n]`; returns `(Xc, ev, iters)`."""
        tg = self._tg64
        opts = self.options
        Xc = Xc.to(torch.float64)
        k = Xc.shape[0]
        iters = 0
        ev = tg.evaluate(Xc)
        ## Per-particle step fraction for a backtracking Newton: a rejected step is retried
        ## at half the length (down to POLISH_ALPHA_MIN), an accepted one lets it grow back.
        ## Without this a rejected step was recomputed identically and rejected again for
        ## every remaining iteration -- the polish was a no-op exactly when it was needed.
        alpha = torch.ones(k, dtype=torch.float64, device=self.device)
        for i in range(int(opts.svgd_polish_iters)):
            if bool((ev.infeas <= opts.svgd_polish_tol).all()):
                break
            if time.perf_counter() >= deadline:
                break
            J_q = tg.jacobian_q(ev)
            J_h, J_g = tg.row_jacobians(ev, J_q)
            act = (ev.g > 0.0).to(ev.g.dtype)
            J = torch.cat([J_h, J_g * act.unsqueeze(2)], dim=1).detach()
            r = torch.cat([ev.h, ev.g * act], dim=1).detach()
            J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            done = (ev.infeas <= opts.svgd_polish_tol).unsqueeze(1)
            dx = al.gn_correction(J, r, opts.svgd_gn_delta)
            dx = al.clamp_q_step(dx, J_q if tg.bp.is_learned else None, opts.svgd_q_step_max)
            dx = torch.where(done, torch.zeros_like(dx), dx * alpha.unsqueeze(1))
            Xn, _ = tg.project(Xc + dx)
            ev_n = tg.evaluate(Xn)
            ## accept where the step did not make things worse (a safeguarded Newton)
            better = (torch.nan_to_num(ev_n.infeas, nan=float("inf"))
                      <= torch.nan_to_num(ev.infeas, nan=float("inf")))
            Xc = torch.where(better.unsqueeze(1), Xn, Xc)
            alpha = torch.where(better, torch.clamp(2.0 * alpha, max=1.0),
                                torch.clamp(0.5 * alpha, min=POLISH_ALPHA_MIN))
            iters += 1
            ev = tg.evaluate(Xc)
        return Xc.detach(), ev, iters

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
        reserve = float(opts.svgd_time_reserve)
        swarm_deadline = t0 + wall * (1.0 - reserve)
        final_deadline = t0 + wall

        self._build()
        tg, tg64 = self._tg, self._tg64
        bp = tg.bp
        self._pool.drain()
        self._pool.reset()
        self._make_runner()
        counts0 = dict(getattr(program, "eval_counts", {}) or {})
        self._init_protocol = opts.svgd_paired_init

        x0_np = np.asarray(program.prog.GetInitialGuess(program.lumped_vars), dtype=float)
        X, clip, init_stats = self._init_particles(x0_np)
        N = X.shape[0]
        extras = dict(init_protocol=self._init_protocol, clip_distance=clip.detach().cpu().numpy().tolist(),
                      init_stats=init_stats, workers=self.workers,
                      step_mode=(self._runner.mode if self._runner is not None else
                                 ("eager (admm_svgd runs its own loop, never compiled)"
                                  if self.method == "admm_svgd" else "eager (autograd path)")),
                      step_reused_template=bool(self._runner is not None and self._runner.reused),
                      pool_overlap=self._overlap)
        phase["init"] = time.perf_counter() - t0

        ## -- AL state --
        rho0 = float(opts.svgd_rho0)
        self._eta0 = 1.0 / rho0 ** 0.1
        S = ALState.init(N, self.n, tg.m_e, tg.m_i, rho0=rho0, eta0=self._eta0,
                         lr0=float(opts.svgd_lr), dtype=self.dtype, device=self.device,
                         gn_lam0=float(opts.svgd_gn_lm))

        ## -- optional CEM warm-up --
        warmup_steps = 0
        if opts.svgd_warmup == "cem" and N > 1:
            from src.svgd.warmup import cem_warmup
            t = time.perf_counter()
            X, wstats = cem_warmup(tg, X, rho0, int(opts.svgd_warmup_iters),
                                   float(opts.svgd_warmup_elite), self._gen,
                                   deadline=swarm_deadline)
            warmup_steps = wstats.get("iters", 0)
            extras["warmup"] = wstats
            phase["warmup"] = time.perf_counter() - t

        def record(x_best):
            program.RecordIterate(x_best.detach().to(device="cpu", dtype=torch.float64).numpy())

        ## -- swarm --
        t = time.perf_counter()
        outer_iters = int(opts.max_iter) if opts.max_iter is not None else int(opts.svgd_outer_iters)
        inner_iters = max(1, int(opts.svgd_inner_iters))
        if self.method == "admm_svgd":
            from src.svgd.admm import AdmmSwarm
            X, S, stop, sstats, last_ev = AdmmSwarm(self).run(X, S, swarm_deadline, outer_iters, record)
        else:
            X, S, stop, sstats, last_ev = self._swarm(X, S, swarm_deadline, outer_iters, inner_iters, record)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        phase["swarm"] = time.perf_counter() - t

        ## -- candidates: each particle's best, plus its current point where better --
        t = time.perf_counter()
        Xb = torch.where(torch.isfinite(S.best_merit).unsqueeze(1), S.best_x, X)
        with torch.no_grad():
            evb = tg64.evaluate(Xb.to(torch.float64), need_grad=False)
        merit_b = tg64.merit(evb)
        all_nan = not bool(torch.isfinite(merit_b).any())
        viol_before = torch.nan_to_num(evb.infeas, nan=float("inf")).cpu().numpy()

        if all_nan:
            x_ret = x0_np.copy()
            status = STATUS_NAN
            drake_ok = False
            solver_ok = False
            selected = 0
            n_feasible = 0
            polish_iters = 0
            viol_after = viol_before
            drake_viol = None
            cands_idx = []
        else:
            k = max(1, min(int(opts.svgd_polish_topk), N))
            order = torch.argsort(torch.nan_to_num(merit_b, nan=float("inf")))
            cands_idx = order[:k]
            Xc, evc, polish_iters = self._polish(Xb[cands_idx], final_deadline)
            phase["polish"] = time.perf_counter() - t
            t = time.perf_counter()
            infeas_c = torch.nan_to_num(evc.infeas.detach(), nan=float("inf"))
            feas_c = infeas_c <= 1.0
            Fc = torch.nan_to_num(evc.F.detach(), nan=float("inf"))
            ## selection key: feasible first, then F; infeasible by infeasibility
            key = torch.where(feas_c, Fc, FEASIBLE_WEIGHT + infeas_c)
            sel_order = torch.argsort(key)
            viol_after = viol_before.copy()
            viol_after[cands_idx.cpu().numpy()] = infeas_c.cpu().numpy()
            n_feasible = int((torch.as_tensor(viol_after) <= 1.0).sum())
            ## exact Drake re-check, in F order, over the top recheck_topk
            x_ret, drake_ok, selected, drake_viol, solver_ok = None, False, -1, None, False
            best_v, best_j = np.inf, None
            Xc_cpu = Xc.detach().cpu().numpy()
            for j in sel_order[:max(1, int(opts.svgd_recheck_topk))].tolist():
                xf = bp.to_drake_x(Xc_cpu[j])
                v = self._drake_max_violation(xf)
                if v <= self.tol:
                    x_ret, drake_ok, selected, drake_viol = Xc_cpu[j].copy(), True, j, v
                    solver_ok = bool(feas_c[j])
                    break
                if v < best_v:
                    best_v, best_j = v, j
            if x_ret is None:
                j = best_j if best_j is not None else int(sel_order[0])
                x_ret, selected, drake_viol = Xc_cpu[j].copy(), j, best_v
                solver_ok = bool(feas_c[j])
            selected = int(cands_idx[selected].item())
            status = STATUS_CONVERGED if drake_ok else (stop if stop in (STATUS_WALL_CLOCK, STATUS_STEP_CAP) else STATUS_INFEASIBLE)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        phase["select"] = time.perf_counter() - t

        counts = dict(getattr(program, "eval_counts", {}) or {})
        map_evals = int(counts.get("map_forward", 0) - counts0.get("map_forward", 0))
        total = time.perf_counter() - t0
        ## The per-outer trace: stacked and moved to the host once, here, never per step.
        rows = sstats.get("trace", [])
        if rows:
            T_np = torch.stack(rows).detach().to(device="cpu", dtype=torch.float64).numpy()
            trace = {name: [float(v) for v in T_np[:, j]] for j, name in enumerate(_Target.TRACE_COLUMNS)}
        else:
            trace = {name: [] for name in _Target.TRACE_COLUMNS}
        extras.update(dict(
            trace=trace, derivative_mode="analytic" if tg.analytic else "autograd",
            violation_before_polish=[float(v) for v in np.asarray(viol_before)],
            violation_after_polish=[float(v) for v in np.asarray(viol_after)],
            polish_iters=int(polish_iters), drake_max_violation=None if drake_viol is None else float(drake_viol),
            best_history=sstats.get("best_history", []), stop_status=status_name(stop),
            total_steps=int(sstats.get("inner", 0)), eval_counts_delta={
                k: int(counts.get(k, 0) - counts0.get(k, 0)) for k in counts},
            pool_calls=int(self._pool.calls), pool_configs=int(self._pool.configs),
            pool_span_seconds=float(self._pool.span_seconds),
            admm_primal_residual=sstats.get("primal_residual"), admm_dual_residual=sstats.get("dual_residual"),
            admm=sstats.get("admm"),
            candidates=[int(i) for i in (cands_idx.cpu().numpy() if hasattr(cands_idx, "cpu") else cands_idx)]))
        details = SvgdSolverDetails(
            status=int(status), status_name=status_name(status), method=self.method,
            n_particles=N, dtype=opts.svgd_dtype,
            iterations=int(sstats.get("outer", 0)), inner_steps=int(sstats.get("inner", 0)),
            map_evals=map_evals, n_feasible=int(n_feasible), n_resampled=int(sstats.get("n_resampled", 0)),
            selected_index=int(selected), phase_times=phase,
            timed_out=(stop == STATUS_WALL_CLOCK), hit_iteration_cap=(stop == STATUS_STEP_CAP),
            solver_feasible=bool(solver_ok), drake_feasible=bool(drake_ok),
            solve_seconds=total, collision_seconds=float(self._pool.seconds),
            stop_reason=str(sstats.get("stop_reason", "")), extras=extras)
        details.extras["warmup_steps"] = int(warmup_steps)
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
        """Pay the pool spawn, the first flow / FK calls, the autograd machinery and -- under
        `svgd_compile` / `svgd_cuda_graph` -- the compile of every stage and the capture of
        every graph this solver's (N, dtype, method, structure) uses, outside any timed cell:
        a few steps of the configured method on particles drawn around the program's current
        guess (both `do_gn` variants where `svgd_gn_every > 1`), plus one float64 polish step.
        Returns the seconds spent; `self.warmup_info` says what was compiled / captured."""
        t0 = time.perf_counter()
        self._build()
        opts = self.options
        self._init_protocol = opts.svgd_paired_init
        x0 = np.asarray(self.program.prog.GetInitialGuess(self.program.lumped_vars), dtype=float)
        x0 = np.nan_to_num(x0)
        X, _, _ = self._init_particles(x0)
        N = X.shape[0]
        rho0 = float(opts.svgd_rho0)
        self._eta0 = 1.0 / rho0 ** 0.1
        S = ALState.init(N, self.n, self._tg.m_e, self._tg.m_i, rho0=rho0, eta0=self._eta0,
                         lr0=float(opts.svgd_lr), dtype=self.dtype, device=self.device,
                         gn_lam0=float(opts.svgd_gn_lm))
        self._pool.drain()
        self._make_runner()
        if self.method == "admm_svgd":
            from src.svgd.admm import AdmmSwarm
            AdmmSwarm(self).run(X, S, time.perf_counter() + 60.0, 2, lambda x: None)
        else:
            step_fn = self._step_split if self._runner is not None else (
                self._step_al if self.method == "al_svgd" else self._step_t)
            variants = (True, False) if int(opts.svgd_gn_every) > 1 else (True,)
            for do_gn in variants:
                for t in range(3):
                    X, S, _ = step_fn(X, S, 0, t, 3, do_gn)
        self._polish(X[:1], time.perf_counter() + 60.0)
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
