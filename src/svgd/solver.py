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
  admm_svgd: NOT IMPLEMENTED in this wave (raises at construction; see the class docstring).

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

`svgd_compile` and `svgd_cuda_graph` are accepted and are NO-OPS in this wave: the step is
written as a pure function with static shapes and no host syncs (except the pool round trip
inside `evaluate`) so it can be captured later, but nothing is compiled yet.
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

from src.svgd import al, kernels
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
    no-op: the pool belongs to the module cache, not to any BatchedProgram."""

    def __init__(self, pool):
        self.pool = pool
        self.seconds = 0.0
        self.calls = 0

    def eval(self, Q, need_grad=True):
        t = time.perf_counter()
        out = self.pool.eval(Q, need_grad=need_grad)
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return out

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
        h = out.h * self.sh
        g = out.g * self.sg
        finite = (torch.isfinite(h).all(dim=1) & torch.isfinite(g).all(dim=1)
                  & torch.isfinite(out.F) & torch.isfinite(out.q).all(dim=1))
        gplus = torch.clamp(g, min=0.0)
        stacked = torch.cat([h.abs(), gplus], dim=1)
        infeas = stacked.amax(dim=1) if stacked.shape[1] else torch.zeros_like(out.F)
        infeas = torch.where(finite, infeas, torch.full_like(infeas, float("inf"))).detach()
        return SimpleNamespace(Xg=Xg, out=out, h=h, g=g, F=out.F, cfg=out.q, finite=finite,
                               infeas=infeas)

    def merit(self, ev):
        """`F` where feasible at the gate, else `F + FEASIBLE_WEIGHT * infeas`."""
        excess = torch.where(ev.infeas <= 1.0, torch.zeros_like(ev.infeas), ev.infeas)
        return al.best_merit(ev.F.detach(), excess, FEASIBLE_WEIGHT)

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
        N = ev.Xg.shape[0]
        J_gen = self.generic_jacobian_cfg(ev) @ J_q                     # [N, nr, n]
        J_ex = self._extra_jacobian(ev.Xg, N)
        J_stack = torch.cat([J_gen, J_ex], dim=1)
        J_h = J_stack[:, self.h_idx] * self.h_coef.view(1, -1, 1)
        J_g = J_stack[:, self.g_idx] * self.g_coef.view(1, -1, 1)
        return J_h, J_g

    ## -- the per-outer-iteration trace ---------------------------------------------------
    TRACE_COLUMNS = ("min_infeas", "med_infeas", "n_feasible", "med_rho", "med_lr",
                     "frac_gn_clamped", "med_h_pos", "med_h_rot", "med_g_plus",
                     "min_F_feasible", "med_dq", "med_nw_infeas")

    def trace_row(self, ev, S, dq, clamped):
        """One `[len(TRACE_COLUMNS)]` tensor of swarm statistics at an outer boundary, all
        on the device (the caller stacks them and moves them to the host ONCE at the end):
        the best and median scaled violation, the feasible count, median rho and lr, the
        fraction of particles whose last GN step hit the q-step clamp (`nan` when none ran),
        the median over particles of the worst position / rotation equality row and of the
        worst violated inequality, the best objective among feasible particles, the median
        configuration step, and the median Nocedal-Wright infeasibility measure."""
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
            minF, dqm, al.infeasibility(ev.h.detach(), ev.g.detach(), S).median()])

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

    `admm_svgd` is NOT implemented in this wave: constructing a solver with it raises
    `NotImplementedError("admm_svgd: next wave")`. The Stein-projected consensus ADMM needs a
    q-space row evaluator (rows as functions of `q` alone, independent of the arm), which the
    batched program does not expose; `al_svgd` and `tsvgd` came first.
    """

    def __init__(self, program, particles_override=None):
        self.program = program
        self.options = program.options
        opts = self.options
        if opts.svgd_method == "admm_svgd":
            raise NotImplementedError("admm_svgd: next wave")
        self.method = opts.svgd_method
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
        """`(K, R, needs_pullback)`: the kernel matrix and the repulsion IN THE KERNEL'S
        SPACE (q for `svgd_kernel = "q"`, x otherwise); the caller pulls `R` back through
        `J_q^T` (one VJP, or the Jacobian when it has it) when `needs_pullback`. Non-finite
        particles are isolated so a NaN cannot spread through the kernel average (their K
        row/column is the identity's)."""
        opts = self.options
        N = X.shape[0]
        which = opts.svgd_kernel
        if which == "none" or N < 2:
            K, R = kernels.identity_terms(N, self.n, self.dtype, self.device)
            return K, R, False
        y = ev.cfg.detach() if which == "q" else X
        finite = ev.finite & torch.isfinite(y).all(dim=1)
        y0 = torch.where(finite.unsqueeze(1), y, torch.zeros_like(y))
        D = kernels.pairwise_sqdist(y0)
        if opts.svgd_bandwidth == "median":
            hbw = kernels.median_bandwidth(D, N, floor=opts.svgd_bandwidth_floor)
        else:
            hbw = torch.tensor(float(opts.svgd_bandwidth) ** 2, dtype=self.dtype, device=self.device)
        K, _ = kernels.rbf_terms(y0, hbw)
        mask = finite.unsqueeze(1) & finite.unsqueeze(0)
        eye = torch.eye(N, dtype=self.dtype, device=self.device)
        K = torch.where(mask, K, eye)
        ## the repulsion with the masked kernel (same formula as rbf_terms, sign included)
        R = (2.0 / hbw) * (K.sum(dim=1, keepdim=True) * y0 - K @ y0)
        R = torch.where(finite.unsqueeze(1), R, torch.zeros_like(R))
        return K, R, which == "q"

    def _gn_step(self, ev, S, J_h, J_g, rows_active):
        """The (unclamped) Gauss-Newton correction onto the equality rows and, when
        `rows_active`, the PHR-active inequality rows."""
        opts = self.options
        if rows_active:
            act = al.active_mask(ev.g, S.mu, S.rho)
            J = torch.cat([J_h, J_g * act.unsqueeze(2)], dim=1)
            r = torch.cat([ev.h, ev.g * act], dim=1).detach()
        else:
            J, r = J_h, ev.h.detach()
        J = torch.nan_to_num(J.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        return al.gn_correction(J, r, opts.svgd_gn_delta)

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
        ch, cg = al.al_constraint_grad_coefficients(ev.h, ev.g, S)
        K, R, pull = self._kernel_terms(ev, X)
        J_q = None
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
            dx, clamped = self._clamp(self._gn_step(ev, S, J_h, J_g, rows_active=True), J_q)
            Xn = Xn + dx
            ev.gn_clamped = clamped
        Xn, _ = tg.project(Xn)
        S = al.track_best(S, X, tg.merit(ev))
        return Xn.detach(), S, ev

    def _step_t(self, X, S, t_outer, t_step, total_steps, do_gn):
        tg = self._tg
        opts = self.options
        ev = tg.evaluate(X)
        J_q = tg.jacobian_q(ev)
        J_h, J_g = tg.row_jacobians(ev, J_q)
        dFc, dFx = tg.cost_parts(ev)
        gF = (J_q.transpose(1, 2) @ dFc.unsqueeze(2)).squeeze(2) + dFx
        J_h = torch.nan_to_num(J_h.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        J_g = torch.nan_to_num(J_g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        _, cg = al.al_constraint_grad_coefficients(ev.h, ev.g, S)
        act = al.active_mask(ev.g, S.mu, S.rho)
        J_A = torch.cat([J_h, J_g * act.unsqueeze(2)], dim=1)
        P = al.tangent_projector(J_A, opts.svgd_tangent_delta)
        gI = (J_g.transpose(1, 2) @ cg.unsqueeze(2)).squeeze(2)
        driving = -(gF + gI)
        fin = ev.finite.unsqueeze(1)
        driving = torch.where(fin, driving, torch.zeros_like(driving))
        driving = (P @ driving.unsqueeze(2)).squeeze(2)
        K, R, pull = self._kernel_terms(ev, X)
        R_pulled = (J_q.transpose(1, 2) @ R.unsqueeze(2)).squeeze(2) if pull else R
        R_pulled = torch.where(fin, R_pulled, torch.zeros_like(R_pulled))
        gamma = kernels.anneal_gamma(t_outer, opts.svgd_gamma_t)
        T = kernels.anneal_T(t_step, opts.svgd_repulsion_T0, opts.svgd_anneal_frac, total_steps)
        phi = kernels.svgd_direction(K, driving, R_pulled, gamma, T)
        phi = (P @ phi.unsqueeze(2)).squeeze(2)
        dx_c = self._gn_step(ev, S, J_h, J_g, rows_active=False)
        step_lr = S.lr.unsqueeze(1) * phi
        step, clamped = self._clamp(step_lr + dx_c, J_q)
        ## The lr schedule sees the Stein step's own configuration motion, not the manifold
        ## correction's (the same defect as al_svgd's: the correction alone saturates the
        ## clamp and drove lr to its floor on every step -- the trace of stage-0 tsvgd).
        if tg.bp.is_learned:
            ev.dq_adam = (J_q @ step_lr.unsqueeze(2)).squeeze(2).abs().amax(dim=1)
        else:
            ev.dq_adam = step_lr.abs().amax(dim=1)
        Xn, _ = tg.project(X + step)
        S = al.track_best(S, X, tg.merit(ev))
        self._last_P = P.detach()
        ev.gn_clamped = clamped
        return Xn.detach(), S, ev

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
                       lr=torch.where(mask, torch.full_like(S.lr, float(opts.svgd_lr)), S.lr))

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
        step_fn = self._step_al if self.method == "al_svgd" else self._step_t
        stats = dict(outer=0, inner=0, n_resampled=0, best_history=[], trace=[])
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
                dq_lr = getattr(ev, "dq_adam", None)
                if dq_lr is None:
                    dq_lr = dq
                S = al.lr_schedule(S, torch.nan_to_num(dq_lr, nan=float("inf")), opts.svgd_q_step_max,
                                   opts.svgd_lr, opts.svgd_lr_min, float(t_outer), opts.svgd_lr_decay_t)
                prev_q = q_now
                last_ev = ev
                t_step += 1
                stats["inner"] += 1
            S = al.update_multipliers(ev.h.detach(), ev.g.detach(), S, opts.svgd_rho_growth,
                                      opts.svgd_rho_max, opts.svgd_multiplier_max)
            stats["outer"] = t_outer + 1
            stats["trace"].append(tg.trace_row(ev, S, dq, getattr(ev, "gn_clamped", None)))
            ## -- outer boundary: sync, record, clock, patience --
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            best_i = int(torch.argmin(torch.nan_to_num(S.best_merit, nan=float("inf"))).item())
            best_m = float(S.best_merit[best_i].item())
            record(S.best_x[best_i])
            stats["best_history"].append(best_m)
            if math.isfinite(best_m) and best_m < best_seen - PATIENCE_REL * max(1.0, abs(best_seen)):
                best_seen = best_m
                stale = 0
            else:
                stale += 1
            if best_seen < FEASIBLE_WEIGHT and stale >= int(opts.svgd_stop_patience):
                status = STATUS_CONVERGED
                break
            if time.perf_counter() >= deadline:
                status = STATUS_WALL_CLOCK
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
        self._pool.seconds = 0.0
        self._pool.calls = 0
        counts0 = dict(getattr(program, "eval_counts", {}) or {})
        self._init_protocol = opts.svgd_paired_init

        x0_np = np.asarray(program.prog.GetInitialGuess(program.lumped_vars), dtype=float)
        X, clip, init_stats = self._init_particles(x0_np)
        N = X.shape[0]
        extras = dict(init_protocol=self._init_protocol, clip_distance=clip.detach().cpu().numpy().tolist(),
                      init_stats=init_stats, workers=self.workers)
        phase["init"] = time.perf_counter() - t0

        ## -- AL state --
        rho0 = float(opts.svgd_rho0)
        self._eta0 = 1.0 / rho0 ** 0.1
        S = ALState.init(N, self.n, tg.m_e, tg.m_i, rho0=rho0, eta0=self._eta0,
                         lr0=float(opts.svgd_lr), dtype=self.dtype, device=self.device)

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
            pool_calls=int(self._pool.calls),
            candidates=[int(i) for i in (cands_idx.cpu().numpy() if hasattr(cands_idx, "cpu") else cands_idx)]))
        details = SvgdSolverDetails(
            status=int(status), status_name=status_name(status), method=self.method,
            n_particles=N, dtype=opts.svgd_dtype,
            iterations=int(sstats.get("outer", 0)), inner_steps=int(sstats.get("inner", 0)),
            map_evals=map_evals, n_feasible=int(n_feasible), n_resampled=int(sstats.get("n_resampled", 0)),
            selected_index=int(selected), phase_times=phase,
            timed_out=(stop == STATUS_WALL_CLOCK), hit_iteration_cap=(stop == STATUS_STEP_CAP),
            solver_feasible=bool(solver_ok), drake_feasible=bool(drake_ok),
            solve_seconds=total, collision_seconds=float(self._pool.seconds), extras=extras)
        details.extras["warmup_steps"] = int(warmup_steps)
        write_log(opts.file_print_name, details)
        return SvgdResult(program, bp.to_drake_x(x_ret), details, success=bool(drake_ok))

    ## ---------------------------------- warm-up ---------------------------------------
    def warm_up(self):
        """Pay the pool spawn, the first flow / FK calls and the autograd machinery outside
        any timed cell: three steps of the configured method at the configured N and dtype on
        particles drawn around the program's current guess, plus one float64 polish step.
        Returns the seconds spent. (`svgd_compile` / `svgd_cuda_graph`: no-ops this wave.)"""
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
                         lr0=float(opts.svgd_lr), dtype=self.dtype, device=self.device)
        step_fn = self._step_al if self.method == "al_svgd" else self._step_t
        for t in range(3):
            X, S, _ = step_fn(X, S, 0, t, 3, True)
        self._polish(X[:1], time.perf_counter() + 60.0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if hasattr(self.program, "ResetEvalCounts"):
            self.program.ResetEvalCounts()
        return time.perf_counter() - t0
