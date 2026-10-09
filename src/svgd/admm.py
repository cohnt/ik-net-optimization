"""`admm_svgd`: Stein-projected consensus ADMM on `q_bar = f(x)`, per particle.

The program's decision variables `x` reach every constraint row through the configuration
`q = f(x)` (the flow plus the correction on the learned arm, the identity on joint space),
except for the EXTRA rows (the z / c boxes and the latent trust region) and the direct
costs `R(x)` (the correction and latent penalties), which act on `x` alone. That split is
STRUCTURAL -- which variables a row or a cost is a function of, read off the batched
program's inventory (`evaluate_extra`, `cost_x_parts`, `cost_cfg_parts`) -- and is the only
thing this module knows about a formulation. Consensus ADMM introduces a copy `q_bar` of
the configuration and alternates, with scaled dual `u` and penalty `rho` per particle:

  x-block   x <- argmin_x  R(x) + (rho/2) ||f(x) - q_bar + u||^2 + (rho/2) ||[h_ex(x); g_ex^+(x)]||^2
            within the TRUE variable bounds: `svgd_admm_x_iters` damped Gauss-Newton steps
            on that objective (Hessian `H_R + rho J^T J + rho J_a^T J_a + delta I`, `J = df/dx`),
            each clamped to `|J dx|_inf <= svgd_q_step_max` AND `|dx|_inf <= svgd_q_step_max`
            (the second bounds the null space of `J`, where the system is nearly singular)
            and projected onto the bounds.
            The extra rows enter as a quadratic penalty (they are scaled by 1/tol, so a
            violation is expensive); a projection onto them is not available in general.
  q-block   q_bar <- proj( f(x) + u + phi_q(q_bar) ),   phi_q = (1/Z)[ K (-grad F_q / rho) + gamma T(t) R ]
            the proximal-gradient point on the configuration-space costs `F_q` (joint
            centering), Stein-smoothed over the swarm with a q-space RBF kernel and pushed
            apart by the repulsion `R` (temperature `T(t)` annealed to zero, gain
            `svgd_admm_gamma`); `proj` is `svgd_admm_q_iters` BACKTRACKING Gauss-Newton
            steps onto the generic rows at q_bar -- `[h(q); g_violated(q)]` through
            `BatchedProgram.evaluate_cfg` (FK, the exact collision pool, the joint-limit
            rows) and `generic_rows_jacobian_cfg` -- then a clip to the configuration box.
  dual      u <- u + f(x) - q_bar
  rho       residual balancing per particle (Boyd et al. 2011, section 3.4.1):
            `r_p = ||f(x) - q_bar||`, `r_d = rho ||q_bar - q_bar_prev||`; `rho *= tau` and
            `u /= tau` where `r_p > mu r_d`, `rho /= tau` and `u *= tau` where `r_d > mu r_p`,
            `mu = 10`, `tau = 2`, bounded to `[ADMM_RHO_MIN, ADMM_RHO_MAX]`.

On the joint-space arm `f = I`, `R = 0` and there are no extra rows, so the x-block is the
SAME CODE with `J = I`, `H_R = 0`: its Gauss-Newton system reads `(rho + delta) dx =
-rho (x - (q_bar - u))`, i.e. `x <- Pi_box(q_bar - u)` to within `delta / rho` in one step --
the degenerate projection split, reported as such. Nothing short-circuits it.

The swarm is driven from `SvgdSolver` (`run`), which supplies the shared `_Target`, the
particle init, the clock, `RecordIterate`, the stall rule, the trace format and -- after
this returns -- the shared float64 polish, selection and exact Drake re-check. The running
best per particle is tracked by the solver's own merit (`_Target.merit`) on a FULL
evaluation of `x` at every outer boundary, so the particle handed on is scored by the
program as written, not by the split. There is no convergence theorem for any of this:
nonconvex consensus ADMM with inexact blocks and a Stein term is a HEURISTIC, fielded to
be measured against `al_svgd` on the same cells.

Compile-friendly like the rest: static shapes, `torch.where` branches, `solve_ex`, no
`.item()` inside a round; the host is touched only at the outer boundary, where the solver
touches it anyway. `ALState` is reused as the per-particle state carrier (`rho` holds the
ADMM penalty, so the trace's `med_rho` column reports it; `lam`, `mu`, `eta`, the Adam
moments and `gn_lam` are unused).
"""

import math
import time
import torch

from src.svgd import al, kernels
from src.svgd.result import STATUS_CONVERGED, STATUS_STEP_CAP, STATUS_WALL_CLOCK

ADMM_MU = 10.0          # residual-balancing ratio (Boyd 3.4.1)
ADMM_TAU = 2.0          # residual-balancing factor
ADMM_RHO_MIN = 1e-2
ADMM_RHO_MAX = 1e6
PROJ_ALPHA_MIN = 1.0 / 16


class AdmmSwarm:
    """One ADMM run on a solver's swarm; see the module docstring."""

    def __init__(self, solver):
        self.solver = solver
        self.tg = solver._tg
        self.bp = self.tg.bp
        self.opts = solver.options
        self.dtype, self.device = self.tg.dtype, self.tg.device
        self.n, self.ndof = self.tg.n, self.tg.ndof
        bp = self.bp
        self._lo_q, self._hi_q = self._config_box()
        ## scaling of the generic rows (the q-space rows), in `evaluate_cfg` order
        self.sh_gen = self.tg.sh[bp.h_generic_mask]
        self.sg_gen = self.tg.sg[bp.g_generic_mask]
        self.sh_ex = self.tg.sh[~bp.h_generic_mask]
        self.sg_ex = self.tg.sg[~bp.g_generic_mask]
        ## the generic rows' positions in the stacked [drake_rows] Jacobian, signed
        self.h_idx_gen = self.tg.h_idx[bp.h_generic_mask]
        self.g_idx_gen = self.tg.g_idx[bp.g_generic_mask]
        self.h_coef_gen = self.tg.h_coef[bp.h_generic_mask]
        self.g_coef_gen = self.tg.g_coef[bp.g_generic_mask]
        self._E_q = self.tg._E_q
        self._eye_n = torch.eye(self.n, dtype=self.dtype, device=self.device)

    def _config_box(self):
        """The configuration's own limits as a box (the plant's `ConfigLimits`), for the
        q-block's clip; +-inf where the program reports none."""
        try:
            lower, upper = self.bp.program.ConfigLimits()
            lo = torch.tensor([float(v) for v in list(lower)[:self.ndof]], dtype=self.dtype, device=self.device)
            hi = torch.tensor([float(v) for v in list(upper)[:self.ndof]], dtype=self.dtype, device=self.device)
        except Exception:
            lo = torch.full((self.ndof,), -float("inf"), dtype=self.dtype, device=self.device)
            hi = torch.full((self.ndof,), float("inf"), dtype=self.dtype, device=self.device)
        return lo, hi

    ## ---------------------------------- the map ---------------------------------------
    def _map(self, X, jacobian):
        """`(f(X), df/dX or None)` -- the flow alone, no kinematics. Counted as a map
        forward (and a Jacobian) like the solver's own evaluations."""
        bp, tg = self.bp, self.tg
        N = X.shape[0]
        if not bp.is_learned:
            cfg = X[:, :self.ndof].detach().clone()
            J = bp.jacobian_q(X) if jacobian else None
            tg._count("map_forward")
            return cfg, J
        if not jacobian:
            with torch.no_grad():
                cfg = bp._config(X.detach())
            tg._count("map_forward")
            return cfg, None
        Xg = X.detach().requires_grad_(True)
        with torch.enable_grad():
            cfg = bp._config(Xg)
            tg._count("map_forward")
            J = tg._batched_grad(cfg, Xg, self._E_q, N)
        tg._count("map_jacobian")
        return cfg.detach(), J.detach()

    ## --------------------------------- x-block ----------------------------------------
    def x_block(self, X, v, rho):
        """`svgd_admm_x_iters` damped GN steps on `R(x) + (rho/2)|f(x) - v|^2 + (rho/2)
        |[h_ex; g_ex^+]|^2` within the variable bounds. Returns `(X, f(X), n_clamped)`."""
        bp, tg, opts = self.bp, self.tg, self.opts
        N = X.shape[0]
        delta = float(opts.svgd_gn_delta)
        clamped_any = torch.zeros(N, dtype=torch.bool, device=self.device)
        cfg = None
        for _ in range(max(1, int(opts.svgd_admm_x_iters))):
            cfg, J = self._map(X, jacobian=True)
            J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            res = torch.nan_to_num(cfg - v, nan=0.0, posinf=0.0, neginf=0.0)
            R, gR, HR = bp.cost_x_parts(X)
            h_ex, g_ex, Jh_ex, Jg_ex = bp.evaluate_extra(X)
            h_ex = h_ex * self.sh_ex
            g_ex = g_ex * self.sg_ex
            Jh_ex = Jh_ex * self.sh_ex.view(1, -1, 1)
            Jg_ex = Jg_ex * self.sg_ex.view(1, -1, 1)
            act = (g_ex > 0.0).to(self.dtype)
            J_a = torch.cat([Jh_ex, Jg_ex * act.unsqueeze(2)], dim=1)
            r_a = torch.cat([h_ex, g_ex * act], dim=1)
            J_a = torch.nan_to_num(J_a, nan=0.0, posinf=0.0, neginf=0.0)
            r_a = torch.nan_to_num(r_a, nan=0.0, posinf=0.0, neginf=0.0)
            rho3 = rho.view(-1, 1, 1)
            H = (HR.unsqueeze(0) + rho3 * (J.transpose(1, 2) @ J) + rho3 * (J_a.transpose(1, 2) @ J_a)
                 + delta * self._eye_n)
            grad = (gR + rho.unsqueeze(1) * (J.transpose(1, 2) @ res.unsqueeze(2)).squeeze(2)
                    + rho.unsqueeze(1) * (J_a.transpose(1, 2) @ r_a.unsqueeze(2)).squeeze(2))
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            ## Solved in float64 whatever the swarm's dtype: the region rows are scaled by
            ## 1/tol, so rho J_a^T J_a reaches ~1e10 beside a `delta` of 1e-6, a condition
            ## number float32 cannot represent (a 20 x 20 solve per particle; cheap).
            dx, info = torch.linalg.solve_ex(H.double(), -grad.double().unsqueeze(2),
                                             check_errors=False)
            dx = dx.squeeze(2).to(self.dtype)
            ok = (info == 0) & torch.isfinite(dx).all(dim=1)
            dx = torch.where(ok.unsqueeze(1), dx, torch.zeros_like(dx))
            Jq = J if bp.is_learned else None
            dq = dx if Jq is None else (Jq @ dx.unsqueeze(2)).squeeze(2)
            clamped_any |= (dq.abs().amax(dim=1) > float(opts.svgd_q_step_max)) | \
                (dx.abs().amax(dim=1) > float(opts.svgd_q_step_max))
            ## Two clamps. In configuration (`|J dx|_inf`, the solver's own unit) and on `dx`
            ## itself: the first alone leaves the null space of `J` unbounded, and the
            ## x-block's system is nearly singular there (only `delta` and the active
            ## region rows act on it), so a violated trust-region row with consensus held
            ## produced |dx| ~ 600 and moved f(x) by 4e7 rad in ONE step (2026-10-08).
            dx = al.clamp_q_step(dx, Jq, opts.svgd_q_step_max)
            dx = al.clamp_q_step(dx, None, opts.svgd_q_step_max)
            X, _ = tg.project(X + dx)
        cfg, _ = self._map(X, jacobian=False)
        return X.detach(), cfg, clamped_any

    ## --------------------------------- q-block ----------------------------------------
    def _rows_q(self, Q):
        """Scaled generic rows and Jacobians at configurations `Q`: `(h, g, J_h, J_g,
        infeas, ev)`."""
        bp = self.bp
        with torch.no_grad():
            ev = bp.evaluate_cfg(Q, row_jacobians=True)
        h = ev.h * self.sh_gen
        g = ev.g * self.sg_gen
        D = torch.nan_to_num(bp.generic_rows_jacobian_cfg(ev), nan=0.0, posinf=0.0, neginf=0.0)
        J_h = D[:, self.h_idx_gen] * self.h_coef_gen.view(1, -1, 1)
        J_g = D[:, self.g_idx_gen] * self.g_coef_gen.view(1, -1, 1)
        finite = torch.isfinite(h).all(dim=1) & torch.isfinite(g).all(dim=1)
        stacked = torch.cat([h.abs(), torch.clamp(g, min=0.0)], dim=1)
        infeas = stacked.amax(dim=1) if stacked.shape[1] else torch.zeros(Q.shape[0], dtype=self.dtype, device=self.device)
        infeas = torch.where(finite, infeas, torch.full_like(infeas, float("inf")))
        return h, g, J_h, J_g, infeas, ev

    def project_q(self, Q):
        """`svgd_admm_q_iters` backtracking GN steps onto `[h(q); g_violated(q)]` from `Q`,
        then the configuration box. Returns `(Q, infeasibility, n_clamped, pool_rounds)`."""
        opts = self.opts
        N = Q.shape[0]
        Q = torch.minimum(torch.maximum(Q, self._lo_q), self._hi_q)
        h, g, J_h, J_g, infeas, _ = self._rows_q(Q)
        alpha = torch.ones(N, dtype=self.dtype, device=self.device)
        clamped_any = torch.zeros(N, dtype=torch.bool, device=self.device)
        rounds = 1
        for _ in range(max(0, int(opts.svgd_admm_q_iters))):
            act = (g > 0.0).to(self.dtype)
            J = torch.cat([J_h, J_g * act.unsqueeze(2)], dim=1)
            r = torch.cat([h, g * act], dim=1)
            J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            dq = al.gn_correction(J, r, opts.svgd_gn_delta)
            clamped_any |= dq.abs().amax(dim=1) > float(opts.svgd_q_step_max)
            dq = al.clamp_q_step(dq, None, opts.svgd_q_step_max) * alpha.unsqueeze(1)
            Qn = torch.minimum(torch.maximum(Q + dq, self._lo_q), self._hi_q)
            hn, gn, J_hn, J_gn, infeas_n, _ = self._rows_q(Qn)
            rounds += 1
            better = torch.nan_to_num(infeas_n, nan=float("inf")) <= torch.nan_to_num(infeas, nan=float("inf"))
            b1 = better.unsqueeze(1)
            b2 = better.view(-1, 1, 1)
            Q = torch.where(b1, Qn, Q)
            h = torch.where(b1, hn, h)
            g = torch.where(b1, gn, g)
            J_h = torch.where(b2, J_hn, J_h)
            J_g = torch.where(b2, J_gn, J_g)
            infeas = torch.where(better, infeas_n, infeas)
            alpha = torch.where(better, torch.clamp(2.0 * alpha, max=1.0),
                                torch.clamp(0.5 * alpha, min=PROJ_ALPHA_MIN))
        return Q, infeas, clamped_any, rounds

    def q_block(self, fx, u, Qbar, rho, T):
        """The Stein-smoothed proximal point, then its projection."""
        opts = self.opts
        N = Qbar.shape[0]
        _, _, gF = self.bp.cost_cfg_parts(Qbar)
        driving = -torch.nan_to_num(gF, nan=0.0, posinf=0.0, neginf=0.0) / rho.unsqueeze(1)
        if opts.svgd_kernel != "none" and N > 1:
            finite = torch.isfinite(Qbar).all(dim=1)
            y0 = torch.where(finite.unsqueeze(1), Qbar, torch.zeros_like(Qbar))
            D = kernels.pairwise_sqdist(y0)
            if opts.svgd_bandwidth == "median":
                hbw = kernels.median_bandwidth(D, N, floor=opts.svgd_bandwidth_floor)
            else:
                hbw = torch.tensor(float(opts.svgd_bandwidth) ** 2, dtype=self.dtype, device=self.device)
            K, _ = kernels.rbf_terms(y0, hbw)
            mask = finite.unsqueeze(1) & finite.unsqueeze(0)
            K = torch.where(mask, K, torch.eye(N, dtype=self.dtype, device=self.device))
            R = (2.0 / hbw) * (K.sum(dim=1, keepdim=True) * y0 - K @ y0)
            R = torch.where(finite.unsqueeze(1), R, torch.zeros_like(R))
        else:
            K, R = kernels.identity_terms(N, self.ndof, self.dtype, self.device)
        phi = kernels.svgd_direction(K, driving, R, 1.0, float(opts.svgd_admm_gamma) * T)
        target = fx + u + phi
        target = torch.where(torch.isfinite(target), target, Qbar)
        return self.project_q(target)

    ## ----------------------------------- run ------------------------------------------
    def run(self, X, S, deadline, outer_iters, record):
        """The outer loop; the same contract as `SvgdSolver._swarm`:
        `(X, S, stop_status, stats, last_ev)`."""
        from src.svgd import solver as solver_mod       # lazy: solver imports this module
        solver, tg, opts = self.solver, self.tg, self.opts
        N = X.shape[0]
        kw = dict(dtype=self.dtype, device=self.device)
        rho = torch.full((N,), float(opts.svgd_admm_rho), **kw)
        S = al.replace(S, rho=rho)
        fx, _ = self._map(X, jacobian=False)
        Qbar = torch.nan_to_num(fx, nan=0.0, posinf=0.0, neginf=0.0).clone()
        u = torch.zeros(N, self.ndof, **kw)
        degenerate = not self.bp.is_learned
        stats = dict(outer=0, inner=0, n_resampled=0, best_history=[], trace=[], pool_rounds=0,
                     primal_residual=[], dual_residual=[], stop_reason="step_cap",
                     admm=dict(
                         degenerate=degenerate,
                         split=("projection split: f = I, so the x-block is x <- Pi_box(q_bar - u) "
                                "to within svgd_gn_delta / rho in one GN step; ADMM here is an "
                                "alternating projection of the configuration onto the constraint "
                                "manifold with a consensus dual, not a split across a map"
                                if degenerate else
                                "consensus split across the flow: x-block GN on f(x) ~ q_bar - u, "
                                "q-block projection of q_bar onto the generic rows"),
                         x_iters=int(opts.svgd_admm_x_iters), q_iters=int(opts.svgd_admm_q_iters),
                         rho0=float(opts.svgd_admm_rho)))
        total_rounds = max(1, int(outer_iters))
        best_seen = float("inf")
        stale = 0
        status = STATUS_STEP_CAP
        last_ev = None
        prev_q = None
        for t in range(total_rounds):
            ## -- resample (runaway / non-finite), at the outer boundary --
            if t % max(1, int(opts.svgd_resample_every)) == 0:
                with torch.no_grad():
                    ev0 = tg.evaluate(X, need_grad=False)
                X, S, n_re = solver._resample(X, S, ev0.cfg, ev0.finite, tg.merit(ev0))
                stats["n_resampled"] += n_re
                if n_re:
                    fx, _ = self._map(X, jacobian=False)
                    redrawn = ~torch.isfinite(Qbar).all(dim=1) | (ev0.cfg.abs().amax(dim=1) > float(opts.svgd_resample_q_max))
                    Qbar = torch.where(redrawn.unsqueeze(1), torch.nan_to_num(fx, nan=0.0, posinf=0.0, neginf=0.0), Qbar)
                    u = torch.where(redrawn.unsqueeze(1), torch.zeros_like(u), u)
                    S = al.replace(S, rho=torch.where(redrawn, torch.full_like(S.rho, float(opts.svgd_admm_rho)), S.rho))
            rho = S.rho
            T = kernels.anneal_T(t, opts.svgd_repulsion_T0, opts.svgd_anneal_frac, total_rounds)
            ## -- the three blocks --
            X, fx, clamped_x = self.x_block(X, Qbar - u, rho)
            Qprev = Qbar
            Qbar, infeas_q, clamped_q, rounds = self.q_block(fx, u, Qbar, rho, T)
            stats["pool_rounds"] += rounds
            fxs = torch.nan_to_num(fx, nan=0.0, posinf=0.0, neginf=0.0)
            u = u + fxs - Qbar
            ## -- residual balancing --
            r_p = (fxs - Qbar).norm(dim=1)
            r_d = rho * (Qbar - Qprev).norm(dim=1)
            up = r_p > ADMM_MU * r_d
            down = r_d > ADMM_MU * r_p
            rho_new = torch.where(up, torch.clamp(rho * ADMM_TAU, max=ADMM_RHO_MAX),
                                  torch.where(down, torch.clamp(rho / ADMM_TAU, min=ADMM_RHO_MIN), rho))
            u = u * (rho / rho_new).unsqueeze(1)
            S = al.replace(S, rho=rho_new)
            stats["inner"] += int(opts.svgd_admm_x_iters) + int(opts.svgd_admm_q_iters)
            ## -- the full target at x: best tracking, trace, stall --
            with torch.no_grad():
                ev = tg.evaluate(X, need_grad=False)
            S = al.track_best(S, X, tg.merit(ev))
            q_now = ev.cfg.detach()
            dq = (torch.nan_to_num((q_now - prev_q).abs().amax(dim=1), nan=float("inf"))
                  if prev_q is not None else torch.zeros(N, **kw))
            prev_q = q_now
            last_ev = ev
            ev.gn_clamped = clamped_q | clamped_x
            stats["outer"] = t + 1
            stats["trace"].append(tg.trace_row(ev, S, dq, ev.gn_clamped))
            stats["primal_residual"].append(r_p)
            stats["dual_residual"].append(r_d)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            best_i = int(torch.argmin(torch.nan_to_num(S.best_merit, nan=float("inf"))).item())
            best_m = float(S.best_merit[best_i].item())
            record(S.best_x[best_i])
            stats["best_history"].append(best_m)
            feasible_seen = best_seen < solver_mod.FEASIBLE_WEIGHT
            rel = float(opts.svgd_stop_rel) if feasible_seen else solver_mod.PATIENCE_REL
            ## (`best_seen` starts at +inf, and `inf - inf` is nan, which compares False:
            ## that is why the first wave's patience never fired, not the 1e-6 threshold.)
            thresh = best_seen - rel * max(1.0, abs(best_seen)) if math.isfinite(best_seen) else float("inf")
            if math.isfinite(best_m) and best_m < thresh:
                best_seen = best_m
                stale = 0
            else:
                stale += 1
            if best_seen < solver_mod.FEASIBLE_WEIGHT and stale >= int(opts.svgd_stop_patience):
                status = STATUS_CONVERGED
                stats["stop_reason"] = "feasible_stall"
                break
            if time.perf_counter() >= deadline:
                status = STATUS_WALL_CLOCK
                stats["stop_reason"] = "wall_clock"
                break
        stats["admm"]["pool_rounds"] = int(stats["pool_rounds"])
        for k in ("primal_residual", "dual_residual"):
            rows = stats[k]
            stats[k] = ([float(v) for v in torch.stack(rows).median(dim=1).values.detach().cpu().numpy()]
                        if rows else [])
        return X, S, status, stats, last_ev
