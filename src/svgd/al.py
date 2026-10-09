"""Per-particle augmented Lagrangian state and updates for the particle solvers.

Pure tensor arithmetic on per-particle quantities. N particles, n decision variables, m_e
equality rows `h` (`h = 0`) and m_i inequality rows `g` (`g <= 0`). Rows arrive SCALED by the
caller (`h~ = h / (tol * s)`), so `||.||_inf <= 1` means feasible at the harness tolerance;
nothing here knows which row is which, where a row's centre is, or what formulation produced
it -- the same code runs the learned arm, joint space and the soft and screw robots.

The Lagrangian is the Powell-Hestenes-Rockafellar (PHR) form,

    L(x; lam, mu, rho) = F + lam.h + (rho/2) ||h||^2
                           + (1 / (2 rho)) sum_j [ max(0, mu_j + rho g_j)^2 - mu_j^2 ],

with the multiplier and penalty updates of Nocedal & Wright, Algorithm 17.4, per particle:
if the infeasibility `||[h ; max(g, -mu/rho)]||_inf` is within the particle's own tolerance
`eta`, the multipliers take a first-order step and the tolerance tightens; otherwise the
penalty grows and the tolerance is reset from it. The penalty parameter here is `rho`, the
reciprocal of NW's `mu`, hence `eta / rho^0.9` and `1 / rho^0.1` where they write
`eta mu^0.9` and `mu^0.1`.

Every function is written for `torch.compile(dynamic=False)` and CUDA-graph capture:

  * no `.item()`, no boolean-mask indexing, no data-dependent shape anywhere; every branch
    is a `torch.where` over the whole batch;
  * linear solves go through `torch.linalg.solve_ex(..., check_errors=False)`, with the
    `info` flag folded in by `torch.where`, because plain `solve` synchronises with the host
    to raise on a singular matrix and is therefore not capturable;
  * nothing raises from inside a step: a singular system yields a zero step, a non-finite
    merit is never taken as a particle's best;
  * explicit `dtype=` / `device=` on every tensor constructed, since `jrl.config` sets
    torch's global defaults at import and whichever module imports it first decides them.

The state is a plain dataclass of tensors treated as IMMUTABLE: every update returns a new
`ALState` (`dataclasses.replace`), so no in-place op ever touches a tensor autograd may still
hold, and a captured graph's static buffers are the caller's business, not this module's.
"""

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class ALState:
    """Per-particle AL multipliers, penalty, tolerance, Adam moments and the running best.

    All tensors share one `(dtype, device)`. `step` is Adam's step count as a float tensor
    `[N]` so that bias correction is a tensor op (one count per particle, though `adam_step`
    advances them together). `best_x` / `best_merit` are maintained by the caller through
    `track_best`, by whatever merit it defines.
    """

    lam: Tensor         # [N, m_e]  equality multipliers
    mu: Tensor          # [N, m_i]  inequality multipliers (>= 0)
    rho: Tensor         # [N]       penalty parameter
    eta: Tensor         # [N]       infeasibility tolerance (NW 17.4)
    adam_m: Tensor      # [N, n]    Adam first moment
    adam_v: Tensor      # [N, n]    Adam second moment
    step: Tensor        # [N]       Adam step count (float)
    lr: Tensor          # [N]       per-particle learning rate
    best_x: Tensor      # [N, n]    best iterate so far (by the caller's merit)
    best_merit: Tensor  # [N]       its merit (+inf until the first `track_best`)
    gn_lam: Tensor = None   # [N]   Levenberg-Marquardt damping of the GN correction (0: off)
    gn_r2: Tensor = None    # [N]   |r|^2 of the GN row set at the last step (nan: none yet)
    gn_pred2: Tensor = None  # [N]  |r + J d|^2, the linear prediction over the step taken
    gn_mask: Tensor = None  # [N, m_e + m_i]  which rows that set was (0/1)

    @staticmethod
    def init(N, n, m_e, m_i, rho0, eta0, lr0, dtype, device, gn_lam0=0.0) -> "ALState":
        """Zero multipliers and moments, `rho0`, `eta0`, `lr0` (and `gn_lam0`) on every
        particle, best unset, no LM prediction recorded yet."""
        kw = dict(dtype=dtype, device=device)
        return ALState(
            lam=torch.zeros(N, m_e, **kw),
            mu=torch.zeros(N, m_i, **kw),
            rho=torch.full((N,), float(rho0), **kw),
            eta=torch.full((N,), float(eta0), **kw),
            adam_m=torch.zeros(N, n, **kw),
            adam_v=torch.zeros(N, n, **kw),
            step=torch.zeros(N, **kw),
            lr=torch.full((N,), float(lr0), **kw),
            best_x=torch.zeros(N, n, **kw),
            best_merit=torch.full((N,), float("inf"), **kw),
            gn_lam=torch.full((N,), float(gn_lam0), **kw),
            gn_r2=torch.full((N,), float("nan"), **kw),
            gn_pred2=torch.full((N,), float("nan"), **kw),
            gn_mask=torch.zeros(N, m_e + m_i, **kw),
        )


## --------------------------------------------------------------------------------------
## The PHR augmented Lagrangian and its row derivatives
## --------------------------------------------------------------------------------------

def al_value(F: Tensor, h: Tensor, g: Tensor, S: ALState) -> Tensor:
    """`L = F + lam.h + (rho/2)||h||^2 + (1/(2 rho)) sum_j [max(0, mu_j + rho g_j)^2 - mu_j^2]`.

    `F [N]`, `h [N, m_e]`, `g [N, m_i]` -> `L [N]`. With `mu = 0` the inequality term is
    `(rho/2) ||max(0, g)||^2`, the plain quadratic penalty.
    """
    rho = S.rho.unsqueeze(1)
    eq = (S.lam * h).sum(1) + 0.5 * S.rho * (h * h).sum(1)
    shifted = torch.clamp(S.mu + rho * g, min=0.0)
    ineq = (shifted * shifted - S.mu * S.mu).sum(1) / (2.0 * S.rho)
    return F + eq + ineq


def al_constraint_grad_coefficients(h: Tensor, g: Tensor, S: ALState):
    """`dL/dh = lam + rho h` and `dL/dg = max(0, mu + rho g)`, shapes of `h` and `g`.

    These are the cotangents a caller hands to ONE vector-Jacobian product through the rows
    (`J_h^T ch + J_g^T cg`), so the AL gradient costs a single reverse pass however many
    rows there are. They are exactly `autograd.grad(al_value, (h, g))`.
    """
    rho = S.rho.unsqueeze(1)
    ch = S.lam + rho * h
    cg = torch.clamp(S.mu + rho * g, min=0.0)
    return ch, cg


def infeasibility(h: Tensor, g: Tensor, S: ALState) -> Tensor:
    """`||[h ; max(g, -mu/rho)]||_inf` per particle (Nocedal-Wright 17.4's measure) -> `[N]`.

    An inequality row deep in the feasible interior with a zero multiplier contributes
    nothing; one with a positive multiplier is measured against `-mu/rho`, the value at
    which the PHR term switches off, so a row the multiplier still "believes" active counts
    as violated until it is pushed back to `g = -mu/rho`.
    """
    N = h.shape[0]
    g_eff = torch.maximum(g, -S.mu / S.rho.unsqueeze(1))
    stacked = torch.cat([h, g_eff], dim=1)
    if stacked.shape[1] == 0:  # static: no rows at all
        return torch.zeros(N, dtype=h.dtype, device=h.device)
    return stacked.abs().amax(dim=1)


def update_multipliers(h: Tensor, g: Tensor, S: ALState, rho_growth, rho_max,
                       multiplier_max, eta_rel=0.0) -> ALState:
    """Nocedal-Wright Algorithm 17.4, per particle, by `torch.where`.

    Where `infeasibility <= eta`:  `lam += rho h`, `mu = max(0, mu + rho g)`,
                                   `eta = max(eta / rho^0.9, eta_rel * infeasibility)`
                                   (rho unchanged).
    Otherwise:                     `rho = min(rho_max, rho * rho_growth)`,
                                   `eta = max(1 / rho_new^0.1, eta_rel * infeasibility)`
                                   (multipliers unchanged).
    Multipliers are clipped to `|lam|, mu <= multiplier_max` so a particle whose rows never
    converge cannot carry its multipliers to infinity and poison every later gradient.

    `eta_rel` (0 = textbook NW) makes the tolerance RELATIVE to the particle's own current
    infeasibility on BOTH branches: the next test passes once it has shrunk by the factor
    `eta_rel`, so rho grows only where a particle's infeasibility stalls. (The passing
    branch's absolute `eta / rho^0.9` alone is a ~8x tightening per pass at rho = 10, which
    a far start's scaled rows cannot follow: one pass, then a fail and a rho growth on
    nearly every later outer step -- rho at its cap in six outer steps on the joint-space
    pose cell, measured 2026-10-08.)
    NW's absolute `rho^-0.1` (~0.6 at rho 1e2) is unreachable from a far start on rows
    scaled by `1/tol` (infeasibility 1e3..1e4), so every test failed, rho saturated at
    `rho_max` within a few outer steps and the multipliers never took a single first-order
    step -- a pure penalty method wearing an AL's clothes (the 2026-10-08 traces).
    """
    v = infeasibility(h, g, S)
    ok = v <= S.eta                                   # [N] bool
    ok_rows = ok.unsqueeze(1)
    rho = S.rho.unsqueeze(1)

    lam_up = torch.clamp(S.lam + rho * h, min=-multiplier_max, max=multiplier_max)
    mu_up = torch.clamp(S.mu + rho * g, min=0.0, max=multiplier_max)
    lam = torch.where(ok_rows, lam_up, S.lam)
    mu = torch.where(ok_rows, mu_up, S.mu)

    rho_grown = torch.clamp(S.rho * rho_growth, max=rho_max)
    rho_new = torch.where(ok, S.rho, rho_grown)
    v_rel = float(eta_rel) * torch.nan_to_num(v, nan=0.0, posinf=0.0)
    eta_reset = torch.maximum(rho_new.pow(-0.1), v_rel)
    eta_tight = torch.maximum(S.eta / S.rho.pow(0.9), v_rel)
    eta_new = torch.where(ok, eta_tight, eta_reset)
    return replace(S, lam=lam, mu=mu, rho=rho_new, eta=eta_new)


def seed_eta(h: Tensor, g: Tensor, S: ALState, eta_rel) -> ALState:
    """`eta = max(eta, eta_rel * infeasibility)` per particle: the relative tolerance of
    `update_multipliers` applied at the START, so the first outer test is "shrink the initial
    infeasibility by `eta_rel`" rather than NW's absolute `rho0^-0.1`. A no-op at
    `eta_rel = 0`."""
    if not eta_rel:
        return S
    v = torch.nan_to_num(infeasibility(h, g, S), nan=0.0, posinf=0.0)
    return replace(S, eta=torch.maximum(S.eta, float(eta_rel) * v))


def active_mask(g: Tensor, mu: Tensor, rho: Tensor) -> Tensor:
    """Inequality rows PHR treats as active, `g > -mu/rho`, as a float 0/1 mask `[N, m_i]`.

    Used to stack the active `g` rows under `h` for the tangent projector and the
    Gauss-Newton correction: multiply the row (and its Jacobian) by the mask rather than
    indexing with it, so the shapes stay static.
    """
    active = g > -mu / rho.unsqueeze(1)
    return active.to(dtype=g.dtype)


## --------------------------------------------------------------------------------------
## The inner optimizer: per-particle Adam with a per-particle learning rate
## --------------------------------------------------------------------------------------

def adam_step(x: Tensor, grad: Tensor, S: ALState, betas=(0.9, 0.999), eps=1e-8):
    """One bias-corrected Adam step on every particle, learning rate `S.lr` per particle.

    `x, grad [N, n]` -> `(x_new, S_new)`. Matches `torch.optim.Adam` (no weight decay, no
    amsgrad) step for step; the step count advances by one on every particle.
    """
    b1, b2 = betas
    step = S.step + 1.0
    m = b1 * S.adam_m + (1.0 - b1) * grad
    v = b2 * S.adam_v + (1.0 - b2) * grad * grad
    bc1 = 1.0 - torch.pow(b1, step).unsqueeze(1)
    bc2 = 1.0 - torch.pow(b2, step).unsqueeze(1)
    m_hat = m / bc1
    v_hat = v / bc2
    x_new = x - S.lr.unsqueeze(1) * m_hat / (torch.sqrt(v_hat) + eps)
    return x_new, replace(S, adam_m=m, adam_v=v, step=step)


def lr_schedule(S: ALState, dq_observed: Tensor, q_step_max, lr0, lr_min, t,
                lr_decay_t) -> ALState:
    """Per-particle learning rate: halve where the observed configuration step was too big,
    otherwise grow by 10 % up to the global decayed rate `lr0 / (1 + t / lr_decay_t)`.

    `dq_observed [N]` is `||q_i(t) - q_i(t-1)||_inf`, measured by the caller in CONFIGURATION
    space (through the network for the learned arm), so a particle whose latent is in a
    high-gain region of the chart slows down without knowing why. Floored at `lr_min`.
    """
    lr_global = lr0 / (1.0 + t / lr_decay_t)
    grown = torch.clamp(1.1 * S.lr, max=lr_global)
    lr = torch.where(dq_observed > q_step_max, 0.5 * S.lr, grown)
    lr = torch.clamp(lr, min=lr_min)
    return replace(S, lr=lr)


## --------------------------------------------------------------------------------------
## Gauss-Newton correction, step clamp and tangent projector
## --------------------------------------------------------------------------------------

def _solve_gram(J: Tensor, rhs: Tensor, delta, lam=None):
    """Solve `(J J^T + delta I + lam diag(J J^T)) Y = rhs` per particle without host syncs.

    `J [N, k, n]`, `rhs [N, k, *]`, `lam [N]` or None -> `(Y [N, k, *], ok [N] bool)`.
    `ok` is False where the factorisation failed; the caller zeroes that particle's result.
    Non-finite output is also flagged, so a `nan` can never leave this function marked as
    success. The `lam` term is Marquardt's scaling of the Levenberg damping -- relative to
    the Gram's own diagonal, so it means the same thing whatever the rows are scaled by.
    """
    k = J.shape[1]
    A = J @ J.transpose(1, 2)
    eye = torch.eye(k, dtype=J.dtype, device=J.device)
    A = A + delta * eye
    if lam is not None:
        A = A + torch.diag_embed(lam.view(-1, 1) * torch.diagonal(A, dim1=1, dim2=2))
    Y, info = torch.linalg.solve_ex(A, rhs, check_errors=False)
    finite = torch.isfinite(Y).flatten(1).all(dim=1)
    ok = (info == 0) & finite
    Y = torch.where(ok.view(-1, *([1] * (Y.dim() - 1))), Y, torch.zeros_like(Y))
    return Y, ok


def gn_correction(J: Tensor, r: Tensor, delta, lam=None) -> Tensor:
    """Gauss-Newton step onto `r = 0`: `dx = -J^T (J J^T + delta I)^-1 r`, `[N, n]`.

    With `lam [N]` the Gram carries Marquardt's `lam_i diag(J_i J_i^T)` as well, which turns
    the step continuously from Gauss-Newton (`lam = 0`) into a short scaled-gradient step
    (`lam` large) -- `lm_gain_update` drives it per particle.

    For a LINEAR row set one step lands exactly on `r = 0`; for a nonlinear one the iteration
    is Newton's on the constraint manifold and converges quadratically. Rows the caller does
    not want (inactive inequalities, masked rows) are to be ZEROED in both `J` and `r`, not
    removed -- the `delta I` keeps the Gram matrix invertible on those rows and a zero `r`
    there gives a zero contribution, so the step is exactly the one the surviving rows alone
    would produce. A singular system (delta = 0, rank-deficient `J`) yields a zero step
    rather than a `nan` or a raise.
    """
    Y, _ = _solve_gram(J, r.unsqueeze(2), delta, lam)
    return -(J.transpose(1, 2) @ Y).squeeze(2)


LM_RATIO_LOW = 0.25     # gain ratio below which the damping grows (Nielsen / More)
LM_RATIO_HIGH = 0.75    # ... and above which it shrinks


def lm_gain_update(S: ALState, h: Tensor, g: Tensor, growth, lam_min, lam_max):
    """Levenberg-Marquardt damping per particle from the GAIN RATIO of the last step:
    `ratio = (|r_prev|^2 - |r_now|^2) / (|r_prev|^2 - |r_prev + J d|^2)`, the actual over
    the linearly predicted reduction of the squared residual on the SAME row set (`gn_mask`)
    over the step `d` the particle actually took (Adam + Stein + GN correction, after the
    bound projection -- the linearisation is tested on what was done, not on the GN part
    alone). `ratio < LM_RATIO_LOW`: `gn_lam *= growth` (capped at `lam_max`); `ratio >
    LM_RATIO_HIGH`: `gn_lam /= growth` (floored at `lam_min`); otherwise unchanged. No
    information -- the first step, a redrawn particle (`gn_r2 = nan`), a step predicted not
    to reduce the residual (denominator <= 0), a non-finite row -- leaves `gn_lam` alone.

    `h`, `g` are the SCALED rows at the current point. Returns `(S, ratio [N])`, `ratio`
    nan where there was no information."""
    r = torch.cat([h, g], dim=1) * S.gn_mask
    actual2 = (r * r).sum(dim=1)
    den = S.gn_r2 - S.gn_pred2
    valid = torch.isfinite(actual2) & torch.isfinite(den) & (den > 0.0)
    ratio = (S.gn_r2 - actual2) / torch.where(valid, den, torch.ones_like(den))
    ratio = torch.where(valid, ratio, torch.full_like(ratio, float("nan")))
    grown = torch.clamp(S.gn_lam * float(growth), max=float(lam_max))
    shrunk = torch.clamp(S.gn_lam / float(growth), min=float(lam_min))
    lam = torch.where(valid & (ratio < LM_RATIO_LOW), grown,
                      torch.where(valid & (ratio > LM_RATIO_HIGH), shrunk, S.gn_lam))
    return replace(S, gn_lam=lam), ratio


def lm_record_prediction(S: ALState, J: Tensor, r: Tensor, mask: Tensor, d: Tensor) -> ALState:
    """Store what `lm_gain_update` tests at the next step: the GN row set's `|r|^2`, its
    linear prediction `|r + J d|^2` over the total step `d [N, n]`, and the row mask.
    `J [N, m, n]`, `r [N, m]` already masked (zeroed where `mask` is 0)."""
    pred = r + (J @ d.unsqueeze(2)).squeeze(2)
    return replace(S, gn_r2=(r * r).sum(dim=1), gn_pred2=(pred * pred).sum(dim=1), gn_mask=mask)


def clamp_q_step(dx: Tensor, Jq, q_step_max) -> Tensor:
    """Scale `dx_i` by `min(1, q_step_max / ||Jq_i dx_i||_inf)`.

    `Jq [N, ndof, n]` is the configuration Jacobian `dq/dx`; `None` means `dx` IS the
    configuration step (joint space), in which case the bound is applied to `dx` itself.
    A step that already respects the bound is returned unchanged, bit for bit.
    """
    if Jq is None:
        dq = dx
    else:
        dq = (Jq @ dx.unsqueeze(2)).squeeze(2)
    norm = dq.abs().amax(dim=1)
    safe = torch.clamp(norm, min=torch.finfo(dx.dtype).tiny)
    scale = torch.where(norm > q_step_max, q_step_max / safe, torch.ones_like(norm))
    return torch.where((norm > q_step_max).unsqueeze(1), dx * scale.unsqueeze(1), dx)


def tangent_projector(J: Tensor, delta) -> Tensor:
    """`P = I - J^T (J J^T + delta I)^-1 J`, `[N, n, n]`: the projector onto the tangent space
    of the (active) constraint manifold, exact at `delta = 0` on full-rank `J`."""
    n = J.shape[2]
    Y, _ = _solve_gram(J, J, delta)                     # (J J^T + dI)^-1 J   [N, k, n]
    eye = torch.eye(n, dtype=J.dtype, device=J.device)
    return eye - J.transpose(1, 2) @ Y


## --------------------------------------------------------------------------------------
## Resampling and the running best
## --------------------------------------------------------------------------------------

def resample_mask(q: Tensor, rows_finite: Tensor, q_max) -> Tensor:
    """Particles to redraw, `[N]` bool: `||q||_inf > q_max`, any non-finite `q`, or a row
    set the caller found non-finite (`rows_finite [N]` bool, True where all rows are finite).

    `q_max` is the gain-ceiling runaway threshold: a configuration of 1e7 rad is the chart
    reporting an astronomically unlikely draw, not a configuration to recover.
    """
    too_big = q.abs().amax(dim=1) > q_max
    nonfinite = ~torch.isfinite(q).all(dim=1)
    return too_big | nonfinite | ~rows_finite


def best_merit(F: Tensor, infeas: Tensor, feasible_weight) -> Tensor:
    """One merit a caller may use with `track_best`: `F + feasible_weight * infeasibility`.

    Kept outside `track_best` so the merit is the caller's choice, stated where it is used.
    """
    return F + feasible_weight * infeas


def track_best(S: ALState, x: Tensor, merit: Tensor) -> ALState:
    """Keep each particle's best `x` by the caller's `merit [N]` (lower is better).

    A non-finite merit never wins, so a particle that blew up cannot overwrite a finite
    best with garbage; the first finite merit always wins against the initial `+inf`.
    """
    better = (merit < S.best_merit) & torch.isfinite(merit)
    best_x = torch.where(better.unsqueeze(1), x, S.best_x)
    best_merit_new = torch.where(better, merit, S.best_merit)
    return replace(S, best_x=best_x, best_merit=best_merit_new)
