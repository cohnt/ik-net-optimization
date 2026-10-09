"""The augmented Lagrangian of the `svgd` solver: per-particle multipliers and penalties.

Pure tensor arithmetic on per-particle quantities. N particles, m_e equality rows `h` (`h = 0`)
and m_i inequality rows `g` (`g <= 0`). Rows arrive SCALED by the caller (`h~ = h / tol`), so
`v_i = ||[h~_i ; max(0, g~_i)]||_inf <= 1` means feasible at the harness gate; nothing here
knows which row is which or what formulation produced it.

The merit is the Powell-Hestenes-Rockafellar (PHR) form, per particle i,

    L_i = f_i + lam_i.h~_i + (rho_i / 2) ||h~_i||^2
              + (1 / (2 rho_i)) sum_j [ max(0, mu_ij + rho_i g~_ij)^2 - mu_ij^2 ].

EVERY QUANTITY OF THE AL IS PER PARTICLE: the multipliers `lam_i`, `mu_i`, the penalty `rho_i`
and the tolerance `eta_i`. A particle stuck in an infeasible local minimum must not drive up
the penalty, or shrink the step, of the particles still making progress.

At every outer check (`update`), for each particle i with violation `v_i`:

  * multipliers (Nocedal & Wright 17.4's first-order step): if `v_i <= eta_i`,
        lam_i <- lam_i + rho_i h~_i,   mu_i <- max(0, mu_i + rho_i g~_i),
        eta_i <- eta_i / rho_i^0.9;
    otherwise unchanged. Multipliers are clipped to `+-multiplier_max` and the clipped
    entries are counted.
  * penalty (Powell's test): if `v_i > gamma * v_i(previous check)`,
        rho_i <- min(beta * rho_i, rho_max);
    otherwise unchanged.

Written for `torch.compile` / CUDA-graph capture like the rest of the step: no `.item()`, no
boolean-mask indexing, every branch a `torch.where`, explicit dtype / device on every tensor
built (jrl sets torch's global defaults at import). The state is a dataclass of tensors
treated as IMMUTABLE: every update returns a new `ALState`.
"""

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class ALState:
    """Per-particle multipliers, penalty, tolerance and the violation at the last check."""

    lam: Tensor         # [N, m_e]  equality multipliers
    mu: Tensor          # [N, m_i]  inequality multipliers (>= 0)
    rho: Tensor         # [N]       penalty
    eta: Tensor         # [N]       multiplier-update tolerance
    v_prev: Tensor      # [N]       violation at the previous check (Powell's test)

    @staticmethod
    def init(N, m_e, m_i, rho0, eta0, dtype, device, v0=None) -> "ALState":
        """Zero multipliers, `rho0` and `eta0` on every particle; `v_prev = v0` (the
        initial swarm's violation) or +inf (no test can fail before a first check)."""
        kw = dict(dtype=dtype, device=device)
        v_prev = torch.full((N,), float("inf"), **kw) if v0 is None else v0.detach().clone()
        return ALState(lam=torch.zeros(N, m_e, **kw), mu=torch.zeros(N, m_i, **kw),
                       rho=torch.full((N,), float(rho0), **kw),
                       eta=torch.full((N,), float(eta0), **kw), v_prev=v_prev)


def al_value(F: Tensor, h: Tensor, g: Tensor, S: ALState) -> Tensor:
    """`L_i` above: `F [N]`, `h [N, m_e]`, `g [N, m_i]` -> `[N]`. With `mu = 0` the
    inequality term is `(rho/2) ||max(0, g)||^2`, the plain quadratic penalty."""
    rho = S.rho.unsqueeze(1)
    eq = (S.lam * h).sum(1) + 0.5 * S.rho * (h * h).sum(1)
    shifted = torch.clamp(S.mu + rho * g, min=0.0)
    ineq = (shifted * shifted - S.mu * S.mu).sum(1) / (2.0 * S.rho)
    return F + eq + ineq


def al_constraint_grad_coefficients(h: Tensor, g: Tensor, S: ALState):
    """`dL/dh~ = lam + rho h~` and `dL/dg~ = max(0, mu + rho g~)`, shapes of `h` and `g`:
    the cotangents of ONE vector-Jacobian product through the rows, so `grad (L - f)` costs a
    single reverse pass however many rows there are."""
    rho = S.rho.unsqueeze(1)
    return S.lam + rho * h, torch.clamp(S.mu + rho * g, min=0.0)


def violation(h: Tensor, g: Tensor) -> Tensor:
    """`v_i = ||[h~_i ; max(0, g~_i)]||_inf` -> `[N]` (0 with no rows at all)."""
    stacked = torch.cat([h.abs(), torch.clamp(g, min=0.0)], dim=1)
    if stacked.shape[1] == 0:  # static: no rows
        return torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
    return stacked.amax(dim=1)


def update(h: Tensor, g: Tensor, S: ALState, beta, gamma, rho_max, multiplier_max):
    """The outer check (module docstring), per particle. `h`, `g` are the scaled rows at the
    check. Returns `(S_new, updated [N] bool, n_clipped [] long)`: which particles took a
    multiplier step and how many multiplier entries the clip bound."""
    v = torch.nan_to_num(violation(h, g), nan=float("inf"))
    ok = v <= S.eta
    ok_rows = ok.unsqueeze(1)
    rho = S.rho.unsqueeze(1)
    lam_raw = S.lam + rho * h
    mu_raw = torch.clamp(S.mu + rho * g, min=0.0)
    lam_up = torch.clamp(lam_raw, min=-multiplier_max, max=multiplier_max)
    mu_up = torch.clamp(mu_raw, max=multiplier_max)
    clipped = ((lam_raw.abs() > multiplier_max) & ok_rows).sum() \
        + ((mu_raw > multiplier_max) & ok_rows).sum()
    lam = torch.where(ok_rows, lam_up, S.lam)
    mu = torch.where(ok_rows, mu_up, S.mu)
    eta = torch.where(ok, S.eta / S.rho.pow(0.9), S.eta)
    grow = v > gamma * S.v_prev
    rho_new = torch.where(grow, torch.clamp(S.rho * beta, max=rho_max), S.rho)
    return replace(S, lam=lam, mu=mu, rho=rho_new, eta=eta, v_prev=v), ok, clipped


def reset(S: ALState, mask: Tensor, rho0, eta0) -> ALState:
    """Fresh AL state on the masked particles (a redrawn particle): zero multipliers, `rho0`,
    `eta0`, and no previous violation (its first Powell test cannot fail)."""
    m1 = mask.unsqueeze(1)
    return replace(S, lam=torch.where(m1, torch.zeros_like(S.lam), S.lam),
                   mu=torch.where(m1, torch.zeros_like(S.mu), S.mu),
                   rho=torch.where(mask, torch.full_like(S.rho, float(rho0)), S.rho),
                   eta=torch.where(mask, torch.full_like(S.eta, float(eta0)), S.eta),
                   v_prev=torch.where(mask, torch.full_like(S.v_prev, float("inf")), S.v_prev))


def resample_mask(q: Tensor, rows_finite: Tensor, q_max) -> Tensor:
    """Particles to redraw, `[N]` bool: `||q||_inf > q_max`, any non-finite `q`, or a row
    set the caller found non-finite (`rows_finite [N]`, True where every row is finite)."""
    too_big = q.abs().amax(dim=1) > q_max
    nonfinite = ~torch.isfinite(q).all(dim=1)
    return too_big | nonfinite | ~rows_finite
