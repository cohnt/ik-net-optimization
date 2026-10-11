"""The augmented Lagrangian of the `svgd` solver: the formulation the swarm samples.

Pure tensor arithmetic on per-particle quantities. N particles, m_e equality rows `h` (`h = 0`)
and m_i inequality rows `g` (`g <= 0`). Rows arrive SCALED by the caller (`h~ = h / tol`), so
`v_i = ||[h~_i ; max(0, g~_i)]||_inf <= 1` means feasible at the harness gate; nothing here
knows which row is which or what formulation produced it.

THE AUGMENTED LAGRANGIAN IS THE FORMULATION, SVGD THE OPTIMIZER (Thomas, 2026-10-09): one
dynamics, no inner/outer structure. The merit is the Powell-Hestenes-Rockafellar (PHR) form
with ONE penalty `rho`, fixed for the whole solve and shared by every particle,

    L_i = f_i + lam_i.h~_i + (rho / 2) ||h~_i||^2
              + (1 / (2 rho)) sum_j [ max(0, mu_ij + rho g~_ij)^2 - mu_ij^2 ],

and the multipliers are per particle. Every K steps the caller takes a dual-ascent step with
its OWN step `alpha` (`svgd_dual_lr`), UNCONDITIONALLY, on every particle (`dual_update`):

    lam_i <- lam_i + alpha h~_i,    mu_i <- max(0, mu_i + alpha g~_i),

`alpha = 0` leaving the multipliers at zero (a pure quadratic penalty). The textbook step
`alpha = rho` is exact only when the primal is minimised between updates, which it is not here
(gradient descent-ascent), so the dual rate is a separate parameter.

Clipped to `+-multiplier_max`, the clipped entries counted. No feasibility gate of any kind.

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
    """Per-particle multipliers."""

    lam: Tensor         # [N, m_e]  equality multipliers
    mu: Tensor          # [N, m_i]  inequality multipliers (>= 0)

    @staticmethod
    def init(N, m_e, m_i, dtype, device) -> "ALState":
        """Zero multipliers on every particle."""
        kw = dict(dtype=dtype, device=device)
        return ALState(lam=torch.zeros(N, m_e, **kw), mu=torch.zeros(N, m_i, **kw))


def al_value(F: Tensor, h: Tensor, g: Tensor, S: ALState, rho) -> Tensor:
    """`L_i` above: `F [N]`, `h [N, m_e]`, `g [N, m_i]`, scalar `rho` -> `[N]`. With `mu = 0`
    the inequality term is `(rho/2) ||max(0, g)||^2`, the plain quadratic penalty."""
    eq = (S.lam * h).sum(1) + 0.5 * rho * (h * h).sum(1)
    shifted = torch.clamp(S.mu + rho * g, min=0.0)
    ineq = (shifted * shifted - S.mu * S.mu).sum(1) / (2.0 * rho)
    return F + eq + ineq


def al_constraint_grad_coefficients(h: Tensor, g: Tensor, S: ALState, rho):
    """`dL/dh~ = lam + rho h~` and `dL/dg~ = max(0, mu + rho g~)`, shapes of `h` and `g`:
    the cotangents of ONE vector-Jacobian product through the rows, so `grad (L - f)` costs a
    single reverse pass however many rows there are."""
    return S.lam + rho * h, torch.clamp(S.mu + rho * g, min=0.0)


def violation(h: Tensor, g: Tensor) -> Tensor:
    """`v_i = ||[h~_i ; max(0, g~_i)]||_inf` -> `[N]` (0 with no rows at all)."""
    stacked = torch.cat([h.abs(), torch.clamp(g, min=0.0)], dim=1)
    if stacked.shape[1] == 0:  # static: no rows
        return torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
    return stacked.amax(dim=1)


def dual_update(h: Tensor, g: Tensor, S: ALState, alpha, multiplier_max, by_row=False):
    """The dual-ascent step on every particle, unconditionally (module docstring), at the
    scaled rows `h`, `g` of the current swarm. Returns `(S_new, n_clipped [] long)`, the
    number of multiplier entries the clip bound; with `by_row=True` also
    `(lam_rows [m_e], mu_rows [m_i])` long, those entries per row summed over the particles
    (`n_clipped` is their total). (A particle with a non-finite row is redrawn, and its
    multipliers zeroed, at the same check.)"""
    lam_raw = S.lam + alpha * h
    mu_raw = torch.clamp(S.mu + alpha * g, min=0.0)
    lam_rows = (lam_raw.abs() > multiplier_max).sum(0)
    mu_rows = (mu_raw > multiplier_max).sum(0)
    clipped = lam_rows.sum() + mu_rows.sum()
    lam = torch.clamp(lam_raw, min=-multiplier_max, max=multiplier_max)
    mu = torch.clamp(mu_raw, max=multiplier_max)
    if by_row:
        return replace(S, lam=lam, mu=mu), clipped, (lam_rows, mu_rows)
    return replace(S, lam=lam, mu=mu), clipped


def by_group(rows: Tensor, row_group: Tensor, n_groups: int) -> Tensor:
    """Sum a per-row vector `[m]` into its groups: `row_group [m]` long, each row's group
    index in `[0, n_groups)` -> `[n_groups]`, on the device of `rows` (no host sync)."""
    out = torch.zeros(n_groups, dtype=rows.dtype, device=rows.device)
    return out.index_add(0, row_group, rows)


def inf_norm_by_group(M: Tensor, row_group: Tensor, n_groups: int) -> Tensor:
    """`|M_i|_inf` restricted to each group's columns: `M [N, m]`, `row_group [m]` ->
    `[N, n_groups]` (0 for a group with no column in `M`)."""
    out = torch.zeros(M.shape[0], n_groups, dtype=M.dtype, device=M.device)
    if M.shape[1] == 0:
        return out
    idx = row_group.unsqueeze(0).expand(M.shape[0], -1)
    return out.scatter_reduce(1, idx, M.abs(), reduce="amax", include_self=True)


def reset(S: ALState, mask: Tensor) -> ALState:
    """Zero multipliers on the masked particles (a redrawn particle)."""
    m1 = mask.unsqueeze(1)
    return replace(S, lam=torch.where(m1, torch.zeros_like(S.lam), S.lam),
                   mu=torch.where(m1, torch.zeros_like(S.mu), S.mu))


def resample_mask(q: Tensor, rows_finite: Tensor, q_max) -> Tensor:
    """Particles to redraw, `[N]` bool: `||q||_inf > q_max`, any non-finite `q`, or a row
    set the caller found non-finite (`rows_finite [N]`, True where every row is finite)."""
    too_big = q.abs().amax(dim=1) > q_max
    nonfinite = ~torch.isfinite(q).all(dim=1)
    return too_big | nonfinite | ~rows_finite
