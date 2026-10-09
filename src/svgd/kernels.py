"""Stein repulsion for the particle solvers, and the two annealing schedules.

Pure tensor math on N particles with a d-dimensional kernel coordinate `y`. The kernel is an
RBF with the median heuristic (Liu & Wang 2016), `K_ij = exp(-||y_i - y_j||^2 / h)`, and the
quantities a Stein step needs are returned separately so the caller composes them: the
kernel matrix `K` for averaging driving forces, and the repulsion `R`, which lives in the
kernel's own space and must be PULLED BACK into the decision variables when the two differ.

Which space the kernel lives in is the caller's choice (`kernel_space`): for the learned
arm the default is CONFIGURATION space, `y = q(x)`, so that two particles are "close" when
their configurations are, not when their latents are -- two latents far apart that decode
to the same configuration are one solution, and the repulsion should say so. The pullback
is then one vector-Jacobian product through the flow, `R_pulled = J_q^T R`, which the
caller performs because the flow is its to evaluate; `pullback_identity` is the x-space
counterpart so both paths read the same.

THE SIGN OF `R` IS THE SVGD ONE. For particle j the repulsion is

    R_j = sum_i grad_{y_i} K(y_i, y_j) = (2/h) sum_i K_ij (y_j - y_i),

the gradient with respect to the OTHER particle, which pushes y_j away from its neighbours
(Liu & Wang eq. 8, second term). It is MINUS the gradient of `sum_i K_ij` with respect to
y_j -- ascending the kernel in a particle's own coordinate pulls the swarm together, and the
2-D Gaussian moment test in `tests/test_svgd_kernels_al.py` is what pins the sign.

`svgd_direction` returns the kernel-AVERAGED part of the Stein step only,

    phi_j = (1 / Z_j) [ gamma sum_i K_ij driving_i + T R_pulled_j ],

the "Q-method" form in which the caller adds its OWN augmented-Lagrangian gradient outside
this average (Tabor & Hermans). `Z_j` defaults to the kernel row sum `sum_i K_ij` (the
particle's effective neighbour count) rather than the textbook `N`; the fixed point is the
same (a positive per-particle factor does not move a zero), and it is what makes the
no-interaction control the SAME CODE PATH: with `K = I` (`identity_terms`) the row sum is
one and `phi` is exactly the driving term, independent of N. `normaliser="n"` restores the
textbook 1/N.

Compile- and CUDA-graph-friendly throughout: N is static, so even the median over the
off-diagonal pairs is computed at fixed shape (see `median_bandwidth`); no `.item()`, no
boolean indexing; explicit `dtype=` / `device=` on every tensor constructed.
"""

import math

import torch
from torch import Tensor


## --------------------------------------------------------------------------------------
## Distances, bandwidth, kernel
## --------------------------------------------------------------------------------------

def pairwise_sqdist(y: Tensor) -> Tensor:
    """`||y_i - y_j||^2` for all pairs, `[N, N]`, without an `[N, N, d]` temporary.

    The expansion `|y_i|^2 + |y_j|^2 - 2 y_i.y_j`, clamped at zero and with an exactly zero
    diagonal (the expansion leaves ~1e-16 noise there in float64 and ~1e-6 in float32, and
    `median_bandwidth` relies on being able to mark the diagonal).
    """
    sq = (y * y).sum(dim=1)
    d = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (y @ y.transpose(0, 1))
    d = torch.clamp(d, min=0.0)
    eye = torch.eye(y.shape[0], dtype=torch.bool, device=y.device)
    return torch.where(eye, torch.zeros_like(d), d)


def median_bandwidth(sqdist: Tensor, N: int, floor) -> Tensor:
    """Median heuristic: `h = max(floor^2, med(dist)^2 / log N)`, a scalar tensor.

    The median is over the N(N-1)/2 OFF-DIAGONAL pairs, at static shape. The trick: the
    `N x N` matrix lists every pair twice and the diagonal N times; set the diagonal to
    `+inf` so it sorts last, and the doubled pair list `p_1, p_1, p_2, p_2, ...` has, at
    1-indexed positions P = N(N-1)/2 and P+1, exactly the two values whose mean is the
    median of the P pairs (both are `p_(P+1)/2` when P is odd; `p_P/2` and `p_P/2+1` when P
    is even). Two `kthvalue` calls at compile-time-constant k, no masking, no
    data-dependent shape. For N = 1 there is no pair (and `log N = 0`), so the bandwidth is
    `floor^2`; that branch is on the static N, not on data.
    """
    kw = dict(dtype=sqdist.dtype, device=sqdist.device)
    floor_sq = torch.full((), float(floor) ** 2, **kw)     # a fill kernel: no H2D copy, capturable
    if N < 2:
        return floor_sq
    dist = torch.sqrt(sqdist)
    eye = torch.eye(N, dtype=torch.bool, device=sqdist.device)
    flat = torch.where(eye, torch.full_like(dist, float("inf")), dist).reshape(-1)
    P = N * (N - 1) // 2
    lo = torch.kthvalue(flat, P).values
    hi = torch.kthvalue(flat, P + 1).values
    med = 0.5 * (lo + hi)
    h = med * med / math.log(N)
    return torch.maximum(h, floor_sq)


def rbf_terms(y: Tensor, h):
    """RBF kernel `K [N, N]` and the SVGD repulsion `R [N, d]` in the kernel's space.

    `K_ij = exp(-||y_i - y_j||^2 / h)`;
    `R_j = sum_i grad_{y_i} K_ij = (2/h) sum_i K_ij (y_j - y_i) = (2/h) (diag(K 1) y - K y)`.
    The sign is the one that pushes particles APART (module docstring).
    """
    sqdist = pairwise_sqdist(y)
    K = torch.exp(-sqdist / h)
    rowsum = K.sum(dim=1, keepdim=True)
    R = (2.0 / h) * (rowsum * y - K @ y)
    return K, R


def identity_terms(N: int, d: int, dtype, device):
    """The no-interaction control's `(K, R)`: `K = I [N, N]`, `R = 0 [N, d]`.

    Same code path as the RBF terms downstream -- `svgd_direction` with these is exactly
    the driving term, so "kernel off" differs from "kernel on" in nothing but these two
    tensors.
    """
    K = torch.eye(N, dtype=dtype, device=device)
    R = torch.zeros(N, d, dtype=dtype, device=device)
    return K, R


## --------------------------------------------------------------------------------------
## The Stein direction, kernel space and pullback
## --------------------------------------------------------------------------------------

def svgd_direction(K: Tensor, driving: Tensor, R_pulled: Tensor, gamma, T,
                   normaliser="rowsum") -> Tensor:
    """`phi_j = (1/Z_j) [ gamma sum_i K_ij driving_i + T R_pulled_j ]`, `[N, n]`.

    `driving [N, n]` is each particle's own descent direction (the negative objective or AL
    gradient); `R_pulled [N, n]` the repulsion already pulled back to the decision
    variables; `gamma` the driving-force gain (`anneal_gamma`), `T` the repulsion
    temperature (`anneal_T`, exactly zero before the polish). `Z_j` is the kernel row sum
    (default) or N (`normaliser="n"`); see the module docstring for why the row sum.
    """
    N = K.shape[0]
    mixed = gamma * (K @ driving) + T * R_pulled
    if normaliser == "n":
        return mixed / N
    if normaliser == "rowsum":
        return mixed / K.sum(dim=1, keepdim=True)
    raise ValueError(f"normaliser must be 'rowsum' or 'n', got {normaliser!r}")


def kernel_space(y_q: Tensor, x: Tensor, which: str):
    """The coordinate the kernel lives in: configuration (`"q"`), decision variables
    (`"x"`), or `None` for no repulsion (the caller then uses `identity_terms`)."""
    if which == "q":
        return y_q
    if which == "x":
        return x
    if which == "none":
        return None
    raise ValueError(f"kernel space must be 'q', 'x' or 'none', got {which!r}")


def pullback_identity(R: Tensor) -> Tensor:
    """The x-space pullback: the kernel coordinate IS the decision variable, so `R` is
    already in the right space. The q-space counterpart is `R_pulled = J_q^T R`, one VJP
    through the flow, performed by the caller."""
    return R


def matrix_kernel_precondition(A_mean: Tensor, delta) -> Tensor:
    """SVN-lite preconditioner: `(A_mean + delta I)^-1`, `[n, n]`, to be applied to `phi`.

    `A_mean` is a swarm-averaged positive semidefinite curvature estimate (e.g. the mean
    Gauss-Newton matrix `J^T J`); the same matrix on every particle keeps this a scalar
    kernel with a constant metric rather than a matrix-valued kernel. Off by default on the
    caller's side; a singular system returns the identity rather than raising.
    """
    n = A_mean.shape[0]
    eye = torch.eye(n, dtype=A_mean.dtype, device=A_mean.device)
    inv, info = torch.linalg.solve_ex(A_mean + delta * eye, eye, check_errors=False)
    ok = (info == 0) & torch.isfinite(inv).all()
    return torch.where(ok, inv, eye)


## --------------------------------------------------------------------------------------
## Schedules
## --------------------------------------------------------------------------------------

def _clamp_max(x, hi):
    return torch.clamp(x, max=hi) if isinstance(x, Tensor) else min(x, hi)


def _clamp_min(x, lo):
    return torch.clamp(x, min=lo) if isinstance(x, Tensor) else max(x, lo)


def anneal_gamma(t, gamma_t, p=1, cycles=1):
    """D'Angelo & Fortuin driving-force gain: `min(1, (mod(t, T/C) / (T/C))^p)` for
    `t < T`, and 1 thereafter, with `T = gamma_t * C`. Each of the C cycles ramps the driving
    force from 0 to 1 over `gamma_t` steps, so repulsion dominates at the start of every
    cycle; once the annealing horizon `T` is past the gain is 1 for the rest of the run
    (for `cycles = 1` that is a single ramp of `gamma_t` steps). `t` may be a number or a
    tensor.
    """
    period = float(gamma_t)
    horizon = period * cycles
    frac = (t % period) / period
    ramp = _clamp_max(frac ** p, 1.0)
    if isinstance(t, Tensor):
        return torch.where(t >= horizon, torch.ones_like(ramp), ramp)
    return 1.0 if t >= horizon else ramp


def anneal_T(t, T0, anneal_frac, total_steps):
    """Repulsion temperature `T0 * max(0, 1 - t / (anneal_frac * total_steps))^2`.

    Reaches EXACTLY 0.0 at `t = anneal_frac * total_steps` and stays there, so the polish
    runs on a swarm with no repulsion at all rather than a vanishingly small one. `t` may
    be a number or a tensor.
    """
    horizon = float(anneal_frac) * float(total_steps)
    s = _clamp_min(1.0 - t / horizon, 0.0)
    return T0 * s * s
