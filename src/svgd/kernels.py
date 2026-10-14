"""The Stein part of the `svgd` step: RBF kernel, median bandwidth, and the update direction.

Pure tensor math on N particles. The kernel is an RBF with the median heuristic (Liu & Wang
2016), `K_ij = exp(-||q_i - q_j||^2 / h)`, `h = med(||q_i - q_j||)^2 / log N` (floored), in
CONFIGURATION space -- two particles are "close" when their configurations are, whatever
decision variables produced them. Its repulsion

    R_j = sum_i grad_{q_i} K(q_i, q_j) = (2/h) sum_i K_ij (q_j - q_i)

lives in configuration space and is pulled back to the decision variables by the caller, one
vector-Jacobian product through the flow (`J_q^T R`); the joint-space arm's map is the identity.

THE SIGN OF `R` IS THE SVGD ONE: the gradient with respect to the OTHER particle, which pushes
q_j away from its neighbours (Liu & Wang eq. 8, second term). The 2-D Gaussian moment test in
`tests/test_svgd_kernels_al.py` pins it.

`stein_direction` is the Tabor-Hermans form (arXiv 2506.00589, the "Q method"): only the
objective's gradient and the repulsion are averaged over the kernel, and each particle's OWN
constraint gradient is added outside the average,

    phi_i = (1/N) sum_j [ K_ji (-grad f_j) + grad_{y_j} K(q_j, q_i) ] - grad (L - f)_i,

or, with `inside=True` (the literal SVGD on exp(-L), an A/B), the whole `-grad L` inside
the average. The no-interaction control (`svgd_kernel = none`) is `kernel=False`: both kernel
terms dropped, `phi_i = -grad L_i`. The target carries no temperature: `svgd_temperature` and
`svgd_metric_temperature` were removed 2026-10-14 (stage SVGD_R22 measured T inert), so T = 1.

Compile- and CUDA-graph-friendly: N is static (even the median over the off-diagonal pairs is
taken at fixed shape), no `.item()`, no boolean indexing, explicit dtype / device.
"""

import math

import torch
from torch import Tensor


def pairwise_sqdist(y: Tensor) -> Tensor:
    """`||y_i - y_j||^2` for all pairs, `[N, N]`, without an `[N, N, d]` temporary; clamped at
    zero with an exactly zero diagonal (`median_bandwidth` marks the diagonal)."""
    sq = (y * y).sum(dim=1)
    d = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (y @ y.transpose(0, 1))
    d = torch.clamp(d, min=0.0)
    eye = torch.eye(y.shape[0], dtype=torch.bool, device=y.device)
    return torch.where(eye, torch.zeros_like(d), d)


def median_bandwidth(sqdist: Tensor, N: int, floor, live: Tensor = None) -> Tensor:
    """Median heuristic: `h = max(floor^2, med(dist)^2 / log N)`, a scalar tensor.

    The median is over the N(N-1)/2 OFF-DIAGONAL pairs, at static shape: with the diagonal
    set to +inf the doubled pair list sorts as `p_1, p_1, p_2, p_2, ...`, so positions P and
    P + 1 (P = N(N-1)/2) hold the two values whose mean is the median. N = 1 has no pair (and
    `log 1 = 0`): the bandwidth is `floor^2`, a branch on the static N.

    `live [N]` bool (the dropped-particle swarm, `svgd_resample = drop` / `keep`): the median
    over the pairs of LIVE particles only and `log n_live` for `log N`, still at static shape
    (a sort and a gather at the data-dependent position); `floor^2` below two live particles.
    With every particle live it returns exactly the unmasked value: the sorted positions hold
    the same elements `kthvalue` returns, and the `log` correction is an exact 1.0."""
    kw = dict(dtype=sqdist.dtype, device=sqdist.device)
    floor_sq = torch.full((), float(floor) ** 2, **kw)     # a fill kernel: capturable
    if N < 2:
        return floor_sq
    dist = torch.sqrt(sqdist)
    eye = torch.eye(N, dtype=torch.bool, device=sqdist.device)
    if live is None:
        flat = torch.where(eye, torch.full_like(dist, float("inf")), dist).reshape(-1)
        P = N * (N - 1) // 2
        med = 0.5 * (torch.kthvalue(flat, P).values + torch.kthvalue(flat, P + 1).values)
        return torch.maximum(med * med / math.log(N), floor_sq)
    pair = live.unsqueeze(1) & live.unsqueeze(0) & ~eye
    srt = torch.sort(torch.where(pair, dist, torch.full_like(dist, float("inf"))).reshape(-1)).values
    n_live = live.sum()
    P = n_live * (n_live - 1) // 2
    lo = torch.clamp(P - 1, min=0).reshape(1)
    hi = torch.clamp(P, max=N * N - 1).reshape(1)
    med = 0.5 * (srt.index_select(0, lo) + srt.index_select(0, hi)).reshape(())
    ratio = torch.where(n_live == N, torch.ones((), **kw),
                        math.log(N) / torch.log(torch.clamp(n_live.to(sqdist.dtype), min=2.0)))
    h = torch.maximum(med * med / math.log(N) * ratio, floor_sq)
    return torch.where(n_live >= 2, h, floor_sq)


def rbf_terms(y: Tensor, h):
    """`K [N, N]` and the repulsion `R [N, d]` in the kernel's space:
    `K_ij = exp(-||y_i - y_j||^2 / h)`, `R_j = (2/h) (diag(K 1) y - K y)_j`."""
    K = torch.exp(-pairwise_sqdist(y) / h)
    R = (2.0 / h) * (K.sum(dim=1, keepdim=True) * y - K @ y)
    return K, R


def stein_direction(K: Tensor, R_y: Tensor, gF_y: Tensor, gC_y: Tensor, inside=False,
                    kernel=True, avg_scale=None) -> Tensor:
    """`phi [N, n]` in the normalised coordinates `y` (module docstring).

    `K [N, N]` the kernel, `R_y [N, n]` its repulsion pulled back to `y`, `gF_y` the
    objective's gradient and `gC_y = grad_y (L - f)` the constraint part of the AL's, both
    `[N, n]`. `kernel=False` is the no-interaction control, `inside=True`
    the literal form with the whole AL gradient under the kernel average. `avg_scale` (a
    scalar tensor, the dropped-particle swarm) rescales the kernel average from `1/N` to
    `1/n_live` -- exactly 1.0 when every particle is live."""
    if not kernel:
        return -(gF_y + gC_y)
    N = K.shape[0]
    if inside:
        avg = (K @ (-(gF_y + gC_y)) + R_y) / N
        return avg if avg_scale is None else avg * avg_scale
    avg = (K @ (-gF_y) + R_y) / N
    if avg_scale is not None:
        avg = avg * avg_scale
    return avg - gC_y
