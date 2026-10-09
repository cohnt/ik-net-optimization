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

    phi_i = (1/N) sum_j [ K_ji (-grad f_j / T) + grad_{y_j} K(q_j, q_i) ] - (1/T) grad (L - f)_i,

or, with `inside=True` (the literal SVGD on exp(-L/T), an A/B), the whole `-grad L / T` inside
the average. The no-interaction control (`svgd_kernel = none`) is `kernel=False`: both kernel
terms dropped, `phi_i = -(1/T) grad L_i`.

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


def median_bandwidth(sqdist: Tensor, N: int, floor) -> Tensor:
    """Median heuristic: `h = max(floor^2, med(dist)^2 / log N)`, a scalar tensor.

    The median is over the N(N-1)/2 OFF-DIAGONAL pairs, at static shape: with the diagonal
    set to +inf the doubled pair list sorts as `p_1, p_1, p_2, p_2, ...`, so positions P and
    P + 1 (P = N(N-1)/2) hold the two values whose mean is the median. N = 1 has no pair (and
    `log 1 = 0`): the bandwidth is `floor^2`, a branch on the static N."""
    kw = dict(dtype=sqdist.dtype, device=sqdist.device)
    floor_sq = torch.full((), float(floor) ** 2, **kw)     # a fill kernel: capturable
    if N < 2:
        return floor_sq
    dist = torch.sqrt(sqdist)
    eye = torch.eye(N, dtype=torch.bool, device=sqdist.device)
    flat = torch.where(eye, torch.full_like(dist, float("inf")), dist).reshape(-1)
    P = N * (N - 1) // 2
    med = 0.5 * (torch.kthvalue(flat, P).values + torch.kthvalue(flat, P + 1).values)
    return torch.maximum(med * med / math.log(N), floor_sq)


def rbf_terms(y: Tensor, h):
    """`K [N, N]` and the repulsion `R [N, d]` in the kernel's space:
    `K_ij = exp(-||y_i - y_j||^2 / h)`, `R_j = (2/h) (diag(K 1) y - K y)_j`."""
    K = torch.exp(-pairwise_sqdist(y) / h)
    R = (2.0 / h) * (K.sum(dim=1, keepdim=True) * y - K @ y)
    return K, R


def stein_direction(K: Tensor, R_y: Tensor, gF_y: Tensor, gC_y: Tensor, T, inside=False,
                    kernel=True) -> Tensor:
    """`phi [N, n]` in the normalised coordinates `y` (module docstring).

    `K [N, N]` the kernel, `R_y [N, n]` its repulsion pulled back to `y`, `gF_y` the
    objective's gradient and `gC_y = grad_y (L - f)` the constraint part of the AL's, both
    `[N, n]`; `T` the temperature. `kernel=False` is the no-interaction control, `inside=True`
    the literal form with the whole AL gradient under the kernel average."""
    if not kernel:
        return -(gF_y + gC_y) / T
    N = K.shape[0]
    if inside:
        return (K @ (-(gF_y + gC_y) / T) + R_y) / N
    return (K @ (-gF_y / T) + R_y) / N - gC_y / T
