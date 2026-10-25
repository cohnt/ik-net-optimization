"""BiLipNet and the LOInK surrogate model, ported to torch: the core of the `loink` chart family.

Why this module exists
----------------------
The chart ablation's next rung (after the cVAE, `src/charts/cvae.py`) is LOInK, *Learned
Optimal Inverse Kinematics* (Somerfield, Abood, Wang, Manchester, arXiv 2609.21275). LOInK
learns a bi-Lipschitz change of coordinates `G : q -> (y_hat, z)` whose first `p` outputs
fit the task and whose last `K = n - p` outputs are a latent in which a learned cost is a
positive-definite quadratic with its minimum at `z = 0`; IK is then `q = G^{-1}(y, z)`.
`G` is a BiLipNet (Wang, Dvijotham, Manchester, *Monotone, Bi-Lipschitz, and
Polyak-Lojasiewicz Networks*, ICML 2024, arXiv 2402.01344), whose inverse has no closed
form and is computed by Davis-Yin operator splitting.

There is no LOInK code online (2026-10-22); BiLipNet's is the official PLNet repository,
https://github.com/acfr/PLNet (MIT), JAX/Flax, `surrogte_loss/rb20d_solver.py` and
`surrogte_loss/layer.py`. This module is a line-by-line port of that code, checked against a
float64 golden file generated from it (`tests/test_loink_reference.py`,
`tests/data/loink_reference/`). The first part of this file is the network core, its
inversion, the LOInK model and its loss; the second ("the chart family", at the end) the task
representation, the training costs, the chart wrapper, the sidecar and the loader.

What is here, and the equations it implements
---------------------------------------------
  * `cayley`        BiLipNet eq. (4), PLNet's `layer.cayley` (orthonormal columns; transposes
                    when the matrix is wide, as PLNet does).
  * `Unitary`       the orthogonal layers O_k of eq. (8), `O(x) = P x + q`, P by Cayley.
  * `MonLipLayer`   the mu-strongly-monotone, nu-Lipschitz feed-through layer, eq. (1)-(7):
                    `z = sigma(V z + sqrt(2 gam) S x + b_hat)`, `y = mu x + sqrt(gam/2) S^T z + b_y`
                    (eq. 6), gam = nu - mu, V_k = 2 B_k A_{k-1}^T, S_k = A_k Q_k - B_k Q_{k-1}
                    (Sec. 3.3). The learned per-unit scaling Psi of eq. (5) is ABSENT, exactly
                    as in PLNet's code (sigma_hat = sigma).
  * `BiLipNet`      eq. (8): uni_0, mon_0, ..., mon_{depth-1}, uni_depth; (prod mu, prod nu).
  * the `mln_*` helpers: PLNet's functions on the explicit parameter dict; `mln_bwd_*` are
                    the inverse model eq. (10), `mln_RA` is the resolvent R_A of eq. (11) by
                    forward substitution, App. B eq. (28).
  * `dys_layer` / `dys_inverse`: the Davis-Yin iteration eq. (11) / App. B eq. (27), as
                    PLNet's `mln_back_solve_dys` + `DavisYinSplit`, for an ARBITRARY target
                    (PLNet's script inverts only at 0): u0 = 0, step alpha = mu/gam (PLNet's
                    `alpha * mu/gam` with its module global alpha = 1.0; note App. B proves
                    convergence on the OPEN interval (0, mu/gam), so this is its edge), relaxation
                    Lambda = 1, x recovered from the LAST z^{k+1}, layers inverted last to first
                    (eq. 9).
  * `invert_bilipnet`: the DYS result, `n_newton` Newton polish steps on the n-dim map
                    (undamped, or damped by `damped_newton_step` over `STEP_SIZES`: the
                    chart's `newton_damping`), and one implicit-function-theorem linearisation
                    that makes its derivative exact.
  * `LoinkModel` / `loink_loss`: LOInK eq. (9)-(11) and (14)-(16).

Deliberate differences from PLNet (LOInK's convention, Sec. V.B: "We fix the bi-Lipschitz
lower bound mu at 0.1, keeping the upper bound nu as a free parameter, but with a small
penalty on the distortion tau := nu/mu")
  * `MonLipLayer.mu` is a FIXED buffer and `nu = exp(lognu)` is learned; rb20d_solver.py's
    MonLipNet learns lognu and FIXES tau (mu = nu / tau). Keeping `nu = exp(lognu)` exactly
    (no softplus reparametrisation) is what makes the golden check exact; nothing enforces
    `nu > mu`, so `check_bounds()` reports it and the trainer penalises it.
  * the tanh activation is supported (PLNet's code is ReLU-only); its DYS prox step is
    `prox_tanh` (below).

Parameter layout: PLNet's, NOT torch's
--------------------------------------
Every matrix is stored in PLNet's orientation (`W` is applied as `x @ R^T`, `Fq` is
(n, sum units), `Fab{j}` is (nz_j + nz_{j-1}, nz_j)), under PLNet's flax names (`Fab0`,
`fab0`, `b0`, ...), so `load_plnet_params` copies the flax tree with NO transposes.

Compile / jacrev / CUDA-graph discipline
----------------------------------------
Pure torch, dtype and device follow the parameters, batched over a leading dimension. Every
loop runs a FIXED number of times (from Python ints and the architecture), there is no
`.item()`, no Python branch on a tensor value (`torch.where` / `torch.clamp` instead), and
linear solves go through `_solve` = `lu_factor_ex(check_errors=False)` + `lu_solve`,
NOT `torch.linalg.solve` or `solve_ex`: the program captures the compiled `jacrev` as a
CUDA graph under `enable_grad`, and in that mode `linalg_solve`'s BACKWARD recomputes its
gradient with a plain `at::linalg_solve`, whose info check (`_linalg_check_errors`) is a
host sync that stream capture forbids ("operation not permitted when stream is
capturing", every soft12 LOINKSMOKE item on torch 2.14, 2026-10-23). `lu_solve`'s
derivatives are `lu_solve` and triangular solves on both sides, with no check anywhere. No
`torch.autograd.Function`: the inverse's derivative comes from the IFT step's structure.
"""

import math

import torch
from torch import nn

FAMILY = "loink"

#: Slope-restricted in [0, 1], hence each the prox of a convex function (BiLipNet App. B.1,
#: Table 2): ReLU = prox of the indicator of [0, inf); tanh = prox^1 of
#: f(z) = 1/2 [ln(1 - z^2) + z ln((1+z)/(1-z)) - z^2], i.e. f'(z) = atanh(z) - z.
ACTIVATIONS = ("relu", "tanh")

#: LOInK's cost-head variants. "m" and "x" build the SAME model (a head H : y_hat -> (b, P)
#: and J_hat = z^T (I/2 + P^T P) z + b, eq. 10-11); the letter names which cost target the
#: trainer fits, which is the trainer's business. "u" has no head and no auxiliary loss.
COSTS = ("m", "x", "u")

#: Fixed Newton steps of the tanh prox solve (see `prox_tanh`).
PROX_TANH_STEPS = 8


def _solve(A, B, left=True):
    """`A^{-1} B` (left) or `B A^{-1}` (left=False), with no info-check sync in either the
    forward or the backward pass (see the module docstring)."""
    LU, pivots, _info = torch.linalg.lu_factor_ex(A, check_errors=False)
    return torch.linalg.lu_solve(LU, pivots, B, left=left)


def _act(activation, x):
    return torch.relu(x) if activation == "relu" else torch.tanh(x)


def _glorot_normal(shape):
    """flax's `glorot_normal`: variance_scaling(1, fan_avg, truncated_normal) -- a normal
    truncated at +-2 std, std corrected by 0.87962566 so the variance is 1/fan_avg.
    Initialisation only: a trained or loaded model overwrites it."""
    fan_avg = 0.5 * (shape[-2] + shape[-1])
    std = math.sqrt(1.0 / fan_avg) / 0.87962566103423978
    W = torch.empty(shape)
    nn.init.trunc_normal_(W, std=std, a=-2 * std, b=2 * std)
    return W


## ---------------------------------------------------------------------- orthogonal layers

def cayley(W):
    """PLNet's Cayley map (BiLipNet eq. 4) on W (m, n), returning an (m, n) matrix with
    orthonormal columns (m >= n) or rows (n > m, via the transpose, as PLNet does).

    U = W[:n], V = W[n:], Z = (U - U^T) + V^T V,
    cayley(W) = [(I + Z)^{-1} (I - Z); -2 V (I + Z)^{-1}].
    (The paper writes Z = G^T - G + H^T H; PLNet's code has U - U^T. Both skew parts give an
    orthonormal result; this is PLNet's, which the golden check pins.)
    """
    m, n = W.shape
    if n > m:
        return cayley(W.T).T
    U, V = W[:n], W[n:]
    I = torch.eye(n, dtype=W.dtype, device=W.device)
    Z = (U - U.T) + V.T @ V
    top = _solve(I + Z, I - Z)
    if m == n:
        return top
    return torch.cat([top, -2 * _solve(I + Z, V, left=False)], dim=0)


class Unitary(nn.Module):
    """An orthogonal affine layer `x -> x R^T + b` with R = cayley((a / |W|_F) W).

    PLNet's `layer.Unitary` (square, with bias): `W` (n, n) glorot-normal, `a` (1,) initialised
    to |W|_F (so R starts at cayley(W)), `b` (n,) zeros."""

    def __init__(self, n):
        super().__init__()
        W = _glorot_normal((n, n))
        self.W = nn.Parameter(W)
        self.a = nn.Parameter(torch.linalg.norm(W).reshape(1))
        self.b = nn.Parameter(torch.zeros(n))

    def R(self):
        return cayley((self.a / torch.linalg.norm(self.W)) * self.W)

    def forward(self, x):
        return x @ self.R().T + self.b

    def inverse(self, y):
        return (y - self.b) @ self.R()

    def explicit(self):
        """`(R, b)`: PLNet's `Unitary.get_params`."""
        return self.R(), self.b


## ---------------------------------------------------------------------- monotone layers

class MonLipLayer(nn.Module):
    """A mu-strongly-monotone, nu-Lipschitz feed-through layer (BiLipNet Sec. 3, eq. 6).

    PLNet's parametrisation exactly (rb20d_solver.py `MonLipNet.__call__`, transcribed line
    by line): free parameters `Fq` (n, sum units), `fq` (1,), and per block j `Fab{j}`
    (nz_j + nz_{j-1}, nz_j) with nz_{-1} = 0, `fab{j}` (1,), `b{j}` (nz_j,); `by` (n,);
    `lognu` (1,). `mu` is a FIXED buffer (LOInK's convention, module docstring) and
    nu = exp(lognu), gam = nu - mu. The bounds hold for every parameter value with nu > mu;
    `check_bounds` reports whether that is so (nothing enforces it).
    """

    def __init__(self, n, units, mu, lognu_init=math.log(2.0), activation="relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation {activation!r} not in {ACTIVATIONS}")
        if not math.exp(lognu_init) > mu > 0:
            raise ValueError(f"need nu = exp(lognu_init) = {math.exp(lognu_init):.4g} > mu = {mu:.4g} > 0")
        self.n, self.units, self.activation = int(n), tuple(int(u) for u in units), activation
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.lognu = nn.Parameter(torch.full((1,), float(lognu_init)))
        self.by = nn.Parameter(torch.zeros(self.n))
        Fq = _glorot_normal((self.n, sum(self.units)))
        self.Fq = nn.Parameter(Fq)
        self.fq = nn.Parameter(torch.linalg.norm(Fq).reshape(1))
        nz_1 = 0
        for j, nz in enumerate(self.units):
            Fab = _glorot_normal((nz + nz_1, nz))
            setattr(self, f"Fab{j}", nn.Parameter(Fab))
            setattr(self, f"fab{j}", nn.Parameter(torch.linalg.norm(Fab).reshape(1)))
            setattr(self, f"b{j}", nn.Parameter(torch.zeros(nz)))
            nz_1 = nz

    def nu(self):
        return torch.exp(self.lognu).squeeze(0)

    def check_bounds(self):
        """`(nu > mu, nu - mu)` as tensors: the layer is (mu, nu) only while gam > 0, and
        sqrt(2 gam) is NaN otherwise. The trainer penalises a violation."""
        gam = self.nu() - self.mu
        return gam > 0, gam

    def _ABT(self, j):
        Fab, fab = getattr(self, f"Fab{j}"), getattr(self, f"fab{j}")
        return cayley((fab / torch.linalg.norm(Fab)) * Fab)

    def forward(self, x):
        nu = torch.exp(self.lognu)
        mu = self.mu
        y = mu * x + self.by
        QT = cayley((self.fq / torch.linalg.norm(self.Fq)) * self.Fq)
        sqrt_2g, sqrt_g2 = torch.sqrt(2.0 * (nu - mu)), torch.sqrt((nu - mu) / 2.0)
        idx, nz_1 = 0, 0
        zk, Ak_1 = None, None
        for k, nz in enumerate(self.units):
            ABT = self._ABT(k)
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]
            QTk = QT[:, idx:idx + nz]
            bk = getattr(self, f"b{k}")
            if k == 0:
                # PLNet runs the k = 0 block through zero-width matrices (zk = x[..., :0],
                # Ak_1 = zeros((0, 0))); the terms they contribute are exactly zero.
                STk = QTk @ ATk
                zk = _act(self.activation, sqrt_2g * x @ STk + bk)
            else:
                QTk_1 = QT[:, idx - nz_1:idx]
                STk = QTk @ ATk - QTk_1 @ BTk
                zk = _act(self.activation, 2 * (zk @ Ak_1) @ BTk + sqrt_2g * x @ STk + bk)
            y = y + sqrt_g2 * zk @ STk.T
            idx += nz
            nz_1 = nz
            Ak_1 = ATk.T
        return y

    def explicit(self):
        """The explicit model (BiLipNet eq. 6-7), PLNet's `MonLipNet.get_params`:
        `{mu, nu, gam, units, activation, V, S, bh, by}` with V the list of V_j (nz_j, nz_{j-1})
        for j >= 1 (V_j = 2 B_j A_{j-1}^T), S (sum units, n) the stacked S_j = A_j Q_j - B_j Q_{j-1},
        bh the stacked b_j."""
        nu = self.nu()
        Q = cayley((self.fq / torch.linalg.norm(self.Fq)) * self.Fq).T
        V, S, bh = [], [], []
        idx = 0
        ATk_1 = Qk_1 = None
        for k, nz in enumerate(self.units):
            Qk = Q[idx:idx + nz, :]
            bh.append(getattr(self, f"b{k}"))
            ABT = self._ABT(k)
            if k > 0:
                Ak, Bk = ABT[:nz, :].T, ABT[nz:, :].T
                V.append(2 * Bk @ ATk_1)
                S.append(Ak @ Qk - Bk @ Qk_1)
            else:
                Ak = ABT.T
                S.append(ABT.T @ Qk)
            ATk_1, Qk_1 = Ak.T, Qk
            idx += nz
        return {"mu": self.mu, "nu": nu, "gam": nu - self.mu, "units": self.units,
                "activation": self.activation, "V": V, "S": torch.cat(S, dim=0),
                "bh": torch.cat(bh, dim=0), "by": self.by}


## PLNet's helpers on the explicit dict (rb20d_solver.py "mln utils"), activation-aware.

def mln_fwd_x2b(p, x):
    return torch.sqrt(2 * p["gam"]) * x @ p["S"].T + p["bh"]


def mln_fwd_b2z(p, b):
    z, idx, zk = [], 0, None
    for k, nz in enumerate(p["units"]):
        if k == 0:
            zk = _act(p["activation"], b[..., idx:idx + nz])
        else:
            zk = _act(p["activation"], zk @ p["V"][k - 1].T + b[..., idx:idx + nz])
        z.append(zk)
        idx += nz
    return torch.cat(z, dim=-1)


def mln_fwd_xz2y(p, x, z):
    return p["mu"] * x + torch.sqrt(p["gam"] / 2) * z @ p["S"] + p["by"]


def mln_fwd(p, x):
    b = mln_fwd_x2b(p, x)
    z = mln_fwd_b2z(p, b)
    return mln_fwd_xz2y(p, x, z)


def mln_bwd_yz2x(p, y, z):
    return (y - p["by"] - torch.sqrt(p["gam"] / 2) * z @ p["S"]) / p["mu"]


def mln_bwd_y2b(p, y):
    """b_z of eq. (10): sqrt(2 gam)/mu S (y - b_y) + b_hat."""
    return torch.sqrt(2 * p["gam"]) / p["mu"] * (y - p["by"]) @ p["S"].T + p["bh"]


def mln_bwd_z2v(p, z):
    """C(z) of Prop. 4.2: (gam/mu) S S^T z."""
    return p["gam"] / p["mu"] * (z @ p["S"]) @ p["S"].T


def mln_RA(p, alpha, bz, zh, uh):
    """z^{k+1} = R_A(u^{k+1/2} - alpha C(z^{k+1/2})) by forward substitution, App. B eq. (28)."""
    zv = mln_bwd_z2v(p, zh)
    vh = bz - zv
    au, av = 1 / (1 + alpha), alpha / (1 + alpha)
    b = av * vh + au * uh
    z, idx, zk = [], 0, None
    for k, nz in enumerate(p["units"]):
        if k == 0:
            zk = b[..., idx:idx + nz]
        else:
            zk = av * zk @ p["V"][k - 1].T + b[..., idx:idx + nz]
        z.append(zk)
        idx += nz
    return torch.cat(z, dim=-1)


## ---------------------------------------------------------------------- the prox step

def prox_tanh(u, alpha, steps=PROX_TANH_STEPS):
    """prox_f^alpha(u) for the f with prox_f^1 = tanh (BiLipNet Table 2), elementwise.

    Derivation. sigma = prox_f^1 = (I + df)^{-1}, so df = sigma^{-1} - I = atanh - I, and
        prox_f^alpha = (I + alpha df)^{-1} = ((1 - alpha) I + alpha atanh)^{-1},
    i.e. v = prox_f^alpha(u) is the unique v in (-1, 1) with (1 - alpha) v + alpha atanh(v) = u.
    (Unique: the left side has derivative (1 - alpha) + alpha / (1 - v^2) >= 1 for alpha > 0.)
    At alpha = 1 it is tanh(u).

    Solved in w = atanh(v), v = tanh(w) -- the same equation, g(w) = (1 - alpha) tanh(w) +
    alpha w = u, but g' lies in [min(alpha, 1), max(alpha, 1)], so there is no singularity at
    |v| -> 1 and no clamp of v is needed (in v, Newton overshoots past 1 from tanh(u) and then
    crawls back along the log singularity). Write a = |u|, c = 1 - alpha. Then w* has the sign
    of u and |w*| lies in [(a - |c|)/alpha, (a + |c|)/alpha] (|tanh| <= 1); g is concave on
    w > 0 for alpha < 1 and convex for alpha > 1, and the start
        |w0| = max(a, (a - c)/alpha)  (alpha < 1, both are lower bounds on |w*|)
        |w0| = min(a, (a - c)/alpha)  (alpha > 1, both are upper bounds)
    sits on the side from which Newton is monotone, so the iterates never overshoot. They are
    clamped to the bracket anyway (the safeguard; inactive in exact arithmetic). |w0| = a is
    the alpha = 1 answer v = tanh(u). A FIXED `steps` (8) reach 1e-14 absolute on |u| <= 60 for
    alpha in [0.01, 100] and 6e-12 at alpha = 1e-3 (measured, float64).
    """
    alpha = torch.as_tensor(alpha, dtype=u.dtype, device=u.device)
    c = 1 - alpha
    a, s = u.abs(), torch.sign(u)
    edge = (a - c) / alpha
    w = torch.where(alpha < 1, torch.maximum(a, edge), torch.minimum(a, edge))
    lo, hi = (a - c.abs()) / alpha, (a + c.abs()) / alpha
    for _ in range(steps):
        t = torch.tanh(w)
        w = torch.clamp(w - (c * t + alpha * w - a) / (c * (1 - t * t) + alpha), lo, hi)
    return s * torch.tanh(w)


def prox(activation, u, alpha):
    """The DYS z^{k+1/2} step, prox_f^alpha(u). ReLU's f is an indicator, so its prox is relu
    for every alpha (PLNet's `DavisYinSplit` hard-codes relu)."""
    return torch.relu(u) if activation == "relu" else prox_tanh(u, alpha)


## ---------------------------------------------------------------------- inversion

def dys_layer(p, y, n_dys, record=False):
    """x = F^{-1}(y) for one monotone layer by `n_dys` Davis-Yin steps (eq. 11), PLNet's
    `DavisYinSplit` loop: u0 = 0, alpha = mu/gam, Lambda = 1, x from the LAST z^{k+1} (eq. 10).
    With `record`, also returns [(z^{k+1/2}, z^{k+1}, u^{k+1}) per step]."""
    bz = mln_bwd_y2b(p, y)
    alpha = p["mu"] / p["gam"]
    u = torch.zeros_like(bz)
    zk = u
    history = []
    for _ in range(n_dys):
        zh = prox(p["activation"], u, alpha)
        uh = 2 * zh - u
        zk = mln_RA(p, alpha, bz, zh, uh)
        u = u + (zk - zh)
        if record:
            history.append((zh, zk, u))
    x = mln_bwd_yz2x(p, y, zk)
    return (x, history) if record else x


def dys_inverse(net, w, n_dys):
    """The plain DYS inverse of a BiLipNet at `w` (eq. 9), no Newton, no IFT step: what
    PLNet's `mln_back_solve_dys` computes, for an arbitrary target. Differentiable through
    the iterations if called with grad enabled (`invert_bilipnet` never does)."""
    y = net.uni[net.depth].inverse(w)
    for k in reversed(range(net.depth)):
        x = dys_layer(net.mon[k].explicit(), y, n_dys)
        y = net.uni[k].inverse(x)
    return y


def bilipnet_jacobian(net, x):
    """dG/dx at each row of `x`, [B, n, n], by `torch.func.jacrev` vmapped over the batch."""
    return torch.func.vmap(torch.func.jacrev(lambda xi: net(xi.unsqueeze(0)).squeeze(0)))(x)


#: The damped Newton polish's candidate step sizes, largest first (`damped_newton_step`).
STEP_SIZES = (1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)

#: The polish's step rule, a per-chart setting (`newton_damping` in the sidecar):
#: "none" = the full Newton step every time, "backtrack" = `damped_newton_step`.
NEWTON_DAMPINGS = ("none", "backtrack")


def damped_newton_step(net, x, w):
    """One damped Newton step on `G(x) = w`, per sample, no autograd (call under no_grad).

    Direction `d = J_G(x)^{-1} (w - G(x))`; `G` is evaluated at every candidate
    `x + s d`, `s` in `STEP_SIZES`, in ONE batched forward ([S * B, n]); each sample takes the
    candidate with the smallest residual `|G(x + s d) - w|_2` (`argmin` returns the FIRST
    minimum, so ties go to the larger step), and keeps `x` itself when no candidate improves on
    it, so the residual is per-sample non-increasing. Why damped: the undamped step
    (`s = 1` always) is non-monotone, and on real soft PCS charts it diverges or wanders on
    rare inputs (|x| 20-40 against ~3 at the solution, max residual 50-70 over 2048 samples at
    an (n_dys, n_newton) that passed on 256), so no fixed step count was reliable. `G` is
    bi-Lipschitz, so `J_G` is nonsingular everywhere and, where `G` is smooth (tanh), `d` is a
    descent direction of `|G(x) - w|^2`: a small enough step always decreases it.

    NOT for every chart: a relu `G` is piecewise linear, and from a point near a kink the
    piece's direction can ascend for every step down to 1/64 (measured on soft12 `urP`: ~5%
    of 2048 samples stall at residual ~10, some even with steps down to 2^-30), where the full
    step crosses the kink and converges. Hence `newton_damping` is a per-chart setting that
    the export ladder chooses ("none" or "backtrack").

    Graph-safe: static shapes, scalar-times-tensor candidates (no host-to-device copy), and
    selection by `argmin` / `take_along_dim` / `where` only, no python branching on tensor
    values, so torch.compile and CUDA-graph capture see one fixed graph. Costs, in passes of
    `G`: the Jacobian (~n + 1) plus the S candidate evaluations (one batched call)."""
    r = w - net(x)
    d = _solve(bilipnet_jacobian(net, x), r.unsqueeze(-1)).squeeze(-1)
    B, n = x.shape
    S = len(STEP_SIZES)
    cand = torch.stack([x + s * d for s in STEP_SIZES])                 # [S, B, n]
    res = (net(cand.reshape(S * B, n)).reshape(S, B, n) - w.unsqueeze(0)).norm(dim=-1)  # [S, B]
    best = res.argmin(dim=0)                                            # [B], first minimum
    x_new = torch.take_along_dim(cand, best.view(1, B, 1), dim=0).squeeze(0)
    res_new = torch.take_along_dim(res, best.view(1, B), dim=0).squeeze(0)
    keep = res_new >= r.norm(dim=-1)
    return torch.where(keep.unsqueeze(-1), x, x_new)


def invert_bilipnet(net, w, n_dys, n_newton=0, damping="none"):
    """x = G^{-1}(w): DYS, then `n_newton` ADDITIONAL Newton polish steps, then always one
    implicit-function-theorem step that carries the derivative.

    The polish's step rule is `damping` (`NEWTON_DAMPINGS`, the chart's `newton_damping`):
    "none" takes the full step `x <- x + J_G(x)^{-1} (w - G(x))`; "backtrack" takes
    `damped_newton_step` (a per-sample choice over `STEP_SIZES`, monotone in the residual).
    The undamped polish was measured to cycle or diverge on rare inputs of a smooth (tanh)
    chart, and the damped one to stall at kinks of a piecewise-linear (relu) chart, so neither
    is right for every chart. Both are graph-safe (static shapes; `damping` is a Python string,
    fixed per chart, so the branch on it is resolved at trace time).

    The DYS iterations and the polish run on `w.detach()` under `torch.no_grad()`, giving x_d.
    The returned point is
        x_out = x_d + J_G(x_d)^{-1} (w - G(x_d)),   x_d and J_G(x_d) detached,
    one more (FULL, undamped) Newton step whose derivative is exactly the IFT one:
    d x_out / d w = J_G(x_d)^{-1}, and, because G(x_d) stays attached to the parameters theta,
    d x_out / d theta = -J_G(x_d)^{-1} dG/dtheta (x_d). The damping touches only the detached
    polish, so this step and its derivative are the same under either rule. Nothing
    differentiates through an iteration, so an outer `jacrev` (the program's) sees a linear
    function of `w` with matrix J_G(x_d)^{-1}. At `n_newton = 0` the step is still taken (the
    DYS point is linearised once).
    """
    if damping not in NEWTON_DAMPINGS:
        raise ValueError(f"newton damping {damping!r} not in {NEWTON_DAMPINGS}")
    with torch.no_grad():
        wd = w.detach()
        x = dys_inverse(net, wd, n_dys)
        for _ in range(n_newton):
            if damping == "backtrack":
                x = damped_newton_step(net, x, wd)
            else:
                J = bilipnet_jacobian(net, x)
                x = x + _solve(J, (wd - net(x)).unsqueeze(-1)).squeeze(-1)
        x = x.detach()
        J = bilipnet_jacobian(net, x).detach()
    return x + _solve(J, (w - net(x)).unsqueeze(-1)).squeeze(-1)


class BiLipNet(nn.Module):
    """G = O_depth o F_{depth-1} o ... o O_1 o F_0 o O_0 (BiLipNet eq. 8), each monotone layer
    at mu_layer = mu_total ** (1/depth), so prod mu = mu_total; nu per layer is learned."""

    def __init__(self, n, units, depth=2, mu_total=0.1, activation="relu",
                 lognu_init=math.log(2.0)):
        super().__init__()
        self.n, self.depth, self.units = int(n), int(depth), tuple(int(u) for u in units)
        self.activation = activation
        mu_layer = float(mu_total) ** (1.0 / self.depth)
        self.uni = nn.ModuleList([Unitary(self.n) for _ in range(self.depth + 1)])
        self.mon = nn.ModuleList([MonLipLayer(self.n, self.units, mu_layer, lognu_init, activation)
                                  for _ in range(self.depth)])

    def forward(self, x):
        for k in range(self.depth):
            x = self.uni[k](x)
            x = self.mon[k](x)
        return self.uni[self.depth](x)

    def bounds(self):
        """(prod mu, prod nu, prod nu/mu) = (lipmin, lipmax, tau), PLNet's `get_bounds`."""
        mu = torch.stack([m.mu for m in self.mon]).prod()
        nu = torch.stack([m.nu() for m in self.mon]).prod()
        return mu, nu, nu / mu

    def check_bounds(self):
        """True (a tensor) when every monotone layer has nu > mu."""
        return torch.stack([m.check_bounds()[0] for m in self.mon]).all()

    def inverse(self, w, n_dys, n_newton=0, damping="none"):
        return invert_bilipnet(self, w, n_dys, n_newton, damping)


def load_plnet_params(net, golden, layer_tau=None):
    """Copy a PLNet flax parameter tree, flattened to `golden["uni_{k}/W"]`-style keys, onto
    `net` (a `BiLipNet`). NO transposes: this module stores every matrix in PLNet's own
    orientation (`x @ R^T` with R = cayley(W) of W as stored; `Fq` (n, sum units); `Fab{j}`
    (nz_j + nz_{j-1}, nz_j)), and each shape is checked. rb20d_solver.py's MonLipNet has
    mu = nu / layer_tau; here mu is a buffer, so it is SET to exp(lognu) / layer_tau
    (`layer_tau` defaults to `golden["layer_tau"]`)."""
    layer_tau = float(golden["layer_tau"]) if layer_tau is None else float(layer_tau)

    def put(param, key):
        src = torch.as_tensor(golden[key], dtype=param.dtype, device=param.device)
        if tuple(src.shape) != tuple(param.shape):
            raise ValueError(f"{key}: golden shape {tuple(src.shape)} != {tuple(param.shape)}")
        param.copy_(src)

    with torch.no_grad():
        for k, uni in enumerate(net.uni):
            for name in ("W", "a", "b"):
                put(getattr(uni, name), f"uni_{k}/{name}")
        for k, mon in enumerate(net.mon):
            names = ["Fq", "fq", "by", "lognu"]
            for j in range(len(mon.units)):
                names += [f"Fab{j}", f"fab{j}", f"b{j}"]
            for name in names:
                put(getattr(mon, name), f"mon_{k}/{name}")
            mon.mu.copy_(torch.exp(mon.lognu).reshape(()) / layer_tau)
    return net


## ---------------------------------------------------------------------- LOInK

class LoinkModel(nn.Module):
    """LOInK's surrogate (arXiv 2609.21275 Sec. IV.A): G : q -> (y_hat, z), eq. (9), and a
    cost J_hat(z, y_hat) = z^T M(y_hat) z + b(y_hat), M = I/2 + P^T P, eq. (10)-(11).

    G is a BiLipNet on the normalised configuration x = q / scale (`scale` a positive (n,)
    buffer, so a model carries the scaling it was trained at). Its first `p` outputs are the
    task estimate y_hat (f_hat = G_1), the last K = n - p the latent z. The head
    H : y_hat -> (b, P) is an MLP p -> head_hidden -> head_hidden -> 1 + K*K with SiLU; the
    paper's "P in R^{p x p}" must be K x K for z^T M z to be defined, so it is K x K here.
    `cost="u"` has no head.

    `forward(z, y)` IS the decoder q = scale * G^{-1}([y, z]), so that a compiled wrapper
    (which compiles `forward` only) compiles the call the program makes, as for the cVAE.
    The task representation and the robot costs are not this class's business: it is
    generic in (n, p).
    """

    def __init__(self, n, p, scale, units=(256,) * 4, depth=2, mu=0.1, activation="relu",
                 cost="m", head_hidden=128, n_dys=100, n_newton=2, newton_damping="none",
                 lognu_init=math.log(2.0)):
        super().__init__()
        if not 0 < p < n:
            raise ValueError(f"need 0 < p < n (a nonempty latent); got p={p}, n={n}")
        if cost not in COSTS:
            raise ValueError(f"cost {cost!r} not in {COSTS}")
        self.n, self.p, self.K = int(n), int(p), int(n) - int(p)
        if newton_damping not in NEWTON_DAMPINGS:
            raise ValueError(f"newton_damping {newton_damping!r} not in {NEWTON_DAMPINGS}")
        self.cost, self.n_dys, self.n_newton = cost, int(n_dys), int(n_newton)
        self.newton_damping = newton_damping
        scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if scale.shape != (self.n,) or not bool((scale > 0).all()):
            raise ValueError(f"scale must be ({self.n},) and positive; got {scale}")
        self.register_buffer("scale", scale)
        self.G = BiLipNet(self.n, units, depth, mu, activation, lognu_init)
        if cost == "u":
            self.H = None
        else:
            self.H = nn.Sequential(nn.Linear(self.p, head_hidden), nn.SiLU(),
                                   nn.Linear(head_hidden, head_hidden), nn.SiLU(),
                                   nn.Linear(head_hidden, 1 + self.K * self.K))

    def forward_G(self, q):
        """(y_hat [B, p], z [B, K]) = G(q / scale)."""
        out = self.G(q / self.scale)
        return out[:, :self.p], out[:, self.p:]

    def encode(self, q):
        return self.forward_G(q)[1]

    def invert(self, q):
        """The chart's inverse direction, q -> (y_hat, z): G itself (explicit)."""
        return self.forward_G(q)

    def decode(self, y, z):
        """q = scale * G^{-1}([y, z]) (eq. 13 at z = 0), derivative exact (`invert_bilipnet`)."""
        w = torch.cat([y, z], dim=-1)
        return invert_bilipnet(self.G, w, self.n_dys, self.n_newton,
                               self.newton_damping) * self.scale

    def forward(self, z, y):
        return self.decode(y, z)

    def cost_hat(self, z, y_hat):
        """J_hat = z^T (I/2 + P^T P) z + b, with (b, P) = H(y_hat), eq. (10)-(11)."""
        h = self.H(y_hat)
        b, P = h[:, 0], h[:, 1:].reshape(-1, self.K, self.K)
        Pz = (P @ z.unsqueeze(-1)).squeeze(-1)
        return 0.5 * (z * z).sum(-1) + (Pz * Pz).sum(-1) + b


def loink_loss(model, q, y, j=None, lam=1.0, tau_penalty=1e-3):
    """LOInK's loss, eq. (14)-(16), plus the distortion penalty of Sec. V.B:
        task_mse = mean_i |y_hat_i - y_i|^2          (eq. 14; summed over task dims)
        aux_mse  = mean_i (J_hat(z_i, y_hat_i) - j_i)^2   (eq. 15; absent when cost == "u")
        dist     = tau_penalty * log(tau),  tau = prod nu / prod mu
        loss     = task_mse + lam * aux_mse + dist   (eq. 16 + dist)
    The paper says only "a small penalty on the distortion"; log tau is this port's choice.
    Returns `(loss, parts)`; every part is a tensor (no `.item()`)."""
    y_hat, z = model.forward_G(q)
    task_mse = ((y_hat - y) ** 2).sum(-1).mean()
    _, nu, tau = model.G.bounds()
    dist = tau_penalty * torch.log(tau)
    loss = task_mse + dist
    parts = {"task_mse": task_mse, "task_rmse": torch.sqrt(task_mse), "dist": dist, "nu": nu,
             "tau": tau}
    if model.H is not None:
        aux_mse = ((model.cost_hat(z, y_hat) - j) ** 2).mean()
        loss = loss + lam * aux_mse
        parts["aux_mse"] = aux_mse
    parts["loss"] = loss
    return loss, parts


## ====================================================================== the chart family
##
## Everything below makes `LoinkModel` a chart of the learned formulation, the way
## `src/charts/cvae.py` does for the cVAE: the task representation, the training costs, the
## `Chart` wrapper, the mandatory sidecar and the loader. Imported here rather than at the top
## so that the core above stays exactly the port the golden check pins.

import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

from src.charts.base import Chart  # noqa: E402

## `src.flow_loading` (SidecarPath, WriteArch) is imported inside the functions that need it:
## it imports ikflow and jrl, and jrl sets torch's default device to CUDA at import, which
## must not happen to a script that only wants the network core above.

#: The task the chart's first `p` outputs fit: `y = [x, y, z, roll, pitch, yaw]`, the
#: program's own `c` variables (Drake `RollPitchYaw`). A stated choice: there is no
#: continuous 6-parameter chart of SE(3), so the rpy cut (roll or yaw = +-pi, pitch = +-pi/2)
#: is an irreducible floor of a Lipschitz map near it; `screen_loink.py` (e) measures it.
TASK_DIM = 6

#: The LOInK eq. 18 barrier's floor, `-log(max(1 - q_hat^2, eps))`.
BARRIER_EPS = 1e-3

#: The fields that define the chart's forward pass and its cost semantics. Everything else
#: in a sidecar (training recipe, loss weights, provenance) is carried for the record only.
ARCH_FIELDS = ("ndof", "task_dim", "dim_latent_space", "units", "depth", "mu", "activation",
               "cost", "head_hidden", "j_mean", "j_std", "n_dys", "n_newton", "newton_damping")

#: ARCH_FIELDS a sidecar written before the field existed may lack, and the value it meant.
ARCH_FIELD_DEFAULTS = {"newton_damping": "none"}

#: Sidecar fields that are a RECORD, not architecture (LOInK phase 2, 2026-10-24): the IK branch
#: a single-branch chart trained on (`src/branches.py`, a list of three ints) and its dataset's
#: name. No weight depends on them, so nothing cross-checks them; `LoadLoinkChart` copies them
#: onto the chart (`LoinkChart.branch` / `.dataset_name`) when the sidecar names them.
RECORD_FIELDS = ("branch", "dataset_name")

#: The one architecture of the campaign (no ladder): the paper's manipulator example.
DEFAULT_ARCH = {"units": (256, 256, 256, 256), "depth": 2, "mu": 0.1, "head_hidden": 128,
                "n_dys": 100, "n_newton": 2, "newton_damping": "none"}


## ---------------------------------------------------------------------- task representation

def Pose7ToXyzRpy(pose7):
    """`[B, 7]` `[x, y, z, qw, qx, qy, qz]` -> `[B, 6]` `[x, y, z, roll, pitch, yaw]`.

    Drake's convention, `R = Rz(yaw) Ry(pitch) Rx(roll)` (extrinsic x-y-z), by the standard
    atan2 formulas on the rotation matrix of the normalised quaternion:
        roll = atan2(R21, R22), pitch = atan2(-R20, hypot(R00, R10)), yaw = atan2(R10, R00),
    so roll, yaw in (-pi, pi] and pitch in [-pi/2, pi/2] -- the principal domain Drake's
    `RollPitchYaw(RotationMatrix(q)).vector()` returns (matched to 1e-12 by
    `tests/test_loink_chart.py`). The quaternion's sign does not matter. Smooth away from
    the cut; at gimbal lock (|pitch| = pi/2 exactly) its derivative is not finite.
    """
    xyz, q = pose7[:, :3], pose7[:, 3:7]
    q = q / torch.linalg.norm(q, dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    R00 = 1 - 2 * (y * y + z * z)
    R10 = 2 * (x * y + w * z)
    R20 = 2 * (x * z - w * y)
    R21 = 2 * (y * z + w * x)
    R22 = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(R21, R22)
    pitch = torch.atan2(-R20, torch.sqrt(R00 * R00 + R10 * R10))
    yaw = torch.atan2(R10, R00)
    return torch.cat([xyz, torch.stack([roll, pitch, yaw], dim=1)], dim=1)


def XyzRpyToPose7(y):
    """`[B, 6]` `[x, y, z, roll, pitch, yaw]` -> `[B, 7]` `[x, y, z, qw, qx, qy, qz]`, the
    quaternion canonicalised to `w >= 0` as Drake's `ToQuaternion` returns it
    (`src.svgd.batched_flow.quat_from_rpy_batched`, the tree's one rpy -> quaternion map).
    Defined for ANY rpy; `Pose7ToXyzRpy(XyzRpyToPose7(y)) == y` only on the principal domain.
    """
    from src.svgd.batched_flow import quat_from_rpy_batched
    return torch.cat([y[:, :3], quat_from_rpy_batched(y[:, 3:6])], dim=1)


def RpyCutDistance(rpy):
    """Distance in radians to the nearest rpy cut, `min(pi - |roll|, pi - |yaw|, pi/2 - |pitch|)`."""
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    return torch.minimum(torch.minimum(math.pi - roll.abs(), math.pi - yaw.abs()),
                         math.pi / 2 - pitch.abs())


## ---------------------------------------------------------------------- training costs

def NominalConfiguration(robot_name, ndof):
    """The program's `NominalConfiguration()` constant for `robot_name`, without a program:
    Panda / iiwa from `src.nominal_configurations` (what their programs read), the screw arm
    from its spec, the soft PCS arms zeros (the generic program's default)."""
    if robot_name == "panda":
        from src.nominal_configurations import PANDA_Q_NOMINAL as q
    elif robot_name == "iiwa14":
        from src.nominal_configurations import IIWA_Q_NOMINAL as q
    elif robot_name.startswith("screw7_"):
        from src.screw_arm.params import GetSpec
        q = GetSpec(robot_name).q_nominal
    elif re.fullmatch(r"soft\d+", robot_name):
        q = np.zeros(ndof)
    else:
        raise ValueError(f"no nominal configuration known for {robot_name!r} "
                         "(panda, iiwa14, screw7_*, soft*)")
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (ndof,):
        raise ValueError(f"{robot_name}: nominal has shape {q.shape}, expected ({ndof},)")
    return q


class _TrainingCost:
    """`q [B, n] -> j [B]`, in configuration units; constants cast to `q`'s dtype/device."""

    def __init__(self, kind, q_nom=None, mid=None, half=None, eps=BARRIER_EPS):
        self.kind, self.eps = kind, float(eps)
        self.q_nom, self.mid, self.half = q_nom, mid, half

    def __call__(self, q):
        if self.kind == "m":
            d = q - self.q_nom.to(dtype=q.dtype, device=q.device)
            return 0.5 * (d * d).sum(-1)
        q_hat = (q - self.mid.to(dtype=q.dtype, device=q.device)) / self.half.to(
            dtype=q.dtype, device=q.device)
        return -torch.log(torch.clamp(1 - q_hat * q_hat, min=self.eps)).sum(-1)


def TrainingCost(robot_name, kind, robot=None):
    """The cost a `kind` chart's head is trained to fit, or None for `kind == "u"`.

      * `m` (matched): `0.5 |q - q_nom|^2`, the program's reported joint-centering cost, with
        `q_nom` = `NominalConfiguration(robot_name)` (soft PCS: `0.5 |strain|^2`).
      * `x` (mismatched): LOInK eq. 18, `-sum_i log(max(1 - q_hat_i^2, eps))`, `eps = 1e-3`,
        on `q_hat = (q - mid) / half` from the robot's joint limits (soft PCS: +-1).
      * `u` (unstructured): no head, no cost -- `None`.

    Both in the configuration units the dataset holds (the soft PCS dataset is normalised
    strain already). `robot` (a jrl `Robot`) is resolved by name when not given.
    """
    if kind not in COSTS:
        raise ValueError(f"cost {kind!r} not in {COSTS}")
    if kind == "u":
        return None
    if robot is None:
        from jrl.robots import get_robot
        robot = get_robot(robot_name)
    ndof = int(robot.ndof)
    if kind == "m":
        q_nom = torch.tensor(NominalConfiguration(robot_name, ndof), dtype=torch.float64,
                             device="cpu")
        return _TrainingCost("m", q_nom=q_nom)
    lo, hi = (torch.tensor(v, dtype=torch.float64, device="cpu")
              for v in zip(*robot.actuated_joints_limits))
    return _TrainingCost("x", mid=0.5 * (lo + hi), half=0.5 * (hi - lo))


## ---------------------------------------------------------------------- the chart

def LoinkDecode(model, z, c6):
    """`(q [B, ndof], None)`: `model.forward(z, c6)` = scale * G^{-1}([c6, z]) at the
    program's RAW task `c6 = [xyz, rpy]` (`task_input = "xyzrpy"`): no rpy -> quaternion ->
    rpy round trip, so the input is continuous where `c`'s roll crosses +-pi (G is defined on
    all of R^n; off the principal domain it extrapolates, the cut's floor, screen (e)).
    Through `forward` so a compiled wrapper compiles this call (`LoinkModel`)."""
    return model(z, c6), None


def LoinkEncode(model, q, c6):
    """`z = G_2(q / scale)`: exact, and the task is IGNORED -- the chart's own task estimate
    `y_hat = G_1(q)` is what reproduces `q`, which `LoinkInvert` returns and `SetStartFromQ`
    uses. Decoding this `z` at any other task (the FK pose's) gives a different `q`."""
    return model.encode(q)


def LoinkInvert(model, q):
    """`(c6 [B, 6], z [B, K]) = G(q / scale)`, `c6` the RAW `y_hat` (not wrapped to the rpy
    principal domain), so `LoinkDecode(model, z, c6) == q` for EVERY `q`."""
    return model.invert(q)


class LoinkChart(Chart):
    """LOInK as a chart: `decode` is `G^{-1}` at the program's raw `c = [xyz, rpy]`
    (`task_input = "xyzrpy"`), `encode` is `G_2`, and `invert(model, q) -> (c6, z)` is the
    chart's own exact inverse (`src/charts/base.py`). Screens holding a dataset pose7 form
    the input with `ChartTaskFromPose7` (`Pose7ToXyzRpy`).

    `decode` / `encode` / `invert` are module-level functions held as plain instance
    attributes -- not bound methods -- for the reason `FlowChart` gives: `MakeFlowInference`
    closes over `decode` and torch.compile guards on what a compiled closure holds, so it must
    carry no reference to this object (a no-argument `functools.partial` would be the same).

    Compile and CUDA graphs: the Davis-Yin loop is `n_dys` fixed iterations per monotone
    layer with no data-dependent control flow, so dynamo unrolls it and a CUDA graph captures
    it (a graph replays whatever kernels ran during capture). Measured on the laptop CPU
    (2026-10-22, untrained Panda chart, float64, 4 threads; `FlowJacobianGen(compile_it=True)`
    compile + first call, then steady state against eager): relu 33 s at n_dys 2, 38 s at
    10, 114 s at 100 (19 ms v 121 ms eager; 103 s re-measured through the raw-c6 path); tanh 163 s at n_dys 20, 566 s at 100 (83 ms v
    213 ms) -- the tanh prox's inner 8-step solve multiplies the unrolled graph. Jacobians
    equal eager to 1e-14. `flow_cuda_graph` compiles a second, value-only function on top.
    The lever on compile time is `n_dys` itself (`screen_loink.py` (b): the Newton polish
    makes the value exact at small counts inside the basin). Wrapping the DYS loop in
    `torch.compiler.disable` is NOT a fallback: dynamo cannot break a graph inside a
    `torch.func` transform, so the whole compiled `jacrev` silently runs eager (measured:
    0.4 s "compile", steady time = eager, bit-identical).
    """

    family = FAMILY
    has_density = False
    exact_inverse = True
    task_input = "xyzrpy"

    def __init__(self, nn_model, robot, arch, arch_source=None):
        self.nn_model = nn_model
        self.robot = robot
        self.arch = dict(arch)
        self.arch_source = arch_source
        ## Record only (LOInK phase 2, 2026-10-24): the IK branch a single-branch chart was
        ## trained on (`src/branches.py`; the signs, None for an unrestricted chart) and the
        ## dataset it trained on (None when the sidecar predates the field). Nothing in the
        ## program reads them; they reach `metadata.chart_arch` through `arch`, and the
        ## exporter's gate and the screens restrict their draws by `branch`.
        self.branch = arch.get("branch")
        self.dataset_name = arch.get("dataset_name")
        self.network_width = int(arch["dim_latent_space"])
        self.ndof = int(robot.ndof)
        self.decode = LoinkDecode
        self.encode = LoinkEncode
        self.invert = LoinkInvert

    def cost_hat_raw(self, z, y_hat, model=None):
        """The head's cost estimate in the training cost's own units,
        `j_mean + j_std * J_hat(z, y_hat)` (the head fits the standardised cost)."""
        if self.arch["cost"] == "u":
            raise ValueError("a cost-'u' LOInK chart has no cost head")
        model = self.nn_model if model is None else model
        return self.arch["j_mean"] + self.arch["j_std"] * model.cost_hat(z, y_hat)


## ---------------------------------------------------------------------- sidecar and loading

def WriteLoinkArch(checkpoint_pkl, arch, provenance=None):
    """Write `<stem>.arch.json` for a LOInK chart, with `"family": "loink"`."""
    from src.flow_loading import WriteArch
    missing = [k for k in ARCH_FIELDS if k not in arch]
    if missing:
        raise ValueError(f"LOInK arch is missing {missing}")
    arch = dict(arch, units=[int(u) for u in arch["units"]])
    return WriteArch(checkpoint_pkl, dict(arch, family=FAMILY), provenance=provenance)


def _Indexed(state_dict, pattern):
    return sorted({int(m.group(1)) for k in state_dict if (m := re.fullmatch(pattern, k))})


def ArchFromLoinkStateDict(state_dict):
    """Recover every field the weights determine from a raw `LoinkModel` state dict.

    From SHAPES: `ndof` (`G.uni.0.W`), `depth` (the number of `G.mon.*`), `units` (each
    `G.mon.k.Fab{j}`), and, when a cost head is present, `task_dim` and `head_hidden`
    (`H.0.weight`) and `dim_latent_space` (`H.4.weight` has 1 + K^2 rows). From a BUFFER'S
    VALUE: `mu`, the product of the per-layer `G.mon.k.mu` (`load_state_dict` would install it
    whatever the sidecar said, so it is checked rather than trusted). Whether there is a head
    (`has_head`, i.e. cost "u" or not) from the keys.

    What NO check can catch -- the reason the sidecar is mandatory, as `rnvp_clamp` is for
    flows (`src/flow_loading.py`) and `activation` for the cVAE: `activation` (relu and tanh
    weights are identical in shape), the cost letter between "m" and "x" (the same head fits
    either target), `j_mean` / `j_std` (the head's standardisation, which only `cost_hat_raw`
    reads), `n_dys` / `n_newton` / `newton_damping` (iteration counts and step rule of the
    inverse), and, on a cost-"u" chart, `task_dim` / `head_hidden`.
    """
    unis = _Indexed(state_dict, r"G\.uni\.(\d+)\.W")
    mons = _Indexed(state_dict, r"G\.mon\.(\d+)\.Fq")
    if not unis or not mons:
        raise ValueError("state dict has no 'G.uni.*.W' / 'G.mon.*.Fq' keys -- not a LOInK chart")
    problems = []
    ndof = int(state_dict["G.uni.0.W"].shape[0])
    depth = len(mons)
    if mons != list(range(depth)) or unis != list(range(depth + 1)):
        problems.append(f"unexpected layer layout: uni {unis}, mon {mons}")
    units = None
    for k in mons:
        blocks = _Indexed(state_dict, rf"G\.mon\.{k}\.Fab(\d+)")
        u = tuple(int(state_dict[f"G.mon.{k}.Fab{j}"].shape[1]) for j in blocks)
        nz_1 = 0
        for j, nz in zip(blocks, u):
            shape = tuple(state_dict[f"G.mon.{k}.Fab{j}"].shape)
            if shape != (nz + nz_1, nz):
                problems.append(f"G.mon.{k}.Fab{j} has shape {shape}, expected ({nz + nz_1}, {nz})")
            nz_1 = nz
        if tuple(state_dict[f"G.mon.{k}.Fq"].shape) != (ndof, sum(u)):
            problems.append(f"G.mon.{k}.Fq has shape {tuple(state_dict[f'G.mon.{k}.Fq'].shape)}, "
                            f"expected ({ndof}, {sum(u)})")
        if units is None:
            units = u
        elif u != units:
            problems.append(f"G.mon.{k} units {u} differ from G.mon.0's {units}")
    for k in unis:
        if tuple(state_dict[f"G.uni.{k}.W"].shape) != (ndof, ndof):
            problems.append(f"G.uni.{k}.W is not ({ndof}, {ndof})")
    mu = float(np.prod([float(state_dict[f"G.mon.{k}.mu"]) for k in mons]))
    if "scale" not in state_dict or tuple(state_dict["scale"].shape) != (ndof,):
        problems.append(f"no 'scale' tensor of shape ({ndof},)")
    out = {"ndof": ndof, "depth": depth, "units": list(units or ()), "mu": mu,
           "has_head": "H.0.weight" in state_dict}
    if out["has_head"]:
        p, hh = int(state_dict["H.0.weight"].shape[1]), int(state_dict["H.0.weight"].shape[0])
        rows = int(state_dict["H.4.weight"].shape[0])
        k = ndof - p
        if rows != 1 + k * k:
            problems.append(f"H.4.weight has {rows} rows, expected 1 + K^2 = {1 + k * k} "
                            f"(K = ndof {ndof} - task_dim {p})")
        out.update(task_dim=p, head_hidden=hh, dim_latent_space=k)
    if problems:
        raise ValueError("not a well-formed LOInK state dict:\n  " + "\n  ".join(problems))
    return out


def ReadLoinkArch(checkpoint):
    """Read the MANDATORY sidecar of a LOInK checkpoint; raise if absent or not a LOInK's."""
    from src.flow_loading import SidecarPath
    sidecar = SidecarPath(checkpoint)
    if not os.path.exists(sidecar):
        raise FileNotFoundError(
            f"no architecture sidecar at {sidecar}. A LOInK chart has no legacy fallback: the "
            "activation, the cost letter, the head's standardisation and the inverse's "
            "iteration counts and step rule cannot be recovered from the weights. Export it with "
            "scripts/training/export_loink.py.")
    with open(sidecar) as f:
        payload = json.load(f)
    payload = {**ARCH_FIELD_DEFAULTS, **payload}
    if payload.get("family") != FAMILY:
        raise ValueError(f"{sidecar} declares family {payload.get('family')!r}, not {FAMILY!r}")
    missing = [k for k in ARCH_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"{sidecar} is missing the LOInK arch fields {missing}")
    return payload, sidecar


def VerifyLoinkArch(declared, state_dict, source):
    """Raise if the sidecar disagrees with what the weights imply, or with itself."""
    actual = ArchFromLoinkStateDict(state_dict)
    d = dict(declared)
    mismatches = []
    for key in ("ndof", "depth"):
        if int(d[key]) != actual[key]:
            mismatches.append(f"{key}: sidecar says {d[key]!r}, weights imply {actual[key]!r}")
    if [int(u) for u in d["units"]] != actual["units"]:
        mismatches.append(f"units: sidecar says {list(d['units'])}, weights imply {actual['units']}")
    if not math.isclose(float(d["mu"]), actual["mu"], rel_tol=1e-6):
        mismatches.append(f"mu: sidecar says {d['mu']!r}, the weights' mu buffers multiply "
                          f"to {actual['mu']!r}")
    if d["cost"] not in COSTS:
        mismatches.append(f"cost: {d['cost']!r} not in {COSTS}")
    elif (d["cost"] != "u") != actual["has_head"]:
        mismatches.append(f"cost: sidecar says {d['cost']!r}, the weights "
                          f"{'have' if actual['has_head'] else 'have no'} cost head")
    if d["activation"] not in ACTIVATIONS:
        mismatches.append(f"activation: {d['activation']!r} not in {ACTIVATIONS}")
    if int(d["task_dim"]) != TASK_DIM:
        mismatches.append(f"task_dim: sidecar says {d['task_dim']!r}; the chart's task is "
                          f"xyz + rpy, {TASK_DIM}")
    if int(d["dim_latent_space"]) != int(d["ndof"]) - int(d["task_dim"]):
        mismatches.append(f"dim_latent_space: sidecar says {d['dim_latent_space']!r}, but "
                          f"ndof - task_dim = {int(d['ndof']) - int(d['task_dim'])}")
    if actual["has_head"]:
        for key in ("task_dim", "head_hidden", "dim_latent_space"):
            if int(d[key]) != actual[key]:
                mismatches.append(f"{key}: sidecar says {d[key]!r}, weights imply {actual[key]!r}")
        if not (d["j_mean"] is not None and d["j_std"] is not None and float(d["j_std"]) > 0):
            mismatches.append(f"j_mean / j_std: a cost head needs a finite standardisation, "
                              f"got {d['j_mean']!r} / {d['j_std']!r}")
    if int(d["n_dys"]) < 1 or int(d["n_newton"]) < 0:
        mismatches.append(f"n_dys {d['n_dys']!r} / n_newton {d['n_newton']!r} out of range")
    if d.get("newton_damping", "none") not in NEWTON_DAMPINGS:
        mismatches.append(f"newton_damping: {d['newton_damping']!r} not in {NEWTON_DAMPINGS}")
    if mismatches:
        raise ValueError(
            f"architecture in {source} does not match the weights:\n  " + "\n  ".join(mismatches)
            + "\nLoading this checkpoint would silently produce a different chart.")


def BuildLoinkModel(arch, scale):
    """An (untrained) `LoinkModel` for `arch`, on the CPU (jrl may have made CUDA the default)."""
    with torch.device("cpu"):
        return LoinkModel(int(arch["ndof"]), int(arch["task_dim"]), scale,
                          units=tuple(int(u) for u in arch["units"]), depth=int(arch["depth"]),
                          mu=float(arch["mu"]), activation=arch["activation"],
                          cost=arch["cost"], head_hidden=int(arch["head_hidden"]),
                          n_dys=int(arch["n_dys"]), n_newton=int(arch["n_newton"]),
                          newton_damping=arch.get("newton_damping", "none"))


def LoadLoinkChart(robot_name, checkpoint, compile_model=None):
    """Build a `LoinkChart` from a bare state-dict `.pkl` and its mandatory sidecar.

    Every field the weights determine is cross-checked and a contradiction RAISES
    (`VerifyLoinkArch`). As for the cVAE: the sidecar's `robot_name` (when present) must be the
    one asked for, and the `scale` buffer must equal the robot's own joint scaling
    (`src.charts.cvae.JointScale`) -- what stops a Panda chart loading against the 7-DoF screw
    arm, which no shape check can tell apart.
    """
    from jrl.config import DEVICE
    from jrl.robots import get_robot

    from src.charts.cvae import JointScale

    payload, source = ReadLoinkArch(checkpoint)
    if payload.get("robot_name") not in (None, robot_name):
        raise ValueError(f"{source} was trained for {payload['robot_name']!r}, "
                         f"not {robot_name!r}")
    arch = {k: payload[k] for k in ARCH_FIELDS}
    ## Record-only sidecar fields carried onto the chart when present (2026-10-24), so an
    ## unrestricted chart's `arch` (and the metadata built from it) is unchanged.
    record = {k: payload[k] for k in RECORD_FIELDS if payload.get(k) is not None}

    with open(checkpoint, "rb") as f:
        state_dict = pickle.load(f)
    VerifyLoinkArch(arch, state_dict, source)

    robot = get_robot(robot_name)
    if int(robot.ndof) != int(arch["ndof"]):
        raise ValueError(f"{source}: chart has ndof {arch['ndof']}, robot {robot_name!r} "
                         f"has {robot.ndof}")
    expected_scale = JointScale(robot)
    if not torch.allclose(state_dict["scale"].float().cpu(), expected_scale, rtol=0, atol=1e-6):
        raise ValueError(f"{source}: the chart's joint scale {state_dict['scale'].tolist()} is "
                         f"not robot {robot_name!r}'s {expected_scale.tolist()} -- trained for "
                         "a different robot?")

    model = BuildLoinkModel(arch, expected_scale)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()
    nn_model = torch.compile(model, **compile_model) if compile_model is not None else model
    return LoinkChart(nn_model, robot, dict(arch, family=FAMILY, robot_name=robot_name, **record),
                      arch_source=source)
