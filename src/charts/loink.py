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
`tests/data/loink_reference/`). This file holds ONLY the network core, its inversion, the
LOInK model and its loss; the chart wrapper, sidecar and loader come later.

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
  * `invert_bilipnet`: the DYS result, `n_newton` Newton polish steps on the n-dim map, and
                    one implicit-function-theorem linearisation that makes its derivative exact.
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
linear solves use `torch.linalg.solve_ex(check_errors=False)`, which unlike
`torch.linalg.solve` does not synchronise on the info check (graph capture). No
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
    """`A^{-1} B` (left) or `B A^{-1}` (left=False), without the info-check sync."""
    return torch.linalg.solve_ex(A, B, left=left, check_errors=False)[0]


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


def invert_bilipnet(net, w, n_dys, n_newton=0):
    """x = G^{-1}(w): DYS, then `n_newton` ADDITIONAL Newton polish steps, then always one
    implicit-function-theorem step that carries the derivative.

    The DYS iterations and the polish run on `w.detach()` under `torch.no_grad()`, giving x_d.
    The returned point is
        x_out = x_d + J_G(x_d)^{-1} (w - G(x_d)),   x_d and J_G(x_d) detached,
    one more Newton step whose derivative is exactly the IFT one: d x_out / d w = J_G(x_d)^{-1},
    and, because G(x_d) stays attached to the parameters theta, d x_out / d theta =
    -J_G(x_d)^{-1} dG/dtheta (x_d). Nothing differentiates through an iteration, so an outer
    `jacrev` (the program's) sees a linear function of `w` with matrix J_G(x_d)^{-1}. At
    `n_newton = 0` the step is still taken (the DYS point is linearised once).
    """
    with torch.no_grad():
        wd = w.detach()
        x = dys_inverse(net, wd, n_dys)
        for _ in range(n_newton):
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

    def inverse(self, w, n_dys, n_newton=0):
        return invert_bilipnet(self, w, n_dys, n_newton)


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
                 cost="m", head_hidden=128, n_dys=100, n_newton=2, lognu_init=math.log(2.0)):
        super().__init__()
        if not 0 < p < n:
            raise ValueError(f"need 0 < p < n (a nonempty latent); got p={p}, n={n}")
        if cost not in COSTS:
            raise ValueError(f"cost {cost!r} not in {COSTS}")
        self.n, self.p, self.K = int(n), int(p), int(n) - int(p)
        self.cost, self.n_dys, self.n_newton = cost, int(n_dys), int(n_newton)
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
        return invert_bilipnet(self.G, w, self.n_dys, self.n_newton) * self.scale

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
