"""`src/svgd/al.py` and `src/svgd/kernels.py` on synthetic problems.

The two modules are pure tensor math on per-particle quantities and know nothing about any
formulation, so they are tested on problems whose answers are known in closed form: a PHR
augmented Lagrangian against autograd and against the quadratic penalty it reduces to; the
Nocedal-Wright 17.4 updates against the KKT point of an equality-constrained quadratic, one
quadratic per particle; Gauss-Newton on linear rows (exact in one step) and on a circle
(quadratic convergence); the tangent projector's algebra; the RBF repulsion against autograd
and, the check that pins its SIGN, plain SVGD on a 2-D standard Gaussian matching the
target's mean and covariance -- then collapsing onto the mode once the repulsion
temperature is annealed to exactly zero. `adam_step` is held against `torch.optim.Adam`.

Everything runs on CPU float64 and, where a CUDA device exists, on CUDA float32 with the
tolerances scaled to that precision. The last test compiles a composition of the AL
functions with `torch.compile(dynamic=False)` and checks it against eager, and greps both
modules for the host-synchronising calls (`.item()`, `.tolist()`, `.numpy()`, `.cpu()`)
that would break CUDA-graph capture.

No pytest config in this repo: plain `test_*` functions with a `__main__` driver.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.svgd import al, kernels
from src.svgd.al import ALState

HERE = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src", "svgd")


def _targets():
    """(dtype, device, tol) triples to run every test on."""
    out = [(torch.float64, torch.device("cpu"), 1e-12)]
    if torch.cuda.is_available():
        out.append((torch.float32, torch.device("cuda"), 1e-5))
    return out


def _gen(seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _randn(*shape, dtype, device, gen):
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype=dtype, device=device)


def _maxabs(a, b=None):
    d = a if b is None else a - b
    return float(d.detach().abs().max()) if d.numel() else 0.0


def _state(N, n, m_e, m_i, dtype, device, gen, rho0=2.0, eta0=1.0, lr0=0.1, random_mult=True):
    S = ALState.init(N, n, m_e, m_i, rho0=rho0, eta0=eta0, lr0=lr0, dtype=dtype, device=device)
    if random_mult:
        S = al.replace(S, lam=_randn(N, m_e, dtype=dtype, device=device, gen=gen),
                       mu=_randn(N, m_i, dtype=dtype, device=device, gen=gen).abs(),
                       rho=0.5 + 3.0 * torch.rand(N, generator=gen, dtype=torch.float64).to(
                           dtype=dtype, device=device))
    return S


## --------------------------------------------------------------------------------------
## 1. The PHR augmented Lagrangian
## --------------------------------------------------------------------------------------

def test_al_value_and_coefficients_agree_with_autograd():
    for dtype, device, tol in _targets():
        gen = _gen(1, device)
        N, n, m_e, m_i = 17, 5, 3, 4
        S = _state(N, n, m_e, m_i, dtype, device, gen)
        F = _randn(N, dtype=dtype, device=device, gen=gen)
        h = _randn(N, m_e, dtype=dtype, device=device, gen=gen).requires_grad_(True)
        g = _randn(N, m_i, dtype=dtype, device=device, gen=gen).requires_grad_(True)
        L = al.al_value(F, h, g, S)
        assert L.shape == (N,)
        dh, dg = torch.autograd.grad(L.sum(), (h, g))
        ch, cg = al.al_constraint_grad_coefficients(h.detach(), g.detach(), S)
        err = max(_maxabs(dh, ch), _maxabs(dg, cg))
        assert err <= tol * 10, f"{dtype} {device}: coefficient error {err:.2e}"
        # Hand-written L against the module's, including the active-set switch.
        rho = S.rho.unsqueeze(1)
        hand = (F + (S.lam * h).sum(1) + 0.5 * S.rho * (h * h).sum(1)
                + ((torch.clamp(S.mu + rho * g, min=0) ** 2 - S.mu ** 2).sum(1)) / (2 * S.rho))
        assert _maxabs(L, hand) <= tol * 10
        print(f"     {str(dtype):14s} {device.type}: dL/dh, dL/dg vs autograd {err:.1e}")
    print("PASS al_value and its row coefficients agree with autograd")


def test_al_with_zero_multipliers_is_the_quadratic_penalty():
    for dtype, device, tol in _targets():
        gen = _gen(2, device)
        N, n, m_e, m_i = 11, 4, 2, 5
        rho0 = 3.5
        S = ALState.init(N, n, m_e, m_i, rho0=rho0, eta0=1.0, lr0=0.1, dtype=dtype, device=device)
        lam = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        S = al.replace(S, lam=lam)  # mu stays 0
        F = _randn(N, dtype=dtype, device=device, gen=gen)
        h = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        g = _randn(N, m_i, dtype=dtype, device=device, gen=gen)
        L = al.al_value(F, h, g, S)
        pen = (F + (lam * h).sum(1) + 0.5 * rho0 * (h * h).sum(1)
               + 0.5 * rho0 * (torch.clamp(g, min=0) ** 2).sum(1))
        assert _maxabs(L, pen) <= tol * 10, f"{_maxabs(L, pen):.2e}"
    print("PASS mu = 0 reduces PHR to the quadratic penalty F + lam.h + rho/2 |h|^2 + rho/2 |g+|^2")


## --------------------------------------------------------------------------------------
## 2. Multiplier updates
## --------------------------------------------------------------------------------------

def test_update_multipliers_takes_the_right_branch_per_particle():
    for dtype, device, tol in _targets():
        gen = _gen(3, device)
        N, n, m_e, m_i = 20, 3, 2, 3
        S = ALState.init(N, n, m_e, m_i, rho0=2.0, eta0=0.5, lr0=0.1, dtype=dtype, device=device)
        lam = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        mu = _randn(N, m_i, dtype=dtype, device=device, gen=gen).abs()
        S = al.replace(S, lam=lam, mu=mu)
        # Even particles: rows well inside eta. Odd particles: a row at 10x eta.
        h = 0.1 * _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        g = -0.1 * _randn(N, m_i, dtype=dtype, device=device, gen=gen).abs() - 0.1
        odd = torch.arange(N, device=device) % 2 == 1
        h = torch.where(odd.unsqueeze(1), h + 5.0, h)
        v = al.infeasibility(h, g, S)
        ok = v <= S.eta
        assert bool((ok == ~odd).all()), "the constructed branch split did not come out as intended"

        rho_growth, rho_max, mult_max = 10.0, 15.0, 1.0
        S2 = al.update_multipliers(h, g, S, rho_growth, rho_max, mult_max)
        rho = S.rho.unsqueeze(1)
        lam_exp = torch.where(ok.unsqueeze(1), torch.clamp(lam + rho * h, -mult_max, mult_max), lam)
        mu_exp = torch.where(ok.unsqueeze(1), torch.clamp(mu + rho * g, 0.0, mult_max), mu)
        rho_exp = torch.where(ok, S.rho, torch.clamp(S.rho * rho_growth, max=rho_max))
        eta_exp = torch.where(ok, S.eta / S.rho ** 0.9, rho_exp ** -0.1)
        for name, got, exp in (("lam", S2.lam, lam_exp), ("mu", S2.mu, mu_exp),
                               ("rho", S2.rho, rho_exp), ("eta", S2.eta, eta_exp)):
            assert _maxabs(got, exp) <= tol * 10, f"{name}: {_maxabs(got, exp):.2e}"
        # Updated particles are clipped; the others keep their multipliers untouched.
        assert float(S2.lam[~odd].abs().max()) <= mult_max and float(S2.mu[~odd].max()) <= mult_max
        assert bool((S2.lam[odd] == lam[odd]).all()) and bool((S2.mu[odd] == mu[odd]).all())
        assert float(S2.mu.min()) >= 0.0
        assert float(S2.rho.max()) <= rho_max
        assert bool((S2.rho[odd] == rho_max).all()), "rho of the failing half must be capped at rho_max"
        assert bool((S2.rho[~odd] == S.rho[~odd]).all())
        # Clipping actually bit on at least one multiplier (so the clamp was exercised).
        assert bool(((lam + rho * h).abs() > mult_max)[~odd].any())
    print("PASS update_multipliers: per-particle branch, clipped multipliers, capped rho")


def _toy_equality_qp(N, n, m, dtype, device, gen):
    """min 1/2 |x - a_i|^2 s.t. B x = c, a different `a_i` per particle, shared (B, c).

    `B` has orthonormal rows scaled by 0.5, so `grad_x L` has Lipschitz constant
    `1 + 0.25 rho` and plain gradient descent is a usable inner solver at every `rho` the
    test reaches; the multipliers' per-round contraction is `1 / (1 + 0.25 rho)`.
    """
    a = _randn(N, n, dtype=torch.float64, device="cpu", gen=gen)
    Q, _ = torch.linalg.qr(_randn(n, m, dtype=torch.float64, device="cpu", gen=gen))
    B = 0.5 * Q.T
    c = _randn(m, dtype=torch.float64, device="cpu", gen=gen)
    # KKT: x* = a - B^T (B B^T)^-1 (B a - c);  lam* = (B B^T)^-1 (B a - c)
    BBt = B @ B.T
    lam_star = torch.linalg.solve(BBt, (a @ B.T - c).T).T
    x_star = a - lam_star @ B
    cast = lambda t: t.to(dtype=dtype, device=device)
    return cast(a), cast(B), cast(c), cast(x_star), cast(lam_star)


def test_al_rounds_converge_to_the_kkt_point_on_every_particle():
    for dtype, device, tol in _targets():
        gen = _gen(4, device)
        N, n, m = 50, 6, 2
        a, B, c, x_star, lam_star = _toy_equality_qp(N, n, m, dtype, device, gen)
        S = ALState.init(N, n, m, 0, rho0=10.0, eta0=1.0, lr0=0.1, dtype=dtype, device=device)
        x = torch.zeros(N, n, dtype=dtype, device=device)
        Bn2 = float(torch.linalg.matrix_norm(B, ord=2) ** 2)
        g = torch.zeros(N, 0, dtype=dtype, device=device)
        rounds, inner = 30, 300
        for _ in range(rounds):
            # Inner: gradient descent on L with a per-particle step 1 / (1 + rho |B|^2),
            # the Lipschitz constant of grad_x L for this quadratic.
            step = (1.0 / (1.0 + S.rho * Bn2)).unsqueeze(1)
            for _ in range(inner):
                h = x @ B.T - c
                ch, _ = al.al_constraint_grad_coefficients(h, g, S)
                grad = (x - a) + ch @ B
                x = x - step * grad
            h = x @ B.T - c
            S = al.update_multipliers(h, g, S, rho_growth=10.0, rho_max=100.0, multiplier_max=1e6)
        h = x @ B.T - c
        h_inf = float(h.abs().max())
        x_err = _maxabs(x, x_star)
        lam_err = _maxabs(S.lam, lam_star)
        gate = 1e-8 if dtype == torch.float64 else 2e-4
        assert h_inf <= gate, f"{dtype}: |h|_inf {h_inf:.2e}"
        assert x_err <= 100 * gate, f"{dtype}: |x - x*| {x_err:.2e}"
        assert lam_err <= 1e4 * gate, f"{dtype}: |lam - lam*| {lam_err:.2e}"
        print(f"     {str(dtype):14s} {device.type}: |h|inf {h_inf:.1e}  |x-x*| {x_err:.1e}  "
              f"|lam-lam*| {lam_err:.1e}  rho in [{float(S.rho.min()):.0f}, {float(S.rho.max()):.0f}]")
    print("PASS AL rounds converge to the KKT point of an equality-constrained quadratic, all 50 particles")


## --------------------------------------------------------------------------------------
## 3. Gauss-Newton correction
## --------------------------------------------------------------------------------------

def test_gn_correction_is_exact_on_linear_rows():
    for dtype, device, tol in _targets():
        gen = _gen(5, device)
        N, k, n = 13, 3, 8
        A = _randn(N, k, n, dtype=dtype, device=device, gen=gen)
        b = _randn(N, k, dtype=dtype, device=device, gen=gen)
        x = _randn(N, n, dtype=dtype, device=device, gen=gen)
        r = (A @ x.unsqueeze(2)).squeeze(2) - b
        dx = al.gn_correction(A, r, delta=0.0)
        r_new = (A @ (x + dx).unsqueeze(2)).squeeze(2) - b
        assert _maxabs(r_new) <= tol * 10, f"{_maxabs(r_new):.2e}"
        # Minimum-norm: dx lies in the row space of A.
        P = al.tangent_projector(A, 0.0)
        assert _maxabs(P @ dx.unsqueeze(2)) <= tol * 10
    print("PASS one GN step lands exactly on r = 0 for linear rows (minimum-norm step)")


def test_gn_correction_converges_quadratically_on_a_circle():
    dtype, device = torch.float64, torch.device("cpu")
    gen = _gen(6, device)
    N = 9
    x = _randn(N, 2, dtype=dtype, device=device, gen=gen) * 0.15 + torch.tensor(
        [1.0, 0.0], dtype=dtype, device=device)
    residuals = []
    for _ in range(5):
        r = (x * x).sum(1, keepdim=True) - 1.0           # [N, 1]
        J = (2.0 * x).unsqueeze(1)                       # [N, 1, 2]
        residuals.append(float(r.abs().max()))
        x = x + al.gn_correction(J, r, delta=0.0)
    residuals.append(float(((x * x).sum(1) - 1.0).abs().max()))
    print("     |r|inf over Newton steps:", " ".join(f"{v:.1e}" for v in residuals))
    assert residuals[4] <= 1e-12, residuals
    # Quadratic: r_{k+1} <= C r_k^2 with a modest C on every step (radial Newton on rho^2 - 1
    # has r_{k+1} = r_k^2 / (4 rho_k) exactly), down to the float64 floor.
    for k in range(4):
        assert residuals[k + 1] <= 2.0 * residuals[k] ** 2 + 1e-15, residuals
    print("PASS GN on the unit circle converges quadratically over 4 steps")


def test_gn_correction_ignores_masked_rows_and_survives_singular_j():
    for dtype, device, tol in _targets():
        gen = _gen(7, device)
        N, k, n = 7, 4, 6
        J = _randn(N, k, n, dtype=dtype, device=device, gen=gen)
        r = _randn(N, k, dtype=dtype, device=device, gen=gen)
        mask = torch.ones(N, k, dtype=dtype, device=device)
        mask[:, 1] = 0.0
        mask[::2, 3] = 0.0
        delta = 1e-6
        dx_masked = al.gn_correction(J * mask.unsqueeze(2), r * mask, delta)
        # Reference: per particle, drop the masked rows and solve the reduced system.
        ref = torch.zeros_like(dx_masked)
        for i in range(N):
            keep = mask[i] > 0.5
            Ji, ri = J[i][keep], r[i][keep]
            Ai = Ji @ Ji.T + delta * torch.eye(int(keep.sum()), dtype=dtype, device=device)
            ref[i] = -Ji.T @ torch.linalg.solve(Ai, ri)
        err = _maxabs(dx_masked, ref)
        assert err <= tol * 100, f"{err:.2e}"
        # Singular J (duplicate rows) with delta > 0: finite; with delta = 0: zero, not nan.
        Js = J.clone()
        Js[:, 1] = Js[:, 0]
        rs = r.clone()
        rs[:, 1] = rs[:, 0]
        dx = al.gn_correction(Js, rs, 1e-8)
        assert bool(torch.isfinite(dx).all())
        dx0 = al.gn_correction(Js, rs, 0.0)
        assert bool(torch.isfinite(dx0).all())
    print("PASS GN ignores zeroed rows exactly and is finite on singular J")


def test_relative_eta_seed_and_lm_gain_ratio():
    """The second wave's three AL additions. (a) `update_multipliers(eta_rel)`: on the
    failing branch the new tolerance is `max(rho_new^-0.1, eta_rel * infeasibility)`, on the
    passing one `max(eta / rho^0.9, eta_rel * infeasibility)`, and `eta_rel = 0` is bitwise
    the textbook rule.
    (b) `seed_eta` raises `eta` to `eta_rel * infeasibility` where that is larger and is a
    no-op at 0. (c) `lm_gain_update` grows the damping by `growth` (capped) where the gain
    ratio (actual / predicted reduction of |r|^2) is below 0.25, shrinks it (floored) above
    0.75, and leaves it alone without information (no prediction yet, a non-decreasing
    prediction); on linear rows the ratio is exactly 1; `gn_correction(lam)` with a large
    `lam` is a short step along the Marquardt-scaled gradient `-J^T D^-1 r / lam`."""
    for dtype, device, tol in _targets():
        gen = _gen(13, device)
        N, n, m_e, m_i = 12, 4, 2, 2
        S = ALState.init(N, n, m_e, m_i, rho0=10.0, eta0=0.5, lr0=0.1, dtype=dtype, device=device)
        h = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        h[::2] *= 1e-3                                           # even: pass; odd: fail
        h[1::2] += 5.0
        g = -torch.ones(N, m_i, dtype=dtype, device=device)
        v = al.infeasibility(h, g, S)
        ok = v <= S.eta
        assert bool(ok[::2].all()) and not bool(ok[1::2].any())
        S_nw = al.update_multipliers(h, g, S, 10.0, 1e4, 1e3)
        S_0 = al.update_multipliers(h, g, S, 10.0, 1e4, 1e3, eta_rel=0.0)
        assert bool((S_nw.eta == S_0.eta).all()) and bool((S_nw.rho == S_0.rho).all())
        S_r = al.update_multipliers(h, g, S, 10.0, 1e4, 1e3, eta_rel=0.5)
        exp_fail = torch.maximum(S_r.rho ** -0.1, 0.5 * v)
        assert _maxabs(S_r.eta[1::2], exp_fail[1::2]) <= tol * 10
        exp_pass = torch.maximum(S_nw.eta, 0.5 * v)
        assert _maxabs(S_r.eta[::2], exp_pass[::2]) <= tol * 10, "passing branch: max(eta/rho^0.9, eta_rel v)"
        assert bool((S_r.lam == S_nw.lam).all()) and bool((S_r.rho == S_nw.rho).all())
        S_s = al.seed_eta(h, g, S, 0.5)
        assert _maxabs(S_s.eta, torch.maximum(S.eta, 0.5 * v)) <= tol
        assert bool((al.seed_eta(h, g, S, 0.0).eta == S.eta).all())
        ## LM: the gain ratio, case by case (m_e = 2, m_i = 2 rows; mask on the h rows)
        kw = dict(dtype=dtype, device=device)
        S_l = al.replace(S, gn_lam=torch.full((N,), 1e-2, **kw),
                         gn_r2=torch.full((N,), 1.0, **kw), gn_pred2=torch.zeros(N, **kw),
                         gn_mask=torch.cat([torch.ones(N, m_e, **kw), torch.zeros(N, m_i, **kw)], 1))
        a2 = torch.tensor([0.9, 0.1, 0.5, 0.5, 0.5, 0.9] * 2, **kw)   # actual |r|^2 per particle
        h_l = torch.zeros(N, m_e, **kw)
        h_l[:, 0] = a2.sqrt()
        g_l = torch.full((N, m_i), 123.0, **kw)                          # masked out: ignored
        S_l = al.replace(S_l, gn_r2=torch.where(torch.arange(N, device=device) % 6 == 3,
                                                 torch.full_like(S_l.gn_r2, float("nan")), S_l.gn_r2),
                         gn_pred2=torch.where(torch.arange(N, device=device) % 6 == 4,
                                              torch.full_like(S_l.gn_pred2, 2.0), S_l.gn_pred2))
        S_l2, ratio = al.lm_gain_update(S_l, h_l, g_l, growth=10.0, lam_min=1e-4, lam_max=1e-1)
        lam = S_l2.gn_lam.double().cpu().numpy()
        rat = ratio.double().cpu().numpy()
        exp_lam = np.array([1e-1, 1e-3, 1e-2, 1e-2, 1e-2, 1e-1] * 2)
        assert np.allclose(lam, exp_lam, rtol=1e-5), f"lam {lam} vs {exp_lam}"
        assert np.allclose(rat[[0, 1, 2]], [0.1, 0.9, 0.5], rtol=1e-5) and np.isnan(rat[[3, 4]]).all()
        S_cap, _ = al.lm_gain_update(S_l2, h_l, g_l, growth=10.0, lam_min=1e-4, lam_max=1e-1)
        assert abs(float(S_cap.gn_lam[0]) - 1e-1) <= 1e-7, "capped at lam_max"
        ## record + update on LINEAR rows with the exact GN step: actual == predicted, ratio 1
        A = _randn(N, m_e, n, dtype=dtype, device=device, gen=gen)
        b = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        x = _randn(N, n, dtype=dtype, device=device, gen=gen)
        r0 = (A @ x.unsqueeze(2)).squeeze(2) - b
        d = 0.5 * al.gn_correction(A, r0, 0.0)                       # half the GN step
        mask = torch.cat([torch.ones(N, m_e, **kw), torch.zeros(N, m_i, **kw)], 1)
        S_p = al.lm_record_prediction(S_l, torch.cat([A, torch.zeros(N, m_i, n, **kw)], 1),
                                      torch.cat([r0, torch.zeros(N, m_i, **kw)], 1), mask, d)
        r1 = (A @ (x + d).unsqueeze(2)).squeeze(2) - b
        _, ratio_lin = al.lm_gain_update(S_p, r1, g_l, growth=10.0, lam_min=1e-4, lam_max=1e-1)
        assert _maxabs(ratio_lin, torch.ones_like(ratio_lin)) <= 1e3 * tol, \
            f"linear rows: ratio {ratio_lin}"
        J = _randn(N, m_e, n, dtype=dtype, device=device, gen=gen)
        r = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        dx0 = al.gn_correction(J, r, 1e-9)
        dx_none = al.gn_correction(J, r, 1e-9, torch.zeros(N, dtype=dtype, device=device))
        assert _maxabs(dx0, dx_none) <= tol, "lam = 0 is plain GN"
        lam = torch.full((N,), 1e6, dtype=dtype, device=device)
        dx = al.gn_correction(J, r, 0.0, lam)
        D = torch.diagonal(J @ J.transpose(1, 2), dim1=1, dim2=2)
        grad_step = -(J.transpose(1, 2) @ (r / D).unsqueeze(2)).squeeze(2) / lam.unsqueeze(1)
        rel = _maxabs(dx, grad_step) / max(_maxabs(grad_step), 1e-30)
        assert rel <= 1e-3, f"large-lam step is not the scaled gradient: rel {rel:.2e}"
    print("PASS relative eta (update + seed), the LM gain ratio and the damped GN step")


## --------------------------------------------------------------------------------------
## 4./5. Tangent projector and the q-step clamp
## --------------------------------------------------------------------------------------

def test_tangent_projector_algebra():
    for dtype, device, tol in _targets():
        gen = _gen(8, device)
        N, k, n = 10, 4, 9
        J = _randn(N, k, n, dtype=dtype, device=device, gen=gen)
        P = al.tangent_projector(J, 0.0)
        t = 1e-10 if dtype == torch.float64 else 1e-4
        assert _maxabs(P @ J.transpose(1, 2)) <= t, f"P J^T: {_maxabs(P @ J.transpose(1, 2)):.2e}"
        assert _maxabs(P @ P, P) <= t, f"P^2 - P: {_maxabs(P @ P, P):.2e}"
        assert _maxabs(P, P.transpose(1, 2)) <= t
        # Rank n - k: trace.
        tr = torch.diagonal(P, dim1=1, dim2=2).sum(1)
        assert _maxabs(tr, torch.full_like(tr, n - k)) <= 100 * t
        # Regularised: P J^T is O(delta), still symmetric.
        Pd = al.tangent_projector(J, 1e-6)
        assert _maxabs(Pd, Pd.transpose(1, 2)) <= t
        assert _maxabs(Pd @ J.transpose(1, 2)) <= 1e-3
    print("PASS tangent projector: P J^T = 0, P^2 = P, symmetric, rank n - k")


def test_clamp_q_step_scales_exactly_to_the_bound():
    for dtype, device, tol in _targets():
        gen = _gen(9, device)
        N, n, ndof = 12, 20, 7
        dx = _randn(N, n, dtype=dtype, device=device, gen=gen)
        Jq = _randn(N, ndof, n, dtype=dtype, device=device, gen=gen)
        dq = (Jq @ dx.unsqueeze(2)).squeeze(2)
        norms = dq.abs().amax(1)
        q_max = float(norms.median())           # about half exceed, half do not
        out = al.clamp_q_step(dx, Jq, q_max)
        dq_out = (Jq @ out.unsqueeze(2)).squeeze(2)
        over = norms > q_max
        new_norm = dq_out.abs().amax(1)
        assert _maxabs(new_norm[over], torch.full_like(new_norm[over], q_max)) <= tol * 100
        assert bool((out[~over] == dx[~over]).all()), "steps inside the bound must be untouched bitwise"
        # Direction preserved.
        ratio = out[over] / dx[over]
        assert _maxabs(ratio, ratio[:, :1].expand_as(ratio)) <= tol * 1e3
        # Joint space: Jq is None, the bound applies to dx itself.
        out2 = al.clamp_q_step(dx, None, 0.5)
        assert float(out2.abs().amax(1).max()) <= 0.5 * (1 + tol * 10)
        small = dx.abs().amax(1) <= 0.5
        assert bool((out2[small] == dx[small]).all())
    print("PASS clamp_q_step scales exactly onto the bound and leaves small steps bitwise")


## --------------------------------------------------------------------------------------
## 6. Kernels
## --------------------------------------------------------------------------------------

def test_pairwise_sqdist_matches_brute_force():
    for dtype, device, tol in _targets():
        gen = _gen(10, device)
        N, d = 33, 7
        y = _randn(N, d, dtype=dtype, device=device, gen=gen)
        D = kernels.pairwise_sqdist(y)
        brute = ((y.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
        err = _maxabs(D, brute)
        assert err <= tol * 1e3, f"{err:.2e}"
        assert bool((torch.diagonal(D) == 0).all())
        assert bool((D >= 0).all())
    print("PASS pairwise_sqdist matches the brute-force [N, N, d] expansion")


def test_median_bandwidth_matches_numpy_on_the_upper_triangle():
    for dtype, device, tol in _targets():
        gen = _gen(11, device)
        for N in (2, 3, 4, 5, 16, 64, 65):
            y = _randn(N, 3, dtype=dtype, device=device, gen=gen)
            D = kernels.pairwise_sqdist(y)
            h = kernels.median_bandwidth(D, N, floor=1e-3)
            Dn = D.detach().cpu().double().numpy()
            iu = np.triu_indices(N, k=1)
            med = np.median(np.sqrt(Dn[iu]))
            expect = max(1e-6, med ** 2 / math.log(N))
            assert abs(float(h) - expect) <= tol * 1e3 * max(1.0, expect), (N, float(h), expect)
        # Floor binds when the swarm has collapsed.
        y0 = torch.zeros(8, 3, dtype=dtype, device=device)
        h0 = kernels.median_bandwidth(kernels.pairwise_sqdist(y0), 8, floor=0.2)
        assert abs(float(h0) - 0.2 ** 2) <= tol
        # N = 1 -> floor^2, no log(1) division.
        h1 = kernels.median_bandwidth(torch.zeros(1, 1, dtype=dtype, device=device), 1, floor=0.3)
        assert abs(float(h1) - 0.3 ** 2) <= tol
    print("PASS median_bandwidth equals numpy's median over the upper triangle (odd and even pair counts)")


def test_rbf_repulsion_is_the_svgd_gradient():
    for dtype, device, tol in _targets():
        gen = _gen(12, device)
        N, d = 21, 4
        y = _randn(N, d, dtype=dtype, device=device, gen=gen)
        h = 1.7
        K, R = kernels.rbf_terms(y, h)
        assert _maxabs(K, K.T) <= tol * 10
        assert _maxabs(torch.diagonal(K), torch.ones_like(torch.diagonal(K))) <= tol * 10
        # R_j = sum_i grad_{y_i} k(y_i, y_j): differentiate the FIRST argument only, and read
        # the gradient on the first argument's particle, summed over the second.
        y1 = y.clone().requires_grad_(True)
        Kf = torch.exp(-((y1.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2) / h)   # [i, j]
        (grad_first,) = torch.autograd.grad(Kf.sum(), y1)
        # grad_first[i] = sum_j grad_{y_i} k(y_i, y_j), the repulsion ON particle i by symmetry
        # of k -- but with the OPPOSITE sign of what we want: grad_{y_i} k(y_i, y_j) points
        # from y_i toward y_j. The repulsion on j is sum_i grad_{y_i} k(y_i, y_j) = -grad_first[j].
        err = _maxabs(R, -grad_first)
        t = 1e-10 if dtype == torch.float64 else 1e-4
        assert err <= t, f"R vs autograd: {err:.2e}"
        # Explicit form.
        explicit = (2.0 / h) * (K.sum(1, keepdim=True) * y - K @ y)
        assert _maxabs(R, explicit) <= t
        # Sign: a particle is pushed AWAY from the swarm's centre of mass (two-particle case).
        y2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype, device=device)
        _, R2 = kernels.rbf_terms(y2, 1.0)
        assert float(R2[1, 0]) > 0 and float(R2[0, 0]) < 0
    print("PASS rbf_terms: K symmetric with unit diagonal; R = sum_i grad_{y_i} K_ij (repulsive sign)")


def test_svgd_direction_with_kernel_none_is_the_driving_term():
    for dtype, device, tol in _targets():
        gen = _gen(13, device)
        N, n = 16, 5
        driving = _randn(N, n, dtype=dtype, device=device, gen=gen)
        assert kernels.kernel_space(torch.zeros(N, 7, dtype=dtype, device=device),
                                   driving, "none") is None
        K, R = kernels.identity_terms(N, n, dtype, device)
        phi = kernels.svgd_direction(K, driving, kernels.pullback_identity(R), gamma=1.0, T=1.0)
        assert bool((phi == driving).all()), "kernel none must reproduce the driving term bitwise"
        phi_n = kernels.svgd_direction(K, driving, R, gamma=1.0, T=1.0, normaliser="n")
        assert _maxabs(phi_n, driving / N) <= tol * 10
        # kernel_space returns the right coordinate.
        yq = _randn(N, 7, dtype=dtype, device=device, gen=gen)
        assert kernels.kernel_space(yq, driving, "q") is yq
        assert kernels.kernel_space(yq, driving, "x") is driving
    print("PASS svgd_direction with the identity kernel is exactly the driving term")


def _run_svgd_gaussian(dtype, device, steps, lr, N=200, anneal=None, normaliser="rowsum"):
    """Plain SVGD on a 2-D standard Gaussian from an offset, tight cloud."""
    gen = _gen(14, device)
    y = 0.3 * _randn(N, 2, dtype=dtype, device=device, gen=gen) + torch.tensor(
        [2.5, -1.5], dtype=dtype, device=device)
    for t in range(steps):
        D = kernels.pairwise_sqdist(y)
        h = kernels.median_bandwidth(D, N, floor=1e-3)
        K, R = kernels.rbf_terms(y, h)
        T = 1.0 if anneal is None else anneal(t)
        phi = kernels.svgd_direction(K, -y, kernels.pullback_identity(R), gamma=1.0, T=T,
                                     normaliser=normaliser)
        y = y + lr * phi
    return y


def test_plain_svgd_matches_a_gaussians_moments():
    for dtype, device, tol in _targets():
        y = _run_svgd_gaussian(dtype, device, steps=500, lr=0.1)
        mean = y.mean(0)
        cov = torch.cov(y.T)
        mean_err = float(mean.abs().max())
        cov_err = float((cov - torch.eye(2, dtype=dtype, device=device)).abs().max())
        print(f"     {str(dtype):14s} {device.type}: mean {mean.tolist()}  "
              f"cov diag {torch.diagonal(cov).tolist()}  off {float(cov[0, 1]):.4f}")
        assert mean_err <= 0.05, f"mean error {mean_err:.3f}"
        assert cov_err <= 0.08, f"covariance error {cov_err:.3f}"
    # The textbook 1/N normaliser reaches the same fixed point (slower per step, so more steps).
    y = _run_svgd_gaussian(torch.float64, torch.device("cpu"), steps=3000, lr=0.5, normaliser="n")
    cov = torch.cov(y.T)
    print(f"     normaliser='n': mean {y.mean(0).tolist()}  cov diag {torch.diagonal(cov).tolist()}")
    assert float(y.mean(0).abs().max()) <= 0.05
    assert float((cov - torch.eye(2, dtype=torch.float64)).abs().max()) <= 0.08
    print("PASS plain SVGD on N(0, I) reproduces mean and covariance (the sign check)")


def test_svgd_with_annealed_temperature_collapses_onto_the_mode():
    for dtype, device, tol in _targets():
        total = 800
        anneal = lambda t: kernels.anneal_T(t, T0=1.0, anneal_frac=0.5, total_steps=total)
        y = _run_svgd_gaussian(dtype, device, steps=total, lr=0.1, anneal=anneal)
        spread = float(y.std(0).max())
        mean_err = float(y.mean(0).abs().max())
        print(f"     {str(dtype):14s} {device.type}: after anneal to T = 0: spread {spread:.1e}, "
              f"|mean| {mean_err:.1e}")
        assert spread <= 1e-2 and mean_err <= 1e-2
    print("PASS annealing T to exactly 0 collapses the swarm onto the mode")


## --------------------------------------------------------------------------------------
## 7. Adam
## --------------------------------------------------------------------------------------

def test_adam_step_matches_torch_optim_adam():
    for dtype, device, tol in _targets():
        gen = _gen(15, device)
        n, lr = 6, 0.05
        x0 = _randn(1, n, dtype=dtype, device=device, gen=gen)
        grads = [_randn(1, n, dtype=dtype, device=device, gen=gen) for _ in range(20)]
        # Reference.
        p = torch.nn.Parameter(x0.clone())
        opt = torch.optim.Adam([p], lr=lr, betas=(0.9, 0.999), eps=1e-8)
        for g in grads:
            opt.zero_grad()
            p.grad = g.clone()
            opt.step()
        # Ours.
        S = ALState.init(1, n, 0, 0, rho0=1.0, eta0=1.0, lr0=lr, dtype=dtype, device=device)
        x = x0
        for g in grads:
            x, S = al.adam_step(x, g, S)
        err = _maxabs(x, p.detach())
        t = 1e-10 if dtype == torch.float64 else 1e-5
        assert err <= t, f"Adam mismatch {err:.2e}"
        assert float(S.step[0]) == 20.0
        # Per-particle lr honoured: same gradient, lr 0.1 vs 0.3 -> first step is lr * sign(g).
        S2 = ALState.init(2, n, 0, 0, rho0=1.0, eta0=1.0, lr0=0.1, dtype=dtype, device=device)
        S2 = al.replace(S2, lr=torch.tensor([0.1, 0.3], dtype=dtype, device=device))
        g2 = grads[0].expand(2, n)
        x2, _ = al.adam_step(torch.zeros(2, n, dtype=dtype, device=device), g2, S2)
        expect = -S2.lr.unsqueeze(1) * torch.sign(g2) * (1.0 / (1.0 + 1e-8 / g2.abs()))
        assert _maxabs(x2, expect) <= t * 100
        print(f"     {str(dtype):14s} {device.type}: 20 Adam steps vs torch.optim.Adam {err:.1e}")
    print("PASS adam_step matches torch.optim.Adam and honours a per-particle lr")


def test_lr_schedule_halves_grows_and_floors():
    dtype, device = torch.float64, torch.device("cpu")
    S = ALState.init(4, 2, 0, 0, rho0=1.0, eta0=1.0, lr0=0.1, dtype=dtype, device=device)
    S = al.replace(S, lr=torch.tensor([0.1, 0.1, 0.01, 0.0004], dtype=dtype))
    dq = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=dtype)
    S2 = al.lr_schedule(S, dq, q_step_max=0.5, lr0=0.1, lr_min=0.0003, t=100.0, lr_decay_t=100.0)
    # lr_global = 0.05: particle 0 halves (0.05), 1 grows to min(0.05, 0.11) = 0.05,
    # 2 grows to 0.011, 3 halves to 0.0002 then floors at 0.0003.
    expect = torch.tensor([0.05, 0.05, 0.011, 0.0003], dtype=dtype)
    assert _maxabs(S2.lr, expect) <= 1e-15, S2.lr
    print("PASS lr_schedule: halve on a large q step, grow 10 % toward the decayed global rate, floor")


## --------------------------------------------------------------------------------------
## 8. Schedules; active mask, resampling, best tracking
## --------------------------------------------------------------------------------------

def test_anneal_T_reaches_exactly_zero_and_anneal_gamma_cycles():
    total, frac = 1000, 0.4
    horizon = frac * total
    assert kernels.anneal_T(0, 2.0, frac, total) == 2.0
    mid = kernels.anneal_T(horizon / 2, 2.0, frac, total)
    assert abs(mid - 2.0 * 0.25) < 1e-15
    assert kernels.anneal_T(horizon, 2.0, frac, total) == 0.0
    assert kernels.anneal_T(horizon + 1, 2.0, frac, total) == 0.0
    assert kernels.anneal_T(total, 2.0, frac, total) == 0.0
    t = torch.arange(0, total + 1, dtype=torch.float64)
    Tt = kernels.anneal_T(t, 2.0, frac, total)
    assert bool((Tt[t >= horizon] == 0.0).all()) and bool((Tt[t < horizon] > 0.0).all())
    assert bool((Tt[1:] <= Tt[:-1]).all()), "T must be non-increasing"

    g = kernels.anneal_gamma
    assert g(0, 100) == 0.0 and abs(g(50, 100) - 0.5) < 1e-15 and g(100, 100) == 1.0 and g(250, 100) == 1.0
    assert abs(g(50, 100, p=2) - 0.25) < 1e-15
    # Three cycles of 100: ramps restart at 100 and 200, gain 1 from 300 on.
    assert abs(g(150, 100, cycles=3) - 0.5) < 1e-15
    assert g(200, 100, cycles=3) == 0.0
    assert abs(g(299, 100, cycles=3) - 0.99) < 1e-12
    assert g(300, 100, cycles=3) == 1.0 and g(450, 100, cycles=3) == 1.0
    gt = g(torch.tensor([0.0, 50.0, 150.0, 200.0, 350.0]), 100, cycles=3)
    assert _maxabs(gt, torch.tensor([0.0, 0.5, 0.5, 0.0, 1.0])) <= 1e-15
    print("PASS anneal_T hits exactly 0.0 at anneal_frac * total_steps and stays; anneal_gamma cycles")


def test_active_mask_resample_mask_and_track_best():
    for dtype, device, tol in _targets():
        gen = _gen(16, device)
        N, m_i = 10, 4
        g = _randn(N, m_i, dtype=dtype, device=device, gen=gen)
        mu = _randn(N, m_i, dtype=dtype, device=device, gen=gen).abs()
        rho = torch.full((N,), 2.0, dtype=dtype, device=device)
        mask = al.active_mask(g, mu, rho)
        assert mask.dtype == dtype
        assert bool((mask == (g > -mu / 2.0).to(dtype)).all())

        q = _randn(N, 7, dtype=dtype, device=device, gen=gen)
        q[1, 3] = 5e3
        q[2, 0] = float("nan")
        q[3, 6] = float("inf")
        rows_finite = torch.ones(N, dtype=torch.bool, device=device)
        rows_finite[4] = False
        rm = al.resample_mask(q, rows_finite, q_max=1000.0)
        expect = torch.zeros(N, dtype=torch.bool, device=device)
        expect[[1, 2, 3, 4]] = True
        assert bool((rm == expect).all()), rm

        S = ALState.init(N, 3, 0, 0, rho0=1.0, eta0=1.0, lr0=0.1, dtype=dtype, device=device)
        x1 = _randn(N, 3, dtype=dtype, device=device, gen=gen)
        m1 = _randn(N, dtype=dtype, device=device, gen=gen)
        S = al.track_best(S, x1, m1)
        assert bool((S.best_x == x1).all()) and bool((S.best_merit == m1).all())
        x2 = _randn(N, 3, dtype=dtype, device=device, gen=gen)
        m2 = m1.clone()
        m2[::2] -= 1.0                     # improve the even particles
        m2[1] = float("nan")               # a nan never wins
        m2[3] = float("inf")
        S = al.track_best(S, x2, m2)
        even = torch.arange(N, device=device) % 2 == 0
        assert bool((S.best_x[even] == x2[even]).all()) and bool((S.best_x[~even] == x1[~even]).all())
        assert bool(torch.isfinite(S.best_merit).all())
        F = _randn(N, dtype=dtype, device=device, gen=gen)
        v = _randn(N, dtype=dtype, device=device, gen=gen).abs()
        assert _maxabs(al.best_merit(F, v, 3.0), F + 3.0 * v) <= tol
    print("PASS active_mask, resample_mask (runaway / nan / inf / non-finite rows) and track_best")


## --------------------------------------------------------------------------------------
## 9. Graph-friendliness
## --------------------------------------------------------------------------------------

def _top_level_commas(src, start):
    """Count the commas at nesting depth 0 between `src[start]` (just past an opening
    paren) and its matching close, so a `torch.where(cond)` -- the one-argument form whose
    output shape depends on the data -- is told apart from the three-argument select."""
    depth, commas = 0, 0
    for ch in src[start:]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return commas
            depth -= 1
        elif ch == "," and depth == 0:
            commas += 1
    raise AssertionError("unbalanced parentheses")


def _code_only(path):
    """The module's source with every string literal and comment blanked out (same length
    and offsets), so a docstring that NAMES `.item()` does not count as a call to it."""
    import io
    import tokenize
    src = open(path).read()
    out = list(src)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.STRING, tokenize.COMMENT):
            lines = src.splitlines(keepends=True)
            start = sum(len(l) for l in lines[:tok.start[0] - 1]) + tok.start[1]
            end = sum(len(l) for l in lines[:tok.end[0] - 1]) + tok.end[1]
            for i in range(start, end):
                if out[i] not in "\n":
                    out[i] = " "
    return "".join(out)


def test_no_host_synchronising_calls_in_the_modules():
    ## `torch.tensor(` too: a Python scalar or list made into a CUDA tensor is a pageable
    ## host-to-device copy, which CUDA-graph capture refuses ("Cannot copy between CPU and
    ## CUDA tensors during CUDA graph capture") -- `median_bandwidth`'s floor was one, found
    ## by the first capture of the split step (2026-10-08). `torch.full(...)` is a fill kernel.
    banned = (".item(", ".tolist(", ".numpy(", ".cpu(", ".nonzero(", "torch.tensor(")
    for name in ("al.py", "kernels.py"):
        src = _code_only(os.path.join(SRC, name))
        for b in banned:
            assert b not in src, f"{name} contains {b!r}"
        pos, n_where = 0, 0
        while True:
            pos = src.find("torch.where(", pos)
            if pos < 0:
                break
            n_where += 1
            assert _top_level_commas(src, pos + len("torch.where(")) >= 2, (
                f"{name}: single-argument torch.where at offset {pos}")
            pos += 1
        assert n_where > 0
    print("PASS no .item()/.tolist()/.numpy()/.cpu()/.nonzero()/torch.tensor()/1-arg where in al.py or kernels.py")


def _compiled_chain(x, F_a, A, b, C, d, lam, mu, rho, eta, adam_m, adam_v, step, lr, best_x,
                    best_merit):
    """al_value -> gradient -> adam_step -> update_multipliers -> gn_correction, all tensors
    in and out (ALState is rebuilt inside so the compiled signature is tensors only)."""
    S = ALState(lam=lam, mu=mu, rho=rho, eta=eta, adam_m=adam_m, adam_v=adam_v, step=step, lr=lr,
                best_x=best_x, best_merit=best_merit)

    def merit(xx):
        F = 0.5 * ((xx - F_a) ** 2).sum(1)
        h = (A @ xx.unsqueeze(2)).squeeze(2) - b
        g = (C @ xx.unsqueeze(2)).squeeze(2) - d
        return al.al_value(F, h, g, S).sum(), (h, g)

    grad, (h, g) = torch.func.grad(merit, has_aux=True)(x)
    x1, S = al.adam_step(x, grad, S)
    S = al.update_multipliers(h, g, S, rho_growth=10.0, rho_max=1e4, multiplier_max=1e3)
    h1 = (A @ x1.unsqueeze(2)).squeeze(2) - b
    dx = al.gn_correction(A, h1, delta=1e-10)
    x2 = x1 + al.clamp_q_step(dx, None, 0.5)
    S = al.track_best(S, x2, al.best_merit(0.5 * ((x2 - F_a) ** 2).sum(1),
                                           al.infeasibility(h1, g, S), 10.0))
    return x2, S.lam, S.mu, S.rho, S.eta, S.adam_m, S.adam_v, S.step, S.lr, S.best_x, S.best_merit


def test_compiled_chain_matches_eager():
    if not torch.cuda.is_available():
        print("SKIP compiled-chain smoke: no CUDA device")
        return
    dtype, device = torch.float64, torch.device("cuda")
    gen = _gen(17, device)
    N, n, m_e, m_i = 32, 6, 2, 3
    x = _randn(N, n, dtype=dtype, device=device, gen=gen)
    a = _randn(N, n, dtype=dtype, device=device, gen=gen)
    A = _randn(N, m_e, n, dtype=dtype, device=device, gen=gen)
    b = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
    C = _randn(N, m_i, n, dtype=dtype, device=device, gen=gen)
    d = _randn(N, m_i, dtype=dtype, device=device, gen=gen)
    S = _state(N, n, m_e, m_i, dtype, device, gen)
    S = al.replace(S, eta=torch.full((N,), 2.0, dtype=dtype, device=device))
    args = (x, a, A, b, C, d, S.lam, S.mu, S.rho, S.eta, S.adam_m, S.adam_v, S.step, S.lr,
            S.best_x, S.best_merit)
    eager = _compiled_chain(*args)
    compiled = torch.compile(_compiled_chain, dynamic=False, fullgraph=True)
    out = compiled(*args)
    out2 = compiled(*args)       # second call: the cached graph
    torch.cuda.synchronize()
    worst = 0.0
    for e, c, c2 in zip(eager, out, out2):
        assert bool((c == c2).all()), "compiled call is not deterministic"
        worst = max(worst, _maxabs(e, c))
    assert worst <= 1e-10, f"compiled vs eager {worst:.2e}"
    print(f"     fullgraph torch.compile(dynamic=False) on CUDA float64 vs eager: {worst:.1e}")
    print("PASS compiled AL chain matches eager (full graph, no breaks)")


if __name__ == "__main__":
    test_al_value_and_coefficients_agree_with_autograd()
    test_al_with_zero_multipliers_is_the_quadratic_penalty()
    test_update_multipliers_takes_the_right_branch_per_particle()
    test_al_rounds_converge_to_the_kkt_point_on_every_particle()
    test_gn_correction_is_exact_on_linear_rows()
    test_gn_correction_converges_quadratically_on_a_circle()
    test_gn_correction_ignores_masked_rows_and_survives_singular_j()
    test_relative_eta_seed_and_lm_gain_ratio()
    test_tangent_projector_algebra()
    test_clamp_q_step_scales_exactly_to_the_bound()
    test_pairwise_sqdist_matches_brute_force()
    test_median_bandwidth_matches_numpy_on_the_upper_triangle()
    test_rbf_repulsion_is_the_svgd_gradient()
    test_svgd_direction_with_kernel_none_is_the_driving_term()
    test_plain_svgd_matches_a_gaussians_moments()
    test_svgd_with_annealed_temperature_collapses_onto_the_mode()
    test_adam_step_matches_torch_optim_adam()
    test_lr_schedule_halves_grows_and_floors()
    test_anneal_T_reaches_exactly_zero_and_anneal_gamma_cycles()
    test_active_mask_resample_mask_and_track_best()
    test_no_host_synchronising_calls_in_the_modules()
    test_compiled_chain_matches_eager()
    print("ALL PASS")
