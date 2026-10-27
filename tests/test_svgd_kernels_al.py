"""`src/svgd/al.py`, `src/svgd/kernels.py` and the step pieces of `src/svgd/fused.py` on synthetic problems.

The modules are pure tensor math on per-particle quantities and know nothing about any
formulation, so they are tested on problems whose answers are known in closed form: the PHR
augmented Lagrangian (one fixed, shared penalty rho) against autograd and against the
quadratic penalty it reduces to; the dual-ascent step -- unconditional, per particle, its own
step alpha (alpha = 0 leaves the multipliers at zero), clipped and counted, zeroed on resample; AL rounds converging to the KKT point of an
equality-constrained quadratic, one per particle; the RBF repulsion against autograd and, the check that pins its SIGN, plain
SVGD on a 2-D standard Gaussian matching the target's mean and covariance; the GN Hessian of
the penalty against autograd on linear rows; one Stein variational Newton step landing on a
quadratic's minimiser; and the identity metric being projected gradient descent with the
per-particle step `svgd_lr / ||H_i||_F`. The kernel spaces (`svgd_kernel`: the `q` fielded until 2026-10-12
reproducing the previous step bit for bit, and x, z, c, cz, qz, pose against autograd) close it.
There is no temperature: `svgd_temperature` and `svgd_metric_temperature` were removed 2026-10-14
(stage SVGD_R22 measured T inert), and the step is the T = 1 one.
Tiers (tests/_tiers.py): the default `fast` runs in ~11 s on one CPU core; no `slow` group.
"""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import _tiers                                                             # noqa: E402

from src.svgd import al, fused, kernels
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
    return torch.randn(*shape, generator=gen, dtype=torch.float64, device="cpu").to(dtype=dtype, device=device)


def _maxabs(a, b=None):
    d = a if b is None else a - b
    return float(d.detach().abs().max()) if d.numel() else 0.0


RHO = 2.5


def _state(N, m_e, m_i, dtype, device, gen, random_mult=True):
    S = ALState.init(N, m_e, m_i, dtype=dtype, device=device)
    if random_mult:
        S = al.replace(S, lam=_randn(N, m_e, dtype=dtype, device=device, gen=gen),
                       mu=_randn(N, m_i, dtype=dtype, device=device, gen=gen).abs())
    return S


## --------------------------------------------------------------------------------------
## 1. The PHR augmented Lagrangian
## --------------------------------------------------------------------------------------

def test_al_value_and_coefficients_agree_with_autograd():
    for dtype, device, tol in _targets():
        gen = _gen(1, device)
        N, m_e, m_i = 17, 3, 4
        S = _state(N, m_e, m_i, dtype, device, gen)
        F = _randn(N, dtype=dtype, device=device, gen=gen)
        h = _randn(N, m_e, dtype=dtype, device=device, gen=gen).requires_grad_(True)
        g = _randn(N, m_i, dtype=dtype, device=device, gen=gen).requires_grad_(True)
        L = al.al_value(F, h, g, S, RHO)
        assert L.shape == (N,)
        dh, dg = torch.autograd.grad(L.sum(), (h, g))
        ch, cg = al.al_constraint_grad_coefficients(h.detach(), g.detach(), S, RHO)
        err = max(_maxabs(dh, ch), _maxabs(dg, cg))
        assert err <= tol * 10, f"{dtype} {device}: coefficient error {err:.2e}"
        hand = (F + (S.lam * h).sum(1) + 0.5 * RHO * (h * h).sum(1)
                + ((torch.clamp(S.mu + RHO * g, min=0) ** 2 - S.mu ** 2).sum(1)) / (2 * RHO))
        assert _maxabs(L, hand) <= tol * 10
        print(f"     {str(dtype):14s} {device.type}: dL/dh, dL/dg vs autograd {err:.1e}")
    print("PASS al_value and its row coefficients agree with autograd")


def test_al_with_zero_multipliers_is_the_quadratic_penalty():
    for dtype, device, tol in _targets():
        gen = _gen(2, device)
        N, m_e, m_i = 11, 2, 5
        rho0 = 3.5
        S = ALState.init(N, m_e, m_i, dtype=dtype, device=device)
        lam = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        S = al.replace(S, lam=lam)  # mu stays 0
        F = _randn(N, dtype=dtype, device=device, gen=gen)
        h = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        g = _randn(N, m_i, dtype=dtype, device=device, gen=gen)
        L = al.al_value(F, h, g, S, rho0)
        pen = (F + (lam * h).sum(1) + 0.5 * rho0 * (h * h).sum(1)
               + 0.5 * rho0 * (torch.clamp(g, min=0) ** 2).sum(1))
        assert _maxabs(L, pen) <= tol * 10, f"{_maxabs(L, pen):.2e}"
    print("PASS mu = 0 reduces PHR to the quadratic penalty F + lam.h + rho/2 |h|^2 + rho/2 |g+|^2")


## --------------------------------------------------------------------------------------
## 2. The outer check: per-particle multipliers, tolerance and penalty
## --------------------------------------------------------------------------------------

def test_dual_update_is_unconditional_and_per_particle():
    """The dual step on EVERY particle, whatever its violation (no feasibility gate): each
    particle's multipliers move by its OWN rows times the dual step alpha (its own parameter,
    not rho); clipped to the box with the clipped entries counted; zeroed on the masked
    (resampled) particles only. alpha = 0 is a no-op that leaves zero multipliers at zero."""
    ALPHA = 0.7
    for dtype, device, tol in _targets():
        gen = _gen(3, device)
        N, m_e, m_i = 20, 2, 3
        S = _state(N, m_e, m_i, dtype, device, gen)
        lam, mu = S.lam.clone(), S.mu.clone()
        h = _randn(N, m_e, dtype=dtype, device=device, gen=gen)
        g = _randn(N, m_i, dtype=dtype, device=device, gen=gen)
        h[::2] *= 1e-3                                  # even: near-feasible; odd: far off
        h[1::2] += 50.0
        mult_max = 20.0
        S2, n_clipped = al.dual_update(h, g, S, ALPHA, mult_max)
        lam_raw, mu_raw = lam + ALPHA * h, torch.clamp(mu + ALPHA * g, min=0.0)
        assert _maxabs(S2.lam, torch.clamp(lam_raw, -mult_max, mult_max)) <= tol * 10
        assert _maxabs(S2.mu, torch.clamp(mu_raw, max=mult_max)) <= tol * 10
        assert bool((S2.lam != lam).any(dim=1).all()), "every particle's lam moves, feasible or not"
        exp_clip = int((lam_raw.abs() > mult_max).sum() + (mu_raw > mult_max).sum())
        assert int(n_clipped) == exp_clip and exp_clip > 0, (int(n_clipped), exp_clip)
        assert float(S2.mu.min()) >= 0.0 and float(S2.lam.abs().max()) <= mult_max
        ## per particle: particle j's step depends on particle j's rows only
        h_alt = h.clone()
        h_alt[5] += 1.0
        S3, _ = al.dual_update(h_alt, g, S, ALPHA, 1e9)
        S4, _ = al.dual_update(h, g, S, ALPHA, 1e9)
        diff = (S3.lam - S4.lam).abs().amax(dim=1)
        assert float(diff[5]) > 0 and float(torch.cat([diff[:5], diff[6:]]).max()) == 0.0
        ## reset: a redrawn particle gets zero multipliers, the others are untouched
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[[1, 4]] = True
        S5 = al.reset(S2, mask)
        assert bool((S5.lam[mask] == 0).all()) and bool((S5.mu[mask] == 0).all())
        assert bool((S5.lam[~mask] == S2.lam[~mask]).all()) and bool((S5.mu[~mask] == S2.mu[~mask]).all())
        ## alpha = 0: a pure quadratic penalty -- zero multipliers stay exactly zero, nothing clips
        Z = ALState.init(N, m_e, m_i, dtype=dtype, device=device)
        Z2, nz = al.dual_update(h, g, Z, 0.0, mult_max)
        assert bool((Z2.lam == 0).all()) and bool((Z2.mu == 0).all()) and int(nz) == 0
    print("PASS dual update: unconditional, per particle, step alpha (0 = no-op), clipped and counted, "
          "zeroed on resample")


def test_dual_update_counts_clips_by_row_group():
    """`dual_update(by_row=True)` counts, per row, the multiplier entries the clip bound
    (summed over particles), and `by_group` sums those into row groups: built so every
    row's clip count is known in advance, with one group spanning h and g's index lists in
    an interleaved order. The total equals `n_clipped`, which `by_row` leaves unchanged;
    `inf_norm_by_group` is the per-group |.|_inf. Neither syncs with the host."""
    M = 10.0
    for dtype, device, _ in _targets():
        kw = dict(dtype=dtype, device=device)
        N, m_e, m_i = 5, 3, 4
        ## groups: 0 "pos" = h cols 0, 2; 1 "rpy" = h col 1; 2 "col" = g col 3;
        ## 3 "jl" = g cols 0, 2; 4 "trust" = g col 1 (never clips)
        h_grp = torch.tensor([0, 1, 0], dtype=torch.long, device=device)
        g_grp = torch.tensor([3, 4, 3, 2], dtype=torch.long, device=device)
        S = ALState(lam=torch.full((N, m_e), 9.0, **kw), mu=torch.full((N, m_i), 9.0, **kw))
        S.lam[:, 1] = -9.0
        h = torch.zeros(N, m_e, **kw)
        g = torch.zeros(N, m_i, **kw)
        h[:2, 0] = 2.0            # col 0: particles 0, 1 cross +M           -> 2 clips
        h[4, 0] = -30.0           # ... particle 4 crosses -M                 -> 1 more
        h[:, 1] = -1.5            # col 1: every particle crosses -M          -> 5 clips
        h[3, 2] = 0.5             # col 2: 9.5, inside the box                -> 0
        g[:3, 0] = 5.0            # col 0: three particles cross +M           -> 3 clips
        g[:, 1] = -20.0           # col 1: clamped at 0 from below, not a clip -> 0
        g[1, 2] = 1.5             # col 2: one particle crosses               -> 1 clip
        g[:, 3] = 4.0             # col 3: all five                           -> 5 clips
        S2, n, (lam_rows, mu_rows) = al.dual_update(h, g, S, 1.0, M, by_row=True)
        S2b, nb = al.dual_update(h, g, S, 1.0, M)
        assert torch.equal(S2.lam, S2b.lam) and torch.equal(S2.mu, S2b.mu) and int(n) == int(nb)
        assert lam_rows.tolist() == [3, 5, 0] and mu_rows.tolist() == [3, 0, 1, 5], (
            lam_rows.tolist(), mu_rows.tolist())
        grp = al.by_group(lam_rows, h_grp, 5) + al.by_group(mu_rows, g_grp, 5)
        assert grp.device == lam_rows.device and grp.dtype == torch.long
        assert grp.tolist() == [3, 5, 5, 4, 0], grp.tolist()
        assert int(grp.sum()) == int(n) == 17
        ## per-group |.|_inf at the clipped state, per particle
        inf = al.inf_norm_by_group(S2.lam, h_grp, 5) + al.inf_norm_by_group(S2.mu, g_grp, 5)
        ref = torch.zeros(N, 5, **kw)
        for j in range(m_e):
            k = int(h_grp[j]); ref[:, k] = torch.maximum(ref[:, k], S2.lam[:, j].abs())
        for j in range(m_i):
            k = int(g_grp[j]); ref[:, k] = torch.maximum(ref[:, k], S2.mu[:, j].abs())
        assert torch.equal(inf, ref) and float(inf[:, 1].min()) == M and float(inf[:, 4].max()) == 0.0
        ## no rows at all: empty counts, a zero total
        E = ALState.init(N, 0, 0, dtype=dtype, device=device)
        _, n0, (l0, m0) = al.dual_update(torch.zeros(N, 0, **kw), torch.zeros(N, 0, **kw), E, 1.0, M,
                                         by_row=True)
        assert int(n0) == 0 and l0.numel() == 0 and m0.numel() == 0
    print("PASS multiplier clips by row group: per-row counts, group sums, total unchanged, |.|_inf")


def _toy_equality_qp(N, n, m, dtype, device, gen):
    """min 1/2 |x - a_i|^2 s.t. B x = c, a different `a_i` per particle, shared (B, c)."""
    a = _randn(N, n, dtype=torch.float64, device="cpu", gen=gen)
    Q, _ = torch.linalg.qr(_randn(n, m, dtype=torch.float64, device="cpu", gen=gen))
    B = 0.5 * Q.T
    c = _randn(m, dtype=torch.float64, device="cpu", gen=gen)
    BBt = B @ B.T
    lam_star = torch.linalg.solve(BBt, (a @ B.T - c).T).T
    x_star = a - lam_star @ B
    cast = lambda t: t.to(dtype=dtype, device=device)
    return cast(a), cast(B), cast(c), cast(x_star), cast(lam_star)


def test_al_rounds_converge_to_the_kkt_point_on_every_particle():
    """Dual ascent on its own: each round minimises L exactly at the fixed rho (the
    quadratic's normal equations, per particle -- not the swarm's step, which is not under
    test here), then runs `al.dual_update`. The multipliers converge to the KKT multipliers
    and the iterate to the KKT point on every particle."""
    for dtype, device, tol in _targets():
        gen = _gen(4, device)
        N, n, m = 50, 6, 2
        a, B, c, x_star, lam_star = _toy_equality_qp(N, n, m, dtype, device, gen)
        rho = 10.0
        S = ALState.init(N, m, 0, dtype=dtype, device=device)
        g = torch.zeros(N, 0, dtype=dtype, device=device)
        I = torch.eye(n, dtype=dtype, device=device)
        for _ in range(30):
            ## argmin_x 1/2|x - a|^2 + lam.(Bx - c) + rho/2 |Bx - c|^2, per particle
            A = I.unsqueeze(0) + rho * (B.T @ B).unsqueeze(0)
            rhs = a - S.lam @ B + rho * (c @ B).unsqueeze(0)
            x = torch.linalg.solve(A, rhs.unsqueeze(2)).squeeze(2)
            S, _ = al.dual_update(x @ B.T - c, g, S, rho, multiplier_max=1e6)
        h_inf = float((x @ B.T - c).abs().max())
        x_err = _maxabs(x, x_star)
        lam_err = _maxabs(S.lam, lam_star)
        gate = 1e-8 if dtype == torch.float64 else 2e-4
        assert h_inf <= gate, f"{dtype}: |h|_inf {h_inf:.2e}"
        assert x_err <= 100 * gate, f"{dtype}: |x - x*| {x_err:.2e}"
        assert lam_err <= 1e4 * gate, f"{dtype}: |lam - lam*| {lam_err:.2e}"
        print(f"     {str(dtype):14s} {device.type}: |h|inf {h_inf:.1e}  |x-x*| {x_err:.1e}  "
              f"|lam-lam*| {lam_err:.1e}  (rho {rho:g} fixed)")
    print("PASS AL rounds converge to the KKT point of an equality-constrained quadratic, all 50 particles")


## --------------------------------------------------------------------------------------
## 3. Kernels and the Stein direction
## --------------------------------------------------------------------------------------

def test_pairwise_sqdist_matches_brute_force():
    for dtype, device, tol in _targets():
        gen = _gen(10, device)
        y = _randn(33, 7, dtype=dtype, device=device, gen=gen)
        D = kernels.pairwise_sqdist(y)
        brute = ((y.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
        assert _maxabs(D, brute) <= tol * 1e3
        assert bool((torch.diagonal(D) == 0).all()) and bool((D >= 0).all())
    print("PASS pairwise_sqdist matches the brute-force [N, N, d] expansion")


def test_median_bandwidth_matches_numpy_on_the_upper_triangle():
    for dtype, device, tol in _targets():
        gen = _gen(11, device)
        for N in (2, 3, 4, 5, 16, 64, 65):
            y = _randn(N, 3, dtype=dtype, device=device, gen=gen)
            D = kernels.pairwise_sqdist(y)
            h = kernels.median_bandwidth(D, N, floor=1e-3)
            Dn = D.detach().cpu().double().numpy()
            med = np.median(np.sqrt(Dn[np.triu_indices(N, k=1)]))
            expect = max(1e-6, med ** 2 / math.log(N))
            assert abs(float(h) - expect) <= tol * 1e3 * max(1.0, expect), (N, float(h), expect)
        y0 = torch.zeros(8, 3, dtype=dtype, device=device)
        assert abs(float(kernels.median_bandwidth(kernels.pairwise_sqdist(y0), 8, floor=0.2)) - 0.04) <= tol
        h1 = kernels.median_bandwidth(torch.zeros(1, 1, dtype=dtype, device=device), 1, floor=0.3)
        assert abs(float(h1) - 0.09) <= tol
    print("PASS median_bandwidth equals numpy's median over the upper triangle (odd and even pair counts)")


def test_rbf_repulsion_is_the_svgd_gradient():
    for dtype, device, tol in _targets():
        gen = _gen(12, device)
        y = _randn(21, 4, dtype=dtype, device=device, gen=gen)
        h = 1.7
        K, R = kernels.rbf_terms(y, h)
        assert _maxabs(K, K.T) <= tol * 10
        y1 = y.clone().requires_grad_(True)
        Kf = torch.exp(-((y1.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2) / h)
        (grad_first,) = torch.autograd.grad(Kf.sum(), y1)
        ## R_j = sum_i grad_{y_i} k(y_i, y_j) = -grad_first[j] by the symmetry of k.
        t = 1e-10 if dtype == torch.float64 else 1e-4
        assert _maxabs(R, -grad_first) <= t, f"R vs autograd: {_maxabs(R, -grad_first):.2e}"
        y2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype, device=device)
        _, R2 = kernels.rbf_terms(y2, 1.0)
        assert float(R2[1, 0]) > 0 and float(R2[0, 0]) < 0, "the repulsion pushes particles apart"
    print("PASS rbf_terms: K symmetric; R = sum_i grad_{y_i} K_ij (repulsive sign)")


def test_stein_direction_forms():
    """The three forms of `stein_direction`: kernel off is `-(gF + gC)` bitwise; the
    Tabor-Hermans form averages only the objective and the repulsion, `(1/N)`; the literal
    form averages the whole AL gradient. With `K = I` and `N = 1` all three agree."""
    for dtype, device, tol in _targets():
        gen = _gen(13, device)
        N, n = 16, 5
        gF = _randn(N, n, dtype=dtype, device=device, gen=gen)
        gC = _randn(N, n, dtype=dtype, device=device, gen=gen)
        R = _randn(N, n, dtype=dtype, device=device, gen=gen)
        y = _randn(N, 3, dtype=dtype, device=device, gen=gen)
        K, _ = kernels.rbf_terms(y, 2.0)
        off = kernels.stein_direction(K, R, gF, gC, kernel=False)
        assert bool((off == -(gF + gC)).all())
        th = kernels.stein_direction(K, R, gF, gC)
        assert _maxabs(th, (K @ (-gF) + R) / N - gC) <= tol * 10
        lit = kernels.stein_direction(K, R, gF, gC, inside=True)
        assert _maxabs(lit, (K @ (-(gF + gC)) + R) / N) <= tol * 10
        one = [kernels.stein_direction(torch.ones(1, 1, dtype=dtype, device=device),
                                       torch.zeros(1, n, dtype=dtype, device=device),
                                       gF[:1], gC[:1], inside=i, kernel=k)
               for i, k in ((False, True), (True, True), (False, False))]
        assert _maxabs(one[0], one[2]) <= tol and _maxabs(one[1], one[2]) <= tol
    print("PASS stein_direction: kernel off, Tabor-Hermans and literal forms; N = 1 collapses them")


def _run_svgd_gaussian(dtype, device, steps, lr, N=200):
    """Plain SVGD (the Tabor-Hermans form with no constraint) on a 2-D standard Gaussian."""
    gen = _gen(14, device)
    y = 0.3 * _randn(N, 2, dtype=dtype, device=device, gen=gen) + torch.tensor(
        [2.5, -1.5], dtype=dtype, device=device)
    zero = torch.zeros_like(y)
    for _ in range(steps):
        h = kernels.median_bandwidth(kernels.pairwise_sqdist(y), N, floor=1e-3)
        K, R = kernels.rbf_terms(y, h)
        y = y + lr * kernels.stein_direction(K, R, y, zero)    # grad f = y
    return y


def test_plain_svgd_matches_a_gaussians_moments():
    for dtype, device, tol in _targets():
        y = _run_svgd_gaussian(dtype, device, steps=3000, lr=0.5)
        mean_err = float(y.mean(0).abs().max())
        cov_err = float((torch.cov(y.T) - torch.eye(2, dtype=dtype, device=device)).abs().max())
        print(f"     {str(dtype):14s} {device.type}: |mean| {mean_err:.3f}  |cov - I| {cov_err:.3f}")
        assert mean_err <= 0.05, f"mean error {mean_err:.3f}"
        assert cov_err <= 0.08, f"covariance error {cov_err:.3f}"
    print("PASS plain SVGD on N(0, I) reproduces mean and covariance (the sign check)")


## --------------------------------------------------------------------------------------
## 4. The update: the GN Hessian, Stein variational Newton, the per-particle identity step
## --------------------------------------------------------------------------------------

def _toy_target(s, lo, hi, H_fx, w_cfg=0.0):
    """What `fused.direction_and_hessian` / `step_from` read off a target: the region scale
    `s`, the cost Hessian pieces and the bound projection."""
    def project(X):
        Xp = torch.minimum(torch.maximum(X, lo), hi)
        return Xp, (Xp - X).norm(dim=1)
    return SimpleNamespace(s=s, H_fx=H_fx, w_cfg=w_cfg, bp=SimpleNamespace(project=project))


def _toy_problem(dtype, device, gen, N=8, n=4, m_e=2, m_i=2):
    """f = 1/2 |x - a_i|^2 (Hessian I), linear rows h = A x - b, g = C x - d."""
    kw = dict(dtype=dtype, device=device)
    A = _randn(m_e, n, dtype=dtype, device=device, gen=gen)
    C = _randn(m_i, n, dtype=dtype, device=device, gen=gen)
    a = _randn(N, n, dtype=dtype, device=device, gen=gen)
    b = _randn(m_e, dtype=dtype, device=device, gen=gen)
    d = _randn(m_i, dtype=dtype, device=device, gen=gen)

    def rows(x):
        return 0.5 * ((x - a) ** 2).sum(1), x @ A.T - b, x @ C.T - d
    J_q = torch.zeros(N, n, n, **kw)          # no configuration-space cost here (w_cfg = 0)
    return A, C, rows, J_q, torch.eye(n, **kw)


def _direction(tg, sc, X, rows, J_q, A, C, S):
    xg = X.clone().requires_grad_(True)
    F, h, g = rows(xg)
    (gF,) = torch.autograd.grad(F.sum(), xg)
    N = X.shape[0]
    K = torch.eye(N, dtype=X.dtype, device=X.device)
    fin = torch.ones(N, dtype=torch.bool, device=X.device)
    J_h = A.unsqueeze(0).expand(N, -1, -1)
    J_g = C.unsqueeze(0).expand(N, -1, -1)
    return fused.direction_and_hessian(tg, sc, X, gF, J_q, J_h, J_g, h.detach(), g.detach(), fin,
                                       K, torch.zeros_like(X), False, S)


def test_gn_hessian_equals_the_autograd_hessian_on_linear_rows():
    """On linear rows the GN Hessian of the penalty part IS its Hessian: rho A^T A on the
    equalities plus rho C^T C on the PHR-active inequalities, in y = x / s -- checked against
    autograd's Hessian of `al_value(0, h, g)` at points away from the active-set kink."""
    for dtype, device, tol in _targets():
        gen = _gen(22, device)
        kw = dict(dtype=dtype, device=device)
        N, n, m_e, m_i = 6, 4, 2, 3
        A = _randn(m_e, n, dtype=dtype, device=device, gen=gen)
        C = _randn(m_i, n, dtype=dtype, device=device, gen=gen)
        S = _state(N, m_e, m_i, dtype, device, gen)
        s = torch.tensor([0.5, 2.0, 1.0, 3.0], **kw)
        X = _randn(N, n, dtype=dtype, device=device, gen=gen)
        g = X @ C.T - 0.1
        H = fused.gn_hessian(s, 0.0, torch.zeros(n, n, **kw), torch.zeros(N, n, n, **kw),
                             A.expand(N, -1, -1), C.expand(N, -1, -1), g, S.mu, RHO)
        for i in range(N):
            def pen(y):
                x = y * s
                return al.al_value(torch.zeros(1, **kw), (x @ A.T - 0.3).unsqueeze(0),
                                   (x @ C.T - 0.1).unsqueeze(0), al.replace(
                                       S, lam=S.lam[i:i + 1], mu=S.mu[i:i + 1]), RHO).sum()
            Ha = torch.autograd.functional.hessian(pen, X[i] / s)
            assert _maxabs(H[i], Ha) <= 1e3 * tol * max(1.0, _maxabs(Ha)), (dtype, i, _maxabs(H[i], Ha))
    print("PASS GN Hessian of the penalty part equals autograd's Hessian on linear rows (active set PHR)")


def test_svn_step_lands_on_the_minimiser_of_a_quadratic():
    """Stein variational Newton with the exact Hessian and delta = 0: on a quadratic L (the
    objective 1/2|x - a|^2 plus equality rows, whose GN Hessian is exact) one step with
    svgd_lr = 1, kernel off, no bound, lands on argmin L."""
    for dtype, device, tol in _targets():
        gen = _gen(23, device)
        kw = dict(dtype=dtype, device=device)
        A, C, rows, J_q, H_fx = _toy_problem(dtype, device, gen, m_i=0)
        N, n = 8, 4
        C = torch.zeros(0, n, **kw)
        S = _state(N, A.shape[0], 0, dtype, device, gen)
        s = torch.tensor([0.5, 2.0, 1.0, 3.0], **kw)
        tg = _toy_target(s, torch.full((n,), -1e9, **kw), torch.full((n,), 1e9, **kw), H_fx)
        sc = fused.StepConfig(kernel="none", bandwidth_floor=0.05, inside=False, lr=1.0,
                              rho=RHO, metric="gn", gn_lm=0.0)
        X = _randn(N, n, dtype=dtype, device=device, gen=gen)
        phi, H, _ = _direction(tg, sc, X, rows, J_q, A, C, S)
        Xn, _, _ = fused.step_from(tg, sc, X, phi, H)
        ## argmin_x 1/2|x - a|^2 + lam.(Ax - b) + rho/2 |Ax - b|^2, per particle (one exact
        ## Newton step from X on the quadratic, by autograd)
        xs = []
        for i in range(N):
            def L(x):
                F, h, g = rows(x.unsqueeze(0).expand(N, -1))
                return al.al_value(F[i:i + 1], h[i:i + 1], g[i:i + 1],
                                   al.replace(S, lam=S.lam[i:i + 1], mu=S.mu[i:i + 1]), RHO).sum()
            x = X[i].clone().requires_grad_(True)
            gr = torch.autograd.grad(L(x), x)[0]
            Hx = torch.autograd.functional.hessian(L, X[i])
            xs.append(X[i] - torch.linalg.solve(Hx, gr))
        x_star = torch.stack(xs)
        err = _maxabs(Xn, x_star) / max(1.0, _maxabs(x_star))
        assert err <= 1e4 * tol, f"{dtype}: SVN step misses the minimiser by {err:.2e}"
        print(f"     {str(dtype):14s} {device.type}: |x_SVN - x*| {err:.1e}")
    print("PASS one SVN step with the exact Hessian (delta = 0) lands on the quadratic's minimiser")


def test_identity_metric_is_projected_gradient_descent_with_the_frobenius_step():
    """With the kernel off and svgd_metric = identity the step IS projected gradient descent
    on L in y = x / s with the per-particle step svgd_lr / ||H_i||_F: checked against a
    hand-rolled PGD through `al_value`'s own autograd gradient, with the bound clamp."""
    for dtype, device, tol in _targets():
        gen = _gen(21, device)
        kw = dict(dtype=dtype, device=device)
        A, C, rows, J_q, H_fx = _toy_problem(dtype, device, gen)
        N, n = 8, 4
        S = _state(N, A.shape[0], C.shape[0], dtype, device, gen)
        s = torch.tensor([0.5, 2.0, 1.0, 3.0], **kw)
        lo = torch.tensor([-0.3, -1e9, -2.0, -1.0], **kw)
        hi = torch.tensor([0.3, 1e9, 2.0, 1.0], **kw)
        tg = _toy_target(s, lo, hi, H_fx)
        sc = fused.StepConfig(kernel="none", bandwidth_floor=0.05, inside=False, lr=1.125,
                              rho=RHO, metric="identity", gn_lm=0.0)
        ## lr 1.125 = the old lr 0.9 at T = 0.8: the same step, so the bound clamp is still hit.
        X = torch.zeros(N, n, **kw)
        X_ref = X.clone()
        clipped = 0
        for _ in range(5):
            phi, H, _ = _direction(tg, sc, X, rows, J_q, A, C, S)
            X, clip, n_clip = fused.step_from(tg, sc, X, phi, H)
            clipped += int(n_clip.sum())
            yg = (X_ref / s).clone().requires_grad_(True)
            F2, h2, g2 = rows(yg * s)
            (gy,) = torch.autograd.grad(al.al_value(F2, h2, g2, S, RHO).sum(), yg)
            fro = H.flatten(1).norm(dim=1).unsqueeze(1)
            y_new = yg.detach() - (sc.lr / fro) * gy
            X_ref = torch.minimum(torch.maximum(y_new * s, lo), hi)
            rel = _maxabs(X, X_ref) / max(1.0, _maxabs(X_ref))
            assert rel <= tol * 100, f"{dtype}: PGD mismatch (relative) {rel:.2e}"
        assert clipped > 0, "the test must exercise the bound clamp"
        ## ||H||_F bounds lambda_max from above (the step is a safe Lipschitz step)
        lam = fused.lambda_max(H)
        assert bool((H.flatten(1).norm(dim=1) >= lam * (1 - 1e-6)).all())
    print("PASS identity metric: PGD on L_rho in y = x / s with the step svgd_lr / ||H_i||_F")


def test_resample_mask():
    for dtype, device, tol in _targets():
        gen = _gen(16, device)
        N = 10
        q = _randn(N, 7, dtype=dtype, device=device, gen=gen)
        q[1, 3] = 50.0
        q[2, 0] = float("nan")
        q[3, 6] = float("inf")
        rows_finite = torch.ones(N, dtype=torch.bool, device=device)
        rows_finite[4] = False
        rm = al.resample_mask(q, rows_finite, q_max=10.0)
        expect = torch.zeros(N, dtype=torch.bool, device=device)
        expect[[1, 2, 3, 4]] = True
        assert bool((rm == expect).all()), rm
    print("PASS resample_mask: runaway (|q| > q_max), nan, inf and non-finite rows")


def test_live_weights_drop_particles_exactly():
    """`svgd_resample = drop` / `keep`: the live mask. With every particle live the masked
    bandwidth, kernel terms and direction are BIT-IDENTICAL to the unmasked ones (the native
    path); with some dropped, the live particles' kernel, repulsion and direction are exactly
    those of the swarm of live particles alone (bandwidth over the live pairs with log n_live,
    average over n_live), and a dropped particle is isolated and frozen. `al.drop` zeroes its
    multipliers and live weight."""
    for dtype, device, tol in _targets():
        gen = _gen(23, device)
        N, d, n = 9, 7, 5
        y = _randn(N, d, dtype=dtype, device=device, gen=gen)
        sq = kernels.pairwise_sqdist(y)
        all_live = torch.ones(N, dtype=torch.bool, device=device)
        h_ref = kernels.median_bandwidth(sq, N, floor=0.05)
        h_all = kernels.median_bandwidth(sq, N, floor=0.05, live=all_live)
        assert torch.equal(h_ref, h_all), (h_ref, h_all)
        live = all_live.clone()
        live[[2, 5, 6]] = False
        sub = y[live]
        h_sub = kernels.median_bandwidth(kernels.pairwise_sqdist(sub), int(live.sum()), floor=0.05)
        h_mask = kernels.median_bandwidth(sq, N, floor=0.05, live=live)
        assert _maxabs(h_sub, h_mask) <= tol * float(h_sub), (h_sub, h_mask)
        one = all_live.clone()
        one[1:] = False
        assert torch.equal(kernels.median_bandwidth(sq, N, floor=0.05, live=one),
                           torch.full((), 0.05 ** 2, dtype=dtype, device=device))

        sc = fused.StepConfig(kernel="q", bandwidth_floor=0.05, inside=False, lr=0.3,
                              rho=10.0)
        X = _randn(N, n, dtype=dtype, device=device, gen=gen)
        finite = torch.ones(N, dtype=torch.bool, device=device)
        K0, R0, _, _ = fused.kernel_terms(sc, y, finite, X)
        K1, R1, _, _ = fused.kernel_terms(sc, y, finite, X, all_live)
        assert torch.equal(K0, K1) and torch.equal(R0, R1)
        gF = _randn(N, n, dtype=dtype, device=device, gen=gen)
        gC = _randn(N, n, dtype=dtype, device=device, gen=gen)
        R_y = _randn(N, n, dtype=dtype, device=device, gen=gen)
        ones = torch.ones((), dtype=dtype, device=device)
        for inside in (False, True):
            p0 = kernels.stein_direction(K0, R_y, gF, gC, inside=inside)
            p1 = kernels.stein_direction(K0, R_y, gF, gC, inside=inside, avg_scale=ones)
            assert torch.equal(p0, p1)

        Km, Rm, _, _ = fused.kernel_terms(sc, y, finite, X, live)
        Xs = X[live]
        Ks, Rs, _, _ = fused.kernel_terms(sc, sub, finite[live], Xs)
        assert _maxabs(Km[live][:, live], Ks) <= tol and _maxabs(Rm[live], Rs) <= tol * 10
        dead = ~live
        assert bool((Km[dead][:, live] == 0).all()) and bool((Km[live][:, dead] == 0).all())
        assert bool((Rm[dead] == 0).all())
        ## The direction over n_live: the live rows of the masked swarm's phi are the sub-swarm's.
        n_live = int(live.sum())
        scale = torch.full((), N / n_live, dtype=dtype, device=device)
        pm = kernels.stein_direction(Km, R_y * live.unsqueeze(1), gF * live.unsqueeze(1), gC,
                                     avg_scale=scale)
        ps = kernels.stein_direction(Ks, R_y[live], gF[live], gC[live])
        assert _maxabs(pm[live], ps) <= tol * 10, _maxabs(pm[live], ps)

        S = ALState.init(N, 3, 2, dtype=dtype, device=device)
        S = ALState(lam=S.lam + 1.0, mu=S.mu + 2.0, live=S.live)
        Sd = al.drop(S, dead)
        assert bool((Sd.live[dead] == 0).all()) and bool((Sd.live[live] == 1).all())
        assert bool((Sd.lam[dead] == 0).all()) and bool((Sd.mu[dead] == 0).all())
        assert bool((Sd.lam[live] == 1).all()) and bool((Sd.mu[live] == 2).all())
        assert fused.live_mask(sc, Sd) is None          # native step config never reads it
        sc_l = fused.StepConfig(**{**sc.__dict__, "live_weights": True})
        assert torch.equal(fused.live_mask(sc_l, Sd), live)
    print("PASS live weights: all-live bit-identical to unmasked; dropped particles isolated, "
          "the live ones exactly the sub-swarm's kernel, bandwidth and direction")


## --------------------------------------------------------------------------------------
## 5. Graph-friendliness
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


def _compiled_chain(x, a, A, b, C, d, lam, mu):
    """al_value -> gradient -> the kernel-off Stein step -> the dual update -> reset of a
    resample mask, all tensors in and out (ALState rebuilt inside, tensors-only signature)."""
    S = ALState(lam=lam, mu=mu)

    def merit(xx):
        h = (A @ xx.unsqueeze(2)).squeeze(2) - b
        g = (C @ xx.unsqueeze(2)).squeeze(2) - d
        return al.al_value(0.5 * ((xx - a) ** 2).sum(1), h, g, S, RHO).sum(), (h, g)

    grad, (h, g) = torch.func.grad(merit, has_aux=True)(x)
    phi = kernels.stein_direction(torch.eye(x.shape[0], dtype=x.dtype, device=x.device),
                                  torch.zeros_like(x), grad, torch.zeros_like(x), kernel=False)
    x1 = x + (0.01 / RHO) * phi
    S, clipped = al.dual_update(h, g, S, RHO, 1e3)
    rm = al.resample_mask(x1, torch.isfinite(x1).all(dim=1), 1.0)
    S = al.reset(S, rm)
    return x1, S.lam, S.mu, clipped, rm


def test_compiled_chain_matches_eager():
    if not torch.cuda.is_available():
        print("SKIP compiled-chain smoke: no CUDA device")
        return
    dtype, device = torch.float64, torch.device("cuda")
    gen = _gen(17, device)
    N, n, m_e, m_i = 32, 6, 2, 3
    r = lambda *sh: _randn(*sh, dtype=dtype, device=device, gen=gen)
    S = _state(N, m_e, m_i, dtype, device, gen)
    args = (r(N, n), r(N, n), r(N, m_e, n), r(N, m_e), r(N, m_i, n), r(N, m_i), S.lam, S.mu)
    eager = _compiled_chain(*args)
    compiled = torch.compile(_compiled_chain, dynamic=False, fullgraph=True)
    out = compiled(*args)
    out2 = compiled(*args)
    torch.cuda.synchronize()
    worst = 0.0
    for e, c, c2 in zip(eager, out, out2):
        assert bool((c == c2).all()), "compiled call is not deterministic"
        worst = max(worst, _maxabs(e.double(), c.double()))
    assert worst <= 1e-10, f"compiled vs eager {worst:.2e}"
    print(f"     fullgraph torch.compile(dynamic=False) on CUDA float64 vs eager: {worst:.1e}")
    print("PASS compiled AL chain matches eager (full graph, no breaks)")


## --------------------------------------------------------------------------------------
## 6. The kernel spaces (2026-10-11)
## --------------------------------------------------------------------------------------

## The fielded step as it stood before the kernel spaces and the metric's temperature
## (svgd tip 5f15f55), VERBATIM, as the reference the fielded configuration must reproduce
## bit for bit: `svgd_kernel = q` (then with `svgd_metric_temperature = off` at T = 1). The one
## edit since: the temperature argument is gone from `stein_direction`'s call, the option having
## been removed 2026-10-14 (at T = 1 the division it made was exact).

def _head_kernel_terms(sc, cfg, finite, X, live=None):
    N = X.shape[0]
    dtype, device = X.dtype, X.device
    if sc.kernel == "none" or N < 2:
        return (torch.eye(N, dtype=dtype, device=device),
                torch.zeros_like(cfg), sc.kernel != "none")
    y = cfg.detach()
    fin = finite & torch.isfinite(y).all(dim=1)
    if live is not None:
        fin = fin & live
    y0 = torch.where(fin.unsqueeze(1), y, torch.zeros_like(y))
    hbw = kernels.median_bandwidth(kernels.pairwise_sqdist(y0), N, floor=sc.bandwidth_floor,
                                   live=live)
    K, _ = kernels.rbf_terms(y0, hbw)
    mask = fin.unsqueeze(1) & fin.unsqueeze(0)
    K = torch.where(mask, K, torch.eye(N, dtype=dtype, device=device))
    R = (2.0 / hbw) * (K.sum(dim=1, keepdim=True) * y0 - K @ y0)
    R = torch.where(fin.unsqueeze(1), R, torch.zeros_like(R))
    return K, R, True


def _head_direction_and_hessian(tg, sc, X, gF, J_q, J_h, J_g, h, g, finite, K, R, kernel_on, S,
                                live=None):
    _bmv = fused._bmv
    N, n = X.shape
    avg_scale = None
    if live is not None:
        finite = finite & live
        n_live = live.sum()
        avg_scale = torch.where(n_live == N, torch.ones((), dtype=X.dtype, device=X.device),
                                N / torch.clamp(n_live, min=1).to(X.dtype))
    ch, cg = al.al_constraint_grad_coefficients(h, g, S, sc.rho)
    gC = _bmv(J_h.transpose(1, 2), ch) + _bmv(J_g.transpose(1, 2), cg)
    R_x = _bmv(J_q.transpose(1, 2), R) if kernel_on else torch.zeros_like(X)
    sv = tg.s.unsqueeze(0)
    fin = finite.unsqueeze(1)
    zero = torch.zeros_like(X)
    gF_y = torch.where(fin, sv * gF, zero)
    gC_y = torch.where(fin, sv * gC, zero)
    R_y = torch.where(fin, sv * R_x, zero)
    use_kernel = sc.kernel != "none"
    phi = kernels.stein_direction(K, R_y, gF_y, gC_y, inside=sc.inside, kernel=use_kernel,
                                  avg_scale=avg_scale)
    rep = R_y / N if use_kernel else zero
    if use_kernel and avg_scale is not None:
        rep = rep * avg_scale
    ratio = rep.norm(dim=1) / torch.clamp((phi - rep).norm(dim=1), min=torch.finfo(X.dtype).tiny)
    ratio = torch.nanmedian(torch.where(finite, ratio, torch.full_like(ratio, float("nan"))))
    H = fused.gn_hessian(tg.s, tg.w_cfg, tg.H_fx, J_q, J_h, J_g, g, S.mu, sc.rho)
    eye = torch.eye(n, dtype=X.dtype, device=X.device).unsqueeze(0).expand(N, -1, -1)
    ok = finite & torch.isfinite(H).flatten(1).all(dim=1)
    H = torch.where(ok.view(-1, 1, 1), H, eye)
    return phi, H, ratio


def _head_step_from(tg, sc, X, phi, H, live=None):
    n = X.shape[1]
    if sc.metric == "gn":
        eye = torch.eye(n, dtype=X.dtype, device=X.device).unsqueeze(0)
        Y, info = torch.linalg.solve_ex(H + sc.gn_lm * eye, phi.unsqueeze(2), check_errors=False)
        dy = sc.lr * Y.squeeze(2)
        good = (info == 0) & torch.isfinite(dy).all(dim=1)
    else:
        fro = torch.clamp(H.flatten(1).norm(dim=1), min=torch.finfo(X.dtype).tiny)
        dy = (sc.lr / fro).unsqueeze(1) * phi
        good = torch.isfinite(dy).all(dim=1)
    dy = torch.where(good.unsqueeze(1), dy, torch.zeros_like(dy))
    sv = tg.s.unsqueeze(0)
    Xn = X + sv * dy
    Xp, _ = tg.bp.project(Xn)
    if live is not None:
        l1 = live.unsqueeze(1)
        Xp = torch.where(l1, Xp, X)
        Xn = torch.where(l1, Xn, X)
    moved = (Xp != Xn) & torch.isfinite(Xn)
    clip = torch.where(moved, (Xp - Xn) / sv, torch.zeros_like(X)).norm(dim=1)
    return Xp.detach(), clip, moved.sum(dim=1)


W_Z, ND = 3, 4                     # the toy learned layout: [c6 | z (3) | q_c (4)], 13 variables
N_VARS = 6 + W_Z + ND


def _toy_learned(dtype, device, gen, N=9, m_e=3, m_i=2, learned=True):
    """A learned-arm-shaped toy: region scale s, a box, the block layout `BatchedProgram`
    exposes (`is_learned`, `width`, `ndof`), a LINEAR configuration map `cfg = X @ B^T` (so
    `J_q = B`, exactly) and a linear task-pose map `pose = cfg @ P^T` (`d pose / d cfg = P`)."""
    kw = dict(dtype=dtype, device=device)
    n = N_VARS if learned else ND
    s = 0.5 + _randn(n, dtype=dtype, device=device, gen=gen).abs()
    lo, hi = torch.full((n,), -3.0, **kw), torch.full((n,), 3.0, **kw)

    def project(X):
        Xp = torch.minimum(torch.maximum(X, lo), hi)
        return Xp, (Xp - X).norm(dim=1)
    bp = SimpleNamespace(project=project, is_learned=learned, width=W_Z if learned else 0,
                         ndof=ND)
    tg = SimpleNamespace(s=s, H_fx=0.1 * torch.eye(n, **kw), w_cfg=0.3, bp=bp)
    B = _randn(ND, n, dtype=dtype, device=device, gen=gen) if learned else torch.eye(ND, **kw)
    P = _randn(7, ND, dtype=dtype, device=device, gen=gen)
    X = _randn(N, n, dtype=dtype, device=device, gen=gen)
    p = SimpleNamespace(
        tg=tg, B=B, P=P, X=X, cfg=X @ B.T, J_q=B.unsqueeze(0).expand(N, -1, -1).contiguous(),
        gF=_randn(N, n, dtype=dtype, device=device, gen=gen),
        J_h=_randn(N, m_e, n, dtype=dtype, device=device, gen=gen),
        J_g=_randn(N, m_i, n, dtype=dtype, device=device, gen=gen),
        h=_randn(N, m_e, dtype=dtype, device=device, gen=gen),
        g=_randn(N, m_i, dtype=dtype, device=device, gen=gen),
        S=_state(N, m_e, m_i, dtype, device, gen),
        finite=torch.ones(N, dtype=torch.bool, device=device))
    p.pose = p.cfg @ P.T
    p.pose_jac = P.unsqueeze(0).expand(N, -1, -1).contiguous()
    return p


def _sc(kernel="q", **kw):
    base = dict(kernel=kernel, bandwidth_floor=0.05, inside=False, lr=0.3, rho=10.0,
                metric="gn", gn_lm=1.0)
    base.update(kw)
    return fused.StepConfig(**base)


def _head_chain(tg, sc, X, cfg, gF, J_q, J_h, J_g, h, g, finite, lam, mu, live=None):
    S = ALState(lam=lam, mu=mu)
    K, R, kon = _head_kernel_terms(sc, cfg, finite, X, live)
    phi, H, ratio = _head_direction_and_hessian(tg, sc, X, gF, J_q, J_h, J_g, h, g, finite, K, R,
                                                kon, S, live)
    Xn, clip, n_clip = _head_step_from(tg, sc, X, phi, H, live)
    return K, R, phi, H, ratio, Xn, clip, n_clip


def _new_chain(tg, sc, X, cfg, gF, J_q, J_h, J_g, h, g, finite, lam, mu, live=None):
    S = ALState(lam=lam, mu=mu)
    K, R, kon, _ = fused.kernel_terms(sc, cfg, finite, X, live, tg=tg)
    phi, H, ratio = fused.direction_and_hessian(tg, sc, X, gF, J_q, J_h, J_g, h, g, finite, K, R,
                                                kon, S, live)
    Xn, clip, n_clip = fused.step_from(tg, sc, X, phi, H, live)
    return K, R, phi, H, ratio, Xn, clip, n_clip


def test_fielded_kernel_and_metric_reproduce_the_previous_step_bitwise():
    """(a) `svgd_kernel = q` -- the step fielded until 2026-10-12, which every svgd round through
    R19 ran (with metric off; the pin is now c x z, and the metric form is no longer an option) -- gives
    the kernel, repulsion, direction, Hessian, ratio and step of the pre-2026-10-11 code
    (`_head_*`, verbatim) BIT FOR BIT, under both metrics, with and without a dropped particle
    and a non-finite one; eager on every target, and compiled on CUDA (the compiled new chain
    against the compiled old one). The end-to-end twin -- a 30-step solve per task x arm x
    eager / compiled / graphed against arrays saved at the old tip -- was run when the change
    landed (docs/svgd-solver.md, the options table)."""
    for dtype, device, tol in _targets():
        gen = _gen(31, device)
        p = _toy_learned(dtype, device, gen)
        N = p.X.shape[0]
        fin = p.finite.clone()
        fin[4] = False
        live = torch.ones(N, dtype=torch.bool, device=device)
        live[2] = False
        for metric in ("gn", "identity"):
            sc = _sc("q", metric=metric)
            for lv in (None, live):
                args = (p.tg, sc, p.X, p.cfg, p.gF, p.J_q, p.J_h, p.J_g, p.h, p.g, fin,
                        p.S.lam, p.S.mu, lv)
                ref, new = _head_chain(*args), _new_chain(*args)
                for a, b in zip(ref, new):
                    assert torch.equal(a, b), (dtype, metric, lv is None)
        if device.type == "cuda":
            sc = _sc("q")
            args = (p.tg, sc, p.X, p.cfg, p.gF, p.J_q, p.J_h, p.J_g, p.h, p.g, fin,
                    p.S.lam, p.S.mu, live)
            ref = torch.compile(_head_chain, dynamic=False)(*args)
            new = torch.compile(_new_chain, dynamic=False)(*args)
            for a, b in zip(ref, new):
                assert torch.equal(a, b), "compiled"
            print(f"     compiled CUDA {dtype}: bit-identical")
    print("PASS svgd_kernel=q (the pre-2026-10-12 pin): the previous step bit for bit")


NEW_SPACES = ("x", "z", "c", "cz", "qz", "pose")


def _support(space, n):
    """The decision-variable coordinates a DIRECT (y-routed) repulsion may touch."""
    c, z = set(range(6)), set(range(6, 6 + W_Z))
    return {"x": set(range(n)), "z": z, "c": c, "cz": c | z}.get(space)


def _kernel_by_autograd(space, tg, Y, B, P, hs):
    """`sum_j K(y_i, y_j)` built from y with autograd's graph, at FIXED bandwidths `hs`, and
    the kernel's features of y -- the same spaces as `fused.kernel_features`, written out."""
    X = Y * tg.s
    cfg = X @ B.T
    pose = cfg @ P.T
    c, z = Y[:, :6], Y[:, 6:6 + W_Z]
    feats = {"q": [cfg], "x": [Y], "z": [z], "c": [c], "cz": [c, z], "qz": [cfg, z],
             "pose": [pose]}[space]
    return feats


def test_new_kernel_spaces():
    """(b) every new space gives a finite direction on a random swarm; its repulsion is the
    exact SVGD repulsion of its (product) RBF -- `R_j = sum_i grad_{y_i} K(y_i, y_j)` -- by
    autograd through the features at the bandwidths it used, with the configuration and pose
    factors pulled back through J_q (and d pose / d cfg); it is zero when the particles
    coincide in that space and nonzero when they do not; and the DIRECT spaces' repulsion lies
    in its block (zero outside it) and does not depend on J_q. `bw` has one bandwidth per
    factor. (c) `cz` at equal bandwidths is the single RBF on the stacked [c; z]."""
    for dtype, device, tol in _targets():
        gen = _gen(33, device)
        p = _toy_learned(dtype, device, gen)
        tg, N, n = p.tg, p.X.shape[0], p.X.shape[1]
        for space in NEW_SPACES:
            sc = _sc(space)
            K, R, kon, bw = fused.kernel_terms(sc, p.cfg, p.finite, p.X, tg=tg, pose=p.pose,
                                               pose_jac=p.pose_jac)
            assert kon and isinstance(R, tuple)
            assert bw.shape == ((2,) if space in ("cz", "qz") else (1,)), (space, bw.shape)
            R_y = fused.repulsion_y(tg, p.X, p.J_q, R, kon, p.finite)
            phi, H, ratio = fused.direction_and_hessian(tg, sc, p.X, p.gF, p.J_q, p.J_h, p.J_g,
                                                        p.h, p.g, p.finite, K, R, kon, p.S)
            assert bool(torch.isfinite(phi).all()) and bool(torch.isfinite(R_y).all()), space
            assert float(R_y.abs().max()) > 1e-6, f"{space}: no repulsion on a random swarm"
            ## the exact repulsion, by autograd at the bandwidths used
            Y = (p.X / tg.s).detach()
            Y1 = Y.clone().requires_grad_(True)
            f1 = _kernel_by_autograd(space, tg, Y1, p.B, p.P, bw)
            f2 = _kernel_by_autograd(space, tg, Y, p.B, p.P, bw)
            expo = sum(((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(2) / hk
                       for a, b, hk in zip(f1, f2, bw))
            Ka = torch.exp(-expo)
            assert _maxabs(K, Ka.detach()) <= 1e3 * tol, (space, _maxabs(K, Ka))
            (G,) = torch.autograd.grad(Ka.sum(), Y1)
            err = _maxabs(R_y, -G) / max(1.0, _maxabs(G))
            assert err <= 1e3 * tol, (space, dtype, err)
            ## direct spaces: in their block, and independent of J_q
            sup = _support(space, n)
            if sup is not None:
                outside = [k for k in range(n) if k not in sup]
                if outside:
                    assert bool((R_y[:, outside] == 0).all()), space
                R_y2 = fused.repulsion_y(tg, p.X, 7.0 * p.J_q, R, kon, p.finite)
                assert torch.equal(R_y, R_y2), space
            ## coincident in the space -> no repulsion; the other blocks are free to differ
            Xc, cfgc, posec = p.X.clone(), p.cfg.clone(), p.pose.clone()
            if space in ("x",):
                Xc[:] = p.X[0]
            if space in ("z", "cz", "qz"):
                Xc[:, 6:6 + W_Z] = p.X[0, 6:6 + W_Z]
            if space in ("c", "cz"):
                Xc[:, :6] = p.X[0, :6]
            if space == "qz":
                cfgc[:] = p.cfg[0]
            if space == "pose":
                posec[:] = p.pose[0]
            Kc, Rc, _, _ = fused.kernel_terms(sc, cfgc, p.finite, Xc, tg=tg, pose=posec,
                                              pose_jac=p.pose_jac)
            Rc_y = fused.repulsion_y(tg, Xc, p.J_q, Rc, True, p.finite)
            ## float32: coincident rows still differ by roundoff in K @ u against K 1 * u, which
            ## the floored bandwidth's 2 / h = 800 amplifies; float64 pins the exact zero.
            small = 1e-9 if dtype == torch.float64 else 2e-2
            assert _maxabs(Rc_y) <= small * max(1.0, _maxabs(R_y)), (space, _maxabs(Rc_y))
            assert _maxabs(Xc, p.X) > 0.1 or space == "x" or space == "pose", space
        ## (c) cz at equal bandwidths = one RBF on [c; z]: a floor far above the median
        sc = _sc("cz", bandwidth_floor=50.0)
        K, R, kon, bw = fused.kernel_terms(sc, p.cfg, p.finite, p.X, tg=tg)
        assert torch.equal(bw, torch.full((2,), 2500.0, dtype=dtype, device=device))
        Y = p.X / tg.s
        u = torch.cat([Y[:, :6], Y[:, 6:6 + W_Z]], dim=1)
        K1, R1 = kernels.rbf_terms(u, 2500.0)
        R_y = fused.repulsion_y(tg, p.X, p.J_q, R, kon, p.finite)
        assert _maxabs(K, K1) <= 1e2 * tol and _maxabs(R_y[:, :6 + W_Z], R1) <= 1e2 * tol * max(1.0, _maxabs(R1))
        assert bool((R_y[:, 6 + W_Z:] == 0).all())
        ## N = 1 and none: no interaction
        K0, R0, kon0, bw0 = fused.kernel_terms(_sc("cz"), p.cfg[:1], p.finite[:1], p.X[:1], tg=tg)
        assert torch.equal(K0, torch.ones(1, 1, dtype=dtype, device=device)) and bw0.shape == (2,)
        Kn, Rn, konn, bwn = fused.kernel_terms(_sc("none"), p.cfg, p.finite, p.X, tg=tg)
        assert not konn and bwn.shape == (0,)
    print("PASS kernel spaces x, z, c, cz, qz, pose: exact repulsion by autograd, block support, "
          "zero on coincidence; cz at equal bandwidths = RBF on [c; z]")


def test_pose_kernel_ignores_self_motion():
    """(d) `pose`: two particles with the SAME task pose and DIFFERENT latents (and
    configurations) -- the flow's self-motion -- repel not at all, while a third with a
    different pose is repelled; the quaternion is canonicalised first, so the same rotation
    given as q and -q is the same pose; `fused.pose_and_jacobian` is the 7-vector
    [position; canonical quaternion]."""
    for dtype, device, tol in _targets():
        gen = _gen(34, device)
        p = _toy_learned(dtype, device, gen, N=3)
        tg = p.tg
        X = p.X.clone()
        X[1, 6:6 + W_Z] = X[0, 6:6 + W_Z] + 1.5                         # a different z ...
        cfg = X @ p.B.T
        quat = torch.nn.functional.normalize(_randn(3, 4, dtype=dtype, device=device, gen=gen), dim=1)
        quat[1] = -quat[0]                                             # ... the same rotation
        pos = _randn(3, 3, dtype=dtype, device=device, gen=gen)
        pos[1] = pos[0]                                                # ... and position
        Jp = _randn(3, 3, ND + 2, dtype=dtype, device=device, gen=gen)
        Jw = _randn(3, 3, ND + 2, dtype=dtype, device=device, gen=gen)
        pose, pose_jac = fused.pose_and_jacobian(quat, pos, Jp, Jw, ND)
        assert pose.shape == (3, 7) and pose_jac.shape == (3, 7, ND)
        assert bool((pose[:, 3] >= 0).all()) and _maxabs(pose[0], pose[1]) <= tol
        sc = _sc("pose")
        for pair in ((0, 1), (0, 2)):
            idx = list(pair)
            K, R, kon, _ = fused.kernel_terms(sc, cfg[idx], p.finite[idx], X[idx], tg=tg,
                                              pose=pose[idx], pose_jac=pose_jac[idx])
            R_y = fused.repulsion_y(tg, X[idx], p.J_q[idx], R, kon, p.finite[idx])
            if pair == (0, 1):
                small = 1e-9 if dtype == torch.float64 else 1e-3
                assert _maxabs(R_y) <= small, ("self-motion repelled", _maxabs(R_y))
                Kq, Rq, _, _ = fused.kernel_terms(_sc("q"), cfg[idx], p.finite[idx], X[idx])
                assert _maxabs(fused.repulsion_y(tg, X[idx], p.J_q[idx], Rq, True, p.finite[idx])) > 1e-3
            else:
                assert _maxabs(R_y) > 1e-3, "different poses must repel"
    print("PASS pose kernel: same pose, different z -> no repulsion; different pose -> repulsion")


def test_pose_jacobian_matches_finite_differences_on_the_panda():
    """`fused.pose_and_jacobian` (the quaternion's derivative from the world angular-velocity
    Jacobian, `1/2 [-v^T; w I - [v]x] Jw`, at the canonical quaternion) against central
    differences of the batched kinematics of the real Panda grasp frame (`between_fingers`,
    `program.frame`), float64. Skipped where the Panda programs cannot be built."""
    try:
        sys.path.insert(0, HERE)
        import test_batched_program as TB
        bp = TB.batched("panda", "mug", "numerical", dtype=torch.float64)
    except Exception as exc:                        # pragma: no cover
        print(f"SKIP pose Jacobian on the Panda: {type(exc).__name__}: {exc}")
        return
    rng = np.random.default_rng(3)
    lo, hi = (b.detach().cpu().numpy() for b in bp.bounds)
    Q = torch.tensor(rng.uniform(lo, hi, size=(6, bp.ndof)), dtype=torch.float64, device=bp.device)
    q_plant, (quat, pos), _, (Jp, Jw) = bp.kinematics(Q, row_jacobians=True)
    pose, J = fused.pose_and_jacobian(quat, pos, Jp, Jw, bp.ndof)
    eps = 1e-6
    Jfd = torch.zeros_like(J)
    for k in range(bp.ndof):
        d = torch.zeros_like(Q)
        d[:, k] = eps
        _, (qa, pa), _, _ = bp.kinematics(Q + d)
        _, (qb, pb), _, _ = bp.kinematics(Q - d)
        pa7 = torch.cat([pa, fused.canonical_quat(qa)], dim=1)
        pb7 = torch.cat([pb, fused.canonical_quat(qb)], dim=1)
        Jfd[:, :, k] = (pa7 - pb7) / (2 * eps)
    err = _maxabs(J, Jfd)
    assert err <= 1e-7, f"pose Jacobian vs central differences: {err:.2e}"
    ## The autograd path's twin (`_Target.pose_and_jacobian_ev`, for a robot behind its own
    ## pose provider) agrees with the closed form.
    from src.svgd.solver import _Target
    tg = _Target(bp, 1e-4)
    Xg = Q.clone().requires_grad_(True)
    out = bp.evaluate(Xg, need_collision=False)
    pose_a, J_a = tg.pose_and_jacobian_ev(SimpleNamespace(out=out, cfg=out.q))
    err_a = max(_maxabs(pose_a, pose), _maxabs(J_a, J))
    assert err_a <= 1e-10, f"autograd pose Jacobian vs closed form: {err_a:.2e}"
    print(f"PASS pose Jacobian (canonical quaternion via Jw) vs central differences on the Panda "
          f"grasp frame: {err:.1e}; the autograd path's agrees to {err_a:.1e}")
    TB.close_all()


def test_joint_space_refuses_latent_kernels():
    """(f) `z`, `c`, `cz` and `qz` name blocks the joint-space arm does not have: refused at
    solver CONSTRUCTION (`fused.check_kernel_for_arm`, called by `SvgdSolver.__init__`, which
    `WarmUpSvgdStep` runs before the first cell), never from inside the step; `q`, `x`, `pose`
    and `none` are accepted there, and every space is accepted on the learned arm. Every space
    passes ProgramOptions, which does not know the arm."""
    from src.generic_program import ProgramOptions
    from src.svgd.solver import SvgdSolver
    for space in fused.KERNEL_SPACES:
        ProgramOptions(which_solver="svgd", svgd_kernel=space)
        fused.check_kernel_for_arm(space, True)
        joint = SimpleNamespace(options=ProgramOptions(which_solver="svgd", svgd_kernel=space))
        learned = SimpleNamespace(options=joint.options, z=None)
        if space in fused.LATENT_KERNELS:
            for call in (lambda: fused.check_kernel_for_arm(space, False),
                         lambda: SvgdSolver(joint)):
                try:
                    call()
                    raise AssertionError(f"svgd_kernel={space} accepted on joint space")
                except ValueError as exc:
                    assert space in str(exc) and "joint-space" in str(exc), str(exc)
        else:
            fused.check_kernel_for_arm(space, False)
            SvgdSolver(joint)
        SvgdSolver(learned)
    try:
        ProgramOptions(svgd_kernel="rbf")
        raise AssertionError("svgd_kernel='rbf' accepted")
    except ValueError:
        pass
    print("PASS z / c / cz / qz refused on the joint-space arm at construction; q / x / pose / none accepted")


GROUPS = {"fast": [test_al_value_and_coefficients_agree_with_autograd,
                   test_al_with_zero_multipliers_is_the_quadratic_penalty,
                   test_dual_update_is_unconditional_and_per_particle,
                   test_dual_update_counts_clips_by_row_group,
                   test_al_rounds_converge_to_the_kkt_point_on_every_particle,
                   test_pairwise_sqdist_matches_brute_force,
                   test_median_bandwidth_matches_numpy_on_the_upper_triangle,
                   test_rbf_repulsion_is_the_svgd_gradient,
                   test_stein_direction_forms,
                   test_plain_svgd_matches_a_gaussians_moments,
                   test_gn_hessian_equals_the_autograd_hessian_on_linear_rows,
                   test_svn_step_lands_on_the_minimiser_of_a_quadratic,
                   test_identity_metric_is_projected_gradient_descent_with_the_frobenius_step,
                   test_resample_mask, test_live_weights_drop_particles_exactly,
                   test_no_host_synchronising_calls_in_the_modules,
                   test_compiled_chain_matches_eager,
                   test_fielded_kernel_and_metric_reproduce_the_previous_step_bitwise,
                   test_new_kernel_spaces, test_pose_kernel_ignores_self_motion,
                   test_pose_jacobian_matches_finite_differences_on_the_panda,
                   test_joint_space_refuses_latent_kernels]}


if __name__ == "__main__":
    _tiers.run(GROUPS)
    print("ALL PASS", flush=True)
    ## Exit without interpreter finalization. With CUDA visible, torch's autograd device thread
    ## (`pt_autograd_0`) can drop its last GraphTask while CPython is finalizing; the decref
    ## asks for the GIL, Python 3.12 answers with pthread_exit, and the forced unwind through a
    ## noexcept destructor is std::terminate -- rc 134 after ALL PASS (gdb, 2026-10-22; also 1
    ## in 20 runs of a bare torch CUDA backward, so torch's, not this repo's). A failing test
    ## has raised before this line.
    sys.stderr.flush()
    os._exit(0)
