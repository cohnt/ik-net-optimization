"""`src/svgd/al.py` and `src/svgd/kernels.py` (and `fused.update_step`) on synthetic problems.

The modules are pure tensor math on per-particle quantities and know nothing about any
formulation, so they are tested on problems whose answers are known in closed form: the PHR
augmented Lagrangian (one fixed, shared penalty rho) against autograd and against the
quadratic penalty it reduces to; the dual-ascent step -- unconditional, per particle, step
rho, clipped and counted, zeroed on resample; AL rounds converging to the KKT point of an
equality-constrained quadratic, one per particle; the RBF repulsion against autograd and, the check that pins its SIGN, plain
SVGD on a 2-D standard Gaussian matching the target's mean and covariance; the Tabor-Hermans
direction with the kernel off being exactly projected gradient descent on L / T; and the
step size being `svgd_lr / rho` with the one shared rho.

Everything runs on CPU float64 and, where a CUDA device exists, on CUDA float32 with the
tolerances scaled to that precision. The last tests grep both modules for the
host-synchronising calls that would break CUDA-graph capture and compile a composition of
the AL functions with `torch.compile(dynamic=False, fullgraph=True)` against eager.

No pytest config in this repo: plain `test_*` functions with a `__main__` driver.
"""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype=dtype, device=device)


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
    particle's multipliers move by its OWN rows times the one shared rho; clipped to the box
    with the clipped entries counted; zeroed on the masked (resampled) particles only."""
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
        S2, n_clipped = al.dual_update(h, g, S, RHO, mult_max)
        lam_raw, mu_raw = lam + RHO * h, torch.clamp(mu + RHO * g, min=0.0)
        assert _maxabs(S2.lam, torch.clamp(lam_raw, -mult_max, mult_max)) <= tol * 10
        assert _maxabs(S2.mu, torch.clamp(mu_raw, max=mult_max)) <= tol * 10
        assert bool((S2.lam != lam).any(dim=1).all()), "every particle's lam moves, feasible or not"
        exp_clip = int((lam_raw.abs() > mult_max).sum() + (mu_raw > mult_max).sum())
        assert int(n_clipped) == exp_clip and exp_clip > 0, (int(n_clipped), exp_clip)
        assert float(S2.mu.min()) >= 0.0 and float(S2.lam.abs().max()) <= mult_max
        ## per particle: particle j's step depends on particle j's rows only
        h_alt = h.clone()
        h_alt[5] += 1.0
        S3, _ = al.dual_update(h_alt, g, S, RHO, 1e9)
        S4, _ = al.dual_update(h, g, S, RHO, 1e9)
        diff = (S3.lam - S4.lam).abs().amax(dim=1)
        assert float(diff[5]) > 0 and float(torch.cat([diff[:5], diff[6:]]).max()) == 0.0
        ## reset: a redrawn particle gets zero multipliers, the others are untouched
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[[1, 4]] = True
        S5 = al.reset(S2, mask)
        assert bool((S5.lam[mask] == 0).all()) and bool((S5.mu[mask] == 0).all())
        assert bool((S5.lam[~mask] == S2.lam[~mask]).all()) and bool((S5.mu[~mask] == S2.mu[~mask]).all())
    print("PASS dual update: unconditional, per particle, step rho, clipped and counted, zeroed on resample")


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
    """The three forms of `stein_direction`: kernel off is `-(gF + gC)/T` bitwise; the
    Tabor-Hermans form averages only the objective and the repulsion, `(1/N)`; the literal
    form averages the whole AL gradient. With `K = I` and `N = 1` all three agree."""
    for dtype, device, tol in _targets():
        gen = _gen(13, device)
        N, n, T = 16, 5, 0.7
        gF = _randn(N, n, dtype=dtype, device=device, gen=gen)
        gC = _randn(N, n, dtype=dtype, device=device, gen=gen)
        R = _randn(N, n, dtype=dtype, device=device, gen=gen)
        y = _randn(N, 3, dtype=dtype, device=device, gen=gen)
        K, _ = kernels.rbf_terms(y, 2.0)
        off = kernels.stein_direction(K, R, gF, gC, T, kernel=False)
        assert bool((off == -(gF + gC) / T).all())
        th = kernels.stein_direction(K, R, gF, gC, T)
        assert _maxabs(th, (K @ (-gF / T) + R) / N - gC / T) <= tol * 10
        lit = kernels.stein_direction(K, R, gF, gC, T, inside=True)
        assert _maxabs(lit, (K @ (-(gF + gC) / T) + R) / N) <= tol * 10
        one = [kernels.stein_direction(torch.ones(1, 1, dtype=dtype, device=device),
                                       torch.zeros(1, n, dtype=dtype, device=device),
                                       gF[:1], gC[:1], T, inside=i, kernel=k)
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
        y = y + lr * kernels.stein_direction(K, R, y, zero, 1.0)    # grad f = y
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
## 4. The update: projected gradient descent with kernel off, step svgd_lr / rho_i
## --------------------------------------------------------------------------------------

def _toy_target(s, lo, hi):
    """The two attributes `fused.update_step` reads off a target: the region scale `s` and
    the bound projection."""
    def project(X):
        Xp = torch.minimum(torch.maximum(X, lo), hi)
        return Xp, (Xp - X).norm(dim=1)
    return SimpleNamespace(s=s, bp=SimpleNamespace(project=project))


def test_kernel_none_is_projected_gradient_descent_on_L():
    """With the kernel off the Tabor-Hermans step IS projected gradient descent on L_rho / T
    in the normalised coordinates: `y <- clamp_B(y - (svgd_lr / rho_i) grad_y L_i / T)`, i.e.
    `x <- clamp_B(x - (svgd_lr / rho_i) s^2 grad_x L_i / T)` -- checked against a hand-rolled
    PGD on a quadratic objective with linear equality and inequality rows, through
    `al_value`'s own autograd gradient, over several steps and with the bound clamping."""
    for dtype, device, tol in _targets():
        gen = _gen(21, device)
        kw = dict(dtype=dtype, device=device)
        N, n, m_e, m_i = 8, 4, 2, 2
        s = torch.tensor([0.5, 2.0, 1.0, 3.0], **kw)
        lo = torch.tensor([-0.3, -1e9, -2.0, -1.0], **kw)
        hi = torch.tensor([0.3, 1e9, 2.0, 1.0], **kw)
        tg = _toy_target(s, lo, hi)
        A = _randn(m_e, n, dtype=dtype, device=device, gen=gen)
        C = _randn(m_i, n, dtype=dtype, device=device, gen=gen)
        a = _randn(N, n, dtype=dtype, device=device, gen=gen)
        S = _state(N, m_e, m_i, dtype, device, gen)
        sc = fused.StepConfig(kernel="none", bandwidth_floor=0.05, inside=False, lr=0.5, T=0.8,
                              rho=RHO)

        def rows(x):
            return 0.5 * ((x - a) ** 2).sum(1), x @ A.T - 0.1, x @ C.T - 0.2
        X = torch.zeros(N, n, **kw)
        X_ref = X.clone()
        clipped = 0
        for _ in range(5):
            xg = X.clone().requires_grad_(True)
            F, h, g = rows(xg)
            (gF,) = torch.autograd.grad(F.sum(), xg)
            ch, cg = al.al_constraint_grad_coefficients(h.detach(), g.detach(), S, RHO)
            gC = ch @ A + cg @ C
            K = torch.eye(N, **kw)
            fin = torch.ones(N, dtype=torch.bool, device=device)
            X, clip, n_clip = fused.update_step(tg, sc, X, gF, gC, K, torch.zeros_like(X), fin)
            clipped += int(n_clip.sum())
            ## the reference: PGD on L / T, by autograd of al_value, in y = x / s
            yg = (X_ref / s).clone().requires_grad_(True)
            F2, h2, g2 = rows(yg * s)
            (gy,) = torch.autograd.grad(al.al_value(F2, h2, g2, S, RHO).sum(), yg)
            y_new = yg.detach() - (sc.lr / RHO) * gy / sc.T
            X_ref = torch.minimum(torch.maximum(y_new * s, lo), hi)
            rel = _maxabs(X, X_ref) / max(1.0, _maxabs(X_ref))
            assert rel <= tol * 100, f"{dtype}: PGD mismatch (relative) {rel:.2e}"
        assert clipped > 0, "the test must exercise the bound clamp"
    print("PASS kernel none: the step is projected gradient descent on L_rho / T in y = x / s")


def test_step_size_is_lr_over_the_shared_rho():
    """`svgd_lr / rho`, the same for every particle: at rho 4 a particle with gradient G moves
    by exactly -(lr / 4) G (kernel off, no bound), and doubling rho halves every step."""
    dtype, device = torch.float64, torch.device("cpu")
    kw = dict(dtype=dtype, device=device)
    s = torch.ones(3, **kw)
    tg = _toy_target(s, torch.full((3,), -1e9, **kw), torch.full((3,), 1e9, **kw))
    X = torch.zeros(2, 3, **kw)
    gF = torch.tensor([[1.0, -2.0, 0.5], [3.0, 1.0, -1.0]], **kw)
    out = []
    for rho in (4.0, 8.0):
        sc = fused.StepConfig(kernel="none", bandwidth_floor=0.05, inside=False, lr=0.1, T=1.0, rho=rho)
        Xn, _, _ = fused.update_step(tg, sc, X, gF, torch.zeros_like(gF), torch.eye(2, **kw),
                                     torch.zeros_like(X), torch.ones(2, dtype=torch.bool))
        out.append(Xn)
    assert _maxabs(out[0], -0.025 * gF) <= 1e-15 and _maxabs(out[1], 0.5 * out[0]) <= 1e-15, out
    print("PASS the step is svgd_lr / rho with the one shared rho")


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
                                  torch.zeros_like(x), grad, torch.zeros_like(x), 1.0, kernel=False)
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


if __name__ == "__main__":
    test_al_value_and_coefficients_agree_with_autograd()
    test_al_with_zero_multipliers_is_the_quadratic_penalty()
    test_dual_update_is_unconditional_and_per_particle()
    test_al_rounds_converge_to_the_kkt_point_on_every_particle()
    test_pairwise_sqdist_matches_brute_force()
    test_median_bandwidth_matches_numpy_on_the_upper_triangle()
    test_rbf_repulsion_is_the_svgd_gradient()
    test_stein_direction_forms()
    test_plain_svgd_matches_a_gaussians_moments()
    test_kernel_none_is_projected_gradient_descent_on_L()
    test_step_size_is_lr_over_the_shared_rho()
    test_resample_mask()
    test_no_host_synchronising_calls_in_the_modules()
    test_compiled_chain_matches_eager()
    print("ALL PASS")
