"""Guards for the BiLipNet / LOInK core (`src/charts/loink.py`) on untrained models.

There is no test runner in this repo; run this by hand, CPU only, under the local memory scope:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=20G \
        -p MemorySwapMax=0 .venv/bin/python tests/test_loink_core.py

Every model here is small (n <= 7, units <= 32) and UNTRAINED, with biases and lognu drawn at
random so no layer sits at its initialisation. The golden comparison against PLNet's JAX code
is `tests/test_loink_reference.py`; this file covers what the golden cannot (tanh, the Newton
polish, the IFT derivative, float32, the LOInK model). What is protected:

1. Cayley orthogonality in both orientations; `mln_fwd(explicit()) == forward` for relu and
   tanh.
2. The guarantees the parametrisation exists for, for both activations: each layer is
   mu-strongly monotone and nu-Lipschitz, the whole net (prod mu, prod nu) bi-Lipschitz,
   on 2000 random pairs.
3. The tanh prox: alpha = 1 returns tanh(u); every alpha solves (1 - alpha) v + alpha atanh(v)
   = u to 1e-12.
4. Inversion: the DYS residual falls with iterations, and after `n_dys = 200, n_newton = 2`
   is below 1e-10 (float64) for both activations and below 1e-5 in float32.
5. The derivative of the inverse: `jacrev` of `decode` w.r.t. (y, z) equals
   diag(scale) J_G(x*)^{-1} to 1e-8 and central differences to 1e-6, with no cross-batch
   coupling -- the IFT step, not the iterations, carries it.
6. `LoinkModel`: shapes, decode(invert(q)) == q, encode == invert()[1], `loink_loss` finite
   and differentiable, cost "u" has no head and no aux term.
7. `torch.compile(model.forward)` on CPU matches eager (SKIP if compile is unavailable).
"""

import math
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO)

from src.charts.loink import (  # noqa: E402
    BiLipNet, LoinkModel, MonLipLayer, bilipnet_jacobian, cayley, dys_inverse,
    invert_bilipnet, loink_loss, mln_fwd, prox_tanh)

CPU = torch.device("cpu")
F64 = torch.float64


def _gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _randomize(module, seed):
    """Biases N(0, 0.1^2) and lognu = log 2 + N(0, 0.1^2), in place, so that no layer is at
    its initialisation (zero biases, nu = 2)."""
    g = _gen(seed)
    with torch.no_grad():
        for name, prm in module.named_parameters():
            leaf = name.split(".")[-1]
            if leaf in ("b", "by") or (leaf.startswith("b") and leaf[1:].isdigit()):
                prm.copy_(0.1 * torch.randn(prm.shape, generator=g, dtype=F64).to(prm.dtype))
            elif leaf == "lognu":
                prm.copy_(math.log(2.0) + 0.1 * torch.randn(prm.shape, generator=g, dtype=F64).to(prm.dtype))
    return module


def _net(activation, n=6, units=(16, 16, 16), depth=2, seed=0, dtype=F64):
    torch.manual_seed(seed)
    return _randomize(BiLipNet(n, units, depth=depth, mu_total=0.1, activation=activation),
                      seed + 100).to(dtype).eval()


def test_cayley_orthogonal():
    g = _gen(1)
    for m, n in ((7, 4), (4, 7), (5, 5), (32, 16), (6, 32)):
        W = torch.randn(m, n, generator=g, dtype=F64)
        R = cayley(W)
        assert R.shape == (m, n)
        G = R.T @ R if m >= n else R @ R.T
        err = (G - torch.eye(min(m, n), dtype=F64)).abs().max().item()
        assert err < 1e-12, f"cayley ({m}, {n}): orthonormality error {err:.2e}"
    print("PASS cayley: orthonormal columns (tall, square) and rows (wide) to 1e-12")


def test_explicit_matches_forward():
    g = _gen(2)
    for act in ("relu", "tanh"):
        torch.manual_seed(3)
        layer = _randomize(MonLipLayer(6, (16, 16, 16), mu=0.3, activation=act), 4).double()
        x = 2 * torch.randn(256, 6, generator=g, dtype=F64)
        with torch.no_grad():
            err = (mln_fwd(layer.explicit(), x) - layer(x)).abs().max().item()
        assert err < 1e-12, f"{act}: mln_fwd(explicit) v forward {err:.2e}"
        print(f"PASS {act}: mln_fwd(explicit) == forward ({err:.1e})")


def test_bilipschitz():
    g = _gen(5)
    for act in ("relu", "tanh"):
        net = _net(act, seed=6)
        x1 = 3 * torch.randn(2000, 6, generator=g, dtype=F64)
        x2 = x1 + torch.randn(2000, 6, generator=g, dtype=F64) * torch.logspace(-3, 1, 2000, dtype=F64)[:, None]
        with torch.no_grad():
            dx = (x1 - x2).norm(dim=1)
            for k, mon in enumerate(net.mon):
                dF = mon(x1) - mon(x2)
                mu, nu = mon.mu, mon.nu()
                mono = ((dF * (x1 - x2)).sum(1) / dx ** 2).min()
                lip = (dF.norm(dim=1) / dx).max()
                assert mono >= mu * (1 - 1e-10), f"{act} mon_{k}: monotonicity {mono:.4g} < mu {mu:.4g}"
                assert lip <= nu * (1 + 1e-10), f"{act} mon_{k}: Lipschitz {lip:.4g} > nu {nu:.4g}"
            mu, nu, tau = net.bounds()
            ratio = (net(x1) - net(x2)).norm(dim=1) / dx
            assert ratio.min() >= mu * (1 - 1e-10) and ratio.max() <= nu * (1 + 1e-10), (
                f"{act}: G ratio [{ratio.min():.4g}, {ratio.max():.4g}] outside [{mu:.4g}, {nu:.4g}]")
            assert bool(net.check_bounds())
        print(f"PASS {act}: each layer monotone >= mu, Lipschitz <= nu; G ratio in "
              f"[{ratio.min():.3f}, {ratio.max():.3f}] within [mu, nu] = [{mu:.3f}, {nu:.3f}] (tau {tau:.2f})")


def test_prox_tanh():
    u = torch.linspace(-6, 6, 4001, dtype=F64)
    err1 = (prox_tanh(u, torch.tensor(1.0, dtype=F64)) - torch.tanh(u)).abs().max().item()
    assert err1 < 1e-15, f"prox_tanh at alpha = 1 is not tanh: {err1:.2e}"
    worst = 0.0
    for alpha in (0.01, 0.05, 0.19, 0.46, 0.9, 1.0, 1.5, 3.0, 10.0):
        v = prox_tanh(u, torch.tensor(alpha, dtype=F64))
        ## |v| <= 1, not < 1: for large |w*| tanh(w*) rounds to +-1 exactly, as tanh(u) does
        ## for |u| > 19 in the forward pass. Nothing downstream takes atanh of it.
        assert bool((v.abs() <= 1).all())
        ## The v equation, evaluated where atanh(v) is well conditioned (|v| < 1 - 1e-3:
        ## near |v| = 1 atanh amplifies v's last ulp beyond 1e-12 whatever solved for it).
        ok = v.abs() < 1 - 1e-3
        r = ((1 - alpha) * v + alpha * torch.atanh(v) - u)[ok].abs().max().item()
        worst = max(worst, r)
        assert r < 1e-12, f"alpha {alpha}: (1-a) v + a atanh v - u = {r:.2e}"
    ## And the wide range, through the w = atanh(v) form the solver works in.
    uw = torch.linspace(-60, 60, 20001, dtype=F64)
    for alpha in (0.01, 0.19, 1.0, 10.0):
        v = prox_tanh(uw, torch.tensor(alpha, dtype=F64))
        assert bool(torch.isfinite(v).all() and (v.abs() <= 1).all())
    print(f"PASS prox_tanh: alpha=1 is tanh ({err1:.1e}); equation residual {worst:.1e} over "
          "alpha in [0.01, 10]")


def _residual_table(net, w):
    rows = {}
    with torch.no_grad():
        for K in (1, 5, 10, 20, 50, 100, 200):
            rows[f"DYS {K}"] = (net(dys_inverse(net, w, K)) - w).abs().max().item()
    for nn_ in (0, 1, 2):
        x = invert_bilipnet(net, w, 200, nn_).detach()
        rows[f"DYS 200 + IFT, n_newton={nn_}"] = (net(x) - w).abs().max().item()
    return rows


def test_inversion():
    g = _gen(7)
    tables = {}
    for act in ("relu", "tanh"):
        net = _net(act, n=7, units=(32, 32, 32, 32), seed=8)
        w = torch.randn(64, 7, generator=g, dtype=F64)
        rows = _residual_table(net, w)
        tables[act] = rows
        r = [rows[f"DYS {K}"] for K in (1, 10, 50, 200)]
        assert r[0] > r[1] > r[2] > r[3], f"{act}: DYS residual not falling {r}"
        final = rows["DYS 200 + IFT, n_newton=2"]
        assert final < 1e-10, f"{act}: residual after DYS 200 + Newton 2 = {final:.2e}"
        ## round trip on G's own outputs
        x0 = torch.randn(64, 7, generator=g, dtype=F64)
        with torch.no_grad():
            x_rec = invert_bilipnet(net, net(x0), 200, 2)
        rt = (x_rec - x0).abs().max().item()
        assert rt < 1e-9, f"{act}: G^-1(G(x)) - x = {rt:.2e}"
    print("PASS inversion float64 (n=7, units 32x4, depth 2, mu_total 0.1), max |G(x) - w| over 64 targets:")
    print(f"      {'':34s} {'relu':>9s} {'tanh':>9s}")
    for key in tables["relu"]:
        print(f"      {key:34s} {tables['relu'][key]:9.1e} {tables['tanh'][key]:9.1e}")

    for act in ("relu", "tanh"):
        net = _net(act, n=7, units=(32, 32, 32, 32), seed=8, dtype=torch.float32)
        w = torch.randn(64, 7, generator=g, dtype=F64).float()
        x = invert_bilipnet(net, w, 200, 2).detach()
        with torch.no_grad():
            r32 = (net(x) - w).abs().max().item()
        assert x.dtype == torch.float32 and r32 < 1e-5, f"{act} float32 residual {r32:.2e}"
        print(f"PASS {act} float32: residual after DYS 200 + Newton 2 = {r32:.1e}")


def _model(act, cost="m", n=6, p=3, seed=9, n_dys=200, n_newton=2):
    torch.manual_seed(seed)
    scale = torch.linspace(0.5, 3.0, n)
    model = LoinkModel(n, p, scale, units=(16, 16, 16), depth=2, mu=0.1, activation=act,
                       cost=cost, head_hidden=32, n_dys=n_dys, n_newton=n_newton)
    _randomize(model.G, seed + 1)
    return model.double().eval()


def test_ift_jacobian():
    g = _gen(10)
    for act in ("relu", "tanh"):
        model = _model(act)
        B, n, p = 3, model.n, model.p
        y = torch.randn(B, p, generator=g, dtype=F64)
        z = torch.randn(B, n - p, generator=g, dtype=F64)
        jy, jz = torch.func.jacrev(model.decode, argnums=(0, 1))(y, z)
        assert jy.shape == (B, n, B, p) and jz.shape == (B, n, B, n - p)
        with torch.no_grad():
            q = model.decode(y, z)
            J_G = bilipnet_jacobian(model.G, q / model.scale)
        J_ref = model.scale[:, None] * torch.linalg.inv(J_G)
        h = 1e-6
        e_inv = e_fd = e_cross = 0.0
        for b in range(B):
            Jb = torch.cat([jy[b, :, b, :], jz[b, :, b, :]], dim=1)
            e_inv = max(e_inv, (Jb - J_ref[b]).abs().max().item())
            for c in range(B):
                if c != b:
                    e_cross = max(e_cross, jy[b, :, c, :].abs().max().item(),
                                  jz[b, :, c, :].abs().max().item())
        w = torch.cat([y, z], 1)
        with torch.no_grad():
            for i in range(n):
                dw = torch.zeros_like(w)
                dw[:, i] = h
                qp = model.decode((w + dw)[:, :p], (w + dw)[:, p:])
                qm = model.decode((w - dw)[:, :p], (w - dw)[:, p:])
                col = (qp - qm) / (2 * h)
                for b in range(B):
                    Jb_i = jy[b, :, b, i] if i < p else jz[b, :, b, i - p]
                    e_fd = max(e_fd, (col[b] - Jb_i).abs().max().item())
        assert e_inv < 1e-8, f"{act}: jacrev(decode) v diag(scale) J_G^-1: {e_inv:.2e}"
        assert e_fd < 1e-6, f"{act}: jacrev(decode) v central differences: {e_fd:.2e}"
        assert e_cross == 0.0, f"{act}: cross-batch Jacobian blocks nonzero ({e_cross:.2e})"
        print(f"PASS {act} IFT: jacrev(decode) v diag(scale) J_G(x*)^-1 {e_inv:.1e}, v central "
              f"differences {e_fd:.1e}, cross-batch blocks exactly 0")


def test_loink_model():
    g = _gen(11)
    for act in ("relu", "tanh"):
        for cost in ("m", "x", "u"):
            model = _model(act, cost=cost)
            B, n, p, K = 32, model.n, model.p, model.K
            q = torch.randn(B, n, generator=g, dtype=F64) * model.scale
            y_hat, z = model.invert(q)
            assert y_hat.shape == (B, p) and z.shape == (B, K)
            assert torch.equal(model.encode(q), z)
            q_rec = model.decode(y_hat, z)
            assert q_rec.shape == (B, n) and q_rec.dtype == F64
            rt = (q_rec - q).abs().max().item()
            assert rt < 1e-9, f"{act}/{cost}: decode(invert(q)) - q = {rt:.2e}"
            assert torch.equal(model(z, y_hat), q_rec)
            y = torch.randn(B, p, generator=g, dtype=F64)
            j = torch.rand(B, generator=g, dtype=F64)
            model.zero_grad()
            loss, parts = loink_loss(model, q, y, j if cost != "u" else None)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(v).all() for v in parts.values())
            grads = [prm.grad for prm in model.parameters() if prm.requires_grad]
            assert all(gr is not None and torch.isfinite(gr).all() for gr in grads), "missing/NaN grads"
            if cost == "u":
                assert model.H is None and "aux_mse" not in parts
                assert torch.allclose(loss, parts["task_mse"] + parts["dist"], rtol=0, atol=1e-14)
            else:
                assert model.H is not None and "aux_mse" in parts
                assert model.cost_hat(z, y_hat).shape == (B,)
            print(f"PASS LoinkModel {act}/{cost}: shapes, encode == invert()[1], round trip {rt:.1e}, "
                  f"loss {loss.item():.4g} finite with finite grads (tau {parts['tau'].item():.2f})")
    ## The cost model's quadratic form: at z = 0 it is b(y_hat); with P = 0 it is |z|^2 / 2 + b.
    model = _model("tanh")
    with torch.no_grad():
        model.H[-1].weight.zero_()
        model.H[-1].bias.zero_()
        model.H[-1].bias[0] = 0.7
        z = torch.randn(5, model.K, generator=g, dtype=F64)
        yh = torch.randn(5, model.p, generator=g, dtype=F64)
        c = model.cost_hat(z, yh)
    assert torch.allclose(c, 0.5 * (z * z).sum(1) + 0.7, rtol=0, atol=1e-14)
    print("PASS cost_hat: P = 0 gives |z|^2/2 + b")


def test_compile():
    """Kept last: compiles `forward` (DYS 10 + Newton 1 + the IFT step) on the CPU."""
    model = _model("tanh", n_dys=10, n_newton=1)
    g = _gen(12)
    y = torch.randn(8, model.p, generator=g, dtype=F64)
    z = torch.randn(8, model.K, generator=g, dtype=F64)
    try:
        t = time.monotonic()
        compiled = torch.compile(model.forward)
        q_c = compiled(z, y)
        dt = time.monotonic() - t
    except Exception as e:  # noqa: BLE001 -- any backend failure is an environment SKIP
        print(f"SKIP torch.compile on CPU: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return
    q_e = model.forward(z, y)
    err = (q_c - q_e).abs().max().item()
    assert err < 1e-10, f"compiled v eager decode: {err:.2e}"
    print(f"PASS torch.compile(model.forward) on CPU matches eager ({err:.1e}; compile+run {dt:.0f} s)")


if __name__ == "__main__":
    t0 = time.monotonic()
    test_cayley_orthogonal()
    test_explicit_matches_forward()
    test_bilipschitz()
    test_prox_tanh()
    test_inversion()
    test_ift_jacobian()
    test_loink_model()
    test_compile()
    print(f"ALL PASS ({time.monotonic() - t0:.1f} s)")
