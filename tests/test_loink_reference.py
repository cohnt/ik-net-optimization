"""The torch BiLipNet port (`src/charts/loink.py`) against the official PLNet JAX/Flax code.

There is no test runner in this repo; run this by hand, CPU only, under the local memory scope:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=20G \
        -p MemorySwapMax=0 .venv/bin/python tests/test_loink_reference.py

The golden file `tests/data/loink_reference/bilipnet_golden.npz` is float64 output of PLNet's
own `rb20d_solver.py` / `layer.py` definitions, unmodified (generator
`scripts/loink_replication/make_bilipnet_golden.py`, contract in the README beside the file):
a BiLipNet n = 5, depth 2, units (16, 16, 16, 16), tau = 10, so each monotone layer has
mu = nu / sqrt(10) with nu = exp(lognu). The torch twin is built with the file's raw
parameters (`load_plnet_params`, no transposes: the port keeps PLNet's orientation) and each
layer's mu buffer SET to exp(lognu)/sqrt(10). Everything is checked to 1e-10 in float64:

1. every explicit quantity (R, mu, nu, gam, S, bh, V_j, the bounds);
2. every layer's output on x, the full G, and the three forward helpers;
3. the plain DYS inverse `dys_inverse(net, w, K)` at K = 1, 5, 20, 100, 500 (the raw DYS
   result, BEFORE the IFT linearisation `invert_bilipnet` adds), the inverse of G(x) at
   K = 500, the single-layer inverse of mon_0, its b_z and alpha, and the first three DYS
   iterates (z^{k+1/2}, z^{k+1}, u^{k+1}) on mon_0.

If the golden file is absent (the agent generating it may not have finished), the test
prints SKIP and exits 0.
"""

import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO)

from src.charts.loink import (  # noqa: E402
    BiLipNet, dys_inverse, dys_layer, load_plnet_params, mln_fwd_b2z, mln_fwd_x2b,
    mln_fwd_xz2y, mln_bwd_y2b)

GOLDEN = os.path.join(REPO, "tests", "data", "loink_reference", "bilipnet_golden.npz")
TOL = 1e-10
CPU = torch.device("cpu")


def _t(golden, key):
    return torch.as_tensor(golden[key], dtype=torch.float64, device=CPU)


def _close(name, got, golden, key, tol=TOL):
    ref = _t(golden, key)
    got = got.detach().reshape(ref.shape) if got.numel() == ref.numel() else got.detach()
    assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
    err = (got - ref).abs().max().item() if ref.numel() else 0.0
    assert err < tol, f"{name} ({key}): max |torch - jax| = {err:.3e} >= {tol:g}"
    return err


def build_twin(golden):
    n, depth = int(golden["n"]), int(golden["depth"])
    units = tuple(int(u) for u in golden["units"])
    assert (n, depth, units) == (5, 2, (16, 16, 16, 16)), (n, depth, units)
    assert abs(float(golden["layer_tau"]) - math.sqrt(10.0)) < 1e-12
    torch.manual_seed(0)
    net = BiLipNet(n, units, depth=depth, mu_total=0.1, activation="relu").double()
    return load_plnet_params(net, golden).eval()


def test_explicit(net, golden):
    worst = 0.0
    for k, uni in enumerate(net.uni):
        R, _ = uni.explicit()
        worst = max(worst, _close(f"uni_{k} R", R, golden, f"uni_{k}/R"))
    for k, mon in enumerate(net.mon):
        p = mon.explicit()
        for name in ("mu", "nu", "gam", "S", "bh"):
            worst = max(worst, _close(f"mon_{k} {name}", p[name], golden, f"mon_{k}/{name}"))
        assert len(p["V"]) == 3
        for j in (1, 2, 3):
            worst = max(worst, _close(f"mon_{k} V{j}", p["V"][j - 1], golden, f"mon_{k}/V{j}"))
    worst = max(worst, _close("bounds", torch.stack(net.bounds()), golden, "bounds"))
    print(f"PASS explicit quantities (R, mu, nu, gam, S, bh, V_j, bounds): worst {worst:.2e}")


def test_forward(net, golden):
    x = _t(golden, "x")
    worst = 0.0
    with torch.no_grad():
        for k, uni in enumerate(net.uni):
            worst = max(worst, _close(f"uni_{k}(x)", uni(x), golden, f"uni_{k}_out"))
        for k, mon in enumerate(net.mon):
            worst = max(worst, _close(f"mon_{k}(x)", mon(x), golden, f"mon_{k}_out"))
        worst = max(worst, _close("G(x)", net(x), golden, "G_out"))
        p = net.mon[0].explicit()
        b = mln_fwd_x2b(p, x)
        z = mln_fwd_b2z(p, b)
        y = mln_fwd_xz2y(p, x, z)
        worst = max(worst, _close("helper b", b, golden, "mon_0_helper_b"))
        worst = max(worst, _close("helper z", z, golden, "mon_0_helper_z"))
        worst = max(worst, _close("helper y", y, golden, "mon_0_helper_y"))
    print(f"PASS layer outputs, G and the mon_0 forward helpers: worst {worst:.2e}")


def test_inverse(net, golden):
    w = _t(golden, "w")
    worst = {}
    with torch.no_grad():
        for K in (1, 5, 20, 100, 500):
            worst[K] = _close(f"DYS K={K}", dys_inverse(net, w, K), golden, f"x_inv_k{K}")
        e_gx = _close("DYS of G(x) K=500", dys_inverse(net, _t(golden, "G_out"), 500), golden,
                      "x_inv_of_Gx_k500")
        p = net.mon[0].explicit()
        e_bz = _close("mon_0 bz", mln_bwd_y2b(p, w), golden, "mon_0_bz")
        e_al = _close("mon_0 alpha", p["mu"] / p["gam"], golden, "mon_0_alpha")
        x0, hist = dys_layer(p, w, 500, record=True)
        e_m0 = _close("mon_0 inverse K=500", x0, golden, "mon_0_inv_k500")
        e_it = 0.0
        for s in (1, 2, 3):
            zh, zk, u = hist[s - 1]
            e_it = max(e_it, _close(f"zh step {s}", zh, golden, f"mon_0_dys_zh_k{s}"))
            e_it = max(e_it, _close(f"zk step {s}", zk, golden, f"mon_0_dys_zk_k{s}"))
            e_it = max(e_it, _close(f"u step {s}", u, golden, f"mon_0_dys_u_k{s}"))
        res = (net(dys_inverse(net, w, 500)) - w).abs().max().item()
    print("PASS DYS inverse v golden, max |torch - jax| per K: "
          + ", ".join(f"K={K} {e:.1e}" for K, e in worst.items()))
    print(f"PASS DYS of G(x) {e_gx:.1e}; mon_0 bz {e_bz:.1e}, alpha {e_al:.1e}, single-layer "
          f"inverse {e_m0:.1e}, iterates 1-3 {e_it:.1e}; torch G residual at K=500 {res:.1e}")


if __name__ == "__main__":
    t0 = time.monotonic()
    if not os.path.exists(GOLDEN):
        print(f"SKIP: golden file {GOLDEN} absent (scripts/loink_replication/make_bilipnet_golden.py)")
        sys.exit(0)
    golden = dict(np.load(GOLDEN))
    net = build_twin(golden)
    test_explicit(net, golden)
    test_forward(net, golden)
    test_inverse(net, golden)
    print(f"ALL PASS ({time.monotonic() - t0:.1f} s)")
