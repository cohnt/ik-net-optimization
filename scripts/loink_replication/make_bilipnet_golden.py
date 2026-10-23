"""Golden reference for the PLNet (BiLipNet) JAX/Flax code, float64, for checking a torch port.

Source: github.com/acfr/PLNet (MIT), surrogte_loss/rb20d_solver.py + layer.py (read-only).
rb20d_solver.py cannot be imported (module-level code loads orbax checkpoints and runs solves), so
this script exec()s only its definitions (the line range from `class MonLipNet` through `mln_RA`)
after `from layer import *`, unmodified.

Throwaway venv (NOT the project .venv), CPU only:
    python3 -m venv <scratch>/venv-jax
    <scratch>/venv-jax/bin/pip install "jax[cpu]" flax numpy
    pip freeze (relevant): jax==0.12.0.1 jaxlib==0.12.0.1 flax==0.12.10 numpy==2.5.4
                           ml_dtypes==0.6.0 scipy==1.18.1   (Python 3.12.3)
Run:
    JAX_PLATFORMS=cpu <scratch>/venv-jax/bin/python scripts/loink_replication/make_bilipnet_golden.py \
        --plnet-dir <PLNet clone>/surrogte_loss
Writes tests/data/loink_reference/bilipnet_golden.npz (keys documented in the README next to it).

Random draw order (numpy RandomState-free: np.random.default_rng(0), one generator, in this order):
  1. Unitary biases b: k = 0,1,2, shape (5,), N(0, 0.1^2)
  2. for mon k = 0,1: b0,b1,b2,b3 (16,) each, then by (5,), then lognu = log 2 + N(0,0.1^2) (1,)
  3. x (64,5) ~ N(0,1)
  4. w (64,5) ~ N(0,1)
Flax init uses jax.random.PRNGKey(0) (W, Fq, Fab* glorot_normal; the rest constants), float32 then cast.
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

ap = argparse.ArgumentParser()
ap.add_argument("--plnet-dir", required=True)
ap.add_argument("--out", default=None)
args = ap.parse_args()
sys.path.insert(0, args.plnet_dir)

from layer import *  # noqa: F401,F403  (cayley, Unitary, nn, ...)
from typing import Sequence, Callable  # noqa: F401

# exec the definitions of rb20d_solver.py: MonLipNet .. mln_RA (before "# solver ---")
src = open(os.path.join(args.plnet_dir, "rb20d_solver.py")).read().split("\n")
start = next(i for i, l in enumerate(src) if l.startswith("class MonLipNet"))
stop = next(i for i, l in enumerate(src) if l.startswith("# solver ---"))
ns = dict(globals())
exec("\n".join(src[start:stop]), ns)
MonLipNet, BiLipNet = ns["MonLipNet"], ns["BiLipNet"]
orth_bwd, mln_fwd, mln_fwd_x2b = ns["orth_bwd"], ns["mln_fwd"], ns["mln_fwd_x2b"]
mln_fwd_b2z, mln_fwd_xz2y = ns["mln_fwd_b2z"], ns["mln_fwd_xz2y"]
mln_bwd_y2b, mln_bwd_yz2x, mln_RA = ns["mln_bwd_y2b"], ns["mln_bwd_yz2x"], ns["mln_RA"]
relu = nn.relu

N, DEPTH, UNITS, TAU, B = 5, 2, (16, 16, 16, 16), 10.0, 64
LAYER_TAU = TAU ** (1 / DEPTH)
ALPHA_G, LAMBDA = 1.0, 1.0  # the module globals `alpha`, `Lambda` of rb20d_solver.py

model = BiLipNet(list(UNITS), tau=TAU, depth=DEPTH)
mon_mod = MonLipNet(list(UNITS), tau=LAYER_TAU)
uni_mod = Unitary()

params = model.init(jax.random.PRNGKey(0), jnp.zeros((B, N), jnp.float64))
params = jax.tree_util.tree_map(lambda a: np.array(a, dtype=np.float64), params)
p = params["params"]
rng = np.random.default_rng(0)
for k in range(DEPTH + 1):
    p[f"uni_{k}"]["b"] = 0.1 * rng.standard_normal(N)
for k in range(DEPTH):
    for j in range(len(UNITS)):
        p[f"mon_{k}"][f"b{j}"] = 0.1 * rng.standard_normal(UNITS[j])
    p[f"mon_{k}"]["by"] = 0.1 * rng.standard_normal(N)
    p[f"mon_{k}"]["lognu"] = np.log(2.0) + 0.1 * rng.standard_normal(1)
x_np = rng.standard_normal((B, N))
w_np = rng.standard_normal((B, N))
params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), params)
p = params["params"]
x, w = jnp.asarray(x_np), jnp.asarray(w_np)

out = {"n": np.array(N), "depth": np.array(DEPTH), "units": np.array(UNITS), "tau": np.array(TAU),
       "layer_tau": np.array(LAYER_TAU)}
paths = {}
for path, v in traverse_util.flatten_dict(p).items():
    key = "/".join(path)
    out[key] = np.asarray(v)
    paths[key] = "params/" + "/".join(path)
for k, v in paths.items():
    print("param path", k, "<-", v, out[k].shape, out[k].dtype)

uni_params, mon_params = [], []
for k in range(DEPTH):
    uni_params.append(uni_mod.apply({"params": p[f"uni_{k}"]}, method=uni_mod.get_params))
    mon_params.append(mon_mod.apply({"params": p[f"mon_{k}"]}, method=mon_mod.get_params))
uni_params.append(uni_mod.apply({"params": p[f"uni_{DEPTH}"]}, method=uni_mod.get_params))
for k in range(DEPTH + 1):
    out[f"uni_{k}/R"] = np.asarray(uni_params[k]["R"])
for k in range(DEPTH):
    mp = mon_params[k]
    mu, gam = mp["mu"], mp["gam"]
    out[f"mon_{k}/mu"], out[f"mon_{k}/gam"], out[f"mon_{k}/nu"] = (np.asarray(mu), np.asarray(gam),
                                                                  np.asarray(mu + gam))
    out[f"mon_{k}/S"], out[f"mon_{k}/bh"] = np.asarray(mp["S"]), np.asarray(mp["bh"])
    for j, V in enumerate(mp["V"]):
        out[f"mon_{k}/V{j + 1}"] = np.asarray(V)
lipmin, lipmax, tau_tot = model.apply(params, method=model.get_bounds)
out["bounds"] = np.array([lipmin, lipmax, tau_tot], dtype=np.float64)

# forward outputs
for k in range(DEPTH + 1):
    out[f"uni_{k}_out"] = np.asarray(uni_mod.apply({"params": p[f"uni_{k}"]}, x))
for k in range(DEPTH):
    out[f"mon_{k}_out"] = np.asarray(mon_mod.apply({"params": p[f"mon_{k}"]}, x))
out["G_out"] = np.asarray(model.apply(params, x))
hb = mln_fwd_x2b(mon_params[0], x)
hz = mln_fwd_b2z(mon_params[0], hb)
hy = mln_fwd_xz2y(mon_params[0], x, hz)
out["mon_0_helper_b"], out["mon_0_helper_z"], out["mon_0_helper_y"] = map(np.asarray, (hb, hz, hy))
assert np.abs(out["mon_0_helper_y"] - out["mon_0_out"]).max() < 1e-10
for k in range(DEPTH):
    assert np.abs(np.asarray(mln_fwd(mon_params[k], x)) - out[f"mon_{k}_out"]).max() < 1e-10
chk = x
for k in range(DEPTH):
    chk = ns["orth_fwd"](uni_params[k], chk)
    chk = mln_fwd(mon_params[k], chk)
chk = ns["orth_fwd"](uni_params[DEPTH], chk)
assert np.abs(np.asarray(chk) - out["G_out"]).max() < 1e-10
out["x"], out["w"] = x_np, w_np


def dys(mp, alpha, bz, k_iters, record=0):
    """Per-layer Davis-Yin loop exactly as DavisYinSplit/body_fn in mln_back_solve_dys."""
    u = jnp.zeros_like(bz)
    zk = u
    rec = []
    for _ in range(k_iters):
        zh = relu(u)
        uh = 2 * zh - u
        zk = mln_RA(mp, alpha, bz, zh, uh)
        u = u + LAMBDA * (zk - zh)
        rec.append((zh, zk, u))
    return zk, rec


def alpha_of(mp):
    return ALPHA_G * mp["mu"] / mp["gam"]


def invert_bilipnet(uni_params, mon_params, w, k_iters):
    y = orth_bwd(uni_params[DEPTH], w)
    for k in reversed(range(DEPTH)):
        mp = mon_params[k]
        bz = mln_bwd_y2b(mp, y)
        zk, _ = dys(mp, alpha_of(mp), bz, k_iters)
        x_ = mln_bwd_yz2x(mp, y, zk)
        y = orth_bwd(uni_params[k], x_)
    return y


for K in (1, 5, 20, 100, 500):
    out[f"x_inv_k{K}"] = np.asarray(invert_bilipnet(uni_params, mon_params, w, K))
xi = invert_bilipnet(uni_params, mon_params, jnp.asarray(out["G_out"]), 500)
out["x_inv_of_Gx_k500"] = np.asarray(xi)
out["recover_err"] = np.array(np.abs(out["x_inv_of_Gx_k500"] - x_np).max())

mp0 = mon_params[0]
bz0 = mln_bwd_y2b(mp0, w)
zk, rec = dys(mp0, alpha_of(mp0), bz0, 500)
out["mon_0_inv_k500"] = np.asarray(mln_bwd_yz2x(mp0, w, zk))
out["mon_0_bz"], out["mon_0_alpha"] = np.asarray(bz0), np.asarray(alpha_of(mp0))
for s in (1, 2, 3):
    zh, zk_s, u_s = rec[s - 1]
    out[f"mon_0_dys_zh_k{s}"], out[f"mon_0_dys_zk_k{s}"], out[f"mon_0_dys_u_k{s}"] = map(
        np.asarray, (zh, zk_s, u_s))
out["mon_0_inv_residual_k500"] = np.array(
    np.abs(np.asarray(mln_fwd(mp0, jnp.asarray(out["mon_0_inv_k500"]))) - w_np).max())
G_res = np.abs(np.asarray(model.apply(params, jnp.asarray(out["x_inv_k500"]))) - w_np).max()
out["G_inv_residual_k500"] = np.array(G_res)
for K in (1, 5, 20, 100, 500):
    print(f"G residual at k={K}:", np.abs(np.asarray(model.apply(params, jnp.asarray(out[f'x_inv_k{K}']))) - w_np).max())
print("bounds (lipmin, lipmax, tau):", out["bounds"])
print("recover_err (inverse of G(x), k=500):", float(out["recover_err"]))
print("mon_0_inv_residual_k500:", float(out["mon_0_inv_residual_k500"]))
print("G_inv_residual_k500:", float(out["G_inv_residual_k500"]))

dest = args.out or os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "loink_reference",
                                "bilipnet_golden.npz")
dest = os.path.abspath(dest)
os.makedirs(os.path.dirname(dest), exist_ok=True)
assert all(np.asarray(v).dtype in (np.float64, np.int64) for v in out.values())
np.savez(dest, **out)
print("wrote", dest)
