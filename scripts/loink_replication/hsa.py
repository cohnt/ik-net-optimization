"""LOInK paper (arXiv 2609.21275) Sec. VII replication: the planar HSA soft manipulator.

    hsa.py data   # results/loink_replication/hsa_n2000000_seed0.npz -> hsa/train.npz
    (train with train_replication.py --data results/loink_replication/hsa/train.npz ...)
    hsa.py eval --model-dir DIR

Dataset (`hsa_dataset.py`): u ~ Uniform([0, phi_max]^6) motor angles (rad), SoRoMoX
equilibrium, y = [p_x, p_y, theta] (m, m, rad). Training arrays (converged rows only):
  * x = u - phi_max/2 (`x_offset` = phi_max/2 stored; train with `--scale phi_max/2`, so the
    network sees u normalised to [-1, 1]; decode returns x, eval adds the offset back);
  * y = the task STANDARDISED per dimension (`y_mean`, `y_sd` stored, raw in `y_raw`): the
    paper does not say how it scales y, and raw metres (~0.1 m spreads) next to radians and
    a cost in rad^2 make eq. (16)'s two MSEs incommensurate;
  * j = |u - u_bar|^2 (eq. 24), RAW rad^2 (the trainer may standardise it; the saved model
    carries j_mean / j_std). `--u-bar zero` (default, the paper's u_bar = 0, `train.npz`)
    or `--u-bar mid` (u_bar = phi_max / 2, the centre of U, `train_ubar_mid.npz`): with
    U = [0, phi_max]^6, u_bar = 0 is a CORNER of U, so the cost's minimiser on a task fibre
    lies on the boundary of the data, where LOInK's z = 0 cannot sit (see the note).
    `u_bar` is stored and every cost and ratio below uses |u - u_bar|^2.

Evaluation (Table II, Fig. 8-10): 2000 held-out targets y_i (from validation samples x_i,
the trainer's `val_idx.npy`), u* = decode(y_i, 0), forward map = the same SoRoMoX solver
(`hsa_dataset.Solver`). The paper's Fig. 9 denominator is an optimisation-based reference
we do not have; we report |u*|^2 / |x_i|^2 against the generating dataset sample, and a
second, stronger reference: the cheapest training sample within 2 mm (position) and
0.02 rad of y_i (`ref_nn`).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from train_replication import CostHatData, LoadModel  # noqa: E402

SRC = os.path.join(REPO, "results", "loink_replication", "hsa_n2000000_seed0.npz")
OUT = os.environ.get("HSA_OUT", os.path.join(REPO, "results", "loink_replication", "hsa"))
CS = (0.0, 0.1, 0.2, 0.3, 0.4)


def MakeData(a):
    d = np.load(a.src)
    ok = d["converged"].astype(bool)
    u, y = d["u"][ok], d["y"][ok]
    os.makedirs(OUT, exist_ok=True)
    meta = json.loads(str(d["metadata"]))
    phi = np.asarray(meta["phi_max"], dtype=np.float64)
    u_lo, u_hi = float(meta.get("u_lo_frac", 0.0)) * phi, phi
    off = 0.5 * (u_lo + u_hi)              # x = u - off; train with --scale (u_hi - u_lo) / 2
    u_bar = np.zeros(6) if a.u_bar == "zero" else off.copy()
    j = ((u - u_bar) ** 2).sum(1)
    name = a.name or ("train.npz" if a.u_bar == "zero" else f"train_ubar_{a.u_bar}.npz")
    y_mean, y_sd = y.mean(0), y.std(0)
    np.savez(os.path.join(OUT, name), x=(u - off).astype(np.float32), x_offset=off, u_bar=u_bar, u_lo=u_lo, u_hi=u_hi,
             y=((y - y_mean) / y_sd).astype(np.float32), y_mean=y_mean, y_sd=y_sd, y_raw=y, j=j,
             src_index=np.flatnonzero(ok), metadata=json.dumps(dict(src=a.src, **meta)))
    info = dict(n_total=int(len(ok)), n_converged=int(ok.sum()), phi_max=meta.get("phi_max"),
                y_min=y.min(0).tolist(), y_max=y.max(0).tolist(), y_mean=y_mean.tolist(),
                y_sd=y_sd.tolist(), x_offset=off.tolist(), u_lo=u_lo.tolist(),
                scale_to_train_with=((u_hi - u_lo) / 2).tolist(), j_mean=float(j.mean()),
                j_std=float(j.std()), j_min=float(j.min()), j_max=float(j.max()))
    json.dump(info, open(os.path.join(OUT, name.replace(".npz", "_info.json")), "w"), indent=1)
    print(json.dumps(info, indent=1))


def _hist_log(ax, v, bins, label, color):
    ax.hist(v, bins=bins, label=label, color=color, alpha=0.8)


def Eval(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from hsa_dataset import Solver

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.makedirs(OUT, exist_ok=True)
    model, ckpt = LoadModel(os.path.join(a.model_dir, "model.pt"), dtype=torch.float64)
    fin = json.load(open(os.path.join(a.model_dir, "final.json")))
    d = np.load(ckpt["config"]["data"])
    off, y_mean, y_sd = d["x_offset"], d["y_mean"], d["y_sd"]
    u_bar = d["u_bar"] if "u_bar" in d.files else np.zeros(6)
    u_lo = d["u_lo"] if "u_lo" in d.files else np.zeros(6)

    def C(u):
        return ((u - u_bar) ** 2).sum(-1)
    X, Y, J = d["x"].astype(np.float64) + off, d["y_raw"].astype(np.float64), d["j"]

    def Dec(m, y_raw, z, dtype=torch.float64):
        """u = decode((y_raw - y_mean) / y_sd, z) + x_offset, as a numpy array."""
        ys = torch.as_tensor((y_raw - y_mean) / y_sd, dtype=dtype)
        return m.decode(ys, torch.as_tensor(z, dtype=dtype)).detach().numpy() + off
    val = np.load(os.path.join(a.model_dir, "val_idx.npy"))
    rng = np.random.default_rng(a.seed)
    sel = rng.choice(val, a.n_targets, replace=False)
    yt, xt = Y[sel], X[sel]
    solver = Solver()
    phi_max = solver.phi_max

    ## u* = decode(y, 0), float64 for accuracy
    us = Dec(model, yt, np.zeros((len(yt), model.K)))
    _, y_star, ok = solver.Solve(us)
    ee_mm = 1e3 * np.linalg.norm(y_star[:, :2] - yt[:, :2], axis=1)
    ## the fit alone: decode at the generating sample's OWN latent z_i = G_2(x_i) (in-domain
    ## by construction), so the error is the task fit's, not z = 0's placement
    with torch.no_grad():
        z_own = model.invert(torch.as_tensor(xt - off))[1].numpy()
    u_own = Dec(model, yt, z_own)
    _, y_own, _ = solver.Solve(u_own)
    ee_own_mm = 1e3 * np.linalg.norm(y_own[:, :2] - yt[:, :2], axis=1)
    th_err = np.abs(y_star[:, 2] - yt[:, 2])
    J_star = C(us)
    ratio_sample = J_star / C(xt)
    in_U = ((us >= u_lo) & (us <= phi_max)).all(1)

    ## stronger reference: the cheapest training sample within 2 mm and 0.02 rad of y_i
    tr = np.setdiff1d(np.arange(len(X)), val)
    Ys = np.c_[Y[tr, :2], 0.1 * Y[tr, 2]]          # 0.02 rad -> 2 mm
    tree = cKDTree(Ys)
    nbrs = tree.query_ball_point(np.c_[yt[:, :2], 0.1 * yt[:, 2]], r=2e-3)
    ref_nn = np.array([J[tr[n]].min() if len(n) else np.nan for n in nbrs])
    n_nbrs = np.array([len(n) for n in nbrs])
    ratio_nn = J_star / ref_nn

    def mmm(v):
        v = v[np.isfinite(v)]
        return dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()),
                    max=float(v.max()))

    res = dict(model_dir=os.path.relpath(a.model_dir, REPO), training=fin["final"],
               config=fin["config"], n_targets=int(a.n_targets),
               solver_converged=float(ok.mean()),
               ee_error_mm=mmm(ee_mm), theta_error_rad=mmm(th_err),
               ee_error_mm_at_own_latent=mmm(ee_own_mm),
               u_own_latent_vs_x_max_abs=float(np.abs(u_own - xt).max()),
               u_star_in_U_frac=float(in_U.mean()),
               ee_error_mm_u_star_in_U=mmm(ee_mm[in_U]) if in_U.sum() > 1 else None,
               ee_error_mm_u_star_outside_U=mmm(ee_mm[~in_U]) if (~in_U).sum() > 1 else None,
               u_star_distance_outside_U_rad=mmm(np.linalg.norm(np.maximum(us - phi_max, 0)
                                                                + np.maximum(u_lo - us, 0), axis=1)),
               cost_ratio_vs_generating_sample_u_star_in_U=(mmm(ratio_sample[in_U])
                                                             if in_U.sum() > 1 else None),
               b_hat_data_units=mmm(CostHatData(model, ckpt, torch.zeros(len(yt), model.K, dtype=torch.float64),
                                                torch.as_tensor((yt - y_mean) / y_sd)).detach().numpy()),
               u_star_max_violation_rad=float(np.maximum(us - phi_max, u_lo - us).max().clip(0)),
               U=dict(lo=u_lo.tolist(), hi=phi_max.tolist()),
               cost_u_star_rad2=mmm(J_star),
               cost_u_star_normalised=mmm((((us - u_bar) / phi_max) ** 2).sum(1)),
               cost_dataset_sample_rad2=mmm(C(xt)),
               u_bar=u_bar.tolist(),
               cost_ratio_vs_generating_sample=mmm(ratio_sample),
               frac_cheaper_than_generating_sample=float((ratio_sample < 1).mean()),
               cost_ratio_vs_nn_reference=mmm(ratio_nn),
               nn_reference=dict(radius="2 mm position, 0.02 rad", targets_with_ref=int(np.isfinite(ref_nn).sum()),
                                 median_neighbours=float(np.median(n_nbrs))))

    ## inference time: float32, batch of n_targets, CPU (CUDA hidden in every run)
    m32, _ = LoadModel(os.path.join(a.model_dir, "model.pt"), dtype=torch.float32)
    y32 = torch.as_tensor((yt - y_mean) / y_sd, dtype=torch.float32)
    z32 = torch.zeros(len(yt), m32.K)
    m32.decode(y32[:10], z32[:10])
    times = []
    for _ in range(5):
        t = time.perf_counter()
        u32 = m32.decode(y32, z32).detach()
        times.append(time.perf_counter() - t)
    t1 = []
    for i in range(20):
        t = time.perf_counter()
        m32.decode(y32[i:i + 1], z32[i:i + 1])
        t1.append(time.perf_counter() - t)
    res["inference"] = dict(device="cpu", threads=torch.get_num_threads(), dtype="float32",
                            n_dys=m32.n_dys, n_newton=m32.n_newton,
                            batch=len(yt), batch_wall_s_median=float(np.median(times)),
                            per_target_ms_amortised=float(1e3 * np.median(times) / len(yt)),
                            single_target_ms_median=float(1e3 * np.median(t1)),
                            float32_vs_float64_max_abs_u=float(np.abs(u32.numpy() + off - us).max()))

    ## Fig. 10: z ~ N(0, c I), 100 draws per target over 50 targets
    fig10 = {}
    sub = np.arange(50)
    ratios = {}
    for c in CS:
        yy = np.repeat(yt[sub], 100, axis=0)
        z = np.sqrt(c) * rng.standard_normal((len(yy), model.K))
        u = Dec(model, yy, z)
        _, yc, okc = solver.Solve(u)
        r = C(u) / np.repeat(J_star[sub], 100)
        ratios[c] = r
        fig10[f"c={c}"] = dict(ratio=mmm(r), frac_below_1=float((r < 1 - 1e-9).mean()),
                               ee_error_mm=mmm(1e3 * np.linalg.norm(yc[:, :2] - yy[:, :2], axis=1)),
                               in_U_frac=float(((u >= u_lo) & (u <= phi_max)).all(1).mean()))
    res["fig10"] = fig10

    ## figures
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(np.log10(max(ee_mm.min(), 1e-4)), np.log10(ee_mm.max()), 100)
    ax.hist(ee_mm, bins=bins, color="tab:purple")
    ax.set_xscale("log")
    ax.set_xlabel("EE error [mm]")
    ax.set_ylabel("Count")
    ax.set_title(f"Fig. 8 replica: LOInK, {a.n_targets} held-out targets")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_ee_error.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (r, lab) in enumerate(((ratio_sample, "generating dataset sample |x_i|^2"),
                                  (ratio_nn, "cheapest dataset sample within 2 mm / 0.02 rad"))):
        r = r[np.isfinite(r)]
        ax[k].hist(r, bins=np.logspace(np.log10(r.min()), np.log10(r.max()), 80), color="tab:purple")
        ax[k].axvline(1, color="k", ls="--")
        ax[k].set_xscale("log")
        ax[k].set_xlabel("|u*|^2 / |u_ref|^2")
        ax[k].set_title(f"reference: {lab}", fontsize=9)
    ax[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_cost_ratio.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(len(CS), 1, figsize=(6, 8), sharex=True)
    for k, c in enumerate(CS):
        ax[k].hist(np.clip(ratios[c], 0.8, 1.6), bins=np.linspace(0.8, 1.6, 41),
                   color=plt.cm.plasma(0.15 + 0.15 * k))
        ax[k].text(0.98, 0.7, f"c = {c}", transform=ax[k].transAxes, ha="right")
    ax[-1].set_xlabel("Cost ratio |u|^2 / |u*|^2 (clipped to [0.8, 1.6])")
    ax[len(CS) // 2].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_sampling.png"), dpi=130)
    plt.close(fig)

    tag = a.tag or os.path.basename(os.path.normpath(a.model_dir))
    path = os.path.join(OUT, "summary.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    summary[tag] = res
    json.dump(summary, open(path, "w"), indent=1)
    for f in ("fig8_ee_error.png", "fig9_cost_ratio.png", "fig10_sampling.png"):
        os.replace(os.path.join(OUT, f), os.path.join(OUT, f.replace(".png", f"_{tag}.png")))
    print(json.dumps({k: v for k, v in res.items() if k not in ("config", "training")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--src", default=SRC)
    d.add_argument("--u-bar", default="zero", choices=["zero", "mid"])
    d.add_argument("--name", default=None, help="output npz name under results/.../hsa/")
    e = sub.add_parser("eval")
    e.add_argument("--model-dir", required=True)
    e.add_argument("--tag", default=None)
    e.add_argument("--n-targets", type=int, default=2000)
    e.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    MakeData(a) if a.cmd == "data" else Eval(a)


if __name__ == "__main__":
    main()
