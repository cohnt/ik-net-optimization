"""LOInK paper (arXiv 2609.21275) Sec. V replication: the planar 3-DoF manipulator.

The robot. Three links of l = 1 m from the origin, q in [0, pi]^3 (elbow up), task
y = p_x (scalar). Link i's absolute angle is theta_1 = q_1, theta_i = theta_{i-1} - q_i
(i >= 2): the base joint measured counter-clockwise from +x, the two relative joints bending
CLOCKWISE, p = sum_i l [cos theta_i, sin theta_i]. This is the convention Fig. 3 draws (at
y = 1.00 the first link points up-left, q ~ (2.03, 1.39, 1.50); at y = 2.90 q ~ (0.46, 0.38,
0.31)). The all-counter-clockwise convention (theta_i = sum_{k<=i} q_k) is its mirror image
(q_1 -> pi - q_1, p_x -> -p_x; J is invariant), so it poses the SAME problem at -y: there
the paper's targets y = 1.00 / 1.95 / 2.90 sit in the sparse, high-cost end of the dataset
(3.5% of samples have p_x > 1) instead of the dense end.

The cost, eq. (18)-(20). q_hat = (q - pi/2)/(pi/2) in [-1, 1];
J_q = -sum_i log(max(1 - q_hat_i^2, eps)), eps = 1e-3 (the paper: "some small conditioning
scalar"). J_com = (l/3) sum_{i=0}^{3} |c^i_x|^2: the paper's sum runs over FOUR indices for
three links; we take c^0 = the base (at the origin, so its term is identically zero) and
c^1..c^3 the link midpoints (each link's centre of mass), i.e.
J_com = (l/3) sum_{i=1}^{3} (x of the midpoint of link i)^2. J = J_q + lam_com J_com,
lam_com not given (default 1.0).

Network input (the paper does not say): the fielded runs use `data --centre` (x = q - pi/2)
and train with `--scale 1.5707963267948966`, so the network sees q_hat in [-1, 1]; eq. (17)
then bounds |dq| <= (pi/2) |dy| / mu, which is what `eval` checks.

    planar3dof.py data --lam-com 1.0 --centre --out results/loink_replication/planar3dof/data_lc1.0.npz
    train_replication.py --data .../data_lc1.0.npz --out-dir .../relu_mu0.1_lam0.01 \
        --scale 1.5707963267948966 --lam 0.01 [--activation tanh] [--mu 0.05|1.5]
    planar3dof.py eval --model relu=DIR --mu-model 0.05=DIR --mu-model 0.1=DIR --mu-model 1.5=DIR
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from src.charts.loink import dys_inverse, invert_bilipnet  # noqa: E402
from train_replication import CostHatData, LoadModel  # noqa: E402

L = 1.0
EPS = 1e-3
OUT = os.environ.get("PLANAR_OUT", os.path.join(REPO, "results", "loink_replication", "planar3dof"))
TARGETS = (1.00, 1.95, 2.90)


SIGN = np.array([1.0, -1.0, -1.0])     # theta_i = theta_{i-1} - q_i for i >= 2 (Fig. 3)


def Theta(q):
    return np.cumsum(SIGN * q, axis=-1)


def Points(q):
    """Joint positions [..., 4, 2] (base, joints 2, 3, end effector)."""
    a = Theta(q)
    d = L * np.stack([np.cos(a), np.sin(a)], -1)
    return np.concatenate([np.zeros(q.shape[:-1] + (1, 2)), np.cumsum(d, axis=-2)], axis=-2)


def Px(q):
    return L * np.cos(Theta(q)).sum(-1)


def Jq(q):
    qh = (q - np.pi / 2) / (np.pi / 2)
    return -np.log(np.maximum(1 - qh ** 2, EPS)).sum(-1)


def Jcom(q):
    P = Points(q)
    mid_x = 0.5 * (P[..., :-1, 0] + P[..., 1:, 0])        # link midpoints c^1..c^3
    return (L / 3) * (mid_x ** 2).sum(-1)


def Cost(q, lam_com):
    return Jq(q) + lam_com * Jcom(q)


@__import__("functools").lru_cache(maxsize=None)
def TrueOptimum(y, lam_com, n_start=64, seed=0):
    """min J(q) s.t. p_x(q) = y, q in [0, pi]^3 by SLSQP from n_start random in-limit starts
    (evaluation reference only)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    best = None
    lo, hi = 1e-6, np.pi - 1e-6
    for _ in range(n_start):
        q0 = rng.uniform(0.05, np.pi - 0.05, 3)
        r = minimize(lambda q: Cost(q, lam_com), q0, method="SLSQP", bounds=[(lo, hi)] * 3,
                     constraints=[{"type": "eq", "fun": lambda q: Px(q) - y}],
                     options=dict(maxiter=500, ftol=1e-12))
        if r.success and abs(Px(r.x) - y) < 1e-8 and (best is None or r.fun < best.fun):
            best = r
    return best.x, float(best.fun)


def MakeData(a):
    rng = np.random.default_rng(a.seed)
    q = rng.uniform(0.0, np.pi, (a.n, 3))
    off = np.full(3, np.pi / 2) if a.centre else np.zeros(3)
    np.savez(a.out, x=(q - off).astype(np.float32), x_offset=off, y=Px(q)[:, None].astype(np.float32),
             j=Cost(q, a.lam_com), metadata=json.dumps(dict(
                 n=a.n, seed=a.seed, lam_com=a.lam_com, eps=EPS, l=L,
                 J_com="(l/3) sum_{i=1..3} (link-i midpoint x)^2; c^0 = base (zero)")))
    j = Cost(q, a.lam_com)
    print(f"wrote {a.out}: n {a.n}, y in [{Px(q).min():.3f}, {Px(q).max():.3f}], "
          f"J mean {j.mean():.3f} min {j.min():.3f} max {j.max():.3f}")


## ---------------------------------------------------------------------- evaluation

def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def Offset(model):
    """x = q - x_offset in the training data (`data --centre`); zeros otherwise."""
    if not hasattr(model, "_x_offset"):
        d = np.load(model._data_path)
        model._x_offset = d["x_offset"] if "x_offset" in d.files else np.zeros(3)
    return model._x_offset


def DecodeBatch(model, y, z):
    q = model.decode(_t(y).reshape(-1, 1), _t(z).reshape(-1, model.K))
    return q.detach().numpy() + Offset(model)


def Fig3(models, lam_com, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {}
    zs = np.linspace(-3, 3, 81)
    Z1, Z2 = np.meshgrid(zs, zs)
    Zg = np.stack([Z1.ravel(), Z2.ravel()], 1)
    for name, (model, ckpt) in models.items():
        r = {}
        fig, ax = plt.subplots(2, 3, figsize=(11, 7))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for k, y in enumerate(TARGETS):
            q = DecodeBatch(model, [y], np.zeros((1, 2)))[0]
            q_opt, J_opt = TrueOptimum(y, lam_com)
            yy = np.full(len(Zg), y)
            Jhat = CostHatData(model, ckpt, _t(Zg), _t(yy)[:, None]).detach().numpy().reshape(Z1.shape)
            Qg = DecodeBatch(model, yy, Zg)
            Jtrue = Cost(Qg, lam_com).reshape(Z1.shape)
            inlim = ((Qg >= 0) & (Qg <= np.pi)).all(1).reshape(Z1.shape)
            iy, ix = np.unravel_index(np.argmin(Jtrue), Jtrue.shape)
            jy, jx = np.unravel_index(np.argmin(Jhat), Jhat.shape)
            J0 = float(Cost(q, lam_com))
            r[f"y={y:.2f}"] = dict(
                q_star=q.tolist(), p_x=float(Px(q)), abs_err=float(abs(Px(q) - y)),
                J=J0, J_q=float(Jq(q)), J_com=float(Jcom(q)),
                J_hat_at_0=float(CostHatData(model, ckpt, _t(np.zeros((1, 2))), _t([[y]])).item()),
                in_limits=bool(((q >= 0) & (q <= np.pi)).all()),
                elbow_up_all_joints_in_0_pi=bool(((q >= 0) & (q <= np.pi)).all()),
                true_opt_q=q_opt.tolist(), true_opt_J=J_opt, optimality_gap=J0 - J_opt,
                grid_true_J_argmin_z=[float(Z1[iy, ix]), float(Z2[iy, ix])],
                grid_true_J_min=float(Jtrue[iy, ix]),
                grid_Jhat_argmin_z=[float(Z1[jy, jx]), float(Z2[jy, jx])],
                grid_frac_in_limits=float(inlim.mean()),
                grid_task_err_max=float(np.abs(Px(Qg) - y).max()))
            levels = np.linspace(min(Jhat.min(), Jtrue.min()), min(Jhat.min(), Jtrue.min()) + 6, 25)
            for row, (F, lab) in enumerate(((Jhat, "J_hat(z, y)"), (Jtrue, "J(decode(y, z))"))):
                cs = ax[row, k].contourf(Z1, Z2, np.minimum(F, levels[-1]), levels=levels, cmap="magma_r")
                ax[row, k].plot(0, 0, "o", color="tab:red", ms=4)
                ax[row, k].plot(Z1[iy, ix] if row else Z1[jy, jx], Z2[iy, ix] if row else Z2[jy, jx],
                                "x", color="tab:cyan", ms=7)
                if row:
                    ax[row, k].contour(Z1, Z2, inlim.astype(float), levels=[0.5], colors="w",
                                       linewidths=0.8, linestyles="--")
                ax[row, k].set_title(f"y = {y:.2f}: {lab}", fontsize=9)
                ax[row, k].set_aspect("equal")
                ax[row, k].set_xlabel("z1")
                ax[row, k].set_ylabel("z2")
                fig.colorbar(cs, ax=ax[row, k], shrink=0.8)
            P = Points(q)
            ax2.plot(P[:, 0], P[:, 1], "-o", lw=3, label=f"y = {y:.2f}")
            ax2.plot([y], [-0.4], "k^")
        fig.suptitle(f"{name}: top J_hat, bottom true J (red o: z = 0; cyan x: grid argmin; "
                     "white dashes: in-limits boundary)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig3_levelsets_{name}.png"), dpi=130)
        ax2.plot(0, 0, "ko", ms=10)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.legend(fontsize=8)
        ax2.set_title(f"{name}: q* = G^-1([y; 0])")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out, f"fig3_configs_{name}.png"), dpi=130)
        plt.close("all")
        res[name] = r
    return res


def Smoothstep(t):
    return 3 * t ** 2 - 2 * t ** 3


def Fig4(models, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.linspace(0, 1, 200)
    y = 1 + 2 * Smoothstep(t)
    ydot = 2 * (6 * t - 6 * t ** 2)
    res = {}
    fig, ax = plt.subplots(4, 1, figsize=(6, 9), sharex=True)
    ax[0].plot(t, y, "r")
    ax[0].set_ylabel("y")
    for name, (model, ckpt) in models.items():
        mu, nu, _ = (float(v) for v in model.G.bounds())
        ## eq. (17) holds for x_net = (q - offset) / scale: |q_dot| <= max(scale) |y_dot| / mu
        smax, smin = float(model.scale.max()), float(model.scale.min())
        q = DecodeBatch(model, y, np.zeros((len(t), 2)))
        qdot = np.gradient(q, t, axis=0)
        speed = np.linalg.norm(qdot, axis=1)
        ratio = speed[1:-1] / np.abs(ydot[1:-1])        # ydot = 0 at the ends
        ## eq. (17) as the Lipschitz statement it is, on every chord between consecutive
        ## samples (both at z = 0): |dq| / |dy| in [min(scale)/nu, max(scale)/mu]
        chord = np.linalg.norm(np.diff(q, axis=0), axis=1) / np.abs(np.diff(y))
        res[name] = dict(mu=mu, nu=nu, max_qdot_norm=float(speed.max()),
                         max_ydot=float(np.abs(ydot).max()),
                         max_ratio_qdot_over_ydot=float(ratio.max()),
                         min_ratio_qdot_over_ydot=float(ratio.min()),
                         scale=smax, bound_scale_over_mu=smax / mu,
                         lower_bound_scale_over_nu=smin / nu,
                         chord_ratio_max=float(chord.max()), chord_ratio_min=float(chord.min()),
                         eq17_upper_holds=bool((chord <= smax / mu * (1 + 1e-9)).all()),
                         eq17_lower_holds=bool((chord >= smin / nu * (1 - 1e-9)).all()),
                         task_err_max=float(np.abs(Px(q) - y).max()),
                         in_limits_all=bool(((q >= 0) & (q <= np.pi)).all()),
                         total_variation_qdot=float(np.abs(np.diff(qdot, axis=0)).sum()))
        for i in range(3):
            ax[i + 1].plot(t, qdot[:, i], label=f"mu = {mu:g} (nu = {nu:.1f})")
            ax[i + 1].set_ylabel(f"qdot{i + 1}")
    ax[1].legend(fontsize=8)
    ax[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_joint_velocities.png"), dpi=130)
    plt.close(fig)
    return res


def InversionTable(model, y, dtype):
    """max / mean |G(x) - [y; 0]| for x from DYS alone and DYS + IFT step + n_newton polish."""
    m = model.to(dtype)
    w = torch.cat([torch.as_tensor(y, dtype=dtype).reshape(-1, 1),
                   torch.zeros(len(y), m.K, dtype=dtype)], 1)
    rows = {}
    with torch.no_grad():
        for K in (10, 50, 100, 200):
            xr = dys_inverse(m.G, w, K)
            r = (m.G(xr) - w).norm(dim=1)
            rows[f"DYS {K}"] = (float(r.max()), float(r.mean()))
    for nn_ in (0, 1, 2):
        x = invert_bilipnet(m.G, w, 100, nn_).detach()
        with torch.no_grad():
            r = (m.G(x) - w).norm(dim=1)
        rows[f"DYS 100 + IFT step, n_newton={nn_}"] = (float(r.max()), float(r.mean()))
    model.to(torch.float64)
    return rows


def Eval(a):
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    summary = {}
    spath = os.path.join(OUT, "summary.json")
    if os.path.exists(spath):
        summary = json.load(open(spath))
    def load(spec):
        name, d = spec.split("=", 1)
        model, ckpt = LoadModel(os.path.join(d, "model.pt"), dtype=torch.float64)
        model._data_path = ckpt["config"]["data"]
        return name, d, model, ckpt
    main_models, mu_models, dirs = {}, {}, {}
    for spec in a.model:
        name, d, m, c = load(spec)
        main_models[name] = (m, c)
        dirs[name] = d
        fin = json.load(open(os.path.join(d, "final.json")))
        summary.setdefault("training", {})[name] = dict(dir=os.path.relpath(d, REPO), **fin["final"],
                                                        config=fin["config"])
    for spec in a.mu_model:
        name, d, m, c = load(spec)
        mu_models[name] = (m, c)
        dirs[f"mu={name}"] = d
        fin = json.load(open(os.path.join(d, "final.json")))
        summary.setdefault("training", {})[f"mu_{name}"] = dict(dir=os.path.relpath(d, REPO),
                                                                **fin["final"], config=fin["config"])
    lam_com = a.lam_com
    summary["lam_com"] = lam_com
    t = time.time()
    if not a.skip_fig3:
        summary["fig3"] = Fig3(main_models, lam_com, OUT)
    if mu_models:
        summary["fig4"] = Fig4({f"mu={k}": v for k, v in mu_models.items()}, OUT)
    ## inversion residual on 2000 held-out targets (y of validation samples)
    rng = np.random.default_rng(0)
    inv = {}
    for name, (m, c) in ({} if a.skip_inversion else
                         {**main_models, **{f"mu={k}": v for k, v in mu_models.items()}}).items():
        vi = np.load(os.path.join(dirs[name], "val_idx.npy"))
        ys = np.load(c["config"]["data"])["y"][rng.choice(vi, 2000, replace=False), 0].astype(np.float64)
        inv[name] = {"float64": InversionTable(m, ys, torch.float64),
                     "float32": InversionTable(m, ys, torch.float32)}
    if not a.skip_inversion:
        summary["inversion_residual_norm_max_mean"] = inv
    summary["eval_wall_s" + ("_fig4" if a.skip_fig3 else "")] = round(time.time() - t, 1)
    with open(spath, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in summary if k != "training"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n", type=int, default=1_000_000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--lam-com", type=float, default=1.0)
    d.add_argument("--centre", action="store_true",
                   help="store x = q - pi/2 (decode returns x; eval adds the offset back)")
    d.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", action="append", default=[], help="name=DIR (Fig. 3 models)")
    e.add_argument("--mu-model", action="append", default=[], help="mu=DIR (Fig. 4 models)")
    e.add_argument("--lam-com", type=float, default=1.0)
    e.add_argument("--skip-fig3", action="store_true")
    e.add_argument("--skip-inversion", action="store_true")
    a = ap.parse_args()
    if a.cmd == "data":
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        MakeData(a)
    else:
        Eval(a)


if __name__ == "__main__":
    main()
