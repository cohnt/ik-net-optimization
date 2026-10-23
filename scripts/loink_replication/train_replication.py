"""Plain-torch trainer for a LOInK model (`src/charts/loink.py`) on an npz dataset.

Replication of the LOInK paper (arXiv 2609.21275) Sec. V / VII. The dataset npz holds
`x` [N, n] (configuration / input), `y` [N, p] (task) and `j` [N] (cost).

Recipe (the paper's, Sec. V.B / VII.B, defaults here): Adam, lr 1e-3, batch 500, 3 epochs,
reduce-on-plateau with patience 500 STEPS, mu = 0.1 fixed, nu free with a small distortion
penalty (`loink_loss`: tau_penalty * log tau), loss = task MSE + lam * aux MSE (eq. 16).

Choices the paper does not fix (recorded in the saved config):
  * plateau signal: the training loss, smoothed by an exponential moving average
    (`--plateau-ema 0.99`, i.e. ~100-step horizon), handed to `ReduceLROnPlateau` EVERY step,
    patience 500 steps, factor 0.5, torch's default threshold (rel 1e-4). A per-step raw
    minibatch loss is noise-dominated; the EMA is what "patience of 500 steps" can sensibly
    watch. (optax's `contrib.reduce_on_plateau`, the likely JAX original, defaults factor
    0.1; ours is 0.5 as instructed, `--plateau-factor`.)
  * nu > mu guard: `loink_loss`'s log-tau penalty pulls each layer's nu toward its fixed mu
    and nothing in the parametrisation keeps nu > mu (sqrt(2 gam) is NaN otherwise). After
    every optimiser step lognu is clamped to >= log(mu (1 + 1e-3)); the number of steps at
    which the clamp was active is logged (`nu_clamped`).
  * `--clip-grad`: global gradient-norm clipping, OFF by default (the paper names none).
    Without it a single loss spike (seen on the HSA run and the planar tanh run) pushes the
    EMA above its pre-spike minimum for good, and the plateau scheduler then halves the
    learning rate every `patience` steps down to ~0.
  * `--scale`: the model's input scaling, x_net = x / scale (a positive scalar or one value
    per coordinate); the paper does not say how inputs are normalised.
  * `--j-standardise`: fit the head to (j - j_mean) / j_std; the saved model carries both,
    and J_hat in data units is j_mean + j_std * cost_hat (still a PD quadratic in z with its
    minimum at z = 0, so the decoder's z = 0 is unchanged in meaning).
  * `--j-scale s`: fit the head to j / s instead (j_mean = 0, j_std = s). Because
    M = I/2 + P^T P bounds J_hat - b below by |z|^2 / 2 and G's lower Lipschitz bound mu
    forces |dz| >= mu |dx| along a fibre, a cost whose curvature along the fibre is below
    ~mu^2 (in the units J_hat is fitted in) cannot be represented near its minimum;
    s < 1 scales the fitted cost up.
  * validation: 20k samples (`--n-val`) held out by a seeded permutation; their indices are
    saved (`val_idx.npy`) so evaluation scripts use the same split.

    python scripts/loink_replication/train_replication.py --data D.npz --out-dir OUT
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

from src.charts.loink import LoinkModel, loink_loss  # noqa: E402


def BuildModel(cfg):
    return LoinkModel(cfg["n"], cfg["p"], cfg["scale"], units=tuple(cfg["units"]),
                      depth=cfg["depth"], mu=cfg["mu"], activation=cfg["activation"],
                      cost="m" if cfg["cost_head"] else "u", head_hidden=cfg["head_hidden"],
                      n_dys=cfg["n_dys"], n_newton=cfg["n_newton"])


def LoadModel(path, device="cpu", dtype=torch.float32):
    """(model, ckpt) from a `model.pt` this script wrote; model in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = BuildModel(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device=device, dtype=dtype).eval(), ckpt


def CostHatData(model, ckpt, z, y_hat):
    """J_hat in the dataset's units (undoing `--j-standardise`)."""
    return ckpt["j_mean"] + ckpt["j_std"] * model.cost_hat(z, y_hat)


def _clamp_nu(model, rel=1e-3):
    hit = False
    with torch.no_grad():
        for mon in model.G.mon:
            lo = math.log(float(mon.mu) * (1 + rel))
            if float(mon.lognu) < lo:
                mon.lognu.fill_(lo)
                hit = True
    return hit


@torch.no_grad()
def Evaluate(model, x, y, j, j_mean, j_std, batch=5000):
    """Validation: task RMSE (sqrt of eq. 14's mean squared norm), aux MSE in standardised
    and in data units."""
    se_t = se_a = 0.0
    n = len(x)
    for i in range(0, n, batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        y_hat, z = model.forward_G(xb)
        se_t += ((y_hat - yb) ** 2).sum().item()
        if model.H is not None:
            jb = (j[i:i + batch] - j_mean) / j_std
            se_a += ((model.cost_hat(z, y_hat) - jb) ** 2).sum().item()
    out = {"val_task_rmse": math.sqrt(se_t / n)}
    if model.H is not None:
        out["val_aux_mse"] = se_a / n
        out["val_aux_mse_data_units"] = se_a / n * j_std ** 2
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-train", type=int, default=None, help="cap on training samples")
    ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--units", default="256,256,256,256")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--cost-head", dest="cost_head", action="store_true", default=True)
    ap.add_argument("--no-cost-head", dest="cost_head", action="store_false")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--tau-penalty", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=500)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--plateau-ema", type=float, default=0.99)
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help="clip the global gradient norm to this (0 = off, the paper's recipe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--j-standardise", action="store_true")
    ap.add_argument("--j-scale", type=float, default=None,
                    help="fit the head to j / j_scale (j_mean = 0); excludes --j-standardise")
    ap.add_argument("--scale", default="1.0", help="scalar or comma list: x_net = x / scale")
    ap.add_argument("--n-dys", type=int, default=100)
    ap.add_argument("--n-newton", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=100)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.device == "cpu" and "OMP_NUM_THREADS" in os.environ:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    dev = torch.device(a.device)

    d = np.load(a.data)
    X, Y, J = d["x"].astype(np.float32), d["y"].astype(np.float32), d["j"].astype(np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    N, n = X.shape
    p = Y.shape[1]
    perm = rng.permutation(N)
    val_idx, tr_idx = perm[:a.n_val], perm[a.n_val:]
    if a.n_train is not None:
        tr_idx = tr_idx[:a.n_train]
    np.save(os.path.join(a.out_dir, "val_idx.npy"), val_idx)
    if a.j_standardise:
        j_mean, j_std = float(J[tr_idx].mean()), float(J[tr_idx].std())
    elif a.j_scale is not None:
        j_mean, j_std = 0.0, float(a.j_scale)
    else:
        j_mean, j_std = 0.0, 1.0
    scale = [float(s) for s in a.scale.split(",")]
    scale = scale * n if len(scale) == 1 else scale

    cfg = dict(n=n, p=p, scale=scale, units=[int(u) for u in a.units.split(",")],
               depth=a.depth, mu=a.mu, activation=a.activation, cost_head=a.cost_head,
               head_hidden=a.head_hidden, n_dys=a.n_dys, n_newton=a.n_newton, lam=a.lam,
               tau_penalty=a.tau_penalty, lr=a.lr, batch=a.batch, epochs=a.epochs,
               patience=a.patience, plateau_factor=a.plateau_factor, clip_grad=a.clip_grad,
               plateau_signal=f"EMA({a.plateau_ema}) of the train loss, every step",
               nu_guard="lognu clamped to >= log(mu (1 + 1e-3)) after each step",
               seed=a.seed, device=a.device, j_standardise=a.j_standardise, j_scale=a.j_scale,
               data=os.path.abspath(a.data), n_train=len(tr_idx), n_val=len(val_idx))
    model = BuildModel(cfg).to(dev)

    Xt = torch.from_numpy(X[tr_idx]).to(dev)
    Yt = torch.from_numpy(Y[tr_idx]).to(dev)
    Jt = torch.from_numpy(((J[tr_idx] - j_mean) / j_std).astype(np.float32)).to(dev)
    Xv = torch.from_numpy(X[val_idx]).to(dev)
    Yv = torch.from_numpy(Y[val_idx]).to(dev)
    Jv = torch.from_numpy(J[val_idx].astype(np.float32)).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=a.plateau_factor,
                                                       patience=a.patience)
    lam = a.lam if a.cost_head else 0.0
    steps_per_epoch = len(tr_idx) // a.batch
    log = open(os.path.join(a.out_dir, "log.jsonl"), "w")
    ema, acc, n_acc, clamped, step, gmax = None, {}, 0, 0, 0, 0.0
    t0 = time.time()

    def save(tag):
        ckpt = dict(state_dict=model.state_dict(), config=cfg, j_mean=j_mean, j_std=j_std,
                    step=step, tag=tag)
        torch.save(ckpt, os.path.join(a.out_dir, "model.pt"))

    for epoch in range(a.epochs):
        order = torch.randperm(len(tr_idx), device=dev)
        model.train()
        for b in range(steps_per_epoch):
            idx = order[b * a.batch:(b + 1) * a.batch]
            loss, parts = loink_loss(model, Xt[idx], Yt[idx], Jt[idx] if a.cost_head else None,
                                     lam=lam, tau_penalty=a.tau_penalty)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                a.clip_grad if a.clip_grad > 0 else float("inf"))
            gmax = max(gmax, gn.item())
            opt.step()
            clamped += _clamp_nu(model)
            step += 1
            lv = loss.item()
            if not math.isfinite(lv):
                raise RuntimeError(f"non-finite loss at step {step}: {parts}")
            ema = lv if ema is None else a.plateau_ema * ema + (1 - a.plateau_ema) * lv
            sched.step(ema)
            for k in ("task_mse", "aux_mse", "dist"):
                if k in parts:
                    acc[k] = acc.get(k, 0.0) + parts[k].item()
            n_acc += 1
            if step % a.log_every == 0:
                rec = {"step": step, "epoch": epoch, "time": round(time.time() - t0, 2),
                       "loss_ema": ema, "lr": opt.param_groups[0]["lr"],
                       "task_rmse": math.sqrt(acc["task_mse"] / n_acc),
                       "nu": parts["nu"].item(), "tau": parts["tau"].item(),
                       "nu_layers": [m.nu().item() for m in model.G.mon],
                       "nu_clamped": clamped, "grad_norm_max": gmax}
                if "aux_mse" in acc:
                    rec["aux_mse"] = acc["aux_mse"] / n_acc
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                acc, n_acc, gmax = {}, 0, 0.0
        model.eval()
        ev = Evaluate(model, Xv, Yv, Jv, j_mean, j_std)
        ev.update(step=step, epoch=epoch, time=round(time.time() - t0, 2), kind="val")
        log.write(json.dumps(ev) + "\n")
        log.flush()
        print(json.dumps(ev), flush=True)
        save(f"epoch{epoch}")

    with torch.no_grad():
        mu, nu, tau = model.G.bounds()
    final = dict(Evaluate(model, Xv, Yv, Jv, j_mean, j_std), steps=step,
                 wall_s=round(time.time() - t0, 1), device=a.device,
                 torch_threads=torch.get_num_threads(), mu=float(mu), nu=float(nu),
                 tau=float(tau), nu_layers=[m.nu().item() for m in model.G.mon],
                 final_lr=opt.param_groups[0]["lr"], nu_clamped_steps=clamped,
                 j_mean=j_mean, j_std=j_std)
    save("final")
    with open(os.path.join(a.out_dir, "final.json"), "w") as f:
        json.dump(dict(config=cfg, final=final), f, indent=1)
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
