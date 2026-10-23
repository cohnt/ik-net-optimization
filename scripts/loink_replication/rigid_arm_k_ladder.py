"""The latent-width (K) ladder that diagnoses why LOInK cannot chart a 7-DoF arm's pose task.

What it shows (docs/notes-chart.md, "Training outcome (2026-10-23)"). LOInK makes the IK map the
first p coordinates of ONE bi-Lipschitz chart G of R^n, leaving a latent of width K = n - p. On
a 7-DoF arm with the 6-D pose task that is K = 1, and the chart fits neither position (~0.27 m
held-out RMSE per axis) nor orientation (rpy at the mean-predictor level) while its distortion
tau grows. Giving the latent room restores learning: padding the input with dummy coordinates
(K = 4, 7), or dropping half the task (rpy only or xyz only, K = 4), lowers the held-out error,
xyz-only to ~0.1 m. So the bottleneck is the latent width LOInK prescribes, not the optimiser or
the port. Each row trains the same `LoinkModel` / `loink_loss` from scratch on iiwa14's local
ikflow dataset and reports the held-out RMSE per task column (position m, rpy rad wrapped to
(-pi, pi]) every 1000 steps.

How it was run (2026-10-23, the numbers in docs/notes-chart.md): on the laptop, CPU only, one
process in the foreground under the memory scope, 4 torch threads; about 2 min per row, ~11 min
for all five:

    CUDA_VISIBLE_DEVICES= systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
        .venv/bin/python scripts/loink_replication/rigid_arm_k_ladder.py

Fixed recipe: cost "u" (no head), units (256,)*4, depth 2, mu 0.1, relu, float64; Adam 1e-3,
batch 500 drawn with replacement, `clip_grad_value_` 1.0, `tau_penalty` 1e-3, 3000 steps, seed 0
(`torch.manual_seed(0)` before each row's model is built). Held-out set: the first 5000 test
samples. The dummy inputs are uniform in [-1, 1] from a generator seeded with the split's sample
count, so they are a deterministic function of the sample index (and differ between the train
and test splits, as an unrelated coordinate should).
"""
import argparse
import json
import math
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

from src.charts.loink import LoinkModel, Pose7ToXyzRpy, loink_loss  # noqa: E402

DATASET_DIR = {"iiwa14": os.path.expanduser("~/.cache/ikflow/datasets/iiwa14/")}
DATASET_TAG = "__tag0=non-self-colliding"
# The iiwa14's per-joint max(|lo|, |hi|) (170/120/170/120/170/120/175 deg) to 4 decimals --
# what `src.charts.cvae.JointScale(jrl.robots.get_robot("iiwa14"))` returns, up to rounding.
# Kept literal: importing jrl sets torch's global default device and dtype (jrl/config.py)
# and probes the GPUs, and the diagnosis's numbers were taken with these rounded values.
JOINT_SCALE = {"iiwa14": [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543]}
COLUMN_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
N_HELD_OUT = 5000
LOG_EVERY = 1000

# name -> (label, dummy inputs, task columns)
ROWS = {
    "p6": ("p6, no pad", 0, [0, 1, 2, 3, 4, 5]),
    "p6pad3": ("p6, pad 3", 3, [0, 1, 2, 3, 4, 5]),
    "p6pad6": ("p6, pad 6", 6, [0, 1, 2, 3, 4, 5]),
    "rpy": ("p3 rpy, no pad", 0, [3, 4, 5]),
    "xyz": ("p3 xyz, no pad", 0, [0, 1, 2]),
}


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def pad(q, k):
    """Append k dummy inputs uniform in [-1, 1], from a generator seeded with q's row count."""
    if k == 0:
        return q
    gen = torch.Generator().manual_seed(q.shape[0])
    tab = torch.rand((q.shape[0], k), generator=gen, dtype=q.dtype) * 2 - 1
    return torch.cat([q, tab], 1)


def load_dataset(robot):
    d = DATASET_DIR[robot]

    def load(name):  # the tensors were saved on CUDA
        return torch.load(os.path.join(d, name + DATASET_TAG), map_location="cpu")

    qtr = load("samples_tr.pt").double()
    ytr = Pose7ToXyzRpy(load("endpoints_tr.pt").double())
    qte = load("samples_te.pt")[:N_HELD_OUT].double()
    yte = Pose7ToXyzRpy(load("endpoints_te.pt")[:N_HELD_OUT].double())
    return qtr, ytr, qte, yte


def run_row(name, data, scale7, steps, lr=1e-3, batch=500):
    label, k_pad, cols = ROWS[name]
    qtr, ytr, qte, yte = data
    torch.manual_seed(0)
    n_q = qtr.shape[1]
    n, p = n_q + k_pad, len(cols)
    scale = torch.cat([scale7, torch.ones(k_pad).double()])
    model = LoinkModel(n, p, scale, units=(256,) * 4, depth=2, mu=0.1, activation="relu",
                       cost="u").double()
    Qtr, Qte = pad(qtr, k_pad), pad(qte, k_pad)
    Ytr, Yte = ytr[:, cols], yte[:, cols]
    ang = [i for i, c in enumerate(cols) if c >= 3]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0, N, log = time.time(), Qtr.shape[0], []
    for s in range(steps + 1):
        idx = torch.randint(0, N, (batch,))
        loss, parts = loink_loss(model, Qtr[idx], Ytr[idx], None, lam=0.01, tau_penalty=1e-3)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        opt.step()
        if s % LOG_EVERY == 0 or s == steps:
            with torch.no_grad():
                r = model.forward_G(Qte)[0] - Yte
                if ang:
                    r[:, ang] = wrap(r[:, ang])
                per = torch.sqrt((r ** 2).mean(0)).tolist()
            wall = time.time() - t0
            tau = parts["tau"].item()
            print(f"{label:28s} K={n - p} step {s:4d} tau {tau:7.1f} held-out rmse/col "
                  f"{[round(v, 3) for v in per]} {wall:.0f}s", flush=True)
            log.append({"step": s, "tau": tau, "wall_s": wall,
                        "rmse": {COLUMN_NAMES[c]: v for c, v in zip(cols, per)}})
    return {"label": label, "K": n - p, "p": p, "n": n, "pad": k_pad,
            "columns": [COLUMN_NAMES[c] for c in cols], "log": log,
            "wall_s": time.time() - t0}


def print_summary(results):
    print("\nfinal held-out RMSE (position m, rpy rad):")
    print(f"{'row':28s} {'K':>2s}  {'x / y / z':23s} roll / pitch / yaw")
    for r in results.values():
        last = r["log"][-1]["rmse"]

        def group(names):
            return " / ".join(f"{last[c]:.3f}" for c in names) if names[0] in last else "-"

        print(f"{r['label']:28s} {r['K']:2d}  {group(['x', 'y', 'z']):23s} "
              f"{group(['roll', 'pitch', 'yaw'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--robot", default="iiwa14")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rows", default=",".join(ROWS),
                    help=f"comma-separated subset of {list(ROWS)} (default: all, in that order)")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (the record: 4)")
    ap.add_argument("--out", default=None,
                    help="JSON output (default results/loink_replication/k_ladder/<robot>.json)")
    args = ap.parse_args()
    if args.robot not in DATASET_DIR:
        sys.exit(f"--robot {args.robot!r}: only {sorted(DATASET_DIR)} is wired up (its local "
                 f"ikflow dataset and joint scale); add the robot's entries to DATASET_DIR and "
                 f"JOINT_SCALE first")
    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    unknown = [r for r in rows if r not in ROWS]
    if unknown:
        sys.exit(f"--rows: unknown {unknown}; choose from {list(ROWS)}")
    out = args.out or os.path.join(REPO, "results", "loink_replication", "k_ladder",
                                   f"{args.robot}.json")
    torch.set_num_threads(args.threads)

    data = load_dataset(args.robot)
    scale7 = torch.tensor(JOINT_SCALE[args.robot]).double()
    results = {}
    for name in rows:
        results[name] = run_row(name, data, scale7, args.steps)
    print_summary(results)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    meta = {"robot": args.robot, "steps": args.steps, "threads": args.threads, "seed": 0,
            "n_train": data[0].shape[0], "n_held_out": data[2].shape[0],
            "dataset_dir": DATASET_DIR[args.robot], "dataset_tag": DATASET_TAG,
            "joint_scale": JOINT_SCALE[args.robot], "torch": torch.__version__,
            "recipe": {"cost": "u", "units": [256] * 4, "depth": 2, "mu": 0.1,
                       "activation": "relu", "dtype": "float64", "optimizer": "Adam",
                       "lr": 1e-3, "batch": 500, "clip_grad_value": 1.0,
                       "tau_penalty": 1e-3}}
    with open(out, "w") as f:
        json.dump({"meta": meta, "rows": results}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
