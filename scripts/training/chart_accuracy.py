"""Chart accuracy: how far is FK(flow(c, z)) from the conditioning pose c?

Comparable to the recorded medians -- lemon-haze-7 16.6 mm / 64.5 mm p90, Panda 3.8 / 9.4 --
and the number acceptance criterion 2 is judged on. Conditioning poses are real task poses
(FK of a random collision-free configuration), the distribution both benchmark tasks draw
targets from, matching scripts/training/pole_at_task_poses.py.

Runaway outputs are EXCLUDED (|q|_inf >= 10 rad): a configuration of 1e12 rad has no
meaningful "chart error", and including it would make a pole-free network look worse than
it is while telling you nothing the pole metric does not already say. The fraction excluded
is reported -- read it alongside the error, since a network can buy accuracy by being
confidently wrong on the rest.

Measured 2026-09-07 (N=5000, seed 0):

    network                        usable    median     p90
    iiwa14 ddp-r1 step620000       99.04%    10.70 mm   85.5 mm
    iiwa14 lemon-haze-7            94.30%    22.14 mm  120.8 mm
    iiwa14 elated-firefly-11 (6b)  99.88%    29.75 mm  122.2 mm

Usage:
    python scripts/training/chart_accuracy.py --n 5000
    python scripts/training/chart_accuracy.py --checkpoint <pkl> --nb_nodes 6
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts/training"))

from pole_metric import load_solver  # noqa: E402

RUNAWAY_RAD = 10.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--nb_nodes", type=int, default=12)
    p.add_argument("--dim_latent_space", type=int, default=8)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    default_ckpt = os.path.join(
        REPO_ROOT, "models/iiwa14/iiwa14__ddp-r1__step620000.pkl")
    ckpt = args.checkpoint or default_ckpt
    solver, width, ndof = load_solver("iiwa14", ckpt, args.nb_nodes, args.dim_latent_space)
    robot = solver.robot

    rng = np.random.default_rng(args.seed)
    lo, hi = np.array(robot.actuated_joints_limits).T
    q_target = rng.uniform(lo, hi, size=(args.n, ndof))
    pose = robot.forward_kinematics(torch.tensor(q_target, dtype=torch.float64))
    pose = pose.detach().cpu().numpy() if hasattr(pose, "detach") else np.asarray(pose)
    z = rng.normal(size=(args.n, width))       # the latent a native start would draw

    model = solver.nn_model.double().eval()
    dev = next(model.parameters()).device
    qs = []
    with torch.no_grad():
        for i in range(0, args.n, args.chunk):
            c = np.zeros((min(args.chunk, args.n - i), 8))
            c[:, :7] = pose[i:i + args.chunk]
            out = model(torch.tensor(z[i:i + args.chunk], dtype=torch.float64, device=dev),
                        c=torch.tensor(c, dtype=torch.float64, device=dev), rev=True)[0]
            qs.append(out[:, :ndof].cpu().numpy())
    q_out = np.concatenate(qs)

    usable = (np.abs(q_out) < RUNAWAY_RAD).all(axis=1)
    fk = robot.forward_kinematics(torch.tensor(q_out[usable], dtype=torch.float64))
    fk = fk.detach().cpu().numpy() if hasattr(fk, "detach") else np.asarray(fk)
    err_mm = np.linalg.norm(fk[:, :3] - pose[usable][:, :3], axis=1) * 1000.0

    res = {"checkpoint": os.path.basename(ckpt), "nb_nodes": args.nb_nodes,
           "n": args.n, "seed": args.seed,
           "usable_fraction": float(usable.mean()), "n_usable": int(usable.sum()),
           "pos_err_mm/median": float(np.median(err_mm)),
           "pos_err_mm/p90": float(np.percentile(err_mm, 90)),
           "pos_err_mm/p99": float(np.percentile(err_mm, 99))}
    for k, v in res.items():
        print(f"{k}: {v}")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
