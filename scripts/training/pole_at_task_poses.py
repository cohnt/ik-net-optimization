"""Pole exposure at the conditioning poses the TASKS actually produce.

`scripts/training/pole_metric.py` draws the conditioning pose uniformly from a 0.5 m box
around [0.4, 0, 0.5] with `RollPitchYaw(uniform(-pi, pi, 3))`. That is the domain every
recorded baseline was measured on, so it stays the comparable number -- but it is NOT the
distribution either benchmark task presents to the flow. Both tasks generate targets by
sampling a collision-free `q` and taking `FK(q)`, and under `--start native` the learned
arm conditions on exactly that pose. Those poses span the whole reachable workspace.

The gap is large and it matters for acceptance decisions (measured 2026-09-07, N=20000):

    network                        box domain   task poses
    iiwa14 ddp-r1 step620000        0.0125%       0.67%      (51x)
    iiwa14 lemon-haze-7             3.34%         6.14%      (1.8x)
    iiwa14 elated-firefly-11 (6b)   0.0%          0.0%

So a chart can look far cleaner than it is where the task operates. Report BOTH columns.

Two traps, both of which were bugs in the first draft of this script:
  * the flow must be evaluated with `rev=True` (latent -> configuration);
  * `pole_metric.load_solver` returns `(solver, width, ndof)` in that order.

Usage:
    python scripts/training/pole_at_task_poses.py --checkpoint <pkl> --n 20000
    python scripts/training/pole_at_task_poses.py --checkpoint <pkl> --nb_nodes 6 --z ball
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="iiwa14 .pkl; default is whatever src/iiwa_program.py loads")
    p.add_argument("--nb_nodes", type=int, default=12,
                   help="coupling blocks in the checkpoint. A wrong value changes the "
                        "forward pass without changing any parameter shape.")
    p.add_argument("--dim_latent_space", type=int, default=8)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z", choices=["prior", "ball"], default="ball",
                   help="prior = N(0,I), what a native start draws; ball = uniform in the "
                        "radius-4.3 ball, what pole_metric.py draws (use this to compare "
                        "against the box column, since it holds the latent distribution "
                        "fixed and varies only the conditioning pose)")
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
    # Exactly how both tasks build targets: a random configuration, then its FK.
    q_target = rng.uniform(lo, hi, size=(args.n, ndof))
    pose = robot.forward_kinematics(torch.tensor(q_target, dtype=torch.float64))
    pose = pose.detach().cpu().numpy() if hasattr(pose, "detach") else np.asarray(pose)

    if args.z == "prior":
        z = rng.normal(size=(args.n, width))
    else:
        g = rng.normal(size=(args.n, width))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        z = g * 4.3 * rng.uniform(size=(args.n, 1)) ** (1.0 / width)

    model = solver.nn_model.double().eval()
    dev = next(model.parameters()).device
    chunks = []
    with torch.no_grad():
        for i in range(0, args.n, args.chunk):
            c = np.zeros((min(args.chunk, args.n - i), 8))
            c[:, :7] = pose[i:i + args.chunk]
            c[:, 7] = 0.0                      # softflow noise column, zero at test time
            out = model(torch.tensor(z[i:i + args.chunk], dtype=torch.float64, device=dev),
                        c=torch.tensor(c, dtype=torch.float64, device=dev), rev=True)[0]
            chunks.append(out[:, :ndof].abs().amax(dim=1).cpu().numpy())
    qinf = np.concatenate(chunks)

    res = {"checkpoint": os.path.basename(ckpt), "nb_nodes": args.nb_nodes,
           "n": args.n, "seed": args.seed, "z": args.z, "domain": "task_poses",
           "pole/frac_gt_1000": float((qinf > 1000).mean()),
           "pole/count_gt_1000": int((qinf > 1000).sum()),
           "pole/frac_gt_3": float((qinf > 3).mean()),
           "pole/p50": float(np.median(qinf)),
           "pole/p99": float(np.percentile(qinf, 99)),
           "pole/max": float(qinf.max())}
    for k, v in res.items():
        print(f"{k}: {v}")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
