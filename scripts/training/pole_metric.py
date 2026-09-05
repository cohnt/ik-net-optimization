"""Pole-fraction diagnostic for IKFlow checkpoints.

The acceptance metric for the iiwa14 retraining campaign: the fraction of the
conditioning domain the flow maps to runaway configurations (|q|_inf > 1000 rad).
Recorded baselines (N=20000, seed 0, float64, canonical quaternions):

    iiwa14  lemon-haze-7          frac_gt_1000 = 0.0334
    panda   lp191_5.25m           frac_gt_1000 = 0.00065

The sampler lives in the vendored fork (ikflow.training.pole_callback) so the
training-time callback and this script share one source of truth. The default
evaluation is batched; --crosscheck verifies the batched path against the batch-1
MakeFlowInference reference (the code path the optimization programs use).

Usage:
    python scripts/training/pole_metric.py --robot iiwa14 --n 20000
    python scripts/training/pole_metric.py --robot panda --n 20000
    python scripts/training/pole_metric.py --robot iiwa14 --checkpoint path/to.pkl --nb_nodes 6
    python scripts/training/pole_metric.py --robot iiwa14 --crosscheck
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(REPO_ROOT)

from ikflow.training.pole_callback import pole_metrics, sample_conditioning_and_latents  # noqa: E402

DEFAULT_IIWA_CKPT = os.path.join(REPO_ROOT, "models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl")


def load_solver(robot: str, checkpoint: str, nb_nodes: int, dim_latent_space: int):
    from ikflow.model import IkflowModelParameters
    from ikflow.ikflow_solver import IKFlowSolver
    from jrl.robots import get_robot

    if robot == "panda":
        # The Panda checkpoint downloads itself; its architecture comes from
        # model_descriptions.yaml (dim_latent_space=7).
        from ikflow.model_loading import get_ik_solver

        solver, _ = get_ik_solver("panda__full__lp191_5.25m")
        return solver, 7, 7

    hparams = {
        "nb_nodes": nb_nodes,
        "dim_latent_space": dim_latent_space,
        "coeff_fn_config": 3,
        "coeff_fn_internal_size": 1024,
        "rnvp_clamp": 2.5,
        "robot_name": robot,
    }
    hyper = IkflowModelParameters()
    hyper.__dict__.update(hparams)
    solver = IKFlowSolver(hyper, get_robot(robot), compile_model=None)
    solver.load_state_dict(checkpoint)
    return solver, dim_latent_space, solver.robot.ndof


def crosscheck(nn_model, width: int, ndof: int, n: int = 100, seed: int = 0) -> float:
    """Max |rel diff| between the batched evaluation and the batch-1 MakeFlowInference
    reference (src/generic_program.py), on identical samples."""
    from src.generic_program import MakeFlowInference

    model = __import__("copy").deepcopy(nn_model).double().eval()
    dev = next(model.parameters()).device
    c_np, z_np = sample_conditioning_and_latents(n, width, seed=seed)

    with torch.no_grad():
        out_batched, _ = model(
            torch.tensor(z_np, dtype=torch.float64, device=dev),
            c=torch.tensor(c_np, dtype=torch.float64, device=dev),
            rev=True,
        )
    q_batched = out_batched[:, :ndof].cpu().numpy()

    flow = MakeFlowInference(model, width, ndof, dev)
    worst = 0.0
    for i in range(n):
        v = torch.tensor(
            np.concatenate([c_np[i, :7], z_np[i], np.zeros(ndof)]), dtype=torch.float64, device=dev
        )
        with torch.no_grad():
            q1, _ = flow(v)
        q1 = q1.cpu().numpy()
        rel = np.abs(q_batched[i] - q1) / np.maximum(np.abs(q1), 1e-12)
        worst = max(worst, float(rel.max()))
    return worst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="iiwa14", choices=["iiwa14", "panda", "iiwa7"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pkl state dict (iiwa default: the shipped lemon-haze-7)")
    parser.add_argument("--nb_nodes", type=int, default=12)
    parser.add_argument("--dim_latent_space", type=int, default=8)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crosscheck", action="store_true", help="Verify batched vs batch-1 MakeFlowInference agreement")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None and args.robot == "iiwa14":
        checkpoint = DEFAULT_IIWA_CKPT

    solver, width, ndof = load_solver(args.robot, checkpoint, args.nb_nodes, args.dim_latent_space)

    if args.crosscheck:
        worst = crosscheck(solver.nn_model, width, ndof, n=100, seed=args.seed)
        print(f"crosscheck: max rel diff batched vs batch-1 reference = {worst:.3e}")
        assert worst < 1e-9, "batched path disagrees with the batch-1 reference"
        print("PASS")

    metrics = pole_metrics(solver.nn_model, width=width, ndof=ndof, n=args.n, seed=args.seed)
    metrics["robot"] = args.robot
    metrics["checkpoint"] = checkpoint or "downloaded"
    for k, v in metrics.items():
        print(f"{k}: {v}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=1)
