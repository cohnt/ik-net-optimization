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

# The adopted chart, matching the defaults in chart_accuracy.py and pole_at_task_poses.py.
# (This used to point at lemon-haze-7 while its two siblings pointed here, so the three
# scripts silently measured different networks when run without --checkpoint.)
DEFAULT_IIWA_CKPT = os.path.join(REPO_ROOT, "models/iiwa14/iiwa14__ddp-r1__step620000.pkl")


## THE SCREENS' DOMAIN AND THRESHOLD ARE ROBOT-SHAPED, and were iiwa-shaped for their whole
## life: a conditioning box around [0.4, 0, 0.5], a radius-4.3 latent ball, and "runaway"
## defined as |q|_inf > 1000 RADIANS.  None of those three transfers to a robot whose
## coordinates are normalized strain in [-1, 1] and whose latent is 12 wide, so they are
## resolved per robot here.  The rigid arms keep their exact previous values, so archived
## screens stay comparable; the soft rungs get the same quantities in their own units --
## the threshold as the same MULTIPLE of the coordinate limit (1000 rad against an iiwa
## limit of ~2.9 rad is ~345x, so 345 against a normalized limit of 1), and the latent ball
## as sqrt(width) + 1.5, the convention the trust region already uses.
##
## CLAUDE.md is explicit that these screens are a SMOKE TEST and not a selection criterion.
## That is the reason to put them in the right units rather than to make them decisive: a
## screen reported in the wrong units is not a weak signal, it is a number that reads 0.000
## forever.
def ScreenDomain(robot):
    """`(position_base, position_slack, latent_radius, runaway_threshold)` for a robot."""
    import math

    from src.soft_arm.params import RUNGS

    if robot in RUNGS:
        spec = RUNGS[robot]
        return ((0.0, 0.0, 0.45), 0.25, round(math.sqrt(spec.ndof) + 1.5, 2), 345.0)
    return ((0.4, 0.0, 0.5), 0.25, 4.3, 1000.0)


def SoftRungNames():
    """The soft arm's rungs, imported lazily so this module stays cheap to import."""
    from src.soft_arm.params import RUNGS

    return sorted(RUNGS)


def load_solver(robot: str, checkpoint: str, nb_nodes: int = None, dim_latent_space: int = None):
    """Build a solver for `checkpoint`, taking its architecture from the `.arch.json`
    sidecar and cross-checking it against the weights (src/flow_loading.py).

    `nb_nodes` / `dim_latent_space` remain only as the fallback for a checkpoint with no
    sidecar; where a sidecar exists it wins. Passing `checkpoint=None` for the Panda uses
    ikflow's downloaded pretrained chart, whose architecture comes from
    model_descriptions.yaml.
    """
    from src.flow_loading import LEGACY_ARCH_BY_ROBOT, LEGACY_IIWA_ARCH, LoadFlowSolver

    if robot == "panda" and checkpoint is None:
        from ikflow.model_loading import get_ik_solver

        solver, _ = get_ik_solver("panda__full__lp191_5.25m")
        solver.arch = dict(LEGACY_ARCH_BY_ROBOT["panda"], robot_name="panda")
        solver.arch_source = "ikflow model_descriptions.yaml"
        return solver, solver.network_width, solver.robot.ndof

    fallback = dict(LEGACY_ARCH_BY_ROBOT.get(robot, LEGACY_IIWA_ARCH))
    if nb_nodes is not None:
        fallback["nb_nodes"] = nb_nodes
    if dim_latent_space is not None:
        fallback["dim_latent_space"] = dim_latent_space

    solver = LoadFlowSolver(robot, checkpoint, fallback_arch=fallback)
    return solver, solver.network_width, solver.robot.ndof


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
    parser.add_argument("--robot", type=str, default="iiwa14", choices=["iiwa14", "panda", "iiwa7"] + SoftRungNames())
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

    ## THIS SCREEN'S DOMAIN LIVES IN THE VENDORED FORK and is deliberately frozen by
    ## tests/test_pole_metric_and_export.py::test_sampler_stream_frozen: a conditioning box
    ## around [0.4, 0, 0.5] and a radius-4.3 latent ball, both iiwa-shaped. Run against a
    ## robot whose coordinates are normalized strain, it would report a runaway fraction of
    ## 0.000 forever -- not a weak signal but a meaningless one, and the project has paid
    ## for a screen that could not distinguish "nothing wrong" from "measuring nothing".
    ##
    ## So it declines, rather than editing the fork to take a domain it has never needed.
    ## The other two screens sample over the ROBOT'S OWN task poses and its own FK, so they
    ## are robot-agnostic and do run: pole_at_task_poses.py is the pole screen for this
    ## robot, and chart_accuracy.py the accuracy one. Exits 0 and writes the reason, so an
    ## export job that screens every robot does not fail on this one.
    if args.robot in SoftRungNames():
        payload = {"robot": args.robot, "skipped": True,
                   "reason": ("box-domain screen is iiwa-shaped (frozen sampler in the "
                              "vendored ikflow fork); use pole_at_task_poses.py, whose "
                              "domain is this robot's own task poses")}
        print(json.dumps(payload, indent=2))
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w") as handle:
                json.dump(payload, handle, indent=2)
        ## The script body runs directly under `__main__`, so this is an exit, not a
        ## return. Exit 0: an export job screens every robot and must not fail on the one
        ## whose domain this screen cannot express.
        raise SystemExit(0)

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
