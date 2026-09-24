"""Fit the learned forward model for one soft-arm rung, and report its accuracy.

Trained against the EXACT map, on uniformly drawn configurations -- the surrogate stands in
for a forward model we happen to have, so that the optimization can be measured with a
forward model we would not have on real hardware. That is the point of the axis: it is the
general mechanism, and it is a control that moves both arms.

Loss is chordal on rotation (mean squared error on the 3x3 entries, which is a proper
metric on SO(3) and does not care about the quaternion's sign ambiguity) plus mean squared
error on translation, weighted so a millimetre of position and a milliradian of rotation
cost about the same.

    python scripts/soft_arm/train_fk_surrogate.py --rung soft12 --steps 20000
"""

import argparse
import os
import subprocess
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.soft_arm import kinematics as K
from src.soft_arm.fk_surrogate import (BackboneFkSurrogate, Save, Screen, SurrogatePath,
                                       sixd_to_rotation)
from src.soft_arm.params import GetSpec, RUNGS
from src.utils import RepoDir


def quaternion_to_rotation(q):
    """`(..., 4)` wxyz -> `(..., 3, 3)`. The TARGET side, so no gradient flows through it."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rung", choices=sorted(RUNGS), default="soft12")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rotation-weight", type=float, default=0.1,
                   help="a radian of rotation against a metre of position; 0.1 makes a "
                        "milliradian and a tenth of a millimetre cost about the same")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-dtype", choices=("float32", "float64"), default="float32",
                   help="FIT in float32, SCREEN and ship in float64. The surrogate's own "
                        "error is the quantity of interest and lands around 1e-4 m at best, "
                        "four orders above float32's noise floor, so the precision buys "
                        "nothing during the fit -- and float64 on a consumer GPU is roughly "
                        "an order of magnitude slower, which is the whole difference between "
                        "a fit that converges and one that runs out of patience.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    spec = GetSpec(args.rung)
    device = args.device
    torch.manual_seed(args.seed)

    train_dtype = getattr(torch, args.train_dtype)
    surrogate = BackboneFkSurrogate(spec, width=args.width, depth=args.depth)
    surrogate.to(device=device, dtype=train_dtype).train()
    optimiser = torch.optim.AdamW(surrogate.parameters(), lr=args.lr, weight_decay=1e-6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.steps)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    started = time.time()
    for step in range(1, args.steps + 1):
        cfg = (torch.rand(args.batch, spec.ndof, generator=generator,
                          dtype=torch.float64) * 2 - 1).to(device)
        with torch.no_grad():
            ## The TARGET is always computed in float64, whatever the fit runs in: it is
            ## the exact map, and there is no reason to hand the network a noisy one.
            target = K.config_to_plant_q(cfg, spec, device=device).reshape(
                args.batch, spec.num_bodies, 7)
            target_rotation = quaternion_to_rotation(target[..., :4]).to(train_dtype)
            target_translation = target[..., 4:].to(train_dtype)

        raw = surrogate.net(cfg.to(train_dtype)).reshape(args.batch, spec.num_bodies, 9)
        rotation = sixd_to_rotation(raw[..., 3:])
        loss_rotation = ((rotation - target_rotation) ** 2).mean()
        loss_translation = ((raw[..., :3] - target_translation) ** 2).mean()
        loss = loss_translation + args.rotation_weight * loss_rotation

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f"  step {step:6d}/{args.steps}  loss {float(loss):.3e}  "
                  f"(pos {float(loss_translation):.3e}  rot {float(loss_rotation):.3e})",
                  flush=True)

    surrogate.eval()
    surrogate.to(device="cpu", dtype=torch.float64)
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    ## Screened on a DIFFERENT seed: the fit draws fresh configurations every step, so there
    ## is no held-out set in the usual sense, but the screen must still not be able to score
    ## the stream it was trained on.
    metrics = Screen(surrogate, spec, n=20000, seed=args.seed + 1000)
    metrics["train_steps"] = args.steps
    metrics["train_dtype"] = args.train_dtype
    metrics["width"] = args.width
    metrics["depth"] = args.depth
    metrics["train_seconds"] = round(time.time() - started, 1)

    path = args.out or SurrogatePath(spec, RepoDir())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=RepoDir()).decode().strip()
    except Exception:
        commit = None
    Save(surrogate, spec, path, metrics,
         provenance={"commit": commit, "seed": args.seed, "lr": args.lr,
                     "batch": args.batch, "rotation_weight": args.rotation_weight})
    print(f"\nwrote {os.path.relpath(path, RepoDir())}")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")
    print("\nThe number that matters is tip_mm/p99 against the 1e-3 m (1 mm) task gate:\n"
          "  a surrogate whose tail is comparable to the gate cannot be told apart from a\n"
          "  formulation that missed, which is why verify() re-measures on exact kinematics.")


if __name__ == "__main__":
    main()
