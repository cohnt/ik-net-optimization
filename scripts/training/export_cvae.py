"""Export a cVAE training checkpoint (.ckpt) to the bare state-dict `.pkl` plus its sidecar.

The cVAE twin of `export_ckpt_to_pkl.py`, and for the same reason: a `.pkl` carries weights
only, and the activation changes the forward pass without changing any shape, so the
architecture travels in a mandatory `<out>.arch.json` (`"family": "cvae"`) that
`src.charts.cvae.LoadCVAEChart` cross-checks against the weights on every load.

The architecture is read from the checkpoint's own `hyper_parameters` (the trainer's
`save_hyperparameters`), never from this script's arguments -- what is recorded is what was
trained. The default output name is `models/<robot>/<robot>__cvae<K>__step<N>.pkl`; the rung
token `cvae<K>` has no underscore, so benchmark tags split on `_` stay unambiguous.

The round trip reloads the export through `LoadCVAEChart`, exactly as the programs do, and
checks its decode against the Lightning checkpoint's own decoder on random inputs.

    python scripts/training/export_cvae.py <run_dir>/checkpoints/last.ckpt
    python scripts/training/export_cvae.py <ckpt> --out models/panda/<name>.pkl

`--untrained` writes a randomly initialised chart in the same contract instead, for local
end-to-end plumbing (every model fit is a cluster job): `<robot>__cvae<K>untrained__step0`,
never tracked (.gitignore), sidecar provenance `untrained: true`. It says nothing about
solve quality, as `scripts/gvs_arm/make_untrained_chart.py` says of the flow's.

    python scripts/training/export_cvae.py --untrained --robot panda --dim_latent_space 7
"""

import argparse
import os
import pickle
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "training"))

from export_ckpt_to_pkl import _provenance_commit  # noqa: E402
from src.charts.cvae import (  # noqa: E402
    ARCH_FIELDS, DECODER_VARIANCES, DEFAULT_DECODER_VARIANCE, POSE_DIM, CVAEModel, JointScale,
    LoadCVAEChart, WriteCVAEArch)

PREFIX = "model."


def RungLabel(k, decoder_variance):
    """`cvae<K>` (learned variance) or `cvaef<K>` (fixed): cluster/chart_runs.txt's labels."""
    return f"cvae{'f' if decoder_variance == 'fixed' else ''}{k}"


def DefaultOutPath(robot_name, k, step, decoder_variance=DEFAULT_DECODER_VARIANCE):
    return os.path.join(REPO, "models", robot_name,
                        f"{robot_name}__{RungLabel(k, decoder_variance)}__step{step}.pkl")


def ArchFromHparams(hp):
    ## A checkpoint from before the field existed trained the learned-variance variant.
    return {"ndof": int(hp["ndof"]), "dim_latent_space": int(hp["dim_latent_space"]),
            "hidden": int(hp["hidden"]), "depth": int(hp["depth"]),
            "activation": hp["activation"], "pose_dim": POSE_DIM,
            "decoder_variance": hp.get("decoder_variance", DEFAULT_DECODER_VARIANCE),
            "robot_name": hp["robot_name"]}


def export(ckpt_path, out_path=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    arch = ArchFromHparams(hp)
    step = int(ckpt.get("global_step", -1))
    out_path = out_path or DefaultOutPath(arch["robot_name"], arch["dim_latent_space"], step,
                                          arch["decoder_variance"])

    state_dict = {k[len(PREFIX):]: v.detach().cpu() for k, v in ckpt["state_dict"].items()
                  if k.startswith(PREFIX)}
    assert state_dict, f"no '{PREFIX}*' keys in {ckpt_path}"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"global_step {step}, {len(state_dict)} tensors -> {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")

    ## The training recipe rides along for the record; only ARCH_FIELDS are enforced.
    recipe = {k: hp[k] for k in ("learning_rate", "gamma", "step_lr_every", "weight_decay",
                                 "gradient_clip") if k in hp}
    sidecar = WriteCVAEArch(out_path, dict(arch, training=recipe), provenance={
        "source_checkpoint": os.path.abspath(ckpt_path),
        "global_step": step,
        **_provenance_commit(),
    })
    print(f"wrote {sidecar}  ({', '.join(f'{k}={arch[k]}' for k in ARCH_FIELDS)})")
    return out_path, arch, hp, state_dict


def roundtrip(out_path, arch, hp, state_dict, n=256, seed=0):
    """Reload through the programs' loader and compare its decode with the checkpoint's."""
    chart = LoadCVAEChart(arch["robot_name"], out_path)
    reference = CVAEModel(arch["ndof"], arch["dim_latent_space"], hp["scale"],
                          hidden=arch["hidden"], depth=arch["depth"],
                          activation=arch["activation"],
                          decoder_variance=arch["decoder_variance"])
    reference.load_state_dict(state_dict, strict=True)
    ## Explicit devices throughout: importing jrl (via the loader) made CUDA torch's default.
    reference = reference.cpu().double().eval()

    loaded = chart.nn_model.double()
    dev = next(loaded.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, arch["dim_latent_space"], generator=g, dtype=torch.float64, device="cpu")
    pose = torch.randn(n, POSE_DIM, generator=g, dtype=torch.float64, device="cpu")
    pose[:, 3:] = pose[:, 3:] / pose[:, 3:].norm(dim=1, keepdim=True)
    with torch.no_grad():
        q_ref = reference(z, pose)
        q_loaded, _ = chart.decode(loaded, z.to(dev), pose.to(dev))
    err = (q_loaded.cpu() - q_ref).abs().max().item()
    assert torch.isfinite(q_ref).all(), "non-finite decode"
    assert err <= 1e-10, f"exported decode differs from the checkpoint's by {err:.3e}"
    print(f"roundtrip OK through LoadCVAEChart: max |dq| = {err:.2e} over {n} samples")


def write_untrained(robot_name, k, out_path=None, seed=0, hidden=512, depth=6,
                    activation="silu", decoder_variance=DEFAULT_DECODER_VARIANCE):
    """A randomly initialised chart in the export contract's own format."""
    from jrl.robots import get_robot

    robot = get_robot(robot_name)
    scale = JointScale(robot)
    ## Built on the CPU so the draw is the same with or without a GPU (jrl made CUDA the
    ## default device on import).
    torch.manual_seed(seed)
    with torch.device("cpu"):
        model = CVAEModel(robot.ndof, k, scale, hidden=hidden, depth=depth,
                          activation=activation, decoder_variance=decoder_variance)
    arch = dict(model.arch, robot_name=robot_name)
    out_path = out_path or os.path.join(
        REPO, "models", robot_name,
        f"{robot_name}__{RungLabel(k, decoder_variance)}untrained__step0.pkl")
    state_dict = {name: v.detach().cpu() for name, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)
    WriteCVAEArch(out_path, arch, provenance={
        "untrained": True, "seed": seed,
        "purpose": "pipeline smoke test; says nothing about solve quality",
        **_provenance_commit()})
    n_params = sum(v.numel() for v in model.parameters())
    print(f"wrote {out_path} (+ .arch.json): untrained, K={k}, {n_params} parameters")
    return out_path, arch, {"scale": scale.tolist()}, state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_path", nargs="?", default=None)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--robot", default=None, help="--untrained only")
    parser.add_argument("--dim_latent_space", type=int, default=None, help="--untrained only")
    parser.add_argument("--seed", type=int, default=0, help="--untrained only")
    parser.add_argument("--decoder_variance", default=DEFAULT_DECODER_VARIANCE,
                        choices=DECODER_VARIANCES, help="--untrained only (a trained "
                        "checkpoint's own hyper_parameters say which it is)")
    parser.add_argument("--out", default=None,
                        help="default models/<robot>/<robot>__cvae<K>__step<N>.pkl")
    parser.add_argument("--skip_roundtrip", action="store_true")
    args = parser.parse_args()
    if args.untrained:
        if args.ckpt_path or not args.robot or not args.dim_latent_space:
            parser.error("--untrained takes --robot and --dim_latent_space, and no checkpoint")
        out_path, arch, hp, state_dict = write_untrained(
            args.robot, args.dim_latent_space, args.out, seed=args.seed,
            decoder_variance=args.decoder_variance)
    else:
        if not args.ckpt_path:
            parser.error("a checkpoint is required (or --untrained)")
        out_path, arch, hp, state_dict = export(args.ckpt_path, args.out)
    if not args.skip_roundtrip:
        roundtrip(out_path, arch, hp, state_dict)
