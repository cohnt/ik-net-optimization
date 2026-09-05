"""Export a Lightning training checkpoint (.ckpt) to the bare-pickle state dict (.pkl)
that this repo's programs load (IKFlowSolver.load_state_dict = pickle.load).

Guards the silent-load hazard: src/iiwa_program.py builds IkflowModelParameters from a
partial dict, so any hparam not in that dict silently keeps the upstream default. This
script asserts the checkpoint's hyperparameters match every value the loader assumes,
then round-trips the exported .pkl through the loader's own construction path.

Usage:
    python scripts/training/export_ckpt_to_pkl.py <run_dir>/checkpoints/last.ckpt \
        models/iiwa14/iiwa14__<name>__global_step_<N>.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# What src/iiwa_program.py:76-90 assumes (nb_nodes..robot_name explicitly; the rest are
# upstream defaults it silently inherits). A checkpoint differing in ANY of these would
# load with matching tensor shapes and wrong behaviour, or break the program's hardcoded
# z-guess width.
LOADER_ASSUMPTIONS = {
    "nb_nodes": 12,
    "dim_latent_space": 8,
    "coeff_fn_config": 3,
    "coeff_fn_internal_size": 1024,
    "rnvp_clamp": 2.5,
    "softflow_enabled": True,
    "softflow_noise_scale": 0.001,
    "sigmoid_on_output": False,
    "coupling_layer": "glow",
    "permute_random_enabled": True,
}


def export(ckpt_path: str, out_path: str, skip_hparam_check: bool = False) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]["base_hparams"]

    if not skip_hparam_check:
        for key, expected in LOADER_ASSUMPTIONS.items():
            actual = getattr(hp, key)
            assert actual == expected, (
                f"hparam mismatch: checkpoint has {key}={actual!r} but the repo loader assumes "
                f"{expected!r}. Loading this checkpoint through src/iiwa_program.py would be "
                "silently wrong. Update the loader (and this table) deliberately, or re-export "
                "with --skip_hparam_check if you know what you are doing."
            )

    prefix = "nn_model."
    state_dict = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    assert state_dict, f"no '{prefix}*' keys in {ckpt_path}"
    print(f"global_step: {ckpt['global_step']}, tensors: {len(state_dict)}")

    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def roundtrip(out_path: str) -> None:
    """Load the exported .pkl exactly the way src/iiwa_program.py does and run one
    forward pass."""
    from ikflow.model import IkflowModelParameters
    from ikflow.ikflow_solver import IKFlowSolver
    from jrl.robots import get_robot

    hparams = {
        "nb_nodes": 12,
        "dim_latent_space": 8,
        "coeff_fn_config": 3,
        "coeff_fn_internal_size": 1024,
        "rnvp_clamp": 2.5,
        "robot_name": "iiwa14",
    }
    hyper = IkflowModelParameters()
    hyper.__dict__.update(hparams)
    solver = IKFlowSolver(hyper, get_robot("iiwa14"), compile_model=None)
    solver.load_state_dict(out_path)

    model = solver.nn_model.double().eval()
    dev = next(model.parameters()).device
    z = torch.zeros((1, 8), dtype=torch.float64, device=dev)
    c = torch.tensor([[0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64, device=dev)
    with torch.no_grad():
        q, _ = model(z, c=c, rev=True)
    q = q[0, :7].cpu().numpy()
    assert np.all(np.isfinite(q)), f"non-finite forward pass: {q}"
    print(f"roundtrip forward pass OK: q = {np.array2string(q, precision=3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--skip_hparam_check", action="store_true")
    parser.add_argument("--skip_roundtrip", action="store_true")
    args = parser.parse_args()

    export(args.ckpt_path, args.out_path, skip_hparam_check=args.skip_hparam_check)
    if not args.skip_roundtrip:
        roundtrip(args.out_path)
