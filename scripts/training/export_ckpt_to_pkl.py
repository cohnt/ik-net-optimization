"""Export a Lightning training checkpoint (.ckpt) to the bare-pickle state dict (.pkl)
that this repo's programs load (IKFlowSolver.load_state_dict = pickle.load), together with
the `<out>.arch.json` sidecar recording the architecture it was trained at.

The sidecar is the point. A `.pkl` carries weights only, and `rnvp_clamp` / `nb_nodes`
change the forward pass without changing any parameter shape -- so a checkpoint loaded at
the wrong architecture is silently a different chart. Previously this script defended
against that by *asserting* the checkpoint matched one hardcoded architecture
(`nb_nodes = 12`, `coeff_fn_internal_size = 1024`, ...), which is exactly the check that
has to go once a ladder of architectures exists. Instead we now record what the
checkpoint actually is, and `src.flow_loading.LoadFlowSolver` cross-checks that record
against the weights on every load.

Usage:
    python scripts/training/export_ckpt_to_pkl.py <run_dir>/checkpoints/last.ckpt \
        models/iiwa14/iiwa14__<name>__step<N>.pkl
"""

import argparse
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.flow_loading import ARCH_FIELDS, LoadFlowSolver, SidecarPath, WriteArch  # noqa: E402


def _provenance_commit():
    """Which code produced this checkpoint.

    Prefers the ikflow fork's own commit, but the cluster copy is an rsync of the working
    tree with .git excluded, so `git rev-parse` finds nothing there. cluster/stage_code.sh
    writes the staged learned-ik commit to `.staged-commit` for exactly this reason, so
    fall back to it rather than recording null -- a checkpoint whose provenance is "null"
    is the case the sidecar exists to prevent.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    try:
        out = subprocess.run(["git", "-C", os.path.join(root, "third_party/ikflow"),
                              "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.stdout.strip():
            return {"ikflow_commit": out.stdout.strip()}
    except Exception:
        pass
    try:
        with open(os.path.join(root, ".staged-commit")) as f:
            return {"staged_learned_ik_commit": f.read().strip()}
    except Exception:
        return {"commit": None}


def export(ckpt_path: str, out_path: str, robot_name: str = None) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]["base_hparams"]
    arch = dict(vars(hp))

    # `robot_name` is not part of IkflowModelParameters; it comes from the training run's
    # own hyper_parameters, or from the caller for a checkpoint that predates it.
    robot = (robot_name or ckpt["hyper_parameters"].get("robot_name")
             or getattr(hp, "robot_name", None))
    assert robot, ("cannot determine robot_name from the checkpoint -- pass --robot_name")
    arch["robot_name"] = robot

    missing = [k for k in ARCH_FIELDS if k not in arch]
    assert not missing, f"checkpoint hyper_parameters are missing {missing}"

    prefix = "nn_model."
    state_dict = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    assert state_dict, f"no '{prefix}*' keys in {ckpt_path}"
    print(f"global_step: {ckpt['global_step']}, tensors: {len(state_dict)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    sidecar = WriteArch(out_path, arch, provenance={
        "source_checkpoint": os.path.abspath(ckpt_path),
        "global_step": int(ckpt.get("global_step", -1)),
        **_provenance_commit(),
    })
    print(f"wrote {sidecar}  (nb_nodes={arch['nb_nodes']}, "
          f"coeff_fn_internal_size={arch['coeff_fn_internal_size']}, "
          f"rnvp_clamp={arch['rnvp_clamp']}, dim_latent_space={arch['dim_latent_space']})")
    return arch


def roundtrip(out_path: str, robot_name: str) -> None:
    """Load the exported .pkl exactly the way the programs do -- through the shared loader,
    so the sidecar's architecture is cross-checked against the weights -- and run one
    forward pass."""
    solver = LoadFlowSolver(robot_name, out_path)

    model = solver.nn_model.double().eval()
    dev = next(model.parameters()).device
    width = solver.network_width
    z = torch.zeros((1, width), dtype=torch.float64, device=dev)
    c = torch.tensor([[0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64, device=dev)
    with torch.no_grad():
        q, _ = model(z, c=c, rev=True)
    q = q[0, :solver.robot.ndof].cpu().numpy()
    assert np.all(np.isfinite(q)), f"non-finite forward pass: {q}"
    print(f"roundtrip forward pass OK (width {width}): q = {np.array2string(q, precision=3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--robot_name", type=str, default=None,
                        help="Override/supply the robot the checkpoint was trained for")
    parser.add_argument("--skip_roundtrip", action="store_true")
    args = parser.parse_args()

    arch = export(args.ckpt_path, args.out_path, robot_name=args.robot_name)
    if not args.skip_roundtrip:
        roundtrip(args.out_path, arch["robot_name"])
