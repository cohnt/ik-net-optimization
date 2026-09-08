"""Write `<checkpoint>.arch.json` sidecars for the checkpoints that predate them.

`src.flow_loading.LoadFlowSolver` falls back to the legacy architecture with a warning
when a checkpoint has no sidecar. The fallback exists so archived runs stay reproducible,
but nothing in the reduced-chart campaign should *depend* on it -- a warning is easy to
miss in a sharded cluster log. So the three checkpoints already on disk get real sidecars
here, each recording how its architecture was established:

- `iiwa14__ddp-r1__step620000.pkl`  -- exact, read from the training checkpoint's own
  `hyper_parameters.base_hparams`.
- `iiwa14__lemon-haze-7__global_step_4.25M.pkl` -- Julia's original chart. No training
  checkpoint survives, so the architecture is the legacy one this repo has always assumed,
  cross-checked against the weights (`nb_nodes=12`, width 1024). `rnvp_clamp = 2.5` is
  unverifiable from weights but is corroborated by the clamp sweep in CLAUDE.md, which
  found a clear optimum at 2.5 for these weights -- the wrong clamp would not have.
- `iiwa14__elated-firefly-11.pkl` -- the 6-block control of unknown provenance. `nb_nodes`
  is read from the weights (6, not the filename's say-so); everything else is assumed legacy.

Idempotent: re-running overwrites with the same content. Run from the repo root.
"""

import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.flow_loading import (ARCH_FIELDS, ArchFromStateDict, LEGACY_IIWA_ARCH,  # noqa: E402
                              SidecarPath, WriteArch)
from src.utils import RepoDir  # noqa: E402


def _legacy_with_measured_shape(pkl_path, note):
    """Legacy architecture, but with every structurally-visible field taken from the weights."""
    with open(pkl_path, "rb") as f:
        measured = ArchFromStateDict(pickle.load(f))
    arch = dict(LEGACY_IIWA_ARCH)
    for key in ("nb_nodes", "coeff_fn_config", "coeff_fn_internal_size"):
        arch[key] = measured[key]
    arch["dim_latent_space"] = measured["ndim_tot"]
    arch["robot_name"] = "iiwa14"
    return arch, {"architecture_source": note,
                  "unverifiable_fields": ["rnvp_clamp", "softflow_noise_scale", "init_scale"]}


def main():
    root = RepoDir()
    written = []

    # 1. ddp-r1: exact, from its own training checkpoint.
    ddp_pkl = os.path.join(root, "models/iiwa14/iiwa14__ddp-r1__step620000.pkl")
    ddp_ckpt = os.path.join(root, "results/train/iiwa14_ddp_r1/keep/ikflow-checkpoint-step=620000.ckpt")
    if os.path.exists(ddp_pkl):
        if os.path.exists(ddp_ckpt):
            import torch
            ckpt = torch.load(ddp_ckpt, map_location="cpu", weights_only=False)
            arch = dict(vars(ckpt["hyper_parameters"]["base_hparams"]))
            arch["robot_name"] = "iiwa14"
            prov = {"architecture_source": "exact: hyper_parameters.base_hparams",
                    "source_checkpoint": ddp_ckpt,
                    "global_step": int(ckpt["global_step"])}
        else:
            arch, prov = _legacy_with_measured_shape(
                ddp_pkl, "legacy defaults + weight shapes (training checkpoint absent)")
        written.append(WriteArch(ddp_pkl, arch, provenance=prov))

    # 2. lemon-haze-7: legacy, corroborated by shapes.
    lh = os.path.join(root, "models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl")
    if os.path.exists(lh):
        arch, prov = _legacy_with_measured_shape(
            lh, "legacy defaults + weight shapes; original training run not available")
        written.append(WriteArch(lh, arch, provenance=prov))

    # 3. elated-firefly-11: nb_nodes comes from the weights, not the filename.
    ff = os.path.join(root, "results/train/controls/iiwa14__elated-firefly-11.pkl")
    if os.path.exists(ff):
        arch, prov = _legacy_with_measured_shape(
            ff, "legacy defaults + weight shapes; provenance unknown -- pilot use only")
        written.append(WriteArch(ff, arch, provenance=prov))

    for path in written:
        print(f"wrote {path}")
    if not written:
        print("no checkpoints found -- nothing to backfill", file=sys.stderr)
        return 1

    # Prove every sidecar loads through the real path.
    from src.flow_loading import LoadFlowSolver
    for sidecar in written:
        pkl = sidecar.replace(".arch.json", ".pkl")
        solver = LoadFlowSolver("iiwa14", pkl)
        print(f"verified {os.path.basename(pkl)}: network_width={solver.network_width}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
