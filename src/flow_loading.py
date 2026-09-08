"""Load an IKFlow chart together with the architecture it was trained at.

Why this module exists
----------------------
`IKFlowSolver.load_state_dict` is a bare `pickle.load` of a raw `nn_model` state dict
(`ikflow_solver.py`), so a `.pkl` carries **weights only**. The architecture has to be
supplied by the caller, and until now it lived in hardcoded dicts at four call sites
(`src/iiwa_program.py`, `src/panda_program.py`, `scripts/training/pole_metric.py`,
`scripts/training/export_ckpt_to_pkl.py`), all of them asserting `nb_nodes = 12`,
`coeff_fn_internal_size = 1024`, `rnvp_clamp = 2.5`.

That is fine while every checkpoint shares one architecture and fatal as soon as they do
not. `rnvp_clamp` and `nb_nodes` change the forward pass **without changing any parameter
shape**, so a mismatched value loads cleanly and silently returns a different chart.

So each checkpoint now carries a sidecar, `<checkpoint>.arch.json`, written by
`scripts/training/export_ckpt_to_pkl.py` from the training checkpoint's own
`hyper_parameters.base_hparams`. `LoadFlowSolver` reads it, builds the solver, and then
**cross-checks the declared architecture against the state dict's actual shapes**.

What the cross-check can and cannot catch
-----------------------------------------
The state dict is structured as

    module_list.0                  FixedLinearTransform   (M, M_inv, b, logDetM)
    module_list.{1,3,5,...}        PermuteRandom          (perm, perm_inv)
    module_list.{2,4,6,...}        GLOWCouplingBlock      (subnet1.*, subnet2.*)

so `nb_nodes`, `coeff_fn_internal_size`, `coeff_fn_config` and `ndim_tot` are all
recoverable from shapes and are verified. **`rnvp_clamp` is not recoverable** — it is a
scalar used in the forward pass and stored nowhere. That is precisely why the sidecar is
mandatory for new checkpoints rather than merely convenient.
"""

import json
import os
import pickle
import re
import warnings

from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
from jrl.robots import get_robot

# The architecture every pre-sidecar checkpoint in this repo was trained at
# (iiwa14__lemon-haze-7, iiwa14__ddp-r1). Used only as the fallback for a checkpoint with
# no sidecar, and still shape-verified afterwards.
LEGACY_IIWA_ARCH = {
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

# The Panda's baseline, from ikflow's `model_descriptions.yaml` entry for
# `panda__full__lp191_5.25m`. Note `dim_latent_space = 7` against the iiwa's 8: each robot's
# ladder holds its own baseline latent width fixed, which keeps the optimization problem the
# same shape across a ladder *and* means a checkpoint loaded against the wrong robot fails
# the shape check instead of loading silently.
LEGACY_PANDA_ARCH = dict(LEGACY_IIWA_ARCH, dim_latent_space=7, softflow_noise_scale=0.01)

# The fields that define the chart. Anything outside this set (loss weights, noise scales
# used only during training) does not change the forward pass and is carried in the
# sidecar for provenance without being enforced.
ARCH_FIELDS = tuple(LEGACY_IIWA_ARCH.keys())

LEGACY_ARCH_BY_ROBOT = {"iiwa14": LEGACY_IIWA_ARCH, "panda": LEGACY_PANDA_ARCH}


def SidecarPath(checkpoint):
    """The architecture sidecar that belongs to `checkpoint`."""
    return os.path.splitext(checkpoint)[0] + ".arch.json"


def ArchFromStateDict(state_dict):
    """Recover the structurally-visible architecture from a raw `nn_model` state dict.

    Returns a dict with `nb_nodes`, `coeff_fn_internal_size`, `coeff_fn_config` and
    `ndim_tot`. `rnvp_clamp` is deliberately absent -- it leaves no trace in the weights.
    """
    indices = sorted({int(m.group(1)) for k in state_dict
                      if (m := re.match(r"module_list\.(\d+)\.", k))})
    if not indices:
        raise ValueError("state dict has no 'module_list.*' keys -- not an IKFlow chart")

    # module 0 is the fixed [-1, 1] rescaling; each coupling block costs a permute + a glow.
    if indices[-1] % 2 != 0 or indices != list(range(indices[-1] + 1)):
        raise ValueError(f"unexpected module_list layout: indices {indices}")
    nb_nodes = indices[-1] // 2

    linears = sorted(int(m.group(1)) for k in state_dict
                     if (m := re.match(r"module_list\.2\.subnet1\.(\d+)\.weight", k)))
    if not linears:
        raise ValueError("no 'module_list.2.subnet1.*.weight' keys -- unexpected block type")

    return {
        "nb_nodes": nb_nodes,
        # subnet is Linear/ReLU stacked, so `coeff_fn_config` hidden layers => that many + 1 Linears.
        "coeff_fn_config": len(linears) - 1,
        "coeff_fn_internal_size": int(state_dict[f"module_list.2.subnet1.{linears[0]}.weight"].shape[0]),
        "ndim_tot": int(state_dict["module_list.0.M"].shape[0]),
    }


def ReadArch(checkpoint, fallback=None):
    """Read a checkpoint's architecture sidecar, or fall back with a warning.

    `fallback` defaults to the legacy iiwa architecture, which is what every checkpoint
    predating the sidecar was trained at.
    """
    sidecar = SidecarPath(checkpoint)
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            arch = json.load(f)
        return {k: v for k, v in arch.items() if k in ARCH_FIELDS}, sidecar

    fallback = dict(LEGACY_IIWA_ARCH if fallback is None else fallback)
    warnings.warn(
        f"no architecture sidecar at {sidecar}; assuming the legacy architecture "
        f"(nb_nodes={fallback['nb_nodes']}, coeff_fn_internal_size="
        f"{fallback['coeff_fn_internal_size']}, rnvp_clamp={fallback['rnvp_clamp']}). "
        "rnvp_clamp cannot be verified against the weights -- if this checkpoint was "
        "trained at a different clamp it will load silently and be wrong. Export it with "
        "scripts/training/export_ckpt_to_pkl.py to get a sidecar.",
        RuntimeWarning, stacklevel=2)
    return fallback, None


def VerifyArch(declared, state_dict, source):
    """Raise if `declared` disagrees with what the weights structurally imply."""
    actual = ArchFromStateDict(state_dict)
    mismatches = [f"{k}: sidecar says {declared[k]!r}, weights imply {actual[k]!r}"
                  for k in ("nb_nodes", "coeff_fn_config", "coeff_fn_internal_size")
                  if k in declared and declared[k] != actual[k]]

    # ndim_tot is max(ndof, dim_latent_space) inside ikflow, so it bounds rather than
    # equals dim_latent_space -- check the direction that is actually implied.
    if "dim_latent_space" in declared and declared["dim_latent_space"] > actual["ndim_tot"]:
        mismatches.append(f"dim_latent_space: sidecar says {declared['dim_latent_space']!r}, "
                          f"but the weights are only {actual['ndim_tot']} wide")

    if mismatches:
        raise ValueError(
            f"architecture in {source or 'the fallback defaults'} does not match the "
            "weights:\n  " + "\n  ".join(mismatches) +
            "\nLoading this checkpoint would silently produce a different chart.")


def LoadFlowSolver(robot_name, checkpoint, fallback_arch=None, compile_model=None):
    """Build an `IKFlowSolver` at the architecture `checkpoint` was trained at.

    The sidecar is authoritative; the weights are then used to verify every part of it
    that leaves a structural trace. `rnvp_clamp` cannot be verified -- see the module
    docstring.
    """
    if fallback_arch is None:
        fallback_arch = LEGACY_ARCH_BY_ROBOT.get(robot_name, LEGACY_IIWA_ARCH)
    arch, source = ReadArch(checkpoint, fallback=fallback_arch)

    with open(checkpoint, "rb") as f:
        state_dict = pickle.load(f)
    VerifyArch(arch, state_dict, source)

    hyper_parameters = IkflowModelParameters()
    hyper_parameters.__dict__.update(arch)
    hyper_parameters.__dict__["robot_name"] = robot_name

    solver = IKFlowSolver(hyper_parameters, get_robot(robot_name), compile_model=compile_model)
    solver.load_state_dict(checkpoint)
    # Attach the resolved architecture so downstream code can *record* what it actually
    # loaded rather than what was requested. IKFlowSolver itself keeps no hparams.
    solver.arch = dict(arch, robot_name=robot_name)
    solver.arch_source = source
    return solver


def WriteArch(checkpoint, arch, provenance=None):
    """Write `<checkpoint>.arch.json`. `arch` should be a full `IkflowModelParameters.__dict__`."""
    payload = dict(arch)
    if provenance:
        payload["provenance"] = provenance
    sidecar = SidecarPath(checkpoint)
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return sidecar
