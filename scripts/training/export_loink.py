"""Export a LOInK training checkpoint (.ckpt) to the bare state-dict `.pkl` plus its sidecar.

The LOInK twin of `export_cvae.py`: a `.pkl` carries weights only, and the activation, the
cost letter, the head's standardisation (`j_mean`, `j_std`) and the inverse's iteration
counts (`n_dys`, `n_newton`) change the chart without changing any shape, so the architecture
travels in a mandatory `<out>.arch.json` (`"family": "loink"`) that
`src.charts.loink.LoadLoinkChart` cross-checks against the weights on every load.

The architecture is read from the checkpoint's own `hyper_parameters`, never from this
script's arguments -- with ONE exception, `--n_dys`: the inverse's iteration count is set
after training from `screen_loink.py` (b), and re-exporting with it is how the screen's
recommendation reaches the sidecar (recorded as `n_dys_source`). The default output name is
`models/<robot>/<robot>__loink<cost><act><recipe>__step<N>.pkl` (e.g. `loinkmrP`; no
underscore inside the label, so benchmark tags split on `_` stay unambiguous).

The round trip reloads the export through `LoadLoinkChart`, as the programs do, and requires,
in float64 on 256 samples, (1) its decode within 1e-10 of the checkpoint's own model and (2)
the inverse exact: `|G(decode(w)) - w|_inf < 1e-8`.

    python scripts/training/export_loink.py <run_dir>/checkpoints/last.ckpt
    python scripts/training/export_loink.py <ckpt> --out models/panda/<name>.pkl [--n_dys 50]

`--untrained` writes a randomly initialised chart in the same contract instead, for local
end-to-end plumbing (every model fit is a cluster job): `<robot>__loink<cost><act>P__untrained`
by default, sidecar provenance `untrained: true`, `j_mean = 0`, `j_std = 1` (no data). It says
nothing about solve quality.

    python scripts/training/export_loink.py --untrained --robot panda --cost m --activation relu
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
from src.charts.loink import (  # noqa: E402
    ACTIVATIONS, ARCH_FIELDS, COSTS, DEFAULT_ARCH, TASK_DIM, BuildLoinkModel, LoadLoinkChart,
    WriteLoinkArch)

PREFIX = "model."
RECIPES = ("P", "A")

#: The round trip's two gates (module docstring).
DECODE_TOL = 1e-10
INVERSE_TOL = 1e-8


def RungLabel(cost, activation, recipe):
    """`loink<cost><act><recipe>`, act the activation's initial: cluster/chart_runs.txt's labels."""
    return f"loink{cost}{activation[0]}{recipe}"


def DefaultOutPath(robot_name, cost, activation, recipe, step):
    return os.path.join(REPO, "models", robot_name,
                        f"{robot_name}__{RungLabel(cost, activation, recipe)}__step{step}.pkl")


def ArchFromHparams(hp, n_dys=None):
    ndof = int(hp["ndof"])
    return {"ndof": ndof, "task_dim": TASK_DIM, "dim_latent_space": ndof - TASK_DIM,
            "units": [int(u) for u in hp["units"]], "depth": int(hp["depth"]),
            "mu": float(hp["mu"]), "activation": hp["activation"], "cost": hp["cost"],
            "head_hidden": int(hp["head_hidden"]),
            "j_mean": None if hp["j_mean"] is None else float(hp["j_mean"]),
            "j_std": None if hp["j_std"] is None else float(hp["j_std"]),
            "n_dys": int(hp["n_dys"] if n_dys is None else n_dys),
            "n_newton": int(hp["n_newton"]), "robot_name": hp["robot_name"]}


def export(ckpt_path, out_path=None, n_dys=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    arch = ArchFromHparams(hp, n_dys)
    step = int(ckpt.get("global_step", -1))
    out_path = out_path or DefaultOutPath(arch["robot_name"], arch["cost"], arch["activation"],
                                          hp["recipe"], step)

    state_dict = {k[len(PREFIX):]: v.detach().cpu() for k, v in ckpt["state_dict"].items()
                  if k.startswith(PREFIX)}
    assert state_dict, f"no '{PREFIX}*' keys in {ckpt_path}"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"global_step {step}, {len(state_dict)} tensors -> {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")

    ## The training recipe and loss weights ride along for the record; only ARCH_FIELDS are
    ## enforced.
    recipe = {k: hp[k] for k in ("recipe", "optimizer", "learning_rate", "weight_decay",
                                 "scheduler", "gamma", "step_lr_every", "plateau_factor",
                                 "plateau_patience", "plateau_ema", "gradient_clip", "lam",
                                 "tau_penalty") if k in hp}
    sidecar = WriteLoinkArch(out_path, dict(
        arch, training=recipe,
        n_dys_source="training default" if n_dys is None else "export --n_dys (screen_loink (b))"),
        provenance={"source_checkpoint": os.path.abspath(ckpt_path), "global_step": step,
                    **_provenance_commit()})
    print(f"wrote {sidecar}  ({', '.join(f'{k}={arch[k]}' for k in ARCH_FIELDS)})")
    return out_path, arch, hp, state_dict


def roundtrip(out_path, arch, hp, state_dict, n=256, seed=0):
    """Reload through the programs' loader; compare its decode with the checkpoint's model and
    check the inverse is exact, float64, on `n` realistic inputs: the chart's own task
    estimate at configurations inside the limits (the raw `c6` the program hands the chart)
    and a latent from the prior."""
    chart = LoadLoinkChart(arch["robot_name"], out_path)
    reference = BuildLoinkModel(arch, hp["scale"])
    reference.load_state_dict(state_dict, strict=True)
    reference = reference.cpu().double().eval()

    loaded = chart.nn_model.double()
    dev = next(loaded.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = (torch.tensor(v, dtype=torch.float64, device="cpu")
              for v in zip(*chart.robot.actuated_joints_limits))
    q = lo + (hi - lo) * torch.rand((n, arch["ndof"]), generator=g, dtype=torch.float64,
                                    device="cpu")
    z = torch.randn((n, arch["dim_latent_space"]), generator=g, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        ## The raw task the chart assigns q (its own `invert`), as the paired start uses it.
        y = reference.forward_G(q)[0]
        q_ref = reference.decode(y, z)
        q_loaded, _ = chart.decode(loaded, z.to(dev), y.to(dev))
        inv_res = (reference.G(q_ref / reference.scale) - torch.cat([y, z], dim=1)).abs().amax(dim=1)
    assert torch.isfinite(q_ref).all(), "non-finite decode"
    err = (q_loaded.cpu() - q_ref).abs().max().item()
    assert err <= DECODE_TOL, f"exported decode differs from the checkpoint's by {err:.3e}"
    res = inv_res.max().item()
    assert res < INVERSE_TOL, (f"|G(decode(w)) - w| = {res:.3e} >= {INVERSE_TOL:g} at n_dys "
                               f"{arch['n_dys']}, n_newton {arch['n_newton']}: the inverse is not "
                               "exact (screen_loink.py (b) recommends an n_dys)")
    print(f"roundtrip OK through LoadLoinkChart: max |dq| = {err:.2e}, max |G(decode(w)) - w| "
          f"= {res:.2e} over {n} samples")


def write_untrained(robot_name, cost, activation, out_path=None, seed=0, recipe="P"):
    """A randomly initialised chart in the export contract's own format."""
    from jrl.robots import get_robot

    from src.charts.cvae import JointScale

    robot = get_robot(robot_name)
    scale = JointScale(robot)
    ndof = int(robot.ndof)
    arch = {"ndof": ndof, "task_dim": TASK_DIM, "dim_latent_space": ndof - TASK_DIM,
            "units": list(DEFAULT_ARCH["units"]), "depth": DEFAULT_ARCH["depth"],
            "mu": DEFAULT_ARCH["mu"], "activation": activation, "cost": cost,
            "head_hidden": DEFAULT_ARCH["head_hidden"],
            "j_mean": None if cost == "u" else 0.0, "j_std": None if cost == "u" else 1.0,
            "n_dys": DEFAULT_ARCH["n_dys"], "n_newton": DEFAULT_ARCH["n_newton"],
            "robot_name": robot_name}
    torch.manual_seed(seed)
    model = BuildLoinkModel(arch, scale)
    out_path = out_path or os.path.join(
        REPO, "models", robot_name,
        f"{robot_name}__{RungLabel(cost, activation, recipe)}__untrained.pkl")
    state_dict = {name: v.detach().cpu() for name, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)
    WriteLoinkArch(out_path, arch, provenance={
        "untrained": True, "seed": seed,
        "purpose": "pipeline smoke test; says nothing about solve quality",
        **_provenance_commit()})
    n_params = sum(v.numel() for v in model.parameters())
    print(f"wrote {out_path} (+ .arch.json): untrained, {cost}/{activation}, K={ndof - TASK_DIM}, "
          f"{n_params} parameters")
    return out_path, arch, {"scale": scale.tolist()}, state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_path", nargs="?", default=None)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--robot", default=None, help="--untrained only")
    parser.add_argument("--cost", default=None, choices=COSTS, help="--untrained only")
    parser.add_argument("--activation", default=None, choices=ACTIVATIONS,
                        help="--untrained only")
    parser.add_argument("--seed", type=int, default=0, help="--untrained only")
    parser.add_argument("--n_dys", type=int, default=None,
                        help="override the inverse's DYS iteration count (screen_loink (b)); "
                             "default: the checkpoint's")
    parser.add_argument("--out", default=None,
                        help="default models/<robot>/<robot>__loink<cost><act><recipe>__step<N>.pkl")
    parser.add_argument("--skip_roundtrip", action="store_true")
    args = parser.parse_args()
    if args.untrained:
        if args.ckpt_path or not (args.robot and args.cost and args.activation):
            parser.error("--untrained takes --robot, --cost and --activation, and no checkpoint")
        out_path, arch, hp, state_dict = write_untrained(args.robot, args.cost, args.activation,
                                                         args.out, seed=args.seed)
    else:
        if not args.ckpt_path:
            parser.error("a checkpoint is required (or --untrained)")
        out_path, arch, hp, state_dict = export(args.ckpt_path, args.out, n_dys=args.n_dys)
    if not args.skip_roundtrip:
        roundtrip(out_path, arch, hp, state_dict)
