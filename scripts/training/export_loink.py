"""Export a LOInK training checkpoint (.ckpt) to the bare state-dict `.pkl` plus its sidecar.

The LOInK twin of `export_cvae.py`: a `.pkl` carries weights only, and the activation, the
cost letter, the head's standardisation (`j_mean`, `j_std`) and the inverse's iteration
counts (`n_dys`, `n_newton`) change the chart without changing any shape, so the architecture
travels in a mandatory `<out>.arch.json` (`"family": "loink"`) that
`src.charts.loink.LoadLoinkChart` cross-checks against the weights on every load.

The architecture is read from the checkpoint's own `hyper_parameters`, never from this
script's arguments -- with ONE exception, the inverse's iteration counts `--n_dys` and
`--n_newton` (the DYS steps and the Newton polish after them), which change no weight. Without
`--n_dys` the exporter finds its own: it round-trips at the checkpoint's `n_dys` (100 by
default) and, while that fails, at each rung of `N_DYS_LADDER` = 200, 400, 800, 1600, 3200 above
it, printing every rung's residuals, and writes the SMALLEST passing count into the sidecar
(`n_dys_source = "export ladder (smallest passing of 100..3200)"`); if no rung passes it exits
non-zero after printing the whole ladder (the sidecar is left at the first rung). A chart
trained to a large distortion `tau` (1e3-1e5) needs more than 100 DYS steps to land in the
Newton polish's basin. An explicit `--n_dys` (e.g. `screen_loink.py` (b)'s recommendation)
is used as given, round-tripped once, no ladder (`n_dys_source = "export --n_dys"`).
`--n_newton` defaults to the checkpoint's. The default output name is
`models/<robot>/<robot>__loink<cost><act><recipe>__step<N>.pkl` (e.g. `loinkmrP`; no
underscore inside the label, so benchmark tags split on `_` stay unambiguous).

The round trip reloads the export through `LoadLoinkChart`, as the programs do, and requires,
in float64 on 256 samples, (1) its decode within 1e-10 of the checkpoint's own model and (2)
the inverse exact: `|G(decode(w)) - w|_inf < 1e-8`. `--skip_roundtrip` also skips the ladder
(the sidecar keeps the checkpoint's `n_dys`).

    python scripts/training/export_loink.py <run_dir>/checkpoints/last.ckpt
    python scripts/training/export_loink.py <ckpt> --out models/panda/<name>.pkl [--n_dys 50] [--n_newton 4]

`--untrained` writes a randomly initialised chart in the same contract instead, for local
end-to-end plumbing (every model fit is a cluster job): `<robot>__loink<cost><act>P__untrained`
by default, sidecar provenance `untrained: true`, `j_mean = 0`, `j_std = 1` (no data),
`n_dys` / `n_newton` the defaults (100 / 2) unless given, and the same ladder. It says
nothing about solve quality.

    python scripts/training/export_loink.py --untrained --robot panda --cost m --activation relu
"""

import argparse
import json
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
from src.flow_loading import SidecarPath  # noqa: E402

PREFIX = "model."
RECIPES = ("P", "A")

#: The round trip's two gates (module docstring).
DECODE_TOL = 1e-10
INVERSE_TOL = 1e-8

#: The DYS iteration counts the exporter tries, in order, above the checkpoint's own, when no
#: `--n_dys` is given (module docstring).
N_DYS_LADDER = (100, 200, 400, 800, 1600, 3200)


def RungLabel(cost, activation, recipe):
    """`loink<cost><act><recipe>`, act the activation's initial: cluster/chart_runs.txt's labels."""
    return f"loink{cost}{activation[0]}{recipe}"


def DefaultOutPath(robot_name, cost, activation, recipe, step):
    return os.path.join(REPO, "models", robot_name,
                        f"{robot_name}__{RungLabel(cost, activation, recipe)}__step{step}.pkl")


def ArchFromHparams(hp, n_dys=None, n_newton=None):
    ndof = int(hp["ndof"])
    return {"ndof": ndof, "task_dim": TASK_DIM, "dim_latent_space": ndof - TASK_DIM,
            "units": [int(u) for u in hp["units"]], "depth": int(hp["depth"]),
            "mu": float(hp["mu"]), "activation": hp["activation"], "cost": hp["cost"],
            "head_hidden": int(hp["head_hidden"]),
            "j_mean": None if hp["j_mean"] is None else float(hp["j_mean"]),
            "j_std": None if hp["j_std"] is None else float(hp["j_std"]),
            "n_dys": int(hp["n_dys"] if n_dys is None else n_dys),
            "n_newton": int(hp["n_newton"] if n_newton is None else n_newton),
            "robot_name": hp["robot_name"]}


def export(ckpt_path, out_path=None, n_dys=None, n_newton=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    arch = ArchFromHparams(hp, n_dys, n_newton)
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
        n_dys_source="training default" if n_dys is None else "export --n_dys"),
        provenance={"source_checkpoint": os.path.abspath(ckpt_path), "global_step": step,
                    **_provenance_commit()})
    print(f"wrote {sidecar}  ({', '.join(f'{k}={arch[k]}' for k in ARCH_FIELDS)})")
    return out_path, arch, hp, state_dict


def roundtrip_residuals(out_path, arch, hp, state_dict, n=256, seed=0):
    """Reload through the programs' loader; compare its decode with the checkpoint's model and
    measure the inverse, float64, on `n` realistic inputs: the chart's own task estimate at
    configurations inside the limits (the raw `c6` the program hands the chart) and a latent
    from the prior. The loaded chart reads `n_dys` / `n_newton` from the sidecar ON DISK, so
    that is what is checked; `arch` (which must agree with it) builds the reference.
    Returns `(finite, max |dq|, max |G(decode(w)) - w|)`."""
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
    finite = bool(torch.isfinite(q_ref).all())
    err = (q_loaded.cpu() - q_ref).abs().max().item()
    return finite, err, inv_res.max().item()


def _passes(finite, err, res):
    return finite and err <= DECODE_TOL and res < INVERSE_TOL


def roundtrip(out_path, arch, hp, state_dict, n=256, seed=0):
    """`roundtrip_residuals` at the sidecar's counts, asserting both gates."""
    finite, err, res = roundtrip_residuals(out_path, arch, hp, state_dict, n, seed)
    assert finite, "non-finite decode"
    assert err <= DECODE_TOL, f"exported decode differs from the checkpoint's by {err:.3e}"
    assert res < INVERSE_TOL, (f"|G(decode(w)) - w| = {res:.3e} >= {INVERSE_TOL:g} at n_dys "
                               f"{arch['n_dys']}, n_newton {arch['n_newton']}: the inverse is not "
                               "exact (screen_loink.py (b) recommends an n_dys)")
    print(f"roundtrip OK through LoadLoinkChart: max |dq| = {err:.2e}, max |G(decode(w)) - w| "
          f"= {res:.2e} over {n} samples")


def NDysLadder(first):
    """The checkpoint's own `n_dys` first, then every `N_DYS_LADDER` rung above it."""
    return [int(first)] + [r for r in N_DYS_LADDER if r > int(first)]


def _set_sidecar_inverse(out_path, n_dys, n_dys_source):
    """Rewrite `n_dys` / `n_dys_source` in the export's sidecar, every other field as written."""
    sidecar = SidecarPath(out_path)
    with open(sidecar) as f:
        payload = json.load(f)
    payload.update(n_dys=int(n_dys), n_dys_source=n_dys_source)
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def roundtrip_ladder(out_path, arch, hp, state_dict, ladder=None, n=256, seed=0):
    """Round-trip the export at each `n_dys` of `ladder` (default `NDysLadder(arch["n_dys"])`)
    in turn, each time through the sidecar on disk, and keep the SMALLEST that passes both
    gates: the sidecar is left at it (`n_dys_source` "export ladder ...") and the arch is
    returned. If none passes, the sidecar is put back at the first rung and an
    AssertionError lists every rung's residuals."""
    ladder = list(ladder or NDysLadder(arch["n_dys"]))
    source = f"export ladder (smallest passing of {ladder[0]}..{ladder[-1]})"
    rows = []
    for n_dys in ladder:
        rung = dict(arch, n_dys=int(n_dys))
        _set_sidecar_inverse(out_path, n_dys, source)
        finite, err, res = roundtrip_residuals(out_path, rung, hp, state_dict, n, seed)
        ok = _passes(finite, err, res)
        rows.append(f"n_dys {n_dys:5d}, n_newton {rung['n_newton']}: max |dq| = {err:.2e}, "
                    f"max |G(decode(w)) - w| = {res:.2e}{'' if finite else ' (non-finite)'}"
                    f" -> {'PASS' if ok else 'fail'}")
        print(f"ladder {rows[-1]}", flush=True)
        if ok:
            print(f"roundtrip OK through LoadLoinkChart at n_dys {n_dys} (the smallest passing "
                  f"rung) over {n} samples; sidecar n_dys = {n_dys}")
            return rung
    _set_sidecar_inverse(out_path, ladder[0],
                         f"training default (no export ladder rung of {ladder[0]}..{ladder[-1]} "
                         "passed)")
    raise AssertionError(
        f"no n_dys in {ladder} passes the round trip (decode <= {DECODE_TOL:g}, "
        f"|G(decode(w)) - w| < {INVERSE_TOL:g}) at n_newton {arch['n_newton']}:\n  "
        + "\n  ".join(rows))


def write_untrained(robot_name, cost, activation, out_path=None, seed=0, recipe="P",
                    n_dys=None, n_newton=None):
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
            "n_dys": int(DEFAULT_ARCH["n_dys"] if n_dys is None else n_dys),
            "n_newton": int(DEFAULT_ARCH["n_newton"] if n_newton is None else n_newton),
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
                        help="the inverse's DYS iteration count, used as given (screen_loink "
                             "(b)); default: the smallest passing rung of the export ladder, "
                             "starting at the checkpoint's")
    parser.add_argument("--n_newton", type=int, default=None,
                        help="the Newton polish steps after DYS; default: the checkpoint's")
    parser.add_argument("--out", default=None,
                        help="default models/<robot>/<robot>__loink<cost><act><recipe>__step<N>.pkl")
    parser.add_argument("--skip_roundtrip", action="store_true")
    args = parser.parse_args()
    if args.untrained:
        if args.ckpt_path or not (args.robot and args.cost and args.activation):
            parser.error("--untrained takes --robot, --cost and --activation, and no checkpoint")
        out_path, arch, hp, state_dict = write_untrained(args.robot, args.cost, args.activation,
                                                         args.out, seed=args.seed,
                                                         n_dys=args.n_dys, n_newton=args.n_newton)
    else:
        if not args.ckpt_path:
            parser.error("a checkpoint is required (or --untrained)")
        out_path, arch, hp, state_dict = export(args.ckpt_path, args.out, n_dys=args.n_dys,
                                                n_newton=args.n_newton)
    if not args.skip_roundtrip:
        if args.n_dys is not None:
            roundtrip(out_path, arch, hp, state_dict)
        else:
            roundtrip_ladder(out_path, arch, hp, state_dict)
