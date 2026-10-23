"""Export a LOInK training checkpoint (.ckpt) to the bare state-dict `.pkl` plus its sidecar.

The LOInK twin of `export_cvae.py`: a `.pkl` carries weights only, and the activation, the
cost letter, the head's standardisation (`j_mean`, `j_std`) and the inverse's iteration
counts and step rule (`n_dys`, `n_newton`, `newton_damping`) change the chart without changing
any shape, so the architecture travels in a mandatory `<out>.arch.json` (`"family": "loink"`)
that `src.charts.loink.LoadLoinkChart` cross-checks against the weights on every load.

The architecture is read from the checkpoint's own `hyper_parameters`, never from this
script's arguments -- with ONE exception, the inverse's settings `--n_dys`, `--n_newton` and
`--newton_damping` (the DYS steps, the Newton polish after them and its step rule), which
change no weight. Without any of them the exporter finds its own on a three-way ladder,
`N_DYS_LADDER` (25..3200) x `N_NEWTON_LADDER` (2..50) x `NEWTON_DAMPINGS` ("none",
"backtrack"): it writes the CHEAPEST passing triple into the sidecar under the cost model
`n_dys + (ndof + 1 + extra) n_newton`, `extra = len(STEP_SIZES)` under "backtrack" and 0 under
"none" (`InverseCost`; ties to "none", then to the smaller `n_newton`; `n_dys_source =
"export ladder (cheapest passing of n_dys x n_newton x newton_damping, ...)"`), printing every
triple it round-tripped; if none passes it exits non-zero after printing them all (the sidecar
is put back at the checkpoint's own settings). Why both step rules: the damped polish
(`src.charts.loink.damped_newton_step`) fixes a smooth (tanh) chart's rare cycling samples
(soft12 `mtP`: (25, 5, backtrack)), but stalls at the kinks of a piecewise-linear (relu)
chart, where the full step crosses them (soft12 `urP`: every backtrack pair through
(50, 50) fails at residual ~1e2). An explicit `--n_dys` / `--n_newton` / `--newton_damping`
is used as given (the others from the checkpoint), round-tripped once, no ladder
(`n_dys_source = "export --n_dys/--n_newton/--newton_damping"`). The default output name is
`models/<robot>/<robot>__loink<cost><act><recipe>__step<N>.pkl` (e.g. `loinkmrP`; no
underscore inside the label, so benchmark tags split on `_` stay unambiguous).

The round trip reloads the export through `LoadLoinkChart`, as the programs do, and requires,
in float64, (1) its decode within 1e-10 of the checkpoint's own model and (2) the inverse
exact: `|G(decode(w)) - w|_inf < 1e-8`. The inputs are `w = [G_1(q), z]` at 2048
configurations `q` uniform in the joint limits (`G_1(q)` the chart's own task estimate, the
raw `c6` the program hands it), each with THREE latents: `prior`, z ~ N(0, I) (the native
start's draw); `box`, z uniform in the program's latent box [-5, 5]^K (`LATENT_BOX`); and
`real`, z = G_2(q) (the paired start, which hands the decode `G(q_init)`). Why three
(measured 2026-10-23 on the six trained soft12 charts): the prior alone, the gate until then,
misses the program's domain -- real latents reach |z|_inf 13-25 on the m/x charts and
100-1100 on the u charts, and at the budgets it chose the decode failed on 15-49% of real
configurations (each an immediate `unrepresentable_start` from the paired start), while the
prior rarely reaches the box's corners, where two charts failed on 0.05-0.15% of samples. The
ladder gates each triple in two nested stages on ONE fixed draw of the 2048 configurations and
their latents: the first 256 of them, then, only if those pass, all 2048 (a failure on the
prefix is a failure on the whole, so nothing is lost). The prior and box draws are hard gates;
a chart with NO triple exact on the real draw (soft12 relu `mrP`, 2026-10-23) gets the
best-effort triple with the smallest fraction of failing real latents, recorded as
`real_latent_fail_fraction` (`roundtrip_ladder`). `--n_dys_max` / `--n_newton_max` cap the
ladder's two counts (default the full ladders). `--skip_roundtrip` also skips the ladder (the
sidecar keeps the checkpoint's settings).

    python scripts/training/export_loink.py <run_dir>/checkpoints/last.ckpt
    python scripts/training/export_loink.py <ckpt> --out models/panda/<name>.pkl [--n_dys 50] [--n_newton 4] [--newton_damping backtrack]

`--untrained` writes a randomly initialised chart in the same contract instead, for local
end-to-end plumbing (every model fit is a cluster job): `<robot>__loink<cost><act>P__untrained`
by default, sidecar provenance `untrained: true`, `j_mean = 0`, `j_std = 1` (no data),
`n_dys` / `n_newton` / `newton_damping` the defaults (100 / 2 / none) unless given, and the
same ladder. It says nothing about solve quality.

    python scripts/training/export_loink.py --untrained --robot panda --cost m --activation relu

`--reladder PKL` re-chooses the inverse's budget of an EXISTING export without its checkpoint
(e.g. one exported under an older gate): it reads the `.pkl` and its sidecar as
`LoadLoinkChart` does, runs the same ladder (or, given `--n_dys` / `--n_newton` /
`--newton_damping`, one round trip at those, the rest from the sidecar), and rewrites only the
sidecar's inverse fields (`n_dys`, `n_newton`, `newton_damping`, `n_dys_source`,
`inverse_ladder`, `real_latent_fail_fraction`) plus a `provenance.reladder` note (commit,
date, the triple it `replaced`); the weights, the rest of
the provenance and every other field are left as written. If the ladder or round trip fails,
the sidecar is restored byte for byte and the exporter exits non-zero.

    python scripts/training/export_loink.py --reladder models/soft12/<name>.pkl
"""

import argparse
import datetime
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
    ACTIVATIONS, ARCH_FIELDS, COSTS, DEFAULT_ARCH, NEWTON_DAMPINGS, STEP_SIZES, TASK_DIM,
    BuildLoinkModel, LoadLoinkChart, ReadLoinkArch, WriteLoinkArch)
from src.flow_loading import SidecarPath  # noqa: E402

PREFIX = "model."
RECIPES = ("P", "A")

#: The round trip's two gates (module docstring), and its sample count: 2048 configurations,
#: each decoded at the three latents of `LATENT_DRAWS` (so 3 x 2048 decodes), because the
#: undamped polish passed pairs on 256 samples that failed (residual 50-70) on 2048. The
#: ladder's first stage gates on the first ROUNDTRIP_PREFIX of the same draw.
DECODE_TOL = 1e-10
INVERSE_TOL = 1e-8
ROUNDTRIP_SAMPLES = 2048
ROUNDTRIP_PREFIX = 256

#: The round trip's three latent draws, all at the same configurations q and task y = G_1(q)
#: (`roundtrip_residuals`): the prior N(0, I) (the native start), uniform in the latent box
#: (anywhere a solve may walk), and the real latent G_2(q) (the paired start).
LATENT_DRAWS = ("prior", "box", "real")

#: The half-width of the programs' latent box. It MIRRORS the literal `-5 <= z <= 5` in
#: `IKFlowProgram.LatentBoxConstraint` (`src/generic_program.py`), copied rather than imported
#: because importing the program pulls in Drake; change both together.
LATENT_BOX = 5.0

#: The export ladder's three axes, searched when none of `--n_dys` / `--n_newton` /
#: `--newton_damping` is given (module docstring, `roundtrip_ladder`).
N_DYS_LADDER = (25, 50, 100, 200, 400, 800, 1600, 3200)
N_NEWTON_LADDER = (2, 5, 10, 20, 30, 50)
DAMPING_LADDER = NEWTON_DAMPINGS


def InverseCost(n_dys, n_newton, ndof, damping="none"):
    """The cost model of one decode, in network passes: a Davis-Yin iteration is ~one pass of
    the n-dim map; a Newton polish step needs the full n x n Jacobian of `G` (by `jacrev`, ~ndof
    + 1 passes) plus an n x n solve, and a damped one (`damping = "backtrack"`,
    `src.charts.loink.damped_newton_step`) also `G` at each of the `len(STEP_SIZES)` candidate
    steps (one batched call, counted as that many passes), so
    `n_dys + (ndof + 1 + extra) * n_newton`, `extra = len(STEP_SIZES)` under "backtrack", 0
    under "none". A model, not a measurement: it ranks the ladder's passing triples, nothing
    else."""
    if damping not in NEWTON_DAMPINGS:
        raise ValueError(f"newton damping {damping!r} not in {NEWTON_DAMPINGS}")
    extra = len(STEP_SIZES) if damping == "backtrack" else 0
    return int(n_dys) + (int(ndof) + 1 + extra) * int(n_newton)


LADDER_SOURCE = ("export ladder (cheapest passing of n_dys x n_newton x newton_damping, "
                 "cost = n_dys + (ndof+1+extra) n_newton, extra = len(STEP_SIZES) under "
                 "backtrack, 0 under none)")
EXPLICIT_SOURCE = "export --n_dys/--n_newton/--newton_damping"
BEST_EFFORT_SOURCE = (LADDER_SOURCE + "; best effort: no triple was exact on the real latents "
                      "z = G_2(q) (the paired start), so of those exact on the prior and box "
                      "draws, the one with the smallest real_latent_fail_fraction (ties to the "
                      "cheapest)")


def CappedLadders(n_dys_max=None, n_newton_max=None):
    """`roundtrip_ladder`'s `n_dys_ladder` / `n_newton_ladder` keywords with the counts above
    `n_dys_max` / `n_newton_max` removed (None: the full `N_DYS_LADDER` / `N_NEWTON_LADDER`), so
    a cluster run does not walk to (3200, 50) for a chart that cannot pass. Raises if a cap
    leaves a ladder empty."""
    ladders = {"n_dys_ladder": tuple(d for d in N_DYS_LADDER
                                     if n_dys_max is None or d <= int(n_dys_max)),
               "n_newton_ladder": tuple(nn for nn in N_NEWTON_LADDER
                                        if n_newton_max is None or nn <= int(n_newton_max))}
    for name, cap in (("n_dys_ladder", n_dys_max), ("n_newton_ladder", n_newton_max)):
        if not ladders[name]:
            raise ValueError(f"cap {cap} leaves {name} empty (smallest rung "
                             f"{(N_DYS_LADDER if name == 'n_dys_ladder' else N_NEWTON_LADDER)[0]})")
    return ladders


def RungLabel(cost, activation, recipe):
    """`loink<cost><act><recipe>`, act the activation's initial: cluster/chart_runs.txt's labels."""
    return f"loink{cost}{activation[0]}{recipe}"


def DefaultOutPath(robot_name, cost, activation, recipe, step):
    return os.path.join(REPO, "models", robot_name,
                        f"{robot_name}__{RungLabel(cost, activation, recipe)}__step{step}.pkl")


def ArchFromHparams(hp, n_dys=None, n_newton=None, newton_damping=None):
    ndof = int(hp["ndof"])
    return {"ndof": ndof, "task_dim": TASK_DIM, "dim_latent_space": ndof - TASK_DIM,
            "units": [int(u) for u in hp["units"]], "depth": int(hp["depth"]),
            "mu": float(hp["mu"]), "activation": hp["activation"], "cost": hp["cost"],
            "head_hidden": int(hp["head_hidden"]),
            "j_mean": None if hp["j_mean"] is None else float(hp["j_mean"]),
            "j_std": None if hp["j_std"] is None else float(hp["j_std"]),
            "n_dys": int(hp["n_dys"] if n_dys is None else n_dys),
            "n_newton": int(hp["n_newton"] if n_newton is None else n_newton),
            "newton_damping": (hp.get("newton_damping", "none") if newton_damping is None
                               else newton_damping),
            "robot_name": hp["robot_name"]}


def export(ckpt_path, out_path=None, n_dys=None, n_newton=None, newton_damping=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    arch = ArchFromHparams(hp, n_dys, n_newton, newton_damping)
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
        n_dys_source=("training default" if n_dys is None and n_newton is None
                      and newton_damping is None else EXPLICIT_SOURCE)),
        provenance={"source_checkpoint": os.path.abspath(ckpt_path), "global_step": step,
                    **_provenance_commit()})
    print(f"wrote {sidecar}  ({', '.join(f'{k}={arch[k]}' for k in ARCH_FIELDS)})")
    return out_path, arch, hp, state_dict


def roundtrip_residuals(out_path, arch, hp, state_dict, n=ROUNDTRIP_SAMPLES, seed=0,
                        n_eval=None, real_tol=INVERSE_TOL):
    """Reload through the programs' loader; compare its decode with the checkpoint's model and
    measure the inverse, float64, on the inputs the program hands the decode: the chart's own
    task estimate `y = G_1(q)` (the raw `c6`) at configurations `q` uniform inside the limits,
    each with the three latents of `LATENT_DRAWS` -- `prior` z ~ N(0, I) (the native start),
    `box` z uniform in [-LATENT_BOX, LATENT_BOX]^K (the program's latent box, whose corners the
    prior rarely reaches), `real` z = G_2(q) (the paired start sets the variables to
    G(q_init)). `n` configurations, then `n` prior latents, then `n` box latents are drawn from
    ONE generator (fixed `seed`) in that order and the FIRST `n_eval` of each (default all)
    evaluated, so a smaller `n_eval` is a prefix of the same samples (the ladder's first
    stage). The three draws are decoded as one batch. The loaded chart reads `n_dys` /
    `n_newton` / `newton_damping` from the sidecar ON DISK, so that is what is checked; `arch`
    (which must agree with it) builds the reference. Returns `(finite, max |dq|,
    max |G(decode(w)) - w|, per_draw, real_fail)`, both maxima over all three draws,
    `per_draw` the inverse residual's maximum on each, `{draw: max}` in `LATENT_DRAWS` order,
    and `real_fail` the fraction of the `n_eval` real-draw samples whose residual is NOT below
    `real_tol` (a non-finite one counts as failing; `real_tol` is `roundtrip_ladder`'s test
    hook)."""
    n_eval = n if n_eval is None else int(n_eval)
    assert 0 < n_eval <= n, (n_eval, n)
    chart = LoadLoinkChart(arch["robot_name"], out_path)
    reference = BuildLoinkModel(arch, hp["scale"])
    reference.load_state_dict(state_dict, strict=True)
    reference = reference.cpu().double().eval()

    loaded = chart.nn_model.double()
    dev = next(loaded.parameters()).device
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = (torch.tensor(v, dtype=torch.float64, device="cpu")
              for v in zip(*chart.robot.actuated_joints_limits))
    k = arch["dim_latent_space"]
    q = lo + (hi - lo) * torch.rand((n, arch["ndof"]), generator=g, dtype=torch.float64,
                                    device="cpu")
    z_prior = torch.randn((n, k), generator=g, dtype=torch.float64, device="cpu")
    z_box = LATENT_BOX * (2 * torch.rand((n, k), generator=g, dtype=torch.float64,
                                         device="cpu") - 1)
    q, z_prior, z_box = q[:n_eval], z_prior[:n_eval], z_box[:n_eval]
    with torch.no_grad():
        ## The raw task the chart assigns q (its own `invert`), as the paired start uses it,
        ## and the latent it assigns q, the paired start's.
        y, z_real = reference.forward_G(q)
        draws = dict(zip(LATENT_DRAWS, (z_prior, z_box, z_real)))
        y = y.repeat(len(draws), 1)
        z = torch.cat([draws[name] for name in LATENT_DRAWS], dim=0)
        q_ref = reference.decode(y, z)
        q_loaded, _ = chart.decode(loaded, z.to(dev), y.to(dev))
        inv_res = (reference.G(q_ref / reference.scale) - torch.cat([y, z], dim=1)).abs().amax(dim=1)
    finite = bool(torch.isfinite(q_ref).all())
    err = (q_loaded.cpu() - q_ref).abs().max().item()
    by_draw = dict(zip(LATENT_DRAWS, inv_res.split(n_eval)))
    per_draw = {name: r.max().item() for name, r in by_draw.items()}
    real_fail = (~(by_draw["real"] < real_tol)).double().mean().item()
    return finite, err, inv_res.max().item(), per_draw, real_fail


def _by_draw(per_draw):
    return ", ".join(f"{name} {per_draw[name]:.2e}" for name in LATENT_DRAWS)


def _passes(finite, err, res):
    """Both gates; `err` and `res` are maxima over the draws they are given."""
    return finite and err <= DECODE_TOL and res < INVERSE_TOL


def _passes_prior_box(finite, err, per_draw):
    """The ladder's HARD gates: decode and finiteness on every draw, the inverse on the draws
    the solver visits (`prior`, `box`) -- the real draw is gated separately."""
    return _passes(finite, err, max(per_draw["prior"], per_draw["box"]))


def _settings(arch):
    return (f"n_dys {arch['n_dys']}, n_newton {arch['n_newton']}, newton_damping "
            f"{arch.get('newton_damping', 'none')}")


def roundtrip(out_path, arch, hp, state_dict, n=ROUNDTRIP_SAMPLES, seed=0):
    """`roundtrip_residuals` at the sidecar's settings on all `n` samples (each at every
    latent draw), asserting both gates. Returns the per-draw residual maxima."""
    finite, err, res, per_draw, _ = roundtrip_residuals(out_path, arch, hp, state_dict, n, seed)
    assert finite, "non-finite decode"
    assert err <= DECODE_TOL, f"exported decode differs from the checkpoint's by {err:.3e}"
    assert res < INVERSE_TOL, (f"|G(decode(w)) - w| = {res:.3e} >= {INVERSE_TOL:g} at "
                               f"{_settings(arch)} (by latent draw: {_by_draw(per_draw)}): the "
                               "inverse is not exact (screen_loink.py (b) recommends an n_dys)")
    print(f"roundtrip OK through LoadLoinkChart: max |dq| = {err:.2e}, max |G(decode(w)) - w| "
          f"= {res:.2e} ({_by_draw(per_draw)}) over {n} samples x {len(LATENT_DRAWS)} latent "
          "draws")
    return per_draw


def _set_sidecar_inverse(out_path, n_dys, n_newton, newton_damping, n_dys_source, drop=(),
                         **extra):
    """Rewrite `n_dys` / `n_newton` / `newton_damping` / `n_dys_source` (and any `extra` record
    fields; any `drop` fields removed) in the export's sidecar, every other field as written."""
    sidecar = SidecarPath(out_path)
    with open(sidecar) as f:
        payload = json.load(f)
    for key in drop:
        payload.pop(key, None)
    payload.update(n_dys=int(n_dys), n_newton=int(n_newton), newton_damping=newton_damping,
                   n_dys_source=n_dys_source, **extra)
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def LadderOrder(n_dys_ladder, n_newton_ladder, ndof, damping_ladder=DAMPING_LADDER):
    """Every `(n_dys, n_newton, newton_damping)` triple of the three ladders, cheapest first
    (`InverseCost`), ties to "none" (the `NEWTON_DAMPINGS` order), then to the smaller
    `n_newton`. At `n_newton = 0` the step rule is moot, so only "none" is listed. Within one
    (`n_newton`, damping) this is `n_dys` ascending."""
    rank = {d: i for i, d in enumerate(NEWTON_DAMPINGS)}
    triples = sorted({(int(d), int(nn), dm) for dm in damping_ladder for nn in n_newton_ladder
                      for d in n_dys_ladder if int(nn) > 0 or dm == "none"},
                     key=lambda t: (InverseCost(t[0], t[1], ndof, t[2]), rank[t[2]], t[1], t[0]))
    return triples


def roundtrip_ladder(out_path, arch, hp, state_dict, n_dys_ladder=N_DYS_LADDER,
                     n_newton_ladder=N_NEWTON_LADDER, damping_ladder=DAMPING_LADDER,
                     n=ROUNDTRIP_SAMPLES, n_prefix=ROUNDTRIP_PREFIX, seed=0,
                     real_tol=INVERSE_TOL):
    """Find the cheapest `(n_dys, n_newton, newton_damping)` whose round trip passes both
    gates on all three latent draws, or failing that the best effort below, and leave the
    sidecar at it. Returns `(arch at the chosen triple, tried)`, `tried` a list of `(n_dys,
    n_newton, newton_damping, cost, samples, finite, max |dq|, max |G(decode(w)) - w|,
    per_draw, real_fail, passes_prior_box, passes)` in the order run (`samples` the stage that
    decided it; both maxima over all three latent draws of `roundtrip_residuals`, `per_draw`
    the residual's maximum on each, `real_fail` the real draw's failing fraction at that
    stage, `passes_prior_box` the hard gates, `passes` those AND the real draw exact).

    Triples are visited in increasing `InverseCost` (`LadderOrder`) and the search stops at the
    first triple passing on all three draws, so every triple cheaper than the choice was
    round-tripped and failed, and no triple dearer than it is spent. (A tanh chart can need
    `n_dys` 3200 at `n_newton` 2 but 50 at 10; walking the `n_newton = 2` row to 3200 first
    would cost minutes for a pair that cannot win.) Each triple is gated in two nested stages
    on one draw of `n` configurations and their three latents: the first `n_prefix` on the
    hard gates (decode and finiteness on every draw, the inverse on `prior` and `box`), then,
    only if those pass, all `n` on the hard gates and the real draw (same samples, so a prefix
    failure is a failure). Each is checked through the sidecar ON DISK.

    The best-effort rule. The prior and box draws are where the solver walks, so they stay
    hard gates; the real draw is only the paired start, whose failure the benchmark records
    per cell (`start_q_error`). So when NO triple is exact on the real draw, the ladder chooses,
    among the triples passing the hard gates on all `n` samples, the one with the smallest
    `real_fail` (ties to the cheapest, i.e. first in ladder order), with `n_dys_source =
    BEST_EFFORT_SOURCE`. The reason (2026-10-23, soft12 relu `mrP`, 2048 real latents): no
    triple of the ladder is exact on the real draw -- undamped Newton leaves 0.3% failing at
    (25, 40), 0.05% at (400, 50) with max residual 4, and backtracking stalls at the relu
    kinks (0.24%, max 2e3, and 0.4% of the box draw) -- while the prior and box draws pass at
    every undamped triple with `n_newton >= 20`. `real_tol` (default `INVERSE_TOL`) is the
    level the real draw is gated at, a TEST HOOK only (0.0 makes every triple fail it).

    The sidecar records the triples tried (`inverse_ladder`, each with its per-draw residuals
    `max_inverse_residual_by_draw`, `real_fail_fraction` and `passes_prior_box`) and
    `real_latent_fail_fraction` (the choice's; 0.0 when it is exact on all three). If no
    triple even passes the hard gates, the sidecar is put back at the checkpoint's own
    settings (`arch`) and an AssertionError lists every triple."""
    ndof = int(arch["ndof"])
    order = LadderOrder(n_dys_ladder, n_newton_ladder, ndof, damping_ladder)
    stages = [m for m in (min(int(n_prefix), n), n) if m > 0]
    stages = sorted(set(stages))
    tried, rows = [], []

    def finish(t, source):
        n_dys, n_newton, damping, cost = t[:4]
        record = [{"n_dys": d, "n_newton": nn, "newton_damping": dm, "cost": c,
                   "samples": ns, "finite": f, "max_dq": e, "max_inverse_residual": r,
                   "max_inverse_residual_by_draw": pd, "real_fail_fraction": rf,
                   "passes_prior_box": pb, "passes": p}
                  for d, nn, dm, c, ns, f, e, r, pd, rf, pb, p in tried]
        _set_sidecar_inverse(out_path, n_dys, n_newton, damping, source, inverse_ladder=record,
                             real_latent_fail_fraction=t[9])
        best_effort = source == BEST_EFFORT_SOURCE
        print(f"ladder CHOSEN n_dys {n_dys}, n_newton {n_newton}, newton_damping {damping} "
              f"(cost {cost}; " + (
                  f"BEST EFFORT: no triple of the {len(order)}-triple ladder is exact on the real "
                  f"latents; of those exact on prior and box, the smallest real-draw failing "
                  f"fraction, {100 * t[9]:.2f}%" if best_effort else
                  f"every cheaper triple of the {len(order)}-triple ladder failed") +
              f"): roundtrip through LoadLoinkChart over {n} samples x {len(LATENT_DRAWS)} "
              f"latent draws ({_by_draw(t[8])}); sidecar n_dys = {n_dys}, n_newton = "
              f"{n_newton}, newton_damping = {damping}", flush=True)
        return dict(arch, n_dys=n_dys, n_newton=n_newton, newton_damping=damping), tried

    for n_dys, n_newton, damping in order:
        rung = dict(arch, n_dys=n_dys, n_newton=n_newton, newton_damping=damping)
        _set_sidecar_inverse(out_path, n_dys, n_newton, damping, LADDER_SOURCE)
        for m in stages:
            finite, err, res, per_draw, real_fail = roundtrip_residuals(
                out_path, rung, hp, state_dict, n, seed, n_eval=m, real_tol=real_tol)
            hard = _passes_prior_box(finite, err, per_draw)
            if not hard:
                break
        ok = hard and m == n and real_fail == 0.0
        cost = InverseCost(n_dys, n_newton, ndof, damping)
        tried.append((n_dys, n_newton, damping, cost, m, finite, err, res, per_draw, real_fail,
                      hard, ok))
        rows.append(f"n_dys {n_dys:5d}, n_newton {n_newton:3d}, {damping:9s} (cost {cost:5d}), "
                    f"{m:4d} samples: max |dq| = {err:.2e}, max |G(decode(w)) - w| = {res:.2e}"
                    f" ({_by_draw(per_draw)}; real failing {100 * real_fail:.2f}%)"
                    f"{'' if finite else ' (non-finite)'} -> "
                    f"{'PASS' if ok else 'prior+box only' if hard and m == n else 'fail'}")
        print(f"ladder {rows[-1]}", flush=True)
        if ok:
            return finish(tried[-1], LADDER_SOURCE)
    ## No triple exact on all three draws: the best effort, if any passed the hard gates.
    candidates = [t for t in tried if t[10] and t[4] == n]
    if candidates:
        return finish(min(candidates, key=lambda t: t[9]), BEST_EFFORT_SOURCE)
    _set_sidecar_inverse(out_path, arch["n_dys"], arch["n_newton"],
                         arch.get("newton_damping", "none"),
                         "training default (no export ladder triple of n_dys "
                         f"{list(n_dys_ladder)} x n_newton {list(n_newton_ladder)} x "
                         f"newton_damping {list(damping_ladder)} passed)",
                         drop=("inverse_ladder", "real_latent_fail_fraction"))
    raise AssertionError(
        f"no (n_dys, n_newton, newton_damping) in {list(n_dys_ladder)} x {list(n_newton_ladder)}"
        f" x {list(damping_ladder)} passes the round trip's hard gates (decode <= "
        f"{DECODE_TOL:g}, |G(decode(w)) - w| < {INVERSE_TOL:g} on the prior and box draws):\n  "
        + "\n  ".join(rows))


def reladder(pkl_path, n_dys=None, n_newton=None, newton_damping=None, **ladder):
    """Re-choose the inverse's budget of an existing export (`--reladder`, module docstring).

    Builds exactly the objects `export()` hands the round trip -- `arch` the sidecar's
    `ARCH_FIELDS` (read by `ReadLoinkArch`, as `LoadLoinkChart` does) plus its `robot_name`,
    `state_dict` the `.pkl` itself (already stripped of `PREFIX`), `hp = {"scale": <its scale
    buffer>}` (all the round trip reads of `hp`) -- and runs `roundtrip_ladder` (`ladder` its
    keyword arguments), or `roundtrip` when any of `n_dys` / `n_newton` / `newton_damping` is
    given (the rest from the sidecar; a stale `inverse_ladder` / `real_latent_fail_fraction`
    is dropped). Only the inverse fields and a `provenance.reladder` note (commit, date, the
    triple it `replaced`) change; on any failure the sidecar is
    restored byte for byte and the exception re-raised. The `.pkl` is never written. Returns
    `(old triple, new triple)`."""
    payload, sidecar = ReadLoinkArch(pkl_path)
    if payload.get("robot_name") is None:
        raise ValueError(f"{sidecar} has no robot_name; cannot rebuild the round trip's robot")
    arch = dict({k: payload[k] for k in ARCH_FIELDS}, robot_name=payload["robot_name"])
    with open(pkl_path, "rb") as f:
        state_dict = pickle.load(f)
    hp = {"scale": state_dict["scale"].tolist()}
    old = (int(arch["n_dys"]), int(arch["n_newton"]), arch["newton_damping"])
    with open(sidecar) as f:
        original = f.read()
    explicit = n_dys is not None or n_newton is not None or newton_damping is not None
    try:
        if explicit:
            if ladder:
                raise ValueError(f"ladder options {sorted(ladder)} with an explicit triple")
            chosen = dict(arch, n_dys=int(old[0] if n_dys is None else n_dys),
                          n_newton=int(old[1] if n_newton is None else n_newton),
                          newton_damping=old[2] if newton_damping is None else newton_damping)
            _set_sidecar_inverse(pkl_path, chosen["n_dys"], chosen["n_newton"],
                                 chosen["newton_damping"], EXPLICIT_SOURCE,
                                 drop=("inverse_ladder", "real_latent_fail_fraction"))
            roundtrip(pkl_path, chosen, hp, state_dict)
        else:
            chosen, _ = roundtrip_ladder(pkl_path, arch, hp, state_dict, **ladder)
    except BaseException:
        with open(sidecar, "w") as f:
            f.write(original)
        print(f"reladder FAILED: {sidecar} restored as it was", flush=True)
        raise
    with open(sidecar) as f:
        payload = json.load(f)
    payload.setdefault("provenance", {})["reladder"] = {
        **_provenance_commit(), "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "replaced": dict(zip(("n_dys", "n_newton", "newton_damping"), old))}
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    new = (int(chosen["n_dys"]), int(chosen["n_newton"]), chosen["newton_damping"])
    print(f"reladder {pkl_path}: (n_dys, n_newton, newton_damping) {old} -> {new}"
          f"{' (unchanged)' if new == old else ''}; sidecar {sidecar}")
    return old, new


def write_untrained(robot_name, cost, activation, out_path=None, seed=0, recipe="P",
                    n_dys=None, n_newton=None, newton_damping=None):
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
            "newton_damping": (DEFAULT_ARCH["newton_damping"] if newton_damping is None
                               else newton_damping),
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
    source = ("training default" if n_dys is None and n_newton is None and newton_damping is None
              else EXPLICIT_SOURCE)
    WriteLoinkArch(out_path, dict(arch, n_dys_source=source), provenance={
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
    parser.add_argument("--reladder", default=None, metavar="PKL",
                        help="re-choose the inverse settings of an existing export's sidecar "
                             "(no checkpoint; the .pkl is not rewritten)")
    parser.add_argument("--robot", default=None, help="--untrained only")
    parser.add_argument("--cost", default=None, choices=COSTS, help="--untrained only")
    parser.add_argument("--activation", default=None, choices=ACTIVATIONS,
                        help="--untrained only")
    parser.add_argument("--seed", type=int, default=0, help="--untrained only")
    parser.add_argument("--n_dys", type=int, default=None,
                        help="the inverse's DYS iteration count, used as given (screen_loink "
                             "(b)); with neither --n_dys nor --n_newton: the cheapest passing "
                             "pair of the export ladder; with only --n_newton: the checkpoint's")
    parser.add_argument("--n_newton", type=int, default=None,
                        help="the Newton polish steps after DYS, used as given; with only "
                             "--n_dys: the checkpoint's")
    parser.add_argument("--newton_damping", default=None, choices=NEWTON_DAMPINGS,
                        help="the polish's step rule, used as given; with neither this nor "
                             "--n_dys / --n_newton: the ladder's; otherwise the checkpoint's")
    parser.add_argument("--n_dys_max", type=int, default=None,
                        help="cap the ladder's n_dys (default the full N_DYS_LADDER)")
    parser.add_argument("--n_newton_max", type=int, default=None,
                        help="cap the ladder's n_newton (default the full N_NEWTON_LADDER)")
    parser.add_argument("--out", default=None,
                        help="default models/<robot>/<robot>__loink<cost><act><recipe>__step<N>.pkl")
    parser.add_argument("--skip_roundtrip", action="store_true")
    args = parser.parse_args()
    explicit = (args.n_dys is not None or args.n_newton is not None
                or args.newton_damping is not None)
    capped = args.n_dys_max is not None or args.n_newton_max is not None
    if explicit and capped:
        parser.error("--n_dys_max / --n_newton_max cap the ladder; an explicit "
                     "--n_dys / --n_newton / --newton_damping runs none")
    try:
        ladder = CappedLadders(args.n_dys_max, args.n_newton_max)
    except ValueError as e:
        parser.error(str(e))
    if args.reladder:
        if args.ckpt_path or args.untrained or args.out or args.skip_roundtrip:
            parser.error("--reladder takes no checkpoint, --untrained, --out or --skip_roundtrip")
        reladder(args.reladder, n_dys=args.n_dys, n_newton=args.n_newton,
                 newton_damping=args.newton_damping, **({} if explicit else ladder))
        sys.exit(0)
    if args.untrained:
        if args.ckpt_path or not (args.robot and args.cost and args.activation):
            parser.error("--untrained takes --robot, --cost and --activation, and no checkpoint")
        out_path, arch, hp, state_dict = write_untrained(args.robot, args.cost, args.activation,
                                                         args.out, seed=args.seed,
                                                         n_dys=args.n_dys, n_newton=args.n_newton,
                                                         newton_damping=args.newton_damping)
    else:
        if not args.ckpt_path:
            parser.error("a checkpoint is required (or --untrained)")
        out_path, arch, hp, state_dict = export(args.ckpt_path, args.out, n_dys=args.n_dys,
                                                n_newton=args.n_newton,
                                                newton_damping=args.newton_damping)
    if not args.skip_roundtrip:
        if explicit:
            roundtrip(out_path, arch, hp, state_dict)
        else:
            roundtrip_ladder(out_path, arch, hp, state_dict, **ladder)
