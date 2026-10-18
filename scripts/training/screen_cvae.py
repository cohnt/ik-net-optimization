"""Training screens for an exported cVAE chart, before any benchmark is spent on it.

Four screens, each answering a question the benchmark would otherwise answer expensively.
All on the dataset's TEST split (the `*_te` files `IkfLitDataset` loads, resolved the same
way), all in float64, written to `results/screen_cvae/<chart name>.json`.

(a) FK error of `decode(c, z ~ N(0, I))` at test poses -- the chart a NATIVE start sees.
    Position mm / angle deg: median, p95, max (over the non-runaway draws, with the runaway
    fraction alongside, as `chart_accuracy.py` does for flows).

(b) Paired-start representability. The paired start is "encoder only": `z = mu(q, c)`, and
    the residual `r = q - decode(c, mu)` goes to the correction, whose box is +-0.1
    (`ProgramOptions.correction_bound`). A start with `max|r| > 0.1` cannot be represented
    and its cell is scored `unrepresentable_start`, so the FRACTION inside the box predicts
    that rate before a single cell runs. Also the fraction of `mu` outside the +-5 latent
    box (a general constraint, so not projected -- reported because IPOPT starts outside).

(c) Active latent units: the batch-mean KL of each latent dimension against the prior. A
    dimension under 0.01 nats is collapsed -- in the flow-width rung the dimensions past the
    redundancy are expected to be, and then they are flat decision variables in the NLP.

(d) The runaway screen, the chart's answer to the project's central finding: the flow
    charts' residual failures were configurations of 1e7-1e16 rad on near-worst-case gain
    regions that the solver is attracted to because `dq/dvars` is as large as `q`. So:
    `z` uniform in the +-5 latent box (the program's whole latent domain) and `z ~ N(0, I)`
    at test poses, reporting max `|q|_inf`, and the max / p99 spectral norm of `dq/dz` and
    `dq/dpose` (`torch.func.jacrev`, through the chart's own `decode`).

    python scripts/training/screen_cvae.py --robot panda \
        --checkpoint models/panda/panda__cvae7__step620000.pkl
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

import src.register_robots  # noqa: E402,F401  -- before anything resolves a robot

from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING  # noqa: E402
from ikflow.utils import get_dataset_directory, get_dataset_filepaths  # noqa: E402
from jrl.math_utils import geodesic_distance_between_quaternions  # noqa: E402

from src.charts.cvae import LoadCVAEChart  # noqa: E402

CORRECTION_BOX = 0.1     # ProgramOptions.correction_bound
LATENT_BOX = 5.0         # IKFlowProgram.LatentBoxConstraint
ACTIVE_KL_NATS = 0.01    # as train_cvae.py
RUNAWAY_RAD = 10.0       # as chart_accuracy.py


def LoadTestSplit(robot_name, n, seed):
    """`n` rows of the test split, the same files the trainer's `IkfLitDataset` reads."""
    directory = get_dataset_directory(robot_name)
    _, _, samples_te, poses_te, _ = get_dataset_filepaths(
        directory, [DATASET_TAG_NON_SELF_COLLIDING])
    q = torch.load(samples_te, map_location="cpu")
    pose = torch.load(poses_te, map_location="cpu")
    if n < q.shape[0]:
        idx = torch.from_numpy(np.random.default_rng(seed).choice(q.shape[0], n, replace=False))
        q, pose = q[idx], pose[idx]
    return q.double(), pose.double(), directory


def _Stats(x, prefix, scale=1.0):
    x = np.asarray(x, dtype=float) * scale
    if x.size == 0:
        return {f"{prefix}_median": None, f"{prefix}_p95": None, f"{prefix}_max": None}
    return {f"{prefix}_median": float(np.median(x)), f"{prefix}_p95": float(np.percentile(x, 95)),
            f"{prefix}_max": float(x.max())}


def _Chunked(fn, *tensors, chunk):
    return torch.cat([fn(*(t[i:i + chunk] for t in tensors))
                      for i in range(0, tensors[0].shape[0], chunk)])


def ScreenFK(chart, model, pose, rng, chunk):
    dev = next(model.parameters()).device
    z = torch.from_numpy(rng.normal(size=(pose.shape[0], chart.network_width)))
    with torch.no_grad():
        q = _Chunked(lambda zz, pp: chart.decode(model, zz.to(dev), pp.to(dev))[0].cpu(),
                     z, pose, chunk=chunk)
    usable = q.abs().amax(dim=1) < RUNAWAY_RAD
    ## At each robot's OWN default dtype: jrl's rigid-arm FK builds float32 intermediates
    ## whatever `dtype` says (passing float64 raises), while this project's robots compute
    ## in float64. float32 FK is micrometre-level, far below what this screen reports.
    realised = chart.robot.forward_kinematics(q[usable], out_device="cpu").double()
    target = pose[usable]
    pos = torch.linalg.norm(realised[:, :3] - target[:, :3], dim=1).numpy()
    ang = geodesic_distance_between_quaternions(realised[:, 3:7], target[:, 3:7]).numpy()
    return {"n": int(pose.shape[0]), "usable_frac": float(usable.double().mean()),
            **_Stats(pos, "pos_mm", 1e3), **_Stats(np.degrees(ang), "ang_deg")}


def ScreenRepresentability(chart, model, q, pose, chunk):
    dev = next(model.parameters()).device

    def residual(qq, pp):
        qq, pp = qq.to(dev), pp.to(dev)
        mu = chart.encode(model, qq, pp)
        q_hat, _ = chart.decode(model, mu, pp)
        return torch.cat([(qq - q_hat).abs().amax(dim=1, keepdim=True),
                          mu.abs().amax(dim=1, keepdim=True)], dim=1).cpu()

    with torch.no_grad():
        out = _Chunked(residual, q, pose, chunk=chunk).numpy()
    r, mu_inf = out[:, 0], out[:, 1]
    return {"n": int(q.shape[0]), "correction_box": CORRECTION_BOX,
            "frac_within_box": float((r <= CORRECTION_BOX).mean()),
            "max_abs_residual": {"median": float(np.median(r)),
                                 "p90": float(np.percentile(r, 90)),
                                 "p99": float(np.percentile(r, 99)), "max": float(r.max())},
            "frac_mu_outside_latent_box": float((mu_inf > LATENT_BOX).mean()),
            "mu_absmax": float(mu_inf.max())}


def ScreenActiveUnits(model, q, pose, chunk):
    dev = next(model.parameters()).device

    def kl(qq, pp):
        mu, logvar = model.encode_dist(qq.to(dev), pp.to(dev))
        return (0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar)).cpu()

    with torch.no_grad():
        per_dim = _Chunked(kl, q, pose, chunk=chunk).mean(dim=0).numpy()
    return {"kl_per_dim": [float(v) for v in per_dim], "kl_total": float(per_dim.sum()),
            "threshold_nats": ACTIVE_KL_NATS, "active_dims": int((per_dim > ACTIVE_KL_NATS).sum()),
            "sigma_q": [float(v) for v in model.log_sigma_q.exp().detach().cpu()]}


def ScreenRunaway(chart, model, pose, rng, n_jac, chunk):
    """Max |q| and Jacobian spectral norms over the latent box and under the prior."""
    dev = next(model.parameters()).device
    k = chart.network_width

    def decode_one(z1, p1):
        return chart.decode(model, z1[None], p1[None])[0][0]

    jac_z = torch.func.vmap(torch.func.jacrev(decode_one, argnums=0))
    jac_p = torch.func.vmap(torch.func.jacrev(decode_one, argnums=1))

    out = {}
    draws = {"box": rng.uniform(-LATENT_BOX, LATENT_BOX, size=(pose.shape[0], k)),
             "prior": rng.normal(size=(pose.shape[0], k))}
    for name, z_np in draws.items():
        z = torch.from_numpy(z_np)
        with torch.no_grad():
            q = _Chunked(lambda zz, pp: chart.decode(model, zz.to(dev), pp.to(dev))[0].cpu(),
                         z, pose, chunk=chunk)
        q_inf = q.abs().amax(dim=1).numpy()
        m = min(n_jac, pose.shape[0])
        jz = _Chunked(lambda zz, pp: torch.linalg.matrix_norm(
            jac_z(zz.to(dev), pp.to(dev)), ord=2).detach().cpu(), z[:m], pose[:m], chunk=chunk)
        jp = _Chunked(lambda zz, pp: torch.linalg.matrix_norm(
            jac_p(zz.to(dev), pp.to(dev)), ord=2).detach().cpu(), z[:m], pose[:m], chunk=chunk)
        jz, jp = jz.numpy(), jp.numpy()
        out[name] = {"n": int(pose.shape[0]), "q_absmax": float(q_inf.max()),
                     "q_inf_p99": float(np.percentile(q_inf, 99)),
                     "frac_runaway": float((q_inf > RUNAWAY_RAD).mean()),
                     "n_jacobian": int(m),
                     "dq_dz_norm_max": float(jz.max()), "dq_dz_norm_p99": float(np.percentile(jz, 99)),
                     "dq_dpose_norm_max": float(jp.max()),
                     "dq_dpose_norm_p99": float(np.percentile(jp, 99))}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=5000, help="test rows for (b), (c), (d)")
    p.add_argument("--n_fk", type=int, default=2000,
                   help="test rows for (a); the soft PCS arm's FK is float64 CPU")
    p.add_argument("--n_jacobian", type=int, default=2000)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json_out", default=None,
                   help="default results/screen_cvae/<checkpoint stem>.json")
    args = p.parse_args(argv)

    t0 = time.monotonic()
    chart = LoadCVAEChart(args.robot, args.checkpoint)
    model = chart.nn_model.double().eval()
    q, pose, directory = LoadTestSplit(args.robot, args.n, args.seed)
    rng = np.random.default_rng(args.seed)

    result = {
        "robot": args.robot,
        "checkpoint": os.path.abspath(args.checkpoint),
        "arch": chart.arch,
        "dataset_dir": directory,
        "seed": args.seed,
        "fk_error": ScreenFK(chart, model, pose[:args.n_fk], rng, args.chunk),
        "representability": ScreenRepresentability(chart, model, q, pose, args.chunk),
        "active_units": ScreenActiveUnits(model, q, pose, args.chunk),
        "runaway": ScreenRunaway(chart, model, pose, rng, args.n_jacobian, args.chunk),
    }
    result["elapsed_s"] = time.monotonic() - t0

    out = args.json_out or os.path.join(
        REPO, "results", "screen_cvae",
        os.path.splitext(os.path.basename(args.checkpoint))[0] + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)

    fk, rep, au, run = (result[k] for k in ("fk_error", "representability", "active_units",
                                            "runaway"))
    print(f"(a) FK error, z ~ N(0, I): pos median {fk['pos_mm_median']:.2f} mm "
          f"p95 {fk['pos_mm_p95']:.2f}; ang median {fk['ang_deg_median']:.2f} deg "
          f"p95 {fk['ang_deg_p95']:.2f}; usable {fk['usable_frac']:.4f}")
    print(f"(b) paired start: {rep['frac_within_box']:.4f} within +-{CORRECTION_BOX} "
          f"(max|r| median {rep['max_abs_residual']['median']:.4f}, "
          f"p99 {rep['max_abs_residual']['p99']:.4f}); mu outside +-{LATENT_BOX}: "
          f"{rep['frac_mu_outside_latent_box']:.4f}")
    print(f"(c) active units: {au['active_dims']} of {chart.network_width} "
          f"(KL per dim {np.array2string(np.array(au['kl_per_dim']), precision=3)})")
    for name, r in run.items():
        print(f"(d) {name}: max|q| {r['q_absmax']:.3g}, |dq/dz| max {r['dq_dz_norm_max']:.3g} "
              f"p99 {r['dq_dz_norm_p99']:.3g}, |dq/dpose| max {r['dq_dpose_norm_max']:.3g} "
              f"p99 {r['dq_dpose_norm_p99']:.3g}")
    print(f"wrote {out} ({result['elapsed_s']:.1f} s)")
    return result


if __name__ == "__main__":
    main()
