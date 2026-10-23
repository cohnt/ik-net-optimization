"""Training screens for an exported LOInK chart, before any benchmark is spent on it.

All in float64 (except where (b) measures float32) on the dataset's TEST split (the `*_te`
files, resolved as `screen_cvae.py` does), 5000 samples unless noted, written to
`results/screen_loink/<chart name>.json`. Notation: `G(q / scale) = (y_hat, z)`, the task
`y = [xyz, rpy]` of the dataset pose (`Pose7ToXyzRpy`), `decode(y, z) = scale * G^{-1}([y, z])`.

(a) Task error of `G_1` v the dataset pose -- LOInK's "EE error": position mm and rotation
    deg (geodesic), median / p95 / mean, and the fraction with position error > 50 mm (the
    disconnected-fibre floor: one connected latent line per pose cannot fit two branches).

(b) The inverse: `|G(x) - w|_inf` for `w = [y, z ~ N(0, I)]` against the Davis-Yin
    iteration count in {10, 20, 50, 100, 200, 400}, DYS alone and with the decode's Newton
    polish (`n_newton` steps + the IFT step, i.e. exactly the decode's value), in float64 and
    float32; and the RECOMMENDED `n_dys`: the smallest count whose polished p99 residual is
    below 1e-10 in float64 (None if no count reaches it). `export_loink.py --n_dys` writes it.

(c) The z = 0 solution, LOInK's own answer on our robots (`--n_cost`, default 1000 targets):
    FK error of `decode(y, 0)` v the target pose (median / p95, mm and deg) and its program
    cost (the `m` cost, `0.5 |q - q_nom|^2`, raw units) against the cheapest of the 1000
    training samples nearest the target pose (pose distance `sqrt(|dp|^2 + (L theta)^2)`,
    `L = 0.1 m/rad`, over up to `--n_pool` training samples): the median ratio, with the
    1000th neighbour's distance reported so the looseness of "nearest" is visible. Then, on
    `m` / `x` charts, the head's `J_hat` (raw units, `cost_hat_raw`) v the true training cost
    at `(y_hat, z) = G(q)` on the test samples: MSE and R^2.

(d) Gain: the spectral norm of `dq / d[y, z]` (`vmap(jacrev(decode))`) over 2000 draws, the
    pose from the test split and `z` uniform in the +-5 latent box; its max, and max |q|. The
    certified bound is `max(scale) / mu_total`; the ratio is reported. And the bi-Lipschitz
    inequality `mu |dx| <= |G(x1) - G(x2)| <= nu |dx|` on 10000 random pairs of test `q`
    (`x = q / scale`): the fraction violated (should be 0).

(e) The rpy cut: (a)'s task error binned by the dataset pose's distance to the nearest cut,
    `d = min(pi - |roll|, pi - |yaw|, pi/2 - |pitch|)`, bins [0, 0.1), [0.1, 0.3), [0.3, 1),
    [1, inf); and the fraction of the test set with d < 0.1.

(f) Representability (2000 samples): `max |decode(invert(q)) - q|` through the chart's own
    route (`LoinkInvert` -> raw `c6 = y_hat` -> `LoinkDecode`, what `SetStartFromQ` does; exact
    for every q, the chart taking the raw `c`), the fraction of `y_hat` whose rpy lies off the
    principal domain (where the chart extrapolates), and `|c - c_FK|` -- the paired start's
    `start_c_error` -- in metres (xyz) and radians (rpy, unwrapped as the program's `c`
    variables are), and its 6-norm.

    python scripts/training/screen_loink.py --robot panda \
        --checkpoint models/panda/panda__loinkmrP__step620000.pkl
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "training"))

import src.register_robots  # noqa: E402,F401  -- before anything resolves a robot

from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING  # noqa: E402
from ikflow.utils import get_dataset_directory, get_dataset_filepaths  # noqa: E402
from jrl.math_utils import geodesic_distance_between_quaternions  # noqa: E402

from screen_cvae import LoadTestSplit  # noqa: E402
from src.charts.loink import (  # noqa: E402
    LoadLoinkChart, Pose7ToXyzRpy, RpyCutDistance, TrainingCost, XyzRpyToPose7,
    bilipnet_jacobian, dys_inverse, invert_bilipnet)

LATENT_BOX = 5.0                       # IKFlowProgram.LatentBoxConstraint
DYS_COUNTS = (10, 20, 50, 100, 200, 400)
RECOMMEND_TOL = 1e-10                  # (b): polished p99 residual, float64
FIBRE_FLOOR_MM = 50.0                  # (a)
CUT_BINS = (0.0, 0.1, 0.3, 1.0, math.inf)
POSE_LENGTH = 0.1                      # (c): metres per radian in the pose distance
N_NEIGHBOURS = 1000                    # (c)


def _Stats(x, prefix, scale=1.0, mean=False):
    x = np.asarray(x, dtype=float) * scale
    if x.size == 0:
        out = {f"{prefix}_median": None, f"{prefix}_p95": None}
        return dict(out, **({f"{prefix}_mean": None} if mean else {}))
    out = {f"{prefix}_median": float(np.median(x)), f"{prefix}_p95": float(np.percentile(x, 95))}
    if mean:
        out[f"{prefix}_mean"] = float(x.mean())
    return out


def _Quantiles(x):
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max())}


def _Chunked(fn, *tensors, chunk):
    outs = [fn(*(t[i:i + chunk] for t in tensors)) for i in range(0, tensors[0].shape[0], chunk)]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat(parts) for parts in zip(*outs))
    return torch.cat(outs)


def _PoseErrors(pose_a, pose_b):
    """(position error m, geodesic angle rad) between two [B, 7] pose batches, as numpy."""
    pos = torch.linalg.norm(pose_a[:, :3] - pose_b[:, :3], dim=1).cpu().numpy()
    ang = geodesic_distance_between_quaternions(pose_a[:, 3:7].double().cpu(),
                                                pose_b[:, 3:7].double().cpu()).numpy()
    return pos, ang


## ---------------------------------------------------------------------- (a) and (e)

def ScreenTaskError(model, q, pose, chunk):
    dev = next(model.parameters()).device
    with torch.no_grad():
        y_hat, _ = _Chunked(lambda qq: model.forward_G(qq.to(dev)), q, chunk=chunk)
    pos, ang = _PoseErrors(XyzRpyToPose7(y_hat.cpu()), pose)
    out = {"n": int(q.shape[0]), **_Stats(pos, "pos_mm", 1e3, mean=True),
           **_Stats(np.degrees(ang), "ang_deg", mean=True),
           "frac_pos_over_50mm": float((pos * 1e3 > FIBRE_FLOOR_MM).mean())}
    return out, pos, ang


def ScreenRpyCut(pose, pos, ang):
    d = RpyCutDistance(Pose7ToXyzRpy(pose)[:, 3:6]).numpy()
    bins = []
    for lo, hi in zip(CUT_BINS[:-1], CUT_BINS[1:]):
        m = (d >= lo) & (d < hi)
        bins.append({"bin": [lo, None if math.isinf(hi) else hi], "n": int(m.sum()),
                     **_Stats(pos[m], "pos_mm", 1e3), **_Stats(np.degrees(ang[m]), "ang_deg")})
    return {"n": int(d.size), "frac_within_0.1rad": float((d < 0.1).mean()), "bins": bins}


## ---------------------------------------------------------------------- (b)

def _Newton(net, x, w, steps):
    """`steps` Newton steps on `G(x) = w`. With `steps = n_newton + 1` from the DYS point this is
    exactly `invert_bilipnet`'s value (its `n_newton` polish steps and the IFT step)."""
    for _ in range(steps):
        J = bilipnet_jacobian(net, x)
        x = x + torch.linalg.solve(J, (w - net(x)).unsqueeze(-1)).squeeze(-1)
    return x


def ScreenInversion(model, pose, rng, chunk):
    n_newton = int(model.n_newton)
    y = Pose7ToXyzRpy(pose)
    z = torch.from_numpy(rng.normal(size=(pose.shape[0], model.K)))
    w64 = torch.cat([y, z], dim=1)
    out = {"n": int(pose.shape[0]), "n_newton": n_newton, "counts": list(DYS_COUNTS)}
    for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        m = model if dtype == torch.float64 else copy.deepcopy(model).to(dtype)
        dev = next(m.parameters()).device
        w = w64.to(dtype=dtype, device=dev)
        table = {}
        with torch.no_grad():
            for n in DYS_COUNTS:
                def run(ww, n=n):
                    x = dys_inverse(m.G, ww, n)
                    xp = _Newton(m.G, x, ww, n_newton + 1)
                    return ((m.G(x) - ww).abs().amax(dim=1).double().cpu(),
                            (m.G(xp) - ww).abs().amax(dim=1).double().cpu())
                plain, polished = _Chunked(run, w, chunk=chunk)
                table[str(n)] = {"dys": _Quantiles(plain), "polished": _Quantiles(polished)}
        out[name] = table
    ## `_Newton` from the DYS point is the decode's own value: checked once, at the model's
    ## n_dys, against `invert_bilipnet` itself.
    dev = next(model.parameters()).device
    with torch.no_grad():
        wc = w64[:min(chunk, 200)].to(dev)
        a = invert_bilipnet(model.G, wc, model.n_dys, n_newton)
        b = _Newton(model.G, dys_inverse(model.G, wc, model.n_dys), wc, n_newton + 1)
    out["polish_matches_decode"] = float((a - b).abs().max())
    rec = [n for n in DYS_COUNTS if out["float64"][str(n)]["polished"]["p99"] < RECOMMEND_TOL]
    out["recommended_n_dys"] = rec[0] if rec else None
    out["recommend_rule"] = f"smallest count with float64 polished p99 < {RECOMMEND_TOL:g}"
    return out


## ---------------------------------------------------------------------- (c)

def LoadTrainPool(robot_name, n_pool, seed):
    directory = get_dataset_directory(robot_name)
    samples_tr, poses_tr, _, _, _ = get_dataset_filepaths(directory, [DATASET_TAG_NON_SELF_COLLIDING])
    q = torch.load(samples_tr, map_location="cpu")
    pose = torch.load(poses_tr, map_location="cpu")
    if n_pool < q.shape[0]:
        idx = torch.from_numpy(np.random.default_rng(seed + 1).choice(q.shape[0], n_pool,
                                                                      replace=False))
        q, pose = q[idx], pose[idx]
    return q, pose


def _NearestCheapest(pose_q, pool_pose, pool_cost, k, dev, chunk=16):
    """Per query pose: the min cost over its `k` nearest pool poses, and the k-th distance."""
    pp = pool_pose.to(device=dev, dtype=torch.float32)
    pc = pool_cost.to(device=dev, dtype=torch.float64)
    best, kth = [], []
    for i in range(0, pose_q.shape[0], chunk):
        qq = pose_q[i:i + chunk].to(device=dev, dtype=torch.float32)
        dp = torch.cdist(qq[:, :3], pp[:, :3])
        dot = (qq[:, None, 3:7] * pp[None, :, 3:7]).sum(-1).abs().clamp(max=1.0)
        theta = 2 * torch.arccos(dot)
        d = torch.sqrt(dp * dp + (POSE_LENGTH * theta) ** 2)
        dk, ik = torch.topk(d, k, dim=1, largest=False)
        best.append(pc[ik].min(dim=1).values.cpu())
        kth.append(dk[:, -1].double().cpu())
    return torch.cat(best), torch.cat(kth)


def ScreenZeroLatent(chart, model, robot_name, q, pose, n_cost, n_pool, seed, chunk):
    dev = next(model.parameters()).device
    m_cost = TrainingCost(robot_name, "m", robot=chart.robot)
    pose_c = pose[:n_cost]
    with torch.no_grad():
        q0 = _Chunked(lambda pp: chart.decode(
            model, torch.zeros((pp.shape[0], chart.network_width), dtype=pp.dtype, device=dev),
            Pose7ToXyzRpy(pp).to(dev))[0].cpu(), pose_c, chunk=chunk)
    realised = chart.robot.forward_kinematics(q0, out_device="cpu").double()
    pos, ang = _PoseErrors(realised, pose_c)
    cost0 = m_cost(q0)
    pool_q, pool_pose = LoadTrainPool(robot_name, n_pool, seed)
    pool_cost = torch.cat([m_cost(pool_q[i:i + 100000].double())
                           for i in range(0, pool_q.shape[0], 100000)])
    k = min(N_NEIGHBOURS, pool_q.shape[0])
    best, kth = _NearestCheapest(pose_c, pool_pose, pool_cost, k, dev)
    ratio = (cost0 / best).numpy()
    out = {"n": int(n_cost), "fk_error": {**_Stats(pos, "pos_mm", 1e3),
                                          **_Stats(np.degrees(ang), "ang_deg")},
           "q_absmax": float(q0.abs().max()),
           "cost_m": {"median": float(cost0.median()),
                      "neighbours": k, "pool": int(pool_q.shape[0]),
                      "pose_distance": f"sqrt(|dp|^2 + ({POSE_LENGTH} m/rad * theta)^2)",
                      "kth_neighbour_distance_median": float(kth.median()),
                      "cheapest_neighbour_median": float(best.median()),
                      "ratio_median": float(np.median(ratio)),
                      "ratio_p95": float(np.percentile(ratio, 95))}}
    if chart.arch["cost"] != "u":
        fn = TrainingCost(robot_name, chart.arch["cost"], robot=chart.robot)
        with torch.no_grad():
            y_hat, z = _Chunked(lambda qq: model.forward_G(qq.to(dev)), q, chunk=chunk)
            j_hat = chart.cost_hat_raw(z, y_hat, model).double().cpu()
        j = fn(q)
        sse = float(((j_hat - j) ** 2).sum())
        sst = float(((j - j.mean()) ** 2).sum())
        out["j_hat"] = {"cost": chart.arch["cost"], "n": int(q.shape[0]),
                        "mse": sse / q.shape[0], "r2": 1.0 - sse / sst if sst > 0 else None,
                        "j_mean": float(j.mean()), "j_std": float(j.std())}
    return out


## ---------------------------------------------------------------------- (d)

def ScreenGain(chart, model, q, pose, rng, n_jac, n_pairs, chunk):
    dev = next(model.parameters()).device
    m = min(n_jac, pose.shape[0])
    y = Pose7ToXyzRpy(pose[:m])
    z = torch.from_numpy(rng.uniform(-LATENT_BOX, LATENT_BOX, size=(m, model.K)))
    w = torch.cat([y, z], dim=1)
    p = model.p

    def decode_one(w1):
        return model.decode(w1[None, :p], w1[None, p:])[0]

    jac = torch.func.vmap(torch.func.jacrev(decode_one))
    norms = _Chunked(lambda ww: torch.linalg.matrix_norm(jac(ww.to(dev)), ord=2).detach().cpu(),
                     w, chunk=min(chunk, 250))
    with torch.no_grad():
        qq = _Chunked(lambda ww: model.decode(ww[:, :p].to(dev), ww[:, p:].to(dev)).cpu(), w,
                      chunk=chunk)
        mu, nu, tau = (float(v) for v in model.G.bounds())
    bound = float(model.scale.max()) / mu
    norms = norms.numpy()

    ## The bi-Lipschitz inequality on random pairs of test configurations.
    i = torch.from_numpy(rng.integers(0, q.shape[0], n_pairs))
    j = torch.from_numpy(rng.integers(0, q.shape[0], n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    with torch.no_grad():
        x1, x2 = (q[i] / model.scale.cpu()).to(dev), (q[j] / model.scale.cpu()).to(dev)
        dG = torch.linalg.norm(model.G(x1) - model.G(x2), dim=1)
        dx = torch.linalg.norm(x1 - x2, dim=1)
        r = (dG / dx).cpu().numpy()
    tol = 1e-9
    violated = (r < mu * (1 - tol)) | (r > nu * (1 + tol))
    return {"n": int(m), "z_box": LATENT_BOX,
            "dq_dw_norm_max": float(norms.max()), "dq_dw_norm_p99": float(np.percentile(norms, 99)),
            "bound_scale_max_over_mu": bound, "ratio_max_to_bound": float(norms.max()) / bound,
            "q_absmax": float(qq.abs().max()),
            "mu": mu, "nu": nu, "tau": tau,
            "bilipschitz": {"pairs": int(keep.sum()), "frac_violated": float(violated.mean()),
                            "ratio_min": float(r.min()), "ratio_max": float(r.max())}}


## ---------------------------------------------------------------------- (f)

def ScreenRepresentability(chart, model, q, pose, chunk):
    dev = next(model.parameters()).device

    def run(qq, pp):
        qq = qq.to(dev)
        c6, z = chart.invert(model, qq)
        q_back = chart.decode(model, z, c6)[0]
        c_fk = Pose7ToXyzRpy(pp.to(dev))
        off = RpyCutDistance(c6[:, 3:6]) < 0
        return ((q_back - qq).abs().amax(dim=1).cpu(), (c6 - c_fk).cpu(), off.cpu())

    with torch.no_grad():
        r, dc, off = _Chunked(run, q, pose, chunk=chunk)
    d_pos = torch.linalg.norm(dc[:, :3], dim=1).numpy()
    d_rot = torch.linalg.norm(dc[:, 3:], dim=1).numpy()
    d_all = torch.linalg.norm(dc, dim=1).numpy()
    return {"n": int(q.shape[0]),
            "roundtrip_max_abs": _Quantiles(r.numpy()),
            "frac_roundtrip_over_1e-6": float((r.numpy() > 1e-6).mean()),
            "frac_y_hat_off_rpy_domain": float(off.double().mean()),
            "c_offset_xyz_m": _Quantiles(d_pos), "c_offset_rpy_rad": _Quantiles(d_rot),
            "start_c_error": _Quantiles(d_all)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=5000, help="test rows for (a), (b), (e) and J_hat")
    p.add_argument("--n_cost", type=int, default=1000, help="targets for (c)'s z = 0 solution")
    p.add_argument("--n_pool", type=int, default=5_000_000,
                   help="training samples (c) searches for neighbours")
    p.add_argument("--n_jacobian", type=int, default=2000, help="(d) draws")
    p.add_argument("--n_pairs", type=int, default=10000, help="(d) bi-Lipschitz pairs")
    p.add_argument("--n_repr", type=int, default=2000, help="(f) samples")
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json_out", default=None,
                   help="default results/screen_loink/<checkpoint stem>.json")
    args = p.parse_args(argv)

    t0 = time.monotonic()
    chart = LoadLoinkChart(args.robot, args.checkpoint)
    model = chart.nn_model.double().eval()
    q, pose, directory = LoadTestSplit(args.robot, args.n, args.seed)
    rng = np.random.default_rng(args.seed)

    task, pos, ang = ScreenTaskError(model, q, pose, args.chunk)
    timings = {}
    result = {"robot": args.robot, "checkpoint": os.path.abspath(args.checkpoint),
              "arch": chart.arch, "dataset_dir": directory, "seed": args.seed,
              "task_error": task}
    for key, fn in (
            ("inversion", lambda: ScreenInversion(model, pose, rng, args.chunk)),
            ("zero_latent", lambda: ScreenZeroLatent(chart, model, args.robot, q, pose,
                                                     args.n_cost, args.n_pool, args.seed,
                                                     args.chunk)),
            ("gain", lambda: ScreenGain(chart, model, q, pose, rng, args.n_jacobian,
                                        args.n_pairs, args.chunk)),
            ("rpy_cut", lambda: ScreenRpyCut(pose, pos, ang)),
            ("representability", lambda: ScreenRepresentability(
                chart, model, q[:args.n_repr], pose[:args.n_repr], args.chunk))):
        t = time.monotonic()
        result[key] = fn()
        timings[key] = time.monotonic() - t
    result["timings_s"] = timings
    result["elapsed_s"] = time.monotonic() - t0

    out = args.json_out or os.path.join(
        REPO, "results", "screen_loink",
        os.path.splitext(os.path.basename(args.checkpoint))[0] + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)

    t, inv, zl, g, cut, rep = (result[k] for k in ("task_error", "inversion", "zero_latent",
                                                   "gain", "rpy_cut", "representability"))
    print(f"(a) task error of G_1: pos median {t['pos_mm_median']:.2f} mm p95 {t['pos_mm_p95']:.2f} "
          f"mean {t['pos_mm_mean']:.2f}; ang median {t['ang_deg_median']:.2f} deg p95 "
          f"{t['ang_deg_p95']:.2f}; > 50 mm: {t['frac_pos_over_50mm']:.4f}")
    for n in DYS_COUNTS:
        a64, a32 = inv["float64"][str(n)], inv["float32"][str(n)]
        print(f"(b) n_dys {n:>3}: float64 p99 dys {a64['dys']['p99']:.2e} polished "
              f"{a64['polished']['p99']:.2e} | float32 p99 dys {a32['dys']['p99']:.2e} polished "
              f"{a32['polished']['p99']:.2e}")
    print(f"(b) recommended n_dys: {inv['recommended_n_dys']} (polish == decode to "
          f"{inv['polish_matches_decode']:.1e})")
    print(f"(c) z = 0: FK pos median {zl['fk_error']['pos_mm_median']:.2f} mm p95 "
          f"{zl['fk_error']['pos_mm_p95']:.2f}; ang median {zl['fk_error']['ang_deg_median']:.2f} "
          f"deg; cost ratio v cheapest of {zl['cost_m']['neighbours']} neighbours: median "
          f"{zl['cost_m']['ratio_median']:.3f} (k-th neighbour at "
          f"{zl['cost_m']['kth_neighbour_distance_median']:.3f})"
          + (f"; J_hat R^2 {zl['j_hat']['r2']:.4f}, MSE {zl['j_hat']['mse']:.3g}"
             if "j_hat" in zl else ""))
    print(f"(d) |dq/dw| max {g['dq_dw_norm_max']:.3g} (bound {g['bound_scale_max_over_mu']:.3g}, "
          f"ratio {g['ratio_max_to_bound']:.3f}); max|q| {g['q_absmax']:.3g}; bi-Lipschitz "
          f"violated {g['bilipschitz']['frac_violated']:.4f} of {g['bilipschitz']['pairs']}")
    print(f"(e) within 0.1 rad of an rpy cut: {cut['frac_within_0.1rad']:.4f}; median pos mm by "
          f"bin: {[b['pos_mm_median'] for b in cut['bins']]}")
    print(f"(f) decode(invert(q)) - q: max {rep['roundtrip_max_abs']['max']:.2e}; y_hat off the "
          f"rpy domain {rep['frac_y_hat_off_rpy_domain']:.4f}; start_c_error median "
          f"{rep['start_c_error']['median']:.3g} max {rep['start_c_error']['max']:.3g}")
    print(f"wrote {out} ({result['elapsed_s']:.1f} s; {', '.join(f'{k} {v:.0f}s' for k, v in timings.items())})")
    return result


if __name__ == "__main__":
    main()
