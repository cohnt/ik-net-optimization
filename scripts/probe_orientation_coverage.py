#!/usr/bin/env python3
"""HOW MUCH of SO(3) does a robot reach at a fixed tip position?

`scripts/probe_orientation_freedom.py` answers only the yes/no question -- whether
the reachable orientation set is 3-dimensional -- by the scaling of a covering
radius.  On the GVS push-rod arm it came back 3-dimensional (1.43x per tripling
against 1.44x for a 3-D set), which settles the DIMENSION and says nothing about
the MEASURE: a 3-dimensional set can still be a sliver.  This measures the
measure.

THE ESTIMATOR.  Draw orientations uniformly over SO(3) (the same bi-invariant
measure the pole screen draws from) and ask what fraction lie within `tau` of an
orientation the robot actually achieved at that position.  That fraction is a
lower bound on the reachable measure which tightens as the achieved sample grows,
and it is only meaningful next to two references, both of which this prints:

  * a UNIFORM CONTROL -- the same number of orientations drawn uniformly over
    SO(3), through the same estimator.  It reads ~1.000 at any usable `tau`, and
    if it does not, `tau` is below the sampling resolution and the run is
    measuring sample sparseness rather than the robot.
  * a SATURATION LADDER over the achieved sample size.  A coverage still climbing
    with N is not converged; one that has flattened is the set's measure.

The ball of geodesic radius `theta` has normalized Haar measure
`(theta - sin theta) / pi`, so at N achieved orientations the expected covering
distance inside a set of measure f is where `(N / f)(theta - sin theta)/pi ~ 1`.
At N = 6000 that is about 7.4 degrees for f = 1, which is why `tau` of 20-30
degrees is well clear of the resolution floor while staying far below the scale
of any real hole.

STRUCTURE, from the same samples: the tip frame's z-axis is the backbone tangent,
so a tip orientation splits into a tangent direction on S^2 and a roll about it.
The probe reports how much of the sphere the tangent covers and how much of the
full 360 degrees of roll is available at a typical tangent, which says WHAT is
missing rather than only how much.

    python scripts/probe_orientation_coverage.py --robot gvs_pushrod9_o1 \\
        --draws 8000000 --workers 48
"""
import argparse
import multiprocessing as mp
import os
import pathlib
import sys
import time

## One thread per worker, before numpy/jax load here or in any spawned child.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402

REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

#: The conditioning box centre both soft arms' pole screens use, always measured so the
#: number stays comparable with `probe_orientation_freedom.py`'s.
SCREEN_CENTRE = (0.0, 0.0, 0.45)


def _ball_measure(theta):
    """Normalized Haar measure of a geodesic ball of radius `theta` in SO(3)."""
    return (theta - np.sin(theta)) / np.pi


def _worker(job):
    """Draw configurations, keep the tip orientations landing near any centre."""
    robot_name, count, seed, index, cpus, batch, centres, radius = job
    if cpus:
        os.sched_setaffinity(0, set(cpus))
    os.environ["GVS_ARM_XLA_THREADS"] = str(max(1, len(cpus) if cpus else 1))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    np.random.seed(int(np.random.SeedSequence([seed, index]).generate_state(1)[0]))
    import torch
    torch.set_num_threads(1)
    import src.register_robots  # noqa: F401
    from jrl.robots import get_robot
    robot = get_robot(robot_name)
    centres = np.asarray(centres)
    ## The GVS arm's forward map can fail to converge on a draw, and jrl's
    ## `forward_kinematics` raises rather than reporting which; its model exposes the
    ## converged mask, so use that where it exists and fall back to the generic call.
    model = getattr(robot, "_model", None)
    batched = getattr(model, "TipPoseBatch", None)

    kept_quat = [[] for _ in range(len(centres))]
    positions = []
    drawn = 0
    start = time.time()
    while drawn < count:
        take = min(batch, count - drawn)
        cfg = robot.sample_joint_angles(take)
        if batched is not None:
            pose, converged = batched(cfg)
            pose = pose[converged]
        else:
            pose = np.asarray(robot.forward_kinematics(
                torch.as_tensor(cfg, dtype=torch.float64, device="cpu")).cpu())
        drawn += take
        ## A thinned copy of the tip positions, for choosing centres and for reporting
        ## where the arm actually goes; the full set would be gigabytes.
        positions.append(pose[::37, :3].copy())
        for c, centre in enumerate(centres):
            near = np.linalg.norm(pose[:, :3] - centre, axis=1) < radius
            if near.any():
                kept_quat[c].append(pose[near, 3:].copy())
    quats = [np.concatenate(k) if k else np.zeros((0, 4)) for k in kept_quat]
    return (index, quats, np.concatenate(positions), drawn, time.time() - start)


def _cpu_slices(workers):
    cpus = sorted(os.sched_getaffinity(0))
    return [cpus[i::workers] for i in range(workers)] if cpus else [[] for _ in range(workers)]


def _draw(robot_name, total, seed, workers, batch, centres, radius, timeout):
    shares = [total // workers + (1 if i < total % workers else 0) for i in range(workers)]
    slices = _cpu_slices(workers)
    jobs = [(robot_name, n, seed, i, slices[i], batch, centres, radius)
            for i, n in enumerate(shares) if n > 0]
    kept = [[] for _ in range(len(centres))]
    positions, drawn = [], 0
    context = mp.get_context("spawn")
    with context.Pool(workers) as pool:
        iterator = pool.imap_unordered(_worker, jobs)
        for k in range(len(jobs)):
            index, quats, pos, n, elapsed = iterator.next(timeout=timeout)
            for c in range(len(centres)):
                if len(quats[c]):
                    kept[c].append(quats[c])
            positions.append(pos)
            drawn += n
            if k < 2 or (k + 1) % 8 == 0 or k == len(jobs) - 1:
                print(f"    worker {index} done: {n:,} draws in {elapsed:.0f} s "
                      f"({elapsed / max(n, 1) * 1e3:.1f} ms each); "
                      f"{k + 1}/{len(jobs)} workers", flush=True)
    return ([np.concatenate(k) if k else np.zeros((0, 4)) for k in kept],
            np.concatenate(positions), drawn)


def _nearest_angles(probes, achieved, chunk=2048):
    """Geodesic angle (degrees) from each probe orientation to the nearest achieved."""
    out = np.empty(len(probes))
    for i in range(0, len(probes), chunk):
        block = probes[i:i + chunk]
        dots = np.abs(block @ achieved.T).clip(0.0, 1.0)
        out[i:i + chunk] = np.degrees(2.0 * np.arccos(dots.max(axis=1)))
    return out


def _uniform_quaternions(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.sign(q[:, :1] + (q[:, :1] == 0))


def _tangent_and_roll(quats):
    """Tip-frame z-axis (the backbone tangent) and the roll about it, in degrees.

    The roll is measured against the shortest-arc frame that carries world z onto the
    tangent, so it is a genuine function of the orientation given the tangent.
    """
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    ## Third column of the rotation matrix: the tip frame's z-axis in world coordinates.
    tangent = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x),
                        1 - 2 * (x * x + y * y)], axis=1)
    ## The shortest arc from world z to the tangent, as a quaternion.
    axis = np.stack([-tangent[:, 1], tangent[:, 0], np.zeros(len(tangent))], axis=1)
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    safe = norm[:, 0] > 1e-12
    axis = np.where(norm > 1e-12, axis / np.where(norm > 1e-12, norm, 1.0),
                    np.array([1.0, 0.0, 0.0]))
    angle = np.arccos(tangent[:, 2].clip(-1.0, 1.0))
    ref = np.concatenate([np.cos(angle / 2)[:, None],
                          axis * np.sin(angle / 2)[:, None]], axis=1)
    ref[~safe] = np.array([1.0, 0.0, 0.0, 0.0])
    ## roll quaternion = ref^-1 * q, whose rotation is about z by the roll angle.
    rw, rx, ry, rz = ref[:, 0], -ref[:, 1], -ref[:, 2], -ref[:, 3]
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    roll_w = rw * qw - rx * qx - ry * qy - rz * qz
    roll_z = rw * qz + rz * qw + rx * qy - ry * qx
    return tangent, np.degrees(2.0 * np.arctan2(roll_z, roll_w))


def _coverage_table(name, achieved, probes, taus, ladder=True):
    """Coverage of SO(3) at each `tau`, plus the saturation ladder."""
    angles = _nearest_angles(probes, achieved)
    row = [float((angles < t).mean()) for t in taus]
    print(f"  {name:<22} N={len(achieved):>7,}  "
          + "  ".join(f"{t:>2.0f}d {c:6.1%}" for t, c in zip(taus, row))
          + f"   median {np.median(angles):5.1f}d  p99 {np.percentile(angles, 99):5.1f}d")
    if ladder and len(achieved) >= 800:
        steps = []
        n = len(achieved)
        while n >= 400:
            sub = _nearest_angles(probes, achieved[:n])
            steps.append((n, float((sub < taus[len(taus) // 2]).mean())))
            n //= 3
        pretty = "  ".join(f"N={n:,}:{c:.1%}" for n, c in reversed(steps))
        print(f"  {'saturation @ ' + str(taus[len(taus) // 2]) + 'd':<22} {pretty}")
    return row, angles


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="gvs_pushrod9_o1")
    ap.add_argument("--draws", type=int, default=8_000_000)
    ap.add_argument("--radius", type=float, default=0.05, help="'same position' ball, metres")
    ap.add_argument("--num_centres", type=int, default=8,
                    help="positions to measure, chosen from a pilot by farthest-point "
                         "selection, plus the screen centre")
    ap.add_argument("--pilot", type=int, default=200_000)
    ap.add_argument("--probes", type=int, default=20_000)
    ap.add_argument("--taus", type=float, nargs="+", default=[10, 20, 30, 45])
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("PROBE_WORKERS", 0)) or None)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker_timeout", type=float, default=10800)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.workers:
        args.workers = max(1, len(os.sched_getaffinity(0)) // 2)

    import src.register_robots as rr  # noqa: F401
    print(f"{args.robot}: {args.draws:,} draws on {args.workers} workers, "
          f"'same position' = a {args.radius * 100:.0f} cm ball", flush=True)

    ## PILOT: where does the tip actually go? Centres are chosen from the achieved
    ## positions themselves (farthest-point, so they span the workspace rather than
    ## crowding the mode), which keeps every centre one the robot reaches often enough
    ## to measure. The screen centre is always included, measured or not.
    t0 = time.time()
    _, positions, _ = _draw(args.robot, args.pilot, args.seed + 7, args.workers,
                            args.batch, np.zeros((1, 3)), 0.0, args.worker_timeout)
    rng = np.random.default_rng(args.seed)
    ## Centres are the DENSEST positions the pilot found, kept `2 * radius` apart so their
    ## balls do not overlap. Density matters: a farthest-point spread lands on rare corners
    ## of the workspace, where too few draws arrive to measure anything. The workspace's
    ## extremes are reported separately by the position summary above.
    voxel = np.round(positions / args.radius).astype(int)
    keys, counts = np.unique(voxel, axis=0, return_counts=True)
    order = np.argsort(-counts)
    centres, taken = [np.asarray(SCREEN_CENTRE)], []
    for idx in order:
        c = keys[idx] * args.radius
        if all(np.linalg.norm(c - other) >= 2 * args.radius for other in centres):
            centres.append(c)
            taken.append(int(counts[idx]))
        if len(centres) >= args.num_centres:
            break
    centres = np.asarray(centres)
    print(f"  pilot: {len(positions):,} tip positions kept, radius "
          f"{np.linalg.norm(positions, axis=1).mean():.3f} m mean; centres chosen in "
          f"{time.time() - t0:.0f} s", flush=True)
    for i, c in enumerate(centres):
        share = "" if i == 0 else f"   pilot density {taken[i - 1] / len(positions):.2%}"
        print(f"    centre {i}: [{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}]"
              + ("   (the pole screen's box centre)" if i == 0 else share))

    achieved, positions, drawn = _draw(args.robot, args.draws, args.seed, args.workers,
                                       args.batch, centres, args.radius, args.worker_timeout)
    print(f"  drew {drawn:,} configurations in {time.time() - t0:.0f} s; "
          f"kept {sum(len(a) for a in achieved):,} across {len(centres)} centres", flush=True)

    probes = _uniform_quaternions(rng, args.probes)
    taus = list(args.taus)
    print(f"\nFRACTION OF SO(3) within tau of a REACHED orientation "
          f"({args.probes:,} uniform probe orientations)\n")
    control_n = max(len(a) for a in achieved)
    _coverage_table("uniform control", _uniform_quaternions(rng, control_n), probes, taus)
    print()
    results = {}
    for i, (centre, quats) in enumerate(zip(centres, achieved)):
        if len(quats) < 400:
            print(f"  centre {i} [{centre[0]:+.2f},{centre[1]:+.2f},{centre[2]:+.2f}]: only "
                  f"{len(quats):,} achieved -- too few to measure")
            continue
        label = f"centre {i} [{centre[0]:+.2f},{centre[1]:+.2f},{centre[2]:+.2f}]"
        row, angles = _coverage_table(label, quats, probes, taus)
        tangent, roll = _tangent_and_roll(quats)
        ## Solid angle the tangent covers, by the same threshold logic on S^2.
        tp = rng.normal(size=(4000, 3))
        tp /= np.linalg.norm(tp, axis=1, keepdims=True)
        cos = (tp @ tangent.T).max(axis=1).clip(-1, 1)
        sphere = float((np.degrees(np.arccos(cos)) < taus[1]).mean())
        ## Roll available at a typical tangent: bin by tangent direction, take the span.
        ## The bin is COARSENED until some cell holds enough samples -- at a fixed
        ## resolution most centres produced no qualifying cell and the statistic read `nan`.
        spans, res = [], 0
        for res in (6, 4, 3, 2):
            keys = np.round(tangent * res).astype(int)
            order = np.lexsort(keys.T)
            spans, start = [], 0
            for j in range(1, len(order) + 1):
                if j == len(order) or (keys[order[j]] != keys[order[start]]).any():
                    if j - start >= 20:
                        r = np.sort(roll[order[start:j]])
                        gap = np.diff(np.concatenate([r, [r[0] + 360]]))
                        spans.append(360.0 - gap.max())
                    start = j
            if spans:
                break
        span = float(np.median(spans)) if spans else float("nan")
        shown = f"{span:.0f}d of 360" if spans else "unmeasured (no cell with 20 samples)"
        print(f"  {'structure':<22} tangent covers {sphere:.1%} of the sphere "
              f"(within {taus[1]:.0f}d); roll spans {shown} at a typical tangent"
              f"  [{len(spans)} tangent cells at 1/{res}]")
        results[label] = dict(centre=centre, coverage=row, angles=angles,
                              sphere=sphere, roll_span=span, n=len(quats))
        print()

    print(f"tau is the tolerance: 'within tau of reachable'. Read the uniform control as the "
          f"\nestimator's ceiling at this sample size, and the saturation ladder as whether the "
          f"\nnumber has converged. Measure of a ball of radius tau in SO(3): "
          + ", ".join(f"{t:.0f}d {_ball_measure(np.radians(t)):.4f}" for t in taus))
    if args.out:
        np.savez_compressed(args.out, centres=centres, probes=probes,
                            **{f"achieved_{i}": a for i, a in enumerate(achieved)})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
