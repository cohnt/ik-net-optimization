#!/usr/bin/env python3
"""Is the flange orientation free given its position?  The question that decides
whether an in-training pole screen is in distribution.

The vendored ikflow fork's pole callback draws its conditioning poses as a
position and an orientation INDEPENDENTLY.  That is only a fair draw if the robot
can reach (most of) SO(3) at a given position.  The soft arm cannot -- `soft12`
activates no torsion, so its tip orientation is essentially determined by its tip
position, an independently drawn orientation is never reachable, and the callback
spends a whole run evaluating the flow out of distribution: `pole/max` 2.4e8
against an in-distribution screen's 5.44 at a STRICTER threshold, eight orders of
magnitude apart and a statement about unreachable poses rather than about the
chart.

`helix7` is a 7-DoF arm with a roll-pitch-roll wrist, so the structural
expectation is that orientation IS free -- but "probably spans SO(3)" is an
assumption, and the soft arm's experience is what it costs to leave one standing.
This measures it.

The method avoids needing an IK solver.  Sample configurations, keep the ones
whose flange lands in a small ball about a point in the conditioning box, and ask
how far an independently drawn orientation sits from the nearest one ACHIEVED
there.  A single such number is uninformative, because it is dominated by how
sparsely SO(3) was sampled.  What discriminates is its SCALING: the covering
radius of N points spread over a d-dimensional set falls as N**(-1/d), so an
orientation set that fills 3-dimensional SO(3) shrinks by 3**(1/3) = 1.44x per
3x in N and never reaches a floor, while one confined to a lower-dimensional
subset PLATEAUS at the distance from a random orientation to that subset.

Measured for helix7_p050 at [0.4, 0, 0.5] (the callback's box centre), 22.5M
draws, 6,505 of them landing within 5 cm: median degrees to the nearest achieved
orientation 24.71 / 16.85 / 11.58 / 8.12 at N = 200 / 600 / 1800 / 5400, i.e.
1.47x, 1.45x and 1.43x per 3x against the predicted 1.44x.  No floor, so the
orientation set is full-dimensional and the callback is in distribution.  That is
why `ScreenDomain` returns the rigid tuple unchanged for this robot and why its
pole numbers are quotable beside the record's.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.helix_arm.kinematics import forward_kinematics
from src.helix_arm.params import PRIMARY, SPECS, GetSpec


def AchievedOrientations(spec, centre, radius, want, rng, batch=2_500_000):
    """Quaternions of flange poses landing within `radius` of `centre`."""
    kept, drawn = [], 0
    while sum(len(k) for k in kept) < want:
        lo = np.array([j.lower for j in spec.joints])
        hi = np.array([j.upper for j in spec.joints])
        cfg = rng.uniform(lo, hi, size=(batch, len(lo)))
        drawn += batch
        pose = forward_kinematics(torch.as_tensor(cfg, dtype=torch.float64), spec).numpy()
        near = np.linalg.norm(pose[:, :3] - centre, axis=1) < radius
        kept.append(pose[near, 3:])
    return np.concatenate(kept), drawn


def NearestAngles(achieved, probes):
    """Degrees from each probe orientation to the nearest achieved one."""
    dots = np.abs(probes @ achieved.T).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots.max(axis=1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=PRIMARY, choices=sorted(SPECS))
    ap.add_argument("--centre", type=float, nargs=3, default=[0.4, 0.0, 0.5],
                    help="position to hold, in metres (default: the conditioning box centre)")
    ap.add_argument("--radius", type=float, default=0.05, help="'same position' ball, metres")
    ap.add_argument("--achieved", type=int, default=6000, help="orientations to collect")
    ap.add_argument("--probes", type=int, default=4000, help="independent orientations to test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = GetSpec(args.robot)
    rng = np.random.default_rng(args.seed)
    centre = np.array(args.centre)

    achieved, drawn = AchievedOrientations(spec, centre, args.radius, args.achieved, rng)
    print(f"{spec.name}: drew {drawn:,} configurations; {len(achieved):,} land within "
          f"{args.radius * 100:.0f} cm of {centre.tolist()}")

    probes = rng.normal(size=(args.probes, 4))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)

    ## The scaling IS the measurement; a single row of it means nothing on its own.
    print(f"\n{'N achieved':>11}  {'median':>8}  {'p99':>8}  {'max':>8}  shrink per 3x "
          f"(1.44 if SO(3)-filling)")
    counts = [n for n in (200, 600, 1800, 5400) if n <= len(achieved)]
    previous = None
    for n in counts:
        ang = NearestAngles(achieved[:n], probes)
        median = float(np.median(ang))
        ratio = "--" if previous is None else f"{previous / median:.2f}x"
        print(f"{n:>11}  {median:>8.2f}  {np.percentile(ang, 99):>8.2f}  "
              f"{ang.max():>8.2f}  {ratio:>12}")
        previous = median

    if len(counts) >= 3:
        first = float(np.median(NearestAngles(achieved[:counts[0]], probes)))
        last = float(np.median(NearestAngles(achieved[:counts[-1]], probes)))
        steps = len(counts) - 1
        print(f"\ngeometric mean shrink per 3x: {(first / last) ** (1.0 / steps):.2f}x "
              f"against 1.44x for a 3-dimensional set")
        print("A ratio near 1.44 with no floor => orientation is free given position, so an "
              "independently\ndrawn conditioning orientation is reachable and an in-training "
              "pole screen is in distribution.")


if __name__ == "__main__":
    main()
