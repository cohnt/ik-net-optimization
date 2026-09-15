"""How often does a random configuration put the target inside a shelf compartment?

THIS SCRIPT GATES THE CAMPAIGN.  `--shelf-inset` decides how hard the hardened task is, and
the rejection sampler that implements it can become unaffordable -- or simply never
terminate -- without any of that being visible in a benchmark summary.  Run this, read the
`P(trip/60)` column, and only then pick an inset and a guard.

WHAT IT MEASURES.  For each (robot, task): draw uniform configurations over the plant's own
position limits, keep the collision-free ones, and record the target frame's world origin.
Then score that one cached set of points against every candidate inset, so the insets are
exactly comparable and the whole sweep costs one pass.  The grasp rows additionally run the
floating-mug penetration screen, but only on the points that already passed containment
(~1% of draws), so it costs nothing.

WHY IT DOES NOT BUILD A PROGRAM.  Containment is a question about geometry, not about the
flow, so this builds the plant directly and needs neither torch nor a checkpoint -- it runs
on a laptop with no iiwa weights.  The collision settings are READ OFF `ProgramOptions()`
rather than retyped, so they cannot drift from what the benchmark actually uses.

THE COLUMN THAT MATTERS.  Acceptance restarts at every accepted target, so the sampler's
guard is a per-target tail bound: P(trip on one target) = (1 - p)^guard, and over a
60-target grid P(trip) = 1 - (1 - that)^60.  A trip raises RuntimeError partway through a
queued run and kills every shard of it, so the criterion is:

    an inset is viable for a (robot, task) pair iff P(trip over 60 targets) < 1%
    AT THE GUARD THE MANIFEST WILL ACTUALLY PASS.

Usage:
    python scripts/probe_shelf_acceptance.py                      # the defaults below
    python scripts/probe_shelf_acceptance.py --draws 50000 --guard 5000,50000
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint

from src.generic_program import ProgramOptions
from src.shelf_regions import PointInShelfCompartments, ShelfCompartmentRegions
from src.target_screening import SCENES, FloatingMugScreen, SceneFile
from src.utils import BuildEnv, HiddenPrints


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robots", default="panda,iiwa")
    p.add_argument("--tasks", default="mug,pose")
    p.add_argument("--insets", default="0,0.05,0.10,0.125")
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--guard", default="5000,50000",
                   help="rejection guards to report the trip probability at")
    p.add_argument("--targets", type=int, default=60,
                   help="targets per grid, for the P(trip) column")
    p.add_argument("--scene", default="hardened",
                   choices=("hardened", "nobin", "legacy"),
                   help="which obstacle set to probe. `nobin` keeps the decorative mugs, "
                        "four of which sit inside shelf compartments, so its acceptance is "
                        "strictly lower than `hardened`'s -- check it before fielding it.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def probe_scene(robot, task, draws, seed, scene="hardened"):
    """Returns (target points of the collision-free draws, n_drawn, seconds, screen)."""
    spec = SCENES[(robot, task)]
    yaml_file = SceneFile(robot, task, scene)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=yaml_file)
    plant = diagram.GetSubsystemByName("plant")
    context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())

    ## Read off the defaults rather than retyping them, so the probe cannot drift from the
    ## benchmark's own collision geometry.
    options = ProgramOptions()
    collision = MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=options.collision_bound,
        influence_distance_offset=options.collision_influence_offset,
        plant_context=context)

    frame = plant.GetFrameByName(spec.target_frame)
    lower, upper = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    rng = np.random.default_rng(seed)

    points, start = [], time.time()
    for _ in range(draws):
        q = rng.uniform(lower, upper)
        plant.SetPositions(context, q)
        if collision.Eval(q) < 1:
            points.append(frame.CalcPoseInWorld(context))
    seconds = time.time() - start

    screen = None
    if task == "mug":
        with HiddenPrints():
            screen = FloatingMugScreen(yaml_file, spec.robot_instances)
    return points, seconds, screen, spec


def main():
    args = parse_args()
    insets = [float(x) for x in args.insets.split(",")]
    guards = [int(x) for x in args.guard.split(",")]

    print("Shelf-containment acceptance, %d uniform draws per scene, seed %d."
          % (args.draws, args.seed))
    print("P(trip/%d) is the chance the sampler's consecutive-rejection guard fires "
          "somewhere in a %d-target grid." % (args.targets, args.targets))

    failures = []
    for robot in args.robots.split(","):
        for task in args.tasks.split(","):
            points, seconds, screen, spec = probe_scene(robot, task, args.draws, args.seed, args.scene)
            n_free = len(points)
            print("\n%s / %s   point = %s   scene = %s"
                  % (robot, task, spec.target_frame, os.path.basename(SceneFile(robot, task, args.scene))))
            print("  collision-free %d/%d (%.1f%%), %.3f ms per draw"
                  % (n_free, args.draws, 100.0 * n_free / args.draws,
                     1000.0 * seconds / args.draws))
            header = ("  %-7s %9s %9s %9s %9s" % ("inset", "in-shelf", "%of free", "%of raw",
                                                  "draws/tgt"))
            header += "".join(" %13s" % ("P(trip/%d)@%d" % (args.targets, g)) for g in guards)
            print(header)
            for inset in insets:
                regions = ShelfCompartmentRegions(inset)
                contained = [X for X in points
                             if PointInShelfCompartments(X.translation(), regions)]
                kept = ([X for X in contained if not screen.Penetrates([X])]
                        if screen is not None else contained)
                n = len(kept)
                p_raw = n / args.draws
                row = ("  %-7.3f %9d %8.3f%% %8.4f%% %9s"
                       % (inset, n, 100.0 * n / max(n_free, 1), 100.0 * p_raw,
                          ("%.0f" % (1.0 / p_raw)) if p_raw else "inf"))
                for g in guards:
                    if p_raw <= 0:
                        row += " %13s" % "1.00e+00"
                        trip = 1.0
                    else:
                        per_target = (1.0 - p_raw) ** g
                        trip = 1.0 - (1.0 - per_target) ** args.targets
                        row += " %13.2e" % trip
                    if trip >= 0.01:
                        failures.append((robot, task, inset, g, trip))
                print(row)
            if screen is not None:
                dropped = len([X for X in points if PointInShelfCompartments(
                    X.translation(), ShelfCompartmentRegions(insets[-1]))])
                print("  (penetration screen runs only on contained candidates: <= %d of %d)"
                      % (dropped, n_free))

    print("\n" + "=" * 78)
    if failures:
        print("FAIL -- these (robot, task, inset, guard) combinations trip >= 1%% of runs:")
        for robot, task, inset, guard, trip in failures:
            print("  %s/%s inset %.3f at guard %d: P(trip/%d) = %.3f"
                  % (robot, task, inset, guard, args.targets, trip))
        print("Either raise the guard for those items or do not field that inset.")
    else:
        print("OK -- every (inset, guard) combination probed stays under 1%%.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
