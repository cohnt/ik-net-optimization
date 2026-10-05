"""How fast does the dataset sampler run? Seconds per sample through the batched equilibrium.

The dataset is `sample_joint_angles_and_poses`: uniform rod forces, a vmapped Newton solve
per draw, the tip pose, and the sphere self-collision screen. Every sample is a root-find,
so this is not the 14 us/config the soft PCS arm's closed form cost, and the size of the
dataset (`DATASET_SIZE`, ikflow's default 25M; LOInK used 2M) and the job's wall time have to
be set from a measurement rather than assumed. Measured at several batch sizes because the
vmap runs every lane to the slowest lane's convergence.

Runs at the lowest scheduling priority and with the thread count given, so it can share the
machine; the number it reports is then an UPPER bound on the per-sample cost.

    GVS_ARM_XLA_THREADS=8 .venv/bin/python scripts/gvs_arm/probe_datagen_rate.py --batches 2000,20000
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

## Low priority when sharing a laptop; a no-op cost on a dedicated node.
os.nice(19)

import src.register_robots  # noqa: E402,F401
from jrl.robots import get_robot  # noqa: E402
from src.gvs_arm.params import PRIMARY, RUNGS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rung", choices=sorted(RUNGS), default=PRIMARY)
    p.add_argument("--batches", default="2000,20000")
    p.add_argument("--repeats", type=int, default=2)
    args = p.parse_args()

    robot = get_robot(args.rung)
    threads = os.environ.get("GVS_ARM_XLA_THREADS", "all")
    print(f"{args.rung}: {robot.ndof} inputs, XLA threads = {threads}")
    for batch in (int(x) for x in args.batches.split(",")):
        ## First call compiles for this batch shape; report the steady state.
        robot.sample_joint_angles_and_poses(batch, only_non_self_colliding=True)
        times = []
        for _ in range(args.repeats):
            start = time.time()
            samples, poses = robot.sample_joint_angles_and_poses(
                batch, only_non_self_colliding=True)
            times.append(time.time() - start)
        per = min(times) / batch
        print(f"  batch {batch:>6}: {min(times):7.2f} s  ->  {per * 1e6:8.1f} us/sample; "
              f"25M would take {25e6 * per / 3600:6.1f} h, 2M {2e6 * per / 3600:5.2f} h; "
              f"rejected unconverged so far: {robot.rejected_unconverged}")
    print(f"  self-collision rate: {np.mean(robot.config_self_collides(np.asarray(samples)).numpy()):.4f} "
          f"among kept samples (should be 0), sample shape {samples.shape}, poses {poses.shape}")


if __name__ == "__main__":
    main()
