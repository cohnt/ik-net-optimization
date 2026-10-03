"""Repair the joint limits Drake's parsers silently drop for a screw joint.

THE TRAP. `ParseJointLimits` is reached only for revolute and prismatic joints -- in the
URDF parser at `detail_urdf_parser.cc:630,656`, and the SDFormat path is the same -- so a
screw joint's `<limit>` element is read and discarded, and `plant.GetPositionLowerLimits()`
reports `[-inf, inf]` on that coordinate. Measured on this project's own model in both
formats.

WHAT IT COSTS IF IT IS NOT REPAIRED, and none of it is a crash:

  * the joint-limit constraint row goes vacuous on that coordinate, so the one row this
    whole robot exists to stress is not enforced;
  * the joint-space arm's `BoundingBoxConstraint` becomes unbounded, and IPOPT is poorly
    behaved on unbounded variables -- the reason the latent trust region is kept;
  * `SampleConfiguration`'s `rng.uniform(lower, upper)` returns `nan`, and the target
    sampler then spins forever rather than failing.

`set_position_limits` works only AFTER `Finalize()`, and `src/utils.py:BuildEnv` finalizes
internally, so the program's `__init__` is the first place this can run. It must also run
BEFORE `ToAutoDiffXd()`: that call makes an independent scalar-converted COPY of the plant,
and a copy taken before the repair carries the infinities for ever with nothing downstream
to say so.
"""

import numpy as np


def ApplyScrewJointLimits(plant, spec):
    """Set the limits the parser discarded. Idempotent; returns `{joint: (lo, hi)}`.

    Idempotence matters because the pose task shares one diagram between the sampler program
    and every cell's program, while the grasp task rebuilds the diagram per target -- so
    this runs once per plant on one path and many times on the other.
    """
    applied = {}
    for joint in spec.joints:
        if joint.kind != "screw":
            continue
        handle = plant.GetMutableJointByName(joint.name)
        handle.set_position_limits(np.array([joint.lower]), np.array([joint.upper]))
        applied[joint.name] = (joint.lower, joint.upper)
    return applied


def RequireFiniteLimits(plant, what="plant"):
    """Raise unless every position limit is finite.

    Belt and braces after `ApplyScrewJointLimits`, and the check that would have caught the
    trap in the first place. A non-finite limit does not crash anything -- it quietly turns
    a uniform draw into `nan` and a constraint row into noise -- so it has to be asserted
    rather than waited for.
    """
    lower = np.asarray(plant.GetPositionLowerLimits(), dtype=float)
    upper = np.asarray(plant.GetPositionUpperLimits(), dtype=float)
    bad = ~(np.isfinite(lower) & np.isfinite(upper))
    if bad.any():
        raise RuntimeError(
            f"{what}: position limits are non-finite at indices {np.flatnonzero(bad).tolist()} "
            f"(lower={lower.tolist()}, upper={upper.tolist()}). A screw joint's <limit> is "
            f"discarded by Drake's parsers; call ApplyScrewJointLimits on the plant before "
            f"ToAutoDiffXd().")
    return lower, upper
