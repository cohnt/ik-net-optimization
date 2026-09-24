"""The soft continuum arm's definition: one dataclass per rung, and nothing else.

Every other module in `src/soft_arm/` derives from these numbers -- the torch kinematics,
the generated SDF, the jrl shim, the dataset sampler -- so the robot has exactly one
definition and a change here propagates rather than needing to be mirrored.

The arm is a spatial Piecewise Constant Strain (PCS) continuum manipulator, the family
SoRoMoX calls `soromox.systems.pcs.pcs.PCS`. Its configuration is the per-segment strain
twist, restricted to an active subset. SoRoMoX orders the twist angular-first as
`[kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]`; we use the same ordering and the
same indices, so a strain vector written here can be handed to SoRoMoX unchanged.

DEVIATION COORDINATES, SYMMETRIC ABOUT ZERO.  `sigma_z` is the *elongation*, not the
stretch ratio: the physical twist carries `1 + sigma_z`, so `sigma_z = 0` is the
unstretched rod and the origin of configuration space is the straight, unstretched arm.
That is not cosmetic. ikflow bakes `1 / max(|lo|, |hi|)` per coordinate into its first
`FixedLinearTransform` (`ikflow/model.py:311-316`) -- a pure scaling with NO offset -- so a
coordinate whose limits are not symmetric about zero hands the flow input its first layer
cannot recentre.

NORMALIZED CONFIGURATION.  What every formulation decides over, what the flow emits and
what the correction perturbs is `cfg = strain / limit`, in `[-1, 1]` per coordinate.
Physical strains appear only inside the kinematics.  This keeps the shared scalars --
`correction_bound`, the joint-centering `w * I`, the latent trust region -- meaning the
same thing on every coordinate and comparable with the rigid arms, where those scalars act
on radians of comparable range.  It is a stated adaptation, not a silent convention.
"""

from dataclasses import dataclass, field
from typing import Tuple

# Indices into SoRoMoX's angular-first strain twist.
KAPPA_X, KAPPA_Y, KAPPA_Z, SIGMA_X, SIGMA_Y, SIGMA_Z = range(6)

#: Human-readable names, indexed as above.
STRAIN_NAMES = ("kappa_x", "kappa_y", "kappa_z", "sigma_x", "sigma_y", "sigma_z")


@dataclass(frozen=True)
class SoftArmSpec:
    """One rung of the soft arm.

    `strain_basis` names the active strain indices *per segment*, so the configuration is
    `num_segments * len(strain_basis)` long, ordered segment-major:
    `[seg0_s0, seg0_s1, ..., seg1_s0, ...]`.

    `strain_limits` is the physical limit of each active strain, in the same order, and is
    symmetric: the admissible set is `[-limit, +limit]`.  Bending limits are in 1/m;
    the elongation limit is dimensionless.
    """

    name: str
    num_segments: int
    total_length: float                      # metres of backbone, summed over segments
    strain_basis: Tuple[int, ...]
    strain_limits: Tuple[float, ...]
    sublinks_per_segment: int                # `K`: the collision discretization
    backbone_radius: float                   # the rod itself
    collision_radius: float                  # the spheres that ARE the robot's geometry
    #: Where the gripper's mount frame sits relative to the last segment's tip.
    tip_frame_name: str = "soft_tip"

    def __post_init__(self):
        if len(self.strain_basis) != len(self.strain_limits):
            raise ValueError(
                f"{self.name}: {len(self.strain_basis)} active strains but "
                f"{len(self.strain_limits)} limits")
        if len(set(self.strain_basis)) != len(self.strain_basis):
            raise ValueError(f"{self.name}: duplicate entries in strain_basis")
        if any(limit <= 0.0 for limit in self.strain_limits):
            raise ValueError(f"{self.name}: strain limits must be positive (they are "
                             f"symmetric half-widths, not bounds)")
        if SIGMA_Z not in self.strain_basis:
            # Without an axial strain the rod cannot stretch, which is a legitimate robot
            # but not one this project fields; flag it rather than silently mis-scaling.
            raise ValueError(f"{self.name}: strain_basis must include SIGMA_Z")

    # -- shapes -----------------------------------------------------------------

    @property
    def strains_per_segment(self) -> int:
        return len(self.strain_basis)

    @property
    def ndof(self) -> int:
        return self.num_segments * self.strains_per_segment

    @property
    def segment_length(self) -> float:
        return self.total_length / self.num_segments

    @property
    def sublink_length(self) -> float:
        """Arc length of one collision sub-link, at zero elongation."""
        return self.segment_length / self.sublinks_per_segment

    @property
    def num_sublinks(self) -> int:
        return self.num_segments * self.sublinks_per_segment

    @property
    def num_plant_positions(self) -> int:
        """Drake positions: each sub-link is a quaternion floating body, 7 apiece."""
        return 7 * self.num_sublinks

    # -- limits -----------------------------------------------------------------

    @property
    def limits_per_dof(self) -> Tuple[float, ...]:
        """The physical limit of each of the `ndof` coordinates, segment-major."""
        return tuple(self.strain_limits) * self.num_segments

    @property
    def dof_names(self) -> Tuple[str, ...]:
        return tuple(f"seg{i}_{STRAIN_NAMES[s]}"
                     for i in range(self.num_segments) for s in self.strain_basis)

    def max_bend_angle_per_segment(self) -> float:
        """Largest bend a single segment can reach, in radians.

        The bending strains act as a vector, so the corner of the box is the worst case.
        Used to bound the sub-link's own rotation, which is what keeps the collision
        spheres covering the arc.
        """
        import math
        bend = [limit for index, limit in zip(self.strain_basis, self.strain_limits)
                if index in (KAPPA_X, KAPPA_Y)]
        if not bend:
            return 0.0
        kappa_max = math.sqrt(sum(b * b for b in bend))
        # A stretched segment is longer, so it bends through a larger angle.
        elongation = self.strain_limits[self.strain_basis.index(SIGMA_Z)]
        return kappa_max * self.segment_length * (1.0 + elongation)

    def sphere_coverage_margin(self) -> float:
        """How much room the collision spheres have, in metres, beyond covering the rod.

        Consecutive sphere centres are `sublink_length` apart along the backbone, so the
        union covers a rod of radius `r` when `sublink_length <= 2 * sqrt(R^2 - r^2)`.
        This returns the slack in that inequality; the containment test measures the real
        thing, this is the design-time bound.  Negative means the spheres leave gaps.
        """
        import math
        if self.collision_radius <= self.backbone_radius:
            return float("-inf")
        reach = 2.0 * math.sqrt(self.collision_radius ** 2 - self.backbone_radius ** 2)
        # A stretched segment spaces its sub-links further apart.
        elongation = self.strain_limits[self.strain_basis.index(SIGMA_Z)]
        return reach - self.sublink_length * (1.0 + elongation)

    def sagitta(self) -> float:
        """Worst-case bulge of one sub-arc away from the chord through its ends.

        Informational.  The collision spheres are centred ON the backbone, at the sub-link
        origins, so the bulge between two consecutive centres is already inside their
        union; what binds coverage is centre spacing, which `sphere_coverage_margin`
        measures.  This number says how far from a straight rod the discretization is.
        """
        import math
        bend = [limit for index, limit in zip(self.strain_basis, self.strain_limits)
                if index in (KAPPA_X, KAPPA_Y)]
        if not bend:
            return 0.0
        kappa_max = math.sqrt(sum(b * b for b in bend))
        elongation = self.strain_limits[self.strain_basis.index(SIGMA_Z)]
        ds = self.sublink_length * (1.0 + elongation)
        return kappa_max * ds * ds / 8.0


#: Bending limit, in 1/m.  Fixed so that the WORST case -- the corner of the (kappa_x,
#: kappa_y) box, on a fully stretched segment -- is exactly half a turn:
#:
#:     sqrt(2) * kappa_max * segment_length * (1 + sigma_z_max) = pi
#:
#: on the 12-DOF rung.  Per axis that is 97 degrees; the vector magnitude at the box
#: corner is sqrt(2) larger and stretch adds another 30%.  "No segment bends more than
#: half a turn" is the statement worth being able to make.
#:
#: Chosen from continuum-arm plausibility BEFORE any acceptance rate or cell count was
#: measured, and not revisited afterwards.  The limits are part of the robot: tuning them
#: to raise a shelf-acceptance rate or a success count would be making the question
#: easier rather than the method better.
_KAPPA_MAX = 8.5           # 1/m
_SIGMA_Z_MAX = 0.30        # 0.7x to 1.3x nominal length
_TOTAL_LENGTH = 0.80       # m of backbone, iiwa-like reach
_SUBLINK_TARGET = 0.0333   # m: the collision discretization, uniform across rungs

#: LIMITS ARE IDENTICAL ON EVERY RUNG, and that is the point of the ladder.  Total
#: backbone length is fixed at 0.8 m, so the total bend the arm can accumulate is
#: `kappa * total_length` regardless of how many segments that length is cut into.
#: Holding `kappa` fixed therefore holds the workspace envelope fixed, and the rungs
#: differ ONLY in how many independent segments -- how much redundancy -- the same arm
#: has.  The visible consequence is that a 3-segment rung bends further per segment
#: (240 degrees at the box corner, against 180) because its segments are longer.
_BEND = (_KAPPA_MAX, _KAPPA_MAX)

#: The primary rung, and the one the status-quo rows are measured on.
SOFT12 = SoftArmSpec(
    name="soft12",
    num_segments=4,
    total_length=_TOTAL_LENGTH,
    strain_basis=(KAPPA_X, KAPPA_Y, SIGMA_Z),
    strain_limits=(*_BEND, _SIGMA_Z_MAX),
    sublinks_per_segment=6,
    backbone_radius=0.025,
    collision_radius=0.035,
)

SOFT9 = SoftArmSpec(
    name="soft9",
    num_segments=3,
    total_length=_TOTAL_LENGTH,
    strain_basis=(KAPPA_X, KAPPA_Y, SIGMA_Z),
    strain_limits=(*_BEND, _SIGMA_Z_MAX),
    sublinks_per_segment=8,
    backbone_radius=0.025,
    collision_radius=0.035,
)

#: Adds torsion.  Torsion does not change the bending envelope -- it spins the frame about
#: the backbone -- so the reachable positions are the 12-DOF rung's, with four more
#: degrees of freedom to reach them and a genuinely torsional backbone.
SOFT16 = SoftArmSpec(
    name="soft16",
    num_segments=4,
    total_length=_TOTAL_LENGTH,
    strain_basis=(KAPPA_X, KAPPA_Y, KAPPA_Z, SIGMA_Z),
    strain_limits=(*_BEND, _KAPPA_MAX, _SIGMA_Z_MAX),
    sublinks_per_segment=6,
    backbone_radius=0.025,
    collision_radius=0.035,
)

RUNGS = {spec.name: spec for spec in (SOFT9, SOFT12, SOFT16)}


def GetSpec(name: str) -> SoftArmSpec:
    """The rung by name, with the available names in the error rather than a KeyError."""
    try:
        return RUNGS[name]
    except KeyError:
        raise ValueError(
            f"unknown soft-arm rung {name!r}; expected one of {sorted(RUNGS)}") from None
