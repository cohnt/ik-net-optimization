"""The GVS push-rod arm's definition: one frozen dataclass per rung, and nothing else.

Every other module in `src/gvs_arm/` derives from these numbers -- the SoRoMoX model the
solver evaluates, the generated Drake model, the jrl shim, the dataset sampler -- so the
robot has exactly one definition and a change here propagates rather than needing to be
mirrored.

WHAT THE ROBOT IS.  A spatial continuum arm whose backbone is a Geometric Variable Strain
(GVS) rod -- the strain along each segment is a Legendre polynomial of `basis_order` in the
normalized arc length, of which the soft PCS arm's constant strain is the order-0 member --
driven by PUSH-PULL RODS: three per segment at 120 degrees, each routed at a fixed fraction
of the local radius and acting only within its own segment (the way LOInK's HSA segments
carry their own actuators, arXiv 2609.21275 sec. VII). The backbone TAPERS from `radius_base`
to `radius_tip`, and that is load-bearing rather than cosmetic: with a uniform section, a
straight-routed rod applies a uniform moment and the generalized stiffness is diagonal in
the Legendre basis, so every coefficient above order 0 is EXACTLY zero at equilibrium and the
order ladder would be vacuous. `EI(s) ~ r(s)^4` is what makes the strain genuinely variable.
Measured on the prototype: a differential rod force at `F_max` gives a Legendre-1 curvature
coefficient of -2.3 /m on the tapered rod and 0.000000 on a uniform one.

THE CONFIGURATION IS THE ROD FORCE VECTOR, NOT THE STRAIN.  The decision variables of every
formulation are the nine rod forces; the backbone strain is whatever static equilibrium
SoRoMoX's model reaches under them (`src/gvs_arm/model.py`). So the forward map is a
root-find on top of a numerically integrated rod, with no closed form anywhere in it -- the
setting LOInK calls case (b), and the reason this robot exists. The kinematic redundancy is
in the INPUTS: nine forces against a 6-D pose task, three spare, six spare on the grasp task
(Thomas: "at least one degree of kinematic redundancy"; `__post_init__` enforces it).

NORMALIZED, SYMMETRIC INPUTS.  What every formulation decides over, what the flow emits and
what the correction perturbs is `cfg = u / F_max` per rod, in `[-1, 1]`: a push-pull rod
carries force of either sign, so the box is symmetric and its centre `cfg = 0` is the
straight, unstressed arm -- which makes the joint-centering cost the effort cost LOInK
minimises (`||u||^2`, weighted by each segment's force scale). ikflow bakes
`1 / max(|lo|, |hi|)` per coordinate into its first `FixedLinearTransform` with NO offset, so
an off-centre box would hand the flow input its first layer cannot recentre; a tension-only
tendon box `[0, F_max]` would have put the rest arm at a corner.

`F_max` IS DERIVED, FROM ONE STATED RULE.  Per segment, the force at which the order-0
differential bend of that segment (one rod pushing at `+F`, the other two pulling at
`-F/2`) reaches `design_curvature = 8.5 /m` -- the soft PCS arm's bending limit, so the two
robots share an envelope statement: `F_max_i = kappa * E * I(s_mid) / (1.5 * d(s_mid))`.
Fixed from plausibility BEFORE any acceptance rate was measured, like the PCS limits, and
not revisited afterwards: tuning limits to raise a success count makes the question easier
rather than the method better.

CONVENTIONS ARE SOROMOX'S, DELIBERATELY.  Strains are named in SoRoMoX's vocabulary --
`kappa_y`, `kappa_z` bend, `sigma_x` stretches -- because SoRoMoX's backbone is its local
x-axis (`sigma_x = 1` is the straight reference) and with its default upright mounting the
backbone points along world +z, exactly where every scene here welds a robot. Nothing in
this package rotates a frame; the model that runs IS the model that was specified.
"""

import math
from dataclasses import dataclass
from typing import Tuple

#: SoRoMoX's angular-first strain twist, local x along the backbone.
STRAIN_NAMES = ("kappa_x", "kappa_y", "kappa_z", "sigma_x", "sigma_y", "sigma_z")

#: The task the redundancy is counted against: a full 6-D pose.
_TASK_DIMENSION = 6


@dataclass(frozen=True)
class GvsArmSpec:
    """One rung of the GVS push-rod arm.

    The rungs differ ONLY in `basis_order`, the fidelity of the backbone's strain field.
    The inputs (nine rod forces), the taper, the material and the force scales are identical,
    so the ladder measures what the forward model's order does to the SAME optimization
    problem -- not a change of problem dimension.
    """

    name: str
    basis_order: int
    num_segments: int = 3
    total_length: float = 0.80                 # metres of backbone, summed over segments
    radius_base: float = 0.030                 # the rod itself, at the base
    radius_tip: float = 0.015                  # ... and at the tip; linear in between
    active_strains: Tuple[str, ...] = ("kappa_y", "kappa_z", "sigma_x")
    basis: str = "legendre"
    num_gauss_points: int = 7                  # SoRoMoX's per-segment quadrature (>= 5)
    young_modulus: float = 1e5                 # Pa, silicone-like
    poisson_ratio: float = 0.45
    density: float = 1000.0                    # kg/m^3; recorded, inert with gravity off
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rods_per_segment: int = 3
    rod_azimuths_deg: Tuple[float, ...] = (0.0, 120.0, 240.0)
    rod_offset_ratio: float = 0.7              # rod at 0.7 r(s), inside the backbone
    design_curvature: float = 8.5              # 1/m, the PCS arm's bending limit
    sublinks_per_segment: int = 12             # collision sub-links per segment
    collision_radius_ratio: float = 1.4        # sphere radius = ratio * local r(s)
    tip_frame_name: str = "gvs_tip"

    def __post_init__(self):
        if self.basis_order < 0:
            raise ValueError(f"{self.name}: basis_order must be >= 0")
        if self.num_gauss_points < 5:
            raise ValueError(f"{self.name}: SoRoMoX's GVS needs >= 5 Gauss points per segment")
        if self.radius_base <= 0.0 or self.radius_tip <= 0.0:
            raise ValueError(f"{self.name}: radii must be positive")
        if len(set(self.rod_azimuths_deg)) != self.rods_per_segment:
            raise ValueError(f"{self.name}: need {self.rods_per_segment} distinct rod azimuths")
        for strain in self.active_strains:
            if strain not in STRAIN_NAMES:
                raise ValueError(f"{self.name}: unknown strain {strain!r}")
        if "sigma_x" not in self.active_strains:
            raise ValueError(f"{self.name}: the backbone must be able to stretch (sigma_x)")
        if self.ninputs <= _TASK_DIMENSION:
            raise ValueError(
                f"{self.name}: {self.ninputs} inputs against a {_TASK_DIMENSION}-D task leaves "
                f"no kinematic redundancy; the comparison needs at least one spare degree")
        if not 0.0 < self.rod_offset_ratio < 1.0:
            raise ValueError(f"{self.name}: rods must sit inside the backbone")

    # -- shapes -----------------------------------------------------------------

    @property
    def ninputs(self) -> int:
        """The decision-variable count of every formulation: one force per rod."""
        return self.num_segments * self.rods_per_segment

    @property
    def ndof(self) -> int:
        """Alias for `ninputs`: what the rest of the project calls the configuration width."""
        return self.ninputs

    @property
    def backbone_dof(self) -> int:
        """SoRoMoX's generalized coordinates: strains x (order + 1) per segment."""
        return self.num_segments * len(self.active_strains) * (self.basis_order + 1)

    @property
    def segment_length(self) -> float:
        return self.total_length / self.num_segments

    @property
    def sublink_length(self) -> float:
        """Arc length of one collision sub-link on the unstretched rod."""
        return self.segment_length / self.sublinks_per_segment

    @property
    def num_sublinks(self) -> int:
        return self.num_segments * self.sublinks_per_segment

    @property
    def num_bodies(self) -> int:
        """Floating bodies: one per sub-link, plus the tip mount (see the PCS arm)."""
        return self.num_sublinks + 1

    @property
    def num_plant_positions(self) -> int:
        """Drake positions: every body is a quaternion floating body, 7 apiece."""
        return 7 * self.num_bodies

    @property
    def tip_link_name(self) -> str:
        return "gvs_tip_link"

    def body_names(self) -> Tuple[str, ...]:
        """The floating bodies, in the order the generated SDF declares them."""
        return tuple(
            [f"seg{i}_sub{j}" for i in range(self.num_segments)
             for j in range(self.sublinks_per_segment)] + [self.tip_link_name])

    def body_arclengths(self) -> Tuple[float, ...]:
        """Where each body's origin sits on the unstretched backbone, tip included."""
        return tuple([j * self.sublink_length for j in range(self.num_sublinks)]
                     + [self.total_length])

    def body_radii(self) -> Tuple[float, ...]:
        """Each body's collision sphere follows the local radius of the taper."""
        return tuple(self.collision_radius_ratio * self.radius_at(s)
                     for s in self.body_arclengths())

    def filter_group_sizes(self) -> Tuple[int, ...]:
        """Collision bodies per segment group; the tip joins the last segment."""
        sizes = [self.sublinks_per_segment] * self.num_segments
        sizes[-1] += 1
        return tuple(sizes)

    def sublinks_below_table(self) -> int:
        """How many leading collision spheres reach below the table top at z = 0.

        The arm is mounted THROUGH the table surface, so the spheres whose centres sit
        within their own radius of z = 0 are inside it by construction and are filtered
        against the tables. Derived so that changing the sub-link count or the radius cannot
        silently leave one unfiltered.
        """
        count = 0
        for s, radius in zip(self.body_arclengths()[:-1], self.body_radii()[:-1]):
            if s < radius:
                count += 1
        return max(1, count)

    # -- geometry and material ---------------------------------------------------

    def radius_at(self, s: float) -> float:
        """The backbone radius at arc length `s`, linear from base to tip."""
        return self.radius_base + (self.radius_tip - self.radius_base) * s / self.total_length

    def segment_radii(self, segment: int) -> Tuple[float, float]:
        """`(base, tip)` radius of one segment: SoRoMoX's `LinearProfile` per link."""
        s0 = segment * self.segment_length
        return self.radius_at(s0), self.radius_at(s0 + self.segment_length)

    def rod_offset_at(self, s: float) -> float:
        return self.rod_offset_ratio * self.radius_at(s)

    @property
    def rod_azimuths(self) -> Tuple[float, ...]:
        return tuple(math.radians(a) for a in self.rod_azimuths_deg)

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def bending_stiffness_at(self, s: float) -> float:
        """`E * I` of the circular section at `s`."""
        return self.young_modulus * math.pi * self.radius_at(s) ** 4 / 4.0

    def axial_stiffness_at(self, s: float) -> float:
        """`E * A` of the circular section at `s`."""
        return self.young_modulus * math.pi * self.radius_at(s) ** 2

    # -- the force scales ---------------------------------------------------------

    def force_limit(self, segment: int) -> float:
        """`F_max` of one segment's rods, from the design-curvature rule in the docstring.

        One rod at `+F` with the other two at `-F/2` is a pure moment of magnitude `1.5 F d`
        about the backbone (the three azimuths sum to zero), so the order-0 curvature of a
        uniform-moment segment is `1.5 F d / EI`. Evaluated at the segment's mid-section,
        since the taper makes both `d` and `EI` vary along it.
        """
        s_mid = (segment + 0.5) * self.segment_length
        return (self.design_curvature * self.bending_stiffness_at(s_mid)
                / (1.5 * self.rod_offset_at(s_mid)))

    @property
    def force_limits(self) -> Tuple[float, ...]:
        """`F_max` per INPUT, segment-major, rods within a segment identical."""
        return tuple(self.force_limit(i)
                     for i in range(self.num_segments) for _ in range(self.rods_per_segment))

    def max_axial_strain(self, segment: int) -> float:
        """Axial strain under three rods at `+F_max`, order 0, mid-section.

        `3 F_max / EA = design_curvature * r / (2 * rod_offset_ratio)` after substitution --
        0.17 at the base segment, 0.11 at the tip. This is what sets the worst sub-link
        spacing for the sphere coverage margin.
        """
        s_mid = (segment + 0.5) * self.segment_length
        return 3.0 * self.force_limit(segment) / self.axial_stiffness_at(s_mid)

    @property
    def input_names(self) -> Tuple[str, ...]:
        return tuple(f"seg{i}_rod{k}" for i in range(self.num_segments)
                     for k in range(self.rods_per_segment))

    dof_names = input_names

    # -- the collision discretization --------------------------------------------

    def sphere_coverage_margin(self) -> float:
        """Slack in the centre SPACING at the worst sub-link, in metres.

        Consecutive sphere centres are `sublink_length * (1 + axial strain)` apart, and a
        rod of radius `r` is covered when that spacing is at most `2 sqrt(R^2 - r^2)`. The
        tip segment is the worst case: smallest radius, largest stretch relative to its
        spacing. Negative means the spheres certainly leave gaps; the containment test
        (`tests/test_gvs_arm_model.py`) measures the fact on the rod's surface.
        """
        worst = float("inf")
        for segment in range(self.num_segments):
            s_end = (segment + 1) * self.segment_length
            r = self.radius_at(s_end)
            R = self.collision_radius_ratio * r
            if R <= r:
                return float("-inf")
            reach = 2.0 * math.sqrt(R * R - r * r)
            spacing = self.sublink_length * (1.0 + self.max_axial_strain(segment))
            worst = min(worst, reach - spacing)
        return worst

    @property
    def latent_radius(self) -> float:
        """The trust-region convention every robot here follows: `sqrt(width) + 1.5`."""
        return round(math.sqrt(self.ninputs) + 1.5, 2)


#: The order ladder. Same inputs, same taper, same material and force scales on every rung.
GVS_PUSHROD9_O0 = GvsArmSpec(name="gvs_pushrod9_o0", basis_order=0)
GVS_PUSHROD9_O1 = GvsArmSpec(name="gvs_pushrod9_o1", basis_order=1)
GVS_PUSHROD9_O2 = GvsArmSpec(name="gvs_pushrod9_o2", basis_order=2)

RUNGS = {spec.name: spec for spec in (GVS_PUSHROD9_O0, GVS_PUSHROD9_O1, GVS_PUSHROD9_O2)}

#: Pre-registered before any cell is read: order 1 is the headline rung, order 2 the
#: fidelity check above it, order 0 the constant-strain control inside the same family.
PRIMARY = "gvs_pushrod9_o1"


def GetSpec(name: str) -> GvsArmSpec:
    """The rung by name, with the available names in the error rather than a KeyError."""
    try:
        return RUNGS[name]
    except KeyError:
        raise ValueError(
            f"unknown GVS arm rung {name!r}; expected one of {sorted(RUNGS)}") from None
