"""The `helix7` arm: one definition, from which everything else is rendered.

A 7-DoF spherical-revolute-spherical arm whose **upper-arm roll is a helical (screw)
pair** -- the upper arm telescopes as it rolls, at a fixed pitch, driven by one
coordinate. That is what takes the arm out of the algebraic class: a revolute joint
contributes `cos q` and `sin q`, which the tangent half-angle substitution makes
rational, but a screw joint contributes `cos q`, `sin q` AND `q`, and `q` is
algebraically independent of `exp(i q)`. Abban, Li and Schicho (arXiv:1312.1060) put it
plainly: algebraic methods "have failed so far for the study of linkages with helical
joints ... because of the presence of some non-algebraic relations".

WHY THE ROBOT IS INVENTED. No real arm with a lone helical joint exists, in hardware or
in any public model, and the reason is structural rather than accidental: a lone helical
pair makes the drive torque and the load's reaction torque the same torque, so every real
screw actuator either grounds the nut against rotation (becoming a prismatic joint behind
a gearbox) or adds a co-axial second motor (becoming a CYLINDRICAL pair, which
re-coordinatises back to an algebraic inverse kinematics by an invertible linear map).
Every SCARA ball-screw-spline shaft is the latter; the spline groove exists precisely to
decouple rotation from translation. The joint itself is an ordinary machine element --
a THK ball-screw/spline quill in "spiral mode", spline nut driven and screw nut braked,
is a lone helical pair at the ball screw's lead -- and it is first-class in Drake, DART,
Simbody, Pinocchio and Simscape. Nobody has put one in an arm. That is the finding; this
file is the response to it.

WHY FROM SCRATCH. The alternative was to declare one joint of an existing benchmark arm a
screw. That is neither real nor clean: it carries a real robot's name, geometry and
published identity while no longer being that robot, so every table is ambiguous about
what was measured. This module is the robot, in the soft arm's idiom -- `generate_sdf.py`
and `kinematics.py` are two RENDERINGS of it, and a test says they agree to 1e-12.

THE FAMILY. Four rungs sharing every number except the pitch, so the pitch ladder is a
dose-response on one robot rather than four robots. `helix7_p000` is a full member with
pitch 0, not a code path taking a different branch: a control that runs different code is
not a control. At pitch 0 the arm is a plain S-R-S arm, for which the closed form is
standard -- so the family contains its own algebraic member, and the ladder runs from "an
analytic column could exist here" to "no analytic column can exist here".

UNITS. `pitch` is METRES PER REVOLUTION, matching Drake's `ScrewJoint::screw_pitch` and
SDFormat's `<screw_thread_pitch>`: the joint coordinate is the ANGLE in radians and the
translation is `pitch * q / (2 * pi)`. Do not copy a pitch in from another library --
Pinocchio's `JointModelHelical` uses metres per RADIAN (a factor of 2*pi), and SDFormat's
deprecated `<thread_pitch>` is radians per metre with the opposite handedness.
"""

import math
from dataclasses import dataclass
from typing import Tuple

#: Metres per revolution -> metres per radian.
TWO_PI = 2.0 * math.pi

#: Sphere spacing along a link's axis, in metres. Drake's collision geometry is a union of
#: spheres rather than a capsule because CAPSULES HANG DRAKE'S PROXIMITY ENGINE -- measured
#: on the soft arm, where a 6-body capsule model with ZERO candidate collision pairs did not
#: complete one `MinimumDistanceLowerBoundConstraint` evaluation in minutes while the
#: identical sphere model evaluated in microseconds. The tree's own
#: `iiwa14_spheres_cylinders_collision.urdf` had already reached the same conclusion.
SPHERE_SPACING = 0.05


@dataclass(frozen=True)
class JointSpec:
    """One joint, in URDF's convention: `origin` places the joint frame in the PARENT link.

    `kind` is `"revolute"` or `"screw"`. `pitch` is metres per revolution and is meaningful
    only for a screw joint; it is carried here rather than on the arm so that a future spec
    can hold more than one screw at more than one pitch.
    """

    name: str
    kind: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    lower: float
    upper: float
    pitch: float = 0.0

    @property
    def metres_per_radian(self) -> float:
        return self.pitch / TWO_PI


@dataclass(frozen=True)
class LinkSpec:
    """One link, and the segment its collision spheres are strung along.

    `a` and `b` are in the LINK'S OWN frame. The spheres are placed along `a -> b` at
    `SPHERE_SPACING`, inclusive of both ends, all of radius `radius`.

    THE SPHERE UNION IS THE ROBOT'S COLLISION GEOMETRY -- declared, not an approximation of
    some underlying rod, exactly as on the soft arm. That matters twice over. Drake's
    collision constraint is then EXACT on the robot as defined, with no containment argument
    to make or test; and the self-collision screen the dataset runs can use the very same
    spheres, so the dataset and the solver cannot be describing different robots. It also
    means there is no capsule anywhere in this robot, which is the point -- see
    `SPHERE_SPACING`.
    """

    name: str
    a: Tuple[float, float, float]
    b: Tuple[float, float, float]
    radius: float
    mass: float = 1.0

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)

    def sphere_centres(self) -> Tuple[Tuple[float, float, float], ...]:
        """Centres along `a -> b`, inclusive of both ends, at most `SPHERE_SPACING` apart."""
        n = max(1, math.ceil(self.length / SPHERE_SPACING))
        return tuple(
            tuple(self.a[k] + (self.b[k] - self.a[k]) * (i / n) for k in range(3))
            for i in range(n + 1)
        )


## ---------------------------------------------------------------------------------------
## The geometry. Shared by every rung: the rungs differ ONLY in the screw joint's pitch, so
## the ladder measures the coupling and not the arm.
##
## Offsets are chosen so the reach band matches the iiwa14 and the Panda (1.31 m of flange
## height fully extended, 0.89 m horizontal from the shoulder against the iiwa's ~0.95),
## which is what lets the existing shelf welds, the two tables, the finray gripper and the
## whole of `src/shelf_regions.py` apply to this robot untouched. A robot that needed its own
## furniture would not be comparable with the record's rows.
## ---------------------------------------------------------------------------------------

#: The upper arm is a TELESCOPING TUBE running through a collar, which is what a helical
#: joint at an upper-arm roll physically means: `upper_housing` is fixed to the shoulder and
#: `upper_arm` rolls and slides through it. The tube's 0.06 m tail is what keeps it captive
#: -- at full extension it still overlaps the collar by 0.07 m -- and that overlap, together
#: with the tube's tail clearing the pedestal, is what fixes the joint's range and the
#: primary pitch together. The geometry and the stroke are one decision, and both are part
#: of the robot.
LINKS = (
    LinkSpec("base_link",     (0.0, 0.0, 0.00),  (0.0, 0.0, 0.20), 0.085, mass=8.0),
    LinkSpec("shoulder",      (0.0, 0.0, -0.09), (0.0, 0.0, 0.09), 0.080, mass=4.0),
    LinkSpec("upper_housing", (0.0, 0.0, -0.02), (0.0, 0.0, 0.12), 0.072, mass=3.0),
    LinkSpec("upper_arm",     (0.0, 0.0, -0.06), (0.0, 0.0, 0.36), 0.055, mass=2.8),
    LinkSpec("forearm",       (0.0, 0.0, 0.00),  (0.0, 0.0, 0.36), 0.055, mass=2.4),
    LinkSpec("wrist",         (0.0, 0.0, -0.05), (0.0, 0.0, 0.05), 0.052, mass=1.2),
    LinkSpec("wrist_pitch",   (0.0, 0.0, 0.00),  (0.0, 0.0, 0.06), 0.048, mass=0.8),
    LinkSpec("flange",        (0.0, 0.0, 0.00),  (0.0, 0.0, 0.07), 0.045, mass=0.6),
)

#: The joint table. Limits are in radians; the screw joint's span TWO FULL REVOLUTIONS,
#: which is deliberate -- many `q3` differing by `2*pi` give the same rotation and a
#: different extension, so the solution set is richly multimodal, which is the property a
#: normalizing flow is supposed to capture.
#:
#: The range is SYMMETRIC about zero, `+-2*pi`, and that is a constraint the chart imposes
#: rather than the mechanism. With `sigmoid_on_output` false -- the configuration every
#: checkpoint in this project is trained at -- ikflow's first layer is
#: `x_i / max(|lo_i|, |hi_i|)`, a PURE SCALING with no offset (`ikflow/model.py`, the
#: `else` branch). A one-sided `[0, 4*pi]` range has the same stroke but lands that
#: coordinate in `[0, 1]` instead of `[-1, 1]`, spending half its input range and handing
#: the flow an offset its first layer cannot remove. The stroke is therefore centred on the
#: nominal extension, which a telescoping actuator is just as entitled to be.
#:
#: Every other joint sits in the band the rigid arms use.
SCREW_JOINT_NAME = "upper_arm_screw"

_JOINTS_AT_ZERO_PITCH = (
    JointSpec("base_yaw", "revolute", "base_link", "shoulder",
              (0.0, 0.0, 0.42), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.90, 2.90),
    JointSpec("shoulder_pitch", "revolute", "shoulder", "upper_housing",
              (0.0, 0.0, 0.00), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.00, 2.00),
    JointSpec(SCREW_JOINT_NAME, "screw", "upper_housing", "upper_arm",
              (0.0, 0.0, 0.06), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
              -2.0 * math.pi, 2.0 * math.pi),
    JointSpec("elbow", "revolute", "upper_arm", "forearm",
              (0.0, 0.0, 0.36), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.60, 2.60),
    JointSpec("forearm_roll", "revolute", "forearm", "wrist",
              (0.0, 0.0, 0.36), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.90, 2.90),
    JointSpec("wrist_pitch_joint", "revolute", "wrist", "wrist_pitch",
              (0.0, 0.0, 0.00), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.00, 2.00),
    JointSpec("wrist_roll", "revolute", "wrist_pitch", "flange",
              (0.0, 0.0, 0.06), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.00, 3.00),
)

#: How far apart two links must be along the chain before the pair is CHECKED. Neighbours
#: meet at their shared joint by construction; a link two apart is separated only by a short
#: housing or collar that both overlap (the telescoping tube and its collar interpenetrate
#: by design, and so do the pedestal and the shoulder); and the WRIST is a compact
#: three-link spherical joint spanning 0.11 m, so links three apart across it are physically
#: neighbours too -- the forearm's last sphere and the flange's first sit 0.06 m apart with
#: radii summing to 0.10. Checking any of those would reject every configuration, exactly as
#: Drake filters joint-adjacent bodies and as jrl's own robots carry a hand-curated
#: never-colliding list.
#:
#: A rule beats a curated list here because the arm is ours: we can build it so the rule is
#: true, and a test asserts it at the home configuration and across the whole screw stroke.
#: What the rule keeps is what matters -- the arm folding onto its own pedestal
#: (`base_link` against `forearm` and everything beyond it) and the wrist driven back into
#: the shoulder -- and those pairs are measured to come within a millimetre of contact over
#: the configuration box, so they are live constraints rather than decoration.
COLLISION_CHAIN_GAP = 4

#: Link-index pairs the self-collision screen checks. If the screen and the URDF's filter
#: groups ever disagree, the dataset would be screened against a different robot than the
#: solver sees, so both are generated from this one tuple.
COLLISION_PAIRS = tuple(
    (i, j)
    for i in range(len(LINKS))
    for j in range(i + COLLISION_CHAIN_GAP, len(LINKS))
)

#: The one pair the rule filters that is nonetheless expected NEVER to touch. Everything
#: else the rule drops interpenetrates by design -- the tube through its collar, the tube
#: through the shoulder, the forearm's end against the flange across the compact wrist --
#: but `base_link` against `upper_arm` is dropped only because they are three links apart,
#: and it stays clear for a reason that is part of the robot: the screw's range is
#: ONE-SIDED, so the tube's tail never travels back down towards the pedestal. A test
#: asserts it over the whole configuration box. If it ever fails, that is the geometry
#: saying the stroke reaches somewhere it should not, and it is a finding rather than a
#: test to loosen.
FILTERED_BUT_CLEAR = (("base_link", "upper_arm"),)


@dataclass(frozen=True)
class HelixArmSpec:
    """One rung: the geometry above at one screw pitch."""

    name: str
    pitch: float                      #: metres per revolution

    def __post_init__(self):
        if not math.isfinite(self.pitch):
            raise ValueError(f"{self.name}: pitch must be finite, got {self.pitch!r}")
        if self.pitch < 0.0:
            raise ValueError(
                f"{self.name}: pitch must be non-negative. A left-handed thread is a "
                f"different robot, not a lower rung of this ladder; give it its own spec.")

    ## -- identity -------------------------------------------------------------------

    @property
    def base_link(self) -> str:
        return LINKS[0].name

    @property
    def flange_link(self) -> str:
        return LINKS[-1].name

    @property
    def links(self) -> Tuple[LinkSpec, ...]:
        return LINKS

    @property
    def joints(self) -> Tuple[JointSpec, ...]:
        """The joint table with this rung's pitch substituted into the screw joint."""
        return tuple(
            JointSpec(j.name, j.kind, j.parent, j.child, j.origin_xyz, j.origin_rpy,
                      j.axis, j.lower, j.upper,
                      self.pitch if j.kind == "screw" else 0.0)
            for j in _JOINTS_AT_ZERO_PITCH
        )

    @property
    def ndof(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def screw_joint_names(self) -> Tuple[str, ...]:
        return tuple(j.name for j in self.joints if j.kind == "screw")

    @property
    def limits(self):
        """`(lower, upper)` as plain tuples, in joint order."""
        return (tuple(j.lower for j in self.joints), tuple(j.upper for j in self.joints))

    @property
    def travel(self) -> float:
        """The screw joint's full axial stroke in metres -- the size of the coupling."""
        return max(((j.upper - j.lower) * j.metres_per_radian
                    for j in self.joints if j.kind == "screw"), default=0.0)

    @property
    def latent_trust_region(self) -> float:
        """The convention the rigid arms and the soft arm share: `sqrt(dim) + 1.5`."""
        return round(math.sqrt(self.ndof) + 1.5, 2)


## The ladder. The PRIMARY rung is pre-registered here, before any acceptance rate or cell
## count has been read, exactly as the soft arm's `n6` was: 0.050 m/rev telescopes the upper
## arm by +-100 mm, a quarter of its length, which is structural rather than a perturbation
## while leaving 0.10 m of tube overlap at the stroke ends. The pitch is part of the robot.
SPECS = {
    s.name: s for s in (
        HelixArmSpec("helix7_p000", 0.000),
        HelixArmSpec("helix7_p025", 0.025),
        HelixArmSpec("helix7_p050", 0.050),
        HelixArmSpec("helix7_p100", 0.100),
    )
}

PRIMARY = "helix7_p050"


def GetSpec(name: str) -> HelixArmSpec:
    if name not in SPECS:
        raise KeyError(f"no such helix rung {name!r}; known: {sorted(SPECS)}")
    return SPECS[name]


def RungNames():
    """Sorted rung names, for `--robot` choices and jrl registration."""
    return sorted(SPECS)
