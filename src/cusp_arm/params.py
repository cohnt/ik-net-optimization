"""The `cusp7` arm: one definition, from which everything else is rendered.

A 7-DoF arm built from a CUSPIDAL 6R plus one added joint at the wrist end. A cuspidal
manipulator can move between two inverse-kinematics solutions of the same pose without
crossing a singularity, so one aspect (a singularity-free region of joint space) holds
several solutions and no branch label is global: any label must jump somewhere the arm is
regular. That is the property under test -- an analytic chart has to pick branches, a
learned latent does not.

THE 6R IS PUBLISHED, NOT OURS. It is the simplified "Transpressor" class of Feeß and Weiß,
*Inverse kinematics of a cuspidal 6R robot* (arXiv 2602.14794, 2026): a research
simplification of the KUKA Transpressor (a KR100P with axis 6 shifted parallel off the wrist
centre), simplified so axes 1/2 and 3/4 intersect. Standard (distal) DH,
`T_{i+1} = T_i Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)`:

    i   d    a    alpha
    1   d1   0    pi/2
    2   0    a2   0
    3   0    0    -pi/2
    4   d4   0    pi/2
    5   0    a5   -pi/2
    6   0    0    0

The paper's example values `a2 = 3, d4 = 2, a5 = 1/2` (unitless) are used, scaled
uniformly by `SCALE` to metres: a uniform scale changes neither the solution count nor
cuspidality. Its inverse kinematics has 0/4/8/12/16 solutions, 16 on two analytic pose
families. Cuspidality is proven for the class by an estimate of `det J` along a straight
joint path between two solutions; the proof's sufficient condition `a2 > 2 d4` is NOT met
by the example values, and the paper reports that the determinant keeps its sign there all
the same (and numerically splits the 16 solutions into four classes of four joined by
singularity-free paths). Tommy, 2026-11-25: the arm is known to be cuspidal; use it.
`d1` is WLOG zero in the paper; here it is a base-to-shoulder height, which moves the arm
and changes nothing else.

THE ADDED JOINT is a tool pitch at the wrist END, after a short flange barrel: locking it at
any angle leaves exactly the published 6R with a fixed tool transform, so every locked slice
is the cuspidal 6R. The wrist end rather than the base because a base joint perpendicular
to axis 1 tilts the whole arm about a point at table height, driving half the configuration
box through the table. It cannot be coaxial with axis 6 (that would be redundancy in name
only), so it is perpendicular to it.

JOINT ZEROS AND LIMITS: the iiwa's ranges by joint ROLE (Tommy: "more like the range
achieved by the iiwa") -- +-170 deg on the rolls (1, 4, 6), +-120 deg on the pitches (2, 3,
5, 7) -- symmetric about zero because ikflow's first layer is an offset-free scaling. For
those ranges to mean what they mean on the iiwa, the joint ZERO is placed as the iiwa's is:
the arm straight up. DH zero is not that (at `theta = 0` the upper arm is horizontal and the
elbow bent 90 deg), so two joints carry a constant offset
`theta_i = q_i + THETA_OFFSET[i]`. An offset re-labels a joint coordinate; it does not
change the robot, and the fidelity test checks the chain against the paper's DH product
with the offsets applied. The paper's own analysis ignores joint limits.
"""

import math
from dataclasses import dataclass
from typing import Tuple

#: Sphere spacing along a link's axis, in metres; spheres because capsules hang Drake's
#: proximity engine (measured on the soft arm; see src/screw_arm/params.py).
SPHERE_SPACING = 0.05

## ----------------------------------------------------------------------------- the 6R --

#: Metres per unit of the paper's example values. 0.16 puts shoulder-to-fingertip at
#: ~1.06 m against the screw arm's ~1.04 m, so the shared shelf scene fits unchanged.
SCALE = 0.16
PAPER_A2, PAPER_D4, PAPER_A5 = 3.0, 2.0, 0.5

A2 = PAPER_A2 * SCALE
D4 = PAPER_D4 * SCALE
A5 = PAPER_A5 * SCALE
#: Base-to-shoulder: joint 1 sits on a 0.20 m pedestal and the shoulder 0.22 m above it,
#: 0.42 m in all, the screw arm's shoulder height.
PEDESTAL_HEIGHT = 0.20
D1 = 0.22

HALF_PI = 0.5 * math.pi

#: The DH table, `(d, a, alpha)` per joint, in metres and radians.
DH = (
    (D1, 0.0, HALF_PI),
    (0.0, A2, 0.0),
    (0.0, 0.0, -HALF_PI),
    (D4, 0.0, HALF_PI),
    (0.0, A5, -HALF_PI),
    (0.0, 0.0, 0.0),
)

#: `theta_i = q_i + THETA_OFFSET[i]` -- puts q = 0 at the arm straight up, as on the iiwa.
#: Joint 2: the upper arm points along +z instead of +x. Joint 3: the forearm continues the
#: upper arm instead of standing at 90 deg. Joint 5 needs none: at theta5 = 0 axis 6
#: already continues the forearm, displaced sideways by the a5 offset. Checked by
#: tests/test_cusp_arm_kinematics.py, which asserts the straight-up home.
THETA_OFFSET = (0.0, HALF_PI, -HALF_PI, 0.0, 0.0, 0.0)

## ------------------------------------------------------------------- the added joint --

#: The flange barrel from axis 6's frame to the tool pitch's axis, along axis 6.
FLANGE_BARREL = 0.06
#: The tool link: from the tool pitch's axis to the gripper mount.
GRIPPER_MOUNT_Z = 0.10

DEG = math.pi / 180.0
ROLL_LIMIT = 170.0 * DEG
PITCH_LIMIT = 120.0 * DEG


## ------------------------------------------------------------- rotation bookkeeping --


def _matmul(A, B):
    return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def _rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _rpy(R):
    """URDF/SDFormat fixed-axis roll-pitch-yaw of `R = Rz(y) Ry(p) Rx(r)`.

    Only ever called on `Rx(alpha) Rz(offset)` with alpha and offset multiples of pi/2, so
    the pitch is never at the gimbal point in practice; the branch is there for honesty."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2][0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(R[2][1], R[2][2])
        yaw = math.atan2(R[1][0], R[0][0])
    else:
        roll = math.atan2(-R[1][2], R[1][1])
        yaw = 0.0
    return tuple(0.0 if abs(v) < 1e-15 else v for v in (roll, pitch, yaw))


## --------------------------------------------------------------------- the model -----


@dataclass(frozen=True)
class JointSpec:
    """One revolute joint, URDF convention: `origin` places the joint frame in the PARENT.

    The field set is the screw arm's, so the shared torch chain
    (`src.screw_arm.kinematics.link_poses`) and SDF blocks read it unchanged; every joint
    here is revolute (`pitch` 0)."""

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
        return 0.0


@dataclass(frozen=True)
class LinkSpec:
    """One link and the segment its collision spheres are strung along, in its own frame.
    The sphere union IS the robot's collision geometry (declared, not approximated)."""

    name: str
    a: Tuple[float, float, float]
    b: Tuple[float, float, float]
    radius: float
    mass: float = 1.0

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)

    def sphere_centres(self) -> Tuple[Tuple[float, float, float], ...]:
        n = max(1, math.ceil(self.length / SPHERE_SPACING))
        return tuple(
            tuple(self.a[k] + (self.b[k] - self.a[k]) * (i / n) for k in range(3))
            for i in range(n + 1)
        )


#: Link frames are the DH frames: link i's frame is `T_{i-1} Rz(theta_i)`, so its geometry
#: lies along `Tz(d_i) Tx(a_i)` -- +z by d, then +x by a. The elbow and the wrist-roll
#: housings sit on their own joint axes.
LINKS = (
    LinkSpec("base_link",   (0.0, 0.0, 0.09), (0.0, 0.0, PEDESTAL_HEIGHT), 0.085, mass=8.0),
    LinkSpec("shoulder",    (0.0, 0.0, 0.0),  (0.0, 0.0, D1), 0.080, mass=4.0),
    LinkSpec("upper_arm",   (0.0, 0.0, 0.0),  (A2, 0.0, 0.0), 0.060, mass=3.0),
    LinkSpec("elbow",       (0.0, 0.0, -0.05), (0.0, 0.0, 0.05), 0.060, mass=1.5),
    LinkSpec("forearm",     (0.0, 0.0, 0.0),  (0.0, 0.0, D4), 0.055, mass=2.4),
    LinkSpec("wrist",       (0.0, 0.0, 0.0),  (A5, 0.0, 0.0), 0.050, mass=1.2),
    LinkSpec("wrist_roll",  (0.0, 0.0, 0.0),  (0.0, 0.0, FLANGE_BARREL), 0.048, mass=0.8),
    LinkSpec("tool",        (0.0, 0.0, 0.02), (0.0, 0.0, GRIPPER_MOUNT_Z), 0.045, mass=0.6),
)

_JOINT_NAMES = ("base_yaw", "shoulder_pitch", "elbow_pitch", "forearm_roll",
                "wrist_pitch", "wrist_roll_joint", "tool_pitch")
_LIMITS = (ROLL_LIMIT, PITCH_LIMIT, PITCH_LIMIT, ROLL_LIMIT,
           PITCH_LIMIT, ROLL_LIMIT, PITCH_LIMIT)


def _build_joints():
    joints = []
    # Joint 1 stands on the pedestal; joint i + 1's origin in link i is the DH step
    # `Tz(d_i) Tx(a_i) Rx(alpha_i)` followed by joint i + 1's constant offset about its own
    # axis (which commutes with the joint's motion, so it belongs in the origin).
    prev_xyz, prev_R = (0.0, 0.0, PEDESTAL_HEIGHT), _rot_z(THETA_OFFSET[0])
    for i in range(6):
        joints.append(JointSpec(
            _JOINT_NAMES[i], "revolute", LINKS[i].name, LINKS[i + 1].name,
            prev_xyz, _rpy(prev_R), (0.0, 0.0, 1.0), -_LIMITS[i], _LIMITS[i]))
        d, a, alpha = DH[i]
        prev_xyz = (a, 0.0, d)
        offset = THETA_OFFSET[i + 1] if i + 1 < 6 else 0.0
        prev_R = _matmul(_rot_x(alpha), _rot_z(offset))
    # The added tool pitch: up the flange barrel along axis 6, rotating about the barrel's
    # own y -- perpendicular to axis 6, so it is a genuine seventh direction.
    joints.append(JointSpec(
        _JOINT_NAMES[6], "revolute", LINKS[6].name, LINKS[7].name,
        (0.0, 0.0, FLANGE_BARREL), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        -_LIMITS[6], _LIMITS[6]))
    return tuple(joints)


JOINTS = _build_joints()

#: Links closer than this along the chain are not checked against each other: neighbours
#: meet at their joint; the upper arm's end and the forearm's start share the elbow point;
#: the wrist offset (a5 = 0.08 m) is shorter than the forearm's and wrist roll's radii
#: combined; and the tool starts 0.06 m up the barrel from the wrist roll's origin. Three
#: apart and beyond are checked, so the tool folding back onto the forearm, the forearm onto
#: the shoulder and the arm onto its pedestal are live constraints. A test asserts the home
#: configuration is clear under this rule.
COLLISION_CHAIN_GAP = 3

COLLISION_PAIRS = tuple(
    (i, j)
    for i in range(len(LINKS))
    for j in range(i + COLLISION_CHAIN_GAP, len(LINKS))
)


@dataclass(frozen=True)
class CuspArmSpec:
    """The arm. One rung (Tommy: cuspidal rung only, no non-cuspidal control)."""

    name: str

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
        return JOINTS

    @property
    def ndof(self) -> int:
        return len(JOINTS)

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return tuple(j.name for j in JOINTS)

    @property
    def screw_joint_names(self) -> Tuple[str, ...]:
        return ()

    @property
    def limits(self):
        return (tuple(j.lower for j in JOINTS), tuple(j.upper for j in JOINTS))

    @property
    def latent_trust_region(self) -> float:
        """The shared convention: `sqrt(dim) + 1.5`."""
        return round(math.sqrt(self.ndof) + 1.5, 2)

    @property
    def q_nominal(self) -> Tuple[float, ...]:
        """The joint-centering target: a NONSINGULAR, collision-free home in joint order.

        q = 0 is the arm straight up, singular as on the iiwa. This leans the shoulder
        forward, bends the elbow ~86 deg and the wrist back, putting the tool at about
        (0.64, 0, 0.63) m, in front of the arm. Measured (2026-11-25, torch chain): 6x7
        Jacobian sigma_min 0.241 (the iiwa's nominal: 0.204), 0.59 rad inside the nearest
        limit, closest checked sphere pair (forearm, tool) 13 mm clear. Asserted in
        tests/test_cusp_arm_kinematics.py."""
        return (0.0, -0.6, -1.5, 0.0, 1.2, 0.0, 0.0)


SPECS = {s.name: s for s in (CuspArmSpec("cusp7"),)}

PRIMARY = "cusp7"


def GetSpec(name: str) -> CuspArmSpec:
    if name not in SPECS:
        raise KeyError(f"no such cuspidal-arm rung {name!r}; known: {sorted(SPECS)}")
    return SPECS[name]


def RungNames():
    return sorted(SPECS)
