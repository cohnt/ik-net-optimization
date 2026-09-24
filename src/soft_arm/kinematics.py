"""The soft arm's forward kinematics, in torch, once.

This module is the ONLY implementation of the arm's kinematics that runs. The float path
of `VarsToQ`, the AutoDiffXd path's Jacobian, the training dataset's poses, the jrl shim's
`forward_kinematics` and the self-collision screen all call it, so there is nothing for a
second implementation to drift from. SoRoMoX is the oracle it is pinned against, offline,
through a committed golden file -- not a second runtime path.

WHY THE DISCRETIZATION IS EXACT.  A segment carries a CONSTANT strain twist `xi`, so its
pose along the backbone is `exp(xi_hat * s)`, a one-parameter subgroup. Therefore

    exp(xi_hat * L) == exp(xi_hat * L/K) ** K

holds exactly, for any K. Splitting a segment into K sub-links and composing their
transforms reproduces the segment's tip pose to machine precision. K buys collision
resolution and nothing else; it does not approximate the kinematics.

WHAT IS AND IS NOT APPROXIMATE.  Nothing here is. The collision spheres centred at the
sub-link origins are DECLARED to be the robot's geometry (see `params.SoftArmSpec`), not an
approximation of a swept rod, so the collision constraint is exact on the robot as defined.
`sphere_coverage_margin` records that the union does contain a rod of `backbone_radius`.

THE SINGULARITY AT ZERO STRAIN IS THE THING TO GET RIGHT.  The exponential carries
`sin(t)/t`, `(1 - cos t)/t^2` and `(t - sin t)/t^3`, and `t = 0` is exactly the straight,
unstretched configuration -- which is the origin of configuration space, the centre of the
joint-centering cost, and the padding `InvertFlow` writes. A naive implementation returns
NaN there. A `torch.where`-guarded one STILL returns NaN gradients, because `where`
evaluates both branches and the untaken one poisons the backward pass. Both branches here
are therefore finite by construction: the exact branch divides by a CLAMPED angle, so it
never sees zero, and the series branch supplies the values and gradients near it.
"""

import math
import torch

from src.soft_arm.params import SoftArmSpec, SIGMA_Z

#: Below this squared angle the series expansions are used. At the crossover the series
#: and the closed form agree to ~1e-18, far inside float64, so the switch is invisible.
_SMALL_ANGLE_SQ = 1e-8

#: The exact branch divides by an angle clamped to this, so it is finite (and so is its
#: gradient) even where the series branch is the one actually selected.
_CLAMP_SQ = 1e-12


def _exp_coefficients(theta_sq):
    """`(sin t)/t`, `(1 - cos t)/t^2`, `(t - sin t)/t^3` and `sin(t/2)/t`, stably.

    Takes the SQUARED angle, because differentiating a Euclidean norm at zero is itself a
    singularity -- the same reason SoRoMoX's constant-strain utilities work in
    `dot(omega, omega)` rather than `|omega|`.
    """
    small = theta_sq < _SMALL_ANGLE_SQ
    # EVERY denominator in the exact branch is the CLAMPED angle, not the true one.
    # `torch.where` evaluates both branches, so a division by the true (possibly zero)
    # angle would produce an inf in the untaken branch and an inf * 0 = NaN in the
    # backward pass -- at exactly the straight configuration, which is the origin of
    # configuration space and the centre of the joint-centering cost. Clamping only the
    # angle and leaving `theta_sq` raw in the denominators reproduces that bug; it was
    # measured doing so before this comment existed.
    theta_sq_safe = torch.clamp(theta_sq, min=_CLAMP_SQ)
    theta = torch.sqrt(theta_sq_safe)
    t2, t4 = theta_sq, theta_sq * theta_sq          # series: the TRUE angle, accurate at 0

    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    a = torch.where(small, 1.0 - t2 / 6.0 + t4 / 120.0, sin_t / theta)
    b = torch.where(small, 0.5 - t2 / 24.0 + t4 / 720.0, (1.0 - cos_t) / theta_sq_safe)
    c = torch.where(small, 1.0 / 6.0 - t2 / 120.0 + t4 / 5040.0,
                    (theta - sin_t) / (theta_sq_safe * theta))
    # The quaternion's vector part is `s * a`, with s = sin(t/2)/t -> 1/2 as t -> 0.
    s = torch.where(small, 0.5 - t2 / 48.0 + t4 / 3840.0, torch.sin(0.5 * theta) / theta)
    w = torch.where(small, 1.0 - t2 / 8.0 + t4 / 384.0, torch.cos(0.5 * theta))
    return a, b, c, s, w


def _cross(u, v):
    return torch.cross(u, v, dim=-1)


def se3_exp(omega, nu):
    """`exp(xi_hat)` for the twist `xi = [omega; nu]`, already scaled by arc length.

    Angular-first, matching SoRoMoX's ordering. Returns `(quaternion_wxyz, translation)`
    with shapes `(..., 4)` and `(..., 3)`.

    The rotation is returned as a quaternion rather than a matrix on purpose: composing
    quaternions is smooth everywhere, whereas recovering one from a rotation matrix needs a
    largest-component branch whose gradient is discontinuous at the branch boundaries.
    """
    theta_sq = (omega * omega).sum(-1, keepdim=True)
    a, b, c, s, w = _exp_coefficients(theta_sq)

    quat = torch.cat([w, s * omega], dim=-1)

    # p = V(omega) nu with V = I + b [omega]x + c [omega]x^2
    cross1 = _cross(omega, nu)
    cross2 = _cross(omega, cross1)
    translation = nu + b * cross1 + c * cross2
    return quat, translation


def quat_multiply(q1, q2):
    """Hamilton product, wxyz. `q1 * q2` composes q2's rotation then q1's."""
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    w = w1 * w2 - (v1 * v2).sum(-1, keepdim=True)
    v = w1 * v2 + w2 * v1 + _cross(v1, v2)
    return torch.cat([w, v], dim=-1)


def quat_rotate(q, p):
    """Rotate `p` by the unit quaternion `q` (wxyz), without forming a matrix."""
    w, v = q[..., :1], q[..., 1:]
    t = 2.0 * _cross(v, p)
    return p + w * t + _cross(v, t)


def _as_batch(cfg, spec, dtype, device):
    tensor = torch.as_tensor(cfg, dtype=dtype, device=device)
    squeeze = tensor.dim() == 1
    if squeeze:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != spec.ndof:
        raise ValueError(f"{spec.name} takes {spec.ndof} coordinates, got {tensor.shape[-1]}")
    return tensor, squeeze


def strain_limits_tensor(spec, dtype=torch.float64, device="cpu"):
    return torch.tensor(spec.limits_per_dof, dtype=dtype, device=device)


def cfg_to_twists(cfg, spec, dtype=torch.float64, device="cpu"):
    """Normalized configuration in `[-1, 1]^ndof` -> the physical strain twist per segment.

    Returns `(omega, nu)`, each `(B, num_segments, 3)`, in SoRoMoX's angular-first order.
    The reference strain is the unstretched rod, so the axial entry is `1 + sigma_z`: a
    configuration of all zeros is the straight arm at its nominal length.
    """
    tensor, squeeze = _as_batch(cfg, spec, dtype, device)
    physical = tensor * strain_limits_tensor(spec, dtype, device)
    physical = physical.reshape(-1, spec.num_segments, spec.strains_per_segment)

    twist = torch.zeros(physical.shape[0], spec.num_segments, 6, dtype=dtype, device=device)
    # `index_copy` rather than fancy indexing so the graph stays simple for jacfwd.
    index = torch.tensor(spec.strain_basis, dtype=torch.long, device=device)
    twist = twist.index_copy(2, index, physical)
    # The reference strain: an unstretched rod advances one unit of arc length per unit of
    # arc length. `sigma_z` is the DEVIATION from that.
    reference = torch.zeros(6, dtype=dtype, device=device)
    reference[SIGMA_Z] = 1.0
    twist = twist + reference
    return twist[..., :3], twist[..., 3:], squeeze


def sublink_poses(cfg, spec, dtype=torch.float64, device="cpu"):
    """Pose of every collision sub-link's origin, plus the arm's tip.

    Returns `(quat, translation, tip_quat, tip_translation)` with the sub-link arrays
    shaped `(B, num_sublinks, 4)` and `(B, num_sublinks, 3)`, in the arm's base frame --
    which the scene welds to the world at identity, so it is also the world frame.

    Sub-link `j` of segment `i` sits at the START of its own sub-arc, so the first
    sub-link of the first segment is the base pose itself. That is what puts a collision
    sphere at every sampled point of the backbone rather than half a step past it.
    """
    omega, nu, squeeze = cfg_to_twists(cfg, spec, dtype, device)
    ds = spec.sublink_length
    # One step transform per segment: every sub-link of a segment shares the segment's
    # strain, so the same relative transform repeats K times.
    step_quat, step_translation = se3_exp(omega * ds, nu * ds)

    batch = step_quat.shape[0]
    quat = torch.zeros(batch, 0, 4, dtype=dtype, device=device)
    translation = torch.zeros(batch, 0, 3, dtype=dtype, device=device)

    current_quat = torch.zeros(batch, 4, dtype=dtype, device=device)
    current_quat[:, 0] = 1.0
    current_translation = torch.zeros(batch, 3, dtype=dtype, device=device)

    quats, translations = [], []
    for segment in range(spec.num_segments):
        sq = step_quat[:, segment, :]
        st = step_translation[:, segment, :]
        for _ in range(spec.sublinks_per_segment):
            quats.append(current_quat)
            translations.append(current_translation)
            current_translation = current_translation + quat_rotate(current_quat, st)
            current_quat = quat_multiply(current_quat, sq)

    quat = torch.stack(quats, dim=1)
    translation = torch.stack(translations, dim=1)
    if squeeze:
        return (quat[0], translation[0], current_quat[0], current_translation[0])
    return quat, translation, current_quat, current_translation


def tip_pose(cfg, spec, dtype=torch.float64, device="cpu"):
    """`(quaternion_wxyz, translation)` of the arm's tip frame."""
    _, _, tip_quat, tip_translation = sublink_poses(cfg, spec, dtype, device)
    return tip_quat, tip_translation


def forward_kinematics(cfg, spec, dtype=torch.float64, device="cpu"):
    """Tip pose as `[x, y, z, qw, qx, qy, qz]`, the convention jrl and ikflow use.

    The quaternion is canonicalised to `w >= 0`, as pydrake's `ToQuaternion` does, so the
    dataset the flow trains on and the conditioning poses the program forms agree in sign.
    """
    tip_quat, tip_translation = tip_pose(cfg, spec, dtype, device)
    sign = torch.where(tip_quat[..., :1] < 0, -1.0, 1.0)
    return torch.cat([tip_translation, tip_quat * sign], dim=-1)


def config_to_plant_q(cfg, spec, dtype=torch.float64, device="cpu"):
    """Configuration -> the Drake plant's position vector.

    Every sub-link is a quaternion floating body, whose 7 positions Drake orders
    `[qw, qx, qy, qz, x, y, z]`; bodies appear in the order the generated SDF declares
    them, which is segment-major. `tests/test_soft_arm_drake.py` pins both orderings
    against the plant rather than trusting this comment.
    """
    quat, translation, _, _ = sublink_poses(cfg, spec, dtype, device)
    return torch.cat([quat, translation], dim=-1).reshape(*quat.shape[:-2], -1)


def sphere_centers(cfg, spec, dtype=torch.float64, device="cpu"):
    """Centres of the collision spheres, `(B, num_sublinks, 3)`.

    These are the sub-link origins: the spheres ARE the robot, so this is its geometry,
    not a proxy for it.
    """
    _, translation, _, _ = sublink_poses(cfg, spec, dtype, device)
    return translation


def backbone_points(cfg, spec, samples_per_sublink=8, dtype=torch.float64, device="cpu"):
    """Densely sampled points on the TRUE backbone curve, for the containment test.

    Walks each sub-arc at `samples_per_sublink` interior abscissae using the same
    exponential, so this is the continuous rod the sphere union has to contain -- not a
    re-sampling of the sphere centres, which would make the test vacuous.
    """
    omega, nu, squeeze = cfg_to_twists(cfg, spec, dtype, device)
    quat, translation, _, _ = sublink_poses(cfg, spec, dtype, device)
    ds = spec.sublink_length

    fractions = torch.arange(1, samples_per_sublink + 1, dtype=dtype, device=device)
    fractions = (fractions / samples_per_sublink) * ds

    points = []
    for segment in range(spec.num_segments):
        for sub in range(spec.sublinks_per_segment):
            index = segment * spec.sublinks_per_segment + sub
            base_q = quat[:, index, :]
            base_p = translation[:, index, :]
            for fraction in fractions:
                step_q, step_t = se3_exp(omega[:, segment, :] * fraction,
                                         nu[:, segment, :] * fraction)
                points.append(base_p + quat_rotate(base_q, step_t))
    stacked = torch.stack(points, dim=1)
    return stacked[0] if squeeze else stacked
