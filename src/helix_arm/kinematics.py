"""The `helix7` arm's forward kinematics in torch, batched.

This is the map the dataset is drawn through, the map ikflow's validation compares against,
and the map `CalibrateFlowFrame` reads. It is written against `params.py` rather than
parsed out of a file, so it and `generate_urdf.py` are two renderings of one definition and
a test asserts they agree to 1e-12 against Drake -- the plant the solver actually
differentiates. That test is a LIVE comparison at freshly drawn configurations, not a
golden file: the soft arm needed a committed `.npz` only because its oracle was JAX in a
separate venv, and a check that silently stops running is indistinguishable from one that
passes.

POSES ARE CARRIED AS QUATERNIONS THROUGHOUT, never as rotation matrices that are converted
back at the end. Axis-angle to quaternion is `[cos(q/2), axis * sin(q/2)]`, which is entire
-- no small-angle branch, no clamped divisor, no `torch.where` whose untaken branch can
poison the backward pass. Matrix-to-quaternion is the operation that needs those guards,
and composing quaternions from the start means it never happens. (The soft arm's
`se3_exp` machinery does not transfer for the same reason: its `sin(t)/t` coefficients have
a removable singularity at the origin of ITS configuration space; a screw joint's
translation is linear in `q` and its axis is a constant.)

A SCREW JOINT IS ONE LINE MORE THAN A REVOLUTE ONE. Rotation about an axis fixes that axis,
so the rotation and the axial translation commute and the joint transform is just the
revolute transform with the translation block filled in:

    R = axis-angle(axis, q),   t = axis * pitch * q / (2 * pi)

with `pitch` in metres per REVOLUTION, matching Drake's `ScrewJoint`.
"""

import math

import torch

from src.helix_arm.params import COLLISION_PAIRS, HelixArmSpec, TWO_PI

_CPU = "cpu"


## -- quaternion algebra, wxyz ------------------------------------------------------------
##
## Built explicitly with a dtype and device on every call. `jrl.config` calls
## `torch.set_default_device` AND `torch.set_default_dtype(torch.float32)` AT IMPORT, so a
## bare `torch.tensor(...)` in this process allocates float32 on cuda -- which then either
## raises somewhere unrelated or, worse, silently downcasts a float64 chain to float32 and
## leaves a 1e-8 noise floor in what is supposed to be a 1e-12 agreement.


def quat_multiply(p, q):
    """Hamilton product, `[..., 4]` wxyz."""
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        (
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ),
        dim=-1,
    )


def quat_rotate(q, v):
    """Rotate `v` `[..., 3]` by the unit quaternion `q` `[..., 4]`."""
    w, u = q[..., 0:1], q[..., 1:4]
    uv = torch.cross(u, v, dim=-1)
    return v + 2.0 * (w * uv + torch.cross(u, uv, dim=-1))


def quat_from_rpy(rpy, dtype, device):
    """A constant orientation from URDF's fixed roll-pitch-yaw, as `[4]` wxyz."""
    roll, pitch, yaw = (float(a) for a in rpy)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return torch.tensor(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=dtype,
        device=device,
    )


def quat_to_matrix(q):
    """`[..., 4]` wxyz -> `[..., 3, 3]`. Polynomial, so safe everywhere; used only to
    compare against Drake, never on the chain itself."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )


## -- the chain -----------------------------------------------------------------------


def _as_batch(cfg, spec, dtype, device):
    """`[ndof]` or `[B, ndof]` -> `([B, ndof], squeeze)`."""
    x = torch.as_tensor(cfg, dtype=dtype, device=device)
    if x.dim() == 1:
        return x.unsqueeze(0), True
    if x.dim() != 2:
        raise ValueError(f"expected [ndof] or [B, ndof], got shape {tuple(x.shape)}")
    if x.shape[1] != spec.ndof:
        raise ValueError(
            f"{spec.name} has {spec.ndof} coordinates, got {x.shape[1]}. A width mismatch "
            f"here is usually a chart loaded against the wrong rung.")
    return x, False


def joint_step(joint, q, dtype, device):
    """The moving part of one joint: `(quat, translation)` for a batch of `q` `[B]`."""
    axis = torch.tensor(joint.axis, dtype=dtype, device=device)
    axis = axis / torch.linalg.norm(axis)
    half = 0.5 * q
    quat = torch.cat((torch.cos(half).unsqueeze(-1),
                      torch.sin(half).unsqueeze(-1) * axis), dim=-1)
    if joint.kind == "screw":
        trans = q.unsqueeze(-1) * (joint.pitch / TWO_PI) * axis
    else:
        trans = torch.zeros(q.shape[0], 3, dtype=dtype, device=device)
    return quat, trans


def link_poses(cfg, spec: HelixArmSpec, dtype=torch.float64, device=_CPU):
    """Every link frame in the base frame.

    Returns `(quat [B, nlinks, 4], trans [B, nlinks, 3])` in `spec.links` order, with the
    base link at index 0 carrying the identity. `spec`'s joint table is a simple chain, so
    link `i + 1` is the child of joint `i`.
    """
    x, squeeze = _as_batch(cfg, spec, dtype, device)
    batch = x.shape[0]

    quat = torch.zeros(batch, 4, dtype=dtype, device=device)
    quat[:, 0] = 1.0
    trans = torch.zeros(batch, 3, dtype=dtype, device=device)
    quats, transs = [quat], [trans]

    for i, joint in enumerate(spec.joints):
        fixed_q = quat_from_rpy(joint.origin_rpy, dtype, device).expand(batch, 4)
        fixed_t = torch.tensor(joint.origin_xyz, dtype=dtype, device=device).expand(batch, 3)
        step_q, step_t = joint_step(joint, x[:, i], dtype, device)

        # parent -> joint frame (fixed), then joint frame -> child (moving).
        trans = trans + quat_rotate(quat, fixed_t)
        quat = quat_multiply(quat, fixed_q)
        trans = trans + quat_rotate(quat, step_t)
        quat = quat_multiply(quat, step_q)

        quats.append(quat)
        transs.append(trans)

    q_out = torch.stack(quats, dim=1)
    t_out = torch.stack(transs, dim=1)
    if squeeze:
        return q_out[0], t_out[0]
    return q_out, t_out


def flange_pose(cfg, spec: HelixArmSpec, dtype=torch.float64, device=_CPU):
    """`(quat [B, 4] wxyz, trans [B, 3])` of the flange, in the base frame."""
    quats, transs = link_poses(cfg, spec, dtype=dtype, device=device)
    return quats[..., -1, :], transs[..., -1, :]


def forward_kinematics(cfg, spec: HelixArmSpec, dtype=torch.float64, device=_CPU):
    """`[B, 7]` as `x, y, z, qw, qx, qy, qz` -- jrl's layout, and ikflow's.

    Canonicalised to `qw >= 0`, as jrl's own robots are, so a pose and its antipode do not
    read as two different training targets. That sign flip is a discontinuity in the
    REPRESENTATION at `qw = 0` and not in the pose; nothing on the solver's path goes
    through this function (`VarsToQ` asks Drake for the end-effector pose), so it costs
    nothing there.
    """
    quat, trans = flange_pose(cfg, spec, dtype=dtype, device=device)
    sign = torch.where(quat[..., 0:1] < 0, -torch.ones_like(quat[..., 0:1]),
                       torch.ones_like(quat[..., 0:1]))
    return torch.cat((trans, quat * sign), dim=-1)


## -- collision geometry ----------------------------------------------------------------


def _sphere_table(spec: HelixArmSpec, dtype, device):
    """`(centres [nspheres, 3], link_index [nspheres], radius [nspheres])` in link frames."""
    centres, owner, radii = [], [], []
    for i, link in enumerate(spec.links):
        for c in link.sphere_centres():
            centres.append(c)
            owner.append(i)
            radii.append(link.radius)
    return (torch.tensor(centres, dtype=dtype, device=device),
            torch.tensor(owner, dtype=torch.int64, device=device),
            torch.tensor(radii, dtype=dtype, device=device))


def sphere_centres_world(cfg, spec: HelixArmSpec, dtype=torch.float64, device=_CPU):
    """Every collision sphere's centre in the base frame, `[B, nspheres, 3]`."""
    quats, transs = link_poses(cfg, spec, dtype=dtype, device=device)
    if quats.dim() == 2:
        quats, transs = quats.unsqueeze(0), transs.unsqueeze(0)
    local, owner, _ = _sphere_table(spec, dtype, device)
    q = quats[:, owner, :]
    t = transs[:, owner, :]
    return t + quat_rotate(q, local.unsqueeze(0).expand(q.shape[0], -1, -1))


def _checked_sphere_pairs(spec: HelixArmSpec, device):
    """Index pairs of spheres whose LINKS are in `COLLISION_PAIRS`.

    Adjacent links are filtered here and in the URDF, and they must agree: the telescoping
    tubes interpenetrate by construction, so a screen that checked them would reject every
    configuration, and a URDF that did not filter them would make every solve infeasible.
    """
    owner = []
    for i, link in enumerate(spec.links):
        owner += [i] * len(link.sphere_centres())
    allowed = set(COLLISION_PAIRS)
    first, second = [], []
    for a in range(len(owner)):
        for b in range(a + 1, len(owner)):
            if (owner[a], owner[b]) in allowed:
                first.append(a)
                second.append(b)
    return (torch.tensor(first, dtype=torch.int64, device=device),
            torch.tensor(second, dtype=torch.int64, device=device))


def self_collision_depth(cfg, spec: HelixArmSpec, dtype=torch.float64, device=_CPU):
    """Worst overlap between two unfiltered collision spheres, `[B]`, positive in collision.

    Sphere-sphere, so this is EXACTLY the geometry Drake's constraint sees rather than a
    conservative stand-in, and it is a `cdist` rather than the per-pair QP jrl uses for
    capsules -- which is what makes a 25M-sample dataset screen affordable.
    """
    centres = sphere_centres_world(cfg, spec, dtype=dtype, device=device)
    _, _, radii = _sphere_table(spec, dtype, device)
    first, second = _checked_sphere_pairs(spec, device)
    gap = (torch.linalg.norm(centres[:, first, :] - centres[:, second, :], dim=-1)
           - (radii[first] + radii[second]))
    return -gap.min(dim=-1).values
