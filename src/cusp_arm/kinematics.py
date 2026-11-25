"""The `cusp7` arm's forward kinematics in torch, batched.

The chain itself is the screw arm's (`src.screw_arm.kinematics.link_poses`), which walks any
spec's URDF-convention joint table with quaternions throughout; every joint here is
revolute, so its screw term is identically zero. What is this robot's own is the collision
table, keyed on `src.cusp_arm.params.COLLISION_PAIRS` so the dataset screen and the SDF's
filter groups are generated from one tuple.

`dh_flange_pose` is the paper's DH product written out independently of the joint table, so
a test can assert the rendered chain is the published 6R (with the joint-zero offsets)
rather than something that merely agrees with itself.
"""

import math

import torch

from src.cusp_arm.params import (COLLISION_PAIRS, DH, FLANGE_BARREL, GRIPPER_MOUNT_Z,
                                 PEDESTAL_HEIGHT, THETA_OFFSET, CuspArmSpec)
from src.screw_arm.kinematics import (  # noqa: F401  -- re-exported for callers
    forward_kinematics, flange_pose, link_poses, quat_rotate, quat_to_matrix)

_CPU = "cpu"


def _sphere_table(spec: CuspArmSpec, dtype, device):
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


def sphere_centres_world(cfg, spec: CuspArmSpec, dtype=torch.float64, device=_CPU):
    """Every collision sphere's centre in the base frame, `[B, nspheres, 3]`."""
    quats, transs = link_poses(cfg, spec, dtype=dtype, device=device)
    if quats.dim() == 2:
        quats, transs = quats.unsqueeze(0), transs.unsqueeze(0)
    local, owner, _ = _sphere_table(spec, dtype, device)
    q = quats[:, owner, :]
    t = transs[:, owner, :]
    return t + quat_rotate(q, local.unsqueeze(0).expand(q.shape[0], -1, -1))


def _checked_sphere_pairs(spec: CuspArmSpec, device):
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


def self_collision_depth(cfg, spec: CuspArmSpec, dtype=torch.float64, device=_CPU):
    """Worst overlap between two unfiltered collision spheres, `[B]`, positive in collision.
    Sphere against sphere: exactly the geometry Drake's constraint sees on this robot."""
    centres = sphere_centres_world(cfg, spec, dtype=dtype, device=device)
    _, _, radii = _sphere_table(spec, dtype, device)
    first, second = _checked_sphere_pairs(spec, device)
    gap = (torch.linalg.norm(centres[:, first, :] - centres[:, second, :], dim=-1)
           - (radii[first] + radii[second]))
    return -gap.min(dim=-1).values


## -- the paper's DH product, independently -----------------------------------------------


def _dh(theta, d, a, alpha):
    """Standard DH `Rz(theta) Tz(d) Tx(a) Rx(alpha)` as a batch of 4x4, `theta` `[B]`."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack((
        torch.stack((ct, -st * ca, st * sa, a * ct), -1),
        torch.stack((st, ct * ca, -ct * sa, a * st), -1),
        torch.stack((zero, zero + sa, zero + ca, zero + d), -1),
        torch.stack((zero, zero, zero, one), -1),
    ), dim=-2)


def dh_6r(theta):
    """The published 6R, `T_0^6` for DH angles `theta` `[B, 6]` (frame 0 on axis 1)."""
    T = torch.eye(4, dtype=theta.dtype).expand(theta.shape[0], 4, 4)
    for i, (d, a, alpha) in enumerate(DH):
        T = T @ _dh(theta[:, i], d, a, alpha)
    return T


def dh_flange_pose(q):
    """The tool frame, `[B, 4, 4]`, built from the paper's DH product: pedestal, then
    `T_0^6` at `theta = q[:6] + THETA_OFFSET`, then the flange barrel and the added tool
    pitch (about the barrel's y). The result is the tool link's own frame, which is what
    `forward_kinematics` returns (the gripper mount is `GRIPPER_MOUNT_Z` further along z)."""
    q = torch.as_tensor(q, dtype=torch.float64)
    theta = q[:, :6] + torch.tensor(THETA_OFFSET, dtype=q.dtype)
    base = torch.eye(4, dtype=q.dtype).expand(q.shape[0], 4, 4).clone()
    base[:, 2, 3] = PEDESTAL_HEIGHT
    T = base @ dh_6r(theta)
    c, s = torch.cos(q[:, 6]), torch.sin(q[:, 6])
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    tool = torch.stack((
        torch.stack((c, zero, s, zero), -1),
        torch.stack((zero, one, zero, zero), -1),
        torch.stack((-s, zero, c, zero + FLANGE_BARREL), -1),
        torch.stack((zero, zero, zero, one), -1),
    ), dim=-2)
    return T @ tool


def det_j_6r(theta):
    """`det J` of the 6R in closed form, at DH angles `theta` `[B, 6]`:

        -a2 c3 [ -a2 a5 c2 s4^2 - a2 d4 c2 s5 + a5 d4 sin(t2 + t3) + d4^2 s5 sin(t2 + t3) ]

    (up to the sign fixed by the Jacobian's row order). Derived symbolically from the DH
    table (2026-11-25) because the paper's printed form, as read from its HTML version,
    `-a2 d4 c3 (((s3 + s2 c3) d4 - a2) s5 + a5 sin(t2 + t3))`, does not match its own table:
    it drops the `s4^2` term (zero on the paper's path, where theta4 = 0) and differs at
    theta4 = 0 too. The paper's cuspidality argument uses the path with theta4 = 0, where
    the two agree in structure; a numerical check of that path on these numbers keeps one
    sign of `det J` throughout. Independent of theta1 and theta6, as any 6R's is."""
    a2, d4, a5 = DH[1][1], DH[3][0], DH[4][1]
    t2, t3, t4, t5 = theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4]
    c2, c3, s4, s5 = torch.cos(t2), torch.cos(t3), torch.sin(t4), torch.sin(t5)
    s23 = torch.sin(t2 + t3)
    return -a2 * c3 * (-a2 * a5 * c2 * s4 ** 2 - a2 * d4 * c2 * s5
                       + a5 * d4 * s23 + d4 ** 2 * s5 * s23)


#: The gripper mount, in the tool link's frame (re-exported for the scene generator).
MOUNT_Z = GRIPPER_MOUNT_Z
