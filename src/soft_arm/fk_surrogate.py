"""A learned forward model for the soft arm, and the honest machinery around it.

WHY THIS EXISTS. The configuration-to-plant-positions map is the whole forward model: the
IK constraint reads the task frame's pose out of it, and the collision constraint reads the
whole body's placement out of it. So replacing that one map with a network replaces the
forward model everywhere at once, and it slots into the same `ConfigToPlantQ` hook and the
same two-Jacobian chain the analytic map uses.

Three reasons it is an axis rather than a fallback, in order of weight:

1. It is the GENERAL mechanism. The analytic path exists only because a piecewise
   constant-strain backbone happens to have a closed-form exponential. SoRoMoX's GVS models
   integrate numerically, and an actuation-space model of a real HSA or tendon-driven arm
   has no closed form at all. This is what makes the robot "a soft arm" rather than "the one
   soft arm we could write down by hand".
2. It is the setting the inspiration paper is in. LOInK's soft-manipulator experiment is
   explicitly its case (b) -- no forward model available, only recorded configurations --
   and its IKFlow baseline is trained on exactly that data.
3. It is a CONTROL, not an advantage. The joint-space arm uses the same surrogate, so any
   effect is a property of the forward model rather than of the formulation -- the same
   structure that makes the solver axis interpretable.

THE ROTATION HEAD IS 6-D ON PURPOSE. Regressing quaternions directly is discontinuous (q
and -q are the same rotation, so the target is two-valued) and regressing Euler angles has
the pitch singularity. The 6-D continuous representation -- two vectors, Gram-Schmidt to an
orthonormal frame -- is smooth, surjective onto SO(3), and unit by construction, so the
plant never sees a non-unit quaternion and the AutoDiffXd path never differentiates a
normalisation that could divide by zero.

WHAT MUST TRAVEL WITH IT. An arm that optimises against its own surrogate must not be
graded by it: `verify()` re-measures the task on the EXACT kinematics through
`VerificationQ`. And `CalibrateFlowFrame` cannot pass its 1e-6 constancy check, because the
flow is trained against the true FK and the offset it measures is then the surrogate's own
error -- so the tolerance becomes a stated number and the measured spread is recorded,
rather than the check being switched off.
"""

import json
import os

import torch

from src.soft_arm import kinematics as K
from src.soft_arm.params import SoftArmSpec

_CPU = "cpu"


def sixd_to_rotation(sixd):
    """Two vectors -> an orthonormal frame, by Gram-Schmidt. `(..., 6) -> (..., 3, 3)`."""
    a, b = sixd[..., :3], sixd[..., 3:]
    e1 = torch.nn.functional.normalize(a, dim=-1)
    b_perp = b - (e1 * b).sum(-1, keepdim=True) * e1
    e2 = torch.nn.functional.normalize(b_perp, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def rotation_to_quaternion(matrix):
    """`(..., 3, 3) -> (..., 4)` wxyz, by the branchless symmetric-sum form.

    The largest-component branch is the usual route and is NOT used: its gradient is
    discontinuous exactly at the branch boundaries, which is a thing the constraint
    Jacobian would inherit. This form differentiates cleanly everywhere a rotation matrix
    is proper, which Gram-Schmidt guarantees.
    """
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-12)) * 0.5
    x = (m[..., 2, 1] - m[..., 1, 2]) / (4.0 * w)
    y = (m[..., 0, 2] - m[..., 2, 0]) / (4.0 * w)
    z = (m[..., 1, 0] - m[..., 0, 1]) / (4.0 * w)
    return torch.stack([w, x, y, z], dim=-1)


class BackboneFkSurrogate(torch.nn.Module):
    """`cfg -> every collision body's pose`, in the analytic map's own layout.

    The output is laid out exactly as `kinematics.config_to_plant_q` lays it out --
    `[qw, qx, qy, qz, x, y, z]` per body, bodies in `spec.body_names()` order -- so the
    program's plant-slot permutation and its Jacobian chain apply unchanged, and the two
    backends are interchangeable at one call site.
    """

    def __init__(self, spec: SoftArmSpec, width: int = 512, depth: int = 4):
        super().__init__()
        self.spec_name = spec.name
        self.ndof = spec.ndof
        self.num_bodies = spec.num_bodies
        self.width = width
        self.depth = depth
        layers, size = [], spec.ndof
        for _ in range(depth):
            layers += [torch.nn.Linear(size, width), torch.nn.SiLU()]
            size = width
        layers.append(torch.nn.Linear(size, spec.num_bodies * 9))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, cfg):
        single = cfg.dim() == 1
        batched = cfg.unsqueeze(0) if single else cfg
        raw = self.net(batched).reshape(batched.shape[0], self.num_bodies, 9)
        translation = raw[..., :3]
        quaternion = rotation_to_quaternion(sixd_to_rotation(raw[..., 3:]))
        out = torch.cat([quaternion, translation], dim=-1).reshape(batched.shape[0], -1)
        return out[0] if single else out


def MakeLearnedConfigToPlantQ(surrogate):
    """A free function `cfg -> plant positions`, closing over the surrogate and nothing else.

    Free, for the same reason the flow's inference is: `torch.compile` guards on everything
    a callable closes over, so compiling a bound method re-triggers dynamo per program.
    """

    def config_to_plant(cfg):
        return surrogate(cfg)

    return config_to_plant


def SurrogatePath(spec: SoftArmSpec, root: str) -> str:
    return os.path.join(root, "models", spec.name, f"{spec.name}__fk_surrogate.pt")


def Save(surrogate, spec, path, metrics, provenance=None):
    torch.save({"state_dict": surrogate.state_dict(), "spec_name": spec.name,
                "ndof": spec.ndof, "num_bodies": spec.num_bodies,
                "width": surrogate.width, "depth": surrogate.depth,
                "metrics": metrics, "provenance": provenance or {}}, path)
    with open(os.path.splitext(path)[0] + ".json", "w") as handle:
        json.dump({"spec_name": spec.name, "width": surrogate.width,
                   "depth": surrogate.depth, "metrics": metrics,
                   "provenance": provenance or {}}, handle, indent=2, sort_keys=True)


def Load(spec: SoftArmSpec, path, device=_CPU):
    """Load a surrogate, refusing one trained for a different robot.

    A surrogate is a forward MODEL. Loading one built for another rung would silently swap
    the robot out from under every constraint, which is the same class of error the flow's
    architecture sidecar exists to catch.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob["spec_name"] != spec.name:
        raise ValueError(f"{path} is a surrogate for {blob['spec_name']!r}, not {spec.name!r}")
    if blob["ndof"] != spec.ndof or blob["num_bodies"] != spec.num_bodies:
        raise ValueError(
            f"{path} has ndof={blob['ndof']} num_bodies={blob['num_bodies']}, but "
            f"{spec.name} is {spec.ndof}/{spec.num_bodies} -- the spec has changed since "
            f"this surrogate was fitted and it no longer describes this robot")
    surrogate = BackboneFkSurrogate(spec, width=blob["width"], depth=blob["depth"])
    surrogate.load_state_dict(blob["state_dict"])
    surrogate.to(device).double().eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    return surrogate, blob["metrics"]


@torch.no_grad()
def Screen(surrogate, spec, n=20000, seed=0, device=_CPU):
    """Tip-pose error against the exact map -- the surrogate's own accuracy report.

    Reported like a chart's: median and tail, in millimetres and degrees. The number that
    matters is the tail against the 1e-3 m task gate, because that is what a solve is
    graded by, and it is measured on configurations the fit never saw.
    """
    generator = torch.Generator().manual_seed(seed)
    cfg = (torch.rand(n, spec.ndof, generator=generator, dtype=torch.float64) * 2 - 1).to(device)
    exact = K.config_to_plant_q(cfg, spec, device=device)
    approx = surrogate(cfg)
    tip = slice(7 * (spec.num_bodies - 1), 7 * spec.num_bodies)
    dp = torch.linalg.norm(exact[:, tip][:, 4:] - approx[:, tip][:, 4:], dim=-1) * 1000.0
    qa, qb = exact[:, tip][:, :4], approx[:, tip][:, :4]
    dot = torch.clamp((qa * qb).sum(-1).abs(), max=1.0)
    da = torch.rad2deg(2.0 * torch.arccos(dot))
    body_dp = torch.linalg.norm(
        (exact.reshape(n, -1, 7)[..., 4:] - approx.reshape(n, -1, 7)[..., 4:]), dim=-1) * 1000.0
    return {"n": int(n), "seed": int(seed),
            "tip_mm/median": float(dp.median()), "tip_mm/p99": float(dp.quantile(0.99)),
            "tip_mm/max": float(dp.max()),
            "tip_deg/median": float(da.median()), "tip_deg/p99": float(da.quantile(0.99)),
            "tip_deg/max": float(da.max()),
            "body_mm/median": float(body_dp.median()), "body_mm/max": float(body_dp.max())}
