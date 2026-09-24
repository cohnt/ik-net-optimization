"""A `jrl.robot.Robot` for the soft arm, so ikflow can train on it unmodified.

ikflow reaches its robot through `jrl.robots.get_robot`, and jrl's `Robot` is URDF- and
klampt-driven: its forward kinematics come from parsing a kinematic chain, which cannot
express strain coordinates. Rather than fork jrl, this SUBCLASSES `Robot` and overrides
`__init__` without calling `super().__init__` -- the base constructor would demand a URDF,
load it into klampt and run a batch FK to warm a cache, none of which applies.

The subclassing is not stylistic: `IKFlowSolver.__init__` asserts
`isinstance(robot, Robot)`, so a duck-typed class fails on the first load. `name` is a
CLASS attribute for the same kind of reason -- `get_robot` compares `clc.name` on the class
before instantiating anything, and a class attribute shadows the base's property, which is
exactly how jrl's own robots declare it.

WHAT IKFLOW ACTUALLY TOUCHES, and therefore what is implemented here:

  `name`, `ndof`                 -- the model's input width and the dataset directory
  `actuated_joints_limits`       -- baked into `module_list.0` as a fixed rescaling
  `sample_joint_angles`          -- dataset draws
  `sample_joint_angles_and_poses`-- dataset draws and their poses; OVERRIDDEN, because
                                    jrl's version goes through klampt one config at a time
  `forward_kinematics`           -- batched, torch, [N, 7] as xyz + wxyz; used by the
                                    dataset, by validation's pose error, and by
                                    `CalibrateFlowFrame`
  `clamp_to_joint_limits`        -- validation
  `config_self_collides`         -- validation and `--only_non_self_colliding`

Everything else jrl exposes that ikflow can reach raises `NotImplementedError` with a
message naming why, so an unreachable path fails immediately and loudly rather than as an
`AttributeError` four hundred thousand steps into a training run.

The limits are `[-1, 1]` on every coordinate because the configuration is NORMALIZED
strain (see `params.py`). That also makes ikflow's first-layer rescaling the identity,
which is the tidiest possible answer to a transform that has no offset to give.
"""

import numpy as np
import torch

from jrl.robot import Robot

from src.soft_arm import kinematics as K
from src.soft_arm.params import RUNGS, SoftArmSpec

#: Everything here is built explicitly on the CPU. `jrl.config` calls
#: `torch.set_default_device(DEVICE)` AT IMPORT, which on a machine with a GPU makes bare
#: `torch.as_tensor` allocate on cuda -- and then `.numpy()` raises, the dataset writer
#: gets device tensors it will later `torch.save`, and the failure surfaces somewhere
#: unrelated. The map is cheap enough that the CPU is the right place for it anyway.
_CPU = "cpu"


def _self_collision_pairs(spec):
    """Sub-link index pairs that are checked, matching the SDF's filter groups exactly.

    Within a segment and between adjacent segments the spheres overlap by construction --
    that is what makes the union a rod rather than a string of beads -- so those pairs are
    filtered there and must be filtered here. If the two ever disagree, the dataset would
    be screened against a different robot than the one the solver sees.
    """
    segment_of = []
    for segment in range(spec.num_segments):
        segment_of += [segment] * spec.sublinks_per_segment
    segment_of.append(spec.num_segments - 1)          # the tip joins the last segment
    first, second = [], []
    for i in range(len(segment_of)):
        for j in range(i + 1, len(segment_of)):
            if abs(segment_of[i] - segment_of[j]) >= 2:
                first.append(i)
                second.append(j)
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


class SoftArmRobot(Robot):
    """One rung of the soft arm, wearing jrl's interface."""

    name = "soft12"
    formal_robot_name = "Soft PCS continuum arm (12 DoF)"
    spec: SoftArmSpec = RUNGS["soft12"]

    def __init__(self, verbose: bool = False):
        # Deliberately no `super().__init__`: it parses a URDF, loads klampt and runs a
        # warm-up batch FK, none of which exists for this robot.
        self._spec = type(self).spec
        self._name = type(self).name
        self._ndof = self._spec.ndof
        self._actuated_joint_limits = [(-1.0, 1.0)] * self._ndof
        self._actuated_joint_names = list(self._spec.dof_names)
        self._actuated_joint_velocity_limits = [1.0] * self._ndof
        self._verbose = verbose
        first, second = _self_collision_pairs(self._spec)
        self._pair_first = torch.from_numpy(first).to(_CPU)
        self._pair_second = torch.from_numpy(second).to(_CPU)
        # Two spheres collide when their centres are closer than the sum of the radii.
        self._collision_distance = 2.0 * self._spec.collision_radius

    # -- identity ---------------------------------------------------------------

    @property
    def ndof(self):
        return self._ndof

    @property
    def actuated_joints_limits(self):
        return self._actuated_joint_limits

    @property
    def actuated_joint_names(self):
        return self._actuated_joint_names

    @property
    def actuated_joints_velocity_limits(self):
        return self._actuated_joint_velocity_limits

    @property
    def urdf_filepath(self):
        raise NotImplementedError(
            f"{self._name} has no URDF: its configuration is per-segment strain, which a "
            f"kinematic chain cannot express. The Drake model in models/soft_arm/ is a "
            f"collision discretization driven from outside, not a description of the "
            f"degrees of freedom.")

    def __str__(self):
        return f"<SoftArmRobot {self._name}, {self._ndof} DoF>"

    __repr__ = __str__

    # -- sampling ---------------------------------------------------------------

    def sample_joint_angles(self, n: int, joint_limit_eps: float = 1e-6) -> np.ndarray:
        """Uniform over the configuration box, inset by `joint_limit_eps` as jrl does."""
        low, high = -1.0 + joint_limit_eps, 1.0 - joint_limit_eps
        return np.random.uniform(low, high, size=(n, self._ndof))

    def sample_joint_angles_and_poses(self, n: int, joint_limit_eps: float = 1e-6,
                                      only_non_self_colliding: bool = True,
                                      tqdm_enabled: bool = False,
                                      return_torch: bool = False):
        """Configurations and their tip poses.

        Overridden rather than inherited: jrl's version computes each pose through klampt
        one configuration at a time, which is where its ~14 us/config comes from. Ours is
        the same batched torch map the solver uses, so the dataset and the program cannot
        be describing different robots.
        """
        batch = max(1, min(n, 100_000))
        configurations, poses = [], []
        remaining = n
        progress = None
        if tqdm_enabled:
            from tqdm import tqdm
            progress = tqdm(total=n, desc=f"{self._name} samples")
        rejected_in_a_row = 0
        while remaining > 0:
            draw = torch.as_tensor(self.sample_joint_angles(batch, joint_limit_eps),
                                   dtype=torch.float64, device=_CPU)
            if only_non_self_colliding:
                keep = ~self.config_self_collides(draw)
                draw = draw[keep]
            if draw.shape[0] == 0:
                rejected_in_a_row += 1
                if rejected_in_a_row > 20:
                    raise RuntimeError(
                        f"{self._name}: {20 * batch} consecutive configurations all "
                        f"self-collide. The strain limits and the collision radius "
                        f"disagree about what this robot is.")
                continue
            rejected_in_a_row = 0
            draw = draw[:remaining]
            configurations.append(draw)
            poses.append(K.forward_kinematics(draw, self._spec, device=_CPU))
            remaining -= draw.shape[0]
            if progress is not None:
                progress.update(draw.shape[0])
        if progress is not None:
            progress.close()
        samples = torch.cat(configurations, dim=0)
        endpoints = torch.cat(poses, dim=0)
        if return_torch:
            return samples, endpoints
        return samples.cpu().numpy(), endpoints.cpu().numpy()

    # -- kinematics -------------------------------------------------------------

    def forward_kinematics(self, x, out_device=None, dtype=torch.float64,
                           return_quaternion: bool = True, return_full_joint_fk: bool = False,
                           return_full_link_fk: bool = False):
        """Tip pose as `[x, y, z, qw, qx, qy, qz]`, batched."""
        if return_full_joint_fk or return_full_link_fk:
            raise NotImplementedError(
                f"{self._name} has no joints or links in jrl's sense; ask the Drake model "
                f"for body poses, or src.soft_arm.kinematics.sublink_poses.")
        if not return_quaternion:
            raise NotImplementedError(
                f"{self._name} returns quaternions only; the rotation-matrix form is "
                f"unused on every path this project takes.")
        ## Computed on the CPU, returned on the CALLER'S device. jrl's own implementation
        ## returns on the device it was given, and ikflow's validation relies on that: it
        ## compares this against target poses that live on cuda, so a CPU return raises
        ## "Expected all tensors to be on the same device" -- inside validation_step, which
        ## is the first eval AFTER training has started, not at construction.
        device = x.device if isinstance(x, torch.Tensor) else _CPU
        tensor = torch.as_tensor(x, dtype=dtype, device=_CPU)
        poses = K.forward_kinematics(tensor, self._spec, dtype=dtype, device=_CPU)
        return poses.to(out_device if out_device is not None else device)

    def clamp_to_joint_limits(self, x):
        ## Device-preserving by construction: clamp does not move anything.
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, -1.0, 1.0)
        return np.clip(x, -1.0, 1.0)

    def config_self_collides(self, x, verbose: bool = False):
        """Do any two UNFILTERED collision spheres overlap?

        Batched: returns a bool tensor for a batch, a plain bool for one configuration --
        the shape jrl's callers expect on each path.
        """
        ## Returns a bool tensor on the caller's device, for the same reason
        ## `forward_kinematics` does: ikflow's validation indexes it against cuda tensors.
        device = x.device if isinstance(x, torch.Tensor) else _CPU
        tensor = torch.as_tensor(x, dtype=torch.float64, device=_CPU)
        single = tensor.dim() == 1
        if single:
            tensor = tensor.unsqueeze(0)
        centres = K.sphere_centers(tensor, self._spec, device=_CPU)
        separation = torch.linalg.norm(centres[:, self._pair_first, :]
                                       - centres[:, self._pair_second, :], dim=-1)
        collides = (separation < self._collision_distance).any(dim=-1)
        return bool(collides[0]) if single else collides.to(device)

    # -- the paths this project never takes, failing loudly rather than silently ----

    def _unsupported(self, what, why):
        raise NotImplementedError(f"{self._name}: {what} is not implemented -- {why}")

    def inverse_kinematics_step_levenburg_marquardt(self, *args, **kwargs):
        self._unsupported(
            "Levenberg-Marquardt IK refinement",
            "it is reached only from generate_ik_solutions(refine_solutions=True), which "
            "no path in this project sets; solving IK is what the Drake program does")

    def forward_kinematics_klampt(self, *args, **kwargs):
        self._unsupported("klampt forward kinematics",
                          "there is no URDF for klampt to load")

    def _x_to_qs(self, *args, **kwargs):
        self._unsupported("the klampt configuration conversion", "there is no klampt model")

    def set_klampt_robot_config(self, *args, **kwargs):
        self._unsupported("setting a klampt configuration", "there is no klampt model")

    @property
    def klampt_world_model(self):
        self._unsupported("the klampt world model", "there is no URDF to load into one")

    def jacobian(self, *args, **kwargs):
        self._unsupported(
            "jrl's task Jacobian",
            "use torch.func on src.soft_arm.kinematics.forward_kinematics, which is the "
            "same map the solver differentiates")

    def self_collision_distances(self, *args, **kwargs):
        self._unsupported(
            "jrl's capsule self-collision distances",
            "this robot's collision geometry is spheres; config_self_collides is the "
            "screen, and the Drake program uses MinimumDistanceLowerBoundConstraint")


def _make_rung(spec: SoftArmSpec):
    return type(f"SoftArmRobot_{spec.name}", (SoftArmRobot,),
                {"name": spec.name, "spec": spec,
                 "formal_robot_name": f"Soft PCS continuum arm ({spec.ndof} DoF)"})


#: One `Robot` subclass per rung, so `get_robot("soft9")` works like any other robot.
ROBOT_CLASSES = tuple(_make_rung(spec) for spec in RUNGS.values())
