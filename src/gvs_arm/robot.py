"""A `jrl.robot.Robot` for the GVS push-rod arm, so ikflow can train on it unmodified.

The soft PCS arm's shim (`src/soft_arm/robot.py`), applied to a robot whose "joint angles" are
rod forces: ikflow reaches its robot through `jrl.robots.get_robot`, `IKFlowSolver.__init__`
asserts `isinstance(robot, Robot)`, and jrl's base constructor would demand a URDF and klampt,
so this SUBCLASSES `Robot` and overrides `__init__` without calling `super().__init__`.
`name` is a CLASS attribute because `get_robot` compares `clc.name` on the class.

WHAT IKFLOW ACTUALLY TOUCHES, and therefore what is implemented here:

  `name`, `ndof`                 -- the model's input width (9) and the dataset directory
  `actuated_joints_limits`       -- `[-1, 1]` per rod, baked into `module_list.0`
  `sample_joint_angles`          -- uniform over the normalized force box
  `sample_joint_angles_and_poses`-- the DATASET: draws, their equilibrium tip poses, and the
                                    self-collision screen, all batched through SoRoMoX
  `forward_kinematics`           -- batched `[N, 7]` as xyz + wxyz; validation's pose error
                                    and `CalibrateFlowFrame`
  `clamp_to_joint_limits`, `config_self_collides` -- validation

Everything else raises `NotImplementedError` naming why, so an unreachable path fails at once
rather than as an `AttributeError` four hundred thousand steps into a training run.

A DRAW THAT DOES NOT REACH EQUILIBRIUM IS REJECTED AND COUNTED, never written. Newton
converged on every draw measured so far (`tests/test_gvs_arm_model.py`), so the count is
expected to stay zero; it is recorded so that a rung whose statics turn multi-valued shows
up as a number rather than as a quietly biased dataset.
"""

import os

import numpy as np
import torch

from jrl.robot import Robot

from src.gvs_arm.model import GetModel
from src.gvs_arm.params import RUNGS, GvsArmSpec

#: Everything here is built explicitly on the CPU. `jrl.config` calls
#: `torch.set_default_device(DEVICE)` AT IMPORT, so a bare `torch.as_tensor` would allocate
#: on cuda and `.numpy()` would raise somewhere unrelated. The model is JAX on the CPU anyway.
_CPU = "cpu"


class GvsArmRobot(Robot):
    """One rung of the GVS push-rod arm, wearing jrl's interface."""

    name = "gvs_pushrod9_o1"
    formal_robot_name = "GVS push-rod continuum arm (9 inputs, order 1)"
    spec: GvsArmSpec = RUNGS["gvs_pushrod9_o1"]

    def __init__(self, verbose: bool = False):
        # Deliberately no `super().__init__`: it parses a URDF, loads klampt and runs a
        # warm-up batch FK, none of which exists for this robot.
        self._spec = type(self).spec
        self._name = type(self).name
        self._ndof = self._spec.ninputs
        self._actuated_joint_limits = [(-1.0, 1.0)] * self._ndof
        self._actuated_joint_names = list(self._spec.input_names)
        self._actuated_joint_velocity_limits = [1.0] * self._ndof
        self._verbose = verbose
        self._model = GetModel(self._spec)
        self.rejected_unconverged = 0

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
            f"{self._name} has no URDF: its configuration is a rod-force vector whose "
            f"kinematics are SoRoMoX's equilibrium. The Drake model in models/{self._name}/ "
            f"is a collision discretization driven from outside, not a description of the "
            f"degrees of freedom.")

    def __str__(self):
        return f"<GvsArmRobot {self._name}, {self._ndof} inputs>"

    __repr__ = __str__

    # -- sampling ---------------------------------------------------------------

    def sample_joint_angles(self, n: int, joint_limit_eps: float = 1e-6) -> np.ndarray:
        """Uniform over the normalized force box, inset by `joint_limit_eps` as jrl does."""
        low, high = -1.0 + joint_limit_eps, 1.0 - joint_limit_eps
        return np.random.uniform(low, high, size=(n, self._ndof))

    def sample_joint_angles_and_poses(self, n: int, joint_limit_eps: float = 1e-6,
                                      only_non_self_colliding: bool = True,
                                      tqdm_enabled: bool = False,
                                      return_torch: bool = False):
        """Rod-force draws and their equilibrium tip poses, batched through SoRoMoX.

        The batch is sized for the JAX `vmap`: every lane runs Newton until the slowest lane
        converges, and the assembly is memory-light, so tens of thousands per call is the
        sweet spot measured on the laptop.
        """
        ## `GVS_ARM_SAMPLE_BATCH` lets a process-parallel build use smaller batches: the
        ## vmapped solve's intermediates scale with the batch, and 96 workers at 20000 each
        ## sat near a 187 GB node's memory limit.
        batch = max(1, min(n, int(os.environ.get("GVS_ARM_SAMPLE_BATCH", "20000"))))
        configurations, poses = [], []
        remaining = n
        progress = None
        if tqdm_enabled:
            from tqdm import tqdm
            progress = tqdm(total=n, desc=f"{self._name} samples")
        rejected_in_a_row = 0
        while remaining > 0:
            draw = self.sample_joint_angles(batch, joint_limit_eps)
            tips, converged = self._model.TipPoseBatch(draw)
            self.rejected_unconverged += int((~converged).sum())
            keep = converged
            if only_non_self_colliding:
                collides, _ = self._model.SelfCollides(draw)
                keep = keep & ~collides
            draw, tips = draw[keep], tips[keep]
            if draw.shape[0] == 0:
                rejected_in_a_row += 1
                if rejected_in_a_row > 20:
                    raise RuntimeError(
                        f"{self._name}: {20 * batch} consecutive draws all self-collide or "
                        f"fail to reach equilibrium; the force scales and the collision "
                        f"radii disagree about what this robot is.")
                continue
            rejected_in_a_row = 0
            draw, tips = draw[:remaining], tips[:remaining]
            configurations.append(draw)
            poses.append(tips)
            remaining -= draw.shape[0]
            if progress is not None:
                progress.update(draw.shape[0])
        if progress is not None:
            progress.close()
        samples = np.concatenate(configurations, axis=0)
        endpoints = np.concatenate(poses, axis=0)
        if return_torch:
            return (torch.as_tensor(samples, dtype=torch.float64, device=_CPU),
                    torch.as_tensor(endpoints, dtype=torch.float64, device=_CPU))
        return samples, endpoints

    # -- kinematics -------------------------------------------------------------

    def forward_kinematics(self, x, out_device=None, dtype=torch.float64,
                           return_quaternion: bool = True, return_full_joint_fk: bool = False,
                           return_full_link_fk: bool = False):
        """Tip-frame pose as `[x, y, z, qw, qx, qy, qz]`, batched, on the CALLER'S device."""
        if return_full_joint_fk or return_full_link_fk:
            raise NotImplementedError(
                f"{self._name} has no joints or links in jrl's sense; ask the Drake model "
                f"for body poses, or src.gvs_arm.model.GvsArmModel.PlantQ.")
        if not return_quaternion:
            raise NotImplementedError(
                f"{self._name} returns quaternions only; the rotation-matrix form is "
                f"unused on every path this project takes.")
        ## Computed on the CPU, returned on the caller's device: ikflow's validation compares
        ## this against target poses that live on cuda.
        device = x.device if isinstance(x, torch.Tensor) else _CPU
        array = np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=float)
        single = array.ndim == 1
        if single:
            array = array[None]
        poses, converged = self._model.TipPoseBatch(array)
        if not converged.all():
            raise RuntimeError(f"{self._name}: {int((~converged).sum())} configurations did "
                               f"not reach equilibrium in forward_kinematics")
        tensor = torch.as_tensor(poses[0] if single else poses, dtype=dtype, device=_CPU)
        return tensor.to(out_device if out_device is not None else device)

    def clamp_to_joint_limits(self, x):
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, -1.0, 1.0)
        return np.clip(x, -1.0, 1.0)

    def config_self_collides(self, x, verbose: bool = False):
        """Do any two UNFILTERED collision spheres overlap? Batched; bool for one config."""
        device = x.device if isinstance(x, torch.Tensor) else _CPU
        array = np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=float)
        single = array.ndim == 1
        if single:
            array = array[None]
        collides, _ = self._model.SelfCollides(array)
        if single:
            return bool(collides[0])
        return torch.as_tensor(collides, device=_CPU).to(device)

    # -- the paths this project never takes, failing loudly rather than silently ----

    def _unsupported(self, what, why):
        raise NotImplementedError(f"{self._name}: {what} is not implemented -- {why}")

    def inverse_kinematics_step_levenburg_marquardt(self, *args, **kwargs):
        self._unsupported("Levenberg-Marquardt IK refinement",
                          "solving IK is what the Drake program does")

    def forward_kinematics_klampt(self, *args, **kwargs):
        self._unsupported("klampt forward kinematics", "there is no URDF for klampt to load")

    def _x_to_qs(self, *args, **kwargs):
        self._unsupported("the klampt configuration conversion", "there is no klampt model")

    def set_klampt_robot_config(self, *args, **kwargs):
        self._unsupported("setting a klampt configuration", "there is no klampt model")

    @property
    def klampt_world_model(self):
        self._unsupported("the klampt world model", "there is no URDF to load into one")

    def jacobian(self, *args, **kwargs):
        self._unsupported("jrl's task Jacobian",
                          "use src.gvs_arm.model.GvsArmModel.PlantJacobian, the same implicit "
                          "map the solver differentiates")

    def self_collision_distances(self, *args, **kwargs):
        self._unsupported("jrl's capsule self-collision distances",
                          "this robot's collision geometry is spheres; config_self_collides "
                          "is the screen, and the Drake program uses "
                          "MinimumDistanceLowerBoundConstraint")


def _make_rung(spec: GvsArmSpec):
    return type(f"GvsArmRobot_{spec.name}", (GvsArmRobot,),
                {"name": spec.name, "spec": spec,
                 "formal_robot_name": (f"GVS push-rod continuum arm ({spec.ninputs} inputs, "
                                       f"order {spec.basis_order})")})


#: One `Robot` subclass per rung, so `get_robot("gvs_pushrod9_o1")` works like any other robot.
ROBOT_CLASSES = tuple(_make_rung(spec) for spec in RUNGS.values())
