"""A `jrl.robot.Robot` for `helix7`, so ikflow can train on it unmodified.

ikflow reaches its robot through `jrl.robots.get_robot`, and jrl's `Robot` is URDF- and
klampt-driven. Neither can carry this arm: `jrl/urdf_utils.py:parse_urdf` matches elements
by `child.tag == "joint"` and knows only `revolute / continuous / prismatic / fixed`, and
jrl's batched forward kinematics asserts on anything else. There is no URDF to give it
either -- URDF has no screw joint type at all.

So this SUBCLASSES `Robot` and overrides `__init__` WITHOUT calling `super().__init__`,
which would demand a URDF, load it into klampt and run a warm-up batch FK. The subclassing
is not stylistic: `IKFlowSolver.__init__` asserts `isinstance(robot, Robot)`, so a
duck-typed class fails on the first load. `name` is a CLASS attribute for a related reason
-- `get_robot` compares `clc.name` on the class before instantiating anything.

WHAT IKFLOW ACTUALLY TOUCHES, and therefore what is implemented here:

  `name`, `ndof`                  -- the model's input width and the dataset directory
  `actuated_joints_limits`        -- baked into `module_list.0` as a fixed rescaling
  `sample_joint_angles`           -- dataset draws
  `sample_joint_angles_and_poses` -- dataset draws and their poses
  `forward_kinematics`            -- batched, torch, [N, 7] as xyz + wxyz; used by the
                                     dataset, by validation's pose error, and by
                                     `CalibrateFlowFrame`
  `clamp_to_joint_limits`         -- validation
  `config_self_collides`          -- validation and `--only_non_self_colliding`

Everything else jrl exposes that ikflow can reach raises `NotImplementedError` naming why,
so an unreachable path fails immediately rather than four hundred thousand steps into a run.

THE JOINT LIMITS REACH THE ARCHITECTURE, so they are not a free choice. With
`sigmoid_on_output` false -- the configuration every checkpoint in this project is trained
at -- ikflow builds its first node as `x_i / max(|lo_i|, |hi_i|)`, a PURE SCALING with no
offset (`ikflow/model.py`, the `else` branch; the affine-with-offset version is the
`sigmoid_on_output` branch, which nothing here uses). A coordinate whose range is not
symmetric about zero therefore lands off-centre in an input range the first layer cannot
recentre. `params.py` keeps the screw joint's range symmetric for exactly that reason, and
it is the same reason the soft arm normalises its strains about zero.
"""

import numpy as np
import torch

from jrl.robot import Robot

from src.helix_arm import kinematics as K
from src.helix_arm.params import SPECS, HelixArmSpec

#: Everything here is COMPUTED on the CPU. `jrl.config` calls `torch.set_default_device`
#: AND `torch.set_default_dtype(torch.float32)` AT IMPORT, so on a machine with a GPU a bare
#: `torch.as_tensor` allocates float32 on cuda -- and then `.numpy()` raises, the dataset
#: writer gets device tensors it will later `torch.save`, and the failure surfaces somewhere
#: unrelated. The map is cheap enough that the CPU is the right place for it anyway.
_CPU = "cpu"


class HelixArmRobot(Robot):
    """One rung of the helical-joint arm, wearing jrl's interface."""

    name = "helix7_p050"
    formal_robot_name = "Helical-joint 7-DoF arm (0.050 m/rev)"
    spec: HelixArmSpec = SPECS["helix7_p050"]

    def __init__(self, verbose: bool = False):
        # Deliberately no `super().__init__`: it parses a URDF, loads klampt and runs a
        # warm-up batch FK, none of which exists for this robot.
        self._spec = type(self).spec
        self._name = type(self).name
        self._ndof = self._spec.ndof
        lower, upper = self._spec.limits
        self._actuated_joint_limits = [(lo, hi) for lo, hi in zip(lower, upper)]
        self._actuated_joint_names = list(self._spec.joint_names)
        self._actuated_joint_velocity_limits = [2.0] * self._ndof
        self._verbose = verbose
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)

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
            f"{self._name} has no URDF: URDF cannot express a screw joint, whose spec lists "
            f"only planar, floating, revolute, continuous, prismatic and fixed. The Drake "
            f"model is SDFormat, at models/{self._name}/{self._name}.sdf, and it is "
            f"generated from src/helix_arm/params.py rather than being the definition.")

    def __str__(self):
        return f"<HelixArmRobot {self._name}, {self._ndof} DoF, pitch {self._spec.pitch} m/rev>"

    __repr__ = __str__

    # -- sampling ---------------------------------------------------------------

    def sample_joint_angles(self, n: int, joint_limit_eps: float = 1e-6) -> np.ndarray:
        """Uniform over the configuration box, inset by `joint_limit_eps` as jrl does."""
        return np.random.uniform(self._lower + joint_limit_eps,
                                 self._upper - joint_limit_eps,
                                 size=(n, self._ndof))

    def sample_joint_angles_and_poses(self, n: int, joint_limit_eps: float = 1e-6,
                                      only_non_self_colliding: bool = True,
                                      tqdm_enabled: bool = False,
                                      return_torch: bool = False):
        """Configurations and their flange poses.

        Overridden rather than inherited: jrl's version computes each pose through klampt
        one configuration at a time. Ours is the same batched torch map the solver's model
        is checked against, so the dataset and the program cannot be describing different
        robots.
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
                draw = draw[~self.config_self_collides(draw)]
            if draw.shape[0] == 0:
                rejected_in_a_row += 1
                if rejected_in_a_row > 20:
                    raise RuntimeError(
                        f"{self._name}: {20 * batch} consecutive configurations all "
                        f"self-collide. The joint limits and the collision spheres disagree "
                        f"about what this robot is.")
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
                           return_quaternion: bool = True,
                           return_full_joint_fk: bool = False,
                           return_full_link_fk: bool = False):
        """Flange pose as `[x, y, z, qw, qx, qy, qz]`, batched."""
        if return_full_joint_fk or return_full_link_fk:
            raise NotImplementedError(
                f"{self._name}: jrl's full-chain forward kinematics returns transforms in "
                f"its own parsed-chain order, which this robot has no parsed chain for. Ask "
                f"src.helix_arm.kinematics.link_poses, or the Drake plant.")
        if not return_quaternion:
            raise NotImplementedError(
                f"{self._name} returns quaternions only; the rotation-matrix form is unused "
                f"on every path this project takes.")
        ## Computed on the CPU, returned on the CALLER'S device. jrl's own implementation
        ## returns on the device it was given, and ikflow's validation relies on that: it
        ## compares this against target poses that live on cuda, so a CPU return raises
        ## "Expected all tensors to be on the same device" -- inside `validation_step`,
        ## which is the first eval AFTER training starts, not at construction. A cluster run
        ## would burn its whole queue wait before saying so.
        device = x.device if isinstance(x, torch.Tensor) else _CPU
        tensor = torch.as_tensor(x, dtype=dtype, device=_CPU)
        poses = K.forward_kinematics(tensor, self._spec, dtype=dtype, device=_CPU)
        return poses.to(out_device if out_device is not None else device)

    def clamp_to_joint_limits(self, x):
        ## Device-preserving by construction: clamping does not move anything.
        if isinstance(x, torch.Tensor):
            lower = torch.as_tensor(self._lower, dtype=x.dtype, device=x.device)
            upper = torch.as_tensor(self._upper, dtype=x.dtype, device=x.device)
            return torch.clamp(x, lower, upper)
        return np.clip(x, self._lower, self._upper)

    def config_self_collides(self, x, verbose: bool = False):
        """Do any two unfiltered collision spheres overlap?

        Sphere against sphere, which is EXACTLY the geometry Drake's constraint sees on this
        robot rather than a conservative stand-in -- and a `cdist` rather than the per-pair
        QP jrl solves for capsules, which is what makes screening a 25M-sample dataset
        affordable.

        Batched: a bool tensor for a batch, a plain bool for one configuration, which are
        the shapes jrl's two calling paths expect. `evaluate_solutions` calls it per
        configuration.
        """
        ## Returns on the caller's device, for the same reason `forward_kinematics` does:
        ## ikflow's validation indexes the result against cuda tensors.
        device = x.device if isinstance(x, torch.Tensor) else _CPU
        tensor = torch.as_tensor(x, dtype=torch.float64, device=_CPU)
        single = tensor.dim() == 1
        if single:
            tensor = tensor.unsqueeze(0)
        collides = K.self_collision_depth(tensor, self._spec, device=_CPU) > 0.0
        return bool(collides[0]) if single else collides.to(device)

    # -- the paths this project never takes, failing loudly rather than silently ----

    def _unsupported(self, what, why):
        raise NotImplementedError(f"{self._name}: {what} is not implemented -- {why}")

    def inverse_kinematics_step_levenburg_marquardt(self, *args, **kwargs):
        self._unsupported(
            "Levenberg-Marquardt IK refinement",
            "it is reached only from generate_ik_solutions(refine_solutions=True), which no "
            "path in this project sets; solving IK is what the Drake program does")

    def forward_kinematics_klampt(self, *args, **kwargs):
        self._unsupported("klampt forward kinematics",
                          "there is no URDF for klampt to load, and klampt has no screw "
                          "joint to load it into")

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
            "its per-joint switch has no screw branch, so it would silently emit a revolute "
            "column and drop the axial term; use torch.func on "
            "src.helix_arm.kinematics.forward_kinematics")

    def jacobian_np(self, *args, **kwargs):
        self._unsupported("jrl's numpy Jacobian", "it goes through klampt's getJacobian")

    def jacobian_batch_np(self, *args, **kwargs):
        self._unsupported("jrl's batched numpy Jacobian", "it goes through klampt")

    def config_collides_with_env(self, *args, **kwargs):
        self._unsupported(
            "jrl's environment collision check",
            "the scene lives in Drake; the program uses MinimumDistanceLowerBoundConstraint")

    def self_collision_distances(self, *args, **kwargs):
        self._unsupported(
            "jrl's capsule self-collision distances",
            "this robot's collision geometry is spheres; config_self_collides is the screen "
            "and the Drake program carries the constraint")


def _make_rung(spec: HelixArmSpec):
    return type(f"HelixArmRobot_{spec.name}", (HelixArmRobot,),
                {"name": spec.name, "spec": spec,
                 "formal_robot_name":
                     f"Helical-joint 7-DoF arm ({spec.pitch:.3f} m/rev)"})


#: One `Robot` subclass per rung, so `get_robot("helix7_p100")` works like any other robot.
ROBOT_CLASSES = tuple(_make_rung(spec) for spec in SPECS.values())
