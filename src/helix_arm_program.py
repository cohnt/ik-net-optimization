"""The four `helix7` programs: learned and joint space, pose task and grasp task.

Built on `src/iiwa_program.py`'s template rather than the soft arm's, and the reason is the
one structural fact about this robot: ITS CONFIGURATION IS THE PLANT'S POSITION VECTOR. So
every hook the shared machinery offers for a robot whose configuration is something else --
`Config`, `ConfigLimits`, `ConfigToPlantQ`, `SampleConfiguration`, `VerificationQ` -- is
correct at its default here, and none of it is overridden. The arm is an ordinary rigid
serial chain; the only thing unusual about it is one joint's motion.

There is no analytic column, and unlike the iiwa's that is not a matter of effort. A
closed-form inverse kinematics needs the forward kinematics to be an ALGEBRAIC function of
the joint variables, and a helical joint contributes `cos q`, `sin q` and `q` at once.
`ALL_ARMS` records this robot as `learned,numerical` for that reason.

THE ONE THING THAT MUST HAPPEN IN A PARTICULAR ORDER is the screw-limit repair; see
`__init__`. Everything else follows the iiwa.
"""

import os

import numpy as np
import torch
from ikflow.config import DEVICE

from pydrake.all import (AutoDiffXd, MathematicalProgram, Quaternion, RigidTransform,
                         RollPitchYaw, RollPitchYaw_, RotationMatrix, RotationMatrix_)

import src.register_robots  # noqa: F401  -- before anything resolves a robot by name
from src.flow_loading import LEGACY_ARCH_BY_ROBOT, LoadFlowSolver
from src.generic_program import *  # noqa: F401,F403
from src.generic_program import (IKFlowConstraints, IKFlowProgram, ProgramOptions,
                                 regularize_jacobian)
from src.helix_arm.limits import ApplyScrewJointLimits, RequireFiniteLimits
from src.helix_arm.params import PRIMARY, GetSpec
from src.utils import Mug


class HelixArmIKProgram(IKFlowProgram):
    """The learned formulation on the pose task: free conditioning pose, latent, correction."""

    def __init__(self, diagram, options=ProgramOptions(), robot=PRIMARY,
                 model=None, checkpoint=None):
        self.spec = GetSpec(robot)
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")

        ## THE SCREW-LIMIT REPAIR, AND IT MUST SIT EXACTLY HERE.
        ##
        ## Drake's parsers call ParseJointLimits only for revolute and prismatic joints, so
        ## a screw joint's <limit> is read and discarded and the plant reports [-inf, inf].
        ## Nothing crashes: the joint-limit row goes vacuous on that coordinate -- the one
        ## row this robot exists to stress -- the joint-space arm's bounding box becomes
        ## unbounded, and the target sampler's rng.uniform(lower, upper) returns nan.
        ##
        ## It must come AFTER Finalize(), which BuildEnv does internally, and BEFORE
        ## ToAutoDiffXd(): that call takes an independent scalar-converted COPY, so a copy
        ## made first carries the infinities for ever with nothing downstream to say so.
        ## tests/test_helix_arm_kinematics.py asserts both halves.
        ##
        ## And it lives in __init__ rather than in the benchmark driver because
        ## GenerateDiagramWithMug builds a FRESH DIAGRAM PER TARGET -- a one-shot repair in
        ## the driver would cover the pose scene and silently miss every grasp scene.
        self._screw_limits = ApplyScrewJointLimits(self.plant, self.spec)

        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context)

        frame_name = self.TargetFrameName()
        self.frame = self.plant.GetFrameByName(frame_name)
        self.autodiff_frame = self.autodiff_plant.GetFrameByName(frame_name)

        ## `ee_frame` is the frame the FLOW speaks in, and it is the flange on every task --
        ## `HelixArmRobot.forward_kinematics` returns the flange pose, so that is what the
        ## chart was trained against. `self.frame` is where the TASK acts, which the grasp
        ## subclass moves to `between_fingers`.
        ##
        ## It is set HERE, before `CalibrateFlowFrame`, and that ordering is the whole
        ## point. `frame_for_flow` falls back to `self.frame` when `ee_frame` is absent, so
        ## a grasp program that set it after calling `super().__init__()` would calibrate
        ## `between_fingers` against the flow and then apply that 0.200 m offset to the
        ## flange -- conditioning the network 0.200 m from where it thinks it is. Nothing
        ## would raise: the offset IS constant, which is all the calibration checks. The
        ## iiwa avoids this by calibrating a second time in its grasp subclass; setting the
        ## frame first means one calibration is enough and there is no second call to
        ## forget.
        self.ee_frame = self.plant.GetFrameByName(self.spec.flange_link)

        self.num_pos = self.plant.num_positions()
        self.num_arm_dof = self.spec.ndof
        ## The conditioning pose is xyz + rpy, as on every other robot.
        self.num_task_vars = 6

        if self.num_pos != self.num_arm_dof:
            ## True on the rigid arms, which carry two finray finger joints. This gripper
            ## model has none, so the two coincide and PadQ's tail is empty -- stated rather
            ## than assumed, because a gripper change would make it false silently.
            pass

        if model is None:
            if checkpoint is None:
                raise ValueError(
                    f"{self.spec.name} has no published chart, so --checkpoint is required. "
                    f"A missing one is not an error you would notice: it is a column of "
                    f"zeros.")
            self.ik_solver = LoadFlowSolver(
                self.spec.name, checkpoint,
                fallback_arch=LEGACY_ARCH_BY_ROBOT[self.spec.name])
        else:
            self.ik_solver = model

        self.options = options
        self.ConfigureNetworkDtype()
        self.constraints = []

        RequireFiniteLimits(self.plant, f"{self.spec.name} plant")
        RequireFiniteLimits(self.autodiff_plant, f"{self.spec.name} autodiff plant")

        ## The flow is conditioned on the frame it was TRAINED on, which is the flange --
        ## the frame `HelixArmRobot.forward_kinematics` returns. `self.frame` is whatever
        ## the scene calls the end effector, and the two are the same frame only by luck.
        ## A skipped calibration is indistinguishable from a correct one until the geometry
        ## it silently relied on changes; on the Panda that cost a whole column, 10/60 with
        ## median max_violation 0.4. Here it is also a free end-to-end witness: if the pitch
        ## sign, the axis normalisation or the SDF's units disagreed with the torch map, the
        ## measured offset would vary with the screw coordinate and this would raise.
        self.CalibrateFlowFrame()

    def TargetFrameName(self):
        """The frame the task's target pose is expressed in. The grasp subclass moves it."""
        return self.spec.flange_link

    ## -- the program -------------------------------------------------------------

    def create_prog(self, target_pose=np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal=None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6)
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width)
        self.correction = self.prog.NewContinuousVariables(self.num_arm_dof)

        self.lumped_vars = np.hstack([self.c, self.z, self.correction])
        if getattr(self.options, "lift_q", False):
            self.q_lift = self.prog.NewContinuousVariables(self.num_arm_dof, "q_lift")
            self.lumped_vars = np.hstack([self.lumped_vars, self.q_lift])

        self.target_pose = target_pose
        self.q_nominal = np.zeros(self.num_pos) if q_nominal is None else q_nominal

        ## `c` is the NETWORK's conditioning input, so it lives in the frame the flow was
        ## trained on rather than in whatever frame the scene calls the end effector.
        X_W_target = RigidTransform(Quaternion(self.target_pose[3:]), self.target_pose[:3])
        X_W_flow = X_W_target @ self.X_ee_flow
        self.initial_guess = np.zeros(6)
        self.initial_guess[:3] = X_W_flow.translation()
        self.initial_guess[3:] = RollPitchYaw(X_W_flow.rotation()).vector()

        self.prog.SetInitialGuess(self.c, self.initial_guess)
        self.prog.SetInitialGuess(self.z, np.zeros(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))

        self.jacobian_gen = self.MakeJacobianGen()
        self.add_constraints()
        self.add_costs()

    ## -- the two maps ------------------------------------------------------------

    def ik_inference(self, vars, add_correction=True):
        """Latent + conditioning pose + correction -> joint coordinates.

        The body lives in `MakeFlowInference` so this eager path and the (possibly compiled)
        Jacobian evaluate literally the same code.
        """
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=self.torch_dtype)
        if not add_correction:
            width = self.ik_solver.network_width
            vars = torch.cat([vars[:7 + width],
                              torch.zeros(self.num_arm_dof, dtype=vars.dtype, device=DEVICE)])
        q, _ = self.FlowInference()(vars)
        return q

    def TaskVarsToPose7(self, task_vars, t):
        xyz = task_vars[:3]
        quaternion = RotationMatrix_[t](RollPitchYaw_[t](task_vars[3:6])).ToQuaternion().wxyz()
        return xyz, quaternion

    def PadQ(self, q_arm):
        """Arm coordinates -> a full plant position vector.

        The tail is empty on this robot: its gripper model carries no joints, so
        `num_pos == num_arm_dof`. Written generically anyway, so a gripper with finger
        joints would not silently leave them at zero.
        """
        q = np.zeros(self.num_pos)
        q[:self.num_arm_dof] = q_arm
        q[self.num_arm_dof:] = 0.04
        return q

    def VarsToQ(self, rpy_vars, add_correction=True):
        ad = isinstance(rpy_vars[0], AutoDiffXd)
        t = AutoDiffXd if ad else float
        width = self.ik_solver.network_width
        n = self.num_task_vars
        ndof = self.num_arm_dof

        vars = np.zeros(7 + width + ndof, dtype=AutoDiffXd if ad else np.float64)
        xyz, quaternion = self.TaskVarsToPose7(rpy_vars[:n], t)
        vars[:3] = xyz
        vars[3:7] = quaternion
        vars[7:7 + width] = rpy_vars[n:n + width]
        ## Explicit end index: under `lift_q` the lumped vector carries the lifted
        ## configuration after the correction, so an open-ended slice would be too wide.
        vars[7 + width:] = rpy_vars[n + width:n + width + ndof]

        if not ad:
            q = np.zeros(self.num_pos)
            q[ndof:] = 0.04
            q[:ndof] = self.ik_inference(vars, add_correction=add_correction).detach().cpu().numpy()
            return q

        vars_values = np.array([v.value() for v in vars])
        vars_gradients = np.array([v.derivatives() for v in vars])
        vars_tensor = torch.tensor(vars_values, dtype=self.torch_dtype, device=DEVICE)
        jacobian, q_tensor = self.jacobian_gen(vars_tensor)
        jacobian_np = regularize_jacobian(jacobian.detach().cpu().numpy(), self.options)

        q_values = np.zeros(self.num_pos)
        q_values[ndof:] = 0.04
        q_values[:ndof] = q_tensor.detach().cpu().numpy()

        q_gradients = np.zeros((self.num_pos, len(rpy_vars)))
        q_gradients[:ndof, :] = jacobian_np @ vars_gradients
        return np.array([AutoDiffXd(q_values[i], q_gradients[i])
                         for i in range(len(q_values))])


class HelixArmMugProgram(HelixArmIKProgram):
    """The learned formulation on the grasp task."""

    def TargetFrameName(self):
        return "between_fingers"

    def __init__(self, diagram, options=ProgramOptions(), robot=PRIMARY,
                 model=None, checkpoint=None):
        super().__init__(diagram, options, robot, model, checkpoint)
        ## The flow conditions on the flange; the grasp constraint acts between the fingers.
        ## Both frames are already in place -- `TargetFrameName` made `self.frame` the grasp
        ## frame from the start and the base `__init__` set `ee_frame` to the flange before
        ## calibrating -- so all that is left is the constant between them, which
        ## `SetStartFromQ` uses to express a seed in the network's frame.
        self.plant.SetPositions(self.plant_context, np.zeros(self.num_pos))
        X_W_flow = self.FlowPoseInWorld()
        X_W_grasp = self.frame.CalcPoseInWorld(self.plant_context)
        self.X_grasp_ee = X_W_grasp.inverse() @ X_W_flow

    def create_prog(self, target_mug=Mug(), q_nominal=None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6)
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width)
        self.correction = self.prog.NewContinuousVariables(self.num_arm_dof)

        self.lumped_vars = np.hstack([self.c, self.z, self.correction])
        if getattr(self.options, "lift_q", False):
            self.q_lift = self.prog.NewContinuousVariables(self.num_arm_dof, "q_lift")
            self.lumped_vars = np.hstack([self.lumped_vars, self.q_lift])

        self.target_mug = target_mug
        self.q_nominal = np.zeros(self.num_pos) if q_nominal is None else q_nominal

        self.prog.SetInitialGuess(self.c, [*target_mug.middle.translation(), 0, 0, 0])
        self.prog.SetInitialGuess(self.z, np.random.randn(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        self.jacobian_gen = self.MakeJacobianGen()

        self.target_pose = np.array([*target_mug.middle.translation(), 0, 0, 0])
        self.add_constraints()
        self.add_costs()

    def CreateIKConstraint(self):
        ## The gripper must lie on the mug's central axis (x = y = 0 EXACTLY, because that
        ## is the task) and within the mug's height along it. Orientation is free: any
        ## approach direction is a valid grasp.
        lb = np.array([0.0, 0.0, -self.options.mug_height])
        ub = np.array([0.0, 0.0, self.options.mug_height])

        def eval_func(vars, q, pose):
            position, _ = pose
            mug_transform = np.linalg.inv(self.target_mug.middle.GetAsMatrix4())
            ## Drop the homogeneous row: identically 1 with a zero gradient, so keeping it
            ## as an equality row costs a rank and breaks LICQ.
            return (mug_transform @ np.array([[*position, 1]]).T).squeeze()[:3]

        self.ik_constraint = IKFlowConstraints(lb, ub, eval_func, description="IKConstraint")
        self.constraints.append(self.ik_constraint)
        return self.ik_constraint

    def BoundingBoxConstraint(self):
        self.LatentBoxConstraint()
        centre = self.target_mug.middle.translation()
        slack = self.options.c_position_slack
        ## A general linear constraint, deliberately NOT a bounding box. IPOPT's bound_push
        ## projects the initial guess into every VARIABLE BOX before evaluating anything,
        ## which would teleport `c` to the face while the latent stayed tuned to the
        ## unprojected pose -- destroying the exact paired start. A general constraint may
        ## start violated; the violation just lands in inf_pr.
        self.c_box = (np.concatenate([centre - slack, -2 * np.pi * np.ones(3)]),
                      np.concatenate([centre + slack, 2 * np.pi * np.ones(3)]))
        self.c_box_constraint = self.prog.AddLinearConstraint(
            np.eye(6), self.c_box[0], self.c_box[1], self.c)
        self.c_box_constraint.evaluator().set_description("CBoxConstraint")
        bound = self.options.correction_bound
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -bound * np.ones(self.num_arm_dof), bound * np.ones(self.num_arm_dof),
            self.correction)
        self.correction_bounding_box_constraint.evaluator().set_description(
            "CorrectionBoundingBoxConstraint")


class _NumericalMixin:
    """The joint-space formulation: the decision variables ARE the joint coordinates.

    A mixin rather than the iiwa's four-way copy, and the record says why: the latent box
    was once repaired in `generic_program.py` while the grasp subclasses carried private
    copies of `BoundingBoxConstraint`, so the pose arms were fixed and the grasp arms
    silently were not. Listed FIRST in the MRO, so the grasp variant keeps the mug's
    `CreateIKConstraint` while taking its variables from here.
    """

    def _CreateVariables(self, q_nominal):
        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(self.num_arm_dof)
        self.lumped_vars = self.q
        self.q_nominal = (np.zeros(self.num_arm_dof) if q_nominal is None else q_nominal)
        self.prog.SetInitialGuess(self.q, self.q_nominal[:self.num_arm_dof])

    def VarsToQ(self, rpy_vars, add_correction=False):
        q = np.zeros(self.num_pos,
                     dtype=AutoDiffXd if isinstance(rpy_vars[0], AutoDiffXd) else float)
        q[self.num_arm_dof:] = 0.04
        q[:self.num_arm_dof] = rpy_vars[:self.num_arm_dof]
        return q

    def SetStartFromQ(self, q_arm):
        return self._SetClipped(self.q, np.asarray(q_arm, dtype=float)[:self.num_arm_dof])

    def SetNativeStart(self, q_init, rng):
        """Joint-space IK restarts from uniformly random configurations, which is what
        `q_init` is, so native and paired coincide for this arm by construction."""
        return self.SetStartFromQ(q_init)

    def BoundingBoxConstraint(self):
        ## Read off the PLANT rather than from a hardcoded table, because the screw joint's
        ## limits arrive by repair rather than from the parser and a second copy of them
        ## here would be a place for the two to disagree.
        lower, upper = RequireFiniteLimits(self.plant, f"{self.spec.name} plant")
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            lower[:self.num_arm_dof], upper[:self.num_arm_dof], self.q)
        self.bounding_box_constraint.evaluator().set_description("QBoundingBoxConstraint")


class HelixArmIKProgramNumerical(_NumericalMixin, HelixArmIKProgram):
    """Joint-space formulation of the pose task."""

    def create_prog(self, target_pose=np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal=None):
        self._CreateVariables(q_nominal)
        self.target_pose = target_pose
        self.add_constraints()
        self.add_costs()


class HelixArmMugProgramNumerical(_NumericalMixin, HelixArmMugProgram):
    """Joint-space formulation of the grasp task."""

    def create_prog(self, target_mug=Mug(), q_nominal=None):
        self._CreateVariables(q_nominal)
        self.target_mug = target_mug
        self.target_pose = np.array([*target_mug.middle.translation(), 0, 0, 0])
        self.add_constraints()
        self.add_costs()
