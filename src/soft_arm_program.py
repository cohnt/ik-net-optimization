"""The soft continuum arm's four programs: learned and joint-space, pose and grasp.

Modelled line for line on `src/iiwa_program.py`, which is the two-arm template -- this
robot has no closed-form IK, so like the iiwa it has no analytic column and the comparison
is learned-vs-joint-space.

WHAT IS DIFFERENT, AND IT IS ONE THING.  On both rigid arms the configuration IS the
plant's position vector: `num_pos == num_arm_dof == 7`, so `VarsToQ` returns joint angles
and the joint-limit row bounds them directly. Here the configuration is 12 normalized
strains and the plant carries 231 floating-body positions whose limits are +-inf. The
separation lives in `VarsToConfigAndQ`, `ConfigLimits` and `ConfigToPlantQ`, all of which
`IKFlowProgram` provides with defaults that make the rigid arms' behaviour literally
unchanged.

WHAT IS NOT DIFFERENT.  The formulation. The learned arm is the draft's eq. (6) -- a free
conditioning pose `c`, a latent `z`, a correction `q_c`, and the task imposed as constraint
rows through `FK(q)`. Nothing here adds a cost, weakens a bound, or gives this robot a term
the others do not have. The joint-space arm decides over the configuration directly and
shares every constraint and cost through the base class, which is what makes the comparison
a statement about the change of variables alone.

TWO JACOBIANS, EACH IN ITS OWN REGIME.  The chain rule is

    d(plant q)/d(vars) = dP/dcfg  @  dflow/dvars

with the flow's Jacobian left exactly as the rigid arms compute it -- one `jacrev`, 12
outputs against 30 inputs, which is reverse mode's regime and the measured verdict this
project already closed -- and the network-free map differentiated in FORWARD mode, 12
inputs against 231 outputs. Composing the map inside `MakeFlowInference` instead would have
made one reverse pass of 231 outputs against 30 inputs and silently voided that verdict.
Measured on this map: jacfwd compiled is 0.158 ms against jacrev's 6.3 ms eager.
"""

import numpy as np
import torch

from ikflow.config import DEVICE
from pydrake.all import (AutoDiffXd, MathematicalProgram, Quaternion, RigidTransform,
                         RollPitchYaw)

import src.soft_arm.register  # noqa: F401  -- registers the rungs with jrl.robots
from src.flow_loading import LoadFlowSolver
from src.generic_program import IKFlowConstraints, IKFlowProgram, ProgramOptions
from src.soft_arm import kinematics as K
from src.soft_arm.params import GetSpec
from src.utils import Mug, RepoDir

_CPU = "cpu"


class SoftArmIKProgram(IKFlowProgram):
    """Learned formulation, pose task: reach a 6-D pose with the tip frame."""

    def __init__(self, diagram, options=ProgramOptions(), rung="soft12", model=None,
                 checkpoint=None, fk="analytic", surrogate=None):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context)

        self.spec = GetSpec(rung)
        self.frame = self.plant.GetFrameByName(self.spec.tip_frame_name)
        self.autodiff_frame = self.autodiff_plant.GetFrameByName(self.spec.tip_frame_name)
        self.num_pos = self.plant.num_positions()
        self.num_arm_dof = self.spec.ndof
        self.num_task_vars = 6

        if self.num_pos != self.spec.num_plant_positions:
            raise RuntimeError(
                f"the scene's plant has {self.num_pos} positions but {self.spec.name} "
                f"declares {self.spec.num_plant_positions}; the model and the spec have "
                f"drifted -- rerun src/soft_arm/generate_sdf.py")

        if model is not None:
            self.ik_solver = model
        elif checkpoint is not None:
            self.ik_solver = LoadFlowSolver(self.spec.name, checkpoint)
        else:
            raise ValueError(
                f"{self.spec.name} has no upstream pretrained chart to fall back on: pass "
                f"a `checkpoint` or a `model`. (The Panda's default is Jeremy's published "
                f"weights; nobody has published a chart for this robot.)")

        self._plant_slots = self._BuildPlantSlotMap()
        ## The FORWARD MODEL. `analytic` is the exact constant-strain exponential;
        ## `learned` swaps in a surrogate over the same backbone frames, which replaces the
        ## forward model for the IK constraint AND the collision geometry at once, because
        ## both read the same map. A constructor argument rather than a ProgramOptions
        ## field: all arms share one options object, and an option naming something only
        ## one robot has is what scored two cluster columns zero.
        self.fk_mode = fk
        self.fk_surrogate = None
        if fk == "learned":
            from src.soft_arm import fk_surrogate as FS
            if surrogate is None:
                path = FS.SurrogatePath(self.spec, RepoDir())
                surrogate, self.fk_surrogate_metrics = FS.Load(self.spec, path)
            else:
                ## Shared across every program in a grid: loading per cell would be 480
                ## loads of the same weights, and -- more to the point -- both arms must
                ## carry the SAME forward model or the control is not a control.
                self.fk_surrogate_metrics = getattr(surrogate, "screen_metrics", {})
            self.fk_surrogate = surrogate
            ## CalibrateFlowFrame cannot pass 1e-6 here. The flow is trained against the
            ## TRUE kinematics, so the scene-to-flow offset it measures is the surrogate's
            ## own error and is not constant. The tolerance becomes a stated number derived
            ## from the surrogate's measured tail, and the spread is recorded -- the check
            ## is loosened on the record, not switched off.
            tail_mm = float(self.fk_surrogate_metrics.get("tip_mm/max", 10.0))
            self.flow_frame_tol = max(1e-6, 10.0 * tail_mm / 1000.0)
        elif fk != "analytic":
            raise ValueError(f"unknown fk backend {fk!r}; expected 'analytic' or 'learned'")
        self.options = options
        self.ConfigureNetworkDtype()
        self.constraints = []
        ## Forward mode, and compiled when the flow's Jacobian is -- one switch, so a
        ## comparison cannot accidentally run one arm compiled and the other not.
        compile_it = bool(getattr(options, "compile_flow_jacobian", False))
        if self.fk_surrogate is not None:
            from src.soft_arm import fk_surrogate as FS
            forward = FS.MakeLearnedConfigToPlantQ(self.fk_surrogate)
            jacobian = torch.func.jacfwd(forward)

            def config_jacobian(cfg):
                return jacobian(cfg), forward(cfg)

            self.config_jacobian = config_jacobian
        else:
            self.config_jacobian = K.ConfigJacobianGen(
                self.spec, compile_it=compile_it, device=_CPU)
        self.CalibrateFlowFrame()

    ## --------------------------- configuration vs plant --------------------------- ##

    def ConfigLimits(self):
        """`[-1, 1]` per coordinate: the configuration is NORMALIZED strain."""
        ones = np.ones(self.num_arm_dof)
        return (-ones, ones)

    def _BuildPlantSlotMap(self):
        """Where each body's 7 positions live in THIS plant's position vector.

        Delegates to `kinematics.PlantSlotMap`, which the acceptance probe uses too, so the
        program and the probe cannot disagree about the layout.
        """
        return K.PlantSlotMap(self.plant, self.spec)

    def ConfigToPlantQ(self, cfg):
        """The forward model, in float: exact, or the surrogate under `--fk learned`."""
        tensor = torch.as_tensor(np.asarray(cfg, dtype=float), dtype=torch.float64,
                                 device=_CPU)
        if self.fk_surrogate is not None:
            canonical = self.fk_surrogate(tensor).detach().cpu().numpy()
        else:
            canonical = K.config_to_plant_q(tensor, self.spec, device=_CPU).cpu().numpy()
        return canonical[self._plant_slots]

    def ExactConfigToPlantQ(self, cfg):
        """The EXACT forward model, whatever this program is optimising against."""
        tensor = torch.as_tensor(np.asarray(cfg, dtype=float), dtype=torch.float64,
                                 device=_CPU)
        return K.config_to_plant_q(tensor, self.spec, device=_CPU).cpu().numpy()[
            self._plant_slots]

    def VerificationQ(self, q):
        """Where `verify()` grades: always the exact kinematics.

        Under `--fk analytic` this is the identity, and `np.array_equal` downstream sees
        that and records nothing extra. Under `--fk learned` it re-derives the plant
        positions from the configuration the solve returned, so the task gate, the pose
        residual, the collision check and the true minimum distance are all measured on the
        robot rather than on the model of it. Without this the learned-FK column would be
        graded by the thing it is measuring.
        """
        if self.fk_surrogate is None:
            return q
        return self.ExactConfigToPlantQ(self._last_returned_cfg)

    ## `PadQ` keeps its meaning -- configuration to plant positions -- so anything that
    ## still calls it by that name gets the map rather than a zero-padded strain vector.
    def PadQ(self, cfg):
        return self.ConfigToPlantQ(cfg)

    def SampleConfiguration(self, rng):
        return rng.uniform(-1.0, 1.0, size=self.num_arm_dof)

    def _LiftingQ(self):
        if bool(getattr(self.options, "lift_q", False)):
            raise NotImplementedError(
                "lift_q assumes the configuration IS the plant position vector, which on "
                "this robot would write a 12-strain guess into 231 floating-body "
                "positions. It is a closed, refuted Stage F diagnostic; rather than "
                "reimplement it here, this refuses loudly.")
        return False

    ## ------------------------------- the program ---------------------------------- ##

    def create_prog(self, target_pose=np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal=None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6)
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width)
        self.correction = self.prog.NewContinuousVariables(self.num_arm_dof)
        self.lumped_vars = np.hstack([self.c, self.z, self.correction])

        self.target_pose = target_pose
        ## In CONFIGURATION space: the quadratic's zero is the straight, unstretched rod,
        ## which is this robot's elastic-energy analogue of joint centering.
        self.q_nominal = (np.zeros(self.num_arm_dof) if q_nominal is None
                          else np.asarray(q_nominal, dtype=float))

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

    ## ------------------------------- the two maps ---------------------------------- ##

    def ik_inference(self, vars, add_correction=True):
        """The flow's forward pass: latent and conditioning pose -> a CONFIGURATION."""
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=self.torch_dtype)
        if not add_correction:
            width = self.ik_solver.network_width
            vars = torch.cat([vars[:7 + width],
                              torch.zeros(self.num_arm_dof, dtype=vars.dtype, device=DEVICE)])
        cfg, _ = self.FlowInference()(vars)
        return cfg

    def TaskVarsToPose7(self, task_vars, t):
        """Task variables -> the `(xyz, wxyz)` the flow is conditioned on.

        The task variables ARE the conditioning pose, written as xyz + rpy, exactly as on
        both rigid arms.
        """
        from pydrake.all import RollPitchYaw_, RotationMatrix_
        xyz = task_vars[:3]
        quaternion = RotationMatrix_[t](RollPitchYaw_[t](task_vars[3:6])).ToQuaternion().wxyz()
        return xyz, quaternion

    def _FlowVars(self, rpy_vars, autodiff):
        """`[xyz | wxyz | z | correction]`, the layout `MakeFlowInference` consumes."""
        width = self.ik_solver.network_width
        n = self.num_task_vars
        vars = np.zeros(7 + width + self.num_arm_dof,
                        dtype=AutoDiffXd if autodiff else np.float64)
        xyz, quaternion = self.TaskVarsToPose7(
            rpy_vars[:n], AutoDiffXd if autodiff else float)
        vars[:3] = xyz
        vars[3:7] = quaternion
        vars[7:7 + width] = rpy_vars[n:n + width]
        vars[7 + width:] = rpy_vars[n + width:n + width + self.num_arm_dof]
        return vars

    def VarsToConfigAndQ(self, rpy_vars, add_correction=True):
        autodiff = isinstance(rpy_vars[0], AutoDiffXd)
        vars = self._FlowVars(rpy_vars, autodiff)

        if not autodiff:
            cfg = self.ik_inference(vars, add_correction=add_correction)
            cfg = cfg.detach().cpu().numpy()
            self._last_returned_cfg = cfg
            return cfg, self.ConfigToPlantQ(cfg)

        values = np.array([v.value() for v in vars])
        gradients = np.array([v.derivatives() for v in vars])
        tensor = torch.tensor(values, dtype=self.torch_dtype, device=DEVICE)
        flow_jacobian, cfg_tensor = self.jacobian_gen(tensor)
        flow_jacobian = flow_jacobian.detach().cpu().numpy()
        cfg = cfg_tensor.detach().cpu().numpy()

        ## d(cfg)/d(vars) first -- (ndof x nvars) -- then the map's (nq x ndof) on top of
        ## it. The other association would build an (nq x nvars) intermediate for nothing.
        cfg_gradients = flow_jacobian @ gradients

        cfg_cpu = torch.tensor(cfg, dtype=torch.float64, device=_CPU)
        map_jacobian, plant_q = self.config_jacobian(cfg_cpu)
        ## Permute into the plant's own slot order, values and rows together.
        map_jacobian = map_jacobian.detach().cpu().numpy()[self._plant_slots]
        plant_q = plant_q.detach().cpu().numpy()[self._plant_slots]
        q_gradients = map_jacobian @ cfg_gradients

        self._last_returned_cfg = cfg
        cfg_ad = np.array([AutoDiffXd(cfg[i], cfg_gradients[i]) for i in range(len(cfg))])
        q_ad = np.array([AutoDiffXd(plant_q[i], q_gradients[i]) for i in range(len(plant_q))])
        return cfg_ad, q_ad

    def VarsToQ(self, rpy_vars, add_correction=True):
        return self.VarsToConfigAndQ(rpy_vars, add_correction=add_correction)[1]


class SoftArmMugProgram(SoftArmIKProgram):
    """Learned formulation, grasp task: the gripper on the mug's axis, orientation free."""

    def __init__(self, diagram, options=ProgramOptions(), rung="soft12", model=None,
                 checkpoint=None, fk="analytic", surrogate=None):
        ## `fk`/`surrogate` are FORWARDED, not defaulted here. The benchmark driver passes
        ## `fk=args.fk` to whichever class the task selects, so an override that dropped
        ## them made the grasp task a TypeError at construction -- every mug item of stage
        ## SOFT12 died in 58 s while the pose items ran, i.e. half a campaign lost to a
        ## signature that only the other half exercised.
        super().__init__(diagram, options, rung, model, checkpoint, fk=fk,
                         surrogate=surrogate)
        ## The flow conditions on the arm's tip; the grasp acts between the fingers.
        self.ee_frame = self.frame
        self.frame = self.plant.GetFrameByName("between_fingers")
        self.autodiff_frame = self.autodiff_plant.GetFrameByName("between_fingers")

        self.CalibrateFlowFrame()
        self.plant.SetPositions(self.plant_context,
                                self.ConfigToPlantQ(np.zeros(self.num_arm_dof)))
        X_W_flow = self.FlowPoseInWorld()
        X_W_grasp = self.frame.CalcPoseInWorld(self.plant_context)
        self.X_grasp_ee = X_W_grasp.inverse() @ X_W_flow

    def create_prog(self, target_mug=Mug(), q_nominal=None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6)
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width)
        self.correction = self.prog.NewContinuousVariables(self.num_arm_dof)
        self.lumped_vars = np.hstack([self.c, self.z, self.correction])

        self.target_mug = target_mug
        self.q_nominal = (np.zeros(self.num_arm_dof) if q_nominal is None
                          else np.asarray(q_nominal, dtype=float))

        ## Seed `c` at the conditioning pose a grasp of this mug would have, not at the
        ## mug itself: the two differ by X_grasp_ee, and seeding at the mug is what put
        ## the Panda's `c` 120 degrees from where the network expects it.
        X_W_ee = target_mug.middle @ self.X_grasp_ee
        self.prog.SetInitialGuess(self.c, np.concatenate(
            [X_W_ee.translation(), X_W_ee.rotation().ToRollPitchYaw().vector()]))
        self.prog.SetInitialGuess(self.z, np.zeros(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))

        self.target_pose = np.array([*target_mug.middle.translation(), 1, 0, 0, 0])
        self.jacobian_gen = self.MakeJacobianGen()
        self.add_constraints()
        self.add_costs()

    def CreateIKConstraint(self):
        # Identical to the rigid arms': on the mug's central axis exactly (x = y = 0, an
        # equality, because that is what the task says) and within the mug's height along
        # it. Orientation free -- any approach direction is a valid grasp.
        lb = np.array([0.0, 0.0, -self.options.mug_height])
        ub = np.array([0.0, 0.0, self.options.mug_height])

        def eval_func(vars, q, pose):
            position, _ = pose
            mug_transform = np.linalg.inv(self.target_mug.middle.GetAsMatrix4())
            # Drop the homogeneous row: identically 1 with a zero gradient, so keeping it
            # as an equality row costs a rank and breaks LICQ.
            return (mug_transform @ np.array([[*position, 1]]).T).squeeze()[:3]

        self.ik_constraint = IKFlowConstraints(lb, ub, eval_func, description="IKConstraint")
        self.constraints.append(self.ik_constraint)
        return self.ik_constraint

    def BoundingBoxConstraint(self):
        self.LatentBoxConstraint()
        centre = self.target_mug.middle.translation()
        slack = self.options.c_position_slack
        ## A general linear constraint, deliberately NOT a bounding box: IPOPT's bound_push
        ## projects the initial guess into every variable box before evaluating anything,
        ## which would silently destroy the exact paired start.
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
    """The joint-space arm: the decision variables ARE the configuration.

    `VarsToConfigAndQ` is the kinematic map and nothing else -- no network is evaluated --
    and every constraint and cost is inherited, which is what makes the comparison a
    statement about the change of variables alone.

    Note what this arm is NOT on this robot: an identity map. On the rigid arms the
    joint-space arm's `VarsToQ` is free, which is why their runtime table shows a flat,
    cheap baseline. Here it still has to place 231 floating-body positions, so the
    baseline carries real per-iteration cost too. That must be said wherever this robot's
    runtimes appear: the learned arm's premium will look smaller than on the rigid arms
    because the BASELINE got more expensive, not because the learned arm got cheaper.
    """

    def _CreateVariables(self):
        self.q = self.prog.NewContinuousVariables(self.num_arm_dof)
        self.lumped_vars = self.q

    def VarsToConfigAndQ(self, rpy_vars, add_correction=False):
        cfg = np.asarray(rpy_vars[:self.num_arm_dof])
        if not isinstance(rpy_vars[0], AutoDiffXd):
            self._last_returned_cfg = np.asarray(cfg, dtype=float)
        else:
            self._last_returned_cfg = np.array([v.value() for v in cfg])
        if isinstance(rpy_vars[0], AutoDiffXd):
            values = np.array([v.value() for v in cfg])
            gradients = np.array([v.derivatives() for v in cfg])
            tensor = torch.tensor(values, dtype=torch.float64, device=_CPU)
            map_jacobian, plant_q = self.config_jacobian(tensor)
            map_jacobian = map_jacobian.detach().cpu().numpy()[self._plant_slots]
            plant_q = plant_q.detach().cpu().numpy()[self._plant_slots]
            q_gradients = map_jacobian @ gradients
            q_ad = np.array([AutoDiffXd(plant_q[i], q_gradients[i])
                             for i in range(len(plant_q))])
            return cfg, q_ad
        return cfg, self.ConfigToPlantQ(cfg)

    def VarsToQ(self, rpy_vars, add_correction=False):
        return self.VarsToConfigAndQ(rpy_vars)[1]

    def SetStartFromQ(self, q_arm):
        return self._SetClipped(self.q, np.asarray(q_arm, dtype=float)[:self.num_arm_dof])

    def SetNativeStart(self, q_init, rng):
        return self.SetStartFromQ(q_init)

    def BoundingBoxConstraint(self):
        lower, upper = self.ConfigLimits()
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(lower, upper, self.q)
        self.bounding_box_constraint.evaluator().set_description("QBoundingBoxConstraint")


class SoftArmIKProgramNumerical(_NumericalMixin, SoftArmIKProgram):
    def create_prog(self, target_pose=np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal=None):
        self.prog = MathematicalProgram()
        self._CreateVariables()
        self.target_pose = target_pose
        self.q_nominal = (np.zeros(self.num_arm_dof) if q_nominal is None
                          else np.asarray(q_nominal, dtype=float))
        self.prog.SetInitialGuess(self.q, self.q_nominal)
        self.add_constraints()
        self.add_costs()


class SoftArmMugProgramNumerical(_NumericalMixin, SoftArmMugProgram):
    def create_prog(self, target_mug=Mug(), q_nominal=None):
        self.prog = MathematicalProgram()
        self._CreateVariables()
        self.target_mug = target_mug
        self.target_pose = np.array([*target_mug.middle.translation(), 1, 0, 0, 0])
        self.q_nominal = (np.zeros(self.num_arm_dof) if q_nominal is None
                          else np.asarray(q_nominal, dtype=float))
        self.prog.SetInitialGuess(self.q, self.q_nominal)
        self.add_constraints()
        self.add_costs()
