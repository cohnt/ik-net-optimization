import os

import pydrake.math
import torch
from ikflow.config import DEVICE
import numpy as np
from dataclasses import dataclass, field
from functools import partial
import numpy as np
from pydrake.all import (
    AutoDiffXd,
    IpoptSolver,
    SnoptSolver,
    SolverOptions,
    CommonSolverOption,
    MinimumDistanceLowerBoundConstraint,
    Quaternion,
    RigidTransform,
    RigidTransform_,
    RotationMatrix,
    RotationMatrix_,
    RollPitchYaw,
    RollPitchYaw_,
    Quaternion_,
)

@dataclass
class ProgramOptions:
    joint_centering_cost: float = field(default=0.0, metadata={"help": "Weight for joint centering cost"})
    collision_avoidance: bool = field(default=True, metadata={"help": "Add collision avoidance constraints"})
    joint_limits: bool = field(default=True, metadata={"help": "Enforce joint limits"})
    ik_constraint_tol: tuple = field(default=(1e-4, 0.01), metadata={"help": "Tolerance for IK constraints: tuple of (position tol, orientation tol in radians). The orientation entry is used only by orientation_error_form='rpy_boxed'; 'rpy' pins the residual to zero"})
    orientation_error_form: str = field(default="rpy", metadata={"help": "'rpy' pins the roll-pitch-yaw residual to zero, as ../codebase's pose constraint does; 'rpy_boxed' allows +-ori_tol on each row"})
    correction_cost_weight: float = field(default=0.0, metadata={"help": "Weight for correction cost to keep close to zero"})

    mug_height: float = field(default=0.035, metadata={"help": "Mug height for valid grasp poses"})

    ## Network evaluation ##
    # The flow is a float32 artifact, but evaluating it in float64 costs ~15% and
    # makes the map smooth well below 1e-4. In float32 the finite-difference error
    # against the analytic Jacobian blows up from 6e-3 (h=1e-4) to 1.8 (h=1e-6),
    # which is what makes SNOPT's derivative check fail and starves the line search.
    use_float64: bool = field(default=True, metadata={"help": "Evaluate the flow in float64 so the map is smooth at solver step sizes"})

    c_position_slack: float = field(default=0.25, metadata={"help": "Half-width of the box on the conditioning position, about the target"})


    ## Evaluation sharing ##
    # Every Drake binding evaluates its own callback, so the joint-centering cost used to
    # run a *second* forward pass and a second jacrev through the flow at the same point
    # as the constraint binding. An IPOPT log of the mug problem records 1276 objective
    # evaluations against 1276 constraint evaluations and 455 objective gradients against
    # 490 constraint Jacobians -- i.e. about half the network work was redundant.
    # On by default: the memoised path returns bit-identical values *and* derivatives, so
    # this is a pure throughput win (measured 12/30 -> 22/30 on the Panda grasp once the
    # frame was also fixed). Turn it off only to reproduce a pre-overhaul measurement.
    share_flow_evaluations: bool = field(default=True, metadata={"help": "Memoise VarsToQ/fk so the cost and constraint bindings share one flow evaluation per point"})

    ## Conditioning of the learned program ##
    # The flow is conditioned on the pose of the frame it was *trained* on. In the finray
    # grasp scene the body called "panda_hand" is a different frame, 27 mm and 120 degrees
    # away, so looking it up by name conditions the network on a pose it never saw. Off by
    # default only so the ladder can measure the repair separately from the redesigns.
    calibrate_flow_frame: bool = field(default=True, metadata={"help": "Express the conditioning pose in the frame the flow was trained on rather than in whichever scene body shares its name"})
    # The latent prior is N(0, I), so |z| concentrates near sqrt(latent_dim); a +-5 box
    # per component lets the optimiser walk to |z| ~ 13, deep into the tail where the flow
    # has seen no training mass and its output stops meaning anything. A norm bound is the
    # learned analogue of the analytic framework's reachability constraint, which is the
    # gap the draft itself names ("outside the reachable set IKFlow's gradients explode").
    latent_trust_region: float = field(default=None, metadata={"help": "Bound on ||z||; None keeps the per-component box only"})
    latent_cost_weight: float = field(default=0.0, metadata={"help": "Weight on ||z||^2, keeping the latent in the flow's typical set"})
    correction_bound: float = field(default=0.1, metadata={"help": "Half-width of the box on the joint-space correction"})
    c_parameterization: str = field(default="free", metadata={"help": "'free': c is a free 6-vector constrained through FK. 'task': c is computed from task parameters so the conditioning pose satisfies the task by construction"})

    ## Solver behaviour ##
    ipopt_mu_strategy: str = field(default=None, metadata={"help": "IPOPT 'mu_strategy'; 'adaptive' often helps on badly scaled problems"})
    max_iter: int = field(default=None, metadata={"help": "Iteration cap (IPOPT max_iter / SNOPT Major iterations limit)"})

    ## Starting point ##
    # A benchmark that starts each formulation somewhere different cannot attribute a
    # success-rate gap to the formulation. `SetStartFromQ` puts every arm at the same
    # configuration; this switch only exists so the old protocol stays reproducible.
    seed_from_q_init: bool = field(default=False, metadata={"help": "Start from a shared q_init instead of the per-formulation default"})

    ## Solver options ##
    which_solver: str = field(default="ipopt", metadata={"help": "Which IKFlow solver to use"})
    acceptable_tol: float = field(default=1e-4, metadata={"help": "Acceptable tolerance for solver convergence"})
    acceptable_dual_inf_tol: float = field(default=1e-4, metadata={"help": "Acceptable dual infeasibility tolerance for solver convergence"})
    acceptable_compl_inf_tol: float = field(default=1e-4, metadata={"help": "Acceptable complementary infeasibility tolerance for solver convergence"})
    acceptable_constr_viol_tol: float = field(default=1e-6, metadata={"help": "Acceptable constraint violation tolerance for solver convergence"})
    acceptable_iter: int = field(default=1, metadata={"help": "Acceptable number of iterations for solver convergence"})
    file_print_level: int = field(default=5, metadata={"help": "File print level for the solver"})
    file_print_name: str = field(default="ikflow_solver_log.txt", metadata={"help": "File name for solver log"})
    max_wall_time: float = field(default=60, metadata={"help": "Maximum wall time for the solver in seconds"})
    snopt_function_precision: float = field(default=None, metadata={"help": "SNOPT 'Function precision'. Leave None in float64; set ~1e-6 if evaluating the flow in float32"})

    vars_file: str = field(default=None, metadata={"help": "If provided, saves variable trajectories to this file"})
    visualize: bool = field(default=False, metadata={"help": "If true, visualizes the IK solving process in Meshcat"})



# ----------------------------- orientation error ------------------------------
#
# The orientation half of the IK pose constraint: three signed rows of the roll-pitch-yaw
# residual, the same shape as the position rows and the same quantity ../codebase's
# EEPoseConstraint imposes. ProgramOptions.orientation_error_form picks the bounds:
#
#   rpy          residual == 0        pinned, as ../codebase pins it  [default]
#   rpy_boxed    |residual| <= ori_tol
#
# Earlier revisions measured the mismatch as a single scalar angle, 2*arccos(|q.q_t|).
# That shape is what put a branch point at zero error -- taking a norm of a
# three-component error is exactly the operation that manufactures one -- and its eps
# clamp additionally handed back an AutoDiffXd with an empty derivative vector at
# convergence. Three signed rows have neither problem: the residual is smooth at the
# solution and its Jacobian is full rank there. See git history (0be5342) for the scalar
# forms and the measurements that retired them.
#
# Only the pose-target programs reach this (PandaIKProgram, Iiwa14IKProgram and their
# ...Numerical variants); the mug programs override CreateIKConstraint, and the analytic
# programs never add an IK constraint at all.


def orientation_error_rpy(orientation, target_rpy):
    '''Signed roll-pitch-yaw residual, rpy(R) - rpy(R_target), wrapped to (-pi, pi].

    The chart degenerates at pitch = +-pi/2 (gimbal lock) and each component wraps at
    +-pi. Wrapping the residual, rather than differencing raw angles, removes the 2*pi
    jump for targets sitting near a branch; it shifts by a constant, so derivatives are
    untouched. Gimbal lock is a property of the *target pose*, not of the error, so it
    does not degrade behaviour at convergence.

    `orientation` is the achieved orientation as a wxyz quaternion, plain or AutoDiffXd;
    `target_rpy` is the target's roll-pitch-yaw as plain floats, precomputed by the caller.
    '''
    ad = isinstance(orientation[0], AutoDiffXd)
    dtype = AutoDiffXd if ad else float
    achieved = RollPitchYaw_[dtype](RotationMatrix_[dtype](Quaternion_[dtype](orientation)))
    residual = achieved.vector() - np.asarray(target_rpy)
    # Wrap each component into (-pi, pi] by subtracting a constant multiple of 2*pi.
    for i in range(3):
        value = residual[i].value() if ad else float(residual[i])
        turns = np.round(value / (2.0 * np.pi))
        if turns != 0.0:
            residual[i] = residual[i] - 2.0 * np.pi * turns
    return residual


ORIENTATION_ERROR_FORMS = ("rpy", "rpy_boxed")

class IKFlowConstraints:
    def __init__(self, lb, ub, eval_func, description=""):
        self.lb = lb
        self.ub = ub
        self.eval_func = eval_func
        self.description = description
    def __len__(self):
        return len(self.lb)

class GraspTaskParamMixin:
    '''The mug grasp with the task folded into the decision variables.

    The baseline program leaves `c` a free 6-vector inside a 0.5 m box around the mug and
    asks a nonlinear constraint on `FK(q)` to put the gripper on the mug's axis. Here the
    decision variables are the grasp pose *in the mug frame*, `X_MG = (x, y, z, r, p, y)`,
    and the conditioning pose is computed from them:

        c  =  X_WM . X_MG . X_GE

    so every `c` the optimiser can name is already a valid grasp. The task constraint on
    `c` is then a plain bounding box -- `x = y = 0`, `z` within the mug height, orientation
    free -- which an NLP solver handles exactly and for free, instead of two nonlinear
    equality rows. This is the shape `eaik-experiment` uses for the same grasp
    (`AddBoundingBoxConstraint([0, 0, -h2, -pi, -pi, -pi], [0, 0, h2, pi, pi, pi], p)`),
    and it is the project's own thesis written literally: optimise in end-effector space,
    treat the flow as an approximate chart, and leave only the chart's error to the
    correction.

    What remains nonlinear is exactly that chart error: `FK(IKFlow(c, z) + q_c)` is not
    `c`, so the mug-axis constraint on `FK(q)` stays. The difference is that it now starts
    small and stays small, rather than being the thing that drags `c` around.

    Note `X_GE`, not `X_EG`: `X_MG` is the pose of the *grasp* frame, and converting it to
    the frame the flow was conditioned on is a conjugation that does not cancel. Getting
    this backwards is a bug that survived a long time in the sibling project.
    '''

    def TaskVarsToPose7(self, task_vars, t):
        X_MG = RigidTransform_[t](RollPitchYaw_[t](task_vars[3:6]), task_vars[:3])
        X_W_ee = self._Templated(self.target_mug.middle, t) @ X_MG @ self._Templated(self.X_grasp_ee, t)
        return X_W_ee.translation(), X_W_ee.rotation().ToQuaternion().wxyz()

    @staticmethod
    def _Templated(transform, t):
        if t is float:
            return transform
        return transform.cast[AutoDiffXd]()

    def _GraspParamsFromQ(self, q_arm):
        '''`X_MG` of the configuration `q_arm`, i.e. the inverse of `TaskVarsToPose7`.'''
        self.plant.SetPositions(self.plant_context, self.PadQ(np.asarray(q_arm, dtype=float)[:self.num_arm_dof]))
        X_W_grasp = self.frame.CalcPoseInWorld(self.plant_context)
        X_MG = self.target_mug.middle.inverse() @ X_W_grasp
        return np.concatenate([X_MG.translation(), X_MG.rotation().ToRollPitchYaw().vector()])

    def create_prog(self, *args, **kwargs):
        super().create_prog(*args, **kwargs)
        # The inherited guess is a conditioning pose; here the variables are grasp
        # parameters, and the centre of the mug's axis is the natural default. Callers
        # that want a specific start use SetStartFromQ.
        self.prog.SetInitialGuess(self.c, np.zeros(6))

    def SetStartFromQ(self, q_arm):
        # Same ordering point as the free-c version: the latent is taken at the
        # configuration's own conditioning pose, and only the grasp parameters are
        # projected onto the manifold the box describes.
        q_arm = np.asarray(q_arm, dtype=float)[:self.num_arm_dof]
        self.plant.SetPositions(self.plant_context, self.PadQ(q_arm))
        X_W_ee = self.FlowFrame().CalcPoseInWorld(self.plant_context)
        c = np.concatenate([X_W_ee.translation(), X_W_ee.rotation().ToRollPitchYaw().vector()])
        z = self.InvertFlow(q_arm, c)
        clipped = self._SetClipped(self.c, self._GraspParamsFromQ(q_arm))
        clipped += self._SetClipped(self.z, z)
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        return clipped

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -5. * np.ones(self.ik_solver.network_width),
            5. * np.ones(self.ik_solver.network_width), self.z)
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        # The grasp itself: exactly on the mug's axis, within its height, orientation
        # free. x and y are pinned because that is what the task is -- the gripper origin
        # lies *on* the axis -- not because a tolerance happened to be chosen.
        height = self.MugHeight()
        self.c_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            np.array([0.0, 0.0, -height, -2 * np.pi, -2 * np.pi, -2 * np.pi]),
            np.array([0.0, 0.0, height, 2 * np.pi, 2 * np.pi, 2 * np.pi]),
            self.c)
        self.c_bounding_box_constraint.evaluator().set_description("GraspParamBoundingBox")
        bound = self.options.correction_bound
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -bound * np.ones(self.num_arm_dof), bound * np.ones(self.num_arm_dof), self.correction)
        self.correction_bounding_box_constraint.evaluator().set_description("CorrectionBoundingBoxConstraint")

    def MugHeight(self):
        '''Half-height of the graspable band along the mug axis.

        `ProgramOptions.mug_height` is the single source of truth. `Mug.height` exists too
        and defaults to 0.04 against the option's 0.035, which is a latent disagreement --
        the grasp constraint and the bound on the grasp parameter have to be the same
        number or the box and the constraint describe different problems.
        '''
        return self.options.mug_height



class IKFlowProgram:
    def __init__(self, diagram, frame, solver, options=ProgramOptions()):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context)

        self.frame = frame
        self.autodiff_frame = self.autodiff_plant.GetBodyByName(frame.name()).body_frame()

        self.ik_solver = solver
        self.ik_solver.nn_model.eval()
        self.options = options

        self.constraints = []

    def add_constraints(self):
        self.CreateIKConstraint()
        if self.options.collision_avoidance:
            self.CreateCollisionFreeConstraint()
        if self.options.joint_limits:
            self.CreateJointLimitsConstraint()
        self.ApplyConstraints()
        self.BoundingBoxConstraint()
        # Only the learned formulations have a latent; the joint-space and analytic arms
        # share this options object so that budgets and tolerances stay identical between
        # them, which means options that name learned-only variables must be guarded.
        if self.options.latent_trust_region is not None and hasattr(self, "z"):
            self.LatentTrustRegion()

    def add_costs(self):
        if self.options.joint_centering_cost > 0.0:
            self.JointCenteringCost()
        if self.options.correction_cost_weight > 0.0:
            self.CorrectionCost()
        if self.options.latent_cost_weight > 0.0 and hasattr(self, "z"):
            self.LatentCost()

    def LatentCost(self):
        '''A quadratic pull towards the centre of the latent prior.

        Excluded from the reported objective by name, the way ../codebase excludes its
        barrier terms, so the learned column of a cost table still measures the same
        objective as the other formulations.'''
        width = self.ik_solver.network_width
        self.latent_cost = self.prog.AddQuadraticCost(
            Q=self.options.latent_cost_weight * np.eye(width),
            b=np.zeros(width), vars=self.z)
        self.latent_cost.evaluator().set_description("LatentRegularizerCost")

    def LatentTrustRegion(self):
        '''`||z||^2 <= r^2`, imposed as an inequality rather than a per-component box.

        The prior mass sits on a shell of radius sqrt(latent_dim), and a per-component box
        of +-5 admits norms far outside it. An inequality is deliberate: its gradient,
        `2 z`, does not vanish where the constraint is active, so it does not reproduce
        the degenerate active set that a norm-residual equality would create.'''
        radius = self.options.latent_trust_region
        width = self.ik_solver.network_width
        self.latent_trust_constraint = self.prog.AddQuadraticConstraint(
            2.0 * np.eye(width), np.zeros(width), -np.inf, radius ** 2, self.z)
        self.latent_trust_constraint.evaluator().set_description("LatentTrustRegion")

    def fk(self, q, matrix = False):
        frame, context = self.SetPositions(q)
        rigid_transform = frame.CalcPoseInWorld(context)
        if matrix:
            return rigid_transform.GetAsMatrix4()
        else:
            return rigid_transform.translation(), rigid_transform.rotation().ToQuaternion().wxyz()

    ## ---------------------- shared flow evaluation ------------------------- ##

    def _FlowCacheKey(self, vars):
        '''Key an iterate by its values *and* its derivative block.

        Drake hands the cost and the constraint the same `vars` at the same point, but an
        AutoDiffXd carries a gradient as well as a value, and the two callbacks can be
        called with different seed matrices. Keying on the value alone would silently
        return a Jacobian computed against the wrong seeds.
        '''
        if isinstance(vars[0], AutoDiffXd):
            values = np.array([v.value() for v in vars])
            derivatives = np.array([v.derivatives() for v in vars])
            return (True, values.tobytes(), derivatives.shape, derivatives.tobytes())
        return (False, np.asarray(vars, dtype=float).tobytes())

    def QAndPose(self, vars):
        '''`(q, pose)` for an iterate, evaluating the flow at most once per point.

        The forward pass and the `jacrev` are the dominant cost of the learned
        formulation -- everything else in a `VarsToQ` evaluation is about 0.01 ms -- so
        sharing them between the constraint binding and the cost binding is close to a
        factor of two on the whole solve.
        '''
        if not getattr(self.options, "share_flow_evaluations", False):
            q = self.VarsToQ(vars)
            return q, self.fk(q)
        cache = getattr(self, "_flow_cache", None)
        if cache is None:
            cache = self._flow_cache = {}
        key = self._FlowCacheKey(vars)
        hit = cache.get(key)
        if hit is None:
            q = self.VarsToQ(vars)
            hit = (q, self.fk(q))
            cache[key] = hit
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
        return hit

    ## ------------------------- shared starting point ----------------------- ##

    def CalibrateFlowFrame(self, samples=4, tol=1e-9):
        '''Measure the offset between the scene's end-effector frame and the frame the
        flow was actually trained on, and cache it.

        This is not a nicety. The flow is conditioned on the pose of a specific frame --
        jrl's `panda_hand`, at the standard Franka offset from `panda_link7`. The mug
        scene welds a finray gripper to `panda_link7` and that model *also* contains a
        body called `panda_hand`, but at translation [0, 0, 0.134] and rpy [90, 0, 45]
        rather than the Franka hand's [0, 0, 0.107] and [0, 0, -45]. Looking the frame up
        by name therefore returns a frame 27 mm and 120 degrees away from the one the
        network means, and every `c` handed to the flow in the grasp experiments was in
        that wrong frame. Measured symptom: inverting a random configuration at the
        scene's frame returns |z| = 67.6, against 2.23 -- essentially sqrt(7), the typical
        norm under the latent prior -- at the correct frame. The network was being asked
        about configurations it considers astronomically unlikely on every iterate.

        Both frames are rigidly welded to the same link, so the offset is a constant; it
        is measured at several configurations and checked rather than assumed.
        '''
        if not self.options.calibrate_flow_frame:
            self.X_ee_flow = RigidTransform()
            return self.X_ee_flow
        lower = self.plant.GetPositionLowerLimits()[:self.num_arm_dof]
        upper = self.plant.GetPositionUpperLimits()[:self.num_arm_dof]
        offsets = []
        for _ in range(samples):
            q_arm = np.random.uniform(lower, upper)
            self.plant.SetPositions(self.plant_context, self.PadQ(q_arm))
            X_scene = self.frame_for_flow.CalcPoseInWorld(self.plant_context)
            pose = self.ik_solver.robot.forward_kinematics(
                torch.tensor(q_arm[None, :], dtype=torch.float64, device=DEVICE))
            pose = pose.detach().cpu().numpy()[0]
            wxyz = pose[3:] / np.linalg.norm(pose[3:])
            X_flow = RigidTransform(Quaternion(wxyz), pose[:3])
            offsets.append(X_scene.inverse() @ X_flow)
        spread = max(np.linalg.norm(offsets[0].translation() - o.translation())
                     + abs((offsets[0].inverse() @ o).rotation().ToAngleAxis().angle())
                     for o in offsets[1:])
        if spread > 1e-6:
            raise RuntimeError(
                f"the flow frame offset is not constant across configurations "
                f"(spread {spread:.3e}); the scene's joint convention does not match the "
                f"one the network was trained with")
        self.X_ee_flow = offsets[0]
        return self.X_ee_flow

    def FlowPoseInWorld(self, context=None):
        '''The pose the flow should be conditioned on, in the world frame.'''
        context = self.plant_context if context is None else context
        X = self.frame_for_flow.CalcPoseInWorld(context)
        offset = getattr(self, "X_ee_flow", None)
        return X if offset is None else X @ offset

    @property
    def frame_for_flow(self):
        return self.FlowFrame()

    def FlowFrame(self):
        '''The frame the flow was conditioned on during training.

        The mug programs move `self.frame` to `between_fingers` because that is where the
        grasp constraint acts, but the network still speaks in terms of the end-effector
        frame it was trained against, so `c` must always be expressed there.
        '''
        return getattr(self, "ee_frame", self.frame)

    def InvertFlow(self, q_arm, c):
        '''The latent that reproduces `q_arm` under conditioning pose `c`.

        IKFlow is a normalizing flow, so this is the network run forwards (`rev=False`)
        and is exact -- not an optimisation. It is what lets the learned formulation start
        from the *same* configuration as the joint-space one, the way ../codebase seeds
        its analytic formulation by recovering `psi` and `GC` from `q_initial`.
        '''
        dtype = self.torch_dtype
        pose7 = self.CToPose7(np.asarray(c, dtype=float))
        c_t = torch.tensor(np.concatenate([pose7, [0.0]])[None, :], dtype=dtype, device=DEVICE)
        x = np.zeros((1, self.ik_solver.network_width))
        x[0, :self.num_arm_dof] = np.asarray(q_arm, dtype=float)[:self.num_arm_dof]
        x_t = torch.tensor(x, dtype=dtype, device=DEVICE)
        with torch.no_grad():
            z, _ = self.ik_solver.nn_model(x_t, c=c_t, rev=False)
        return z.squeeze(0).detach().cpu().numpy().astype(float)

    def SetStartFromQ(self, q_arm):
        '''Start this program at the configuration `q_arm`, in its own variables.

        Returns how far the start had to be clipped to sit inside the variable bounds; a
        start outside the box is not the same start, and the amount matters when reading
        a paired comparison.
        '''
        q_arm = np.asarray(q_arm, dtype=float)[:self.num_arm_dof]
        self.plant.SetPositions(self.plant_context, self.PadQ(q_arm))
        pose = self.FlowPoseInWorld()
        c = np.concatenate([pose.translation(), pose.rotation().ToRollPitchYaw().vector()])
        # Invert at the *unclipped* conditioning pose, then clip. The temptation is to do
        # it the other way round so that q(start) is exactly q_arm, and that is wrong: a
        # random collision-free configuration is not a grasp of this mug, so its
        # conditioning pose sits outside the program's box, and asking the flow for the
        # latent that produces q_arm under a *projected* pose returns |z| ~ 1e7 -- the
        # flow correctly reporting that this configuration is astronomically unlikely
        # there. Measured: |z| goes from 1.6 to 6.2e7 between the two orders. Inverting
        # first keeps the latent inside the typical set and lets the box move only the
        # conditioning pose, which is the quantity the box is actually about.
        z = self.InvertFlow(q_arm, c)
        clipped = self._SetClipped(self.c, c) + self._SetClipped(self.z, z)
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        return clipped

    def _SetClipped(self, variables, values):
        '''Set an initial guess, clipped into the program's bounding box on it.'''
        values = np.asarray(values, dtype=float)
        lower = np.full(len(values), -np.inf)
        upper = np.full(len(values), np.inf)
        names = {v.get_id(): i for i, v in enumerate(variables)}
        for binding in self.prog.bounding_box_constraints():
            evaluator = binding.evaluator()
            for row, var in enumerate(binding.variables()):
                index = names.get(var.get_id())
                if index is not None:
                    lower[index] = max(lower[index], evaluator.lower_bound()[row])
                    upper[index] = min(upper[index], evaluator.upper_bound()[row])
        clipped = np.clip(values, lower, upper)
        self.prog.SetInitialGuess(variables, clipped)
        return float(np.linalg.norm(clipped - values))

    ## ------------------------- multi-start seeding ------------------------- ##

    @property
    def torch_dtype(self):
        return torch.float64 if self.options.use_float64 else torch.float32

    def ConfigureNetworkDtype(self):
        '''Cast the flow to the working dtype. Idempotent, so it is safe to call on a
        solver instance shared between programs.'''
        self.ik_solver.nn_model.to(self.torch_dtype)
        self.ik_solver.nn_model.eval()

    def PadQ(self, q_arm):
        '''Arm joint angles -> a full plant position vector.'''
        q = np.zeros(self.num_pos)
        q[:self.num_arm_dof] = q_arm
        q[self.num_arm_dof:] = 0.04  # fixed gripper joints
        return q

    @staticmethod
    def CToPose7(c):
        '''Conditioning variable (xyz + rpy) -> the xyz + wxyz the flow is conditioned on.'''
        return np.concatenate([c[:3], RotationMatrix(RollPitchYaw(c[3:6])).ToQuaternion().wxyz()])

    ## These are Robot Specific need to be implemented in each file ##
    def ik_inference(self, vars):
        pass
    def VarsToQ(self, vars):
        pass

    def SetPositions(self, q):
        if isinstance(q[0], AutoDiffXd):
            self.autodiff_plant.SetPositions(self.autodiff_context, q)
            return self.autodiff_frame, self.autodiff_context
        else:
            self.plant.SetPositions(self.plant_context, q)
            return self.frame, self.plant_context

    def EvalAllConstraints(self, vars):
        '''Parallelize as much of the VarsToQ as possible to shorten computation time'''
        q, pose = self.QAndPose(vars)  ## one flow evaluation for every constraint row
        total_length = sum(len(constraint) for constraint in self.constraints)
        result = np.full(total_length, q[0]) ## q datatype
        idx = 0
        for constraint in self.constraints:
            l = len(constraint)
            result[idx:idx + l] = constraint.eval_func(vars = vars, q = q, pose = pose)
            idx += l
        return result

    
    def ApplyConstraints(self):
        total_lb = np.hstack([constraint.lb for constraint in self.constraints])
        total_ub = np.hstack([constraint.ub for constraint in self.constraints])
        self.all_constraints = self.prog.AddConstraint(
            func=self.EvalAllConstraints,
            lb=total_lb,
            ub=total_ub,
            vars=self.lumped_vars
        )
        self.all_constraints.evaluator().set_description("AllIKFlowConstraints")


    def CreateIKConstraint(self):
        '''Six rows: the per-axis position error, then the roll-pitch-yaw residual.

        "rpy" pins the orientation residual to zero, as ../codebase's EEPoseConstraint
        does with lb == ub; "rpy_boxed" allows +-ori_tol on each row instead.
        '''
        pos_tol, ori_tol = self.options.ik_constraint_tol
        form = self.options.orientation_error_form
        if form not in ORIENTATION_ERROR_FORMS:
            raise ValueError(f"Unknown orientation_error_form {form!r}; expected one of "
                             f"{sorted(ORIENTATION_ERROR_FORMS)}")
        rpy_tol = 0.0 if form == "rpy" else ori_tol

        # The target's rpy is fixed for the life of the program, so compute it once
        # rather than per constraint evaluation.
        target_rpy = RollPitchYaw(RotationMatrix(Quaternion(self.target_pose[3:]))).vector()
        lb = np.array([-pos_tol] * 3 + [-rpy_tol] * 3)
        ub = np.array([pos_tol] * 3 + [rpy_tol] * 3)

        def eval_func(vars, q, pose):
            position, orientation = pose
            pos_error = position - self.target_pose[:3]
            return np.concatenate([pos_error, orientation_error_rpy(orientation, target_rpy)])

        self.ik_constraint = IKFlowConstraints(lb, ub, eval_func, description="IKConstraint")
        self.constraints.append(self.ik_constraint)
        return self.ik_constraint

    def CreateCollisionFreeConstraint(self):
        self.collision_free_constraint_eval = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=1e-3,
            influence_distance_offset=1e-1,
            plant_context=self.plant_context
        )
        def eval_func(vars = None, q = np.zeros(7), pose = None):
            return 0.1 * self.collision_free_constraint_eval.Eval(q)
        lb = np.array([-np.inf])
        ub = np.array([0.1])
        self.collision_free_constraint = IKFlowConstraints(lb, ub, eval_func, description="CollisionFreeConstraint")
        self.constraints.append(self.collision_free_constraint)
        return self.collision_free_constraint
    
    def CreateJointLimitsConstraint(self):
        lower_limits = self.plant.GetPositionLowerLimits()
        upper_limits = self.plant.GetPositionUpperLimits()
        def eval_func(vars = None, q = None, pose = None):
            return q
        self.joint_limit_constraint = IKFlowConstraints(lower_limits, upper_limits, eval_func, description="JointLimitsConstraint")
        self.constraints.append(self.joint_limit_constraint)
        return self.joint_limit_constraint


    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -5. * np.ones(self.ik_solver.network_width), 5. * np.ones(self.ik_solver.network_width), self.z
        )
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        self.c_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            self.initial_guess - 1, self.initial_guess + 1,self.c
        )
        self.c_bounding_box_constraint.evaluator().set_description("CBoundingBoxConstraint")
        bound = self.options.correction_bound
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -bound * np.ones(7), bound * np.ones(7), self.correction
        )
        self.correction_bounding_box_constraint.evaluator().set_description("CorrectionBoundingBoxConstraint")
    

    
    def JointCenteringCost(self):
        self.joint_centering_cost = self.prog.AddCost(
            func = self.EvalJointCenteringCost,
            vars = self.lumped_vars
        )
        self.joint_centering_cost.evaluator().set_description("JointCenteringCost")
    
    def EvalJointCenteringCost(self, vars):
        # Shares the constraint binding's flow evaluation when share_flow_evaluations is
        # on; otherwise this is a second full forward pass / jacrev at the same point.
        q, _ = self.QAndPose(vars)
        diff = q[:7] - self.q_nominal
        return 0.5 * diff @ (self.options.joint_centering_cost * np.eye(7)) @ diff
    
    def CorrectionCost(self):
        self.correction_cost = self.prog.AddQuadraticCost(
            Q=self.options.correction_cost_weight * np.eye(7),
            b=np.zeros(7),
            vars=self.correction
        )
        self.correction_cost.evaluator().set_description("CorrectionCost")
    

    def Solve(self):

        if os.path.exists(self.options.file_print_name):
            with open(self.options.file_print_name, "r+") as f:
                f.seek(0)
                f.truncate()
        
        if self.options.which_solver == "ipopt":
            solver = IpoptSolver()
            solver_options = SolverOptions()
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_tol", self.options.acceptable_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_constr_viol_tol", self.options.acceptable_constr_viol_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_dual_inf_tol", self.options.acceptable_dual_inf_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_compl_inf_tol", self.options.acceptable_compl_inf_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "file_print_level", self.options.file_print_level)
            solver_options.SetOption(IpoptSolver().solver_id(), "print_user_options", "yes")
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_iter", self.options.acceptable_iter)
            solver_options.SetOption(IpoptSolver().solver_id(), "max_wall_time", self.options.max_wall_time)
            if self.options.ipopt_mu_strategy is not None:
                solver_options.SetOption(IpoptSolver().solver_id(), "mu_strategy", self.options.ipopt_mu_strategy)
            if self.options.max_iter is not None:
                solver_options.SetOption(IpoptSolver().solver_id(), "max_iter", int(self.options.max_iter))
            
        if self.options.which_solver == 'snopt':
            solver = SnoptSolver()
            solver_options = SolverOptions()
            solver_options.SetOption(SnoptSolver.id(), "Major print level", self.options.file_print_level)
            solver_options.SetOption(SnoptSolver.id(), "Timing Level", 3)
            solver_options.SetOption(SnoptSolver.id(), "Time Limit", self.options.max_wall_time)
            solver_options.SetOption(SnoptSolver.id(), "Major optimality tolerance", self.options.acceptable_tol)
            solver_options.SetOption(SnoptSolver.id(), "Minor optimality tolerance", self.options.acceptable_tol)
            solver_options.SetOption(SnoptSolver.id(), "Major feasibility tolerance", self.options.acceptable_constr_viol_tol)
            if self.options.max_iter is not None:
                solver_options.SetOption(SnoptSolver.id(), "Major iterations limit", int(self.options.max_iter))
            if self.options.snopt_function_precision is not None:
                # SNOPT otherwise assumes the constraints are accurate to ~1e-13 and
                # probes derivatives at h=5.5e-7, which is pure noise for a float32 flow.
                solver_options.SetOption(SnoptSolver.id(), "Function precision", self.options.snopt_function_precision)
            # solver_options.SetOption(SnoptSolver.id(), "Major Iteration Limit", 4 * self.options.max_wall_time)


        
        solver_options.SetOption(CommonSolverOption.kPrintFileName, self.options.file_print_name)



        self.prog.AddVisualizationCallback(
            partial(visualization_callback, diagram=self.diagram, diagram_context=self.diagram_context,
                                                plant=self.plant, plant_context=self.plant_context,
                                                vars_to_q=self.VarsToQ, vars_file = self.options.vars_file, visualize = self.options.visualize),
            self.lumped_vars
        )
        
        return solver.Solve(self.prog, solver_options=solver_options)


def visualization_callback(vars, diagram, diagram_context, plant, plant_context, vars_to_q, vars_file, visualize):
    if visualize or vars_file is not None:
        q = vars_to_q(vars)
        if visualize:
            plant.SetPositions(plant_context, q)
            diagram.ForcedPublish(diagram_context)
        if vars_file is not None:
            with open(vars_file, "a") as f:
                f.write(",".join([str(val) for val in vars]) + "\n")