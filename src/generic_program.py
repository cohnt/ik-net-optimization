import os
import time

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
    # The shape of the collision row, exposed so it can be swept. The defaults are the
    # values that were hardcoded in CreateCollisionFreeConstraint, so nothing moves unless
    # they are set. `collision_influence_offset` is the distance at which a geometry pair
    # starts contributing to Drake's smooth penalty, i.e. it sets the gradient the solver
    # has to follow while it is still far from contact; `collision_row_scale` scales the
    # whole row (and its upper bound with it, since the binding's own threshold is 1).
    collision_bound: float = field(default=1e-3, metadata={"help": "MinimumDistanceLowerBoundConstraint 'bound' (metres)"})
    collision_influence_offset: float = field(default=1e-1, metadata={"help": "MinimumDistanceLowerBoundConstraint 'influence_distance_offset' (metres)"})
    collision_row_scale: float = field(default=0.1, metadata={"help": "Scaling applied to the collision constraint row and its upper bound"})
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

    # torch.compile on the jacrev: measured 17.98 -> 13.55 ms (1.33x) at batch 1 in float64,
    # one dynamo graph with no recompiles across iterates, agreeing with eager to 1e-14 on a
    # Jacobian of magnitude 12, for a 14.2 s one-off compile penalty. The Jacobian is 84% of
    # a learned solve, so inside a fixed wall-clock cap this is roughly 30% more iterations
    # -- which means it *moves the learned arm's success rate* and every arm of a reported
    # comparison has to be run with the same setting. Off by default so a solve costs no
    # compile; the benchmark scripts turn it on and warm it up before the grid.
    compile_flow_jacobian: bool = field(default=False, metadata={"help": "torch.compile the flow Jacobian once per process and share it between programs"})


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
    # Which discrete branch set the Panda analytic chart uses: 4 is the historical chart
    # (elbow branch pinned, the half far from the joint limits); 8 adds the mirrored elbow
    # branch, taking round-trip coverage from 89.4% to 99.4% of random configurations. The
    # default stays 4 so archived runs remain reproducible; the residual ~0.6% at 8 is a
    # measured property of this chart, left as future work (arXiv:2503.03992 may help).
    analytic_branches: int = 4
    # Restores the pre-2026-08-31 paired start, in which the conditioning pose was clipped
    # into its box before the solve and the correction started at zero -- leaving the
    # learned arm 1.2-3.3 rad from the shared q_init. The repaired default sets the guess
    # exactly (Drake accepts an infeasible initial guess; IPOPT projects bounds itself),
    # for reproducing archived runs only.
    legacy_paired_start: bool = False
    # Degrade the flow's chart by a deterministic smooth perturbation of this magnitude
    # (rad, per-joint sin features of [c; z]); 0 disables. Experimental knob for the
    # chart-error dose-response -- see MakeFlowInference.
    chart_error_scale: float = 0.0
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

## ---------------------------- the flow evaluation ----------------------------- ##
#
# The network forward pass lives here as a *free function of the lumped variables* rather
# than as a method, for one reason: torch.compile guards on everything the compiled callable
# closes over, so a bound method would carry the program instance into the guards and each of
# the thirty programs in a benchmark grid would re-trigger dynamo. Closing over the network
# alone lets one graph be compiled once per process and reused by every program that shares
# it. (This is what the old "200 ms compilation penalty per program" comment was measuring:
# the penalty was per instance because the compiled thing was per instance.)


def MakeFlowInference(nn_model, width, num_arm_dof, device, chart_error_scale=0.0):
    """`vars -> (q, q)`, the shape `jacrev(..., has_aux=True)` wants.

    Returning q twice is what lets one reverse pass yield both dq/dvars and q, instead of
    evaluating the network again for the value. `vars` is [conditioning pose as xyz + wxyz,
    latent, correction]; the trailing zero on the conditioning row is the padding the flow
    was trained with.

    `chart_error_scale` adds a deterministic, smooth, seeded perturbation
    `eps * sin(W [c; z] + b)` to the network's output -- an experimental knob that
    degrades the chart's accuracy without touching anything else, so success can be
    measured against chart error with the scene, kinematics and solver held fixed (the
    dose-response experiment: the Panda's 3.8 mm chart pushed through the iiwa's
    16.6-64 mm regime). It is a function of the conditioning pose and latent only, so the
    correction's analytic identity block in the Jacobian is untouched.
    """
    if chart_error_scale:
        # Explicit device="cpu": ikflow installs a global default-device override that
        # would otherwise pair the CPU generator with a CUDA allocation and fail.
        gen = torch.Generator(device="cpu").manual_seed(0)
        W = torch.randn((7 + width, num_arm_dof), generator=gen,
                        dtype=torch.float64, device="cpu").to(device)
        b = torch.randn(num_arm_dof, generator=gen,
                        dtype=torch.float64, device="cpu").to(device)

    def flow_inference(vars):
        c, z, correction = vars[:7], vars[7:7 + width], vars[7 + width:]
        c_torch = torch.cat(
            [c.unsqueeze(0), torch.zeros((1, 1), dtype=vars.dtype, device=device)], dim=1)
        output, _ = nn_model(z.unsqueeze(0), c=c_torch, rev=True)
        q = output[0, :num_arm_dof] + correction
        if chart_error_scale:
            q = q + chart_error_scale * torch.sin(
                vars[:7 + width].to(W.dtype) @ W + b).to(q.dtype)
        return q, q
    return flow_inference


_COMPILED_JACOBIANS = {}


def FlowJacobianGen(nn_model, width, num_arm_dof, device, compile_it, chart_error_scale=0.0):
    """`vars -> (dq/dvars, q)`, compiled once per (network, shape, dtype) if asked.

    Reverse mode is the right primitive at this shape -- 7 outputs against 21 inputs, of
    which 13 reach the network -- and the measurements behind that are in CLAUDE.md.
    """
    jacobian_gen = torch.func.jacrev(
        MakeFlowInference(nn_model, width, num_arm_dof, device, chart_error_scale),
        has_aux=True)
    if not compile_it:
        return jacobian_gen
    key = (id(nn_model), width, num_arm_dof, str(device),
           next(nn_model.parameters()).dtype, float(chart_error_scale))
    if key not in _COMPILED_JACOBIANS:
        _COMPILED_JACOBIANS[key] = torch.compile(jacobian_gen)
    return _COMPILED_JACOBIANS[key]


class IKFlowConstraints:
    def __init__(self, lb, ub, eval_func, description=""):
        self.lb = lb
        self.ub = ub
        self.eval_func = eval_func
        self.description = description
    def __len__(self):
        return len(self.lb)


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
        ## `correction` is a learned-only decision variable, so this must be guarded the
        ## same way `latent_cost_weight` and `latent_trust_region` are -- the baseline
        ## programs share this options object. Ungated, a `--set correction_cost_weight`
        ## run raises AttributeError inside every numerical/analytic program's
        ## construction and scores that whole column 0 in about 10 ms per cell.
        if self.options.correction_cost_weight > 0.0 and hasattr(self, "correction"):
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

    def FlowInference(self):
        """The eager forward pass, built once per program and shared with the compiled
        Jacobian so both paths run identical code."""
        fn = getattr(self, "_flow_inference", None)
        if fn is None:
            fn = self._flow_inference = MakeFlowInference(
                self.ik_solver.nn_model, self.ik_solver.network_width,
                self.num_arm_dof, DEVICE, self.options.chart_error_scale)
        return fn

    def MakeJacobianGen(self):
        return FlowJacobianGen(
            self.ik_solver.nn_model, self.ik_solver.network_width, self.num_arm_dof,
            DEVICE, self.options.compile_flow_jacobian, self.options.chart_error_scale)

    def WarmUpJacobian(self):
        """Pay torch.compile's one-off cost outside any timed solve.

        Returns the seconds spent. The benchmark scripts call this on the sampler program,
        which holds the same network every later program is handed, so the grid never sees
        the compile.
        """
        import time
        width = self.ik_solver.network_width
        vars = np.zeros(7 + width + self.num_arm_dof)
        vars[3] = 1.0                                  # a unit quaternion, w first
        tensor = torch.tensor(vars, dtype=self.torch_dtype, device=DEVICE)
        start = time.time()
        gen = self.MakeJacobianGen()
        gen(tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time() - start

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
        # A local, fixed-seed generator, for two reasons. The offset is constant by
        # construction, but it is *measured*, so it carries ~1e-8 of numerical noise that
        # differs with the configurations it was measured at; drawing them from the global
        # stream gave every program a slightly different X_ee_flow, and the flow amplifies
        # 1e-8 in the conditioning pose to 1e-6 in q -- enough that two arms of the same
        # cell were not solving quite the same problem. It also stops this call from
        # consuming global draws, which used to shift the benchmark's target grid depending
        # on whether calibrate_flow_frame was on.
        rng = np.random.default_rng(0)
        offsets = []
        for _ in range(samples):
            q_arm = rng.uniform(lower, upper)
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
        # The c region is a general linear constraint now, not a variable bound, so
        # _SetClipped/_BoxDistance no longer see it; measure and (for legacy) apply the
        # clip against the stored region instead.
        c_lo, c_hi = self.c_box
        z_lo, z_hi = self.z_box
        c_clip_distance = float(np.linalg.norm(np.clip(c, c_lo, c_hi) - c))
        z_clip_distance = float(np.linalg.norm(np.clip(z, z_lo, z_hi) - z))
        if self.options.legacy_paired_start:
            self.prog.SetInitialGuess(self.c, np.clip(c, c_lo, c_hi))
            self.prog.SetInitialGuess(self.z, np.clip(z, z_lo, z_hi))
            clipped = c_clip_distance + z_clip_distance
            self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
            return clipped
        # The exact paired start. The conditioning pose is set *unclipped* -- a Drake
        # initial guess need not satisfy the bounds, and IPOPT projects variables into
        # their box itself -- so flow(c, z) reproduces q_arm to the network's noise floor
        # (measured ~1e-6 in float32, tighter in float64), and the correction closes that
        # residual. q(start) is then q_arm to float precision, which is what "paired"
        # claims; the pre-clipped version started 1.2-3.3 rad away. The distance from c to
        # its box is returned as the clip distance: it is how far the solver's own
        # projection will move the first iterate.
        self.prog.SetInitialGuess(self.c, c)
        self.prog.SetInitialGuess(self.z, z)
        clipped = c_clip_distance + z_clip_distance
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        residual = q_arm - np.asarray(
            self.VarsToQ(self.prog.GetInitialGuess(self.lumped_vars)), dtype=float)[:self.num_arm_dof]
        bound = self.options.correction_bound
        residual = np.nan_to_num(residual, nan=0.0, posinf=bound, neginf=-bound)
        self.prog.SetInitialGuess(self.correction, np.clip(residual, -bound, bound))
        return clipped

    def SetNativeStart(self, q_init, rng):
        """This formulation's *own* initialisation, as it would be run outside a comparison.

        The learned formulation's natural procedure is the flow's inference procedure:
        condition on the pose the task hands you and draw the latent from the prior the
        network was trained against, with no correction. `create_prog` has already set the
        conditioning pose from the target, so only the latent is drawn here.

        This is a *sample*, not a search. Nothing in it looks at the problem's constraints
        or its objective, and no candidate is scored or selected -- which is the line that
        separates a formulation's natural initialisation from solving part of the problem
        outside the solver.

        `q_init` is accepted so that formulations whose natural start *is* a configuration
        can use it; this one ignores it. Returns a clip distance, for symmetry with
        `SetStartFromQ`.
        """
        self.prog.SetInitialGuess(self.z, rng.standard_normal(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        return 0.0

    def _VariableBounds(self, variables):
        '''The tightest bounding box the program imposes on `variables`.'''
        lower = np.full(len(variables), -np.inf)
        upper = np.full(len(variables), np.inf)
        names = {v.get_id(): i for i, v in enumerate(variables)}
        for binding in self.prog.bounding_box_constraints():
            evaluator = binding.evaluator()
            for row, var in enumerate(binding.variables()):
                index = names.get(var.get_id())
                if index is not None:
                    lower[index] = max(lower[index], evaluator.lower_bound()[row])
                    upper[index] = min(upper[index], evaluator.upper_bound()[row])
        return lower, upper

    def _SetClipped(self, variables, values):
        '''Set an initial guess, clipped into the program's bounding box on it.'''
        values = np.asarray(values, dtype=float)
        lower, upper = self._VariableBounds(variables)
        clipped = np.clip(values, lower, upper)
        self.prog.SetInitialGuess(variables, clipped)
        return float(np.linalg.norm(clipped - values))

    def _BoxDistance(self, variables, values):
        '''How far `values` sits outside the program's bounding box on `variables` --
        the projection distance IPOPT will apply at its first iterate when the guess is
        set unclipped.'''
        values = np.asarray(values, dtype=float)
        lower, upper = self._VariableBounds(variables)
        return float(np.linalg.norm(np.clip(values, lower, upper) - values))

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
        scale = self.options.collision_row_scale
        self.collision_free_constraint_eval = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=self.options.collision_bound,
            influence_distance_offset=self.options.collision_influence_offset,
            plant_context=self.plant_context
        )
        def eval_func(vars = None, q = np.zeros(7), pose = None):
            return scale * self.collision_free_constraint_eval.Eval(q)
        lb = np.array([-np.inf])
        # The binding's raw value is "in collision" above 1, so the scaled bound is the
        # scale itself; keeping the two coupled means changing the scale reshapes the
        # gradient without moving the feasible set.
        ub = np.array([scale])
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


    def LatentBoxConstraint(self):
        '''`-5 <= z <= 5`, as a general linear constraint rather than a variable bound.

        Defined once here and called by every program that owns a latent -- the pose
        programs through `BoundingBoxConstraint` below and the mug programs through their
        overrides -- because the three copies this replaces are how the bug below survived
        the first repair: the pose arms were fixed and the mug arms silently were not.

        The reason it is not a bounding box is the same one that applies to the
        conditioning pose, but the failure was worse, because we applied the projection
        ourselves rather than leaving it to IPOPT: `SetStartFromQ` clipped the inverted
        latent into this box before the solver ever ran. The flow is a bijection, so
        `flow(c, InvertFlow(q, c))` reproduces `q` exactly -- but only at the *unclipped*
        latent. On the iiwa pose task the inversion routinely returns components past +-5
        (measured |z| ~ 9.1), so the clip moved the start several radians, the +-0.1
        correction could not close the residual, and 49 of 60 paired cells were recorded
        as `unrepresentable_start`: an arm scored as unable to represent a configuration
        it represents exactly. The feasible set is unchanged; what changes is only the
        guess's freedom to start outside it.
        '''
        width = self.ik_solver.network_width
        self.z_box = (-5. * np.ones(width), 5. * np.ones(width))
        self.bounding_box_constraint = self.prog.AddLinearConstraint(
            np.eye(width), self.z_box[0], self.z_box[1], self.z)
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        return self.bounding_box_constraint

    def BoundingBoxConstraint(self):
        self.LatentBoxConstraint()
        # A general linear constraint, deliberately NOT a bounding box, and the
        # distinction is load-bearing. IPOPT (an interior-point method) requires every
        # iterate to sit strictly inside the *variable bounds* -- its bound_push projects
        # the initial guess into the box before evaluating anything, which silently
        # destroyed the exact paired start: `c` was teleported to the box face while the
        # latent stayed tuned to the unprojected pose, so the first evaluated point was
        # 1-3 rad from q_init and bit-identical to the old pre-clipped protocol (measured:
        # identical iterate-0 lines in the IPOPT logs). General constraints carry no such
        # interiority requirement -- they may start violated, the violation just lands in
        # inf_pr -- so with the box written this way the solver genuinely starts at the
        # guess and walks `c` into the region continuously while `z` and the correction
        # adapt, instead of being jolted onto the face at iterate 0.
        self.c_box = (self.initial_guess - 1, self.initial_guess + 1)
        self.c_box_constraint = self.prog.AddLinearConstraint(
            np.eye(len(self.c)), self.c_box[0], self.c_box[1], self.c)
        self.c_box_constraint.evaluator().set_description("CBoxConstraint")
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



        inner = partial(visualization_callback, diagram=self.diagram, diagram_context=self.diagram_context,
                                                plant=self.plant, plant_context=self.plant_context,
                                                vars_to_q=self.VarsToQ, vars_file = self.options.vars_file, visualize = self.options.visualize)

        def record_iterate(vars):
            # Keep the newest iterate on the program, in memory, always. A solve that ends
            # abnormally -- an exception, a harness kill, the (measured, once in 1740
            # cells) C++-level wedge inside a single IPOPT iteration -- can then still be
            # verified from the point the solver actually had, instead of the point being
            # discarded with the solve. The predecessor design raised SolveTimeout from
            # the constraint callback, which both threw the iterate away and could not
            # fire during the wedge (no Python ran for 102 minutes); when a wedge
            # releases, IPOPT's own max_wall_time ends the solve at the next iteration
            # boundary with the iterate intact, which needs no help from us.
            self.last_iterate = np.array(vars, dtype=float)
            inner(vars)

        self.prog.AddVisualizationCallback(record_iterate, self.lumped_vars)
        
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