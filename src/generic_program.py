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
    RotationMatrix,
    RollPitchYaw,
)

@dataclass
class ProgramOptions:
    joint_centering_cost: float = field(default=0.0, metadata={"help": "Weight for joint centering cost"})
    collision_avoidance: bool = field(default=True, metadata={"help": "Add collision avoidance constraints"})
    joint_limits: bool = field(default=True, metadata={"help": "Enforce joint limits"})
    ik_constraint_tol: tuple = field(default=(1e-4, 0.01), metadata={"help": "Tolerance for IK constraints: tuple of (position tol, orientation tol)"})
    correction_cost_weight: float = field(default=0.0, metadata={"help": "Weight for correction cost to keep close to zero"})

    mug_height: float = field(default=0.035, metadata={"help": "Mug height for valid grasp poses"})

    ## Network evaluation ##
    # The flow is a float32 artifact, but evaluating it in float64 costs ~15% and
    # makes the map smooth well below 1e-4. In float32 the finite-difference error
    # against the analytic Jacobian blows up from 6e-3 (h=1e-4) to 1.8 (h=1e-6),
    # which is what makes SNOPT's derivative check fail and starves the line search.
    use_float64: bool = field(default=True, metadata={"help": "Evaluate the flow in float64 so the map is smooth at solver step sizes"})

    ## Multi-start seeding ##
    # A batched forward pass is nearly free (n=256 costs the same as n=1 on GPU),
    # so draw many candidates and start from the most feasible one.
    num_seed_samples: int = field(default=256, metadata={"help": "Candidate (c, z) pairs drawn to pick an initial guess. 0 disables seeding"})
    seed_refine_top_k: int = field(default=8, metadata={"help": "Candidates re-scored against every constraint (incl. collision)"})
    seed_latent_scale: float = field(default=1.0, metadata={"help": "Std dev of the sampled latents"})
    c_position_slack: float = field(default=0.25, metadata={"help": "Half-width of the box on the conditioning position, about the target"})


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



# TODO: replace this with a custom evaluator that computes theta^2 and its derivative
# exactly as a function of d = |q . q_target|, instead of composing arccos and clamping.
#
# Why: theta = 2*arccos(d) has an infinite derivative at d = 1, i.e. exactly at
# convergence. The eps clamp below does not merely bound that -- when it binds,
# pydrake.math.min returns the float bound promoted to an AutoDiffXd with an *empty*
# derivative vector, so the orientation row reports no gradient at all inside
# theta < 2.8e-4 rad. Squaring removes the branch point (theta^2 ~ 8*(1-d), linear in d),
# but writing (2*arccos(d))**2 literally still evaluates 0 * inf at d = 1 and yields NaN.
#
# Implementation notes:
#   value:      theta^2 = 4*arccos(d)^2
#   derivative: d(theta^2)/dd = -8*arccos(d)/sqrt(1-d^2),  which is 0/0 at d = 1 with
#               limit -8, so branch to a series near coincidence. With u = 1-d,
#               arccos(1-u) = sqrt(2u)*(1 + u/12 + 3u^2/160 + ...), hence
#               theta^2 = 8u*(1 + u/6 + ...) and d(theta^2)/dd = -8*(1 + u/3 + ...).
#   Assemble the result as AutoDiffXd(value, dtheta2_dd * d.derivatives()).
#
# Callers must also change with it:
#   - CreateIKConstraint's lb for this row must become -inf rather than 0. theta^2 >= 0
#     automatically, and grad(theta^2) = 0 at the solution (grad(d) vanishes there
#     because d is at its maximum), so an lb of 0 makes the row active with a zero
#     gradient at convergence -- the same LICQ failure as the mug constraint's
#     homogeneous row. This holds however exactly theta^2 is computed.
#   - ub becomes ori_tol**2, which is 1e-4 for the current 0.01 rad and therefore the
#     same order as acceptable_constr_viol_tol; normalize the row (theta^2/ori_tol^2 <= 1)
#     so the solver's slack does not widen the effective angular tolerance by ~40%.
#   - This changes the meaning of ik_constraint_tol[1], so archived solver logs stop
#     being comparable across the change.
#
# Only the pose-target programs reach this (PandaIKProgram, Iiwa14IKProgram and their
# ...Numerical variants); the mug programs override CreateIKConstraint and leave
# orientation free.
def orientation_error(orientation, target_orientation, eps=1e-8):
    dot_product = np.dot(orientation, target_orientation)
    clipped_value = pydrake.math.min(1.0 - eps, pydrake.math.max(-1.0 + eps, np.abs(dot_product)))
    return 2.0 * pydrake.math.arccos(clipped_value)

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

    def add_costs(self):
        if self.options.joint_centering_cost > 0.0:
            self.JointCenteringCost()
        if self.options.correction_cost_weight > 0.0:
            self.CorrectionCost()

    def fk(self, q, matrix = False):
        frame, context = self.SetPositions(q)
        rigid_transform = frame.CalcPoseInWorld(context)
        if matrix:
            return rigid_transform.GetAsMatrix4()
        else:
            return rigid_transform.translation(), rigid_transform.rotation().ToQuaternion().wxyz()

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

    def BatchInference(self, c_candidates, z_candidates):
        '''Evaluate the flow on a whole batch of (c, z) pairs at once.

        Batching is nearly free in float32 (batch 256 costs 8.8 ms vs 5.9 ms for batch 1,
        because the evaluation is CPU-dispatch bound, not FLOP bound) but NOT in float64,
        where the same batch costs 84 ms vs 6.8 ms. See scripts/profiling/profile_flow.py.
        '''
        n = len(c_candidates)
        dtype = self.torch_dtype
        pose7 = np.array([self.CToPose7(c) for c in c_candidates])
        c_t = torch.tensor(pose7, dtype=dtype, device=DEVICE)
        c_t = torch.cat([c_t, torch.zeros((n, 1), dtype=dtype, device=DEVICE)], dim=1)
        z_t = torch.tensor(np.asarray(z_candidates), dtype=dtype, device=DEVICE)
        with torch.no_grad():
            output, _ = self.ik_solver.nn_model(z_t, c=c_t, rev=True)
        return output[:, :self.num_arm_dof].detach().cpu().numpy().astype(float)

    def SeedCandidates(self, n):
        '''Candidate (c, z) pairs to seed from. Subclasses whose task leaves part of the
        target pose free should override this and sample that freedom too.'''
        c = np.tile(np.asarray(self.prog.GetInitialGuess(self.c), dtype=float), (n, 1))
        z = self.options.seed_latent_scale * np.random.randn(n, self.ik_solver.network_width)
        return c, z

    @staticmethod
    def ConstraintViolation(constraint, vars, q, pose):
        value = np.asarray(constraint.eval_func(vars=vars, q=q, pose=pose), dtype=float)
        return float(np.sum(np.maximum(0.0, constraint.lb - value) + np.maximum(0.0, value - constraint.ub)))

    def SeedInitialGuess(self):
        '''Pick an initial guess by drawing candidates from the flow and keeping the most
        feasible one, instead of starting from a single arbitrary latent draw.

        Scoring is two-stage: every candidate is ranked on the IK constraint (forward
        kinematics only), then the best few are re-scored against every constraint,
        because the collision query is the expensive part.
        '''
        n = self.options.num_seed_samples
        if n <= 0:
            return None

        c_candidates, z_candidates = self.SeedCandidates(n)
        q_batch = self.BatchInference(c_candidates, z_candidates)

        qs, poses = [], []
        ik_violation = np.empty(n)
        for i in range(n):
            q = self.PadQ(q_batch[i])
            pose = self.fk(q)
            qs.append(q)
            poses.append(pose)
            ik_violation[i] = self.ConstraintViolation(self.ik_constraint, None, q, pose)

        top_k = np.argsort(ik_violation)[:max(1, self.options.seed_refine_top_k)]

        best_index, best_violation = int(top_k[0]), np.inf
        for i in top_k:
            vars_i = np.concatenate([c_candidates[i], z_candidates[i], np.zeros(self.num_arm_dof)])
            violation = sum(self.ConstraintViolation(c, vars_i, qs[i], poses[i]) for c in self.constraints)
            if violation < best_violation:
                best_violation, best_index = violation, int(i)

        self.prog.SetInitialGuess(self.c, c_candidates[best_index])
        self.prog.SetInitialGuess(self.z, z_candidates[best_index])
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        return best_violation

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
        q = self.VarsToQ(vars) ## this is ran once for all constraints
        pose = self.fk(q) ## this is ran once for all constraints
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
        pos_tol, ori_tol = self.options.ik_constraint_tol
        lb = np.array([-pos_tol] * 3 + [0])
        ub = np.array([pos_tol] * 3 + [ori_tol])
        def eval_func(vars, q, pose):
            position, orientation = pose
            pos_error = position - self.target_pose[:3]
            orientation_err = orientation_error(orientation, self.target_pose[3:])
            return np.concatenate([pos_error, np.array([orientation_err])])
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
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -0.1 * np.ones(7), 0.1 * np.ones(7), self.correction
        )
        self.correction_bounding_box_constraint.evaluator().set_description("CorrectionBoundingBoxConstraint")
    

    
    def JointCenteringCost(self):
        self.joint_centering_cost = self.prog.AddCost(
            func = self.EvalJointCenteringCost,
            vars = self.lumped_vars
        )
        self.joint_centering_cost.evaluator().set_description("JointCenteringCost")
    
    def EvalJointCenteringCost(self, vars):
        q = self.VarsToQ(vars)
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
        if self.options.which_solver == 'snopt':
            solver = SnoptSolver()
            solver_options = SolverOptions()
            solver_options.SetOption(SnoptSolver.id(), "Major print level", self.options.file_print_level)
            solver_options.SetOption(SnoptSolver.id(), "Timing Level", 3)
            solver_options.SetOption(SnoptSolver.id(), "Time Limit", self.options.max_wall_time)
            solver_options.SetOption(SnoptSolver.id(), "Major optimality tolerance", self.options.acceptable_tol)
            solver_options.SetOption(SnoptSolver.id(), "Minor optimality tolerance", self.options.acceptable_tol)
            solver_options.SetOption(SnoptSolver.id(), "Major feasibility tolerance", self.options.acceptable_constr_viol_tol)
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