from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE
import torch
import numpy as np
from src.utils import Mug
from src.generic_program import *
import numpy as np
from pydrake.all import (
    MathematicalProgram,
    RollPitchYaw,
    AutoDiffXd,
    Quaternion,
    RotationMatrix,
    RotationMatrix_,
    RollPitchYaw_,
    RigidTransform,
    RigidTransform_,
    Quaternion_,
)
from src.panda_analytic_ik import Analytic_IK_Panda




class PandaIKProgram(IKFlowProgram):
    def __init__(self, diagram, options = ProgramOptions(), model = None):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context) 

        self.frame = self.plant.GetBodyByName("panda_hand").body_frame()
        self.autodiff_frame = self.autodiff_plant.GetBodyByName("panda_hand").body_frame()
        self.num_pos = self.plant.num_positions()
        self.num_arm_dof = 7

        if model is None:
            model_name = "panda__full__lp191_5.25m"
            self.ik_solver, _ = get_ik_solver(model_name)
        else:
            self.ik_solver = model

        self.options = options
        self.ConfigureNetworkDtype()
        self.constraints = []
        # Size of the first block of decision variables, from which the conditioning pose
        # is built. Six either way: a free conditioning pose (xyz + rpy), or the grasp
        # pose in the mug frame under c_parameterization="task".
        self.num_task_vars = 6



    def create_prog(self, target_pose = np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal = None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6) # x y z roll pitch yaw into nn model
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width) # latent variables
        self.correction = self.prog.NewContinuousVariables(7) ## small correction term to q

        self.lumped_vars = np.hstack([self.c, self.z, self.correction])

        ## TODO: Change the initial guess to something smarter

        self.target_pose = target_pose
        if q_nominal is None:
            self.q_nominal = np.zeros(7)
        else:
            self.q_nominal = q_nominal

        self.initial_guess = np.zeros(6)
        self.initial_guess[:3] = self.target_pose[:3]
        self.initial_guess[3:] = RotationMatrix(Quaternion(self.target_pose[3:])).ToRollPitchYaw().vector()


        self.prog.SetInitialGuess(self.c, self.initial_guess)
        self.prog.SetInitialGuess(self.z, np.random.randn(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(7))


        # jacrev already computes the forward pass, so ask it for the value too
        # (has_aux) rather than evaluating the network a second time in VarsToQ.
        # torch.compile is deliberately not used here: measured 39.5 ms either way,
        # against a 200 ms compilation penalty per program.
        self.jacobian_gen = torch.func.jacrev(self.ik_inference_with_value, has_aux=True)

        ## Add Constraints
        self.add_constraints()

        self.add_costs()


    def ik_inference(self, vars, add_correction = True):
        '''Given a latent + target + correction, returns corresponding joint angles
        vars can be either numpy array or torch tensor (for gradient computation)'''
        # Convert to tensor only if not already a tensor
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=self.torch_dtype)

        c, z, correction = (vars[:7], vars[7:7+self.ik_solver.network_width], vars[7+self.ik_solver.network_width:])
        # Work directly with tensor slices - don't call torch.tensor() again!
        c_torch = torch.cat([c.unsqueeze(0), torch.zeros((1, 1), dtype=vars.dtype, device=DEVICE)], dim=1)
        z_batch = z.unsqueeze(0)

        output, _ = self.ik_solver.nn_model(z_batch, c=c_torch, rev=True)
        q = output[:, :7].squeeze(0)
        if add_correction:
            return q + correction
        else: return q

    def ik_inference_with_value(self, vars):
        '''jacrev(..., has_aux=True) target: returns q twice so one reverse pass yields
        both dq/dvars and q.'''
        q = self.ik_inference(vars)
        return q, q



    def TaskVarsToPose7(self, task_vars, t):
        '''Task variables -> the (xyz, wxyz) the flow is conditioned on.

        The default parameterisation is the identity in all but coordinates: the task
        variables *are* the conditioning pose, written as xyz + rpy. Subclasses that fold
        the task into the variables override this, and because the whole expression is
        evaluated on Drake's templated types the AutoDiffXd derivatives flow through it
        without any extra chain rule.
        '''
        xyz = task_vars[:3]
        quaternion = RotationMatrix_[t](RollPitchYaw_[t](task_vars[3:6])).ToQuaternion().wxyz()
        return xyz, quaternion

    def VarsToQ(self, rpy_vars, add_correction = True):

        ad = isinstance(rpy_vars[0], AutoDiffXd)
        t = AutoDiffXd if ad else float
        width = self.ik_solver.network_width
        n = self.num_task_vars

        vars = np.zeros(7 + width + 7, dtype=AutoDiffXd if ad else np.float64)
        xyz, quaternion = self.TaskVarsToPose7(rpy_vars[:n], t)
        vars[:3] = xyz
        vars[3:7] = quaternion
        vars[7:7 + width] = rpy_vars[n:n + width]
        vars[7 + width:] = rpy_vars[n + width:]


        if not ad:
            q = np.zeros(self.num_pos)
            q[7:] = [0.04] * (self.num_pos - 7)  # fixed gripper joints
            q[:7] = self.ik_inference(vars, add_correction=add_correction).detach().cpu().numpy()
            return q

        else: # Compute AutoDiffXd with Jacobian_Gen
            # Extract values and gradients from AutoDiffXd
            vars_values = np.array([v.value() for v in vars])
            vars_gradients = np.array([v.derivatives() for v in vars])

            # One reverse pass gives both dq/dvars and q.
            vars_tensor = torch.tensor(vars_values, dtype=self.torch_dtype, device=DEVICE)
            jacobian, q_tensor = self.jacobian_gen(vars_tensor)
            jacobian_np = jacobian.detach().cpu().numpy()

            q_values = np.zeros(self.num_pos)
            q_values[7:] = [0.04] * (self.num_pos - 7)
            q_values[:7] = q_tensor.detach().cpu().numpy()

            # Chain rule: dq/dvars @ dvars = dq
            # For each element of q, compute gradient via chain rule
            q_gradients = np.zeros((self.num_pos, len(rpy_vars)))
            q_gradients[:7, :] = jacobian_np @ vars_gradients

            # Create AutoDiffXd objects with value and gradient
            q_ad = np.array([AutoDiffXd(q_values[i], q_gradients[i]) for i in range(len(q_values))])
            return q_ad




class PandaMugProgram(PandaIKProgram):
    '''Program for grasping pose of a mug for Panda'''
    def __init__(self, diagram, options = ProgramOptions(), model = None):
        super().__init__(diagram, options, model)
        # The flow is conditioned on the pose of the frame it was trained against
        # (panda_hand); the grasp constraint acts on the point between the fingers.
        # Keep both so seeds can be drawn in the frame the network understands.
        self.ee_frame = self.frame
        self.frame = self.plant.GetFrameByName("between_fingers")
        self.autodiff_frame = self.autodiff_plant.GetFrameByName("between_fingers")

        # Calibrate the conditioning frame before anything derived from it. `X_grasp_ee`
        # then maps the grasp frame to the frame the *flow* speaks in, not to whichever
        # body the scene happens to call "panda_hand".
        self.CalibrateFlowFrame()
        self.plant.SetPositions(self.plant_context, np.zeros(self.num_pos))
        X_W_flow = self.FlowPoseInWorld()
        X_W_grasp = self.frame.CalcPoseInWorld(self.plant_context)
        self.X_grasp_ee = X_W_grasp.inverse() @ X_W_flow


    def create_prog(self, target_mug = Mug(), q_nominal = None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6) # x y z qw qx qy qz into nn model
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width) # latent variables
        self.correction = self.prog.NewContinuousVariables(7) ## small correction term to q

        self.lumped_vars = np.hstack([self.c, self.z, self.correction])

        self.target_mug = target_mug
        if q_nominal is None:
            self.q_nominal = np.zeros(self.num_pos)
        else:
            self.q_nominal = q_nominal

        # `c` is the pose of the frame the flow was trained on, not of the grasp point, so
        # the mug centre is not a valid guess for it -- the two differ by X_grasp_ee, 0.1 m
        # for this gripper. Seeding or SetStartFromQ overwrites this, but a wrong default
        # is a trap for any caller that runs neither.
        X_W_ee = target_mug.middle @ self.X_grasp_ee
        self.prog.SetInitialGuess(self.c, np.concatenate(
            [X_W_ee.translation(), X_W_ee.rotation().ToRollPitchYaw().vector()]))
        self.prog.SetInitialGuess(self.z, np.random.randn(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(7))
        self.jacobian_gen = torch.func.jacrev(self.ik_inference_with_value, has_aux=True)

        self.target_pose = np.array([*target_mug.middle.translation(), 1, 0, 0, 0]) ## for bounding box
        self.add_constraints()
        self.add_costs()

    def CreateIKConstraint(self):
        # The gripper must lie on the mug's central axis (x = y = 0 exactly, as
        # ../codebase's MugConstraint imposes it) and within the mug's height along it.
        # Orientation is left free: any approach direction is a valid grasp.
        # x = y = 0 exactly: the grasp point lies *on* the mug's axis, and that equality
        # is the definition of the task rather than a tolerance to be chosen. (The
        # analytic arm's own mug constraint still uses +-ik_constraint_tol[0] on these
        # rows, which holds it to a slightly easier problem; the repair is to pin that
        # one too, not to loosen this one.)
        lb = np.array([0.0, 0.0, -self.options.mug_height])
        ub = np.array([0.0, 0.0, self.options.mug_height])
        def eval_func(vars, q, pose):
            position, _ = pose
            mug_transform = np.linalg.inv(self.target_mug.middle.GetAsMatrix4())
            # Row 3 of the homogeneous product is identically 1 with a zero gradient;
            # keeping it as an equality row costs a rank and breaks LICQ.
            return (mug_transform @ np.array([[*position, 1]]).T).squeeze()[:3]
        self.ik_constraint = IKFlowConstraints(lb, ub, eval_func, description="IKConstraint")
        self.constraints.append(self.ik_constraint)
        return self.ik_constraint

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -5. * np.ones(self.ik_solver.network_width), 5. * np.ones(self.ik_solver.network_width), self.z
        )
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        # Keep the conditioning pose near the mug. A +-5 m box lets the optimizer walk
        # the flow far outside the workspace it was trained on, where its output is
        # meaningless. Orientation stays free (+-2*pi avoids clipping rpy wraparound).
        centre = self.target_mug.middle.translation()
        slack = self.options.c_position_slack
        self.c_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            np.concatenate([centre - slack, -2 * np.pi * np.ones(3)]),
            np.concatenate([centre + slack, 2 * np.pi * np.ones(3)]),
            self.c
        )
        self.c_bounding_box_constraint.evaluator().set_description("CBoundingBoxConstraint")
        bound = self.options.correction_bound
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -bound * np.ones(7), bound * np.ones(7), self.correction
        )
        self.correction_bounding_box_constraint.evaluator().set_description("CorrectionBoundingBoxConstraint")



class PandaMugProgramTaskParam(GraspTaskParamMixin, PandaMugProgram):
    '''The Panda grasp with the task folded into the decision variables.

    All of the work is in `GraspTaskParamMixin`; it is robot-agnostic, so the iiwa gets the
    same formulation from the same code.'''


class PandaIKProgramNumerical(PandaIKProgram):
    '''Program for Inverse Kinematics of a end-effector pose'''
    def __init__(self, diagram, options = ProgramOptions(), model = None):
        super().__init__(diagram, options, model)

    def create_prog(self, target_pose = np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal = None):
        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(7)


        self.lumped_vars = self.q

        self.target_pose = target_pose
        if q_nominal is None:
            self.q_nominal = np.zeros(7)
        else:
            self.q_nominal = q_nominal
        self.prog.SetInitialGuess(self.q, self.q_nominal)
        self.add_constraints()
        self.add_costs()


    def VarsToQ(self, rpy_vars, add_correction = False):
        q = np.zeros(self.num_pos, dtype=AutoDiffXd if isinstance(rpy_vars[0], AutoDiffXd) else float)
        q[7:] = [0.04] * (self.num_pos - 7)  # fixed gripper joints
        q[:7] = rpy_vars[:7]
        return q

    def SetStartFromQ(self, q_arm):
        '''The joint-space arm's variables *are* the configuration, so the shared start
        needs no conversion.'''
        return self._SetClipped(self.q, np.asarray(q_arm, dtype=float)[:7])

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
                    -10. * np.ones(7), 10. * np.ones(7), self.q
        )

class PandaIKProgramAnalytic(PandaIKProgram):
    def __init__(self, diagram, options = ProgramOptions(), model = None):
        super().__init__(diagram, options, model)
        self.analytic_ik = Analytic_IK_Panda()

    def create_prog(self, target_pose = np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal = None, pose_offset = None, gc = None):
        self.prog = MathematicalProgram()
        self.xyz_rpy = self.prog.NewContinuousVariables(6) ## x y z roll pitch yaw
        self.psi = self.prog.NewContinuousVariables(1) ## redundancy parameter
        self.lumped_vars = np.hstack([self.xyz_rpy, self.psi])

        self.target_pose = target_pose
        if q_nominal is None:
            self.q_nominal = np.zeros(7)
        else:
            self.q_nominal = q_nominal
        if gc is None:
            opts = np.array([[1,1],[1,2],[2,1],[2,2]])
            self.gc = opts[np.random.randint(len(opts))]
        else:
            self.gc = gc
        self.pose_offset = pose_offset

        self.target_rpy = [*target_pose[:3], *RotationMatrix(Quaternion(target_pose[3:])).ToRollPitchYaw().vector()]

        self.prog.SetInitialGuess(self.xyz_rpy, self.target_rpy)
        self.prog.SetInitialGuess(self.psi, [.5])

        self.add_constraints()
        self.add_costs()

    def SetStartFromQ(self, q_arm):
        '''Recover the analytic parameterisation of `q_arm` by inversion.

        `psi` and the branch `GC` are read straight back out of the configuration, the way
        ../codebase seeds its new formulation with `analytic_ik.psi(q_initial)` and
        `.GC(q_initial)`. Choosing the branch from the seed rather than at random is also
        the honest baseline: a random branch is feasible only some of the time, and
        charging the analytic arm for that is not a statement about the formulation.
        '''
        q_arm = np.asarray(q_arm, dtype=float)[:7]
        self.gc = self.analytic_ik.gc(q_arm)
        self.plant.SetPositions(self.plant_context, self.PadQ(q_arm))
        # self.frame, not FlowFrame(): there is no flow in this formulation, and the
        # variables are the pose of the frame the closed-form map is written against --
        # panda_hand here, between_fingers in the grasp scene.
        pose = self.frame.CalcPoseInWorld(self.plant_context)
        xyz_rpy = np.concatenate([pose.translation(),
                                  pose.rotation().ToRollPitchYaw().vector()])
        clipped = self._SetClipped(self.xyz_rpy, xyz_rpy)
        clipped += self._SetClipped(self.psi, [self.analytic_ik.psi(q_arm)])
        return clipped

    def add_constraints(self):
        if self.options.collision_avoidance:
            self.CreateCollisionFreeConstraint()
        if self.options.joint_limits:
            self.CreateJointLimitsConstraint()
        self.ApplyConstraints()
        self.ReachabilityConstraint()
        self.BoundingBoxConstraint()

    def VarsToQ(self, rpy_vars, add_correction = False):
        xyz_rpy = rpy_vars[:6]
        psi = rpy_vars[6]
        ad = isinstance(xyz_rpy[0], AutoDiffXd)
        dtype = AutoDiffXd if ad else float
        q = np.zeros(self.num_pos, dtype=dtype)
        pose = RigidTransform_[dtype](RollPitchYaw_[dtype](xyz_rpy[3:]), xyz_rpy[:3])
        q[:7] = self.analytic_ik.IK(pose, psi, GC=self.gc, pose_offset=self.pose_offset)
        q[7:] = [0.04] * (self.num_pos - 7)  # fixed gripper joints
        return q


    def BoundingBoxConstraint(self):
        ik_tol = self.options.ik_constraint_tol
        ik_bb = np.array([ik_tol[0], ik_tol[0], ik_tol[0], ik_tol[1], ik_tol[1], ik_tol[1]])
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
                    self.target_rpy - ik_bb, self.target_rpy + ik_bb, self.xyz_rpy
        )
        self.bounding_box_constraint_psi = self.prog.AddBoundingBoxConstraint(
                    -5., 5., self.psi
        )

    def ReachabilityConstraint(self):

        self.reachability_constraint = self.prog.AddConstraint(
            func=lambda vars: self.EvalReachabilityConstraint(vars),
            lb=np.array([-1., -1., -1., -1.]),
            ub=np.array([1., 1., 1., 1.]),
            vars=self.lumped_vars,
            description="ReachabilityConstraint"
        )
        return self.reachability_constraint

    def EvalReachabilityConstraint(self, vars):
        xyz_rpy = vars[:6]
        psi = vars[6]
        ad = isinstance(xyz_rpy[0], AutoDiffXd)
        dtype = AutoDiffXd if ad else float
        pose = RigidTransform_[dtype](RollPitchYaw_[dtype](xyz_rpy[3:]), xyz_rpy[:3])
        q = self.analytic_ik.IK(pose, psi, GC=self.gc, pose_offset=self.pose_offset, return_unclipped_vals=True)
        return q



class PandaMugProgramNumerical(PandaMugProgram):
    '''Program for Inverse Kinematics of a end-effector pose'''
    def __init__(self, diagram, options = ProgramOptions(), model = None):
        super().__init__(diagram, options, model)

    def create_prog(self, target_mug = Mug(), q_nominal = None):
        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(7)


        self.lumped_vars = self.q

        self.target_mug = target_mug
        if q_nominal is None:
            self.q_nominal = np.zeros(7)
        else:
            self.q_nominal = q_nominal
        self.prog.SetInitialGuess(self.q, self.q_nominal)
        self.add_constraints()
        self.add_costs()


    def VarsToQ(self, rpy_vars, add_correction = False):
        q = np.zeros(self.num_pos, dtype=AutoDiffXd if isinstance(rpy_vars[0], AutoDiffXd) else float)
        q[7:] = [0.04] * (self.num_pos - 7)  # fixed gripper joints
        q[:7] = rpy_vars[:7]
        return q

    def SetStartFromQ(self, q_arm):
        return self._SetClipped(self.q, np.asarray(q_arm, dtype=float)[:7])

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
                    -10. * np.ones(7), 10. * np.ones(7), self.q
        )


class PandaMugProgramAnalytic(PandaIKProgramAnalytic):
    def __init__(self, diagram, options = ProgramOptions(), model = None):
        super().__init__(diagram, options, model)
        # `xyz_rpy` is the pose of the *grasp* frame: Analytic_IK_Panda.IK with
        # MUG_ANALYTIC_OFFSET inverts the between_fingers pose back to q (to 5e-3 rad, the
        # residual between the DH parameters and the finray URDF), while panda_hand's pose
        # gives 4.5 rad. Leaving self.frame on panda_hand made the inherited SetStartFromQ
        # write a pose 0.1 m and 90 degrees from the one the variables mean, so the analytic
        # arm began 3-5.6 rad from the shared q_init -- i.e. it was not the paired start.
        self.frame = self.plant.GetFrameByName("between_fingers")
        self.autodiff_frame = self.autodiff_plant.GetFrameByName("between_fingers")
    def create_prog(self, target_mug = Mug(), q_nominal = None, pose_offset = None, gc = None):
        self.prog = MathematicalProgram()
        self.xyz_rpy = self.prog.NewContinuousVariables(6) ## x y z roll pitch yaw
        self.psi = self.prog.NewContinuousVariables(1) ## redundancy parameter
        self.lumped_vars = np.hstack([self.xyz_rpy, self.psi])

        self.target_mug = target_mug
        if q_nominal is None:
            self.q_nominal = np.zeros(7)
        else:
            self.q_nominal = q_nominal
        if gc is None:
            opts = np.array([[1,1],[1,2],[2,1],[2,2]])
            self.gc = opts[np.random.randint(len(opts))]
        else:
            self.gc = gc
        self.pose_offset = pose_offset

        self.prog.SetInitialGuess(self.xyz_rpy, [*target_mug.middle.translation(), 1, 0, 0])
        # self.prog.SetInitialGuess(self.xyz_rpy, self.target_rpy)
        self.prog.SetInitialGuess(self.psi, [.5])

        self.add_constraints()
        self.add_costs()


    def IKConstraint(self):
        # x = y = 0 exactly, the same equality the learned arm is held to. These rows used
        # to be +-ik_constraint_tol[0], which held the analytic arm to a slightly easier
        # problem than the formulation it is being compared against.
        self.ik_constraint = self.prog.AddConstraint(
            func = lambda vars: self.EvalIKMugConstraint(vars),
            lb = np.array([0.0, 0.0, -self.options.mug_height]),
            ub = np.array([0.0, 0.0, self.options.mug_height]),
            vars = self.lumped_vars,
            description = "IKMugConstraint"
        )
    def EvalIKMugConstraint(self, vars):
        xyz_rpy = vars[:6]
        psi = vars[6]
        mug_transform = np.linalg.inv(self.target_mug.middle.GetAsMatrix4())
        # Drop the homogeneous row: it is identically 1 with a zero gradient.
        return (mug_transform @ np.array([[*(xyz_rpy[:3]), 1]]).T).squeeze()[:3]

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
                    -5 * np.ones(6), 5 * np.ones(6), self.xyz_rpy
        )
        self.bounding_box_constraint_psi = self.prog.AddBoundingBoxConstraint(
                    -5., 5., self.psi
        )

    def add_constraints(self):
        if self.options.collision_avoidance:
            self.CreateCollisionFreeConstraint()
        if self.options.joint_limits:
            self.CreateJointLimitsConstraint()
        self.ApplyConstraints()
        self.IKConstraint()
        self.ReachabilityConstraint()
        self.BoundingBoxConstraint()