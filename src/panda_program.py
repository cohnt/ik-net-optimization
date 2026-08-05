from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE
import torch
import numpy as np
from src.utils import Mug
from src.generic_program import *
import numpy as np
from pydrake.all import (
    MathematicalProgram,
    AutoDiffXd,
    Quaternion,
    RotationMatrix, 
    RotationMatrix_,
    RollPitchYaw_,
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

        if model is None:
            model_name = "panda__full__lp191_5.25m"
            self.ik_solver, _ = get_ik_solver(model_name)
            self.ik_solver.nn_model.eval()
        else:
            self.ik_solver = model

        self.options = options
        self.constraints = []



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


        self.jacobian_gen = torch.compile(torch.func.jacrev(self.ik_inference)) ## function that can compute jacobian dq/dvars

        ## Add Constraints
        self.add_constraints()

        self.add_costs()


    def ik_inference(self, vars, add_correction = True):
        '''Given a latent + target + correction, returns corresponding joint angles
        vars can be either numpy array or torch tensor (for gradient computation)'''
        # Convert to tensor only if not already a tensor
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=torch.float32)
        
        c, z, correction = (vars[:7], vars[7:7+self.ik_solver.network_width], vars[7+self.ik_solver.network_width:])
        # Work directly with tensor slices - don't call torch.tensor() again!
        c_torch = torch.cat([c.unsqueeze(0), torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)], dim=1)
        z_batch = z.unsqueeze(0)

        output, _ = self.ik_solver.nn_model(z_batch, c=c_torch, rev=True)
        q = output[:, :7].squeeze(0)
        if add_correction:
            return q + correction
        else: return q
    
        

    def VarsToQ(self, rpy_vars, add_correction = True):


        ad = isinstance(rpy_vars[0], AutoDiffXd)
        vars = np.zeros(21, dtype=AutoDiffXd if ad else np.float32)
        t = AutoDiffXd if ad else float

        vars[:3] = rpy_vars[:3]
        vars[3:7] = RotationMatrix_[t](RollPitchYaw_[t](rpy_vars[3:6])).ToQuaternion().wxyz()
        vars[7:14] = rpy_vars[6:13]
        vars[14:21] = rpy_vars[13:20]


        if not ad:
            q = np.zeros(self.num_pos)
            q[7:] = [0.04] * (self.num_pos - 7)  # fixed gripper joints
            q[:7] = self.ik_inference(vars, add_correction=add_correction).detach().cpu().numpy()
            return q
        
        else: # Compute AutoDiffXd with Jacobian_Gen
            # Extract values and gradients from AutoDiffXd
            vars_values = np.array([v.value() for v in vars])
            vars_gradients = np.array([v.derivatives() for v in vars])
            
            # Compute q values
            q_values = np.zeros(self.num_pos)
            q_values[7:] = [0.04] * (self.num_pos - 7)
            q_values[:7] = self.ik_inference(vars_values, add_correction=add_correction).detach().cpu().numpy()
            
            # Compute Jacobian dq/dvars
            vars_tensor = torch.tensor(vars_values, dtype=torch.float32, device=DEVICE, requires_grad=True)
            jacobian = self.jacobian_gen(vars_tensor)
            jacobian_np = jacobian.detach().cpu().numpy()
            
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
        self.frame = self.frame = self.plant.GetFrameByName("between_fingers")
        self.autodiff_frame = self.autodiff_plant.GetFrameByName("between_fingers")


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

        self.prog.SetInitialGuess(self.c, [*target_mug.middle.translation(), 1, 0, 0])
        self.prog.SetInitialGuess(self.z, np.random.randn(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(7))
        self.jacobian_gen = torch.func.jacrev(self.ik_inference) ##

        self.target_pose = np.array([*target_mug.middle.translation(), 1, 0, 0, 0]) ## for bounding box
        self.add_constraints()
        self.add_costs()
    
    def CreateIKConstraint(self):
        ik_tol, _ = self.options.ik_constraint_tol
        lb = np.array([-ik_tol, -ik_tol, -self.options.mug_height, 1])
        ub = np.array([ik_tol, ik_tol, self.options.mug_height, 1])
        def eval_func(vars, q, pose):
            position, _ = pose
            mug_transform = np.linalg.inv(self.target_mug.middle.GetAsMatrix4())
            return (mug_transform @ np.array([[*position, 1]]).T).squeeze()
        self.ik_constraint = IKFlowConstraints(lb, ub, eval_func, description="IKConstraint")
        self.constraints.append(self.ik_constraint)
        return self.ik_constraint

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -5. * np.ones(self.ik_solver.network_width), 5. * np.ones(self.ik_solver.network_width), self.z
        )
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        self.c_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -5 * np.ones(6), 5 * np.ones(6), self.c
        )
        self.c_bounding_box_constraint.evaluator().set_description("CBoundingBoxConstraint")
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -0.1 * np.ones(7), 0.1 * np.ones(7), self.correction
        )
        self.correction_bounding_box_constraint.evaluator().set_description("CorrectionBoundingBoxConstraint")



class PandaIKProgramNumerical(PandaIKProgram):
    '''Program for Inverse Kinematics of a end-effector pose'''
    def __init__(self, diagram, options = ProgramOptions()):
        super().__init__(diagram, options)

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

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
                    -10. * np.ones(7), 10. * np.ones(7), self.q
        )

class PandaIKProgramAnalytic(PandaIKProgram):
    def __init__(self, diagram, options = ProgramOptions()):
        super().__init__(diagram, options)


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
            opts = np.array([[1,1],[1,2],[2,1],[1,2]])
            self.gc = opts[np.random.randint(len(opts))]
        else:
            self.gc = gc
        self.pose_offset = pose_offset

        self.target_rpy = [*target_pose[:3], *RotationMatrix(Quaternion(target_pose[3:])).ToRollPitchYaw().vector()]

        self.prog.SetInitialGuess(self.xyz_rpy, self.target_rpy)
        self.prog.SetInitialGuess(self.psi, [.5])

        self.add_constraints()
        self.add_costs()

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

