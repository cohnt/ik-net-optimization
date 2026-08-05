from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
from jrl.robots import get_robot
from ikflow.config import DEVICE
import torch
import numpy as np
from src.utils import Mug, RepoDir
from src.generic_program import *
import numpy as np
from pydrake.all import (
    MathematicalProgram,
    AutoDiffXd, 
    RigidTransform_,
    Quaternion,
    RotationMatrix,
    RotationMatrix_,
    RollPitchYaw_,
)
import os
import pydrake.math

from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper
    



class Iiwa14IKProgram(IKFlowProgram):
    def __init__(self, diagram, options = ProgramOptions(), model_instance = None):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context) 

        if model_instance is None:
            self.frame = self.plant.GetBodyByName("iiwa_link_7").body_frame()
            self.autodiff_frame = self.autodiff_plant.GetBodyByName("iiwa_link_7").body_frame()
        else:
            model_instance_name = self.plant.GetModelInstanceName(model_instance)
            autodiff_model_instance = self.autodiff_plant.GetModelInstanceByName(model_instance_name)
            self.frame = self.plant.GetBodyByName("iiwa_link_7", model_instance).body_frame()
            self.autodiff_frame = self.autodiff_plant.GetBodyByName("iiwa_link_7", autodiff_model_instance).body_frame()
        self.num_pos = self.plant.num_positions()

        hparams = {'nb_nodes': 12, 
        'dim_latent_space': 8, 
        'coeff_fn_config': 3, 
        'coeff_fn_internal_size': 1024, 
        'rnvp_clamp': 2.5, 
        'robot_name': 'iiwa14'}

        robot = get_robot(hparams['robot_name'])
        hyper_parameters = IkflowModelParameters()
        hyper_parameters.__dict__.update(hparams)
        self.ik_solver = IKFlowSolver(hyper_parameters, robot, compile_model=None)
        print(RepoDir())
        self.ik_solver.load_state_dict(os.path.join(RepoDir(), "models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl"))
        self.ik_solver.nn_model.eval()
        
        self.options = options
        self.constraints = []

    def create_prog(self, target_pose = np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal = None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6)
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width)
        self.correction = self.prog.NewContinuousVariables(7)

        self.lumped_vars = np.hstack([self.c, self.z, self.correction])

        self.target_pose = target_pose
        if q_nominal is None:
            self.q_nominal = np.zeros(self.num_pos)
        else:
            self.q_nominal = q_nominal

        self.initial_guess = np.zeros(6)
        self.initial_guess[:3] = self.target_pose[:3]
        self.initial_guess[3:] = RotationMatrix(Quaternion(self.target_pose[3:])).ToRollPitchYaw().vector()



        self.prog.SetInitialGuess(self.c, self.initial_guess)
        self.prog.SetInitialGuess(self.z, np.zeros(8))
        self.prog.SetInitialGuess(self.correction, np.zeros(7))

        self.jacobian_gen = torch.compile(torch.func.jacrev(self.ik_inference))
        # self.rev_jac_gen = torch.func.jacrev(self.reverse_inference)

        self.add_constraints()
        self.add_costs()

    def ik_inference(self, vars, add_correction = True):
        '''Given a latent + target + correction, returns corresponding joint angles
        vars can be either numpy array or torch tensor (for gradient computation)
        '''
        # Convert to tensor only if not already a tensor
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=torch.float32)
        
        c, z, correction = (vars[:7], vars[7:7+self.ik_solver.network_width], vars[7+self.ik_solver.network_width:])


        c_torch = torch.cat([c.unsqueeze(0), torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)], dim=1)

        # print(z_batch, c_torch)
        output, _ = self.ik_solver.nn_model(z.unsqueeze(0), c=c_torch, rev=True)

        q = output[0, :7]
        if add_correction:
            return q + correction
        else: return q

    def reverse_inference(self, vars, pad = 0.0):
        '''vars := [q + pose]
        run reverse inference to find associated z value'''

        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=torch.float32)
        
        q = vars[:7]
        pose = vars[7:]
        c_torch = torch.cat([pose, torch.tensor([0.0], dtype=torch.float32, device=DEVICE)]).unsqueeze(0)

        q_pad = torch.cat([q, torch.tensor([pad], dtype=torch.float32, device=DEVICE)]).unsqueeze(0)  # [1, 8]
        z_out, _ = self.ik_solver.nn_model(q_pad, c=c_torch, rev=False)
        return z_out



    def VarsToQ(self, rpy_vars, add_correction = True):
        ad = isinstance(rpy_vars[0], AutoDiffXd)
        vars = np.zeros(22, dtype=AutoDiffXd if ad else np.float32)
        t = AutoDiffXd if ad else float
        vars[:3] = rpy_vars[:3]
        vars[3:7] = RotationMatrix_[t](RollPitchYaw_[t](rpy_vars[3:6])).ToQuaternion().wxyz()
        vars[7:15] = rpy_vars[6:14]
        vars[15:22] = rpy_vars[14:21]

        if not ad:
            q = np.zeros(self.num_pos)
            q[7:] = [0.04] * (self.num_pos - 7)  # fixed gripper joints
            q[:7] = self.ik_inference(vars, add_correction=add_correction).detach().cpu().numpy()
            return q
        else:
            vars_values = np.array([v.value() for v in vars])
            vars_gradients = np.array([v.derivatives() for v in vars])
            
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
            
            # print(sum(q_ad**2))
            return q_ad


class IiwaMugProgram(Iiwa14IKProgram):
    def __init__(self, diagram, options = ProgramOptions()):
        super().__init__(diagram, options)
        self.frame = self.frame = self.plant.GetFrameByName("between_fingers")
        self.autodiff_frame = self.autodiff_plant.GetFrameByName("between_fingers")

    def create_prog(self, target_mug = Mug(), q_nominal = None):
        
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(6) # x y z roll pitch yaw
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width) # latent variables
        self.correction = self.prog.NewContinuousVariables(7) ## small correction term to q

        self.lumped_vars = np.hstack([self.c, self.z, self.correction])

        self.target_mug = target_mug
        if q_nominal is None:
            self.q_nominal = np.zeros(self.num_pos)
        else:
            self.q_nominal = q_nominal


        self.prog.SetInitialGuess(self.c, [*target_mug.middle.translation(), 0, 0, 0])
        self.prog.SetInitialGuess(self.z, np.random.randn(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(7))
        self.jacobian_gen = torch.compile(torch.func.jacrev(self.ik_inference)) ## function that can compute jacobian dq/dvars

        self.target_pose = np.array([*target_mug.middle.translation(), 0, 0, 0]) ## for bounding box
        self.add_constraints()
        self.add_costs()



       
    def CreateIKConstraint(self):
        ik_tol, _ = self.options.ik_constraint_tol
        lb = np.array([-ik_tol, -ik_tol, -self.target_mug.height, 1])
        ub = np.array([ik_tol, ik_tol, self.target_mug.height, 1])
        def eval_func(vars, q, pose):
            position, _ = pose
            mug_transform = np.linalg.inv(self.target_mug.middle.GetAsMatrix4())
            return (mug_transform @ np.array([[*position, 1]]).T).squeeze()
        self.ik_constraint = IKFlowConstraints(lb, ub, eval_func, description="IKConstraint")
        self.constraints.append(self.ik_constraint)
        return self.ik_constraint
