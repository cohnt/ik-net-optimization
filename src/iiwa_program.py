from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
from jrl.robots import get_robot
from ikflow.config import DEVICE
import torch
import numpy as np
from src.utils import Mug
from src.generic_program import *
import numpy as np
from pydrake.all import (
    MathematicalProgram,
    AutoDiffXd, 
    RigidTransform_,
)



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
        self.ik_solver.load_state_dict("iiwa14__lemon-haze-7__global_step_4.25M.pkl")
        self.ik_solver.nn_model.eval()
        
        self.options = options
        self.constraints = []

    def create_prog(self, target_pose = np.zeros(7), q_nominal = None):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(7)
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width)
        self.correction = self.correction = self.prog.NewContinuousVariables(7)

        self.lumped_vars = np.hstack([self.c, self.z, self.correction])

        self.target_pose = target_pose
        if q_nominal is None:
            self.q_nominal = np.zeros(self.num_pos)
        else:
            self.q_nominal = q_nominal

        self.prog.SetInitialGuess(self.c, target_pose)
        self.prog.SetInitialGuess(self.z, np.random.randn(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(7))

        self.jacobian_gen = torch.func.jacrev(self.ik_inference)
        self.rev_jac_gen = torch.func.jacrev(self.reverse_inference)

        self.add_constraints()
        self.add_costs()

    def ik_inference(self, vars, add_correction = False):
        '''Given a latent + target + correction, returns corresponding joint angles
        vars can be either numpy array or torch tensor (for gradient computation)'''
        # Convert to tensor only if not already a tensor
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=torch.float32)
        
        c, z, correction = (vars[:7], vars[7:7+self.ik_solver.network_width], vars[7+self.ik_solver.network_width:])
        # Work directly with tensor slices - don't call torch.tensor() again!

        c_torch = torch.cat([c.unsqueeze(0), torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)], dim=1)

        z_batch = z.unsqueeze(0)

        # print(z_batch, c_torch)
        output, extra = self.ik_solver.nn_model(z_batch, c=c_torch, rev=True)
        # print(output, extra)
        # Keep this as a torch tensor so correction addition is elementwise and type-safe.
        q = output[0, :7]
        if add_correction:
            return q + correction
        else: return q

    def reverse_inference(self, vars):
        '''vars := [q + pose]
        run reverse inference to find associated z value'''
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=torch.float32)
        
        q = vars[:7]
        pose = vars[7:]
        c_torch = torch.cat([pose, torch.tensor([0.0], dtype=torch.float32, device=DEVICE)]).unsqueeze(0)

        q_pad = torch.cat([q, torch.tensor([0.0], dtype=torch.float32, device=DEVICE)]).unsqueeze(0)  # [1, 8]
        z_out, _ = self.ik_solver.nn_model(q_pad, c=c_torch, rev=False)
        return z_out



    def VarsToQ(self, vars, add_correction = True):
        ad = isinstance(vars[0], AutoDiffXd)

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
            q_gradients = np.zeros((self.num_pos, len(vars)))
            q_gradients[:7, :] = jacobian_np @ vars_gradients
            
            # Create AutoDiffXd objects with value and gradient
            q_ad = np.array([AutoDiffXd(q_values[i], q_gradients[i]) for i in range(len(q_values))])
            
            # print(sum(q_ad**2))
            return q_ad

    