"""The GVS push-rod arm's four programs: learned and joint-space, pose and grasp.

The soft PCS arm's programs (`src/soft_arm_program.py`) with ONE thing swapped: the forward
model. There the configuration is strain and the map is a closed-form exponential in torch;
here the configuration is the nine rod forces and the map is SoRoMoX solved to static
equilibrium (`src/gvs_arm/model.py`), differentiated implicitly. Everything the PCS programs
do with a map -- the free conditioning pose `c`, the latent `z`, the correction `q_c`, the
task imposed as constraint rows through `FK(q)`, the joint-space arm deciding over the
configuration directly, the two-Jacobian chain -- is inherited unchanged, which is what keeps
this a statement about the change of variables and the forward model, not about a new
formulation.

The chain is the same shape as the PCS arm's:

    d(plant q)/d(vars) = dP/dcfg  @  dflow/dvars

with the flow's `jacrev` exactly as every robot computes it, and `dP/dcfg` now the implicit
Jacobian `d(poses)/d(q*) . dq*/du . F_max` from the model -- forward mode through a root-find,
9 inputs against 259 outputs. Measured: ~14 ms per map evaluation and ~18 ms per Jacobian on
one CPU core, beside the flow's ~17 ms; that is this robot's per-iteration price, to be
reported and not attacked.

`fk="learned"` keeps the PCS arm's learned-forward-model hook (a surrogate over the same
`cfg -> plant q` layout); the fit is a cluster job and is not part of this build.
"""

import numpy as np

import src.register_robots  # noqa: F401  -- registers every project robot with jrl.robots
from src.flow_loading import LoadFlowSolver
from src.generic_program import ProgramOptions
from src.gvs_arm.model import GetModel, PlantSlotMap
from src.gvs_arm.params import GetSpec, PRIMARY
from src.soft_arm_program import SoftArmIKProgram, SoftArmMugProgram, _NumericalMixin
from src.utils import Mug

_CPU = "cpu"


class GvsArmIKProgram(SoftArmIKProgram):
    """Learned formulation, pose task: reach a 6-D pose with the tip frame."""

    def __init__(self, diagram, options=ProgramOptions(), rung=PRIMARY, model=None,
                 checkpoint=None, fk="analytic", surrogate=None):
        ## Deliberately NOT `super().__init__`: the PCS constructor resolves a PCS spec and
        ## a torch map by rung name. Everything it sets up is set up here, in the same order.
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
        self.num_arm_dof = self.spec.ninputs
        self.num_task_vars = 6

        if self.num_pos != self.spec.num_plant_positions:
            raise RuntimeError(
                f"the scene's plant has {self.num_pos} positions but {self.spec.name} "
                f"declares {self.spec.num_plant_positions}; the model and the spec have "
                f"drifted -- rerun src/gvs_arm/generate_sdf.py")

        if model is not None:
            self.ik_solver = model
        elif checkpoint is not None:
            self.ik_solver = LoadFlowSolver(self.spec.name, checkpoint)
        else:
            raise ValueError(
                f"{self.spec.name} has no pretrained chart to fall back on: pass a "
                f"`checkpoint` or a `model`.")

        self.model = GetModel(self.spec)
        self._plant_slots = self._BuildPlantSlotMap()
        self.fk_mode = fk
        self.fk_surrogate = None
        if fk == "learned":
            from src.soft_arm import fk_surrogate as FS
            if surrogate is None:
                raise ValueError(
                    f"{self.spec.name}: pass the fitted surrogate explicitly; no surrogate "
                    f"has been fitted for this robot yet (it is a cluster job).")
            self.fk_surrogate_metrics = getattr(surrogate, "screen_metrics", {})
            self.fk_surrogate = surrogate
            tail_mm = float(self.fk_surrogate_metrics.get("tip_mm/max", 10.0))
            self.flow_frame_tol = max(1e-6, 10.0 * tail_mm / 1000.0)
            forward = FS.MakeLearnedConfigToPlantQ(self.fk_surrogate)
            import torch
            jacobian = torch.func.jacfwd(forward)

            def config_jacobian(cfg):
                return (jacobian(cfg).detach().cpu().numpy(),
                        forward(cfg).detach().cpu().numpy())

            self._config_jacobian_np = None
            self._config_jacobian_torch = config_jacobian
        elif fk != "analytic":
            raise ValueError(f"unknown fk backend {fk!r}; expected 'analytic' or 'learned'")
        self.options = options
        self.ConfigureNetworkDtype()
        self.constraints = []
        ## The model's jit is the analogue of `--compile` on the torch maps: always on,
        ## because JAX has no eager fast path worth comparing against. The switch still
        ## governs the flow's Jacobian exactly as on every other robot.
        self.CalibrateFlowFrame()

    ## --------------------------- configuration vs plant --------------------------- ##

    def _BuildPlantSlotMap(self):
        return PlantSlotMap(self.plant, self.spec)

    def ConfigToPlantQ(self, cfg):
        """The forward model, in float: SoRoMoX at equilibrium, or the surrogate."""
        cfg = np.asarray(cfg, dtype=float)
        if self.fk_surrogate is not None:
            import torch
            canonical = self.fk_surrogate(
                torch.as_tensor(cfg, dtype=torch.float64, device=_CPU)).detach().cpu().numpy()
        else:
            canonical = self.model.PlantQ(cfg)
        return canonical[self._plant_slots]

    def ExactConfigToPlantQ(self, cfg):
        """The EXACT forward model, whatever this program is optimising against."""
        return self.model.PlantQ(np.asarray(cfg, dtype=float))[self._plant_slots]

    def MapJacobian(self, cfg):
        """`(d(plant q)/d(cfg), plant q)` in the CANONICAL body order."""
        if self.fk_surrogate is not None:
            import torch
            return self._config_jacobian_torch(
                torch.as_tensor(cfg, dtype=torch.float64, device=_CPU))
        return self.model.PlantJacobian(np.asarray(cfg, dtype=float))

    ## The PCS `VarsToConfigAndQ` calls `self.config_jacobian(torch tensor)`; give it a
    ## callable with that contract so the inherited chain runs unchanged.
    @property
    def config_jacobian(self):
        def jacobian_of(cfg_tensor):
            import torch
            jacobian, value = self.MapJacobian(cfg_tensor.detach().cpu().numpy())
            return (torch.as_tensor(jacobian, dtype=torch.float64, device=_CPU),
                    torch.as_tensor(value, dtype=torch.float64, device=_CPU))
        return jacobian_of


class GvsArmMugProgram(GvsArmIKProgram, SoftArmMugProgram):
    """Learned formulation, grasp task: the gripper on the mug's axis, orientation free."""

    def __init__(self, diagram, options=ProgramOptions(), rung=PRIMARY, model=None,
                 checkpoint=None, fk="analytic", surrogate=None):
        GvsArmIKProgram.__init__(self, diagram, options, rung, model, checkpoint, fk=fk,
                                 surrogate=surrogate)
        ## The flow conditions on the arm's tip; the grasp acts between the fingers. Same
        ## recalibration as the PCS grasp program.
        self.ee_frame = self.frame
        self.frame = self.plant.GetFrameByName("between_fingers")
        self.autodiff_frame = self.autodiff_plant.GetFrameByName("between_fingers")
        self.CalibrateFlowFrame()
        self.plant.SetPositions(self.plant_context,
                                self.ConfigToPlantQ(np.zeros(self.num_arm_dof)))
        X_W_flow = self.FlowPoseInWorld()
        X_W_grasp = self.frame.CalcPoseInWorld(self.plant_context)
        self.X_grasp_ee = X_W_grasp.inverse() @ X_W_flow

    ## `create_prog`, `CreateIKConstraint` and `BoundingBoxConstraint` come from
    ## `SoftArmMugProgram` through the MRO; `ConfigToPlantQ` and friends from the GVS class.


class GvsArmIKProgramNumerical(_NumericalMixin, GvsArmIKProgram):
    def create_prog(self, target_pose=np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal=None):
        from pydrake.all import MathematicalProgram
        self.prog = MathematicalProgram()
        self._CreateVariables()
        self.target_pose = target_pose
        self.q_nominal = (np.zeros(self.num_arm_dof) if q_nominal is None
                          else np.asarray(q_nominal, dtype=float))
        self.prog.SetInitialGuess(self.q, self.q_nominal)
        self.add_constraints()
        self.add_costs()


class GvsArmMugProgramNumerical(_NumericalMixin, GvsArmMugProgram):
    def create_prog(self, target_mug=Mug(), q_nominal=None):
        from pydrake.all import MathematicalProgram
        self.prog = MathematicalProgram()
        self._CreateVariables()
        self.target_mug = target_mug
        self.target_pose = np.array([*target_mug.middle.translation(), 1, 0, 0, 0])
        self.q_nominal = (np.zeros(self.num_arm_dof) if q_nominal is None
                          else np.asarray(q_nominal, dtype=float))
        self.prog.SetInitialGuess(self.q, self.q_nominal)
        self.add_constraints()
        self.add_costs()


