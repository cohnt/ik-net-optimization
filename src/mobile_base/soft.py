"""The soft PCS arm on the mobile base.

Its sub-links are quaternion FLOATING bodies whose poses the strain map writes into the plant
(`src/soft_arm/kinematics.py`), so its base cannot ride the platform through a weld the way the
rigid arms' do. The strain map gives every sub-link's pose in the arm base frame `A` (on the
fixed-base scene `A` is the world, which is why that scene welds the base at the origin); here
each is composed with `X_WA(b)` before it is written:

    plant q = [ b ; X_WA(b) * sublink_pose_A(strain) for every sub-link ]

in torch, forward mode, exactly as the fixed-base map is differentiated -- with 15 inputs
(`[b | strain]`) instead of 12. The rest is the approved formulation unchanged: the learned arm
decides `[b | c | z | q_c]` and the flow is conditioned on `X_WA(b)^-1 X(c)`; joint space
decides `[b | strain]`; the configuration (limits, centering) is the strain alone.

The GRID is in configuration space on this robot, as on the fixed base: a target or a start is
the vector `[b | strain]`, mapped to the plant by `GridToPlantQ`.
"""

from functools import cached_property

import numpy as np
import torch
from pydrake.all import AutoDiffXd

from src.mobile_base.program import MobileJointMixin, MobileLearnedMixin
from src.screw_arm.kinematics import quat_multiply, quat_rotate
from src.soft_arm import kinematics as K
from src.soft_arm_program import (SoftArmIKProgram, SoftArmIKProgramNumerical,
                                  SoftArmMugProgram, SoftArmMugProgramNumerical)

_CPU = "cpu"


def MakeMobileConfigToPlant(spec, X_FA, dtype=torch.float64, device=_CPU):
    """`[b | strain] -> the arm's floating positions in the WORLD`, canonical body order."""
    q_FA = torch.tensor(X_FA.rotation().ToQuaternion().wxyz(), dtype=dtype, device=device)
    p_FA = torch.tensor(X_FA.translation(), dtype=dtype, device=device)

    def mobile_config_to_plant(bc):
        b, cfg = bc[..., :3], bc[..., 3:]
        canon = K.config_to_plant_q(cfg, spec, dtype=dtype, device=device)
        canon = canon.reshape(*canon.shape[:-1], -1, 7)
        q_A, t_A = canon[..., :4], canon[..., 4:]
        half = 0.5 * b[..., 2]
        zero = torch.zeros_like(half)
        q_z = torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)
        q_WA = quat_multiply(q_z, q_FA.expand_as(q_z))
        p_WA = torch.stack((b[..., 0], b[..., 1], zero), dim=-1) + quat_rotate(
            q_z, p_FA.expand(*q_z.shape[:-1], 3))
        q_W = quat_multiply(q_WA.unsqueeze(-2).expand_as(q_A), q_A)
        t_W = quat_rotate(q_WA.unsqueeze(-2).expand_as(q_A), t_A) + p_WA.unsqueeze(-2)
        out = torch.cat([q_W, t_W], dim=-1)
        return out.reshape(*out.shape[:-2], -1)

    return mobile_config_to_plant


class MobileSoftMixin:
    """List FIRST, ahead of the mobile learned/joint mixin and the soft program class."""

    EXTRA_PLANT_POSITIONS = 3

    def _BuildPlantSlotMap(self):
        """`slots` with `plant_q[slots] = canonical_q`: each body's 7 positions, read from the
        plant (the slot order is not the declaration order -- the soft arm's own trap)."""
        slots = np.empty(7 * len(self.spec.body_names()), dtype=np.int64)
        for index, name in enumerate(self.spec.body_names()):
            start = self.plant.GetBodyByName(name).floating_positions_start()
            if start < 0:
                raise RuntimeError(f"{name} is not a floating body in this plant")
            slots[7 * index:7 * index + 7] = np.arange(start, start + 7)
        covered = np.zeros(self.plant.num_positions(), dtype=bool)
        covered[slots] = True
        covered[self.BaseSlots()] = True
        if not covered.all():
            raise RuntimeError("the mobile soft scene has positions neither the base nor the "
                               "strain map drives")
        self._soft_slots = slots
        return slots

    def ConfigLimits(self):
        ones = np.ones(self.num_arm_dof)
        return (-ones, ones)

    ## ------------------------------- the map ----------------------------------- ##

    @cached_property
    def _mobile_map(self):
        return MakeMobileConfigToPlant(self.spec, self._mb.X_FA)

    @cached_property
    def _mobile_map_jacobian(self):
        fn = self._mobile_map
        jac = torch.func.jacfwd(fn)
        if bool(getattr(self.options, "compile_flow_jacobian", False)):
            jac, fn = torch.compile(jac), torch.compile(fn)
        return lambda bc: (jac(bc), fn(bc))

    def _Plant(self, b, floating):
        q = np.asarray(self.plant.GetDefaultPositions(), dtype=float).copy()
        q[self.BaseSlots()] = b
        q[self._soft_slots] = floating
        return q

    def ConfigToPlantQ(self, cfg, b=None):
        if getattr(self, "fk_surrogate", None) is not None:
            raise NotImplementedError("the learned forward model is closed on the mobile base")
        if b is None:
            b = np.asarray(self.plant.GetDefaultPositions(), dtype=float)[self.BaseSlots()]
        bc = torch.as_tensor(np.concatenate([np.asarray(b, float),
                                             np.asarray(cfg, float)[:self.num_arm_dof]]),
                             dtype=torch.float64, device=_CPU)
        return self._Plant(b, self._mobile_map(bc).detach().cpu().numpy())

    ExactConfigToPlantQ = ConfigToPlantQ

    def PadQ(self, cfg):
        return self.ConfigToPlantQ(cfg)

    def CalibrationPlantQ(self, q_arm, rng):
        return self.ConfigToPlantQ(q_arm, rng.uniform(*self.BaseLimits()))

    def TorchConfigToPlantQ(self, dtype=torch.float64, device=_CPU):
        raise NotImplementedError("the batched mobile soft map is built by BatchedProgram")

    ## ------------------------------ grid vectors -------------------------------- ##

    def SplitPlantQ(self, v):
        v = np.asarray(v, dtype=float)
        return v[:3], v[3:3 + self.num_arm_dof]

    def GridVector(self, b, cfg):
        return np.concatenate([np.asarray(b, float), np.asarray(cfg, float)])

    def GridToPlantQ(self, v):
        b, cfg = self.SplitPlantQ(v)
        return self.ConfigToPlantQ(cfg, b)

    ## ---------------------------- vars -> (cfg, q) ------------------------------ ##

    def _ComposeAD(self, b_ad, cfg_vals, cfg_grads):
        """AutoDiffXd plant vector from AutoDiffXd base entries and the strain's value and
        gradient rows, through the forward-mode Jacobian of the composed map."""
        b_vals = np.array([v.value() for v in b_ad])
        b_grads = np.array([v.derivatives() for v in b_ad])
        bc = torch.tensor(np.concatenate([b_vals, cfg_vals]), dtype=torch.float64,
                          device=_CPU)
        J, floating = self._mobile_map_jacobian(bc)
        J = J.detach().cpu().numpy()
        floating = floating.detach().cpu().numpy()
        grads = J @ np.vstack([b_grads, cfg_grads])
        n_vars = b_grads.shape[1]
        q = np.empty(self.num_pos, dtype=AutoDiffXd)
        default = np.asarray(self.plant.GetDefaultPositions(), dtype=float)
        for i in range(self.num_pos):
            q[i] = AutoDiffXd(default[i], np.zeros(n_vars))
        q[self.BaseSlots()] = b_ad
        for k, slot in enumerate(self._soft_slots):
            q[slot] = AutoDiffXd(floating[k], grads[k])
        return q

    def VarsToConfigAndQ(self, vars, add_correction=True):
        autodiff = isinstance(vars[0], AutoDiffXd)
        b = vars[:3]
        if hasattr(self, "z"):
            flow_vars = self._FlowVars(vars, autodiff)
            if not autodiff:
                cfg = self.ik_inference(flow_vars, add_correction=add_correction)
                cfg = cfg.detach().cpu().numpy()
                self._last_returned_cfg = cfg
                return cfg, self.ConfigToPlantQ(cfg, np.asarray(b, float))
            values = np.array([v.value() for v in flow_vars])
            gradients = np.array([v.derivatives() for v in flow_vars])
            tensor = torch.tensor(values, dtype=self.torch_dtype, device=self._FlowDevice())
            flow_jacobian, cfg_tensor = self.jacobian_gen(tensor)
            cfg = cfg_tensor.detach().cpu().numpy().astype(float)
            cfg_grads = flow_jacobian.detach().cpu().numpy() @ gradients
        else:
            cfg_ad = np.asarray(vars[3:3 + self.num_arm_dof])
            if not autodiff:
                cfg = np.asarray(cfg_ad, dtype=float)
                self._last_returned_cfg = cfg
                return cfg, self.ConfigToPlantQ(cfg, np.asarray(b, float))
            cfg = np.array([v.value() for v in cfg_ad])
            cfg_grads = np.array([v.derivatives() for v in cfg_ad])
        self._last_returned_cfg = cfg
        cfg_out = np.array([AutoDiffXd(cfg[i], cfg_grads[i]) for i in range(len(cfg))])
        return cfg_out, self._ComposeAD(b, cfg, cfg_grads)

    def _FlowDevice(self):
        from ikflow.config import DEVICE
        return DEVICE

    def VarsToQ(self, vars, add_correction=True):
        return self.VarsToConfigAndQ(vars, add_correction=add_correction)[1]


def _Learned(base):
    class Mobile(MobileSoftMixin, MobileLearnedMixin, base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._InitMobileLearned()
    Mobile.__name__ = Mobile.__qualname__ = "Mobile" + base.__name__
    return Mobile


def _Joint(base):
    class Mobile(MobileSoftMixin, MobileJointMixin, base):
        pass
    Mobile.__name__ = Mobile.__qualname__ = "Mobile" + base.__name__
    return Mobile


MOBILE_SOFT_CLASSES = {
    ("soft", "pose", "learned"): _Learned(SoftArmIKProgram),
    ("soft", "mug", "learned"): _Learned(SoftArmMugProgram),
    ("soft", "pose", "numerical"): _Joint(SoftArmIKProgramNumerical),
    ("soft", "mug", "numerical"): _Joint(SoftArmMugProgramNumerical),
}
