"""The formulations on a mobile base (approved by Thomas, 2026-11-25; docs/notes-mobile-base.md).

THE BASE.  `b = (x, y, yaw)`, the three joints of `models/mobile_base/kmr_platform.urdf`
(world -> prismatic x -> prismatic y -> revolute yaw -> platform), boxed by the plant's own
limits: x, y in [-2, 2] m, yaw in [-2 pi, 2 pi]. A variable bound in both arms, because every
start is drawn inside it.

LEARNED.  Variables `[b | c | z | q_c]`. `c` keeps its fixed-base meaning -- the WORLD pose of
the frame the flow was trained on -- with its box, its start and the correction unchanged. What
the network is handed is that pose expressed in the arm's base frame `A`, the frame jrl's FK
(and so the training data) uses:

    VarsToQ(b, c, z, q_c) = [ b ; flow( X_WA(b)^-1 X(c), z ) + q_c ]

It is implemented as the robot class's own `VarsToQ` with `num_task_vars = 9` and the
`TaskVarsToPose7` hook overridden, so the AutoDiffXd chain rule is the existing one: the
derivatives of the arm-frame pose with respect to `b` and `c` come out of Drake's templated
`RigidTransform_` and are multiplied into the flow's `jacrev` exactly as `c`'s always were.

Where the chart is exact (`FK(flow(p, z)) = p`, `q_c = 0`) the end effector's world pose is
`X(c)` whatever `b` and `z` are: the base joins the latent as a coordinate ALONG the IK fibre.

JOINT SPACE.  Variables `[b | q_arm]`, identity into the plant slots.

SHARED.  The configuration (`Config`, the joint-limit row, joint centering) is the ARM's
coordinates only: the base has no home, so the base block of the centering cost is zero
(`../codebase`'s rule, `kmriiwa_experiments.py` `centering_cost_mat[:3, :3] = 0`). Collision is
evaluated on the full plant vector, platform included.

STARTS take a FULL plant vector (`q_init[base slots]` is the cell's base start, drawn by the
grid under one of the three base-start rules; `src/mobile_base/starts.py`), so both arms of a
cell start their base at the same place in the same coordinates.

The mixin wraps the record's robot classes rather than editing them: each one fills the arm
into `q[:ndof]` of a plant-sized vector, and the mixin moves it into the arm's real slots and
puts `b` in front. Every slot is read from the plant (`position_start`), never assumed.
"""

from functools import cached_property

import numpy as np
from pydrake.all import (AutoDiffXd, RigidTransform, RigidTransform_, RollPitchYaw,
                         RollPitchYaw_)

from src.mobile_base.generate_scenes import BASE_JOINTS, MOBILE_MODEL, MOUNT_FRAME


class _MobileGeometry:
    """What the mixin reads off a finalized plant, once."""

    def __init__(self, program):
        plant = program.plant
        instance = plant.GetModelInstanceByName(MOBILE_MODEL)
        joints = [plant.GetJointByName(name, instance) for name in BASE_JOINTS]
        self.base_slots = np.array([j.position_start() for j in joints])
        lower, upper = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        self.base_limits = (lower[self.base_slots].copy(), upper[self.base_slots].copy())
        if not (np.all(np.isfinite(self.base_limits[0]))
                and np.all(np.isfinite(self.base_limits[1]))):
            raise RuntimeError(f"the mobile base's limits are not finite: {self.base_limits}")

        ## The arm's base body: the one body welded to the platform's mount frame.
        mount_body = plant.GetBodyByName(MOUNT_FRAME, instance)
        children = [plant.get_joint(ji).child_body()
                    for ji in plant.GetJointIndices()
                    if plant.get_joint(ji).parent_body().index() == mount_body.index()]
        if len(children) != 1:
            raise RuntimeError(f"expected one body welded to {MOUNT_FRAME}, found "
                               f"{[b.scoped_name() for b in children]}")
        self.arm_base_frame = children[0].body_frame()

        ## The arm's coordinates: every non-base position, in plant order, which is the
        ## order the robot class fills `q[:ndof]` in on the fixed-base plant. Checked to be
        ## exactly `num_arm_dof` wide, so a gripper that grew joints would fail here.
        rest = np.setdiff1d(np.arange(plant.num_positions()), self.base_slots)
        self.other_slots = rest
        self.footprint = joints[-1].child_body().body_frame()
        context = plant.CreateDefaultContext()
        ## X_FA: the yaw joint's child frame to the arm base. Constant (welded); read from the
        ## plant, never typed (../codebase F5: a hand-typed mount yaw was 3.7e-6 rad off).
        self.X_FA = plant.CalcRelativeTransform(context, self.footprint, self.arm_base_frame)
        ## The joint chain world -> x -> y -> yaw must be the plain planar map this file
        ## writes in closed form; verified against the plant at a few base poses.
        rng = np.random.default_rng(0)
        for _ in range(3):
            b = rng.uniform(*self.base_limits)
            q = plant.GetPositions(context)
            q[self.base_slots] = b
            plant.SetPositions(context, q)
            X_plant = plant.CalcRelativeTransform(context, plant.world_frame(),
                                                  self.arm_base_frame)
            X_mine = PlanarTransform(b, float) @ self.X_FA
            err = np.max(np.abs(X_plant.GetAsMatrix4() - X_mine.GetAsMatrix4()))
            if err > 1e-12:
                raise RuntimeError(f"the base chain is not the planar map (err {err:.2e})")


def PlanarTransform(b, t):
    """`X_WF(b)`: translation (x, y, 0) and a rotation of `yaw` about +z, in scalar type `t`."""
    zero = b[0] * 0.0 if t is AutoDiffXd else 0.0
    return RigidTransform_[t](RollPitchYaw_[t](zero, zero, b[2]),
                              np.array([b[0], b[1], zero]))


class MobileBaseMixin:
    """List it FIRST in the bases, ahead of the robot's program class."""

    MOBILE = True

    @cached_property
    def _mb(self):
        return _MobileGeometry(self)

    ## ----------------------------- geometry ------------------------------------ ##

    def ArmBaseFrame(self):
        """The frame `A` the flow's training FK is expressed in (`CalibrateFlowFrame`)."""
        return self._mb.arm_base_frame

    def BaseSlots(self):
        return self._mb.base_slots

    def ArmSlots(self):
        return self._mb.other_slots[:self.num_arm_dof]

    def BaseLimits(self):
        return self._mb.base_limits

    def X_WA(self, b, t=float):
        """`X_WA(b) = X_WF(b) X_FA`: the arm base in the world, in scalar type `t`."""
        X_FA = self._mb.X_FA if t is float else self._mb.X_FA.cast[AutoDiffXd]()
        return PlanarTransform(b, t) @ X_FA

    ## ------------------------- configuration vs plant --------------------------- ##

    def ConfigLimits(self):
        lower, upper = self.plant.GetPositionLowerLimits(), self.plant.GetPositionUpperLimits()
        slots = self.ArmSlots()
        return lower[slots], upper[slots]

    def ConfigToPlantQ(self, cfg, b=None):
        """Arm configuration (+ base, default the plant's default) -> a full plant vector."""
        q = np.asarray(self.plant.GetDefaultPositions(), dtype=float).copy()
        q[self.ArmSlots()] = np.asarray(cfg, dtype=float)[:self.num_arm_dof]
        if b is not None:
            q[self.BaseSlots()] = b
        return q

    def PadQ(self, q_arm):
        return self.ConfigToPlantQ(q_arm)

    def CalibrationPlantQ(self, q_arm, rng):
        """`CalibrateFlowFrame`'s plant vector: a random base too, so the constancy check
        also proves the offset does not depend on where the base is."""
        return self.ConfigToPlantQ(q_arm, rng.uniform(*self.BaseLimits()))

    def SplitPlantQ(self, q_full):
        q_full = np.asarray(q_full, dtype=float)
        return q_full[self.BaseSlots()], q_full[self.ArmSlots()]

    def _Assemble(self, b, q_fixed):
        """`b` and the robot class's fixed-base vector (arm in `[:ndof]`) -> plant vector."""
        ad = isinstance(q_fixed[0], AutoDiffXd) or isinstance(b[0], AutoDiffXd)
        q = np.empty(self.num_pos, dtype=AutoDiffXd if ad else float)
        if ad:
            n_vars = len(self.lumped_vars)
            zero = AutoDiffXd(0.0, np.zeros(n_vars))
            q[:] = zero
        else:
            q[:] = np.asarray(self.plant.GetDefaultPositions(), dtype=float)
        q[self.BaseSlots()] = b
        q[self.ArmSlots()] = q_fixed[:self.num_arm_dof]
        return q

    def VarsToConfigAndQ(self, vars):
        q = self.VarsToQ(vars)
        return q[self.ArmSlots()], q

    ## ------------------------------ variables ----------------------------------- ##

    def add_constraints(self):
        """Prepend the base to the variables the robot class just declared, then build."""
        if getattr(self.options, "lift_q", False):
            raise NotImplementedError("lift_q is not defined on a mobile base")
        self.b = self.prog.NewContinuousVariables(3, "b")
        self.lumped_vars = np.hstack([self.b, self.lumped_vars])
        self.prog.SetInitialGuess(self.b, np.asarray(
            self.plant.GetDefaultPositions(), dtype=float)[self.BaseSlots()])
        super().add_constraints()

    def BoundingBoxConstraint(self):
        super().BoundingBoxConstraint()
        lower, upper = self.BaseLimits()
        self.base_bounding_box = self.prog.AddBoundingBoxConstraint(lower, upper, self.b)
        self.base_bounding_box.evaluator().set_description("BaseBoundingBoxConstraint")

    def RobotSettings(self):
        out = super().RobotSettings()
        out["base_limits"] = [[float(v) for v in bound] for bound in self.BaseLimits()]
        out["X_FA"] = self._mb.X_FA.GetAsMatrix34().tolist()
        return out


class MobileLearnedMixin(MobileBaseMixin):
    """The learned formulation on a base (module docstring)."""

    def _InitMobileLearned(self):
        self.num_task_vars = 9

    def TaskVarsToPose7(self, task_vars, t):
        b, c = task_vars[:3], task_vars[3:9]
        X_WC = RigidTransform_[t](RollPitchYaw_[t](c[3:6]), np.asarray(c[:3]))
        X_AC = self.X_WA(b, t).inverse() @ X_WC
        return X_AC.translation(), X_AC.rotation().ToQuaternion().wxyz()

    def VarsToQ(self, vars, add_correction=True):
        q_fixed = super().VarsToQ(vars, add_correction=add_correction)
        return self._Assemble(vars[:3], q_fixed)

    def ConditioningPoseInArmFrame(self, x_lumped):
        """The pose the network is handed at a point, as (xyz, rpy) -- a diagnostic."""
        b, c = np.asarray(x_lumped[:3], float), np.asarray(x_lumped[3:9], float)
        X = self.X_WA(b).inverse() @ RigidTransform(RollPitchYaw(c[3:6]), c[:3])
        return np.concatenate([X.translation(), X.rotation().ToRollPitchYaw().vector()])

    def SetStartFromQ(self, q_full):
        """The paired start on a base: invert at the arm-frame pose, then clip (not fielded)."""
        b, q_arm = self.SplitPlantQ(q_full)
        self.plant.SetPositions(self.plant_context, np.asarray(q_full, dtype=float))
        pose = self.FlowPoseInWorld()
        c = np.concatenate([pose.translation(), pose.rotation().ToRollPitchYaw().vector()])
        X_AC = self.X_WA(b).inverse() @ pose
        c_A = np.concatenate([X_AC.translation(), X_AC.rotation().ToRollPitchYaw().vector()])
        z = self.InvertFlow(q_arm, c_A)
        c_lo, c_hi = self.c_box
        z_lo, z_hi = self.z_box
        clipped = (float(np.linalg.norm(np.clip(c, c_lo, c_hi) - c))
                   + float(np.linalg.norm(np.clip(z, z_lo, z_hi) - z)))
        self.prog.SetInitialGuess(self.b, b)
        self.prog.SetInitialGuess(self.c, c)
        self.prog.SetInitialGuess(self.z, z)
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        residual = q_arm - np.asarray(
            self.Config(self.prog.GetInitialGuess(self.lumped_vars)), dtype=float)
        bound = self.options.correction_bound
        residual = np.nan_to_num(residual, nan=0.0, posinf=bound, neginf=-bound)
        self.prog.SetInitialGuess(self.correction, np.clip(residual, -bound, bound))
        return clipped

    def SetNativeStart(self, q_init, rng):
        """The base at the cell's base start, shared with joint space; everything else is the
        fixed-base native start (c at the target, z from the prior, no correction)."""
        b, _ = self.SplitPlantQ(q_init)
        self.prog.SetInitialGuess(self.b, b)
        return super().SetNativeStart(q_init, rng)


class MobileJointMixin(MobileBaseMixin):
    """The joint-space formulation on a base: `[b | q_arm]`."""

    def VarsToQ(self, vars, add_correction=False):
        q_fixed = super().VarsToQ(vars[3:], add_correction=add_correction)
        return self._Assemble(vars[:3], q_fixed)

    def SetStartFromQ(self, q_full):
        b, q_arm = self.SplitPlantQ(q_full)
        return self._SetClipped(self.b, b) + self._SetClipped(self.q, q_arm)

    def SetNativeStart(self, q_init, rng):
        return self.SetStartFromQ(q_init)
