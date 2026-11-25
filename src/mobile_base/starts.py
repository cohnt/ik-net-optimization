"""The mobile-base grid: target draws and the three base-start rules (Thomas, 2026-11-25).

A cell's start is ONE full plant vector `q_init`: its base entries come from the cell's base-start
rule, its arm entries from the uniform draw inside the joint limits every grid already uses, and
the pair is redrawn until it is collision-free (the record's guesses are collision-free draws).
Both arms read the same `q_init`: joint space starts at all of it, the learned arm at its base
(`MobileLearnedMixin.SetNativeStart`), so the base starts at the same place in the same
coordinates in both. Thomas asked for all three rules to be tried:

  uniform  x, y ~ U(-2, 2) (the base box), yaw ~ U(-pi, pi), independent of the target --
           joint space's native rule extended to the base.
  facing   ../codebase's adopted `facing_axis` rule (src/util.py:TargetAwareBaseGuess with
           jitter): heading = the target's approach axis projected on the floor, the base set
           back from the target along it by 1.0 m x U(0.6, 1.4), heading jittered by +-20 deg.
           The approach axis is the gripper's wrist -> fingertip direction expressed in the
           target frame, so it is the same physical axis on every robot and both tasks (on the
           iiwa pose task, where the target is `iiwa_link_7`, it is that frame's +z, exactly
           ../codebase's choice). On the grasp task the mug's frame is the generating grasp
           frame, so this reads the generating approach direction -- the information
           `GraspCStart` already uses, stated rather than hidden.
  home     ../codebase's KMR directive default (1.5, 0, 0), for every cell.

Every draw comes from a generator of the cell's own (`CellRng`), so the targets -- drawn first,
from the grid's stream -- never move with the rule, and a cell's start does not depend on how
many others were drawn before it.
"""

import math

import numpy as np

from src.mobile_base.generate_scenes import HOME_BASE

BASE_START_RULES = ("uniform", "facing", "home")

#: ../codebase src/util.py: the KMR row's stand-off and the jitter modifier.
FACING_REACH = 1.0
JITTER_REACH_FRACTION = (0.6, 1.4)
JITTER_YAW_RADIANS = math.radians(20.0)

#: A bounded redraw: a facing stand-off that puts the platform inside the table, say, admits no
#: collision-free arm at all, so the base is redrawn too (uniform, facing) after this many arms.
ARM_DRAWS_PER_BASE = 50
MAX_START_DRAWS = 20000

_START_STREAM = 0x6d627374   # "mbst"


def CellRng(seed, i, j, rule):
    """Cell (i, j)'s generator under `rule`: four words plus the rule's index, so it never
    coincides with the grid's own stream or another rule's."""
    return np.random.default_rng([int(seed), _START_STREAM, BASE_START_RULES.index(rule),
                                  int(i), int(j)])


class MobileGrid:
    """Draws on one mobile program's plant (`MobileBaseMixin`)."""

    def __init__(self, program, collision_free, wrist_frame, fingertip_frame="between_fingers"):
        self.program = program
        self.collision_free = collision_free
        self.base_slots = program.BaseSlots()
        self.arm_slots = program.ArmSlots()
        self.base_lower, self.base_upper = program.BaseLimits()
        self.arm_lower, self.arm_upper = program.ConfigLimits()
        self.default = np.asarray(program.plant.GetDefaultPositions(), dtype=float)
        self.approach_T = self._ApproachAxis(wrist_frame, fingertip_frame)

    ## ------------------------------- draws ---------------------------------- ##

    def UniformBase(self, rng):
        return np.array([rng.uniform(self.base_lower[0], self.base_upper[0]),
                         rng.uniform(self.base_lower[1], self.base_upper[1]),
                         rng.uniform(-math.pi, math.pi)])

    def Assemble(self, b, q_arm):
        q = self.default.copy()
        q[self.base_slots] = b
        q[self.arm_slots] = q_arm
        return q

    def Draw(self, rng):
        """A uniform full configuration -- the TARGET draw: base uniform in its box (yaw in
        [-pi, pi]), arm uniform in its limits."""
        b = self.UniformBase(rng)
        return self.Assemble(b, rng.uniform(self.arm_lower, self.arm_upper))

    ## ----------------------------- base starts ------------------------------ ##

    def _ApproachAxis(self, wrist, fingertip):
        """The gripper's wrist -> fingertip unit vector in the TARGET frame (`program.frame`).
        Constant -- all three frames ride the last link -- and checked at two configurations."""
        plant, ctx = self.program.plant, self.program.plant_context
        axes = []
        for seed in (0, 1):
            q = self.Draw(np.random.default_rng(seed))
            plant.SetPositions(ctx, q)
            X_WT = self.program.frame.CalcPoseInWorld(ctx)
            p_w = plant.GetFrameByName(wrist).CalcPoseInWorld(ctx).translation()
            p_f = plant.GetFrameByName(fingertip).CalcPoseInWorld(ctx).translation()
            u = (p_f - p_w) / np.linalg.norm(p_f - p_w)
            axes.append(X_WT.rotation().inverse() @ u)
        if np.max(np.abs(axes[0] - axes[1])) > 1e-9:
            raise RuntimeError("the approach axis is not constant in the target frame")
        return axes[0]

    def FacingBase(self, X_WT, rng):
        """../codebase `TargetAwareBaseGuess(target, "facing_axis", jitter=True)`, with the
        approach axis for the frame's +z (module docstring)."""
        d = (X_WT.rotation() @ self.approach_T)[:2]
        mid = np.asarray(X_WT.translation(), dtype=float)[:2]
        if math.hypot(d[0], d[1]) < 1e-9:
            yaw = 0.0
        else:
            yaw = math.atan2(d[1], d[0])
        reach = FACING_REACH * float(rng.uniform(*JITTER_REACH_FRACTION))
        yaw = yaw + float(rng.uniform(-JITTER_YAW_RADIANS, JITTER_YAW_RADIANS))
        b = np.array([mid[0] - reach * math.cos(yaw), mid[1] - reach * math.sin(yaw), yaw])
        ## Inside the base box: a stand-off past +-2 m cannot be a start (the box is a
        ## variable bound and every start must respect it), so it is redrawn by the caller.
        return b

    def _InBox(self, b):
        return bool(np.all(b >= self.base_lower) and np.all(b <= self.base_upper))

    def Start(self, rule, X_WT, rng):
        """One cell's `q_init` under `rule`, plus how many draws it took."""
        if rule not in BASE_START_RULES:
            raise ValueError(f"base start rule {rule!r} is not one of {BASE_START_RULES}")
        draws = 0
        b = None
        while draws < MAX_START_DRAWS:
            if b is None or (rule != "home" and draws % ARM_DRAWS_PER_BASE == 0):
                if rule == "uniform":
                    b = self.UniformBase(rng)
                elif rule == "facing":
                    b = self.FacingBase(X_WT, rng)
                    if not self._InBox(b):
                        draws += 1
                        b = None
                        continue
                else:
                    b = np.array(HOME_BASE, dtype=float)
            q = self.Assemble(b, rng.uniform(self.arm_lower, self.arm_upper))
            draws += 1
            if self.collision_free(q):
                return q, draws
        raise RuntimeError(f"no collision-free start under rule {rule!r} in {MAX_START_DRAWS} "
                           f"draws")
