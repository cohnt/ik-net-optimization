"""The four `cusp7` programs: learned and joint space, pose task and grasp task.

The cuspidal arm is the same kind of robot as the screw arm as far as the program is
concerned -- a rigid serial chain whose configuration IS the plant's position vector, a
gripper with no joints, a flow conditioned on the last link -- so these are the screw arm's
classes (`src/screw_arm_program.py`) with the spec looked up in `src.cusp_arm.params`. The
screw-limit repair those classes run is a no-op here (no joint is a screw), and
`RequireFiniteLimits` still asserts every limit the parser read.

There is no analytic column (Tommy, 2026-11-25): on a cuspidal arm no branch label is
global, so an analytic formulation's branch tracking cannot be trusted. Arms are
`learned,numerical`.
"""

import numpy as np

from src.cusp_arm.params import PRIMARY, GetSpec
from src.generic_program import ProgramOptions
from src.screw_arm_program import (ScrewArmIKProgram, ScrewArmIKProgramNumerical,
                                   ScrewArmMugProgram, ScrewArmMugProgramNumerical)
from src.utils import Mug  # noqa: F401  -- re-exported for the benchmark


class CuspArmIKProgram(ScrewArmIKProgram):
    """Learned formulation, pose task."""
    ResolveSpec = staticmethod(GetSpec)

    def __init__(self, diagram, options=ProgramOptions(), robot=PRIMARY, model=None,
                 checkpoint=None):
        super().__init__(diagram, options, robot, model, checkpoint)


class CuspArmMugProgram(ScrewArmMugProgram):
    """Learned formulation, grasp task. The legacy mug-centred `c` start is the screw/iiwa
    control's; this robot postdates it, and `legacy_robot_settings` is a control only."""
    ResolveSpec = staticmethod(GetSpec)

    def __init__(self, diagram, options=ProgramOptions(), robot=PRIMARY, model=None,
                 checkpoint=None):
        super().__init__(diagram, options, robot, model, checkpoint)


class CuspArmIKProgramNumerical(ScrewArmIKProgramNumerical):
    """Joint-space formulation, pose task."""
    ResolveSpec = staticmethod(GetSpec)

    def __init__(self, diagram, options=ProgramOptions(), robot=PRIMARY, model=None,
                 checkpoint=None):
        super().__init__(diagram, options, robot, model, checkpoint)


class CuspArmMugProgramNumerical(ScrewArmMugProgramNumerical):
    """Joint-space formulation, grasp task."""
    ResolveSpec = staticmethod(GetSpec)

    def __init__(self, diagram, options=ProgramOptions(), robot=PRIMARY, model=None,
                 checkpoint=None):
        super().__init__(diagram, options, robot, model, checkpoint)
