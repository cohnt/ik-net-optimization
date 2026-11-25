"""The mobile-base program classes: each record robot class with the mobile mixin in front.

`MOBILE_CLASSES[(robot, task, arm)]` is what the benchmark drivers pick from under
`--scene mobile`. The learned classes set `num_task_vars = 9` ([b | c]) after the robot class's
own constructor has run -- that constructor calibrates the flow frame, which reads only the
mixin's lazily built geometry, never the variable layout.
"""

from src.iiwa_program import (Iiwa14IKProgram, Iiwa14IKProgramNumerical, IiwaMugProgram,
                              IiwaMugProgramNumerical)
from src.mobile_base.program import MobileJointMixin, MobileLearnedMixin
from src.panda_program import (PandaIKProgram, PandaIKProgramNumerical, PandaMugProgram,
                               PandaMugProgramNumerical)
from src.screw_arm_program import (ScrewArmIKProgram, ScrewArmIKProgramNumerical,
                                   ScrewArmMugProgram, ScrewArmMugProgramNumerical)


def _Learned(base):
    class Mobile(MobileLearnedMixin, base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._InitMobileLearned()
    Mobile.__name__ = Mobile.__qualname__ = "Mobile" + base.__name__
    return Mobile


def _Joint(base):
    class Mobile(MobileJointMixin, base):
        pass
    Mobile.__name__ = Mobile.__qualname__ = "Mobile" + base.__name__
    return Mobile


MOBILE_CLASSES = {}
MOBILE_CLASSES.update({
    ("panda", "pose", "learned"): _Learned(PandaIKProgram),
    ("panda", "mug", "learned"): _Learned(PandaMugProgram),
    ("panda", "pose", "numerical"): _Joint(PandaIKProgramNumerical),
    ("panda", "mug", "numerical"): _Joint(PandaMugProgramNumerical),
    ("iiwa", "pose", "learned"): _Learned(Iiwa14IKProgram),
    ("iiwa", "mug", "learned"): _Learned(IiwaMugProgram),
    ("iiwa", "pose", "numerical"): _Joint(Iiwa14IKProgramNumerical),
    ("iiwa", "mug", "numerical"): _Joint(IiwaMugProgramNumerical),
    ("screw", "pose", "learned"): _Learned(ScrewArmIKProgram),
    ("screw", "mug", "learned"): _Learned(ScrewArmMugProgram),
    ("screw", "pose", "numerical"): _Joint(ScrewArmIKProgramNumerical),
    ("screw", "mug", "numerical"): _Joint(ScrewArmMugProgramNumerical),
})


def _SoftClasses():
    from src.mobile_base.soft import MOBILE_SOFT_CLASSES
    return MOBILE_SOFT_CLASSES


MOBILE_CLASSES.update(_SoftClasses())
