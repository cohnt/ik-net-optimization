"""A `jrl.robot.Robot` for `cusp7`, so ikflow can train on it unmodified.

The jrl interface is the screw arm's shim (`src.screw_arm.robot.ScrewArmRobot`: no
`super().__init__`, batched torch FK on the CPU returned on the caller's device, loud
`NotImplementedError` on every path ikflow never takes). Its forward kinematics walks the
spec's joint table generically, so it serves this all-revolute chain unchanged. What is this
robot's own is the self-collision screen, keyed on `src.cusp_arm.params.COLLISION_PAIRS`
-- the same tuple the SDF's filter groups come from -- and the identity strings.

There is a URDF-free reason too: the robot is defined by `params.py` and rendered as
SDFormat, so jrl's URDF/klampt path is not the definition, exactly as on the screw arm.
"""

import torch

from src.cusp_arm import kinematics as K
from src.cusp_arm.params import SPECS, CuspArmSpec
from src.screw_arm.robot import ScrewArmRobot

_CPU = "cpu"


class CuspArmRobot(ScrewArmRobot):
    """The cuspidal arm, wearing jrl's interface."""

    name = "cusp7"
    formal_robot_name = "Cuspidal 7-DoF arm (Transpressor 6R + tool pitch)"
    spec: CuspArmSpec = SPECS["cusp7"]

    @property
    def urdf_filepath(self):
        raise NotImplementedError(
            f"{self._name} has no URDF: the robot is src/cusp_arm/params.py, rendered as "
            f"SDFormat at models/{self._name}/{self._name}.sdf (Drake reads collision "
            f"filter groups from SDFormat only).")

    def __str__(self):
        return f"<CuspArmRobot {self._name}, {self._ndof} DoF>"

    __repr__ = __str__

    def config_self_collides(self, x, verbose: bool = False):
        """Do any two unfiltered collision spheres overlap? Batched -> bool tensor on the
        caller's device; one configuration -> plain bool (jrl's two calling shapes)."""
        device = x.device if isinstance(x, torch.Tensor) else _CPU
        tensor = torch.as_tensor(x, dtype=torch.float64, device=_CPU)
        single = tensor.dim() == 1
        if single:
            tensor = tensor.unsqueeze(0)
        collides = K.self_collision_depth(tensor, self._spec, device=_CPU) > 0.0
        return bool(collides[0]) if single else collides.to(device)


ROBOT_CLASSES = (CuspArmRobot,)
