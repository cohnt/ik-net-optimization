"""Make `cusp7` visible to `jrl.robots.get_robot`, and therefore to ikflow.

Appending to `jrl.robots.ALL_CLCS` is the whole mechanism (see src/screw_arm/register.py).
Import `src.register_robots` rather than this module directly. Idempotent.
"""

import jrl.robots

from src.cusp_arm.robot import ROBOT_CLASSES


def Register():
    known = {clc.name for clc in jrl.robots.ALL_CLCS}
    for clc in ROBOT_CLASSES:
        if clc.name not in known:
            jrl.robots.ALL_CLCS.append(clc)
    jrl.robots.ALL_ROBOT_NAMES = [clc.name for clc in jrl.robots.ALL_CLCS]
    return tuple(clc.name for clc in ROBOT_CLASSES)


REGISTERED = Register()
