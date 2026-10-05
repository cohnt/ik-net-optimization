"""Make the GVS push-rod arm visible to `jrl.robots.get_robot`, and therefore to ikflow.

Same mechanism as `src/soft_arm/register.py`: `get_robot` scans `jrl.robots.ALL_CLCS` for a
class whose `name` matches, so appending to that list is the whole thing. Import this
through `src.register_robots`, which is the one list every by-name lookup site imports.
"""

import jrl.robots

from src.gvs_arm.robot import ROBOT_CLASSES


def Register():
    """Append every rung to jrl's registry. Safe to call repeatedly."""
    known = {clc.name for clc in jrl.robots.ALL_CLCS}
    for clc in ROBOT_CLASSES:
        if clc.name not in known:
            jrl.robots.ALL_CLCS.append(clc)
    jrl.robots.ALL_ROBOT_NAMES = [clc.name for clc in jrl.robots.ALL_CLCS]
    return tuple(clc.name for clc in ROBOT_CLASSES)


REGISTERED = Register()
