"""Make `helix7` visible to `jrl.robots.get_robot`, and therefore to ikflow.

`get_robot` scans `jrl.robots.ALL_CLCS` for a class whose `name` matches, so appending to
that list is the whole mechanism -- no fork of jrl, no edit to the vendored ikflow fork, and
no monkeypatched function whose behaviour depends on import order beyond this one.

Import `src.register_robots` rather than this module directly; that is the seam every other
robot will join, and it is what the program, `src/flow_loading.py` and the training entry
point all go through. Idempotent, so importing it twice is harmless.
"""

import jrl.robots

from src.helix_arm.robot import ROBOT_CLASSES


def Register():
    """Append every rung to jrl's registry. Safe to call repeatedly."""
    known = {clc.name for clc in jrl.robots.ALL_CLCS}
    for clc in ROBOT_CLASSES:
        if clc.name not in known:
            jrl.robots.ALL_CLCS.append(clc)
    jrl.robots.ALL_ROBOT_NAMES = [clc.name for clc in jrl.robots.ALL_CLCS]
    return tuple(clc.name for clc in ROBOT_CLASSES)


REGISTERED = Register()
