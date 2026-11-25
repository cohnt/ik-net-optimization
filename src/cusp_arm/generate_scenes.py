"""Emit both scene YAMLs for `cusp7`, by the screw arm's generator.

`src.screw_arm.generate_scenes.RenderScene` reads only the spec's `name`, `base_link` and
`flange_link`, plus the gripper mount; the furniture (finray gripper, tables, shelves at
`src/shelf_regions.py`'s welds, the legacy bin) is shared by URI with every robot of the
record, which is what makes a row here comparable. The base is welded at the world origin
with no `X_PC` (`CalibrateFlowFrame` needs it), and the gripper at the tool link's
`GRIPPER_MOUNT_Z` with the iiwa's rotation.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import src.screw_arm.generate_scenes as _screw_scenes
from src.cusp_arm.params import GRIPPER_MOUNT_Z, SPECS, CuspArmSpec


def RenderScene(spec: CuspArmSpec, legacy: bool = False) -> str:
    text = _screw_scenes.RenderScene(spec, legacy)
    # The shared generator hardcodes the screw arm's mount (0.10) and names itself in the
    # header; both are rewritten so the file says what it is.
    if _screw_scenes.GRIPPER_MOUNT_Z != GRIPPER_MOUNT_Z:
        raise RuntimeError("the screw arm's gripper mount changed; render the weld here")
    return text.replace("src/screw_arm/generate_scenes.py", "src/cusp_arm/generate_scenes.py")


def ScenePath(spec: CuspArmSpec, root: str, legacy: bool = False) -> str:
    return _screw_scenes.ScenePath(spec, root, legacy)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    for spec in SPECS.values():
        for legacy in (True, False):
            path = ScenePath(spec, root, legacy)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(RenderScene(spec, legacy))
            print(f"wrote {os.path.relpath(path, root)}")


if __name__ == "__main__":
    main()
