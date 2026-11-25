"""In-memory model directives that weld spherical obstacles into a scene.

pydrake-only imports on purpose: `src/svgd/collision_backend.py` imports this, and must not
pull in `src.utils` (cycles).  The same function builds the per-target solve diagram, the
screen's rebuilt scenes and the svgd collision checker's, so every path sees the same
obstacles in the same directive order (plant indexing and pair sets then match).
"""

import numpy as np
from pydrake.common.schema import Transform as SchemaTransform
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import AddModel, AddWeld, ModelDirective
from pydrake.multibody.tree import ModelInstanceIndex

OBSTACLE_SDF = "package://combining_kinematics/models/assets/obstacle_sphere_r040.sdf"
OBSTACLE_MODEL_FMT = "obstacle_%03d"
OBSTACLE_LINK = "sphere_link"


def ObstacleDirectives(centres, sdf=OBSTACLE_SDF):
    """add_model + add_weld per centre: `obstacle_%03d` welded to the world, pure translation."""
    out = []
    for k, c in enumerate(centres):
        c = np.asarray(c, dtype=np.float64).reshape(3)
        name = OBSTACLE_MODEL_FMT % k
        add = ModelDirective()
        add.add_model = AddModel(name=name, file=sdf)
        weld = ModelDirective()
        weld.add_weld = AddWeld(parent="world", child="%s::%s" % (name, OBSTACLE_LINK),
                                X_PC=SchemaTransform(RigidTransform(c)))
        out += [add, weld]
    return out


def ObstacleInstanceNames(plant):
    """Names of the model instances called `obstacle_###` in `plant`."""
    names = [plant.GetModelInstanceName(ModelInstanceIndex(i))
             for i in range(plant.num_model_instances())]
    return [n for n in names if n.startswith("obstacle_")]
