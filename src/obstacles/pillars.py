"""Family PILLAR: floor-to-ceiling cylinders in front of each shelf unit (a fixed scene).

THE RULE.  For each shelf unit in `SHELF_WELDS`, with t = (tx, ty) and local x axis
u = (cos yaw, sin yaw):

* the base-facing sign is s = sign(u . (0 - t)); the unit is open at both x faces, and this
  picks the one that faces the robot base (the world origin, where every record robot is
  welded);
* the base-facing opening centre is o = t + s * 0.15 * u (xy; 0.15 is the half-depth);
* the pillar is a vertical cylinder of radius r, z from 0.0 to 2.0 m, whose axis lies on the
  horizontal segment from the base (0, 0) to o, at distance (0.05 + r) from o along that
  segment -- so its SURFACE is 0.05 m from o along the segment.

Rungs: `PILLAR_RADII = (0.04, 0.08, 0.12)` = `pillars_r040`, `pillars_r080`, `pillars_r120`.
A pillar scene is the hardened scene's YAML text verbatim plus four appended directives
(`pillar1..pillar4`), so everything else is byte-identical to the record scene.
"""

import numpy as np

from src.shelf_regions import SHELF_WELDS

PILLAR_RADII = (0.04, 0.08, 0.12)
PILLAR_SCENE_NAMES = tuple("pillars_r%03d" % round(r * 1000) for r in PILLAR_RADII)
PILLAR_HEIGHT = 2.0
SHELF_HALF_DEPTH = 0.15
GAP = 0.05


def PillarSceneName(r):
    return "pillars_r%03d" % round(r * 1000)


def PillarCentres(r):
    """The four pillar axis points (x, y), in `SHELF_WELDS` order."""
    out = []
    for _, tx, ty, _tz, yaw_deg in SHELF_WELDS:
        t = np.array([tx, ty])
        u = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
        s = 1.0 if float(u @ (-t)) > 0 else -1.0
        o = t + s * SHELF_HALF_DEPTH * u
        d = -o / np.linalg.norm(o)
        out.append(o + (GAP + r) * d)
    return np.array(out)
