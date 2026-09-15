"""The shelf compartments of the hardened scenes, as boxes a target may be required to lie in.

This is the geometry half of the target hardening: pure numpy, no pydrake, so it imports in
microseconds and can be tested without building a plant.  `src/target_screening.py` is the
Drake half (the penetration screen and the sampler).

WHY IT EXISTS.  The grasp benchmark used to sample a collision-free `q`, weld a mug wherever
the gripper landed, and accept it unconditionally, so targets sat in free air far more often
than in clutter and the collision-avoidance half of the problem barely bound.  Requiring the
target to land inside a shelf compartment is what makes the scene's obstacles part of the
problem.  Ported from `../codebase`'s `MugAdversarialRegions` / `MugCenterInAdversarialRegions`
(adopted there 2026-09-15), minus the bin case -- the hardened scenes have no bin, so an
`include_bin` parameter would be a knob with one legal value.

THE DERIVATION, from `models/assets/shelves.sdf`.  Link `shelves_body` carries two side walls
("right_wall"/"left_wall"), boxes 0.3 x 0.016 x 0.783 at local y = +/-0.295, so their INNER
faces sit at y = +/-(0.295 - 0.008) = +/-0.287.  Link `top_and_bottom` -- fixed-jointed to
`shelves_body` with no offset, so it shares the same local frame -- carries four horizontal
boards, each 0.3 (x) x 0.6 (y) x 0.016 (z), at local z = -0.3995 (bottom), -0.13115 (lower
shelf), +0.13115 (upper shelf), +0.3995 (top).  Their usable faces are +/-0.008 off those
centres, which divides the unit into the three z compartments below.  Each compartment's
interior xy footprint is the boards' own extent: x in [-0.15, 0.15], y in [-0.287, 0.287], in
the LOCAL (unrotated) frame.

THE INSET IS SYMMETRIC, and that is not an oversight.  `shelves.sdf` has NO BACK WALL: the
unit is a tunnel open at both +/-x faces, so which x face is "front" depends on the weld's
yaw.  The side walls already close off y and the boards close off z, so depth is the only free
axis and the only one worth insetting.  `shelf_depth_inset` shrinks each compartment to
|x| <= 0.15 - inset.  Inset 0.0 reproduces the un-inset boxes exactly.

A WORLD-FRAME AABB WOULD NOT DO.  These units are welded at yaw 135 and 235 degrees, where an
axis-aligned bounding box of the rotated footprint over-approximates the area by about 4x
(hx=0.15, hy=0.287 against a ~0.31 x 0.31 AABB).  It would accept "in the shelves" targets
that are really in free air beside the unit.  So a region is stored as its LOCAL box plus the
weld's translation and yaw, and the predicate rotates the query point into the region's own
frame rather than rotating the box into the world.
"""

import numpy as np

## The boards' x extent and the inner faces of the side walls, both local to a shelf unit.
SHELF_HALF_EXTENTS = (0.15, 0.287)

## The three interior compartments, as local z intervals between consecutive board faces.
SHELF_Z_COMPARTMENTS = ((-0.3915, -0.13915), (-0.12315, 0.11515), (0.13915, 0.3915))

## (name, tx, ty, tz, yaw_deg) of each shelves.sdf weld.  Identical in all three hardened
## scenes -- and in the legacy ones, which is why this table is shared rather than per-scene.
## `tests/test_shelf_placement_screens.py` reads these poses back out of a built plant and
## fails if they drift, so the table cannot silently diverge from the YAMLs.
SHELF_WELDS = (
    ("shelves",   0.30,  0.55, 0.4, 235.0),
    ("shelves2",  0.30, -0.55, 0.4, 135.0),
    ("shelves3", -0.45, -0.55, 0.4, 235.0),
    ("shelves4", -0.45,  0.55, 0.4, 135.0),
)

## Adopted 2026-09-15, matching `../codebase`.  See `scripts/probe_shelf_acceptance.py` for
## the acceptance rates this buys on each robot, and for why 0.125 is not fielded.
DEFAULT_SHELF_DEPTH_INSET = 0.10


def ShelfCompartmentRegions(shelf_depth_inset=0.0):
    """The twelve shelf compartments, as (translation, yaw_rad, lo_local, hi_local).

    Four units x three compartments.  `lo_local`/`hi_local` are in the unit's own frame;
    `PointInShelfCompartments` is what interprets them.  Raises ValueError unless
    0 <= shelf_depth_inset < 0.15, since at or past the boards' half-depth every compartment
    is empty and the sampler would reject for ever rather than reporting a bad setting.
    """
    shelf_hx, shelf_hy = SHELF_HALF_EXTENTS
    if not 0.0 <= shelf_depth_inset < shelf_hx:
        raise ValueError(
            "shelf_depth_inset must be in [0, %g) or the shelf compartments are empty; "
            "got %r" % (shelf_hx, shelf_depth_inset))
    shelf_hx -= shelf_depth_inset

    regions = []
    for _, tx, ty, tz, yaw_deg in SHELF_WELDS:
        translation = np.array([tx, ty, tz])
        yaw = np.deg2rad(yaw_deg)
        for z_lo, z_hi in SHELF_Z_COMPARTMENTS:
            regions.append((translation, yaw,
                            np.array([-shelf_hx, -shelf_hy, z_lo]),
                            np.array([shelf_hx, shelf_hy, z_hi])))
    return regions


def PointInShelfCompartments(point_world, regions):
    """True if `point_world` lies inside any region's true (rotated) box.

    Rotates the point into each region's own frame -- p_local = Rz(-yaw) @ (p - t) -- rather
    than testing a world-frame axis-aligned bound, which at these welds would be about 4x too
    permissive.  See the module docstring.
    """
    point_world = np.asarray(point_world, dtype=float)
    for translation, yaw, lo_local, hi_local in regions:
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy, dz = point_world - translation
        p_local = np.array([c * dx + s * dy, -s * dx + c * dy, dz])
        if np.all(p_local >= lo_local) and np.all(p_local <= hi_local):
            return True
    return False


def ShelfRegionsFromPlant(plant, plant_context, shelf_depth_inset=0.0):
    """The same table, rebuilt from the poses a finalized plant actually reports.

    Production never calls this: the hardcoded table needs no plant, and a scene is built once
    per process where this would rebuild per call.  It exists so the test can assert the two
    agree, which is what makes it impossible for `SHELF_WELDS` and the scene YAMLs to drift
    apart -- the failure mode being a containment test silently scoring the wrong boxes.

    Raises if the scene does not carry exactly the four expected shelf instances.
    """
    ## Imported here, not at module scope, to keep this module pydrake-free for its callers.
    from pydrake.math import RigidTransform  # noqa: F401  (documents the return type)

    shelf_hx, shelf_hy = SHELF_HALF_EXTENTS
    if not 0.0 <= shelf_depth_inset < shelf_hx:
        raise ValueError("shelf_depth_inset must be in [0, %g); got %r"
                         % (shelf_hx, shelf_depth_inset))
    shelf_hx -= shelf_depth_inset

    regions = []
    for name, _, _, _, _ in SHELF_WELDS:
        if not plant.HasModelInstanceNamed(name):
            raise RuntimeError("scene has no shelf model instance %r" % name)
        instance = plant.GetModelInstanceByName(name)
        X_W = plant.GetFrameByName("shelves_body", instance).CalcPoseInWorld(plant_context)
        rpy = X_W.rotation().ToRollPitchYaw().vector()
        if abs(rpy[0]) > 1e-9 or abs(rpy[1]) > 1e-9:
            raise RuntimeError(
                "shelf %r is not welded with a pure yaw (rpy=%r); the compartment table "
                "assumes yaw-only welds" % (name, rpy))
        translation = X_W.translation()
        for z_lo, z_hi in SHELF_Z_COMPARTMENTS:
            regions.append((translation, float(rpy[2]),
                            np.array([-shelf_hx, -shelf_hy, z_lo]),
                            np.array([shelf_hx, shelf_hy, z_hi])))
    return regions
