"""The hardened scenes, the shelf compartment table, and the two target screens.

Guards three things that are each silent when they break:

* the hardcoded `SHELF_WELDS` table drifting from the scene YAMLs, which would leave the
  containment test scoring boxes that are not where the shelves are;
* a hardened scene drifting from its legacy twin by more than the removals it is defined as;
* the penetration screen's robot filter silently matching nothing after a model instance is
  renamed, which rejects every candidate as "penetrating" and looks exactly like a too-deep
  inset -- i.e. it burns a campaign rather than failing.

No pytest config in this repo, so these are plain `test_*` functions with a `__main__`
driver.  `_LocalToWorld` is written here rather than imported, so this cross-checks
`src/shelf_regions.py` instead of restating it.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import LoadModelDirectives
from pydrake.multibody.tree import ModelInstanceIndex

from src.shelf_regions import (SHELF_HALF_EXTENTS, SHELF_WELDS, SHELF_Z_COMPARTMENTS,
                               PointInShelfCompartments, ShelfCompartmentRegions,
                               ShelfRegionsFromPlant)
from src.target_screening import (SCENES, ContainmentPose, FloatingMugScreen,
                                  SampleShelfTargets, SceneFile)
from src.utils import BuildEnv, RepoDir

DECORATIVE_MUGS = ("mug", "mug2", "mug3", "mug4", "rmug", "rmug2", "rmug3")
STATIC_FURNITURE = {"table", "table2", "shelves", "shelves2", "shelves3", "shelves4"}

## (robot, task) -> the instances the legacy twin carries that the hardened one must not.
## The panda GRASP scene never had decorative mugs, so it loses only the bin: asserting the
## same removal list for all three would make the test pass for the wrong reason.
## No ("panda", "pose") entry: the Panda pose scene IS the Panda grasp scene since the stock
## hand was removed, so it would test the same file twice. Both robots now carry the same
## finray on both tasks, which is what makes reaching into a shelf geometrically comparable
## between them.
REMOVED = {
    ("panda", "mug"): ("binF",),
    ("iiwa", "mug"): ("binF",) + DECORATIVE_MUGS,
}

_CACHE = {}


def _scene(robot, task, scene="hardened"):
    """Built diagram + plant + context, memoised (building a scene is seconds, not ms)."""
    key = (robot, task, scene)
    if key not in _CACHE:
        diagram = BuildEnv(meshcat=None, directives_file=SceneFile(robot, task, scene))
        plant = diagram.GetSubsystemByName("plant")
        context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
        _CACHE[key] = (diagram, plant, context)
    return _CACHE[key]


def _instance_names(plant):
    return {plant.GetModelInstanceName(ModelInstanceIndex(i))
            for i in range(plant.num_model_instances())} - {"WorldModelInstance",
                                                            "DefaultModelInstance"}


def _LocalToWorld(translation, yaw, p_local):
    """Independent of src/shelf_regions.py on purpose: Rz(yaw) @ p_local + t."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = p_local
    return np.array([translation[0] + c * x - s * y,
                     translation[1] + s * x + c * y,
                     translation[2] + z])


def test_region_table_matches_scene_welds():
    """The hardcoded welds are where Drake actually puts the shelves, in every scene."""
    for robot, task in REMOVED:
        _, plant, context = _scene(robot, task)
        shelves = {n for n in _instance_names(plant) if n.startswith("shelves")}
        assert shelves == {w[0] for w in SHELF_WELDS}, (robot, task, shelves)
        for name, tx, ty, tz, yaw_deg in SHELF_WELDS:
            instance = plant.GetModelInstanceByName(name)
            X = plant.GetFrameByName("shelves_body", instance).CalcPoseInWorld(context)
            assert np.allclose(X.translation(), [tx, ty, tz], atol=1e-12), (robot, name)
            ## Matrices, not rpy degrees: Drake reports the 235-degree welds as -125, so a
            ## naive degree comparison fails on a scene that is perfectly correct.
            expected = RotationMatrix.MakeZRotation(np.deg2rad(yaw_deg))
            assert X.rotation().IsNearlyEqualTo(expected, 1e-12), (robot, name)

        for inset in (0.0, 0.10):
            a = ShelfRegionsFromPlant(plant, context, inset)
            b = ShelfCompartmentRegions(inset)
            assert len(a) == len(b) == 12
            for (t1, y1, lo1, hi1), (t2, y2, lo2, hi2) in zip(a, b):
                assert np.allclose(t1, t2, atol=1e-12)
                ## Compare as a direction: -125 and 235 degrees are the same weld.
                assert np.isclose(np.cos(y1), np.cos(y2), atol=1e-12)
                assert np.isclose(np.sin(y1), np.sin(y2), atol=1e-12)
                assert np.allclose(lo1, lo2) and np.allclose(hi1, hi2)
    print("PASS region table matches every scene's welds")


def test_hardened_scenes_have_no_bin_or_decorative_mugs():
    for (robot, task), removed in REMOVED.items():
        _, hard_plant, _ = _scene(robot, task, "hardened")
        _, legacy_plant, _ = _scene(robot, task, "legacy")
        hard, legacy = _instance_names(hard_plant), _instance_names(legacy_plant)
        for name in removed:
            assert not hard_plant.HasModelInstanceNamed(name), (robot, task, name)
            ## The legacy twin MUST have it, or the assertion above is vacuous.
            assert legacy_plant.HasModelInstanceNamed(name), (robot, task, name)
        assert legacy - hard == set(removed), (robot, task, legacy - hard)
        assert STATIC_FURNITURE <= hard, (robot, task, hard)
    print("PASS hardened scenes drop exactly the bin and the decorative mugs")


def test_nobin_scene_drops_only_the_bin():
    """`nobin` is the scene that separates containment from clutter -- it must keep both."""
    for robot, task in (("iiwa", "mug"), ("iiwa", "pose")):
        _, plant, context = _scene(robot, task, "nobin")
        present = _instance_names(plant)
        assert not plant.HasModelInstanceNamed("binF"), (robot, task)
        ## The whole point: the decorative mugs STAY. Without this the scene is just the
        ## hardened one under a different name and the disambiguation measures nothing.
        for name in DECORATIVE_MUGS:
            assert plant.HasModelInstanceNamed(name), (robot, task, name)
        assert STATIC_FURNITURE <= present, (robot, task)
        ## And it must still be a legal containment scene: same four shelves, same poses.
        for name, tx, ty, tz, yaw_deg in SHELF_WELDS:
            X = plant.GetFrameByName(
                "shelves_body", plant.GetModelInstanceByName(name)).CalcPoseInWorld(context)
            assert np.allclose(X.translation(), [tx, ty, tz], atol=1e-12), (robot, name)
            assert X.rotation().IsNearlyEqualTo(
                RotationMatrix.MakeZRotation(np.deg2rad(yaw_deg)), 1e-12), (robot, name)
        ## Exactly the bin separates it from legacy.
        _, legacy_plant, _ = _scene(robot, task, "legacy")
        assert _instance_names(legacy_plant) - present == {"binF"}, (robot, task)
    ## The Panda never had decorative mugs on either task -- its pose scene IS its grasp
    ## scene -- so hardened already IS its nobin scene, on both tasks.
    for task in ("mug", "pose"):
        assert SceneFile("panda", task, "nobin") == SceneFile("panda", task, "hardened")
    print("PASS nobin scene drops the bin and keeps the clutter")


def test_hardened_matches_legacy_minus_removals():
    """Semantic comparison of the directives Drake actually sees, in both directions."""
    for (robot, task), removed in REMOVED.items():
        hard = LoadModelDirectives(SceneFile(robot, task, "hardened")).directives
        legacy = LoadModelDirectives(SceneFile(robot, task, "legacy")).directives

        def names(d):
            out = []
            if d.add_model is not None:
                out.append(d.add_model.name)
            if d.add_weld is not None:
                out.append(d.add_weld.child.split("::")[0])
            return out

        kept = [d for d in legacy if not any(n in removed for n in names(d))]
        assert [repr(d) for d in hard] == [repr(d) for d in kept], (robot, task)
    print("PASS hardened scenes are their legacy twins minus the removals")


def test_scene_registry_matches_plants():
    """Every robot-filter name exists, the complement is exactly furniture, frames resolve."""
    for (robot, task), spec in SCENES.items():
        _, plant, _ = _scene(robot, task)
        present = _instance_names(plant)
        for name in spec.robot_instances:
            assert plant.HasModelInstanceNamed(name), (robot, task, name)
        assert present - set(spec.robot_instances) == STATIC_FURNITURE, (robot, task, present)
        assert plant.HasFrameNamed(spec.target_frame), (robot, task, spec.target_frame)
    print("PASS scene registry matches the built plants")


def test_exact_containment_vs_world_aabb():
    """A world AABB over-approximates ~4x at these welds; the predicate must not."""
    regions = ShelfCompartmentRegions(0.0)
    hx, hy = SHELF_HALF_EXTENTS
    checked = 0
    for translation, yaw, lo_local, hi_local in regions:
        z_mid = 0.5 * (lo_local[2] + hi_local[2])
        inside = _LocalToWorld(translation, yaw, [0.9 * hx, 0.9 * hy, z_mid])
        assert PointInShelfCompartments(inside, regions)
        ## 0.95 of the ROTATED footprint's axis-aligned half-extents: inside the AABB a
        ## lazier implementation would test, outside the real box.
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        aabb_hx, aabb_hy = hx * c + hy * s, hx * s + hy * c
        outside = np.array([translation[0] + 0.95 * aabb_hx,
                            translation[1] + 0.95 * aabb_hy,
                            translation[2] + z_mid])
        assert not PointInShelfCompartments(outside, regions), (translation, yaw)
        checked += 1
    assert checked == 12
    print("PASS exact rotated-box containment beats the world AABB")


def test_inset_shrinks_depth_only():
    hx, hy = SHELF_HALF_EXTENTS
    base, inset = ShelfCompartmentRegions(0.0), ShelfCompartmentRegions(0.10)
    for (_, _, lo0, hi0), (_, _, lo1, hi1) in zip(base, inset):
        assert np.isclose(hi1[0], hx - 0.10) and np.isclose(lo1[0], -(hx - 0.10))
        assert np.isclose(hi1[1], hi0[1]) and np.isclose(lo1[1], lo0[1])
        assert np.isclose(hi1[2], hi0[2]) and np.isclose(lo1[2], lo0[2])
    ## A point 0.10 m deep is inside un-inset and outside at inset 0.10.
    translation, yaw = base[0][0], base[0][1]
    z_mid = 0.5 * (base[0][2][2] + base[0][3][2])
    p = _LocalToWorld(translation, yaw, [0.10, 0.0, z_mid])
    assert PointInShelfCompartments(p, base)
    assert not PointInShelfCompartments(p, inset)
    for bad in (-0.01, 0.15, 0.2):
        try:
            ShelfCompartmentRegions(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("inset %r should have raised" % bad)
    print("PASS the inset shrinks depth only, and validates its range")


def _screen_for(robot, task):
    spec = SCENES[(robot, task)]
    return FloatingMugScreen(SceneFile(robot, task), spec.robot_instances)


def test_penetration_screen_accepts_and_rejects():
    for robot, task in (("iiwa", "mug"), ("panda", "mug")):
        screen = _screen_for(robot, task)
        name, tx, ty, tz, yaw_deg = SHELF_WELDS[0]
        R = RotationMatrix.MakeZRotation(np.deg2rad(yaw_deg))
        t = np.array([tx, ty, tz])

        def world(local):
            return RigidTransform(R, t + R.multiply(np.asarray(local, dtype=float)))

        z_mid = 0.5 * (SHELF_Z_COMPARTMENTS[1][0] + SHELF_Z_COMPARTMENTS[1][1])
        assert not screen.Penetrates([world([0, 0, z_mid])]), robot
        ## Centred on the lower board, and driven through a side wall.
        assert screen.Penetrates([world([0, 0, -0.13115])]), robot
        assert screen.Penetrates([world([0, 0.295, z_mid])]), robot
        ## No poses at all: every slot parks far away and nothing collides.
        assert not screen.Penetrates([]), robot
    print("PASS penetration screen accepts compartment centres, rejects boards and walls")


def test_screen_ignores_robot_pairs():
    """Mug-vs-robot must be ignored, or every candidate is rejected as 'penetrating'."""
    for robot, task in (("iiwa", "mug"), ("panda", "mug")):
        screen = _screen_for(robot, task)
        ## Sitting inside the robot's own base geometry.
        assert not screen.Penetrates([RigidTransform([0.0, 0.0, 0.2])]), robot
    print("PASS the screen ignores mug-vs-robot contact")


def test_containment_points_agree_across_robots():
    """`--placement-point fingertips` must resolve on every scene, and differ where it should.

    Both robots now carry the SAME finray gripper on BOTH tasks -- the Panda pose scene used
    to be `panda_jrl.urdf` with its own stock Franka hand, which made the pose task's
    collision geometry differ between robots for no reason. With that gone, every scene has
    a real `between_fingers` frame and reaching into a shelf is geometrically comparable
    across robots.
    """
    seps = {}
    for (robot, task), spec in SCENES.items():
        assert spec.wrist_frame and spec.fingertip_frame, (robot, task)
        _, plant, context = _scene(robot, task)
        for mode in ("wrist", "fingertips"):
            pose_of, label = ContainmentPose(plant, context, spec, mode)
            q = np.zeros(plant.num_positions())
            X = pose_of(q)
            assert np.all(np.isfinite(X.translation())), (robot, task, mode)
            assert label
        ## The separation must be the SAME on both robots, or rejection sampling is keyed on
        ## different points on the hand and the two are not comparable problems.
        wrist, _ = ContainmentPose(plant, context, spec, "wrist")
        tip, _ = ContainmentPose(plant, context, spec, "fingertips")
        q = np.zeros(plant.num_positions())
        d = float(np.linalg.norm(wrist(q).translation() - tip(q).translation()))
        seps.setdefault(task, set()).add(round(d, 9))
        if task == "mug":
            ## the mug is welded at between_fingers, so the two must coincide there
            assert d < 1e-12, (robot, task, d)
        else:
            assert abs(d - 0.100) < 1e-9, (robot, task, d)
    ## The decisive assertion: one separation value per task, shared by both robots.
    for task, values in seps.items():
        assert len(values) == 1, (task, values)
    print("PASS containment points are the same point on the hand for both robots")


def test_free_placement_reproduces_the_legacy_stream():
    """With no regions and no screen the sampler is the pre-hardening comprehension."""
    lower, upper = np.zeros(7), np.ones(7)

    def legacy(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < 5:
            q = rng.uniform(lower, upper)
            if q[0] < 0.7:          # a stand-in for the collision filter
                out.append(q)
        return out

    rng = np.random.default_rng(0)
    qs, stats = SampleShelfTargets(
        5, draw=lambda: rng.uniform(lower, upper),
        collision_free=lambda q: q[0] < 0.7, target_pose=None)
    assert all(np.array_equal(a, b) for a, b in zip(qs, legacy(0)))
    assert stats["accepted"] == 5 and stats["containment_rejected"] == 0
    print("PASS free placement reproduces the legacy stream exactly")


def test_rejection_guard_trips_with_a_useful_message():
    rng = np.random.default_rng(0)
    try:
        SampleShelfTargets(1, draw=lambda: rng.uniform(0, 1, 7),
                           collision_free=lambda q: False, target_pose=None,
                           max_consecutive_rejections=50, label="iiwa/pose inset 0.100")
    except RuntimeError as exc:
        assert "iiwa/pose inset 0.100" in str(exc)
        assert "on collision" in str(exc) and "probe_shelf_acceptance" in str(exc)
    else:
        raise AssertionError("the guard should have tripped")
    print("PASS the rejection guard trips and says what was rejected and why")


if __name__ == "__main__":
    test_region_table_matches_scene_welds()
    test_hardened_scenes_have_no_bin_or_decorative_mugs()
    test_nobin_scene_drops_only_the_bin()
    test_hardened_matches_legacy_minus_removals()
    test_scene_registry_matches_plants()
    test_exact_containment_vs_world_aabb()
    test_inset_shrinks_depth_only()
    test_penetration_screen_accepts_and_rejects()
    test_screen_ignores_robot_pairs()
    test_containment_points_agree_across_robots()
    test_free_placement_reproduces_the_legacy_stream()
    test_rejection_guard_trips_with_a_useful_message()
    print("ALL PASS")
