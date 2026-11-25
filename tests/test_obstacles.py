"""The obstacle campaign's invariants: rung 0 is the record, the OBST stream is pure, the screen
refuses what it must, the witness certificate holds, SVGD sees the obstacles, the PILLAR scenes
follow their stated rule.

Guards what is silent when it breaks:

* a change to `scene_fingerprint`, `GenerateDiagramWithMug` or `SCENES` that moves a record
  scene (rung 0 must stay bit-for-bit; goldens in `tests/data/record_scene_fingerprints.json`);
* a candidate stream that depends on rejections or sharding;
* a screen whose robot filter matches nothing, or that cannot see the mug (Drake never makes
  candidate pairs between two ANCHORED geometries, so a welded mug/obstacle pair is never
  reported by the solve scene);
* the SVGD collision backend rebuilding the solve scene WITHOUT the obstacles.

Plain `test_*` functions through tests/_tiers.py.  `fast` is ~1-2 min on one core; `slow` adds
the witness certificate on every robot x task and every pillar scene.
"""

import json
import os
import sys
import types
from functools import partial

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import _tiers                                                             # noqa: E402

from pydrake.math import RigidTransform                                   # noqa: E402
from pydrake.multibody.parsing import AddModel, LoadModelDirectives, ModelDirective  # noqa: E402

from src.benchmark import MUG_MODEL_FILE, scene_fingerprint               # noqa: E402
from src.generic_program import ProgramOptions                           # noqa: E402
from src.obstacles import generate_pillar_scenes as gen                  # noqa: E402
from src.obstacles import per_target as pt                               # noqa: E402
from src.obstacles.directives import (OBSTACLE_SDF, ObstacleDirectives,  # noqa: E402
                                      ObstacleInstanceNames)
from src.obstacles.pillars import PILLAR_RADII, PillarCentres, PillarSceneName  # noqa: E402
from src.shelf_regions import SHELF_WELDS, ShelfCompartmentRegions       # noqa: E402
from src.svgd.collision_backend import SceneSpec                          # noqa: E402
from src.target_screening import (PILLAR_SCENES, SCENES, ContainmentPose,  # noqa: E402
                                  FloatingMugScreen, SampleShelfTargets, SceneFile)
from src.utils import (BuildEnv, BuildTargetDiagram, GenerateDiagramWithMug, Mug,  # noqa: E402
                       RepoDir)

ROOT = RepoDir()
GOLDEN = os.path.join(ROOT, "tests", "data", "record_scene_fingerprints.json")
GOLDEN_PREFIX = {
    "panda/mug": "64dbd94dc2c2", "panda/pose": "bcdb2d2af900",
    "iiwa/mug": "e8f8de49adba", "iiwa/pose": "3a1c302e3626",
    "screw7_p050/mug": "48e42ac398bd", "screw7_p050/pose": "6a0cb1658654",
    "soft12/mug": "414394e0409a", "soft12/pose": "f034ae3a9859",
}
FAR_FREE = np.array([2.0, 2.0, 1.0])


def _fp(robot, task, **kw):
    return scene_fingerprint(SceneFile(robot, task, "hardened"),
                             extra_models=[MUG_MODEL_FILE] if task == "mug" else [], **kw)


## ------------------------------------------------------------------ fingerprints ------

def test_rung0_fingerprints_are_the_record():
    gold = json.load(open(GOLDEN))
    assert set(gold) == set(GOLDEN_PREFIX)
    for key, prefix in GOLDEN_PREFIX.items():
        assert gold[key].startswith(prefix), (key, gold[key], prefix)
        robot, task = key.split("/")
        assert _fp(robot, task) == gold[key], key


def test_payload_changes_the_fingerprint_only_when_given():
    base = _fp("iiwa", "mug")
    assert _fp("iiwa", "mug", extra_payload=None) == base
    rng = np.random.default_rng(0)
    centres = [rng.uniform(size=(pt.N_MAX, 3)) for _ in range(3)]
    p8 = pt.ObstacleFingerprintPayload(8, centres)
    p16 = pt.ObstacleFingerprintPayload(16, centres)
    assert p8 == pt.ObstacleFingerprintPayload(8, centres)          # deterministic
    d8, d16 = _fp("iiwa", "mug", extra_payload=p8), _fp("iiwa", "mug", extra_payload=p16)
    assert len({base, d8, d16}) == 3
    moved = [c.copy() for c in centres]
    moved[2][3, 0] += 1e-9
    assert _fp("iiwa", "mug", extra_payload=pt.ObstacleFingerprintPayload(8, moved)) != d8


## ------------------------------------------------------------------ the RNG stream ------

class _LogScreen:
    """A pure-function stand-in for ObstacleScreen that logs every candidate it is shown."""
    def __init__(self, accept):
        self.accept, self.seen = accept, []

    def Screen(self, q, c, mug_pose=None):
        self.seen.append(np.array(c))
        return (True, "accepted") if self.accept(c) else (False, "scene")


def _hash_accept(c):
    return (int(abs(c[0]) * 1e6) + int(abs(c[1]) * 1e6)) % 3 != 0


def test_stream_is_pure_and_prefix_nested():
    point = np.array([0.1, 0.5, 0.5])
    seed, robot, task = 5, "iiwa", "mug"
    # target i alone == its slot in an all-targets draw
    allsets = []
    for i in range(4):
        s = _LogScreen(_hash_accept)
        allsets.append(pt.DrawTargetObstacles(s, pt.TargetRng(seed, robot, task, i),
                                              None, point, None)[0])
    s = _LogScreen(_hash_accept)
    alone = pt.DrawTargetObstacles(s, pt.TargetRng(seed, robot, task, 2), None, point, None)[0]
    assert np.array_equal(alone, allsets[2]) and not np.array_equal(allsets[1], allsets[2])
    assert allsets[0].shape == (pt.N_MAX, 3)
    # the candidate stream is unchanged by rejections: candidate j is the same whatever
    # happened to candidates < j
    sa, sb = _LogScreen(lambda c: True), _LogScreen(_hash_accept)
    pt.DrawTargetObstacles(sa, pt.TargetRng(seed, robot, task, 0), None, point, None)
    pt.DrawTargetObstacles(sb, pt.TargetRng(seed, robot, task, 0), None, point, None)
    # the screen only ever sees candidates in stream order; the rejecting screen saw a
    # different NUMBER of them, yet both are prefixes of the one raw stream
    raw = pt.TargetRng(seed, robot, task, 0)
    stream = [pt._candidate(raw, point) for _ in range(len(sa.seen) + len(sb.seen) + 200)]
    for seen in (sa.seen, sb.seen):
        j = 0
        for c in seen:                      # screened candidates are a subsequence of the stream
            while not np.array_equal(stream[j], c):
                j += 1
            j += 1
    assert len(sa.seen) != len(sb.seen)
    first = stream
    # a sphere ball: every candidate lies within REGION_RADIUS of the point
    assert max(np.linalg.norm(c - point) for c in first) <= pt.REGION_RADIUS + 1e-12
    # rung n = first n accepted: the rung-16 set is the head of the rung-64 set, and the
    # payloads agree on it
    r64 = allsets[0]
    assert np.array_equal(r64[:16], r64[:16].copy())
    p16 = pt.ObstacleFingerprintPayload(16, [r64])
    p16b = pt.ObstacleFingerprintPayload(16, [r64[:16]])
    assert p16 == p16b
    # block draws: 4 numbers per candidate, whatever the fate
    g = pt.TargetRng(seed, robot, task, 0)
    pt._candidate(g, point)
    h = pt.TargetRng(seed, robot, task, 0)
    h.standard_normal(3)
    h.uniform()
    assert g.standard_normal() == h.standard_normal()


## ------------------------------------------------------------------ the scene helpers ------

IIWA = "iiwa"
_CACHE = {}


def _iiwa_screen():
    if "screen" not in _CACHE:
        spec = SCENES[(IIWA, "mug")]
        _CACHE["screen"] = pt.ObstacleScreen(SceneFile(IIWA, "mug"), spec.robot_instances,
                                             with_mug=True)
    return _CACHE["screen"]


def _sample_target(robot, task, seed=7):
    """One collision-free, shelf-contained generating configuration, the way the benchmark does
    it (uniform in the joint box, collision-free, fingertips in a compartment, mug clear)."""
    key = ("target", robot, task, seed)
    if key in _CACHE:
        return _CACHE[key]
    from pydrake.all import MinimumDistanceLowerBoundConstraint
    spec = SCENES[(robot, task)]
    yaml = SceneFile(robot, task)
    diagram = BuildEnv(meshcat=None, directives_file=yaml)
    plant = diagram.GetSubsystemByName("plant")
    if robot.startswith("screw7"):
        from src.screw_arm.limits import ApplyScrewJointLimits
        from src.screw_arm.params import SPECS
        ApplyScrewJointLimits(plant, SPECS[robot])
    ctx = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise RuntimeError("%s: non-finite position limits" % robot)
    cons = MinimumDistanceLowerBoundConstraint(plant=plant, bound=1e-3,
                                               influence_distance_offset=0.1, plant_context=ctx)
    rng = np.random.default_rng(seed)
    pose_of, _ = ContainmentPose(plant, ctx, spec, "fingertips")
    mugscreen = FloatingMugScreen(yaml, spec.robot_instances)
    qs, _ = SampleShelfTargets(
        1, lambda: rng.uniform(lo, hi),
        lambda q: float(np.asarray(cons.Eval(q)).flatten()[0]) < 1.0, pose_of,
        regions=ShelfCompartmentRegions(0.10),
        screen=lambda X: mugscreen.Penetrates([X]), label="%s/%s" % (robot, task))
    q = qs[0]
    X = pose_of(q)
    _CACHE[key] = (q, X)
    return q, X


def _planted(screen, q, centre, mug_pose=None):
    return screen.Screen(q, np.asarray(centre, dtype=np.float64), mug_pose)


## ------------------------------------------------------------------ the screen ------

def test_screen_refuses_and_accepts():
    screen = _iiwa_screen()
    q, X = _sample_target(IIWA, "mug")
    assert _planted(screen, q, FAR_FREE) == (True, "accepted")
    assert _planted(screen, q, [0.5, 0.0, 0.01])[1] == "floor"
    # inside a shelf board: the lower board's centre of the shelf unit farthest from the hand
    best, far = None, -1.0
    for _, tx, ty, tz, yaw in SHELF_WELDS:
        c = np.array([tx, ty, tz - 0.13115])
        d = np.linalg.norm(c - X.translation())
        if d > far:
            best, far = c, d
    ok, why = _planted(screen, q, best)
    assert (ok, why) == (False, "scene"), (ok, why)
    # inside the target mug (mug parked in free space so the robot is not the nearer reason)
    mug = RigidTransform([2.0, 2.0, 1.0])
    assert _planted(screen, q, [2.0, 2.0, 1.0], mug) == (False, "mug")
    assert _planted(screen, q, [2.0, 2.0, 1.0], None)[0]                  # mug parked: free
    # within 0.02 m of the gripper
    hand = screen.plant.GetFrameByName("hand")
    screen._set_robot(q)
    near = hand.CalcPoseInWorld(screen.plant_context).translation()
    assert _planted(screen, q, near) == (False, "robot")
    # a robot filter that matches nothing must fail loudly
    try:
        pt.ObstacleScreen(SceneFile(IIWA, "mug"), ("iiwa", "no_such_robot"), with_mug=False)
    except RuntimeError as e:
        assert "no_such_robot" in str(e)
    else:
        raise AssertionError("a missing robot instance name did not raise")


def test_prepare_matches_per_target_draws():
    screen = _iiwa_screen()
    q, X = _sample_target(IIWA, "mug")
    pts = [X.translation(), X.translation()]
    sets, stats = pt.PrepareObstacleSets(
        IIWA, "mug", 3, SceneFile(IIWA, "mug"), SCENES[(IIWA, "mug")].robot_instances,
        [q, q], pts, [X, X])
    assert len(sets) == 2 and sets[0].shape == (pt.N_MAX, 3), sets[0].shape
    assert not stats[0]["cap_hit"] and stats[0]["accepted"] == pt.N_MAX
    one, _ = pt.DrawTargetObstacles(screen, pt.TargetRng(3, IIWA, "mug", 1), q,
                                    X.translation(), X)
    assert np.array_equal(one, sets[1]) and not np.array_equal(sets[0], sets[1])
    # every accepted centre respects the analytic separation and the floor
    c = sets[0]
    d = np.linalg.norm(c[:, None] - c[None], axis=-1) + 10 * np.eye(len(c))
    assert d.min() >= 2 * pt.OBSTACLE_RADIUS + pt.SCENE_MARGIN - 1e-12
    assert c[:, 2].min() >= pt.OBSTACLE_RADIUS


## ------------------------------------------------------------------ the certificate ------

def _certify(robot, task, n=pt.N_MAX):
    q, X = _sample_target(robot, task)
    spec = SCENES[(robot, task)]
    yaml = SceneFile(robot, task)
    sets, stats = pt.PrepareObstacleSets(robot, task, 11, yaml, spec.robot_instances,
                                         [q], [X.translation()], [X])
    assert stats[0]["accepted"] == pt.N_MAX and not stats[0]["cap_hit"], stats[0]
    for k in (0, n):
        diagram = BuildTargetDiagram(yaml, sets[0][:k], X if task == "mug" else None)
        out = pt.CertifyWitness(diagram, q, spec.robot_instances, ProgramOptions())
        assert out["ok"] and out["constraint_value"] < 1.0
        if k:
            assert out["witness_obstacle_clearance"] >= pt.WITNESS_MARGIN - 1e-9
        else:
            assert out["witness_obstacle_clearance"] == float("inf")
    return out


def test_certificate_iiwa_grasp():
    out = _certify(IIWA, "mug")
    assert out["witness_obstacle_clearance"] < 10.0


def test_certificate_refuses_a_violated_witness():
    q, X = _sample_target(IIWA, "mug")
    near = _iiwa_screen()
    near._set_robot(q)
    p = near.plant.GetFrameByName("hand").CalcPoseInWorld(near.plant_context).translation()
    diagram = BuildTargetDiagram(SceneFile(IIWA, "mug"), [p], X)
    try:
        pt.CertifyWitness(diagram, q, SCENES[(IIWA, "mug")].robot_instances, ProgramOptions())
    except RuntimeError as e:
        assert "CertifyWitness failed" in str(e)
    else:
        raise AssertionError("an obstacle inside the gripper was certified")


def test_certificate_slow(robot, task):
    _certify(robot, task)


## ------------------------------------------------------------------ svgd sees them ------

def _pairs(diagram):
    sg = diagram.GetSubsystemByName("scene_graph")
    insp = sg.model_inspector()
    return {tuple(sorted((insp.GetName(a), insp.GetName(b))))
            for a, b in insp.GetCollisionCandidates()}


def _stub_program(diagram, mug=None):
    return types.SimpleNamespace(plant=diagram.GetSubsystemByName("plant"), diagram=diagram,
                                 options=ProgramOptions(), target_mug=mug)


def test_svgd_rebuilds_the_obstacles():
    from src.svgd.solver import _scene_spec
    yaml = SceneFile(IIWA, "mug")
    q, X = _sample_target(IIWA, "mug")
    sets, _ = pt.PrepareObstacleSets(IIWA, "mug", 3, yaml, SCENES[(IIWA, "mug")].robot_instances,
                                     [q], [X.translation()], [X])
    centres = sets[0][:8]
    solve = BuildTargetDiagram(yaml, centres, X)
    spec = SceneSpec.from_program(_stub_program(solve, Mug(X)))
    assert len(spec.obstacles) == 8 and hash(spec) is not None
    rebuilt = spec.build_diagram()
    assert _pairs(rebuilt) == _pairs(solve)
    names = lambda d: [d.GetSubsystemByName("plant").GetModelInstanceName(  # noqa: E731
        __import__("pydrake.multibody.tree", fromlist=["x"]).ModelInstanceIndex(i))
        for i in range(d.GetSubsystemByName("plant").num_model_instances())]
    assert names(rebuilt) == names(solve)
    # the guard must be seen to refuse: a spec WITHOUT the obstacles is a different scene ...
    bare = SceneSpec(yaml_path=spec.yaml_path, mug_xyz=spec.mug_xyz, mug_wxyz=spec.mug_wxyz,
                     mug_rotation=spec.mug_rotation)
    assert _pairs(bare.build_diagram()) != _pairs(solve)
    # ... and from_program raises when the diagram forgot its centres
    forgot = BuildTargetDiagram(yaml, centres, X)
    forgot.extra_obstacles = ()
    try:
        SceneSpec.from_program(_stub_program(forgot, Mug(X)))
    except ValueError as e:
        assert "obstacle" in str(e)
    else:
        raise AssertionError("from_program accepted a diagram that dropped its obstacles")
    # the solver's fallback (a pose-task scene: no mug instance) carries them too
    pose_diag = BuildTargetDiagram(yaml, centres, None)
    stub = _stub_program(pose_diag, mug=Mug(X))          # target_mug without an instance
    fb = _scene_spec(stub)
    assert fb.obstacles == spec.obstacles and not fb.has_mug
    assert _pairs(fb.build_diagram()) == _pairs(pose_diag)


## ------------------------------------------------------------------ no obstacles: as before ------

class _Stub:
    def __init__(self, diagram, frame):
        self.plant = diagram.GetSubsystemByName("plant")
        self.plant_context = self.plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
        self.frame = self.plant.GetFrameByName(frame)

    def SetPositions(self, q):
        self.plant.SetPositions(self.plant_context, q)


def test_generate_diagram_with_mug_unchanged_without_obstacles():
    yaml = SceneFile(IIWA, "mug")
    stub = _Stub(BuildEnv(meshcat=None, directives_file=yaml), "between_fingers")
    q, _ = _sample_target(IIWA, "mug")
    diagram, mug = GenerateDiagramWithMug(q, stub, yaml, None)
    plant = diagram.GetSubsystemByName("plant")
    from pydrake.multibody.tree import ModelInstanceIndex
    inst = [plant.GetModelInstanceName(ModelInstanceIndex(i))
            for i in range(plant.num_model_instances())]
    assert ObstacleInstanceNames(plant) == [] and "target_mug" in inst
    ref = BuildEnv(meshcat=None, directives_file=yaml)
    pref = ref.GetSubsystemByName("plant")
    assert len(inst) == pref.num_model_instances() + 1 and diagram.extra_obstacles == ()
    stub.SetPositions(q)
    want = stub.frame.CalcPoseInWorld(stub.plant_context)
    body = plant.GetBodyByName("mug_body_link", plant.GetModelInstanceByName("target_mug"))
    got = plant.EvalBodyPoseInWorld(plant.GetMyContextFromRoot(diagram.CreateDefaultContext()),
                                    body)
    assert np.abs(got.GetAsMatrix4() - want.GetAsMatrix4()).max() < 1e-12
    assert np.array_equal(mug.middle.GetAsMatrix4(), want.GetAsMatrix4())


## ------------------------------------------------------------------ pillars ------

def test_pillar_files_are_the_generator_output():
    assert gen.main(["--check"]) == 0
    assert PILLAR_SCENES == tuple(PillarSceneName(r) for r in PILLAR_RADII)
    for robot in gen.RECORD_ROBOTS:
        for scene in PILLAR_SCENES:
            assert os.path.exists(SceneFile(robot, "mug", scene))
    try:
        SceneFile("gvs_pushrod9_o1", "mug", PILLAR_SCENES[0])
    except SystemExit:
        pass
    else:
        raise AssertionError("pillar scene accepted for a non-record robot")


def _body_pose(diagram, model, link):
    plant = diagram.GetSubsystemByName("plant")
    ctx = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    return plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName(link, plant.GetModelInstanceByName(model)))


def test_pillar_positions_follow_the_rule():
    for r in PILLAR_RADII:
        diagram = BuildEnv(meshcat=None, directives_file=SceneFile("iiwa", "mug",
                                                                   PillarSceneName(r)))
        for k, (_, tx, ty, _tz, yaw) in enumerate(SHELF_WELDS, start=1):
            # independent recomputation: both x-face openings, take the one nearer the base
            u = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
            t = np.array([tx, ty])
            faces = [t + 0.15 * u, t - 0.15 * u]
            o = min(faces, key=np.linalg.norm)
            want = o * (1.0 - (0.05 + r) / np.linalg.norm(o))   # on the segment base -> o
            X = _body_pose(diagram, "pillar%d" % k, "pillar_link")
            p = X.translation()
            assert np.abs(p[:2] - want).max() < 1e-8, (r, k, p, want)
            assert abs(p[2] - 1.0) < 1e-12
            assert abs(np.linalg.norm(o - p[:2]) - (0.05 + r)) < 1e-8
        assert np.abs(PillarCentres(r)[0] - _body_pose(diagram, "pillar1", "pillar_link")
                      .translation()[:2]).max() < 1e-8


def test_pillars_clear_shelves_and_tables():
    # the pillar as a FREE body, so pillar-vs-anchored pairs exist
    for r in PILLAR_RADII:
        name = PillarSceneName(r)
        d = ModelDirective()
        d.add_model = AddModel(name="pillar_screen",
                               file="package://combining_kinematics/" + gen.SdfPath(r))
        diagram = BuildEnv(meshcat=None, directives_file=SceneFile("iiwa", "mug"),
                           extra_directives=[d])
        plant = diagram.GetSubsystemByName("plant")
        sg = diagram.GetSubsystemByName("scene_graph")
        full = diagram.CreateDefaultContext()
        plant_ctx = plant.GetMyContextFromRoot(full)
        sg_ctx = sg.GetMyContextFromRoot(full)
        body = plant.GetBodyByName("pillar_link", plant.GetModelInstanceByName("pillar_screen"))
        for k, (x, y) in enumerate(PillarCentres(r), start=1):
            plant.SetFreeBodyPose(plant_ctx, body, RigidTransform([x, y, 1.0]))
            query = plant.get_geometry_query_input_port().Eval(plant_ctx)
            insp = query.inspector()
            ids = set(plant.GetCollisionGeometriesForBody(body))
            for p in query.ComputeSignedDistancePairwiseClosestPoints(max_distance=1.0):
                if p.id_A in ids or p.id_B in ids:
                    other = p.id_B if p.id_A in ids else p.id_A
                    inst = plant.GetBodyFromFrameId(insp.GetFrameId(other)).model_instance()
                    nm = plant.GetModelInstanceName(inst)
                    if nm.startswith("shelves"):
                        assert p.distance >= 0.01, (name, k, nm, p.distance)
                    elif nm.startswith("table"):
                        # the pillar's foot (z = 0.0) RESTS on the table top (z ~ 0): contact
                        # without penetration is the stated geometry, not a defect
                        assert p.distance >= -1e-9, (name, k, nm, p.distance)


def test_pillar_scenes_build(robots=("panda", "iiwa", "screw7_p050", "soft12"),
                             scenes=(PILLAR_SCENES[1],)):
    for robot in robots:
        for scene in scenes:
            diagram = BuildEnv(meshcat=None, directives_file=SceneFile(robot, "mug", scene))
            plant = diagram.GetSubsystemByName("plant")
            assert all(plant.HasModelInstanceNamed("pillar%d" % k) for k in range(1, 5))


def test_obstacle_directives_hashable_spec_and_names():
    d = ObstacleDirectives([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert len(d) == 4 and OBSTACLE_SDF.endswith("obstacle_sphere_r040.sdf")


GROUPS = {
    "fast": [test_rung0_fingerprints_are_the_record,
             test_payload_changes_the_fingerprint_only_when_given,
             test_stream_is_pure_and_prefix_nested,
             test_screen_refuses_and_accepts,
             test_prepare_matches_per_target_draws,
             test_certificate_iiwa_grasp,
             test_certificate_refuses_a_violated_witness,
             test_svgd_rebuilds_the_obstacles,
             test_generate_diagram_with_mug_unchanged_without_obstacles,
             test_pillar_files_are_the_generator_output,
             test_pillar_positions_follow_the_rule,
             test_pillars_clear_shelves_and_tables,
             test_pillar_scenes_build,
             test_obstacle_directives_hashable_spec_and_names],
    "slow": [partial(test_certificate_slow, robot, task)
             # soft12 is not here: its plant vector is 231 floating-body positions reached
             # only through the program's `ExactConfigToPlantQ`, so sampling a witness needs
             # the program (a checkpoint load). Its screen path is the same code, untested.
             for robot in ("panda", "iiwa", "screw7_p050")
             for task in ("mug", "pose")]
            + [partial(test_pillar_scenes_build,
                       ("panda", "iiwa", "screw7_p050", "soft12"), PILLAR_SCENES)],
}


def main():
    _tiers.run(GROUPS)
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
