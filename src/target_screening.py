"""Screening a sampled target: is it inside a shelf, and does the object fit there?

The Drake half of the target hardening; `src/shelf_regions.py` is the geometry half.  Three
pieces, in dependency order:

* `SCENES` -- which YAML each (robot, task) builds, which model instances are the robot, and
  which frame's origin is the point a containment test applies to.
* `FloatingMugScreen` -- does a mug at this pose intersect the static scene?
* `SampleShelfTargets` -- the rejection-sampling loop both benchmark scripts share.

WHY THE PENETRATION SCREEN NEEDS ITS OWN DIAGRAM.  Drake's SceneGraph never generates
collision candidates between two ANCHORED geometries, and `GenerateDiagramWithMug` WELDS the
target mug into the scene.  So on the solve scene a mug-vs-shelf overlap is silently never
reported -- the mug could be halfway through a shelf board and nothing would say so.  The
screen therefore runs on a separate diagram carrying the mug as a FREE body, where
mug-vs-anchored pairs do get generated.  This is not redundant with the arm's own clearance
check: that one runs on the solve scene with no target mug present at all, and checks the arm.
"""

import os
from dataclasses import dataclass

from pydrake.geometry import SceneGraph  # noqa: F401  (documents what BuildEnv returns)
from pydrake.multibody.parsing import ModelDirective, AddModel

from src.shelf_regions import PointInShelfCompartments
from src.utils import BuildEnv, RepoDir

MUG_URDF = "package://combining_kinematics/models/mug/mug_simple_red.urdf"

## Past this many candidates rejected BACK TO BACK the sampler gives up rather than spinning
## for ever.  It is a TAIL BOUND, not a budget: acceptance restarts at each accepted target,
## so P(trip on one target) = (1 - p)^guard and a 60-target grid gets 60 chances to trip.
##
## `../codebase` uses 5000.  We deliberately do not: acceptance here is about an order of
## magnitude lower than it measured (0.23% of raw draws on iiwa pose at inset 0.10, against
## its 4.6% in-region), and at 5000 that row trips somewhere in a 60-target grid **41% of the
## time** -- and a trip raises partway through a queued run and kills every shard of it.
## Measured by scripts/probe_shelf_acceptance.py, 20000 draws per scene, P(trip over 60
## targets) at inset 0.10:
##
##     guard   panda/mug  panda/pose  iiwa/mug  iiwa/pose
##      5000     6.3e-05     1.2e-02   6.0e-04    4.1e-01
##     50000     0           0         0          0
##
## 50000 costs nothing -- it bounds only the tail, and the expected draw count per target is
## unchanged at ~400-1100 (~12 s of sampling per grid) -- so it is the default, and no
## manifest has to remember to raise it.
MAX_CONSECUTIVE_REJECTIONS = 50000


@dataclass(frozen=True)
class SceneSpec:
    """Everything about a (robot, task) scene that the target screening needs."""
    key: str
    hardened: str
    legacy: str
    ## Bin removed, clutter kept -- the scene that separates "targets must be in a shelf"
    ## from "the scene lost obstacles".  Only the iiwa needs one: the Panda GRASP scene had
    ## no decorative mugs to begin with, so its hardened scene already IS its nobin scene.
    ## None where the distinction does not exist.
    nobin: str
    ## EXACT model-instance names, not prefixes.  `../codebase` filters with str.startswith,
    ## which would also swallow a future `panda_table`; an exact set is testable, and
    ## tests/test_shelf_placement_screens.py asserts both that every name here exists and
    ## that the complement is exactly the static furniture.
    robot_instances: tuple
    ## The frame whose world origin a containment test applies to.  For the grasp task this
    ## is the grasp point, which is also where the mug is welded, so "mug centre in a
    ## compartment" and "target point in a compartment" are the same test.  For the pose task
    ## it is the frame the target pose IS.
    target_frame: str
    ## THE CONTAINMENT POINTS, and they are deliberately NOT `target_frame`.
    ##
    ## Rejection sampling must key on the same physical point on the hand for both robots or
    ## the two are not solving comparable problems.  `target_frame` fails that: the iiwa
    ## pose task targets `iiwa_link_7`, its arm FLANGE, 0.184 m behind `between_fingers`,
    ## while the Panda targets `panda_hand`, the GRIPPER MOUNT, 0.100 m behind it -- 84 mm
    ## apart, different points on the hand.  Both robots carry the same finray gripper, so
    ## the gripper's own frames are the consistent choice and are exact:
    ##
    ##     wrist      = the gripper base link -- `hand` (iiwa) / `panda_hand` (Panda),
    ##                  measured at exactly 0.100 m behind `between_fingers` on BOTH
    ##     fingertips = `between_fingers` itself, identical on both
    ##
    ## So the wrist/fingertip choice is one 0.1 m step along the gripper, the same step on
    ## each robot, and which of the two better expresses the task is a measurement.
    ##
    ## On the GRASP task both resolve to `between_fingers`: the mug is welded there, so the
    ## mug centre IS the grasp point, and a point 0.1 m behind it is not the mug.
    wrist_frame: str = None
    fingertip_frame: str = None


SCENES = {
    ("panda", "mug"): SceneSpec(
        "panda_mug",
        "models/panda/panda_finray_collision_hardened.yaml",
        "models/panda/panda_finray_collision.yaml",
        "models/panda/panda_finray_collision_hardened.yaml",
        ("panda", "finray"), "between_fingers",
        wrist_frame="between_fingers", fingertip_frame="between_fingers"),
    ## The Panda POSE scene is the Panda GRASP scene -- the same file, as the iiwa has always
    ## done. It used to be `panda_jrl.urdf`, the arm with its own stock Franka hand, on the
    ## reasoning that the flow was trained against jrl's Panda. That reasoning does not
    ## survive inspection: arm joints 1-7 and their limits are byte-identical between
    ## `panda_jrl.urdf` and `panda_no_hand.urdf`, the flow is conditioned on a frame welded
    ## to link 7, and `CalibrateFlowFrame` measures whatever offset the gripper introduces
    ## and checks it is constant. So the stock hand bought nothing and cost three things: a
    ## second gripper on one robot (making the pose task's collision geometry differ between
    ## robots for no reason), two prismatic finger joints that the sampler drew uniformly
    ## while every solve pinned them to 0.04, and no `between_fingers` frame, which forced
    ## the fingertip containment point to be a hardcoded TCP offset instead of a measured
    ## frame. Removed 2026-09-15 along with the stock hand itself.
    ("panda", "pose"): SceneSpec(
        "panda_pose",
        "models/panda/panda_finray_collision_hardened.yaml",
        "models/panda/panda_finray_collision.yaml",
        "models/panda/panda_finray_collision_hardened.yaml",
        ("panda", "finray"), "panda_hand",
        wrist_frame="panda_hand", fingertip_frame="between_fingers"),
    ("iiwa", "mug"): SceneSpec(
        "iiwa_mug",
        "models/iiwa14/iiwa14_collision_hardened.yaml",
        "models/iiwa14/iiwa14_collision.yaml",
        "models/iiwa14/iiwa14_collision_nobin.yaml",
        ("iiwa", "finray"), "between_fingers",
        wrist_frame="between_fingers", fingertip_frame="between_fingers"),
    ("iiwa", "pose"): SceneSpec(
        "iiwa_pose",
        "models/iiwa14/iiwa14_collision_hardened.yaml",
        "models/iiwa14/iiwa14_collision.yaml",
        "models/iiwa14/iiwa14_collision_nobin.yaml",
        ("iiwa", "finray"), "iiwa_link_7",
        wrist_frame="hand", fingertip_frame="between_fingers"),
}


def SceneFile(robot, task, scene="hardened"):
    """Absolute path to the scene YAML for this (robot, task) and scene mode.

    `hardened` drops the bin and the decorative mugs; `nobin` drops only the bin, keeping
    the clutter; `legacy` is the pre-2026-09-15 scene.  `nobin` raises where the robot has
    no such variant, rather than silently falling back to `hardened` -- a run labelled
    `nobin` that quietly measured `hardened` would be worse than no run at all.
    """
    spec = SCENES[(robot, task)]
    path = {"hardened": spec.hardened, "legacy": spec.legacy, "nobin": spec.nobin}[scene]
    if path is None:
        raise SystemExit(
            "--scene nobin: %s/%s has no bin-free-with-clutter variant. The Panda grasp "
            "scene never had decorative mugs, so its hardened scene already is its nobin "
            "scene; the Panda pose scene has no variant built." % (robot, task))
    return os.path.join(RepoDir(), path)


class FloatingMugScreen:
    """Does a mug at a given pose penetrate anything in the scene that is not the robot?

    Builds the scene once with `num_mugs` UNWELDED mugs appended in memory (an `add_model`
    with no matching `add_weld` leaves a free body), then reuses it per query.  Mug-vs-robot
    contact is expected and ignored: the mug is placed AT the gripper, and the screen diagram
    holds the robot at whatever configuration the scene defaults to, which is not the
    configuration the target came from.  Only mug-vs-static and mug-vs-mug matter here.
    """

    def __init__(self, directives_file, robot_instances, num_mugs=1):
        extra = []
        for k in range(num_mugs):
            d = ModelDirective()
            d.add_model = AddModel(name="mug_screen_%d" % k, file=MUG_URDF)
            extra.append(d)
        self.diagram = BuildEnv(meshcat=None, directives_file=directives_file,
                                extra_directives=extra)
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.scene_graph = self.diagram.GetSubsystemByName("scene_graph")
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.scene_graph_context = self.scene_graph.GetMyContextFromRoot(self.context)
        self.mug_bodies = [
            self.plant.GetBodyByName("mug_body_link",
                                     self.plant.GetModelInstanceByName("mug_screen_%d" % k))
            for k in range(num_mugs)]
        missing = [n for n in robot_instances if not self.plant.HasModelInstanceNamed(n)]
        if missing:
            raise RuntimeError(
                "FloatingMugScreen: scene %s has no model instance(s) %r -- the robot filter "
                "would then reject every candidate as 'penetrating'."
                % (os.path.basename(directives_file), missing))
        self._robot_instances = {self.plant.GetModelInstanceByName(n)
                                 for n in robot_instances}

    def Penetrates(self, poses):
        """True if any supplied mug pose penetrates non-robot geometry.

        Unused mug slots are parked far away rather than left at the origin, where they would
        sit inside the robot's base and inside each other.
        """
        for k, body in enumerate(self.mug_bodies):
            X = poses[k] if k < len(poses) else None
            if X is None:
                from pydrake.math import RigidTransform
                X = RigidTransform([0.0, 0.0, 1000.0 + 2.0 * k])
            self.plant.SetFreeBodyPose(self.plant_context, body, X)
        query = self.plant.get_geometry_query_input_port().Eval(self.plant_context)
        inspector = query.inspector()
        for pair in query.ComputePointPairPenetration():
            a = self.plant.GetBodyFromFrameId(inspector.GetFrameId(pair.id_A))
            b = self.plant.GetBodyFromFrameId(inspector.GetFrameId(pair.id_B))
            if (a.model_instance() in self._robot_instances
                    or b.model_instance() in self._robot_instances):
                continue
            return True
        return False


def SampleShelfTargets(n, draw, collision_free, target_pose, regions=None, screen=None,
                       max_consecutive_rejections=MAX_CONSECUTIVE_REJECTIONS,
                       label="", progress=None):
    """Draw `n` targets, rejecting until each is collision-free and (optionally) in a shelf.

    `draw()` returns a candidate configuration -- the caller passes a closure over its OWN
    generator, so the random stream stays where the benchmark scripts manage it.
    `collision_free(q)` and `target_pose(q)` are likewise the caller's, so this function needs
    no plant and no program.  `regions=None` disables containment, `screen=None` disables the
    penetration screen; with both disabled the loop consumes exactly one `draw()` per
    iteration and therefore reproduces the pre-hardening stream bit for bit, which is what
    keeps `--scene legacy --target-placement free` able to reproduce an archived grid.

    The loop is FLAT -- collision rejections and placement rejections share one counter --
    because that is the quantity the acceptance probe measures and the only one the guard's
    arithmetic is valid for.

    Returns (qs, stats).
    """
    qs = []
    stats = dict(drawn=0, collision_rejected=0, containment_rejected=0,
                 penetration_rejected=0, accepted=0)
    consecutive = 0
    while len(qs) < n:
        q = draw()
        stats["drawn"] += 1
        rejected = None
        if not collision_free(q):
            rejected = "collision_rejected"
        elif regions is not None:
            X = target_pose(q)
            if not PointInShelfCompartments(X.translation(), regions):
                rejected = "containment_rejected"
            elif screen is not None and screen(X):
                rejected = "penetration_rejected"
        if rejected is not None:
            stats[rejected] += 1
            consecutive += 1
            if consecutive > max_consecutive_rejections:
                raise RuntimeError(
                    "%s: %d consecutive candidates rejected (%d on collision, %d outside "
                    "every shelf compartment, %d penetrating the scene). Accepted %d of %d "
                    "targets from %d draws. Either the inset is too deep for this robot or "
                    "the guard is too tight -- see scripts/probe_shelf_acceptance.py, which "
                    "prints the trip probability for each inset."
                    % (label or "SampleShelfTargets", consecutive,
                       stats["collision_rejected"], stats["containment_rejected"],
                       stats["penetration_rejected"], len(qs), n, stats["drawn"]))
            continue
        consecutive = 0
        stats["accepted"] += 1
        qs.append(q)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    stats["accept_rate"] = stats["accepted"] / max(stats["drawn"], 1)
    survived = stats["drawn"] - stats["collision_rejected"]
    stats["accept_rate_of_collision_free"] = stats["accepted"] / max(survived, 1)
    return qs, stats


def FormatTargetStats(label, stats):
    """One line for the job log, so a viability problem is visible rather than inferred."""
    return ("%s target sampling: %d accepted from %d draws (%.2f%%; %.2f%% of collision-free) "
            "-- rejected %d on collision, %d outside a shelf, %d penetrating"
            % (label, stats["accepted"], stats["drawn"], 100 * stats["accept_rate"],
               100 * stats["accept_rate_of_collision_free"], stats["collision_rejected"],
               stats["containment_rejected"], stats["penetration_rejected"]))


def ContainmentPose(plant, plant_context, spec, placement_point="wrist"):
    """A callable q -> RigidTransform whose ORIGIN is what containment tests.

    Keyed on the GRIPPER, not on the task's target frame, so the same physical point on the
    hand is used for both robots -- see SceneSpec for why the target frame fails that.
    `wrist` is the gripper base, 0.100 m behind the fingers on both robots; `fingertips` is
    `between_fingers`.  On the grasp task the two coincide at the mug centre.

    The caller still hands the FULL target pose to the penetration screen, and the target the
    arms are given is untouched either way: this is a sampling filter, not a redefinition of
    the task.
    """
    frame_name = {"wrist": spec.wrist_frame,
                  "fingertips": spec.fingertip_frame}.get(placement_point)
    if frame_name is None:
        raise SystemExit("unknown or unavailable placement point %r for %s"
                         % (placement_point, spec.key))
    frame = plant.GetFrameByName(frame_name)

    def pose_of(q):
        plant.SetPositions(plant_context, q)
        return frame.CalcPoseInWorld(plant_context)
    return pose_of, frame_name
