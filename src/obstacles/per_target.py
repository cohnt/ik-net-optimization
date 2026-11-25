"""Family OBST: random spherical obstacles near each target, certified feasible by a witness.

Every target comes from a collision-free generating configuration `q*` (the WITNESS).  For
each target we draw up to `N_MAX` sphere centres uniformly in a ball of radius `REGION_RADIUS`
about the target point and keep a candidate only if the witness stays feasible with it:

* at least `WITNESS_MARGIN` from every ROBOT geometry (gripper included) at `q*`;
* at least `SCENE_MARGIN` from every non-robot geometry (shelves, tables, the target mug);
* at least `2 * OBSTACLE_RADIUS + SCENE_MARGIN` from every previously accepted obstacle
  (centre distance; analytic, sphere against sphere);
* centre z >= `OBSTACLE_RADIUS` (nothing under the work surface).

So q* solves the problem with every obstacle present: the instance is feasible by
construction, and the benchmark's own `MinimumDistanceLowerBoundConstraint` (bound 1e-3) is
satisfied at q* with `WITNESS_MARGIN - 1e-3` to spare.  Rung n of `LADDER` is the first n
accepted obstacles, so rungs are prefix-nested; rung 0 is no obstacles and reproduces the
record exactly (same scene, same draws, same fingerprint -- `extra_payload=None`).

THE STREAM IS A PURE FUNCTION of (seed, robot, task, target_index): one generator per target,
and each candidate consumes a FIXED block (3 standard normals for the direction, 1 uniform for
the radius fraction) whatever its fate.  Rejections, sharding and the other targets cannot
move it.  Drake never generates collision candidates between two ANCHORED geometries, so the
screen uses a FREE sphere (and a FREE mug) exactly as `FloatingMugScreen` does; the welded
mug and the welded obstacles of the real solve scene are never tested against each other by
Drake, which is why the screen, not the solve scene, is what keeps them apart.
"""

import zlib

import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import AddModel, ModelDirective

from src.obstacles.directives import OBSTACLE_SDF, ObstacleInstanceNames

## Sphere radius of `models/assets/obstacle_sphere_r040.sdf` (metres).
OBSTACLE_RADIUS = 0.04
## Centres are uniform in a ball of this radius about the target point (metres).
REGION_RADIUS = 0.30
## Minimum signed distance from a candidate to any ROBOT geometry at q* (metres).
WITNESS_MARGIN = 0.03
## Minimum signed distance to any non-robot geometry and to earlier obstacles (metres).
SCENE_MARGIN = 0.01
## Rungs: the number of obstacles per target.  Rung n is the first n accepted.
LADDER = (0, 8, 16, 32, 64)
N_MAX = 64
## The cap on candidates per target is N_MAX * CANDIDATES_PER_OBSTACLE.
CANDIDATES_PER_OBSTACLE = 1000
RULE_VERSION = "obst-v1"
## "obst" in ASCII; the second word of every target's seed sequence.
TAG = 0x6F627374

MUG_URDF = "package://combining_kinematics/models/mug/mug_simple_red.urdf"
_PARK = RigidTransform([0.0, 0.0, 1000.0])
_SPHERE_NAME = "obstacle_screen"
_MUG_NAME = "target_mug_screen"


def TargetRng(seed, robot, task, target_index):
    rid = zlib.crc32(f"{robot}/{task}".encode())
    return np.random.default_rng([int(seed), TAG, rid, int(target_index)])


class ObstacleScreen:
    """Does a sphere at `centre` keep the witness `q*` feasible?  Built once per run."""

    def __init__(self, directives_file, robot_instances, with_mug):
        from src.utils import BuildEnv
        extra = []
        d = ModelDirective()
        d.add_model = AddModel(name=_SPHERE_NAME, file=OBSTACLE_SDF)
        extra.append(d)
        if with_mug:
            d = ModelDirective()
            d.add_model = AddModel(name=_MUG_NAME, file=MUG_URDF)
            extra.append(d)
        self.diagram = BuildEnv(meshcat=None, directives_file=directives_file,
                                extra_directives=extra)
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        plant = self.plant
        missing = [n for n in robot_instances if not plant.HasModelInstanceNamed(n)]
        if missing:
            raise RuntimeError(
                "ObstacleScreen: scene %s has no model instance(s) %r -- the robot filter "
                "would match nothing." % (directives_file, missing))
        self._robot = {plant.GetModelInstanceByName(n) for n in robot_instances}
        sphere_inst = plant.GetModelInstanceByName(_SPHERE_NAME)
        self._sphere_body = plant.GetBodyByName("sphere_link", sphere_inst)
        self._mug_body = (plant.GetBodyByName("mug_body_link",
                                              plant.GetModelInstanceByName(_MUG_NAME))
                          if with_mug else None)
        self._sphere_ids = set(plant.GetCollisionGeometriesForBody(self._sphere_body))
        if not self._sphere_ids:
            raise RuntimeError("ObstacleScreen: the screen sphere has no collision geometry")
        self._mug_instance = (plant.GetModelInstanceByName(_MUG_NAME) if with_mug else None)
        ## Positions that belong to the robot: everything except the free bodies' 7 each, in
        ## the program's own relative order.
        free = [self._sphere_body] + ([self._mug_body] if with_mug else [])
        drop = set()
        for b in free:
            s = b.floating_positions_start()
            drop.update(range(s, s + 7))
        self._q_idx = np.array([i for i in range(plant.num_positions()) if i not in drop])

    def _set_robot(self, q_star):
        q_star = np.asarray(q_star, dtype=np.float64)
        if len(q_star) != len(self._q_idx):
            raise ValueError("ObstacleScreen: q* has %d entries, the scene's robot has %d"
                             % (len(q_star), len(self._q_idx)))
        x = np.array(self.plant.GetPositions(self.plant_context))
        x[self._q_idx] = q_star
        self.plant.SetPositions(self.plant_context, x)

    def Screen(self, q_star, centre, mug_pose=None):
        """(accepted, reason).  Reasons: floor, robot, mug, scene (obstacle-vs-obstacle is the
        caller's analytic test)."""
        centre = np.asarray(centre, dtype=np.float64)
        if centre[2] < OBSTACLE_RADIUS:
            return False, "floor"
        plant, ctx = self.plant, self.plant_context
        self._set_robot(q_star)
        plant.SetFreeBodyPose(ctx, self._sphere_body, RigidTransform(centre))
        if self._mug_body is not None:
            plant.SetFreeBodyPose(ctx, self._mug_body,
                                  _PARK if mug_pose is None else mug_pose)
        query = plant.get_geometry_query_input_port().Eval(ctx)
        inspector = query.inspector()
        pairs = query.ComputeSignedDistancePairwiseClosestPoints(
            max_distance=WITNESS_MARGIN + 0.05)
        reason = None
        for p in pairs:
            if p.id_A in self._sphere_ids:
                other = p.id_B
            elif p.id_B in self._sphere_ids:
                other = p.id_A
            else:
                continue
            inst = plant.GetBodyFromFrameId(inspector.GetFrameId(other)).model_instance()
            if inst in self._robot:
                if p.distance < WITNESS_MARGIN:
                    return False, "robot"
            elif inst == self._mug_instance:
                if p.distance < SCENE_MARGIN:
                    reason = "mug"
            elif p.distance < SCENE_MARGIN and reason is None:
                reason = "scene"
        if reason is not None:
            return False, reason
        return True, "accepted"


def _candidate(rng, target_point):
    """One fixed-size block of draws: 3 normals + 1 uniform."""
    g = rng.standard_normal(3)
    u = rng.uniform()
    n = np.linalg.norm(g)
    direction = g / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    return np.asarray(target_point, dtype=np.float64) + REGION_RADIUS * u ** (1.0 / 3.0) * direction


def DrawTargetObstacles(screen, rng, q_star, target_point, mug_pose):
    """Up to N_MAX accepted centres [k, 3] float64 (short when the cap is hit) and stats."""
    cap = N_MAX * CANDIDATES_PER_OBSTACLE
    centres = []
    stats = dict(candidates=0, accepted=0, rejected={}, cap_hit=False)
    min_sep = 2.0 * OBSTACLE_RADIUS + SCENE_MARGIN
    while len(centres) < N_MAX:
        if stats["candidates"] >= cap:
            stats["cap_hit"] = True
            break
        c = _candidate(rng, target_point)
        stats["candidates"] += 1
        if centres and np.min(np.linalg.norm(np.asarray(centres) - c, axis=1)) < min_sep:
            ok, reason = False, "obstacle"
        else:
            ok, reason = screen.Screen(q_star, c, mug_pose)
        if ok:
            centres.append(c)
        else:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
    stats["accepted"] = len(centres)
    return np.asarray(centres, dtype=np.float64).reshape(-1, 3), stats


def PrepareObstacleSets(robot, task, seed, yaml_file, robot_instances, target_qs,
                        target_points, mug_poses):
    """For ALL targets: (list of [<=N_MAX, 3] centre arrays, list of stats).  Geometry only;
    every shard calls it for every target, so shards agree by construction."""
    screen = ObstacleScreen(yaml_file, robot_instances, with_mug=(task == "mug"))
    sets, stats = [], []
    for i, (q, pt, mp) in enumerate(zip(target_qs, target_points, mug_poses)):
        c, s = DrawTargetObstacles(screen, TargetRng(seed, robot, task, i), q, pt, mp)
        sets.append(c)
        stats.append(s)
    return sets, stats


def ObstacleFingerprintPayload(n, centres_per_target):
    """Bytes for `scene_fingerprint(extra_payload=...)`: rule version, constants, n, and every
    target's first-n centres (float64) in target order."""
    head = ("%s|r=%r|region=%r|witness=%r|scene=%r|nmax=%d|cands=%d|ladder=%r|n=%d|T=%d"
            % (RULE_VERSION, OBSTACLE_RADIUS, REGION_RADIUS, WITNESS_MARGIN, SCENE_MARGIN,
               N_MAX, CANDIDATES_PER_OBSTACLE, LADDER, int(n), len(centres_per_target)))
    out = [head.encode()]
    for c in centres_per_target:
        c = np.ascontiguousarray(np.asarray(c, dtype=np.float64).reshape(-1, 3)[:n])
        out.append(b"|%d|" % len(c) + c.tobytes())
    return b"".join(out)


def CertifyWitness(diagram, q_star, robot_instances, options):
    """On the ACTUAL solve diagram: the sampler's own criterion holds at q* and every
    robot-vs-obstacle pair clears WITNESS_MARGIN.  Returns a dict; raises RuntimeError if not."""
    from pydrake.all import MinimumDistanceLowerBoundConstraint
    plant = diagram.GetSubsystemByName("plant")
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, np.asarray(q_star, dtype=np.float64))
    constraint = MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=options.collision_bound,
        influence_distance_offset=options.collision_influence_offset,
        plant_context=plant_context)
    value = float(np.asarray(constraint.Eval(np.asarray(q_star, dtype=np.float64))).flatten()[0])
    missing = [n for n in robot_instances if not plant.HasModelInstanceNamed(n)]
    if missing:
        raise RuntimeError("CertifyWitness: no model instance(s) %r" % (missing,))
    robot = {plant.GetModelInstanceByName(n) for n in robot_instances}
    obst = {plant.GetModelInstanceByName(n) for n in ObstacleInstanceNames(plant)}
    clearance = float("inf")
    if obst:
        sg = diagram.GetSubsystemByName("scene_graph")
        sg_ctx = sg.GetMyContextFromRoot(context)
        query = sg.get_query_output_port().Eval(sg_ctx)
        inspector = query.inspector()
        for p in query.ComputeSignedDistancePairwiseClosestPoints(max_distance=10.0):
            a = plant.GetBodyFromFrameId(inspector.GetFrameId(p.id_A)).model_instance()
            b = plant.GetBodyFromFrameId(inspector.GetFrameId(p.id_B)).model_instance()
            if (a in obst and b in robot) or (b in obst and a in robot):
                clearance = min(clearance, float(p.distance))
    ok = value < 1.0 and clearance >= WITNESS_MARGIN - 1e-9
    result = dict(constraint_value=value, witness_obstacle_clearance=clearance, ok=bool(ok))
    if not ok:
        raise RuntimeError(
            "CertifyWitness failed: collision constraint value %.6g (need < 1), nearest "
            "robot-obstacle clearance %.6g m (need >= %g) -- the obstacle set does not "
            "keep the generating configuration feasible." % (value, clearance, WITNESS_MARGIN))
    return result
