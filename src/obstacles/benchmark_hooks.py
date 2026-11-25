"""The obstacle campaign's hooks into the four record benchmark scripts (one place, not four).

Rung 0 (`--obstacles none`, the default) is the record: every method here is then a no-op or
returns what the script computed before obstacles existed (`payload()` is None, `tag_tokens()`
is empty, `centres()` is `()`, `pose_diagram()` returns the shared diagram), so the scripts'
draws, grid_hash, scene_fingerprint, tags and diagrams are unchanged.  The one addition rung 0
makes is `metadata["obstacles"] == {"family": "none"}`.

Obstacles never touch the run's `rng` or the global numpy stream: `PrepareObstacleSets` draws
from per-target generators that are pure in (seed, robot, task, target).  Every shard prepares
and certifies EVERY target, so the obstacle metadata is identical across shards.
"""

import json
import os

import numpy as np

from src.obstacles import per_target as pt
from src.utils import BuildTargetDiagram, HiddenPrints, MugTargetPose


def add_arguments(parser):
    parser.add_argument("--obstacles", choices=("none", "per_target"), default="none",
                        help="`none` is the record scene. `per_target` welds N random spheres "
                             "near each target (family OBST; src/obstacles/per_target.py), "
                             "certified feasible by the target's generating configuration. "
                             "The grid is drawn first and is identical either way.")
    parser.add_argument("--obstacle-count", type=int, default=None, metavar="N",
                        help="spheres per target, one of %r (with --obstacles per_target; "
                             "rung 0 is `--obstacles none`)" % (pt.LADDER,))
    parser.add_argument("--prepare-only", action="store_true",
                        help="draw the grid, prepare and certify the obstacles, write "
                             "<log_dir>/prepare.json (incl. each guess's collision value on "
                             "its target's solve scene), and exit without solving or compiling")


def check_arguments(args):
    if args.obstacles == "none":
        if args.obstacle_count is not None:
            raise SystemExit("--obstacle-count needs --obstacles per_target "
                             "(rung 0 is --obstacles none)")
        return
    if args.obstacle_count is None or args.obstacle_count not in pt.LADDER:
        raise SystemExit("--obstacles per_target needs --obstacle-count in %r" % (pt.LADDER,))
    if args.obstacle_count == 0:
        raise SystemExit("--obstacle-count 0 is rung 0: use --obstacles none")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def CollisionValues(diagram, qs, options):
    """The program's own collision criterion (MinimumDistanceLowerBoundConstraint, the
    options' bound and offset) on `diagram` at each plant vector of `qs`; > 1 is in collision."""
    from pydrake.all import MinimumDistanceLowerBoundConstraint
    plant = diagram.GetSubsystemByName("plant")
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    constraint = MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=options.collision_bound,
        influence_distance_offset=options.collision_influence_offset,
        plant_context=plant_context)
    out = []
    for q in qs:
        q = np.asarray(q, dtype=np.float64)
        plant.SetPositions(plant_context, q)
        out.append(float(np.asarray(constraint.Eval(q)).flatten()[0]))
    return out


class ObstaclePlan:
    def __init__(self, args, robot_key, yaml_file, robot_instances, to_plant=None):
        """`robot_key` is the SCENES key ("panda", "iiwa", "screw7_p050", "soft12");
        `to_plant` maps a grid configuration to plant positions (identity on rigid arms)."""
        check_arguments(args)
        self.family = args.obstacles
        self.n = int(args.obstacle_count or 0)
        self.prepare_only = bool(args.prepare_only)
        self.robot_key, self.task, self.seed = robot_key, args.task, args.seed
        self.yaml_file, self.robot_instances = yaml_file, tuple(robot_instances)
        self.to_plant = to_plant or (lambda q: q)
        self.sets, self.stats, self.certificates = None, None, {}
        self.guess_collision_values = {}
        self._pose_diagrams = {}
        self.target_stats = None

    @property
    def active(self):
        return self.family == "per_target"

    ## ------------------------------------------------------------ preparation ------

    def set_grid(self, sampler, target_cfgs, guesses, point_of_plant_q, target_stats=None):
        """Called once the grid is drawn.  Records the plant-coordinate witnesses and, for
        per_target, draws every target's obstacle set (all targets, whatever the shard)."""
        self.target_stats = target_stats
        self.guesses = guesses
        self.q_stars = [np.asarray(self.to_plant(c), dtype=np.float64) for c in target_cfgs]
        if not self.active:
            return
        points = [np.asarray(point_of_plant_q(q).translation()) for q in self.q_stars]
        mug_poses = [MugTargetPose(q, sampler) if self.task == "mug" else None
                     for q in self.q_stars]
        with HiddenPrints():
            self.sets, self.stats = pt.PrepareObstacleSets(
                self.robot_key, self.task, self.seed, self.yaml_file, self.robot_instances,
                self.q_stars, points, mug_poses)

    def centres(self, ti):
        """Target `ti`'s first-N centres, or `()` at rung 0."""
        return () if not self.active else self.sets[ti][:self.n]

    def payload(self):
        return (pt.ObstacleFingerprintPayload(self.n, self.sets) if self.active else None)

    def tag_tokens(self):
        return ["obst%d" % self.n] if self.active else []

    def wanted(self, cells, n_targets):
        """Targets that need a solve diagram: this shard's (all, under --prepare-only)."""
        if cells is None or self.prepare_only:
            return set(range(n_targets))
        return {ti for ti, _ in cells}

    ## ------------------------------------------------------------ diagrams ------

    def on_diagram(self, ti, diagram, options):
        """Certify target `ti`'s ACTUAL solve diagram (obstacle scenes), and under
        --prepare-only measure its guesses' collision values.  Raises on a failed certificate."""
        if self.active:
            self.certificates[ti] = pt.CertifyWitness(
                diagram, self.q_stars[ti], self.robot_instances, options)
        if self.prepare_only:
            self.guess_collision_values[ti] = CollisionValues(
                diagram, [self.to_plant(g) for g in self.guesses[ti]], options)

    def pose_diagrams(self, shared_diagram, wanted, options):
        """Pose task: per-target obstacle diagrams (certified for every target); at rung 0
        the shared diagram is kept and nothing is built."""
        if not self.active:
            if self.prepare_only:
                for ti in range(len(self.q_stars)):
                    self.on_diagram(ti, shared_diagram, options)
            return
        for ti in range(len(self.q_stars)):
            with HiddenPrints():
                d = BuildTargetDiagram(self.yaml_file, self.centres(ti))
                self.on_diagram(ti, d, options)
            if ti in wanted:
                self._pose_diagrams[ti] = d

    def pose_diagram(self, ti, shared_diagram):
        return self._pose_diagrams[ti] if self.active else shared_diagram

    ## ------------------------------------------------------------ records ------

    def metadata(self):
        if not self.active:
            return {"family": "none"}
        n = self.n
        return dict(
            family=self.family, n=n, rule=pt.RULE_VERSION, radius=pt.OBSTACLE_RADIUS,
            region_radius=pt.REGION_RADIUS, witness_margin=pt.WITNESS_MARGIN,
            scene_margin=pt.SCENE_MARGIN, n_max=pt.N_MAX, ladder=list(pt.LADDER),
            achieved=[min(len(s), n) for s in self.sets],
            short_targets=[i for i, s in enumerate(self.sets) if len(s) < n],
            cap_hits=[i for i, st in enumerate(self.stats) if st["cap_hit"]],
            candidates=[st["candidates"] for st in self.stats],
            rejected=[st["rejected"] for st in self.stats],
            certificates=[self.certificates.get(i) for i in range(len(self.sets))],
            centres=[np.asarray(s[:n]).tolist() for s in self.sets])

    def write_prepare(self, log_dir, grid_hash, scene_fp):
        """--prepare-only's output: printed and written to <log_dir>/prepare.json."""
        report = dict(grid_hash=grid_hash, scene_fingerprint=scene_fp,
                      target_stats=self.target_stats, obstacles=self.metadata(),
                      guess_collision_values=[self.guess_collision_values.get(i)
                                              for i in range(len(self.q_stars))])
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "prepare.json")
        with open(path, "w") as fh:
            json.dump(report, fh, default=_json_default, indent=1)
        gv = [v for row in report["guess_collision_values"] if row for v in row]
        shown = dict(report, guess_collision_values="(%d values)" % len(gv),
                     obstacles={k: v for k, v in report["obstacles"].items() if k != "centres"})
        print(json.dumps(shown, default=_json_default))
        print("guesses starting in collision on their solve scene: %d of %d"
              % (sum(v > 1.0 for v in gv), len(gv)))
        print("wrote %s" % path)
        return report
