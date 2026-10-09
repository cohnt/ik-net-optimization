#!/usr/bin/env python3
"""Does the target mug, welded where `GenerateDiagramWithMug` puts it, clear the gripper?

`GenerateDiagramWithMug` welds the mug at the FULL pose of `between_fingers` at the generating
configuration q*, handle along the frame's +x (three r = 9 mm cylinders reaching x = 90 mm).
Whether the handle lands in the finger gap or inside a finger plate is therefore decided by the
YAW of that frame in the gripper's SDF -- and nothing downstream can catch a bad one: the
sampler's `collision_free` runs on the scene WITHOUT the mug, `FloatingMugScreen` filters the
robot out by design, and Drake never reports anchored-vs-anchored contact, which is what a
welded mug against a welded-to-the-arm finger is on the solve scene.

So this probe measures it directly, per robot scene: draw collision-free q*, weld the mug
exactly as the benchmark does (through `GenerateDiagramWithMug` itself, on a stub program),
and evaluate the mug scene's own `MinimumDistanceLowerBoundConstraint` at q* beside the true
signed distance between the mug and every robot body.

    python scripts/probe_mug_contact.py                       # every scene with a local model
    python scripts/probe_mug_contact.py --robots iiwa,panda --draws 150

Found 2026-10-08: the wsg finray's frame had yaw 0, pointing the handle INTO the right finger
by 18.0 mm on 150/150 draws of the iiwa scene, while the Panda's finray frame (yaw 1.57)
points it through the gap with +9.5 mm to spare. The wsg frame now carries the same yaw; this
probe is the check that it stays so on every robot carrying that gripper.
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint

from src.generic_program import ProgramOptions
from src.gvs_arm.params import RUNGS as _GVS_RUNGS
from src.screw_arm.limits import ApplyScrewJointLimits, RequireFiniteLimits
from src.screw_arm.params import SPECS as SCREW_SPECS
from src.soft_arm.params import RUNGS as _SOFT_RUNGS
from src.target_screening import SCENES, SceneFile
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints

DEFAULT_ROBOTS = "panda,iiwa,soft12,screw7_p050,gvs_pushrod9_o1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robots", default=DEFAULT_ROBOTS,
                   help="comma-separated robot keys of src/target_screening.py's SCENES")
    p.add_argument("--draws", type=int, default=150,
                   help="collision-free generating configurations per scene")
    p.add_argument("--scene", default="hardened", choices=("hardened", "nobin", "legacy"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class _StubProgram:
    """The three attributes `GenerateDiagramWithMug` reads, and nothing else."""

    def __init__(self, plant, context):
        self.plant = plant
        self.plant_context = context
        self.frame = plant.GetFrameByName("between_fingers")

    def SetPositions(self, q):
        self.plant.SetPositions(self.plant_context, q)


def make_drawer(robot, plant, rng):
    """q -> plant positions, drawn the way scripts/probe_shelf_acceptance.py draws them."""
    soft_spec = _SOFT_RUNGS.get(robot)
    gvs_spec = _GVS_RUNGS.get(robot)
    if gvs_spec is not None:
        from src.gvs_arm.model import GetModel, PlantSlotMap as GvsPlantSlotMap
        gvs_model = GetModel(gvs_spec)
        picks = GvsPlantSlotMap(plant, gvs_spec)
        return lambda: gvs_model.PlantQ(rng.uniform(-1.0, 1.0, size=gvs_spec.ninputs))[picks]
    if soft_spec is not None:
        import torch
        from src.soft_arm import kinematics as SK
        picks = SK.PlantSlotMap(plant, soft_spec)

        def draw():
            cfg = rng.uniform(-1.0, 1.0, size=soft_spec.ndof)
            tensor = torch.as_tensor(cfg, dtype=torch.float64, device="cpu")
            return SK.config_to_plant_q(tensor, soft_spec, device="cpu").numpy()[picks]
        return draw
    lower, upper = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    return lambda: rng.uniform(lower, upper)


def probe_scene(robot, draws, seed, scene):
    spec = SCENES[(robot, "mug")]
    yaml_file = SceneFile(robot, "mug", scene)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=yaml_file)
    plant = diagram.GetSubsystemByName("plant")
    context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    if robot in SCREW_SPECS:
        ## Drake's parsers discard a screw joint's <limit>; the benchmark repairs it inside
        ## the program's __init__, which this probe never constructs.
        ApplyScrewJointLimits(plant, SCREW_SPECS[robot])
    if robot not in _SOFT_RUNGS and robot not in _GVS_RUNGS:
        ## The continuum arms' plants carry quaternion floating bodies with +-inf limits by
        ## construction; their configuration is drawn in strain / force space instead.
        RequireFiniteLimits(plant, f"{robot}/mug")

    options = ProgramOptions()
    collision = MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=options.collision_bound,
        influence_distance_offset=options.collision_influence_offset, plant_context=context)
    rng = np.random.default_rng(seed)
    draw = make_drawer(robot, plant, rng)
    stub = _StubProgram(plant, context)

    values, distances, closest = [], [], Counter()
    drawn, start = 0, time.time()
    while len(values) < draws:
        q = draw()
        drawn += 1
        if collision.Eval(q) >= 1:
            continue
        with HiddenPrints():
            mug_diagram, _ = GenerateDiagramWithMug(q, stub, yaml_file, None)
        mplant = mug_diagram.GetSubsystemByName("plant")
        mcontext = mplant.GetMyContextFromRoot(mug_diagram.CreateDefaultContext())
        ## The welded mug carries no positions, so q* indexes the mug scene as it did the
        ## solve scene -- exactly the assumption the benchmark makes when it solves on it.
        mug_collision = MinimumDistanceLowerBoundConstraint(
            plant=mplant, bound=options.collision_bound,
            influence_distance_offset=options.collision_influence_offset,
            plant_context=mcontext)
        values.append(float(np.asarray(mug_collision.Eval(q)).ravel()[0]))
        mplant.SetPositions(mcontext, q)
        query = mplant.get_geometry_query_input_port().Eval(mcontext)
        inspector = query.inspector()
        mug_body = mplant.GetBodyByName(
            "mug_body_link", mplant.GetModelInstanceByName("target_mug"))
        robot_instances = {mplant.GetModelInstanceByName(n) for n in spec.robot_instances}
        best, best_name = np.inf, None
        for sd in query.ComputeSignedDistancePairwiseClosestPoints(1.0):
            a = mplant.GetBodyFromFrameId(inspector.GetFrameId(sd.id_A))
            b = mplant.GetBodyFromFrameId(inspector.GetFrameId(sd.id_B))
            if mug_body.index() not in (a.index(), b.index()):
                continue
            other = b if a.index() == mug_body.index() else a
            if other.model_instance() not in robot_instances:
                continue
            if sd.distance < best:
                best, best_name = sd.distance, other.name()
        distances.append(best)
        closest[best_name] += 1
    return dict(values=np.array(values), distances=np.array(distances), closest=closest,
                drawn=drawn, seconds=time.time() - start)


def main():
    args = parse_args()
    print("Mug welded at between_fingers(q*) as GenerateDiagramWithMug does; the mug scene's "
          "MinimumDistanceLowerBoundConstraint(bound=%g, influence=%g) at q*, and the true "
          "signed distance from the mug to the nearest ROBOT body. `y >= 1` is contact or "
          "penetration -- what a solve on that scene starts from, and what the sampler's "
          "collision_free (no mug) and FloatingMugScreen (robot filtered out) cannot see."
          % (ProgramOptions().collision_bound, ProgramOptions().collision_influence_offset))
    header = ("%-18s %6s %9s %10s %10s %10s  %s"
              % ("robot", "draws", "frac y>=1", "median mm", "min mm", "max mm", "closest body"))
    print(header)
    print("-" * len(header))
    for robot in args.robots.split(","):
        robot = robot.strip()
        if (robot, "mug") not in SCENES:
            print("%-18s   no grasp scene in SCENES -- skipped" % robot)
            continue
        try:
            r = probe_scene(robot, args.draws, args.seed, args.scene)
        except Exception as exc:  # a missing asset, not a measurement
            print("%-18s   could not build: %s" % (robot, str(exc).splitlines()[0][:100]))
            continue
        frac = float(np.mean(r["values"] >= 1.0))
        d = 1e3 * r["distances"]
        top = ", ".join("%s %d" % (k, v) for k, v in r["closest"].most_common(3))
        print("%-18s %6d %9.3f %10.1f %10.1f %10.1f  %s"
              % (robot, len(d), frac, np.median(d), d.min(), d.max(), top))
        print("%-18s   %d raw draws, %.0f s" % ("", r["drawn"], r["seconds"]))


if __name__ == "__main__":
    main()
