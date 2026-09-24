"""The soft arm's kinematics, and the Drake model's agreement with them.

Each test can only fail for its own reason, so a failure names its own source:

* `test_discretization_is_exact` -- the claim the whole model rests on. A segment carries a
  constant twist, so `exp(xi*L) == exp(xi*L/K)**K`; if that ever stopped holding, K would
  silently become an accuracy knob instead of a collision-resolution knob.
* `test_matches_closed_form_arc` -- a bending segment IS a circular arc, so it can be
  checked against a formula written independently of the implementation.
* `test_gradients_finite_at_zero_strain` -- the trap. `exp` carries sin(t)/t, (1-cos t)/t^2
  and (t-sin t)/t^3, and t = 0 is the straight arm: the origin of configuration space, the
  centre of the joint-centering cost, and the padding `InvertFlow` writes. The first
  version of this module returned NaN gradients there while returning correct VALUES, so a
  test on values alone would have passed.
* `test_drake_matches_torch` -- the generated SDF and the torch map are two descriptions of
  one robot. Tolerance is 1e-12, not 1e-6: both are float64 compositions of the same
  matrices, so anything looser would be hiding a disagreement rather than allowing for
  numerics.
* `test_spheres_contain_the_backbone` -- one-sided, and that is the point. The spheres ARE
  the robot's declared geometry; what has to be true is that they CONTAIN the rod, not that
  they approximate it.
* `test_collision_filters_are_what_they_claim` -- a filter that matches nothing is
  indistinguishable from a correct one until a campaign is burnt on it, which is the lesson
  `FloatingMugScreen` already carries.
* `test_generated_sdf_is_current` -- the generator and its committed output cannot drift.

No pytest config in this repo, so these are plain `test_*` functions with a `__main__`
driver: `.venv/bin/python tests/test_soft_arm_kinematics.py`.
"""

import dataclasses
import itertools
import math
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder,
                         MinimumDistanceLowerBoundConstraint, Parser)

from src.soft_arm import kinematics as K
from src.soft_arm.generate_sdf import OutputPath, RenderSdf
from src.soft_arm.params import RUNGS, SOFT12
from src.utils import RepoDir

DTYPE = torch.float64


def _configurations(spec, count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, spec.ndof, generator=generator, dtype=DTYPE) * 2 - 1


def _build(spec):
    """The bare arm, welded at the origin -- no scene, so this tests the model alone."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    parser.package_map().AddPackageXml(os.path.join(RepoDir(), "package.xml"))
    parser.AddModels(OutputPath(spec, RepoDir()))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.Finalize()
    diagram = builder.Build()
    context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    return plant, scene_graph, context


def test_discretization_is_exact():
    for spec in RUNGS.values():
        configurations = _configurations(spec, 8)
        reference = K.forward_kinematics(configurations, spec)
        worst = 0.0
        for sublinks in (1, 2, 3, 4, 6, 8, 12, 20):
            varied = dataclasses.replace(spec, sublinks_per_segment=sublinks)
            error = (K.forward_kinematics(configurations, varied) - reference).abs().max()
            worst = max(worst, float(error))
        assert worst < 1e-13, f"{spec.name}: tip moved by {worst:.3e} when K changed"
        print(f"PASS {spec.name}: tip pose is K-independent to {worst:.1e}")


def test_matches_closed_form_arc():
    """One segment bending about +x sweeps a circle of radius 1/kappa in the y-z plane."""
    spec = dataclasses.replace(SOFT12, name="one_segment", num_segments=1, total_length=0.2)
    length = spec.total_length
    for kappa in (0.0, 0.5, 3.0, 8.5):
        configuration = torch.zeros(spec.ndof, dtype=DTYPE)
        configuration[0] = kappa / spec.strain_limits[0]
        pose = K.forward_kinematics(configuration, spec)
        if kappa == 0.0:
            expected = torch.tensor([0.0, 0.0, length], dtype=DTYPE)
        else:
            radius, angle = 1.0 / kappa, kappa * length
            expected = torch.tensor(
                [0.0, -radius * (1 - math.cos(angle)), radius * math.sin(angle)], dtype=DTYPE)
        error = float((pose[:3] - expected).abs().max())
        bend = 2 * math.acos(min(1.0, abs(float(pose[3]))))
        assert error < 1e-14, f"kappa={kappa}: position off by {error:.3e}"
        assert abs(bend - kappa * length) < 1e-12, f"kappa={kappa}: bent {bend} rad"
    print("PASS a bending segment reproduces the closed-form circular arc to 1e-14")


def test_gradients_finite_at_zero_strain():
    for spec in RUNGS.values():
        for scale in (0.0, 1e-14, 1e-10, 1e-6, 1e-2):
            configuration = torch.full((spec.ndof,), scale, dtype=DTYPE)
            forward = torch.func.jacfwd(lambda x: K.config_to_plant_q(x, spec))(configuration)
            reverse = torch.func.jacrev(lambda x: K.forward_kinematics(x, spec))(configuration)
            assert torch.isfinite(forward).all(), f"{spec.name}: NaN jacfwd at {scale:g}"
            assert torch.isfinite(reverse).all(), f"{spec.name}: NaN jacrev at {scale:g}"
        # ... and the value at exactly zero is right, not merely finite.
        zero = torch.zeros(spec.ndof, dtype=DTYPE)
        pose = K.forward_kinematics(zero, spec)
        expected = torch.tensor([0.0, 0.0, spec.total_length, 1.0, 0.0, 0.0, 0.0], dtype=DTYPE)
        assert float((pose - expected).abs().max()) < 1e-15
    print("PASS gradients are finite at exactly zero strain on every rung")


def test_gradients_match_central_differences():
    spec = SOFT12
    step = 1e-6
    identity = torch.eye(spec.ndof, dtype=DTYPE)
    for configuration in _configurations(spec, 3, seed=7):
        forward = torch.func.jacfwd(lambda x: K.config_to_plant_q(x, spec))(configuration)
        numeric = torch.stack(
            [(K.config_to_plant_q(configuration + step * identity[i], spec)
              - K.config_to_plant_q(configuration - step * identity[i], spec)) / (2 * step)
             for i in range(spec.ndof)], dim=-1)
        relative = float((forward - numeric).abs().max() / forward.abs().max())
        assert relative < 1e-8, f"jacfwd vs central differences: {relative:.3e}"
    print("PASS jacfwd agrees with central differences to better than 1e-8 relative")


def test_drake_matches_torch():
    for spec in RUNGS.values():
        plant, _, context = _build(spec)
        assert plant.num_positions() == spec.num_plant_positions, (
            f"{spec.name}: plant has {plant.num_positions()} positions, spec says "
            f"{spec.num_plant_positions}")
        starts = [plant.GetBodyByName(name).floating_positions_start()
                  for name in spec.body_names()]
        assert starts == list(range(0, 7 * spec.num_bodies, 7)), (
            f"{spec.name}: Drake does not lay the bodies out in declaration order, so "
            f"config_to_plant_q writes into the wrong slots")

        worst_position, worst_quaternion = 0.0, 0.0
        for configuration in _configurations(spec, 25, seed=3):
            plant.SetPositions(context, K.config_to_plant_q(configuration, spec).numpy())
            quaternions, translations, tip_quaternion, tip_translation = K.sublink_poses(
                configuration, spec)
            for index, name in enumerate(spec.body_names()[:-1]):
                pose = plant.GetBodyByName(name).EvalPoseInWorld(context)
                worst_position = max(worst_position, float(
                    np.abs(pose.translation() - translations[index].numpy()).max()))
                # q and -q are the same rotation, so compare magnitudes.
                worst_quaternion = max(worst_quaternion, float(np.abs(
                    np.abs(pose.rotation().ToQuaternion().wxyz())
                    - np.abs(quaternions[index].numpy())).max()))
            tip = plant.GetFrameByName(spec.tip_frame_name).CalcPoseInWorld(context)
            worst_position = max(worst_position, float(
                np.abs(tip.translation() - tip_translation.numpy()).max()))
            worst_quaternion = max(worst_quaternion, float(np.abs(
                np.abs(tip.rotation().ToQuaternion().wxyz())
                - np.abs(tip_quaternion.numpy())).max()))
        assert worst_position < 1e-12, f"{spec.name}: positions differ by {worst_position:.3e}"
        assert worst_quaternion < 1e-12, (
            f"{spec.name}: rotations differ by {worst_quaternion:.3e}")
        print(f"PASS {spec.name}: Drake and torch agree to {worst_position:.1e} m, "
              f"{worst_quaternion:.1e} in quaternion")


def test_spheres_contain_the_backbone():
    """Every point of the true rod is inside the sphere union -- a superset, not a fit."""
    for spec in RUNGS.values():
        # Random draws almost never reach a corner of the configuration box, and the
        # corners are where the spacing is worst (full stretch) and the bend is tightest.
        # Sweep them explicitly: the map is segment-local, so one segment's corner
        # repeated across segments is the worst sub-arc that segment can produce.
        corners = [torch.tensor(list(corner) * spec.num_segments, dtype=DTYPE)
                   for corner in itertools.product((-1.0, 0.0, 1.0),
                                                   repeat=spec.strains_per_segment)]
        worst_clearance = float("inf")
        for configuration in list(_configurations(spec, 40, seed=11)) + corners:
            surface = K.backbone_surface_points(configuration, spec,
                                                samples_per_sublink=12, directions=12)
            centres = K.sphere_centers(configuration, spec)
            distance = torch.cdist(surface, centres).min(dim=-1).values
            clearance = spec.collision_radius - float(distance.max())
            worst_clearance = min(worst_clearance, clearance)
        assert worst_clearance > 0.0, (
            f"{spec.name}: the spheres leave the rod exposed by "
            f"{-worst_clearance * 1000:.2f} mm")
        print(f"PASS {spec.name}: the rod is inside the sphere union with "
              f"{worst_clearance * 1000:.2f} mm to spare")


def test_collision_filters_are_what_they_claim():
    for spec in RUNGS.values():
        _, scene_graph, _ = _build(spec)
        pairs = scene_graph.model_inspector().GetCollisionCandidates()
        live = set()
        inspector = scene_graph.model_inspector()
        for first, second in pairs:
            live.add(frozenset((inspector.GetName(first).split("::")[-1],
                                inspector.GetName(second).split("::")[-1])))

        def segment_of(geometry_name):
            # The tip sphere is a member of the last segment's group.
            if geometry_name.startswith(spec.tip_link_name):
                return spec.num_segments - 1
            return int(geometry_name.split("_")[0][3:])

        for pair in live:
            first, second = sorted(pair)
            assert abs(segment_of(first) - segment_of(second)) >= 2, (
                f"{spec.name}: {first} and {second} are within one segment of each other "
                f"and should have been filtered")
        sizes = spec.filter_group_sizes()
        expected = sum(sizes[i] * sizes[j]
                       for i in range(spec.num_segments) for j in range(spec.num_segments)
                       if j - i >= 2)
        assert len(live) == expected, (
            f"{spec.name}: {len(live)} live pairs, expected {expected} -- a filter that "
            f"matches nothing looks exactly like a correct one")
        print(f"PASS {spec.name}: {len(live)} live self-collision pairs, all non-adjacent")


def test_a_curled_arm_collides_with_itself():
    """Non-adjacent segments are live for a reason: this arm can curl through itself."""
    spec = SOFT12
    plant, _, context = _build(spec)
    constraint = MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=1e-3, influence_distance_offset=0.1, plant_context=context)
    straight = torch.zeros(spec.ndof, dtype=DTYPE)
    value = float(np.asarray(constraint.Eval(
        K.config_to_plant_q(straight, spec).numpy())).flat[0])
    assert value < 1.0, f"the straight arm reads as in collision ({value:.4f})"
    curled = torch.zeros(spec.ndof, dtype=DTYPE)
    for segment in range(spec.num_segments):
        curled[segment * spec.strains_per_segment] = 1.0      # full kappa_x, every segment
    value = float(np.asarray(constraint.Eval(
        K.config_to_plant_q(curled, spec).numpy())).flat[0])
    assert value > 1.0, (
        f"a fully curled arm reads as collision-free ({value:.4f}); the non-adjacent "
        f"segment pairs are not actually being checked")
    print("PASS straight is clear and fully curled self-collides")


def test_generated_sdf_is_current():
    for spec in RUNGS.values():
        path = OutputPath(spec, RepoDir())
        with open(path) as handle:
            on_disk = handle.read()
        assert on_disk == RenderSdf(spec), (
            f"{os.path.relpath(path, RepoDir())} is out of date -- rerun "
            f"src/soft_arm/generate_sdf.py rather than hand-editing it")
    print("PASS the committed SDFs are exactly what the generator emits")


if __name__ == "__main__":
    test_discretization_is_exact()
    test_matches_closed_form_arc()
    test_gradients_finite_at_zero_strain()
    test_gradients_match_central_differences()
    test_drake_matches_torch()
    test_spheres_contain_the_backbone()
    test_collision_filters_are_what_they_claim()
    test_a_curled_arm_collides_with_itself()
    test_generated_sdf_is_current()
    print("ALL PASS")
