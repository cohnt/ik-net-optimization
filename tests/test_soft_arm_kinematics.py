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
                         MinimumDistanceLowerBoundConstraint, Parser, Quaternion,
                         RotationMatrix)

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


#: SoRoMoX's base frame runs the backbone along -x; ours runs it along +z. The two models
#: are otherwise the SAME map, so they differ by this exact constant rotation -- integer
#: entries, zero translation, determinant 1. Declared rather than fitted, so this test pins
#: the frame convention as well as the equivalence: a change to either is caught.
SOROMOX_FROM_OURS = np.array([[0.0, 0.0, -1.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])

GOLDEN = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                      "data", "soft_arm_fk_golden.npz")


def _segment_tip_transforms(configurations, spec):
    """Our map's per-segment tip transforms as 4x4s, to match `forward_kinematics_tips`.

    The quaternion-to-matrix step goes through pydrake rather than through our own
    `kinematics` helpers: a test that reuses the implementation it is checking would not
    catch an error in that conversion.
    """
    tensor = torch.as_tensor(configurations, dtype=DTYPE)
    quaternions, translations, tip_quaternion, tip_translation = K.sublink_poses(tensor, spec)
    per_segment = spec.sublinks_per_segment
    out = np.zeros((tensor.shape[0], spec.num_segments, 4, 4))
    out[:, :, 3, 3] = 1.0
    for segment in range(spec.num_segments):
        if segment == spec.num_segments - 1:
            quaternion, translation = tip_quaternion, tip_translation
        else:
            index = (segment + 1) * per_segment
            quaternion, translation = quaternions[:, index, :], translations[:, index, :]
        for row in range(tensor.shape[0]):
            wxyz = quaternion[row].numpy()
            rotation = RotationMatrix(Quaternion(wxyz / np.linalg.norm(wxyz)))
            out[row, segment, :3, :3] = rotation.matrix()
            out[row, segment, :3, 3] = translation[row].numpy()
    return out


def test_matches_soromox():
    """Our torch map IS the SoRoMoX model, up to a declared constant base frame.

    This is what makes "the analytic forward kinematics from the soft robot repo" a checked
    statement rather than a provenance claim. The golden file is committed and carries its
    own metadata (including that JAX ran in float64 -- a 1e-12 claim against a float32
    oracle would be a fiction), so this runs everywhere with no JAX installed.
    """
    golden = np.load(GOLDEN)
    assert bool(golden["jax_enable_x64"]), (
        "the golden file was generated in float32; regenerate it with jax_enable_x64")
    worst = 0.0
    for name in sorted(RUNGS):
        spec = RUNGS[name]
        configurations, soromox = golden[f"{name}/cfg"], golden[f"{name}/tips"]
        ours = _segment_tip_transforms(configurations, spec)
        expected = SOROMOX_FROM_OURS[None, None] @ ours
        error = float(np.abs(expected - soromox).max())
        worst = max(worst, error)
        assert error < 1e-12, f"{name}: disagrees with SoRoMoX by {error:.3e}"
        print(f"PASS {name}: matches SoRoMoX over {configurations.shape[0]} configurations "
              f"to {error:.1e}")
    print(f"PASS the torch map is SoRoMoX's model to {worst:.1e}, up to R_y(90 deg)")


def test_golden_file_is_reproducible():
    """Regenerating the golden file must not move it. Opt-in: needs the oracle venv.

    Runs the generator through the ORACLE VENV'S OWN INTERPRETER rather than this one. The
    obvious spelling -- `try: import soromox` -- would skip forever: this test file imports
    pydrake, which the oracle venv deliberately does not have, so the two interpreters can
    never be the same one. A check that can only ever skip is exactly the failure this
    golden file exists to avoid, so it is the VENV's presence that gates it, not an import.
    """
    import subprocess
    oracle = os.path.join(RepoDir(), ".venv-soromox", "bin", "python")
    if not os.path.exists(oracle):
        print(f"SKIP golden-file regeneration (no oracle venv at "
              f"{os.path.relpath(oracle, RepoDir())}; "
              f"python3 -m venv .venv-soromox && pip install soromox 'jax[cpu]')")
        return
    with open(GOLDEN, "rb") as handle:
        before = handle.read()
    subprocess.check_call(
        [oracle, os.path.join(RepoDir(), "scripts", "soft_arm", "generate_fk_golden.py")],
        stdout=subprocess.DEVNULL)
    with open(GOLDEN, "rb") as handle:
        after = handle.read()
    assert after == before, (
        "regenerating the golden file changed it -- either SoRoMoX's model moved or the "
        "spec did, and the committed numbers no longer describe this robot")
    print("PASS the golden file regenerates identically from SoRoMoX")


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
    test_matches_soromox()
    test_golden_file_is_reproducible()
    test_generated_sdf_is_current()
    print("ALL PASS")
