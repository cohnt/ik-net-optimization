"""The GVS push-rod arm's forward model, and the Drake model's agreement with it.

The model IS SoRoMoX (`src/gvs_arm/model.py`), so nothing here re-derives its physics; what
is tested is the WRAPPER and the DESIGN:

* `test_rest_and_axial` -- `cfg = 0` is the straight arm at z = 0.8 with an identity tip
  frame; three equal rod forces give pure axial strain and no bend.
* `test_single_rod_bends_a_circular_arc` -- on the order-0 rung a differential rod force is a
  uniform moment, so segment 0 is a circular arc of radius `1/kappa`, with `kappa` within a
  few percent of the design curvature the force scale was derived from. This pins which
  coordinate is curvature about which axis, and the sign of the rod input.
* `test_taper_makes_the_strain_variable` -- THE TRAP. With a uniform section every Legendre
  coefficient above order 0 is exactly zero at equilibrium; with the taper they are not. If
  this ever stopped holding the order ladder would measure nothing.
* `test_jacobian_matches_central_differences` -- the implicit `dq*/du` composed with the
  poses and the quaternion conversion, against finite differences of the same map.
* `test_equilibrium_converges_and_is_unique` -- Newton converges on every draw of a batch,
  and restarting from random backbone states reaches the same equilibrium; the cold-start
  selection rule is therefore not selecting among alternatives.
* `test_drake_matches_model` -- the generated SDF and the SoRoMoX poses are two descriptions
  of one robot: body poses agree to 1e-12, and the tip FRAME's rotation is the body's
  pitched by +90 degrees so its z-axis runs along the backbone.
* `test_spheres_contain_the_backbone` -- one-sided: the (tapered) sphere union must CONTAIN
  the rod's surface, at the box corners where stretch and bend are worst.
* `test_collision_filters_are_what_they_claim`, `test_a_curled_arm_collides_with_itself`,
  `test_generated_files_are_current`, `test_redundancy_is_enforced`.

Run with the project venv: `GVS_ARM_XLA_THREADS=2 .venv/bin/python tests/test_gvs_arm_model.py`.
"""

import dataclasses
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder,  # noqa: E402
                         MinimumDistanceLowerBoundConstraint, Parser)

from src.gvs_arm import generate_scenes as GS  # noqa: E402
from src.gvs_arm import generate_sdf as GSDF  # noqa: E402
from src.gvs_arm.model import GetModel, GvsArmModel, TIP_FRAME_PITCH  # noqa: E402
from src.gvs_arm.params import RUNGS, GetSpec, GvsArmSpec  # noqa: E402
from src.utils import RepoDir  # noqa: E402

PRIMARY = GetSpec("gvs_pushrod9_o1")


def _configurations(spec, count, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(count, spec.ninputs))


def _build(spec):
    """The bare arm, welded at the origin -- no scene, so this tests the model alone."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    parser.package_map().AddPackageXml(os.path.join(RepoDir(), "package.xml"))
    parser.AddModels(GSDF.OutputPath(spec, RepoDir()))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.Finalize()
    diagram = builder.Build()
    context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    return plant, scene_graph, context


def _differential(spec, segment, rod=0, level=1.0):
    """One rod at `+level`, the other two at `-level/2`: a pure moment on that segment."""
    cfg = np.zeros(spec.ninputs)
    base = segment * spec.rods_per_segment
    cfg[base:base + spec.rods_per_segment] = -level / 2
    cfg[base + rod] = level
    return cfg


def test_rest_and_axial():
    for spec in RUNGS.values():
        model = GetModel(spec)
        plant_q = model.PlantQ(np.zeros(spec.ninputs)).reshape(-1, 7)
        tip = model.TipPose(np.zeros(spec.ninputs))
        assert np.allclose(plant_q[-1, 4:], [0, 0, spec.total_length], atol=1e-14)
        assert np.allclose(tip, [0, 0, spec.total_length, 1, 0, 0, 0], atol=1e-14), tip
        ## Three equal forces on segment 0: stretch, no bend, nothing downstream bends either.
        cfg = np.zeros(spec.ninputs)
        cfg[:spec.rods_per_segment] = 1.0
        q = model.Equilibrium(cfg)
        per = len(spec.active_strains) * (spec.basis_order + 1)
        block = q[:per].reshape(len(spec.active_strains), spec.basis_order + 1)
        assert np.abs(block[:2]).max() < 1e-12, f"{spec.name}: equal forces bend"
        assert block[2, 0] > 0.05, f"{spec.name}: equal pushes do not stretch"
        assert np.abs(q[per:]).max() < 1e-12, f"{spec.name}: a loaded segment moved another"
        tip = model.TipPose(cfg)
        assert abs(tip[0]) < 1e-12 and abs(tip[1]) < 1e-12 and tip[2] > spec.total_length
        print(f"PASS {spec.name}: rest is straight at z = {spec.total_length}; equal forces "
              f"stretch segment 0 by {block[2, 0]:.3f} with no bend")


def test_single_rod_bends_a_circular_arc():
    spec = GetSpec("gvs_pushrod9_o0")
    model = GetModel(spec)
    cfg = _differential(spec, 0)
    q = model.Equilibrium(cfg)
    ## Order 0, active strains (kappa_y, kappa_z, sigma_x): one coefficient each.
    kappa_y, kappa_z, sigma_x = q[:3]
    assert abs(kappa_y) < 1e-12, "rod 0 sits on local +y and must bend about z only"
    ## Not EXACTLY zero: each rod's force acts along its own tangent, and on a bent segment
    ## the outer and inner rods tilt differently, so their axial components no longer cancel
    ## to the last digit. Physical, small, and the arc below accounts for it.
    assert abs(sigma_x) < 0.02, f"a pure moment stretches by {sigma_x}"
    kappa = abs(kappa_z)
    assert abs(kappa - spec.design_curvature) / spec.design_curvature < 0.05, kappa
    poses = model.PosesOfBackbone(q)
    tip0 = poses[spec.sublinks_per_segment][:3, 3]           # segment 0's end
    L = spec.segment_length
    stretch = 1.0 + sigma_x
    expected = np.array([0.0, -stretch * (1 - np.cos(kappa * L)) / kappa,
                         stretch * np.sin(kappa * L) / kappa])
    assert np.abs(tip0 - expected).max() < 1e-10, (tip0, expected)
    print(f"PASS a differential rod force bends segment 0 into a circular arc of "
          f"{kappa:.3f} /m (design {spec.design_curvature}), pushing toward -y, with "
          f"{sigma_x:+.4f} residual stretch from the rods' tilt")


def test_taper_makes_the_strain_variable():
    for name in ("gvs_pushrod9_o1", "gvs_pushrod9_o2"):
        spec = GetSpec(name)
        uniform = dataclasses.replace(spec, name=f"{name}_uniform", radius_tip=spec.radius_base)
        tapered, flat = GetModel(spec), GvsArmModel(uniform)
        order = spec.basis_order
        cfg = _differential(spec, 0)
        q_t = tapered.Equilibrium(cfg)[:3 * (order + 1)].reshape(3, order + 1)
        q_u = flat.Equilibrium(cfg)[:3 * (order + 1)].reshape(3, order + 1)
        higher_t = np.abs(q_t[:, 1:]).max()
        higher_u = np.abs(q_u[:, 1:]).max()
        assert higher_u < 1e-10, f"{name}: uniform section has variable strain {higher_u}"
        assert higher_t > 0.5, f"{name}: taper gives no variable strain ({higher_t})"
        print(f"PASS {name}: higher-order curvature {higher_t:.3f} /m on the taper, "
              f"{higher_u:.1e} on a uniform section")


def test_jacobian_matches_central_differences():
    spec = PRIMARY
    model = GetModel(spec)
    step = 1e-6
    for cfg in _configurations(spec, 3, seed=7, scale=0.6):
        jacobian, value = model.PlantJacobian(cfg)
        assert np.allclose(value, model.PlantQ(cfg))
        numeric = np.stack([(model.PlantQ(cfg + step * e) - model.PlantQ(cfg - step * e))
                            / (2 * step) for e in np.eye(spec.ninputs)], axis=-1)
        relative = float(np.abs(jacobian - numeric).max() / np.abs(jacobian).max())
        assert relative < 1e-6, f"implicit Jacobian vs central differences: {relative:.3e}"
    print(f"PASS the implicit Jacobian agrees with central differences to {relative:.1e} relative")


def test_equilibrium_converges_and_is_unique():
    spec = PRIMARY
    model = GetModel(spec)
    cfgs = _configurations(spec, 2000, seed=1)
    q, ok, steps = model.EquilibriumBatch(cfgs)
    assert ok.all(), f"{int((~ok).sum())} of {len(cfgs)} draws did not converge"
    rng = np.random.default_rng(2)
    worst = 0.0
    for i in range(50):
        scale = np.abs(q[i]).max() + 1.0
        q0 = rng.uniform(-scale, scale, size=q.shape[1])
        q_restart, ok_i, _ = model.EquilibriumFrom(cfgs[i], q0)
        assert ok_i, f"restart {i} did not converge"
        worst = max(worst, float(np.abs(q_restart - q[i]).max()))
    assert worst < 1e-8, f"a restart found a different equilibrium ({worst:.3e})"
    print(f"PASS Newton converges on 2000/2000 draws (median {int(np.median(steps))} steps); "
          f"50 random restarts agree with the cold start to {worst:.1e}")


def test_drake_matches_model():
    for spec in RUNGS.values():
        model = GetModel(spec)
        plant, _, context = _build(spec)
        assert plant.num_positions() == spec.num_plant_positions
        starts = [plant.GetBodyByName(name).floating_positions_start()
                  for name in spec.body_names()]
        assert starts == list(range(0, 7 * spec.num_bodies, 7)), "bodies not in declaration order"
        worst_p, worst_R = 0.0, 0.0
        for cfg in _configurations(spec, 10, seed=3):
            plant.SetPositions(context, model.PlantQ(cfg))
            poses = model.PosesOfBackbone(model.Equilibrium(cfg))
            for index, name in enumerate(spec.body_names()):
                pose = plant.GetBodyByName(name).EvalPoseInWorld(context)
                worst_p = max(worst_p, float(np.abs(pose.translation() - poses[index, :3, 3]).max()))
                worst_R = max(worst_R, float(np.abs(pose.rotation().matrix()
                                                    - poses[index, :3, :3]).max()))
            ## The tip FRAME: pitched by +90 degrees so its z-axis runs along the backbone.
            frame = plant.GetFrameByName(spec.tip_frame_name).CalcPoseInWorld(context)
            tip = model.TipPose(cfg)
            worst_p = max(worst_p, float(np.abs(frame.translation() - tip[:3]).max()))
            worst_R = max(worst_R, float(np.abs(np.abs(frame.rotation().ToQuaternion().wxyz())
                                                - np.abs(tip[3:])).max()))
            tangent = poses[-1, :3, 0]                           # body x = backbone tangent
            assert np.abs(frame.rotation().matrix()[:, 2] - tangent).max() < 1e-12
        assert worst_p < 1e-12 and worst_R < 1e-12, (spec.name, worst_p, worst_R)
        print(f"PASS {spec.name}: Drake and SoRoMoX agree to {worst_p:.1e} m, {worst_R:.1e} in "
              f"rotation; the tip frame's z-axis is the backbone tangent (pitch {TIP_FRAME_PITCH:.4f})")


def test_spheres_contain_the_backbone():
    """Every point of the true rod surface is inside the sphere union -- a superset, not a fit."""
    for spec in RUNGS.values():
        model = GetModel(spec)
        radii = np.asarray(spec.body_radii())
        corners = []
        for pattern in ((1, -1, -1), (1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, 1, 1)):
            corners.append(np.array(list(pattern) * spec.num_segments, dtype=float))
        worst = float("inf")
        for cfg in list(_configurations(spec, 30, seed=11)) + corners:
            surface = model.BackboneSurfacePoints(cfg, samples=240, directions=12)
            centres = model.PlantQ(cfg).reshape(-1, 7)[:, 4:]
            distance = np.linalg.norm(surface[:, None, :] - centres[None], axis=-1)
            clearance = float((radii[None] - distance).max(axis=1).min())
            worst = min(worst, clearance)
        assert worst > 0.0, f"{spec.name}: the rod is exposed by {-worst * 1000:.2f} mm"
        print(f"PASS {spec.name}: the rod is inside the sphere union with {worst * 1000:.2f} mm "
              f"to spare (spacing margin {spec.sphere_coverage_margin() * 1000:.2f} mm)")


def test_collision_filters_are_what_they_claim():
    for spec in RUNGS.values():
        _, scene_graph, _ = _build(spec)
        inspector = scene_graph.model_inspector()
        live = set()
        for first, second in inspector.GetCollisionCandidates():
            live.add(frozenset((inspector.GetName(first).split("::")[-1],
                                inspector.GetName(second).split("::")[-1])))

        def segment_of(geometry_name):
            if geometry_name.startswith(spec.tip_link_name):
                return spec.num_segments - 1
            return int(geometry_name.split("_")[0][3:])

        for pair in live:
            first, second = sorted(pair)
            assert abs(segment_of(first) - segment_of(second)) >= 2, (first, second)
        sizes = spec.filter_group_sizes()
        expected = sum(sizes[i] * sizes[j]
                       for i in range(spec.num_segments) for j in range(spec.num_segments)
                       if j - i >= 2)
        assert len(live) == expected, (len(live), expected)
        print(f"PASS {spec.name}: {len(live)} live self-collision pairs, all non-adjacent")


def test_a_curled_arm_collides_with_itself():
    spec = PRIMARY
    model = GetModel(spec)
    plant, _, context = _build(spec)
    constraint = MinimumDistanceLowerBoundConstraint(
        plant=plant, bound=1e-3, influence_distance_offset=0.1, plant_context=context)
    straight = float(np.asarray(constraint.Eval(model.PlantQ(np.zeros(spec.ninputs)))).flat[0])
    assert straight < 1.0, f"the straight arm reads as in collision ({straight:.4f})"
    curled = np.concatenate([_differential(spec, i)[i * 3:(i + 1) * 3]
                             for i in range(spec.num_segments)])
    value = float(np.asarray(constraint.Eval(model.PlantQ(curled))).flat[0])
    assert value > 1.0, f"a fully curled arm reads as collision-free ({value:.4f})"
    print("PASS straight is clear and fully curled self-collides")


def test_sampler_path_is_one_solve():
    """`TipAndCentresBatch` (the dataset sampler's single solve per draw) agrees with the
    separate tip-pose and sphere-centre paths, and its collision screen with `SelfCollides`."""
    spec = PRIMARY
    model = GetModel(spec)
    cfgs = _configurations(spec, 64, seed=3)
    tips, centres, ok = model.TipAndCentresBatch(cfgs)
    tips_ref, ok_tip = model.TipPoseBatch(cfgs)
    centres_ref, ok_centres = model.SphereCentresBatch(cfgs)
    assert ok.all() and ok_tip.all() and ok_centres.all()
    assert np.abs(tips - tips_ref).max() < 1e-12, np.abs(tips - tips_ref).max()
    assert np.abs(centres - centres_ref).max() < 1e-12, np.abs(centres - centres_ref).max()
    collides_ref, _ = model.SelfCollides(cfgs)
    assert (model.CollidesFromCentres(centres) == collides_ref).all()
    print(f"PASS one-solve sampler path matches the separate paths on 64 draws "
          f"({int(collides_ref.sum())} self-colliding)")


def test_generated_files_are_current():
    for spec in RUNGS.values():
        with open(GSDF.OutputPath(spec, RepoDir())) as handle:
            assert handle.read() == GSDF.RenderSdf(spec), f"{spec.name}.sdf is stale"
        for legacy in (False, True):
            with open(GS.ScenePath(spec, RepoDir(), legacy)) as handle:
                assert handle.read() == GS.RenderScene(spec, legacy), f"{spec.name} scene stale"
    print("PASS the committed SDFs and scenes are exactly what the generators emit")


def test_redundancy_is_enforced():
    for spec in RUNGS.values():
        assert spec.ninputs > 6
    try:
        GvsArmSpec(name="two_rods", basis_order=1, rods_per_segment=2, rod_azimuths_deg=(0, 180))
    except ValueError as error:
        assert "redundancy" in str(error)
        print("PASS a 6-input variant is refused: at least one degree of redundancy is required")
        return
    raise AssertionError("a spec with no kinematic redundancy was accepted")


if __name__ == "__main__":
    test_rest_and_axial()
    test_single_rod_bends_a_circular_arc()
    test_taper_makes_the_strain_variable()
    test_jacobian_matches_central_differences()
    test_equilibrium_converges_and_is_unique()
    test_drake_matches_model()
    test_spheres_contain_the_backbone()
    test_collision_filters_are_what_they_claim()
    test_a_curled_arm_collides_with_itself()
    test_sampler_path_is_one_solve()
    test_generated_files_are_current()
    test_redundancy_is_enforced()
    print("ALL PASS")
