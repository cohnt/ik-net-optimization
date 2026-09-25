"""The `helix7` model: the generator, the screw joint, and the torch map against Drake.

DRAKE IS THE ORACLE HERE, and that is worth more than the golden file the soft arm needed.
Its reference model lives in JAX in a separate venv, so equivalence had to be frozen into a
committed `.npz` -- and a check that silently stops running is indistinguishable from one
that passes. `pydrake` is imported by every file in this directory, so this comparison runs
on every invocation, in the same environment the solver runs in, and it re-derives
agreement at freshly drawn configurations against THE VERY PLANT the solver differentiates.
A scene change, a Drake pin move or a parser regression is caught here; a golden file would
sail through all three.

The screw joint's limits are asserted BOTH before and after the repair. Only half of that
is about the repair working: the other half is about the test still being able to tell that
there is something to repair, so that a future Drake which starts honouring `<limit>` on a
screw joint shows up as a failure here rather than as silence.

No pytest config in this repo, so these are plain `test_*` functions with a `__main__`
driver.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.framework import DiagramBuilder

from src.helix_arm import kinematics as K
from src.helix_arm.generate_scenes import RenderScene, ScenePath
from src.helix_arm.generate_sdf import OutputPath, RenderSdf
from src.helix_arm.limits import ApplyScrewJointLimits, RequireFiniteLimits
from src.helix_arm.params import (COLLISION_PAIRS, FILTERED_BUT_CLEAR, LINKS, PRIMARY,
                                  SPECS, GetSpec)
from src.utils import RepoDir

REPO = RepoDir()


def _bare_plant(spec, repair=True):
    """The model alone, welded at the world origin, optionally with the limits repaired."""
    plant = MultibodyPlant(0.0)
    Parser(plant).AddModels(OutputPath(spec, REPO))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(spec.base_link))
    plant.Finalize()
    if repair:
        ApplyScrewJointLimits(plant, spec)
    return plant


def _position_order(plant, spec):
    """Plant position index per joint, read from the plant rather than assumed.

    Declaration order and position order agree on this model today, but nothing guarantees
    it -- on the soft arm, welding the gripper moved a link from position slot 168 to slot
    0, and the symptom was not a crash but a calibration that read as a frame mismatch.
    """
    return [plant.GetJointByName(name).position_start() for name in spec.joint_names]


def test_generated_models_are_current():
    for spec in SPECS.values():
        path = OutputPath(spec, REPO)
        assert os.path.exists(path), f"{path} is missing; run src/helix_arm/generate_sdf.py"
        with open(path) as f:
            on_disk = f.read()
        assert on_disk == RenderSdf(spec), (
            f"{os.path.relpath(path, REPO)} is not what generate_sdf.py emits. The model is "
            f"generated from params.py; edit the spec, not the SDF.")
    print("PASS the committed models are byte-exactly what the generator emits")


def test_generated_scenes_are_current():
    for spec in SPECS.values():
        for legacy in (True, False):
            path = ScenePath(spec, REPO, legacy)
            assert os.path.exists(path), (
                f"{path} is missing; run src/helix_arm/generate_scenes.py")
            with open(path) as f:
                on_disk = f.read()
            assert on_disk == RenderScene(spec, legacy), (
                f"{os.path.relpath(path, REPO)} is not what generate_scenes.py emits. The "
                f"hardened/legacy relation holds by construction here; edit the generator.")
    print("PASS the committed scenes are byte-exactly what the generator emits")


def test_screw_pitch_round_trips_through_the_plant():
    """Pins the UNITS from the plant rather than from documentation.

    `<screw_thread_pitch>` is metres per revolution; its deprecated predecessor
    `<thread_pitch>` was radians per metre AND left-handed. Reading the pitch back off the
    joint is the only way to know which one this Drake understood.
    """
    for spec in SPECS.values():
        plant = _bare_plant(spec)
        for name in spec.screw_joint_names:
            got = plant.GetJointByName(name).screw_pitch()
            assert abs(got - spec.pitch) < 1e-12, f"{spec.name}/{name}: {got} != {spec.pitch}"
    print("PASS screw_pitch round-trips from params through the SDF to the plant")


def test_the_screw_joints_limits_are_dropped_by_the_parser():
    spec = GetSpec(PRIMARY)
    plant = _bare_plant(spec, repair=False)
    joint = plant.GetJointByName(spec.screw_joint_names[0])
    assert not np.isfinite(joint.position_lower_limits()[0]), (
        "Drake now honours <limit> on a screw joint. That is good news, but it means "
        "ApplyScrewJointLimits is no longer the only thing standing between this robot and "
        "a nan sampler -- check the repair is still consistent with what the parser reads.")
    assert not np.isfinite(joint.position_upper_limits()[0])
    print("PASS the parser drops a screw joint's <limit>, which is why limits.py exists")


def test_the_repair_restores_the_limits_the_sdf_records():
    for spec in SPECS.values():
        plant = _bare_plant(spec)
        lower, upper = RequireFiniteLimits(plant, spec.name)
        order = _position_order(plant, spec)
        for joint, slot in zip(spec.joints, order):
            assert abs(lower[slot] - joint.lower) < 1e-12
            assert abs(upper[slot] - joint.upper) < 1e-12
    print("PASS the repair restores exactly the limits the SDF records")


def test_the_repair_must_precede_the_autodiff_copy():
    """`ToAutoDiffXd()` copies the plant, so the order in the program's `__init__` matters.

    Asserting BOTH directions is the point: the wrong order is not a crash, it is an
    autodiff plant that carries the infinities for ever with nothing to say so.
    """
    spec = GetSpec(PRIMARY)

    good = _bare_plant(spec, repair=True).ToAutoDiffXd()
    RequireFiniteLimits(good, "autodiff plant, repaired first")

    bad = _bare_plant(spec, repair=False).ToAutoDiffXd()
    try:
        RequireFiniteLimits(bad, "autodiff plant, repaired second")
    except RuntimeError as exc:
        assert "ApplyScrewJointLimits" in str(exc)
    else:
        raise AssertionError(
            "an autodiff copy taken before the repair should still carry +-inf; if it no "
            "longer does, ToAutoDiffXd has started sharing limits and the ordering comment "
            "in the program's __init__ is stale")
    print("PASS the limits repair must precede ToAutoDiffXd, in both directions")


def test_torch_forward_kinematics_matches_drake():
    rng = np.random.default_rng(11)
    for spec in SPECS.values():
        plant = _bare_plant(spec)
        ctx = plant.CreateDefaultContext()
        order = _position_order(plant, spec)
        lo, hi = (np.array(x) for x in spec.limits)
        worst_p = worst_r = 0.0
        for _ in range(256):
            q = rng.uniform(lo, hi)
            full = np.zeros(plant.num_positions())
            for k, slot in enumerate(order):
                full[slot] = q[k]
            plant.SetPositions(ctx, full)
            X = plant.CalcRelativeTransform(ctx, plant.GetFrameByName(spec.base_link),
                                            plant.GetFrameByName(spec.flange_link))
            ours = K.forward_kinematics(torch.as_tensor(q, dtype=torch.float64), spec).numpy()
            worst_p = max(worst_p, np.abs(X.translation() - ours[:3]).max())
            R = K.quat_to_matrix(torch.as_tensor(ours[3:], dtype=torch.float64)).numpy()
            worst_r = max(worst_r, np.abs(X.rotation().matrix() - R).max())
        assert worst_p < 1e-12, f"{spec.name}: position off by {worst_p:.3e}"
        assert worst_r < 1e-12, f"{spec.name}: rotation off by {worst_r:.3e}"
    print("PASS the torch map is Drake's, to 1e-12, on every rung")


def test_link_poses_match_drake():
    """Not just the flange: every link frame, because the collision screen reads them all."""
    spec = GetSpec(PRIMARY)
    plant = _bare_plant(spec)
    ctx = plant.CreateDefaultContext()
    order = _position_order(plant, spec)
    rng = np.random.default_rng(12)
    lo, hi = (np.array(x) for x in spec.limits)
    worst = 0.0
    for _ in range(64):
        q = rng.uniform(lo, hi)
        full = np.zeros(plant.num_positions())
        for k, slot in enumerate(order):
            full[slot] = q[k]
        plant.SetPositions(ctx, full)
        _, trans = K.link_poses(torch.as_tensor(q, dtype=torch.float64), spec)
        for i, link in enumerate(spec.links):
            X = plant.CalcRelativeTransform(ctx, plant.GetFrameByName(spec.base_link),
                                            plant.GetFrameByName(link.name))
            worst = max(worst, np.abs(X.translation() - trans[i].numpy()).max())
    assert worst < 1e-12, f"link frames off by {worst:.3e}"
    print("PASS every link frame agrees with Drake, to 1e-12")


def test_gradients_are_finite_and_correct():
    """`jacfwd` and `jacrev` both, at the home configuration and at a random one.

    The home configuration is where a badly written rotation would show a removable
    singularity. There is none on this path -- axis-angle to quaternion is entire and the
    screw translation is linear -- and this is what says so rather than the docstring.
    """
    spec = GetSpec(PRIMARY)
    lo, hi = (np.array(x) for x in spec.limits)
    rng = np.random.default_rng(13)
    for label, q in (("home", np.zeros(spec.ndof)), ("random", rng.uniform(lo, hi))):
        x = torch.as_tensor(q, dtype=torch.float64)
        f = lambda v: K.forward_kinematics(v, spec)          # noqa: E731
        jf = torch.func.jacfwd(f)(x)
        jr = torch.func.jacrev(f)(x)
        assert torch.isfinite(jf).all(), f"{label}: jacfwd is not finite"
        assert torch.isfinite(jr).all(), f"{label}: jacrev is not finite"
        assert torch.allclose(jf, jr, atol=1e-12), f"{label}: jacfwd != jacrev"

        eps = 1e-6
        fd = torch.zeros_like(jf)
        for k in range(spec.ndof):
            step = torch.zeros(spec.ndof, dtype=torch.float64)
            step[k] = eps
            fd[:, k] = (f(x + step) - f(x - step)) / (2 * eps)
        err = (jf - fd).abs().max().item()
        assert err < 1e-7, f"{label}: analytic vs central differences off by {err:.3e}"
    print("PASS gradients are finite at the home configuration and match differences")


def test_the_screw_is_what_makes_the_arm_move_axially():
    """One revolution of the screw joint advances the flange by exactly the pitch.

    Also the multimodality the range exists for: `q3` and `q3 + 2*pi` give the SAME
    orientation and a different position, which is precisely what an algebraic inverse
    kinematics cannot enumerate and what the flow is being asked to represent.
    """
    for spec in SPECS.values():
        k = spec.joint_names.index(spec.screw_joint_names[0])
        base = torch.zeros(spec.ndof, dtype=torch.float64)
        turned = base.clone()
        turned[k] = 2.0 * math.pi
        p0 = K.forward_kinematics(base, spec)
        p1 = K.forward_kinematics(turned, spec)
        assert abs(float(p1[2] - p0[2]) - spec.pitch) < 1e-12, (
            f"{spec.name}: one revolution moved the flange {float(p1[2] - p0[2]):.6f} m, "
            f"expected the pitch {spec.pitch:.6f} m")
        assert torch.allclose(p0[3:].abs(), p1[3:].abs(), atol=1e-12), (
            f"{spec.name}: a full revolution should leave the orientation unchanged")
    print("PASS one revolution advances the flange by exactly the pitch, orientation fixed")


def test_pitch_zero_is_a_plain_revolute_arm():
    """The control rung is the same arm with the coupling switched off, and nothing else.

    The sharp statement is about PERIODICITY, not about the flange standing still -- at
    pitch 0 the joint is still a revolute joint, so turning it moves everything downstream.
    What changes is that a full revolution becomes the identity again: `q` and `q + 2*pi`
    are the same configuration. At any other pitch they differ by exactly the pitch, which
    is the multimodality an algebraic inverse kinematics cannot enumerate, and here it is
    switched off. The two rungs are otherwise the same arm, which the second half asserts.
    """
    p000, primary = GetSpec("helix7_p000"), GetSpec(PRIMARY)
    k = primary.joint_names.index(primary.screw_joint_names[0])
    rng = np.random.default_rng(14)
    lo, hi = (np.array(x) for x in primary.limits)
    for _ in range(32):
        q = rng.uniform(lo, hi)
        q[k] = rng.uniform(0.0, 2.0 * math.pi)          # room to add a turn
        x = torch.as_tensor(q, dtype=torch.float64)
        turned = x.clone()
        turned[k] = x[k] + 2.0 * math.pi

        flat = K.forward_kinematics(x, p000)
        assert torch.allclose(flat, K.forward_kinematics(turned, p000), atol=1e-12), (
            "at pitch 0 a full revolution of the screw coordinate must be the identity")

        moved = K.forward_kinematics(turned, primary) - K.forward_kinematics(x, primary)
        assert abs(float(torch.linalg.norm(moved[:3])) - primary.pitch) < 1e-12, (
            "at the primary pitch the same revolution must advance the flange by the pitch")
        assert torch.allclose(moved[3:], torch.zeros(4, dtype=torch.float64), atol=1e-12)

        # ...and with the coupling at rest the two rungs are the same robot.
        rest = x.clone()
        rest[k] = 0.0
        assert torch.allclose(K.forward_kinematics(rest, p000),
                              K.forward_kinematics(rest, primary), atol=1e-14)

    assert abs(_bare_plant(p000).GetJointByName(p000.screw_joint_names[0]).screw_pitch()) < 1e-15
    print("PASS helix7_p000 is the same arm with the coupling switched off")


## There is deliberately no `RationalForwardKinematics` test here. It refuses this model --
## "only weld, revolute and prismatic joints are supported" -- which is Drake's
## tangent-half-angle machinery saying the arm is outside the algebraic class, and it looks
## like an attractive witness. It is not one. It keys on the JOINT TYPE, so it refuses
## `helix7_p000` exactly as readily as the primary rung even though that rung's coupling is
## switched off, and it therefore distinguishes nothing about this ladder. Nothing in this
## project uses that machinery either. `test_screw_pitch_round_trips_through_the_plant`
## already asserts what the model is, from the plant, and asserts it per rung.


def _checked_pairs_in_drake(spec):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant, scene_graph).AddModels(OutputPath(spec, REPO))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(spec.base_link))
    plant.Finalize()
    inspector = scene_graph.model_inspector()
    index = {link.name: i for i, link in enumerate(LINKS)}
    pairs = set()
    for a, b in inspector.GetCollisionCandidates():
        na = plant.GetBodyFromFrameId(inspector.GetFrameId(a)).name()
        nb = plant.GetBodyFromFrameId(inspector.GetFrameId(b)).name()
        if na != nb:
            pairs.add(tuple(sorted((index[na], index[nb]))))
    return pairs


def test_collision_filters_are_what_the_spec_claims():
    """The screen and the model must check the same pairs.

    If they disagree the dataset is screened against a different robot from the one the
    solver sees, and nothing anywhere says so.
    """
    spec = GetSpec(PRIMARY)
    assert _checked_pairs_in_drake(spec) == set(COLLISION_PAIRS)
    print(f"PASS Drake checks exactly the {len(COLLISION_PAIRS)} pairs COLLISION_PAIRS names")


def test_the_home_configuration_and_the_whole_stroke_are_clear():
    spec = GetSpec(PRIMARY)
    lo, hi = (np.array(x) for x in spec.limits)
    k = spec.joint_names.index(spec.screw_joint_names[0])
    sweep = torch.zeros(65, spec.ndof, dtype=torch.float64)
    sweep[:, k] = torch.linspace(float(lo[k]), float(hi[k]), 65, dtype=torch.float64)
    depth = K.self_collision_depth(sweep, spec)
    assert float(depth.max()) < 0.0, (
        f"the arm self-collides somewhere in its own screw stroke, worst depth "
        f"{float(depth.max()):+.4f} m")
    print(f"PASS home and the whole screw stroke are clear by {-float(depth.max()):.3f} m")


def test_the_filtered_pair_that_must_stay_clear():
    """`base_link` against `upper_arm` is filtered only because it is close along the chain.

    It stays clear for a reason that is part of the robot -- the screw's range is one-sided,
    so the tube's tail never travels back down towards the pedestal. A failure here is the
    geometry saying the stroke reaches somewhere it should not.
    """
    spec = GetSpec(PRIMARY)
    index = {link.name: i for i, link in enumerate(LINKS)}
    rng = np.random.default_rng(15)
    lo, hi = (np.array(x) for x in spec.limits)
    q = torch.as_tensor(rng.uniform(lo, hi, size=(50000, spec.ndof)), dtype=torch.float64)
    centres = K.sphere_centres_world(q, spec)
    owner, radii = [], []
    for i, link in enumerate(LINKS):
        n = len(link.sphere_centres())
        owner += [i] * n
        radii += [link.radius] * n
    owner = np.asarray(owner)
    radii = torch.as_tensor(radii, dtype=torch.float64)
    for a_name, b_name in FILTERED_BUT_CLEAR:
        a = np.flatnonzero(owner == index[a_name])
        b = np.flatnonzero(owner == index[b_name])
        gap = (torch.cdist(centres[:, a, :], centres[:, b, :])
               - (radii[a].unsqueeze(1) + radii[b].unsqueeze(0)))
        worst = float(gap.min())
        assert worst > 0.0, (
            f"{a_name} and {b_name} are filtered but touch, worst gap {worst:+.4f} m")
        print(f"     {a_name} vs {b_name}: clear by {worst:.4f} m over 50k draws")
    print("PASS the pairs the chain-gap rule filters but must not touch stay clear")


if __name__ == "__main__":
    test_generated_models_are_current()
    test_generated_scenes_are_current()
    test_screw_pitch_round_trips_through_the_plant()
    test_the_screw_joints_limits_are_dropped_by_the_parser()
    test_the_repair_restores_the_limits_the_sdf_records()
    test_the_repair_must_precede_the_autodiff_copy()
    test_torch_forward_kinematics_matches_drake()
    test_link_poses_match_drake()
    test_gradients_are_finite_and_correct()
    test_the_screw_is_what_makes_the_arm_move_axially()
    test_pitch_zero_is_a_plain_revolute_arm()
    test_collision_filters_are_what_the_spec_claims()
    test_the_home_configuration_and_the_whole_stroke_are_clear()
    test_the_filtered_pair_that_must_stay_clear()
    print("ALL PASS")
