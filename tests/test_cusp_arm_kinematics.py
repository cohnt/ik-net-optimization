"""The `cusp7` model: the generators, the torch map against Drake, and fidelity to the paper.

These check that we built the published robot -- the simplified Transpressor 6R of Feeß and
Weiß (arXiv 2602.14794) plus a tool pitch -- not that it is cuspidal: the paper proves that,
and Tommy ruled (2026-11-25) that a known cuspidal arm is used as known. A mis-rendered DH
table, a wrong joint-zero offset or a sign slip in the added joint would build some other
robot, and would show up here rather than as a silently different experiment.

Drake is the oracle for the torch map (the plant the solver differentiates); the paper's DH
product, written out independently in `kinematics.dh_flange_pose`, is the oracle for the
joint table. Plain `test_*` functions with a `__main__` driver; tiers per tests/_tiers.py,
the default `fast` group only.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import _tiers                                                             # noqa: E402

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.framework import DiagramBuilder

from src.cusp_arm import kinematics as K
from src.cusp_arm.generate_scenes import RenderScene, ScenePath
from src.cusp_arm.generate_sdf import OutputPath, RenderSdf
from src.cusp_arm.params import (A2, A5, COLLISION_PAIRS, D4, DH, LINKS, PAPER_A2,
                                 PAPER_A5, PAPER_D4, PRIMARY, SCALE, SPECS, THETA_OFFSET,
                                 GetSpec)
from src.utils import RepoDir

REPO = RepoDir()


def _bare_plant(spec):
    plant = MultibodyPlant(0.0)
    Parser(plant).AddModels(OutputPath(spec, REPO))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(spec.base_link))
    plant.Finalize()
    return plant


def _position_order(plant, spec):
    """Plant position slot per joint, read from the plant rather than assumed."""
    return [plant.GetJointByName(name).position_start() for name in spec.joint_names]


def _set(plant, ctx, spec, q):
    full = np.zeros(plant.num_positions())
    for k, slot in enumerate(_position_order(plant, spec)):
        full[slot] = q[k]
    plant.SetPositions(ctx, full)


def test_generated_models_are_current():
    for spec in SPECS.values():
        path = OutputPath(spec, REPO)
        assert os.path.exists(path), f"{path} is missing; run src/cusp_arm/generate_sdf.py"
        with open(path) as f:
            assert f.read() == RenderSdf(spec), (
                f"{os.path.relpath(path, REPO)} is not what generate_sdf.py emits; edit "
                f"params.py, not the SDF.")
    print("PASS the committed model is byte-exactly what the generator emits")


def test_generated_scenes_are_current():
    for spec in SPECS.values():
        for legacy in (True, False):
            path = ScenePath(spec, REPO, legacy)
            assert os.path.exists(path), f"{path} missing; run src/cusp_arm/generate_scenes.py"
            with open(path) as f:
                assert f.read() == RenderScene(spec, legacy), (
                    f"{os.path.relpath(path, REPO)} is not what generate_scenes.py emits")
    print("PASS the committed scenes are byte-exactly what the generator emits")


def test_the_dh_table_is_the_papers():
    """The numbers themselves: the paper's table, its example values, one uniform scale."""
    assert (PAPER_A2, PAPER_D4, PAPER_A5) == (3.0, 2.0, 0.5)
    assert (A2, D4, A5) == (3.0 * SCALE, 2.0 * SCALE, 0.5 * SCALE)
    h = 0.5 * math.pi
    expected = ((None, 0.0, h), (0.0, A2, 0.0), (0.0, 0.0, -h),
                (D4, 0.0, h), (0.0, A5, -h), (0.0, 0.0, 0.0))
    for i, ((d, a, alpha), (ed, ea, ealpha)) in enumerate(zip(DH, expected)):
        assert (ed is None or d == ed) and a == ea and alpha == ealpha, f"DH row {i + 1}"
    print("PASS the DH table is Feeß & Weiß's, at their example values x %.2f m" % SCALE)


def test_torch_forward_kinematics_matches_drake():
    spec = GetSpec(PRIMARY)
    plant = _bare_plant(spec)
    ctx = plant.CreateDefaultContext()
    lo, hi = (np.array(x) for x in spec.limits)
    rng = np.random.default_rng(11)
    worst_p = worst_r = worst_link = 0.0
    for _ in range(256):
        q = rng.uniform(lo, hi)
        _set(plant, ctx, spec, q)
        X = plant.CalcRelativeTransform(ctx, plant.GetFrameByName(spec.base_link),
                                        plant.GetFrameByName(spec.flange_link))
        ours = K.forward_kinematics(torch.as_tensor(q, dtype=torch.float64), spec).numpy()
        worst_p = max(worst_p, np.abs(X.translation() - ours[:3]).max())
        R = K.quat_to_matrix(torch.as_tensor(ours[3:], dtype=torch.float64)).numpy()
        worst_r = max(worst_r, np.abs(X.rotation().matrix() - R).max())
        _, trans = K.link_poses(torch.as_tensor(q, dtype=torch.float64), spec)
        for i, link in enumerate(spec.links):
            Xl = plant.CalcRelativeTransform(ctx, plant.GetFrameByName(spec.base_link),
                                             plant.GetFrameByName(link.name))
            worst_link = max(worst_link, np.abs(Xl.translation() - trans[i].numpy()).max())
    assert worst_p < 1e-12 and worst_r < 1e-12, f"flange off by {worst_p:.3e} / {worst_r:.3e}"
    assert worst_link < 1e-12, f"link frames off by {worst_link:.3e}"
    for joint in spec.joints:
        j = plant.GetJointByName(joint.name)
        assert abs(j.position_lower_limits()[0] - joint.lower) < 1e-12
        assert abs(j.position_upper_limits()[0] - joint.upper) < 1e-12
    print("PASS the torch map, every link frame and the joint limits are Drake's, to 1e-12")


def test_the_chain_is_the_papers_dh_product():
    """The joint table against the paper's product `Rz Tz Tx Rx`, written out separately,
    with `theta = q + THETA_OFFSET` and the tool pitch appended. This is what catches a
    mis-rendered table or a wrong offset; Drake agreeing with torch cannot."""
    spec = GetSpec(PRIMARY)
    lo, hi = (torch.tensor(x, dtype=torch.float64) for x in spec.limits)
    g = torch.Generator().manual_seed(12)
    q = lo + (hi - lo) * torch.rand(512, spec.ndof, generator=g, dtype=torch.float64)
    quat, trans = K.flange_pose(q, spec)
    T = K.dh_flange_pose(q)
    err_r = (K.quat_to_matrix(quat) - T[:, :3, :3]).abs().max().item()
    err_p = (trans - T[:, :3, 3]).abs().max().item()
    assert err_r < 1e-12 and err_p < 1e-12, f"chain != DH product: {err_r:.3e} / {err_p:.3e}"
    print("PASS the rendered chain is the paper's DH product (with the joint-zero offsets)")


def test_the_added_joint_leaves_the_6r_intact():
    """Locking the tool pitch at any angle leaves the published 6R with a fixed tool: the
    6R's frame 6 does not depend on q7, and q7 only rotates the tool about the barrel's y."""
    spec = GetSpec(PRIMARY)
    g = torch.Generator().manual_seed(13)
    q = (torch.rand(64, spec.ndof, generator=g, dtype=torch.float64) * 2 - 1) * 2.0
    quats, transs = K.link_poses(q, spec)
    q2 = q.clone()
    q2[:, 6] = q2[:, 6] + 0.7
    quats2, transs2 = K.link_poses(q2, spec)
    six = spec.links.index(next(l for l in LINKS if l.name == "wrist_roll"))
    assert torch.allclose(quats[:, six], quats2[:, six], atol=1e-14)
    assert torch.allclose(transs[:, six], transs2[:, six], atol=1e-14)
    assert torch.allclose(transs[:, -1], transs2[:, -1], atol=1e-14), "q7 moved the pivot"
    print("PASS the 6R sub-chain is independent of the added joint (every locked slice is it)")


def test_det_j_matches_the_closed_form():
    """Drake's 6x6 Jacobian of the 6R sub-chain against the closed form `det_j_6r` derived
    from the DH table (the paper's printed form does not match its table; see the
    docstring there). Frame-invariant up to the row-order sign, so the base offset is
    irrelevant; the formula is in metres, so the scale is carried by it."""
    spec = GetSpec(PRIMARY)
    plant = _bare_plant(spec)
    ctx = plant.CreateDefaultContext()
    rng = np.random.default_rng(14)
    lo, hi = (np.array(x) for x in spec.limits)
    order = _position_order(plant, spec)
    ratios = []
    for _ in range(64):
        q = rng.uniform(lo, hi)
        _set(plant, ctx, spec, q)
        frame6 = plant.GetFrameByName("wrist_roll")
        J = plant.CalcJacobianSpatialVelocity(
            ctx, JacobianWrtVariable.kQDot, frame6, np.zeros(3),
            plant.world_frame(), plant.world_frame())
        J6 = J[:, order[:6]]
        theta = torch.tensor([q[:6] + np.array(THETA_OFFSET)], dtype=torch.float64)
        ours = float(np.linalg.det(J6))
        paper = float(K.det_j_6r(theta)[0])
        if abs(paper) > 1e-3:
            ratios.append(ours / paper)
    ratios = np.array(ratios)
    assert np.allclose(ratios, ratios[0], rtol=1e-9) and abs(abs(ratios[0]) - 1) < 1e-9, (
        f"det J is not the closed form: ratios {ratios[:6]}")
    print(f"PASS det J of the 6R is the closed form (ratio {ratios[0]:+.0f})")


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
    """The dataset screen and the solver's model must check the same pairs."""
    spec = GetSpec(PRIMARY)
    assert _checked_pairs_in_drake(spec) == set(COLLISION_PAIRS)
    print(f"PASS Drake checks exactly the {len(COLLISION_PAIRS)} pairs COLLISION_PAIRS names")


def test_home_and_nominal():
    """q = 0 is the arm straight up (the iiwa's zero); the nominal is in limits, clear of
    self-collision and nonsingular."""
    spec = GetSpec(PRIMARY)
    _, trans = K.link_poses(torch.zeros(spec.ndof, dtype=torch.float64), spec)
    elbow, wrist = trans[3].numpy(), trans[5].numpy()
    assert np.allclose(elbow[:2], 0, atol=1e-12) and np.allclose(wrist[:2], 0, atol=1e-12), (
        "q = 0 is not the arm straight up: the joint-zero offsets are wrong")
    assert abs(elbow[2] - (0.42 + A2)) < 1e-12 and abs(wrist[2] - (0.42 + A2 + D4)) < 1e-12

    qn = np.array(spec.q_nominal)
    lo, hi = (np.array(x) for x in spec.limits)
    assert np.all(qn > lo + 0.5) and np.all(qn < hi - 0.5), "nominal within 0.5 rad of a limit"
    depth = float(K.self_collision_depth(torch.tensor([qn], dtype=torch.float64), spec)[0])
    assert depth < -0.01, f"nominal self-collision depth {depth:+.4f}"
    plant = _bare_plant(spec)
    ctx = plant.CreateDefaultContext()
    _set(plant, ctx, spec, qn)
    J = plant.CalcJacobianSpatialVelocity(
        ctx, JacobianWrtVariable.kQDot, plant.GetFrameByName(spec.flange_link), np.zeros(3),
        plant.world_frame(), plant.world_frame())[:, _position_order(plant, spec)]
    smin = np.linalg.svd(J, compute_uv=False).min()
    assert smin > 0.2, f"nominal sigma_min {smin:.3f}"
    print(f"PASS q = 0 is straight up; nominal clear by {-depth:.3f} m, sigma_min {smin:.3f}")


def test_gradients_are_finite_and_correct():
    spec = GetSpec(PRIMARY)
    lo, hi = (np.array(x) for x in spec.limits)
    rng = np.random.default_rng(15)
    for label, q in (("home", np.zeros(spec.ndof)), ("random", rng.uniform(lo, hi))):
        x = torch.as_tensor(q, dtype=torch.float64)
        f = lambda v: K.forward_kinematics(v, spec)          # noqa: E731
        jf = torch.func.jacfwd(f)(x)
        jr = torch.func.jacrev(f)(x)
        assert torch.isfinite(jf).all() and torch.allclose(jf, jr, atol=1e-12), label
    print("PASS gradients are finite and forward/reverse agree")


GROUPS = {"fast": [test_generated_models_are_current, test_generated_scenes_are_current,
                   test_the_dh_table_is_the_papers,
                   test_torch_forward_kinematics_matches_drake,
                   test_the_chain_is_the_papers_dh_product,
                   test_the_added_joint_leaves_the_6r_intact,
                   test_det_j_matches_the_closed_form,
                   test_collision_filters_are_what_the_spec_claims,
                   test_home_and_nominal,
                   test_gradients_are_finite_and_correct]}


def main():
    _tiers.run(GROUPS)
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
