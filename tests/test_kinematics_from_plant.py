"""The batched torch FK extracted from a plant, against the plant itself.

DRAKE IS THE ORACLE, at freshly drawn configurations, on the very scenes the solver
builds: both hardened rigid scenes, a Panda grasp scene with its welded target mug, and
the screw arm's scene with its limit repair applied. A golden file would miss a scene
change, a Drake pin move and a parser regression alike; this runs in the solver's own
environment on every invocation.

Four things are pinned. (1) Every body's world pose and the task frames
(`between_fingers`, the gripper body) agree with `CalcRelativeTransform` to 1e-13, on
CPU and, where there is one, on CUDA. (2) `rpy_from_quat` is Drake's OWN algorithm --
agreement to 1e-12 on random orientations and the same degradation near gimbal lock --
because the orientation rows difference rpy angles and two equivalent-but-different
charts would disagree by more than the row tolerance exactly where it matters. (3) The
autograd Jacobian equals central differences and `jacfwd == jacrev`. (4) The joints are
replayed in TOPOLOGICAL order with slots read from `position_start()`: the Panda scene
registers the finger welds before the hand weld they hang from, and the test asserts
that inversion is present so a regression to declaration order would be caught by a
scene that actually exercises it.

No pytest config in this repo, so these are plain `test_*` functions with a `__main__`
driver.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.multibody.tree import BodyIndex

from src.generic_program import orientation_error_rpy
from src.screw_arm.limits import ApplyScrewJointLimits, RequireFiniteLimits
from src.screw_arm.params import PRIMARY as SCREW_PRIMARY, GetSpec
from src.svgd.kinematics_from_plant import (BatchedFK, KinematicTree, canonical_quat, pose7,
                                            quat_from_rpy_batched, rpy_from_quat,
                                            wrap_residual)
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints, RepoDir

REPO = RepoDir()
PANDA_YAML = os.path.join(REPO, "models/panda/panda_finray_collision_hardened.yaml")
IIWA_YAML = os.path.join(REPO, "models/iiwa14/iiwa14_collision_hardened.yaml")
SCREW_YAML = os.path.join(REPO, f"models/{SCREW_PRIMARY}/{SCREW_PRIMARY}_collision_hardened.yaml")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_CACHE = {}


## -- scenes ------------------------------------------------------------------------------


class _Scene:
    """A plant, a context and the two task frames, by name."""

    def __init__(self, name, diagram, gripper, repair=None):
        self.name = name
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        if repair is not None:
            repair(self.plant)
        self.context = self.plant.CreateDefaultContext()
        self.gripper = self.plant.GetBodyByName(gripper)
        self.between_fingers = self.plant.GetFrameByName("between_fingers")
        self.lower = np.asarray(self.plant.GetPositionLowerLimits(), dtype=float)
        self.upper = np.asarray(self.plant.GetPositionUpperLimits(), dtype=float)
        RequireFiniteLimits(self.plant, name)

    def random_q(self, n, seed):
        rng = np.random.default_rng(seed)
        return rng.uniform(self.lower, self.upper, size=(n, self.plant.num_positions()))

    def home_q(self):
        return np.asarray(self.plant.GetPositions(self.plant.CreateDefaultContext()), dtype=float)


class _MugStub:
    """The three things `GenerateDiagramWithMug` reads off a program."""

    def __init__(self, scene):
        self.plant = scene.plant
        self.plant_context = scene.context
        self.frame = scene.between_fingers

    def SetPositions(self, q):
        self.plant.SetPositions(self.plant_context, q)


def _screw_repair(plant):
    ApplyScrewJointLimits(plant, GetSpec(SCREW_PRIMARY))


def scenes():
    if "scenes" not in _CACHE:
        with HiddenPrints():
            panda = _Scene("panda", BuildEnv(None, PANDA_YAML), "panda_hand")
            iiwa = _Scene("iiwa14", BuildEnv(None, IIWA_YAML), "hand")
            screw = _Scene(SCREW_PRIMARY, BuildEnv(None, SCREW_YAML), "hand", repair=_screw_repair)
            ## A grasp scene: the mug welded at the gripper pose of a random configuration,
            ## exactly as the benchmark builds one per target.
            q_mug = panda.random_q(1, seed=7)[0]
            diagram_mug, _ = GenerateDiagramWithMug(q_mug, _MugStub(panda), PANDA_YAML, None)
            panda_mug = _Scene("panda+mug", diagram_mug, "panda_hand")
        assert panda_mug.plant.HasBodyNamed("mug_body_link")
        _CACHE["scenes"] = [panda, iiwa, panda_mug, screw]
    return _CACHE["scenes"]


def _drake_body_poses(scene, q):
    """`(quat [nb, 4] wxyz, pos [nb, 3])` from Drake at one configuration."""
    plant, ctx = scene.plant, scene.context
    plant.SetPositions(ctx, q)
    quats, poss = [], []
    for i in range(plant.num_bodies()):
        X = plant.CalcRelativeTransform(ctx, plant.world_frame(),
                                        plant.get_body(BodyIndex(i)).body_frame())
        quats.append(X.rotation().ToQuaternion().wxyz())
        poss.append(X.translation())
    return np.asarray(quats), np.asarray(poss)


def _quat_diff_up_to_sign(a, b):
    """Max over the batch of `min(|a - b|, |a + b|)`."""
    return float(np.minimum(np.abs(a - b).max(-1), np.abs(a + b).max(-1)).max())


## -- 1. every body against Drake -------------------------------------------------------


def test_body_poses_match_drake_on_every_scene():
    n = 256
    for scene in scenes():
        tree = KinematicTree.from_plant(scene.plant)
        assert tree.num_bodies == scene.plant.num_bodies()
        assert tree.num_positions == scene.plant.num_positions()
        Q = scene.random_q(n, seed=1)
        ref_q, ref_p = zip(*(_drake_body_poses(scene, q) for q in Q))
        ref_q, ref_p = np.stack(ref_q), np.stack(ref_p)       # [n, nb, 4], [n, nb, 3]
        for device in DEVICES:
            fk = BatchedFK(tree, dtype=torch.float64, device=device)
            quat, pos = fk.body_poses(torch.as_tensor(Q, dtype=torch.float64, device=device))
            assert quat.shape == (n, tree.num_bodies, 4) and pos.shape == (n, tree.num_bodies, 3)
            assert quat.dtype == torch.float64 and quat.device.type == device
            dq = _quat_diff_up_to_sign(quat.cpu().numpy(), ref_q)
            dp = float(np.abs(pos.cpu().numpy() - ref_p).max())
            assert dp <= 1e-13, f"{scene.name}/{device}: body position off by {dp:.2e}"
            assert dq <= 1e-13, f"{scene.name}/{device}: body quaternion off by {dq:.2e}"
            ## The quaternion itself is not unit-normalised anywhere on the chain; it must
            ## stay unit by composition alone.
            norm_err = float((torch.linalg.norm(quat, dim=-1) - 1.0).abs().max())
            assert norm_err <= 1e-13, f"{scene.name}/{device}: |quat| - 1 = {norm_err:.2e}"
            print(f"     {scene.name:10s} {device:4s} {tree.num_bodies:2d} bodies x {n}: "
                  f"pos {dp:.1e}  quat {dq:.1e}  |q|-1 {norm_err:.1e}")
    print("PASS every body's world pose matches CalcRelativeTransform on every scene")


def test_task_frames_match_drake():
    """`between_fingers` (a FixedOffsetFrame) and the gripper body frame, through
    `frame_pose_from_q`, plus the canonical quaternion against `ToQuaternion().wxyz()`."""
    n = 128
    for scene in scenes():
        tree = KinematicTree.from_plant(scene.plant)
        Q = scene.random_q(n, seed=2)
        for frame, label in ((scene.between_fingers, "between_fingers"),
                             (scene.gripper.body_frame(), scene.gripper.name())):
            ref = []
            for q in Q:
                scene.plant.SetPositions(scene.context, q)
                X = frame.CalcPoseInWorld(scene.context)
                ref.append(np.concatenate((X.translation(), X.rotation().ToQuaternion().wxyz())))
            ref = np.stack(ref)
            for device in DEVICES:
                fk = BatchedFK(tree, dtype=torch.float64, device=device)
                quat, pos = fk.frame_pose_from_q(
                    torch.as_tensor(Q, dtype=torch.float64, device=device), frame)
                got = pose7(quat, pos).cpu().numpy()
                ## pose7 is CANONICAL, so this is a plain difference, not up to sign.
                err = float(np.abs(got - ref).max())
                assert err <= 1e-13, f"{scene.name}/{device}/{label}: pose7 off by {err:.2e}"
                assert bool((canonical_quat(quat)[:, 0] >= 0).all())
                print(f"     {scene.name:10s} {device:4s} {label:16s} pose7 {err:.1e}")
    print("PASS task frames and canonical quaternions match Drake")


## -- 2. the orientation algebra ---------------------------------------------------------


def _random_unit_quats(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _drake_rpy(q_wxyz):
    return RollPitchYaw(RotationMatrix(Quaternion(q_wxyz))).vector()


def test_rpy_from_quat_is_drakes_algorithm():
    Q = _random_unit_quats(10000, seed=3)
    ref = np.stack([_drake_rpy(q) for q in Q])
    got = rpy_from_quat(torch.as_tensor(Q, dtype=torch.float64, device="cpu")).numpy()
    err = float(np.abs(got - ref).max())
    assert err <= 1e-12, f"rpy_from_quat differs from Drake by {err:.2e}"
    print(f"     10k random orientations: max |rpy - Drake rpy| = {err:.1e}")

    ## Near gimbal lock. Roll and yaw are individually ill-conditioned there; what is
    ## asserted is that both algorithms still describe the SAME rotation (the
    ## reconstructed matrices agree), and the raw angle difference is REPORTED.
    rng = np.random.default_rng(4)
    rpy = np.stack((rng.uniform(-math.pi, math.pi, 200),
                    rng.choice([-1, 1], 200) * (math.pi / 2) + rng.uniform(-1e-6, 1e-6, 200),
                    rng.uniform(-math.pi, math.pi, 200)), axis=1)
    Qg = np.stack([RotationMatrix(RollPitchYaw(r)).ToQuaternion().wxyz() for r in rpy])
    ref = np.stack([_drake_rpy(q) for q in Qg])
    got = rpy_from_quat(torch.as_tensor(Qg, dtype=torch.float64, device="cpu")).numpy()
    raw = float(np.abs(got - ref).max())
    pitch_err = float(np.abs(got[:, 1] - ref[:, 1]).max())
    mat_err = max(float(np.abs(RotationMatrix(RollPitchYaw(g)).matrix()
                               - RotationMatrix(RollPitchYaw(r)).matrix()).max())
                  for g, r in zip(got, ref))
    assert np.isfinite(got).all()
    assert pitch_err <= 1e-12, f"pitch differs by {pitch_err:.2e} within 1e-6 of gimbal lock"
    assert mat_err <= 1e-9, f"near gimbal lock the two rpy triples describe rotations {mat_err:.2e} apart"
    print(f"     200 within 1e-6 of pitch = +-pi/2: max |rpy - Drake| = {raw:.1e} "
          f"(pitch {pitch_err:.1e}; reconstructed R differ by {mat_err:.1e})")
    print("PASS rpy_from_quat reproduces RollPitchYaw(RotationMatrix(Quaternion(q)))")


def test_quat_from_rpy_batched_round_trips_through_drake():
    rng = np.random.default_rng(5)
    rpy = np.stack((rng.uniform(-math.pi, math.pi, 5000),
                    rng.uniform(-math.pi / 2, math.pi / 2, 5000),
                    rng.uniform(-math.pi, math.pi, 5000)), axis=1)
    ref = np.stack([RotationMatrix(RollPitchYaw(r)).ToQuaternion().wxyz() for r in rpy])
    got = canonical_quat(quat_from_rpy_batched(
        torch.as_tensor(rpy, dtype=torch.float64, device="cpu"))).numpy()
    err = float(np.abs(got - ref).max())
    assert err <= 1e-14, f"quat_from_rpy_batched differs from Drake by {err:.2e}"
    ## And back through rpy_from_quat: the chart round-trips away from gimbal lock.
    back = rpy_from_quat(torch.as_tensor(got, dtype=torch.float64, device="cpu")).numpy()
    rt = float(np.abs(wrap_residual(torch.as_tensor(back - rpy, dtype=torch.float64,
                                                    device="cpu")).numpy()).max())
    assert rt <= 1e-12, f"rpy -> quat -> rpy round trip off by {rt:.2e}"
    print(f"     5k rpy: quat vs Drake {err:.1e}, rpy round trip {rt:.1e}")
    print("PASS quat_from_rpy_batched matches RotationMatrix(RollPitchYaw).ToQuaternion()")


def test_wrap_residual_matches_orientation_error_rpy():
    """`orientation_error_rpy(identity, -v)` returns `0 - (-v) = v`, wrapped by the loop
    this helper reproduces -- so the comparison is against the program's own function,
    not a re-typed copy of its loop. Exact multiples of 2pi are included on purpose:
    they are where `round` has to land on an integer."""
    rng = np.random.default_rng(6)
    values = np.concatenate((rng.uniform(-30, 30, 3000),
                             2 * math.pi * np.arange(-5, 6),
                             2 * math.pi * np.arange(-5, 6) + math.pi,
                             [0.0, math.pi, -math.pi, math.nextafter(math.pi, 4), 1e-300]))
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    ref = np.array([orientation_error_rpy(identity, [-v, 0.0, 0.0])[0] for v in values])
    got = wrap_residual(torch.as_tensor(values, dtype=torch.float64, device="cpu")).numpy()
    err = float(np.abs(got - ref).max())
    assert err == 0.0, f"wrap_residual differs from orientation_error_rpy by {err:.2e}"
    assert bool((got > -math.pi - 1e-12).all() and (got <= math.pi + 1e-12).all())
    print(f"     {len(values)} values incl. exact multiples of 2pi: bitwise equal")
    print("PASS wrap_residual is orientation_error_rpy's wrap")


## -- 3. gradients -------------------------------------------------------------------------


def test_jacobians_match_central_differences():
    h = 1e-6
    for scene in scenes():
        tree = KinematicTree.from_plant(scene.plant)
        fk = BatchedFK(tree, dtype=torch.float64, device="cpu")
        frame = scene.between_fingers

        def f(q):
            quat, pos = fk.frame_pose_from_q(q.unsqueeze(0), frame)
            ## The raw quaternion rather than pose7: canonicalisation flips sign at w = 0,
            ## and a frame rotated 180 deg from the world at home sits exactly there.
            return torch.cat((pos[0], quat[0]))

        def f_rpy(q):
            quat, _ = fk.frame_pose_from_q(q.unsqueeze(0), frame)
            return rpy_from_quat(quat)[0]

        for label, q in (("home", scene.home_q()), ("random", scene.random_q(1, seed=9)[0])):
            qt = torch.as_tensor(q, dtype=torch.float64, device="cpu")
            J_rev = torch.func.jacrev(f)(qt)
            J_fwd = torch.func.jacfwd(f)(qt)
            J_fd = torch.zeros_like(J_rev)
            for i in range(len(q)):
                e = torch.zeros_like(qt)
                e[i] = h
                J_fd[:, i] = (f(qt + e) - f(qt - e)) / (2 * h)
            fd_err = float((J_rev - J_fd).abs().max())
            fr_err = float((J_fwd - J_rev).abs().max())
            assert fd_err <= 1e-7, f"{scene.name}/{label}: jacrev vs central differences {fd_err:.2e}"
            assert fr_err <= 1e-12, f"{scene.name}/{label}: jacfwd vs jacrev {fr_err:.2e}"
            assert torch.isfinite(J_rev).all()

            ## The orientation rows use rpy; its Jacobian is checked too, where the chart
            ## is not at gimbal lock (the test skips it there and says so).
            pitch = float(f_rpy(qt)[1])
            if abs(abs(pitch) - math.pi / 2) > 1e-2:
                Jr = torch.func.jacrev(f_rpy)(qt)
                Jr_fd = torch.zeros_like(Jr)
                for i in range(len(q)):
                    e = torch.zeros_like(qt)
                    e[i] = h
                    Jr_fd[:, i] = wrap_residual(f_rpy(qt + e) - f_rpy(qt - e)) / (2 * h)
                rpy_err = float((Jr - Jr_fd).abs().max())
                assert rpy_err <= 1e-6, f"{scene.name}/{label}: rpy jacrev vs FD {rpy_err:.2e}"
                rpy_note = f"rpy-FD {rpy_err:.1e}"
            else:
                rpy_note = f"rpy at gimbal lock (pitch {pitch:+.4f}), skipped"
            print(f"     {scene.name:10s} {label:6s} jacrev-FD {fd_err:.1e}  "
                  f"jacfwd-jacrev {fr_err:.1e}  {rpy_note}")
    print("PASS autograd Jacobians match central differences; jacfwd == jacrev")


def test_rpy_gradient_is_finite_at_drakes_singular_guard():
    """Drake zeroes `atan2(yA, xA)` when both arguments are within eps of 0. Applying
    that guard with `torch.where` on the RESULT leaves `0 * nan = nan` in the backward
    pass; the module substitutes safe arguments instead, and this pins it."""
    ## q with e1 + e3 = 0 and e0 - e2 = 0 exactly: e.g. the 180-degree rotation about y.
    q = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64, device="cpu",
                     requires_grad=True)
    rpy = rpy_from_quat(q)
    (grad,) = torch.autograd.grad(rpy.sum(), q)
    assert torch.isfinite(rpy).all() and torch.isfinite(grad).all(), (rpy, grad)
    ref = _drake_rpy(q.detach()[0].numpy())
    assert float(np.abs(rpy.detach()[0].numpy() - ref).max()) <= 1e-12
    print("PASS rpy_from_quat has a finite gradient where Drake's singular guard fires")


## -- 4. topological order ----------------------------------------------------------------


def test_tree_is_topological_and_slots_come_from_position_start():
    for scene in scenes():
        tree = KinematicTree.from_plant(scene.plant)
        plant = scene.plant
        ## Parents before children.
        have = {0}
        for r in tree.joints:
            assert r.parent in have, f"{scene.name}: {r.name} replayed before its parent's pose"
            have.add(r.child)
        assert have == set(range(tree.num_bodies))
        ## Slots are the plant's own, and cover q exactly once.
        slots = []
        for r in tree.joints:
            joint = plant.GetJointByName(r.name) if r.kind != "weld" else None
            if r.position_slot is None:
                assert r.kind == "weld"
                continue
            assert r.position_slot == joint.position_start(), (scene.name, r.name)
            slots.append(r.position_slot)
        assert sorted(slots) == list(range(plant.num_positions()))
        ## Anchored = welded to world, world included, and nothing moving reaches them.
        for r in tree.joints:
            if r.child in tree.anchored:
                assert r.kind == "weld" and r.parent in tree.anchored, (scene.name, r.name)
        assert 0 in tree.anchored
        print(f"     {scene.name:10s} {len(tree.joints)} joints, {len(tree.anchored)} anchored "
              f"bodies, slots {slots}")

    ## The Panda scene is the one where declaration order would have been WRONG: the
    ## finger welds (parent `panda_hand`) are registered before the hand weld. If that
    ## ever stops being so, this scene no longer exercises the walk and the assertion
    ## should move to one that does.
    panda = scenes()[0]
    plant = panda.plant
    index_of = {plant.get_joint(ji).name(): int(ji) for ji in plant.GetJointIndices()}
    tree = KinematicTree.from_plant(plant)
    incoming = {r.child: r for r in tree.joints}
    inverted = [r.name for r in tree.joints
                if r.parent != 0 and index_of[incoming[r.parent].name] > index_of[r.name]]
    assert inverted, (
        "no Panda joint is declared before its parent's joint any more; the topological "
        "walk is no longer exercised by this scene -- point the assertion at one that does.")
    print(f"     Panda joints declared before their parent's joint: {inverted}")
    print("PASS joints are replayed in topological order with slots from position_start()")


def test_from_plant_refuses_a_floating_body():
    """The soft PCS arm's sub-links are floating bodies; the extractor must say so by
    name rather than return a tree that is missing them."""
    from pydrake.multibody.plant import MultibodyPlant
    plant = MultibodyPlant(0.0)
    from pydrake.multibody.tree import SpatialInertia
    plant.AddRigidBody("loose", SpatialInertia.SolidSphereWithMass(1.0, 0.1))
    plant.Finalize()
    try:
        KinematicTree.from_plant(plant)
    except (NotImplementedError, RuntimeError) as e:
        assert "loose" in str(e) or "Floating" in str(e) or "floating" in str(e).lower(), str(e)
        print(f"     refused: {str(e).splitlines()[0][:90]}...")
    else:
        raise AssertionError("from_plant accepted a plant with a floating body")
    print("PASS from_plant raises on a robot whose configuration is not the plant's q")


if __name__ == "__main__":
    test_body_poses_match_drake_on_every_scene()
    test_task_frames_match_drake()
    test_rpy_from_quat_is_drakes_algorithm()
    test_quat_from_rpy_batched_round_trips_through_drake()
    test_wrap_residual_matches_orientation_error_rpy()
    test_jacobians_match_central_differences()
    test_rpy_gradient_is_finite_at_drakes_singular_guard()
    test_tree_is_topological_and_slots_come_from_position_start()
    test_from_plant_refuses_a_floating_body()
    print("ALL PASS")
