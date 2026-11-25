"""The mobile-base campaign's invariants (src/mobile_base/, docs/notes-mobile-base.md).

Each check guards something that would otherwise fail silently:

* the generated platform, table and scene files drifting from their generator;
* the mobile room's compartment table drifting from the shelves the scenes actually weld;
* a base slot assumed rather than read, or the closed-form planar map disagreeing with the
  plant's own joint chain (every program's `X_WA(b)` rests on it);
* the flow-frame calibration depending on where the base is, or differing from the
  fixed-base calibration of the same chart (the conditioning frame trap, moved);
* the learned map's AutoDiffXd chain disagreeing with finite differences in the base and `c`
  columns, or the paired start not reproducing its configuration exactly;
* the start rules not being reproducible per cell, leaving the base box, or the facing rule's
  stand-off not being ../codebase's.

Tiers (tests/_tiers.py): `fast` only, ~1-2 min on one CPU core (it builds two Panda programs
and one soft PCS program).
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import _tiers                                                             # noqa: E402

from pydrake.all import (ExtractGradient, ExtractValue, InitializeAutoDiff, RigidTransform,
                         RollPitchYaw)

from src.mobile_base import generate_scenes as G
from src.mobile_base.program import PlanarTransform, _MobileGeometry
from src.mobile_base.starts import (FACING_REACH, JITTER_REACH_FRACTION, CellRng, MobileGrid)
from src.shelf_regions import (MOBILE_SHELF_WELDS, PointInShelfCompartments,
                               ShelfCompartmentRegions, ShelfRegionsFromPlant)
from src.utils import BuildEnv, HiddenPrints, RepoDir

PANDA_CKPT = "models/panda/panda__n6__step620000.pkl"
SOFT_CKPT = "models/soft12/soft12__n6__step620000.pkl"


def _scene(robot):
    return os.path.join(RepoDir(), G.ROBOTS[robot][1])


class _PlantOnly:
    def __init__(self, diagram):
        self.plant = diagram.GetSubsystemByName("plant")


_CACHE = {}


def _panda_programs():
    if "panda" not in _CACHE:
        from dataclasses import replace
        from src.generic_program import ProgramOptions
        from src.mobile_base.robots import MOBILE_CLASSES
        from src.panda_program import PandaIKProgram
        opts = ProgramOptions(visualize=False, use_float64=True, joint_centering_cost=1e-4,
                              correction_cost_weight=10.0)
        with HiddenPrints():
            d = BuildEnv(None, _scene("panda"))
            L = MOBILE_CLASSES[("panda", "pose", "learned")](d, options=opts,
                                                             checkpoint=PANDA_CKPT)
            J = MOBILE_CLASSES[("panda", "pose", "numerical")](
                d, options=replace(opts, joint_centering_cost=1.0), model=L.ik_solver)
            fixed = PandaIKProgram(BuildEnv(None, os.path.join(
                RepoDir(), "models/panda/panda_finray_collision_hardened.yaml")),
                options=opts, model=L.ik_solver)
        _CACHE["panda"] = (L, J, fixed)
    return _CACHE["panda"]


def _target_of(program, v):
    program.plant.SetPositions(program.plant_context, program.GridToPlantQ(v))
    X = program.frame.CalcPoseInWorld(program.plant_context)
    return np.array([*X.translation(), *X.rotation().ToQuaternion().wxyz()])


## ------------------------------------- files ------------------------------------------ ##

def test_generated_files_match_generator():
    for rel, text in G.Outputs().items():
        with open(os.path.join(RepoDir(), rel)) as f:
            assert f.read() == text, f"{rel} differs from src/mobile_base/generate_scenes.py"
    print("PASS every generated mobile-base file is byte-identical to its generator")


def test_mobile_regions_match_plant():
    with HiddenPrints():
        d = BuildEnv(None, _scene("iiwa"))
    plant = d.GetSubsystemByName("plant")
    ctx = plant.CreateDefaultContext()
    mine = ShelfCompartmentRegions(0.10, welds=MOBILE_SHELF_WELDS)
    theirs = ShelfRegionsFromPlant(plant, ctx, 0.10, welds=MOBILE_SHELF_WELDS)
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1.2, -1.2, 0.9], [1.2, 1.2, 1.9], size=(20000, 3))
    a = [PointInShelfCompartments(p, mine) for p in pts]
    b = [PointInShelfCompartments(p, theirs) for p in pts]
    assert a == b and sum(a) > 100, (sum(a), sum(b))
    print(f"PASS the mobile compartment table matches the plant ({sum(a)} of 20000 inside)")


## ----------------------------------- geometry ----------------------------------------- ##

def test_base_slots_limits_and_planar_map():
    for robot in G.ROBOTS:
        with HiddenPrints():
            d = BuildEnv(None, _scene(robot))
        geo = _MobileGeometry(_PlantOnly(d))     # raises if the planar map disagrees
        plant = d.GetSubsystemByName("plant")
        inst = plant.GetModelInstanceByName(G.MOBILE_MODEL)
        slots = [plant.GetJointByName(j, inst).position_start() for j in G.BASE_JOINTS]
        assert list(geo.base_slots) == slots
        lo, hi = geo.base_limits
        assert np.allclose(lo, [-2, -2, -2 * math.pi]) and np.allclose(hi, [2, 2, 2 * math.pi])
        X = PlanarTransform(np.array([0.3, -0.4, 0.7]), float)
        assert np.allclose(X.translation(), [0.3, -0.4, 0.0])
    print("PASS base slots read from the plant, limits [-2, 2]^2 x [-2pi, 2pi], planar map exact")


## ------------------------------------ programs ---------------------------------------- ##

def test_panda_calibration_is_base_invariant_and_matches_fixed_base():
    L, _, fixed = _panda_programs()
    assert L.flow_frame_spread < 1e-6
    d = L.X_ee_flow.inverse() @ fixed.X_ee_flow
    err = np.linalg.norm(d.translation()) + abs(d.rotation().ToAngleAxis().angle())
    assert err < 1e-6, err
    print(f"PASS mobile calibration (random bases) spread {L.flow_frame_spread:.1e}, "
          f"matches the fixed-base offset to {err:.1e}")


def test_panda_learned_map_start_and_gradients():
    L, J, _ = _panda_programs()
    rng = np.random.default_rng(3)
    grid = MobileGrid(L, lambda v: True, "panda_hand")
    v = grid.Draw(rng)
    with HiddenPrints():
        L.create_prog(_target_of(L, v))
        J.create_prog(_target_of(J, v))
    L.SetStartFromQ(v)
    x0 = L.prog.GetInitialGuess(L.lumped_vars)
    assert np.max(np.abs(L.VarsToQ(x0) - v)) < 1e-6
    ## The fibre property: with the paired start's correction, the end effector IS X(c).
    pos, _ = L.fk(L.VarsToQ(x0))
    c = x0[3:9]
    X_c = RigidTransform(RollPitchYaw(c[3:6]), c[:3]) @ L.X_ee_flow.inverse()
    assert np.max(np.abs(X_c.translation() - pos)) < 1e-6
    x = x0 + 0.01 * rng.standard_normal(len(x0))
    qa = L.VarsToQ(InitializeAutoDiff(x).ravel())
    assert np.max(np.abs(ExtractValue(qa).ravel() - L.VarsToQ(x))) < 1e-10
    Ja, eps = ExtractGradient(qa), 1e-6
    Jfd = np.stack([(L.VarsToQ(x + eps * e) - L.VarsToQ(x - eps * e)) / (2 * eps)
                    for e in np.eye(len(x))], axis=1)
    rel = np.max(np.abs(Ja - Jfd)) / max(1.0, np.max(np.abs(Jfd)))
    assert rel < 1e-6, rel
    J.SetNativeStart(v, rng)
    xj = J.prog.GetInitialGuess(J.lumped_vars)
    assert np.array_equal(J.VarsToQ(xj), v) and np.array_equal(J.Config(xj), v[J.ArmSlots()])
    print(f"PASS learned map: exact paired start, fibre property, AD vs FD rel {rel:.1e}; "
          f"joint space [b | q] lands in the plant's slots")


def test_soft_composed_map():
    from dataclasses import replace
    from src.generic_program import ProgramOptions
    from src.mobile_base.robots import MOBILE_CLASSES
    opts = ProgramOptions(visualize=False, use_float64=True, joint_centering_cost=1e-4,
                          correction_cost_weight=10.0)
    with HiddenPrints():
        d = BuildEnv(None, _scene("soft12"))
        L = MOBILE_CLASSES[("soft", "pose", "learned")](d, options=opts, rung="soft12",
                                                        checkpoint=SOFT_CKPT)
    assert L.flow_frame_spread < 1e-9
    rng = np.random.default_rng(5)
    v = np.concatenate([[0.3, -0.7, 1.1], rng.uniform(-1, 1, L.num_arm_dof)])
    poses = []
    for b in ([0.3, -0.7, 1.1], [0.0, 0.0, 0.0], [-1.5, 1.2, -2.0]):
        L.plant.SetPositions(L.plant_context, L.ConfigToPlantQ(v[3:], np.array(b)))
        poses.append(L.frame.CalcPose(L.plant_context, L.ArmBaseFrame()).GetAsMatrix4())
    assert max(np.max(np.abs(p - poses[0])) for p in poses) < 1e-12
    with HiddenPrints():
        L.create_prog(_target_of(L, v))
    L.SetStartFromQ(v)
    x0 = L.prog.GetInitialGuess(L.lumped_vars)
    assert np.max(np.abs(L.Config(x0) - v[3:])) < 1e-6
    x = x0 + 0.01 * rng.standard_normal(len(x0))
    Ja, eps = ExtractGradient(L.VarsToQ(InitializeAutoDiff(x).ravel())), 1e-6
    Jfd = np.stack([(L.VarsToQ(x + eps * e) - L.VarsToQ(x - eps * e)) / (2 * eps)
                    for e in np.eye(len(x))], axis=1)
    rel = np.max(np.abs(Ja - Jfd)) / max(1.0, np.max(np.abs(Jfd)))
    assert rel < 1e-6, rel
    print(f"PASS soft PCS on the base: tip-in-A base-invariant, exact start, AD vs FD {rel:.1e}")


## ------------------------------------- starts ----------------------------------------- ##

def test_start_rules():
    L, _, _ = _panda_programs()
    free = lambda v: L.collision_free_constraint_eval.Eval(L.GridToPlantQ(v)) < 1
    grid = MobileGrid(L, free, "panda_hand")
    ## A shelf-contained target, as the drivers draw them.
    regions = ShelfCompartmentRegions(0.10, welds=MOBILE_SHELF_WELDS)
    rng = np.random.default_rng(11)
    while True:
        v = grid.Draw(rng)
        L.plant.SetPositions(L.plant_context, v)
        p = L.plant.GetFrameByName("between_fingers").CalcPoseInWorld(L.plant_context)
        if free(v) and PointInShelfCompartments(p.translation(), regions):
            break
    X_WT = L.frame.CalcPoseInWorld(L.plant_context)
    for rule in ("uniform", "facing", "home"):
        a, _, ok_a = grid.Start(rule, X_WT, CellRng(1, 2, 3, rule))
        b, _, _ = grid.Start(rule, X_WT, CellRng(1, 2, 3, rule))
        assert np.array_equal(a, b), rule
        base, _ = L.SplitPlantQ(a)
        assert ok_a == free(a)
        assert np.all(base >= grid.base_lower) and np.all(base <= grid.base_upper)
        if rule == "home":
            assert np.array_equal(base, np.array(G.HOME_BASE))
        if rule == "facing":
            reach = np.linalg.norm(base[:2] - X_WT.translation()[:2])
            lo, hi = (FACING_REACH * f for f in JITTER_REACH_FRACTION)
            assert lo - 1e-9 <= reach <= hi + 1e-9, reach
    other, _, _ = grid.Start("uniform", X_WT, CellRng(1, 2, 4, "uniform"))
    assert not np.array_equal(other, a)
    ## The fallback: with nothing collision-free, the last draw comes back flagged, not a raise.
    never = MobileGrid(L, lambda v: False, "panda_hand")
    q, draws, ok = never.Start("facing", X_WT, CellRng(1, 0, 0, "facing"))
    assert not ok and draws > 0 and q is not None
    print("PASS the three base-start rules: reproducible per cell, flagged when in collision, "
          "inside the box, home at (1.5, 0, 0), facing stand-off in 1.0 m x [0.6, 1.4]; the "
          "no-free-start fallback returns the last draw flagged")


def test_driver_refusals():
    from src.mobile_base.harness import CheckMobileArgs
    from src.target_screening import MAX_CONSECUTIVE_REJECTIONS
    ns = lambda **kw: argparse.Namespace(**{**dict(
        scene="mobile", base_start="uniform", arms="learned,numerical", guess_filter="none",
        max_target_rejections=MAX_CONSECUTIVE_REJECTIONS), **kw})
    for bad in (dict(base_start=None), dict(arms="learned,analytic"),
                dict(scene="hardened"), dict(guess_filter="charted")):
        try:
            CheckMobileArgs(ns(**bad))
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")
    args = ns()
    assert CheckMobileArgs(args) and args.max_target_rejections == 2_000_000
    print("PASS the drivers refuse a mobile run without a rule, with an analytic arm, or a "
          "rule without the mobile scene; the rejection cap rises to 2e6")


GROUPS = {"fast": [test_generated_files_match_generator,
                   test_mobile_regions_match_plant,
                   test_base_slots_limits_and_planar_map,
                   test_driver_refusals,
                   test_panda_calibration_is_base_invariant_and_matches_fixed_base,
                   test_panda_learned_map_start_and_gradients,
                   test_start_rules,
                   test_soft_composed_map]}


def main():
    _tiers.run(GROUPS)
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
