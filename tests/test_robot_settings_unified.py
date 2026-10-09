"""Pin the cross-robot settings unified on 2026-10-08, and the control that undoes them.

No pytest config in this repo; run by hand:

    python tests/test_robot_settings_unified.py

An inventory ahead of re-measuring the grasp rows found five settings that differed between
robots for no deliberate reason. The PI's decisions:

  1. the native grasp start for `c` is `mug.middle @ X_grasp_ee` as xyz + rpy on EVERY robot
     (the iiwa and screw arms seeded `[mug xyz, 0, 0, 0]`);
  2. the joint-space arm's variable bound on `q` is `ConfigLimits()` on every robot (the
     Panda used +-10 rad, the iiwa a hand-typed table);
  3. `q_nominal` is a NONSINGULAR HOME pose per robot, held in one place per robot (it was
     zeros everywhere: outside q4's range on the Panda and singular on every S-R-S arm);
  4. the latent trust region is NOT unified but A/B-tested through one flag,
     `latent_trust_region_rule`, which sizes it `round(sqrt(dim_latent) + 1.5, 2)` from the
     loaded chart (moves the Panda 4.0 -> 4.15 and the iiwa 4.3 -> 4.33, nothing else);
  5. the grasp `c` box is centred on that same flow-frame pose, +-c_position_slack.

`legacy_robot_settings=True` restores 1, 2, 3 and 5 together and is the control arm of the
re-run, so its old values are pinned here too: a control that drifts is not a control.

This reads the bounds and guesses Drake was actually handed, never the docstrings. Every
robot's programs are built: the Panda and iiwa on their local `n6` charts, the screw, soft
PCS and GVS arms on UNTRAINED in-process charts (their trained charts are not local), which
exercises every bound- and start-forming path and says nothing about solve quality.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generic_program import ProgramOptions                          # noqa: E402
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []
CHECKS = [0]
SIGMA_MIN_FLOOR = 0.05


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          {detail}")
        FAILURES.append(name)


def find(prog, description):
    for b in prog.GetAllConstraints():
        if b.evaluator().get_description() == description:
            return b
    return None


def bounds_of(binding):
    e = binding.evaluator()
    return np.asarray(e.lower_bound(), dtype=float), np.asarray(e.upper_bound(), dtype=float)


def untrained_chart(robot_name, width_arch):
    """An in-process, untrained IKFlow chart of the robot's legacy architecture."""
    import torch
    from ikflow.ikflow_solver import IKFlowSolver
    from ikflow.model import IkflowModelParameters
    from jrl.robots import get_robot
    import src.register_robots  # noqa: F401
    parameters = IkflowModelParameters()
    parameters.__dict__.update(dict(width_arch, nb_nodes=4))
    torch.manual_seed(0)
    return IKFlowSolver(parameters, get_robot(robot_name))


def sigma_min(program, q_plant):
    """Smallest singular value of the 6 x n manipulator Jacobian at the flow frame."""
    from pydrake.all import JacobianWrtVariable
    plant, context = program.plant, program.plant_context
    plant.SetPositions(context, q_plant)
    J = plant.CalcJacobianSpatialVelocity(
        context, JacobianWrtVariable.kQDot, program.frame_for_flow, np.zeros(3),
        plant.world_frame(), plant.world_frame())
    return float(np.linalg.svd(J[:, :program.num_arm_dof], compute_uv=False)[-1])


def collision_value(program, q_plant):
    return float(program.collision_free_constraint_eval.Eval(q_plant)[0])


def grasp_pose_xyz_rpy(program, mug):
    X = mug.middle @ program.X_grasp_ee
    return np.concatenate([X.translation(), X.rotation().ToRollPitchYaw().vector()])


class Robot:
    """How to build one robot's four programs, under given options."""

    def __init__(self, name, scene, learned_pose, numerical_pose, learned_mug, numerical_mug,
                 kwargs, legacy_q_bounds, legacy_mug_centred_c, expect_rule_radius,
                 nominal_sigma=True):
        self.name = name
        self.scene = os.path.join(REPO, scene)
        self.classes = dict(learned_pose=learned_pose, numerical_pose=numerical_pose,
                            learned_mug=learned_mug, numerical_mug=numerical_mug)
        self.kwargs = kwargs
        self.legacy_q_bounds = legacy_q_bounds
        self.legacy_mug_centred_c = legacy_mug_centred_c
        self.expect_rule_radius = expect_rule_radius
        self.nominal_sigma = nominal_sigma

    def build(self, opts):
        with HiddenPrints():
            diagram = BuildEnv(meshcat=None, directives_file=self.scene)
            pose_sampler = self.classes["learned_pose"](diagram, options=opts, **self.kwargs)
            pose_sampler.create_prog()
            rng = np.random.default_rng(0)
            cfg = pose_sampler.SampleConfiguration(rng)
            target = np.concatenate(pose_sampler.fk(pose_sampler.ConfigToPlantQ(cfg)))
            mug_sampler = self.classes["learned_mug"](diagram, options=opts, **self.kwargs)
            mug_sampler.create_prog()
            q_m = mug_sampler.ConfigToPlantQ(mug_sampler.SampleConfiguration(rng))
            diagram_with_mug, mug = GenerateDiagramWithMug(q_m, mug_sampler, self.scene, None)
            out = {}
            for key, cls in self.classes.items():
                d = diagram_with_mug if key.endswith("mug") else diagram
                p = cls(d, options=opts, **self.kwargs)
                if key.endswith("mug"):
                    p.create_prog(target_mug=mug)
                else:
                    p.create_prog(target)
                out[key] = p
        return out, mug


def robots():
    from src.flow_loading import LEGACY_ARCH_BY_ROBOT, LoadFlowSolver
    from src.panda_program import (PandaIKProgram, PandaIKProgramNumerical, PandaMugProgram,
                                   PandaMugProgramNumerical, PANDA_LEGACY_Q_BOUNDS)
    from src.iiwa_program import (Iiwa14IKProgram, Iiwa14IKProgramNumerical, IiwaMugProgram,
                                  IiwaMugProgramNumerical)
    from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper
    from src.screw_arm.params import PRIMARY as SCREW, GetSpec as ScrewSpec
    from src.screw_arm_program import (ScrewArmIKProgram, ScrewArmIKProgramNumerical,
                                       ScrewArmMugProgram, ScrewArmMugProgramNumerical)
    from src.soft_arm_program import (SoftArmIKProgram, SoftArmIKProgramNumerical,
                                      SoftArmMugProgram, SoftArmMugProgramNumerical)
    from src.gvs_arm.params import PRIMARY as GVS
    from src.gvs_arm_program import (GvsArmIKProgram, GvsArmIKProgramNumerical,
                                     GvsArmMugProgram, GvsArmMugProgramNumerical)

    with HiddenPrints():
        panda_chart = LoadFlowSolver(
            "panda", os.path.join(REPO, "models/panda/panda__n6__step620000.pkl"))
        iiwa_chart = LoadFlowSolver(
            "iiwa14", os.path.join(REPO, "models/iiwa14/iiwa14__n6__step620000.pkl"))
        screw_chart = untrained_chart(SCREW, LEGACY_ARCH_BY_ROBOT[SCREW])
        soft_chart = untrained_chart("soft12", LEGACY_ARCH_BY_ROBOT["soft12"])
        gvs_chart = untrained_chart(GVS, LEGACY_ARCH_BY_ROBOT[GVS])
    print("  note  screw7, soft12 and GVS charts are UNTRAINED in-process charts (their trained "
          "charts are not local); only bounds, starts and nominals are checked on them")
    return [
        Robot("panda", "models/panda/panda_finray_collision_hardened.yaml",
              PandaIKProgram, PandaIKProgramNumerical, PandaMugProgram, PandaMugProgramNumerical,
              dict(model=panda_chart), PANDA_LEGACY_Q_BOUNDS, False, 4.15),
        Robot("iiwa14", "models/iiwa14/iiwa14_collision_hardened.yaml",
              Iiwa14IKProgram, Iiwa14IKProgramNumerical, IiwaMugProgram, IiwaMugProgramNumerical,
              dict(model=iiwa_chart), (iiwa_limits_lower, iiwa_limits_upper), True, 4.33),
        Robot(SCREW, f"models/{SCREW}/{SCREW}_collision_hardened.yaml",
              ScrewArmIKProgram, ScrewArmIKProgramNumerical, ScrewArmMugProgram,
              ScrewArmMugProgramNumerical, dict(robot=SCREW, model=screw_chart),
              None, True, ScrewSpec(SCREW).latent_trust_region),
        Robot("soft12", "models/soft12/soft12_collision_hardened.yaml",
              SoftArmIKProgram, SoftArmIKProgramNumerical, SoftArmMugProgram,
              SoftArmMugProgramNumerical, dict(rung="soft12", model=soft_chart),
              None, False, 4.96, nominal_sigma=False),
        Robot(GVS, f"models/{GVS}/{GVS}_collision_hardened.yaml",
              GvsArmIKProgram, GvsArmIKProgramNumerical, GvsArmMugProgram,
              GvsArmMugProgramNumerical, dict(rung=GVS, model=gvs_chart),
              None, False, 4.5, nominal_sigma=False),
    ]


def check_unified(robot, programs, mug, opts):
    name = robot.name
    learned_mug = programs["learned_mug"]
    want_c = grasp_pose_xyz_rpy(learned_mug, mug)
    slack = opts.c_position_slack

    for key in ("numerical_pose", "numerical_mug"):
        p = programs[key]
        b = find(p.prog, "QBoundingBoxConstraint")
        lo, hi = (np.asarray(x, dtype=float)[:p.num_arm_dof] for x in p.ConfigLimits())
        if b is None:
            check(f"{name} {key}: QBoundingBoxConstraint present", False, "binding not found")
            continue
        blo, bhi = bounds_of(b)
        check(f"{name} {key}: q bound == ConfigLimits()",
              np.array_equal(blo, lo) and np.array_equal(bhi, hi),
              f"bound {blo}..{bhi}\n          limits {lo}..{hi}")
        check(f"{name} {key}: q bound is finite",
              bool(np.all(np.isfinite(blo)) and np.all(np.isfinite(bhi))), f"{blo}..{bhi}")

    ## q_nominal, on every program: one value per robot, inside the limits, nonsingular,
    ## collision-free in the hardened scene.
    nominals = {k: np.asarray(p.q_nominal, dtype=float) for k, p in programs.items()}
    first = next(iter(nominals.values()))
    check(f"{name}: q_nominal identical across all four programs",
          all(np.array_equal(v, first) for v in nominals.values()),
          f"{ {k: v.tolist() for k, v in nominals.items()} }")
    p = programs["learned_pose"]
    lo, hi = (np.asarray(x, dtype=float)[:p.num_arm_dof] for x in p.ConfigLimits())
    margin = float(np.min(np.minimum(first - lo, hi - first)))
    check(f"{name}: q_nominal {np.round(first, 3).tolist()} inside ConfigLimits "
          f"(closest joint {margin:.3f} from a limit)", margin > 0.0,
          f"q_nominal={first.tolist()} limits={lo.tolist()}..{hi.tolist()}")
    q_plant = p.ConfigToPlantQ(first)
    if robot.nominal_sigma:
        s = sigma_min(p, q_plant)
        check(f"{name}: q_nominal is nonsingular (sigma_min {s:.4f} > {SIGMA_MIN_FLOOR})",
              s > SIGMA_MIN_FLOOR, f"sigma_min={s}")
        s0 = sigma_min(p, p.ConfigToPlantQ(np.zeros(p.num_arm_dof)))
        print(f"  note  {name}: for the record, zeros has sigma_min {s0:.2e}")
    else:
        print(f"  note  {name}: straight rod is the natural nominal; no manipulator-Jacobian "
              f"check on a floating-body plant")
    cv = collision_value(p, q_plant)
    check(f"{name}: q_nominal is collision-free in the hardened scene (value {cv:.3f} < 1)",
          cv < 1.0, f"collision value {cv}")

    ## The grasp start and box, on the learned mug program.
    c0 = learned_mug.prog.GetInitialGuess(learned_mug.c)
    check(f"{name} learned_mug: native c start == [xyz, rpy] of mug.middle @ X_grasp_ee",
          np.allclose(c0, want_c, atol=1e-12), f"start {c0}\n          want  {want_c}")
    b = find(learned_mug.prog, "CBoxConstraint")
    if b is None:
        check(f"{name} learned_mug: CBoxConstraint present", False, "binding not found")
    else:
        blo, bhi = bounds_of(b)
        check(f"{name} learned_mug: c box centred on that pose's xyz with half-width "
              f"{slack}",
              np.allclose(blo[:3], want_c[:3] - slack) and np.allclose(bhi[:3], want_c[:3] + slack),
              f"box {blo[:3]}..{bhi[:3]} centre {want_c[:3]}")
        check(f"{name} learned_mug: c box leaves orientation free (+-2pi)",
              np.allclose(blo[3:], -2 * np.pi) and np.allclose(bhi[3:], 2 * np.pi),
              f"{blo[3:]}..{bhi[3:]}")
        check(f"{name} learned_mug: c box is a general constraint, not a bounding box",
              all(bb.evaluator().get_description() != "CBoxConstraint"
                  for bb in learned_mug.prog.bounding_box_constraints()),
              "bound_push would project the paired start")
    check(f"{name} learned_mug: X_grasp_ee is a real standoff "
          f"({np.linalg.norm(learned_mug.X_grasp_ee.translation()):.3f} m), so the centre moved",
          np.linalg.norm(learned_mug.X_grasp_ee.translation()) > 0.05,
          "a zero standoff would make items 1 and 5 vacuous on this robot")
    settings = learned_mug.RobotSettings()
    check(f"{name} learned_mug: RobotSettings records the unified state",
          settings["legacy_robot_settings"] is False
          and settings["grasp_c_box_centre"] == "flow_pose"
          and np.allclose(settings["grasp_c_start"], want_c), json.dumps(settings)[:200])


def check_legacy(robot, programs, mug, opts):
    name = robot.name
    learned_mug = programs["learned_mug"]
    informed = grasp_pose_xyz_rpy(learned_mug, mug)
    mug_xyz = mug.middle.translation()
    slack = opts.c_position_slack

    for key in ("numerical_pose", "numerical_mug"):
        p = programs[key]
        b = find(p.prog, "QBoundingBoxConstraint")
        blo, bhi = bounds_of(b)
        if robot.legacy_q_bounds is not None:
            lo, hi = (np.asarray(x, dtype=float) for x in robot.legacy_q_bounds)
            check(f"{name} {key} LEGACY: q bound == the old per-robot bound "
                  f"({lo[0]:.6g}..{hi[0]:.6g} on joint 1)",
                  np.array_equal(blo, lo) and np.array_equal(bhi, hi), f"{blo}..{bhi}")
        else:
            lo, hi = (np.asarray(x, dtype=float)[:p.num_arm_dof] for x in p.ConfigLimits())
            check(f"{name} {key} LEGACY: q bound == ConfigLimits() (this robot never had "
                  f"another)", np.array_equal(blo, lo) and np.array_equal(bhi, hi),
                  f"{blo}..{bhi}")
    for key, p in programs.items():
        check(f"{name} {key} LEGACY: q_nominal is zeros",
              np.array_equal(np.asarray(p.q_nominal, dtype=float), np.zeros(p.num_arm_dof)),
              f"{np.asarray(p.q_nominal).tolist()}")
    c0 = learned_mug.prog.GetInitialGuess(learned_mug.c)
    if robot.legacy_mug_centred_c:
        check(f"{name} learned_mug LEGACY: native c start == [mug xyz, 0, 0, 0]",
              np.allclose(c0, [*mug_xyz, 0, 0, 0]), f"{c0}")
    else:
        check(f"{name} learned_mug LEGACY: native c start was already the informed pose",
              np.allclose(c0, informed), f"{c0}")
    blo, bhi = bounds_of(find(learned_mug.prog, "CBoxConstraint"))
    check(f"{name} learned_mug LEGACY: c box centred on the MUG with half-width {slack}",
          np.allclose(blo[:3], mug_xyz - slack) and np.allclose(bhi[:3], mug_xyz + slack),
          f"box {blo[:3]}..{bhi[:3]} mug {mug_xyz}")
    settings = learned_mug.RobotSettings()
    check(f"{name} learned_mug LEGACY: RobotSettings records the control",
          settings["legacy_robot_settings"] is True
          and settings["grasp_c_box_centre"] == "mug", json.dumps(settings)[:200])


def check_latent_rule(robot, programs_rule, programs_fixed):
    name = robot.name
    for key in ("learned_pose", "learned_mug"):
        p = programs_rule[key]
        width = p.ik_solver.network_width
        want = round(float(np.sqrt(width)) + 1.5, 2)
        b = find(p.prog, "LatentTrustRegion")
        if b is None:
            check(f"{name} {key} latent_rule: LatentTrustRegion present", False, "not found")
            continue
        _, ub = bounds_of(b)
        check(f"{name} {key} latent_rule: radius == round(sqrt({width})+1.5, 2) == "
              f"{robot.expect_rule_radius}",
              np.isclose(ub[0], want ** 2) and want == robot.expect_rule_radius
              and p.LatentTrustRadius() == want, f"ub={ub} want={want ** 2}")
        q = programs_fixed[key]
        _, ub_fixed = bounds_of(find(q.prog, "LatentTrustRegion"))
        check(f"{name} {key} latent (fixed 4.0): radius stays 4.0, rule flag off",
              np.isclose(ub_fixed[0], 16.0) and q.LatentTrustRadius() == 4.0, f"{ub_fixed}")
    for key in ("numerical_pose", "numerical_mug"):
        check(f"{name} {key} latent_rule: no latent region on the joint-space arm",
              find(programs_rule[key].prog, "LatentTrustRegion") is None, "found one")


def check_fingerprint():
    from src import benchmark as bm
    from scripts import collate
    print("\n--- scene fingerprint ---")
    iiwa = os.path.join(REPO, "models/iiwa14/iiwa14_collision_hardened.yaml")
    files = bm.scene_model_files(iiwa)
    check("iiwa scene lists the wsg finray SDF among its model files (the yaw-fix file)",
          any(f.endswith("wsg50_110_finray_fingers_box_collision.sdf") for f in files),
          str(files))
    check("every listed model file exists", all(os.path.exists(f) for f in files), str(files))
    fp = bm.scene_fingerprint(iiwa)
    check("fingerprint is a 40-hex sha1 and deterministic",
          len(fp) == 40 and fp == bm.scene_fingerprint(iiwa), fp)
    legacy = os.path.join(REPO, "models/iiwa14/iiwa14_collision.yaml")
    check("hardened and legacy iiwa scenes fingerprint differently",
          fp != bm.scene_fingerprint(legacy), fp)
    check("the grasp task's in-memory mug changes the fingerprint",
          fp != bm.scene_fingerprint(iiwa, extra_models=[os.path.join(REPO, bm.MUG_MODEL_FILE)]),
          fp)
    ## A referenced model file's CONTENT is what is hashed: copy the scene into a temp
    ## package-free layout is not possible (package:// URIs resolve through package.xml),
    ## so hash the same YAML against a one-byte edit of a referenced file's bytes by hand.
    by_hand = bm.hashlib.sha1()
    with open(iiwa, "rb") as f:
        by_hand.update(f.read())
    for path in files:
        by_hand.update(b"\0" + os.path.basename(path).encode() + b"\0")
        with open(path, "rb") as f:
            by_hand.update(f.read())
    check("fingerprint == sha1(YAML bytes, then each referenced model file's bytes)",
          by_hand.hexdigest() == fp, fp)

    ## collate --pair refuses differing fingerprints the way it refuses differing solvers.
    def summary(fingerprint, grid="deadbeef0000-mug"):
        return dict(summary={"learned": {}},
                    metadata=dict(grid_hash=grid, scene="s.yaml", scene_fingerprint=fingerprint,
                                  target_placement="shelf", shelf_inset=0.1, start="paired",
                                  solver="ipopt", checkpoint=None, task="mug"),
                    records={"learned": [dict(target=0, guess=0, feasible=True),
                                         dict(target=0, guess=1, feasible=False)]})
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, fingerprint in enumerate(("a" * 40, "b" * 40, "a" * 40)):
            d = os.path.join(tmp, f"run{i}")
            os.makedirs(d)
            paths.append(os.path.join(d, "summary.json"))
            with open(paths[-1], "w") as f:
                json.dump(summary(fingerprint), f)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            collate.pair("learned", paths)
        lines = out.getvalue().splitlines()
        row1 = [l for l in lines if l.startswith("run1")]
        row2 = [l for l in lines if l.startswith("run2")]
        check("collate --pair flags a run whose scene_fingerprint differs as NOT comparable",
              row1 and "scene_fingerprint" in row1[0] and "not comparable" in row1[0],
              "\n".join(lines))
        check("collate --pair accepts a run with the same grid AND fingerprint",
              row2 and "not comparable" not in row2[0], "\n".join(lines))


def main():
    base = dict(collision_avoidance=True, joint_limits=True, use_float64=True,
                share_flow_evaluations=True)
    unified = ProgramOptions(**base, latent_trust_region=4.0)
    legacy = ProgramOptions(**base, latent_trust_region=4.0, legacy_robot_settings=True)
    rule = ProgramOptions(**base, latent_trust_region_rule=True)
    check("ProgramOptions defaults to the UNIFIED settings",
          ProgramOptions().legacy_robot_settings is False
          and ProgramOptions().latent_trust_region_rule is False, "")

    for robot in robots():
        print(f"\n--- {robot.name}: unified defaults ---")
        programs, mug = robot.build(unified)
        check_unified(robot, programs, mug, unified)
        print(f"\n--- {robot.name}: legacy_robot_settings=True (the control) ---")
        programs_legacy, mug_legacy = robot.build(legacy)
        check_legacy(robot, programs_legacy, mug_legacy, legacy)
        print(f"\n--- {robot.name}: latent_trust_region_rule ---")
        programs_rule, _ = robot.build(rule)
        check_latent_rule(robot, programs_rule, programs)

    check_fingerprint()

    print(f"\n{CHECKS[0]} checks, {len(FAILURES)} failures")
    for f in FAILURES:
        print(f"  - {f}")
    if FAILURES:
        return 1
    print("ROBOT SETTINGS OK -- unified by default, the legacy control reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
