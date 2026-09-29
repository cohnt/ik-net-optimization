"""Paired-grid benchmark on the soft continuum arm, learned against joint space.

The same harness as `scripts/iiwa/iiwa_benchmark.py`; there is no analytic arm, for the
same reason there is none on the iiwa and a stronger one -- a redundant continuum arm has
no closed-form IK to write.

ONE THING IS GENUINELY DIFFERENT, AND IT IS THE SAMPLER. The rigid arms draw targets and
guesses as `rng.uniform(plant lower, plant upper)`, which works there because the plant's
positions ARE the configuration. Here the plant carries quaternion floating bodies whose
limits are `+-inf`, so that draw is `nan`. Everything is drawn over the CONFIGURATION box
and mapped through `ConfigToPlantQ` wherever a plant vector is wanted. `q_init` is a
configuration throughout, which is what `SetStartFromQ` has always meant.

Usage:
    python soft_arm_benchmark.py --task mug --rung soft12 --checkpoint <path.pkl> \
        --targets 15 --guesses 2 --wall-time 20
"""
import argparse
import hashlib
import math
import os
import sys
from ast import literal_eval
from dataclasses import fields, replace

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import RepoDir, BuildEnv, GenerateDiagramWithMug, HiddenPrints
from src import benchmark as bm
from src.shelf_regions import DEFAULT_SHELF_DEPTH_INSET, ShelfCompartmentRegions
from src.target_screening import (MAX_CONSECUTIVE_REJECTIONS, SCENES, ContainmentPose,
                                  FloatingMugScreen, FormatTargetStats,
                                  SampleShelfTargets, SceneFile)
from src.generic_program import ProgramOptions, orientation_error_rpy
from src.soft_arm.params import GetSpec, RUNGS
from src.soft_arm_program import (SoftArmIKProgram, SoftArmIKProgramNumerical,
                                  SoftArmMugProgram, SoftArmMugProgramNumerical)
from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix
from pydrake.geometry import Meshcat
from tqdm import tqdm


def Configs(spec):
    """The ladder, with the trust region sized to THIS rung's latent.

    Same convention as the rigid arms -- `sqrt(dim_latent) + ~1.5`, which is the Panda's
    4.0 at 7 and the iiwa's 4.3 at 8 -- so 4.96 at 12. Computed rather than tabulated,
    because the DOF ladder varies the latent width and a stale constant would quietly make
    one rung's region tighter than another's.
    """
    radius = round(math.sqrt(spec.ndof) + 1.5, 2)
    return {
        "baseline": dict(calibrate_flow_frame=False, share_flow_evaluations=False),
        "frame":    dict(share_flow_evaluations=False),
        "eval":     dict(share_flow_evaluations=True),
        "latent":   dict(share_flow_evaluations=True, latent_trust_region=radius),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mug", "pose"], default="mug")
    p.add_argument("--rung", choices=sorted(RUNGS), default="soft12",
                   help="which arm. The rungs hold total backbone length and the strain "
                        "limits fixed, so they share a workspace envelope and differ only "
                        "in redundancy; targets are NOT comparable across rungs unless a "
                        "stage deliberately shares them.")
    p.add_argument("--targets", type=int, default=15)
    p.add_argument("--guesses", type=int, default=2)
    p.add_argument("--wall-time", type=float, default=20.0)
    p.add_argument("--solver", choices=["ipopt", "snopt", "nlopt"], default="ipopt",
                   help="three METHOD CLASSES: interior point, SQP, augmented Lagrangian. "
                        "Each at its own defaults; see the iiwa script.")
    p.add_argument("--start", choices=["paired", "native"], default="paired")
    p.add_argument("--arms", default="learned,numerical")
    p.add_argument("--config", default="latent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-tol", type=float, default=1e-3,
                   help="task-space gate, deliberately an order of magnitude looser than "
                        "any bound the solver optimises against. See the iiwa script.")
    p.add_argument("--checkpoint", default=None,
                   help="path to this rung's IKFlow .pkl. REQUIRED: unlike the Panda there "
                        "is no published chart for this robot to fall back on, and unlike "
                        "the iiwa there is no shipped one. The architecture comes from the "
                        "checkpoint's own .arch.json sidecar, so there is no --nb-nodes: a "
                        "sidecar is written at export and a contradicting one raises.")
    p.add_argument("--fk", choices=("analytic", "learned"), default="analytic",
                   help="the FORWARD MODEL. `analytic` is the exact constant-strain "
                        "exponential. `learned` swaps in a surrogate over the same backbone "
                        "frames, which replaces the forward model for the IK constraint and "
                        "the collision geometry at once, because both read the same map. It "
                        "is the general mechanism -- SoRoMoX's variable-strain models "
                        "integrate numerically and a real actuation-space arm has no closed "
                        "form -- and it is a CONTROL rather than an advantage, because the "
                        "joint-space arm uses the same surrogate. Success is always verified "
                        "against the EXACT kinematics, so an arm is never graded by its own "
                        "model of the robot.")
    p.add_argument("--tag", default=None)
    p.add_argument("--cells", default=None, metavar="TI:GI[,TI:GI...]")
    p.add_argument("--shard", default=None, metavar="K/N")
    p.add_argument("--cell-timeout", type=float, default=None)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile BOTH network-free Jacobians -- the flow's and the "
                        "kinematic map's. One switch, because both change how many "
                        "iterations fit in a fixed cap, so compared runs must set it the "
                        "same way. The map's compiled forward-mode Jacobian is worth 68x.")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="NAME=VALUE")
    p.add_argument("--scene", choices=("hardened", "nobin", "legacy"), default="hardened")
    p.add_argument("--target-placement", choices=("shelf", "free", "auto"), default="auto")
    p.add_argument("--placement-point", choices=("wrist", "fingertips"), default="fingertips")
    p.add_argument("--shelf-inset", type=float, default=DEFAULT_SHELF_DEPTH_INSET)
    p.add_argument("--max-target-rejections", type=int, default=MAX_CONSECUTIVE_REJECTIONS)
    return p.parse_args()


def apply_overrides(options, overrides):
    names = {f.name for f in fields(ProgramOptions)}
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if name not in names:
            raise SystemExit(f"--set: no ProgramOptions field {name!r}")
        try:
            parsed[name] = literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[name] = value
    return replace(options, **parsed), parsed


def main():
    args = parse_args()
    spec = GetSpec(args.rung)
    if args.checkpoint is None:
        raise SystemExit(
            f"--checkpoint is required for {args.rung}: no chart ships with this robot. "
            f"Train one with cluster/submit_train.sh and point at the exported .pkl.")

    shard = bm.parse_shard(args.shard) if args.shard else None
    if shard is not None and args.cells:
        raise SystemExit("--shard and --cells are mutually exclusive")

    ckpt_tok = [os.path.basename(args.checkpoint).replace(".pkl", "")]
    tag = args.tag or "_".join(
        [args.rung, args.task, args.config, args.solver, args.start]
        + ([] if args.fk == "analytic" else [f"fk{args.fk}"]) + ckpt_tok
        + [f"{k}{v}" for k, v in (i.split("=", 1) for i in args.overrides)]
        + (["compiled"] if args.compile else []))
    if shard is not None:
        tag = f"{tag}_shard{shard[0]}of{shard[1]}"
    log_dir = os.path.join(RepoDir(), f"results/{args.rung}/benchmark", tag)
    out_path = os.path.join(log_dir, "summary.json")

    base_options = ProgramOptions(
        visualize=False, joint_centering_cost=1e-4, max_wall_time=args.wall_time,
        which_solver=args.solver, acceptable_tol=1e-3,
        acceptable_constr_viol_tol=1e-4, ik_constraint_tol=(1e-4, 0.01),
        mug_height=0.04)
    base_options = replace(base_options, **Configs(spec)[args.config],
                           compile_flow_jacobian=args.compile)
    base_options, overrides = apply_overrides(base_options, args.overrides)
    _, ori_tol = base_options.ik_constraint_tol
    slack = base_options.acceptable_constr_viol_tol

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    meshcat = Meshcat() if base_options.visualize else None

    scene_spec = SCENES[(args.rung, args.task)]
    yaml_file = SceneFile(args.rung, args.task, args.scene)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
        sampler_cls = SoftArmMugProgram if args.task == "mug" else SoftArmIKProgram
        sampler = sampler_cls(diagram, options=base_options, rung=args.rung,
                              checkpoint=args.checkpoint, fk=args.fk)
        sampler.create_prog()
    ik_solver = sampler.ik_solver

    ## THE SAMPLER, in configuration space. `draw` returns a CONFIGURATION; everything that
    ## wants a plant vector maps it. On the rigid arms these two are the same object and
    ## the wrapping is invisible; here it is the difference between a draw and `nan`.
    def to_plant(cfg):
        ## EXACT, always. The grid is a property of the robot, not of whichever forward
        ## model the program carries -- drawing targets through a surrogate would make the
        ## analytic and learned columns un-pairable, which is the one thing this axis needs.
        return sampler.ExactConfigToPlantQ(cfg)

    def sample_collision_free():
        while True:
            cfg = sampler.SampleConfiguration(rng)
            q = to_plant(cfg)
            sampler.plant.SetPositions(sampler.plant_context, q)
            if sampler.collision_free_constraint_eval.Eval(q) < 1:
                return cfg

    placement = args.target_placement
    if placement == "auto":
        placement = "shelf"
    regions = (ShelfCompartmentRegions(args.shelf_inset) if placement == "shelf" else None)
    pose_of_plant_q, placement_point = ContainmentPose(
        sampler.plant, sampler.plant_context, scene_spec, args.placement_point)
    mug_screen = None
    if regions is not None and args.task == "mug":
        with HiddenPrints():
            mug_screen = FloatingMugScreen(yaml_file, scene_spec.robot_instances)

    target_cfgs, target_stats = SampleShelfTargets(
        args.targets,
        draw=lambda: sampler.SampleConfiguration(rng),
        collision_free=lambda cfg: sampler.collision_free_constraint_eval.Eval(
            to_plant(cfg)) < 1,
        target_pose=lambda cfg: pose_of_plant_q(to_plant(cfg)),
        regions=regions,
        screen=(lambda X: mug_screen.Penetrates([X])) if mug_screen else None,
        max_consecutive_rejections=args.max_target_rejections,
        label=f"{args.rung}/{args.task}/{placement}/{args.placement_point}",
        progress=tqdm(total=args.targets, desc="targets"))
    print(FormatTargetStats(f"{args.rung}/{args.task}", target_stats))

    guesses = [[sample_collision_free() for _ in range(args.guesses)]
               for _ in range(args.targets)]
    ## The task is a SUFFIX, following the iiwa: the two tasks draw from the same seed over
    ## the same box and would otherwise hash identically, so `collate.py --pair` would
    ## happily compare a grasp run against a pose one.
    grid_hash = hashlib.sha1(np.asarray(
        target_cfgs + [g for row in guesses for g in row]).tobytes()).hexdigest()[:12]
    grid_hash = f"{grid_hash}-{args.task}"

    if shard is not None:
        cells = bm.shard_cells(*shard, args.targets, args.guesses)
    elif args.cells:
        cells = [tuple(map(int, c.split(":"))) for c in args.cells.split(",")]
    else:
        cells = None

    compile_seconds = None
    if args.compile:
        compile_seconds = sampler.WarmUpJacobian()
        print(f"compiled the flow Jacobian in {compile_seconds:.1f} s")

    if args.task == "mug":
        mug_meshcat = Meshcat() if base_options.visualize else None
        wanted = {ti for ti, _ in cells} if cells is not None else set(range(args.targets))
        targets = [None] * len(target_cfgs)
        for ti, cfg in enumerate(tqdm(target_cfgs, desc="mugs")):
            if ti not in wanted:
                continue
            with HiddenPrints():
                targets[ti] = GenerateDiagramWithMug(to_plant(cfg), sampler, yaml_file,
                                                     mug_meshcat)

        def task_gate(program, q):
            program.plant.SetPositions(program.plant_context, q)
            grasp = program.plant.GetFrameByName("between_fingers")
            p_W = grasp.CalcPoseInWorld(program.plant_context).translation()
            p_M = program.target_mug.middle.inverse() @ p_W
            axis_error = float(np.linalg.norm(p_M[:2]))
            height = float(abs(p_M[2]))
            ok = (axis_error <= args.task_tol
                  and height <= program.options.mug_height + args.task_tol)
            return ok, dict(axis_error=axis_error, height=height)
    else:
        targets = []
        for cfg in target_cfgs:
            sampler.plant.SetPositions(sampler.plant_context, to_plant(cfg))
            pose = sampler.frame.CalcPoseInWorld(sampler.plant_context)
            targets.append(np.array([*pose.translation(),
                                     *pose.rotation().ToQuaternion().wxyz()]))

        def task_gate(program, q):
            translation, wxyz = program.fk(q)
            target = program.target_pose
            axis_max = float(np.max(np.abs(np.asarray(translation) - target[:3])))
            target_rpy = RollPitchYaw(RotationMatrix(Quaternion(target[3:]))).vector()
            rpy_max = float(np.max(np.abs(
                np.asarray(orientation_error_rpy(wxyz, target_rpy), dtype=float))))
            ok = axis_max <= args.task_tol and rpy_max <= ori_tol
            return ok, dict(pos_error=axis_max, rpy_error=rpy_max)

    numerical_options = replace(base_options, joint_centering_cost=1e0)
    mug = args.task == "mug"

    def build(cls, options, target, q_init, cell):
        if mug:
            diagram_with_mug, target_mug = target
            with HiddenPrints():
                program = cls(diagram_with_mug, options=options, rung=args.rung,
                              model=ik_solver, fk=args.fk,
                              surrogate=sampler.fk_surrogate)
                program.create_prog(target_mug=target_mug)
        else:
            with HiddenPrints():
                program = cls(diagram, options=options, rung=args.rung, model=ik_solver,
                              fk=args.fk, surrogate=sampler.fk_surrogate)
                program.create_prog(target)
        with HiddenPrints():
            if args.start == "paired":
                program.clip_distance = program.SetStartFromQ(q_init)
            else:
                program.clip_distance = program.SetNativeStart(
                    q_init, np.random.default_rng([args.seed, *cell]))
        return program

    learned_cls = SoftArmMugProgram if mug else SoftArmIKProgram
    numerical_cls = SoftArmMugProgramNumerical if mug else SoftArmIKProgramNumerical
    all_arms = {
        "learned": bm.Arm("learned",
                          lambda t, g, c: build(learned_cls, base_options, t, g, c),
                          base_options.joint_centering_cost),
        "numerical": bm.Arm("numerical",
                            lambda t, g, c: build(numerical_cls, numerical_options, t, g, c),
                            numerical_options.joint_centering_cost),
    }
    arms = [all_arms[name] for name in args.arms.split(",")]

    n_cells = len(cells) if cells is not None else args.targets * args.guesses
    bar = tqdm(total=len(arms) * n_cells, desc=tag)
    records = bm.run_grid(
        arms, targets, guesses, task_gate, log_dir, out_path, tol=slack,
        relaxed_tol=args.task_tol,
        cell_timeout=args.cell_timeout or (5 * args.wall_time + 300),
        cells=cells,
        unrepresentable_tol=(1e-3 if args.start == "paired" else None),
        progress=lambda *a: bar.update(1),
        metadata=dict(robot=args.rung, rung=args.rung, ndof=spec.ndof, task=args.task,
                      fk=args.fk,
                      fk_surrogate_metrics=getattr(sampler, "fk_surrogate_metrics", None),
                      flow_frame_spread=getattr(sampler, "flow_frame_spread", None),
                      flow_frame_tol=getattr(sampler, "flow_frame_tol", 1e-6),
                      solver=args.solver, config=args.config,
                      wall_time=args.wall_time, seed=args.seed,
                      grid_hash=grid_hash, compiled=args.compile,
                      scene=os.path.basename(yaml_file), scene_mode=args.scene,
                      target_placement=placement,
                      shelf_inset=(args.shelf_inset if placement == "shelf" else None),
                      target_screen=(mug_screen is not None),
                      placement_point=placement_point,
                      placement_point_mode=args.placement_point,
                      target_candidates_drawn=target_stats["drawn"],
                      target_accept_rate=target_stats["accept_rate"],
                      compile_seconds=compile_seconds,
                      overrides=overrides, start=args.start,
                      n_targets=args.targets, n_guesses=args.guesses,
                      shard=args.shard, checkpoint=args.checkpoint,
                      **bm.provenance()))
    bar.close()
    print()
    bm.print_table(bm.summarise(records, arms, args.targets, args.guesses),
                   [a.name for a in arms])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
