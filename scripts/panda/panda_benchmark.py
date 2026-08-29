"""Paired-grid benchmark of the three IK formulations.

    learned     c (conditioning pose), z (latent), q_c (joint-space correction);
                q = IKFlow(c, z) + q_c
    numerical   the joint angles q
    analytic    the end-effector pose and redundancy parameter, through closed-form IK

Every formulation goes through the same `IKFlowProgram` machinery, is given the same
targets, the same wall-clock cap, the same tolerances and -- under the default
`--start paired` -- the same starting configuration. Success is verified from the
returned point rather than taken from the solver's status; see `src/benchmark.py`.

Usage:
    python panda_benchmark.py --task mug   --targets 20 --guesses 3 --wall-time 20
    python panda_benchmark.py --task pose  --targets 20 --guesses 3 --config seeded
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import (RepoDir, BuildEnv, GenerateDiagramWithMug, HiddenPrints,
                       CalculateError)
from src import benchmark as bm
from src.generic_program import ProgramOptions, orientation_error_rpy
from src.panda_program import (PandaIKProgram, PandaIKProgramNumerical,
                               PandaIKProgramAnalytic, PandaMugProgram,
                               PandaMugProgramNumerical, PandaMugProgramAnalytic,
                               PandaMugProgramTaskParam)
from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.geometry import Meshcat
from tqdm import tqdm


# The analytic map is written against a different end-effector convention than the
# scene's; this is the offset that reconciles them (unchanged from the older scripts).
MUG_ANALYTIC_OFFSET = RigidTransform(
    RotationMatrix([[0, 0., 1.], [0, -1, 0.], [1., 0, 0.]]),
    np.array([-0.0236, -1.87933e-05, 0.0]))
POSE_ANALYTIC_OFFSET = RigidTransform(RotationMatrix.Identity(),
                                      np.array([0.0, 0.0, 0.1034]))

# Ladder configurations. Each is an override of the base options; "baseline" is the
# code exactly as it stood before this overhaul, so every later row is a delta from a
# measured starting point rather than from an assumption.
CONFIGS = {
    # The ladder is cumulative: each row adds one change to the row above it, so a
    # success-rate move can be attributed. "baseline" is the code as it stood before the
    # overhaul, including the wrong conditioning frame, so every later row is a delta from
    # something measured rather than from something assumed.
    "baseline":     dict(calibrate_flow_frame=False, share_flow_evaluations=False),
    "frame":        dict(share_flow_evaluations=False),
    "eval":         dict(share_flow_evaluations=True),
    "task":         dict(share_flow_evaluations=True, c_parameterization="task"),
    "latent":       dict(share_flow_evaluations=True, c_parameterization="task",
                         latent_trust_region=4.0),
    "axis":         dict(share_flow_evaluations=True, c_parameterization="task",
                         latent_trust_region=4.0, mug_axis_tol=1e-4),
    # Same as "axis" but with the free conditioning pose, to separate the task
    # parameterisation from everything stacked on top of it.
    "axis-free-c":  dict(share_flow_evaluations=True, latent_trust_region=4.0,
                         mug_axis_tol=1e-4),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mug", "pose"], default="mug")
    p.add_argument("--targets", type=int, default=20)
    p.add_argument("--guesses", type=int, default=3)
    p.add_argument("--wall-time", type=float, default=20.0)
    p.add_argument("--solver", choices=["ipopt", "snopt"], default="ipopt")
    p.add_argument("--start", choices=["paired", "native"], default="paired")
    p.add_argument("--arms", default="learned,numerical,analytic")
    p.add_argument("--config", default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-tol", type=float, default=1e-3,
                   help="task-space gate: metres off the mug axis / per-axis position "
                        "error. Deliberately looser than the solver's constraint "
                        "tolerance -- the gate should measure whether the arm actually "
                        "reached the target, not which formulation's rounding is "
                        "smaller. The raw errors are stored per record, so a stricter "
                        "gate can be recomputed from the summary without re-running.")
    p.add_argument("--tag", default=None, help="subdirectory under results/")
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.tag or f"{args.task}_{args.config}_{args.start}_{args.solver}"
    log_dir = os.path.join(RepoDir(), "results/panda/benchmark", tag)
    out_path = os.path.join(log_dir, "summary.json")

    base_options = ProgramOptions(
        visualize=False,
        joint_centering_cost=1e-4,
        max_wall_time=args.wall_time,
        which_solver=args.solver,
        acceptable_tol=1e-3,
        acceptable_constr_viol_tol=1e-4,
        ik_constraint_tol=(1e-4, 0.01),
        mug_height=0.04,
        # Under the paired protocol the start comes from q_init, so the multi-start
        # seeding pass is off by default; `--config seeded` turns it back on as its own
        # row rather than leaving it as an advantage only one arm enjoys.
        num_seed_samples=0,
    )
    base_options = replace(base_options, **CONFIGS[args.config])
    if args.start == "native":
        base_options = replace(base_options, num_seed_samples=256)

    pos_tol, _ = base_options.ik_constraint_tol
    slack = base_options.acceptable_constr_viol_tol

    np.random.seed(args.seed)
    meshcat = Meshcat()
    scene = "panda_finray_collision.yaml" if args.task == "mug" else "panda_collision.yaml"
    yaml_file = os.path.join(RepoDir(), "models/panda", scene)

    with HiddenPrints():
        diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
        # One program, used only to sample targets and to hold the loaded network so
        # every later program shares it (the flow is ~1.8 s to load).
        sampler_cls = PandaMugProgram if args.task == "mug" else PandaIKProgram
        sampler = sampler_cls(diagram, options=replace(base_options, num_seed_samples=0))
        sampler.create_prog()
    ik_solver = sampler.ik_solver
    lower = sampler.plant.GetPositionLowerLimits()
    upper = sampler.plant.GetPositionUpperLimits()

    def sample_collision_free():
        while True:
            q = np.random.uniform(lower, upper)
            sampler.plant.SetPositions(sampler.plant_context, q)
            if sampler.collision_free_constraint_eval.Eval(q) < 1:
                return q

    ## ------------------------------- targets ------------------------------- ##
    # Sampling a configuration and taking its gripper pose guarantees every target is
    # reachable and, for the mug, that at least one valid grasp exists.
    target_qs = [sample_collision_free() for _ in tqdm(range(args.targets), desc="targets")]
    guesses = [sample_collision_free() for _ in range(args.guesses)]

    if args.task == "mug":
        mug_meshcat = Meshcat()
        targets = []
        for q in tqdm(target_qs, desc="mugs"):
            with HiddenPrints():
                targets.append(GenerateDiagramWithMug(q, sampler, yaml_file, mug_meshcat))
    else:
        targets = []
        for q in target_qs:
            sampler.plant.SetPositions(sampler.plant_context, q)
            pose = sampler.frame.CalcPoseInWorld(sampler.plant_context)
            targets.append(np.array([*pose.translation(),
                                     *pose.rotation().ToQuaternion().wxyz()]))

    ## ------------------------------ task gates ----------------------------- ##
    if args.task == "mug":
        def task_gate(program, q):
            # Ask for `between_fingers` by name rather than using `program.frame`.
            # PandaMugProgramAnalytic inherits from the *pose* analytic class, which never
            # moves `self.frame` off `panda_hand`, so reading `program.frame` here would
            # silently measure a point 0.1 m (|X_grasp_ee|) away from the grasp.
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
        def task_gate(program, q):
            translation, wxyz = program.fk(q)
            target = program.target_pose
            axis_max = float(np.max(np.abs(np.asarray(translation) - target[:3])))
            target_rpy = RollPitchYaw(RotationMatrix(Quaternion(target[3:]))).vector()
            rpy_max = float(np.max(np.abs(
                np.asarray(orientation_error_rpy(wxyz, target_rpy), dtype=float))))
            angle, distance = CalculateError(
                RigidTransform(Quaternion(wxyz), translation),
                RigidTransform(Quaternion(target[3:]), target[:3]))
            ok = axis_max <= args.task_tol and rpy_max <= 10 * args.task_tol
            return ok, dict(pos_error=axis_max, rpy_error=rpy_max,
                            pos_dist=float(distance), ori_error=float(angle))

    ## -------------------------------- the arms ----------------------------- ##
    numerical_options = replace(base_options, joint_centering_cost=1e0)

    def build(cls, options, target, q_init, **extra):
        if args.task == "mug":
            diagram_with_mug, mug = target
            with HiddenPrints():
                program = cls(diagram_with_mug, options=options, model=ik_solver)
                program.create_prog(target_mug=mug, **extra)
        else:
            with HiddenPrints():
                program = cls(diagram, options=options, model=ik_solver)
                program.create_prog(target, **extra)
        if args.start == "paired":
            with HiddenPrints():
                program.clip_distance = program.SetStartFromQ(q_init)
        return program

    mug = args.task == "mug"
    learned_cls = PandaIKProgram
    if mug:
        learned_cls = (PandaMugProgramTaskParam
                       if base_options.c_parameterization == "task" else PandaMugProgram)
    all_arms = {
        "learned": bm.Arm(
            "learned",
            lambda t, g: build(learned_cls, base_options, t, g),
            base_options.joint_centering_cost),
        "numerical": bm.Arm(
            "numerical",
            lambda t, g: build(PandaMugProgramNumerical if mug else PandaIKProgramNumerical,
                               numerical_options, t, g),
            numerical_options.joint_centering_cost),
        "analytic": bm.Arm(
            "analytic",
            lambda t, g: build(PandaMugProgramAnalytic if mug else PandaIKProgramAnalytic,
                               base_options, t, g,
                               pose_offset=MUG_ANALYTIC_OFFSET if mug else POSE_ANALYTIC_OFFSET),
            base_options.joint_centering_cost),
    }
    arms = [all_arms[name] for name in args.arms.split(",")]

    bar = tqdm(total=len(arms) * args.targets * args.guesses, desc=tag)
    records = bm.run_grid(
        arms, targets, guesses, task_gate, log_dir, out_path,
        tol=slack,
        progress=lambda *a: bar.update(1),
        metadata=dict(task=args.task, solver=args.solver, config=args.config,
                      start=args.start, wall_time=args.wall_time, seed=args.seed,
                      n_targets=args.targets, n_guesses=args.guesses))
    bar.close()

    summary = bm.summarise(records, arms, args.targets, args.guesses)
    print()
    bm.print_table(summary, [a.name for a in arms])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
