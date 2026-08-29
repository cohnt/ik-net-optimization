"""Paired-grid benchmark on the iiwa14, learned against joint space.

The same harness as `scripts/panda/panda_benchmark.py`; only the robot, the scene and the
program classes differ. There is no analytic arm here yet: `src/iiwa_analytic_ik.py`
exposes a different signature from the Panda one (`IK(pose, GC, psi)` with the gripper
offset baked into `gripper_ik` rather than passed as a `pose_offset`), so wiring it up is
a frame-conventions job in its own right and is deliberately left out rather than done
carelessly -- a mis-specified offset would silently measure the wrong frame, which is
exactly the failure this overhaul just found in the Panda grasp scene.

Usage:
    python iiwa_benchmark.py --task mug  --targets 15 --guesses 2 --wall-time 20
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
from src.iiwa_program import (Iiwa14IKProgram, Iiwa14IKProgramNumerical,
                              IiwaMugProgram, IiwaMugProgramNumerical,
                              IiwaMugProgramTaskParam)
from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.geometry import Meshcat
from tqdm import tqdm

CONFIGS = {
    "baseline": dict(calibrate_flow_frame=False, share_flow_evaluations=False),
    "frame":    dict(share_flow_evaluations=False),
    "eval":     dict(share_flow_evaluations=True),
    "task":     dict(share_flow_evaluations=True, c_parameterization="task"),
    # The Panda ladder's winner. The iiwa latent is 8-dimensional, so the trust-region
    # radius is sqrt(8) + ~1.5 rather than the Panda's sqrt(7) + ~1.4.
    "latent":   dict(share_flow_evaluations=True, c_parameterization="task",
                     latent_trust_region=4.3),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mug", "pose"], default="mug")
    p.add_argument("--targets", type=int, default=15)
    p.add_argument("--guesses", type=int, default=2)
    p.add_argument("--wall-time", type=float, default=20.0)
    p.add_argument("--solver", choices=["ipopt", "snopt"], default="ipopt")
    p.add_argument("--arms", default="learned,numerical")
    p.add_argument("--config", default="axis")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-tol", type=float, default=1e-3)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.tag or f"iiwa_{args.task}_{args.config}"
    log_dir = os.path.join(RepoDir(), "results/iiwa/benchmark", tag)
    out_path = os.path.join(log_dir, "summary.json")

    base_options = ProgramOptions(
        visualize=False, joint_centering_cost=1e-4, max_wall_time=args.wall_time,
        which_solver=args.solver, acceptable_tol=1e-3,
        acceptable_constr_viol_tol=1e-4, ik_constraint_tol=(1e-4, 0.01),
        mug_height=0.04)
    base_options = replace(base_options, **CONFIGS[args.config])
    slack = base_options.acceptable_constr_viol_tol

    np.random.seed(args.seed)
    meshcat = Meshcat()
    yaml_file = os.path.join(RepoDir(), "models/iiwa14/iiwa14_collision.yaml")
    with HiddenPrints():
        diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
        sampler_cls = IiwaMugProgram if args.task == "mug" else Iiwa14IKProgram
        sampler = sampler_cls(diagram, options=base_options)
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

    target_qs = [sample_collision_free() for _ in tqdm(range(args.targets), desc="targets")]
    guesses = [sample_collision_free() for _ in range(args.guesses)]

    if args.task == "mug":
        mug_meshcat = Meshcat()
        targets = []
        for q in tqdm(target_qs, desc="mugs"):
            with HiddenPrints():
                targets.append(GenerateDiagramWithMug(q, sampler, yaml_file, mug_meshcat))

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
        for q in target_qs:
            sampler.plant.SetPositions(sampler.plant_context, q)
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
            ok = axis_max <= args.task_tol and rpy_max <= 10 * args.task_tol
            return ok, dict(pos_error=axis_max, rpy_error=rpy_max)

    numerical_options = replace(base_options, joint_centering_cost=1e0)
    mug = args.task == "mug"

    def build(cls, options, target, q_init):
        if mug:
            diagram_with_mug, target_mug = target
            with HiddenPrints():
                program = cls(diagram_with_mug, options=options, model=ik_solver)
                program.create_prog(target_mug=target_mug)
        else:
            with HiddenPrints():
                program = cls(diagram, options=options, model=ik_solver)
                program.create_prog(target)
        with HiddenPrints():
            program.clip_distance = program.SetStartFromQ(q_init)
        return program

    learned_cls = Iiwa14IKProgram
    if mug:
        learned_cls = (IiwaMugProgramTaskParam
                       if base_options.c_parameterization == "task" else IiwaMugProgram)
    all_arms = {
        "learned": bm.Arm("learned",
                          lambda t, g: build(learned_cls, base_options, t, g),
                          base_options.joint_centering_cost),
        "numerical": bm.Arm("numerical",
                            lambda t, g: build(
                                IiwaMugProgramNumerical if mug else Iiwa14IKProgramNumerical,
                                numerical_options, t, g),
                            numerical_options.joint_centering_cost),
    }
    arms = [all_arms[name] for name in args.arms.split(",")]

    bar = tqdm(total=len(arms) * args.targets * args.guesses, desc=tag)
    records = bm.run_grid(arms, targets, guesses, task_gate, log_dir, out_path, tol=slack,
                          progress=lambda *a: bar.update(1),
                          metadata=dict(robot="iiwa14", task=args.task,
                                        solver=args.solver, config=args.config,
                                        wall_time=args.wall_time,
                                        seed=args.seed))
    bar.close()
    print()
    bm.print_table(bm.summarise(records, arms, args.targets, args.guesses),
                   [a.name for a in arms])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
