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
import hashlib
import os
import sys
from ast import literal_eval
from dataclasses import fields, replace

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
    p.add_argument("--start", choices=["paired", "native"], default="paired",
                   help="paired: every arm starts at the same q_init, in its own variables "
                        "(SetStartFromQ). native: every arm uses its own initialisation -- "
                        "the flow's latent drawn from its prior, the analytic map's "
                        "redundancy parameter and branch drawn from theirs, the joint-space "
                        "arm from a random configuration. Sampled, never searched: no "
                        "candidate is scored against the problem in either mode.")
    p.add_argument("--arms", default="learned,numerical")
    # "axis" named a config that was removed with the mug-axis tolerance, so the default
    # raised KeyError; "latent" is the configuration the Panda ladder settled on.
    p.add_argument("--config", default="latent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-tol", type=float, default=1e-3)
    p.add_argument("--tag", default=None)
    p.add_argument("--cells", default=None, metavar="TI:GI[,TI:GI...]",
                   help="run only these (target, guess) cells of the seeded grid")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the flow Jacobian, once per process, warmed up "
                        "before the grid. Moves the learned arm's success rate inside a "
                        "fixed cap, so runs being compared must set it the same way.")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override any ProgramOptions field, e.g. --set correction_bound=0.4")
    return p.parse_args()


def apply_overrides(options, overrides):
    parsed = {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        name, _, value = item.partition("=")
        name = name.strip()
        if not any(f.name == name for f in fields(ProgramOptions)):
            raise SystemExit(f"--set: no such ProgramOptions field {name!r}")
        try:
            parsed[name] = literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[name] = value
    return replace(options, **parsed), parsed


def main():
    args = parse_args()
    tag = args.tag or "_".join(
        ["iiwa", args.task, args.config, args.start]
        + [f"{k}{v}" for k, v in (i.split("=", 1) for i in args.overrides)]
        + (["compiled"] if args.compile else []))
    log_dir = os.path.join(RepoDir(), "results/iiwa/benchmark", tag)
    out_path = os.path.join(log_dir, "summary.json")

    base_options = ProgramOptions(
        visualize=False, joint_centering_cost=1e-4, max_wall_time=args.wall_time,
        which_solver=args.solver, acceptable_tol=1e-3,
        acceptable_constr_viol_tol=1e-4, ik_constraint_tol=(1e-4, 0.01),
        mug_height=0.04)
    base_options = replace(base_options, **CONFIGS[args.config],
                           compile_flow_jacobian=args.compile)
    base_options, overrides = apply_overrides(base_options, args.overrides)
    slack = base_options.acceptable_constr_viol_tol

    # A local generator for the grid: see the note in the Panda script -- draws made during
    # program construction used to shift which targets a configuration was measured on.
    rng = np.random.default_rng(args.seed)
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
            q = rng.uniform(lower, upper)
            sampler.plant.SetPositions(sampler.plant_context, q)
            if sampler.collision_free_constraint_eval.Eval(q) < 1:
                return q

    target_qs = [sample_collision_free() for _ in tqdm(range(args.targets), desc="targets")]
    guesses = [sample_collision_free() for _ in range(args.guesses)]
    grid_hash = hashlib.sha1(np.asarray(target_qs + guesses).tobytes()).hexdigest()[:12]

    compile_seconds = None
    if args.compile:
        compile_seconds = sampler.WarmUpJacobian()
        print(f"compiled the flow Jacobian in {compile_seconds:.1f} s")

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

    def build(cls, options, target, q_init, cell):
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
            if args.start == "paired":
                program.clip_distance = program.SetStartFromQ(q_init)
            else:
                # A generator per cell, so a native start varies from guess to guess and
                # from target to target while staying reproducible from --seed.
                program.clip_distance = program.SetNativeStart(
                    q_init, np.random.default_rng([args.seed, *cell]))
        return program

    learned_cls = Iiwa14IKProgram
    if mug:
        learned_cls = (IiwaMugProgramTaskParam
                       if base_options.c_parameterization == "task" else IiwaMugProgram)
    all_arms = {
        "learned": bm.Arm("learned",
                          lambda t, g, c: build(learned_cls, base_options, t, g, c),
                          base_options.joint_centering_cost),
        "numerical": bm.Arm("numerical",
                            lambda t, g, c: build(
                                IiwaMugProgramNumerical if mug else Iiwa14IKProgramNumerical,
                                numerical_options, t, g, c),
                            numerical_options.joint_centering_cost),
    }
    arms = [all_arms[name] for name in args.arms.split(",")]

    bar = tqdm(total=len(arms) * args.targets * args.guesses, desc=tag)
    records = bm.run_grid(arms, targets, guesses, task_gate, log_dir, out_path, tol=slack, cell_timeout=5 * args.wall_time + 300,
        cells=([tuple(map(int, c.split(":"))) for c in args.cells.split(",")]
               if args.cells else None),
                          progress=lambda *a: bar.update(1),
                          metadata=dict(robot="iiwa14", task=args.task,
                                        solver=args.solver, config=args.config,
                                        wall_time=args.wall_time, seed=args.seed,
                                        grid_hash=grid_hash, compiled=args.compile,
                                        compile_seconds=compile_seconds,
                                        overrides=overrides, start=args.start,
                                        n_targets=args.targets, n_guesses=args.guesses))
    bar.close()
    print()
    bm.print_table(bm.summarise(records, arms, args.targets, args.guesses),
                   [a.name for a in arms])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
