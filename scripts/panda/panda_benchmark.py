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
from src.panda_program import (PandaIKProgram, PandaIKProgramNumerical,
                               PandaIKProgramAnalytic, PandaMugProgram,
                               PandaMugProgramNumerical, PandaMugProgramAnalytic)
from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.geometry import Meshcat
from tqdm import tqdm


# The analytic map is written against its own end-effector convention: link 7, rotated
# -pi/4 about z and translated by d7 + d8 = 0.2104 m. Reconciling it with a scene frame is
# therefore a measurement, not a constant -- the fitted MUG_ANALYTIC_OFFSET this replaces
# was 0.046 degrees off, because the finray SDF writes 1.57 where the constant assumed
# pi/2, which cost 1.5e-3 to 4.9e-3 rad of round-trip error in the paired start (0.019 mm
# at the gripper, so it never showed up in the task gate).
X_L7_ANALYTIC = RigidTransform(RotationMatrix.MakeZRotation(-np.pi / 4),
                               np.array([0.0, 0.0, 0.2104]))


def AnalyticPoseOffset(plant, context, frame_name):
    """`X_FA`: the analytic map's frame, expressed in the frame its variables denote.

    Both are welded to link 7, so this is constant and one configuration suffices.
    """
    plant.SetPositions(context, np.zeros(plant.num_positions()))
    X_WF = plant.GetFrameByName(frame_name).CalcPoseInWorld(context)
    X_WL7 = plant.GetFrameByName("panda_link7").CalcPoseInWorld(context)
    return X_WF.inverse() @ X_WL7 @ X_L7_ANALYTIC

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
    "latent":       dict(share_flow_evaluations=True,
                         latent_trust_region=4.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mug", "pose"], default="mug")
    p.add_argument("--targets", type=int, default=20)
    p.add_argument("--guesses", type=int, default=3)
    p.add_argument("--wall-time", type=float, default=20.0)
    p.add_argument("--solver", choices=["ipopt", "snopt"], default="ipopt")
    p.add_argument("--start", choices=["paired", "native"], default="paired",
                   help="paired: every arm starts at the same q_init, in its own variables "
                        "(SetStartFromQ). native: every arm uses its own initialisation -- "
                        "the flow's latent drawn from its prior, the analytic map's "
                        "redundancy parameter and branch drawn from theirs, the joint-space "
                        "arm from a random configuration. Sampled, never searched: no "
                        "candidate is scored against the problem in either mode.")
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
    p.add_argument("--guess-filter", choices=["none", "charted"], default="none",
                   help="'charted' rejection-samples the shared guess list into the four "
                        "wide analytic branches (elbow branch A=+1). Applied once, to the "
                        "guesses every arm receives, before any solve -- pairing is "
                        "preserved and nothing is scored against the problem. The grid is "
                        "conditioned on the analytic chart, so its table answers 'given a "
                        "start in a wide bundle, how do the arms compare?' and is reported "
                        "separately with that caveat.")
    p.add_argument("--cells", default=None, metavar="TI:GI[,TI:GI...]",
                   help="run only these (target, guess) cells of the seeded grid -- "
                        "a debugging aid for reproducing a single archived cell")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the flow Jacobian (1.33x on it, ~1.5x on an "
                        "AutoDiffXd VarsToQ). Compiled once per process and warmed up "
                        "before the grid, so no cell pays the ~10 s penalty -- but it "
                        "moves the learned arm's success rate inside a fixed wall-clock "
                        "cap, so every run being compared must set it the same way.")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override any ProgramOptions field, e.g. --set correction_bound=0.4. "
                        "Applied after --config, recorded in the metadata and the tag.")
    return p.parse_args()


def apply_overrides(options, overrides):
    """`--set name=value` -> a replace() on the options, validated against the dataclass."""
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
        [args.task, args.config, args.solver, args.start]
        + [f"{k}{v}" for k, v in (i.split("=", 1) for i in args.overrides)]
        + (["compiled"] if args.compile else []))
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
    )
    base_options = replace(base_options, **CONFIGS[args.config],
                           compile_flow_jacobian=args.compile)
    base_options, overrides = apply_overrides(base_options, args.overrides)

    pos_tol, _ = base_options.ik_constraint_tol
    slack = base_options.acceptable_constr_viol_tol

    # A local generator for the grid, so that the targets and guesses cannot be shifted by
    # anything else drawing from the global stream -- which is exactly what used to happen:
    # CalibrateFlowFrame drew four configurations in __init__ and returned early when it was
    # disabled, so the ladder's baseline rung ran on a different grid from every other rung
    # and no cross-rung comparison was paired. The global seed stays for incidental draws.
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    meshcat = Meshcat()
    scene = "panda_finray_collision.yaml" if args.task == "mug" else "panda_collision.yaml"
    yaml_file = os.path.join(RepoDir(), "models/panda", scene)

    with HiddenPrints():
        diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
        # One program, used only to sample targets and to hold the loaded network so
        # every later program shares it (the flow is ~1.8 s to load).
        sampler_cls = PandaMugProgram if args.task == "mug" else PandaIKProgram
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

    ## ------------------------------- targets ------------------------------- ##
    # Sampling a configuration and taking its gripper pose guarantees every target is
    # reachable and, for the mug, that at least one valid grasp exists.
    target_qs = [sample_collision_free() for _ in tqdm(range(args.targets), desc="targets")]

    def sample_guess():
        # A property of the *grid*, identical for every arm: with --guess-filter charted,
        # keep drawing until the guess lies in the half of the analytic chart the
        # historical 4-branch map covers (elbow branch A = +1; ~90% of configurations).
        while True:
            q = sample_collision_free()
            if args.guess_filter == "none":
                return q
            from src.panda_analytic_ik import Analytic_IK_Panda
            if Analytic_IK_Panda().gc(np.asarray(q, dtype=float)[:7], branches=3)[2] > 0:
                return q

    guesses = [sample_guess() for _ in range(args.guesses)]
    # Identical cells across runs are what makes the ladder and the sweeps paired; record a
    # hash of them so a mismatch is visible in the summary rather than silently compared.
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
            # Ask for `between_fingers` by name rather than using `program.frame`: the
            # gate should measure the same physical point whatever each formulation calls
            # its own frame. (PandaMugProgramAnalytic used to leave `self.frame` on
            # `panda_hand`, 0.1 m away from the grasp; it no longer does, but the gate has
            # no business depending on that.)
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

    def build(cls, options, target, q_init, cell, **extra):
        if args.task == "mug":
            diagram_with_mug, mug = target
            with HiddenPrints():
                program = cls(diagram_with_mug, options=options, model=ik_solver)
                program.create_prog(target_mug=mug, **extra)
        else:
            with HiddenPrints():
                program = cls(diagram, options=options, model=ik_solver)
                program.create_prog(target, **extra)
        with HiddenPrints():
            if args.start == "paired":
                program.clip_distance = program.SetStartFromQ(q_init)
            else:
                # A generator per cell, so a native start varies from guess to guess and
                # from target to target while staying reproducible from --seed.
                program.clip_distance = program.SetNativeStart(
                    q_init, np.random.default_rng([args.seed, *cell]))
        return program

    mug = args.task == "mug"
    analytic_offset = AnalyticPoseOffset(
        sampler.plant, sampler.plant_context,
        "between_fingers" if mug else "panda_hand")
    learned_cls = PandaIKProgram
    if mug:
        learned_cls = PandaMugProgram
    all_arms = {
        "learned": bm.Arm(
            "learned",
            lambda t, g, c: build(learned_cls, base_options, t, g, c),
            base_options.joint_centering_cost),
        "numerical": bm.Arm(
            "numerical",
            lambda t, g, c: build(PandaMugProgramNumerical if mug else PandaIKProgramNumerical,
                               numerical_options, t, g, c),
            numerical_options.joint_centering_cost),
        "analytic": bm.Arm(
            "analytic",
            lambda t, g, c: build(PandaMugProgramAnalytic if mug else PandaIKProgramAnalytic,
                               base_options, t, g, c, pose_offset=analytic_offset),
            base_options.joint_centering_cost),
        # The same formulation on the full 8-branch chart. A separate arm rather than a
        # config so the 4- and 8-branch columns land in one run on identical cells; the
        # only difference is the discrete branch set (and so which q_init the paired start
        # can represent -- the 4-branch chart cannot express ~10% of configurations).
        "analytic8": bm.Arm(
            "analytic8",
            lambda t, g, c: build(PandaMugProgramAnalytic if mug else PandaIKProgramAnalytic,
                               replace(base_options, analytic_branches=8),
                               t, g, c, pose_offset=analytic_offset),
            base_options.joint_centering_cost),
    }
    arms = [all_arms[name] for name in args.arms.split(",")]

    bar = tqdm(total=len(arms) * args.targets * args.guesses, desc=tag)
    records = bm.run_grid(
        arms, targets, guesses, task_gate, log_dir, out_path,
        tol=slack, cell_timeout=5 * args.wall_time + 300,
        cells=([tuple(map(int, c.split(":"))) for c in args.cells.split(",")]
               if args.cells else None),
        # Paired means starting AT q_init: a cell whose q_init the arm's variables cannot
        # represent is an immediate failure (fail_reason "unrepresentable_start"), not a
        # solve from a projection. Native starts are the formulation's own draw, so the
        # rule does not apply there.
        unrepresentable_tol=(1e-3 if args.start == "paired" else None),
        progress=lambda *a: bar.update(1),
        metadata=dict(task=args.task, solver=args.solver, config=args.config,
                      wall_time=args.wall_time, seed=args.seed, grid_hash=grid_hash,
                      compiled=args.compile, compile_seconds=compile_seconds,
                      overrides=overrides, start=args.start, guess_filter=args.guess_filter,
                      n_targets=args.targets, n_guesses=args.guesses))
    bar.close()

    summary = bm.summarise(records, arms, args.targets, args.guesses)
    print()
    bm.print_table(summary, [a.name for a in arms])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
