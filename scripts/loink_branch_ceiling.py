"""The branch reach ceiling: how many probe targets a SINGLE-BRANCH chart could solve at all.

Phase 2 of the LOInK campaign retrains the Panda and iiwa charts on ONE IK branch, the
nominal configuration's (`RecoverBranch(NominalConfiguration())` of the analytic program:
Panda (B, C, A) = (1, 2, +1) on the 8-branch He-Liu chart, iiwa (g2, g4, g6) = (+1, -1, +1)
on Faria's S-R-S chart -- computed here, never hard-coded). Such an arm can only return
configurations on that branch, so a success count like 12/20 must be read against how many
of the 20 targets are reachable ON THE BRANCH at all. This script measures that number.

WHAT "REACHABLE" MEANS. A target is reachable on a branch iff some candidate configuration

    q = closed-form IK(pose, psi, branch)

on the analytic arm's own map (`src/analytic_program.py`, the program the benchmark fields as
`analytic8`, so the frame offset, the clip and the reachability rows are the benchmark's) is
scored `feasible` by `src.benchmark.verify` -- the benchmark's own per-binding re-evaluation of
a returned point, called with the record's tolerances (`tol = acceptable_constr_viol_tol =
1e-4`, `relaxed_tol = task_tol = 1e-3`) and the benchmark's task gate for the task. So it is
exactly what a benchmark cell means by success: every binding of the analytic program within
1e-4 (the pose-target equality or the mug-axis rows, the reachability rows -- which keep the
arccos/arcsin clip inactive, so q IS the closed form -- the joint-limit row, the collision row,
the psi and pose boxes), collision-free in the hardened shelf scene (the welded target mug
included on the grasp task), and the task gate passed. In addition the configuration must
classify back to the branch (`RecoverBranch(q) == branch`), so it lies on the branch the
single-branch dataset is built from.

THE GRID MAKES THIS A LOWER BOUND on the true ceiling. The candidates are a finite grid:
  * psi: `--psi-steps` (default 360) points over the analytic program's own psi bound
    `[-PSI_BOUNDS, PSI_BOUNDS]` (Panda: +-5, psi = q7; iiwa: +-10, the arm angle);
  * pose task: the pose is the target (the program's equality pins it), nothing else is free;
  * grasp task: the program constrains only the POSITION of `between_fingers` in the mug
    frame (x = y = 0, |z| <= mug_height); ALL THREE rotational freedoms are free. Swept: the
    yaw about the mug axis (`--yaw-steps`, default 36, over a full turn, including 0) and the
    height z (`--z-steps`, default 5, over [-h, h], including 0 when odd); the two tilts about
    axes perpendicular to the mug axis are held at zero (the generating configuration's).
A feasible configuration between grid points, or at a tilted grasp, is missed, so "reachable"
here is certain and "unreachable" is not. A cheap vectorised prefilter (the batched torch twin
of the map, `src/svgd/batched_analytic.py`, pinned to the pydrake map at <= 1e-12 by
tests/test_batched_analytic.py; reachability rows, joint limits) and a single collision
evaluation discard candidates before `verify`; `verify` alone decides feasibility. Candidates
are tried nearest-to-the-generating-grasp first (yaw, then z, by magnitude) and the search
stops at the first feasible one per target and branch.

ALSO REPORTED: the same count over ALL eight branches of the analytic chart (a target is
reachable if any branch reaches it; every branch is searched, so each target lists the
branches that reach it), so the ceiling's cost against a multi-branch chart is visible; and a sanity column, the generating configuration q* itself scored by `verify` at
its own exact (pose, psi, branch) -- every target is built from a collision-free q*, so this
should be feasible except where q* sits in the analytic clip band or the gate.

THE TARGETS ARE THE BENCHMARK'S, EXACTLY: the record grid (`--targets 60 --guesses 8 --seed 1
--scene hardened --target-placement shelf --placement-point fingertips --config latent`) is
drawn here with the benchmark scripts' own random-stream plumbing around the shared sampler
`src.target_screening.SampleShelfTargets` (the plumbing is inline in each script's `main()`,
so it is transcribed below line for line), all 60 targets and 480 guesses, so the printed
`grid_hash` can be checked against a record summary's `metadata["grid_hash"]`; the record's
value is embedded and compared. The LOInK probe solves guess 0 of targets 0..19
(LOINK_PROBE_TARGETS = 20 in cluster/gen_manifest.py), i.e. the first `--targets` targets.
The chart loaded to build the programs (the classes require one) affects nothing measured.

Usage (CPU only; analytic, never touches the GPU; the full run is a cluster job,
cluster/ceiling_job.sh):
    python scripts/loink_branch_ceiling.py --robot panda --task pose
    python scripts/loink_branch_ceiling.py --robot iiwa --task mug --targets 20
Output: results/loink_branch_ceiling/<robot>_<task>.json and a printed table.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch  # noqa: E402

from src.utils import (RepoDir, BuildEnv, GenerateDiagramWithMug, HiddenPrints)  # noqa: E402
from src import benchmark as bm  # noqa: E402
from src.shelf_regions import DEFAULT_SHELF_DEPTH_INSET, ShelfCompartmentRegions  # noqa: E402
from src.target_screening import (MAX_CONSECUTIVE_REJECTIONS, SCENES,  # noqa: E402
                                  ContainmentPose, FloatingMugScreen, FormatTargetStats,
                                  SampleShelfTargets, SceneFile)
from src.generic_program import ProgramOptions, orientation_error_rpy  # noqa: E402
from src.analytic_program import CheckClipCannotEscapeGate  # noqa: E402
from src.svgd.batched_analytic import BatchedAnalyticMap  # noqa: E402
from pydrake.all import (Quaternion, RigidTransform, RollPitchYaw,  # noqa: E402
                         RotationMatrix)

#: The record grid (stage REMEASURE / PAPER): fixed, so the hash can be checked.
RECORD_SEED = 1
RECORD_N_TARGETS = 60
RECORD_N_GUESSES = 8
#: `metadata["grid_hash"]` of the record's summaries (read 2026-10-24 from
#: results/panda/benchmark/sc_REMEASURE_panda_n6_ipopt_{posetip,mugshelf}_480_180_paired and
#: results/iiwa/benchmark/sc_PAPER_iiwa_n4_ipopt_{posetip,mugshelf}_480_180_paired).
RECORD_GRID_HASH = {
    ("panda", "pose"): "f872dcf34dd1",
    ("panda", "mug"): "734d0f70e7bb",
    ("iiwa", "pose"): "4361e3d807f1-pose",
    ("iiwa", "mug"): "b8a7b81e4972-mug",
}
#: The record's charts, used when present (they do not affect anything measured here).
RECORD_CHECKPOINT = {"panda": "models/panda/panda__n6__step620000.pkl",
                     "iiwa": "models/iiwa14/iiwa14__n4__step620000.pkl"}
#: The benchmark's `--config latent` and the record's `--set` overrides that are not GPU-only.
CONFIG_LATENT = dict(share_flow_evaluations=True, latent_trust_region=4.0)
RECORD_OVERRIDES = dict(correction_cost_weight=10.0, max_iter=1000000)
TASK_TOL = 1e-3  # the benchmark's --task-tol default


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--robot", choices=("panda", "iiwa"), required=True)
    p.add_argument("--task", choices=("pose", "mug"), required=True)
    p.add_argument("--targets", type=int, default=20,
                   help="the first N targets of the record grid (the LOInK probe: 20)")
    p.add_argument("--psi-steps", type=int, default=360)
    p.add_argument("--yaw-steps", type=int, default=36, help="grasp task only")
    p.add_argument("--z-steps", type=int, default=5, help="grasp task only")
    p.add_argument("--branches", choices=("both", "nominal"), default="both",
                   help="`nominal` skips the all-eight-branches comparison")
    p.add_argument("--checkpoint", default=None,
                   help="chart to construct the programs with (irrelevant to the result); "
                        "default the record's if present under the repo, else the program's")
    p.add_argument("--json_out", "--json-out", dest="json_out", default=None)
    return p.parse_args()


## ------------------------------ the record grid ------------------------------- ##

def program_classes(robot):
    if robot == "panda":
        from src.panda_program import (PandaIKProgram, PandaMugProgram,
                                       PandaIKProgramAnalytic, PandaMugProgramAnalytic)
        return dict(pose=PandaIKProgram, mug=PandaMugProgram,
                    pose_analytic=PandaIKProgramAnalytic, mug_analytic=PandaMugProgramAnalytic)
    from src.iiwa_program import (Iiwa14IKProgram, IiwaMugProgram, Iiwa14IKProgramAnalytic,
                                  IiwaMugProgramAnalytic)
    return dict(pose=Iiwa14IKProgram, mug=IiwaMugProgram,
                pose_analytic=Iiwa14IKProgramAnalytic, mug_analytic=IiwaMugProgramAnalytic)


def base_options():
    """The benchmark scripts' `base_options` (identical on both robots) + --config latent +
    the record's non-GPU overrides. Only the collision options touch the grid's draws."""
    options = ProgramOptions(
        visualize=False, joint_centering_cost=1e-4, max_wall_time=180.0,
        which_solver="ipopt", acceptable_tol=1e-3, acceptable_constr_viol_tol=1e-4,
        ik_constraint_tol=(1e-4, 0.01), mug_height=0.04)
    return replace(options, **CONFIG_LATENT, **RECORD_OVERRIDES)


def draw_record_grid(robot, task, options, checkpoint):
    """The record grid, drawn exactly as scripts/{panda,iiwa}/*_benchmark.py `main()` draws it
    (transcribed: the stream plumbing is inline there; the sampler is the shared function).
    Returns the scene, the sampler program, the 60 target configurations, the targets in the
    form the benchmark solves (pose 7-vectors / (diagram, Mug) built lazily) and the hash."""
    classes = program_classes(robot)
    yaml_file = SceneFile(robot, task, "hardened")
    scene_fp = bm.scene_fingerprint(
        yaml_file, extra_models=([os.path.join(RepoDir(), bm.MUG_MODEL_FILE)]
                                 if task == "mug" else ()))
    rng = np.random.default_rng(RECORD_SEED)
    np.random.seed(RECORD_SEED)
    spec = SCENES[(robot, task)]
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=yaml_file)
        kw = dict(options=options, checkpoint=checkpoint)
        if robot == "iiwa":
            kw["nb_nodes"] = 12  # the benchmark's default; a sidecar overrides it
        sampler = classes[task](diagram, **kw)
        sampler.create_prog()
    lower = sampler.plant.GetPositionLowerLimits()
    upper = sampler.plant.GetPositionUpperLimits()

    def sample_collision_free():
        while True:
            q = rng.uniform(lower, upper)
            sampler.plant.SetPositions(sampler.plant_context, q)
            if sampler.collision_free_constraint_eval.Eval(q) < 1:
                return q

    regions = ShelfCompartmentRegions(DEFAULT_SHELF_DEPTH_INSET)
    target_pose_of, placement_point = ContainmentPose(
        sampler.plant, sampler.plant_context, spec, "fingertips")
    mug_screen = None
    if task == "mug":
        with HiddenPrints():
            mug_screen = FloatingMugScreen(yaml_file, spec.robot_instances)
    target_qs, target_stats = SampleShelfTargets(
        RECORD_N_TARGETS,
        draw=lambda: rng.uniform(lower, upper),
        collision_free=lambda q: sampler.collision_free_constraint_eval.Eval(q) < 1,
        target_pose=target_pose_of, regions=regions,
        screen=(lambda X: mug_screen.Penetrates([X])) if mug_screen else None,
        max_consecutive_rejections=MAX_CONSECUTIVE_REJECTIONS,
        label=f"{robot}/{task}/shelf/fingertips")
    print(FormatTargetStats(f"{robot}/{task}", target_stats))
    guesses = [[sample_collision_free() for _ in range(RECORD_N_GUESSES)]
               for _ in range(RECORD_N_TARGETS)]
    grid_hash = hashlib.sha1(np.asarray(
        target_qs + [g for row in guesses for g in row]).tobytes()).hexdigest()[:12]
    if robot == "iiwa":
        grid_hash = f"{grid_hash}-{task}"  # the iiwa script's suffix; the Panda's has none
    return dict(yaml_file=yaml_file, scene_fingerprint=scene_fp, diagram=diagram,
                sampler=sampler, target_qs=target_qs, grid_hash=grid_hash)


## ------------------------------- the task gates -------------------------------- ##
## Transcribed from the benchmark scripts' `task_gate` closures (both robots gate alike).

def make_task_gate(task, ori_tol):
    if task == "mug":
        def task_gate(program, q):
            program.plant.SetPositions(program.plant_context, q)
            grasp = program.plant.GetFrameByName("between_fingers")
            p_W = grasp.CalcPoseInWorld(program.plant_context).translation()
            p_M = program.target_mug.middle.inverse() @ p_W
            axis_error = float(np.linalg.norm(p_M[:2]))
            height = float(abs(p_M[2]))
            ok = (axis_error <= TASK_TOL and height <= program.options.mug_height + TASK_TOL)
            return ok, dict(axis_error=axis_error, height=height)
    else:
        def task_gate(program, q):
            translation, wxyz = program.fk(q)
            target = program.target_pose
            axis_max = float(np.max(np.abs(np.asarray(translation) - target[:3])))
            target_rpy = RollPitchYaw(RotationMatrix(Quaternion(target[3:]))).vector()
            rpy_max = float(np.max(np.abs(
                np.asarray(orientation_error_rpy(wxyz, target_rpy), dtype=float))))
            ok = axis_max <= TASK_TOL and rpy_max <= ori_tol
            return ok, dict(pos_error=axis_max, rpy_error=rpy_max)
    return task_gate


## --------------------------------- the search ---------------------------------- ##

def set_branch(program, gc):
    """Fix the program's branch and drop every per-iterate memo: `QAndPose` keys on the
    decision vector's values only, so a point re-scored on another branch would otherwise
    read the previous branch's q."""
    program.gc = np.asarray(gc)
    program._flow_cache = {}
    program._analytic_aux = {}
    program._last_config = None


def pose_candidates(program, task, yaw_steps, z_steps):
    """`[M, 6]` xyz_rpy candidates for the task-frame pose, nearest the generating grasp first,
    with their (yaw, z) labels."""
    if task == "pose":
        return np.asarray(program.target_rpy, dtype=float)[None, :], [(None, None)]
    h = program.options.mug_height
    yaws = np.angle(np.exp(1j * 2 * np.pi * np.arange(yaw_steps) / yaw_steps))
    yaws = sorted(yaws, key=lambda y: (abs(y), y))
    zs = np.linspace(-h, h, z_steps) if z_steps > 1 else np.zeros(1)
    zs = sorted(zs, key=lambda z: (abs(z), z))
    X_WM = program.target_mug.middle
    rows, labels = [], []
    for yaw in yaws:
        for z in zs:
            X = X_WM @ RigidTransform(RotationMatrix.MakeZRotation(float(yaw)), [0.0, 0.0, z])
            rows.append([*X.translation(), *RollPitchYaw(X.rotation()).vector()])
            labels.append((float(yaw), float(z)))
    return np.asarray(rows, dtype=float), labels


def search_branch(program, gc, poses, labels, psis, task_gate, tol, stats):
    """First feasible candidate on branch `gc` (pose-major, psi-minor), or None."""
    set_branch(program, gc)
    tmap = BatchedAnalyticMap.from_program(program, dtype=torch.float64, device="cpu", gc=gc)
    n_psi = len(psis)
    X = np.concatenate([np.repeat(poses, n_psi, axis=0),
                        np.tile(psis, len(poses))[:, None]], axis=1)
    with torch.no_grad():
        q, u = tmap(torch.as_tensor(X, dtype=torch.float64))
    q, u = q.numpy(), u.numpy()
    n = program.num_arm_dof
    lo, hi = (np.asarray(b, dtype=float)[:n] for b in program.ConfigLimits())
    reach = 1.0 - program.options.analytic_reach_stepback
    ok = (np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(u), axis=1)
          & np.all(np.abs(u) <= reach + tol, axis=1)
          & np.all(q >= lo - tol, axis=1) & np.all(q <= hi + tol, axis=1))
    stats["candidates"] += len(X)
    stats["prefilter_pass"] += int(ok.sum())
    scale = getattr(program.options, "collision_row_scale", 0.1)
    gc_t = tuple(float(v) for v in np.ravel(gc))
    for i in np.flatnonzero(ok):
        q_full = program.PadQ(q[i])
        stats["collision_evals"] += 1
        if float(np.asarray(program.collision_free_constraint_eval.Eval(q_full)
                            ).flatten()[0]) > 1.0 + tol / scale:
            continue
        stats["verify_calls"] += 1
        verdict = bm.verify(program, None, task_gate, tol, x_lumped=X[i], relaxed_tol=TASK_TOL)
        if not verdict.feasible:
            continue
        q_ver = np.asarray(verdict.detail["q"], dtype=float)[:n]
        if tuple(float(v) for v in np.ravel(program.RecoverBranch(q_ver))) != gc_t:
            stats["branch_roundtrip_rejects"] += 1
            continue
        yaw, z = labels[i // n_psi]
        return dict(branch=list(gc_t), psi=float(X[i, 6]), yaw=yaw, z=z,
                    xyz_rpy=[float(v) for v in X[i, :6]], q=[float(v) for v in q_ver],
                    max_violation=verdict.detail.get("max_violation"),
                    min_distance=verdict.detail.get("min_distance"))
    return None


def build_analytic(robot, task, grid, ti, options, ik_solver):
    cls = program_classes(robot)[f"{task}_analytic"]
    sampler, q = grid["sampler"], grid["target_qs"][ti]
    with HiddenPrints():
        if task == "mug":
            diagram_with_mug, mug = GenerateDiagramWithMug(q, sampler, grid["yaml_file"], None)
            program = cls(diagram_with_mug, options=options, model=ik_solver)
            program.create_prog(target_mug=mug)
        else:
            sampler.plant.SetPositions(sampler.plant_context, q)
            pose = sampler.frame.CalcPoseInWorld(sampler.plant_context)
            target = np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()])
            program = cls(grid["diagram"], options=options, model=ik_solver)
            program.create_prog(target)
    return program


def generator_check(program, q_star, task_gate, tol):
    """q* scored at its own exact (pose, psi, branch): the sanity column."""
    n = program.num_arm_dof
    q_arm = np.asarray(q_star, dtype=float)[:n]
    set_branch(program, program.RecoverBranch(q_arm))
    program.plant.SetPositions(program.plant_context, program.PadQ(q_arm))
    X = program.frame.CalcPoseInWorld(program.plant_context)
    x = np.array([*X.translation(), *X.rotation().ToRollPitchYaw().vector(),
                  np.nan_to_num(program.RecoverPsi(q_arm))])
    verdict = bm.verify(program, None, task_gate, tol, x_lumped=x, relaxed_tol=TASK_TOL)
    return dict(feasible=bool(verdict.feasible), fail_reason=verdict.fail_reason,
                violations=verdict.detail.get("violations"),
                branch=[float(v) for v in np.ravel(program.gc)])


## ----------------------------------- main -------------------------------------- ##

def main():
    args = parse_args()
    t0 = time.time()
    if not 1 <= args.targets <= RECORD_N_TARGETS:
        raise SystemExit(f"--targets must be in 1..{RECORD_N_TARGETS}")
    checkpoint = args.checkpoint
    if checkpoint is None:
        cand = os.path.join(RepoDir(), RECORD_CHECKPOINT[args.robot])
        checkpoint = cand if os.path.exists(cand) else None
    options = base_options()
    grid = draw_record_grid(args.robot, args.task, options, checkpoint)
    expected = RECORD_GRID_HASH[(args.robot, args.task)]
    hash_ok = grid["grid_hash"] == expected
    print(f"grid_hash {grid['grid_hash']} (record {expected}): "
          f"{'MATCH' if hash_ok else 'MISMATCH'}; scene_fingerprint "
          f"{grid['scene_fingerprint'][:12]}; {time.time() - t0:.1f} s")

    ## The analytic arm's options exactly as the benchmark builds `analytic8`.
    analytic_options = replace(options, joint_centering_cost=1e0, analytic_branches=8)
    tol = analytic_options.acceptable_constr_viol_tol
    CheckClipCannotEscapeGate(analytic_options, tol)
    task_gate = make_task_gate(args.task, analytic_options.ik_constraint_tol[1])
    ik_solver = grid["sampler"].ik_solver

    records, nominal_branch, psi_bound = [], None, None
    stats = {k: dict(candidates=0, prefilter_pass=0, collision_evals=0, verify_calls=0,
                     branch_roundtrip_rejects=0) for k in ("nominal", "others")}
    for ti in range(args.targets):
        t_target = time.time()
        program = build_analytic(args.robot, args.task, grid, ti, analytic_options, ik_solver)
        nominal = tuple(float(v) for v in np.ravel(
            program.RecoverBranch(program.NominalConfiguration())))
        if nominal_branch is None:
            nominal_branch, psi_bound = nominal, float(program.PSI_BOUNDS)
        assert nominal == nominal_branch
        assert tuple(float(v) for v in np.ravel(program.gc)) == nominal  # create_prog's default
        psis = np.linspace(-psi_bound, psi_bound, args.psi_steps)
        poses, labels = pose_candidates(program, args.task, args.yaw_steps, args.z_steps)
        q_star = grid["target_qs"][ti]
        gen = generator_check(program, q_star, task_gate, tol)
        hit = search_branch(program, nominal, poses, labels, psis, task_gate, tol,
                            stats["nominal"])
        ## Every other branch is searched too (not stopped at the first reaching one), so each
        ## target records WHICH branches reach it -- the multi-branch comparison and a check
        ## that the search can say "no".
        per_branch = {}
        if args.branches == "both":
            per_branch[str(nominal)] = hit
            for gc in program.BranchSet():
                key = tuple(float(v) for v in np.ravel(gc))
                if key == nominal:
                    continue
                per_branch[str(key)] = search_branch(program, gc, poses, labels, psis,
                                                     task_gate, tol, stats["others"])
        reaching = [k for k, v in per_branch.items() if v is not None]
        rec = dict(target=ti, reachable=hit is not None, first_feasible=hit,
                   reachable_any_branch=bool(reaching) if args.branches == "both" else None,
                   branches_reaching=reaching if args.branches == "both" else None,
                   first_feasible_other_branches={k: v for k, v in per_branch.items()
                                                  if k != str(nominal)} or None,
                   generator_branch=gen["branch"],
                   generator_on_nominal=tuple(gen["branch"]) == nominal,
                   generator_feasible=gen["feasible"], generator_fail_reason=gen["fail_reason"],
                   generator_violations=gen["violations"],
                   seconds=time.time() - t_target)
        records.append(rec)
        print(f"  target {ti:2d}: nominal {'REACH' if rec['reachable'] else '-----'}"
              + (f"  any {'REACH' if rec['reachable_any_branch'] else '-----'}"
                 f" ({len(reaching)}/{len(per_branch)} branches)"
                 if args.branches == "both" else "")
              + f"  q* branch {tuple(int(v) for v in gen['branch'])}"
              f" verify {'ok' if gen['feasible'] else gen['fail_reason']}"
              + (f"  first psi {hit['psi']:+.3f}"
                 + (f" yaw {hit['yaw']:+.3f} z {hit['z']:+.3f}" if args.task == 'mug' else '')
                 if hit else "")
              + f"  ({rec['seconds']:.1f} s)", flush=True)

    n = len(records)
    n_nom = sum(r["reachable"] for r in records)
    n_any = sum(bool(r["reachable_any_branch"]) for r in records)
    by_branch = {}
    if args.branches == "both":
        for gc in program.BranchSet():
            key = str(tuple(float(v) for v in np.ravel(gc)))
            by_branch[key] = sum(key in r["branches_reaching"] for r in records)
    n_gen = sum(r["generator_feasible"] for r in records)
    n_gen_nom = sum(r["generator_on_nominal"] for r in records)
    out = dict(
        robot=args.robot, task=args.task, n_targets=n, targets=list(range(n)),
        branch=list(nominal_branch), reachable=n_nom,
        reachable_any_branch=n_any if args.branches == "both" else None,
        reachable_by_branch=by_branch or None,
        generator_feasible=n_gen, generator_on_nominal_branch=n_gen_nom,
        grid_hash=grid["grid_hash"], record_grid_hash=expected, grid_hash_match=hash_ok,
        scene_fingerprint=grid["scene_fingerprint"],
        record_grid=dict(seed=RECORD_SEED, n_targets=RECORD_N_TARGETS,
                         n_guesses=RECORD_N_GUESSES, scene="hardened", target_placement="shelf",
                         placement_point="fingertips", config="latent"),
        search_grid=dict(psi_steps=args.psi_steps, psi_range=[-psi_bound, psi_bound],
                         yaw_steps=args.yaw_steps if args.task == "mug" else None,
                         z_steps=args.z_steps if args.task == "mug" else None,
                         z_range=([-analytic_options.mug_height, analytic_options.mug_height]
                                  if args.task == "mug" else None),
                         tilt="held at zero (not swept)" if args.task == "mug" else None,
                         lower_bound=True),
        gate=dict(tol=tol, relaxed_tol=TASK_TOL, ori_tol=analytic_options.ik_constraint_tol[1],
                  feasible_means="src.benchmark.verify(...).feasible and RecoverBranch(q) == "
                                 "branch"),
        analytic_options=dict(analytic_branches=8,
                              analytic_clip_stepback=analytic_options.analytic_clip_stepback,
                              analytic_reach_stepback=analytic_options.analytic_reach_stepback),
        checkpoint=checkpoint, search_stats=stats, seconds=time.time() - t0,
        per_target=records, **bm.provenance())
    json_out = args.json_out or os.path.join(
        RepoDir(), "results/loink_branch_ceiling", f"{args.robot}_{args.task}.json")
    os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(out, fh, indent=1, default=bm._json_default)

    branch_s = "(" + ", ".join(f"{v:+g}" if args.robot == "iiwa" or k == 2 else f"{v:g}"
                               for k, v in enumerate(nominal_branch)) + ")"
    print()
    print(f"{args.robot} {args.task}: reachable {n_nom} / {n} on branch {branch_s}, "
          f"psi steps {args.psi_steps}"
          + (f", yaw steps {args.yaw_steps}, z steps {args.z_steps}" if args.task == "mug" else "")
          + f", grid_hash {grid['grid_hash']} ({'matches' if hash_ok else 'DOES NOT MATCH'} "
          f"the record)")
    if args.branches == "both":
        print(f"{args.robot} {args.task}: reachable {n_any} / {n} on ANY of the "
              f"{len(program.BranchSet())} analytic branches (same grid)")
        print(f"{args.robot} {args.task}: reachable per branch: "
              + ", ".join(f"{k} {v}" for k, v in by_branch.items()))
    print(f"{args.robot} {args.task}: generator q* verified feasible {n_gen} / {n}; "
          f"q* on the nominal branch {n_gen_nom} / {n}")
    print(f"wrote {json_out} ({time.time() - t0:.1f} s)")
    if not hash_ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
