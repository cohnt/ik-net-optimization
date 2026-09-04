"""Assert that every constraint says what the task means.

There is no test suite in this repo; run this by hand:

    python tests/test_constraint_bounds.py

The rule it enforces is Thomas's, given after the pose constraint was found carrying a
`+-1e-4` box on its position rows: *"IK constraint tol should always be zero. The whole
point is that it's an equality constraint, satisfied exactly. Tolerance should be zero in
the mathematical program, only appearing in solver tolerance."*

Writing an equality as `lb = -tol, ub = +tol` does not loosen it slightly, it changes its
kind -- an interior-point method parks ON the face of an inequality rather than driving
the residual to zero. That failure is invisible in success rates and obvious in the
residuals: 67-97% of every arm's successful pose solves sat at exactly 1e-4 while the
orientation rows of the same constraint, correctly written `lb == ub == 0`, reached ~1e-9.

So this reads the ACTUAL bounds Drake was handed, never the docstrings, and checks both
directions: rows that must be equalities are equalities, and rows that are genuinely
inequalities (a collision bound, joint limits, a task's own freedom such as the grasp
height band, the analytic chart's reachability rows) have not been tightened by mistake.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generic_program import ProgramOptions                          # noqa: E402
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints    # noqa: E402
from src.panda_program import (PandaIKProgram, PandaIKProgramNumerical, # noqa: E402
                               PandaIKProgramAnalytic, PandaMugProgram,
                               PandaMugProgramNumerical, PandaMugProgramAnalytic)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []
CHECKS = [0]


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          {detail}")
        FAILURES.append(name)


def bounds_of(binding):
    e = binding.evaluator()
    return np.asarray(e.lower_bound(), dtype=float), np.asarray(e.upper_bound(), dtype=float)


def find(prog, description):
    for b in prog.GetAllConstraints():
        if b.evaluator().get_description() == description:
            return b
    return None


def stacked_rows(prog, label):
    """Bounds of the single stacked `AllIKFlowConstraints` binding.

    ApplyConstraints concatenates every registered row into one Drake binding so the
    network is evaluated once per iterate; CreateIKConstraint registers its rows first,
    so the task rows are the leading block.
    """
    b = find(prog, "AllIKFlowConstraints")
    if b is None:
        check(f"{label}: AllIKFlowConstraints present", False, "binding not found")
        return None, None
    return bounds_of(b)


def main():
    opts = ProgramOptions(collision_avoidance=True, joint_limits=True, use_float64=True)

    # ---------------------------------------------------------------- pose task
    print("\n--- pose task: the six IK rows ---")
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None,
                           directives_file=os.path.join(REPO, "models/panda/panda_collision.yaml"))
        sampler = PandaIKProgram(diagram, options=opts)
        sampler.create_prog()
    # A reachable target, the way the benchmark makes them: the gripper pose of a real
    # configuration. A fabricated pose could be unreachable and mask a bounds error.
    rng = np.random.default_rng(0)
    q_t = rng.uniform(sampler.plant.GetPositionLowerLimits(),
                      sampler.plant.GetPositionUpperLimits())
    translation, wxyz = sampler.fk(q_t)
    target = np.concatenate([translation, wxyz])
    ik_solver = sampler.ik_solver

    for cls, label in ((PandaIKProgram, "learned"), (PandaIKProgramNumerical, "numerical")):
        with HiddenPrints():
            p = cls(diagram, options=opts, model=ik_solver)
            p.create_prog(target)
        lb, ub = stacked_rows(p.prog, label)
        if lb is None:
            continue
        check(f"{label}: position rows are an EQUALITY at 0",
              np.array_equal(lb[:3], np.zeros(3)) and np.array_equal(ub[:3], np.zeros(3)),
              f"lb={lb[:3]} ub={ub[:3]}  <- a +-tol box here is the defect this test exists for")
        check(f"{label}: orientation rows are an EQUALITY at 0 (orientation_error_form='rpy')",
              np.array_equal(lb[3:6], np.zeros(3)) and np.array_equal(ub[3:6], np.zeros(3)),
              f"lb={lb[3:6]} ub={ub[3:6]}")
        numerical_prog = p.prog

    # The analytic arm pins its pose VARIABLES to the target; that row carries its whole
    # pose target. It used to be a box of +-(1e-4, 0.01), so this baseline got 0.01 rad of
    # orientation freedom per axis while the arms it is compared against were pinned to
    # zero -- it was solving an easier problem, and it used all of it (median rpy_error
    # 8.5e-3, p90 exactly on the bound).
    print("\n--- pose task: the analytic arm's pose target ---")
    from scripts.panda.panda_benchmark import AnalyticPoseOffset
    with HiddenPrints():
        offset = AnalyticPoseOffset(sampler.plant, sampler.plant_context, "panda_hand")
        pa = PandaIKProgramAnalytic(diagram, options=opts, model=ik_solver)
        pa.create_prog(target, pose_offset=offset)
    b = find(pa.prog, "PoseTargetConstraint")
    if b is None:
        check("analytic: PoseTargetConstraint present", False,
              "binding not found (was it renamed, or still called PoseTargetBoxConstraint?)")
    else:
        lb, ub = bounds_of(b)
        check("analytic: pose target is an EQUALITY (lb == ub)",
              np.array_equal(lb, ub), f"lb={lb} ub={ub}")
        check("analytic: pinned to the target pose itself",
              np.allclose(lb, pa.target_rpy), f"lb={lb} target={pa.target_rpy}")
    check("analytic: the pose target is NOT a bounding box",
          all(bb.evaluator().get_description() != "PoseTargetConstraint"
              for bb in pa.prog.bounding_box_constraints()),
          "as a variable bound IPOPT's bound_push projects the initial guess into it "
          "before evaluating anything, which silently replaces the paired start")

    print("\n--- rows that must STAY inequalities ---")
    # Collision and joint limits are rows INSIDE the stacked binding, not separate
    # bindings, so they have to be checked positionally. add_constraints registers
    # IK (6 rows), then collision (1), then joint limits (n_q) -- so everything after
    # the leading six must contain genuine inequalities. Looking them up by description
    # silently finds nothing and passes vacuously, which is how an earlier draft of this
    # test reported "ok" while checking nothing.
    lb, ub = stacked_rows(numerical_prog, "numerical")
    n_q = sampler.plant.num_positions()
    check("numerical: the stack carries the trailing collision + joint-limit rows",
          lb is not None and len(lb) >= 6 + 1 + n_q,
          f"stack has {0 if lb is None else len(lb)} rows, expected at least {6 + 1 + n_q}")
    if lb is not None and len(lb) > 6:
        tail_lb, tail_ub = lb[6:], ub[6:]
        check("numerical: collision row is one-sided (-inf, scale)",
              np.isneginf(tail_lb[0]) and np.isclose(tail_ub[0], opts.collision_row_scale),
              f"lb={tail_lb[0]} ub={tail_ub[0]}")
        check("numerical: every trailing row is a strict inequality, none tightened",
              bool(np.all(tail_ub > tail_lb)),
              f"equal rows at indices {np.flatnonzero(tail_ub <= tail_lb).tolist()}")
    bb = find(pa.prog, "ReachabilityConstraint")
    if bb is not None:
        lb, ub = bounds_of(bb)
        check("analytic: ReachabilityConstraint is the +-1 chart inequality",
              np.allclose(lb, -1) and np.allclose(ub, 1), f"lb={lb} ub={ub}")

    # ---------------------------------------------------------------- grasp task
    print("\n--- grasp task: mug-axis equality, height band ---")
    with HiddenPrints():
        mug_diagram = BuildEnv(
            meshcat=None,
            directives_file=os.path.join(REPO, "models/panda/panda_finray_collision.yaml"))
        mug_sampler = PandaMugProgram(mug_diagram, options=opts, model=ik_solver)
        mug_sampler.create_prog()
        q_m = rng.uniform(mug_sampler.plant.GetPositionLowerLimits(),
                          mug_sampler.plant.GetPositionUpperLimits())
        diagram_with_mug, mug = GenerateDiagramWithMug(
            q_m, mug_sampler, os.path.join(REPO, "models/panda/panda_finray_collision.yaml"),
            None)

    for cls, label in ((PandaMugProgram, "learned"), (PandaMugProgramNumerical, "numerical")):
        with HiddenPrints():
            pm = cls(diagram_with_mug, options=opts, model=ik_solver)
            pm.create_prog(target_mug=mug)
        lb, ub = stacked_rows(pm.prog, f"mug {label}")
        if lb is None:
            continue
        check(f"mug {label}: axis rows x, y are an EQUALITY at 0 (the task's definition)",
              np.array_equal(lb[:2], np.zeros(2)) and np.array_equal(ub[:2], np.zeros(2)),
              f"lb={lb[:2]} ub={ub[:2]}")
        check(f"mug {label}: height row is a BAND, deliberately not an equality",
              not np.isclose(lb[2], ub[2])
              and np.isclose(ub[2], opts.mug_height) and np.isclose(lb[2], -opts.mug_height),
              f"lb={lb[2]} ub={ub[2]} mug_height={opts.mug_height}")

    with HiddenPrints():
        pma = PandaMugProgramAnalytic(diagram_with_mug, options=opts, model=ik_solver)
        pma.create_prog(target_mug=mug, pose_offset=offset)
    b = find(pma.prog, "IKMugConstraint") or find(pma.prog, "MugConstraint")
    if b is not None:
        lb, ub = bounds_of(b)
        check("mug analytic: axis rows x, y are an EQUALITY at 0",
              np.array_equal(lb[:2], np.zeros(2)) and np.array_equal(ub[:2], np.zeros(2)),
              f"lb={lb[:2]} ub={ub[:2]}")
    # The mug analytic arm's pose variables are NOT pinned to a target -- the grasp is
    # imposed through the mug rows -- so its +-5 box is a variable region and stays one.
    check("mug analytic: variable regions on xyz_rpy/psi are present",
          len(pma.prog.bounding_box_constraints()) > 0,
          "add_constraints must still call BoundingBoxConstraint; IPOPT is documented in "
          "this repo as behaving poorly on unbounded variables")

    print(f"\n{CHECKS[0]} checks, {len(FAILURES)} failures")
    for f in FAILURES:
        print(f"  - {f}")
    if FAILURES:
        return 1
    print("CONSTRAINT BOUNDS OK -- equalities are exact; tolerance lives in the solver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
