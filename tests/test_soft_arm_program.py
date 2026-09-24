"""The soft arm's four programs: gradients, the paired start, and the two forward models.

`test_soft_arm_kinematics.py` pins the MAP. This pins what the PROGRAM does with it, which is
where the robot meets the shared machinery:

* `test_gradients_match_central_differences` -- the whole constraint stack under AutoDiffXd,
  every row block reported separately so a failure names its own source. This is the chain
  that has no other check: the flow's `jacrev`, the map's `jacfwd`, and Drake's own collision
  derivative, composed.
* `test_paired_start_is_exact` -- `SetStartFromQ` must land ON the configuration it was given.
  The flow is a bijection, so this is exact rather than approximate, and a regression here
  shows up in the benchmark as `unrepresentable_start` on every cell rather than as an error.
* `test_config_is_a_memo_hit` -- the joint-limit row calls `Config(vars)`, and it must NOT
  cost a second network pass. The rule this protects is the one about never adding a
  constraint that recomputes `VarsToQ`.
* `test_joint_limits_row_bounds_the_configuration` -- 12 rows in [-1, 1], not 231 vacuous
  rows against the plant's +-inf floating-body limits.
* `test_verification_q_is_exact_under_a_learned_forward_model` -- an arm must never be graded
  by its own model of the robot.

Run with the project venv: `.venv/bin/python tests/test_soft_arm_program.py`. Uses an
UNTRAINED chart on purpose -- it exercises every code path while saying nothing about solve
quality, and the properties here are ones no amount of training would fix.
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from ikflow.ikflow_solver import IKFlowSolver
from ikflow.model import IkflowModelParameters
from jrl.robots import get_robot
from pydrake.all import AutoDiffXd, InitializeAutoDiff

import src.soft_arm.register  # noqa: F401
from src.flow_loading import LEGACY_SOFT_ARCH
from src.generic_program import ProgramOptions
from src.soft_arm.params import GetSpec
from src.soft_arm_program import (SoftArmIKProgram, SoftArmIKProgramNumerical,
                                  SoftArmMugProgram)
from src.soft_arm import kinematics as K
from src.utils import BuildEnv, HiddenPrints

RUNG = "soft12"
SPEC = GetSpec(RUNG)
_CACHE = {}


def _solver():
    if "solver" not in _CACHE:
        parameters = IkflowModelParameters()
        parameters.__dict__.update(dict(LEGACY_SOFT_ARCH[RUNG], nb_nodes=4))
        torch.manual_seed(0)
        _CACHE["solver"] = IKFlowSolver(parameters, get_robot(RUNG))
    return _CACHE["solver"]


def _diagram():
    if "diagram" not in _CACHE:
        with HiddenPrints():
            _CACHE["diagram"] = BuildEnv(
                meshcat=None,
                directives_file=f"models/{RUNG}/{RUNG}_collision_hardened.yaml")
    return _CACHE["diagram"]


def _program(cls=SoftArmIKProgram, **kwargs):
    options = ProgramOptions(joint_centering_cost=1e-4, correction_cost_weight=10.0)
    with HiddenPrints():
        program = cls(_diagram(), options=options, rung=RUNG, model=_solver(), **kwargs)
    return program


def _a_target(program, seed=0):
    rng = np.random.default_rng(seed)
    cfg = program.SampleConfiguration(rng)
    pose = K.forward_kinematics(torch.as_tensor(cfg, dtype=torch.float64), SPEC).numpy()
    return cfg, pose


def test_gradients_match_central_differences():
    for cls, label in ((SoftArmIKProgram, "learned"),
                       (SoftArmIKProgramNumerical, "joint space")):
        program = _program(cls)
        cfg, pose = _a_target(program)
        with HiddenPrints():
            program.create_prog(target_pose=pose)
            program.SetStartFromQ(cfg)
        x0 = program.prog.GetInitialGuess(program.lumped_vars)

        analytic = np.array([row.derivatives() for row in
                             program.EvalAllConstraints(InitializeAutoDiff(x0).flatten())])
        step = 1e-6
        numeric = np.zeros_like(analytic)
        for i in range(len(x0)):
            bump = np.zeros(len(x0))
            bump[i] = step
            plus = np.asarray(program.EvalAllConstraints(x0 + bump), dtype=float)
            minus = np.asarray(program.EvalAllConstraints(x0 - bump), dtype=float)
            numeric[:, i] = (plus - minus) / (2 * step)

        blocks, index = {"IK": 6, "collision": 1, "strain limits": SPEC.ndof}, 0
        for name, count in blocks.items():
            rows = slice(index, index + count)
            worst = float(np.abs(analytic[rows] - numeric[rows]).max())
            assert worst < 1e-6, f"{label}/{name}: gradient off by {worst:.3e}"
            index += count
        scale = max(float(np.abs(analytic).max()), 1e-12)
        relative = float(np.abs(analytic - numeric).max() / scale)
        print(f"PASS {label}: {analytic.shape[0]} constraint rows x {len(x0)} variables "
              f"agree with central differences to {relative:.1e} relative")


def test_paired_start_is_exact():
    """`SetStartFromQ` lands ON the configuration, because the flow is a bijection."""
    for cls, label in ((SoftArmIKProgram, "learned"),
                       (SoftArmIKProgramNumerical, "joint space")):
        for seed in (0, 1, 2):
            program = _program(cls)
            cfg, pose = _a_target(program, seed=seed)
            with HiddenPrints():
                program.create_prog(target_pose=pose)
                program.SetStartFromQ(cfg)
            x0 = program.prog.GetInitialGuess(program.lumped_vars)
            error = float(np.abs(np.asarray(program.Config(x0), dtype=float) - cfg).max())
            assert error < 1e-9, f"{label} seed {seed}: start is {error:.3e} from q_init"
        print(f"PASS {label}: the paired start is exact to 1e-9 on three configurations")


def test_config_is_a_memo_hit():
    """`Config` must not cost a second network pass -- the joint-limit row calls it."""
    program = _program()
    cfg, pose = _a_target(program)
    with HiddenPrints():
        program.create_prog(target_pose=pose)
        program.SetStartFromQ(cfg)
    x0 = program.prog.GetInitialGuess(program.lumped_vars)
    program.ResetEvalCounts()
    program.EvalAllConstraints(x0)
    after_stack = dict(program.eval_counts)
    for _ in range(5):
        program.Config(x0)
    assert dict(program.eval_counts) == after_stack, (
        f"Config() evaluated the map again: {program.eval_counts} vs {after_stack}")
    print(f"PASS Config() is a pure memo hit "
          f"({after_stack['map_forward']} forward map evaluations for the whole stack, "
          f"unchanged by five Config() calls)")


def test_joint_limits_row_bounds_the_configuration():
    program = _program()
    cfg, pose = _a_target(program)
    with HiddenPrints():
        program.create_prog(target_pose=pose)
    row = next(c for c in program.constraints if c.description == "JointLimitsConstraint")
    assert len(row) == SPEC.ndof, (
        f"the joint-limit row has {len(row)} rows; it must bound the {SPEC.ndof} "
        f"configuration coordinates, not the plant's {program.num_pos} positions")
    assert np.allclose(row.lb, -1.0) and np.allclose(row.ub, 1.0), (row.lb, row.ub)
    print(f"PASS the joint-limit row is {len(row)} strain rows in [-1, 1], "
          f"not {program.num_pos} vacuous plant rows")


def test_verification_q_is_exact_under_a_learned_forward_model():
    """Identity on the exact model; the exact kinematics under a surrogate.

    Checked WITHOUT a fitted surrogate, by standing a deliberately wrong map in for one: the
    property under test is that `verify()` grades somewhere other than where the program
    optimised, and that must hold however bad the surrogate is.
    """
    program = _program()
    cfg, pose = _a_target(program)
    with HiddenPrints():
        program.create_prog(target_pose=pose)
    exact = program.ConfigToPlantQ(cfg)
    assert np.array_equal(program.VerificationQ(exact), exact), (
        "under --fk analytic, VerificationQ must be the identity")

    class _WrongModel:
        def __call__(self, tensor):
            return K.config_to_plant_q(tensor, SPEC, device="cpu") + 0.25

    program.fk_surrogate = _WrongModel()
    program._last_returned_cfg = np.asarray(cfg, dtype=float)
    graded = program.VerificationQ(program.ConfigToPlantQ(cfg))
    assert np.abs(graded - exact).max() < 1e-12, (
        "VerificationQ returned the surrogate's positions; an arm would be graded by its own "
        "model of the robot")
    print("PASS verify() grades on exact kinematics even when the forward model is wrong")


if __name__ == "__main__":
    test_gradients_match_central_differences()
    test_paired_start_is_exact()
    test_config_is_a_memo_hit()
    test_joint_limits_row_bounds_the_configuration()
    test_verification_q_is_exact_under_a_learned_forward_model()
    print("ALL PASS")
