"""The GVS push-rod arm's four programs: gradients, the paired start, and the forward model.

`test_gvs_arm_model.py` pins the MODEL. This pins what the PROGRAM does with it, which is
where the robot meets the shared machinery -- the same properties the soft PCS arm's programs
are held to, since these programs inherit theirs:

* `test_gradients_match_central_differences` -- the whole constraint stack under AutoDiffXd,
  every row block reported separately. This is the chain with no other check: the flow's
  `jacrev`, the IMPLICIT Jacobian through SoRoMoX's equilibrium, and Drake's own collision
  derivative, composed.
* `test_paired_start_is_exact` -- `SetStartFromQ` must land ON the rod-force vector it was
  given; the flow is a bijection so this is exact.
* `test_config_is_a_memo_hit` -- the joint-limit row calls `Config(vars)` and must NOT cost a
  second equilibrium solve.
* `test_joint_limits_row_bounds_the_configuration` -- 9 rows in [-1, 1], not 259 vacuous
  rows against the plant's +-inf floating-body limits.
* `test_verification_q_is_exact_under_a_learned_forward_model` -- an arm must never be graded
  by its own model of the robot.
* `test_every_program_accepts_what_the_driver_passes` -- one keyword set across all four.

Run with the project venv: `GVS_ARM_XLA_THREADS=2 .venv/bin/python tests/test_gvs_arm_program.py`.
Uses an UNTRAINED chart on purpose.
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from ikflow.ikflow_solver import IKFlowSolver  # noqa: E402
from ikflow.model import IkflowModelParameters  # noqa: E402
from jrl.robots import get_robot  # noqa: E402
from pydrake.all import InitializeAutoDiff  # noqa: E402

import src.register_robots  # noqa: E402,F401
from src.flow_loading import LEGACY_GVS_ARCH  # noqa: E402
from src.generic_program import ProgramOptions  # noqa: E402
from src.gvs_arm.model import GetModel  # noqa: E402
from src.gvs_arm.params import GetSpec, PRIMARY  # noqa: E402
from src.gvs_arm_program import (GvsArmIKProgram, GvsArmIKProgramNumerical,  # noqa: E402
                                 GvsArmMugProgram, GvsArmMugProgramNumerical)
from src.utils import BuildEnv, HiddenPrints  # noqa: E402

RUNG = PRIMARY
SPEC = GetSpec(RUNG)
_CACHE = {}


def _solver():
    if "solver" not in _CACHE:
        parameters = IkflowModelParameters()
        parameters.__dict__.update(dict(LEGACY_GVS_ARCH[RUNG], nb_nodes=4))
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


def _program(cls=GvsArmIKProgram, **kwargs):
    options = ProgramOptions(joint_centering_cost=1e-4, correction_cost_weight=10.0,
                             latent_trust_region=SPEC.latent_radius)
    with HiddenPrints():
        program = cls(_diagram(), options=options, rung=RUNG, model=_solver(), **kwargs)
    return program


def _a_target(program, seed=0):
    rng = np.random.default_rng(seed)
    cfg = program.SampleConfiguration(rng)
    return cfg, GetModel(SPEC).TipPose(cfg)


def test_every_program_accepts_what_the_driver_passes():
    import inspect

    required = {"diagram", "options", "rung", "model", "checkpoint", "fk", "surrogate"}
    for cls in (GvsArmIKProgram, GvsArmMugProgram,
                GvsArmIKProgramNumerical, GvsArmMugProgramNumerical):
        params = inspect.signature(cls.__init__).parameters
        missing = required - set(params) - {"diagram"}
        assert not missing, f"{cls.__name__}.__init__ does not accept {sorted(missing)}"
    print("PASS all four GVS arm programs accept the driver's keyword set")


def test_gradients_match_central_differences():
    for cls, label in ((GvsArmIKProgram, "learned"),
                       (GvsArmIKProgramNumerical, "joint space")):
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

        blocks, index = {"IK": 6, "collision": 1, "rod-force limits": SPEC.ninputs}, 0
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
    for cls, label in ((GvsArmIKProgram, "learned"),
                       (GvsArmIKProgramNumerical, "joint space")):
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
    assert len(row) == SPEC.ninputs, (
        f"the joint-limit row has {len(row)} rows; it must bound the {SPEC.ninputs} rod "
        f"forces, not the plant's {program.num_pos} positions")
    assert np.allclose(row.lb, -1.0) and np.allclose(row.ub, 1.0), (row.lb, row.ub)
    print(f"PASS the joint-limit row is {len(row)} rod-force rows in [-1, 1], "
          f"not {program.num_pos} vacuous plant rows")


def test_verification_q_is_exact_under_a_learned_forward_model():
    program = _program()
    cfg, pose = _a_target(program)
    with HiddenPrints():
        program.create_prog(target_pose=pose)
    exact = program.ConfigToPlantQ(cfg)
    assert np.array_equal(program.VerificationQ(exact), exact), (
        "under --fk analytic, VerificationQ must be the identity")

    class _WrongModel:
        def __call__(self, tensor):
            return torch.as_tensor(GetModel(SPEC).PlantQ(tensor.detach().cpu().numpy())) + 0.25

    program.fk_surrogate = _WrongModel()
    program._last_returned_cfg = np.asarray(cfg, dtype=float)
    graded = program.VerificationQ(program.ConfigToPlantQ(cfg))
    assert np.abs(graded - exact).max() < 1e-12, (
        "VerificationQ returned the surrogate's positions; an arm would be graded by its own "
        "model of the robot")
    print("PASS verify() grades on the exact model even when the forward model is wrong")


def test_grasp_programs_build_and_calibrate():
    """The grasp classes construct, recalibrate on `between_fingers`, and keep the GVS map."""
    program = _program(GvsArmMugProgram)
    assert program.frame.name() == "between_fingers"
    assert program.model is GetModel(SPEC)
    numerical = _program(GvsArmMugProgramNumerical)
    assert numerical.frame.name() == "between_fingers"
    print("PASS both grasp programs build on the GVS model and act at between_fingers")


if __name__ == "__main__":
    test_every_program_accepts_what_the_driver_passes()
    test_gradients_match_central_differences()
    test_paired_start_is_exact()
    test_config_is_a_memo_hit()
    test_joint_limits_row_bounds_the_configuration()
    test_verification_q_is_exact_under_a_learned_forward_model()
    test_grasp_programs_build_and_calibrate()
    print("ALL PASS")
