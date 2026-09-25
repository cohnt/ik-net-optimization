"""The four `helix7` programs: gradients, the paired start, and the frame the flow speaks in.

`test_helix_arm_kinematics.py` pins the MAP and `test_helix_arm_robot.py` pins the SHIM.
This pins what the PROGRAM does with them, which is where the robot meets the shared
machinery.

* `test_gradients_match_central_differences` -- the whole constraint stack under AutoDiffXd,
  each row block reported separately so a failure names its own source. This is the chain
  with no other check: the flow's `jacrev`, the chain rule into the conditioning pose, and
  Drake's own collision derivative, composed. It is also the only place the SCREW joint's
  derivative meets the flow's.
* `test_paired_start_is_exact` -- `SetStartFromQ` must land ON the configuration it was
  given. The flow is a bijection, so this is exact rather than approximate, and a regression
  shows up in the benchmark as `unrepresentable_start` on every cell rather than as an error.
* `test_the_flow_frame_is_the_flange_on_both_tasks` -- the silent one. See its docstring.
* `test_joint_limits_row_bounds_the_screw_coordinate` -- the row this robot exists to
  stress, carrying the spec's own finite numbers rather than the parser's `+-inf`.

Uses an UNTRAINED chart on purpose: it exercises every code path while saying nothing about
solve quality, and the properties here are ones no amount of training would fix.

Run with the project venv: `.venv/bin/python tests/test_helix_arm_program.py`.
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

from src.flow_loading import LEGACY_ARCH_BY_ROBOT
from src.generic_program import ProgramOptions
from src.helix_arm.params import PRIMARY, GetSpec
from src.helix_arm_program import (HelixArmIKProgram, HelixArmIKProgramNumerical,
                                   HelixArmMugProgram, HelixArmMugProgramNumerical)
from src.utils import BuildEnv, HiddenPrints, Mug

SPEC = GetSpec(PRIMARY)
_CACHE = {}


def _solver():
    if "solver" not in _CACHE:
        parameters = IkflowModelParameters()
        parameters.__dict__.update(dict(LEGACY_ARCH_BY_ROBOT[PRIMARY], nb_nodes=4))
        torch.manual_seed(0)
        _CACHE["solver"] = IKFlowSolver(parameters, get_robot(PRIMARY))
    return _CACHE["solver"]


def _diagram():
    if "diagram" not in _CACHE:
        with HiddenPrints():
            _CACHE["diagram"] = BuildEnv(
                meshcat=None,
                directives_file=f"models/{PRIMARY}/{PRIMARY}_collision_hardened.yaml")
    return _CACHE["diagram"]


def _program(cls, options=None):
    options = options or ProgramOptions(latent_trust_region=SPEC.latent_trust_region)
    with HiddenPrints():
        return cls(_diagram(), options=options, robot=PRIMARY, model=_solver())


def _build(program, q_target):
    """Give the program a target this robot can actually reach: its own FK at `q_target`."""
    program.plant.SetPositions(program.plant_context, program.PadQ(q_target))
    X = program.frame.CalcPoseInWorld(program.plant_context)
    with HiddenPrints():
        if isinstance(program, HelixArmMugProgram):
            program.create_prog(target_mug=Mug(middle=X))
        else:
            target = np.concatenate([X.translation(),
                                     X.rotation().ToQuaternion().wxyz()])
            program.create_prog(target_pose=target)
    return program


def _draw(rng):
    lower, upper = (np.array(x) for x in SPEC.limits)
    return rng.uniform(lower, upper)


LEARNED = (("learned pose", HelixArmIKProgram), ("learned grasp", HelixArmMugProgram))
NUMERICAL = (("joint-space pose", HelixArmIKProgramNumerical),
             ("joint-space grasp", HelixArmMugProgramNumerical))


def test_gradients_match_central_differences():
    rng = np.random.default_rng(0)
    worst = {}
    for label, cls in LEARNED + NUMERICAL:
        program = _build(_program(cls), _draw(rng))
        x0 = np.asarray(program.prog.GetInitialGuess(program.lumped_vars), dtype=float)
        ## Somewhere generic, not at the guess: a start that happens to sit on a constraint
        ## face would compare gradients where the interesting terms are smallest.
        x0 = x0 + 0.05 * rng.standard_normal(x0.shape)

        for constraint in program.constraints:
            def block(x):
                q = program.VarsToQ(x)
                return np.asarray(constraint.eval_func(x, q, program.fk(q)), dtype=float)

            ## `.reshape(-1)`: InitializeAutoDiff returns a COLUMN vector, and with that
            ## shape `rpy_vars[0]` is a length-1 array rather than an AutoDiffXd, so
            ## VarsToQ would take its float branch and silently compare a float Jacobian
            ## against central differences of the same floats.
            x_ad = InitializeAutoDiff(x0).reshape(-1)
            q_ad = program.VarsToQ(x_ad)
            rows = constraint.eval_func(x_ad, q_ad, program.fk(q_ad))
            analytic = np.array([r.derivatives() if isinstance(r, AutoDiffXd)
                                 else np.zeros(len(x0)) for r in np.atleast_1d(rows)])

            eps = 1e-6
            numeric = np.zeros_like(analytic)
            for k in range(len(x0)):
                step = np.zeros_like(x0)
                step[k] = eps
                numeric[:, k] = (block(x0 + step) - block(x0 - step)) / (2 * eps)

            scale = max(1.0, float(np.abs(analytic).max()))
            error = float(np.abs(analytic - numeric).max() / scale)
            worst[(label, constraint.description)] = error
            assert error < 1e-5, (
                f"{label} / {constraint.description}: analytic and central differences "
                f"disagree by {error:.3e} (relative to {scale:.3g})")
    for (label, description), error in sorted(worst.items()):
        print(f"     {label:<18} {description:<26} {error:.2e}")
    print(f"PASS all {len(worst)} constraint blocks match central differences")


def test_paired_start_is_exact():
    """`SetStartFromQ` must land on the configuration it was handed, on every arm.

    Exact rather than approximate: the flow is a bijection, so `InvertFlow` returns the
    latent that reproduces `q_init` and the correction closes whatever is left. A regression
    here is not an error -- it is every cell of the grid scored `unrepresentable_start`.

    Note what the returned number is and is NOT. It is the distance from the unclipped `c`
    to its region, i.e. how far IPOPT's own `bound_push` will move the first iterate; the
    guess itself is set UNCLIPPED, which is the whole point of the region being a general
    linear constraint rather than a variable bound. So it is legitimately large when
    `q_init` has nothing to do with the target, and it must be zero when they coincide --
    both are asserted, because only the pair of them says the semantics are right.
    """
    rng = np.random.default_rng(1)
    for label, cls in LEARNED + NUMERICAL:
        for _ in range(3):
            q_init = _draw(rng)
            program = _build(_program(cls), _draw(rng))
            program.SetStartFromQ(q_init)
            x0 = program.prog.GetInitialGuess(program.lumped_vars)
            q0 = np.asarray([float(v) for v in program.VarsToQ(x0)], dtype=float)
            error = float(np.abs(q0[:SPEC.ndof] - q_init).max())
            assert error < 1e-9, f"{label}: start is {error:.3e} from q_init"

        ## Start AT the target's own configuration: the conditioning pose is then the
        ## target's, so it sits at the centre of its own region and nothing is projected.
        q_target = _draw(rng)
        program = _build(_program(cls), q_target)
        clip = program.SetStartFromQ(q_target)
        assert clip < 1e-9, (
            f"{label}: a start at the target's own configuration was reported {clip:.3e} "
            f"outside the conditioning region")
    print("PASS the paired start is exact on all four arms, and unprojected at the target")


def test_the_flow_frame_is_the_flange_on_both_tasks():
    """`X_ee_flow` must be the IDENTITY here, and the grasp program is where that can break.

    `HelixArmRobot.forward_kinematics` returns the flange pose, and the scene's flange frame
    is that same frame, so the measured offset between them is the identity. That makes this
    assertion an end-to-end witness over the whole chain: if the SDF's `<screw_thread_pitch>`
    units, the pitch's sign, the axis normalisation or the joint-origin convention disagreed
    between `generate_sdf.py` and `kinematics.py`, the offset would not be the identity.

    The grasp program is the case that matters. `frame_for_flow` falls back to `self.frame`
    when `ee_frame` is unset, so setting `ee_frame` AFTER `super().__init__()` -- which is
    what the iiwa's structure invites -- calibrates `between_fingers` against the flow and
    then applies that 0.200 m offset to the flange. Nothing raises: the offset IS constant,
    which is all `CalibrateFlowFrame` checks. It was written that way once here, and this is
    the assertion that caught it.
    """
    for label, cls in LEARNED:
        program = _program(cls)
        assert program.frame_for_flow.name() == SPEC.flange_link, (
            f"{label}: the flow is being conditioned on "
            f"{program.frame_for_flow.name()!r}, not the flange")
        offset = program.X_ee_flow
        assert np.abs(offset.translation()).max() < 1e-9, (
            f"{label}: X_ee_flow is {offset.translation()}, not the identity")
        assert abs(offset.rotation().ToAngleAxis().angle()) < 1e-9, label
    ## And the grasp program must still know where the fingers are relative to it.
    grasp = _program(HelixArmMugProgram)
    assert abs(float(np.linalg.norm(grasp.X_grasp_ee.translation())) - 0.200) < 1e-9
    print("PASS the flow is conditioned on the flange on both tasks, offset exactly identity")


def test_joint_limits_row_bounds_the_screw_coordinate():
    """The row this robot exists to stress, with finite bounds rather than the parser's."""
    lower, upper = (np.array(x) for x in SPEC.limits)
    k = SPEC.joint_names.index(SPEC.screw_joint_names[0])
    for label, cls in LEARNED + NUMERICAL:
        program = _build(_program(cls), _draw(np.random.default_rng(2)))
        rows = [c for c in program.constraints if c.description == "JointLimitsConstraint"]
        assert len(rows) == 1, label
        row = rows[0]
        assert len(row.lb) == SPEC.ndof, (label, len(row.lb))
        assert np.all(np.isfinite(row.lb)) and np.all(np.isfinite(row.ub)), label
        assert np.allclose(row.lb, lower, atol=1e-12), label
        assert np.allclose(row.ub, upper, atol=1e-12), label
        assert abs(row.lb[k] - lower[k]) < 1e-12 and abs(row.ub[k] - upper[k]) < 1e-12
    print(f"PASS the joint-limit row carries {SPEC.ndof} finite rows, screw coordinate "
          f"bounded at +-{upper[k]:.4f}")


def test_the_numerical_box_is_the_plants_own_limits():
    """The joint-space arm's box must come from the repaired plant, not a second copy."""
    lower, upper = (np.array(x) for x in SPEC.limits)
    for label, cls in NUMERICAL:
        program = _build(_program(cls), _draw(np.random.default_rng(3)))
        binding = program.bounding_box_constraint.evaluator()
        assert np.allclose(binding.lower_bound(), lower, atol=1e-12), label
        assert np.allclose(binding.upper_bound(), upper, atol=1e-12), label
        assert np.all(np.isfinite(binding.lower_bound())), label
    print("PASS the joint-space box is the plant's own repaired limits")


def test_a_missing_checkpoint_is_an_error_not_a_column_of_zeros():
    try:
        with HiddenPrints():
            HelixArmIKProgram(_diagram(), robot=PRIMARY)
    except ValueError as exc:
        assert "checkpoint" in str(exc)
    else:
        raise AssertionError("a missing chart must raise, not produce a column of zeros")
    print("PASS a missing checkpoint raises rather than scoring zero")


if __name__ == "__main__":
    test_the_flow_frame_is_the_flange_on_both_tasks()
    test_joint_limits_row_bounds_the_screw_coordinate()
    test_the_numerical_box_is_the_plants_own_limits()
    test_paired_start_is_exact()
    test_a_missing_checkpoint_is_an_error_not_a_column_of_zeros()
    test_gradients_match_central_differences()
    print("ALL PASS")
