"""The jrl shim: what ikflow can reach, and what it must not reach by accident.

`test_helix_arm_kinematics.py` pins the MAP against Drake. This pins the WRAPPER -- the
object ikflow is handed, which is where a robot that is not in jrl meets a library that
assumes every robot is.

Two failure modes here are silent rather than loud, and both have precedent on the soft arm.
A method that quietly returns a wrong answer (jrl's Jacobian has no screw branch, so it
would emit a revolute column and drop the axial term) is worse than one that raises, so
everything unreachable raises with a named reason and this asserts that it does. And a
tensor returned on the wrong device raises inside ikflow's `validation_step` -- the first
eval AFTER training starts, so a cluster job burns its whole queue wait before saying so.

No pytest config in this repo, so these are plain `test_*` functions with a `__main__`
driver.
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from jrl.robots import get_robot

import src.register_robots  # noqa: F401  -- before anything resolves a robot by name
from src.helix_arm import kinematics as K
from src.helix_arm.params import PRIMARY, SPECS, GetSpec

SPEC = GetSpec(PRIMARY)


def test_every_rung_resolves_by_name():
    for name, spec in SPECS.items():
        robot = get_robot(name)
        assert robot.ndof == spec.ndof
        assert robot.spec.pitch == spec.pitch, f"{name} resolved to the wrong pitch"
        assert tuple(robot.actuated_joint_names) == spec.joint_names
    ## Registration runs at import and must be safe to repeat: three separate modules
    ## import the seam, and on the training path it happens again inside a delegated script.
    before = len(get_robot(PRIMARY).actuated_joint_names)
    import src.register_robots as seam
    seam.RegisterAll()
    assert len(get_robot(PRIMARY).actuated_joint_names) == before
    print(f"PASS all {len(SPECS)} rungs resolve through get_robot, idempotently")


def test_limits_are_the_specs_and_symmetric_where_it_matters():
    """The limits are not decoration: ikflow bakes them into `module_list.0`.

    With `sigmoid_on_output` false the first node is `x_i / max(|lo_i|, |hi_i|)`, a pure
    scaling with no offset, so a range not symmetric about zero lands that coordinate
    off-centre in an input range the layer cannot recentre.
    """
    robot = get_robot(PRIMARY)
    lower, upper = SPEC.limits
    assert robot.actuated_joints_limits == [(lo, hi) for lo, hi in zip(lower, upper)]
    for name, lo, hi in zip(SPEC.joint_names, lower, upper):
        assert abs(lo + hi) < 1e-12, (
            f"{name} is not symmetric about zero: [{lo}, {hi}]. ikflow's first layer cannot "
            f"recentre it, so half that coordinate's input range would go unused.")
    print("PASS the limits are the spec's, and every coordinate is symmetric about zero")


def test_sampling_returns_poses_that_are_the_map():
    robot = get_robot(PRIMARY)
    configurations, poses = robot.sample_joint_angles_and_poses(
        512, only_non_self_colliding=True)
    assert configurations.shape == (512, SPEC.ndof)
    assert poses.shape == (512, 7)

    ours = K.forward_kinematics(
        torch.as_tensor(configurations, dtype=torch.float64, device="cpu"), SPEC).numpy()
    assert np.abs(ours - poses).max() == 0.0, (
        "the dataset's poses must be the SAME map the program is checked against, to the "
        "bit; anything else and the dataset describes a different robot")

    lower, upper = (np.array(x) for x in SPEC.limits)
    assert (configurations >= lower).all() and (configurations <= upper).all()
    depth = K.self_collision_depth(
        torch.as_tensor(configurations, dtype=torch.float64, device="cpu"), SPEC)
    assert float(depth.max()) < 0.0, "a screened draw self-collides"
    print("PASS sampled poses are bit-identical to the map, in limits, and screened")


def test_self_collision_shapes_match_both_calling_paths():
    """jrl calls this per configuration from `evaluate_solutions` and batched elsewhere."""
    robot = get_robot(PRIMARY)
    one = robot.sample_joint_angles(1)[0]
    single = robot.config_self_collides(one)
    assert isinstance(single, bool), f"expected a plain bool, got {type(single)}"

    batch = torch.as_tensor(robot.sample_joint_angles(32), dtype=torch.float64, device="cpu")
    many = robot.config_self_collides(batch)
    assert isinstance(many, torch.Tensor) and many.dtype == torch.bool
    assert many.shape == (32,)
    assert bool(many[0]) == robot.config_self_collides(batch[0].numpy())
    print("PASS config_self_collides returns a bool for one and a tensor for a batch")


def test_results_come_back_on_the_callers_device():
    """ikflow's validation compares these against cuda tensors.

    A CPU return raises "Expected all tensors to be on the same device" INSIDE
    `validation_step`, which is the first eval after training starts -- so the failure
    lands after the queue wait, not at construction.
    """
    robot = get_robot(PRIMARY)
    if not torch.cuda.is_available():
        print("SKIP device round-trip (no cuda on this machine)")
        return
    x = torch.as_tensor(robot.sample_joint_angles(8), dtype=torch.float64, device="cuda")
    assert robot.forward_kinematics(x).device.type == "cuda"
    assert robot.config_self_collides(x).device.type == "cuda"
    assert robot.clamp_to_joint_limits(x).device.type == "cuda"
    cpu = x.cpu()
    assert robot.forward_kinematics(cpu).device.type == "cpu"
    print("PASS forward kinematics and the screen return on the caller's device")


def test_clamping_and_out_of_range_input():
    robot = get_robot(PRIMARY)
    lower, upper = (np.array(x) for x in SPEC.limits)
    far = np.full(SPEC.ndof, 1e3)
    assert np.allclose(robot.clamp_to_joint_limits(far), upper)
    assert np.allclose(robot.clamp_to_joint_limits(-far), lower)
    print("PASS clamping lands on the spec's own limits")


def test_the_unreachable_paths_raise_with_a_reason():
    """Everything that would go through klampt, or emit a revolute Jacobian column."""
    robot = get_robot(PRIMARY)
    cases = [
        ("urdf_filepath", lambda: robot.urdf_filepath),
        ("klampt_world_model", lambda: robot.klampt_world_model),
        ("forward_kinematics_klampt", lambda: robot.forward_kinematics_klampt(None)),
        ("_x_to_qs", lambda: robot._x_to_qs(None)),
        ("set_klampt_robot_config", lambda: robot.set_klampt_robot_config(None)),
        ("jacobian", lambda: robot.jacobian(None)),
        ("jacobian_np", lambda: robot.jacobian_np(None)),
        ("jacobian_batch_np", lambda: robot.jacobian_batch_np(None)),
        ("config_collides_with_env", lambda: robot.config_collides_with_env(None, None)),
        ("self_collision_distances", lambda: robot.self_collision_distances(None)),
        ("inverse_kinematics_step_levenburg_marquardt",
         lambda: robot.inverse_kinematics_step_levenburg_marquardt(None)),
    ]
    for name, call in cases:
        try:
            call()
        except NotImplementedError as exc:
            ## The message has to carry a reason, not just a name: this is what a future
            ## reader hits when ikflow grows a call into one of these, and "not
            ## implemented" alone would send them to reimplement it rather than to ask why
            ## it is absent.
            assert PRIMARY in str(exc) and len(str(exc)) > 60, (
                f"{name} raises without saying why: {exc}")
        else:
            raise AssertionError(f"{name} did not raise; it would answer as if revolute")

    x = torch.as_tensor(robot.sample_joint_angles(2), dtype=torch.float64, device="cpu")
    for kwargs in ({"return_full_link_fk": True}, {"return_full_joint_fk": True},
                   {"return_quaternion": False}):
        try:
            robot.forward_kinematics(x, **kwargs)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"forward_kinematics{kwargs} did not raise")
    print(f"PASS all {len(cases) + 3} unreachable paths raise with a named reason")


def test_the_flow_accepts_this_robot():
    """An untrained `IKFlowSolver` builds, and the bijection round-trips.

    This is the integration the shim exists for, and it is where the joint limits actually
    bite: the first node divides each coordinate by `max(|lo|, |hi|)`, and a chart built
    against a robot whose width or limits disagree fails here rather than in training.
    """
    from ikflow.config import DEVICE
    from ikflow.ikflow_solver import IKFlowSolver
    from ikflow.model import IkflowModelParameters

    from src.flow_loading import LEGACY_ARCH_BY_ROBOT

    robot = get_robot(PRIMARY)
    parameters = IkflowModelParameters()
    parameters.__dict__.update(dict(LEGACY_ARCH_BY_ROBOT[PRIMARY], nb_nodes=4))
    torch.manual_seed(0)
    solver = IKFlowSolver(parameters, robot)
    assert solver.network_width == SPEC.ndof, (
        f"{PRIMARY} is {SPEC.ndof} wide but the chart is {solver.network_width}; check "
        f"LEGACY_ARCH_BY_ROBOT, or InvertFlow will write past the end of its buffer")
    solver.nn_model.to(torch.float64).eval()

    q = robot.sample_joint_angles(16)
    pose = robot.forward_kinematics(
        torch.as_tensor(q, dtype=torch.float64, device="cpu")).cpu().numpy()
    x = np.zeros((16, solver.network_width))
    x[:, :SPEC.ndof] = q
    conditioning = np.concatenate([pose, np.zeros((16, 1))], axis=1)
    with torch.no_grad():
        latent, _ = solver.nn_model(
            torch.tensor(x, dtype=torch.float64, device=DEVICE),
            c=torch.tensor(conditioning, dtype=torch.float64, device=DEVICE), rev=False)
        back, _ = solver.nn_model(
            latent, c=torch.tensor(conditioning, dtype=torch.float64, device=DEVICE), rev=True)
    error = float((back[:, :SPEC.ndof].cpu() - torch.as_tensor(q, device="cpu")).abs().max())
    assert error < 1e-5, f"the flow did not round-trip: {error:.3e}"
    print(f"PASS an untrained chart builds at width {solver.network_width} and round-trips "
          f"to {error:.1e}")


if __name__ == "__main__":
    test_every_rung_resolves_by_name()
    test_limits_are_the_specs_and_symmetric_where_it_matters()
    test_sampling_returns_poses_that_are_the_map()
    test_self_collision_shapes_match_both_calling_paths()
    test_results_come_back_on_the_callers_device()
    test_clamping_and_out_of_range_input()
    test_the_unreachable_paths_raise_with_a_reason()
    test_the_flow_accepts_this_robot()
    print("ALL PASS")
