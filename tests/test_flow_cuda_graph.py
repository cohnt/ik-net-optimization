"""Guard the CUDA-graph replay of the flow (`ProgramOptions.flow_cuda_graph`).

There is no test suite in this repo; run this by hand (needs a CUDA device):

    python tests/test_flow_cuda_graph.py

A replayed graph reads its input from, and writes its outputs to, fixed buffers, and checks
nothing at replay -- so its two characteristic failures are silent: returning the previous
call's outputs (a stale buffer), and handing a caller a tensor the next replay overwrites.
Both look like a slightly worse solver rather than a bug. So this compares, at a sequence of
DISTINCT iterates, everything `VarsToQ` returns on both paths -- the float configuration,
and the AutoDiffXd values and their derivative blocks -- between a graphed program and a
compiled-only one sharing the same network, and checks that a value held across a later
call is not overwritten. It also checks the two guards: the option refuses to run without
--compile, and no graph may be captured after `WarmUpJacobian` froze them.
"""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydrake.autodiffutils import AutoDiffXd, InitializeAutoDiff   # noqa: E402

import src.generic_program as gp                                     # noqa: E402
from src.generic_program import ProgramOptions                       # noqa: E402
from src.panda_program import PandaMugProgram                        # noqa: E402
from src.utils import BuildEnv, HiddenPrints                         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(REPO, "models/panda/panda__n6__step620000.pkl")
FAILURES = []
CHECKS = [0]


def check(name, condition, detail=""):
    CHECKS[0] += 1
    print(f"  [{'ok' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def ad_parts(q_ad):
    return (np.array([v.value() for v in q_ad]),
            np.vstack([v.derivatives() for v in q_ad]))


def main():
    if not torch.cuda.is_available():
        print("no CUDA device: nothing to test")
        return 0

    print("\n--- the option refuses to run without --compile ---")
    try:
        ProgramOptions(flow_cuda_graph=True)
        check("flow_cuda_graph without compile_flow_jacobian raises", False)
    except ValueError:
        check("flow_cuda_graph without compile_flow_jacobian raises", True)

    plain = ProgramOptions(compile_flow_jacobian=True, collision_avoidance=False)
    graphed = ProgramOptions(compile_flow_jacobian=True, flow_cuda_graph=True,
                             collision_avoidance=False)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=os.path.join(
            REPO, "models/panda/panda_finray_collision_hardened.yaml"))
        ref = PandaMugProgram(diagram, options=plain, checkpoint=CHECKPOINT)
        ref.create_prog()
        fast = PandaMugProgram(diagram, options=graphed, model=ref.ik_solver)
        fast.create_prog()
    gp.FreezeFlowGraphs(False)
    fast.WarmUpJacobian()

    print("\n--- graphed VarsToQ agrees with compiled-only at distinct iterates ---")
    rng = np.random.default_rng(0)
    n = len(ref.lumped_vars)
    width = ref.ik_solver.network_width
    worst_value = worst_ad = worst_grad = 0.0
    held = None
    for i in range(40):
        x = np.zeros(n)
        x[:3] = rng.uniform([0.3, -0.3, 0.2], [0.7, 0.3, 0.7])
        x[3:6] = rng.uniform(-np.pi, np.pi, 3)
        x[6:6 + width] = rng.standard_normal(width)
        x[6 + width:6 + width + 7] = rng.uniform(-0.1, 0.1, 7)
        x_ad = InitializeAutoDiff(x).flatten()

        q_ref, q_fast = ref.VarsToQ(x), fast.VarsToQ(x)
        worst_value = max(worst_value, float(np.max(np.abs(q_ref - q_fast))))
        (v_ref, g_ref), (v_fast, g_fast) = ad_parts(ref.VarsToQ(x_ad)), ad_parts(fast.VarsToQ(x_ad))
        worst_ad = max(worst_ad, float(np.max(np.abs(v_ref - v_fast))))
        worst_grad = max(worst_grad, float(np.max(
            np.abs(g_ref - g_fast) / np.maximum(1.0, np.abs(g_ref)))))

        if i == 0:
            live = fast.ik_inference(_lumped(fast, x))
            held = (live.detach().clone(), live)
    ## The float path is eager in `ref` and compiled in `fast`, so the two agree to compile's
    ## rounding, not bitwise; the Jacobian path is the same compiled kernels either way.
    check("float configuration matches", worst_value < 1e-10, f"max |dq| {worst_value:.1e}")
    check("AutoDiffXd values match", worst_ad < 1e-10, f"max |dq| {worst_ad:.1e}")
    check("AutoDiffXd derivative blocks match", worst_grad < 1e-10, f"max rel {worst_grad:.1e}")

    print("\n--- a returned tensor is not overwritten by a later replay ---")
    snapshot, live = held
    check("tensor from the first call survives 40 later calls",
          torch.equal(snapshot, live), f"max diff {float((snapshot - live).abs().max()):.1e}")

    print("\n--- no capture after the warm-up froze the graphs ---")
    other = gp.GraphedFlowCall(lambda v: v * 2.0, grad_enabled=False)
    try:
        other(torch.zeros(3, dtype=torch.float64, device="cuda"))
        check("a capture after WarmUpJacobian raises", False)
    except RuntimeError:
        check("a capture after WarmUpJacobian raises", True)
    gp.FreezeFlowGraphs(False)

    print(f"\n{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
    return 1 if FAILURES else 0


def _lumped(program, rpy_vars):
    """The network's input for `rpy_vars`, built the way `VarsToQ` builds it."""
    width = program.ik_solver.network_width
    xyz, quaternion = program.TaskVarsToPose7(rpy_vars[:6], float)
    v = np.zeros(7 + width + 7)
    v[:3], v[3:7] = xyz, quaternion
    v[7:] = rpy_vars[6:6 + width + 7]
    return v


if __name__ == "__main__":
    sys.exit(main())
