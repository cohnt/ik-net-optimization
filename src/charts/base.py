"""The interface between the learned formulation and whatever network is its chart.

Why this module exists
----------------------
The learned formulation is `q = chart(c, z) + correction`: the program needs a map from a
conditioning pose and a latent to a configuration, its derivative, and (for the paired start)
a way back from a configuration to a latent. Until the chart ablation every chart was an
IKFlow normalizing flow, so the program called the FrEIA module directly --
`nn_model(z, c=[pose7, 0], rev=True)` -- and that call, the 8-wide softflow conditioning row,
the zero-padding of `q` to the network width and the log-det all leaked into
`generic_program.py` and `svgd/batched_flow.py`.

A `Chart` owns those details. The program holds one as `self.ik_solver` (the old name is
kept so that the flow path reads exactly as it did) and reaches the network only through:

    decode(model, z [B, K], pose7 [B, 7]) -> (q [B, ndof], logdet [B] or None)
    encode(model, q [B, ndof], pose7 [B, 7]) -> z [B, K]

`pose7` is `[x, y, z, qw, qx, qy, qz]` in the frame the chart was trained on (see
`CalibrateFlowFrame`). Both take the module as an ARGUMENT rather than closing over
`self.nn_model`, because callers evaluate private dtype copies of it (`batched_flow`) and
compile closures over the module alone (`FlowJacobianGen`). Both must be pure torch,
`jacrev`- and `torch.compile`-friendly, with no data-dependent Python control flow.

Attributes every chart carries (the ones the code already read off an `IKFlowSolver`):

    nn_model        the torch module (dtype, device, compile/graph memo keys, private copies)
    network_width   K, the latent width = the number of `z` decision variables
    robot           the jrl `Robot` it was trained on (FK for frame calibration, limits)
    arch            the resolved architecture (the sidecar), recorded in run metadata
    arch_source     where `arch` came from
    family          "ikflow", "cvae", "loink", ...
    has_density     True when `decode` returns a log-det (only a flow has one)
    exact_inverse   True when the chart can express `q` EXACTLY in its own variables: for a
                    chart without `invert`, `decode(encode(q, pose7), pose7) == q` at the FK
                    pose (a flow, a bijection for every pose); for a chart with `invert`,
                    `decode(*reversed(invert(q)))` -- i.e. at the task the chart itself
                    assigns `q`. False when `encode` is an encoder's estimate and the residual
                    is left to the correction (the cVAE)
    task_input      what `decode` / `encode` take as the task: "pose7" (the default; the
                    flow, the cVAE) is `[x, y, z, qw, qx, qy, qz]` as above; "xyzrpy" (LOInK)
                    is the program's RAW `c = [x, y, z, roll, pitch, yaw]`, unconverted. The
                    program lumps `[c7 | z | correction]` or `[c6 | z | correction]`
                    accordingly (`IKFlowProgram.ChartTaskDim`, `_FlowVars`), and so does the
                    swarm (`BatchedFlow.task`). An xyzrpy chart's input is continuous across
                    the rpy cut, where `c`'s roll may cross +-pi; a pose7 chart's is not
                    defined in rpy at all. `ChartTaskFromPose7` forms the input from a
                    dataset / FK pose7 for screens.

One OPTIONAL member, `None` unless a chart provides it:

    invert(model, q [B, ndof]) -> (task [B, task_dim], z [B, K])

for a chart whose task output is its OWN: the task (in the chart's `task_input` form) is the
chart's estimate of `q`'s pose, not `q`'s FK pose, and only the two together reproduce `q`
(LOInK, whose `G : q -> (y_hat, z)` fits the task approximately, `task` = the raw `y_hat`).
`SetStartFromQ` uses it when present -- `c` the returned task, `z` as returned -- and records
how far that `c` sits from the FK pose's as `start_c_error`. A chart with `invert = None` is
started as before: `c` at the FK pose, `z = encode(q, pose7_FK)`.
"""


TASK_INPUTS = ("pose7", "xyzrpy")


def ChartTaskFromPose7(chart, pose7):
    """The chart's task input for a `[B, 7]` pose (a dataset pose, an FK pose): `pose7`
    itself for a "pose7" chart, its xyz + Drake rpy for an "xyzrpy" one."""
    if getattr(chart, "task_input", "pose7") == "pose7":
        return pose7
    from src.charts.loink import Pose7ToXyzRpy
    return Pose7ToXyzRpy(pose7)


class Chart:
    family = None
    has_density = False
    exact_inverse = False
    invert = None
    task_input = "pose7"

    @property
    def num_dof(self):
        return int(self.robot.ndof)

    def decode(self, model, z, pose7):
        raise NotImplementedError

    def encode(self, model, q, pose7):
        raise NotImplementedError
