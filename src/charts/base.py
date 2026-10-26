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
    family          "ikflow", "cvae", ...
    has_density     True when `decode` returns a log-det (only a flow has one)
    exact_inverse   True when `encode` reproduces `q` exactly (a bijection); False when it
                    is an encoder's estimate and the residual is left to the correction
"""


class Chart:
    family = None
    has_density = False
    exact_inverse = False

    @property
    def num_dof(self):
        return int(self.robot.ndof)

    def decode(self, model, z, pose7):
        raise NotImplementedError

    def encode(self, model, q, pose7):
        raise NotImplementedError
