"""The IKFlow normalizing flow as a `Chart` (see `src/charts/base.py`).

`FlowChart` wraps an `IKFlowSolver` and is what every learned program held as `ik_solver`
before the interface existed, plus `decode` / `encode`: the two network calls that used to
be written out at each call site. The ops are exactly the ones those call sites ran, in the
same order, so every flow result is bit-identical to the pre-interface code
(`tests/test_chart_interface.py` pins it with `torch.equal`).

What is flow-specific, and therefore lives here and nowhere else:

  * the conditioning row is EIGHT wide, `[x, y, z, qw, qx, qy, qz, 0]`: the trailing zero is
    ikflow's softflow noise-scale column, held at zero at inference;
  * the network is square, `network_width = max(ndof, dim_latent_space)` wide in and out,
    so `decode` keeps the first `ndof` outputs and `encode` zero-pads `q` to the width (the
    padding the flow was trained with -- see `src/soft_arm/kinematics.py` for why it must
    be zero). A width below `ndof` would make `encode` write past its buffer, so the
    constructor refuses it;
  * the log-det: the FrEIA call's second return, `log|det dx/dz|` of the reverse map, which
    is why `has_density` is True only here.
"""

import functools

import torch

from src.charts.base import Chart


def FlowDecode(model, z, pose7, num_dof):
    """`(q [B, ndof], log|det dx/dz| [B])`: the flow run in reverse (`rev=True`)."""
    c = torch.cat([pose7, torch.zeros((pose7.shape[0], 1), dtype=pose7.dtype,
                                      device=pose7.device)], dim=1)
    output, logdet = model(z, c=c, rev=True)
    return output[:, :num_dof], logdet


def FlowEncode(model, q, pose7, width):
    """The latent reproducing `q` under `pose7`: the flow run forwards (`rev=False`), exact."""
    c = torch.cat([pose7, torch.zeros((pose7.shape[0], 1), dtype=pose7.dtype,
                                      device=pose7.device)], dim=1)
    x = torch.cat([q, torch.zeros((q.shape[0], width - q.shape[1]), dtype=q.dtype,
                                  device=q.device)], dim=1)
    z, _ = model(x, c=c, rev=False)
    return z


class FlowChart(Chart):
    family = "ikflow"
    has_density = True
    exact_inverse = True

    def __init__(self, solver):
        # Through __dict__ so that __getattr__ below never sees a half-built instance.
        self.__dict__["_solver"] = solver
        ndof, width = int(solver.robot.ndof), int(solver.network_width)
        if ndof > width:
            raise ValueError(f"flow chart: ndof {ndof} exceeds the network width {width}; "
                             "encode would write past its buffer")
        # `decode` / `encode` are partials of module-level functions, NOT bound methods.
        # `MakeFlowInference` closes over `decode`, and torch.compile guards on everything a
        # compiled closure holds (the header above `MakeFlowInference` says why that has to be
        # the network and nothing per-program); a partial carries the two ints and no
        # reference to this object or the jrl robot behind it.
        self.decode = functools.partial(FlowDecode, num_dof=ndof)
        self.encode = functools.partial(FlowEncode, width=width)

    @property
    def solver(self):
        """The wrapped `IKFlowSolver`, for the flow-only code that needs it whole."""
        return self._solver

    @property
    def arch(self):
        """The resolved architecture `LoadFlowSolver` attached, with the family. The
        upstream Panda chart (`get_ik_solver`) carries none: ikflow reads its
        hyperparameters from its own model_descriptions.yaml."""
        return dict(getattr(self._solver, "arch", None) or {}, family=self.family)

    @property
    def arch_source(self):
        return getattr(self._solver, "arch_source", None)

    def __getattr__(self, name):
        # Only reached for names this object does not carry: `nn_model`, `network_width`,
        # `robot`, `load_state_dict`, ... read straight off the solver, as they always were.
        # An AttributeError (not a KeyError) before `_solver` exists, so that copy/pickle's
        # probes for `__deepcopy__` / `__setstate__` on a bare instance behave.
        solver = self.__dict__.get("_solver")
        if solver is None:
            raise AttributeError(name)
        return getattr(solver, name)
