"""A batched, differentiable torch wrapper around the IKFlow network.

The program's own flow evaluation (`MakeFlowInference` in `src/generic_program.py`) is
written for one iterate at a time: a lumped vector `[c7 | z | q_c]` with the conditioning
pose already converted to xyz + wxyz, run through the network at batch size 1. A particle
method needs the same map for N particles at once, with gradients from `torch.autograd`
rather than from `jacrev` into Drake's `AutoDiffXd`. This module is that map, and nothing
more: it reproduces, column for column, what `VarsToQ` computes on the plain float path,

    q = flow(c, z)[:, :ndof] + q_c,

and its inverse (`rev=False`), which is what the paired start relies on.

Two things it deliberately does NOT do.

  * It never calls `.to()` on the shared network. `ConfigureNetworkDtype` casts the one
    `ik_solver.nn_model` every program in a grid shares, and the `cuda-graphs` branch bakes
    parameter addresses into a captured graph; a `.to(float32)` in place swaps the storage
    under it and the graph silently returns garbage. A swarm wanting a dtype other than the
    network's gets a PRIVATE `copy.deepcopy`, cast once and cached per process keyed on
    `(id(nn_model), dtype)` -- with the original kept alive in the cache entry, because
    Python reuses ids once an object is collected.
  * It takes the conditioning pose as the PROGRAM's decision variable -- `c6 = [xyz, rpy]`,
    the order of `lumped_vars` -- not as the 7-vector `MakeFlowInference` takes. The rpy ->
    quaternion conversion that `VarsToQ` performs with Drake's templated types per iterate
    is done here in torch (`quat_from_rpy_batched`), so autograd carries `dq/dc6` through
    it the way Drake's autodiff does on the program's side.

House rules observed throughout: explicit `dtype=`/`device=` on every tensor (`jrl.config`
sets torch's global default dtype to float32 and default device to cuda at import), no
Python loop over the batch, no in-place ops on autograd tensors.
"""

import copy

import torch
from torch import Tensor

## Private, dtype-cast copies of a shared network, keyed on (id(nn_model), dtype). The value
## holds the ORIGINAL as well as the copy so that the id in the key cannot be recycled by a
## later, unrelated module while the entry is alive.
_PRIVATE_COPIES = {}


def quat_from_rpy_batched(rpy: Tensor) -> Tensor:
    """Roll-pitch-yaw [B, 3] -> unit quaternion [B, 4] as wxyz, canonicalised to w >= 0.

    Drake's convention: `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`, so the quaternion is
    `q_z(yaw) * q_y(pitch) * q_x(roll)`. `RotationMatrix.ToQuaternion()` -- what `CToPose7`
    and `TaskVarsToPose7` go through -- returns the representative with `w >= 0`, so the same
    sign is applied here. The sign is DETACHED: it is a piecewise-constant choice of
    representative, and at `w = 0` it is a representation discontinuity the Drake path has
    too (its autodiff differentiates the chosen branch and nothing else). Document it; do not
    smooth it.

    NOTE: `src/svgd/kinematics_from_plant.py` carries a helper of the same name and
    semantics, written concurrently. The two are meant to be unified into one later; until
    then they must agree, and the test here pins this one against Drake.
    """
    if rpy.dim() != 2 or rpy.shape[1] != 3:
        raise ValueError(f"rpy must be [B, 3], got {tuple(rpy.shape)}")
    half = 0.5 * rpy
    cr, cp, cy = torch.cos(half[:, 0]), torch.cos(half[:, 1]), torch.cos(half[:, 2])
    sr, sp, sy = torch.sin(half[:, 0]), torch.sin(half[:, 1]), torch.sin(half[:, 2])
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = torch.stack([w, x, y, z], dim=1)
    # +1 where w >= 0 (including exactly 0, as Drake's `w >= 0` reads), -1 where w < 0.
    sign = torch.where(w.detach() < 0,
                       torch.tensor(-1.0, dtype=rpy.dtype, device=rpy.device),
                       torch.tensor(1.0, dtype=rpy.dtype, device=rpy.device))
    return quat * sign.unsqueeze(1)


def _private_copy(nn_model, dtype):
    """The cached private copy of `nn_model` cast to `dtype` (see module docstring)."""
    key = (id(nn_model), dtype)
    entry = _PRIVATE_COPIES.get(key)
    if entry is None or entry[0] is not nn_model:
        model = copy.deepcopy(nn_model).to(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        entry = _PRIVATE_COPIES[key] = (nn_model, model)
    return entry[1]


class BatchedFlow:
    """`(c6, z, q_c) -> q` for a batch of particles, and the inverse `(q, c6) -> z`.

    Layout of a lumped particle `X[i]` is the program's `lumped_vars`: `[c6 (6) | z (width)
    | q_c (ndof)]`, with `c6 = [xyz, roll, pitch, yaw]`. Anything after that (the `lift_q`
    block, where present) is ignored, as `VarsToQ`'s explicit end index ignores it.
    """

    def __init__(self, nn_model, width, ndof, dtype, device, chart_error_scale=0.0):
        self.width = int(width)
        self.ndof = int(ndof)
        if self.ndof > self.width:
            raise ValueError(f"ndof {self.ndof} exceeds the network width {self.width}")
        self.dtype = dtype
        self.device = torch.device(device)
        self.chart_error_scale = float(chart_error_scale)
        self.shared_model = nn_model
        network_dtype = next(nn_model.parameters()).dtype
        if network_dtype == dtype:
            self.model = nn_model
        else:
            self.model = _private_copy(nn_model, dtype)
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            raise ValueError(f"the network lives on {model_device}, asked for {self.device}; "
                             "this wrapper never moves the shared network")
        if self.chart_error_scale:
            # The SAME seeded perturbation `MakeFlowInference` adds: a CPU generator at seed
            # 0, W of shape (7 + width, ndof) then b of shape (ndof,), both float64, moved to
            # the device. Drawn in this order so the matrices are bit-identical.
            gen = torch.Generator(device="cpu").manual_seed(0)
            W = torch.randn((7 + self.width, self.ndof), generator=gen,
                            dtype=torch.float64, device="cpu").to(self.device)
            b = torch.randn(self.ndof, generator=gen,
                            dtype=torch.float64, device="cpu").to(self.device)
            self._W, self._b = W, b

    @property
    def nvars(self):
        """Width of a lumped particle: `6 + width + ndof`."""
        return 6 + self.width + self.ndof

    @staticmethod
    def from_program(program, dtype=None, device=None):
        """A wrapper around the network `program` holds, at the program's dtype by default.

        Reads `program.ik_solver.nn_model`, `ik_solver.network_width`, `num_arm_dof`,
        `options.chart_error_scale` and `torch_dtype`. The device defaults to wherever the
        network already is.
        """
        nn_model = program.ik_solver.nn_model
        if dtype is None:
            dtype = program.torch_dtype
        if device is None:
            device = next(nn_model.parameters()).device
        return BatchedFlow(nn_model, program.ik_solver.network_width, program.num_arm_dof,
                           dtype, device, getattr(program.options, "chart_error_scale", 0.0))

    def conditioning(self, c6: Tensor) -> Tensor:
        """`[xyz, rpy]` [B, 6] -> the network's conditioning row `[xyz, wxyz, 0]` [B, 8].

        The trailing zero is ikflow's softflow noise-scale column, the padding the flow was
        trained with; `MakeFlowInference` appends the same zero.
        """
        if c6.dim() != 2 or c6.shape[1] != 6:
            raise ValueError(f"c6 must be [B, 6], got {tuple(c6.shape)}")
        pad = torch.zeros((c6.shape[0], 1), dtype=c6.dtype, device=c6.device)
        return torch.cat([c6[:, :3], quat_from_rpy_batched(c6[:, 3:6]), pad], dim=1)

    def q(self, c6: Tensor, z: Tensor, qc: Tensor) -> Tensor:
        """`flow(c, z)[:, :ndof] + q_c`, [B, ndof]. Differentiable in all three arguments."""
        cond = self.conditioning(c6)
        output, _ = self.model(z, c=cond, rev=True)
        q = output[:, :self.ndof] + qc
        if self.chart_error_scale:
            # A function of the conditioning pose and latent only, exactly as on the
            # program's side, so the correction's identity block is untouched.
            inputs = torch.cat([cond[:, :7], z], dim=1).to(self._W.dtype)
            q = q + self.chart_error_scale * torch.sin(inputs @ self._W + self._b).to(q.dtype)
        return q

    def q_from_vars(self, X: Tensor) -> Tensor:
        """Lumped particles `[c6 | z | q_c]` [B, >= 6 + width + ndof] -> q [B, ndof]."""
        if X.dim() != 2 or X.shape[1] < self.nvars:
            raise ValueError(f"X must be [B, >= {self.nvars}], got {tuple(X.shape)}")
        w, n = self.width, self.ndof
        return self.q(X[:, :6], X[:, 6:6 + w], X[:, 6 + w:6 + w + n])

    def invert(self, q: Tensor, c6: Tensor) -> Tensor:
        """The latent that reproduces `q` under conditioning `c6`: the flow run forwards.

        Batched `InvertFlow`: `x = [q, 0...]` padded to the network width, `rev=False`.
        Exact (the flow is a bijection), evaluated under `no_grad` -- it initialises
        particles and is never differentiated through.
        """
        if q.dim() != 2 or q.shape[1] != self.ndof:
            raise ValueError(f"q must be [B, {self.ndof}], got {tuple(q.shape)}")
        with torch.no_grad():
            pad = torch.zeros((q.shape[0], self.width - self.ndof), dtype=q.dtype,
                              device=q.device)
            x = torch.cat([q, pad], dim=1)
            z, _ = self.model(x, c=self.conditioning(c6), rev=False)
        return z
