"""The Drake program's rows and costs, evaluated for N particles at once in torch.

`BatchedProgram.from_program(program)` reads a constructed `IKFlowProgram` -- learned or
joint-space arm, pose or grasp task -- and `evaluate(X)` returns, for a batch `X [N, nvars]`
of decision-variable vectors, EXACTLY the quantities Drake evaluates at each `X[i]`:

    * `drake_rows`  the `AllIKFlowConstraints` vector in Drake's row order (pose or mug
                    rows, then the collision row, then the joint-limit rows), unscaled;
    * `extra_rows`  the separate linear / quadratic bindings the learned arm carries (the
                    z box, the c box, the latent trust region), each in its binding's order;
    * `F`           the program's full objective (every cost binding) and `F_reported`,
                    the shared objective only (what `benchmark.reported_cost` measures);
    * `h`, `g`      the same rows split into equalities (`lb == ub`, as `value - lb`) and
                    inequalities `<= 0` (both finite sides), multiplied by `RowScaling`.

All of it is differentiable by autograd, so a particle solver takes per-particle gradients
from one backward. `tests/test_batched_program.py` pins every row against
`program.prog.EvalBinding` on all four rigid program classes x both arms, the gradients
against the program's own AutoDiffXd chain, and `init_from_q` against `SetStartFromQ`.

WHAT IS READ FROM THE PROGRAM, AND WHAT IS NOT. The row inventory is NOT hard-coded by task:
it is built by walking `program.constraints` -- the very list `ApplyConstraints` stacked
into the generic binding -- and dispatching on each entry's description, with the bounds
taken from `program.all_constraints.evaluator()` (the bounds Drake was handed) and
cross-checked against the list. The extra bindings are read from `prog.GetAllConstraints()`
by type (linear: `A x`; quadratic: `0.5 x'Qx + b'x`), the costs from `prog.GetAllCosts()`
by description and type. A constraint or cost this file does not know how to replay raises
at construction, by name, rather than being silently dropped from a swarm's target. Nothing
here encodes which arm "should" have which rows.

THE THREE PIECES IT BUILDS ON, each pinned by its own test:
  `batched_flow.BatchedFlow`            the network: `q = flow(c6, z)[:, :ndof] + q_c`;
  `kinematics_from_plant.BatchedFK`     the plant's kinematics in torch, body poses ONCE per
                                        evaluate, task frames composed with `frame_pose`
                                        (`frame_pose_from_q` per frame would run the whole
                                        chain twice -- launch-bound on CUDA, ~22 ms);
  `collision_backend.collision_row`     Drake's own `MinimumDistanceLowerBoundConstraint`,
                                        evaluated exactly, in process (no proxy).

ROBOTS WHOSE CONFIGURATION IS NOT THE PLANT'S q (the soft PCS arm) plug in behind two
hooks on `from_program`: `body_pose_provider` (a `BodyPoseProvider`: `body_names`,
`body_poses(q_plant) -> (quat, pos)`) and `config_to_plant_q` (a differentiable torch map
`cfg [N, ndof] -> q_plant [N, num_pos]`). The defaults are `KinematicTree.from_plant` and
the rigid arms' `PadQ` (gripper joints at 0.04), and the default `config_to_plant_q` is
CHECKED against `program.ConfigToPlantQ` at construction, so a robot that needs the hooks
is refused by name rather than evaluated on the wrong plant vector.

House rules: explicit `dtype=`/`device=` on every tensor (`jrl.config` mutates torch's
defaults at import); no Python loop over particles; no in-place ops on autograd tensors;
`evaluate` never raises on a non-finite particle -- NaN propagates through the flow, the
kinematics and the collision evaluator (which screens non-finite rows before Drake sees them).
"""

import os
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.solvers import (BoundingBoxConstraint, LinearConstraint, LinearCost,
                             QuadraticConstraint, QuadraticCost)

from src.svgd.batched_flow import BatchedFlow
from src.svgd.collision_backend import CollisionEvaluator, collision_row
from src.svgd.kinematics_from_plant import (BatchedFK, BodyPoseProvider, KinematicTree,
                                            canonical_quat, rpy_from_quat, wrap_residual)

## Mirrors `src.benchmark._REGULARIZER_COSTS` (asserted equal by the test rather than
## imported, so constructing a swarm does not import the benchmark harness).
REGULARIZER_COSTS = ("LatentRegularizerCost", "CorrectionCost", "JointLimitPenaltyCost")

## Descriptions of the separate learned-arm bindings -> the short keys of `extra_rows`.
EXTRA_ROW_KEYS = {
    "ZBoundingBoxConstraint": "z_box",
    "CBoxConstraint": "c_box",
    "LatentTrustRegion": "trust",
}
GENERIC_BINDING = "AllIKFlowConstraints"


## ------------------------------------------------------------------------------------ ##
##                                       Dataclasses                                     ##
## ------------------------------------------------------------------------------------ ##

@dataclass(frozen=True)
class RowScaling:
    """Multipliers applied to `h` and `g` ONLY, by row group. `drake_rows` and `extra_rows`
    are never scaled. The z and c boxes carry no scale (1.0)."""
    position: float = 1.0
    rotation: float = 1.0
    collision: float = 1.0
    joint_limit: float = 1.0
    mug: float = 1.0
    trust: float = 1.0

    def of(self, group: str) -> float:
        return float(getattr(self, group)) if group else 1.0


@dataclass(frozen=True)
class RowSpec:
    """Where one entry of `h` or `g` came from.

    `kind` is "eq" (an `h` row, `value - lb`), "lo" (a `g` row, `lb - value`) or "hi" (a
    `g` row, `value - ub`); `drake_binding` is the Drake binding's description and
    `drake_row` the row within it; `group` names the `RowScaling` field applied (empty for
    the z and c boxes). `row_group` is the row's DIAGNOSTIC label (`BatchedProgram.row_groups`:
    which block of rows it belongs to, e.g. for counting multiplier clips by group); it is
    never read by the step.
    """
    name: str
    kind: str
    drake_binding: str
    drake_row: int
    lb: float
    ub: float
    group: str = ""
    row_group: str = ""


@dataclass
class Evaluation:
    F: Tensor                      # [N] the program's FULL objective (all cost bindings)
    F_reported: Tensor             # [N] the shared objective (regularizers excluded)
    h: Tensor                      # [N, m_e] equality residuals (value - lb), scaled
    g: Tensor                      # [N, m_i] inequalities <= 0: all "lo" rows then all "hi"
    q: Tensor                      # [N, ndof] the configuration
    q_plant: Tensor                # [N, num_pos]
    pose_frame: Tuple[Tensor, Tensor]   # (quat [N, 4] wxyz, pos [N, 3]) of program.frame
    pose_flow: Tuple[Tensor, Tensor]    # (quat, pos) of the flow's conditioning frame
    collision_y: Optional[Tensor]  # [N] the UNSCALED Drake penalty; None if not evaluated
    drake_rows: Tensor             # [N, n_rows] AllIKFlowConstraints in Drake order, unscaled
    extra_rows: Dict[str, Tensor]  # "z_box", "c_box", "trust": each binding's value vector
    extras: Dict[str, object] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)   # filled when profile=True


## ------------------------------------------------------------------------------------ ##
##                                   Internal row records                                ##
## ------------------------------------------------------------------------------------ ##

@dataclass
class _RowBlock:
    """One contiguous block of the stacked value vector `[drake_rows | extra bindings]`."""
    binding: str          # Drake description
    kind: str             # "pose_pos" | "pose_rpy" | "mug" | "collision" | "joint_limit" | "linear" | "quadratic"
    start: int            # offset into the stacked vector
    size: int
    lb: np.ndarray
    ub: np.ndarray
    group: str            # RowScaling group ("" for unscaled)
    names: List[str]
    key: str = ""         # extra_rows key for extra bindings
    labels: Optional[List[str]] = None   # per-row diagnostic label; None: `key or kind`
    A: Optional[Tensor] = None       # linear: [size, nv]
    Q: Optional[Tensor] = None       # quadratic: [nv, nv]
    b: Optional[Tensor] = None       # quadratic: [nv]
    var_idx: Optional[Tensor] = None   # indices into the lumped vector


@dataclass
class _CostTerm:
    description: str
    kind: str             # "joint_centering" | "quadratic" | "linear"
    reported: bool
    Q: Optional[Tensor] = None
    b: Optional[Tensor] = None
    c: float = 0.0
    var_idx: Optional[Tensor] = None


def _normalise_device(device):
    """`torch.device("cuda")` with the current index filled in, so it compares equal to the
    `cuda:0` a parameter reports (`BatchedFlow` compares devices literally)."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return d


def _rigid_to_pose(X: RigidTransform):
    q = X.rotation().ToQuaternion().wxyz()
    return (tuple(float(v) for v in q), tuple(float(v) for v in X.translation()))


## ------------------------------------------------------------------------------------ ##
##                                      BatchedProgram                                   ##
## ------------------------------------------------------------------------------------ ##

class BatchedProgram:
    """See the module docstring. Construct through `from_program`."""

    def __init__(self):
        raise TypeError("use BatchedProgram.from_program(program, ...)")

    ## -------------------------------- construction -------------------------------- ##

    @staticmethod
    def from_program(program, dtype=torch.float64, device=None,
                     collision=None, row_scaling=None, body_pose_provider=None,
                     config_to_plant_q=None, regions_as_rows=True, profile=False):
        """Read everything off a constructed program (after `create_prog`).

        `device` defaults to wherever the program's network lives (learned arm) or CUDA if
        available (joint space). `collision` is a `CollisionEvaluator` to share; otherwise the
        program's own (`CollisionEvaluator.from_program`: its constraint, in this process).

        `body_pose_provider` / `config_to_plant_q` are the hooks for a robot whose
        configuration is not the plant's position vector (module docstring).
        `regions_as_rows=True` puts the z box, the c box and the trust region into `g`, as
        the general constraints they are in Drake; `g_spec` records which binding each row
        came from so a solver can treat them specially. Either way they are in `extra_rows`.
        """
        self = BatchedProgram.__new__(BatchedProgram)
        self.program = program
        self.options = program.options
        self.dtype = dtype
        self.profile = bool(profile)
        self.row_scaling = row_scaling if row_scaling is not None else RowScaling()
        self.regions_as_rows = bool(regions_as_rows)
        prog = program.prog

        self.is_learned = hasattr(program, "z")
        self.is_mug = hasattr(program, "target_mug")
        self.ndof = int(program.num_arm_dof)
        self.num_pos = int(program.num_pos)
        if getattr(self.options, "lift_q", False) or getattr(program, "q_lift", None) is not None:
            raise NotImplementedError("BatchedProgram does not replay the `lift_q` formulation")
        if getattr(self.options, "joint_limit_penalty_weight", 0.0) > 0.0 and self.is_learned:
            raise NotImplementedError("JointLimitPenaltyCost is not replayed; it is a refuted "
                                      "remedy whose knob is off everywhere")

        self.lumped_vars = np.asarray(program.lumped_vars)
        self.nvars = int(len(self.lumped_vars))
        self._lumped_idx = np.asarray(prog.FindDecisionVariableIndices(self.lumped_vars), dtype=int)
        self._num_vars_full = int(prog.num_vars())
        self._lumped_pos = {int(k): i for i, k in enumerate(self._lumped_idx)}

        if self.is_learned:
            if device is None:
                device = next(program.ik_solver.nn_model.parameters()).device
            self.device = _normalise_device(device)
            self.flow = BatchedFlow.from_program(program, dtype=dtype, device=self.device)
            self.width = self.flow.width
            if self.nvars != 6 + self.width + self.ndof:
                raise NotImplementedError(
                    f"{type(program).__name__}: lumped_vars has {self.nvars} entries, expected "
                    f"6 + {self.width} + {self.ndof} = [c6 | z | q_c]")
        else:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = _normalise_device(device)
            self.flow = None
            self.width = 0
            if self.nvars != self.ndof:
                raise NotImplementedError(
                    f"{type(program).__name__}: a joint-space arm's lumped_vars should be q "
                    f"({self.ndof}), got {self.nvars}; analytic formulations are not replayed")

        ## -- kinematics: body poses and the configuration -> plant map -----------------
        plant = program.plant
        if body_pose_provider is None:
            try:
                tree = KinematicTree.from_plant(plant)
            except (NotImplementedError, RuntimeError) as exc:
                raise NotImplementedError(
                    f"{type(program).__name__}: KinematicTree.from_plant refused this plant "
                    f"({exc}). This robot's configuration is not the plant's position vector; "
                    f"pass body_pose_provider= and config_to_plant_q= to from_program.") from exc
            body_pose_provider = BatchedFK(tree, dtype=dtype, device=self.device)
        self.fk: BodyPoseProvider = body_pose_provider
        self.plant_q_is_padded_cfg = config_to_plant_q is None
        if config_to_plant_q is None:
            config_to_plant_q = self._default_config_to_plant_q()
            self._check_config_to_plant_q(config_to_plant_q)
        self.config_to_plant_q: Callable[[Tensor], Tensor] = config_to_plant_q

        ## -- frames: program.frame and the flow's conditioning frame -------------------
        body_names = list(self.fk.body_names)
        self._frame_body, self._frame_X_BF = self._frame_offset(program.frame, body_names)
        X_ee_flow = getattr(program, "X_ee_flow", None)
        if X_ee_flow is None:
            X_ee_flow = RigidTransform()
        self.X_ee_flow = X_ee_flow
        flow_frame = program.frame_for_flow
        fb, (fq, fp) = self._frame_offset(flow_frame, body_names)
        X_BF_flow = RigidTransform(Quaternion(np.array(fq)), np.array(fp)) @ X_ee_flow
        self._flow_body, self._flow_X_BF = fb, _rigid_to_pose(X_BF_flow)
        ## The fast path: only the joints on the two frames' root paths, fixed transforms
        ## pre-composed (`FrameChain`). Available when the provider is the plant replay;
        ## a robot's own `BodyPoseProvider` goes through `body_poses` as before.
        self._frame_chain = None
        if isinstance(self.fk, BatchedFK):
            self._frame_chain = self.fk.frame_chain(
                [(self._frame_body, self._frame_X_BF), (self._flow_body, self._flow_X_BF)])

        ## -- the task --------------------------------------------------------------------
        if self.is_mug:
            M = program.target_mug.middle.GetAsMatrix4()
            Minv = np.linalg.inv(M)                 # the same op the Drake row performs
            self._MinvT = torch.tensor(Minv.T, dtype=dtype, device=self.device)
            self.mug_height = float(self.options.mug_height)
        else:
            tp = np.asarray(program.target_pose, dtype=float)
            self._target_pos = torch.tensor(tp[:3], dtype=dtype, device=self.device)
            target_rpy = RollPitchYaw(RotationMatrix(Quaternion(tp[3:]))).vector()
            self._target_rpy = torch.tensor(np.asarray(target_rpy, dtype=float), dtype=dtype,
                                            device=self.device)

        ## -- the row inventory -----------------------------------------------------------
        self._blocks: List[_RowBlock] = []
        self._build_generic_blocks()
        self._n_drake_rows = sum(b.size for b in self._blocks)
        self._build_extra_blocks()
        self._n_rows_total = sum(b.size for b in self._blocks)
        self._build_h_g_index()

        ## -- costs -----------------------------------------------------------------------
        self._costs: List[_CostTerm] = []
        self._build_costs()
        n = self.ndof
        q_nom = np.asarray(program.q_nominal, dtype=float)[:n]
        self._q_nominal = torch.tensor(q_nom, dtype=dtype, device=self.device)
        self._w_centering = float(self.options.joint_centering_cost)

        ## -- bounds and regions ----------------------------------------------------------
        lo, hi = self._variable_bounds()
        self.bounds = (torch.tensor(lo, dtype=dtype, device=self.device),
                       torch.tensor(hi, dtype=dtype, device=self.device))
        rlo, rhi = self._regions()
        self.regions = (torch.tensor(rlo, dtype=dtype, device=self.device),
                        torch.tensor(rhi, dtype=dtype, device=self.device))
        self.correction_bound = float(self.options.correction_bound)

        ## -- the collision row: the program's own constraint, in this process --------------
        self._has_collision = any(b.kind == "collision" for b in self._blocks)
        if collision is None and self._has_collision:
            collision = CollisionEvaluator.from_program(program)
        self.collision = collision
        self._native_c = self._compute_native_c()
        return self

    ## -- construction helpers --

    def _default_config_to_plant_q(self):
        """The rigid arms' `PadQ`: `[cfg, 0.04 x (num_pos - ndof)]`, differentiable."""
        n_pad = self.num_pos - self.ndof
        pad = torch.full((1, max(n_pad, 0)), 0.04, dtype=self.dtype, device=self.device)

        def pad_q(cfg):
            if n_pad == 0:
                return cfg
            return torch.cat([cfg, pad.expand(cfg.shape[0], n_pad)], dim=1)
        return pad_q

    def _check_config_to_plant_q(self, fn):
        """The default pad must agree with the program's own map, or this is not a robot
        the default serves."""
        rng = np.random.default_rng(0)
        lower, upper = self.program.ConfigLimits()
        lower = np.asarray(lower, dtype=float)[:self.ndof]
        upper = np.asarray(upper, dtype=float)[:self.ndof]
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise NotImplementedError(
                f"{type(self.program).__name__}: ConfigLimits are not finite; pass "
                f"config_to_plant_q= (the default pad is for the rigid arms).")
        cfg = rng.uniform(lower, upper, size=(3, self.ndof))
        mine = fn(torch.tensor(cfg, dtype=torch.float64, device=self.device)).detach().cpu().numpy()
        theirs = np.stack([np.asarray(self.program.ConfigToPlantQ(c), dtype=float) for c in cfg])
        if mine.shape != theirs.shape or np.abs(mine - theirs).max() > 1e-12:
            raise NotImplementedError(
                f"{type(self.program).__name__}: the default configuration -> plant map (PadQ) "
                f"does not reproduce program.ConfigToPlantQ; pass config_to_plant_q= and "
                f"body_pose_provider= for this robot.")

    def _frame_offset(self, frame, body_names):
        """`(body_index, (quat4, pos3))` of a Drake frame on this provider's bodies."""
        body = int(frame.body().index())
        if body >= len(body_names) or body_names[body] != frame.body().name():
            raise RuntimeError(
                f"frame {frame.name()!r} rides on body {frame.body().name()!r} (index {body}), "
                f"which the body-pose provider does not list at that index")
        return body, _rigid_to_pose(frame.GetFixedPoseInBodyFrame())

    def _build_generic_blocks(self):
        program = self.program
        ev = program.all_constraints.evaluator()
        if ev.get_description() != GENERIC_BINDING:
            raise RuntimeError(f"expected the generic binding to be {GENERIC_BINDING!r}, got "
                               f"{ev.get_description()!r}")
        lb_all = np.asarray(ev.lower_bound(), dtype=float).reshape(-1)
        ub_all = np.asarray(ev.upper_bound(), dtype=float).reshape(-1)
        lb_list = np.hstack([np.asarray(c.lb, dtype=float) for c in program.constraints])
        ub_list = np.hstack([np.asarray(c.ub, dtype=float) for c in program.constraints])
        if not (np.array_equal(lb_all, lb_list) and np.array_equal(ub_all, ub_list)):
            raise RuntimeError("the generic binding's bounds differ from program.constraints")
        start = 0
        for c in program.constraints:
            size = len(c)
            lb, ub = lb_all[start:start + size], ub_all[start:start + size]
            desc = c.description
            if desc == "IKConstraint":
                if self.is_mug:
                    if size != 3:
                        raise NotImplementedError(f"mug IKConstraint with {size} rows")
                    ## x, y pin the gripper to the mug's axis (the equalities); z keeps it
                    ## within the mug's height (the inequality).
                    self._blocks.append(_RowBlock(GENERIC_BINDING, "mug", start, 3, lb, ub, "mug",
                                                  ["mug_x", "mug_y", "mug_z"],
                                                  labels=["mug_xy", "mug_xy", "mug_z"]))
                else:
                    if size != 6:
                        raise NotImplementedError(f"pose IKConstraint with {size} rows")
                    self._blocks.append(_RowBlock(GENERIC_BINDING, "pose_pos", start, 3, lb[:3],
                                                  ub[:3], "position", ["pos_x", "pos_y", "pos_z"]))
                    self._blocks.append(_RowBlock(GENERIC_BINDING, "pose_rpy", start + 3, 3,
                                                  lb[3:], ub[3:], "rotation",
                                                  ["rpy_roll", "rpy_pitch", "rpy_yaw"]))
            elif desc == "CollisionFreeConstraint":
                if size != 1:
                    raise NotImplementedError(f"collision block with {size} rows")
                self._blocks.append(_RowBlock(GENERIC_BINDING, "collision", start, 1, lb, ub,
                                              "collision", ["collision"]))
                self.collision_scale = float(self.options.collision_row_scale)
                if not np.isclose(ub[0], self.collision_scale):
                    raise RuntimeError("collision row's upper bound is not collision_row_scale")
            elif desc == "JointLimitsConstraint":
                if size != self.ndof:
                    raise NotImplementedError(f"joint-limit block with {size} rows, ndof {self.ndof}")
                self._blocks.append(_RowBlock(GENERIC_BINDING, "joint_limit", start, size, lb, ub,
                                              "joint_limit", [f"joint_limit_{i}" for i in range(size)]))
            else:
                raise NotImplementedError(
                    f"{type(program).__name__}: constraint block {desc!r} is not replayed")
            start += size

    def _binding_var_idx(self, binding):
        full = self.program.prog.FindDecisionVariableIndices(binding.variables())
        pos = []
        for k in full:
            if int(k) not in self._lumped_pos:
                raise RuntimeError(f"binding {binding.evaluator().get_description()!r} touches a "
                                   f"variable outside lumped_vars")
            pos.append(self._lumped_pos[int(k)])
        return torch.tensor(pos, dtype=torch.long, device=self.device)

    def _build_extra_blocks(self):
        prog = self.program.prog
        start = self._n_drake_rows
        n_generic = 0
        for binding in prog.GetAllConstraints():
            ev = binding.evaluator()
            desc = ev.get_description()
            if desc == GENERIC_BINDING:
                n_generic += 1
                continue
            if isinstance(ev, BoundingBoxConstraint):
                continue      # a true variable bound: `bounds`, not a row
            lb = np.asarray(ev.lower_bound(), dtype=float).reshape(-1)
            ub = np.asarray(ev.upper_bound(), dtype=float).reshape(-1)
            key = EXTRA_ROW_KEYS.get(desc, desc or type(ev).__name__)
            var_idx = self._binding_var_idx(binding)
            if isinstance(ev, LinearConstraint):
                A = np.asarray(ev.GetDenseA(), dtype=float)
                size = A.shape[0]
                self._blocks.append(_RowBlock(
                    desc, "linear", start, size, lb, ub, "", [f"{key}_{i}" for i in range(size)],
                    key=key, A=torch.tensor(A, dtype=self.dtype, device=self.device),
                    var_idx=var_idx))
            elif isinstance(ev, QuadraticConstraint):
                Q = np.asarray(ev.Q(), dtype=float)
                b = np.asarray(ev.b(), dtype=float).reshape(-1)
                group = "trust" if key == "trust" else ""
                self._blocks.append(_RowBlock(
                    desc, "quadratic", start, 1, lb, ub, group, [key], key=key,
                    Q=torch.tensor(Q, dtype=self.dtype, device=self.device),
                    b=torch.tensor(b, dtype=self.dtype, device=self.device), var_idx=var_idx))
                size = 1
            else:
                raise NotImplementedError(
                    f"{type(self.program).__name__}: constraint binding {desc!r} of type "
                    f"{type(ev).__name__} is not replayed")
            start += size
        if n_generic != 1:
            raise RuntimeError(f"expected exactly one {GENERIC_BINDING} binding, found {n_generic}")

    def _build_h_g_index(self):
        """Split the stacked rows into h (lb == ub) and g (finite lo sides, then finite hi
        sides), recording a RowSpec per entry and the RowScaling multiplier."""
        eq_idx, eq_lb, eq_s, eq_spec = [], [], [], []
        lo_idx, lo_lb, lo_s, lo_spec = [], [], [], []
        hi_idx, hi_ub, hi_s, hi_spec = [], [], [], []
        for blk in self._blocks:
            is_region = blk.binding in ("ZBoundingBoxConstraint", "CBoxConstraint")
            if is_region and not self.regions_as_rows:
                continue
            scale = self.row_scaling.of(blk.group)
            for r in range(blk.size):
                lb, ub = float(blk.lb[r]), float(blk.ub[r])
                idx = blk.start + r
                base = dict(name=blk.names[r], drake_binding=blk.binding, drake_row=r
                            if blk.binding != GENERIC_BINDING else idx, lb=lb, ub=ub,
                            group=blk.group, row_group=blk.labels[r] if blk.labels is not None
                            else (blk.key or blk.kind))
                if lb == ub:
                    eq_idx.append(idx); eq_lb.append(lb); eq_s.append(scale)
                    eq_spec.append(RowSpec(kind="eq", **base))
                else:
                    if np.isfinite(lb):
                        lo_idx.append(idx); lo_lb.append(lb); lo_s.append(scale)
                        lo_spec.append(RowSpec(kind="lo", **base))
                    if np.isfinite(ub):
                        hi_idx.append(idx); hi_ub.append(ub); hi_s.append(scale)
                        hi_spec.append(RowSpec(kind="hi", **base))
        t = lambda v, dt=self.dtype: torch.tensor(np.asarray(v, dtype=float), dtype=dt, device=self.device)
        li = lambda v: torch.tensor(np.asarray(v, dtype=int), dtype=torch.long, device=self.device)
        self._eq = (li(eq_idx), t(eq_lb), t(eq_s))
        self._lo = (li(lo_idx), t(lo_lb), t(lo_s))
        self._hi = (li(hi_idx), t(hi_ub), t(hi_s))
        self.h_spec: List[RowSpec] = eq_spec
        self.g_spec: List[RowSpec] = lo_spec + hi_spec
        self._label_row_groups()
        ## The same split restricted to the GENERIC rows (functions of the configuration
        ## alone), for `evaluate_cfg`: `h_generic_mask[i]` says whether `h[:, i]` is one,
        ## and `evaluate_cfg`'s `h` is `h[:, h_generic_mask]` in the same order.
        nd = self._n_drake_rows
        gen = lambda idx: [k for k, i in enumerate(idx) if i < nd]
        ke, kl, kh = gen(eq_idx), gen(lo_idx), gen(hi_idx)
        pick = lambda v, ks: [v[k] for k in ks]
        self._eq_gen = (li(pick(eq_idx, ke)), t(pick(eq_lb, ke)), t(pick(eq_s, ke)))
        self._lo_gen = (li(pick(lo_idx, kl)), t(pick(lo_lb, kl)), t(pick(lo_s, kl)))
        self._hi_gen = (li(pick(hi_idx, kh)), t(pick(hi_ub, kh)), t(pick(hi_s, kh)))
        self.h_generic_mask = torch.tensor([i < nd for i in eq_idx], dtype=torch.bool, device=self.device)
        self.g_generic_mask = torch.tensor([i < nd for i in lo_idx] + [i < nd for i in hi_idx],
                                           dtype=torch.bool, device=self.device)
        ## ... and the complement, the EXTRA rows (functions of the decision variables
        ## alone), with their indices shifted into the extra-rows vector (`evaluate_extra`).
        ex = lambda idx: [k for k, i in enumerate(idx) if i >= nd]
        xe, xl, xh = ex(eq_idx), ex(lo_idx), ex(hi_idx)
        shift = lambda v, ks: [v[k] - nd for k in ks]
        self._eq_ex = (li(shift(eq_idx, xe)), t(pick(eq_lb, xe)), t(pick(eq_s, xe)))
        self._lo_ex = (li(shift(lo_idx, xl)), t(pick(lo_lb, xl)), t(pick(lo_s, xl)))
        self._hi_ex = (li(shift(hi_idx, xh)), t(pick(hi_ub, xh)), t(pick(hi_s, xh)))

    def _label_row_groups(self):
        """Finish every `h` / `g` entry's `row_group` label and build `row_groups`,
        `h_row_group`, `g_row_group` (`RowSpec`'s docstring). The label is the block's own
        (`_RowBlock.labels`, else its extra-rows key, else its kind: `pose_pos`, `pose_rpy`,
        `mug_xy`, `mug_z`, `collision`, `joint_limit`, `z_box`, `c_box`, `trust`, ...),
        read off the rows built here and never chosen per robot. A `g` label
        with rows on both sides is split into `<label>_lo` / `<label>_hi`; an `h` label that
        also names `g` rows becomes `<label>_eq`, so `h` and `g` never share a group. Plain
        Python (lists of str / int), so it adds nothing to the step's tensors."""
        g_base = [s.row_group for s in self.g_spec]
        sides = {}
        for s, lab in zip(self.g_spec, g_base):
            sides.setdefault(lab, set()).add(s.kind)
        g_lab = [f"{lab}_{s.kind}" if len(sides[lab]) > 1 else lab
                 for s, lab in zip(self.g_spec, g_base)]
        h_lab = [f"{s.row_group}_eq" if s.row_group in sides else s.row_group for s in self.h_spec]
        self.h_spec = [replace(s, row_group=lab) for s, lab in zip(self.h_spec, h_lab)]
        self.g_spec = [replace(s, row_group=lab) for s, lab in zip(self.g_spec, g_lab)]
        groups: List[str] = []
        for lab in h_lab + g_lab:
            if lab not in groups:
                groups.append(lab)
        self.row_groups: List[str] = groups                          # h groups, then g groups
        self.h_row_group: List[int] = [groups.index(lab) for lab in h_lab]   # [m_e] -> group
        self.g_row_group: List[int] = [groups.index(lab) for lab in g_lab]   # [m_i] -> group

    def _build_costs(self):
        prog = self.program.prog
        for binding in prog.GetAllCosts():
            ev = binding.evaluator()
            desc = ev.get_description()
            reported = desc not in REGULARIZER_COSTS
            if desc == "JointCenteringCost":
                self._costs.append(_CostTerm(desc, "joint_centering", reported))
            elif desc == "JointLimitPenaltyCost":
                raise NotImplementedError("JointLimitPenaltyCost is not replayed")
            elif isinstance(ev, QuadraticCost):
                self._costs.append(_CostTerm(
                    desc, "quadratic", reported,
                    Q=torch.tensor(np.asarray(ev.Q(), dtype=float), dtype=self.dtype, device=self.device),
                    b=torch.tensor(np.asarray(ev.b(), dtype=float).reshape(-1), dtype=self.dtype,
                                   device=self.device),
                    c=float(ev.c()), var_idx=self._binding_var_idx(binding)))
            elif isinstance(ev, LinearCost):
                self._costs.append(_CostTerm(
                    desc, "linear", reported,
                    b=torch.tensor(np.asarray(ev.a(), dtype=float).reshape(-1), dtype=self.dtype,
                                   device=self.device),
                    c=float(ev.b()), var_idx=self._binding_var_idx(binding)))
            else:
                raise NotImplementedError(
                    f"{type(self.program).__name__}: cost binding {desc!r} of type "
                    f"{type(ev).__name__} is not replayed")

    def _variable_bounds(self):
        """The tightest TRUE bounding box on lumped_vars (`IKFlowProgram._VariableBounds`)."""
        lower = np.full(self.nvars, -np.inf)
        upper = np.full(self.nvars, np.inf)
        prog = self.program.prog
        for binding in prog.bounding_box_constraints():
            ev = binding.evaluator()
            idx = prog.FindDecisionVariableIndices(binding.variables())
            lb = np.asarray(ev.lower_bound(), dtype=float).reshape(-1)
            ub = np.asarray(ev.upper_bound(), dtype=float).reshape(-1)
            for row, k in enumerate(idx):
                i = self._lumped_pos.get(int(k))
                if i is not None:
                    lower[i] = max(lower[i], lb[row])
                    upper[i] = min(upper[i], ub[row])
        return lower, upper

    def _regions(self):
        """The c box and z box as per-variable intervals (informational; general
        constraints in Drake). Read from the linear bindings whose matrix is the identity
        on their variables; +-inf elsewhere and on the joint-space arm."""
        lower = np.full(self.nvars, -np.inf)
        upper = np.full(self.nvars, np.inf)
        for blk in self._blocks:
            if blk.kind != "linear" or blk.key not in ("z_box", "c_box"):
                continue
            A = blk.A.cpu().numpy()
            if A.shape[0] != A.shape[1] or not np.array_equal(A, np.eye(A.shape[0])):
                warnings.warn(f"{blk.binding} is not an identity box; regions ignore it")
                continue
            idx = blk.var_idx.cpu().numpy()
            for r, i in enumerate(idx):
                lower[i] = max(lower[i], blk.lb[r])
                upper[i] = min(upper[i], blk.ub[r])
        return lower, upper

    def _compute_native_c(self):
        """The conditioning pose `create_prog` set, recomputed from the program's own data
        (the initial guess may since have been overwritten by `SetStartFromQ`).

        Pose programs record it as `program.initial_guess`. Grasp programs seed `c` in one
        of two ways: the Panda and soft PCS programs at the flow-frame pose of a grasp
        (`target_mug.middle @ X_grasp_ee`), the iiwa and screw programs at the mug centre
        with zero rpy (`[t, 0, 0, 0]`). Which rule applies is read from the class's MRO;
        an unknown class falls back to the current initial guess with a warning.
        """
        p = self.program
        if not self.is_learned:
            return None
        if not self.is_mug:
            ig = getattr(p, "initial_guess", None)
            if ig is not None:
                return np.asarray(ig, dtype=float).copy()
        else:
            names = [k.__name__ for k in type(p).__mro__]
            if any(n.startswith(("Panda", "Soft")) for n in names) and hasattr(p, "X_grasp_ee"):
                X_W_ee = p.target_mug.middle @ p.X_grasp_ee
                return np.concatenate([X_W_ee.translation(),
                                       X_W_ee.rotation().ToRollPitchYaw().vector()])
            if any(n.startswith(("Iiwa", "Screw")) for n in names):
                return np.array([*p.target_mug.middle.translation(), 0.0, 0.0, 0.0])
        warnings.warn(f"{type(p).__name__}: no rule for the native c; using the current "
                      f"initial guess of c")
        return np.asarray(p.prog.GetInitialGuess(p.c), dtype=float).copy()

    ## ------------------------------------ API --------------------------------------- ##

    @property
    def n_rows(self):
        """Width of `drake_rows`."""
        return self._n_drake_rows

    def native_c(self) -> Tensor:
        """The `c` that `create_prog` set (learned arm only), as a `[6]` tensor."""
        if self._native_c is None:
            raise AttributeError("the joint-space arm has no conditioning pose")
        return torch.tensor(self._native_c, dtype=self.dtype, device=self.device)

    def _as_X(self, X) -> Tensor:
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if X.dim() != 2 or X.shape[1] != self.nvars:
            raise ValueError(f"X must be [N, {self.nvars}], got {tuple(X.shape)}")
        return X

    def _split(self, X):
        """`(c6, z, qc)` of a learned batch."""
        w = self.width
        return X[:, :6], X[:, 6:6 + w], X[:, 6 + w:6 + w + self.ndof]

    def _config(self, X) -> Tensor:
        if self.is_learned:
            c6, z, qc = self._split(X)
            return self.flow.q(c6, z, qc)
        return X[:, :self.ndof]

    def _count(self, bucket):
        counts = getattr(self.program, "eval_counts", None)
        if counts is None:
            reset = getattr(self.program, "ResetEvalCounts", None)
            if reset is None:
                return
            reset()
            counts = self.program.eval_counts
        counts[bucket] = counts.get(bucket, 0) + 1

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def evaluate(self, X, need_collision=True, detach_kinematics=False,
                 row_jacobians=False, collision=None) -> Evaluation:
        """All rows and costs at every particle; one flow pass, one FK pass, one collision call.

        `need_collision=False` skips the collision row: the collision entries of `drake_rows` and
        `g` are then NaN and `collision_y` is None -- a diagnostic/timing mode, never a
        solver step's target. Counted once into `program.eval_counts["map_forward"]`.

        `row_jacobians=True` also forms the frames' geometric Jacobians (available when
        `has_analytic_row_jacobians`), so `generic_rows_jacobian_cfg` can be called on the
        result; `detach_kinematics=True` then builds the kinematics, rows and costs on a
        DETACHED copy of the configuration -- `q` keeps its graph back to `X` (the one
        backward a solver still needs), nothing downstream of it does. A solver that takes
        every row derivative in closed form uses both; the tests, which differentiate the
        rows by autograd, use neither.

        THE COLLISION GRADIENT. Where the plant vector carries an autograd graph the row
        goes through `collision_row` (its backward is the evaluator's stored gradient, which is
        also exposed as `extras["collision_grad"]`). Where it does NOT -- the joint-space
        arm, and every detached-kinematics evaluation -- the evaluator is called directly, with
        the gradient requested iff `row_jacobians`. (Until 2026-10-08 the gradient was read
        off the autograd node only, so in exactly those two cases `collision_grad` was None
        and the solver's collision-row Jacobian was silently ZERO on both arms.)

        `collision=(row [N], grad [N, nq] or None)`, scaled as the evaluator returns them,
        supplies the collision row precomputed (the solver's split step: Drake runs on the
        CPU while the GPU does the Jacobians); the evaluator is then not called.

        The pieces -- `kinematics`, `collision_eval`, `assemble` -- are public so a fused
        step can run them in stages with the collision call between; `evaluate` is exactly
        their composition.
        """
        X = self._as_X(X)
        timing = {}
        prof = self.profile
        t0 = time.perf_counter() if prof else 0.0

        cfg = self._config(X)
        self._count("map_forward")
        if prof:
            self._sync(); timing["flow"] = time.perf_counter() - t0; t0 = time.perf_counter()

        cfg_k = cfg.detach() if detach_kinematics else cfg
        kin = self.kinematics(cfg_k, row_jacobians)
        if prof:
            self._sync(); timing["fk"] = time.perf_counter() - t0; t0 = time.perf_counter()

        col_row, col_grad = None, None
        if self._has_collision and need_collision:
            if prof:
                self._sync(); tc = time.perf_counter()
            if collision is not None:
                col_row, col_grad = collision
            elif kin[0].requires_grad:
                col_row = collision_row(kin[0], self.collision)
                ## The evaluator's own `d row / d q_plant` [N, nq] rides on the autograd node
                ## (`CollisionRow.save_for_backward`); exposed so a solver assembling a
                ## row Jacobian reuses it instead of re-evaluating the row.
                col_grad = col_row.grad_fn.saved_tensors[0]
            else:
                col_row, col_grad = self.collision_eval(kin[0], need_grad=row_jacobians)
            if prof:
                timing["collision"] = time.perf_counter() - tc
        ev = self.assemble(X, cfg, cfg_k, kin, col_row, col_grad)
        if prof:
            self._sync(); timing["rows_costs"] = time.perf_counter() - t0 - timing.get("collision", 0.0)
        ev.timing = timing
        return ev

    def kinematics(self, cfg_k, row_jacobians=False):
        """`(q_plant, pose_frame, pose_flow, frame_jac)` at configurations `cfg_k`: the
        plant vector, the two frames' poses and (with `row_jacobians`) the program frame's
        geometric Jacobians `(Jp, Jw)`, else None. No collision, no rows."""
        q_plant = self.config_to_plant_q(cfg_k)
        frames = self._frame_poses(q_plant, jacobians=row_jacobians)
        frame_jac = (frames[0][2], frames[0][3]) if row_jacobians else None
        return q_plant, frames[0][:2], frames[1][:2], frame_jac

    def collision_eval(self, q_plant, need_grad=True):
        """`(row [N], grad [N, nq] or None)`: the SCALED collision row (and its
        gradient) at `q_plant`, as tensors on `q_plant`'s device and dtype. No autograd
        node. A host round trip (Drake runs on the CPU in float64)."""
        Q = q_plant.detach().to("cpu", torch.float64).numpy()
        value, grad = self.collision.eval(Q, need_grad=need_grad)
        row = torch.as_tensor(value).to(device=q_plant.device, dtype=q_plant.dtype)
        g = None if grad is None else torch.as_tensor(grad).to(device=q_plant.device, dtype=q_plant.dtype)
        return row, g

    def assemble(self, X, cfg, cfg_k, kin, col_row, col_grad) -> Evaluation:
        """Rows, costs and the `Evaluation` from the pieces: decision variables `X`, the
        configuration `cfg` (with its graph, if any) and its kinematics copy `cfg_k`, the
        `kinematics(cfg_k)` tuple, and the collision row and gradient (`None` row: the
        collision entries are NaN). Pure tensor arithmetic over a fixed block inventory --
        no Drake call, no host sync -- so a fused step can compile and capture it."""
        N = X.shape[0]
        q_plant, pose_frame, pose_flow, frame_jac = kin

        ## -- the stacked value vector, block by block, in Drake order --
        values = []
        collision_y = None
        rpy = None
        for blk in self._blocks:
            k = blk.kind
            if k == "pose_pos":
                values.append(pose_frame[1] - self._target_pos)
            elif k == "pose_rpy":
                rpy = rpy_from_quat(canonical_quat(pose_frame[0]))
                values.append(wrap_residual(rpy - self._target_rpy))
            elif k == "mug":
                ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
                hom = torch.cat([pose_frame[1], ones], dim=1) @ self._MinvT
                values.append(hom[:, :3])
            elif k == "collision":
                if col_row is not None:
                    collision_y = col_row / self.collision_scale
                    values.append(col_row.unsqueeze(1))
                else:
                    values.append(torch.full((N, 1), float("nan"), dtype=X.dtype, device=X.device))
            elif k == "joint_limit":
                values.append(cfg_k[:, :blk.size])
            elif k == "linear":
                values.append(X[:, blk.var_idx] @ blk.A.T)
            elif k == "quadratic":
                xv = X[:, blk.var_idx]
                values.append((0.5 * ((xv @ blk.Q) * xv).sum(dim=1) + xv @ blk.b).unsqueeze(1))
            else:                                   # pragma: no cover
                raise RuntimeError(k)
        V = torch.cat(values, dim=1)
        drake_rows = V[:, :self._n_drake_rows]
        extra_rows = {blk.key: V[:, blk.start:blk.start + blk.size]
                      for blk in self._blocks if blk.key}

        eq_idx, eq_lb, eq_s = self._eq
        lo_idx, lo_lb, lo_s = self._lo
        hi_idx, hi_ub, hi_s = self._hi
        h = (V[:, eq_idx] - eq_lb) * eq_s
        g = torch.cat([(lo_lb - V[:, lo_idx]) * lo_s, (V[:, hi_idx] - hi_ub) * hi_s], dim=1)

        ## -- costs --
        F = torch.zeros(N, dtype=X.dtype, device=X.device)
        F_rep = torch.zeros(N, dtype=X.dtype, device=X.device)
        for term in self._costs:
            if term.kind == "joint_centering":
                d = cfg_k[:, :self.ndof] - self._q_nominal
                val = 0.5 * self._w_centering * (d * d).sum(dim=1)
            elif term.kind == "quadratic":
                xv = X[:, term.var_idx]
                val = 0.5 * ((xv @ term.Q) * xv).sum(dim=1) + xv @ term.b + term.c
            else:
                xv = X[:, term.var_idx]
                val = xv @ term.b + term.c
            F = F + val
            if term.reported:
                F_rep = F_rep + val

        extras = {"q_inf": cfg.detach().abs().max(dim=1).values,
                  "collision_grad": col_grad, "frame_jac": frame_jac,
                  "rpy": None if rpy is None else rpy.detach()}
        if self.is_learned:
            c6, z, qc = self._split(X)
            extras["z_norm"] = z.detach().norm(dim=1)
            extras["qc_inf"] = qc.detach().abs().max(dim=1).values
        return Evaluation(F=F, F_reported=F_rep, h=h, g=g, q=cfg, q_plant=q_plant,
                          pose_frame=pose_frame, pose_flow=pose_flow, collision_y=collision_y,
                          drake_rows=drake_rows, extra_rows=extra_rows, extras=extras)

    def evaluate_cfg(self, Q, need_collision=True, row_jacobians=True) -> Evaluation:
        """The GENERIC rows (pose / mug, collision, joint limits -- the rows that are
        functions of the configuration alone) and the configuration-space costs at a
        batch of CONFIGURATIONS `Q [N, ndof]`, with no flow and no decision variables:
        the q-space evaluation an ADMM q-block or any projection onto the constraint
        manifold needs, the same code path as `evaluate` downstream of the configuration.

        Returns an `Evaluation` whose `h` / `g` are the generic entries only (in the order
        `h[:, h_generic_mask]` / `g[:, g_generic_mask]` of a full `evaluate`), whose `F` /
        `F_reported` carry the configuration-space cost terms only, whose `extra_rows` is
        empty, and whose `extras` carry `frame_jac` / `collision_grad` exactly as
        `evaluate(..., row_jacobians=True)` does, so `generic_rows_jacobian_cfg` applies.
        Identical to the matching slices of `evaluate(X)` at `Q = q(X)` (a test pins it).
        Not counted as a map evaluation: the map is not evaluated."""
        Q = torch.as_tensor(Q, dtype=self.dtype, device=self.device)
        if Q.dim() != 2 or Q.shape[1] != self.ndof:
            raise ValueError(f"Q must be [N, {self.ndof}], got {tuple(Q.shape)}")
        N = Q.shape[0]
        cfg = Q
        q_plant = self.config_to_plant_q(cfg)
        frames = self._frame_poses(q_plant, jacobians=row_jacobians)
        pose_frame, pose_flow = frames[0][:2], frames[1][:2]
        frame_jac = (frames[0][2], frames[0][3]) if row_jacobians else None
        values = []
        collision_y = None
        collision_grad = None
        rpy = None
        for blk in self._blocks:
            if blk.binding != GENERIC_BINDING:
                continue
            k = blk.kind
            if k == "pose_pos":
                values.append(pose_frame[1] - self._target_pos)
            elif k == "pose_rpy":
                rpy = rpy_from_quat(canonical_quat(pose_frame[0]))
                values.append(wrap_residual(rpy - self._target_rpy))
            elif k == "mug":
                ones = torch.ones((N, 1), dtype=Q.dtype, device=Q.device)
                hom = torch.cat([pose_frame[1], ones], dim=1) @ self._MinvT
                values.append(hom[:, :3])
            elif k == "collision":
                if need_collision:
                    row, collision_grad = self.collision_eval(q_plant, need_grad=row_jacobians)
                    collision_y = row / self.collision_scale
                    values.append(row.unsqueeze(1))
                else:
                    values.append(torch.full((N, 1), float("nan"), dtype=Q.dtype, device=Q.device))
            elif k == "joint_limit":
                values.append(cfg[:, :blk.size])
            else:                                   # pragma: no cover
                raise RuntimeError(k)
        V = torch.cat(values, dim=1)
        eq_idx, eq_lb, eq_s = self._eq_gen
        lo_idx, lo_lb, lo_s = self._lo_gen
        hi_idx, hi_ub, hi_s = self._hi_gen
        h = (V[:, eq_idx] - eq_lb) * eq_s
        g = torch.cat([(lo_lb - V[:, lo_idx]) * lo_s, (V[:, hi_idx] - hi_ub) * hi_s], dim=1)
        F, F_rep, _ = self.cost_cfg_parts(cfg)
        extras = {"q_inf": cfg.detach().abs().max(dim=1).values, "collision_grad": collision_grad,
                  "frame_jac": frame_jac, "rpy": None if rpy is None else rpy.detach()}
        return Evaluation(F=F, F_reported=F_rep, h=h, g=g, q=cfg, q_plant=q_plant,
                          pose_frame=pose_frame, pose_flow=pose_flow, collision_y=collision_y,
                          drake_rows=V, extra_rows={}, extras=extras, timing={})

    def extra_rows_values(self, X) -> Tensor:
        """The extra bindings' value vectors at `X`, stacked in `extra_blocks` order,
        `[N, n_extra]` (the quantity `extra_row_jacobian` is the derivative of). No graph."""
        X = self._as_X(X).detach()
        N = X.shape[0]
        vals = []
        for blk in self._blocks:
            if not blk.key:
                continue
            if blk.kind == "linear":
                vals.append(X[:, blk.var_idx] @ blk.A.T)
            else:
                xv = X[:, blk.var_idx]
                vals.append((0.5 * ((xv @ blk.Q) * xv).sum(dim=1) + xv @ blk.b).unsqueeze(1))
        if not vals:
            return torch.zeros(N, 0, dtype=X.dtype, device=X.device)
        return torch.cat(vals, dim=1)

    def evaluate_extra(self, X):
        """`(h_ex, g_ex, J_h_ex, J_g_ex)`: the EXTRA rows -- the entries of `h` / `g` that
        are NOT generic, `h[:, ~h_generic_mask]` and `g[:, ~g_generic_mask]` of a full
        `evaluate`, same order and same `RowScaling` -- and their closed-form Jacobians
        w.r.t. `X` (`[N, m, nvars]`). The x-only half of the program, for an ADMM x-block.
        Empty (zero-width) on the joint-space arm."""
        X = self._as_X(X).detach()
        V = self.extra_rows_values(X)
        J = self.extra_row_jacobian(X)
        eq_idx, eq_lb, eq_s = self._eq_ex
        lo_idx, lo_lb, lo_s = self._lo_ex
        hi_idx, hi_ub, hi_s = self._hi_ex
        h = (V[:, eq_idx] - eq_lb) * eq_s
        g = torch.cat([(lo_lb - V[:, lo_idx]) * lo_s, (V[:, hi_idx] - hi_ub) * hi_s], dim=1)
        J_h = J[:, eq_idx] * eq_s.view(1, -1, 1)
        J_g = torch.cat([-J[:, lo_idx] * lo_s.view(1, -1, 1), J[:, hi_idx] * hi_s.view(1, -1, 1)], dim=1)
        return h, g, J_h, J_g

    def cost_cfg_parts(self, cfg):
        """The cost terms that are functions of the CONFIGURATION (the joint-centering
        term): `(F [N], F_reported [N], dF/dcfg [N, ndof])` at `cfg [N, ndof]`."""
        cfg = torch.as_tensor(cfg, dtype=self.dtype, device=self.device)
        N = cfg.shape[0]
        F = torch.zeros(N, dtype=cfg.dtype, device=cfg.device)
        F_rep = torch.zeros_like(F)
        grad = torch.zeros(N, self.ndof, dtype=cfg.dtype, device=cfg.device)
        for term in self._costs:
            if term.kind != "joint_centering":
                continue
            d = cfg[:, :self.ndof] - self._q_nominal
            val = 0.5 * self._w_centering * (d * d).sum(dim=1)
            F = F + val
            if term.reported:
                F_rep = F_rep + val
            grad = grad + self._w_centering * d
        return F, F_rep, grad

    def cost_x_parts(self, X):
        """The cost terms that act on the DECISION VARIABLES directly (the quadratic /
        linear bindings): `(R [N], dR/dX [N, nvars], d2R/dX2 [nvars, nvars])` at `X`. The
        Hessian is constant (every such term is at most quadratic) and symmetrised. Zero
        everything on an arm without such terms (joint space)."""
        X = self._as_X(X).detach()
        N = X.shape[0]
        R = torch.zeros(N, dtype=X.dtype, device=X.device)
        g = torch.zeros(N, self.nvars, dtype=X.dtype, device=X.device)
        H = torch.zeros(self.nvars, self.nvars, dtype=X.dtype, device=X.device)
        for term in self._costs:
            if term.kind == "quadratic":
                xv = X[:, term.var_idx]
                R = R + 0.5 * ((xv @ term.Q) * xv).sum(dim=1) + xv @ term.b + term.c
                g[:, term.var_idx] += 0.5 * (xv @ term.Q + xv @ term.Q.transpose(0, 1)) + term.b
                Qs = 0.5 * (term.Q + term.Q.transpose(0, 1))
                H[term.var_idx.unsqueeze(1), term.var_idx.unsqueeze(0)] += Qs
            elif term.kind == "linear":
                xv = X[:, term.var_idx]
                R = R + xv @ term.b + term.c
                g[:, term.var_idx] += term.b
        return R, g, H

    def _frame_poses(self, q_plant, jacobians=False):
        """`(pose_frame, pose_flow)`, each `(quat [N, 4], pos [N, 3])` -- or, with
        `jacobians=True`, `(quat, pos, Jp, Jw)` -- through the `FrameChain` fast path
        where the provider is the plant replay, else by composing onto the provider's
        full `body_poses` (no Jacobians there: `has_analytic_row_jacobians` is False)."""
        if self._frame_chain is not None:
            a, b = self._frame_chain.poses(q_plant, jacobians=jacobians)
            return a, b
        if jacobians:
            raise NotImplementedError("row Jacobians in closed form need the plant replay "
                                      "(FrameChain); this provider has none")
        quat_all, pos_all = self.fk.body_poses(q_plant)
        return (self.fk.frame_pose(quat_all, pos_all, self._frame_body, self._frame_X_BF),
                self.fk.frame_pose(quat_all, pos_all, self._flow_body, self._flow_X_BF))

    @property
    def has_analytic_row_jacobians(self):
        """True when `generic_rows_jacobian_cfg` and `cost_gradient_parts` are available:
        the frames come from the plant replay (`FrameChain`) and the plant vector is the
        padded configuration (so a plant-space gradient is a configuration-space one by a
        slice). A robot behind its own hooks goes through autograd instead."""
        return self._frame_chain is not None and self.plant_q_is_padded_cfg

    def generic_rows_jacobian_cfg(self, ev: Evaluation) -> Tensor:
        """`d drake_rows / d cfg`, `[N, n_rows, ndof]`, in closed form from an
        `evaluate(..., row_jacobians=True)` result -- UNSCALED rows, Drake order:

          pose_pos     Jp                       (the frame origin's geometric Jacobian)
          pose_rpy     Einv(rpy) Jw             (rpy rates from the world angular velocity,
                                                 R = Rz(y) Ry(p) Rx(r); singular at |p| = pi/2
                                                 exactly where the row's own derivative is)
          mug          Minv[:3, :3] Jp          (the homogeneous row minus its constant)
          collision    the evaluator's d row / d q   (`extras["collision_grad"]`)
          joint_limit  I

        The `wrap_residual` shift and the target are constants. Plant columns beyond the
        configuration (the gripper pad) are dropped."""
        if ev.extras.get("frame_jac") is None:
            raise ValueError("generic_rows_jacobian_cfg needs evaluate(..., row_jacobians=True)")
        if not self.has_analytic_row_jacobians:
            raise NotImplementedError("closed-form row Jacobians need has_analytic_row_jacobians")
        Jp, Jw = ev.extras["frame_jac"]
        N = Jp.shape[0]
        nd = self.ndof
        J = torch.zeros(N, self._n_drake_rows, nd, dtype=Jp.dtype, device=Jp.device)
        for blk in self._blocks:
            if blk.binding != GENERIC_BINDING:
                continue
            sl = slice(blk.start, blk.start + blk.size)
            if blk.kind == "pose_pos":
                J[:, sl] = Jp[:, :, :nd]
            elif blk.kind == "pose_rpy":
                rpy = ev.extras["rpy"]
                cy, sy = torch.cos(rpy[:, 2]), torch.sin(rpy[:, 2])
                cp, sp = torch.cos(rpy[:, 1]), torch.sin(rpy[:, 1])
                z = torch.zeros_like(cy)
                o = torch.ones_like(cy)
                Einv = torch.stack([torch.stack([cy / cp, sy / cp, z], dim=1),
                                    torch.stack([-sy, cy, z], dim=1),
                                    torch.stack([cy * sp / cp, sy * sp / cp, o], dim=1)], dim=1)
                J[:, sl] = Einv @ Jw[:, :, :nd]
            elif blk.kind == "mug":
                R = self._MinvT[:3, :3].transpose(0, 1)
                J[:, sl] = R.unsqueeze(0) @ Jp[:, :, :nd]
            elif blk.kind == "collision":
                cg = ev.extras.get("collision_grad")
                if cg is not None:
                    J[:, blk.start] = cg[:, :nd]
            elif blk.kind == "joint_limit":
                J[:, sl] = torch.eye(nd, dtype=Jp.dtype, device=Jp.device)[:blk.size].unsqueeze(0)
        return J

    def cost_gradient_parts(self, X, cfg):
        """`(dF/dcfg [N, ndof], dF/dX|direct [N, nvars])` of the FULL objective in closed
        form from the cost inventory: the joint-centering term in the configuration, the
        quadratic / linear bindings directly in the decision variables. The total gradient
        is `J_q^T dF/dcfg + dF/dX|direct`."""
        X = self._as_X(X).detach()
        cfg = cfg.detach()
        N = X.shape[0]
        g_cfg = torch.zeros(N, self.ndof, dtype=X.dtype, device=X.device)
        g_x = torch.zeros(N, self.nvars, dtype=X.dtype, device=X.device)
        for term in self._costs:
            if term.kind == "joint_centering":
                g_cfg = g_cfg + self._w_centering * (cfg[:, :self.ndof] - self._q_nominal)
            elif term.kind == "quadratic":
                xv = X[:, term.var_idx]
                g_x[:, term.var_idx] += 0.5 * (xv @ term.Q + xv @ term.Q.transpose(0, 1)) + term.b
            else:
                g_x[:, term.var_idx] += term.b
        return g_cfg, g_x

    @property
    def generic_blocks(self):
        """`[(kind, start, size), ...]` of the `drake_rows` blocks, in Drake order; `kind`
        is one of pose_pos / pose_rpy / mug / collision / joint_limit. What a solver routes
        on when it assembles the generic rows' derivative w.r.t. the configuration: the
        joint-limit rows ARE the configuration, the collision row's gradient is the evaluator's
        (`Evaluation.extras["collision_grad"]`), and only the task rows need autograd."""
        return [(b.kind, b.start, b.size) for b in self._blocks if b.binding == GENERIC_BINDING]

    @property
    def extra_blocks(self):
        """`[(key, size), ...]` of the extra bindings, in `extra_rows` key order."""
        return [(b.key, b.size) for b in self._blocks if b.key]

    def extra_row_jacobian(self, X) -> Tensor:
        """`d extra_rows / dX` in closed form, `[N, n_extra, nvars]`, extra bindings in
        `extra_blocks` order: a linear block is its `A` on `var_idx`, a quadratic block's
        row is `Q x + b` on `var_idx`. No autograd, no graph."""
        X = self._as_X(X).detach()
        N = X.shape[0]
        n_extra = sum(b.size for b in self._blocks if b.key)
        J = torch.zeros(N, n_extra, self.nvars, dtype=X.dtype, device=X.device)
        off = 0
        for blk in self._blocks:
            if not blk.key:
                continue
            if blk.kind == "linear":
                J[:, off:off + blk.size, blk.var_idx] = blk.A.unsqueeze(0).expand(N, -1, -1)
            else:
                xv = X[:, blk.var_idx]
                J[:, off, blk.var_idx] = xv @ blk.Q + blk.b
            off += blk.size
        return J

    def project(self, X, regions=False):
        """Clip to the TRUE variable bounds (and to the c/z regions if `regions=True`),
        ALWAYS returning the per-particle Euclidean clip distance. The solver never calls
        this on a start without recording the distance."""
        X = self._as_X(X)
        lo, hi = self.bounds
        if regions:
            lo = torch.maximum(lo, self.regions[0])
            hi = torch.minimum(hi, self.regions[1])
        Xp = torch.minimum(torch.maximum(X, lo), hi)
        return Xp, (Xp - X).norm(dim=1)

    def init_from_q(self, q_init) -> Tensor:
        """Batched `SetStartFromQ`: the arm's own variables at configuration `q_init
        [N, ndof]`. Learned: `c` = the flow-frame pose at q (UNCLIPPED), `z` = the flow
        inverted at that c (UNCLIPPED), `q_c` = the residual `q - flow(c, z)` clipped to
        the correction box (NaN -> 0, +-inf -> +-bound, as the program does). Joint space:
        `q_init` itself, unprojected -- the program's own `SetStartFromQ` clips it into the
        q box, which `project()` reproduces while returning the distance."""
        q = torch.as_tensor(q_init, dtype=self.dtype, device=self.device)
        if q.dim() != 2 or q.shape[1] < self.ndof:
            raise ValueError(f"q_init must be [N, >= {self.ndof}], got {tuple(q.shape)}")
        q = q[:, :self.ndof]
        if not self.is_learned:
            return q.clone()
        with torch.no_grad():
            _, (quat, pos) = self._frame_poses(self.config_to_plant_q(q))
            c6 = torch.cat([pos, rpy_from_quat(canonical_quat(quat))], dim=1)
            z = self.flow.invert(q, c6)
            q_flow = self.flow.q(c6, z, torch.zeros_like(q))
            residual = q - q_flow
            bound = self.correction_bound
            residual = torch.nan_to_num(residual, nan=0.0, posinf=bound, neginf=-bound)
            qc = residual.clamp(-bound, bound)
        return torch.cat([c6, z, qc], dim=1)

    def to_drake_x(self, x_i) -> np.ndarray:
        """A lumped vector -> the full `prog.num_vars()` vector `EvalBinding` takes
        (`benchmark.verify`'s scatter)."""
        x_i = np.asarray(x_i, dtype=float).reshape(-1)
        if x_i.size != self.nvars:
            raise ValueError(f"expected {self.nvars} entries, got {x_i.size}")
        x = np.zeros(self._num_vars_full)
        x[self._lumped_idx] = x_i
        return x

    def from_drake_x(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self._num_vars_full:
            raise ValueError(f"expected {self._num_vars_full} entries, got {x.size}")
        return x[self._lumped_idx].copy()

    def jacobian_q(self, X) -> Tensor:
        """`dq/dX` per particle, `[N, ndof, nvars]` -- the single derivative funnel.

        Particles are independent through the flow (no cross-sample layer in eval mode),
        so `d(sum_i q_ik)/dX_j = dq_jk/dX_j` and the Jacobian is `ndof` reverse passes with
        no loop over particles. Identity (padded) on the joint-space arm. Counted into
        `program.eval_counts["map_jacobian"]`.
        """
        X = self._as_X(X)
        N = X.shape[0]
        if not self.is_learned:
            eye = torch.eye(self.ndof, dtype=X.dtype, device=X.device)
            return eye.unsqueeze(0).expand(N, self.ndof, self.nvars).clone()
        Xg = X.detach().requires_grad_(True)
        cfg = self._config(Xg)
        self._count("map_jacobian")
        rows = []
        for k in range(self.ndof):
            (gk,) = torch.autograd.grad(cfg[:, k].sum(), Xg, retain_graph=k < self.ndof - 1)
            rows.append(gk)
        return torch.stack(rows, dim=1)

    def vjp_q(self, X, R) -> Tensor:
        """`R^T dq/dX` per particle (`R [N, ndof]`), one backward. Counted as a Jacobian."""
        X = self._as_X(X)
        R = torch.as_tensor(R, dtype=self.dtype, device=self.device)
        if not self.is_learned:
            return R[:, :self.nvars].clone()
        Xg = X.detach().requires_grad_(True)
        cfg = self._config(Xg)
        self._count("map_jacobian")
        (g,) = torch.autograd.grad((cfg * R).sum(), Xg)
        return g

    def close(self):
        """Nothing to release: the collision row runs in this process on the program's own
        constraint. Kept so `with BatchedProgram...` and existing callers still work."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
