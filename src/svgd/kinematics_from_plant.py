"""Batched, differentiable torch forward kinematics read out of a Drake `MultibodyPlant`.

A particle optimizer wants the program's pose and grasp rows for N configurations in one
pass. Drake evaluates one configuration per context, so the kinematic tree is extracted
ONCE at construction -- every joint's fixed offsets, axis, pitch and position slot -- and
replayed as batched torch over `[B, ndof]`, in float64, with autograd through the whole
chain. Drake stays the oracle: `tests/test_kinematics_from_plant.py` compares every body
of every fielded scene against `plant.CalcRelativeTransform` at freshly drawn
configurations, so a scene change, a Drake pin move or a parser regression is caught
there rather than in a solver that quietly optimises the wrong row.

POSES ARE (QUAT wxyz, POSITION) THROUGHOUT, never rotation matrices converted back at
the end -- `src/screw_arm/kinematics.py` says why: axis-angle to quaternion is entire,
and matrix-to-quaternion is the operation that needs guarded branches, so composing
quaternions from the start means it never runs on the chain. The helpers are IMPORTED
from there, not copied, so there is one quaternion algebra in the tree.

TWO DRAKE FACTS THIS FILE IS BUILT AROUND.

  1. JOINT INDICES ARE NOT TOPOLOGICAL. On the Panda scene the finger welds (joints 8, 9)
     are registered before the hand weld that is their parent (joint 10), and the
     world weld of the base is joint 11. Replaying joints in index order would compose a
     finger onto a hand whose pose has not been computed. So the tree is walked from the
     world body, and each joint's coordinate comes from its own `position_start()`
     rather than from its index -- the screw-arm lesson
     (`tests/test_screw_arm_kinematics.py:_position_order`; on the soft arm, welding
     the gripper moved a link from position slot 168 to slot 0 with no error).

  2. pydrake BINDS `ScrewJoint.screw_pitch()` BUT NOT `screw_axis()`. The axis is
     recovered from the plant itself: with the joint alone displaced by one radian,
     `CalcRelativeTransform(F, M)` is a rotation by exactly 1 rad about the axis, which
     `ToAngleAxis()` returns, and its translation must equal `axis * pitch / 2pi` --
     both are checked at extraction, so a sign convention this file has wrong fails
     here rather than in a row.

Every tensor is created with an explicit `dtype=` and `device=`: `jrl.config` sets
torch's global default dtype to float32 and default device to cuda AT IMPORT, so a bare
`torch.tensor(...)` in the benchmark process silently downcasts a float64 chain and
leaves a 1e-8 noise floor in what is supposed to be a 1e-13 agreement.

Robots whose configuration is not the plant's position vector (the soft PCS arm, whose
sub-links are floating bodies with no joint) do not go through `KinematicTree`: they
implement `BodyPoseProvider` with their own batched map, and `from_plant` raises on them
by name rather than producing a tree that is missing bodies.
"""

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple  # noqa: F401 (Sequence: Protocol attr)

import numpy as np
import torch

from pydrake.multibody.tree import (BodyIndex, PrismaticJoint, RevoluteJoint,
                                    ScrewJoint, WeldJoint)

from src.screw_arm.kinematics import quat_multiply, quat_rotate, quat_to_matrix

TWO_PI = 2.0 * math.pi
_CPU = "cpu"

Quat4 = Tuple[float, float, float, float]
Pos3 = Tuple[float, float, float]
Pose = Tuple[Quat4, Pos3]


## -- the extracted tree ----------------------------------------------------------------


@dataclass(frozen=True)
class JointRecord:
    """One joint as the batched chain replays it: `X_PC(q) = X_PF * X_FM(q) * X_MC`.

    `X_PF` is the parent-body-to-F offset and `X_MC` the M-to-child-body offset, both
    fixed; `X_FM(q)` is the moving part -- rotation about `axis` by `q` (revolute),
    translation `axis * q` (prismatic), both at once with `axis * pitch * q / 2pi`
    (screw), or the identity (weld). For a WELD the joint's own fixed `X_FM` is folded
    into `X_PF`, so the replay is one rule for every kind. Offsets are plain float
    tuples: the record is frozen and device-free, and `BatchedFK` builds the tensors on
    whatever (dtype, device) it is asked for.
    """
    name: str
    kind: str                       # "revolute" | "weld" | "screw" | "prismatic"
    parent: int                     # BodyIndex of the parent body
    child: int                      # BodyIndex of the child body
    position_slot: Optional[int]    # index into the plant's q; None for a weld
    X_PF: Pose
    X_MC: Pose
    axis: Optional[Pos3]            # unit axis expressed in F; None for a weld
    pitch: float                    # metres per REVOLUTION (Drake's convention); 0 unless screw


_SUPPORTED = {RevoluteJoint: "revolute", WeldJoint: "weld", ScrewJoint: "screw",
              PrismaticJoint: "prismatic"}


def _pose_tuple(X) -> Pose:
    """A `RigidTransform` as `((w, x, y, z), (x, y, z))` of Python floats."""
    q = X.rotation().ToQuaternion().wxyz()
    p = X.translation()
    return (tuple(float(v) for v in q), tuple(float(v) for v in p))


def _screw_axis_from_plant(plant, joint):
    """Recover a screw joint's axis (expressed in F) from the plant, since pydrake does
    not bind `ScrewJoint.screw_axis()`.

    Displace this joint alone by 1 rad and read `X_FM`: a rotation of exactly one radian
    about the axis plus a translation of `pitch / 2pi` along it. One radian rather than
    pi, where `ToAngleAxis` is sign-ambiguous, and rather than a small angle, where the
    axis would be read off a near-identity matrix. Both the angle and the translation are
    asserted, so a wrong sign in this file's understanding of the joint fails here.
    """
    context = plant.CreateDefaultContext()
    q = np.array(plant.GetPositions(context), dtype=float)
    q[joint.position_start()] = 1.0
    plant.SetPositions(context, q)
    X_FM = plant.CalcRelativeTransform(context, joint.frame_on_parent(), joint.frame_on_child())
    aa = X_FM.rotation().ToAngleAxis()
    if abs(aa.angle() - 1.0) > 1e-9:
        raise RuntimeError(
            f"screw joint {joint.name()!r}: displacing it by 1 rad rotated F->M by "
            f"{aa.angle()} rad; the joint is not a rotation by q about a fixed axis.")
    axis = np.asarray(aa.axis(), dtype=float)
    expected = axis * joint.screw_pitch() / TWO_PI
    if np.linalg.norm(np.asarray(X_FM.translation()) - expected) > 1e-12:
        raise RuntimeError(
            f"screw joint {joint.name()!r}: translation at q = 1 is {X_FM.translation()}, "
            f"expected axis * pitch / 2pi = {expected}. Pitch units or axis sign differ from "
            f"what this file assumes (Drake: metres per revolution, along the rotation axis).")
    return axis


@dataclass(frozen=True)
class KinematicTree:
    """A plant's joint graph in an order the batched replay can follow.

    `joints` is topological (parents before children, found by walking from the world
    body), `body_names[i]` is the name of `BodyIndex(i)` -- NOT unique on the hardened
    scenes, whose four shelf units all carry a `shelves_body`, so bodies are addressed by
    index -- and `anchored` holds every body index welded (transitively) to the world,
    world included, whose pose is a constant the replay need not recompute per batch.
    """
    joints: Tuple[JointRecord, ...]
    num_bodies: int
    num_positions: int
    body_names: Tuple[str, ...]
    anchored: frozenset

    @staticmethod
    def from_plant(plant) -> "KinematicTree":
        """Extract the tree from a finalized plant. Raises on anything that is not a
        single tree of revolute / weld / screw / prismatic joints over every body."""
        if not plant.is_finalized():
            raise RuntimeError("KinematicTree.from_plant needs a finalized plant.")
        nb = plant.num_bodies()
        body_names = tuple(plant.get_body(BodyIndex(i)).name() for i in range(nb))

        ## Index the joints by parent body. Drake's joint indices are declaration order,
        ## which is not topological (see the module docstring), so this map is what the
        ## walk below reads, never `GetJointIndices()` in sequence.
        by_parent = {}
        incoming = {}
        for ji in plant.GetJointIndices():
            joint = plant.get_joint(ji)
            kind = None
            for cls, name in _SUPPORTED.items():
                if isinstance(joint, cls):
                    kind = name
            if kind is None:
                raise NotImplementedError(
                    f"joint {joint.name()!r} is a {type(joint).__name__}; this extractor "
                    f"replays revolute, weld, screw and prismatic joints only. A robot whose "
                    f"configuration is not the plant's q (the soft PCS arm's floating "
                    f"sub-links) provides its own BodyPoseProvider instead.")
            child = int(joint.child_body().index())
            if child in incoming:
                raise RuntimeError(
                    f"body {body_names[child]!r} is the child of two joints "
                    f"({incoming[child].name()!r} and {joint.name()!r}); not a tree.")
            incoming[child] = joint
            by_parent.setdefault(int(joint.parent_body().index()), []).append(joint)

        ## Walk from the world body. Each joint is recorded when its parent's pose is
        ## already available, which is the only order the replay can run in.
        records = []
        seen = {0}
        frontier = [0]
        while frontier:
            parent = frontier.pop(0)
            for joint in by_parent.get(parent, []):
                records.append(KinematicTree._record(plant, joint, body_names))
                child = int(joint.child_body().index())
                seen.add(child)
                frontier.append(child)
        missing = sorted(set(range(nb)) - seen)
        if missing:
            raise RuntimeError(
                f"bodies {[body_names[i] for i in missing]} are reached by no joint from the "
                f"world: they are floating (or the tree has a cycle). This robot's "
                f"configuration is not the plant's position vector, so it provides its own "
                f"BodyPoseProvider rather than going through KinematicTree.")

        ## Every plant position must be owned by exactly one recorded joint, or a row of
        ## `q` would be silently ignored.
        slots = sorted(r.position_slot for r in records if r.position_slot is not None)
        if slots != list(range(plant.num_positions())):
            raise RuntimeError(
                f"the recorded joints own position slots {slots} but the plant has "
                f"{plant.num_positions()} positions.")

        anchored = frozenset(int(b.index()) for b in plant.GetBodiesWeldedTo(plant.world_body()))
        return KinematicTree(joints=tuple(records), num_bodies=nb,
                             num_positions=plant.num_positions(), body_names=body_names,
                             anchored=anchored | {0})

    @staticmethod
    def _record(plant, joint, body_names) -> JointRecord:
        X_PF = joint.frame_on_parent().GetFixedPoseInBodyFrame()
        X_CM = joint.frame_on_child().GetFixedPoseInBodyFrame()
        X_MC = X_CM.inverse()
        axis, pitch, slot = None, 0.0, None
        if isinstance(joint, WeldJoint):
            ## Fold the weld's fixed F->M into the parent offset so the replay's
            ## `X_PF * X_FM(q) * X_MC` holds with X_FM(q) = I.
            X_PF = X_PF @ joint.X_FM()
        else:
            if joint.num_positions() != 1:
                raise RuntimeError(f"joint {joint.name()!r} has {joint.num_positions()} positions")
            slot = int(joint.position_start())
            if isinstance(joint, RevoluteJoint):
                axis = np.asarray(joint.revolute_axis(), dtype=float)
            elif isinstance(joint, PrismaticJoint):
                axis = np.asarray(joint.translation_axis(), dtype=float)
            elif isinstance(joint, ScrewJoint):
                axis = _screw_axis_from_plant(plant, joint)
                pitch = float(joint.screw_pitch())
            axis = tuple(float(v) for v in axis / np.linalg.norm(axis))
        kind = next(name for cls, name in _SUPPORTED.items() if isinstance(joint, cls))
        return JointRecord(name=joint.name(), kind=kind,
                           parent=int(joint.parent_body().index()),
                           child=int(joint.child_body().index()),
                           position_slot=slot, X_PF=_pose_tuple(X_PF), X_MC=_pose_tuple(X_MC),
                           axis=axis, pitch=pitch)

    def frame_offset(self, frame):
        """`(body_index, quat4, pos3)` of a Drake `Frame`: the body it rides on and its
        fixed pose in that body. A body frame returns the identity offset."""
        quat, pos = _pose_tuple(frame.GetFixedPoseInBodyFrame())
        return int(frame.body().index()), quat, pos

## -- the interface a robot plugs in behind ---------------------------------------------


class BodyPoseProvider(Protocol):
    """World poses of every plant body for a batch of configurations.

    `body_names[i]` names `BodyIndex(i)`; `body_poses(cfg)` returns `(quat [B, nb, 4]
    wxyz, pos [B, nb, 3])`, world at index 0 as the identity, differentiable in `cfg`.
    `BatchedFK` implements it for any robot whose configuration is the plant's `q`; the
    soft PCS arm implements it over its strain coordinates.
    """
    body_names: Sequence[str]

    def body_poses(self, cfg: torch.Tensor):
        ...


## -- the batched replay ---------------------------------------------------------------


class BatchedFK:
    """Replay a `KinematicTree` over `q [B, ndof]` in torch.

    Constants are stacked once onto `(dtype, device)` at construction; the only per-call
    work is the chain itself. Anchored bodies (welded to the world) have constant poses,
    computed once here and broadcast per batch, so the scene's shelves and tables cost
    nothing per evaluation -- on GPU each joint is ~10 kernel launches, and the hardened
    Panda scene has more furniture welds than arm joints.
    """

    def __init__(self, tree: KinematicTree, dtype=torch.float64, device=_CPU):
        self.tree = tree
        self.dtype = dtype
        self.device = torch.device(device)
        self.body_names = tree.body_names
        self.num_positions = tree.num_positions
        self.num_bodies = tree.num_bodies

        def t(values, n):
            return torch.tensor(values, dtype=dtype, device=self.device).reshape(n)

        self._joints = []
        for r in tree.joints:
            self._joints.append(dict(
                record=r,
                qPF=t(r.X_PF[0], 4), pPF=t(r.X_PF[1], 3),
                qMC=t(r.X_MC[0], 4), pMC=t(r.X_MC[1], 3),
                axis=None if r.axis is None else t(r.axis, 3),
                pitch_over_2pi=r.pitch / TWO_PI,
            ))
        self._identity_q = t((1.0, 0.0, 0.0, 0.0), 4)
        self._zero_p = t((0.0, 0.0, 0.0), 3)

        ## Anchored poses: run the chain once at batch 1. No joint on an anchored body's
        ## path moves, so the result is independent of q and is cached as constants.
        q0 = torch.zeros(1, tree.num_positions, dtype=dtype, device=self.device)
        quat0, pos0 = self._chain(q0, skip_anchored=False)
        self._anchored_quat = {b: quat0[0, b].detach().clone() for b in tree.anchored}
        self._anchored_pos = {b: pos0[0, b].detach().clone() for b in tree.anchored}

    ## -- the chain --

    def _joint_motion(self, j, q, B):
        """`(quat [B, 4], trans [B, 3])` of X_FM(q) for one joint and its coordinate
        `q [B]` (None for a weld)."""
        r = j["record"]
        if r.kind == "weld":
            return (self._identity_q.expand(B, 4), self._zero_p.expand(B, 3))
        axis = j["axis"]
        if r.kind == "prismatic":
            return (self._identity_q.expand(B, 4), q.unsqueeze(-1) * axis)
        ## Axis-angle to quaternion is entire -- no branch, no clamped divisor.
        half = 0.5 * q
        quat = torch.cat((torch.cos(half).unsqueeze(-1), torch.sin(half).unsqueeze(-1) * axis),
                         dim=-1)
        if r.kind == "screw":
            ## Rotation about an axis fixes that axis, so the rotation and the axial
            ## translation commute and a screw is a revolute with the translation filled in.
            return quat, q.unsqueeze(-1) * (j["pitch_over_2pi"] * axis)
        return quat, self._zero_p.expand(B, 3)

    def _chain(self, q, skip_anchored=True):
        B = q.shape[0]
        quats = [None] * self.num_bodies
        poss = [None] * self.num_bodies
        quats[0] = self._identity_q.expand(B, 4)
        poss[0] = self._zero_p.expand(B, 3)
        for j in self._joints:
            r = j["record"]
            if skip_anchored and r.child in self._anchored_quat:
                quats[r.child] = self._anchored_quat[r.child].expand(B, 4)
                poss[r.child] = self._anchored_pos[r.child].expand(B, 3)
                continue
            qp, pp = quats[r.parent], poss[r.parent]
            ## parent body -> F (fixed) ...
            p = pp + quat_rotate(qp, j["pPF"].expand(B, 3))
            Q = quat_multiply(qp, j["qPF"].expand(B, 4))
            ## ... F -> M (the joint's motion) ...
            coord = None if r.position_slot is None else q[:, r.position_slot]
            step_q, step_p = self._joint_motion(j, coord, B)
            p = p + quat_rotate(Q, step_p)
            Q = quat_multiply(Q, step_q)
            ## ... M -> child body (fixed).
            p = p + quat_rotate(Q, j["pMC"].expand(B, 3))
            Q = quat_multiply(Q, j["qMC"].expand(B, 4))
            quats[r.child], poss[r.child] = Q, p
        return torch.stack(quats, dim=1), torch.stack(poss, dim=1)

    def body_poses(self, q):
        """World poses of ALL bodies: `(quat [B, nb, 4] wxyz, pos [B, nb, 3])`, in
        `BodyIndex` order with the world at 0 as the identity."""
        q = torch.as_tensor(q, dtype=self.dtype, device=self.device)
        if q.dim() != 2 or q.shape[1] != self.num_positions:
            raise ValueError(
                f"expected q of shape [B, {self.num_positions}], got {tuple(q.shape)}")
        return self._chain(q)

    ## -- frames --

    def frame_pose(self, quat, pos, body_index, X_BF):
        """Compose a fixed body-frame offset `X_BF = (quat4, pos3)` onto body
        `body_index` of a `body_poses` result: `(quat [B, 4], pos [B, 3])`."""
        qB, pB = quat[:, body_index], pos[:, body_index]
        qBF = torch.as_tensor(X_BF[0], dtype=self.dtype, device=self.device)
        pBF = torch.as_tensor(X_BF[1], dtype=self.dtype, device=self.device)
        B = qB.shape[0]
        return (quat_multiply(qB, qBF.expand(B, 4)), pB + quat_rotate(qB, pBF.expand(B, 3)))

    def frame_pose_from_q(self, q, frame):
        """World pose of a Drake `Frame` for a batch of `q`: `(quat [B, 4], pos [B, 3])`."""
        body, qBF, pBF = self.tree.frame_offset(frame)
        quat, pos = self.body_poses(q)
        return self.frame_pose(quat, pos, body, (qBF, pBF))

    def frame_chain(self, frames):
        """A `FrameChain` evaluating ONLY the joints on the paths from the world to the
        given frames (`[(body_index, (quat4, pos3)), ...]`), with every fixed transform
        between two moving joints pre-composed. See `FrameChain`."""
        return FrameChain(self, frames)


class FrameChain:
    """The fast path for a few frames: the moving joints on their root paths and nothing else.

    `BatchedFK.body_poses` replays every joint of the plant and is launch-bound on CUDA
    (~22 joints x ~10 small kernels, flat in the batch size), while the rows of a program
    need two frames on the same arm. Here the path from the world to each requested frame
    is walked once at construction: anchored ancestors are constants, a weld is a fixed
    transform, and every run of fixed transforms between two MOVING joints -- `X_MC` of
    one, the welds between, `X_PF` of the next -- is composed once into a single offset.
    The chain state is the pose of each moving joint's M frame; a requested frame is one
    fixed compose from the M frame of its nearest moving ancestor (or a constant when no
    joint on its path moves). Exact: only the association of constant products changes.

    `poses(q) -> [(quat [B, 4], pos [B, 3]), ...]` in the order the frames were given.
    """

    def __init__(self, fk: BatchedFK, frames):
        self.fk = fk
        tree = fk.tree
        joint_of_child = {r.child: j for j, r in zip(fk._joints, tree.joints)}
        ident = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        def const_pose(b):
            return (tuple(float(v) for v in fk._anchored_quat[b].detach().cpu().tolist()),
                    tuple(float(v) for v in fk._anchored_pos[b].detach().cpu().tolist()))

        ## resolve(body) -> (step index of the nearest moving ancestor or None, fixed offset
        ## from that step's M frame -- or from the world -- to the body).
        self._steps = []            # dicts: joint, parent (step index or None), pre (pose) or anchor
        memo = {}

        def resolve(body):
            if body in memo:
                return memo[body]
            if body == 0 or body in tree.anchored:
                out = (None, const_pose(body))
            else:
                j = joint_of_child[body]
                r = j["record"]
                parent_step, off = resolve(r.parent)
                if r.kind == "weld":
                    out = (parent_step, _compose(_compose(off, r.X_PF), r.X_MC))
                else:
                    pre = _compose(off, r.X_PF)
                    self._steps.append(dict(joint=j, parent=parent_step,
                                            pre=None if parent_step is None else pre,
                                            anchor=pre if parent_step is None else None))
                    out = (len(self._steps) - 1, r.X_MC)
            memo[body] = out
            return out

        self._frames = []
        for body, X_BF in frames:
            step, off = resolve(int(body))
            self._frames.append((step, _compose(off, (tuple(X_BF[0]), tuple(X_BF[1])))))

        dtype, device = fk.dtype, fk.device

        def t(v, n):
            return torch.tensor(v, dtype=dtype, device=device).reshape(n)
        for s in self._steps:
            for key in ("pre", "anchor"):
                if s[key] is not None:
                    s[key + "_q"], s[key + "_p"] = t(s[key][0], 4), t(s[key][1], 3)
        self._frame_consts = [(t(off[0], 4), t(off[1], 3)) for _, off in self._frames]
        self.num_moving = len(self._steps)
        self.num_positions = fk.num_positions

        ## Geometric-Jacobian bookkeeping per frame: its ancestor steps (root first), their
        ## position slots, and per ancestor the coefficients of the two column formulas
        ##     dp/dq_j = rot_j * (w_j x (p_frame - o_j)) + lin_j * w_j,   dw/dq_j = rot_j * w_j
        ## with w_j the joint axis in the world and o_j a point on it (the F origin):
        ## revolute (rot 1, lin 0), prismatic (rot 0, lin 1), screw (rot 1, lin pitch/2pi).
        self._frame_anc = []
        for step, _ in self._frames:
            anc = []
            while step is not None:
                anc.append(step)
                step = self._steps[step]["parent"]
            anc.reverse()
            kinds = [self._steps[a]["joint"]["record"] for a in anc]
            self._frame_anc.append(dict(
                steps=anc,
                slots=torch.tensor([r.position_slot for r in kinds], dtype=torch.long, device=device),
                rot=t([0.0 if r.kind == "prismatic" else 1.0 for r in kinds], (1, len(anc), 1)),
                lin=t([1.0 if r.kind == "prismatic" else (r.pitch / TWO_PI if r.kind == "screw" else 0.0)
                       for r in kinds], (1, len(anc), 1))))

    def poses(self, q, jacobians=False):
        """`[(quat [B, 4], pos [B, 3]), ...]` per frame; with `jacobians=True` each entry
        is `(quat, pos, Jp [B, 3, nq], Jw [B, 3, nq])` -- the geometric Jacobian of the
        frame's origin and of its angular velocity (world frame, `d omega / d qdot`) with
        respect to the plant's positions, by the column formulas above. Plain forward
        tensor ops, so a solver gets the rows' derivative without a backward through the
        chain; `tests/test_svgd_solver.py` pins it against autograd of these very poses."""
        fk = self.fk
        q = torch.as_tensor(q, dtype=fk.dtype, device=fk.device)
        if q.dim() != 2 or q.shape[1] != self.num_positions:
            raise ValueError(f"expected q of shape [B, {self.num_positions}], got {tuple(q.shape)}")
        B = q.shape[0]
        M, W, O = [], [], []
        for s in self._steps:
            j = s["joint"]
            r = j["record"]
            if s["parent"] is None:
                Q, p = s["anchor_q"].expand(B, 4), s["anchor_p"].expand(B, 3)
            else:
                qp, pp = M[s["parent"]]
                p = pp + quat_rotate(qp, s["pre_p"].expand(B, 3))
                Q = quat_multiply(qp, s["pre_q"].expand(B, 4))
            if jacobians:
                W.append(quat_rotate(Q, j["axis"].expand(B, 3)))      # axis in the world
                O.append(p)                                           # a point on it
            step_q, step_p = fk._joint_motion(j, q[:, r.position_slot], B)
            p = p + quat_rotate(Q, step_p)
            Q = quat_multiply(Q, step_q)
            M.append((Q, p))
        out = []
        for k, ((step, _), (oq, op)) in enumerate(zip(self._frames, self._frame_consts)):
            if step is None:
                quat, pos = oq.expand(B, 4), op.expand(B, 3)
            else:
                Q, p = M[step]
                quat, pos = quat_multiply(Q, oq.expand(B, 4)), p + quat_rotate(Q, op.expand(B, 3))
            if not jacobians:
                out.append((quat, pos))
                continue
            Jp = torch.zeros(B, 3, self.num_positions, dtype=fk.dtype, device=fk.device)
            Jw = torch.zeros_like(Jp)
            anc = self._frame_anc[k]
            if anc["steps"]:
                Wk = torch.stack([W[a] for a in anc["steps"]], dim=1)     # [B, K, 3]
                Ok = torch.stack([O[a] for a in anc["steps"]], dim=1)
                lever = torch.cross(Wk, pos.unsqueeze(1) - Ok, dim=-1)
                cols_p = anc["rot"] * lever + anc["lin"] * Wk
                cols_w = anc["rot"] * Wk
                Jp[:, :, anc["slots"]] = cols_p.transpose(1, 2)
                Jw[:, :, anc["slots"]] = cols_w.transpose(1, 2)
            out.append((quat, pos, Jp, Jw))
        return out


def _compose(a: Pose, b: Pose) -> Pose:
    """`X_a * X_b` on `(quat4, pos3)` tuples, with the chain's own quaternion algebra on
    CPU float64 so a pre-composed offset equals what the full replay would have formed."""
    qa = torch.tensor(a[0], dtype=torch.float64, device=_CPU).reshape(1, 4)
    pa = torch.tensor(a[1], dtype=torch.float64, device=_CPU).reshape(1, 3)
    qb = torch.tensor(b[0], dtype=torch.float64, device=_CPU).reshape(1, 4)
    pb = torch.tensor(b[1], dtype=torch.float64, device=_CPU).reshape(1, 3)
    q = quat_multiply(qa, qb)[0]
    p = (pa + quat_rotate(qa, pb))[0]
    return (tuple(float(v) for v in q.tolist()), tuple(float(v) for v in p.tolist()))


## -- pose algebra the rows need --------------------------------------------------------


def canonical_quat(q):
    """`q * sign(w)` so `w >= 0`, as Drake's `RotationMatrix.ToQuaternion()` returns.

    The sign is a piecewise constant and is detached, so the result's derivative is the
    sign times `q`'s -- which is what `ToQuaternion` on an AutoDiffXd matrix gives too.
    `w == 0` keeps `+1`, matching Drake's `if (w < 0) negate`.
    """
    w = q[..., 0:1]
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w)).detach()
    return q * sign


def pose7(quat, pos):
    """`[B, 7]` as `[x, y, z, qw, qx, qy, qz]` with the quaternion canonical -- jrl's
    and ikflow's layout, and `forward_kinematics`'s in the robot shims."""
    return torch.cat((pos, canonical_quat(quat)), dim=-1)


def wrap_residual(r):
    """Wrap to (-pi, pi] by subtracting a constant multiple of 2pi: exactly the loop in
    `generic_program.orientation_error_rpy` (`r - 2pi * round(r / 2pi)`, round half to
    even in both numpy and torch), so a residual the solver sees and one the swarm sees
    agree bitwise. A constant shift, so the derivative is untouched."""
    return r - TWO_PI * torch.round(r / TWO_PI)


def quat_from_rpy_batched(rpy):
    """`[..., 3]` roll-pitch-yaw (Drake's SpaceXYZ convention) -> `[..., 4]` wxyz,
    differentiable; the batched twin of `screw_arm.kinematics.quat_from_rpy`."""
    half = 0.5 * rpy
    cr, sr = torch.cos(half[..., 0]), torch.sin(half[..., 0])
    cp, sp = torch.cos(half[..., 1]), torch.sin(half[..., 1])
    cy, sy = torch.cos(half[..., 2]), torch.sin(half[..., 2])
    return torch.stack((cr * cp * cy + sr * sp * sy,
                        sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy), dim=-1)


def rpy_from_quat(q):
    """Roll-pitch-yaw of a unit quaternion `[..., 4]` wxyz, by DRAKE'S algorithm.

    This reproduces `RollPitchYaw(RotationMatrix(Quaternion(q)))` -- the call
    `orientation_error_rpy` makes -- step for step
    (`CalcRollPitchYawFromQuaternionAndRotationMatrix`, drake/math/roll_pitch_yaw.cc):
    pitch from the matrix as `atan2(-R20, sqrt((R22^2 + R21^2 + R10^2 + R00^2) / 2))`,
    then roll and yaw as the difference and sum of two half-angle `atan2`s on the
    quaternion's components, each wrapped once into [-pi, pi]. Using the same algorithm
    rather than any equivalent one is the point: the two agree to ~1e-14 away from
    gimbal lock and degrade in the same way near pitch = +-pi/2, so a row the swarm
    evaluates and the row Drake evaluates are the same function.

    Two things Drake does implicitly are done explicitly here. `RollPitchYaw(R)` takes
    its quaternion from `R.ToQuaternion()`, which has `w >= 0`, so the input is
    canonicalised first (a sign flip shifts `zA` and `zB` each by pi, which the single
    wrap step absorbs -- but only for one side of the boundary). And Drake's singular
    guards (`|yA| <= eps and |xA| <= eps` -> 0) are applied by substituting safe
    arguments BEFORE `atan2`, not by `torch.where` on its result: `atan2`'s backward at
    the origin is `0/0`, and `where` multiplies the untaken branch's gradient by zero,
    which leaves nan rather than removing it.
    """
    q = canonical_quat(q)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    R = quat_to_matrix(q)
    R00, R10 = R[..., 0, 0], R[..., 1, 0]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    Rsum = torch.sqrt((R22 * R22 + R21 * R21 + R10 * R10 + R00 * R00) / 2)
    pitch = torch.atan2(-R20, Rsum)

    e0, e1, e2, e3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yA, xA = e1 + e3, e0 - e2
    yB, xB = e3 - e1, e0 + e2
    eps = torch.finfo(q.dtype).eps
    zA = _guarded_atan2(yA, xA, eps)
    zB = _guarded_atan2(yB, xB, eps)
    roll = _wrap_once(zA - zB)
    yaw = _wrap_once(zA + zB)
    return torch.stack((roll, pitch, yaw), dim=-1)


def _guarded_atan2(y, x, eps):
    singular = (y.abs() <= eps) & (x.abs() <= eps)
    y_safe = torch.where(singular, torch.zeros_like(y), y)
    x_safe = torch.where(singular, torch.ones_like(x), x)
    return torch.atan2(y_safe, x_safe)


def _wrap_once(a):
    """Drake's `if (a > pi) a -= 2pi; if (a < -pi) a += 2pi` -- one step each way, not a
    full `round`, so values are wrapped exactly as Drake wraps them."""
    a = torch.where(a > math.pi, a - TWO_PI, a)
    return torch.where(a < -math.pi, a + TWO_PI, a)
