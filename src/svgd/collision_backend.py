"""The program's collision row, evaluated EXACTLY by Drake, for a batch of configurations.

No geometric proxy anywhere (Thomas, 2026-10-08: "I don't like the proxy idea at all"). The row
a particle optimizer sees is the one `IKFlowProgram.CreateCollisionFreeConstraint` hands IPOPT:

    row(q) = collision_row_scale * MinimumDistanceLowerBoundConstraint(
                 plant, bound=collision_bound,
                 influence_distance_offset=collision_influence_offset).Eval(q)

with upper bound `collision_row_scale`, so `verify()` and the optimizer see the same functional
and there is no proxy-versus-Drake disagreement to adjudicate. `tests/test_collision_backend.py`
pins the batched value AND gradient bitwise against the program's own binding.

**In process, no process pool (Thomas, 2026-10-09).** `CollisionEvaluator` is a Python loop
over the N particles calling the PROGRAM'S OWN constraint
(`program.collision_free_constraint_eval`, on the program's own plant context) on AutoDiffXd --
`Eval(InitializeAutoDiff(q))`, exactly what the program's binding does -- and returns the value
`[N]` and gradient `[N, nq]`. It holds the GIL for the whole batch (pydrake does not release it in
`Eval`), so it is serial and costs roughly N x the per-configuration Eval. A batched C++
clearance-with-Jacobians call that releases the GIL is a Drake-side item for Thomas, not patched
here. (`from_spec` builds the same constraint on a rebuilt scene, for callers holding no program.)

**A non-finite configuration must never reach Drake.** `MultibodyPlant::SetPositions` has a
`DRAKE_DEMAND(AllFinite(q))`; in pydrake that failure surfaces as `SystemExit` (a BaseException,
not an Exception). Rows are screened for finiteness before any Drake call and come back as NaN
value and NaN gradient; the per-row `except BaseException` behind that screen is belt-and-braces
for anything else Drake aborts on.

**`SceneSpec`** is a picklable description of the scene (directives YAML plus, for the grasp
task, the welded target mug). `GenerateDiagramWithMug` welds the mug through
`pydrake.common.schema.Transform(RigidTransform)`, which stores the rotation as RPY in degrees; a
quaternion round-trip of that pose is NOT bit-exact, while `RigidTransform(RotationMatrix(R), p)`
reproduces the original exactly. The spec therefore carries the 3x3 rotation for reconstruction
and the quaternion only for a human reader.

**`ParallelCollisionChecker` is the boolean, C++-parallel path** (in-process Drake threads) for
where a boolean suffices (collision-free draws, redraws): `SceneGraphCollisionChecker
.CheckConfigsCollisionFree` does 1024 configurations in ~16-33 ms. Its padding defaults to
`collision_bound`, so "in collision" means the same thing as the row's `> scale`.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.utils import BuildEnv, HiddenPrints, RepoDir

MUG_URDF = "package://combining_kinematics/models/mug/mug_simple_red.urdf"
MUG_MODEL_NAME = "target_mug"
MUG_BODY_NAME = "mug_body_link"


## ------------------------------------------------------------------------------------ ##
##                                       SceneSpec                                       ##
## ------------------------------------------------------------------------------------ ##

@dataclass(frozen=True)
class SceneSpec:
    """A picklable description from which a worker rebuilds the solve scene exactly.

    `mug_rotation` (row-major 3x3) is what reconstructs the weld bit-for-bit; `mug_wxyz` is the
    same rotation for a reader and is NOT used to rebuild anything (see the module docstring).
    """
    yaml_path: str
    mug_xyz: tuple = None
    mug_wxyz: tuple = None
    mug_rotation: tuple = None
    mug_urdf: str = MUG_URDF
    mug_model_name: str = MUG_MODEL_NAME
    mug_body_name: str = MUG_BODY_NAME
    collision_bound: float = 1e-3
    influence_distance_offset: float = 0.1
    row_scale: float = 0.1

    @staticmethod
    def from_program(program, yaml_path=None):
        """Read the spec off a constructed program.

        The program does not record the YAML it was built from -- the benchmark scripts build
        the diagram themselves and hand it to the constructor -- so it is taken, in order, from
        the `yaml_path` argument, a `scene_yaml` attribute on the program, or one on its diagram
        (pydrake `Diagram`s accept dynamic attributes, so `BuildEnv`'s caller can tag it).

        The mug comes from `program.target_mug.middle`, which IS the `RigidTransform` the weld
        was formed from, so the rebuilt weld is bit-identical. If the plant carries a
        `target_mug` instance but the program has no `target_mug` attribute, the weld's own
        `X_FM` is read back instead -- that pose has already been through the schema's RPY
        degrees once, so a second pass is exact to ~1e-16 rather than bitwise. A `target_mug`
        attribute on a program whose plant has no such instance is an inconsistent scene and
        raises.
        """
        yaml = yaml_path
        if yaml is None:
            yaml = getattr(program, "scene_yaml", None)
        if yaml is None:
            yaml = getattr(program.diagram, "scene_yaml", None)
        if yaml is None:
            raise ValueError(
                "SceneSpec.from_program: the program records no scene YAML. Pass yaml_path=, "
                "or set `program.scene_yaml` / `program.diagram.scene_yaml` where the diagram "
                "is built (the benchmark scripts know the path; the program never did).")
        yaml = os.path.abspath(yaml)
        if not os.path.exists(yaml):
            raise FileNotFoundError(yaml)

        plant = program.plant
        has_instance = plant.HasModelInstanceNamed(MUG_MODEL_NAME)
        mug = getattr(program, "target_mug", None)
        X_mug = None
        if mug is not None:
            if not has_instance:
                raise ValueError(
                    "SceneSpec.from_program: program has a `target_mug` but its plant has no "
                    "`%s` model instance -- the mug would be added to a scene the program does "
                    "not have." % MUG_MODEL_NAME)
            X_mug = mug.middle
        elif has_instance:
            from pydrake.multibody.tree import WeldJoint
            instance = plant.GetModelInstanceByName(MUG_MODEL_NAME)
            welds = [plant.get_joint(j) for j in plant.GetJointIndices(instance)]
            welds = [j for j in welds if isinstance(j, WeldJoint)
                     and j.parent_body().index() == plant.world_body().index()]
            if len(welds) != 1:
                raise ValueError("SceneSpec.from_program: expected exactly one world weld on "
                                 "`%s`, found %d" % (MUG_MODEL_NAME, len(welds)))
            X_mug = welds[0].X_FM()

        opts = program.options
        return SceneSpec(
            yaml_path=yaml,
            mug_xyz=None if X_mug is None else tuple(float(v) for v in X_mug.translation()),
            mug_wxyz=None if X_mug is None else tuple(
                float(v) for v in X_mug.rotation().ToQuaternion().wxyz()),
            mug_rotation=None if X_mug is None else tuple(
                float(v) for v in X_mug.rotation().matrix().reshape(-1)),
            collision_bound=float(opts.collision_bound),
            influence_distance_offset=float(opts.collision_influence_offset),
            row_scale=float(opts.collision_row_scale),
        )

    @property
    def has_mug(self):
        return self.mug_rotation is not None

    def mug_pose(self):
        """The mug's world pose as a `RigidTransform`, or None on a pose-task scene."""
        if not self.has_mug:
            return None
        from pydrake.math import RigidTransform, RotationMatrix
        R = np.array(self.mug_rotation, dtype=np.float64).reshape(3, 3)
        return RigidTransform(RotationMatrix(R), np.array(self.mug_xyz, dtype=np.float64))

    def extra_directives(self):
        """The in-memory `add_model`/`add_weld` pair `GenerateDiagramWithMug` appends."""
        if not self.has_mug:
            return []
        from pydrake.common.schema import Transform as SchemaTransform
        from pydrake.multibody.parsing import AddModel, AddWeld, ModelDirective
        add_mug = ModelDirective()
        add_mug.add_model = AddModel(name=self.mug_model_name, file=self.mug_urdf)
        weld_mug = ModelDirective()
        weld_mug.add_weld = AddWeld(
            parent="world",
            child="%s::%s" % (self.mug_model_name, self.mug_body_name),
            X_PC=SchemaTransform(self.mug_pose()),
        )
        return [add_mug, weld_mug]

    def build_diagram(self):
        """The solve scene, headless, exactly as the program's diagram was built."""
        with HiddenPrints():
            return BuildEnv(meshcat=None, directives_file=self.yaml_path,
                            extra_directives=self.extra_directives() or None)


## ------------------------------------------------------------------------------------ ##
##                         CollisionEvaluator (in process, exact)                        ##
## ------------------------------------------------------------------------------------ ##

class CollisionEvaluator:
    """The scaled collision row and its gradient for a batch, by a per-particle loop over a
    `MinimumDistanceLowerBoundConstraint` in THIS process (module docstring).

    `eval(Q)` returns `scale * y` and `scale * dy/dq` for every row of `Q` (`[N, nq]`, the FULL
    plant vector); rows that are non-finite, or on which Drake aborts, come back NaN.
    """

    def __init__(self, constraint, nq, scale):
        self.constraint = constraint
        self.nq = int(nq)
        self.scale = float(scale)

    @staticmethod
    def from_program(program):
        """The program's OWN constraint (`collision_free_constraint_eval`, its plant context)
        and row scale."""
        return CollisionEvaluator(program.collision_free_constraint_eval,
                                  program.plant.num_positions(),
                                  program.options.collision_row_scale)

    @staticmethod
    def from_spec(spec):
        """The same constraint on a scene rebuilt from `spec` (a caller with no program)."""
        from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint
        diagram = spec.build_diagram()
        plant = diagram.GetSubsystemByName("plant")
        context = diagram.CreateDefaultContext()
        constraint = MinimumDistanceLowerBoundConstraint(
            plant=plant, bound=spec.collision_bound,
            influence_distance_offset=spec.influence_distance_offset,
            plant_context=plant.GetMyContextFromRoot(context))
        ev = CollisionEvaluator(constraint, plant.num_positions(), spec.row_scale)
        ev._keepalive = (diagram, context)          # the constraint holds raw pointers into these
        return ev

    def eval(self, Q, need_grad=True):
        from pydrake.autodiffutils import ExtractGradient, ExtractValue, InitializeAutoDiff
        Q = np.asarray(Q, dtype=np.float64)
        n = Q.shape[0]
        if Q.ndim != 2 or Q.shape[1] != self.nq:
            raise ValueError("collision row: got shape %r, the scene has %d plant positions"
                             % (Q.shape, self.nq))
        scale = self.scale
        values = np.full(n, np.nan)
        grads = np.full((n, self.nq), np.nan) if need_grad else None
        for i in range(n):
            q = Q[i]
            # A NaN or inf position is a DRAKE_DEMAND abort inside SetPositions, which pydrake
            # surfaces as SystemExit. Never let it reach Drake; the row is NaN by contract.
            if not np.all(np.isfinite(q)):
                continue
            try:
                if need_grad:
                    y = self.constraint.Eval(InitializeAutoDiff(q))
                    values[i] = scale * np.ravel(ExtractValue(y))[0]
                    g = np.ravel(ExtractGradient(y))
                    if g.size == self.nq:
                        grads[i] = scale * g
                    elif g.size == 0:
                        # An empty derivative vector is Drake's "constant": nothing within
                        # the influence distance, so the gradient is identically zero.
                        grads[i] = 0.0
                    else:
                        raise RuntimeError("unexpected gradient size %d" % g.size)
                else:
                    values[i] = scale * np.ravel(self.constraint.Eval(q))[0]
            except KeyboardInterrupt:
                raise
            except BaseException:
                # Leave the row NaN. The hot path must never raise.
                continue
        return values, grads


class CollisionRow(torch.autograd.Function):
    """`forward(q_plant [N, nq], evaluator) -> value [N]`, same dtype and device as the input.

    Drake runs in float64 on the CPU whatever the input is; the value comes back in the input's
    dtype and device and the stored `[N, nq]` gradient drives `backward`
    (`grad_output[:, None] * grad`). NaN rows propagate as NaN."""

    @staticmethod
    def forward(ctx, q_plant, evaluator):
        Q = q_plant.detach().to("cpu", torch.float64).numpy()
        value, grad = evaluator.eval(Q, need_grad=True)
        ctx.save_for_backward(torch.as_tensor(grad).to(device=q_plant.device, dtype=q_plant.dtype))
        return torch.as_tensor(value).to(device=q_plant.device, dtype=q_plant.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad_output.unsqueeze(1) * grad, None


def collision_row(q_plant, evaluator):
    """The scaled collision row for a batch of plant vectors, differentiable in torch."""
    return CollisionRow.apply(q_plant, evaluator)


## ------------------------------------------------------------------------------------ ##
##                      ParallelCollisionChecker (boolean, C++-parallel)                 ##
## ------------------------------------------------------------------------------------ ##

class ParallelCollisionChecker:
    """`SceneGraphCollisionChecker` over the same scene, for where a boolean suffices.

    **Robot model instances are derived, not named**: an instance is part of the robot iff it
    owns a body Drake does not consider anchored to the world, i.e. anything that moves with
    the configuration. The arm's links move; the finray gripper is welded to the last link, so
    it moves too; tables, shelves and every welded mug (decorative or target) are anchored and
    therefore environment; a floating-body continuum arm is all moving bodies. On both rigid
    hardened scenes this reproduces `SCENES[...].robot_instances` exactly, and
    `robot_instance_names` exposes the result so a caller can check.

    `padding` defaults to the spec's `collision_bound`, so `collision_free` is False exactly
    where the program's row would exceed its upper bound (some pair closer than `bound`), up to
    the smoothing of Drake's `SmoothOverMax`. Pass `padding=0.0` for the plain penetration
    test.
    """

    def __init__(self, spec, padding=None, edge_step_size=0.05):
        from pydrake.multibody.parsing import (LoadModelDirectives, ModelDirectives,
                                               ProcessModelDirectives)
        from pydrake.multibody.tree import ModelInstanceIndex
        from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker
        self.spec = spec
        self.padding = spec.collision_bound if padding is None else float(padding)
        with HiddenPrints():
            builder = RobotDiagramBuilder()
            parser = builder.parser()
            parser.package_map().AddPackageXml(os.path.join(RepoDir(), "package.xml"))
            directives = LoadModelDirectives(spec.yaml_path)
            extra = spec.extra_directives()
            if extra:
                combined = ModelDirectives()
                combined.directives = list(directives.directives) + list(extra)
                directives = combined
            ProcessModelDirectives(directives, builder.plant(), parser)
            self.model = builder.Build()
            plant = self.model.plant()
            robot = []
            for i in range(plant.num_model_instances()):
                mi = ModelInstanceIndex(i)
                if any(not plant.IsAnchored(plant.get_body(b)) for b in plant.GetBodyIndices(mi)):
                    robot.append(mi)
            if not robot:
                raise RuntimeError("ParallelCollisionChecker: no moving model instance in %s"
                                   % spec.yaml_path)
            self.robot_instance_names = tuple(plant.GetModelInstanceName(m) for m in robot)
            self.nq = plant.num_positions()
            self.checker = SceneGraphCollisionChecker(
                model=self.model, robot_model_instances=robot,
                edge_step_size=edge_step_size,
                env_collision_padding=self.padding, self_collision_padding=self.padding)

    def collision_free(self, q_plant):
        """bool[N]: True where no checked pair is closer than `padding`. Non-finite rows are
        False (never handed to Drake)."""
        from pydrake.common import Parallelism
        Q = np.ascontiguousarray(np.asarray(q_plant, dtype=np.float64))
        if Q.ndim != 2 or Q.shape[1] != self.nq:
            raise ValueError("q_plant must be [N, %d], got %r" % (self.nq, Q.shape))
        finite = np.all(np.isfinite(Q), axis=1)
        out = np.zeros(Q.shape[0], dtype=bool)
        if finite.any():
            ok = self.checker.CheckConfigsCollisionFree(Q[finite], Parallelism.Max())
            out[finite] = np.asarray(ok, dtype=bool)
        return out


## ------------------------------------------------------------------------------------ ##
##                                        Timing                                         ##
## ------------------------------------------------------------------------------------ ##

def measure_collision(spec, Ns=(64, 256, 1024), repeats=3, seed=0, lower=None, upper=None,
                      checker=True, out=print):
    """Print ms per batch and us per configuration of the in-process row (and the boolean
    checker). Configurations are uniform in `[lower, upper]` (defaults: the scene plant's
    position limits). Each cell is the median of `repeats` after one warm-up batch."""
    rng = np.random.default_rng(seed)
    ev = CollisionEvaluator.from_spec(spec)
    if lower is None or upper is None:
        plant = ev._keepalive[0].GetSubsystemByName("plant")
        lower, upper = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError("measure_collision: plant limits are not finite; pass lower/upper")
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    rows = []
    out("%-10s %8s %12s %12s" % ("backend", "N", "ms/batch", "us/config"))
    ev.eval(rng.uniform(lower, upper, size=(8, lower.size)))
    for N in Ns:
        Q = rng.uniform(lower, upper, size=(N, lower.size))
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            ev.eval(Q)
            times.append(time.perf_counter() - t0)
        ms = 1e3 * float(np.median(times))
        rows.append(dict(backend="row", N=N, ms=ms, us_per_config=1e3 * ms / N))
        out("%-10s %8d %12.1f %12.1f" % ("row (AD)", N, ms, 1e3 * ms / N))
    if checker:
        cc = ParallelCollisionChecker(spec)
        cc.collision_free(rng.uniform(lower, upper, size=(64, lower.size)))
        for N in Ns:
            Q = rng.uniform(lower, upper, size=(N, lower.size))
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                cc.collision_free(Q)
                times.append(time.perf_counter() - t0)
            ms = 1e3 * float(np.median(times))
            rows.append(dict(backend="bool", N=N, ms=ms, us_per_config=1e3 * ms / N))
            out("%-10s %8d %12.1f %12.1f" % ("bool C++", N, ms, 1e3 * ms / N))
    return rows
