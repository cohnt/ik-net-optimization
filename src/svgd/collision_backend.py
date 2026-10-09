"""The program's collision row, evaluated EXACTLY by Drake, for a batch of configurations.

No geometric proxy anywhere (Thomas, 2026-10-08: "I don't like the proxy idea at all"). The row
a particle optimizer sees is the one `IKFlowProgram.CreateCollisionFreeConstraint` hands IPOPT:

    row(q) = collision_row_scale * MinimumDistanceLowerBoundConstraint(
                 plant, bound=collision_bound,
                 influence_distance_offset=collision_influence_offset).Eval(q)

with upper bound `collision_row_scale`, so `verify()` and the optimizer see the same functional
and there is no proxy-versus-Drake disagreement to adjudicate. `tests/test_collision_backend.py`
pins the pooled value AND gradient bitwise against the program's own binding.

**Parallelism is by processes, because pydrake holds the GIL.** Measured 2026-10-08 on this
laptop (20 threads): a ThreadPool over `MinimumDistanceLowerBoundConstraint.Eval` is flat at
~3k evals/s from 1 to 16 threads; a 16-process pool does Panda N = 256 / 1024 / 4096 in
~15-20 / 51 / 183 ms. Per configuration the pool AMORTISES with batch size (78 -> 50 -> 45 us),
because what shrinks is the per-task IPC overhead -- so a batch is sent as ONE chunk per worker
(`np.array_split`), never as per-configuration tasks. Workers are `spawn`ed, never forked: the
parent has usually initialised CUDA by the time a pool exists, and a forked CUDA context is
undefined behaviour. Workers build their scene lazily on first use, so constructing a pool is
cheap and the scene cost is paid once per worker.

**A non-finite configuration must never reach Drake.** `MultibodyPlant::SetPositions` has a
`DRAKE_DEMAND(AllFinite(q))`; in pydrake that failure surfaces as `SystemExit` (a BaseException,
not an Exception), so an unguarded worker would simply die. Rows are screened for finiteness
before any Drake call and come back as NaN value and NaN gradient; the per-row `except
BaseException` behind that screen is belt-and-braces for anything else Drake aborts on.

**The scene is rebuilt from a picklable description, not shipped.** `SceneSpec` carries the
directives YAML and, for the grasp task, the welded target mug. `GenerateDiagramWithMug` welds
the mug through `pydrake.common.schema.Transform(RigidTransform)`, which stores the rotation as
RPY in degrees; a quaternion round-trip of that pose is NOT bit-exact (measured 1.1e-16 off, and
the schema RPY differs), while `RigidTransform(RotationMatrix(R), p)` reproduces the original
exactly. The spec therefore carries the 3x3 rotation for reconstruction and the quaternion only
for a human reader.

**`ParallelCollisionChecker` is the boolean, C++-parallel path** for where a boolean suffices
(collision-free draws, a feasibility mask): `SceneGraphCollisionChecker.CheckConfigsCollisionFree`
does 1024 configurations in ~16-22 ms. Its padding defaults to `collision_bound`, so "in
collision" means the same thing as the row's `> scale`: Drake's penalty exceeds 1 iff some
candidate pair is closer than `bound` (0 disagreements in 1500 random Panda draws), and with
padding 0 the checker would pass the 0 <= d < bound sliver the row rejects (7 of 1500).

Upstream item for Thomas, not patched here: releasing the GIL in the `Eval` /
`CalcRobotClearance` bindings would turn this pool into a thread pool with no IPC.
"""

import atexit
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass

import numpy as np

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
##                                  The worker (spawned)                                 ##
## ------------------------------------------------------------------------------------ ##

class _WorkerScene:
    """One worker's diagram, context and constraint. Built on the first batch."""

    def __init__(self, spec):
        from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint
        self.spec = spec
        self.diagram = spec.build_diagram()
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.nq = self.plant.num_positions()
        self.constraint = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=spec.collision_bound,
            influence_distance_offset=spec.influence_distance_offset,
            plant_context=self.plant_context,
        )

    def eval(self, Q, need_grad):
        from pydrake.autodiffutils import ExtractGradient, ExtractValue, InitializeAutoDiff
        Q = np.asarray(Q, dtype=np.float64)
        n = Q.shape[0]
        if Q.shape[1] != self.nq:
            raise ValueError("collision pool: got %d plant positions, scene has %d"
                             % (Q.shape[1], self.nq))
        scale = self.spec.row_scale
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


def _worker_main(spec, conn):
    scene = None
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            Q, need_grad = msg
            try:
                if scene is None:
                    scene = _WorkerScene(spec)
                values, grads = scene.eval(Q, need_grad)
                conn.send(("ok", values, grads))
            except BaseException:
                conn.send(("error", traceback.format_exc()))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


## ------------------------------------------------------------------------------------ ##
##                                   DrakeCollisionPool                                  ##
## ------------------------------------------------------------------------------------ ##

class DrakeCollisionPool:
    """`workers` spawned processes, each owning one copy of the scene described by `spec`.

    `eval(Q)` returns the SCALED row value `scale * y` and its gradient `scale * dy/dq` for
    every row of `Q` (shape `[N, num_positions]` -- the FULL plant vector, which on the soft PCS
    arm is 231 floating-body coordinates, never assumed to be 7). One chunk per worker; rows
    that are non-finite, or on which Drake aborts, come back NaN. Use as a context manager or
    call `close()`; an `atexit` hook closes anything left open.
    """

    def __init__(self, spec, workers=None, ctx="spawn", eval_timeout=600.0):
        if workers is None:
            workers = max(1, (os.cpu_count() or 2) // 2)
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.spec = spec
        self.workers = int(workers)
        self.eval_timeout = float(eval_timeout)
        self._ctx = mp.get_context(ctx)
        self._procs = []
        self._conns = []
        self._closed = False
        self._broken = None
        for _ in range(self.workers):
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(target=_worker_main, args=(spec, child_conn), daemon=True)
            proc.start()
            child_conn.close()
            self._procs.append(proc)
            self._conns.append(parent_conn)
        atexit.register(self.close)

    # -- lifecycle ------------------------------------------------------------------- #
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self, join_timeout=5.0):
        """Ask every worker to exit, then terminate whatever is still alive. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        for conn in self._conns:
            try:
                conn.send(None)
            except Exception:
                pass
        for proc in self._procs:
            proc.join(join_timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(1.0)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass

    @property
    def closed(self):
        return self._closed

    def alive(self):
        return [p.is_alive() for p in self._procs]

    # -- evaluation ------------------------------------------------------------------ #
    def eval(self, q_plant, need_grad=True):
        """(value[N], grad[N, nq]) of the scaled row at every configuration; grad None if
        `need_grad=False` (a plain `Eval`, a little cheaper when only values are wanted).
        Exactly `collect(submit(q_plant, need_grad))`."""
        return self.collect(self.submit(q_plant, need_grad))

    def submit(self, q_plant, need_grad=True):
        """Send one chunk of `q_plant` to every worker and return at once with a ticket for
        `collect`. Between the two the workers run Drake while the caller does other work
        (the solver's split step launches the flow Jacobian on the GPU here). At most one
        ticket may be outstanding: the pipes are request/reply, so a second `submit` before
        the first `collect` would interleave replies -- it raises instead."""
        if self._closed:
            raise RuntimeError("DrakeCollisionPool is closed")
        if self._broken is not None:
            raise RuntimeError("DrakeCollisionPool is broken: %s" % self._broken)
        if getattr(self, "_outstanding", None) is not None:
            raise RuntimeError("DrakeCollisionPool.submit: a ticket is already outstanding")
        Q = np.ascontiguousarray(np.asarray(q_plant, dtype=np.float64))
        if Q.ndim != 2:
            raise ValueError("q_plant must be [N, num_positions], got shape %r" % (Q.shape,))
        N, nq = Q.shape
        chunks = np.array_split(Q, self.workers) if N else []
        sent = []
        for k, chunk in enumerate(chunks):
            if chunk.shape[0] == 0:
                continue
            try:
                self._conns[k].send((chunk, need_grad))
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                self._broken = "worker %d is gone (%s)" % (k, type(e).__name__)
                raise RuntimeError(self._broken) from e
            sent.append(k)
        offsets = np.cumsum([0] + [c.shape[0] for c in chunks])
        ticket = (N, nq, bool(need_grad), sent, offsets)
        self._outstanding = ticket
        return ticket

    def drain(self):
        """Collect and discard an outstanding ticket, if any: a caller that raised between
        `submit` and `collect` must not leave stale replies in the pipes for the next one."""
        ticket = getattr(self, "_outstanding", None)
        if ticket is not None:
            self.collect(ticket)

    def collect(self, ticket):
        """Wait for the workers' replies to `ticket` and assemble `(value, grad)`."""
        if ticket is not getattr(self, "_outstanding", None):
            raise RuntimeError("DrakeCollisionPool.collect: not the outstanding ticket")
        self._outstanding = None
        N, nq, need_grad, sent, offsets = ticket
        values = np.full(N, np.nan)
        grads = np.full((N, nq), np.nan) if need_grad else None
        for k in sent:
            conn = self._conns[k]
            if not conn.poll(self.eval_timeout):
                self._broken = "worker %d did not answer within %.0f s" % (k, self.eval_timeout)
                raise RuntimeError(self._broken)
            try:
                reply = conn.recv()
            except (EOFError, ConnectionResetError, OSError) as e:
                self._broken = ("worker %d died (%s) -- a Drake abort escaped the per-row "
                                "guard, or the scene failed to build" % (k, type(e).__name__))
                raise RuntimeError(self._broken) from e
            if reply[0] != "ok":
                self._broken = "worker %d raised:\n%s" % (k, reply[1])
                raise RuntimeError(self._broken)
            _, v, g = reply
            lo, hi = offsets[k], offsets[k + 1]
            values[lo:hi] = v
            if need_grad:
                grads[lo:hi] = g
        return values, grads


## ------------------------------------------------------------------------------------ ##
##                      The autograd Function (torch imported lazily)                    ##
## ------------------------------------------------------------------------------------ ##
## `torch` is deliberately not imported at module level: a spawned worker re-imports this
## module to find `_worker_main`, and sixteen workers each importing torch is a second and
## a few hundred MB apiece for nothing. `CollisionRow` is built on first access through the
## module-level `__getattr__` (PEP 562), so `from ... import CollisionRow` still works.

_COLLISION_ROW_CLASS = None


def _collision_row_class():
    global _COLLISION_ROW_CLASS
    if _COLLISION_ROW_CLASS is not None:
        return _COLLISION_ROW_CLASS
    import torch

    class CollisionRow(torch.autograd.Function):
        """`forward(q_plant [N, nq], pool) -> value [N]`, same dtype and device as the input.

        Drake runs in float64 on the CPU whatever the input is; the value comes back in the
        input's dtype and device and the stored `[N, nq]` gradient drives `backward`
        (`grad_output[:, None] * grad`). NaN rows propagate as NaN."""

        @staticmethod
        def forward(ctx, q_plant, pool):
            Q = q_plant.detach().to("cpu", torch.float64).numpy()
            value, grad = pool.eval(Q, need_grad=True)
            ctx.save_for_backward(
                torch.as_tensor(grad).to(device=q_plant.device, dtype=q_plant.dtype))
            return torch.as_tensor(value).to(device=q_plant.device, dtype=q_plant.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (grad,) = ctx.saved_tensors
            return grad_output.unsqueeze(1) * grad, None

    _COLLISION_ROW_CLASS = CollisionRow
    return CollisionRow


def __getattr__(name):
    if name == "CollisionRow":
        return _collision_row_class()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def collision_row(q_plant, pool):
    """The scaled collision row for a batch of plant vectors, differentiable in torch."""
    return _collision_row_class().apply(q_plant, pool)


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

def measure_pool(spec, Ns=(64, 256, 1024, 4096), workers=(8, 16), repeats=3, seed=0,
                 lower=None, upper=None, checker=True, out=print):
    """Print ms per batch and us per configuration of the pool (and the boolean checker).

    Configurations are uniform in `[lower, upper]` (defaults: the scene plant's position
    limits, which are finite on the rigid arms; pass them explicitly for a floating-body
    robot). Each cell is the median of `repeats` after one warm-up batch, so the lazy scene
    build is not in the number. Returns the rows as a list of dicts.
    """
    rng = np.random.default_rng(seed)
    if lower is None or upper is None:
        diagram = spec.build_diagram()
        plant = diagram.GetSubsystemByName("plant")
        lower, upper = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError("measure_pool: plant limits are not finite; pass lower/upper")
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    rows = []
    out("%-10s %8s %12s %12s" % ("backend", "N", "ms/batch", "us/config"))
    for K in workers:
        with DrakeCollisionPool(spec, workers=K) as pool:
            pool.eval(rng.uniform(lower, upper, size=(max(K, 8), lower.size)))
            for N in Ns:
                Q = rng.uniform(lower, upper, size=(N, lower.size))
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    pool.eval(Q)
                    times.append(time.perf_counter() - t0)
                ms = 1e3 * float(np.median(times))
                rows.append(dict(backend="pool", workers=K, N=N, ms=ms, us_per_config=1e3 * ms / N))
                out("%-10s %8d %12.1f %12.1f" % ("pool K=%d" % K, N, ms, 1e3 * ms / N))
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
            rows.append(dict(backend="bool", workers=None, N=N, ms=ms, us_per_config=1e3 * ms / N))
            out("%-10s %8d %12.1f %12.1f" % ("bool C++", N, ms, 1e3 * ms / N))
    return rows
