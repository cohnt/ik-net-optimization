"""The GVS push-rod arm's forward model: SoRoMoX, solved to static equilibrium, in JAX.

THIS IS SOROMOX, NOT A PORT OF IT.  The soft PCS arm re-derived its (one-line) constant-strain
exponential in torch and pinned it to SoRoMoX through an offline golden file. For a
variable-strain rod with routed actuators that would mean re-deriving the Magnus
integration, the Legendre strain basis, the section-projected stiffness and the rod-force
quadrature -- a second implementation of the reference that can only drift from it. Thomas,
2026-09-29: "the point is to use that repo." So SoRoMoX is a runtime dependency here:
`GVS.from_segments` builds the rod from `params.py`, `actuation_force` and `potential_force`
are its own, and the ONE thing this module adds is the equilibrium root-find SoRoMoX does
not ship (it simulates to rest; LOInK did exactly that to build its dataset).

WHAT A FORWARD EVALUATION IS.  `cfg -> u = cfg * F_max -> q* : actuation_force(q*, u) =
potential_force(q*)  ->  poses at the sub-link arc lengths -> Drake's floating-body layout`.
Newton on the residual from the straight arm (a cold start, so the map is a deterministic
function of `u` and not of the solver's history), implicit differentiation for `dq*/du`
(optimistix's `ImplicitAdjoint`), forward-mode autodiff for everything downstream.
Measured on the prototype: 3 Newton steps, residual 1e-16, `dq*/du` against central
differences 9e-10, ~18 ms per solve and ~20 ms per Jacobian jitted on one CPU core.

JAX IS CPU-ONLY AND FLOAT64 HERE.  The flow runs in torch on the GPU; this map runs beside it
on the CPU. XLA's thread pool takes every core by default, which is right for a dataset build
on an exclusive node and wrong for eight benchmark workers sharing one: `GVS_ARM_XLA_THREADS=1`
in the worker's environment pins it, and `cluster/run_items.sh` sets exactly that.
`jax_enable_x64` is set before any array exists, because JAX defaults to float32 and every
numerical argument in this project is made in float64.

FRAMES.  SoRoMoX's backbone is its local x-axis; with its default upright mounting the base
frame is rotated so that local x points along world +z, which is where every scene here
welds a robot. So the straight arm's bodies all carry the rotation `R_y(-90 deg)`, and the
TIP FRAME -- the frame the gripper welds to and the flow is conditioned on -- is declared in
the SDF as `R_y(+90 deg)` relative to the tip body, so that its z-axis runs along the
backbone exactly as the soft PCS arm's does and the scene furniture is byte-identical. The
body quaternions are canonicalised to `w >= 0`; the pose is the same either way, and a
consistent sign is what the finite-difference tests need.

NUMPY IN, NUMPY OUT.  The program's AutoDiffXd chain (`src/gvs_arm_program.py`, inherited
from the PCS arm's) composes `map_jacobian @ cfg_gradients` in numpy, so it never sees a JAX
array and the torch side never sees one either.
"""

import math
import os

## Thread policy must be decided before XLA initialises, i.e. before `import jax`.
_THREADS = os.environ.get("GVS_ARM_XLA_THREADS")
if _THREADS is not None:
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={_THREADS}")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optimistix as optx  # noqa: E402
from soromox.actuation import ThreadlikeActuator, ThreadlikeRouting  # noqa: E402
from soromox.systems.components import JointSpec, LinearProfile, LinkSpec  # noqa: E402
from soromox.systems.gvs import GVS, GVSSegment, StrainBasisSpec  # noqa: E402

from src.gvs_arm.params import GvsArmSpec  # noqa: E402

#: The straight, unstretched rod in SoRoMoX's convention: unit axial strain along local x.
_XI_REF = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

#: Newton on the equilibrium residual. `rtol`/`atol` bound the STEP (optimistix's Cauchy
#: termination), so a converged solve sits at the residual's round-off floor.
_NEWTON = dict(rtol=1e-10, atol=1e-12)
_MAX_NEWTON_STEPS = 64

#: The tip frame relative to the tip body: `R_y(+90 deg)` takes the body's z-axis onto its
#: x-axis, i.e. onto the backbone. Also written into the generated SDF as the frame's pose.
TIP_FRAME_PITCH = math.pi / 2
_R_TIP = jnp.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def _canonical(quat):
    """`w >= 0`, applied along the last axis."""
    return jnp.where(quat[..., :1] < 0, -quat, quat)


def BuildSoromoxModel(spec: GvsArmSpec):
    """The SoRoMoX GVS model of one rung, from the spec and nothing else."""
    segments = []
    for i in range(spec.num_segments):
        r0, r1 = spec.segment_radii(i)
        segments.append(GVSSegment(
            link=LinkSpec.circular(length=spec.segment_length, radius=LinearProfile(r0, r1),
                                   density=spec.density, reference_strain=list(_XI_REF),
                                   young_modulus=spec.young_modulus,
                                   poisson_ratio=spec.poisson_ratio),
            joint=JointSpec.fixed(),
            basis=StrainBasisSpec(type=spec.basis, strain_selector=tuple(spec.active_strains),
                                  basis_order=spec.basis_order),
            num_gauss_points=spec.num_gauss_points))

    ## One rod per input, segment-major, each spanning ONLY its own segment. The routing
    ## offset is `rod_offset_ratio * r(s)` with `s` the GLOBAL arc length, which SoRoMoX's
    ## linear routing expresses as intercept + slope * s. Local x must be zero: the
    ## backbone is the local x-axis.
    intercepts, slopes, starts, ends = [], [], [], []
    slope_r = spec.rod_offset_ratio * (spec.radius_tip - spec.radius_base) / spec.total_length
    for i in range(spec.num_segments):
        for azimuth in spec.rod_azimuths:
            d0 = spec.rod_offset_ratio * spec.radius_base
            intercepts.append([0.0, d0 * np.cos(azimuth), d0 * np.sin(azimuth)])
            slopes.append([0.0, slope_r * np.cos(azimuth), slope_r * np.sin(azimuth)])
            starts.append(i)
            ends.append(i)
    routing = ThreadlikeRouting.linear(
        intercept=jnp.asarray(intercepts), slope=jnp.asarray(slopes),
        start_segment_index=tuple(starts), end_segment_index=tuple(ends))
    ## `push_rods`: SoRoMoX's sign convention for a rod that lengthens its path under a
    ## positive force. Its `[0, inf)` bound is metadata only -- nothing clips `u` -- so a
    ## negative force is simply a pull, which is what "push-pull" means.
    rods = ThreadlikeActuator.push_rods(routing)
    return GVS.from_segments(segments, gravity=jnp.asarray(spec.gravity, dtype=jnp.float64),
                             actuators=rods)


def _quaternion_from_rotation(R):
    """`(..., 3, 3) -> (..., 4)` wxyz by Shepperd's method, finite everywhere.

    Four candidate formulas, one per largest diagonal term, each computed with a CLAMPED
    square root so the untaken branches are finite too -- `jnp.where` evaluates every
    branch, and a NaN in an untaken one poisons forward-mode derivatives exactly as it does
    in torch. Within a branch's region all four agree up to a global sign, so the selected
    value is continuous and its derivative is the derivative of one smooth function.
    """
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    r01, r02, r10 = R[..., 0, 1], R[..., 0, 2], R[..., 1, 0]
    r12, r20, r21 = R[..., 1, 2], R[..., 2, 0], R[..., 2, 1]
    candidates = jnp.stack([1.0 + t, 1.0 + 2.0 * r00 - t, 1.0 + 2.0 * r11 - t,
                            1.0 + 2.0 * r22 - t], axis=-1)
    pick = jnp.argmax(candidates, axis=-1)
    s = jnp.sqrt(jnp.clip(candidates, 1e-12, None))          # (..., 4) = 2 * |component|
    inv = 0.5 / s
    q_w = jnp.stack([0.5 * s[..., 0], (r21 - r12) * inv[..., 0], (r02 - r20) * inv[..., 0],
                     (r10 - r01) * inv[..., 0]], axis=-1)
    q_x = jnp.stack([(r21 - r12) * inv[..., 1], 0.5 * s[..., 1], (r01 + r10) * inv[..., 1],
                     (r02 + r20) * inv[..., 1]], axis=-1)
    q_y = jnp.stack([(r02 - r20) * inv[..., 2], (r01 + r10) * inv[..., 2], 0.5 * s[..., 2],
                     (r12 + r21) * inv[..., 2]], axis=-1)
    q_z = jnp.stack([(r10 - r01) * inv[..., 3], (r02 + r20) * inv[..., 3],
                     (r12 + r21) * inv[..., 3], 0.5 * s[..., 3]], axis=-1)
    stacked = jnp.stack([q_w, q_x, q_y, q_z], axis=-2)          # (..., 4 candidates, 4)
    return jnp.take_along_axis(stacked, pick[..., None, None], axis=-2)[..., 0, :]


class GvsArmModel:
    """One rung's forward model, built once per process (`GetModel`) and jitted.

    Every public method takes the NORMALIZED input `cfg` in `[-1, 1]^ninputs` and returns
    numpy. The batched methods take `(B, ninputs)` and are what the dataset sampler and the
    self-collision screen use.
    """

    def __init__(self, spec: GvsArmSpec):
        self.spec = spec
        self.robot = BuildSoromoxModel(spec)
        assert self.robot.num_actuators == spec.ninputs
        self.backbone_dof = int(self.robot.num_internal_dofs)
        assert self.backbone_dof == spec.backbone_dof, (
            self.backbone_dof, spec.backbone_dof)
        self._force_limits = jnp.asarray(spec.force_limits, dtype=jnp.float64)
        self._arclengths = jnp.asarray(spec.body_arclengths(), dtype=jnp.float64)
        self._radii = jnp.asarray(spec.body_radii(), dtype=jnp.float64)
        robot = self.robot

        def residual(q, u):
            return robot.actuation_force(q, u) - robot.potential_force(q)

        def solve_from(cfg, q0):
            u = cfg * self._force_limits
            sol = optx.root_find(residual, optx.Newton(**_NEWTON), q0, args=u,
                                 max_steps=_MAX_NEWTON_STEPS, throw=False)
            ok = sol.result == optx.RESULTS.successful
            return sol.value, ok, sol.stats["num_steps"]

        def solve(cfg):
            """Cold start from the straight arm: the map is a function of `cfg` alone."""
            return solve_from(cfg, jnp.zeros(self.backbone_dof))

        def poses(q):
            """`(num_bodies, 4, 4)` world transforms at the body arc lengths."""
            return robot.forward_kinematics_abscissa_batched(q, self._arclengths)

        def plant_q(cfg):
            q, _, _ = solve(cfg)
            g = poses(q)
            quat = _canonical(_quaternion_from_rotation(g[:, :3, :3]))
            return jnp.concatenate([quat, g[:, :3, 3]], axis=-1).reshape(-1)

        def plant_jacobian(cfg):
            return jax.jacfwd(lambda c: (plant_q(c), plant_q(c)), has_aux=True)(cfg)

        def tip_pose(cfg):
            """The TIP FRAME's pose, `[x, y, z, qw, qx, qy, qz]` with `w >= 0` (jrl's layout)."""
            q, ok, _ = solve(cfg)
            g = poses(q)[-1]
            quat = _canonical(_quaternion_from_rotation(g[:3, :3] @ _R_TIP))
            return jnp.concatenate([g[:3, 3], quat]), ok

        def sphere_centres(cfg):
            q, ok, _ = solve(cfg)
            return poses(q)[:, :3, 3], ok

        def tip_and_centres(cfg):
            """Tip pose AND sphere centres from ONE equilibrium solve: the dataset sampler
            needs both for every draw, and solving twice doubled its cost."""
            q, ok, _ = solve(cfg)
            g = poses(q)
            quat = _canonical(_quaternion_from_rotation(g[-1, :3, :3] @ _R_TIP))
            return jnp.concatenate([g[-1, :3, 3], quat]), g[:, :3, 3], ok

        self._solve = jax.jit(solve)
        self._tip_and_centres_batch = jax.jit(jax.vmap(tip_and_centres))
        self._solve_from = jax.jit(solve_from)
        self._plant_q = jax.jit(plant_q)
        self._plant_jacobian = jax.jit(plant_jacobian)
        self._tip_pose = jax.jit(tip_pose)
        self._solve_batch = jax.jit(jax.vmap(solve))
        self._tip_pose_batch = jax.jit(jax.vmap(tip_pose))
        self._sphere_centres_batch = jax.jit(jax.vmap(sphere_centres))
        self._poses_of_q = jax.jit(poses)

    # -- the forward model ------------------------------------------------------------

    def _cfg(self, cfg):
        cfg = jnp.asarray(np.asarray(cfg, dtype=float))
        if cfg.shape != (self.spec.ninputs,):
            raise ValueError(f"{self.spec.name} takes {self.spec.ninputs} inputs, got {cfg.shape}")
        return cfg

    def Equilibrium(self, cfg):
        """The backbone's generalized coordinates at equilibrium; raises if Newton failed."""
        q, ok, steps = self._solve(self._cfg(cfg))
        if not bool(ok):
            raise RuntimeError(f"{self.spec.name}: equilibrium did not converge in "
                               f"{int(steps)} Newton steps at cfg={np.asarray(cfg)}")
        return np.asarray(q)

    def EquilibriumFrom(self, cfg, q0):
        """Newton from an arbitrary backbone state -- the uniqueness probe's tool, not the map's."""
        q, ok, steps = self._solve_from(self._cfg(cfg), jnp.asarray(q0, dtype=jnp.float64))
        return np.asarray(q), bool(ok), int(steps)

    def PlantQ(self, cfg):
        """Every body's `[qw, qx, qy, qz, x, y, z]`, bodies in `spec.body_names()` order."""
        return np.asarray(self._plant_q(self._cfg(cfg)))

    def PlantJacobian(self, cfg):
        """`(d(plant q)/d(cfg), plant q)` -- the PCS arm's `ConfigJacobianGen` contract."""
        jacobian, value = self._plant_jacobian(self._cfg(cfg))
        return np.asarray(jacobian), np.asarray(value)

    def TipPose(self, cfg):
        """`[x, y, z, qw, qx, qy, qz]` of the tip, `w >= 0`."""
        pose, ok = self._tip_pose(self._cfg(cfg))
        if not bool(ok):
            raise RuntimeError(f"{self.spec.name}: equilibrium did not converge")
        return np.asarray(pose)

    def PosesOfBackbone(self, q):
        """World transforms `(num_bodies, 4, 4)` of a given backbone state (tests)."""
        return np.asarray(self._poses_of_q(jnp.asarray(q, dtype=jnp.float64)))

    # -- batched, for the dataset and the screens -----------------------------------

    def EquilibriumBatch(self, cfgs):
        q, ok, steps = self._solve_batch(jnp.asarray(np.asarray(cfgs, dtype=float)))
        return np.asarray(q), np.asarray(ok), np.asarray(steps)

    def TipPoseBatch(self, cfgs):
        """`(B, 7)` tip poses and a `(B,)` convergence mask."""
        poses, ok = self._tip_pose_batch(jnp.asarray(np.asarray(cfgs, dtype=float)))
        return np.asarray(poses), np.asarray(ok)

    def SphereCentresBatch(self, cfgs):
        """`(B, num_bodies, 3)` collision-sphere centres and a `(B,)` convergence mask."""
        centres, ok = self._sphere_centres_batch(jnp.asarray(np.asarray(cfgs, dtype=float)))
        return np.asarray(centres), np.asarray(ok)

    def SelfCollides(self, cfgs):
        """Does any UNFILTERED pair of spheres overlap? `(B,)` bool, plus the converged mask.

        Pairs at least two segments apart, matching the SDF's filter groups exactly; the
        radii are the per-body taper radii, so two spheres collide when their centres are
        closer than the sum of THEIR radii.
        """
        centres, ok = self.SphereCentresBatch(cfgs)
        return self.CollidesFromCentres(centres), ok

    def CollidesFromCentres(self, centres):
        """The self-collision screen on `(B, num_bodies, 3)` centres already in hand."""
        first, second = self.self_collision_pairs()
        separation = np.linalg.norm(centres[:, first, :] - centres[:, second, :], axis=-1)
        radii = np.asarray(self.spec.body_radii())
        return (separation < radii[first] + radii[second]).any(axis=-1)

    def TipAndCentresBatch(self, cfgs):
        """`(B, 7)` tip poses, `(B, num_bodies, 3)` sphere centres and the converged mask,
        from one solve per draw -- the dataset sampler's path."""
        tips, centres, ok = self._tip_and_centres_batch(jnp.asarray(np.asarray(cfgs, dtype=float)))
        return np.asarray(tips), np.asarray(centres), np.asarray(ok)

    def self_collision_pairs(self):
        spec = self.spec
        segment_of = []
        for segment in range(spec.num_segments):
            segment_of += [segment] * spec.sublinks_per_segment
        segment_of.append(spec.num_segments - 1)             # the tip joins the last segment
        first, second = [], []
        for i in range(len(segment_of)):
            for j in range(i + 1, len(segment_of)):
                if abs(segment_of[i] - segment_of[j]) >= 2:
                    first.append(i)
                    second.append(j)
        return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)

    # -- the rod's surface, for the containment test --------------------------------

    def BackboneSurfacePoints(self, cfg, samples=300, directions=12):
        """Points on the TRUE rod surface at `samples` arc lengths: what the spheres must contain."""
        q = self.Equilibrium(cfg)
        s = jnp.linspace(0.0, self.spec.total_length, samples)
        g = np.asarray(self.robot.forward_kinematics_abscissa_batched(jnp.asarray(q), s))
        radii = np.asarray([self.spec.radius_at(float(x)) for x in np.asarray(s)])
        angles = np.arange(directions) * (2 * np.pi / directions)
        ## The cross-section is the local y-z plane: local x is the backbone.
        offsets = np.stack([np.zeros_like(angles), np.cos(angles), np.sin(angles)], axis=-1)
        points = (g[:, None, :3, 3] + radii[:, None, None]
                  * np.einsum("nij,dj->ndi", g[:, :3, :3], offsets))
        return points.reshape(-1, 3)

    def WarmUp(self):
        """Pay every jit's compile cost now rather than inside the first solver iterate."""
        zero = np.zeros(self.spec.ninputs)
        self.PlantQ(zero)
        self.PlantJacobian(zero)
        self.TipPose(zero)


_MODELS = {}


def GetModel(spec: GvsArmSpec) -> GvsArmModel:
    """One model per rung per process: building and jitting are seconds, not milliseconds."""
    if spec.name not in _MODELS:
        _MODELS[spec.name] = GvsArmModel(spec)
    return _MODELS[spec.name]


def PlantSlotMap(plant, spec: GvsArmSpec):
    """Where each body's 7 positions live in THIS plant's position vector.

    Read from the plant, never assumed: welding the gripper to the tip moves that body to
    the FRONT of Drake's ordering (the PCS arm measured `soft_tip_link` going from slot 168
    to slot 0). Returns `picks` with `plant_q = canonical_q[picks]`.
    """
    num_positions = plant.num_positions()
    picks = np.empty(num_positions, dtype=np.int64)
    seen = np.zeros(num_positions, dtype=bool)
    for index, name in enumerate(spec.body_names()):
        start = plant.GetBodyByName(name).floating_positions_start()
        if start < 0:
            raise RuntimeError(f"{name} is not a floating body in this plant -- welded?")
        picks[start:start + 7] = np.arange(7 * index, 7 * index + 7)
        seen[start:start + 7] = True
    if not seen.all():
        raise RuntimeError(
            f"{spec.name}: {int((~seen).sum())} of the plant's {num_positions} positions "
            f"belong to no body of this arm; the scene has degrees of freedom the forward "
            f"model does not drive")
    return picks
