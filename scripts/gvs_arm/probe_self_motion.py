#!/usr/bin/env python3
"""Is the GVS push-rod arm's redundancy REAL -- is there a self-motion manifold?

`GvsArmSpec.__post_init__` refuses a rung with `ninputs <= 6`, so the arm has nine
inputs against a six-dimensional pose task and three degrees of redundancy BY
CONSTRUCTION.  That is an arithmetic claim about variable counts.  Whether it is a
KINEMATIC claim depends on the task Jacobian `d(tip pose)/d(rod forces)` actually
having rank six, and on the null space being something the arm can travel along a
finite distance rather than an infinitesimal direction that curves away immediately.

Both are measured here, and they are different questions:

  RANK.  The 6 x 9 spatial Jacobian (translation rows in metres, rotation rows as an
  angular velocity in radians, recovered from the quaternion derivative as
  `omega = 2 Im(conj(q) qdot)` so the rows are a genuine twist and not four
  constrained quaternion components).  Rank six means a three-dimensional null
  space at that configuration.  The SMALLEST singular value matters as much as the
  rank: a technically-full-rank but ill-conditioned Jacobian has a self-motion the
  arm can barely execute.

  TRAVEL.  A finite walk along the null space with a corrector: step along a null
  direction, then pull the pose error back to zero with the Jacobian's pseudo-
  inverse, and keep going while the pose stays inside `--pos_tol` / `--rot_tol` and
  the forces stay inside their box.  The distance travelled in normalized force
  space is the answer: the self-motion manifold is as big as that walk, and every
  configuration on it is a different arm shape holding the SAME tip pose.

Ill-conditioning here is expected to grow near the force box, where the arm
saturates -- the same saturation that makes the joint-space baseline fail -- so the
distribution matters, not one draw.

    python scripts/gvs_arm/probe_self_motion.py --rung gvs_pushrod9_o1 --samples 200
"""
import argparse
import os
import pathlib
import sys

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
os.nice(19)  # a probe; it shares the machine

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.gvs_arm.model import GetModel  # noqa: E402
from src.gvs_arm.params import RUNGS, GetSpec  # noqa: E402


def _task_jacobian_factory(model):
    """`cfg -> (6 x 9 spatial Jacobian, tip pose)`, jitted.

    The tip pose is `[x, y, z, qw, qx, qy, qz]`; its quaternion block has a unit-norm
    constraint, so differentiating it raw would give four rows of rank three and an
    SVD whose singular values mean nothing. Mapping the quaternion derivative to an
    angular velocity gives six rows in the units the task actually uses.
    """
    tip = model._tip_pose  # the jitted `cfg -> (pose, converged)` the sampler uses

    def pose_only(cfg):
        pose, _ = tip(cfg)
        return pose

    def jacobian(cfg):
        seven = jax.jacfwd(pose_only)(cfg)          # (7, 9)
        pose = pose_only(cfg)
        q = pose[3:]
        dq = seven[3:, :]                           # (4, 9)
        ## omega = 2 * Im(conj(q) (x) qdot), column by column.
        w, x, y, z = q[0], q[1], q[2], q[3]
        dw, dx, dy, dz = dq[0], dq[1], dq[2], dq[3]
        omega = 2.0 * jnp.stack([w * dx - x * dw - y * dz + z * dy,
                                 w * dy + x * dz - y * dw - z * dx,
                                 w * dz - x * dy + y * dx - z * dw])
        return jnp.concatenate([seven[:3, :], omega], axis=0), pose

    return jax.jit(jacobian)


def _pose_error(pose, target):
    """Position error (m) and rotation error (rad) between two `[xyz, quat]` poses."""
    dp = np.linalg.norm(pose[:3] - target[:3])
    dot = abs(float(np.dot(pose[3:], target[3:])))
    return dp, 2.0 * np.arccos(min(1.0, dot))


def _walk(jacobian, model, start, pos_tol, rot_tol, step, max_steps, seed):
    """Travel along the self-motion manifold from `start`, correcting back onto it.

    Returns the arc length travelled in normalized force space and why it stopped.
    """
    rng = np.random.default_rng(seed)
    _, target = jacobian(start)
    target = np.asarray(target)
    cfg = np.array(start)
    travelled, direction = 0.0, None
    for _ in range(max_steps):
        J, pose = jacobian(cfg)
        J = np.asarray(J)
        ## The null space of the 6 x 9 task Jacobian: the directions that move the arm
        ## without moving the tip.
        _, s, vt = np.linalg.svd(J)
        if s[5] <= 1e-9:
            return travelled, "rank deficient"
        null = vt[6:, :]
        if direction is None:
            direction = null.T @ rng.normal(size=null.shape[0])
        else:
            ## Keep going the same way: project the previous direction onto the new
            ## null space, so the walk follows one curve instead of diffusing.
            direction = null.T @ (null @ direction)
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return travelled, "null space turned"
        direction /= norm
        trial = cfg + step * direction
        if np.abs(trial).max() > 1.0:
            return travelled, "force box"
        ## Corrector: one Gauss-Newton pull back onto the level set of the pose.
        Jt, pose_t = jacobian(trial)
        dp, dr = _pose_error(np.asarray(pose_t), target)
        if dp > pos_tol or dr > rot_tol:
            Jt = np.asarray(Jt)
            residual = np.concatenate([np.asarray(pose_t)[:3] - target[:3],
                                       _rotation_residual(np.asarray(pose_t), target)])
            trial = trial - np.linalg.pinv(Jt) @ residual
            if np.abs(trial).max() > 1.0:
                return travelled, "force box"
            _, pose_t = jacobian(trial)
            dp, dr = _pose_error(np.asarray(pose_t), target)
            if dp > pos_tol or dr > rot_tol:
                return travelled, "corrector lost the pose"
        travelled += step
        cfg = trial
    return travelled, "step budget"


def _rotation_residual(pose, target):
    """Rotation vector taking the target orientation onto `pose`'s, in radians."""
    q, t = pose[3:], target[3:]
    if np.dot(q, t) < 0:
        t = -t
    ## conj(t) (x) q, as a rotation vector.
    w = t[0] * q[0] + t[1] * q[1] + t[2] * q[2] + t[3] * q[3]
    v = np.array([t[0] * q[1] - t[1] * q[0] - t[2] * q[3] + t[3] * q[2],
                  t[0] * q[2] + t[1] * q[3] - t[2] * q[0] - t[3] * q[1],
                  t[0] * q[3] - t[1] * q[2] + t[2] * q[1] - t[3] * q[0]])
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v * (2.0 * np.arctan2(n, w) / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", default="gvs_pushrod9_o1", choices=list(RUNGS))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--walks", type=int, default=20)
    ap.add_argument("--step", type=float, default=0.01, help="normalized force units")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--pos_tol", type=float, default=1e-4, help="metres")
    ap.add_argument("--rot_tol", type=float, default=1e-3, help="radians")
    ap.add_argument("--scale", type=float, default=1.0, help="draw in [-scale, scale]^9")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = GetSpec(args.rung)
    model = GetModel(spec)
    jacobian = _task_jacobian_factory(model)
    rng = np.random.default_rng(args.seed)
    draws = rng.uniform(-args.scale, args.scale, size=(args.samples, spec.ninputs))

    singular = np.empty((args.samples, 6))
    for i, cfg in enumerate(draws):
        J, _ = jacobian(cfg)
        singular[i] = np.linalg.svd(np.asarray(J), compute_uv=False)[:6]

    rank6 = int((singular[:, 5] > 1e-9).sum())
    print(f"{args.rung}: {spec.ninputs} inputs, 6-D pose task -> "
          f"{spec.ninputs - 6} degrees of redundancy by construction\n")
    print(f"TASK JACOBIAN RANK over {args.samples} uniform draws in "
          f"[-{args.scale}, {args.scale}]^{spec.ninputs}")
    print(f"  full rank 6 (so a {spec.ninputs - 6}-dimensional null space): "
          f"{rank6}/{args.samples}")
    names = ["s1 (largest)", "s2", "s3", "s4", "s5", "s6 (smallest)"]
    print(f"  {'':<14} {'median':>10} {'p5':>10} {'min':>10}")
    for k, name in enumerate(names):
        col = singular[:, k]
        print(f"  {name:<14} {np.median(col):>10.4f} {np.percentile(col, 5):>10.4f} "
              f"{col.min():>10.2e}")
    cond = singular[:, 0] / np.maximum(singular[:, 5], 1e-300)
    print(f"  condition number s1/s6: median {np.median(cond):.1f}, "
          f"p95 {np.percentile(cond, 95):.1f}, max {cond.max():.3g}")

    print(f"\nSELF-MOTION TRAVEL from {args.walks} starts (step {args.step} in normalized "
          f"force units,\nholding the tip pose to {args.pos_tol * 1e3:.1f} mm and "
          f"{np.degrees(args.rot_tol):.2f} deg)")
    lengths, reasons = [], {}
    for w in range(args.walks):
        length, why = _walk(jacobian, model, draws[w], args.pos_tol, args.rot_tol,
                            args.step, args.max_steps, args.seed + w)
        lengths.append(length)
        reasons[why] = reasons.get(why, 0) + 1
    lengths = np.asarray(lengths)
    print(f"  arc length travelled: median {np.median(lengths):.2f}, "
          f"min {lengths.min():.2f}, max {lengths.max():.2f} "
          f"(the box itself is 2.0 wide per axis)")
    print("  stopped because: " + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
    print(f"\nA median walk of {np.median(lengths):.2f} in normalized force units at a FIXED "
          f"tip pose is the\nself-motion manifold being traversed, not merely existing "
          f"infinitesimally.")


if __name__ == "__main__":
    main()
