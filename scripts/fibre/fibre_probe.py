"""IK-fibre connectivity and flow-latent jumps, per robot (an investigation for future work:
constrained motion planning in the chart's coordinates; see docs/notes-fibre.md).

Everything is in the chart's OWN world: targets are jrl FK of random in-limit configs (the
reachable domain the flow was trained on), the flow is conditioned on those poses, and the
fibre is {u : FK(u) = target}, u = normalized config (q = mid + half * u). No Drake, no
collision.

(1) connectivity, WITHOUT limits by default (`--limits` imposes the box): joint, strain and
    force limits are additional constraints, not kinematic or topological obstructions.
    Revolute joints are then periodic (configuration space is a torus; differences wrap).
    Solutions from flow draws AND uniform random starts are Newton-projected onto the fibre.
    A one-dimensional fibre (a 7-DoF arm) is a union of closed loops, traced EXACTLY
    (`trace_loops`). Higher-dimensional fibres (soft PCS, GVS) join pairs by a greedy
    fibre-following path OR a projected chord; union-find components are an UPPER bound.
    Rigid arms carry the analytic branch label as a purity check (on the iiwa the GC signs
    are separated by singular configurations, so a component must be label-pure).
(2) latent lines: straight lines z0 -> z1 (both N(0,I)) at fixed c; per-step |du|,
    jumpiness = max/median, pose error along the line, branch flips (rigid).
(3) pose lines: z fixed, c moving along a reachable pose path (FK of a straight
    configuration-space segment): does the chart tear as the TASK moves? This is the
    measurement constrained motion planning needs, (2) only covers self-motion.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

## The repo root: scripts/fibre/ -> ../..; LIK_REPO overrides it (a copy run outside the tree).
REPO = os.environ.get("LIK_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/training"))
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)  # jrl builds per-joint transforms in the default dtype

from pole_metric import load_solver  # noqa: E402

CKPT = {
    "panda": "models/panda/panda__n6__step620000.pkl",
    "iiwa14": "models/iiwa14/iiwa14__n4__step620000.pkl",   # the record's rung (cluster tree)
    "soft9": "models/soft9/soft9__n6__step620000.pkl",
    "soft12": "models/soft12/soft12__n6__step620000.pkl",
    "soft16": "models/soft16/soft16__n6__step620000.pkl",
    "gvs_pushrod9_o1": "models/gvs_pushrod9_o1/gvs_pushrod9_o1__n6__step620000.pkl",
    "gvs_pushrod9_o2": "models/gvs_pushrod9_o2/gvs_pushrod9_o2__n6__step620000.pkl",
}


# ---------------------------------------------------------------- quaternions / error
# Error convention, everywhere: e = [p - p_t ; rotvec(quat * conj(quat_t))], 6 rows, quats wxyz.
def qmul(a, b):
    w1, v1 = a[..., :1], a[..., 1:]
    w2, v2 = b[..., :1], b[..., 1:]
    return np.concatenate([w1 * w2 - (v1 * v2).sum(-1, keepdims=True),
                           w1 * v2 + w2 * v1 + np.cross(v1, v2)], -1)


def rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., :1])
    scale = np.where(s > 1e-12, ang / np.maximum(s, 1e-300), 2.0)
    return v * scale


def _err_generic(xp, p, tgt, cross, norm, atan2, where):
    """The error in any array library (numpy-like `xp`), differentiable where `p` is."""
    pt, qt = tgt[..., :3], tgt[..., 3:]
    w1, v1 = p[..., 3:4], p[..., 4:]
    w2, v2 = qt[..., :1], -qt[..., 1:]
    w = w1 * w2 - (v1 * v2).sum(-1, keepdims=True)
    v = w1 * v2 + w2 * v1 + cross(v1, v2)
    flip = where(w < 0, -1.0, 1.0)
    w, v = w * flip, v * flip
    s = norm(v)
    big = s > 1e-9
    s_safe = where(big, s, xp.ones_like(s))          # double-where: no NaN gradient at s = 0
    scale = where(big, 2 * atan2(s_safe, w) / s_safe, 2.0 / w)
    return xp.concatenate([p[..., :3] - pt, v * scale], -1)


def _err_torch(p, tgt):
    return _err_generic(torch, p, tgt, lambda a, b: torch.linalg.cross(a, b),
                        lambda v: torch.linalg.norm(v, dim=-1, keepdim=True), torch.atan2,
                        lambda c, a, b: torch.where(c, a, torch.as_tensor(b, dtype=p.dtype)))


#: Largest GVS batch handed to JAX at once. The vmapped equilibrium solve's memory grows with
#: the batch: 48 workers each solving 16384 configurations OOM-killed a 192 GB node.
GVS_CHUNK = 1024


def _bucket(x, floor=8):
    """`x` padded along axis 0 to the next power of two (repeating row 0), and the true size.
    JAX compiles one program per batch SHAPE; the path-follower's batch shrinks every step,
    so without bucketing every step of a GVS solve paid a fresh compilation."""
    B = len(x)
    size = max(floor, 1 << max(B - 1, 0).bit_length())
    if size == B:
        return x, B
    return np.concatenate([x, np.repeat(x[:1], size - B, 0)]), B


class Robot:
    """The chart, its robot's FK in the chart's own frame, and EXACT error Jacobians:
    torch autograd through jrl / the soft PCS map, JAX (implicit differentiation through the
    equilibrium solve, optimistix's default ImplicitAdjoint) for the GVS arm. Central
    differences remain only as a self-check at construction (`jac_check`)."""

    def __init__(self, name):
        self.name = name
        self.chart, self.width, self.ndof = load_solver(name, os.path.join(REPO, CKPT[name]))
        self.jrl = self.chart.robot
        lo, hi = np.array(self.jrl.actuated_joints_limits, dtype=np.float64).T
        self.mid, self.half = (lo + hi) / 2, (hi - lo) / 2
        self.model = self.chart.nn_model.double().cpu().eval()
        self.labeler = None
        # jrl builds per-joint transforms in the DEFAULT dtype (and its config resets it at
        # import), so float64 FK needs the default set here and its float32 cache dropped.
        torch.set_default_dtype(torch.float64)
        cache = getattr(self.jrl, "_parent_T_joint_cache", None)
        if cache:
            for dev in cache:
                cache[dev].clear()
        if name == "panda":
            from src.panda_analytic_ik import Analytic_IK_Panda
            a = Analytic_IK_Panda()
            self.labeler = lambda q: tuple(int(x) for x in a.gc(q, branches=3))
        elif name == "iiwa14":
            self.labeler = lambda q: tuple(int(np.sign(q[k]) or 1) for k in (1, 3, 5))
        ## Revolute joints are periodic: one turn in normalized units (None for strain / force).
        self.period = (2 * np.pi / self.half) if name in ("panda", "iiwa14") else None
        self.gvs = name.startswith("gvs_")
        if self.gvs:
            self._build_gvs()
        self.jac_check = self._check_jacobian()

    # -- GVS: the model's own jitted tip pose (cfg is the normalized force, u == cfg) ----
    def _build_gvs(self):
        import jax
        import jax.numpy as jnp
        gm = self.jrl._model
        tip = gm._tip_pose  # jitted (pose7, converged) from ONE equilibrium solve

        def err_one(cfg, tgt):
            pose, ok = tip(cfg)
            e = _err_generic(jnp, pose, tgt, jnp.cross,
                             lambda v: jnp.linalg.norm(v, axis=-1, keepdims=True),
                             jnp.arctan2, jnp.where)
            return e, ok

        self._gvs_err = jax.jit(jax.vmap(err_one))
        self._gvs_jac = jax.jit(jax.vmap(jax.jacfwd(err_one, has_aux=True)))
        self._jnp = jnp

    def q(self, u):
        return self.mid + self.half * u

    def u(self, q):
        return (q - self.mid) / self.half

    def fk(self, u):
        """[B,7] in the chart's frame; NaN rows where the GVS equilibrium did not converge."""
        if self.gvs:
            if len(u) > GVS_CHUNK:  # a vmapped solve over 16k configs OOM-killed 48 workers
                return np.concatenate([self.fk(u[i:i + GVS_CHUNK]) for i in range(0, len(u), GVS_CHUNK)])
            x, B = _bucket(self.q(u))
            poses, ok = self.jrl._model.TipPoseBatch(x)
            poses, ok = np.array(poses, dtype=np.float64)[:B], np.asarray(ok)[:B]
            poses[~np.asarray(ok)] = np.nan
            return poses
        with torch.no_grad():
            out = self.jrl.forward_kinematics(torch.as_tensor(self.q(u), dtype=torch.float64),
                                              dtype=torch.float64)
        return out.detach().cpu().numpy().astype(np.float64)

    def err(self, u, tgt):
        """[B,6] = [dp; rotvec(q q_t^-1)], tgt [B,7]."""
        p = self.fk(u)
        conj = tgt[:, 3:] * np.array([1, -1, -1, -1])
        return np.concatenate([p[:, :3] - tgt[:, :3], rotvec(qmul(p[:, 3:], conj))], -1)

    def jac(self, u, tgt):
        """[B,6,n] exact d err / d u. Rows that come back non-finite fall back to `jac_fd`."""
        if self.gvs:
            if len(u) > GVS_CHUNK:
                return np.concatenate([self.jac(u[i:i + GVS_CHUNK], tgt[i:i + GVS_CHUNK])
                                       for i in range(0, len(u), GVS_CHUNK)])
            x, B = _bucket(self.q(u))
            t, _ = _bucket(tgt)
            J, ok = self._gvs_jac(self._jnp.asarray(x), self._jnp.asarray(t))
            J, ok = np.array(J, dtype=np.float64)[:B], np.asarray(ok)[:B]
            J = J * self.half[None, None, :]
            J[~np.asarray(ok)] = np.nan
        else:
            ut = torch.as_tensor(u, dtype=torch.float64).clone().requires_grad_(True)
            qt = torch.as_tensor(self.mid) + torch.as_tensor(self.half) * ut
            e = _err_torch(self.jrl.forward_kinematics(qt, dtype=torch.float64),
                           torch.as_tensor(tgt, dtype=torch.float64))
            rows = [torch.autograd.grad(e[:, i].sum(), ut, retain_graph=i < 5)[0] for i in range(6)]
            J = torch.stack(rows, 1).detach().numpy()
        bad = ~np.isfinite(J).all((1, 2))
        if bad.any() and not self.gvs:
            J[bad] = self.jac_fd(u[bad], tgt[bad])
        return J

    def jac_fd(self, u, tgt, eps=1e-6):
        B, n = u.shape
        E = np.eye(n) * eps
        up = (u[:, None, :] + E[None]).reshape(-1, n)
        um = (u[:, None, :] - E[None]).reshape(-1, n)
        t2 = np.repeat(tgt, n, 0)
        d = (self.err(up, t2) - self.err(um, t2)).reshape(B, n, 6) / (2 * eps)
        return d.transpose(0, 2, 1)  # [B,6,n]

    def _check_jacobian(self, B=8, seed=12345):
        """Exact vs central differences at random in-box points, a few cm from a target."""
        rng = np.random.default_rng(seed)
        u = rng.uniform(-0.9, 0.9, (B, self.ndof))
        tgt = self.fk(np.clip(u + 0.05 * rng.standard_normal(u.shape), -1, 1))
        Je, Jf = self.jac(u, tgt), self.jac_fd(u, tgt)
        ok = np.isfinite(Je).all((1, 2)) & np.isfinite(Jf).all((1, 2))
        rel = float(np.abs(Je[ok] - Jf[ok]).max() / max(np.abs(Jf[ok]).max(), 1e-12)) if ok.any() else None
        print(f"{self.name}: exact-vs-FD Jacobian max rel diff {rel} on {int(ok.sum())}/{B} points",
              flush=True)
        if rel is None or rel > 1e-4:
            raise RuntimeError(f"{self.name}: exact Jacobian disagrees with central differences "
                               f"(max rel diff {rel}); refusing to run on it")
        return rel

    def decode(self, z, pose7):
        with torch.no_grad():
            q, _ = self.chart.decode(self.model, torch.as_tensor(z, dtype=torch.float64),
                                     torch.as_tensor(pose7, dtype=torch.float64))
        return self.u(q.cpu().numpy())


def pinv(J, act=None, lam=1e-12):
    if act is not None:
        J = J * (~act)[:, None, :]
    JJt = J @ J.transpose(0, 2, 1) + lam * np.eye(6)
    return J.transpose(0, 2, 1) @ np.linalg.inv(JJt), J


# ------------------------------------------------------------------- connectivity
# LIMITS: whether the joint / strain / force box is imposed on the fibre. OFF by default:
# limits are additional constraints, not kinematic or topological obstructions (Thomas,
# 2026-11-09). Off, revolute joints are periodic -- the arm's configuration space is a
# torus -- so differences are wrapped and solutions are stored with angles in (-pi, pi].
LIMITS = False


def wrap(R, d):
    """Differences in u, wrapped onto the torus for revolute joints (no-op with LIMITS or for
    strain / force coordinates, which live in R^n)."""
    if LIMITS or R.period is None:
        return d
    return d - R.period * np.round(d / R.period)


def canon(R, u):
    """u with every revolute angle in (-pi, pi]."""
    if LIMITS or R.period is None:
        return u
    return R.u(np.mod(R.q(u) + np.pi, 2 * np.pi) - np.pi)


def dist(R, a, b):
    return np.linalg.norm(wrap(R, a - b), axis=-1)


def project(R, u, tgt, iters=60, maxstep=0.2, tol=1e-10):
    """Min-norm Newton onto the fibre, P points at once. With LIMITS the box is an active set
    and a solution must lie in it. Rows whose FK fails (GVS: equilibrium not converged) stay
    put and come back unconverged. Returns (u, converged)."""
    u = u.copy()
    for _ in range(iters):
        e = R.err(u, tgt)
        if np.isfinite(e).all() and np.abs(e).max() < tol:
            break
        J = R.jac(u, tgt)
        bad = ~(np.isfinite(e).all(1) & np.isfinite(J).all((1, 2)))
        e[bad], J[bad] = 0.0, 0.0
        Jp, _ = pinv(J)
        du = -(Jp @ e[..., None])[..., 0]
        if LIMITS:
            act = (np.abs(u) >= 1 - 1e-12) & (np.sign(du) * np.sign(u) > 0)
            if act.any():
                Jp, _ = pinv(J, act)
                du = -(Jp @ e[..., None])[..., 0]
        nrm = np.linalg.norm(du, axis=1, keepdims=True)
        u = u + du * np.minimum(1, maxstep / np.maximum(nrm, 1e-300))
        if LIMITS:
            u = np.clip(u, -1, 1)
    e = np.abs(R.err(u, tgt)).max(1)
    ok = np.nan_to_num(e, nan=np.inf) < 1e-9
    if LIMITS:
        ok &= np.abs(u).max(1) <= 1 + 1e-9
    return canon(R, u), ok


def track(R, ua, ub, tgt, H=0.05, maxsteps=400, stall_max=40):
    """Greedy fibre-following path ua -> ub for P pairs at once: a null-space step toward the
    (wrapped) other end, a chord-Newton corrector, the box as an active set under LIMITS.
    Returns ok [P]. Can stall where the fibre is connected, so joins are a lower bound."""
    P = len(ua)
    u = ua.copy()
    h = np.full(P, H)
    done = np.zeros(P, bool)
    ok = np.zeros(P, bool)
    best = dist(R, ub, u)
    stall = np.zeros(P, int)
    lim = 1.0 if LIMITS else np.inf
    for _ in range(maxsteps):
        idx = np.where(~done)[0]
        if len(idx) == 0:
            break
        U, D = u[idx], wrap(R, ub[idx] - u[idx])
        dn = np.linalg.norm(D, axis=1)
        close = dn < h[idx]
        ok[idx[close]] = True
        done[idx[close]] = True
        keep = ~close
        idx, U, D, dn = idx[keep], U[keep], D[keep], dn[keep]
        if len(idx) == 0:
            continue
        T = tgt[idx]
        act = ((np.abs(U) >= 1 - 1e-9) & (np.sign(D) * np.sign(U) > 0)) if LIMITS else np.zeros_like(U, bool)
        J = R.jac(U, T)
        badJ = ~np.isfinite(J).all((1, 2))
        J[badJ] = 0.0
        Jp, Jm = pinv(J, act)
        Dm = D * ~act
        step = Dm - (Jp @ (Jm @ Dm[..., None]))[..., 0]
        sn = np.linalg.norm(step, axis=1)
        stuck = (sn < 1e-3 * dn) | badJ
        done[idx[stuck]] = True
        step = step * (np.minimum(h[idx], sn) / np.maximum(sn, 1e-300))[:, None]
        Up = np.clip(U + step, -lim, lim)
        for _ in range(8):
            e = np.nan_to_num(R.err(Up, T))
            Up = np.clip(Up - (Jp @ e[..., None])[..., 0], -lim, lim)
        e = np.nan_to_num(np.abs(R.err(Up, T)).max(1), nan=np.inf)
        moved = np.linalg.norm(Up - U, axis=1)
        good = (e < 1e-8) & (moved < 3 * h[idx]) & ~stuck
        gi = idx[good]
        u[gi] = Up[good]
        bi = idx[~good & ~stuck]
        h[bi] /= 2
        done[bi[h[bi] < H / 64]] = True
        h[gi] = np.minimum(h[gi] * 1.5, H)
        nd = dist(R, ub[idx], u[idx])
        imp = nd < best[idx] - 1e-4
        best[idx[imp]] = nd[imp]
        stall[idx[imp]] = 0
        stall[idx[~imp]] += 1
        done[idx[stall[idx] > stall_max]] = True
    return ok


def chord(R, ua, ub, tgt, ds=0.02, iters=6):
    """Second joining method: follow the straight chord ua -> (nearest lift of) ub, projecting
    each point onto the fibre (predictor along the chord, chord-Newton corrector, the box as
    an active set under LIMITS). Returns ok [P]."""
    P, n = ua.shape
    ub = ua + wrap(R, ub - ua)
    lim = 1.0 if LIMITS else np.inf
    K = int(max(20, np.ceil(np.linalg.norm(ub - ua, axis=1).max() / ds)))
    du = (ub - ua) / K
    u = ua.copy()
    ok = np.ones(P, bool)
    for _ in range(K):
        idx = np.where(ok)[0]
        if len(idx) == 0:
            break
        U, T = u[idx], tgt[idx]
        Up = np.clip(U + du[idx], -lim, lim)
        J = R.jac(Up, T)
        badJ = ~np.isfinite(J).all((1, 2))
        J[badJ] = 0.0
        for _ in range(iters):
            e = np.nan_to_num(R.err(Up, T))
            Jp, _ = pinv(J)
            step = -(Jp @ e[..., None])[..., 0]
            if LIMITS:
                act = (np.abs(Up) >= 1 - 1e-12) & (np.sign(step) * np.sign(Up) > 0)
                if act.any():
                    Jp, _ = pinv(J, act)
                    step = -(Jp @ e[..., None])[..., 0]
            Up = np.clip(Up + step, -lim, lim)
        e = np.nan_to_num(np.abs(R.err(Up, T)).max(1), nan=np.inf)
        moved = np.linalg.norm(Up - U, axis=1)
        good = (e < 1e-8) & (moved < 3 * np.maximum(np.linalg.norm(du[idx], axis=1), 0.01)) & ~badJ
        u[idx[good]] = Up[good]
        ok[idx[~good]] = False
    return ok & (np.linalg.norm(u - ub, axis=1) < 0.05)


def trace_loops(R, sols, tgt7, h=0.01, tol=0.02, maxsteps=30000):
    """EXACT components of a one-dimensional fibre without limits (a 7-DoF arm): every
    component is a closed loop on the torus, so trace each loop from an unlabelled solution
    (tangent = the Jacobian's null vector, predictor-corrector) until it closes, labelling
    every solution it passes within `tol`. Returns labels, loops closed, loops left open."""
    m = len(sols)
    labels = -np.ones(m, int)
    closed = opened = 0
    T1 = tgt7[None]
    for i in range(m):
        if labels[i] >= 0:
            continue
        labels[i] = i
        u, tprev, travelled, step, ok_loop = sols[i].copy(), None, 0.0, h, False
        for _ in range(maxsteps):
            J = R.jac(u[None], T1)
            t = np.linalg.svd(J[0])[2][-1]
            if tprev is not None and t @ tprev < 0:
                t = -t
            Jp, _ = pinv(J)
            while True:
                up = u + step * t
                for _ in range(8):
                    up = up - Jp[0] @ np.nan_to_num(R.err(up[None], T1)[0])
                e = np.nan_to_num(np.abs(R.err(up[None], T1)).max(), nan=np.inf)
                if e < 1e-9 and np.linalg.norm(up - u) < 2 * step:
                    break
                step /= 2
                if step < h / 64:
                    break
            if step < h / 64:
                break
            travelled += np.linalg.norm(up - u)
            u, tprev, step = up, t, min(step * 1.5, h)
            near = dist(R, sols, u[None]) < tol
            for j in np.where(near & (labels != i))[0]:
                if labels[j] < 0:
                    labels[j] = i
                else:  # a loop through an already-labelled solution: one component
                    labels[labels == labels[j]] = i
            if travelled > 10 * h and dist(R, u, sols[i]) < h:
                ok_loop = True
                break
        closed += ok_loop
        opened += not ok_loop
    return labels, closed, opened


PROGRESS = False


def _progress(msg):
    """Progress lines inside one target (--progress): the GVS arm's targets take long enough
    that the per-target line alone cannot tell slow from stuck."""
    if PROGRESS:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def components(R, sols, tgt7, knn=4, cross=3, rounds=4):
    """Component labels of the sampled solutions. A one-dimensional fibre without limits is
    traced exactly (`trace_loops`); otherwise pairs are joined by `track` OR `chord` (kNN
    pairs, then the closest `cross` pairs of every two components, `rounds` times) -- an
    upper bound on the component count. Returns (labels, info)."""
    m = len(sols)
    if not LIMITS and R.ndof - 6 == 1:
        t0 = time.time()
        lab, closed, opened = trace_loops(R, sols, tgt7)
        _progress(f"  traced {closed} closed / {opened} open loops, {time.time() - t0:.0f} s")
        return lab, {"method": "trace", "loops_closed": closed, "loops_open": opened}
    parent = list(range(m))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    D = dist(R, sols[:, None], sols[None])
    tried = set()

    def attempt(pairs):
        pairs = [p for p in pairs if p not in tried and find(p[0]) != find(p[1])]
        if not pairs:
            return 0
        tried.update(pairs)
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        T = np.repeat(tgt7[None], len(pairs), 0)
        ok = track(R, sols[a], sols[b], T)
        if (~ok).any():
            ok[~ok] = chord(R, sols[a[~ok]], sols[b[~ok]], T[~ok])
        for (i, j), o in zip(pairs, ok):
            if o:
                parent[find(i)] = find(j)
        return len(pairs)

    t0 = time.time()
    n_tried = attempt([(i, int(j)) for i in range(m) for j in np.argsort(D[i])[1:knn + 1]])
    _progress(f"  kNN joins: {len({find(i) for i in range(m)})} components, {n_tried} pairs, "
              f"{time.time() - t0:.0f} s")
    for rnd in range(rounds):  # closest members of each pair of distinct components
        roots = sorted({find(i) for i in range(m)})
        if len(roots) == 1:
            break
        mem = {r: [i for i in range(m) if find(i) == r] for r in roots}
        pairs = []
        for x in range(len(roots)):
            for y in range(x + 1, len(roots)):
                A, B = mem[roots[x]], mem[roots[y]]
                sub = D[np.ix_(A, B)]
                for f in np.argsort(sub, axis=None)[:cross]:
                    i, j = np.unravel_index(f, sub.shape)
                    pairs.append((A[i], B[j]))
        if attempt(pairs) == 0:
            break
        n_tried += len(pairs)
        _progress(f"  cross round {rnd}: {len({find(i) for i in range(m)})} components, "
                  f"{n_tried} pairs, {time.time() - t0:.0f} s")
    return np.array([find(i) for i in range(m)]), {"method": "track+chord", "pairs": n_tried}



def line_metrics(R, U, C, L, S):
    """Jumps along L lines of S decoded configurations U [L*S, n] against poses C [L*S, 7].
    A jump is a step > 10x the line's median step AND > 0.05 normalized. Lines whose pose
    path has a non-converged (NaN) pose are dropped (GVS only)."""
    n = U.shape[1]
    E = R.err(U, C)
    U = U.reshape(L, S, n)
    good = np.isfinite(E).all(1).reshape(L, S).all(1)
    U, E = U[good], E.reshape(L, S, 6)[good]
    perr = np.linalg.norm(E[..., :3], axis=-1)
    rerr = np.linalg.norm(E[..., 3:], axis=-1)
    st = np.linalg.norm(np.diff(U, axis=1), axis=-1)  # [L, S-1]
    med = np.median(st, axis=1)
    jump = (st > 10 * med[:, None]) & (st > 0.05)
    out = {
        "lines": int(good.sum()),
        "median_step": float(np.median(med)),
        "jumpiness_median": float(np.median(st.max(1) / med)),
        "jumpiness_p90": float(np.percentile(st.max(1) / med, 90)),
        "max_step_median": float(np.median(st.max(1))),
        "lines_with_jump": int(jump.any(1).sum()),
        "jumps_total": int(jump.sum()),
        "pos_err_median_mm": float(1e3 * np.median(perr)),
        "pos_err_p99_mm": float(1e3 * np.percentile(perr, 99)),
        "rot_err_median_deg": float(np.degrees(np.median(rerr))),
        "out_of_box_frac": float((np.abs(U) > 1).any(-1).mean()),
    }
    if R.labeler is not None:
        lab = np.array([[hash(R.labeler(R.q(U[l, k]))) for k in range(S)] for l in range(len(U))])
        flip = lab[:, 1:] != lab[:, :-1]
        out.update({
            "branch_flips_total": int(flip.sum()),
            "lines_with_flip": int(flip.any(1).sum()),
            "flips_at_jump": int((flip & jump).sum()),
            "distinct_branches_on_lines": int(len(np.unique(lab))),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, choices=list(CKPT))
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--nflow", type=int, default=32)
    ap.add_argument("--nrand", type=int, default=32)
    ap.add_argument("--lines", type=int, default=64)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--pose_seg", type=float, default=0.6,
                    help="length of the configuration-space segment whose FK is a pose line")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_conn", action="store_true")
    ap.add_argument("--skip_lines", action="store_true",
                    help="connectivity only (the GVS arm: ~20-40 ms per equilibrium solve)")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--limits", action="store_true",
                    help="impose the joint / strain / force box on the fibre (default: off)")
    a = ap.parse_args()
    global PROGRESS, LIMITS
    PROGRESS, LIMITS = a.progress, a.limits
    rng = np.random.default_rng(a.seed)
    R = Robot(a.robot)
    n = R.ndof
    print(f"{a.robot}: ndof {n}, width {R.width}, ckpt {CKPT[a.robot]}", flush=True)
    res = {"robot": a.robot, "ckpt": CKPT[a.robot], "jac_check_rel": R.jac_check, "limits": LIMITS,
           "args": vars(a), "targets": []}
    for t in range(a.targets):
        t0 = time.time()
        u_star = rng.uniform(-1, 1, (1, n))
        tgt7 = R.fk(u_star)[0]
        rec = {}
        # ---------------- latent lines: c fixed at the target, z0 -> z1
        ## The draws below happen with or without --skip_lines, so the same seed gives the same
        ## targets either way; only the decodes and the FK are skipped.
        z0 = rng.standard_normal((a.lines, R.width))
        z1 = rng.standard_normal((a.lines, R.width))
        s = np.linspace(0, 1, a.steps)
        Z = (z0[:, None] * (1 - s)[None, :, None] + z1[:, None] * s[None, :, None]).reshape(-1, R.width)
        C = np.repeat(tgt7[None], len(Z), 0)
        if not a.skip_lines:
            rec["latent"] = line_metrics(R, R.decode(Z, C), C, a.lines, a.steps)
        # ---------------- pose lines: z fixed, c along a REACHABLE pose path (the FK of a
        # straight segment of length `pose_seg` in normalized configuration space)
        ua = rng.uniform(-1, 1, (a.lines, n))
        d = rng.standard_normal((a.lines, n))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        ub = np.clip(ua + a.pose_seg * d, -1, 1)
        Upath = (ua[:, None] * (1 - s)[None, :, None] + ub[:, None] * s[None, :, None]).reshape(-1, n)
        Z = np.repeat(rng.standard_normal((a.lines, R.width)), a.steps, 0)
        if not a.skip_lines:
            C = R.fk(Upath)
            rec["pose_lines"] = line_metrics(R, R.decode(Z, np.nan_to_num(C)), C, a.lines, a.steps)
        _progress(f"target {t}: lines done, {time.time() - t0:.0f} s")
        # ---------------- connectivity
        if not a.skip_conn:
            uf = R.decode(rng.standard_normal((a.nflow, R.width)), np.repeat(tgt7[None], a.nflow, 0))
            ur = rng.uniform(-1, 1, (a.nrand, n))
            T = np.repeat(tgt7[None], a.nflow + a.nrand, 0)
            starts = np.vstack([uf, ur])
            up, okp = project(R, np.clip(starts, -1, 1) if LIMITS else starts, T)
            _progress(f"target {t}: {int(okp.sum())}/{len(okp)} projected, {time.time() - t0:.0f} s")
            src = np.array(["flow"] * a.nflow + ["rand"] * a.nrand)
            sols = np.vstack([up[okp], canon(R, u_star)])
            ssrc = list(src[okp]) + ["target"]
            keep = []
            for i in range(len(sols)):
                if all(dist(R, sols[i], sols[j]) > 1e-2 for j in keep):
                    keep.append(i)
            sols = sols[keep]
            ssrc = [ssrc[i] for i in keep]
            comp, info = components(R, sols, tgt7)
            roots, sizes = np.unique(comp, return_counts=True)
            c = {"n_solutions": int(len(sols)), "flow_converged": int(okp[:a.nflow].sum()),
                 "rand_converged": int(okp[a.nflow:].sum()), "components": int(len(roots)),
                 "component_sizes": sorted(sizes.tolist(), reverse=True), **info,
                 "outside_box_solutions": int((np.abs(sols) > 1 + 1e-9).any(1).sum()),
                 "flow_components": int(len({comp[i] for i in range(len(sols)) if ssrc[i] == "flow"})),
                 "target_component_size": (int(sizes[list(roots).index(comp[ssrc.index("target")])])
                                           if "target" in ssrc else None),
                 "solutions": sols.round(6).tolist(), "labels": comp.tolist()}
            if R.labeler is not None:
                labs = [R.labeler(R.q(x)) for x in sols]
                c["distinct_branches"] = len(set(labs))
                c["impure_components"] = int(sum(len({labs[i] for i in range(len(sols)) if comp[i] == r}) > 1
                                                 for r in roots))
                c["branches_split"] = int(sum(len({comp[i] for i in range(len(sols)) if labs[i] == L}) > 1
                                              for L in set(labs)))
            rec["conn"] = c
        rec["seconds"] = round(time.time() - t0, 1)
        res["targets"].append(rec)
        brief = {k: ({kk: vv for kk, vv in v.items() if kk not in ("solutions", "labels")}
                     if isinstance(v, dict) else v) for k, v in rec.items()}
        print(f"target {t}: {json.dumps(brief)}", flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
