"""Dataset for the LOInK paper's Sec. VII soft manipulator (arXiv 2609.21275).

THE ROBOT. A planar Handed Shearing Auxetics (HSA) arm in SoRoMoX (`PlanarHSA`, after
Stolzle, Rus & Della Santina 2024), N = 3 segments of 0.1 m flexible length, two HSA rods per
segment. SoRoMoX ships no preset; the per-segment material/geometry parameters are the
upstream repo's `assets/robot_parameters/planar_hsa/{fpu,epu}_control.npz` (Stolzle et al.
2024 identification; vendored in `params/`), tiled to 3 segments with `length = 0.1`.
Default variant `fpu_control` (the paper does not name one; `--variant epu_control` selects
the other).

THE INPUT. With `consider_underactuation=True` (the default) the actuation input is one MOTOR
ROTATION `phi` (rad) per rod: u_i = [phi_i1, phi_i2] per segment, u in R^6 (the sum of the
two rods' twists elongates, the difference bends). The admissible set U is the motor range of
the parameter file, `phi_max` (FPU 200 deg = 3.4907 rad; EPU 270 deg): U = [0, phi_max]^6
(`--u-lo-frac` moves the lower end; 0 is the lower limit used here, an ASSUMPTION: the paper
does not print U).

THE EQUILIBRIUM. SoRoMoX has no static solve; LOInK simulated to rest. Rest of the dynamics
`B qdd + C qd + G + K(q) + D qd = tau_u(q, phi)` is `tau_u - G - K = 0` (qd = qdd = 0), so we
Newton-solve that residual with optimistix from the straight rest configuration q = 0, as
`src/gvs_arm/model.py` does for the GVS arm. This is the fixed point a damped rollout reaches
when it is stable (no hysteresis here), and ~100x cheaper. `converged` is optimistix's flag.

GRAVITY. `G = grad(_gravitational_energy)`; SoRoMoX's `_gravitational_force`/`dynamics_terms`
raise for more than one HSA segment in 0.5.0 (equal to the gradient for one segment, checked).

MOUNTING. `base_pose` = SoRoMoX's planar mounting pose. DEFAULT `standing`: the horizontal
pose with gravity negated, so it pulls toward the base and compresses the straight rest pose
(the paper's Fig. 7 draws the arm standing on its base, pointing up; sigma_a < 0 at u = 0).
`horizontal` (theta = 0): the
backbone runs along +y of the base frame and u = 0 is the straight rest pose y = [0, ~0.456, 0]
(the parameter file's gravity (0, 9.81) enters SoRoMoX's potential as -m g.y, so it slightly
STRETCHES this pose; length 0.453 without it). SoRoMoX's own default when `base_pose` is not
given is `upright` (theta = pi/2): backbone along -x, gravity perpendicular to it, so the arm
sags (theta ~ 0.95 rad at u = 0). `--mounting upright|hanging` select those; the paper does
not say which it used.

THE OUTPUT. `y = [p_x, p_y, theta]` from `forward_kinematics_end_effector(q)` (SoRoMoX returns
[theta, x, y]; reordered), in the BASE frame (base pose 0), the pose of the end effector =
distal cap + platform of the last segment (offset 0), theta in rad, `q` the 9 strain
coordinates [kappa_b, sigma_sh, sigma_a] x 3 (deviation from the reference strain).

    python scripts/loink_replication/hsa_dataset.py --n 2000000 --seed 0 --out X.npz
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optimistix as optx  # noqa: E402
import soromox  # noqa: E402
from soromox.systems.pcs.params import PlanarHSAParams  # noqa: E402
from soromox.systems.pcs.planar_hsa import PlanarHSA  # noqa: E402
from soromox.systems.pcs.structures import PlanarHSAStructure  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
SEG_LEN = 0.1
NEWTON = dict(rtol=1e-10, atol=1e-12)
MAX_STEPS = 64


def BuildModel(variant="fpu_control", num_segments=3, seg_len=SEG_LEN, mounting="standing"):
    one = PlanarHSAParams.from_npz(os.path.join(HERE, "params", f"{variant}.npz"))
    kw = {}
    # tile every per-segment field (leading axis 1) to num_segments
    per_segment = ("length", "proximal_cap_length", "distal_cap_length", "rod_height",
                   "rod_outer_radius", "rod_inner_radius", "rod_offset", "strain_coupling",
                   "platform_dimension", "rod_density", "platform_density", "end_cap_density",
                   "nominal_bending_stiffness", "nominal_shear_stiffness",
                   "nominal_axial_stiffness", "bending_shear_stiffness",
                   "bending_stiffness_correction", "shear_stiffness_correction",
                   "axial_stiffness_correction", "bending_damping", "shear_damping",
                   "axial_damping", "phi_max", "reference_strain")
    for name in per_segment:
        a = jnp.asarray(getattr(one, name))
        kw[name] = jnp.tile(a, (num_segments,) + (1,) * (a.ndim - 1))
    kw["length"] = jnp.full((num_segments,), seg_len)
    params = one.replace(**kw) if hasattr(one, "replace") else __import__("dataclasses").replace(one, **kw)
    from soromox.utils.geometry import poses
    ## `standing`: the paper's Fig. 7 (the arm on its base, pointing up, gravity compressing it):
    ## the horizontal pose with the parameter file's gravity negated, so it pulls toward the base.
    base = poses.planar_mounting_pose("horizontal" if mounting == "standing" else mounting)
    if mounting == "standing":
        params = params.replace(gravity=-jnp.asarray(params.gravity)) if hasattr(params, "replace") \
            else __import__("dataclasses").replace(params, gravity=-jnp.asarray(params.gravity))
    return PlanarHSA(params, PlanarHSAStructure(consider_underactuation=True),
                     base_pose=base), params


class Solver:
    def __init__(self, variant="fpu_control", u_lo_frac=0.0, mounting="standing"):
        self.robot, self.params = BuildModel(variant, mounting=mounting)
        r = self.robot
        self.phi_max = np.asarray(self.params.phi_max).reshape(-1)       # (6,)
        self.u_lo = u_lo_frac * self.phi_max
        self.ndof = r.num_internal_dofs

        ## Gravity as grad of SoRoMoX's own potential energy: `_gravitational_force` (and
        ## `dynamics_terms`) raise for >1 segment in soromox 0.5.0 (tip-Jacobian shape bug in
        ## `_mass_geometry_from_integration`); for 1 segment the two agree to machine precision.
        grav = jax.grad(r._gravitational_energy)

        def residual(q, u):
            return r.actuation_force(q, u) - grav(q) - r.elastic_force(q)

        def one(u):
            sol = optx.root_find(residual, optx.Newton(**NEWTON), jnp.zeros(self.ndof),
                                 args=u, max_steps=MAX_STEPS, throw=False)
            q = sol.value
            ok = sol.result == optx.RESULTS.successful
            res = jnp.max(jnp.abs(residual(q, u)))
            ee = r.forward_kinematics_end_effector(q)             # [theta, x, y]
            return q, jnp.stack([ee[1], ee[2], ee[0]]), ok, res

        self.batch = jax.jit(jax.vmap(one))
        self.rest = jax.jit(one)

    def Draw(self, n, rng):
        return self.u_lo + rng.random((n, 6)) * (self.phi_max - self.u_lo)

    def Solve(self, u):
        q, y, ok, res = self.batch(jnp.asarray(u))
        ok = np.asarray(ok) & np.isfinite(np.asarray(y)).all(1) & (np.asarray(res) < 1e-6)
        return np.asarray(q), np.asarray(y), ok


def Merge(chunk_dir, out, n, meta):
    files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npz")))
    parts = [np.load(f) for f in files]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("u", "q", "y", "converged")}
    assert len(cat["u"]) == n, (len(cat["u"]), n)
    np.savez(out, **cat, metadata=json.dumps(meta))
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--variant", default="fpu_control")
    ap.add_argument("--mounting", default="standing", choices=["standing", "upright", "horizontal", "hanging"])
    ap.add_argument("--u-lo-frac", type=float, default=0.0)
    ap.add_argument("--chunk-dir", default=None)
    ap.add_argument("--start", type=int, default=0, help="first chunk index (sharding)")
    ap.add_argument("--stride", type=int, default=1, help="chunk stride (sharding)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    cdir = a.chunk_dir or (os.path.splitext(a.out)[0] + "_chunks")
    os.makedirs(cdir, exist_ok=True)
    s = None if a.merge_only else Solver(a.variant, a.u_lo_frac, a.mounting)
    nchunks = -(-a.n // a.chunk)
    meta = dict(soromox=soromox.__version__ if hasattr(soromox, "__version__") else "0.5.0",
                jax=jax.__version__, variant=a.variant, mounting=a.mounting, num_segments=3, segment_length=SEG_LEN,
                u_range="[u_lo_frac*phi_max, phi_max] per motor, uniform",
                u_lo_frac=a.u_lo_frac, phi_max=None if s is None else s.phi_max.tolist(),
                solver="optimistix Newton from q=0 on tau_u - G - K(q) = 0",
                solver_tol=NEWTON, max_steps=MAX_STEPS, converged_def="optx successful & |res|_inf<1e-6",
                y="[p_x, p_y, theta] base frame, end effector, rad", seed=a.seed, n=a.n,
                chunk=a.chunk)
    t0, done = time.time(), 0
    for c in (range(0) if a.merge_only else range(a.start, nchunks, a.stride)):
        path = os.path.join(cdir, f"chunk_{c:06d}.npz")
        if os.path.exists(path):
            continue
        m = min(a.chunk, a.n - c * a.chunk)
        u = s.Draw(m, np.random.default_rng([a.seed, c]))
        t = time.time()
        q, y, ok = s.Solve(u)
        np.savez(path + ".tmp.npz", u=u, q=q, y=y, converged=ok)
        os.replace(path + ".tmp.npz", path)
        done += m
        print(f"chunk {c}/{nchunks} {m} samples {m / (time.time() - t):.0f}/s "
              f"conv {ok.mean():.4f}", flush=True)
    if a.no_merge:
        return
    cat = Merge(cdir, a.out, a.n, meta)
    print(f"merged {a.out}: conv {cat['converged'].mean():.4f}; wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
