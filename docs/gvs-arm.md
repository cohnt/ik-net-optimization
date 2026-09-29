# The GVS push-rod arm: what was built, what was measured, what is queued

Branch `gvs-actuated-arm`, 2026-09-29. Design decisions and standing rules are in `CLAUDE.md`
("The GVS push-rod arm"); this file holds the measured numbers behind them and the state of
the build. Operations (what to submit, in what order, what to check) are in
`cluster/GVS_ARM_RUNBOOK.md`.

## The robot, in numbers

| | value | where it comes from |
| --- | --- | --- |
| segments | 3 x 0.2667 m | `GvsArmSpec.num_segments`, `total_length = 0.80` |
| backbone radius | 0.030 m at the base, 0.015 m at the tip, linear | `radius_base`, `radius_tip` |
| material | E = 1e5 Pa, nu = 0.45 (G = 3.45e4 Pa); density 1000 kg/m^3, gravity off | spec |
| strains | `kappa_y`, `kappa_z`, `sigma_x` (SoRoMoX's names; local x is the backbone) | spec |
| backbone order | Legendre 0 / 1 / 2 -> 9 / 18 / 27 generalized coordinates | the rungs |
| quadrature | 7 Gauss points per segment | `num_gauss_points` |
| rods | 3 per segment at 0, 120, 240 deg, at 0.7 r(s), each on its own segment | spec |
| inputs | 9 rod forces; `F_max` = 13.22 / 7.24 / 3.41 N per segment (base to tip) | `force_limits`, derived from `design_curvature = 8.5 /m` |
| max axial strain | 0.167 / 0.137 / 0.106 per segment at three rods at `+F_max` | `max_axial_strain` |
| collision bodies | 12 sub-links per segment + tip = 37 spheres, radius 1.4 r(s) (42 mm -> 21 mm) | `sublinks_per_segment`, `collision_radius_ratio` |
| plant positions | 259 (37 quaternion floating bodies) | generated SDF |
| latent (`dim_latent_space`) | 9 on every rung; trust region 4.5 | `latent_radius` |

## Measurements (laptop, float64, `tests/test_gvs_arm_model.py` and `scripts/gvs_arm/probe_*.py`)

**The vacuity trap is real and the taper defeats it.** A differential rod force on segment 0
(`cfg = [1, -1/2, -1/2, 0, ...]`) gives a Legendre-1 curvature coefficient of **2.315 /m**
on the tapered order-1 rung and **2.356 /m** on order 2; on a uniform-section variant of
the same spec both read **< 1e-15**. The constant term is 8.36 /m against the design value
8.5 /m (the rule evaluates `EI` and the rod offset at the mid-section; the equilibrium
integrates them along the taper).

**Equilibrium is unique and cheap.** Newton converges on 2000/2000 uniform draws in a
median of 3 steps to a residual of 1e-16; 50 random restarts of the backbone state reach the
cold-start equilibrium to 3e-15. Jitted, one solve costs ~14 ms and the full implicit
Jacobian `d(plant q)/d(cfg)` (259 x 9) ~18 ms on one CPU core; against central differences
the Jacobian agrees to 2e-10 relative, and the whole constraint stack under AutoDiffXd
(flow `jacrev` -> implicit `dq*/du` -> body poses -> Drake collision) to 7e-10.

**The discretization is an approximation, and here is its size.** Tip error against a
40-point Gauss reference over 200 uniform draws (`probe_convergence.py`):

| Gauss points | order 1: median / p99 / max (mm) | order 2: median / p99 / max (mm) |
| --- | --- | --- |
| 5 | 0.0106 / 0.0255 / 0.0292 | 0.0085 / 0.0202 / 0.0249 |
| **7 (fielded)** | **0.0032 / 0.0077 / 0.0088** | **0.0026 / 0.0061 / 0.0075** |
| 9 | 0.0013 / 0.0030 / 0.0035 | 0.0010 / 0.0024 / 0.0030 |
| 12 | 0.0004 / 0.0010 / 0.0012 | 0.0003 / 0.0008 / 0.0010 |
| 16 | 0.0001 / 0.0003 / 0.0004 | 0.0001 / 0.0003 / 0.0003 |

Order 0 is exact at every order (constant strain, as it must be). At the fielded 7 points
the worst tip error is two orders of magnitude below the 1e-3 m task gate; orientation
error rounds to 0.0000 deg throughout.

**Containment.** The tapered sphere union contains the rod's surface with 2.2 / 2.0 / 2.0 mm
to spare (orders 0 / 1 / 2) over 30 random draws and five box corners; the spacing margin
the spec computes at the worst sub-link is 4.8 mm.

**Self-collision.** 156 live sphere pairs per rung, all at least two segments apart; the
straight arm reads 0.96 on the collision penalty and a fully curled one reads above 1.
Uniform draws self-collide at ~0.25%.

**Pipeline smoke** (untrained chart, pose task, 2 targets x 1 guess, 20 s): the sampler,
shelf-contained targets, both arms, `verify()` and the summary all run;
`results/gvs_pushrod9_o1/benchmark/smoke_pose/`. Says nothing about solve quality.

**Datagen rate** (`probe_datagen_rate.py`): see the runbook; the number sizes
`DATASET_SIZE` and the job wall time. Measured under a peer's load at nice 19, so an upper
bound.

## What is queued and what is not

Built and tested locally: spec, model, generated SDF/scenes, jrl shim, registration seam,
four programs, driver, probes, stage `GVS` in the manifest generator, the chained-dataset
path and the preflight job. **Nothing has been submitted.** Data generation waits for the
CPU nodes to free up and for Thomas's go-ahead; training and evaluation are out of this
session's scope; the FK surrogate (`--fk learned`) is a cluster job that has not been run.

## How this compares to LOInK, once

LOInK's soft experiment (arXiv 2609.21275 sec. VII): SoRoMoX's planar HSA, 3 segments x 0.1 m,
two actuators per segment (6 inputs), task `[px, py, theta]`, forward map "the evaluation
of the complex SoRoMoX model" simulated to equilibrium, 2 x 10^6 uniform input draws, IKFlow
trained on the same data, cost `||u||^2`. Ours: spatial, 3 x 0.267 m, three push-pull rods
per segment (9 inputs), 6-D pose and grasp tasks in a shelf scene with collision
constraints, static root-find instead of a dynamic rollout, a tapered GVS backbone where
theirs is constant strain, and a joint-space optimization baseline they do not run. The
contribution is orthogonal (Thomas: wrapping a network in an optimization problem), so the
setups need not match; the differences are stated here and not engineered away.
