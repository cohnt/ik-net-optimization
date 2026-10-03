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

**Shelf acceptance** (`scripts/probe_shelf_acceptance.py --robots gvs_pushrod9_o1`, 20000
uniform draws, inset 0.10 m): 34.8% of uniform rod-force draws are collision-free in the
hardened scene; of those, **1.70%** land in a compartment on the grasp task (0.59% of raw
draws, ~169 draws per target) and **0.91%** on the pose task (0.32% of raw, ~317 per target).
Higher than the rigid arms' 0.2-0.7% and in the soft PCS arm's range; `P(trip)` at the
fielded guard of 50000 is 0 to machine precision on both tasks, so the grid cannot trip it.

**Pipeline smoke** (untrained chart, 2 targets x 1 guess, 20 s, both tasks): the sampler,
shelf-contained targets, mug diagrams, both arms, `verify()` and the summary all run
(`results/gvs_pushrod9_o1/benchmark/smoke_{pose,mug}/`). Says nothing about solve quality.
A 12-solve diagnostic of the joint-space arm alone: from the target, from straight and from
random starts it converges to 1e-8 on 9 of 12; the 3 failures are IPOPT's "converged to a
point of local infeasibility" with rod forces on the +-1 box -- force saturation, a property
of the problem to report like every other baseline's.

**Orientation given position** (`scripts/probe_orientation_freedom.py --robot gvs_pushrod9_o1`,
17.1M uniform draws, 6004 tips within 5 cm of `[0, 0, 0.45]`): the nearest-neighbour
distance between tip orientations shrinks by **1.43x per tripling** of the sample (24.3 /
17.0 / 11.9 / 8.3 deg median at 200 / 600 / 1800 / 5400), against 1.44x for a 3-dimensional
set and 1.0 for a floor. So the reachable orientations at a fixed tip position form a
3-dimensional set even though no strain is torsional: three segments bending about
different axes compose to a rotation about the tangent. The soft PCS arm's "no torsion, so
orientation is not free" was an inference that this measurement does not support in general;
it is not re-measured here.

**How much of SO(3), in fractions** (`scripts/probe_orientation_coverage.py` via
`cluster/orientation_coverage_job.sh`, job 5782975: 8,000,000 uniform force draws on 48
xeon-p8 workers, 37,298 tips kept within 5 cm of 8 positions chosen as the densest pilot
voxels, scored against 20,000 uniform probe orientations). Dimension is not fraction, so this
is the question answered directly: what fraction of SO(3) lies within `tau` of an orientation
the arm actually reaches at a fixed tip position.

| tip position | N | within 10 deg | within 20 deg | within 30 deg | within 45 deg | median / p99 to nearest reached |
| --- | --- | --- | --- | --- | --- | --- |
| uniform control (not the robot) | 5,681 | 79.8% | 100.0% | 100.0% | 100.0% | 7.6 / 14.0 deg |
| `[0.00, 0.00, 0.45]` (the pole screen's) | 3,012 | 50.1% | 85.2% | **93.3%** | 99.0% | 10.0 / 45.2 deg |
| `[0.25, -0.35, 0.35]` | 4,797 | 39.1% | 65.0% | 79.4% | 92.4% | 12.9 / 63.8 deg |
| `[-0.35, -0.25, 0.35]` | 4,743 | 38.6% | 64.9% | 79.4% | 92.2% | 12.9 / 64.9 deg |
| `[0.35, -0.35, 0.10]` | 3,819 | 31.2% | 60.8% | 80.2% | 94.4% | 15.9 / 57.3 deg |
| `[-0.10, -0.45, 0.40]` | 5,313 | 34.0% | 50.3% | 63.4% | 81.5% | 19.8 / 80.9 deg |
| `[-0.15, -0.30, 0.55]` | 5,681 | 31.3% | 46.8% | 59.8% | 78.1% | 22.3 / 89.5 deg |
| `[0.15, -0.45, 0.45]` | 5,361 | 26.0% | 39.5% | 51.1% | 69.1% | 29.1 / 94.0 deg |
| `[0.25, 0.30, 0.60]` (worst) | 4,572 | 16.7% | 27.1% | **37.5%** | 54.2% | 41.3 / 111.7 deg |

**It is a large fraction, not a sliver: 37-93% of SO(3) within 30 deg and 54-99% within
45 deg, median 71% and 87% over the eight positions.** Three things make that readable. The
uniform control is the estimator's ceiling at these sample sizes and it saturates (100% at
20 deg), so the robot's shortfall is the robot's, not the sample's. The saturation ladder
still RISES with N at every centre (the worst runs 28.7 -> 33.7 -> 37.5% at N = 508 / 1,524 /
4,572), so every number here is a **lower bound** on what the arm reaches. And a geodesic ball
of 30 deg is only 0.75% of SO(3) by Haar measure, so "93% within 30 deg" is a statement about
a genuinely spread set rather than about a loose tolerance.

**The structure is that the pointing direction is nearly free and the ROLL is what is
restricted**: the backbone tangent covers 86-100% of the sphere within 20 deg at every
centre, while the roll about that tangent spans only about 50-103 deg of 360 where enough
samples share a tangent cell. So the torsion intuition was right about the mechanism and
wrong about the size -- a quarter turn of roll is not "not free". Coverage is best at the pole
screen's own position and worst at the far-reach corners, which is the force box saturating.
The practical consequence is unchanged: an independently drawn orientation is not guaranteed
reachable, so the pole screen keeps drawing in-distribution poses, and the benchmark restricts
itself to known-feasible initial guesses.

**Redundancy is real, not just arithmetic** (`scripts/gvs_arm/probe_self_motion.py`). Nine
inputs against a 6-D pose task gives three degrees of redundancy by construction; what makes
it kinematic is the 6 x 9 spatial task Jacobian having rank 6, which it does on **400 of 400**
uniform draws. The rotation rows are a genuine twist (`omega = 2 Im(conj(q) qdot)`), not four
constrained quaternion components, so the singular values carry units: 3.29 / 2.69 / 1.70 /
0.50 / 0.26 / 0.073 at the median, condition number median 46 and p95 142.

The null space is also TRAVERSABLE, which rank alone does not show. Walking along it with a
Gauss-Newton corrector that holds the tip pose to 0.1 mm and 0.06 deg, the arc length covered
in normalized force units is **1.38 median (1.04 to 2.77)** from starts in the inner half of
the box, and 0.44 median from starts drawn over the whole box. **All 40 walks stop at the
force box, none at a singularity or a curvature failure**, and the difference between those
two rows is the reason: the self-motion manifold is bounded by actuation limits, not by
kinematics, which is the same saturation that makes the joint-space baseline fail. The box is
2.0 wide per axis, so a median walk crosses a substantial part of it at a FIXED tip pose.

**Datagen rate.** One process through the batched JAX solve costs 12-15 ms/sample on a
cluster node with XLA unpinned -- the vmapped Newton does not spread across cores, and
`jax.pmap` over host devices is refused by lineax under optimistix -- so the dataset is
built process-parallel (`scripts/gvs_arm/build_dataset_parallel.py`, one single-threaded JAX
per CPU, ikflow's exact file layout); the runbook has the table and the launch. Two facts
found by the first cluster builds (2026-09-30): the sampler had solved every draw twice
(tip pose, then sphere centres for the self-collision screen) and now solves once, 23.5 ->
11.2 ms/sample per worker on the laptop; and the vmapped solve's peak resident memory is
about 2.05 GB + 0.7 MB per lane (kernel high-water mark 2.38 GB at batch 512, 2.73 GB at
1024), which is why 48 workers at batch 4096 were OOM-killed on a 192 GB node and the
builder now runs batch 512 with a memory-derived worker cap. **Measured on the cluster**
(48 workers, one xeon-p8 node, 25M samples): 436.8 us/sample over the node, 21.0 ms/sample
per worker, 3 h 6 min wall, peak 99.5 GB of 192, `rejected_unconverged` 0 on all 25,000,000
training and 15,000 test draws. **Order 2 costs 1.83x that per sample** (801.5 us over the node,
38.3-38.5 ms per worker, 5 h 37 min for 25M, peak 121.6 GB, `rejected_unconverged` 0) -- 27
generalized coordinates in the Newton system against 18 -- which also exposed the builder's
per-result timeout: the first `imap_unordered.next()` waits for a worker's whole 521k share,
and at this rate that share takes ~20,000 s, the old default itself (the fastest worker
finished 67 s under it). The default is now 11 h, under the wall. Both datasets are
downloaded and verified: unit quaternions to 1.2e-7, 64 random rows each re-solved against the
stored endpoints to 6.3e-8 (`o1`) and 5.8e-8 (`o2`), the float32 storage floor.

## Pre-registered columns and predictions (written 2026-10-02, before any trained chart existed)

Agreed with Thomas on 2026-09-30, and committed here before any trained-chart cell was read,
because this robot's argument rests on them. Both are computed by `scripts/report_gvs.py stage`
from the recorded cells, with no extra runs.

1. **Time-matched joint space.** For every learned cell (target t, guess g), the budget is the
   wall time the learned arm spent on that cell, success or failure. Joint space runs target t's
   8 recorded solves in a random order and stops at the first success. The learned cell's
   counterpart succeeds if that first success lands within the budget; the figure is averaged
   over 2,000 random orders. Printed beside it:
   - the STARVED share: cells whose budget exceeds all 8 joint-space solves, where the true
     multi-start figure would be higher;
   - the any-of-8 ceiling.

   This is an analysis of recorded guesses, not multi-start machinery. It uses
   `scripts/report_time_matched.py`'s own `row_stats`, which reproduces the record's figures
   (soft PCS arm under IPOPT: 71.9 / 69.2 / 74.8 / 90.6%).
2. **Iterations on mutual successes**: the median, over cells BOTH arms solved, of the per-cell
   learned/joint-space iteration ratio. Per-arm medians over each arm's own successes compare
   different cells.

Reported beside them: the per-EVALUATION premium (ms per network-and-map evaluation,
`solver_seconds / eval_counts["map_jacobian"]`, per arm and as a ratio) and ms per iteration.

**Predictions:**
- The per-evaluation premium is ~1.3-1.5x, because both arms pay the equilibrium solve (~14 ms
  plus ~18 ms per Jacobian on one core) and the flow adds ~17 ms.
- Learned success SURVIVES the time-matched column on every row where single-start joint space
  leaves room.

**Reference points, IPOPT:**
- Rigid arms: only Panda contained grasp survives (learned 98-99% against matched 54-68%).
- Soft PCS arm: grasp 98-99% against 69-72% and pose native 99.4% against 74.8% survive; pose
  paired is a tie (90.8% against 90.6%).

That arm's own per-evaluation premium, read from SOFT12 by the same reader, is 2.95-3.45x under
IPOPT and 2.25-2.43x under SNOPT.

**Solvers: IPOPT and SNOPT, no NLopt** (Thomas, 2026-10-02: *"it's a waste of time"*), each at its
adopted configuration. These rows replace the soft PCS arm's IPOPT and SNOPT rows.

## The go/no-go pre-check (before any training)

Thomas agreed this gate on 2026-09-30. Two numbers decide whether training can show anything,
and neither needs a trained chart:
- **room to win**: single-start joint-space success on stage GVS's own cells (stage `GVSJS`);
- **the premium**: measured with the untrained `n6` chart (stage `GVSPREM`). A chart's cost per
  evaluation is set by its architecture, not its weights.

Proposed thresholds: proceed if joint space leaves real room (below ~90% on the rows) AND the
premium is below ~2x. `cluster/calibrate.sh` measures workers per node for this robot in the same
allocation, with both arms. Read all three with `scripts/report_gvs.py precheck`.

### Pre-check results so far (cluster, 2026-10-02)

**Contention, and why every GVS stage runs at PROCS=2.** `calibrate.sh` gpu-procs (o1 grasp,
IPOPT, both arms, pinned, the same 8 cells at every level) gives this, with iterations at a
median of 62 on every level:

| workers per node | learned ms/eval | joint-space ms/eval |
| --- | --- | --- |
| 1 | 37.8 | 23.3 |
| 2 | 34.8 | 22.9 |
| 4 | 38.5 | 25.8 |
| 8 | 42.1 | 28.9 |

The joint-space arm is pure JAX on the CPU, so it degrades about twice as fast: +24% at 8
workers against +11% for the learned arm. At the record's 8 workers that would inflate every
joint-space solve relative to a learned one and bias the time-matched column toward the
learned arm by ~12%. Thomas chose **PROCS=2** (2026-10-02), where both arms are within ~2% of
uncontended. GVS runtime columns are therefore at a different PROCS from the record's rows;
timing is not compared across robots.

**The per-evaluation premium is 1.17-1.54x** (stage GVSPREM, untrained n6 chart, 8 cells per
row, at PROCS=8): IPOPT 1.54 / 1.51 (o1 grasp / pose) and 1.36 / 1.35 (o2); SNOPT 1.27 / 1.29
(o1) and 1.17 / 1.20 (o2). That is inside the predicted 1.3-1.5x and under the ~2x threshold.
Order 2's premium is smaller because its equilibrium solve is costlier while the flow is the
same. Uncontended, the calibration's o1 grasp ratio is 1.62x, so contention compresses the
premium as well. GVSPREM2 re-measures it at PROCS=2.

## What is queued and what is not

Built, tested and on the cluster: the robot and its programs, both datasets (25M + 15k,
`rejected_unconverged` 0), the tree `~/learned-ik-gvs`.

Stage GVS (128 items: 2 rungs x 2 experiments x 2 protocols x IPOPT/SNOPT x 8 shards) has two
derived forms on the same grid, and `gen_manifest.py --selftest` checks every grid argument
against it:
- `GVSJS`: joint space alone, one protocol;
- `GVSL`: learned alone, to be joined to GVSJS. Fielded only if GVSPREM shows that a joint-space
  solve costs the same wall time in either kind of job.

GVS workers run pinned to their own physical cores (`cluster/cpu_slice.py`), because unpinned
JAX processes grow ~5 threads per visible CPU.

The learned forward model (`--fk learned`) is CLOSED as not worth it (2026-09-30; `CLAUDE.md`,
the soft PCS arm's section). This robot's exact forward model is expensive, and that is what keeps
a time-matched joint-space baseline from fitting cheap restarts into one learned solve.

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
