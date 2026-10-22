# The analytic chart and the non-rigid / non-algebraic robots

The long form of a part of `CLAUDE.md`, moved here verbatim on 2026-10-18 to bring `CLAUDE.md` under
40k characters. `CLAUDE.md` keeps the verdicts and rulings; this file keeps the mechanisms, numbers and
history behind them. "Above"/"below" and section names refer to the old single-file layout, now split
across `docs/notes-*.md`. A reference elsewhere to "CLAUDE.md's ... section" means the matching notes file.

## The analytic chart: eight branches, and the clip ladder

The Panda map's discrete set is three binary choices — wrist (B), shoulder (C), elbow (A) — and it
historically charted only A = +1. The missing half is a *single sign*: negate both triangle angles
`O2O4O6` and `O2O6O4` (elbow relations `q3 = theta + q3_add - 2*pi` for A = +1, `-theta + q3_add` for
A = -1, partitioning at `q3 = -0.467`). `ProgramOptions.analytic_branches` selects the chart (default
4); `gc(q, branches=3)` recovers all three indices. The iiwa's Faria map charts all eight by GC signs,
so analytic4-vs-analytic8 is a **Panda-only** experiment.

**The old "residual 0.6%" was the arccos clip, not the chart** (2026-10-12): the same map at clip
1e-6 reproduces 99.37% of uniform configurations to 1e-3, at clip 1e-8 100%. Uniform round trip at
clip 1e-8 is **99.90% at 1e-6 and 100% at 1e-2 on the Panda's 8 branches, 99.95% / 100% on the
iiwa** (`tests/test_analytic_ik.py` prints the curves). q6 is also now wrapped by 2pi into its window
(seam mid-infeasible-arc), as pullback-metrics' header does. arXiv:2503.03992 is not needed for it.

**The clip ladder (Thomas, 2026-10-12: measure both).** The map clips every arccos/arcsin argument
into `+-(1 - analytic_clip_stepback)` with zero derivative; the `ReachabilityConstraint` rows hold the
unclipped arguments inside `+-(1 - analytic_reach_stepback)`, so at any point the verifier passes the
clip is inactive (`CheckClipCannotEscapeGate`: reach - clip >= the verifier's slack, raised before a
grid). Those rows exclude a band ~sqrt(2 * reach) rad wide around every seam: at ../codebase's
**1e-4 / 1e-3** (default; tag `ANALYTIC`) that is 15.8% (Panda 8), 11.8% (Panda 4) and 8.6% (iiwa)
of uniform configurations; at the tight **1e-8 / 1.1e-4** (tag `ANALYTICT`) 5.6 / 4.3 / 3.0%. Both
are fielded. The arguments come from the SAME pass as `q` and are read through the `QAndPose` memo,
and the pose-target and mug rows are linear constraints on the pose variables (../codebase's `PinVars`
default encoding). SNOPT satisfies linear constraints in its first minors, so under the paired start
iterate 0 jumps to the target pose with psi still at the start's value. **At 480 cells that is not a
defect**: iiwa pose paired SNOPT is the analytic arm's best row (448, against IPOPT 376; as
../codebase's Arm on a Table, 89.7 v 29.5), Panda pose paired SNOPT 212 against IPOPT 236 (the 4-cell
smoke's 0/4 was noise). ../codebase's nonlinear pin (`PIN_AS_NONLINEAR`) was measured only on its
Bimanual row and dropped (Thomas, 2026-09-22); not fielded here.

**Two backends for the map, both measured (Thomas, 2026-10-13).** `analytic_backend="cpp"`
(`cpp/`, ctypes, one g++ call; built on the cluster by `cluster/build_cpp_job.sh`, a stale build
refused by its source hash) equals the Python AutoDiffXd path bit for bit and the float path to
~4e-12 (numpy's BLAS), so IPOPT trajectories are not bitwise (iterations equal on 44 of 108 local
cells, 5 verdicts differ: the per-cell instability any setting change shows). Its ms/it is level
with joint space's (2.2-4.3 against 2.1-2.9 on the same cells; Python 4.7-9.2). Stage ANALYTIC_CPP
fields it on the default ladder's IPOPT / SNOPT rows; everything else runs the Python reference.

**`analytic8` against `analytic4` is the unbalanced-bundle pathology, both signs intact.** Under
`paired` the 8-branch chart wins the grasp task (418 against 376) because it can represent starts the
4-branch chart forfeits; under `native` it loses badly on both tasks (387 against 450, and 155
against 259 on the pose task), because a uniform draw over eight branches lands in the narrow
near-limit bundles half the time against roughly 10% of configuration-space volume. A genuine finding
about unbalanced discrete solution bundles in optimization-IK, and exactly what the column was added
to expose. **The whole `analytic4` disadvantage is start coverage**: drawing `q_init` by rejection so
it falls only in the four wide bundles, the two charts become identical (59/60 and 59/60 grasp, 34/60
and 34/60 pose, same iteration counts). **Nothing about the near-limit bundles makes the *solve*
harder; they are simply configurations that arm cannot be given.**

## The soft PCS arm: a robot whose configuration is not the plant's position vector

Nothing remains open: the learned forward model was closed as not worth it (2026-09-30,
below). Tables: **`docs/soft-arm-ladders.md`**. Operations:
**`cluster/SOFT_ARM_RUNBOOK.md`**.

**The name is deliberately "the soft PCS arm", never "the soft arm"**: other soft-arm models (a GVS
arm) are being added, so "the soft arm" is no longer a unique referent. A spatial **Piecewise
Constant Strain (PCS)** continuum arm, defined with **SoRoMoX** and modelled on the soft experiment
in **LOInK** (arXiv 2609.21275) -- whose baseline, IKFlow, is what we field, not their BiLipNet
method. No closed-form IK, so a two-way comparison like the iiwa's; unlike either rigid arm its
**configuration is strain**, the first robot here whose configuration is not the plant's position
vector.

| rung | segments | strains per segment | DOF | decision vars | plant positions |
| --- | --- | --- | --- | --- | --- |
| `soft9` | 3 x 0.2667 m | kx, ky, sz | 9 | 24 | 217 |
| `soft12` | 4 x 0.2000 m | kx, ky, sz | 12 | 30 | 231 |
| `soft16` | 4 x 0.2000 m | kx, ky, kz, sz | 16 | 38 | 231 |

Total backbone length is fixed at 0.800 m and the strain limits are identical on every rung, so the
accumulable bend is the same and the rungs differ ONLY in redundancy -- which is what makes the DOF
ladder a measurement rather than three robots. `soft12` is primary. `dim_latent_space` is the rung's
own DOF and is not a free choice: `InvertFlow` writes `x[0, :num_arm_dof]` into a buffer of width
`network_width`, so a narrower latent is a buffer overrun. Trust region follows `sqrt(dim) + 1.5`.

**Normalized strain coordinates, symmetric about zero.** Configuration is `strain / limit` in
`[-1, 1]`, `sigma_z` the elongation so zero is the straight unstretched rod. Symmetry is FORCED by
ikflow, which bakes `1 / max(|lo|, |hi|)` per coordinate into its first `FixedLinearTransform` -- a
pure scaling with no offset, so an off-centre coordinate hands the flow input its first layer cannot
recentre. Normalization is also what keeps `correction_bound = 0.1`, the joint-centering `w * I` and
the trust region meaning one thing across coordinates of incompatible units. A **stated adaptation
for the write-up**. Limits (`|kappa| <= 8.5 /m`, `|sigma_z| <= 0.3`) were fixed from plausibility
BEFORE any acceptance rate was measured and are part of the robot.

**Our torch map IS SoRoMoX's model, and it is CHECKED.** Over 256 configurations per rung,
`g_soromox = A @ g_ours` with one constant `A` to **1.1e-15**; `A` is exactly `R_y(90 deg)` because
SoRoMoX runs the backbone along -x where we run +z. The conjugated form does NOT fit, so the two
disagree about the BASE frame only. The check is a **committed golden file**
(`tests/data/soft_arm_fk_golden.npz`), not a live import -- a test that runs only where SoRoMoX is
installed is one disposable venv on one laptop, and a check that silently stops running is
indistinguishable from one that passes. Its metadata records that JAX ran in float64, because **JAX
defaults to float32 and a 1e-12 claim against a float32 oracle is a fiction**.

**The Drake discretization is EXACT.** A segment carries a constant twist, so
`exp(xi*L) == exp(xi*L/K)**K` identically; K buys collision resolution and nothing else (measured
agreement across K = 1..20 to 2.7e-15). Each sub-link is a body no joint references, which SDFormat
makes a quaternion floating body -- chosen over a prismatic/revolute chain because it has no gimbal
lock, one body per sub-link instead of six, and `+-inf` limits so a sampler over plant limits fails
LOUDLY with nan rather than quietly.

**CAPSULES HANG DRAKE'S PROXIMITY ENGINE** -- a 6-body capsule model with ZERO candidate pairs did
not complete one `MinimumDistanceLowerBoundConstraint` evaluation in minutes; the identical sphere
model evaluates in microseconds. So collision geometry is spheres, as the tree's own
`iiwa14_spheres_cylinders_collision.urdf` already is: 0.03 ms float, 0.07 ms AutoDiffXd, 1.3 ms
curled into self-collision. **The sphere union IS the robot's geometry**, declared rather than
approximated, so the constraint is exact on the robot as defined; what must then hold is one-sided
containment, measured on the rod's surface at the configuration box's corners (worst-case clearance
3.5-4.2 mm).

**Configuration vs plant positions.** `QAndPose` memoises `(q, pose, cfg)` and still returns
`(q, pose)`; `Config(vars)` reads the third slot from the same memo, so a constraint row costs no
second network pass. `ConfigLimits()`, `ConfigToPlantQ()` and `SampleConfiguration()` default to the
plant's limits, `PadQ` and a uniform draw. On both rigid arms the default `cfg` IS `q`, the same
object, so the separation is bit-identical there by construction.

**Two Jacobians, each in its own regime, and a 68x.** `d(plant q)/d(vars) = dP/dcfg @ dflow/dvars`,
with the flow's Jacobian left exactly as the rigid arms compute it. Composing the map inside
`MakeFlowInference` would have made one reverse pass of 231 outputs against 30 inputs and silently
voided the closed measurement that reverse mode wins on shape. The map is differentiated in FORWARD
mode and compiled: `jacfwd` 10.7 ms eager -> **0.158 ms compiled (67.9x)**, where `jacrev` compiles
*worse* (6.3 -> 15.2 ms). Tied to the same `--compile` switch as the flow's.

**The base must be welded at the world origin.** `CalibrateFlowFrame` compares a WORLD-frame scene
pose against the flow's BASE-frame pose and requires a constant offset, which holds only at
`X_W_base = I`. Both rigid arms are welded at the origin, which is *why* the check has always passed.
A plinth would break it by its height and would more quietly hand the network a conditioning pose in
the wrong frame -- the failure that collapsed the Panda pose task to 10/60. Measured here: `X_ee_flow`
is the identity with spread 5e-17. **And welding the gripper REORDERS Drake's bodies** --
`soft_tip_link` moves from slot 168 to slot 0 -- so the slot map is read from the plant rather than
assumed from SDF declaration order.

**No fork of `jrl`, no edit to the vendored ikflow fork.** `SoftArmRobot` subclasses `jrl.robot.Robot`
(required: `IKFlowSolver.__init__` asserts it) and overrides `__init__` without calling `super()`,
which would demand a URDF and klampt; `name` is a class attribute because `get_robot` compares it on
the class. `sample_joint_angles_and_poses` is overridden rather than inherited -- ours is the same
batched torch map the solver differentiates, at 16 us/config including the self-collision screen.
Everything else jrl exposes raises `NotImplementedError` naming why. Every tensor is built explicitly
on the CPU: `jrl.config` calls `set_default_device` AT IMPORT.

**TWO POLE SCREENS, TWO DOMAINS -- do not quote them interchangeably.** The in-training callback had
the iiwa's constants and drew position and orientation INDEPENDENTLY; `soft12` has no torsion, so its
tip orientation is not free given tip position and nearly every sample was unreachable. On
`soft12__n6__step620000` that reads `pole/max` 2.4e8 against the standalone in-distribution screen's
**5.44** at a stricter threshold -- eight orders, and a statement about unreachable poses rather than
about the chart. **Quote the standalone number.** Fixed 2026-09-28: the fork's domain is now
injectable with its defaults untouched, and `scripts/training/ikflow_entry.py` supplies this
project's convention for soft rungs only.

**The primary chart is `soft12__n6__step620000`.** `frac_gt_threshold` is 0.0 at every checkpoint on the
in-distribution screen.

**Acceptance is HIGHER than the rigid arms'** (grasp 1.06-1.40%, pose 0.44-0.73% of collision-free
draws, against 0.55-0.68% and 0.23-0.37%), so this robot is about twice as easy to place in a
compartment. `P(trip)` at the fielded `MAX_CONSECUTIVE_REJECTIONS = 50000` is 0 on every rung, task
and inset. Collision-free fraction rises monotonically with DOF because the 3-segment rung bends
furthest per segment and self-collides most.

**Do not build datasets for several robots CONCURRENTLY.** Three datagen jobs launched together;
soft9 exited 1 after 6:21 with its five files already byte-perfect on disk, because ikflow's
end-of-run summary scans EVERY dataset in the shared cache and read a sibling's half-written tensor.
`build_dataset_job.sh` writes its `.DONE` sentinel only on RC == 0, so the dataset looked unfinished
when it was complete, and `train_flow.sh` hard-fails without the sentinel.

### What the four stages measured

**Every grasp number in this subsection ran on the defective wsg scene (the mug-handle trap) and is
void; the pose numbers stand.** The record's soft PCS rows are stage REMEASURE's: grasp native 480 v
472 (IPOPT) and 478 v 377 (SNOPT) learned wins, grasp paired 477 v 472 a tie and 350 v 377 a
joint-space win; the ladders' grasp rows were not re-measured.

Stages are SEPARATE from `ADOPTED_RUNGS` because the status quo was accepted work and this robot was
not; the chart ladder is **benchmarked, not merely screened**, since neither intrinsic screen predicts
cells; the fielded rung is pre-registered at `n6` before any cell is read. DOF rungs are **not paired
cell-for-cell** -- each draws its own grid, so they are compared by target-level success rate with a
bootstrap CI, and McNemar does not apply across them.

**The chart ladder is FLAT** -- learned wins 10 of 12, ties 2 with the iteration budget lifted, no rung separates from another (spread
3-10 cells of 480 across n4/n6/n8). `n8` sits above the ~1e7 runaway band and does NOT degrade, where
an above-ceiling rung is strictly worse on the iiwa. That agrees with the in-distribution screen
reading 0.0 everywhere: **this robot has no runaway population, so the gain-ceiling criterion is
satisfied vacuously and the ladder has nothing to measure.** A negative result about the robot, not a
refutation of the selection rule.

**The DOF ladder measures the BASELINE, not the formulation** -- learned wins 9, ties 2, loses 1 with
the iteration budget lifted (stage ITCAP; 11 / 0 / 1 as first measured). On
pose native the learned arm is at the ceiling (478/477/480 across 9/12/16 DOF) while joint space
climbs **280 -> 334 -> 441**: extra redundancy is worth 161 cells to the arm that has headroom, and
the learned arm has none left to show it in. The one loss,
`soft16` pose paired, is now ESTABLISHED (407 v 441, p = 7.7e-04, 0 cells at the budget).

Stage SOFTCAP (NLopt grasp cap ladder) ran on the defective scene and is void, and `soft12_n6`'s
exact two-stage reproduction: `docs/soft-arm-ladders.md`.

**One caveat that must travel with this robot's runtime table.** The joint-space arm here is not
free: on the rigid arms its `VarsToQ` is the identity, where here every iteration costs a full
33-body kinematics-and-constraint evaluation. Measured 4.6-11.8 ms/it against roughly 2 ms on the
rigid arms, while the learned arm's 18-56 ms/it is what it costs there too. So the per-iteration
premium is 2.5-6.0x rather than 10-13x **because the baseline got more expensive, not because the
learned arm got cheaper** -- and joint space is still faster per solved problem on every IPOPT and
SNOPT row. The learned arm does win on ITERATIONS on grasp (82-92 against 123).

**Learned forward kinematics is CLOSED as not worth it, on BOTH soft arms (Thomas,
2026-09-30).** Do not fit a surrogate, run stage SOFTFK, or field `--fk learned`.

- **It cannot change who wins.** The chart outputs a configuration, and every constraint row of
  BOTH arms goes through FK of it. So a surrogate is a control that moves both arms alike; its only
  effect on the comparison is its own approximation error.
- **It weakens the paper's main claim.** Making FK cheap makes a joint-space solve cheap again, and
  the learned arm's per-iteration premium returns toward the rigid arms' 10-13x. That lets a
  time-matched, restart-enabled joint space fit several tries into one learned solve. An expensive forward model is what keeps the comparison fair to the
  learned arm, so it is kept.
- **It buys no realism on the PCS arm**, whose exact map is already a cheap closed-form torch
  function.
- **The deployed-hardware argument** (only learned FK exists there) applies to both arms equally
  and is not this paper's contribution.

The hook stays in the tree, unused, as refuted remedies' knobs do: `fk="learned"`,
`src/soft_arm/fk_surrogate.py`, `scripts/soft_arm/train_fk_surrogate.py` and
`cluster/fk_surrogate_job.sh`. `verify()` still re-measures every solution on exact kinematics
through `VerificationQ`.

## The GVS push-rod arm: a robot with no closed-form forward model at all

**MERGED TO MAIN 2026-10-05 from branch `gvs-actuated-arm` (closed), outside the campaign of record;
its stage REMEASURE rows (2026-10-09) are reported beside the record.** Numbers:
`docs/gvs-arm.md`; tables `docs/gvs-arm-tables.md` (`scripts/report_gvs.py stage`). Operations: `cluster/GVS_ARM_RUNBOOK.md`. **It does NOT replace the soft PCS arm** (Thomas, 2026-10-05, on stage GVS: *"the GVS arm doesn't
help our story. We can still merge it into main, but it certainly doesn't replace the other soft
arm"*). The PCS rows stay in the record, and GVS is merged as a measured robot outside it.

**What it is.** A spatial continuum arm driven in ACTUATION SPACE: three segments of a **tapered**
(30 mm -> 15 mm) Geometric Variable Strain rod, strain a Legendre polynomial of order 0/1/2 per
segment (rungs `gvs_pushrod9_o0/o1/o2`, primary `o1`), **three push-pull rods per segment at 120
degrees**, each routed at 0.7 r(s) and acting only within its own segment. The configuration of every
formulation is the **nine rod forces**, normalized to `[-1, 1]` (`F_max` = 13.2 / 7.2 / 3.4 N per
segment, derived from one stated rule: a differential rod force reaches the PCS arm's 8.5 /m at the
segment's mid-section). The forward map is **SoRoMoX's own GVS model solved to static equilibrium**
(`src/gvs_arm/model.py`: Newton via optimistix, implicit differentiation, `jax.jacfwd` through the
body poses), no gravity (a spec field, off, stated). LOInK's soft experiment is the reference; ours is spatial, 9 inputs, with an optimization baseline they
do not run, and Thomas rules the setups need not match: *"our contribution is orthogonal to LOInK."*

**SoRoMoX IS the model, not an oracle.** Thomas: *"the point is to use that repo."* No torch
re-derivation, no golden file; `soromox` + `jax[cpu]` + `optimistix` are runtime dependencies of the
project venv (CPU jaxlib only -- the flow owns the GPU). What is tested is the WRAPPER: conventions,
the rod input's sign, the implicit Jacobian, convergence, uniqueness. The PCS arm's torch map is the
exception that motivated the old pattern, not a precedent.

**Kinematic redundancy is in the inputs, enforced, and MEASURED.** Nine forces against a 6-D
pose task (Thomas: *"at least one degree of kinematic redundancy"*); `GvsArmSpec.__post_init__`
refuses a spec with `ninputs <= 6` and a test pins it. That is arithmetic; the kinematic claim
is that the 6 x 9 spatial task Jacobian has rank 6, which it does on **400 of 400** uniform
draws, and that the resulting 3-D null space can be
TRAVELLED: a corrected walk holding the tip pose to 0.1 mm and 0.06 deg covers a median
**1.38** of the 2.0-wide normalized force box, and **every walk stops at the force box, not at
a singularity**. So the self-motion manifold is bounded by actuation limits rather than by
kinematics -- the same saturation that makes the joint-space baseline fail. `dim_latent_space = 9` on every rung: the order ladder
changes the forward model's fidelity, not the problem's width.

**THE VACUITY TRAP, and why the backbone tapers.** With a uniform section, a straight-routed rod
applies a uniform moment and the generalized stiffness is diagonal in the Legendre basis, so every
coefficient above order 0 is EXACTLY zero at equilibrium -- an order ladder on such a rod would measure
nothing. `EI(s) ~ r(s)^4` is what makes the strain variable. Measured and pinned by a test: a
differential rod force gives 2.3 /m of Legendre-1 curvature on the taper and 1e-16 on a uniform rod.

**Conventions are SoRoMoX's, deliberately.** Strains are `kappa_y, kappa_z, sigma_x` (local x is the
backbone, `sigma_x = 1` the straight reference); with its default upright mounting the backbone runs
along world +z at the origin, where every scene welds a robot, so nothing rotates a frame. Two
consequences: the tip FRAME `gvs_tip` is declared in the SDF as `R_y(+90 deg)` on the tip body so its
z runs along the rod (the gripper weld and the flow's conditioning pose are the same geometry as on
every other robot), and body quaternions are canonicalised to `w >= 0`. **Tip orientation
given tip position is 3-dimensional here despite no torsional strain** (nearest-neighbour
shrink 1.43x per tripling against 1.44x for a 3-D set) and covers a LARGE fraction of SO(3). The backbone
tangent covers 86-100% of the sphere; what is restricted is the ROLL about it, ~50-103 deg of
360. So "no torsion, so orientation is not free" was an inference, right about the mechanism
and wrong about the size. It is still not all of SO(3), so the pole screen draws
in-distribution poses.

**The discretization is an approximation and its size is measured**: at the fielded 7 Gauss points
per segment the tip error against a 40-point reference is 0.009 mm max on order 1 and 0.0075 mm on
order 2 (order 0 exact) -- two orders below the 1 mm task gate. The sphere union follows the taper
(radius 1.4 r(s), 37 bodies, 259 plant positions) and contains the rod with ~2 mm to spare.

**The forward model is a root-find, and that is the per-iteration price**: ~14 ms per evaluation and
~18 ms per 259 x 9 Jacobian on one CPU core, beside the flow's ~17 ms. Newton converges in a median
of 3 steps on 2000/2000 draws; random restarts reach the same equilibrium to 3e-15, so the cold start
from the straight arm is a deterministic map and not a selection among alternatives. A draw that
does not converge is REJECTED AND COUNTED in the dataset sampler, never written. The whole AutoDiffXd
chain (flow `jacrev` -> implicit `dq*/du` -> poses -> Drake collision) matches central differences to
7e-10. XLA sizes its pools from the process's CPU affinity (jax 0.11 ignores the threads flag), so
`run_items.sh` sets `GVS_ARM_XLA_THREADS=1` and the dataset builder pins each worker to its own CPU
slice before JAX loads.

**The joint-space arm's failures are force saturation, not a wiring fault.** From the target, from
straight and from random starts it converges to 1e-8; when it fails IPOPT reports local infeasibility
with rod forces on the +-1 box. A property of the problem to report, like every other baseline's.

**What stage GVS measured (2026-10-05; 16 logical runs, IPOPT and SNOPT, PROCS=2).** Its grasp
rows ran on the defective wsg scene and are void; its stage REMEASURE rows, reported beside the
record in `docs/status-quo-tables.md`, replace them (IPOPT 3 learned / 1 tie, SNOPT 2 / 2: grasp
native 480 v 467 and 476 v 366 are now learned wins, grasp paired 470 v 467 and 343 v 366 ties).
**Stage GVSO2 (2026-10-22) measured `o2` the same way** (REMEASURE's items, rung swapped, PROCS=2):
the same verdict on every row, so 10 learned / 6 ties / 0 losses over both rungs; no row
budget-bound; no rung separates. Tables, time-matched columns and acceptance:
`docs/gvs-arm-tables.md`.

**What is queued and what is not.** Stage `GVS` (`cluster/gen_manifest.py`: two trained rungs x two
experiments x two protocols x **IPOPT and SNOPT, no NLopt** -- Thomas, 2026-10-02: *"it's a waste of
time"*) is status-quo-shaped. **Its two columns -- time-matched joint space and iterations
on mutual successes -- and their predictions are PRE-REGISTERED in `docs/gvs-arm.md`** (written
before any trained chart existed); `scripts/report_gvs.py` reads both stages.
`o0` is spec-only. **Learned FK is closed for this robot too** (the soft PCS arm's section says why): its
exact forward model is the expensive one, which is what made it the candidate for a
time-matched claim, so the primary rows keep it. `scripts/gvs_arm/make_untrained_chart.py` writes a gitignored untrained chart so the
pipeline can be smoked without training, and was: both tasks, both arms, end to end.

## The screw-joint arm: a robot no algebraic method can chart

**MERGED TO MAIN 2026-10-02 from branch `non-analytic-arm` (closed). All five charts trained, stages
SCREW / SCREWCHART / SCREWPITCH / SCREWCAP measured**; results below.
`cluster/SCREW_ARM_RUNBOOK.md` holds the operations and the screens. **The record's screw rows are
stage REMEASURE's** (IPOPT and SNOPT) and REMEASURE_NLOPT's (NLopt).
Stage SCREW's grasp numbers below ran on the defective wsg scene and are void; its pose numbers stand.

**The identifiers say `screw` everywhere, as the prose does** -- robot `screw7_p*`, `src/screw_arm/`,
stages SCREW / SCREWCHART / SCREWPITCH. Until 2026-10-02 they
were spelled `helix`; the rename (Thomas, 2026-09-28, *"purge mentions of a 'helix arm'
everywhere"*) moved the code, regenerated the models and manifests from the renamed generators
(byte-identical modulo the name) and renamed the downloaded results. The cluster tree the campaign
ran in, `~/learned-ik-helix`, was **deleted on 2026-10-02** (Thomas's call) after its screens were
pulled down; the wandb runs keep the old name. The five final charts, under the new names, live in
`~/learned-ik`. `git log --follow` crosses the rename.

**Why the robot exists.** An analytic column needs `FK(q)` to be *algebraic*: for a revolute arm
every entry is a polynomial in `(cos q_i, sin q_i)`, and `c^2 + s^2 = 1` turns IK into a polynomial
system elimination solves. A **screw joint** rotates by `q` *and* translates
`pitch * q / (2*pi)` along the same axis, so `q` enters both trigonometrically and linearly and is
algebraically independent of `e^{iq}` (Lindemann-Weierstrass). Abban, Li & Schicho
(arXiv:1312.1060) state it for linkages: algebraic-geometry methods "have failed so far ... because
of the presence of some non-algebraic relations". So this is a **generality demonstration**, like
the soft arm — a robot class the algebraic baselines cannot touch, where the learned + optimization
formulation needs no change at all. Two arms, `learned,numerical`.

**The robot is INVENTED, and invented from scratch.** We could not find a 7+-DoF arm with a lone
screw joint in hardware or in any public model, and the reason is structural: a lone screw pair
carries the drive torque and the load's reaction torque through the same thread. The next section
gives the evidence. In all of public GitHub exactly two robot models use an SDFormat `screw` joint
and neither is an arm DOF. Given the arm must be invented it is invented **from scratch**, not by
perturbing a benchmark arm — which would carry a real robot's name and published identity while no
longer being that robot.

**`screw7`.** A 7-DoF S-R-S arm whose **upper-arm roll is a screw joint**: the upper arm telescopes as it
rolls, and every downstream link is offset from that axis, which is what makes the coupling
irreducible. `src/screw_arm/params.py` **is** the robot; the SDFormat model and the batched torch FK
are two renderings of it and a test says they agree, so there is no second source of truth and
nothing to drift. The screw coordinate is `±2π`, deliberately symmetric about zero because
**ikflow's first layer is `x_i / max(|lo_i|, |hi_i|)` — a pure scaling with no offset** — so a
one-sided range would land that coordinate in `[0, 1]`. Many `q3` differing by `2π` give the same
rotation at a different extension, so the solution set is richly multimodal, which is the property a
flow is supposed to capture. Reach matches the iiwa (flange at z 1.26 home, 0.89 m horizontal from a
shoulder at z 0.42), so every shelf weld, table and containment screen applies untouched.

**Four rungs, one robot: pitch `{0, 0.025, 0.050, 0.100}` m/rev, primary `screw7_p050`, fixed before
any number was read.** Every other number is shared, so the ladder is a dose-response rather than
four robots. **`screw7_p000` is a full spec, not a code path** — a control that takes a different
code path is not a control — and at pitch 0 the arm is an ordinary S-R-S arm for which the closed
form is standard, so the family contains its own degenerate, algebraic member.

**That member is NOT trained, by decision** (Thomas, 2026-09-25: *"Seems like a waste of time to
train a model for [screw7_p000]. We already have analytic arms, we don't need a specific control
example here."*). The project already fields two S-R-S arms **with** analytic columns, so a seventh
chart would spend 620k steps rediscovering that an algebraic arm is algebraic. The spec and its
dataset stay — the tests use it, and it is what makes the pitch a *parameter* rather than a fact
about one robot — but the trained ladder is the three screw rungs, and the "an analytic column
could exist here" end of the scale is held by the Panda and the iiwa. One consequence to keep in
view: with the limit box held at ±2π for comparability, the zero-pitch member's screw coordinate
covers the circle twice, so `q` and `q+2π` are the same configuration there and different
everywhere else.

**TRAP: every Drake parser silently discards a screw joint's `<limit>`.** `ParseJointLimits` is
reached only for revolute and prismatic joints, in URDF and SDFormat alike, so the plant reports
`[-inf, inf]` on that coordinate and nothing raises. The joint-limit row — the one row this robot
exists to stress — goes vacuous, the joint-space arm's box goes unbounded, and the target sampler's
`rng.uniform(lower, upper)` returns `nan` and spins for ever. `src/screw_arm/limits.py` repairs it
in the program's `__init__`, after `Finalize()` and **before `ToAutoDiffXd()`**, which takes an
independent copy that would otherwise carry the infinities for ever. Anything that builds this plant
without constructing a program must repair it itself; `scripts/probe_shelf_acceptance.py` does.

**TRAP, and it is the conditioning-frame lesson a second time: `ee_frame` must be set BEFORE
`CalibrateFlowFrame`.** The grasp program set it after `super().__init__()`, which is what the
iiwa's structure invites, so the calibration measured `between_fingers` and applied that 0.2 m
offset to the flange. Nothing raised: `frame_for_flow` falls back to `self.frame`, and the offset
**is** constant, so the constancy check passes. `X_ee_flow` must be exactly the identity on both
tasks — which doubles as a free end-to-end witness over the whole SDFormat-versus-torch chain — and
a test pins it.

Three smaller ones worth not rediscovering. **Capsules hang Drake's proximity engine**, so collision
geometry is a sphere union along each capsule's segment. A `<drake:collision_filter_group>` named
after its own link raises "Non-unique name detected 2 times", hence the `cfg_` prefix. And `jrl`
cannot parse or evaluate a screw joint at all, so `src/screw_arm/robot.py` follows the soft arm's
shim pattern — no `super().__init__()`, and every jrl method we do not implement raises
`NotImplementedError` naming why. `RationalForwardKinematics` refusing this arm is **not** a test:
it keys on the joint *type*, so it refuses `screw7_p000` just as readily and distinguishes nothing.

Laptop verification before queueing (FK and gradient agreement, acceptance, `P(trip)`):
`cluster/SCREW_ARM_RUNBOOK.md`.

**This robot's pole screens are IN DISTRIBUTION, and that is now measured rather than argued.** The
vendored fork's in-training callback draws its conditioning pose as a position and an orientation
**independently**, which is a fair draw only where the arm reaches most of SO(3) at a given position. The structural expectation here (7 DoF, roll-pitch-roll wrist) is that orientation
is free, but that is exactly the kind of assumption the soft arm's experience says to stop leaving
standing. `scripts/probe_orientation_freedom.py` settles it without an IK solver: hold the flange
within 5 cm of the box centre `[0.4, 0, 0.5]` and measure how far an independently drawn orientation
sits from the nearest one **achieved** there. A single such number is meaningless, being set by how
sparsely SO(3) was sampled; **the SCALING is the measurement**, since a covering radius over a
`d`-dimensional set falls as `N**(-1/d)`. Measured median degrees 24.1 / 16.8 / 11.7 at
`N` = 200 / 600 / 1800 — **1.44x per 3x against the 1.44x a 3-dimensional set predicts, with no
floor.** So the orientation set is full-dimensional, the callback's draw is reachable, and
`ScreenDomain` returning the rigid tuple unchanged is correct. **Quote this robot's in-training pole
curve directly beside the record's**, unlike the soft arm's. A lower-dimensional set would instead
have plateaued at the distance from a random orientation to it — which is the general test, not a
screw7 fact.

### Why a screw joint is a sensible thing to build

Asked for directly (Thomas, 2026-09-25), because an invented robot has to be defensible as a
*machine* and not only as a test case. Claims below were checked against vendor documentation and
patents; the three things that did **not** survive checking are named at the end, because the
tempting version of this story is more confident than the evidence.

**The pair is textbook, not exotic.** The screw pair is one of the standard lower pairs,
symbol **H**, with **one** degree of freedom — the same as R and P, imposing five constraints
between two spatial bodies (Lynch & Park, *Modern Robotics*, §2.2.1 and Table 2.1). It is the
general case of which R and P are the degenerate limits: pitch 0 is a pure rotation and pitch → ∞ a
pure translation (ibid., Def. 3.24). So `screw7`'s pitch ladder is a sweep along a standard
one-parameter family, and its zero-pitch member is the R end of it.

**Mind the pitch units; three conventions are in play.** Drake's `screw_pitch` — and this repo's
`params.py` — is **metres per revolution**, so translation is `pitch · q / 2π` with `q` in radians.
*Modern Robotics* defines pitch `h` in **metres per radian**, giving `d = h·θ` with no 2π; Pinocchio
follows that convention. Machine-tool practice adds a third trap: for a single-start thread "pitch"
equals "lead", but for a multi-start thread lead = n × pitch, and a ball-screw catalogue quantity is
the **lead**. Never copy a pitch between libraries without converting.

**Hardware realises a lone H pair in exactly two ways, and neither is sold as a robot joint.**

*As an internal element.* The **Newport Picomotor** is a genuine lone screw pair. And it shows exactly why a lone H pair is hard to use as a joint: the drive torque
and the load's reaction torque pass through the same thread, so Newport publishes a **torsional load
limit of 0.018 N·m** above which the actuator stalls, and specifies pushing against a smooth flat
pad rather than bolting a load to the tip.

*As a constrained operating mode.* A **ball screw/spline** puts a ball-screw groove and a ball-spline groove on one
shaft with two independently rotatable nuts. THK names three modes: *"rotational, linear, and
**spiral**"*. **Spiral mode is the screw pair**: drive the spline nut with the screw nut held and
the shaft advances at the screw's lead per turn.

**So the joint is buildable from catalogue parts — but the honest statement is narrower than "an
actuator with this structure exists".** Spiral mode is a *constrained mode of a 2-DoF device*: the
hardware has two independent inputs and braking one is a control choice, not a kinematic constraint
built into the pair. We are **not aware of any commercially available actuator that realises a lone
screw pair as a robot joint.** Every rotary-linear product on the market is either a 2-DoF
**cylindrical** actuator with two independent drives (LinMot's PR01 linear-rotary motors; the
ball-screw/spline SCARA quill, which patents from Epson, Fanuc, ABB, Yaskawa, Denso Wave, Mitsubishi
Electric and Nidec Sankyo all show driven by two motors) or a screw with an anti-rotation feature,
which makes it **prismatic**. Screw joints in the robotics literature are pedagogical — Lynch &
Park's RPH and HRR chains are exercises.

**That is the reportable finding, and it is why the arm had to be invented rather than downloaded.**
The joint is a standard pair, buildable, and first-class in Drake, DART, Simbody and Pinocchio;
nobody has put one in an arm. The C-pair alternative would not have served: a screw in series with a
prismatic or revolute joint **on the same axis** is a cylindrical pair, and its IK re-coordinatises
back to an algebraic problem under an invertible linear map — so it would look non-algebraic and not
be. What makes `screw7`'s coupling irreducible is that the H pair is the upper-arm **roll**, so its
translation telescopes the link it rotates about, and every downstream link is offset from that
axis.

**Three claims that did not survive checking, recorded so they do not creep back.** NSK, Hiwin and
Nook are **not** established ball-screw/spline suppliers — "spline" does not appear in NSK's
sitemap, Hiwin lists ball splines only, Nook could not be checked; only THK, NB and PMI are
confirmed. Kawasaki and Omron are **not** confirmed users of a ball-screw/spline SCARA quill, unlike
the seven makers named above. And there is **no** non-rotating-tip Picomotor variant marketed for
attaching loads — the only rotating/non-rotating distinction Newport actually sells is the 8341NF
*rotary-output* actuator, which is the opposite.

### What the three stages measured (2026-10-01/02)

Stages `SCREW` (status-quo-shaped: 2 experiments x 2 protocols x 3 solvers), `SCREWCHART` (`nb_nodes`
4/6/8 on `p050`, IPOPT) and `SCREWPITCH` (the three trained pitches at `n6`, IPOPT): 36 logical runs
of 480 cells at 180 s, seed 1, `--compile`, contained placement, Drake nightly `0.0.20260918`. **Separate stages, never entries in `ADOPTED_RUNGS`.** Tables,
with the cap and runaway columns inline: `python scripts/report_screw.py`; operations and screens:
`cluster/SCREW_ARM_RUNBOOK.md`. **The pitch rungs do not pair** (each draws its own grid), so
McNemar stays within a pitch, between the arms.

**Stage SCREWCAP re-measured every budget-bound row with the iteration budgets lifted**
(2026-10-02): the 17 rows where either arm had >= 24 of 480 cells at a budget, each regenerated from
its own builder -- same seed, grid, chart and **180 s clock** -- with `max_iter` 100000 and, for SNOPT,
`snopt_iterations_limit` too (both its INFO 31 and 32 count as the iteration cap). The clock stays
because Thomas ruled it the usability limit: *"Don't raise the wall clock timeout. If it's that slow,
it's not usable."* So the remaining failures are 180 s stops, and each verdict below is a result at the
fielded clock rather than budget-bound. Cell-for-cell table against the originals: `python scripts/report_screw.py
SCREWCAP`.

**The chart ladder: `n4` is still the WORST rung, but by less** -- the iiwa's best. It wins pose native and still loses pose paired decisively (200 against
313, p = 3.6e-15, at 0 capped cells and 246 timeouts). `n6` and `n8` agree (both pose rows won, both
grasp rows tied). `n4`'s failures are not the gain-ceiling runaway: on pose paired they end at a
median `|q|_inf` of 30 rad against ±3 rad limits, none above 1000 -- moderate out-of-limits excursions,
what its training-time validation ratio (up to 14.7, unclamped over clamped) had flagged. The
pre-registered `n6` stands.

**The pitch ladder found the GAIN-CEILING RUNAWAY on this robot, at the SMALLEST pitch.** `p025_n6`
is the one rung the lifted budget does not rescue: it still loses grasp native (363 against 420,
p = 3.5e-6, 111 timeouts), ties grasp paired (398 against 420, p = 0.053) and pose paired, and wins
pose native. **91 of its 124 original grasp failures and 158 of 188 pose-paired failures return
`|q|_inf > 1000` rad, 73 and 58 above 1e7** — the record's 1e7-1e16 band. `p050` has none and `p100`
some (55 of 132 on pose paired, a learned win, 359 against 310 lifted). Unlike the iiwa's single ray,
it is a two-joint family, shoulder pitch against wrist roll at opposite sign, not the screw
coordinate. **This is the one place a screen predicted cells**: `p025_n6` is the chart whose box
screen ended at 9.3e5 rad, 92% of its log ceiling, against 20-27% for the other two. One chart is not
a reversal of "the screen is a smoke test", but it is the first agreement, and it says the runaway is
a property of the trained chart rather than of the robot class.

**The curvilinear rail is deferred, not dropped.** `<drake:joint type="curvilinear">` parses in the
pinned build with finite limits. Two things to know on return: on each piece the map is algebraic (a
circular arc is a revolute joint about the arc centre, relabelled), so the non-algebraicity is
global rather than pointwise; and Drake's trajectory is planar and piecewise line-and-arc only,
which matches every real curved track and admits no clothoid or spline.

