# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Keep this file compact.** It is read at the start of every session, so its cost is paid
repeatedly. When a measurement supersedes an older one, **delete the old table and keep the
conclusion**; git history holds the numbers.

## Project

Research code (ROS-style package `combining_kinematics`) for solving inverse kinematics with a
**normalizing-flow IK network (IKFlow) placed inside a Drake optimization program**, so that
collision avoidance, joint limits and task costs are imposed on the network's *output*. The point
of the repo is the three-way comparison of formulations for the same IK problem:

| Formulation | Decision variables | `VarsToQ` |
| --- | --- | --- |
| **learned** (`Panda/IiwaIKProgram`, `...MugProgram`) | conditioning pose `c` (xyz+rpy, 6), latent `z` (`network_width`), `correction` (7) | forward pass of the IKFlow model + `correction` |
| **numerical** (`...ProgramNumerical`) | joint angles `q` (7) | identity |
| **analytic** (`...ProgramAnalytic`) | end-effector pose `xyz_rpy` (6) + redundancy parameter `psi` (1) | closed-form S-R-S IK (`src/*_analytic_ik.py`) |

All three go through the same `IKFlowProgram` machinery, so a change to constraints/costs affects
all of them. `workshop-paper-draft.pdf` is the write-up — an early rough draft, orientation only,
not a replication target. The sibling repo `../codebase/` is the analytic-vs-numerical project this
one builds on (`scripts/iiwa/iiwa_collision.py` imports from there); treat it as read-only.

## Environment and running

No package manifest. Dependencies: `pydrake` from a local Drake build (`~/opt/rlg/drake-build`,
already on `PYTHONPATH`; provides IPOPT and SNOPT), `torch`, `numpy`, `tqdm`, and `ikflow` + `jrl`
(Jeremy Morgan's packages — **not** in the default env; check `python -c "import ikflow"` first).

Scripts append the repo root to `sys.path` themselves, so run them from anywhere:

```bash
python scripts/panda/panda_benchmark.py --task mug  --targets 15 --guesses 3 --config latent
python scripts/panda/panda_benchmark.py --task pose --targets 15 --guesses 3 --config latent
python scripts/iiwa/iiwa_benchmark.py   --task mug  --targets 12 --guesses 2 --config latent
python scripts/collate.py 'results/*/benchmark/*/summary.json'
```

The older per-experiment scripts (`panda_mug*.py`, `panda_pose_headtohead.py`, `*_collision.py`,
`iiwa_mug.py`) are kept for comparability with archived results and are configured by editing the
`####### Options #######` block at the top; `iiwa_collision.py` needs `../codebase` on `sys.path`.
There is no test suite beyond `tests/` (invariant guards, not a suite) and no lint config. Solver
logs and summary JSON go to `results/` (gitignored). Visualization goes to Meshcat; the mug
experiments start a *second* Meshcat because `GenerateDiagramWithMug` rebuilds the diagram per
target.

Panda weights download via `ikflow` (`panda__full__lp191_5.25m`); iiwa weights are local pickles
under `models/iiwa14/` (gitignored — obtained separately). `models/panda/*.urdf` use
`package://combining_kinematics/...` URIs with vendored meshes, so the scene is portable.

## Architecture

### `src/generic_program.py` — the shared program

`ProgramOptions` is the single dataclass configuring everything (costs, tolerances, solver,
seeding, dtype, logging). `IKFlowProgram` owns the Drake diagram, the plant, its `ToAutoDiffXd()`
copy and both contexts.

Constraints are **not** added to Drake one at a time. Each `Create*Constraint` builds an
`IKFlowConstraints(lb, ub, eval_func)` and appends it to `self.constraints`; `ApplyConstraints`
adds a *single* Drake generic constraint whose evaluator (`EvalAllConstraints`) computes
`q = VarsToQ(vars)` and the forward kinematics **once** and dispatches the cached `(vars, q, pose)`
to every `eval_func`. The network pass dominates cost, so **never add a constraint that recomputes
`VarsToQ` itself**.

`Solve()` configures the solver from the options, registers a visualization callback (which also
appends every iterate to `options.vars_file` when set), keeps `program.last_iterate` so any
abnormal exit can still be verified, and returns Drake's `MathematicalProgramResult`.

### Per-robot subclasses (`src/panda_program.py`, `src/iiwa_program.py`)

Each robot implements `__init__` (frames, plant sizes, model loading), `create_prog` (decision
variables, initial guesses, `self.jacobian_gen`, `add_constraints`/`add_costs`), `ik_inference` and
`VarsToQ`. The `...MugProgram` subclasses swap `self.frame` from the end-effector (the frame the
flow was *trained* on) to `between_fingers` (the frame the grasp acts on), keeping `X_grasp_ee` so
seeds can still be expressed in the network's frame. A mug grasp constrains only the gripper's
position in the mug frame (`x = y = 0` exactly — an equality, because that is the task; `z` within
`mug_height`), leaving orientation free — hence the overridden `CreateIKConstraint` and
`SeedCandidates`.

### Checkpoints carry their architecture (`src/flow_loading.py`)

A `.pkl` is a bare `nn_model` state dict — `IKFlowSolver.load_state_dict` is a plain
`pickle.load` — so **weights travel without their architecture**. `nb_nodes` and `rnvp_clamp`
change the forward pass **without changing any parameter shape**, so a mismatch loads cleanly and
is silently a different chart.

Each checkpoint has a sidecar `<name>.arch.json`, written by
`scripts/training/export_ckpt_to_pkl.py` from the training checkpoint's own
`hyper_parameters.base_hparams`. `LoadFlowSolver(robot, checkpoint)` reads it and cross-checks
against the weights: `nb_nodes` (module-list length / 2), `coeff_fn_config`,
`coeff_fn_internal_size` and the network width are all recoverable from state-dict shapes and are
verified; a contradicting sidecar **raises**. **`rnvp_clamp` is the one field no check can
catch**, which is why the sidecar is mandatory. The resolved architecture is attached as
`solver.arch`. No sidecar → legacy architecture with a `RuntimeWarning`; all on-disk checkpoints
are backfilled (`scripts/training/backfill_arch_sidecars.py`).

**Hold `dim_latent_space` at each robot's baseline** — iiwa14 8, Panda 7. The latent width *is* the
decision-variable count (21 iiwa, 20 Panda), so varying it changes the optimization problem rather
than the chart; holding it also means a checkpoint loaded against the wrong robot fails the shape
check instead of loading silently.

### Gradients through the flow

`VarsToQ` is dual-path: under `float` a plain forward pass; under `AutoDiffXd` it calls
`self.jacobian_gen` (one reverse pass yields both `dq/dvars` and `q`) and chain-rules
`jacobian @ vars_gradients` into fresh `AutoDiffXd` objects. Both go through
`MakeFlowInference(nn_model, ...)`, a free function of the lumped variables closing over the
network and nothing else — which is what lets `FlowJacobianGen` memoise `torch.compile(jacrev(...))`
per process rather than per program. Analytic formulations instead evaluate `pydrake.math` trig on
templated types so Drake's own autodiff propagates.

Numerical facts worth not rediscovering, several encoded in `ProgramOptions` defaults:

- Evaluate the flow in **float64** (`use_float64=True`). Gradients are analytic, so this is not
  about differencing: a float32 network produces *values* with a ~1e-7 noise floor, which corrupts
  every quantity computed as a difference over a small step — line-search actual-vs-predicted
  reduction, convergence tests, SNOPT's derivative verification. `snopt_function_precision` tells
  SNOPT that noise floor when running in float32.
- **`ik_constraint_tol` forms no constraint bound.** The pose rows are a hard equality
  (`lb = ub = 0`); what survives of the option is the benchmark's gate (see the tolerance ladder).
- The IK pose constraint is six rows: per-axis position error, then the **roll-pitch-yaw residual**
  `rpy(FK(q)) - rpy(target)` wrapped to (-pi, pi] (`orientation_error_rpy`).
  `orientation_error_form` picks the bounds — `rpy` (default) pins the residual to zero,
  `rpy_boxed` allows `±ori_tol` per row. **Three signed rows are deliberate**: earlier revisions
  used a scalar angle `2*arccos(|q.q_target|)`, and taking a norm of a three-component error puts a
  branch point at zero error — infinite derivative, plus an `eps` clamp returning an `AutoDiffXd`
  with an *empty* derivative vector. Commit `0be5342` holds the retired forms.
- Constraint rows with an identically-zero gradient (e.g. the homogeneous row of a transformed
  point) break LICQ — the mug constraints deliberately drop it.
- The conditioning variable `c` is boxed near the target (`c_position_slack=0.25`) to keep the flow
  inside its trained workspace — a heuristic, not a correctness requirement, since the IK
  constraint is on `FK(q)` and an out-of-distribution `c` cannot produce a false solution. Note the
  exposure cliff at 0.5 below: the default sits just under it, by luck rather than design.
- **There is no seeding search, deliberately.** A previous revision drew 256 `(c, z)` candidates,
  scored them against the problem's own constraints and started from the best — solving part of the
  problem outside the solver, which only the learned formulation can afford. The machinery is
  removed, not disabled. `SetStartFromQ(q_init)` is the only way to set an initial guess, and every
  formulation in a comparison gets the same `q_init`.

### Why the Jacobian is a `jacrev`, not a JVP

Measured, so it does not get re-litigated. The single `J = dq/dvars` (7 x 21) serves every consumer
— all eleven constraint rows and the objective gradient — and on one Panda grasp solve **the
Jacobian is 84% of the solve and the flow altogether 96%**. Reverse mode wins on shape: 7 outputs
against 21 inputs of which only 13 reach the network. Measured: `jacrev` + matmul 17.1 ms and a
vmapped VJP 16.4 ms (bit-identical) against a vmapped JVP at 47 ms — **forward mode 2.8x slower**,
with 13 tangents costing the same as 20 (CPU-dispatch bound). The one place a VJP would have won,
the objective-gradient path, is already covered by sharing the constraint's Jacobian.

**`torch.compile` on the `jacrev` is worth taking**: **1.48x** on a whole AutoDiffXd `VarsToQ`,
agreeing with eager to 3.5e-15, one dynamo graph, for a one-off 8-14 s local / ~35 s cluster cold
cost. An old comment saying otherwise was measuring a *bound method* — `torch.compile` guards on
everything the callable closes over, so each of thirty programs re-triggered dynamo. It is off by
default, on with `--compile`, because more iterations inside a fixed cap **moves the learned arm's
success rate** and only that arm benefits — so **every run being compared must set it the same way**.

### Profiling

No profiler in the tree; recover one with `git show ab3ea15:scripts/profiling/profile_flow.py`. At
batch size 1 the flow evaluation is **entirely CPU-bound**: the GPU is never behind the CPU, and
float64 and float32 cost the same wall time despite a 3.4x difference in GPU kernel time. Roughly
70% of a `jacrev` is CPU-side dispatch (PyTorch eager + FrEIA Python; `cudaLaunchKernel` only ~17%),
so runtime is bounded by how fast the CPU can describe 2853 operations and **even zero-overhead
execution leaves only a ~3x ceiling**. **Reducing that dispatch cost is out of scope** (Thomas: infra
fixes for the CPU bottleneck are "future work/possibly not in scope at all") — a number to report,
not a project.

### The conditioning frame (read this before touching the learned formulation)

The flow is conditioned on the pose of **the frame it was trained on**, and in both grasp scenes
that is *not* the frame the code used to look up by name. `panda_finray.sdf` contains its own
`panda_hand` welded to `panda_link7` at `[0, 0, 0.134]`, rpy `[90, 0, 45]`, whereas jrl's Panda —
the model IKFlow was trained against — puts `panda_hand` at `[0, 0, 0.107]`, rpy `[0, 0, -45]`.
`GetBodyByName("panda_hand")` returns the finray one, **27 mm and 120 degrees** away. The iiwa's
`iiwa_link_7` is 45 mm short of the flow's frame.

The symptom is unmistakable. Running the flow *forwards* on a random configuration (`rev=False`,
which inverts it exactly) returns the latent that would have produced it:

| robot | at the scene frame | at the calibrated frame | typical `\|z\|` under the prior |
| --- | --- | --- | --- |
| Panda | 67.6 | 2.23 | sqrt(7) = 2.65 |
| iiwa14 | 12.1 | 2.45 | sqrt(8) = 2.83 |

A latent of 67 is the network reporting that the configuration is astronomically unlikely for that
conditioning pose. `IKFlowProgram.CalibrateFlowFrame` measures the offset against
`ik_solver.robot.forward_kinematics` at several configurations, checks it is constant (both frames
are welded to the same link, so it must be) and caches it as `self.X_ee_flow`; `FlowPoseInWorld()`
is what should be used wherever a conditioning pose is formed.
`ProgramOptions.calibrate_flow_frame=False` restores the old behaviour for ablations.

**It was called only by the grasp subclasses until 2026-09-16**, and the omission was invisible
because the Panda pose scene used `panda_jrl.urdf` (whose `panda_hand` really is the trained frame)
and the iiwa's 45 mm offset is a pure translation. Welding the finray into the pose scene destroyed
that coincidence and the Panda pose task collapsed to **10/60 with median `max_violation` 0.4**,
against 58/60 and 1.8e-08 for the grasp task in the *same* scene with the *same* chart. Both base
pose programs now calibrate; it is a no-op where the frames agree and draws from its own fixed-seed
generator so it cannot shift the grid. **Every pose column measured before this is superseded.**
The general lesson: **a calibration that is skipped is indistinguishable from one that is correct,
until the geometry it silently relied on changes** — and a miscalibration and an architectural
weakness produce the same symptom, each masking the size of the other.

### The latent trust region, and what the ablation ladder attributed

Panda grasp, learned arm only, 60 cells, 20 s, paired, one grid: baseline (uncalibrated frame, no
sharing) 11/60 → + conditioning-frame calibration **29/60** → + shared flow evaluation 30/60 → +
latent trust region 34/60; per-rung exact McNemar 26/8 **p = 0.0029**, 1/0 p = 1.0, 15/11 p = 0.56,
whole stack 28/5 **p = 6.6e-5**. **The frame calibration is worth 18 of the 23 cells and is the only
significant rung**; sharing is worth one cell, as it must be, being bit-identical.

`latent_trust_region` is +4 cells and not significant. It stays because IPOPT is poorly behaved on
unbounded variables and a nonbinding constraint still shapes an interior-point trajectory — Thomas:
*"The fact that it's nonbinding doesn't mean it didn't have a positive influence on the solver, since
IPOPT is interior point."* It is a **stated deviation from eq. (6) that must be documented in the
paper draft**; do not argue from binding fractions that it is inert.

### Sharing the flow evaluation between bindings

Each Drake binding evaluates its own callback, so `EvalJointCenteringCost` used to run a second
forward pass and `jacrev` exactly where `EvalAllConstraints` had just evaluated (an archived IPOPT
log shows 1276 objective evaluations against 1276 constraint evaluations — about half the network
work redundant). `IKFlowProgram.QAndPose` memoises `(q, pose)` on the iterate, keyed on the values
**and** the AutoDiffXd derivative block (keying on the value alone would hand back a Jacobian
computed against the wrong seed matrix), behind `share_flow_evaluations`, which **defaults on** —
the memoised path is bit-identical, so there is no reason to run without it except to reproduce a
pre-overhaul measurement.

### Scenes and utilities (`src/utils.py`, `models/`)

`BuildEnv(meshcat, directives_file, extra_directives=None)` builds the diagram from a Drake
model-directives YAML, registering `package.xml` so `package://combining_kinematics/...` resolves;
`extra_directives` is appended to the loaded ones **in memory**, so a caller can add models without
writing to the tracked YAML. `GenerateDiagramWithMug(q, program, yaml_file, meshcat)` uses exactly
that: an `add_model`/`add_weld` pair for a mug at the gripper pose of `q` (the weld pose passed as a
`pydrake.common.schema.Transform`, not formatted into text). The YAML on disk is never modified, so
a crash cannot leave a stray mug in a tracked scene — it used to append-then-truncate, which could.
`BuildEnv(meshcat=None)` skips visualization outright, which is *not* the same as passing `None`
through to `ApplyVisualizationConfig` (Drake would start its own).

Targets in the mug experiments are generated by sampling collision-free `q` and welding a mug at
the resulting gripper pose, so every target is known to admit a valid grasp. `HiddenPrints`
suppresses Drake/ikflow output at the file-descriptor level.

Notebooks in `notebooks/` are the exploratory counterpart to `scripts/`, run from `notebooks/`.

### The hardened scene and shelf-contained targets

Accepting the gripper pose *wherever a collision-free `q` landed* put targets in free air far more
often than in clutter, so collision avoidance barely bound. `../codebase` hardened its Grasp
Selection the same way (`38f4eac`); this is the same treatment, same shelves, same inset.

**The scene.** `models/panda/panda_finray_collision_hardened.yaml`,
`models/panda/panda_collision_hardened.yaml`, `models/iiwa14/iiwa14_collision_hardened.yaml` are
their legacy twins minus `binF` and (where present) the seven welded decorative mugs. Four shelves
and two tables remain. Legacy files are untouched and remain the default for the older scripts.

**The target must land in a shelf.** `src/shelf_regions.py` holds the twelve compartments — four
units x three bays — each as its **local** box plus the weld's translation and yaw, and
`PointInShelfCompartments` rotates the query point into the region's own frame. A world-frame AABB
is not usable: at these 135°/235° welds it over-approximates the footprint about **4x**. The inset
is symmetric because `shelves.sdf` has **no back wall** — the unit is a tunnel and which `x` face is
"front" depends on the yaw. `shelf_depth_inset = 0.10 m`, matching the sibling.

**And the object must fit.** **Drake never generates collision candidates between two ANCHORED
geometries**, and `GenerateDiagramWithMug` *welds* the target mug, so a mug/board overlap is
silently invisible on the solve scene. `FloatingMugScreen` therefore runs on its own diagram with
the mug appended **unwelded**. Its robot filter is exact model-instance names, and a filter
matching nothing would reject every candidate as "penetrating" — indistinguishable from a too-deep
inset — so `tests/test_shelf_placement_screens.py` pins it.

**Acceptance is 0.2-0.7%**, an order of magnitude below the sibling's, measured by
`scripts/probe_shelf_acceptance.py`, as a fraction of collision-free draws (at inset 0.10: panda
grasp 0.675%, panda pose 0.371%, iiwa grasp 0.552%, iiwa pose 0.228%). That is ~400-1100 draws per
target, ~12 s per grid — negligible. What it breaks is the sibling's rejection guard: acceptance
restarts at every accepted target, so the guard is a per-target tail bound and a 60-target grid
gets 60 chances to trip; at the sibling's 5000 the iiwa pose row trips 41% of runs, and a trip kills
every shard of a queued run. **`MAX_CONSECUTIVE_REJECTIONS = 50000`**, where every row is zero to
machine precision.

**Two things to know before reading a hardened number.** The pose task's containment point is the
frame its target *is*, recorded per run as `placement_point`. And the panda *grasp* scene never had
decorative mugs, so hardening removes strictly less from it than from the other two; say so
wherever grasp and pose deltas appear in one table.

**Guesses are deliberately not containment-filtered.** They are initial configurations, not
targets; filtering them would couple the start distribution to the target distribution.

`--scene legacy --target-placement free` reproduces the pre-hardening sampler exactly (a test pins
it), so an archived grid is still re-runnable. `c_position_slack` is **not** touched: a ±0.25 m
conditioning box around a mug in a 0.10 m compartment is loose relative to the free space and
plausibly hurts the learned arm specifically, but sweeping it here would confound hardening with
slack.

## Rules the campaign established

Each was learned by getting it wrong, at the cost of whole tables. None is negotiable without
Thomas.

### The tolerance ladder: constraint bounds exact, then solver tolerance, then a looser gate

Thomas: *"IK constraint tol should always be zero. The whole point is that it's an equality
constraint, satisfied exactly. Tolerance should be zero in the mathematical program, only appearing
in solver tolerance."*

`lb = -tol, ub = +tol` does not loosen an equality, it **changes its kind**, and an interior-point
method parks *on* the face of an inequality. The evidence, from 480 persisted cells: orientation
was already a true equality and converged five orders tighter than position, in the same
constraint, in the same solve — learned median `pos_error` 9.999e-05 against `rpy_error` 1.38e-08,
with 67-84% of solutions sitting exactly on the box. **The analytic arm's version was a fairness
defect, not merely a numerical one**: its pose target was a box on its decision variables carrying
the whole `ik_constraint_tol` tuple, so it got ±0.01 rad of orientation freedom per axis while the
arms it baselines were pinned to zero, and `max_violation` reported 0.00 because a box is satisfied
right up to its face.

The fix collapsed residuals by five to nine orders of magnitude and is guarded:
`tests/test_constraint_bounds.py` reads the bounds Drake was actually handed and fails if any drifts
back. Both sibling projects already followed this ladder. **No ranking moved when it landed** —
pooled over eight experiments and six caps, 225 cells better against 204 worse, p = 0.334 — because a
boxed solution stopped at 1e-4 and the gate is 1e-3, so it passed anyway. **The defect was in
solution quality and in fairness between the arms, not in the rankings.** The equality is also ~30%
*cheaper* in iterations and wall clock: it is unambiguously active, so there is no active-set
question and IPOPT handles it directly rather than through barrier terms on two inequality faces.

**Rung 3: the gate stays deliberately looser, and the gap must not be closed.**
`ik_constraint_tol = 1e-4` for the program's rows, `task_tol = 1e-3` for the task gate. Thomas:
*"go back to 1e-4 actual tol and 1e-3 task tol, to avoid this issue (that's why I did it in the
first place)."* On 480 iiwa pose cells the joint-space arm's median `pos_error` was 1.0001e-04
against a 1e-4 bound, with 64% a rounding error above it and none above 1.01e-4. A gate at exactly
1e-4 scores which side the last ulp fell on, and it *appeared to reverse* a row: 296-vs-332
(p = 0.016 against) became 199-vs-120 (p = 2.5e-08 in favour). A coin toss dressed as a result.

**Never set an acceptance gate equal to a bound the solver is optimising against**, and before
proposing to tighten one, check the distribution of the gated quantity: if solutions are pinned to
the bound, tightening measures noise. The collision gate carries the binding's own slack for the
same reason. A second verdict `feasible_relaxed` is recorded at `task_tol` so the question stays
re-analysable; the relaxation is worth +10 to +18 cells of 480 on the grasp task and **exactly zero
on the pose task**, moving no ordering.

### A region an initial guess may violate must be a general constraint, never a variable bound

IPOPT's `bound_push` projects the initial guess into every *bounding box* before evaluating
anything, so a box silently reshapes the start protocol. This bit twice.

**The conditioning-pose box.** Pre-clipping `c` teleported it to the box face while the latent
stayed tuned to the unprojected pose, making the "exact" and old "pre-clipped" protocols land on
bit-identical iterate-0 lines. It is now `AddLinearConstraint(I, lb, ub, c)` (`CBoxConstraint`).

**The latent's own `±5` box** was left as a bounding box after that repair, and was worse —
`SetStartFromQ` clipped the inverted latent itself, so the projection was ours. The flow is a
bijection, so `flow(c, InvertFlow(q, c))` reproduces `q` exactly, but only at the *unclipped*
latent; the inversion routinely returns components past ±5, so the clip moved the start by radians
and the cell was scored `unrepresentable_start` — an arm recorded as unable to represent a
configuration it represents exactly. Measured on iiwa pose paired, 20 s, same grid: learned success
11/60 → **40/60**, cells scored `unrepresentable_start` 49 → 0, median `|q(start) - q_init|`
3.79 → 0.0000. The arm starts at `|z| ~ 7.9`, outside the region, and the solver walks it to
`|z| ~ 2.9` on its own — the whole point of the region being a constraint. **Every archived paired
learned column predating this is void.**

Two structural notes. The box lives in **one** method, `LatentBoxConstraint()`, because the first
repair fixed `generic_program.py` while the mug subclasses overrode `BoundingBoxConstraint` and
carried their own copies — the pose arms were fixed and the grasp arms silently were not. And
nothing may project a guess without recording that it did (`clip_distance`). Variable bounds remain
fine for regions a start always respects (the correction's ±0.1), and infeasible initial guesses
are acceptable by policy — Thomas: *"we're not assuming feasible initial guesses."*

### No invented formulations, and no weakening of the problem

A "task-parameterised" grasp reformulation (decision variable = the grasp pose in the mug frame)
existed here and was fielded as the benchmark's "learned" arm. **It is not the paper's formulation
and should never have existed** — Thomas: *"You were never supposed to do the task-parameterized
version... Constructing new formulations and passing them off as ones I've already written is
completely unacceptable."* The learned formulation is eq. (6) of the draft, exactly as
`PandaMugProgram`/`IiwaMugProgram` implement it: free conditioning pose `c`, latent `z`, correction
`q_c`, the grasp imposed as constraint rows through `FK(q)`. The machinery is **removed outright**,
mirroring the seeding precedent, and every number produced with it is void and has been re-measured.

The same principle bans weakening the problem. The mug-axis rows are an equality because that *is*
the task — a `mug_axis_tol` option that widened them was removed, not defaulted to zero.
Improvements must come from the formulation or the solver, never from making the question easier.

### The correction penalty is a stated part of the learned formulation

`correction_cost_weight = 10`, approved by Thomas on 2026-09-02 (*"A penalty on the correction term
is acceptable"*). The draft says `q_c ~ 0` without specifying how that is imposed; this is what
imposes it. So the weight is a **stated** part of the formulation and must appear wherever the
learned arm is described, and every table must still show what the penalty buys and costs.

**Options naming learned-only decision variables must be guarded.** `add_costs` applied
`correction_cost_weight` unconditionally, but `correction` exists only on the learned arm and all
three formulations share one `ProgramOptions`. So `--set correction_cost_weight=10` raised
`AttributeError` inside every numerical/analytic program's construction and each of those columns
scored **0 of 480 in about 10 ms per cell**. The failure mode is worth remembering: a whole column
of zeroes with `median_max_violation = nan`, three orders of magnitude below the cap, with
`fail_reason = "error"` rather than a named task gate. **Any arm reporting a per-cell wall time
three orders below the cap is not solving badly, it is not solving at all.** `_abort_on_dead_arm`
in `src/benchmark.py` now aborts when an arm fails identically, in under a second, on its first
three cells — this pattern cost two whole columns of cluster campaigns.

### Every result is told in success, iterations, cost and wall clock

Thomas's standing reporting rule. Iterations are hardware-independent and describe the
*formulation*; seconds describe this implementation on this machine and are **never compared across
machines**; cost says what the solution is worth. Reporting only seconds makes the cap story look
arbitrary; only iterations hides that the learned arm's iteration is ten to thirty times more
expensive; only success hides that on the grasp task its solutions cost roughly twice the
baseline's.

Two musts, both learned by getting them wrong: **cost is compared only on cells *both* arms
solved** (a median over each arm's own successes compares different cell sets, and the easy cells
are exactly the ones a weaker arm also solves, so that form flatters whichever arm fails more); and
**learned-only regularizers are excluded from the reported objective** (`reported_cost`), so the
column measures the objective every formulation shares.

**Joint space is the comparison's target; the analytic columns are baselines.** Lead with
learned-vs-joint-space McNemar, never claim a win off beating analytic alone, and state ties as
ties. Baselines get bug fixes and standard treatments but no novel research (Thomas, on
`analytic8`'s uniform branch draw: *"since it's a baseline, we're not trying to do novel research
things to make it better"*).

**A cap check before reporting any loss**: does the losing arm have meaningful timeouts? Timeouts
~0 → the cap is innocent and the result is a formulation result. Timeouts significant → the cap is
measuring throughput, not formulation; raise it and re-measure. The case that established this:
iiwa `n4` contained grasp scored 391 v 442 (p = 1.4e-06, a clear loss) at 45 s with 88 timeouts; at
180 s with 0 timeouts it is 447 v 442, a tie. The arms were tied all along. Relatedly, **the cap is a
budget for the arm that evaluates a network, not a shared budget**: across 5/10/20/45/90/180 s every
baseline is flat, with one exception — on iiwa grasp paired the joint-space arm is itself cap-bound
below 20 s, with cells running 1300-1430 iterations against that arm's median of 70. So the
joint-space arm is not uniformly cheap; it has a tail.

## Benchmarking (`src/benchmark.py`, `scripts/*/[a-z]*_benchmark.py`)

- **Paired grid.** `num_targets x num_guesses` cells, one solve per cell, no retry-on-failure,
  every formulation on identical cells — so success can be compared with an exact McNemar test and
  the CI can bootstrap over whole *targets* (guesses within a target are correlated). Guesses are
  drawn **per target** (`guesses[ti][gi]`), not shared across targets: sharing across *arms* is
  what pairing needs, sharing across *targets* quantizes start-dependent effects into target-sized
  blocks.
- **Both start protocols are measured** (`--start`), and both are first-class. `paired` puts every
  arm at the same `q_init` in its own variables via `SetStartFromQ`: joint space at `q_init`,
  analytic at `FK(q_init)` with `psi`/`GC` recovered by inversion, learned at `c = FK(q_init)` with
  `z` from running the flow forwards. Order matters — invert **first**, then clip; clipping first
  and inverting at the projected pose returns `|z| ~ 1e7`, because a random configuration is not a
  grasp of this mug and the flow is right to say so. `native` gives each formulation the
  initialisation it would have outside a comparison. The joint-space arm's two protocols coincide
  (its native start *is* a random configuration), so any difference between the tables is
  attributable to the others. Neither protocol searches, and the paired start is *measured* rather
  than assumed: every cell records `clip_distance` and `start_q_error`.
- **Success verified from the returned point**, not from `result.is_success()`: every binding is
  re-evaluated at the solution and the task re-measured from `q`, with a named `fail_reason`. Every
  learned failure in the archived runs was a wall-clock timeout, and a timeout that landed on a
  valid grasp is a success. Two gates that are easy to get wrong: an interior-point method parks
  *on* the collision constraint (value 1 + 1e-7) so that gate needs the binding's own slack; and
  `PandaMugProgramAnalytic` inherits from the *pose* analytic class, so the grasp must be measured
  by asking for `between_fingers` by name.
- **Raw per-cell state is persisted**, not only derived summaries — the returned `q` (and the
  recovered last iterate's `q` on abnormal exit), the decision-variable vector, per-binding signed
  violations, the true `min_distance` and `min_distance_pair` beside `collision_value`, and the
  start. Without this, no geometric quantity can be recomputed after a run without re-solving the
  whole grid. Decide the recorded schema *before* launching: the grid is the expensive thing, not
  the disk.

**Abnormal exits keep the iterate.** Thomas rejected a watchdog that raised from inside the
flow-evaluation callback: *"I don't like the idea of messing with QAndPose to force kill it, since
then we don't get an intermediate solution?"* So `Solve()` keeps `program.last_iterate`; any
abnormal exit is verified from that point and recorded as `recovered_feasible`/`recovered_cost`
alongside the failure reason; and the *process* is bounded from outside (OS-level `timeout`), never
the solve from inside a hot-path callback. `SolveTimeout`, `CheckDeadline` and `hard_time_factor`
were **deleted** and must not come back.

Beyond the verdict a record carries `max_violation` and `detail["violations_all"]`;
`collision_value`, `min_distance`, `min_distance_pair`; `start_q_error`, `clip_distance`, `z_norm`;
`median_correction_inf` and `correction_binding` (how much of the ±0.1 box solutions use — the
check that the learned arm is not quietly becoming a reparameterised joint-space arm); and `q`,
plus `q_lift` and `q_flow` separately under `lift_q`. `median_max_violation` separates the arms by
six orders of magnitude and is worth reading next to any success count.

Three switches. `--compile` turns on the compiled flow Jacobian (see above: set it identically
everywhere being compared). `--set NAME=VALUE` overrides any `ProgramOptions` field, so a sweep
needs no code edit; it lands in the metadata and the default tag. And the grid is drawn from a
generator local to the script and hashed into `metadata["grid_hash"]` (with the task as a *suffix*,
since the iiwa's mug and pose grids otherwise hashed identically), so runs not measured on the same
cells cannot be compared by accident — `python scripts/collate.py --pair learned '<glob>'` runs
exact McNemar between runs on matching cells and refuses a grid mismatch.

### How paired the paired start actually is

`SetStartFromQ` gives every arm the same `q_init` expressed in its own variables, but a formulation
can only represent a configuration its variables reach.

| arm | `\|q(start) - q_init\|` at the guess | why |
| --- | --- | --- |
| joint space | 0 exactly | its variables are the configuration |
| learned, free `c` | ~1e-6 (pose task: 0.0 measured) | exact: unclipped `c` + inverted latent + correction |
| analytic, 8 branches | 1e-11, or several radians on ~0.6% of starts | exact where the chart covers the configuration |
| analytic, 4 branches | 1e-11, or several radians on ~10% of starts | the historical chart; the `analytic` column |

Two projections that were once necessary have been removed — the learned arm's pre-clipping of `c`,
and the pose analytic arm's clipping into its `xyz_rpy` box (which had it always beginning at the
target pose, a median 2.7 rad from the shared `q_init`). `legacy_paired_start=True` restores the old
behaviour. `start_q_error` measures the *initial guess*; where a guess sits outside a variable's
bounds IPOPT projects it at iterate 0 and `clip_distance` records that. The two numbers together
describe honestly how much survives the solver's own bound projection.

### `collision_value` is a penalty, not a clearance

`detail["collision_value"]` is the **raw** value of Drake's `MinimumDistanceLowerBoundConstraint` —
a smooth penalty aggregated over every geometry pair inside the influence distance (`bound=1e-3`,
`influence_distance_offset=0.1`). It is a pure number, not a length. Calibrated against the true
minimum signed distance over 4000 random iiwa configurations: raw < 1.0 is clear, 1.0-1.05 is
roughly 0 to -1 mm, 1.2-1.5 is -12 to -19 mm, 2.0-4.0 is -59 to -124 mm. Every *success* sits at
0.9997-1.0005 — parked exactly on contact, which is why the gate carries the binding's own slack.
**`verify()` records the true signed `min_distance` in metres and the pair attaining it**, so read
that instead. The row's shape is three `ProgramOptions` fields (`collision_bound`,
`collision_influence_offset`, `collision_row_scale`).

## The solver axis: interior point, SQP, augmented Lagrangian

**Three METHOD CLASSES, not three vendors.** Thomas: *"for NLOPT, we want to use it as an augmented
lagrangian solver. This is important! We don't care about SLSQP, since SNOPT is already SQP. The
point is that we test interior point, augmented lagrangian, and SQP."* So `--solver` is `ipopt`
(interior point), `snopt` (SQP), `nlopt` (**`LD_AUGLAG`** — not `LD_AUGLAG_EQ`, which absorbs only
*equality* constraints into the AL and leaves inequalities to the inner solver, and this program
carries both). Any solver added later must be justified by the class it contributes. **The solver
is a reporting axis, never a choice**: Thomas, *"we would not pick one solver or the other, but
rather report the performance for both solvers."* It is shared across arms within a run and varied
across runs.

All three take `kGenericConstraint`/`kGenericCost`/`kCallback`, and all three call
`EvalVisualizationCallbacks` **inside their objective evaluation**, so `last_iterate` recovery works
unchanged under each.

### Each solver converges at its own defaults

The SNOPT branch used to read IPOPT's `acceptable_tol`/`acceptable_constr_viol_tol` as its `Major
optimality`/`Major feasibility tolerance`. That is rung 2 of the tolerance ladder done wrong:
IPOPT's `acceptable_*` family is its **relaxed early-stop** criterion, not what it converges to (its
real `tol` is 1e-8). **Do not transplant one solver's option values onto another.** Unset
`snopt_*`/`nlopt_*` fields are simply not passed, so each solver sits at its own defaults and the
shared, deliberately looser task gate decides success.

**But "each at its own defaults" is a CHOICE, and not a symmetric one.** `tol`, `constr_viol_tol`,
`dual_inf_tol` and `compl_inf_tol` were never `ProgramOptions` fields at all, so every archived run
took IPOPT's own defaults — `constr_viol_tol` **1e-4** against SNOPT's Major feasibility **1e-6**,
`dual_inf_tol` **1** against Major optimality **2e-6**. So the interior-point column was allowed
100x the constraint violation and six orders more dual infeasibility than the SQP column, on top of
an early stop (`acceptable_tol=1e-3`, `acceptable_iter=1`) SNOPT has no counterpart for. That is a
property of the HARNESS, not of interior-point methods. The four fields are now plumbed. **The task
gate is untouched by any of it** — success is verified from the returned point at `task_tol = 1e-3`,
two orders above every tolerance involved.

### What each solver will and will not tell you

**No solver reports an iteration count through Drake.** `SnoptSolverDetails` carries `info`,
`solve_time` and multipliers but no count; `NloptSolverDetails` carries a single `status`.

| | IPOPT | SNOPT | NLopt |
| --- | --- | --- | --- |
| print file | yes | yes | **none, and `kPrintFileName` is silently ignored** |
| `iterations` | "Number of Iterations" | "No. of major iterations" | -- |
| eval counts | 4 separate counts | one `User function calls (total)` | -- |
| seconds | log | `details.solve_time` | -- |
| status | -- | `details.info` | `details.status` |

`iterations` means **majors** under both solvers that report it, because IPOPT's count is majors —
the column is only comparable if it means the same thing. SNOPT's minors are kept separately. snOptA
has one user function, so IPOPT's four eval counts have no SNOPT counterpart and stay `None` rather
than being filled with a different quantity.

**So the program counts evaluations itself** (`IKFlowProgram.ResetEvalCounts`, counted in
`QAndPose`, the one funnel every arm's solve passes through). `map_jacobian` is the AutoDiffXd
count — one `jacrev` through the network each for the learned arm. It is the only cost measure the
NLopt column has, and it cross-validates: on the SNOPT smoke run it equalled `User function calls
(total)` **exactly** on every cell. It is *not* an iteration count — a line search evaluates the map
several times per accepted step. `collate.py` prints `--`, never `nan` or `0`, where a solver
reports nothing.

**Status is decoded numerically, not from log text.** SNOPT INFO 34 is the time limit and Drake
leaves it as a generic solver error with no distinctive exit string; NLopt has no text at all.
Matching exit strings alone would report `timeouts: 0` for a capped run of either. NLopt status 5 is
`MAXEVAL_REACHED`, recorded as `hit_eval_cap`, since NLopt has no notion of an iteration to cap.

### Traps, all found by probing rather than by reading

**A solver option can be ACCEPTED and do nothing**, and only the solver's own parameter echo tells
the two apart. `"Timing Level"` was set for the life of this repo and silently wrote no timing
block, because SNOPT's parser is case-sensitive on the second word while Drake raises only on
keywords SNOPT does not know at all. Every option this branch exposes was verified reaching the
solver — for IPOPT by requiring `used = yes` in the `print_user_options` block, for SNOPT by
requiring the value in the parameter echo — and that check earned its place three times:

- **`Hessian updates` is inert** at these sizes: SNOPT picks full-memory mode below 75 variables and
  the programs have 20-21, so the echo keeps reporting 99999999 however it is set. `Hessian
  frequency` is the one that bites.
- **`Nonderivative linesearch` is a VALUELESS keyword** — passing 0 turns it ON exactly as 1 does,
  so it is a bool emitted only when True. The echo abbreviates it `Nonderiv.  linesearch`, which is
  why a first probe grepping its full name wrongly called it inert.
- **`linear_solver=mumps` does not exist.** Drake's IPOPT is built against SPRAL and offers only
  `spral` and `custom`, so there is no linear-solver axis on this problem.

**Read a solver's defaults out of the solver, not out of memory.** IPOPT's
`print_options_documentation` dump contradicted the obvious assumption:
`acceptable_dual_inf_tol` defaults to **1e+10** and the two acceptable infeasibility tolerances to
**1e-2**. A sweep arm labelled "IPOPT's defaults" that was not cost a resubmission.

**The Drake on the laptop is not the Drake on the cluster.** The workstation is a source build; the
cluster runs the official 1.56.0 tarball, whose `NloptSolver` exposes exactly six options
(`algorithm`, `constraint_tol`, `xtol_rel`, `xtol_abs`, `max_eval`, `max_time`) against the source
build's eleven. Drake validates NLopt names strictly and **raises** on an unknown one, so code
written against the newer API passes locally and fails on every cell of a cluster run.
`tests/test_solver_plumbing.py` pins the emitted keys to 1.56.0's six; do not relax it.

**`max_eval` is not "unset" by default** — Drake defaults it to 1000, a cap that binds here, so it
must be set deliberately or the NLopt column silently measures an evaluation budget rather than the
wall clock.

### The result: IPOPT > SNOPT >>> NLopt, as predicted

Stage SOLVER2, 480 cells (60 targets x 8 guesses), 45 s, seed 1, `--compile`, adopted rungs (Panda
`n6`, iiwa `n4`), `learned,numerical`, both protocols, three placements. The IPOPT column is
measured on this grid and this code, not quoted from an archive.

Learned arm, successes of 480:

| | native IPOPT | native SNOPT | paired IPOPT | paired SNOPT |
| --- | --- | --- | --- | --- |
| iiwa grasp free | **444** | 348 | **456** | 333 |
| iiwa grasp contained | **392** | 175 | **406** | 212 |
| iiwa pose fingertip | **470** | 442 | **422** | 210 |
| Panda grasp free | **474** | 450 | **474** | 397 |
| Panda grasp contained | **461** | 438 | **437** | 280 |
| Panda pose fingertip | **461** | 438 | **405** | 251 |

**IPOPT wins all 24 rows** (12 learned + 12 joint space), 23 significant, p from 3.6e-08 to 4.8e-44;
the exception is Panda contained-grasp joint space, p = 0.09.

**This is a CONFIRMATION, not a finding.** Thomas: *"SNOPT performing worse than IPOPT is not
surprising. In my experience, IPOPT is more robust to ill-posed problems, and our neural network
gradients are definitely ill-posed. I expect to see IPOPT > SNOPT >>> NLOPT."* Write it up as the
size and mechanism of a predicted gap.

**It is a property of the PROBLEM, not of the learned formulation.** The joint-space arm degrades
too, on every row and by comparable margins (457→398, 442→292, 453→404, 323→298, 299→236, 217→169),
and that arm never evaluates the network.

**The gap is much larger under `paired`** — on four of six learned row-pairs the `native` gap is
23-28 cells and the `paired` gap is 77-212 — so **SNOPT copes far worse with an infeasible start**.
A 60-cell triage read this backwards in both directions before 480 cells settled it. **Do not draw a
protocol conclusion from 60 cells.**

**And it is NOT a budget artefact.** Pooled over all 3,952 SNOPT failures: `nonlinear
infeasibilities minimized` (INFO 13) **50.7%**, `current point cannot be improved` (INFO 41)
**36.2%**, iteration limit 9.5%, time limit **3.6%**. About 87% are convergence failures and 11%
budget, and SNOPT times out *less* often than IPOPT does. The two fail in opposite ways: IPOPT's
learned-arm failures are mostly wall-clock — still descending when the clock runs out — where
SNOPT's are INFO 41, giving up at a feasible-but-wrong point. **INFO 41 is the documented signature
of inaccurate or badly scaled derivatives**, which is Thomas's explanation with a mechanism
attached. This is distinct from the gain-ceiling runaway, absent here: `n4`'s solved cells return
violations ~1e-08, so what SNOPT struggles with is ordinary ill-conditioning of an *exact* Jacobian.

**Iterations and cost.** SNOPT takes 2-4x the iterations on cells it does solve and its solutions
are worse on the joint-space arm by a wide margin (median cost 3.627 against 1.828 on iiwa grasp
free, on cells both solved). On the learned arm cost is closer and occasionally favours SNOPT — i.e.
where it converges it sometimes finds a better optimum; it just converges far less often.

**Harness self-check passes.** Joint space is bit-identical between protocols in all six row-pairs,
and the IPOPT column reproduces the archive (456/457 against the archived 457/457 on the same
`grid_hash`, learned inside the ±1 cell that cap-bound cells are reproducible to). **So the eleven
new option fields and the rewritten three-way dispatch left the IPOPT path where it was.**

### NLopt: the augmented Lagrangian is not competitive on this problem

Fielded at 60 cells (stage SOLVER's grid, pairing against the IPOPT and SNOPT triage columns cell for
cell) rather than 480, because a column that may be empty does not need campaign scale to be honest.
On the three grasp rows under `paired` it scores **1/60, 0/60 and 1/60** against IPOPT's 59/60,
timing out on every cell, burning 4,900-6,100 network Jacobians per cell and landing 4.5e-02 to
9.7e-02 from feasible. The one place it works is the pose task under `native` — 38/60 and 39/60
against IPOPT's 58 and 60, reaching 1e-06 — so it is not broken, it is far too slow to satisfy tight
equality rows through an ill-conditioned chart inside any budget here. **`max_time` also binds much
more tightly than SNOPT's `Time limit`** (20.0-20.1 s against 24-28 s in a local probe), because
SNOPT only checks at major-iteration boundaries; wall-clock columns are not capped equally across
solvers.

### Stage SWEEP: solver settings, and what IPOPT's early stop is worth

43 settings (20 SNOPT, 23 IPOPT), one factor at a time against each solver's own default, 240 pooled
cells, `paired`. **Step rejection is deliberately absent** — those five fields are a separate
question (Thomas, 2026-09-16), and `stage_SWEEP` **raises** if an entry names one.

**SNOPT: no setting rescues it.** Against its own 141/240 the best is the gradient-free line search
at 153 (p = 0.20); **not one of the twenty reaches significance**, and the extremes hurt. A solver
losing on INFO 13/41 is not losing for want of tuning.

**IPOPT: the convergence tolerances are INERT and the acceptable-point machinery is everything.**
`convsnopt` (IPOPT held to SNOPT's convergence numbers) scores 212 against the fielded 211 and every
single-factor convergence row is within noise — IPOPT already converges far tighter than either
default, so the 1e-4-against-1e-6 asymmetry above was real on paper and worth **zero cells**. The
`acceptable_*` family, solved of 240: `accviolloose` 213 (66 median iterations, 0 timeouts, violation
1.26e-06) > **as fielded 211** (147, 3, 1.29e-08) > **`ipoptdefault` 200** (840 iterations, 157
timeouts) > `acctight` 121 > `accoff` 62 > `fair` 34; *SNOPT at its own defaults 141* (410, 16,
2.03e-08).

**The fairness question is answered: the ordering survives it.** At each solver's own defaults —
what rung 2 asks for — **IPOPT 200, SNOPT 141, p = 4.1e-09**, and on the 119 cells both solve IPOPT's
solutions also cost less (5.574 against 6.841).

**But the early stop is worth 39 cells and a 9x speedup, and that has to be stated** — and it is
**NOT returning sloppy points**: fielded successes sit at 1.29e-08, five orders inside the gate and
the same quality as SNOPT's. Turning it off drives the violation to 2.22e-15 and success to **62** —
IPOPT without it keeps polishing a solution it already has until the clock kills it. So
`acceptable_iter = 1` does not let IPOPT scrape past the gate, it lets IPOPT **recognise it is
already done and stop**, which under a wall-clock cap is a real capability. SNOPT has no counterpart
and would not benefit: only 3.6% of its failures are time limits. The trade is quality against
throughput — `accviolloose` is fastest and best on success but **worst on cost** of the live arms.
**Nothing is adopted**: the alternatives move success by at most +2 cells of 240 and changing it
would break comparability with every archived run.

### The grasp-containment lever is closed for the solver axis

Grasp containment was the standing "revisit whenever another knob moves" lever, because on the
contained task joint space needs 970 median iterations against 48 on the free one. Stage SOLVER2
fielded those rows under both solvers. **It does not move.** SNOPT never flips a verdict toward the
learned arm and flips one away from it (Panda contained paired goes from a decisive learned win
under IPOPT, 437-323 p = 2.3e-20, to a tie under SNOPT, 280-298 p = 0.26). The containment verdicts
stand exactly as measured under IPOPT; step rejection is the remaining candidate.

### Future work on this axis

- **NLopt settings are unswept, by decision** (Thomas: *"Store testing NLOPT settings as future
  work"*) — `LD_AUGLAG` vs `LD_AUGLAG_EQ`, `constraint_tol`, `xtol_*`, `max_eval`. The column is not
  competitive at Drake's defaults by a very wide margin, so a sweep would only say *why*. The AL's
  inner local optimizer is not selectable on the cluster's Drake at all — leaving it unset is a
  supported state (NLopt supplies LD_LBFGS) — **TODO** when that Drake carries the local-optimizer
  PRs.
- **The step-rejection family is untouched and is the next question.** `snopt_major_step_limit` and
  `snopt_violation_limit` are plumbed and unset, as are IPOPT's three. `Major step limit` bounds
  `||dx|| <= limit*(1+||x||)` per major iteration and `Violation limit` is the counterpart of
  `ipopt_theta_max_fact`.
- **Do not extend Drake to get better instrumentation.** Thomas: *"NLOPT might not have the robust
  logging we need btw, work with what you have, don't write new logging stuff in Drake or
  anything."*

## Results: the current campaign

All numbers below are on the **hardened scene** with the corrected program (true pose equality,
`correction_cost_weight = 10`, calibrated conditioning frame on every task, `--compile`, IPOPT),
480 cells = 60 targets x 8 guesses, seed 1 (out of sample), 45 s unless stated. Earlier campaigns
(final3-5, Stages A-D, the legacy-scene headline tables, the pre-calibration pose tables) are
**superseded and their tables removed**; git history holds the originals, and every conclusion of
theirs that still stands is restated here on corrected numbers.

Adopted defaults: **grasp targets free** (containment available behind `--target-placement shelf`),
**pose containment at the fingertips** where containment is used at all. Adopted rungs: Panda `n6`,
iiwa `n4`.

Harness self-check, which passes everywhere below: joint space is identical across every chart rung
of a robot within an experiment, and `median_start_q_error` is 0.0 exactly under `paired`.

### Grasp task, adopted default (hardened scene, free targets)

| panda | upstream | n12 | n8 | **n6** | n4 | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp native | 432 | 437 | 465 | **474** | 479 | 456 | 453 |
| grasp paired | 417 | 409 | 463 | **475** | 474 | 443 | 453 |

| iiwa | ddpr1 | n8 | n6 | **n4** | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- |
| grasp native | 282 | 318 | 294 | **446** | 311 | 457 |
| grasp paired | 286 | 336 | 297 | **457** | 336 | 457 |

Best rung against joint space: Panda `n6` **474 v 453** native (p = 0.00032) and **475 v 453**
paired (p = 0.00011); iiwa `n4` 446 v 457 native (p = 0.14, tie) and **457 v 457** paired (exact
parity — the first time that row has not been a deficit).

### Pose task

| panda | upstream | n12 | n8 | **n6** | n4 | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pose free, native | 466 | 471 | 455 | **458** | 453 | 447 | 201 |
| pose free, paired | 294 | 293 | 418 | **438** | 390 | 303 | 201 |

| iiwa | ddpr1 | n8 | n6 | **n4** | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- |
| pose free, native | 442 | 443 | 452 | **470** | 431 | 332 |
| pose free, paired | 284 | 267 | 269 | **447** | 354 | 332 |

**The learned arm wins every pose row decisively**, at every placement — e.g. wrist-contained
paired, Panda `n6` 382 against 185 at **p = 2e-42** and iiwa `n4` 384 against 268 at **p = 5e-16**. On the `paired` protocol the full-depth
charts collapse on both robots (Panda `upstream`/`n12` 242-294 against `n6`'s 382-438; iiwa
`n8`/`n6` 232-269 against `n4`'s 384-447) — that is the gain-ceiling runaway, present in the free
columns too, not a containment effect.

**Containment at the fingertips beats containment at the wrist, on both counts.** The two candidate
points are one 0.100 m step apart along the gripper — the same step on both robots, which is why
containment is keyed on the gripper rather than on each task's own target frame. Best rung, learned
v joint space:

| | free | wrist | **fingertip** |
| --- | --- | --- | --- |
| panda `n6` native | 458 v 201 | 428 v 185 | **461 v 217** |
| panda `n6` paired | 438 v 201 | 382 v 185 | **405 v 217** |
| iiwa `n4` native | 470 v 332 | 433 v 268 | **470 v 299** |
| iiwa `n4` paired | 447 v 332 | 384 v 268 | **422 v 299** |

Both arms score higher at the fingertips on every row, which is what the geometry predicts: putting
the *hand* in a compartment is a shallower reach than driving the *wrist* in behind it, so the wrist
definition silently demands 0.1 m more penetration into a 0.10 m compartment. Fingertip is also the
more faithful statement of the task, and preserves the learned margin at least as well as the wrist
on three rows of four.

**Pose containment is a genuine difficulty increase, not a free win**, and this reverses a
pre-calibration conclusion that recommended adopting it. On the corrected program containment costs
the learned arm 2-3.5x what it costs joint space on the Panda (margin −14 native, −40 paired) and is
a wash on the iiwa (+27, +1). **Whether to keep it is Thomas's call.**

### What the success counts hide: headroom and rescue rate

The grasp rows read as a narrow Panda win and an iiwa tie. That is an artefact of **headroom** —
joint space is at 94-95% there, so only 23-27 cells of 480 are available to win at all:

| config | L | JS | L only | JS only | both | neither | of JS's failures, rescued |
| --- | --- | --- | --- | --- | --- | --- | --- |
| panda native, free | 474 | 453 | 27 | 6 | 447 | 0 | **27/27 = 100%** |
| panda paired, free | 475 | 453 | 27 | 5 | 448 | 0 | **27/27 = 100%** |
| iiwa native, free | 446 | 457 | 17 | 28 | 429 | 6 | 17/23 = 74% |
| iiwa paired, free | 457 | 457 | 17 | 17 | 440 | 6 | 17/23 = 74% |
| panda native, contained | 462 | 323 | 148 | 9 | -- | -- | **148/157 = 94%** |
| panda paired, contained | 444 | 323 | 142 | 21 | -- | -- | **142/157 = 90%** |
| iiwa native, contained | 391 | 442 | 30 | 81 | -- | -- | 30/38 = 79% |
| iiwa paired, contained | 407 | 442 | 30 | 65 | -- | -- | 30/38 = 79% |

**On the Panda the learned arm solves every single cell joint space cannot** — 27 of 27 under both
protocols, with `neither` = 0. And the iiwa's 457-vs-457 is not the same 457 cells: 17 each way, so
the arms are genuinely complementary even where the totals agree. **The rescue rate is high and
stable everywhere — 74-100%** — and that is the quantity the success counts obscure. Containment is
what creates headroom (27 cells → 157 on the Panda), which is the argument for revisiting it.

### The last deficit was budget, and it is gone at 180 s

The iiwa's contained-grasp loss was predicted to be cap-bound near-misses rather than divergence.
iiwa `n4`, contained grasp, 480 cells, caps against the existing 45 s column on the same grid (the
cap does not enter target sampling, so this pairs cell for cell):

| start | cap | learned | js | timeouts | median iters | median `max_violation` | vs 45 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native | 45 | 391 | 442 | 88 | 434 | 3.0e-08 | **p = 1.4e-06 (js wins)** |
| native | **180** | **447** | 442 | **0** | 629 | 2.0e-08 | +57/-1, p = 4.1e-16; **tie, p = 0.60** |
| native | 360 | 447 | 442 | 0 | 629 | 2.0e-08 | identical to 180 |
| paired | 45 | 407 | 442 | 75 | 398 | 2.7e-08 | **p = 0.00042 (js wins)** |
| paired | **180** | **453** | 442 | **0** | 570 | 1.9e-08 | +46/-0, p = 2.8e-14; **tie, p = 0.19** |
| paired | 360 | 453 | 442 | 0 | 570 | 1.9e-08 | identical to 180 |

Timeouts 88 → 29 → 0, the 360 s column reproduces 180 s *exactly*, and the gain is almost purely
one-directional (+57/-1, +46/-0), which is what recovering stalled cells looks like rather than
resampling noise.

**So the last deficit in the project is an implementation property, not a formulation or chart
property.** It is not that the learned formulation cannot express these grasps; it is that one
iteration costs 35 ms against joint space's 3.6 ms. Reported honestly, that parity costs **180 s
against 1.6 s** — 629 median iterations to joint space's 448, at ~10x the per-iteration price, so
roughly a 14x wall-clock premium for a tie. **Anything that lifts the learned arm to ~450 cells
inside 45 s closes this row**; step rejection is the candidate, and a better chart is not, because
`n4`'s violations here are 2e-08 — it is converging correctly, just slowly.

The learned arm's own residual failure set (28/17 cells free, 81/65 contained, cells joint space
solves) is every one `fail_reason = "constraint"` with median `max_violation` 0.009-0.037 —
**centimetres off, not astronomical** — at 286-434 median iterations and 34-36 ms/it. The runaway
signature is `max_violation` >= 1e+03; this is 1e-02. **A convergence problem, not a runaway.**

### Iterations, cost and wall clock

Medians over succeeded cells; cost on cells **both** arms solved, learned-only regularizers
excluded.

| experiment | L iters | L s | ms/it | JS iters | JS s | n both | L cost | JS cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| panda `n6` grasp native (contained) | 200 | 8.77 | 44 | 970 | 4.26 | 314 | 7.510 | **5.305** |
| panda `n6` grasp paired (contained) | 301 | 13.39 | 44 | 970 | 4.24 | 302 | 6.846 | **5.164** |
| panda `n6` pose contained paired | 142 | 5.56 | 39 | 44 | 0.14 | 140 | **9.226** | 9.901 |
| iiwa `n4` grasp native (contained) | 434 | 14.65 | 34 | 448 | 1.63 | 361 | 4.930 | **2.827** |
| iiwa `n4` pose contained paired | 179 | 4.86 | 27 | 231 | 0.10 | 231 | **5.743** | 6.065 |

Two things the success columns hide. **The hardened grasp task costs the joint-space arm its
cheapness** — 970 and 448 median iterations against its archived 48 and 66 on the soft problem, so
its per-cell wall clock rose ~30x on the Panda, and the learned arm's per-iteration penalty is
correspondingly smaller here (~10x rather than the ~13-30x of the soft problem). And **the cost
split by task survives hardening**: the learned arm wins on cost on the pose task on both robots
and loses by ~1.4-1.7x on the grasp task. On the soft problem the pose-task cost win held in all
four rows (1-10%), which is the draft's central claim holding on the second of its two axes.

**What the correction penalty costs.** On cells both solved, `w = 10` against `w = 0`: pose task
0.5-5% more expensive, grasp task **30-100%** more expensive — and the grasp task is exactly where
it buys its cells (+68 to +157 of 480, p ≤ 2.1e-10 on all four grasp rows, every pose row a tie).
So the penalty is a **trade**, not a free improvement: it converts objective value into
feasibility. The mechanism is the redundancy it was adopted to break — with `q_c` free the arm can
nudge `q` toward a well-centred configuration for nothing; pinning `q_c` to zero means `q` is
whatever the flow emits at `(c, z)`. This is not a bookkeeping artefact of the penalty term
appearing in the objective: at `w = 10` the correction is driven to `|q_c|_inf ~ 1.8e-05`, so the
term contributes ~2e-08 to a cost of ~5.

Deeper mechanism, and it is *not* that the correction box was binding (`on the box` is 0.00 at
every weight): with `c` and `q_c` both free, many pairs give the same `q`, so the active constraint
gradients are rank-deficient and IPOPT spends its budget on a degenerate direction. Penalising
`q_c` breaks that degeneracy, which is why it bites hardest where the active set is largest. As the
weight rises the correction is driven to zero and the median constraint violation falls three
orders on the iiwa (2.65e-02 at w=0.001 → 2.61e-08 at w=10) while the latent stays put. Weight 30
is flat or worse on three of four 60-cell rows, so **10 is at or near the optimum**.

### Settled negative results on the knobs — do not re-sweep

- **Collision shaping** (`collision_influence_offset`, `collision_row_scale`): peaks weakly around
  0.2-0.4, within noise of the default.
- **`ipopt_mu_strategy=adaptive`**: inert on both robots despite the archived logs looking like its
  textbook case.
- **`latent_cost_weight`**: helps the Panda (48 at 0.1), hurts the iiwa (14). Not a general win.
- **`correction_bound`** swept 0.1/0.2/0.4/0.8: inert. The box never binds, the solver takes more of
  it when given more (median `|q_c|` 0.054 → 0.484) and gets nothing for it.
- **The shelf-depth inset** (0 / 0.05 / 0.10 / 0.125): success moves 2-10 cells of 60 across the
  whole span with no monotone trend and **no ordering flip** on any row. What it moves steeply is
  acceptance (0.60% → 0.10%), so it buys sampling cost, not difficulty. **0.10 m stands.**
- **Scene clutter** (stage HARDMUG: the iiwa's seven welded decorative mugs, four inside
  compartments, kept against removed): worth −6 to +5 cells of 480 to the learned arm and −14 to
  +12 to joint space. The suspected Panda/iiwa confound — that the Panda grasp scene never had those
  mugs, so hardening it is near-pure containment — was real in principle and **immaterial in
  practice**. The iiwa/Panda divergence is a property of the robots. Do not re-open this.

## The chart: why the rungs differ and how one is chosen

The iiwa's upstream checkpoint was the project's bottleneck. `iiwa14_ddp_r1` (620k steps) cut
`frac_gt_1000` from 3.34% to 0.0125% but left `pole/max` at 7.8e9 and iiwa grasp at 270/480 — the
headroom was still there, that run had only moved less of the domain into it. The follow-on trained
deliberately **simpler, less accurate** charts: `nb_nodes` 12/8/6/4 lowers the architectural gain
ceiling `exp(2.4975*nb_nodes)` from 1e13 to 2e4, with a width-only rung (12 blocks,
`coeff_fn_internal_size` 256) as the control separating "less accurate" from "less headroom". Nine
runs, both robots, 620k steps each; `cluster/ladder_runs.txt` is the spec.

**Reducing depth from 12 helps on both robots, and the optimum rung differs by robot** — `n4` on the
iiwa, `n6` on the Panda. "The smallest chart wins" was an iiwa-only result.

**The two mechanisms differ by robot.** On the iiwa a smaller chart buys cells by eliminating runaway
configurations; on the Panda, which has no runaway at all (`median_max_violation` 1e-08 on every
rung), it buys them by fitting more iterations inside the fixed cap. Per-iteration cost falls
monotonically with depth — iiwa grasp native 80 / 55 / 42 / 33 ms for n12 / n8 / n6 / n4, with
`n12w256` at 72 holding depth — and timeouts collapse with it. The learned arm's per-iteration
penalty is now **~10-13x, down from ~30x**. That second mechanism is an implementation-and-hardware
property, which is why ms/it must be reported beside success. A net-faster chart that needs more
steps is a **trade-off to report, not a confound to remove** (Thomas: *"More iterations but faster
net is a trade-off, not automatically good or bad"*) — do not redesign to a fixed iteration cap.

**Two controls land as intended.** Panda `upstream` and `n12` agree within noise on all rows, so our
training recipe reproduces Jeremy's and no reduced-Panda result is confounded with recipe. And
`n12_w256`, the accuracy-only control, never beats `n12` significantly while keeping the slow
iteration: width costs accuracy without buying either headroom or speed. Depth buys the speed.

**Chart accuracy is a clean monotone dose curve and it runs BACKWARDS to cells.** Median FK error
over 5000 poses, 4/6/8/12 blocks: 20.0 / 12.1 / 11.3 / 10.1 mm on the iiwa, 14.7 / 9.5 / 7.2 / 6.1
mm on the Panda; the width rungs are 75.1 and 22.9 mm. The *least* accurate depth rung solves the
most on the iiwa. 20 mm chart error does not appear in the solutions — the IK constraint is on
`FK(q)`, so `n4`'s solved cells return `median_max_violation` 1.1e-08; the chart only decides where
the solver starts.

**Neither intrinsic screen predicts cells, in either direction.** iiwa `n8` screens cleanest of the
four (task-pose `frac_gt_1000` 0.0, `pole/max` 202) and is the worst rung on pose paired; `n4`
screens dirtier and solves best. A chart can be clean everywhere the sampler looks and catastrophic
everywhere the Newton step goes. **The screen is a smoke test, not a selection criterion.**

**Pole mass is CREATED BY TRAINING, monotonically, on every rung.** Across all 31 kept checkpoints
per rung, task-pose `pole/max` over 20k..620k steps runs 27 → 2.5e3 (iiwa `n4`), 409 → 1.3e5 (iiwa
`n6`), 50 → 3.5e11 (Panda `n12`). The headroom is present at initialisation and barely used; SGD
walks the network into it while buying accuracy. Accuracy itself is converged by ~480k.

**Stage TRAJ measured the training-step axis directly: training makes the iiwa `n6` chart WORSE and
does nothing at all for `n4`.** Exact McNemar, earliest checkpoint against 620k: `n6` **loses three
of four rows to its own first checkpoint** — grasp native 22/159 (p = 8.3e-27), grasp paired 15/142
(p = 4.1e-27), pose paired 63/162 (p = 3.1e-11); `n4` moves on **none** of the four (p = 0.088 to
1.0). `n4` at 20k has ~78 mm median FK error against 620k's 20 mm — a 4x accuracy gain over 600k
steps, worth **zero cells** — while `n6` buys the same accuracy and *pays* 137 grasp cells, its
`pole/max` climbing 409 → 1.3e5 and its pose-paired `median_max_violation` crossing 1e-08 → **1e+03**
between 240k and 400k. **The sharpest statement the campaign has that accuracy is not the quantity
that matters.** It is **not** a licence to pick early checkpoints: that is selecting on the test set
and confounds architecture with selection.

### The chart-selection rule

**Choose an architecture whose gain ceiling `exp(2.4976 * nb_nodes)` sits below the ~1e7 runaway band
— on the iiwa that is `nb_nodes = 4` — then take the final checkpoint, because below the ceiling the
training-step axis is flat and above it later checkpoints are strictly worse.**

The ceiling is `atan` and the block count, nothing else: FrEIA's coupling block is `y = exp(s)*x + t`
with `s = clamp * 0.636 * atan(s_raw)`, so `|s| < clamp*0.636*pi/2 = 2.4976` at `rnvp_clamp = 2.5`
and one block amplifies by at most 12.15x. Over `nb_nodes`: 2.2e4 / 3.2e6 / 4.8e8 / 1.0e13 at
n4/n6/n8/n12. It bounds the weights, so no amount of training escapes it — and **training reliably
walks 78-89% of the way up whatever log-ceiling it is given**, so the ceiling predicts where a
trained chart lands rather than merely bounding it. The configurations that actually kill solves are
**1e7 to 1e16 rad**; `frac_gt_1000`'s threshold is a bimodality separator (ordinary configurations
~2.5 rad), *not* the level at which a solve dies — `n4`'s ceiling is above 1000 and is fine. 6→4 is a
cliff, not a dose curve. Every competing criterion is disqualified by measurement (accuracy,
intrinsic screening, "take the last checkpoint", "take whichever benchmarks best"). If a chart above
its ceiling ever *must* be selected among, the only sound version is a held-out optimization grid.

**`rnvp_clamp = 2.5` is confirmed correct** — worth checking because `src/iiwa_program.py` hardcodes
the iiwa's hyperparameters and `rnvp_clamp` changes the forward pass without changing any parameter
shape. Sweeping it against the same weights, fraction `> 1000`: 0.893 / 0.410 / 0.090 / **0.035** /
0.126 / 0.999 at clamp 1.0 / 1.5 / 2.0 / **2.5** / 3.0 / 5.0. A clear optimum.

**A path bug cost a rung its first measurement (fixed in `6fbff55`).** `train_flow.sh` reassigns
`HOME="$ROOT/home"` so ikflow resolves `DATASET_DIR` at import; `export_and_screen_job.sh` derived
its own `ROOT` from `$HOME`, so the inline export resolved every path one level deep and died. The
rung trained all 620k steps and exported nothing, and four interleaved benchmark jobs fired at a
checkpoint that did not exist — **failing per cell rather than fast**. `submit_bench.sh` now refuses
when a manifest names a checkpoint absent from the cluster (`dd24cc3`).

## The gain ceiling: the project's central scientific finding

This explains the residual learned-arm failures on the full-depth charts and the historical iiwa
grasp deficit.

**The violated binding is `AllIKFlowConstraints`, and inside it the joint-limits row.** On cells that
never converge, `max_violation` equals `|q|_inf` exactly (Spearman 1.0 on 55 of 64 badly violating
cells), with joint angles of **1e7 to 1e16 radians**. Everything else follows: a configuration of
1e8 rad puts the gripper anywhere, so "deeply in collision, a metre off target" is a *consequence*,
and the collision penalty is a bystander (Spearman −0.15).

**Every runaway lies on one ray, and it is a property of the network, not of the solve.** Normalising
the exploded `q` vectors: the iiwa's ray is `[.001, -.001, .016, -.000, .003, .978, -.208]` and the
Panda's `[-.016, .033, .998, -.032, -.002, .027, .023]`, both at pairwise `|cos| = 1.0000` across two
tasks and both protocols — dominated by a single joint (iiwa wrist, Panda elbow).

**Sampling the network directly reproduces it, with no Drake and no solver involved.** Draw `c`
uniformly in its ±0.25 m box, a uniform unit quaternion, and `z` uniformly in the ball of radius 4.3
— strictly inside the allowed region — and evaluate `MakeFlowInference` in float64: median `|q|_inf`
~2.5-2.65, but p100 of 20000 is 4.1e+12 (Panda) and 5.5e+16 (iiwa), with **fraction > 1000 rad
0.00065 (Panda) against 0.0334 (iiwa) — a factor of 51** — and the ray recovered from those samples
matches the one the solver landed on to `|cos|` 0.9976-0.9999. The distribution is bimodal, not
heavy-tailed: a draw is either an ordinary configuration or it is astronomical.

**The mechanism is architectural headroom, not a numerical bug** — the gain ceiling above. These are
near-worst-case gain regions of a bounded map, not poles. Both checkpoints have the same headroom;
they differ only in how much of the conditioning domain sits near it.

**Why the solver finds a 3%-measure set 30% of the time.** It does not sample; it follows gradients,
and `dq/dvars` in these regions is as large as `q` is, so a Newton step is *attracted* to them. That
is also why more budget never helps: cap-bound cells at 180 s are the same cells that were cap-bound
at 20 s. **The learned arm does control `q`** — Thomas: *"the network does get to control the joint
limits a bit, since it can adjust z. That's the whole point of differentiating through the network —
we take the constraint gradient for joint limits and pull it back through the network to z."* What
differs from the baselines is *when* the limits hold (only at convergence), and that the gradient
into a high-gain region is itself enormous.

**Neither region knob avoids them, because they are not at the edges.** Fraction of the region with
`|q|_inf > 1000`: the latent trust-region radius over 1.0..8.0 leaves the iiwa at 0.026-0.071 and the
Panda flat at 0.0018 — shrinking to `R = 1` changes nothing, because the blow-up regions are spread
through the domain including at `|z| <= 1`. **This is why the trust-region sweep measured inert.**
`c_position_slack` is flat from 0.05 to 0.25 and then a **cliff** at 0.5, where exposure jumps 6x on
the iiwa and 76x on the Panda; the default sits just under it by luck, worth knowing before anyone
widens it.

### Every optimization-side remedy has been measured and refuted

Thomas's ranking: **(1) a better chart is preferred over everything else** — *"All of these actions
are less preferred than just getting a better iiwa chart"*; (2) lifting `q` into a bounded decision
variable, permitted but disliked (*"we're effectively adding a nonlinear equality constraint"*); (3)
a joint-limit penalty, permitted but disliked. **"You can try it, but I don't like it" means measure
it and report it as a stated deviation, not adopt it if the numbers look good.**

| remedy | verdict |
| --- | --- |
| **chart accuracy** (`chart_error_scale`, a smooth seeded perturbation) | **not the mechanism.** Degrading the Panda's chart to the iiwa's accuracy (~12-20 mm) costs it 1-3 cells of 60 (35 → 34 → 32); the iiwa is 23 cells worse. The standing chart-accuracy hypothesis does not survive its own experiment. Smooth `sin` error also adds no high-gain regions, so this could never have reproduced the real pathology. |
| **IPOPT scaling** (`nlp_scaling_method=none`, `nlp_scaling_max_gradient` 1e4/1e8) | **inert**, five experiments x 60 cells, largest movement ±2 cells, no p < 0.5. Not a scaling artefact: IPOPT is being handed a chart with gain ~1e13 and following the gradient into it. (`equilibration-based` needs HSL MC19 and raises on construction.) |
| **joint-limit penalty** at 1/10/100 on four rows | **inert** — smallest p = 0.115, no consistent direction, runaway counts unmoved, median violation unchanged at ~1e-08. Adding a penalty on a quantity a constraint row already governs buys nothing, vindicating Thomas's objection. |
| **lifting `q`** (`lift_q`: `q` a bounded variable, chart as a 7-row equality) | **net negative, 38 better against 116 worse over eight experiments, p = 2.2e-10.** It delivers exactly one thing universally — **0 runaway cells in all eight** — but **the runaway does not stop, it relocates**: the flow still reaches 1.19e10 and it lands in `max_violation` instead of `q` (median 3e+06 to 7e+06 on the collapsing pose rows). The split is by **task**: neutral-to-positive on grasp with *fewer* iterations, catastrophic on pose paired (40 → 9 Panda, 40 → 2 iiwa). Mechanism: the pose task already pins the end-effector with six equality rows, so lifting adds seven more — thirteen equalities in 27 variables with the flow's badly scaled Jacobian inside seven. **An interior-point method's tolerance for a badly scaled row depends on how many equalities it is already carrying.** The 480-cell replication was deliberately not run. |
| **Jacobian regularization** (Frobenius clipping, Tikhonov/LM damping of the singular values, an SVD floor) | **clear negative, line closed. 310 cells better against 1,509 worse** over ten variants x eight experiments. Not one setting is a significant improvement; the only one not net-worse (`jacobian_max_norm=1000`, +6, p = 0.59) does not reduce the runaway it was introduced to prevent, and the most aggressive damping scores **0/480**. |

**Why damping cannot work, the transferable part.** The flow's Jacobian is the *exact* derivative of
an explicit function. Where the gain approaches its ceiling, a sensitivity of 1e13 is the correct
answer, not an artifact. Damping it breaks the correspondence between the constraint values IPOPT
evaluates and the gradients it is handed, leaving an inconsistent nonlinear program — which is why
the most aggressive damping fails hardest while *increasing* the runaway count. LM damping is sound
applied to the **Newton step** rather than to a reported derivative, but Drake's IPOPT does not
expose the step computation. **Thomas's ruling: *"I think we can conclude gradient regularization and
the other strategies isn't worth it."*** The knobs stay in the tree, off.

A lesson these stages share: **suppressing the runaway does not buy success.** `tikhonov=10` cut
runaway cells 31 → 7 for exactly 19/19 on success; `lift_q` reached zero runaways while losing 78
cells net.

### Step ACCEPTANCE is a different lever, and the best open lead

Everything refuted above altered the *derivatives* the solver was handed. Filter tuning leaves the
program exactly as written and changes only which trial points are accepted. The mechanism it
attacks: IPOPT holds **variable bounds** at every iterate but general constraints only at
convergence, and in the learned formulation `q` is not a decision variable, so the joint-limit rows
are general constraints and an iterate may sit at `|q| = 1e8`. That is why `lift_q` gave zero
runaways, at the cost of seven equality rows. `ipopt_theta_max_fact` attacks it without touching the
formulation: IPOPT rejects any trial point whose constraint violation exceeds
`theta_max_fact * max(1, theta(x_0))`.

The trajectories say why this should work. Recording every iterate through `VarsToQ` on five runaway
cells, **the solve is well behaved for 22 to 110 iterations and then jumps in a single step** — on
three of the five `|q|_inf` goes from ~2.5 to past 1e3 in one iterate. Not a slow drift the solver
could be nursed through; **one accepted catastrophic step, which is exactly what a filter ceiling can
refuse.**

Local probe only, 16 cells, one seed, one chart — a lead, not a measurement: default 11/16 with 5
runaways at median cost 7.61; `ipopt_theta_max_fact=1` **14/16 with 2 runaways at cost 6.75**; `=10`
bit-identical to the default; `ipopt_watchdog_trigger=0` and `ipopt_max_soc=8` both 11/16. Three
cells gained, **none lost**, cost *improved*, effect sharply thresholded between 1 and 10 — the first
intervention that both suppresses the runaway **and** converts it into success. All three fields are
plumbed and default to None. **Not measured**: needs the grasp task, the Panda, both protocols, 480
cells.

**Step control is not only runaway prevention.** Thomas: *"step rejection (and trust region ideas)
can still help even if we're not running away. There's a reason people like trust region solvers."*
The runaway *mechanism* is absent on the adopted `n4`/`n6` rungs, and that does not retire the lever:
a neural-network chart produces ill-conditioned derivatives whether or not any iterate blows up, and
the live targets are the cells that converge too slowly and the cells that stall at a moderate
violation with budget left. Trust-region *options inside an existing solver* are in scope;
implementing a trust-region *solver* is a different project.

## The analytic chart: eight branches, and what the last 0.6% is

The closed-form map's discrete set is three binary choices — wrist (B), shoulder (C), elbow (A) — and
the implementation historically charted only A = +1, the half away from the joint limits. The missing
half is a *single sign*: negate both triangle angles `O2O4O6` and `O2O6O4`. The old commented-out
"Case A1" line matches no configuration. The measured elbow relations are `q3 = theta + q3_add - 2*pi`
(A = +1) and `-theta + q3_add` (A = -1), partitioning at `q3 = q3_add - pi = -0.467`.
`ProgramOptions.analytic_branches` selects the chart (default 4, so archived runs stay reproducible);
`gc(q, branches=3)` recovers all three indices with zero mislabels in 4000 samples. Round-trip
coverage of `IK(FK(q), psi(q), gc(q)) == q`: 89.4% (4 branches) against 99.40% (8) at 1e-6, rising to
99.83% at 1e-2.

**The residual is not singularities** (measure zero; this set has positive measure) **and not branch
mislabelling** (the 24/4000 misses are reproduced by *no* branch of the eight). Two are off by ~4 rad
and 22 by 1e-3 to 1e-2, clustered where the wrist arcsin argument approaches 1. Consistent with the
<=16 self-motion-manifold bound (Burdick/Luck). **Left as future work by decision** —
arXiv:2503.03992 is the starting point — and until then coverage is reported as the curve above,
never as "100% up to singularities". The iiwa needs no such column: its Faria/SRS implementation
already charts all eight branches, so analytic4-vs-analytic8 is a **Panda-only** experiment.

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

## Running on MIT SuperCloud (`cluster/`)

`cluster/README.md` is the playbook and `~/.claude/skills/supercloud/SKILL.md` carries the standing
rules; this is what a reader of *this* file needs. Allocation: **4 nodes on `xeon-g6-volta`**, each
40 Xeon Gold 6248 cores and 2x V100 32 GB.

**Timing is never compared across machines.** Thomas: *"There's never a need to compare wall-clock
(or really, performance in general) between laptop and cluster. But wall clock limits can be
adjusted on the cluster."* So the wall-clock cap stays as the measurement — no switch to iteration
caps for portability's sake — but its value comes from `cluster/calibrate.sh` on that hardware.
`metadata.host`/`metadata.device` exist so a cluster run cannot be paired against a laptop one. The
corollary: **CPU contention still corrupts the measurement**, so workers per node is a measured
quantity.

**`--shard K/N` is a no-op by construction.** It splits **target-major** — whole targets per shard,
never a target's guesses split — because `success_ci` bootstraps over whole targets and
`solved_within_k` counts restarts within one, and it appends `_shardKofN` to the tag (without which
two shards overwrite each other's `summary.json`). `cluster/merge_shard_summaries.py` pools the
records and **re-runs `summarise`** rather than stitching per-shard numbers, preserving arm order so
`_mcnemar`'s pair directions survive. `bash cluster/verify_sharding.sh` proves the round trip in ~2
minutes — **run it after any change to sharding, the merger, or grid construction.**

**Two cluster facts that shaped the design.** The account's `xeon-g6-volta` limit is a Slurm
**`GrpTRES` group** cap (`node=4`, `MaxSubmit=240`), not a per-job `MaxNodes`: work beyond it is
accepted and **queued**, so a whole stage is submitted at once and Slurm meters it — and the cap is
shared with everything else the account runs. Because jobs start at different times there is no
stable rank space to deal work into, so `cluster/run_items.sh` claims items with an atomic
`mkdir <id>.claim`. And **PyTorch 2.11's cu128 wheels dropped sm_70**, so the V100s need a cu126
build; the wrong wheel imports cleanly, reports a CUDA device, and fails only at the first kernel
launch, which is why `cluster/smoke.sh` launches a real kernel. Three smaller adaptations: Meshcat is
optional (`BuildEnv(meshcat=None)`), mug scenes are built only for the shard's targets, and
`hit_iteration_cap` is the counterpart to `timed_out` (`is_timeout` does not match IPOPT's "Maximum
Number of Iterations Exceeded").

**The calibration.** Four arms, one per node, each a full job on a real partition, on the **Panda
grasp** task — that task specifically, because it is the one that binds against the cap. A first
attempt ran the *pose* task and measured nothing: a pose cell converges in ~74 iterations here, so
its iteration count is identical at every cap and however contended the node is. **A converged solve
takes the iterations it takes**; only its wall time moves.
- *Workers per node*, median iterations inside a fixed 20 s cap: 202 / 196 / 194 / 186 / 114 / 70 at
  P = 1 / 2 / 4 / 8 / 20 / 40. `PROCS=4` is conservative, `PROCS=8` is what exploratory stages ran at
  (Thomas ruled the 8% acceptable for sweeps, reserving uncontended runs for hard comparisons and
  paper numbers). A worker that gets less done inside a wall-clock cap is a **different
  measurement**, so the count is held fixed across everything compared.
- *CPU-only is not competitive*: at one worker the GPU reaches 202 median iterations against 62,
  solving 4 of 8 cells against 1 of 8. Not a contradiction of the CPU-bound profiling result — that
  says the GPU is never the bottleneck *while a GPU is present*.
- *The cap*: median iterations 142 / 203 / **338** / 338 / 338 at 10 / 20 / 45 / 90 / 180 s. **45 s
  is the campaign's cap** — medians saturate by 45 s with 2x headroom; beyond that only the tail
  gains, which is what the cap curve is for.
- *Startup*: 40 concurrent `import torch, pydrake, ikflow, jrl` take 10 s total, so the venv can stay
  on the shared filesystem (copying to node-local `$TMPDIR` costs 231 s and buys nothing).
  `torch.compile` costs ~35 s cold, ~17 s warm per process.

**Solver logs are node-local, one archive per run.** `src/benchmark.py` once wrote one ~20 KB IPOPT
log per (cell x arm) onto the shared filesystem — **35,596 of them, 87% of every collection's file
count**, exactly the many-small-files pattern SuperCloud warns against; Lustre is metadata-op bound
at that size and a routine collection had drifted from three minutes to thirty. Per-cell logs now go
to node-local `$TMPDIR` (keyed on run tag *and* pid, since a node runs eight workers) and roll into
one `solver_logs.tar.gz` at the end of `run_grid`. After: 40,733 files → 2,964, 950 MB → 316 MB,
~30 min → **43 s**. `collect_results.sh` is incremental by default and never ships `state/`.

**Any cluster-wide check on a shared account must be scoped to this project's own jobs.** Both
scripts used to refuse while *any* job was running (`LLstat | grep -c RUNNI`), which fired on an
unrelated campaign's job since SuperCloud accounts are shared across Thomas's projects. Both guards
now filter by this project's job name and count `PENDING` as well as `RUNNING`.

**The laptop suspends when idle.** Every multi-hour stall this repo recorded — the archived 6106 s
cell and four overnight "wedges" — was the machine going to sleep (GNOME suspends after 900 s idle
*even on AC*; `journalctl` matches every stall to the minute). Three wrong solver-level theories each
fit part of the evidence: it struck only *unattended* runs (16/16 attended reproductions clean),
`timeout`/`sleep` run on CLOCK_MONOTONIC which pauses across suspend, and a CUDA context straddling a
suspend leaves torch spinning at 100% CPU. **Any long unattended run here must hold a sleep
inhibitor** (`systemd-inhibit --what=sleep:idle --mode=block`, launched with `setsid`), and a "hung"
unattended process is diagnosed by checking `journalctl -b | grep "suspend now"` against the stall
window *first*. Long benchmarks now run on the cluster.

**Reproducibility at the cap is ±1 cell** — two runs of the same configuration scored 34/60 and
35/60, the differing cell hitting the wall clock in both. Worth remembering before reading a
one-cell difference as a real effect.

**Plan around SuperCloud's monthly maintenance** — second Tuesday, compute down Monday evening to
Wednesday morning, nothing survives it. Next window: 2026-10-12 to 10-14.

## Where the project stands, and what is next

**The learned formulation wins the pose task decisively on both robots and every placement, wins
the Panda grasp task, and ties the iiwa grasp task** (at parity under `paired` on the adopted free
default; at 180 s under containment). The one honest caveat everywhere is per-iteration cost: ~10x
joint space's, which is an implementation property with a known ~3x dispatch floor and is **out of
scope to fix** (Thomas: architecture/infra work on the CPU dispatch bottleneck is "future
work/possibly not in scope at all"). It is a number to report.

Thomas's roadmap (2026-09-04), with status: **(1) iiwa checkpoint training — DONE**, the
reduced-capacity ladder is trained, measured and has a selection rule; **(2) SNOPT and NLOPT —
DONE**, see the solver axis; **(3) performance tuning and formulation tweaks for getting the best
results with the learned formulation** — the live item. Note that Thomas naming formulation work as
a work item is **not** a standing licence to invent formulations; what is compared remains his call,
made explicitly in advance.

Live items:

- **Step rejection (IPOPT filter tuning)** is the best open lead and the remaining candidate for the
  iiwa grasp row — see "Step ACCEPTANCE" above. Needs the grasp task, the Panda, both protocols,
  480 cells. `snopt_major_step_limit` / `snopt_violation_limit` are its SNOPT counterparts.
- **Whether pose containment is adopted** is Thomas's call; fingertip is the better point if it is.
- **A harder problem formulation** beyond the hardened scene, if he wants one — his idea, his call.

### Smaller open items

- **Fold a small optimization smoke run into checkpoint validation.** Thomas's idea, explicitly
  deferred (*"Obviously, not worth it right now, but a cool idea for the future"*).
  `cluster/export_and_screen_job.sh` screens on intrinsic metrics only, and those do not predict
  cells. A handful of cells through `src/benchmark.py` at export time would catch a chart that
  screens clean and solves badly; the export job already loads the solver.
- **More guesses per target** in the paired grid — same guesses for every arm — reported as "solved
  within k restarts". The only honest form of multi-start, and the harness already does it.
- **Non-dimensionalise the conditioning pose's translation against its rotation**, the way
  `eaik-experiment` scales its Jacobian rows by a 1.12 m length scale. Never tried.
- **A `q_c == 0` arm** (the draft's eq. 4) would quantify what the correction buys, now that the
  penalty has established the `c`/`q_c` redundancy is real.
- **The analytic chart's residual 0.6%** — future work by decision; arXiv:2503.03992.
- **Least-squares domain extension**, from Thomas's unreleased IFT-IK paper, is deferred but belongs
  to *this* project. (Trust-region *solving* belongs to a different project and is out of scope.)
- `stage_H` in `cluster/gen_manifest.py` is a generic cross-test harness kept for whatever knob next
  needs one.
- One latent bug worth remembering: `scripts/iiwa/iiwa_benchmark.py`'s default tag omitted
  `args.solver` where the Panda's has always included it, so an iiwa SNOPT run and an iiwa IPOPT run
  with otherwise identical flags resolved to the same `summary.json` and overwrote each other — the
  same trap `--shard` and `--checkpoint` were each fixed for. Fixed.
