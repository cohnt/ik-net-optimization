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
| **analytic** (`PandaIKProgramAnalytic`, `PandaMugProgramAnalytic` — **Panda only**) | end-effector pose `xyz_rpy` (6) + redundancy parameter `psi` (1) | closed-form S-R-S IK (`src/*_analytic_ik.py`) |

**There is no iiwa analytic *program*.** `src/iiwa_analytic_ik.py` holds the closed-form map and `src/iiwa_program.py` imports it, but no `Iiwa14IKProgramAnalytic` exists and
`scripts/iiwa/iiwa_benchmark.py` registers only `learned` and `numerical`. The comparison is
therefore three-way on the Panda and two-way on the iiwa. Writing that arm is future work or
possibly not done at all (Thomas, 2026-09-19); the closed-form map that would have served it was
deleted rather than maintained unreached, so only the joint limits remain in that file.

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
catch**, hence the mandatory sidecar. The resolved architecture is attached as `solver.arch`. No sidecar → legacy architecture with a `RuntimeWarning`; all on-disk checkpoints
are backfilled (`scripts/training/backfill_arch_sidecars.py`).

**Hold `dim_latent_space` at each robot's baseline** — iiwa14 8, Panda 7. The latent width *is* the
decision-variable count (21 iiwa, 20 Panda), so varying it changes the optimization problem rather
than the chart; holding it also means a checkpoint loaded against the wrong robot fails the shape
check instead of loading silently.

### Gradients through the flow

`VarsToQ` is dual-path: under `float` a plain forward pass; under `AutoDiffXd` it calls
`self.jacobian_gen` (one reverse pass yields both `dq/dvars` and `q`) and chain-rules
`jacobian @ vars_gradients` into fresh `AutoDiffXd`. Both go through `MakeFlowInference(nn_model,
...)`, a free function closing over the network and nothing else, letting `FlowJacobianGen`
memoise `torch.compile(jacrev(...))` per process rather than per program. Analytic formulations
instead evaluate `pydrake.math` trig on templated types, so Drake's own autodiff propagates.

Numerical facts, several encoded in `ProgramOptions` defaults:

- **float64** (`use_float64=True`). Not about differencing — gradients are analytic. A float32
  network produces *values* with a ~1e-7 noise floor, corrupting everything computed as a difference
  over a small step: line-search actual-vs-predicted, convergence tests, SNOPT's derivative
  verification. `snopt_function_precision` declares that floor when running float32.
- **`ik_constraint_tol` forms no constraint bound.** Pose rows are a hard equality (`lb = ub = 0`);
  what survives of the option is the benchmark's gate (tolerance ladder).
- The IK pose constraint is six rows: per-axis position error, then the **rpy residual**
  `rpy(FK(q)) - rpy(target)` wrapped to (-pi, pi] (`orientation_error_rpy`).
  `orientation_error_form` picks the bounds — `rpy` (default) pins it to zero, `rpy_boxed` allows
  `±ori_tol` per row. **Three signed rows are deliberate**: a scalar angle
  `2*arccos(|q.q_target|)` puts a branch point at zero error (infinite derivative), and its `eps`
  clamp returned an `AutoDiffXd` with an *empty* derivative vector. Retired forms: `0be5342`.
- Rows with an identically-zero gradient (e.g. the homogeneous row of a transformed point) break
  LICQ — the mug constraints drop it.
- `c` is boxed near the target (`c_position_slack=0.25`) to keep the flow in its trained workspace.
  A heuristic, not a correctness requirement: the IK constraint is on `FK(q)`, so an
  out-of-distribution `c` cannot produce a false solution. The default sits just under the exposure
  cliff at 0.5, by luck rather than design.
- **No seeding search, deliberately.** A previous revision drew 256 `(c, z)` candidates, scored them
  against the problem's own constraints and started from the best — solving part of the problem
  outside the solver, which only the learned arm can afford. Machinery removed, not disabled.
  `SetStartFromQ(q_init)` is the only way to set a guess, and every formulation gets the same one.

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
cost. It must be applied to the free function: `torch.compile` guards on everything the callable
closes over, so compiling a *bound method* re-triggers dynamo per program. It is off by default, on
with `--compile`, because more iterations inside a fixed cap **moves the learned arm's success
rate** and only that arm benefits, so **every run being compared must set it the same way**.

### Profiling

No profiler in the tree; recover one with `git show ab3ea15:scripts/profiling/profile_flow.py`. At
batch size 1 the flow evaluation is **entirely CPU-bound**: the GPU is never behind the CPU, and
float64 and float32 cost the same wall time despite a 3.4x difference in GPU kernel time. Roughly
70% of a `jacrev` is CPU-side dispatch (PyTorch eager + FrEIA Python; `cudaLaunchKernel` only ~17%),
so runtime is bounded by how fast the CPU can describe 2853 operations and **even zero-overhead
execution leaves only a ~3x ceiling**. **Reducing that dispatch cost is out of scope** (Thomas: infra
fixes for the CPU bottleneck are "future work/possibly not in scope at all") — report the number,
do not make it a project.

### The conditioning frame (read this before touching the learned formulation)

The flow is conditioned on the pose of **the frame it was trained on**, and in both grasp scenes
that is *not* the frame the code used to look up by name. `panda_finray.sdf` contains its own
`panda_hand` welded to `panda_link7` at `[0, 0, 0.134]`, rpy `[90, 0, 45]`, whereas jrl's Panda —
the model IKFlow was trained against — puts `panda_hand` at `[0, 0, 0.107]`, rpy `[0, 0, -45]`.
`GetBodyByName("panda_hand")` returns the finray one, **27 mm and 120 degrees** away. The iiwa's
`iiwa_link_7` is 45 mm short of the flow's frame.

The symptom is unmistakable: running the flow *forwards* on a random configuration (`rev=False`,
which inverts it exactly) returns the latent that would have produced it:

| robot | at the scene frame | at the calibrated frame | typical `\|z\|` under the prior |
| --- | --- | --- | --- |
| Panda | 67.6 | 2.23 | sqrt(7) = 2.65 |
| iiwa14 | 12.1 | 2.45 | sqrt(8) = 2.83 |

A latent of 67 is the network reporting that the configuration is astronomically unlikely for that
conditioning pose. `IKFlowProgram.CalibrateFlowFrame` measures the offset against
`ik_solver.robot.forward_kinematics` at several configurations, checks it is constant (both frames
are welded to the same link, so it must be) and caches it as `self.X_ee_flow`. Use
`FlowPoseInWorld()` wherever a conditioning pose is formed;
`ProgramOptions.calibrate_flow_frame=False` restores the old behaviour for ablations.

**It was called only by the grasp subclasses until 2026-09-16**, invisibly, because the Panda pose
scene used `panda_jrl.urdf` (whose `panda_hand` really is the trained frame) and the iiwa's 45 mm
offset is a pure translation. Welding the finray into the pose scene destroyed that coincidence and
the Panda pose task collapsed to **10/60 with median `max_violation` 0.4**, against 58/60 and
1.8e-08 for the grasp task in the *same* scene with the *same* chart. Both base pose programs now
calibrate; it is a no-op where the frames agree and draws from its own fixed-seed generator so it
cannot shift the grid. **Every pose column measured before this is superseded.** The lesson:
**a calibration that is skipped is indistinguishable from one that is correct, until the geometry it
silently relied on changes** — and a miscalibration and an architectural weakness produce the same
symptom, each masking the size of the other.

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
computed against the wrong seed matrix), behind `share_flow_evaluations`, which **defaults on**:
the memoised path is bit-identical, so run without it only to reproduce a pre-overhaul
measurement.

### Scenes and utilities (`src/utils.py`, `models/`)

`BuildEnv(meshcat, directives_file, extra_directives=None)` builds the diagram from a Drake
model-directives YAML, registering `package.xml` so `package://combining_kinematics/...` resolves;
`extra_directives` is appended to the loaded ones **in memory**, so a caller can add models without
writing to the tracked YAML. `GenerateDiagramWithMug(q, program, yaml_file, meshcat)` uses exactly
that: an `add_model`/`add_weld` pair for a mug at the gripper pose of `q` (the weld pose passed as a
`pydrake.common.schema.Transform`, not formatted into text). The YAML on disk is never modified, so
a crash cannot leave a stray mug in a tracked scene. `BuildEnv(meshcat=None)` skips visualization outright, which is *not* the same as passing `None`
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
(p = 0.016 against) became 199-vs-120 (p = 2.5e-08 in favour).

**Never set an acceptance gate equal to a bound the solver is optimising against**, and before
proposing to tighten one, check the distribution of the gated quantity: if solutions are pinned to
the bound, tightening measures noise. The collision gate carries the binding's own slack for the
same reason. A second verdict `feasible_relaxed` is recorded at `task_tol` so the question stays
re-analysable; the relaxation is worth +10 to +18 cells of 480 on the grasp task and **exactly zero
on the pose task**, moving no ordering.

### A region an initial guess may violate must be a general constraint, never a variable bound

IPOPT's `bound_push` projects the initial guess into every *bounding box* before evaluating
anything, so a box silently reshapes the start protocol. Two instances:

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

Two structural notes. The box lives in **one** method, `LatentBoxConstraint()`: the first repair
fixed `generic_program.py` while the mug subclasses overrode `BoundingBoxConstraint` with their own
copies, so the pose arms were fixed and the grasp arms silently were not. And nothing may project a
guess without recording that it did (`clip_distance`). Variable bounds remain
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
is acceptable"*). The draft says `q_c ~ 0` without specifying how; this imposes it. The weight is
therefore a **stated** part of the formulation and must appear wherever the learned arm is
described, and every table must still show what the penalty buys and costs.

**Options naming learned-only decision variables must be guarded.** `add_costs` applied
`correction_cost_weight` unconditionally, but `correction` exists only on the learned arm and all
three formulations share one `ProgramOptions`. `--set correction_cost_weight=10` therefore raised
`AttributeError` inside every numerical/analytic program's construction and each of those columns
scored **0 of 480 in about 10 ms per cell**. The failure mode: a whole column
of zeroes with `median_max_violation = nan`, three orders of magnitude below the cap, with
`fail_reason = "error"` rather than a named task gate. **Any arm reporting a per-cell wall time
three orders below the cap is not solving badly, it is not solving at all.** `_abort_on_dead_arm`
in `src/benchmark.py` now aborts when an arm fails identically, in under a second, on its first
three cells; the pattern cost two whole columns of cluster campaigns.

### Every result is told in success, iterations, cost and wall clock

Thomas's standing reporting rule. Iterations are hardware-independent and describe the
*formulation*; seconds describe this implementation on this machine and are **never compared across
machines**; cost says what the solution is worth. Reporting only seconds makes the cap story look
arbitrary; only iterations hides that the learned arm's iteration is ten to thirty times more
expensive; only success hides that on the grasp task its solutions cost roughly twice the
baseline's.

Two musts: **cost is compared only on cells *both* arms solved** (a median over each arm's own successes compares different cell sets, and the easy cells
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
measuring throughput, not formulation; raise it and re-measure. Established by iiwa `n4` contained
grasp: 391 v 442 (p = 1.4e-06, a clear loss) at 45 s with 88 timeouts; at 180 s with 0 timeouts it is
447 v 442, a tie — the arms were tied all along. **The cap is a budget for the arm that evaluates a
network, not a shared budget**: across 5/10/20/45/90/180 s every baseline is flat, with one
exception — on iiwa grasp paired the joint-space arm is itself cap-bound below 20 s, with cells
running 1300-1430 iterations against that arm's median of 70. That arm is not uniformly cheap; it
has a tail.

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
  grasp of this mug. `native` gives each formulation the
  initialisation it would have outside a comparison. The joint-space arm's two protocols coincide
  (its native start *is* a random configuration), so any difference between the tables is
  attributable to the others. Neither protocol searches, and the paired start is *measured* rather
  than assumed: every cell records `clip_distance` and `start_q_error`.
- **Success verified from the returned point**, not from `result.is_success()`: every binding is
  re-evaluated at the solution and the task re-measured from `q`, with a named `fail_reason`. Every
  learned failure in the archived runs was a wall-clock timeout, and a timeout that landed on a
  valid grasp is a success. Two gates to get right: an interior-point method parks
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
then we don't get an intermediate solution?"* `Solve()` therefore keeps `program.last_iterate`; any
abnormal exit is verified from that point and recorded as `recovered_feasible`/`recovered_cost`
alongside the failure reason; and the *process* is bounded from outside (OS-level `timeout`), never
the solve from inside a hot-path callback. `SolveTimeout`, `CheckDeadline` and `hard_time_factor`
were **deleted** and must not come back.

Beyond the verdict a record carries `max_violation` and `detail["violations_all"]`;
`collision_value`, `min_distance`, `min_distance_pair`; `start_q_error`, `clip_distance`, `z_norm`;
`median_correction_inf` and `correction_binding` (how much of the ±0.1 box solutions use — the
check that the learned arm is not quietly becoming a reparameterised joint-space arm); and `q`,
plus `q_lift` and `q_flow` separately under `lift_q`. `median_max_violation` separates the arms by
six orders of magnitude and should be read next to any success count.

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
bounds IPOPT projects it at iterate 0 and `clip_distance` records that, so the two numbers together
describe how much survives the solver's own bound projection.

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
rather report the performance for both solvers."* It is shared across arms within a run, varied
across runs.

All three take `kGenericConstraint`/`kGenericCost`/`kCallback`, and all three call
`EvalVisualizationCallbacks` **inside their objective evaluation**, so `last_iterate` recovery works
unchanged under each.

### Each solver converges at its own defaults

The SNOPT branch used to read IPOPT's `acceptable_tol`/`acceptable_constr_viol_tol` as its `Major
optimality`/`Major feasibility tolerance`. That is rung 2 of the tolerance ladder done wrong:
IPOPT's `acceptable_*` family is its **relaxed early-stop** criterion, not what it converges to (its
real `tol` is 1e-8). **Do not transplant one solver's option values onto another.** Unset
`snopt_*`/`nlopt_*` fields are not passed, so each solver sits at its own defaults and the shared,
deliberately looser task gate decides success.

**But "each at its own defaults" is a CHOICE, and not a symmetric one.** `tol`, `constr_viol_tol`,
`dual_inf_tol` and `compl_inf_tol` were never `ProgramOptions` fields at all, so every archived run
took IPOPT's own defaults — `constr_viol_tol` **1e-4** against SNOPT's Major feasibility **1e-6**,
`dual_inf_tol` **1** against Major optimality **2e-6** — the interior-point column was allowed
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

**The program counts evaluations itself** (`IKFlowProgram.ResetEvalCounts`, counted in
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
requiring the value in the parameter echo. It caught three:

- **`Hessian updates` is inert** at these sizes: SNOPT picks full-memory mode below 75 variables and
  the programs have 20-21, so the echo keeps reporting 99999999 however it is set. `Hessian
  frequency` is the one that bites.
- **`Nonderivative linesearch` is a VALUELESS keyword** — passing 0 turns it ON exactly as 1 does,
  so it is a bool emitted only when True. The echo abbreviates it `Nonderiv.  linesearch`, so a
  probe grepping its full name wrongly calls it inert.
- **`linear_solver=mumps` does not exist.** Drake's IPOPT is built against SPRAL and offers only
  `spral` and `custom`, so there is no linear-solver axis on this problem.

**Read a solver's defaults out of the solver, not out of memory.** IPOPT's
`print_options_documentation` dump contradicted the obvious assumption:
`acceptable_dual_inf_tol` defaults to **1e+10** and the two acceptable infeasibility tolerances to
**1e-2**. A sweep arm labelled "IPOPT's defaults" that was not cost a resubmission.

**The Drake on the laptop is not necessarily the Drake on the cluster** — the reason for the
option-surface checks. The cluster once ran the 1.56.0 tarball, whose `NloptSolver` exposes exactly
six options (`algorithm`, `constraint_tol`, `xtol_rel`,
`xtol_abs`, `max_eval`, `max_time`) against a source build's eleven and this pin's sixteen. Drake
validates NLopt names strictly and **raises** on an unknown one — from inside `Solve`, so it lands in
`run_grid`'s per-cell `except` and becomes a **full column of instant failures** rather than an error.
`ProgramOptions.__post_init__` therefore refuses an unavailable option before the first cell, and
`tests/test_solver_plumbing.py` bounds the emitted keys by the *running* Drake's surface; keep both.

**`max_eval` is not "unset" by default** — Drake defaults it to 1000, a cap that binds here, so it
must be set explicitly or the NLopt column silently measures an evaluation budget, not the wall
clock.

### The result: IPOPT > SNOPT >>> NLopt, and the whole axis is CLOSED

All three method classes have been swept to exhaustion. **Do not re-sweep any of them.** The
per-stage tables are deleted; each stage's reader regenerates them from the persisted runs
(`scripts/report_snopttune.py`, `report_snoptcombo.py`, `report_nlopttune.py`, `report_step.py`,
each of which implements its own pre-registered rule inline so it cannot drift). The success
numbers of record are stage STATUSQUO's, under "Results" below.

**This is a CONFIRMATION, not a finding.** Thomas: *"SNOPT performing worse than IPOPT is not
surprising. In my experience, IPOPT is more robust to ill-posed problems, and our neural network
gradients are definitely ill-posed. I expect to see IPOPT > SNOPT >>> NLOPT."* Write it up as the
size and mechanism of a predicted gap. On stage SOLVER2's 480-cell grid IPOPT won all 24 rows, 23
significantly, p from 3.6e-08 to 4.8e-44.

**It is a property of the PROBLEM, not of the learned formulation.** The joint-space arm degrades
under SNOPT too, on every row and by comparable margins, and that arm never evaluates the network.
The same holds for every solver setting below: each moves both arms.

**The mechanism, and it is not budget.** Pooled over all 3,952 SNOPT failures: `nonlinear
infeasibilities minimized` (INFO 13) **50.7%**, `current point cannot be improved` (INFO 41)
**36.2%**, iteration limit 9.5%, time limit **3.6%** — about 87% convergence failures against 11%
budget, and SNOPT times out *less* often than IPOPT. The two fail in opposite ways: IPOPT's
learned-arm failures are mostly wall-clock, still descending when the clock runs out, where SNOPT's
are INFO 41, giving up at a feasible-but-wrong point. **INFO 41 is the documented signature of
inaccurate or badly scaled derivatives**, which is Thomas's explanation with a mechanism attached.
It is not the gain-ceiling runaway: `n4`'s solved cells return violations ~1e-08, so what SNOPT
struggles with is ordinary ill-conditioning of an *exact* Jacobian. SNOPT also takes 2-4x the
iterations on cells it does solve, and its solutions are worse on the joint-space arm by a wide
margin; on the learned arm cost is closer and occasionally favours SNOPT — where it converges it
sometimes finds a better optimum, it just converges far less often.

**The gap is much larger under `paired`**, so SNOPT copes far worse with an infeasible start. A
60-cell triage read this backwards in both directions before 480 cells settled it: **do not draw a
protocol conclusion from 60 cells.**

**Wall-clock columns are not capped equally across solvers.** NLopt's `max_time` binds much more
tightly than SNOPT's `Time limit` (20.0-20.1 s against 24-28 s in a local probe), because SNOPT only
checks at major-iteration boundaries.

**One cap-rule lesson, the unusual case.** At its defaults NLopt timed out on 60 of 60 grasp cells,
which looks exactly like the throughput case the cap rule is written for — but at 180 s ten of twelve
rows are identical to 45 s. The default configuration never terminates its *inner* solve, so more
wall clock buys no outer progress. The cap rule still holds; it just needed the cap arm to be **run
rather than assumed**.

### What solver tuning is worth: IPOPT's early stop, and one SNOPT setting

**Per-solver tuning is permitted and per-problem tuning is not** (Thomas, 2026-09-17: *"I'm okay with
playing with solver settings on a per-solver basis, as long as it's not per-problem"*), so a setting
qualifies only if it wins **uniformly across rows**, never by picking the robot it helps. Every rule
below was pre-registered before any result was read, and every one is **counted by row, never
pooled**.

**IPOPT: the convergence tolerances are INERT and the acceptable-point machinery is everything**
(stage SWEEP, 23 settings, 240 cells). Holding IPOPT to SNOPT's convergence numbers is within noise —
IPOPT already converges far tighter than either default, so the 1e-4-against-1e-6 asymmetry documented
above was real on paper and worth **zero cells**. At each solver's own defaults, which is what rung 2
of the tolerance ladder asks for, **IPOPT 200, SNOPT 141, p = 4.1e-09**, and on the cells both solve
IPOPT's solutions also cost less. **The fairness question is answered: the ordering survives it.**

**The early stop is worth 39 cells and a 9x speedup, and that has to be stated** — and it is **NOT
returning sloppy points.** Fielded successes sit at 1.29e-08, five orders inside the gate and the same
quality as SNOPT's. Turning it off drives the violation to 2.22e-15 and success *down* to 62 of 240:
IPOPT without it keeps polishing a solution it already has until the clock kills it.
`acceptable_iter = 1` does not let IPOPT scrape past the gate, it lets IPOPT **recognise it is already
done and stop**, a real capability under a wall-clock cap. SNOPT has no counterpart and would not
benefit. Nothing was adopted: the alternatives move success by at most +2 cells of 240, and changing
the default would break comparability with every archived run.

**SNOPT: one setting of thirteen survives 480 cells x 12 rows, and no combination beats it.** The
motivation was fairness, not rescue — IPOPT's column ran a tuned configuration while SNOPT's ran bare
Drake defaults, a property of the harness rather than of SQP. Stage SNOPTTUNE's rule (>= 9 of 12 rows
better on the learned arm, none significantly worse, >= 1 significantly better) passes **`Major step
limit = 0.5`** alone. Stage SNOPTCOMBO then crossed it with the three other positive factors under two
bars — that rule against Drake's defaults, plus >= 8 of 12 better against the survivor itself — and
**nothing clears both**, so the combination question is closed. Four combinations are *valid* SNOPT
configurations in that they beat Drake's defaults; none improves on the survivor.

**The pooled numbers disagree with the row rule, and the rule is what stands.** Pooled over 5,760
learned cells a four-factor stack beats the survivor by +221 cells and would have been fielded on that
basis. **It is a TASK TRADE**: it gains 35-58 cells on each iiwa grasp row, all significant, and loses
on every pose row — exactly what pooling hides, and why the rule reads rows. It is also bought with
work rather than insight, running ~50% more iterations and roughly twice the timeouts.

**The mechanism is diffuse, and the SNOPT step-limit hypothesis is REFUTED at both scales.** A smaller
`Major step limit` was predicted to convert INFO 41 into convergence. It does not: INFO 41 supplies
less than a third of the gain and falls by under 2% of itself, and the 60-cell story that INFO 13
falls did **not** replicate — INFO 13 is flat. What the setting actually does is stop SNOPT exhausting
its iteration and time budgets.

**NLopt: naming the inner optimizer is worth nothing; truncating it is worth a great deal.** Stage
NLOPTTUNE fielded nine settings plus a 180 s budget arm across 12 rows. Its pre-registered gate
(>= +8 cells of 60 on >= 6 of 12 rows, or any row from <= 2/60 to >= 20/60) is cleared by
**nothing**. `LD_MMA` named alone matches Drake's default column cell for cell with identical median
violations — so PR 25002's *algorithm selector*, the thing that unblocked the stage, is inert here.
What moves cells is the inner **budget and tolerance** options that shipped with it: on Panda
contained grasp the default burns 3,532 network Jacobians per cell and lands 3.7e-04 from feasible,
where loose inner tolerances use **66** and reach 6.8e-07. That is the classic augmented-Lagrangian
failure — an inner subproblem solved to convergence before the multipliers are ever updated — with the
classic fix.

**But it is a trade, not an improvement, and the sign flips by row.** On cells both columns solve,
loose inner tolerances cost more nearly everywhere, and on **Panda pose native**, a row the default
already solves 38 of 60, truncation *loses* cells monotonically in how aggressive it is. The inner
budget buys feasibility where feasibility was the binding problem and costs optimality everywhere
else.

**Three things no NLopt setting changes.** The **iiwa grasp rows are 0-3 of 60** under every setting
and at 180 s — nothing Drake exposes makes the augmented Lagrangian solve that task at all. The
**joint-space arm moves with these settings too**. And the **ordering is untouched**; what changed is
that the third column is now measured across Drake's whole option surface instead of left unswept by
decision.

**The `LD_SLSQP` caveat, unresolved by design.** It puts an SQP method inside the augmented
Lagrangian. The outer method is still AL and the inner subproblem bound-constrained only (see the
mechanism below), so it reads as an AL column — but the three-method-classes rule is Thomas's
(2026-09-18: *"For now, it's okay to have LD_SLSQP as the inner optimizer. We can always decide
later."*), so the caveat travels with the number rather than resolving it. Mechanistically it behaves
like SNOPT: **0 timeouts on every row** and 2.3-4.4 s mean wall clock against the default's 27-45 s.

### What is fielded, and what each adoption is and is not

Exactly **two** solver settings are fielded anywhere in this project, both adopted 2026-09-19.
Everything else — all 16 `ipopt_*` fields, 15 of 19 `snopt_*`, 10 of 15 `nlopt_*`, and the four
remaining step-rejection knobs — stays plumbed and `None`.

**SNOPT: `snopt_major_step_limit` defaults to 0.5** (Drake/SNOPT's own default is 2.0). **It was
adopted for FAIRNESS, not for the comparison, and that distinction must survive into the write-up.**
On the learned-vs-joint-space question it changes nothing: five learned wins, four joint-space wins
and three ties before and after, with **zero verdict flips** and the same rows in each bucket. It
gives the learned arm +161 cells of 5,760 and joint space +164 — the same size — and the per-row
margins (L - JS) move between -32 and +15 and sum to **-3**. What justifies it is that IPOPT's column
runs a tuned configuration while SNOPT's ran bare Drake defaults. **Do not present it as helping the
learned formulation.** It is also a property of SNOPT rather than of the chart: the joint-space arm
improves on all twelve rows.

Two consequences. **The SNOPT numbers of record are stage SNOPTCOMBO's `mstep0p5` column**, not
`sc_SOLVER2_*_snopt_*`, which was measured at Drake's defaults. And **"set nothing" no longer means
Drake's SNOPT defaults** — a stage whose column means that must say `--set
snopt_major_step_limit=None`, which emits the option not at all. `tests/test_solver_plumbing.py`
pins both directions.

**NLopt: `LD_AUGLAG` + `LD_MMA` inner + inner `xtol_rel = ftol_rel = 1e-3`.** Fielded on Thomas's
criterion — *"Feasibility is the name of the game, objective cost is secondary."* Against Drake's NLopt
defaults on the learned arm it is better on 5 of 12 rows, worse on 1, unchanged on 6 (all six rows
where nothing solves), and it collapses both the residual and the work per cell — Panda grasp contained
native 12 -> 42 of 60 (p = 1.9e-09) at 3532 -> 66 network Jacobians is the clearest instance.

**Three things must be reported with it.** It **failed** stage NLOPTTUNE's pre-registered gate, which
asked whether to spend 480-cell compute and not whether the setting is the best configuration —
state the gate failure alongside the setting so it does not read as a configuration chosen where it
helps. The **Panda pose native row is a genuine regression on the adoption's own criterion**: five
cells one-directionally and a residual four times worse (9.3e-07 -> 3.6e-06), and that is the honest
cost. And it **flips one learned-vs-joint-space verdict** (Panda pose paired, tie -> learned win),
unlike the SNOPT adoption which flipped none.

**A caveat we are not re-sweeping.** The sibling `ik-tune` project, sweeping the same inner tolerance
on its own problems, puts the optimum near **1e-4** and finds loosening past it costs — and reached
that only after discovering its control had been running at an *implicit* 1e-4. So our adopted 1e-3
is in a sensible region but is **not** an optimum this project established; it is the value NLOPTTUNE
happened to field. Revisiting it is Thomas's call.

**`LD_AUGLAG` is kept, and the reason is checkable — but the mechanism is in NLopt, not in Drake.**
Under `LD_AUGLAG` the inner optimizer solves a **bound-constrained** subproblem and every constraint
sits in the augmented-Lagrangian penalty, whatever the inner algorithm's own method class. That
retires the `LD_SLSQP` taxonomy worry, and it is why `LD_AUGLAG_EQ` is **not** used: `_EQ` absorbs
only equalities and enforces inequalities on the subproblem directly, and this program carries both
kinds. What decides it is the algorithm name alone, verified in the nlopt bundled in Drake:
`src/api/optimize.c:934` passes `sub_has_fc` computed purely from the enum, and
`src/algs/auglag/auglag.c:98-101` branches on it (`if (sub_has_fc) d.m = 0; else m = 0;`), i.e.
inequalities go either into the penalty or onto the subproblem, never both.

**An earlier version of this argued from Drake's side — that Drake never adds a constraint to the
inner `local_opt` — and that reasoning must not come back.** It is true and causes nothing:
`auglag.c:110-119` overrides the subproblem's objective, bounds and stopval, removes its constraints
and repopulates from the *outer* problem's list, so whatever Drake put on `local_opt` is discarded.
The wrong chain also predicts the wrong thing, since under `LD_AUGLAG_EQ` the subproblem is *not*
bound-constrained even though Drake adds nothing to `local_opt`.

**A live hypothesis this hands us, not acted on.** `ik-tune` measured both parents across six robot
experiments and found the plain-vs-`_EQ` difference large and one-directional: its two experiments
whose *inequality* structure carries the problem collapse under plain `LD_AUGLAG` at every encoding
and inner tolerance tried (14.4-19.8% against 54-63% under `_EQ`). Our iiwa grasp rows are 0-3 of 60
under every NLopt setting and at 180 s, and the grasp task is exactly where the collision inequality
binds — so penalised-rather-than-enforced inequalities is now a mechanism candidate for a row this
project had recorded only as inexplicable. Their problems are not ours, so this is a thing to watch
rather than a prediction. **Switching to `LD_AUGLAG_EQ` is a method-class decision and therefore
Thomas's**, and the reason recorded above for preferring one honest augmented Lagrangian is unchanged
by any of it.

**Requires a Drake carrying PR 25002.** `local_optimizer_ftol_rel` is absent from 1.56.0 and from a
pre-PR source build, so `CheckNloptOptions`'s availability refusal is scoped to
`which_solver == "nlopt"`; unconditional, it would refuse `ProgramOptions()` itself and kill every
IPOPT and SNOPT cell over options they never emit.

### The grasp-containment lever is closed on both axes

Grasp containment was the standing "revisit whenever another knob moves" lever, because on the
contained task joint space needs 970 median iterations against 48 on the free one. Under SNOPT it
never flips a verdict *toward* the learned arm and flips one away from it (Panda contained paired
goes from a decisive learned win under IPOPT to a tie under SNOPT). Step rejection was the remaining
candidate and is refuted too, so the lever is closed on both axes and the containment verdicts stand
as measured under IPOPT.

### ONE DRAKE, AND IT IS THE PIN

The nightly is installed at `$ROOT/drake` by `cluster/setup_supercloud.sh` and there is no second
install, no `DRAKE=nightly` per-item sentinel and no `stage_DRAKEBUMP` — all three were deleted on
2026-09-19. Every arm of every campaign runs on the current pin. Thomas, ruling on it for the second
time:

> stop running IPOPT and SNOPT on the installed 1.56.0. I've said this already. The difference
> between 1.56.0 and the current nightly is negligible. I think we've even measured this. Stop making
> this mistake, it's getting tiresome. Run everything on the current nightly. Even if you think I'm
> wrong, it doesn't matter, because we're not going to pin IPOPT and SNOPT back to an earlier version
> as Drake moves ahead — that would be a regression that you report so I can fix in Drake upstream,
> and/or further tuning to fix it.

So **archive pairing is not a reason to keep an old install**: a cross-version caveat is stated once
and a version-induced regression is reported and fixed upstream, never pinned around. **Do not
recreate `stage_DRAKEBUMP`.** The empirical backing was free — stage NLOPTTUNE's `default` column ran
on this nightly against a 1.56.0 archive and reproduced it exactly on eleven of twelve rows.

**Why a nightly at all.** The AL column's inner local optimizer became selectable when **PR 25002**
(merged 2026-09-17, `5a73436c`) took `NloptSolver` from **six** option names to **sixteen**. Neither
set is in a release — 1.57.0 was cut before both and still declares six — so
`drake-0.0.20260918-noble.tar.gz`, whose tip is exactly that merge commit, is the pin. Two
properties of it still bite: cluster Drake versions are per-project, so the install lives inside this
project's tree; and it needs **its own** `drake_models` cache warm, because the cache key includes the
models commit that Drake version pins. Nightlies publish no `.sha256`, so setup verifies against a
hash recorded in the repo — the bytes the local tests ran against. Nightly artifacts **expire after
45 days** (~2026-11-02), so **move the pin to 1.58.0 the moment it carries PR 25002** and restore the
published-checksum path. One caveat before comparing across nightlies: `ik-tune` sees
inner-local-optimizer behaviour differ between the 09-16 and 09-18 nightlies at an identical recorded
configuration, while a control naming no inner optimizer is identical on both.

**The Luksan trap, a new instance of an old one.** Drake's NLopt is built without the LGPL Luksan
sources, so `LD_LBFGS`, the `LD_VAR*` family and every `LD_TNEWTON*` variant are listed by
`ParseNloptAlgorithm` as valid choices and then refused *inside the solve* with `attempting to use
NLOPT_LD_LBFGS, but Luksan code disabled`, returning `kInvalidInput` and status 0. The symptom is
quiet and misleading: a **0.5 s cell with `q=None`, `max_violation=None` and `fail_reason` unset**,
which reads like a harness bug. Exactly eight algorithms are refused; **`LD_MMA`, `LD_CCSAQ`,
`LD_SLSQP`, `LN_COBYLA` and `LN_BOBYQA` all work**, so the usable *gradient-based* inner optimizers
are the first three and nothing else. `NLOPT_LUKSAN_DISABLED` refuses them at configuration time, for
both the outer and inner algorithm.

**This refutes a claim that once stood here**: that NLopt supplies `LD_LBFGS` when the inner
optimizer is unset. It cannot — naming `LD_LBFGS` *fails* while leaving it unset *solves*. So there
is **no "name what NLopt already picks" control available**: `default` against any named inner
algorithm unavoidably mixes "Drake called `set_local_optimizer` at all" with "which algorithm".

Two more Drake behaviours worth not rediscovering. Every `local_optimizer_*` option is read
unconditionally but **applied only inside `if (!parsed_options.local_optimizer_algorithm.empty())`**
(`nlopt_solver.cc:546-564`), so an inner budget or tolerance without a named inner algorithm is
accepted and inert. And an **unknown** NLopt name does not crash a run: Drake accepts it at
`SetOption` and raises from inside `Solve`, which lands in `run_grid`'s per-cell `except Exception`
and is recorded as `fail_reason="error"`, i.e. a **full column of instant failures** rather than an
error. Hence the check lives in `ProgramOptions.__post_init__`, before the first cell.

**Do not extend Drake to get better instrumentation.** Thomas: *"NLOPT might not have the robust
logging we need btw, work with what you have, don't write new logging stuff in Drake or anything."*

## Results: the current campaign

**Stage STATUSQUO is the campaign of record**, measured 2026-09-19/20: hardened scene,
shelf-contained targets at the **fingertips for both tasks**, **180 s**, 480 cells = 60 targets x 8
guesses, seed 1 (out of sample), `--compile`, adopted rungs (Panda `n6`, iiwa `n4`), arms
`learned,numerical`, both start protocols, all three solvers at their adopted configurations, Drake
nightly `0.0.20260918`.

**THE RECORD IS 24 LOGICAL RUNS: two robots x TWO experiments x two protocols x three solvers,
11,520 solves.** The experiments are grasp and pose, both shelf-contained at the fingertips, and
there are no others. The run also produced 12 further logical runs on `--target-placement free`,
which **is a vestigial setting of the grasp experiment and not a third experiment** (Thomas,
2026-09-21: *"preserving old settings and old experimental setups is contrary to that mission"*).
Gathering that data was not the error -- the cluster was idle -- but it is **not part of the record
and is not reported**: `scripts/report_statusquo.py` no longer prints it and `stage_STATUSQUO`'s
selftest now refuses any non-contained placement. Those summaries remain on disk for anyone who
deliberately wants to look.

Earlier campaigns (final3-5, Stages A-D, the legacy-scene headline tables, the pre-calibration pose
tables, and the 45 s / free-grasp tables this section used to hold) are **superseded and their tables
removed**; git history holds the originals, and every conclusion of theirs that still stands is
restated here on the status-quo numbers. The ladder tables under "The chart" are the deliberate
exception: they remain the 45 s / free-grasp record, because nothing affecting the charts changed.

**Do not quote a 45 s number as current.** Where one appears below it is labelled as the pairing
reference for a cap effect.

Harness self-check, which passes on all 24 rows: joint space is identical between protocols on every
solver x row pair, `median_start_q_error` is 0.0 exactly under `paired`, and every row has 480 cells
on both arms.

### The status quo, ESTABLISHED 2026-09-21

Decided 2026-09-19, measured 2026-09-19/20, **accepted by Thomas on 2026-09-21**. This is the
project's baseline now: two experiments per robot (grasp and pose, both shelf-contained at the
fingertips), 180 s, both start protocols, three solvers at their adopted configurations, adopted
chart rungs. **Any future result is stated against it**, and a change to any of those choices is a
new decision rather than a variation.

Thomas's call, replacing both the free-grasp default and the 45 s campaign cap:

> Why don't we set grasp fingertips in shelf as default, use the 180s timeout, run the other
> experiments/solvers at 180s at the new status quo benchmark, and just have a note that if the
> story radically changes as a result, flag it?

**What changed.** `--target-placement auto` now resolves to `shelf` for **both** tasks (it was
`shelf` for pose and `free` for grasp), at `--placement-point fingertips`. The campaign cap is
**180 s**. `--target-placement free` still reproduces any earlier grasp column, and containment
changes which draws are accepted, so a contained grid has a different `grid_hash` from a free one
and the two cannot be paired. The **cap** does not enter target sampling, so 45 s and 180 s columns
on the same placement DO pair cell for cell.

**Why the two changes are one decision.** Containment is what creates headroom — on free targets the
joint-space arm sits at 94-95% and only ~25 cells of 480 are winnable at all, while contained it
falls to 300-323 and needs 965-970 median iterations against 125-176. But at 45 s the iiwa's
contained-grasp rows are cap-bound (64-74 learned timeouts of 480) and score as joint-space wins;
at 180 s they are ties with **zero** timeouts, and the 360 s column reproduces 180 s exactly, so
180 s is saturated in the tail as well as the median. Adopting containment at 45 s would have
fielded a cap artefact as a result.

**The cost of parity must be reported, and the status-quo IPOPT table below carries it**: the iiwa
contained-grasp tie is bought at a **14x** wall-clock premium (Table 3: 27.29 s against 1.90 s
native, 25.25 against 1.90 paired), and joint space is flat across the whole cap ladder (442-443,
1.7-1.9 s), so the entire effect is the learned arm's per-iteration price.

**WHAT WAS FLAGGED, and the ANSWERS.** Three criteria were named in advance so "did the story change"
would be a printed verdict rather than a judgement call; `scripts/report_statusquo.py` evaluates them
inline. All three are answered and **nothing moved against the learned arm**:

1. **Verdict moves: four, every one predicted.** iiwa contained grasp went joint-space-win -> **tie**
   under both protocols, as expected. Panda contained grasp stayed a decisive learned win with a
   growing margin (paired 437 -> 471), also as predicted. The other six status-quo rows hold their
   45 s verdict. (Two further moves were measured on the retired free-grasp placement, tie ->
   learned win under both protocols; they are not part of the record -- see below.)
2. **The IPOPT-SNOPT gap widened, and the mechanism is measured rather than inferred.** Median
   learned-arm gap 102 cells at 45 s -> **120 at 180 s**, IPOPT ahead on every row at both caps.
   See "The cap effect, per row" below for where it comes from.
3. **NLopt under the adopted configuration: ten of twelve rows solve something, all four pose rows are
   decisive learned wins, and the solver ordering is untouched.** The one story-level surprise is not
   on this list: **the augmented Lagrangian is extraordinarily start-sensitive** (Panda contained
   grasp 327 native -> 0 paired), which nobody predicted and which means that column must be read per
   protocol and never pooled.

**A killed item cannot corrupt a result.** Partial writes go to `summary.json.partial`
(`src/benchmark.py`) and `run_items.sh` publishes to the collection point only on exit 0, so a killed
shard reads as *absent* rather than truncated; `merge_shard_summaries.py` then refuses both on a
missing shard index and, separately, on any unsolved cell of the full grid. What a killed item costs
is its compute plus a stale claim needing `collect_results.sh --reclaim`. **So shard sizing is a
THROUGHPUT question, not a data-integrity one.**

**The caps were also genuinely misconfigured, and that is fixed.** `ITEM_TIMEOUT` was 4 h and
`submit_bench.sh` requested `--time=04:00:00` -- *numerically equal*, which made the bad case normal:
one long item consumed the whole job and Slurm's kill hit all `PROCS` workers at once, leaving
`PROCS` stale claims. `xeon-g6-volta` allows **4-04:00:00 (100 h)**. Now: job wall **48 h**,
`ITEM_TIMEOUT` **8 h** (a backstop for a wedged solve, not a budget -- a 102-minute stall inside one
IPOPT iteration is on record), `timeout -k 60` so a TERM-ignoring solve is actually killed, and
`run_items.sh` **refuses to claim an item the remaining job time cannot cover**, leaving it unclaimed
for the next job instead. That last part is the actual fix: raising the numbers alone still lets a
worker claim a multi-hour item minutes before its job ends. **Keep `WALL` > `ITEM_TIMEOUT`.**

### The status quo measured: stage STATUSQUO

480 cells = 60 targets x 8 guesses, seed 1, **180 s**, `--compile`, adopted rungs (Panda `n6`,
iiwa `n4`), hardened scene, shelf-contained targets at the fingertips, `learned,numerical`, both
protocols, all three solvers at their adopted configurations, Drake nightly `0.0.20260918`. **Two
experiments per robot, eight rows per solver, 24 logical runs, 11,520 solves.**
`scripts/report_statusquo.py` owns these tables and evaluates the three pre-registered flag criteria
inline.

**Acceptance checks all pass.** iiwa `n4` contained grasp under IPOPT reproduces
`sc_CAP_iiwa_n4_mug_180_{native,paired}` **exactly** — learned 447/453, joint space 442/442, zero
timeouts on every arm, same `grid_hash` — which validates the new stage, the raised cluster caps
and the Drake pin move in one row. `median_start_q_error` is 0.0 on every paired row and joint
space is bit-identical between protocols on all twelve solver x row pairs. Every row that had no
timeouts at 45 s reproduces its 45 s cell count **exactly** (sole exception: Panda pose tip paired,
+2), which is a tighter reproducibility statement than the +/-2-cell band.

**Timeouts are essentially gone at 180 s** — at most 2 cells of 480 on any IPOPT or SNOPT row. So
these are formulation results, not cap results, and this is also what closes the 180 s chart-ladder
re-measurement: heavy timeouts were its trigger and there are none.

#### The four tables

Laid out as in `writing/tro-paper/tables/*_alternate_organization.tex` so the two papers read side by
side: rows are experiments, each solver is a block with the learned and joint-space arms **adjacent**,
and there is **one metric per table**. `IP` = interior point (IPOPT), `AL` = augmented Lagrangian
(NLOPT), `SQP` = sequential quadratic programming (SNOPT). **Bold** is the better of each pair (both
bold when tied); (star) marks the best in the row. **Every row prints, zeros included** -- an omitted
row reads as missing data rather than as a measurement of zero.
`scripts/report_statusquo.py` regenerates all four, so do not hand-maintain them.

**Table 1 -- success rate of 480 cells.** Higher is better.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | **0.931** (star) | 0.921 | **0.000** | **0.000** | 0.419 | **0.633** |
| iiwa grasp paired | **0.944** (star) | 0.921 | **0.004** | 0.000 | 0.500 | **0.633** |
| iiwa pose native | **0.979** (star) | 0.623 | **0.621** | 0.065 | **0.925** | 0.562 |
| iiwa pose paired | **0.879** (star) | 0.623 | **0.246** | 0.065 | 0.515 | **0.562** |
| panda grasp native | **0.992** (star) | 0.673 | **0.681** | 0.000 | **0.919** | 0.635 |
| panda grasp paired | **0.981** (star) | 0.673 | **0.000** | **0.000** | 0.627 | **0.635** |
| panda pose native | **0.960** (star) | 0.452 | **0.602** | 0.025 | **0.917** | 0.371 |
| panda pose paired | **0.848** (star) | 0.452 | **0.237** | 0.025 | **0.573** | 0.371 |

**The learned arm wins 15 of the 24 solver x experiment cells, ties 7 and loses 2** — interior point
6/2/0, augmented Lagrangian 5/3/0, SQP 4/2/2 — and interior point is the best entry in every row.
**Both losses are iiwa contained grasp under SQP**; there is no other losing cell anywhere in the
table. Verdicts are by exact McNemar on the cells the table shows, which is also what decides a tie:
a numeric reading of the same table put two rows down as losses that the text called ties.
`scripts/report_statusquo.py` prints this tally beside the table so the two cannot drift apart.

**Table 2 -- optimal cost**, on cells **both** arms solved, learned-only regularizers excluded. Lower
is better; `N/A` means fewer than 10 shared solved cells, so no comparison exists.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | 4.985 | **2.839** (star) | N/A | N/A | 5.409 | **5.393** |
| iiwa grasp paired | 4.998 | **2.826** (star) | N/A | N/A | 5.607 | **4.515** |
| iiwa pose native | **6.242** | 6.757 | 6.511 | **4.988** (star) | **5.975** | 6.858 |
| iiwa pose paired | **6.011** | 6.704 | 7.062 | **5.807** (star) | 6.863 | **6.757** |
| panda grasp native | 7.504 | **5.305** (star) | N/A | N/A | **7.040** | 7.458 |
| panda grasp paired | 6.846 | **5.358** (star) | N/A | N/A | **7.370** | 7.567 |
| panda pose native | **11.136** | 11.861 | 9.346 | **9.088** (star) | **10.366** | 10.774 |
| panda pose paired | **10.654** (star) | 11.762 | N/A | N/A | 10.807 | **10.749** |

**The cost split is by TASK, not by solver**: learned is cheaper on pose and ~1.4-1.8x more expensive
on grasp, under every solver that produces a comparison.

**Table 3 -- mean runtime in seconds over all cells.** Lower is better. **This machine only** --
never compared across machines.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | 27.29 | **1.90** (star) | 180.06 | **180.06** | 17.72 | **4.29** |
| iiwa grasp paired | 25.25 | **1.90** (star) | **179.33** | 180.06 | 20.06 | **4.30** |
| iiwa pose native | 2.19 | **0.09** (star) | **80.01** | 174.26 | 2.61 | **0.25** |
| iiwa pose paired | 4.51 | **0.09** (star) | **140.98** | 174.27 | 11.84 | **0.25** |
| panda grasp native | 11.79 | **5.91** | **57.81** | 180.06 | 5.55 | **3.80** (star) |
| panda grasp paired | 20.93 | **5.91** | 180.07 | **180.06** | 17.59 | **3.80** (star) |
| panda pose native | 2.20 | **0.29** | **81.87** | 177.15 | 3.18 | **0.26** (star) |
| panda pose paired | 7.04 | **0.29** | **143.48** | 177.16 | 10.82 | **0.26** (star) |

Joint space wins every IP and SQP cell, which is the per-iteration price stated as a number.
**Under AL the ordering inverts on five rows** -- the learned arm is genuinely faster there, because
it converges while joint space burns the whole 180 s.

**Table 4 -- median major iterations over solved cells.** Lower is better. **The AL column is `N/A`
BY CONSTRUCTION**: `NloptSolverDetails` carries a single `status`, and NLopt has no major iteration to
count. Its work proxy is `eval_counts["map_jacobian"]` (3,881-12,923 per learned cell against
9,550-13,417 for joint space), which is **not comparable across arms** -- identity map on the
joint-space side -- so it does not belong in this table.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | 381 | **212** | N/A | N/A | 798 | **168** (star) |
| iiwa grasp paired | 348 | **212** | N/A | N/A | 644 | **168** (star) |
| iiwa pose native | 44 | **28** | N/A | N/A | 73 | **22** (star) |
| iiwa pose paired | 102 | **28** | N/A | N/A | 316 | **22** (star) |
| panda grasp native | **169** | 744 | N/A | N/A | 165 | **164** (star) |
| panda grasp paired | **261** | 744 | N/A | N/A | 443 | **164** (star) |
| panda pose native | 36 | **33** | N/A | N/A | 47 | **21** (star) |
| panda pose paired | 100 | **33** | N/A | N/A | 238 | **21** (star) |

**The Panda grasp rows are where hardening shows**: joint space needs 744 median iterations against
the learned arm's 169, so the learned formulation wins on *iterations* there under interior point
despite costing ~10x per iteration. Containment costs the joint-space arm its cheapness.

#### Headroom and the rescue rate: what the success counts hide

Containment is what creates the headroom the comparison needs. Under IPOPT, cells joint space
fails, and how many of them the learned arm solves:

| config | L | JS | L only | JS only | both | neither | of JS's failures, rescued |
| --- | --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp contained native | 447 | 442 | 31 | 26 | 416 | 7 | 31/38 = 82% |
| iiwa grasp contained paired | 453 | 442 | 35 | 24 | 418 | 3 | 35/38 = **92%** |
| iiwa pose tip native | 470 | 299 | 175 | 4 | 295 | 6 | 175/181 = **97%** |
| iiwa pose tip paired | 422 | 299 | 158 | 35 | 264 | 23 | 158/181 = 87% |
| panda grasp contained native | 476 | 323 | 156 | 3 | 320 | 1 | 156/157 = **99%** |
| panda grasp contained paired | 471 | 323 | 155 | 7 | 316 | 2 | 155/157 = **99%** |
| panda pose tip native | 461 | 217 | 248 | 4 | 213 | 15 | 248/263 = 94% |
| panda pose tip paired | 407 | 217 | 219 | 29 | 188 | 44 | 219/263 = 83% |

**The rescue rate is 82-99% on every row of both robots and both protocols**, and containment is what
makes it matter: there are 157-263 joint-space failures available to rescue. The iiwa grasp ties are
not the same cells either -- 31-35 each way -- so the arms are complementary even where the totals
agree.

#### The cap effect, per row: where the IPOPT-SNOPT gap comes from

The verdict is under "WHAT WAS FLAGGED" above; this is the measurement behind it. Learned arm, cells
gained going from the 45 s pairing reference to 180 s on the same grid:

| | IPOPT gains 45 s -> 180 s | SNOPT gains |
| --- | --- | --- |
| every grasp row | +5 to +55 | -1 to +7 |
| every pose row | **exactly +0** | +0 to +2 |

IPOPT's learned-arm failures are wall-clock, so a 4x cap recovers them; only 3.6% of SNOPT's are the
time limit, so it has nothing to recover. **The pose rows being exactly +0 is also the campaign's
tightest reproducibility statement** -- every row with no timeouts at 45 s reproduces its 45 s cell
count exactly, the sole exception being Panda pose tip paired at +2.

#### The status quo under NLopt (augmented Lagrangian), `LD_MMA` inner + loose inner tolerances

Work, wall clock and violation are over **all 480 cells**, not over successes: this column times out
most cells, so a median over successes would describe the handful it got right. `jac/cell` is the
program's own network-Jacobian counter and is **not** comparable across arms — for the learned arm
each is a reverse pass through the flow, for joint space the identity map.

| row | L | JS | p | L jac/cell | JS jac/cell | L s | JS s | L timeouts | JS timeouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp contained native | 0 | 0 | 1 tie | 12923 | 12407 | 180.06 | 180.06 | 480 | 480 |
| iiwa grasp contained paired | 2 | 0 | 0.50 tie | 12709 | 12407 | 179.33 | 180.06 | 478 | 480 |
| iiwa pose tip native | **298** | 31 | **5.1e-68** | 4025 | 9550 | 80.01 | 174.26 | 199 | 463 |
| iiwa pose tip paired | **118** | 31 | **2.0e-17** | 7066 | 9550 | 140.98 | 174.27 | 369 | 463 |
| panda grasp contained native | **327** | 0 | **7.3e-99** | 3881 | 13417 | 57.81 | 180.06 | 153 | **480** |
| panda grasp contained paired | 0 | 0 | 1 tie | 11916 | 13416 | 180.07 | 180.06 | 480 | 480 |
| panda pose tip native | **289** | 12 | **5.8e-82** | 3988 | 10455 | 81.87 | 177.15 | 203 | 469 |
| panda pose tip paired | **114** | 12 | **5.4e-26** | 6864 | 10469 | 143.48 | 177.16 | 371 | 469 |

**ALL FOUR pose rows are decisive learned wins** — iiwa 298 v 31 native (discordant 278 to 11) and
118 v 31 paired (101 to 14); Panda 289 v 12 native (278 to 1) and 114 v 12 paired (107 to 5), every
p between 5.4e-26 and 5.1e-68. **The joint-space arm never exceeds 31 of 480 anywhere in this
column.** Under an augmented Lagrangian the learned formulation solves this problem
and the joint-space formulation essentially does not — 96% of its cells hit the wall clock. Report
it as the result it is, not as a spoiled column: it is **attributable**, because only the solver
differs and the joint-space arm is the *easier* problem (7 variables, no network), so its collapse
is a property of NLopt on this program rather than of a harness that favours us. Cost exists here
but on only 20 and 17 common cells (learned 6.511 / 7.062 against 4.988 / 5.807), so quote it with
that n. The paired protocol is much harder for the augmented Lagrangian, as it is for SNOPT: 298 ->
118 cells and 80 s -> 141 s mean, while the joint-space arm is unchanged at 31 by construction.

**Panda contained grasp is the cleanest statement the project contains: learned 327 of 480 against
joint space ZERO of 480**, p = 7.3e-99, discordant 327 to 0, with the joint-space arm timing out on
every single cell while the learned arm converges in 58 s mean on 3,881 network Jacobians against
joint space's 13,417. There is no cost comparison because the arms share no solved cell -- print a
dash, and note that the dash here means the baseline solved nothing, not that the data is missing.

**Both iiwa grasp rows are 0-2 of 480 on BOTH arms**, replicating at 480 cells what stage
NLOPTTUNE found at 60: nothing Drake exposes makes the augmented Lagrangian solve an iiwa grasp.
Rows where both arms sit at the floor carry **no comparison and no cost column** — that is the one
narrow caveat, and it does not touch the pose rows where the result is.

**The augmented Lagrangian is extraordinarily sensitive to the starting point, far more than either
other solver, and that is a finding in its own right.** On Panda contained grasp the `native`
protocol gives 327 of 480 and the `paired` protocol gives **zero**, with every cell timing out; on
iiwa pose it is 298 -> 118 and on Panda pose 289 -> 114. IPOPT's largest protocol effect on the same rows
is 476 -> 471, and SNOPT's is 441 -> 301. So an AL started at a shared infeasible `q_init` cannot get
its multipliers moving before the clock runs out, where an interior-point method barely notices. This
is why the NLopt column must be read per protocol and never pooled, and it is the sharpest
demonstration in the project that **the two start protocols answer different questions**.

**The adopted configuration is doing exactly what it was adopted for.** On the pose row the learned
arm uses 4,025 network Jacobians per cell against the joint-space arm's 9,550 and finishes in 80 s
mean against 174 s (paired: 7,066 and 141 s against the same 9,550 and 174 s) — i.e. it converges rather than exhausting the budget, which is the feasibility
criterion the configuration was fielded on.

That answers flag criterion 3, untested before this campaign because the flat 180 s arm predated the
adopted configuration: six of the eight rows solve something and the solver ordering is untouched.

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

**The second mechanism was MEASURED AT A 45 s CAP, so it should shrink at 180 s; the first should
not.** "A cheaper chart fits more iterations inside the cap" is cap-dependent by construction, whereas
"a smaller chart eliminates runaway configurations" is not. **The ladder is deliberately not
re-measured and the question is CLOSED rather than deferred**: nothing touching the charts changed,
the rungs are selected by the gain ceiling rather than by cells so a new grid cannot revise the
choice, and the one condition under which re-measuring would have been worth ~350 core-hours — heavy
timeouts at 180 s, the regime where the cap-dependent half does the work — is answered above:
timeouts are at most 2 cells of 480. Quote the ladder tables as the 45 s / free-grasp record and do
not restate this mechanism as a property of the current status quo.

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

### Step rejection: measured, refuted, and it exposed something bigger (stage STEP)

**The last lever on the solver axis, and it is closed.** Filter tuning leaves the program exactly
as written and changes only which trial points are accepted, so it was never touched by the
gradient-damping refutation. Two stages settled it. `stage_STEP` in `cluster/gen_manifest.py` owns
the five knobs and reuses `STEP_REJECTION_KNOBS` as a **whitelist**, the mirror of stage SWEEP's
blacklist, so the two questions cannot merge from either side. All five are now proven to reach
their solver from its own parameter echo (`tests/test_solver_plumbing.py`) — they never had been.

**The screen: 16 settings x 6 rows x 60 cells, and not one reached significance.** Best pooled over
the 240 selection cells was IPOPT `theta2`/`soc0`/`theta1` at 203 against 191 (p = 0.10-0.16) and
SNOPT `mstep0p5` at 127 against 110 (p = 0.060). The `theta_max_fact` table was narrowed in advance
by the old probe's own archive, which is still on disk: `full_theta10` is **bit-identical** to
`full_base` to fourteen digits, so >= 10 is a proven null and the selftest refuses one.

**The confirmation at 480 cells x 12 rows refuted both promoted settings, and reversed the screen.**
`theta1` is **significantly worse** — pooled +322/-421 over 2,880 cells, p = 0.00032 — and on the
very row the screen liked best, Panda pose paired, the screen's 45 -> 53 became **404 -> 391**.
`soc0` is a clean null (+302/-321, p = 0.47). **The mechanism column reversed too**: at 60 cells
`theta1` appeared to cut restoration on Panda pose 34.7% -> 24.9%, and at 480 cells it *raises* it,
26.8% -> 31.4%. A plausible mechanism agreeing with a spurious outcome did **not** protect against
the false positive; both were the same noise.

**The SNOPT hypothesis is refuted with a named replacement.** A smaller `Major step limit` was
predicted to convert INFO 41 (`current point cannot be improved`, a line search that cannot find a
step) into convergence. It does not: `mstep0p5` gains 15 cells of 360 while INFO 41 stays at
**exactly 81**, unchanged. What falls is INFO 13 `nonlinear infeasibilities minimized`, 51 -> 42. The
step limit helps SNOPT *reach* feasibility, not escape a stalled line search. It is also a **trade,
not a win** — +5/+7 on the iiwa against -7/-6 on the Panda — which is why the pre-registered harm
clause excluded it. Measured at 60 cells only.

#### What it exposed: per-cell outcomes are unstable, and that is the real finding

Pooled over 5,760 cell-comparisons, with a same-configuration re-run as the control:

| comparison | failures that became successes | successes that became failures | net |
| --- | --- | --- | --- |
| same config, different run | 27/458 = **5.9%** | 14/5302 = **0.3%** | +13 |
| default -> `theta1` | 322/445 = **72.4%** | 421/5315 = **7.9%** | -99 |
| default -> `soc0` | 302/445 = **67.9%** | 321/5315 = **6.0%** | -19 |

**About 70% of the learned arm's residual failures are recovered by a single filter-option change —
and the same change breaks 6-8% of the cells that already worked.** Successes outnumber failures
about 12:1, so the small proportional loss cancels the large proportional gain almost exactly. This
is not a cap artefact: the same-config control flips 5.9%, and even failures that **converged
wrong rather than timing out** recover at 48-67%.

So the residual failures are **not intrinsically hard**. On the iiwa contained-grasp row `soc0`
recovers 35 of the 44 cells that only a 180 s budget otherwise reaches — but it also recovers 73% of
the cells that *even 180 s cannot solve*, so the recovery is undirected, not targeted at the
budget-bound ones. The honest statement is that a cell's outcome is close to a coin weighted by
trajectory, and **no single global setting wins, because the reshuffle is symmetric in proportion.**

**The implication is for multi-start, not for tuning.** If ~70% of failures yield to *some* setting,
the value is in varying the setting per attempt and reporting "solved within k restarts" — the
harness already supports exactly that (`solved_within_k`), and CLAUDE.md already lists more guesses
per target as an open item. Picking one filter setting globally is refuted; picking several and
taking the best is a different, untested, and **legitimate** proposition, since it searches over
*solver configurations* rather than over initial guesses.

**Nothing is adopted** (Thomas's call in advance: report, do not field). The knobs stay plumbed and
`None`. The prior 16-cell lead that motivated this — `theta_max_fact=1` at 14/16 against 11/16 — was
measured on `ddp-r1` when the runaway was live, and does not survive the adopted rungs.

**A methodological number worth keeping**: CLAUDE.md's "reproducibility at the cap is +/-1 cell" was
measured on 60-cell grids. At 480 cells with 52-88 timeouts a same-configuration re-run moves up to
**7 net and 19 discordant** cells. Every one of the 41 discordant cells across 12 such rows was
cap-bound, and the three rows with zero timeouts had zero discordance — so the solve path is exactly
reproducible on cells that converge, and the band scales with the cap-bound population, not the grid.

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
- *The cap*: median iterations 142 / 203 / **338** / 338 / 338 at 10 / 20 / 45 / 90 / 180 s, i.e.
  medians saturate by 45 s. That is why 45 s was the cap for the exploratory campaigns; **the
  campaign cap is now 180 s**, adopted with containment because beyond the median only the *tail*
  gains and the contained-grasp rows live in the tail.
- *Startup*: 40 concurrent `import torch, pydrake, ikflow, jrl` take 10 s total, so the venv can stay
  on the shared filesystem (copying to node-local `$TMPDIR` costs 231 s and buys nothing).
  `torch.compile` costs ~35 s cold, ~17 s warm per process.

**A shard set spanning THREE collections cannot be merged by `collect_results.sh` alone**, which
searches every prior staging directory as of 2026-09-20 but originally searched only the previous
one. The symptom is the merger reporting 22 of 24 shards missing from directories that *exist*,
because incremental rsync creates the directory and skips a `summary.json` it already shipped.
Nothing is lost — the shard summaries are all on local disk, split across staging directories — and
the remedy is to build the union locally, which is verifiable rather than trusted: re-merging rows
that had merged normally reproduced all seven exactly. **Collect less often than an item takes, or
expect to build the union.**

**Solver logs are node-local, one archive per run.** Writing one ~20 KB log per (cell x arm) onto
Lustre put 35,596 small files — 87% of a collection's file count — on a filesystem that is metadata-op
bound at that size, and a routine collection had drifted from three minutes to thirty. Per-cell logs
now go to node-local `$TMPDIR` (keyed on run tag *and* pid, since a node runs eight workers) and roll
into one `solver_logs.tar.gz` at the end of `run_grid`: 40,733 files → 2,964, 950 MB → 316 MB,
~30 min → **43 s**. `collect_results.sh` is incremental by default and never ships `state/`.

**Any cluster-wide check on a shared account must be scoped to this project's own jobs, by JOB
name.** The account is shared across Thomas's projects, so `LLstat | grep -c RUNNI` refuses whenever
anything at all is running. But the scoped fix then filtered `squeue -n run_items.sh` — the payload
script's *filename* — and a job's name comes from the submitter
(`--job-name=lik_bench_${MANIFEST%.txt}`), so `--reclaim`'s guard was unconditionally 0 and never
refused for its whole life: it would have stolen items from 32 live workers, each of which would then
have written its shard to a claim someone else re-ran. Now `lik_bench_$MANIFEST_NAME`, counting
`PENDING` as well as `RUNNING`, and verified live in both directions. **The transferable lesson: a
guard nobody has observed refusing has not been tested.** `cluster/README.md` carries the filter
itself.

**The laptop suspends when idle**, which accounted for every multi-hour stall this repo ever
recorded (GNOME suspends after 900 s idle *even on AC*). Three plausible solver-level theories each
fit part of the evidence before `journalctl` matched every stall to the minute. **Any long unattended
run here must hold a sleep inhibitor** (`systemd-inhibit --what=sleep:idle --mode=block`, launched
with `setsid`), and a "hung" unattended process is diagnosed by checking
`journalctl -b | grep "suspend now"` against the stall window *first*. Long benchmarks now run on the
cluster.

**Reproducibility at the cap scales with the cap-bound population, not the grid.** On 60-cell grids
two runs of the same configuration scored 34/60 and 35/60, the differing cell hitting the wall clock
in both; at 480 cells with 52-88 timeouts a same-configuration re-run moves up to **7 net and 19
discordant** cells, and rows with zero timeouts have zero discordance. So the solve path is exactly
reproducible on cells that converge — see stage STEP's measurement of this. Worth remembering before
reading a small difference as a real effect.

**Plan around SuperCloud's monthly maintenance** — second Tuesday, compute down Monday evening to
Wednesday morning, nothing survives it. Next window: 2026-10-12 to 10-14.

## Where the project stands, and what is next

**MEASURED AT THE STATUS QUO (stage STATUSQUO): the learned arm wins 15 of the 24 solver x experiment
cells, ties 7 and loses 2** — interior point 6/2/0, augmented Lagrangian 5/3/0, SQP 4/2/2. Under
IPOPT it wins six of eight rows decisively (every pose row on both robots, p from 2.9e-37 to
4.6e-68, and Panda contained grasp by ~150 cells) and ties the two iiwa contained-grasp rows, with
no losses. **Both losses in the whole table are iiwa contained grasp under SQP**, so one verdict is
solver-dependent — which is what the solver axis exists to expose, and it is a property of SQP on
this problem rather than of the formulation, since the joint-space arm degrades under SNOPT too.
Under NLopt all four pose rows are decisive learned wins against a joint-space arm that never
exceeds 31 of 480.

The one honest caveat everywhere is per-iteration cost: **14x** joint space's on the iiwa
contained-grasp tie (Table 3, 27.29 s against 1.90 s), an implementation property with a known ~3x
dispatch floor and **out of scope to fix** (Thomas: architecture/infra work on the CPU dispatch
bottleneck is "future work/possibly not in scope at all"). It is a number to report. It is also not
a constant: on Panda contained grasp joint space needs 744 median iterations against the learned
arm's 169, so the premium there is ~2-3x — hardening the task costs the joint-space arm its
cheapness.

**The rescue rate is the quantity the success counts hide: 82-99% on every row** (see the table
under "Headroom and the rescue rate"). Containment is what makes that matter, leaving 157-263
joint-space failures available to rescue.

Thomas's roadmap (2026-09-04), with status: **(1) iiwa checkpoint training — DONE**, the
reduced-capacity ladder is trained, measured and has a selection rule; **(2) SNOPT and NLOPT —
DONE**, see the solver axis; **(3) performance tuning and formulation tweaks for getting the best
results with the learned formulation** — the live item. Note that Thomas naming formulation work as
a work item is **not** a standing licence to invent formulations; what is compared remains his call,
made explicitly in advance.

**Four things are closed and must not be reopened.** The **solver axis**, on all three method
classes, with exactly two settings fielded. **Placement**: shelf-contained at the fingertips for both
tasks, `--target-placement auto` resolving to `shelf` for both, and `free` surviving only as the
reproducer for archived grasp columns — a retired SETTING, never a row again
(`stage_STATUSQUO`'s selftest refuses a non-contained placement). **The status quo itself**, accepted
2026-09-21, which is what any future result is stated against. And **multi-start**, which stage STEP's
instability finding appears to invite and which Thomas ruled out: *"Let's not go into multi-starting
yet, treat it as somewhat out-of-scope for now, and possibly for this project altogether."* So do not
build it, do not propose it as the next step, and do not lead a report with it; `solved_within_k`
already exists in the harness and needs no work. The instability measurement stands as a reported
finding on its own.

**Stage `STATUSQUO` is the campaign of record** (ran 2026-09-19/20, jobs 5681825-5681828, 480 items,
~19 h wall on 4 nodes x 8 workers). **24 logical runs** = both adopted rungs x 2 experiments x both
protocols x all three solvers, 480 cells each at 180 s, seed 1, arms `learned,numerical`, each solver
at its adopted configuration and **no settings axis** (the selftest refuses one). Read it with
`scripts/report_statusquo.py`, which prints the quartet per row, the 24-cell tally and the three flag
criteria mechanically; `cluster/STATUSQUO_RUNBOOK.md` holds the run's own record. All acceptance
checks passed, including the decisive one: iiwa `n4` contained grasp under IPOPT reproduces
`sc_CAP_iiwa_n4_mug_180_*` exactly on all four numbers, across a different stage, the raised cluster
caps and the Drake pin.

### Smaller open items

Everything here is genuinely open; closed questions live above.

- **A `q_c == 0` arm** (the draft's eq. 4) would quantify what the correction buys, now that the
  penalty has established the `c`/`q_c` redundancy is real.
- **Non-dimensionalise the conditioning pose's translation against its rotation**, the way
  `eaik-experiment` scales its Jacobian rows by a 1.12 m length scale. Never tried.
- **Least-squares domain extension**, from Thomas's unreleased IFT-IK paper, is deferred but belongs
  to *this* project. (Trust-region *solving* belongs to a different project and is out of scope.)
- **The analytic chart's residual 0.6%** — future work by decision; arXiv:2503.03992.
- **Fold a small optimization smoke run into checkpoint validation.** Thomas's idea, explicitly
  deferred (*"Obviously, not worth it right now, but a cool idea for the future"*).
  `cluster/export_and_screen_job.sh` screens on intrinsic metrics only, and those do not predict
  cells. A handful of cells through `src/benchmark.py` at export time would catch a chart that
  screens clean and solves badly; the export job already loads the solver.
- **A harder problem formulation** beyond the hardened scene, if he wants one — his idea, his call.
- **The Drake pin is a nightly that expires ~2026-11-02.** Move it to 1.58.0 once that carries
  PR 25002, and restore the published-checksum path.
- `stage_H` in `cluster/gen_manifest.py` is a generic cross-test harness, kept for whatever knob next
  needs one. It is the one stage in that file not tied to a closed question.
