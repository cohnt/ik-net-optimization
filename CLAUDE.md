# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code (ROS-style package `combining_kinematics`) for solving inverse kinematics with a **normalizing-flow IK network (IKFlow) placed inside a Drake optimization program**, so that collision avoidance, joint limits, and task costs are imposed on the network's *output*. The point of the repo is the three-way comparison of formulations for the same IK problem:

| Formulation | Decision variables | `VarsToQ` |
| --- | --- | --- |
| **learned** (`Panda/IiwaIKProgram`, `...MugProgram`) | conditioning pose `c` (xyz+rpy, 6), latent `z` (`network_width`), `correction` (7) | forward pass of the IKFlow model + `correction` |
| **learned, task-parameterised** (`...MugProgramTaskParam`) | grasp pose in the mug frame `X_MG` (6), latent `z`, `correction` (7) | `c` computed from `X_MG`, then as above |
| **numerical** (`...ProgramNumerical`) | joint angles `q` (7) | identity |
| **analytic** (`...ProgramAnalytic`) | end-effector pose `xyz_rpy` (6) + redundancy parameter `psi` (1) | closed-form S-R-S IK (`src/*_analytic_ik.py`) |

All three go through the same `IKFlowProgram` machinery, so a change to constraints/costs affects all of them. `workshop-paper-draft.pdf` is the write-up. The sibling repo `../codebase/` is the analytic-vs-numerical project this one builds on (`scripts/iiwa/iiwa_collision.py` imports `src.iiwa_experiments` from there); it has its own CLAUDE.md and should be treated as read-only from here.

## Environment and running

No package manifest. Dependencies: `pydrake` from a local Drake build (`~/opt/rlg/drake-build`, already on `PYTHONPATH`; provides IPOPT and SNOPT), `torch`, `numpy`, `tqdm`, and `ikflow` + `jrl` (Jeremy Morgan's IKFlow / Jrl packages — **not** installed in the default env at the time of writing; check `python -c "import ikflow"` before assuming a script can run).

Scripts append the repo root to `sys.path` themselves, so run them from anywhere:

```bash
# The current harness (paired grid, feasibility-verified success -- prefer these):
python scripts/panda/panda_benchmark.py --task mug  --targets 15 --guesses 3 --config latent
python scripts/panda/panda_benchmark.py --task pose --targets 15 --guesses 3 --config latent
python scripts/iiwa/iiwa_benchmark.py   --task mug  --targets 12 --guesses 2 --config latent
python scripts/collate.py 'results/*/benchmark/*/summary.json'

# The older per-experiment scripts, kept for comparability with archived results:
python scripts/panda/panda_mug.py            # 3-way mug-grasp comparison, panda
python scripts/panda/panda_mug_headtohead.py [num_tests] [max_wall_time]
python scripts/panda/panda_mug_ablation.py [config_index] [num_tests] [max_wall_time]
python scripts/panda/panda_pose_headtohead.py [num_tests] [max_wall_time]
python scripts/panda/panda_collision.py
python scripts/iiwa/iiwa_mug.py
python scripts/iiwa/iiwa_collision.py        # needs ../codebase on sys.path
```

There is no test suite, lint config, or argparse: scripts are configured by editing the `####### Options #######` block at the top (trial count, `seed`, `visualize`, one or more `ProgramOptions`). Solver logs and summary JSON go to `results/` (gitignored). Visualization goes to Meshcat via `StartMeshcat()`; the mug experiments start a *second* Meshcat instance because `GenerateDiagramWithMug` rebuilds the whole diagram per target.

`models/panda/panda_no_hand.urdf` and `panda_jrl.urdf` used to contain **hardcoded absolute mesh paths** from the machine the repo was developed on (`/home/tangles/Urop/ikflow/...`), which made the panda scene fail to load anywhere else with a Drake "URI resolved to ... which does not exist" error. Both now use `package://combining_kinematics/models/panda/collision_geometries/...` URIs and the meshes are vendored, so the scene is portable; that fix is committed (`c5f7ea0`).

The panda model weights are downloaded by `ikflow` (`panda__full__lp191_5.25m`); the iiwa weights are a local pickle, `models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl` (gitignored — must be obtained separately).

## Architecture

### `src/generic_program.py` — the shared program

`ProgramOptions` is the single dataclass configuring everything (costs, tolerances, solver, seeding, dtype, logging). `IKFlowProgram` owns the Drake diagram, the plant, its `ToAutoDiffXd()` copy, and both contexts.

Constraints are **not** added to Drake one at a time. Each `Create*Constraint` method builds an `IKFlowConstraints(lb, ub, eval_func)` and appends it to `self.constraints`; `ApplyConstraints` then adds a *single* Drake generic constraint whose evaluator (`EvalAllConstraints`) computes `q = VarsToQ(vars)` and the forward kinematics **once** and dispatches the cached `(vars, q, pose)` to every `eval_func`. That sharing is the reason for the indirection — the network forward/backward pass is the dominant cost, so never add a constraint that recomputes `VarsToQ` itself.

`Solve()` configures IPOPT or SNOPT from the options, registers a visualization callback (which also appends every iterate to `options.vars_file` when set), and returns Drake's `MathematicalProgramResult`.

### Per-robot subclasses (`src/panda_program.py`, `src/iiwa_program.py`)

Each robot implements `__init__` (frames, plant sizes, model loading), `create_prog` (declares decision variables, sets initial guesses, builds `self.jacobian_gen`, calls `add_constraints` / `add_costs`), `ik_inference`, and `VarsToQ`. The `...MugProgram` subclasses additionally swap `self.frame` from the end-effector (the frame the flow was *trained* on) to `between_fingers` (the frame the grasp constraint acts on), keeping `X_grasp_ee` so seeds can still be expressed in the network's frame. A mug grasp constrains only the gripper's position in the mug frame (`x = y = 0` exactly -- an equality, because that is what the task is; `z` within `mug_height`), leaving orientation free — hence the overridden `CreateIKConstraint` and `SeedCandidates`.

### Gradients through the flow

`VarsToQ` is dual-path: under `float` it returns a plain forward pass; under `AutoDiffXd` it calls `self.jacobian_gen` (one reverse pass yields both `dq/dvars` and `q`) and chain-rules `jacobian @ vars_gradients` into fresh `AutoDiffXd` objects. Both paths go through `MakeFlowInference(nn_model, ...)` in `src/generic_program.py`, a free function of the lumped variables that closes over the network and nothing else — which is what lets `FlowJacobianGen` memoise `torch.compile(jacrev(...))` per process instead of per program (`ProgramOptions.compile_flow_jacobian`, and see the Jacobian section below). Analytic formulations instead evaluate `pydrake.math` trig on templated types (`RigidTransform_[T]`, `RollPitchYaw_[T]`) so Drake's own autodiff propagates.

Numerical facts worth not rediscovering — several are recorded in code comments and encoded in `ProgramOptions` defaults:

- Evaluate the flow in **float64** (`use_float64=True`). Gradients are analytic (`jacrev` through the flow, chain-ruled into `AutoDiffXd`), so this is not about the solver differencing anything: it is that a float32 network produces *values* with a ~1e-7 noise floor, which corrupts every quantity computed as a difference over a small step — line-search actual-vs-predicted reduction, convergence tests, and SNOPT's optional derivative verification. `snopt_function_precision` tells SNOPT that noise floor when running in float32.
- Don't ask for a position tolerance below the flow's noise floor: `ik_constraint_tol=(1e-4, 0.01)`.
- The IK pose constraint is six rows: the per-axis position error, then the **roll-pitch-yaw residual** `rpy(FK(q)) - rpy(target)` wrapped to (-pi, pi] (`orientation_error_rpy` in `src/generic_program.py`). `ProgramOptions.orientation_error_form` picks the bounds: `rpy` (the default) pins the residual to zero, as `../codebase`'s `EEPoseConstraint` does with `lb == ub`; `rpy_boxed` allows `±ori_tol` per row. Three signed rows are deliberate. Earlier revisions used a single scalar angle `2*arccos(|q.q_target|)`, and taking a norm of a three-component error is exactly what puts a branch point at zero error — its derivative is infinite there, and its `eps` clamp additionally returned an `AutoDiffXd` with an *empty* derivative vector while freezing the row's value at 2.83e-4. Three rows have neither problem: the residual is smooth at the solution with a full-rank Jacobian, and the chart's degeneracies (gimbal lock at `pitch = ±pi/2`, the `±pi` wrap) are properties of the *target pose*, not of the error. Commit `0be5342` holds the retired scalar forms and the measurements behind the decision. `scripts/panda/panda_pose_headtohead.py` compares the joint-space and learned formulations under it.
- Constraint rows with an identically-zero gradient (e.g. the homogeneous row of a transformed point) break LICQ — the mug constraints deliberately drop it.
- The conditioning variable `c` is boxed near the target (`c_position_slack=0.25`) to keep the flow inside its trained workspace. This is a conditioning heuristic, not a correctness requirement — the IK constraint is imposed on `FK(q)`, so an out-of-distribution `c` cannot produce a false solution. It is also loose enough not to exclude valid grasps (`between_fingers` sits 0.1 m from `panda_hand`, so with free orientation and `mug_height=0.04` the valid `c` positions lie within 0.14 m of the mug centre). No sweep in this repo isolates its benefit.
- **There is no seeding search, deliberately.** A previous revision drew 256 `(c, z)` candidates, scored them against the problem's own constraints and started from the best. That is not initialisation, it is solving part of the problem outside the solver, and only the learned formulation can afford it — so it flatters exactly the column under test. The machinery is removed, not merely disabled. `SetStartFromQ(q_init)` is the only way to set a program's initial guess, and every formulation in a comparison must be given the same `q_init`.

### Profiling

There is no profiler in the tree. A standalone one (no Drake, no mug scene) that mirrors the
solver hot path and attributes its cost lives only in history — recover it with
`git show ab3ea15:scripts/profiling/profile_flow.py`, and see that commit's message for the
full measurements. The headline result is that at batch size 1 the flow evaluation is
**entirely CPU-bound**: the GPU is never behind the CPU, and float64 and float32 cost the same
wall time despite a 3.4x difference in actual GPU kernel time. Roughly 70% of a `jacrev` is
CPU-side work — mostly PyTorch/FrEIA Python dispatch, with `cudaLaunchKernel` itself only about
17% — so runtime is bounded by how fast the CPU can describe 2853 operations, not by GPU math.
Even zero-overhead execution would leave only a ~3x ceiling.

### The conditioning frame (read this before touching the learned formulation)

The flow is conditioned on the pose of **the frame it was trained on**, and in both grasp
scenes that is *not* the frame the code used to look up by name. `panda_finray.sdf`
contains its own body called `panda_hand`, welded to `panda_link7` at `[0, 0, 0.134]` with
`rpy [90, 0, 45]`, whereas jrl's Panda -- the model IKFlow was trained against -- puts
`panda_hand` at `[0, 0, 0.107]` with `rpy [0, 0, -45]`. `GetBodyByName("panda_hand")`
returns the finray one, which is **27 mm and 120 degrees** away. The iiwa has the same
class of error: the scene's `iiwa_link_7` is 45 mm short of the flow's frame.

The symptom is quantitative and unmistakable. Running the flow *forwards* on a random
configuration (`rev=False`, which inverts it exactly) returns the latent that would have
produced it:

| robot | at the scene frame | at the calibrated frame | typical `|z|` under the prior |
| --- | --- | --- | --- |
| Panda | 67.6 | 2.23 | sqrt(7) = 2.65 |
| iiwa14 | 12.1 | 2.45 | sqrt(8) = 2.83 |

A latent of 67 is the network reporting that the configuration is astronomically unlikely
for that conditioning pose. Every iterate of every grasp solve was in that regime.
`IKFlowProgram.CalibrateFlowFrame` now measures the offset against `ik_solver.robot.
forward_kinematics` at several configurations, checks it is constant (both frames are
welded to the same link, so it must be), and caches it as `self.X_ee_flow`;
`FlowPoseInWorld()` is what should be used wherever a conditioning pose is formed.
`ProgramOptions.calibrate_flow_frame=False` restores the old behaviour for ablations.

Measured effect on the 256-sample seed for the Panda mug: the best candidate starts
**0.0005 m** off the mug axis instead of **0.098 m**.

### Sharing the flow evaluation between bindings

Each Drake binding evaluates its own callback, so `EvalJointCenteringCost` used to run a
second forward pass and a second `jacrev` at exactly the point `EvalAllConstraints` had
just evaluated. An archived IPOPT log shows 1276 objective evaluations against 1276
constraint evaluations and 455 objective gradients against 490 constraint Jacobians --
about half the network work was redundant. `IKFlowProgram.QAndPose` memoises `(q, pose)`
on the iterate, keyed on the values **and** the AutoDiffXd derivative block (keying on the
value alone would hand back a Jacobian computed against the wrong seed matrix), behind
`ProgramOptions.share_flow_evaluations`, which **defaults on** -- the memoised path returns
bit-identical values and derivatives, so there is no reason to run without it except to
reproduce a pre-overhaul measurement.

### Task-parameterised conditioning (`c_parameterization="task"`)

`GraspTaskParamMixin` in `src/generic_program.py` replaces the free 6-vector `c` with the
grasp pose in the mug frame, `X_MG`, and *computes* the conditioning pose as
`c = X_WM . X_MG . X_GE`. Every `c` the optimiser can name is then a valid grasp, and the
task constraint on `c` becomes a plain bounding box (`x = y = 0`, `z` within the mug
height, orientation free) instead of two nonlinear equality rows -- the shape
`../../minimal-coordinates/ift/eaik-experiment` uses for the same grasp. What stays
nonlinear is the flow's *chart error*, which is what the correction `q_c` exists to
absorb; `q_c` remains a 7-vector in joint space and is untouched by this.

Note `X_GE`, not `X_EG`: converting the grasp pose to the frame the flow speaks in is a
conjugation that does not cancel, and getting it backwards is a bug that survived a long
time in the sibling project. `PandaMugProgramTaskParam` and `IiwaMugProgramTaskParam` are
one-line subclasses of the mixin.

### Benchmarking (`src/benchmark.py`, `scripts/*/[a-z]*_benchmark.py`)

The older head-to-head scripts remain for comparability, but new measurements should use
`src/benchmark.py`, which fixes three things they got wrong:

- **Paired grid.** `num_targets x num_guesses` cells, one solve per cell, no
  retry-on-failure, every formulation on the identical cells -- so success can be compared
  with an exact McNemar test and the CI can bootstrap over whole *targets* (guesses within
  a target are correlated).
- **A shared starting configuration** (`--start paired`). `SetStartFromQ(q_init)` puts each
  arm at the same configuration in its own variables: the joint-space arm at `q_init`, the
  analytic arm at `FK(q_init)` with `psi`/`GC` recovered by inversion, the learned arm at
  `c = FK(q_init)` with `z` from running the flow forwards. Order matters: invert **first**,
  then clip `c` into its box. Clipping first and inverting at the projected pose returns
  `|z| ~ 1e7`, because a random configuration is not a grasp of this mug and the flow is
  right to say so.
- **Both start protocols are measured** (`--start`). `paired` puts every arm at the same
  `q_init` in its own variables. `native` gives each formulation the initialisation it
  would have outside a comparison: the learned arm conditions on the pose the task hands it
  and draws its latent from the prior the flow was trained against, the analytic arm takes
  the target pose and draws its redundancy parameter and branch, and the joint-space arm
  takes a random configuration -- which is what `q_init` already is, so that arm's two
  protocols coincide and any difference between the tables is attributable to the other
  formulations. Neither protocol searches: nothing is scored against the problem's
  constraints or objective before the solve, which is the line the removed 256-sample
  seeding crossed.
- **The paired start is measured, not assumed.** Every cell records `clip_distance`, the
  arm's actual `start_q_error = |q(start) - q_init|` and (for learned arms) the norm of the
  latent it started from, because only the joint-space arm can hold the shared start
  exactly. See "How paired the paired start actually is" below -- the numbers are large
  enough that quoting the comparison without them would misdescribe it.

- **Success verified from the returned point**, not from `result.is_success()`. Every
  binding is re-evaluated at the solution and the task is re-measured from `q`, with a
  named `fail_reason`. This matters because *every* learned failure in the archived runs
  was a wall-clock timeout, and a timeout that landed on a valid grasp is a success. Two
  gates that are easy to get wrong: an interior-point method parks *on* the collision
  constraint (value 1 + 1e-7), so the collision gate needs the same slack the binding has;
  and `PandaMugProgramAnalytic` inherits from the *pose* analytic class and never moves
  `self.frame` off `panda_hand`, so the grasp must be measured by asking for
  `between_fingers` by name (that class now moves `self.frame` onto the grasp frame, but
  the gate should not depend on it).

Three switches the scripts grew for this round of measurements. `--compile` turns on the
compiled flow Jacobian and warms it up before the grid, so no cell pays the ~10 s penalty;
because it changes how many iterations the learned arm fits inside a fixed cap, **every run
being compared has to set it the same way**. `--set NAME=VALUE` overrides any
`ProgramOptions` field, so a sweep needs no code edit; it lands in the metadata and the
default tag. And the grid is drawn from a generator local to the script and hashed into
`metadata["grid_hash"]`, so runs that were not measured on the same cells cannot be
compared by accident -- `python scripts/collate.py --pair learned '<glob>'` runs exact
McNemar between runs on matching cells and refuses a grid mismatch.

### How paired the paired start actually is (repaired 2026-08-31)

`SetStartFromQ` gives every arm the same `q_init` expressed in its own variables, but a
formulation can only represent a configuration its variables reach, so what each arm
actually gets is `q_init` *projected onto its own representable set*. Two of the four
projections were unnecessary and have been removed:

- **The learned free-`c` arm now starts exactly at `q_init`.** The 1.2-3.3 rad error of
  the previous protocol came entirely from pre-clipping `c` into its box before the solve,
  which was never required: a Drake initial guess need not satisfy the bounds, and IPOPT
  projects variables into their box itself (`bound_push`). The repaired start sets
  `c = FK(q_init)` unclipped, inverts the flow there (measured: `flow(c, z)` then
  reproduces `q_init` to ~1e-6, the network's noise floor), and closes that residual with
  the correction. `clip_distance` now records how far `c` sits outside its box -- the
  projection IPOPT will apply at its first iterate -- and `legacy_paired_start=True`
  restores the old behaviour for reproducing archived runs.
- **The analytic arm can now represent ~99.4% of configurations instead of ~89%**, with
  `analytic_branches=8` (see the chart section below). Where the chart covers `q_init` the
  start is exact to 1e-11; `start_q_error` keeps recording the remainder per cell.

| arm | `\|q(start) - q_init\|` at the guess | why |
| --- | --- | --- |
| joint space | 0 exactly | its variables are the configuration |
| learned, free `c` | ~1e-6 (pose task: 0.0 measured) | exact: unclipped conditioning pose + inverted latent + correction |
| analytic, 8 branches | 1e-11, or several radians on ~0.6% of starts | exact where the chart covers the configuration |
| analytic, 4 branches | 1e-11, or several radians on ~10% of starts | the historical chart; kept as the `analytic` column |
| learned, task-parameterised | ~3 rad, occasionally 1e16 | `c` encodes a grasp and `q_init` is not one -- exact matching impossible by construction |

A third defect surfaced while verifying the repair: **the pose task's analytic arm was
never given the paired start either.** Its formulation pins `xyz_rpy` to the target with a
`+-ik_constraint_tol` *bounding box*, and `SetStartFromQ` clipped the guess into it -- so
the arm always began at (target pose, `psi(q_init)`, `gc(q_init)`), a median 2.7 rad from
the shared `q_init` in the archived pose tables, regardless of chart coverage. Fixed the
same way as the learned `c`: the guess is set unclipped and the box projection is the
solver's first move. (The mug analytic arm never had this problem -- its task rows are
generic constraints, not variable bounds -- which is why it measured 8e-11.)

The number to keep in mind reading any of this: `start_q_error` measures the *initial
guess*. Where a guess sits outside a variable's bounds, IPOPT projects it at iterate 0,
and `clip_distance` now records exactly that projection distance per cell. "Paired" is
exact at the guess; how much of it survives the solver's own bound projection is a
per-formulation property that the two numbers together describe honestly.

The **task-parameterised learned arm** keeps the latent from inverting at `q_init`'s own
conditioning pose and then moves `c` onto the mug axis; holding `z` fixed while `c` moves
that far can put the flow's output at 1e16 (GLOW's clamped exponentials amplify by up to
`exp(2.5)` per coupling block, twelve blocks deep). Re-inverting at the projected pose is
worse, not better -- the ordering note above measures `|z| ~ 1e7` -- so the start stands as
a projection, the correction closes the (at most +-0.1 rad per joint) part it can
representably close, and `start_q_error` reports the rest per cell.

### The analytic chart: eight branches, and what the last 0.6% is (2026-08-31)

The closed-form map's discrete set is three binary choices -- wrist (B), shoulder (C), and
elbow (A) -- and the implementation historically charted only A = +1, the half away from
the joint limits, following the Panda analytic IK paper. The missing half is a *single
sign*: negate both triangle angles `O2O4O6` and `O2O6O4` (the arm plane's signed angles
flipping together; the elbow reflected across the shoulder-wrist axis). The old
commented-out "Case A1" line (`O2O4O6 - q3_add`) matches no configuration and was a dead
end; the measured elbow relations are `q3 = theta + q3_add - 2*pi` (A = +1) and
`-theta + q3_add` (A = -1), partitioning exactly at `q3 = q3_add - pi = -0.467`.

`ProgramOptions.analytic_branches` selects the chart (default 4, so archived runs stay
reproducible; the benchmark's `analytic8` arm runs the 8-branch chart on the same cells).
`gc(q, branches=3)` recovers all three indices with zero mislabels in 4000 samples.
Round-trip coverage of `IK(FK(q), psi(q), gc(q)) == q`, 4000 random configurations:

| tolerance | 4 branches | 8 branches |
| --- | --- | --- |
| 1e-6 | 89.4% | 99.40% |
| 1e-3 | -- | 99.58% |
| 1e-2 | -- | 99.83% |

**The residual is not singularities** (those are measure zero; this set has positive
measure) **and not branch mislabelling** (the 24/4000 misses are reproduced by *no* branch
of the eight). Two are off by ~4 rad -- a genuinely distinct solution -- and 22 by 1e-3 to
1e-2, clustered where the wrist arcsin argument approaches 1, i.e. near a branch-merge
locus. Consistent with the <=16 self-motion-manifold bound (Burdick/Luck): three binary
indices need not enumerate them all for an arm with the Panda's link offsets. **Left as
future work by decision** -- a recent Panda IK paper with alternative self-motion
parameterisations (arXiv:2503.03992) is the suggested starting point -- and until then
coverage is reported as the curve above, never as "100% up to singularities".



### The ablation ladder, re-run (2026-08-29, RTX 3080 Ti laptop, IPOPT, 20 s cap, compiled)

Panda grasp, learned arm only, 15 targets x 2 guesses, paired start, `--compile`, one grid
for every rung (`grid_hash 64f0c9cdf9be`), so the rungs are comparable cell by cell. Each
rung adds one change to the one above it; `latent-free-c` is `latent` without the task
parameterisation, the control that isolates it.

| rung | success | iters | wall (s) | `\|z\|` at start | median start error |
| --- | --- | --- | --- | --- | --- |
| baseline (uncalibrated frame, no sharing) | 19/30 | 116 | 12.0 | 11.93 | 3.91 |
| + conditioning-frame fix | 21/30 | 107 | 11.2 | 3.03 | 2.32 |
| + shared flow evaluation | 20/30 | 107 | 10.1 | 3.03 | 2.32 |
| + task parameterisation | 22/30 | 198 | 12.9 | 3.03 | 3.58 |
| + latent trust region | **24/30** | 172 | 11.0 | 3.03 | 3.58 |
| latent trust region *without* the task parameterisation | 15/30 | 160 | 13.9 | 3.03 | 2.32 |

Exact McNemar on the shared cells, each change against the rung below it:

| change | success | better / worse | p |
| --- | --- | --- | --- |
| conditioning frame | 21 vs 19 | 7 / 5 | 0.77 |
| shared flow evaluation | 20 vs 21 | 1 / 2 | 1.0 |
| task parameterisation | 22 vs 20 | 6 / 4 | 0.75 |
| latent trust region, with the task parameterisation | 24 vs 22 | 7 / 5 | 0.77 |
| latent trust region, on the free conditioning pose | 15 vs 20 | 3 / 8 | 0.23 |
| the whole stack | 24 vs 19 | 8 / 3 | 0.23 |

**Nothing here is significant.** The stack is worth +5 cells over the baseline and no single
rung is distinguishable from the one below it, so the honest statement is that the task
parameterisation and the latent trust region remain **unproven**, exactly as they were
before -- the ladder did not rescue them, it measured them. Three things it does establish:

- The old "12/30 -> 21/30" framing is gone. Under the repaired starts and the compiled
  Jacobian the *baseline* reaches 19/30, so most of that apparent gain was the broken
  latent start and the iteration budget, not the redesigns.
- The one consistent direction in the table is **negative**: the latent trust region on the
  free conditioning pose loses 8 cells and wins 3. It appears to be worth something only
  alongside the task parameterisation, which is an interaction, not a main effect.
- The task parameterisation nearly doubles the iteration count (198 against 107) for its two
  cells, so it is buying success with work rather than with better conditioning.

Detecting a 5-cell difference at 30 cells is hopeless; a grid several times larger is what
these two design choices would need, and that is a cheaper experiment than it looks now that
a rung costs about six minutes.

### The three-way comparison, both start protocols (2026-08-29, IPOPT, 20 s cap, compiled)

15 targets x 2 guesses, one grid per experiment, `--compile`, feasibility-verified success.
`paired` starts every arm at the same `q_init`; `native` gives each formulation its own
initialisation. The joint-space arm's two protocols coincide by construction, and it does
score identically in all four experiments, which is the harness checking itself.

| experiment | start | learned | numerical | analytic |
| --- | --- | --- | --- | --- |
| Panda pose | paired | **23/30** | 15/30 | 21/30 |
| Panda pose | native | **28/30** | 15/30 | 16/30 |
| Panda grasp | paired | 24/30 | **30/30** | 26/30 |
| Panda grasp | native | 25/30 | **30/30** | **30/30** |
| iiwa pose | paired | 16/30 | 14/30 | -- |
| iiwa pose | native | **30/30** | 14/30 | -- |
| iiwa grasp | paired | 12/30 | **29/30** | -- |
| iiwa grasp | native | 7/30 | **29/30** | -- |

**On the pose task the learned formulation wins, on both robots, under both protocols.**
Exact McNemar against joint space: Panda 23 vs 15 (p = 0.039) paired and 28 vs 15
(p = 0.00098) native; iiwa 16 vs 14 (p = 0.77) paired and 30 vs 14 (p = 3.1e-5) native.
Against the analytic arm on the Panda, 28 vs 16 native (p = 0.0042) but 23 vs 21 paired
(p = 0.77). It also wins on cost wherever it wins on success. This is the draft's central
claim, and it now holds under two different initialisation protocols rather than one.

**On the grasp task it loses to both baselines at this cap** -- Panda 24 vs 30 paired
(p = 0.031) and 25 vs 30 native (p = 0.063), iiwa 12/30 and 7/30 against 29/30 -- but read
the next section before concluding anything from that: on the Panda the entire deficit is
the 20 s cap, and it disappears at 45 s.

**Neither protocol is uniformly kind**, which is the reason to report both:

| arm and experiment | paired | native | p |
| --- | --- | --- | --- |
| learned, iiwa pose | 16/30 | 30/30 | 0.00012 |
| learned, Panda pose | 23/30 | 28/30 | 0.125 |
| analytic, Panda grasp | 26/30 | 30/30 | 0.125 |
| analytic, Panda pose | 21/30 | 16/30 | 0.227 |
| learned, iiwa grasp | 12/30 | 7/30 | 0.227 |
| numerical, all four | -- | identical | 1.0 |

The pattern is that the paired start **handicaps the learned arm on the pose task** -- its
natural start conditions the flow on the target pose and draws the latent from the prior,
whereas the paired start hands it the latent of an unrelated random configuration -- and
handicaps the **analytic arm on the grasp task**, where the shared `q_init` often falls in
the half of the chart it does not cover. Both effects are protocol artefacts rather than
statements about the formulations, and they point in opposite directions, so the two
qualitative conclusions above are robust to the choice.

### What the wall-clock cap was actually measuring (20 s against 45 s)

The same eight experiments at a 45 s cap. **Every baseline is bit-identical at both caps in
all four experiments**, and so is every arm on the pose task -- same successes, same
iteration counts, same solver exit strings. Only the learned arm on the grasp task moves:

| experiment | start | 20 s | 45 s | gained | p |
| --- | --- | --- | --- | --- | --- |
| Panda grasp | native | 25/30 | **30/30** | 5 | 0.063 |
| Panda grasp | paired | 24/30 | 27/30 | 3 | 0.25 |
| iiwa grasp | native | 7/30 | 14/30 | 7 | 0.016 |
| iiwa grasp | paired | 12/30 | 16/30 | 4 | 0.125 |
| everything else | either | -- | identical | 0 | 1.0 |

So **the Panda grasp result at 20 s was measuring the cap, not the formulation**. At 45 s
with its native start the learned arm solves 30/30, the same as joint space and the same as
the analytic arm; the five cells it lost at 20 s all exit "Maximum wallclock time exceeded"
there and "Solved To Acceptable Level" here. The iiwa grasp deficit is real and survives the
cap (14/30 and 16/30 against 29/30), as does the pose result in the learned arm's favour.

The honest way to state the Panda grasp row is therefore in iterations, which are
hardware-independent, rather than in seconds. On the 45 s native run the learned arm reaches
30/30 in a mean of **147 iterations**, against 166 for joint space and 247 for the analytic
arm -- it needs *fewer* solver iterations than either baseline. What it does not have is
their per-iteration cost: each of its iterations carries a network Jacobian at ~14 ms
compiled, against ~2 ms of Drake kinematics, and the profiling says that gap is CPU dispatch
rather than arithmetic. That is an implementation property; the iteration count is the
formulation property, and it is the one to report.

### The two knob sweeps: both are inert (2026-08-29, 20 s cap, paired, learned only)

One factor at a time on the grasp task, 15 x 2 on the same grid as the finals, whose learned
column supplies the default point rather than re-running it.

| `correction_bound` | 0.1 | 0.2 | 0.4 | 0.8 |
| --- | --- | --- | --- | --- |
| iiwa success | 12/30 | 10/30 | 9/30 | 11/30 |
| iiwa median `\|q_c\|` | 0.054 | 0.080 | 0.271 | 0.484 |
| Panda success | 24/30 | 19/30 | 23/30 | 20/30 |

| `latent_trust_region` | 3.0 | 4.0 / 4.3 | 6.0 | off |
| --- | --- | --- | --- | --- |
| iiwa success | 9/30 | 12/30 | 13/30 | 13/30 |
| Panda success | 24/30 | 24/30 | 22/30 | 20/30 |

Nothing in either table is significant against its default (smallest p = 0.27), and neither
has a monotone trend. Two specific things they settle:

- **The correction box is not binding, on either robot.** The fraction of solutions sitting
  on it is **0.00 at every point of both sweeps**, and at the default the median `|q_c|` is
  0.054 against a bound of 0.1. The standing hypothesis -- that the iiwa's +-0.1 rad box was
  sized for a chart 3.8 mm off and must be strangling a chart 16.6 mm off -- is **refuted**.
  The solver takes more of the box when it is given more (0.054 -> 0.484 as the bound goes
  0.1 -> 0.8) and gets nothing for it, which says the extra freedom is used but useless.
- **The latent trust region does nothing that its absence does not.** Combined with the
  ladder, where it was worth +2 cells with the task parameterisation and -5 without it,
  there is now no measurement anywhere in this repo that supports it. The one honest
  observation is that removing it makes solves *pathological* rather than merely worse: the
  Panda `off` run contains a cell that ran 6106 s against a 20 s cap (diagnosed below --
  a nondeterministic C++-level wedge, not a property of the formulation or the iterate).

So the iiwa's grasp deficit is not the correction box and not the latent bound. What remains
of the original suspicion is the chart itself (16.6 mm / 6.4 deg median against the Panda's
3.8 mm / 0.71 deg), which no knob in this program can repair -- which makes asking Julia
about the provenance of `iiwa14__lemon-haze-7__global_step_4.25M.pkl` the next thing worth
doing, and a cheap one.

### The final4 comparison, 20 s cap (2026-09-01, exact paired start, 8-branch column)

15 x 2, `--compile`, seed 0, same grid as final3 (`64f0c9cdf9be` on the Panda mug), so
`collate.py --pair` compares cell by cell. `analytic` is the historical 4-branch chart;
`analytic8` the full chart. Joint space is the comparison's target; analytic is a baseline.

| experiment | start | learned | numerical | analytic | analytic8 |
| --- | --- | --- | --- | --- | --- |
| Panda pose | paired | **25/30** | 15/30 | 21/30 | 14/30 |
| Panda pose | native | **28/30** | 15/30 | 16/30 | 10/30 |
| Panda grasp | paired | 22/30 | **30/30** | 26/30 | 21/30 |
| Panda grasp | native | 27/30 | **30/30** | 29/30 | 24/30 |
| iiwa pose | paired | 18/30 | 14/30 | -- | -- |
| iiwa pose | native | **30/30** | 14/30 | -- | -- |
| iiwa grasp | either | (wedged; queued for re-run) | | -- | -- |

Against joint space, exact McNemar: Panda pose **12/2, p = 0.013** paired (the repaired
start *strengthened* the final3 result, 23-vs-15 p = 0.039 -> 25-vs-15 p = 0.013) and
14/1, p = 0.00098 native; Panda grasp 0/8, p = 0.0078 paired and 0/3, p = 0.25 native at
this cap; iiwa pose 8/4, p = 0.39 paired (a tie, as before) and 16/0, p = 3.1e-5 native.

**The 8-branch chart makes the analytic formulation worse, uniformly.** analytic8 trails
analytic in all four Panda experiments -- 0/6 p = 0.031 (pose native), 9/2 p = 0.065
(pose paired), 7/2 (grasp paired), 5/0 p = 0.0625 (grasp native) -- despite (because of)
representing the start exactly: under the paired start it lands *in* the mirrored
near-limit bundle whenever `q_init` does, and under the native start it draws that bundle
half the time. Starting a solve inside a bundle pinned against the joint limits is worse
than starting at the wide-bundle chart's projection of the same configuration. The
historical 4-branch chart was, in effect, performing branch selection for free -- which is
a genuine finding about unbalanced discrete solution bundles in optimization-IK, and
exactly the phenomenon the `analytic8` column was added to expose.

### The 6106-second cell, diagnosed (2026-08-31): a rare C++ wedge, and what now bounds a solve

The cell (`sweep3_panda_latent_None`, target 0 guess 1) is fully explained and its
watchdog has been **removed**, not tuned. The evidence:

- Its IPOPT log ends after iteration 125 (an `H` step -- regularised Hessian) with no
  `EXIT:` line, and nothing was written for 102 minutes; IPOPT writes one line per
  iteration, so the process was wedged **inside a single iteration**, below Python.
- Re-running the cell (`--cells 0:1`, same grid hash) is **bit-identical through iteration
  125** -- same objective, infeasibility, and step sizes line for line -- and then simply
  continues, solving in 8 s / 154 iterations. Sixteen consecutive attempts all did exactly
  that. So the computation is deterministic and the wedge is not: a rare scheduling-level
  stall (once in 1740 cells), not a numerical pathology of the iterate. The kernel journal
  shows nothing in the window.
- The wedge **released on its own**: the old `SolveTimeout` fired from the constraint
  callback at 6106 s, which is the deadline check finally being *reachable* again -- proof
  that a callback-based kill can never catch this class of stall, only add insult by
  discarding the iterate afterwards.

The design that replaced it, per the standing rule that a watchdog must not throw away the
solver's point:

- `Solve()`'s per-iteration callback stores `program.last_iterate` (in memory, always).
- On any exception, `run_grid` re-verifies that iterate exactly as a returned solution
  (`recovered_feasible`, `recovered_fail_reason`, `recovered_detail` in the record).
- A wedge that releases now ends *cleanly*: the next iteration trips IPOPT's own
  `max_wall_time` and the solve returns its point, which is verified normally. The only
  remaining bound on a wedge that never releases is the OS (`timeout` in the queue
  script), which is where such a bound belongs.
- `SolveTimeout`, `CheckDeadline`, `hard_time_factor` and the `QAndPose` deadline poll are
  deleted -- the poll cost a `time.time()` on the hottest path in the program and could not
  fire when it mattered.

**The mechanism, caught live the same night (2026-09-01, ~00:20).** The first `final4`
iiwa grasp run wedged for five hours at its fourth learned cell, and this time the
per-cell `faulthandler` dumps caught it: eighteen identical stack dumps over the whole
window show the main thread inside a *torch op in the eager float forward pass*
(`VarsToQ` -> `QAndPose` -> `EvalJointCenteringCost`), i.e. **a CUDA call that never
returned** -- and the process survived `timeout`'s SIGTERM for three hours, the signature
of an *uninterruptible* ioctl. The machine runs the GPU with fine-grained **Runtime D3
enabled**, runtime PM `auto`, persistence mode off: when the solver spends minutes in the
numerical/analytic arms (zero GPU work), the device autosuspends, and the learned arm's
next torch op must resume it -- a resume that occasionally hangs for hours and then
releases. That fits every observation: the nondeterminism, the bit-identical re-runs that
sail through, no Python for the whole window, the self-release, SIGTERM immunity, both
robots, an empty kernel journal, and the timing (the wedge struck immediately after a
long GPU-idle numerical solve). Mitigation while the queue runs, root-free: a sidecar
process launching one trivial CUDA kernel per second so the device never goes idle long
enough to suspend. The durable fix needs root -- persistence mode (`nvidia-smi -pm 1`) or
`NVreg_DynamicPowerManagement=0x00` -- and is recorded as an open item. Note the OS-level
`timeout` is *also* powerless against the wedge itself (nothing delivers a signal to a
process stuck in an uninterruptible syscall); it fires when the wedge releases, which is
another reason the recovery path, not a kill, is the right design.

### Measured results (2026-08-28, RTX 3080 Ti laptop, IPOPT, 20 s cap)

Produced with `src/benchmark.py`; raw records in `results/*/benchmark/*/summary.json`.
Success is feasibility-verified from the returned point, solver status reported alongside.
Every arm starts from the same `q_init` and none of them searches for a good start.

**Superseded, but the last thing measured** (15 targets x 2 guesses):

| experiment | learned | numerical | analytic |
| --- | --- | --- | --- |
| Panda grasp, before the overhaul | 12/30 | 27/30 | 28/30 |
| Panda grasp, after | 21/30 | 29/30 | 30/30 |
| Panda pose, after | 24/30 | 15/30 | 14/30 |

On the pose experiment the learned formulation won on success (McNemar p = 0.023 against
joint space, 0.013 against analytic) *and* on cost (median 8.91 against 10.77 and 9.30 on
the cells all three solved) -- the draft's central claim, with paired statistics. On the
grasp experiment it was still behind both baselines (p = 0.008, 0.004).

The grasp rows were produced with two defects in the shared start, both since repaired, so
they are being re-run rather than quoted (`ladder3_*`, `final3_*` in `results/`):

- `GraspTaskParamMixin.SetStartFromQ` inverted the flow at the **uncalibrated** frame, so
  the task-parameterised arm -- the one behind every `task` and `latent` number -- started
  from a latent of norm 40 to 5.7e7 that was then clipped into the +-5 box. The calibrated
  inversion returns 2.8 to 4.1, against the prior's `sqrt(7) = 2.65`.
- `PandaMugProgramAnalytic` left `self.frame` on `panda_hand` while its variables denote
  `between_fingers`, so the analytic arm started 3.0 to 5.6 rad from the shared `q_init`.

The analytic arm's grasp constraint was also loosened to `+-ik_constraint_tol[0]` on the two
axis rows where the learned arm is pinned; it is now pinned too.

**Withdrawn.** Every other table this branch produced was run with the searched 256-sample
start that has since been removed: the Panda grasp 35/45, the Panda pose 45/45, both iiwa
tables, and the whole seven-rung ablation ladder. They measure the seeding, not the change
under test. The 12/30 -> 21/30 above survives because both ends had seeding off, but the
**attribution of that gain to the individual changes is not established**.

Of the five changes in that gain, two need no benchmark to justify:

- the **conditioning frame** was a straight bug -- the network was being asked about poses
  27 mm and 120 degrees from the frame it was trained on, and `|z| = 67.6` against 2.23 is
  a direct measurement of that, independent of any solve;
- **sharing the flow evaluation** returns bit-identical values and derivatives at roughly
  twice the throughput, so it cannot cost anything.

The other two -- the **task parameterisation** and the **latent trust region** -- are
genuine design choices whose value is currently unproven. (A third, relaxing the mug-axis
equality, has been removed rather than re-measured: that equality is the definition of the
grasp task, so widening it solves an easier problem instead of solving this one better.)
Re-running the ladder is the first thing to do:

```bash
for CFG in baseline frame eval task latent latent-free-c; do
  python scripts/panda/panda_benchmark.py --task mug --targets 15 --guesses 2 \
      --wall-time 20 --arms learned --config $CFG --tag ladder2_$CFG
done            # roughly 10 minutes per configuration
```

**What the grasp failures are.** Every learned grasp failure is a wall-clock timeout, and
on the valid run 8 of 9 sit within 1 cm of the mug axis while 9 of 9 are in collision
(median collision value 1.34 against a limit of 1.0). The binding difficulty is collision
avoidance, not the learned chart. Successes take a median of 74 iterations and 5.8 s of the
20 s cap.

**Chart accuracy, which no protocol question touches.** `|FK(flow(c, z)) - c|` over 200
random targets:

| robot | position, median / p90 | orientation, median / p90 |
| --- | --- | --- |
| Panda | 3.8 mm / 9.4 mm | 0.71 deg / 2.76 deg |
| iiwa14 | 16.6 mm / 64.5 mm | 6.39 deg / 30.2 deg |

The iiwa flow is a four- to eightfold worse chart than the Panda's, which no amount of
optimisation will repair. Two things to check before any more iiwa work: `correction_bound`
is +-0.1 rad, chosen for a chart 3.8 mm off and almost certainly binding at 16.6 mm; and
`models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl` is a local checkpoint of
unknown provenance, unlike the Panda's published weights, so a 4x worse chart may be a
statement about the checkpoint rather than about the method.

### Why the Jacobian is a `jacrev` and not a JVP

Measured, so it does not get re-litigated. The network Jacobian is computed in exactly one
place -- `VarsToQ`'s AutoDiffXd branch builds `J = dq/dvars` (7 x 21) and chain-rules
`J @ vars_gradients` into fresh `AutoDiffXd` objects -- and that single `J` serves every
consumer: all eleven constraint rows and, since the flow-sharing commit, the objective
gradient too, because Drake composes `dg/dq` itself through AutoDiffXd.

On one Panda grasp solve (task-parameterised, 27 IPOPT iterations, 1.66 s): 35 float
`VarsToQ` calls at 5.8 ms and 35 AutoDiffXd calls at 39.6 ms, so **the Jacobian is 84% of
the solve and the flow altogether is 96%**. IPOPT asked for 28 constraint Jacobians and 28
objective gradients; the memoisation served all 56 from 35 evaluations.

Reverse mode is the right primitive because of the shape: 7 outputs against 21 inputs, of
which only 13 reach the network (`dq/dq_c = I` analytically). Reverse needs 7 passes,
forward needs 13-20, so 7 is optimal. Measured at that shape:

| method | time | max diff vs current |
| --- | --- | --- |
| `jacrev` + matmul (current) | 17.1 ms | -- |
| vmapped VJP, 7 cotangents | 16.4 ms | 0 (bit-identical) |
| vmapped JVP, 20 tangents | 47.1 ms | 6.2e-8 |
| vmapped JVP, 13 tangents, correction analytic | 47.3 ms | 4.7e-8 |
| plain forward pass | 5.3 ms | -- |

Forward mode is 2.8x slower, and 13 tangents cost the same as 20 -- confirming the profiling
result that this is CPU-dispatch bound, not FLOP bound, so reducing the tangent count buys
nothing. It also disagrees with reverse by 6e-8, float32-level error in a float64 model.

**`torch.compile` on the `jacrev` is now implemented** and is worth taking. Measured in this
environment at batch 1 in float64: 17.98 ms eager against 13.55 ms compiled on the `jacrev`
alone, and 21.3 ms against 14.4 ms on a whole AutoDiffXd `VarsToQ` (**1.48x**), agreeing
with eager to 3.5e-15 relative on both value and derivatives, with dynamo reporting a single
graph and no recompiles across iterates. The one-off cost is 8-14 s. The reason the old
comment in `panda_program.py` said it was not worth it ("39.5 ms either way, against a
200 ms compilation penalty per program") is that the compiled object was a *bound method*:
`torch.compile` guards on everything the callable closes over, so each of the thirty
programs in a grid re-triggered dynamo. `MakeFlowInference` closes over the network alone
and `FlowJacobianGen` memoises on `(network, shape, dtype)`, so one graph now serves every
program in the process. It is off by default and turned on with the benchmark scripts'
`--compile`, because more iterations inside a fixed wall-clock cap *moves the learned arm's
success rate*, and only the learned arm benefits.

The one place a VJP genuinely would have won is already gone: the objective-gradient path
used to compute the whole 7 x 21 Jacobian to produce a single 1 x 20 row, where one VJP
with cotangent `dcost/dq` would have replaced seven passes. Sharing the constraint's
Jacobian captured that instead. The remaining runtime levers are iteration count and
`torch.compile` (~1.3x on the `jacrev`), not the AD mode.

### Next steps

Ordered by what the evidence actually supports. Nothing here weakens the problem: the
grasp axis stays an equality, every arm starts from the same `q_init`, and no formulation
searches for a start.

**Required before any new claim**

1. Re-run the ablation ladder under the valid protocol (`scripts/run_queue.sh`, stage 1).
   The 12/30 -> 21/30 gain on the Panda grasp is real but unattributed; the frame fix and
   the shared flow evaluation are justified without it, the task parameterisation and the
   latent trust region are not. Only now is the ladder *paired*: until the grid moved to a
   local generator, `CalibrateFlowFrame`'s draws meant the `baseline` rung ran on different
   targets from every other rung.

**Panda grasp -- the failures are collision, not the chart**

Every failure is a timeout, 8 of 9 land within 1 cm of the mug axis, and 9 of 9 are in
collision. So target the collision constraint and the iteration budget, not the task rows.

2. Collision-constraint shaping: `influence_distance_offset` (0.1) and the 0.1 scaling on
   that row set the gradient the solver has to follow while it is far from contact. Never
   swept.
3. IPOPT `mu_strategy="adaptive"`, `max_iter`, `nlp_scaling_method`. Plumbed, never swept.
   The archived logs show `lg(mu)` collapsing to -8 within 50 iterations and then hundreds
   of iterations of tiny steps, which is the signature `adaptive` exists for.
4. ~~`torch.compile` on the `jacrev`~~ **done**: `--compile`, 1.48x on an AutoDiffXd
   `VarsToQ`, one graph per process. Note it is not quite "pure throughput, no protocol
   question" as this list used to claim -- only the learned arm evaluates a network, so
   inside a fixed cap it moves that arm's success rate and nothing else's.
5. `correction_bound` swept (see 9; inert). `correction_cost_weight` (0) still unexamined. The
   draft only says `q_c ~ 0`. Widening the bound is a *formulation* change, not tuning: at
   the limit the correction can represent any configuration and the learned arm becomes a
   reparameterised joint-space arm, so a success gain has to be read against how much of
   the box the solutions actually use (`median_correction_inf`, `correction_binding`).
6. ~~`latent_trust_region` radius~~ **swept, inert** (3.0 / 4.0 / 6.0 / off: 24, 24, 22, 20
   on the Panda; 9, 12, 13, 13 on the iiwa). No measurement in this repo supports it now.
   The soft form (`latent_cost_weight`) is still unmeasured, but there is little reason to
   expect more of it than of the hard form.
7. Report success against the wall-clock cap as a curve rather than one number, plus
   iteration counts, which are hardware-independent. Successes take 5.8 s median of a 20 s
   cap, so the single number is mostly describing the tail.
8. More guesses per target in the paired grid -- same guesses for every arm -- reported as
   "solved within k restarts". This is the only honest form of multi-start and the harness
   already does it.

**iiwa -- model quality first, optimisation second**

The iiwa flow is a 4-8x worse chart than the Panda's (16.6 mm / 6.4 deg median against
3.8 mm / 0.71 deg). Optimisation cannot repair that, so establish it before tuning.

9. ~~Sweep `correction_bound` upward~~ **done, and the premise was wrong**: the box is not
   binding on either robot (0.00 of solutions sit on it; median `|q_c|` is 0.054 against a
   bound of 0.1), and widening it to 0.8 changes nothing.
10. Ask Julia about `iiwa14__lemon-haze-7__global_step_4.25M.pkl`. It is a local checkpoint
    of unknown provenance against the Panda's published weights; if it is undertrained then
    the iiwa grasp row is a statement about the checkpoint, not about the method. Cheap and
    decisive.
11. Only then, the divergent cells -- a few reach 3.4e7 constraint violation, which is the
    latent or the conditioning pose escaping despite the trust region.

**Fairness repairs owed to the baselines** (12 and 13 done; 16 is new and open)

12. ~~Pin the analytic arm's two mug-axis rows~~ **done**.
13. ~~`PandaMugProgramAnalytic` never moves `self.frame` off `panda_hand`~~ **done**, and it
    was worse than a trap: it made the analytic arm start 3-5.6 rad from the shared
    `q_init`. Its pose offset is now measured from the scene rather than fitted, too.
16. ~~The analytic chart cannot represent ~30% of configurations~~ **not a defect**: it
    charts four of the eight branches deliberately, the others being very close to the
    joint limits (see the Panda analytic IK paper). The consequence stands and is measured
    per cell as `start_q_error` -- in the uncharted region the analytic arm cannot be given
    the shared start -- but nothing here is to be repaired.

**Speculative, in the project's spirit**

14. The `c` / `q_c` redundancy: with both free, many pairs give the same `q`, so the active
    constraint gradients are rank-deficient. The task parameterisation addresses part of
    this; a `q_c == 0` arm (the draft's eq. 4) would quantify what the correction buys.
15. Non-dimensionalise the conditioning pose's translation against its rotation, the way
    `eaik-experiment` scales its Jacobian rows by a 1.12 m length scale, so the `c` block is
    dimensionally coherent.

### Scenes and utilities (`src/utils.py`, `models/`)

`BuildEnv(meshcat, directives_file, extra_directives=None)` builds the diagram from a Drake model-directives YAML, registering `package.xml` so `package://combining_kinematics/...` URIs resolve; `extra_directives` is a list of `ModelDirective` objects appended to the loaded ones **in memory**, so a caller can add models to a scene without writing to the tracked YAML. `GenerateDiagramWithMug(q, program, yaml_file, meshcat)` uses exactly that: it constructs an `add_model`/`add_weld` pair for a mug at the gripper pose of `q` (the weld pose is passed as a `pydrake.common.schema.Transform`, not formatted into text) and rebuilds the diagram. The YAML on disk is never modified, so a crash or interrupt mid-call cannot leave a stray mug in a tracked scene — it used to append-then-truncate the file, which could. Targets in the mug experiments are generated by sampling collision-free `q` and welding a mug at the resulting gripper pose, so every target is known to admit a valid grasp.

`HiddenPrints` suppresses Drake/ikflow output at the file-descriptor level and is used around program construction inside sweeps.

Notebooks in `notebooks/` are the exploratory counterpart to `scripts/` and import the same `src/` modules; they run from the `notebooks/` directory (they append `../` to `sys.path`).
