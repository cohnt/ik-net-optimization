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
python scripts/panda/panda_benchmark.py --task mug  --targets 15 --guesses 3 --config latent --start native
python scripts/panda/panda_benchmark.py --task pose --targets 15 --guesses 3 --config latent --start native
python scripts/iiwa/iiwa_benchmark.py   --task mug  --targets 12 --guesses 2 --config latent --start native
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

Each robot implements `__init__` (frames, plant sizes, model loading), `create_prog` (declares decision variables, sets initial guesses, builds `self.jacobian_gen`, calls `add_constraints` / `add_costs` / `SeedInitialGuess`), `ik_inference`, and `VarsToQ`. The `...MugProgram` subclasses additionally swap `self.frame` from the end-effector (the frame the flow was *trained* on) to `between_fingers` (the frame the grasp constraint acts on), keeping `X_grasp_ee` so seeds can still be expressed in the network's frame. A mug grasp constrains only the gripper's position in the mug frame (`x = y = 0` exactly, `z` within `mug_height`), leaving orientation free — hence the overridden `CreateIKConstraint` and `SeedCandidates`.

### Gradients through the flow

`VarsToQ` is dual-path: under `float` it returns a plain forward pass; under `AutoDiffXd` it calls `self.jacobian_gen = torch.func.jacrev(ik_inference_with_value, has_aux=True)` (one reverse pass yields both `dq/dvars` and `q`) and chain-rules `jacobian @ vars_gradients` into fresh `AutoDiffXd` objects. Analytic formulations instead evaluate `pydrake.math` trig on templated types (`RigidTransform_[T]`, `RollPitchYaw_[T]`) so Drake's own autodiff propagates.

Numerical facts worth not rediscovering — several are recorded in code comments and encoded in `ProgramOptions` defaults:

- Evaluate the flow in **float64** (`use_float64=True`). Gradients are analytic (`jacrev` through the flow, chain-ruled into `AutoDiffXd`), so this is not about the solver differencing anything: it is that a float32 network produces *values* with a ~1e-7 noise floor, which corrupts every quantity computed as a difference over a small step — line-search actual-vs-predicted reduction, convergence tests, and SNOPT's optional derivative verification. `snopt_function_precision` tells SNOPT that noise floor when running in float32.
- Don't ask for a position tolerance below the flow's noise floor: `ik_constraint_tol=(1e-4, 0.01)`.
- The IK pose constraint is six rows: the per-axis position error, then the **roll-pitch-yaw residual** `rpy(FK(q)) - rpy(target)` wrapped to (-pi, pi] (`orientation_error_rpy` in `src/generic_program.py`). `ProgramOptions.orientation_error_form` picks the bounds: `rpy` (the default) pins the residual to zero, as `../codebase`'s `EEPoseConstraint` does with `lb == ub`; `rpy_boxed` allows `±ori_tol` per row. Three signed rows are deliberate. Earlier revisions used a single scalar angle `2*arccos(|q.q_target|)`, and taking a norm of a three-component error is exactly what puts a branch point at zero error — its derivative is infinite there, and its `eps` clamp additionally returned an `AutoDiffXd` with an *empty* derivative vector while freezing the row's value at 2.83e-4. Three rows have neither problem: the residual is smooth at the solution with a full-rank Jacobian, and the chart's degeneracies (gimbal lock at `pitch = ±pi/2`, the `±pi` wrap) are properties of the *target pose*, not of the error. Commit `0be5342` holds the retired scalar forms and the measurements behind the decision. `scripts/panda/panda_pose_headtohead.py` compares the joint-space and learned formulations under it.
- Constraint rows with an identically-zero gradient (e.g. the homogeneous row of a transformed point) break LICQ — the mug constraints deliberately drop it.
- The conditioning variable `c` is boxed near the target (`c_position_slack=0.25`) to keep the flow inside its trained workspace. This is a conditioning heuristic, not a correctness requirement — the IK constraint is imposed on `FK(q)`, so an out-of-distribution `c` cannot produce a false solution. It is also loose enough not to exclude valid grasps (`between_fingers` sits 0.1 m from `panda_hand`, so with free orientation and `mug_height=0.04` the valid `c` positions lie within 0.14 m of the mug centre). No sweep in this repo isolates its benefit.
- `SeedInitialGuess` draws `num_seed_samples` `(c, z)` pairs in one batched forward pass, ranks them on the IK constraint alone, then re-scores the top `seed_refine_top_k` against *all* constraints because the collision query is expensive. Set `num_seed_samples=0` for programs used only to sample targets or share a loaded network. Note that batching is only free in float32: measured on this machine a batch-256 forward costs 8.8 ms in float32 (1.5x a batch of 1) but 84 ms in float64 (12x), because float64 is FLOP-bound past about batch 4 while float32 stays overhead-bound. Seeding therefore pays ~80 ms per program for a precision it does not need — it only ranks candidates.

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
  right to say so. `--start native` instead gives each arm its own best initialisation
  (256-sample seeding for the learned arm), which is the "best practice" comparison.
- **Success verified from the returned point**, not from `result.is_success()`. Every
  binding is re-evaluated at the solution and the task is re-measured from `q`, with a
  named `fail_reason`. This matters because *every* learned failure in the archived runs
  was a wall-clock timeout, and a timeout that landed on a valid grasp is a success. Two
  gates that are easy to get wrong: an interior-point method parks *on* the collision
  constraint (value 1 + 1e-7), so the collision gate needs the same slack the binding has;
  and `PandaMugProgramAnalytic` inherits from the *pose* analytic class and never moves
  `self.frame` off `panda_hand`, so the grasp must be measured by asking for
  `between_fingers` by name.

### Measured results (2026-08-28, RTX 3080 Ti laptop, IPOPT, 20 s cap)

Produced with `src/benchmark.py`; raw records in `results/*/benchmark/*/summary.json`.
Success is feasibility-verified from the returned point, solver status reported alongside.
These supersede the archived `results/panda/mug/headtohead` numbers, which were scored by
`result.is_success()` and were run before the conditioning-frame fix.

**The ladder** (Panda grasp, learned arm only, 15 targets x 2 guesses, 256-sample seeding),
cumulative -- each row adds one change to the row above:

| configuration | success | timeouts | mean iters | mean jacrevs | wall (s) |
| --- | --- | --- | --- | --- | --- |
| baseline (scene frame, separate cost evaluation) | 12/30 | 18 | 144 | 145 | 17.7 |
| + calibrated flow frame | 18/30 | 12 | 103 | 104 | 14.2 |
| + shared flow evaluation | 22/30 | 9 | 129 | 130 | 12.3 |
| + task-parameterised `c` | 22/30 | 8 | **72** | **73** | 9.0 |
| + latent trust region `||z|| <= 4` | **23/30** | 7 | 80 | 81 | 8.8 |
| + mug axis relaxed to +-1e-4 | 20/30 | 10 | 94 | 95 | 11.2 |
| latent trust region + axis, but free `c` | 19/30 | 11 | 108 | 109 | 11.9 |

Two things to keep: the task parameterisation buys no extra successes but cuts iterations
by 44%, which is a conditioning result, not a feasibility one; and **relaxing the mug-axis
equality made things worse** (23 -> 20), so the two rows stay pinned. `mug_axis_tol` is
kept as an option only to record that the experiment was run.

**Head to head, shared start** (`--start paired`, 15 targets x 2 guesses). The learned
column is the same protocol before and after, so the two are directly comparable:

| experiment | learned | numerical | analytic |
| --- | --- | --- | --- |
| Panda grasp, before | 12/30 | 27/30 | 28/30 |
| Panda grasp, after | **21/30** | 29/30 | 30/30 |
| Panda pose, after | **24/30** | 15/30 | 14/30 |

On the pose experiment the learned formulation wins on success (McNemar p = 0.023 against
joint space, 0.013 against analytic) *and* on cost (median 8.91 against 10.77 and 9.30 on
the cells all three solved). That is the draft's central claim, now with paired statistics.
On the grasp experiment it is still behind both baselines (p = 0.008, 0.004), where before
it was behind them by roughly twice as much.

**Where it did not work: the iiwa grasp.** Every configuration lands in the same place --
baseline 3/24, frame fix with free `c` 5/24, the Panda's winning configuration 4/24, with
18-21 of 24 cells timing out in all three. The Panda's gains do not transfer. The iiwa
grasp scene is far more cluttered (four shelf units and seven decorative mugs against the
Panda scene's two shelves and a bin) so the collision query dominates, and the iiwa flow is
a different artifact (a local `lemon-haze-7` pickle rather than the published Panda
weights). Diagnosing that is open work; do not assume it is the same problem as the Panda's
was. The iiwa *pose* experiment is fine: learned 21/24 against joint space 22/24, at 25
solver iterations against 31.

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

The one place a VJP genuinely would have won is already gone: the objective-gradient path
used to compute the whole 7 x 21 Jacobian to produce a single 1 x 20 row, where one VJP
with cotangent `dcost/dq` would have replaced seven passes. Sharing the constraint's
Jacobian captured that instead. The remaining runtime levers are iteration count and
`torch.compile` (~1.3x on the `jacrev`), not the AD mode.

### Scenes and utilities (`src/utils.py`, `models/`)

`BuildEnv(meshcat, directives_file, extra_directives=None)` builds the diagram from a Drake model-directives YAML, registering `package.xml` so `package://combining_kinematics/...` URIs resolve; `extra_directives` is a list of `ModelDirective` objects appended to the loaded ones **in memory**, so a caller can add models to a scene without writing to the tracked YAML. `GenerateDiagramWithMug(q, program, yaml_file, meshcat)` uses exactly that: it constructs an `add_model`/`add_weld` pair for a mug at the gripper pose of `q` (the weld pose is passed as a `pydrake.common.schema.Transform`, not formatted into text) and rebuilds the diagram. The YAML on disk is never modified, so a crash or interrupt mid-call cannot leave a stray mug in a tracked scene — it used to append-then-truncate the file, which could. Targets in the mug experiments are generated by sampling collision-free `q` and welding a mug at the resulting gripper pose, so every target is known to admit a valid grasp.

`HiddenPrints` suppresses Drake/ikflow output at the file-descriptor level and is used around program construction inside sweeps.

Notebooks in `notebooks/` are the exploratory counterpart to `scripts/` and import the same `src/` modules; they run from the `notebooks/` directory (they append `../` to `sys.path`).
