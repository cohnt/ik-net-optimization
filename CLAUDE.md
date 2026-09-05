# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code (ROS-style package `combining_kinematics`) for solving inverse kinematics with a **normalizing-flow IK network (IKFlow) placed inside a Drake optimization program**, so that collision avoidance, joint limits, and task costs are imposed on the network's *output*. The point of the repo is the three-way comparison of formulations for the same IK problem:

| Formulation | Decision variables | `VarsToQ` |
| --- | --- | --- |
| **learned** (`Panda/IiwaIKProgram`, `...MugProgram`) | conditioning pose `c` (xyz+rpy, 6), latent `z` (`network_width`), `correction` (7) | forward pass of the IKFlow model + `correction` |
| **numerical** (`...ProgramNumerical`) | joint angles `q` (7) | identity |
| **analytic** (`...ProgramAnalytic`) | end-effector pose `xyz_rpy` (6) + redundancy parameter `psi` (1) | closed-form S-R-S IK (`src/*_analytic_ik.py`) |

All three go through the same `IKFlowProgram` machinery, so a change to constraints/costs affects all of them. `workshop-paper-draft.pdf` is the write-up. The sibling repo `../codebase/` is the analytic-vs-numerical project this one builds on (`scripts/iiwa/iiwa_collision.py` imports `src.iiwa_experiments` from there); it has its own CLAUDE.md and should be treated as read-only from here.

## Environment and running

No package manifest. Dependencies: `pydrake` from a local Drake build (`~/opt/rlg/drake-build`, already on `PYTHONPATH`; provides IPOPT and SNOPT), `torch`, `numpy`, `tqdm`, and `ikflow` + `jrl` (Jeremy Morgan's IKFlow / Jrl packages — **not** installed in the default env; check `python -c "import ikflow"` before assuming a script can run).

Scripts append the repo root to `sys.path` themselves, so run them from anywhere:

```bash
# The current harness (paired grid, feasibility-verified success -- prefer these):
python scripts/panda/panda_benchmark.py --task mug  --targets 15 --guesses 3 --config latent
python scripts/panda/panda_benchmark.py --task pose --targets 15 --guesses 3 --config latent
python scripts/iiwa/iiwa_benchmark.py   --task mug  --targets 12 --guesses 2 --config latent
python scripts/collate.py 'results/*/benchmark/*/summary.json'

# The older per-experiment scripts (panda_mug*.py, panda_pose_headtohead.py, *_collision.py,
# iiwa_mug.py) are kept for comparability with archived results; iiwa_collision.py needs
# ../codebase on sys.path.
```

There is no test suite or lint config; the older scripts are configured by editing the `####### Options #######` block at the top. Solver logs and summary JSON go to `results/` (gitignored). Visualization goes to Meshcat via `StartMeshcat()`; the mug experiments start a *second* Meshcat instance because `GenerateDiagramWithMug` rebuilds the whole diagram per target.

`models/panda/panda_no_hand.urdf` and `panda_jrl.urdf` used to contain hardcoded absolute mesh paths from the original development machine, which made the panda scene fail to load anywhere else. Both now use `package://combining_kinematics/...` URIs and the meshes are vendored, so the scene is portable (fixed in `c5f7ea0`).

The panda model weights are downloaded by `ikflow` (`panda__full__lp191_5.25m`); the iiwa weights are a local pickle, `models/iiwa14/iiwa14__lemon-haze-7__global_step_4.25M.pkl` (gitignored — must be obtained separately).

## Architecture

### `src/generic_program.py` — the shared program

`ProgramOptions` is the single dataclass configuring everything (costs, tolerances, solver, seeding, dtype, logging). `IKFlowProgram` owns the Drake diagram, the plant, its `ToAutoDiffXd()` copy, and both contexts.

Constraints are **not** added to Drake one at a time. Each `Create*Constraint` method builds an `IKFlowConstraints(lb, ub, eval_func)` and appends it to `self.constraints`; `ApplyConstraints` then adds a *single* Drake generic constraint whose evaluator (`EvalAllConstraints`) computes `q = VarsToQ(vars)` and the forward kinematics **once** and dispatches the cached `(vars, q, pose)` to every `eval_func`. That sharing is the reason for the indirection — the network forward/backward pass is the dominant cost, so never add a constraint that recomputes `VarsToQ` itself.

`Solve()` configures IPOPT or SNOPT from the options, registers a visualization callback (which also appends every iterate to `options.vars_file` when set), keeps `program.last_iterate` so any abnormal exit can still be verified, and returns Drake's `MathematicalProgramResult`.

### Per-robot subclasses (`src/panda_program.py`, `src/iiwa_program.py`)

Each robot implements `__init__` (frames, plant sizes, model loading), `create_prog` (declares decision variables, sets initial guesses, builds `self.jacobian_gen`, calls `add_constraints` / `add_costs`), `ik_inference`, and `VarsToQ`. The `...MugProgram` subclasses additionally swap `self.frame` from the end-effector (the frame the flow was *trained* on) to `between_fingers` (the frame the grasp constraint acts on), keeping `X_grasp_ee` so seeds can still be expressed in the network's frame. A mug grasp constrains only the gripper's position in the mug frame (`x = y = 0` exactly — an equality, because that is what the task is; `z` within `mug_height`), leaving orientation free — hence the overridden `CreateIKConstraint` and `SeedCandidates`.

### Gradients through the flow

`VarsToQ` is dual-path: under `float` it returns a plain forward pass; under `AutoDiffXd` it calls `self.jacobian_gen` (one reverse pass yields both `dq/dvars` and `q`) and chain-rules `jacobian @ vars_gradients` into fresh `AutoDiffXd` objects. Both paths go through `MakeFlowInference(nn_model, ...)`, a free function of the lumped variables that closes over the network and nothing else — which is what lets `FlowJacobianGen` memoise `torch.compile(jacrev(...))` per process instead of per program (`ProgramOptions.compile_flow_jacobian`). Analytic formulations instead evaluate `pydrake.math` trig on templated types (`RigidTransform_[T]`, `RollPitchYaw_[T]`) so Drake's own autodiff propagates.

Numerical facts worth not rediscovering, several encoded in `ProgramOptions` defaults:

- Evaluate the flow in **float64** (`use_float64=True`). Gradients are analytic, so this is not about the solver differencing anything: a float32 network produces *values* with a ~1e-7 noise floor, which corrupts every quantity computed as a difference over a small step — line-search actual-vs-predicted reduction, convergence tests, and SNOPT's optional derivative verification. `snopt_function_precision` tells SNOPT that noise floor when running in float32.
- **`ik_constraint_tol` forms no constraint bound.** The pose rows are a hard equality (`lb = ub = 0`); what survives of the option is the benchmark's gate. See "The tolerance ladder" below.
- The IK pose constraint is six rows: the per-axis position error, then the **roll-pitch-yaw residual** `rpy(FK(q)) - rpy(target)` wrapped to (-pi, pi] (`orientation_error_rpy`). `ProgramOptions.orientation_error_form` picks the bounds — `rpy` (the default) pins the residual to zero as `../codebase`'s `EEPoseConstraint` does, `rpy_boxed` allows `±ori_tol` per row. **Three signed rows are deliberate**: earlier revisions used a scalar angle `2*arccos(|q.q_target|)`, and taking a norm of a three-component error is exactly what puts a branch point at zero error — infinite derivative there, plus an `eps` clamp that returned an `AutoDiffXd` with an *empty* derivative vector. Three rows have neither problem, and the chart's degeneracies (gimbal lock, the `±pi` wrap) are properties of the *target pose*, not of the error. Commit `0be5342` holds the retired forms and the measurements behind the decision.
- Constraint rows with an identically-zero gradient (e.g. the homogeneous row of a transformed point) break LICQ — the mug constraints deliberately drop it.
- The conditioning variable `c` is boxed near the target (`c_position_slack=0.25`) to keep the flow inside its trained workspace — a conditioning heuristic, not a correctness requirement, since the IK constraint is imposed on `FK(q)` and an out-of-distribution `c` cannot produce a false solution. Note the exposure cliff at 0.5 recorded under "The flow's own gain": the default sits just under it, which is luck rather than design.
- **There is no seeding search, deliberately.** A previous revision drew 256 `(c, z)` candidates, scored them against the problem's own constraints and started from the best — not initialisation but solving part of the problem outside the solver, which only the learned formulation can afford, so it flatters exactly the column under test. The machinery is removed, not disabled. `SetStartFromQ(q_init)` is the only way to set an initial guess, and every formulation in a comparison gets the same `q_init`.

### Why the Jacobian is a `jacrev` and not a JVP

Measured, so it does not get re-litigated. The single `J = dq/dvars` (7 x 21) serves every consumer — all eleven constraint rows and the objective gradient — and on one Panda grasp solve **the Jacobian is 84% of the solve and the flow altogether is 96%**.

Reverse mode is right because of the shape: 7 outputs against 21 inputs, of which only 13 reach the network, so reverse needs 7 passes where forward needs 13-20. Measured, `jacrev` + matmul 17.1 ms and a vmapped VJP 16.4 ms (bit-identical) against a vmapped JVP at 47 ms — **forward mode is 2.8x slower**, 13 tangents cost the same as 20 (confirming CPU-dispatch rather than FLOP binding), and it disagrees with reverse by 6e-8, float32-level error in a float64 model. The one place a VJP would have won, the objective-gradient path, is already covered by sharing the constraint's Jacobian.

**`torch.compile` on the `jacrev` is implemented and worth taking**: **1.48x** on a whole AutoDiffXd `VarsToQ`, agreeing with eager to 3.5e-15, one dynamo graph and no recompiles, for a one-off 8-14 s local / ~35 s cold on the cluster. An old comment saying it was not worth it was measuring a *bound method* — `torch.compile` guards on everything the callable closes over, so each of thirty programs re-triggered dynamo. It is off by default and turned on with `--compile`, because more iterations inside a fixed cap **moves the learned arm's success rate** and only the learned arm benefits — so every run being compared must set it the same way.

### Profiling

No profiler in the tree. A standalone one (no Drake, no mug scene) lives only in history — recover it with `git show ab3ea15:scripts/profiling/profile_flow.py`. The headline result is that at batch size 1 the flow evaluation is **entirely CPU-bound**: the GPU is never behind the CPU, and float64 and float32 cost the same wall time despite a 3.4x difference in actual GPU kernel time. Roughly 70% of a `jacrev` is CPU-side work — mostly PyTorch/FrEIA Python dispatch, with `cudaLaunchKernel` itself only about 17% — so runtime is bounded by how fast the CPU can describe 2853 operations. Even zero-overhead execution would leave only a ~3x ceiling.

### The conditioning frame (read this before touching the learned formulation)

The flow is conditioned on the pose of **the frame it was trained on**, and in both grasp scenes that is *not* the frame the code used to look up by name. `panda_finray.sdf` contains its own body called `panda_hand`, welded to `panda_link7` at `[0, 0, 0.134]` with `rpy [90, 0, 45]`, whereas jrl's Panda — the model IKFlow was trained against — puts `panda_hand` at `[0, 0, 0.107]` with `rpy [0, 0, -45]`. `GetBodyByName("panda_hand")` returns the finray one, which is **27 mm and 120 degrees** away. The iiwa has the same class of error: the scene's `iiwa_link_7` is 45 mm short of the flow's frame.

The symptom is quantitative and unmistakable. Running the flow *forwards* on a random configuration (`rev=False`, which inverts it exactly) returns the latent that would have produced it:

| robot | at the scene frame | at the calibrated frame | typical `|z|` under the prior |
| --- | --- | --- | --- |
| Panda | 67.6 | 2.23 | sqrt(7) = 2.65 |
| iiwa14 | 12.1 | 2.45 | sqrt(8) = 2.83 |

A latent of 67 is the network reporting that the configuration is astronomically unlikely for that conditioning pose. Every iterate of every grasp solve was in that regime. `IKFlowProgram.CalibrateFlowFrame` now measures the offset against `ik_solver.robot.forward_kinematics` at several configurations, checks it is constant (both frames are welded to the same link, so it must be), and caches it as `self.X_ee_flow`; `FlowPoseInWorld()` is what should be used wherever a conditioning pose is formed. `ProgramOptions.calibrate_flow_frame=False` restores the old behaviour for ablations. **This fix is worth 18 of the ablation ladder's 23 cells** — see the ladder below.

### Sharing the flow evaluation between bindings

Each Drake binding evaluates its own callback, so `EvalJointCenteringCost` used to run a second forward pass and a second `jacrev` at exactly the point `EvalAllConstraints` had just evaluated. An archived IPOPT log shows 1276 objective evaluations against 1276 constraint evaluations and 455 objective gradients against 490 constraint Jacobians — about half the network work was redundant. `IKFlowProgram.QAndPose` memoises `(q, pose)` on the iterate, keyed on the values **and** the AutoDiffXd derivative block (keying on the value alone would hand back a Jacobian computed against the wrong seed matrix), behind `ProgramOptions.share_flow_evaluations`, which **defaults on** — the memoised path returns bit-identical values and derivatives, so there is no reason to run without it except to reproduce a pre-overhaul measurement.

### Scenes and utilities (`src/utils.py`, `models/`)

`BuildEnv(meshcat, directives_file, extra_directives=None)` builds the diagram from a Drake model-directives YAML, registering `package.xml` so `package://combining_kinematics/...` URIs resolve; `extra_directives` is a list of `ModelDirective` objects appended to the loaded ones **in memory**, so a caller can add models to a scene without writing to the tracked YAML. `GenerateDiagramWithMug(q, program, yaml_file, meshcat)` uses exactly that: it constructs an `add_model`/`add_weld` pair for a mug at the gripper pose of `q` (the weld pose passed as a `pydrake.common.schema.Transform`, not formatted into text) and rebuilds the diagram. The YAML on disk is never modified, so a crash mid-call cannot leave a stray mug in a tracked scene — it used to append-then-truncate the file, which could. `BuildEnv(meshcat=None)` skips visualization outright, which is *not* the same as passing `None` through to `ApplyVisualizationConfig` (Drake would start its own).

Targets in the mug experiments are generated by sampling collision-free `q` and welding a mug at the resulting gripper pose, so every target is known to admit a valid grasp. `HiddenPrints` suppresses Drake/ikflow output at the file-descriptor level and is used around program construction inside sweeps.

Notebooks in `notebooks/` are the exploratory counterpart to `scripts/` and import the same `src/` modules; they run from the `notebooks/` directory.

## Rules the campaign established

These were each learned by getting them wrong, at the cost of whole tables. They govern the
formulation and the harness, and none of them is negotiable without Thomas.

### The tolerance ladder: constraint bounds exact, then solver tolerance, then a looser gate

Thomas's ruling: *"IK constraint tol should always be zero. The whole point is that it's an
equality constraint, satisfied exactly. Tolerance should be zero in the mathematical program,
only appearing in solver tolerance."*

Until 2026-09-03 the pose constraint's position rows were `lb = -1e-4, ub = +1e-4`. That is
not a slightly looser equality: it is an inequality, and an interior-point method parks *on*
the face of one instead of driving the residual to zero. The evidence, from 480 persisted
cells — orientation was already a true equality and converged five orders tighter than
position, in the same constraint, in the same solve:

| arm | median `pos_error` (boxed) | median `rpy_error` | on the box |
| --- | --- | --- | --- |
| learned | 9.999e-05 | 1.38e-08 | 67-84% |
| joint space | 1.000e-04 | 6.65e-10 | 93-97% |
| analytic | 2.35e-05 | **8.46e-03** (p90 = 1.00e-02) | orientation, always |

**The analytic arm's was a fairness defect, not merely a numerical one.** Its pose target was
imposed by a box on its decision variables carrying the whole `ik_constraint_tol` tuple, so it
received ±0.01 rad of orientation freedom per axis while the arms it is a baseline for were
pinned to zero. It used all of it, and `max_violation` reported 0.00, because a box is
satisfied right up to its face.

The fix collapsed the residuals by five to nine orders of magnitude (see EQ1 below) and is
guarded: `tests/test_constraint_bounds.py` reads the bounds Drake was actually handed and
fails if any of them drifts back. Both sibling projects already followed this ladder —
`../codebase`'s `EEPoseConstraint` passes `lb = ub = extract_xyzrpy(target)`;
`eaik-experiment`'s reachability row is `lb=[0], ub=[0]` under a comment reading "(no slack)".

#### Rung 3: the gate stays deliberately looser, and the gap must not be closed

**`ik_constraint_tol = 1e-4` for the program's rows, `task_tol = 1e-3` for the task gate.**
Thomas: *"go back to 1e-4 actual tol and 1e-3 task tol, to avoid this issue (that's why I did it
in the first place)."* The reason is the same parking behaviour: on 480 iiwa pose cells the
joint-space arm's median `pos_error` was 1.0001e-04 against a 1e-4 bound, with 64% of its
solutions a rounding error above it and none above 1.01e-4 (learned: 9.9971e-05, 33% above). A
gate at exactly 1e-4 scores which side the last ulp fell on, and it *appeared to reverse* the one
row the learned arm loses — iiwa pose paired going from 296-vs-332 (p = 0.016 against) to
199-vs-120 (p = 2.5e-08 in favour). A coin toss dressed as a result.

**Never set an acceptance gate equal to a bound the solver is optimising against**, and before
proposing to tighten one, check the distribution of the gated quantity: if the solutions are
pinned to the bound, tightening measures noise. The collision gate carries the binding's own
slack for the same reason. A second verdict `feasible_relaxed` is recorded at `task_tol` so the
question stays re-analysable; the relaxation is worth +10 to +18 cells of 480 on the grasp task
and **exactly zero on the pose task**, moving no ordering.

### A region an initial guess may violate must be a general constraint, never a variable bound

IPOPT's `bound_push` projects the initial guess into every *bounding box* before evaluating
anything, so a box silently reshapes the start protocol. This bit twice.

**The conditioning-pose box.** Pre-clipping `c` into its box teleported it to the box face
while the latent stayed tuned to the unprojected pose, making the "exact" and old
"pre-clipped" protocols land on bit-identical iterate-0 lines. It is now
`AddLinearConstraint(I, lb, ub, c)` (`CBoxConstraint`) at all three sites.

**The latent's own `±5` box** was left as a bounding box after that repair, and this one was
worse — `SetStartFromQ` clipped the inverted latent itself, so the projection was ours, not
IPOPT's. The flow is a bijection, so `flow(c, InvertFlow(q, c))` reproduces `q` exactly, but
only at the *unclipped* latent; the inversion routinely returns components past ±5, so the
clip moved the start by radians and the cell was scored `unrepresentable_start` — an arm
recorded as unable to represent a configuration it represents exactly. Measured on iiwa pose
paired, 20 s, same grid:

| | before | after |
| --- | --- | --- |
| learned success | 11/60 | **40/60** |
| cells scored `unrepresentable_start` | 49 | 0 |
| median `\|q(start) - q_init\|` | 3.79 | 0.0000 |

The arm starts at `\|z\| ~ 7.9`, outside the region, and the solver walks it to `\|z\| ~ 2.9` on
its own — the whole point of the region being a constraint rather than a bound. **Every archived
paired learned column predating this is void.**

Two structural notes. The box now lives in **one** method, `LatentBoxConstraint()`, because the
first repair fixed `generic_program.py` while the mug subclasses overrode `BoundingBoxConstraint`
and carried their own copies — the pose arms were fixed and the grasp arms silently were not. And
nothing may project a guess without recording that it did (`clip_distance`). Variable bounds
remain fine for regions a start always respects (the correction's ±0.1), and infeasible initial
guesses are acceptable by policy — Thomas: *"we're not assuming feasible initial guesses."*

### No invented formulations, and no weakening of the problem

A "task-parameterised" grasp reformulation (`GraspTaskParamMixin`, decision variable = the
grasp pose in the mug frame) existed here and was fielded as the benchmark's "learned" arm.
**It is not the paper's formulation and should never have existed** — Thomas: *"You were
*never* supposed to do the task-parameterized version... Constructing new formulations and
passing them off as ones I've already written is completely unacceptable."* The learned
formulation is eq. (6) of the draft, exactly as `PandaMugProgram`/`IiwaMugProgram` implement
it: free conditioning pose `c`, latent `z`, correction `q_c`, the grasp imposed as constraint
rows through `FK(q)` (mug-axis equality, height band, orientation free). The machinery is
**removed outright**, mirroring the seeding precedent, and every number produced with it is
void and has been re-measured.

The same principle bans weakening the problem. The mug-axis rows are an equality because that
*is* the task — a `mug_axis_tol` option that widened them was removed, not defaulted to zero.
Improvements must come from the formulation or the solver, never from making the question
easier.

### The correction penalty is a stated part of the learned formulation

`correction_cost_weight = 10`, approved by Thomas on 2026-09-02 (*"A penalty on the correction
term is acceptable"*). The draft says `q_c ~ 0` without specifying how that is imposed; this is
what imposes it. Two consequences: the weight is a **stated** part of the formulation and must
appear wherever the learned arm is described, and every table must still show what the penalty
buys and costs — the with/without comparison is paired on the same grid, never dropped once
adopted.

**Options naming learned-only decision variables must be guarded.** `add_costs` applied
`correction_cost_weight` unconditionally, but `correction` exists only on the learned arm and
all three formulations share one `ProgramOptions`. So `--set correction_cost_weight=10` raised
`AttributeError` inside every numerical/analytic program's construction and each of those
columns scored **0 of 480 in about 10 ms per cell**. The failure mode is worth remembering:
a whole column of zeroes with `median_max_violation = nan`, at three orders of magnitude below
the cap, with `fail_reason = "error"` rather than a named task gate. **Any arm reporting a
per-cell wall time three orders below the cap is not solving badly, it is not solving at all.**
`_abort_on_dead_arm` in `src/benchmark.py` now aborts a run when an arm fails identically, in
under a second, on its first three cells — this pattern has cost two whole columns of cluster
campaigns, both caught only during analysis after the compute was spent.

## Benchmarking (`src/benchmark.py`, `scripts/*/[a-z]*_benchmark.py`)

The older head-to-head scripts remain for comparability, but new measurements use
`src/benchmark.py`, which fixes what they got wrong:

- **Paired grid.** `num_targets x num_guesses` cells, one solve per cell, no retry-on-failure,
  every formulation on the identical cells — so success can be compared with an exact McNemar
  test and the CI can bootstrap over whole *targets* (guesses within a target are correlated).
  Guesses are drawn **per target** (`guesses[ti][gi]`), not shared across targets: sharing
  across *arms* is what pairing needs, sharing across *targets* quantizes start-dependent
  effects into target-sized blocks.
- **Both start protocols are measured** (`--start`). `paired` puts every arm at the same `q_init`
  in its own variables via `SetStartFromQ`: joint space at `q_init`, analytic at `FK(q_init)` with
  `psi`/`GC` recovered by inversion, learned at `c = FK(q_init)` with `z` from running the flow
  forwards. Order matters — invert **first**, then clip; clipping first and inverting at the
  projected pose returns `|z| ~ 1e7`, because a random configuration is not a grasp of this mug
  and the flow is right to say so. `native` gives each formulation the initialisation it would
  have outside a comparison: the learned arm conditions on the pose the task hands it and draws
  its latent from the prior, the analytic arm takes the target pose with a random redundancy
  parameter and branch, and the joint-space arm takes a random configuration — which is what
  `q_init` already is, so that arm's two protocols coincide and any difference between the tables
  is attributable to the others. Neither protocol searches, and the paired start is *measured*
  rather than assumed: every cell records `clip_distance` and `start_q_error`, because only the
  joint-space arm can hold the shared start exactly.
- **Success verified from the returned point**, not from `result.is_success()`: every binding is
  re-evaluated at the solution and the task re-measured from `q`, with a named `fail_reason`. This
  matters because *every* learned failure in the archived runs was a wall-clock timeout, and a
  timeout that landed on a valid grasp is a success. Two gates that are easy to get wrong: an
  interior-point method parks *on* the collision constraint (value 1 + 1e-7), so the collision gate
  needs the binding's own slack; and `PandaMugProgramAnalytic` inherits from the *pose* analytic
  class, so the grasp must be measured by asking for `between_fingers` by name.
- **Raw per-cell state is persisted**, not only derived summaries — the returned `q` (and the
  recovered last iterate's `q` on abnormal exit), the decision-variable vector, per-binding
  signed violations, the true `min_distance` and `min_distance_pair` beside `collision_value`,
  and the start. Without this, no geometric quantity can be recomputed after a run without
  re-solving the whole grid, which is exactly what once blocked answering a question from the
  archive.

### Abnormal exits keep the iterate

**Any time-limit or kill mechanism must be paired with recovery of the last iterate.** Thomas
rejected a watchdog that raised from inside the flow-evaluation callback: *"I don't like the idea
of messing with QAndPose to force kill it, since then we don't get an intermediate solution?"*
Every learned-arm failure in the archived runs was a timeout, and a timeout that landed on a valid
grasp is a success. So `Solve()` keeps `program.last_iterate`; any abnormal exit is verified from
that point and recorded as `recovered_feasible` / `recovered_cost` alongside the failure reason;
and the *process* is bounded from outside (OS-level `timeout`), never the solve from inside a
hot-path callback. `SolveTimeout`, `CheckDeadline` and `hard_time_factor` were **deleted** and must
not come back — a callback deadline poll was doubly wrong, unable to fire while the machine slept
and destroying the iterate when it finally did.

Beyond the verdict a record carries `max_violation` and `detail["violations_all"]`;
`collision_value`, `min_distance`, `min_distance_pair`; `start_q_error`, `clip_distance`, `z_norm`;
`median_correction_inf` and `correction_binding` (how much of the ±0.1 box the solutions use — the
check that the learned arm is not quietly becoming a reparameterised joint-space arm); and `q`,
plus `q_lift` and `q_flow` separately under `lift_q`. `median_max_violation` separates the arms by
six orders of magnitude and is worth reading next to any success count.

Three switches. `--compile` turns on the compiled flow Jacobian and warms it up before the grid;
because it changes how many iterations the learned arm fits inside a fixed cap, **every run being
compared has to set it the same way**. `--set NAME=VALUE` overrides any `ProgramOptions` field,
so a sweep needs no code edit; it lands in the metadata and the default tag. And the grid is drawn
from a generator local to the script and hashed into `metadata["grid_hash"]` (with the task as a
*suffix*, since the iiwa's mug and pose grids otherwise hashed identically), so runs not measured
on the same cells cannot be compared by accident — `python scripts/collate.py --pair learned
'<glob>'` runs exact McNemar between runs on matching cells and refuses a grid mismatch.

### How paired the paired start actually is

`SetStartFromQ` gives every arm the same `q_init` expressed in its own variables, but a
formulation can only represent a configuration its variables reach.

| arm | `\|q(start) - q_init\|` at the guess | why |
| --- | --- | --- |
| joint space | 0 exactly | its variables are the configuration |
| learned, free `c` | ~1e-6 (pose task: 0.0 measured) | exact: unclipped conditioning pose + inverted latent + correction |
| analytic, 8 branches | 1e-11, or several radians on ~0.6% of starts | exact where the chart covers the configuration |
| analytic, 4 branches | 1e-11, or several radians on ~10% of starts | the historical chart; the `analytic` column |

Two projections that were once necessary have been removed — the learned arm's pre-clipping of
`c`, and the pose analytic arm's clipping into its `xyz_rpy` box (which had it always beginning
at the target pose, a median 2.7 rad from the shared `q_init`, regardless of chart coverage).
`legacy_paired_start=True` restores the old behaviour for reproducing archived runs.

The number to keep in mind: `start_q_error` measures the *initial guess*. Where a guess sits
outside a variable's bounds IPOPT projects it at iterate 0, and `clip_distance` records that
projection per cell. "Paired" is exact at the guess; the two numbers together describe honestly
how much survives the solver's own bound projection.

### `collision_value` is a penalty, not a clearance

`detail["collision_value"]` is the **raw** value of Drake's `MinimumDistanceLowerBoundConstraint`
— a smooth penalty aggregated over every geometry pair inside the influence distance
(`bound=1e-3`, `influence_distance_offset=0.1`). It is a pure number, not a length, so "1.26
against a limit of 1.0" says nothing about penetration depth. Calibrated against the true minimum
signed distance over 4000 random iiwa configurations, raw < 1.0 is clear, 1.0-1.05 is roughly 0
to -1 mm, 1.2-1.5 is -12 to -19 mm, and 2.0-4.0 is -59 to -124 mm. Every *success* sits at raw
0.9997-1.0005 — parked exactly on contact, which is why the gate carries the binding's own slack.

**`verify()` now records the true signed `min_distance` in metres and the pair attaining it**, so
read that instead; the raw value is kept only because archived records carry it. The collision
row's shape is three `ProgramOptions` fields (`collision_bound`, `collision_influence_offset`,
`collision_row_scale`), defaulting to what was once hardcoded.

## Results: the corrected campaign

Everything below was measured on a program whose pose rows are a true equality, with the
approved `correction_cost_weight = 10`, `--compile`, and the draft's own learned formulation.
**Earlier campaigns (final3, final4, final5, Stages A-D) are superseded and their tables have
been removed from this file** — they were measured either with the boxed pose rows, the
task-parameterised arm, or the latent bounding box. Where one of them established something
that still stands, it is restated here on corrected numbers. The git history holds the
originals.

### THE HEADLINE TABLE: the three-way comparison at 480 cells (2026-09-04)

60 targets x 8 per-target guesses = **480 cells**, 45 s cap, both protocols, both robots, both
tasks, **seed 1** — out of sample, no tuning decision was made on this grid. Joint space is the
comparison's target; the analytic columns are baselines.

| experiment | start | learned | joint space | analytic4 | analytic8 | L vs js | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda pose | native | **474/480** | 228/480 | 259/480 | 155/480 | 250 / 4 | **1.2e-68** |
| Panda pose | paired | **339/480** | 228/480 | 244/480 | 251/480 | 169 / 58 | **9.0e-14** |
| iiwa pose | native | **408/480** | 325/480 | -- | -- | 130 / 47 | **3.4e-10** |
| iiwa pose | paired | 299/480 | 325/480 | -- | -- | 92 / 118 | 0.084 (tie) |
| Panda grasp | native | 447/480 | 457/480 | 450/480 | 387/480 | 20 / 30 | 0.20 (tie) |
| Panda grasp | paired | 424/480 | **457/480** | 376/480 | 418/480 | 18 / 51 | **8.8e-05** |
| iiwa grasp | native | 229/480 | **462/480** | -- | -- | 11 / 244 | **2.2e-58** |
| iiwa grasp | paired | 235/480 | **462/480** | -- | -- | 7 / 234 | **5.0e-60** |

**The harness checks itself and passes.** The joint-space arm is bit-identical between the two
protocols in all four experiments — 228/228, 457/457, 325/325, 462/462 — as it must be, since
its native start *is* a random configuration. Every difference in the other columns is
therefore attributable to their initialisation. `median_start_q_error` is 0.0 exactly for the
learned and joint-space arms under `paired`, and ~1e-11 for the analytic arms where the chart
covers `q_init`.

**The draft's central claim is stronger on a correct program, not weaker.** The learned
formulation wins the pose task on three of four rows — decisively on the Panda under both
protocols and on the iiwa under `native` — and ties on the fourth. On the grasp task it ties on
the Panda under `native` and loses under `paired`. The iiwa grasp row remains the one large
deficit, and its cause is the flow's own gain (see below), not anything in the optimization.

Against the same grid measured on the boxed program, **two conclusions moved, both in the
learned arm's favour**: Panda grasp native went from a loss (437/457, p = 0.013) to a tie, and
iiwa pose paired from a loss (296/332, p = 0.016) to a tie. The learned columns barely moved
(466 → 474, 338 → 339, 407 → 408, 227 → 229). What moved is **joint space, and only on the
pose task**: 249 → 228 on the Panda and 332 → 325 on the iiwa, its grasp columns unchanged.
With the box gone that arm has to actually reach the target rather than stop 1e-4 away, and on
the pose task it pays about 20 cells for it.

**`analytic8` against `analytic4` is the unbalanced-bundle pathology, both signs intact.** Under
`paired` the 8-branch chart wins the grasp task (418 against 376) because it can represent
starts the 4-branch chart forfeits; under `native` it loses badly on both tasks (387 against
450, and **155 against 259** on the pose task), because a uniform draw over eight branches lands
in the narrow near-limit bundles half the time against roughly 10% of configuration-space
volume. This is a genuine finding about unbalanced discrete solution bundles in
optimization-IK, and exactly what the `analytic8` column was added to expose.

### EQ1: what the equality fix actually changed

15 targets x 4 = 60 cells, 45 s, seed 0, all arms, both protocols, both robots, both tasks;
the grid and seed match the boxed run exactly, so every comparison is paired cell for cell.

**The residuals collapse.** Medians over solved cells, boxed → equality:

| experiment | arm | `pos_error` | `rpy_error` |
| --- | --- | --- | --- |
| Panda pose native | learned | 1.00e-04 → **1.82e-09** | 1.77e-08 → 7.03e-09 |
| Panda pose native | analytic | 2.13e-05 → **2.48e-12** | **9.18e-03 → 9.29e-12** |
| Panda pose paired | analytic8 | 1.02e-05 → **2.45e-12** | **6.81e-03 → 8.04e-12** |

Five orders of magnitude on position for every arm, and **nine** on the analytic arm's
orientation — it had been returning poses half a degree off target and scoring them as
successes.

**But no conclusion moved, and this is now measured on 2,880 cells.** At 60 cells, twenty-four
arm-by-arm paired comparisons were **not one significant** (smallest p = 0.52, largest change ±4
cells); pooling the learned arm over all eight experiments and all six caps of the cap curve
below gives **225 better, 204 worse, p = 0.334**. The grasp task is bit-identical, as it must be:
those rows were already `0 == 0` and the mug subclasses override `CreateIKConstraint`. The reason
is clear in hindsight — a boxed solution stopped at 1e-4 and the task gate is 1e-3, so it
**passed the gate anyway**. The box never made a cell easier to succeed at; what it did was
return solutions four to nine orders looser than the program claimed. **The defect was in
solution quality and in fairness between the arms, not in the rankings.**

**The equality is also cheaper**, which is the opposite of what a tighter constraint suggests.
Median iterations on cells both runs solved: learned 52 → 34, 103 → 75, 56 → 38, 118 → 83;
joint space 29 → 24; analytic 10 → 8, 37 → 26. Roughly **30% fewer iterations and 30% less wall
clock**, objective unchanged to within 3%. An equality is unambiguously active, so there is no
active-set question and IPOPT handles it directly rather than through barrier terms on two
inequality faces. That saving does **not** convert into cells — consistent with the diagnosis
that the learned arm's cap-bound failures are a frozen divergent set rather than slow
convergence. Making a diverged solve 30% faster does not rescue it.

### EQ3: the cap curve — the claim holds at an adequate budget

Eight experiments x six caps (5 / 10 / 20 / 45 / 90 / 180 s), 60 cells, seed 0, all arms, same
grids as the boxed run.

**At 180 s, where every arm has saturated:**

| experiment | start | learned | joint space | better / worse | p |
| --- | --- | --- | --- | --- | --- |
| Panda pose | native | **59/60** | 30/60 | 30 / 1 | **3.0e-08** |
| Panda pose | paired | **41/60** | 30/60 | 18 / 7 | **0.043** |
| iiwa pose | native | **60/60** | 41/60 | 19 / 0 | **3.8e-06** |
| iiwa pose | paired | 44/60 | 41/60 | 13 / 10 | 0.68 (tie) |
| Panda grasp | native | 58/60 | 55/60 | 5 / 2 | 0.45 (tie) |
| Panda grasp | paired | 55/60 | 55/60 | 5 / 5 | 1.0 (exact parity) |
| iiwa grasp | paired | 53/60 | 58/60 | 2 / 7 | 0.18 (tie) |
| iiwa grasp | native | 45/60 | **58/60** | 1 / 14 | **0.00098** |

**This is the draft's central claim on a correct program at an adequate budget**: the learned
formulation wins the pose task on both robots under `native` and on the Panda under `paired`,
ties on the iiwa under `paired`, and reaches parity on the Panda grasp under both protocols.
**Only one row of eight goes significantly against it** — the iiwa grasp under `native` — and
its paired counterpart is no longer significant where at 45 s it was 18 vs 58.

**The cap is a budget for the arm that evaluates a network, not a shared budget.** Every
baseline is flat across all six caps, with one exception worth recording: on **iiwa grasp
paired the joint-space arm is itself cap-bound below 20 s**, scoring 59 / 56 / 58 / 58 / 58 / 58
with three cells exiting at the wall clock at 5 s and 10 s. Those cells run 1300-1430 iterations
against that arm's median of 70 (p90 485, max 3000) — so the joint-space arm is not uniformly
cheap, it has a tail. The same wobble predates the equality fix. Everywhere else the baselines
are flat to the cell.

Note the medians are over each arm's *succeeded* cells, and that set grows with the cap, so a
median that rises from 5 s to 180 s is partly composition rather than the same cells taking
longer — which is why comparisons are drawn at the one cap where every arm has saturated.

### EQ4: the correction penalty replicates at 480 cells

480 cells, 45 s, both protocols, both robots, both tasks, **seed 1** (out of sample).
`correction_cost_weight = 10` against the same formulation with the penalty off, exact McNemar
over all 480 shared cells:

| experiment | start | penalty | no penalty | better / worse | p |
| --- | --- | --- | --- | --- | --- |
| iiwa grasp | native | **229/480** | 72/480 | 173 / 16 | **1.8e-34** |
| iiwa grasp | paired | **235/480** | 98/480 | 170 / 33 | **2.0e-23** |
| Panda grasp | native | **447/480** | 379/480 | 93 / 25 | **2.1e-10** |
| Panda grasp | paired | **424/480** | 345/480 | 116 / 37 | **1.1e-10** |
| iiwa pose | paired | 299/480 | 277/480 | 82 / 60 | 0.078 (tie) |
| iiwa pose | native | 408/480 | 397/480 | 33 / 22 | 0.18 (tie) |
| Panda pose | native | 474/480 | 469/480 | 7 / 2 | 0.18 (tie) |
| Panda pose | paired | 339/480 | 343/480 | 64 / 68 | 0.79 (tie) |

**The penalty is a grasp-task effect and only a grasp-task effect** — every grasp row
significant at 1e-10 or below, every pose row a tie. That is what the mechanism predicts, so it is
not a general success multiplier read off a lucky grid, and it costs the pose task nothing.

**The mechanism**, and it is *not* that the correction box was binding (it never is — `on the box`
is 0.00 at every weight): with `c` and `q_c` both free, many pairs give the same `q`, so the
active constraint gradients are rank-deficient and IPOPT spends its budget on a degenerate
direction. Penalising `q_c` breaks that degeneracy, which is why it bites hardest where the active
set is largest. The instrumentation shows it directly — as the weight rises the correction is
driven to zero and the median constraint violation falls three orders of magnitude on the iiwa,
from grossly infeasible to the joint-space arm's own level, while the latent stays put:

| `correction_cost_weight` | 0.001 | 0.01 | 0.1 | 1.0 | 10 (adopted) | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| Panda paired / native (60 cells) | 37 | 44 | 46 | **51** | 50 / **58** | 49 / 51 |
| iiwa paired / native (60 cells) | 13 | 24 | 31 | 34 | **45** / 39 | 41 / **42** |
| median `\|q_c\|`, iiwa | 4.99e-02 | 3.32e-02 | 1.98e-03 | 2.29e-04 | **2.09e-05** | -- |
| median max violation, iiwa | 2.65e-02 | 1.28e-02 | 8.52e-05 | 2.90e-05 | **2.61e-08** | -- |

At 480 cells the same instrumentation reproduces (medians, grasp paired): median `|q_c|`
7.48e-02 / 6.72e-02 (Panda / iiwa) without the penalty against **2.06e-05 / 5.39e-04** with it,
and median max violation 4.63e-05 → **3.44e-08** on the Panda, 9.04e-02 → **2.86e-04** on the
iiwa. Three of the four 60-cell rows are flat or worse at weight 30, so **10 is at or near the
optimum** rather than merely the largest value tried; read the per-row wobble as noise at 60
cells, with the 480-cell table above carrying the penalty's case.

**The other knobs keep their character too**, all swept and none worth revisiting: the
collision-shaping pair (`collision_influence_offset`, `collision_row_scale`) peaks weakly around
0.2-0.4 and is within noise of the default; `ipopt_mu_strategy=adaptive` is inert on both robots
(42 and 19) despite the archived logs looking like its textbook case; `latent_cost_weight` helps
the Panda (48 at 0.1) while hurting the iiwa (14), so not a general win; and `correction_bound`
swept upward (0.1 / 0.2 / 0.4 / 0.8) is inert — the box is not binding on either robot, and the
solver takes more of it when given more (median `|q_c|` 0.054 → 0.484) and gets nothing for it.

### Iterations, cost and wall clock: the three numbers a result is told in

**Standing reporting rule, Thomas's: every result is told in iteration count and in objective
cost, as well as in runtime.** Iterations are hardware-independent and describe the *formulation*;
seconds describe this implementation on this machine and are never compared across machines; cost
says what the solution is worth. Reporting only seconds makes the cap story look arbitrary; only
iterations hides that the learned arm's iteration is thirty times more expensive; only success
hides that on the grasp task its solutions cost roughly twice the baseline's.

Medians over each arm's succeeded cells at the 180 s cap, where the cap binds on nothing:

| experiment | start | learned (solved / iters / s / ms-per-iter) | joint space |
| --- | --- | --- | --- |
| Panda pose | native | 58 / 52 / 4.57 / **69.7** | 29 / 34 / 0.10 / 2.9 |
| Panda pose | paired | 41 / 101 / 7.26 / **72.3** | (same, both protocols) |
| Panda grasp | native | 58 / 187 / 15.00 / **83.8** | 55 / 48 / 0.14 / 2.8 |
| Panda grasp | paired | 55 / 215 / 18.12 / **84.2** | (same) |
| iiwa pose | native | 60 / 57 / 3.71 / **59.0** | 39 / 30 / 0.06 / 2.3 |
| iiwa pose | paired | 40 / 126 / 6.50 / **52.6** | (same) |
| iiwa grasp | native | 45 / 207 / 15.77 / **83.4** | 58 / 66 / 0.17 / 2.6 |
| iiwa grasp | paired | 53 / 213 / 19.07 / **82.2** | (same) |

The analytic arms sit between: 10-13 iterations at 6.5 ms on the Panda pose task, 98-113 at
8.0 ms on the grasp task.

**The per-iteration cost is the honest headline, and it is a factor of 25-30**: 53-84 ms against
joint space's 2.3-2.9 ms and the analytic arms' 6-8 ms, stable across robots, tasks and protocols
— as it must be, being one network Jacobian against Drake kinematics. Profiling says that gap is
CPU dispatch with a known ~3x floor, so it is an implementation property, not something tuning
removes.

**Iteration count is the formulation property, and it splits by task.** On the pose task the
learned arm wins on success while taking a comparable number of steps (52 against 34 on the Panda
under `native`, 57 against 30 on the iiwa) — it finds solutions the joint-space arm does not,
rather than grinding longer. On the grasp task it takes **3-4x** as many steps *and* pays 30x per
step; the two multiply to roughly 100x, which is the whole of the cap story and why 5 s is not a
measurement of the grasp task and 180 s barely is. So "the learned arm reaches parity on the Panda
grasp at 180 s" must always be stated as: parity in success at 55/60 each, at 215 median
iterations against 48, and 18.1 s against 0.14 s.

#### Cost, on the cells both arms solved

Two things had to be right first. **Costs are compared only on cells *both* arms solved** — a
median over each arm's own successes compares different cell sets, and the easy cells are exactly
the ones a weaker arm also solves, so that form flatters whichever arm fails more. And **the
learned-only regularizers are excluded from the reported objective** (`reported_cost`), so the
column measures the objective every formulation shares. 480 cells, 45 s:

| experiment | start | n both | learned | joint space |
| --- | --- | --- | --- | --- |
| Panda pose | native | 242 | **10.459** | 10.567 |
| Panda pose | paired | 182 | **10.383** | 10.643 |
| iiwa pose | native | 285 | **6.457** | 6.776 |
| iiwa pose | paired | 209 | **6.367** | 7.112 |
| Panda grasp | native | 417 | 5.322 | **2.826** |
| Panda grasp | paired | 398 | 4.858 | **2.687** |
| iiwa grasp | native | 216 | 5.952 | **2.647** |
| iiwa grasp | paired | 230 | 5.299 | **2.618** |

**On the pose task the learned arm wins on cost as well as on success**, on both robots and
under both protocols — modestly (1-10%) but with the same sign in all four rows. This is the
draft's central claim holding on the second of its two axes. **On the grasp task it loses on
cost by roughly a factor of two, in every row.**

#### What the correction penalty costs, and it is not nothing

The same comparison against the penalty-free arm, on the cells both solved:

| task | `w = 10` | `w = 0` | penalty costs |
| --- | --- | --- | --- |
| pose (4 rows, n = 206-460) | 6.364-9.893 | **6.061-9.459** | 0.5-5% |
| grasp (4 rows, n = 58-343) | 4.815-7.185 | **2.444-4.663** | 30-100% |

**The penalty is nearly free on the pose task (0.5-5%) and expensive on the grasp task
(30-100%)** — and the grasp task is exactly where it buys its cells. So the penalty is a
*trade*, not a free improvement, and must be reported as one: it converts objective value into
feasibility. The mechanism is the redundancy it was adopted to break — with `q_c` free the arm
can nudge `q` toward a well-centred configuration for nothing; pinning `q_c` to zero means `q`
is whatever the flow emits at `(c, z)`, which is less centred. The whole grasp cost gap against
joint space is this.

This is **not** a bookkeeping artefact of the penalty term appearing in the objective: at
`w = 10` the correction is driven to `|q_c|_inf ~ 1.8e-05`, so the term contributes about 2e-08
to a cost of ~5, six orders too small to explain the gap. An earlier caution in this file
claiming otherwise named the wrong mechanism; the columns are comparable, and the rise in cost
with the weight is a real change in which solutions the solver returns.

### The ablation ladder: the frame fix is the whole stack

Panda grasp, learned arm only, 60 cells, 20 s, paired, one grid for every rung (the finals'
grid), so the rungs are cell-comparable with each other and with the finals' learned column.

| rung | success | iters | at the cap | `\|z\|` at start | median `\|q_c\|` |
| --- | --- | --- | --- | --- | --- |
| baseline (uncalibrated frame, no sharing) | 11/60 | 135 | 41 | **426** | 0.090 |
| + conditioning-frame calibration | **29/60** | 126 | 31 | 2.81 | 0.086 |
| + shared flow evaluation | 30/60 | 134 | 31 | 2.81 | 0.085 |
| + latent trust region | 34/60 | 155 | 26 | 2.81 | 0.074 |

Exact McNemar, each rung against the one below: frame calibration 26/8, **p = 0.0029**; shared
evaluation 1/0, p = 1.0; latent trust region 15/11, p = 0.56; the whole stack 28/5,
**p = 6.6e-5**. The median start error is 0 (exact) at every rung.

**This is the first ladder in the repo that measures what it claims to** — earlier ladders were
confounded by the latent bounding box (which silently projected every start) and by two rungs
running the unauthorized task parameterisation. With the box a general constraint and the start
exact at every rung, the attribution is clean and it is almost entirely one change:

- **The conditioning-frame calibration is worth 18 cells and is the only significant rung.** Its
  mechanism is the `|z|` column: uncalibrated, `SetStartFromQ` inverts the flow at a pose 27 mm
  and 120 degrees from the trained frame, and the network answers with a latent of norm **426** —
  and, the latent region now being a constraint, the solver actually *starts* there instead of
  being quietly clipped. The old ladders could not see this because the clip hid it.
- **Sharing the flow evaluation is worth one cell**, as it must be: bit-identical values and
  derivatives, so its only effect is throughput inside a fixed cap.
- **The latent trust region (`latent_trust_region`) is +4 cells and not significant.** It stays for the reason recorded
  separately — IPOPT is poorly behaved on unbounded variables, and a nonbinding constraint still
  shapes an interior-point trajectory — and remains a stated deviation from eq. (6) rather than
  a proven improvement.

### The analytic chart: eight branches, and what the last 0.6% is

The closed-form map's discrete set is three binary choices — wrist (B), shoulder (C), elbow (A)
— and the implementation historically charted only A = +1, the half away from the joint limits,
following the Panda analytic IK paper. The missing half is a *single sign*: negate both triangle
angles `O2O4O6` and `O2O6O4` (the elbow reflected across the shoulder-wrist axis). The old
commented-out "Case A1" line matches no configuration. The measured elbow relations are
`q3 = theta + q3_add - 2*pi` (A = +1) and `-theta + q3_add` (A = -1), partitioning at
`q3 = q3_add - pi = -0.467`.

`ProgramOptions.analytic_branches` selects the chart (default 4, so archived runs stay
reproducible). `gc(q, branches=3)` recovers all three indices with zero mislabels in 4000
samples. Round-trip coverage of `IK(FK(q), psi(q), gc(q)) == q`, 4000 random configurations:
89.4% (4 branches) against 99.40% (8 branches) at 1e-6, rising to 99.83% at 1e-2.

**The residual is not singularities** (measure zero; this set has positive measure) **and not
branch mislabelling** (the 24/4000 misses are reproduced by *no* branch of the eight). Two are
off by ~4 rad — a genuinely distinct solution — and 22 by 1e-3 to 1e-2, clustered where the
wrist arcsin argument approaches 1, i.e. near a branch-merge locus. Consistent with the <=16
self-motion-manifold bound (Burdick/Luck). **Left as future work by decision** — arXiv:2503.03992
is the suggested starting point — and until then coverage is reported as the curve above, never
as "100% up to singularities". Note the iiwa needs no such column: its Faria/SRS implementation
already charts all eight branches, so analytic4-vs-analytic8 is a **Panda-only** experiment.

**A separate grid confirms the whole `analytic4` disadvantage is start coverage.** Drawing
`q_init` by rejection so it falls only in the four wide bundles the 4-branch chart covers (applied
once to the shared guess list, so pairing is preserved — but it changes the cells, so that table is
*not* cell-comparable with the finals), `analytic4` and `analytic8` become identical: 59/60 and
59/60 on the grasp task, 34/60 and 34/60 on the pose task, with the same mean iteration counts and
the same `start_q_error`. They must be, since every `q_init` then lies in a bundle both charts
cover. **Nothing about the near-limit bundles makes the *solve* harder; they are simply
configurations that arm cannot be given.**

## The one open question: the residual failures are the flow's own gain

This is the project's central scientific finding and the explanation of the iiwa grasp deficit.

**The violated binding is `AllIKFlowConstraints`, and inside it the joint-limits row.** On the cells that never converge,
`max_violation` equals `|q|_inf` exactly (Spearman 1.0, agreeing to the digit, on 55 of the 64
badly-violating learned cells across all eight 180 s runs). The returned configurations have
joint angles of **1e7 to 1e16 radians**. Everything else about those cells follows: a
configuration of 1e8 rad puts the gripper anywhere, so "deeply in collision, a metre off
target" is a *consequence* of the blow-up, and the collision penalty is a bystander
(`max_violation` does not track `collision_value` at all, Spearman -0.15).

**Every runaway lies on one ray, and it is a property of the network, not of the solve.**
Normalising the exploded `q` vectors and taking pairwise `|cos|`:

| | ray (unit, joint order) | pairwise `\|cos\|` |
| --- | --- | --- |
| iiwa (mug native, mug paired, pose paired) | `[0.001, -0.001, 0.016, -0.000, 0.003, 0.978, -0.208]` | **1.0000** |
| Panda (mug native, mug paired, pose paired) | `[-0.016, 0.033, 0.998, -0.032, -0.002, 0.027, 0.023]` | **1.0000** |

The same ray on both tasks and under both start protocols, dominated by a single joint — the
iiwa's wrist (joint 6) and the Panda's elbow (joint 3).

**Sampling the network directly reproduces it, with no Drake and no solver involved.** Draw `c`
position uniformly in its ±0.25 m box, a uniform unit quaternion, and `z` uniformly in the ball
of radius 4.3 — strictly inside the region the formulation allows — and evaluate
`MakeFlowInference` in float64:

| | Panda `lp191_5.25m` | iiwa14 `lemon-haze-7` |
| --- | --- | --- |
| median `\|q\|_inf` | 2.65 | 2.50 |
| p99 | 3.71 | 8.7e+05 |
| p100 of 20000 | 4.1e+12 | 5.5e+16 |
| fraction `> 3` rad (outside joint limits) | 0.159 | 0.142 |
| **fraction `> 1000` rad** | **0.00065** | **0.0334** |
| ray recovered from those samples | `[-0.017, 0.035, 0.998, ...]` | `[0.008, 0.030, 0.017, ..., 0.984, -0.167]` |
| `\|cos\|` against the ray the *solver* landed on | **0.9999** | **0.9976** |

The distribution is bimodal, not heavy-tailed: on the Panda, 14 of 20000 exceed 10 rad and 13 of
those exceed 1000. A draw is either an ordinary configuration or it is astronomical.

**This is the answer to the iiwa grasp deficit.** The iiwa checkpoint puts **3.34% of the
allowed region** into the blow-up regime against the Panda's **0.065% — a factor of 51**.

**The mechanism is architectural headroom, not a numerical bug.** FrEIA's coupling blocks
soft-clamp the log-scale to `clamp * 0.636 * atan(s/clamp)`, bounded by `clamp * 0.636 * pi/2`,
so with `rnvp_clamp = 2.5` over `nb_nodes = 12` the worst-case output gain is about
`e^(2.5*12) ~ 1e13` — exactly the scale of the observed maxima. These are near-worst-case gain
regions of a bounded map, not poles. Both checkpoints have the same headroom; they differ only
in how much of the conditioning domain sits near it.

**`rnvp_clamp = 2.5` is confirmed correct**, worth checking because `src/iiwa_program.py`
hardcodes the iiwa's hyperparameters and `rnvp_clamp` changes the forward pass without changing
any parameter shape, so a wrong value would load silently. Sweeping it against the same weights,
fraction `> 1000`: 0.893 / 0.410 / 0.090 / **0.035** / 0.126 / 0.999 at clamp 1.0 / 1.5 / 2.0 /
**2.5** / 3.0 / 5.0. A clear optimum, so the 3.34% is a property of the weights.

**Why the solver finds a 3%-measure set 30% of the time.** It does not sample; it follows
gradients, and `dq/dvars` in these regions is as large as `q` is. A Newton step is *attracted*
to them. That is also why more budget never helps: the cap-bound cells at 180 s are the same
cells that were cap-bound at 20 s, and on iiwa pose paired the set is frozen at 19 cells across
every cap tested.

**Note the learned arm does control `q`** — Thomas: *"the network does get to control the joint
limits a bit, since it can adjust z. That's the whole point of differentiating through the
network — we take the constraint gradient for joint limits and pull it back through the network
to z."* An earlier claim in this file that it imposes limits on something it cannot control was
wrong. What differs from the baselines is *when* the limits hold (only at convergence), and that
the gradient into a high-gain region is itself enormous, so the Newton step is attracted rather
than repelled.

### Neither region knob avoids them, because they are not at the edges

Fraction of the region with `|q|_inf > 1000`:

| knob | values | iiwa | Panda |
| --- | --- | --- | --- |
| latent trust-region radius | 1.0 / 2.0 / 3.0 / 4.3 / 6.0 / 8.0 | 0.033 / 0.026 / 0.032 / 0.035 / 0.040 / 0.071 | 0.0018 flat |
| `c_position_slack` | 0.05 / 0.10 / 0.25 (default) / 0.50 | 0.039 / 0.041 / 0.035 / **0.225** | 0.000 / 0.000 / 0.0018 / **0.137** |

Shrinking the trust region to `R = 1` leaves the iiwa's exposure unchanged at 3.3%: the blow-up
regions are spread through the domain, including at `|z| <= 1`. **This is why the trust-region
sweep measured inert** — it was never able to exclude them. The `c` box is flat from 0.05 to
0.25 and then a **cliff** at 0.5, where exposure jumps 6x on the iiwa and 76x on the Panda; the
default sits just under it, which is luck rather than design, and worth knowing before anyone
widens it.

### Every optimization-side remedy has been measured and refuted

Thomas's ranking of the candidates: **(1) a better chart is preferred over everything else** —
*"All of these actions are less preferred than just getting a better iiwa chart"*; (2) lifting
`q` into a bounded decision variable, permitted but disliked (*"we're effectively adding a
nonlinear equality constraint"*); (3) a joint-limit penalty, permitted but disliked (*"we should
be able to rely on the constraint to handle it"*). "You can try it, but I don't like it" means
measure it and report it as a stated deviation, not adopt it if the numbers look good.

**Chart accuracy is not the mechanism.** `chart_error_scale = eps` adds a deterministic, smooth,
seeded perturbation `eps * sin(W [c; z] + b)` to the flow's output, degrading the chart while
holding the scene, kinematics, solver, grid and start protocol fixed. Panda grasp, 60 cells,
20 s, paired:

| `eps` (rad) | nominal median chart error | success | at the cap |
| --- | --- | --- | --- |
| 0 (the Panda flow as trained) | 3.8 mm | 35/60 | 25 |
| 0.016 | ~12 mm | 34/60 | 26 |
| 0.032 | ~20 mm | 32/60 | 30 |
| 0.064 | ~43 mm | 22/60 | 40 |
| 0.128 | ~83 mm | 1/60 | 1 |

The iiwa's measured chart is 16.6 mm median / 64.5 mm p90 against the Panda's 3.8 / 9.4, so it
sits between `eps = 0.016` and `0.032`, where the Panda still solves **34/60 and 32/60**. The
iiwa solves 12/60. **Degrading the Panda's chart to the iiwa's accuracy costs it one to three
cells; the iiwa is twenty-three cells worse** — the standing chart-accuracy hypothesis does not
survive its own experiment. (The `eps = 0.128` row measures something else: 58 of its 60 cells
fail as `unrepresentable_start`, 0.128 rad per joint exceeding what the ±0.1 correction can
absorb. That row describes the correction box, not the dose curve.) Note also why this experiment
could never have reproduced the real pathology: smooth `sin` error degrades accuracy while adding
no high-gain regions.

**IPOPT's scaling is not the lever.** `nlp_scaling_method=none` and `nlp_scaling_max_gradient`
at 1e4 and 1e8, five experiments x 60 cells: every variant inert, largest movement ±2 cells, no
comparison reaching p < 0.5, and the three settings reproducing each other almost cell for cell
(which is what raising the cap far enough should do). So the runaway is not a scaling artefact —
IPOPT is not mis-scaling a row it could have handled, it is being handed a chart with gain ~1e13
and following the gradient into it. (`equilibration-based` is unavailable in Drake's IPOPT, needing
HSL MC19; it raised `RuntimeError` on construction and scored 0/60 in ~10 ms a cell — a crash, not
a measurement.)

**The joint-limit penalty is inert**, which vindicates Thomas's objection to it. Across twelve
measurements (`joint_limit_penalty_weight` at 1, 10, 100 on four rows) the smallest p against
the default is 0.115, no weight has a consistent direction on either robot, the runaway counts
do not move, and the median max violation is unchanged at ~1e-08. Adding a penalty on a quantity
a constraint row already governs buys nothing. The knob stays in the tree, off.

**Lifting `q` is net negative, and it is a task effect.** `lift_q` adds `q` as a decision
variable whose bounding box is the joint limits and imposes the chart as a 7-row equality. All
eight experiments, 60 cells each, against the default:

| experiment | start | default | `liftq` | b/w | p | iters (def → lift) | runaway cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda grasp | paired | 50 | **58** | 9/1 | **0.022** | 188 → 136 | 4 → 0 |
| grasp, other 3 rows | | 39-57 | 43-59 | | 0.33-0.82 | 170-201 → 114-209 | 1-14 → 0 |
| Panda pose | native | 58 | 52 | 2/8 | 0.11 | 52 → 44 | 0 → 0 |
| iiwa pose | native | 60 | 44 | 0/16 | **3.1e-05** | 57 → 30 | 0 → 0 |
| Panda pose | paired | 40 | **9** | 4/35 | **3.3e-07** | 100 → 161 | 13 → 0 |
| iiwa pose | paired | 40 | **2** | 0/38 | **7.3e-12** | 126 → 317 | 18 → 0 |
| **all eight** | | | | **38 / 116** | **2.2e-10** | | |

**It delivers exactly one thing, universally: 0 runaway cells in all eight experiments** — the
returned configuration is inside the joint limits by construction, always (max `|q_lift|` 3.05 rad
on the iiwa against its 3.054 limit). **But the runaway does not stop, it relocates.** The flow
still reaches 1.19e10, and since the chart is now an equality row that lands in `max_violation`
instead of in `q` — median violation **3e+06 to 7e+06** on the collapsing pose rows. The cell
fails either way; only the row it fails in changes.

The split is by **task**, not robot: neutral-to-positive on the grasp task with consistently
*fewer* iterations, negative on all four pose rows and catastrophic under `paired`. The mechanism
is in the violation column — the pose task already pins the end-effector with six equality rows,
so lifting adds seven more, giving thirteen equalities in 27 variables with the flow's badly
scaled Jacobian inside seven of them, where the grasp task's rows are mostly inequalities with
only the two mug-axis equalities. **An interior-point method's tolerance for a badly scaled row
depends on how many equalities it is already carrying** — Thomas's objection with a mechanism
attached. An arm scoring 2/60 and 9/60 on two of eight experiments is disqualified whatever it
does elsewhere, so the 480-cell replication was **deliberately not run**: the negative result is
complete at 60 cells.

**Jacobian regularization is a clear negative and the line of investigation is closed.**
`regularize_jacobian()` in `src/generic_program.py` implements three strategies acting on the
Jacobian before the chain rule, so the solver sees a damped gradient while `q` is unchanged:
Frobenius norm clipping (`jacobian_max_norm`), Tikhonov/LM damping of the singular values
(`s_damped = s * λ / (s + λ)` — the correct shape for the runaway, since large singular values
are damped more than small ones), and a singular-value floor (`jacobian_svd_floor`). Ten variants
x eight experiments x 60 cells:
| variant | success / 480 | delta | p | runaway cells / 480 |
| --- | --- | --- | --- | --- |
| *(default)* | 389 | | | 56 |
| `jacobian_max_norm=1000` (the best) | 395 | +6 | 0.59 | 48 |
| the six middling settings | 362-382 | -7 to -27 | 0.022-0.53 | 35-64 |
| `tikhonov=10` / `max_norm=10` | 232 / 202 | -157 / -187 | 1e-27 / 4e-36 | 42 / 77 |
| `tikhonov=1` / `=0.1` (most aggressive) | 2 / **0** | -387 / -389 | 6e-115 / 2e-117 | 106 / 142 |

Aggregated over every row and variant: **310 cells better against 1,509 worse.** Not one setting
is a significant improvement, and the only one not net-worse is indistinguishable from no
regularization and does not reduce the runaway it was introduced to prevent.

**Why it cannot work, which is the part to remember.** The flow's Jacobian is the *exact*
derivative of an explicit function. Where the gain approaches its architectural ceiling of ~1e13,
a sensitivity of 1e13 is the correct answer, not an artifact to be regularized away. Damping it
does not regularize the problem — it breaks the correspondence between the constraint values
IPOPT evaluates and the gradients it is handed, leaving an inconsistent nonlinear program. That
is why the *most* aggressive damping fails hardest while *increasing* the runaway count: with the
gradient scaled to nothing, the solver has neither the signal that would carry it into a
high-gain region nor the one that would carry it out. LM damping is sound applied to the **Newton
step** rather than to a reported derivative, but Drake's IPOPT does not expose the step
computation, so the well-posed version is unreachable here and the reachable one is refuted.

**Thomas's ruling: *"I think we can conclude gradient regularization and the other strategies
isn't worth it."*** The knobs stay in the tree, off, with this table as the reason not to revisit
them. (One pattern, recorded as post-hoc because it was chosen after seeing the data and is not
significant: on the two runaway-heavy rows moderate damping is positive, pooled 27/15, p = 0.088,
runaway cells 31 → 11. Note `tikhonov=10` cuts the runaway hardest there, 31 → 7, while scoring
exactly 19/19 on success — **suppressing the runaway does not buy success**, the same conclusion
lifting `q` reached.)

**So a better iiwa chart is the preferred remedy and now the only untried one.**

## Running on MIT SuperCloud (`cluster/`)

`cluster/README.md` is the playbook and `~/.claude/skills/supercloud/SKILL.md` carries the
standing rules; what follows is what a reader of *this* file needs. The allocation is **4 nodes
on `xeon-g6-volta`**, each 40 Xeon Gold 6248 cores and 2x V100 32 GB.

**Timing is never compared across machines.** Thomas: *"There's never a need to compare
wall-clock (or really, performance in general) between laptop and cluster. But wall clock limits
can be adjusted on the cluster."* So the wall-clock cap stays as the measurement — no switch to
iteration caps for portability's sake — but its value is chosen from `cluster/calibrate.sh` on
that hardware. `metadata.host` and `metadata.device` exist so a cluster run cannot be paired
cell-for-cell against a laptop one. The corollary is that **CPU contention still corrupts the
measurement**, so how many worker processes may share a node is a measured quantity.

**`--shard K/N` is the sharding primitive**, and it is a no-op by construction. It splits
**target-major** — whole targets per shard, never a target's guesses split — because `success_ci`
bootstraps over whole targets and `solved_within_k` counts restarts within one, and it appends
`_shardKofN` to the tag (without which two shards resolve to the same `summary.json` and
overwrite each other). `cluster/merge_shard_summaries.py` pools the records and **re-runs
`summarise`** rather than stitching per-shard numbers, preserving arm order so `_mcnemar`'s pair
directions survive. `bash cluster/verify_sharding.sh` proves the round trip locally in ~2 minutes,
bounding solves with `max_iter` rather than the wall clock deliberately. **Run it after any change
to sharding, the merger, or grid construction.**

**Two cluster facts that shaped the design.** The account's `xeon-g6-volta` limit is a Slurm
**`GrpTRES` group** cap (`node=4`, `MaxSubmit=240`), not a per-job `MaxNodes`: work beyond it is
accepted and **queued**, so a whole stage is submitted at once and Slurm meters it — and the cap
is shared with everything else the account runs. Because jobs therefore start at different times
there is no stable rank space to deal work into, so `cluster/run_items.sh` claims items with an
atomic `mkdir <id>.claim`. And **PyTorch 2.11's cu128 wheels dropped sm_70**, so the V100s need a
cu126 build; the wrong wheel imports cleanly, reports a CUDA device, and fails only at the first
kernel launch, which is why `cluster/smoke.sh` launches a real kernel rather than trusting
`get_arch_list()`. Three smaller adaptations: Meshcat is optional (`BuildEnv(meshcat=None)`),
mug scenes are built only for the shard's targets, and `hit_iteration_cap` is the counterpart to
`timed_out` (`is_timeout` does not match IPOPT's "Maximum Number of Iterations Exceeded", so a
`--set max_iter` run reported `timeouts: 0`).

### The calibration (`xeon-g6-volta`, V100)

Four arms, one per node, each a full job on a real partition. The workload is the **Panda grasp**
task, learned arm only, 4 targets x 2 guesses, `--compile` — that task specifically, because it is
the one that binds against the cap. A first attempt ran the *pose* task and measured nothing: a
pose cell converges in ~74 iterations and ~6 s here, so its iteration count is identical at every
cap and however contended the node is. **A converged solve takes the iterations it takes**; only
its wall time moves. Both sweeps were structurally incapable of showing an effect, whatever the
truth.

**Workers per node.** Median iterations achieved inside a fixed 20 s cap, on the GPU:

| workers | 1 | 2 | 4 | 8 | 20 | 40 |
| --- | --- | --- | --- | --- | --- | --- |
| median iters | 202 | 196 | **194** | 186 | 114 | 70 |
| vs P=1 | 1.00x | 0.97x | **0.96x** | 0.92x | 0.56x | 0.35x |

**`PROCS=4` is the conservative setting and `PROCS=8` is what exploratory stages ran at** — four
workers cost 4% of the per-cell iteration count where 8 costs 8% and 20 costs 44%. Thomas ruled
the 8% acceptable for sweeps, reserving instrumented uncontended runs for hard comparisons and
paper numbers. The P=1 row is the *noisiest* (one worker's median against forty at P=40), so
0.96-0.97x at P=2 and P=4 is within noise of unity while the collapse at P>=20 plainly is not.
Since the benchmark is wall-clock capped, a worker that gets less done is a **different
measurement**, not merely a slower one — so the worker count is held fixed across everything being
compared.

**CPU-only is not competitive and the campaign runs on the GPU**: at one worker the GPU reaches
202 median iterations against 62, solving 4 of 8 cells against 1 of 8, and CPU-only degrades more
gracefully under contention (0.74x at P=40 against 0.35x) only from a starting point 3.3x worse.
This does not contradict the profiling result that the flow is CPU-bound at batch 1 — that says
the GPU is never the bottleneck *while a GPU is present*, not that torch on CPU is as fast.

**The cap.** Single worker, 8 cells: median iterations 142 / 203 / **338** / 338 / 338 at 10 / 20
/ 45 / 90 / 180 s, with the median cell finishing at ~22 s and 3 / 4 / 5 / 5 / 6 feasible.
**45 s is the campaign's cap**: medians saturate by 45 s and do not move at 90 or 180, so the
median cell has converged with 2x headroom. Beyond that only the tail gains, which is what the
cap curve is for rather than something to buy with a bigger default.

**Staging and startup**: 40 concurrent `import torch, pydrake, ikflow, jrl` take **10 s** total,
so Lustre read amplification is not a problem and the venv can stay on the shared filesystem —
copying it to node-local `$TMPDIR` costs 231 s against Drake's 13 s and buys nothing.
`torch.compile` of the flow Jacobian costs ~35 s cold and ~17 s warm per process.

### Solver logs: node-local, and one archive per run

`src/benchmark.py` once wrote one ~20 KB IPOPT log per (cell x arm) straight onto the shared
filesystem — **35,596 of them, 87% of every collection's file count**, exactly the many-small-files
pattern SuperCloud's guidance warns against. Lustre is metadata-op bound on files that size, so a
routine collection had drifted from three minutes to thirty and worsened with every stage. Now
per-cell logs go to node-local `$TMPDIR` (keyed on the run tag *and* the pid, since a node runs
eight workers) and are rolled into one `solver_logs.tar.gz` per run at the end of `run_grid` —
3.7x compression, every log still recoverable with `tar xf`. On a laptop run they stay in `log_dir`
and are rolled up in place. After the fix: 40,733 files → 2,964, 950 MB → 316 MB, ~30 min →
**43 s**. `collect_results.sh` is incremental by default (`--full` forces the old behaviour) and
never ships `state/`, whose done markers are load-bearing on the cluster and never read locally;
`cluster/compact_logs.sh` backfills pre-change runs from inside a debug-cpu job.

**A guard bug this surfaced, worth remembering.** Both scripts refused while *any* job was running,
via `LLstat | grep -c RUNNI`. SuperCloud accounts are **shared across Thomas's projects**, so that
fired on an unrelated campaign's job. Both guards now filter by this project's own job name and
count `PENDING` as well as `RUNNING`. **Any cluster-wide check on a shared account must be scoped
to this project's own jobs.**

### The laptop suspends when idle

Every multi-hour stall this repo recorded — the archived 6106 s cell and four overnight "wedges"
— was **the machine going to sleep**. GNOME suspends after 900 s idle *even on AC*
(`sleep-inactive-ac-type='suspend'`), and `journalctl` matches every stall to the minute. Three
wrong solver-level theories (SPRAL, GPU runtime-D3, a torch spin) each fit part of the evidence:
it struck only *unattended* runs (16/16 attended reproductions ran clean), `timeout`/`sleep` run
on CLOCK_MONOTONIC which pauses across suspend, and a CUDA context straddling a suspend leaves
torch spinning at 100% CPU afterwards.

**Any long unattended run on this machine must hold a sleep inhibitor**:
`systemd-inhibit --what=sleep:idle --mode=block sleep infinity &`, launched with `setsid` so a
session teardown cannot take it down with the queue. When an unattended process appears hung,
check `journalctl -b | grep "suspend now"` against the stall window *before* any solver- or
GPU-level theory. Long benchmarks now run on the cluster instead.

**Reproducibility at the cap is ±1 cell.** Two runs of the same configuration on the same grid
scored 34/60 and 35/60; the single differing cell hit the wall clock in both, reaching 264
iterations in one and 286 in the other. Cells that exit at the cap are reproducible only up to
machine load — worth remembering before reading a one-cell difference anywhere as a real effect.

## Next steps

**Thomas's roadmap, in priority order (2026-09-04)**, given once the corrected campaign finished:
*"iiwa checkpoint training infra (and launching the multi-day training job), SNOPT and NLOPT, and
then performance tuning and formulation tweaks for getting the best results with the learned
formulation."*

1. **Retrain the iiwa chart.** Build the training infrastructure, then launch the multi-day run.
   This is the project's one open scientific question and a *training* task — new territory for a
   repo that has only ever run benchmarks. The diagnosis it answers is above: `lemon-haze-7` puts
   3.34% of the conditioning domain into the flow's worst-case-gain regime against the Panda's
   0.065%, and that factor of fifty-one is the whole of the iiwa grasp deficit. Every
   optimization-side remedy has been measured and refuted. Establish the checkpoint's provenance
   with Julia first, and **plan the run around SuperCloud's monthly maintenance** — second Tuesday,
   compute down Monday evening to Wednesday morning, nothing survives it.

2. **SNOPT and NLOPT.** All ~395 archived runs are IPOPT; both scripts already accept `--solver`
   and Drake supplies all three on both machines. The point is to *report* every solver, not to
   pick one. **Known blocker, found by a smoke test and not fixed:** SNOPT runs, but `parse_log`
   matches only IPOPT's log format, so `iterations`, the evaluation counts, `solver_seconds` and
   `exit` all come back `None`. Iterations is the hardware-independent number, so the parser needs
   per-solver formats before a solver comparison means anything.

3. **Performance tuning and formulation tweaks.** Note this is Thomas naming formulation work as a
   work item, not a standing licence — what is compared remains his call, made explicitly in
   advance.

### Smaller open items

- **More guesses per target** in the paired grid — same guesses for every arm — reported as
  "solved within k restarts". The only honest form of multi-start, and the harness already does
  it.
- **Non-dimensionalise the conditioning pose's translation against its rotation**, the way
  `eaik-experiment` scales its Jacobian rows by a 1.12 m length scale, so the `c` block is
  dimensionally coherent. Never tried.
- **A `q_c == 0` arm** (the draft's eq. 4) would quantify what the correction buys, now that the
  penalty has established the `c`/`q_c` redundancy is real.
- **The analytic chart's residual 0.6%** — left as future work by decision; arXiv:2503.03992 is
  the suggested starting point.
- **Least-squares domain extension**, from Thomas's unreleased IFT-IK paper, is deferred but
  belongs to *this* project. (Trust-region solving belongs to a different project and is out of
  scope here entirely.)
- `stage_H` in `cluster/gen_manifest.py` is a generic cross-test harness, built for a regularization
  cross that died with Stage G. Kept for whatever knob next needs one.
