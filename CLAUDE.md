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

### Checkpoints carry their architecture (`src/flow_loading.py`)

A `.pkl` is a bare `nn_model` state dict — `IKFlowSolver.load_state_dict` is a plain
`pickle.load` — so **weights travel without their architecture**. It used to be hardcoded at
four call sites, all asserting `nb_nodes = 12`, `coeff_fn_internal_size = 1024`,
`rnvp_clamp = 2.5`. `nb_nodes` and `rnvp_clamp` change the forward pass **without changing
any parameter shape**, so a mismatch loads cleanly and is silently a different chart.

Each checkpoint now has a sidecar, `<name>.arch.json`, written by
`scripts/training/export_ckpt_to_pkl.py` from the training checkpoint's own
`hyper_parameters.base_hparams`. `LoadFlowSolver(robot, checkpoint)` reads it, builds the
solver, and cross-checks it against the weights: `nb_nodes` (module-list length / 2),
`coeff_fn_config`, `coeff_fn_internal_size` and the network width are all recoverable from
state-dict shapes and are verified; a sidecar contradicting them **raises**. **`rnvp_clamp`
is the one field no check can catch** — a scalar used in the forward pass and stored nowhere
— which is why the sidecar is mandatory for new checkpoints rather than merely convenient.
The resolved architecture is attached as `solver.arch` so runs record what they loaded
rather than what they asked for. A checkpoint with no sidecar falls back to the legacy
architecture with a `RuntimeWarning`; sidecars are backfilled for all three checkpoints on
disk (`scripts/training/backfill_arch_sidecars.py`), so nothing depends on that path.

**Hold `dim_latent_space` at each robot's baseline** — iiwa14 8, Panda 7. The latent width
*is* the program's decision-variable count (21 for the iiwa, 20 for the Panda), so varying it
changes the optimization problem rather than the chart; holding it also means a checkpoint
loaded against the wrong robot fails the shape check instead of loading silently.

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

### The hardened scene and shelf-contained targets (2026-09-15)

Sampling a collision-free `q` and accepting the gripper pose *wherever it landed* put targets
in free air far more often than in clutter, so the collision-avoidance half of the problem
barely bound. `../codebase` hit the same weakness in its Grasp Selection experiment and
hardened it (`38f4eac`); this is the same treatment, on the same shelves, at the same inset.

**The scene.** Three new YAMLs — `models/panda/panda_finray_collision_hardened.yaml`,
`models/panda/panda_collision_hardened.yaml`, `models/iiwa14/iiwa14_collision_hardened.yaml`
— are their legacy twins minus `binF` and (on the two scenes that had them) the seven welded
decorative mugs. Four shelves and two tables remain. The legacy files are untouched and stay
the default for the older per-experiment scripts, so archived runs still reproduce.

**The target must land in a shelf.** `src/shelf_regions.py` holds the twelve compartments —
four units x three bays — each as its **local** box plus the weld's translation and yaw, and
`PointInShelfCompartments` rotates the query point into the region's own frame. A world-frame
AABB is not usable: at these 135°/235° welds it over-approximates the footprint about **4x**
and would accept targets in free air beside the unit. The inset is symmetric because
`shelves.sdf` has **no back wall** — the unit is a tunnel and which `x` face is "front" depends
on the yaw. `shelf_depth_inset = 0.10 m` is adopted, matching `../codebase`.

**And the object must fit.** Containment alone is not enough — a mug centred in a compartment
can still intersect a board. **Drake never generates collision candidates between two
ANCHORED geometries**, and `GenerateDiagramWithMug` *welds* the target mug, so that overlap is
silently invisible on the solve scene. `FloatingMugScreen` therefore runs on its own diagram
with the mug appended **unwelded**. Its robot filter is exact model-instance names, and a
filter that matched nothing would reject every candidate as "penetrating" — indistinguishable
from a too-deep inset — so `tests/test_shelf_placement_screens.py` pins it.

**Acceptance is ~0.2-0.7%, an order of magnitude below the sibling's**, measured by
`scripts/probe_shelf_acceptance.py` (20000 draws per scene), as a fraction of collision-free
draws:

| inset | 0.0 | 0.05 | **0.10** | 0.125 |
| --- | --- | --- | --- | --- |
| panda grasp (`between_fingers`) | 2.000% | 1.215% | **0.675%** | 0.319% |
| panda pose (`panda_hand`) | 1.320% | 0.764% | **0.371%** | 0.175% |
| iiwa grasp (`between_fingers`) | 1.895% | 1.236% | **0.552%** | 0.216% |
| iiwa pose (`iiwa_link_7`) | 0.972% | 0.540% | **0.228%** | 0.132% |

That costs ~400-1100 draws per target, about **12 s of sampling per grid** at 0.19-0.33 ms a
draw — negligible. What it *does* break is the sibling's rejection guard. Acceptance restarts
at every accepted target, so the guard is a per-target tail bound and a 60-target grid gets 60
chances to trip; at `../codebase`'s 5000 the **iiwa pose row trips 41% of runs**, and a trip
raises partway through a queued run and kills every shard of it. **`MAX_CONSECUTIVE_REJECTIONS
= 50000`**, where every row is zero to machine precision. It bounds only the tail, so it costs
nothing and no manifest has to remember to raise it. **0.125 is not fielded**: 1818 draws per
target on iiwa pose, and 98% of runs trip at the sibling's guard.

**Two things to know before reading a hardened number.** The pose task's containment point is
the frame its target *is* — `iiwa_link_7` or `panda_hand`, i.e. **the wrist, not the
fingertips** — so "pose target in a compartment" is a harder and differently-hard condition on
each robot. It is recorded per run as `placement_point`. And the panda *grasp* scene never had
decorative mugs, so hardening removes strictly less from it than from the other two; say so
wherever grasp and pose deltas appear in one table.

**Guesses are deliberately not containment-filtered.** They are initial configurations, not
targets; filtering them would couple the start distribution to the target distribution and
change what "given a random collision-free start" means.

`--scene legacy --target-placement free` reproduces the pre-hardening sampler exactly —
`SampleShelfTargets` consumes one draw per iteration with both screens off, and a test pins
that — so an archived grid is still re-runnable. `c_position_slack` is **not** touched: a
±0.25 m conditioning box around a mug in a 0.10 m-deep compartment is now loose relative to
the free space and plausibly hurts the learned arm specifically, but sweeping it here would
confound hardening with slack.

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

## The solver axis: interior point, SQP, augmented Lagrangian (2026-09-16)

**Three METHOD CLASSES, not three vendors.** Thomas: *"for NLOPT, we want to use it as an
augmented lagrangian solver. This is important! We don't care about SLSQP, since SNOPT is
already SQP. The point is that we test interior point, augmented lagrangian, and SQP."* So
`--solver` is `ipopt` (interior point), `snopt` (SQP), `nlopt` (**`LD_AUGLAG`**). `LD_SLSQP`
is the wrong NLopt default -- it is an SQP method and would leave the comparison with two SQP
columns and no AL column. `LD_AUGLAG` rather than `LD_AUGLAG_EQ` because `_EQ` absorbs only
*equality* constraints into the AL and leaves inequalities to the inner solver, and this
program carries both. Any solver added later must be justified by the class it contributes.

All three take `kGenericConstraint`/`kGenericCost`/`kCallback`, and all three call
`EvalVisualizationCallbacks` **inside their objective evaluation** -- so `last_iterate`
recovery, which every abnormal exit depends on, works unchanged under each. Nothing about the
watchdog design needed revisiting.

### Each solver converges at its own defaults

The SNOPT branch used to read IPOPT's `acceptable_tol` and `acceptable_constr_viol_tol` as its
`Major optimality`/`Major feasibility tolerance`. That is rung 2 of the tolerance ladder done
wrong: IPOPT's `acceptable_*` family is its **relaxed early-stop** criterion, not what it
converges to (its real `tol` is 1e-8), so SNOPT was being asked for 1e-3 while the solver it
is compared against drove to 1e-8. **Do not transplant one solver's option values onto
another.** Unset `snopt_*`/`nlopt_*` fields are simply not passed, so each solver sits at its
own defaults and the shared, deliberately looser task gate decides success. No archived SNOPT
run existed, so nothing was invalidated.

### What each solver will and will not tell you

**No solver reports an iteration count through Drake.** `SnoptSolverDetails` carries `info`,
`solve_time` and the multipliers but no count; `NloptSolverDetails` carries a single `status`
field. So iteration counts come from the print file, and NLopt has none at all.

| | IPOPT | SNOPT | NLopt |
| --- | --- | --- | --- |
| print file | yes | yes | **none, and `kPrintFileName` is silently ignored** |
| `iterations` | "Number of Iterations" | "No. of major iterations" | -- |
| eval counts | 4 separate counts | one `User function calls (total)` | -- |
| seconds | log | `details.solve_time` | -- |
| status | -- | `details.info` | `details.status` |

`iterations` means **majors** under both solvers that report it, because IPOPT's count is
majors -- this column is only comparable if it means the same thing in both. SNOPT's minors
are kept separately. snOptA has one user function, so there is no funobj/funcon split and
IPOPT's four eval counts have no SNOPT counterpart; they stay `None` rather than being filled
with a different quantity.

**So the program counts map evaluations itself** (`IKFlowProgram.ResetEvalCounts`, counted in
`QAndPose`, the one funnel every arm's solve passes through). `map_jacobian` is the
AutoDiffXd count -- one `jacrev` through the network each for the learned arm. It is the only
cost measure the NLopt column has, and it cross-validates: on the SNOPT smoke run
`map_jacobian` equalled `User function calls (total)` **exactly** on every cell (377, 236,
332, 272). It is *not* an iteration count -- a line search evaluates the map several times per
accepted step -- so it measures work done, not steps taken. `collate.py` prints `--`, never
`nan` or `0`, where a solver reports nothing: a `0` there would read as "converged instantly"
rather than "does not tell us".

**Status is decoded numerically, not from log text.** SNOPT INFO 34 is the time limit and
Drake leaves it as a generic solver error with no distinctive exit string; NLopt has no text
at all. Matching exit strings alone would report `timeouts: 0` for a capped run of either --
the same trap `is_iteration_cap` was written for. NLopt status 5 is `MAXEVAL_REACHED`, an
*evaluation* cap recorded as `hit_eval_cap`, since NLopt has no notion of an iteration to cap.

### Four traps, all found by probing rather than by reading

- **The laptop's Drake is not the cluster's.** The cluster runs the official **1.56.0**
  tarball, whose `NloptSolver` exposes exactly six options: `algorithm`, `constraint_tol`,
  `xtol_rel`, `xtol_abs`, `max_eval`, `max_time`. A workstation source build additionally
  offers five `local_optimizer_*` options for choosing the AL's inner solver. **Drake
  validates NLopt option names strictly and raises on one it does not know**, so code written
  against the local API passes locally and fails on *every cell* of a cluster run.
  `tests/test_solver_plumbing.py` pins the emitted keys to 1.56.0's six. **A Drake feature
  must be checked against the cluster's version before code depends on it.**
- **`"Timing Level"` is accepted by Drake and silently INERT.** SNOPT's parser is case
  sensitive on the second word; only `"Timing level"` writes the timing block. That option had
  been dead in this repo. Drake raises only on a keyword SNOPT's table does not know at all,
  so **a SNOPT option can be accepted and do nothing** -- confirm anything set here in the
  print file.
- **Drake defaults NLopt's `max_eval` to 1000.** That is a cap, not "unset", and it binds
  here. Left alone the NLopt column would silently measure a 1000-evaluation budget instead of
  the wall-clock cap every other column is measured under. `max_time` carries the cap and
  `max_eval` is set explicitly.
- **SNOPT's print file opens with `SNMEMA EXIT 100 -- finished successfully`** from the
  memory-estimation pass. A bare `EXIT` regex reports that instead of the solve's and calls a
  failed solve a success; the parse anchors on `SNOPTA` and takes the last match.

Two smaller notes. `Solution No` drops the end-of-file row/column dump, a fifth of the print
file that nothing parses -- one log per cell over 480 cells is the many-small-files pattern
this project already had to fix once. And **SNOPT's `Time limit` is checked at major-iteration
boundaries**, so a cell overshoots the cap by one major iteration: measured 24-28 s against a
20 s cap on the learned arm, whose iteration is expensive. That is inside `cell_timeout`
(`5*wall + 300`) and the 4 h `ITEM_TIMEOUT`, but it means SNOPT wall-clock is not capped as
tightly as IPOPT's.

### The local smoke runs, and the one early signal in them

Panda pose, 4 cells, 20 s cap, learned + joint space. **Far too small to rank anything** --
this is plumbing verification, and the ranking question is what stage SOLVER is for. But two
things in it are worth knowing before reading that stage.

| solver | cells solved | wall clock per cell | what bound |
| --- | --- | --- | --- |
| IPOPT | converged, 24-89 majors | 6-8 s | nothing |
| SNOPT | 1/4 | 20-28 s | `Time limit`, all four cells |
| NLopt (`LD_AUGLAG`) | 0/4 | 20.0-20.1 s exactly | `max_time`, all four cells |

**NLopt did not fail to run -- it failed to converge in 20 s, which is a different thing.**
It did 212-237 map evaluations on the learned arm and 2848-5053 on the joint-space arm, and
came out at `max_violation` 0.16-2.41, i.e. making real but incomplete progress. That is the
expected shape for an augmented Lagrangian against tight equality rows, and it is the reason
the axis is worth measuring rather than assumed. Note the joint-space arm, which IPOPT solves
in a fraction of a second, also timed out under NLopt at 20 s.

**`max_time` binds far more tightly than SNOPT's `Time limit`**: 20.0-20.1 s against 24-28 s,
because SNOPT only checks at major-iteration boundaries and one of the learned arm's majors is
expensive. Worth remembering when reading wall-clock columns across solvers -- they are not
capped equally.

### Future work on this axis

- **NLopt settings are unswept**, by decision (Thomas, 2026-09-16: *"Store testing NLOPT
  settings as future work"*). `LD_AUGLAG` vs `LD_AUGLAG_EQ`, `constraint_tol`, `xtol_rel`,
  `xtol_abs`, `max_eval` -- all at Drake's defaults, none measured.
- **The AL's inner local optimizer is not selectable** on the cluster's Drake. Leaving it
  unset is a supported state: Drake does not call `set_local_optimizer`, so NLopt supplies its
  own (LD_LBFGS for the gradient-based AUGLAG families), which is the right shape because an
  AL's inner problem is bound-constrained only. **TODO** when the cluster's Drake carries the
  local-optimizer PRs: expose `local_optimizer_algorithm` and sweep it.
- **`snopt_major_step_limit` and `snopt_violation_limit` are plumbed and unset.** These are
  SNOPT's analogues of the repo's best open IPOPT lead -- `Major step limit` bounds
  `||dx|| <= limit*(1+||x||)` per major iteration (the trust region the runaway wants, since
  it is *one accepted catastrophic step* out of a well-behaved trajectory), and
  `Violation limit` is the counterpart of `ipopt_theta_max_fact`.
- **Do not extend Drake to get better instrumentation.** Thomas: *"NLOPT might not have the
  robust logging we need btw, work with what you have, don't write new logging stuff in Drake
  or anything."* Instrument on our side and report honestly what a solver does not expose.

**One latent bug this surfaced:** `scripts/iiwa/iiwa_benchmark.py`'s default tag omitted
`args.solver` where the Panda's has always included it, so an iiwa SNOPT run and an iiwa IPOPT
run with otherwise identical flags resolved to the same `summary.json` and overwrote each
other -- the same trap `--shard` and `--checkpoint` were each fixed for. Fixed; archived iiwa
runs were all tagged explicitly from manifests, so no archived path moved.

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

**Step ACCEPTANCE is a different lever from step damping, and the first one that has worked.**
Everything refuted above altered the *derivatives* the solver was handed, breaking the
correspondence between the values IPOPT evaluates and the gradients it uses. Filter tuning leaves
the program exactly as written and changes only which trial points are accepted. The mechanism it
attacks: IPOPT holds **variable bounds** at every iterate but general constraints only at
convergence, and in the learned formulation `q` is not a decision variable, so the joint-limit
rows are general constraints and an iterate may sit at `|q| = 1e8`. That is why `lift_q` gave zero
runaways — at the cost of seven equality rows. `ipopt_theta_max_fact` attacks it without touching
the formulation: IPOPT rejects any trial point whose constraint violation exceeds
`theta_max_fact * max(1, theta(x_0))`.

The trajectories say why this should work. Recording every iterate through `VarsToQ` on five
runaway cells (iiwa pose paired, ddp-r1, 20 s), **the solve is well behaved for 22 to 110
iterations and then jumps in a single step** — on three of the five, `|q|_inf` goes from ~2.5 to
past 1e3 in one iterate. The runaway is not a slow drift the solver could be nursed through; it is
one accepted catastrophic step, which is exactly what a filter ceiling can refuse.

**Local probe only, 16 cells, one seed, one chart, laptop GPU — a lead, not a measurement:**

| variant | solved | runaway | median iters | median cost |
| --- | --- | --- | --- | --- |
| default | 11/16 | 5 | 139 | 7.61 |
| `ipopt_theta_max_fact=1` | **14/16** | **2** | 131 | **6.75** |
| `ipopt_theta_max_fact=10` | 11/16 | 5 | 138 | 7.61 (bit-identical to default) |
| `ipopt_watchdog_trigger=0` | 11/16 | 5 | — | — |
| `ipopt_max_soc=8` | 11/16 | 5 | — | — |

Three cells gained, **none lost**, cost *improved*, effect sharply thresholded between 1 and 10.
This is the first intervention that both suppresses the runaway **and** converts it into success —
`tikhonov=10` cut runaways 31 → 7 for exactly 19/19, and `lift_q` reached zero runaways while
losing 78 cells net. `ipopt_theta_max_fact`, `ipopt_watchdog_trigger` and `ipopt_max_soc` are
plumbed and default to None (IPOPT's own defaults). **Not measured**: needs the grasp task, the
Panda, both start protocols and 480 cells, which waits for the ladder to finish.

SNOPT has the closer analogue of a trust region — `Major step limit` (default 2.0) bounds
`||dx|| <= limit*(1+||x||)` per major iteration, and `Violation limit` is its theta ceiling.
Neither is plumbed, and **`parse_log` must learn SNOPT's log format first** (iterations, eval
counts and exit all return None under SNOPT), since iterations is the hardware-independent number.

**So a better iiwa chart is the preferred remedy, and filter step-rejection is now a live second one.**

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

## THE HARDENED PROBLEM AT 480 CELLS (2026-09-15)

All eleven rungs on the hardened scene, one grid per experiment, 60 targets x 8 guesses,
seed 1, 45 s, `--compile`, `learned,numerical`. Grasp targets always shelf-contained; the
pose task fielded **both** ways, because whether containment belongs there was the open
question. 66 runs, 528 items, ~5 h on four nodes.

**Not cell-comparable with `sc_LADDER_*`, by construction** — the hardened scene admits a
different target set, `grid_hash` differs, and `collate.py` refuses the pairing. Compare
HARD columns with each other; the archived columns are quoted below only as "what the same
arm scored on the soft problem", never as a paired test.

| panda | upstream | n12 | n8 | **n6** | n4 | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp native | 432 | 407 | 455 | **462** | 468 | 437 | 323 |
| grasp paired | 347 | 347 | 429 | **444** | 450 | 380 | 323 |
| pose contained, native | 436 | 434 | 439 | **429** | 437 | 417 | 167 |
| pose contained, paired | 271 | 275 | 387 | **392** | 352 | 272 | 167 |
| pose free, native | 466 | 460 | 460 | **460** | 461 | 449 | 220 |
| pose free, paired | 341 | 330 | 422 | **438** | 400 | 347 | 220 |

| iiwa | ddpr1 | n8 | n6 | **n4** | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- |
| grasp native | 266 | 316 | 300 | **391** | 297 | 442 |
| grasp paired | 320 | 338 | 368 | **407** | 334 | 442 |
| pose contained, native | 432 | 441 | 436 | **438** | 417 | 268 |
| pose contained, paired | 287 | 214 | 206 | **411** | 338 | 268 |
| pose free, native | 426 | 428 | 439 | **468** | 425 | 332 |
| pose free, paired | 306 | 208 | 211 | **452** | 332 | 332 |

The harness checks itself and passes: joint space is identical across every rung of a robot
within each experiment (323 / 167 / 220 panda, 442 / 268 / 332 iiwa), and
`median_start_q_error` is 0.0 exactly under `paired`.

### THE HEADLINE: hardening flipped the Panda grasp task, and only the Panda's

Exact McNemar against joint space on the shared 480 cells:

| experiment | learned | js | p | on the SOFT problem |
| --- | --- | --- | --- | --- |
| panda grasp native (`n6`) | **462** | 323 | **1.5e-33** | 471 vs 457, p = 0.02 |
| panda grasp paired (`n6`) | **444** | 323 | **2.9e-23** | 471 vs 457, p = 0.02 |
| panda pose paired (`n6`, contained) | **392** | 167 | **6.2e-47** | 435 vs 228 |
| iiwa grasp native (`n4`) | 391 | **442** | **1.4e-06** | 448 vs 462, p = 0.05 |
| iiwa grasp paired (`n4`) | 407 | **442** | **0.00042** | 449 vs 462, p = 0.07 |
| iiwa pose paired (`n4`, contained) | **411** | 268 | **9.4e-24** | 448 vs 325 |

**On the Panda the learned formulation now wins all six experiments** at p <= 2.9e-23 —
including the grasp task, which the full-depth charts *lost significantly* on the soft
problem. The mechanism is entirely on the baseline's side: joint space fell **457 -> 323**
on Panda grasp while `n6` fell 471 -> 462. Its iteration count tells the same story — 970
median iterations against its archived 48, and 4.26 s against 0.14 s. The hardened grasp
task is a genuinely hard problem for a joint-space formulation, and that is exactly the
saturation the change was meant to remove.

**On the iiwa it went the other way**: joint space barely moved (462 -> 442) while `n4` fell
449 -> 407, taking a row that was at near-parity (p = 0.07) to a clear loss (p = 0.0004).

**The two robots are not measuring the same intervention, and this is the confound to
state.** The Panda *grasp* scene never had decorative mugs, so hardening it is essentially
pure containment (the bin sits at `[0.75, 0, 0]`, nowhere near the shelves). The iiwa scene
lost the bin **and seven welded mugs**, four of them inside shelf compartments — so its
grasp task gained a containment requirement while *losing* obstacles. Any claim of the form
"hardening helps/hurts the learned arm" has to carry that caveat until the iiwa is re-run
with its decorative mugs kept.

### THE CLUTTER WAS NOT THE EXPLANATION (stage HARDMUG, 2026-09-15)

The obvious suspicion about the table above was that the two robots did not receive the same
intervention: the Panda GRASP scene never had decorative mugs, so hardening it is near-pure
containment, while the iiwa scene lost seven of them, four inside shelf compartments. Stage
HARDMUG re-ran all five iiwa rungs on `--scene nobin` — bin removed, **clutter kept**, which
is the iiwa's match for what the Panda got. 30 runs, 240 items, same grid shape and seed.

**It changes essentially nothing.** `n4`, against the hardened (clutter-free) columns:

| experiment | learned nobin | learned hardened | Δ | js nobin | js hardened | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| grasp native | 392 | 391 | **+1** | 428 | 442 | -14 |
| grasp paired | 412 | 407 | **+5** | 428 | 442 | -14 |
| pose contained, paired | 412 | 411 | **+1** | 280 | 268 | +12 |
| pose free, paired | 449 | 452 | **-3** | 321 | 332 | -11 |

Seven welded obstacles, four of them in the compartments targets are drawn from, are worth
**-6 to +5 cells of 480 to the learned arm and -14 to +12 to joint space**. The one verdict
that moves is iiwa grasp paired, from a joint-space win (p = 0.00042) to a tie (412 vs 428,
p = 0.13); grasp native stays a loss (392 vs 428, p = 0.0012).

**So the confound was real in principle and immaterial in practice, and the iiwa/Panda
divergence is a property of the robots, not an artifact of the scene.** Recorded because the
hypothesis was explicit and is now refuted — do not re-open it.

### What each hardening step is actually worth

With clutter measured at ~0, the three steps separate cleanly:

| step | what it does | measured worth |
| --- | --- | --- |
| remove the bin | `binF` sits at `[0.75, 0, 0]`, nowhere near the shelves | nil by construction; `../codebase` measured it "statistically free" |
| remove the decorative mugs | seven welded obstacles, four in compartments | **±14 cells of 480, both arms** (above) |
| **contain the target in a shelf** | the target must be reachable *inside* a compartment | **everything** (below) |

**Containment is the whole intervention, and its sign is opposite on the two robots:**

| | Δ joint space | Δ learned |
| --- | --- | --- |
| **panda** grasp native (`n6`) | 457 -> 323, **-134** | 471 -> 462, **-9** |
| **panda** grasp paired (`n6`) | 457 -> 323, **-134** | 471 -> 444, **-27** |
| **iiwa** grasp native (`n4`) | 462 -> 442, **-20** | 448 -> 391, **-57** |
| **iiwa** grasp paired (`n4`) | 462 -> 442, **-20** | 449 -> 407, **-42** |

On the Panda containment costs joint space 5-15x what it costs the learned arm; on the iiwa
it costs the learned arm 2-3x what it costs joint space. The iteration columns show why the
Panda moved: reaching into a compartment takes its joint-space arm **970 median iterations
against its archived 48**, where the iiwa's takes 448. The hardened grasp task is near the
edge of what a joint-space formulation does cheaply on the Panda and is not on the iiwa.

### POSE RE-MEASURED ON THE CORRECTED PROGRAM (stage POSE2, 2026-09-16)

480 cells, 45 s, all eleven rungs, both placements, both protocols, on the corrected scene
(one finray gripper for both robots, both tasks), with the conditioning frame calibrated and
`c` seeded in the flow's frame, and containment keyed on the gripper base -- the same point
on the hand for both robots. **This supersedes every earlier pose table in this file.**

| panda | upstream | n12 | n8 | **n6** | n4 | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pose contained, native | 435 | 434 | 444 | **428** | 432 | 394 | 185 |
| pose contained, paired | 242 | 242 | 369 | **382** | 347 | 250 | 185 |
| pose free, native | 466 | 471 | 455 | **458** | 453 | 447 | 201 |
| pose free, paired | 294 | 293 | 418 | **438** | 390 | 303 | 201 |

| iiwa | ddpr1 | n8 | n6 | **n4** | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- |
| pose contained, native | 434 | 430 | 433 | **433** | 420 | 268 |
| pose contained, paired | 285 | 234 | 232 | **384** | 331 | 268 |
| pose free, native | 442 | 443 | 452 | **470** | 431 | 332 |
| pose free, paired | 284 | 267 | 269 | **447** | 354 | 332 |

Joint space is identical across every rung of a robot within each experiment (185 / 201
Panda, 268 / 332 iiwa), and the learned arm wins every one of the eight rows decisively --
Panda `n6` contained paired 382 against 185 at **p = 2e-42**, iiwa `n4` 384 against 268 at
**p = 5e-16**.

#### The containment verdict REVERSES on the corrected program

The earlier table had containment costing joint space 53-64 cells against the learned arm's
30-46, which read as "containment hardens the task without narrowing the claim". That was
measured on a program whose pose path never calibrated its conditioning frame. Corrected:

| | learned | joint space | effect on the learned margin |
| --- | --- | --- | --- |
| panda `n6` native | -30 | -16 | **-14** |
| panda `n6` paired | -56 | -16 | **-40** |
| iiwa `n4` native | -37 | -64 | +27 |
| iiwa `n4` paired | -63 | -64 | +1 |

**On the Panda containment now costs the learned arm 2-3.5x what it costs joint space, and
shrinks the margin by 14-40 cells; on the iiwa it is a wash.** So pose containment does not
strengthen the comparison -- it makes the problem harder and, on one robot, harder for the
arm under test specifically. It remains a legitimate harder problem; it is simply not the
free win the pre-fix table suggested. **Whether to keep it is Thomas's call**, and the
fingertip variant (stage FINGER) is the other half of the evidence.

The rungs behave as the ladder predicts throughout: `n4` leads the iiwa on every row, and on
the `paired` protocol the full-depth charts collapse on both robots (Panda `upstream`/`n12`
242-294 against `n6`'s 382-438; iiwa `n8`/`n6` 232-269 against `n4`'s 384-447), which is the
gain-ceiling runaway and not a containment effect -- it is present in the free columns too.

### THE ADOPTED GRASP DEFAULT: HARDENED SCENE, FREE TARGETS (stage GRASPFREE, 2026-09-16)

Grasp containment defaults **off** (Thomas, 2026-09-15: its sign is opposite on the two robots
and too many other knobs are in flight). That configuration -- hardened scene, free grasp
targets -- had never been measured: the archived columns are the legacy scene and stage HARD
is the contained one. 480 cells, all eleven rungs, both protocols.

| panda | upstream | n12 | n8 | **n6** | n4 | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp native | 432 | 437 | 465 | **474** | 479 | 456 | 453 |
| grasp paired | 417 | 409 | 463 | **475** | 474 | 443 | 453 |

| iiwa | ddpr1 | n8 | n6 | **n4** | n12w256 | **js** |
| --- | --- | --- | --- | --- | --- | --- |
| grasp native | 282 | 318 | 294 | **446** | 311 | 457 |
| grasp paired | 286 | 336 | 297 | **457** | 336 | 457 |

| best rung | contained (stage HARD) | free (adopted) | p, free |
| --- | --- | --- | --- |
| panda `n6` native | 462 v 323 | **474 v 453** | 0.00032 |
| panda `n6` paired | 444 v 323 | **475 v 453** | 0.00011 |
| iiwa `n4` native | 391 v 442 | 446 v 457 | 0.14 (tie) |
| iiwa `n4` paired | 407 v 442 | **457 v 457** | 1.0 (exact parity) |

**On the adopted default the learned arm wins the Panda grasp task and ties the iiwa's.**
That is a better position than either the soft legacy problem (where the Panda's full-depth
charts lost) or the contained one (where the iiwa lost at p = 0.0004). The iiwa reaches
**exact parity at 457/457** under `paired`, which is the first time that row has not been a
deficit.

Note the baseline is back near saturation on this configuration (453 / 457), which is the
ceiling the hardening was meant to remove -- so **grasp containment remains the lever to
revisit**, and it is kept available behind `--target-placement shelf`. It should be re-tried
whenever another knob moves the picture, a solver change most of all: on the contained task
joint space needs 970 median iterations against its 48 here, so a solver that changes how the
baseline copes with a hard active set could change that verdict.

### WHY THE GRASP TASK LOOKS LIKE "ONLY A TIE", AND WHAT IS ACTUALLY THERE

Success counts on the adopted grasp default read as a narrow Panda win and an iiwa tie. That
reading is an artefact of **headroom**: joint space is at 94-95% there, so only 23-27 cells
of 480 are available to win at all. Decomposed:

| config | L | JS | L only | JS only | both | neither | of JS's failures, rescued |
| --- | --- | --- | --- | --- | --- | --- | --- |
| panda native, free | 474 | 453 | 27 | 6 | 447 | 0 | **27/27 = 100%** |
| panda paired, free | 475 | 453 | 27 | 5 | 448 | 0 | **27/27 = 100%** |
| iiwa native, free | 446 | 457 | 17 | 28 | 429 | 6 | 17/23 = 74% |
| iiwa paired, free | 457 | 457 | 17 | 17 | 440 | 6 | 17/23 = 74% |

**On the Panda the learned arm solves every single cell joint space cannot** -- 27 of 27,
under both protocols, with `neither` = 0. It is not marginally better; the task has nothing
left to give. And the iiwa's 457-vs-457 is not the same 457 cells: 17 each way, so the arms
are genuinely complementary even where the totals agree.

**Containment is what creates headroom, which is the argument for revisiting it:**

| config | L | JS | JS fails | rescued | p |
| --- | --- | --- | --- | --- | --- |
| panda native, contained | 462 | 323 | 157 | **148 (94%)** | 1.5e-33 |
| panda paired, contained | 444 | 323 | 157 | **142 (90%)** | 2.9e-23 |
| iiwa native, contained | 391 | 442 | 38 | 30 (79%) | 1.4e-06 |
| iiwa paired, contained | 407 | 442 | 38 | 30 (79%) | 0.00042 |

Containment takes the Panda's headroom from 27 cells to 157 and the learned arm takes 90-94%
of it. **The learned arm's rescue rate is high and stable everywhere -- 74-100% -- on both
robots and both configurations.** That is the quantity the success counts obscure.

#### The iiwa grasp deficit is a convergence problem, not a runaway

So the iiwa deficit is not a failure to rescue; it is that the iiwa learned arm carries **its
own failure set** that joint space does not share (28 and 17 cells free, 81 and 65
contained). Every one of those is `fail_reason = "constraint"` with median `max_violation`
0.009-0.037 -- **centimetres off, not astronomical** -- against timeout counts that track the
lost counts closely (38/27 free, 88/75 contained) at 286-434 median iterations and 34-36
ms/it.

**These are cap-bound near-misses, not divergence.** The runaway signature is
`max_violation` >= 1e+03; this is 1e-02. So the remaining iiwa grasp deficit is the learned
arm converging too slowly on a subset of cells, which is a different problem from the
gain-ceiling runaway and is plausibly reachable by the deferred solver work -- step
rejection, or SNOPT/NLOPT -- rather than by another chart.

### WHAT THE CALIBRATION FIX WAS WORTH ON THE IIWA, ISOLATED

The iiwa's `posefree` columns isolate the fix exactly: its scene never changed, and with no
containment the containment-point change cannot reach the sampler, so **the pre-fix (stage
HARD) and post-fix (stage POSE2) runs are on the same grid** -- `grid_hash` matches and the
comparison is a paired McNemar over all 480 cells. Joint space is 332 -> 332 on every row, as
it must be, since it never touches the flow.

| rung | start | pre-fix | post-fix | better/worse | p | median `max_violation` |
| --- | --- | --- | --- | --- | --- | --- |
| ddpr1 | native | 426 | 442 | 34/18 | 0.036 | 1.2e-08 -> 1.3e-08 |
| ddpr1 | paired | 306 | 284 | 82/104 | 0.12 | 8.4e-08 -> 9.6e-08 |
| **n8** | **paired** | **208** | **267** | 138/79 | **7.5e-05** | **3.1e+03 -> 7.2e-07** |
| **n6** | **paired** | **211** | **269** | 138/80 | **0.0001** | **9.6e+04 -> 2.3e-07** |
| n8 | native | 428 | 443 | 31/16 | 0.04 | 1.4e-08 -> 1.3e-08 |
| n6 | native | 439 | 452 | 28/15 | 0.066 | 1.3e-08 -> 1.2e-08 |
| n4 | native | 468 | 470 | 8/6 | 0.79 | 5.6e-09 -> 5.9e-09 |
| n4 | paired | 452 | 447 | 25/30 | 0.59 | 1.0e-08 -> 1.0e-08 |
| n12w256 | paired | 332 | 354 | 94/72 | 0.10 | 1.9e-08 -> 1.7e-08 |

**The gain is concentrated exactly where the chart had headroom to run away into.** On `n8`
and `n6` under `paired` the fix is worth ~58 cells apiece and collapses the median violation
by **ten orders of magnitude** -- 3.1e+03 and 9.6e+04 down to ~1e-07 -- i.e. it removes the
runaway outright. On `n4`, whose gain ceiling (2.2e4) sits below the runaway band, it is worth
nothing at all: 468 -> 470 and 452 -> 447, both well inside noise, with `max_violation`
unchanged at ~1e-08 because there was never anything wrong to fix.

**This revises, but does not overturn, the runaway story.** The "n8/n6 pose-paired runaway"
recorded in the ladder was *partly* this bug: conditioning the network 45 mm off its trained
frame was pushing those charts into headroom they have and `n4` does not. After the fix they
sit at 267/269 -- still far below `n4`'s 447, so the gain ceiling remains the dominant effect
and the chart-selection rule is unchanged. What changes is the size of the deficit
attributable to architecture alone.

The lesson for reading this file: **a miscalibration and an architectural weakness produce the
same symptom, and the one masks the size of the other.** Every pre-fix pose number
over-attributed to the ceiling whatever the 45 mm was contributing.

### FINGERTIP CONTAINMENT BEATS WRIST CONTAINMENT (stage FINGER, 2026-09-16)

The pose task's containment point is a choice, and the two candidates are one 0.100 m step
apart along the gripper -- the same step on both robots, which is the whole reason containment
is keyed on the gripper rather than on each task's own target frame. 480 cells, all eleven
rungs, both protocols; the wrist arm is stage POSE2's `posein` columns on the same grid.

| best rung | free | wrist | **fingertip** |
| --- | --- | --- | --- |
| panda `n6` native | 458 v 201 (+257) | 428 v 185 (+243) | **461 v 217** (+244) |
| panda `n6` paired | 438 v 201 (+237) | 382 v 185 (+197) | **405 v 217** (+188) |
| iiwa `n4` native | 470 v 332 (+138) | 433 v 268 (+165) | **470 v 299** (+171) |
| iiwa `n4` paired | 447 v 332 (+115) | 384 v 268 (+116) | **422 v 299** (+123) |

**Both arms score higher at the fingertips than at the wrist, on every row** -- learned
405-470 against 382-433, joint space 217/299 against 185/268. That is what the geometry
predicts: putting the *hand* in a compartment is a shallower reach than driving the *wrist*
in behind it, so the wrist definition silently demands 0.1 m more penetration into a 0.10 m
deep compartment.

The learned arm wins decisively under all three definitions -- under fingertip containment,
p = 4.6e-68 / 9.0e-37 (Panda native/paired) and 1.1e-46 / 7.3e-20 (iiwa).

**Fingertip is the better containment point on both counts.** It is the more faithful
statement of the task -- "the gripper reaches into the shelf", not "the wrist does" -- and it
preserves the learned margin at least as well as the wrist on three rows of four (iiwa +171
against +165 and +123 against +116; Panda native +244 against +243), the exception being
Panda paired (+188 against +197). Against no containment at all it still costs margin on the
Panda (+188 against free's +237) and gains it on the iiwa (+123 against +115), so containment
remains a genuine difficulty increase rather than a free win -- see the POSE2 section.

### THE SHELF-DEPTH INSET DOES NOT CHANGE THE STORY (stage INSET, 2026-09-16)

0.10 m was adopted from `../codebase` and this repo had never swept it. 60 cells, 45 s, one
rung per robot (iiwa `n4`, Panda `n6`), both tasks contained, both protocols, insets
0 / 0.05 / 0.10 / 0.125.

**Success moves by 2-10 cells of 60 across the entire span, with no monotone trend and no
ordering flip.** Every row's learned-vs-joint-space verdict is the same at every inset: the
iiwa loses the grasp task and wins pose at all four, the Panda wins the grasp task at all
four. At 60 cells, where reproducibility at the cap is +/-1 cell, a range of 9-10 is
wobble rather than a dose curve.

What the inset *does* move, steeply, is acceptance -- 0.60% -> 0.10% of raw draws on Panda
grasp, 0.69% -> 0.06% on iiwa pose. So the inset buys sampling cost, not difficulty, and
**0.10 m stands**: it is the sibling's value, it is comfortably samplable at the 50000 guard,
and nothing downstream depends on it. Do not re-sweep it without a reason.

### THE POSE TASK NEVER CALIBRATED THE CONDITIONING FRAME (found 2026-09-16)

`CalibrateFlowFrame` was called only by the *grasp* subclasses. On the pose task it was
skipped entirely, and that was invisible for two different reasons per robot:

* the **Panda** pose scene was `panda_jrl.urdf`, whose `panda_hand` IS the Franka offset the
  network was trained on, so the offset really was identity;
* the **iiwa** pose scene's `iiwa_link_7` is 45 mm from the flow's frame -- a pure
  translation, no rotation, mild enough to never announce itself.

Removing the stock Panda hand destroyed the first coincidence. The pose scene now welds the
finray, whose own `panda_hand` sits **27 mm and 120 degrees** away -- the exact trap
`CalibrateFlowFrame`'s docstring was written about -- and the Panda pose task collapsed to
**10/60 with median `max_violation` 0.4**, against 58/60 and 1.8e-08 for the grasp task in
the *same scene* with the *same chart*. The grasp task was fine precisely because it was the
only path that calibrated.

Both base pose programs now calibrate. It is a no-op wherever the frames already agree, and
it draws from its own fixed-seed generator so it cannot shift the grid. **Every archived
pose column predates this**, including the iiwa's, which ran 45 mm off its trained frame
throughout -- so the pose halves of the ladder, the training-step sweep and stage HARD are
all superseded, not merely re-based.

The lesson generalises past this bug: **a calibration that is skipped is indistinguishable
from a calibration that is correct, until the geometry it was silently relying on changes.**
The pose path had no test asserting `X_ee_flow` was ever measured.

### The pose-placement verdict: containment costs the baseline roughly twice what it costs the learned arm

| | js free | js contained | learned free | learned contained |
| --- | --- | --- | --- | --- |
| panda (`n6` paired) | 220 | **167** (-53) | 438 | **392** (-46) |
| iiwa (`n4` paired) | 332 | **268** (-64) | 452 | **411** (-41) |
| iiwa (`n4` native) | 332 | **268** (-64) | 468 | **438** (-30) |

Containment is a real difficulty increase for both arms, and it is **not** symmetric: it
costs joint space 53-64 cells against the learned arm's 30-46. So it hardens the pose task
without narrowing the claim — the learned margin widens. **Recommend adopting `posein` as
the pose default**, with `posefree` retained as the ablation that shows what containment
did. Note the caveat recorded above: the pose containment point is the frame the target
*is* — `iiwa_link_7` / `panda_hand`, the **wrist, not the fingertips** — so it is a
different and differently-hard condition on each robot.

### Iterations, cost and wall clock

Medians over succeeded cells; cost on the cells **both** arms solved, learned-only
regularizers excluded.

| experiment | L iters | L s | ms/it | JS iters | JS s | n both | L cost | JS cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| panda `n6` grasp native | 200 | 8.77 | 44 | 970 | 4.26 | 314 | 7.510 | **5.305** |
| panda `n6` grasp paired | 301 | 13.39 | 44 | 970 | 4.24 | 302 | 6.846 | **5.164** |
| panda `n6` pose contained paired | 142 | 5.56 | 39 | 44 | 0.14 | 140 | **9.226** | 9.901 |
| iiwa `n4` grasp native | 434 | 14.65 | 34 | 448 | 1.63 | 361 | 4.930 | **2.827** |
| iiwa `n4` pose contained paired | 179 | 4.86 | 27 | 231 | 0.10 | 231 | **5.743** | 6.065 |

Two things the success columns hide. **The hardened grasp task costs the joint-space arm its
cheapness**: 970 and 448 median iterations against its archived 48 and 66, so its per-cell
wall clock rose 30x on the Panda. The learned arm's per-iteration penalty is accordingly
much smaller here — 44 ms against 4.4 ms on the Panda grasp row, ~10x rather than the ~13-30x
of the soft problem. And **the cost split by task survives hardening**: the learned arm wins
on cost on the pose task on both robots and loses by ~1.4-1.7x on the grasp task.

### The rungs still behave as the ladder said

`n4` is the iiwa's best rung on every row; on the Panda `n6`/`n4` lead and the full-depth
`upstream`/`n12` trail by 40-100 cells on the grasp task. The `n8`/`n6` pose-paired runaway
replicates on the iiwa (206-214 against `n4`'s 411), so the gain-ceiling selection rule is
unaffected by the scene change.

## Next steps

**Thomas's roadmap, in priority order (2026-09-04)**, given once the corrected campaign finished:
*"iiwa checkpoint training infra (and launching the multi-day training job), SNOPT and NLOPT, and
then performance tuning and formulation tweaks for getting the best results with the learned
formulation."*

1. **Retrain the iiwa chart. DONE — the reduced-capacity ladder is trained and measured.**
   `iiwa14_ddp_r1` (620k steps, adopted at `9b887c0`) cut `frac_gt_1000` from 3.34% to 0.0125%,
   but left `pole/max` at 7.8e9 and iiwa grasp at 270/480 against joint space's 462 — the
   headroom was still there, that run had only moved less of the domain into it. The follow-on
   trained deliberately **simpler, less accurate** charts: `nb_nodes` 12/8/6/4 lowers the
   architectural gain ceiling `exp(2.4975·nb_nodes)` from 1e13 to 2e4, with a width-only rung
   (12 blocks, `coeff_fn_internal_size` 256) as the control separating "less accurate" from
   "less headroom". Nine runs, both robots, 620k steps each at `ddp_r1`'s optimiser settings.
   `cluster/ladder_runs.txt` is the spec; `--stage LADDER` measures it on stage CKPT's grid.

   **THE LADDER AT 480 CELLS (2026-09-14).** All eleven rungs, one grid, 60 targets x 8 guesses,
   seed 1, 45 s, `--compile`, both tasks, both protocols, joint space in every run.

   | experiment | `ddpr1` | `n8` | `n6` | `n4` | `n12w256` | **iiwa js** |
   | --- | --- | --- | --- | --- | --- | --- |
   | grasp native | 267 | 309 | 288 | **448** | 349 | 462 |
   | grasp paired | 301 | 327 | 302 | **449** | 344 | 462 |
   | pose native | 432 | 426 | 441 | **463** | 422 | 325 |
   | pose paired | 307 | 221 | 232 | **448** | 337 | 325 |

   | experiment | `upstream` | `n12` | `n8` | `n6` | `n4` | `n12w256` | **Panda js** |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | grasp native | 443 | 430 | 465 | 471 | **476** | 459 | 457 |
   | grasp paired | 418 | 416 | 467 | 471 | **474** | 447 | 457 |
   | pose native | **474** | 463 | 461 | 462 | 459 | 451 | 228 |
   | pose paired | 340 | 339 | 419 | **435** | 407 | 332 | 228 |

   **Reducing depth from 12 helps on both robots, and the optimum rung differs by robot** — `n4`
   on the iiwa, `n6` on the Panda (435 against `n4`'s 407 on pose paired). "The smallest chart
   wins" was an iiwa-only result and does not generalise.

   **Against joint space**, exact McNemar within each run (better/worse for learned):

   | experiment | iiwa `ddpr1` | iiwa `n4` | Panda `n12` | Panda `n6` |
   | --- | --- | --- | --- | --- |
   | grasp native | JS 9/204, **3e-49** | JS 15/29, 0.05 | JS 20/47, **0.001** | L 23/9, **0.02** |
   | grasp paired | JS 8/169, **2e-40** | JS 15/28, 0.07 | JS 22/63, **1e-05** | L 22/8, **0.02** |
   | pose native | L 139/32, **4e-17** | L 147/9, **3e-33** | L 248/13, **2e-57** | L 239/5, **5e-64** |
   | pose paired | JS 92/110, 0.2 | L 143/20, **4e-24** | L 170/59, **1e-13** | L 224/17, **3e-47** |

   **Three things only 480 cells show, two of which move a claim:**

   - **iiwa `n4` does NOT reach grasp parity.** It is ~14 cells of 480 short, p = 0.05 and 0.07.
     The exact parity seen at 60 cells was the small grid saturating joint space at 60/60 — there
     are ~18 winnable cells there and `n4` takes most but not all. Still a transformation of
     `ddp_r1`'s 9/204. **This is the project's one remaining deficit: one robot, one task.**
   - **The Panda's full-depth charts significantly LOSE the grasp task** (`n12` p = 0.001 native,
     1e-05 paired; `upstream` the same shape) while every reduced-depth rung wins it. At 60 cells
     those rows were 59/60 against 54/60 and not significant, so the reduced-chart case on the
     Panda is *stronger* than the triage suggested, not weaker.
   - **The iiwa runaway on `n8`/`n6` pose paired replicates at scale** — `median_max_violation`
     2e+03 and 1e+03, scoring 221 and 232, *below* `ddp_r1`'s 307. Only `n4` (1e-08) and
     `n12w256` (2e-08) are clean. Depth is not a dose curve; there is a cliff between 6 and 4.

   **The two mechanisms, and they differ by robot.** On the iiwa a smaller chart buys cells by
   eliminating runaway configurations; on the Panda, which has no runaway at all
   (`median_max_violation` 1e-08 on every rung), it buys them by fitting more iterations inside
   the fixed cap. Per-iteration cost falls monotonically with depth — iiwa grasp native 80 / 55 /
   42 / 33 ms for n12 / n8 / n6 / n4, with `n12w256` at 72 holding depth — and timeouts collapse
   with it (iiwa `n4` 33/31/6/10 against `ddp_r1`'s 212/194/42/170). The learned arm's
   per-iteration penalty against joint space is now **~13x, down from ~30x**. The second
   mechanism is an implementation-and-hardware property, not a better-shaped chart, which is why
   ms/it must be reported beside success.

   **Two controls land as intended.** Panda `upstream` and `n12` agree within noise on all four
   rows, so our training recipe reproduces Jeremy's and no reduced-Panda result is confounded
   with "our recipe vs. his". And `n12_w256`, the accuracy-only control, never beats `n12`
   significantly while keeping the slow iteration: width costs accuracy without buying either
   headroom or speed. Depth buys the speed.

   **Chart accuracy is a clean monotone dose curve and it runs BACKWARDS to cells.** Median FK
   error over 5000 poses, 4/6/8/12 blocks: 20.0 / 12.1 / 11.3 / 10.1 mm on the iiwa,
   14.7 / 9.5 / 7.2 / 6.1 mm on the Panda; the width rungs are 75.1 and 22.9 mm. The *least*
   accurate depth rung solves the most on the iiwa. Note 20 mm chart error does not appear in the
   solutions — the IK constraint is on `FK(q)`, so `n4`'s solved cells return
   `median_max_violation` 1.1e-08; the chart only decides where the solver starts.

   **Neither intrinsic screen predicts cells, in either direction.** iiwa `n8` screens cleanest of
   all four (task-pose `frac_gt_1000` 0.0, `pole/max` 202) and is the worst rung on pose paired;
   `n4` screens dirtier and solves best. A chart can be clean everywhere the sampler looks and
   catastrophic everywhere the Newton step goes. **The screen is a smoke test, not a selection
   criterion** — which is the argument for folding a few optimization cells into checkpoint
   validation (see "Smaller open items").

   **Pole mass is CREATED BY TRAINING, monotonically, on every rung.** Screening all 31 kept
   checkpoints per rung: iiwa `n4`'s task-pose `pole/max` runs 27 -> 2.5e3 over 20k..620k steps,
   iiwa `n6`'s 409 -> 1.3e5, Panda `n12`'s 50 -> 3.5e11. The headroom is present at initialisation
   and barely used; SGD walks the network into it while buying accuracy. `n4`'s trajectory is
   smooth; `n6`'s swings four orders between adjacent checkpoints, which predicts exactly the
   erratic `n6` rows above. **Accuracy itself is converged by ~480k** — the last eight checkpoints
   of every rung are within 3%, and 620000 is best or within 2% of best on all nine — so there is
   no better checkpoint to hunt, and selecting one per rung would confound architecture with
   selection. `--stage TRAJ` measured the training-step axis directly, and its answer is below.

   **THE TRAINING-STEP SWEEP (2026-09-15): training makes the iiwa `n6` chart WORSE, and does
   nothing at all for `n4`.** Same 480-cell grid, seed 1, 45 s, `--compile`, both tasks, both
   protocols, learned arm against joint space in every run; the 620k column is the ladder's own.

   | iiwa `n4` | 20k | 100k | 200k | 400k | 620k | js |
   | --- | --- | --- | --- | --- | --- | --- |
   | grasp native | 459 | 449 | 447 | 456 | 448 | 462 |
   | grasp paired | 461 | 456 | 453 | 452 | 449 | 462 |
   | pose native | 460 | 463 | 464 | 467 | 463 | 325 |
   | pose paired | 448 | 455 | 447 | 451 | 448 | 325 |

   | iiwa `n6` | 40k | 120k | 240k | 400k | 620k | js |
   | --- | --- | --- | --- | --- | --- | --- |
   | grasp native | **425** | 317 | 299 | 298 | 288 | 462 |
   | grasp paired | **429** | 325 | 318 | 305 | 302 | 462 |
   | pose native | 446 | 440 | 450 | 443 | 441 | 325 |
   | pose paired | **331** | 268 | 247 | 228 | 232 | 325 |
   | `median_max_violation`, pose paired | 3.8e-08 | 4.1e-07 | 4.0e-06 | **5.1e+03** | 1e+03 | |

   Exact McNemar, earliest checkpoint against 620k: **`n6` loses three of four rows to its own
   first checkpoint** — grasp native 22/159 (**p = 8.3e-27**), grasp paired 15/142
   (**p = 4.1e-27**), pose paired 63/162 (**p = 3.1e-11**), pose native 28/33 (p = 0.61). **`n4`
   moves on none of the four** — 17/28 (p = 0.14), 15/27 (p = 0.088), 16/13 (p = 0.71), 27/27
   (p = 1.0).

   **This is the pole-mass screen's prediction confirmed downstream, and it is the sharpest
   statement the campaign has that accuracy is not the quantity that matters.** `n4` at 20k
   steps has ~78 mm median FK error against 620k's 20 mm — a 4x accuracy gain over 600k steps
   of training, worth **zero cells** in all four experiments. Meanwhile `n6` buys the same kind
   of accuracy and *pays* 137 grasp cells for it, because its `pole/max` climbs 409 -> 1.3e5 over
   the same interval and the violation column shows the runaway arriving: `median_max_violation`
   on pose paired crosses from 1e-08 to **1e+03** between 240k and 400k. Training is what walks
   a chart into its architectural headroom; `n4`'s ceiling of 2.2e4 is below the runaway band, so
   there is nothing for training to walk it into and its curve is flat.

   **This is not a licence to pick early checkpoints.** Selecting a checkpoint on this grid is
   selecting on the test set, and it would confound architecture with selection — which is why
   every ladder rung is reported at 620k. What the sweep licenses is the opposite conclusion:
   **the rung to pick is the one whose ceiling makes the choice moot.**

   #### THE CHART-SELECTION RULE

   **Choose an architecture whose gain ceiling `exp(2.4976 · nb_nodes)` sits below the ~1e7
   runaway band — on the iiwa that is `nb_nodes = 4` — then take the final checkpoint, because
   below the ceiling the training-step axis is flat and above it later checkpoints are strictly
   worse.**

   The ceiling is `atan` and the block count, nothing else: FrEIA's coupling block is
   `y = exp(s)·x + t` with `s = clamp · 0.636 · atan(s_raw)`, so `|s| < clamp·0.636·π/2 = 2.4976`
   at `rnvp_clamp = 2.5` and one block amplifies by at most 12.15x. Over `nb_nodes`: 2.2e4 / 3.2e6
   / 4.8e8 / 1.0e13 at n4 / n6 / n8 / n12. It bounds the weights, so no amount of training escapes
   it — and **training reliably walks 78-89% of the way up whatever log-ceiling it is given**
   (observed `pole/max` at 620k: `n4` 2.5e3, `n6` 1.3e5, Panda `n12` 3.5e11), so the ceiling
   predicts where a trained chart lands rather than merely bounding it.

   Note the band: the configurations that actually kill solves are **1e7 to 1e16 rad**.
   `frac_gt_1000`'s threshold of 1000 is a bimodality separator (ordinary configurations sit at
   ~2.5 rad), *not* the level at which a solve dies — `n4`'s ceiling is above 1000 and is fine.
   Every competing criterion is disqualified by measurement: accuracy (converged by 480k, and
   backwards across rungs), intrinsic pole screening (`n8` screens cleanest and solves worst),
   "take the last checkpoint" (true only below the ceiling), and "take whichever benchmarks best"
   (selection on the reporting grid). If a chart above its ceiling ever *must* be selected among,
   the only sound version is a held-out optimization grid — which is the deferred export-time
   smoke-cells item, now with a design.

   The harness checks itself and passes: joint space is identical across every rung of a robot
   AND identical to the archived 480-cell columns (462 / 325 / 457 / 228), grid hashes match, and
   `median_start_q_error` is 0.0000 exactly under `paired`.


   **A path bug cost this rung its first measurement (2026-09-09, fixed in `6fbff55`).**
   `train_flow.sh` reassigns `HOME="$ROOT/home"` so ikflow resolves `DATASET_DIR` at import;
   `export_and_screen_job.sh` derives its own `ROOT` from `$HOME`, so the inline export resolved
   every path one level deep and died with `no checkpoints under .../learned-ik/home/learned-ik/...`.
   The rung trained all 620k steps and exported nothing, and the four interleaved benchmark jobs
   fired at a checkpoint that did not exist — **failing per cell rather than fast**, so they burned
   16 minutes each before anyone noticed. `submit_bench.sh` now refuses when a manifest names a
   checkpoint absent from the cluster (`dd24cc3`), which is the guard that would have caught it.

   **Plan around SuperCloud's monthly maintenance** — second Tuesday, compute down Monday evening
   to Wednesday morning, nothing survives it. The window closed 2026-09-08; the next is
   2026-10-12 to 10-14.

2. **SNOPT and NLOPT. DONE as infrastructure** -- the blocker is fixed and all three solve the
   real program. See "The solver axis" below. What remains is to *run* it: the first measurement
   is stage SOLVER, a 60-cell triage on the adopted rungs.

3. **Performance tuning and formulation tweaks.** Note this is Thomas naming formulation work as a
   work item, not a standing licence — what is compared remains his call, made explicitly in
   advance.

**Thomas's update once the ladder landed (2026-09-14):** *"Sounds like we're making big strides
in getting things better, but joint space is still a bit better than learned. I have a couple
ideas for things that might help (using a harder problem formulation, trying some sort of step
rejection tricks), but first, let's get these results in to see where we're at."* So the two
live items, after the results are in:

- **A harder problem formulation. DONE — measured at 480 cells, see "THE HARDENED PROBLEM"
  above.** The scene keeps its four shelves but loses the bin and the decorative mugs, and a
  target is accepted only if it lands inside a shelf compartment (0.10 m depth inset), mirroring
  `../codebase`'s hardened Grasp Selection. It did what it was for **on the Panda**: joint space
  fell 457 → 323 on the grasp task and the learned arm went from losing that row to winning it
  at p = 1.5e-33. **On the iiwa it did not** — joint space barely moved (462 → 442) while `n4`
  fell to 407, so that row got worse, not better.

  The suspected confound — that the two robots did not receive the same intervention, the
  Panda grasp scene never having had decorative mugs — was measured by stage HARDMUG and
  **refuted**: keeping the iiwa's clutter moves either arm by at most 14 cells of 480. The
  divergence is a property of the robots. What remains open: **adopt `posein` as the pose
  default** — containment costs joint space
  53-64 cells against the learned arm's 30-46, so it hardens the task without narrowing the
  claim. Third, the pose containment point is the **wrist** (`iiwa_link_7` / `panda_hand`), not
  the fingertips; if that should be defined differently it must change before the numbers are
  written up.
- **Step rejection (IPOPT filter tuning).** Already plumbed and carrying the best lead in the
  repo — see "Step ACCEPTANCE is a different lever" above. `ipopt_theta_max_fact=1` gained three
  cells and lost none on a 16-cell probe, cut runaways 5 → 2, and improved cost, with `=10`
  bit-identical to the default. Needs the grasp task, the Panda, both protocols, 480 cells. It
  was explicitly deferred until the ladder finished, which it now has.

### Smaller open items

- **Fold a small optimization smoke run into checkpoint validation.** Thomas's idea,
  2026-09-08, explicitly deferred (*"Obviously, not worth it right now, but a cool idea for
  the future"*). `cluster/export_and_screen_job.sh` currently screens a checkpoint on
  intrinsic metrics only — pole exposure on both domains, chart accuracy — but what actually
  matters is whether the chart makes the *optimization* work, and those are not the same
  thing (the `chart_error_scale` experiment showed accuracy barely predicts cells). A handful
  of cells through `src/benchmark.py` at export time would catch a chart that screens clean
  and solves badly, without waiting for a full grid. Cheap: the export job already loads the
  solver.
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
