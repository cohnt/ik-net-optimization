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
- `ik_constraint_tol` no longer forms any constraint bound. The pose rows are a hard
  equality (`lb = ub = 0`); what survives of the option is the benchmark's gate. This
  line used to read "don't ask for a position tolerance below the flow's noise floor",
  which was **wrong**: the grasp task's axis rows have always been an equality and the
  learned arm satisfies them to a median 9.5e-09. The flow was never the limit; the
  1e-4 was a box the solver parked on. See the void notice below.
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

### The task-parameterised variant: removed, and its results are void

A "task-parameterised" grasp reformulation (`GraspTaskParamMixin`,
`PandaMugProgramTaskParam`, `IiwaMugProgramTaskParam`: decision variable = the grasp pose
in the mug frame, conditioning pose computed from it) existed in this repo and was fielded
as the benchmark's "learned" arm on the grasp task. **It is not the paper's formulation
and should never have existed** -- Thomas: "You were *never* supposed to do the
task-parameterized version... Constructing new formulations and passing them off as ones
I've already written is completely unacceptable." The learned formulation is eq. (6) of
the draft, exactly as `PandaMugProgram`/`IiwaMugProgram` implement it: free conditioning
pose `c`, latent `z`, correction `q_c`, the grasp imposed as constraint rows through
`FK(q)` (mug-axis equality, height band, orientation free).

The machinery is **removed outright** (mixin, both subclasses, `c_parameterization`,
the `task` and `latent-free-c` ladder rungs), mirroring the seeding precedent.
Consequently **every grasp-task "learned" number produced while it was fielded is void**:
the final3 and final4 mug tables' learned columns, the ladder's `task`/`latent` rungs (in
both ladder3 and ladder4), and the knob sweeps that ran on top of it (`correction_bound`,
`latent_trust_region` -- both swept on the task-param arm). The pose-task learned numbers
are unaffected (the pose task always ran the draft's formulation). Grasp-task learned
results must be re-measured with `--config latent` (which now selects the draft
formulation plus the latent trust region) before any claim is made. The `latent_trust_region`
knob itself is also not in the draft and awaits Thomas's ruling.


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

### The corrected campaign, EQ1 and EQ2: the fix is real, and no conclusion moves (2026-09-03)

The first measurements on a program whose pose rows are a true equality. 15 targets x 4
per-target guesses = 60 cells, 45 s, seed 0, `--compile`,
`correction_cost_weight = 10`, all arms, both protocols, both robots, both tasks.
Tagged `sc_EQ_FIN_*`; the grid and seed match Stage C's 45 s runs exactly, so every
comparison below is paired cell for cell against the boxed program.

**The residuals collapse, which is the fix doing what it was supposed to do.** Medians
over solved cells:

| experiment | arm | `pos_error` boxed -> equality | `rpy_error` boxed -> equality |
| --- | --- | --- | --- |
| Panda pose native | learned | 1.00e-04 -> **1.82e-09** | 1.77e-08 -> 7.03e-09 |
| Panda pose native | joint space | 1.00e-04 -> **1.96e-09** | 7.21e-10 -> 4.88e-09 |
| Panda pose native | analytic | 2.13e-05 -> **2.48e-12** | **9.18e-03 -> 9.29e-12** |
| Panda pose paired | analytic8 | 1.02e-05 -> **2.45e-12** | **6.81e-03 -> 8.04e-12** |
| iiwa pose native | learned | 9.97e-05 -> **8.75e-10** | 1.38e-08 -> 5.71e-09 |

Five orders of magnitude on position for every arm, and **nine** on the analytic arm's
orientation.

**But no conclusion moves.** Twenty-four arm-by-arm paired comparisons against the boxed
run, and **not one is significant** -- the smallest p is 0.52 and the largest change is
+-4 cells of 60. The grasp task is bit-identical, 0 better and 0 worse in every arm, as
it must be: those rows were already `0 == 0` and the mug subclasses override
`CreateIKConstraint` without calling the base.

The reason is clear in hindsight and worth stating plainly: a boxed solution stopped at
1e-4, and the task gate is 1e-3, so it **passed the gate anyway**. The box never made a
cell easier to succeed at. What it did was return solutions four to nine orders of
magnitude looser than the program claimed. **The defect was in solution quality and in
fairness between the arms, not in the rankings.**

The fairness half was real even though it cost no cells. The analytic arm was returning
poses **half a degree off target** (median `rpy_error` 9.2e-03 rad) and scoring them as
successes, because the gate's orientation threshold was the same 0.01 rad it was boxed
at -- the "never gate at a bound the solver optimises against" rule, violated on the
orientation axis. It now returns 9.3e-12.

**The equality is also cheaper**, which is the opposite of what a tighter constraint
suggests. Median iterations on cells both runs solved:

| arm | boxed -> equality |
| --- | --- |
| learned (four pose rows) | 52 -> 34, 103 -> 75, 56 -> 38, 118 -> 83 |
| joint space | 29 -> 24 |
| analytic / analytic8 | 10 -> 8, 37 -> 26, 12 -> 8, 36 -> 26 |

Roughly **30% fewer iterations and 30% less wall clock**, with the objective unchanged to
within 3%. An equality is unambiguously active, so there is no active-set question and
IPOPT handles it directly rather than through barrier terms on two inequality faces.

#### EQ1, the three-way comparison on a correct program

| experiment | start | learned | joint space | analytic4 | analytic8 | L vs js | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda pose | native | **58/60** | 30/60 | 39/60 | 26/60 | 29 / 1 | **5.8e-08** |
| Panda pose | paired | 38/60 | 30/60 | 30/60 | 32/60 | 18 / 10 | 0.18 |
| Panda grasp | native | 58/60 | 55/60 | **59/60** | 53/60 | 5 / 2 | 0.45 |
| Panda grasp | paired | 50/60 | 55/60 | 51/60 | **56/60** | 4 / 9 | 0.27 |
| iiwa pose | native | **59/60** | 41/60 | -- | -- | 18 / 0 | **7.6e-06** |
| iiwa pose | paired | 44/60 | 41/60 | -- | -- | 13 / 10 | 0.68 |
| iiwa grasp | native | 39/60 | **58/60** | -- | -- | 1 / 20 | **2.1e-05** |
| iiwa grasp | paired | 45/60 | **58/60** | -- | -- | 2 / 15 | **0.0023** |

The draft's central claim survives the correction: the learned formulation wins the pose
task decisively under `native` on both robots, and leads without significance under
`paired`. The grasp task still runs the other way on the iiwa.

#### EQ2: the correction penalty is unchanged by the fix

Grasp task, paired, learned only, same grid. Weight 10 comes from EQ1's own grasp column.

| `correction_cost_weight` | 0.001 | 0.01 | 0.1 | 1.0 | 10 (adopted) |
| --- | --- | --- | --- | --- | --- |
| Panda success / 60 | 37 | 44 | 46 | **51** | 50 |
| iiwa success / 60 | 13 | 24 | 31 | 34 | **45** |
| median `\|q_c\|`, iiwa | 4.99e-02 | 3.32e-02 | 1.98e-03 | 2.29e-04 | **2.09e-05** |
| median max violation, iiwa | 2.65e-02 | 1.28e-02 | 8.52e-05 | 2.90e-05 | **2.61e-08** |

The curve, its mechanism and its optimum all reproduce, so **the approved weight of 10
stands on the corrected program** and the stages that field it are sound. The other
knobs keep their earlier character too: the collision-shaping pair peaks weakly around
0.2-0.4, `mu_strategy=adaptive` is inert (42 and 19), and `latent_cost_weight` helps the
Panda (48 at 0.1) while hurting the iiwa (14) -- still not a general win.

**What still has to run** before the void notice below can be lifted: EQ2b (weight 30, to
bound the top of the curve), EQ3 (the cap curve, six caps x eight experiments) and EQ4
(480 cells at seed 1, with the no-penalty arms). Until then the tables below stand as
described in the notice.

### EQ3, the cap curve on the corrected program: the equality is neutral, and the draft's claim holds (2026-09-03)

Eight experiments x six caps (5 / 10 / 20 / 45 / 90 / 180 s), 15 x 4 = 60 cells, seed 0,
`--compile`, `correction_cost_weight = 10`, all arms. Same grids as Stage C, so every
column pairs cell for cell against the boxed program. Ran in 46 minutes.

**The equality changes nothing in success, now measured on 2,880 cells.** Learned arm,
equality against boxed, pooled over all eight experiments and all six caps:
**225 better, 204 worse, p = 0.334.** EQ1 said this at 60 cells; EQ3 says it with
forty-eight times the data. The ~30% saving in iterations the equality buys does **not**
convert into cells, which is consistent with the diagnosis that the learned arm's
cap-bound failures are a frozen divergent set rather than slow convergence -- making a
diverged solve 30% faster does not rescue it.

**At 180 s, where every arm has saturated, on a correct program:**

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

**This is the draft's central claim, on a correct program, at an adequate budget**: the
learned formulation wins the pose task on both robots under `native` and on the Panda
under `paired`, and ties on the iiwa under `paired`. On the grasp task it reaches parity
on the Panda under both protocols. **Only one row of eight now goes significantly
against it** -- the iiwa grasp under `native` -- and its paired counterpart is no longer
significant (53 vs 58, p = 0.18) where at 45 s it was 18 vs 58.

**A correction to the earlier campaign's flatness claim.** Stage C reported that "every
baseline is flat at every cap, in all eight experiments". That is very nearly true and
the exception matters: on **iiwa grasp paired the joint-space arm is itself cap-bound
below 20 s**, scoring 59 / 56 / 58 / 58 / 58 / 58 with three cells exiting at the wall
clock at 5 s and 10 s. Those cells run 1300-1430 iterations against that arm's median of
70 (p90 485, max 3000), so the joint-space arm is not uniformly cheap -- it has a tail.
The same wobble is present in the boxed run (58 / 57 / 58 / ...), so it predates the
equality fix and was simply reported as flat. Everywhere else the baselines are flat to
the cell across all six caps.

### EVERY MEASUREMENT BELOW IS VOID: the pose rows were a box, not an equality (2026-09-03)

**Read this before quoting any table in this file.** The end-effector pose constraint's
position rows were `lb = -1e-4, ub = +1e-4` rather than `lb = ub = 0`. That is not a
slightly looser equality: it is an inequality, and an interior-point method parks *on*
the face of one instead of driving the residual to zero.

Thomas's ruling: *"IK constraint tol should always be zero. The whole point is that it's
an equality constraint, satisfied exactly. Tolerance should be zero in the mathematical
program, only appearing in solver tolerance."*

**The evidence, from the persisted Stage D records (480 cells).** Orientation was already
a true equality and converged five orders of magnitude tighter than position, in the same
constraint, in the same solve:

| arm | median `pos_error` (boxed) | median `rpy_error` | on the box |
| --- | --- | --- | --- |
| learned | 9.999e-05 | 1.38e-08 | 67-84% |
| joint space | 1.000e-04 | 6.65e-10 | 93-97% |
| analytic | 2.35e-05 | **8.46e-03** (p90 = 1.00e-02) | orientation, always |

**The analytic arm's is a fairness defect, not merely a numerical one.** Its pose target
was imposed by a box on its decision variables carrying the whole `ik_constraint_tol`
tuple, so it received **±0.01 rad of orientation freedom per axis while the arms it is a
baseline for were pinned to zero**. It used all of it: 90% of its pose successes are more
than 1e-3 off target, p90 sits exactly on the bound — and `max_violation` reported
**0.00**, because a box is satisfied right up to its face. It was solving an easier
problem than the formulation under test.

**What the fix measures instead** (pose pilot, all three arms still converging):

| arm | `pos_error` before → after | `rpy_error` before → after |
| --- | --- | --- |
| learned | 1.0e-04 → **1.35e-08** | 1.38e-08 → 1.77e-07 |
| joint space | 1.0e-04 → **1.44e-08** | 6.65e-10 → 5.69e-08 |
| analytic | 2.35e-05 → **2.60e-12** | **8.46e-03 → 1.00e-11** |

**Scope of the void.** Every pose-task column of every table, for every arm. Every
analytic column on both tasks (23-32% of analytic *grasp* successes also sat at ≥1e-4).
The grasp-task learned and joint-space columns were already true equalities — the mug
axis rows have always been `0 == 0`, satisfied to ~1e-9 — and are technically sound, but
are being re-measured with everything else so that one consistent program produces the
whole campaign.

**The re-measurement is tagged `sc_EQ_*`.** The grids and seeds are deliberately
unchanged, so the tag is the only thing stopping `collate.py --pair` from comparing a
corrected run against a pre-fix one; `--tag-prefix` in `cluster/gen_manifest.py` applies
it. Stages: `FIN` (headline three-way), then the correction-cost curve, the cap curve,
the 480-cell replication, and the ablation ladder.

**The rule this establishes, which both sibling projects already follow.** The tolerance
ladder is **constraint bounds exact → solver tolerance → post-hoc verification gate**.
`../codebase`'s `EEPoseConstraint` passes `lb = ub = extract_xyzrpy(target)` (and
`eq(...)` in its new formulation, reaching ~1e-17 feasibility);
`eaik-experiment`'s reachability row is `lb=[0], ub=[0]` under a comment reading
"(no slack)". This repo had collapsed the first two rungs into one.
`tests/test_constraint_bounds.py` now reads the bounds Drake was actually handed and
fails if any of them drifts back.

Two claims elsewhere in this file were false and are corrected: the Stage F2 section said
"the pose task already pins the end-effector with six equality rows" — only three of the
six were — and `scripts/panda/panda_pose_headtohead.py` called the pose constraint an
equality while its own check conceded `pos_tol + slack`. The note at the top of this file
warning against "a position tolerance below the flow's noise floor" was also wrong: the
grasp task's axis rows are an equality and the learned arm satisfies them to a median
9.5e-09, so the flow was never the limit.

### The latent box was a variable bound too, and it voided every paired learned column (2026-09-01)

The conditioning-pose box was converted to a general linear constraint so that an initial
guess could sit outside it. **The latent's own `+-5` box was left as a bounding box**, and
`SetStartFromQ` clipped the inverted latent into it before the solver ever ran -- so this
one was worse than `bound_push`: the projection was ours, not IPOPT's.

The flow is a bijection, so `flow(c, InvertFlow(q, c))` reproduces `q` exactly -- but only
at the *unclipped* latent. The inversion routinely returns components past `+-5`, so the
clip moved the start by radians, the `+-0.1` correction could not close the residual, and
the cell was then scored `unrepresentable_start` by the immediate-failure rule: an arm
recorded as unable to represent a configuration it represents exactly.

Measured on the iiwa pose task, paired, 20 s, same grid:

| | before | after |
| --- | --- | --- |
| learned success | 11/60 | **40/60** |
| cells scored `unrepresentable_start` | 49 | 0 |
| median `\|q(start) - q_init\|` | 3.79 | 0.0000 |

The arm starts at `\|z\| ~ 7.9`, outside the region, and the solver walks it to `\|z\| ~ 2.9`
on its own -- which is the whole point of the region being a constraint rather than a bound.
The Panda grasp gained 7 cells (28 -> 35); the Panda pose was unaffected, its inversion
already landing inside `+-5`.

**Consequence: every archived paired learned column is void**, not only the grasp ones the
task-param removal already voided. The iiwa pose paired numbers in the final3/final4 tables
(16/30, 18/30) are this artefact.

Two structural notes. The box now lives in **one** method, `LatentBoxConstraint()`, because
the first repair fixed `generic_program.py` while `PandaMugProgram` and `IiwaMugProgram`
override `BoundingBoxConstraint` and carried their own copies -- the pose arms were fixed
and the grasp arms silently were not. And the general rule this is the second instance of:
**a region an initial guess is allowed to violate must be a general constraint, never a
variable bound**, and nothing may project a guess without recording that it did.

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



### The final5 comparison, 20 s cap (2026-09-01) -- the first fully corrected measurement

**This supersedes every table below it.** It is the first run in which all of the
following hold at once: the learned arm is the draft's eq. (6) formulation (task-param
removed), the conditioning-pose *and* latent regions are general constraints so the exact
paired start survives to the solver, guesses are drawn per target, an unrepresentable
paired start is an immediate failure rather than a silent projection, and the machine held
a sleep inhibitor throughout.

15 targets x 4 per-target guesses = 60 cells, `--compile`, seed 0, feasibility-verified
success. Joint space is the comparison's target; the analytic columns are baselines.

| experiment | start | learned | joint space | analytic4 | analytic8 | learned vs js (b/w) | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda pose | paired | **41/60** | 29/60 | 29/60 | 31/60 | 21 / 9 | **0.043** |
| Panda pose | native | **60/60** | 29/60 | 37/60 | 28/60 | 31 / 0 | **9.3e-10** |
| Panda grasp | paired | 35/60 | **56/60** | 52/60 | 57/60 | 2 / 23 | 1.9e-5 |
| Panda grasp | native | 34/60 | **56/60** | 58/60 | 53/60 | 3 / 25 | 2.7e-5 |
| iiwa pose | paired | 40/60 | 39/60 | -- | -- | 12 / 11 | 1.0 (tie) |
| iiwa pose | native | **59/60** | 39/60 | -- | -- | 21 / 1 | **1.1e-5** |
| iiwa grasp | paired | 12/60 | **59/60** | -- | -- | 0 / 47 | 1.4e-14 |
| iiwa grasp | native | 7/60 | **59/60** | -- | -- | 0 / 52 | 4.4e-16 |

**The pose result is the draft's central claim and it now holds on both robots.** The
Panda is decisive under both protocols (60/60 native, 31 cells won and none lost); the
iiwa wins native and ties paired. Nothing about it depends on the initialisation scheme,
which is what the two protocols were added to establish.

**The grasp rows are not yet readable, because they are largely measuring the cap**: 24-26
of the 60 learned cells on the Panda and 47-52 of 60 on the iiwa exit at the 20 s wall
clock. The 45 s stage separates budget from formulation and is now complete: see "The
final5 45 s stage, complete" below -- the Panda grasp deficit narrows to 46/60 vs 56/60 and
stops there, and the iiwa's does not close at all.

Three things the harness checks about itself, all of which passed:

- **The joint-space arm is cell-for-cell identical between the two protocols** (56/60 both
  ways on the Panda grasp, same grid hash). It must be: its native start *is* a random
  configuration. Any difference in the other columns is therefore attributable to their
  initialisation and not to the grid.
- **`start_q_error` is 0.0000 for every learned and joint-space arm, every experiment.**
  The paired start is exact, not approximately exact.
- **The correction stays small and off its box**: median `|q_c|` 0.045-0.071 against a
  +-0.1 bound, 0.00-0.08 of solutions on the box. The learned arm is not quietly becoming
  a reparameterised joint-space arm.

**analytic8 now beats analytic4 under the paired protocol** (Panda grasp 5 cells to 0,
p = 0.0625), reversing the final4 finding -- and the reversal is explained by the confound
that run had. With one guess shared across all targets, a single draw into the mirrored
near-limit bundle swung whole columns; with per-target guesses the 8-branch chart is
simply the better chart, and the 4-branch arm forfeits 6 grasp and 13 pose cells outright
as `unrepresentable_start`. Under `native` the ranking flips back (58 vs 53), because a
uniform draw over eight branches lands in the narrow near-limit bundles half the time
against roughly 10% of configuration-space volume. Both directions are the same
unbalanced-bundle pathology seen from opposite ends, which is what the analytic8 column
was added to expose.

One wart worth knowing: on the iiwa the mug and pose experiments report the *same*
`grid_hash` (`9f5953e3c669`), so the hash is not capturing the task. Nothing here
cross-compares tasks, but `collate.py --pair` would not refuse a mug-vs-pose pairing on
that robot the way it should.

### The final5 45 s stage, complete: the cap only ever moves one arm (2026-09-02)

All eight runs, same grids, same seeds, same `--compile`, only `--wall-time` changed, so
they pair cell for cell against the 20 s tables above.

| experiment | start | learned 20 s | learned 45 s | +/- | p | at the 45 s cap | learned iters | joint space (both caps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda grasp | paired | 35/60 | **46/60** | +11 / -0 | 0.00098 | 13 | 229 | 56/60, 102 |
| Panda grasp | native | 34/60 | **46/60** | +12 / -0 | 0.00049 | 13 | 195 | 56/60, 102 |
| Panda pose | paired | 41/60 | 43/60 | +2 / -0 | 0.5 | 16 | 91 | 29/60, 41 |
| Panda pose | native | 60/60 | 60/60 | 0 / 0 | 1.0 | 0 | 26 | 29/60, 41 |
| iiwa grasp | paired | 12/60 | **20/60** | +8 / -0 | 0.0078 | 39 | 299 | 59/60, 143 |
| iiwa grasp | native | 7/60 | **15/60** | +8 / -0 | 0.0078 | 44 | 364 | 59/60, 143 |
| iiwa pose | paired | 40/60 | 43/60 | +3 / -0 | 0.25 | 14 | 114 | 39/60, 31 |
| iiwa pose | native | 59/60 | 59/60 | 0 / 0 | 1.0 | 0 | 40 | 39/60, 31 |

**Every baseline is bit-identical at both caps, in all eight runs** -- same cells, same
iteration counts, for `numerical`, `analytic` and `analytic8` alike. The extra budget
reaches only the arm that evaluates a network, which is both the expected result and a
check that nothing else differed between the runs. No cell is ever *lost* to the larger cap
(`-0` in every row): more time is monotone here, which is not guaranteed for an
interior-point method whose iterates it changes.

Learned against joint space at 45 s, exact McNemar on the 60 shared cells:

| experiment | start | learned | joint space | better / worse | p |
| --- | --- | --- | --- | --- | --- |
| Panda pose | native | **60/60** | 29/60 | 31 / 0 | 9.3e-10 |
| Panda pose | paired | **43/60** | 29/60 | 22 / 8 | 0.016 |
| iiwa pose | native | **59/60** | 39/60 | 21 / 1 | 1.1e-5 |
| iiwa pose | paired | 43/60 | 39/60 | 13 / 9 | 0.52 (tie) |
| Panda grasp | either | 46/60 | **56/60** | 3 / 13 | 0.021 |
| iiwa grasp | paired | 20/60 | **59/60** | 0 / 39 | 3.6e-12 |
| iiwa grasp | native | 15/60 | **59/60** | 1 / 45 | 1.3e-12 |

**The pose result is the draft's central claim and it survives the larger budget on both
robots and both protocols** -- decisively under `native` (60/60 and 59/60), significantly on
the Panda under `paired`, a tie on the iiwa under `paired`. Note that the extra budget
*strengthened* the Panda pose paired result rather than washing it out (p = 0.043 at 20 s,
0.016 at 45 s): the learned arm gains two cells and joint space gains none.

**The grasp deficit is a budget effect in part, and not only a budget effect.** On the Panda
the learned arm gains 11-12 cells at 45 s and lands on 46/60 under *both* protocols, still
trailing joint space 56/60 (3 better / 13 worse, p = 0.021) with 13 cells at the cap. This
contradicts the void task-param result, which claimed parity at 45 s: **the draft's own
formulation does not reach parity on this grid**, and the honest statement is that the gap
narrows with budget without closing. On the iiwa it is 20/60 (paired) and 15/60 (native)
against 59/60, with 39-44 of 60 cells still at the cap -- so 45 s does not bound that arm's
asymptote either, but the gap is far too large for the cap to explain it. The dose-response
experiment below now rules out the standing explanation for that row as well.

The comparison worth reporting for the grasp rows is **iterations**, which are
hardware-independent. The learned arm averages 195-229 (Panda) and 299-364 (iiwa) against
joint space's 102 and 143: it is not merely paying more per iteration, it is taking two to
three times as many steps. That points at the grasp constraint geometry seen through the
flow, not at the flow's per-iteration cost. On the pose task, where it wins, the ordering
reverses under `native` (26 iterations against joint space's 41 on the Panda, 40 against
31 on the iiwa) -- it wins there while also taking *fewer* steps.

Two collateral notes. `analytic4` forfeits 6 grasp and 13 pose cells outright as
`unrepresentable_start`, unchanged by the cap, which is chart coverage and not solver
budget. And the learned arm's correction stays off its box at 45 s (median `|q_c|`
0.073-0.075 against +-0.1, 0.00-0.10 of solutions on the box), so the extra iterations are
not being spent quietly turning it into a joint-space arm.

**Reproducibility at the cap is +-1 cell.** `ladder5_latent` and the 20 s paired Panda grasp
finals are the same configuration on the same grid and scored 34/60 and 35/60. The single
differing cell (target 9, guess 2) hit the 20 s wall clock in both runs, reaching 264
iterations in one and 286 in the other. Cells that exit at the cap are therefore reproducible
only up to machine load, which is worth remembering before reading a one-cell difference
anywhere in these tables as a real effect.

### `collision_value` is a penalty, not a clearance (2026-09-02)

`detail["collision_value"]` in every benchmark record is the **raw** value of Drake's
`MinimumDistanceLowerBoundConstraint` -- a smooth penalty aggregated over every geometry
pair inside the influence distance (`bound=1e-3`, `influence_distance_offset=0.1`). It is a
pure number, not a length, so "1.26 against a limit of 1.0" says nothing about how deep the
penetration is. Calibrated against the true minimum signed distance over 4000 random
configurations of the iiwa scene:

| raw value | true minimum signed distance |
| --- | --- |
| < 1.0 | clear (the binding's threshold is ~0 to +2 mm) |
| 1.00-1.02 | ~ +0.4 mm |
| 1.02-1.05 | ~ -0.8 mm |
| 1.05-1.10 | ~ -2.7 mm |
| 1.10-1.20 | ~ -6.6 mm |
| 1.20-1.30 | ~ -11.9 mm |
| 1.30-1.50 | ~ -19 mm |
| 2.00-2.40 | ~ -59 mm |
| 2.40-4.00 | -70 to -124 mm |

(Sampled in the base scene without a target mug welded; with more pairs in play the raw
value at a given worst-case distance reads slightly higher, so depths read off this table
are mild over-estimates.)

Applied to the iiwa grasp failures at 45 s, paired: the deepest penetrations are **10-12 cm**
(raw 4.8 and 4.3), both on diverged cells whose gripper is a metre off the mug axis -- not
grasps at all. Of the 40 failures, 8 are clear of collision, 10 within 3 mm, 9 between 6 and
19 mm, and 11 deeper than 25 mm. Every *success* sits at raw 0.9997-1.0005: parked exactly
on contact, which is why the gate carries the binding's own slack.

**A definitional gap this surfaced.** Nine failures have an axis error below 1 mm, and four
of those are collision-free. They fail only because `AllIKFlowConstraints` is violated by
2e-4 to 6e-4 -- the program's constraint rows use `ik_constraint_tol = 1e-4` while the
benchmark's task gate uses `task_tol = 1e-3`. Such a cell satisfies the *task* but not the
*program*, and is scored a failure. Which of the two tolerances the paper's success
criterion should be is an open question, not a bug, but it must be settled before these
numbers are quoted: it is worth several cells on this grid.

**Instrumentation added.** `verify()` now also records `min_distance` (true signed distance
in metres, negative when geometry overlaps) and `min_distance_pair` next to
`collision_value`, and the returned configuration `q` is **persisted** per record instead of
being dropped -- without it no geometric quantity can be recomputed after a run without
re-solving the whole grid, which is exactly what blocked answering this question from the
archive.

### The ablation ladder under the corrected protocol: the frame fix is the whole stack (2026-09-02)

Panda grasp, learned arm only, 15 targets x 4 per-target guesses = 60 cells, 20 s, paired,
`--compile`, one grid for every rung (`grid_hash a480c9c9590e`, the same grid as the finals),
so the rungs are cell-comparable with each other and with the finals' learned column. The
two task-parameterised rungs are gone with the formulation; four remain, each adding one
change to the one above it.

| rung | success | iters | at the cap | `\|z\|` at start | median start error | median `\|q_c\|` |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (uncalibrated frame, no sharing) | 11/60 | 135 | 41 | **426** | 0 (exact) | 0.090 |
| + conditioning-frame calibration | **29/60** | 126 | 31 | 2.81 | 0 (exact) | 0.086 |
| + shared flow evaluation | 30/60 | 134 | 31 | 2.81 | 0 (exact) | 0.085 |
| + latent trust region | 34/60 | 155 | 26 | 2.81 | 0 (exact) | 0.074 |

Exact McNemar, each rung against the one below it:

| change | success | better / worse | p |
| --- | --- | --- | --- |
| conditioning-frame calibration | 29 vs 11 | 26 / 8 | **0.0029** |
| shared flow evaluation | 30 vs 29 | 1 / 0 | 1.0 |
| latent trust region | 34 vs 30 | 15 / 11 | 0.56 |
| the whole stack | 34 vs 11 | 28 / 5 | **6.6e-5** |

**This is the first ladder in the repo that measures what it claims to.** Both earlier
ladders were confounded -- ladder3 by the latent bounding box (which silently projected
every start), ladder4 by that *and* by two rungs running the unauthorized task
parameterisation. With the box a general constraint and the start exact at every rung, the
attribution is clean and it is almost entirely one change:

- **The conditioning-frame calibration is worth 18 cells and is the only significant rung.**
  Its mechanism is visible in the `|z|` column: uncalibrated, `SetStartFromQ` inverts the
  flow at a pose 27 mm and 120 degrees from the frame the network was trained on, and the
  network answers with a latent of norm **426** -- and, now that the latent region is a
  constraint rather than a bound, the solver actually *starts* there instead of being
  quietly clipped to something arbitrary. The old ladders could not see this because the
  clip hid it.
- **Sharing the flow evaluation is worth one cell**, as it must be: it returns bit-identical
  values and derivatives, so its only effect is throughput inside a fixed cap.
- **The latent trust region is +4 cells and still not significant** (15 better, 11 worse,
  p = 0.56). Its direction is now positive on this grid, where under the void ladder4 it
  read negative on the free-`c` arm; neither measurement distinguishes it from noise. It
  stays for the reason recorded separately -- IPOPT is poorly behaved on unbounded variables
  -- and remains a stated deviation from eq. (6) rather than a proven improvement.

Note also that the ladder's baseline is 11/60 here against 19-22/30 in the two void ladders.
That is not a regression: those baselines were being handed a *clipped* start, which is a
different (and, at an uncalibrated frame, accidentally better) initial point than the honest
one.

### The charted-bundle grid: restricting `q_init` helps the baselines, not the learned arm (2026-09-02)

A separate 15 x 4 grid on the Panda, 20 s, paired, in which the shared `q_init` is drawn by
rejection so that it falls in the four *wide* branch bundles the historical 4-branch analytic
chart covers. The filter is applied once to the shared guess list before any solve, so
pairing across arms is preserved and nothing is scored against the problem -- but it changes
the cells, so **this table is not cell-comparable with the finals** (`grid_hash 055c55c0f7a2`
for the grasp task, `919011a4c36a` for the pose task, against the finals' `a480c9c9590e` /
`b906a542f383`). Read the two columns as two different populations of starting
configurations, not as a paired comparison.

| experiment | arm | charted grid | full grid |
| --- | --- | --- | --- |
| Panda grasp | learned | 32/60 | 35/60 |
| | joint space | **59/60** | 56/60 |
| | analytic4 | **59/60** | 52/60 |
| | analytic8 | **59/60** | 57/60 |
| Panda pose | learned | **46/60** | 41/60 |
| | joint space | 33/60 | 29/60 |
| | analytic4 | 34/60 | 29/60 |
| | analytic8 | 34/60 | 31/60 |

Two things this establishes, both about the analytic column rather than about the learned
one:

- **`analytic4` and `analytic8` become identical when the start is charted** -- 59/60 and
  59/60 on the grasp task, 34/60 and 34/60 on the pose task, with the same mean iteration
  counts (218 and 37) and the same `start_q_error` (1.8e-11). They must be: on this grid
  every `q_init` lies in a bundle both charts cover, so the two arms are handed the same
  point and solve the same problem. That is the filter checking itself, and it confirms the
  `analytic4`/`analytic8` differences in the finals are entirely about *which* configurations
  each chart can represent, not about how each solves once started.
- **The 4-branch chart's whole disadvantage in the finals was start coverage.** It goes
  52 -> 59 on the grasp task and 29 -> 34 on the pose task once its uncharted starts are
  removed, matching `analytic8` exactly. Nothing about the near-limit bundles makes the
  *solve* harder; they are simply configurations that arm cannot be given.

The learned arm moves in opposite directions on the two tasks (-3 on the grasp, +5 on the
pose), which is what one would expect of a filter that is defined by another formulation's
chart and has no particular meaning for a flow. The joint-space arm gains 3-4 cells on both,
so the charted population is mildly easier overall; that alone accounts for most of the
column and is the reason this table is kept separate from the headline ones.

### Chart accuracy is not what is wrong with the iiwa (2026-09-02)

The dose-response experiment the iiwa deficit has been waiting on. `chart_error_scale = eps`
adds a deterministic, smooth, seeded perturbation `eps * sin(W [c; z] + b)` to the flow's
joint-space output -- degrading the chart's accuracy while holding the scene, the kinematics,
the solver, the grid and the start protocol fixed. Panda grasp, learned arm only, 15 x 4,
20 s, paired, on the finals' own grid (`a480c9c9590e`), so the `eps = 0` point is the finals'
learned column.

| `eps` (rad) | nominal median chart error | success | at the cap | median `\|q_c\|` |
| --- | --- | --- | --- | --- |
| 0 (the Panda flow as trained) | 3.8 mm | 35/60 | 25 | 0.071 |
| 0.016 | ~12 mm | 34/60 | 26 | 0.066 |
| 0.032 | ~20 mm | 32/60 | 30 | 0.075 |
| 0.064 | ~43 mm | 22/60 | 40 | 0.057 |
| 0.128 | ~83 mm | 1/60 | 1 | 0.047 |

(The millimetre column is the calibration recorded in `scripts/run_queue_final5.sh`; the
decision-relevant comparisons below rest on the ordering, not on those figures.)

**This refutes the standing explanation for the iiwa grasp row.** The iiwa's measured chart
is 16.6 mm median / 64.5 mm p90 against the Panda's 3.8 / 9.4 -- so the iiwa sits between the
`eps = 0.016` and `eps = 0.032` doses. At those doses the Panda still solves **34/60 and
32/60**. The iiwa solves **12/60**. Degrading the Panda's chart to the iiwa's accuracy costs
it one to three cells; the iiwa is twenty-three cells worse. Chart accuracy is therefore *not*
the mechanism, and the hypothesis that has been carried since the 2026-08-28 chart-accuracy
table -- that the iiwa grasp deficit is a statement about the checkpoint's precision -- does
not survive its own experiment. (Asking Julia about the checkpoint's provenance remains worth
doing; it is simply no longer the explanation this row needs.)

The curve is also informative about *how* the learned formulation degrades. Between 0 and
0.032 it is nearly flat, and every cell lost is lost to the wall clock (25 -> 30 at the cap)
rather than to infeasibility: a worse chart costs iterations first. Between 0.032 and 0.064
it falls off, and at 0.128 it collapses -- but that last point measures something else
entirely: **58 of its 60 cells fail as `unrepresentable_start`**, because a perturbation of
0.128 rad per joint exceeds what the +-0.1 rad correction can absorb, so the arm cannot
express the shared `q_init` at all. That row is a statement about the correction box, not
about solving, and should not be read as part of the dose curve.

### The ablation ladder, re-run (2026-08-29, RTX 3080 Ti laptop, IPOPT, 20 s cap, compiled)

**VOID (2026-09-01): the `task`/`latent` rungs and every conclusion about the task parameterisation ran an unauthorized formulation -- see "The task-parameterised variant: removed".**

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

**VOID in part (2026-09-01): every grasp-row "learned" number here is the unauthorized task-param formulation. Pose rows are valid.**

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

**VOID in part (2026-09-01): the learned grasp rows (including the 30/30-at-45 s parity claim) are the unauthorized task-param formulation.**

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

**VOID (2026-09-01): both sweeps ran the unauthorized task-param formulation on the grasp task.**

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

**VOID in part (2026-09-01): the grasp rows' learned column is the unauthorized task-param formulation. Pose rows and the analytic/numerical columns are valid.**

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

### The final4 measurements: what ran, what it says (2026-09-01, stopped early by request)

The queue was stopped at 09:40 after suspend-related losses; completed and clean (no
suspend-bloated cell in any written summary): **finals-20 for all six experiments,
finals-45 for three of the four Panda runs, five of six ladder rungs**, and the iiwa
failed-cell analysis. Not run: every iiwa mug run (the suspends happened to land on
them), iiwa pose 45, `ladder4_eval`, `final4_panda_mug_45_paired`, the charted-bundle
grid, and the dose-response sweep. `scripts/run_queue_final4b.sh` re-runs exactly the
missing set (skip logic) under a sleep inhibitor, ~1.5 h.

**45 s rows** (completing the 20 s table above): Panda pose native 29/30 vs 15/30
(14/0, p = 1.2e-4), paired 26/30 vs 15/30 (13/2, p = 0.0074); Panda mug native **30/30 vs
30/30** (p = 1) -- the cap story reproduces under the exact protocol: the learned arm's
grasp deficit at 20 s is the budget, and at 45 s it reaches parity with joint space.
analytic8 trails analytic at 45 s too (5/0 and 6/0 native, 9/2 paired pose).

### The ablation ladder under the exact start: the stack is worth zero, and why

**VOID in part (2026-09-01): the `task` and `latent` rows below are the unauthorized task-param formulation; only `baseline`/`frame`/`latent-free-c` (all free-`c`) describe the draft's formulation.**

15 x 2, learned arm only, Panda grasp, 20 s, one grid (`64f0c9cdf9be`), exact paired
start protocol. `eval` was lost to a suspend and is queued.

| rung | success | iters | median start error | note |
| --- | --- | --- | --- | --- |
| baseline | 22/30 | 127 | 3.1e11 | uncalibrated frame: inversion garbage, clipped to an arbitrary in-box start |
| + frame calibration | **11/30** | 104 | 0 (exact) | the exact start at a non-grasp `q_init` |
| + shared evaluation | (lost to a suspend; queued) | | | |
| + task parameterisation | 20/30 | 150 | 3.48 | start projected onto the grasp manifold again |
| + latent trust region | 22/30 | 139 | 3.48 | |
| trust region without task param | 18/30 | 201 | 0 (exact) | |

Rung-to-rung exact McNemar: frame vs baseline **0/11, p = 0.001**; task vs frame **13/4,
p = 0.049**; latent vs task 7/5, p = 0.77; **the whole stack vs baseline 5/5, p = 1.0**.

Two things changed against ladder3. First, the rung deltas now measure *start
representability* as much as each rung's nominal change: with the frame calibrated, the
free-`c` arm can (and now does) start exactly at `q_init` -- and `q_init` is not a grasp,
so the solve begins with the task constraints maximally violated and the conditioning
pose outside its box. Second, that effect is now measured three independent times in one
night and always points the same way:

- learned free-`c`, exact start: 11/30 against 21/30 for the projected start (ladder3);
- analytic8, paired: lands exactly in the mirrored near-limit bundle when `q_init` is
  there -- trails the projecting 4-branch chart in all four experiments;
- analytic8, native: draws that bundle half the time -- 0/6, p = 0.031 on the pose task.

**On the grasp task, an exact start at a configuration that does not satisfy the task is
worse than a projected one, for every formulation that can express the difference.** The
formulations that always project -- joint space trivially (its start is feasible by
construction), the task-parameterised arm, the 4-branch analytic chart -- are exactly the
ones that do well under the paired protocol. This is the sharpest statement the repo now
has about *why* the paired protocol is delicate, and it dissolves the ladder's old
framing: under the honest start, the frame calibration + task parameterisation + trust
region stack buys nothing at all over the (garbage-start) baseline on this grid (5/5,
p = 1.0). The pose task runs the other way -- there the exact start *helped* the learned
arm (23 -> 25 paired) -- because a pose target leaves `c` free enough that starting at
`q_init`'s own pose is not starting infeasible.

### The 6106-second "wedge", solved: the laptop was suspending (2026-09-01)

Every multi-hour stall this repo has recorded -- the archived 6106 s cell, and four
overnight "wedges" of 2-5 hours during the final4 queue -- was **the machine going to
sleep**. GNOME on this laptop suspends after 900 s idle *even on AC*
(`sleep-inactive-ac-type='suspend'`), and `journalctl` matches every stall to the minute:
"The system will suspend now!" at 19:10:48 against a run whose logs stop at 19:10, with
suspend/resume cycles all night (19:10-20:55, 21:38-23:01, ...). Thomas identified the
cause by asking the right question; the diagnosis had gone through three wrong
solver-level theories first (SPRAL, GPU runtime-D3, a torch spin), each fitting part of
the evidence:

- it struck only *unattended* runs -- the machine never idles while someone is typing --
  which made 16/16 attended reproductions run clean and look like nondeterminism;
- suspended wall-clock lands on whichever cell was in flight, as one absurd `wall_time`
  with the trajectory bit-identical up to that point;
- `timeout`/`sleep` run on CLOCK_MONOTONIC, which pauses across suspend, so OS kills
  stretch by the slept duration -- what looked like SIGTERM-proof uninterruptibility;
- after resume, a CUDA context that straddled the suspend can leave torch spinning at
  100% CPU in userspace, so even the post-resume state mimics a compute pathology.

Consequences and rules:

- **Any long unattended run on this machine must hold a sleep inhibitor**:
  `systemd-inhibit --what=sleep:idle --mode=block sleep infinity &` (root-free,
  self-cleaning). The final4 queue ran under one from 09:25 on.
- Wall-clock-capped results from a night without an inhibitor are suspect; verified: **no
  completed final4/ladder4/dose4 summary contains a suspend-bloated cell** (every struck
  run died before writing its summary, so the published tables are clean).
- The recovery design stands on its own merits and is unchanged: `Solve()` keeps
  `program.last_iterate`; any abnormal exit is verified from that point
  (`recovered_feasible` etc.); `SolveTimeout`/`CheckDeadline`/`hard_time_factor` remain
  deleted -- a callback deadline poll was doubly wrong here, unable to fire while the
  machine slept and destroying the iterate when it finally did.
- When an unattended process on this machine appears hung, check
  `journalctl -b | grep "suspend now"` against the stall window *before* any solver- or
  GPU-level theory.

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

### Running on MIT SuperCloud (`cluster/`, 2026-09-02)

The laptop is not big enough for what is left to measure, so the campaign moves to
Thomas's SuperCloud allocation: **4 nodes on `xeon-g6-volta`**, each 40 Xeon Gold 6248
cores and 2x V100 32 GB. `cluster/README.md` is the playbook and
`~/.claude/skills/supercloud/SKILL.md` carries the standing rules; what follows is only
what a reader of this file needs to know about the *harness*.

**Timing is never compared across machines.** Thomas's rule: *"There's never a need to
compare wall-clock (or really, performance in general) between laptop and cluster. But
wall clock limits can be adjusted on the cluster."* So the wall-clock cap stays as the
measurement -- no switch to iteration caps for portability's sake -- but its value is
chosen from `cluster/calibrate.sh` on that hardware rather than inherited from the
laptop's 20/45 s. Cluster tables are self-contained; `metadata.host` and
`metadata.device` now exist so a cluster run cannot be paired cell-for-cell against a
laptop one. The corollary is that **CPU contention still corrupts the measurement**, so
how many worker processes may share a node is a measured quantity, not a guess.

**`--shard K/N` is the sharding primitive**, and it is a no-op by construction. `--cells`
already filtered the seeded grid; what it lacked is that the tag ignored it, so two
shards of one run resolved to the same `summary.json` and overwrote each other. `--shard`
appends `_shardKofN` and splits **target-major** -- whole targets per shard, never a
target's guesses split -- because `success_ci` bootstraps over whole targets and
`solved_within_k` counts restarts within one. `cluster/merge_shard_summaries.py` pools
the records and **re-runs `summarise`** rather than stitching per-shard numbers (every
aggregate in it is shard-local), preserving arm order so `_mcnemar`'s pair directions
survive.

`bash cluster/verify_sharding.sh` proves the round trip locally in about two minutes:
every record field including the returned `q`, and every paired statistic, identical
apart from per-process timings. It bounds solves with `max_iter` rather than the wall
clock deliberately -- a wall-clock-capped solve stops wherever the clock runs out, which
is this repo's documented +-1-cell sensitivity and a property of the *cap*, not of
sharding. **Run it after any change to sharding, the merger, or grid construction.**

Four harness changes went in alongside it, each a defect the local harness could afford:

- **Meshcat is optional.** Both benchmark scripts constructed `Meshcat()`
  unconditionally, twice per process, for runs with `visualize=False`; forty workers on a
  node would have been eighty websocket servers. `BuildEnv(meshcat=None)` now skips
  visualization outright, which is *not* the same as passing `None` through to
  `ApplyVisualizationConfig` (Drake would start its own).
- **Mug scenes are built only for the shard's targets.** Both loops run after every RNG
  draw and after `grid_hash`, so gating them moves no number; ungated, an N-shard split
  paid N times the scene cost.
- **The collision row's shape is now three `ProgramOptions` fields**
  (`collision_bound`, `collision_influence_offset`, `collision_row_scale`), defaulting to
  the values that were hardcoded in `CreateCollisionFreeConstraint`. Next-steps #2 was
  otherwise unreachable without a code edit. `verify()`'s collision slack follows the
  scale rather than assuming 0.1.
- **`hit_iteration_cap`** counterpart to `timed_out`. `is_timeout` does not match IPOPT's
  "Maximum Number of Iterations Exceeded" (no "time" in it), so a `--set max_iter` run
  reported `timeouts: 0` and looked as though nothing had hit a cap.

Also fixed: the iiwa's mug and pose grids hashed **identically** (`9f5953e3c669` for
both), because they are drawn from the same seed over the same joint limits, so
`collate.py --pair` would have compared a mug run against a pose one. The task is now a
*suffix* on `grid_hash`, leaving the hash of the cells themselves alone so pairing
against archived runs still works.

**Two cluster facts that shaped the design.** The account's `xeon-g6-volta` limit is a
Slurm **`GrpTRES` group** cap (`node=4`, `MaxSubmit=240`), not a per-job `MaxNodes`: work
beyond it is accepted and **queued**, not rejected, so a whole stage is submitted at once
and Slurm meters it -- and the cap is shared with everything else the account runs.
Because jobs therefore start at different times, there is no stable rank space to deal
work into, so `cluster/run_items.sh` claims items with an atomic `mkdir <id>.claim`
instead of the sibling project's golden-ratio rank stride. And **PyTorch 2.11's cu128
wheels dropped sm_70**, so the V100s need a cu126 build; the wrong wheel imports cleanly,
reports a CUDA device, and fails only at the first kernel launch, which is why
`cluster/smoke.sh` launches a real kernel rather than trusting `get_arch_list()`.


### The SuperCloud calibration (2026-09-02, `xeon-g6-volta`, V100)

Four arms, one per node, each a full job on a real partition. The workload is the
**Panda grasp** task, learned arm only, 4 targets x 2 guesses, `--compile` -- the
grasp task specifically, because it is the one that binds against the wall-clock
cap. A first attempt ran the *pose* task and measured nothing: a pose cell
converges in ~74 iterations and ~6 s here, so its iteration count is identical at
10, 20, 45, 90 and 180 s and identical however contended the node is. A converged
solve takes the iterations it takes; only its wall time moves. Both the cap sweep
and the contention sweep were therefore structurally incapable of showing an
effect, whatever the truth.

**Workers per node.** Median iterations achieved inside a fixed 20 s cap, against
the single-worker run. The node has 40 cores and 2 V100s.

| workers | median iters (GPU) | vs P=1 | median iters (CPU-only) | vs P=1 |
| --- | --- | --- | --- | --- |
| 1 | 202 | 1.00x | 62 | 1.00x |
| 2 | 196 | 0.97x | 62 | 1.00x |
| 4 | **194** | **0.96x** | 62 | 0.99x |
| 8 | 186 | 0.92x | 61 | 0.97x |
| 20 | 114 | 0.56x | 58 | 0.93x |
| 40 | 70 | 0.35x | 46 | 0.74x |

**`PROCS=4` is the campaign's setting**: four workers per node cost 4% of the
per-cell iteration count, where 8 costs 8% and 20 costs 44%. Note the P=1 row is
the *noisiest* in the table (it is one worker's median, against forty at P=40),
so the 0.96-0.97x at P=2 and P=4 is within noise of unity while the collapse at
P>=20 plainly is not. Since the benchmark is wall-clock capped, a worker that
gets less done is a *different measurement*, not merely a slower one.

**CPU-only is not competitive and the campaign runs on the GPU.** At one worker
the GPU reaches 202 median iterations against 62, and solves 4 of 8 cells against
1 of 8. This does not contradict the profiling result that the flow is CPU-bound
at batch 1 -- that says the GPU is never the bottleneck *while a GPU is present*,
not that torch on CPU is as fast. CPU-only does degrade far more gracefully
(0.74x at P=40 against 0.35x), but from a starting point 3.3x worse, and its
aggregate node throughput is lower at every P.

**The cap.** Single worker, same grid, 8 cells:

| cap (s) | median iters | median wall (s) | feasible of 8 |
| --- | --- | --- | --- |
| 10 | 142 | 10.04 | 3 |
| 20 | 203 | 17.21 | 4 |
| 45 | **338** | 22.62 | 5 |
| 90 | 338 | 23.47 | 5 |
| 180 | 338 | 21.56 | 6 |

**45 s is the campaign's cap.** Median iterations saturate at 338 by 45 s and do
not move at 90 or 180, and the median cell finishes at ~22 s -- so at 45 s the
median cell has converged with 2x headroom. Beyond 45 s only the tail gains (one
further cell of eight between 45 s and 180 s), which is what the Stage C cap
curve is for rather than something to buy with a bigger default. The laptop's
20/45 s were not inherited; this is measured on this hardware, and cluster and
laptop timings are never compared.

**Staging and startup**, from the parity arm: 40 concurrent `import torch,
pydrake, ikflow, jrl` take **10 s** total, so Lustre read amplification is not a
problem and the venv can stay on the shared filesystem -- copying it to node-local
`$TMPDIR` costs 231 s against Drake's 13 s and the repo's 1 s, and buys nothing.
`torch.compile` of the flow Jacobian costs ~35 s cold and ~17 s warm per process.

Qualitative parity holds: a pose grid and a grasp grid at the archived seed give
202 and 206 median iterations with 4 of 8 feasible, and the paired-start invariant
(`start_q_error == 0` for the learned and joint-space arms) reproduces on this
hardware with a different torch build and a different GPU.

### Stage B on SuperCloud: the correction cost is the one knob that works (2026-09-02)

Panda and iiwa grasp, learned arm only, 15 x 4 = 60 cells, 45 s cap, paired start,
`--compile`, 8 workers per node. One factor at a time; the sc_A grasp finals on the same
grid supply the default point rather than being re-run. Success is the strict criterion
(the program's own rows at `ik_constraint_tol`).

| knob | value | Panda | iiwa |
| --- | --- | --- | --- |
| *(default)* | | 41/60 | 18/60 |
| `collision_influence_offset` (0.1) | 0.02 / 0.05 / 0.2 / 0.4 | 37 / 42 / 44 / 44 | 13 / 18 / **25** / 20 |
| `collision_row_scale` (0.1) | 0.02 / 0.05 / 0.2 / 0.5 | 39 / 40 / 42 / 39 | 19 / 15 / 24 / 17 |
| `ipopt_mu_strategy` | adaptive | 43 | 17 |
| `latent_cost_weight` (0) | 0.001 / 0.01 / 0.1 | 40 / 44 / **50** | 11 / 12 / 15 |
| **`correction_cost_weight` (0)** | 0.001 / 0.01 / 0.1 / 1.0 | 38 / 44 / 47 / **50** | 13 / 24 / 30 / **33** |

**`correction_cost_weight` is the only knob with a strong, monotone effect in the same
direction on both robots** -- +9 cells on the Panda and +15 on the iiwa, rising all the
way to the largest weight tested. Everything else is either mild, non-monotone, or
robot-specific: `mu_strategy=adaptive` does nothing on either robot despite the archived
IPOPT logs looking like its textbook case; the collision-shaping knobs peak weakly around
0.2 and are within noise of the default; and `latent_cost_weight` **helps the Panda by 9
cells and hurts the iiwa by 3**, so it is not a general win and should not be read as one.

The mechanism is visible in the instrumentation, and it is not that the correction box was
binding (it never is -- `on the box` is 0.00 at every point of the sweep):

| `correction_cost_weight` | 0 | 0.001 | 0.01 | 0.1 | 1.0 |
| --- | --- | --- | --- | --- | --- |
| median `\|q_c\|` | 0.0797 | 0.0133 | 0.0045 | 0.0006 | 0.0001 |
| median max constraint violation, Panda | 5.7e-05 | 7.2e-05 | 4.1e-05 | 2.0e-05 | **3.6e-06** |
| median max constraint violation, iiwa | 1.7e-02 | 3.5e-02 | 8.7e-03 | 2.7e-04 | **2.9e-05** |
| median `\|\|z\|\|` | 1.78 | 1.33 | 1.81 | 1.55 | 1.60 |

As the weight rises the correction is driven to zero and **the median constraint violation
falls by three orders of magnitude on the iiwa**, from grossly infeasible to feasible,
while the latent stays put. That is the signature of a degeneracy being removed rather
than of a better search: with `c` and `q_c` both free, many pairs give the same `q`, so the
active constraint gradients are rank-deficient -- exactly the redundancy recorded as
next-steps #14 and never before tested. Penalising `q_c` breaks it.

Two cautions on reading this table, the first of which has since been **corrected**.
The `cost` column *is* comparable across the `corrcost` sweep: the rise from 2.63 to 4.87
on the Panda is not the penalty term entering the reported objective (at `w >= 1` that
term is ~1e-08, the penalty having driven `|q_c|` to ~1e-05) but a real change in which
solutions come back, and it is the price the penalty charges for its feasibility. See
"What the correction penalty costs" further down. And the smallest weight, 0.001, is *worse* than zero on both robots (38 vs 41,
13 vs 18) -- either noise at this grid size or a weak penalty perturbing without
regularising, and not currently distinguishable.

**Thomas approved the penalty on 2026-09-02** ("A penalty on the correction term is
acceptable"), so a nonzero `correction_cost_weight` is now part of the learned
formulation rather than an unauthorised change to it. The draft says `q_c ~ 0` without
specifying how that is imposed, and this is what imposes it. Two consequences for how it
is reported: the weight is a *stated* part of the formulation and must appear wherever
the learned arm is described, and every table must still show what the penalty buys --
the with/against-without comparison is paired on the same grid (`sc_A` against
`sc_B_*_corrcost_*`, and at full power in Stage D), never dropped once the penalty is
adopted.

### The complete correction-cost curve, and where it peaks (2026-09-02)

Sweeps B, B2 and B3 together, grasp task, learned arm only, 60 cells at 45 s on one grid
(`grid_hash a480c9c9590e`, seed 0), so every point pairs cell for cell.

| robot / start | 0 | 0.001 | 0.01 | 0.1 | 1 | 3 | 10 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda paired | 41 | 38 | 44 | 47 | 50 | **54** | 48 | 51 |
| Panda native | 42 | | | | | 52 | **58** | 50 |
| iiwa paired | 18 | 13 | 24 | 30 | 33 | 37 | **47** | 42 |
| iiwa native | 16 | | | | | 37 | 39 | **40** |
| total of the four | 117 | | | | | 180 | **192** | 183 |

**The curve turns over between 10 and 30**, which is what B3 was run to establish: at 30
three of the four experiments are flat or worse, so 10 is near the optimum rather than
merely the largest value anyone tried. Stage D fields 10 on that basis.

Read the per-experiment wobble as noise, not structure. Panda paired goes 54 -> 48 -> 51
across weights 3, 10 and 30, which is a +-3 to 6 cell swing on a 60-cell grid; the
aggregate and the monotone rise out of 0 are the real signal, and resolving the top of
the curve properly is one of the things 480 cells buys in Stage D.

What the penalty does, from the instrumentation rather than by inference:

| | w = 0 | w = 10 |
| --- | --- | --- |
| median `\|q_c\|` | 0.080 | 0.0000 |
| median max constraint violation, Panda | 5.7e-05 | 6.4e-08 |
| median max constraint violation, iiwa | 1.7e-02 | 2.6e-08 |
| timeouts, iiwa paired (of 60) | 40 | 15 |
| median `\|\|z\|\|` | 1.78 | ~1.6 |

The correction is driven to zero, the median constraint violation falls to the
**joint-space arm's own level** (1.1e-08), and the timeouts more than halve, while the
latent does not move. This is the `c`/`q_c` redundancy of next-steps #14 being removed:
with both free many pairs give the same `q`, so the active constraint gradients are
rank-deficient, and IPOPT spends its budget on a degenerate direction.

**The headline consequence.** On the Panda grasp task under `native`, the learned arm
reaches **58/60 against joint space's 55/60** -- the grasp task was the one place the
learned formulation lost, and the deficit is substantially an artefact of leaving the
correction free rather than a property of the formulation. The iiwa grasp arm goes
18/60 to 47/60 against joint space's 58/60: still behind, but the gap falls from 40 cells
to 11. Both statements are on a 60-cell grid and on the *same* grid the weight was chosen
on, which is why Stage D re-measures them at 480 cells on a different seed.

### Stage D: the correction penalty holds out of sample, at 480 cells (2026-09-02)

The big-N replication, and the first table in this repo measured on a grid that no
tuning decision was made on. 60 targets x 8 per-target guesses = **480 cells**, 45 s cap,
`--compile`, both start protocols, both robots, both tasks, learned arm only in the
paired penalty/no-penalty form. **Seed 1**, deliberately: targets are drawn sequentially
from the seed, so the 60-target *seed-0* grid literally contains the 15-target seed-0
sweep grid as a prefix, and reporting the headline table there would quote a weight that
was chosen on a quarter of the very cells being reported.

`correction_cost_weight = 10` (Thomas's approved penalty) against the same formulation
with the penalty off, exact McNemar on all 480 shared cells:

| experiment | start | penalty | no penalty | better / worse | p |
| --- | --- | --- | --- | --- | --- |
| Panda grasp | paired | **417/480** | 347/480 | 114 / 44 | **2.4e-08** |
| Panda grasp | native | **437/480** | 377/480 | 94 / 34 | **1.1e-07** |
| iiwa grasp | paired | **237/480** | 98/480 | 169 / 30 | **1.1e-24** |
| iiwa grasp | native | **227/480** | 77/480 | 169 / 19 | **3.0e-31** |
| Panda pose | paired | 338/480 | 331/480 | 71 / 64 | 0.61 (tie) |
| Panda pose | native | 466/480 | 468/480 | 6 / 8 | 0.79 (tie) |
| iiwa pose | paired | 296/480 | 280/480 | 90 / 74 | 0.24 (tie) |
| iiwa pose | native | 407/480 | 394/480 | 37 / 24 | 0.12 (tie) |

**The penalty is a grasp-task effect and only a grasp-task effect**, which is exactly
what the mechanism predicts. The `c`/`q_c` redundancy costs the solver most where the
active constraint set is largest, and the pose task's four rows are a tie under every
protocol -- so the penalty is not a general success multiplier being read off a lucky
grid, and it costs the pose task nothing to adopt.

The instrumentation reproduces the Stage B mechanism at 8x the cells:

| | penalty off | penalty (w = 10) |
| --- | --- | --- |
| median `\|q_c\|` (Panda / iiwa) | 0.074 / 0.072 | 1.8e-05 / 1.9e-05 |
| fraction on the +-0.1 box, Panda | 0.012-0.013 | **0.000** |
| median max violation, Panda paired | 4.4e-05 | **3.7e-08** |
| median max violation, iiwa paired | 9.4e-02 | **2.5e-04** |
| timeouts of 480, iiwa paired | 355 | **235** |
| timeouts of 480, Panda paired | 135 | **79** |

Two cautions carried forward from Stage B, one of which has since been **corrected**.
The rise in median cost from 2.53 to 4.88 on the Panda paired grasp is *not* the penalty
term appearing in the objective -- at `w = 10` that term is about 2e-08 -- it is a real
change in which solutions the solver returns, and the columns are comparable. See
"What the correction penalty costs" above: the penalty trades objective value for
feasibility, and on the grasp task the trade is 30-100%. And the
relaxed criterion is worth +10 to +18 cells of 480 on the grasp task and **exactly zero
on the pose task**, moving no ordering anywhere.

**The baseline columns of this run are void and are being re-measured** -- see the
correction-cost guard below. The learned columns are unaffected.

### `correction_cost_weight` silently voided every baseline column of Stage D

`IKFlowProgram.add_costs` applied `correction_cost_weight` unconditionally, but
`correction` is a **learned-only decision variable** and all three formulations share one
`ProgramOptions` object. So `--set correction_cost_weight=10` raised
`AttributeError: 'PandaIKProgramNumerical' object has no attribute 'correction'` inside
every numerical/analytic/analytic8 program's construction, and each of those columns
scored **0 of 480 in about 10 ms per cell**.

The guard is the one `latent_cost_weight` and `latent_trust_region` already carry, and
the comment sitting directly above them already stated the rule -- *"options that name
learned-only variables must be guarded"*. This cost was simply missed when it was added.
It is fixed, and a baselines-only re-run (`--stage Dbase`) re-measures those columns on
the same seed, grid, cap and start protocols, so they pair cell for cell against the
learned records above.

The failure mode is worth remembering because of how it presents: a whole column of
zeroes with `median_max_violation = nan` and `mean_iterations = nan`, at ~10 ms a cell.
**Any arm reporting a per-cell wall time three orders of magnitude below the cap is not
solving badly, it is not solving at all** -- and `fail_reason` was `"error"` rather than
any of the named task gates, which is the tell.

### Stage D, the three-way comparison at 480 cells (2026-09-02)

The baseline columns, re-measured after the correction-cost guard on the same seed-1
grid, cap and start protocols as the learned columns above, so every pairing below is
exact McNemar over all 480 shared cells. The learned arm is the draft's eq. (6) with the
approved correction penalty (`correction_cost_weight = 10`); joint space is the
comparison's target and the analytic columns are baselines.

| experiment | start | learned | joint space | analytic4 | analytic8 | L vs js (b/w) | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda pose | paired | **338/480** | 249/480 | 221/480 | 228/480 | 156 / 67 | **2.3e-09** |
| Panda pose | native | **466/480** | 249/480 | 242/480 | 142/480 | 224 / 7 | **3.8e-57** |
| iiwa pose | native | **407/480** | 332/480 | -- | -- | 122 / 47 | **7.2e-09** |
| iiwa pose | paired | 296/480 | **332/480** | -- | -- | 87 / 123 | 0.016 |
| Panda grasp | native | 437/480 | **457/480** | 450/480 | 387/480 | 20 / 40 | 0.013 |
| Panda grasp | paired | 417/480 | **457/480** | 376/480 | 418/480 | 19 / 59 | 6.4e-06 |
| iiwa grasp | paired | 237/480 | **462/480** | -- | -- | 7 / 232 | 1.9e-59 |
| iiwa grasp | native | 227/480 | **462/480** | -- | -- | 11 / 246 | 5.9e-59 |

**The harness checks itself and passes.** The joint-space arm is bit-identical between the
two protocols in all four experiments -- 249/249, 457/457, 332/332, 462/462, with the same
mean iteration counts -- as it must be, since its native start *is* a random
configuration. Every difference in the other columns is therefore attributable to their
initialisation. `median_start_q_error` is **0.0 exactly** for the learned and joint-space
arms under `paired`, and 1e-11 for the analytic arms where the chart covers `q_init`.

**The pose result is the draft's central claim and it holds on both robots under
`native`, and on the Panda under both protocols** -- decisively so (466/480 against
249/480, 224 cells won and 7 lost). It also wins on **cost** wherever it wins on success --
10.383 against joint space's 10.643 on the Panda paired pose and 6.457 against 6.776 on
the iiwa native pose, medians on the 182 and 285 cells *both* arms solved. (The figures
this section previously quoted, 9.46 and 6.40, were medians over each arm's own
successes, which compares different cell sets; the direction is unchanged.) On the
**grasp** task the ordering reverses and the learned arm costs roughly twice what joint
space does -- see "Iterations, cost and wall clock" above, which also shows that gap is
the correction penalty being paid for in objective value.

**One conclusion changes at this sample size, and it changes against us.** The iiwa pose
under `paired` read as a tie at 60 cells (40/60 vs 39/60, p = 1.0); at 480 cells it is a
**modest loss**, 296 against 332 (p = 0.016). The extra power resolved it rather than
confirming it, which is exactly what Stage D was run for, and it should be reported that
way. Note the learned arm still has the *lower* median cost on that row (6.54 against
6.79) and **172 of its 480 cells exit at the cap** against zero for joint space, so this
row is at least partly a budget statement -- which is what Stage C's cap curve is for.

**The Panda grasp deficit is now small.** With the penalty the learned arm reaches 437/480
and 417/480 against joint space's 457/480 -- 20 and 40 cells of 480, against the 110-cell
gap the same grid shows without the penalty. The iiwa grasp remains the outlier at
237/480, with 235 cells at the cap and a median max violation of 2.5e-04, still above
`ik_constraint_tol`.

**analytic8 against analytic4, at 8x the power, is still the unbalanced-bundle
pathology** and it now shows both signs cleanly. Under `paired` the 8-branch chart wins
the grasp task (418 against 376) because it can represent starts the 4-branch chart
forfeits; under `native` it loses badly on both tasks (387 against 450, and **142 against
242** on the pose task), because a uniform draw over eight branches lands in the narrow
near-limit bundles half the time against roughly 10% of configuration-space volume. Its
median max violation on that worst row is **1.2e-01** -- those cells are not near-misses,
they are solves begun inside a bundle pinned against the joint limits.

The relaxed criterion (`task_tol = 1e-3` instead of `ik_constraint_tol = 1e-4`) is worth
+13 to +18 cells of 480 to the learned arm on the grasp task, +2 to +6 to the baselines,
and **exactly zero to anything on the pose task**. It moves no ordering in the table.

### Stage C, the success-vs-cap curve: the cap only ever bound one arm, and now it is spent (2026-09-02)

Next-steps #7, finally a curve instead of two points. Eight experiments x six caps
(5 / 10 / 20 / 45 / 90 / 180 s), 15 targets x 4 per-target guesses = 60 cells, seed 0,
`--compile`, both start protocols, all arms, against the **approved formulation**
(`correction_cost_weight = 10`). One grid per experiment across all six caps, so every
column pairs cell for cell.

| experiment | start | 5 s | 10 s | 20 s | 45 s | 90 s | 180 s | joint space (all caps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda pose | native | 39 | 53 | 56 | 58 | 58 | **58** | 29 |
| Panda pose | paired | 18 | 32 | 36 | 40 | 41 | **41** | 29 |
| iiwa pose | native | 44 | 51 | 59 | 60 | 60 | **60** | 39 |
| iiwa pose | paired | 21 | 34 | 40 | 40 | 40 | **40** | 39 |
| Panda grasp | native | 21 | 30 | 45 | 57 | 59 | **58** | 55 |
| Panda grasp | paired | 15 | 24 | 35 | 50 | 53 | **55** | 55 |
| iiwa grasp | native | 11 | 17 | 30 | 39 | 42 | **45** | 58 |
| iiwa grasp | paired | 13 | 21 | 31 | 45 | 52 | **53** | 58 |

**Every baseline is flat at every cap, in all eight experiments.** Joint space scores
identically at 5 s and at 180 s (29, 39, 55, 58) with the same mean iteration counts, and
the analytic arms saturate by 10-20 s. Thirty-six baseline measurements, one number each.
This is the strongest form of a result the laptop saw at two points: **the wall-clock cap
is not a shared budget, it is a budget for the arm that evaluates a network**, and any
comparison that quotes one cap is quoting a point on one curve against four constants.

**The Panda grasp deficit closes.** At 180 s, paired, the learned arm scores **55/60
against joint space's 55/60** -- exact parity, 5 cells better and 5 worse, p = 1.0 -- and
under `native` 58/60 against 55/60 (5/2, p = 0.45, a nominal lead that this grid cannot
resolve). This supersedes the standing statement that "the gap narrows with budget without
closing": that was measured without the correction penalty. With the penalty *and* an
adequate budget it closes on the Panda. The iiwa grasp gap narrows the same way without
closing -- 53/60 against 58/60 paired is **no longer significant** (2/7, p = 0.18), where
at 45 s it was 18/60 against 58/60.

**The pose result strengthens with budget rather than washing out**: Panda 58 vs 29 native
(30/1, p = 3.0e-08) and 41 vs 29 paired (22/10, p = 0.050); iiwa 60/60 vs 39 native
(21/0, p = 9.5e-07) and 40 vs 39 paired (13/12, p = 1.0, a tie).

**What the curve costs Stage D.** Stage D ran at 45 s, and the two grasp rows are still
climbing there: 45 -> 180 s is worth +5 cells on the Panda paired grasp, +6 on the iiwa
native and +8 on the iiwa paired. **The Stage D grasp columns are therefore
budget-limited**, and the iiwa grasp row in particular (237/480 with 235 cells at the cap)
should be read as a lower bound, not as that formulation's asymptote. The pose rows are
unaffected -- they saturate by 45 s under `native` and by 20 s under `paired`.

#### The residual failures are divergence, not slowness (next-steps #11, measured)

The cells still at the cap at 180 s are the same cells that were at the cap at 20 s and at
45 s -- literally the same `(target, guess)` pairs, checked as sets -- and they are not
converging slowly. On iiwa pose paired the cap-bound set is **frozen at 19 cells from 20 s
through 180 s** and success never moves off 40, which is why that row saturates so early.

| | diverged (at the 180 s cap) | succeeded |
| --- | --- | --- |
| n (iiwa pose paired) | 19 | 40 |
| median `\|q_c\|` | **0.082**, against the +-0.1 box | 1.3e-05 |
| median `\|z\|` at exit | 3.58 | 1.54 |
| median position error | **0.81 m** | 1.0e-04 |
| median true min distance | **-0.085 m** (in collision) | clear |
| median iterations | 587 | -- |

Three things this settles. The violated binding is `AllIKFlowConstraints` on **19 of 19**
cells, while *every* variable region reports slack -- latent trust region -5.7, `c` box
-0.24, correction box -0.018 -- so this is not the latent or the conditioning pose
escaping, which is what next-steps #11 assumed. The **correction is pinned near its box**
(0.082 of 0.1) on exactly the cells the penalty fails to drive it to zero on, against
1e-05 on the successes: the penalty is winning or losing per cell, not on average. And the
returned point is deeply in collision (median -8.5 cm, worst -17.8 cm) and roughly a metre
off target -- these solves wandered somewhere infeasible and stayed there.

That row was identified from the persisted `q`, and it changes the reading of the table
above: it is the **joint-limits row**, on configurations of 1e7 to 1e16 radians. The
collision depth and the metre of position error are *consequences* of that, not causes --
see "The residual failures are the flow's own gain" below, which is where this thread ends.

Finally, note the numbers here are on **cluster hardware at PROCS=8**, and per the
standing rule are never compared cell-for-cell against a laptop table; `metadata.host` and
`metadata.device` make that mechanical.

### The residual failures are the flow's own gain, and it is 51x worse on the iiwa (2026-09-02)

Stage C left one question: which row inside `AllIKFlowConstraints` reaches 1e7-1e11 on the
cells that never converge. It is the **joint-limits row**, and `max_violation` equals
`|q|_inf` exactly on those cells (Spearman 1.0, values agreeing to the digit, on 55 of the
64 badly-violating learned cells across all eight 180 s runs). The returned configurations
have joint angles of **1e7 to 1e16 radians**.

Everything else about those cells follows from that. A configuration of 1e8 rad puts the
gripper anywhere, so the "deeply in collision, a metre off target" reading is a
*consequence* of the blow-up, not the cause, and the collision penalty is a bystander --
`max_violation` does not track `collision_value` at all (Spearman -0.15).

**Every runaway lies on one ray, and it is a property of the network, not of the solve.**
Normalising the exploded `q` vectors and taking pairwise `|cos|`:

| | ray (unit, joint order) | pairwise `\|cos\|` |
| --- | --- | --- |
| iiwa, from the benchmark records (mug native, mug paired, pose paired) | `[0.001, -0.001, 0.016, -0.000, 0.003, 0.978, -0.208]` | **1.0000** across all three runs |
| Panda, from the benchmark records (mug native, mug paired, pose paired) | `[-0.016, 0.033, 0.998, -0.032, -0.002, 0.027, 0.023]` | **1.0000** |

The same ray on both tasks and under both start protocols. It is dominated by a single
joint -- the iiwa's wrist (joint 6, coefficient 0.978) and the Panda's elbow (joint 3,
0.998).

**Sampling the network directly reproduces it, with no Drake and no solver involved.**
Draw `c` position uniformly in its +-0.25 m box, a uniform unit quaternion, and `z`
uniformly in the ball of radius 4.3 -- i.e. strictly inside the region the formulation
allows -- and evaluate `MakeFlowInference` in float64:

| | Panda `lp191_5.25m` | iiwa14 `lemon-haze-7` |
| --- | --- | --- |
| median `\|q\|_inf` | 2.65 | 2.50 |
| p99 | 3.71 | 8.7e+05 |
| p100 of 20000 | 4.1e+12 | 5.5e+16 |
| fraction `> 3` rad (outside joint limits) | 0.159 | 0.142 |
| **fraction `> 1000` rad** | **0.00065** | **0.0334** |
| fraction `> 1e6` rad | 0.00060 | 0.00885 |
| ray recovered from those samples | `[-0.017, 0.035, 0.998, -0.032, -0.002, 0.029, 0.025]` | `[0.008, 0.030, 0.017, -0.007, 0.047, 0.984, -0.167]` |
| `\|cos\|` against the ray the *solver* landed on | **0.9999** | **0.9976** |

The distribution is bimodal, not heavy-tailed: on the Panda, 14 of 20000 exceed 10 rad and
13 of those exceed 1000. A draw is either an ordinary configuration or it is astronomical.

**This is the answer to the iiwa grasp deficit**, the campaign's one open scientific
question. The iiwa checkpoint puts **3.34% of the allowed region** into the blow-up regime
against the Panda's **0.065% -- a factor of 51**. And it explains why the dose-response
experiment refuted the chart-accuracy hypothesis: `chart_error_scale` adds smooth
`eps*sin(...)` error, which degrades accuracy while adding no such regions, so it was never
capable of reproducing this. The iiwa's problem was never that its chart is imprecise; it is
that its chart has fifty times more of these regions to fall into.

**The mechanism is architectural headroom, not a numerical bug.** FrEIA's coupling blocks
soft-clamp the log-scale to `clamp * 0.636 * atan(s/clamp)`, bounded by `clamp * 0.636 *
pi/2`, so with `rnvp_clamp = 2.5` over `nb_nodes = 12` the worst-case output gain is about
`e^(2.5*12) ~ 1e13` -- which is exactly the scale of the observed maxima. These are
near-worst-case gain regions of a bounded map, not poles. Both checkpoints have the same
headroom; they differ only in how much of the conditioning domain sits near it.

**`rnvp_clamp = 2.5` is confirmed correct**, which was worth checking because
`src/iiwa_program.py` hardcodes the iiwa's hyperparameters and `rnvp_clamp` changes the
forward pass without changing any parameter shape -- a wrong value would load silently.
Sweeping it against the same weights:

| `rnvp_clamp` | 1.0 | 1.5 | 2.0 | **2.5** | 3.0 | 5.0 |
| --- | --- | --- | --- | --- | --- | --- |
| median `\|q\|_inf` | 2.8e+06 | 28.3 | 4.24 | **2.49** | 2.00 | 6.9e+08 |
| fraction `> 1000` | 0.893 | 0.410 | 0.090 | **0.035** | 0.126 | 0.999 |

2.5 is a clear optimum, so the checkpoint is being loaded as trained and the 3.34% is a
property of the weights.

**Why the solver finds a 3%-measure set 30% of the time.** It does not sample; it follows
gradients, and `dq/dvars` in these regions is as large as `q` is. A Newton step is
*attracted* to them. That is also why more budget never helps: the cap-bound cells at 180 s
are the same cells that were cap-bound at 20 s, and on iiwa pose paired the set is frozen at
19 cells across every cap tested.

**Neither existing region knob avoids them, because they are not at the edges.** Fraction
of the region with `|q|_inf > 1000`, sweeping the two knobs that already exist:

| latent trust-region radius | 1.0 | 2.0 | 3.0 | 4.3 | 6.0 | 8.0 |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa | 0.033 | 0.026 | 0.032 | 0.035 | 0.040 | 0.071 |
| Panda | 0.0018 | 0.0020 | 0.0020 | 0.0018 | 0.0018 | 0.0018 |

| `c_position_slack` | 0.05 | 0.10 | 0.25 (default) | 0.50 |
| --- | --- | --- | --- | --- |
| iiwa | 0.039 | 0.041 | 0.035 | **0.225** |
| Panda | 0.000 | 0.000 | 0.0018 | **0.137** |

Shrinking the trust region to `R = 1` leaves the iiwa's exposure unchanged at 3.3%: the
blow-up regions are spread through the domain, including at `|z| <= 1`, not banished to its
outskirts. **This is why the trust-region sweep measured inert** -- it was never able to
exclude them, so its only effect was on conditioning, which is the reason it stays (IPOPT
and unbounded variables) rather than a success argument.

The `c` box tells a different and mildly alarming story: flat from 0.05 to 0.25 and then a
**cliff** at 0.5, where exposure jumps 6x on the iiwa and 76x on the Panda. The default
`c_position_slack = 0.25` sits just under that cliff, which is luck rather than design --
worth knowing before anyone widens it.

**What this does not license.** Adding a bound or a penalty that keeps `q` in range would be
a change to the formulation under test, not tuning, so it is Thomas's call and not one to
make here -- see Stage E below. (He has since authorised both as *diagnostics*, disliking
both; Stage F measures them and neither fixes this. The penalty is inert; lifting bounds the
returned configuration but relocates the runaway into the equality residual.) Note also that
the learned arm is structurally alone in this: joint space carries joint limits as *variable
bounds*, satisfied at every iterate by construction, and the analytic arm's chart is a closed
form with no such regions. The learned arm is the only one whose limits are imposed on the
output of a map -- but **not**, as an earlier revision of this file claimed, on something it
cannot control: the limit row's gradient is pulled back through the network to `z` exactly,
which is the whole point of differentiating through the flow. What differs is *when* the
limits hold (only at convergence), and that the gradient into a high-gain region is itself
enormous, so the Newton step is attracted rather than repelled.

### Stage E: IPOPT's scaling is not the lever either (2026-09-02)

The one remaining thing that could be tried without changing the formulation. The runaway
diagnosed above puts a constraint row reaching 1e11 in front of a scaling scheme that
computes its factors from the gradients at the *starting* point and caps them at
`nlp_scaling_max_gradient = 100` -- so a row that only becomes enormous later is scaled as
though it were ordinary. Learned arm only, 15 x 4 = 60 cells, 45 s, `--compile`, seed 0, on
the finals' grids. The default point is **not** re-run: Stage C's 45 s learned columns are
exactly it, on the same grid, seed and cap.

| experiment | default | `nlp_scaling_method=none` | `max_gradient=1e4` | `max_gradient=1e8` |
| --- | --- | --- | --- | --- |
| iiwa grasp paired | 45/60 | 45 | 45 | 45 |
| iiwa grasp native | 39/60 | 38 | 38 | 39 |
| iiwa pose paired | 40/60 | 41 | 41 | 41 |
| Panda pose paired | 40/60 | 42 | 42 | 42 |
| Panda grasp paired | 50/60 | 48 | 49 | 49 |

**Every variant is inert.** The largest movement is +-2 cells and no comparison against the
default reaches p < 0.5, let alone significance (exact McNemar, 60 shared cells). Note also
that `max_gradient = 1e4` and `1e8` reproduce `none` almost cell for cell, which is what
raising the cap far enough should do -- three settings, one behaviour.

So the runaway is **not a scaling artefact**. IPOPT is not mis-scaling a row it could have
handled; it is being handed a chart with regions of gain ~1e13 and following the gradient
into them. Nothing in the solver's options fixes that, which -- with the trust region and
the `c` box already ruled out -- leaves only the two remedies that are Thomas's to
authorise: a different iiwa checkpoint, or something acting on `q`.

**`equilibration-based` is unavailable in Drake's IPOPT** (it needs the HSL MC19 routine)
and raised `RuntimeError: Error setting IPOPT string option` on construction, scoring
0/60 in ~10 ms a cell. Its column is a crash, not a measurement, and is excluded above.
IPOPT's own error message lists what it will accept: `user-scaling`, `gradient-based`,
`none`.

**That failure mode is now trapped.** `_abort_on_dead_arm` in `src/benchmark.py` aborts a
run when an arm fails **identically, in under a second, on its first three cells** -- the
unmistakable signature of a misconfiguration rather than a hard problem, since a
configuration error is deterministic and instant while a numerical failure varies and costs
real time. This has now cost two whole columns of cluster campaigns (the
`correction_cost_weight` guard incident and this one), both caught only during analysis
after the compute was spent.

### Iterations, cost and wall clock: the three numbers a result is told in (2026-09-02)

**Standing reporting rule, Thomas's: every result is told in iteration count and in
objective cost, as well as in runtime.** Iterations are the hardware-independent quantity
and the one that describes the *formulation*; seconds describe this implementation on this
machine, and are never compared across machines; cost says what the solution is worth,
which success alone cannot. Reporting only seconds makes the cap story look arbitrary;
reporting only iterations hides that the learned arm's iteration is thirty times more
expensive than joint space's; reporting only success hides that on the grasp task the
learned arm's solutions cost roughly twice what the baseline's do. All three, always.

Median over each arm's *succeeded* cells, Stage C at the 180 s cap (60 cells, seed 0,
`--compile`, `correction_cost_weight = 10`), so every arm has converged and the cap binds
on nothing:

| experiment | start | arm | solved | median iters | median s | ms / iter |
| --- | --- | --- | --- | --- | --- | --- |
| Panda pose | native | **learned** | 58 | 52 | 4.57 | **69.7** |
| | | joint space | 29 | 34 | 0.10 | 2.9 |
| | | analytic4 / analytic8 | 37 / 27 | 10 / 13 | 0.07 / 0.09 | 6.5 / 6.4 |
| Panda pose | paired | **learned** | 41 | 101 | 7.26 | **72.3** |
| | | joint space | 29 | 34 | 0.10 | 2.9 |
| Panda grasp | native | **learned** | 58 | 187 | 15.00 | **83.8** |
| | | joint space | 55 | 48 | 0.14 | 2.8 |
| | | analytic4 / analytic8 | 59 / 53 | 113 / 98 | 0.91 / 0.74 | 8.1 / 8.0 |
| Panda grasp | paired | **learned** | 55 | 215 | 18.12 | **84.2** |
| | | joint space | 55 | 48 | 0.14 | 2.8 |
| iiwa pose | native | **learned** | 60 | 57 | 3.71 | **59.0** |
| | | joint space | 39 | 30 | 0.06 | 2.3 |
| iiwa pose | paired | **learned** | 40 | 126 | 6.50 | **52.6** |
| iiwa grasp | native | **learned** | 45 | 207 | 15.77 | **83.4** |
| | | joint space | 58 | 66 | 0.17 | 2.6 |
| iiwa grasp | paired | **learned** | 53 | 213 | 19.07 | **82.2** |

**The per-iteration cost is the honest headline, and it is a factor of 25-30.** The learned
arm pays **53-84 ms** an iteration against joint space's **2.3-2.9 ms** and the analytic
arms' **6-8 ms**, and the ratio is remarkably stable across robots, tasks and protocols --
as it must be, since it is one network Jacobian per iteration against Drake kinematics. The
profiling section says that gap is CPU dispatch through PyTorch/FrEIA rather than
arithmetic, and that even zero-overhead execution leaves only a ~3x ceiling, so it is an
implementation property with a known floor rather than something tuning will remove.

**Iteration count is the formulation property, and it splits by task.** On the pose task
the learned arm wins on success while taking a comparable number of steps -- 52 against 34
on the Panda under `native`, 57 against 30 on the iiwa -- so its advantage there is that it
finds solutions the joint-space arm does not, not that it grinds longer. On the grasp task
it takes **3-4x** as many steps (187-215 against 48, 207-213 against 66) *and* pays 30x per
step, which together are the whole of the wall-clock cap story: the two multiply to roughly
100x, which is why 5 s is not a measurement of the grasp task and 180 s barely is.

**So "the learned arm reaches parity on the Panda grasp at 180 s" should always be stated
as: parity in success at 55/60 each, at 215 median iterations against 48, and 18.1 s
against 0.14 s.** All three numbers are true and only the set of them is honest. The same
applies in the learned arm's favour on the pose task, where 466/480 against 249/480 comes
at 52 iterations against 34 -- a win on success, on cost, and on step count, but still
~45x the wall clock.

#### Cost, on the cells both arms solved

Success says how often a formulation returns a valid configuration; cost says what that
configuration is worth. They can and here do point in different directions, so the cost
column is not optional.

Two things had to be right before the number meant anything. **Costs are compared only on
cells *both* arms solved** -- a median over each arm's own successes compares different
cell sets, and the easy cells are exactly the ones a weaker arm also solves, so that form
flatters whichever arm fails more. And **the learned-only regularizers are excluded from
the reported objective** (`reported_cost` in `src/benchmark.py`), so the column measures
the objective every formulation shares -- the joint-centering cost -- rather than the
learned arm's objective plus its penalties.

Stage D, 480 cells, 45 s, `correction_cost_weight = 10`, median cost on the cells both
arms solved:

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

**On the pose task the learned arm wins on cost as well as on success**, on both robots
and under both protocols -- modestly (1-10%) but with the same sign in all four rows, and
on 182-285 shared cells rather than on the whole-run medians the earlier claim quoted.
This is the draft's central claim holding on the second of its two axes.

**On the grasp task it loses on cost by roughly a factor of two, in every row.** This was
not being reported at all, and it changes how the grasp result reads: the honest statement
is no longer "the learned arm reaches parity at a large enough budget" but "it reaches
parity in *success* at a large enough budget, at roughly 4x the iterations, 30x the
seconds per iteration, and 2x the objective value."

#### What the correction penalty costs, and it is not nothing

The same comparison against the penalty-free arm, on the cells both solved -- the paired
form of the Stage D penalty table, which previously reported success only:

| experiment | start | n both | `w = 10` | `w = 0` |
| --- | --- | --- | --- | --- |
| Panda pose | native | 460 | 9.893 | **9.372** |
| Panda pose | paired | 267 | 9.502 | **9.459** |
| iiwa pose | native | 370 | 6.364 | **6.061** |
| iiwa pose | paired | 206 | 6.476 | **6.352** |
| Panda grasp | native | 343 | 5.441 | **2.903** |
| Panda grasp | paired | 303 | 4.815 | **2.444** |
| iiwa grasp | native | 58 | 6.168 | **4.606** |
| iiwa grasp | paired | 68 | 7.185 | **4.663** |

**The penalty is nearly free on the pose task (0.5-5%) and expensive on the grasp task
(30-100%)** -- and the grasp task is exactly where it buys its 70-169 cells of success.
So the penalty is a *trade*, not a free improvement, and it must be reported as one: it
converts objective value into feasibility. The mechanism is the same redundancy it was
adopted to break -- with `q_c` free the arm can nudge `q` toward a well-centred
configuration for nothing, and pinning `q_c` to zero means `q` is whatever the flow emits
at `(c, z)`, which is less centred. The whole grasp cost gap against joint space above is
this: 5.32 against 2.83 with the penalty, 2.90 without it.

Note this is *not* a bookkeeping artefact of the penalty term appearing in the objective.
At `w = 10` the correction is driven to `|q_c|_inf ~ 1.8e-05`, so the term itself
contributes about **2e-08** to a cost of ~5 -- six orders of magnitude too small to
explain the gap. The superseded caution in the Stage B and D sections ("the `cost` column
was long said not to be comparable across the sweep on the grounds that the penalty is
part of the objective being reported; that reasoning is wrong (the term is ~1e-08 at the
weights adopted) and the column is comparable -- what it shows is the objective value the
penalty costs") named the wrong mechanism: the columns *are* comparable, and the rise in cost
with the weight is a real change in which solutions the solver returns.

#### Two cautions on the iteration table

The medians are over each arm's *succeeded* cells, and that set
grows with the cap, so a median that rises from 5 s to 180 s is partly composition (harder
cells joining the set) rather than the same cells taking longer -- which is why the
comparison above is drawn at the one cap where every arm has saturated. And the ms/iter
column is cluster hardware at `PROCS=8`; per the standing rule it is never compared against
a laptop figure, only against the other arms measured beside it.

### Stage F: the two `q`-side interventions, and what each of them actually does (2026-09-03)

**Both arms below are STATED DEVIATIONS from the draft's eq. (6)**, run as diagnostics of
the runaway diagnosed above. Thomas authorised both while saying he dislikes both, and
ranked them **below simply getting a better iiwa chart**; nothing here is fielded as "the
learned formulation". `lift_q` adds `q` as a decision variable whose bounding box is the
joint limits and imposes the chart as a 7-row equality; `joint_limit_penalty_weight` adds a
quadratic hinge on limit violation to the objective.

A framing correction first, because it governs how this whole thread should be described.
The earlier claim that the learned arm "imposes joint limits on an output it does not
control" is **wrong**, and Thomas said so: *"the network does get to control the joint
limits a bit, since it can adjust z. That's the whole point of differentiating through the
network -- we take the constraint gradient for joint limits and pull it back through the
network to z."* The arm controls `q` exactly and analytically. The pathology is
**attraction**: where the chart's gain is ~1e13, `dq/dz` is as large as `q`, so a Newton
step is drawn into the region rather than repelled from it.

Pilot: 4 rows (the three where the pathology lives, plus a Panda control), learned arm only,
15 x 4 = 60 cells, 45 s, seed 0, `--compile`, `correction_cost_weight = 10`, on Stage C's
own grids -- so **Stage C's 45 s learned column is the default point and was not re-run**.
Exact McNemar against it on the 60 shared cells.

| experiment | variant | success | vs default | b/w | p | median iters | median cost | cells returning `\|q\|_inf > 1000` | median max violation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp paired | *(default)* | 45/60 | | | | 170 | 5.56 | 6 | 2.6e-08 |
| | `liftq` | 43 | -2 | 9/11 | 0.82 | 209 | 5.51 | **0** | 3.9e-08 |
| | `jlpen1` / `10` / `100` | 37 / 37 / 40 | -8 / -8 / -5 | | 0.12-0.41 | 172-214 | | 11 / 9 / 5 | ~1e-07 |
| iiwa grasp native | *(default)* | 39/60 | | | | 201 | 5.94 | 14 | 5.4e-08 |
| | `liftq` | 45 | +6 | 11/5 | 0.21 | 151 | 7.63 | **0** | 1.2e-07 |
| | `jlpen1` / `10` / `100` | 39 / 40 / 35 | 0 / +1 / -4 | | 0.45-1.0 | 166-199 | | 10 / 11 / 12 | ~1e-07 |
| iiwa pose paired | *(default)* | 40/60 | | | | 126 | 7.83 | 18 | 5.9e-08 |
| | `liftq` | **3** | **-37** | **1/38** | **2e-11** | 606 | 4.79 | **0** | **7.0e+06** |
| | `jlpen1` / `10` / `100` | 39 / 45 / 40 | -1 / +5 / 0 | | 0.42-1.0 | 118-148 | | 19 / 13 / 18 | ~6e-08 |
| Panda grasp paired | *(default)* | 50/60 | | | | 188 | 5.26 | 4 | 4.7e-08 |
| | `liftq` | **59** | **+9** | **9/0** | **0.004** | **137** | 5.65 | **0** | 8.7e-09 |
| | `jlpen1` / `10` / `100` | 52 / 54 / 53 | +2 / +4 / +3 | | 0.39-0.75 | 177-204 | | 6 / 2 / 2 | ~3e-08 |

**The penalty is inert, and that vindicates Thomas's objection to it.** He said *"we should
be able to rely on the constraint to handle it"*, and across twelve measurements the
smallest p against the default is **0.115**, no weight has a consistent direction on either
robot, the runaway counts do not move (11/9/5 against 6; 10/11/12 against 14; 19/13/18
against 18), and the median max violation is unchanged at ~1e-08. Adding a penalty on a
quantity a constraint row already governs buys nothing. The knob stays in the tree, off,
with this table as the reason not to revisit it.

**Lifting does exactly what it promises and does not fix the problem.** Read its two
columns together:

- **The returned configuration is never out of limits again** -- 0 cells above 1000 rad in
  all four rows, with `max |q_lift|` = 3.05 rad on the iiwa (its joint-7 limit is 3.054) and
  3.41 on the Panda, across 240 cells. That is a hard guarantee the default formulation
  cannot make, and it is the one thing lifting genuinely delivers.
- **But the runaway does not stop; it relocates.** The flow still reaches **1.19e10**, and
  since the chart is now an equality row, that lands in `max_violation` instead of in `q`.
  On the iiwa pose row the median violation is **7.0e+06**. The cell fails either way; only
  the row it fails in has changed.

Because `verify()` records `q` from the flow rather than from the lifted variable, the two
agree on converged cells (57 of 60 on the Panda grasp) and diverge on exactly the cells that
never converge -- which is what makes `q_lift` and `q_flow` being persisted separately worth
the two extra fields.

**The result splits by task, sharply, and in opposite directions:**

- **iiwa pose paired collapses to 3/60 against 40/60** (1 better, 38 worse, p = 2e-11), at
  **606 median iterations**. This is Thomas's objection measured: the badly scaled Jacobian
  moves out of an inequality row and into an equality one, and an interior-point method
  handles it worse there. It is worst on precisely the row where the flow's poles are worst.
- **Panda grasp paired improves to 59/60 against 50/60** (9 better, **0 worse**, p = 0.004)
  and takes **fewer** iterations (137 against 188) -- the only intervention anywhere in this
  campaign to gain cells while also getting cheaper.
- The two iiwa grasp rows are ties (-2 at p = 0.82, +6 at p = 0.21).

So lifting is not a fix for the runaway and is not adoptable as a formulation -- an arm that
scores 3/60 on one of the eight experiments is disqualified whatever it does elsewhere. What
it is, is evidence that **the grasp task and the pose task want different things from the
chart's algebraic presentation**, which is a more interesting finding than the one the stage
was run to get. Stage F2 expanded `liftq` alone to all eight experiments; it
reproduces both effects and settles them as a **task** split, net negative overall
(38 better / 116 worse, p = 2.2e-10). See the next section.

**Neither intervention changes the standing conclusion**, which is unchanged from the
diagnosis: the flow's own gain is the mechanism, and a better iiwa chart is the remedy
Thomas prefers over anything done on the optimization side.

### Stage F2: lifting, on all eight experiments -- it is a task effect, and it is net negative (2026-09-03)

`lift_q` alone, expanded from the pilot's four rows to all eight experiments. 15 x 4 = 60
cells each, 45 s, seed 0, `--compile`, `correction_cost_weight = 10`, on Stage C's grids, so
every row is exact McNemar against Stage C's 45 s learned column on 60 shared cells.

| experiment | start | default | `liftq` | +/- | b/w | p | iters (def -> lift) | runaway cells (def -> lift) | median max violation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda grasp | paired | 50 | **58** | +8 | 9/1 | **0.022** | 188 -> 136 | 4 -> 0 | 8.7e-09 |
| iiwa grasp | native | 39 | 44 | +5 | 11/6 | 0.33 | 201 -> 150 | 14 -> 0 | 1.2e-07 |
| Panda grasp | native | 57 | 59 | +2 | 3/1 | 0.63 | 185 -> 114 | 1 -> 0 | 8.3e-09 |
| iiwa grasp | paired | 45 | 43 | -2 | 9/11 | 0.82 | 170 -> 209 | 6 -> 0 | 3.9e-08 |
| Panda pose | native | 58 | 52 | -6 | 2/8 | 0.11 | 52 -> 44 | 0 -> 0 | 1.0e-08 |
| iiwa pose | native | 60 | 44 | -16 | 0/16 | **3.1e-05** | 57 -> 30 | 0 -> 0 | 2.0e-08 |
| Panda pose | paired | 40 | **9** | -31 | 4/35 | **3.3e-07** | 100 -> 161 | 13 -> 0 | **3.3e+06** |
| iiwa pose | paired | 40 | **2** | -38 | 0/38 | **7.3e-12** | 126 -> 317 | 18 -> 0 | **7.1e+06** |
| **all eight** | | | | | **38 / 116** | **2.2e-10** | | | |

**Net, lifting is clearly worse: 38 cells better against 116 worse over 480 shared cells,
p = 2.2e-10.** The pilot's one significant gain (Panda grasp paired, +8, p = 0.022)
reproduces, and so does the collapse, and the collapse is much the larger effect.

**It is a task effect, not a robot effect**, and that is the finding worth keeping:

- **On the grasp task lifting is neutral-to-positive on both robots and both protocols**
  (+8, +5, +2, -2; only the Panda paired row is significant), and it consistently takes
  **fewer iterations** -- 136 against 188, 114 against 185, 150 against 201.
- **On the pose task it is negative in all four rows** and catastrophic under `paired`
  (Panda 40 -> 9, iiwa 40 -> 2), where it takes **2-3x more** iterations.

The mechanism is visible in the violation column. The pose task already pins the
end-effector with six rows -- three of them equalities at the time, the position rows
being a +-1e-4 box that has since been corrected; lifting adds seven more, so the solver faces thirteen
equalities in 27 variables with the flow's badly scaled Jacobian inside seven of them, and
on the two collapsing rows the median residual is **3e+06 to 7e+06** -- those solves never
close the chart equality at all. The grasp task's rows are mostly inequalities (the height
band, collision) with only the two mug-axis equalities, so the same seven rows are a much
smaller addition. **An interior-point method's tolerance for a badly scaled row depends on
how many equalities it is already carrying**, which is Thomas's objection with a mechanism
attached: *"we're effectively adding a nonlinear equality constraint (whereas currently, one
just has a free variable)."*

**The one thing lifting delivers, it delivers universally: 0 runaway cells in all eight
experiments**, against 0-18 for the default. The returned configuration is inside the joint
limits by construction, everywhere, always. It simply does not follow that the solve
succeeds -- the runaway moves into the equality residual and the cell fails there instead.

**Stage F3 (the 480-cell replication) was NOT run, and should not be.** The expansion was
gated on the pilot moving the runaway metric; F2 has now resolved that it does not -- the
pathology is relocated, not removed -- and an arm that scores 2/60 and 9/60 on two of the
eight experiments is disqualified as a formulation whatever it does on the other six.
Replicating it at 480 cells would spend one to two hours of contended cluster time
characterising something already ruled out. The negative result is complete at 60 cells.

**Where this leaves the runaway.** Every remedy that does not touch the chart has now been
measured and none works: the latent trust region, the `c` box, all of IPOPT's scaling
options (Stage E), a joint-limit penalty (Stage F, inert), and lifting `q` (Stage F2, net
negative). **A better iiwa chart remains the preferred remedy and is now also the only
untried one.** Gradient regularization was the remaining hope when this was written --
the finding above looked suggestive, since what fails in F2 is the conditioning of an
equality row carrying the chart's Jacobian, which is precisely what
Levenberg-Marquardt-style damping addresses. Stage G measured it across all eight
experiments and it is a clear negative; see that section for why damping an exact
derivative cannot work. The least-squares domain extension remains deferred.

### Stage G: Jacobian regularization (2026-09-03)

Three strategies for damping the flow's Jacobian before the chain rule, so the solver
sees bounded gradients while the value `q` is unchanged. The implementation is
`regularize_jacobian()` in `src/generic_program.py`, called from the AutoDiffXd branch
of `VarsToQ` in both robot programs.

**1. Frobenius norm clipping (`jacobian_max_norm`).**
If `||J||_F > max_norm`, scale `J ← J * (max_norm / ||J||_F)`. Isotropic — damps all
directions equally, including the well-conditioned ones.

**2. Tikhonov / LM damping (`jacobian_tikhonov_lambda`).**
Damp the singular values: `s_damped = s * λ / (s + λ)`. As `s → ∞`, `s_damped → λ`
(bounded); as `s → 0`, `s_damped → 0` (no amplification). This is the correct shape for
the runaway: large singular values are damped more than small ones.

**3. Singular value floor (`jacobian_svd_floor`).**
Truncated pseudoinverse: replace `s < floor` with `floor`. Prevents vanishing gradients
(the IFT paper's problem) as well as exploding ones.

**Sweep.** 10 variants × 8 experiments = 80 runs, 4 shards each = 320 items, ~1h on 4
nodes at PROCS=8. All compared against Stage C's 45s learned column. The variants are:

| knob | values |
| --- | --- |
| `jacobian_max_norm` | 10, 100, 1000, 10000 |
| `jacobian_tikhonov_lambda` | 0.1, 1.0, 10.0, 100.0 |
| `jacobian_svd_floor` | 0.1, 1.0 |

**Result: a clear negative, and the line of investigation is closed.** All eight
experiments, 60 cells each, 45 s, seed 0, `--compile`, `correction_cost_weight = 10`,
learned arm only, on Stage C's own grids -- so Stage C's 45 s learned column is the
default point and was not re-run. Exact McNemar on the 60 shared cells per row;
success is pooled over all eight rows below (480 cells per variant).

| variant | success / 480 | delta | better / worse | p | runaway cells / 480 |
| --- | --- | --- | --- | --- | --- |
| *(default)* | 389 | | | | 56 |
| `jacobian_max_norm=1000` | 395 | +6 | 47 / 41 | 0.59 | 48 |
| `jacobian_svd_floor=0.1` | 382 | -7 | 43 / 50 | 0.53 | 64 |
| `jacobian_svd_floor=1.0` | 378 | -11 | 45 / 56 | 0.32 | 63 |
| `jacobian_max_norm=10000` | 376 | -13 | 14 / 27 | 0.060 | 54 |
| `jacobian_max_norm=100` | 362 | -27 | 51 / 78 | 0.022 | 35 |
| `jacobian_tikhonov_lambda=100` | 362 | -27 | 45 / 72 | 0.016 | 40 |
| `jacobian_tikhonov_lambda=10` | 232 | -157 | 34 / 191 | 1.0e-27 | 42 |
| `jacobian_max_norm=10` | 202 | -187 | 30 / 217 | 3.7e-36 | 77 |
| `jacobian_tikhonov_lambda=1` | 2 | -387 | 1 / 388 | 6.2e-115 | 106 |
| `jacobian_tikhonov_lambda=0.1` | **0** | -389 | 0 / 389 | 1.6e-117 | 142 |

Aggregated over every row and variant: **310 cells better against 1,509 worse.** Not one
setting is a significant improvement, and the only one that is not net-worse
(`jacobian_max_norm = 1000`, +6, p = 0.59) is indistinguishable from no regularization
and does not reduce the runaway it was introduced to prevent (48 against 56).

**Why it cannot work, which is the part to remember.** The flow's Jacobian is the
*exact* derivative of an explicit function. Where the network's gain approaches its
architectural ceiling of ~1e13, a sensitivity of 1e13 is the correct answer, not an
artifact to be regularized away. Damping it does not regularize the problem -- it breaks
the correspondence between the constraint values IPOPT evaluates and the gradients it is
handed, leaving an inconsistent nonlinear program. That is why the *most* aggressive
damping fails hardest, scoring 0 and 2 of 480 while *increasing* the runaway count to 142
and 106: with the gradient scaled to nothing, the solver has neither the signal that
would carry it into a high-gain region nor the one that would carry it out.

Levenberg-Marquardt damping is sound where it applies to the **Newton step** -- the
linear system -- rather than to a reported derivative. Drake's IPOPT does not expose the
step computation, so the well-posed version of the idea is not reachable from this repo,
and the reachable version is the one measured and refuted here.

One post-hoc pattern, recorded as post-hoc rather than as a finding: on the two
runaway-heavy rows (paired pose, 13 and 18 runaway cells at the default), moderate
damping is positive -- `jacobian_max_norm=100` is +4 on the Panda and +8 on the iiwa,
pooled 27 / 15, p = 0.088, with runaway cells 31 -> 11. It was chosen after seeing the
data and is not significant. Note also that `jacobian_tikhonov_lambda=10` cuts the
runaway hardest on those rows (31 -> 7) while scoring exactly 19 / 19 on success:
**suppressing the runaway does not buy success**, the same conclusion Stage F2 reached
about lifting `q`.

**Thomas's ruling (2026-09-03): "I think we can conclude gradient regularization and the
other strategies isn't worth it."** The knobs stay in the tree, off, with this table as
the reason not to revisit them.

**Stage H was built and not run.** It would have crossed the winning regularization
against the other knobs this campaign has swept (`correction_cost_weight` at 0 and 30,
`latent_cost_weight`, `collision_influence_offset`, `ipopt_mu_strategy=adaptive`,
`lift_q`), 256 items. Its premise is a regularization setting worth crossing, and there
is none, so it dies with Stage G. `stage_H` in `cluster/gen_manifest.py` is a generic
cross-test harness and is kept for whatever knob next needs one.

**Where this leaves the runaway.** Every remedy that does not touch the chart has now
been measured and none works: the latent trust region and the `c` box (inert, and unable
to exclude the regions at all), IPOPT's scaling options (Stage E), a joint-limit penalty
(Stage F), lifting `q` (Stage F2, net negative), and Jacobian regularization (here).
**A better iiwa checkpoint remains the preferred remedy and is now the only untried
one.**

### Solver logs: node-local, and one archive per run (2026-09-03)

`src/benchmark.py` wrote one ~20 KB IPOPT log per (cell x arm) straight into the run
directory on the shared filesystem. By Stage G the campaign had accumulated **35,596 of
them -- 87% of every collection's file count and 78% of its bytes** -- which is exactly
the many-small-files pattern SuperCloud's guidance warns against. Lustre is metadata-op
bound on files that size, so a routine collection had drifted from three minutes to
**thirty**, and it got worse with every stage.

Three changes, and the numbers they bought:

| | before | after |
| --- | --- | --- |
| files in a collection | 40,733 | 2,964 |
| bytes | 950 MB | 316 MB |
| wall clock, full collection | ~30 min | **43 s** |
| loose `.txt` under `results/` on the cluster | 35,596 | **0** |

- **Per-cell logs go to node-local `$TMPDIR`** (keyed on the run tag *and* the pid --
  `$TMPDIR` is per node and a node runs eight workers) and are rolled into one
  `solver_logs.tar.gz` per run at the end of `run_grid`. Measured 3.7x compression on
  real IPOPT logs, and every log stays individually recoverable with `tar xf`, so this is
  a storage change and not a loss of diagnostic detail. With no scratch directory -- a
  laptop run -- the logs stay in `log_dir` and are rolled up in place, so the two
  environments differ only in where the intermediates live.
- **`collect_results.sh` is incremental by default**, shipping only what has been written
  since the last *successful* collection (`.last_collect` on the cluster; the watermark
  advances only after the extract and merge succeed, so a half-finished transfer is
  retried in full rather than silently skipping data). `--full` forces the old behaviour.
  `state/` is no longer shipped at all -- its done markers and claim directories are
  load-bearing for resume *on the cluster* and are never read locally.
- **`cluster/compact_logs.sh`** retroactively folds the logs of runs predating the change
  into the same archives. It runs **inside a debug-cpu job**, not on the login node
  (compression is a job's work); only `--dry-run` and `--status` go over ssh. It archives
  before removing, removes only the names it archived, and folds into an existing archive
  rather than clobbering it, so it is safe to re-run. It compacted all 35,596 logs across
  1,360 run directories with 0 failures; all 1,360 archives were then verified to contain
  exactly the loose files they replaced, with content hashes checked on a random sample.

**A guard bug this surfaced, worth remembering.** Both `compact_logs.sh` and
`collect_results.sh --reclaim` refused while *any* job was running, via
`LLstat | grep -c RUNNI`. SuperCloud accounts are **shared across Thomas's projects**, so
that check fired on a `run_matrix.sh` job belonging to an unrelated campaign. Only
`run_items.sh` writes into `~/learned-ik/results`; both guards now filter by that job
name, and count `PENDING` as well as `RUNNING` since a queued worker could start part way
through. Any cluster-wide check on a shared account must be scoped to this project's own
jobs.

### The success criterion is settled: the gap is deliberate and stays (2026-09-03)

**Thomas's ruling: `ik_constraint_tol = 1e-4` for the program's rows, `task_tol = 1e-3`
for the task gate -- the order-of-magnitude gap is a design choice and must not be
closed.** In his words: *"go back to 1e-4 actual tol and 1e-3 task tol, to avoid this
issue (that's why I did it in the first place)."* This settles the open question the
section below records, in the opposite direction from the one a session briefly took: the
two tolerances were matched at 1e-4, the consequences were measured, and the match was
reverted.

**Why the gap has to be there, measured.** An interior-point method parks *on* an active
constraint, so at convergence the gated quantity sits at the bound plus or minus
rounding, not comfortably inside it. Stage D, iiwa pose paired, 480 cells:

| arm | median `pos_error` | > 1e-4 | > 1.01e-4 | > 2e-4 |
| --- | --- | --- | --- | --- |
| learned | 9.9971e-05 | 33% | 1% | 0% |
| joint space | 1.0001e-04 | **64%** | 0% | 0% |

Every solution is pinned to 1e-4. A gate at exactly 1e-4 therefore scores which side of
the bound the last ulp fell on -- and it does not fail quietly: it *appeared to reverse*
the one row the learned arm loses, iiwa pose paired going from 296-vs-332 (p = 0.016
against) to 199-vs-120 (p = 2.5e-08 in favour). That reversal is a coin toss dressed as a
result, and it is exactly what the looser gate exists to prevent. The gate answers "did
the arm reach the target", not "whose rounding is smaller".

**The general rule, which this repo already applies elsewhere: never set an acceptance
gate equal to a bound the solver is optimising against.** The collision gate carries the
binding's own slack for the same reason (a converged solve reports a collision value of
1 + 1e-7). Before proposing to tighten any gate, check the distribution of the gated
quantity against the bound; if the solutions are pinned to it, tightening measures noise.

**Every table in this file is unaffected** -- the restored gate reproduces the stored
verdict on all 8,339 Stage D task-gated records, exactly.

**One repair kept from the exercise.** The pose gate's orientation bound was
`10 * task_tol`, which equals `ik_constraint_tol[1] = 0.01` only by coincidence at the
default `task_tol = 1e-3`; overriding `--task-tol` would silently have dragged the
orientation gate with it. It now reads `ik_constraint_tol[1]` directly -- identical at the
default, trap removed.

### The dual success criterion, first measurements (2026-09-02)

**The open question this section records is now settled** -- see the ruling above: the
gap between the two tolerances is deliberate and stays. Kept for the measurements.


Every record now carries `max_violation` (the largest constraint violation at the returned
point), `detail["violations_all"]` (every binding's worst *signed* violation, so slack
reads as negative), and a second verdict `feasible_relaxed` scored with the program's
constraint tolerance relaxed from `ik_constraint_tol = 1e-4` to the task gate's
`task_tol = 1e-3`. Which of the two the paper's criterion should be is open; recording the
continuous quantity means settling it later is a re-analysis and not a re-run.

What the relaxation is worth on the sc_A grasp finals, per 60 cells: learned +1 to +3,
numerical 0 to +2, analytic 0 to +1. **It moves no qualitative conclusion** -- the iiwa
learned arm goes 18 -> 21 against numerical's 58 -> 60.

The more informative number turned out to be `median_max_violation`, which separates the
arms by six orders of magnitude:

| arm | median max violation |
| --- | --- |
| analytic / analytic8 | 2e-16 to 5e-09 (machine precision) |
| numerical | ~1.1e-08 |
| learned, Panda | 4e-05 to 6e-05 (below `ik_constraint_tol`) |
| **learned, iiwa** | **1.7e-02 to 2.9e-02** |

The iiwa learned arm's *median* cell is three orders of magnitude outside tolerance, so
**its deficit is not a near-miss story and no tolerance choice rescues it** -- the typical
cell is grossly infeasible when the clock runs out, not sitting just outside the gate.
That is consistent with 40-42 of its 60 cells exiting at the cap, and it is what the
correction-cost result above then largely repairs (2.9e-05 at weight 1.0).

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
3. IPOPT `mu_strategy="adaptive"` swept (inert, both robots). `nlp_scaling_method` was
   **not** in fact plumbed despite this list saying so; it and `nlp_scaling_max_gradient`
   now are (`ipopt_nlp_scaling_method`, `ipopt_nlp_scaling_max_gradient`) and Stage E
   sweeps them, because the runaway diagnosed above puts a 1e11 row in front of a scaling
   scheme that computes its factors at the starting point and caps them at 100.
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
7. ~~Report success against the wall-clock cap as a curve rather than one number~~
   **done** -- Stage C, six caps from 5 s to 180 s on all eight experiments. Every
   baseline is flat at every cap; only the learned arm moves. The Panda grasp deficit
   closes at 180 s and the residual failures turn out to be a frozen divergent set rather
   than slow convergence.
8. More guesses per target in the paired grid -- same guesses for every arm -- reported as
   "solved within k restarts". This is the only honest form of multi-start and the harness
   already does it.

**iiwa -- and the chart-accuracy hypothesis is now dead**

The iiwa flow is a 4-8x worse chart than the Panda's (16.6 mm / 6.4 deg median against
3.8 mm / 0.71 deg), and that was the standing explanation for its grasp deficit. **The
dose-response experiment refutes it** (see "Chart accuracy is not what is wrong with the
iiwa"): degrading the Panda's own chart to the iiwa's accuracy costs 1-3 cells of 60, where
the iiwa is 23 cells worse. Whatever the iiwa row is, it is not chart precision, so the
open question is now *what* it is -- start with the 39-44 cells that exit at the cap having
taken 299-364 iterations, roughly 2.5x the joint-space arm.

9. ~~Sweep `correction_bound` upward~~ **done, and the premise was wrong**: the box is not
   binding on either robot (0.00 of solutions sit on it; median `|q_c|` is 0.054 against a
   bound of 0.1), and widening it to 0.8 changes nothing.
10. Ask Julia about `iiwa14__lemon-haze-7__global_step_4.25M.pkl` -- **now the sharpest
    question in the project**. The checkpoint puts 3.34% of the conditioning domain into
    the network's worst-case-gain regime against the Panda's 0.065%, and that 51x is the
    iiwa grasp deficit. `rnvp_clamp = 2.5` is confirmed correct, so this is the weights,
    not the loading. The superseded note read: The dose-response experiment shows
    chart error of the iiwa's magnitude costs the Panda 1-3 cells, not 23, so the checkpoint's
    *accuracy* cannot be the mechanism. Its provenance is still worth knowing (retraining is
    already planned) -- it simply is not the explanation this row needs.
11. ~~The divergent cells~~ **solved** -- see "The residual failures are the flow's own
    gain". The row is the joint-limits row on configurations of 1e7-1e16 rad; every
    runaway on a given robot lies on one ray; and direct sampling of the network
    reproduces it with no solver involved. The old note read:
    Stage C measured them: the violated binding is `AllIKFlowConstraints` on 19 of 19
    cells while every variable region reports slack, so it is *not* the latent or the
    conditioning pose escaping the trust region. The cells are deeply in collision
    (median -8.5 cm) about a metre off target, with the correction pinned near its box.
    Identify which row inside `AllIKFlowConstraints` reaches 1e7-1e11 -- it does not track
    `collision_value` -- which is a re-analysis of the persisted `q`, not a re-run.

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
