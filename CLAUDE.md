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
python scripts/panda/panda_mug.py            # 3-way mug-grasp comparison, panda
python scripts/panda/panda_mug_headtohead.py [num_tests] [max_wall_time]
python scripts/panda/panda_mug_ablation.py [config_index] [num_tests] [max_wall_time]
python scripts/panda/panda_orientation_headtohead.py [num_tests] [max_wall_time]
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

Each robot implements `__init__` (frames, plant sizes, model loading), `create_prog` (declares decision variables, sets initial guesses, builds `self.jacobian_gen`, calls `add_constraints` / `add_costs` / `SeedInitialGuess`), `ik_inference`, and `VarsToQ`. The `...MugProgram` subclasses additionally swap `self.frame` from the end-effector (the frame the flow was *trained* on) to `between_fingers` (the frame the grasp constraint acts on), keeping `X_grasp_ee` so seeds can still be expressed in the network's frame. A mug grasp constrains only the gripper's position in the mug frame (xy tight, z within `mug_height`), leaving orientation free — hence the overridden `CreateIKConstraint` and `SeedCandidates`.

### Gradients through the flow

`VarsToQ` is dual-path: under `float` it returns a plain forward pass; under `AutoDiffXd` it calls `self.jacobian_gen = torch.func.jacrev(ik_inference_with_value, has_aux=True)` (one reverse pass yields both `dq/dvars` and `q`) and chain-rules `jacobian @ vars_gradients` into fresh `AutoDiffXd` objects. Analytic formulations instead evaluate `pydrake.math` trig on templated types (`RigidTransform_[T]`, `RollPitchYaw_[T]`) so Drake's own autodiff propagates.

Numerical facts worth not rediscovering — several are recorded in code comments and encoded in `ProgramOptions` defaults:

- Evaluate the flow in **float64** (`use_float64=True`). Gradients are analytic (`jacrev` through the flow, chain-ruled into `AutoDiffXd`), so this is not about the solver differencing anything: it is that a float32 network produces *values* with a ~1e-7 noise floor, which corrupts every quantity computed as a difference over a small step — line-search actual-vs-predicted reduction, convergence tests, and SNOPT's optional derivative verification. `snopt_function_precision` tells SNOPT that noise floor when running in float32.
- Don't ask for a position tolerance below the flow's noise floor: `ik_constraint_tol=(1e-4, 0.01)`.
- The orientation row of the IK constraint has three selectable evaluators in `src/generic_program.py`, chosen by `ProgramOptions.orientation_error_form`. `legacy` is the original `2*arccos(|q.q_target|)`, whose derivative is infinite at zero error and whose `eps` clamp silently produces an *empty* derivative vector when it binds — freezing the row's value at 2.83e-4 as well. `angle` keeps the same metric but evaluates it through the series for `u = 1-d` and assembles the `AutoDiffXd` by hand, so the value stays accurate and the derivative is finite (bounded at 1.41e6 by flooring `u`). `squared` (the default) uses `theta^2/ori_tol**2`, which has no branch point at all: the derivative tends to −8. `lb` for the squared row must be `-inf`, since `theta^2 >= 0` with a vanishing gradient at the solution would otherwise be an active row with no gradient. All three impose the same feasible set, `theta <= ori_tol`, and `ik_constraint_tol[1]` stays an angle in radians. Note that writing `(2*arccos(d))**2` directly does **not** work — the chain still differentiates through `arccos` at the branch point. `scripts/panda/panda_orientation_headtohead.py` compares the three.
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

### Scenes and utilities (`src/utils.py`, `models/`)

`BuildEnv(meshcat, directives_file, extra_directives=None)` builds the diagram from a Drake model-directives YAML, registering `package.xml` so `package://combining_kinematics/...` URIs resolve; `extra_directives` is a list of `ModelDirective` objects appended to the loaded ones **in memory**, so a caller can add models to a scene without writing to the tracked YAML. `GenerateDiagramWithMug(q, program, yaml_file, meshcat)` uses exactly that: it constructs an `add_model`/`add_weld` pair for a mug at the gripper pose of `q` (the weld pose is passed as a `pydrake.common.schema.Transform`, not formatted into text) and rebuilds the diagram. The YAML on disk is never modified, so a crash or interrupt mid-call cannot leave a stray mug in a tracked scene — it used to append-then-truncate the file, which could. Targets in the mug experiments are generated by sampling collision-free `q` and welding a mug at the resulting gripper pose, so every target is known to admit a valid grasp.

`HiddenPrints` suppresses Drake/ikflow output at the file-descriptor level and is used around program construction inside sweeps.

Notebooks in `notebooks/` are the exploratory counterpart to `scripts/` and import the same `src/` modules; they run from the `notebooks/` directory (they append `../` to `sys.path`).
