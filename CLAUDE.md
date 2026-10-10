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

**There is no iiwa analytic *program*, which is not a statement about the iiwa.** The iiwa14 is an
S-R-S arm and has a closed-form solution (Faria et al.); what this repo lacks is an implementation of
it. `src/iiwa_analytic_ik.py` is down to the joint limits -- the closed-form map was deleted rather
than maintained unreached -- no `Iiwa14IKProgramAnalytic` exists, and `scripts/iiwa/iiwa_benchmark.py`
registers only `learned` and `numerical`. Writing that arm is future work or possibly not done at all
(Thomas, 2026-09-19). **Comparing against analytic IK is not the objective**: the Panda carries an
analytic column only because we happen to have its solution implemented, so describe the asymmetry as
an accident of implementation, never as three-way-vs-two-way being a property of the robots.

All three go through the same `IKFlowProgram` machinery, so a change to constraints/costs affects
all of them. `workshop-paper-draft.pdf` is the write-up — an early rough draft, orientation only,
not a replication target. The sibling repo `../codebase/` is the analytic-vs-numerical project this
one builds on (`scripts/iiwa/iiwa_collision.py` imports from there); treat it as read-only.

`docs/closed-axes.md` carries the **evidence** for questions this file records only the **verdict** of
(the solver axis, the chart ladder, the refuted remedies, step rejection). Read it before reopening or
re-sweeping any of them; do not summarise it back into this file.

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

### Profiling, and CUDA graphs (`flow_cuda_graph`)

Recover the old profiler with `git show ab3ea15:scripts/profiling/profile_flow.py`. At batch size 1
the flow evaluation **was CPU-dispatch-bound**: ~70% of a `jacrev` is describing ~2853 ops to the GPU.
**`--compile --set flow_cuda_graph=True` removes that** (Thomas asked for it, 2026-10-08). The
compiled `jacrev` and a compiled `no_grad` forward pass are each captured once per process as a CUDA
graph and then replayed (`GraphedFlowCall`). Afterwards a call is GPU-bound: kernel time ≈ call time.
The old "~3x ceiling" is superseded.

- **How it is used.**
  - It requires `--compile`.
  - Graphs are captured in `WarmUpJacobian` before the grid, then frozen. A capture inside a timed
    solve raises, because compile and capture are offline compute.
  - Like `--compile`, it moves the learned arm's success within a cap, so **every compared run sets
    it the same way**.
- **Why manual capture.**
  - Eager `jacrev` cannot be captured: jrl's import-time `set_default_device` makes it do a
    host-to-device copy.
  - `mode="reduce-overhead"` exceeds the cudagraph-trees re-record limit and falls back to ungraphed.
  - Probe: `scripts/profiling/probe_cuda_graphs.py`.
- **Why the forward pass too.**
  - Under IPOPT, Drake evaluates constraint values in double and costs in AutoDiffXd. So each trial
    point is one `jacrev` plus one plain forward pass.
  - That forward pass was eager and recorded autograd even under `--compile`: 7.1 ms against the
    compiled `jacrev`'s 4.4 ms on a V100.
- **Exactness.**
  - Replay is bit-identical to the compiled Jacobian.
  - The forward pass, now compiled, agrees with eager to ~1e-14.

**Measured end to end** (stages CUDAGRAPH / CUDAGRAPHP2 / CUDAGRAPHMPS; IPOPT, the record's 8 rows
on the record's grids; `scripts/report_cudagraph.py [P2|MPS]`), at one process per V100:

- Learned ms/it falls **2.3-2.6x**.
- Iterations are identical on 100% of mutually solved cells, and no cell is lost.
- The per-iteration premium over joint space falls from ~8-11x to **3.1-4.2x**.
- Both iiwa contained-grasp ties become learned wins: 472 and 476 v 452, p ≤ 0.016.

**The record (stage REMEASURE) runs with graphs on.**

**Processes per GPU now matter.** Without MPS, processes sharing a V100 are time-sliced, and a graphed
process is GPU-bound. One IPOPT trial point (one `jacrev` plus one forward pass):

| processes per GPU | speedup from graphs |
| --- | --- |
| 1 | 3.6x |
| 4, without MPS | 1.35x |
| 4, with `MPS=1` (`submit_bench.sh` starts a job-local daemon) | 3.0x |

End to end (stage CUDAGRAPHMPS, the same 8 rows at PROCS=8 under `MPS=1`), graphed learned ms/it is
**1.12-1.19x** slower than at one per GPU, against 1.31-1.91x without MPS. That is about 3.4x the node
throughput. Iterations are identical on 100% of shared cells, nothing is lost, and the success verdicts
match one-per-GPU on every row.

The old record ran PROCS=8 without MPS (a **1.15-1.30x GPU-contention penalty** joint space did not
pay); stage REMEASURE ran PROCS=8 under `MPS=1`, so its learned wall times carry the 1.12-1.19x above
and are development-grade. **Paper numbers run at one solve per GPU
(PROCS=2)** (Thomas: *"those are the conditions in which the final paper results will be drawn"*).
4 per GPU under `MPS=1` is for development throughput only. Never run 4 per GPU without MPS once graphs
are on.

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

**TRAP: the mug handle must point through the finger gap, and on the wsg finray it did not.**
`GenerateDiagramWithMug` welds the mug at the FULL pose of `between_fingers` at q*, handle along
that frame's +x. `panda_finray.sdf` gives the frame yaw 1.57, so the handle runs out past the
fingertips; `wsg50_110_finray_fingers_box_collision.sdf` -- the gripper of the iiwa, soft PCS, screw
and GVS scenes -- gave it yaw **0**, so the handle pointed **18 mm into the right finger plate** on
150/150 collision-free q*. Nothing could catch it: the sampler's `collision_free` runs without the
mug, `FloatingMugScreen` filters the robot out, and welded mug against welded finger is
anchored-vs-anchored, which Drake never reports. Found 2026-10-08 by `scripts/probe_mug_contact.py`
(after the fix: 0/150 contact, +9.5 mm clearance, Panda unchanged); the fix is the one-token yaw in
that SDF (452d784), and no code depends on the yaw. **The Panda never had it. Every wsg grasp number
before stage REMEASURE is void** -- stages STATUSQUO, SOFT12/SOFTCHART/SOFTDOF/SOFTCAP,
SCREW/SCREWCHART/SCREWPITCH/SCREWCAP, GVS and ITCAP on the iiwa, soft PCS, screw and GVS grasp rows.
Every run now records `metadata["scene_fingerprint"]` (sha1 over the directives YAML, every model
file it references and the in-memory mug; first 8 hex in the auto tag), because `grid_hash` hashes
only q's and a defective-scene run shares a grid with a fixed-scene one; `collate --pair` refuses a
fingerprint mismatch. **Never pair cells across a scene change** -- compare verdicts within runs.

**Cross-robot settings, ADOPTED 2026-10-09** (Thomas: *"Adopt the settings unification everywhere,
including the panda joint limits."*). An inventory found settings that differed between robots for
no deliberate reason; each is now one code path in `generic_program.py`, measured by stage
REMEASURE:

1. Native grasp start `c` = the flow-frame pose of the generating configuration,
   `mug.middle @ X_grasp_ee` as xyz + rpy (`GraspCStart`; the iiwa and screw arms seeded
   `[mug xyz, 0, 0, 0]`). The PI accepts that this conditions the learned arm on a known-valid grasp.
2. The joint-space variable bound is `ConfigLimits()` everywhere (`QBoundingBoxConstraint`). **The
   Panda used +-10 rad**, the iiwa a hand-typed table 1e-6 off the plant's. The only setting that
   moved a cell: Panda grasp joint space **374 -> 449** (+96/-21, p = 1.2e-12), median iterations
   908 -> 206.
3. `q_nominal` is a nonsingular, in-limits, collision-free home per robot (`NominalConfiguration`):
   Panda Franka's ready pose, iiwa and screw7 `[0, 0.6, 0, -1.75, 0, 1.0, 0]`, the soft PCS and GVS
   arms the straight rod. Zeros was singular on the rigid arms and, on the Panda, outside q4's range.
4. The grasp `c` box is centred on that same flow-frame pose, +-`c_position_slack`
   (`GraspCBoxConstraint`), not on the mug; the five per-robot copies of the box are gone.
5. The latent trust region was an **A/B, and it is inert**: `latent_trust_region_rule`
   (`--config latent_rule`, radius `sqrt(dim) + 1.5`: Panda 4.0 -> 4.15, iiwa 4.3 -> 4.33) ties on
   all eight Panda/iiwa IPOPT rows, so **the per-robot radii stand**.

`--set legacy_robot_settings=True` restores 1-4 together and is **a control only** (stage
REMEASURE_LEGACY), never a record setting; what each arm built with lands in
`metadata["robot_settings"]`, and `tests/test_robot_settings_unified.py` pins both states on all five
robots.

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

**A cap check before reporting any loss**: does the losing arm sit at a **budget** — and **read BOTH
`timed_out` AND `hit_iteration_cap`, never timeouts alone.** Neither ~0 → the cap is innocent and the
result is a formulation result. Either significant → the cap is measuring throughput, not
formulation; raise it and re-measure. Established by iiwa `n4` contained grasp: 391 v 442
(p = 1.4e-06, a clear loss) at 45 s with 88 timeouts; at 180 s with 0 timeouts it is 447 v 442, a tie
— the arms were tied all along.

**But that 180 s reading used timeouts only, and `max_iter` is the budget it missed.** `max_iter`
defaults to `None`, so IPOPT runs at **its own default of 3000 iterations**, and a cell can reach
3000 *inside* the wall-clock cap — a stop `timed_out` does not record and `hit_iteration_cap` does.
On the record's own grasp rows (2026-09-28) the iiwa learned arm had 27-29 of its 27-33 failures at
3000 iterations with a median ~100 s of 180 s unspent, and Panda grasp's joint-space arm 80 cells at
a median 14 s; the 45 s → 180 s move had converted a wall-clock stop into an iteration-limit stop.
**Every such row has now been re-measured with the budgets lifted** -- the screw arm's by stage
SCREWCAP, the other 23 rows (STATUSQUO, SOFT12, SOFTCHART, SOFTDOF) by **stage ITCAP, 2026-10-08**
-- on the same grid, chart and 180 s clock, IPOPT at `max_iter` 1e6 and SNOPT at 1e5 majors / 1e8
minors (`scripts/report_itcap.py`); stage REMEASURE carries the lifted budgets on every row. **No
verdict of the then-record moved** (its wsg grasp numbers are since void, see the mug-handle trap): the two iiwa contained-grasp
ties are now ESTABLISHED ties (461 v 452, 464 v 452), every SQP loss stands, and IPOPT cells at the
new budget are 0 everywhere. **What the budget had been binding was mostly the BASELINE**: joint
space gains 10-12 cells on every soft grasp row, 27-28 on `soft16` and 51 on Panda grasp, the
learned arm 0-14. Three soft LADDER verdicts moved (soft PCS arm's section). Thomas reversed his
2026-09-28 "record the caveat, re-measure nothing": **a row with >= 24 of 480 cells at a budget
carries no verdict until it is re-measured with the iteration budget lifted (the clock never is)**;
within a push, flag it and re-run it as a follow-up. Lifting SNOPT's has one cost:
**SNOPT does not check its time limit on a major with zero minors**, so a cycling cell runs to the
major limit -- 0.4-2.2 h at 1e5 -- and none has ever been scored feasible, so no verdict depends on
it, but mean-wall columns carry the overrun and an item timeout must allow for it. **The cap is a budget for the arm that evaluates a
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
behaviour. `start_q_error` measures the *initial guess*, and `clip_distance` how far the start sat
outside its own regions — but **that second number is summed over regions of two different kinds and
must not be read as a projection everywhere.** Where the region is a genuine variable bound (the
joint-space arms' `q`, the analytic arm's `psi`, and its `xyz_rpy` on the grasp task) IPOPT's
`bound_push` really does move iterate 0 by that much. Where it is a general constraint — the learned
arm's `c` and `z`, the analytic arm's `xyz_rpy` on the pose task, which is the whole point of those
regions not being bounds — nothing projects, iterate 0 is the guess as written, and a nonzero value
means only that the solver was handed a start outside the region and walked it in. On the latent
that is the intended behaviour and the measurement that established it (start `|z| ~ 7.9` against a
radius-4.96 region, solution `|z| ~ 2.9`), not a loss.

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
point is that we test interior point, augmented lagrangian, and SQP."* So `--solver` is `ipopt`,
`snopt`, and `nlopt` (**`LD_AUGLAG`** -- not `LD_AUGLAG_EQ`, which absorbs only equalities and leaves
inequalities to the inner solver, and this program carries both). **The solver is a reporting axis,
never a choice**: *"we would not pick one solver or the other, but rather report the performance for
both solvers."* Shared across arms within a run, varied across runs.

**The whole axis is CLOSED on all three method classes. Do not re-sweep any of them.** Evidence, the
per-stage arithmetic and the refuted mechanisms live in **`docs/closed-axes.md`**; each stage's reader
(`scripts/report_{snopttune,snoptcombo,nlopttune,step}.py`) regenerates its tables with its
pre-registered rule inline so it cannot drift.

### Each solver at its own defaults, and why that is a CHOICE

Unset `snopt_*`/`nlopt_*` fields are not passed, so each solver sits at its own defaults and the
shared, deliberately looser task gate decides success. **Do not transplant one solver's option values
onto another** -- the SNOPT branch once read IPOPT's `acceptable_*` family (its relaxed early stop,
not what it converges to) as SNOPT's Major tolerances.

But "own defaults" is not symmetric. `tol`, `constr_viol_tol`, `dual_inf_tol` and `compl_inf_tol`
were never `ProgramOptions` fields, so every archived run gave IPOPT `constr_viol_tol` **1e-4**
against SNOPT's **1e-6** and `dual_inf_tol` **1** against **2e-6** -- 100x the constraint violation
and six orders more dual infeasibility, plus an early stop SNOPT has no counterpart for. A property
of the HARNESS, not of interior-point methods. All four are now plumbed. The task gate is untouched
by any of it.

### What each solver will and will not tell you

**No solver reports an iteration count through Drake.** IPOPT's comes from its print file; SNOPT's
`details` carries `info`, `solve_time` and multipliers but no count; `NloptSolverDetails` carries a
single `status` and NLopt writes no print file at all (`kPrintFileName` is silently ignored).
`iterations` means **majors** under both solvers that report it. snOptA has one user function, so
IPOPT's four eval counts have no SNOPT counterpart and stay `None` rather than being filled with a
different quantity.

**The program counts evaluations itself** in `QAndPose`, the one funnel every arm's solve passes
through. `map_jacobian` is the AutoDiffXd count -- the NLopt column's only work measure -- and it
cross-validates: on the SNOPT smoke run it equalled `User function calls (total)` exactly on every
cell. It is NOT an iteration count, and it is **not comparable across arms**, being the identity map
on the joint-space side. **Status is decoded numerically, not from log text**: SNOPT INFO 34 is the
time limit with no distinctive exit string, NLopt has no text at all, so matching strings would report
`timeouts: 0` for a capped run of either. NLopt status 5 is `MAXEVAL_REACHED`, recorded as
`hit_eval_cap`.

### The result: IPOPT > SNOPT >>> NLopt

**A CONFIRMATION, not a finding.** Thomas: *"IPOPT is more robust to ill-posed problems, and our
neural network gradients are definitely ill-posed. I expect to see IPOPT > SNOPT >>> NLOPT."* Write it
up as the size and mechanism of a predicted gap. On stage SOLVER2's 480-cell grid IPOPT won all 24
rows, 23 significantly.

**It is a property of the PROBLEM, not of the learned formulation** -- the joint-space arm degrades
under SNOPT too, on every row and by comparable margins, and that arm never evaluates the network.
The same holds for every solver setting: each moves both arms.

**The mechanism, and it is not budget.** Pooled over all 3,952 SNOPT failures: INFO 13 (nonlinear
infeasibilities minimized) **50.7%**, INFO 41 (current point cannot be improved) **36.2%**, iteration
limit 9.5%, time limit **3.6%** -- about 87% convergence failures against 11% budget, and SNOPT times
out *less* often than IPOPT. The two fail in opposite ways: IPOPT's learned-arm failures are mostly
wall-clock, still descending; SNOPT's are INFO 41, giving up at a feasible-but-wrong point. **INFO 41
is the documented signature of inaccurate or badly scaled derivatives** -- ordinary ill-conditioning
of an *exact* Jacobian, not the gain-ceiling runaway. The gap is much larger under `paired`, so SNOPT
copes far worse with an infeasible start; a 60-cell triage read that backwards in both directions
before 480 cells settled it, so **do not draw a protocol conclusion from 60 cells**.

**Wall-clock columns are not capped equally across solvers**: NLopt's `max_time` binds much more
tightly than SNOPT's `Time limit` (20.0-20.1 s against 24-28 s), because SNOPT only checks at major
boundaries.

### What tuning is worth: two settings fielded, everything else plumbed and `None`

**Per-solver tuning is permitted and per-problem tuning is not** (*"I'm okay with playing with solver
settings on a per-solver basis, as long as it's not per-problem"*), so a setting qualifies only by
winning **uniformly across rows**, **counted by row, never pooled** -- the transferable lesson, since
a four-factor SNOPT stack beats the fielded survivor by +221 pooled cells and is a **task trade**
pooling hides, gaining on every iiwa grasp row and losing on every pose row.

The verdicts: IPOPT's convergence tolerances are **inert** and its acceptable-point machinery is
everything -- at each solver's own defaults **IPOPT 200, SNOPT 141, p = 4.1e-09**, so the fairness
question is answered and the ordering survives it. IPOPT's early stop is worth 39 cells and a 9x
speedup and is **not** returning sloppy points (disabling it drives the violation to 2.22e-15 and
success *down*, because IPOPT then polishes a solution it already has until the clock kills it).
Naming NLopt's inner optimizer is worth nothing; truncating its inner tolerances is worth a great
deal, as a trade whose sign flips by row. **Three things no NLopt setting changes**: the iiwa grasp
rows are 0-3 of 60 under every setting and at 180 s (on the defective wsg scene; fixed, native is 313 of 480); the joint-space arm moves with these settings
too; the ordering is untouched.

**Exactly two settings are fielded**, both adopted 2026-09-19.

**SNOPT `snopt_major_step_limit = 0.5`** (Drake's own default is 2.0), sole survivor of thirteen, and
no combination beats it. **Adopted for FAIRNESS, not for the comparison** -- five learned wins, four
joint-space wins and three ties before and after, **zero verdict flips**, +161 cells to learned and
+164 to joint space, per-row margins summing to **-3**. What justifies it is that IPOPT's column ran
a tuned configuration while SNOPT's ran bare Drake defaults. **Do not present it as helping the
learned formulation.** Two consequences: the SNOPT numbers of record are stage SNOPTCOMBO's
`mstep0p5` column, and **"set nothing" no longer means Drake's SNOPT defaults** -- a stage meaning
that must say `--set snopt_major_step_limit=None`.

**NLopt `LD_AUGLAG` + `LD_MMA` inner + inner `xtol_rel = ftol_rel = 1e-3`**, on Thomas's criterion
*"Feasibility is the name of the game, objective cost is secondary."* Better on 5 of 12 rows, worse
on 1, unchanged on 6, and it collapses both residual and work per cell (Panda grasp contained native
12 -> 42 of 60 at 3532 -> 66 network Jacobians). **Three things must be reported with it**: it
FAILED stage NLOPTTUNE's pre-registered gate, which asked whether to spend 480-cell compute rather
than whether the setting is best -- state the gate failure alongside it; the Panda pose native row is
a genuine regression on the adoption's own criterion (five cells, residual 9.3e-07 -> 3.6e-06); and
it flips one learned-vs-joint-space verdict, unlike the SNOPT adoption which flipped none. A caveat
we are not re-sweeping: the sibling `ik-tune` project puts the same inner tolerance's optimum near
1e-4, so our 1e-3 is in a sensible region but is not an optimum this project established.

**`LD_AUGLAG` is kept for a reason in NLopt, not in Drake**: under it the inner optimizer solves a
**bound-constrained** subproblem and every constraint sits in the AL penalty, whatever the inner
algorithm's own class -- which retires the `LD_SLSQP` taxonomy worry and is why `_EQ` is not used.
Verified in the bundled nlopt source. **An earlier version argued from Drake's side -- that Drake
never adds a constraint to the inner `local_opt` -- and that reasoning must not come back**: true,
causes nothing, and predicts the wrong thing under `_EQ`. A live hypothesis this hands us, not acted
on: `ik-tune` finds the plain-vs-`_EQ` difference large and one-directional, and our iiwa grasp rows
are exactly where the collision inequality binds -- so penalised-rather-than-enforced inequalities is
a mechanism candidate. **Switching is a method-class decision and therefore Thomas's.**

### Step rejection: measured and refuted (stage STEP), with one real finding

A 16-setting screen at 60 cells reached significance nowhere; the 480-cell confirmation **refuted
both promoted settings** -- `theta1` is significantly *worse* (p = 0.00032) and `soc0` is a clean
null. **The mechanism column reversed too**, so a plausible mechanism agreeing with a spurious
outcome did not protect against the false positive. Nothing adopted, by Thomas's call made in
advance.

What it exposed is the real finding: **per-cell outcomes are unstable.** Pooled over 5,760
comparisons, a single filter-option change recovers ~70% of the learned arm's residual failures while
breaking 6-8% of the cells that already worked; successes outnumber failures ~12:1, so the two cancel
almost exactly. A same-configuration control flips 5.9%/0.3%, and even failures that converged wrong
rather than timing out recover at 48-67%. So the residual failures are **not intrinsically hard**,
and no single global setting wins because the reshuffle is symmetric in proportion. **The implication
is for multi-start, which Thomas ruled out of scope** -- report the instability as a finding, do not
propose acting on it.

**A methodological number worth keeping**: "reproducibility at the cap is +/-1 cell" was measured on
60-cell grids. At 480 cells with 52-88 timeouts a same-configuration re-run moves up to **7 net and
19 discordant** cells; every one of the 41 discordant cells across 12 such rows was cap-bound, and
rows with zero timeouts had zero discordance. The band scales with the cap-bound population, not the
grid.

### Traps, all found by probing rather than by reading

**A solver option can be ACCEPTED and do nothing**, and only the solver's own parameter echo tells
the two apart -- `"Timing Level"` was set for the life of this repo and silently wrote no timing
block, because SNOPT's parser is case-sensitive on the second word while Drake raises only on
keywords SNOPT does not know at all. Every option this branch exposes was verified reaching its
solver (IPOPT via `used = yes` in `print_user_options`, SNOPT via the parameter echo). It caught
three: `Hessian updates` is inert below 75 variables; `Nonderivative linesearch` is a VALUELESS
keyword, so passing 0 turns it ON; and `linear_solver=mumps` does not exist, Drake's IPOPT being
built against SPRAL.

**Read a solver's defaults out of the solver.** IPOPT's own dump contradicts the obvious assumption:
`acceptable_dual_inf_tol` defaults to **1e+10** and the two acceptable infeasibility tolerances to
**1e-2**. A sweep arm labelled "IPOPT's defaults" that was not cost a resubmission.

**`max_eval` is not "unset" by default** -- Drake defaults it to 1000, a cap that binds here, so it
must be set explicitly or the NLopt column silently measures an evaluation budget rather than the
wall clock.

**The Luksan trap.** Drake's NLopt is built without the LGPL Luksan sources, so `LD_LBFGS`, the
`LD_VAR*` family and every `LD_TNEWTON*` variant are listed as valid and then refused *inside* the
solve, returning `kInvalidInput` and status 0. The symptom is quiet and misleading: a **0.5 s cell
with `q=None`, `max_violation=None` and `fail_reason` unset**, which reads like a harness bug.
`LD_MMA`, `LD_CCSAQ`, `LD_SLSQP`, `LN_COBYLA` and `LN_BOBYQA` all work. It also means there is **no
"name what NLopt already picks" control available**. Two more Drake behaviours: every
`local_optimizer_*` option is read but applied only when an inner algorithm is NAMED, so an inner
budget without one is accepted and inert; and an **unknown** NLopt name is accepted at `SetOption`
and raises from inside `Solve`, landing in `run_grid`'s per-cell `except` as a **full column of
instant failures** rather than an error. Hence the check lives in `ProgramOptions.__post_init__`,
before the first cell, and `tests/test_solver_plumbing.py` bounds the emitted keys by the *running*
Drake's surface.

**Do not extend Drake to get better instrumentation** -- *"NLOPT might not have the robust logging we
need btw, work with what you have, don't write new logging stuff in Drake or anything."*

### ONE DRAKE, AND IT IS THE PIN

The nightly `drake-0.0.20260918-noble.tar.gz` at `$ROOT/drake`; no second install, no per-item
version sentinel, no `stage_DRAKEBUMP` (all three deleted 2026-09-19). Thomas, ruling for the second
time:

> stop running IPOPT and SNOPT on the installed 1.56.0. I've said this already. ... Run everything on
> the current nightly. Even if you think I'm wrong, it doesn't matter, because we're not going to pin
> IPOPT and SNOPT back to an earlier version as Drake moves ahead -- that would be a regression that
> you report so I can fix in Drake upstream, and/or further tuning to fix it.

So **archive pairing is not a reason to keep an old install**. A nightly is needed at all because
**PR 25002** took `NloptSolver` from six option names to sixteen, which is what made the inner local
optimizer selectable; no release carries it, and `local_optimizer_ftol_rel` is absent from 1.56.0, so
`CheckNloptOptions`'s refusal is scoped to `which_solver == "nlopt"` -- unconditional, it would refuse
`ProgramOptions()` itself and kill every IPOPT and SNOPT cell over options they never emit. Nightly
artifacts **expire after 45 days (~2026-11-02)**, so **move the pin to 1.58.0 the moment it carries
PR 25002** and restore the published-checksum path.

### The grasp-containment lever is closed on both axes

It was the standing "revisit whenever another knob moves" lever, because on the contained task joint
space needs 970 median iterations against 48 on the free one. Under SNOPT it never flips a verdict
*toward* the learned arm and flips one away from it. Step rejection was the remaining candidate and is
refuted too, so the containment verdicts stand as measured under IPOPT.

## Results: the campaign of record

**Stage REMEASURE, measured 2026-10-09 and accepted by Thomas 2026-10-09.** It re-measured the whole
record because three things changed under it at once: the wsg scene fix (the mug-handle trap), the
five unified settings, and CUDA graphs with lifted iteration budgets. Conditions: hardened scene,
shelf-contained targets at the **fingertips for both tasks**, **180 s**, 480 cells = 60 targets x 8
guesses, seed 1 (out of sample), `--compile --set flow_cuda_graph=True`, IPOPT `max_iter` 1e6 and
SNOPT 1e5 majors / 1e8 minors, adopted rungs (Panda `n6`, iiwa `n4`, soft PCS `soft12` `n6`, screw
`screw7_p050` `n6`), arms `learned,numerical`, both start protocols, each
solver at its adopted configuration, Drake nightly `0.0.20260918`, **PROCS=8 under `MPS=1`** -- a
development-throughput condition, not the paper's one solve per GPU, so the wall-clock columns are
development-grade.

**THE RECORD IS 48 LOGICAL RUNS: four robots x TWO experiments (grasp, pose) x two protocols x three
solvers**: the 32 IPOPT and SNOPT runs from stage REMEASURE, the 16 NLopt runs from stage
**REMEASURE_NLOPT** (same conditions and tag family, queued last; jobs 5868202/03/04/07, collected
2026-10-10). **The GVS arm is OUTSIDE the record**:
stage REMEASURE measured it under identical conditions and the reporter prints its eight rows as a
separate block with their own tally (IPOPT 3 learned / 1 tie, SNOPT 2 / 2), but it is merged as a
measured robot outside the record and does not replace the soft PCS arm (its section). The old
record's NLopt runs stay on disk, read only by `--legacy`. `--target-placement free` is a retired setting, not a third experiment
(Thomas: *"preserving old settings and old experimental setups is contrary to that mission"*), and is
not reported. `legacy_robot_settings=True` (stage REMEASURE_LEGACY) and `latent_trust_region_rule`
(stage REMEASURE_RULE) are controls, never record rows. Earlier campaigns are superseded and their
tables deleted; git history holds the numbers.

The tables, the verdict tally and the three pre-registered flag criteria live in
**`docs/status-quo-tables.md`**, regenerated by `scripts/report_statusquo.py` (`--legacy` prints the
superseded record). The before/after, the attribution and the acceptance checks are
`scripts/report_remeasure.py`. Layout: rows are experiments, each solver a block with the learned and
joint-space arms **adjacent**, the better of each pair starred, and **every row prints, zeros
included**.

### What the tables say

**Success: interior point 15 learned / 1 tie / 0 joint space, SQP 10 / 1 / 5, augmented Lagrangian
11 / 3 / 2**, against 12/4/0, 7/4/5 and 9/7/0 on the old record over the same sixteen rows; five
verdicts flipped under IPOPT and SNOPT each, eight under NLopt. Verdicts
are exact McNemar within each run; the reporter prints the tally beside the table so text and table
cannot drift.

- **Interior point loses nothing anywhere.** Its one tie is soft PCS grasp paired (477 v 472, a
  learned win before: the scene fix bought joint space +17).
- **The SQP contained-grasp losses survive only from the paired start**: iiwa 276 v 348, screw 259 v
  346 and soft PCS 350 v 377 stand, while the iiwa and screw *native* losses are now learned wins
  (436 v 348, 400 v 346). Two new SQP losses, neither a scene effect: **Panda grasp paired** (300 v
  361, a tie before; joint space gained 55 cells, the joint-limit bound as under IPOPT) and **iiwa pose paired**
  (233 v 272, p = 0.014; a tie before, and that scene did not change).
- **What moved, and why** (attribution, IPOPT, cell-paired where the scene is shared): the scene fix
  moved the wsg grasp rows only, the learned arm by +1.0 to +9.6 points (screw grasp 427 -> 470 under
  the old settings); the settings were null on every row but one, **Panda grasp joint space 374 ->
  449** (+96/-21, p = 1.2e-12), the +-10 rad bound becoming `ConfigLimits()`. No pose row moved
  significantly.
- **The trust-region rule A/B is eight ties**, so the per-robot radii stand.

**Cost (cells both arms solved) is dearer for the learned arm on grasp under IPOPT, 1.9-5.1x on all
four robots** (screw paired 9.36 against 1.83 is the widest); under SNOPT grasp is near-level (1.0-1.3x)
but for soft PCS paired (2.0x) and soft PCS native, where the learned arm is cheaper. It is **cheaper
on rigid-arm pose** (10 of 12 rows; the exception is screw pose paired under both solvers). On soft PCS pose it is dearer under IPOPT and level under SNOPT.
`N/A` means fewer than 10 shared solved cells.

**Runtime (Table 3, mean over all cells clamped at 180 s) is where CUDA graphs show.** Joint space is
faster on most IPOPT rows, but the learned arm is now faster on Panda grasp both protocols (3.52 /
5.86 s against 6.24 / 6.27 s) and soft PCS grasp native, and under SNOPT on three of four grasp native
rows (all but screw). The ratio is largest on the pose paired rows, where a joint-space
solve takes a fraction of a second (soft PCS 9.27 s against 0.37 s, iiwa 1.82 against 0.10); the iiwa
grasp wins cost 3.01 / 7.10 s against 1.09 / 1.10 s, where the old record's tie cost 13x.

**Iterations**: on Panda grasp the joint-space median fell 908 -> 206 with its bound fixed, so the
old "containment costs joint space its cheapness" was largely the +-10 rad box; the learned arm still
uses fewer there under IPOPT (134 / 199) and on soft PCS grasp native. **The augmented
Lagrangian column is `N/A` BY CONSTRUCTION** -- NLopt has no major iteration to count.

**The rescue rate** (cells only the learned arm solved, against joint space's failures) is printed
per row as `L+` beside `JS` in the per-solver detail; the old record's 83-100% is not re-quoted.

**Time-matched joint space, re-measured on stage REMEASURE's own cells** (`scripts/report_time_matched.py
--prefix sc_REMEASURE_ --exclude sc_REMEASURE_LEGACY_ --exclude sc_REMEASURE_RULE_`): joint space gets
best-of-k from each target's 8 recorded starts, random order, stopping at the first success, within
the learned arm's wall time on the same cell. Where the budget outlasts all 8 starts ("starved") the
joint-space figure is a lower bound.

| | learned | time-matched joint space | verdict |
| --- | --- | --- | --- |
| **IPOPT grasp** native / paired: Panda, iiwa, soft PCS | 99.4-100% | 60-87% (starved <= 34%) | **survives** |
| IPOPT grasp, screw | 99.0 / 98.3% | >= 90.4 / 95.1% (starved 43-61%) | unestablished |
| **IPOPT pose native**: soft PCS / Panda | 99.4 / 95.4% | 56.8 / 58.5% | **survives** |
| IPOPT pose native: iiwa / screw | 96.2 / 95.6% | >= 77.5 / 82.4% (starved 37-43%) | unestablished |
| IPOPT pose paired | soft PCS 91.0%, Panda 82.3%, iiwa / screw 84.6 / 89.4% | 82.2%, 84.9%, 96.2 / 97.1% | soft PCS survives, Panda level, **joint space wins** iiwa and screw |
| SNOPT grasp native | 90.8-99.6% (screw 83.3) | 69.1-70.7% (screw 83.5) | **survives** on three, screw level |
| SNOPT pose native: Panda, iiwa, soft PCS / screw | 91.5-99.8 / 87.3% | >= 77.6-91.2 (starved 35-48%) / 92.4% | unestablished, screw to joint space |
| SNOPT grasp paired, SNOPT pose paired | 48.5-81.9% | 78.2-96.1% | **joint space wins** every row |
| **NLopt native**, all eight rows | 62.3-75.0% | 0-11.3% (starved <= 1%) | **survives** |
| NLopt pose paired | soft PCS / iiwa 67.7 / 23.3%, screw / Panda 19.4 / 27.1% | 1.9 / 5.4%, 12.8 / 20.6% | survives soft PCS and iiwa; learned ahead on screw and Panda, single-start ties |
| NLopt grasp paired | 0.6-2.3% | iiwa / Panda 0%, screw / soft PCS 9.9 / 7.7% | floor on iiwa and Panda, **joint space wins** screw and soft PCS |
| *GVS, outside the record* | IPOPT grasp 100 / 97.9%, pose native 100% | 55.2 / 80.8%, 29.8% | survives; pose paired (79.8 v 83.4%) and every SNOPT paired row go to joint space |

**This reverses the old caveat on interior-point grasp**: on the old record only Panda contained grasp
survived on the rigid arms; with the scene fixed and graphs cutting the per-iteration premium it
survives on three of the four record robots (all but screw), and on native pose on two. The paired-start pose rows still go to
a restarting joint space. Wall clock here is PROCS=8 under MPS, which charges the learned arm
1.12-1.19x over one solve per GPU and joint space little, so paper conditions should be no less kind
to the learned arm. **State single-start wins as single-start, and rest a time-matched claim on the
"survives" rows above.** Under NLopt a joint-space solve itself runs to the clock, so restarts buy it
almost nothing (starved <= 3% on every row).

**The augmented Lagrangian, read per protocol and never pooled** (it is extraordinarily
start-sensitive on the learned arm).
- **Native: learned wins all eight rows** -- grasp 313-360 of 480 against 0-48 (iiwa 313 v 0, Panda
  322 v 0, screw 323 v 48, soft PCS 360 v 37), pose 299-358 against 41-141.
- **Paired: the learned arm's grasp collapses to 3-11 of 480**: iiwa 11 v 0 a learned win at the
  floor, Panda 3 v 0 a tie, and **screw 4 v 48 and soft PCS 4 v 37 the column's two joint-space
  wins**. Pose paired: iiwa 112 v 41 and soft PCS 325 v 51 learned wins, Panda 130 v 141 and screw
  93 v 84 ties. Learned pose falls 299-304 -> 93-130 on the rigid arms; soft PCS barely moves.
- **Joint space solves nothing on iiwa and Panda grasp** (all 480 cells time out) -- a property of
  NLopt on this program, joint space being the easier problem.
- **What moved.** The old record's wsg grasp rows were 0-3 of 480 on BOTH arms; native they are now
  learned wins (and joint space 0 -> 37-48 on screw and soft PCS), so that floor was the mug-handle
  scene or the settings -- under IPOPT the settings were null on every wsg row, but no NLopt control
  separates them. Panda pose joint space 12 -> 141 (its scene did not change, so most plausibly the settings) and
  screw pose 30 -> 84 make both pose paired rows ties.
- **Cost** (shared cells, 12-101 per row) is dearer for the learned arm on every comparable row but
  soft PCS pose paired. **Runtime**: learned means 50-79 s native against joint space's 143-180 s.

**Cap check on the record: no row is budget-bound.** IPOPT has 0 cells at an iteration budget on
every row; SNOPT at most 10; NLopt 0 at `hit_eval_cap` and `hit_iteration_cap` on all 16 rows, both
arms. Learned timeouts are at most 18 under IPOPT and 17 under SNOPT (both soft PCS), joint space's at
most 8 and 34. **NLopt is clock-bound everywhere** -- learned 123-194 timeouts native and 161-477
paired, joint space 356-480 -- which is the fielded clock, never lifted, so these are results at it.
SOFTCAP's "quadrupling the clock moves nothing" was measured on the defective scene and is void.

**Acceptance checks all pass.** Every run on one robot x task scene carries the same
`scene_fingerprint`, NLopt's included; joint space is bit-identical between protocols on 42 of 42
protocol pairs across every sub-stage and the GVS arm (cap-bound cells excluded, which leaves the
eight NLopt pairs only 0-124 cells each); and the Panda, whose scene did not change, **reproduces the old record
under the legacy settings** with every discordant cell cap-bound.

### The honest caveats, and what is closed

The one caveat everywhere is **per-iteration cost**, an implementation property; the record now
carries CUDA graphs (see Profiling) and it is still a number to report, not a constant.

Thomas's roadmap, with status: **(1) iiwa checkpoint training DONE**; **(2) SNOPT and NLOPT DONE**;
**(3) performance tuning and formulation tweaks** -- the live item. Naming formulation work as a work
item is **not** a standing licence to invent formulations.

**Four things are closed and must not be reopened.** The **solver axis**, on all three method classes,
with exactly two settings fielded. **Placement**: shelf-contained at the fingertips for both tasks,
with `free` surviving only as the reproducer for archived columns -- a retired SETTING, never a row
again. **The status quo itself** (now stage REMEASURE). And **multi-start**, which stage STEP's
instability finding appears to invite and which Thomas ruled out: *"Let's not go into multi-starting
yet, treat it as somewhat out-of-scope for now, and possibly for this project altogether."* Do not
build it, do not propose it as the next step, and do not lead a report with it.

## Settled negative results on the knobs — do not re-sweep

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

The iiwa's upstream checkpoint was the project's bottleneck, so a follow-on trained deliberately
**simpler, less accurate** charts: `nb_nodes` 12/8/6/4 lowers the architectural gain ceiling
`exp(2.4975*nb_nodes)` from 1e13 to 2e4, with a width-only rung as the control separating "less
accurate" from "less headroom". Nine runs, both robots, 620k steps each; `cluster/ladder_runs.txt` is
the spec and `docs/closed-axes.md` holds the per-rung tables.

**Reducing depth from 12 helps on both robots, and the optimum rung differs by robot** — `n4` on the
iiwa, `n6` on the Panda. "The smallest chart wins" was an iiwa-only result.

**The two mechanisms differ by robot.** On the iiwa a smaller chart buys cells by eliminating runaway
configurations; on the Panda, which has no runaway at all, it buys them by fitting more iterations
inside the fixed cap. Per-iteration cost falls monotonically with depth (iiwa grasp native 80 / 55 / 42
/ 33 ms for n12 / n8 / n6 / n4, the width rung holding depth at 72) and timeouts collapse with it. The
learned arm's per-iteration penalty is now **~10-13x, down from ~30x**. That second mechanism is an
implementation-and-hardware property, which is why ms/it must be reported beside success. A net-faster
chart that needs more steps is a **trade-off to report, not a confound to remove** (Thomas: *"More
iterations but faster net is a trade-off, not automatically good or bad"*) — do not redesign to a fixed
iteration cap.

**The second mechanism was MEASURED AT A 45 s CAP, so it should shrink at 180 s; the first should not.**
**The ladder is deliberately not re-measured and the question is CLOSED rather than deferred**: nothing
touching the charts changed, the rungs are selected by the gain ceiling rather than by cells so a new
grid cannot revise the choice, and the one condition under which re-measuring would have been worth
~350 core-hours — heavy timeouts at 180 s — is answered: timeouts are at most 2 cells of 480. Quote the
ladder tables as the 45 s / free-grasp record.

**Two controls land as intended.** Panda `upstream` and `n12` agree within noise on all rows, so our
training recipe reproduces Jeremy's and no reduced-Panda result is confounded with recipe. And
`n12_w256`, the accuracy-only control, never beats `n12` significantly while keeping the slow iteration:
width costs accuracy without buying either headroom or speed. Depth buys the speed.

**Chart accuracy is a clean monotone dose curve and it runs BACKWARDS to cells.** Median FK error over
5000 poses, 4/6/8/12 blocks: 20.0 / 12.1 / 11.3 / 10.1 mm on the iiwa, 14.7 / 9.5 / 7.2 / 6.1 mm on the
Panda. The *least* accurate depth rung solves the most on the iiwa. 20 mm chart error does not appear in
the solutions — the IK constraint is on `FK(q)`, so `n4`'s solved cells return `median_max_violation`
1.1e-08; the chart only decides where the solver starts.

**Neither intrinsic screen predicts cells, in either direction.** iiwa `n8` screens cleanest of the four
and is the worst rung on pose paired; `n4` screens dirtier and solves best. A chart can be clean
everywhere the sampler looks and catastrophic everywhere the Newton step goes. **The screen is a smoke
test, not a selection criterion.**

**Pole mass is CREATED BY TRAINING, monotonically, on every rung.** Task-pose `pole/max` over
20k..620k steps runs 27 -> 2.5e3 (iiwa `n4`), 409 -> 1.3e5 (iiwa `n6`), 50 -> 3.5e11 (Panda `n12`). The
headroom is present at initialisation and barely used; SGD walks the network into it while buying
accuracy, which is itself converged by ~480k.

**Stage TRAJ measured the training-step axis directly: training makes the iiwa `n6` chart WORSE and does
nothing at all for `n4`.** `n6` **loses three of four rows to its own first checkpoint** (exact McNemar,
p = 8.3e-27 to 3.1e-11); `n4` moves on **none** of the four. `n4` at 20k has ~78 mm median FK error
against 620k's 20 mm — a 4x accuracy gain over 600k steps, worth **zero cells** — while `n6` buys the
same accuracy and *pays* 137 grasp cells. **The sharpest statement the campaign has that accuracy is not
the quantity that matters.** It is **not** a licence to pick early checkpoints: that is selecting on the
test set and confounds architecture with selection.

**A path bug cost a rung its first measurement (fixed in `6fbff55`).** `train_flow.sh` reassigns
`HOME="$ROOT/home"` so ikflow resolves `DATASET_DIR` at import; `export_and_screen_job.sh` derived its
own `ROOT` from `$HOME`, so the inline export resolved every path one level deep and died. The rung
trained all 620k steps and exported nothing, and four interleaved benchmark jobs fired at a checkpoint
that did not exist — **failing per cell rather than fast**. `submit_bench.sh` now refuses when a
manifest names a checkpoint absent from the cluster (`dd24cc3`).

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

Thomas's ranking: **(1) a better chart is preferred over everything else** — *"All of these actions are
less preferred than just getting a better iiwa chart"*; (2) lifting `q` into a bounded decision
variable, permitted but disliked (*"we're effectively adding a nonlinear equality constraint"*); (3) a
joint-limit penalty, permitted but disliked. **"You can try it, but I don't like it" means measure it
and report it as a stated deviation, not adopt it if the numbers look good.**

All four lines are closed and net-negative or inert: **chart accuracy** (`chart_error_scale`) is **not
the mechanism** — degrading the Panda's chart to the iiwa's accuracy costs it 1-3 cells of 60 where the
iiwa is 23 worse, so the standing chart-accuracy hypothesis does not survive its own experiment;
**IPOPT scaling** is inert across five experiments, so this is not a scaling artefact; a **joint-limit
penalty** is inert at 1/10/100, vindicating Thomas's objection that a penalty on a quantity a
constraint row already governs buys nothing; and **lifting `q`** is net negative (38 better against 116
worse, p = 2.2e-10), delivering 0 runaway cells in all eight experiments while **the runaway does not
stop, it relocates** into `max_violation`. **Jacobian regularization** is a clear negative over ten
variants — 310 cells better against 1,509 worse — and Thomas's ruling closed it: *"I think we can
conclude gradient regularization and the other strategies isn't worth it."* The knobs stay in the tree,
off. Per-remedy numbers: `docs/closed-axes.md`.

**Why damping cannot work, the transferable part.** The flow's Jacobian is the *exact* derivative of an
explicit function. Where the gain approaches its ceiling, a sensitivity of 1e13 is the correct answer,
not an artifact. Damping it breaks the correspondence between the constraint values IPOPT evaluates and
the gradients it is handed, leaving an inconsistent nonlinear program — which is why the most aggressive
damping fails hardest while *increasing* the runaway count. LM damping is sound applied to the **Newton
step** rather than to a reported derivative, but Drake's IPOPT does not expose the step computation.

**A lesson these stages share: suppressing the runaway does not buy success.** `tikhonov=10` cut runaway
cells 31 -> 7 for exactly 19/19 on success; `lift_q` reached zero runaways while losing 78 cells net.

### Step rejection: measured and refuted (stage STEP)

**The last lever on the solver axis, and it is closed.** Filter tuning leaves the program exactly as
written and changes only which trial points are accepted, so it was never touched by the
gradient-damping refutation. A 16-setting screen at 60 cells reached significance nowhere; the 480-cell
confirmation then **refuted both promoted settings** — IPOPT's `theta1` is significantly *worse*
(+322/-421 over 2,880 cells, p = 0.00032) and `soc0` is a clean null. **The mechanism column reversed
too**, so a plausible mechanism agreeing with a spurious outcome did **not** protect against the false
positive: both were the same noise. **Nothing is adopted** (Thomas's call in advance: report, do not
field); the five knobs stay plumbed and `None`.

`stage_STEP` in `cluster/gen_manifest.py` owns them and reuses `STEP_REJECTION_KNOBS` as a
**whitelist**, the mirror of stage SWEEP's blacklist, so the two questions cannot merge from either
side. All five are proven to reach their solver from its own parameter echo
(`tests/test_solver_plumbing.py`) — they never had been. Evidence, the screen table and the refuted
SNOPT step-limit mechanism: `docs/closed-axes.md`, regenerated by `scripts/report_step.py`.

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

## The soft PCS arm: a robot whose configuration is not the plant's position vector

**On branch `soft-manipulator`.** Model, kinematics, scenes, four programs and the ikflow shim have
landed and are tested; all five charts are trained; stages SOFT12, SOFTCHART, SOFTDOF and SOFTCAP are
measured. Nothing remains open: the learned forward model was closed as not worth it (2026-09-30,
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

**The primary chart is `soft12__n6__step620000`.** Median tip error over 5000 held-out poses runs
9.40 / 4.77 / 3.62 / 3.13 / 2.90 / 2.74 mm at 20k..620k steps -- better than either rigid arm's
adopted rung, which is worth stating but not celebrating, since the record's own finding is that
chart accuracy runs BACKWARDS to cells. `frac_gt_threshold` is 0.0 at every checkpoint on the
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

**The chart ladder is FLAT** -- learned wins 10 of 12, ties 2 with the iteration budget lifted (stage
ITCAP turned `n4` grasp paired from a win into a tie), no rung separates from another (spread
3-10 cells of 480 across n4/n6/n8). `n8` sits above the ~1e7 runaway band and does NOT degrade, where
an above-ceiling rung is strictly worse on the iiwa. That agrees with the in-distribution screen
reading 0.0 everywhere: **this robot has no runaway population, so the gain-ceiling criterion is
satisfied vacuously and the ladder has nothing to measure.** A negative result about the robot, not a
refutation of the selection rule.

**The DOF ladder measures the BASELINE, not the formulation** -- learned wins 9, ties 2, loses 1 with
the iteration budget lifted (stage ITCAP; 11 / 0 / 1 as first measured). On
pose native the learned arm is at the ceiling (478/477/480 across 9/12/16 DOF) while joint space
climbs **280 -> 334 -> 441**: extra redundancy is worth 161 cells to the arm that has headroom, and
the learned arm has none left to show it in. **Lifting the budget shows the same on grasp**: `soft16`
joint space gains 27-28 cells and both its grasp rows become ties (476 v 476, 476 v 477). The one loss,
`soft16` pose paired, is now ESTABLISHED (407 v 441, p = 7.7e-04, 0 cells at the budget).

**Stage SOFTCAP closes the cap rule on the NLopt grasp rows: the floor is REAL.** Native runs
90/180/**360** s and is 0/480 on both arms at every cap -- quadrupling the wall clock moves exactly
zero cells; paired runs 90/180 s at 2/480 against 0/480. `hit_eval_cap` is 0 throughout, confirming
that `nlopt_max_eval` defaults to 0 and `max_time` is what binds. So these rows are a property of the
augmented Lagrangian on this program, not of the clock -- and they still carry **no verdict**, both
arms being at the floor. **Void with the scene**: on the fixed one soft PCS NLopt grasp native is
360 v 37, paired 4 v 37 (stage REMEASURE_NLOPT). The 180 s rung doubles as a same-configuration reproducibility control and
reproduces SOFT12 **cell for cell**. The 360 s paired rung was **retired unmeasured** to give nodes
back to a sibling campaign; it is absent, not null.

**`soft12_n6` was measured twice, in two separately generated and submitted stages, and reproduces
EXACTLY on all four rows** (474/468/477/436, identical `a_only`, `b_only`, p). Only ms/it moves
(77.7 -> 86.2), which is node contention.

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
  time-matched, restart-enabled joint space fit several tries into one learned solve. The
  record's own cells (2026-09-30) show that is exactly where the learned arm's success advantage
  fails to survive. Under IPOPT, giving joint space best-of-k from each target's 8 recorded starts
  within the learned cell's own wall time, only Panda contained grasp survives on the rigid arms
  (learned 98-99% against 54-68%). The soft PCS arm, where FK already costs a joint-space solve
  3-4x more, survives on 3 of 4 rows (grasp 98-99% against 69-72%; pose native 99.4% against
  74.8%; pose paired a tie). An expensive forward model is what keeps the comparison fair to the
  learned arm, so it is kept.
- **It buys no realism on the PCS arm**, whose exact map is already a cheap closed-form torch
  function.
- **The deployed-hardware argument** (only learned FK exists there) applies to both arms equally
  and is not this paper's contribution.

The hook stays in the tree, unused, as refuted remedies' knobs do: `fk="learned"`,
`src/soft_arm/fk_surrogate.py`, `scripts/soft_arm/train_fk_surrogate.py` and
`cluster/fk_surrogate_job.sh`. `verify()` still re-measures every solution on exact kinematics
through `VerificationQ`. The one laptop fit, 4k steps, reached 11 mm median against a 1 mm gate,
and its weights were deleted.

## The GVS push-rod arm: a robot with no closed-form forward model at all

**MERGED TO MAIN 2026-10-05 from branch `gvs-actuated-arm` (closed), outside the campaign of record;
its stage REMEASURE rows (2026-10-09) are reported beside the record.** Built and tested; both datasets built; go/no-go pre-check done 2026-10-02/03; charts `o1_n6` and `o2_n6` trained and proven 2026-10-04; **stage GVS measured 2026-10-05** (results below). Numbers:
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
body poses), no gravity (a spec field, off, stated). LOInK's soft experiment is the reference
(arXiv 2609.21275 sec. VII: SoRoMoX's planar HSA, 3 segments, 2 actuators each, simulated to
equilibrium, IKFlow on the same data); ours is spatial, 9 inputs, with an optimization baseline they
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
draws (condition number median 46, p95 142), and that the resulting 3-D null space can be
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
shrink 1.43x per tripling against 1.44x for a 3-D set) and covers a LARGE fraction of SO(3):
measured over 8M draws at eight tip positions, **37-93% of SO(3) within 30 deg of a reached
orientation and 54-99% within 45 deg** (medians 71% and 87%), against a uniform control that
saturates at 100% by 20 deg, and every figure still rising with sample size. The backbone
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
slice before JAX loads. **The dataset build is process-parallel and memory-bound**: one worker's
vmapped solve peaks at ~2 GB + 0.7 MB per lane, so the batch is 512 and the worker count is
capped by the node's memory (48 x 4096 was OOM-killed on 192 GB); the sampler solves each draw
ONCE for both the tip pose and the collision screen. **The fielded build measured 436.8
us/sample over a 48-worker xeon-p8 node, 3 h 6 min for 25M, `rejected_unconverged` 0**; order 2
costs 1.83x that per sample (27 generalized coordinates against 18), and a build's per-worker
timeout must exceed `share x ms_each`, since the first result waits for a worker's whole share.

**The joint-space arm's failures are force saturation, not a wiring fault.** From the target, from
straight and from random starts it converges to 1e-8; when it fails IPOPT reports local infeasibility
with rod forces on the +-1 box. A property of the problem to report, like every other baseline's.

**What stage GVS measured (2026-10-05; 16 logical runs, IPOPT and SNOPT, PROCS=2).** Its grasp
rows ran on the defective wsg scene and are void; its stage REMEASURE rows, reported beside the
record in `docs/status-quo-tables.md`, replace them (IPOPT 3 learned / 1 tie, SNOPT 2 / 2: grasp
native 480 v 467 and 476 v 366 are now learned wins, grasp paired 470 v 467 and 343 v 366 ties).
- **Success: learned wins 7, ties 9, loses 0.** IPOPT gives 5 wins and 3 ties, SNOPT 2 and 6.
  - Pose native is 479-480 of 480 against 252-315 on both solvers.
  - IPOPT grasp leaves joint space only 25-33 failures, so three of its four rows are ties.
  - **No SQP grasp loss**, unlike the other wsg robots' paired SQP grasp rows.
- **The per-evaluation premium prediction holds**: 1.31-1.48x under IPOPT and 1.15-1.24x under
  SNOPT, against a predicted 1.3-1.5x.
- **The time-matched prediction holds by sign on 11 of 16 rows and FAILS on every pose paired row**
  (IPOPT 80% against 87-88%, SNOPT 57% against 94-97%).
  - Holds decisively: IPOPT grasp (95-97% against 71-75%) and pose native.
  - Level: the SNOPT grasp rows (-1.8 to +5.9 points).
  - **The exact forward model did NOT make restarts expensive enough on pose paired**: from the
    paired start the learned arm needs 3x (IPOPT) to 9x (SNOPT) its native iterations, while a
    joint-space pose solve takes under a second. That is the soft PCS arm's one tie, reproduced.
- **Cap check:** `hit_iteration_cap` is at most 11 of 480 everywhere, so no row is
  iteration-budget-bound. The SNOPT rows carry 51-130 wall-clock timeouts, and those verdicts are
  results at the fielded clock.
- **Joint space reproduces cell for cell** across protocols and against GVSJS on every converged
  cell.
- **o1 and o2 do not separate on any row** (target-level bootstrap). Only the cost per evaluation
  differs.

**What is queued and what is not.** Stage `GVS` (`cluster/gen_manifest.py`: two trained rungs x two
experiments x two protocols x **IPOPT and SNOPT, no NLopt** -- Thomas, 2026-10-02: *"it's a waste of
time"*) is status-quo-shaped. Its go/no-go pre-check (`GVSJS` joint-space cells, `GVSPREM` premium,
calibration) runs before any training. **Its two columns -- time-matched joint space and iterations
on mutual successes -- and their predictions are PRE-REGISTERED in `docs/gvs-arm.md`** (written
before any trained chart existed); `scripts/report_gvs.py` reads both stages.
Datasets for `o1` and `o2` are BUILT (25M + 15k each, `rejected_unconverged` 0 on every draw),
downloaded and verified, 2026-09-30; `o0`
is spec-only. **Learned FK is closed for this robot too** (the soft PCS arm's section says why): its
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
`~/learned-ik`. Its own tree `~/learned-ik-screw` was folded in and deleted on 2026-10-07, after the
merge check passed (`cluster/SCREW_ARM_RUNBOOK.md`). `git log --follow` crosses the rename.

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

**Measured on the laptop, before anything was queued.** Torch FK against Drake 4.4e-16 position and
1.3e-15 rotation, every link frame under 1e-12; all twelve AutoDiffXd constraint gradient blocks
within 5.2e-10 of central differences; `X_ee_flow` exactly the identity; paired start
`|q(start) - q_init| = 0.0` on all four rungs; 54.1-54.7% of uniform draws collision-free;
acceptance at inset 0.10 of **0.31-0.39% grasp and 0.33-0.34% pose**, inside the rigid arms' band
(0.55-0.68% and 0.23-0.37%), at 465-588 draws per target, with `P(trip)` zero at the fielded
`MAX_CONSECUTIVE_REJECTIONS = 50000`. Acceptance falls monotonically with pitch on the grasp row,
which is the stroke carrying more of the configuration box out of the shelves. The dataset builder,
a 200-step training smoke and the export round trip all run clean.

**This robot's pole screens are IN DISTRIBUTION, and that is now measured rather than argued.** The
vendored fork's in-training callback draws its conditioning pose as a position and an orientation
**independently**, which is a fair draw only where the arm reaches most of SO(3) at a given position.
The soft arm does not, and its callback consequently read `pole/max` 2.4e8 against an
in-distribution screen's 5.44 — eight orders, and a statement about unreachable poses rather than
about the chart. The structural expectation here (7 DoF, roll-pitch-roll wrist) is that orientation
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

*As an internal element.* The **Newport Picomotor** is a genuine lone screw pair: a precision
80-threads-per-inch screw clamped in a split nut and advanced by piezo stick-slip, so the screw —
and with it the ball tip — rotates as it translates, rigidly coupled at 317.5 µm per revolution.
Newport's own closed-loop arithmetic confirms the coupling (6000 encoder counts per revolution at
52.9 nm each). And it shows exactly why a lone H pair is hard to use as a joint: the drive torque
and the load's reaction torque pass through the same thread, so Newport publishes a **torsional load
limit of 0.018 N·m** above which the actuator stalls, and specifies pushing against a smooth flat
pad rather than bolting a load to the tip.

*As a constrained operating mode.* A **ball screw/spline** — THK's BNS-type "Precision Ball
Screw/Spline", NB's SPBR, PMI's PBSA — puts a ball-screw groove and a ball-spline groove on one
shaft with two independently rotatable nuts. THK names three modes: *"rotational, linear, and
**spiral**"*. **Spiral mode is the screw pair**: drive the spline nut with the screw nut held and
the shaft advances at the screw's lead per turn. Drive the screw nut with the spline nut held and
you get translation; drive both together and the screw nut's rotation cancels the translation,
giving pure rotation.

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
*rotary-output* actuator, which is the opposite. Note also that only the BNS/SPBR/PBSA-type models
have both nuts rotatable; THK's NS type and NB's SPBF have a fixed spline nut and are linear-only.

### What the three stages measured (2026-10-01/02)

Stages `SCREW` (status-quo-shaped: 2 experiments x 2 protocols x 3 solvers), `SCREWCHART` (`nb_nodes`
4/6/8 on `p050`, IPOPT) and `SCREWPITCH` (the three trained pitches at `n6`, IPOPT): 36 logical runs
of 480 cells at 180 s, seed 1, `--compile`, contained placement, Drake nightly `0.0.20260918`,
352 items with zero worker failures. **Separate stages, never entries in `ADOPTED_RUNGS`.** Tables,
with the cap and runaway columns inline: `python scripts/report_screw.py`; operations and screens:
`cluster/SCREW_ARM_RUNBOOK.md`. **The pitch rungs do not pair** (each draws its own grid), so
McNemar stays within a pitch, between the arms.

**Stage SCREW reproduces the record's pattern on a third robot class: learned wins 5, ties 5,
loses 2.** Interior point 2/2/0, augmented Lagrangian 2/2/0, SQP 1/1/2, and **both losses are
contained grasp under SQP** (161 and 211 against 298). On the fixed scene the native one is a
learned win (400 v 346) and the paired one stands (259 v 346): **the SQP contained-grasp losses
survive only from the paired start**, on the iiwa and the soft PCS arm too. The learned arm takes every pose row except SQP paired (a clean tie): IPOPT 457 and 421
against 313, NLopt **303 against 30**. It rescues 84-93% of joint space's IPOPT failures. Cost splits
by task again, ~2.2x dearer on grasp (7.8 against 3.5) and cheaper on pose; the per-iteration
premium is ~14x (48 against 3.3 ms). NLopt grasp was 0-3 of 480 on both arms (defective scene; fixed: 323 v 48 native, 4 v 48 paired).

**Stage SCREWCAP re-measured every budget-bound row with the iteration budgets lifted**
(2026-10-02): the 17 rows where either arm had >= 24 of 480 cells at a budget, each regenerated from
its own builder -- same seed, grid, chart and **180 s clock** -- with `max_iter` 100000 and, for SNOPT,
`snopt_iterations_limit` too (both its INFO 31 and 32 count as the iteration cap). The clock stays
because Thomas ruled it the usability limit: *"Don't raise the wall clock timeout. If it's that slow,
it's not usable."* So the remaining failures are 180 s stops, and each verdict below is a result at the
fielded clock rather than budget-bound. Iteration-capped cells are now 0 on every IPOPT row; SNOPT keeps
10 per row that reached 100000 majors on a few thousand minors -- cycling, not under-budgeted, and under
the 24-cell threshold. Cell-for-cell table against the originals: `python scripts/report_screw.py
SCREWCAP`.

**The IPOPT grasp ties were ESTABLISHED ties** (427 v 424, 429 v 424; the learned arm gains 16 and
10 cells, losing none) -- on the defective scene; on the fixed one both are learned wins (475 and 472
v 442). The SQP grasp losses stand unchanged (163 and 212 against 300), so stage SCREW's
tally is unchanged at 5 / 5 / 2.

**The chart ladder: `n4` is still the WORST rung, but by less** -- the iiwa's best. Its two grasp
losses were the iteration cap, not the chart: lifted, `n4` gains 43 and 50 cells (losing 1) and ties
(410 and 415 against 424). It wins pose native and still loses pose paired decisively (200 against
313, p = 3.6e-15, at 0 capped cells and 246 timeouts). `n6` and `n8` agree (both pose rows won, both
grasp rows tied). `n4`'s failures are not the gain-ceiling runaway: on pose paired they end at a
median `|q|_inf` of 30 rad against ±3 rad limits, none above 1000 -- moderate out-of-limits excursions,
what its training-time validation ratio (up to 14.7, unclamped over clamped) had flagged. The
pre-registered `n6` stands. `p050_n6` reproduces across all three stages (411/412/411 and
419/420/420 on grasp, exactly on pose).

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
a property of the trained chart rather than of the robot class. Joint space climbs with pitch on
grasp (420 / 424 / 434 lifted) and is flat on pose; it too gains 2-14 cells when the cap lifts.

**Build datasets one at a time** — ikflow's end-of-run summary scans the shared cache directory, so
a concurrent sibling's half-written tensor makes a finished job exit 1 with its data correct on
disk and its `.DONE` sentinel missing, which `train_flow.sh` hard-fails without.

**The curvilinear rail is deferred, not dropped.** `<drake:joint type="curvilinear">` parses in the
pinned build with finite limits. Two things to know on return: on each piece the map is algebraic (a
circular arc is a revolute joint about the arc centre, relabelled), so the non-algebraicity is
global rather than pointwise; and Drake's trajectory is planar and piecewise line-and-arc only,
which matches every real curved track and admits no clothoid or spline.

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
quantity. So is **GPU** contention, which only the learned arm pays. Paper runs use one solve per
GPU (PROCS=2); development stages may use PROCS=8 with `MPS=1` (Profiling).

**`--shard K/N` is a no-op by construction.** It splits **target-major** — whole targets per shard,
never a target's guesses split — because `success_ci` bootstraps over whole targets and
`solved_within_k` counts restarts within one, and it appends `_shardKofN` to the tag (without which
two shards overwrite each other's `summary.json`). `cluster/merge_shard_summaries.py` pools the
records and **re-runs `summarise`** rather than stitching per-shard numbers, preserving arm order so
`_mcnemar`'s pair directions survive. `bash cluster/verify_sharding.sh` proves the round trip in ~2
minutes — **run it after any change to sharding, the merger, or grid construction.**

**Benchmarks run on the GPU partition, and that is settled** (Thomas, 2026-09-25: *"benchmarks
have to use GPU. We've tried this before."*). So benchmark jobs and chart training compete for the
same 4-node cap, and the lever when the cluster is saturated is **ordering the queued work**, never
relocating benchmarks to `xeon-p8`. Datagen is the exception and does not generalise: it is pure
CPU and `build_dataset_job.sh` runs on `xeon-p8`.

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

The measured calibration — workers per node, GPU-vs-CPU, the cap ladder and process startup — is in
`cluster/README.md`; it is a property of that hardware, not of the project.

Three operational post-mortems live in `cluster/README.md` in more detail than a project record
needs, and the transferable lesson of each is stated there. **A shard set straddling any number of
collections merges normally** — `collect_results.sh` passes `--also` for every prior staging
directory — so collect when you want results and act only on an actual `INCOMPLETE`; the old advice
here to "collect less often than an item takes" was stale, and a stale MITIGATION costs more than a
stale diagnosis because it gets acted on. Per-cell solver logs go to node-local `$TMPDIR` and roll
into one archive, because 35,596 small files on Lustre took a routine collection from three minutes
to thirty. And **any cluster-wide check on a shared account must be scoped to this project's own
jobs, by JOB name** — the fix that filtered the payload script's filename instead meant `--reclaim`'s
guard was unconditionally 0 and never refused for its whole life, so **a guard nobody has observed
refusing has not been tested.**

**The laptop does NOT suspend on AC**, and no run here needs a sleep inhibitor. Thomas, 2026-09-28,
having checked the power settings: *"I just went into my settings and confirmed that my computer
doesn't sleep on AC. So no need to worry about manually preventing it from sleeping."* A multi-hour
gap in a session is him closing it deliberately or hitting a usage cap -- neither is a fault, and
neither is diagnosed or mitigated. This file previously asserted the opposite and told a reader to
hold `systemd-inhibit --what=sleep:idle`; that guarded nothing. Long benchmarks run on the cluster
anyway, where a laptop's state is irrelevant.

**Local compute must never take the laptop down (Thomas, 2026-10-09: *"If a pytest sweep crashes my
machine, that is unacceptable"*).** On 2026-10-09 at 16:02 the svgd solver's Drake collision PROCESS
pools -- one per cached BatchedProgram in a test file plus 20-worker pools in benchmark subprocesses,
run concurrently by a subagent -- reached 81 worker processes and 44.8 GB on a 62 GB machine with no
swap, and the kernel OOM killer took his Slack. **The pool is deleted outright** (Thomas, the same
evening: the only parallelism is the GPU over particles and, behind it, Drake's own C++ threads). The
collision row is evaluated IN PROCESS on the program's own `MinimumDistanceLowerBoundConstraint`
(`src/svgd/collision_backend.py`, `CollisionEvaluator`), serially, ~0.3 ms per particle -- 92-96% of a
step at N = 64-256 -- until Drake gains a batched, GIL-releasing
`CollisionChecker::CalcRobotClearances` (brief: `~/Downloads/drake-CalcRobotClearances-plan.md`;
locally pydrake comes from his own build at `~/opt/rlg/drake-build`, so it lands here before any
nightly). Rules that survive, in practice and in briefs: no process pools in this project; every local
test file, smoke or end-to-end run that can build a program runs ONE AT A TIME, in the foreground,
under `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0` (the 8 GB of swap is his
emergency buffer and is never counted or used), so a runaway kills the run and not his applications;
and loadavg is logged before and after every timed run, any run at load above 24 re-run.

Two details of that paragraph survive it, being about process handling rather than power: detach a
long local process with **`setsid`, not `nohup`** -- `nohup` only ignores SIGHUP, so a teardown group
kill takes the process *and* anything it was guarding -- and `pgrep -f <script>` run from a Bash tool
call often matches **the calling shell's own command line**, so a dead process reads as alive.

**Reproducibility at the cap scales with the cap-bound population, not the grid.** On 60-cell grids
two runs of the same configuration scored 34/60 and 35/60, the differing cell hitting the wall clock
in both; at 480 cells with 52-88 timeouts a same-configuration re-run moves up to **7 net and 19
discordant** cells, and rows with zero timeouts have zero discordance. So the solve path is exactly
reproducible on cells that converge — see stage STEP's measurement of this. Worth remembering before
reading a small difference as a real effect.

**Plan around SuperCloud's monthly maintenance** — second Tuesday, compute down Monday evening to
Wednesday morning, nothing survives it. Next window: 2026-10-12 to 10-14.

## Open items

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
