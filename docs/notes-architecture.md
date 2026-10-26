# Architecture notes

The long form of a part of `CLAUDE.md`, moved here verbatim on 2026-10-18 to bring `CLAUDE.md` under
40k characters. `CLAUDE.md` keeps the verdicts and rulings; this file keeps the mechanisms, numbers and
history behind them. "Above"/"below" and section names refer to the old single-file layout, now split
across `docs/notes-*.md`. A reference elsewhere to "CLAUDE.md's ... section" means the matching notes file.

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
  - **TRAP: a linear solve inside the network (2026-10-23, the LOInK chart).** The jacrev graph is
    captured under `enable_grad`, and in that mode torch's backward of `linalg_solve` /
    `solve_ex` recomputes its gradient through a plain `at::linalg_solve`, whose info check
    (`_linalg_check_errors`) is a host sync: "CUDA error: operation not permitted when stream is
    capturing", at `WarmUpJacobian`, on every item (torch 2.14 on the cluster). The forward
    pass's `check_errors=False` is not enough. Use `lu_factor_ex(check_errors=False)` +
    `lu_solve` (`src/charts/loink.py` `_solve`), whose derivatives are graph-safe both ways, and
    before graphing any new network trace its GRAD-ENABLED compiled graph (a recording
    `aot_autograd` backend) for `_linalg_check_errors`; a CPU run proves nothing about capture.
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
pay). Stage REMEASURE ran PROCS=8 under `MPS=1`, and stage PAPER + SVGD_R2 re-ran its 32 IPOPT and
SNOPT runs at one solve per GPU, which **measures the premium end to end**: mean wall **1.06-1.22x**
learned (median 1.13; soft PCS IPOPT native excepted, see Results) and **1.01-1.09x** joint space,
with success and iterations unchanged (`scripts/report_paper.py`). **Paper numbers run at one solve per GPU (PROCS=2)** (Thomas: *"those are
the conditions in which the final paper results will be drawn"*). 4 per GPU under `MPS=1` is for
development throughput only: its verdicts carry over, its seconds do not. Never run 4 per GPU without
MPS once graphs are on.

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

