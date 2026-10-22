# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Keep this file under 40k characters.** It is read at the start of every session. It holds the
verdicts, rulings and traps; the mechanisms, numbers and history behind them live in
**`docs/notes-*.md`** (architecture, rules-and-benchmarking, solver-axis, svgd, record, chart, robots,
cluster) — **read the matching notes file before changing or re-measuring anything in that area.**
When a measurement supersedes an older one, delete the old table and keep the conclusion; git history
holds the numbers. Evidence for closed questions: `docs/closed-axes.md`.

## Project

Research code (ROS-style package `combining_kinematics`) for solving IK with a **normalizing-flow IK
network (IKFlow) inside a Drake optimization program**, so collision avoidance, joint limits and task
costs are imposed on the network's *output*. The point is a comparison of formulations of the same
problem:

| Formulation | Decision variables | `VarsToQ` |
| --- | --- | --- |
| **learned** (`Panda/IiwaIKProgram`, `...MugProgram`) | conditioning pose `c` (xyz+rpy, 6), latent `z` (`network_width`), `correction` (7) | IKFlow forward pass + `correction` |
| **numerical** (`...ProgramNumerical`) | joint angles `q` (7) | identity |
| **analytic** (`Panda/Iiwa14IKProgramAnalytic`, `...MugProgramAnalytic`) | task-frame pose (6) + redundancy `psi` (1) | closed form (`src/*_analytic_ik.py`), branch fixed per solve |

**The analytic arm** (both rigid arms, merged 2026-10-14; Panda He-Liu `psi = q7` on 4 or 8 branches,
iiwa Faria S-R-S on 8; shared code `src/analytic_program.py`) is **a reference target, possibly not in
the paper** (Thomas, 2026-10-13); the paper's case is robots with no closed form, so **learned v joint
space is the comparison of record**. Measured (stages ANALYTIC*, 2026-10-14) it is **not an upper
bound**: single start it loses success to learned on every IPOPT and most SNOPT rows, leads under
NLopt; its failures are targets the cell's fixed branch cannot reach. Its IPOPT/SNOPT/NLopt seconds are
from xeon-p8, not volta. Rulings `memory/analytic-baseline-rulings.md`; ops
`cluster/ANALYTIC_RUNBOOK.md`; tables `docs/analytic-arm-tables.md`.

All formulations share `IKFlowProgram`, so a change to constraints/costs affects all of them.
`workshop-paper-draft.pdf` is an early rough draft — orientation only. The sibling `../codebase/`
(analytic-vs-numerical) is **read-only**; `scripts/iiwa/iiwa_collision.py` imports from it.

## Environment and running

No package manifest. `pydrake` from `~/opt/rlg/drake-build` (on `PYTHONPATH`; IPOPT and SNOPT),
`torch`, `numpy`, `tqdm`, and `ikflow` + `jrl` (Jeremy Morgan's; **not** in the default env — check
`python -c "import ikflow"`). Scripts add the repo root to `sys.path`:

```bash
python scripts/panda/panda_benchmark.py --task mug  --targets 15 --guesses 3 --config latent
python scripts/panda/panda_benchmark.py --task pose --targets 15 --guesses 3 --config latent
python scripts/iiwa/iiwa_benchmark.py   --task mug  --targets 12 --guesses 2 --config latent
python scripts/collate.py 'results/*/benchmark/*/summary.json'
```

Older per-experiment scripts (`panda_mug*.py`, `*_collision.py`, `iiwa_mug.py`, ...) are kept for
archived comparability, configured by their `####### Options #######` block. `tests/` holds invariant
guards, not a suite; no lint config. Output goes to `results/` (gitignored). Visualization is Meshcat;
mug experiments start a second one (`GenerateDiagramWithMug` rebuilds the diagram per target). Panda
weights download via `ikflow` (`panda__full__lp191_5.25m`); iiwa weights are local pickles under
`models/iiwa14/` (gitignored). URDFs use `package://combining_kinematics/...` with vendored meshes.

## Architecture (long form: `docs/notes-architecture.md`)

**`src/generic_program.py`.** `ProgramOptions` is the single dataclass configuring everything.
`IKFlowProgram` owns the diagram, plant, its `ToAutoDiffXd()` copy and contexts. Each
`Create*Constraint` appends an `IKFlowConstraints(lb, ub, eval_func)`; `ApplyConstraints` adds **one**
Drake generic constraint whose `EvalAllConstraints` computes `q = VarsToQ(vars)` and FK **once** and
dispatches `(vars, q, pose)` to every `eval_func`. The network dominates cost, so **never add a
constraint that recomputes `VarsToQ` itself**. `QAndPose` memoises `(q, pose)` per iterate, keyed on
values **and** the AutoDiffXd derivative block (`share_flow_evaluations`, default on, bit-identical),
so costs reuse the constraint's pass. `Solve()` registers a visualization callback (also writes
iterates to `options.vars_file`), keeps `program.last_iterate`, returns the `MathematicalProgramResult`.

**Per-robot subclasses** (`src/panda_program.py`, `src/iiwa_program.py`) implement `__init__`,
`create_prog`, `ik_inference`, `VarsToQ`. `...MugProgram` swaps `self.frame` from the trained
end-effector to `between_fingers`, keeping `X_grasp_ee`. A mug grasp constrains only position in the
mug frame (`x = y = 0` an equality, `z` within `mug_height`), orientation free.

**Checkpoints carry their architecture** (`src/flow_loading.py`). A `.pkl` is a bare state dict, and
`nb_nodes` / `rnvp_clamp` change the forward pass without changing any shape. Each checkpoint has a
mandatory `<name>.arch.json` sidecar (`scripts/training/export_ckpt_to_pkl.py`); `LoadFlowSolver`
cross-checks every field recoverable from shapes and **raises** on contradiction. **`rnvp_clamp` is
the one field no check can catch.** **Hold `dim_latent_space` at each robot's baseline** (iiwa 8,
Panda 7): it is the decision-variable count, so varying it changes the problem, not the chart.

**Gradients.** `VarsToQ` is dual-path: float = plain forward pass; AutoDiffXd = `self.jacobian_gen`
(one reverse pass gives `dq/dvars` and `q`), chain-ruled into fresh AutoDiffXd. Both go through the
free function `MakeFlowInference(nn_model, ...)` so `FlowJacobianGen` memoises
`torch.compile(jacrev(...))` per process (**compiling a bound method re-triggers dynamo per
program**). Analytic formulations use `pydrake.math` trig and Drake's own autodiff.
- **`jacrev`, not JVP — measured, do not re-litigate**: one `J` (7 x 21) serves every row and the
  objective; the Jacobian is 84% of a solve, the flow 96%; forward mode is 2.8x slower.
- **`--compile`** is 1.48x and **`--set flow_cuda_graph=True`** (requires `--compile`; graphs captured
  in `WarmUpJacobian`, a capture inside a timed solve raises) a further 2.3-2.6x on learned ms/it,
  iterations identical. Both move the learned arm's success within a cap, so **every compared run sets
  them identically. The record runs with both on.**
- **Processes per GPU matter**: graphed processes are GPU-bound, so 4/GPU without MPS kills the gain.
  **Paper numbers run at one solve per GPU (PROCS=2)** (Thomas). 4/GPU under `MPS=1` is development
  throughput only (verdicts carry over, seconds do not). **Never run >1/GPU without MPS with graphs on.**
- **float64** (`use_float64=True`): float32 values have a ~1e-7 floor that corrupts every difference
  (line search, convergence tests, SNOPT derivative checks).
- The IK pose constraint is six rows: per-axis position error, then the **rpy residual** wrapped to
  (-pi, pi]. **Three signed rows are deliberate** (a scalar angle has a branch point at zero error).
  Rows with an identically-zero gradient break LICQ (the mug constraints drop the homogeneous row).
- `c` is boxed near the target (`c_position_slack=0.25`, a heuristic; it sits just under an exposure
  **cliff at 0.5** — do not widen it).
- **No seeding search, deliberately** (removed, not disabled): `SetStartFromQ(q_init)` is the only way
  to set a guess, and every formulation gets the same one.

**The conditioning frame — read before touching the learned formulation.** The flow is conditioned on
the pose of the frame **it was trained on**, which is not the scene's frame of that name (finray
`panda_hand` is 27 mm / 120 deg off jrl's; iiwa `iiwa_link_7` 45 mm short). Symptom: running the flow
forwards on a random `q` returns `|z|` ~67 instead of ~sqrt(dim). `CalibrateFlowFrame` measures the
offset, checks it is constant and caches `X_ee_flow`; **use `FlowPoseInWorld()` wherever a
conditioning pose is formed**. Skipping it collapsed Panda pose to 10/60. **A skipped calibration is
indistinguishable from a correct one until the geometry it relied on changes.** It requires the base
welded at the world origin.

**Ablation ladder (Panda grasp):** frame calibration is worth 18 of 23 cells and is the only
significant rung. **`latent_trust_region`** (+4, n.s.) **stays**: IPOPT is interior point, so a
nonbinding constraint still shapes the trajectory (Thomas). It is a **stated deviation from eq. (6) the
paper must document**; do not argue from binding fractions that it is inert.

**Scenes** (`src/utils.py`). `BuildEnv(meshcat, directives_file, extra_directives=None)` appends
directives **in memory**; `GenerateDiagramWithMug` welds a mug at the gripper pose of `q` that way, so
tracked YAML is never modified. `BuildEnv(meshcat=None)` skips visualization (passing `None` to
`ApplyVisualizationConfig` would start Drake's own). Mug targets come from collision-free `q`, so every
target admits a valid grasp. `HiddenPrints` silences at the fd level. Notebooks run from `notebooks/`.

**The hardened scene and shelf-contained targets** (`models/*/*_hardened.yaml`: legacy minus `binF`
and decorative mugs). Targets must land in one of twelve shelf compartments (`src/shelf_regions.py`,
local boxes rotated into each region's frame — a world AABB over-approximates ~4x; inset symmetric,
`shelf_depth_inset = 0.10`, because `shelves.sdf` has no back wall). **Drake never generates collision
candidates between two ANCHORED geometries**, so `FloatingMugScreen` checks the mug **unwelded** on its
own diagram (`tests/test_shelf_placement_screens.py` pins its filter). Acceptance is 0.2-0.7%, so
**`MAX_CONSECUTIVE_REJECTIONS = 50000`**. The pose task's containment point is recorded as
`placement_point`. Guesses are **deliberately not** containment-filtered. `--scene legacy
--target-placement free` reproduces the pre-hardening sampler (retired setting, never a row).

**TRAP: the mug handle must point through the finger gap.** The wsg finray SDF gave `between_fingers`
yaw 0, putting the handle 18 mm into a finger plate on every q* — invisible to every check
(anchored-vs-anchored). Fixed by the yaw in that SDF (452d784; `scripts/probe_mug_contact.py`). **Every
wsg grasp number before stage REMEASURE is void.** Runs record `metadata["scene_fingerprint"]`;
`collate --pair` refuses a mismatch. **Never pair cells across a scene change.**

**Cross-robot settings, ADOPTED 2026-10-09** (Thomas): one code path each in `generic_program.py` —
native grasp start `c` = flow-frame pose of the generating configuration (`GraspCStart`; the PI accepts
it conditions on a known-valid grasp); joint-space bound `ConfigLimits()` everywhere (the Panda had
+-10 rad: grasp joint space 374 -> 449); `q_nominal` a nonsingular in-limits home
(`NominalConfiguration`); the grasp `c` box centred on that pose (`GraspCBoxConstraint`); per-robot
trust-region radii stand (the `sqrt(dim) + 1.5` rule A/B is inert). `--set legacy_robot_settings=True`
is **a control only**; `tests/test_robot_settings_unified.py` pins both states.

## Rules the campaign established (long form: `docs/notes-rules-and-benchmarking.md`)

Each was learned by getting it wrong, at the cost of whole tables. None is negotiable without Thomas.

**The tolerance ladder.** Thomas: *"IK constraint tol should always be zero. The whole point is that
it's an equality constraint, satisfied exactly. Tolerance should be zero in the mathematical program,
only appearing in solver tolerance."* `lb = -tol, ub = +tol` changes an equality's **kind**, and an
interior-point method parks on the face. Pose rows are `lb = ub = 0`; `tests/test_constraint_bounds.py`
reads the bounds Drake was handed. `ik_constraint_tol` forms no bound. **Rung 3: the gate stays
deliberately looser** — `ik_constraint_tol = 1e-4` (solver), `task_tol = 1e-3` (gate); Thomas: *"go
back to 1e-4 actual tol and 1e-3 task tol, to avoid this issue"*. **Never set an acceptance gate equal
to a bound the solver optimises against**: solutions pinned to the bound make the gate score the last
ulp (it appeared to reverse a row). Check the gated quantity's distribution before tightening any gate.
The collision gate carries the binding's own slack. `feasible_relaxed` is recorded at `task_tol`.

**A region an initial guess may violate is a general constraint, never a variable bound**: IPOPT's
`bound_push` projects the guess into every box before evaluating. The `c` box is
`AddLinearConstraint` (`CBoxConstraint`); the latent's box lives in **one** method,
`LatentBoxConstraint()` (subclass copies once left the grasp arms unfixed). Clipping the inverted latent
had scored exact starts `unrepresentable_start` (iiwa pose 11 -> 40 of 60 when fixed; **every older
paired learned column is void**). Nothing may project a guess without recording it (`clip_distance`).
Variable bounds remain fine for regions a start always respects (the correction's +-0.1). Thomas:
*"we're not assuming feasible initial guesses."*

**No invented formulations, no weakening of the problem.** A "task-parameterised" grasp formulation
once fielded as "learned" was removed outright — Thomas: *"Constructing new formulations and passing
them off as ones I've already written is completely unacceptable."* The learned formulation is eq. (6)
exactly as `PandaMugProgram`/`IiwaMugProgram` implement it. Constraints that *are* the task stay
equalities (a `mug_axis_tol` was removed). Improvements come from the formulation or the solver, never
from making the question easier.

**The correction penalty** `correction_cost_weight = 10` (Thomas, 2026-09-02) is a **stated** part of
the learned formulation; describe it wherever the arm is described. **Options naming learned-only
variables must be guarded** — an unguarded one once zeroed two columns at ~10 ms/cell.
**An arm reporting per-cell wall three orders below the cap is not solving at all**; `_abort_on_dead_arm`
in `src/benchmark.py` aborts on it.

**Reporting: success, iterations, cost and wall clock, always together** (Thomas). Iterations describe
the formulation, seconds this implementation on this machine (**never compared across machines**),
cost what a solution is worth. **Cost only on cells both arms solved**; learned-only regularizers
excluded (`reported_cost`). **Joint space is the comparison's target; analytic columns are baselines**:
lead with learned-vs-joint-space McNemar, never claim a win off analytic alone, state ties as ties.
Baselines get bug fixes and standard treatments, no novel research (Thomas).

**Cap check before reporting any loss: read BOTH `timed_out` AND `hit_iteration_cap`.** `max_iter`
`None` means IPOPT's own 3000, reachable inside the clock. **A row with >= 24 of 480 cells at a budget
carries no verdict until re-measured with the iteration budget lifted — never the clock** (Thomas:
*"Don't raise the wall clock timeout. If it's that slow, it's not usable."*). Record budgets: IPOPT
`max_iter` 1e6, SNOPT 1e5 majors / 1e8 minors. SNOPT does not check its time limit on a major with
zero minors, so a cycling cell can run 0.4-2.2 h (never scored feasible; allow for it in item
timeouts). The cap is a budget for the arm that evaluates a network: baselines are flat across caps.

## Benchmarking (`src/benchmark.py`; long form in the rules notes)

- **Paired grid**: `num_targets x num_guesses`, one solve per cell, no retry, identical cells for every
  arm — exact McNemar for success, bootstrap CI over whole targets. Guesses drawn **per target**.
- **Two start protocols, both first-class** (`--start`): `paired` puts every arm at the same `q_init`
  in its own variables (`SetStartFromQ`; **invert first, then clip**); `native` gives each its own
  initialisation. Joint space's two protocols coincide. Every cell records `clip_distance` (summed over
  bounds *and* general constraints — not a projection everywhere) and `start_q_error`.
- **Paired-start exactness**: joint space 0; learned ~1e-6; analytic 8-branch 1e-11; Panda 4-branch
  loses ~13% of starts (`unrepresentable_start`). A start in the analytic clip band is solved as an
  infeasible start, flagged `start_in_clip_band`. `legacy_paired_start=True` restores old projections.
- **Success is verified from the returned point**, every binding re-evaluated, with a named
  `fail_reason`. `PandaMugProgramAnalytic` inherits from the pose class, so the grasp is measured via
  `between_fingers` by name.
- **Raw per-cell state is persisted** (q, last iterate, vars, per-binding violations, `min_distance`
  and its pair, start, `z_norm`, `correction_binding`, ...). Decide the schema **before** launching.
- **Abnormal exits keep the iterate** (Thomas rejected a watchdog in `QAndPose`): verified as
  `recovered_feasible`/`recovered_cost`; the process is bounded from outside (OS `timeout`).
  `SolveTimeout`, `CheckDeadline`, `hard_time_factor` were **deleted and must not come back**.
- `--compile`, `--set NAME=VALUE` (any `ProgramOptions` field, recorded in metadata and tag), and
  `metadata["grid_hash"]` (task as suffix); `collate.py --pair learned '<glob>'` refuses a grid mismatch.
- **`collision_value` is a penalty, not a clearance** (raw `MinimumDistanceLowerBoundConstraint`;
  successes sit at 0.9997-1.0005). Read `min_distance` (metres) instead.

## The solver axis (long form: `docs/notes-solver-axis.md`; evidence `docs/closed-axes.md`)

**Three METHOD CLASSES**: `--solver ipopt` (interior point), `snopt` (SQP), `nlopt` = **`LD_AUGLAG`**
(augmented Lagrangian; not `_EQ`, which leaves inequalities to the inner solver). Thomas: *"we test
interior point, augmented lagrangian, and SQP."* **A reporting axis, never a choice**; shared across
arms within a run. **The whole axis is CLOSED. Do not re-sweep it.**

- **Each solver at its own defaults**; never transplant one solver's option values onto another.
  IPOPT's `tol`/`constr_viol_tol`/`dual_inf_tol`/`compl_inf_tol` are now plumbed. Per-solver tuning is
  permitted, per-problem is not; a setting qualifies only by winning **uniformly by row, never pooled**.
- **Fielded settings, exactly two** (2026-09-19): SNOPT `snopt_major_step_limit = 0.5` (adopted for
  **fairness**, zero verdict flips — **do not present it as helping the learned arm**; "Drake defaults"
  now means `--set snopt_major_step_limit=None`); NLopt `LD_AUGLAG` + `LD_MMA` inner + inner
  `xtol_rel = ftol_rel = 1e-3` (Thomas: *"Feasibility is the name of the game"*; report with it that
  it failed NLOPTTUNE's gate, regresses Panda pose native, flips one verdict).
- **Result IPOPT > SNOPT >>> NLopt — a confirmation of Thomas's prediction**, a property of the
  problem (joint space degrades too). SNOPT's failures are ~87% convergence (INFO 13, INFO 41 = badly
  scaled exact derivatives), not budget.
- **No solver reports iterations through Drake**: IPOPT's from its print file, SNOPT majors, NLopt
  none. The program counts evaluations in `QAndPose` (`map_jacobian`; not comparable across arms).
  **Status is decoded numerically**, never from log text. NLopt status 5 = `hit_eval_cap`.
- **Step rejection refuted** (stage STEP). Its real finding: **per-cell outcomes are unstable** — one
  filter change recovers ~70% of failures and breaks 6-8% of successes. Report it; it points at
  multi-start, which is **out of scope** (below). At 480 cells a same-config re-run moves up to 7 net /
  19 discordant cells, all cap-bound.
- **Traps**: an option can be accepted and inert (check the solver's own echo; SNOPT is
  case-sensitive on the second word; `Nonderivative linesearch` is valueless, so 0 turns it ON; no
  `mumps`, Drake's IPOPT uses SPRAL). Read defaults out of the solver (IPOPT `acceptable_dual_inf_tol`
  is 1e10). Drake's NLopt `max_eval` defaults to 1000 — set it. **Luksan trap**: `LD_LBFGS`, `LD_VAR*`,
  `LD_TNEWTON*` are listed but refused inside the solve (0.5 s cells, `q=None`); `LD_MMA`, `LD_CCSAQ`,
  `LD_SLSQP`, `LN_COBYLA`, `LN_BOBYQA` work. Unknown NLopt names raise inside `Solve`, hence
  `ProgramOptions.__post_init__` checks (`tests/test_solver_plumbing.py`). `local_optimizer_*` applies
  only when an inner algorithm is named. **Do not extend Drake for instrumentation** (Thomas).
- **`LD_AUGLAG`'s justification is in NLopt's source** (bound-constrained inner subproblem, every
  constraint in the AL penalty) — never the old Drake-side argument. Switching to `_EQ` is a
  method-class decision and Thomas's.
- **Grasp containment is closed** on both axes; the containment verdicts stand as measured under IPOPT.

**ONE DRAKE, AND IT IS THE PIN**: the nightly `drake-0.0.20260918-noble.tar.gz` at `$ROOT/drake`, no
second install (Thomas, twice: *"Run everything on the current nightly"* — a regression is reported
upstream, never pinned around; archive pairing is no reason to keep an old install). A nightly because
**PR 25002** (sixteen NLopt options) is in no release; `CheckNloptOptions` refuses only under
`which_solver == "nlopt"`. **Nightlies expire ~2026-11-02: move to 1.58.0 once it carries PR 25002.**
One stated exception: the svgd rounds' clearance wheel (`DRAKE_ENV=crc`, `~/learned-ik/venv-crc`,
`CalcRobotClearances`), a strict speed improvement, cluster only. The API is in master (#25073,
2026-10-12) and the default local pydrake: **never point local code at a non-default Drake** (Thomas,
2026-10-20); the cluster exception goes when the pin moves.

## AL-SVGD, the fourth method class (long form: `docs/notes-svgd.md`; evidence `docs/svgd-solver.md`)

A batch-parallel particle method (`--solver svgd`) on exactly the existing programs; branch
`svgd-tune` merged (0efb579), **the push is CLOSED (2026-10-18); next is network-architecture
ablations, in another session.** Ops `cluster/SVGD_RUNBOOK.md`; tables `docs/svgd-2x2-tables.md`
(`scripts/report_svgd_2x2.py --write`).

**Rulings**: SVGD-style only; exact Drake collision, no proxy; no formulation-specific information in
the solver (kernel SPACE the one exception; `svgd_stop_select_objective` a second, measured inert); AL
is the formulation and SVGD the optimizer, one dynamics; K a separate parameter (do not re-propose the
one-loop strip-down); "resample" is the term; **the AL with its true dual update stays** (Thomas,
2026-10-16) though the pure penalty ties it; compute footprint is a paper explanation, not a fairness
worry. **The goal is learned-under-svgd against learned-under-IPOPT; joint-space svgd is an ablation.**

**The pin `SVGD_PINNED_V4`**: PHR AL, Tabor-Hermans direction, dual step alpha = rho every K = 10;
rho 1e5; kernel `cz` (median bandwidth, floor 0.05; **a joint-space run must name `svgd_kernel=q`**);
Stein variational Newton, block-diagonal Gauss-Newton, Levenberg delta 10 (fixes the gauge of the
(c, z) redundancy); lr 0.3; N = 64 float32, particle 0 at the guess, the rest from the native draw
(`svgd_paired_init=native`); runaways (`|q|_inf` > 10) resampled; multiplier box 1e4; stop = best
feasible objective stalls 3 checks at rel 1e-3, confirmed by the float64 Drake re-check (**gate-edge
rule**: a failed re-check tightens the level x0.9 and continues); stop/selection read the `full`
objective. Every setting's reason is in the notes. Refuted/inert, not fielded: temperature (removed),
latent prior, flow density, correction weights, drive-only prior (**soft PCS pose finding, ruled
2026-10-18: reported, NOT fielded** — one pin for all robots).

**The record: R29 `pin4` (learned) + R30 `pin4kq` (joint space)**, 16 rows; R30 `pin4s2` replicates on
seed 2. **The seed-to-seed band on a row's median cost is 0.94-1.10** — a ratio inside it is a tie.

**The result, a double contribution** (Thomas, 2026-10-18; metrics success, cost, wall, time-matched;
no iterations). Proposed = learned + AL-SVGD; ablation A = learned under the Drake solvers; ablation B
= joint space under AL-SVGD; baselines = joint space under the Drake solvers.
- **Success: 480/480 on all 16 rows**, zero clock or step-cap stops; each ingredient alone takes most
  rows near the ceiling, only the combination reaches it everywhere (the screw arm needs both).
- **Cost**: cheaper than A-IPOPT on all 12 rigid rows, costlier on soft PCS; **the premium against joint
  space on grasp and soft PCS is the learned formulation's, under every optimizer.**
- **Wall** 0.70-2.00 s every row; **time-matched** the proposed method is 100% everywhere.
- **Fair characterisation** (agreed 2026-10-18): on success and on wall at matched time AL-SVGD >=
  ablations > baselines under interior point; on cost the learned formulation pays a premium on grasp
  and soft PCS. **Strong on success, robustness to the start, matched-time wall and generality; NOT a
  solution-quality result against joint space, and the paper must not claim one.**

**Open**: twelve screw-arm segfaults in R22/R23 (a race at PROCS=2 is what is left; recorded, not
chased). Float64 Newton polish being smoked on branch `svgd-polish` (must not break inequality rows).
GVS under svgd deferred. The soft PCS arm takes the split step (forward-mode row Jacobians).

## Results: the campaign of record (long form: `docs/notes-record.md`)

**Stage REMEASURE (accepted by Thomas 2026-10-09)**: hardened scene, shelf-contained at the fingertips
for both tasks, **180 s**, 480 cells = 60 targets x 8 guesses, seed 1, `--compile --set
flow_cuda_graph=True`, lifted budgets, rungs Panda `n6` / iiwa `n4` / soft PCS `soft12 n6` / screw
`screw7_p050 n6`, arms `learned,numerical`, both protocols, Drake nightly. Wall clocks at paper
conditions (one solve per V100) from stages PAPER + SVGD_R2 (success identical within one cell). **The
record is 64 logical runs: four robots x grasp/pose x two protocols x IPOPT / SNOPT / NLopt
(REMEASURE_NLOPT) / AL-SVGD.** GVS is measured under identical conditions but **outside the record**.
Tables, tally and flag criteria: **`docs/status-quo-tables.md`** (`scripts/report_statusquo.py`;
`--paper`, `--legacy`); attribution `scripts/report_remeasure.py`; pairing `scripts/report_paper.py`.
Layout: rows experiments, each solver a block, learned and joint space adjacent, better starred,
every row printed. Earlier campaigns are superseded and deleted.

- **Success tally (learned / tie / joint space)**: IPOPT 15/1/0, SNOPT 10/1/5, NLopt 11/3/2, AL-SVGD
  2/14/0. Interior point loses nothing. SQP losses survive only from the paired start (plus Panda grasp
  paired and iiwa pose paired). NLopt is extraordinarily start-sensitive: native learned wins all
  eight; paired learned grasp collapses to 3-11 of 480; joint space solves nothing on iiwa/Panda grasp.
  **Read NLopt per protocol, never pooled.**
- **Cost (shared cells)**: learned dearer on grasp under IPOPT (1.9-5.1x, all four robots), near-level
  under SNOPT; cheaper on rigid-arm pose (10 of 12 rows).
- **Wall**: joint space faster on most IPOPT rows; learned faster on Panda grasp and soft PCS grasp
  native under IPOPT. **Iterations**: NLopt and AL-SVGD columns are `N/A` by construction.
- **Time-matched joint space** (`scripts/report_time_matched.py`; restarts from the target's 8 starts
  inside the learned wall; **rule since 2026-10-18: the running solve finishes and counts**, Thomas):
  IPOPT grasp **level** on all four robots; IPOPT pose native **survives** on all four; IPOPT pose
  paired goes to joint space on the rigid arms; SNOPT paired rows all go to joint space; NLopt native
  survives everywhere. **State single-start wins as single-start; rest a time-matched claim only on the
  "survives" rows.**
- **Cap check: no row is budget-bound.** NLopt is clock-bound everywhere, at the fielded clock.
- **Acceptance checks pass**: one `scene_fingerprint` per robot x task, joint space bit-identical
  between protocols on 42/42 pairs, the Panda reproduces the old record under legacy settings.

**The caveat everywhere is per-iteration cost**, an implementation property to report, not a constant.

**Four things are CLOSED and must not be reopened**: the solver axis; placement (shelf-contained;
`free` only as an archive reproducer); the status quo itself; and **multi-start** — Thomas: *"Let's
not go into multi-starting yet, treat it as somewhat out-of-scope for now, and possibly for this
project altogether."* Do not build it, propose it as the next step, or lead a report with it. Thomas's
roadmap: (1) iiwa checkpoints DONE, (2) SNOPT/NLopt DONE, (3) performance tuning and formulation
tweaks live — **not a standing licence to invent formulations.**

**Settled negative knobs — do not re-sweep**: collision shaping (`collision_influence_offset`,
`collision_row_scale`), `ipopt_mu_strategy=adaptive`, `latent_cost_weight` (helps Panda, hurts iiwa),
`correction_bound` (never binds), shelf-depth inset (0.10 stands), scene clutter (stage HARDMUG; the
iiwa/Panda divergence is the robots').

## The chart and the gain ceiling (long form: `docs/notes-chart.md`)

**Chart ladder** (`nb_nodes` 12/8/6/4 + a width control; `cluster/ladder_runs.txt`): reducing depth
helps on both robots; **optimum `n4` iiwa, `n6` Panda**. iiwa: fewer runaways; Panda: cheaper
iterations inside the cap (measured at 45 s — quote the ladder as the 45 s / free-grasp record). **The
ladder is CLOSED, not re-measured.** A net-faster chart needing more steps is a trade-off to report
(Thomas). **Chart accuracy runs backwards to cells**, and stage TRAJ showed training makes iiwa `n6`
worse and does nothing for `n4` — accuracy is not the quantity that matters; that is **not** a licence
to pick early checkpoints (selecting on the test set). **Intrinsic screens do not predict cells**: a
smoke test, not a selection criterion.

**The chart-selection rule**: choose an architecture whose gain ceiling `exp(2.4976 * nb_nodes)` sits
below the ~1e7 runaway band (iiwa: `nb_nodes = 4`), then take the final checkpoint. The ceiling is
FrEIA's `atan` clamp (`|s| < 2.4976` at `rnvp_clamp = 2.5`, 12.15x per block); training walks 78-89%
of the way up whatever log-ceiling it is given. `rnvp_clamp = 2.5` is confirmed optimal.

**The central scientific finding**: residual learned failures on full-depth charts are the joint-limit
row with `|q|` of **1e7-1e16 rad**, all on one ray per robot — a property of the network (sampling the
flow alone reproduces it; iiwa 51x the Panda's exposure). Near-worst-case gain regions of a bounded map,
not poles; the solver is *attracted* to them because `dq/dvars` is as large as `q`. More budget never
helps; neither region knob avoids them (they are spread through the domain). **Every
optimization-side remedy is measured and refuted**: chart accuracy (not the mechanism), IPOPT scaling,
joint-limit penalty, lifting `q` (the runaway relocates), Jacobian regularization (Thomas: *"not worth
it"*). Thomas's ranking: a better chart beats everything; *"You can try it, but I don't like it"* means
measure and report as a stated deviation, not adopt. **Damping an exact Jacobian makes the NLP
inconsistent**; damping belongs on the Newton step, which Drake's IPOPT does not expose. **Suppressing
the runaway does not buy success.**

**Chart ablation, rung one CLOSED (branch `chart-cvae`, merged to main 2026-10-21).** The
program reaches its network only through `src/charts/` (`LoadChart`; `FlowChart` bit-identical to the
old calls), built so LOInK (the next push, its own campaign) plugs in. Rung: a standard cVAE decoder
as the chart, learned and fixed (textbook) decoder variance x K = flow width / redundancy, IPOPT +
AL-SVGD. **Verdict: a loss.** Every rigid native row loses to the flow and to joint space under both
solvers (one IPOPT tie with joint space); screw 0 on every probe; paired start 0/480 everywhere
(encoder round trip 0.75-2.8 rad off); iiwa K=8 learned variance diverged twice (recorded, not
benchmarked). Soft PCS ties the flow, with cheaper cells, only because the decoder returns a low-strain
configuration. **Mechanism: every chart's latent is collapsed or nearly so** (active-unit screen) --
the decoder returns E[q | pose], unusable where the IK fibre is disconnected branches. The screw
dataset rebuild was bit-identical (`n6r` = `n6`). Long form `docs/notes-chart.md`; ops
`cluster/CVAE_RUNBOOK.md`.

## Other robots (long form: `docs/notes-robots.md`)

**The analytic chart.** Panda's 8 branches = wrist/shoulder/elbow signs (`analytic_branches`, default
4; `gc(q, branches=3)`); iiwa charts all 8, so 4-v-8 is Panda-only. The old "residual 0.6%" was the
arccos clip. **Clip ladder** (Thomas: measure both): default 1e-4 / 1e-3 (`ANALYTIC`) and tight 1e-8 /
1.1e-4 (`ANALYTICT`); `ReachabilityConstraint` rows keep the clip inactive at any gate-passing point
(`CheckClipCannotEscapeGate`). Pose-target and mug rows are linear on the pose variables (SNOPT jumps
to them; not a defect). Backends `analytic_backend="cpp"` (`cpp/`, bit-identical AutoDiffXd; stage
ANALYTIC_CPP) and the Python reference. **`analytic8` v `analytic4` is the unbalanced-bundle
pathology**: 8 wins paired, loses native; the whole 4-branch disadvantage is start coverage.

**The soft PCS arm** — always "the soft PCS arm", never "the soft arm". Spatial Piecewise Constant
Strain continuum arm defined with SoRoMoX, after LOInK (arXiv 2609.21275); **configuration is strain**,
normalized to [-1, 1] symmetric about zero (forced by ikflow's offset-free first layer; a stated
adaptation). Rungs `soft9/12/16` differ only in redundancy; `soft12` primary, chart
`soft12__n6__step620000`. Our torch map equals SoRoMoX's (committed golden file
`tests/data/soft_arm_fk_golden.npz`; JAX in float64). **Capsules hang Drake's proximity engine** —
spheres, declared as the robot's geometry. `QAndPose` memoises `(q, pose, cfg)`; `Config(vars)` reads
it. The strain map is differentiated in forward mode, compiled. **The base must be welded at the world
origin**; welding the gripper reorders Drake's bodies (read slots from the plant). No fork of `jrl`
(`SoftArmRobot` shim). **Quote the standalone in-distribution pole screen, never the in-training one.**
Do not build datasets for several robots concurrently. Chart ladder flat (no runaway population); the
DOF ladder measures the baseline. **Learned FK is CLOSED on both soft arms** (Thomas, 2026-09-30): it
cannot change who wins and an expensive forward model is what keeps the comparison fair; the hook
stays, unused. Tables `docs/soft-arm-ladders.md`; ops `cluster/SOFT_ARM_RUNBOOK.md`.

**The GVS push-rod arm** — merged 2026-10-05, **outside the record, does not replace the soft PCS arm**
(Thomas: *"the GVS arm doesn't help our story"*). Tapered three-segment GVS rod driven by nine
normalized rod forces; forward map is **SoRoMoX's own static equilibrium** (Thomas: *"the point is to
use that repo"* — SoRoMoX is the model, not an oracle; test the wrapper). Redundancy enforced
(`ninputs > 6`) and measured; the taper avoids the vacuity trap (a uniform rod zeroes higher Legendre
orders). Joint-space failures are force saturation. IPOPT and SNOPT only, no NLopt (Thomas: *"it's a
waste of time"*). `GVS_ARM_XLA_THREADS=1`. Notes `docs/gvs-arm.md`, tables `docs/gvs-arm-tables.md`,
ops `cluster/GVS_ARM_RUNBOOK.md`.

**The screw-joint arm** (`screw7_p*`, `src/screw_arm/`) — **always "screw", never "helix"**. A
generality demonstration: a screw joint makes FK non-algebraic, so no algebraic IK applies. Invented
from scratch (`src/screw_arm/params.py` **is** the robot; SDF and torch FK agree by test); pitch rungs
{0, 0.025, 0.050, 0.100} m/rev (Drake: metres per revolution), primary `p050`; `p000` is a spec but
**not trained** (Thomas). **TRAP: every Drake parser silently discards a screw joint's `<limit>`** —
`src/screw_arm/limits.py` repairs it after `Finalize()`, before `ToAutoDiffXd()`; anything building the
plant without a program must repair it too. **TRAP: set `ee_frame` BEFORE `CalibrateFlowFrame`**
(`X_ee_flow` must be identity; a test pins it). Collision filter groups need the `cfg_` prefix.
Measured: chart ladder `n6` stands; `p025_n6` shows the gain-ceiling runaway (a property of the trained
chart). `screw7_p050`'s dataset is rebuilt (bit-identical, proved by retraining `n6`); the other pitch rungs' are still gone. Ops and screens
`cluster/SCREW_ARM_RUNBOOK.md`; `python scripts/report_screw.py`. The curvilinear rail is deferred.

## Running on MIT SuperCloud (long form: `docs/notes-cluster.md`; playbook `cluster/README.md`)

4 nodes on `xeon-g6-volta` (40 cores, 2x V100 each), a Slurm `GrpTRES` cap (`node=4`,
`MaxSubmit=240`) shared with the account: submit whole stages and let Slurm meter them; items are
claimed by atomic `mkdir <id>.claim`. **Timing is never compared across machines** (Thomas); cap values
come from `cluster/calibrate.sh`; CPU and GPU contention are measured quantities. **Benchmarks run on
the GPU partition** (Thomas: *"benchmarks have to use GPU"*); the lever when saturated is queue order.
Exceptions on `xeon-p8`: datagen, and arms that never touch the GPU (the analytic arm's
IPOPT/SNOPT/NLopt; `submit_bench.sh` refuses learned/compile/svgd there). PyTorch's cu128 wheels drop
sm_70 — use cu126 (`cluster/smoke.sh` launches a real kernel). `--shard K/N` splits target-major and
the merger re-runs `summarise`; **run `bash cluster/verify_sharding.sh` after any change to sharding,
the merger or grid construction.** Shard sets straddling collections merge normally. **Cluster-wide
checks on the shared account are scoped by JOB name; a guard nobody has seen refuse is untested.**
Monthly maintenance: second Tuesday, Monday evening to Wednesday morning.

**Local compute must never take the laptop down** (Thomas, 2026-10-09: *"If a pytest sweep crashes my
machine, that is unacceptable"*). **No process pools in this project.** Every local test file, smoke or
end-to-end run that builds a program runs ONE AT A TIME, in the foreground, under `systemd-run --user
--scope -p MemoryMax=20G -p MemorySwapMax=0` (swap is his emergency buffer); log loadavg before and
after timed runs, re-run any at load > 24. Detach with `setsid`, not `nohup`; `pgrep -f` can match the
calling shell. The laptop does not sleep on AC; session gaps are deliberate, never diagnosed.

## Open items

- **Analytic arm timing on one machine** (xeon-p8 v volta): deferred (Thomas, 2026-10-13).
- **A `q_c == 0` arm** (eq. 4) to quantify what the correction buys.
- **Non-dimensionalise the conditioning pose's translation against rotation** (cf. `eaik-experiment`'s
  1.12 m length scale). Never tried.
- **Least-squares domain extension** (Thomas's unreleased IFT-IK paper): deferred, belongs here.
  Trust-region *solving* is another project's.
- **A small optimization smoke run in checkpoint validation** (`cluster/export_and_screen_job.sh`):
  Thomas's idea, explicitly deferred.
- **A harder problem formulation** beyond the hardened scene — his idea, his call.
- **The Drake nightly pin expires ~2026-11-02**: move to 1.58.0 once it carries PR 25002.
- `stage_H` in `cluster/gen_manifest.py` is a generic cross-test harness kept for the next knob.
