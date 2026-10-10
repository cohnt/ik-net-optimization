# The svgd solver: pre-registration

**Written 2026-10-08, on branch `svgd` at `39a5db5`, BEFORE any smoke number was read.** Everything
below the "Smoke results" heading is filled in afterwards; nothing above it may be edited to fit what
the smoke shows. A change to a rule above is a new pre-registration, dated, with the reason, and the
old text is kept.

Driver: `scripts/svgd/smoke.py`. Reader: `scripts/report_svgd.py` (it imports the matrix from the
driver, so the two cannot disagree). Reader's guards: `tests/test_report_svgd.py`.

**Re-registered 2026-10-09 (Thomas: the fielded solver is PLAIN augmented-Lagrangian SVGD, a
clear, simple method whose every part is nameable).** The method, its options and the variant
matrix below replace the 2026-10-08 text, which is in git at `f2cb943`; everything beyond the
method is removed, not switched off (the list at the end of this section). The rules from "Go /
no-go" on stand; the smoke must be re-run before any of them is read again.

## What the fourth method class is, and is not

The solver axis is three method classes -- interior point (IPOPT), SQP (SNOPT), augmented Lagrangian
(NLopt `LD_AUGLAG`) -- and that axis is closed. Thomas reopened it for a **fourth class**:
batch-parallel, **SVGD-style** particle methods in torch on the GPU, which exploit the fact that the
flow and the forward kinematics evaluate a batch of N configurations at nearly the cost of one.
Like the other three it is a **reporting axis, never a choice**: shared by both arms within a run,
reported beside the other three classes, not instead of them.

**The method (`svgd_method = al_svgd`, the only one).** Every arm hands the solver its decision
vector x, the full objective f(x), equality rows h(x) = 0 and inequality rows g(x) <= 0 -- the
program's own rows, replayed for N particles by `BatchedProgram` -- and its TRUE variable bounds B
(learned: the correction box `+-correction_bound`; joint space: the joint limits). The c box and the
latent ball are inequality ROWS, never bounds.

1. **Rows in natural units** (Thomas, 2026-10-09). The augmented Lagrangian, its multipliers
   and its dual step use the rows exactly as the program's bindings evaluate them: metres,
   radians, Drake's collision penalty with the program's `collision_row_scale`, joint-limit rows
   in rad, the trust and c-box rows as Drake has them. One length scale relates the row groups:
   orientation-type rows (the rpy residual rows and the c box's rpy rows -- any row in radians
   EXCEPT the joint limits) are multiplied by `svgd_row_length_scale` (default 1.0 m: literally
   the raw rows). The tolerance survives ONLY in the feasibility test: particle i is feasible
   iff `|h_ij| <= tol` for every equality row and `g_ij <= tol` for every inequality row, `tol =
   acceptable_constr_viol_tol` (the old `v_i <= 1`, unchanged; the tolerance ladder is
   untouched). `svgd_row_units = tolerance` is the control: every AL row divided by `tol`, the
   earlier path bit for bit -- equivalent to penalising the natural rows with `rho / tol^2 =
   1e9`, the ill-conditioning this change removes.
2. **Merit**, per particle: the PHR augmented Lagrangian
   `L_i = f + lam_i.h + (rho/2)|h|^2 + (1/(2 rho)) sum_j [max(0, mu_ij + rho g_j)^2 - mu_ij^2]`;
   target density `exp(-L / T)`, `T = svgd_temperature` and `rho = svgd_rho` both fixed for the
   whole solve, `rho` ONE scalar shared by every particle. **The augmented Lagrangian is the
   FORMULATION and SVGD the optimizer running on it** (Thomas, 2026-10-09): one dynamics, no
   inner/outer structure.
3. **Coordinates**: particles move in `y = x / s`, `s` the region half-widths per block (learned:
   `c_position_slack` on the conditioning position, pi on its orientation, the latent trust radius --
   the +-5 box where none is set -- and `correction_bound`; joint space: half of each joint's range),
   read from the program's own options and bounds and recorded as `extras["region_scale"]`.
4. **Direction**, the Tabor-Hermans form (arXiv 2506.00589, the "Q method"):
   `phi_i = (1/N) sum_j [K(q_j, q_i)(-grad_y f_j / T) + grad_{y_j} K(q_j, q_i)] - (1/T) grad_y (L - f)_i`
   -- only the objective's gradient and the repulsion are averaged over the kernel, each particle's
   OWN constraint gradient is added outside the average. **Metric** (`svgd_metric`):
   - `gn` (the fielded configuration): **Stein variational Newton**, block-diagonal (Detommaso,
     Cui, Marzouk, Spantini & Scheichl, "A Stein variational Newton method", NeurIPS 2018):
     `dy_i = svgd_lr (H_i + delta I)^-1 phi_i`, the metric applied to the whole direction,
     repulsion included (SVN's definition), `delta = svgd_gn_lm` (Levenberg damping),
     `svgd_lr = 0.3` (a damped fraction of the Newton step; option table). `H_i` is the Gauss-Newton Hessian of the AL at particle i in y:
     `S (w J_q^T J_q + H_fx + rho J_h^T J_h + rho J_g,act^T J_g,act) S` -- the joint-centering
     cost through dq/dx, the constant Hessian of the costs quadratic in x (the correction
     penalty), and the rows with the PHR active set `mu_ij + rho g_ij > 0`. Solved per particle
     by a batched `solve_ex` (graph-capturable under the cuSOLVER pin).
   - `identity`: the plain direction with a per-particle step `dy_i = (svgd_lr / ||H_i||_F)
     phi_i`. The Frobenius norm is a safe upper bound on `lambda_max(H_i)` (at most
     sqrt(rank) above it) and a pure reduction, so the step stays graph-capturable; the exact
     `lambda_max` (eager `eigvalsh`, which a CUDA graph cannot capture) is recorded as a
     diagnostic only, at the dual-update checks under both metrics, with the median ratio
     `||H||_F / lambda_max`.

   Then `y_i <- clamp_B(y_i + dy_i)`; the clamp's distance in y is recorded (`bound_clip`). No
   momentum, no adaptation. The median ratio `|repulsion| / |drive|` within phi is recorded
   per solve (`repulsion_ratio_median`): the kernel's actual share of the step.
5. **Kernel**: RBF on the configuration q (flow output + correction on the learned arm, q itself on
   joint space), median bandwidth `h = med^2 / log N` floored at `svgd_bandwidth_floor`; the
   repulsion is pulled back to y through the flow's VJP. `svgd_kernel = none` drops both kernel
   terms (`phi_i = -(1/T) grad_y L_i`, N independent projected gradient descents, batched).
6. **Dual ascent with step `svgd_dual_lr`, on a fixed cadence, rho fixed.** Every K =
   `svgd_inner_iters` steps, on EVERY particle and unconditionally (no feasibility gate of any
   kind): `lam_i <- lam_i + alpha h_i`, `mu_i <- max(0, mu_i + alpha g_i)`, `alpha =
   svgd_dual_lr`, at the swarm's current rows; clipped to `+-svgd_multiplier_max`, the clipped
   entries counted. The textbook step `alpha = rho` is exact only when the primal is minimised
   between updates, which it is not here (gradient descent-ascent), so the dual rate is a separate
   parameter (Thomas, 2026-10-09). `alpha = 0` is a pure quadratic penalty: the update is still
   taken and recorded, and leaves the multipliers at zero. The multipliers are per particle, the
   penalty is not. Recorded per cell: the dual updates taken, the median and largest
   `|lam_i|_inf` and `|mu_i|_inf` at stop, and the clip count.
7. **Resampling** at every check: a particle with `|q|_inf > svgd_resample_q_max` or a non-finite row
   is redrawn from the arm's NATIVE start distribution with zero multipliers (`n_resampled`).
8. **Stop**: the wall clock (`max_wall_time`, less `svgd_time_reserve` for the re-check), the
   outer-step cap (`svgd_outer_iters`), or -- once some particle is feasible at the gate -- when the
   best feasible particle's objective f has improved by less than `svgd_stop_rel` (relative) over
   `svgd_stop_patience` consecutive checks. `stop_reason` is `converged`, `wall_clock` or
   `step_cap`.
9. **Selection**: among the particles feasible on the batched rows (in the swarm's dtype), in
   objective order, the top `svgd_recheck_topk` are re-checked exactly in Drake (`prog.EvalBinding`
   in float64 on the same x, clamped onto B in float64 to undo the swarm dtype's rounding of a
   bound); the first passer is returned, else the smallest Drake violation, scored infeasible.
   `verify()` then scores it exactly as it scores IPOPT's point. The solver's verdict
   (`solver_feasible`), its Drake re-check (`drake_feasible`) and `verify()` must agree; a
   disagreement is a bug, not a result.
10. **Precision**: a float32 swarm; `svgd_dtype = float64` is a control.
11. **Initialisation**: particle 0 is the cell's initial guess exactly, never clipped; the others are
    `y0 + svgd_jitter * N(0, I)` (`svgd_paired_init = jitter`) or the arm's native draw (`native`),
    clamped onto B. The CEM warm-up (`svgd_warmup = cem`) is a phase of the method, A/B tested.
12. **The literal form** is reachable as an A/B: `svgd_constraint_inside_kernel = True` puts the
    whole `-grad L / T` inside the kernel average. Nothing else.

What it **is not**: not multi-start (N particles inside ONE solve on ONE clock is the method;
restarting a solve is multi-start, ruled out of scope); not a re-sweep of the closed Drake classes
(the IPOPT twin runs the adopted configuration plus `flow_cuda_graph=True`); and it uses **no
formulation-specific information** (no latent prior, no assumption that the correction is
zero-centred, no arm-specific kernel; the only structure used is that every arm has a q and that
every variable block has a region). Recorded future work, not current scope: using
formulation-specific information rigorously with SVGD. **Exact Drake collision, no proxy**: the
collision row is the program's own Drake `MinimumDistanceLowerBoundConstraint`, evaluated IN
PROCESS (Thomas, 2026-10-09: no process pool): a Python loop over the N particles calling it on
AutoDiffXd (`Eval(InitializeAutoDiff(q))`, exactly the program's binding), returning value and
gradient. It is serial and GIL-bound (pydrake holds the GIL in `Eval`); a batched C++
clearance-with-Jacobians call that releases it is a Drake-side item for Thomas. Its seconds are
reported as the collision row's share of the wall.

### The options, one table

| option | default | part of the method |
| --- | --- | --- |
| `svgd_method` | `al_svgd` | the method (one value) |
| `svgd_n` | 64 | N particles (1 = the single-particle control) |
| `svgd_dtype` | `float32` | swarm precision (`float64` a control) |
| `svgd_seed` | 0 | particle draws, mixed with a CRC of the initial guess |
| `svgd_paired_init` | `jitter` | 11: `jitter` or `native` |
| `svgd_jitter` | 0.1 | 11: jitter sigma in the normalised coordinates |
| `svgd_temperature` | 1.0 | 2: T, fixed |
| `svgd_lr` | 0.3 | 4: a fraction of the metric's step: `svgd_lr (H + delta I)^-1 phi` (gn) or `svgd_lr / \|\|H\|\|_F` (identity). Chosen 2026-10-10 by the screen at rho 1e3, delta 10 on 6 grasp + 6 pose Panda cells, most feasible cells with ties to the larger lr: lr 0.3 solved 24/24 (learned 12/12, joint space 12/12) resampling 0.07 of N per check, lr 1.0 solved 22/24 resampling 0.12 |
| `svgd_metric` | `gn` | 4: `gn` (Stein variational Newton) or `identity` |
| `svgd_gn_lm` | 10 | 4: delta, the Levenberg damping of the GN metric. Chosen 2026-10-10 by the most feasible cells (learned + joint space) on 6 grasp + 6 pose Panda cells, ties to the larger: delta 1e-2 / 1 / 10 gave 2 / 3 / 6 (1e-4 diverged); then svgd_lr 1.0 / 0.3 / 0.1 at delta 10 gave 6 / 0 / 0 -- all at rho 10, and the 0.3 / 0.1 cells were mostly `error` records from the since-fixed eigvalsh crash, so that lr reading is void; `svgd_lr` was chosen later by the screen at rho 1e3 (its row) |
| `svgd_row_units` | `natural` | 1: `natural` or `tolerance` (the control) |
| `svgd_row_length_scale` | 1.0 | 1: metres per radian on the orientation-type rows |
| `svgd_kernel` | `q` | 5: `q` or `none` |
| `svgd_bandwidth_floor` | 0.05 | 5: floor on the median bandwidth |
| `svgd_constraint_inside_kernel` | False | 12: the literal form, an A/B |
| `svgd_rho` | 1000 | 2, 4: the penalty -- one scalar, fixed, shared. Chosen 2026-10-10 on 6 grasp + 6 pose Panda cells at delta 10, lr 1: rho 10 / 1e3 / 1e4 / 1e5 solved 5 / 12 / 12 / 12 learned and 0 / 10 / 9 / 11 joint space; 1e3 is the smallest that works and resamples least (0.08-0.12 of N per check against 0.17-0.21 at 1e5). With the Newton metric a large rho no longer costs conditioning. At rho = 10 the repulsion was 0.97 of the step and swamped the drive (N = 1 solved grasp 6/6 where N = 64 solved 0/6) |
| `svgd_dual_lr` | None (= `svgd_rho`) | 6: alpha, the dual-ascent step; 0 = pure quadratic penalty. The default is pending the alpha ladder |
| `svgd_multiplier_max` | 1e4 | 6: multiplier clip |
| `svgd_inner_iters` | 10 | 6, 7, 8: K, the cadence of the dual step, resampling and the stop rule |
| `svgd_outer_iters` | 300 | 8: the step cap (`max_iter`, when set, also binds) |
| `svgd_resample_q_max` | 10.0 | 7: runaway threshold, rad |
| `svgd_stop_patience` | 5 | 8 |
| `svgd_stop_rel` | 1e-3 | 8 |
| `svgd_recheck_topk` | 10 | 9 |
| `svgd_time_reserve` | 0.1 | 8: fraction of the cap kept for selection and re-check |
| `svgd_warmup`, `svgd_warmup_iters`, `svgd_warmup_elite` | `none`, 10, 0.1 | 11: the CEM phase |
| `svgd_compile`, `svgd_cuda_graph` | False, False | execution: the split step eager / compiled / graphed |

**The step size.** Under `gn` the metric carries the scale, so `svgd_lr = 1` would be the full Newton step
on the GN model of the AL; the fielded 0.3 is a damped fraction of it (screen, option table). Under `identity`, `svgd_lr / ||H_i||_F <= svgd_lr / lambda_max(H_i)`
is a per-particle Lipschitz step. (The earlier fixed step, `1e-10 / rho` on rows divided by tol,
was forced by the `rho / tol^2` curvature: pose-row curvature ~1e10 against objective curvature
1e-4..10, which froze the self-motion and left the order-1 repulsion numerically dead. Its
measurements are in git, a9758e1.)

**Memory.** No worker processes: the collision row runs in the solver's own process on the
program's own scene. Local runs still go under `systemd-run --user --scope -p MemoryMax=...
-p MemorySwapMax=0` (the smoke driver does this by default).

### Removed, 2026-10-09

- **Annealed repulsion temperature** (`svgd_repulsion_T0`, `svgd_anneal_frac`) and the
  **driving-force ramp** (`svgd_gamma_t`): T is fixed.
- **Gauss-Newton equality correction** (`svgd_gn_every`, `svgd_gn_delta`) with its
  **Levenberg-Marquardt damping, gain ratio** (the old `svgd_gn_lm*` family) and the
  cost-weighted metric. (The name `svgd_gn_lm` returned later the same day with a different
  meaning: the fixed Levenberg damping of the Stein variational Newton metric, item 4.)
- **q-step clamp** (`svgd_q_step_max`).
- **float64 polish** (`svgd_polish_iters`, `svgd_polish_tol`, `svgd_polish_topk`): the re-check is
  on the swarm's own point.
- **Adam** and the **per-particle learning-rate schedule** (`svgd_lr_decay_t`, `svgd_lr_min`): a
  plain gradient step.
- **Relative-eta multiplier test** (`svgd_eta_rel`) and NW's failure-branch tolerance reset.
- **Running best per particle**: selection reads the swarm at stop.
- **`tsvgd`** (tangent-space SVGD, `svgd_tsvgd_switch_infeas`, `svgd_tangent_delta`) and
  **`admm_svgd`** (`src/svgd/admm.py`, `svgd_admm_*`).
- **Kernel in x** (`svgd_kernel = x`), the **fixed bandwidth** (`svgd_bandwidth`), the
  **row-sum kernel normaliser** (the average is the textbook 1/N).
- **Orientation-row scale** (`svgd_row_scale_rot`), **per-block jitter sigmas** (`svgd_jitter_z`,
  `_c_pos`, `_c_rot`, `_qc`, `_q`; one `svgd_jitter`), **resample period** (`svgd_resample_every`;
  every check).
- **The cuSOLVER pin** (`preferred_linalg_library`): no batched linear solve remained. The
  recompile-limit raise and the inductor options stay (the profiler's many structures need them).
  (The pin RETURNED later the same day with the Stein variational Newton solve, item 4: a batched
  `solve_ex` is graph-capturable only under it.)
- **The per-particle penalty and every gate on the multipliers** (later the same day): the
  per-particle `rho_i` with Powell's growth test (`svgd_rho0`, `svgd_rho_growth`, `svgd_rho_gamma`,
  `svgd_rho_max`), the feasibility tolerance `eta_i` and its `rho0^-0.1` start, `v_prev`, and the
  `rho_median` / `rho_max` / `rho_at_stop` details. Thomas: the augmented Lagrangian is the
  formulation and SVGD the optimizer; the eta / Powell gating was the bilevel method of
  multipliers. `rho` is now one fixed scalar (`svgd_rho`) and the dual step is unconditional on a
  fixed cadence. (Its first 6-cell check, 9756ff4: every particle's rho reached the 1e6 cap within
  six checks and no multiplier ever updated -- learned 2/6 grasp, 0/6 pose.)
- **The process pool for the collision row** (`DrakeCollisionPool`, its worker processes and
  `_worker_main`, the live-worker registry and memory guard `admit` / `PoolRefused` /
  `mem_available_gb` / `SVGD_MAX_LIVE_WORKERS`, `resolve_workers`, `POOL_CACHE_SIZE`, and the
  options `svgd_collision_workers` and `svgd_pool_overlap`). Thomas: no process pool; the row
  runs in process on the program's own constraint. The pool's processes, each a whole Drake scene,
  OOM-killed the laptop on 2026-10-09.

## The variants, every option named in full

Every svgd column is `al_svgd` and sets, by `--set`, the settings below; every other `svgd_*` field
is at its `ProgramOptions` default, and the run's metadata records what was handed to the solver
(`solver_options_emitted`). Each column differs from the primary `al64` in ONE setting, so the
reporter reads each as an A/B against it.

| column id | `svgd_n` | `svgd_kernel` | `svgd_constraint_inside_kernel` | `svgd_paired_init` | `svgd_warmup` | role |
| --- | --- | --- | --- | --- | --- | --- |
| `al1` | 1 | `q` | False | jitter | none | single-particle control |
| `al64none` | 64 | `none` | False | jitter | none | no-interaction control (batched AL) |
| `al64` | 64 | `q` | False | jitter | none | **primary** |
| `al256` | 256 | `q` | False | jitter | none | the N ladder's upper rung |
| `al64lit` | 64 | `q` | True | jitter | none | the literal SVGD form |
| `al64native` | 64 | `q` | False | native | none | the native swarm init |
| `al64cem` | 64 | `q` | False | jitter | cem | the CEM warm-up A/B |

On every svgd column also `svgd_dtype=float32`, `svgd_compile=True`, `svgd_cuda_graph=True` (mode
`graphed`). The IPOPT twin
(`ipopt`) is `--solver ipopt --set flow_cuda_graph=True` with everything else as the record's IPOPT
column.

Tags name every setting: `smoke_SVGD_<robot>_<rung>_<variant>_<row>_<cells>_<cap>_<start>`, variant
`svgd-al_svgd-n<N>-kernel_<kernel>-constraint_inside_kernel_<bool>-warmup_<warmup>-init_<paired_init>-<dtype>-<mode>`
or `ipopt-flow_cuda_graph`.

## The smoke matrix

- **Grid**: the record's own, `--targets 60 --guesses 8 --seed 1 --scene hardened --shelf-inset 0.1
  --config latent --set correction_cost_weight=10.0 --compile`, with stage STATUSQUO's placement per
  row, restricted by `--cells 0:0,0:1,13:0,13:1,26:0,26:1,39:0,39:1,52:0,52:1` (10 cells: two guesses of
  every 13th target). `grid_hash` hashes the whole grid, so every smoke run carries the record's hash.
- **Robots**: Panda `n6` (the record's rung); iiwa `n6` -- **not the record's `n4`, which is not on
  this laptop**, so the iiwa learned arm pairs against its IPOPT twin only (rung token
  `n6-not-record-n4`).
- **Rows**: grasp contained (`mugshelf`) and pose contained at the fingertips (`posetip`), each under
  `paired` and `native`: 8 rows.
- **Arms**: `learned,numerical`.
- **Columns**: the seven svgd columns above plus the IPOPT twin: 8 per row, 64 runs.
- **Cap**: 20 s per cell (`--wall-time`, a driver option). The record ran at 180 s on SuperCloud V100s;
  the record pairs on successes only, never on seconds, and the twin is the same-machine, same-cap
  comparison.
- **The iiwa grasp rows wait for the mug-handle fix.** The record's wsg-gripper grasp rows predate
  `452d784` (branch `mug-handle-yaw`) and are void; those rows are run once that fix is merged into
  `svgd`, and are read against the twin alone (the reporter refuses the record there).
- Sequential on the laptop GPU (`--jobs 1`).

## Go / no-go for a cluster stage

All must hold; the reporter evaluates (1)-(7) mechanically where the summaries decide them.

1. zero `error` records and no dead-arm abort, tests green;
2. on >= 1 row, >= 1 variant's learned successes >= the record's IPOPT learned count on those cells,
   and joint space not dead;
3. the solver's own feasibility verdict agrees with `verify()` on every cell;
4. wall <= cap + 1 s everywhere and `recovered_*` populated on a kill test;
5. ms/step within 2x of `profile_step`;
6. resampling < 10% of N;
7. N=1 does not beat N=64 on the same method;
8. the selected variant(s) and N are written here BEFORE any cluster manifest is generated.

Readings fixed in advance. In (2), "the record's IPOPT learned count on those cells" is the record's
learned successes on the same 10 cells where the record's chart is the smoke's (Panda); on the iiwa,
and on any row whose record is void, it is the IPOPT twin's learned count on those cells. In (3),
"agrees" is checked twice per cell: `solver_feasible == drake_feasible` and
`drake_feasible == feasible`. In (5), `profile_step` means the same arm, method, N, dtype and mode on
the same task, Panda (the profiler builds Panda programs only); smoke ms/step is the swarm
phase over outer steps. In (6), per cell `n_resampled / svgd_n`, the worst cell. In (7), success
counts of `al1` against `al64`, per arm, per row. Beside every success count the reporter prints
the two population metrics: feasible particles at stop and the median pairwise q-distance among
them (`feasible_q_spread`).

A flag registered before the smoke was read (2026-10-09 01:45, from the solver wave's 6-cell
end-to-end checks, not from the smoke): on learned `al_svgd` pose cells the median correction
`|q_c|_inf` was 0.100, every solution on the +-0.1 box, where IPOPT's record median is ~0.054.
CLAUDE.md names this exactly ("the check that the learned arm is not quietly becoming a
reparameterised joint-space arm"). The smoke's `median_correction_inf` and `correction_binding`
are therefore read beside every success count; a variant whose solutions sit on the box is reported
with that beside it and is not selected over one that does not on success alone. The suspected
mechanism is the minimum-norm Gauss-Newton correction spending the residual on `q_c`, the cheapest
direction in its unscaled metric; it is unaddressed in the fielded code and is NOT a formulation
change.

## The cluster stage's per-row analysis rule

For each row (robot x experiment x start protocol), per selected variant, on 480 cells at the
record's 180 s:

**The question is the LEARNED arm's performance under svgd** (Thomas, 2026-10-10: *"the goal is
performance on the learned arm. joint space SVGD is an ablation, not a baseline"*). The joint-space
svgd column is an ablation -- the swarm without the network -- never the comparison target the record's
Drake columns have.

- **Learned under svgd vs learned under IPOPT**, on the same cells, exact two-sided McNemar -- the
  record's IPOPT column where its chart and scene are the run's, else an IPOPT column measured beside
  it. A verdict is a win, a tie (p >= 0.05) or a loss, and ties are stated as ties. This is the lead
  result of every row.
- **Joint space under svgd, as an ablation**: printed beside the learned column with its own McNemar
  against the record's joint-space IPOPT column, so the solver's effect on each formulation is visible;
  the learned-vs-joint-space McNemar under svgd is printed but is NOT a headline and no verdict-flip
  tally is built on it.
- **The cap rule**: read both `timed_out` and `hit_iteration_cap` (the svgd step cap counts as the
  iteration budget). A row with >= 24 of 480 cells at the iteration budget carries no verdict until
  re-measured with the budget lifted; the clock is never raised, so clock-bound rows are results at the
  fielded clock and are flagged as such.
- **Both start protocols are reported separately, never pooled.**
- **Every result is told in success, iterations, cost and wall clock**: success with the McNemar;
  svgd outer steps, labelled as NOT comparable to IPOPT majors; cost as the reported cost (learned-only
  regularizers excluded) on cells both arms solved, `N/A` under 10 such cells; mean wall over all cells
  and ms/step. Tables put learned and joint space adjacent, mark the better, print every row including
  zeros, and name every setting in full.
- Beside them, from `record["svgd"]`: feasible particles, resampled fraction, `selected_index`, the
  collision row's share of the wall, `stop_reason` counts, warm-up/compile seconds, and the
  feasibility-agreement check, which must read zero.

## Smoke results

### The re-registered method, measured 2026-10-10

Measured 2026-10-10 00:50-04:39 EDT on the laptop (RTX 3080 Ti, 20 cores), in five steps:
the delta and lr probes, then a rho screen, then the attribution, then a one-setting-at-a-time
screen, and finally the re-smoke. Each step chose a default, and the next step ran at it. The
first four steps run on a **development grid**, not the smoke matrix:

- the Panda `n6`, both tasks (`mugshelf` grasp, `posetip` pose), `paired` start;
- `--targets 3 --guesses 2 --seed 1`, which gives 6 cells per task, 12 per arm and 24 per setting;
- a 20 s cap, N = 64, `svgd_kernel=q`, a float32 swarm, compiled and graphed;
- `--set correction_cost_weight=10.0 --config latent --scene hardened --shelf-inset 0.1
  --target-placement shelf`, with arms `learned,numerical`.

Every run went alone, in the foreground, under `systemd-run --user --scope -p MemoryMax=20G -p
MemorySwapMax=0`, with loadavg logged at both ends. No run in these tables exceeded 2.8, so none
is contaminated.

**Every table prints its `error` cells.** The `lambda_max` diagnostic (an eager `eigvalsh` at the
checks) raised `LinAlgError` on an ill-conditioned float32 H. The benchmark records such a cell as
`fail_reason = error`, so the cell is not counted as a solver failure.

- The fix: the diagnostic now runs in float64, with NaN on failure (7dec630).
- Affected: the delta and lr probes and the first (rho 10) attribution, which carry error cells.
- Clean: every run from the rho screen on has zero.
- The affected runs were not re-run, because the rho screen moved the default under them
  (coordinator's call).

**1. The delta probe** (`svgd_gn_lm`; rho 10, lr 1, alpha = rho, K 10). The rule, fixed before
the probe: most feasible cells, ties to the larger delta.

| delta | learned errors | learned grasp / pose | joint space grasp / pose | resampled / N per check, learned median (grasp / pose) |
| --- | --- | --- | --- | --- |
| 1e-4 | 0 | 0/6 / not run | 0/6 / not run | 1.00 (all 64, every check) / not run: diverges |
| 1e-2 | 0 | 1/6 / 0/6 | 0/6 / 1/6 | 1.00 / 1.00 |
| 1 | 4 | 0/6 / 2/6 | 0/6 / 1/6 | 0.06 / 0.02 |
| 10 | 4 | 0/6 / 5/6 | 0/6 / 1/6 | 0.01 / 0.01 |

delta = 10 was taken. Resampling falls with the damping, from every particle at every check to 1%.
It was committed as provisional (7dec630). **The lr probe that followed is void**: at delta 10, lr
0.3 and lr 0.1 each scored 0/12 on both arms, but in each run 9 of the 12 learned cells were `error`
records.

**2. The rho screen** (delta 10, lr 1, alpha = rho, K 10). At rho 10 nothing worked: learned 5/12
with 4 error cells, joint space 0/12. Grasp failed everywhere, the repulsion was 0.97 of the step,
and N = 1 solved grasp 6/6 where N = 64 solved 0/6.

| rho | learned errors | learned grasp / pose | joint space grasp / pose | total / 24 | resampled / N per check (learned, worst task) |
| --- | --- | --- | --- | --- | --- |
| 10 | 4 | 0/6 / 5/6 | 0/6 / 0/6 | 5 | 0.01 |
| 1e3 | 0 | 6/6 / 6/6 | 4/6 / 6/6 | 22 | 0.12 |
| 1e4 | 0 | 6/6 / 6/6 | 3/6 / 6/6 | 21 | 0.21 |
| 1e5 | 0 | 6/6 / 6/6 | 5/6 / 6/6 | 23 | 0.21 |

rho = 1e3 was taken (76fb323) because it is the smallest rho that works and it resamples least.
1e5's one extra cell is a joint-space grasp cell. With the Newton metric a larger rho costs no
conditioning, only churn. The same settings run twice, the screen's rho 1e3 run and attribution
(a), **reproduce success cell for cell** (learned 12, joint space 10). The outer-step and
`lambda_max` medians reproduce too. Only the failing joint-space cells' final violation moves (0.68 against
0.56); those cells stop on the wall clock.

**3. The attribution** at the fielded configuration of the time (rho 1e3, delta 10, lr 1, alpha =
rho, K 10). It changes the row units and the metric one at a time:

| config | arm | error cells | grasp | pose | med outer steps | med failing max_viol | med \|lam\|_max | multiplier clips | lambda_max med / max | \|\|H\|\|_F / lambda_max | repulsion / drive | collision share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) natural + gn (the fielded configuration) | learned | 0 | 6/6 | 6/6 | 10.5 | -- | 1.9e2 | 12276 | 2.6e4 / 3.4e38 | 1.02 | 0.16 | 0.85 |
| | joint space | 0 | 4/6 | 6/6 | 22 | 0.56 | 1e4 | 24512 | 2.4e4 / 1.2e8 | 1.05 | 1.3e-5 | 0.95 |
| (b) natural + identity | learned | 0 | 6/6 | 2/6 | 36.5 | 3.4e-4 | 8.2e3 | 13319 | 6.1e4 / 2.9e38 | 1.01 | 0.0029 | 0.85 |
| | joint space | 0 | 0/6 | 0/6 | 74.5 | 0.13 | 1e4 | 6916 | 1.6e4 / 1.9e8 | 1.12 | 5.9e-5 | 0.94 |
| (c) tolerance + gn | learned | 0 | 0/6 | 0/6 | 61 | 1.0e-3 | 0 | 932090 | 1 / 2.8e38 | 4.47 | 0 | 0.85 |
| | joint space | 0 | 3/6 | 6/6 | 12 | 0.15 | 1e4 | 89551 | 1.8e12 / 4.2e14 | 1.12 | 2.7e-13 | 0.96 |
| (d) tolerance + identity | learned | 0 | 6/6 | 2/6 | 31.5 | 0.059 | 1e4 | 207004 | 2.5e13 / 2.9e38 | 1.00 | 3.7e-11 | 0.86 |
| | joint space | 0 | 2/6 | 0/6 | 69.5 | 0.0092 | 1e4 | 263916 | 2e12 / 2.4e13 | 1.15 | 1.5e-10 | 0.95 |

(b) and (d) run at `svgd_lr=1.0` under `identity`. A `lambda_max` maximum near 3e38 is float32's
ceiling: it is a runaway particle's H at the check, before resampling redraws it. Medians are over
cells.

- **The learned arm needs both changes together.** Natural units with the Newton metric gives
  12/12. Either alone gives 8/12 (b, d). The Newton metric on tolerance-scaled rows (c) gives
  **0/12**, and every particle is redrawn at every check: 3,700-4,000 redraws in 58-63 checks per
  cell. So no multiplier survives (median `|lam|_max` 0). The median particle's H is the identity
  the solver substitutes when H or the rows are non-finite: `lambda_max` is 1 and
  `||H||_F / lambda_max` is sqrt(20).
- **Joint space needs the metric, and the units barely matter.** The Newton metric gives 10 and 9
  (a, c), against 0 and 2 under identity (b, d).
- **The kernel acts only where both changes are in.** The repulsion is 0.16 of the learned arm's
  step under (a) and at most 0.003 anywhere else.
- **The conditioning change removed the 1e12-1e13 curvature** of the tolerance-scaled rows.
  Under natural units the median `lambda_max` is 1.6-6.1e4.
- `||H||_F` stays within 1.0-1.15x of `lambda_max` everywhere except (c)'s learned arm, so the
  Frobenius step under `identity` is close to the exact Lipschitz step.

The same ladder at rho 10, the earlier default, read (learned / joint space of 12): (a) 5/0 with 4
error cells, (b) 0/0 with 6 error cells, (c) 3/0, (d) 6/1. A fifth column, (e), was the committed
a9758e1 (tolerance units, before the metric existed), which read 2/0. Nothing worked at rho 10,
which is why the rho screen came first.

**4. The screen**: one setting at a time from the fielded configuration (rho 1e3, delta 10, lr 1,
alpha = rho, K 10, N 64, T 1, kernel `q`, jitter paired init, Tabor-Hermans form). The verdicts
were fixed before the screen and are mechanical:

- DIVERGES: either arm resamples more than 0.10 of N per check (median, worst task).
- UNSTABLE: the total is at least 2 below the reference's.
- SLOW: the total ties the reference's, but the median check at which a first particle is feasible
  is more than twice the reference's.
- OK: otherwise.

| setting | errors (L / J) | learned grasp / pose | joint space grasp / pose | total / 24 | med failing viol, J | resampled / N per check (L) | stops, learned | stops, joint space | ms / inner step (L / J) | med check to 1st feasible | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference (lr 1) | 0 / 0 | 6/6 / 6/6 | 4/6 / 6/6 | 22 | 0.56 | 0.12 | 12 converged | 10 converged, 2 clock | 28.5 / 25.8 | 3 | (reference) |
| `svgd_rho=10` | 4 / 0 | 0/6 / 5/6 | 0/6 / 0/6 | 5 | 0.047 | 0.01 | 5 converged, 3 clock, 4 error | 12 clock | 27.7 / 25.6 | 29 | UNSTABLE |
| `svgd_rho=1e4` | 0 / 0 | 6/6 / 6/6 | 3/6 / 6/6 | 21 | 0.62 | 0.21 | 12 converged | 9 converged, 3 clock | 28.8 / 26.9 | 2 | DIVERGES |
| `svgd_rho=1e5` | 0 / 0 | 6/6 / 6/6 | 5/6 / 6/6 | 23 | 0.061 | 0.21 | 12 converged | 11 converged, 1 clock | 28.3 / 27.6 | 2 | DIVERGES |
| `svgd_inner_iters=1` | 0 / 0 | 6/6 / 6/6 | 0/6 / 0/6 | 12 | 0.89 | 0.09 | 12 converged | 10 step cap, 2 clock | 60.1 / 50.7 | 6 | UNSTABLE |
| `svgd_inner_iters=100` | 0 / 0 | 6/6 / 6/6 | 2/6 / 5/6 | 19 | 4.3e-4 | 0.13 | 5 converged, 7 clock | 3 converged, 9 clock | 26.7 / 25.3 | 2 | DIVERGES |
| `svgd_n=1` | 0 / 0 | 6/6 / 6/6 | 0/6 / 3/6 | 15 | 1.2 | 0.01 | 10 converged, 2 step cap | 3 converged, 9 step cap | 4.7 / 1.7 | 3 | UNSTABLE |
| `svgd_n=256` | 0 / 0 | 6/6 / 6/6 | 5/6 / 6/6 | 23 | 0.079 | 0.11 | 12 converged | 6 converged, 6 clock | 103.4 / 106.6 | 1 | DIVERGES |
| `svgd_n=1024` | 0 / 0 | 6/6 / 6/6 | 6/6 / 6/6 | 24 | -- | 0.20 | 12 clock | 12 clock | 417.4 / 437.1 | 1 | DIVERGES |
| `svgd_kernel=none` | 0 / 0 | 6/6 / 6/6 | 0/6 / 6/6 | 18 | 0.60 | 0.07 | 6 converged, 6 clock | 6 converged, 6 clock | 28.6 / 26.2 | 2 | UNSTABLE |
| `svgd_temperature=0.1` | 0 / 0 | 0/6 / 0/6 | 0/6 / 0/6 | 0 | 1.2 | 1.00 | 12 clock | 12 clock | 29.1 / 24.4 | -- | DIVERGES |
| `svgd_temperature=10` | 0 / 0 | 6/6 / 6/6 | 3/6 / 5/6 | 20 | 1.5e-3 | 0.03 | 12 converged | 8 converged, 4 clock | 30.0 / 26.5 | 18 | UNSTABLE |
| `svgd_paired_init=native` | 0 / 0 | 6/6 / 6/6 | 4/6 / 6/6 | 22 | 0.67 | 0.05 | 12 converged | 9 converged, 3 clock | 29.4 / 25.7 | 3 | OK |
| `svgd_constraint_inside_kernel=True` | 0 / 0 | 0/6 / 0/6 | 0/6 / 0/6 | 0 | 0.49 | 0.84 | 12 clock | 12 clock | 34.0 / 28.5 | -- | DIVERGES |
| **`svgd_lr=0.3`** | 0 / 0 | 6/6 / 6/6 | 6/6 / 6/6 | **24** | -- | 0.07 | 12 converged | 12 converged | 28.0 / 26.1 | 7 | **OK** |

The `svgd_rho=10` row is attribution (a) at rho 10. The `svgd_inner_iters=1` row ran at rho 1e3.

- **The reference sits over its own resampling line** (0.12 of N per check), so DIVERGES at
  0.11-0.21 marks the reference's churn level or more. It does not mark a swarm that fails: rho
  1e4, rho 1e5, N 256 and K 100 still solve 19-23 of 24. Only `svgd_temperature=0.1` and the
  literal form diverge in the plain sense: 0 of 24, with 84-100% of the swarm redrawn at every
  check.
- **`svgd_lr=0.3` is the one setting that beats the reference without churning**: 24/24, with
  resampling 0.07 of N per check. By the rule fixed beforehand (most feasible cells, ties to the
  larger lr), it is the fielded step (a397b75). The cost is about twice the checks to a first
  feasible particle (7 against 3). Wall clock per inner step is unchanged.
- **The kernel is worth 4 cells, all on joint-space grasp.** `svgd_kernel=none` takes joint-space
  grasp from 4/6 to 0/6, and the learned arm is unaffected.
- **N = 1 does not beat N = 64** (pre-registered check 7): it is 15 against 22, losing 7 joint-space
  cells to the step cap. N = 1024 solves 24/24 but never stops on its own (12 + 12 wall clock, 417 ms per inner
  step).
- The dual cadence has an interior optimum. K = 1 starves joint space into the step cap (0/12), and
  K = 100 runs to the clock on 16 of 24 cells.
- The temperature has one too. T = 0.1 diverges, and T = 10 slows the first feasible particle six
  times.
- The native swarm init ties the jitter init exactly (22/24, the same cells on both arms).
- The collision row is 85-86% of the learned arm's step time and 94-96% of joint space's, in every
  configuration above (attribution table), so ms per step is the in-process Drake loop's.

**5. The re-smoke** at the new defaults (rho 1e3, delta 10, lr 0.3; a397b75), measured 04:21-04:39
EDT. This is the pre-registered matrix's Panda rows (`scripts/svgd/smoke.py --rows panda --columns ipopt,al64`):
the record's 10 cells, both tasks, both starts, the primary `al64` beside the IPOPT twin, at 20 s,
under the 24G cap per child. All 8 runs wrote a summary, with 0 `error` records, and load stayed
at 2.1 or below at both ends. Full tables: `python scripts/report_svgd.py --rows panda --columns
ipopt,al64`.

| row | learned svgd | joint space svgd | McNemar (svgd) | learned IPOPT twin | joint space IPOPT twin | record's IPOPT learned |
| --- | --- | --- | --- | --- | --- | --- |
| grasp paired | **10**/10 | 4/10 (6 clock) | 6 / 0, p = 0.031: learned | 10/10 | 9/10 | 9 |
| grasp native | **10**/10 | 4/10 (6 clock) | 6 / 0, p = 0.031: learned | 9/10 (1 clock) | 9/10 | 10 |
| pose paired | 10/10 | 10/10 (1 clock) | tie | 9/10 | 6/10 | 6 |
| pose native | 10/10 | 10/10 (1 clock) | tie | 10/10 | 6/10 | 9 |

Error cells: 0 on every row. Only the learned arm runs the flow, so the record has no joint-space
column to pair against here.

| row | arm | outer steps (median) | ms / outer step | mean wall, s (IPOPT twin) | median max_viol | cost, both solved | feasible particles at stop | resampled / N, cumulative median / max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grasp paired | learned | 24 | 309 | 7.55 (5.06) | 2.2e-5 | N/A (4 cells) | 6 | 0.31 / 1.00 |
| | joint space | 32 | 285 | 14.68 (1.27) | 1.7e-4 | | 0 | 0 / 0 |
| grasp native | learned | 12 | 304 | 4.00 (4.39) | 2.8e-5 | N/A (4 cells) | 13 | 0 / 0.016 |
| | joint space | 32 | 283 | 14.64 (1.20) | 2.0e-4 | | 0 | 0 / 0 |
| pose paired | learned | 12 | 264 | 3.90 (3.00) | 5.6e-5 | 6.74 | 19.5 | 0.45 / 1.00 |
| | joint space | 18 | 263 | 6.95 (0.13) | 6.4e-5 | **4.56** | 4 | 0 / 0 |
| pose native | learned | 14 | 251 | 3.95 (1.03) | 1.3e-5 | 7.73 | 39 | 0.016 / 0.125 |
| | joint space | 18 | 247 | 6.64 (0.13) | 6.4e-5 | **4.86** | 4 | 0 / 0 |

An outer step is K = 10 inner steps, so svgd's step count is NOT comparable to IPOPT's majors. The
ms per inner step, 25-31, matches the screen's 28.

**The learned arm solves 40 of 40.**

- It matches or beats its IPOPT twin on every row (+0 to +1 cell) and the record on every row
  (+0 to +4).
- It stops on its own on all 40 cells (`converged`), in 12-24 outer steps.
- Its wall clock is 3.9-7.6 s per cell, against the IPOPT twin's 1.0-5.1 s.
- Solutions sit at `max_violation` 1e-5 to 6e-5: inside the task gate, but four orders above
  IPOPT's 1e-8. The swarm is float32 and its own test is `|h| <= tol`.

**Joint space under svgd is the weak arm on grasp.**

- It solves 4 of 10 on both grasp rows against the twin's 9. The 6 failures are the same 6 cells
  under both protocols, every one stopped by the clock.
- Its multipliers sit at the `svgd_multiplier_max` clip: median `|lam|_inf` 1e4, with 10-11
  thousand clipped entries per run.
- Its failing cells end at `max_violation` 1e-4 to 0.4. Three are within 3e-4, so they are still
  descending.
- On pose it solves 10 of 10 against the twin's 6. The twin's 4 failures end at `max_violation` 0.04-0.49.
- **Joint space is identical between protocols on all 20 svgd cells**, as it must be (its
  native start is its paired start). Outer steps differ by 0-4 on clock-bound cells only.

**The kernel is doing work on the learned arm.**

- Feasible particles at stop: a median of 6-39 of 64.
- The pairwise q-spread among them is 4.1-5.6 rad, against joint space's 1.5-1.7 among 0-4.
  This is the solution set, not one point.
- Cost on cells both arms solved (pose only; grasp has 4 shared cells, so it is N/A): the learned
  solutions are dearer, 6.74 against 4.56 and 7.73 against 4.86. That is the direction the record
  shows for IPOPT on grasp. Here it is on pose.

**The pre-registered correction-box flag is CLEARED.** Under the re-registered method the learned
arm's median `|q_c|_inf` on solved cells is 0.0014-0.0084, with **0 of 40** solutions on the ±0.1
box. The 2026-10-09 solver read 0.072-0.100 and 20-60% on the box. The IPOPT twin reads ~2e-5,
so svgd still spends more correction than IPOPT does, by two orders, but nowhere near the box.

**Go / no-go, mechanically** (the reporter's own evaluation):

| check | verdict | detail |
| --- | --- | --- |
| (1) | PASS on the summaries | 0 error records, 0 missing runs. The tests are run separately. |
| (2) | PASS on all 4 rows | |
| (3) | PASS | 0 disagreements between the solver's verdict, the Drake re-check and `verify()` |
| (4) | PASS | 0 cells over cap + 1 s. The kill test is still owed. |
| (5) | not read | no `--profile` given |
| (6) | **FAIL as written** | see below |
| (7) | not decidable | the re-smoke ran no `al1` |
| (8) | manual | |

Check (6) is per cell `n_resampled / svgd_n`, CUMULATIVE over the run:

- The worst learned cell reads 1.00 on grasp paired and pose paired: 64 redraws over 12-24
  checks, about 0.04-0.08 of N per check.
- Native starts read 0.016-0.125.
- The screen's per-check reading at lr 0.3 was 0.07.
- Redraws happen only on the learned arm, and mostly from the paired start.

The check as pre-registered fails, and reading it per check is a change to a rule above, which is
Thomas's.

### Superseded: the 2026-10-08 solver, smoked 2026-10-09

**SUPERSEDED (2026-10-09).** Everything in this section and the next was measured on the
2026-10-08 solver (Adam, the Gauss-Newton correction, the float64 polish, tsvgd / admm_svgd and the
other removed pieces), not on plain AL-SVGD. It is kept as the record of that solver and is read for
nothing now; the re-smoke of the re-registered method replaces it.

Measured 2026-10-09 01:40-05:43 EDT on the laptop (RTX 3080 Ti, 20 cores), `scripts/svgd/smoke.py`
at its defaults: 88 runs, 8 rows (Panda `n6` and iiwa `n6` x grasp/pose x paired/native) x 11
columns, 10 cells x 2 arms each, 20 s cap, graphed mode, float32 swarm, `init_jitter`, one run at a
time. Full tables: `python scripts/report_svgd.py --profile
results/profiling/svgd_step_tuxedo-stellaris_20261009T045946Z.json`. The iiwa rows are PLUMBING
evidence only (the record's rung is `n4`, not local); the Panda rows pair against the record.

**Go / no-go.** (1) PASS: 0 error records, 88 of 88 summaries, every test file green. (2) PASS on all
8 rows: `al_svgd` N=64 (kernel `q` and kernel `none` alike) solves **10 of 10 learned cells on every
row**, against the IPOPT twin's 10/9/9/10 (Panda grasp p/n, pose p/n) and 7/10/6/9 (iiwa) and the
record's own 9/10/6/9 on the Panda cells. (3) PASS: 0 disagreements between the solver's verdict, the
Drake re-check and `verify()` on 1,760 svgd cells. (4) PASS: 0 cells over cap + 1 s (the kill test
is still owed). (5) PASS once compared per INNER step (the profiler times one fused step; an outer
step holds `svgd_inner_iters` = 10): every Panda block is 0.83-1.37x its profile. (6) **FAIL as
written, and the failure is informative**: `al_svgd` resamples 0.000 of N on every Panda cell and
0.19-0.45 N (cumulative over the run) on a few iiwa cells; `tsvgd` redraws 2-6 N on the Panda and up
to **15.7 N** on iiwa pose, `admm_svgd` up to 6.9 N. (7) PASS: N=1 never beats N=64 (N=64 gains 0-4
cells per arm and row, 64q+ / 1+ = 32 / 0 pooled).

**Success, learned v joint space, al_svgd N=64 kernel q, no warm-up** (IPOPT twin in brackets):
Panda grasp paired 10 v 10 [10 v 9], native 10 v 10 [9 v 9]; Panda pose paired 10 v 7 [9 v 6],
native 10 v 7 [10 v 6]; iiwa grasp paired 10 v 10 [7 v 10], native 10 v 10 [10 v 10]; iiwa pose
paired 10 v 10 [6 v 9], native 10 v 10 [9 v 9]. Joint space under svgd hits the 300-outer-step cap
on 1-2 Panda pose cells per row (so those rows carry no verdict under the cap rule) and on 4-6 cells
per row at N=1; **the cluster stage lifts `svgd_outer_iters` so only the clock binds.** Solved cells
reach `max_violation` 1e-9 to 1e-12, below IPOPT's 1e-8.

**Wall, same machine, mean over all cells, learned arm**: `al_svgd` N=64 1.0-1.8 s against the
IPOPT twin's 1.0-10.1 s (Panda grasp paired 1.8 v 4.9; iiwa grasp paired 1.5 v 10.1; iiwa pose paired
1.7 v 9.2). Per outer step 65-78 ms (6.5-7.8 per inner step, ~50-60% of it the exact collision pool),
against IPOPT's 15-25 ms per major; `al_svgd` needs 10-22 outer steps where IPOPT needs 32-195 majors.
Joint space under svgd is SLOWER than under IPOPT everywhere (0.9-6.0 s against 0.04-1.3 s): a
batched first-order method on a 7-variable problem buys nothing over a second-order one.

**CEM warm-up is inert**: over 176 arm-rows the A/B flips 0 cells on 168 of them, +2/-0 on two
`admm_svgd` learned rows and 0/-1 on four iiwa pose joint-space rows (p = 0.5-1 everywhere); outer
steps and wall move by noise. `svgd_kernel=none` against `q` is also indistinguishable on success
(identical counts on every row); the kernel's effect is on the solution set, not on whether one is
found.

**tsvgd** solves 10 of 10 learned cells on every row too, but from the paired start it runs to the
clock on 6-10 cells per row (196-224 outer steps, 15-18.5 s) while already feasible: its stop rule
does not fire in tangent mode, so its wall column is the cap. **admm_svgd** is the weak method on the
learned arm from the paired start (8/10, 7/10, 9/10, **5/10**), clock-bound, and its joint-space
degenerate form sits at the step cap on 9-10 cells per iiwa row.

**The pre-registered correction-box flag is CONFIRMED and is the main open item.** Under `al_svgd`
N=64 the learned arm's median `|q_c|_inf` is 0.072-0.100 with 20-60% of solutions ON the +-0.1 box
(`correction_binding`), against the IPOPT twin's 2e-5 and 0% on the same cells; the median reported
cost over each arm's own solved cells is correspondingly higher (Panda grasp paired 7.99 against
IPOPT's 4.22; pose paired 7.56 against 5.08; not mutual-cell medians, so indicative only). `tsvgd`
sits at 0.005-0.099 with 0-40% binding. The learned svgd arm is finding feasible points by spending
the correction, i.e. partly as a reparameterised joint-space arm. The likely mechanism is the
minimum-norm Gauss-Newton correction, which is cheapest in `q_c` under the unscaled metric. The
candidate fix is solver-internal and formulation-agnostic (weight the GN metric by the program's own
cost Hessian, so the correction direction pays its `w_c = 10`), **not applied; Thomas's call**, and
the smoke's numbers stand as the pre-fix measurement.

## Selected variant

**SUPERSEDED (2026-10-09)** with the section above: the selection below was of the removed solver;
a new selection is written after the re-smoke, before any cluster manifest.

Written 2026-10-09 06:05 EDT, before any manifest exists; **pending Thomas's gate** on the
correction-box item above, which may change the fielded solver and therefore void this selection.

- **Primary: `al_svgd`, N = 64, `svgd_kernel=q`, `svgd_warmup=none`, `init_jitter`, float32 swarm +
  float64 polish, graphed** -- the only method that passes (2), (3), (5), (6) on the Panda and stops
  on its own (`feasible_stall` on 100% of cells).
- **Controls in the stage**: `svgd_kernel=none` at N = 64 (the no-interaction control; the smoke
  cannot separate it from `q` on success, so the cluster's 480 cells decide), and N = 1 (the
  single-particle control). N = 256 only on the survivors, as the ladder's upper rung.
- **Dropped from the first cluster manifest**: the CEM warm-up on every row (inert to the cell on
  176 arm-rows; one A/B column on the primary variant is kept so the A/B is a 480-cell statement, not
  a 10-cell one); `admm_svgd` (weak and clock-bound on the learned arm, degenerate on joint space);
  `tsvgd` is kept as the second candidate only if its stop rule is fixed first, since a method that
  polishes to the cap cannot have a wall column.
- **Settings for the stage**: `svgd_outer_iters` lifted so only the 180 s clock binds (cap rule);
  `PROCS=2`; the comparison columns are the record (no IPOPT re-run), paired by `(target, guess)`
  after the `grid_hash` check.
