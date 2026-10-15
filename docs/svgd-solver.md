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

   `svgd_metric_temperature = on` puts `H_i / T` in place of `H_i` under either metric (options
   table); `off` is the step above.

   Then `y_i <- clamp_B(y_i + dy_i)`; the clamp's distance in y is recorded (`bound_clip`). No
   momentum, no adaptation. The median ratio `|repulsion| / |drive|` within phi is recorded
   per solve (`repulsion_ratio_median`): the kernel's actual share of the step.
5. **Kernel**: RBF on the configuration q (flow output + correction on the learned arm, q itself on
   joint space), median bandwidth `h = med^2 / log N` floored at `svgd_bandwidth_floor`; the
   repulsion is pulled back to y through the flow's VJP. `svgd_kernel = none` drops both kernel
   terms (`phi_i = -(1/T) grad_y L_i`, N independent projected gradient descents, batched).
   That is the fielded `q`; the other kernel spaces (`x`, `z`, `c`, `cz`, `qz`, `pose`) are
   defined in the options table.
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
   is redrawn from the arm's NATIVE start distribution with zero multipliers (`n_resampled`). That
   is `svgd_resample = native`, the fielded mode. `drop` removes such a particle from the swarm
   instead (`n_dropped`). A dropped particle is frozen. It is left out of every other particle's
   kernel average, repulsion and median bandwidth, the average being over the `n_live` live
   particles, and it is never feasible or selected. `keep` lets a runaway keep stepping, and drops a
   non-finite particle as `drop` does, since it cannot step without poisoning the kernel average.
   If every particle is dropped, the swarm stops with `all_dead` and the cell is scored infeasible.
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
zero-centred, no arm-specific kernel in the fielded configuration -- `svgd_kernel = z / c / cz /
qz` are screen arms; the only structure used is that every arm has a q and that every variable
block has a region). Recorded future work, not current scope: using
formulation-specific information rigorously with SVGD. **Exact Drake collision, no proxy**: the
collision row is the program's own Drake `MinimumDistanceLowerBoundConstraint`, evaluated IN
PROCESS (Thomas, 2026-10-09: no process pool): a Python loop over the N particles calling it on
AutoDiffXd (`Eval(InitializeAutoDiff(q))`, exactly the program's binding), returning value and
gradient. It is serial and GIL-bound (pydrake holds the GIL in `Eval`). The batched C++
clearance-with-Jacobians call that releases it now exists on a Drake branch and is the
`clearances` backend ("Batched clearance row", below); the default stays this one. Its seconds
are reported as the collision row's share of the wall.

### The options, one table

| option | default | part of the method |
| --- | --- | --- |
| `svgd_method` | `al_svgd` | the method (one value) |
| `svgd_n` | 64 | N particles (1 = the single-particle control) |
| `svgd_dtype` | `float32` | swarm precision (`float64` a control) |
| `svgd_seed` | 0 | particle draws, mixed with a CRC of the initial guess |
| `svgd_paired_init` | `native` (fielded 2026-10-12; was `jitter`) | 11: `jitter` or `native`; particle 0 is the shared guess exactly under either |
| `svgd_jitter` | 0.1 | 11: jitter sigma in the normalised coordinates |
| `svgd_temperature` | **REMOVED 2026-10-14** (R22; T = 1, the only value ever fielded) | 2: T, fixed |
| `svgd_lr` | 0.3 | 4: a fraction of the metric's step: `svgd_lr (H + delta I)^-1 phi` (gn) or `svgd_lr / \|\|H\|\|_F` (identity). Chosen 2026-10-10 by the screen at rho 1e3, delta 10 on 6 grasp + 6 pose Panda cells, most feasible cells with ties to the larger lr: lr 0.3 solved 24/24 (learned 12/12, joint space 12/12) resampling 0.07 of N per check, lr 1.0 solved 22/24 resampling 0.12 |
| `svgd_metric` | `gn` | 4: `gn` (Stein variational Newton) or `identity` |
| `svgd_metric_temperature` | **REMOVED 2026-10-14** with `svgd_temperature` (R22: T inert at the pin; the step is the T = 1 form below, bit for bit) -- was `on` (fielded 2026-10-12; `off` before) | 4: `off`: the metric is H, so the drive `-grad L / T` is not matched by it and the Newton step scales as 1/T. `on`: the metric is H / T, `dy_i = svgd_lr (H_i / T + delta I)^-1 phi_i`, so T sets only the repulsion-to-drive ratio; under `identity` the step bound becomes `\|\|H_i\|\|_F / T` (the analogue, not refused). Both are swept and neither is "the" definition (Thomas, 2026-10-11). `on` at T = 1 is `off` bit for bit |
| `svgd_gn_lm` | 10 | 4: delta, the Levenberg damping of the GN metric. Chosen 2026-10-10 by the most feasible cells (learned + joint space) on 6 grasp + 6 pose Panda cells, ties to the larger: delta 1e-2 / 1 / 10 gave 2 / 3 / 6 (1e-4 diverged); then svgd_lr 1.0 / 0.3 / 0.1 at delta 10 gave 6 / 0 / 0 -- all at rho 10, and the 0.3 / 0.1 cells were mostly `error` records from the since-fixed eigvalsh crash, so that lr reading is void; `svgd_lr` was chosen later by the screen at rho 1e3 (its row) |
| `svgd_row_units` | `natural` | 1: `natural` or `tolerance` (the control) |
| `svgd_row_length_scale` | 1.0 | 1: metres per radian on the orientation-type rows |
| `svgd_kernel` | `cz` (fielded 2026-10-12; was `q`) | 5: the RBF's space, median bandwidth per factor (floored by `svgd_bandwidth_floor` in that factor's units). `q`: the configuration, repulsion pulled back to y through the flow's VJP. `x`: all of y = x / s, one bandwidth, repulsion directly in y (on joint space: q over its half-range). `z` / `c`: the latent / conditioning-pose block of y alone, repulsion on that block only. `cz`: the product of the `c` and `z` RBFs, each its own bandwidth (an anisotropic RBF on [c; z]); repulsion the sum of the two blocks' terms. `qz`: the product of the `q` RBF (pulled back) and the `z` RBF (direct), each its own bandwidth. `pose`: the task frame's (`program.frame`) [position; quaternion wxyz, w >= 0], pulled back through d pose / d q -- `Jp`, and `1/2 [-v^T; w I - [v]x] Jw` for the quaternion -- then the VJP. `none`: no interaction. `z` / `c` / `cz` / `qz` are learned-only, refused on joint space at solver construction. Recorded per cell: `kernel`, `bandwidths` at the last step, `feasible_spread_kernel` (the feasible particles' median pairwise distance in the kernel's own space and units; a 2-list for `cz` / `qz`). `q` reproduces the previous step bit for bit (a 30-step solve, both tasks x both arms x eager / compiled / graphed, against arrays saved at 5f15f55, 2026-10-11) |
| `svgd_bandwidth_floor` | 0.05 | 5: floor on the median bandwidth |
| `svgd_constraint_inside_kernel` | False | 12: the literal form, an A/B |
| `svgd_rho` | 1e5 (fielded 2026-10-12 with `cz`, R14-R19; was 1000) | 2, 4: the penalty -- one scalar, fixed, shared. The 1000 was chosen 2026-10-10 on 6 grasp + 6 pose Panda cells at delta 10, lr 1: rho 10 / 1e3 / 1e4 / 1e5 solved 5 / 12 / 12 / 12 learned and 0 / 10 / 9 / 11 joint space; 1e3 is the smallest that works and resamples least (0.08-0.12 of N per check against 0.17-0.21 at 1e5). With the Newton metric a large rho no longer costs conditioning. At rho = 10 the repulsion was 0.97 of the step and swamped the drive (N = 1 solved grasp 6/6 where N = 64 solved 0/6) |
| `svgd_dual_lr` | None (= `svgd_rho`) | 6: alpha, the dual-ascent step; 0 = pure quadratic penalty. The default is pending the alpha ladder |
| `svgd_multiplier_max` | 1e4 | 6: multiplier clip |
| `svgd_inner_iters` | 10 | 6, 7, 8: K, the cadence of the dual step, resampling and the stop rule |
| `svgd_outer_iters` | 300 | 8: the step cap (`max_iter`, when set, also binds) |
| `svgd_resample_q_max` | 10.0 | 7: runaway threshold, rad |
| `svgd_resample` | `native` | 7: a runaway or non-finite particle is redrawn (`native`), dropped (`drop`), or, if a finite runaway, kept stepping (`keep`, which still drops a non-finite particle). Dropping is a static-shape live weight in the AL state. Round R11 |
| `svgd_stop_patience` | 5 | 8 |
| `svgd_stop_rel` | 1e-3 | 8 |
| `svgd_recheck_topk` | 10 | 9; the same float64 re-check gates a stall stop (the gate-edge rule, below RULINGS 2026-10-12) |
| `svgd_time_reserve` | 0.1 | 8: fraction of the cap kept for selection and re-check |
| `svgd_warmup`, `svgd_warmup_iters`, `svgd_warmup_elite` | `none`, 10, 0.1 | 11: the CEM phase |
| `svgd_compile`, `svgd_cuda_graph` | False, False | execution: the split step eager / compiled / graphed |
| `svgd_collision_backend` | `autodiff` | execution: how the exact collision row is evaluated -- `autodiff` (the program's constraint on AutoDiffXd, one configuration at a time) or `clearances` (the same functional from one batched `CalcRobotClearances` call; needs Drake branch `calc-robot-clearances`, refused elsewhere). See "Batched clearance row" |

**The step size.** Under `gn` the metric carries the scale, so `svgd_lr = 1` would be the full Newton step
on the GN model of the AL; the fielded 0.3 is a damped fraction of it (screen, option table). Under `identity`, `svgd_lr / ||H_i||_F <= svgd_lr / lambda_max(H_i)`
is a per-particle Lipschitz step. (The earlier fixed step, `1e-10 / rho` on rows divided by tol,
was forced by the `rho / tol^2` curvature: pose-row curvature ~1e10 against objective curvature
1e-4..10, which froze the self-motion and left the order-1 repulsion numerically dead. Its
measurements are in git, a9758e1.)

**Memory.** No worker processes: the collision row runs in the solver's own process on the
program's own scene. Local runs still go under `systemd-run --user --scope -p MemoryMax=...
-p MemorySwapMax=0` (the smoke driver does this by default).

### Batched clearance row (`svgd_collision_backend=clearances`), measured 2026-10-11

Thomas's `CollisionChecker::CalcRobotClearances` (Drake branch `calc-robot-clearances`, commit
15a2c97: batched `CalcRobotClearance`, GIL released, parallel over the checker's per-thread
contexts) is **temporarily installed at `/home/tommy/opt/rlg/drake-build-crc`** and used by
prepending `/home/tommy/opt/rlg/drake-build-crc/lib/python3.12/site-packages` to `PYTHONPATH`.
**Neither this repo's default pydrake nor the cluster's Drake has it, so the default stays
`autodiff`**, and `clearances` is refused at options construction where the binding is missing.

`ClearanceEvaluator` (`src/svgd/collision_backend.py`, whose docstring has the formula) is the same
functional: one `CalcRobotClearances(Q, bound + influence_distance_offset)` on an unpadded checker
of the solve scene, then `MinimumValueLowerBoundConstraint`'s aggregation replicated in numpy
float64 (quadratically smoothed hinge on `(d - v_infl) / (v_infl - bound)` for `d < v_infl`, scaled
by `1 / phi(-1)`, `SoftOverMax` at alpha 100 padded with zeros to the scene's candidate count; the
gradient through the checker's pair Jacobians). The pair sets are checked at construction (302
pose, 342 grasp candidates on the Panda, identical). `tests/test_collision_clearances.py`, 64
uniform draws per Panda scene (37 penetrating) plus NaN rows: **values bitwise equal** to the
program's binding, **gradients within 4.1e-14 of their largest entry** (the two derivative paths,
`CalcJacobianTranslationalVelocity` with the query's normal against `CalcJacobianPositionVector`
with the witness-point difference, differ in roundoff), bitwise independent of thread count and of
the batch.

Row cost on this laptop (20 threads; `scripts/svgd/profile_clearances.py`, uniform draws, us per
configuration; the cluster's cores are ~2x slower per configuration):

| scene | N | autodiff | clearances, 20 threads | clearances, 1 thread |
| --- | --- | --- | --- | --- |
| pose | 64 / 256 / 1024 | 171 / 178 / 172 | 25.0 / 18.3 / 17.3 | 143 / 148 / 141 |
| grasp | 64 / 256 / 1024 | 194 / 178 / 179 | 24.9 / 18.7 / 17.6 | 158 / 146 / 146 |

7-10x at 20 threads; single-threaded it is 1.2x (no per-configuration AutoDiffXd). End to end,
Panda, paired, learned arm only, the smoke's 10 cells, fielded defaults graphed, 60 s
(`probe_CLR_<backend>_<task>`): **success 10/10 on both backends, both tasks**; 18 of 20 cells
take identical outer steps with identical `max_violation`; the other two are pose 13:0, where
autodiff stopped at the wall clock (235 steps, 54 s) and clearances at the 300-step cap (19 s),
same selected particle and cost to 1e-6, and pose 13:1, same 17 steps but a slightly different
returned particle (`max_violation` 1.8e-5 against 6.8e-6), float32 trajectories separated by
the roundoff above. Median wall per cell 3.68 -> 1.04 s (pose) and 7.61 -> 1.65 s (grasp); the
collision share of the wall 0.88 -> 0.26 and 0.91 -> 0.38. What remains of the row is mostly the
per-result pybind conversion and the numpy aggregation (the Drake call is ~75-85% of it at N =
64).

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
| (4) | PASS | 0 cells over cap + 1 s. Kill test PASSED 2026-10-11 (`tests/test_svgd_result.py::test_svgd_kill_keeps_the_iterate`): a solve killed on its 4th outer check leaves `last_iterate` at the swarm's best particle and the harness scores it into `recovered_*`. |
| (5) | not read | no `--profile` given |
| (6) | **FAIL as written; read as GO for now** | see below. Thomas, 2026-10-11: the per-check / per-run wording is DEFERRED; err on the side of go until R11 (`svgd_resample` drop / keep) says what the quantity should be. |
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
passed on 2026-10-11, see the table above). (5) PASS once compared per INNER step (the profiler times one fused step; an outer
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

## Stage SVGD_R1: the fielded method on the record's four Panda rows, 480 cells, 180 s

**Measured 2026-10-10 on SuperCloud (jobs 5884549-52, svgd tip adf534c restaged; PROCS=2, no MPS,
graphed), read by `scripts/report_svgd.py --stage SVGD --variants kq`.** The `kq` variant is the
fielded configuration exactly as "Smoke results" registered it (natural units, Stein variational
Newton with the Levenberg shift 10, rho = 1000, lr = 0.3, alpha = rho, K = 10, N = 64, T = 1, kernel on
q). The IPOPT twin is stage SVGD_R2: the same items under IPOPT at paper conditions, on the same grids.
Framing, per Thomas (2026-10-10): the learned arm is the question, joint space under svgd an ablation.

**A first submission on the stale staged code (42a5893, jobs 5882597-5882601) is VOID and was moved
aside on the cluster**: that code predates the `lambda_max` fix (7dec630), so the eigvalsh diagnostic
raised inside the solve and the learned arm was scored `error` on 85-90% of grasp cells and 14-23% of
pose cells. The re-run below has 0 error cells.

**Lead: learned under svgd vs learned under IPOPT, same cells, exact McNemar.**

| row | learned svgd | learned IPOPT | svgd-only / IPOPT-only | p | verdict |
| --- | --- | --- | --- | --- | --- |
| grasp paired | **480** / 480 | 479 | 1 / 0 | 1 | tie |
| grasp native | **480** / 480 | 480 | 0 / 0 | 1 | tie |
| pose paired | **480** / 480 | 395 | 85 / 0 | 5.2e-26 | **svgd** |
| pose native | **480** / 480 | 458 | 22 / 0 | 4.8e-07 | **svgd** |

The learned arm solves every cell on every row. The gain is on pose, where IPOPT loses 85 and 22
cells from a bad start and the swarm loses none; on grasp both are at the ceiling.

**What the swarm rescued (read from the R2 twin's records).** Every one of the 85 + 22 pose cells
and the 1 grasp cell that the learned arm solves under svgd and not under IPOPT is an IPOPT
`constraint` failure: converged, at a median 118 majors and 1.5-1.7 s, to a locally infeasible point
with `max_violation` 0.25 (paired) / 0.06 (native), `timed_out` 0, no runaway (`|q|_inf` < 1000 on
all 108). So the gain is not budget: a single interior-point trajectory from that start ends in the
wrong basin. **It is not the swarm's coverage either -- R3 below shows N = 1 rescues the same 85 + 22
cells**; what differs is the solver's dynamics from that start (the augmented-Lagrangian Newton step,
and on the paired rows the redraw of a particle whose configuration blows up). The swarm spends a
median 5.7 s on those cells against IPOPT's 1.6 s to fail.

**What it costs** (mean wall over all cells, each clamped at the cap; svgd steps are OUTER steps of 10
inner steps, not comparable to majors):

| row | svgd wall | IPOPT wall | svgd outer steps (median) | ms / outer step | viol svgd | viol IPOPT | cost svgd / IPOPT (shared cells) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 12.26 s | 5.01 s | 24 | 502 | 4.7e-05 | 1.5e-08 | 8.38 / 4.14 (338) |
| grasp native | 7.14 s | 3.00 s | 13 | 492 | 4.4e-05 | 1.2e-08 | 8.41 / 4.51 (338) |
| pose paired | 6.56 s | 2.02 s | 13 | 429 | 3.5e-05 | 1.3e-08 | 5.61 / 6.40 (451 / 170) |
| pose native | 6.97 s | 0.77 s | 13 | 421 | 1.7e-05 | 9.0e-09 | 7.50 / 6.71 (451 / 189) |

- **2.4-9x the wall of IPOPT**, and the step is the serial Drake collision row: 88-91% of the
  learned arm's wall (94-95% of joint space's), ~44 of ~49 ms per inner step at N = 64 (64 configs
  at ~690 us each on the node's cores). With Drake's batched CPU-parallel `CalcRobotClearances` the
  step is ~6-7 ms (~7x), which puts the learned arm at or below IPOPT's wall on every row; a GPU
  collision row would add only ~1.2-1.4x more because the flow then dominates (ruled not worth
  building, 2026-10-10).
- Solutions sit at a median violation of 2e-5 to 5e-5 against IPOPT's 1e-8, inside the 1e-4 / 1e-3
  gates: the swarm stops at the first feasible check, IPOPT polishes.
- The cost column is the shared objective on the cells both arms of THAT run solved (so the two
  runs' cell sets differ); on grasp the returned particle is ~2x IPOPT's learned cost, on pose level.
- Stop reasons: `converged` on 1919 of 1920 learned cells, one `wall_clock` (pose paired); 0 at the
  iteration budget; `recovered_*` on 0 cells. Feasible particles at stop, median: 6 / 13 / 21 / 31 of
  64 (grasp p/n, pose p/n), q-spread among them 4.1-5.9 rad.
- Multiplier clips: 252k (grasp paired), 9.6k, 367k, 38k updates at the 1e4 clip on the learned
  arm. Which row group hits it is not recorded (phase-2: per-group clip counters).

**Step profile at the fielded configuration (laptop, RTX 3080 Ti, graphed float32, 20 timed steps;
`scripts/svgd/profile_step.py --dtypes float32 --modes graphed`, 2026-10-11;
`results/profiling/svgd_step_fielded_{pose,mug}_laptop_20261011.json`).** ms per INNER step, learned
arm, pose / grasp scene, with the serial collision row's share:

| N | 1 | 16 | 64 | 256 | 1024 |
| --- | --- | --- | --- | --- | --- |
| ms / step (pose / grasp) | 2.1 / 2.1 | 7.9 / 6.4 | 21.9 / 23.5 | 90.6 / 84.0 | 355 / 335 |
| collision share | 0.15 / 0.17 | 0.84 / 0.80 | 0.92 / 0.92 | 0.96 / 0.96 | 0.94 / 0.94 |
| collision us / config | 308 / 360 | 414 / 318 | 315 / 339 | 339 / 313 | 326 / 306 |

Everything but the collision row costs ~1.8 ms at N = 1, ~2-4 ms at N = 64 and ~21 ms at N = 1024
on this GPU; the row is a flat ~310-340 us per configuration, so the step is linear in N from
N = 16 up. Joint space is 0.8 / 16.6 / 267 ms at N = 1 / 64 / 1024 (pose), its row cheaper per
configuration (~250 us) because the plant has no gripper-to-flow frame work to do. **Go/no-go (5)
("ms/step within 2x of `profile_step`") was written for a same-machine comparison and is read as
such**: R1's V100 nodes run ~49 ms per inner step at N = 64 (490 ms per outer step of 10), 2.2x this
laptop, entirely the serial row at ~690 against ~330 us per configuration on slower cores -- a
host difference, not a solver one, and consistent with the 88-91% collision share measured in R1
itself. Timing is never compared across machines (CLAUDE.md); the cluster's own share is the check.

**The joint-space ablation** (the swarm without the network; its own McNemar is against the
joint-space IPOPT twin, and it is NOT a baseline): grasp 338 / 480 on both protocols (143 cells at the
clock; IPOPT 449 / 448, p ~ 1e-22 to IPOPT), pose 451 / 480 on both (IPOPT 200, p = 6e-68 to svgd).
Its multipliers pin at the clip (median |lam|_inf 1e4 on grasp, 8.7M clipped updates) and a
grasp cell takes a median 660 inner steps at 468 ms per outer step. Joint space is bit-identical
between the two protocols, as it must be (its native start is a random configuration).

**Cap check**: learned 0 cells at the step budget on every row, 1 at the clock; the joint-space grasp
rows are clock-bound (143 of 480) and carry the flag "a result at the fielded clock".

### R1K: the kernel off (`knone`), measured 2026-10-11

**Jobs 5884553-56 on the restaged tip, the same four rows, both arms, `svgd_kernel=none` and nothing
else changed; read by `scripts/report_svgd.py --stage SVGD --variants kq,knone`.** 0 error cells.
The round took ~10 h on four nodes against R1's ~4 h, because the joint-space arm runs to the clock on
most grasp cells with the kernel off (which is why R3 onward drop that arm).

**Lead: the kernel buys the learned arm no success.** `knone` solves 480 / 480 on every row, +0 / -0
against `kq` cell for cell, and carries the same verdicts against the IPOPT twin (tie, tie, svgd,
svgd; the same 85 + 22 + 1 rescued cells).

**What it changes is where the swarm stops.** On the cells all three solved (learned arm; IPOPT twin,
`kq`, `knone`; median over those cells):

| row | n | cost IPOPT / kq / knone | viol kq / knone | wall kq / knone | outer steps kq / knone | knone cells at the clock | knone cost better / worse than kq |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 479 | 4.14 / 8.32 / **3.15** | 4.7e-05 / 2.0e-07 | 12.0 / 135.6 s | 24 / 281 | 215 | 475 / 4 |
| grasp native | 480 | 4.52 / 8.58 / **3.25** | 4.4e-05 / 2.0e-07 | 6.4 / 93.7 s | 13 / 192 | 123 | 472 / 8 |
| pose paired | 395 | 6.44 / 5.62 / **4.45** | 3.5e-05 / 6.2e-07 | 5.8 / 7.0 s | 13 / 15 | 0 | 326 / 33 |
| pose native | 458 | 6.76 / 7.50 / **4.99** | 1.6e-05 / 6.2e-07 | 5.7 / 20.0 s | 13 / 46 | 0 | 381 / 67 |

- With the kernel off the swarm keeps descending: the stop rule (the best feasible objective
  unchanged over 5 checks) fires late on pose and on grasp mostly never (215 and 123 of 480 cells run
  to the 180 s clock), and the returned particle is cheaper than the kernel-on one on 472-475 of 480
  grasp cells and cheaper than IPOPT's (3.2 against 4.1-4.5 on grasp, 4.5-5.0 against 6.4-6.8 on
  pose), at a violation two orders tighter (2e-7 against 5e-5). All 64 particles are feasible at stop
  on grasp (48-62 on pose) against 6-31 with the kernel on; the q-spread among them is 2.4-4.5 rad
  against 4.1-5.9.
- With the kernel on the best particle stalls within 13-24 outer steps at ~2x IPOPT's cost. **So
  the grasp cost premium in R1's table is where the kernel-on swarm stops, not a limit of the
  formulation under this solver**: the same particles, left to descend without repulsion, pass
  IPOPT's objective. Whether that is the repulsion holding the swarm at a worse objective or the
  stall rule reading a repulsion-perturbed best particle as stalled is not separated here (K = 100
  in R6 and lr in R4 / R6 bear on it).
- Wall: the kernel-on swarm is 10-15x faster to stop on grasp and 1.2-3.5x on pose, for the same
  success and a worse objective. Both are the same ~480 ms per outer step.

**The joint-space ablation: the kernel is what gets the swarm without the network to feasibility on
grasp.** Kernel off, grasp falls 338 -> **87** / 480 on both protocols (+6 / -257 against kernel on,
p = 6e-68; 393 cells at the clock), with 0 feasible particles at stop on the median cell, a q-spread
of 0.002 rad (the 64 particles collapse to one point), violation 0.4 and every multiplier at the
1e4 clip (22M clipped updates). Pose is a tie, 457 against 451 (+16 / -10, p = 0.33). So on the
learned arm the kernel is inert for success and on joint-space grasp it is worth 251 net cells: the
repulsion keeps the swarm from collapsing into the same infeasible point, which the learned arm's
start distribution (64 latents) does on its own.

**Cap check**: learned `knone` is at the clock on 215 / 123 grasp cells, flagged "a result at the
fielded clock"; since every cell is solved the flag touches no verdict. 0 cells at the step budget on
any row. `recovered_*` on 0 cells.

### R3: N = 1 and N = 256 (`n1`, `n256`), learned arm only, measured 2026-10-11

**Jobs 5886385-88, 128 items, 0 error cells; `scripts/report_svgd.py --stage SVGD --variants
kq,knone,n1,n256`.** Each differs from `kq` in `svgd_n` alone.

**Lead: N buys the learned arm no success.** N = 1 solves 480 / 480 / 478 / 480 (grasp paired,
grasp native, pose paired, pose native), N = 256 480 on every row; against the IPOPT twin both carry
`kq`'s verdicts (tie, tie, svgd, svgd), and **N = 1 rescues the same 85 + 22 pose cells** that N = 64
does. The two N = 1 losses on pose paired are `constraint` stops at the clock (violation 1e-3 and
1.2e-2 at 162 s, the cap less the polish reserve). Go/no-go (7), "N = 1 does not beat N = 64", holds
as a tie (+0 / -2 on pose paired, +0 / -0 elsewhere).

**So the pre-registered reading of R1 -- that the swarm rescues by basin coverage -- is refuted.**
A single particle under the same augmented-Lagrangian Newton dynamics rescues the same cells. Two
mechanisms are in play and the records separate them partly:

- **The redraw.** A particle whose configuration passes 10 rad (or whose rows go non-finite) is
  redrawn from the arm's native start distribution and counted (`n_resampled`). At N = 1 that redraw
  is a restart of the one trajectory. On the paired rows it fires on most cells: the median N = 1
  cell is redrawn once on grasp paired (391 of 480 cells) and pose paired (353 of 480), against 5 and
  35 of 480 native. **Of the 85 pose-paired cells N = 1 rescues, 68 passed through a redraw; of the
  22 pose-native rescues, 9 did.** So 17 + 13 rescues are the Newton dynamics from the given start
  alone, and the rest are a redraw from the native draw after the paired start blew up. At N = 64 the
  median paired cell redraws 21 (grasp) / 48 (pose) of its 64 particles over the run. **Ruled
  (Thomas, 2026-10-11): a redraw is the swarm's own dynamics**, distinct from a single-trajectory
  solver's RESTART and from the MULTISTART protocol of the time-budget column; the one-solve column
  may run the fielded `native` mode, and R11 (`svgd_resample` drop / keep) characterises what the
  redraw buys rather than repairing a fairness defect. Terminology from here on (Thomas's choice): RESAMPLE
  (a particle replaced inside a running swarm, the Monte Carlo sense), restart (a solver relaunched
  from a new start), multistart (the time-budget protocol).
- **The step.** On the native rows, where almost nothing is redrawn, N = 1 still ties N = 64 and beats
  IPOPT on pose (480 against 458, 13 of the 22 rescues without any redraw): the damped
  Gauss-Newton step on the augmented Lagrangian reaches a feasible point from starts where IPOPT's
  interior-point trajectory converges to a locally infeasible one.

**Cost, violation and wall on the cells all four solved** (learned arm, median):

| row | n | cost IPOPT / N=64 / N=1 / N=256 | viol N=64 / N=1 / N=256 | wall N=64 / N=1 / N=256 | outer steps N=1 / N=256 (N=64) |
| --- | --- | --- | --- | --- | --- |
| grasp paired | 479 | 4.14 / 8.32 / 5.63 / 6.33 | 4.7e-05 / 2.2e-07 / 4.6e-05 | 12.0 / 19.0 / 39.4 s | 406 / 21 (24) |
| grasp native | 480 | 4.52 / 8.58 / 5.60 / 7.22 | 4.4e-05 / 2.1e-07 / 4.0e-05 | 6.4 / 17.4 / 23.2 s | 376 / 12 (13) |
| pose paired | 393 | 6.44 / 5.62 / 7.46 / 4.87 | 3.5e-05 / 6.4e-07 / 2.8e-05 | 5.8 / 6.7 / 20.6 s | 132 / 13 (13) |
| pose native | 458 | 6.76 / 7.50 / 6.93 / 5.99 | 1.6e-05 / 6.9e-07 / 1.5e-05 | 5.7 / 3.3 / 22.4 s | 69 / 14 (13) |

- N = 1 runs 70-400 outer steps at ~2 ms per inner step (collision share 0.14-0.17) and stops
  `converged` on 1918 of 1920 cells, at a violation of 2e-7 (the single particle polishes, as the
  kernel-off swarm did) and a cost between IPOPT's and N = 64's on grasp. Its wall is 1.5-3x N = 64's
  on grasp and 0.6-1.2x on pose.
- N = 256 stops in the same 12-21 outer steps as N = 64 at 3-4x the wall (~1.9 s per outer step,
  collision share 0.95-0.96, the serial row again), returns 25-128 feasible particles, and its cost
  is better than N = 64's on every row (6.3-7.2 against 8.3-8.6 on grasp, 4.9-6.0 against 5.6-7.5 on
  pose) -- more particles, a better best.
- Multiplier clips at N = 1: 7.9k / 94 / 7.2k / 883 (grasp p / n, pose p / n), so the clip is
  reached on the paired rows even by one particle.

**Cap check**: 0 cells at the step budget on every N = 1 and N = 256 row; 2 N = 1 cells at the clock
(pose paired). The flag is not raised.

### R4: rho = 1e4 and lr = 1.0 (`rho1e4`, `lr1`), learned arm only, measured 2026-10-11

**Jobs 5886389 / 90 / 92 / 96, 192 items, 0 error cells; `--variants kq,knone,n1,n256,rho1e4,lr1`.**
Each differs from `kq` in one setting (`svgd_rho` 1000 -> 10000; `svgd_lr` 0.3 -> 1.0).

**Lead: both are inert on success.** 480 / 480 on every row for both, +0 / -0 against `kq` cell for
cell, the same verdicts against the IPOPT twin and the same 85 + 22 rescued cells. That is now six
variants (kernel off, N = 1, N = 256, rho x10, lr x3.3, and the primary) with identical success on
1920 cells: **on this grid the learned arm's success under the augmented-Lagrangian swarm is
insensitive to every setting varied so far**, and the two pose rows are won against IPOPT by all of
them.

**What they change** (learned arm; cost / violation / wall are medians over the cells all four
solved; redraws, clips and feasible particles over all 480 cells):

| row | cost IPOPT / kq / rho1e4 / lr1 | wall kq / rho1e4 / lr1 | outer steps kq / rho1e4 / lr1 | median particles redrawn per cell (of 64) kq / rho1e4 / lr1 | clipped updates kq / rho1e4 / lr1 | feasible at stop kq / rho1e4 / lr1 |
| --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 4.14 / 8.32 / **6.23** / 9.70 | 12.0 / 7.8 / 7.1 s | 24 / 16 / 14 | 21 / 61 / 58 | 252k / 732k / 517k | 6 / 14 / 5 |
| grasp native | 4.52 / 8.58 / 7.69 / 10.18 | 6.4 / 7.0 / 7.3 s | 13 / 14 / 14 | 0 / 10 / 42 | 9.6k / 154k / 381k | 13 / 17 / 4 |
| pose paired | 6.44 / 5.62 / **5.17** / 5.26 | 5.8 / 5.4 / 4.2 s | 13 / 12 / 9 | 48 / 65 / 75 | 367k / 821k / 725k | 21 / 27 / 41 |
| pose native | 6.76 / 7.50 / 6.70 / **6.15** | 5.7 / 5.8 / 4.7 s | 13 / 13 / 11 | 3 / 7 / 21 | 38k / 147k / 223k | 31 / 30 / 45 |

- **rho = 1e4** stops earlier on grasp paired (16 against 24 outer steps, 7.8 against 12.0 s) at a
  better objective (6.2 against 8.3) with more feasible particles (14 against 6), and is level
  elsewhere. It triples the redraws (the median paired cell redraws 61 of its 64 particles over the
  run against 21) and the clipped updates: the stiffer penalty throws more particles past 10 rad
  and pins more multipliers at the 1e4 clip. Violations are unchanged (1.5e-5 to 4.3e-5).
- **lr = 1.0** is the setting the screen demoted for churn, and the 480-cell rows agree on the
  mechanism without agreeing on the cost: it stops fastest (9-14 outer steps, 4.2-7.3 s), with the
  most redraws (42 of 64 on grasp native against 0 at lr = 0.3) and the most clipped updates, and
  returns the worst grasp objective (9.7 / 10.2 against 8.3 / 8.6) but the best pose one
  (5.3 / 6.2 against 5.6 / 7.5), with 41-45 feasible particles on pose against 21-31.
- Neither moves a verdict, a violation band or the collision share (0.89-0.91). The step is the
  same ~480 ms per outer step.

**Cap check**: 0 cells at the step budget, 0 at the clock, on all eight rows.

### R5: N = 16 and T = 10 (`n16`, `T10`), learned arm only, measured 2026-10-11

**Jobs 5886397-400, 160 items, 0 error cells; `--variants ...,n16,T10`.** Each differs from `kq` in
one setting (`svgd_n` 64 -> 16; `svgd_temperature` 1 -> 10).

**Lead: both inert on success.** N = 16 solves 480 / 479 / 480 / 480, T = 10 480 / 480 / 479 / 480;
each is +0 / -1 against `kq` on one row, and both carry `kq`'s verdicts against the IPOPT twin.
**The two lost cells are the first go/no-go (3) disagreements in 3840 cells, and the reporter
flags them as BUG lines**: on each the swarm's float32 rows read its one feasible particle as
inside the 1e-4 gate (`solver_feasible`), the float64 Drake re-check read 1.0030e-4 and 1.0005e-4
(`drake_feasible` False), and the cell returned that particle scored infeasible (`feasible_relaxed`
True at the 1e-3 gate). The mechanism is the gate rule of CLAUDE.md -- the stop fires at the gate
the particle is pinned to, so the last ulp decides -- not a code defect; the fix candidates are a
margin inside the gate for the stop rule, or the float64 polish before the verdict (phase 2). Eight variants now
sit within one cell of each other on every row.

**What they change** (learned arm; cost / violation / wall medians over the cells all four solved;
the rest over all 480):

| row | cost IPOPT / kq / N=16 / T=10 | viol kq / N=16 / T=10 | wall kq / N=16 / T=10 | outer steps kq / N=16 / T=10 | feasible at stop kq / N=16 / T=10 | resampled / N kq / N=16 / T=10 |
| --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 4.14 / 8.32 / 10.27 / 11.83 | 4.7e-05 / 4.9e-05 / 6.6e-05 | 12.0 / 4.8 / 59.6 s | 24 / 31 / 116 | 6 / 2 / 1 | 0.33 / 0.62 / 0.07 |
| grasp native | 4.52 / 8.58 / 10.38 / 10.80 | 4.4e-05 / 4.3e-05 / 6.7e-05 | 6.4 / 2.3 / 21.3 s | 13 / 15 / 42 | 13 / 3 / 2 | 0 / 0 / 0 |
| pose paired | 6.51 / 5.62 / 7.55 / 9.34 | 3.5e-05 / 2.6e-05 / 7.1e-05 | 5.8 / 2.1 / 26.9 s | 13 / 14 / 61 | 21 / 5 / 1 | 0.74 / 0.75 / 0.22 |
| pose native | 6.76 / 7.50 / 8.53 / 8.99 | 1.6e-05 / 1.6e-05 / 6.8e-05 | 5.7 / 1.9 / 23.5 s | 13 / 14 / 55 | 31 / 7 / 4 | 0.05 / 0.06 / 0.02 |

- **N = 16** stops in the same 13-31 outer steps at a quarter of the step cost, so its wall is
  2.5-3x lower than N = 64's (1.9-4.8 s, at or below IPOPT's learned wall on three rows), with the
  same violation band, 2-7 feasible particles at stop, and a worse objective (the N ladder's cost
  ordering is monotone: N = 256 < 64 < 16 on every row). On grasp paired it resamples 0.62 of its
  swarm per cell against 0.33 at N = 64.
- **T = 10 is read under the fielded metric, where the drive carries 1/T and the metric does not,
  so it is a 0.1x Newton step as much as a hotter target** (the H / T option, `svgd_metric_temperature`,
  exists since c61b7ce and is swept locally first). Measured: 3-5x more outer steps, 4-5x the wall
  (59.6 s on grasp paired), the worst objective of any variant (11.8 / 10.8 / 9.3 / 9.0), a wider
  violation band (7e-5), and 1-4 feasible particles at stop against 6-31. **The spread among the
  feasible particles does not grow** (6.0 / 5.7 / 4.9 / 3.5 rad against 5.9 / 5.2 / 5.0 / 4.1), so
  "temperature = coverage" did not show as coverage here; what showed is the smaller step and a
  hotter swarm with fewer particles inside the gate. Resampling falls (0.07 against 0.33 on grasp
  paired), consistent with the smaller step. All 480 stop `converged`.

**Cap check**: 0 cells at the step budget, 0 at the clock, on all eight rows.

### R11: what happens to a runaway particle (`drop`, `keep`), learned arm only, measured 2026-10-11

**Jobs 5892392-95, 192 items, 0 error cells; `--variants ...,drop,keep`. The first round on the
batched clearance row under the clearances venv** (`drake_version_txt` `0.0.20261011+crc.15a2c97` on
every record; `collision_threads` 40). Each differs from `kq` in `svgd_resample` only: `native` (the
fielded redraw) -> `drop` (a runaway or non-finite particle leaves the swarm) or `keep` (a runaway
keeps stepping; a non-finite particle is still dropped). The backend is a second difference against
R1's `kq` in seconds and, through the clock, in success; it evaluates the same functional bitwise.

| row | `kq` (R1) | `drop` | `keep` |
| --- | --- | --- | --- |
| grasp paired | 480 | **336** (143 swarms died) | **241** (239 at the clock) |
| grasp native | 480 | 480 | 480 |
| pose paired | 480 | **335** (145 died, 5 clock) | **208** (276 clock, 1 died) |
| pose native | 480 | 479 | 479 |

**Lead: the redraw is what carries the paired start, and nothing else moves.** On both native rows
the three modes are within one cell (`drop` and `keep` 480 / 479, `kq` 480 / 480, the two lost pose
cells sharing a target) and keep `kq`'s verdicts against the IPOPT twin (tie on grasp, win on pose).
On both paired rows `drop` loses 144-145 cells and `keep` 239-272 against `kq` (+0/-144 and
+0/-239 on grasp, p < 1e-40), turning `kq`'s tie and win into losses against IPOPT (336 v 479;
335 v 395 with 52 cells only svgd solved, 112 only IPOPT).

**Mechanism, read from the records.** `kq` redraws on 458 / 480 grasp paired and 479 / 480 pose
paired cells (median 21 and 48 of 64 particles over the solve), against 146 / 383 of the native
cells at a median of 0 and 3. The paired swarm starts around a random configuration in a scene
where most of the AL descent passes through the high-gain regions, so most particles run away at
some point. Under `drop` the swarm empties: a median 46 (grasp) / 52 (pose) of 64 particles leave,
the median solved cell ends with 18 / 12 live, and in 143 / 145 cells every particle is gone
(`all_dead`; scored `fail_reason` "nan", no particle to return). Only 21 and 14 of those dead swarms
had a feasible particle at their last recorded iterate, so the phase-2 change of returning that
iterate would recover ~15% of them, not the gap. Under `keep` a runaway keeps stepping, drags its
neighbours through the kernel-averaged objective gradient (the Tabor-Hermans form keeps the
constraint gradient outside the kernel and the objective gradient inside), overflows to non-finite
in float32 and is then dropped anyway (median 49 / 55 dropped), and the live remainder sits at the
multiplier box (`|mu|_inf` median 1e4 on both paired rows, 8.8e7 and 1.0e8 clips against `kq`'s
2.5e5) until the 180 s clock: every one of the 239 and 271 clock stops returned a particle at
`|q|_inf` ~ 1e22 rad. The survivors converge on the same median outer-step count as `kq` (23 v 24
grasp) -- the mode changes who survives, not how the survivors converge.

**Cost and stop on the surviving cells.** `drop` returns median cost 11.5 / 9.3 (grasp / pose
paired) against `kq`'s 8.4 / 6.6 on its own solved cells, from a swarm a third the size (feasible
particles at stop median 2 / 5 against 6 / 21); pose paired takes 69 median outer steps against 13.
On the native rows all three modes agree on steps (12-13), feasible particles (13 / 28-31) and cost
(8.6 / 7.9 against 8.4 / 7.0), so the fielded `native` costs nothing where it does not fire.

**The clearance row, measured on the cluster.** Outer step 80-83 ms against `kq`'s 502 ms on the
autodiff row (6.1x); collision share of the solve 0.15-0.30 against ~0.9. On the native rows,
where the solver's path is the same (same steps, same success), the per-cell wall ratio `kq` / this
round on the 479-480 mutually converged cells is **4.8-5.7x** (mean wall grasp native 7.14 -> 1.40 s,
pose native 6.97 -> 2.38 s). On pose paired `drop` the ratio is 1.0x because the thinned swarm takes
5x the outer steps. `collision_threads` reads 40 per solve at PROCS=2, i.e. the job's `nproc` is
80 hyperthreads, so two solves share 40 cores at 40 threads each; left as is for the rest of the
chain (changing it mid-chain would add a third difference), and the share says it is not the
bottleneck.

**One more go/no-go (3) disagreement** (the reporter's BUG line): grasp paired `drop` cell (17, 4),
`solver_feasible` True on the float32 rows, Drake float64 1.0648e-4 against the 1e-4 gate. Same
mechanism as R5's two; now three in 7680 cells, all within 7% of the gate.

**What it decides.** Against Thomas's two paper columns: for a constant-time multistart budget the
resample is swarm dynamics and `native` stands; for the one-solve column `native` is also the only
mode that holds the paired rows, and on the native rows the three modes are indistinguishable, so
no column needs a different mode. `drop` is the honest "no redraw" control for the paper (a swarm
that cannot refill solves 70% of the paired cells), `keep` the control that shows why the threshold
exists.

### R7: the seed control (`kq@seed2` against `ipopt@seed2`), measured 2026-10-11

**Jobs 5892403-06 (svgd, 96 items, clearance row) and 5892401-02 (the IPOPT twins, 32 items, both
arms, the nightly tarball), 0 error cells.** Grid seed 2 draws new targets and guesses; the reporter
pairs these two with each other only (`SVGD_TWIN_OF`), never with a seed-1 run. The svgd items are
R1's `kq` verbatim on the new grid.

| row | learned svgd (`kq@seed2`) | learned IPOPT (`ipopt@seed2`) | verdict | seed 1 (R1 v R2) |
| --- | --- | --- | --- | --- |
| grasp paired | 480 | 480 (joint space 448) | tie | 480 v 479, tie |
| grasp native | 480 | 480 (447) | tie | 480 v 480, tie |
| pose paired | 480 | 386 (211) | **svgd**, +94 / -0, p = 1e-28 | 480 v 395, svgd |
| pose native | 480 | 461 (211) | **svgd**, +19 / -0, p = 4e-6 | 480 v 458, svgd |

**Lead: seed 1 replicates on a fresh grid.** The learned arm under svgd solves 480 / 480 on all four
rows again, the IPOPT twin's counts move by at most 9 cells (386 v 395, 461 v 458, 480 v 479), and
every verdict and its direction is the same: ties on grasp, wins on pose where IPOPT leaves 19-94
cells unsolved and svgd none. The IPOPT twin's own learned-vs-joint-space line also replicates
(480 v 448, 480 v 447, 386 v 211, 461 v 211 against seed 1's 479 v 449, 480 v 448, 395 v 200,
458 v 200). Median outer steps 24 / 12 / 14 / 13 against seed 1's 24 / 13 / 13 / 13; feasible
particles at stop 6 / 13 / 20 / 28 against 6 / 13 / 21 / 31.

**Cost on shared cells** carries the seed-1 pattern: grasp 8.4 / 9.0 against IPOPT's 4.0 / 4.7
(~2x, the kernel-on stall of R1K); pose paired 5.9 against 6.2 (svgd cheaper), pose native 7.5
against 6.6. Violation 2.3e-5 to 4.8e-5 against 1e-8.

**Wall is on the clearance row** (not comparable to R1's): mean 2.10 / 1.21 / 1.30 / 1.43 s against
the IPOPT twin's 4.45 / 2.41 / 2.55 / 1.32 s on the tarball at the same one solve per GPU -- the
learned arm under svgd is now faster than under IPOPT on three of four rows and level on pose
native, where under the autodiff row (R1) it was 2.5-8x slower on every row. The maximum cell is
3.0-9.6 s; no cell is near the clock.

### R6: lr = 0.1, K = 1, K = 100 (`lr0p1`, `K1`, `K100`), learned arm only, measured 2026-10-11

**Jobs 5892407/09/10/11, 288 items, clearance row, 0 error cells; `--variants ...,lr0p1,K1,K100`.** Each
differs from `kq` in one setting: `svgd_lr` 0.3 -> 0.1; `svgd_inner_iters` (the SVN steps per check,
K) 10 -> 1 or 100. On the clearance row, so seconds are not R1's; the cleanest same-backend wall
reference is R7's `kq@seed2` (2.10 / 1.21 / 1.30 / 1.43 s on another grid).

| variant | success (gp / gn / pp / pn) | cost on shared cells | outer steps (median) | feasible particles | mean wall s |
| --- | --- | --- | --- | --- | --- |
| `kq` (R1, autodiff row) | 480 / 480 / 480 / 480 | 8.38 / 8.41 / 5.61 / 7.50 | 24 / 13 / 13 / 13 | 6 / 13 / 21 / 31 | 12.3 / 7.1 / 6.6 / 7.0 |
| `lr0p1` | 480 / 480 / 480 / 480 | 8.77 / 8.19 / 6.93 / 7.92 | 43 / 18 / 21 / 21 | 2 / 17 / 5 / 24 | 3.66 / 1.63 / 2.06 / 2.44 |
| `K1` | 480 / 480 / 480 / **479** | 7.96 / 8.78 / 6.71 / 8.15 | 34 / 36 / 38 / 46 (= inner steps) | **2 / 2 / 4 / 4** | **1.08 / 1.14 / 1.26 / 1.51** |
| `K100` | 480 / 480 / 480 / 480 | 9.97 / 9.30 / 7.16 / 7.59 | 18 / 10 / 8 / 8 (x100 inner) | **30 / 35 / 40 / 55** | 11.6 / 7.9 / 6.1 / 8.3 |

**Lead: all three tie `kq` on success** (the one `K1` loss is the fourth go/no-go (3) gate-edge
disagreement, pose native cell (15, 2), Drake 1.0070e-4 against the 1e-4 gate) and carry its verdicts
against the IPOPT twin. Eleven variants now sit within one cell of each other on every row.

**K is the knob that trades swarm convergence for wall, and it is the strip-down question Thomas
posed.** `K1` checks (dual update, resample, stop test, clock) after every SVN step: it converges in
34-46 inner steps where `kq` takes 130-240 (13-24 outer x 10), at 28-30 ms per outer step (one SVN
step plus the Drake check) against `kq`'s 82 ms on this row, so it is the fastest variant measured,
1.1-1.5 s mean, max 2.3-3.6 s, every cell converged, and its cost is within -5% / +9% of `kq`'s.
What it gives up is the swarm: the stall rule ("best feasible objective stalls for 5 checks") now
fires after 5 single steps instead of 50, so it stops with 2-4 feasible particles of 64 where `kq`
has 6-31 and `K100` 30-55. **For the one-solve column `K1` is a tie on success, cost and stop
behaviour and the simpler method; for a constant-budget multistart column that wants several
feasible particles per solve, the stall rule under K = 1 stops too early.** Stripping to one loop
would also mean re-expressing the stall window in inner steps (5 checks x K steps), not just
K = 1 -- which is the same loop with a 50-step stall window. **RULED (Thomas, 2026-10-11 evening):
K stays a separate parameter, fielded at 10, and the inner/outer structure is kept** -- *"I want to
better understand its effects as we start doing more complicated things. The balance of inner to
outer steps in AL seems like something we need to handle delicately."* The one-loop strip-down is
not done. `K100` is the other end: the most feasible particles, the lowest
violation (2.6e-6 to 3.0e-5), the worst grasp cost (9.3-10.0, the kernel-on stall having 100 steps
to act between dual updates) and 5-8x the wall; it earns nothing on these rows.

**lr = 0.1** is inert on success and slower: 1.5-1.8x the outer steps, cost within 10% either way,
fewer feasible particles on the paired rows (2 / 5 against 6 / 21). With R4's `lr1` (fastest stop,
worst grasp cost, most redraws) the step-size ladder reads 0.1 < 0.3 < 1.0 as wall-vs-cost with the
fielded 0.3 in the middle; nothing moves success.

### R8: N = 4, 32, 128 (`n4`, `n32`, `n128`), learned arm only, measured 2026-10-11

**Jobs 5892412-15, 288 items, clearance row, 0 error cells; `--variants ...,n4,n32,n128`.** With R3's
N = 1 and 256 and R5's N = 16 the particle-count ladder is complete at 1 / 4 / 16 / 32 / 64 / 128 / 256.

| N | success (gp / gn / pp / pn) | cost on shared cells | outer steps | feasible particles | redraws / N (paired grasp, median) | mean wall s (row) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 480 / 480 / 480 / **479** | 10.9 / 11.2 / 8.5 / 8.7 | 41 / 27 / 17 / 15 | 1 / 1 / 2 / 2 | 1.00 | 2.3 / 1.8 / 2.5 / 2.2 |
| 32 | 480 x 4 | 9.3 / 9.5 / 6.7 / 8.2 | 26 / 13 / 14 / 13 | 3 / 5 / 10 / 15 | 0.44 | 1.8 / 1.1 / 1.2 / 1.7 |
| 64 (`kq`, autodiff row) | 480 x 4 | 8.4 / 8.4 / 5.6 / 7.5 | 24 / 13 / 13 / 13 | 6 / 13 / 21 / 31 | 0.33 | 12.3 / 7.1 / 6.6 / 7.0 |
| 128 | 480 x 4 | 7.5 / 7.9 / 5.2 / 6.8 | 21 / 12 / 13 / 14 | 12 / 28 / 41 / 61 | 0.24 | 2.7 / 1.7 / 1.7 / 1.9 |

**Lead: N buys no success at any rung** -- 480 / 480 on 11 of the 12 rows, the twelfth the same
gate-edge cell as `K1`'s (below) -- and the verdicts against the IPOPT twin are `kq`'s on every rung.
With R3 and R5 that is the whole ladder from 1 to 256 within one cell on every row. **What N buys is
cost, monotonically**: on the shared cells 10.9 -> 9.3 -> 8.4 -> 7.5 (R3: 6.3-7.2 at 256) on grasp
paired, 8.5 -> 6.7 -> 5.6 -> 5.2 on pose paired, every row in the same order, with the feasible
particle count rising 1 -> 61 and the per-particle redraw rate falling 1.0 -> 0.24. The N = 4 swarm
is one feasible particle plus three being redrawn (`selected idx` median 2 of 4, redraws 1.0 per
particle per cell); N = 128 keeps 12-61 feasible and chooses among them. Outer steps fall with N on
the paired rows (41 -> 21 grasp) and are flat on native (12-13), so the swarm's coverage shortens
the paired search but not the native one.

**Wall on the clearance row is flat in N from 32 to 128** (1.1-1.8 against 1.7-2.7 s mean): the step
cost grows 66 -> 82 -> 114 ms per outer step (N = 32 / 64 / 128) while the step count falls, and
below that N = 4 is slower than N = 32 (49 ms per step but 1.5-3x the steps). So on this hardware the
cost-vs-wall trade is nearly free up to N = 128, and R3's N = 256 (3-4x `kq`'s wall on the autodiff
row) is where the GPU step stops amortising. **The one-solve column can run N = 128 at N = 32's
wall and a 20% lower cost**; that is the setting to carry into the kernel-space round, where the
latent kernel is expected to raise the feasible count further.

**The gate-edge cell is the same cell.** Pose native (15, 2) is the one lost cell for both `K1`
(Drake 1.0070e-4) and `n4` (1.0347e-4); `kq` returns it at 1.6e-5 and IPOPT at 8e-10. Five go/no-go
(3) disagreements in 11,520 cells, all on the float32 rows reading a particle inside the 1e-4 gate
that float64 Drake reads 0.05-7% outside it: the phase-2 fix (a stop-rule margin inside the gate, or
the float64 polish before the verdict) is a one-line change of policy, Thomas's call.

### R9: rho = 100, rho = 1e5, multiplier clip 1e6 (`rho100`, `rho1e5`, `clip1e6`), learned arm only, measured 2026-10-11

**Jobs 5892416-19, 288 items, clearance row, 0 error cells; `--variants ...,rho100,rho1e5,clip1e6`.** With
R4's `rho1e4` the penalty ladder is 1e2 / 1e3 / 1e4 / 1e5; `clip1e6` lifts the multiplier box from
1e4 to 1e6 at the fielded rho.

| variant | success | cost on shared cells (gp / gn / pp / pn) | outer steps | feasible particles | redraws / N (gp) | mean wall s |
| --- | --- | --- | --- | --- | --- | --- |
| `rho100` | 480 x 4 | 10.9 / 10.6 / 8.7 / 8.6 | 52 / 19 / 24 / 20 | 2 / 3 / 5 / 19 | 0.28 | 4.5 / 1.8 / 2.4 / 2.5 |
| `kq` (rho 1e3, autodiff row) | 480 x 4 | 8.4 / 8.4 / 5.6 / 7.5 | 24 / 13 / 13 / 13 | 6 / 13 / 21 / 31 | 0.33 | 12.3 / 7.1 / 6.6 / 7.0 |
| `rho1e4` (R4, autodiff row) | 480 x 4 | 6.2 / 7.5 / 5.3 / 6.9 | 16 / 13 / 12 / 13 | -- | 1.0 | -- |
| `rho1e5` | 480 x 4 | **5.9 / 7.0 / 5.2 / 6.2** | **12 / 14 / 12 / 13** | 16 / 16 / 29 / 31 | 1.05 | **1.1 / 1.3 / 1.2 / 1.3** |
| `clip1e6` | 480 x 4 | 7.9 / 8.6 / 5.8 / 7.4 | 23 / 13 / 13 / 13 | 8 / 13 / 24 / 29 | 0.48 | 2.1 / 1.3 / 1.3 / 1.4 |

**Lead: all three tie `kq` on success** (480 / 480 on all twelve rows, no gate-edge cell this time) and
carry its verdicts against the IPOPT twin. Fourteen variants now sit within one cell of each other on
every row. **rho is the cost lever, monotonically, and it costs nothing in wall**: on the shared
cells grasp paired cost reads 10.9 -> 8.4 -> 6.2 -> 5.9 over rho = 1e2 / 1e3 / 1e4 / 1e5, every row in
the same order, with outer steps falling 52 -> 24 -> 16 -> 12 and the feasible count rising 2 -> 6 ->
16. At rho = 1e5 the swarm converges in the fewest outer steps of any variant that keeps a swarm
(12-14, where `K1` takes 34-46 single steps) and returns the lowest grasp cost of the rho ladder,
within 1.5x of IPOPT's 4.1 against `kq`'s 2x -- so a large share of R1K's "kernel-on stall" premium
is the penalty being too soft, not the kernel. The price is the redraw rate: at rho = 1e5 the median
cell redraws every particle about once (1.05 per particle against 0.33 at 1e3), and the multiplier
clips at the 1e4 box ten times as often; neither moves success. rho = 100 is the other end: the
penalty too weak to pull the swarm onto the manifold (2-3 feasible particles on grasp), 2x the
steps, the worst cost of the ladder.

**The multiplier box is a stated limit and not the stall's cause.** Lifting it to 1e6 ties `kq` on
steps (23 / 13 / 13 / 13), lands cost between `kq` and `rho1e4` (7.9 / 8.6 / 5.8 / 7.4), and the
`|mu|_inf` maximum does reach 1e6 on every row, so the 1e4 box binds on some particles without
deciding anything. This closes the phase-2 question from the smoke ("JS multipliers at the clip"):
on the learned arm the clip is a bystander.

**Toward the kernel-space round.** rho = 1e5 is the one single-factor change in the ladder that
lowers cost on every row at `kq`'s wall or less; the SVN metric makes a large rho free (the
conditioning commit's reasoning). It is a candidate for the fielded pin once it is seen beside the
kernel space -- the stall it shortens is the q-kernel's, and under a latent kernel the swarm already
converges (stage A); the round should carry rho as a factor rather than assume the two add.

### R10: the paired swarm drawn from the native distribution (`initnative`), paired rows only, measured 2026-10-11

**Jobs 5892420-23, 48 items, clearance row, 0 error cells; `--variants ...,initnative`.** One change:
`svgd_paired_init` `jitter` -> `native`: particle 0 still sits exactly at the shared `q_init` in the
arm's own variables, but the other 63 are drawn from the arm's native start distribution (`c` at the
target pose, `z ~ N(0, I)`, `q_c = 0`) instead of jittered around the guess. The native rows are
unchanged by construction and were not run.

| row | success | cost on shared cells (`kq` / `initnative`) | outer steps | feasible particles | redraws / N | mean wall s |
| --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 480 | 8.38 / **5.78** | 24 / **10** | 6 / **17** | 0.33 / **0.00** | 12.3 (autodiff) / **1.00** |
| pose paired | 480 | 5.61 / **4.87** | 13 / **10** | 21 / **31** | 0.75 / **0.02** | 6.6 (autodiff) / **1.05** |

**Lead: the paired-start penalty on the learned arm is the jittered swarm, and the native draw
removes it.** Success is a tie (480 / 480 either way, the same verdicts against the IPOPT twin), but
every other column moves to the native rows' values or better: grasp paired cost 8.4 -> 5.8 (the
native row's `kq` is 8.4; this is the lowest grasp cost of any single-factor variant, below
`rho1e5`'s 5.9), outer steps 24 -> 10, feasible particles 6 -> 17, and the redraw rate from a third
of the swarm per cell to zero. The whole R11 story -- the paired swarm running away and surviving
only by the redraw -- is a property of the 63 jittered particles, which start where the shared guess
is (a random configuration, far from any grasp) and run into the high-gain regions together; drawn
from the native distribution they start where the flow puts a grasp of this mug and almost never
run away. Particle 0 keeps the protocol honest: the shared guess is in the swarm, exactly, and the
record says how often it was the particle returned (`selected idx`).

**What it means for the two paper columns.** The paired protocol asks each arm to start from the
same configuration; under a particle method "the same start" is one particle. `initnative` is the
natural reading of the paired protocol for a swarm, and it costs nothing on success. Fielding it
would make the two protocols nearly identical for the learned arm under svgd (10 / 10 outer steps,
1.0 / 1.0 s, 17 / 13 feasible), so the paired row would stop carrying information of its own --
which is itself the finding: the formulation-plus-swarm does not depend on the start. Thomas's
call whether the fielded paired init becomes `native` (phase 2).

### R12: the iiwa (`n4`), the primary `kq` in a second context, measured 2026-10-11

**Jobs 5892397-400, 96 items, clearance row, 0 error cells; `scripts/report_svgd.py --stage SVGD --rows
iiwa`.** The fielded method with zero overrides on the iiwa14 at its adopted chart `n4`, the four
rows at 480 cells, paired against the record's IPOPT learned arm from stage PAPER (the iiwa's
twin, same cells and scene). The joint-space smoke (SVGD_SMOKE_IIWA, 4 cells per row, 60 s) ran
first: learned 4 / 4 everywhere, joint space 3 / 3 / 4 / 4, no error cells.

| row | learned svgd | learned IPOPT (PAPER) | verdict | cost (svgd / IPOPT) | outer steps | feasible | mean wall s (svgd / IPOPT) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 480 | 479 (joint space 453) | tie | 8.6 / 7.6 | 19 | 6 | 1.46 / 6.27 |
| grasp native | 480 | 480 (453) | tie | 10.9 / 8.0 | 13 | 12 | 1.03 / 2.68 |
| pose paired | 480 | 406 (312) | **svgd** +74 / -0, p = 1e-22 | **5.4** / 9.0 | 14 | 25 | 1.07 / 1.61 |
| pose native | 480 | 462 (312) | **svgd** +18 / -0, p = 8e-6 | **7.3** / 9.0 | 14 | 28 | 1.27 / 0.67 |

**Lead: the Panda picture transfers to the iiwa unchanged.** 480 / 480 on all four rows, ties on
grasp (the IPOPT twin is at 479-480 there) and wins on pose where IPOPT leaves 18-74 cells. The
grasp cost premium is smaller than the Panda's (1.1-1.4x against 2x) and on pose svgd is cheaper
than IPOPT by 20-40%. Outer steps (13-19), feasible particles (6-28) and redraw rates (0.02-0.94
per particle, highest on pose paired) are the Panda's numbers within noise; every cell converged on
its own stop rule. Wall on the clearance row is 1.0-1.5 s, below the IPOPT twin on three rows (the
iiwa IPOPT learned arm is slow on grasp paired, 6.3 s) and above it only on pose native (1.27
against 0.67). The iiwa's runaway population (the gain-ceiling finding of the record) did not
surface: the per-particle redraw handles it as on the Panda, and no row needed the clock.

### R15: the screw arm (`screw7_p050`, `n6`), the primary `kq` in a third context, measured 2026-10-12

**Jobs 5895118-21, 96 items, clearance row, 0 error cells; `--rows screw7_p050`.** Zero overrides, the
four rows at 480 cells, paired against stage PAPER's screw IPOPT learned arm. The smoke
(SVGD_SMOKE_SCREW, 4 cells per row, 60 s, both arms) ran first: learned 4 / 4 everywhere, joint space
1 / 1 / 4 / 4 (its grasp cells at the 60 s clock, the ablation's known weakness), no error cells.

| row | learned svgd | learned IPOPT (PAPER) | verdict | cost (svgd / IPOPT) | outer steps | feasible | mean wall s (svgd / IPOPT) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 480 | 473 (joint space 442) | **svgd** +7 / -0, p = 0.016 | 10.5 / 9.5 | 16 | 3 | 1.54 / 11.11 |
| grasp native | 480 | 475 (442) | tie, +5 / -0, p = 0.06 | 12.2 / 9.0 | 13 | 10 | 1.38 / 6.93 |
| pose paired | 480 | 429 (320) | **svgd** +51 / -0, p = 9e-16 | **4.2** / 13.1 | 14 | 28 | 1.44 / 1.95 |
| pose native | 480 | 459 (320) | **svgd** +21 / -0, p = 1e-6 | **7.6** / 11.5 | 15 | 30 | 1.59 / 1.10 |

**Lead: the robot no algebraic method can chart is where the swarm wins most.** 480 / 480 on all
four rows; three wins and a near-win against the IPOPT learned arm, none of the Panda's and iiwa's
grasp ties, because the screw arm's IPOPT learned column is the record's slowest (398 majors and
11 s mean on grasp paired, 6 cells at the clock) while the swarm converges in 13-16 outer steps and
1.4-1.6 s on every row. Cost: the grasp premium is 1.1-1.35x as on the iiwa; on pose svgd returns a
third to two thirds of IPOPT's cost (4.2 against 13.1 on paired). The paired grasp swarm is thin
(3 feasible of 64, redraws 0.38 per particle, the Panda's pattern) and the pose swarms full
(28-30). Every cell converged on the stop rule; no clock, no iteration cap, no gate-edge cell.
Wall is below the IPOPT twin on three rows and 1.4x above it on pose native (1.59 against 1.10).
The gain-ceiling runaway of this robot's `p025` chart is not at issue here (`p050` has none); the
per-particle redraw rate is the Panda's.

### R13: the kernel space at 480 cells (`kz`, `kcz`, `kqz`, `kx`), learned arm only, measured 2026-10-12

**Jobs 5895126-29, 384 items, clearance row, 0 error cells; `--variants kq,...,kz,kcz,kqz,kx`.** One
change each against `kq`: `svgd_kernel` q -> z (the latent), cz (the product of the conditioning-pose
and latent blocks, per-block median bandwidths), qz (configuration x latent), x (the whole decision
vector). The 480-cell confirmation of the laptop screen's stage A.

| kernel | success | cost on shared cells (gp / gn / pp / pn) | outer steps | feasible particles | redraws / N (gp) | mean wall s | stops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q (`kq`, R1, autodiff row) | 480 x 4 | 8.38 / 8.41 / 5.61 / 7.50 | 24 / 13 / 13 / 13 | 6 / 13 / 21 / 31 | 0.33 | (12.3 / 7.1 / 6.6 / 7.0) | 1919 converged, 1 clock |
| z | 480 x 4 | 4.89 / 6.95 / 4.70 / 6.78 | 26 / 12 / 12 / 13 | 16 / 31 / 27 / 25 | 0.59 | 2.54 / 1.42 / 1.38 / 1.48 | all converged |
| c x z | 480 x 4 | **4.73 / 6.61 / 4.79 / 6.33** | 26 / **10** / 11 / 12 | **18 / 54 / 36 / 45** | 0.45 | 2.64 / 1.34 / 1.56 / 1.35 | 1919 converged, 1 clock |
| q x z | 480 x 4 | 5.73 / 7.05 / 4.87 / 6.40 | 22 / 10 / 13 / 12 | 11 / 44 / 31 / 39 | 0.25 | 2.07 / 1.33 / 1.68 / 1.45 | all converged |
| x | 480 x 4 | 5.18 / 7.42 / 4.78 / 7.11 | 22 / **9** / 11 / 11 | 26 / 38 / 28 / 37 | 1.02 | 2.11 / **0.82** / **1.08** / **1.12** | all converged |
| IPOPT learned (R2), for scale | 479 / 480 / 395 / 458 | 4.14 / 4.66 / 6.15 / 6.55 | -- | -- | -- | 5.0 / 3.0 / 2.0 / 0.8 | |

**Lead: the kernel space is the cost lever Thomas predicted, and it costs nothing in success or
wall.** All four spaces are 480 / 480 on every row with `kq`'s verdicts against the IPOPT twin
(nineteen variants within one cell on every row now). On grasp paired the cost falls from 8.38 to
4.7-5.7, i.e. from 2.0x IPOPT's 4.14 to **1.14x under c x z and 1.18x under z**; on grasp native from
8.41 to 6.6-7.4 (1.4-1.6x IPOPT); on both pose rows the swarm was already cheaper than IPOPT and the
new spaces take another 10-15% off. The feasible count at stop triples (c x z: 18 / 54 / 36 / 45 of
64 against 6 / 13 / 21 / 31) while the feasible q-spread FALLS (3.2 against 5.2 rad on grasp native):
the latent kernel keeps the swarm on the manifold and lets it settle into fewer, better basins, where
the q kernel holds particles apart in configuration space at the cost of feasibility -- the stage A
reading, at 480 cells. **The stop-rule interaction feared from the screen did not cost anything**:
c x z has one clock stop in 1920 cells (pose paired, 162 s, solved), every other cell converged on
the stall rule, and its outer-step counts are `kq`'s or lower. Wall is 1.3-2.6 s on every space,
`kq@seed2`'s 1.2-2.1 s on the same row (grasp paired is 0.4-0.5 s slower under z and c x z, the
price of the lower cost; x is the fastest variant on three rows at 0.8-1.1 s).

**Ranking by cost, grasp paired / native: c x z < z < x < q x z << q**; by feasible particles c x z >
q x z > x > z >> q; q x z is the most conservative (lowest redraw rate, 0.25) and x the most churned
(1.02, every particle redrawn once per cell) because the x kernel's bandwidth is dominated by the
latent block without the per-block scaling. **c x z is the fielding candidate** -- the lowest cost
on three of four rows, the fullest feasible swarm everywhere, one clock stop in 1920 -- with z as the
single-block alternative that gives up 3% of cost for a simpler kernel. Thomas's call, after R14
says whether the gain adds to rho = 1e5's and to N = 128's (both lower cost by a similar amount on
the q kernel) or substitutes for them.

**Against the record.** Under IPOPT the learned arm's grasp cost premium over joint space was
1.9-5.1x (CLAUDE.md, the record's cost paragraph); under svgd with the fielded q kernel it was 2x
IPOPT's own learned cost on top of that. With the latent kernel the learned arm under svgd returns
grasp solutions within 15% of IPOPT's learned cost on the paired start and 40-60% above on native,
and cheaper than IPOPT on pose, at 480 / 480 success and 1-3 s wall.

### R16: the soft PCS arm (`soft12`, `n6`), the primary `kq` in a fourth context, measured 2026-10-12

**Jobs 5895122-25, 96 items, clearance row, 0 error cells; `--rows soft12`.** Zero overrides, the four
rows at 480 cells, paired against stage PAPER's soft PCS IPOPT learned arm. The smoke (SVGD_SMOKE_SOFT)
ran first: learned 4 / 4 everywhere, joint space 4 / 4 / 4 / 3, no error cells. **This robot's svgd
step is the EAGER path**: `SoftArmIKProgram.TorchConfigToPlantQ` runs under autograd with no analytic
row Jacobians (`has_analytic_row_jacobians` False), so an outer step costs 800-835 ms against 82 ms
on the rigid arms -- the "rewrite the PCS and GVS forward steps in torch" TODO is what this row
measures the price of.

| row | learned svgd | learned IPOPT (PAPER) | verdict | cost (svgd / IPOPT) | outer steps | feasible | mean wall s (svgd / IPOPT) | clock stops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grasp paired | 480 | 477 (joint space 472) | tie, +3 / -0 | 2.03 / 0.70 | 20 | 12 | 16.4 / 6.1 | 0 |
| grasp native | 480 | 480 (472) | tie | 2.67 / 0.63 | 11 | 19 | 9.6 / 1.2 | 0 |
| pose paired | 480 | 437 (334) | **svgd** +43 / -0, p = 2e-13 | 2.33 / 0.99 | 12 | 21 | 26.4 / 8.7 | **24** (all solved) |
| pose native | 480 | 477 (334) | tie, +3 / -0 | 2.52 / 1.37 | 11 | 24 | 10.1 / 0.6 | 0 |

**Lead: success transfers (480 / 480 on all four rows, the one row where IPOPT is not at the
ceiling is a +43 win), but this is the robot where the swarm pays most in cost and wall.** Cost is
2-4x IPOPT's learned cost on every row (absolute values are small on this arm: 2.0-2.7 against
0.6-1.4), where the rigid arms were 1.1-2x on grasp and cheaper on pose. Wall is 10-26 s against
IPOPT's 0.6-8.7 s, the first context where svgd is slower than IPOPT on every row, and pose paired
has 24 cells at the 180 s clock (all 24 feasible when the clock fired; the reporter marks the row
clock-bound). Outer steps (11-20) and feasible counts (12-24) are the rigid arms' numbers, so the
swarm's dynamics are unchanged: the premium is the 10x step cost of the eager forward map and
nothing else. **The fix is the TODO, not the method**: a compiled torch PCS forward step with
analytic row Jacobians (the rigid arms' path) should return this robot to the 1-3 s band. Until
then the soft PCS row is reported at this cost and the comparison stays on success and iterations,
where it is the Panda's picture.

### R14: do the levers add? Declared two-factor combinations, learned arm only, measured 2026-10-12

**Jobs 5895130-34, 384 items, clearance row, 0 error cells; `--variants kq,rho1e5,n128,kz,kcz,kzHonT0p1,kzrho1e5,kzn128,kczrho1e5`.**
The first declared two-factor round (`SVGD_DECLARED_FACTORS`; the selftest holds each variant to
exactly its two factors): the latent kernel z with the H / T metric at T = 0.1 (one factor: at T = 1
the two metric forms coincide), z with rho = 1e5, z with N = 128, and c x z with rho = 1e5.

| variant | success | cost on shared cells (gp / gn / pp / pn) | outer steps | feasible particles | redraws / N (gp) | mean wall s |
| --- | --- | --- | --- | --- | --- | --- |
| `kq` (R1) | 480 x 4 | 8.38 / 8.41 / 5.61 / 7.50 | 24 / 13 / 13 / 13 | 6 / 13 / 21 / 31 | 0.33 | (autodiff row) |
| `kz` (R13) | 480 x 4 | 4.89 / 6.95 / 4.70 / 6.78 | 26 / 12 / 12 / 13 | 16 / 31 / 27 / 25 | 0.59 | 2.54 / 1.42 / 1.38 / 1.48 |
| `kcz` (R13) | 480 x 4 | 4.73 / 6.61 / 4.79 / 6.33 | 26 / 10 / 11 / 12 | 18 / 54 / 36 / 45 | 0.45 | 2.64 / 1.34 / 1.56 / 1.35 |
| `kzHonT0p1` | 480 x 4 | 4.11 / 5.86 / 4.53 / 6.08 | 18 / 13 / 11 / 12 | 35 / 32 / 32 / 28 | 1.17 | 2.66 / 1.56 / 1.13 / 1.31 |
| `kzrho1e5` | 480 x 4 | 4.09 / 5.19 / 4.52 / 5.44 | 14 / 14 / 10 / 13 | 35 / 32 / 32 / 28 | 1.06 | 1.71 / 1.57 / 1.13 / 1.39 |
| `kzn128` | 480 x 4 | 4.82 / 6.44 / 4.51 / 6.18 | 25 / 11 / 11 / 13 | 28 / 68 / 61 / 58 | 0.48 | 3.27 / 1.99 / 1.52 / 1.97 |
| `kczrho1e5` | 480 x 4 | **3.39 / 4.93 / 4.62 / 5.14** | 21 / 13 / 10 / 12 | **57 / 54 / 47 / 47** | 1.06 | 2.66 / 1.95 / 1.06 / 1.31 |
| IPOPT learned (R2) | 479 / 480 / 395 / 458 | 4.14 / 4.66 / 6.15 / 6.55 | | | | 5.0 / 3.0 / 2.0 / 0.8 |

**Lead: the kernel space and rho ADD; N does not add to the latent kernel; the H / T form at T = 0.1
is worth the same as rho = 1e5 on the latent kernel.** All four combinations are 480 / 480 on every
row with `kq`'s verdicts (twenty-three variants within one cell now), every cell converged on the
stop rule (the c x z clock stop of R13 is gone), wall 1.1-2.7 s.
- **c x z + rho = 1e5 returns grasp paired cost 3.39, BELOW IPOPT's learned 4.14 on the same cells**
  (0.82x), where the fielded `kq` sat at 2.0x; grasp native 4.93 against 4.66 (1.06x), pose 25-30%
  below IPOPT. The feasible swarm is 47-57 of 64 on every row. Against its parents: c x z alone 4.73,
  rho = 1e5 alone 5.94 -- the two gains stack almost fully (4.73 -> 3.39 is -28%, rho alone bought
  -29% on `kq`).
- **z + rho = 1e5** reads the same way (4.89 -> 4.09, -16%), with the fewest outer steps of the
  table (10-14).
- **z + H / T at T = 0.1** lands on z + rho = 1e5's numbers almost exactly (4.11 / 5.86 / 4.53 / 6.08
  against 4.09 / 5.19 / 4.52 / 5.44; the same 35 / 32 / 32 / 28 feasible). Under H / T a colder
  temperature scales the repulsion-to-drive ratio the way a larger rho scales the constraint-to-
  objective ratio; on this program the two knobs move the swarm the same way. Both roughly triple the
  redraw rate (1.06-1.17 per particle against 0.59), the same churn R9 saw at rho = 1e5.
- **z + N = 128 does not add**: cost 4.82 against z's 4.89 at twice the feasible particles and 1.3x
  the wall. N's cost gain on the q kernel (R8) came from having more feasible particles to choose
  among; the latent kernel already supplies them, so the swarm size stops mattering above 64.

**Fielding candidate after R13 + R14: c x z + rho = 1e5** (or z + rho = 1e5 as the single-block
form). What it changes against today's pin is two settings and one kernel definition; what it
buys is a learned arm under svgd that is 480 / 480 on all four Panda rows at 1-3 s with grasp cost
at or below IPOPT's learned arm. The open items it does not settle: H / T as the metric form (a
definition, bit-identical at T = 1; T stays 1 under this candidate), the native-draw paired init
(R10, orthogonal), and whether the gain transfers to the iiwa, screw and soft PCS contexts (not
measured; their rounds ran `kq`).

### R17: the kernel space transfers to the iiwa (`kcz`, `kczrho1e5` on `n4`), measured 2026-10-12

**Jobs 5895823-26, 192 items, clearance row, 0 error cells; `--rows iiwa --variants kq,kcz,kczrho1e5`.**
The c x z kernel's first runs off the Panda (latent width 8 here), alone and with rho = 1e5, against
R12's `kq` and stage PAPER's iiwa IPOPT learned arm.

| variant | success | cost on shared cells (gp / gn / pp / pn) | outer steps | feasible particles | mean wall s |
| --- | --- | --- | --- | --- | --- |
| `kq` (R12) | 480 x 4 | 8.65 / 10.93 / 5.44 / 7.29 | 19 / 13 / 14 / 14 | 6 / 12 / 25 / 28 | 1.46 / 1.03 / 1.07 / 1.27 |
| `kcz` | 480 x 4 | 5.18 / 8.45 / 4.00 / 5.69 | 20 / 11 / 14 / 14 | 13 / 49 / 34 / 40 | 1.60 / 1.21 / 1.20 / 1.34 |
| `kczrho1e5` | 480 x 4 | **3.50 / 4.47 / 3.79 / 4.77** | 14 / 14 / 12 / 13 | **53 / 52 / 38 / 37** | 1.57 / 1.69 / 1.01 / 1.29 |
| IPOPT learned (PAPER) | 479 / 480 / 406 / 462 | 7.57 / 7.95 / 9.03 / 8.99 | | | 6.27 / 2.68 / 1.61 / 0.67 |

**Lead: the Panda's result transfers whole.** 480 / 480 on every row for both variants, `kq`'s
verdicts against IPOPT (ties on grasp, +74 and +18 on pose), every cell converged, wall 1.0-1.7 s.
The cost moves as on the Panda and by the same proportions: c x z alone -40% / -23% / -26% / -22%
against `kq`, c x z + rho = 1e5 -60% / -59% / -30% / -35%, and the combination's cost is **below the
IPOPT learned arm's on all four rows** (0.46x / 0.56x / 0.42x / 0.53x) where `kq` was above it on
grasp (1.14x / 1.37x). Feasible particles at stop 37-53 of 64 against 6-28. The latent kernel on a
width-8 latent behaves as on the width-7 one; nothing about the iiwa's runaway population (the
record's gain-ceiling finding) shows in the redraw rate (0.3-1.2 per particle, the Panda's band at
rho = 1e5).

### R17S: the kernel space transfers to the screw arm (`kcz`, `kczrho1e5` on `p050 n6`), measured 2026-10-12

**Jobs 5895827-30, 192 items, clearance row, 0 error cells; `--rows screw7_p050 --variants kq,kcz,kczrho1e5`.**

| variant | success | cost on shared cells (gp / gn / pp / pn) | outer steps | feasible particles | mean wall s |
| --- | --- | --- | --- | --- | --- |
| `kq` (R15) | 480 x 4 | 10.49 / 12.19 / 4.19 / 7.57 | 16 / 13 / 14 / 15 | 3 / 10 / 28 / 30 | 1.54 / 1.38 / 1.44 / 1.59 |
| `kcz` | 480 x 4 | 6.83 / 9.55 / 3.58 / 5.95 | 17 / 13 / 14 / 16 | 6 / 25 / 35 / 38 | 1.76 / 1.55 / 1.66 / 2.17 |
| `kczrho1e5` | 480 x 4 | **3.36 / 4.03 / 3.57 / 4.89** | 14 / 15 / 13 / 14 | **49 / 46 / 37 / 36** | 1.97 / 1.89 / 1.50 / 1.68 |
| IPOPT learned (PAPER) | 473 / 475 / 429 / 459 | 9.45 / 9.02 / 13.12 / 11.48 | | | 11.1 / 6.9 / 2.0 / 1.1 |

**Lead: transfers whole on the third robot too.** 480 / 480 on every row, R15's verdicts (a grasp
paired win, a native near-win, two pose wins), every cell converged, wall 1.5-2.2 s. Cost against
`kq`: c x z alone -35% / -22% / -15% / -21%, with rho = 1e5 -68% / -67% / -15% / -35%; the
combination sits at 0.36x / 0.45x / 0.27x / 0.43x of the IPOPT learned arm's cost on the same cells
(the screw arm's IPOPT learned column is the record's most expensive). Feasible particles 36-49 of
64 against 3-30. The paired grasp swarm that `kq` and `kcz` leave thin (3 and 6 feasible) fills to 49
under rho = 1e5, at a redraw rate of 1.5 per particle per cell -- the churn is the price on this
arm as on the others, and nothing in success or stop behaviour pays for it. **Three robots, twelve
rows, one setting change (c x z + rho = 1e5): 480 / 480 everywhere, cost below the IPOPT learned arm
on 11 of 12 rows** (the twelfth, Panda grasp native, 1.06x).

### R18: a third factor on the candidate pin (`kczrho1e5n128`, `kczrho1e5initnative`), measured 2026-10-12

**Jobs 5895831-34, 144 items, clearance row, 0 error cells.** Declared three-factor stacks on the
Panda: c x z + rho = 1e5 + N = 128 (four rows) and c x z + rho = 1e5 + the native-draw paired init
(paired rows only, as R10).

| variant | success | cost on shared cells (gp / gn / pp / pn) | outer steps | feasible particles | redraws / N (gp / pp) | mean wall s |
| --- | --- | --- | --- | --- | --- | --- |
| `kczrho1e5` (R14) | 480 x 4 | 3.39 / 4.93 / 4.62 / 5.14 | 21 / 13 / 10 / 12 | 57 / 54 / 47 / 47 of 64 | 1.06 / 1.06 | 2.66 / 1.95 / 1.06 / 1.31 |
| `kczrho1e5n128` | 480 x 4 | 3.26 / 4.55 / 4.46 / 4.89 | 27 / 13 / 10 / 12 | 120 / 111 / 100 / 101 of 128 | 1.06 / 1.06 | 4.61 / 2.71 / 1.39 / 1.81 |
| `kczrho1e5initnative` | 480 / -- / 480 / -- | 3.37 / -- / 4.63 / -- | 19 / -- / 9 / -- | 58 / -- / 47 / -- | **0.05 / 0.08** | 2.51 / -- / 0.98 / -- |

**Lead: neither third factor moves success, and they separate on what they cost.** N = 128 buys
4-8% of cost for 1.4-1.7x the wall -- the same verdict as R14's z + N = 128: once the latent kernel
fills the swarm (94% of particles feasible at either N), a bigger swarm only costs steps. **The
native-draw paired init buys nothing in cost (3.37 against 3.39) and removes the churn**: redraws
fall from 1.06 to 0.05 per particle on grasp paired and from 1.06 to 0.08 on pose paired, outer
steps 21 -> 19 and 10 -> 9, wall 2.66 -> 2.51 and 1.06 -> 0.98 s, feasible count unchanged (58 / 47).
So rho = 1e5's one visible price under the candidate pin -- every particle redrawn about once per
paired cell -- is entirely the jittered start, and the R10 reading holds under the new kernel: the
paired protocol's swarm should be drawn from the native distribution with particle 0 at the shared
guess. Both stacks converged on every cell.

**Where this leaves the pin.** The candidate is c x z + rho = 1e5 at N = 64, with the native-draw
paired init as the paired protocol's swarm; H / T as the metric form is a definition change that is
bit-identical at T = 1 and leaves T = 1 in the pin. Measured on three robots and twelve rows at
480 / 480 with cost below the IPOPT learned arm on eleven. Open: the soft PCS arm has not run the
new kernel; a seed-2 replication of the candidate has not run; and fielding is Thomas's decision,
after which the record rows are re-run at the new pin.

### R19: the candidate pin replicates on the seed-2 grid (`kczrho1e5@seed2`), measured 2026-10-12

**Jobs 5896065-68, 96 items, clearance row, 0 error cells; `--variants kq,kczrho1e5`.** c x z + rho = 1e5
on the Panda's seed-2 grid, paired with the seed-2 IPOPT twin (R7IPOPT) and read beside R7's
`kq@seed2`; never paired with a seed-1 run (the reporter refuses).

| | seed 1 (R14) | seed 2 (R19) | IPOPT learned, seed 2 |
| --- | --- | --- | --- |
| success (gp / gn / pp / pn) | 480 x 4 | 480 x 4 | 480 / 480 / 386 / 461 |
| verdict vs IPOPT | T / T / W / W | T / T / W +94 / W +19 | |
| cost on shared cells | 3.39 / 4.93 / 4.62 / 5.14 | 3.48 / 4.70 / 4.37 / 4.90 | 4.00 / 4.66 / 6.15 / 6.55 |
| feasible particles | 57 / 54 / 47 / 47 | 57 / 54 / 48 / 48 | |
| outer steps | 21 / 13 / 10 / 12 | 21 / 13 / 10 / 12 | |
| mean wall s | 2.66 / 1.95 / 1.06 / 1.31 | 2.60 / 2.05 / 1.06 / 1.27 | 4.45 / 2.41 / 2.55 / 1.32 |

**Lead: the candidate replicates to the digit.** 480 / 480 on all four rows, the same verdicts, cost
within 3-5% of seed 1 on every row and below the seed-2 IPOPT learned arm on three (grasp native
4.70 against 4.66, level), the same feasible counts and step counts, every cell converged, and the
wall 1.1-2.6 s against the twin's 1.3-4.5 s. The one-cell reproducibility band the record quotes
for cap-bound rows does not apply here: nothing is cap-bound.

### R17P: the kernel space on the soft PCS arm (`kcz`, `kczrho1e5` on `soft12 n6`), measured 2026-10-12

**Jobs 5896069-72, 192 items, clearance row, 0 error cells; `--rows soft12 --variants kq,kcz,kczrho1e5`.**
The c x z kernel's first runs on a width-12 latent and on the eager forward step (R16).

| variant | success | cost on shared cells (gp / gn / pp / pn) | outer steps | feasible particles | mean wall s | clock stops |
| --- | --- | --- | --- | --- | --- | --- |
| `kq` (R16) | 480 x 4 | 2.03 / 2.67 / 2.33 / 2.52 | 20 / 11 / 12 / 11 | 12 / 19 / 21 / 24 | 16.4 / 9.6 / 26.4 / 10.1 | 0 / 0 / 24 / 0 |
| `kcz` | 480 x 4 | 1.02 / 1.78 / 1.58 / 1.77 | 20 / 8 / 11 / 10 | 60 / 58 / 43 / 60 | 18.2 / 7.5 / 37.7 / 9.5 | 3 / 0 / 43 / 0 |
| `kczrho1e5` | 480 x 4 | **0.90 / 1.48 / 1.36 / 1.58** | **9 / 8 / 9 / 9** | **61 / 58 / 59 / 59** | **10.1 / 8.6 / 7.5 / 8.6** | **0 / 0 / 0 / 0** |
| IPOPT learned (PAPER) | 477 / 480 / 437 / 477 | 0.70 / 0.63 / 0.99 / 1.37 | | | 6.1 / 1.2 / 8.7 / 0.6 | |

**Lead: transfers on the fourth robot, and here it also fixes the clock.** 480 / 480 on every row
for both variants with R16's verdicts (three ties at IPOPT's ceiling, +43 on pose paired). Cost
against `kq`: c x z alone -50% / -33% / -32% / -30%, with rho = 1e5 -56% / -45% / -42% / -37%; the
combination sits at 1.1-1.5x the IPOPT learned arm (0.90 against 0.70, 1.48 against 0.63, 1.36
against 0.99, 1.58 against 1.37), down from `kq`'s 1.8-4.3x, with 58-61 of 64 particles feasible on
every row. **The combination halves the outer steps (9 / 8 / 9 / 9 against 20 / 11 / 12 / 11) and
with them the wall**: 7.5-10.1 s against `kq`'s 9.6-26.4 s, and R16's 24 clock stops on pose paired
are gone (c x z alone has 43 there -- the fuller swarm improving slowly under the stall rule, the
stage A interaction, which rho = 1e5 removes by converging in 9 steps). The per-step cost is
unchanged at 780-850 ms (the eager forward map), so this robot's remaining wall premium over IPOPT
(1.2-7x by row) is entirely the torch-rewrite TODO.

**Four robots, sixteen rows, one setting change (c x z + rho = 1e5): 480 / 480 everywhere, every
cell converged, cost below the IPOPT learned arm on 11 of 16 rows and within 1.5x on the other five
(Panda grasp native 1.06x, the four soft PCS rows 1.1-1.5x), wall 1-3 s on the rigid arms and
7.5-10 s on the soft PCS arm.**

### RULINGS 2026-10-12 (Thomas, morning): the fielded pin, and native is the method

**Fielded from here on: c x z + rho = 1e5 (N = 64, T = 1) with the H / T metric form and the
native-draw paired swarm** -- items 1 and 2 of the decision list, approved as proposed. **"Native draw
is the method and paired is the ablation"**: the native start protocol is the primary row of the
learned arm under svgd; the paired start measures start sensitivity and is reported as an ablation,
as joint-space svgd is. The defaults in `ProgramOptions` move to the pin once item 3 (the gate-edge
rule) is decided, so the record rows are re-run under one change set.

**Still queued**, learned arm only (Thomas, 2026-10-10: *"drop joint space ablation arm from R3, R4,
R5"*, extended to R6-R10), **and from here on every svgd round runs the batched clearance row under
the clearances venv** (Thomas, 2026-10-11 14:45: *"restage everything using batch clearances"*; R6 had
72 of 288 items done on the default row and was cancelled and voided with R8-R10 unstarted, so no
round mixes the two rows). Submitted 2026-10-11 15:00 in this order: R11 (resample `drop` / `keep`),
the iiwa joint-space smoke, R12 (iiwa `n4`, `kq`, after the smoke), the seed-2 IPOPT twins (on the
nightly tarball), R7 (`kq` at seed 2), R6 (lr = 0.1, K = 1, K = 100), R8 (N = 4, 32, 128), R9
(rho = 100, rho = 1e5, clip 1e6), R10 (paired init from the native draw). Their sections are appended
as they land. **Wall from these rounds is not comparable to R1-R5's**, and a variant read against R1's
`kq` carries the collision backend as a second difference (seconds, and through the clock possibly
success); the clearance row evaluates the same functional bitwise, so success differences at the
clock are the only channel.

### Gate-edge rule (2026-10-12)

Thomas, on the proposal: *"Yes, I like the proposed fix."* **A stall stop stands only once the
float64 Drake re-check passes.** When the stop rule would stop with reason `converged`, the solver
runs the end selection's own re-check there and then (`SvgdSolver._select`: the top
`svgd_recheck_topk` particles feasible on the batched rows, by objective, each through
`prog.EvalBinding` in float64). If a candidate passes, the swarm stops and that selection IS the
end selection -- one re-check, as before, never two. If none passes, the swarm does not stop: the
stall counter resets, the solve's internal feasibility level `feas_level` -- the per-row test
`v <= feas_level`, 1.0 at the start, used by the stop rule, the trace and the selection's mask
alike -- is multiplied by 0.9 (0.9, 0.81, ...), the failure is recorded, and the swarm continues
to the next stall, where the re-check runs again. The wall-clock, step-cap and `all_dead` stops are
unchanged (select at the end, at the current level). Recorded per cell: `n_recheck_failures`,
`feas_level` at stop (both in `_SVGD_DETAIL_KEYS`), `solver_feasible` / `drake_feasible` as
before, and each refused stop in `extras["recheck_failures"]`.

**Why.** Five cells in 11,520 stopped converged on a float32 particle inside the 1e-4 gate that
float64 Drake read just outside it, and each had exactly ONE float32-feasible particle, so the
re-check had nothing else to pass. A fixed margin (a 0.9999e-4 gate inside the swarm) covers none
of them: the float32-vs-float64 gap is the network's float32 error amplified by the chart's gain,
0.05-6.5% of the gate, not a rounding constant. Continuing the swarm at a tighter level lets it
fill in particles that pass. **No per-iteration cost** (Thomas: *"This is something that has
happened 5 times in 11,520 -- incredibly rare! And we're going to add a 1 ms cost to every outer
iteration?"*): nothing float64 runs except at a stop event, and a cell whose first re-check passes
is bit-identical to the pre-rule solver (`tests/test_svgd_pin_and_gate.py`, against arrays captured
at d200121; a forced first failure continues, records one failure, stops at level 0.9 and returns
a Drake-feasible point).

**The pin moved with it** (items 1 and 2 of the ruling): `ProgramOptions` defaults are now
`svgd_kernel=cz`, `svgd_rho=1e5`, `svgd_metric_temperature=on`, `svgd_paired_init=native` (N = 64,
T = 1, lr = 0.3, delta = 10, K = 10, resample `native` and the stop rule unchanged). `cz` is
learned-only, so a joint-space svgd solve must now name its kernel (e.g. `q`). Every existing
manifest (R1-R19, the smokes) ran the old pin, regenerates byte-identically
(`SVGD_PINNED_V1`; kernel q / metric off / init jitter were never named, and
`SVGD_HISTORICAL_DEFAULTS` records them for the reporter), and a stage from R20 on takes
`SVGD_PINNED_V2`, which names all four moved settings.

### R20: the record rows at the fielded pin (`pin2`, `pin2kq`), measured 2026-10-12

**Ran 2026-10-12 13:02-14:52 EDT, before the maintenance** (Thomas: an idle cluster until the evening
is wasted compute). Tip 86efbb0 restaged (the pin, the gate-edge rule and the soft PCS split step
all in the staged tree; `.staged-commit` verified), jobs 5897146 (SVGD_SMOKE_R20, one node) and
5897148-51 (R20, four nodes, `afterok`), `DRAKE_ENV=crc PROCS=2`, the clearance row. **6 node-hours
against the 25 allotted**: the joint-space ablation, charged at 60 s per grasp cell, converged in
1.5 s. `cluster/SVGD_RUNBOOK.md`, "R20". The first stage on `SVGD_PINNED_V2` (c x z, rho 1e5, H / T,
the native-draw paired swarm, gn_lm 10, lr 0.3, every one named), the record's sixteen rows, four
robots x grasp / pose x native / paired, 480 cells at 180 s on seed 1, zero further overrides
(`pin2`), read against the same IPOPT twins the single-robot rounds paired with (SVGD_R2 on the
Panda, stage PAPER on the others). **Native is the primary row and paired its ablation** (the
ruling above). The joint-space ablation (`pin2kq`, the pin with `svgd_kernel=q`) ran on the
Panda's four rows in its own items.

**The smoke passed every gate item** (eight items, Panda, 4 cells at 60 s, both arms as separate
items): 32 / 32 cells solved, zero `error` records, `svgd_kernel` `q` on the `pin2kq` items and
`cz` on `pin2`, `svgd_warmup_seconds` present, `n_recheck_failures` and `feas_level` on every record
(the restaged solver ran), the solver's verdict equal to Drake's on 32 / 32, slowest cell 10.1 s.
Joint space at the new pin solved its 16 cells in 1.1-1.6 s.

**Learned under svgd at the fielded pin, against learned under IPOPT, same cells** (exact McNemar;
cost = median reported cost, svgd on the cells it shares with the twin's learned arm, IPOPT the
reporter's column on the cells both of its arms solved; wall = mean over all cells clamped at 180 s;
steps = median svgd outer steps; max = the slowest svgd cell):

| row | svgd | IPOPT | verdict | cost svgd / IPOPT | wall s svgd / IPOPT | outer | ms / outer | max s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda grasp native | 480 | 480 | tie | 3.39 / 4.51 | 2.47 / 3.00 | 18 | 84 | 19.7 |
| Panda grasp paired | 480 | 479 | tie | 3.37 / 4.14 | 2.49 / 5.01 | 19 | 83 | 14.5 |
| Panda pose native | 480 | 458 | **svgd** p = 4.8e-7 | 4.61 / 6.71 | 1.01 / 0.77 | 9 | 85 | 6.5 |
| Panda pose paired | 480 | 395 | **svgd** p = 5.2e-26 | 4.63 / 6.40 | 1.00 / 2.02 | 9 | 86 | 5.6 |
| iiwa grasp native | 480 | 480 | tie | 3.46 / 7.95 | 1.44 / 2.68 | 11 | 66 | 13.9 |
| iiwa grasp paired | 480 | 479 | tie | 3.41 / 7.57 | 1.28 / 6.27 | 11 | 66 | 10.3 |
| iiwa pose native | 480 | 462 | **svgd** p = 7.6e-6 | 3.88 / 8.99 | 0.89 / 0.67 | 11 | 68 | 3.5 |
| iiwa pose paired | 480 | 406 | **svgd** p = 1.1e-22 | 3.85 / 9.03 | 0.92 / 1.61 | 11 | 68 | 3.8 |
| screw grasp native | 480 | 475 | tie p = 0.062 | 3.28 / 9.02 | 1.76 / 6.93 | 13 | 84 | 10.9 |
| screw grasp paired | 480 | 473 | **svgd** p = 0.016 | 3.27 / 9.45 | 1.95 / 11.11 | 13 | 85 | 51.1 |
| screw pose native | 480 | 459 | **svgd** p = 9.5e-7 | 3.74 / 11.48 | 1.28 / 1.10 | 12 | 86 | 7.8 |
| screw pose paired | 480 | 429 | **svgd** p = 8.9e-16 | 3.66 / 13.12 | 1.29 / 1.95 | 11 | 87 | 9.6 |
| soft PCS grasp native | 480 | 480 | tie | 0.909 / 0.626 | 2.58 / 1.21 | 8 | 213 | 19.5 |
| soft PCS grasp paired | 480 | 477 | tie p = 0.25 | 0.889 / 0.704 | 2.54 / 6.12 | 8 | 214 | 22.7 |
| soft PCS pose native | 480 | 477 | tie p = 0.25 | 1.389 / 1.368 | 1.51 / 0.61 | 8 | 177 | 7.6 |
| soft PCS pose paired | 480 | 437 | **svgd** p = 2.3e-13 | 1.370 / 0.985 | 1.52 / 8.70 | 8 | 174 | 4.8 |

- **480 / 480 on all sixteen rows; 8 wins, 8 ties, 0 losses** against the learned arm under IPOPT.
  Zero clock stops and zero step-cap stops on any row (IPOPT's twins: screw grasp 6 / 3 and soft
  PCS 3 / 18 timeouts). Every cell `converged`.
- **Success is identical cell for cell to the V1 rounds on every row** (`pin2` against `kq`:
  +0 / -0 on all sixteen; the VARIANT vs VARIANT block). The pin moves cost and wall, not cells.
- **Cost below the IPOPT learned arm on the twelve rigid-arm rows** (0.55-0.81x on the Panda,
  0.43-0.45x on the iiwa, 0.28-0.36x on the screw arm); above it on three soft PCS rows (1.26-1.45x)
  and level on soft PCS pose native (1.02x).
- **Wall (cluster, one solve per GPU, cap-clamped mean): 0.9-2.6 s on every row.** Below the IPOPT
  learned arm on all eight paired rows and on the four grasp-native rows but soft PCS; above it on
  pose native (IPOPT 0.6-1.1 s there).
- **Residual at stop** 4e-6 to 3e-5 (median `max_violation`) against IPOPT's 1e-8; the task gate is
  1e-3. Feasible particles at stop: median 58 of 64 on Panda grasp paired (the q kernel's 6).
- **`n_recheck_failures` is 0 on every cell of every row**: the gate-edge rule never fired at 480
  x 20 = 9600 cells.

**Learned under svgd against JOINT SPACE under IPOPT, same cells** (the paper's comparison target;
`scripts/svgd/cross/learned_svgd_vs_js_ipopt.py`). Time-matched = joint-space IPOPT allowed restarts
from the target's other recorded starts, random order, inside svgd's wall on that cell (200
permutations per target); starved = the budget outlasted all eight starts (0.0-0.4% everywhere):

| row | learned svgd | JS IPOPT | L+ / JS+ | p | JS time-matched | cost L / JS (shared) | wall s L / JS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda grasp native / paired | 480 | 448 / 449 | 32 / 0, 31 / 0 | 5e-10, 9e-10 | 64.8% / 64.2% | 3.37 / 2.25 | 2.47 / 6.07 |
| Panda pose native / paired | 480 | 200 | 280 / 0 | 1e-84 | 84.4% / 84.6% | 4.51 / 7.71 | 1.01 / 0.53 |
| iiwa grasp native / paired | 480 | 453 | 27 / 0 | 1.5e-8 | 74.4% / 74.3% | 3.44 / 2.91 | 1.44 / 1.02 |
| iiwa pose native / paired | 480 | 312 | 168 / 0 | 5e-51 | 99.2% / 99.3% | 3.87 / 9.20 | 0.89 / 0.09 |
| screw grasp native / paired | 480 | 442 | 38 / 0 | 7e-12 | 83.7% / 84.1% | 3.25 / 1.90 | 1.76 / 1.08 |
| screw pose native / paired | 480 | 320 | 160 / 0 | 1e-48 | 97.9% / 98.2% | 3.58 / 12.38 | 1.28 / 0.12 |
| soft PCS grasp native / paired | 480 | 472 | 8 / 0 | 0.008 | 82.2% / 81.3% | 0.90 / 0.24 | 2.58 / 3.19 |
| soft PCS pose native / paired | 480 | 334 | 146 / 0 | 2e-44 | 92.4% / 92.3% | 1.37 / 0.55 | 1.51 / 0.34 |

Every discordant cell on every row goes to svgd; time-matched joint-space IPOPT reaches 64-99% and
100% nowhere, closest on iiwa and screw pose, where one joint-space solve is 0.1 s. Joint-space IPOPT
solutions are cheaper on every grasp row (1.2x iiwa, 1.5x Panda, 1.7x screw, 3.7x soft PCS) and on
soft PCS pose (2.5x); the learned svgd solutions are cheaper on rigid-arm pose (1.7-3.5x). Joint-space
IPOPT is faster on every pose row (0.09-0.5 s a cell), svgd on three of four grasp rows.

**The soft PCS split step on the cluster** (`pin2` R20 against `kczrho1e5` R17P, the same pin's two
method factors on the autograd / eager step, same grid):

| soft PCS row | ms / outer step R17P -> R20 | wall s | max s | outer steps | cost |
| --- | --- | --- | --- | --- | --- |
| grasp native | 804 -> 213 | 8.57 -> 2.58 | 96 -> 20 | 8 -> 8 | 1.48 -> 0.91 |
| grasp paired | 780 -> 214 | 10.11 -> 2.54 | 88 -> 23 | 9 -> 8 | 0.90 -> 0.89 |
| pose native | 850 -> 177 | 8.57 -> 1.51 | 127 -> 8 | 9 -> 8 | 1.58 -> 1.39 |
| pose paired | 832 -> 174 | 7.51 -> 1.52 | 64 -> 5 | 9 -> 8 | 1.36 -> 1.37 |

3.7-4.8x per outer step and 3.3-5.7x in wall, success and step counts unchanged (+0 / -0 on every
row against R17P's 480s). The soft PCS arm now sits at 2.4-3.2x the rigid arms' ms per outer
step (the clearance row over 231 plant positions), where it was 10x.

**The joint-space ablation at the pin (`pin2kq`, Panda, kernel q)** against joint space under IPOPT,
same cells (not a baseline):

| row | svgd | IPOPT | verdict | cost svgd / IPOPT | wall s svgd / IPOPT | outer | clock stops svgd (R1 `kq`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp native | 480 | 448 | svgd p = 4.7e-10 | 3.39 / 2.25 | 1.49 / 6.07 | 20 | 0 (143) |
| grasp paired | 480 | 449 | svgd p = 9.3e-10 | 3.39 / 2.23 | 1.45 / 6.06 | 20 | 0 (143) |
| pose native | 480 | 200 | svgd p = 1.0e-84 | 4.43 / 7.69 | 0.83 / 0.53 | 14 | 0 (29) |
| pose paired | 480 | 200 | svgd p = 1.0e-84 | 4.43 / 7.58 | 0.83 / 0.53 | 14 | 0 (29) |

At the V1 pin (R1, kernel q, rho 1e3, jitter) joint-space svgd was 338 / 480 on grasp with 143 cells
at the clock and 451 on pose; **at the V2 pin it is 480 / 480 on all four rows with no clock stop**,
at 0.8-1.5 s a cell. Its grasp cost is 1.5x IPOPT's joint-space cost (the learned arm's is 0.75x
IPOPT's learned), its pose cost 0.6x. Feasible particles at stop: median 4 of 64 (the learned arm's
58). Multiplier clips per run: 2.8e5 (`mug_xy`) and 5.2e5 (`collision`) on the joint-space grasp
rows against 0.9e4 and 1.2e4 on the learned arm's, so the swarm without the network leans on the
clip where the learned arm does not. Which of rho 1e5, the H / T metric and the native init did this
is not separated (the ablation ran the pin as a block, by design).

### R21: the joint-space ablation at the pin on the other three robots (`pin2kq`), measured 2026-10-12

**Ran 2026-10-12 16:18-17:54 EDT** (Thomas, on R20's Panda ablation: *"Run the joint-space svgd
ablation on the other three robots"*), stage `SVGD_R21`, jobs 5900901 / 5900902 / 5900909 / 5900910,
four nodes, `DRAKE_ENV=crc PROCS=2`, the staged 86efbb0 (no restage; the manifest scp'd and
md5-checked). 12 runs / 288 items, the iiwa `n4`, the screw arm `p050 n6` and the soft PCS arm
`soft12 n6`, grasp / pose x native / paired, joint space at the pin with `svgd_kernel=q`, its own
items. Same token, pin and overrides as R20's Panda column, so every robot now has `pin2` (learned)
and `pin2kq` (joint space) at the pin on the same grid as its IPOPT twin.

**Joint space under svgd against joint space under IPOPT, same cells** (not a baseline; `clock` =
cells whose swarm stopped at the 180 s clock, `feas` = median feasible particles of 64 at stop):

| row | JS svgd | JS IPOPT | J+ / I+ | p | cost svgd / IPOPT | wall s svgd / IPOPT | outer | clock | feas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Panda grasp native / paired (R20) | 480 / 480 | 448 / 449 | 32 / 0, 31 / 0 | 5e-10 | 3.39 / 2.25 | 1.49 / 6.07 | 20 | 0 | 4 |
| Panda pose native / paired (R20) | 480 / 480 | 200 / 200 | 280 / 0 | 1e-84 | 4.42 / 7.71 | 0.83 / 0.53 | 14 | 0 | 16 |
| iiwa grasp native / paired | 480 / 480 | 453 / 453 | 27 / 0 | 1.5e-8 | 3.49 / 2.91 | 1.93 / 1.02 | 27 | 0 | 5 |
| iiwa pose native / paired | 480 / 480 | 312 / 312 | 168 / 0 | 5e-51 | 3.37 / 9.20 | 1.08 / 0.09 | 16 | 0 | 28 |
| screw grasp native | 463 | 442 | 34 / 13 | 3.1e-3 | 3.16 / 1.82 | 7.47 / 1.08 | 24 | 17 | 4 |
| screw grasp paired | 460 | 442 | 32 / 14 | 0.011 | 3.17 / 1.82 | 8.54 / 1.08 | 24 | 20 | 4 |
| screw pose native / paired | 477 / 477 | 320 / 320 | 157 / 0 | 1e-47 | 2.56 / 12.38 | 2.24 / 0.12 | 18 | 3 | 27 |
| soft PCS grasp native / paired | 480 / 480 | 472 / 472 | 8 / 0 | 0.008 | 0.83 / 0.24 | 3.95 / 3.19 | 12 | 0 | 13 |
| soft PCS pose native / paired | 480 / 480 | 334 / 334 | 146 / 0 | 2e-44 | 0.79 / 0.55 | 2.74 / 0.34 | 12 | 0 | 43 |

- **Joint space under svgd at the pin beats joint space under IPOPT on all sixteen rows**, 480 / 480
  on twelve of them. The screw arm is the one robot where it does not solve everything: 17 / 20 grasp
  cells and 3 / 3 pose cells per protocol run to the clock still infeasible (`constraint`, median
  `max_violation` 0.06-0.10 on grasp, 0.025 on pose; no resample, no re-check failure), fewer than
  the 24-cell cap threshold, so a result at the fielded clock. Those cells are also what lifts the
  screw arm's mean wall to 7.5-8.5 s (median 1.5 s, p90 3.0 s).
- **Joint space's svgd solutions are dearer than its IPOPT solutions on every grasp row** (1.2x iiwa,
  1.5x Panda, 1.7x screw, 3.5x soft PCS) and cheaper on every rigid-arm pose row (0.2-0.6x); on soft
  PCS pose 1.4x dearer.
- **The V1-pin picture is gone**: at kernel q / rho 1e3 / jitter (R1) joint-space svgd was 338 / 480
  on Panda grasp with 143 clock stops; at the pin the only clock stops anywhere are the screw arm's.
  Which of rho 1e5, the H / T metric and the native init did it is not separated.

**Learned under svgd against joint space under svgd, same cells** -- the comparison inside the
method, now on every robot (cost = median on the cells BOTH svgd arms and the IPOPT joint-space arm
solved):

| row | learned | joint space | L+ / J+ | p | cost L / JS | wall s L / JS | feas L / JS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Panda grasp native / paired | 480 | 480 | 0 / 0 | 1 | 3.37 / 3.39 | 2.47 / 1.49 | 58 / 4 |
| Panda pose native / paired | 480 | 480 | 0 / 0 | 1 | 4.51 / 4.42 | 1.01 / 0.83 | -- / 16 |
| iiwa grasp native / paired | 480 | 480 | 0 / 0 | 1 | 3.44 / 3.49 | 1.44 / 1.93 | -- / 5 |
| iiwa pose native / paired | 480 | 480 | 0 / 0 | 1 | 3.87 / 3.37 | 0.89 / 1.08 | -- / 28 |
| screw grasp native | 480 | 463 | 17 / 0 | **1.5e-5, learned** | 3.21 / 3.16 | 1.76 / 7.47 | -- / 4 |
| screw grasp paired | 480 | 460 | 20 / 0 | **1.9e-6, learned** | 3.21 / 3.17 | 1.95 / 8.54 | -- / 4 |
| screw pose native / paired | 480 | 477 | 3 / 0 | 0.25, tie | 3.58 / 2.56 | 1.28 / 2.24 | -- / 27 |
| soft PCS grasp native / paired | 480 | 480 | 0 / 0 | 1 | 0.90 / 0.83 | 2.58 / 3.95 | -- / 13 |
| soft PCS pose native / paired | 480 | 480 | 0 / 0 | 1 | 1.37 / 0.79 | 1.51 / 2.74 | -- / 43 |

- **Inside svgd: the learned arm wins 2 rows (screw grasp, both protocols), ties 14, loses 0.** The
  two wins are the cells the joint-space swarm leaves at the clock; the learned swarm has none.
- **Cost inside svgd is level on grasp** (within 0.07 on every rigid row; soft PCS 0.90 v 0.83) and
  favours joint space on pose (iiwa 3.87 v 3.37, screw 3.58 v 2.56, soft PCS 1.37 v 0.79; Panda level).
- **Wall inside svgd**: the learned arm is slower on the Panda (2.5 v 1.5 s grasp, 1.0 v 0.8 pose)
  and faster on the iiwa grasp, the screw arm and the soft PCS arm (where joint space's step is the
  231-position plant replay for both arms).
- **The joint-space swarm keeps 4-5 feasible particles of 64 at stop on the rigid grasp rows against
  the learned arm's 58** (Panda): the q-space kernel spreads a joint-space swarm off the manifold
  the way it did the learned arm's in R13, and the latent kernel is learned-only.

## The tuning push (branch `svgd-tune`, from 2026-10-14): strip, then replace with structure

**Thomas's end goal (2026-10-14): a high-performing AL-SVGD solver for the learned formulation that
leverages the special structure of the formulation, with a clear explanation for why each decision
helps.** The work pattern is his four phases: (1) a simple method working [R20]; (2) STRIP every
piece that does not earn its place; (3) replace what special treatment remains with formulation
insight; (4) add insight. Success is at the ceiling on every record row, so from here a round is
read on **cost on the cells shared with the pin, wall, residual at stop, feasible particles at stop
and outer steps**, and on each variant's own diagnostic.

Rules for every round of the push, fixed here before the first manifest:

- **Each variant pre-registers a mechanism, a diagnostic, a prediction and a decision rule.** The
  mechanism says why the change should move which metric; the diagnostic is a recorded per-cell
  quantity that the mechanism predicts; a variant whose outcome matches the prediction while its
  diagnostic does not is reported as *outcome without mechanism* (the stage-STEP lesson) and is
  never adopted on outcome alone.
- **Noise band**: "within noise" means the median cost on the cells shared with the pin within
  [0.95, 1.05] of the pin's (R19's seed-to-seed band is 3-5%), feasible particles within +-3, outer
  steps within +-2, wall within +-15%, residual in the same decade. Success is exact McNemar
  against the pin on the same cells, ties stated as ties.
- **Cap check** reads `timed_out` AND `stop_reason` (`wall_clock`, `step_cap`); >= 24 of 480 cells
  at a budget means no verdict. `n_recheck_failures` is printed per row.
- **A strip that is inert is removed, not defaulted off** (the seeding precedent), in its own
  commit after Thomas's word, with `tests/test_svgd_pin_and_gate.py` re-pinned.
- **Joint-space svgd stays an ablation** and is never tuned as a side effect; where a strip touches
  a setting the ablation arm leans on, one joint-space confirmation runs beside it.
- Structure (phase 3 / 4) enters the LEARNED arm's target only; the program's rows, bounds and task
  are untouched; every added term is a learned-only regulariser excluded from `reported_cost`.

Roadmap, each round its own pre-registration shown to Thomas before its manifest: **R22** (this
section) strips at the pin; **R23** ablates the three 2026-10-09 conditioning changes (natural
units, per-particle step, SVN with Levenberg delta), which were phase-1 exceptions and were A/B'd
only on 12 laptop cells at rho 1e3; **R24** tunes wall and cost at fixed success (N 16 / 32, rho
1e6, the stall patience, the dual step alpha, the c x z bandwidth rule); **R25** a float64 polish of
the returned particle (code; his call); **R26** the latent prior N(0, I) on z; **R27** the flow as an
exact density (the log-det, aimed at the runaway); **R28** the correction regulariser (w_c ladder,
weight 10 as control; the zero-centred q_c prior as the structural alternative to penalty and box).

### Stage SVGD_R22: stripping at the pin (pre-registered 2026-10-14, before any manifest)

**What runs.** The learned arm on the record's sixteen rows (Panda `n6`, iiwa `n4`, `screw7_p050`
`n6`, `soft12` `n6` x grasp / pose x native / paired), 480 cells, 180 s, seed 1, `SVGD_PINNED_V2`
plus exactly ONE override per variant, graphed, the lifted step cap, the clearance row
(`DRAKE_ENV=crc PROCS=2`), 24 shards; and ONE joint-space item set on the Panda's four rows. Every
variant is read against R20's `pin2` on the same cells (the VARIANT vs VARIANT block) and against
the robot's IPOPT twin (SVGD_R2 on the Panda, stage PAPER elsewhere). Native is the primary row,
paired the ablation.

| token | override | what it strips or tests | mechanism | diagnostic | prediction | decision rule |
| --- | --- | --- | --- | --- | --- | --- |
| `clipoff` | `svgd_multiplier_max=3e38` (`inf` does not survive `literal_eval` and would arrive as a string; 1e300 overflows the float32 scalar cast inside `torch.clamp` and errors every cell, found by the local smoke; 3e38 is float32-representable and `clamp` at +-3e38 is a no-op for any real multiplier) | the multiplier box (item 6) | PHR multipliers on the learned arm are O(1e2-1e4); the box caps only the few particles whose rows are still large after a dual step, which are the ones about to be resampled or already infeasible; it cannot move a feasible particle | `n_multiplier_clipped` (0 by construction), `lam_inf_max` / `mu_inf_max` (how far unboxed multipliers run), `n_resampled`, feasible count | ties on all 16 rows; cost within noise; `mu_inf_max` reaches 1e5-1e7 on a few particles without moving the selected one | inert on all 16 rows AND on the joint-space confirmation -> the box is deleted; inert on the learned arm only -> it stays as a stated setting of the ablation arm; a learned row where cost moves > 5% or the feasible count falls > 3 -> the box earns its place, explained from `multiplier_clips_by_group` |
| `clipoffjs` | `svgd_kernel=q` + `svgd_multiplier_max=3e38`, `--arms numerical`, Panda only (declared two factors, scoped as R20's `pin2kq`) | the box on the ablation arm, which leans on it (2.8e5 / 5.2e5 clips per grasp run, R20) | joint space has no latent to absorb a dual step; its multipliers on the `mug_xy` and `collision` groups climb to the box; unboxed they either keep climbing harmlessly or the swarm loses the grasp cells it only just holds at the pin | as `clipoff`, against R20's `pin2kq` | A: 480 / 480, `mu_inf_max` 1e5-1e7, cost within noise; B: grasp rows lose cells to the clock | decides whether `clipoff`'s deletion is licensed (above) |
| `drop2` | `svgd_resample=drop` | the redraw from the native draw (item 7); a dropped particle is frozen out of the swarm, the minimal handling of a non-finite particle | at the pin the swarm is native-drawn under BOTH protocols, so almost no particle runs away (R18: 0.05-0.08 redraws per particle on paired, ~0 native); the redraw that carried the jittered V1 swarm (R11) has nothing left to carry | `n_dropped`, `n_live` at stop, against `n_resampled` on `pin2` for the same cells; feasible count | ties on all 16 rows; median `n_dropped` <= 3 of 64; cost and feasible within noise; the native rows close to `pin2` cell for cell where `n_resampled` was 0 | inert on all 16 rows -> the redraw never earns its place at the pin: propose deleting the `native` and `keep` modes, the frozen drop staying as the sole non-finite handling (his approval); a paired row that loses cells -> the redraw stays and the loss is reported with `n_dropped` |
| `T0p1` | `svgd_temperature=0.1` | whether T earns a role: under the pinned H / T metric T sets ONLY the repulsion-to-drive ratio (bit-identical to the pin at T = 1) | 10x less repulsion moves the swarm toward kernel-off behaviour (R1K: keeps descending, lower cost, later stall stop); R14 found T = 0.1 ~ rho = 1e5 on `kz` at rho 1e3, so at rho 1e5 it may add little | `repulsion_ratio_median`, outer steps, `stop_reason` (clock stops as R1K), feasible count, cost | cost -0 to -10% on grasp, more outer steps, possibly clock stops on grasp; feasible count unchanged or lower | with `T10v2`: both within noise on all 16 rows -> **T is inert at the pin: T = 1 is fixed and `svgd_temperature` and `svgd_metric_temperature` are deleted** (the method then has no temperature); T = 0.1 lowers cost >= 10% at no loss of success or wall on >= 12 of 16 rows -> T earns a role, which is a tuning adoption AND phase-4 item 6 (T as coverage), so it goes to Thomas as a question, not fielded |
| `T10v2` | `svgd_temperature=10` | the other side of the bracket | 10x more repulsion: fewer feasible particles, higher cost, more steps (R5's `T10` under the old metric form was also a 0.1x Newton step; here it is the clean repulsion effect) | as `T0p1` | cost up, feasible count down, success unchanged | as `T0p1` |

**Not re-proposed, and why.** K (`svgd_inner_iters`): Thomas's 2026-10-11 ruling keeps the inner /
outer split as a parameter to be understood; a K ladder at the pin runs only on his word. The
gate-edge rule: fired 0 times in 9600 cells, costs nothing per step, is the exactness guarantee.
The time reserve (`svgd_time_reserve = 0.1`, 18 s of 180): an execution setting, and no learned cell
at the pin is within 129 s of the clock (the slowest cell is 51 s), so it never acts; it is left
alone. A float64 polish: none exists (removed 2026-10-09; `_select` only clamps onto the bounds),
so the residual question is R25 and needs code. The bandwidth floor (`svgd_bandwidth_floor = 0.05`)
is read from R20's recorded `bandwidths` on all sixteen rows rather than re-run: see "Strips decided
from the record" below.

**Allotment.** 68 logical runs (4 learned variants x 4 robots x 2 tasks x 2 starts + the Panda
joint-space set), 1632 items. The tool's figure (`gen_manifest.py --stage SVGD_R22 --allotment`,
5 s per learned cell + 180 s per item) and R20's measured scaling (1.0-2.6 s per learned cell, 6
node-hours for 480 items) are both stated in `cluster/SVGD_RUNBOOK.md`; the worst case puts every
cell at the clock. The only variant with a clock risk is `T0p1` (kernel-off behaviour). Items are claimed
in id order within the learned group (so per robot and row the four variants run as T0p1, T10v2,
clipoff, drop2, interleaved across the queue), and the joint-space items go last; the order sets
only when a verdict can first be read, not the total.

**Read from the record before the round, no compute** (R20's and R21's `summary.json`, every cell's
`record["svgd"]`, 2026-10-14; this sharpens the predictions above and was written before any R22
cell ran):

- **The bandwidth floor is NOT inert, so it is not stripped here; it joins R24's bandwidth-rule
  round.** `bandwidths` records the squared bandwidth `h = max(floor^2, med^2 / log N)`, so "at the
  floor" is `h <= 0.0025`. The z block never touches it on any row of either arm (median h 0.22-0.28).
  The c block does, on the POSE task only: Panda pose 9.2% / 10.8% of cells (native / paired), soft
  PCS pose **98.5% / 99.2%**; every grasp row and the iiwa / screw pose rows 0%; the joint-space `q`
  kernel never (h 1.2-12). Mechanism, to be tested in R24 rather than assumed: on the pose task every
  particle conditions on the same target pose, so the swarm's c-spread collapses and the c factor of
  the c x z product would become a delta without the floor -- the floor is what keeps the c block
  from switching the kernel off on those rows. That makes the c block's role on pose a question for
  the bandwidth round (per-block median against a shared bandwidth, or the z kernel alone on pose).
- **The resample fires on most learned cells at the pin, not "almost never".** Median redraws per
  cell (of 64 particles, cumulative over the solve): Panda 2-5, iiwa 8-9, screw 13-30, soft PCS 0-1;
  p90 6-57; cells with any redraw 81-100% on the rigid arms, 12-93% on soft PCS; native and paired
  alike (R18's "0.05-0.08 per particle" is the same 3-5 of 64 on the Panda). Joint space redraws on
  **no** cell of any row. So `drop2`'s prediction is sharpened: median `n_dropped` 2-9 on Panda /
  iiwa, 13-30 on the screw arm, 0-1 on soft PCS, and the screw grasp rows -- where a swarm would lose
  up to a third of its particles -- are where a `drop2` loss, if any, will show. The decision rule
  is unchanged.
- **The multiplier box is active on the learned arm, on every row group.** `lam_inf_max` reaches the
  1e4 box on the worst particle of the median cell on every iiwa, screw and Panda-pose row (Panda
  grasp 131-164, soft PCS 2-18); the clips land on every group the arm has (task rows, collision,
  joint limits, the z and c boxes, the trust row), 100-830 clipped entries per median cell. Joint
  space clips only its task rows and the collision row, 1e5-1e6 entries per run (and never
  resamples). So `clipoff` is a real test, not a formality: the prediction that the box does not move
  the selected particle stands, and `mu_inf_max` / `lam_inf_max` will say how far the unboxed
  multipliers run.
- `repulsion_ratio_median` at the pin: 0.005-0.014 on the rigid grasp rows, 0.0002-0.001 on rigid
  pose, 0.08-0.09 on soft PCS grasp -- the kernel's share of the step is small everywhere, which is
  the baseline `T0p1` / `T10v2` move by 10x each way.
- Every one of the 32 runs: 480 of 480 `converged`, `n_recheck_failures` 0 (the screw joint-space
  rows: 17 / 20 / 3 / 3 `wall_clock`, as R21 reported).

**R22 results (ran 14:12-19:40 EDT 2026-10-14 on four nodes, 5.5 h; jobs 5918035-38; 1631 of
1632 items, the screw pose native `T0p1` shard 6 of 24 lost to a one-off segmentation fault and
resubmitted as job 5924933, so that row reads on 23 shards until it lands).** Reader:
`scripts/report_svgd.py --stage SVGD --rows <robot> --variants pin2[,pin2kq],clipoff[,clipoffjs],drop2,T10v2,T0p1`.
Cost below is the median over cells of the per-cell ratio to `pin2` on cells both solved; wall is
the ratio of cap-clamped means; outer steps and feasible particles are medians, as differences from
`pin2`. **Cap check: 0 learned cells at the step cap or the clock on every row of every variant;
`n_recheck_failures` 0 on all 67 runs; residual in the same decade as the pin (2e-6 to 3e-5) on
every row. Every learned variant keeps the pin's verdict against the IPOPT twin on every row (8
wins / 8 ties, cell for cell 480 / 480 except `drop2` on the screw arm).**

| variant | success (16 rows) | cost ratio to pin | wall ratio | feasible of 64, delta | outer steps, delta | the diagnostic | verdict by the pre-registered rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `clipoff` | 480 / 480 on all 16 | 0.999-1.001 | 0.97-1.07 | -1 to +2 | -1 to +1 | `n_multiplier_clipped` 0 by construction; unboxed multipliers run to `lam_inf_max` 1.3e5-1.2e6 and `mu_inf_max` 1.2e6-1.2e11 on the worst particle without moving the selected one | **inert on the learned arm on all 16 rows**, as predicted |
| `clipoffjs` (joint space, Panda) | grasp **464 / 480** both protocols against `pin2kq`'s 480 (McNemar +0 / -16, p = 3.1e-5), 16 clock stops each; pose 479 / 480 (+0 / -1, p = 1), 1 clock stop each | 0.946 grasp (on the 433 shared cells), 1.000 pose | **4.6-4.7 grasp**, 1.55-1.58 pose | 3 v 4 grasp, 10 v 16 pose | +1 / 0 | `mu_inf_max` 1.6-1.8e8 grasp, 5-6e7 pose; the swarm loses the grasp cells it holds at the pin to the clock | **prediction B: the box is NOT inert on the ablation arm**, so by the rule it stays as a stated setting of that arm (the learned arm never needs it: no clip in 30,720 learned cells moved a selected particle) |
| `drop2` | 480 / 480 on 12 rows; **screw arm 477 / 477 / 477 / 478** (grasp native / paired, pose native / paired), every loss a swarm with every particle dropped (`stop_reason = all_dead`), McNemar vs `pin2` +0 / -3, -3, -3, -2 (p = 0.25-0.5 per row, 11 cells over the four rows) | 0.998-1.002 on 12 rows; screw grasp **1.037 / 1.018** | 0.93-1.02 | Panda / iiwa -2 to -6, soft PCS -1, **screw -16 / -17 grasp, -7 pose** | 0 to -1 | median `n_dropped` per cell: Panda 2-4, iiwa 7-9, soft PCS 0-1, **screw 20-22 grasp, 11-12 pose** (R20's `n_resampled` medians, as predicted) | **the resample earns its place, on the screw arm only, for the runaway it was built for**; inert elsewhere. The prediction's "ties on all 16" held on 12 and failed where the pre-round record read said a loss would show. Nothing deleted; R27's density term is the structural answer to the screw arm's runaway |
| `T10v2` | 480 / 480 on all 16 | 0.994-1.011 on 15 rows, screw grasp native **1.041** | 0.85-1.07 | -3 to +1 on 15 rows, screw grasp **-10** | -2 to +1 | `repulsion_ratio_median` 10x the pin's on every row (e.g. Panda grasp 0.16 v 0.016, soft PCS grasp 0.8 v 0.09) | within noise on 15 of 16 rows (screw grasp native at the band's edge on cost and outside it on feasible particles); the predicted cost-up / feasible-down shows only on the screw arm |
| `T0p1` | 480 / 480 on all 15 rows read (screw pose native pending) | 0.984-1.008 | **1.01-1.58**: outside +15% on 4 rows (Panda pose native 1.22 / paired 1.17, iiwa grasp native 1.16, screw pose paired 1.58) | -3 to +2 | 0 to +3 | `repulsion_ratio_median` 0.1x the pin's on every row; no clock stop anywhere (the kernel-off clock stops of R1K do not appear at rho 1e5) | never lowers cost by 10% on any row, so T does not earn a role; T = 0.1 only costs wall. The rule's letter ("both within noise on all 16") is missed by `T0p1`'s wall on 4 rows; its conclusion, that nothing is gained from T in either direction, holds. **Thomas ruled the removal 2026-10-14; `svgd_temperature` and `svgd_metric_temperature` are deleted in c308934** (the T = 1 step is bit-identical: two-cell reference and the pin test's pre-change npz) |

Read against the mechanisms: every outcome matches its pre-registered mechanism, none is "outcome
without mechanism". The box's learned-arm inertness is the dual-step argument (the clip touches
particles about to be redrawn); its joint-space activity is the missing latent (no z to absorb a
dual step, multipliers climb to 1e8, the swarm drifts to the clock). The resample's single
dependency is the screw arm's runaway population, which R20 had already measured as 13-30 redraws
per cell; `drop2` converts those redraws into dead particles and, three cells per row, into dead
swarms. The temperature moves only the repulsion share, by exactly its factor, and the repulsion
share at the pin (0.0002-0.09) is too small for a 10x either way to reach the selected particle.

**What R22 strips, and what stays with a reason**: the temperature (two options, gone); the
multiplier box stays for the ablation arm and is documented as inert on the learned arm; the
resample stays for the screw arm. CLAUDE.md's svgd section carries the verdicts.

### Stage SVGD_R23: the phase-1 conditioning exceptions at the pin (pre-registered 2026-10-14, before any screen cell was read)

**What it strips.** The three 2026-10-09 conditioning changes (the first section of this document:
rows in natural units, the Stein variational Newton metric with its Levenberg delta, and the
per-particle step). Thomas allowed them as phase-1 exceptions to "no special treatment", to be
ablated in phase 2; their only A/B is the 12-cell attribution at rho 1e3, delta 10, lr 1 (section
"3. The attribution" above), never at 480 cells or at the pin. The per-particle step has no
option of its own: under `gn` the step is a fraction of each particle's Newton step, under
`identity` it is `svgd_lr / ||H_i||_F` per particle, so a global step would need code and is not
proposed; `identity` is the ablation of the metric with the per-particle Frobenius bound kept.

**What runs.** R22's shape: the learned arm on the record's sixteen rows, 480 cells, 180 s, seed
1, `SVGD_PINNED_V2` plus exactly ONE override per variant, graphed, the lifted step cap, the
clearance row, 24 shards, read against R20's `pin2` on the same cells and the IPOPT twin. No
joint-space item: these settings are shared by both arms, the ablation arm is not being tuned, and
its attribution reading (joint space needs the metric, the units barely matter) stands as measured.

**A laptop screen gates the manifest, and it is part of the pre-registration.** Two of the five
variants are predicted dead at the pin and one is predicted structurally singular (below); a
480-cell row of clock stops carries no verdict (the cap rule) and costs ~48 node-hours per variant.
So every variant first runs on 12 Panda cells (targets 0-5, guess 0, grasp and pose, native, 180 s,
the pin, one process at a time on the laptop; tags `screen_R23_panda_n6_svgd_*_6_180_native_<tok>`
beside a `pin2` reference on the same cells). The rule, fixed before the screen ran: a variant that
solves **fewer than 6 of its 12 cells, or stops on the clock on 3 or more**, is DEAD AT THE PIN and
is reported from the screen (labelled preliminary, 12 cells); the rest go to the sixteen rows.
Thomas can override either way.

| token | override | what it strips | mechanism | diagnostic | prediction | decision rule |
| --- | --- | --- | --- | --- | --- | --- |
| `ident` | `svgd_metric=identity` | the SVN metric (item 4): the direction is applied with the per-particle scalar step `svgd_lr / \|\|H_i\|\|_F` instead of `svgd_lr (H_i + delta I)^-1` | at the pin the AL's curvature is `rho J^T J` on the row directions, `lambda_max` median 1e5-1e7 (R20's `lambda_max_median`), against O(1-10) on the objective-only directions (joint centering, the correction penalty) along the feasible set. A scalar step bounded by `1 / \|\|H\|\|_F` moves the row directions at the right rate and the objective directions 1e-5 to 1e-7 as fast: the swarm reaches the rows and then barely descends the objective, so it stalls at a high cost or runs to the clock. The attribution at rho 1e3 read 8/12 and 36 outer steps against 12/12 and 10; rho 1e5 widens the gap by 100x | outer steps, `stop_reason`, cost on shared cells, `frobenius_over_lambda_median` (1.00-1.07 at the pin: the Frobenius bound is tight, so the step is the Lipschitz step, not a loose one), `repulsion_ratio_median`, feasible count | DEAD on the screen (pose cells to the clock, grasp cells at a cost far above the pin); if alive, cost up and outer steps up >= 3x on every row | success lost or cost > 5% above the pin on any row -> **the Newton metric earns its place**, and the explanation is the 1e5-1e7 spread between row and objective curvature that rho = 1e5 creates and the metric absorbs (why "a larger rho costs no conditioning" held in the rho screen). Within noise on all rows -> the metric is unnecessary and is stripped (not predicted) |
| `tolunits` | `svgd_row_units=tolerance` | natural units (item 1): every AL row divided by `tol = 1e-4`, the pre-2026-10-09 path | Gauss-Newton is covariant under a row rescaling, so with an exact Newton step the units would be inert. The Levenberg delta is not covariant: dividing the rows by 1e-4 multiplies `rho J^T J` by 1e8, so delta 10 becomes a relative damping of 1e-7 and the float32 solve of a 1e13-curvature system (the attribution's (c): `lambda_max` median 1e12, every particle redrawn at every check, 0/12) returns garbage steps that the resample absorbs. At rho 1e5 the curvature is 1e15 | `lambda_max_median`, `n_resampled` (prediction: ~N per check), feasible count, `stop_reason` | DEAD on the screen: 0 feasible, every particle redrawn, clock stops | dead -> **natural units earn their place**, and the explanation is the units x delta x float32 interaction, not "ill-conditioning" in the abstract; alive (not predicted) -> the units are a free choice and the simpler one is reported |
| `lm0` | `svgd_gn_lm=0` | the Levenberg delta (item 4): the plain GN step | **the learned formulation is 6-fold redundant in (c, z) by construction**: `q = flow(c, z) + q_c`, so `dq / d(c, z)` is 7 x 13 and every row of the AL (task, collision, joint limits -- all functions of q) and the joint-centering cost see the 13 conditioning-and-latent coordinates only through it. `H` on that block is therefore rank <= 7 of 13 wherever the c-box and latent-trust rows are inactive (nearly everywhere: both are regions, rarely active), with a 6-dimensional null space along the fibre `{(c, z): flow(c, z) = const}`. Only the correction penalty's constant Hessian (10 I on q_c) and delta see the fibre. **Delta is what fixes the gauge along the formulation's own redundancy**; at delta 0 `solve_ex` on a singular float32 block returns a step along the fibre of size ~1/eps that the bound clamp and the resample absorb | `n_resampled`, `bound_clip`, `lambda_max_median` unchanged, feasible count, the `solve_ex` failure count (recorded as a zero step; shows as frozen particles) | DEAD or near-dead on the screen: resamples explode, feasible count collapses. (Joint space has no fibre -- its block is `J_q = I` -- so this is a learned-arm fact; it is not run) | dead -> **delta earns its place, and the reason is structural, not numerical**: it regularises the (c, z) gauge. That names the phase-3 replacement: a term that lives on the fibre -- the latent prior `w_z |z|^2 / 2` of R26 gives the z block the curvature `w_z I`, so R26 adds `w_z = 1` with `svgd_gn_lm = 0` as a declared two-factor variant (the structure replacing the damping), subject to Thomas's go for R26 |
| `lm1` | `svgd_gn_lm=1` | one decade below the pin | delta damps the directions whose curvature is below it: at the pin (delta 10) a direction of curvature 1 (joint centering, `w = 1`) takes `lr / 11` of its Newton step, the correction directions (curvature 10) `lr / 2`, the rows (1e5+) the full `lr`. Delta 1 lets the objective directions move 5x faster per outer step: more objective descent before the stall, so lower cost, at the risk of overshoot on the fibre (more resamples). The 2026-10-10 probe's "most feasible at delta 10" was at rho 10 and is void at the pin | cost on shared cells, outer steps, `n_resampled`, feasible count, residual | cost 5-15% below the pin on grasp (where the objective is the mug-frame cost and the self-motion is real), within noise on pose; outer steps unchanged or fewer; resamples up by a few per cell | `lm1` lowers cost >= 5% on >= 12 of 16 rows at no loss of success, wall within noise, resamples within +3 -> a tuning adoption that goes to Thomas as a question (phase-2 result, not fielded by me); within noise everywhere -> delta is inert over a decade and 10 stands with its structural explanation |
| `lm100` | `svgd_gn_lm=100` | one decade above the pin | the objective directions take `lr / 101` of their Newton step: slower objective descent, more outer steps to the stall or an earlier stall at a higher cost; the row directions unchanged (curvature 1e5+) | as `lm1` | cost 5-15% above the pin on grasp, outer steps up; feasible count unchanged (the rows are untouched) | as `lm1`; together they say whether cost is monotone in delta, which is the mechanism's signature |

**Strips decided without compute.** `svgd_row_length_scale` (1.0 m, "literally the raw rows"):
the default is the identity, so there is nothing to strip, and a ladder on it is R24 material at
most. The eager `lambda_max` diagnostic costs a `eigvalsh` per dual-update check and is a
recorded diagnostic, not a solver setting.

**Allotment.** 5 learned variants x 16 rows = 80 logical runs, 1920 items before the screen;
each dead variant removes 16 runs. The tool's figure (5 s per cell + 180 s per item) is 71 node-h
for the full set; R20's scaling (1-2.6 s per cell) makes the alive variants (`lm1`, `lm100`,
and whichever of the others survives) ~4 node-h per variant. Worst case per clock-bound variant 48
node-h, which is what the screen avoids.

**The screen (laptop, 2026-10-14 14:25-15:47 EDT; PRELIMINARY, 12 Panda cells, read after the
section above was committed as 4d3eaf8).** Targets 0-5, guess 0, native, 180 s, the pin plus one
override; `pin2` re-run on the same cells as the reference (6/6 and 6/6, the cost ratio's
denominator). Medians over the 6 cells of each task; cost ratio over the cells both solved.

| variant | grasp solved / clock | pose solved / clock | outer steps grasp / pose (pin 23.5 / 8) | wall s grasp / pose (pin 4.2 / 1.2) | feasible of 64 grasp / pose (pin 52.5 / 42) | resamples per cell grasp / pose (pin 2.5 / 3.5) | cost ratio to pin grasp / pose | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lm1` | 6/6, 0 | 6/6, 0 | 11 / 9 | 2.0 / 1.5 | 12.5 / 2.5 | 17.5 / 80 | **1.27 / 1.23** | alive -> cluster |
| `lm100` | 6/6, 0 | 6/6, 0 | **45** / 8 | **9.6** / 1.4 | 61.5 / 54 | 0 / 0.5 | 1.00 / 0.99 | alive -> cluster |
| `lm0` | **0/6, 6** | **0/6, 6** | 575 / 599 | 162 / 162 | 0 / 0 | **36,782 / 37,333** | -- | DEAD at the pin |
| `ident` | 6/6, 0 | 6/6, 0 | 7 / **133** | 1.4 / **25.2** | 20.5 / **1** | 0 / 0 | **1.25 / 1.24** | alive -> cluster |
| `tolunits` | **0/6, 6** | **0/6, 6** | 622 / 653 | 162 / 162 | 0 / 0 | **39,443 / 40,975** | -- | DEAD at the pin |

What the screen says against the predictions, before the 480-cell round:

- **`lm0` and `tolunits` are dead exactly as predicted, with the predicted signature**: every
  particle redrawn at every check (~60 redraws per outer step, N = 64), zero feasible particles
  ever, every cell to the clock, and the recorded `lambda_max` median is 1 with
  `||H||_F / lambda_max = sqrt(20) = 4.47` -- the identity the solver substitutes for a non-finite
  H. For `lm0` that is the singular (c, z) block: the plain GN solve of a rank-7-of-13 block in
  float32 returns a non-finite step. For `tolunits` it is the 1e15 curvature of the rescaled rows
  at rho 1e5. Neither goes to the cluster; both are reported from here (12 cells, preliminary) as
  "the setting earns its place" with its mechanism. **The structural reading stands: delta is what
  fixes the gauge along the learned formulation's own (c, z) redundancy**, which is the phase-3
  candidate R26 carries (the latent prior as the term that lives on the fibre).
- **`lm1` is alive but worse on everything except success, and the cost prediction had the wrong
  sign.** Cost +23-27%, feasible particles 12.5 / 2.5 of 64 against 52.5 / 42, resamples 7x / 23x
  the pin's. The mechanism's second half (overshoot along the fibre, absorbed by the resample) is
  what happened; the first half (faster objective descent, lower cost) did not show, because a
  particle that overshoots is redrawn rather than descending. So delta 10 is not slowing the
  objective; it is keeping particles on the manifold. The 480-cell round reads it under the
  pre-registered rule.
- **`lm100` is level on cost and success, with 2x the outer steps and wall on grasp, zero
  resamples and the most feasible particles of any variant** (61.5 / 54 of 64); its repulsion share
  is 0.28 on grasp against the pin's 0.01, because the damped drive leaves the kernel term a larger
  fraction of the step. The cost-up prediction did not show at 12 cells; the steps-up one did.
- **`ident` is alive, against the prediction of dead, and lands in the mechanism's other branch**:
  it stalls early at a high cost rather than running to the clock on grasp (7 outer steps, 1.4 s,
  cost +25%, 20.5 feasible), and on pose it crawls (133 outer steps, 25 s, ONE feasible particle,
  cost +24%). Both halves of the mechanism are visible: the scalar step reaches the rows and
  barely descends the objective. Under the pre-registered rule a cost > 5% above the pin on any row
  is already "the Newton metric earns its place"; the 480-cell round measures it on all sixteen.

Manifest `cluster/manifest_stageSVGD_R23.txt` (48 runs, 1152 items: `ident`, `lm1`, `lm100` on
the sixteen rows) was generated after this table was written; it is submitted on Thomas's word.

### Stage SVGD_R24: tuning wall and cost at fixed success (pre-registered 2026-10-14 evening, before any manifest)

**What it is.** Phase-2's last round: the settings that R22 / R23 leave standing, each pushed one
step from the pin to find out whether the pin's VALUE earns its place or only the piece. Every
variant is a single `--set` on the pin **`SVGD_PINNED_V3`** (= V2 without the removed
`svgd_metric_temperature=on`; bit-identical to V2, and identical on the staged cluster tree, where
that option's default is `on`), the learned arm, the record's sixteen rows, R22's shape, read
against R20's `pin2` on the same cells and the IPOPT twin. No joint-space items. Success is at
the ceiling, so every rule below reads cost / wall / feasible / steps / residual with the push's
noise band, and a variant that moves wall or cost by less than the band on 16 rows is reported
inert and the pin's value stands with its reason.

| token | override | what it tests | mechanism | diagnostic | prediction | decision rule |
| --- | --- | --- | --- | --- | --- | --- |
| `N16`, `N32` | `svgd_n=16`, `svgd_n=32` | the swarm size (64), the wall lever | the step is GPU-batched over N, but the collision row (clearances, 92-96% of a step at N = 64-256 before the batched row; still the largest share) and the dual check scale with N, so wall per outer step falls with N; the latent kernel keeps the swarm on the manifold at any N (R14 / R18: N does not add to `cz`); the selection picks the best feasible particle, so fewer particles = a smaller pool, cost up slightly and the paired start carried by fewer draws | ms per outer step, outer steps, feasible count (scaled: fraction of N), cost, `n_resampled` | wall 0.4-0.7x at N 16, 0.6-0.85x at N 32; cost +0-5%; success unchanged (R-early: N 1-256 within a cell at V1); feasible fraction unchanged | N 32 within the cost band and wall <= 0.85x on >= 12 rows -> a tuning adoption question for Thomas (the pin's N is 64 for no measured reason beyond "works"); cost outside the band on any rigid row -> 64 earns its place as the selection pool |
| `rho1e6` | `svgd_rho=1e6` | the penalty (1e5), one decade further along the lever that bought the record's cost (1e3 -> 1e5) | with the SVN metric a larger rho costs no conditioning (the metric absorbs `rho J^T J`), only churn: the PHR active set hardens, the dual step `alpha = rho` moves multipliers faster, particles overshoot the rows more and are redrawn more; the objective is weighed less against the rows, so cost may fall again (as 1e3 -> 1e5 did) or saturate | cost, `n_resampled`, residual (should fall ~10x), `lam_inf_max` / `mu_inf_max` (box at 1e4 bites harder), feasible count | cost within noise (the lever saturated at 1e5) or -0 to -5%; resamples up 2x; residual 1e-6 | cost <= 0.95x on >= 12 rows at no success / wall loss -> a question for Thomas (and R26-R28's structure then runs at 1e6); within noise -> 1e5 stands as the saturation point; resamples > +3 per cell or any row losing cells -> 1e5 earns its place |
| `pat2`, `pat3` | `svgd_stop_patience=2`, `=3` | the stall patience (5): of the 8-19 median outer steps per cell, 5 are the stall itself | the stop fires when the best feasible objective has not improved by `svgd_stop_rel` for `patience` checks (then the float64 re-check); a shorter patience saves (5 - p) outer steps on every cell and risks stopping during a plateau the swarm would have left | outer steps (prediction: exactly -3 / -2 medians), wall, cost (the price of stopping early), `n_recheck_failures` | wall 0.7-0.85x (`pat2`), 0.8-0.9x (`pat3`); cost +0-3%; success unchanged | `pat3` (or `pat2`) within the cost band on all 16 rows with wall <= 0.85x on >= 12 -> adoption question; cost outside the band anywhere -> 5 earns its place as the plateau guard |
| `alpha0p1`, `alpha10`, `alpha0` | `svgd_dual_lr=1e4`, `=1e6`, `=0` (the dual step; default `None` = rho, the exact PHR update) | the dual-ascent step: declared "pending the alpha ladder" on 2026-10-09 and never swept | `alpha = rho` is the method-of-multipliers step that makes the AL's stationary point the KKT point in one shot per check; alpha < rho converges the multipliers geometrically slower (more outer steps to the same residual), alpha > rho overshoots them (oscillation, more resamples); **`alpha = 0` is the pure quadratic penalty**, whose minimiser violates the rows by `|lam*| / rho` = 1e-3 to 1e-1 at the record's multipliers (1e2-1e4) against the 1e-4 gate, so it should fail most cells -- the multipliers earning their place in one row | outer steps, residual, `lam_inf_median` (0 by construction under `alpha0`), `n_resampled`, success | `alpha0p1`: steps +2 to +5, cost within noise; `alpha10`: resamples up, cost within noise or worse; **`alpha0`: success collapses on the task-row-bound cells (grasp first)** | `alpha0` losing cells on every row -> the multipliers earn their place (phase 2's cleanest "why"); `alpha0p1` / `alpha10` within noise -> alpha = rho is not a knob and stays implied (the option could go); either beating the pin on cost at no wall loss -> question |
| `floor1e-3`, `floor0p2`, `kz` | `svgd_bandwidth_floor=0.001`, `=0.2`, `svgd_kernel=z` | the c x z bandwidth rule and its floor (0.05), which R20's record shows binding on the pose task's c block (Panda 9-11% of cells, soft PCS 98-99%) and nowhere else | per block `h = max(floor^2, med^2 / log N)`; on the pose task every particle conditions near one target, the c-spread is below the floor and the c factor is held flat by it. `floor1e-3` lets the median rule run (a sharper c kernel on pose: more repulsion in c), `floor0p2` holds the c factor flat on far more cells (grasp rows too, where the c-spread is 0.1-0.3), and `kz` drops the c factor altogether (the latent kernel alone, R14's `kz` at the pin's rho) | `bandwidths` (the floor's binding fraction per block), `repulsion_ratio_median`, `feasible_q_spread`, feasible count, cost | `kz` ~ pin on the pose rows (the floor already makes the c factor ~1 there) and within noise on grasp; `floor1e-3` within noise on grasp, pose rows: repulsion share up, feasible count down a few; `floor0p2` within noise | `kz` within noise on all 16 -> **the c factor earns nothing and `cz` is simplified to `z`** (one bandwidth, no floor question on pose; a question for Thomas since `cz` was his sweep's winner at rho 1e3); the floor variants say whether 0.05 is a value or an accident |

**Not proposed.** `svgd_lr` (0.3): a damped Newton fraction, chosen by the 2026-10-10 screen at
rho 1e3; a ladder at the pin is a numerical knob with no structural question and R23's delta
ladder covers the same axis (how far the objective directions move per step). `svgd_stop_rel`
(1e-3): paired with the patience, one of the two is enough. `svgd_recheck_topk` (10): an execution
setting of the gate-edge rule. `svgd_inner_iters` (K): not re-proposed.

**Allotment.** 11 variants x 16 rows = 176 runs, 4224 items; the tool's figure (5 s per cell +
180 s per item) ~165 node-h, R22's measured pace (1632 items in 5.5 h on 4 nodes) ~14 h on 4 nodes;
`alpha0` may run its failing cells to the clock (16 rows x up to 480 x 180 s = 48 node-h worst),
which the cap rule then reads as a result at the fielded clock. Local smoke of every value never
run before (`svgd_n` 16 / 32 at the pin, rho 1e6, patience 2 / 3, the three alpha values, the two
floors) on 2 Panda pose cells each before the manifest; `kz` ran in R14. No restage: every option
exists in the staged 7812715 and V3 omits only a setting whose staged default is the pin's value.

### Stage SVGD_R26: the latent prior N(0, I) on z (phase 3 / 4; pre-registered 2026-10-14 evening, before any cell)

**Why this round needs no new code, and what that means.** The formulation already carries the
term: `ProgramOptions.latent_cost_weight = w` adds Drake's `QuadraticCost` `0.5 w |z|^2` on the
learned arm's latent (`LatentCost`, description `LatentRegularizerCost`), excluded by name from
`reported_cost`; the svgd batched program folds every quadratic cost into its target and its
constant Hessian into the SVN metric. So **`latent_cost_weight = 1` is exactly `|z|^2 / 2`, the
negative log density of the flow's latent prior N(0, I)**, up to a constant, inside the augmented
Lagrangian -- the structure the chart was trained to invert, entering the LEARNED arm's target
only (joint space has no z). It was swept once under IPOPT (CLAUDE.md: `latent_cost_weight` helps
the Panda at 0.1, hurts the iiwa; a settled negative there) and never under svgd. R25 (the float64
polish) is skipped in sequence because it needs code and Thomas's call; R26 runs first.

**What runs.** R22's shape (the learned arm, the record's sixteen rows, 480 cells, 180 s, seed 1,
`SVGD_PINNED_V3`, 24 shards), four variants read against R20's `pin2` (= `w = 0`) on the same
cells and the IPOPT twin; two are declared two-factor variants:

| token | override | the term | mechanism | diagnostic | prediction | decision rule |
| --- | --- | --- | --- | --- | --- | --- |
| `lp1` | `latent_cost_weight=1` | `+ \|z\|^2 / 2`: the exact log prior | the flow is accurate where the latent is typical (`\|z\| ~ sqrt(d)` = 2.65 Panda, 2.83 iiwa); the task rows leave the self-motion free and the kernel's repulsion spreads particles along it; the prior gives every particle an objective-side drive toward the typical set, so the feasible swarm sits where the chart is trustworthy and the selected particle is a configuration the chart represents well. On the objective it is one more quadratic, so cost (which excludes it) may rise slightly where the prior and the task cost disagree | `z_norm` of the selected particle and the swarm (recorded; R20's selected `\|z\|` is 2.2-2.6), `n_resampled` (a particle drifting toward the runaway has atypical z: prediction, fewer redraws), feasible count, cost, residual | `\|z\|` of the selected particle down 10-30%; resamples down on the rigid arms (Panda 2-5 -> 1-3, screw 13-30 -> under 10); cost within noise or +0-5%; success unchanged | resamples down >= 30% on the screw rows at cost within noise -> the prior does structurally what the resample does numerically, and it is the candidate to field (a question for Thomas, with R27's log-det as the stronger form); within noise everywhere -> inert, reported, not fielded |
| `lp10` | `latent_cost_weight=10` | a 10x prior, the dose | as `lp1`, with the prior now competing with the task cost: cost up, `\|z\|` down further, possibly feasible particles down where the typical set and the grasp manifold are far apart | as `lp1` | cost +5-15%, `\|z\|` 1.5-2.0, resamples down further | says whether the effect is monotone in w (the mechanism's signature) |
| `lp1notr` | `latent_cost_weight=1` + `latent_trust_region=None` (two factors) | the soft prior REPLACING the hard latent trust region (`\|z\| <= 4.0 / 4.3`, a stated deviation from eq. (6) that IPOPT keeps because a nonbinding row still shapes an interior-point trajectory; svgd has no barrier, so that reason does not transfer) | with the row gone the swarm's only hold on z is the prior (and the +-5 component box, a region of last resort); if the prior suffices, the formulation moves TOWARD eq. (6) at no cost | `z_norm` max over the swarm (does any particle leave the old radius?), `bound_clip` on the z box, feasible count, cost, resamples | within noise of `lp1` on all 16 rows; `\|z\|` max 3-4, no particle at the +-5 box | within noise of `lp1` everywhere -> the trust region earns nothing under svgd once the prior is in: a phase-3 replacement, Thomas's call since it changes the program the arm is measured on; a row losing cells or resampling more -> the row stays |
| `lp1lm0` | `latent_cost_weight=1` + `svgd_gn_lm=0` (two factors) | the prior REPLACING the Levenberg delta (R23's screen: delta 0 is dead because the (c, z) block of the GN Hessian is rank <= 7 of 13 along the formulation's redundancy fibre; the prior adds `I` on the z block, and no fibre direction is purely in c because `dq / dc` has full column rank, so the block becomes nonsingular by structure, not by damping) | if alive, the structural reading of R23 is confirmed: the formulation's own prior fixes the gauge the damping was fixing | `lambda_max_median` (no longer the substituted 1), `frobenius_over_lambda_median`, `n_resampled`, outer steps, cost | ALIVE on the screen; outer steps and cost near `lp1`'s; if resamples are high the undamped step overshoots on the c directions, which the prior does not touch | alive and within noise of `lp1` -> **delta is replaced by structure** (phase 3, the cleanest result this push can produce; fielding it is Thomas's call); alive but worse -> delta stays with the prior; dead -> the c directions need damping too and the structural claim is narrowed to the z block |

**A laptop screen gates `lp1lm0` only** (12 Panda cells, as R23's: targets 0-5 guess 0, native,
both tasks, 180 s; rule: fewer than 6 of 12 solved or 3 or more clock stops = dead at the pin,
reported from the screen). `lp1`, `lp10` and `lp1notr` go to the sixteen rows after a 2-cell
parse smoke each.

**Added after the parse smoke, before the manifest (2026-10-14 20:50 EDT): a fifth variant
`lp0p1`, `latent_cost_weight=0.1`.** The 2-cell smoke (Panda pose native, targets 0-1, 60 s)
already shows the dose at w = 1 is strong: the selected particle's `|z|` falls from 3.9 (the pin
on the same two cells) to 0.34 / 0.34 (`lp1`), 0.12 / 0.18 (`lp10`), 0.27 / 0.36 (`lp1notr`),
against the prior's typical `sqrt(7) = 2.65`; outer steps rise from 4-8 to 85-132 (`lp1`), 68-74
(`lp10`), 79-81 (`lp1notr`); cost 3.73 / 5.48 against 3.35 / 4.62; feasible particles 44-58 of
64; 2 / 2 solved everywhere. The mechanism is visible already: at a typical latent the prior is
worth `|z|^2 / 2 ~ 3.5`, larger than the task cost itself (2-5), so the target's descent along the
self-motion is dominated by the prior and the stall rule (on the full target) keeps the swarm
walking toward the prior's MODE at z = 0 rather than its typical set, at 10x the outer steps. The
prediction "`|z|` down 10-30%" was therefore wrong in size before the round; the round measures
the dose. The 0.1 rung is the one IPOPT's sweep found helpful on the Panda (CLAUDE.md) and the one
at which the prior (~0.35 at a typical z) is small against the task cost. Decision rules are
unchanged, read per rung.

**The `lp1lm0` screen (laptop, 2026-10-14 20:48-21:05 EDT, grasp half; the pose half ran after
this was written): DEAD at the pin.** 2 of 6 grasp cells solved and all 6 stopped on the clock
(556-765 outer steps, 35,000-49,000 redraws per cell, 0-1 feasible particles at stop, the
substituted identity H on 5 of 6 cells) -- the same signature as R23's `lm0` (0 of 6, 560-617
steps, 36,000-39,000 redraws) with two cells landing on a feasible particle before the clock. So
**the prior on z does not replace the Levenberg delta, and the structural claim is narrowed**: the
prior adds `I` on the z block, which makes the (c, z) block nonsingular in exact arithmetic (no
fibre direction is purely in c), but a fibre direction whose z component is small keeps a
curvature of order `|dz|^2`, and the undamped float32 step along it is still enormous. What delta
regularises is the whole soft subspace, not only its z part; a term that would replace it has to
act on c too (the conditioning pose has no prior in the formulation), so delta stays, with this as
its reason. `lp1lm0` is not run at 480 cells.

*The pose half, read after the paragraph above was committed (21:05-21:08 EDT): ALIVE.* 6 of 6
converged, 7-31 outer steps, wall 3.7-12.9 s against the pin's 1.2, feasible particles 20-40 of
64, redraws 5-26 per cell, `lambda_max` median 2.6e6-1.7e7 (a real Hessian, not the substituted
identity), `|z|` 0.7-1.4. So the prior DOES fix the gauge where the six pose rows pin `q` (the
soft subspace is then the (c, z) fibre alone, and `I` on z is enough for the float32 solve), and
does not where the grasp leaves a self-motion in `q` (the soft subspace gains the self-motion's
pull-back into (c, z), on which the prior has no hold and delta is the only damping). The variant
stays off the cluster by the rule (6 clock stops on grasp), and the narrowed claim reads: the
formulation's prior replaces the damping exactly on the task that has no redundancy left for it
to damp.

**Allotment.** 4 variants x 16 rows = 64 runs, 1536 items: the tool's ~60 node-h; by the smoke's
outer steps (10x the pin's on the w >= 1 rungs) the real figure is 8-12 h on 4 nodes rather than
R22's pace; no restage (`latent_cost_weight`, `latent_trust_region` and `svgd_gn_lm` all exist in
the staged 7812715).

**R28 (the correction regulariser) is pre-registered after R26 and R27 read**, since it runs on
their better term.

### Stage SVGD_R27: the flow as an exact density (phase 3 / 4; pre-registered 2026-10-14 night, before any cell; code ac4f5c2)

**The term.** `flow_logdet_weight = w` adds `w log|det dx/dz|` of the IKFlow map `x = flow(c, z)`
to the learned arm's objective (Drake cost `FlowLogDetCost`, excluded from `reported_cost` by
name; svgd cost kind `flow_logdet`, in the target and the drive). With `latent_cost_weight = w`
beside it the pair is `w (|z|^2 / 2 + log|det dx/dz|) = -w log p(x | c)` up to a constant: the
density of configurations the chart actually represents, the conditional the flow was trained to
model. Facts that shape the reading, found while building it (the scoping and the parity tests):
- The log-det is a free by-product of the forward pass (FrEIA returns it; IKFlow discards it), and
  its gradient is one extra row of the existing reverse pass, taken only when `w > 0` -- the extra
  row perturbs the q rows at 1e-16, so the flag keeps the pin bit for bit (pin test untouched; the
  term-off graphed smoke equals the pre-change reference on every field).
- **On the iiwa the map is 8-wide against 7 joints** (the eighth output is the training-time
  padding), so there the term is the joint density of `(q, pad)`, not `p(q | c)`; the Panda, screw
  and soft PCS maps are square. It is the density of `flow(c, z)`, not of `q = flow + q_c`
  (exact to `|q_c| ~ 2e-5` at the approved penalty), and not a density under `chart_error_scale`
  or `lift_q` (both refused with it).
- **The gradient is dominated by the conditioning pose**: on a sample point ~20 on c's position,
  1-2 on its orientation, 0.003-0.5 on z. The term is a conditional density optimised over a free
  c: it moves the conditioning pose to where the configuration is typical, more than it moves the
  latent. The latent half (`|z|^2 / 2`) is what R26 measures alone.
- **The term's size shifts the stall rule.** The log-det is -23 to -34 on the Panda against a
  task cost of 2-5, so with the term on the target is large and negative; the stall test
  `f_best < best_seen - rel |best_seen|` then has a threshold ~6-10x looser in absolute terms (and
  ill-defined where the target crosses zero). The stop rule is left exactly as it is (the term
  joins the target as `LatentRegularizerCost` does), and outer steps are read with this in mind:
  a stall reached in fewer steps than `lp1`'s is partly the rule, not only the term. Changing the
  rule (stalling on the shared objective) is a separate candidate, not folded in here.
- SVN metric: no block for the log-det (its curvature is indefinite and a second-order pass is
  2-3x a flow Jacobian per step); the `|z|^2 / 2` half enters through `LatentCost`'s constant
  Hessian as in R26. The exact-Hessian metric is a possible second rung, only if the term matters.
- The 2-cell smoke (Panda pose native, compiled + graphed, `w = 1` on both halves): 2 / 2
  converged in 31 / 33 outer steps (pin ~8, `lp1` 85 / 132), `|z|` 0.64 / 0.79 (`lp1` 0.34 / 0.34),
  cost 3.68 / 4.81 (pin 3.35 / 4.62, `lp1` 3.73 / 5.48), the negative log density at the returned
  point -32.8 / -33.7, 60-61 feasible particles, 0-2 redraws.
- The same two cells under IPOPT with the term on (the Drake cost path, 60 s): both feasible at
  the returned point, 559 / 384 majors to the clock, cost 6.43 / 4.79, `|z|` 1.3 / 0.78, density
  -31.2 / -34.2; 0 error cells. The term is a stated deviation from the measured learned formulation
  and is NOT fielded on the Drake solvers; the smoke only shows the shared cost evaluator runs.

**What runs.** R22's shape (the learned arm, the record's sixteen rows, 480 cells, 180 s, seed 1,
24 shards, the V3 pin), **on a RESTAGED cluster tree** (the term is new code; the restage waits
until R26 has drained so no queued item changes tree mid-round, and the pin test's bit-identity is
what licenses running R24 / R26's pending items on it if that ever happens). Three variants, read
against `pin2`, R26's `lp0p1` / `lp1` on the same cells, and the IPOPT twin:

| token | override | the term | mechanism | diagnostic | prediction | decision rule |
| --- | --- | --- | --- | --- | --- | --- |
| `ld1` | `flow_logdet_weight=1` | the log-det alone (the Jacobian-volume half, no prior) | high-gain regions of the chart are where `log|det dx/dz|` is large, so the term penalises a particle BEFORE its `q` explodes, where the resample acts only after (`|q|_inf > 10`): the structural answer to the screw arm's runaway R22 showed the resample still carries. Alone it also rewards low-volume regions, which with a free c can pull the conditioning pose rather than the latent | `n_resampled` / `n_dropped` (prediction: down on the screw rows, 13-30 -> under 10 per cell), `flow_neg_log_density` at the returned point (recorded for every learned cell, pin included), `z_norm`, `|q|_inf` excursions, feasible count, cost, outer steps | resamples down >= 30% on the screw rows; cost within noise or +0-5%; success unchanged; `|z|` near the pin's (the latent half is absent) | resamples down >= 30% on the screw rows at cost within noise -> the log-det does structurally what the resample does numerically (then R26's resample question is re-asked with the term in); within noise everywhere -> inert, reported, not fielded |
| `fd0p1` | `latent_cost_weight=0.1` + `flow_logdet_weight=0.1` (declared two factors) | `-0.1 log p(x | c)`, the full density at the dose R26's `lp0p1` carries | as `ld1` plus the prior's pull toward the typical set; read against `lp0p1` to separate the log-det's contribution from the prior's | as `ld1`, plus `z_norm` against `lp0p1`'s | within noise of `lp0p1` on cost and wall; resamples below `lp0p1`'s on the screw rows if the log-det does the work the prior does not | the pair of readings (`fd0p1` - `lp0p1`, `fd1` - `lp1`) attributes the effect to the log-det or to the prior; the candidate to field is whichever rung lowers resamples on the screw rows at cost and wall within noise -- a question for Thomas |
| `fd1` | `latent_cost_weight=1` + `flow_logdet_weight=1` (declared two factors) | `-log p(x | c)`, the exact density | as above at the smoke's dose; `lp1`'s 10x outer steps may shrink (the smoke: 31-33 against 85-132) partly through the stall rule's shifted threshold | as above, plus outer steps against `lp1` | cost between the pin's and `lp1`'s (+5-10%), `|z|` ~0.6-0.8, outer steps 3-5x the pin, resamples down on the screw rows | as `fd0p1` |

**Allotment.** 3 variants x 16 rows = 48 runs, 1152 items: the tool's ~45 node-h; by the smoke's
outer steps (4x the pin on `fd1`) ~5-8 h on 4 nodes. Restage first (`BRANCH=svgd-tune
cluster/make_staging_clone.sh`, after R26 drains), then the usual scp / md5 / submit.

## Kernel-space screen, stage A (laptop, 2026-10-11 evening; PRELIMINARY, 12 cells)

`scripts/svgd/screens/screen_kernel.sh`, run on Thomas's go-ahead: 3 targets x 2 guesses on each
Panda task, paired start, 20 s, N = 64, learned arm only, the fielded configuration except the kernel
space (`svgd_kernel`), autodiff collision row, laptop GPU. Every variant solves 12 / 12, so the screen
separates on the swarm, not on success. Columns: feasible particles at stop (median of 6 cells), q-spread
among them, outer steps, median wall, cumulative redraws per cell, median reported cost on the solved
cells (the same 6 cells for every row), median max violation.

```
space   task  feas  spr_q  outer  wall   redraw  cost   viol     stop
q       mug    7.0  6.300    14    4.1    30     7.75   4.3e-05  converged 6      <- the fielded kernel
q       pose  29.5  5.022    12    3.0    50     4.58   2.4e-05  converged 6
z       mug   33.5  4.771    14    4.2    64     3.53   1.2e-05  converged 6
z       pose  30.5  4.525    11    2.9    64     3.51   1.5e-05  converged 6
x       mug   29.0  4.541    14    4.2    64     3.79   5.3e-06  converged 6
x       pose  27.5  4.584    10    2.7    64     3.56   7.4e-06  converged 6
c       mug    4.5  5.259    16    5.2    64     4.81   3.1e-05  converged 6
c       pose  16.5  4.759    17    4.6    64     7.79   4.9e-05  converged 6
cz      mug   56.5  4.544    16    5.4    64     3.33   1.7e-05  converged 5, clock 1
cz      pose  42.5  4.520     9    2.6    64     3.63   1.7e-05  converged 6
qz      mug   22.0  4.716    13    4.0    31     4.31   7.3e-06  converged 6
qz      pose  38.0  4.558     9    2.4    64     4.13   7.4e-06  converged 6
pose    mug    2.0  4.873    17    5.5   139     4.54   6.9e-05  converged 6      (task-space kernel, grasp only)
none    mug   63.5  4.514    58   18.1    52     3.39   4.5e-07  clock 6          <- R1K's kernel-off
none    pose  53.5  4.559    10    2.6    42     4.15   1.3e-06  converged 6
```

**What it says, at 12 cells.** (i) The fielded q-space kernel is the odd one out: it spreads the
swarm in configuration space (q-spread 6.3 against 4.5-4.8 for every other space and for no kernel
at all), leaves 7 of 64 particles feasible on grasp and returns the worst cost of the table (7.75,
~2x IPOPT's ~4 on these rows) -- R1K's "kernel-on stall" is a property of WHERE the kernel acts, not
of having one. (ii) Repulsion in the flow's own variables -- the latent `z`, the product `c x z`, and
`x` (the whole decision vector, dominated by `z`) -- converges most of the swarm onto the feasible set
(28-57 of 64) at a cost of 3.3-3.8, below IPOPT's and equal to kernel-off's 3.39, **while stopping in
9-16 outer steps where kernel-off runs to the clock on every grasp cell** (58 steps, 18 s). The
feasible q-spread under `z` is the same 4.5-4.8 rad as with no kernel: the swarm's coverage of
distinct configurations comes from the start distribution and the flow's multimodality, and the
latent kernel preserves it without pushing particles off the manifold. (iii) `c` alone is the worst
space on the pose task (cost 7.8, 17 feasible): repulsion in the conditioning pose pushes particles
away from the one pose they must all condition on. (iv) The task-space kernel on grasp redraws 139
particles per cell (2x N) and keeps 2 feasible -- repulsion along the mug's free orientation drives
particles into the runaway regions. (v) `c x z` has the most feasible particles (57 / 43) and one
grasp cell at the clock: a swarm that keeps most particles feasible has more of them improving the
best objective, so the stall rule fires later -- the stop rule and the kernel space interact, which is
the interaction Thomas asked not to be assumed away. Stage B (temperature 1e-1 / 1e-2 / 1e-4 under
the fielded metric H and the swept H/T form, for q, z, x, cz, qz) follows; a cluster round on the
480-cell rows for the spaces that separate comes after it. **A kernel in `z` or `c` exists only on
the learned arm**; the joint-space svgd ablation can run the q kernel only, so the kernel space is a
setting of the learned formulation's solver, not a shared one.

## Kernel-space screen, stage B: temperature under both metric forms (laptop, 2026-10-11/12; PRELIMINARY, 12 cells)

Same 12 cells and settings as stage A; for each of the five spaces q, z, x, c x z, q x z the
temperature T in {1e-1, 1e-2, 1e-4} under the fielded metric form (`svgd_metric_temperature=off`,
the SVN metric is H) and under the swept form (`on`, the metric is H / T). Stage A is the T = 1
row of both forms, where they coincide. 60 runs; the screen was paused 21:50-00:05 for Thomas and
resumed on the missing cells only. Full table: scratchpad `kernel_stageB.txt`; the doc keeps the
reading.

**Under the fielded metric H, every temperature below 1 diverges, in every space.** 0 or 1 of 6
cells on all 30 runs, 2000-4000 redraws per cell (30-60 x N), every cell at the clock. This is the
reading from the code of 2026-10-11 02:15 confirmed: with the drive `-grad L / T` and no 1 / T in H,
the SVN step `H^-1 phi` scales as 1 / T, so T = 0.1 is a 10x step and the swarm leaves the basin on
the first check. **Under the fielded form T is a step multiplier, not a temperature**, and R5's
`T10` (a 0.1x step, 3-5x the outer steps, no extra spread) is the same fact from the other side.

**Under H / T, temperature is a well-posed knob and it is inert over two decades.** T = 0.1 and
0.01 solve 12 / 12 in every space, with feasible counts, q-spread among the feasible (4.5-4.8 rad in
every non-q space at every T) and cost within noise of T = 1:

```
space   T       feas (mug/pose)  cost (mug/pose)   outer (mug/pose)  stops
q       1       7.0 / 29.5       7.75 / 4.58       14 / 12           converged
q       1e-1   16.5 / 34.0       6.10 / 4.12       15 / 10           converged
q       1e-2   12.0 / 15.0       6.63 / 5.26       12 / 10           converged
q       1e-4    0.5 /  0.0       9.86 / 7.44       51 / 58           3/6, 2/6; clock 5 + 5, redraws ~2000
z       1      33.5 / 30.5       3.53 / 3.51       14 / 11           converged
z       1e-1   31.5 / 29.5       3.54 / 3.57       12 / 12           converged
z       1e-2   31.0 / 32.0       3.37 / 3.50       18 / 14           converged
z       1e-4    8.5 /  0.5       3.34 / 3.36       34 / 40           5/6, 3/6; clock 3 + 4, redraws ~1600
x       1      29.0 / 27.5       3.79 / 3.56       14 / 10           converged
x       1e-1   38.5 / 32.0       3.71 / 3.64       20 / 10           converged
x       1e-2   39.5 / 34.0       3.40 / 4.16       24 / 12           clock 2 (mug)
x       1e-4    5.5 /  1.0       3.60 / 4.71       28 / 28           5/6, 4/6; clock 2 + 2, redraws ~700-950
cz      1      56.5 / 42.5       3.33 / 3.63       16 /  9           clock 1 (mug)
cz      1e-1   57.0 / 39.5       2.96 / 4.15       13 /  9           clock 1 (mug)
cz      1e-2   57.5 / 38.5       2.98 / 3.63       40 /  9           clock 5 (mug)
cz      1e-4   43.5 / 16.0       2.12 / 4.66       36 / 18           clock 3 (mug); redraws 80 / 218
qz      1      22.0 / 38.0       4.31 / 4.13       13 /  9           converged
qz      1e-1   38.0 / 44.5       3.67 / 3.55       10 / 10           converged
qz      1e-2   47.0 / 46.0       3.67 / 3.53       16 /  9           converged
qz      1e-4   35.0 / 37.5       2.16 / 3.52       27 / 13           clock 1 (mug)
```

**What it says, at 12 cells.** (i) The "temperature = coverage" reading is not seen: lowering T by
two decades under a consistent metric changes neither the spread of the feasible set nor the
feasible count in any space. The repulsion-to-drive ratio T sets is not what limits coverage here;
the start distribution and the flow's multimodality are (stage A). (ii) At T = 1e-4 the single
spaces q, z, x break (3-5 of 6, 700-2000 redraws, clock stops): the repulsion is gone and the SVN
drive alone walks particles into the runaway regions, the N = 1 failure mode of R3 at N = 64. The
two PRODUCT spaces survive 1e-4 at 6 / 6 (c x z 43 / 16 feasible, q x z 35 / 38) and return the
lowest costs of the table (2.1-2.2 on grasp), so the second block's repulsion holds the swarm
together when the first's is too cold. (iii) **c x z has the most feasible particles and the lowest
grasp cost at every T, and it is the one space that runs to the clock**: 1 / 1 / 5 / 3 grasp cells
at T = 1 / 1e-1 / 1e-2 / 1e-4, with 57 of 64 particles feasible -- a swarm that keeps nearly every
particle feasible keeps improving the best objective in small steps and the stall rule (1e-3
relative over 5 checks) fires late or not at all. That is the stop rule interacting with the kernel
space, as flagged in stage A, and a 480-cell row will say whether it costs wall only or success.
(iv) **q x z is the robust space**: 6 / 6 at every T, 35-47 feasible, cost 3.5-3.7 at T >= 1e-2 and
2.2 at 1e-4, converged on every grasp cell but one; it carries the q kernel's geometric spread and
the latent kernel's convergence. (v) The q kernel improves under H / T at T = 0.1 (17 feasible,
cost 6.1 against 7 / 7.75) and is still the worst space at any temperature.

**For the cluster.** R13 (z, c x z, q x z, x at T = 1, the fielded metric; submitted 2026-10-12
00:00) settles stage A at 480 cells. R14 carries `kzHonT0p1` (z under H / T at T = 0.1) as the
metric-form row; this table says it should tie `kz`, which is the test that the H / T form is the
consistent one. If the stop interaction of c x z costs cells at 480, the fix is the stall rule's
window or threshold, not the space. **The H / T form should become the fielded metric**: it makes
T mean what the method says it means, and at T = 1 it is bit-identical to today's -- a definition
decision for Thomas, not a tuning one.

## Other robots: screw and soft PCS plumbing (2026-10-11)

`BatchedProgram.from_program` now takes all four programs of the screw arm (`screw7_p050`, chart
`n6`) and of the soft PCS arm (`soft12`, chart `n6`). `tests/test_batched_program.py` runs the same
oracle checks on them at the same tolerances: rows against `EvalBinding`, costs, gradients against
the AutoDiffXd chain, `init_from_q` against `SetStartFromQ`, and `to_drake_x` through `verify()`.
`tests/test_kinematics_from_plant.py` checks every body against `CalcRelativeTransform` on the
screw pose and grasp scenes (the `ScrewJoint` asserted present) and on the soft PCS scene.

- **Screw arm: the fielded path.** The plant walker replays the `ScrewJoint`, and `FrameChain`'s
  closed-form row Jacobians carry its `pitch / 2pi` column. So the step is the compiled / graphed
  split step, as on the Panda and the iiwa.
- **The native grasp `c` is the program's own `GraspCStart()` on every robot.**
  `_compute_native_c` (used by `svgd_paired_init = native` and by every resample) now calls it.
  Until this change the iiwa and screw grasp arms were replayed at `[mug centre, 0, 0, 0]`, the
  rule from before the settings unification, so the iiwa swarm's native draws started from a
  different `c` than `IiwaMugProgram` seeds. The Panda and soft PCS values are unchanged bit for
  bit. `tests/test_batched_program.py` pins `native_c()` and the solver's native draw against
  `SetNativeStart` on every learned program, to 1e-12.
- **Soft PCS arm: the split step since 2026-10-12** (the autograd / eager path before). Its 33
  sub-links are `QuaternionFloatingJoint`s. `KinematicTree` now replays those too (Drake normalises
  the quaternion, and so does the replay), so the arm needs no provider of its own. The
  configuration-to-plant map is the program's `TorchConfigToPlantQ`: the exact strain kinematics
  of `src/soft_arm/kinematics.py`, permuted into the plant's slots. It is checked against
  `ConfigToPlantQ` in float64 at construction. `FrameChain`'s closed-form Jacobians through a
  floating joint still refuse by name; the row Jacobians are instead taken w.r.t. the
  configuration (next paragraph), so `has_analytic_row_jacobians` is True and the solver takes
  the split step in all three modes (eager, compiled, graphed).
- **GVS** has no torch forward map. `GvsArmIKProgram.TorchConfigToPlantQ` raises by name, so
  `BatchedProgram.from_program` refuses the arm and it never reaches either step. Its torch port
  is deferred (Thomas, 2026-10-12).

**How the soft PCS arm's split step gets its derivatives (2026-10-12).** The rigid arms' plant
vector is the padded configuration, so a plant-space Jacobian is a configuration-space one by a
slice. The soft PCS arm's is 231 floating-body positions from 12 strains, so the Jacobians are
taken w.r.t. the configuration directly (`BatchedProgram._kinematics_cfg_jacobians`). The
composite `cfg -> TorchConfigToPlantQ -> FrameChain.poses` (pure forward tensor ops) is pushed
forward along the 12 coordinate directions in one vmapped `torch.func.jvp`. That gives
`J_plant = d q_plant / d cfg` (231 x 12), the task frame's `Jp` and, from the quaternion's
tangent, the world angular velocity's `Jw = 2 [-v, w I + [v]x] d quat / d cfg`. Stage 3 pulls
the collision row's plant gradient back through `J_plant`. The joint-limit rows and the
joint-centering cost are on the configuration already. The rigid arms do not enter this path.
`tests/test_svgd_soft_fastpath.py` pins it:

- On the screw arm and the Panda the forward-mode `Jp` / `Jw` equal `FrameChain`'s closed form to
  1.9e-15, so the `Jw` convention is the one the rigid step uses.
- On the soft PCS pose and grasp programs at B = 1 and 64, float64: rows identical to the old path
  (0.0), row Jacobians against autograd of the same rows 3.3e-16 to 1.0e-15 relative, the
  collision pullback 2.6e-18 to 4.4e-16, the cost gradient 0.0. On the learned arm the solver's
  `J_h` / `J_g` agree with the autograd target to 2.6e-16 to 3.2e-15. Frame Jacobians against
  central differences 1.5e-10 to 1.8e-10.
- float32 (reported): task-row Jacobians 2.3e-7 to 8.6e-7 relative. The collision pullback is 1.6e-9 to 1.9e-6
  on three batches and 4.4e-3 on one grasp batch of 64, where the float32 and float64 plant
  vectors sit on two sides of a Drake gradient switch.
- Steps, from the same native-draw swarm at the fielded pin (N = 16, c x z, rho 1e5, H / T): the
  split step against the old autograd step differs by 9.2e-12 to 1.6e-11 after one step in
  float64 and by 5.1e-4 to 1.5e-2 in float32. After 30 steps the differences are 2.3e-10 to
  8.7e-7 (float64) and 0.11 to 0.32 (float32). Every figure is within 3x of the old path's own
  sensitivity: the old path against itself from a start one ulp away moves by 1.2e-10 to 1.9e-10
  after one step and 3.2e-10 to 5.1e-7 after 30 (float64), 2.7e-2 to 0.11 and 0.10 to 0.19
  (float32). In float32 one GN solve at rho = 1e5 turns rounding into 1e-4 to 1e-1.
- Compiled and graphed: the three stages capture, the replay is bitwise equal to the compiled
  stages, and compiled agrees with eager to 4.2e-7 relative.
- EAGER mode is slower than the old path. The vmapped `jvp` is dispatch-bound on the GPU, so the
  eager split step takes 112-119 ms against the autograd step's 92-95 ms (N = 16, float64, the
  step test above, load < 6). The gain is in compiled and graphed mode, which the smokes run.

The rigid arms' split step is unchanged bit for bit. Thirty steps from the same swarm on six
rigid programs (Panda pose learned and joint space, Panda grasp learned `cz`, screw grasp learned
`pose`, screw pose joint space `pose`, iiwa grasp learned), each in eager and graphed mode,
return bitwise-identical swarms before and after the change.
- **Recording.** `record["svgd"]` now carries `step_mode` and `derivative_mode`.

**Smokes** (laptop, one process at a time under the 20G scope): the record's grid (`--targets 60
--guesses 8 --seed 1`) restricted to `--cells 0:0,0:1`, paired, `--wall-time 20`, `svgd_n=16`,
`svgd_kernel=q`, `svgd_compile`, `svgd_cuda_graph`, `flow_cuda_graph`, `--compile --config latent
--set correction_cost_weight=10.0 --scene hardened --shelf-inset 0.1 --target-placement shelf`,
and `--placement-point fingertips` on pose. Tags are `smoke_SVGDPLUMB_<robot>_n6_svgd-n16-graphed_<row>_2_20_paired`.

The table uses these abbreviations:

- **L / JS:** learned / joint space.
- **ms / step:** the svgd log's swarm seconds divided by its inner steps.
- **coll. share:** collision seconds over swarm seconds.

| robot / row | step | L success | JS success | L n_feasible | JS n_feasible | L stop | JS stop | L ms / step | JS ms / step | L coll. share | JS coll. share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| screw pose | graphed, closed form | 2/2 | 2/2 | 6, 7 | 1, 3 | converged x2 | converged x2 | 12.6, 7.8 | 11.3, 6.4 | 0.60, 0.67 | 0.79, 0.79 |
| screw grasp | graphed, closed form | 2/2 | 0/2 | 3, 1 | 0, 0 | converged x2 | wall clock x2 | 14.8, 8.8 | 8.7, 7.5 | 0.51, 0.72 | 0.77, 0.82 |
| soft PCS pose | eager, autograd | 2/2 | 1/2 | 2, 3 | 0, 13 | converged, wall clock | wall clock, converged | 114, 92 | 73, 70 | 0.62, 0.58 | 0.72, 0.72 |
| soft PCS grasp | eager, autograd | 1/2 | 2/2 | 2, 0 | 10, 10 | wall clock x2 | wall clock, converged | 126, 124 | 93, 80 | 0.71, 0.74 | 0.83, 0.81 |

**The soft PCS pose row was measured twice.** The first run was at load 21.8 with the laptop's
memory 52 of 62 GB used by other applications. It read learned 0/2 and joint space 1/2, every cell
at the wall clock, at 300-318 / 201-282 ms per step. The re-run at load 4.5 is the row above. A
2-cell smoke shows that the plumbing runs; it says nothing about success.

**What the soft PCS step costs.** One step is 70-126 ms at N = 16, against 6-15 ms graphed on the
screw arm. The collision row is 58-83% of it. Most of the rest is the eager autograd path through
231 plant positions.

**The soft PCS smokes on the split step (2026-10-12).** The same two rows, same grid (`grid_hash`
and `scene_fingerprint` equal to the SVGDPLUMB runs), same settings, after the change. Tags are
`smoke_SVGDPCS_soft12_n6_svgd-n16-graphed_<row>_2_20_paired`. Load 3.5-6.4 throughout.

| robot / row | step | L success | JS success | L n_feasible | JS n_feasible | L stop | JS stop | L ms / step | JS ms / step | L coll. share | JS coll. share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soft PCS pose (2026-10-11) | eager, autograd | 2/2 | 1/2 | 2, 3 | 0, 13 | converged, wall clock | wall clock, converged | 114, 92 | 73, 70 | 0.62, 0.58 | 0.72, 0.72 |
| soft PCS pose (2026-10-12) | graphed, forward mode | 2/2 | 1/2 | 4, 7 | 0, 13 | converged, wall clock | wall clock, converged | 70.7, 53.7 | 51.0, 49.1 | 0.95, 0.95 | 0.97, 0.97 |
| soft PCS grasp (2026-10-11) | eager, autograd | 1/2 | 2/2 | 2, 0 | 10, 10 | wall clock x2 | wall clock, converged | 126, 124 | 93, 80 | 0.71, 0.74 | 0.83, 0.81 |
| soft PCS grasp (2026-10-12) | graphed, forward mode | 2/2 | 2/2 | 3, 1 | 11, 11 | converged x2 | converged x2 | 87.1, 85.0 | 74.1, 69.9 | 0.97, 0.95 | 0.98, 0.98 |

A step is 1.45-1.71x faster on the learned arm and 1.14-1.43x on joint space (different swarms
per cell, so per-cell ratios are indicative). The collision row is now 95-98% of a step, so it
bounds the gain at this backend. The rest of a step went from 32-43 ms to 2.6-4.3 ms on the
learned arm and from 15-20 ms to 1.4-1.5 ms on joint space. A same-process timing on the learned arm separates the two
(`svgd_compile`, `svgd_cuda_graph`, 10 steps x 3 repeats from the initial swarm, the in-process
autodiff collision row on both paths):

| N | row | graphed split step: total / collision / rest, ms | eager autograd step: total / collision / rest, ms |
| --- | --- | --- | --- |
| 16 | pose | 126.3 / 124.6 / 1.7 | 149.5 / 117.7 / 31.8 |
| 16 | grasp | 71.3 / 69.5 / 1.8 | 104.0 / 70.6 / 33.3 |
| 64 | pose | 521.4 / 519.0 / 2.4 | 540.4 / 506.4 / 34.0 |
| 64 | grasp | 320.0 / 317.7 / 2.4 | 335.8 / 305.1 / 30.8 |

The part of the step that is not the collision row falls 13-18x, from 31-34 ms to 1.7-2.4 ms.

**At the branch defaults (N = 64, `svgd_kernel = q`, rho 1e3; the N / kernel overrides dropped,
tags `..._svgd-default-graphed_...`).** These are an outer step of 2.1-3.8 s on the learned arm
(1.8-3.5 s on joint space), 98-99% of it the collision row:

| row | L success | JS success | L n_feasible | JS n_feasible | L stop | JS stop | L ms / outer step | JS ms / outer step | L coll. share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pose | 2/2 | 1/2 | 21, 5 | 0, 43 | wall clock x2 | wall clock x2 | 2590, 2131 | 1837, 1893 | 0.99, 0.98 |
| grasp | 0/2 | 2/2 | 0, 0 | 1, 42 | wall clock x2 | wall clock x2 | 3098, 3764 | 3466, 2873 | 0.99, 0.99 |

These outer steps cannot be set against the cluster's 800-835 ms. The cluster rounds run the
batched clearance row. The laptop's pydrake has no `CalcRobotClearances`
(`tests/test_collision_clearances.py` skips), so these runs use the serial autodiff row, which is
4.3-8.1 ms per particle here. What carries over is the non-collision part: 31-34 ms per step
eagerly against 2.4 ms graphed at N = 64, i.e. about 0.3 s of each outer step at K = 10 against
0.024 s. A
cluster figure needs a cluster run of the clearance row. The success counts here are 2-cell
smokes at a 20 s clock and say nothing about success.

TODO (Thomas, 2026-10-11): rewrite the PCS and GVS forward steps in torch

- **PCS: done 2026-10-12.** Configuration-space Jacobians by forward mode through the exact
  strain map (above); the arm takes the split step.
- **GVS: deferred (Thomas, 2026-10-12).** The forward map is a JAX root-find with implicit
  differentiation (`src/gvs_arm/model.py`). It needs a torch port, or a dlpack bridge, before
  `BatchedProgram` can replay it.

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
