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

1. **Rows** are divided by `tol = acceptable_constr_viol_tol`, so `v_i = |(h~_i, max(0, g~_i))|_inf
   <= 1` is "feasible at the harness gate". Equalities stay equalities; the tolerance ladder is
   untouched.
2. **Merit**, per particle: the PHR augmented Lagrangian
   `L_i = f + lam_i.h~ + (rho/2)|h~|^2 + (1/(2 rho)) sum_j [max(0, mu_ij + rho g~_j)^2 - mu_ij^2]`;
   target density `exp(-L / T)`, `T = svgd_temperature` and `rho = svgd_rho` both fixed for the
   whole solve, `rho` ONE scalar shared by every particle. **The augmented Lagrangian is the
   FORMULATION and SVGD the optimizer running on it** (Thomas, 2026-10-09): one dynamics, no
   inner/outer structure.
3. **Coordinates**: particles move in `y = x / s`, `s` the region half-widths per block (learned:
   `c_position_slack` on the conditioning position, pi on its orientation, the latent trust radius --
   the +-5 box where none is set -- and `correction_bound`; joint space: half of each joint's range),
   read from the program's own options and bounds and recorded as `extras["region_scale"]`.
4. **Step**, the Tabor-Hermans form (arXiv 2506.00589, the "Q method"):
   `phi_i = (1/N) sum_j [K(q_j, q_i)(-grad_y f_j / T) + grad_{y_j} K(q_j, q_i)] - (1/T) grad_y (L - f)_i`
   -- only the objective's gradient and the repulsion are averaged over the kernel, each particle's
   OWN constraint gradient is added outside the average -- then the plain gradient step
   `y_i <- clamp_B(y_i + (svgd_lr / rho) phi_i)`: no momentum, no adaptation. The clamp's
   distance in y is recorded (`bound_clip`).
5. **Kernel**: RBF on the configuration q (flow output + correction on the learned arm, q itself on
   joint space), median bandwidth `h = med^2 / log N` floored at `svgd_bandwidth_floor`; the
   repulsion is pulled back to y through the flow's VJP. `svgd_kernel = none` drops both kernel
   terms (`phi_i = -(1/T) grad_y L_i`, N independent projected gradient descents, batched).
6. **Dual ascent on a fixed cadence, rho fixed.** Every K = `svgd_inner_iters` steps, on EVERY
   particle and unconditionally (no feasibility gate of any kind), the textbook dual-ascent step
   with step rho: `lam_i <- lam_i + rho h~_i`, `mu_i <- max(0, mu_i + rho g~_i)`, at the swarm's
   current rows; clipped to `+-svgd_multiplier_max`, the clipped entries counted. The multipliers
   are per particle, the penalty is not. Recorded per cell: the dual updates taken, the median and
   largest `|lam_i|_inf` and `|mu_i|_inf` at stop, and the clip count.
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
collision row is Drake's `MinimumDistanceLowerBoundConstraint`, batched through a process pool.

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
| `svgd_lr` | 1e-10 | 4: the step is `svgd_lr / rho` (see "The step size") |
| `svgd_kernel` | `q` | 5: `q` or `none` |
| `svgd_bandwidth_floor` | 0.05 | 5: floor on the median bandwidth |
| `svgd_constraint_inside_kernel` | False | 12: the literal form, an A/B |
| `svgd_rho` | 10 | 2, 4, 6: the penalty -- one scalar, fixed, shared; the dual step |
| `svgd_multiplier_max` | 1e4 | 6: multiplier clip |
| `svgd_inner_iters` | 10 | 6, 7, 8: K, the cadence of the dual step, resampling and the stop rule |
| `svgd_outer_iters` | 300 | 8: the step cap (`max_iter`, when set, also binds) |
| `svgd_resample_q_max` | 10.0 | 7: runaway threshold, rad |
| `svgd_stop_patience` | 5 | 8 |
| `svgd_stop_rel` | 1e-3 | 8 |
| `svgd_recheck_topk` | 10 | 9 |
| `svgd_time_reserve` | 0.1 | 8: fraction of the cap kept for selection and re-check |
| `svgd_warmup`, `svgd_warmup_iters`, `svgd_warmup_elite` | `none`, 10, 0.1 | 11: the CEM phase |
| `svgd_collision_workers` | None | execution: pool size (`cpu_count // PROCS` in a Slurm job, `min(that, 8)` elsewhere) |
| `svgd_compile`, `svgd_cuda_graph`, `svgd_pool_overlap` | False, False, True | execution: the split step eager / compiled / graphed, pool overlapped |

**The step size.** `svgd_lr` is the one number the method has no textbook value for. With the rows
scaled by `1/tol`, the penalty's curvature in y is `rho * lambda`, `lambda` the largest eigenvalue
of `J~ J~^T` over the active rows, so `eps = svgd_lr / rho` makes the stability limit
`svgd_lr < 2 / lambda`, independent of rho. Measured at tol 1e-4 on 64 random particles of each of
the four Panda programs (2026-10-09): median lambda 8e7-3.3e9, 90th percentile 1.1e10 on joint space
and 3.8e10-5.5e10 on the learned arm, where the latent ball's row is active and dominates; it scales
as `1/tol^2`. The default 1e-10 is `1/lambda` at the rigid rows' 90th percentile. On one learned pose
cell (a test cell, not the smoke grid) 1e-9 diverged within the first check, 1e-10 lost the jittered
initial swarm to resampling once and then descended, and 1e-11 was stable and slower; that probe
informed the choice and is not a tuning result.

**Memory.** Every collision pool is admitted by a guard (`src/svgd/collision_backend.py`): refused
if `workers x 0.75 GB > 0.5 x MemAvailable` (from `/proc/meminfo`; swap is never counted) or if the
process's live workers would exceed `SVGD_MAX_LIVE_WORKERS` (default `os.cpu_count()`). The solver
keeps ONE pool per process. Local runs set `svgd_collision_workers=4` and run under `systemd-run
--user --scope -p MemoryMax=... -p MemorySwapMax=0` (the smoke driver does both by default).

### Removed, 2026-10-09

- **Annealed repulsion temperature** (`svgd_repulsion_T0`, `svgd_anneal_frac`) and the
  **driving-force ramp** (`svgd_gamma_t`): T is fixed.
- **Gauss-Newton equality correction** (`svgd_gn_every`, `svgd_gn_delta`) with its
  **Levenberg-Marquardt damping, gain ratio** (`svgd_gn_lm*`) and the cost-weighted metric.
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
- **The cuSOLVER pin** (`preferred_linalg_library`): no batched linear solve remains. The
  recompile-limit raise and the inductor options stay (the profiler's many structures need them).
- **The per-particle penalty and every gate on the multipliers** (later the same day): the
  per-particle `rho_i` with Powell's growth test (`svgd_rho0`, `svgd_rho_growth`, `svgd_rho_gamma`,
  `svgd_rho_max`), the feasibility tolerance `eta_i` and its `rho0^-0.1` start, `v_prev`, and the
  `rho_median` / `rho_max` / `rho_at_stop` details. Thomas: the augmented Lagrangian is the
  formulation and SVGD the optimizer; the eta / Powell gating was the bilevel method of
  multipliers. `rho` is now one fixed scalar (`svgd_rho`) and the dual step is unconditional on a
  fixed cadence. (Its first 6-cell check, 9756ff4: every particle's rho reached the 1e6 cap within
  six checks and no multiplier ever updated -- learned 2/6 grasp, 0/6 pose.)

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
`graphed`) and `svgd_collision_workers=4` (an execution setting, not in the tag). The IPOPT twin
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
the same task, overlap on, Panda (the profiler builds Panda programs only); smoke ms/step is the swarm
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

- **Learned vs joint space under svgd**: exact two-sided McNemar on the paired cells; a verdict is a
  win, a tie (p >= 0.05) or a loss, and ties are stated as ties.
- **Learned under svgd vs learned under IPOPT**, on the same cells, exact McNemar -- the record's IPOPT
  column where its chart and scene are the run's, else an IPOPT column measured beside it.
- **Verdict flips against the record**: the svgd learned-vs-joint-space verdict beside the record's
  IPOPT verdict for the same row; every flip is listed.
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
  collision pool's share of the wall, `stop_reason` counts, warm-up/compile seconds, and the
  feasibility-agreement check, which must read zero.

## Smoke results

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
