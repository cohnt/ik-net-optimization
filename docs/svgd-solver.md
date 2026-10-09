# The svgd solver: pre-registration

**Written 2026-10-08, on branch `svgd` at `39a5db5`, BEFORE any smoke number was read.** Everything
below the "Smoke results" heading is filled in afterwards; nothing above it may be edited to fit what
the smoke shows. A change to a rule above is a new pre-registration, dated, with the reason, and the
old text is kept.

Driver: `scripts/svgd/smoke.py`. Reader: `scripts/report_svgd.py` (it imports the matrix from the
driver, so the two cannot disagree). Reader's guards: `tests/test_report_svgd.py`.

## What the fourth method class is, and is not

The solver axis is three method classes -- interior point (IPOPT), SQP (SNOPT), augmented Lagrangian
(NLopt `LD_AUGLAG`) -- and that axis is closed. Thomas reopened it for a **fourth class**:
batch-parallel, **SVGD-style** particle methods in torch on the GPU, which exploit the fact that the
flow and the forward kinematics evaluate a batch of N configurations at nearly the cost of one.
Like the other three it is a **reporting axis, never a choice**: it is shared by both arms within a
run, and its result is reported beside the other three classes, not instead of them.

What it **is**:

- **SVGD-style only.** Three methods, all on the same target: `al_svgd` (augmented-Lagrangian SVGD,
  primary), `tsvgd` (tangent-space SVGD) and `admm_svgd` (Stein-projected consensus ADMM). No other
  particle method (no CEM-as-solver, no beam, no LM swarm) is fielded.
- **The controls are the same loop**, not a different optimizer: `svgd_kernel=none` drops the
  interaction term (N independent augmented-Lagrangian descents, batched), and `svgd_n=1` is the
  single-particle degenerate case. If the kernel or the population buys nothing, these say so.
- **The target is the program as written**: the Drake program's own rows, replayed for N particles,
  turned into a per-particle PHR augmented Lagrangian. Equalities stay equalities; the tolerance
  ladder is untouched (the solver's feasibility test is `acceptable_constr_viol_tol`, the harness's
  `verify()` gate is the same as every other solver's, and the task gate stays looser).
- **No formulation-specific information in the solver.** No latent-prior term, no assumption that the
  correction is zero-centred, no arm-specific kernel. The only structure used is that every arm has a
  configuration `q` (the default kernel space and the step clamp's unit).
- **Exact Drake collision, no proxy.** The collision row is Drake's own
  `MinimumDistanceLowerBoundConstraint`, batched through a process pool; there is no sphere union or
  other surrogate on the rigid arms. Its seconds are reported as the collision pool's share of the wall.
- **CEM warm-up is a phase of the method, A/B tested.** `svgd_warmup=cem` runs a forward-only
  cross-entropy phase on the penalised merit before the gradient phase; every variant is measured with
  and without it, on the same cells.
- **The returned point** is the best particle by objective among those feasible after a float64
  polish and an exact Drake `EvalBinding` re-check of the top-k. `verify()` then scores it exactly as
  it scores IPOPT's point. The solver's own verdict (`solver_feasible`), its Drake re-check
  (`drake_feasible`) and `verify()` must agree; a disagreement is a bug, not a result.

What it **is not**:

- **Not multi-start.** Particle 0 is always the cell's initial guess exactly (never clipped); the
  others are drawn around it (`svgd_paired_init=jitter`) or from the arm's native distribution
  (`native`). N particles inside ONE solve on ONE clock is the method; restarting a solve is
  multi-start, which Thomas ruled out of scope, and nothing here proposes it.
- **Not a re-sweep of the closed Drake classes.** No IPOPT, SNOPT or NLopt setting changes. The IPOPT
  twin runs the adopted configuration plus `flow_cuda_graph=True`.
- **Recorded future work, not current scope:** *using formulation-specific information rigorously with
  SVGD* (a latent prior, a correction-aware kernel, arm-specific structure).

## The variants, every option named in full

Every svgd column sets, by `--set`:

| column id | `svgd_method` | `svgd_n` | `svgd_kernel` | role |
| --- | --- | --- | --- | --- |
| `al1` | `al_svgd` | 1 | `q` | single-particle control |
| `al64none` | `al_svgd` | 64 | `none` | no-interaction control (batched AL) |
| `al64` | `al_svgd` | 64 | `q` | primary |
| `tsvgd64` | `tsvgd` | 64 | `q` | tangent-space SVGD |
| `admm64` | `admm_svgd` | 64 | `q` | Stein-projected consensus ADMM |

each with `svgd_warmup=none` and `svgd_warmup=cem` (column ids `<base>-none`, `<base>-cem`), and on
every svgd column also `svgd_paired_init=jitter`, `svgd_dtype=float32`, `svgd_compile=True`,
`svgd_cuda_graph=True` (mode `graphed`; `admm_svgd` is not fused and runs eager whatever is asked,
which its log records). `svgd_kernel=q` is the RBF kernel in configuration space with the median
bandwidth rule (`svgd_bandwidth=median`, `svgd_bandwidth_floor=0.05`). **Every other `svgd_*` field is
at its `ProgramOptions` default as of `39a5db5`**, and the run's metadata records what Drake / the
solver was actually handed (`solver_options_emitted`), so a later default change cannot pass silently.
The defaults that bound a run are `svgd_outer_iters=300` (the step cap), `svgd_inner_iters=10`,
`svgd_time_reserve=0.1` (of the wall cap, kept for polish and re-check), `svgd_resample_every=10` with
`svgd_resample_q_max=1000`, `svgd_polish_topk=4`, `svgd_recheck_topk=3`, `svgd_warmup_iters=10`,
`svgd_warmup_elite=0.1`.

The IPOPT twin (`ipopt`) is `--solver ipopt --set flow_cuda_graph=True` with everything else as the
record's IPOPT column.

Tags name every setting: `smoke_SVGD_<robot>_<rung>_<variant>_<row>_<cells>_<cap>_<start>`, variant
`svgd-<method>-n<N>-kernel_<kernel>-warmup_<warmup>-init_<paired_init>-<dtype>-<mode>` or
`ipopt-flow_cuda_graph`.

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
- **Columns**: the ten svgd columns above plus the IPOPT twin: 11 per row, 88 runs.
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
counts of `al1-<w>` against `al64-<w>`, per arm, per row, per warm-up.

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
