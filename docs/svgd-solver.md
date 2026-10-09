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

*(empty -- to be filled from `scripts/report_svgd.py` after the smoke runs; not before)*

## Selected variant

*(empty -- written here, with N and the reason, BEFORE any cluster manifest is generated)*
