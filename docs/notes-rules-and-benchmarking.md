# Campaign rules and the benchmark harness

The long form of a part of `CLAUDE.md`, moved here verbatim on 2026-10-18 to bring `CLAUDE.md` under
40k characters. `CLAUDE.md` keeps the verdicts and rulings; this file keeps the mechanisms, numbers and
history behind them. "Above"/"below" and section names refer to the old single-file layout, now split
across `docs/notes-*.md`. A reference elsewhere to "CLAUDE.md's ... section" means the matching notes file.

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
major limit -- 0.4-2.2 h at 1e5 (~3 callbacks per major) -- and in the record stages none has ever
been scored feasible, so no verdict depends on it, but the record's mean-wall columns carry the
overrun. **Since 2026-10-21 it is bounded from outside** (`cluster/run_items.sh`'s watcher,
`src/benchmark.py`'s persistence section; ops in `cluster/README.md`): a solve past 1.5x its clock
is SIGKILLed, the item relaunches and resumes from its per-cell JSONL, and the cell is recorded from
its last streamed iterate as `fail_reason="killed_overrun"`, `timed_out=True` (Thomas: the clock
stopped it), `hit_iteration_cap=False`, with `recovered_*`. Overrun tail across all local SNOPT
shards: ~50 cells 10-90 s past the clock, ~280 more than 300 s past; every cell >45 s past is
infeasible except the analytic arm's iiwa pose native (59,0), which converges 643-1443 s late in
ANALYTIC/C/T and becomes `killed_overrun` under the watcher (accepted, Thomas 2026-10-21).
**Stage OVR (2026-10-22) re-ran under the watcher every REMEASURE / PAPER / ANALYTIC* SNOPT shard
with a cell past 1.5x** (128 shards, folded into 38 runs; 204 cells killed): no success count or
verdict moved anywhere; time-matched <= 0.3 points; SQP budget stops per row learned 0-10 -> 0, joint
space 0-3 -> 0-2; the analytic (59,0) cell is now a failure on all three ladders. **The cap is a budget for the arm that evaluates a
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
| analytic, 8 branches (both robots) | 1e-11 | exact; a start inside the clip band is still matched (below) |
| analytic, 4 branches (Panda) | 1e-11, or several radians on ~13% of starts | the uncharted elbow half (A = -1), scored `unrepresentable_start` |

**A start inside the analytic clip band is matched, not unrepresentable.** Within ~sqrt(2 * clip) rad
of a chart seam the CLIPPED map returns a configuration ~0.01 rad from `q_init` while the variables
are exactly `q_init`'s pose, psi and branch; `run_grid` re-runs the closed form with the clip relaxed
(`UnclippedStartError`) and solves such a cell as an infeasible start on the reachability rows, as
`../codebase` does, flagging it `start_in_clip_band`. Before 2026-10-12 those cells were scored
`unrepresentable_start` unsolved -- the "0.6%" of `analytic8`'s paired columns.

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

