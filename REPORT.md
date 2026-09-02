# Session report, 2026-09-01/02 -- the corrected-protocol measurement (`final5_*`)

Raw records in `results/*/benchmark/final5_*`, `ladder5_*`, `dose5_*`; full detail in
`CLAUDE.md`. This file is deliberately uncommitted.

**Status: complete.** All 26 runs finished overnight (queue ended 02:21), under a sleep
inhibitor, with no run reporting a failure and no suspend-bloated cell. That is the 16
finals (2 robots x 2 tasks x 2 protocols x 2 caps), 4 ladder rungs, 2 charted-bundle grids
and 4 dose-response points.

## What this run is, and why it supersedes everything before it

`final5` is the first measurement in which all of the following hold simultaneously:

- the learned arm is the draft's eq. (6) formulation, the task-parameterised invention
  having been removed outright;
- the conditioning-pose **and** latent regions are general linear constraints, so an exact
  paired start actually reaches the solver instead of being projected first;
- guesses are drawn **per target**, so no single unlucky draw can swing a whole column;
- a paired start an arm cannot represent is an **immediate, recorded failure** rather than
  a silent projection;
- the machine held a sleep inhibitor throughout.

15 targets x 4 per-target guesses = 60 cells per experiment, `--compile`, seed 0, success
verified from the returned point rather than from the solver's exit status. Joint space is
the comparison's target; the analytic columns are baselines.

## The headline: the draft's central claim holds, on both robots, at both caps

| Panda pose | learned | joint space | better/worse | p |
| --- | --- | --- | --- | --- |
| paired, 20 s | 41/60 | 29/60 | 21 / 9 | 0.043 |
| paired, 45 s | 43/60 | 29/60 | 22 / 8 | **0.016** |
| native, 20 s | **60/60** | 29/60 | 31 / 0 | 9.3e-10 |
| native, 45 s | **60/60** | 29/60 | 31 / 0 | 9.3e-10 |

| iiwa pose | learned | joint space | better/worse | p |
| --- | --- | --- | --- | --- |
| paired, 20 s | 40/60 | 39/60 | 12 / 11 | 1.0 (tie) |
| paired, 45 s | 43/60 | 39/60 | 13 / 9 | 0.52 (tie) |
| native, 20 s | **59/60** | 39/60 | 21 / 1 | 1.1e-5 |
| native, 45 s | **59/60** | 39/60 | 21 / 1 | 1.1e-5 |

The Panda result is decisive under both protocols and *strengthens* with budget (the
learned arm gains two cells at 45 s, joint space gains none); the iiwa wins native and ties
paired. The claim therefore depends neither on the initialisation scheme nor on the wall
clock, which is what running two protocols at two caps was for. Under `native` the learned
arm also wins while taking **fewer** iterations than joint space (26 against 41 on the
Panda, 40 against 31 on the iiwa).

## The grasp task: the cap buys real cells, and does not close the gap

All eight 45 s runs are done, on the same grids and seeds as the 20 s tables -- only
`--wall-time` changed, so they pair cell for cell.

| experiment | start | learned 20 s | learned 45 s | change | joint space |
| --- | --- | --- | --- | --- | --- |
| Panda grasp | paired | 35/60 | **46/60** | +11, -0 (p = 0.00098) | 56/60 |
| Panda grasp | native | 34/60 | **46/60** | +12, -0 (p = 0.00049) | 56/60 |
| iiwa grasp | paired | 12/60 | **20/60** | +8, -0 (p = 0.0078) | 59/60 |
| iiwa grasp | native | 7/60 | **15/60** | +8, -0 (p = 0.0078) | 59/60 |

**Every baseline is unchanged cell-for-cell at both caps, in all eight runs** -- `numerical`,
`analytic` and `analytic8` alike, same successes and same iteration counts. The extra budget
reaches only the arm that evaluates a network: expected, and a check that nothing else
differed between the runs. No cell is ever lost to the larger cap either.

**The deficit narrows without closing.** At 45 s the Panda learned arm lands on 46/60 under
*both* protocols against joint space's 56/60 (3 better / 13 worse, p = 0.021), with 13 cells
still at the cap. This contradicts the void task-param result, which claimed parity at 45 s:
the draft's own formulation does not reach parity on this grid.

The iiwa is starker -- 20/60 paired and 15/60 native against 59/60 -- with 39-44 of 60 cells
still at the cap, so 45 s does not bound that arm's asymptote. The gap is far too large for
the cap to explain, and the dose-response experiment below now rules out the explanation we
had been carrying for it.

The number to report for these rows is **iterations**, which are hardware-independent: the
learned arm averages 195-229 (Panda) and 299-364 (iiwa) against joint space's 102 and 143.
It is not losing merely because each iteration is expensive -- it is taking two to three
times as many steps, which points at the grasp constraint geometry seen through the flow
rather than at the flow's per-iteration cost.

## The finding I did not expect: chart accuracy is not what is wrong with the iiwa

`chart_error_scale = eps` adds a deterministic, smooth, seeded perturbation
`eps * sin(W [c; z] + b)` to the flow's output, degrading the chart's accuracy with the
scene, kinematics, solver, grid and start protocol all held fixed. Panda grasp, learned
only, 15 x 4, 20 s, paired, on the finals' own grid -- so `eps = 0` *is* the finals column.

| `eps` (rad) | nominal median chart error | success | at the cap |
| --- | --- | --- | --- |
| 0 (the Panda flow as trained) | 3.8 mm | 35/60 | 25 |
| 0.016 | ~12 mm | 34/60 | 26 |
| 0.032 | ~20 mm | 32/60 | 30 |
| 0.064 | ~43 mm | 22/60 | 40 |
| 0.128 | ~83 mm | 1/60 | 1 |

The iiwa's measured chart is 16.6 mm median / 64.5 mm p90 against the Panda's 3.8 / 9.4, so
it sits between the second and third rows. **At those doses the Panda still solves 34/60 and
32/60. The iiwa solves 12/60.** Degrading the Panda's chart to the iiwa's accuracy costs one
to three cells; the iiwa is twenty-three cells worse. The hypothesis carried since the
2026-08-28 chart-accuracy table -- that the iiwa grasp deficit is a statement about the
checkpoint's precision -- does not survive its own experiment.

Two secondary readings. Between `eps = 0` and `0.032` the curve is nearly flat and every
cell lost is lost to the *wall clock* (25 -> 30 at the cap) rather than to infeasibility: a
worse chart costs iterations before it costs solutions. And the `eps = 0.128` row is not
part of the curve at all -- 58 of its 60 cells fail as `unrepresentable_start`, because a
perturbation of 0.128 rad per joint exceeds what the `+-0.1` correction can absorb, so it
measures the correction box rather than the solve.

## The ablation ladder, for the first time unconfounded

Panda grasp, learned only, 60 cells, 20 s, paired, one grid shared with the finals. Both
earlier ladders were confounded -- ladder3 by the latent bounding box that silently
projected every start, ladder4 by that *and* by two rungs running the unauthorized task
parameterisation. Neither survives; this one is clean.

| rung | success | iters | `\|z\|` at start | vs the rung below | p |
| --- | --- | --- | --- | --- | --- |
| baseline (uncalibrated frame, no sharing) | 11/60 | 135 | **426** | -- | -- |
| + conditioning-frame calibration | **29/60** | 126 | 2.81 | 26 / 8 | **0.0029** |
| + shared flow evaluation | 30/60 | 134 | 2.81 | 1 / 0 | 1.0 |
| + latent trust region | 34/60 | 155 | 2.81 | 15 / 11 | 0.56 |
| the whole stack vs the baseline | 34 vs 11 | | | 28 / 5 | **6.6e-5** |

The stack is worth 23 cells and **18 of them are the conditioning-frame calibration**, the
only rung that is individually significant. Its mechanism is visible in the `|z|` column:
uncalibrated, inverting the flow at a pose 27 mm and 120 degrees from the frame the network
was trained on returns a latent of norm 426 -- and now that the latent region is a
constraint rather than a bound, the solver genuinely *starts* there instead of being clipped
to something arbitrary. That clip is exactly what hid this in the older ladders.

Sharing the flow evaluation is worth the one cell it must be (it returns bit-identical
values and derivatives; its only effect is throughput inside a fixed cap). The latent trust
region is +4 and still indistinguishable from noise -- its direction is now positive, where
under the void ladder4 it read negative, and neither measurement resolves it. It stays for
the reason you gave (IPOPT is poorly behaved on unbounded variables) and remains a stated
deviation from eq. (6) rather than a proven improvement.

## The charted-bundle grid: it is the analytic column it explains

A separate 15 x 4 Panda grid in which the shared `q_init` is rejection-sampled into the four
wide branch bundles the 4-branch analytic chart covers. The filter is applied once to the
shared guess list before any solve, so pairing across arms survives and nothing is scored --
but it changes the cells, so this table is **not** cell-comparable with the finals.

| experiment | learned | joint space | analytic4 | analytic8 |
| --- | --- | --- | --- | --- |
| Panda grasp, charted | 32/60 | **59/60** | **59/60** | **59/60** |
| Panda grasp, full grid | 35/60 | 56/60 | 52/60 | 57/60 |
| Panda pose, charted | **46/60** | 33/60 | 34/60 | 34/60 |
| Panda pose, full grid | **41/60** | 29/60 | 29/60 | 31/60 |

**analytic4 and analytic8 become identical once every start is charted** -- same successes,
same mean iterations (218 grasp, 37 pose), same `start_q_error` (1.8e-11). They must be: on
this grid every `q_init` lies in a bundle both charts cover, so the two arms are handed the
same point and solve the same problem. That is the filter checking itself, and it settles
what the `analytic4`/`analytic8` difference in the finals is: **entirely start coverage, not
solve difficulty.** The 4-branch chart goes 52 -> 59 and 29 -> 34 once its uncharted starts
are removed. Nothing about the near-limit bundles makes a solve harder; they are simply
configurations that arm cannot be given.

The learned arm moves in opposite directions on the two tasks (-3 grasp, +5 pose), which is
what one expects of a filter defined by another formulation's chart. Joint space gains 3-4
on both, so the charted population is mildly easier overall -- which is why this stays a
separate table.

## The defect this session found: the latent box was a variable bound

The conditioning-pose box had already been converted to a general constraint so a guess
could start outside it. The latent's own `+-5` box was left as a **bounding box**, and
`SetStartFromQ` clipped the inverted latent into it before the solver ran -- so this
projection was ours, not IPOPT's.

The flow is a bijection, so `flow(c, InvertFlow(q, c))` reproduces `q` exactly, but only at
the *unclipped* latent. The inversion routinely returns components past `+-5`, so the clip
moved the start by radians, the `+-0.1` correction could not close the residual, and the
cell was scored `unrepresentable_start`: an arm recorded as unable to represent a
configuration it represents exactly.

| iiwa pose, paired, 20 s | before | after |
| --- | --- | --- |
| learned success | 11/60 | **40/60** |
| cells scored `unrepresentable_start` | 49 | 0 |
| median `\|q(start) - q_init\|` | 3.79 | 0.0000 |

The arm starts at `\|z\| ~ 7.9` and the solver walks it to `\|z\| ~ 2.9` on its own, which
is precisely what a region being a constraint rather than a bound buys. The Panda grasp
gained 7 cells (28 -> 35); the Panda pose was unaffected, its inversion already landing
inside the box.

**Consequence: every archived paired learned column is void**, not only the grasp ones the
task-param removal had already voided. The iiwa pose paired numbers in the final3/final4
tables (16/30, 18/30) are this artefact.

The box now lives in one method, `LatentBoxConstraint()`. The first repair fixed
`generic_program.py` only, while `PandaMugProgram` and `IiwaMugProgram` override
`BoundingBoxConstraint` and carried their own copies -- the pose arms were fixed and the
grasp arms silently were not. The general rule, now the second instance of it: **a region
an initial guess may violate must be a general constraint, never a variable bound**, and
nothing may project a guess without recording that it did.

## The two analytic charts trade places by protocol

analytic8 beats analytic4 under `paired` (Panda grasp, 5 cells to 0, p = 0.0625),
**reversing the final4 finding** -- and the reversal is explained by the confound that run
had. With one guess shared across all targets, a single draw into the mirrored near-limit
bundle swung whole columns. With per-target guesses the 8-branch chart is simply the better
chart, and the 4-branch arm forfeits 6 grasp and 13 pose cells outright as
`unrepresentable_start` -- which the charted grid above now confirms is the entire effect.

Under `native` the ranking flips back (58 vs 53), because a uniform draw over eight branches
lands in the narrow near-limit bundles half the time, against roughly 10% of
configuration-space volume. Both directions are the same unbalanced-bundle pathology seen
from opposite ends -- a known pathology of this baseline, which by your ruling we measure
rather than repair.

## Harness self-checks, all passed

- **Joint space is cell-for-cell identical across the two protocols** (56/60 both ways on
  the Panda grasp, same grid hash). It must be: its native start *is* a random
  configuration. Any difference in the other columns is therefore attributable to their
  initialisation, not to the grid.
- **`start_q_error` is 0.0000 for every learned and joint-space arm, in every experiment**,
  including every ladder rung.
- **The correction stays off its box**: median `\|q_c\|` 0.045-0.075 against `+-0.1`, with
  0.00-0.10 of solutions on the box, at both caps. The learned arm is not degenerating into
  a reparameterised joint-space arm.
- **Reproducibility at the cap is +-1 cell.** `ladder5_latent` and the 20 s paired Panda
  grasp finals are the same configuration on the same grid and scored 34/60 and 35/60; the
  one differing cell hit the wall clock in both runs, at 264 and 286 iterations. Cells that
  exit at the cap are reproducible only up to machine load, so a one-cell difference
  anywhere in these tables is not by itself an effect.

## Known wart

On the iiwa, the mug and pose experiments report the **same** `grid_hash`
(`9f5953e3c669`), so the hash is not capturing the task. Nothing here cross-compares tasks,
but `collate.py --pair` would not refuse a mug-vs-pose pairing on that robot the way it is
designed to. Left unchanged so as not to alter the harness mid-queue; safe to fix now that
the queue is done, though doing so will change future hashes and break pairing against these
runs unless the task is folded in as a suffix rather than into the hash input.

## Still open

- **What the iiwa grasp row actually is.** Chart accuracy is now ruled out. The visible
  facts are 39-44 of 60 cells at the cap having taken 299-364 iterations, roughly 2.5x the
  joint-space arm, with the correction nowhere near its box -- so the next question is what
  those iterations are doing, not how accurate the network is.
- The residual 0.6% of configurations no branch of the 8-branch chart reproduces --
  deliberately future work; arXiv:2503.03992 is the suggested starting point.
- The iiwa checkpoint's provenance, still worth knowing (retraining is planned anyway), but
  no longer the explanation this row needs.
- Documenting the latent trust region in the paper draft as a stated deviation from eq. (6),
  kept because IPOPT struggles with unbounded variables.
