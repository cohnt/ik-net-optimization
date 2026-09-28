# The soft arm's two ladders: the chart ladder and the DOF ladder

Measured 2026-09-28 from stages `SOFTCHART` and `SOFTDOF`, both at the status-quo shape --
hardened scene, shelf-contained targets at the fingertips, 180 s, 480 cells = 60 targets x 8
guesses, seed 1, `--compile`, arms `learned,numerical`, both start protocols, **IPOPT only**
(the solver axis is closed; these questions are about the chart and the robot).

Jobs 5752615-18 (SOFTCHART, drained 06:19-06:48, rc=0) and 5752619-22 (SOFTDOF, drained
07:25-07:36, rc=0). 96 of 96 items produced a summary in each stage; 8 shards merged per
logical run, 480 cells x 2 arms each. 24 logical runs, 23,040 solves.

**Read the cap caveat at the bottom before quoting any row of either table.** One row reads as
a loss and it does not yet carry a verdict.

## The chart ladder: FLAT, and that is the result

`nb_nodes` 4 / 6 / 8 on the primary rung `soft12`. Verdicts by exact McNemar, learned against
joint space, from each run's own `_mcnemar`.

| row | learned | joint | L-only | J-only | p | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| n4 grasp native | 471 | 445 | 34 | 8 | 6.9e-05 | **learned** |
| n4 grasp paired | 463 | 445 | 31 | 13 | 9.6e-03 | **learned** |
| n4 pose native | **480** | 334 | 146 | 0 | 2.2e-44 | **learned** |
| n4 pose paired | 429 | 334 | 128 | 33 | 2.1e-14 | **learned** |
| n6 grasp native | 474 | 445 | 33 | 4 | 1.1e-06 | **learned** |
| n6 grasp paired | 468 | 445 | 31 | 8 | 2.9e-04 | **learned** |
| n6 pose native | 477 | 334 | 144 | 1 | 6.6e-42 | **learned** |
| n6 pose paired | 436 | 334 | 132 | 30 | 1.8e-16 | **learned** |
| n8 grasp native | 473 | 445 | 31 | 3 | 7.7e-07 | **learned** |
| n8 grasp paired | 459 | 445 | 32 | 18 | 0.065 | tie |
| n8 pose native | **480** | 334 | 146 | 0 | 2.2e-44 | **learned** |
| n8 pose paired | 426 | 334 | 121 | 29 | 1.5e-14 | **learned** |

**Learned wins 11 of 12 and ties 1. No losses.** The joint-space arm is identical across the
three rungs by construction -- same grid, same arm, no chart -- which is a harness check: 445
on every grasp row and 334 on every pose row.

**No rung separates from another.** Across n4 / n6 / n8: grasp native 471 / 474 / 473, grasp
paired 463 / 468 / 459, pose native 480 / 477 / 480, pose paired 429 / 436 / 426. The whole
spread is 3-10 cells of 480, inside the reproducibility band the record measures for rows with
a cap-bound population.

**The interesting part is that `n8` does not degrade.** Its gain ceiling `exp(2.4976 * 8)` is
4.8e8, well above the ~1e7 runaway band, and on the iiwa an above-ceiling rung is strictly
worse. Here it is indistinguishable from `n4` at 2.2e4. That agrees with the standalone
in-distribution pole screen, which reports `frac_gt_threshold` 0.0 at **every** checkpoint of
this robot's charts: there is no runaway population to select against, so the gain-ceiling
criterion is satisfied vacuously and the ladder has nothing to measure. The pre-registered
`n6` stands, and it stands for the pre-registration's reason, not because it won.

So the chart-selection rule's *premise* does not hold on this robot. That is a negative result
about the robot, not a refutation of the rule: the rule says to pick below the runaway band, and
every rung here is effectively below it.

## The DOF ladder: the baseline is what moves

The three rungs at the pre-registered `n6`. Total backbone length is held at 0.800 m, so the
rungs differ only in how much redundancy the same arm has. **Not paired across rungs** -- each
draws its own grid -- so the rungs are compared by rate, and McNemar is used only within a row.

| rung | grasp native | grasp paired | pose native | pose paired |
| --- | --- | --- | --- | --- |
| `soft9` (9 DOF) | 474 / 436 | 471 / 436 | 478 / **280** | 410 / 280 |
| `soft12` (12 DOF) | 474 / 445 | 468 / 445 | 477 / 334 | 436 / 334 |
| `soft16` (16 DOF) | 476 / 450 | 476 / 450 | 480 / 441 | **405 / 441** |

learned / joint space, of 480. Verdicts: **learned wins 11 of 12, loses 1.**

**The learned arm is at the ceiling and the baseline climbs to meet it.** On pose native the
learned arm is 478 / 477 / 480 -- flat, and at 480 twice -- while joint space runs
**280 -> 334 -> 441**, a 161-cell gain from redundancy alone. The DOF axis is therefore not
measuring the learned formulation at all on that task; it is measuring how much easier extra
redundancy makes the *baseline's* problem. The learned arm has no headroom left to show it in.

The same ordering holds on grasp, much more weakly: joint space 436 / 445 / 450, learned
474 / 474 / 476.

**The one loss is `soft16` pose paired, 405 against 441** (p = 3.6e-04), and it is the only
row in either stage where joint space wins. See the cap caveat -- it does not carry a verdict.

**`soft16` has torsion and `soft9`/`soft12` do not**, which is the rung difference that is not
just a DOF count: `kappa_z` makes the tip orientation free given the tip position. That is also
why `soft16`'s in-training pole callback reads clean (2.6-3.2, `frac` 0.0) from the first
checkpoint where `soft12_n6`'s ended at 2.4e8 -- the callback draws position and orientation
independently, which is out of distribution for a torsion-free arm and in distribution for this
one. Same callback, same constants; the rung whose kinematics match the callback's assumption
is the one that screens clean. Confirmation of the OOD diagnosis from the other direction.

## Reproducibility: `soft12_n6` was measured twice, in two stages, and agrees exactly

`soft12_n6` appears in both stages -- as the middle rung of the chart ladder and as the middle
rung of the DOF ladder -- generated separately, submitted as separate jobs, run on different
nodes. All four rows reproduce **exactly**: 474 / 468 / 477 / 436, with identical `a_only`,
`b_only` and p on every row. This is this robot's tightest reproducibility statement and it
validates the manifest generator, the sharding and the merger in one comparison.

It also shows what does NOT reproduce, and it is only the timing: ms/it moves 77.7 -> 86.2 on
grasp native between the two stages, which is node contention, not the measurement.

## The quartet, and the caveat this robot's runtime table carries

Per-row medians, `soft12_n6`:

| | learned | joint space |
| --- | --- | --- |
| iterations, grasp | 146-176 | **322** |
| iterations, pose | 25 native / 66 paired | 26 |
| ms/it, grasp | 78-91 | 22 |
| ms/it, pose | 44-144 | 14 |
| wall, grasp | 11.4-16.0 s | 7.2 s |
| wall, pose | 1.1-9.6 s | 0.36 s |

**The learned arm wins on ITERATIONS on the grasp task** -- 146-176 against 322 -- and still
costs more wall clock, because its iteration is 3.5-4x the price. And the per-iteration premium
is 3.5-6x here against the rigid arms' ~10-13x, which must be reported with its cause: **the
baseline got more expensive, not the learned arm cheaper.** Joint space places 231 floating-body
positions on this robot where on the rigid arms its `VarsToQ` is the identity, so it costs
14-25 ms/it here against ~2 ms there.

On `soft16` the baseline's iteration count rises to 414 on grasp, consistent with the same story.

## THE CAP CAVEAT: `max_iter` binds, not the 180 s wall clock

**`ProgramOptions.max_iter` is `None` on every row of both stages, so IPOPT runs at its own
default of 3000 iterations -- and on this robot that limit is reached inside the 180 s cap.**
Timeouts are near zero everywhere, which under the record's cap rule reads as "the cap is
innocent"; that reading is wrong here, because the budget that bound is the iteration count and
`timed_out` does not record it. `hit_iteration_cap` does, and the harness has always written it.

Where it binds, by stage:

| column | cells at 3000 iterations | median wall on those cells | of a 180 s budget |
| --- | --- | --- | --- |
| joint space, every grasp row | 23-37 of 480 | 49-56 s | **~70% unused** |
| learned, every pose *paired* row | 18-55 of 480 | 102-146 s | 19-43% unused |
| learned, grasp rows | 0-8 of 480 | 150-178 s | ~0-17% unused |
| learned, pose *native* rows | 0 | -- | -- |

On the losing row, `soft16` pose paired: **55 of the learned arm's 75 failures stopped at the
iteration limit**, at a median 123 s of the 180 s available, with median `max_violation` 1.73 in
normalized strain units (limit 1.0) -- a solver that had not converged, not a runaway, and not
a wall-clock timeout. Under the cap rule that row measures the iteration budget and **carries
no verdict until `max_iter` is raised and it is re-measured.**

The same applies, in the opposite direction, to every grasp row: the joint-space arm is cut off
at 3000 iterations after ~52 s, so its grasp column is not measured at the declared budget
either. That does not threaten the grasp verdicts -- the margins are 26-43 cells and the
cap-bound populations are smaller -- but it does mean the runtime table's flat, cheap baseline
is partly an artifact of stopping early.

**Pose native is clean on both arms** (`hit_iteration_cap` 0), and those are the rows carrying
the largest effects (146-198 cells, p down to 5.0e-60). Those verdicts are safe.

### This is not confined to the soft arm

The record's own IPOPT rows have it, and it was never checked, because the flag criteria read
`timed_out` alone:

| status-quo row | arm | cells at 3000 it | of which failures | total failures | median wall |
| --- | --- | --- | --- | --- | --- |
| iiwa n4 grasp native | learned | 30 | **29** | 33 | 103 s |
| iiwa n4 grasp paired | learned | 28 | **27** | 27 | 99 s |
| iiwa n4 grasp native/paired | joint | 11 | 10 | 38 | 12 s |
| panda n6 grasp native/paired | joint | 80 | 75 | 157 | 14 s |
| panda n6 grasp | learned | 3-8 | 3-7 | 4-9 | 128-142 s |
| all four pose rows | both | 0-4 | -- | -- | -- |

**On the two iiwa contained-grasp rows the record calls ties, 27-29 of the learned arm's 27-33
failures stopped at the iteration limit with ~43% of the wall clock unspent.** The record builds
its cap story on exactly that row -- "391 v 442 at 45 s with 88 timeouts; at 180 s with 0
timeouts it is 447 v 442, a tie -- the arms were tied all along". The 45 s -> 180 s move did not
remove the budget from those cells; it converted a wall-clock stop into an iteration-limit stop
and the rule stopped seeing it.

What this does and does not threaten, stated conservatively:

* **Untouched:** all four status-quo pose rows (`hit_iteration_cap` 0-4), which carry the
  record's largest and most significant effects.
* **Untouched in verdict, overstated in margin:** Panda contained grasp. 75 cap-bound baseline
  failures cannot close a ~150-cell gap.
* **In question:** the two iiwa contained-grasp **ties**, which are the record's only
  non-wins under IPOPT.

The fix is cheap and is what the status quo already intends -- set `max_iter` explicitly high
enough that the declared wall-clock cap is the binding budget. `ProgramOptions`'s own comment on
`snopt_major_iterations_default` says the intent outright: "3000 matches IPOPT's own default so
the wall clock is what binds". That premise is false on any row where a cell reaches 3000 inside
the cap.

**Thomas's ruling, 2026-09-28: do not re-run, on either the soft arm or the record.** The soft
tables stand as measured with the losing row reported as budget-bound and carrying no verdict, and
the status quo is left untouched with the caveat recorded against the two iiwa contained-grasp ties.
No compute was spent on this. Do not re-open either question.
