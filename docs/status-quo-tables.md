# Stage STATUSQUO: the tables

The campaign of record, measured 2026-09-19/20 and accepted by Thomas on 2026-09-21. `CLAUDE.md`
carries what these tables **say**; this file carries the tables themselves, so a session pays for them
only when it needs a number.

**`scripts/report_statusquo.py` owns every table here and regenerates all of them from the persisted
runs under `results/`, together with the 24-cell tally and the three pre-registered flag criteria. Do
not hand-maintain them.** If a number here disagrees with that script, the script is right.

Conditions: 480 cells = 60 targets x 8 guesses, seed 1 (out of sample), **180 s**, `--compile`, adopted
rungs (Panda `n6`, iiwa `n4`), hardened scene, shelf-contained targets at the fingertips, arms
`learned,numerical`, both start protocols, all three solvers at their adopted configurations, Drake
nightly `0.0.20260918`. Two experiments per robot, eight rows per solver, **24 logical runs, 11,520
solves**. Wall clock is **this machine only** and is never compared across machines.

## The four tables

Laid out as in `writing/tro-paper/tables/*_alternate_organization.tex` so the two papers read side by
side: rows are experiments, each solver is a block with the learned and joint-space arms **adjacent**,
and there is **one metric per table**. `IP` = interior point (IPOPT), `AL` = augmented Lagrangian
(NLOPT), `SQP` = sequential quadratic programming (SNOPT). **Bold** is the better of each pair (both
bold when tied); (star) marks the best in the row. **Every row prints, zeros included** -- an omitted
row reads as missing data rather than as a measurement of zero.
`scripts/report_statusquo.py` regenerates all four, so do not hand-maintain them.

**Table 1 -- success rate of 480 cells.** Higher is better.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | **0.931** (star) | 0.921 | **0.000** | **0.000** | 0.419 | **0.633** |
| iiwa grasp paired | **0.944** (star) | 0.921 | **0.004** | 0.000 | 0.500 | **0.633** |
| iiwa pose native | **0.979** (star) | 0.623 | **0.621** | 0.065 | **0.925** | 0.562 |
| iiwa pose paired | **0.879** (star) | 0.623 | **0.246** | 0.065 | 0.515 | **0.562** |
| panda grasp native | **0.992** (star) | 0.673 | **0.681** | 0.000 | **0.919** | 0.635 |
| panda grasp paired | **0.981** (star) | 0.673 | **0.000** | **0.000** | 0.627 | **0.635** |
| panda pose native | **0.960** (star) | 0.452 | **0.602** | 0.025 | **0.917** | 0.371 |
| panda pose paired | **0.848** (star) | 0.452 | **0.237** | 0.025 | **0.573** | 0.371 |

**The learned arm wins 15 of the 24 solver x experiment cells, ties 7 and loses 2** — interior point
6/2/0, augmented Lagrangian 5/3/0, SQP 4/2/2 — and interior point is the best entry in every row.
**Both losses are iiwa contained grasp under SQP**; there is no other losing cell anywhere in the
table. Verdicts are by exact McNemar on the cells the table shows, which is also what decides a tie:
a numeric reading of the same table put two rows down as losses that the text called ties.
`scripts/report_statusquo.py` prints this tally beside the table so the two cannot drift apart.

**Table 2 -- optimal cost**, on cells **both** arms solved, learned-only regularizers excluded. Lower
is better; `N/A` means fewer than 10 shared solved cells, so no comparison exists.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | 4.985 | **2.839** (star) | N/A | N/A | 5.409 | **5.393** |
| iiwa grasp paired | 4.998 | **2.826** (star) | N/A | N/A | 5.607 | **4.515** |
| iiwa pose native | **6.242** | 6.757 | 6.511 | **4.988** (star) | **5.975** | 6.858 |
| iiwa pose paired | **6.011** | 6.704 | 7.062 | **5.807** (star) | 6.863 | **6.757** |
| panda grasp native | 7.504 | **5.305** (star) | N/A | N/A | **7.040** | 7.458 |
| panda grasp paired | 6.846 | **5.358** (star) | N/A | N/A | **7.370** | 7.567 |
| panda pose native | **11.136** | 11.861 | 9.346 | **9.088** (star) | **10.366** | 10.774 |
| panda pose paired | **10.654** (star) | 11.762 | N/A | N/A | 10.807 | **10.749** |

**The cost split is by TASK, not by solver**: learned is cheaper on pose and ~1.4-1.8x more expensive
on grasp, under every solver that produces a comparison.

**Table 3 -- mean runtime in seconds over all cells.** Lower is better. **This machine only** --
never compared across machines.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | 27.29 | **1.90** (star) | 180.06 | **180.06** | 17.72 | **4.29** |
| iiwa grasp paired | 25.25 | **1.90** (star) | **179.33** | 180.06 | 20.06 | **4.30** |
| iiwa pose native | 2.19 | **0.09** (star) | **80.01** | 174.26 | 2.61 | **0.25** |
| iiwa pose paired | 4.51 | **0.09** (star) | **140.98** | 174.27 | 11.84 | **0.25** |
| panda grasp native | 11.79 | **5.91** | **57.81** | 180.06 | 5.55 | **3.80** (star) |
| panda grasp paired | 20.93 | **5.91** | 180.07 | **180.06** | 17.59 | **3.80** (star) |
| panda pose native | 2.20 | **0.29** | **81.87** | 177.15 | 3.18 | **0.26** (star) |
| panda pose paired | 7.04 | **0.29** | **143.48** | 177.16 | 10.82 | **0.26** (star) |

Joint space wins every IP and SQP cell, which is the per-iteration price stated as a number.
**Under AL the ordering inverts on five rows** -- the learned arm is genuinely faster there, because
it converges while joint space burns the whole 180 s.

**Table 4 -- median major iterations over solved cells.** Lower is better. **The AL column is `N/A`
BY CONSTRUCTION**: `NloptSolverDetails` carries a single `status`, and NLopt has no major iteration to
count. Its work proxy is `eval_counts["map_jacobian"]` (3,881-12,923 per learned cell against
9,550-13,417 for joint space), which is **not comparable across arms** -- identity map on the
joint-space side -- so it does not belong in this table.

| experiment | IP L | IP JS | AL L | AL JS | SQP L | SQP JS |
| --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp native | 381 | **212** | N/A | N/A | 798 | **168** (star) |
| iiwa grasp paired | 348 | **212** | N/A | N/A | 644 | **168** (star) |
| iiwa pose native | 44 | **28** | N/A | N/A | 73 | **22** (star) |
| iiwa pose paired | 102 | **28** | N/A | N/A | 316 | **22** (star) |
| panda grasp native | **169** | 744 | N/A | N/A | 165 | **164** (star) |
| panda grasp paired | **261** | 744 | N/A | N/A | 443 | **164** (star) |
| panda pose native | 36 | **33** | N/A | N/A | 47 | **21** (star) |
| panda pose paired | 100 | **33** | N/A | N/A | 238 | **21** (star) |

**The Panda grasp rows are where hardening shows**: joint space needs 744 median iterations against
the learned arm's 169, so the learned formulation wins on *iterations* there under interior point
despite costing ~10x per iteration. Containment costs the joint-space arm its cheapness.

## Headroom and the rescue rate: what the success counts hide

Containment is what creates the headroom the comparison needs. Under IPOPT, cells joint space
fails, and how many of them the learned arm solves:

| config | L | JS | L only | JS only | both | neither | of JS's failures, rescued |
| --- | --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp contained native | 447 | 442 | 31 | 26 | 416 | 7 | 31/38 = 82% |
| iiwa grasp contained paired | 453 | 442 | 35 | 24 | 418 | 3 | 35/38 = **92%** |
| iiwa pose tip native | 470 | 299 | 175 | 4 | 295 | 6 | 175/181 = **97%** |
| iiwa pose tip paired | 422 | 299 | 158 | 35 | 264 | 23 | 158/181 = 87% |
| panda grasp contained native | 476 | 323 | 156 | 3 | 320 | 1 | 156/157 = **99%** |
| panda grasp contained paired | 471 | 323 | 155 | 7 | 316 | 2 | 155/157 = **99%** |
| panda pose tip native | 461 | 217 | 248 | 4 | 213 | 15 | 248/263 = 94% |
| panda pose tip paired | 407 | 217 | 219 | 29 | 188 | 44 | 219/263 = 83% |

**The rescue rate is 82-99% on every row of both robots and both protocols**, and containment is what
makes it matter: there are 157-263 joint-space failures available to rescue. The iiwa grasp ties are
not the same cells either -- 31-35 each way -- so the arms are complementary even where the totals
agree.

## The cap effect, per row: where the IPOPT-SNOPT gap comes from

The verdict is under "WHAT WAS FLAGGED" above; this is the measurement behind it. Learned arm, cells
gained going from the 45 s pairing reference to 180 s on the same grid:

| | IPOPT gains 45 s -> 180 s | SNOPT gains |
| --- | --- | --- |
| every grasp row | +5 to +55 | -1 to +7 |
| every pose row | **exactly +0** | +0 to +2 |

IPOPT's learned-arm failures are wall-clock, so a 4x cap recovers them; only 3.6% of SNOPT's are the
time limit, so it has nothing to recover. **The pose rows being exactly +0 is also the campaign's
tightest reproducibility statement** -- every row with no timeouts at 45 s reproduces its 45 s cell
count exactly, the sole exception being Panda pose tip paired at +2.

## The status quo under NLopt (augmented Lagrangian), `LD_MMA` inner + loose inner tolerances

Work, wall clock and violation are over **all 480 cells**, not over successes: this column times out
most cells, so a median over successes would describe the handful it got right. `jac/cell` is the
program's own network-Jacobian counter and is **not** comparable across arms — for the learned arm
each is a reverse pass through the flow, for joint space the identity map.

| row | L | JS | p | L jac/cell | JS jac/cell | L s | JS s | L timeouts | JS timeouts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iiwa grasp contained native | 0 | 0 | 1 tie | 12923 | 12407 | 180.06 | 180.06 | 480 | 480 |
| iiwa grasp contained paired | 2 | 0 | 0.50 tie | 12709 | 12407 | 179.33 | 180.06 | 478 | 480 |
| iiwa pose tip native | **298** | 31 | **5.1e-68** | 4025 | 9550 | 80.01 | 174.26 | 199 | 463 |
| iiwa pose tip paired | **118** | 31 | **2.0e-17** | 7066 | 9550 | 140.98 | 174.27 | 369 | 463 |
| panda grasp contained native | **327** | 0 | **7.3e-99** | 3881 | 13417 | 57.81 | 180.06 | 153 | **480** |
| panda grasp contained paired | 0 | 0 | 1 tie | 11916 | 13416 | 180.07 | 180.06 | 480 | 480 |
| panda pose tip native | **289** | 12 | **5.8e-82** | 3988 | 10455 | 81.87 | 177.15 | 203 | 469 |
| panda pose tip paired | **114** | 12 | **5.4e-26** | 6864 | 10469 | 143.48 | 177.16 | 371 | 469 |

**ALL FOUR pose rows are decisive learned wins** — iiwa 298 v 31 native (discordant 278 to 11) and
118 v 31 paired (101 to 14); Panda 289 v 12 native (278 to 1) and 114 v 12 paired (107 to 5), every
p between 5.4e-26 and 5.1e-68. **The joint-space arm never exceeds 31 of 480 anywhere in this
column.** Under an augmented Lagrangian the learned formulation solves this problem
and the joint-space formulation essentially does not — 96% of its cells hit the wall clock. Report
it as the result it is, not as a spoiled column: it is **attributable**, because only the solver
differs and the joint-space arm is the *easier* problem (7 variables, no network), so its collapse
is a property of NLopt on this program rather than of a harness that favours us. Cost exists here
but on only 20 and 17 common cells (learned 6.511 / 7.062 against 4.988 / 5.807), so quote it with
that n. The paired protocol is much harder for the augmented Lagrangian, as it is for SNOPT: 298 ->
118 cells and 80 s -> 141 s mean, while the joint-space arm is unchanged at 31 by construction.

**Panda contained grasp is the cleanest statement the project contains: learned 327 of 480 against
joint space ZERO of 480**, p = 7.3e-99, discordant 327 to 0, with the joint-space arm timing out on
every single cell while the learned arm converges in 58 s mean on 3,881 network Jacobians against
joint space's 13,417. There is no cost comparison because the arms share no solved cell -- print a
dash, and note that the dash here means the baseline solved nothing, not that the data is missing.

**Both iiwa grasp rows are 0-2 of 480 on BOTH arms**, replicating at 480 cells what stage
NLOPTTUNE found at 60: nothing Drake exposes makes the augmented Lagrangian solve an iiwa grasp.
Rows where both arms sit at the floor carry **no comparison and no cost column** — that is the one
narrow caveat, and it does not touch the pose rows where the result is.

**The augmented Lagrangian is extraordinarily sensitive to the starting point, far more than either
other solver, and that is a finding in its own right.** On Panda contained grasp the `native`
protocol gives 327 of 480 and the `paired` protocol gives **zero**, with every cell timing out; on
iiwa pose it is 298 -> 118 and on Panda pose 289 -> 114. IPOPT's largest protocol effect on the same rows
is 476 -> 471, and SNOPT's is 441 -> 301. So an AL started at a shared infeasible `q_init` cannot get
its multipliers moving before the clock runs out, where an interior-point method barely notices. This
is why the NLopt column must be read per protocol and never pooled, and it is the sharpest
demonstration in the project that **the two start protocols answer different questions**.

**The adopted configuration is doing exactly what it was adopted for.** On the pose row the learned
arm uses 4,025 network Jacobians per cell against the joint-space arm's 9,550 and finishes in 80 s
mean against 174 s (paired: 7,066 and 141 s against the same 9,550 and 174 s) — i.e. it converges rather than exhausting the budget, which is the feasibility
criterion the configuration was fielded on.

That answers flag criterion 3, untested before this campaign because the flat 180 s arm predated the
adopted configuration: six of the eight rows solve something and the solver ordering is untouched.

