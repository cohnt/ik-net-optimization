# Stage GVS: the tables

Measured 2026-10-04/05 (jobs 5826941-44, 128 items, 4 nodes on `xeon-g6-volta`, **PROCS=2**, staged
commit `9e98604`), collected and merged 2026-10-05. `docs/gvs-arm.md` carries the robot, the
pre-registration and the pre-check; `CLAUDE.md` carries what these tables **say**.

**`python scripts/report_gvs.py stage` owns the table below and regenerates it from the merged runs
(`sc_GVS_*` under `results/`, including `results/_cluster_staging/*`). Do not hand-maintain it.** If a
number here disagrees with that script, the script is right.

**Conditions.** These match the record's except for the PROCS and the solvers:
- grid: 480 cells = 60 targets x 8 guesses, seed 1;
- **180 s** cap, `--compile`, hardened scene, shelf-contained targets at the fingertips;
- arms `learned,numerical`, both start protocols;
- **IPOPT and SNOPT only** (no NLopt: Thomas, 2026-10-02), each at its adopted configuration, with
  `max_iter` at the record's default;
- Drake nightly `0.0.20260918`;
- the exact forward model, SoRoMoX's GVS solved to static equilibrium;
- charts `gvs_pushrod9_o1__n6__step620000` and `gvs_pushrod9_o2__n6__step620000`.

**PROCS=2 differs from the record's 8** (see `docs/gvs-arm.md`, contention), so runtime columns are not
comparable to any other robot's, which they never are anyway. The two rungs draw **different targets**,
so they are compared only by the target-level bootstrap at the bottom, never by McNemar.

## Acceptance checks (all pass)

- **Cells.** Each of the 16 logical runs merges to 480 cells per arm, with 0 `error` cells on either arm.
- **Grid.** `grid_hash` is identical to stage GVSJS's on every row: `bdc69d9a378d-mug`,
  `b0083a15b347-pose`, `3bc41879586c-mug` and `0f8c95b12130-pose`.
- **Joint space reproduces cell for cell.** Compared native against paired, and against GVSJS (a
  separate stage, joint space alone), every iteration difference is on a cell that ran to the 180 s
  clock in at least one run. **There are zero differences on cells that converged.** One verdict
  differs: o1 IPOPT grasp, cell (30, 0), which timed out in all three runs, stopped at iteration 2779
  against 2775/2793, and landed feasible in one. Hence 455 native against 454 paired.
- **Start protocol.** `median_start_q_error` is 0.0 on every paired learned row.

## The table

Columns, all from the reporter:
- **L / JS:** successes of 480.
- **p:** exact McNemar. A tie is p >= 0.05.
- **it / s:** median iterations and median wall seconds, each over the arm's own successes.
- **it L/JS:** pre-registered. The median over cells BOTH arms solved of the per-cell iteration ratio.
- **cost:** median `reported_cost` on cells both arms solved.
- **rescue:** the joint-space arm's failures that the learned arm solved.
- **TM JS:** pre-registered. Time-matched joint space; see `docs/gvs-arm.md`.
- **L-TM:** the learned success rate minus TM JS, in points.
- **starv:** the share of cells where the budget exceeds all 8 joint-space solves, so the true
  multi-start figure is higher.
- **any8:** the joint-space ceiling.
- **ms/ev:** ms per network-and-map evaluation, `solver_seconds / map_jacobian`.
- **to/ic:** `timed_out` / `hit_iteration_cap`.

```
=== STAGE GVS -- 480 cells, 180 s, seed 1, exact forward model
Verdicts by exact McNemar; cost on cells both arms solved (reported_cost); 'TM' = time-matched joint space (pre-registered).
row                                     L   JS        p     verdict  L it JS it it L/JS  L cost JS cost    L s   JS s   rescue     L %  TM JS  L-TM starv  any8 ms/ev L    JS     x L to/ic JS to/ic
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
o1 ipopt grasp native                 463  455     0.26         tie   114    80    1.22    0.80    0.35   12.0    3.0    23/25   96.5%  74.7% +21.7   14%  100%    33.1  22.6  1.46    17/0      6/0
o1 ipopt grasp paired                 465  454    0.099         tie   103    80    1.32    0.78    0.35    8.9    2.9    24/26   96.9%  70.7% +26.2   11%  100%    33.3  22.6  1.47    15/0      6/0
o1 ipopt pose native                  480  311  2.7e-51     learned    18    29    0.59    1.06    0.71    0.9    0.8  169/169  100.0%  37.8% +62.2    1%  100%    34.5  23.3  1.48     0/0      0/0
o1 ipopt pose paired                  382  311  6.8e-08     learned    56    29    1.79    0.84    0.73    4.4    0.8  122/169   79.6%  88.0%  -8.5   32%  100%    33.2  23.1  1.44    32/0      0/0
o2 ipopt grasp native                 467  447   0.0037     learned   106    75    1.34    0.80    0.35   14.3    3.9    32/33   97.3%  71.5% +25.8   15%   98%    42.8  32.2  1.33    12/0     15/0
o2 ipopt grasp paired                 455  447     0.32         tie   112    75    1.32    0.76    0.35   13.1    3.9    29/33   94.8%  70.8% +24.0   13%   98%    43.0  32.2  1.33    27/0     15/0
o2 ipopt pose native                  480  315  4.3e-50     learned    19    29    0.63    1.06    0.73    1.2    1.2  165/165  100.0%  38.2% +61.8    1%  100%    44.1  32.8  1.35     0/0      0/0
o2 ipopt pose paired                  388  315  1.5e-08     learned    58    29    1.74    0.84    0.73    6.2    1.2  120/165   80.8%  86.5%  -5.7   32%  100%    42.8  32.7  1.31    30/0      0/0
o1 snopt grasp native                 332  308     0.12         tie   558   144    3.86    0.60    0.42   35.5    4.8  119/172   69.2%  63.3%  +5.9    5%  100%    23.6  19.2  1.23    51/8     80/5
o1 snopt grasp paired                 321  308     0.39         tie   604   144    4.01    0.66    0.42   40.1    4.8  104/172   66.9%  64.3%  +2.6    4%  100%    23.8  19.1  1.24   62/10     78/5
o1 snopt pose native                  480  252  4.6e-69     learned    67    26    2.85    0.71    0.72    1.7    0.5  228/228  100.0%  78.3% +21.7   19%   97%    23.2  19.3  1.20     0/0      4/0
o1 snopt pose paired                  275  252     0.11         tie   644    26   20.26    0.73    0.68   48.9    0.5  109/228   57.3%  93.6% -36.4   86%   97%    22.8  19.1  1.20   128/0      4/0
o2 snopt grasp native                 306  322     0.31         tie   554   130    3.44    0.59    0.37   42.7    5.8  100/158   63.8%  61.0%  +2.8    4%  100%    31.6  27.2  1.16   55/11    102/5
o2 snopt grasp paired                 311  322     0.47         tie   610   130    4.06    0.64    0.35   50.1    5.8   89/158   64.8%  66.5%  -1.8    4%  100%    31.9  27.3  1.17   102/7    102/5
o2 snopt pose native                  479  275    4e-60     learned    68    24    3.04    0.71    0.73    2.2    0.6  205/205   99.8%  82.8% +17.0   21%  100%    31.4  27.3  1.15     0/0      6/1
o2 snopt pose paired                  274  275        1         tie   594    24   21.44    0.73    0.71   56.6    0.6   92/205   57.1%  97.0% -40.0   84%  100%    31.0  27.1  1.15   130/0      6/1

Tally, ipopt: learned 5, tie 3

Tally, snopt: learned 2, tie 6
'L-TM' = learned success rate minus time-matched joint space, in points: the pre-registered prediction is that it is positive wherever single-start joint space leaves room. Read it with 'starv', the share of cells where the true multi-start figure is higher.

Cap check: 'to/ic' = timed_out / hit_iteration_cap. A loss or tie where the losing arm has a budget-bound population that could close the gap carries NO verdict.

=== o1 against o2: target-level success rate, bootstrap CI over targets (unpaired grids)
  ipopt grasp native learned    o1 - o2 = -0.8 pts [-4.0, +1.9]
  ipopt grasp native numerical  o1 - o2 = +1.7 pts [-3.1, +6.9]
  ipopt grasp paired learned    o1 - o2 = +2.1 pts [-1.2, +5.2]
  ipopt grasp paired numerical  o1 - o2 = +1.5 pts [-3.3, +6.7]
  ipopt pose native learned    o1 - o2 = +0.0 pts [+0.0, +0.0]
  ipopt pose native numerical  o1 - o2 = -0.8 pts [-7.3, +5.6]
  ipopt pose paired learned    o1 - o2 = -1.3 pts [-8.1, +5.6]
  ipopt pose paired numerical  o1 - o2 = -0.8 pts [-7.3, +5.6]
  snopt grasp native learned    o1 - o2 = +5.4 pts [-1.0, +12.1]
  snopt grasp native numerical  o1 - o2 = -2.9 pts [-10.0, +3.5]
  snopt grasp paired learned    o1 - o2 = +2.1 pts [-5.0, +9.2]
  snopt grasp paired numerical  o1 - o2 = -2.9 pts [-10.0, +3.5]
  snopt pose native learned    o1 - o2 = +0.2 pts [+0.0, +0.6]
  snopt pose native numerical  o1 - o2 = -4.8 pts [-12.7, +3.1]
  snopt pose paired learned    o1 - o2 = +0.2 pts [-7.9, +8.1]
  snopt pose paired numerical  o1 - o2 = -4.8 pts [-12.7, +3.1]
```

## Reading it

**Success: learned wins 7, ties 9, loses 0.** IPOPT gives 5 wins and 3 ties, SNOPT 2 wins and 6
ties.
- **Pose native** is the largest effect on both solvers: learned 479-480 of 480, against joint space
  252-315, with p from 4e-50 to 4.6e-69.
- **IPOPT pose paired** is a win too (382 and 388 against 311 and 315).
- **IPOPT grasp** has joint space at 447-455 of 480, which leaves 25-33 failures to win on. The
  learned arm rescues 23-32 of them, so three of the four grasp rows are ties and o2 native is a
  narrow win (p = 0.0037).
- **There is no SQP grasp loss here.** All four SNOPT grasp rows are ties. All five of the record's
  losses were contained grasp under SQP, on the iiwa, the soft PCS arm and the screw-joint arm.

**The cap check.**
- **`hit_iteration_cap` is at most 11 cells of 480 on any arm and row**, below the 24-cell threshold
  stage SCREWCAP used. So **no row is iteration-budget-bound**, and none needs the re-run that rule
  calls for.
- What binds is the **wall clock**:
  - the learned arm times out on 51-130 cells on the four SNOPT grasp rows and the two SNOPT pose
    paired rows;
  - joint space times out on 78-102 on the SNOPT grasp rows;
  - the learned arm times out on 27-32 on IPOPT o2 grasp paired and both IPOPT pose paired rows.
- The clock is the usability limit and is not raised (Thomas, 2026-10-02), so **these verdicts are
  results at the fielded 180 s clock.** Say so beside the SNOPT ties, which carry large timed-out
  populations on both arms.

**Pre-registered prediction 1, the per-evaluation premium of ~1.3-1.5x, HOLDS under IPOPT:**
- IPOPT: o1 1.44-1.48x, o2 1.31-1.35x.
- SNOPT: below the band at 1.15-1.24x.
- This matches the untrained-chart reading (GVSPREM2: 1.16-1.52x), as it should, since a chart's
  cost per evaluation is set by its architecture.
- o2's premium is smaller because its equilibrium solve costs more (32 against 23 ms per joint-space
  evaluation), while the flow costs the same.

**Pre-registered prediction 2, learned success survives the time-matched column, HOLDS by sign on
11 of 16 rows and FAILS on all four pose paired rows.**
- **Holds decisively:**
  - IPOPT grasp: 94.8-97.3% against 70.7-74.7%;
  - pose native, both solvers: 99.8-100% against 37.8-82.8%.
- **Level:** the four SNOPT grasp rows, at -1.8 to +5.9 points. Three are positive. The fifth
  negative row is o2 grasp paired, at -1.8.
- **Fails:**
  - IPOPT pose paired: 79.6% and 80.8% against 88.0% and 86.5%;
  - SNOPT pose paired: 57.3% and 57.1% against 93.6% and 97.0%.
- On the pose paired rows the 32% (IPOPT) and 84-86% (SNOPT) starved shares mean the true multi-start
  figure is higher still, so those failures are understated, not overstated.
- **The mechanism is the learned arm's cost on that row.** From the paired start the learned arm needs
  56-58 iterations against 18-19 native under IPOPT, and 594-644 against 67-68 under SNOPT, where it
  times out on 128-130 cells. Joint space solves a pose cell in under a second. So one learned
  solve buys several joint-space restarts, and its 180 s timeouts buy all eight.
- **This is the soft PCS arm's pattern**: its pose paired row was the one time-matched tie (90.8%
  against 90.6%). The one difference is that this robot's exact forward model did not make
  joint-space restarts expensive enough to change it.

**On pose native the time-matched figure (37.8-38.2% under IPOPT) is BELOW single-start joint space
(64.8-65.6%).** That is correct, not a bug. The learned arm solves a pose native cell in about the
time of one joint-space solve (median 0.9 against 0.8 s, and 1.2 against 1.2 s), so the matched
budget often does not cover even one joint-space attempt.

**Pre-registered column 2, iterations on mutual successes.**
- The learned arm needs fewer iterations only on **pose native under IPOPT, at 0.59-0.63x**.
- Elsewhere it needs more:
  - IPOPT grasp 1.22-1.34x and IPOPT pose paired 1.74-1.79x;
  - SNOPT grasp 3.4-4.1x and SNOPT pose native 2.9-3.0x;
  - **SNOPT pose paired 20-21x**.
- So unlike the rigid arms' hardened grasp, containment does not cost this robot's joint-space arm
  its cheapness: 75-80 median iterations on grasp under IPOPT.

**Cost** (median `reported_cost` on cells both arms solved) is **higher for the learned arm on every
IPOPT row**:
- grasp 0.76-0.80 against 0.35, about 2.2x;
- pose native 1.06 against 0.71-0.73;
- pose paired 0.84 against 0.73.

Under SNOPT, grasp is 0.59-0.66 against 0.35-0.42, and the pose rows are level (0.71-0.73 against
0.68-0.73). Unlike the rigid arms, the learned arm is not cheaper on pose.

**Runtime:**
- On every row but IPOPT pose native, joint space is faster per solved cell: IPOPT grasp 3-4 s
  against 9-14 s, and SNOPT grasp 5-6 s against 36-50 s.
- On IPOPT pose native the two are level, 0.9 against 0.8 s and 1.2 against 1.2 s, with the learned
  arm solving 480 against 311-315.

**The rescue rate travels with the record's caveat.** The learned arm solves 100% of joint space's
pose native failures, and 88-97% of its IPOPT grasp failures. But joint space solves 97-100% of
targets from at least one of its 8 starts (`any8`), so a rescue is a different start, not an
unreachable target.

**o1 against o2: no rung separates on any row.** Every target-level 95% interval includes zero, on
both arms. The two orders are interchangeable on success, as they were on every intrinsic chart
number, and differ only in cost per evaluation.
