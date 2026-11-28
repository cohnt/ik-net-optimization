# The cuspidal arm (cusp7): tables

Generated 2026-11-28 from stages CUSP, SVGD_CUSP and CUSPCHART (jobs 6346164-75; `cluster/CUSP_ARM_RUNBOOK.md`). Numbers and verdicts only. Outside the record. Regenerate:

```bash
python scripts/report_statusquo.py --cusp
python scripts/report_time_matched.py --prefix sc_CUSP_        # chart n6
python scripts/report_time_matched.py --prefix sc_CUSPCHART_   # chart n4
python scripts/report_svgd_2x2.py --cusp
```

## Learned vs joint space, per solver (`report_statusquo.py --cusp`)

```text
THE CUSPIDAL ARM (cusp7) -- 480 cells, 180 s cap, seed 1; OUTSIDE the record
Published simplified KUKA Transpressor 6R (Feess & Weiss) + a tool pitch, iiwa-like limits; no
  analytic arm. The record's conditions: hardened scene, shelf-contained at the fingertips,
  --compile + flow_cuda_graph, lifted budgets (IPOPT max_iter 1e6; SNOPT 1e5 majors, 1e8
  minors), correction_cost_weight 10, PROCS=2 no MPS (one solve per V100), Drake 1.58.0.
Chart n6 (pre-registered): stage CUSP, 8 of 8 IPOPT + SNOPT runs; stage SVGD_CUSP,
  4 of 4 AL-SVGD rows, each TWO runs (learned _pin4, joint space _pin4kq kernel q),
  the V4 pin, clearance collision row; grid_hash and scene_fingerprint checked against IPOPT.
Chart n4: stage CUSPCHART, 8 of 8 IPOPT + SNOPT runs, printed as its own block.
No NLopt (new robots field IPOPT and SNOPT only), so there is no AL column.

=== HEADLINE TABLES -- cusp7 n6 (learned vs joint space)
  IP = interior point (IPOPT), AL = augmented Lagrangian (NLOPT),
  SQP = sequential quadratic programming (SNOPT). *better* of each pair is starred;
  a trailing * marks the best in the row. Every row prints, zeros included.
  PS = AL-SVGD (particle, the svgd branch's fielded pin): each arm its own run, learned
  _pin4, joint space _pin4kq (kernel q), both from stage SVGD_CUSP; the V4 pin.

  Table 1 -- success rate of 480 cells
  higher is better; ties are by exact McNemar (p >= 0.05), not numeric equality
  experiment                                   IP L        IP JS        SQP L       SQP JS         PS L        PS JS
  cusp7 grasp contained native              *0.992*        0.815      *0.885*        0.537     *1.000**        0.912
  cusp7 grasp contained paired              *0.975*        0.815      *0.592*      *0.537*     *1.000**        0.910
  cusp7 pose contained native               *0.908*        0.358      *0.790*        0.321     *1.000**        0.981
  cusp7 pose contained paired               *0.815*        0.358      *0.448*        0.321     *1.000**        0.981

  Of the 12 solver x experiment cells, learned wins 11, ties 1, loses 0  (verdicts by exact McNemar on the same cells the table shows)
    IP   learned 4, ties 0, joint space 0
    SQP  learned 3, ties 1, joint space 0
    PS   learned 4, ties 0, joint space 0

  Table 2 -- optimal cost, cells BOTH arms solved, learned-only regularizers excluded
  lower is better; N/A means fewer than 10 shared solved cells, so no comparison exists
  experiment                                   IP L        IP JS        SQP L       SQP JS         PS L        PS JS
  cusp7 grasp contained native                7.242     *3.144**      *6.278*        6.736      *3.443*        4.610
  cusp7 grasp contained paired                6.711     *3.142**      *6.392*        6.774      *3.504*        4.611
  cusp7 pose contained native               *7.599*        8.609      *7.154*        8.279        3.948     *3.868**
  cusp7 pose contained paired                 8.856      *8.609*      *6.875*        7.797        3.893     *3.868**

  Table 3 -- mean runtime, s, over ALL cells, each clamped at the 180 s clock
  lower is better; this machine only, never compared across machines. Clamped because SNOPT overruns its clock on cycling cells (none feasible) once the iteration budget is lifted
  every column at PROCS=2, no MPS; PS on the clearance collision row
  experiment                                   IP L        IP JS        SQP L       SQP JS         PS L        PS JS
  cusp7 grasp contained native                 3.51      *1.61**       *3.12*        11.24       *1.68*        18.20
  cusp7 grasp contained paired                 6.95       *1.61*        16.35      *11.28*      *1.42**        18.30
  cusp7 pose contained native                  1.47      *0.17**         1.72       *0.82*       *0.75*         4.10
  cusp7 pose contained paired                  3.29      *0.17**         7.43       *0.82*       *0.76*         4.09

  Table 4 -- median major iterations over solved cells
  lower is better; AL is N/A BY CONSTRUCTION -- NloptSolverDetails carries a single status and NLopt has no major iteration to count
  PS is N/A BY CONSTRUCTION too -- svgd counts OUTER steps (K = 10 inner steps each), not majors
  experiment                                   IP L        IP JS        SQP L       SQP JS         PS L        PS JS
  cusp7 grasp contained native               *113**          132          163        *118*          N/A          N/A
  cusp7 grasp contained paired                  210        *132*          413       *118**          N/A          N/A
  cusp7 pose contained native                    37         *30*           44        *19**          N/A          N/A
  cusp7 pose contained paired                   103         *30*          213        *19**          N/A          N/A

=== PER-SOLVER DETAIL -- cusp7 n6 (discordant counts, McNemar p, timeouts)

=== IPOPT (interior point)   [acceptable-point early stop (acceptable_tol 1e-3, acceptable_iter 1)]

  cusp7 n6
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  cusp7 grasp contained native              476  391   87    2 1.29e-23      learned      113    132    1.45   0.34   7.242   3.144  389    0    0
  cusp7 grasp contained paired              468  391   86    9 6.61e-17      learned      210    132    2.64   0.34   6.711   3.142  382    6    0
  cusp7 pose contained (tip) native         436  172  273    9 7.27e-69      learned       37     30    0.37   0.06   7.599   8.609  163    1    0
  cusp7 pose contained (tip) paired         391  172  248   29 1.57e-44      learned      103     30    1.16   0.06   8.856   8.609  143    4    0

=== SNOPT (SQP)   [Major step limit = 0.5]

  cusp7 n6
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  cusp7 grasp contained native              425  258  198   31 5.68e-31      learned      163    118    1.35   0.33   6.278   6.736  227    1   19
  cusp7 grasp contained paired              284  258  118   92   0.0843          tie      413    118    4.93   0.33   6.392   6.774  166   18   19
  cusp7 pose contained (tip) native         379  154  244   19 5.89e-51      learned       44     19    0.27   0.03   7.154   8.279  135    1    2
  cusp7 pose contained (tip) paired         215  154  126   65 1.21e-05      learned      213     19    2.37   0.03   6.875   7.797   89    6    2

=== AL-SVGD (particle, the svgd branch's fielded pin)   [SVGD_PINNED_V2: N = 64, K = 10 inner steps per outer step, rho = 1e5, lr = 0.3, H / T metric, kernel cz (learned) / q (joint space), native-draw paired init]

  cusp7 n6
  iterations print -- BY CONSTRUCTION: svgd counts OUTER steps (K = 10 inner steps each),
  not majors. Each arm is its own run (see the header); LTO / JTO count clock stops,
  timed_out or svgd.stop_reason == 'wall_clock'.
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  cusp7 grasp contained native              480  438   42    0 4.55e-13      learned       --     --    0.95   1.39   3.443   4.610  438    0   42
  cusp7 grasp contained paired              480  437   43    0 2.27e-13      learned       --     --    0.87   1.44   3.504   4.611  437    0   43
  cusp7 pose contained (tip) native         480  471    9    0  0.00391      learned       --     --    0.69   0.78   3.948   3.868  471    0    9
  cusp7 pose contained (tip) paired         480  471    9    0  0.00391      learned       --     --    0.69   0.77   3.893   3.868  471    0    9


=== CAP CHECK (per solver class and arm, over the rows above)
  clock = timed_out, or svgd.stop_reason == 'wall_clock'; budget = hit_iteration_cap or
  hit_eval_cap, or svgd.stop_reason == 'step_cap'. A row with >= 24 of 480 cells at a BUDGET
  carries no verdict; the 180 s clock is the fielded one and is never lifted.
  class      L clock  L budget    JS clock  JS budget rows >= 24 budget
  IP             0-6         0           0          0                 0
  SQP           1-18         0        2-19          1                 0
  PS               0         0        9-43          0                 0
         AL-SVGD rows with any clock stop: cusp7 grasp contained native (L 0, JS 42), cusp7 grasp contained paired (L 0, JS 43), cusp7 pose contained native (L 0, JS 9), cusp7 pose contained paired (L 0, JS 9)


=== ACCEPTANCE (AL-SVGD)
--- grid_hash and scene_fingerprint: on all 4 AL-SVGD rows the two runs (one per arm)
  agree with each other and with the row's IPOPT record run (checked at load; this script
  refuses to pair otherwise).
--- scene_fingerprint per robot x task scene, over every run in these tables (Drake runs and
  both AL-SVGD arms):
  cusp7         grasp contained   identical: 0a23e9e9cace
  cusp7         pose contained    identical: 47794748fa9b
--- joint space bit-identical between protocols is a DRAKE-solver check (scripts/report_remeasure.py
  4b): same q_init, deterministic solver. AL-SVGD is EXCLUDED from it: its paired swarm draws 63
  of its 64 particles from the native distribution (only particle 0 is the given start), so its
  native and paired joint-space solves are different solves by design.


=== HEADLINE TABLES -- cusp7 n4 (chart rung; learned vs joint space)
  IP = interior point (IPOPT), AL = augmented Lagrangian (NLOPT),
  SQP = sequential quadratic programming (SNOPT). *better* of each pair is starred;
  a trailing * marks the best in the row. Every row prints, zeros included.

  Table 1 -- success rate of 480 cells
  higher is better; ties are by exact McNemar (p >= 0.05), not numeric equality
  experiment                                   IP L        IP JS        SQP L       SQP JS
  cusp7 grasp contained native             *0.988**        0.815      *0.850*        0.537
  cusp7 grasp contained paired             *0.967**        0.815      *0.573*      *0.537*
  cusp7 pose contained native              *0.885**        0.358      *0.729*        0.321
  cusp7 pose contained paired              *0.750**        0.358      *0.375*      *0.321*

  Of the 8 solver x experiment cells, learned wins 6, ties 2, loses 0  (verdicts by exact McNemar on the same cells the table shows)
    IP   learned 4, ties 0, joint space 0
    SQP  learned 2, ties 2, joint space 0

  Table 2 -- optimal cost, cells BOTH arms solved, learned-only regularizers excluded
  lower is better; N/A means fewer than 10 shared solved cells, so no comparison exists
  experiment                                   IP L        IP JS        SQP L       SQP JS
  cusp7 grasp contained native                7.016     *3.142**      *6.273*        6.799
  cusp7 grasp contained paired                7.394     *3.149**        7.715      *6.941*
  cusp7 pose contained native               *8.345*        8.571     *7.658**        8.480
  cusp7 pose contained paired               *8.564*        8.609        8.341     *8.341**

  Table 3 -- mean runtime, s, over ALL cells, each clamped at the 180 s clock
  lower is better; this machine only, never compared across machines. Clamped because SNOPT overruns its clock on cycling cells (none feasible) once the iteration budget is lifted
  every column at PROCS=2, no MPS
  experiment                                   IP L        IP JS        SQP L       SQP JS
  cusp7 grasp contained native                 3.93      *1.63**       *4.49*        11.26
  cusp7 grasp contained paired                 7.41      *1.63**        14.57      *11.27*
  cusp7 pose contained native                  1.38      *0.17**         2.26       *0.82*
  cusp7 pose contained paired                  9.57      *0.17**        11.52       *0.82*

  Table 4 -- median major iterations over solved cells
  lower is better; AL is N/A BY CONSTRUCTION -- NloptSolverDetails carries a single status and NLopt has no major iteration to count
  experiment                                   IP L        IP JS        SQP L       SQP JS
  cusp7 grasp contained native                *131*          132          229       *118**
  cusp7 grasp contained paired                  234        *132*          545       *118**
  cusp7 pose contained native                    44         *30*           64        *19**
  cusp7 pose contained paired                   101         *30*          230        *19**

=== PER-SOLVER DETAIL -- cusp7 n4

=== IPOPT (interior point)   [acceptable-point early stop (acceptable_tol 1e-3, acceptable_iter 1)]

  cusp7 n4
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  cusp7 grasp contained native              474  391   88    5 1.11e-20      learned      131    132    1.31   0.34   7.016   3.142  386    3    0
  cusp7 grasp contained paired              464  391   86   13 2.29e-14      learned      234    132    2.41   0.34   7.394   3.149  378    5    0
  cusp7 pose contained (tip) native         425  172  265   12 2.89e-63      learned       44     30    0.36   0.06   8.345   8.571  160    0    0
  cusp7 pose contained (tip) paired         360  172  215   27 1.52e-37      learned      101     30    0.92   0.06   8.564   8.609  145   19    0

=== SNOPT (SQP)   [Major step limit = 0.5]

  cusp7 n4
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  cusp7 grasp contained native              408  258  187   37 2.61e-25      learned      229    118    1.68   0.32   6.273   6.799  221    1   19
  cusp7 grasp contained paired              275  258  119  102    0.282          tie      545    118    4.87   0.32   7.715   6.941  156   16   19
  cusp7 pose contained (tip) native         350  154  226   30 2.26e-38      learned       64     19    0.40   0.03   7.658   8.480  124    1    2
  cusp7 pose contained (tip) paired         180  154  103   77   0.0621          tie      230     19    1.98   0.03   8.341   8.341   77   17    2


=== CAP CHECK (per solver class and arm, over the rows above)
  clock = timed_out, or svgd.stop_reason == 'wall_clock'; budget = hit_iteration_cap or
  hit_eval_cap, or svgd.stop_reason == 'step_cap'. A row with >= 24 of 480 cells at a BUDGET
  carries no verdict; the 180 s clock is the fielded one and is never lifted.
  class      L clock  L budget    JS clock  JS budget rows >= 24 budget
  IP            0-19         0           0          0                 0
  SQP           1-17         0        2-19          1                 0


=== CHART RUNG: learned n4 against learned n6, same cells (exact McNemar)
  row                                             n6    n4   n6+   n4+         p  better
  ipopt grasp contained native                   476   474     6     4     0.754     tie
  ipopt grasp contained paired                   468   464    15    11     0.557     tie
  ipopt pose contained (tip) native              436   425    37    26     0.207     tie
  ipopt pose contained (tip) paired              391   360    88    57    0.0125      n6
  snopt grasp contained native                   425   408    57    40     0.104     tie
  snopt grasp contained paired                   284   275   113   104     0.587     tie
  snopt pose contained (tip) native              379   350    76    47    0.0113      n6
  snopt pose contained (tip) paired              215   180   117    82    0.0157      n6


=== ACCEPTANCE (Drake solvers)
  identical = same feasibility on every cell, same q to 1e-12 on every cell that stopped on
  its own; a cell the clock stopped in both runs is exempt from the q check (counted).
--- joint space identical between protocols: 4/4 pairs (21 clock-stopped cells exempt)
--- joint space identical between stages CUSP (n6) and CUSPCHART (n4), which never
    evaluates the chart: 8/8 runs (42 clock-stopped cells exempt)
```

## Time-matched joint space, Drake solvers (`report_time_matched.py`)

```text
--prefix sc_CUSP_
solver row                             learned   JS x1 JS matched  JS any  starved starts  L wall  JS wall
----------------------------------------------------------------------------------------------------------
ipopt  cusp7 grasp native                99.2%   81.5%      92.3%  100.0%       0%      1   1.47s    0.35s
ipopt  cusp7 grasp paired                97.5%   81.5%      94.7%  100.0%       0%      1   2.74s    0.35s
ipopt  cusp7 pose native                 90.8%   35.8%      67.0%   93.3%       1%      1   0.41s    0.10s
ipopt  cusp7 pose paired                 81.5%   35.8%      83.9%   93.3%       3%      2   1.18s    0.10s
snopt  cusp7 grasp native                88.5%   53.8%      80.3%  100.0%       0%      1   1.40s    0.35s
snopt  cusp7 grasp paired                59.2%   53.8%      88.6%  100.0%       0%      1   4.76s    0.35s
snopt  cusp7 pose native                 79.0%   32.1%      81.0%   91.7%       7%      2   0.43s    0.03s
snopt  cusp7 pose paired                 44.8%   32.1%      89.3%   91.7%       8%      3   2.22s    0.03s

--prefix sc_CUSPCHART_
solver row                             learned   JS x1 JS matched  JS any  starved starts  L wall  JS wall
----------------------------------------------------------------------------------------------------------
ipopt  cusp7 grasp native                98.8%   81.5%      92.3%  100.0%       0%      1   1.34s    0.35s
ipopt  cusp7 grasp paired                96.7%   81.5%      94.4%  100.0%       0%      1   2.50s    0.35s
ipopt  cusp7 pose native                 88.5%   35.8%      67.5%   93.3%       2%      1   0.38s    0.10s
ipopt  cusp7 pose paired                 75.0%   35.8%      83.0%   93.3%       3%      2   1.00s    0.10s
snopt  cusp7 grasp native                85.0%   53.8%      81.8%  100.0%       0%      1   1.68s    0.35s
snopt  cusp7 grasp paired                57.3%   53.8%      87.7%  100.0%       0%      1   4.58s    0.35s
snopt  cusp7 pose native                 72.9%   32.1%      81.8%   91.7%       6%      2   0.59s    0.03s
snopt  cusp7 pose paired                 37.5%   32.1%      88.6%   91.7%       8%      3   2.13s    0.03s

```

## AL-SVGD as a double contribution: the 2x2 on the cuspidal arm (cusp7, chart n6; outside the record)

Generated by `python scripts/report_svgd_2x2.py --cusp`; numbers and verdicts only. Proposed = learned formulation under AL-SVGD (stage SVGD_CUSP `pin4`, the V4 pin). Ablation A = the same formulation under IPOPT / SNOPT (stage CUSP). Ablation B = joint space under AL-SVGD (SVGD_CUSP `pin4kq`, kernel q, never tuned). Baselines = joint space under IPOPT / SNOPT (stage CUSP). No NLopt. Same cells throughout (grid hash and scene fingerprint asserted); 480 cells, 180 s, seed 1, one solve per GPU. Verdicts: exact McNemar at p < 0.05; a cost ratio inside [0.94, 1.1] (the record's seed-to-seed band, SVGD_R30) is a tie.

### Summary (the condensed read; every number below is in the per-row tables that follow)

Proposed: 480/480 on all 4 rows, wall 0.75-1.68 s, 0 clock stops, 31-54 feasible particles of 64 at stop.

#### Success: range over the 4 rows and the McNemar tally against the proposed method

| corner | range | proposed wins / ties / losses | the wins |
| --- | --- | --- | --- |
| B: JS + AL-SVGD | 437-471 | 4 / 0 / 0 | all 4 |
| A: learned + IPOPT | 391-476 | 3 / 1 / 0 | cusp7 grasp paired, cusp7 pose native, cusp7 pose paired |
| A: learned + SNOPT | 215-425 | 4 / 0 / 0 | all 4 |
| base: JS + IPOPT | 172-391 | 4 / 0 / 0 | all 4 |
| base: JS + SNOPT | 154-258 | 4 / 0 / 0 | all 4 |

#### Cost: ratio corner / proposed on cells both solved (range; cheaper-proposed / tie / cheaper-corner rows)

| corner | rigid grasp | rigid pose |
| --- | --- | --- |
| B: JS + AL-SVGD | 1.32-1.34 (2 / 0 / 0) | 0.98-0.99 (0 / 2 / 0) |
| A: learned + IPOPT | 1.94-2.16 (2 / 0 / 0) | 2.02-2.16 (2 / 0 / 0) |
| A: learned + SNOPT | 1.89-1.93 (2 / 0 / 0) | 1.98-2.06 (2 / 0 / 0) |
| base: JS + IPOPT | 0.91-0.93 (0 / 0 / 2) | 2.19-2.26 (2 / 0 / 0) |
| base: JS + SNOPT | 2.07-2.08 (2 / 0 / 0) | 2.13-2.17 (2 / 0 / 0) |

#### Wall: mean per cell (range over the 4 rows); proposed faster / level / slower (10% either way)

| corner | wall range | faster / level / slower | the slower rows |
| --- | --- | --- | --- |
| B: JS + AL-SVGD | 4.09-18.30 s | 4 / 0 / 0 | none |
| A: learned + IPOPT | 1.47-6.95 s | 4 / 0 / 0 | none |
| A: learned + SNOPT | 1.72-16.35 s | 4 / 0 / 0 | none |
| base: JS + IPOPT | 0.17-1.61 s | 1 / 1 / 2 | cusp7 pose native, cusp7 pose paired |
| base: JS + SNOPT | 0.82-11.28 s | 2 / 2 / 0 | none |

#### Time-matched: the corner restarting inside the proposed method's wall on each cell

| corner | range over the 4 rows | rows at or above 99% | lowest rows |
| --- | --- | --- | --- |
| A: learned + IPOPT | 86.1-99.2% | 1 | cusp7 pose paired 86.1%, cusp7 pose native 93.6%, cusp7 grasp paired 97.5% |
| A: learned + SNOPT | 48.3-92.1% | 0 | cusp7 pose paired 48.3%, cusp7 grasp paired 64.4%, cusp7 pose native 81.6% |
| base: JS + IPOPT | 79.7-91.8% | 0 | cusp7 pose native 79.7%, cusp7 pose paired 80.2%, cusp7 grasp paired 91.5% |
| base: JS + SNOPT | 78.4-87.0% | 0 | cusp7 grasp paired 78.4%, cusp7 grasp native 80.0%, cusp7 pose native 86.6% |

### Success (feasible cells of 480; proposed-only / corner-only cells, p)

| row | proposed | B: JS + AL-SVGD | A: learned + IPOPT | A: learned + SNOPT | base: JS + IPOPT | base: JS + SNOPT |
| --- | --- | --- | --- | --- | --- | --- |
| cusp7 grasp native | 480 | 438 (+42 / -0, p=4.5e-13) | 476 (+4 / -0, p=1.2e-01) | 425 (+55 / -0, p=5.6e-17) | 391 (+89 / -0, p=3.2e-27) | 258 (+222 / -0, p=3.0e-67) |
| cusp7 grasp paired | 480 | 437 (+43 / -0, p=2.3e-13) | 468 (+12 / -0, p=4.9e-04) | 284 (+196 / -0, p=2.0e-59) | 391 (+89 / -0, p=3.2e-27) | 258 (+222 / -0, p=3.0e-67) |
| cusp7 pose native | 480 | 471 (+9 / -0, p=3.9e-03) | 436 (+44 / -0, p=1.1e-13) | 379 (+101 / -0, p=7.9e-31) | 172 (+308 / -0, p=3.8e-93) | 154 (+326 / -0, p=1.5e-98) |
| cusp7 pose paired | 480 | 471 (+9 / -0, p=3.9e-03) | 391 (+89 / -0, p=3.2e-27) | 215 (+265 / -0, p=3.4e-80) | 172 (+308 / -0, p=3.8e-93) | 154 (+326 / -0, p=1.5e-98) |

| corner | range over the 4 rows | proposed wins / ties / losses |
| --- | --- | --- |
| B: JS + AL-SVGD | 437-471 | 4 / 0 / 0 |
| A: learned + IPOPT | 391-476 | 3 / 1 / 0 |
| A: learned + SNOPT | 215-425 | 4 / 0 / 0 |
| base: JS + IPOPT | 172-391 | 4 / 0 / 0 |
| base: JS + SNOPT | 154-258 | 4 / 0 / 0 |

### Cost (median on cells both solved, shared objective; ratio = corner / proposed, n shared)

| row | proposed | B: JS + AL-SVGD | A: learned + IPOPT | A: learned + SNOPT | base: JS + IPOPT | base: JS + SNOPT |
| --- | --- | --- | --- | --- | --- | --- |
| cusp7 grasp native | 3.44 | 1.34 (438) proposed cheaper | 2.16 (476) proposed cheaper | 1.89 (425) proposed cheaper | 0.93 (391) corner cheaper | 2.08 (258) proposed cheaper |
| cusp7 grasp paired | 3.50 | 1.32 (437) proposed cheaper | 1.94 (468) proposed cheaper | 1.93 (284) proposed cheaper | 0.91 (391) corner cheaper | 2.07 (258) proposed cheaper |
| cusp7 pose native | 3.95 | 0.98 (471) tie | 2.02 (436) proposed cheaper | 2.06 (379) proposed cheaper | 2.19 (172) proposed cheaper | 2.13 (154) proposed cheaper |
| cusp7 pose paired | 3.89 | 0.99 (471) tie | 2.16 (391) proposed cheaper | 1.98 (215) proposed cheaper | 2.26 (172) proposed cheaper | 2.17 (154) proposed cheaper |

### Wall (mean over all cells, clamped at 180 s; AL-SVGD corners also clock stops and median feasible particles of 64)

| row | proposed | B: JS + AL-SVGD | A: learned + IPOPT | A: learned + SNOPT | base: JS + IPOPT | base: JS + SNOPT |
| --- | --- | --- | --- | --- | --- | --- |
| cusp7 grasp native | 1.68 s, 0 clock, 54 feas | 18.20 s, 42 clock, 1 feas | 3.51 s | 3.12 s | 1.61 s | 11.24 s |
| cusp7 grasp paired | 1.42 s, 0 clock, 54 feas | 18.30 s, 43 clock, 1 feas | 6.95 s | 16.35 s | 1.61 s | 11.28 s |
| cusp7 pose native | 0.75 s, 0 clock, 31 feas | 4.10 s, 9 clock, 11 feas | 1.47 s | 1.72 s | 0.17 s | 0.82 s |
| cusp7 pose paired | 0.76 s, 0 clock, 31 feas | 4.09 s, 9 clock, 11 feas | 3.29 s | 7.43 s | 0.17 s | 0.82 s |

| corner | wall range | proposed faster (ratio < 0.9) / level / slower |
| --- | --- | --- |
| B: JS + AL-SVGD | 4.09-18.30 s | 4 / 0 / 0 |
| A: learned + IPOPT | 1.47-6.95 s | 4 / 0 / 0 |
| A: learned + SNOPT | 1.72-16.35 s | 4 / 0 / 0 |
| base: JS + IPOPT | 0.17-1.61 s | 1 / 1 / 2 |
| base: JS + SNOPT | 0.82-11.28 s | 2 / 2 / 0 |

### Time-matched (the corner restarts inside the proposed method's wall on each cell, the running solve finishing; success share, starved share, median starts taken). The proposed method is 100% on every row.

| row | A: learned + IPOPT | A: learned + SNOPT | base: JS + IPOPT | base: JS + SNOPT |
| --- | --- | --- | --- | --- |
| cusp7 grasp native | 99.2% (starved 0.0%, 1 starts) | 92.1% (starved 0.0%, 1 starts) | 91.8% (starved 0.0%, 1 starts) | 80.0% (starved 0.0%, 1 starts) |
| cusp7 grasp paired | 97.5% (starved 0.0%, 1 starts) | 64.4% (starved 0.0%, 1 starts) | 91.5% (starved 0.0%, 1 starts) | 78.4% (starved 0.0%, 1 starts) |
| cusp7 pose native | 93.6% (starved 0.0%, 1 starts) | 81.6% (starved 0.0%, 1 starts) | 79.7% (starved 0.0%, 2 starts) | 86.6% (starved 7.3%, 2 starts) |
| cusp7 pose paired | 86.1% (starved 0.0%, 1 starts) | 48.3% (starved 0.0%, 1 starts) | 80.2% (starved 0.0%, 2 starts) | 87.0% (starved 7.1%, 2 starts) |

| corner | time-matched range over the 4 rows | rows at or above 99% |
| --- | --- | --- |
| A: learned + IPOPT | 86.1-99.2% | 1 |
| A: learned + SNOPT | 48.3-92.1% | 0 |
| base: JS + IPOPT | 79.7-91.8% | 0 |
| base: JS + SNOPT | 78.4-87.0% | 0 |

