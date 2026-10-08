# The campaign of record: the tables

Measured 2026-09-19/20 (stages STATUSQUO, panda + iiwa), 2026-09-28 (stage SOFT12, soft12) and
2026-10-01/02 (stage SCREW, screw7_p050), accepted by Thomas on 2026-09-21, extended to the soft PCS
arm on 2026-09-28 and to the screw-joint arm on 2026-10-02 (accepted with its merge). `CLAUDE.md` carries
what these tables **say**; this file carries the tables themselves, so a session pays for them only
when it needs a number.

**`scripts/report_statusquo.py` owns every table here and regenerates all of them from the persisted
runs under `results/`, together with the verdict tally and the three pre-registered flag criteria. Do
not hand-maintain them.** If a number here disagrees with that script, the script is right.

Conditions, identical across all three stages: 480 cells = 60 targets x 8 guesses, seed 1 (out of sample),
**180 s**, `--compile`, adopted rungs (Panda `n6`, iiwa `n4`, soft `n6`, screw `n6`), hardened scene,
shelf-contained targets at the fingertips, arms `learned,numerical`, both start protocols, all three
solvers at their adopted configurations, Drake nightly `0.0.20260918`. **48 logical runs**, four
robots x two experiments x two protocols x three solvers.

The soft PCS arm joins from SOFT12 rather than from a re-run under `stage_STATUSQUO`, because the
conditions are identical and a re-run would re-measure the same thing on equivalent nodes (Thomas,
2026-09-28: *"I don't see 12-by-3 vs 36 as a substantial difference ... all nodes are created equal
on supercloud"*). Only the provenance is split. The screw-joint arm joins from stage SCREW on the
same terms; `stage_SCREW` refuses any cap but 180 s.

## The grasp rows were iteration-budget-bound, and stage ITCAP re-measured them

**The tables below report every iteration-budget-bound row AT THE LIFTED BUDGET** (Thomas,
2026-10-08): the reporter substitutes each stage ITCAP / SCREWCAP run under its original's tag, 17 of
the 48. At the record's original budgets those grasp rows were iteration-cap-bound. `max_iter` defaults to `None`, so IPOPT ran at its own 3000 and SNOPT at
3000 majors, and cells reached that well inside the 180 s wall clock -- which `timed_out` does not
record and `hit_iteration_cap` does. **Stage ITCAP (2026-10-08) re-measured every such row** on the
same seed, grid, chart, `--compile` and 180 s clock with the budgets lifted (IPOPT `max_iter` 1e6;
SNOPT 1e5 majors and 1e8 minors), pairing cell for cell with these runs. **No verdict of the record
moved**, so the tally stands, and the affected verdicts are ESTABLISHED rather than
provisional. Per row, learned / joint space, original budget -> lifted (tabled):

| row | IPOPT | SNOPT |
| --- | --- | --- |
| iiwa grasp native | 447 / 442 -> **461 / 452, tie** (p = 0.23) | 201 / 304 -> 204 / 308, loss |
| iiwa grasp paired | 453 / 442 -> **464 / 452, tie** (p = 0.088) | 240 / 304 -> 243 / 308, loss |
| iiwa pose paired | not cap-bound | 247 / 270 -> 251 / 270, tie |
| Panda grasp native | 476 / 323 -> 476 / 374, win | 441 / 305 -> 441 / 306, win |
| Panda grasp paired | 471 / 323 -> 474 / 374, win | 301 / 305 -> 302 / 306, tie |
| soft12 grasp native | 474 / 445 -> 474 / 456, win (p = 9.1e-04) | 340 / 358 -> 347 / 360, tie |
| soft12 grasp paired | 468 / 445 -> 468 / 455, win (p = 0.024) | 315 / 358 -> 320 / 360, loss |

IPOPT cells at the new budget: 0 on every row. SNOPT: 1-11 per row, every one but one cycling on
zero-minor majors past the clock (SNOPT does not check its time limit there), so no verdict depends
on where they stopped. Panda grasp's joint-space arm gains **51** cells -- its margin was overstated,
as predicted, and the win is untouched. Read the margins from `python scripts/report_itcap.py`,
which prints the quartet for every re-measured row.

**The screw-joint arm's grasp rows had the same re-run earlier, as stage SCREWCAP.** Its two IPOPT grasp ties sit at the
iteration cap here (59-66 of the learned arm's 61-69 failures at 3000), so they are printed as
measured under the record's conditions -- but stage SCREWCAP re-measured them at the same 180 s with
the iteration budgets lifted (Thomas: the wall clock is not raised) and **they hold as ties**, 427 v
424 and 429 v 424. Its two SQP grasp losses hold too (163 and 212 against 300). Table:
`python scripts/report_screw.py SCREWCAP`.

---

THE CAMPAIGN OF RECORD -- 480 cells, 180 s cap, seed 1
Stages STATUSQUO (panda, iiwa) + SOFT12 (soft12) + SCREW (screw7_p050), identical conditions.
Arms: learned vs joint space (numerical). No analytic baseline is fielded.
Iteration-budget-bound rows are reported at the LIFTED budget, same 180 s clock: 17 of 48 runs
  from stages ITCAP (IPOPT max_iter 1e6; SNOPT 1e5 majors, 1e8 minors) and SCREWCAP.
NOTE: solver options move the JOINT-SPACE arm too -- that arm never evaluates the
      network, so a moving JS column is a property of the problem, not drift.

=== HEADLINE TABLES (learned vs joint space, per solver)
  IP = interior point (IPOPT), AL = augmented Lagrangian (NLOPT),
  SQP = sequential quadratic programming (SNOPT). *better* of each pair is starred;
  a trailing * marks the best in the row. Every row prints, zeros included.

  Table 1 -- success rate of 480 cells
  higher is better; ties are by exact McNemar (p >= 0.05), not numeric equality
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native              *0.960**      *0.942*      *0.000*      *0.000*        0.425      *0.642*
  iiwa grasp contained paired              *0.967**      *0.942*      *0.004*      *0.000*        0.506      *0.642*
  iiwa pose contained native               *0.979**        0.623      *0.621*        0.065      *0.925*        0.562
  iiwa pose contained paired               *0.879**        0.623      *0.246*        0.065      *0.523*      *0.562*
  panda grasp contained native             *0.992**        0.779      *0.681*        0.000      *0.919*        0.637
  panda grasp contained paired             *0.988**        0.779      *0.000*      *0.000*      *0.629*      *0.637*
  panda pose contained native              *0.960**        0.452      *0.602*        0.025      *0.917*        0.371
  panda pose contained paired              *0.848**        0.452      *0.237*        0.025      *0.573*        0.371
  screw7_p050 grasp contained native       *0.890**      *0.883*      *0.000*      *0.000*        0.340      *0.625*
  screw7_p050 grasp contained paired       *0.894**      *0.883*      *0.006*      *0.000*        0.442      *0.625*
  screw7_p050 pose contained native        *0.952**        0.652      *0.631*        0.062      *0.885*        0.560
  screw7_p050 pose contained paired        *0.877**        0.652      *0.181*        0.062      *0.540*      *0.560*
  soft12 grasp contained native            *0.988**        0.950      *0.000*      *0.000*      *0.723*      *0.750*
  soft12 grasp contained paired            *0.975**        0.948      *0.004*      *0.000*        0.667      *0.750*
  soft12 pose contained native              *0.994*        0.696      *0.746*        0.106     *0.998**        0.631
  soft12 pose contained paired             *0.908**        0.696      *0.677*        0.108      *0.819*        0.631

  Of the 48 solver x experiment cells, learned wins 28, ties 15, loses 5  (verdicts by exact McNemar on the same cells the table shows)
    IP   learned 12, ties 4, joint space 0
    AL   learned 9, ties 7, joint space 0
    SQP  learned 7, ties 4, joint space 5

  Table 2 -- optimal cost, cells BOTH arms solved, learned-only regularizers excluded
  lower is better; N/A means fewer than 10 shared solved cells, so no comparison exists
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native                 4.978     *2.839**          N/A          N/A        5.421      *5.388*
  iiwa grasp contained paired                 5.013     *2.839**          N/A          N/A        5.625      *4.728*
  iiwa pose contained native                *6.242*        6.757        6.511     *4.988**      *5.975*        6.858
  iiwa pose contained paired                *6.011*        6.704        7.062     *5.807**        6.877      *6.756*
  panda grasp contained native                7.626     *5.602**          N/A          N/A      *7.040*        7.437
  panda grasp contained paired                7.067     *5.602**          N/A          N/A      *7.384*        7.567
  panda pose contained native              *11.136*       11.861        9.346     *9.088**     *10.366*       10.774
  panda pose contained paired             *10.654**       11.762          N/A          N/A       10.807     *10.749*
  screw7_p050 grasp contained native          7.884     *3.523**          N/A          N/A        8.068      *5.691*
  screw7_p050 grasp contained paired          7.506     *3.522**          N/A          N/A        7.697      *6.063*
  screw7_p050 pose contained native         *8.372*        9.874       10.135     *4.922**      *8.300*        9.647
  screw7_p050 pose contained paired         *9.303*       10.062          N/A          N/A     *8.861**        9.725
  soft12 grasp contained native               0.847     *0.332**          N/A          N/A        0.578      *0.453*
  soft12 grasp contained paired               0.796     *0.332**          N/A          N/A        0.845      *0.432*
  soft12 pose contained native                1.368     *0.551**        1.570      *1.051*        0.588      *0.579*
  soft12 pose contained paired                0.985     *0.551**      *0.855*        0.978        0.603      *0.579*

  Table 3 -- mean runtime, s, over ALL cells, each clamped at the 180 s clock
  lower is better; this machine only, never compared across machines. Clamped because SNOPT overruns its clock on cycling cells (none feasible) once the iteration budget is lifted
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native                 28.66      *2.14**     *180.00*     *180.00*        24.00      *19.42*
  iiwa grasp contained paired                 26.56      *2.14**     *179.25*       180.00        27.54      *19.42*
  iiwa pose contained native                   2.19      *0.09**      *79.99*       174.21         2.61       *0.25*
  iiwa pose contained paired                   4.51      *0.09**     *140.93*       174.21        15.64       *1.03*
  panda grasp contained native                11.62       *8.11*      *57.79*       180.00      *5.46**        13.05
  panda grasp contained paired                18.79      *7.95**     *180.00*     *180.00*        20.51      *13.05*
  panda pose contained native                  2.20       *0.29*      *81.85*       177.10         3.18      *0.26**
  panda pose contained paired                  7.04       *0.29*     *143.39*       177.09        10.82      *0.26**
  screw7_p050 grasp contained native          55.56      *2.37**     *180.00*     *180.00*        28.54      *15.10*
  screw7_p050 grasp contained paired          54.77      *2.37**     *178.88*       180.00        24.17      *15.06*
  screw7_p050 pose contained native            3.88      *0.11**      *78.84*       173.25         5.30       *0.23*
  screw7_p050 pose contained paired            6.91      *0.11**     *152.72*       173.21        10.52       *0.23*
  soft12 grasp contained native               12.33     *10.31**     *180.00*     *180.00*        30.39      *22.92*
  soft12 grasp contained paired               16.24     *10.60**     *179.29*       180.00        34.69      *22.96*
  soft12 pose contained native                 2.21      *0.36**      *50.78*       178.68         6.52       *1.48*
  soft12 pose contained paired                10.27      *0.36**      *65.01*       178.63        28.10       *1.47*

  Table 4 -- median major iterations over solved cells
  lower is better; AL is N/A BY CONSTRUCTION -- NloptSolverDetails carries a single status and NLopt has no major iteration to count
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native                   405        *229*          N/A          N/A          805       *173**
  iiwa grasp contained paired                   365        *229*          N/A          N/A          652       *173**
  iiwa pose contained native                     44         *28*          N/A          N/A           73        *22**
  iiwa pose contained paired                    102         *28*          N/A          N/A          323        *22**
  panda grasp contained native                *169*          908          N/A          N/A          165       *164**
  panda grasp contained paired                *264*          908          N/A          N/A          446       *164**
  panda pose contained native                    36         *33*          N/A          N/A           47        *21**
  panda pose contained paired                   100         *33*          N/A          N/A          238        *21**
  screw7_p050 grasp contained native            559        *283*          N/A          N/A         1008       *190**
  screw7_p050 grasp contained paired            611        *283*          N/A          N/A          822       *190**
  screw7_p050 pose contained native              70         *32*          N/A          N/A          126        *22**
  screw7_p050 pose contained paired             128         *32*          N/A          N/A          304        *22**
  soft12 grasp contained native               *92**          126          N/A          N/A          670        *195*
  soft12 grasp contained paired               *82**          126          N/A          N/A          648        *195*
  soft12 pose contained native                *19**           22          N/A          N/A          205         *40*
  soft12 pose contained paired                   40        *22**          N/A          N/A          609         *40*

=== PER-SOLVER DETAIL (discordant counts, McNemar p, timeouts)

=== IPOPT (interior point)   [acceptable-point early stop (acceptable_tol 1e-3, acceptable_iter 1)]

  THE STATUS QUO (contained targets)
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  iiwa grasp contained native               461  452   27   18    0.233          tie      405    229   13.62   0.67   4.978   2.839  434   15    0
  iiwa grasp contained paired               464  452   27   15   0.0884          tie      365    229   11.14   0.67   5.013   2.839  437   16    0
  iiwa pose contained (tip) native          470  299  175    4  1.1e-46      learned       44     28    1.14   0.06   6.242   6.757  295    0    0
  iiwa pose contained (tip) paired          422  299  158   35 7.29e-20      learned      102     28    2.72   0.06   6.011   6.704  264    0    0
  panda grasp contained native              476  374  105    3 1.29e-27      learned      169    908    6.31   4.11   7.626   5.602  371    3    0
  panda grasp contained paired              474  374  106    6 9.76e-25      learned      264    908    9.82   4.00   7.067   5.602  368    7    0
  panda pose contained (tip) native         461  217  248    4 4.61e-68      learned       36     33    1.12   0.09  11.136  11.861  213    0    0
  panda pose contained (tip) paired         407  217  219   29 2.88e-37      learned      100     33    4.11   0.09  10.654  11.762  188    0    0
  screw7_p050 grasp contained native        427  424   50   47    0.839          tie      559    283   27.43   0.91   7.884   3.523  377   53    1
  screw7_p050 grasp contained paired        429  424   47   42    0.672          tie      611    283   27.88   0.92   7.506   3.522  382   49    1
  screw7_p050 pose contained (tip) native   457  313  155   11 1.08e-33      learned       70     32    2.58   0.07   8.372   9.874  302    0    0
  screw7_p050 pose contained (tip) paired   421  313  136   28 3.16e-18      learned      128     32    5.10   0.07   9.303  10.062  285    0    0
  soft12 grasp contained native             474  456   23    5 0.000912      learned       92    126    5.69   1.52   0.847   0.332  451    6    7
  soft12 grasp contained paired             468  455   21    8   0.0241      learned       82    126    4.00   1.56   0.796   0.332  447   11    8
  soft12 pose contained (tip) native        477  334  144    1 6.55e-42      learned       19     22    0.55   0.17   1.368   0.551  333    0    0
  soft12 pose contained (tip) paired        436  334  132   30 1.83e-16      learned       40     22    1.53   0.17   0.985   0.551  304    3    0

=== SNOPT (SQP)   [Major step limit = 0.5]

  THE STATUS QUO (contained targets)
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  iiwa grasp contained native               204  308   65  169 7.49e-12  joint space      805    173   12.76   0.47   5.421   5.388  139   21   32
  iiwa grasp contained paired               243  308   86  151 2.89e-05  joint space      652    173   11.11   0.48   5.625   4.728  157   23   32
  iiwa pose contained (tip) native          444  270  186   12 2.87e-41      learned       73     22    0.99   0.04   5.975   6.858  258    0    0
  iiwa pose contained (tip) paired          251  270  102  121    0.228          tie      323     22    5.79   0.04   6.877   6.756  149    2    1
  panda grasp contained native              441  306  157   22 2.56e-26      learned      165    164    2.80   0.55   7.040   7.437  284    2   21
  panda grasp contained paired              302  306  106  110    0.838          tie      446    164    9.86   0.55   7.384   7.567  196   10   21
  panda pose contained (tip) native         440  178  277   15 1.35e-63      learned       47     21    0.64   0.04  10.366  10.774  163    0    0
  panda pose contained (tip) paired         275  178  158   61 4.22e-11      learned      238     21    5.06   0.04  10.807  10.749  117    0    0
  screw7_p050 grasp contained native        163  300   52  189 1.85e-19  joint space     1008    190   19.73   0.54   8.068   5.691  111   13   29
  screw7_p050 grasp contained paired        212  300   72  160 7.46e-09  joint space      822    190   17.45   0.54   7.697   6.063  140    5   29
  screw7_p050 pose contained (tip) native   425  269  185   29 5.32e-29      learned      126     22    2.33   0.04   8.300   9.647  240    0    0
  screw7_p050 pose contained (tip) paired   259  269  105  115    0.544          tie      304     22    7.69   0.04   8.861   9.725  154    0    0
  soft12 grasp contained native             347  360   83   96     0.37          tie      670    195   20.24   2.02   0.578   0.453  264    9   38
  soft12 grasp contained paired             320  360   78  118  0.00521  joint space      648    195   19.76   2.03   0.845   0.432  242   23   38
  soft12 pose contained (tip) native        479  303  177    1 9.34e-52      learned      205     40    3.72   0.19   0.588   0.579  302    0    0
  soft12 pose contained (tip) paired        393  303  143   53 9.64e-11      learned      609     40   15.90   0.18   0.603   0.579  250    0    0

=== NLopt (augmented Lagrangian)   [LD_AUGLAG + LD_MMA inner + inner xtol_rel = ftol_rel = 1e-3]

  THE STATUS QUO (contained targets)
  work, wall clock and violation are MEANS/MEDIANS OVER ALL 480 CELLS here, not over
  successes: this column times out most cells, so a median over successes would
  describe the handful it got right. 'jac/cell' is the program's own network-Jacobian
  counter and is NOT comparable across arms (identity map on the joint-space arm).
  row                                         L   JS   L+  JS+        p      verdict  L jac/c  JS jc     L s   JS s   Lcost  JScost    n  LTO  JTO
  iiwa grasp contained native                 0    0    0    0        1          tie    12923  12407  180.06 180.06      --      --    0  480  480
  iiwa grasp contained paired                 2    0    2    0      0.5          tie    12709  12407  179.33 180.06      --      --    0  478  480
  iiwa pose contained (tip) native          298   31  278   11 5.08e-68      learned     4025   9550   80.01 174.26   6.511   4.988   20  199  463
  iiwa pose contained (tip) paired          118   31  101   14 1.98e-17      learned     7066   9550  140.98 174.27   7.062   5.807   17  369  463
  panda grasp contained native              327    0  327    0 7.32e-99      learned     3881  13417   57.81 180.06      --      --    0  153  480
  panda grasp contained paired                0    0    0    0        1          tie    11916  13416  180.07 180.06      --      --    0  480  480
  panda pose contained (tip) native         289   12  278    1 5.77e-82      learned     3988  10455   81.87 177.15   9.346   9.088   11  203  469
  panda pose contained (tip) paired         114   12  107    5 5.42e-26      learned     6864  10469  143.48 177.16      --      --    7  371  469
  screw7_p050 grasp contained native          0    0    0    0        1          tie    12925  12526  180.12 180.09      --      --    0  480  480
  screw7_p050 grasp contained paired          3    0    3    0     0.25          tie    12385  12526  178.97 180.07      --      --    0  477  480
  screw7_p050 pose contained (tip) native   303   30  280    7 2.44e-73      learned     3794   9811   78.87 173.32  10.135   4.922   23  189  455
  screw7_p050 pose contained (tip) paired    87   30   81   24 2.08e-08      learned     7313   9829  152.82 173.28      --      --    6  400  455
  soft12 grasp contained native               0    0    0    0        1          tie     6210   3719  180.10 180.09      --      --    0  480  480
  soft12 grasp contained paired               2    0    2    0      0.5          tie     6868   3754  179.39 180.09      --      --    0  478  480
  soft12 pose contained (tip) native        358   51  318   11 1.96e-79      learned     1757   4404   50.80 178.76   1.570   1.051   40  126  469
  soft12 pose contained (tip) paired        325   52  284   11 9.98e-70      learned     2319   4443   65.05 178.72   0.855   0.978   41  161  469


=== PRE-REGISTERED FLAG CRITERIA
Named in advance so 'did the story change' is a printed verdict, not a judgement.

--- 1. Did any learned-vs-joint-space verdict move?  (45 s -> 180 s, same grid)
  iiwa grasp contained native / ipopt                 joint space -> tie           MOVED  <-- FLAG
  iiwa grasp contained paired / ipopt                 joint space -> tie           MOVED  <-- FLAG
  iiwa pose contained (tip) native / ipopt                learned -> learned     
  iiwa pose contained (tip) paired / ipopt                learned -> learned     
  panda grasp contained native / ipopt                    learned -> learned     
  panda grasp contained paired / ipopt                    learned -> learned     
  panda pose contained (tip) native / ipopt               learned -> learned     
  panda pose contained (tip) paired / ipopt               learned -> learned     
  screw7_p050 grasp contained native / ipopt                      -> tie          UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_mugshelf_480_45_native)
  screw7_p050 grasp contained paired / ipopt                      -> tie          UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_mugshelf_480_45_paired)
  screw7_p050 pose contained (tip) native / ipopt                 -> learned      UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_posetip_480_45_native)
  screw7_p050 pose contained (tip) paired / ipopt                 -> learned      UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_posetip_480_45_paired)
  soft12 grasp contained native / ipopt                           -> learned      UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_mugshelf_480_45_native)
  soft12 grasp contained paired / ipopt                           -> learned      UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_mugshelf_480_45_paired)
  soft12 pose contained (tip) native / ipopt                      -> learned      UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_posetip_480_45_native)
  soft12 pose contained (tip) paired / ipopt                      -> learned      UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_posetip_480_45_paired)
  iiwa grasp contained native / snopt                 joint space -> joint space 
  iiwa grasp contained paired / snopt                 joint space -> joint space 
  iiwa pose contained (tip) native / snopt                learned -> learned     
  iiwa pose contained (tip) paired / snopt                    tie -> tie         
  panda grasp contained native / snopt                    learned -> learned     
  panda grasp contained paired / snopt                        tie -> tie         
  panda pose contained (tip) native / snopt               learned -> learned     
  panda pose contained (tip) paired / snopt               learned -> learned     
  screw7_p050 grasp contained native / snopt                      -> joint space  UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_mugshelf_480_45_native_mstep0p5)
  screw7_p050 grasp contained paired / snopt                      -> joint space  UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_mugshelf_480_45_paired_mstep0p5)
  screw7_p050 pose contained (tip) native / snopt                 -> learned      UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_posetip_480_45_native_mstep0p5)
  screw7_p050 pose contained (tip) paired / snopt                 -> tie          UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_posetip_480_45_paired_mstep0p5)
  soft12 grasp contained native / snopt                           -> tie          UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_mugshelf_480_45_native_mstep0p5)
  soft12 grasp contained paired / snopt                           -> joint space  UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_mugshelf_480_45_paired_mstep0p5)
  soft12 pose contained (tip) native / snopt                      -> learned      UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_posetip_480_45_native_mstep0p5)
  soft12 pose contained (tip) paired / snopt                      -> learned      UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_posetip_480_45_paired_mstep0p5)
  iiwa grasp contained native / nlopt                         tie -> tie           [ref is 60 cells, NOT paired]
  iiwa grasp contained paired / nlopt                         tie -> tie           [ref is 60 cells, NOT paired]
  iiwa pose contained (tip) native / nlopt                learned -> learned       [ref is 60 cells, NOT paired]
  iiwa pose contained (tip) paired / nlopt                learned -> learned       [ref is 60 cells, NOT paired]
  panda grasp contained native / nlopt                    learned -> learned       [ref is 60 cells, NOT paired]
  panda grasp contained paired / nlopt                        tie -> tie           [ref is 60 cells, NOT paired]
  panda pose contained (tip) native / nlopt               learned -> learned       [ref is 60 cells, NOT paired]
  panda pose contained (tip) paired / nlopt               learned -> learned       [ref is 60 cells, NOT paired]
  screw7_p050 grasp contained native / nlopt                      -> tie          UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_mugshelf_60_45_native_mmaloose)
  screw7_p050 grasp contained paired / nlopt                      -> tie          UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_mugshelf_60_45_paired_mmaloose)
  screw7_p050 pose contained (tip) native / nlopt                 -> learned      UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_posetip_60_45_native_mmaloose)
  screw7_p050 pose contained (tip) paired / nlopt                 -> learned      UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_posetip_60_45_paired_mmaloose)
  soft12 grasp contained native / nlopt                           -> tie          UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_mugshelf_60_45_native_mmaloose)
  soft12 grasp contained paired / nlopt                           -> tie          UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_mugshelf_60_45_paired_mmaloose)
  soft12 pose contained (tip) native / nlopt                      -> learned      UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_posetip_60_45_native_mmaloose)
  soft12 pose contained (tip) paired / nlopt                      -> learned      UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_posetip_60_45_paired_mmaloose)
  => 2 verdict(s) moved, 24 row(s) unpaired

--- 2. The IPOPT-vs-SNOPT gap (learned arm, per row). Size is the reported quantity.
  row                                       IPOPT  SNOPT   gap   ITO   STO
  iiwa grasp contained native                 461    204   257    15    21
  iiwa grasp contained paired                 464    243   221    16    23
  iiwa pose contained (tip) native            470    444    26     0     0
  iiwa pose contained (tip) paired            422    251   171     0     2
  panda grasp contained native                476    441    35     3     2
  panda grasp contained paired                474    302   172     7    10
  panda pose contained (tip) native           461    440    21     0     0
  panda pose contained (tip) paired           407    275   132     0     0
  screw7_p050 grasp contained native          427    163   264    53    13
  screw7_p050 grasp contained paired          429    212   217    49     5
  screw7_p050 pose contained (tip) native     457    425    32     0     0
  screw7_p050 pose contained (tip) paired     421    259   162     0     0
  soft12 grasp contained native               474    347   127     6     9
  soft12 grasp contained paired               468    320   148    11    23
  soft12 pose contained (tip) native          477    479    -2     0     0
  soft12 pose contained (tip) paired          436    393    43     3     0
  => IPOPT ahead on 15/16 rows, median gap 140 cells. Compare against the 45 s gaps in CLAUDE.md; a widening gap is the predicted direction, so report its SIZE.

--- 3. NLopt at 180 s under the adopted configuration (previously untested)
  12 of 16 rows solve anything at all on the learned arm.
  learned successes: iiwa mugshelf nat 0, iiwa mugshelf pai 2, iiwa posetip nat 298, iiwa posetip pai 118, pand mugshelf nat 327, pand mugshelf pai 0, pand posetip nat 289, pand posetip pai 114, scre mugshelf nat 0, scre mugshelf pai 3, scre posetip nat 303, scre posetip pai 87, soft mugshelf nat 0, soft mugshelf pai 2, soft posetip nat 358, soft posetip pai 325
  The 180 s arm measured at Drake's NLopt defaults was flat against 45 s on ten
  of twelve rows, but that predates the adopted configuration -- so this is the
  first measurement of the two changes together.
  HOW TO READ THIS COLUMN, and it is a RESULT IN OUR FAVOUR, not a spoiled
  comparison: under an augmented Lagrangian the joint-space arm is near-dead on
  every row while the learned arm solves a substantial fraction of the pose rows.
  That is the strongest form the comparison takes anywhere -- the baseline is at
  the floor -- and it is attributable, because only the solver differs and the
  joint-space arm is the EASIER problem (7 variables, no network), so its
  collapse is a property of NLopt on this program and not of the harness.
  Two narrow caveats, neither touching the pose rows: rows where BOTH arms are
  near zero (historically the four iiwa grasp rows) carry no comparison, and cost
  needs cells both arms solved, of which there are few -- hence the dashes.
