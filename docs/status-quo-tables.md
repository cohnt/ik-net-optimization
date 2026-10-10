# The campaign of record: the tables

**Stage REMEASURE**, measured and accepted by Thomas 2026-10-09: the whole record re-measured on the
fixed wsg scene (`between_fingers` yaw 0 -> 1.57, 452d784; the iiwa, soft PCS, screw and GVS grasp
targets previously started with the mug handle 18 mm inside a finger), with the five unified robot
settings adopted (0381807; Thomas: *"Adopt the settings unification everywhere, including the panda
joint limits"*), CUDA graphs and the lifted iteration budgets. `CLAUDE.md` carries what these tables
**say**; this file carries the tables themselves, so a session pays for them only when it needs a
number.

**`scripts/report_statusquo.py` owns every table here and regenerates all of them from the persisted
runs under `results/`, together with the verdict tally and the three pre-registered flag criteria. Do
not hand-maintain them.** If a number here disagrees with that script, the script is right. Regenerate
with `{ <this header>; echo ---; echo; scripts/report_statusquo.py; } > docs/status-quo-tables.md`.
`scripts/report_statusquo.py --legacy` prints the superseded record (STATUSQUO + SOFT12 + SCREW with
the ITCAP / SCREWCAP lifted rows substituted); `scripts/report_remeasure.py` prints the before/after,
the attribution of every move to the scene fix or the settings, the trust-region A/B and the
acceptance checks.

Conditions: 480 cells = 60 targets x 8 guesses, seed 1 (out of sample), **180 s**, `--compile --set
flow_cuda_graph=True`, IPOPT `max_iter` 1e6, SNOPT 1e5 majors / 1e8 minors, adopted rungs (Panda `n6`,
iiwa `n4`, soft PCS `soft12` `n6`, screw `screw7_p050` `n6`), hardened
scene, shelf-contained targets at the fingertips, arms `learned,numerical`, both start protocols, each
solver at its adopted configuration, Drake nightly `0.0.20260918`. **Run at PROCS=8 under `MPS=1`, a
development-throughput condition, not the paper's one solve per GPU**, so the wall-clock columns are
development-grade. **48 logical runs: four robots x two experiments x two protocols x three
solvers**, the IPOPT and SNOPT 32 from stage REMEASURE and the NLopt 16 from stage REMEASURE_NLOPT
(same conditions and tag family, queued last; jobs 5868202/03/04/07, collected 2026-10-10). The old
record's NLopt runs stay on disk and are read only by `--legacy`.

**The GVS arm is OUTSIDE the record** (Thomas, 2026-10-05: *"the GVS arm doesn't help our story. We
can still merge it into main, but it certainly doesn't replace the other soft arm"*). Stage REMEASURE
measured it under identical conditions (`gvs_pushrod9_o1` `n6`, IPOPT and SNOPT; it never ran
NLopt), and its eight rows print as a separate block at the end, with their own tally. They enter
none of the record's tables, tally or flags.

The flag criteria's 45 s references predate REMEASURE, so flag 1 on a wsg grasp row conflates the
cap with the scene fix and the settings; NLopt's references are 60-cell runs and are not paired at
all. `scripts/report_remeasure.py nlopt` prints the NLopt rows' before/after against the old record.

---

THE CAMPAIGN OF RECORD -- stage REMEASURE (2026-10-09), 480 cells, 180 s cap, seed 1
Conditions: the FIXED wsg scene (finray between_fingers yaw 0 -> 1.57, 452d784), the five
  UNIFIED robot settings (0381807; the Panda's joint-space bound is ConfigLimits(), no longer
  +-10 rad), CUDA graphs (--compile, flow_cuda_graph=True), LIFTED iteration budgets (IPOPT
  max_iter 1e6; SNOPT 1e5 majors, 1e8 minors), adopted rungs (panda n6, iiwa n4, soft12 n6,
  screw7_p050 n6), hardened scene, shelf-contained at the fingertips.
Run at PROCS=8 under MPS=1: DEVELOPMENT throughput, NOT the paper's one-solve-per-GPU
  (PROCS=2) condition, so the wall-clock columns are development-grade.
IPOPT + SNOPT: 32 of 32 runs from stage REMEASURE (four robots x two experiments
  x two protocols x two solvers).
NLopt: 16 of 16 runs from stage REMEASURE_NLOPT (2026-10-10; same conditions, same
  tag family). 48 logical runs in all.
The GVS arm (gvs_pushrod9_o1) was measured in the same stage under identical
  conditions and is OUTSIDE the record: its rows print as a separate block at the end,
  with their own tally, and enter none of the record's tables, tally or flags.
Arms: learned vs joint space (numerical). No analytic baseline is fielded.
NOTE: solver options move the JOINT-SPACE arm too -- that arm never evaluates the
      network, so a moving JS column is a property of the problem, not drift.

=== HEADLINE TABLES (learned vs joint space, per solver)
  IP = interior point (IPOPT), AL = augmented Lagrangian (NLOPT),
  SQP = sequential quadratic programming (SNOPT). *better* of each pair is starred;
  a trailing * marks the best in the row. Every row prints, zeros included.

  Table 1 -- success rate of 480 cells
  higher is better; ties are by exact McNemar (p >= 0.05), not numeric equality
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native              *1.000**        0.944      *0.652*        0.000      *0.908*        0.725
  iiwa grasp contained paired              *0.998**        0.944      *0.023*        0.000        0.575      *0.725*
  iiwa pose contained native               *0.963**        0.650      *0.623*        0.085      *0.921*        0.567
  iiwa pose contained paired               *0.846**        0.650      *0.233*        0.085        0.485      *0.567*
  panda grasp contained native             *1.000**        0.935      *0.671*        0.000      *0.960*        0.752
  panda grasp contained paired             *0.998**        0.935      *0.006*      *0.000*        0.625      *0.752*
  panda pose contained native              *0.954**        0.417      *0.633*        0.294      *0.915*        0.402
  panda pose contained paired              *0.823**        0.417      *0.271*      *0.294*      *0.588*        0.402
  screw7_p050 grasp contained native       *0.990**        0.921      *0.673*        0.100      *0.833*        0.721
  screw7_p050 grasp contained paired       *0.983**        0.921        0.008      *0.100*        0.540      *0.721*
  screw7_p050 pose contained native        *0.956**        0.667      *0.629*        0.175      *0.873*        0.554
  screw7_p050 pose contained paired        *0.894**        0.667      *0.194*      *0.175*      *0.560*      *0.554*
  soft12 grasp contained native            *1.000**        0.983      *0.750*        0.077      *0.996*        0.785
  soft12 grasp contained paired            *0.994**      *0.983*        0.008      *0.077*        0.729      *0.785*
  soft12 pose contained native              *0.994*        0.696      *0.746*        0.106     *0.998**        0.631
  soft12 pose contained paired             *0.910**        0.696      *0.677*        0.106      *0.819*        0.631

  Of the 48 solver x experiment cells, learned wins 36, ties 5, loses 7  (verdicts by exact McNemar on the same cells the table shows)
    IP   learned 15, ties 1, joint space 0
    AL   learned 11, ties 3, joint space 2
    SQP  learned 10, ties 1, joint space 5

  Table 2 -- optimal cost, cells BOTH arms solved, learned-only regularizers excluded
  lower is better; N/A means fewer than 10 shared solved cells, so no comparison exists
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native                 7.952     *2.911**          N/A          N/A        7.474      *6.928*
  iiwa grasp contained paired                 7.573     *2.943**          N/A          N/A        7.823      *6.514*
  iiwa pose contained native                *8.987*        9.195        9.147     *4.994**      *9.134*        9.387
  iiwa pose contained paired                *9.030*        9.166       10.603     *5.482**      *7.907*        9.362
  panda grasp contained native                4.506     *2.231**          N/A          N/A        3.730      *3.704*
  panda grasp contained paired                4.135     *2.231**          N/A          N/A        4.153      *3.760*
  panda pose contained native               *6.707*        7.693        7.138     *5.065**      *6.457*        6.968
  panda pose contained paired               *6.403*        7.580        5.537     *4.368**      *6.620*        7.147
  screw7_p050 grasp contained native          9.020      *1.906*       13.178     *0.996**        8.689      *6.977*
  screw7_p050 grasp contained paired          9.363     *1.829**          N/A          N/A        8.657      *6.802*
  screw7_p050 pose contained native        *11.479*       12.463       11.300     *2.945**     *11.505*       12.431
  screw7_p050 pose contained paired          13.124     *12.002*       13.284     *2.892**       13.053     *12.707*
  soft12 grasp contained native               0.626      *0.244*        1.628     *0.135**      *0.260*        0.287
  soft12 grasp contained paired               0.704     *0.244**          N/A          N/A        0.608      *0.305*
  soft12 pose contained native                1.368     *0.551**        1.570      *1.051*        0.588      *0.579*
  soft12 pose contained paired                0.985     *0.551**      *0.855*        0.978        0.603      *0.579*

  Table 3 -- mean runtime, s, over ALL cells, each clamped at the 180 s clock
  lower is better; this machine only, never compared across machines. Clamped because SNOPT overruns its clock on cycling cells (none feasible) once the iteration budget is lifted
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native                  3.01      *1.09**      *63.48*       180.00       *3.45*        14.81
  iiwa grasp contained paired                  7.10      *1.10**     *175.89*       180.00        15.92      *14.88*
  iiwa pose contained native                   0.75      *0.10**      *78.74*       169.95       *1.49*         1.60
  iiwa pose contained paired                   1.82      *0.10**     *142.06*       169.95        13.96       *1.60*
  panda grasp contained native              *3.52**         6.24      *60.01*       180.00       *4.18*        13.12
  panda grasp contained paired              *5.86**         6.27     *178.88*       180.00        13.34      *13.10*
  panda pose contained native                  0.93      *0.54**      *77.68*       142.98         1.91       *1.43*
  panda pose contained paired                  2.37      *0.55**     *138.10*       142.98         9.11       *1.45*
  screw7_p050 grasp contained native           7.81      *1.12**      *60.71*       162.88        10.37       *9.28*
  screw7_p050 grasp contained paired          12.18      *1.12**       178.52     *162.87*        14.54       *9.36*
  screw7_p050 pose contained native            1.23      *0.13**      *78.08*       159.39         2.86       *1.81*
  screw7_p050 pose contained paired            2.28      *0.13**     *149.11*       159.37         6.22       *1.80*
  soft12 grasp contained native             *2.37**         3.44      *49.85*       166.74       *7.73*        20.41
  soft12 grasp contained paired                6.62      *3.39**       178.52     *166.74*        28.86      *20.35*
  soft12 pose contained native                 0.78      *0.37**      *50.56*       178.70         4.74       *2.49*
  soft12 pose contained paired                 9.27      *0.37**      *64.88*       178.70        24.96       *2.51*

  Table 4 -- median major iterations over solved cells
  lower is better; AL is N/A BY CONSTRUCTION -- NloptSolverDetails carries a single status and NLopt has no major iteration to count
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  iiwa grasp contained native                   134       *118**          N/A          N/A          264        *128*
  iiwa grasp contained paired                   246       *118**          N/A          N/A          599        *128*
  iiwa pose contained native                     50         *28*          N/A          N/A           78        *21**
  iiwa pose contained paired                    106         *28*          N/A          N/A          331        *21**
  panda grasp contained native                *134*          206          N/A          N/A          191       *129**
  panda grasp contained paired                *199*          206          N/A          N/A          426       *129**
  panda pose contained native                    37         *33*          N/A          N/A           47        *21**
  panda pose contained paired                   107         *33*          N/A          N/A          247        *21**
  screw7_p050 grasp contained native            227       *116**          N/A          N/A          544        *126*
  screw7_p050 grasp contained paired            398       *116**          N/A          N/A          818        *126*
  screw7_p050 pose contained native              65         *32*          N/A          N/A          125        *22**
  screw7_p050 pose contained paired             143         *32*          N/A          N/A          360        *22**
  soft12 grasp contained native               *38**           53          N/A          N/A          367        *149*
  soft12 grasp contained paired                  57        *53**          N/A          N/A          660        *149*
  soft12 pose contained native                *19**           22          N/A          N/A          205         *40*
  soft12 pose contained paired                   40        *22**          N/A          N/A          609         *40*

=== PER-SOLVER DETAIL (discordant counts, McNemar p, timeouts)

=== IPOPT (interior point)   [acceptable-point early stop (acceptable_tol 1e-3, acceptable_iter 1)]

  THE STATUS QUO (contained targets)
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  iiwa grasp contained native               480  453   27    0 1.49e-08      learned      134    118    1.54   0.33   7.952   2.911  453    0    0
  iiwa grasp contained paired               479  453   27    1 2.16e-07      learned      246    118    2.83   0.34   7.573   2.943  452    0    0
  iiwa pose contained (tip) native          462  312  160   10 6.04e-36      learned       50     28    0.46   0.07   8.987   9.195  302    0    0
  iiwa pose contained (tip) paired          406  312  138   44 1.71e-12      learned      106     28    1.08   0.07   9.030   9.166  268    0    0
  panda grasp contained native              480  449   31    0 9.31e-10      learned      134    206    2.20   0.77   4.506   2.231  449    0    8
  panda grasp contained paired              479  449   30    0 1.86e-09      learned      199    206    3.35   0.79   4.135   2.231  449    0    8
  panda pose contained (tip) native         458  200  269   11 1.83e-65      learned       37     33    0.46   0.10   6.707   7.693  189    0    1
  panda pose contained (tip) paired         395  200  225   30 3.99e-38      learned      107     33    1.59   0.10   6.403   7.580  170    0    1
  screw7_p050 grasp contained native        475  442   38    5  2.5e-07      learned      227    116    3.42   0.28   9.020   1.906  437    3    1
  screw7_p050 grasp contained paired        472  442   38    8 9.25e-06      learned      398    116    5.53   0.28   9.363   1.829  434    7    1
  screw7_p050 pose contained (tip) native   459  320  152   13  3.1e-31      learned       65     32    0.81   0.08  11.479  12.463  307    0    0
  screw7_p050 pose contained (tip) paired   429  320  141   32 1.59e-17      learned      143     32    1.73   0.08  13.124  12.002  288    0    0
  soft12 grasp contained native             480  472    8    0  0.00781      learned       38     53    0.88   0.53   0.626   0.244  472    0    0
  soft12 grasp contained paired             477  472    8    3    0.227          tie       57     53    1.32   0.52   0.704   0.244  469    3    0
  soft12 pose contained (tip) native        477  334  144    1 6.55e-42      learned       19     22    0.29   0.18   1.368   0.551  333    0    0
  soft12 pose contained (tip) paired        437  334  133   30 1.12e-16      learned       40     22    0.84   0.18   0.985   0.551  304   18    0

=== SNOPT (SQP)   [Major step limit = 0.5]

  THE STATUS QUO (contained targets)
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  iiwa grasp contained native               436  348  119   31 2.34e-13      learned      264    128    2.15   0.34   7.474   6.928  317    0   27
  iiwa grasp contained paired               276  348   71  143 9.69e-07  joint space      599    128    5.89   0.35   7.823   6.514  205   11   27
  iiwa pose contained (tip) native          442  272  188   18 6.97e-37      learned       78     21    0.61   0.04   9.134   9.387  254    0    2
  iiwa pose contained (tip) paired          233  272  101  140   0.0142  joint space      331     21    3.28   0.04   7.907   9.362  132    8    2
  panda grasp contained native              461  361  114   14 1.16e-20      learned      191    129    2.17   0.40   3.730   3.704  347    0   24
  panda grasp contained paired              300  361   64  125 1.08e-05  joint space      426    129    5.84   0.40   4.153   3.760  236    5   24
  panda pose contained (tip) native         439  193  259   13 1.48e-60      learned       47     21    0.38   0.04   6.457   6.968  180    1    1
  panda pose contained (tip) paired         282  193  153   64 1.37e-09      learned      247     21    3.34   0.04   6.620   7.147  129    6    1
  screw7_p050 grasp contained native        400  346  110   56 3.36e-05      learned      544    126    6.26   0.31   8.689   6.977  290    1   16
  screw7_p050 grasp contained paired        259  346   72  159 1.03e-08  joint space      818    126   10.55   0.31   8.657   6.802  187    3   17
  screw7_p050 pose contained (tip) native   419  266  182   29 2.75e-28      learned      125     22    1.36   0.04  11.505  12.431  237    0    3
  screw7_p050 pose contained (tip) paired   269  266  119  116    0.896          tie      360     22    4.85   0.04  13.053  12.707  150    0    3
  soft12 grasp contained native             478  377  103    2 2.74e-28      learned      367    149    5.58   1.20   0.260   0.287  375    0   34
  soft12 grasp contained paired             350  377   74  101   0.0491  joint space      660    149   16.39   1.22   0.608   0.305  276   17   33
  soft12 pose contained (tip) native        479  303  177    1 9.34e-52      learned      205     40    2.85   0.19   0.588   0.579  302    0    4
  soft12 pose contained (tip) paired        393  303  143   53 9.64e-11      learned      609     40   13.49   0.20   0.603   0.579  250    8    4

=== NLopt (augmented Lagrangian)   [LD_AUGLAG + LD_MMA inner + inner xtol_rel = ftol_rel = 1e-3]

  THE STATUS QUO (contained targets)
  work, wall clock and violation are MEANS/MEDIANS OVER ALL 480 CELLS here, not over
  successes: this column times out most cells, so a median over successes would
  describe the handful it got right. 'jac/cell' is the program's own network-Jacobian
  counter and is NOT comparable across arms (identity map on the joint-space arm).
  row                                         L   JS   L+  JS+        p      verdict  L jac/c  JS jc     L s   JS s   Lcost  JScost    n  LTO  JTO
  iiwa grasp contained native               313    0  313    0  1.2e-94      learned     4557  12749   63.51 180.07      --      --    0  168  480
  iiwa grasp contained paired                11    0   11    0 0.000977      learned    12640  12749  175.98 180.08      --      --    0  469  480
  iiwa pose contained (tip) native          299   41  272   14 3.45e-63      learned     3900   9172   78.77 170.01   9.147   4.994   27  193  448
  iiwa pose contained (tip) paired          112   41  100   29 2.47e-10      learned     7102   9172  142.13 170.02  10.603   5.482   12  374  448
  panda grasp contained native              322    0  322    0 2.34e-97      learned     4053  13458   60.03 180.07      --      --    0  159  480
  panda grasp contained paired                3    0    3    0     0.25          tie    12114  13458  178.97 180.08      --      --    0  477  480
  panda pose contained (tip) native         304  141  203   40 1.91e-27      learned     3711   7810   77.71 143.03   7.138   5.065  101  185  356
  panda pose contained (tip) paired         130  141   83   94    0.452          tie     6821   7810  138.17 143.04   5.537   4.368   47  358  356
  screw7_p050 grasp contained native        323   48  284    9 5.03e-72      learned     4580  10754   60.74 162.95  13.178   0.996   39  160  432
  screw7_p050 grasp contained paired          4   48    4   48 1.31e-10  joint space    12546  10754  178.60 162.94      --      --    0  476  432
  screw7_p050 pose contained (tip) native   302   84  242   24 1.63e-46      learned     3694   8649   78.12 159.45  11.300   2.945   60  194  412
  screw7_p050 pose contained (tip) paired    93   84   76   67    0.504          tie     7304   8649  149.17 159.43  13.284   2.892   17  394  412
  soft12 grasp contained native             360   37  332    9 7.09e-86      learned     1755   3548   50.02 166.82   1.628   0.135   28  123  444
  soft12 grasp contained paired               4   37    4   37 1.03e-07  joint space     6956   3551  178.82 166.83      --      --    0  476  444
  soft12 pose contained (tip) native        358   51  318   11 1.96e-79      learned     1819   4421   50.60 178.81   1.570   1.051   40  126  469
  soft12 pose contained (tip) paired        325   51  284   10 7.42e-71      learned     2374   4384   64.91 178.79   0.855   0.978   41  161  469


=== PRE-REGISTERED FLAG CRITERIA
Named in advance so 'did the story change' is a printed verdict, not a judgement.
Their 45 s references predate stage REMEASURE: old settings everywhere, and on every wsg
grasp row (iiwa, soft12, screw7_p050) the defective scene. A move there conflates the cap
with the scene fix and the settings; scripts/report_remeasure.py attributes them.

--- 1. Did any learned-vs-joint-space verdict move?  (45 s -> 180 s, same grid)
  iiwa grasp contained native / ipopt                 joint space -> learned       MOVED  <-- FLAG
  iiwa grasp contained paired / ipopt                 joint space -> learned       MOVED  <-- FLAG
  iiwa pose contained (tip) native / ipopt                learned -> learned     
  iiwa pose contained (tip) paired / ipopt                learned -> learned     
  panda grasp contained native / ipopt                    learned -> learned     
  panda grasp contained paired / ipopt                    learned -> learned     
  panda pose contained (tip) native / ipopt               learned -> learned     
  panda pose contained (tip) paired / ipopt               learned -> learned     
  screw7_p050 grasp contained native / ipopt                      -> learned      UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_mugshelf_480_45_native)
  screw7_p050 grasp contained paired / ipopt                      -> learned      UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_mugshelf_480_45_paired)
  screw7_p050 pose contained (tip) native / ipopt                 -> learned      UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_posetip_480_45_native)
  screw7_p050 pose contained (tip) paired / ipopt                 -> learned      UNPAIRED (no sc_SOLVER2_screw7_p050_n6_ipopt_posetip_480_45_paired)
  soft12 grasp contained native / ipopt                           -> learned      UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_mugshelf_480_45_native)
  soft12 grasp contained paired / ipopt                           -> tie          UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_mugshelf_480_45_paired)
  soft12 pose contained (tip) native / ipopt                      -> learned      UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_posetip_480_45_native)
  soft12 pose contained (tip) paired / ipopt                      -> learned      UNPAIRED (no sc_SOLVER2_soft12_n6_ipopt_posetip_480_45_paired)
  iiwa grasp contained native / snopt                 joint space -> learned       MOVED  <-- FLAG
  iiwa grasp contained paired / snopt                 joint space -> joint space 
  iiwa pose contained (tip) native / snopt                learned -> learned     
  iiwa pose contained (tip) paired / snopt                    tie -> joint space   MOVED  <-- FLAG
  panda grasp contained native / snopt                    learned -> learned     
  panda grasp contained paired / snopt                        tie -> joint space   MOVED  <-- FLAG
  panda pose contained (tip) native / snopt               learned -> learned     
  panda pose contained (tip) paired / snopt               learned -> learned     
  screw7_p050 grasp contained native / snopt                      -> learned      UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_mugshelf_480_45_native_mstep0p5)
  screw7_p050 grasp contained paired / snopt                      -> joint space  UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_mugshelf_480_45_paired_mstep0p5)
  screw7_p050 pose contained (tip) native / snopt                 -> learned      UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_posetip_480_45_native_mstep0p5)
  screw7_p050 pose contained (tip) paired / snopt                 -> tie          UNPAIRED (no sc_SNOPTCOMBO_screw7_p050_n6_snopt_posetip_480_45_paired_mstep0p5)
  soft12 grasp contained native / snopt                           -> learned      UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_mugshelf_480_45_native_mstep0p5)
  soft12 grasp contained paired / snopt                           -> joint space  UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_mugshelf_480_45_paired_mstep0p5)
  soft12 pose contained (tip) native / snopt                      -> learned      UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_posetip_480_45_native_mstep0p5)
  soft12 pose contained (tip) paired / snopt                      -> learned      UNPAIRED (no sc_SNOPTCOMBO_soft12_n6_snopt_posetip_480_45_paired_mstep0p5)
  iiwa grasp contained native / nlopt                         tie -> learned       [ref is 60 cells, NOT paired]  MOVED  <-- FLAG
  iiwa grasp contained paired / nlopt                         tie -> learned       [ref is 60 cells, NOT paired]  MOVED  <-- FLAG
  iiwa pose contained (tip) native / nlopt                learned -> learned       [ref is 60 cells, NOT paired]
  iiwa pose contained (tip) paired / nlopt                learned -> learned       [ref is 60 cells, NOT paired]
  panda grasp contained native / nlopt                    learned -> learned       [ref is 60 cells, NOT paired]
  panda grasp contained paired / nlopt                        tie -> tie           [ref is 60 cells, NOT paired]
  panda pose contained (tip) native / nlopt               learned -> learned       [ref is 60 cells, NOT paired]
  panda pose contained (tip) paired / nlopt               learned -> tie           [ref is 60 cells, NOT paired]  MOVED  <-- FLAG
  screw7_p050 grasp contained native / nlopt                      -> learned      UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_mugshelf_60_45_native_mmaloose)
  screw7_p050 grasp contained paired / nlopt                      -> joint space  UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_mugshelf_60_45_paired_mmaloose)
  screw7_p050 pose contained (tip) native / nlopt                 -> learned      UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_posetip_60_45_native_mmaloose)
  screw7_p050 pose contained (tip) paired / nlopt                 -> tie          UNPAIRED (no sc_NLOPTTUNE_screw7_p050_n6_nlopt_posetip_60_45_paired_mmaloose)
  soft12 grasp contained native / nlopt                           -> learned      UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_mugshelf_60_45_native_mmaloose)
  soft12 grasp contained paired / nlopt                           -> joint space  UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_mugshelf_60_45_paired_mmaloose)
  soft12 pose contained (tip) native / nlopt                      -> learned      UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_posetip_60_45_native_mmaloose)
  soft12 pose contained (tip) paired / nlopt                      -> learned      UNPAIRED (no sc_NLOPTTUNE_soft12_n6_nlopt_posetip_60_45_paired_mmaloose)
  => 8 verdict(s) moved, 24 row(s) unpaired

--- 2. The IPOPT-vs-SNOPT gap (learned arm, per row). Size is the reported quantity.
  row                                       IPOPT  SNOPT   gap   ITO   STO
  iiwa grasp contained native                 480    436    44     0     0
  iiwa grasp contained paired                 479    276   203     0    11
  iiwa pose contained (tip) native            462    442    20     0     0
  iiwa pose contained (tip) paired            406    233   173     0     8
  panda grasp contained native                480    461    19     0     0
  panda grasp contained paired                479    300   179     0     5
  panda pose contained (tip) native           458    439    19     0     1
  panda pose contained (tip) paired           395    282   113     0     6
  screw7_p050 grasp contained native          475    400    75     3     1
  screw7_p050 grasp contained paired          472    259   213     7     3
  screw7_p050 pose contained (tip) native     459    419    40     0     0
  screw7_p050 pose contained (tip) paired     429    269   160     0     0
  soft12 grasp contained native               480    478     2     0     0
  soft12 grasp contained paired               477    350   127     3    17
  soft12 pose contained (tip) native          477    479    -2     0     0
  soft12 pose contained (tip) paired          437    393    44    18     8
  => IPOPT ahead on 15/16 rows, median gap 60 cells. Compare against the 45 s gaps in CLAUDE.md; a widening gap is the predicted direction, so report its SIZE.

--- 3. NLopt at 180 s under the adopted configuration (previously untested)
  Per row, native against paired (successes of 480; joint space's two protocols
  coincide, its native start being a random configuration). Verdict by exact McNemar.
  row                                 L nat  L pai  JS nat  JS pai  verdict nat  verdict pai
  iiwa grasp contained                  313     11       0       0      learned      learned
  iiwa pose contained (tip)             299    112      41      41      learned      learned
  panda grasp contained                 322      3       0       0      learned          tie
  panda pose contained (tip)            304    130     141     141      learned          tie
  screw7_p050 grasp contained           323      4      48      48      learned  joint space
  screw7_p050 pose contained (tip)      302     93      84      84      learned          tie
  soft12 grasp contained                360      4      37      37      learned  joint space
  soft12 pose contained (tip)           358    325      51      51      learned      learned
  Joint space solves NOTHING on 4 of 16 rows: iiwa grasp contained native, iiwa grasp contained paired, panda grasp contained native, panda grasp contained paired
  -- a property of NLopt on this program, joint space being the EASIER problem (no
  network), not of the harness.
  Both arms under 5% (24 cells) on 2 row(s), a verdict there being about a floor: iiwa grasp contained paired, panda grasp contained paired
  Cost needs cells both arms solved, of which there are few -- hence the N/A.


=== GVS ARM, MEASURED UNDER IDENTICAL CONDITIONS, OUTSIDE THE RECORD
Stage REMEASURE, the same conditions as the record above; IPOPT and SNOPT only (the GVS
arm never ran NLopt, so its AL column is N/A). Thomas, 2026-10-05: "the GVS arm doesn't
help our story ... it certainly doesn't replace the other soft arm".

=== HEADLINE TABLES -- GVS arm, measured under identical conditions, outside the record
  IP = interior point (IPOPT), AL = augmented Lagrangian (NLOPT),
  SQP = sequential quadratic programming (SNOPT). *better* of each pair is starred;
  a trailing * marks the best in the row. Every row prints, zeros included.

  Table 1 -- success rate of 480 cells
  higher is better; ties are by exact McNemar (p >= 0.05), not numeric equality
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  gvs_pushrod9_o1 grasp contained native     *1.000**        0.973          N/A          N/A      *0.992*        0.762
  gvs_pushrod9_o1 grasp contained paired     *0.979**      *0.973*          N/A          N/A      *0.715*      *0.762*
  gvs_pushrod9_o1 pose contained native     *1.000**        0.648          N/A          N/A      *1.000*        0.525
  gvs_pushrod9_o1 pose contained paired     *0.798**        0.648          N/A          N/A      *0.573*      *0.525*

  Of the 8 solver x experiment cells, learned wins 5, ties 3, loses 0  (verdicts by exact McNemar on the same cells the table shows)
    IP   learned 3, ties 1, joint space 0
    SQP  learned 2, ties 2, joint space 0

  Table 2 -- optimal cost, cells BOTH arms solved, learned-only regularizers excluded
  lower is better; N/A means fewer than 10 shared solved cells, so no comparison exists
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  gvs_pushrod9_o1 grasp contained native        0.682     *0.259**          N/A          N/A        0.261      *0.259*
  gvs_pushrod9_o1 grasp contained paired        0.596     *0.259**          N/A          N/A        0.313      *0.259*
  gvs_pushrod9_o1 pose contained native        1.057      *0.707*          N/A          N/A     *0.706**        0.720
  gvs_pushrod9_o1 pose contained paired        0.836      *0.733*          N/A          N/A        0.734     *0.684**

  Table 3 -- mean runtime, s, over ALL cells, each clamped at the 180 s clock
  lower is better; this machine only, never compared across machines. Clamped because SNOPT overruns its clock on cycling cells (none feasible) once the iteration budget is lifted
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  gvs_pushrod9_o1 grasp contained native      *4.86**         5.00          N/A          N/A      *14.32*        23.15
  gvs_pushrod9_o1 grasp contained paired        16.88      *4.75**          N/A          N/A        55.06      *22.84*
  gvs_pushrod9_o1 pose contained native      *1.24**         1.57          N/A          N/A         3.80       *2.35*
  gvs_pushrod9_o1 pose contained paired        20.84      *1.51**          N/A          N/A        68.80       *2.29*

  Table 4 -- median major iterations over solved cells
  lower is better; AL is N/A BY CONSTRUCTION -- NloptSolverDetails carries a single status and NLopt has no major iteration to count
  experiment                                   IP L        IP JS         AL L        AL JS        SQP L       SQP JS
  gvs_pushrod9_o1 grasp contained native        *36**           51          N/A          N/A          237        *102*
  gvs_pushrod9_o1 grasp contained paired           88        *51**          N/A          N/A          613        *102*
  gvs_pushrod9_o1 pose contained native        *18**           29          N/A          N/A           67         *26*
  gvs_pushrod9_o1 pose contained paired           56         *29*          N/A          N/A          644        *26**

=== PER-SOLVER DETAIL -- GVS arm, measured under identical conditions, outside the record

=== IPOPT (interior point)   [acceptable-point early stop (acceptable_tol 1e-3, acceptable_iter 1)]

  GVS arm, measured under identical conditions, outside the record
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  gvs_pushrod9_o1 grasp contained native    480  467   13    0 0.000244      learned       36     51    2.82   2.03   0.682   0.259  467    0    1
  gvs_pushrod9_o1 grasp contained paired    470  467   11    8    0.648          tie       88     51    7.36   1.92   0.596   0.259  459    9    1
  gvs_pushrod9_o1 pose contained (tip) native  480  311  169    0 2.67e-51      learned       18     29    0.77   0.92   1.057   0.707  311    0    0
  gvs_pushrod9_o1 pose contained (tip) paired  383  311  122   50 3.95e-08      learned       56     29    3.80   0.88   0.836   0.733  261   31    0

=== SNOPT (SQP)   [Major step limit = 0.5]

  GVS arm, measured under identical conditions, outside the record
  row                                         L   JS   L+  JS+        p      verdict  L iters  JS it     L s   JS s   Lcost  JScost    n  LTO  JTO
  gvs_pushrod9_o1 grasp contained native    476  366  114    4 4.78e-29      learned      237    102    8.94   3.05   0.261   0.259  362    1   49
  gvs_pushrod9_o1 grasp contained paired    343  366   76   99    0.096          tie      613    102   35.93   2.85   0.313   0.259  267   67   49
  gvs_pushrod9_o1 pose contained (tip) native  480  252  228    0 4.64e-69      learned       67     26    1.79   0.54   0.706   0.720  252    0    4
  gvs_pushrod9_o1 pose contained (tip) paired  275  252  109   86    0.115          tie      644     26   47.94   0.52   0.734   0.684  166  130    4
