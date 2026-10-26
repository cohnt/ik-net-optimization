# The analytic arm: tables (stages ANALYTIC*, collected 2026-10-14)

Generated output, regenerate rather than edit: `python scripts/report_analytic.py` (analytic beside the
record, pairwise) and `python scripts/report_threeway.py` (learned / joint space / analytic, three-way,
single start and multi-start). Raw runs: `results/_cluster_staging/20261014-015030` and `-105935`
(local, gitignored). Numbers only; the reading is Thomas's.

SNOPT runs carry stage OVR's re-run shards (2026-10-22; overrun watcher, `cluster/README.md`;
superseded merges in `results/archive_preoverrun/`). No verdict moved. iiwa posetip native SNOPT
analytic (59,0), feasible 643-1443 s past the 180 s clock originally, is now `killed_overrun` on
all three ladders (A 435 -> 434, ANALYTICC 434 -> 433, ANALYTICT 436 -> 435; accepted, Thomas
2026-10-21); analytic iteration-budget stops are 0 on every SNOPT row. The svgd rows' L and JS
columns are the AL-SVGD record (R29 / R30), regenerated 2026-10-22.

## report_analytic.py

```
96 analytic runs, 64 record runs (IPOPT / SNOPT at paper conditions, NLopt at PROCS=8 MPS=1)
Verdicts are exact McNemar on the paired cells, naming the side that solved more where
p < 0.05 ('*'). Cost is the median on cells both arms solved (N/A under 10 shared cells,
count in brackets). ms/it is per IPOPT/SNOPT iteration, per map evaluation under NLopt.
A row with >= 24 analytic cells at an iteration budget carries no verdict.

=== clip 1e-4 / reach 1e-3 (../codebase) ===
robot row      start  solver arm               L   JS    A        L v JS           A v L          A v JS       it L/JS/A        ms/it L/JS/A        cost A:L       cost A:JS         wall L/JS/A clk A bud A unrep band
iiwa  mugshelf native ipopt  analytic/cpp    480  453  455 27/0       L*    0/25      L*    26/24    tie  134  118    62   10.3   2.5    3.6     6.22:  7.95     6.22:  2.83   2.68  1.02   3.80     2     0     0    0
iiwa  mugshelf paired ipopt  analytic/cpp    479  453  440 27/1       L*    1/40      L*    26/39    tie  246  118   196   10.3   2.5    4.2     6.34:  7.42     6.25:  2.84   6.27  1.01   4.53     1     0     0   18
iiwa  posetip  native ipopt  analytic/cpp    462  312  441160/10      L*   17/38      L*  157/28      A*   50   28     7    8.0   2.2    3.1     9.24:  9.05     9.00:  9.16   0.67  0.09   0.05     0     0     0    0
iiwa  posetip  paired ipopt  analytic/cpp    406  312  369138/44      L*   44/81      L*  120/63      A*  106   28    27    8.5   2.2    3.0     9.46:  9.14     9.26:  9.11   1.61  0.09   0.11     0     0     0   21
iiwa  mugshelf native snopt  analytic/cpp    436  348  370119/31      L*   35/101     L*   104/82    tie  264  128    32    7.3   2.5    3.0     6.53:  7.51     6.36:  6.69   3.14 14.55  19.05    40     0     0    0
iiwa  mugshelf paired snopt  analytic/cpp    276  348  243 71/143    JS*   89/122     L*   65/170    JS*  599  128    89    8.8   2.6    4.2     6.42:  7.63     6.64:  6.19  14.84 14.55  22.74    44     0     0   18
iiwa  posetip  native snopt  analytic/cpp    442  272  433188/18      L*    29/38    tie  187/26      A*   78   21     6    6.4   1.7    2.4     9.35:  9.02     9.38:  9.34   1.28  1.58   0.70     1     0     0    0
iiwa  posetip  paired snopt  analytic/cpp    233  272  447101/140    JS*  228/14      A*  190/15      A*  331   21     6    8.8   1.7    2.4     9.49:  8.72     9.69:  9.40  13.14  1.58   0.50     1     0     0   21
panda mugshelf native ipopt  analytic/cpp    480  448  441 32/0       L*    0/39      L*    26/33    tie  134  205   128   13.6   3.5    4.5     3.82:  4.51     3.73:  2.23   3.00  6.07   5.25     2     0     0    0
panda mugshelf native ipopt  analytic8/cpp   480  448  257 32/0       L*    0/223     L*   15/206    JS*  134  205   151   13.6   3.5    4.7     3.97:  4.42     3.91:  2.29   3.00  6.07   7.73     5     0     0    0
panda mugshelf paired ipopt  analytic/cpp    479  449  383 30/0       L*    0/96      L*   18/84     JS*  199  206   191   14.0   3.5    5.2     3.50:  4.17     3.48:  2.27   5.01  6.06   5.19     2     0    67   15
panda mugshelf paired ipopt  analytic8/cpp   479  449  393 30/0       L*    0/86      L*   18/74     JS*  199  206   194   14.0   3.5    5.2     3.54:  4.15     3.53:  2.23   5.01  6.06   7.13     4     0     0   28
panda posetip  native ipopt  analytic/cpp    458  200  298269/11      L*   11/171     L*  162/64      A*   37   33     8    9.8   2.7    3.8     7.32:  6.76     6.90:  7.72   0.77  0.53   6.41    11     0     0    0
panda posetip  native ipopt  analytic8/cpp   458  200  150269/11      L*    4/312     L*   87/137    JS*   37   33     8    9.8   2.7    3.7     7.37:  7.03     6.76:  8.08   0.77  0.53   6.15     8     0     0    0
panda posetip  paired ipopt  analytic/cpp    395  200  233225/30      L*   32/194     L*  112/79      A*  107   33    29   12.0   2.7    3.9     7.02:  6.08     7.22:  7.78   2.02  0.53   3.38     6     0    68   16
panda posetip  paired ipopt  analytic8/cpp   395  200  233225/30      L*   32/194     L*  112/79      A*  107   33    29   12.0   2.7    3.9     7.02:  6.08     7.22:  7.78   2.02  0.53   3.55     6     0     0   33
panda mugshelf native snopt  analytic/cpp    461  361  363114/14      L*   17/115     L*    86/84    tie  191  129    65    9.2   3.0    4.5     4.13:  3.73     4.04:  3.60   3.54 12.96  12.02    19     0     0    0
panda mugshelf native snopt  analytic8/cpp   461  361  198114/14      L*   13/276     L*   47/210    JS*  191  129    68    9.2   3.0    4.8     4.12:  3.75     4.04:  3.70   3.54 12.96  11.32    19     1     0    0
panda mugshelf paired snopt  analytic/cpp    300  361  222 64/125    JS*   74/152     L*   44/183    JS*  426  129   122   11.7   3.0    5.9     4.30:  4.22     4.53:  3.88  12.13 12.92  15.44    30     0    67   15
panda mugshelf paired snopt  analytic8/cpp   300  361  228 64/125    JS*   78/150     L*   49/182    JS*  426  129   122   11.7   3.0    5.9     4.33:  4.22     4.53:  3.88  12.13 12.92  17.11    33     0     0   28
panda posetip  native snopt  analytic/cpp    439  193  244259/13      L*   18/213     L*  132/81      A*   47   21     6    6.7   2.0    2.7     6.90:  6.76     6.58:  6.91   1.67  1.39   1.21     3     0     0    0
panda posetip  native snopt  analytic8/cpp   439  193  120259/13      L*    9/328     L*   70/143    JS*   47   21     5    6.7   2.0    2.7     6.69:  6.76     6.47:  7.94   1.67  1.39   1.53     4     0     0    0
panda posetip  paired snopt  analytic/cpp    282  193  212153/64      L*   67/137     L*   102/83    tie  247   21     6   11.2   2.0    3.0     6.60:  6.22     6.63:  6.91   8.30  1.41   0.82     2     0    68   16
panda posetip  paired snopt  analytic8/cpp   282  193  212153/64      L*   67/137     L*   102/83    tie  247   21     6   11.2   2.0    2.9     6.60:  6.22     6.63:  6.91   8.30  1.41   0.82     2     0     0   33
iiwa  mugshelf native nlopt  analytic        313    0  321313/0       L*    97/89    tie  321/0       A*   --   --    --    5.3    --   16.5     9.86:  9.52         N/A (0)  63.48180.00 154.83   412     0     0    0
iiwa  mugshelf paired nlopt  analytic         11    0   15 11/0       L*    13/9     tie   15/0       A*   --   --    --    4.6    --    3.4         N/A (2)         N/A (0) 175.89180.00 174.41   465     0     0   18
iiwa  posetip  native nlopt  analytic        299   41  449272/14      L*  166/16      A*  411/3       A*   --   --    --    5.8  14.7    3.3     8.33:  9.20     7.14:  5.31  78.74169.95  19.68    51     0     0    0
iiwa  posetip  paired nlopt  analytic        112   41  396100/29      L*  297/13      A*  362/7       A*   --   --    --    5.8  14.6    3.2     7.28:  8.78     9.36:  5.31 142.06169.95  32.17    84     0     0   21
panda mugshelf native nlopt  analytic        322    0  174322/0       L*   48/196     L*  174/0       A*   --   --    --    7.2    --    4.3     7.52:  7.27         N/A (0)  60.01180.00 146.09   389     0     0    0
panda mugshelf native nlopt  analytic8       322    0   86322/0       L*   23/259     L*   86/0       A*   --   --    --    7.2    --   12.8     7.20:  6.90         N/A (0)  60.01180.00 165.11   440     0     0    0
panda mugshelf paired nlopt  analytic          3    0    2   3/0     tie     2/3     tie     2/0     tie   --   --    --    6.2    --    3.6         N/A (0)         N/A (0) 178.88180.00 154.13   411     0    67   15
panda mugshelf paired nlopt  analytic8         3    0    2   3/0     tie     2/3     tie     2/0     tie   --   --    --    6.2    --    3.6         N/A (0)         N/A (0) 178.88180.00 179.25   478     0     0   28
panda posetip  native nlopt  analytic        304  141  302203/40      L*   110/112   tie  210/49      A*   --   --    --    8.3  18.3    3.8     5.98:  7.00     6.06:  4.50  77.68142.98  75.71   200     0     0    0
panda posetip  native nlopt  analytic8       304  141  151203/40      L*   49/202     L*   106/96    tie   --   --    --    8.3  18.3    3.8     5.88:  7.22     5.29:  4.44  77.68142.98 128.24   341     0     0    0
panda posetip  paired nlopt  analytic        130  141  273  83/94    tie  190/47      A*  187/55      A*   --   --    --    8.0  18.4    3.7     5.99:  5.79     6.46:  4.30 138.10142.98  53.69   140     0    68   16
panda posetip  paired nlopt  analytic8       130  141  273  83/94    tie  190/47      A*  187/55      A*   --   --    --    8.0  18.4    3.7     5.99:  5.79     6.46:  4.30 138.10142.98  79.19   208     0     0   33
iiwa  mugshelf native svgd/q analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   22    10   76.1  54.7   61.1     6.79:  3.51     6.79:  4.06   1.12  1.48   0.73     0     0     0    0
iiwa  mugshelf native svgd/x analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   22    11   76.1  54.7   60.2     6.95:  3.51     6.95:  4.06   1.12  1.48   0.75     0     0     0    0
iiwa  mugshelf paired svgd/q analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   22    10   76.7  53.4   62.9     6.29:  3.46     6.29:  4.06   0.99  1.45   0.74     0     0     0   18
iiwa  mugshelf paired svgd/x analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   22    11   76.7  53.4   61.8     6.92:  3.46     6.92:  4.06   0.99  1.45   0.78     0     0     0   18
iiwa  posetip  native svgd/q analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   77.5  51.9   61.5     7.99:  3.93     7.99:  3.44   0.72  0.90   0.43     0     0     0    0
iiwa  posetip  native svgd/x analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   77.5  51.9   60.5     7.97:  3.93     7.97:  3.44   0.72  0.90   0.45     0     0     0    0
iiwa  posetip  paired svgd/q analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   76.9  51.6   61.0     8.15:  3.88     8.15:  3.44   0.70  0.89   0.42     0     0     0   21
iiwa  posetip  paired svgd/x analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   76.9  51.6   60.8     8.15:  3.88     8.15:  3.44   0.70  0.89   0.45     0     0     0   21
panda mugshelf native svgd/q analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie   14   16    10   90.1  56.5   70.3     5.91:  3.41     5.91:  3.65   1.91  1.14   0.89     0     0     0    0
panda mugshelf native svgd/q analytic8       480  480  261   0/0     tie    0/219     L*    0/219    JS*   14   16    10   90.1  56.5   68.9     5.95:  3.37     5.95:  3.67   1.91  1.14  74.90   219     0     0    0
panda mugshelf native svgd/x analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie   14   16    13   90.1  56.5   69.2     6.37:  3.41     6.37:  3.65   1.91  1.14   1.35     0     0     0    0
panda mugshelf native svgd/x analytic8       480  480  257   0/0     tie    0/223     L*    0/223    JS*   14   16    13   90.1  56.5   68.7     6.28:  3.39     6.28:  3.69   1.91  1.14  76.17   223     0     0    0
panda mugshelf paired svgd/q analytic        480  480  413   0/0     tie    0/67      L*    0/67     JS*   15   16    10   90.1  56.7   69.9     5.37:  3.42     5.37:  3.65   2.00  1.14   0.72     0     0    67   15
panda mugshelf paired svgd/q analytic8       480  480  417   0/0     tie    0/63      L*    0/63     JS*   15   16    11   90.1  56.7   69.2     5.40:  3.40     5.40:  3.65   2.00  1.14  22.10    63     0     0   28
panda mugshelf paired svgd/x analytic        480  480  413   0/0     tie    0/67      L*    0/67     JS*   15   16    13   90.1  56.7   69.1     6.40:  3.42     6.40:  3.65   2.00  1.14   1.08     0     0    67   15
panda mugshelf paired svgd/x analytic8       480  480  417   0/0     tie    0/63      L*    0/63     JS*   15   16    13   90.1  56.7   68.5     6.40:  3.40     6.40:  3.65   2.00  1.14  22.50    63     0     0   28
panda posetip  native svgd/q analytic        480  480  450   0/0     tie    0/30      L*    0/30     JS*    7   11     7   96.1  57.6   63.3     6.27:  4.53     6.27:  4.44   0.77  0.71  10.58    30     0     0    0
panda posetip  native svgd/q analytic8       480  480  234   0/0     tie    0/246     L*    0/246    JS*    7   11     7   96.1  57.6   62.2     6.19:  4.54     6.19:  4.47   0.77  0.71  83.35   246     0     0    0
panda posetip  native svgd/x analytic        480  480  447   0/0     tie    0/33      L*    0/33     JS*    7   11     7   96.1  57.6   62.6     6.21:  4.53     6.21:  4.44   0.77  0.71  11.63    33     0     0    0
panda posetip  native svgd/x analytic8       480  480  233   0/0     tie    0/247     L*    0/247    JS*    7   11     7   96.1  57.6   61.9     6.15:  4.53     6.15:  4.47   0.77  0.71  83.71   247     0     0    0
panda posetip  paired svgd/q analytic        480  480  390   0/0     tie    0/90      L*    0/90     JS*    7   11     7   96.7  57.4   63.4     6.21:  4.50     6.21:  4.42   0.76  0.70   7.84    22     0    68   16
panda posetip  paired svgd/q analytic8       480  480  390   0/0     tie    0/90      L*    0/90     JS*    7   11     7   96.7  57.4   62.8     6.21:  4.50     6.21:  4.42   0.76  0.70  30.81    90     0     0   33
panda posetip  paired svgd/x analytic        480  480  389   0/0     tie    0/91      L*    0/91     JS*    7   11     7   96.7  57.4   63.1     6.23:  4.50     6.23:  4.42   0.76  0.70   8.18    23     0    68   16
panda posetip  paired svgd/x analytic8       480  480  389   0/0     tie    0/91      L*    0/91     JS*    7   11     7   96.7  57.4   62.5     6.23:  4.50     6.23:  4.42   0.76  0.70  31.15    91     0     0   33
iiwa  mugshelf native ipopt  analytic        480  453  444 27/0       L*    0/36      L*    23/32    tie  134  118    54   10.3   2.5    6.3     6.48:  7.94     6.28:  2.91   2.68  1.02   5.02     1     0     0    0
iiwa  mugshelf paired ipopt  analytic        479  453  439 27/1       L*    1/41      L*    24/38    tie  246  118   189   10.3   2.5    8.8     5.71:  7.52     5.71:  2.85   6.27  1.01   8.21     2     0     0   18
iiwa  posetip  native ipopt  analytic        462  312  442160/10      L*   17/37      L*  157/27      A*   50   28     7    8.0   2.2    5.0     9.24:  9.04     8.99:  9.15   0.67  0.09   0.09     0     0     0    0
iiwa  posetip  paired ipopt  analytic        406  312  376138/44      L*   43/73      L*  127/63      A*  106   28    27    8.5   2.2    5.3     9.44:  9.10     9.26:  9.11   1.61  0.09   0.22     0     0     0   21
iiwa  mugshelf native snopt  analytic        436  348  374119/31      L*   34/96      L*   105/79    tie  264  128    32    7.3   2.5    5.7     6.92:  7.53     6.52:  6.76   3.14 14.55  21.04    46     0     0    0
iiwa  mugshelf paired snopt  analytic        276  348  250 71/143    JS*    95/121   tie   66/164    JS*  599  128    82    8.8   2.6    7.9     6.52:  7.63     6.68:  6.03  14.84 14.55  32.41    64     0     0   18
iiwa  posetip  native snopt  analytic        442  272  434188/18      L*    29/37    tie  187/25      A*   78   21     6    6.4   1.7    4.2     9.35:  9.02     9.40:  9.33   1.28  1.58   1.01     1     0     0    0
iiwa  posetip  paired snopt  analytic        233  272  448101/140    JS*  229/14      A*  190/14      A*  331   21     6    8.8   1.7    4.3     9.49:  8.72     9.71:  9.39  13.14  1.58   0.80     2     0     0   21
panda mugshelf native ipopt  analytic        480  448  440 32/0       L*    0/40      L*    29/37    tie  134  205   106   13.6   3.5    8.0     3.62:  4.51     3.54:  2.16   3.00  6.07   8.54     3     0     0    0
panda mugshelf native ipopt  analytic8       480  448  251 32/0       L*    0/229     L*   15/212    JS*  134  205   135   13.6   3.5    8.3     3.88:  4.39     3.84:  2.03   3.00  6.07  12.02     7     0     0    0
panda mugshelf paired ipopt  analytic        479  449  380 30/0       L*    0/99      L*   22/91     JS*  199  206   194   14.0   3.5    9.0     3.54:  4.12     3.54:  2.36   5.01  6.06   7.72     3     0    67   15
panda mugshelf paired ipopt  analytic8       479  449  396 30/0       L*    0/83      L*   22/75     JS*  199  206   201   14.0   3.5    9.0     3.64:  4.09     3.62:  2.36   5.01  6.06   9.92     3     0     0   28
panda posetip  native ipopt  analytic        458  200  297269/11      L*   11/172     L*  159/62      A*   37   33     8    9.8   2.7    5.7     7.19:  6.77     6.90:  7.72   0.77  0.53   7.87    12     0     0    0
panda posetip  native ipopt  analytic8       458  200  147269/11      L*    4/315     L*   84/137    JS*   37   33     8    9.8   2.7    5.6     7.33:  7.03     6.76:  8.08   0.77  0.53   9.74    13     0     0    0
panda posetip  paired ipopt  analytic        395  200  236225/30      L*   33/192     L*  117/81      A*  107   33    28   12.0   2.7    6.0     7.02:  6.08     7.35:  7.78   2.02  0.53   4.13     8     0    68   16
panda posetip  paired ipopt  analytic8       395  200  236225/30      L*   33/192     L*  117/81      A*  107   33    28   12.0   2.7    6.0     7.02:  6.08     7.35:  7.78   2.02  0.53   4.39     8     0     0   33
panda mugshelf native snopt  analytic        461  361  369114/14      L*   16/108     L*    88/80    tie  191  129    69    9.2   3.0    8.1     4.12:  3.77     4.02:  3.70   3.54 12.96  19.12    38     0     0    0
panda mugshelf native snopt  analytic8       461  361  200114/14      L*   11/272     L*   48/209    JS*  191  129    69    9.2   3.0    8.3     4.17:  3.88     4.06:  3.72   3.54 12.96  14.19    28     0     0    0
panda mugshelf paired snopt  analytic        300  361  213 64/125    JS*   70/157     L*   50/198    JS*  426  129   114   11.7   3.0    8.7     4.34:  4.08     4.32:  3.88  12.13 12.92  13.93    25     0    67   15
panda mugshelf paired snopt  analytic8       300  361  219 64/125    JS*   73/154     L*   54/196    JS*  426  129   115   11.7   3.0    8.7     4.39:  4.08     4.46:  3.88  12.13 12.92  16.15    30     0     0   28
panda posetip  native snopt  analytic        439  193  244259/13      L*   18/213     L*  132/81      A*   47   21     6    6.7   2.0    4.4     6.90:  6.76     6.58:  6.91   1.67  1.39   1.33     3     0     0    0
panda posetip  native snopt  analytic8       439  193  120259/13      L*    9/328     L*   70/143    JS*   47   21     5    6.7   2.0    4.5     6.69:  6.76     6.47:  7.94   1.67  1.39   1.57     4     0     0    0
panda posetip  paired snopt  analytic        282  193  212153/64      L*   67/137     L*   102/83    tie  247   21     6   11.2   2.0    4.4     6.60:  6.22     6.63:  6.91   8.30  1.41   0.97     2     0    68   16
panda posetip  paired snopt  analytic8       282  193  212153/64      L*   67/137     L*   102/83    tie  247   21     6   11.2   2.0    4.5     6.60:  6.22     6.63:  6.91   8.30  1.41   0.98     2     0     0   33
  NOTE  analytic runs on node class cpu, the record on gpu: the wall and ms/it columns cross machines and are not compared across them
  cells the analytic arm passes only at the relaxed gate (task_tol): 376

=== clip 1e-8 / reach 1.1e-4 (tight) ===
robot row      start  solver arm               L   JS    A        L v JS           A v L          A v JS       it L/JS/A        ms/it L/JS/A        cost A:L       cost A:JS         wall L/JS/A clk A bud A unrep band
iiwa  mugshelf native nlopt  analytic        313    0  321313/0       L*    97/89    tie  321/0       A*   --   --    --    5.3    --   17.6     9.76:  9.63         N/A (0)  63.48180.00 156.38   416     0     0    0
iiwa  mugshelf paired nlopt  analytic         11    0   13 11/0       L*    11/9     tie   13/0       A*   --   --    --    4.6    --    3.0         N/A (2)         N/A (0) 175.89180.00 175.15   467     0     0    0
iiwa  posetip  native nlopt  analytic        299   41  453272/14      L*  168/14      A*  415/3       A*   --   --    --    5.8  14.7    2.9     8.31:  9.28     7.14:  5.31  78.74169.95  18.89    49     0     0    0
iiwa  posetip  paired nlopt  analytic        112   41  399100/29      L*  304/17      A*  364/6       A*   --   --    --    5.8  14.6    3.2     7.82:  8.78     9.02:  5.46 142.06169.95  31.10    81     0     0    0
panda mugshelf native nlopt  analytic        322    0  180322/0       L*   49/191     L*  180/0       A*   --   --    --    7.2    --    4.2     7.16:  7.13         N/A (0)  60.01180.00 143.16   381     0     0    0
panda mugshelf native nlopt  analytic8       322    0   93322/0       L*   25/254     L*   93/0       A*   --   --    --    7.2    --    4.2     7.15:  6.91         N/A (0)  60.01180.00 161.37   430     0     0    0
panda mugshelf paired nlopt  analytic          3    0    2   3/0     tie     2/3     tie     2/0     tie   --   --    --    6.2    --    3.7         N/A (0)         N/A (0) 178.88180.00 154.13   411     0    67    0
panda mugshelf paired nlopt  analytic8         3    0    2   3/0     tie     2/3     tie     2/0     tie   --   --    --    6.2    --    3.7         N/A (0)         N/A (0) 178.88180.00 179.25   478     0     0    0
panda posetip  native nlopt  analytic        304  141  307203/40      L*   109/106   tie  213/47      A*   --   --    --    8.3  18.3    3.7     6.13:  7.02     6.38:  4.50  77.68142.98  73.92   195     0     0    0
panda posetip  native nlopt  analytic8       304  141  155203/40      L*   52/201     L*   109/95    tie   --   --    --    8.3  18.3    3.8     5.91:  7.30     5.49:  4.42  77.68142.98 127.60   339     0     0    0
panda posetip  paired nlopt  analytic        130  141  270  83/94    tie  191/51      A*  183/54      A*   --   --    --    8.0  18.4    3.7     5.84:  5.79     6.23:  4.38 138.10142.98  54.27   142     0    68    0
panda posetip  paired nlopt  analytic8       130  141  270  83/94    tie  191/51      A*  183/54      A*   --   --    --    8.0  18.4    3.7     5.84:  5.79     6.23:  4.38 138.10142.98  79.77   210     0     0    0
iiwa  mugshelf native svgd/q analytic        480  480  478   0/0     tie     0/2     tie     0/2     tie    8   22     9   76.1  54.7   64.3     7.05:  3.51     7.05:  4.06   1.12  1.48   1.35     2     0     0    0
iiwa  mugshelf native svgd/x analytic        480  480  472   0/0     tie    0/8       L*    0/8      JS*    8   22    10   76.1  54.7   64.8     7.28:  3.51     7.28:  4.07   1.12  1.48   3.42     8     0     0    0
iiwa  mugshelf paired svgd/q analytic        480  480  472   0/0     tie    0/8       L*    0/8      JS*    8   22     9   76.7  53.4   66.0     6.97:  3.46     6.97:  4.06   0.99  1.45   3.39     8     0     0    0
iiwa  mugshelf paired svgd/x analytic        480  480  476   0/0     tie     0/4     tie     0/4     tie    8   22    10   76.7  53.4   64.0     7.05:  3.46     7.05:  4.07   0.99  1.45   2.04     4     0     0    0
iiwa  posetip  native svgd/q analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   77.5  51.9   59.9     7.95:  3.93     7.95:  3.44   0.72  0.90   0.42     0     0     0    0
iiwa  posetip  native svgd/x analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   77.5  51.9   60.2     7.97:  3.93     7.97:  3.44   0.72  0.90   0.47     0     0     0    0
iiwa  posetip  paired svgd/q analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   76.9  51.6   61.6     8.15:  3.88     8.15:  3.44   0.70  0.89   0.43     0     0     0    0
iiwa  posetip  paired svgd/x analytic        480  480  480   0/0     tie     0/0     tie     0/0     tie    8   13     6   76.9  51.6   59.7     8.15:  3.88     8.15:  3.44   0.70  0.89   0.45     0     0     0    0
panda mugshelf native svgd/q analytic        480  480  465   0/0     tie    0/15      L*    0/15     JS*   14   16    10   90.1  56.5   71.1     6.16:  3.42     6.16:  3.64   1.91  1.14   5.81    15     0     0    0
panda mugshelf native svgd/q analytic8       480  480  248   0/0     tie    0/232     L*    0/232    JS*   14   16    10   90.1  56.5   70.6     6.28:  3.37     6.28:  3.68   1.91  1.14  78.86   232     0     0    0
panda mugshelf native svgd/x analytic        480  480  457   0/0     tie    0/23      L*    0/23     JS*   14   16    13   90.1  56.5   70.2     6.49:  3.44     6.49:  3.64   1.91  1.14   8.77    23     0     0    0
panda mugshelf native svgd/x analytic8       480  480  245   0/0     tie    0/235     L*    0/235    JS*   14   16    12   90.1  56.5   68.9     6.43:  3.37     6.43:  3.72   1.91  1.14  79.97   235     0     0    0
panda mugshelf paired svgd/q analytic        480  480  409   0/0     tie    0/71      L*    0/71     JS*   15   16    10   90.1  56.7   70.4     5.81:  3.42     5.81:  3.65   2.00  1.14   2.04     4     0    67    0
panda mugshelf paired svgd/q analytic8       480  480  410   0/0     tie    0/70      L*    0/70     JS*   15   16    10   90.1  56.7   69.7     5.81:  3.42     5.81:  3.65   2.00  1.14  24.33    70     0     0    0
panda mugshelf paired svgd/x analytic        480  480  390   0/0     tie    0/90      L*    0/90     JS*   15   16    12   90.1  56.7   70.3     6.48:  3.42     6.48:  3.65   2.00  1.14   8.62    23     0    67    0
panda mugshelf paired svgd/x analytic8       480  480  393   0/0     tie    0/87      L*    0/87     JS*   15   16    12   90.1  56.7   70.0     6.54:  3.42     6.54:  3.65   2.00  1.14  30.28    87     0     0    0
panda posetip  native svgd/q analytic        480  480  448   0/0     tie    0/32      L*    0/32     JS*    7   11     7   96.1  57.6   63.0     6.23:  4.54     6.23:  4.45   0.77  0.71  11.27    32     0     0    0
panda posetip  native svgd/q analytic8       480  480  233   0/0     tie    0/247     L*    0/247    JS*    7   11     7   96.1  57.6   62.3     6.21:  4.54     6.21:  4.47   0.77  0.71  83.70   247     0     0    0
panda posetip  native svgd/x analytic        480  480  441   0/0     tie    0/39      L*    0/39     JS*    7   11     7   96.1  57.6   62.8     6.21:  4.53     6.21:  4.44   0.77  0.71  13.65    39     0     0    0
panda posetip  native svgd/x analytic8       480  480  232   0/0     tie    0/248     L*    0/248    JS*    7   11     7   96.1  57.6   62.6     6.19:  4.54     6.19:  4.47   0.77  0.71  84.05   248     0     0    0
panda posetip  paired svgd/q analytic        480  480  389   0/0     tie    0/91      L*    0/91     JS*    7   11     7   96.7  57.4   63.8     6.51:  4.50     6.51:  4.42   0.76  0.70   8.17    23     0    68    0
panda posetip  paired svgd/q analytic8       480  480  389   0/0     tie    0/91      L*    0/91     JS*    7   11     7   96.7  57.4   63.1     6.51:  4.50     6.51:  4.42   0.76  0.70  31.14    91     0     0    0
panda posetip  paired svgd/x analytic        480  480  384   0/0     tie    0/96      L*    0/96     JS*    7   11     7   96.7  57.4   63.5     6.41:  4.50     6.41:  4.42   0.76  0.70   9.88    28     0    68    0
panda posetip  paired svgd/x analytic8       480  480  384   0/0     tie    0/96      L*    0/96     JS*    7   11     7   96.7  57.4   62.7     6.41:  4.50     6.41:  4.42   0.76  0.70  32.85    96     0     0    0
iiwa  mugshelf native ipopt  analytic        480  453  446 27/0       L*    0/34      L*    25/32    tie  134  118    68   10.3   2.5    6.7     6.59:  7.96     6.58:  2.84   2.68  1.02   6.98     2     0     0    0
iiwa  mugshelf paired ipopt  analytic        479  453  445 27/1       L*    1/35      L*    23/31    tie  246  118   163   10.3   2.5    8.3     6.24:  7.59     6.15:  2.84   6.27  1.01   9.07     2     0     0    0
iiwa  posetip  native ipopt  analytic        462  312  439160/10      L*   17/40      L*  156/29      A*   50   28     7    8.0   2.2    5.1     9.21:  9.04     8.99:  9.15   0.67  0.09   0.08     0     0     0    0
iiwa  posetip  paired ipopt  analytic        406  312  363138/44      L*   56/99      L*  115/64      A*  106   28    27    8.5   2.2    5.3     9.32:  9.10     9.26:  9.11   1.61  0.09   0.22     0     0     0    0
iiwa  mugshelf native snopt  analytic        436  348  373119/31      L*   35/98      L*   103/78    tie  264  128    37    7.3   2.5    6.1     6.53:  7.45     6.53:  6.69   3.14 14.55  21.89    43     0     0    0
iiwa  mugshelf paired snopt  analytic        276  348  225 71/143    JS*   84/135     L*   64/187    JS*  599  128    74    8.8   2.6    7.7     6.67:  7.62     6.09:  6.28  14.84 14.55  26.87    51     0     0    0
iiwa  posetip  native snopt  analytic        442  272  435188/18      L*    30/37    tie  187/24      A*   78   21     6    6.4   1.7    4.3     9.35:  9.02     9.38:  9.34   1.28  1.58   1.01     1     0     0    0
iiwa  posetip  paired snopt  analytic        233  272  448101/140    JS*  229/14      A*  190/14      A*  331   21     6    8.8   1.7    4.3     9.46:  8.72     9.71:  9.39  13.14  1.58   0.80     2     0     0    0
panda mugshelf native ipopt  analytic        480  448  450 32/0       L*    0/30      L*    31/29    tie  134  205   175   13.6   3.5    9.1     3.59:  4.51     3.47:  2.18   3.00  6.07   9.06     4     0     0    0
panda mugshelf native ipopt  analytic8       480  448  255 32/0       L*    0/225     L*   17/210    JS*  134  205   201   13.6   3.5    9.2     3.65:  4.42     3.59:  2.03   3.00  6.07  13.17     7     0     0    0
panda mugshelf paired ipopt  analytic        479  449  378 30/0       L*    0/101     L*   19/90     JS*  199  206   262   14.0   3.5    9.8     3.46:  4.14     3.45:  2.16   5.01  6.06   7.74     1     0    67    0
panda mugshelf paired ipopt  analytic8       479  449  389 30/0       L*    0/90      L*   20/80     JS*  199  206   264   14.0   3.5    9.8     3.53:  4.13     3.47:  2.20   5.01  6.06   9.69     1     0     0    0
panda posetip  native ipopt  analytic        458  200  311269/11      L*   12/159     L*  164/53      A*   37   33     8    9.8   2.7    5.7     7.16:  6.87     7.07:  7.78   0.77  0.53   7.03    10     0     0    0
panda posetip  native ipopt  analytic8       458  200  156269/11      L*    6/308     L*   88/132    JS*   37   33     8    9.8   2.7    5.7     7.21:  7.06     6.76:  8.07   0.77  0.53   7.07     8     0     0    0
panda posetip  paired ipopt  analytic        395  200  254225/30      L*   40/181     L*  122/68      A*  107   33    29   12.0   2.7    6.1     6.91:  6.20     7.12:  7.75   2.02  0.53   4.88    10     0    68    0
panda posetip  paired ipopt  analytic8       395  200  254225/30      L*   40/181     L*  122/68      A*  107   33    29   12.0   2.7    6.1     6.91:  6.20     7.12:  7.75   2.02  0.53   5.35    10     0     0    0
panda mugshelf native snopt  analytic        461  361  333114/14      L*   14/142     L*   78/106    JS*  191  129    75    9.2   3.0    8.1     4.19:  3.77     4.05:  3.64   3.54 12.96  16.20    32     0     0    0
panda mugshelf native snopt  analytic8       461  361  185114/14      L*   11/287     L*   43/219    JS*  191  129    81    9.2   3.0    8.6     4.16:  3.90     4.08:  3.69   3.54 12.96  12.57    23     0     0    0
panda mugshelf paired snopt  analytic        300  361  210 64/125    JS*   74/164     L*   50/201    JS*  426  129   128   11.7   3.0    8.8     4.16:  4.27     4.37:  3.48  12.13 12.92  13.46    22     0    67    0
panda mugshelf paired snopt  analytic8       300  361  212 64/125    JS*   74/162     L*   50/199    JS*  426  129   128   11.7   3.0    8.8     4.19:  4.27     4.42:  3.41  12.13 12.92  14.78    24     0     0    0
panda posetip  native snopt  analytic        439  193  241259/13      L*   19/217     L*  131/83      A*   47   21     6    6.7   2.0    4.3     7.04:  6.90     6.70:  6.91   1.67  1.39   2.68     7     0     0    0
panda posetip  native snopt  analytic8       439  193  119259/13      L*    9/329     L*   68/142    JS*   47   21     5    6.7   2.0    4.4     6.71:  6.89     6.46:  7.94   1.67  1.39   2.69     7     0     0    0
panda posetip  paired snopt  analytic        282  193  213153/64      L*   68/137     L*   103/83    tie  247   21     6   11.2   2.0    4.5     6.64:  6.22     6.58:  6.91   8.30  1.41   1.55     4     0    68    0
panda posetip  paired snopt  analytic8       282  193  213153/64      L*   68/137     L*   103/83    tie  247   21     6   11.2   2.0    4.4     6.64:  6.22     6.58:  6.91   8.30  1.41   1.93     5     0     0    0
  NOTE  analytic runs on node class cpu, the record on gpu: the wall and ms/it columns cross machines and are not compared across them
  cells the analytic arm passes only at the relaxed gate (task_tol): 360

```

## report_threeway.py

```
SINGLE START: success /480 | mean wall s (clamped 180) | median cost on cells all three solved;
best = top arm(s) not significantly beaten (McNemar / Wilcoxon, p < 0.05)
ipopt iiwa grasp native      480 / 453 / 444 L             2.68 / 1.02 / 5.02 JS        7.93 / 2.91 / 6.28 [421] JS
ipopt iiwa grasp paired      479 / 453 / 439 L             6.27 / 1.01 / 8.21 JS        7.42 / 2.88 / 5.68 [414] JS
ipopt panda grasp native     480 / 448 / 440 L             3.00 / 6.07 / 8.54 L=A       4.43 / 2.16 / 3.54 [411] JS
ipopt panda grasp paired     479 / 449 / 380 L             5.01 / 6.06 / 7.72 L=A       4.09 / 2.36 / 3.54 [358] JS
ipopt iiwa pose native       462 / 312 / 442 L             0.67 / 0.09 / 0.09 JS        9.12 / 9.15 / 9.00 [275] all tie
ipopt iiwa pose paired       406 / 312 / 376 L             1.61 / 0.09 / 0.22 JS        9.14 / 9.26 / 9.36 [221] all tie
ipopt panda pose native      458 / 200 / 297 L             0.77 / 0.53 / 7.87 JS=A      6.76 / 7.72 / 6.87 [131] all tie
ipopt panda pose paired      395 / 200 / 236 L             2.02 / 0.53 / 4.13 JS        6.08 / 7.72 / 7.19 [107] L
snopt iiwa grasp native      436 / 348 / 374 L           3.14 / 14.55 / 21.04 L         7.51 / 6.89 / 6.64 [245] all tie
snopt iiwa grasp paired      276 / 348 / 250 JS         14.84 / 14.55 / 32.41 JS        7.39 / 5.66 / 6.60 [118] JS=A
snopt panda grasp native     461 / 361 / 369 L           3.54 / 12.96 / 19.12 L         3.69 / 3.64 / 3.99 [270] all tie
snopt panda grasp paired     300 / 361 / 213 JS         12.13 / 12.92 / 13.93 L         4.08 / 3.84 / 4.28 [112] JS=A
snopt iiwa pose native       442 / 272 / 434 L=A           1.28 / 1.58 / 1.01 A         9.16 / 9.29 / 9.44 [234] all tie
snopt iiwa pose paired       233 / 272 / 448 A            13.14 / 1.58 / 0.80 A         8.22 / 9.21 / 9.03 [124] all tie
snopt panda pose native      439 / 193 / 244 L             1.67 / 1.39 / 1.33 A         6.39 / 6.91 / 6.54 [107] all tie
snopt panda pose paired      282 / 193 / 212 L             8.30 / 1.41 / 0.97 A         6.41 / 7.03 / 6.16 [84] all tie
nlopt iiwa grasp native        313 / 0 / 321 A=L      63.48 / 180.00 / 154.83 L               -- / -- / -- [0] N/A
nlopt iiwa grasp paired          11 / 0 / 15 A=L     175.89 / 180.00 / 174.41 A=L             -- / -- / -- [0] N/A
nlopt panda grasp native       322 / 0 / 174 L        60.01 / 180.00 / 146.09 L               -- / -- / -- [0] N/A
nlopt panda grasp paired           3 / 0 / 2 all tie 178.88 / 180.00 / 154.13 A               -- / -- / -- [0] N/A
nlopt iiwa pose native        299 / 41 / 449 A         78.74 / 169.95 / 19.68 A        10.11 / 4.99 / 6.04 [25] JS=A
nlopt iiwa pose paired        112 / 41 / 396 A        142.06 / 169.95 / 32.17 A        10.60 / 5.48 / 7.20 [10] JS=A
nlopt panda pose native      304 / 141 / 302 L=A       77.68 / 142.98 / 75.71 A         6.62 / 4.47 / 5.98 [66] JS
nlopt panda pose paired      130 / 141 / 273 A        138.10 / 142.98 / 53.69 A         5.54 / 5.07 / 5.24 [31] JS=A
svgd  iiwa grasp native      480 / 480 / 480 all tie       1.12 / 1.48 / 0.73 A         3.51 / 4.06 / 6.79 [480] L
svgd  iiwa grasp paired      480 / 480 / 480 all tie       0.99 / 1.45 / 0.74 A         3.46 / 4.06 / 6.29 [480] L
svgd  panda grasp native     480 / 480 / 480 all tie       1.91 / 1.14 / 0.89 A         3.41 / 3.65 / 5.91 [480] L=JS
svgd  panda grasp paired     480 / 480 / 413 L=JS          2.00 / 1.14 / 0.72 A         3.42 / 3.65 / 5.37 [413] L=JS
svgd  iiwa pose native       480 / 480 / 480 all tie       0.72 / 0.90 / 0.43 A         3.93 / 3.44 / 7.99 [480] JS
svgd  iiwa pose paired       480 / 480 / 480 all tie       0.70 / 0.89 / 0.42 A         3.88 / 3.44 / 8.15 [480] JS
svgd  panda pose native      480 / 480 / 450 L=JS         0.77 / 0.71 / 10.58 JS        4.53 / 4.44 / 6.27 [450] JS
svgd  panda pose paired      480 / 480 / 390 L=JS          0.76 / 0.70 / 7.84 JS        4.50 / 4.42 / 6.21 [390] JS

MULTI-START over each target's 8 recorded starts: % targets solved within 10 s / 60 s / any;
median time to first success; best-of-8 cost on targets all three solved
ipopt iiwa grasp native         96 / 99 / 87  100 / 100 / 98 100 / 100 / 100      1.86 / 0.58 / 3.69    2.78 / 1.59 / 1.82 [60]
ipopt iiwa grasp paired         88 / 99 / 77   98 / 100 / 98 100 / 100 / 100      3.83 / 0.60 / 6.25    2.44 / 1.59 / 1.81 [60]
ipopt panda grasp native        96 / 92 / 82   100 / 98 / 96 100 / 100 / 100      2.63 / 1.98 / 4.45    2.80 / 1.80 / 2.10 [60]
ipopt panda grasp paired        90 / 92 / 79    99 / 98 / 96 100 / 100 / 100      3.82 / 2.00 / 7.15    2.08 / 1.80 / 2.09 [60]
ipopt iiwa pose native       100 / 100 / 100 100 / 100 / 100 100 / 100 / 100      0.65 / 0.11 / 0.05    4.12 / 5.33 / 4.96 [60]
ipopt iiwa pose paired        98 / 100 / 100 100 / 100 / 100 100 / 100 / 100      1.68 / 0.11 / 0.23    4.57 / 5.33 / 4.67 [60]
ipopt panda pose native        100 / 95 / 86   100 / 95 / 91  100 / 95 / 100      0.61 / 0.33 / 0.64    4.36 / 5.29 / 4.47 [57]
ipopt panda pose paired         96 / 95 / 91   100 / 95 / 93   100 / 95 / 97      2.18 / 0.33 / 0.67    4.32 / 5.26 / 5.15 [56]
snopt iiwa grasp native         94 / 84 / 80   100 / 90 / 87 100 / 100 / 100    3.08 / 12.55 / 25.01    2.44 / 2.24 / 2.09 [60]
snopt iiwa grasp paired         58 / 84 / 61    90 / 90 / 71 100 / 100 / 100   16.34 / 12.83 / 48.81    3.08 / 2.24 / 2.58 [60]
snopt panda grasp native        93 / 87 / 78   100 / 91 / 87 100 / 100 / 100     3.22 / 4.00 / 23.62    2.49 / 1.87 / 2.40 [60]
snopt panda grasp paired        60 / 87 / 68    94 / 91 / 84  100 / 100 / 98    13.78 / 3.87 / 12.37    2.22 / 1.89 / 2.78 [59]
snopt iiwa pose native          99 / 95 / 99   100 / 97 / 99  100 / 98 / 100      1.00 / 0.11 / 0.03    4.19 / 5.35 / 5.00 [59]
snopt iiwa pose paired          64 / 95 / 99    90 / 97 / 99   98 / 98 / 100     13.31 / 0.11 / 0.03    4.99 / 5.34 / 4.75 [58]
snopt panda pose native         97 / 93 / 97   100 / 93 / 97   100 / 95 / 98      1.08 / 0.16 / 0.06    4.39 / 5.35 / 4.90 [56]
snopt panda pose paired         74 / 93 / 94    95 / 93 / 94   100 / 95 / 95      8.06 / 0.16 / 0.05    4.48 / 5.24 / 5.23 [54]
nlopt iiwa grasp native          64 / 0 / 14     65 / 0 / 14   100 / 0 / 100     85.40 / -- / 202.14          -- / -- / -- [0]
nlopt iiwa grasp paired            2 / 0 / 3       2 / 0 / 3     15 / 0 / 20    640.01 / -- / 617.59          -- / -- / -- [0]
nlopt panda grasp native         66 / 0 / 19     67 / 0 / 19     97 / 0 / 93     55.68 / -- / 329.17          -- / -- / -- [0]
nlopt panda grasp paired           1 / 0 / 0       1 / 0 / 0       5 / 0 / 3    632.58 / -- / 540.43          -- / -- / -- [0]
nlopt iiwa pose native           45 / 3 / 89     58 / 6 / 89  100 / 43 / 100  103.68 / 648.88 / 0.88    4.99 / 5.31 / 3.91 [26]
nlopt iiwa pose paired           18 / 3 / 83     22 / 6 / 83    88 / 43 / 98 374.17 / 649.01 / 22.20    6.09 / 5.40 / 4.26 [25]
nlopt panda pose native          48 / 4 / 57    56 / 20 / 58   98 / 63 / 100  96.49 / 217.47 / 98.44    4.74 / 4.47 / 4.46 [38]
nlopt panda pose paired          17 / 4 / 63    23 / 20 / 64    87 / 63 / 93 364.31 / 217.65 / 54.19    5.02 / 4.37 / 4.31 [35]
svgd  iiwa grasp native       100 / 99 / 100 100 / 100 / 100 100 / 100 / 100      0.98 / 1.23 / 0.73    2.89 / 2.31 / 2.39 [60]
svgd  iiwa grasp paired       100 / 99 / 100 100 / 100 / 100 100 / 100 / 100      0.85 / 1.23 / 0.72    2.69 / 2.31 / 2.51 [60]
svgd  panda grasp native       99 / 99 / 100 100 / 100 / 100 100 / 100 / 100      1.65 / 0.98 / 0.83    2.94 / 2.35 / 2.96 [60]
svgd  panda grasp paired       99 / 99 / 100 100 / 100 / 100 100 / 100 / 100      1.70 / 0.99 / 0.81    2.94 / 2.35 / 2.95 [60]
svgd  iiwa pose native       100 / 100 / 100 100 / 100 / 100 100 / 100 / 100      0.69 / 0.85 / 0.42    3.31 / 3.21 / 4.18 [60]
svgd  iiwa pose paired       100 / 100 / 100 100 / 100 / 100 100 / 100 / 100      0.67 / 0.85 / 0.41    3.30 / 3.21 / 4.03 [60]
svgd  panda pose native       100 / 100 / 94  100 / 100 / 94 100 / 100 / 100      0.73 / 0.69 / 0.48    4.35 / 4.40 / 4.45 [60]
svgd  panda pose paired       100 / 100 / 95  100 / 100 / 95 100 / 100 / 100      0.74 / 0.69 / 0.51    4.29 / 4.40 / 4.50 [60]
```
