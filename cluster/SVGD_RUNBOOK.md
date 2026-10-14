# Stages SVGD_SMOKE, SVGD_R2, SVGD_R1 / SVGD_R1K: the svgd solver on the cluster

The operational half. The design is the SVGD block of `cluster/gen_manifest.py`; the method and
its per-row analysis rule are `docs/svgd-solver.md`. The cluster playbook is `cluster/README.md`.

## What it is

Panda `n6` only, at the record's Panda conditions: hardened scene, shelf-contained targets at the
fingertips, 480 cells = 60 x 8, seed 1, 180 s, `--compile --set flow_cuda_graph=True`,
`--set correction_cost_weight=10.0`, arms `learned,numerical`, both start protocols, grasp
(`mugshelf`) and pose (`posetip`). **Every manifest runs at PROCS=2 with no MPS** (one solve per
V100, the paper's condition).

| manifest | what | logical runs | items | est. node-h (PROCS=2) | h on 4 idle nodes |
| --- | --- | --- | --- | --- | --- |
| `manifest_stageSVGD_SMOKE.txt` | Panda posetip paired, svgd defaults (`kq`), 2 x 2 = 4 cells, 60 s | 1 | 1 | 0.1 | ~0.2 (one node) |
| `manifest_stageSVGD_R2.txt` | the IPOPT twin: stage REMEASURE's four Panda IPOPT rows verbatim, re-tagged, 8 shards | 4 | 32 | 2.6 | 0.6 |
| `manifest_stageSVGD_R1.txt` | svgd at its defaults, `kq` (kernel on q, N = 64), 24 shards | 4 | 96 | 23.7 (assumed) / 98 (worst) | 5.9 / 24.6 |
| `manifest_stageSVGD_R1K.txt` | the same with `--set svgd_kernel=none` (`knone`), 24 shards, **held** | 4 | 96 | 23.7 (assumed) / 98 (worst) | 5.9 / 24.6 |

R1 and R1K are one stage, split by variant so the weekend can be ordered. They share the
`sc_SVGD_R1_` tag family and one builder, and the selftest holds that together they are exactly
the stage's item set.

`python cluster/gen_manifest.py --stage SVGD_R1 --allotment` prints this arithmetic.

- **R2** is charged at the record's Panda IPOPT means (Table 3 of `docs/status-quo-tables.md`, stage
  REMEASURE at PROCS=8 under MPS, so slightly pessimistic at PROCS=2): grasp native 3.52 + 6.24 s,
  grasp paired 5.86 + 6.27 s, pose native 0.93 + 0.54 s, pose paired 2.37 + 0.55 s per cell. There
  is also +180 s per item.
- **R1 and R1K assume 40 s per cell per arm.** Both arms run svgd, so a cell costs 80 s. That is
  an **assumption**, not a measurement.
  - The worst case puts every cell at the 180 s clock on both arms. That is 98 node-hours per
    manifest, about 24.6 h on 4 idle nodes. Both manifests at the worst case (~49 h) **would run
    into the maintenance window**.
  - The latest local end-to-end run at the staged defaults points to the worst case.
    `e2e_delta_mug_1e-2` ran 6 cells at 20 s, and all 12 arm-cells stopped on `wall_clock`.
  - Within each manifest the claim order is grasp, then pose.
  - **Read the smoke's mean wall per cell against its 60 s clock** before trusting the 40 s line.
- Every R1 item fits inside run_items.sh's 8 h `ITEM_TIMEOUT` even with every cell at the clock
  (20 cells x 2 arms x 180 s = 2 h). A maintenance kill therefore loses at most 2 h per item.

**What the svgd items (SMOKE, R1, R1K) set beyond the defaults, and why** (the gen_manifest block says it in full):

- `svgd_compile=True svgd_cuda_graph=True`. This is the `graphed` mode of `scripts/svgd/smoke.py`:
  `WarmUpSvgdStep` compiles and captures before the first timed cell. It is an execution setting,
  not part of the method.
- `svgd_outer_iters=1000000`, in place of the record's `max_iter`, which would otherwise BE the
  svgd step cap. This is the cap rule applied up front: the default of 300 outer checks would stop
  cells before the clock (~0.3 s per check locally), and a row with >= 24 of 480 cells at a budget
  carries no verdict. To field the default cap instead, empty `SVGD_BUDGET` and regenerate.
- `svgd_rho=1000 svgd_gn_lm=10 svgd_lr=0.3`, pinned explicitly on every svgd item (SMOKE, R1,
  R1K). These are the values the local probe chose. The code staged on the cluster (42a5893)
  predates them: its `svgd_gn_lm` default is 1e-2. The tree cannot be restaged while PAPER runs, so
  the manifests carry them (`SVGD_PINNED_V1`; a stage from R20 on takes `SVGD_PINNED_V2`, the 2026-10-12 pin), and the selftest fails if any svgd item lacks them.
- **Nothing else.** `svgd_n`, `svgd_paired_init` and every other svgd option stay at their defaults.
  The selftest reads the relied-on defaults out of `src/generic_program.py`'s text, and fails if
  one moves. (2026-10-12: the defaults moved to the new pin -- kernel cz, rho 1e5, metric on, init
  native. The v1 stages' unnamed kernel q / metric off / init jitter are recorded in
  `SVGD_HISTORICAL_DEFAULTS`; their manifests are unchanged, and a stage from R20 on names every
  moved setting through `SVGD_PINNED_V2`.)

**Maintenance: compute is down Mon 2026-10-12 evening to Wed 10-14 morning.** Running jobs are
killed and queued jobs do not survive. Everything must finish before Monday evening. If the window
lands mid-stage, run `--reclaim` on each manifest and resubmit afterwards.

## Stage the code from the svgd tip

Commit `cluster/gen_manifest.py`, the five manifests (SVGD_SMOKE, SVGD_R2, SVGD_R1, SVGD_R1K, PAPER), `cluster/make_staging_clone.sh` and this
file on `svgd` first. `stage_code.sh` stamps `.staged-commit` from HEAD, while rsync pushes the
working tree.

The svgd worktree (`.claude/worktrees/particle-solvers`) cannot be staged from as it stands, for two
reasons. Its `models/*.pkl` are **symlinks**, and its `third_party/ikflow` submodule is
**uninitialised**. `stage_code.sh` refuses both (exits 5 and 4), because an rsync `--delete` from
it would damage the cluster tree. So stage from a throwaway clone pinned at the tip.
`cluster/make_staging_clone.sh` builds or refreshes that clone:

- It clones the repository locally and detaches at the `svgd` tip.
- It initialises `third_party/ikflow` from the main checkout's own module store, with no network.
  If that fails, it falls back to GitHub.
- It brings in the Panda `n6` checkpoint as a real file: a hardlink where both are on one
  filesystem, a verified copy on the tmpfs scratchpad. The `.arch.json` is tracked, so it comes
  with the clone and is only checked against the main checkout.
- It refuses if any symlink is left under `models/`.
- It warns if the worktree has uncommitted changes, because those are NOT in the clone.

```bash
MAIN=~/Documents/programming/work/rlg/analytic-and-optimization-ik/learned-ik
WT=$MAIN/.claude/worktrees/particle-solvers          # branch svgd
STAGE=/tmp/claude-1000/-home-tommy-Documents-programming-work-rlg-analytic-and-optimization-ik-learned-ik/27aa4506-d9d8-4485-b48d-8f0ef51ffca8/scratchpad/stage-svgd
git -C $WT status --porcelain --untracked-files=no   # must be empty
python $WT/cluster/gen_manifest.py --selftest        # must end "selftest OK" (also catches stale manifests)
bash $WT/cluster/make_staging_clone.sh "$STAGE"      # re-run after every commit: it refreshes to the tip
git -C $STAGE log -1 --oneline                       # must be the svgd tip you mean to stage
```

**Which cluster tree.** Option A is the default; use B only if A refuses.

- **A: `~/learned-ik`, the default `SC_ROOT`.** It has the venv, Drake pin, caches and the Panda
  `n6` checkpoint already.
  - `stage_code.sh` refuses while any job of that tree is queued or running, for example a pending
    `REMEASURE_NLOPT`. **Do not `FORCE_STAGE=1` over a live campaign.**
  - `svgd` contains `main` (34497a7), so the tree only gains code. But every later run from that
    tree runs svgd's `generic_program.py` / `benchmark.py` until main is restaged.
- **B: an isolated tree, `SC_ROOT=learned-ik-svgd` (job prefix `svgd`).** Export it in every
  command below. It needs the full setup first: `setup_supercloud.sh` as a job on `download`
  (~7 GB, MaxJobs=1 there), then `smoke.sh`, per `cluster/README.md` "Order of operations"
  steps 0-2, with `~/learned-ik` replaced by `~/learned-ik-svgd`. That costs hours, not minutes,
  so it does not fit tonight's plan.

```bash
bash $STAGE/cluster/stage_code.sh                    # refuses while a job of this tree runs
ssh tcohn@txe1-login.mit.edu 'cat ~/learned-ik/repo/.staged-commit'   # must print $TIP
```

## Submit: SMOKE, R2, R1 (gated on the smoke), R1K (gated and held)

`submit_bench.sh` already passes `DEPENDENCY=` through as `#SBATCH --dependency`, so no script
change was needed. A chained submission checks the checkpoint at job start rather than at submit.
Run from `$STAGE` (it reads the local manifest for its item count and checkpoint list):

```bash
cd $STAGE
PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_SMOKE.txt 1   # -> "Submitted batch job <SMOKE>"
PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R2.txt 2      # independent; ~1.3 h on 2 nodes
DEPENDENCY=afterok:<SMOKE> PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R1.txt 4
DEPENDENCY=afterok:<SMOKE> PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R1K.txt 4
ssh tcohn@txe1-login.mit.edu 'scontrol hold <R1K job ids>'     # held until the local screen is read
# later, to let it run:  ssh tcohn@txe1-login.mit.edu 'scontrol release <R1K job ids>'
```

R1K's four jobs queue behind R1 under the 4-node cap anyway. The hold keeps them from taking a node
the moment one frees, so they start only when you release them.

- **Never `MPS=1` here, and never PROCS above 2.** The point of R2 is one solve per GPU.
- The smoke takes ~10-15 min on one node: 4 cells x 2 arms x <= 60 s, plus process start, the flow
  compile and graph capture, and the svgd warm-up.
- **`afterok` gates on the smoke JOB's exit status, not on its numbers.**
  - `run_items.sh` exits nonzero if any item's benchmark process exited nonzero. That covers a
    crash, a failed compile or capture, and `_abort_on_dead_arm`.
  - If the smoke fails, R1's and R1K's jobs sit `PENDING (DependencyNeverSatisfied)`. `scancel`
    them.
  - A smoke that runs cleanly but solves nothing still releases R1. To gate R1 on the numbers too,
    hold it the same way as R1K, read the smoke, then `scontrol release`.
- What to read in the smoke's `summary.json` before R1 matters:
  - zero `error` records;
  - `metadata.svgd_warmup_seconds` present for both arms;
  - the solver and `verify()` feasibility agreeing on every cell (`solver_feasible`,
    `drake_feasible`, `feasible`);
  - wall <= cap + 1 s;
  - **mean wall per cell**, which is the R1 allotment's real base.

## Weekend filler: stage PAPER

These are the record's rows again at the **paper's condition, one solve per GPU**. Stage REMEASURE ran
PROCS=8 under `MPS=1`, a development condition whose learned wall times carry a 1.12-1.19x premium.

`manifest_stagePAPER.txt` holds stage REMEASURE's primary items **verbatim** (identical args,
lifted budgets, record flags), re-tagged `sc_PAPER_` in place of `sc_REMEASURE_` with the same
suffix. It covers four robots: Panda `n6`, iiwa `n4`, soft PCS `soft12` `n6` and screw
`screw7_p050` `n6`. Each runs grasp and pose, both protocols, under IPOPT and SNOPT. That is 32
runs, minus the 4 Panda IPOPT runs that `SVGD_R2` already measures: **28 logical runs, 224 items**,
sharded 8-way as REMEASURE was. It excludes GVS, NLopt and the LEGACY / RULE controls. The selftest
holds every item equal to its REMEASURE item apart from the tag.

The claim order is:

1. Panda SNOPT
2. iiwa IPOPT
3. iiwa SNOPT
4. screw IPOPT
5. screw SNOPT
6. soft PCS IPOPT
7. soft PCS SNOPT

Within each, grasp runs before pose.

| robot / solver | runs | items | node-h (PROCS=2) | max item h |
| --- | --- | --- | --- | --- |
| Panda SNOPT | 4 | 32 | 5.6 | 0.72 |
| iiwa IPOPT | 4 | 32 | 1.7 | 0.17 |
| iiwa SNOPT | 4 | 32 | 7.0 | 0.73 |
| screw IPOPT | 4 | 32 | 2.3 | 0.25 |
| screw SNOPT | 4 | 32 | 5.5 | 0.60 |
| soft PCS IPOPT | 4 | 32 | 2.4 | 0.21 |
| soft PCS SNOPT | 4 | 32 | 9.2 | 1.15 |
| **total** | **28** | **224** | **33.6** (8.4 h on 4 idle nodes) | |

`python cluster/gen_manifest.py --stage PAPER --allotment` prints this table. The scaling
assumptions are:

- **Base:** stage REMEASURE's own *unclamped* mean wall per cell, from its merged summaries.
  Unclamped, because SNOPT at the lifted budget overruns its clock on cycling cells, and that is
  node time; soft PCS grasp paired has one 2.2 h cell.
- **Learned arm:** its flow share (0.96 rigid, 0.75 soft PCS) sheds the MPS premium, divided by
  1.155, the middle of 1.12-1.19x. The rest of the learned iteration and all of joint space are
  CPU-bound and left as measured, which is slightly pessimistic.
- **Throughput:** two solves in flight per node instead of eight. That is 4x fewer, and it is where
  most of the cost comes from: about 34 node-hours here against REMEASURE's 17 for 40 runs.
- **Item length:** the figures are means. A shard holding several cycling SNOPT cells runs longer,
  but REMEASURE ran these exact shards inside the 8 h `ITEM_TIMEOUT`.

Every checkpoint it names (`panda__n6`, `iiwa14__n4`, `soft12__n6`, `screw7_p050__n6`) is already on
`~/learned-ik` from stage REMEASURE. The staging clone carries only the Panda one, and rsync's
protect filter keeps the others. `submit_bench.sh` checks them at submit time.

Submit it **held**, behind R1, at one solve per GPU and **never** under MPS:

```bash
cd $STAGE
PROCS=2 bash cluster/submit_bench.sh manifest_stagePAPER.txt 4
ssh tcohn@txe1-login.mit.edu 'scontrol hold <PAPER job ids>'
# when R1 (and R1K, if released) no longer need the nodes:
ssh tcohn@txe1-login.mit.edu 'scontrol release <PAPER job ids>'
```

The tags pair cell for cell with `sc_REMEASURE_*`: same grid, scene, args and code (svgd contains
main). Use the record for verdicts and `sc_PAPER_*` for paper-condition seconds.

## Weekend rounds R3-R5

These are three single-factor rounds, each on R1's four Panda rows (`mugshelf` and `posetip`, paired
and native), **learned arm only** (`--arms learned`, `SVGD_ROUND_ARMS`). Thomas, 2026-10-10 evening:
*"drop joint space ablation arm from R3, R4, R5."* Joint-space svgd is an ablation, measured at the
default setting by R1 (`kq`) and R1K (`knone`); with the kernel off it runs to the 180 s clock on most
grasp cells, which made R1K about three times slower than R1 and would have pushed the variant ladder
past the 10-12 maintenance window. The first submission (jobs 5883538-49, then 5884557-68) carried
both arms and was cancelled unrun; the learned-only manifests were resubmitted as jobs 5886385-88 (R3),
5886389/90/92/96 (R4) and 5886397-400 (R5) at 21:50 EDT behind R1K.

**Shared settings.** All items use the record flags and PROCS=2 with no MPS. They are graphed and
carry the lifted step cap.

**Pinned values.** Each item carries `svgd_rho=1000 svgd_gn_lm=10 svgd_lr=0.3` (`SVGD_PINNED_V1`; a stage from R20 on takes `SVGD_PINNED_V2`, the 2026-10-12 pin), plus
**one** `--set` per variant. An override of a pinned name **replaces** that value in place, so every
item names each option exactly once. The selftest holds this, along with:

- the counts;
- each item's method settings being exactly the pinned values plus its one override;
- every item being its IPOPT row with only the documented swap;
- the allowed `--set` list, extended by `svgd_n` and `svgd_temperature`.

The tags are `sc_SVGD_R<k>_panda_n6_svgd_<row>_480_180_<start>_<variant>`.

| manifest | variant | the one change against R1's `kq` | shards | runs | items | node-h (assumed) | max item h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `manifest_stageSVGD_R3.txt` | `n1` / `n256` | `svgd_n=1` / `svgd_n=256` | 8 / 24 | 8 | 128 | 5.9 | 1.5 |
| `manifest_stageSVGD_R4.txt` | `rho1e4` / `lr1` | `svgd_rho=10000` / `svgd_lr=1.0` (replacing 1000 / 0.3) | 24 / 24 | 8 | 192 | 7.5 | 1.9 |
| `manifest_stageSVGD_R5.txt` | `n16` / `T10` | `svgd_n=16` / `svgd_temperature=10` | 16 / 24 | 8 | 160 | 6.7 | 1.7 |

Learned only, that totals about 20 node-hours, or about 5 h on 4 idle nodes, after R1K (the two-arm
estimate was 116 node-hours).

**The cost estimate is an assumption, not a measurement.** It charges 5 s per learned cell. R1 measured
a 6-12 s median on the V100s at N = 64 (`n256` costs about 4x that per step, `n1` far less), so expect
R3 to run longer than the estimate.
`python cluster/gen_manifest.py --stage SVGD_R3 --allotment` prints this together with the other
svgd manifests.

**Item length.** Every item fits inside the 8 h `ITEM_TIMEOUT` even with every cell at the clock.
The worst is `n1` at 8 shards: 60 cells x 2 arms x 180 s = 6 h.

**Maintenance.** At this estimate the rounds will not all finish before the Monday-evening window.
The queue is FIFO, so R3 is the most likely to land. After the window, run `--reclaim` and resubmit.

Submit with no dependency flags. The rounds queue FIFO behind R1K:

```bash
cd $STAGE
PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R3.txt 4
PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R4.txt 4
PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R5.txt 4
```

`submit_bench.sh` refuses any manifest that is not staged on the cluster. These three are new, so the
staged tree must hold them first.

## Rounds R6, R8, R9, R10

These continue the ladder into phase 2, stripping out what is not important (the coordinator and
Thomas, 2026-10-11 overnight: *"You can use local and cluster compute"*). Each rung attributes **one**
setting of the primary `kq`. The conventions are those of R3-R5: learned arm only, `SVGD_PINNED_V1`
plus one override that replaces a pinned name in place, the record flags, graphed, the lifted step
cap, and PROCS=2 with no MPS. R7, the seed control, needed IPOPT twins on a new grid and waited for
Thomas. He approved it on 2026-10-11, and it has its own section below.

| manifest | variant | the one change against `kq` | what it attributes |
| --- | --- | --- | --- |
| `manifest_stageSVGD_R6.txt` | `lr0p1` | `svgd_lr=0.1` (replacing 0.3) | the step size, below the pin (R4's `lr1` is above it) |
| | `K1` / `K100` | `svgd_inner_iters=1` / `=100` (default 10) | the dual-ascent / resample / stop-check period K |
| `manifest_stageSVGD_R8.txt` | `n4` / `n32` / `n128` | `svgd_n=4` / `32` / `128` (default 64) | the particle count, filling in R3/R5's 1 / 16 / 256 |
| `manifest_stageSVGD_R9.txt` | `rho100` / `rho1e5` | `svgd_rho=100` / `=100000` (replacing 1000) | the AL penalty, either side of R4's `rho1e4` |
| | `clip1e6` | `svgd_multiplier_max=1000000` (default 1e4) | the multiplier box |
| `manifest_stageSVGD_R10.txt` | `initnative` | `svgd_paired_init=native` (default `jitter`) | the paired swarm's draw: around the shared guess, or the arm's own distribution |

**`initnative` runs the PAIRED rows only.** The native rows already draw the native distribution, so
a native item would rerun `kq`. The row filter is the third field of each `SVGD_ROUNDS` entry
(`starts`), and the selftest's expected counts and rows follow it.

**The variant token is `lr0p1`, not `lr0.1`**, following the tree's tag convention (`mstep0p5`):
no dot in a tag that names a results directory.

**Under `K1` and `K100`, the `iterations` column is not comparable with `kq`'s.** It counts outer
checks, and a check is every K steps. Compare these rungs on success and wall. The step cap
(`svgd_outer_iters=1000000` checks) is 1e6 steps at K = 1. That binds within the 180 s clock only
if a step takes under 0.18 ms, so read `hit_iteration_cap` anyway.

**Shards.** Every variant uses 24 shards, as R4 does, `n128` included (R3's `n256` uses 24 too). Each
item is 20 cells. With every cell at the clock, a learned-only item takes 1 h, far inside the 8 h
`ITEM_TIMEOUT`.

**Allotment.** `python cluster/gen_manifest.py --stage SVGD_R6 --allotment` prints these lines. They
assume 5 s per learned cell, which is optimistic, since R1 measured a 6-12 s median at N = 64:

```
  manifest      runs  cells  items  solve-h  node-h    h@4  max item h
  SVGD_R6         12   5760    288     22.4    11.2    2.8        0.08
  SVGD_R8         12   5760    288     22.4    11.2    2.8        0.08
  SVGD_R9         12   5760    288     22.4    11.2    2.8        0.08
  SVGD_R10         2    960     48      3.7     1.9    0.5        0.08
```

Together that is about 35 node-hours, or about 9 h on 4 idle nodes, queued FIFO behind R3-R5. The
Monday-evening maintenance window (2026-10-12 to 10-14) kills whatever is still running. After it,
`--reclaim` and resubmit.

**Staging the manifests.** `stage_code.sh` refuses while campaign jobs of the tree are queued or
running, so it cannot restage the code for these. They need no new code, because every option they
set is already in the staged tree's `ProgramOptions`. So copy the four manifests into the staged
tree by hand:

```bash
scp $WT/cluster/manifest_stageSVGD_R{6,8,9,10}.txt tcohn@txe1-login.mit.edu:~/learned-ik/repo/cluster/
```

**Check before submitting:** confirm that the staged commit has `svgd_inner_iters`,
`svgd_multiplier_max` and `svgd_paired_init` (with `native`). An unknown option name raises at
options construction and turns every cell into an instant failure.

```bash
ssh tcohn@txe1-login.mit.edu 'grep -c "svgd_inner_iters\|svgd_multiplier_max\|svgd_paired_init" ~/learned-ik/repo/src/generic_program.py'
```

Then submit with no dependency flags. **These four run on the clearance row under `DRAKE_ENV=crc`**
(next section): they were submitted on the default row on 2026-10-11 overnight, and when the row
landed Thomas (14:45, *"restage everything using batch clearances"*) had them cancelled -- R6 at 72
of 288 items, R8-R10 unstarted -- and regenerated with the pin. R6's old-row items are void, kept on
the cluster as `~/learned-ik/void_oldrow_R6_*`, never collected.

```bash
cd $STAGE
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R6.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R8.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R9.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R10.txt 4
```

## From R11 on: the clearance collision row, under `DRAKE_ENV=crc`

Thomas, 2026-10-11: the batched clearance row (`svgd_collision_backend=clearances`,
`src/svgd/collision_backend.py`) is a strict improvement and the svgd rounds use it from now on.
**Every svgd item of R6, R8, R9, R10, R11, R12, SMOKE_IIWA and R7 (and of R13-R16, SMOKE_SCREW and
SMOKE_SOFT, their own section below) carries `--set svgd_collision_backend=clearances`** (`SVGD_COLLISION` / `SVGD_CLEARANCE_STAGES` in
`cluster/gen_manifest.py`); nothing else changed in those manifests, and every other manifest
regenerates byte-identical. R1-R5 finished on the default `autodiff` row and keep it; **the IPOPT twins (R2, R7IPOPT, and stage PAPER for the iiwa) stay on
the tarball pin** and never carry it. The selftest holds all of that, plus the value being one
`ProgramOptions` accepts.

**Why** (local measurement, `docs/svgd-solver.md`, "Batched clearance row"): the same functional
from one batched, GIL-released, Drake-threaded `CalcRobotClearances` call -- values bitwise equal
to the program's binding, gradients within 4.1e-14 of their largest entry. It is 7-10x per
configuration at 20 threads, and end to end (Panda paired, learned arm, 10 cells, 60 s) it gives
3.5-4.6x lower median wall per cell (pose 3.68 -> 1.04 s, grasp 7.61 -> 1.65 s) at identical
success (10/10 on both backends, both tasks). It is an EXECUTION setting like `svgd_compile`: the
selftest does not count it as a round's one factor, and the reporter names it in full in every
setting diff.

**The binding exists only on Thomas's Drake branch `calc-robot-clearances`** (commit 15a2c97228), so
these four manifests run under the second environment, `~/learned-ik/venv-crc`: the project venv's
exact freeze plus that branch's wheel, with the tarball OFF `PYTHONPATH` (`cluster/README.md`, "A
second Drake for the svgd clearance row"). Submit them with **`DRAKE_ENV=crc`**; `submit_bench.sh`
refuses them without it, refuses `DRAKE_ENV=crc` on any manifest that is not all-clearances, and
refuses `crc` until `~/learned-ik/verify_venv_crc.DONE` reads `OK`. Under `crc`, `run_items.sh` sets
`DRAKE_NUM_THREADS` = the job's CPUs / PROCS (logged in each worker's header); without it the worker's `OMP_NUM_THREADS=1`
would leave the row on one thread (~1.2x). A tree whose code predates the option (or a pydrake
without the binding) refuses `clearances` when `ProgramOptions` is built, so every cell would fail
instantly: hence the guards.

**venv-crc is built and verified (2026-10-11).** Job 5892137 (`download`) built it; the freeze
gained exactly `drake @ file://...crc.15a2c97...whl`, `Mosek==11.1.2` and `pydot==4.0.1` and changed
nothing else (`~/learned-ik/venv-crc-build/freeze.*.txt`; 5892040 was the same build stopped by a
bad import check, rebuilt from scratch). It is 7.7 GB; `home-crc`'s drake_models cache is 74 MB and
carries the same key as `home`'s. Job 5892228 (`debug-cpu`, no internet) passed every check in
`verify_venv_crc.sh`. That covers the SNOPT, IPOPT and NLopt solves, the plant step, all four
svgd-context scenes built offline, and `tests/test_collision_clearances.py` ALL PASS at 16 Drake
threads under `OMP_NUM_THREADS=1`: values bitwise, gradients within 4.1e-14 relative, 16 threads ==
1 thread bitwise. Job 5892222 passed at only 1 thread, because `nproc` honours `OMP_NUM_THREADS`,
the fix that `run_items.sh` now carries. The test ran from `~/learned-ik/scratch-crc`, because the
campaign tree (adf534c) predates it.

**Read these rounds with that in mind.** The clearance row cuts the collision share of a step from
~0.9 to ~0.3, so at the 180 s clock a cell gets roughly 3-4x more steps. R6, R8-R11's variants are
read against R1's `kq`, and R7's `kq@seed2` beside the seed-1 `kq`; both baselines ran the
`autodiff` row, so **those comparisons carry the backend as a second difference** (in seconds and,
through the clock, possibly in success). Wall from these rounds is not comparable to R1-R5's. The IPOPT-twin pairings (R12 against PAPER, R7 against
R7IPOPT) are unaffected in kind -- the twin is a different solver anyway -- but now also run on a
different Drake build. Record metadata says which: `drake_version_txt` reads
`0.0.20261011+crc.15a2c97 15a2c97...` and `drake_path` is inside `venv-crc`.

## Round R11: what happens to a runaway particle

R11 attributes the primary's handling of a runaway (`|q|_inf > svgd_resample_q_max`) or non-finite
particle, `svgd_resample` (2026-10-11). The conventions are R4's: the four Panda rows, learned arm
only, N = 64, `SVGD_PINNED_V1` plus the one override, 24 shards, both protocols, PROCS=2 with no MPS.

| manifest | variant | the one change against `kq` | what it attributes |
| --- | --- | --- | --- |
| `manifest_stageSVGD_R11.txt` | `drop` / `keep` | `svgd_resample=drop` / `=keep` (default `native`) | `native` redraws the particle from the arm's native start; `drop` removes it from the swarm; `keep` lets a runaway keep stepping and drops only a non-finite particle |

**Allotment** (the same 5 s per learned cell assumption as R4):

```
  manifest      runs  cells  items  solve-h  node-h    h@4  max item h
  SVGD_R11         8   3840    192     14.9     7.5    1.9        0.08
```

**R11 needs a code restage.** `svgd_resample` is a new option, so the staged tree must carry this
commit. A tree without it refuses the option when `ProgramOptions` is built, so every cell fails
instantly. `stage_code.sh` refuses while campaign jobs are queued or running, so restage once the
queue is idle. Then check the staged tree and submit:

```bash
ssh tcohn@txe1-login.mit.edu 'grep -c "svgd_resample\"" ~/learned-ik/repo/src/generic_program.py'   # must be >= 1
ssh tcohn@txe1-login.mit.edu 'grep -c "svgd_collision_backend" ~/learned-ik/repo/src/generic_program.py'   # must be >= 1
cd $STAGE
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R11.txt 4
```

Read the results with `scripts/report_svgd.py --stage SVGD --variants kq,drop,keep`. Each record
carries `resample_mode`, `n_dropped`, `n_live` and `n_resampled`. A cell whose every particle was
dropped stops with `stop_reason = all_dead`. It is scored infeasible, and `recovered_*` holds the
last iterate it recorded.

## R12 (iiwa n4) and SVGD_SMOKE_IIWA

Thomas (2026-10-11) wants the other robots. R12 is the primary `kq` on the **iiwa at its adopted
`n4`** (`models/iiwa14/iiwa14__n4__step620000.pkl`, already on `~/learned-ik` from stage
REMEASURE/PAPER): the record's four iiwa rows (`mugshelf` / `posetip` x paired / native), 480 cells,
180 s, `--solver svgd --compile` and exactly R1's `kq` `--set` list plus the clearance row. Diffed
against R1's `kq` line, an R12 item differs in the script (`scripts/iiwa/iiwa_benchmark.py`), the
checkpoint, `--arms learned`, the tag and `--set svgd_collision_backend=clearances` (from R11 on,
section above; submitted under `DRAKE_ENV=crc`); nothing else. The same holds for SMOKE_IIWA and R7.

- **Learned arm only**, 24 shards, as R4. Tags `sc_SVGD_R12_iiwa_n4_svgd_<row>_480_180_<start>_kq`.
- **Its IPOPT twin is stage PAPER's iiwa IPOPT runs**, `sc_PAPER_iiwa_n4_ipopt_<row>_480_180_<start>`:
  the record's rows at paper conditions (PROCS=2, no MPS), already collected. No new twin manifest.
  The selftest holds R12's IPOPT rows, built at PAPER's 8 shards under PAPER's tag, equal to PAPER's
  iiwa IPOPT items (ids, scripts, args). So grid, scene and cap flags are PAPER's by construction.
- The robot is a **context** in `gen_manifest.py` (`SVGD_STAGE_CONTEXT`, `SVGD_TWIN_OF`). The Panda
  manifests regenerate byte-identical.

`SVGD_SMOKE_IIWA` is the joint-space safety smoke. It has **one item per row** (4 items), each a
2 x 2 grid (4 cells) at 60 s, `--arms learned,numerical`, at the same primary setting. Tags are
`sc_SVGDSMOKE_iiwa_n4_svgd_<row>_4_60_<start>_kq`. It exists so the joint-space arm is seen to run
on the iiwa before anyone trusts the learned-only R12. It is not a result and pairs with nothing.

**Allotment** (`python cluster/gen_manifest.py --stage SVGD_R12 --allotment`). The assumption is the
rounds' **5 s per learned cell** (`SVGD_ROUND_CELL_S`), with no iiwa svgd measurement behind it. R1
measured a 6-12 s median per cell on the Panda at N = 64, so expect R12 to run longer. The smoke is
charged at its clock, both arms.

```
  manifest          runs  cells  items  solve-h  node-h    h@4  max item h
  SVGD_R12             4   1920     96      7.5     3.7    0.9        0.08
  SVGD_SMOKE_IIWA      4     16      4      0.7     0.4    0.1        0.18
```

The worst case puts every cell at the 180 s clock: 96 solve-h, or 48 node-h, about 12 h on 4 idle
nodes. Each item is then 20 cells x 180 s = 1 h, far inside `ITEM_TIMEOUT`.

**Submit order.** SMOKE_IIWA goes first. R12 goes `afterok` on it, and both go after R11 in FIFO.
Both need the same restage R11 needs: the manifests are new, and the staged tree must carry this
commit. Do this after the restage (R11's section):

```bash
cd $STAGE
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_SMOKE_IIWA.txt 1   # -> <SMOKE_IIWA>
DEPENDENCY=afterok:<SMOKE_IIWA> DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R12.txt 4
```

`afterok` gates on the smoke JOB's exit status, not on its numbers, as with SVGD_SMOKE. A smoke that
runs and solves nothing still releases R12. Before reading R12, read the smoke's four
`summary.json`:

- zero `error` records on **both** arms (`_abort_on_dead_arm` would have failed the job);
- `metadata.svgd_warmup_seconds` present for both arms;
- solver / Drake / `verify()` feasibility agreeing;
- wall <= cap + 1 s.

## R7: the seed control (SVGD_R7IPOPT, SVGD_R7)

Approved by Thomas, 2026-10-11. These are the Panda `n6` rows on a **second grid, `--seed 2`**,
where everything else runs seed 1. `--seed` is what both benchmark scripts draw the grid from
(`np.random.default_rng(args.seed)` for the targets and guesses). `grid_hash` hashes those draws, so
a seed-2 run cannot share R2's hash. There are two manifests:

| manifest | what | runs | items | node-h (PROCS=2) | h on 4 idle nodes |
| --- | --- | --- | --- | --- | --- |
| `manifest_stageSVGD_R7IPOPT.txt` | the IPOPT twins: stage SVGD_R2's items exactly, `--seed 2`, both arms, 8 shards; tags `sc_SVGD_R7_panda_n6_ipopt_<row>_480_180_<start>` | 4 | 32 | 2.6 | 0.6 |
| `manifest_stageSVGD_R7.txt` | svgd at `kq`, learned arm only, 24 shards; tags `sc_SVGD_R7_panda_n6_svgd_<row>_480_180_<start>_kq` | 4 | 96 | 3.7 (assumed) | 0.9 |

The selftest holds every R7IPOPT item equal to its SVGD_R2 item apart from `--seed` and the tag. It
also holds every R7 item equal to its R7IPOPT row plus R1's `kq` swap.

**The R7IPOPT figure is R2's.** It is charged at the same Panda IPOPT means, so the estimate is 2.6
node-h, about **0.6 h on 4 idle nodes**. (The "~0.6" quoted for R2 is hours on 4 idle nodes, not
node-hours.) What R2 measured: 3.2 solve-h of solves, plus 32 x 180 s per-item overhead. That is
about 2.4 node-h at PROCS=2, consistent with the estimate. R7 uses the rounds' 5 s per learned cell
assumption.

**Submit order.** Both manifests can run any time after the restage. They are independent of each
other, with no `afterok` needed, and they queue FIFO behind R12:

```bash
cd $STAGE
PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R7IPOPT.txt 2              # the tarball: no DRAKE_ENV
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R7.txt 4
```

**How it is read.** `scripts/report_svgd.py --stage SVGD` prints R7 as the column `kq@seed2` in the
Panda rows' LEAD table, beside `kq`. It is paired only with its own twin, `ipopt@seed2` (R7IPOPT). The
seed is in both the provenance check (`metadata.seed` must equal the item's `--seed`) and every
pairing's must-match list, beside `grid_hash`. So R7 is **refused** against R2's twin, and a seed-2
run that carries a seed-1 grid is refused too. The A/B against the seed-1 primary and the
variant-vs-variant tables do not pair across grids. They print a note line instead.

## R13-R16 and two robot smokes (Thomas, 2026-10-11 ~23:35)

Thomas approved three new rounds together: *"Launch them all, we don't want to waste GPU time."*
All of them follow the R11 / R12 / SMOKE_IIWA / R6-R10 conventions:

- `SVGD_PINNED_V1` plus each variant's overrides. An override of a pinned name replaces it in place.
- The record flags, graphed mode and the lifted step cap.
- 24 shards for every 480-cell row, and PROCS=2 with no MPS.
- **Every svgd item carries the clearance row** (`svgd_collision_backend=clearances`). So every one
  of these manifests is submitted with `DRAKE_ENV=crc`.
- The rounds run the learned arm only. Only the smokes run both arms.

### R13: the kernel space, single factor

The Panda `n6` and its four rows, one override of `svgd_kernel` per variant. `kq` (the kernel on
q, the default, unset) is the reference.

| variant | the one change against `kq` | the repulsion acts on |
| --- | --- | --- |
| `kz` | `svgd_kernel=z` | the latent block of y alone |
| `kcz` | `svgd_kernel=cz` | the product of the conditioning-pose RBF and the latent RBF, each with its own bandwidth |
| `kqz` | `svgd_kernel=qz` | the product of the q RBF (pulled back through the flow) and the latent RBF (direct) |
| `kx` | `svgd_kernel=x` | the whole normalised decision vector y = x / s |

The four are `z / cz / qz / x` in `SVGD_ENUM_FIELDS["svgd_kernel"]` (`src/generic_program.py`).
`z`, `cz` and `qz` are refused on the joint-space arm at solver construction, which a learned-only
round never reaches. The selftest now checks every round override of a choice option against that
tuple, read from the source text, so a value outside it fails the selftest rather than every cell.

### R14: declared two-factor combinations

The same Panda rows. This is the one round allowed more than one factor.
`SVGD_TWO_FACTOR_ROUNDS` (since R18, `SVGD_DECLARED_FACTORS`) in `cluster/gen_manifest.py` declares each variant's two factors. The
selftest holds every R14 item to **exactly** those overrides, with nothing missing and nothing
extra. It still holds every other round to one factor, or zero for the primary carried into a new
context.

| variant | factor 1 | factor 2 |
| --- | --- | --- |
| `kzHonT0p1` | `svgd_kernel=z` | `svgd_metric_temperature=on` + `svgd_temperature=0.1`, which together are one factor: the temperature under the H/T metric |
| `kzrho1e5` | `svgd_kernel=z` | `svgd_rho=100000`, replacing the pinned 1000 (R9's cost lever) |
| `kzn128` | `svgd_kernel=z` | `svgd_n=128` (R8's) |
| `kczrho1e5` | `svgd_kernel=cz` | `svgd_rho=100000` |

**Why the temperature pair is one factor.** At T = 1 the H/T metric is H, so the metric form cannot
be varied on its own: switching `svgd_metric_temperature` on changes nothing until T moves. A
temperature under the default metric is a different factor (R5's `T10`). Read `kzHonT0p1` against
`kz` for the effect of the temperature, and against `kq` only as the combination.

### R15 (screw arm) and R16 (soft PCS arm), with SVGD_SMOKE_SCREW / SVGD_SMOKE_SOFT

These copy R12's shape onto the two other robots of the record. Each runs the primary `kq` with
**zero overrides** on the robot's four rows: learned arm only, 480 cells, 180 s, 24 shards.

- **R15** is the screw arm `screw7_p050` at its fielded `n6`
  (`models/screw7_p050/screw7_p050__n6__step620000.pkl`, `scripts/screw_arm/screw_arm_benchmark.py`).
  Tags are `sc_SVGD_R15_screw7_p050_n6_svgd_<row>_480_180_<start>_kq`.
- **R16** is the soft PCS arm `soft12` at its fielded `n6`
  (`models/soft12/soft12__n6__step620000.pkl`, `scripts/soft_arm/soft_arm_benchmark.py`).
  Tags are `sc_SVGD_R16_soft12_n6_svgd_<row>_480_180_<start>_kq`.
- **Two rounds, not one.** `SVGD_STAGE_CONTEXT` gives each stage one context.
- **Their IPOPT twins are stage PAPER's runs on each robot**, as R12's is for the iiwa:
  `sc_PAPER_screw7_p050_n6_ipopt_<row>_480_180_<start>` and `sc_PAPER_soft12_n6_ipopt_...`. Both
  are collected, so there is no new twin manifest.
  - Neither robot is in `ADOPTED_RUNGS`, so their rows come from `stage_SCREW` / `stage_SOFT12`,
    as stage REMEASURE and PAPER build them. `SVGD_RUNGS` maps each svgd chart to its builder.
  - The selftest holds each robot's IPOPT rows, built at PAPER's 8 shards under PAPER's tag, equal
    to PAPER's items (ids, scripts, args).
  - The checkpoint paths are therefore the ones PAPER already ran on `~/learned-ik`.
- **The two smokes** have SMOKE_IIWA's shape: one 2 x 2 item per row (4 items), 60 s, `--arms
  learned,numerical`, `kq`. Tags are `sc_SVGDSMOKE_<robot>_n6_svgd_<row>_4_60_<start>_kq`. They
  exist so the joint-space arm, and the clearance row on these two scenes, are seen to run before
  the learned-only rounds are read. They are not results.

**Which svgd path each arm takes.** These come from the local plumbing smoke (`docs/svgd-solver.md`,
"Other robots"). The screw arm runs the compiled / graphed split step, as the rigid arms do. The
soft PCS arm has no closed-form row Jacobians through its floating joints, so it takes the eager
autograd step. Locally that cost 70-126 ms per step at N = 16, against 6-15 ms on the screw arm.
**Expect R16 to run far above the 5 s per learned cell assumption**, and possibly to the clock.
With every cell at the clock, an item is 20 cells x 180 s = 1 h, still far inside `ITEM_TIMEOUT`.

**Never exercised: the clearance row on these two scenes.** venv-crc's verification and
`tests/test_collision_clearances.py` cover the Panda (and the iiwa through R12 / SMOKE_IIWA). The
local plumbing smoke ran the default `autodiff` row. `ClearanceEvaluator` refuses at construction
when the checker's pair set is not the constraint's. The soft PCS arm's 33 floating sub-links make
that the case to watch. A refusal fails every cell of the arm. `_abort_on_dead_arm` then fails the
smoke job, and `afterok` holds its round. **That is what the smokes are for.**

### Allotment

`python cluster/gen_manifest.py --stage SVGD_R13 --allotment` prints these lines. The rounds assume
5 s per learned cell, which is optimistic: R1 measured a 6-12 s median at N = 64, and `kzn128`
runs N = 128. The smokes are charged at the clock on both arms.

```
  manifest          runs  cells  items  solve-h  node-h    h@4  max item h
  SVGD_R13            16   7680    384     29.9    14.9    3.7        0.08
  SVGD_R14            16   7680    384     29.9    14.9    3.7        0.08
  SVGD_R15             4   1920     96      7.5     3.7    0.9        0.08
  SVGD_R16             4   1920     96      7.5     3.7    0.9        0.08
  SVGD_SMOKE_SCREW     4     16      4      0.7     0.4    0.1        0.18
  SVGD_SMOKE_SOFT      4     16      4      0.7     0.4    0.1        0.18
```

At the assumption that is about 38 node-hours, or about 9.5 h on 4 idle nodes. The worst case puts
every cell at the clock: 384 items x 1 h for each of R13 and R14 (192 node-h each) and 96 x 1 h for
each of R15 and R16 (48 node-h each), about 120 h on 4 nodes. That cannot finish before the
maintenance window (Mon 2026-10-12 evening to Wed 10-14 morning). Whatever is still running then is
killed: `--reclaim` each manifest and resubmit after it.

### Staging: copy the six manifests, no restage

The staged tree is **7ca45a4** (`R6-R10 on the clearance row ...`). The code needs no restage:
`git diff 7ca45a4 <this commit> -- src/ scripts/ cluster/*.sh` changes only
`scripts/report_svgd.py`, which runs locally. Since 7ca45a4 the staged tree has carried every
option these manifests set:

- `svgd_kernel` with `z / cz / qz / x` and `svgd_metric_temperature` with `on / off`, in
  `SVGD_ENUM_FIELDS`;
- `svgd_temperature`, `svgd_n`, `svgd_rho` and `svgd_collision_backend=clearances`;
- the screw and soft PCS support in `src/svgd/batched_program.py` and
  `src/svgd/kinematics_from_plant.py` (f26cfaa, an ancestor of 7ca45a4);
- `--solver svgd` in both robots' benchmark scripts.

So copy the manifests into the staged tree by hand, as for R6-R10, and check it:

```bash
scp $WT/cluster/manifest_stageSVGD_{R13,R14,R15,R16,SMOKE_SCREW,SMOKE_SOFT}.txt tcohn@txe1-login.mit.edu:~/learned-ik/repo/cluster/
ssh tcohn@txe1-login.mit.edu 'cat ~/learned-ik/repo/.staged-commit'          # must print 7ca45a4...
ssh tcohn@txe1-login.mit.edu 'grep -c "\"svgd_metric_temperature\": (\"off\", \"on\")\|\"svgd_kernel\": (\"q\", \"none\", \"x\", \"z\", \"c\", \"cz\", \"qz\", \"pose\")" ~/learned-ik/repo/src/generic_program.py'   # must print 2
ssh tcohn@txe1-login.mit.edu 'grep -c "TorchConfigToPlantQ" ~/learned-ik/repo/src/svgd/batched_program.py'   # must be >= 1
ssh tcohn@txe1-login.mit.edu 'ls -l ~/learned-ik/repo/models/screw7_p050/screw7_p050__n6__step620000.{pkl,arch.json} ~/learned-ik/repo/models/soft12/soft12__n6__step620000.{pkl,arch.json}'
```

### Submit

Submit from the worktree `$WT`. `submit_bench.sh` reads the manifest from the checkout it lives in,
and the staging clone `$STAGE` (at 7ca45a4) does not have these six. Its scripts are unchanged
since 7ca45a4. The smokes go first, each robot round `afterok` on its own smoke, and R13 then R14
in FIFO with no dependency. The unchained submissions check at submit time that their checkpoints
exist on the cluster. The chained ones check at job start.

```bash
cd $WT
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_SMOKE_SCREW.txt 1   # -> <SMOKE_SCREW>
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_SMOKE_SOFT.txt 1    # -> <SMOKE_SOFT>
DEPENDENCY=afterok:<SMOKE_SCREW> DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R15.txt 4
DEPENDENCY=afterok:<SMOKE_SOFT> DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R16.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R13.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R14.txt 4
```

`afterok` gates on the smoke job's exit status, not on its numbers, as with SMOKE_IIWA. Before
reading R15 or R16, read the smoke's four `summary.json` files against the same checklist as
SMOKE_IIWA's: zero `error` records on both arms, `metadata.svgd_warmup_seconds` present for both
arms, solver / Drake / `verify()` feasibility agreeing, and wall <= cap + 1 s.

### Read

```bash
.venv/bin/python scripts/report_svgd.py --stage SVGD --variants kz,kcz,kqz,kx              # R13 beside kq
.venv/bin/python scripts/report_svgd.py --stage SVGD --variants kz,kzHonT0p1,kzrho1e5,kzn128,kcz,kczrho1e5,rho1e5,n128
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows screw7_p050                     # R15 against PAPER
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows soft12                          # R16 against PAPER
```

The reporter takes the new rounds and robots from `gen_manifest.py`. `--rows` now also accepts the
robots the stage builds, not just smoke.py's. Collected runs land in
`results/screw_arm/benchmark/` and `results/soft12/benchmark/`. Promote them the way the R12
section does: `cp -r <STAGING>/results/<dir>/benchmark/sc_SVGD_R1{5,6}_*` and `sc_SVGDSMOKE_*`.

## R17, R17S and R18 (queued 2026-10-12, the cluster idle until the maintenance)

R13 and R14 found that the kernel space `cz` and rho = 1e5 add: `kczrho1e5` is 480/480 on all four
Panda rows with grasp cost below IPOPT's learned arm. Two questions remain, and the cluster is idle
until the 10-12 maintenance, so they are queued now (Thomas, 2026-10-11: *"Launch them all, we
don't want to waste GPU time"*). The conventions are R13 / R14's exactly:

- `SVGD_PINNED_V1` plus each variant's overrides, the record flags, graphed mode, the lifted step cap.
- Learned arm only, 24 shards per 480-cell run, PROCS=2 with no MPS.
- **Every svgd item carries the clearance row**, so every manifest goes in with `DRAKE_ENV=crc`.

### R17 (iiwa) and R17S (screw arm): does the kernel-space gain transfer?

One context per stage, as before. `SVGD_R17` runs on R12's context (the iiwa at `n4`), `SVGD_R17S`
on R15's (the screw arm `screw7_p050` at `n6`). Each runs the robot's four rows with two variants:

| variant | factors | read against |
| --- | --- | --- |
| `kcz` | `svgd_kernel=cz` (one factor, R13's) | that robot's `kq` (R12 / R15) |
| `kczrho1e5` | `svgd_kernel=cz` + `svgd_rho=100000` (declared two, exactly R14's entry) | `kcz` for rho, `kq` only as the combination |

Tags are `sc_SVGD_R17_iiwa_n4_svgd_<row>_480_180_<start>_<variant>` and
`sc_SVGD_R17S_screw7_p050_n6_svgd_<row>_480_180_<start>_<variant>`. The IPOPT twins are the ones R12
and R15 already pair with: stage PAPER's runs on each robot, collected, so no new twin manifest.

**A variant token now means one setting everywhere.** The selftest checks that a token carries the
same overrides in every round that runs it. So `kcz` on the iiwa is R13's `kcz` carried there, and
the reporter's per-robot columns line up with the Panda's.

### R18 (Panda): declared three-factor stacks

The Panda `n6`, default context. Both variants stack a third factor on `kczrho1e5`:

| variant | factors | rows |
| --- | --- | --- |
| `kczrho1e5n128` | `svgd_kernel=cz` + `svgd_rho=100000` + `svgd_n=128` (R8's / R14's N lever) | all four |
| `kczrho1e5initnative` | `svgd_kernel=cz` + `svgd_rho=100000` + `svgd_paired_init=native` (R10's) | paired only |

`initnative` changes only how the paired swarm is drawn. A native item would be `kczrho1e5`'s run
again, so it runs the paired rows only, as R10's `initnative` does. Read each against R14's
`kczrho1e5` for the third factor.

### The declared-factor rule, generalised

`SVGD_TWO_FACTOR_ROUNDS` became `SVGD_DECLARED_FACTORS`: round -> {variant: its factors}. A variant's
factor count is the length of its declaration, so a round can declare three. The selftest:

- holds every declared variant's overrides to **exactly** its declared factors, as before;
- pins each declared variant's factor count: R14's four at two (unchanged), R17 / R17S's
  `kczrho1e5` at two, and R18's two at three;
- holds every undeclared variant of every round to one factor, or zero for the primary carried into
  a non-default context.

A round may now mix the two kinds. R17 / R17S's `kcz` is undeclared and one factor, while
`kczrho1e5` is declared two, and that is allowed in a non-default context exactly as in the default
one. R14's check is unchanged: all four of its variants are still declared, at two.

### Allotment

`python cluster/gen_manifest.py --stage SVGD_R17 --allotment` prints these lines (the base is the
assumed 5 s per learned cell; R18's `kczrho1e5n128` runs N = 128, where R8 found wall flat to 128):

```
  manifest          runs  cells  items  solve-h  node-h    h@4  max item h
  SVGD_R17             8   3840    192     14.9     7.5    1.9        0.08
  SVGD_R17S            8   3840    192     14.9     7.5    1.9        0.08
  SVGD_R18             6   2880    144     11.2     5.6    1.4        0.08
```

At the assumption that is about 21 node-hours, or about 5 h on 4 idle nodes. In the worst case every
cell is at the clock, each item 20 cells x 180 s = 1 h: 192 + 192 + 144 = 528 node-h at PROCS=2
over 4 nodes is about 66 h. That cannot finish before the maintenance window (Mon 2026-10-12 evening
to Wed 10-14 morning). Whatever is still running then is killed: `--reclaim` each manifest and
resubmit after it.

### Staging: copy the three manifests, no restage

The staged tree is still **7ca45a4**, and these manifests need no restage. Since 7ca45a4 the code
has changed only in `scripts/report_svgd.py`, which runs locally (`git diff 7ca45a4 <this commit> --
src/ scripts/ cluster/*.sh`). And 7ca45a4 already carries every option these manifests set:

- `svgd_kernel` with `cz` and `svgd_paired_init` with `native`, in `SVGD_ENUM_FIELDS`;
- `svgd_rho`, `svgd_n` and `svgd_collision_backend=clearances`.

R10 ran `svgd_paired_init=native` there, R14 `svgd_n=128` and `kczrho1e5`, and R12 / R15 ran the
iiwa and the screw arm. So copy the manifests in by hand, as for R13-R16, and check:

```bash
scp $WT/cluster/manifest_stageSVGD_{R17,R17S,R18}.txt tcohn@txe1-login.mit.edu:~/learned-ik/repo/cluster/
ssh tcohn@txe1-login.mit.edu 'cat ~/learned-ik/repo/.staged-commit'          # must print 7ca45a4...
ssh tcohn@txe1-login.mit.edu 'grep -c "\"svgd_paired_init\": (\"jitter\", \"native\")\|\"svgd_kernel\": (\"q\", \"none\", \"x\", \"z\", \"c\", \"cz\", \"qz\", \"pose\")" ~/learned-ik/repo/src/generic_program.py'   # must print 2
```

### Submit

Submit from the worktree `$WT`, as for R13-R16: `submit_bench.sh` reads the manifest from the
checkout it lives in. The order is R17, R17S, R18, in FIFO with no dependency. There is no smoke:
SMOKE_IIWA and SMOKE_SCREW already ran both arms on these scenes with the clearance row, and R12 /
R15 ran the learned arm.

```bash
cd $WT
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R17.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R17S.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R18.txt 4
```

### Read

```bash
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows iiwa --variants kq,kcz,kczrho1e5          # R17 against R12 and PAPER
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows screw7_p050 --variants kq,kcz,kczrho1e5   # R17S against R15 and PAPER
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows panda --variants kcz,kczrho1e5,kczrho1e5n128,kczrho1e5initnative,n128,initnative
```

The reporter takes the new rounds from `gen_manifest.py` with no edit. The A/B table's header says
each column differs from `kq` in one setting. For the declared variants that is not so: read
`kczrho1e5` against `kcz`, and R18's against `kczrho1e5`, in the variant-vs-variant table. Promote
as for R12 / R15: `cp -r <STAGING>/results/iiwa/benchmark/sc_SVGD_R17_*`,
`<STAGING>/results/screw_arm/benchmark/sc_SVGD_R17S_*` and
`<STAGING>/results/panda/benchmark/sc_SVGD_R18_*`.

**Not exercised before the cluster.** The `cz` kernel has run only on the Panda (R13 / R14). On the
iiwa (latent width 8) and the screw arm it runs for the first time in R17 / R17S. A refusal at
solver construction would fail every cell of the item. `_abort_on_dead_arm` then fails the job
within its first three cells, so read the first collected shard of each before trusting the round.

## R19 and R17P (queued 2026-10-12, the cluster still idle until the maintenance)

After R13 / R14 / R17 / R17S / R18 the candidate fielded pin is `kczrho1e5`: `svgd_kernel=cz` plus
`svgd_rho=100000`, a declared two-factor variant. Two measurements of it are missing, and the
cluster is idle until the 10-12 evening maintenance, so both are queued now (Thomas: *"we don't
want to waste GPU time"*). The conventions are R17 / R17S's exactly:

- `SVGD_PINNED_V1` plus the variant's overrides, the record flags, graphed mode, the lifted step cap.
- Learned arm only, 24 shards per 480-cell run, PROCS=2 with no MPS.
- **Every svgd item carries the clearance row**, so both manifests go in with `DRAKE_ENV=crc`.
- `kczrho1e5` is declared two-factor in both rounds exactly as in R14 / R17 / R17S, and the selftest
  holds the token to one setting in every round that runs it.

### R19 (Panda, grid seed 2): the seed replication of the candidate

`SVGD_R19` runs on R7's context, the Panda `n6` at **`--seed 2`**: `kczrho1e5` on the four rows.
It is read against R7's `kq` (the primary on the same seed-2 grids) and twinned with
**SVGD_R7IPOPT**, already collected, so there is no new twin manifest. The selftest holds R19, R7 and
R7IPOPT to one second-seed context.

Tags are `sc_SVGD_R19_panda_n6_svgd_<row>_480_180_<start>_kczrho1e5`. The reporter prints it as the
column **`kczrho1e5@seed2`** beside R14's seed-1 `kczrho1e5`. It pairs only with `ipopt@seed2`, and
`metadata.seed` is in both the provenance check and every pairing's must-match list, so it is
**refused** against R2's seed-1 twin and never pairs with a seed-1 run. The A/B against the seed-1
primary skips it with a note line. The seed-2 block of the variant-vs-variant table pairs it with
`kq@seed2` (R7) on the same cells, which is the kernel-and-rho effect on the second grid.

### R17P (soft PCS arm): the kernel space on the fourth robot

`SVGD_R17P` runs on R16's context, the soft PCS arm `soft12` at `n6`, with R17 / R17S's two
variants on its four rows:

| variant | factors | read against |
| --- | --- | --- |
| `kcz` | `svgd_kernel=cz` (one factor, R13's) | R16's `kq` |
| `kczrho1e5` | `svgd_kernel=cz` + `svgd_rho=100000` (declared two, exactly R14's entry) | `kcz` for rho, `kq` only as the combination |

Tags are `sc_SVGD_R17P_soft12_n6_svgd_<row>_480_180_<start>_<variant>`. The IPOPT twins are the ones
R16 already pairs with, stage PAPER's soft12 runs, collected.

**This round takes hours, not the allotment's 2 h.** This robot's svgd step is the EAGER path
(`SoftArmIKProgram.TorchConfigToPlantQ` under autograd, no analytic row Jacobians): about 800 ms per
outer step against 82 ms on the rigid arms. R16 measured a mean wall of 10-26 s per cell, so the
allotment's assumed 5 s is far too low here. At R16's walls an item is 20 cells x 10-26 s + ~180 s
overhead, 0.11-0.19 h, and the round is about 10-19 node-h, **about 2.5-4.7 h on 4 idle nodes**.
It is queued after R19 and may be cut by the maintenance window (Mon 2026-10-12 evening to Wed 10-14
morning). If so, `--reclaim` it after the window and resubmit.

**Not exercised before the cluster: the `cz` kernel on the soft PCS arm.** It has run on the Panda
(R13 / R14 / R18), the iiwa (R17) and the screw arm (R17S), never on this robot's eager step or its
12-wide latent. A refusal at solver construction would fail every cell of an item, and
`_abort_on_dead_arm` then fails the job within its first three cells. Read the first collected shard
before trusting the round.

### Allotment

`python cluster/gen_manifest.py --stage SVGD_R19 --allotment` prints these lines (the base is the
assumed 5 s per learned cell):

```
  manifest          runs  cells  items  solve-h  node-h    h@4  max item h
  SVGD_R19             4   1920     96      7.5     3.7    0.9        0.08
  SVGD_R17P            8   3840    192     14.9     7.5    1.9        0.08
```

R19's line should hold: R14's `kczrho1e5` ran 1.1-2.7 s mean per cell on the Panda. R17P's should
not (see above). In the worst case every cell is at the clock, each item 20 cells x 180 s = 1 h:
96 + 192 = 288 node-h at PROCS=2 over 4 nodes, about 36 h.

### Staging: copy the two manifests, no restage

The staged tree is still **7ca45a4**, and these manifests need no restage. Since 7ca45a4 the code
has changed only in `scripts/report_svgd.py`, which runs locally (`git diff 7ca45a4 <this commit> --
src/ scripts/ cluster/*.sh`). And 7ca45a4 already carries every option these manifests set:
`svgd_kernel` with `cz` in `SVGD_ENUM_FIELDS`, `svgd_rho`, and
`svgd_collision_backend=clearances`. R14 ran `kczrho1e5` there, R7 the seed-2 Panda grids and R16
the soft PCS arm. So copy the manifests in by hand, as for R17, and check:

```bash
scp $WT/cluster/manifest_stageSVGD_{R19,R17P}.txt tcohn@txe1-login.mit.edu:~/learned-ik/repo/cluster/
ssh tcohn@txe1-login.mit.edu 'cat ~/learned-ik/repo/.staged-commit'          # must print 7ca45a4...
ssh tcohn@txe1-login.mit.edu 'grep -c "\"svgd_kernel\": (\"q\", \"none\", \"x\", \"z\", \"c\", \"cz\", \"qz\", \"pose\")" ~/learned-ik/repo/src/generic_program.py'   # must print 1
```

### Submit

Submit from the worktree `$WT`, as for R17: `submit_bench.sh` reads the manifest from the checkout it
lives in. The order is R19, then R17P, in FIFO with no dependency, so the short Panda round lands
before the long soft PCS one. There is no smoke: R7 ran the seed-2 Panda grids and SMOKE_SOFT / R16
ran this scene with the clearance row.

```bash
cd $WT
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R19.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R17P.txt 4
```

### Read

```bash
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows panda --variants kq,kczrho1e5      # R19 (kczrho1e5@seed2) beside R14 and R7
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows soft12 --variants kq,kcz,kczrho1e5 # R17P against R16 and PAPER
```

The reporter takes both rounds from `gen_manifest.py` with no edit to its pairing. Its variant-name
fields now widen to the longest column, so `kczrho1e5@seed2` and R18's `kczrho1e5initnative` no
longer run into the round token. Promote as for R7 / R16:
`cp -r <STAGING>/results/panda/benchmark/sc_SVGD_R19_*` and
`<STAGING>/results/soft12/benchmark/sc_SVGD_R17P_*`.

## R22 (branch `svgd-tune`, 2026-10-14): stripping at the pin

**Submitted 2026-10-14 14:12 EDT: jobs 5918035-5918038, four nodes, `DRAKE_ENV=crc PROCS=2`, the
staged 7812715 (no restage; manifest scp'd, md5 `0874898b`), after the local smokes of `clipoff`
(3e38) and `T0p1` solved 2 / 2 each.**

The first round of the tuning push (`docs/svgd-solver.md`, "The tuning push"). Stage `SVGD_R22`
is R20's shape -- `SVGD_PINNED_V2`, the four record contexts, 480 cells at 180 s on seed 1, graphed,
the lifted step cap, the clearance row, 24 shards, PROCS=2 with no MPS -- with one override per
learned variant and one declared two-factor joint-space set:

| token | override | arm / rows |
| --- | --- | --- |
| `clipoff` | `svgd_multiplier_max=3e38` | learned, all 16 rows |
| `drop2` | `svgd_resample=drop` | learned, all 16 rows |
| `T10v2` | `svgd_temperature=10` | learned, all 16 rows |
| `T0p1` | `svgd_temperature=0.1` | learned, all 16 rows (claimed last of the learned items: the one clock risk) |
| `clipoffjs` | `svgd_kernel=q` + `svgd_multiplier_max=3e38` | joint space, Panda `n6` only, its own items after every learned item |

**The tokens are new on purpose**: `drop` (R11) and `T10` (R5) exist on the V1 pin, and the selftest
holds a token to one (pin, overrides) across every round. `3e38` rather than `inf`: the benchmark's
`--set` parser is `literal_eval`, which does not parse `inf`, so the value would arrive as a string
and `torch.clamp` would raise on every cell; `1e300` overflows the float32 scalar cast inside `clamp` (the local smoke errored on every cell); `clamp` at +-3e38 is a no-op and the value is float32-representable.

**No restage.** The staged tree is `7812715` (the analytic campaign's), whose `src/svgd/`,
`src/generic_program.py`, `src/benchmark.py` and `cluster/*` are byte-identical to main's and to
`svgd-tune`'s at this round; every option R22 sets exists there (`svgd_multiplier_max`,
`svgd_resample` with `drop`, `svgd_temperature`). So the manifest is copied by hand and checked:

```bash
WT=~/Documents/programming/work/rlg/analytic-and-optimization-ik/learned-ik/.claude/worktrees/svgd-tune
scp $WT/cluster/manifest_stageSVGD_R22.txt tcohn@txe1-login.mit.edu:~/learned-ik/repo/cluster/
md5sum $WT/cluster/manifest_stageSVGD_R22.txt; ssh tcohn@txe1-login.mit.edu 'md5sum ~/learned-ik/repo/cluster/manifest_stageSVGD_R22.txt; cat ~/learned-ik/repo/.staged-commit; grep -c "svgd_multiplier_max\|\"svgd_resample\": (\"native\", \"drop\", \"keep\")\|svgd_temperature" ~/learned-ik/repo/src/generic_program.py'
```

**Local smoke before submitting** (the two values never run before, `svgd_multiplier_max=3e38` and
`svgd_temperature=0.1`): 2 cells x 20 s, Panda pose paired, learned arm, one process under
`systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`; read `n_multiplier_clipped == 0`
on `clipoff` and the override echoed in `metadata.overrides`. Not a result.

**Submit** (all four nodes are this push's, Thomas 2026-10-14):

```bash
cd $WT
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R22.txt 4
```

**Allotment**: `python cluster/gen_manifest.py --stage SVGD_R22 --allotment` (5 s per learned cell,
180 s per item; the joint-space items at R21's measured 1.5 / 0.9 s). R20 measured 6 node-hours for
480 items against 25 allotted, so expect about a third of the tool's figure.

**Read**:

```bash
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows panda --variants pin2,pin2kq,clipoff,clipoffjs,drop2,T10v2,T0p1
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows iiwa --variants pin2,clipoff,drop2,T10v2,T0p1        # and screw7_p050, soft12
```

Promote with `cp -r <STAGING>/results/<robot>/benchmark/sc_SVGD_R22_* results/<robot>/benchmark/`
(`grep -v _shard`). The decision rule per variant is in the pre-registration; a strip the rule
licenses is removed in its own commit after Thomas's word.

## R21 (submitted 2026-10-12 16:18 EDT): the joint-space ablation at the pin on the other three robots

Thomas, after R20's Panda ablation came in 480 / 480: *"Run the joint-space svgd ablation on the other
three robots."* Stage `SVGD_R21` = R20's `pin2kq` (the pin with `svgd_kernel=q`, `--arms numerical`,
its own items) on the iiwa `n4`, `screw7_p050` `n6` and `soft12` `n6` contexts: 12 runs / 288 items,
`manifest_stageSVGD_R21.txt`, no restage (the staged 86efbb0 runs it; the manifest was scp'd and
md5-checked), jobs 5900901 / 5900902 / 5900909 / 5900910, `DRAKE_ENV=crc PROCS=2`, four nodes.
Allotment 8.2 node-h at the Panda's measured 1.5 s per grasp cell and 0.9 s per pose cell
(`SVGD_R21_JS_CELL_S`); worst case every cell at the clock. Same token in two rounds: the selftest
holds `pin2kq` to one (pin, overrides) across R20 and R21, and the reporter reads it as each robot's
own column (`--variants pin2,pin2kq` on every robot). Maintenance may cut it: `--reclaim` after.

## R20 (ran 2026-10-12 13:02-14:52 EDT, before the maintenance)

**Done.** Restaged 86efbb0, smoke job 5897146 then R20 jobs 5897148-51 `afterok`, collected into staging
`20261012-145309`, promoted, reported (`docs/svgd-solver.md`, "R20"). **6 node-hours against the 25
allotted** below: the joint-space ablation converged in 1.5 s a cell, not the 60 s assumed. The
section below is the checklist as written before the run; it was followed as is.

**The record rows at the fielded pin.** Thomas, 2026-10-12: *"Agree re 1 and 2. Native draw is the
method and paired is the ablation."* The pin is `SVGD_PINNED_V2`: kernel `cz`, rho 1e5, the H / T
metric, the native-draw paired swarm, gn_lm 10, lr 0.3, all six NAMED on every item. R20 is the first
v2 stage, and with its smoke the only one; R1-R19 keep `SVGD_PINNED_V1` and regenerate
byte-identical. Joint-space svgd is an ablation, not a baseline. The conventions are the rounds':

- 480 cells, 180 s, seed 1, the record flags, graphed mode, the lifted step cap.
- 24 shards per 480-cell run, PROCS=2 with no MPS, one solve per GPU.
- **Every svgd item carries the clearance row**, so both manifests go in with `DRAKE_ENV=crc`.

| manifest | variant | arm | robots | settings | runs / items |
| --- | --- | --- | --- | --- | --- |
| `manifest_stageSVGD_R20.txt` | `pin2` | learned (`--arms learned`) | Panda `n6`, iiwa `n4`, `screw7_p050` `n6`, `soft12` `n6` | the pin, zero overrides | 16 / 384 |
| | `pin2kq` | joint space (`--arms numerical`) | Panda `n6` only | the pin + `svgd_kernel=q` | 4 / 96 |
| `manifest_stageSVGD_SMOKE_R20.txt` | `pin2`, `pin2kq` | each its own | Panda `n6` | as R20, 2 x 2 cells at 60 s | 8 / 8 |

- **One stage, four robots.** `SVGD_STAGE_CONTEXTS` holds R20's four contexts, each exactly the
  context its single-robot rounds ran. So each row pairs with the IPOPT twin those did: **SVGD_R2**
  for the Panda, **stage PAPER** for the iiwa, the screw arm and the soft PCS arm. All are collected,
  so there is no new twin manifest.
- **The two arms are separate items.** A joint-space grasp cell can run to the 180 s clock, and in a
  shared item it would hold the learned shard. The joint-space items queue after every learned item
  (`order` 2 / 3 against 0 / 1), so the LEAD rows land first.
- **`pin2kq` names its kernel.** `cz` is learned-only and is refused on joint space at warm-up, and
  `q` is joint space's own kernel.
- **New tokens, on purpose.** `kq` meant zero overrides on the V1 pin. The selftest holds a token to
  the same overrides on the same pin in every round, so neither `kq` nor `kczrho1e5` could name these
  rows. `SVGD_DECLARED_FACTORS` declares `pin2` at zero factors and `pin2kq` at one (the kernel).
- **Tags.** The learned rows are `sc_SVGD_R20_<robot>_<rung>_svgd_<row>_480_180_<start>_pin2`, the
  ablation `sc_SVGD_R20_panda_n6_svgd_<row>_480_180_<start>_pin2kq`, and the smoke
  `sc_SVGDSMOKE_panda_n6_svgd_<row>_4_60_<start>_<pin2|pin2kq>`.
- **This is the single-start table's source.** Native is the primary row, paired its ablation.

**Why a smoke.** Joint space has never run under svgd at rho 1e5, the H / T metric or the native
init, nor on the clearance row: R1 / R1K ran it on the V1 pin and the autodiff row. A dead
joint-space column would otherwise surface only in the 480-cell rows. The smoke runs both arms at the
R20 settings, as SEPARATE items, on the Panda's four rows.

### Allotment

`python cluster/gen_manifest.py --stage SVGD_R20 --allotment` prints these lines:

```
  manifest          runs  cells  items  solve-h  node-h    h@4  max item h
  SVGD_R20            20   9600    480     49.9    25.0    6.2        0.38
  SVGD_SMOKE_R20       8     32      8      0.9     0.5    0.1        0.12
  SVGD_R20 WORST CASE (every cell at the clock, one arm per item): 504.0 solve-h, 252.0 node-h,
  63.0 h on 4 idle nodes; longest item 1.05 h.
```

Each item is charged 20 cells x the mean wall per cell, plus the usual 180 s of item overhead. That is
480 cells per run, 2 processes per node, on 4 nodes.

- **The learned arm, MEASURED (`SVGD_R20_MEAN_WALL`).** The base is `kczrho1e5`'s mean wall per cell
  (the pin's two method factors, clearance row, PROCS=2) from R19 (Panda), R17 (iiwa), R17S (screw
  arm) and R17P (soft PCS arm). That is 1.0-2.6 s on the rigid arms and 7.5-10.1 s on the soft PCS
  arm. Per robot this gives 2.9 / 2.8 / 2.9 / 4.7 node-h (Panda / iiwa / screw / soft PCS), or 13.2
  node-h together, **about 3.3 h on 4 idle nodes**. The 180 s overhead is most of a rigid-arm item.
- **The soft PCS rows run on the EAGER forward step**, about 800 ms per outer step against ~80 ms on
  the rigid arms. The fast step pending on branch `svgd-pcs` would cut that. It is not in this
  manifest's code: R20 runs whatever the restaged tree carries.
- **The joint-space ablation is ASSUMED (`SVGD_R20_JS_CELL_S`)**: 60 s per grasp cell and 10 s per
  pose cell. On R1's autodiff row 143 of 480 grasp cells per protocol ran to the clock. That gives
  11.7 node-h, **about 2.9 h on 4 nodes**, after the learned items.
- **Together: 25.0 node-h, about 6.2 h on 4 idle nodes.** The worst case puts every cell at the clock
  and costs 252 node-h, about 63 h. Every item stays inside `ITEM_TIMEOUT` even then.
- **The smoke takes at most ~30 min on one node** (0.5 node-h): 8 items of 4 cells, charged with
  every cell at the 60 s clock, plus process start and warm-up.

### Staging: RESTAGE first

**R20 needs a code restage.** The solver changed after 7ca45a4 (the staged tree): b4666d7 fielded the
pin's defaults and **the gate-edge rule** in `src/svgd/solver.py`, with `n_recheck_failures` and
`feas_level` in the records. The difference is not a refusal. 7ca45a4 already accepts every option
these manifests set, since R14 / R18 ran `cz`, rho 1e5, H / T and the native init there. So an
unrestaged tree would run R20 **silently on the old solver**, without the gate-edge rule. Commit, then
stage from the clone as in "Stage the code from the svgd tip". `stage_code.sh` refuses while campaign
jobs are queued or running, so restage once the queue is idle. The maintenance window empties it.
Then check the staged tree:

```bash
git -C $WT status --porcelain --untracked-files=no   # must be empty
python $WT/cluster/gen_manifest.py --selftest        # must end "selftest OK"
bash $WT/cluster/make_staging_clone.sh "$STAGE"      # refreshes the clone to the svgd tip
git -C $STAGE log -1 --oneline                       # must be the commit you mean to stage
bash $STAGE/cluster/stage_code.sh                    # refuses while a job of this tree runs
ssh tcohn@txe1-login.mit.edu 'cat ~/learned-ik/repo/.staged-commit'          # must print that commit
ssh tcohn@txe1-login.mit.edu 'grep -c n_recheck_failures ~/learned-ik/repo/src/svgd/result.py'   # must be >= 1
```

The four checkpoints are already on `~/learned-ik`, because R12 / R17 (iiwa `n4`), R15 / R17S (screw
arm) and R16 / R17P (soft PCS arm) ran from there. `stage_code.sh` protects
`models/*/*__step*.pkl` from `--delete`, and `submit_bench.sh` refuses a manifest whose checkpoint
is missing on the cluster.

### Submit

Submit from `$STAGE`, as after any restage. The smoke goes first and R20 `afterok` on it:

```bash
cd $STAGE
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_SMOKE_R20.txt 1   # -> <SMOKE_R20>
DEPENDENCY=afterok:<SMOKE_R20> DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageSVGD_R20.txt 4
```

As with every smoke, **`afterok` gates on the smoke JOB's exit status, not on its numbers.** A crash
fails the job. So does a refusal at solver construction (a joint-space kernel the arm cannot take,
say): it fails every cell of the item, and `_abort_on_dead_arm` then fails the job within its first
three cells. A smoke that runs and solves nothing still releases R20. To gate R20 on the
numbers too, hold it (`scontrol hold`), read the smoke, then `scontrol release`. Read the smoke's
eight `summary.json` files:

- zero `error` records, on the `pin2kq` (joint-space) items above all;
- `metadata.overrides` naming `svgd_kernel='q'` on the `pin2kq` items and `'cz'` on the `pin2` ones;
- `metadata.svgd_warmup_seconds` present;
- `n_recheck_failures` and `feas_level` in the svgd records, which shows the restaged solver ran;
- solver / Drake / `verify()` feasibility agreeing;
- wall <= cap + 1 s.

### Read

```bash
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows panda --variants pin2,pin2kq   # LEAD + the joint-space ablation
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows iiwa --variants pin2           # and screw7_p050, soft12
```

The reporter takes R20 from `gen_manifest.py`. `pin2` is a column of each robot's block, paired with
that robot's twin. `pin2kq` is a Panda column: its LEAD line reads `not run (joint space arm only)`,
and its ABLATION line pairs it with the twin's joint-space arm. The setting diff against a robot's
`kq` names the four moved settings (kernel, rho, metric, paired init), plus the clearance row against
the Panda's R1. Promote as for the earlier rounds:
`cp -r <STAGING>/results/<robot>/benchmark/sc_SVGD_R20_* results/<robot>/benchmark/` for `panda`,
`iiwa`, `screw7_p050` and `soft12`, and
`<STAGING>/results/panda/benchmark/sc_SVGDSMOKE_panda_n6_svgd_*_pin2*` for the smoke.

## Collect, promote and report R12, SMOKE_IIWA and R7

These follow the Collect section below:

```bash
bash $WT/cluster/collect_results.sh --status
bash $WT/cluster/collect_results.sh                  # rsync + merge shards; prints the staging dir
# review, then promote each merged run (the collector prints this line with its staging dir):
cp -r <STAGING>/results/iiwa/benchmark/sc_SVGD_R12_iiwa_n4_svgd_* results/iiwa/benchmark/
cp -r <STAGING>/results/iiwa/benchmark/sc_SVGDSMOKE_iiwa_n4_svgd_* results/iiwa/benchmark/
cp -r <STAGING>/results/panda/benchmark/sc_SVGD_R7_panda_n6_* results/panda/benchmark/
# the report reads staged runs too, so it can be run before promoting:
.venv/bin/python scripts/report_svgd.py --stage SVGD --rows iiwa              # the iiwa block only
.venv/bin/python scripts/report_svgd.py --stage SVGD --variants kq            # kq, kq@seed2 and the iiwa
bash $WT/cluster/collect_results.sh --reclaim manifest_stageSVGD_R12          # (and _SMOKE_IIWA, _R7IPOPT, _R7) only once the queue is idle
```

The report exits 0 with runs missing, 2 on a refused pairing and 3 on a BUG line. The smoke is not
read by the stage report. Read its `summary.json` files directly (the checklist above).

## Collect

Run collection from the svgd worktree `$WT`. It has the `.venv` the merger needs and `results/`, and
`collect_results.sh` merges into the checkout it runs from. Use the same `SC_ROOT` as the staging:

```bash
bash $WT/cluster/collect_results.sh --status
bash $WT/cluster/collect_results.sh                  # rsync + merge shards, incremental
bash $WT/cluster/collect_results.sh --reclaim manifest_stageSVGD_R1   # (and _R1K, PAPER, _R3-_R19, _R17S, _R17P, _R20, the smokes) only once the queue is idle
```

Tags: `sc_SVGDSMOKE_panda_n6_svgd_posetip_4_60_paired_kq`,
`sc_SVGD_R2_panda_n6_ipopt_<row>_480_180_<start>` and
`sc_SVGD_R1_panda_n6_svgd_<row>_480_180_<start>_<kq|knone>`; R12, SMOKE_IIWA, R7, R13-R19, R17S, R17P,
R20, SMOKE_R20 and the screw / soft PCS smokes are listed in their own sections above.

- R2 pairs cell for cell with the record's `sc_REMEASURE_panda_n6_ipopt_*`: same grid, same scene,
  same args.
- R1 pairs cell for cell with R2. The record ran the same args at PROCS=8 under MPS, so for seconds
  compare R1 against R2, never against the record.

## Not verified locally (no GPU, no cluster)

- That the svgd path compiles and captures on sm_70 / cu126. That is the smoke's job.
- That `merge_shard_summaries.py` merges sharded svgd runs. This is its first sharded svgd input.
  It re-runs the branch's own `summarise`, which knows the svgd records.
- `verify_sharding.sh` was not run, since it solves. Nothing about sharding or grid construction
  changed: R1 and R2 use the standard target-major `--shard`.
