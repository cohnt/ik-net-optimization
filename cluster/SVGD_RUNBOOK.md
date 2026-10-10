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
  the manifests carry them (`SVGD_PINNED`), and the selftest fails if any svgd item lacks them.
- **Nothing else.** `svgd_n`, `svgd_paired_init` and every other svgd option stay at their defaults.
  The selftest reads `svgd_n = 64` and `svgd_paired_init = "jitter"` out of
  `src/generic_program.py`'s text, and fails if either default moves.

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
and native).

**Shared settings.** All items use the record flags and PROCS=2 with no MPS. They are graphed and
carry the lifted step cap.

**Pinned values.** Each item carries `svgd_rho=1000 svgd_gn_lm=10 svgd_lr=0.3` (`SVGD_PINNED`), plus
**one** `--set` per variant. An override of a pinned name **replaces** that value in place, so every
item names each option exactly once. The selftest holds this, along with:

- the counts;
- each item's method settings being exactly the pinned values plus its one override;
- every item being its IPOPT row with only the documented swap;
- the allowed `--set` list, extended by `svgd_n` and `svgd_temperature`.

The tags are `sc_SVGD_R<k>_panda_n6_svgd_<row>_480_180_<start>_<variant>`.

| manifest | variant | the one change against R1's `kq` | shards | runs | items | node-h (assumed) | max item h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `manifest_stageSVGD_R3.txt` | `n1` / `n256` | `svgd_n=1` / `svgd_n=256` | 8 / 24 | 8 | 128 | 37.9 | 1.13 |
| `manifest_stageSVGD_R4.txt` | `rho1e4` / `lr1` | `svgd_rho=10000` / `svgd_lr=1.0` (replacing 1000 / 0.3) | 24 / 24 | 8 | 192 | 39.5 | 0.41 |
| `manifest_stageSVGD_R5.txt` | `n16` / `T10` | `svgd_n=16` / `svgd_temperature=10` | 16 / 24 | 8 | 160 | 38.7 | 0.59 |

That totals about 116 node-hours, or about 29 h on 4 idle nodes, after R1 and R1K.

**The cost estimate is an assumption, not a measurement.** It charges 5 s for the learned arm plus
60 s for joint space per cell. The cluster smoke only covered pose paired, on 4 cells at 60 s, where
learned took 4-7 s and joint space 9-42 s. Grasp rows, `n256` and `T10` may cost more per cell.
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

## Collect

Run collection from the svgd worktree `$WT`. It has the `.venv` the merger needs and `results/`, and
`collect_results.sh` merges into the checkout it runs from. Use the same `SC_ROOT` as the staging:

```bash
bash $WT/cluster/collect_results.sh --status
bash $WT/cluster/collect_results.sh                  # rsync + merge shards, incremental
bash $WT/cluster/collect_results.sh --reclaim manifest_stageSVGD_R1   # (and _R1K, PAPER, _R3-_R5) only once the queue is idle
```

Tags: `sc_SVGDSMOKE_panda_n6_svgd_posetip_4_60_paired_kq`,
`sc_SVGD_R2_panda_n6_ipopt_<row>_480_180_<start>` and
`sc_SVGD_R1_panda_n6_svgd_<row>_480_180_<start>_<kq|knone>`.

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
