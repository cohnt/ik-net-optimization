# Live run: stage SNOPTTUNE

**Delete this file when the campaign closes and its conclusion is in CLAUDE.md.** It exists so a
paused session can be resumed by any session — including one with no memory of launching it.

## What is running

Stage SNOPTTUNE, submitted 2026-09-17 12:46 from branch `step-rejection`, staged at commit
`4578996`. Slurm jobs **5667606-5667609**, one per node on `xeon-g6-volta`.
SNOPT's own configuration at campaign scale: **13 settings columns x 12 rows x 480 cells**,
1,248 sharded items, `PROCS=8`, 4 nodes on `xeon-g6-volta`. Allotment ~290 core-hours, ~12 h wall.

Rows are `SOLVER2_ROWS` (grasp-free, grasp-contained, pose-fingertip) x both start protocols x
both adopted rungs (Panda `n6`, iiwa `n4`), seed 1, 45 s cap, hardened scene. Every column runs
every row — uniform coverage is load-bearing and the selftest asserts it.

The manifest is **ordered by column priority**, so the baseline and the four most informative
columns land first and partial results are readable long before the run completes:
`default`, `nonderivls`, `ndlsmstep`, `ndlshess20mstep`, `hessfreq20`, `mstep0p5`, then the rest.

## Why it is running

Every campaign table fields IPOPT with the acceptable-point early stop and SNOPT at bare Drake
defaults — a tuned interior-point method against an untuned SQP. Thomas, 2026-09-17: *"I'm okay
with playing with solver settings on a per-solver basis, as long as it's not per-problem."* So
SNOPT is entitled to one configuration, chosen once and applied to every row.

## The pre-registered decision rule — do not change it after seeing results

13 columns x 12 rows is 156 McNemar tests. A setting is adoptable only if, on the **learned** arm,
it beats `default`:

1. on **at least 9 of the 12 rows**,
2. is **significantly worse (p < 0.05) on none**, and
3. is **significantly better on at least one**.

Counted by row. **Do not pool experiments** (Thomas: *"Do not pool experiments, that is useless"*).
One configuration for all rows; a per-experiment pick is what the per-solver rule forbids.

**Expect no verdict to flip.** Candidates are worth ~+24 of 480 and SNOPT loses iiwa grasp by
50-65 cells. This buys a defensible SNOPT column, not a rescue.

## Resuming

```bash
# 1. Is it still going?
bash -c 'source cluster/ssh_common.sh; sc_run "LLstat | grep -c SNOPTTUNE"'
# NOTE the per-manifest subdirectory -- run_items.sh uses $ROOT/state/$MANIFEST_NAME, and a
# flat ~/learned-ik/state/*.done glob silently reports 0 forever.
D=~/learned-ik/state/manifest_stageSNOPTTUNE
bash -c "source cluster/ssh_common.sh; sc_run 'echo done \$(ls $D/*.done 2>/dev/null | wc -l) of 1248, claims \$(ls -d $D/*.claim 2>/dev/null | wc -l)'"

# 2. Sleep inhibitor (the laptop suspends after 15 idle min even on AC, which has
#    silently frozen every unattended run this repo ever recorded).
pgrep -x systemd-inhibit || setsid systemd-inhibit --what=sleep:idle --mode=block \
  --who=learned-ik --why="SNOPTTUNE" sleep 57600 </dev/null >/dev/null 2>&1 &

# 3. Collect. NOTE results land in results/_cluster_staging/<timestamp>/ and must be
#    PROMOTED into results/<robot>/benchmark/ -- a glob over the promoted location while a
#    run is in flight matches NOTHING, so any check written against it passes vacuously.
#    Read staged output at results/_cluster_staging/*/results/*/benchmark/<tag>/summary.json
#    and promote only complete, merged tags (per-shard dirs are *_shardKofN -- do not promote
#    those, they inflate every later glob).
bash cluster/collect_results.sh

# 4. Read it. report_step.py takes its baseline BY NAME -- `crash0` sorts before `default`.
scripts/report_step.py --ref default 'results/*/benchmark/sc_SNOPTTUNE_iiwa_n4_snopt_mugfree_480_45_paired_*'
```

Known failure modes: an item that dies leaves a `.claim` with no `.done` —
`bash cluster/collect_results.sh --reclaim`. Next SuperCloud maintenance is **2026-10-12 to
10-14**; nothing survives it.

## Acceptance checks before reading any number

- The fresh `default` column must reproduce the archived `sc_SOLVER2_*_snopt_*` counts within the
  cap-bound reproducibility band (at 480 cells with 20-90 timeouts that is up to ~7 net, not +/-1
  — the +/-1 figure in CLAUDE.md was measured on 60-cell grids).
- `median_start_q_error` is 0.0 exactly under `paired`.
- Compare **within this run**, never against the archive: the fresh baseline is what absorbs any
  difference in code version or node contention.

**The baseline check PASSED at full scale, 2026-09-17 14:00.** The `default` column finished
first (96 of 96 shards) and 11 of its 12 rows reproduce the archived `sc_SOLVER2_*_snopt_*`
columns within +/-2 cells on the learned arm: iiwa grasp-free 349/335 against 348/333, iiwa
pose-fingertip 442/210 against 442/210 exactly, iiwa contained-grasp paired 214 against 212,
Panda grasp-free 449/398 against 450/397, Panda contained-grasp 438/281 against 438/280, Panda
pose-fingertip 438/252 against 438/251. **The joint-space arm is EXACT on all eight rows carrying
an archived value (+0 every one)** and identical across protocols on all five checkable pairs.
Zero `fail_reason = "error"` anywhere. So no SNOPT option was rejected, no arm is dead, and this
branch did not perturb the SNOPT path.

The twelfth row (iiwa contained-grasp native) had its shards straddle two collections and reports
"could not be merged" -- that is `collect_results.sh`'s known split-shard case, not a failure; it
merges on a later collection once all eight shards sit under one staging directory.

## Results as they land

**`Nonderivative linesearch` — REFUTED, 2026-09-17 15:20.** The lead candidate, and the only one
with a mechanism tied to the gradients. On 11 of 12 rows it is **5 better, 6 worse, none
significant**, pooled 3806 -> 3774. The pre-registered rule needs >= 9 of 12 better with at least
one significant, so it fails decisively rather than narrowly.

Two things worth keeping from it. Its real mechanism is a **throughput cost**: median iterations
591 -> 761 and wall clock 10.2 -> 20.2 s on the grasp rows, timeouts 22 -> 70, because a
gradient-free line search needs more function evaluations per major and here an evaluation is a
flow Jacobian. And the **churn is enormous** — +95/-94 on iiwa contained-grasp paired, +98/-94 on
Panda contained-grasp paired — the same symmetric reshuffle stage STEP measured, where ~20% of
cells flip each way for a net of one to four.

This is stage STEP repeating: a 60-cell lead (+12/240, p = 0.20) with a plausible mechanism, gone
at 480 cells. **Weaken the prior on the two combinations built on this setting.**

**The two `Nonderivative linesearch` combinations — 2026-09-17 16:40, and the pattern is a ROBOT
TRADE.** `+ Major step limit 0.5` is the best showing so far (total 3806 -> 3850, 8 rows better, 2
worse) but **Panda grasp-free native is significantly WORSE** (449 -> 434, p = 0.036), which trips
clause 2. The three-factor version (6 rows in) does the same thing harder: iiwa contained-grasp
paired 214 -> 247 significantly BETTER (p = 0.025) while Panda grasp-free native 449 -> 428
significantly WORSE (p = 0.0065).

Same two rows, opposite directions, both significant. This is the iiwa/Panda split `Major step
limit = 0.5` showed at 60 cells (iiwa +17, Panda -2), confirmed at scale — and under the
per-solver-not-per-problem rule it is **unadoptable by construction**, not merely unproven. If the
remaining columns keep this shape, that is the stage's result: SNOPT's useful settings are
robot-specific, so SNOPT has no single best configuration on this problem.

## Where it stood when the session closed, 2026-09-17 17:05

**467 of 1248 items done (37%), all four jobs healthy, 499 claims, ETA ~00:20.** Five columns
complete or nearly so: `default`, `nonderivls`, `ndlsmstep`, `ndlshess20mstep` (all 96/96) and
`hessfreq20` at 83/96. Nothing collected since 16:34, so the last three columns' worth of results
are still cluster-side.

**On resuming, in this order:**

1. `LLstat` — if zero jobs and `.done` < 1248, some items died: `bash cluster/collect_results.sh --reclaim`.
2. `bash cluster/collect_results.sh` (several minutes; it merges every complete shard group).
3. Read every column against the baseline. The comparison must be per row — **no pooling across
   experiments** — with the pre-registered rule applied as written above.
4. Promote only complete merged tags into `results/<robot>/benchmark/`; never the `*_shardKofN`
   directories, which inflate later globs.

**The live hypothesis to test against the remaining columns.** The three settings read so far all
move iiwa contained-grasp paired UP and Panda grasp-free native DOWN, both significantly in the
combinations. If the single factors that do NOT involve the nonderivative line search
(`hessfreq20`, `mstep0p5`, `lstol0p99`, `lstol0p1`, `majopt1em08`, `elastic1e2`, `crash0`,
`hessfreq100`) show the same split, the stage's result is that **SNOPT's useful settings are
robot-specific, so SNOPT has no single best configuration on this problem** — which answers the
fairness question that motivated the stage, just not the way it was expected to. If instead the
split is confined to the gradient-free line search and its combinations, then it is a property of
that line search rather than of SNOPT on this problem, and the remaining singles decide whether any
uniform setting exists.

## 2026-09-17 21:50 — `Major step limit = 0.5` leads, and the earlier robot-trade reading was WRONG

Ten columns read at 11 of 12 rows. Rows better / worse, then significantly better / worse:

| setting | better | worse | sig+ | sig- |
| --- | --- | --- | --- | --- |
| **`Major step limit = 0.5`** | **10** | 1 | **2** | **0** |
| `Hessian frequency = 20` | 7 | 4 | 1 | 0 |
| `Major optimality tolerance = 1e-8` | 7 | 3 | 0 | 0 |
| `Nonderivative linesearch` + `Major step limit = 0.5` | 8 | 2 | 0 | 1 |
| `Nonderiv LS` + `Hessian freq 20` + `Major step 0.5` | 6 | 4 | 1 | 2 |
| `Linesearch tolerance = 0.99` | 5 | 5 | 0 | 0 |
| `Nonderivative linesearch` | 5 | 6 | 0 | 0 |
| `Crash option = 0` | 4 | 7 | 0 | 0 |
| `Nonderivative linesearch` + `Hessian frequency = 20` | 3 | 8 | 0 | 1 |

**`Major step limit = 0.5` is on track to PASS** (needs >= 9 of 12 better, 0 significantly worse,
>= 1 significantly better). It gains exactly where SNOPT is weakest — iiwa pose-tip paired
210 -> 247 (p = 0.0020), Panda pose-tip paired 252 -> 275 (p = 0.043), iiwa contained-grasp paired
214 -> 239 (p = 0.071) — and its only losing row is Panda grasp-free native, 449 -> 442, p = 0.14,
not significant. **It also improves the JOINT-SPACE arm on every row** (398->406, 236->270,
404->416, 298->305, 169->178), so it is a property of SNOPT on this problem rather than of the
chart.

**The robot-trade hypothesis recorded at 17:05 is REFUTED, and it was my own overreach:** the split
belongs to `Nonderivative linesearch`, not to the step limit. The line search is net-negative alone
and poisons every combination it enters — `+ Hessian frequency 20` is the stage's worst column at
3 better / 8 worse with a significant loss. Generalising from the two combinations before the
single factor finished produced a conclusion the single factor contradicts. **Do not read a
combination as evidence about its parts.**

Still running: `Linesearch tolerance = 0.1` (3 rows in), `Elastic weight = 100`,
`Hessian frequency = 100`. The twelfth row of every column (iiwa contained-grasp native) is the
straddled-shard group and merges on a later collection.

## 2026-09-17 23:05 — the twelfth row landed, and `Major step limit = 0.5` PASSES

The missing twelfth row was never a run problem: the **baseline** row
`sc_SNOPTTUNE_iiwa_n4_snopt_mugshelf_480_45_native_default` had its eight shards straddle
*four* consecutive collections (staging 132130 / 132731 / 135938 / 150412), and
`collect_results.sh` hands the merger only this staging directory plus the **previous one**
(`--also`), so no single merger invocation ever saw all eight. Every shard was complete on disk
the whole time. Note the shard sizes are **64,64,64,64,56,56,56,56** — 60 targets split
target-major over 8 shards — so a 56-record shard is COMPLETE, not truncated; reading 56 as
partial is what made this look like data loss.

Repaired by hand: the eight complete shard directories were copied into
`results/_cluster_staging/manualmerge-row12/results/iiwa/benchmark/` and
`cluster/merge_shard_summaries.py` run on that directory alone. Merged cleanly, 480 cells x 2 arms.

**Result, pre-registered rule applied unchanged (>= 9 of 12 rows better on the learned arm, none
significantly worse, >= 1 significantly better):**

`Major step limit = 0.5` — **11 of 12 rows better, 3 significantly better, 0 significantly worse
-> PASSES.** Nothing else does. Every other setting is 4-9 rows better and fails on one or both
of the other two conditions.

Still in flight: `Hessian frequency = 100` (27 of 96 items). `Elastic weight = 100` is complete
on the cluster (96/96) and needs only a collection.
