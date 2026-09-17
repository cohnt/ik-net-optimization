# Live run: stage SNOPTTUNE

**Delete this file when the campaign closes and its conclusion is in CLAUDE.md.** It exists so a
paused session can be resumed by any session — including one with no memory of launching it.

## What is running

Stage SNOPTTUNE, submitted 2026-09-17 from branch `step-rejection` at commit `e2a3642`.
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
bash -c 'source cluster/ssh_common.sh; sc_run "ls ~/learned-ik/state/*.done 2>/dev/null | wc -l"'   # of 1248

# 2. Sleep inhibitor (the laptop suspends after 15 idle min even on AC, which has
#    silently frozen every unattended run this repo ever recorded).
pgrep -x systemd-inhibit || setsid systemd-inhibit --what=sleep:idle --mode=block \
  --who=learned-ik --why="SNOPTTUNE" sleep 57600 </dev/null >/dev/null 2>&1 &

# 3. Collect (incremental by default; never ships state/)
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
