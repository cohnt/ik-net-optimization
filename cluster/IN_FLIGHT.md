# In flight on SuperCloud — 2026-09-14

Disposable handoff note. **Delete it once these runs are collected and written up**; the
findings belong in `CLAUDE.md`, not here.

Nothing about these jobs depends on a Claude session. Slurm owns them, they are chained
with `--dependency=afterany`, and they keep running with nobody attached. But **nothing
pulls results down automatically** — collection is a manual step (below).

## What is queued

Three stages, each 4 jobs, each waiting on all four of the stage before it. All on
`xeon-g6-volta` (all 4 nodes of the group cap), `PROCS=8`, 45 s cap, `--compile`,
`correction_cost_weight=10`, seed 1, 480 cells (60 targets x 8 guesses), arms
`learned,numerical`, 12 shards per run.

| stage | jobs | manifest | tag prefix | runs | items |
| --- | --- | --- | --- | --- | --- |
| 1 | 5620624–27 | `manifest_stageLADDER480.txt` | `sc_LADDER_` | `n4`, `n6` @620k, both robots | 192 |
| 2 | 5620718–21 | `manifest_stageLADDER480b.txt` | `sc_LADDER_` | the other 7 rungs @620k | 336 |
| 3 | 5620723–26 | `manifest_stageTRAJ.txt` | `sc_TRAJ_` | iiwa `n4`/`n6` across training steps | 384 |

Estimated 7–9 h of wall clock for all three, from the 60-cell triage's own per-cell wall
times. The manifests' own item-hour figures are the all-cells-hit-the-cap worst case.

Stages 1 and 2 together are the complete 480-cell ladder. Stage 3 is the training-step
sweep (see `stage_TRAJ`'s header in `cluster/gen_manifest.py` for what it tests and why).

## Resuming: what to run

```bash
# 1. where things stand -- one read-only ssh, no filesystem walk
bash cluster/collect_results.sh --status
ssh <sc> "squeue -u \$USER -h -o '%.10i %.12T %.10M %R'"     # or via sc_run

# 2. pull down whatever has landed (incremental; merges shards as it goes)
bash cluster/collect_results.sh

# 3. promote the merged summaries out of staging into the tracked tree
cp -r results/_cluster_staging/<stamp>/results/<robot>/benchmark/<tag> results/<robot>/benchmark/

# 4. read them
python scripts/collate.py 'results/*/benchmark/sc_LADDER_*/summary.json'
python scripts/collate.py --pair learned 'results/iiwa/benchmark/sc_TRAJ_iiwa_n6*/summary.json'
```

Collection is safe to run while later stages are still going — it is incremental, and a
stage that has not finished simply has no merged summary yet.

## What to check before believing any of it

- **The numerical arm must be identical across every run of a robot+task+start.** It does
  not touch the flow. If it moves, the grid or the harness drifted, not the chart.
- **`median_start_q_error` must be 0.0000 under `paired`.**
- **Grid hashes must match the archived 480-cell columns** — `bbd92baaa967-{mug,pose}` for
  the iiwa, `c5f34cedcd81` / `2f84b6d1d64f` for the Panda — or these are not
  cell-comparable with the headline table and `collate.py --pair` will refuse.
- **An arm failing identically in ~10 ms a cell is not solving at all.** `_abort_on_dead_arm`
  should catch it, but check `fail_reason` is a named gate rather than `"error"`.

## If a stage died

`collect_results.sh --reclaim <manifest>` clears claims with no done marker, then
resubmitting the same manifest picks up exactly the unfinished items. It refuses while any
job is active, which is correct — an item still running elsewhere must not be stolen.
