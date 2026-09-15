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

## Results so far — the complete 480-cell ladder (stages 1+2, collected 2026-09-14)

Promoted into `results/`, which is gitignored, so these numbers live only here until the
campaign ends and CLAUDE.md gets its table. **Stage 3 (`sc_TRAJ`) is still running.**

Harness checks pass: joint space identical across every rung of a robot AND identical to
the archived 480-cell columns (iiwa 462 grasp / 325 pose, Panda 457 / 228); grid hashes
match; `median_start_q_error` 0.0 exactly under `paired`.

| experiment | `ddpr1` | `n8` | `n6` | `n4` | `n12w256` | **iiwa js** |
| --- | --- | --- | --- | --- | --- | --- |
| grasp native | 267 | 309 | 288 | **448** | 349 | 462 |
| grasp paired | 301 | 327 | 302 | **449** | 344 | 462 |
| pose native | 432 | 426 | 441 | **463** | 422 | 325 |
| pose paired | 307 | 221 | 232 | **448** | 337 | 325 |

| experiment | `upstream` | `n12` | `n8` | `n6` | `n4` | `n12w256` | **Panda js** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grasp native | 443 | 430 | 465 | 471 | **476** | 459 | 457 |
| grasp paired | 418 | 416 | 467 | 471 | **474** | 447 | 457 |
| pose native | **474** | 463 | 461 | 462 | 459 | 451 | 228 |
| pose paired | 340 | 339 | 419 | **435** | 407 | 332 | 228 |

Three things 480 cells show that 60 did not:

1. **iiwa `n4` does not reach parity on grasp.** JS 15/29 (p = 0.05) native and 15/28
   (p = 0.07) paired — close, and a transformation of `ddp_r1`'s 9/204, but the exact
   parity at 60 cells was the small grid saturating joint space at 60/60.
2. **The Panda's full-depth charts significantly LOSE the grasp task**, `upstream`
   JS 21/35 (p = 0.08) and `n12` JS 20/47 (**p = 0.001**) native, JS 18/57 and 22/63
   (p = 7e-06, 1e-05) paired — while every reduced-depth rung wins it. At 60 cells these
   rows were 59/60 vs 54/60 and not significant.
3. **The iiwa runaway on `n8`/`n6` pose paired replicates at scale**: `median_max_violation`
   2e+03 and 1e+03, scoring 221 and 232, *below* `ddp_r1`'s 307. Only `n4` (1e-08) and
   `n12w256` (2e-08) are clean.

Per-iteration cost falls monotonically with depth on both robots (iiwa grasp native
80 / 55 / 42 / 33 ms for n12 / n8 / n6 / n4; `n12w256` 72, holding depth), and timeouts
collapse with it (iiwa `n4` 33/31/6/10 against `ddp_r1`'s 212/194/42/170). `n4`'s
per-iteration penalty against joint space is now ~13x, from ~30x.

The optimum rung differs by robot — `n4` on the iiwa, `n6` on the Panda (435 vs `n4`'s 407
on pose paired) — reproducing the triage ordering at 8x the cells.

## Do not prune the training tree until stage 3 is done

The cluster tree is **186 G**, most of it `~/learned-ik/results/train/` (310 `.ckpt`, plus
30 `.pkl` + sidecar per rung) and ~64 G of pre-fix `collect_*.tar` in `~/learned-ik/`.
The tars are disposable and are Thomas's call. **The `pkl/` directories are not** — stage 3
benchmarks `../results/train/iiwa14_n{4,6}/pkl/*.pkl` directly, so deleting them mid-chain
silently kills 384 items. Clear the tars if space is needed; leave `results/train/*/pkl/`
alone until stage 3 has been collected.

## If a stage died

`collect_results.sh --reclaim <manifest>` clears claims with no done marker, then
resubmitting the same manifest picks up exactly the unfinished items. It refuses while any
job is active, which is correct — an item still running elsewhere must not be stolen.
