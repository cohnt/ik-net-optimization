# In flight: the hardened problem (stage HARD), 2026-09-15

## What is running

| manifest | jobs | items | state |
| --- | --- | --- | --- |
| `manifest_stageHARDTRI.txt` | 5625360-61 | 24 | **done, collected, promoted** |
| `manifest_stageHARD_iiwa.txt` | 5625541-44 | 240 | running |
| `manifest_stageHARD_panda.txt` | 5625548-51 | 288 | pending, `afterany` the iiwa four |

45 s cap, seed 1, 60 x 8 cells, `--compile`, `correction_cost_weight=10`, `PROCS=8`,
8 shards per run, `learned,numerical`. Eleven rungs x (grasp x 2 starts + pose x 2
placements x 2 starts) = 66 logical runs.

**The allocation is shared.** `../codebase` had `run_matrix.sh` jobs on the same account
when this was submitted; the `xeon-g6-volta` `node=4` cap is a GrpTRES *group* cap across
everything Thomas runs, so surplus queues rather than being rejected.

## When it lands

```bash
bash cluster/collect_results.sh --status
bash cluster/collect_results.sh --reclaim manifest_stageHARD_iiwa    # queue idle only
bash cluster/submit_bench.sh manifest_stageHARD_iiwa.txt 2           # one mop-up pass
bash cluster/collect_results.sh
cp -r results/_cluster_staging/<stamp>/results/<robot>/benchmark/sc_HARD_* results/<robot>/benchmark/
python scripts/collate.py 'results/*/benchmark/sc_HARD_*/summary.json'
```

## Sanity checks before believing anything

- `median_start_q_error` exactly 0 under `paired` (held in all 12 HARDTRI runs).
- The numerical arm identical across every run sharing robot + task + start + **placement**.
- Grid hashes differ from the archived 480-cell columns *by design* -- the hardened scene
  admits a different target set. `collate.py` refuses the pairing, which is correct, not a
  bug. Compare HARD columns against each other, never against `sc_LADDER_*`.
- An arm failing identically in ~10 ms a cell is not solving at all.

## What HARDTRI already showed (24 items, 20 s cap, 60 cells, two control rungs)

The hardening bites and **the grasp baseline is no longer saturated**: Panda grasp joint
space fell from 457/480 (95%) archived to 36/60 (60%), which was the whole point -- there
was almost no room left to win cells.

| run (20 s) | learned | joint space |
| --- | --- | --- |
| panda grasp native | **46/60** | 36/60 |
| panda grasp paired | 27/60 | **36/60** |
| panda pose native, contained | **57/60** | 24/60 |
| panda pose native, free | **57/60** | 30/60 |
| iiwa pose native, contained | **54/60** | 38/60 |
| iiwa pose native, free | **53/60** | 42/60 |

First signal on the open pose question: **containment costs joint space ~6 cells on both
robots and costs the learned arm nothing.** 20 s is not the campaign's cap and two rungs are
not eleven, so this is a lead, not the answer.
