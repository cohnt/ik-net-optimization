# Mobile-base runbook (branch `mobile-base`)

Design and rulings: `docs/notes-mobile-base.md`. Cluster ops: `cluster/CSAIL_RUNBOOK.md` (load the
`csail-cluster` skill first, every session). Each submission is cleared through learned-ik-orchestrator
(cluster, partition, GPU type, count, prefix `mb_`, duration), and Thomas OKs the pilot.

## Items

`python cluster/csail/gen_mobile_manifest.py {smoke,pilot} -o cluster/csail/manifest_mb_<stage>.txt`
(committed outputs; regenerate, never hand-edit).
- Each item is a record item with the scene swapped. IPOPT is stage REMEASURE's run (both arms in one
  process). AL-SVGD is the pin, R29 `pin4` / R30 `pin4kq`, with the kernel `bcz` (learned) / `bq` (joint
  space) in separate runs on the same grid.
- `--scene mobile --base-start {uniform,facing,home}`.
- 180 s, lifted budgets, `--compile --set flow_cuda_graph=True`, native starts, seed 1.

| stage | items | cells | what |
| --- | --- | --- | --- |
| `mb_SMOKE_` | 9 | 36 | plumbing: Panda pose, screw grasp, soft PCS pose; facing; 2x2, 60 s |
| `mb_PILOT_` | 72 | 6,144 | 4 robots x grasp/pose x 3 rules x {ipopt, svgdL, svgdJ}; 16 x 4 |

## Hardware (one kind for the whole campaign)

- `GPU=rtx3090`: on CSAIL, the `torralba-3090-{1,2,3}` nodes. `EXCLUDE_NODES` the `improbablex001-004`
  nodes, so the CPU model is one, EPYC 7773X (learned-ik-orchestrator, 2026-11-26).
- Read `logs/<jobid>/machine.a<N>.txt` before comparing any wall clock.
- These seconds never sit beside the V100 record.

## Blockers (2026-11-26)

- Resolved: the iiwa `n4` chart is in the worktree (sha256 a55e9c39..., copied from SuperCloud; Tommy
  OK'd this one pickle as-is, 2026-11-26). A 1x1 mobile iiwa item ran on it. Never substitute `n6`.
- Resolved: placement is CSAIL RTX 3090 under the lab account, Tommy's choice for obstacles' OBST.
- **Open: Thomas's go on the pilot.** Then send learned-ik-orchestrator the ask: smoke 9 items, then
  pilot 72 items, `MAX_RUNNING` 8.

## Local checks done before the smoke

- The fast tier passes on the merge with main 65cc48f.
- `tests/test_mobile_svgd.py` pins the `clearances` backend against the autodiff constraint on the
  mobile scenes, with the platform included.
- Two smoke items ran on CPU at 1x1 with the GPU flags dropped, so the driver flags are verified: screw
  grasp svgdL and soft PCS pose svgdJ, both feasible.

## Reading the pilot

`python scripts/report_mobile.py --prefix mb_PILOT_` gives success, McNemar against the proposed,
cost on shared cells, wall, time-matched, stops and start reach per robot x task x rule. Read the cap
check before calling any loss. The verdict "harder" is defined in `docs/notes-mobile-base.md`.
