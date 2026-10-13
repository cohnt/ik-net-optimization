# Stages ANALYTIC: the analytic + optimization arm on the Panda and the iiwa

The operational half. The design is the ANALYTIC block of `cluster/gen_manifest.py`; the
formulation is `src/analytic_program.py`. **Nothing here is submitted until Thomas says go**
(2026-10-12), even once maintenance ends.

## What it is

The analytic arm (`analytic` and `analytic8` on the Panda, `analytic` on the iiwa) on the
record's own Panda `n6` and iiwa `n4` rows: each item is the record's REMEASURE item for the same
row with ONLY `--arms`, the tag and (tight ladder) two `--set`s changed, so `grid_hash` and
`scene_fingerprint` equal the record's and `scripts/report_analytic.py` pairs it cell for cell
against the record's learned and joint-space arms. An upper-bound reference column, not a
competitor.

Svgd normalisation (`src/svgd/solver.py:region_scale`, chosen 2026-10-13, Thomas no preference):
`c_position_slack` on position, pi on rpy and on psi. A/B against psi's bound half-width (5 / 10)
on 192 smoke cells: identical success on every row.

**Partitions (Thomas, 2026-10-13).** The analytic arm never touches the GPU except under
AL-SVGD, so ANALYTIC_SMOKE, ANALYTIC and ANALYTIC_NLOPT run on the CPU partition `xeon-p8`
(48 cores, no GPU; a group cap of 8 nodes shared with `~/ik-tune`, so check `LLstat` and ask the
ik-tune session before sizing). Their items drop the record's learned-only `--compile` and
`flow_cuda_graph=True`, and `submit_bench.sh` refuses those switches, `--solver svgd` and learned
arms on xeon-p8. Only the two SVGD manifests go to `xeon-g6-volta`. So the analytic arm's
IPOPT / SNOPT / NLopt seconds come from a different machine from the record's, and
`report_analytic.py` prints a NOTE on every such row: compare its seconds within its own runs only.

Two clip ladders, both measured (Thomas, 2026-10-12): `ANALYTIC` is ../codebase's 1e-4 / 1e-3,
`ANALYTICT` the tight 1e-8 / 1.1e-4.

| manifest | what | conditions | runs | items | est. node-h |
| --- | --- | --- | --- | --- | --- |
| `manifest_stageANALYTIC_SMOKE.txt` | 2 x 2 cells at 60 s, every robot x task x protocol x solver x ladder | xeon-p8, PROCS=2 | 48 | 48 | <= 6 |
| `manifest_stageANALYTIC.txt` | IPOPT and SNOPT, both ladders | xeon-p8, PROCS=2 (paper) | 32 | 256 | 26 |
| `manifest_stageANALYTIC_NLOPT.txt` | NLopt, both ladders (tags `ANALYTICN`, `ANALYTICNT`) | xeon-p8, PROCS=8 (as the record's NLopt; no MPS, no GPU) | 16 | 384 | <= 74 |
| `manifest_stageANALYTIC_SVGD_SMOKE.txt` | 2 x 2 cells at 60 s, every svgd run | PROCS=2, `DRAKE_ENV=crc` | 32 | 32 | <= 2 |
| `manifest_stageANALYTIC_SVGD.txt` | AL-SVGD at R20's pin, kernels `x` (`pin2kx`) and `q` (`pin2kq`), both ladders (`ANALYTICS`, `ANALYTICST`) | PROCS=2, `DRAKE_ENV=crc` | 32 | 768 | 161 |

Estimates (`python cluster/gen_manifest.py --stage ANALYTIC --allotment`) are MEASURED per row,
2026-10-13: IPOPT / SNOPT from 12 cells of each record grid on the laptop at the record's lifted
budgets (laptop and cluster seconds agree within 0.95-1.13x on joint space), svgd from the 8-cell
local smokes; a cell that reached the local clock is charged the full 180 s. NLopt is charged the
clock on every cell, an upper bound. ANALYTIC_SVGD is dominated by Panda `analytic8`, whose failing
swarms run to the clock. Worst item: ~6 h (ANALYTIC shard, every cell at the clock), under the 8 h
item cap.

The C++ backend (`--set analytic_backend=cpp`) is NOT in these manifests; if Thomas fields it,
run `bash cpp/build.sh` in the cluster tree after `stage_code.sh` (one g++ call; a missing library
raises at program construction, before the first cell) and regenerate the manifests with the set.

## Submit (only after the go-ahead)

```bash
git rev-parse --abbrev-ref HEAD                  # analytic-baseline; tree clean
python cluster/gen_manifest.py --selftest
cluster/stage_code.sh
## CPU nodes (xeon-p8): the smoke, then IPOPT / SNOPT, then NLopt
PARTITION=xeon-p8 PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC_SMOKE.txt 1      # -> <SMOKE>
DEPENDENCY=afterok:<SMOKE> PARTITION=xeon-p8 PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC.txt 4
DEPENDENCY=afterany:<ANALYTIC jobs> PARTITION=xeon-p8 PROCS=8 bash cluster/submit_bench.sh manifest_stageANALYTIC_NLOPT.txt 4
## GPU nodes (xeon-g6-volta): AL-SVGD only
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC_SVGD_SMOKE.txt 1   # -> <SSMOKE>
DEPENDENCY=afterok:<SSMOKE> DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC_SVGD.txt 4
```

Collect with `cluster/collect_results.sh` and read with `python scripts/report_analytic.py`.
