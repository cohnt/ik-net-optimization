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

Two clip ladders, both measured (Thomas, 2026-10-12): `ANALYTIC` is ../codebase's 1e-4 / 1e-3,
`ANALYTICT` the tight 1e-8 / 1.1e-4.

| manifest | what | conditions | runs | items | est. node-h |
| --- | --- | --- | --- | --- | --- |
| `manifest_stageANALYTIC_SMOKE.txt` | 2 x 2 cells at 60 s, every robot x task x protocol x solver x ladder | PROCS=2 | 48 | 48 | <= 6 |
| `manifest_stageANALYTIC.txt` | IPOPT and SNOPT, both ladders | PROCS=2, no MPS (paper) | 32 | 256 | 41 |
| `manifest_stageANALYTIC_NLOPT.txt` | NLopt, both ladders (tags `ANALYTICN`, `ANALYTICNT`) | PROCS=8, `MPS=1` (as the record's NLopt) | 16 | 384 | 74 |
| `manifest_stageANALYTIC_SVGD_SMOKE.txt` | 2 x 2 cells at 60 s, every svgd run | PROCS=2, `DRAKE_ENV=crc` | 32 | 32 | <= 2 |
| `manifest_stageANALYTIC_SVGD.txt` | AL-SVGD at R20's pin, kernels `x` (`pin2kx`) and `q` (`pin2kq`), both ladders (`ANALYTICS`, `ANALYTICST`) | PROCS=2, `DRAKE_ENV=crc` | 32 | 768 | 35 |

Estimates:
`python cluster/gen_manifest.py --stage ANALYTIC --allotment` (2x joint space's mean wall per
cell per analytic arm, capped at the clock).

## Submit (only after the go-ahead)

```bash
git rev-parse --abbrev-ref HEAD                  # analytic-baseline; tree clean
python cluster/gen_manifest.py --selftest
cluster/stage_code.sh
PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC_SMOKE.txt 1      # -> <SMOKE>
DEPENDENCY=afterok:<SMOKE> PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC_SVGD_SMOKE.txt 1   # -> <SSMOKE>
DEPENDENCY=afterok:<SSMOKE> DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageANALYTIC_SVGD.txt 4
DEPENDENCY=afterany:<ANALYTIC and ANALYTIC_SVGD jobs> PROCS=8 MPS=1 bash cluster/submit_bench.sh manifest_stageANALYTIC_NLOPT.txt 4
```

Collect with `cluster/collect_results.sh` and read with `python scripts/report_analytic.py`.
