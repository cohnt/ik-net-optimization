# Stage REMEASURE: the record re-measured on the fixed scene

The operational half. The design is the REMEASURE block of `cluster/gen_manifest.py`; the scene
fix and the settings it re-measures are in `CLAUDE.md` ("The hardened scene and shelf-contained
targets"). The cluster playbook is `cluster/README.md`.

## What it is

The PI ruled (2026-10-08) that the whole record must be re-measured: every grasp target on the
iiwa, soft PCS, screw and GVS arms started with the mug handle 18 mm inside a wsg finger (yaw
fix 452d784), and five cross-robot settings were unified at the same time (0381807). The stage is
the record's own rows, generated FROM the status-quo builders -- STATUSQUO (Panda `n6`, iiwa
`n4`), SOFT12, SCREW (`screw7_p050` `n6`), GVS (`o1` `n6`, IPOPT and SNOPT only) -- so rungs,
checkpoints, seed 1, 60 x 8 contained cells and the 180 s clock are the record's by construction.
Every item adds `--set flow_cuda_graph=True` and the lifted iteration budgets (IPOPT `max_iter`
1e6; SNOPT 1e5 majors, 1e8 minors). The clock is never raised. **NLopt was split out** (Thomas,
2026-10-08: *"Skip remeasuring NLopt, it's way too slow"*, then: queue it at the very end, to run
only if the nodes would otherwise idle): its rows run to the 180 s clock on both arms and were ~85
of the primary's node-hours, so they are their own manifest, `REMEASURE_NLOPT`, queued last.

## Four manifests, in this order

| manifest | what | logical runs | items | est. node-h |
| --- | --- | --- | --- | --- |
| `manifest_stageREMEASURE.txt` | the primary, IPOPT and SNOPT, `--config latent` | 40 | 320 | 17 |
| `manifest_stageREMEASURE_LEGACY.txt` | IPOPT, all five robots, `--set legacy_robot_settings=True`: settings vs scene | 20 | 160 | 5.3 |
| `manifest_stageREMEASURE_RULE.txt` | IPOPT, Panda and iiwa, `--config latent_rule`: the trust-region A/B | 8 | 64 | 1.4 |
| `manifest_stageREMEASURE_NLOPT.txt` | the primary's NLopt rows (four record robots), same tag family; queued LAST on a dependency, run only if the nodes would otherwise idle, killable unrun | 16 | 384 | 85 |

Within the primary, items are claimed wsg grasp (IPOPT, then SNOPT), Panda grasp, then pose. The
estimate is `python cluster/gen_manifest.py --allotment` (about 24 node-hours, a quarter of a day
on 4 idle nodes, at PROCS=8 under MPS); its assumptions are
printed with it. Regenerate a manifest with `--stage REMEASURE[_LEGACY|_RULE] -o
cluster/manifest_stage<...>.txt`; `--selftest` fails if a committed manifest is stale.

**Maintenance: compute is down Mon 2026-10-12 evening to Wed 10-14 morning.** Running jobs are
killed and queued jobs do not survive. Submit by Sat 10-10 or after the window; if the window
lands mid-stage, `--reclaim` each manifest and resubmit afterwards.

## Submit (PROCS=8 under MPS, not PROCS=2)

The cluster tree `~/learned-ik` must be staged from a checkout of branch **`mug-handle-yaw`** --
every item's `scene_fingerprint` depends on the fixed SDF, and the reporter refuses a mixture.

```bash
git rev-parse --abbrev-ref HEAD                  # must print mug-handle-yaw; tree clean
python cluster/gen_manifest.py --selftest
cluster/stage_code.sh                            # refuses while a job of this tree runs
ssh ... 'cat ~/learned-ik/repo/.staged-commit'   # must be this checkout's HEAD
PROCS=8 MPS=1 bash cluster/submit_bench.sh manifest_stageREMEASURE.txt 4
PROCS=8 MPS=1 bash cluster/submit_bench.sh manifest_stageREMEASURE_LEGACY.txt 2
PROCS=8 MPS=1 bash cluster/submit_bench.sh manifest_stageREMEASURE_RULE.txt 1
# NLopt last, gated on every job above (Thomas: run only if there is nothing else; kill if he
# wakes early). A dependency defers the checkpoint check to job start.
DEPENDENCY=afterany:<primary>:<legacy>:<rule job ids> PROCS=8 MPS=1 bash cluster/submit_bench.sh manifest_stageREMEASURE_NLOPT.txt 4
```

The control jobs queue behind the primary's four under the 4-node cap and start as it drains.
`submit_bench.sh` refuses if a manifest is not staged or names a checkpoint absent from the
cluster (five charts: `panda__n6`, `iiwa14__n4`, `soft12__n6`, `screw7_p050__n6`,
`gvs_pushrod9_o1__n6`, all `step620000`). `MPS=1` makes `run_items.sh` start a job-local MPS
daemon; **never 4 solves per GPU without it** (CLAUDE.md, Profiling). This is development
throughput, not the paper's one-solve-per-GPU condition, and the report says so.

## Collect and report

```bash
cluster/collect_results.sh --status
cluster/collect_results.sh                       # rsync + merge shards, incremental
cluster/collect_results.sh --reclaim manifest_stageREMEASURE   # only once the queue is idle
python scripts/report_remeasure.py               # all sections; `ipopt` etc. to filter
python scripts/report_remeasure.py --dry-record  # the tables on the record, no new runs needed
```

The merger refuses shards that disagree on `scene_fingerprint`. The reporter never pairs cells
across the scene change: record vs re-measurement is within-run verdicts and unpaired
target-level bootstraps.

## Acceptance checks (section 4 of the report)

1. **One fingerprint per scene.** Every REMEASURE run carries `metadata["scene_fingerprint"]`, and
   all runs on one robot x task scene (all three manifests, both protocols, every solver) share
   it. The reporter refuses to print anything otherwise.
2. **Joint space is bit-identical between protocols** on every converged cell (verdict,
   iterations, cost, q), cap-bound cells excluded and counted. Its native start is a random
   configuration, so it is its paired start.
3. **The Panda reproduces the record under the legacy settings.** Its gripper SDF did not change,
   so `REMEASURE_LEGACY` Panda rows are paired cell for cell against the record (same grid_hash),
   and every discordant cell must be cap-bound in one of the two runs. CUDA graphs may add
   learned cells (+0-11 per IPOPT row in stage CUDAGRAPH) and never removed one.
4. **The cap check on every row**: `timed_out` and `hit_iteration_cap` both printed; >= 24 of 480
   at an iteration budget on either arm voids the verdict until re-measured.

## Status

All four manifests are collected, merged and promoted to `results/<robot>/benchmark/`.
**REMEASURE_NLOPT** (jobs 5868202/03/04/07, manifest 9b78144, code staged at 93120c6, PROCS=8
under `MPS=1`) was **collected 2026-10-10 01:14 EDT and promoted**: 16 runs x 480 cells x 2 arms,
`sc_REMEASURE_<robot>_<rung>_nlopt_<row>_480_180_<start>`. Every run shares its scene's
fingerprint with the IPOPT/SNOPT runs; `hit_eval_cap` and `hit_iteration_cap` are 0 everywhere.
`scripts/report_statusquo.py` reads them as the record's NLopt rows (no old-record fallback);
`scripts/report_remeasure.py nlopt` prints their before/after: learned 9 / tie 7 / joint space 0
-> 11 / 3 / 2, eight verdicts flipped: the three wsg grasp native rows and iiwa grasp paired to
learned, soft PCS and screw grasp paired to joint space, Panda and screw pose paired to ties.

**Paper conditions.** Stage **PAPER** (28 runs, jobs 5882397-5882400) and stage **SVGD_R2** (the
four Panda IPOPT runs) re-ran the primary's 32 IPOPT and SNOPT runs on 2026-10-10 at PROCS=2 with
no MPS (one solve per V100), from the svgd-branch staging commit 42a5893. That commit is main's record
code plus the svgd branch, and its IPOPT and SNOPT paths are byte-identical to main's. R2 was
collected at 02:31 EDT and PAPER at 10:07 EDT, and both were promoted as
`sc_PAPER_<suffix of sc_REMEASURE_>` and `sc_SVGD_R2_panda_n6_ipopt_<row>_480_180_<start>`. Every
pair shares its REMEASURE twin's scene fingerprint and grid hash. `scripts/report_paper.py` prints
the pairing, and `scripts/report_statusquo.py --paper` prints the record at those seconds.
