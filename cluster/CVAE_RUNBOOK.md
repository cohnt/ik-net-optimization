# The cVAE chart ablation: the run's own record

What is submitted, in what order, what it depends on, and what a resuming session checks first.
The design is the plan (branch `chart-cvae`): the record's learned formulation with the IKFlow
chart replaced by a standard conditional VAE's decoder, everything else unchanged. Findings go to
`CLAUDE.md` / `docs/notes-chart.md`; tables to `docs/chart-ablation-tables.md`; this file is
procedure only. `cluster/README.md` and `~/.claude/skills/supercloud/SKILL.md` carry the standing
rules, which override anything here.

**If you are even remotely unsure about a cluster action, stop and ask Thomas.**

## STATE AS OF 2026-10-19 15:00 -- SECOND ROUND QUEUED

**What the night did.** Seven cVAE runs trained and exported. `iiwa14_cvae8` hit a NaN loss at
step 143472 (job 5955589 FAILED). Because `submit_bench.sh` checks every checkpoint in a manifest
when a chained job starts, the missing one made all eight stage-CVAE / CVAE_SVGD jobs refuse
(exit 3, 0 s), so **no cVAE cell was measured**. The chain continued: `screw7_p050_n6r` trained
(15.4 h, 5955608), SCREWREDO + SCREWREDO_SVGD done, SCREWREDO_NLOPT running.

**Screens of the trained charts** (`results/screen_cvae/`): rigid arms unusable as charts --
median FK error at prior samples Panda 609-624 mm / ~105 deg, iiwa K=1 253 mm / 118 deg, screw
746 mm / 130 deg with 0 active latent dims; soft12 21-22 mm / 4 deg. Paired-start representability
0% on every chart (encoder round-trip residual median 0.75-2.8 rad against the 0.1 box), so every
paired cell scores `unrepresentable_start`. No runaway (max |q| 8.6). The learned per-joint
decoder variance absorbed Panda joints 1-3 as noise (sigma ~0.55) -- my design choice, hence:

**Thomas (2026-10-19): both** -- benchmark the trained charts as they are AND train the
textbook fixed-variance variant (`cvaef<K>`, MSE + KL) and benchmark it.

| step | jobs | depends on |
| --- | --- | --- |
| `iiwa14_cvae8` resumed from step 140000 | 5966176 | -- |
| cvaef training, `ONLY='_cvaef[0-9]+$' PAIRS=1 --all` | 5966292-95 | -- |
| CVAE without iiwa14 cvae8 (232), PROCS=2 | 5966296-99 | -- |
| CVAE_SVGD without iiwa14 cvae8 (672), crc, PROCS=2 | 5966300-03 | -- |
| CVAEF (256), PROCS=2 | 5966304-07 | afterany: 5966292-95 |
| CVAEF_SVGD (768), crc, PROCS=2 | 5966308-11 | afterany: 5966292-95 |
| CVAE / CVAE_SVGD iiwa14 cvae8 items only (32 / 96), 2 jobs each | 5966315-16, 5966317-18 (crc) | afterok: 5966176 |

**Outcomes since (2026-10-19 16:00).** `iiwa14_cvae8` diverged AGAIN on resume (inf loss at step
142636; 5966176 FAILED): the learned-variance objective is reproducibly unstable there. Thomas:
**record it as diverged** -- no chart, no benchmark for that rung; its four `_iiwa8` bench jobs
(5966315-18) were cancelled. The fixed-variance `iiwa14_cvaef8` covers K = 8.
**`screw7_p050_n6r` is bit-identical to `n6`** (all 112 tensors equal; the .pkl bytes differ only in
pickling), so the rebuilt dataset reproduced the deleted one exactly; SCREWREDO matches the record
cell for cell on IPOPT/SNOPT (svgd within its run-to-run noise). The record's screw flow numbers
stand; SCREWREDO is a confirmation, not a replacement.

Restaged twice under live jobs with `FORCE_STAGE=1` (29106cc, then the iiwa8 manifests): the only
source change was `src/charts/cvae.py` (learned path bit-identical by test), which no running flow
item imports. The `_noiiwa8` / `_iiwa8` manifests are complements of the full ones with the same
ids; never also submit the full manifests (separate state folders -> double runs).

## STATE AS OF 2026-10-18 18:30 -- THE WHOLE CHAIN IS QUEUED (staged commit 9396d99)

Submitted unattended for the night (Thomas travelling; he ruled the screen check-in before the
benchmarks is skipped -- read the screens alongside the results). Smokes were skipped: the plumbing
was checked locally instead (untrained cVAE chart through IPOPT paired/native, AL-SVGD
paired/native on CPU, and the CUDA-graph path on the laptop GPU -- all ran, no dead-arm abort).

| step | jobs | depends on |
| --- | --- | --- |
| screw7_p050 dataset rebuild (xeon-p8) | 5953556, COMPLETED 7.5 min | -- |
| cVAE training, `PAIRS=1 --all 24:00:00` (two runs per 2-GPU job, so the 4-node cap is never in question) | 5955588 panda, 5955589 iiwa14, 5955590 screw7_p050, 5955591 soft12 | -- |
| stage CVAE (IPOPT, 264 items incl. the joint-space acceptance item), PROCS=2 | 5955592-94, 5955601 | afterany: the four training jobs |
| stage CVAE_SVGD (768 items), DRAKE_ENV=crc PROCS=2 | 5955602-05 | afterany: the four training jobs |
| screw flow retrain `screw7_p050_n6r` (4 nodes, chain_ladder) | 5955608 | afterany: all eight CVAE bench jobs |
| SCREWREDO (IPOPT+SNOPT, 64), PROCS=2 | 5955609-12 | afterany: 5955608 |
| SCREWREDO_SVGD (96), DRAKE_ENV=crc PROCS=2 | 5955613-16 | afterany: 5955608 |
| SCREWREDO_NLOPT (96), PROCS=8 MPS=1 | 5955617-20 | afterany: 5955608 |

**Early training read (step 40k of 620k, 18:37).** Mean FK error of `decode(pose, z ~ N(0,I))`
over 500 val poses: Panda 0.64 m / 1.8 rad, iiwa 0.22 m / 2.0 rad, screw 0.74 m / 2.2 rad (both
screw runs fully collapsed, KL 0), soft12 4.5 cm / 0.17 rad. About one active latent dim, ~2 nats
of KL: the decoder behaves like an averaging regressor, the standard Gaussian-decoder cVAE's known
failure on multimodal targets. Thomas: **let it run** to 620k and benchmark what comes out (a poor
chart is itself an answer, and tests whether the optimization rescues it). A recipe change (KL
warm-up, free bits, fixed decoder variance) would be a deviation from "standard cVAE" and is his
call. wandb is offline on the nodes; sync from the laptop (`environment-and-tooling` memory).

**Why the screw flow is retrained** (Thomas, 2026-10-18): the screw dataset was deleted with the
old tree and rebuilt (same generator, seed 0) but cannot be shown identical to the one `n6` trained
on, so the fielded rung is retrained on the rebuilt data (`n6r`, same recipe) and the screw
record's learned rows re-run with it -- flow and cVAE then see the same data. No sweep.

**Resume checks, in order**: `bash cluster/submit_chart_train.sh --status` (all eight at 620000
steps, exported to `models/<robot>/<robot>__cvae<K>__step620000.*`, screens in
`results/screen_cvae/`); `sacct` the bench job ids above (a bench job whose checkpoints were
missing refuses at start and exits -- resubmit it); `bash cluster/submit_ladder.sh --status` for
`screw7_p050_n6r`; then `python scripts/report_chart_ablation.py --write`. `afterany` means a dead
predecessor does not stall the chain: always check, never infer from the queue.

## What runs

| piece | file | what it is |
| --- | --- | --- |
| the eight runs | `cluster/chart_runs.txt` | `<family> <robot> <run_name> <trainer args>`; a sibling of `ladder_runs.txt`, NOT rows in it (`submit_ladder.sh --next` would launch them as 4-node flow rungs) |
| submitter | `cluster/submit_chart_train.sh` | one single-GPU job per run (`--all`), or two runs per 2-GPU job (`--pair`, `PAIRS=1 --all`); `--status`, `--smoke`, `--smoke-pair` |
| payloads | `cluster/train_chart.sh`, `cluster/train_chart_pair.sh` | train on one GPU, then export the step-620000 checkpoint and screen it, inline |
| benchmark | `cluster/gen_manifest.py` stages `CVAE`, `CVAE_SVGD`, `CVAESMOKE`, `CVAESMOKE_SVGD` | committed manifests `cluster/manifest_stage*.txt` |
| report | `scripts/report_chart_ablation.py --write` | `docs/chart-ablation-tables.md` |

### The rungs, both pre-registered, both benchmarked, final checkpoint

| robot (jrl name) | DOF | K = flow width | K = redundancy | decision variables (6 + K + DOF) | trust radius, K-flow / K-red |
| --- | ---: | --- | --- | --- | --- |
| Panda (`panda`) | 7 | `panda_cvae7` | `panda_cvae1` | 20 / 14 | 4.0 (the record's) / 2.5 |
| iiwa (`iiwa14`) | 7 | `iiwa14_cvae8` | `iiwa14_cvae1` | 21 / 14 | 4.3 (the record's) / 2.5 |
| screw (`screw7_p050`) | 7 | `screw7_p050_cvae7` | `screw7_p050_cvae1` | 20 / 14 | 4.15 (the record's) / 2.5 |
| soft PCS (`soft12`) | 12 | `soft12_cvae12` | `soft12_cvae6` | 30 / 24 | 4.96 (the record's) / 3.95 |

**The trust radius is a setting of the rung.** The K = flow-width rung keeps exactly the record's
`--config latent` radius, so its program is the record's but for the chart. The K = redundancy rung
takes the rule `round(sqrt(K) + 1.5, 2)` at its own width, named on each item as
`--set latent_trust_region=<r>` (the record's constants are that rule at the flow's width; the
screw and soft PCS configs compute it from the robot's DOF, which is why it is named rather than
left to `--config`). `gen_manifest.py --selftest` holds every item to it.

Each run exports to `models/<robot>/<robot>__cvae<K>__step620000.{pkl,arch.json}` (the sidecar
carries `"family": "cvae"`), which is the path stage CVAE's items name; the selftest's
`_ladder_paths_match_export` checks both directions against `chart_runs.txt`.

## Order of operations, and why it is this order

```
0. stage the code (BEFORE anything is queued: stage_code.sh refuses while any job of this tree runs)
1. screw dataset rebuild (xeon-p8, ~10 min)  |  training smokes, panda + soft12 (volta, 30 min)
        |                                          + the co-scheduling / GrpTRES measurement
        |                                          + watch the same-run guard refuse once
        v
2. the eight runs (singles, or PAIRS=1 -- decided by step 1's measurement)
        |   export + screen run INSIDE each job on rc=0
        v
3. collect; read the screens; CHECK IN WITH THOMAS        <- no benchmark before his go
        v
4. stage; CVAESMOKE + CVAESMOKE_SVGD (minutes)
        v
5. CVAE (IPOPT + acceptance) then CVAE_SVGD
        v
6. collect, merge, promote; report_chart_ablation.py --write
```

### 0. Stage

```bash
cluster/stage_code.sh
```

It refuses (exit 3) while ANY job submitted from this tree is queued or running -- including a
cVAE training job, since the trainer imports repo code. So stage once, before step 1, with
everything the training half needs (the model, trainer, exporter, screen, both payloads). The
benchmark half is staged again at step 4, after training has drained. The exported `.pkl` files
written into `repo/models/` on the cluster survive restaging (`--filter='P models/*/*__step*.pkl'`).

### 1a. The screw dataset: rebuild, and it is a NEW dataset

The screw arm's datasets were deleted with `~/learned-ik-helix` on 2026-10-02. Check first what
the cache holds (`panda`, `iiwa14`, `soft12` should be there with `.DONE`; `screw7_p050` should not):

```bash
source cluster/ssh_common.sh
sc_run 'for d in ~/learned-ik/home/.cache/ikflow/datasets/*/; do echo "$d $(cat $d/.DONE 2>/dev/null || echo NO-DONE)"; done'
```

Then, from the login node, on the CPU partition, ONE dataset at a time (ikflow's end-of-run summary
scans every dataset in the cache, so a concurrent build's half-written tensor withholds a finished
build's `.DONE`):

```bash
sc_run 'cd ~/learned-ik/repo && DATASET_ROBOT=screw7_p050 LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-p8 -T 02:00:00 -J lik_dataset'
```

25M samples + 15k test, seed 0, `--only_non_self_colliding`, as the originals. **A rebuild is NOT
the original dataset** (Thomas, 2026-10-02; the originals' `.DONE` held no checksum, so equality
cannot be shown). The screw cVAEs therefore train on different samples than the screw FLOW chart
they are compared against (which trained on the deleted one); every other robot's cVAE trains on
its flow's own dataset. State this wherever the screw rows are reported. `train_chart.sh` refuses
(exit 4) without the `.DONE` sentinel.

### 1b. Training smokes, the co-scheduling measurement, and the guard

Two 200-step smokes on robots whose datasets already exist, submitted together -- they are the
plumbing check (train, step-named checkpoint, export, round trip, screen) AND the co-scheduling
test:

```bash
bash cluster/submit_chart_train.sh --smoke panda 1
bash cluster/submit_chart_train.sh --smoke soft12 6
```

Each exports into its own `results/train/smoke_<robot>_cvae<K>/pkl/`, never `models/`.

**Watch the same-run guard refuse, once** (a guard nobody has observed refusing has not been
tested). While the first smoke is queued or running, repeat its command:

```bash
bash cluster/submit_chart_train.sh --smoke panda 1     # MUST print REFUSING and exit 3
```

**Measure co-scheduling and how the cap counts it**, while both smokes run and nothing else of the
account holds volta nodes (check `squeue -u $USER` first; another project's jobs change the count):

```bash
sc_run "squeue -u \$USER -h -o '%i %j %T %N %b'"                   # both on ONE node?
sc_run "scontrol show assoc_mgr users=\$USER flags=assoc | grep -o 'GrpTRES=[^ ]*' | sort -u"
```

`GrpTRES=node=4(N)`: N is the account's nodes in use. Two smokes on one node reading N = 1 means
the cap counts nodes, and eight single jobs fill 4 nodes x 2 GPUs: submit singles. N = 2 (Slurm's
documented behaviour: each job's node allocation is counted separately) means eight single jobs run
four at a time with half the GPUs idle: submit **pairs**. If the two smokes landed on different
nodes, the co-scheduling question is unanswered -- resubmit with one node's other smoke excluded
rather than guess. If `assoc_mgr` is not readable to the account, pairs are the safe choice: a
pair job is one node and both its GPUs whichever way the cap counts. If pairs are chosen, smoke that mode too:

```bash
bash cluster/submit_chart_train.sh --smoke-pair panda 7 1
```

A smoke passes when `results/train/smoke_*/node0.SENTINEL` reads `rc=0` (pair: the pair's
`launch.log` reports both rcs 0), `status.json` reads step 200, the `.pkl` and `.arch.json` exist
under `pkl/`, and `results/screen_cvae/<robot>__smoke_<robot>_cvae<K>__step200.json` exists. Read
the trainer's throughput from the launch log (`train.log-<jobid>` in pair mode) and set the
walltime from it: `620000 / steps-per-second` with a 30% margin, under the partition's 100 h
ceiling. A smoke is resumable like any run, so **delete `results/train/smoke_*` before re-smoking**
-- otherwise the rerun resumes at step 200 and trains nothing.

### 2. The eight runs

When the screw dataset reads `.DONE` and the smokes passed:

```bash
bash cluster/submit_chart_train.sh --all <walltime>              # GrpTRES counts the node once
PAIRS=1 bash cluster/submit_chart_train.sh --all <walltime>      # it counts each job: two per job
```

Pairs are consecutive rows of `chart_runs.txt`, i.e. each robot's two rungs on one node. Any row
already at 620000 steps is skipped. `--all` reports and continues past a refused row; it never
bypasses the guard. Where to look:

```bash
bash cluster/submit_chart_train.sh --status    # every row's step / losses, the exports, this tree's queue
```

**Do not run the flow ladder at the same time.** `submit_train.sh`'s one-rung-at-a-time guard
matches `/_train_/`, which `_trainc_` does not contain, so neither family's guard refuses the
other: a flow rung would start and take the cap from under these runs.

### 3. Export, screen, collect -- then the check-in

On rc=0 each job exports `cvae-checkpoint-step=620000.ckpt` (the STEP-NAMED checkpoint, never
`last.ckpt`) through `scripts/training/export_cvae.py --out`, which round-trips the export through
`LoadCVAEChart`, then runs `scripts/training/screen_cvae.py` into
`~/learned-ik/results/screen_cvae/<robot>__cvae<K>__step620000.json`. Collect with
`cluster/collect_results.sh` (checkpoints and `pkl/` are not shipped; the screens are). The
`.pkl` files stay on the cluster in `repo/models/`; benchmarks read them there.

**Check in with Thomas before any benchmark item is submitted**, with the four screens per run:
(a) FK error of `decode(c, z ~ N(0, I))`; (b) paired-start representability, which predicts the
`unrepresentable_start` rate; (c) active latent units (the flow-width rung's surplus dimensions are
expected to collapse to the prior); (d) the runaway screen (max `|q|`, Jacobian norms over the
latent box) -- the chart's answer to the central finding.

### 4. Stage, then the benchmark smokes

```bash
cluster/stage_code.sh                                            # training drained first
PROCS=2 bash cluster/submit_bench.sh manifest_stageCVAESMOKE.txt 2
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageCVAESMOKE_SVGD.txt 2
```

32 + 32 items, one per logical run, two cells each (`--cells 0:0,31:5` of the record's grid),
`sc_CVAESMOKE_<robot>_cvae<K>_<solver>_<row>_2_180_<start>`. They pass when every item exits 0, no
`_abort_on_dead_arm`, `metadata["chart_arch"]["family"] == "cvae"`, and the paired rows'
`unrepresentable_start` counts are in line with screen (b). `submit_bench.sh` refuses the CVAE
manifests until every `--checkpoint` they name exists on the cluster -- that is expected before
training finishes and is the guard working.

### 5. The stage

```bash
PROCS=2 bash cluster/submit_bench.sh manifest_stageCVAE.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageCVAE_SVGD.txt 4
```

Paper conditions, one solve per V100, no MPS, as the record. **Two manifests because the Drake
environment is per submission**: `CVAE` = the 32 IPOPT runs (8 shards each, 256 items) on the
tarball pin plus the ACCEPTANCE run (8 items, claimed first: joint space on Panda pose paired,
the record's flags, `sc_CVAEACC_panda_n6_ipopt_posetip_480_180_paired`); `CVAE_SVGD` = the 32
AL-SVGD runs at the V4 pin (24 shards each, 768 items) on the clearance row. Estimates
(`gen_manifest.py --stage <S> --summary --procs 2`): CVAE 29 item-hours, ~3.6 h on four idle
nodes; CVAE_SVGD 60 item-hours, ~7.5 h, two thirds of it per-item startup. The IPOPT rates are the
flow record's, so the cVAE's own per-iteration cost moves them. Mop-up: resubmit the same manifest
once the queue drains; `cluster/collect_results.sh --reclaim <manifest>` only when nothing of it
runs.

### 6. Report

```bash
cluster/collect_results.sh                 # merges shards; promote each run as it prints
python scripts/report_chart_ablation.py --write
```

**Read the acceptance section first**: it must print `IDENTICAL` (q, iterations, cost, verdict on
all 480 cells against the record's joint-space arm). If it does not, the refactor moved joint space
and no cVAE column pairs with the record until that is understood. The report refuses (exit 2) any
cVAE run whose `grid_hash`, `scene_fingerprint`, cap or cells differ from its record row, and
prints `not yet measured` for runs not collected. The cap check (clock AND budget, >= 24 budget
cells voids a row) is in its diagnostics table; a VOID row is re-measured with the iteration budget
lifted, never the clock.

## The fixed-variance variant (`cvaef<K>`, Thomas 2026-10-19)

The textbook cVAE objective (`--decoder_variance=fixed`: squared reconstruction error in normalised
units + KL, beta = 1; no `log_sigma_q`), everything else identical. Eight rows at the end of
`cluster/chart_runs.txt`; the learned-variance rows stay and are NOT resubmitted. Submit exactly
the eight with the run-name filter (awk ERE on the run name; a regex matching nothing refuses):

```bash
ONLY='_cvaef[0-9]+$' PAIRS=1 bash cluster/submit_chart_train.sh --all 24:00:00
```

Exports `models/<robot>/<robot>__cvaef<K>__step620000.pkl` (the label is the run name minus
`<robot>_`). Benchmarks: stages `CVAEF` (IPOPT, 256 items) and `CVAEF_SVGD` (AL-SVGD, 768 items,
`DRAKE_ENV=crc`), item for item stage CVAE / CVAE_SVGD with only the chart swapped and no
acceptance item; `scripts/report_chart_ablation.py` prints them as columns `cVAEf-K` / `cVAEf-k`.

**Filtered stage-CVAE manifests** `manifest_stageCVAE_noiiwa8.txt` (232 items) and
`manifest_stageCVAE_SVGD_noiiwa8.txt` (672) are the full manifests minus every `iiwa14__cvae8`
item (`gen_manifest.py --stage CVAE --exclude-rung iiwa14:cvae8`; the selftest holds them to it),
with the full manifests' item ids. **But run_items.sh keys its `.claim` / `.done` state by the
MANIFEST'S BASENAME** (`$ROOT/state/<manifest name>/<id>.done`), so the full manifest does NOT see
items done under the filtered one. Before submitting the full manifest afterwards, carry the done
markers across (ids are identical, so the full manifest then skips them and runs only the iiwa14
K = 8 items):

```bash
for m in manifest_stageCVAE manifest_stageCVAE_SVGD; do
  mkdir -p ~/learned-ik/state/$m
  cp -n ~/learned-ik/state/${m}_noiiwa8/*.done ~/learned-ik/state/$m/
done
```

(`~/learned-ik` is the default tree; use the tree the stage ran in. Copy `.done` only, never `.claim`: a claim without a done marker is a dead item, and copying one
would hide it from the full manifest.) Mind any job already queued on the FULL manifest: it and a
filtered submission use different state directories and would run the same items twice.

## Resume checks

- **Is a run trained?** Only `--status` says so: `step 620000`. A job leaving the queue is not
  evidence. Every exit code of the payload is in `results/train/<run>/node0.SENTINEL` and the
  launch log: 4 = no dataset `.DONE`; 5 = training exited 0 without a step-620000 checkpoint;
  6 = export or its round trip failed; 7 = the screen failed.
- **Walltime kill / node failure**: resubmit the same run (or pair) -- `--ckpt_path=auto` resumes
  from `last.ckpt`. `--all` does it for every unfinished row.
- **Trained but not exported** (rc 6/7, or `RUN_EXPORT=0`): resubmit that run by name with a short
  walltime; it resumes at step 620000, Lightning should return at once (check the log says so),
  and the in-job export runs again.
  `ROBOT=panda bash cluster/submit_chart_train.sh panda_cvae7 01:00:00 -- --dim_latent_space=7`
  (the trailing args are the row's own; `--all` skips finished rows, so it will not do this).
- **A pair where one run died**: resubmit the pair; the finished run resumes, returns at once and
  re-exports, the other continues. Or resubmit the dead run alone -- the guard refuses while the
  pair job still holds either name.
- **Restaging mid-training** is refused by `stage_code.sh`; never `FORCE_STAGE=1` past it -- the
  queued and running jobs re-read the repo's code.
