# The LOInK chart family: the run's own record

What is submitted, in what order, what it depends on, and what a resuming session checks first.
The design is the plan (branch `chart-loink`): the record's learned formulation with the IKFlow
chart replaced by a LOInK chart (a bi-Lipschitz network `G(q) = [y_hat; z]` inverted by Davis-Yin
splitting, `src/charts/loink.py`), everything else unchanged. Findings go to `CLAUDE.md` /
`docs/notes-chart.md`; this file is procedure only. `cluster/README.md` and
`~/.claude/skills/supercloud/SKILL.md` carry the standing rules, which override anything here.
The cVAE campaign's runbook (`cluster/CVAE_RUNBOOK.md`) is the precedent this one follows.

**If you are even remotely unsure about a cluster action, stop and ask Thomas.**

## STATE -- NOTHING SUBMITTED YET

Plumbing written (chart_runs.txt rows, train_chart.sh `loink)` case, gen_manifest stages, the
committed LOINKSMOKE / LOINKPROBE manifests, empty LOINKEXT manifests). Phase 0 (the paper
replication, `docs/loink-replication.md`) gates the training; Thomas reads it first.

| step | jobs | depends on | state |
| --- | --- | --- | --- |
| stage_code (training half) | -- | -- | |
| smoke `panda --cost=m --activation=relu --recipe=P` | | -- | |
| smoke `soft12 --cost=u --activation=tanh --recipe=A` | | -- | |
| co-scheduling / GrpTRES measurement (or skip: pairs) | -- | both smokes running | |
| P recipe, `ONLY='_loink[mxu][rt]P$' PAIRS=1 --all` (24 runs, 12 jobs) | | smokes passed | |
| A recipe, `ONLY='_loink[mxu][rt]A$' PAIRS=1 --all` (24 runs, 12 jobs) | | smokes passed | |
| collect; read the screens; **check in with Thomas** | -- | all 48 exported | |
| stage_code (benchmark half) | -- | training drained | |
| LOINKSMOKE (192), PROCS=2 | | | |
| LOINKSMOKE_SVGD (192), crc, PROCS=2 | | | |
| LOINKPROBE (536 incl. acceptance), PROCS=2 | | smokes passed | |
| LOINKPROBE_SVGD (1296), crc, PROCS=2 | | smokes passed | |
| collect probes; `make_loink_ext.py`; commit; stage | -- | probes collected | |
| LOINKEXT (<= 1152), PROCS=2 | | | |
| LOINKEXT_SVGD (<= 3456), crc, PROCS=2 | | | |
| collect, merge, promote; the report | -- | | |

## What runs

| piece | file | what it is |
| --- | --- | --- |
| the 48 runs | `cluster/chart_runs.txt` (the `loink` rows) | `<family> <robot> <run_name> <trainer args>`; each row carries its recipe's `--batch_size` / `--max_steps` |
| submitter | `cluster/submit_chart_train.sh` | one single-GPU job per run (`--all`), or two runs per 2-GPU job (`--pair`, `PAIRS=1 --all`); `--status`, `--smoke loink`, `--smoke-pair loink` |
| payloads | `cluster/train_chart.sh` (`FAMILY=loink`), `cluster/train_chart_pair.sh` | train on one GPU (`scripts/training/train_loink.py`), then export the step-named final checkpoint (`export_loink.py`) and screen it (`screen_loink.py`), inline |
| benchmark | `cluster/gen_manifest.py` stages `LOINKSMOKE(_SVGD)`, `LOINKPROBE(_SVGD)`, `LOINKEXT(_SVGD)` | committed manifests `cluster/manifest_stageLOINK*.txt`; EXT written by `cluster/make_loink_ext.py` |
| report | (written separately; finds runs by the `sc_LOINK_` / `sc_LOINKACC_` tag stems) | |

### The charts: twelve per robot, all pre-registered, final checkpoint

Label `loink<cost><act><recipe>`: cost `m` (matched: the program's own `0.5 |q - q_nominal|^2`; soft
PCS `0.5 |strain|^2`), `x` (mismatched: LOInK's eq. 18 log-barrier), `u` (unstructured: no cost
head); activation `r` (relu) / `t` (tanh); recipe `P` (project: AdamW 1.5e-4, StepLR, 620000 steps,
batch 4096) / `A` (paper: Adam 1e-3, reduce-on-plateau, 150000 steps, batch 500). Same datasets as
the flow charts on every robot (the screw dataset is the rebuild, proved bit-identical: `n6r == n6`).

| robot (jrl name) | DOF | K = DOF - 6 | decision variables (6 + K + DOF) | trust radius (rule) | exports |
| --- | ---: | ---: | ---: | ---: | --- |
| Panda (`panda`) | 7 | 1 | 14 | 2.5 | `models/panda/panda__loink<c><a><R>__step<N>.pkl` |
| iiwa (`iiwa14`) | 7 | 1 | 14 | 2.5 | `models/iiwa14/iiwa14__loink...` |
| screw (`screw7_p050`) | 7 | 1 | 14 | 2.5 | `models/screw7_p050/screw7_p050__loink...` |
| soft PCS (`soft12`) | 12 | 6 | 24 | 3.95 | `models/soft12/soft12__loink...` |

`N` = 620000 (P) or 150000 (A): `train_chart.sh` reads `MAX_STEPS`, `BATCH` and `FINAL_STEP` from
the row's own args (they override the submitter's environment, which is why a pair job can mix
recipes), and the trainer's per-recipe checkpoint interval (P 20000, A 10000) divides both. The
sidecar carries `"family": "loink"`. **The trust region is always the rule**
`round(sqrt(K) + 1.5, 2)`, named on every item as `--set latent_trust_region=<r>`; there is no
flow-width rung, so no item keeps `--config latent`'s robot-sized radius. `dim_latent_space` is not
passed (the chart's). `gen_manifest.py --selftest` holds every item to its record twin with only
arms, checkpoint, tag and that radius changed, and `_ladder_paths_match_export` checks the 48 export
paths against `chart_runs.txt` in both directions.

## Order of operations, and why it is this order

```
0. stage the code (BEFORE anything is queued: stage_code.sh refuses while any job of this tree runs)
1. training smokes, one relu row + one tanh row (volta, 30 min)
        + the co-scheduling / GrpTRES measurement (or go straight to pairs)
        + watch the same-run guard refuse once
        v
2. the 48 runs: the P recipe and the A recipe as two PAIRS=1 submissions, walltimes from step 1
        |   export + screen run INSIDE each job on rc=0
        v
3. collect; read the six screen sections; CHECK IN WITH THOMAS     <- no benchmark before his go
        v
4. stage; LOINKSMOKE + LOINKSMOKE_SVGD (minutes)
        v
5. LOINKPROBE (IPOPT + acceptance) and LOINKPROBE_SVGD
        v
6. collect the probes; make_loink_ext.py; commit; stage; LOINKEXT + LOINKEXT_SVGD
        v
7. collect, merge, promote; the report
```

### 0. Stage

```bash
cluster/stage_code.sh
```

It refuses (exit 3) while ANY job submitted from this tree is queued or running -- including a
training job, since the trainer imports repo code. So stage once, before step 1, with everything the
training half needs (`src/charts/loink.py`, the trainer, exporter, screen, both payloads). The
benchmark half is staged again at step 4, after training has drained. Exported `.pkl` files in
`repo/models/` on the cluster survive restaging (`--filter='P models/*/*__step*.pkl'`). Check the
datasets first (`panda`, `iiwa14`, `screw7_p050`, `soft12` must all read `.DONE`):

```bash
source cluster/ssh_common.sh
sc_run 'for d in ~/learned-ik/home/.cache/ikflow/datasets/*/; do echo "$d $(cat $d/.DONE 2>/dev/null || echo NO-DONE)"; done'
```

### 1. Training smokes, the co-scheduling measurement, and the guard

Two 200-step smokes, one per activation, one per recipe, submitted together -- the plumbing check
(train, step-named checkpoint, export, round trip, screen) AND the co-scheduling test:

```bash
bash cluster/submit_chart_train.sh --smoke loink panda  --cost=m --activation=relu --recipe=P
bash cluster/submit_chart_train.sh --smoke loink soft12 --cost=u --activation=tanh --recipe=A
```

Run names `smoke_panda_loinkmrP`, `smoke_soft12_loinkutA`; each exports into its own
`results/train/smoke_<robot>_<label>/pkl/`, never `models/`. The smoke trains at its recipe's batch
(the submitter adds `--batch_size=4096` / `500`) for 200 steps (it refuses a `--max_steps`).
(`--smoke cvae <robot> <K>` and the old `--smoke <robot> <K>` are the cVAE's, unchanged.)

**Watch the same-run guard refuse, once**, while the first smoke is queued or running:

```bash
bash cluster/submit_chart_train.sh --smoke loink panda --cost=m --activation=relu --recipe=P   # MUST print REFUSING, exit 3
```

**Co-scheduling and the cap** -- the cVAE campaign skipped this measurement and used pairs, which
are one node and both its GPUs however the cap counts. Pairs are the default here too. To measure
anyway, while both smokes run and nothing else of the account holds volta nodes:

```bash
sc_run "squeue -u \$USER -h -o '%i %j %T %N %b'"                   # both on ONE node?
sc_run "scontrol show assoc_mgr users=\$USER flags=assoc | grep -o 'GrpTRES=[^ ]*' | sort -u"
```

`GrpTRES=node=4(N)` with N = 1 for two smokes on one node means singles fill 4 nodes x 2 GPUs;
N = 2 (Slurm's documented behaviour) means pairs. Smoke the pair mode before relying on it:

```bash
bash cluster/submit_chart_train.sh --smoke-pair loink panda \
    '--cost=x --activation=relu --recipe=P' '--cost=x --activation=tanh --recipe=P'
```

A smoke passes when `results/train/smoke_*/node0.SENTINEL` reads `rc=0` (pair: the pair's
`launch.log` reports both rcs 0), `status.json` reads step 200, the `.pkl` and `.arch.json` exist
under `pkl/`, and `results/screen_loink/<robot>__smoke_<robot>_<label>__step200.json` exists.
**Walltime, per recipe**, from the launch log's throughput (`train.log-<jobid>` in pair mode):
`steps / steps-per-second x 1.3`, with steps 620000 (P, read off the panda smoke) and 150000 (A,
read off the soft12 smoke; batch 500 is not batch 4096's rate), under the partition's 100 h ceiling.
The 200-step rate includes compile and warm-up, so it is pessimistic; if the P estimate exceeds the
ceiling, stop and ask. A smoke is resumable like any run, so **delete `results/train/smoke_*` before
re-smoking** -- otherwise the rerun resumes at step 200 and trains nothing.

### 2. The 48 runs

Submit the two recipes SEPARATELY, so each pair job holds two runs of one length (`--all`'s pairing
is over consecutive matching rows; with these filters it pairs relu with tanh at one robot, cost and
recipe -- `chart_runs.txt`'s header):

```bash
ONLY='_loink[mxu][rt]P$' PAIRS=1 bash cluster/submit_chart_train.sh --all <P walltime>   # 12 jobs
ONLY='_loink[mxu][rt]A$' PAIRS=1 bash cluster/submit_chart_train.sh --all <A walltime>   # 12 jobs
```

**Always pass `ONLY=`.** A bare `PAIRS=1 --all` would pair each chart's P run with its A run (valid,
but the A GPU idles for most of the job), and it also walks the cVAE rows: the finished ones are
skipped, but `iiwa14_cvae8` (diverged twice, recorded, never at 620000) would be RESUBMITTED. A row whose
`status.json` already reached its own budget (620000 / 150000) is skipped; a pair never spans two
families. `--all` reports and continues past a refused row; it never bypasses the guard.

```bash
bash cluster/submit_chart_train.sh --status    # every row's step / losses, the exports, this tree's queue
```

**Do not run the flow ladder at the same time** (its guard matches `/_train_/`, which `_trainc_`
does not contain, so neither family's guard refuses the other).

### 3. Export, screen, collect -- then the check-in

On rc=0 each job exports `loink-checkpoint-step=<N>.ckpt` (the STEP-NAMED checkpoint, never
`last.ckpt`) through `export_loink.py --out`, which round-trips through the LOInK loader (`decode`
within 1e-10 and `G(decode(w)) - w` below 1e-8, float64) and writes `n_dys` into the sidecar, then
runs `screen_loink.py` into `~/learned-ik/results/screen_loink/<robot>__<label>__step<N>.json`.
Collect with `cluster/collect_results.sh` (checkpoints and `pkl/` are not shipped; the screens are).
The `.pkl` files stay on the cluster in `repo/models/`; benchmarks read them there.

**Read the six screen sections for all 48 charts, then check in with Thomas before any benchmark
item is submitted.** The gates are pass/fail; the rest is what the report quotes:

| section | what it measures | gate |
| --- | --- | --- |
| (a) task error | `G_1` v FK, median / p95 mm and deg; fraction > 50 mm (the disconnected-fibre floor on the rigid arms) | none (the hypothesis: a floor on the rigid arms, not on soft12) |
| (b) inversion | `G(G^-1(w)) - w` v DYS iterations, with / without the Newton polish, float64 / float32; sets `n_dys` | **polished residual < 1e-10 (p99) at the chosen `n_dys`** |
| (c) z = 0 solution | FK error and program cost of `decode(y, 0)` v the dataset's per-pose minimum; `J_hat` v `J` (m, x only) | none |
| (d) gain | spectral norm of `dq/d[y, z]` over the +-5 latent box and the `c` box, max `|q|`, the bi-Lipschitz inequality on random pairs | **gain <= the certified bound (1/mu x scale); no pair violates the inequality** |
| (e) the rpy cut | task error binned by distance to the nearest roll / pitch / yaw cut; fraction of targets within 0.1 rad | none |
| (f) representability | `decode(invert(q)) - q` max abs; `|y_hat - y_FK|` (the paired start's `c` offset) | **round trip ~1e-10** (predicts paired starts are exact, unlike the cVAE's 0/480) |

A chart failing (b), (d) or (f) is a bug in the chart or its export, not a result: stop and report,
do not benchmark it.

### 4. Stage, then the benchmark smokes

```bash
cluster/stage_code.sh                                            # training drained first
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKSMOKE.txt 2
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKSMOKE_SVGD.txt 2
```

192 + 192 items, one per logical run (48 charts x {grasp, pose} x {paired, native}), two cells each
(`--cells 0:0,31:5` of the record's grid), `sc_LOINKSMOKE_<robot>_<label>_<solver>_<row>_2_180_<start>`.
They pass when every item exits 0, no `_abort_on_dead_arm`, `metadata["chart_arch"]["family"] ==
"loink"`, and the paired cells' `start_q_error` is ~1e-6 or better (screen (f)) -- no
`unrepresentable_start`. `submit_bench.sh` refuses a manifest until every `--checkpoint` it names
exists on the cluster: the guard working, before training finishes.

### 5. The probe

```bash
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE_SVGD.txt 4
```

Paper conditions, one solve per V100, no MPS, as the record. **Probe first** (the cVAE precedent,
Thomas 2026-10-19), but with paired probes too -- the inverse is exact, so a paired cell is a real
solve here:

- rigid arms (Panda, iiwa, screw): every chart x task x start as ONE unsharded item of 60 cells,
  guess 0 of every target (`--cells 0:0,...,59:0`), tag `sc_LOINK_..._60_180_native` /
  `..._60_180_paired` -- 144 items per solver;
- soft12: the full 480-cell rows, sharded 8 (IPOPT) / 24 (AL-SVGD) -- 384 / 1152 items;
- the joint-space ACCEPTANCE run in LOINKPROBE, claimed first: CVAE_ACCEPT's row (Panda pose
  paired IPOPT, the record's flags, `--arms numerical`), `sc_LOINKACC_panda_n6_ipopt_posetip_480_180_paired`,
  8 items. It must be bit-identical to the record (q, iterations, cost, verdict per cell) after this
  branch's refactor, or no LOInK column pairs with the record's joint-space arm until understood.

Estimates (`gen_manifest.py --stage <S> --summary --procs 2`, at the FLOW's per-cell rates and
AL-SVGD's 5 s/cell allotment, +180 s per item; a chart that runs to the clock costs up to 180 s a
cell, i.e. 3 h per rigid probe item, still inside ITEM_TIMEOUT's 8 h):

| stage | items | item-hours | LPT wall on 4 nodes x 2 |
| --- | ---: | ---: | ---: |
| LOINKSMOKE | 192 | 10.0 | ~1.2 h |
| LOINKSMOKE_SVGD | 192 | 10.1 | ~1.3 h |
| LOINKPROBE | 536 (144 rigid probes 15.7 h, 384 soft12 46.6 h, acceptance 0.5 h) | 62.8 | ~7.8 h |
| LOINKPROBE_SVGD | 1296 (144 rigid probes 19.2 h, 1152 soft12 89.6 h) | 108.8 | ~13.6 h |
| LOINKEXT, at most (every rigid row qualifies) | 1152 | 125.7 | ~15.7 h |
| LOINKEXT_SVGD, at most | 3456 | 268.8 | ~33.6 h |

**No verdict is read off a probe.** It decides which rigid rows are run in full, nothing else; a
row's success, cost and wall are quoted from its 480 cells or not at all.

### 6. The extension

Collect the probe summaries (they are unsharded: no merge), promote them under `results/` (or leave
them in a `results/_cluster_staging/<stamp>/results/` tree, which the EXT generator also reads), then:

```bash
python cluster/make_loink_ext.py --summary --procs 2   # writes manifest_stageLOINKEXT(_SVGD).txt
python cluster/make_loink_ext.py --check               # exit 1 if the committed ones are stale
python cluster/gen_manifest.py --selftest              # EXT must be whole rigid rows of the full stage
```

The rule, fixed before any probe ran: **every rigid chart x row (solver, task, start) whose 60-cell
probe solved >= 1 cell gets its full 480-cell row**, sharded as stage CVAE; its items are the
would-be full stage's own, byte for byte (`stage_LOINK("LOINKFULL")`), so the probe's 60 cells are
re-solved inside the row. A probe counts only once its summary holds all 60 learned records; one not
yet collected keeps its row out and is listed. Before any probe is collected the manifests are
written EMPTY and the script says so -- never submit an empty manifest. Commit the manifests, stage,
then:

```bash
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKEXT.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKEXT_SVGD.txt 4
```

Regenerating EXT after more probes arrive keeps every item id, and `run_items.sh` keys its state by
the manifest's BASENAME, so resubmitting the same manifest skips the items already done. Mind a job
still queued on the old EXT manifest when restaging (`stage_code.sh` refuses; never `FORCE_STAGE=1`
past it unless the only change is a manifest no running item reads).

### 7. Collect, merge, report

```bash
cluster/collect_results.sh                 # merges shards; promote each run as it prints
```

If `collect_results.sh` times out walking the whole tree (it did in the cVAE campaign), tar the
stage's summaries on the cluster (`results/<robot>/benchmark/sc_LOINK*_shard*of*/summary.json` for
the sharded rows, `sc_LOINK_*_60_180_*/summary.json` for the probes), pull one tarball, extract into a
new `results/_cluster_staging/<stamp>/`, run `cluster/merge_shard_summaries.py <staging>/results`,
and copy each merged run to `results/<robot>/benchmark/`. **Read the acceptance run first** (must be
IDENTICAL to the record's joint-space arm). Every LOInK run's `grid_hash`, `scene_fingerprint`, cap
and cells must match its record row; the cap check (clock AND budget, >= 24 budget cells voids a
row) applies as everywhere -- a VOID row is re-measured with the iteration budget lifted, never the
clock.

## Resume checks and failure modes

- **Is a run trained?** Only `--status` says so: step 620000 (P) or 150000 (A). A job leaving the
  queue is not evidence. The payload's exit code is in the launch log (`launch.log-<jobid>`; pair:
  `train.log-<jobid>` in the run dir), and for 0 / 5-7 also in `results/train/<run>/node0.SENTINEL`
  (2 and 4 exit before training, so they leave no sentinel):
  - 2 = refused before training (unknown FAMILY, or a row's `--checkpoint_every` that does not
    divide its final step -- no step-named checkpoint would exist to export);
  - 4 = no dataset `.DONE`;
  - 5 = training exited 0 without the step-`<N>` checkpoint;
  - 6 = export or its round trip failed (`decode` / `G(decode(w)) - w` tolerances: a chart bug);
  - 7 = the screen failed.
- **Walltime kill / node failure**: resubmit the same run (or pair) -- `--ckpt_path=auto` resumes
  from `last.ckpt`. `--all` (with the same `ONLY=`) does it for every unfinished row.
- **Trained but not exported** (rc 6/7, or `RUN_EXPORT=0`): resubmit that run by name with a short
  walltime; it resumes at its final step, Lightning returns at once (check the log says so), and the
  in-job export runs again. A row of `chart_runs.txt` supplies its own FAMILY, ROBOT and args:
  `bash cluster/submit_chart_train.sh panda_loinkmrA 01:00:00` (`--all` skips finished rows).
- **A pair where one run died**: resubmit the pair; the finished run resumes, returns at once and
  re-exports, the other continues. Or resubmit the dead run alone -- the guard refuses while the
  pair job still holds either name.
- **A benchmark item killed or a solve overrunning**: `run_items.sh`'s watcher kills a solve past
  1.5x its clock (270 s at 180 s) and relaunches the item, which resumes from its per-cell JSONL;
  the killed cell is `killed_overrun`, `timed_out=True`, scored from its last iterate
  (`cluster/README.md`, "The overrun watcher"). A dead item is reclaimed with
  `cluster/collect_results.sh --reclaim <manifest>` only when nothing of that manifest runs, then
  the same manifest is resubmitted.
- **Restaging mid-training** is refused by `stage_code.sh`; never `FORCE_STAGE=1` past it -- the
  queued and running jobs re-read the repo's code.
- **PROCS=2 always, no MPS** for every LOInK benchmark manifest (paper conditions: one solve per
  V100); `DRAKE_ENV=crc` for every `_SVGD` manifest and never for the others (`submit_bench.sh`
  refuses a mix).
