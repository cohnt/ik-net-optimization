# `helix7`, the helical-joint arm: the run's own record

What would be queued, what it depends on, and what a resuming session should check first.
The design and the measured facts live in `CLAUDE.md`; this is the operational half.

**STATUS: datasets BUILT; the training ladder is QUEUED behind the soft arm's campaign.**
No benchmark stage is submitted. The rest of this file is written out so it can be
launched deliberately.

### The queued chain, 2026-09-25

```
5738845  helix_train_smoke_helix7_p050_n4   1 node, 20 min   afterany:5733058 (soft arm tail)
5738846  helix7_p050_n6   4 nodes, 620k     afterok:5738845
5738847  helix7_p050_n4   4 nodes, 620k     afterok:5738845,afterany:5738846
5738848  helix7_p050_n8   4 nodes, 620k     afterok:5738845,afterany:5738847
5738849  helix7_p025_n6   4 nodes, 620k     afterok:5738845,afterany:5738848
5738850  helix7_p100_n6   4 nodes, 620k     afterok:5738845,afterany:5738849  <- the tail
```

**EVERY rung carries the smoke gate, not just the first** (a comma-separated dependency
list is an AND in Slurm), because gating only the first rung is correct or not depending on
cluster configuration. Here `DependencyParameters` is null and `kill_invalid_depend` is
unset, so a failed `afterok` leaves a job PENDING for ever with
`DependencyNeverSatisfied` -- the ladder would be blocked, but only by the accident that
rung 1 never *terminates* and so never satisfies rung 2's `afterany`. On a cluster that
sets `kill_invalid_depend`, rung 1 would be cancelled, its termination would satisfy rung
2, and the whole ladder would train on a robot whose smoke failed. `GATE_TYPE` in
`chain_ladder.sh` now emits the AND form, which is right under both.

**AN UNSATISFIED DEPENDENCY CANCELS NOTHING AND NOTIFIES NOBODY.** The jobs sit PENDING
looking exactly like queued work. So watch the smoke's exit rather than assuming the chain
is advancing, and if it fails, `scancel` the stalled rungs promptly -- another campaign is
chained behind this one's tail and a stall propagates into it silently.

Queued **explicitly behind** the soft arm's chain rather than left to backfill. Backfilling
looks more polite and is worse: that campaign is back-to-back 4-node links, so the only
windows to backfill into are the drains between them, and a 1-node job starting in one
head-of-line blocks the next 620k-step run for its whole duration.

**The zero-pitch rung is not in the chain.** See below.

### This branch runs from its OWN cluster tree, `~/learned-ik-helix/`

`~/learned-ik` is an rsync of a working tree rather than a version-controlled clone, so
there is no branch there to switch and restaging means `rsync --delete` over whatever is
present — which, while another campaign is live, changes the code its queued items re-read
when they start. Select the tree with `SC_ROOT=learned-ik-helix` when staging or
submitting; `LEARNED_IK_ROOT` is what the job-side scripts read, and the submitters now
forward it (they used to build paths from `SC_ROOT` while the payload silently fell back to
the default tree).

Its own: `repo/`, `home/` (its own ikflow dataset cache), `state/`, `results/`.
Symlinked to the default tree and used **strictly read-only**: `venv/`, `drake/`,
`sysdeps/`, `home/.cache/drake`. Read-only means no `pip install` of any kind — the other
campaign runs out of that venv, and a package added here would land inside its run
invisibly. If this branch ever needs a package the default tree lacks, **copy** the venv.
`rm -rf ~/learned-ik-helix` removes the tree without touching anything else.

**Job names carry a prefix derived from the tree** (`learned-ik` -> `lik`,
`learned-ik-helix` -> `helix`), because `stage_code.sh`'s live-campaign guard and
`submit_train.sh`'s RUN_DIR guard both match on job NAMES. Unscoped, two campaigns refuse
each other's submissions for as long as either runs — which happened, and is fixed in both
places rather than forced past.

**Check a tree before trusting it:** `LEARNED_IK_ROOT=$HOME/learned-ik-helix ROBOT=helix7_p050
LLsub ./cluster/preflight_root.sh -s 8 -q debug-cpu -T 00:20:00 -J helix_cal_preflight`.
It exercises what a *benchmark worker* needs, which is strictly more than a dataset build
needs: `run_items.sh`, not the payload, is what puts Drake on `PYTHONPATH`, the extracted
libraries on `LD_LIBRARY_PATH` and the `drake_models` cache where `ProcessModelDirectives`
looks. It passed on 2026-09-25 (job 5738820): Drake imports, all four rungs register, the
scene builds at 7 positions and 22 bodies, the screw row reads `-inf` before the repair and
`+-6.2832` after, and the dataset resolves with its sentinel.

### Datasets, built 2026-09-25

Jobs 5738219-5738222 on `xeon-p8`, chained `afterok` by `cluster/chain_datasets.sh` so
exactly one ran at a time on one node. 25M training samples + 15k test each, seed 0,
`--only_non_self_colliding`, 1.4 GB per rung, **1:06 to 1:22 each and 5.5 minutes for all
four** -- far cheaper than the soft arm's, which is what a 7-DoF closed-form forward
kinematics buys over a 9-DoF numerical one.

They are four genuinely different robots, not one robot four times, and the datasets say so
physically: the joint-angle columns agree across rungs (same limits, same seed, screw
coordinate spanning +-6.2788 = +-2pi), while the reachable set grows with the stroke --
flange `z` reaches 1.260 at pitch 0 against 1.359 at 0.100 m/rev, a difference of 0.0994 m,
which is the one-sided travel of a full revolution to three decimals. Horizontal reach moves
the same way, 0.840 to 0.935.

A 20000-sample plumbing smoke ran first as a job on `debug-cpu`, in a throwaway root: this
job writes a `.DONE` sentinel on success and `train_flow.sh` hard-fails without one, so a
smoke landing beside the real cache would leave a sentinel indistinguishable from a
finished 25M build. `helix7_p000`'s dataset was built before that rung left the ladder and
is kept, so reviving the rung costs one training run rather than a rebuild.

## Order of operations, and why it is this order

```
datagen (one rung at a time)        CPU partition (xeon-p8), no GPU used
        |
        v   train_flow.sh HARD-FAILS (exit 4) without the dataset's .DONE sentinel
chart ladder, sequential at 4 nodes:
        helix7_p050_n6   <- the FIELDED rung, pre-registered
        helix7_p050_n4       the chart ladder measurement
        helix7_p050_n8       the chart ladder measurement
        helix7_p025_n6       the pitch ladder
        helix7_p100_n6
        |
        v   export + screening run INSIDE each training job, on node 0, on rc=0
benchmark stages: HELIX, HELIXCHART, HELIXPITCH
```

**Each pitch is a different robot**, so each needs its own 25M-sample dataset and its own
620k steps, and cannot borrow another rung's chart, because the arm's links are its own.

**`helix7_p000` IS NOT TRAINED.** At pitch 0 the arm is an ordinary S-R-S manipulator, and
the project already fields two of those *with* analytic columns, so a chart for it would
spend 620k steps rediscovering that an algebraic arm is algebraic. Thomas, 2026-09-25:
*"Seems like a waste of time to train a model for helix7_p000. We already have analytic
arms, we don't need a specific control example here."* The spec stays — the tests use it,
and it is the degenerate member that makes the family a family — and its dataset is already
built, so the rung can be added later for the price of one training run. The ladder is
therefore **5 training runs**: three architectures on the primary rung, plus the two other
pitches at the adopted architecture. What the pitch ladder measures is a dose-response
among helical arms; the "an analytic column could exist here" end of the scale is held by
the Panda and the iiwa, which already have one.

## Submitting

**ONE DATASET AT A TIME.** The soft arm measured the failure: ikflow's end-of-run summary
scans every dataset in the shared `~/.cache/ikflow/datasets/`, so a concurrent sibling's
half-written tensor makes a finished job exit 1 with its data already correct on disk. The
data is fine; the missing `.DONE` sentinel is not, and `train_flow.sh` hard-fails without it.

```bash
# datagen -- ALREADY RUN on 2026-09-25, kept for the record and for a rebuild.
# One node, one rung at a time, from ~/learned-ik-helix/repo on the login node:
#   LEARNED_IK_ROOT=$HOME/learned-ik-helix bash cluster/chain_datasets.sh \
#       helix7_p050 helix7_p000 helix7_p025 helix7_p100
# (p000's dataset was built before that rung was dropped from the ladder; it is kept
#  because it costs 1.4 GB and would otherwise have to be rebuilt to revive the rung.)
# The single-rung form below is the equivalent by hand., on the CPU
# partition. `build_dataset_job.sh` exports CUDA_VISIBLE_DEVICES="" itself, so a GPU node
# would be wasted on it, and the `xeon-g6-volta` group cap stays entirely available for
# training. Verified locally with no GPU visible: the sampling path runs unchanged.
DATASET_ROBOT=helix7_p050 LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-p8 -T 02:00:00 -J lik_dataset
DATASET_ROBOT=helix7_p000 LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-p8 -T 02:00:00 -J lik_dataset
# ...then p025 and p100 if the full pitch ladder is being run.

# a 200-step smoke first: a FAILED SMOKE MUST BLOCK the long chain
bash cluster/submit_ladder.sh --smoke helix7_p050

# then the ladder, chained so it advances with no session attached
bash cluster/chain_ladder.sh <smoke_jobid> helix7_p050_n6 helix7_p050_n4 helix7_p050_n8 \
    helix7_p025_n6 helix7_p100_n6

# the benchmark stages, once the charts exist
python cluster/gen_manifest.py --stage HELIX      --wall-time 180 --targets 60 --guesses 8 --shards 8
python cluster/gen_manifest.py --stage HELIXCHART --wall-time 180 --targets 60 --guesses 8 --shards 8
python cluster/gen_manifest.py --stage HELIXPITCH --wall-time 180 --targets 60 --guesses 8 --shards 8
```

`chain_ladder.sh` uses `afterany`, deliberately: a dead rung must not stall every rung behind
it. The cost is that a dead rung's successor starts anyway, so **always check
`submit_ladder.sh --status` after a chain** — a rung short of `MAX_STEPS` was not trained,
whatever the queue says, and resubmitting resumes from `last.ckpt`.

## What a resuming session should check FIRST

0. **DID THE SMOKE PASS?** `sacct -X -o JobID,JobName%30,State,Elapsed -j 5738845`. This is
   first because it is the one failure with NO SIGNAL: if the smoke failed, the five rungs
   gated on it sit PENDING for ever with `Reason=DependencyNeverSatisfied`, looking exactly
   like ordinary queued work. Nothing exits, nothing notifies. Check with
   `squeue -u $USER -o '%i %j %T %r'` and look for `DependencyNeverSatisfied`.
   **If it failed: `scancel` the stalled rungs and TELL THE SOFT ARM SESSION** — it has work
   chained behind this ladder's tail (5738850), so a stall here silently stalls that too.
1. `SC_ROOT=learned-ik-helix bash cluster/submit_ladder.sh --status` — global_step per rung
   against 620000. A rung short of it was not trained, whatever the queue says;
   resubmitting resumes from `last.ckpt`.
2. `LLstat` — this tree's jobs are named `helix_*` (`helix_train_<run>`, `helix_bench_*`,
   `helix_cal_*`). `lik_*` jobs belong to the soft arm campaign and are not ours.
3. That each dataset has its `.DONE` sentinel, **in this tree's own cache**:
   `ls ~/learned-ik-helix/home/.cache/ikflow/datasets/helix7_p050/`
4. `python cluster/gen_manifest.py --selftest` before generating any manifest.

## Traps specific to THIS robot

* **A SCREW JOINT'S `<limit>` IS DISCARDED BY EVERY DRAKE PARSER.** `ParseJointLimits` is
  reached only for revolute and prismatic joints, in URDF and SDFormat alike, so the plant
  reports `[-inf, inf]` on that coordinate. Nothing crashes. The joint-limit row — the one
  row this robot exists to stress — goes vacuous, the joint-space arm's box goes unbounded,
  and the target sampler's `rng.uniform(lower, upper)` returns `nan` and spins for ever.
  `src/helix_arm/limits.py` repairs it inside the program's `__init__`, after `Finalize()`
  and **before `ToAutoDiffXd()`** — that call takes an independent copy, and one made first
  carries the infinities for ever. Anything that builds this plant WITHOUT constructing a
  program must repair it itself; `scripts/probe_shelf_acceptance.py` does.
* **A missing checkpoint is a column of zeros, not an error.** The driver requires
  `--checkpoint`; there is no published chart to fall back on. `gen_manifest`'s selftest
  asserts every helix item carries one, and that it belongs to the rung the item names.
* **The fielded rung is PRE-REGISTERED at `n6`.** `n4` and `n8` are measured and reported,
  never selected from. Picking whichever benchmarks best is selecting on the test set.
* **The pitch rungs do not pair.** Each draws its own grid — the stroke changes the
  reachable set — so compare by target-level success rate with a bootstrap CI over targets.
  McNemar applies within a rung, between the arms, and not across rungs.
* **Run a cap ladder (45 / 180 / 360 s) before reporting any verdict**, per the record's own
  precedent: a row that read as a clear loss at 45 s was a tie at 180 s with zero timeouts.
* **Datagen belongs on the CPU partition; BENCHMARKS DO NOT.** The dataset builder is
  pure sampling -- our batched torch FK and jrl's capsule distances -- and its job script
  already forces `CUDA_VISIBLE_DEVICES=""`, so it should never occupy a GPU node. The
  benchmark jobs are the opposite case: moving them to the CPU partition to dodge the
  4-node `xeon-g6-volta` group cap has been tried on this project and does not work. Do
  not read one as licence for the other.
* **The status quo is not this robot's to change.** Its rows stand beside the record's until
  Thomas merges to main, which is the acceptance gate.

### Two ways to misread the training telemetry, both met on this ladder

* **`val_l2_error` in `status.json` is the UNCLAMPED mean, and it is tail-driven.** It is
  `val/l2_error` from the fork's pole callback (`third_party/ikflow/ikflow/training/pole_callback.py`),
  a mean over a 4000-sample validation draw that includes samples the flow put outside the joint
  limits. A single such sample moves it by an order of magnitude: `helix7_p050_n4` read 0.2093 at
  step 360000 and **2.9880** at 380000, on the same row as `val/l2_error_std` **219.7**, while
  `val_clamped/l2_error` went 0.2060 -> **0.2037** with a std of 0.179. The clamped series is smooth
  on both rungs measured so far; the unclamped one spiked three times on `n4` alone (0.345 at 200k,
  0.328 at 280k, 2.988 at 380k) and never on `n6`. **Read `val_clamped/l2_error` out of
  `metrics/version_0/metrics.csv`; treat the `status.json` number and `submit_ladder.sh --status`'s
  `val_l2` column as a liveness check only.** Both that helper and `watch_ladder.sh` print the
  unclamped field, so the hazard is in the tooling a monitoring session will reach for first. It is
  also NOT the pole screen saying anything: `pole/frac_gt_1000` was 0.0 at every one of those spikes.
* **The last screen printed in an export log is not the final checkpoint.** The export job iterates
  checkpoints in lexicographic order, so `step80000` sorts *after* `step620000` and is what a `tail`
  shows. On `helix7_p050_n6` that is 81.7 mm against the final chart's **35.0 mm** — quoting the log
  tail would report more than twice the true error. Select the screen by step number, never by
  position in the log.

### The out-of-limits tail is CREATED by training, and only on the shallow rung

Quantified once both `n4` and `n6` had reached 620000, so the statement is made on finished runs.
Taking the per-checkpoint ratio `val/l2_error` divided by `val_clamped/l2_error` as the detector — how
much of the unclamped mean is tail rather than fit:

| rung | checkpoints with ratio > 1.3, first half | second half | largest ratio | steps at which it spiked |
| --- | --- | --- | --- | --- |
| `helix7_p050_n4` | 1 of 12 | **5 of 13** | 1.6 -> **14.7** | 200k, 280k, 380k, 440k, 460k, 480k |
| `helix7_p050_n6` | 0 of 31 | 0 of 31 | 1.00 | none |

So the out-of-limits tail is not a property of the validation draw, and not noise: on `n4` it is
**created by training**, it arrives in the second half of the run, and it grows. On `n6` it never
appears at all — the unclamped and clamped means agree to two decimal places at every one of 31
checkpoints. **`pole/frac_gt_1000` reads 0.0 at every one of those six spikes**, so the
unclamped-over-clamped ratio is a *more sensitive* detector of out-of-limits mass than the pole
screen is at its fielded 1000-rad threshold — which is expected, since 1000 rad is a bimodality
separator and a sample need only cross a joint limit to enter the clamped/unclamped gap.

This is the same direction as the record's gain-ceiling finding and the same inversion of it. `n4`'s
log ceiling is `2.4976 * 4` (gain 2.2e4) and its final task-pose `pole/max` is **7017**, or 89% of the
way up in log terms — squarely inside the 78-89% band the record says training walks. `n6`'s ceiling
is 3.2e6 and it reached only **57.0**, or 27%. So the *shallower* rung is the one that conforms to the
rule here, and the deeper one has headroom it never uses. Read beside the screens, that is consistent:
`n4` is the rung with 2 of 20000 task-pose samples above 1000 rad and a 148.9 mm median chart error,
against `n6`'s zero and 35.0 mm. **The pre-registered rung `n6` is also the clean one on every
intrinsic measure**, which is worth stating precisely because the record's own finding is that
intrinsic screens do not predict cells — it is a reason to trust the pre-registration, not a reason to
select on it.

### The p050 chart ladder, complete: training uses LESS of its ceiling the deeper the chart

All three rungs of the primary pitch reached 620000 and exported `rc=0`. Screens selected by step
number from `results/pole/helix7_p050_n*/`, chart error over 5000 held-out poses:

| | `n4` | `n6` (pre-registered) | `n8` |
| --- | --- | --- | --- |
| median chart error | 148.9 mm | 35.0 mm | **28.9 mm** |
| p90 / p99 | 468 / 780 mm | - / 654 mm | 173 / 559 mm |
| `pole/max`, task pose / box | 7017 / 2052 | 57.0 / 33.5 | 22.4 / 21.0 |
| `pole/frac_gt_1000` | 0.0001 | 0.0 | 0.0 |
| `pole/frac_gt_3` | - | - | 0.599 |
| log gain ceiling `exp(2.4976 * nb_nodes)` | 2.2e4 | 3.2e6 | 4.8e8 |
| **fraction of that ceiling used, in log terms** (task pose) | **89%** | **27%** | **16%** |
| training wall clock | 10:44:51 | 14:53:56 | 19:26:55 |

Two things to take from this, both of which bear on the record's gain-ceiling finding rather than on
this robot alone.

**The record's rule that "training walks 78-89% of the way up whatever log ceiling it is given" holds
at `n4` and then fails progressively with depth.** 89% / 27% / 16% is monotone, so on this robot the
ceiling is not a predictor of where a trained chart lands -- it is only a bound, which is the weaker
of the two claims the record makes for it. Anyone quoting the 78-89% band should say which robots it
was measured on.

**`n8`'s ceiling is 4.8e8, far above the 1e7-1e16 band where solves actually die, and `n8` is
nevertheless the CLEANEST rung on every intrinsic measure.** That is the soft PCS arm's result
reproduced on a second constructed robot: an above-band rung does not degrade, where on the iiwa it
is strictly worse. So the gain-ceiling selection criterion is again satisfied vacuously, because this
robot has no runaway population for the criterion to protect against -- `frac_gt_1000` is 0.0 at
every checkpoint of `n6` and `n8`, and 0.0001 at `n4`'s worst.

**Accuracy is monotone in depth and the pre-registered rung is not the most accurate one.** `n8` beats
`n6` by 6 mm of median error. The record's standing finding is that chart accuracy runs BACKWARDS to
cells, so this is not a reason to revisit the pre-registration; HELIXCHART will benchmark all three and
that is the measurement entitled to an opinion.

### The pitch moves runaway mass at a FIXED architecture: `p025_n6` against `p050_n6`

`helix7_p025_n6` finished 620000 steps and exported `rc=0` (14:58:51, against `p050_n6`'s 14:53:56).
Same architecture as the pre-registered rung, half the screw pitch -- so the closer of the two to the
ordinary revolute arm at pitch 0. Screens selected by step number:

| | `p050_n6` (pre-registered) | `p025_n6` |
| --- | --- | --- |
| median / p90 / p99 chart error | 35.0 / 226 / 654 mm | **24.4** / 183 / 656 mm |
| task-pose `pole/max`, fraction of log ceiling | 57.0, 27% | **689, 44%** |
| box `pole/max`, fraction of log ceiling | 33.5, 23% | **9.3e5, 92%** |
| `pole/frac_gt_1000`, task pose / box | 0 / 0 | 0 / 0.00005 (1 of 20000) |
| in-training checkpoints with any sample > 1000 rad | 0 of 31 | **8 of 31**, steps 320k-500k, gone by 520k |
| worst in-training `pole/max` | 35.5 | 1.4e4 |
| largest unclamped / clamped validation ratio | 1.00 | 1.05 |

**At the same depth, the smaller pitch walks much further up its gain ceiling**: the box screen reaches
92% of the log ceiling against 23%, which is inside the record's 78-89% band and above it, where
`p050_n6` sits far below. So the architecture alone does not decide how much headroom training uses on
this robot; the pitch does too, and in the direction that makes the more algebraic member the
spikier one. The in-training tail came and went as `n4`'s did, rather than growing.

What it does NOT show. The worst value anywhere is 9.3e5 rad, an order below the 1e7-1e16 band where
solves die, and the validation draw shows no out-of-limits tail at all (ratio 1.05, against `n4`'s
14.7). It is also more accurate than the pre-registered rung by 11 mm of median error, which the
record says runs backwards to cells anyway. **None of this is a selection input**: the rung is fixed
at `n6` and the pitch is part of the robot. HELIXPITCH's cells are the measurement entitled to say
whether the difference matters to the optimization.

## Measured on the laptop, before anything was queued

| quantity | value |
| --- | --- |
| torch FK against Drake, flange | 4.4e-16 position, 1.3e-15 rotation |
| every link frame against Drake | < 1e-12 |
| constraint gradients vs central differences, 12 blocks | <= 5.2e-10 |
| `X_ee_flow` (scene flange against the chart's frame) | identity, both tasks |
| paired start, `\|q(start) - q_init\|` | 0.0 on all four arms |
| collision-free fraction of uniform draws | 54.1-54.7% |
| acceptance at inset 0.10, grasp / pose (of collision-free) | 0.31-0.39% / 0.33-0.34% |
| draws per target | 465-588 |
| `P(trip)` at the fielded guard 50000 | 0 on every rung and task |
| dataset build | 20000 samples in 0.48 s, so 25M in ~10 min |
| 200-step training smoke | 9.1 steps/s, validation clean, `status.json` written |
| export round-trip | `.pkl` + `.arch.json`, forward pass OK at width 7 |

The rigid arms sit at 0.55-0.68% grasp and 0.23-0.37% pose acceptance, so this robot is in
their band. Acceptance falls monotonically with pitch on the grasp row, which is the stroke
carrying more of the configuration box out of the shelves.

## The cap check, PRE-REGISTERED before any cell is read

Written 2026-09-29, while the chart ladder was still training and no benchmark row existed, so this
rule cannot be shaped by the numbers it judges.

**`max_iter` stays at `None`, i.e. IPOPT's own default of 3000.** Thomas's call, asked explicitly:
*"Keep 3000 to match the record. Flag if that cap is binding -- rerunning cap-bound experiments is a
followup task (out of scope for now)."* So these rows are measured under stage STATUSQUO's exact
conditions and can stand beside them; raising the cap to pre-empt a budget-bound row was rejected,
because differing conditions cost more than a possible re-run.

**What must be read, on every row, before any verdict is stated.** The cap rule requires BOTH
`timed_out` and `hit_iteration_cap`; `summarise` emits them per arm as `timeouts` and
`iteration_capped`. Note which tool shows what:

* `scripts/collate.py` prints both, as its `timeouts` and `icap` columns. **Use it.**
* `scripts/report_statusquo.py` reads `iteration_capped` **nowhere at all** -- its own header
  promises "with timeouts" and that is literally what it prints. This is why the record's
  `max_iter` binding had to be discovered by hand after the fact rather than read off a table, and
  it is a blind spot in accepted work rather than something this branch introduced. Do not infer a
  clean cap from a reporter that cannot see half of it.

**The rule, fixed in advance.** For each row, `capped = max(iteration_capped)` over the two arms:

| condition | verdict |
| --- | --- |
| `timeouts` ~0 and `capped` ~0 | the budget is innocent; the row is a formulation result and carries its verdict |
| either is material on the arm that LOST or floored | **the row carries NO verdict**; state it as budget-bound and name which budget |
| either is material only on the WINNING arm | the verdict stands, and the margin is a lower bound |

"Material" is not a fresh judgement per row: it is >=5% of 480 cells, i.e. 24, chosen now so it
cannot be tuned later. Report the count either way -- a row with 3 capped cells says so.

**And the obligation stops at flagging.** Detecting and reporting a budget-bound row is this push's
deliverable; re-measuring it at a raised cap is a separate task Thomas has taken out of scope for
now. So do not hold the write-up waiting on a re-run, and do not spend cluster time on one unasked.

## The two ladders form a CROSS, not a grid -- pre-registered before any cell is read

The trained ladder is five charts: `p050` at `n4`/`n6`/`n8`, plus `p025` and `p100` at `n6`.
HELIXCHART walks `nb_nodes` at pitch 0.050; HELIXPITCH walks pitch at `nb_nodes = 6`. They
share the centre cell `(p050, n6)` and between them visit 5 of the 3 x 3 = 9 combinations. The
four corners -- `p025`/`p100` x `n4`/`n8` -- are **unmeasured, and the interaction between
pitch and chart depth is therefore not estimated by this design.** Say so wherever either
ladder is reported; a one-factor-at-a-time design cannot speak about a corner.

**The contingency, fixed here in advance.** If HELIXCHART's best rung is not `n6` AND
HELIXPITCH's best pitch is not `p050`, then the configuration the two ladders jointly point at
is a corner neither visited, and no measured row supports it. The minimum that would close it
is **one chart** (that pitch at that depth -- the dataset already exists per robot, so it is a
training run, not a datagen one) and **four logical runs** (2 experiments x 2 protocols, IPOPT
only, same cap and seed), reported as a named followup stage rather than folded into either
ladder. If only one ladder moves off the centre, no extra run is needed: that winner is already
a measured cell.

**What the extra run would and would not license.** It closes a REPORTING gap, not a selection
one. The fielded rung is pre-registered at `n6` and the primary robot at `p050`, and both rules
exist because picking whichever benchmarks best is selecting on the test set -- the record
disqualified "take whichever benchmarks best" by measurement, and the pitch is part of the
robot, fixed before any acceptance rate was read. So a corner that wins does **not** thereby
become the fielded configuration; re-fielding is Thomas's call on the pre-registration, made
explicitly, and would be stated as a deviation. Absent that, the corner is a reported cell.

**It may well not be needed.** The screw arm's `pole/frac_gt_1000` is 0.0 at every checkpoint of
both rungs measured so far, so the gain-ceiling criterion is satisfied vacuously exactly as it
is on the soft PCS arm -- whose chart ladder came out FLAT, with a 3-10 cell spread of 480 and
no rung separating from another. A flat HELIXCHART leaves nothing to disagree with and the
question does not arise.

## Still to build

* The cap ladder (45 / 180 / 360 s), before any verdict is reported. It needs trained
  charts, so it is the campaign's first methodological step rather than part of this
  infrastructure push.

The record's own section for this robot is written: **"The helical-joint arm: a robot no
algebraic method can chart"** in `CLAUDE.md`, which holds the design, the two traps and the
locally measured numbers. This file stays the operational half.
