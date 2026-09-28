# Soft arm: the run's own record

What is queued, what it depends on, and what a resuming session should check first. The
design and the measured facts live in `CLAUDE.md`; this is the operational half.

## STATE AS OF 2026-09-28 11:55 -- READ THIS FIRST ON RESUME

The session that queued all of this was paused here deliberately. **Nothing needs a human or an
agent to advance; the whole remaining chain is expressed in Slurm dependencies and will run to
completion unattended.** Branch `soft-manipulator` is clean and pushed at `c1c6bdd`.

DONE, collected, merged, promoted and reported:

| stage | jobs | outcome |
| --- | --- | --- |
| all five charts trained | 5732986, 5733055-56, 5733057-58 | 620k steps each, exports rc=0 |
| SOFT12 (3 solvers x 2 tasks x 2 protocols) | earlier | 12 logical runs, in `CLAUDE.md` |
| SOFTCHART (`soft12` n4/n6/n8, IPOPT) | 5752615-18 | 96/96 items, rc=0, 12 logical runs |
| SOFTDOF (soft9/12/16 at n6, IPOPT) | 5752619-22 | 96/96 items, rc=0, 12 logical runs |

Tables and verdicts: **`docs/soft-arm-ladders.md`**. Both ladders are reported; the record is
updated; the `max_iter` caveat is recorded and Thomas ruled on 2026-09-28 to re-measure nothing.

STILL RUNNING -- the only live work:

* **SOFTCAP** (`5752635-38`), the NLopt grasp cap ladder. **The 360 s rung was RETIRED at
  11:55 on Thomas's call**, so the stage now delivers a **two-point ladder at 90 and 180 s**.

  | rung | items | state |
  | --- | --- | --- |
  | 180 s | 48 | **complete**, and it is the same-configuration reproducibility control against SOFT12 |
  | 90 s | 48 | queued, runs as workers come free |
  | 360 s | 96 | 19 done, 32 finishing, **45 blocked and never run** |

  **Why retired.** Both NLopt grasp rows sit at 0/480 and 2/480 with 478-480 cells cap-bound on
  BOTH arms and median `max_violation` ~4 cm, so the row carries no verdict until the cap moves --
  that is what the stage is for. But the record predicts the floor holds: the iiwa grasp rows are
  0-3 of 60 under *every* NLopt setting and at 180 s, and 4 cm is not a solve about to converge.
  Against that, the 360 s rung is half the stage and ~8-10 h, and the screw-joint arm's five-rung
  training campaign is queued behind these four jobs on `afterany`. Thomas took the 90/180
  comparison and gave the nodes back.

  **How, and this is the part worth reusing.** Not `scancel`: the 90 s items sit at manifest lines
  145-192, BEHIND all 96 of the 360 s items (`CAP_SWEEP` is `(90, 180, 360)` but the manifest came
  out ordered 180, 360, 90), so cancelling would have left 180 s alone -- one cap, no ladder -- and
  a resubmission of the 90 s rung would have queued behind ~5 days of screw-joint training.
  Instead `cluster/retire_stage.sh manifest_stageSOFTCAP --skip 360 --yes` pre-created the
  **claims** on the 45 un-run 360 s items, which is what `run_items.sh` tests before taking an
  item, so the jobs already running step over that block and carry on to the 90 s rung with no
  resubmission and no queue wait. Claims, not `.done` markers: a claim never inflates the done
  count and is the documented dead-item state, so `--status` stays honest about what ran.
  Reversible with `collect_results.sh --reclaim manifest_stageSOFTCAP` once no job is active.

  **What to expect.** Workers reach the 90 s rung as each finishes its current 360 s item (up to
  ~4 h), then 48 cheap items at up to 32 concurrent, so **the stage should exhaust its manifest
  and the jobs exit this evening**, releasing `afterany` and starting the screw-joint chain. Final
  state will be ~147 done of 192 with 45 blocked -- that shortfall is the retirement, not a
  failure.

  **The 51 completed 360 s items are NOT a row.** Shards are target-major and neither protocol's
  shard set completes, so they cannot be merged into a 480-cell row. They stay on disk; do not
  report them as a 360 s column.

  Readings, for rate intuition: 17 done / 49 claimed at 09:24, 32 / 64 at 10:05, 39 / 71 at 11:08,
  67 / 99 at 11:55. **Do not extrapolate a single rate** -- the 09:24-11:08 window was the cheap
  180 s rung draining and read as ~7-22 items/h, which badly misled a 20 h estimate for work that
  was really ~8-10 h. Derive the remaining time from the PER-RUNG breakdown
  (`retire_stage.sh <manifest> --groups 90,180,360`), never from a done-count slope.

* **The screw-joint arm** (`5738845` smoke, `5738846-50` training), another agent's work, PENDING
  on `afterany:5752635:5752636:5752637:5752638`. It fires on its own when SOFTCAP drains. Do not
  touch it. Its owner watches for its own `afterok` hazard (a failed smoke leaves the five rungs
  on `DependencyNeverSatisfied` forever) and will scancel them itself if needed.

  The Slurm job names are `helix_*` and the agent's own robot key is `helix7`, so match on those
  strings when reading `LLstat` -- but **call the mechanism a screw joint** in prose.
  Thomas, 2026-09-28: *"stop calling it a helix joint. It's a screw joint. That's standard
  terminology (e.g. in URDF or SDF)."* Renaming that agent's stages or jobs is its call, not ours.

  **Should its session ever be absent, do not scancel these on its behalf** -- they are its work,
  a cancel is hard to reverse, and `DependencyNeverSatisfied` wastes no compute. Report a nonzero
  smoke to Thomas and let him decide.

WHAT TO DO ON RESUME, in order:

1. `bash cluster/collect_results.sh --status` -- SOFTCAP reads **~147 done of 192**, not 192.
   The 45-item shortfall is the retired 360 s rung (see above), not a failure.
2. `bash cluster/collect_results.sh` and then promote the merged tags out of
   `results/_cluster_staging/<stamp>/` into `results/<robot>/benchmark/`.

   **The 180 s shard set straddles two collections, and the tooling ALREADY handles it.**
   SOFTCAP was collected once mid-flight (staging `20260928-092551`) at 17 items done, so the
   24 shards of each `sc_SOFTCAP_soft12_n6_nlopt_mugshelf_480_180_{native,paired}` row span that
   directory and the final one. An earlier note here said to expect to build the union by hand;
   **that was stale.** `collect_results.sh` passes `--also` for EVERY prior timestamp-shaped
   staging directory (its lines 176-206), which was added on 2026-09-20 after stage STATUSQUO
   put one row's shards across THREE collections, and was verified by re-merging seven
   already-merged rows and reproducing them exactly. So run the normal collect, then READ THE
   MERGER'S REPORT for those two tags and intervene only if it says `INCOMPLETE`.
   If it ever does, `cluster/merge_shard_summaries.py <newest> --also <older> --only <tag>` is
   the manual form -- it re-runs `summarise` rather than stitching per-shard numbers. Do not
   create a hand-made directory inside `results/_cluster_staging/`: a non-timestamp name there
   was once mistaken for a collection and silently cost two rows their merge.
3. Report SOFTCAP against the cap rule it exists to answer, **as a 90-vs-180 two-point ladder**:
   the two NLopt grasp rows came back 0/480 native and 2/480 paired with 478-480 cap-bound ON
   BOTH ARMS and median `max_violation` ~4 cm. The question the surviving rungs answer is
   whether HALVING the cap to 90 s changes anything; if it does not, the floor is not a budget
   artefact. The 360 s direction was retired unmeasured and must be reported as such, never as
   a null result. Check `hit_eval_cap` as well as
   `timed_out` -- `nlopt_max_eval` defaults to 0 so it should be disabled, but verify rather
   than assume, and remember the 180 s rung is a same-configuration reproducibility control
   against stage SOFT12's own NLopt grasp rows.
4. Then the FK surrogate fit, which is the last open item on this robot. It is a CLUSTER job,
   never local. Two things are known before spending a GPU hour: float32 for the fit and float64
   for the screen and shipped weights, and a per-segment architecture composed analytically
   rather than one net over all 33 body poses.


## Order of operations, and why it is this order

```
datagen (soft12, soft9, soft16)   CPU-only, ~11 min each, xeon-g6-volta
        |
        v   train_flow.sh HARD-FAILS (exit 4) without the dataset's .DONE sentinel
chart ladder, sequential at 4 nodes:
        soft12_n6   <- the FIELDED rung, pre-registered
        soft12_n4       the ladder measurement
        soft12_n8       the control ABOVE the ~1e7 gain ceiling
        soft9_n6        the DOF ladder
        soft16_n6
        |
        v   export + screening run INSIDE each training job, on node 0, on rc=0
benchmark stages: SOFT12, SOFTCHART, SOFTDOF   (SOFTFK needs the surrogate)
```

Datagen goes to `xeon-g6-volta` rather than `xeon-p8` because another project's
`run_matrix.sh` occupies the CPU partition, and this job never touches a GPU anyway
(`CUDA_VISIBLE_DEVICES=""`, and the shim builds every tensor on the CPU on purpose).

## Submitting

**ONE AT A TIME.** Launching all three together made soft9 exit 1 after its data was already
written: ikflow's end-of-run summary scans every dataset in the shared cache directory and read
a sibling's half-written tensor. The data was fine; the missing `.DONE` sentinel was not, and
`train_flow.sh` hard-fails without it.

```bash
# datagen, one per rung -- from ~/learned-ik/repo on the login node
DATASET_ROBOT=soft12 LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-g6-volta -T 02:00:00 -J lik_dataset
DATASET_ROBOT=soft9  LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-g6-volta -T 02:00:00 -J lik_dataset
DATASET_ROBOT=soft16 LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-g6-volta -T 02:00:00 -J lik_dataset

# the chart ladder, chained so it advances with no session attached
# ALREADY SUBMITTED 2026-09-24:
#   datasets  5732960 soft12 / 5732961 soft9 / 5732962 soft16   (running)
#   smoke     5732984  afterany:5732960   -- 200 steps, 1 node, 20 min
#   soft12_n6 5732986  afterok:5732984    -- a FAILED SMOKE BLOCKS the long chain
# The resuming session chains the remaining four rungs:
bash cluster/chain_ladder.sh 5732986 soft12_n4 soft12_n8 soft9_n6 soft16_n6
```

`chain_ladder.sh` uses `afterany`, deliberately: a dead rung must not stall every rung
behind it. The cost is that a dead rung's successor starts anyway, so **always check
`submit_ladder.sh --status` after a chain** -- a rung short of `MAX_STEPS` was not trained,
whatever the queue says, and resubmitting resumes from `last.ckpt`.

## What a resuming session should check FIRST

1. `bash cluster/submit_ladder.sh --status` -- global_step per rung against 620000.
2. `LLstat` -- our jobs are named `lik_dataset` / `lik_train_<run>`.
3. That each dataset has its `.DONE` sentinel:
   `ls ~/learned-ik/home/.cache/ikflow/datasets/soft12/`
4. wandb: runs are offline on the cluster. Sync from the LAPTOP, never the cluster:
   ```bash
   source cluster/ssh_common.sh
   sc_rsync -az "$SC_DEST:learned-ik/results/train/<RUN>/wandb" results/train/<RUN>/
   .venv/bin/wandb sync --legacy results/train/<RUN>/wandb/offline-run-*
   ```
   `--legacy` is mandatory while a job is running. Do not rebuild the workspace view.

## Traps specific to THIS robot

* **A missing checkpoint is a column of zeros, not an error.** The soft benchmark driver
  requires `--checkpoint`; there is no published chart to fall back on. `gen_manifest`'s
  selftest asserts every soft item carries one.
* **The fielded rung is pre-registered at `n6`** -- the most expressive the gain-ceiling
  rule admits below the ~1e7 band. `n4` and `n8` are measured and reported, NOT selected
  from. Picking whichever benchmarks best is selecting on the test set.
* **The DOF rungs do not pair.** Each draws its own grid, so compare by target-level
  success rate with a bootstrap CI over targets. McNemar does not apply across rungs.
* **Run a cap ladder (45/180/360 s) before reporting any verdict.** Both arms are more
  expensive per iteration on this robot, and the record's own precedent is a row that read
  as a clear loss at 45 s and was a tie at 180 s with zero timeouts.
* **The status quo is not this robot's to change.** Its rows stand beside the record until
  Thomas merges to main, which is the acceptance gate.

## Still to build

* The FK surrogate fit, as a cluster job. Fit in float32, screen and ship in float64. The
  4k-step laptop attempt reached 11 mm median against a 1 mm task gate and was deleted.
* The SoRoMoX golden-file equivalence test. `.venv-soromox` (soromox 0.5.0, jax 0.11.2 CPU)
  is built; the generator and the test are not written. Until they are, "the analytic model
  from the soft robot repo" is a provenance claim rather than a checked one.

## Run record: the primary chart (added 2026-09-25)

`5732986 lik_train_soft12_n6` **COMPLETED 0:0**, 13:55:34 wall, 4 nodes x 8 ranks, 620000
steps, ~12.9 steps/s once startup washed out. The inline export wrote all 31 checkpoints to
`models/soft12/soft12__n6__step*.pkl` with architecture sidecars (`export rc=0`), and screened
every one. The fielded checkpoint is `soft12__n6__step620000.pkl`.

Final numbers, both screens, on the SAME checkpoint -- keep them labelled:

| quantity | value |
| --- | --- |
| standalone `pos_err_mm/median` (5000 poses) | 2.74 |
| standalone `pos_err_mm/p99` | 24.36 |
| standalone `pole/max` (task poses, threshold 345, radius 4.96) | 5.44 |
| standalone `pole/frac_gt_threshold` | 0.0 |
| in-training `pole/max` (iiwa box, RPY-uniform orientation) | 2.4e8 |
| in-training `pole/frac_gt_1000` | 0.0198 |

The eight-order gap is the orientation draw, not the chart: the fork's callback draws position
and orientation independently, and `soft12` has no torsion, so an independently drawn
orientation is essentially never reachable. See CLAUDE.md for the full statement.

Accuracy over training (median tip error, mm): 9.40 (20k) / 4.77 (100k) / 3.62 (200k) /
3.13 (300k) / 2.90 (400k) / 2.74 (620k). Validation flattened around 460-500k.

Queue handoff worked as predicted: on release, `5733053` (the soft9 dataset re-run) claimed
nodes ahead of the chained `5733055 soft12_n4`, which is what its lower job ID buys. Nothing
had to be resubmitted.

## Run record: stage SOFT12 lost its grasp half (2026-09-25)

**What happened.** Job 5737149, the two-item smoke, exited 1. Its pose item ran and wrote a
summary; its grasp item died in 58 s with

```
TypeError: SoftArmMugProgram.__init__() got an unexpected keyword argument 'fk'
```

The stage is chained `afterany` on the smoke, so 5737163-66 started anyway and the same
failure repeated for every grasp item: **all 80 `mugshelf` items claimed and none completed,
while all 80 `posetip` items ran normally.** Half a campaign, from one constructor.

**The bug.** `SoftArmMugProgram.__init__` overrode the base signature and dropped `fk` and
`surrogate`. `scripts/soft_arm/soft_arm_benchmark.py` picks the class from `--task` and then
calls it with one fixed keyword set including `fk=args.fk`, so only the grasp branch raised.
It was in the tree from the moment `--fk` was plumbed in; the existing `--fk` test exercised
the pose program only, which is exactly why it survived. Fixed in `197e636`, with
`test_every_program_accepts_what_the_driver_passes` comparing all four classes' signatures
against the keyword set the driver passes -- introspection, so it needs neither a scene nor a
checkpoint and cannot be skipped into uselessness.

**Why this one escaped `_abort_on_dead_arm`.** That guard watches for an arm failing
identically and in under a second on its first three cells. This failure is at CONSTRUCTION,
before the first cell, so the item exits non-zero and the guard never runs. The record's rule
-- an arm whose per-cell wall time is orders below the cap is not solving badly, it is not
solving at all -- has a companion: **an item that exits in under a minute never solved
anything, and no per-cell guard will tell you so.** Read `sacct` ExitCode, not just the
summaries.

**What the smoke would have caught, and when.** The smoke ran at 20:13 and failed by 20:20.
The stage started immediately behind it and had burned all 80 grasp items within the hour.
The whole point of a smoke on `afterany` is that a human reads it inside that hour; nobody
did. Either gate the stage on `afterok`, or accept that the smoke only saves the campaign if
someone is awake. Note that on this cluster `kill_invalid_depend` is unset, so an `afterok`
gate leaves the tail PENDING for ever rather than cancelling it -- blocked, but silent.

**Provenance split, which must travel with these rows.** The pose half ran against the tree
staged as `fd95c783`; the grasp half reruns against `197e636`. The only runtime differences
are the constructor fix above and a slot-map refactor into `K.PlantSlotMap` that was verified
behaviour-preserving before the campaign. Note also that `fd95c783` is **not an ancestor of
HEAD** -- it carried the since-removed `.venv-soromox` and history was rewritten after it, so
the recorded staged-commit marker is orphaned. Trust the tree comparison, not the marker.

`cluster/manifest_stageSOFT12MUG.txt` is the rerun: the 80 dead items, byte-identical to the
`mugshelf` lines of `manifest_stageSOFT12.txt`, so tags, shards and grid hashes match the pose
half already on disk.

## The soft arm's first real rows: stage SOFT12, pose half (2026-09-25)

Six logical runs, 480 cells each, 2,880 cells, 180 s, seed 1, `soft12__n6__step620000`, arms
`learned,numerical`. The grasp half of this stage died at construction (see above) and reruns
separately. All shards merged cleanly: 8 / 8 / 24 / 24 / 8 / 8.

| solver | start | learned | joint space | exact McNemar | L ms/it | N ms/it | L timeouts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPOPT | native | **477**/480 | 334/480 | 6.5e-42 | 89.8 | 13.8 | 0 |
| IPOPT | paired | **436**/480 | 334/480 | 1.8e-16 | -- | -- | 3 |
| SNOPT | native | **479**/480 | 303/480 | 9.3e-52 | 25.5 | 29.3 | 0 |
| SNOPT | paired | **393**/480 | 303/480 | 9.6e-11 | -- | -- | 0 |
| NLopt | native | **358**/480 | 51/480 | 2.0e-79 | -- | -- | 126 |
| NLopt | paired | **325**/480 | 52/480 | 1.0e-69 | -- | -- | 161 |

**The learned arm wins all six rows decisively.** That is the same direction as the record's
rigid-arm pose rows, where every pose row under every solver is a decisive learned win.

**The per-iteration premium is ~6.5x, not 10-30x** (IPOPT native, 89.8 ms/it against 13.8),
which is the predicted consequence of this robot's joint-space arm not being free: it places
231 floating-body positions where the rigid arms' `VarsToQ` is the identity. Report it as the
baseline getting more expensive, never as the learned arm getting cheaper.

**Harness self-check passes**: the joint-space arm is identical between protocols on IPOPT
(334/334) and SNOPT (303/303), and 51 against 52 under NLopt, so protocol differences are
attributable to the learned arm alone.

**The paired protocol costs the learned arm 41 to 86 cells on every solver** (IPOPT 477 ->
436, SNOPT 479 -> 393, NLopt 358 -> 325), while the joint-space arm is unmoved (334/334,
303/303, 51/52). So the protocol effect belongs entirely to the learned arm, and it is larger
here than the record's largest rigid-arm protocol effect under IPOPT (476 -> 471).

**It is NOT a clipping defect, and `median_clip_distance` 1.39 does not mean anything was
clipped.** `clip_distance` is `c_clip_distance + z_clip_distance`
(`src/generic_program.py:1139-1157`): the distance of the unprojected `c` and `z` from their
REGIONS. Both regions are general linear constraints, not variable bounds, so IPOPT never
projects them -- the number is a diagnostic of how far outside its region the start sits, and
the comment at `:1152-1154` calling it "how far the solver's own projection will move the first
iterate" is stale for the non-legacy path. Measured directly: at the paired start the
correction is **0 to machine precision** on every seed tried, `|z|` is inside the region on an
untrained chart, and `median_start_q_error` is 0.0 on every paired row -- the arm represents
`q_init` exactly and keeps doing so. On the trained chart the smoke recorded `start_z_norm`
7.18, and `z_box` is +-5 per component, so a componentwise excess summing to ~1.39 is exactly
what the diagnostic should read.

So the paired penalty is the DESIGNED behaviour the record describes -- the arm starts outside
the latent region and the solver walks it in -- not a bound projecting a start it should not.
What is worth reporting is its SIZE on this robot relative to the rigid arms, which is a
result about the soft arm, not a bug.

NLopt's joint-space column is at 51-52 of 480 with `median_max_violation` **1.37e-01**, which
is the record's augmented-Lagrangian collapse reproduced on a third robot.

## Stage SOFT12 complete: all twelve logical runs (2026-09-27)

The grasp half landed 2026-09-26/27 (jobs `5745556-58`, `5745563`, 80 items, 8 + 8 + 24 shards
per protocol pair, all merged cleanly). With the pose half above that is **12 logical runs,
5,760 solves**, the shape the record's own stage STATUSQUO uses: 2 experiments x 2 protocols x
3 solvers, 480 cells, 180 s, seed 1, `soft12__n6__step620000`, arms `learned,numerical`.

| solver | task | start | learned | joint space | McNemar (learned / joint discordant) | p | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IPOPT | pose  | native | **477**/480 | 334/480 | 144 / 1 | 6.5e-42 | LEARNED |
| IPOPT | pose  | paired | **436**/480 | 334/480 | 132 / 30 | 1.8e-16 | LEARNED |
| SNOPT | pose  | native | **479**/480 | 303/480 | 177 / 1 | 9.3e-52 | LEARNED |
| SNOPT | pose  | paired | **393**/480 | 303/480 | 143 / 53 | 9.6e-11 | LEARNED |
| NLopt | pose  | native | **358**/480 | 51/480 | 318 / 11 | 2.0e-79 | LEARNED |
| NLopt | pose  | paired | **325**/480 | 52/480 | 284 / 11 | 1.0e-69 | LEARNED |
| IPOPT | grasp | native | **474**/480 | 445/480 | 33 / 4 | 1.1e-06 | LEARNED |
| IPOPT | grasp | paired | **468**/480 | 445/480 | 31 / 8 | 0.00029 | LEARNED |
| SNOPT | grasp | native | 340/480 | 358/480 | 83 / 101 | 0.21 | tie |
| SNOPT | grasp | paired | 315/480 | **358**/480 | 77 / 120 | 0.0027 | joint space |
| NLopt | grasp | native | 0/480 | 0/480 | 0 / 0 | 1.0 | floor, no comparison |
| NLopt | grasp | paired | 2/480 | 0/480 | 2 / 0 | 0.5 | floor, no comparison |

**Learned wins 8, ties 1, loses 1, and two rows carry no comparison.** The single loss is
**grasp under SQP paired**, which mirrors the rigid arms exactly: the record's only two losses
across 24 cells are iiwa contained grasp under SQP. A third robot reproducing the same solver x
task corner is the strongest evidence yet that it is a property of SQP on this problem class
rather than of a robot.

### The grasp quartet, IPOPT native (441 shared cells)

| | learned | joint space |
| --- | --- | --- |
| success | **474**/480 | 445/480 |
| median reported cost | 0.840 | **0.328** |
| mean wall (s) | 11.10 | **5.37** |
| median major iterations | **149** | 320 |
| median `max_violation` | **5.4e-09** | 1.2e-08 |
| ms per iteration | 89.0 | 22.4 |
| timeouts | 5 | 0 |

**The per-iteration premium is 4.0x here (89.0 against 22.4 ms) and the wall-clock premium 2.1x,
against 14x on the iiwa** -- and CLAUDE.md's prediction for why is confirmed in the direction it
was made: the baseline got more expensive, not the learned arm cheaper. The joint-space arm
places 231 floating-body positions per evaluation where the rigid arms' `VarsToQ` is the
identity, and it costs 22.4 ms/it against the iiwa's ~2. Under SNOPT the two arms are within 13%
of each other per iteration (32.7 against 32.4 native), so on this robot the premium is a
property of the solver's iteration mix, not a constant.

**Cost does NOT split by task here, and that is a departure from the record.** On the rigid arms
Table 2 has the learned arm cheaper on pose and ~1.4-1.8x more expensive on grasp. On this robot
joint space is cheaper on **nine of the ten comparable rows**, pose included -- IPOPT pose native
1.37 against 0.55 (2.5x), IPOPT grasp native 0.84 against 0.33 (2.6x), SNOPT pose essentially
level at 1.02-1.04x. The single row where the learned arm is cheaper is NLopt pose paired (0.85
against 0.98) on 41 shared cells, which is thin. So on the soft arm the learned formulation buys
feasibility and pays for it in objective value on both tasks, where on the rigid arms it bought
pose cost outright. State it as a robot-level difference, not as a task split.

`correction_binding` is 0 and `median_start_q_error` is exactly 0 under paired on every row, so
the harness self-check passes on all twelve.

### The two NLopt grasp rows are a FLOOR, not a result

Both arms at 0/480 and 2/480, with **478-480 of 480 cells timing out on both arms** and
`median_max_violation` 3.7e-02 (learned) against 4.7e-02 (joint space) -- roughly 4 cm, so
neither arm is near a solution when the clock stops. This is the record's iiwa pattern
reproduced on a third robot: "rows where both arms sit at the floor carry no comparison and no
cost column". Report them as printed rows with no verdict; an omitted row reads as missing data.

**The cap rule is NOT satisfied on these two rows and must not be quoted as if it were.** Both
arms are 100% cap-bound, which is exactly the condition under which the rule says to raise the
cap and re-measure rather than conclude. The record's NLopt precedent says the likely answer is
that more wall clock buys nothing, because the augmented Lagrangian's inner solve never
terminates -- but that precedent was established by RUNNING the cap arm, not by assuming it. A
45 / 180 / 360 s ladder on `nlopt grasp native` is the outstanding measurement before these two
rows are written up as anything at all.

## The chart ladder's pole screens: an OOD-only anomaly (2026-09-27)

All three `soft12` rungs finished 620k steps (`n6` job 5732986, `n4` 5733055, `n8` 5733056,
exports rc=0). The two screens disagree by eight orders of magnitude on one rung and agree on
the other two, which is worth recording because it looks like a bad chart and is not.

| rung | gain ceiling | in-distribution `pole/max` | frac > 345 | in-training (OOD) `pole/max` | OOD frac > 1000 | `val_l2_error` |
| --- | --- | --- | --- | --- | --- | --- |
| `n4` | 2.2e4 | 7.99 | 0.0 | 40.1 | 0.0 | 5.34 mm |
| `n6` | 3.2e6 | 5.44 | 0.0 | **2.44e8** | **0.0198** | 4.04 mm |
| `n8` | 4.8e8 | 4.56 | 0.0 | 5.51 | 0.0 | 2.76 mm |

**On the in-distribution screen the ladder is clean and monotone**: `pole/max` falls 7.99 ->
5.44 -> 4.56 with depth, every rung is two orders below its own threshold of 345, and
`frac_gt_threshold` is 0.0 everywhere. Accuracy is the usual monotone dose curve, improving
with depth.

**The anomaly is confined to the in-training callback, and there it is non-monotone in the gain
ceiling**: `n6` ends at 2.44e8 with 2% of draws past 1000 while `n8`, whose ceiling is 150x
higher, sits at 5.51 and never climbed -- flat at ~5 from step 20k onward. `n6` was already at
7.6e6 at its first evaluation. A ceiling bounds a chart; it does not predict where inside the
bound one lands, and on this robot two rungs used almost none of theirs while the middle one
used most of its own.

**The resolution is the domain, not the chart.** CLAUDE.md's "two pole screens, two domains"
section already names the mechanism: the fork's callback draws position in the iiwa's box and
orientation INDEPENDENTLY from `RollPitchYaw(uniform(-pi, pi, 3))`, and `soft12` has no torsion,
so tip orientation is not free given tip position and an independently drawn orientation is
essentially never reachable. The callback is therefore evaluating all three charts almost
entirely out of distribution, and 2.44e8 is a statement about unreachable poses.

**Quote the in-distribution row.** The in-training curve stays useful as a within-run trend and
as the thing that would catch a chart diverging mid-run; its LEVEL is not a property of the
chart. Neither screen predicts cells in any case -- that is closed -- so this is a labelling
hazard, not a measurement one. Closing the gap means passing the domain into the fork's
callback, which cannot be done while jobs are queued, since staging is refused under a live
campaign.

### `soft16` confirms the diagnosis from the other direction (2026-09-28)

The OOD explanation above predicts something falsifiable: a rung WITH torsion has a tip
orientation that IS free given its tip position, so the callback's independent orientation draw
lands in distribution and the callback should read clean from the first evaluation. `soft16`
activates `kappa_z` and is that rung. Job 5733058, 620k steps, export rc=0 at 05:29.

| screen | `pole/max` | frac > threshold | p50 | p99 |
| --- | --- | --- | --- | --- |
| in-distribution (`pole_at_task_poses.py`, threshold 345, radius 5.5) | 3.49-3.66 | **0.0** | 1.12-1.16 | 1.87-1.97 |
| in-training callback (OOD for a torsion-free arm, in-distribution here) | **2.85** | 0.0 | 0.94 | 1.96 |

**The prediction holds: 2.85, flat from the start, against `soft12_n6`'s 2.44e8 on the SAME
callback with the SAME constants.** The rung whose kinematics match the callback's assumption is
the rung that screens clean, which is the mechanism confirmed rather than merely re-asserted --
and it is confirmed by a rung that was not designed to test it.

Two consequences worth keeping. The in-training callback's LEVEL is now demonstrated to be a
statement about the ROBOT's reachability under the callback's sampler, not about the chart, so it
must never be compared across rungs of differing strain bases. And `soft16` screens cleaner than
`soft12` in distribution too (3.5-3.7 against 5.44), which lines up with what the benchmark then
showed: **no rung of this robot has a runaway population, on any screen, and the chart ladder is
correspondingly flat** (`docs/soft-arm-ladders.md`). The gain-ceiling criterion is satisfied
vacuously here; the screens were never going to separate anything.

**What the ladder does NOT yet say** is anything about cells. `n4` and `n8` are trained and
exported but not benchmarked; the fielded rung stays the pre-registered `n6`, and stage
SOFTCHART is what turns the ladder into a measurement. Selecting a rung on these screens would
be selecting on a criterion the project has already measured to be uninformative.

## Run record: `soft9_n6`, the first DOF-ladder rung (2026-09-27)

Job `5733057`, 14:00:16 wall on 4 nodes x 8 ranks, 620k steps at 11.9 steps/s, export rc=0 at
14:32:44, 30 checkpoints with sidecars. `val_l2_error` 3.23 mm. `soft16_n6` (`5733058`) took the
nodes automatically on the `afterany` dependency.

| screen | `pole/max` | `frac > thr` | p50 | p99 |
| --- | --- | --- | --- | --- |
| in-distribution (radius 4.5 = sqrt(9)+1.5, threshold 345) | 5.80 | 0.0 | 1.01 | 2.27 |
| in-training callback (OOD) | 13.09 | 0.0 | 1.17 | 4.88 |

Median tip error over held-out poses is **2.02 mm**, the most accurate rung in the push so far
(`soft12_n6` is 2.74 mm). Fewer DOF, less redundancy, an easier map -- and by the record's own
finding that accuracy runs BACKWARDS to cells, this predicts nothing about the benchmark.

**A mid-training transient that must not be read as a result.** At 60k steps this rung's
in-training (OOD) screen stood at `pole/max` 5.6e7 with `frac_gt_1000` 0.011, which looked like
`soft12_n6`'s excursion reproducing on a second `nb_nodes=6` run. It is not: by 620k the same
screen reads 13.09 with `frac_gt_1000` **0.0**. The excursion resolved during training.
`soft12_n6` remains the only rung that FINISHES high on the OOD screen, and the general lesson
is that the in-training callback's level is not even monotone within a run, which is a further
reason to quote the standalone screen and read the callback only as a trend.
