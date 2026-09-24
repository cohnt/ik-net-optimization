# Soft arm: the run's own record

What is queued, what it depends on, and what a resuming session should check first. The
design and the measured facts live in `CLAUDE.md`; this is the operational half.

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
