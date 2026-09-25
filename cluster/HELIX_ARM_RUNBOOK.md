# `helix7`, the helical-joint arm: the run's own record

What would be queued, what it depends on, and what a resuming session should check first.
The design and the measured facts live in `CLAUDE.md`; this is the operational half.

**NOTHING HAS BEEN SUBMITTED.** Every command below is written out so it can be launched
deliberately; none of them has been run. The whole branch was built and verified on the
laptop, which is what the "verified locally" lines below mean.

## Order of operations, and why it is this order

```
datagen (one rung at a time)        CPU partition (xeon-p8), no GPU used
        |
        v   train_flow.sh HARD-FAILS (exit 4) without the dataset's .DONE sentinel
chart ladder, sequential at 4 nodes:
        helix7_p050_n6   <- the FIELDED rung, pre-registered
        helix7_p050_n4       the chart ladder measurement
        helix7_p050_n8       the chart ladder measurement
        helix7_p000_n6       the pitch ladder's control
        helix7_p025_n6
        helix7_p100_n6
        |
        v   export + screening run INSIDE each training job, on node 0, on rc=0
benchmark stages: HELIX, HELIXCHART, HELIXPITCH
```

**Each pitch is a different robot**, so each needs its own 25M-sample dataset and its own
620k steps. The full ladder is 4 datasets and 6 training runs, comparable to the soft arm's
push. Narrowing to `{p000, p050}` is 2 datasets and 2 charts and still answers the pitch
ladder's control question, at one dose instead of three — a budget decision, not a design
one. `helix7_p000` cannot borrow another robot's chart: at pitch 0 it is still *this* arm,
with these links.

## Submitting

**ONE DATASET AT A TIME.** The soft arm measured the failure: ikflow's end-of-run summary
scans every dataset in the shared `~/.cache/ikflow/datasets/`, so a concurrent sibling's
half-written tensor makes a finished job exit 1 with its data already correct on disk. The
data is fine; the missing `.DONE` sentinel is not, and `train_flow.sh` hard-fails without it.

```bash
# datagen, one per rung -- from ~/learned-ik/repo on the login node, on the CPU
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
    helix7_p000_n6 helix7_p025_n6 helix7_p100_n6

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

1. `bash cluster/submit_ladder.sh --status` — global_step per rung against 620000.
2. `LLstat` — our jobs are named `lik_dataset` / `lik_train_<run>`.
3. That each dataset has its `.DONE` sentinel:
   `ls ~/learned-ik/home/.cache/ikflow/datasets/helix7_p050/`
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

## Still to build

* The cap ladder (45 / 180 / 360 s), before any verdict is reported. It needs trained
  charts, so it is the campaign's first methodological step rather than part of this
  infrastructure push.

The record's own section for this robot is written: **"The helical-joint arm: a robot no
algebraic method can chart"** in `CLAUDE.md`, which holds the design, the two traps and the
locally measured numbers. This file stays the operational half.
