# GVS push-rod arm: the run's own record

What is queued, what it depends on, and what a resuming session should check first. The
design and the measured facts live in `CLAUDE.md` and `docs/gvs-arm.md`; this is the
operational half. Branch `gvs-actuated-arm`; cluster tree `~/learned-ik-gvs` (its own, per
the two-tree rule: the default `~/learned-ik` carries Thomas's soft PCS arm jobs and must
not have its venv or code changed under them).

## STATE AS OF 2026-09-30 12:45 -- FIRST DATASET DONE, second building

| item | state |
| --- | --- |
| spec, model, SDF/scenes, shim, programs, driver, tests | done, all passing locally |
| stage `GVS` in `cluster/gen_manifest.py`; `cluster/manifest_stageGVS.txt` | generated (64 items), never submitted |
| cluster tree `~/learned-ik-gvs` | created; Drake, sysdeps and the drake_models cache symlinked read-only from `~/learned-ik`; its own venv built on `download` (job 5779961, `setup.DONE: OK`: torch cu126 with sm_70, jax 0.11.2 CPU, soromox 0.5.0, optimistix) |
| staged commit | `79e7c00` |
| preflight `gvs_cal_preflight` (5780312, debug-cpu) | `PREFLIGHT OK`: 259 positions / 52 bodies, warm-up 15.8 s, PlantQ 20.1 ms, PlantJacobian 27.5 ms, 500-sample batch 18.4 s |
| rate `gvs_cal_rate` (5780313, xeon-p8) | 14.9 ms/sample (batch 2000) and 11.9 ms/sample (batch 20000) single-process with XLA unpinned on 96 logical CPUs -- the batched JAX solve does not spread across cores; hence the process-parallel builder. Cancelled as stale once that was known |
| datasets `gvs_pushrod9_o1`, `gvs_pushrod9_o2` | **`o1` COMPLETE** (5781386, `.DONE` at 2026-09-30T12:38): 25,000,000 train + 15,000 test, 3 h 6 min, **436.8 us/sample over the node** (21.0 ms/sample per worker x 48), peak 99.5 GB of 192, `rejected_unconverged` **0** on every draw; downloaded to the laptop's ikflow cache and verified (unit quaternions to 1.2e-7, 64 rows re-solved against the stored endpoints to 6.3e-8, the float32 storage floor). `o2` RUNNING as 5781387 since 12:38, same settings, expected ~15:45. Attempt 4 (5781015) was OUT_OF_MEMORY at batch 4096 and attempts 1-3 died or hung at startup -- the traps below |
| charts `gvs_pushrod9_o{1,2}_n6` | NOT trained -- out of scope (behind the screw-arm queue and the PCS cleanup) |
| FK surrogate (`--fk learned`) | NOT fitted -- a cluster job, not started |

**The launch, verbatim** (what was run; steps 0 and 3 below):

    SC_ROOT=learned-ik-gvs bash cluster/stage_code.sh
    # then, on the login node:
    cd ~/learned-ik-gvs/repo && LEARNED_IK_ROOT=$HOME/learned-ik-gvs DATASET_SIZE=25000000 \
        WALL=12:00:00 CPUS=48 bash cluster/chain_datasets.sh gvs_pushrod9_o1 gvs_pushrod9_o2

`CPUS=96` is refused: Slurm counts 48 CPUs on an xeon-p8 node (the 96 the rate job printed
are hyperthreads); the builder sizes its workers from the job's cpuset (48 workers) and then
caps that by the node's memory. The 12 h wall is deliberately generous because no cluster
per-worker rate is yet on record -- attempt 4 died before printing one. READ THE FIRST
PROGRESS LINES ~15 min in ("worker k: n/N at t s (x ms each), RSS y GB") and project
`521k x ms_each` per worker; cancel and resize if that exceeds the wall. Sentinels: `~/learned-ik-gvs/home/.cache/ikflow/datasets/gvs_pushrod9_o*/.DONE`.
`DATASET_SIZE=2000000` (LOInK's) would take minutes if a smaller set is ever wanted.

## Order of operations, and why it is this order

0. **Stage the isolated tree** (login node is fine; rsync only):
   `SC_ROOT=learned-ik-gvs bash cluster/stage_code.sh`. Prints a NOTE that the
   live-campaign refusal is scoped to the default tree; nothing queued out of
   `~/learned-ik` reads this one.
1. **Build the tree's environment as a job on `download`** (the only non-login partition
   with internet; `MaxJobs=1`, so check nobody else has one queued first):
   `LEARNED_IK_ROOT=$HOME/learned-ik-gvs LLsub ./cluster/setup_supercloud.sh -s 8 -q download`.
   Drake and `sysdeps` can be shared from `~/learned-ik` by symlink beforehand
   (`ln -s ~/learned-ik/drake ~/learned-ik-gvs/drake; cp ~/learned-ik/.drake-ok ...`; same
   for `sysdeps`) -- same project, same pin -- and the script skips them; the venv is this
   tree's OWN, because it gains `jax[cpu]` + `soromox` + `optimistix` and the default tree's
   venv must not change under running jobs.
2. **Preflight, on debug-cpu** (a smoke test: imports, the scene, one equilibrium, one
   sampler batch; produces no records):
   `LEARNED_IK_ROOT=$HOME/learned-ik-gvs ROBOT=gvs_pushrod9_o1 LLsub ./cluster/preflight_root.sh -s 8 -q debug-cpu -T 00:20:00 -J gvs_cal_preflight`
   and read `PREFLIGHT OK` in its log.
3. **Datasets, ONE AT A TIME, on the CPU partition** -- this is the step that waits for the
   go-ahead:
   `LEARNED_IK_ROOT=$HOME/learned-ik-gvs bash cluster/chain_datasets.sh gvs_pushrod9_o1 gvs_pushrod9_o2`
   (a Slurm `afterok` chain: exactly one build runs at a time, because ikflow's end-of-run
   summary reads every dataset in the shared cache and a sibling's half-written tensor
   makes a finished build exit 1 without its `.DONE`). Set `DATASET_SIZE` from the measured
   rate (below) and `WALL` to cover it; the sampler solves an equilibrium per draw, so it is
   NOT the soft PCS arm's 11 minutes.
4. **Charts** (out of scope here): `ROBOT=gvs_pushrod9_o1 bash cluster/submit_train.sh
   gvs_pushrod9_o1_n6 4 -- --nb_nodes=6 --dim_latent_space=9`, then `_o2`; the in-job
   export writes `models/<rung>/<rung>__n6__step620000.pkl` and the screens run.
   `ikflow_entry.py` retargets the pole screen to in-distribution poses for these rungs
   automatically. Measured (`docs/gvs-arm.md`): orientation given position IS 3-dimensional
   on this robot despite the absence of torsion, but it does not cover SO(3), so the
   in-distribution screen stays the safe choice.
5. **Benchmark stage** (out of scope here): the committed `cluster/manifest_stageGVS.txt`
   (64 items: two rungs x two experiments x two protocols x 8 shards, IPOPT, 180 s), which
   `gen_manifest.py --selftest` checks against the stage definition; regenerate ONLY with
   `python cluster/gen_manifest.py --stage GVS --wall-time 180 --targets 60 --guesses 8
   --shards 8 --solvers ipopt --starts paired,native -o cluster/manifest_stageGVS.txt` -- the
   generator's CLI defaults belong to other stages and once produced a SNOPT-only,
   paired-only file. Then `submit_bench.sh manifest_stageGVS.txt 4`, which refuses while
   the checkpoints it names do not exist. IPOPT only; the solver axis is closed.

## The datagen rate, and what it sizes

Every sample is a Newton solve on SoRoMoX's rod, so the rate is measured, not assumed.

| condition | us / sample |
| --- | --- |
| laptop, 2 XLA threads, idle, batch 2000 | ~3,400 |
| laptop, 8 XLA threads, under a peer's 16-worker pool, batch 2000 / 20000 | 17,550 / 19,330 |
| **cluster xeon-p8, XLA unpinned, ONE process, batch 2000 / 20000** | **14,900 / 11,900** |
| laptop, 4 single-threaded worker PROCESSES (contended), 6000 samples | ~3,800 effective (~15,000 per worker) |
| laptop, ONE worker on 2 CPUs under load 17, batch 4096, TWO solves per draw (before 2026-09-30) | 23,500 per worker |
| laptop, ONE worker on 2 CPUs under load 17, batch 1024, one solve per draw | **11,200 per worker** |
| **cluster xeon-p8, 48 pinned workers, batch 512, 25M samples (the fielded build)** | **437 over the node; 21,000 per worker** |

Every row above the last two predates the single-solve sampler and counts two solves per
draw.

The batched JAX solve does NOT spread across a node's cores, and `jax.pmap` over host CPU
devices is refused by lineax inside optimistix's root-find (`pytree does not match
out_structure`). So `scripts/gvs_arm/build_dataset_parallel.py` runs one single-threaded
JAX per CPU of the job's cpuset and writes ikflow's exact files; `build_dataset_job.sh`
routes `gvs_*` robots to it. At 48 workers and the laptop's per-worker 5-11 ms/sample the
effective rate would be 100-230 us/sample, i.e. 25M in 0.7-1.6 h; **no cluster per-worker
number is on record yet**, which is what the progress lines now produce.

**MEASURED, 2026-09-30**: 436.8 us/sample over the node, 21.0 ms/sample per worker, so 25M
takes **3 h 6 min** on one xeon-p8 node. A 6 h wall is therefore ample and the 12 h used for
attempt 5 was belt-and-braces; `rejected_unconverged` was 0 on all 25,015,000 draws. `rejected_unconverged` was 0 on every draw so far
(22,000 on the laptop, 500 in the preflight, 7,500 in the local builder test). The build's
own log prints the measured us/sample over the node; record it here when the first `.DONE`
lands.

## What a resuming session should check FIRST

- `squeue -u $USER` for anything of ours (`lik_ds_gvs_*`, `gvs_cal_preflight`).
- `~/learned-ik-gvs/home/.cache/ikflow/datasets/gvs_pushrod9_o*/.DONE` -- the sentinels.
- `~/learned-ik-gvs/repo/.staged-commit` against `git rev-parse HEAD` on `gvs-actuated-arm`.
- That the default tree `~/learned-ik` was not touched: its venv has no `jax`.

## Traps specific to THIS robot

- **The forward model is a root-find.** `Equilibrium` raises on non-convergence in the
  solver; the sampler rejects and COUNTS such draws (`robot.rejected_unconverged`). Measured
  0 of 2000 locally; a nonzero count on the cluster is a finding, not noise.
- **XLA threads.** Unpinned, JAX takes every core. `run_items.sh` and `preflight_root.sh`
  set `GVS_ARM_XLA_THREADS=1` per worker; the dataset job leaves it unset on purpose.
- **Quaternion sign.** Body quaternions are canonicalised to `w >= 0`; the same pose either
  way, but a finite-difference check on the raw plant vector must compare poses.
- **The tip frame is pitched.** SoRoMoX's backbone is local x; `gvs_tip` is `R_y(+90 deg)` on
  the tip body so its z runs along the rod. The flow is conditioned on the FRAME
  (`TipPose`), not the body.
- **No chart ships.** The driver refuses without `--checkpoint`;
  `scripts/gvs_arm/make_untrained_chart.py` writes a gitignored untrained one for smoke runs.
- **A process-parallel build needs one thread per worker AND a raised process limit.**
  The first cluster build spawned 96 workers whose OpenBLAS and torch OpenMP pools each
  tried to create a thread per core; the node's soft `RLIMIT_NPROC 4096` ran out during
  `import numpy`, the pool kept respawning dying workers, and the job sat "RUNNING" writing
  a 10 MB log of `pthread_create failed` while producing nothing. Pinning
  `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` and `GVS_ARM_XLA_THREADS=1` was not enough: the
  second attempt died the same way at `GetPjRtCpuClient`, because every JAX process still
  creates XLA's per-core Eigen pool (idle) and its compiler threads, ~100 per process, and
  no jax-0.11 flag turns that off. The hard limit is ~770k, so `build_dataset_job.sh` raises
  the soft process and open-file limits for `gvs_*`, and the builder caps its worker count
  by the soft limit so a job that cannot raise it runs slowly instead of dying. A cluster
  job's `nproc` prints 1 under `OMP_NUM_THREADS=1`; the builder reads `sched_getaffinity`.
- **Unpinned JAX workers hang the node.** The third attempt (5780456) ran 96 workers with
  no affinity: XLA sizes its per-process Eigen and compiler pools from the CPUs the process
  may run on (jax 0.11 ignores the `intra_op_parallelism_threads` flag), so every worker
  grew ~420 threads, the node was oversubscribed ~10x with ~184 GB RSS, and after 3.8 h it
  sat entirely idle -- load 0.11, every worker's main thread on a futex, the parent waiting
  on `pool.map`, no output -- until cancelled. Slurm's accounting (`sstat`/`sacct`) showed
  frozen CPU time throughout, so it cannot tell a hang from work; `ssh <node>` and look. The
  builder now pins each worker's affinity to its own CPU slice BEFORE JAX loads (one
  worker per physical core by default, `DATASET_WORKERS` overrides), streams results with
  a per-worker timeout, and prints a line per finished worker so the log shows progress.
- **The vmapped solve's PEAK memory scales with the batch, and 48 x 4096 exceeds a node.**
  The fourth attempt (5781015) ran 48 pinned workers at batch 4096: the test set (batch
  capped at its 312 draws per worker) passed in 168 s, then the training set's first
  4096-lane compile ran and 14 workers were OOM-killed at a node-wide 192 GB
  (`sacct` state `OUT_OF_MEMORY`, `MaxRSS` 191.8 GB, `TotalCPU` 188 h over 3.1 h -- it was
  computing, not hung). The parent saw nothing: a killed worker never returns, so
  `imap_unordered` waited out the 3 h per-worker timeout and raised `TimeoutError` with no
  rate on record. Measured on the laptop with the kernel's high-water mark: 0.96 GB after
  import, **2.38 GB at batch 512 and 2.73 GB at batch 1024** -- about 2.05 GB of process
  and jitted code plus 0.7 MB per lane, which extrapolates to 4.8 GB at 4096 and
  48 x 4.8 = 230 GB on a 192 GB node, exactly what happened. The builder now defaults to
  batch 512 (~2.4 GB peak per worker, 48 workers ~115 GB of 192 GB), caps the
  worker count by `MemTotal`, prints the projection, and every worker reports its
  progress, rate and RSS every ~10 minutes so a slow or growing worker is visible in the
  log long before any limit. The dependent `o2` job (5781016) sat `DependencyNeverSatisfied`
  and was cancelled by hand: an `afterok` chain does not clean up after a failed parent.
- **The sampler solved every draw TWICE** -- once for the tip pose and once for the
  self-collision screen's sphere centres -- until 2026-09-30. `TipAndCentresBatch` returns
  both from one solve; measured on the laptop 23.5 -> 11.2 ms/sample per worker
  (`tests/test_gvs_arm_model.py::test_sampler_path_is_one_solve` pins the agreement).
- **Two training-tooling hazards, reported by the screw-arm session (its runbook has the
  wording), not yet fixed in the shared scripts.** `status.json`'s `val_l2_error` (what
  `submit_ladder.sh --status` and `watch_ladder.sh` print) is the UNCLAMPED validation mean
  and is tail-driven -- one out-of-limits sample moves it an order of magnitude; read
  `val_clamped/l2_error` in `results/train/<rung>/metrics/version_0/metrics.csv` instead. And
  an export log's LAST screen is not the final checkpoint: the export job iterates
  checkpoints lexicographically, so `step80000` sorts after `step620000`.
