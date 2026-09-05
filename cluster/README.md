# Running the `learned-ik` benchmarks on MIT SuperCloud

Everything here mirrors the conventions of `../../codebase/cluster/`, which is the
reference implementation from the sibling project's campaign. Read
`~/.claude/skills/supercloud/SKILL.md` first — it carries the standing rules, and
they override anything convenient.

**If you are even remotely unsure about a cluster action, stop and ask Thomas.**

## The four rules this layout exists to satisfy

1. **Environments stay separate.** `~/ik-tune` on the cluster is a *different*
   project's tree — its own venv, its own Drake, its own extracted-deb prefix —
   and it has jobs running. Nothing here reads, writes, or symlinks into it.
   This project lives entirely under `~/learned-ik`, and `rm -rf ~/learned-ik`
   undoes all of it.
2. **Timing is never compared across machines.** The wall-clock cap stays as the
   measurement, but its *value* is recalibrated here; the laptop's 20 s and 45 s
   carry no meaning on a Xeon Gold 6248. Every arm of a comparison is measured on
   the same machine, and `metadata.host` / `metadata.device` are recorded so a
   cluster run cannot be paired cell-for-cell against a laptop one by accident.
3. **Debug partitions and login nodes take quick smoke tests only.** `smoke.sh`
   is the only thing that goes to `debug-gpu`. The calibration is a real
   experiment and runs on `xeon-g6-volta`; the multi-GB installs run as a job on
   `download`.
4. **The queue is laddered short → long**, so a defect costs minutes rather than
   a night.

## Allocation, and why you do not need to hand-throttle

`sacctmgr` gives this account `GrpTRES node=4` on `xeon-g6-volta` with
`MaxJobs/MaxSubmit = 240`. That is a **group** cap over everything the account is
running in the partition, not a per-job cap — so:

- **Surplus jobs queue, they are not rejected.** Submit twenty one-node jobs and
  four run while sixteen sit `PENDING (AssocGrpNodeLimit)`, starting as earlier
  ones finish. Submit a whole stage and let Slurm meter it.
- **The cap is shared with everything else the account runs**, including another
  project's jobs and another agent's. Check `LLstat` for what is already running,
  not just `LLfree` for what the cluster has free, and say in the plan how many
  nodes a campaign intends to hold.

## Layout on the cluster

```
~/learned-ik/
    drake/       own Drake v1.56.0 noble tarball (sha256-verified)
    sysdeps/     own dpkg-deb -x of libfmt9 + libspdlog1.12 (runtime only)
    venv/        own cp312 venv: torch cu126, ikflow, jrl, ...
    home/        a fake HOME holding the pre-warmed ikflow + jrl caches
    repo/        rsync of this tree (+ .staged-commit)
    state/<manifest>/   <id>.claim, <id>.done, <host>_p<n>.SENTINEL, logs/
    results/     per-run summary.json
    calib/       calibration logs
```

## Order of operations

```bash
# 0. stage the code and the gitignored iiwa checkpoint (small; login node is fine)
cluster/stage_code.sh

# 1. build the environment -- as a JOB on `download`, which is the only non-login
#    partition with internet. ~7 GB. MaxJobs=1 there, so never queue two.
ssh tcohn@txe1-login.mit.edu 'cd ~/learned-ik/repo && LLsub ./cluster/setup_supercloud.sh -s 8 -q download'
cluster/collect_results.sh --status          # wait for setup.DONE == OK

# 2. smoke, on debug-gpu, minutes. The decisive check is a real kernel launch.
ssh ... 'cd ~/learned-ik/repo && LLsub ./cluster/smoke.sh -g volta:2 -s 40 -q debug-gpu -T 0:20:00'

# 3. calibration -- four REAL jobs on REAL nodes (see calibrate.sh's header)
for ARM in gpu-procs cpu-procs caps parity; do
  ssh ... "cd ~/learned-ik/repo && CALIB_ARM=$ARM LLsub ./cluster/calibrate.sh \
           -g volta:2 -s 40 -q xeon-g6-volta -T 3:00:00 -J lik_cal_$ARM"
done
# -> gives (workers-per-node, device, cap). Record the table in CLAUDE.md.

# 4. a stage of the campaign
python cluster/gen_manifest.py --stage A --wall-time <cap> --shards <N> \
       --procs <K> --summary -o cluster/manifest_stageA.txt
cluster/stage_code.sh                        # push the manifest up
# submit one job per node; extras queue behind the 4-node cap
ssh ... 'cd ~/learned-ik/repo && MANIFEST=cluster/manifest_stageA.txt PROCS=<K> \
         LLsub ./cluster/run_items.sh -g volta:2 -s 40 -q xeon-g6-volta -T 12:00:00'

# 5. mop up, then collect (cluster storage is NOT backed up)
cluster/collect_results.sh --status
cluster/collect_results.sh --reclaim manifest_stageA   # only once the queue is idle
cluster/collect_results.sh                             # incremental since the last success
cluster/collect_results.sh --full                      # ...or the whole results tree
```

## Keeping the file count down (this is a policy requirement, not an optimisation)

SuperCloud's guidance is explicit: prefer fewer, larger files (1 MB minimum, ~100 MB
target), under 1000 per directory. Lustre is metadata-op bound, so tens of thousands of
small files are slow in a way that has nothing to do with their size.

This campaign drifted badly out of compliance before anyone noticed. `src/benchmark.py`
wrote one ~20 KB IPOPT log per (cell x arm) directly into the run directory, and by
Stage G there were **35,596 of them — 87% of every collection's file count** and 78% of
its bytes. A routine collection had gone from three minutes to **thirty**, and it got
worse with every stage. The symptom looked like network contention and was not.

What keeps it in compliance now:

- **Per-cell solver logs go to node-local `$TMPDIR`** and are rolled into one
  `solver_logs.tar.gz` per run at the end of `run_grid`. Recover an individual log with
  `tar xzf solver_logs.tar.gz learned_3_2.txt`. Never write per-cell files straight into
  a run directory on the shared filesystem.
- **Collection is incremental** — `.last_collect` on the cluster is the watermark, and it
  advances only after the local extract and merge succeed, so a failed transfer is
  retried in full rather than silently skipped. `state/` is not shipped at all.
- **`cluster/compact_logs.sh`** fixes runs that predate the change, as a debug-cpu job
  (compression is a job's work, never the login node's). `--dry-run` and `--status` are
  read-only ssh.

Measured effect: a full collection went from 40,733 files / 950 MB / ~30 min to
2,964 files / 316 MB / **43 s**.

Note the compaction is done by **the benchmark job itself** — `run_grid` rolls its logs
up before it returns — so no separate pass is needed for new runs.
`cluster/compact_logs.sh` exists only to backfill runs made before that change.

### Retention: logs are deleted once mined

Thomas's rule: *"once obsolete or mined, they can be deleted, since we can always
recreate them and you have saved their knowledge."* Lots of files temporarily is fine;
keeping them forever is not. `cluster/prune_logs.sh` does it (`--dry-run` first,
`--cluster` to include the cluster's copy, which is deliberately not the default).

What is kept, and why the line falls there: **`summary.json` is the primary record** —
per-cell entries, the returned `q`, every binding's violation, and the statistics — and a
wall-clock-capped run would *not* reproduce it bit for bit, so it is not recreatable in
the way the logs are. A `summary.json.partial` with no completed sibling is the only
record of a crashed run and is kept for the same reason. The solver logs are traces of
solves the summaries already describe, and every table in `CLAUDE.md` is written up
before the logs behind it are dropped.

First application, 2026-09-03: local `results/` went from **845 MB to 95 MB** (370
summaries), and the Stage G aggregate still reproduces exactly from what remains.

**Any check that asks the cluster whether it is busy must be scoped to this project.**
The account is shared with Thomas's other campaigns, so `LLstat | grep -c RUNNI` refuses
whenever anything at all is running — it fired on an unrelated `run_matrix.sh`. Filter by
`squeue -u $USER -n run_items.sh`, and count `PENDING` as well as `RUNNING`.

## How work is claimed, and how to recover

`run_items.sh` claims an item by `mkdir "$STATE/<id>.claim"` — atomic on POSIX,
so a failed `mkdir` means another worker owns it — and touches `<id>.done` only
on success. Any number of workers, in any number of jobs, started at any time,
therefore drain one manifest cooperatively with no coordination.

This replaces the sibling project's golden-ratio rank stride, which is the right
answer when a fixed set of ranks all start together. Here they do not: jobs start
whenever the group cap frees a node, so there is no stable rank space to deal
into.

**Re-submitting the same manifest is the mop-up mechanism**, and one mop-up pass
should be budgeted per stage — a COMPLETED job does not mean all its subprocesses
ran. A `.claim` with no `.done` is a dead item; `--reclaim` clears exactly those,
and refuses to run while any job is still active.

## Sharding

`--shard K/N` splits a grid **target-major** (whole targets per shard). The grid
is drawn and hashed before any filtering, so a shard is bit-identical to those
cells of the unsharded run; `merge_shard_summaries.py` pools the records and
re-runs `summarise` over the whole grid, because every statistic in it —
`success_ci`, `solved_within_k`, `_mcnemar`, `_common_cells` — is shard-local and
could not be stitched.

`bash cluster/verify_sharding.sh` proves the round trip is a no-op, locally, in a
couple of minutes. **Run it after any change to sharding, the merger or the grid.**
It bounds the solves by `max_iter` rather than the wall clock on purpose: a
wall-clock-capped solve legitimately varies with machine load, so mixing that in
would make the test unable to fail informatively.

Sharding is only *legitimate* above one worker per node if the calibration says
so — the benchmark is wall-clock capped, so contending workers change what is
measured, not just how long it takes. `PROCS=1` is always safe.

## Traps that cost time here

- **`source /etc/profile` before `set -u`.** `/etc/profile.d/Z97-byobu.sh` reads
  an unset `LC_BYOBU` and kills the job with a one-line log.
- **torch cu128 cannot run on a V100.** PyTorch 2.11's cu128/cu129 wheels dropped
  sm_70. A wrong wheel imports fine, reports a CUDA device, and fails at the
  first kernel launch — so the smoke job launches a real kernel.
- **`jrl` truncates and rewrites `~/.cache/jrl/urdfs/<robot>_...urdf` at every
  `Robot()` construction**, then klampt reads it back. With several workers
  starting at once on a shared `$HOME` that is a truncate-while-reading race, so
  every worker gets its own `$HOME` under `$TMPDIR`.
- **Drake downloads `drake_models` lazily, inside `ProcessModelDirectives`.**
  Every mesh the scenes reference as `package://drake_models/...` comes from
  there, so a compute node fails with "Network is unreachable" *after* passing
  every other check. Setup warms it by building all three scenes; it must be
  warmed with the cluster's Drake, since the cache key includes the models commit
  that Drake version pins.
- **Slurm exports `CUDA_VISIBLE_DEVICES` as GPU UUIDs, not indices.** Pin a
  worker by selecting an entry from Slurm's own list, never by writing `0`/`1`.
- **Compute nodes have no internet.** The panda weights auto-download from GCS
  and must be warmed on the `download` partition first; ikflow computes its
  cache dir from `expanduser("~")` at import, so `HOME` must be set explicitly
  when warming.
- **Two GPUs take code paths one GPU never does.** `jrl.config._get_device()`
  polls nvml (via `torch.cuda.memory_usage`) to pick the least-used card, and
  short-circuits only when one device is visible — always true on the laptop,
  false on a node holding `-g volta:2`, where the ikflow import then dies with
  "nvidia-ml-py does not seem to be installed". Setup installs `nvidia-ml-py`
  and every job script pins `CUDA_VISIBLE_DEVICES` to a single device.
- **`ikflow` declares `Requires-Python "<3.12,>=3.10"` and pip believes it.** The
  cluster's only cp312 interpreter is 3.12.3, so pip refuses the install. The pin
  is stale rather than real — the laptop runs that same ikflow 0.2.0 on 3.12.3
  and imports it fine; it never surfaced there because the local venv was built
  with `uv`, which does not enforce `Requires-Python`. Hence
  `--ignore-requires-python` in setup, which makes the cluster match the laptop
  rather than depart from it.
- **A bare `wait` never returns in a script that does `exec > >(tee ...)`.**
  Bash counts the `tee` of a process substitution as one of its background
  children, and that `tee` cannot exit while it holds the script's stdout — so
  `wait` blocks forever *after* having already reaped every worker it was meant
  to wait for. `jobs -p` has the same defect. This hung three of the four
  calibration arms, each holding a GPU node and doing nothing, and it is
  invisible from the outside: the job stays RUNNING with a live bash and no
  compute. Collect worker PIDs with `pid=$!` and wait on those.
- **A contention or cap probe must run a workload that BINDS against the cap.**
  The first calibration measured the pose task, where a cell converges in ~74
  iterations and ~6 s; its iteration count is therefore identical at 10, 20, 45,
  90 and 180 s, and identical however contended the node is, because a converged
  solve takes the iterations it takes. Such a probe reports "no effect" whatever
  is true. The grasp task, where ~40% of cells exit at the cap, is the workload
  that makes throughput visible.
- **LLsub triples mode cannot run a multi-node job.** `"[N,1,40]"` generates a
  Slurm job ARRAY of N independent single-node jobs (`--array=1-N`), so the
  nodes never share an allocation — no common `SLURM_JOB_NODELIST`, no c10d
  rendezvous. Its generated wrapper also execs the payload directly (the exec
  bit is load-bearing, unlike `-s` mode where sbatch spools the script) and
  ends in a bare `wait`, which returns 0 whatever the payload did — a failed
  payload reports COMPLETED 0:0 with empty logs (smoke_1g, job 5544517,
  2026-09-05: chmod-less script, "Permission denied" lost in an unwaited
  process substitution). Multi-node work uses `#SBATCH` directives + `srun`
  via `cluster/submit_train.sh`'s generated launcher; in that directives mode
  LLsub DROPS `-g/-q/-T/-J`, so resources must live in the directives. The
  generated wrapper of any past job is recoverable:
  `sacct -j <jobid> --batch-script`.
- **`LLstat` shows placeholder resources (1 CPU / 4 G) for PENDING jobs.** That
  is not evidence of a bad submission; verify after it starts before killing it.
