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
cluster/collect_results.sh
```

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
- **Compute nodes have no internet.** The panda weights auto-download from GCS
  and must be warmed on the `download` partition first; ikflow computes its
  cache dir from `expanduser("~")` at import, so `HOME` must be set explicitly
  when warming.
- **`ikflow` declares `Requires-Python "<3.12,>=3.10"` and pip believes it.** The
  cluster's only cp312 interpreter is 3.12.3, so pip refuses the install. The pin
  is stale rather than real — the laptop runs that same ikflow 0.2.0 on 3.12.3
  and imports it fine; it never surfaced there because the local venv was built
  with `uv`, which does not enforce `Requires-Python`. Hence
  `--ignore-requires-python` in setup, which makes the cluster match the laptop
  rather than depart from it.
- **Quote the triples spec** if you ever use triples mode (`"[4,1,40]"`) —
  unquoted brackets are a bash glob.
- **`LLstat` shows placeholder resources (1 CPU / 4 G) for PENDING jobs.** That
  is not evidence of a bad submission; verify after it starts before killing it.
