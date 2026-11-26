# Engaging runbook (MIT ORCD, GPU benchmarks)

Scripts in `cluster/engaging/`; mirrors the SuperCloud flow (`cluster/README.md`). Read the
`engaging-cluster` and `compute-allotment` skills first. **Logging in is Thomas's act**: every script
uses the skill's one existing connection (`eg_run`/`eg_rsync`) and stops with exit 255 when it is down.
Never `eg_open`, never retry a login, never bare ssh.

## Layout on the cluster (`~/lik-engaging`, not `~/learned-ik`)

    image/lik.sif (+ .sha256 .commit .pip-freeze.txt)   immutable; built on a compute node from cluster/engaging/lik.def
    home-cache/.cache/{ikflow,drake}                    warmed once by the plumbing job (internet is used there only)
    c-<12 hex of commit>/repo      the staged commit (immutable)     /state   claims, done markers, hardware.log
                                   /results  published results       /logs    job logs

One tree per commit: a requeued job re-reads the code it started with, restaging never disturbs a
running campaign. `~/learned-ik` is the old (2026-11-10) CPU attempt; this flow does not touch it.

## The image

`lik.def`: Ubuntu 24.04; **Drake 1.58.0 release tarball** (`DRAKE_*` variables at its top, checked
against both the pinned sha256 and the release's published `.sha256`); torch 2.14.0+cu126 and the pip
pins of `setup_supercloud.sh`; jrl at the workstation's commit; ikflow is the staged repo's vendored
fork on `PYTHONPATH`. `CollisionChecker.CalcRobotClearances` is in 1.58.0, so no crc venv is needed.
**cu126 has no sm_100: never request B200.** L40S (default) and H200 run it.
To move the pin edit only the `DRAKE_*` variables; the plumbing check reads them back.

## The sequence

```
bash cluster/engaging/stage.sh                 # clean commit only; manifest; cluster-side sha256sum -c
bash cluster/engaging/submit.sh build          # mit_normal, CPU, 1 h; polls: poll.sh wait <id>
bash cluster/engaging/submit.sh record-sif     # reads build.DONE's sha256 into logs/staged/engaging-sif.sha256
ANY_CPU=1 bash cluster/engaging/submit.sh plumb        # 1 GPU: nvidia-smi, torch kernel, Drake pin, warm caches
bash cluster/engaging/submit.sh survey         # read-only: queue, GPU nodes, QOS limits, quota
NODES=... CPU_MODEL='...' bash cluster/engaging/submit.sh bench cluster/<manifest>.txt
bash cluster/engaging/poll.sh                  # one small call;  poll.sh wait <jobid>  loops every 5 min
bash cluster/engaging/collect.sh               # data-only allowlist rsync, merge, condense into results/
```

Every submit re-verifies the staged tree from the workstation (the cluster's own `sha256sum` against the
laptop's manifest) and hands the manifest hash to the job, which re-checks it, hashes the image against
the value anchored at build, and points `PYTHONPYCACHEPREFIX` at a fresh dir. **A STOP is read, never
bypassed**: read the listed files, fix the script, commit, restage, tell Thomas. Fixes ship as code:
never edit in the cluster tree.

## Hardware: a campaign never mixes it

Timing is never compared across machines, and within a campaign every arm of every item must run on
identical hardware. `submit.sh` therefore **requires** both pins (or `ANY_CPU=1`, smoke tests only):

- `GPU=l40s` (default) becomes `-G l40s:1`; the job refuses a node whose `nvidia-smi` name lacks `L40S`.
- `NODES=node4100-4107,node4200` becomes the node-name OR-constraint (node features are only names);
  `CPU_MODEL='EPYC 9474F'` is checked against `lscpu` in the job, which refuses any other model.
  Find a node group's model by running `plumb` with `ANY_CPU=1` and `NODES=<group>` and reading
  `HARDWARE ...` in its log (also `state/*/hardware.log`). GPU nodes of one type can mix CPU models.

Both are recorded in `state/submissions.log` (every submit, with commit, partition, allotment) and
`state/<manifest>/hardware.log` (every job: host, GPU, CPU model); `collect.sh` brings them home. Run
every part of a campaign with the same `GPU`, `NODES`, `CPU_MODEL`, `PARTITION` class and `COMMIT`.
(src/benchmark.py's own machine metadata is added on main separately.)

**One GPU type per job; there is no way to widen it.** GPU types on Engaging are Slurm gres types, not node
features, and gres has no OR over types. A type-less `--gres=gpu:1` is not "any GPU": the cluster's submit
plugin fills in `l40s` (`scontrol show job` reads `TresPerNode=gres/gpu:l40s:1`), so a node-name constraint
over A100/H100/H200 nodes only narrows the job to the L40S nodes inside it. To run on another type, submit
with that type (`GPU=h200`, `-G h200:1`) as a separate campaign pinned to it. Slurm also rejects a
constraint longer than about 1 KB, which limits a node-name OR list to roughly 100 nodes. The queue is the
cost: on 2026-11-25 a 1 x L40S `mit_preemptable` job sat PENDING (Priority) for over 3 h behind ~4,000
queued jobs, while a `mit_preemptable` image build (CPU) started within seconds.

## Long campaigns on `mit_preemptable`

`PARTITION=mit_preemptable` (4 GPUs per user, 48 h, `--requeue`, `--array=0-3%4`) or the default
`mit_normal_gpu` (2 GPUs, 6 h). **One solve per GPU, no MPS** (one array task = one GPU = `PROCS=1`),
the record's `--compile --set flow_cuda_graph=True` conditions. Tasks drain a manifest by atomic
`mkdir <id>.claim`, so any number of tasks cooperate and re-submitting the manifest is the mop-up.
`ITEM_TIMEOUT` is set to the walltime minus 25 min (max 8 h): run_items.sh never claims an item the job
cannot finish.

**The requeue trap.** A requeued job keeps its job id and restarts from the top of the script; its dead
attempt's claims have no `.done`, so every worker would skip those items. `run_items.sh` (env-gated:
`LIK_TAG_SUFFIX`; SuperCloud unchanged) writes the Slurm job id as the 3rd field of each claim's owner file (shared with CSAIL's `job.sh`), and
`bench_job.sh` calls `eg_release_own_claims` before draining: it removes exactly this job id's claims
that lack `.done`, never another job's. The item then resumes from its per-cell JSONL
(`LEARNED_IK_RESUME=1`). A job that died without a requeue (walltime, node failure, `scancel`) leaves
claims that nothing releases: `collect.sh --reclaim <manifest>` (refuses while any `lik_bench_*` job is
queued or running). Pending mop-up: resubmit the same manifest.
Test your own requeue only with `scontrol requeue <your job id>` on your own `lik_*` job.

## Rules in force

Job names `lik_*` (never touch `dfk_*`); cancel only your own ids; nothing computes on login nodes
(image build `mit_normal`; GPU work `mit_normal_gpu`/`mit_preemptable`); L40S first, H200 only if
needed; no credentials on the cluster (wandb offline); data only comes back, loaded with
`allow_pickle=False`, never run; a host-key change or any sign of compromise: stop and tell Thomas.
Maintenance: monthly, 3rd Tuesday (next reservation 2026-12-15 00:00): end jobs before 00:00 Tuesday; the
login nodes restart Mondays 07:00, which drops the connection (a new login is Thomas's).

## Tests

`python tests/test_engaging_scripts.py` (fast tier, ~12 s, no cluster): stage/submit/claims/hardware pin
against stubs. The first real run of everything above is recorded in `docs/notes-cluster.md`'s Engaging
section once it has happened.
