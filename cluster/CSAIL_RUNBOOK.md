# Running learned-ik benchmarks on CSAIL's shared preemptible GPUs

Procedure only. The code is `cluster/csail/`; the platform facts live in the `csail-cluster` skill, which is read
FIRST (and `compute-allotment`). Findings belong in CLAUDE.md. **If even remotely unsure about a cluster action, stop
and ask Thomas.** The SuperCloud flow (`cluster/README.md`) is the model: same staging checks, same payload
(`cluster/run_items.sh`, unchanged except one field, below), same collection.

## Rules that shape everything here

- **All traffic goes through the skill's `cs_run` / `cs_rsync` over the existing connection** (`CS_DEST` from
  `~/.claude/csail-cluster.local.md`). Nothing here opens a login, uses bare ssh/scp/rsync, or touches a compute
  node. `cs_status` failing is a STOP: a login is Thomas's act, with Duo. Never retry one.
- **Never the lab node.** `locomotion-h200` / `tedrake-h200-1` never appear as a target; `vision-shared-h200` always
  carries `--exclude=tedrake-h200-1` (`csail_refuse_lab_node` enforces it; a test pins it).
- Variables go in front of `sbatch`, **never `--export`** (root cancels the job). Jobs run from `/data` with absolute
  `--output/--error`. Job names are `lik_*`, unique per submission; after an sbatch error check
  `squeue --me -n <name>` before resubmitting. No "claude" in any remote path or name. No credentials on the cluster.
- **Stage only committed code, never edit it there.** Group scratch can be modified by others, so the checks are
  mandatory: laptop manifest -> cluster `sha256sum -c` before every submit, the same again inside every job
  (`lik_job_check`), and the **image sha256** (recorded at build, anchored on the laptop) checked in every job.
- Copy back data only (`collect.sh` lists the archive and refuses links/odd paths before extracting).
- Local runs of the scripts' tests only (`python tests/test_csail_scripts.py`, stubs, no network, ~20 s).

## One-time: the image (Apptainer, built on CSAIL, as a job)

CSAIL's nodes are Ubuntu 22.04 (glibc 2.35); the Drake tarball is `noble` (24.04), so Drake cannot run natively.
`cluster/csail/learned-ik.def` builds Ubuntu 24.04 with:

- **Drake 1.58.0**, the `drake-1.58.0-noble.tar.gz` release tarball (Drake on `PYTHONPATH`, the SuperCloud layout),
  NOT the PyPI wheel and NOT a nightly (Thomas, 2026-11-25; the 09-18 nightly expired: its URL answers 403
  InvalidObjectState, storage class GLACIER). The version, URL, sha256 and commit are the `.def`'s `%arguments`, the
  one place the pin lives for the image; `build_image_job.sh` reads them there, compares the sha256 with the release's
  published `.sha256`, and the build refuses unless `VERSION.TXT` is `1.58.0 09bc913c...`.
  1.58.0 carries PR 25002 and `CollisionChecker.CalcRobotClearances`, so no separate crc venv.
- torch from the cu126 index (`TORCH_SPEC`, default 2.14.0, SuperCloud's measured build), and the exact pins of
  `cluster/setup_supercloud.sh` step 5 (the source of truth), `jrl` pinned to the laptop venv's commit, the vendored
  ikflow fork `--no-deps`, CPU jaxlib + soromox. `/opt/lik/freeze.txt` is the image's `pip freeze`, collected as
  `images/<name>.freeze.txt`: **diff it against SuperCloud's freeze** before trusting "same versions".
- Warm caches baked in (`/opt/lik/home/.cache`: the Panda ikflow weights, jrl URDFs, `drake_models`); `job.sh` links
  them into each worker's HOME, so no task downloads anything.

```bash
export CS_DEST=tcohn@slurm-login.csail.mit.edu          # username from ~/.claude/csail-cluster.local.md
git status --porcelain                                    # must be empty; commit first
git submodule update --init --recursive                   # third_party/ikflow (a worktree does not check it out)
bash cluster/csail/stage.sh                               # stages HEAD under $CS_ROOT/roots/<c12>/, verifies
bash cluster/csail/build_image.sh submit                  # ONE tig-cpu job (-A csail -p tig-cpu -q tig-main, 2 h)
bash cluster/csail/build_image.sh status <jobid>          # then, when .DONE says OK:
bash cluster/csail/build_image.sh record <image name>     # anchors the sha256 ON THE LAPTOP
```

`FAKEROOT=auto|yes|no` (default auto: `--fakeroot` only if `/etc/subuid` lists the user). **Not yet known**: whether an
unprivileged build (no subuid entry was visible on the login node) can run `apt` in `%post`; the build log records
the mode and the userns probe. If it fails there, the fix is committed to the `.def`, never patched on the cluster.
`APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` are pointed at `/data` scratch always (tig-cpu `/tmp` is tiny).

## The GPU plumbing check

```bash
GPU=rtx3090 bash cluster/csail/submit_check.sh <image name>
```
One preemptible job, 1 GPU, 20 min: tree and image hashes, machine record, `nvidia-smi`, `torch.cuda` (exit non-zero
if unavailable: the skill's trap 5), one real kernel, device name vs `EXPECT_GPU`, pydrake from the pin
(`VERSION.TXT`, IPOPT, `CalcRobotClearances`), project imports (ikflow is the vendored fork), the hardened Panda scene
builds. `check.<jobid>.DONE` says `OK` only if all of it passed. A faulty node is appended to `$CS_ROOT/bad_nodes.log`
and every later submit excludes it automatically; report job id, node, PCI bus ID to Thomas.

## A campaign: pinned hardware, many items, frequent preemption

```bash
GPU=l40s bash cluster/csail/submit.sh cluster/<manifest>.txt <image name> [tasks]   # MAX_RUNNING=<n> caps how many run at once (default: tasks)
bash cluster/csail/poll.sh [<c12>]                         # read-only; every 5-10 min at most
bash cluster/csail/collect.sh [<c12>]                      # data only, into results/ (merge + condense as SuperCloud)
```

- **`GPU` is required and pins the hardware**: `rtx3090 rtx4090 l40s a100 h200 csail-h200` set the partition
  (`vision-shared-<gpu>` with `-A locomotion`, or `-A csail -p csail-shared-h200`), `-q shared-if-available --requeue
  --open-mode=append`, the `--constraint` (`nvidia_l40s`, `nvidia_a100-sxm4-80gb`, ...), and `EXPECT_GPU`, which each
  job checks against `nvidia-smi` (fail closed). **A campaign never mixes GPU types**: the first submission writes
  `state/<manifest>/HARDWARE`, a later one with another kind is refused. Every arm of a comparison runs in the SAME
  campaign on the same kind. Prefer non-H200 unless needed; look at `sinfo`/`squeue` first (A100/L40S queues were
  150+ pending in October; the RTX 3090 queue is short).
- **CPU models can differ between nodes of one partition.** There is NO CPU constraint to pin: Slurm features here
  are the GPU, the datacenter (`holyoke`/`stata`) and `jammy` only (checked 2026-11-25, `sinfo -o %f`). Every attempt
  writes `logs/<jobid>/machine.a<N>.txt` (node, GPU, CPU model, image sha, commit); `poll.sh` shows them. Read them
  before comparing any wall clock; if they differ, narrow with `EXCLUDE_NODES=<comma list>` (the exclusion list can only
  remove nodes). CSAIL timing is never compared with SuperCloud's.
- One array task = one GPU = one worker process (`PROCS=1` forced: one solve per GPU; never >1/GPU). Defaults: 8 CPUs,
  48G, `--time=1-00:00:00`, `tasks = min(items, 8)` throttled to that many running. The paper-condition rules of
  CLAUDE.md stand: `--compile --set flow_cuda_graph=True` identical for every compared run (they are in the items).
- State the plan before submitting (account, partition, QoS, CPUs, memory, GPUs and type, walltime, count, name,
  preemptible): `submit.sh` prints it as `PLAN:`; Thomas clears submissions.

### Preemption, requeue, and the stale-claim trap (the normal case here)

GraceTime is 0: a preempted job gets SIGTERM and is killed, then (`--requeue`) re-queues under the **same job id**
after ~2 min and starts again from the top of `job.sh`. Progress survives because the benchmark appends each finished
cell to its JSONL on `/data` and resumes (`LEARNED_IK_RESUME=1`). The trap: `run_items.sh` claims an item with
`mkdir state/<manifest>/<id>.claim`, and nothing ever cleared a claim whose process died, so a killed attempt orphans
its items. Handling, in `job.sh`:

- `run_items.sh`'s owner file now ends with the Slurm job id (**the only shared-file change**).
- At the start of every attempt, `job.sh` releases claims with no `.done` that are owned by THIS job id (the old
  attempt is dead) or by a job that is not RUNNING (gone from `squeue`, or PENDING after its own requeue). A RUNNING
  owner's claim, and a claim with no job id, are never touched. Between rounds it releases only other jobs' dead
  claims, then runs `run_items.sh` again while unclaimed items remain (it walks the manifest once).
- SIGTERM trap: kills the worker subtree so no further item starts, copies this attempt's node-local worker logs to
  `logs/<jobid>/a<N>.worker_*.log` (also copied every 60 s, since the trap may never run), exits 143.
- `run_items.sh`'s own guard refuses an item the job's remaining wall cannot cover; inside the image `squeue` does not
  exist, so `cluster/csail/shim/squeue` answers that one query from `JOB_END_EPOCH`.
- An item re-run after preemption resumes from its JSONL; a cell killed mid-solve is simply re-solved.
- `tests/test_csail_scripts.py` simulates this end to end (stub Slurm and apptainer, real `run_items.sh`): SIGTERM
  mid-item, then the same job id restarts, releases its claim and finishes the item.
- **To test it for real: `scontrol requeue <your own job id>`** (only works on `--requeue` jobs; never anyone else's).
  Watch `sacct -D -j <id>` (REQUEUED, then the next attempt) and the log for `released stale claim`.

A task whose attempt has too little wall left for another item exits; the requeue or a sibling finishes the rest.
Re-submitting the same manifest is the mop-up (idempotent: done items are skipped). Do not cancel anything that is not
yours (`dfk_*` and everyone else's jobs are off limits): `scancel` only job ids this session printed.

## What each script does

| script | where it runs | what |
| --- | --- | --- |
| `stage.sh` | laptop | clean-tree commit -> manifest -> rsync (+ `--checksum` pass) -> `roots/<c12>/repo` -> verify |
| `build_image.sh` / `build_image_job.sh` / `learned-ik.def` | laptop / tig-cpu job | the image; `record` anchors its sha256 |
| `submit_check.sh` / `check_job.sh` / `gpu_check.py` | laptop / GPU job | plumbing check |
| `submit.sh` / `job.sh` / `job_lib.sh` | laptop / GPU array job | campaign; claims, release, logs |
| `poll.sh`, `collect.sh` | laptop | read-only status; data-only collection |
| `csail_common.sh`, `csail_verify.sh`, `csail_submit.sh`, `submit_lib.sh` | laptop | plumbing, checks, sbatch wrapper |

Layout on the cluster: `/data/locomotion/scratch/tcohn/learned-ik/{images,cache,bad_nodes.log,roots/<c12>/{repo,state,
results,logs,manifests,home}}`. One staged commit = one root, so restaging after a commit never changes code under a
running or requeued job. Only what the staging and build create is ever written there; nothing is read from older
trees.

## STOP conditions (report, never bypass)

Manifest or image hash mismatch; an unlisted code file or symlink in the tree; a changed host key; `cs_status` down; an
unexpected job, file or notice under the account; a `collect.sh` archive with links. Read what the check lists, fix the
script (committed), restage, and tell Thomas. The integrity checks of 2026-11-13 had only been tested against a stub:
expect a false STOP the first time and fix the script, do not loosen the check.

## Verified end to end (2026-11-26, branch cluster-csail)

Image build job 3203882 (tig-cpu, 1 h 56 min: apt ~25 min and pip ~40 min on NFS, mksquashfs ~2 h of that is the 4.5 GB SIF
write; allow 3 h): unprivileged, no `--fakeroot` (no `/etc/subuid` entry; apptainer 1.5.5 falls back to a root-mapped
namespace and runs `%post` under fakeroot). Check job 3204758 (RTX 3090, EPYC 7773X): torch 2.14.0+cu126, Drake 1.58.0
`09bc913c`, `CalcRobotClearances`, vendored ikflow, hardened Panda scene. Smoke 3204770 (2 items) and a `scontrol requeue`
of that same job: attempt 0 was SIGTERMed mid-item (logs saved), attempt 1 (same job id, ~9 min later on the 3090
queue) released the stale claim, resumed `csail_smoke_a` from its cell JSONL (`metadata.resume`: `records_loaded` 2,
`killed_cell` null) and finished both items. Lessons that shaped the scripts, so they are not repeated: Slurm runs a COPY
of the batch script (never locate the tree from `BASH_SOURCE`); `pip -c` belongs to `pip install`; roma/FrEIA need torch
installed first; ikflow is a namespace package; run the `%post` in docker (`--cpus=1 --memory=3g`) before any build.
