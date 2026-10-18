# Running on MIT SuperCloud: project notes

The long form of a part of `CLAUDE.md`, moved here verbatim on 2026-10-18 to bring `CLAUDE.md` under
40k characters. `CLAUDE.md` keeps the verdicts and rulings; this file keeps the mechanisms, numbers and
history behind them. "Above"/"below" and section names refer to the old single-file layout, now split
across `docs/notes-*.md`. A reference elsewhere to "CLAUDE.md's ... section" means the matching notes file.

## Running on MIT SuperCloud (`cluster/`)

`cluster/README.md` is the playbook and `~/.claude/skills/supercloud/SKILL.md` carries the standing
rules; this is what a reader of *this* file needs. Allocation: **4 nodes on `xeon-g6-volta`**, each
40 Xeon Gold 6248 cores and 2x V100 32 GB.

**Timing is never compared across machines.** Thomas: *"There's never a need to compare wall-clock
(or really, performance in general) between laptop and cluster. But wall clock limits can be
adjusted on the cluster."* So the wall-clock cap stays as the measurement — no switch to iteration
caps for portability's sake — but its value comes from `cluster/calibrate.sh` on that hardware.
`metadata.host`/`metadata.device` exist so a cluster run cannot be paired against a laptop one. The
corollary: **CPU contention still corrupts the measurement**, so workers per node is a measured
quantity. So is **GPU** contention, which only the learned arm pays. Paper runs use one solve per
GPU (PROCS=2); development stages may use PROCS=8 with `MPS=1` (Profiling).

**`--shard K/N` is a no-op by construction.** It splits **target-major** — whole targets per shard,
never a target's guesses split — because `success_ci` bootstraps over whole targets and
`solved_within_k` counts restarts within one, and it appends `_shardKofN` to the tag (without which
two shards overwrite each other's `summary.json`). `cluster/merge_shard_summaries.py` pools the
records and **re-runs `summarise`** rather than stitching per-shard numbers, preserving arm order so
`_mcnemar`'s pair directions survive. `bash cluster/verify_sharding.sh` proves the round trip in ~2
minutes — **run it after any change to sharding, the merger, or grid construction.**

**Benchmarks run on the GPU partition, and that is settled** (Thomas, 2026-09-25: *"benchmarks
have to use GPU. We've tried this before."*). So benchmark jobs and chart training compete for the
same 4-node cap, and the lever when the cluster is saturated is **ordering the queued work**, never
relocating benchmarks to `xeon-p8`. Datagen is the exception and does not generalise: it is pure
CPU and `build_dataset_job.sh` runs on `xeon-p8`. **So are arms that never touch the GPU**
(Thomas, 2026-10-13): the analytic arm's IPOPT / SNOPT / NLopt stages run on `xeon-p8`
(`PARTITION=xeon-p8`; `submit_bench.sh` refuses `--compile`, CUDA graphs, svgd and learned arms
there), so their seconds come from another machine than the record's; only its AL-SVGD stage uses
the GPUs.

**Two cluster facts that shaped the design.** The account's `xeon-g6-volta` limit is a Slurm
**`GrpTRES` group** cap (`node=4`, `MaxSubmit=240`), not a per-job `MaxNodes`: work beyond it is
accepted and **queued**, so a whole stage is submitted at once and Slurm meters it — and the cap is
shared with everything else the account runs. Because jobs start at different times there is no
stable rank space to deal work into, so `cluster/run_items.sh` claims items with an atomic
`mkdir <id>.claim`. And **PyTorch 2.11's cu128 wheels dropped sm_70**, so the V100s need a cu126
build; the wrong wheel imports cleanly, reports a CUDA device, and fails only at the first kernel
launch, which is why `cluster/smoke.sh` launches a real kernel. Three smaller adaptations: Meshcat is
optional (`BuildEnv(meshcat=None)`), mug scenes are built only for the shard's targets, and
`hit_iteration_cap` is the counterpart to `timed_out` (`is_timeout` does not match IPOPT's "Maximum
Number of Iterations Exceeded").

The measured calibration — workers per node, GPU-vs-CPU, the cap ladder and process startup — is in
`cluster/README.md`; it is a property of that hardware, not of the project.

Three operational post-mortems live in `cluster/README.md` in more detail than a project record
needs, and the transferable lesson of each is stated there. **A shard set straddling any number of
collections merges normally** — `collect_results.sh` passes `--also` for every prior staging
directory — so collect when you want results and act only on an actual `INCOMPLETE`; the old advice
here to "collect less often than an item takes" was stale, and a stale MITIGATION costs more than a
stale diagnosis because it gets acted on. Per-cell solver logs go to node-local `$TMPDIR` and roll
into one archive, because 35,596 small files on Lustre took a routine collection from three minutes
to thirty. And **any cluster-wide check on a shared account must be scoped to this project's own
jobs, by JOB name** — the fix that filtered the payload script's filename instead meant `--reclaim`'s
guard was unconditionally 0 and never refused for its whole life, so **a guard nobody has observed
refusing has not been tested.**

**The laptop does NOT suspend on AC**, and no run here needs a sleep inhibitor. Thomas, 2026-09-28,
having checked the power settings: *"I just went into my settings and confirmed that my computer
doesn't sleep on AC. So no need to worry about manually preventing it from sleeping."* A multi-hour
gap in a session is him closing it deliberately or hitting a usage cap -- neither is a fault, and
neither is diagnosed or mitigated. Long benchmarks run on the cluster
anyway, where a laptop's state is irrelevant.

**Local compute must never take the laptop down (Thomas, 2026-10-09: *"If a pytest sweep crashes my
machine, that is unacceptable"*).** On 2026-10-09 at 16:02 the svgd solver's Drake collision PROCESS
pools -- one per cached BatchedProgram in a test file plus 20-worker pools in benchmark subprocesses,
run concurrently by a subagent -- reached 81 worker processes and 44.8 GB on a 62 GB machine with no
swap, and the kernel OOM killer took his Slack. **The pool is deleted outright** (Thomas, the same
evening: the only parallelism is the GPU over particles and, behind it, Drake's own C++ threads). The
collision row is evaluated IN PROCESS on the program's own `MinimumDistanceLowerBoundConstraint`
(`src/svgd/collision_backend.py`, `CollisionEvaluator`), serially, ~0.3 ms per particle -- 92-96% of a
step at N = 64-256 -- until Drake gains a batched, GIL-releasing
`CollisionChecker::CalcRobotClearances` (brief: `~/Downloads/drake-CalcRobotClearances-plan.md`;
locally pydrake comes from his own build at `~/opt/rlg/drake-build`, so it lands here before any
nightly). Rules that survive, in practice and in briefs: no process pools in this project; every local
test file, smoke or end-to-end run that can build a program runs ONE AT A TIME, in the foreground,
under `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0` (the 8 GB of swap is his
emergency buffer and is never counted or used), so a runaway kills the run and not his applications;
and loadavg is logged before and after every timed run, any run at load above 24 re-run.

Two details of that paragraph survive it, being about process handling rather than power: detach a
long local process with **`setsid`, not `nohup`** -- `nohup` only ignores SIGHUP, so a teardown group
kill takes the process *and* anything it was guarding -- and `pgrep -f <script>` run from a Bash tool
call often matches **the calling shell's own command line**, so a dead process reads as alive.

Reproducibility at the cap: the solver axis's stage STEP section.

**Plan around SuperCloud's monthly maintenance** — second Tuesday, compute down Monday evening to
Wednesday morning, nothing survives it. Next window: 2026-10-12 to 10-14.

