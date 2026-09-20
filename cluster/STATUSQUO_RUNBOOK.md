# Stage STATUSQUO: the campaign's own resume point

**Read this first if you are picking the campaign up cold** — after a pause, a session
restart, or a context compaction. It is the state of record; conversation context is not.
Keep the "Progress" section below current as steps complete, and commit it, so the next
reader needs nothing else.

## What this campaign is

The measurement that replaces every results table in `CLAUDE.md`, all of which predate the
2026-09-19 decisions: shelf-contained grasp targets (were free), shelf-contained pose at the
fingertips (the last open placement question, now decided), a **180 s** cap (was 45 s), and
adopted SNOPT and NLopt configurations (were Drake's defaults).

- 36 logical runs = 2 robots (Panda `n6`, iiwa `n4`) x 3 rows x 2 protocols x 3 solvers.
- 480 cells each (60 targets x 8 guesses), seed 1, `--compile`, hardened scene, inset 0.10,
  `correction_cost_weight=10`, arms `learned,numerical`.
- **480 manifest items**: IPOPT and SNOPT 8 shards per run, NLopt **24** (its cells nearly all
  run the full cap on both arms, ~44 h per logical run).
- Rows: `mugshelf` and `posetip` **are** the status quo; `mugfree` is a **legacy** column,
  included only because iiwa free grasp is unmeasured above 45 s. Never report it as status quo.
- No settings axis: each solver runs its adopted configuration, which is now its default. The
  selftest refuses any `--set` other than the correction penalty.

## Commands, in order

```bash
# regenerate the manifest (idempotent; the spec lives in cluster/gen_manifest.py)
python cluster/gen_manifest.py --stage STATUSQUO --wall-time 180 --targets 60 --guesses 8 \
       --shards 8 --starts paired,native --solvers ipopt,snopt,nlopt \
       -o cluster/manifest_stageSTATUSQUO.txt
python cluster/gen_manifest.py --selftest

cluster/stage_code.sh                    # refuses while a run_items/train job is live
cluster/submit_bench.sh manifest_stageSTATUSQUO.txt 4      # 4 jobs, 48 h wall, PROCS=8

# monitoring (safe to run any time, from anywhere)
cluster/collect_results.sh --status      # done vs claimed per manifest
source cluster/ssh_common.sh && sc_run 'LLstat'

# mop-up, ONLY once the queue is idle: clears claims with no .done, then resubmit
cluster/collect_results.sh --reclaim manifest_stageSTATUSQUO
cluster/submit_bench.sh manifest_stageSTATUSQUO.txt 4

# collect, merge shards, report
cluster/collect_results.sh
python cluster/merge_shard_summaries.py <staging>/results
python scripts/report_statusquo.py '<staging>/results/*/benchmark/sc_STATUSQUO_*/summary.json'
```

**Resuming is free and is the designed mechanism**: items are claimed with an atomic
`mkdir <id>.claim` and completed with `<id>.done`, so re-submitting the same manifest drains
whatever is left, from any number of jobs started at any time. A pause needs no action at all —
running jobs keep going, and nothing here depends on this session staying alive.

## Progress

Update as steps land. `PENDING` / `DONE` / `DONE (n/480)`.

- [x] **Local: plumbing + stage + reporter committed and pushed.** `f6685e8` (stage, caps, one
      Drake), `e2fe757` (emitted solver options + Drake stamp in metadata), `54bd528` (smoke
      covers SNOPT and NLopt). All four test files pass, `gen_manifest --selftest` OK,
      `verify_sharding.sh` OK, and the claim guard is verified in both directions.
- [x] **Cluster: one Drake install, and it is the pin.** `~/learned-ik/drake` is
      `0.0.20260918 5a73436c...`; 1.56.0 deleted. Install is relocatable (`$ORIGIN` RPATH), so
      this was a swap not a re-download. Both hardened scenes build offline under it, i.e. the
      `drake_models` cache resolves.
- [x] **Cluster: code staged.** `.staged-commit` = `e2fe757`, matching local HEAD at the time.
      Re-stage before submitting to pick up `54bd528`.
- [x] **Cluster: smoke check PASSED** on debug-gpu at the staged commit — cu126/sm_70 kernel
      launch, IPOPT and SNOPT available, flows and both robots load offline, one cell per
      (robot, task) with `--compile`, and `start_q_error` **exactly 0.0** on every arm of all
      four, which is the check that the flow inversion and conditioning-frame calibration are
      right on this machine and under this Drake.
- [x] **SUBMITTED 2026-09-19 ~21:47 ET.** Jobs 5681825-5681828, `lik_bench_manifest_stageSTATUS`,
      4 x `PROCS=8` = 32 concurrent workers, 48 h wall each (`TimeLimit=2-00:00:00` confirmed via
      `scontrol`). 32 items claimed within 90 s and the new "don't claim what the job cannot
      finish" guard is correctly silent, as it must be on a 48 h job.
- [ ] Items complete ( / 480)
- [ ] Collected and merged
- [ ] Reported, acceptance checks passed
- [ ] `CLAUDE.md` tables replaced

**Two things this campaign records that earlier ones did not**, both worth using when reading it:
`metadata["solver_options_emitted"]` is the options Drake was actually handed, per arm — which is
how you tell a run at today's adopted defaults from one at Drake's, since `overrides` records only
`--set` — and `metadata["drake_version_txt"]` is the exact build stamp and commit.

## Acceptance checks, all free

- **iiwa `n4` contained grasp under IPOPT must reproduce `sc_CAP_iiwa_n4_mug_180_{native,paired}`**:
  learned 447 / 453, joint space 442, **0 timeouts**, same `grid_hash`. The one row already
  measured at the new status quo, so it checks the new stage, the raised caps and the Drake pin
  at once. **If this row disagrees, stop and diagnose before reading anything else.**
- Other rows pair cell-for-cell against their 45 s counterparts at the same adopted setting:
  IPOPT vs `sc_SOLVER2_*_ipopt_*_480_45_*`, SNOPT vs `sc_SNOPTCOMBO_*_mstep0p5`. NLopt's
  adopted configuration was only ever measured at 60 cells, so that column compares on
  direction and magnitude, **not** cell for cell — say so rather than implying a pairing.
- `median_start_q_error` 0.0 exactly under `paired`; joint space bit-identical between protocols.
- Reproducibility band: +/-2 cells per row at 480 with a cap-bound population, 0 where a row has
  no timeouts.

## The three flag criteria, pre-registered

`scripts/report_statusquo.py` evaluates these; they are not judgement calls.

1. **Any learned-vs-joint-space verdict moving** between 45 s and 180 s. Expected: iiwa
   contained grasp joint-space-win -> tie; Panda contained grasp stays a decisive learned win
   with a growing margin (12/43 learned timeouts at 45 s); iiwa free grasp unmeasured above
   45 s and may move.
2. **The IPOPT-vs-SNOPT gap widening.** IPOPT's learned-arm failures are mostly wall-clock
   while only 3.6% of SNOPT's are the time limit, so a 4x cap should help IPOPT much more. The
   ordering is predicted; its SIZE is the reported quantity.
3. **NLopt at 180 s under the adopted configuration.** Untested — the flat 180 s arm predates it.

## Known follow-ups, deliberately not in this campaign

- **Re-measuring the chart ladder at 180 s**: available, **low priority**. Declined because
  nothing touching the charts changed and the rungs are chosen by the gain ceiling, not by
  cells. Worth reconsidering only if these tables show heavy timeouts, since that is the regime
  where the cap-dependent half of the depth mechanism does the work.
- **Analytic baselines**: the Panda has `analytic`/`analytic8` arms; the iiwa has no analytic
  program at all. Fielding them is future work or possibly not done at all.
- **Pose placement** is decided (shelf / fingertips, both tasks), so there is no open
  placement question.
