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
- [ ] Items complete (192 / 480 at 2026-09-20 00:25 ET, 2 h 40 m in). **IPOPT 96/96 DONE,
      SNOPT 96/96 DONE**, NLopt 0/288 with all 32 workers on its first round. The manifest is
      emitted in solver order rather than longest-first, so the entire NLopt block is the tail:
      288 items over 32 workers is 9 rounds. The oldest NLopt claim is 79 min old with none
      finished yet, consistent with the ~108 min/item estimate, which puts completion at
      **~15:30 ET Sunday**. Refine once the first round lands -- no NLopt cell had ever been
      timed at 180 s under the adopted configuration.
- [x] **Collected and merged, incrementally, mid-run.** 18 of the 24 IPOPT/SNOPT logical runs
      merged, including **all twelve iiwa rows under both solvers**. The 6 outstanding are Panda
      shards still in flight. Collection mid-run is safe and is the designed path: a shard is
      published to the collection point only on exit 0.
- [x] **ACCEPTANCE CHECKS PASS, including the decisive one.** iiwa `n4` contained grasp under
      IPOPT reproduces `sc_CAP_iiwa_n4_mug_180_{native,paired}` **exactly** -- learned 447 / 453,
      joint space 442 / 442, 0 timeouts on every arm, `grid_hash fa692df81e7d-mug` on both sides,
      delta +0 on all four numbers. That is a different stage, a different Drake (nightly
      `0.0.20260918` against the archive's 1.56.0) and the raised caps, all validated at once.
      `median_start_q_error` is 0.0 on all 8 paired rows and joint space is identical between
      protocols on all 8 solver x row pairs.
- [x] **IPOPT and SNOPT reported in full** -- all 24 of their logical runs merged and read
      (11,520 of the campaign's 17,280 solves). Three groups whose shards straddled two
      collections merged into the *previous* staging directory, which is the documented
      behaviour, not a fault.
- [ ] NLopt column reported. Three of its twelve rows are in: **iiwa contained grasp native is
      0/480 on BOTH arms with 480/480 timeouts each**, and both legacy iiwa free-grasp rows are
      learned 3 / joint space 11. So NLOPTTUNE's 60-cell finding -- nothing Drake exposes makes the
      augmented Lagrangian solve an iiwa grasp -- replicates at 480 cells. Rows where both arms sit
      at the floor carry no comparison and no cost column; the rows that will carry the decisive
      NLopt result are the pose ones, still pending.
- [x] **The new results section is DRAFTED** at
      `scratchpad/statusquo_section.md` (IPOPT and SNOPT tables, headroom/rescue-rate table, the
      legacy free-grasp table, and flag criteria 1 and 2 answered). Held out of `CLAUDE.md`
      deliberately until the NLopt column lands, so the file is edited once rather than twice.
      Note when splicing it in: it replaces the four table subsections between
      "Grasp task, adopted default" and "Settled negative results on the knobs", and the 45 s
      reference values in it are the MEASURED `sc_SOLVER2`/`sc_SNOPTCOMBO` pairings, not the older
      numbers quoted elsewhere in `CLAUDE.md` (e.g. Panda contained grasp paired pairs at 437, not
      the 444 an older table shows).
- [ ] `CLAUDE.md` tables replaced

**IPOPT and SNOPT, complete. Two of the three flag criteria have final verdicts.**
Criterion 1 fires on **four rows, and both moves were named in advance**: iiwa contained grasp
goes joint-space-win -> **tie** under both protocols (447 v 442, 453 v 442), exactly as predicted;
and iiwa FREE grasp -- the row flagged as unmeasured above 45 s -- goes tie -> **learned win**
under both (471 v 457, p = 0.016; 475 v 457, p = 1.2e-04). No verdict moved against us.
**Timeouts are essentially zero at 180 s** -- at most 2 cells of 480 on any of the 24 rows -- so
these are formulation results rather than cap results, and by the same token the deferred 180 s
chart-ladder re-measurement stays unwarranted: heavy timeouts were its trigger and there are none.

**Criterion 2 has a final answer with a mechanism.** Learned arm, IPOPT minus SNOPT, median gap
**102 cells at 45 s -> 120 cells at 180 s**, IPOPT ahead on 12 of 12 rows at both caps. The
widening is entirely attributable: raising the cap gains IPOPT +5 to +55 cells on every *grasp*
row and **exactly +0 on every pose row**, while SNOPT gains -1 to +7 anywhere. That is the
predicted mechanism measured directly -- IPOPT's learned-arm failures are wall-clock, SNOPT's are
convergence failures (3.6% time limit) -- and it doubles as the tightest reproducibility check the
campaign has: every row that had no timeouts at 45 s reproduces its 45 s cell count **exactly**,
with the single exception of Panda pose tip paired at +2.

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
