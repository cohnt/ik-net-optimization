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

## Regenerating the manifest

```bash
python cluster/gen_manifest.py --stage STATUSQUO --wall-time 180 --targets 60 --guesses 8 \
    --shards 8 --solvers ipopt,snopt,nlopt --starts paired,native > cluster/manifest_stageSTATUSQUO.txt
```

**Every one of those flags is load-bearing, because the CLI defaults are narrower than this stage
and fail silently.** `--solvers` defaults to `"snopt"`, `--starts` to `"paired"`, `--shards` to `1`,
and `--wall` is not an option at all (it is `--wall-time`, and an unknown flag makes argparse take
every default). Omitting them regenerates a *different, smaller* campaign that still looks like a
valid manifest: I wrote a 32-item file over the real one that way on 2026-09-21. Check the count --
**320 items** (8 rows x [8 + 8 + 24] shards x 2 robots... precisely: ipopt 64, snopt 64, nlopt 192)
-- and check `grep -c free` is 0, before trusting a regenerated manifest.

## Progress

**CAMPAIGN COMPLETE, 2026-09-20 16:26 ET.** 480/480 items, 36/36 logical runs merged with zero
failures, 17,280 solves -- of which **the RECORD is 24 runs / 11,520 solves: two robots x TWO
experiments (grasp and pose, both shelf-contained at the fingertips) x two protocols x three
solvers.** The other 12 runs were `--target-placement free`, a vestigial SETTING of the grasp
experiment rather than a third experiment; the data is on disk and is not reported, the stage no
longer fields it, and the selftest refuses a non-contained placement. Thomas, 2026-09-21: gathering
it was fine on an idle cluster, but *"the intent of status quo was in part to select the experiments
we care about"*. Ran 2026-09-19 21:47 -> 2026-09-20 16:26, ~18.7 h wall on jobs
5681825-5681828 (4 nodes x `PROCS=8`). No stale claims, no errors, no reclaims needed, and the
"don't claim what the job cannot finish" guard never had to fire on a 48 h job.

- [x] Local plumbing, stage, reporter committed. Cluster: one Drake (the pin), code staged, smoke
      passed on debug-gpu.
- [x] All 480 items complete. Item cost split sharply by robot: iiwa NLopt items 90-144 min (the
      bimodality is exact -- target-major sharding of 60 targets over 24 gives twelve 3-target and
      twelve 2-target shards, so 48 or 32 solves, and 48 x 180 s = 144 min), Panda NLopt items
      27-78 min. IPOPT and SNOPT items ran ~16 min.
- [x] Collected, merged, promoted: 36/36.
- [x] **ALL ACCEPTANCE CHECKS PASS.** The decisive one: iiwa `n4` contained grasp under IPOPT
      reproduces `sc_CAP_iiwa_n4_mug_180_{native,paired}` **exactly** -- learned 447/453, joint space
      442/442, `grid_hash fa692df81e7d-mug` both sides, delta +0 on all four numbers -- validating the
      new stage, the raised caps and the Drake pin at once. Also: every row 480 cells on both arms;
      `median_start_q_error` 0.0 on all 18 paired rows; joint space identical between protocols on all
      18 solver x row pairs; provenance uniform (one Drake `0.0.20260918`, `Major step limit = 0.5` and
      the `LD_MMA` inner configuration emitted everywhere they should be).
- [x] Reported in full, all three flag criteria answered.
- [x] `CLAUDE.md` results tables replaced; the 45 s / free-grasp tables and stage SOLVER2's success
      table deleted rather than accumulated.

## Results, in one place

**IPOPT, the status quo's 8 rows: six decisive learned wins, two ties, no losses.** Every pose row on
both robots (p 2.9e-37 to 4.6e-68); Panda contained grasp +148/+153 cells; iiwa contained grasp a tie
at 447/453 v 442. **SNOPT: four wins, two ties, two losses** (both iiwa contained grasp) -- one
solver-dependent verdict. **NLopt: all four pose rows decisive learned wins** (iiwa 298/118 v 31,
Panda 289/114 v 12), Panda contained grasp native 327 v **0**, and the four iiwa grasp rows at the
floor on both arms.

**Three findings worth carrying forward.** The rescue rate is 82-100% on every IPOPT row, with
157-263 joint-space failures available to rescue under containment against 23-27 free. The cap effect
is entirely on grasp rows (+5 to +55 for IPOPT, exactly +0 on every pose row), which is both the
IPOPT-SNOPT widening mechanism and the campaign's tightest reproducibility check. And **the augmented
Lagrangian is extraordinarily start-sensitive** -- Panda contained grasp 327 native -> 0 paired,
against IPOPT's largest protocol effect of 476 -> 471 -- so the NLopt column must be read per protocol
and never pooled.

## Harness defects this campaign found and fixed

1. **`--reclaim`'s guard had never refused anything.** It filtered `squeue -n run_items.sh`, a name no
   job carries (`submit_bench.sh` sets `lik_bench_<manifest>`), so `BUSY` was unconditionally 0 and a
   mop-up during a live campaign would have cleared claims from under 32 working workers. Now scoped
   to `lik_bench_$MANIFEST_NAME`, verified live in both directions.
2. **The merger could only see two staging directories.** NLopt items ran 27-144 min against hourly
   collections, so one row's shards landed across three collections and it reported 22 of 24 missing
   from directories that existed. `--also` is `action=append`; it now gets every prior directory.
3. **The NLopt column had no work measure.** The reporter read `jacobian_evals`, parsed from a solver
   print file NLopt never writes, so it was `nan`. Now reads the program's own
   `eval_counts["map_jacobian"]`.
4. **Medians over successes misdescribe a 99%-failing column.** On iiwa free grasp the 3 solved cells
   report 173 Jacobians and 0.98 s against the typical cell's 12,344 and 180 s -- the median would have
   called the augmented Lagrangian cheap. The NLopt section now reports means over all 480 cells.
5. **Cost on too few shared cells.** One NLopt row shares exactly one solved cell with joint space,
   where the medians were 17.376 against 0.596. Cost now prints `--` below 10 shared cells.

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
