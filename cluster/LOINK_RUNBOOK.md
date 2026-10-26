# The LOInK chart family: the run's own record

What is submitted, in what order, what it depends on, and what a resuming session checks first.
The design is the plan (branch `chart-loink`): the record's learned formulation with the IKFlow
chart replaced by a LOInK chart (a bi-Lipschitz network `G(q) = [y_hat; z]` inverted by Davis-Yin
splitting, `src/charts/loink.py`), everything else unchanged. Findings go to `CLAUDE.md` /
`docs/notes-chart.md`; this file is procedure only. `cluster/README.md` and
`~/.claude/skills/supercloud/SKILL.md` carry the standing rules, which override anything here.
The cVAE campaign's runbook (`cluster/CVAE_RUNBOOK.md`) is the precedent this one follows.

**If you are even remotely unsure about a cluster action, stop and ask Thomas.**

## STATE -- 2026-10-26 00:40 cluster time: 16 of 24 exported, 4 killed (iiwa Bx/Bu A, LR collapsed), 4 P runs left; best chart 526 mm.
Resume: `bash cluster/submit_chart_train.sh --status | grep loinkB`; the watcher
`results/_watch/train_watch.sh` (laptop, background) fires on a new `node0.SENTINEL`, the kill
rule, or an empty `lik_trainc` queue.
 0. Done at 620k (val on the BRANCH test split; constant predictor Panda 581 mm / 136 deg, iiwa
    700 mm / 130 deg): Panda BmrP 743 mm / 83 deg rc 6; BxrP 743 / 83 rc 6; BurP 769 / 79 rc 6
    (relu: no triple of the 96-triple ladder passes the round trip, as every phase-1 relu chart);
    Panda BmtP 683 / 114 rc 0 ladder (100, 10, backtrack) cost 250, (a) pos median 615 mm;
    BxtP 692 / 114 rc 0 (25, 20, backtrack) cost 325, (a) 625 mm; ButP 791 / 109 rc 0
    (800, 50, backtrack) cost 1550, (a) 713 mm, off-branch |z|_inf median 85.6, in the box 0.02;
    iiwa BmrP 662 / 77 rc 6; BmtP 664 / 104 rc 0 (25, 20, backtrack) cost 325, (a) 545 mm p95
    1174, > 50 mm 0.9996, (g) off-branch pos median 1183 mm, in the box 0.72.
    Exports in `~/learned-ik/repo/models/<robot>/<robot>__loinkB*P__step620000.pkl` (5 tanh).
 1. Panda recipe A, done 22:10 (150k steps, batch 500): **all four met the kill rule** --
    `tr/learning_rate` < 1e-6 at step 13k (mtA), 18k (mrA), 20k (xrA), 12k (xtA), i.e. "recipe A:
    LR collapsed", frozen at 1.5e-8 from there -- but the 2 h watcher cadence missed the 30-min
    window, so they ran out (2 h each; nothing lost, the final chart is the 20k chart). Exports:
    BmrA rc 0 ladder (1600, 20, none) cost 1760, (a) 585 mm / 74 deg; BxrA rc 0 (1600, 30, none)
    cost 1840, (a) 577 mm / 73 deg (recipe-A relu charts DO invert, at 5-7x the tanh decode
    cost; P relu did not); BmtA rc 0 (25, 10, backtrack) cost 175, (a) 579 mm / 113 deg; BxtA
    rc 0 (25, 10, backtrack) cost 175, (a) 573 mm / 115 deg. (g) off-branch pos median
    765-1257 mm (relu p95 21-23 m), in the box 0.70-0.81. Val pos RMSE 629-680 mm: the best of
    the ladder, still at the constant predictor (581 mm). Watcher now polls every 15 min so the
    rule can fire on the remaining A pairs (Panda Bu A, iiwa Bm/Bx/Bu A).
    Running: iiwa Bx P pair (220k at 20:30) and Bu P pair (140k), ~01:30.
 2. 00:19 Sun: more recipe A. Panda BurA (lr < 1e-6 at 19k, ran out): rc 0 (1600, 30, none)
    cost 1840, (a) 602 mm / 71 deg, off-branch in the box 0.25. Panda ButA (14k, ran out): rc 0
    (100, 10, backtrack) cost 250, (a) 629 mm / 110 deg, in the box 0.32. iiwa BmtA (14k, ran
    out): rc 0 (25, 10, backtrack) cost 175, (a) 526 mm / 103 deg -- the best fit of the ladder,
    still half a metre. **iiwa BxrA + BxtA: KILLED under the rule** (watcher caught them at step
    30k, lr 1.5e-8 < 1e-6 before 50k): `scancel 6031776` at 00:19 with
    `checkpoints/last.ckpt` (step 40k) kept in both run dirs; no export, no screen; reason
    "recipe A: LR collapsed". `results/_watch/killed.txt` lists them for the watcher. The iiwa
    Bu A pair (6031777) started 00:19 and will meet the rule the same way; iiwa BmrA (relu A,
    6031775) is exporting; the iiwa Bx/Bu P pairs are at hours 5.7 / 4.6 of ~7.
 3. 00:35 Sun: **iiwa BurA + ButA KILLED** the same way (step 20k, lr 1.5e-8): `scancel 6031777`,
    `checkpoints/last.ckpt` (step 20k) kept, "recipe A: LR collapsed". So recipe A on iiwa:
    m pair ran out (started before the 15-min cadence), x and u pairs killed. iiwa BmrA (lr <
    1e-6 at 17k, ran out): rc 0 ladder (1600, 5, none) cost 1640, (a) 511 mm / 62 deg (the
    lowest of the ladder), (g) off-branch 1966 mm, in the box 0.75.
    Left: iiwa Bx P and Bu P pairs (~01:00 / ~02:00 + export). Watcher fix: the kill-rule awk
    now matches only status lines (`/^[a-z].* step /`), the `.pkl` export listing had tripped it.
 4. 01:45 Sun: iiwa BxtP done (620k): val 678 mm / 103 deg; rc 0 ladder (25, 10, backtrack) cost
    175, (a) 551 mm / 89 deg, (g) off-branch 1195 mm, in the box 0.71. iiwa BxrP (val 666 mm /
    77 deg) exporting; iiwa Bu P pair at hour 6.2.
 5. Downstream (`LOINKSMOKE_branch`, `LOINKPROBE_branch`): NOT submitted, Thomas's call.

## STATE -- 2026-10-25 14:50 cluster time: soft12 PROBE done (0/20 everywhere); the 24 branch runs training; downstream stages NOT started.
Resume: `ssh tcohn@txe1-login.mit.edu 'squeue -u tcohn'` (12 `lik_trainc_*loinkB*` pair jobs, 4 running);
`bash cluster/submit_chart_train.sh --status | grep loinkB`.
 0. Both chains ran to completion overnight (logs `results/_watch/probe_chain.log`, `phase2_chain.log`,
    `ceiling_chain.log`): PROBE submitted 00:06; test job rc 0 (663 s, ALL PASS); datasets
    `panda_branch` (drawn 112.9M, on-branch 0.2359, kept 25.015M, 01:48) and `iiwa14_branch`
    (drawn 200.8M, on-branch 0.1250, kept 25.015M, 02:04) on xeon-p8; training smoke
    `smoke_panda_loinkBmrP` rc 0 at 11:14 (ladder (25, 5, none), three draws + off-branch pass);
    the 12 pair jobs 6031766-6031777 submitted 11:17 (P at 14:00:00, A at 05:00:00, PAIRS=1).
    Reach ceiling (4 CPU jobs, grid hashes match the record): Panda pose 16/20 on the nominal
    branch (A = -1 branches reach 0), Panda grasp 20/20, iiwa pose and grasp 20/20 on every branch;
    JSON in `results/loink_branch_ceiling/`.
 1. soft12 PROBE: 56 items done (IPOPT 32 incl. the 8 acceptance shards, SVGD 24), every learned
    cell `constraint` at the 180 s clock, 0/20 on all 24 rows; IPOPT 20-30 it/cell relu, 60-150
    tanh; AL-SVGD 7-23 steps in 180 s. Collected (`collect_results.sh`, staging 20261025-143705),
    promoted to `results/soft12/benchmark/`, screens to `results/screen_loink/`,
    `report_loink.py --write` (acceptance IDENTICAL 479 + 1), `make_loink_ext.py --summary`
    (0 rows qualify; EXT manifests empty, header now 24 of 288 collected), selftest OK.
 2. Training at 14:30 (steps 180k-300k of 620k, lr 1.2e-5 - 3.3e-5, nothing near the kill
    rule): val pos RMSE Panda 680-778 mm, iiwa 659 mm; rot RMSE 78-115 deg. The constant
    predictor on the SAME branch test split is Panda 581 mm / 136 deg, iiwa 700 mm / 130 deg:
    position is at or below the constant predictor, exactly the phase-1 plateau (Panda 767 mm v
    610 mm on the full split, flat from step 20k). Restricting to one branch did not move the
    fit; topology was not the binding constraint. The runs finish anyway (poor fit is a result).
    Expected: P wave 1 ~18:30, P wave 2 (iiwa x, u) ~01:30, A pairs after.
 3. NOT started, awaiting Thomas: `LOINKSMOKE_branch` (+ `_SVGD`) and `LOINKPROBE_branch`
    (192 + 192 GPU items, ~1 h each) on charts that so far cannot fit. Reported to him 14:45 with
    the recommendation to decide once the screens are in. Still to do regardless: screens of the
    24 exports ((a), (g), gates), sidecars to the laptop's `models/`, notes, tables, CLAUDE.md.

## STATE -- 2026-10-24 22:30 cluster time: PROBE cut to 20 cells; single-branch retrain approved (phase 2).
Resume: `ssh tcohn@txe1-login.mit.edu 'squeue -u tcohn'`; PROBE jobs present means the watcher ran.
 0. Thomas (22:00): LOInK phase 2 = retrain the Panda and iiwa charts on ONE IK branch (the nominal
    configuration's: Panda gc (1, 2, +1), iiwa GC (+1, -1, +1)), 12 charts per robot, collapsed
    runs killed (lr < 1e-6 before 1/3 of the budget), datasets rebuilt on the branch by rejection
    sampling on xeon-p8 (never locally), the soft12 PROBE cut to 20 cells. Plan:
    `~/.claude/plans/it-s-time-to-handle-sunny-rainbow.md`; labels `loinkB<cost><act><recipe>`,
    datasets `panda_branch` / `iiwa14_branch`. Screw rows stay "chart does not fit (K = 1)".
 1. `LOINK_PROBE_TARGETS = 20` (gen_manifest; report_loink and make_loink_ext read it). PROBE and
    `_soft12P` manifests regenerated (tags `..._20_180_<start>`; 200 / 192 / 32 / 24 items), EXT
    manifests rewritten (empty), selftest OK, `make_loink_ext.py --check` rc 0, SHARDING OK.
 2. The 19:30 watcher DIED (the session scratchpad was wiped at 22:11 by a context compaction);
    a new one runs from `results/_watch/probe_chain.sh` (log `results/_watch/probe_chain.log`,
    gitignored, detached with setsid): same chain as item 2 of the 19:10 state (smoke check ->
    `stage_code.sh` -> the two `_soft12P` PROBE submissions at PROCS=2, SVGD under
    DRAKE_ENV=crc). Smokes at 22:30: IPOPT 24/24 done, SVGD 14/24 done (4 running, 6 queued),
    expected to drain 23:30-01:00. By hand if the log shows an exit code 2-5.
 3. 23:00: phase-2 code is COMMITTED (1636df0 manifests/report, 18fa551 classifiers/datasets/
    trainer plumbing; both implemented by Opus agents in worktrees, reviewed, local checks pass:
    tests/test_branches.py ALL PASS, selftest OK, SHARDING OK, the four new test_loink_chart
    checks pass alone). A second detached chain, `results/_watch/phase2_chain.sh` (log
    `phase2_chain.log`), waits for "PROBE SUBMITTED" in the first chain's log (the tree is
    staged then) and runs, unattended: the test job (`TEST=tests/test_loink_chart.py
    cluster/test_job.sh`, xeon-p8); `DATASET_ROBOT=panda_branch` then `iiwa14_branch`
    (`cluster/build_dataset_job.sh`, xeon-p8, one at a time, each waited for by its `.DONE`);
    the training smoke `submit_chart_train.sh --smoke loink panda --cost=m --activation=relu
    --recipe=P --dataset_name=panda_branch --branch=nominal` (GPU, queues behind PROBE); on its
    sentinel `rc=0`, `ONLY='_loinkB[mxu][rt]P$' PAIRS=1 ... --all 14:00:00` then the `A$` rows
    at 05:00:00. Exit codes 2-7 in the script header; by hand from the failed step otherwise.
    NOT staged (stage_code refuses under PROBE): nothing yet; the reach-ceiling script
    (scripts/loink_branch_ceiling.py, in progress) will be scp'd like test_job.sh was.
 4. Then by hand: poll `bash cluster/submit_chart_train.sh --status` every ~2 h; KILL any run
    whose status.json learning_rate < 1e-6 before 1/3 of its budget (`scancel`, keep last.ckpt,
    record "recipe A: LR collapsed" in report_loink.REASON and here). After the runs: read the
    screens (results/screen_loink/*loinkB*.json: (a) task error, (b)/(d)/(f) gates, (g)
    off-branch), copy the 24 sidecars to the laptop's models/, then `stage_code.sh` (once no
    job runs) and `PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKSMOKE_branch.txt 4`
    + `DRAKE_ENV=crc ... manifest_stageLOINKSMOKE_SVGD_branch.txt 4`; smoke check with
    `LOINK_SMOKE_ROBOTS=panda,iiwa LOINK_SMOKE_MATCH=_loinkB LOINK_SMOKE_EXPECTED=192
    LOINK_SMOKE_OFF_BRANCH_OK=1 python3 -`; then the `_branch` PROBE manifests, make_loink_ext,
    EXT, collect, merge, `report_loink.py --write`, notes, CLAUDE.md, memory.
 5. Phase-2 plan as approved (kept for reference): `src/branches.py` + `tests/test_branches.py`;
    `scripts/training/build_branch_dataset.py` + the `*_branch` case of
    `cluster/build_dataset_job.sh` (submit panda_branch, then iiwa14_branch, xeon-p8 4 h);
    `--dataset_name` / `--branch` through train_loink / export_loink / screen_loink /
    train_chart.sh; 24 chart_runs rows; LOINK_BRANCH_* rungs, `_branch` manifests, report block;
    training smoke; the 24 runs (`ONLY='_loinkB[mxu][rt]P$'` then `A$`, PAIRS=1) queued behind
    PROBE; kill rule polled every ~2 h; screens; LOINKSMOKE_branch -> LOINKPROBE_branch -> EXT.

## STATE -- 2026-10-24 19:10 cluster time: smokes running; PROBE cancelled, to be resubmitted as 60-cell probes.
Resume: `ssh tcohn@txe1-login.mit.edu 'squeue -u tcohn'`, then `python3 - < cluster/loink_smoke_check.py` over ssh.
 0. Smokes 6025468/69 (IPOPT) and 6025470/72 (SVGD, crc) running since 17:00; 20 of 48 items in
    at 18:50, 0 flagged, every paired start exact (start_q_error 0.0). Every cell so far is
    `constraint` at the clock, 0/40 successes, violations 1-125 (urP IPOPT paired runs away:
    z 42-162, violation 1e3-1e4). Per-item warm-up (`compile_seconds`, the unrolled fixed-count
    DYS loop under dynamo + graph capture) is 5-14 min on the (25,10) charts and 30 min-1.9 h on
    mrP/xrP/urP/utP; IPOPT gets 1.5-9 s per iteration, 20-130 iterations per cell.
 1. Therefore (Thomas, 2026-10-24: "do whatever you think is best, LOInK is a stretch goal") the
    queued PROBE jobs 6025498-6025505 were held and `scancel`led before starting (no state dir,
    no claims). Stage LOINKPROBE is regenerated so soft12 is probed like the rigid arms: one
    unsharded 60-cell item per chart x task x start (`--cells t:0`), 24 + 8 acceptance IPOPT
    items and 24 SVGD items instead of 204 + 580 sharded full-row items. LOINKEXT then covers
    soft12 too (a full row only where a probe solves >= 1 cell). No warm-up code change: not
    worth a code change on a stretch goal; ~12 slot-hours of compiling is the price.
 2. A detached laptop watcher (`probe_chain.sh` in the session scratchpad, log
    `.../scratchpad/probe_chain.log`, pid 106071, started 19:30) does the following on its own;
    a resuming session checks `squeue` first: PROBE jobs present means it ran. By hand otherwise.
    When the smokes leave the queue: run the smoke check (0 flagged expected; a relu paired
    cell may be `unrepresentable_start` at the recorded 0.05-0.10% rate, report it), then
    `bash cluster/stage_code.sh` (refuses while jobs run; the staged tree is otherwise d417150's
    code plus hand-copied `test_job.sh`), then from the laptop
    `PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE_soft12P.txt 4` and
    `DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE_SVGD_soft12P.txt 4`.
    Then collect, merge, `scripts/report_loink.py --write` (acceptance IDENTICAL), notes,
    CLAUDE.md (< 40k), memory.
 3. `tests/test_loink_chart.py` ran on the cluster: job 6025996 ALL PASS (47 checks, 636 s on
    xeon-p8; job 6025515 before it died on a missing PYTHONPATH, fixed in 38fe3cd). Log
    `~/learned-ik/results/tests/test_loink_chart_6025996.log`.

### Earlier state (2026-10-24 17:05)
STATE -- 2026-10-24 17:05 cluster time: smokes and PROBE resubmitted on the re-laddered sidecars.
Resume: `ssh tcohn@txe1-login.mit.edu 'squeue -u tcohn'`, then `python3 - < cluster/loink_smoke_check.py` over ssh.
 0. Re-ladder jobs 6019621-26 all rc=0 by 19:50 on 2026-10-23 (logs and before/after sidecars in
    `~/learned-ik/results/reladder/`). Chosen: mrP (400,50,none) best effort 0.05%; xrP (200,50,none)
    exact; mtP, xtP (25,10,backtrack) exact; utP (200,10,backtrack) exact; urP (25,50,none)
    unchanged, best effort 0.10%. The six sidecars are committed (`models/soft12/*loink*.arch.json`).
 1. Old-budget smokes (48/48 published, 3 flagged, all relu pose paired) moved to
    `~/learned-ik/results/soft12/benchmark_loink_prefix_budget/{published,rundirs}` (kept, not
    deleted); both smoke manifests' `.done` removed and claims reclaimed.
 2. Submitted 17:00 from the laptop (`submit_bench.sh`, PROCS=2): smokes 6025468/69
    (LOINKSMOKE_soft12P) and 6025470/72 (LOINKSMOKE_SVGD_soft12P, crc); PROBE queued behind them
    with `DEPENDENCY=afterany:6025468:6025469:6025470:6025472`: 6025498-501 (LOINKPROBE_soft12P,
    204 items) and 6025502-05 (LOINKPROBE_SVGD_soft12P, crc, 580 items).
    **If the smoke check flags any item, `scancel 6025498 6025499 6025500 6025501 6025502 6025503
    6025504 6025505` before they start.** Pass = 0 flagged; on the relu charts mrP/urP a paired
    cell may still be `unrepresentable_start` at the recorded 0.05-0.10% rate (report it, do not
    flag it). Then as before: collect, merge, `scripts/report_loink.py --write` (acceptance
    IDENTICAL), notes, CLAUDE.md (< 40k), memory.
 3. Still owed: one cluster run of `tests/test_loink_chart.py` (its extended ladder test never ran).

### Earlier state (2026-10-23 19:05)
STATE -- 2026-10-23 19:05 cluster time: smokes flagged, PROBE cancelled, re-ladder jobs queued.
Resume: `ssh tcohn@txe1-login.mit.edu 'squeue -u tcohn; ls ~/learned-ik/results/reladder/'`.
 0. What happened: the smoke check (`cluster/loink_smoke_check.py`, run as its first line says; 40 of 48 items at 18:55)
    flagged 3 items, all relu charts on the pose task with the paired start: cells scored
    `unrepresentable_start` (start_q_error 0.76-1.67). Mechanism: the exporter's ladder chose
    each chart's decode budget on latents from N(0, I); real configurations' latents (|z| 13-25
    on m/x charts) fail to invert there (relu 47-49%, tanh 15-21% of 2048 real q). The 8
    PROBE jobs (6018815-18, 6018819/23/24/25) were `scancel`led before starting (nothing
    touched). The IPOPT smokes 6018466/67 finished; SVGD smokes 6018468/69 were still running.
 1. Fix (commit d417150, staged by copying the three files from a GitHub clone at
    `~/learned-ik/src-clone` into `~/learned-ik/repo` -- stage_code.sh was too slow for the
    laptop's deadline; the chart module is unchanged): the gate covers three latent draws
    (prior, +-5 box, real), `export_loink.py --reladder PKL` rechooses an exported chart's
    budget in place, caps `--n_dys_max/--n_newton_max`, and a best-effort rule for a chart no
    budget inverts exactly on real latents (the relu charts: undamped Newton leaves 0.05-0.3%
    failing at every budget, backtracking stalls at the kinks; prior+box stay hard gates, the
    real draw's failure fraction is minimised and written as `real_latent_fail_fraction`).
    Tested: the three-draw gate and --reladder passed tests/test_loink_chart.py (ALL PASS);
    the best-effort rule, the caps and the --reladder CLI passed a localized untrained-chart
    check (19:03; `real_tol=0.0` -> BEST EFFORT fraction 1.0, exact path unchanged, caps cut the
    ladder, provenance.reladder with `replaced`). The file's own extended ladder test has not
    run; run it ONCE on the cluster (a CPU job, OMP_NUM_THREADS=8) before merge, never as a
    local gate (Thomas: a 10-30 min test is too expensive).
    `~/learned-ik/repo/.staged-commit` was set by hand to d417150 (full hash) so provenance
    is right; the next real `stage_code.sh` overwrites it.
 2. Queued 19:03: jobs 6019621-26 (`lik_reladder_{mrP,xrP,mtP,xtP,urP,utP}`, xeon-p8, -s 8,
    8 h, `cluster/reladder_loink_job.sh`, N_DYS_MAX 400, N_NEWTON_MAX 50). Each rewrites ONLY
    the inverse fields of `~/learned-ik/repo/models/soft12/<stem>.arch.json`; logs and
    before/after sidecars under `~/learned-ik/results/reladder/`. Read every log's CHOSEN line
    (or its failure) first. If a job failed on a code error, fix locally, re-stage the file the
    same way, resubmit that chart.
 3. Then rerun the smokes on the new budgets: move the old results aside
    (`~/learned-ik/results/soft12/benchmark/sc_LOINKSMOKE_*` -> `..._prefix_budget/` or delete
    after mining; they hold nothing the tables need), `collect_results.sh --reclaim` both smoke
    manifests AND clear their `.done` markers (`rm ~/learned-ik/state/manifest_stageLOINKSMOKE*_soft12P/*.done`),
    resubmit (`submit_bench.sh`, PROCS=2, DRAKE_ENV=crc for SVGD), run `cluster/loink_smoke_check.py`: pass =
    0 flagged (paired `start_q_error` ~0 except the recorded `real_latent_fail_fraction` share on
    relu charts). Copy the six new sidecars back to the laptop's `models/soft12/` (gitignored)
    and record the chosen triples and the per-iteration wall change in `docs/notes-chart.md`.
 4. Then PROBE as before (gen_manifest LOINKPROBE / LOINKPROBE_SVGD, DEPENDENCY on the smokes).
    Thomas emails the authors early the week of 2026-10-26; the HSA Table II gap is the artefact.

### Earlier state (16:20)
STATE -- 2026-10-23 16:20 cluster time: the soft12 smokes are queued on the graph-capture fix

What a resuming session checks FIRST:

```bash
ssh tcohn@txe1-login.mit.edu 'squeue -u tcohn; for m in manifest_stageLOINKSMOKE_soft12P manifest_stageLOINKSMOKE_SVGD_soft12P; do s=~/learned-ik/state/$m; echo "$m claims $(ls -d $s/*.claim 2>/dev/null | wc -l) done $(ls $s/*.done 2>/dev/null | wc -l)"; done; ls ~/learned-ik/results/soft12/benchmark/ | grep -c LOINKSMOKE'
```

-1. **Jobs 6018466 / 6018467 (LOINKSMOKE_soft12P, PROCS=2) and 6018468 / 6018469
   (LOINKSMOKE_SVGD_soft12P, crc, PROCS=2)**, submitted 16:17 on staged commit 9be5e9b. The
   first submission (6016941-44, 13:41) lost all 48 items at `WarmUpJacobian`'s CUDA-graph
   capture: `_linalg_check_errors(..., 'torch.linalg.solve')` -> "operation not permitted when
   stream is capturing". The chart's forward used the non-syncing solve; the jacrev graph is
   captured under `enable_grad`, and in that mode `linalg_solve`'s BACKWARD recomputes through a
   plain `at::linalg_solve` with its host-syncing info check (torch 2.14 on the cluster; local
   2.11 on CPU never hit it). Fix 9be5e9b: `_solve` = `lu_factor_ex` + `lu_solve` (derivatives
   graph-safe both ways); tests pass; the six sidecar triples re-pass the 2048-sample round trip
   (max 4e-12). Dead claims reclaimed (`collect_results.sh --reclaim`, 48). Worker logs appear
   in `state/<manifest>/logs/` ONLY when a worker finishes; a run dir under
   `~/learned-ik/repo/results/soft12/benchmark/` is the first sign an item got past warm-up.
   **Smoke pass = every item exits 0, family loink, no dead arm, paired `start_q_error` ~0
   with `clip_distance` ~8.6 recorded (the latent starts outside the box, by diagnosis).**
   16:54: the first 6 smoke items published, 0 flagged (paired `start_q_error` 0, clip 8.9-124,
   every cell `constraint` within the clock). **PROBE queued 17:00 behind the smokes**
   (`DEPENDENCY=afterany:6018466:...:6018469`): jobs 6018815-18 (LOINKPROBE_soft12P, 204 items:
   200 x 60-cell shards + the 4 `panda__n6` acceptance items, must print IDENTICAL) and
   6018819/23/24/25 (LOINKPROBE_SVGD_soft12P, crc, 580 items). **If the full smoke check
   (`scratchpad/smoke_check.py` over `~/learned-ik/results/soft12/benchmark/sc_LOINKSMOKE_*`)
   flags any item, `scancel` the eight PROBE jobs before they start** (claims are atomic; a
   PROBE job that has not started has touched nothing). Estimate: SVGD ~half a day, IPOPT up
   to 3 days if every LOInK cell runs to the 180 s clock. Then collect, merge,
   `scripts/report_loink.py --write`, notes, CLAUDE.md (< 40k).
   Thomas emails the authors early the week of 2026-10-26 (the questions are at the end of
   this file); the cells should be in by then.

### Earlier state (13:10): the six soft12 charts being re-exported and screened (done, rc=0 all)

What a resuming session checks FIRST (everything local died at any pause; the cluster side is
fire-and-forget):

```bash
ssh tcohn@txe1-login.mit.edu 'squeue -u tcohn; cd ~/learned-ik/results/train; grep -h rc= pair_soft12_*/launch.log-601685*; ls ~/learned-ik/results/screen_loink/; ls ~/learned-ik/repo/models/soft12/'
```

0. **Jobs 6016852 / 6016853 / 6016854** (the same three pairs resubmitted 13:08 with 03:00:00,
   staged commit 18a2bd2): training resumes at step 620000 and returns at once, then the
   in-job export runs the NEW ladder (n_dys x n_newton x newton_damping, nested 256/2048
   gate; locally the tanh chart chose (25, 5, backtrack), the relu chart (25, 50, none))
   and the screen. `rc=6` now means the ladder found no passing triple (the log lists all
   96); `rc=7` the screen failed. Then step 4 below.

1. (done 12:30) **Jobs 6012963 / 6012964 / 6012965** (pairs `soft12_loinkmrP+mtP`, `xrP+xtP`, `urP+utP`,
   recipe P, K = 6, 14 h walltime, started 06:43) were at 340-360k of 620k steps at 09:56 (~20k
   steps per 11.5 min): expected done ~12:35 cluster time. Each then runs the IN-JOB export with
   the STAGED (old) exporter, n_dys = 100 only, so expect `rc=6` (round trip fails) on some or
   all: that is the old exporter, not news. The checkpoints are what matter
   (`results/train/<run>/checkpoints/loink-checkpoint-step=620000.ckpt`).
2. Validation at 620k steps (constant predictor 504 mm): pos RMSE 205-332 mm, rot 29-58 deg,
   tau 1.1e4-4.6e4. The charts learn
   something (unlike the rigid arms) but plateau far from a usable fit. Do not read a verdict
   off this; the screens decide.
3. (done 13:08) **After the queue drains** (nothing of ours in `squeue`): `cluster/stage_code.sh` (stages
   the exporter ladder, ba583f0, and the `_soft12P` manifests), then re-export + screen each
   run by name with a short walltime (the "Trained but not exported" recipe below):
   `for r in soft12_loink{m,x,u}{r,t}P; do bash cluster/submit_chart_train.sh $r 01:00:00; done`
   (the GrpTRES cap meters them). A run whose ladder passes at no rung (exit 6 again, the log
   lists all six rungs) has no usable chart: record it and leave its rung out of the manifests
   (add its label to `LOINK_EXCLUDED_RUNGS`, regenerate, commit, restage).
4. Read `results/screen_loink/soft12__loink*.json` (collect with `cluster/collect_results.sh`
   or scp): gates (b) polished residual < 1e-10, (d) gain <= bound, (f) representability
   ~1e-10; (a)/(c) are the task-error and z = 0 numbers to report.
5. Only then the four `_soft12P` manifests ("What actually ran" below), smokes first.
6. Rigid-arm rows and every `A` rung: cancelled, "chart does not fit" (`docs/notes-chart.md`);
   `scripts/report_loink.py` still prints `not yet measured` for them -- change that before
   `--write`. Decisions reserved for Thomas: a padded-latent rung, a workable paper recipe.

Cancelled 2026-10-23 ~06:40: 6012955, 6012958-62, 6012966-77 (rigid P/A, every A pair; each
resumable from `last.ckpt`, but there is no reason to).

Original step table (what was planned; the rigid and A rows were cancelled after wave 1):

| step | jobs | depends on | state |
| --- | --- | --- | --- |
| stage_code (training half) | -- | -- | done 2026-10-22 |
| smokes | | -- | passed |
| P recipe, `ONLY='_loink[mxu][rt]P$' PAIRS=1 --all` (24 runs, 12 jobs) | 6012954-65 | smokes passed | rigid cancelled; soft12 running |
| A recipe, `ONLY='_loink[mxu][rt]A$' PAIRS=1 --all` (24 runs, 12 jobs) | 6012966-77 | smokes passed | cancelled (LR collapse) |
| collect; read the screens; **check in with Thomas** | -- | soft12 exported | |
| stage_code (benchmark half) | -- | training drained | |
| LOINKSMOKE_soft12P (24), PROCS=2 | | | |
| LOINKSMOKE_SVGD_soft12P (24), crc, PROCS=2 | | | |
| LOINKPROBE_soft12P (200 incl. acceptance), PROCS=2 | | smokes passed | |
| LOINKPROBE_SVGD_soft12P (576), crc, PROCS=2 | | smokes passed | |
| LOINKEXT: empty by construction (no rigid probe) | -- | -- | n/a |
| collect, merge, promote; the report | -- | | |

## What runs

| piece | file | what it is |
| --- | --- | --- |
| the 48 runs | `cluster/chart_runs.txt` (the `loink` rows) | `<family> <robot> <run_name> <trainer args>`; each row carries its recipe's `--batch_size` / `--max_steps` |
| submitter | `cluster/submit_chart_train.sh` | one single-GPU job per run (`--all`), or two runs per 2-GPU job (`--pair`, `PAIRS=1 --all`); `--status`, `--smoke loink`, `--smoke-pair loink` |
| payloads | `cluster/train_chart.sh` (`FAMILY=loink`), `cluster/train_chart_pair.sh` | train on one GPU (`scripts/training/train_loink.py`), then export the step-named final checkpoint (`export_loink.py`) and screen it (`screen_loink.py`), inline |
| benchmark | `cluster/gen_manifest.py` stages `LOINKSMOKE(_SVGD)`, `LOINKPROBE(_SVGD)`, `LOINKEXT(_SVGD)` | committed manifests `cluster/manifest_stageLOINK*.txt`; EXT written by `cluster/make_loink_ext.py` |
| report | (written separately; finds runs by the `sc_LOINK_` / `sc_LOINKACC_` tag stems) | |

### The charts: twelve per robot, all pre-registered, final checkpoint

Label `loink<cost><act><recipe>`: cost `m` (matched: the program's own `0.5 |q - q_nominal|^2`; soft
PCS `0.5 |strain|^2`), `x` (mismatched: LOInK's eq. 18 log-barrier), `u` (unstructured: no cost
head); activation `r` (relu) / `t` (tanh); recipe `P` (project: AdamW 1.5e-4, StepLR, 620000 steps,
batch 4096) / `A` (paper: Adam 1e-3, reduce-on-plateau, 150000 steps, batch 500). Same datasets as
the flow charts on every robot (the screw dataset is the rebuild, proved bit-identical: `n6r == n6`).

| robot (jrl name) | DOF | K = DOF - 6 | decision variables (6 + K + DOF) | trust radius (rule) | exports |
| --- | ---: | ---: | ---: | ---: | --- |
| Panda (`panda`) | 7 | 1 | 14 | 2.5 | `models/panda/panda__loink<c><a><R>__step<N>.pkl` |
| iiwa (`iiwa14`) | 7 | 1 | 14 | 2.5 | `models/iiwa14/iiwa14__loink...` |
| screw (`screw7_p050`) | 7 | 1 | 14 | 2.5 | `models/screw7_p050/screw7_p050__loink...` |
| soft PCS (`soft12`) | 12 | 6 | 24 | 3.95 | `models/soft12/soft12__loink...` |

`N` = 620000 (P) or 150000 (A): `train_chart.sh` reads `MAX_STEPS`, `BATCH` and `FINAL_STEP` from
the row's own args (they override the submitter's environment, which is why a pair job can mix
recipes), and the trainer's per-recipe checkpoint interval (P 20000, A 10000) divides both. The
sidecar carries `"family": "loink"`. **The trust region is always the rule**
`round(sqrt(K) + 1.5, 2)`, named on every item as `--set latent_trust_region=<r>`; there is no
flow-width rung, so no item keeps `--config latent`'s robot-sized radius. `dim_latent_space` is not
passed (the chart's). `gen_manifest.py --selftest` holds every item to its record twin with only
arms, checkpoint, tag and that radius changed, and `_ladder_paths_match_export` checks the 48 export
paths against `chart_runs.txt` in both directions.

## Order of operations, and why it is this order

```
0. stage the code (BEFORE anything is queued: stage_code.sh refuses while any job of this tree runs)
1. training smokes, one relu row + one tanh row (volta, 30 min)
        + the co-scheduling / GrpTRES measurement (or go straight to pairs)
        + watch the same-run guard refuse once
        v
2. the 48 runs: the P recipe and the A recipe as two PAIRS=1 submissions, walltimes from step 1
        |   export + screen run INSIDE each job on rc=0
        v
3. collect; read the six screen sections; CHECK IN WITH THOMAS     <- no benchmark before his go
        v
4. stage; LOINKSMOKE + LOINKSMOKE_SVGD (minutes)
        v
5. LOINKPROBE (IPOPT + acceptance) and LOINKPROBE_SVGD
        v
6. collect the probes; make_loink_ext.py; commit; stage; LOINKEXT + LOINKEXT_SVGD
        v
7. collect, merge, promote; the report
```

### 0. Stage

```bash
cluster/stage_code.sh
```

It refuses (exit 3) while ANY job submitted from this tree is queued or running -- including a
training job, since the trainer imports repo code. So stage once, before step 1, with everything the
training half needs (`src/charts/loink.py`, the trainer, exporter, screen, both payloads). The
benchmark half is staged again at step 4, after training has drained. Exported `.pkl` files in
`repo/models/` on the cluster survive restaging (`--filter='P models/*/*__step*.pkl'`). Check the
datasets first (`panda`, `iiwa14`, `screw7_p050`, `soft12` must all read `.DONE`):

```bash
source cluster/ssh_common.sh
sc_run 'for d in ~/learned-ik/home/.cache/ikflow/datasets/*/; do echo "$d $(cat $d/.DONE 2>/dev/null || echo NO-DONE)"; done'
```

### 1. Training smokes, the co-scheduling measurement, and the guard

Two 200-step smokes, one per activation, one per recipe, submitted together -- the plumbing check
(train, step-named checkpoint, export, round trip, screen) AND the co-scheduling test:

```bash
bash cluster/submit_chart_train.sh --smoke loink panda  --cost=m --activation=relu --recipe=P
bash cluster/submit_chart_train.sh --smoke loink soft12 --cost=u --activation=tanh --recipe=A
```

Run names `smoke_panda_loinkmrP`, `smoke_soft12_loinkutA`; each exports into its own
`results/train/smoke_<robot>_<label>/pkl/`, never `models/`. The smoke trains at its recipe's batch
(the submitter adds `--batch_size=4096` / `500`) for 200 steps (it refuses a `--max_steps`).
(`--smoke cvae <robot> <K>` and the old `--smoke <robot> <K>` are the cVAE's, unchanged.)

**Watch the same-run guard refuse, once**, while the first smoke is queued or running:

```bash
bash cluster/submit_chart_train.sh --smoke loink panda --cost=m --activation=relu --recipe=P   # MUST print REFUSING, exit 3
```

**Co-scheduling and the cap** -- the cVAE campaign skipped this measurement and used pairs, which
are one node and both its GPUs however the cap counts. Pairs are the default here too. To measure
anyway, while both smokes run and nothing else of the account holds volta nodes:

```bash
sc_run "squeue -u \$USER -h -o '%i %j %T %N %b'"                   # both on ONE node?
sc_run "scontrol show assoc_mgr users=\$USER flags=assoc | grep -o 'GrpTRES=[^ ]*' | sort -u"
```

`GrpTRES=node=4(N)` with N = 1 for two smokes on one node means singles fill 4 nodes x 2 GPUs;
N = 2 (Slurm's documented behaviour) means pairs. Smoke the pair mode before relying on it:

```bash
bash cluster/submit_chart_train.sh --smoke-pair loink panda \
    '--cost=x --activation=relu --recipe=P' '--cost=x --activation=tanh --recipe=P'
```

A smoke passes when `results/train/smoke_*/node0.SENTINEL` reads `rc=0` (pair: the pair's
`launch.log` reports both rcs 0), `status.json` reads step 200, the `.pkl` and `.arch.json` exist
under `pkl/`, and `results/screen_loink/<robot>__smoke_<robot>_<label>__step200.json` exists.
**Walltime, per recipe**, from the launch log's throughput (`train.log-<jobid>` in pair mode):
`steps / steps-per-second x 1.3`, with steps 620000 (P, read off the panda smoke) and 150000 (A,
read off the soft12 smoke; batch 500 is not batch 4096's rate), under the partition's 100 h ceiling.
The 200-step rate includes compile and warm-up, so it is pessimistic; if the P estimate exceeds the
ceiling, stop and ask. A smoke is resumable like any run, so **delete `results/train/smoke_*` before
re-smoking** -- otherwise the rerun resumes at step 200 and trains nothing.

### 2. The 48 runs

Submit the two recipes SEPARATELY, so each pair job holds two runs of one length (`--all`'s pairing
is over consecutive matching rows; with these filters it pairs relu with tanh at one robot, cost and
recipe -- `chart_runs.txt`'s header):

```bash
ONLY='_loink[mxu][rt]P$' PAIRS=1 bash cluster/submit_chart_train.sh --all <P walltime>   # 12 jobs
ONLY='_loink[mxu][rt]A$' PAIRS=1 bash cluster/submit_chart_train.sh --all <A walltime>   # 12 jobs
```

**Always pass `ONLY=`.** A bare `PAIRS=1 --all` would pair each chart's P run with its A run (valid,
but the A GPU idles for most of the job), and it also walks the cVAE rows: the finished ones are
skipped, but `iiwa14_cvae8` (diverged twice, recorded, never at 620000) would be RESUBMITTED. A row whose
`status.json` already reached its own budget (620000 / 150000) is skipped; a pair never spans two
families. `--all` reports and continues past a refused row; it never bypasses the guard.

```bash
bash cluster/submit_chart_train.sh --status    # every row's step / losses, the exports, this tree's queue
```

**Do not run the flow ladder at the same time** (its guard matches `/_train_/`, which `_trainc_`
does not contain, so neither family's guard refuses the other).

### 3. Export, screen, collect -- then the check-in

On rc=0 each job exports `loink-checkpoint-step=<N>.ckpt` (the STEP-NAMED checkpoint, never
`last.ckpt`) through `export_loink.py --out`, which round-trips through the LOInK loader (`decode`
within 1e-10 and `G(decode(w)) - w` below 1e-8, float64) and writes `n_dys` into the sidecar, then
runs `screen_loink.py` into `~/learned-ik/results/screen_loink/<robot>__<label>__step<N>.json`.
Collect with `cluster/collect_results.sh` (checkpoints and `pkl/` are not shipped; the screens are).
The `.pkl` files stay on the cluster in `repo/models/`; benchmarks read them there.

**Read the six screen sections for all 48 charts, then check in with Thomas before any benchmark
item is submitted.** The gates are pass/fail; the rest is what the report quotes:

| section | what it measures | gate |
| --- | --- | --- |
| (a) task error | `G_1` v FK, median / p95 mm and deg; fraction > 50 mm (the disconnected-fibre floor on the rigid arms) | none (the hypothesis: a floor on the rigid arms, not on soft12) |
| (b) inversion | `G(G^-1(w)) - w` v DYS iterations, with / without the Newton polish, float64 / float32; sets `n_dys` | **polished residual < 1e-10 (p99) at the chosen `n_dys`** |
| (c) z = 0 solution | FK error and program cost of `decode(y, 0)` v the dataset's per-pose minimum; `J_hat` v `J` (m, x only) | none |
| (d) gain | spectral norm of `dq/d[y, z]` over the +-5 latent box and the `c` box, max `|q|`, the bi-Lipschitz inequality on random pairs | **gain <= the certified bound (1/mu x scale); no pair violates the inequality** |
| (e) the rpy cut | task error binned by distance to the nearest roll / pitch / yaw cut; fraction of targets within 0.1 rad | none |
| (f) representability | `decode(invert(q)) - q` max abs; `|y_hat - y_FK|` (the paired start's `c` offset) | **round trip ~1e-10** (predicts paired starts are exact, unlike the cVAE's 0/480) |

A chart failing (b), (d) or (f) is a bug in the chart or its export, not a result: stop and report,
do not benchmark it.

### What actually ran: the four `_soft12P` manifests

Only soft12's six project-recipe charts exist (`models/soft12/soft12__loink{m,x,u}{r,t}P__step620000.pkl`):
every rigid-arm chart never fit the task at K = 1 and every paper-recipe (`A`) chart collapsed, so
their training was cancelled (`docs/notes-chart.md`, "Training outcome (2026-10-23)"). The
benchmark is submitted from FILTERED manifests, committed, `gen_manifest.py`'s
`LOINK_FILTERED_MANIFESTS`: each full manifest minus the 42 `LOINK_EXCLUDED_RUNGS` (three rigid model
dirs x twelve labels + soft12's six `A` labels; `--exclude-rung`, the cVAE `_noiiwa8` precedent; the
selftest holds them to it). Steps 4 and 5 below run these, not the full manifests:

```bash
cluster/stage_code.sh
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKSMOKE_soft12P.txt 2                 # 24 items
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKSMOKE_SVGD_soft12P.txt 2   # 24
# smokes passed:
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE_soft12P.txt 4                 # 200
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE_SVGD_soft12P.txt 4   # 576
```

| manifest | items | item-hours | LPT wall on 4 nodes x 2 |
| --- | ---: | ---: | ---: |
| LOINKSMOKE_soft12P | 24 (6 charts x 4 rows) | 1.3 | ~0.2 h |
| LOINKSMOKE_SVGD_soft12P | 24 | 1.3 | ~0.2 h |
| LOINKPROBE_soft12P | 200 (6 x 4 rows x 8 shards + the 8-item acceptance run) | 23.8 | ~3.0 h |
| LOINKPROBE_SVGD_soft12P | 576 (6 x 4 x 24 shards) | 44.8 | ~5.6 h |

The acceptance run stays in `LOINKPROBE_soft12P` (its `--checkpoint` is the Panda flow
`panda__n6`, which no exclusion matches). With no rigid probe, step 6's LOINKEXT is empty by
construction: nothing to submit. **Do NOT submit the full manifests**: item ids are the full
manifests', but `run_items.sh` keys its `.claim` / `.done` state by the MANIFEST'S BASENAME, so a
full manifest would not see the filtered one's items as done and would run them again
(`submit_bench.sh` refuses the full ones anyway while any `--checkpoint` they name is missing). If a full manifest is ever
wanted afterwards, carry the `.done` markers across first, as `cluster/CVAE_RUNBOOK.md` shows.

### 4. Stage, then the benchmark smokes

```bash
cluster/stage_code.sh                                            # training drained first
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKSMOKE.txt 2
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKSMOKE_SVGD.txt 2
```

192 + 192 items, one per logical run (48 charts x {grasp, pose} x {paired, native}), two cells each
(`--cells 0:0,31:5` of the record's grid), `sc_LOINKSMOKE_<robot>_<label>_<solver>_<row>_2_180_<start>`.
They pass when every item exits 0, no `_abort_on_dead_arm`, `metadata["chart_arch"]["family"] ==
"loink"`, and the paired cells' `start_q_error` is ~1e-6 or better (screen (f)) -- no
`unrepresentable_start`. `submit_bench.sh` refuses a manifest until every `--checkpoint` it names
exists on the cluster: the guard working, before training finishes.

### 5. The probe

```bash
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKPROBE_SVGD.txt 4
```

Paper conditions, one solve per V100, no MPS, as the record. **Probe first** (the cVAE precedent,
Thomas 2026-10-19), but with paired probes too -- the inverse is exact, so a paired cell is a real
solve here:

- rigid arms (Panda, iiwa, screw): every chart x task x start as ONE unsharded item of 60 cells,
  guess 0 of every target (`--cells 0:0,...,59:0`), tag `sc_LOINK_..._60_180_native` /
  `..._60_180_paired` -- 144 items per solver;
- soft12: the full 480-cell rows, sharded 8 (IPOPT) / 24 (AL-SVGD) -- 384 / 1152 items;
- the joint-space ACCEPTANCE run in LOINKPROBE, claimed first: CVAE_ACCEPT's row (Panda pose
  paired IPOPT, the record's flags, `--arms numerical`), `sc_LOINKACC_panda_n6_ipopt_posetip_480_180_paired`,
  8 items. It must be bit-identical to the record (q, iterations, cost, verdict per cell) after this
  branch's refactor, or no LOInK column pairs with the record's joint-space arm until understood.

Estimates (`gen_manifest.py --stage <S> --summary --procs 2`, at the FLOW's per-cell rates and
AL-SVGD's 5 s/cell allotment, +180 s per item; a chart that runs to the clock costs up to 180 s a
cell, i.e. 3 h per rigid probe item, still inside ITEM_TIMEOUT's 8 h):

| stage | items | item-hours | LPT wall on 4 nodes x 2 |
| --- | ---: | ---: | ---: |
| LOINKSMOKE | 192 | 10.0 | ~1.2 h |
| LOINKSMOKE_SVGD | 192 | 10.1 | ~1.3 h |
| LOINKPROBE | 536 (144 rigid probes 15.7 h, 384 soft12 46.6 h, acceptance 0.5 h) | 62.8 | ~7.8 h |
| LOINKPROBE_SVGD | 1296 (144 rigid probes 19.2 h, 1152 soft12 89.6 h) | 108.8 | ~13.6 h |
| LOINKEXT, at most (every rigid row qualifies) | 1152 | 125.7 | ~15.7 h |
| LOINKEXT_SVGD, at most | 3456 | 268.8 | ~33.6 h |

**No verdict is read off a probe.** It decides which rigid rows are run in full, nothing else; a
row's success, cost and wall are quoted from its 480 cells or not at all.

### 6. The extension

Collect the probe summaries (they are unsharded: no merge), promote them under `results/` (or leave
them in a `results/_cluster_staging/<stamp>/results/` tree, which the EXT generator also reads), then:

```bash
python cluster/make_loink_ext.py --summary --procs 2   # writes manifest_stageLOINKEXT(_SVGD).txt
python cluster/make_loink_ext.py --check               # exit 1 if the committed ones are stale
python cluster/gen_manifest.py --selftest              # EXT must be whole rigid rows of the full stage
```

The rule, fixed before any probe ran: **every rigid chart x row (solver, task, start) whose 60-cell
probe solved >= 1 cell gets its full 480-cell row**, sharded as stage CVAE; its items are the
would-be full stage's own, byte for byte (`stage_LOINK("LOINKFULL")`), so the probe's 60 cells are
re-solved inside the row. A probe counts only once its summary holds all 60 learned records; one not
yet collected keeps its row out and is listed. Before any probe is collected the manifests are
written EMPTY and the script says so -- never submit an empty manifest. Commit the manifests, stage,
then:

```bash
PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKEXT.txt 4
DRAKE_ENV=crc PROCS=2 bash cluster/submit_bench.sh manifest_stageLOINKEXT_SVGD.txt 4
```

Regenerating EXT after more probes arrive keeps every item id, and `run_items.sh` keys its state by
the manifest's BASENAME, so resubmitting the same manifest skips the items already done. Mind a job
still queued on the old EXT manifest when restaging (`stage_code.sh` refuses; never `FORCE_STAGE=1`
past it unless the only change is a manifest no running item reads).

### 7. Collect, merge, report

```bash
cluster/collect_results.sh                 # merges shards; promote each run as it prints
```

If `collect_results.sh` times out walking the whole tree (it did in the cVAE campaign), tar the
stage's summaries on the cluster (`results/<robot>/benchmark/sc_LOINK*_shard*of*/summary.json` for
the sharded rows, `sc_LOINK_*_60_180_*/summary.json` for the probes), pull one tarball, extract into a
new `results/_cluster_staging/<stamp>/`, run `cluster/merge_shard_summaries.py <staging>/results`,
and copy each merged run to `results/<robot>/benchmark/`. **Read the acceptance run first** (must be
IDENTICAL to the record's joint-space arm). Every LOInK run's `grid_hash`, `scene_fingerprint`, cap
and cells must match its record row; the cap check (clock AND budget, >= 24 budget cells voids a
row) applies as everywhere -- a VOID row is re-measured with the iteration budget lifted, never the
clock.

## Resume checks and failure modes

- **Is a run trained?** Only `--status` says so: step 620000 (P) or 150000 (A). A job leaving the
  queue is not evidence. The payload's exit code is in the launch log (`launch.log-<jobid>`; pair:
  `train.log-<jobid>` in the run dir), and for 0 / 5-7 also in `results/train/<run>/node0.SENTINEL`
  (2 and 4 exit before training, so they leave no sentinel):
  - 2 = refused before training (unknown FAMILY, or a row's `--checkpoint_every` that does not
    divide its final step -- no step-named checkpoint would exist to export);
  - 4 = no dataset `.DONE`;
  - 5 = training exited 0 without the step-`<N>` checkpoint;
  - 6 = export or its round trip failed (`decode` / `G(decode(w)) - w` tolerances: a chart bug);
  - 7 = the screen failed.
- **Walltime kill / node failure**: resubmit the same run (or pair) -- `--ckpt_path=auto` resumes
  from `last.ckpt`. `--all` (with the same `ONLY=`) does it for every unfinished row.
- **Trained but not exported** (rc 6/7, or `RUN_EXPORT=0`): resubmit that run by name with a short
  walltime; it resumes at its final step, Lightning returns at once (check the log says so), and the
  in-job export runs again. A row of `chart_runs.txt` supplies its own FAMILY, ROBOT and args:
  `bash cluster/submit_chart_train.sh panda_loinkmrA 01:00:00` (`--all` skips finished rows).
- **A pair where one run died**: resubmit the pair; the finished run resumes, returns at once and
  re-exports, the other continues. Or resubmit the dead run alone -- the guard refuses while the
  pair job still holds either name.
- **A benchmark item killed or a solve overrunning**: `run_items.sh`'s watcher kills a solve past
  1.5x its clock (270 s at 180 s) and relaunches the item, which resumes from its per-cell JSONL;
  the killed cell is `killed_overrun`, `timed_out=True`, scored from its last iterate
  (`cluster/README.md`, "The overrun watcher"). A dead item is reclaimed with
  `cluster/collect_results.sh --reclaim <manifest>` only when nothing of that manifest runs, then
  the same manifest is resubmitted.
- **Restaging mid-training** is refused by `stage_code.sh`; never `FORCE_STAGE=1` past it -- the
  queued and running jobs re-read the repo's code.
- **PROCS=2 always, no MPS** for every LOInK benchmark manifest (paper conditions: one solve per
  V100); `DRAKE_ENV=crc` for every `_SVGD` manifest and never for the others (`submit_bench.sh`
  refuses a mix).
