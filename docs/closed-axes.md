# Evidence for closed questions

`CLAUDE.md` carries the **verdict** of every question below; this file carries the **evidence**, so
the record a session reads at startup does not pay for questions that are settled. Nothing here is
live: each section names a closed axis, and reopening one is Thomas's call, not a reader's.

Numbers are reproduced verbatim from the campaign record. Where a `scripts/report_*.py` regenerates a
table, it is named — that script and the persisted runs under `results/` are the source of truth, and
this file is a narrative index into them.

## What solver tuning is worth: IPOPT's early stop, and one SNOPT setting

**Per-solver tuning is permitted and per-problem tuning is not** (Thomas, 2026-09-17: *"I'm okay with
playing with solver settings on a per-solver basis, as long as it's not per-problem"*), so a setting
qualifies only if it wins **uniformly across rows**, never by picking the robot it helps. Every rule
below was pre-registered before any result was read, and every one is **counted by row, never
pooled**.

**IPOPT: the convergence tolerances are INERT and the acceptable-point machinery is everything**
(stage SWEEP, 23 settings, 240 cells). Holding IPOPT to SNOPT's convergence numbers is within noise —
IPOPT already converges far tighter than either default, so the 1e-4-against-1e-6 asymmetry documented
above was real on paper and worth **zero cells**. At each solver's own defaults, which is what rung 2
of the tolerance ladder asks for, **IPOPT 200, SNOPT 141, p = 4.1e-09**, and on the cells both solve
IPOPT's solutions also cost less. **The fairness question is answered: the ordering survives it.**

**The early stop is worth 39 cells and a 9x speedup, and that has to be stated** — and it is **NOT
returning sloppy points.** Fielded successes sit at 1.29e-08, five orders inside the gate and the same
quality as SNOPT's. Turning it off drives the violation to 2.22e-15 and success *down* to 62 of 240:
IPOPT without it keeps polishing a solution it already has until the clock kills it.
`acceptable_iter = 1` does not let IPOPT scrape past the gate, it lets IPOPT **recognise it is already
done and stop**, a real capability under a wall-clock cap. SNOPT has no counterpart and would not
benefit. Nothing was adopted: the alternatives move success by at most +2 cells of 240, and changing
the default would break comparability with every archived run.

**SNOPT: one setting of thirteen survives 480 cells x 12 rows, and no combination beats it.** The
motivation was fairness, not rescue — IPOPT's column ran a tuned configuration while SNOPT's ran bare
Drake defaults, a property of the harness rather than of SQP. Stage SNOPTTUNE's rule (>= 9 of 12 rows
better on the learned arm, none significantly worse, >= 1 significantly better) passes **`Major step
limit = 0.5`** alone. Stage SNOPTCOMBO then crossed it with the three other positive factors under two
bars — that rule against Drake's defaults, plus >= 8 of 12 better against the survivor itself — and
**nothing clears both**, so the combination question is closed. Four combinations are *valid* SNOPT
configurations in that they beat Drake's defaults; none improves on the survivor.

**The pooled numbers disagree with the row rule, and the rule is what stands.** Pooled over 5,760
learned cells a four-factor stack beats the survivor by +221 cells and would have been fielded on that
basis. **It is a TASK TRADE**: it gains 35-58 cells on each iiwa grasp row, all significant, and loses
on every pose row — exactly what pooling hides, and why the rule reads rows. It is also bought with
work rather than insight, running ~50% more iterations and roughly twice the timeouts.

**The mechanism is diffuse, and the SNOPT step-limit hypothesis is REFUTED at both scales.** A smaller
`Major step limit` was predicted to convert INFO 41 into convergence. It does not: INFO 41 supplies
less than a third of the gain and falls by under 2% of itself, and the 60-cell story that INFO 13
falls did **not** replicate — INFO 13 is flat. What the setting actually does is stop SNOPT exhausting
its iteration and time budgets.

**NLopt: naming the inner optimizer is worth nothing; truncating it is worth a great deal.** Stage
NLOPTTUNE fielded nine settings plus a 180 s budget arm across 12 rows. Its pre-registered gate
(>= +8 cells of 60 on >= 6 of 12 rows, or any row from <= 2/60 to >= 20/60) is cleared by
**nothing**. `LD_MMA` named alone matches Drake's default column cell for cell with identical median
violations — so PR 25002's *algorithm selector*, the thing that unblocked the stage, is inert here.
What moves cells is the inner **budget and tolerance** options that shipped with it: on Panda
contained grasp the default burns 3,532 network Jacobians per cell and lands 3.7e-04 from feasible,
where loose inner tolerances use **66** and reach 6.8e-07. That is the classic augmented-Lagrangian
failure — an inner subproblem solved to convergence before the multipliers are ever updated — with the
classic fix.

**But it is a trade, not an improvement, and the sign flips by row.** On cells both columns solve,
loose inner tolerances cost more nearly everywhere, and on **Panda pose native**, a row the default
already solves 38 of 60, truncation *loses* cells monotonically in how aggressive it is. The inner
budget buys feasibility where feasibility was the binding problem and costs optimality everywhere
else.

**Three things no NLopt setting changes.** The **iiwa grasp rows are 0-3 of 60** under every setting
and at 180 s — nothing Drake exposes makes the augmented Lagrangian solve that task at all. The
**joint-space arm moves with these settings too**. And the **ordering is untouched**; what changed is
that the third column is now measured across Drake's whole option surface instead of left unswept by
decision.

**The `LD_SLSQP` caveat, unresolved by design.** It puts an SQP method inside the augmented
Lagrangian. The outer method is still AL and the inner subproblem bound-constrained only (see the
mechanism below), so it reads as an AL column — but the three-method-classes rule is Thomas's
(2026-09-18: *"For now, it's okay to have LD_SLSQP as the inner optimizer. We can always decide
later."*), so the caveat travels with the number rather than resolving it. Mechanistically it behaves
like SNOPT: **0 timeouts on every row** and 2.3-4.4 s mean wall clock against the default's 27-45 s.


## The result: IPOPT > SNOPT >>> NLopt, and the whole axis is CLOSED

All three method classes have been swept to exhaustion. **Do not re-sweep any of them.** The
per-stage tables are deleted; each stage's reader regenerates them from the persisted runs
(`scripts/report_snopttune.py`, `report_snoptcombo.py`, `report_nlopttune.py`, `report_step.py`,
each of which implements its own pre-registered rule inline so it cannot drift). The success
numbers of record are stage STATUSQUO's, under "Results" below.

**This is a CONFIRMATION, not a finding.** Thomas: *"SNOPT performing worse than IPOPT is not
surprising. In my experience, IPOPT is more robust to ill-posed problems, and our neural network
gradients are definitely ill-posed. I expect to see IPOPT > SNOPT >>> NLOPT."* Write it up as the
size and mechanism of a predicted gap. On stage SOLVER2's 480-cell grid IPOPT won all 24 rows, 23
significantly, p from 3.6e-08 to 4.8e-44.

**It is a property of the PROBLEM, not of the learned formulation.** The joint-space arm degrades
under SNOPT too, on every row and by comparable margins, and that arm never evaluates the network.
The same holds for every solver setting below: each moves both arms.

**The mechanism, and it is not budget.** Pooled over all 3,952 SNOPT failures: `nonlinear
infeasibilities minimized` (INFO 13) **50.7%**, `current point cannot be improved` (INFO 41)
**36.2%**, iteration limit 9.5%, time limit **3.6%** — about 87% convergence failures against 11%
budget, and SNOPT times out *less* often than IPOPT. The two fail in opposite ways: IPOPT's
learned-arm failures are mostly wall-clock, still descending when the clock runs out, where SNOPT's
are INFO 41, giving up at a feasible-but-wrong point. **INFO 41 is the documented signature of
inaccurate or badly scaled derivatives**, which is Thomas's explanation with a mechanism attached.
It is not the gain-ceiling runaway: `n4`'s solved cells return violations ~1e-08, so what SNOPT
struggles with is ordinary ill-conditioning of an *exact* Jacobian. SNOPT also takes 2-4x the
iterations on cells it does solve, and its solutions are worse on the joint-space arm by a wide
margin; on the learned arm cost is closer and occasionally favours SNOPT — where it converges it
sometimes finds a better optimum, it just converges far less often.

**The gap is much larger under `paired`**, so SNOPT copes far worse with an infeasible start. A
60-cell triage read this backwards in both directions before 480 cells settled it: **do not draw a
protocol conclusion from 60 cells.**

**Wall-clock columns are not capped equally across solvers.** NLopt's `max_time` binds much more
tightly than SNOPT's `Time limit` (20.0-20.1 s against 24-28 s in a local probe), because SNOPT only
checks at major-iteration boundaries.

**One cap-rule lesson, the unusual case.** At its defaults NLopt timed out on 60 of 60 grasp cells,
which looks exactly like the throughput case the cap rule is written for — but at 180 s ten of twelve
rows are identical to 45 s. The default configuration never terminates its *inner* solve, so more
wall clock buys no outer progress. The cap rule still holds; it just needed the cap arm to be **run
rather than assumed**.

## Step rejection: measured, refuted, and it exposed something bigger (stage STEP)

**The last lever on the solver axis, and it is closed.** Filter tuning leaves the program exactly
as written and changes only which trial points are accepted, so it was never touched by the
gradient-damping refutation. Two stages settled it. `stage_STEP` in `cluster/gen_manifest.py` owns
the five knobs and reuses `STEP_REJECTION_KNOBS` as a **whitelist**, the mirror of stage SWEEP's
blacklist, so the two questions cannot merge from either side. All five are now proven to reach
their solver from its own parameter echo (`tests/test_solver_plumbing.py`) — they never had been.

**The screen: 16 settings x 6 rows x 60 cells, and not one reached significance.** Best pooled over
the 240 selection cells was IPOPT `theta2`/`soc0`/`theta1` at 203 against 191 (p = 0.10-0.16) and
SNOPT `mstep0p5` at 127 against 110 (p = 0.060). The `theta_max_fact` table was narrowed in advance
by the old probe's own archive, which is still on disk: `full_theta10` is **bit-identical** to
`full_base` to fourteen digits, so >= 10 is a proven null and the selftest refuses one.

**The confirmation at 480 cells x 12 rows refuted both promoted settings, and reversed the screen.**
`theta1` is **significantly worse** — pooled +322/-421 over 2,880 cells, p = 0.00032 — and on the
very row the screen liked best, Panda pose paired, the screen's 45 -> 53 became **404 -> 391**.
`soc0` is a clean null (+302/-321, p = 0.47). **The mechanism column reversed too**: at 60 cells
`theta1` appeared to cut restoration on Panda pose 34.7% -> 24.9%, and at 480 cells it *raises* it,
26.8% -> 31.4%. A plausible mechanism agreeing with a spurious outcome did **not** protect against
the false positive; both were the same noise.

**The SNOPT hypothesis is refuted with a named replacement.** A smaller `Major step limit` was
predicted to convert INFO 41 (`current point cannot be improved`, a line search that cannot find a
step) into convergence. It does not: `mstep0p5` gains 15 cells of 360 while INFO 41 stays at
**exactly 81**, unchanged. What falls is INFO 13 `nonlinear infeasibilities minimized`, 51 -> 42. The
step limit helps SNOPT *reach* feasibility, not escape a stalled line search. It is also a **trade,
not a win** — +5/+7 on the iiwa against -7/-6 on the Panda — which is why the pre-registered harm
clause excluded it. Measured at 60 cells only.

## Every optimization-side remedy has been measured and refuted

Thomas's ranking: **(1) a better chart is preferred over everything else** — *"All of these actions
are less preferred than just getting a better iiwa chart"*; (2) lifting `q` into a bounded decision
variable, permitted but disliked (*"we're effectively adding a nonlinear equality constraint"*); (3)
a joint-limit penalty, permitted but disliked. **"You can try it, but I don't like it" means measure
it and report it as a stated deviation, not adopt it if the numbers look good.**

| remedy | verdict |
| --- | --- |
| **chart accuracy** (`chart_error_scale`, a smooth seeded perturbation) | **not the mechanism.** Degrading the Panda's chart to the iiwa's accuracy (~12-20 mm) costs it 1-3 cells of 60 (35 → 34 → 32); the iiwa is 23 cells worse. The standing chart-accuracy hypothesis does not survive its own experiment. Smooth `sin` error also adds no high-gain regions, so this could never have reproduced the real pathology. |
| **IPOPT scaling** (`nlp_scaling_method=none`, `nlp_scaling_max_gradient` 1e4/1e8) | **inert**, five experiments x 60 cells, largest movement ±2 cells, no p < 0.5. Not a scaling artefact: IPOPT is being handed a chart with gain ~1e13 and following the gradient into it. (`equilibration-based` needs HSL MC19 and raises on construction.) |
| **joint-limit penalty** at 1/10/100 on four rows | **inert** — smallest p = 0.115, no consistent direction, runaway counts unmoved, median violation unchanged at ~1e-08. Adding a penalty on a quantity a constraint row already governs buys nothing, vindicating Thomas's objection. |
| **lifting `q`** (`lift_q`: `q` a bounded variable, chart as a 7-row equality) | **net negative, 38 better against 116 worse over eight experiments, p = 2.2e-10.** It delivers exactly one thing universally — **0 runaway cells in all eight** — but **the runaway does not stop, it relocates**: the flow still reaches 1.19e10 and it lands in `max_violation` instead of `q` (median 3e+06 to 7e+06 on the collapsing pose rows). The split is by **task**: neutral-to-positive on grasp with *fewer* iterations, catastrophic on pose paired (40 → 9 Panda, 40 → 2 iiwa). Mechanism: the pose task already pins the end-effector with six equality rows, so lifting adds seven more — thirteen equalities in 27 variables with the flow's badly scaled Jacobian inside seven. **An interior-point method's tolerance for a badly scaled row depends on how many equalities it is already carrying.** The 480-cell replication was deliberately not run. |
| **Jacobian regularization** (Frobenius clipping, Tikhonov/LM damping of the singular values, an SVD floor) | **clear negative, line closed. 310 cells better against 1,509 worse** over ten variants x eight experiments. Not one setting is a significant improvement; the only one not net-worse (`jacobian_max_norm=1000`, +6, p = 0.59) does not reduce the runaway it was introduced to prevent, and the most aggressive damping scores **0/480**. |

**Why damping cannot work, the transferable part.** The flow's Jacobian is the *exact* derivative of
an explicit function. Where the gain approaches its ceiling, a sensitivity of 1e13 is the correct
answer, not an artifact. Damping it breaks the correspondence between the constraint values IPOPT
evaluates and the gradients it is handed, leaving an inconsistent nonlinear program — which is why
the most aggressive damping fails hardest while *increasing* the runaway count. LM damping is sound
applied to the **Newton step** rather than to a reported derivative, but Drake's IPOPT does not
expose the step computation. **Thomas's ruling: *"I think we can conclude gradient regularization and
the other strategies isn't worth it."*** The knobs stay in the tree, off.

A lesson these stages share: **suppressing the runaway does not buy success.** `tikhonov=10` cut
runaway cells 31 → 7 for exactly 19/19 on success; `lift_q` reached zero runaways while losing 78
cells net.

## ONE DRAKE, AND IT IS THE PIN

The nightly is installed at `$ROOT/drake` by `cluster/setup_supercloud.sh` and there is no second
install, no `DRAKE=nightly` per-item sentinel and no `stage_DRAKEBUMP` — all three were deleted on
2026-09-19. Every arm of every campaign runs on the current pin. Thomas, ruling on it for the second
time:

> stop running IPOPT and SNOPT on the installed 1.56.0. I've said this already. The difference
> between 1.56.0 and the current nightly is negligible. I think we've even measured this. Stop making
> this mistake, it's getting tiresome. Run everything on the current nightly. Even if you think I'm
> wrong, it doesn't matter, because we're not going to pin IPOPT and SNOPT back to an earlier version
> as Drake moves ahead — that would be a regression that you report so I can fix in Drake upstream,
> and/or further tuning to fix it.

So **archive pairing is not a reason to keep an old install**: a cross-version caveat is stated once
and a version-induced regression is reported and fixed upstream, never pinned around. **Do not
recreate `stage_DRAKEBUMP`.** The empirical backing was free — stage NLOPTTUNE's `default` column ran
on this nightly against a 1.56.0 archive and reproduced it exactly on eleven of twelve rows.

**Why a nightly at all.** The AL column's inner local optimizer became selectable when **PR 25002**
(merged 2026-09-17, `5a73436c`) took `NloptSolver` from **six** option names to **sixteen**. Neither
set is in a release — 1.57.0 was cut before both and still declares six — so
`drake-0.0.20260918-noble.tar.gz`, whose tip is exactly that merge commit, is the pin. Two
properties of it still bite: cluster Drake versions are per-project, so the install lives inside this
project's tree; and it needs **its own** `drake_models` cache warm, because the cache key includes the
models commit that Drake version pins. Nightlies publish no `.sha256`, so setup verifies against a
hash recorded in the repo — the bytes the local tests ran against. Nightly artifacts **expire after
45 days** (~2026-11-02), so **move the pin to 1.58.0 the moment it carries PR 25002** and restore the
published-checksum path. One caveat before comparing across nightlies: `ik-tune` sees
inner-local-optimizer behaviour differ between the 09-16 and 09-18 nightlies at an identical recorded
configuration, while a control naming no inner optimizer is identical on both.

**The Luksan trap, a new instance of an old one.** Drake's NLopt is built without the LGPL Luksan
sources, so `LD_LBFGS`, the `LD_VAR*` family and every `LD_TNEWTON*` variant are listed by
`ParseNloptAlgorithm` as valid choices and then refused *inside the solve* with `attempting to use
NLOPT_LD_LBFGS, but Luksan code disabled`, returning `kInvalidInput` and status 0. The symptom is
quiet and misleading: a **0.5 s cell with `q=None`, `max_violation=None` and `fail_reason` unset**,
which reads like a harness bug. Exactly eight algorithms are refused; **`LD_MMA`, `LD_CCSAQ`,
`LD_SLSQP`, `LN_COBYLA` and `LN_BOBYQA` all work**, so the usable *gradient-based* inner optimizers
are the first three and nothing else. `NLOPT_LUKSAN_DISABLED` refuses them at configuration time, for
both the outer and inner algorithm.

**This refutes a claim that once stood here**: that NLopt supplies `LD_LBFGS` when the inner
optimizer is unset. It cannot — naming `LD_LBFGS` *fails* while leaving it unset *solves*. So there
is **no "name what NLopt already picks" control available**: `default` against any named inner
algorithm unavoidably mixes "Drake called `set_local_optimizer` at all" with "which algorithm".

Two more Drake behaviours worth not rediscovering. Every `local_optimizer_*` option is read
unconditionally but **applied only inside `if (!parsed_options.local_optimizer_algorithm.empty())`**
(`nlopt_solver.cc:546-564`), so an inner budget or tolerance without a named inner algorithm is
accepted and inert. And an **unknown** NLopt name does not crash a run: Drake accepts it at
`SetOption` and raises from inside `Solve`, which lands in `run_grid`'s per-cell `except Exception`
and is recorded as `fail_reason="error"`, i.e. a **full column of instant failures** rather than an
error. Hence the check lives in `ProgramOptions.__post_init__`, before the first cell.

**Do not extend Drake to get better instrumentation.** Thomas: *"NLOPT might not have the robust
logging we need btw, work with what you have, don't write new logging stuff in Drake or anything."*

## The chart: why the rungs differ and how one is chosen

The iiwa's upstream checkpoint was the project's bottleneck. `iiwa14_ddp_r1` (620k steps) cut
`frac_gt_1000` from 3.34% to 0.0125% but left `pole/max` at 7.8e9 and iiwa grasp at 270/480 — the
headroom was still there, that run had only moved less of the domain into it. The follow-on trained
deliberately **simpler, less accurate** charts: `nb_nodes` 12/8/6/4 lowers the architectural gain
ceiling `exp(2.4975*nb_nodes)` from 1e13 to 2e4, with a width-only rung (12 blocks,
`coeff_fn_internal_size` 256) as the control separating "less accurate" from "less headroom". Nine
runs, both robots, 620k steps each; `cluster/ladder_runs.txt` is the spec.

**Reducing depth from 12 helps on both robots, and the optimum rung differs by robot** — `n4` on the
iiwa, `n6` on the Panda. "The smallest chart wins" was an iiwa-only result.

**The two mechanisms differ by robot.** On the iiwa a smaller chart buys cells by eliminating runaway
configurations; on the Panda, which has no runaway at all (`median_max_violation` 1e-08 on every
rung), it buys them by fitting more iterations inside the fixed cap. Per-iteration cost falls
monotonically with depth — iiwa grasp native 80 / 55 / 42 / 33 ms for n12 / n8 / n6 / n4, with
`n12w256` at 72 holding depth — and timeouts collapse with it. The learned arm's per-iteration
penalty is now **~10-13x, down from ~30x**. That second mechanism is an implementation-and-hardware
property, which is why ms/it must be reported beside success. A net-faster chart that needs more
steps is a **trade-off to report, not a confound to remove** (Thomas: *"More iterations but faster
net is a trade-off, not automatically good or bad"*) — do not redesign to a fixed iteration cap.

**The second mechanism was MEASURED AT A 45 s CAP, so it should shrink at 180 s; the first should
not.** "A cheaper chart fits more iterations inside the cap" is cap-dependent by construction, whereas
"a smaller chart eliminates runaway configurations" is not. **The ladder is deliberately not
re-measured and the question is CLOSED rather than deferred**: nothing touching the charts changed,
the rungs are selected by the gain ceiling rather than by cells so a new grid cannot revise the
choice, and the one condition under which re-measuring would have been worth ~350 core-hours — heavy
timeouts at 180 s, the regime where the cap-dependent half does the work — is answered above:
timeouts are at most 2 cells of 480. Quote the ladder tables as the 45 s / free-grasp record and do
not restate this mechanism as a property of the current status quo.

**Two controls land as intended.** Panda `upstream` and `n12` agree within noise on all rows, so our
training recipe reproduces Jeremy's and no reduced-Panda result is confounded with recipe. And
`n12_w256`, the accuracy-only control, never beats `n12` significantly while keeping the slow
iteration: width costs accuracy without buying either headroom or speed. Depth buys the speed.

**Chart accuracy is a clean monotone dose curve and it runs BACKWARDS to cells.** Median FK error
over 5000 poses, 4/6/8/12 blocks: 20.0 / 12.1 / 11.3 / 10.1 mm on the iiwa, 14.7 / 9.5 / 7.2 / 6.1
mm on the Panda; the width rungs are 75.1 and 22.9 mm. The *least* accurate depth rung solves the
most on the iiwa. 20 mm chart error does not appear in the solutions — the IK constraint is on
`FK(q)`, so `n4`'s solved cells return `median_max_violation` 1.1e-08; the chart only decides where
the solver starts.

**Neither intrinsic screen predicts cells, in either direction.** iiwa `n8` screens cleanest of the
four (task-pose `frac_gt_1000` 0.0, `pole/max` 202) and is the worst rung on pose paired; `n4`
screens dirtier and solves best. A chart can be clean everywhere the sampler looks and catastrophic
everywhere the Newton step goes. **The screen is a smoke test, not a selection criterion.**

**Pole mass is CREATED BY TRAINING, monotonically, on every rung.** Across all 31 kept checkpoints
per rung, task-pose `pole/max` over 20k..620k steps runs 27 → 2.5e3 (iiwa `n4`), 409 → 1.3e5 (iiwa
`n6`), 50 → 3.5e11 (Panda `n12`). The headroom is present at initialisation and barely used; SGD
walks the network into it while buying accuracy. Accuracy itself is converged by ~480k.

**Stage TRAJ measured the training-step axis directly: training makes the iiwa `n6` chart WORSE and
does nothing at all for `n4`.** Exact McNemar, earliest checkpoint against 620k: `n6` **loses three
of four rows to its own first checkpoint** — grasp native 22/159 (p = 8.3e-27), grasp paired 15/142
(p = 4.1e-27), pose paired 63/162 (p = 3.1e-11); `n4` moves on **none** of the four (p = 0.088 to
1.0). `n4` at 20k has ~78 mm median FK error against 620k's 20 mm — a 4x accuracy gain over 600k
steps, worth **zero cells** — while `n6` buys the same accuracy and *pays* 137 grasp cells, its
`pole/max` climbing 409 → 1.3e5 and its pose-paired `median_max_violation` crossing 1e-08 → **1e+03**
between 240k and 400k. **The sharpest statement the campaign has that accuracy is not the quantity
that matters.** It is **not** a licence to pick early checkpoints: that is selecting on the test set
and confounds architecture with selection.

## What is fielded, and what each adoption is and is not

Exactly **two** solver settings are fielded anywhere in this project, both adopted 2026-09-19.
Everything else — all 16 `ipopt_*` fields, 15 of 19 `snopt_*`, 10 of 15 `nlopt_*`, and the four
remaining step-rejection knobs — stays plumbed and `None`.

**SNOPT: `snopt_major_step_limit` defaults to 0.5** (Drake/SNOPT's own default is 2.0). **It was
adopted for FAIRNESS, not for the comparison, and that distinction must survive into the write-up.**
On the learned-vs-joint-space question it changes nothing: five learned wins, four joint-space wins
and three ties before and after, with **zero verdict flips** and the same rows in each bucket. It
gives the learned arm +161 cells of 5,760 and joint space +164 — the same size — and the per-row
margins (L - JS) move between -32 and +15 and sum to **-3**. What justifies it is that IPOPT's column
runs a tuned configuration while SNOPT's ran bare Drake defaults. **Do not present it as helping the
learned formulation.** It is also a property of SNOPT rather than of the chart: the joint-space arm
improves on all twelve rows.

Two consequences. **The SNOPT numbers of record are stage SNOPTCOMBO's `mstep0p5` column**, not
`sc_SOLVER2_*_snopt_*`, which was measured at Drake's defaults. And **"set nothing" no longer means
Drake's SNOPT defaults** — a stage whose column means that must say `--set
snopt_major_step_limit=None`, which emits the option not at all. `tests/test_solver_plumbing.py`
pins both directions.

**NLopt: `LD_AUGLAG` + `LD_MMA` inner + inner `xtol_rel = ftol_rel = 1e-3`.** Fielded on Thomas's
criterion — *"Feasibility is the name of the game, objective cost is secondary."* Against Drake's NLopt
defaults on the learned arm it is better on 5 of 12 rows, worse on 1, unchanged on 6 (all six rows
where nothing solves), and it collapses both the residual and the work per cell — Panda grasp contained
native 12 -> 42 of 60 (p = 1.9e-09) at 3532 -> 66 network Jacobians is the clearest instance.

**Three things must be reported with it.** It **failed** stage NLOPTTUNE's pre-registered gate, which
asked whether to spend 480-cell compute and not whether the setting is the best configuration —
state the gate failure alongside the setting so it does not read as a configuration chosen where it
helps. The **Panda pose native row is a genuine regression on the adoption's own criterion**: five
cells one-directionally and a residual four times worse (9.3e-07 -> 3.6e-06), and that is the honest
cost. And it **flips one learned-vs-joint-space verdict** (Panda pose paired, tie -> learned win),
unlike the SNOPT adoption which flipped none.

**A caveat we are not re-sweeping.** The sibling `ik-tune` project, sweeping the same inner tolerance
on its own problems, puts the optimum near **1e-4** and finds loosening past it costs — and reached
that only after discovering its control had been running at an *implicit* 1e-4. So our adopted 1e-3
is in a sensible region but is **not** an optimum this project established; it is the value NLOPTTUNE
happened to field. Revisiting it is Thomas's call.

**`LD_AUGLAG` is kept, and the reason is checkable — but the mechanism is in NLopt, not in Drake.**
Under `LD_AUGLAG` the inner optimizer solves a **bound-constrained** subproblem and every constraint
sits in the augmented-Lagrangian penalty, whatever the inner algorithm's own method class. That
retires the `LD_SLSQP` taxonomy worry, and it is why `LD_AUGLAG_EQ` is **not** used: `_EQ` absorbs
only equalities and enforces inequalities on the subproblem directly, and this program carries both
kinds. What decides it is the algorithm name alone, verified in the nlopt bundled in Drake:
`src/api/optimize.c:934` passes `sub_has_fc` computed purely from the enum, and
`src/algs/auglag/auglag.c:98-101` branches on it (`if (sub_has_fc) d.m = 0; else m = 0;`), i.e.
inequalities go either into the penalty or onto the subproblem, never both.

**An earlier version of this argued from Drake's side — that Drake never adds a constraint to the
inner `local_opt` — and that reasoning must not come back.** It is true and causes nothing:
`auglag.c:110-119` overrides the subproblem's objective, bounds and stopval, removes its constraints
and repopulates from the *outer* problem's list, so whatever Drake put on `local_opt` is discarded.
The wrong chain also predicts the wrong thing, since under `LD_AUGLAG_EQ` the subproblem is *not*
bound-constrained even though Drake adds nothing to `local_opt`.

**A live hypothesis this hands us, not acted on.** `ik-tune` measured both parents across six robot
experiments and found the plain-vs-`_EQ` difference large and one-directional: its two experiments
whose *inequality* structure carries the problem collapse under plain `LD_AUGLAG` at every encoding
and inner tolerance tried (14.4-19.8% against 54-63% under `_EQ`). Our iiwa grasp rows are 0-3 of 60
under every NLopt setting and at 180 s, and the grasp task is exactly where the collision inequality
binds — so penalised-rather-than-enforced inequalities is now a mechanism candidate for a row this
project had recorded only as inexplicable. Their problems are not ours, so this is a thing to watch
rather than a prediction. **Switching to `LD_AUGLAG_EQ` is a method-class decision and therefore
Thomas's**, and the reason recorded above for preferring one honest augmented Lagrangian is unchanged
by any of it.

**Requires a Drake carrying PR 25002.** `local_optimizer_ftol_rel` is absent from 1.56.0 and from a
pre-PR source build, so `CheckNloptOptions`'s availability refusal is scoped to
`which_solver == "nlopt"`; unconditional, it would refuse `ProgramOptions()` itself and kill every
IPOPT and SNOPT cell over options they never emit.
