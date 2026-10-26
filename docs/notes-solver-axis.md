# The solver axis (IPOPT / SNOPT / NLopt)

The long form of a part of `CLAUDE.md`, moved here verbatim on 2026-10-18 to bring `CLAUDE.md` under
40k characters. `CLAUDE.md` keeps the verdicts and rulings; this file keeps the mechanisms, numbers and
history behind them. "Above"/"below" and section names refer to the old single-file layout, now split
across `docs/notes-*.md`. A reference elsewhere to "CLAUDE.md's ... section" means the matching notes file.

## The solver axis: interior point, SQP, augmented Lagrangian

**Three METHOD CLASSES, not three vendors.** Thomas: *"for NLOPT, we want to use it as an augmented
lagrangian solver. This is important! We don't care about SLSQP, since SNOPT is already SQP. The
point is that we test interior point, augmented lagrangian, and SQP."* So `--solver` is `ipopt`,
`snopt`, and `nlopt` (**`LD_AUGLAG`** -- not `LD_AUGLAG_EQ`, which absorbs only equalities and leaves
inequalities to the inner solver, and this program carries both). **The solver is a reporting axis,
never a choice**: *"we would not pick one solver or the other, but rather report the performance for
both solvers."* Shared across arms within a run, varied across runs.

**The whole axis is CLOSED on all three method classes. Do not re-sweep any of them.** Evidence, the
per-stage arithmetic and the refuted mechanisms live in **`docs/closed-axes.md`**; each stage's reader
(`scripts/report_{snopttune,snoptcombo,nlopttune,step}.py`) regenerates its tables with its
pre-registered rule inline so it cannot drift.

### Each solver at its own defaults, and why that is a CHOICE

Unset `snopt_*`/`nlopt_*` fields are not passed, so each solver sits at its own defaults and the
shared, deliberately looser task gate decides success. **Do not transplant one solver's option values
onto another** -- the SNOPT branch once read IPOPT's `acceptable_*` family (its relaxed early stop,
not what it converges to) as SNOPT's Major tolerances.

But "own defaults" is not symmetric. `tol`, `constr_viol_tol`, `dual_inf_tol` and `compl_inf_tol`
were never `ProgramOptions` fields, so every archived run gave IPOPT `constr_viol_tol` **1e-4**
against SNOPT's **1e-6** and `dual_inf_tol` **1** against **2e-6** -- 100x the constraint violation
and six orders more dual infeasibility, plus an early stop SNOPT has no counterpart for. A property
of the HARNESS, not of interior-point methods. All four are now plumbed. The task gate is untouched
by any of it.

### What each solver will and will not tell you

**No solver reports an iteration count through Drake.** IPOPT's comes from its print file; SNOPT's
`details` carries `info`, `solve_time` and multipliers but no count; `NloptSolverDetails` carries a
single `status` and NLopt writes no print file at all (`kPrintFileName` is silently ignored).
`iterations` means **majors** under both solvers that report it. snOptA has one user function, so
IPOPT's four eval counts have no SNOPT counterpart and stay `None` rather than being filled with a
different quantity.

**The program counts evaluations itself** in `QAndPose`, the one funnel every arm's solve passes
through. `map_jacobian` is the AutoDiffXd count -- the NLopt column's only work measure -- and it
cross-validates: on the SNOPT smoke run it equalled `User function calls (total)` exactly on every
cell. It is NOT an iteration count, and it is **not comparable across arms**, being the identity map
on the joint-space side. **Status is decoded numerically, not from log text**: SNOPT INFO 34 is the
time limit with no distinctive exit string, NLopt has no text at all, so matching strings would report
`timeouts: 0` for a capped run of either. NLopt status 5 is `MAXEVAL_REACHED`, recorded as
`hit_eval_cap`.

### The result: IPOPT > SNOPT >>> NLopt

**A CONFIRMATION, not a finding.** Thomas: *"IPOPT is more robust to ill-posed problems, and our
neural network gradients are definitely ill-posed. I expect to see IPOPT > SNOPT >>> NLOPT."* Write it
up as the size and mechanism of a predicted gap. On stage SOLVER2's 480-cell grid IPOPT won all 24
rows, 23 significantly.

**It is a property of the PROBLEM, not of the learned formulation** -- the joint-space arm degrades
under SNOPT too, on every row and by comparable margins, and that arm never evaluates the network.
The same holds for every solver setting: each moves both arms.

**The mechanism, and it is not budget.** Pooled over all 3,952 SNOPT failures: INFO 13 (nonlinear
infeasibilities minimized) **50.7%**, INFO 41 (current point cannot be improved) **36.2%**, iteration
limit 9.5%, time limit **3.6%** -- about 87% convergence failures against 11% budget, and SNOPT times
out *less* often than IPOPT. The two fail in opposite ways: IPOPT's learned-arm failures are mostly
wall-clock, still descending; SNOPT's are INFO 41, giving up at a feasible-but-wrong point. **INFO 41
is the documented signature of inaccurate or badly scaled derivatives** -- ordinary ill-conditioning
of an *exact* Jacobian, not the gain-ceiling runaway. The gap is much larger under `paired`, so SNOPT
copes far worse with an infeasible start; a 60-cell triage read that backwards in both directions
before 480 cells settled it, so **do not draw a protocol conclusion from 60 cells**.

**Wall-clock columns are not capped equally across solvers**: NLopt's `max_time` binds much more
tightly than SNOPT's `Time limit` (20.0-20.1 s against 24-28 s), because SNOPT only checks at major
boundaries.

### What tuning is worth: two settings fielded, everything else plumbed and `None`

**Per-solver tuning is permitted and per-problem tuning is not** (*"I'm okay with playing with solver
settings on a per-solver basis, as long as it's not per-problem"*), so a setting qualifies only by
winning **uniformly across rows**, **counted by row, never pooled** -- the transferable lesson, since
a four-factor SNOPT stack beats the fielded survivor by +221 pooled cells and is a **task trade**
pooling hides, gaining on every iiwa grasp row and losing on every pose row.

The verdicts: IPOPT's convergence tolerances are **inert** and its acceptable-point machinery is
everything -- at each solver's own defaults **IPOPT 200, SNOPT 141, p = 4.1e-09**, so the fairness
question is answered and the ordering survives it. IPOPT's early stop is worth 39 cells and a 9x
speedup and is **not** returning sloppy points (disabling it drives the violation to 2.22e-15 and
success *down*, because IPOPT then polishes a solution it already has until the clock kills it).
Naming NLopt's inner optimizer is worth nothing; truncating its inner tolerances is worth a great
deal, as a trade whose sign flips by row. **Three things no NLopt setting changes**: the iiwa grasp
rows are 0-3 of 60 under every setting and at 180 s (on the defective wsg scene; fixed, native is 313 of 480); the joint-space arm moves with these settings
too; the ordering is untouched.

**Exactly two settings are fielded**, both adopted 2026-09-19.

**SNOPT `snopt_major_step_limit = 0.5`** (Drake's own default is 2.0), sole survivor of thirteen, and
no combination beats it. **Adopted for FAIRNESS, not for the comparison** -- five learned wins, four
joint-space wins and three ties before and after, **zero verdict flips**, +161 cells to learned and
+164 to joint space, per-row margins summing to **-3**. What justifies it is that IPOPT's column ran
a tuned configuration while SNOPT's ran bare Drake defaults. **Do not present it as helping the
learned formulation.** Two consequences: the SNOPT numbers of record are stage SNOPTCOMBO's
`mstep0p5` column, and **"set nothing" no longer means Drake's SNOPT defaults** -- a stage meaning
that must say `--set snopt_major_step_limit=None`.

**NLopt `LD_AUGLAG` + `LD_MMA` inner + inner `xtol_rel = ftol_rel = 1e-3`**, on Thomas's criterion
*"Feasibility is the name of the game, objective cost is secondary."* Better on 5 of 12 rows, worse
on 1, unchanged on 6, and it collapses both residual and work per cell (Panda grasp contained native
12 -> 42 of 60 at 3532 -> 66 network Jacobians). **Three things must be reported with it**: it
FAILED stage NLOPTTUNE's pre-registered gate, which asked whether to spend 480-cell compute rather
than whether the setting is best -- state the gate failure alongside it; the Panda pose native row is
a genuine regression on the adoption's own criterion (five cells, residual 9.3e-07 -> 3.6e-06); and
it flips one learned-vs-joint-space verdict, unlike the SNOPT adoption which flipped none. A caveat
we are not re-sweeping: the sibling `ik-tune` project puts the same inner tolerance's optimum near
1e-4, so our 1e-3 is in a sensible region but is not an optimum this project established.

**`LD_AUGLAG` is kept for a reason in NLopt, not in Drake**: under it the inner optimizer solves a
**bound-constrained** subproblem and every constraint sits in the AL penalty, whatever the inner
algorithm's own class -- which retires the `LD_SLSQP` taxonomy worry and is why `_EQ` is not used.
Verified in the bundled nlopt source. **An earlier version argued from Drake's side -- that Drake
never adds a constraint to the inner `local_opt` -- and that reasoning must not come back**: true,
causes nothing, and predicts the wrong thing under `_EQ`. A live hypothesis this hands us, not acted
on: `ik-tune` finds the plain-vs-`_EQ` difference large and one-directional, and our iiwa grasp rows
are exactly where the collision inequality binds -- so penalised-rather-than-enforced inequalities is
a mechanism candidate. **Switching is a method-class decision and therefore Thomas's.**

### Step rejection: measured and refuted (stage STEP), with one real finding

A 16-setting screen at 60 cells reached significance nowhere; the 480-cell confirmation **refuted
both promoted settings** -- `theta1` is significantly *worse* (p = 0.00032) and `soc0` is a clean
null. **The mechanism column reversed too**, so a plausible mechanism agreeing with a spurious
outcome did not protect against the false positive. Nothing adopted, by Thomas's call made in
advance.

What it exposed is the real finding: **per-cell outcomes are unstable.** Pooled over 5,760
comparisons, a single filter-option change recovers ~70% of the learned arm's residual failures while
breaking 6-8% of the cells that already worked; successes outnumber failures ~12:1, so the two cancel
almost exactly. A same-configuration control flips 5.9%/0.3%, and even failures that converged wrong
rather than timing out recover at 48-67%. So the residual failures are **not intrinsically hard**,
and no single global setting wins because the reshuffle is symmetric in proportion. **The implication
is for multi-start, which Thomas ruled out of scope** -- report the instability as a finding, do not
propose acting on it.

**A methodological number worth keeping**: "reproducibility at the cap is +/-1 cell" was measured on
60-cell grids. At 480 cells with 52-88 timeouts a same-configuration re-run moves up to **7 net and
19 discordant** cells; every one of the 41 discordant cells across 12 such rows was cap-bound, and
rows with zero timeouts had zero discordance. The band scales with the cap-bound population, not the
grid.

### Traps, all found by probing rather than by reading

**A solver option can be ACCEPTED and do nothing**, and only the solver's own parameter echo tells
the two apart -- `"Timing Level"` was set for the life of this repo and silently wrote no timing
block, because SNOPT's parser is case-sensitive on the second word while Drake raises only on
keywords SNOPT does not know at all. Every option this branch exposes was verified reaching its
solver (IPOPT via `used = yes` in `print_user_options`, SNOPT via the parameter echo). It caught
three: `Hessian updates` is inert below 75 variables; `Nonderivative linesearch` is a VALUELESS
keyword, so passing 0 turns it ON; and `linear_solver=mumps` does not exist, Drake's IPOPT being
built against SPRAL.

**Read a solver's defaults out of the solver.** IPOPT's own dump contradicts the obvious assumption:
`acceptable_dual_inf_tol` defaults to **1e+10** and the two acceptable infeasibility tolerances to
**1e-2**. A sweep arm labelled "IPOPT's defaults" that was not cost a resubmission.

**`max_eval` is not "unset" by default** -- Drake defaults it to 1000, a cap that binds here, so it
must be set explicitly or the NLopt column silently measures an evaluation budget rather than the
wall clock.

**The Luksan trap.** Drake's NLopt is built without the LGPL Luksan sources, so `LD_LBFGS`, the
`LD_VAR*` family and every `LD_TNEWTON*` variant are listed as valid and then refused *inside* the
solve, returning `kInvalidInput` and status 0. The symptom is quiet and misleading: a **0.5 s cell
with `q=None`, `max_violation=None` and `fail_reason` unset**, which reads like a harness bug.
`LD_MMA`, `LD_CCSAQ`, `LD_SLSQP`, `LN_COBYLA` and `LN_BOBYQA` all work. It also means there is **no
"name what NLopt already picks" control available**. Two more Drake behaviours: every
`local_optimizer_*` option is read but applied only when an inner algorithm is NAMED, so an inner
budget without one is accepted and inert; and an **unknown** NLopt name is accepted at `SetOption`
and raises from inside `Solve`, landing in `run_grid`'s per-cell `except` as a **full column of
instant failures** rather than an error. Hence the check lives in `ProgramOptions.__post_init__`,
before the first cell, and `tests/test_solver_plumbing.py` bounds the emitted keys by the *running*
Drake's surface.

**Do not extend Drake to get better instrumentation** -- *"NLOPT might not have the robust logging we
need btw, work with what you have, don't write new logging stuff in Drake or anything."*

### ONE DRAKE, AND IT IS THE PIN

The nightly `drake-0.0.20260918-noble.tar.gz` at `$ROOT/drake`; no second install, no per-item
version sentinel, no `stage_DRAKEBUMP` (all three deleted 2026-09-19). Thomas, ruling for the second
time:

> stop running IPOPT and SNOPT on the installed 1.56.0. I've said this already. ... Run everything on
> the current nightly. Even if you think I'm wrong, it doesn't matter, because we're not going to pin
> IPOPT and SNOPT back to an earlier version as Drake moves ahead -- that would be a regression that
> you report so I can fix in Drake upstream, and/or further tuning to fix it.

So **archive pairing is not a reason to keep an old install**. A nightly is needed at all because
**PR 25002** took `NloptSolver` from six option names to sixteen, which is what made the inner local
optimizer selectable; no release carries it, and `local_optimizer_ftol_rel` is absent from 1.56.0, so
`CheckNloptOptions`'s refusal is scoped to `which_solver == "nlopt"` -- unconditional, it would refuse
`ProgramOptions()` itself and kill every IPOPT and SNOPT cell over options they never emit. Nightly
artifacts **expire after 45 days (~2026-11-02)**, so **move the pin to 1.58.0 the moment it carries
PR 25002** and restore the published-checksum path.

**One stated exception, for the svgd rounds only (Thomas, 2026-10-11).** The batched collision row
needs `CollisionChecker::CalcRobotClearances`, which the pinned nightly predates (master since #25073,
2026-10-12), so a wheel built from his
branch `calc-robot-clearances` (`0.0.20261011+crc.15a2c97`) lives in a SECOND cluster venv,
`~/learned-ik/venv-crc`, selected per submission by `DRAKE_ENV=crc` (`cluster/run_items.sh`,
`cluster/submit_bench.sh`; refused on a manifest without the clearance row and vice versa). The
IPOPT / SNOPT / NLopt record stages and every svgd IPOPT twin stay on the pinned nightly; a run
records which Drake it ran under (`drake_version_txt`). It is a strict speed improvement (the row is
bitwise the same functional, 7-10x per configuration), not a second pin: when Drake ships the API,
the wheel and the venv go.

### The grasp-containment lever is closed on both axes

It was the standing "revisit whenever another knob moves" lever, because on the contained task joint
space needs 970 median iterations against 48 on the free one. Under SNOPT it never flips a verdict
*toward* the learned arm and flips one away from it. Step rejection was the remaining candidate and is
refuted too, so the containment verdicts stand as measured under IPOPT.

