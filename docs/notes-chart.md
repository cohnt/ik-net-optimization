# The chart ladder and the gain ceiling

The long form of a part of `CLAUDE.md`, moved here verbatim on 2026-10-18 to bring `CLAUDE.md` under
40k characters. `CLAUDE.md` keeps the verdicts and rulings; this file keeps the mechanisms, numbers and
history behind them. "Above"/"below" and section names refer to the old single-file layout, now split
across `docs/notes-*.md`. A reference elsewhere to "CLAUDE.md's ... section" means the matching notes file.

## The chart: why the rungs differ and how one is chosen

The iiwa's upstream checkpoint was the project's bottleneck, so a follow-on trained deliberately
**simpler, less accurate** charts: `nb_nodes` 12/8/6/4 lowers the architectural gain ceiling
`exp(2.4975*nb_nodes)` from 1e13 to 2e4, with a width-only rung as the control separating "less
accurate" from "less headroom". Nine runs, both robots, 620k steps each; `cluster/ladder_runs.txt` is
the spec and `docs/closed-axes.md` holds the per-rung tables.

**Reducing depth from 12 helps on both robots, and the optimum rung differs by robot** — `n4` on the
iiwa, `n6` on the Panda. "The smallest chart wins" was an iiwa-only result.

**The two mechanisms differ by robot.** On the iiwa a smaller chart buys cells by eliminating runaway
configurations; on the Panda, which has no runaway at all, it buys them by fitting more iterations
inside the fixed cap. Per-iteration cost falls monotonically with depth (iiwa grasp native 80 / 55 / 42
/ 33 ms for n12 / n8 / n6 / n4, the width rung holding depth at 72) and timeouts collapse with it. That second mechanism is an
implementation-and-hardware property, which is why ms/it must be reported beside success. A net-faster
chart that needs more steps is a **trade-off to report, not a confound to remove** (Thomas: *"More
iterations but faster net is a trade-off, not automatically good or bad"*) — do not redesign to a fixed
iteration cap.

**The second mechanism was MEASURED AT A 45 s CAP, so it should shrink at 180 s; the first should not.**
**The ladder is deliberately not re-measured and the question is CLOSED rather than deferred**: nothing
touching the charts changed, the rungs are selected by the gain ceiling rather than by cells so a new
grid cannot revise the choice, and the one condition under which re-measuring would have been worth
~350 core-hours — heavy timeouts at 180 s — is answered: timeouts are at most 2 cells of 480. Quote the
ladder tables as the 45 s / free-grasp record.

**Two controls land as intended.** Panda `upstream` and `n12` agree within noise on all rows, so our
training recipe reproduces Jeremy's and no reduced-Panda result is confounded with recipe. And
`n12_w256`, the accuracy-only control, never beats `n12` significantly while keeping the slow iteration:
width costs accuracy without buying either headroom or speed. Depth buys the speed.

**Chart accuracy is a clean monotone dose curve and it runs BACKWARDS to cells.** Median FK error over
5000 poses, 4/6/8/12 blocks: 20.0 / 12.1 / 11.3 / 10.1 mm on the iiwa, 14.7 / 9.5 / 7.2 / 6.1 mm on the
Panda. The *least* accurate depth rung solves the most on the iiwa. 20 mm chart error does not appear in
the solutions — the IK constraint is on `FK(q)`, so `n4`'s solved cells return `median_max_violation`
1.1e-08; the chart only decides where the solver starts.

**Neither intrinsic screen predicts cells, in either direction.** iiwa `n8` screens cleanest of the four
and is the worst rung on pose paired; `n4` screens dirtier and solves best. A chart can be clean
everywhere the sampler looks and catastrophic everywhere the Newton step goes. **The screen is a smoke
test, not a selection criterion.**

**Pole mass is CREATED BY TRAINING, monotonically, on every rung.** Task-pose `pole/max` over
20k..620k steps runs 27 -> 2.5e3 (iiwa `n4`), 409 -> 1.3e5 (iiwa `n6`), 50 -> 3.5e11 (Panda `n12`). The
headroom is present at initialisation and barely used; SGD walks the network into it while buying
accuracy, which is itself converged by ~480k.

**Stage TRAJ measured the training-step axis directly: training makes the iiwa `n6` chart WORSE and does
nothing at all for `n4`.** `n6` **loses three of four rows to its own first checkpoint** (exact McNemar,
p = 8.3e-27 to 3.1e-11); `n4` moves on **none** of the four. `n4` at 20k has ~78 mm median FK error
against 620k's 20 mm — a 4x accuracy gain over 600k steps, worth **zero cells** — while `n6` buys the
same accuracy and *pays* 137 grasp cells. **The sharpest statement the campaign has that accuracy is not
the quantity that matters.** It is **not** a licence to pick early checkpoints: that is selecting on the
test set and confounds architecture with selection.

### The chart-selection rule

**Choose an architecture whose gain ceiling `exp(2.4976 * nb_nodes)` sits below the ~1e7 runaway band
— on the iiwa that is `nb_nodes = 4` — then take the final checkpoint, because below the ceiling the
training-step axis is flat and above it later checkpoints are strictly worse.**

The ceiling is `atan` and the block count, nothing else: FrEIA's coupling block is `y = exp(s)*x + t`
with `s = clamp * 0.636 * atan(s_raw)`, so `|s| < clamp*0.636*pi/2 = 2.4976` at `rnvp_clamp = 2.5`
and one block amplifies by at most 12.15x. Over `nb_nodes`: 2.2e4 / 3.2e6 / 4.8e8 / 1.0e13 at
n4/n6/n8/n12. It bounds the weights, so no amount of training escapes it — and **training reliably
walks 78-89% of the way up whatever log-ceiling it is given**, so the ceiling predicts where a
trained chart lands rather than merely bounding it. The configurations that actually kill solves are
**1e7 to 1e16 rad**; `frac_gt_1000`'s threshold is a bimodality separator (ordinary configurations
~2.5 rad), *not* the level at which a solve dies — `n4`'s ceiling is above 1000 and is fine. 6→4 is a
cliff, not a dose curve. Every competing criterion is disqualified by measurement (accuracy,
intrinsic screening, "take the last checkpoint", "take whichever benchmarks best"). If a chart above
its ceiling ever *must* be selected among, the only sound version is a held-out optimization grid.

**`rnvp_clamp = 2.5` is confirmed correct**. Sweeping it against the same weights, fraction `> 1000`: 0.893 / 0.410 / 0.090 / **0.035** /
0.126 / 0.999 at clamp 1.0 / 1.5 / 2.0 / **2.5** / 3.0 / 5.0. A clear optimum.
## The gain ceiling: the project's central scientific finding

This explains the residual learned-arm failures on the full-depth charts and the historical iiwa
grasp deficit.

**The violated binding is `AllIKFlowConstraints`, and inside it the joint-limits row.** On cells that
never converge, `max_violation` equals `|q|_inf` exactly (Spearman 1.0 on 55 of 64 badly violating
cells), with joint angles of **1e7 to 1e16 radians**. Everything else follows: a configuration of
1e8 rad puts the gripper anywhere, so "deeply in collision, a metre off target" is a *consequence*,
and the collision penalty is a bystander (Spearman −0.15).

**Every runaway lies on one ray, and it is a property of the network, not of the solve.** Normalising
the exploded `q` vectors: the iiwa's ray is `[.001, -.001, .016, -.000, .003, .978, -.208]` and the
Panda's `[-.016, .033, .998, -.032, -.002, .027, .023]`, both at pairwise `|cos| = 1.0000` across two
tasks and both protocols — dominated by a single joint (iiwa wrist, Panda elbow).

**Sampling the network directly reproduces it, with no Drake and no solver involved.** Draw `c`
uniformly in its ±0.25 m box, a uniform unit quaternion, and `z` uniformly in the ball of radius 4.3
— strictly inside the allowed region — and evaluate `MakeFlowInference` in float64: median `|q|_inf`
~2.5-2.65, but p100 of 20000 is 4.1e+12 (Panda) and 5.5e+16 (iiwa), with **fraction > 1000 rad
0.00065 (Panda) against 0.0334 (iiwa) — a factor of 51** — and the ray recovered from those samples
matches the one the solver landed on to `|cos|` 0.9976-0.9999. The distribution is bimodal, not
heavy-tailed: a draw is either an ordinary configuration or it is astronomical.

**The mechanism is architectural headroom, not a numerical bug** — the gain ceiling above. These are
near-worst-case gain regions of a bounded map, not poles. Both checkpoints have the same headroom;
they differ only in how much of the conditioning domain sits near it.

**Why the solver finds a 3%-measure set 30% of the time.** It does not sample; it follows gradients,
and `dq/dvars` in these regions is as large as `q` is, so a Newton step is *attracted* to them. That
is also why more budget never helps: cap-bound cells at 180 s are the same cells that were cap-bound
at 20 s. **The learned arm does control `q`** — Thomas: *"the network does get to control the joint
limits a bit, since it can adjust z. That's the whole point of differentiating through the network —
we take the constraint gradient for joint limits and pull it back through the network to z."* What
differs from the baselines is *when* the limits hold (only at convergence), and that the gradient
into a high-gain region is itself enormous.

**Neither region knob avoids them, because they are not at the edges.** Fraction of the region with
`|q|_inf > 1000`: the latent trust-region radius over 1.0..8.0 leaves the iiwa at 0.026-0.071 and the
Panda flat at 0.0018 — shrinking to `R = 1` changes nothing, because the blow-up regions are spread
through the domain including at `|z| <= 1`. **This is why the trust-region sweep measured inert.**
`c_position_slack` is flat from 0.05 to 0.25 and then a **cliff** at 0.5, where exposure jumps 6x on
the iiwa and 76x on the Panda; the default sits just under it by luck, worth knowing before anyone
widens it.

### Every optimization-side remedy has been measured and refuted

Thomas's ranking: **(1) a better chart is preferred over everything else** — *"All of these actions are
less preferred than just getting a better iiwa chart"*; (2) lifting `q` into a bounded decision
variable, permitted but disliked (*"we're effectively adding a nonlinear equality constraint"*); (3) a
joint-limit penalty, permitted but disliked. **"You can try it, but I don't like it" means measure it
and report it as a stated deviation, not adopt it if the numbers look good.**

All four lines are closed and net-negative or inert: **chart accuracy** (`chart_error_scale`) is **not
the mechanism** — degrading the Panda's chart to the iiwa's accuracy costs it 1-3 cells of 60 where the
iiwa is 23 worse, so the standing chart-accuracy hypothesis does not survive its own experiment;
**IPOPT scaling** is inert across five experiments, so this is not a scaling artefact; a **joint-limit
penalty** is inert at 1/10/100, vindicating Thomas's objection that a penalty on a quantity a
constraint row already governs buys nothing; and **lifting `q`** is net negative (38 better against 116
worse, p = 2.2e-10), delivering 0 runaway cells in all eight experiments while **the runaway does not
stop, it relocates** into `max_violation`. **Jacobian regularization** is a clear negative over ten
variants — 310 cells better against 1,509 worse — and Thomas's ruling closed it: *"I think we can
conclude gradient regularization and the other strategies isn't worth it."* The knobs stay in the tree,
off. Per-remedy numbers: `docs/closed-axes.md`.

**Why damping cannot work, the transferable part.** The flow's Jacobian is the *exact* derivative of an
explicit function. Where the gain approaches its ceiling, a sensitivity of 1e13 is the correct answer,
not an artifact. Damping it breaks the correspondence between the constraint values IPOPT evaluates and
the gradients it is handed, leaving an inconsistent nonlinear program — which is why the most aggressive
damping fails hardest while *increasing* the runaway count. LM damping is sound applied to the **Newton
step** rather than to a reported derivative, but Drake's IPOPT does not expose the step computation.

**A lesson these stages share: suppressing the runaway does not buy success.** `tikhonov=10` cut runaway
cells 31 -> 7 for exactly 19/19 on success; `lift_q` reached zero runaways while losing 78 cells net.

### Step rejection (stage STEP)

Measured and refuted; the verdict and the per-cell instability finding are under the solver
axis above, the evidence in `docs/closed-axes.md`.


## The cVAE chart ablation: verdict (branch `chart-cvae`, closed 2026-10-21)

The record's learned formulation with the flow chart replaced by a standard conditional VAE decoder
`q = D(c, z)`, nothing else changed. Two objectives (learned decoder variance `cvae<K>`; fixed
variance, the textbook MSE + KL, `cvaef<K>`) x two latent widths (K = the flow's width; k = the
redundancy), IPOPT and AL-SVGD (pin V4), 480 cells, record conditions. Tables
`docs/chart-ablation-tables.md`; ops `cluster/CVAE_RUNBOOK.md`.

**Rigid arms: a loss on every row, under both solvers.** Native start, success of 480 (flow 480 on
every row under both solvers except IPOPT Panda pose 458 / iiwa pose 462); columns cVAE-K / cVAE-k /
cVAEf-K / cVAEf-k:

| row | IPOPT | AL-SVGD |
| --- | --- | --- |
| Panda grasp | 353 / 228 / 251 / 146 | 350 / 245 / 196 / 154 |
| Panda pose | 58 / 44 / 8 / 6 | 179 / 117 / 51 / 41 |
| iiwa grasp | diverged / 415 / 465 / 343 | diverged / 432 / 444 / 318 |
| iiwa pose | diverged / 2 / 22 / 7 | diverged / 70 / 277 / 213 |

Every cVAE row loses to the flow (McNemar p <= 6e-5) and to joint space, except IPOPT iiwa grasp
cVAEf-K, level with joint space (26 v 14, p = 0.08). No row is budget-bound. Under IPOPT the
learned-variance rows fail by runaway (next section); under AL-SVGD (runaways resampled) nearly
every failure is a cell that never reached a feasible point inside the clock. AL-SVGD lifts pose
(iiwa fixed variance 7-22 -> 213-277) but reaches the ceiling nowhere. Cost on shared cells
(median cVAE / median flow) is mixed -- 0.67-0.98 on IPOPT Panda rows and iiwa grasp, 1.04-1.98 on
AL-SVGD rows except Panda fixed variance (0.85-0.93) -- and is taken over the subset of cells the
cVAE solved, so it carries no verdict where success is this far below the flow's. AL-SVGD wall is
1.4-37x the flow's. **Screw**: 0/60 on every native probe row, both solvers
(not extended). **Paired start**: 0/480 on every row of every robot (soft PCS learned variance 1-2)
-- the encoder round trip lands 0.75-2.8 rad from the start, far outside the +-0.1 correction box.

**Soft PCS arm**: success ties the flow on every native row, both solvers, and the per-cell cost is
lower (section below) -- because the latent is collapsed and the decoder returns a low-strain
configuration, not because the chart is better.

**Mechanism**: every chart's latent is collapsed or nearly so (last section); a standard cVAE
decoder trained on these datasets does not represent the self-motion manifold, which is the property
the formulation needs. Reading: the simplest network that nominally represents self-motion does not
do so once trained, so it is not a usable chart; which property of the flow (exact inverse,
bijectivity, the likelihood objective) is responsible is not separated by this rung.

## The cVAE chart's runaways (chart ablation, 2026-10-20)

The learned-variance cVAE rows fail by the same signature as the flow: on the six full IPOPT native
rows, 87-405 failed cells per row end at `|q|_inf` 1e3-6e7 rad, every row's exploded `q` on **one
ray** (pairwise `|cos|` >= 0.996, top direction >= 99.9% of the spread), dominated by one joint (Panda
joint 4, iiwa joint 3 — not the flow's joints). The mechanism is different:

- **It is extrapolation outside the `c` box, not a gain region inside it.** Every runaway cell
  violates `CBoxConstraint` (median violation 2.2 m to 1.1e4 m; the box is a general constraint, so
  IPOPT's iterates may leave it) and usually the latent trust region. The decoder is a SiLU MLP with
  a linear output, asymptotically linear in its input, so `q` grows linearly with the distance `c`
  travels.
- **Sampling the decoder alone reproduces the ray** (`scripts/training/cvae_gain_probe.py`, 20000
  draws, float64): with the `c` position box scaled x10 / x100, `|q|_inf` grows ~2e3 / ~2e4 and its
  top direction matches the solver's ray to two decimals on all three charts (Panda K = 7 and 1,
  iiwa K = 1). Scaling the `z` ball instead gives no single direction.
- **Inside the allowed region the learned-variance decoders are already steep**: `|q|_inf` p99.9
  440-1086 rad, max 577-1316 (Panda K = 1: 0.25% of draws > 1000) — against ~2.4-2.8 at the median.
  The fixed-variance decoders stay at max 12-31 rad inside and grow ~10x slower outside; their rows
  show no runaways.

So both families have a single runaway ray per chart, but the flow's sits inside the domain (a
bounded map near its gain ceiling) and the cVAE's is the unbounded decoder's linear asymptote, reached
by leaving the `c` box. Cell analysis: `scripts/cvae_runaway_cells.py`.

## The cVAE on the soft PCS arm, native start: cheaper solutions, per cell (2026-10-20)

The one place the cVAE does not lose: success ties the flow on every row. Per cell
(`scripts/soft_cvae_cost_cells.py`; cost = 0.5 w |strain|^2 here, so cheaper = a less-strained rod;
median per-cell ratio cVAE / flow, 95% bootstrap CI over targets):

| | IPOPT grasp | IPOPT pose | AL-SVGD grasp | AL-SVGD pose |
| --- | --- | --- | --- | --- |
| learned variance (K 12 / 6) | 1.59 / 1.53 (costlier) | 0.81 / 0.82 | 0.66 / 0.70 | 0.56 / 0.61 |
| fixed variance (K 12 / 6) | 0.94 / 0.96 (CI spans 1: tie) | 0.53 / 0.52 | 0.56 / 0.53 | 0.50 / 0.51 |

**The fixed-variance decoder returns one configuration per target whatever the start**: the spread of
its solutions across a target's 8 guesses is 0.003 under IPOPT pose (flow 0.97; joint space 0.000,
it converges to the same point from all 8), and its IPOPT pose cost is 1.07x joint space's. The latent
is collapsed (checked: the active-unit screen gives `soft12__cvaef{12,6}` a total KL of 0.032 nats,
one dimension barely over the 0.01 threshold; `results/screen_cvae/`), so the chart is
effectively one configuration per pose and the correction and `c` do the rest: the solver lands next to joint space's minimum-strain solution, where the flow's
self-motion coverage lets it settle on a start-dependent, more strained one. Under AL-SVGD every cVAE
column is also cheaper than joint space (0.54-0.93). Read as: on this arm the formulation without a
real self-motion chart finds the cheap solution; not as a better chart.

### Why collapse costs the soft PCS arm least (2026-10-21)

Active-unit screen, every exported chart (total KL in nats; median FK error at prior draws):

| chart | KL learned var (K / k) | KL fixed var (K / k) | FK error learned / fixed |
| --- | --- | --- | --- |
| Panda | 4.82 / 3.29 | 0.00 / 0.00 | 624 / 600 mm |
| iiwa | diverged / 3.04 | 0.42 / 0.43 | 253 / 559-567 mm |
| screw | 0.00 / 0.00 | 0.00 / 0.00 | 745 / 745 mm |
| soft PCS | 2.78 / 2.88 | 0.03 / 0.03 | 21-22 / 44 mm |

Every chart is collapsed or nearly so; the learned-variance ones carry 1-2 dimensions and put the
rest of the self-motion into the decoder variance (sigma ~0.55, i.e. about the whole normalized
range, on Panda joints 1-3, iiwa 5-7, every screw joint, and every third soft PCS strain). A
collapsed decoder returns the conditional mean E[q | pose] (exactly so under MSE). The damage
therefore depends on how far that mean sits from the IK fibre. Rigid arms: the fibre is a union of
disconnected branches, each wrapping through large joint ranges, so the mean is no solution (0.25-0.75
m). Soft PCS arm: a single connected fibre in strain space on which the tip pose is close to linear
in strain, so the mean stays within ~4 cm, which the c box (+-0.25 m) and the correction absorb.
The same mean is shrunk toward zero strain (|E q| <= E|q|), and cost here is strain energy, which is
the reading of the cheaper solutions above. The fibre-geometry part is an inference from the FK
errors, not measured on the fibres themselves.

## The LOInK chart (branch `chart-loink`, started 2026-10-22)

Not an ablation: the question is what happens when the chart is a network **designed to have a
well-structured latent space**, and (Thomas's nuance) whether a latent structured for ONE cost
helps under a DIFFERENT objective. LOInK (Somerfield, Abood, Wang, Manchester, arXiv 2609.21275;
`docs/papers/`) learns a bi-Lipschitz `G: R^n -> R^n` (BiLipNet, Wang-Dvijotham-Manchester ICML
2024) with `G(q) = [y_hat; z]`, `y_hat in R^6` the task and `z in R^(n-6)` the latent, trained with
`MSE(y_hat, y) + lambda MSE(J_hat(z, y_hat), J)`, `J_hat = |z|^2_M + b(y_hat)`, `M = I/2 + P^T P`,
so `z = 0` is the cost minimiser on each task fibre. The chart is the INVERSE: `decode(c, z) =
G^{-1}([c; z])` by Davis-Yin splitting per monotone layer (fixed `n_dys`, the PLNet reference
algorithm) plus a Newton polish; the derivative is the implicit-function Jacobian `J_G(q)^{-1}`,
exact, with nothing differentiated through the iterations. `mu = 0.1` fixed, `nu = exp(lognu)`
learned per layer under a `1e-3 log(tau)` penalty. Code `src/charts/loink.py`; the port matches
the official JAX code (github.com/acfr/PLNet) to ~1e-15 (`tests/test_loink_reference.py`).

**Three properties, pre-registered.** (1) Certified gain `|dq/d[c,z]| <= scale/mu`: the flow's
runaway is impossible by construction. (2) Exact inverse: `decode(invert(q)) == q` to 1e-15 for
every `q`, so the paired start is exact (`start_q_error` 0; the cVAE was 0/480). (3) The latent
origin is a cost minimiser, so the trust region and the N(0, I) native draw have cost semantics
(the trust region bounds `J_hat` above the per-task minimum; its binding fraction is reported).
**The stated risk**: `G` is one diffeomorphism, so each task's fibre is one connected line; the
rigid arms' IK fibres are disconnected branches, so `G_1 ~ FK` can only be exact on one branch per
region -- a task-error floor the paper itself names as a limitation. The soft PCS arm (one
connected fibre) is the favourable case.

**Adaptations, stated.** The task is the program's own `c = [xyz, rpy]` (Drake `RollPitchYaw`),
handed to the chart RAW (`task_input = "xyzrpy"`; no quaternion round trip), so `decode` is
continuous across the rpy cut and the paired start is exact off the principal domain too; the
training targets lie on the principal domain, so near the cut the chart extrapolates (screen (e)).
There is no continuous 6-parameter chart of SE(3); this floor is intrinsic to a Lipschitz task
output. `K = n - 6` is forced (Panda/iiwa/screw 1, soft12 6); trust region by the rule
`round(sqrt(K) + 1.5, 2)`; latent box +-5. `J` is standardised by its dataset mean/std (sidecar
`j_mean`, `j_std`) so `z = O(1)`. The cost head's `P` is `K x K` (the paper's `p x p` cannot
multiply `z`).

**The ladder (Thomas, 2026-10-22: "try both" on every axis)**: cost `m` (the program's joint
centering about `q_nominal`; soft PCS strain energy) / `x` (LOInK's own eq. 18 log-barrier,
`eps = 1e-3`) / `u` (lambda = 0, no head: bi-Lipschitz structure alone); activation `r` (relu, as
published; the chart is piecewise linear) / `t` (tanh); recipe `P` (the flow's and cVAE's: AdamW
1.5e-4, StepLR, 620k x 4096) / `A` (the paper's: Adam 1e-3, reduce-on-plateau patience 500,
batch 500, 3 epochs = 150k steps, plus gradient clipping at 1.0 which the replication needed).
12 charts per robot, one architecture (2 monotone layers x 4 x 256, head 2 x 128, the paper's
manipulator example). Labels `loink<cost><act><recipe>`. Native start `z ~ N(0, I)` as every
chart; the `z = 0` "LOInK alone" solution is a screen quantity.

**Replication first (`docs/loink-replication.md`; no code is online).** Sec. V (planar 3-DoF)
reproduces: `G^{-1}([y; 0])` within 2.7-3.7 mm of the three targets, elbow-up, in limits, the
true cost minimised at `z = 0`; eq. 17 holds at every `mu`, larger `mu` smoother -- once `lambda
<< 1` (`lambda = 1` underfits the task, 0.207 m v 0.021 m validation RMSE; `lambda = 0.01`
adopted). Sec. VII (planar HSA in SoRoMoX) does NOT reproduce Table II: with `u_bar = 0` at a
corner of the actuation box the fibre's cheapest point lies on the boundary of the data, so the
surrogate's `z = 0` extrapolates outside it (EE error 159 mm against the paper's 0.44 mm; the
fibre-local fit is 2.3 mm; a symmetric box and a centred cost do not recover it; their `U` is not
printed). Carried into the campaign as an expectation: wherever a task fibre's cheapest point sits
on a joint limit, cost `m`'s `z = 0` extrapolates; the barrier cost `x` cannot. tanh was worse than
relu on every replication number. Inversion on TRAINED models: `tau ~ 100`, so DYS alone at 100
iterations leaves 1e-2; the Newton polish takes it to 1e-15 (float32 2e-6) -- the exporter writes
screen (b)'s recommended `n_dys` into the sidecar, and compile time scales with it (relu 100: ~100 s
on CPU; tanh 100: ~570 s).

**Screens** (`scripts/training/screen_loink.py`, `results/screen_loink/`): (a) task error of
`G_1`; (b) inversion residual v `n_dys`, polished and not, both dtypes; (c) the `z = 0` solution's
FK error and cost against the dataset's cheapest neighbour; (d) the gain bound and the
bi-Lipschitz inequality; (e) task error by distance to the rpy cut; (f) representability and
`start_c_error` (the paired start's `c` is the chart's own `y_hat`, not the FK pose).
Stages LOINKSMOKE -> LOINKPROBE (60-cell native AND paired probes on the rigid arms, soft12 in
full) -> LOINKEXT (full rows where a probe had a success); ops `cluster/LOINK_RUNBOOK.md`; tables
`docs/loink-tables.md` (`scripts/report_loink.py --write`). Results: below.

### Training outcome (2026-10-23): the rigid-arm charts never fit the task; the mechanism is K = 1

**What the cluster returned.** Every project-recipe (`P`) chart that finished (Panda m/x/u x
relu/tanh, iiwa m x relu/tanh; 540k-620k steps) sits at the mean-predictor level on validation:
position RMSE 640-790 mm (the iiwa's constant predictor is 698 mm), rotation 97-121 deg, with
the distortion `tau` grown monotonically to 2e4-1.4e5 and the inverse no longer recoverable
(round-trip residual 1e1-4e3 at `n_dys` 100 + 2 Newton; exit code 6). Train task MSE went 6.9 ->
4.5 over 620k steps and the held-out error of the finished iiwa `m` chart is WORSE than the mean
(0.54/0.42/0.38 m per axis against stds 0.41/0.42/0.37; rpy 1.38/0.62/1.57 rad against stds
1.73/0.69/1.81). The paper-recipe (`A`) runs died another way: the pre-registered plateau schedule
(patience 500 steps on the EMA train loss) halved the learning rate every 500 steps from the start,
reaching ~1e-8 by step 40k on every robot (Panda 626-696 mm; soft12 200-290 mm / 42-87 deg); the
same schedule worked in the planar replication only because that loss fell within 1000 steps.
Those runs were cancelled at 60-100k steps (`last.ckpt` kept; `--ckpt_path=auto` resumes). The
rigid-arm `P` jobs still pending (iiwa x/u, screw m/x/u) were cancelled unstarted. The soft12 `P`
pairs (K = 6) run.

**Diagnosis (local CPU, `scripts/loink_replication/rigid_arm_k_ladder.py`, iiwa14's local
dataset: 100k train, 5000 held-out; the same `LoinkModel`/`loink_loss`; Adam 1e-3, batch 500,
clip 1.0, cost `u`, 3000 steps; held-out RMSE per task column, position m / rpy rad wrapped):**

| chart input -> task | K | x / y / z | roll / pitch / yaw |
| --- | --- | --- | --- |
| q -> [xyz, rpy] (the campaign's) | 1 | 0.29 / 0.27 / 0.27 | 1.65 / 0.70 / 1.79 |
| [q, 3 dummies] -> [xyz, rpy] | 4 | 0.31 / 0.31 / 0.32 | 1.23 / 0.51 / 1.33 |
| [q, 6 dummies] -> [xyz, rpy] | 7 | 0.18 / 0.19 / 0.15 | 1.02 / 0.34 / 1.14 |
| q -> rpy only | 4 | - | 1.29 / 0.58 / 1.41 |
| q -> xyz only | 4 | 0.10 / 0.10 / 0.08 | - |
| plain MLP 3 x 256 -> [xyz, rpy] (1500 steps, train subset) | - | 0.12 / 0.13 / 0.10 | 0.89 / 0.32 / 1.01 |

Ruled out first: the dataset is aligned (jrl FK v stored poses 1e-7); a wrapped angular residual
in the loss and training only on samples away from the rpy cut (57% of the data) change nothing
(rotation stays at 115-120 deg), so the cut is not the mechanism; `scale`, `mu` (0.01),
`lognu_init` (log 20), tanh, a smaller net and depth 1 change nothing either.

**Mechanism.** LOInK asks the IK map to be the first `p` coordinates of ONE bi-Lipschitz chart of
`R^n`, i.e. every task fibre a straight line `{y} x R^K` of a global product. With `n = 7`, `p =
6` that leaves `K = 1`: the 7-DoF arms' fibres are one-dimensional sets of closed curves and
arcs (several per pose) that no line carries, and the inverse-Lipschitz bound `mu` makes the
conflict global rather than local -- the net cannot fit position past ~0.27 m and orientation at
all, and training drives `tau` up without bound. Giving the latent room (dummy inputs) restores
learning, so the failure is the latent width LOInK prescribes, not the optimiser or the port
(which matches PLNet to 1e-15 and reproduces the paper's planar example, `K = 2`). The paper's
examples all have `K >= 2` (planar 2, HSA 3, Magneto 12). **A verdict on the rigid arms, not a
bug**: at its own `K = n - p` LOInK cannot chart the 6-D pose task of a 7-DoF arm. Whether a
padded-latent LOInK (`K` = the flow's width, the cVAE's "K = flow width" rung; `decode` drops the
dummy coordinates) is a rung worth fielding is a modification of the method and Thomas's call;
so is a usable paper recipe (their schedule is one of the questions for the authors).

### The soft PCS charts (2026-10-23): K = 6 learns, but no chart fits, and the latent is not where the box is

The six project-recipe charts (`m/x/u` x `relu/tanh`, 620k steps, jobs 6012963-65) do learn,
unlike the rigid arms, and plateau far from usable by 340k steps. Validation at 620k (constant
predictor 504 mm) and the screens (`results/screen_loink/soft12__loink*__step620000.json`, test
split, float64):

| chart | val pos / rot RMSE | tau | task error median pos / ang (p95) | z = 0 solution: FK error, cost ratio v cheapest neighbour | J_hat R^2 | inverse (n_dys, n_newton, damping) |
| --- | --- | --- | --- | --- | --- | --- |
| mrP | 291 mm / 41 deg | 2.1e4 | 241 mm / 17 deg (500 / 106) | 1236 mm, 96x | -5.9 | 25, 10, none |
| mtP | 332 / 58 | 1.1e4 | 294 / 28 (543 / 143) | 967 mm, 41x | -9.8 | 25, 5, backtrack |
| xrP | 293 / 41 | 2.0e4 | 243 / 16 (503 / 106) | 1097 mm, 86x | -5.6 | 25, 10, none |
| xtP | 331 / 56 | 1.2e4 | 291 / 27 (548 / 138) | 1015 mm, 29x | -9.6 | 50, 5, backtrack |
| urP | 205 / 29 | 4.6e4 | 177 / 11 (343 / 53) | 679 mm, 8x | - | 25, 50, none |
| utP | 224 / 45 | 4.5e4 | 188 / 20 (386 / 117) | 513 mm, 49x | - | 25, 10, backtrack |

98-100% of test poses have more than 50 mm of task error; the unstructured charts (`u`, no cost
head) fit best, so the cost head costs task accuracy here too. The z = 0 point, LOInK's own
solution, is 0.5-1.2 m off the target with 8-96x the cheapest neighbouring sample's cost, and the
cost head's `J_hat` has negative R^2 against the true cost: the latent structure the method is
built for did not form. tau grew throughout training to 1e4-4.6e4 (nu = 1e3-4.6e3 at mu = 0.1),
i.e. the net spent its inverse-Lipschitz room as distortion, not as fit. Gates (d) (gain
0.16-0.92 of the certified bound `scale/mu`, bi-Lipschitz inequality violated on 0 of 10k pairs)
and (b) (decode residual at the sidecar's setting: float64 p99 1e-13 to 8e-13; float32 plateau
3e-5 to 3e-4) pass on all six; urP keeps one unconverged decode in ~5000 (residual ~5) that no
fixed budget removes.

**Inverting the charts: the polish, not the splitting, does the work** (`src/charts/loink.py`,
`scripts/training/export_loink.py`, commit 18a2bd2). Davis-Yin alone leaves a residual of 5-20
on the relu charts at any count (the per-layer step `alpha = mu/gamma` shrinks as nu grows), and
the old exporter ladder (n_dys up to 3200, n_newton = 2) passed only mtP, at 3200. Measured on the
440k/480k checkpoints: the undamped Newton polish cycles on rare inputs of the smooth tanh charts
(a pair passing a 256-sample round trip fails on 2048 at residual ~60), and a greedy backtracking
(per-sample best of steps 1 .. 1/64 on the residual norm, static and graph-safe) stalls at the
kinks of the piecewise-linear relu charts (every pair up to 50 steps fails at ~1e2) while the
full step crosses them. So the step rule is a per-chart sidecar field, `newton_damping`
(`none` | `backtrack`), and the exporter walks `n_dys x n_newton x damping` in order of decode
cost (`n_dys + (ndof + 1 + candidates) * n_newton` network passes), gating each triple on the
first 256 of 2048 fixed samples and then on all 2048. Result: every chart inverts at 25-50 DYS
iterations (tanh with 5-10 damped steps, relu with 10-50 undamped ones). The attached IFT step
and the derivative are unchanged by the damping.

**Gate (f), representability, fails on the four cost-head charts (48-51% relu, 14-22% tanh) and
is an artefact of where the latent lives, not of the conversion.** The true latent `z = G_2(q)` of
EVERY real configuration lies outside the benchmark's +-5 box: |z|_inf median 13-16 (max 25) on
the `m`/`x` charts and 100-1100 on the `u` charts (2000 uniform q per chart, scratch diagnosis).
Out there the sidecar's polish budget, chosen on in-box draws, is unfinished (DYS residual 5-20,
polish not converged; 2x the Newton steps repairs 96-100% of the failures, 2x the DYS count
17-68%); y_hat off the principal rpy domain is 0.2-3% of samples and no more frequent among the
failures. Inside the box (z uniform in [-5, 5]^6, y either the chart's own y_hat or the true FK
pose) the decode converges on all six charts: float64 median 1e-13, max 1e-12 to 4e-5 with at
most 3 of 2000 above 1e-6 (urP's one outlier at ~1); float32 1e-5 to 1e-4. **What this means for
the cells**: every solve stays in the box, so the chart is exact where the solver visits; the
paired start inverts `q_init` to a latent 2-5x outside the box and is clipped on every cell
(recorded `clip_distance`, `start_q_error`), so the paired protocol starts from a projected
point as the cVAE's did, for a different reason (the cVAE's latent was collapsed; LOInK's is
real but on a scale the +-5 box and the `sqrt(K) + 1.5` trust region were never set for). The
`N(0, I)` native draw and `z = 0` sit in a region the training data's latent never occupies.
Standardising `J` to z = O(1) assumed the cost head would fit; it did not.

**The decode budget was re-chosen on three latent draws (2026-10-24).** The smokes on the budgets
above scored 3 of 12 relu paired cells `unrepresentable_start` (start_q_error 0.76-1.67): the
paired start sets the variables to `G(q_init)` exactly, and the decode at the sidecar's budget did
not come back to `q_init`, because the ladder had gated only on latents from N(0, I). The gate now
covers the prior draw, the +-5 box (uniform; N(0, I) never reached its corners, where 0.05-0.15% of
samples failed on two charts) and the real latent `z = G_2(q)` (commit d417150;
`export_loink.py --reladder` re-chooses an exported chart's budget in place, run per chart by
`cluster/reladder_loink_job.sh`). Result (2048 samples x 3 draws, float64): mtP and xtP
(25, 10, backtrack), utP (200, 10, backtrack) and xrP (200, 50, none) are exact on all three; for
the relu charts mrP and urP NO budget up to (400, 50) is exact on real latents (undamped Newton
cycles on a few samples, backtracking stalls at the kinks and fails the box too), so the ladder's
stated best-effort rule takes the cheapest budget exact on prior and box with the fewest real-draw
failures: mrP (400, 50, none) at 0.05%, urP (25, 50, none) at 0.10%, written to the sidecar as
`real_latent_fail_fraction`. The price is per-iteration cost: the cost model's passes per decode
go from 155 to 1050 on mrP, 155 to 850 on xrP, 90 to 155 on mtP/xtP, 155 to 400 on utP (urP
unchanged); the cells already ran to the clock, so expect fewer iterations per cell on the relu
charts. The smokes and PROBE were rerun on the new sidecars (runbook).

Benchmarked anyway, soft12 only (the four `_soft12P` manifests, PROCS=2): the question the
rung asks is answered by cells, and the cVAE precedent ran its collapsed charts. Cells
`docs/loink-tables.md`.

### Phase 2 (2026-10-25): single-branch charts on the rigid arms -- the fit does not move

**Why.** The K = 1 diagnosis above says a bijection of `R^7` cannot carry the several disconnected
arcs of a 7-DoF arm's pose fibre on one line. Thomas's call (2026-10-24): train the Panda and
iiwa charts on ONE IK branch, where the fibre is a single arc parameterised by the redundancy,
and accept the single-branch nature as a stated limitation of the arm. Branch = the nominal
configuration's (`src/branches.py`, analytic classifiers, never hard-coded): Panda `gc(q, 3) =
(1, 2, +1)`, iiwa `GC = (+1, -1, +1)`. Datasets `panda_branch` / `iiwa14_branch` (25M + 15k,
seed 0, rejection sampling on xeon-p8; on-branch acceptance 0.236 / 0.125). Labels
`loinkB<cost><act><recipe>`, 12 per robot, the same `K = 1`, trust region 2.5, 180 s. Reach
ceiling of a single-branch arm on the 20 probe targets (`scripts/loink_branch_ceiling.py`,
analytic IK on the nominal branch over a psi grid, verified by the program): Panda pose 16/20
(the `A = -1` branches reach none), Panda grasp 20/20, iiwa pose and grasp 20/20.

**Result: the branch charts sit at the constant predictor, like the full-fibre ones.** Validation
on the BRANCH test split at the end of training (constant predictor Panda 581 mm / 136 deg, iiwa
700 mm / 130 deg); exports via the 96-triple ladder (`n_dys x n_newton x damping`, three latent
draws: prior, +-5 box, real on-branch `q`); screen (a) = task error of `G^{-1}([y; 0])` on the
branch test split; (g) = the same on the FULL test split's off-branch rows (what a paired start
or an off-branch target meets), with the fraction of their true latents inside the +-5 box:

| chart | val pos / rot RMSE | tau | export (n_dys, n_newton, damping), cost | (a) median pos / ang | (g) off-branch pos, in box |
| --- | --- | --- | --- | --- | --- |
| Panda BmrP | 743 mm / 83 deg | 1.3e5 | rc 6, no triple passes | - | - |
| Panda BmtP | 683 / 114 | 3.9e4 | (100, 10, backtrack) 250 | 615 mm / 105 deg | 886 mm, 0.72 |
| Panda BxrP | 743 / 83 | 1.2e5 | rc 6 | - | - |
| Panda BxtP | 692 / 114 | 4.1e4 | (25, 20, backtrack) 325 | 625 / 105 | 905, 0.68 |
| Panda BurP | 769 / 79 | 1.5e5 | rc 6 | - | - |
| Panda ButP | 791 / 109 | 1.4e5 | (800, 50, backtrack) 1550 | 713 / 98 | 981, 0.02 (z median 86) |
| Panda BmrA | 678 / 93 | 5.1e3 | (1600, 20, none) 1760 | 585 / 74 | 1257 (p95 23 m), 0.81 |
| Panda BmtA | 648 / 119 | 1.0e3 | (25, 10, backtrack) 175 | 579 / 113 | 792, 0.74 |
| Panda BxrA | 680 / 93 | 6.0e3 | (1600, 30, none) 1840 | 577 / 73 | 1193 (p95 21 m), 0.80 |
| Panda BxtA | 629 / 120 | 7.0e2 | (25, 10, backtrack) 175 | 573 / 115 | 765, 0.70 |
| Panda BurA | 702 / 91 | 6.3e3 | (1600, 30, none) 1840 | 602 / 71 | 1186 (p95 14 m), 0.25 |
| Panda ButA | 679 / 117 | 1.1e3 | (100, 10, backtrack) 250 | 629 / 110 | 857, 0.32 |
| iiwa BmrP | 662 / 77 | 7.6e4 | rc 6 | - | - |
| iiwa BmtP | 664 / 104 | 2.9e4 | (25, 20, backtrack) 325 | 545 / 89 | 1183, 0.72 |
| iiwa BxrP | 666 / 77 | 7.9e4 | rc 6 | - | - |
| iiwa BxtP | 678 / 103 | 2.8e4 | (25, 10, backtrack) 175 | 551 / 89 | 1195, 0.71 |
| iiwa BurP | 618 / 74 | 1.1e5 | rc 6 | - | - |
| iiwa ButP | 725 / 99 | 1.1e5 | (800, 50, backtrack) 1550 | 593 / 83 | 1448, 0.03 (z median 69) |
| iiwa BmrA | 635 / 85 | 3.5e3 | (1600, 5, none) 1640 | 511 / 62 | 1966, 0.75 |
| iiwa BmtA | 622 / 113 | 9.6e2 | (25, 10, backtrack) 175 | 526 / 103 | 1051, 0.70 |
| iiwa BxrA | 638 / 85 at 40k | 3.5e3 | killed (LR rule), `last.ckpt` | - | - |
| iiwa BxtA | 470 / 118 at 30k | 8.5e2 | killed, `last.ckpt` | - | - |
| iiwa BurA | 589 / 82 at 20k | 4.1e3 | killed, `last.ckpt` | - | - |
| iiwa ButA | 656 / 108 at 20k | 1.1e3 | killed, `last.ckpt` | - | - |

Every exported chart puts more than 50 mm of task error on 99.9% of its own branch's test poses;
the best of the ladder (iiwa BmrA) is 511 mm at the median with a p95 above a metre. Recipe P
reproduces phase 1 step for step: the Panda `m` relu curve is flat from step 20k (721 -> 743 mm),
`tau` grows to 3e4-1.5e5, and every relu chart ends NON-INVERTIBLE (no triple of the ladder
passes the round trip on any draw; the tanh charts pass at 175-325 passes per decode, ButP at
1550). Recipe A collapses as in phase 1 (the plateau schedule halves the LR every 500 steps;
`lr < 1e-6` at step 12-20k on every run): the Panda pairs and the iiwa `m` pair started before
the 15-min watcher cadence and ran out (frozen at 1.5e-8, so the final chart is the step-20k
chart); the iiwa `x` and `u` pairs were `scancel`led under Thomas's rule at steps 30k / 20k with
`checkpoints/last.ckpt` kept (exportable by hand with `export_loink.py`). Two things the A runs
show: their relu charts DO invert, because `tau` stayed at 3e3-6e3, at 5-10x the tanh decode cost
(1640-1840 passes: 1600 Davis-Yin iterations, the polish cannot be damped at the kinks); and on
the iiwa they sit BELOW the constant predictor (470-656 mm v 700) where the P charts and every
Panda chart sit at or above it -- a fit, but half a metre from usable. Off the branch (g), where
a paired start or a target reachable only elsewhere lands, the error roughly doubles (0.8-2 m
medians, the relu A charts' p95 at 14-23 m) and 20-30% of the true latents leave the +-5 box
(ButP and BurA: 70-98%), so the paired protocol would clip as it does on soft12.

**Conclusion.** Restricting the fibre to one arc did not move the fit; the disconnected topology
was not the binding constraint. The gap is between what a `K = 1` bi-Lipschitz chart of `R^7` can
represent for a 6-D pose task and what training finds (the diagnosis table above: padding the
latent is what restores learning), so the open questions for the authors stand: the training
recipe for a 7-DoF arm, why `tau` runs away, and whether a padded latent is in the method's
spirit. **Cells on the branch charts** (`LOINKSMOKE_branch`, `LOINKPROBE_branch`, 192 + 192 GPU
items) are NOT run by default on charts that put the end effector half a metre off: Thomas's call
(the report prints `not yet measured` on those rows until it is made). Sidecars of the 14 exported
charts (and the six rc-6 ones, carrying the training default and the reason) are in
`models/panda/`, `models/iiwa14/`.

### Closed: dismissed on per-iteration cost (Thomas, 2026-10-26)

LOInK is dismissed as a chart for this framework, and the campaign branch is archived as tag
`archive/chart-loink` (never merged; the code under `src/charts/loink.py`, `src/branches.py`,
`scripts/training/*loink*`, the 96-triple exporter ladder, the tables `docs/loink-tables.md`, the
runbook `cluster/LOINK_RUNBOOK.md` and the manifests are all there). The deciding number is not
the fit but the cost of a call: the chart is an inverse, so one decode is `n_dys` Davis-Yin
iterations plus Newton polish steps, 155-1840 network passes at the budgets the exporter had to
choose, against ONE forward pass (plus one `jacrev`) for the flow, whose trial point is ~3 ms on a
V100 with CUDA graphs (`docs/notes-architecture.md`). In the soft12 probe cells that was 20-30
IPOPT iterations in 180 s on the relu charts and 60-150 on tanh, 1-9 s per iteration, ~100x the
flow's 15-26 ms; even a chart that fitted would not fit inside the clock. The branch-chart cells
were never run. The questions that remain are the authors' (training recipe for a 7-DoF arm, why
`tau` runs away under the project recipe, whether a padded latent is in the method's spirit);
Thomas takes them to the authors himself. **Dismissed because inverting the network is too slow
for our purposes; what was needed is answered and the campaign is fully closed** (Thomas,
2026-10-26). The cluster artefacts were deleted on his instruction (12 GB: `results/train/*loink*`,
`repo/models/*/*loink*`, the 104 `sc_LOINK*` result dirs, `results/screen_loink/`, the four
`state/manifest_stageLOINK*` claim dirs, and the branch datasets `panda_branch` / `iiwa14_branch`),
after every result and screen had been copied to the laptop (`results/soft12/benchmark/sc_LOINK*`,
`results/panda/benchmark/sc_LOINKACC*`, `results/screen_loink/`, gitignored). **To replicate**
(approximate replication is the recipe, not bit-exactness): check out tag `archive/chart-loink`,
rebuild a branch dataset with `cluster/build_dataset_job.sh` (`DATASET_ROBOT=panda_branch`, seed 0,
xeon-p8, ~2 h), train a row of `cluster/chart_runs.txt` with `cluster/submit_chart_train.sh` (P:
~14 h on one V100; A: ~2 h), and run the `manifest_stageLOINK*` stages per `cluster/LOINK_RUNBOOK.md`.
The 20 branch sidecars are committed on the tag under `models/{panda,iiwa14}/`.
