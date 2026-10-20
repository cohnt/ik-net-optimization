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
