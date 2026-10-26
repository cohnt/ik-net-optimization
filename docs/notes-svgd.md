# AL-SVGD: verdicts and the V4 pin

The long form of a part of `CLAUDE.md`, moved here verbatim on 2026-10-18 to bring `CLAUDE.md` under
40k characters. `CLAUDE.md` keeps the verdicts and rulings; this file keeps the mechanisms, numbers and
history behind them. "Above"/"below" and section names refer to the old single-file layout, now split
across `docs/notes-*.md`. A reference elsewhere to "CLAUDE.md's ... section" means the matching notes file.

## The fourth method class: AL-SVGD on the fixed programs (branches `svgd` and `svgd-tune`, 2026-10-08 to 10-18)

**Evidence lives in `docs/svgd-solver.md`** (pre-registration and results of every round R1-R30, the
kernel screens); operations in `cluster/SVGD_RUNBOOK.md`; readers `scripts/report_svgd.py --stage SVGD`
(round reads) and **`scripts/report_svgd_2x2.py --write docs/svgd-2x2-tables.md` (the result against
ablations and baselines, the condensed summary first)**. This file keeps the verdicts and the rulings.

**What it is.** Thomas reopened the closed solver axis for a fourth METHOD CLASS: a batch-parallel
particle method on exactly the existing programs. **The goal is the LEARNED arm under svgd against the
learned arm under IPOPT on the same cells; joint-space svgd is an ablation, not a baseline** (Thomas,
2026-10-10). Rulings: SVGD-style only; exact Drake collision, no proxy; no formulation-specific
information inside the solver (kernel SPACE is the one deliberate exception; `svgd_stop_select_objective`
was approved as a second, about regularisers as a class, and measured inert); AL is the formulation and
SVGD the optimizer, one dynamics; `--solver svgd`; K stays a separate parameter (2026-10-11, do not
re-propose the one-loop strip-down); "resample" is the term; the clearances wheel is a strict
improvement every svgd round uses (the one stated exception to ONE DRAKE); compute footprint (GPU + 40
threads vs one core) is a paper explanation, not a fairness worry; **the augmented Lagrangian with its
true dual update stays the method** (2026-10-16: *"more flexible (and arguably more principled) ... leaves
a lot of future directions open, like ADMM"*) although the pure quadratic penalty ties it on every row.

**THE FIELDED PIN IS `SVGD_PINNED_V4` (2026-10-16; stages R29 on), and every piece of it has a measured
reason** (the tune-and-simplify push R22-R30, branch `svgd-tune`, 2026-10-14 to 10-18; each round
pre-registered mechanism, diagnostic, prediction and decision rule before its manifest):

| setting | value | why it stays (the round) |
| --- | --- | --- |
| method | PHR augmented Lagrangian, Tabor-Hermans direction, exact dual step alpha = rho every K = 10 inner steps | alpha = rho / 10 stalls earlier with fewer feasible particles, alpha = 10 rho destroys the swarm; alpha = 0 ties (R24), AL kept by ruling |
| penalty rho | 1e5 | bought the record's cost (1e3 -> 1e5); 1e6 saturated, the residual set by the stall stop (R24) |
| kernel space | `cz` (product of the c and z RBFs, median bandwidth, floor 0.05) | `kz` costs 9-26% on every grasp row (particles sharing z but differing in c read as duplicates and are repelled off the feasible set); floor 0.05 is a value on soft PCS grasp (R24). Learned-only: **a joint-space svgd run must name `svgd_kernel=q`** |
| metric | Stein variational Newton, block-diagonal Gauss-Newton, Levenberg delta 10 | identity crawls on pose and stalls high on grasp (it cannot absorb the 1e5-1e7 row / objective curvature spread rho creates); **delta fixes the gauge along the formulation's own 6-fold (c, z) redundancy** -- delta 0 is dead, delta 1 resamples 7-100x, 100 is a wall trade (R23) |
| step | lr 0.3 | the 2026-10-10 screen; never moved a cell |
| particles | N = 64, float32 swarm, particle 0 at the guess, the rest from the native draw under both protocols (`svgd_paired_init=native`) | N is the selection pool: N 32 / 16 trade cost for wall (R24); the native draw removes the jittered swarm's paired penalty (R10 / R18) |
| resample | runaways (`\|q\|_inf` > 10) redrawn from the native draw | inert on three robots, earns its place on the screw arm (`drop` loses 2-3 cells per screw row, R22) |
| multiplier box | 1e4 | inert on the learned arm (no clip in 30,720 cells moved a selection) but the joint-space ablation loses 16 grasp cells without it (R22) |
| stop | best feasible objective stalls 3 checks at rel 1e-3, confirmed by the float64 Drake re-check | patience 3 ADOPTED from R24 (same cells, wall 0.66-0.81x; R1-R28 ran 5); **the gate-edge rule** (2026-10-12): a stall stop stands only if the top-k float64 re-check passes, else the feasibility level tightens x0.9 and the swarm continues (`n_recheck_failures`, `feas_level`) -- five float32 cells in 11,520 had sat 0.05-7% outside the gate |
| objective the stop and selection read | `full` (regularisers included) | `shared` costs grasp wall 10-60% for nothing (R29) |

**Stripped with compute**: the temperature (`svgd_temperature`, `svgd_metric_temperature`; inert both ways,
removed c308934, R22). **Refuted or inert, not in the pin** (all on the rigid arms; every one moves soft
PCS pose): the latent prior N(0, I) on z as a target term (cost +16-335%, wall 2-8x, R26); the flow as an
exact density (the log-det alone is the push's only success loss, screw grasp 456 / 458, driving the
swarm into the runaway it was meant to prevent, R27); the correction weight 0 / 1 / 100 against 10 (inert,
the box binds on 1 cell of 23,040, R28); the drive-only prior (inert, R29). **Phase-3 replacements that
did NOT happen**: the prior replacing the trust region (moot once the prior is refuted) and replacing the
Levenberg delta. **One robot-specific result not fielded**: on soft PCS pose the drive-only prior at w = 1
gives 0.50x the pin's cost (0.50 / 0.69x the IPOPT twin) for 6 s against 1.1 s; a flat or absent c factor
cuts 18%. **Ruled 2026-10-18: reported as a finding, NOT fielded** (one pin for all robots).

**THE SVGD RECORD IS R29 `pin4` (learned) + R30 `pin4kq` (joint space), 2026-10-17**, the record's sixteen
rows (four robots x grasp / pose x native / paired, 480 cells, 180 s, seed 1, one solve per GPU, the
clearance row); R30 `pin4s2` replicates it on seed 2 (480 / 480 on all 16 rows, every median step count
equal; **the seed-to-seed band on a row's median cost is 0.94-1.10**, so a ratio inside it is a tie, while
cell-paired reads on one grid reproduce within 0.3%). The V2 record (R20 / R21) is superseded on the same
cells: success identical, cost and wall within the band. `docs/status-quo-tables.md` column `PS` reads
these runs. Pin history: V1 (R1-R19, rho 1e3, kernel q, jitter), V2 (R20-R23: `cz`, rho 1e5, native draw),
V3 (R24-R28: V2 without the removed temperature), V4 (V3 + patience 3); `SVGD_HISTORICAL_DEFAULTS` records
what each stage ran unnamed and every manifest regenerates byte-identically.

**The result, as a double contribution (Thomas, 2026-10-18: the formulation and the optimizer; metrics
success, cost, wall and the time-matched axis -- no iterations, svgd counts outer steps, not majors).**
Proposed = learned + AL-SVGD; ablation A = the learned formulation under IPOPT / SNOPT / NLopt; ablation
B = joint space under AL-SVGD; baselines = joint space under the Drake solvers. Tables:
`docs/svgd-2x2-tables.md`.
- **Success: the proposed method is 480 / 480 on all 16 rows with zero clock or step-cap stops**, 8 / 8 / 0
  against A-IPOPT (395-480), 14 / 2 / 0 against A-SNOPT (233-479), 16 / 0 / 0 against A-NLopt (3-360),
  2 / 14 / 0 against B (463-480; the screw arm's clock stops, 16-17 grasp and 4 pose cells, under the
  24-cell threshold), 16 / 0 / 0 against every baseline (joint space under IPOPT 200-472). Under IPOPT
  the formulation is worth +8 to +280 cells over joint space; under joint space the optimizer is worth
  +8 to +280 over IPOPT; **each alone takes most rows to or near the ceiling, the combination is the only
  corner at the ceiling everywhere, and the screw arm is where only the combination gets there.**
- **Cost** (ratio corner / proposed on cells both solved, shared objective): the proposed method is
  cheaper than its own formulation under IPOPT on all 12 rigid rows (1.22-3.48x) and costlier on soft PCS
  (0.70-0.81 grasp); joint space under IPOPT is cheaper on rigid grasp (0.54-0.85) and every soft PCS row
  (0.27-0.42), the proposed cheaper on rigid pose (1.68-3.46). Inside AL-SVGD joint space is level on grasp
  and cheaper on pose (0.73-0.98). **So the cost premium against joint space on grasp and on the soft PCS
  arm is the learned formulation's, under every optimizer; AL-SVGD removes it only against its own
  formulation under IPOPT, by selecting the best of a 42-58-particle feasible swarm.** Residual at stop
  4e-6 to 3e-5 against IPOPT's 1e-8, both inside the 1e-3 gate.
- **Wall** (cap-clamped mean) 0.70-2.00 s on every row: faster than A-IPOPT on 12 rows (slower on soft PCS
  grasp and pose native), than A-SNOPT and every SNOPT / NLopt corner on all 16, than B on 12 (slower on
  Panda grasp); joint space under IPOPT is faster on 10 rows (every pose row at 0.1-0.5 s).
- **Time-matched** (the corner restarts from the target's recorded starts inside the proposed method's wall
  on that cell, the running solve finishing): A-IPOPT 84.6-100% (7 rows at >= 99%), joint space under
  IPOPT 83.1-99.6% (2), the SNOPT corners 51-94%, NLopt 0-75%; the proposed 100% everywhere.
- **The fair one-line characterisation** (agreed 2026-10-18): on success and on wall at matched time the
  ordering is AL-SVGD >= ablations > baselines under interior point, the proposed method at the ceiling on
  every row; on solution cost the learned formulation pays a premium on grasp and on the soft PCS arm
  under every optimizer. The exceptions are discussed, not hidden. **The contribution is clear and strong
  on success, robustness to the start, wall at matched time and generality across four robots; it is not
  a solution-quality result against joint space and the paper must not claim one.**

**Mechanisms settled along the way**: the kernel-off swarm keeps descending to the clock (R1K); the q-space
kernel spreads particles OFF the manifold (3-4 feasible of 64 on rigid grasp against 42-58 with `cz`),
which is why joint space under AL-SVGD is slower on the screw and soft PCS arms; the paired-start penalty
was the jittered swarm (R10); the resample carries the paired start under V1 (R11) and the screw arm under
V4 (R22); the multipliers at the record's rows are O(0.2-3), so the penalty alone sits inside the gate
(R24); the soft PCS arm takes the split step (row Jacobians by forward mode through the exact strain map,
3.7-4.8x per outer step, merge a2a5569).

**Implementation findings, open**: twelve screw-arm items segfaulted in R22 / R23 (native code, not
deterministic per cell, clean under faulthandler alone on a node, concentrated on runaway-heavy variants;
R24-R30 ran 0 of ~9,000 items crashed) -- a race at PROCS=2 is what is left. The float64 Newton polish of
the returned particle (R25) is being SMOKED on branch `svgd-polish` (2026-10-18; Thomas's worry: a polish must not
break inequality rows); the segfault is left recorded (Thomas, 2026-10-18). The GVS arm under
svgd is DEFERRED (2026-10-12; its JAX forward model runs on the CPU). R24's remaining `alpha10` pose rows
(168 items) are left held on the cluster; `--reclaim` would release them. **The push is CLOSED
(2026-10-18); the next step is network-architecture ablations, in another session.** The staged cluster
tree is f312793 (svgd-tune; patience 3, `svgd_stop_select_objective`), swapped in by job 5943005.

