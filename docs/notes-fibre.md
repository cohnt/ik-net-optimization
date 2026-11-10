# IK-fibre topology and chart continuity (branch `fibre-topology`, 2026-11-09)

**Status: an investigation for future work (constrained motion planning in the chart's
coordinates). Not part of the record, not merged.** Question (Thomas): rigid arms' IK
solutions split into branches; is that so for the soft PCS arm, and if not, does the flow's
latent have fewer jumps? Code `scripts/fibre/`, job `cluster/fibre_job.sh`.

## What is measured

Everything is in each chart's OWN frame: targets are jrl FK of random configurations drawn
inside the limits (the domain the flow was trained on), and the flow is conditioned on those
poses. No Drake scene, no collision.

- **Connectivity** of the fibre `{u : FK(u) = target}`, **without joint / strain / force
  limits** (Thomas: limits are additional constraints, not kinematic or topological
  obstructions; `--limits` restores the box for comparison). Revolute joints are then
  periodic, so the rigid arms' configuration space is a torus.
  - Solutions: flow draws and uniform starts, Newton-projected onto the fibre.
  - 7-DoF arms (one-dimensional fibre): every component is a closed loop, **traced exactly**.
  - Soft PCS (soft9/12/16: 3, 6, 10-dimensional fibres) and GVS (3-dimensional): pairs joined by a greedy fibre-following
    path or a projected chord; the component count is an **upper bound**.
- **Latent lines**: `z0 -> z1` at a fixed pose. **Pose lines**: `z` fixed, the pose moving
  along a reachable path (FK of a configuration segment) -- the measurement a planner needs.
  A jump is a step > 10x the line's median step and > 0.05 normalized.
- Error Jacobians are exact (torch autograd; JAX implicit differentiation through the GVS
  equilibrium, never through the Newton iterations), checked against central differences at
  start-up (agreement ~1e-10 on all six robots).

Charts: Panda `n6`, iiwa `n4` (the record's rungs), `soft9__n6`, `soft12__n6`, `soft16__n6`,
`gvs_pushrod9_o{1,2}__n6`. 96 targets per robot (48 seeds x 2), 64 lines of each kind per
target, one single-threaded process per seed: SuperCloud `xeon-p8`, except soft9 and the GVS
arm, run on Engaging `mit_preemptable` (`cluster/engaging/`, CPU image) while SuperCloud was down.

## Results (limits off)

p99 columns are the median over targets of each target's p99. GVS settings differ (next section).

| | soft9 | soft12 | soft16 | GVS o1 | GVS o2 | Panda | iiwa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DOF / fibre dimension | 9 / 3 | 12 / 6 | 16 / 10 | 9 / 3 | 9 / 3 | 7 / 1 | 7 / 1 |
| fibre components, median (p10-p90) | 4 (3-6) | 2 (2-3) | 2 (2-3) | 2 (1-4) | 2 (1-4) | 4 (4-8) | **8 (8-8)** |
| components containing an in-limit solution | 1 on 84/96 | **1 on 92/96** | 2 (1 on 25/96) | 1 on 45/48 | 1 on 46/48 | 2 | 8 |
| latent lines with a jump | 6.8% | 4.9% | 29% | **1.2%** | **1.1%** | 60% | 89% |
| latent jumpiness (max/median step), median | 1.3 | 1.6 | 1.7 | 1.3 | 1.3 | 35 | 110 |
| latent-line position error, median / p99 | 1.7 / 4.3 mm | 2.1 / 6.8 mm | 4.8 / 16 mm | 1.6 / 4.0 mm | 1.6 / 4.2 mm | 6.6 / 142 mm | 15 / 256 mm |
| pose lines with a jump | 23% | 13% | 20% | 12.7% | 14.4% | 51% | 58% |
| pose-line position error, median / p99 | 2.0 / 21 mm | 2.8 / 26 mm | 6.1 / 34 mm | 1.7 / 13 mm | 1.8 / 14 mm | 9.5 / 184 mm | 19 / 402 mm |

- **iiwa: exactly its 8 GC classes**, every loop closed (764/764), 5 of 763 components mix two
  GC labels (loops passing within the 0.02 labelling tolerance near a singular
  configuration). The calibration of the method.
- **Panda: 4-8 components**, 2 of which contain in-limit solutions; the rest exist only
  outside the joint limits. Its He-Liu branch labels are defined at fixed `q7` and are not
  fibre invariants (274/494 loops cross labels), so they are no check here.
- **soft12: one component holds every in-limit solution on 92/96 targets.** The extra
  components are made only of strongly coiled configurations (closest point at ~1.9x the
  strain limit, more than pi of bend per segment). Whether they are separate or merely far
  away is not settled: the joins are an upper bound.
- **soft9 (the soft PCS arm at the GVS arm's redundancy) fragments more than soft12**: median 4
  components against 2, all in-limit solutions in one component on 84/96 against 92/96, and more
  latent and pose-line jumps, at identical settings. One pair of rungs, so it suggests rather
  than shows that less redundancy means more pieces. Against the GVS arm at the same fibre
  dimension, the GVS arm jumps less (latent 1.2% v 6.8%, pose 13-14% v 23%); its component
  counts are not comparable (16 + 16 starts against 32 + 32; fewer starts find fewer pieces).
- **soft16: a binary split** that reaches into the limits (two or more in-limit components on 71/96).
  Total torsion does not separate the two (separates 7/87 with the box; spans ~10.7 rad
  inside one component), so it is not 2-pi twist wrap-around. Cause unknown; a method
  artefact is not ruled out.
- **The latent is smooth where the fibre is connected.** soft12's lines barely jump and stay
  near the fibre (p99 7 mm); the rigid arms' jumps are where the flow stretches across the
  gap between components, through points that are not solutions (p99 140-260 mm, 70% of the
  iiwa's branch flips are jumps). The Panda shows the jumps without exposing a runaway, so
  jumps and the gain-ceiling runaway are different phenomena.
- **For planning**: with the pose moving and `z` fixed, soft12 tears on ~1 line in 8 against
  ~1 in 2 on the rigid arms. Better than the rigid arms, not tear-free; and obstacles split
  any robot's free fibre.

## With limits (`--limits`, `fibre_diag.py`)

The box adds components: soft12 one component on 60/96 (greedy), 67/96 (+ chord); the box
alone accounts for most of the rest (85/96 joined once it is removed). The greedy and chord
joins cannot certify a one-dimensional fibre (iiwa 13-15 pieces against 8 classes with the
box removed), which is why the rigid arms are traced.

## GVS push-rod arm

Complete (2026-11-10, Engaging, both rungs, 48 targets each). One equilibrium solve is ~20-40 ms
per single-threaded worker, so connectivity ran without lines (`--skip_lines`, 48 seeds x 1
target, **16 + 16 starts, not 32 + 32**) and the lines separately (`--skip_conn`, 12 seeds x 4
targets); FK and Jacobians go to JAX in chunks of 1024 (16k at once OOM-killed a node). The
connectivity targets and the line targets are different draws. Numbers in the table above.

Read as soft12's pattern: one component holds the in-limit solutions, the extra ones are
large-force configurations (closest point at ~2.7-2.9x the force limit). o1 and o2 do not
separate. The forward map is a cold-start Newton equilibrium; whether it is continuous
everywhere was not checked (Thomas: the GVS arm is not the focus).

## Open

- What separates soft16's two components.
- Whether soft12's coiled components join the main one.
- Pose lines that cross between components (where the fibre's topology changes across the
  workspace) are not distinguished from chart tears.
