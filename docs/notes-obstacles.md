# Obstacles: a harder benchmark (campaign notes)

Branch `obstacles`, from 2026-11-25. Thomas asked for the open item "a harder problem
formulation", starting with obstacles: on native starts learned + AL-SVGD is 480/480 on every
row, and joint space + AL-SVGD and learned + IPOPT are within a few cells of it
(`docs/svgd-2x2-tables.md`). Merging is Thomas's call.

**What did not make the problem harder, and the lesson.** Scene clutter (stage HARDMUG) moved
-6 to +5 cells of 480. The shelf-depth inset moved 2-10 of 60, with no trend. Obstacles away from
the target's neighbourhood do not bind; they have to be where the arm has to go.

## The two families (both fixed before any result was seen)

The ladders and every constant below were fixed in the plan Thomas approved on 2026-11-25,
before any solve. **Never re-ladder, never select a rung, and report every rung.**

**All four record robots share one layout**, so one rule gives every robot the same obstacles.
- The base is welded at the world origin.
- The four shelf units are at (0.30, +-0.55) and (-0.45, +-0.55), z 0.4, yaw 235 / 135
  (`SHELF_WELDS`).
- Each unit is open at both faces; the face toward the base is 4-16 deg off the base line.

### OBST: random spheres near each target (`src/obstacles/per_target.py`)

- **Geometry.** Spheres of radius 0.04 m, welded in memory (`obstacle_###`), checked by the same
  `MinimumDistanceLowerBoundConstraint` as all other geometry, for every arm and solver.
- **Placement.** Centres are uniform in a ball of radius 0.30 m about the target's containment
  point: `between_fingers`, which is the mug centre on the grasp task.
- **Screen.** Exact Drake signed distance on a diagram that holds the candidate (and, on the
  grasp task, the mug) as FREE bodies, so the anchored-vs-anchored trap cannot hide an overlap.
  A candidate is rejected if any of these holds:
  - it is closer than 0.03 m to any robot or gripper geometry at the witness q* (the target's
    generating configuration);
  - it is closer than 0.01 m to any shelf, table or the target mug;
  - its centre is closer than 2r + 0.01 to an obstacle already accepted;
  - its centre z is below r.
- **Certificate.** `CertifyWitness` checks q* on the actual solve diagram: the collision
  constraint's value is below 1 and the robot-obstacle clearance is at least 0.03 m. A failure
  raises; nothing is redrawn silently.
- **RNG.** `default_rng([seed, "obst", crc32(robot/task), target])`. Every candidate consumes a
  fixed block of draws, so the sets are pure in (seed, robot, task, target), independent of
  sharding, and leave the grid's own stream untouched.
- **Ladder: {0, 8, 16, 32, 64} obstacles per target.** Rung n is the first n accepted, so the
  rungs are nested. The cap is 1000 candidates per obstacle; achieved counts are recorded,
  never padded.
- **Same grid on every rung.** Targets, guesses and `grid_hash` are identical across rungs.
  Only the fingerprint moves (its payload is the rule plus every centre). Rung 0 is the
  record's scene and grid, bit for bit (`tests/data/record_scene_fingerprints.json`).
- **Guesses** are drawn on the base scene, the precedent the mug set, so a start may sit in
  collision ("we're not assuming feasible initial guesses"). Every cell records
  `start_collision_value`.

### PILLAR: floor-to-ceiling cylinders (`src/obstacles/pillars.py`)

- **Rule.** One cylinder per shelf unit, z from 0 to 2 m. Its axis is on the horizontal
  segment from the base to the centre of the unit's base-facing opening, with its surface
  0.05 m from that centre along the segment.
- **Ladder: radius {0, 0.04, 0.08, 0.12} m.** The rungs are nested.
- **Scenes.** Generated as `models/<robot>/*_pillars_rXXX.yaml`: the hardened YAML verbatim
  plus four pillars (`src/obstacles/generate_pillar_scenes.py`; a test checks they are
  byte-equal to the generator's output).
- **Targets and guesses are redrawn on each pillar scene**, so the grids differ between rungs.
  Sampler acceptance falls with radius; measure it with `scripts/probe_shelf_acceptance.py
  --scene`, against the trip criterion at `MAX_CONSECUTIVE_REJECTIONS`.

## Traps

- **AL-SVGD rebuilds its scene from `SceneSpec`**, in its collision checker, its clearances
  backend and its native redraws. `SceneSpec.obstacles` and the `_scene_spec` fallback carry
  the centres, and a plant/diagram count check raises on any mismatch. Anything that rebuilds a
  solve scene must go through `BuildTargetDiagram` or `SceneSpec.build_diagram`.
- **`diagram.extra_obstacles` is a Python attribute.** A diagram rebuilt any other way loses
  it.
- **The fingerprint payload must be passed by every caller**; nothing forces it.
- **Obstacle sets come from Drake distance queries.** Shards on different CPU models could in
  principle flip a candidate at the margin. The merger's `scene_fingerprint` check refuses
  such a merge, so it cannot pass silently.

## Measurement

- **Arms:** learned and joint space.
- **Solvers:** AL-SVGD (pin V4, clearances backend, learned kernel `cz`, joint-space kernel
  `q`) and IPOPT. SNOPT is optional and Thomas's call.
- **Rows:** the four record robots, grasp and pose, native starts, 480 cells, 180 s clock,
  seed 1, `--compile --set flow_cuda_graph=True`, the record's budgets.
- **Each item is the record's own item** with only the obstacle arguments and tag changed.
- **Compute: Engaging / CSAIL preemptible GPUs, not SuperCloud** (Tommy, 2026-11-25: *"I'm not
  super worried about results not all being comparable, as long as there's A/B within the
  campaign"*).
  - Each family runs on ONE cluster and ONE GPU type, its own rung 0 included, at one solve per
    GPU. On Engaging the CPU model is pinned too.
  - **Rung 0 is therefore re-run on the campaign's hardware.** Its seconds never sit beside the
    record's V100 rows.
  - All rungs share one Drake: release 1.58.0, the pin Tommy approved on 2026-11-25.
  - A preempted job resumes from its per-cell JSONL; a cell killed mid-solve is re-solved from
    scratch.
- **Order:** OBST first and reported, then PILLAR. Every submission is cleared with
  learned-ik-orchestrator first.
- **Reporting:** success, cost on shared cells, wall clock, and time-matched success, per family
  x rung x row. Exact McNemar within a rung; never pair cells across rungs (different
  fingerprints). Cap check (`timed_out` and `hit_iteration_cap`) on every row before any
  verdict.
