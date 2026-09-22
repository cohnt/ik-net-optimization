#!/usr/bin/env python3
"""Emit a work-item manifest for cluster/run_items.sh.

============================ STANDING REMINDER ============================
If even REMOTELY unsure about a SuperCloud action, STOP and ask Thomas.
===========================================================================

A work item is one invocation of a benchmark script: one `(robot, tag, flags)`
tuple, optionally split into `--shard K/N`. The manifest line format is four
`|`-separated fields:

    <id>|<env assignments or ->|<script>|<args>

`<id>` doubles as the claim/done-marker key in `state/<manifest>/`, so it must be
unique; `<env>` and `<args>` are word-split by the runner, so no token in either
may contain whitespace (asserted below).

Items are emitted **longest-estimate-first** (LPT). That is bin-packing *within*
a stage and is not in tension with the campaign's short-to-long stage ordering:
stages are laddered by cost so a defect is cheap, and inside a stage LPT keeps the
tail from being one long item finishing alone.

The estimates are guidance for `--summary` only -- nothing schedules on them.

Usage:
    python cluster/gen_manifest.py --stage A --wall-time 20 -o cluster/manifest_stageA.txt
    python cluster/gen_manifest.py --stage B --summary
    python cluster/gen_manifest.py --selftest
"""
import argparse
import re
import os
import sys

SCRIPTS = {"panda": "scripts/panda/panda_benchmark.py",
           "iiwa": "scripts/iiwa/iiwa_benchmark.py"}

# Arms per robot. The iiwa has no analytic arm: no Iiwa14IKProgramAnalytic exists, and
# writing one is future work or possibly not done at all (Thomas, 2026-09-19).
ALL_ARMS = {"panda": "learned,numerical,analytic,analytic8", "iiwa": "learned,numerical"}

# Seconds per (cell x arm), used only for the LPT ordering and the --summary
# estimate. Deliberately pessimistic: the learned arm is the one that can sit at
# the cap, so a cell costs about the cap when it is binding and much less when not.
SEC_PER_CELL_ARM = 0.55


def item(robot, tag, flags, targets, guesses, arms, wall_time, shards=1, env="-", seed=0):
    """One logical run, expanded into `shards` manifest items."""
    n_arms = len(arms.split(","))
    est = targets * guesses * n_arms * wall_time * SEC_PER_CELL_ARM / shards
    base = ["--targets", str(targets), "--guesses", str(guesses),
            "--wall-time", str(wall_time), "--arms", arms,
            "--seed", str(seed), "--compile", "--tag", tag] + flags
    out = []
    for k in range(shards):
        args = base + (["--shard", f"{k}/{shards}"] if shards > 1 else [])
        ident = tag + (f"_shard{k}of{shards}" if shards > 1 else "")
        out.append(dict(id=ident, env=env, script=SCRIPTS[robot], args=args,
                        seconds=est, robot=robot))
    return out


def stage_A(wall, targets, guesses, shards):
    """Instrumented re-runs of the grasp finals.

    The shortest useful work, and the only way to get per-cell `q`, `min_distance`
    and `min_distance_pair` into the record -- every archived summary predates the
    commit that persists them, which is what blocked answering the iiwa question
    from the archive without re-solving.
    """
    items = []
    for robot in ("panda", "iiwa"):
        for start in ("paired", "native"):
            items += item(robot, f"sc_A_{robot}_mug_{int(wall)}_{start}",
                          ["--task", "mug", "--config", "latent", "--start", start],
                          targets, guesses, ALL_ARMS[robot], wall, shards)
    return items


def stage_B(wall, targets, guesses, shards):
    """One-factor knob sweeps, learned arm only -- CLAUDE.md Next-steps #2/#3/#5/#6.

    All four are reachable through `--set` (the collision-shaping fields were added
    to ProgramOptions for exactly this; they default to the previously hardcoded
    values, so the sweep's centre point is the finals' own configuration).
    """
    knobs = ([("collinf", f"collision_influence_offset={v}") for v in (0.02, 0.05, 0.2, 0.4)]
             + [("collscale", f"collision_row_scale={v}") for v in (0.02, 0.05, 0.2, 0.5)]
             + [("mu", "ipopt_mu_strategy=adaptive")]
             + [("corrcost", f"correction_cost_weight={v}") for v in (1e-3, 1e-2, 1e-1, 1.0)]
             + [("latcost", f"latent_cost_weight={v}") for v in (1e-3, 1e-2, 1e-1)])
    items = []
    for robot in ("panda", "iiwa"):
        for name, override in knobs:
            safe = override.split("=")[1].replace(".", "p").replace("-", "m")
            items += item(robot, f"sc_B_{robot}_{name}_{safe}",
                          ["--task", "mug", "--config", "latent", "--start", "paired",
                           "--set", override],
                          targets, guesses, "learned", wall, shards)
    return items


## The correction penalty Thomas approved on 2026-09-02. B2/B3 chart the curve; this is
## the value the headline table fields.
D_CORRECTION_COST = 10.0

def stage_C(caps, targets, guesses, shards):
    """Success against the wall-clock cap, as a curve rather than two points (#7).

    Run against the **approved** learned formulation, i.e. with the correction penalty
    (`D_CORRECTION_COST`). An earlier version of this stage was generated before Thomas
    approved the penalty and was killed unrun: a cap curve measured on a formulation
    nobody is fielding answers a question nobody asked, and the penalty changes the
    quantity the curve is about -- it more than halves the iiwa's timeouts, which is
    precisely what a cap curve measures.
    """
    items = []
    for robot in ("panda", "iiwa"):
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                for cap in caps:
                    items += item(robot, f"sc_C_{robot}_{task}_{int(cap)}_{start}",
                                  ["--task", task, "--config", "latent", "--start", start,
                                   "--set", f"correction_cost_weight={D_CORRECTION_COST}"],
                                  targets, guesses, ALL_ARMS[robot], cap, shards)
    return items


## Stage D runs on a DIFFERENT SEED from every sweep that chose this weight. Targets are
## drawn sequentially from the seed, so a 60-target seed-0 grid literally contains the
## 15-target seed-0 sweep grid as a prefix -- reporting the headline table on it would be
## quoting a weight that was selected on a quarter of the very cells being reported.
## Seed 1 makes the choice out-of-sample.
D_SEED = 1


def stage_D(wall, targets, guesses, shards):
    """The big-N replication of the finals -- the statistical power the design
    questions actually need (a 5-cell difference at 60 cells is undetectable).

    Two learned arms, run as separate items on the same grid so they pair cell for cell:
    the approved formulation with the correction penalty, and the same formulation with
    the penalty off. The second is not optional -- once the penalty is adopted, every
    table still has to show what it buys, and this is where that comparison gets its
    power.
    """
    items = []
    for robot in ("panda", "iiwa"):
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                base = ["--task", task, "--config", "latent", "--start", start]
                items += item(robot, f"sc_D_{robot}_{task}_{int(wall)}_{start}",
                              base + ["--set", f"correction_cost_weight={D_CORRECTION_COST}"],
                              targets, guesses, ALL_ARMS[robot], wall, shards, seed=D_SEED)
                items += item(robot, f"sc_D_{robot}_{task}_{int(wall)}_{start}_nopenalty",
                              base, targets, guesses, "learned", wall, shards, seed=D_SEED)
    return items


def stage_D_baselines(wall, targets, guesses, shards):
    """Stage D's baseline columns, re-measured after the correction-cost guard.

    Stage D applied `--set correction_cost_weight=10` to every arm, but `correction` is
    a learned-only decision variable and the formulations share one `ProgramOptions`, so
    each numerical/analytic/analytic8 program raised AttributeError during construction
    and its whole column scored 0. The learned columns are unaffected -- `run_grid`
    builds a fresh program per cell per arm -- so only the baselines are re-run, on the
    same seed, grid, cap and start protocol, which is what lets them pair cell for cell
    against the Stage D learned records.

    The override is kept even though it is now a no-op for these arms: it keeps
    `metadata["overrides"]` identical to the run these will be paired against.
    """
    items = []
    for robot in ("panda", "iiwa"):
        arms = ",".join(a for a in ALL_ARMS[robot].split(",") if a != "learned")
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                items += item(
                    robot, f"sc_D_{robot}_{task}_{int(wall)}_{start}_baselines",
                    ["--task", task, "--config", "latent", "--start", start,
                     "--set", f"correction_cost_weight={D_CORRECTION_COST}"],
                    targets, guesses, arms, wall, shards, seed=D_SEED)
    return items


## Stage E targets the runaway diagnosed in Stage C: the learned arm's joint-limit row is
## evaluated on the flow's output, whose worst-case gain is ~1e13, and 0.065% (Panda) to
## 3.34% (iiwa) of the allowed conditioning region sits in that regime. Neither existing
## region knob avoids it -- the blow-up regions are spread through the domain, not at its
## edges -- so what is left that does NOT change the formulation is how IPOPT scales a
## problem containing such a row. Its default `gradient-based` scaling computes factors at
## the starting point and caps them at `nlp_scaling_max_gradient = 100`, so a row that only
## becomes enormous later is scaled as though it were ordinary.
##
## Learned arm only, on the five rows with a real deficit or a frozen cap-bound set. The
## default point is NOT re-run: the Stage C 45 s learned columns are exactly it, on the
## same grid, seed and cap.
E_SETTINGS = [
    ("scalenone",  ["ipopt_nlp_scaling_method=none"]),
    ("scaleequil", ["ipopt_nlp_scaling_method=equilibration-based"]),
    ("maxgrad1e4", ["ipopt_nlp_scaling_max_gradient=1e4"]),
    ("maxgrad1e8", ["ipopt_nlp_scaling_max_gradient=1e8"]),
]
E_ROWS = [("iiwa", "mug", "paired"), ("iiwa", "mug", "native"),
          ("iiwa", "pose", "paired"), ("panda", "pose", "paired"),
          ("panda", "mug", "paired")]


def stage_E(wall, targets, guesses, shards):
    """Can IPOPT's scaling survive a constraint row that reaches 1e11? (solver options only)"""
    items = []
    for robot, task, start in E_ROWS:
        for name, sets in E_SETTINGS:
            flags = ["--task", task, "--config", "latent", "--start", start,
                     "--set", f"correction_cost_weight={D_CORRECTION_COST}"]
            for kv in sets:
                flags += ["--set", kv]
            items += item(robot, f"sc_E_{robot}_{task}_{int(wall)}_{start}_{name}",
                          flags, targets, guesses, "learned", wall, shards)
    return items


## ------------------------------- Stage F ---------------------------------- ##
## The two `q`-side interventions against the flow's runaway regions. BOTH ARE STATED
## DEVIATIONS from the draft's eq. (6), authorised by Thomas as experiments while he said
## he dislikes both -- so these arms are diagnostics and must never be reported as "the
## learned formulation". The preferred remedy is a better iiwa chart; this stage exists to
## close out the alternatives that do not require one.
##
## The pilot rows are the three where the pathology lives plus one Panda control where the
## exposure is 0.065% and nothing should move. The default point is NOT re-run: Stage C's
## 45 s learned column is the same grid, seed, cap and formulation.
F_ROWS = [("iiwa", "mug", "paired"),     # worst row: 39-44 of 60 cells at the cap
          ("iiwa", "mug", "native"),
          ("iiwa", "pose", "paired"),    # the frozen 19-cell divergent set
          ("panda", "mug", "paired")]    # control: exposure 0.065%, expect no movement

F_SETTINGS = [("liftq",    ["lift_q=True"]),
              ("jlpen1",   ["joint_limit_penalty_weight=1.0"]),
              ("jlpen10",  ["joint_limit_penalty_weight=10.0"]),
              ("jlpen100", ["joint_limit_penalty_weight=100.0"])]

## Stage F2/F3 expand the winning variant, gated on the pilot moving the runaway metric.
## F3 runs on Stage D's seed and grid so Stage D's learned columns are the paired control.
F2_ROWS = [(r, t, s) for r in ("panda", "iiwa")
           for t in ("mug", "pose") for s in ("paired", "native")]


def _f_variant(name):
    """The single Stage F setting named, for the F2/F3 expansions."""
    if name is None:
        raise SystemExit("stages F2/F3 need --f-variant NAME (one of: "
                         + ", ".join(n for n, _ in F_SETTINGS) + ")")
    match = [(n, kv) for n, kv in F_SETTINGS if n == name]
    if not match:
        raise SystemExit(f"unknown --f-variant {name!r}; expected one of "
                         + ", ".join(n for n, _ in F_SETTINGS))
    return match


def stage_F(wall, targets, guesses, shards, rows=None, settings=None, seed=0, tag="F"):
    """Do either q-side intervention stop the runaway? (stated deviations, diagnostics)"""
    items = []
    for robot, task, start in (rows or F_ROWS):
        for name, sets in (settings or F_SETTINGS):
            flags = ["--task", task, "--config", "latent", "--start", start,
                     "--set", f"correction_cost_weight={D_CORRECTION_COST}"]
            for kv in sets:
                flags += ["--set", kv]
            items += item(robot, f"sc_{tag}_{robot}_{task}_{int(wall)}_{start}_{name}",
                          flags, targets, guesses, "learned", wall, shards, seed=seed)
    return items


def stage_B2(wall, targets, guesses, shards):
    """Extension of the `correction_cost_weight` sweep, which had not peaked at 1.0.

    Stage B found the only knob with a strong, monotone, same-signed effect on BOTH
    robots: success rises all the way to the largest weight tested, so the sweep has to
    be pushed further before anything can be said about where the optimum is. Run on
    both start protocols this time, because a regulariser that changes the objective
    could plausibly interact with where the solve begins.
    """
    items = []
    for robot in ("panda", "iiwa"):
        for start in ("paired", "native"):
            for w in (3.0, 10.0):
                tagw = str(w).replace(".", "p")
                items += item(robot, f"sc_B2_{robot}_corrcost_{tagw}_{start}",
                              ["--task", "mug", "--config", "latent", "--start", start,
                               "--set", f"correction_cost_weight={w}"],
                              targets, guesses, "learned", wall, shards)
    return items


def stage_B3(wall, targets, guesses, shards):
    """Bound the top of the `correction_cost_weight` curve.

    B2 left it ambiguous: the Panda paired arm peaks at 3.0 and falls at 10.0, while the
    iiwa and the Panda native arm are still climbing at 10.0. Weight 30 says whether 10
    is near the optimum or merely the largest value anyone tried, which is the difference
    between reporting an optimum and reporting an edge of a sweep.
    """
    items = []
    for robot in ("panda", "iiwa"):
        for start in ("paired", "native"):
            items += item(robot, f"sc_B3_{robot}_corrcost_30_{start}",
                          ["--task", "mug", "--config", "latent", "--start", start,
                           "--set", "correction_cost_weight=30.0"],
                          targets, guesses, "learned", wall, shards)
    return items


G_ROWS = [
    ("panda", "mug", "paired"), ("panda", "mug", "native"),
    ("panda", "pose", "paired"), ("panda", "pose", "native"),
    ("iiwa", "mug", "paired"), ("iiwa", "mug", "native"),
    ("iiwa", "pose", "paired"), ("iiwa", "pose", "native"),
]

# Jacobian regularization settings: (name, list of --set args)
# Cross with correction_cost_weight: 0 (no penalty) and 10 (approved penalty)
G_SETTINGS = [
    # Norm clipping
    ("jnorm10", ["jacobian_max_norm=10"]),
    ("jnorm100", ["jacobian_max_norm=100"]),
    ("jnorm1k", ["jacobian_max_norm=1000"]),
    ("jnorm10k", ["jacobian_max_norm=10000"]),
    # Tikhonov/LM damping
    ("jtik0p1", ["jacobian_tikhonov_lambda=0.1"]),
    ("jtik1", ["jacobian_tikhonov_lambda=1.0"]),
    ("jtik10", ["jacobian_tikhonov_lambda=10.0"]),
    ("jtik100", ["jacobian_tikhonov_lambda=100.0"]),
    # SVD floor
    ("jsvd0p1", ["jacobian_svd_floor=0.1"]),
    ("jsvd1", ["jacobian_svd_floor=1.0"]),
]


def stage_G(wall, targets, guesses, shards, corr_cost=10.0):
    """Gradient regularization: Jacobian damping / clipping / SVD floor.

    The flow's worst-case gain is ~1e13; this damps the Jacobian before the chain rule
    so the solver sees bounded gradients while the value q is unchanged. Three strategies:
    - jacobian_max_norm: clip the Frobenius norm (isotropic)
    - jacobian_tikhonov_lambda: LM damping on singular values (anisotropic)
    - jacobian_svd_floor: floor on singular values (pseudoinverse-style)

    All 8 experiments, learned arm only, compared against Stage C's 45s column.
    """
    items = []
    for robot, task, start in G_ROWS:
        for name, sets in G_SETTINGS:
            flags = ["--task", task, "--config", "latent", "--start", start,
                     "--set", f"correction_cost_weight={corr_cost}"]
            for kv in sets:
                flags += ["--set", kv]
            items += item(robot, f"sc_G_{robot}_{task}_{int(wall)}_{start}_{name}",
                          flags, targets, guesses, "learned", wall, shards)
    return items


# Stage H: cross-testing the best Jacobian regularization against the other knobs
# this campaign has swept. The regularization setting is chosen from Stage G and
# passed in by name (--reg), so the cross is against a measured winner rather than
# a guess. Each entry is (name, list of --set args) layered on top of the
# regularization; the "none" entry is the regularization alone, which is Stage G's
# own cell and the control for every cross below it.
H_CROSSES = [
    ("alone",        []),                                   # regularization by itself
    ("corr0",        ["correction_cost_weight=0"]),         # does reg substitute for the penalty?
    ("corr30",       ["correction_cost_weight=30"]),        # ...or does it move the curve's peak?
    ("latcost0p01",  ["latent_cost_weight=0.01"]),          # helped Panda, hurt iiwa in Stage B
    ("latcost0p1",   ["latent_cost_weight=0.1"]),
    ("collinf0p2",   ["collision_influence_offset=0.2"]),   # Stage B's weak peak
    ("muadaptive",   ["ipopt_mu_strategy=adaptive"]),       # inert alone; maybe not with bounded gradients
    ("liftq",        ["lift_q=True"]),                      # Stage F2's pose collapse is an equality-row
                                                            # conditioning failure -- exactly what LM damping
                                                            # addresses, so this is the sharpest cross here
]


def stage_H(wall, targets, guesses, shards, reg, corr_cost=10.0):
    """Cross-test the winning Jacobian regularization against the other knobs.

    `reg` names an entry of G_SETTINGS. Every item carries that regularization plus
    one further change, so each row is a paired comparison against the `alone` cell
    (which is Stage G's own measurement of the same setting, re-run here so the
    cross is against a cell from this queue rather than across queues).

    correction_cost_weight defaults to the approved 10 and is overridden by the
    corr0/corr30 crosses, which come later in the --set list and therefore win.
    """
    reg_sets = dict(G_SETTINGS).get(reg)
    if reg_sets is None:
        raise SystemExit(f"--reg must be one of {sorted(dict(G_SETTINGS))}")
    items = []
    for robot, task, start in G_ROWS:
        for name, sets in H_CROSSES:
            flags = ["--task", task, "--config", "latent", "--start", start,
                     "--set", f"correction_cost_weight={corr_cost}"]
            for kv in reg_sets + sets:
                flags += ["--set", kv]
            items += item(robot, f"sc_H_{robot}_{task}_{int(wall)}_{start}_{reg}_{name}",
                          flags, targets, guesses, "learned", wall, shards)
    return items


# The corrected-formulation campaign (2026-09-03). The IK pose rows were a +-1e-4 box
# rather than an equality, so the solver parked on the face of them and the analytic arm
# was additionally handed +-0.01 rad of orientation freedom the arms it baselines did not
# get. Every table produced before the fix is void; these stages re-measure them.
#
# Runs are tagged `sc_EQ_*` via --tag-prefix so a post-fix run can never be paired against
# a pre-fix one by accident -- the grids and seeds are unchanged, so the tag is the only
# thing keeping them apart.
CORR_COST = 10.0     # Thomas's approved correction penalty


def stage_FIN(wall, targets, guesses, shards, corr_cost=CORR_COST):
    """The headline three-way comparison: both robots, both tasks, both protocols.

    All arms, so the learned/joint-space/analytic table comes from one run per cell.
    This is the first thing to run after the equality fix -- it is the table everything
    else is read against, and it surfaces trouble before the cap curve and the 480-cell
    replication are committed.
    """
    items = []
    for robot in ("panda", "iiwa"):
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                items += item(robot, f"sc_FIN_{robot}_{task}_{int(wall)}_{start}",
                              ["--task", task, "--config", "latent", "--start", start,
                               "--set", f"correction_cost_weight={corr_cost}"],
                              targets, guesses, ALL_ARMS[robot], wall, shards)
    return items


# The retrained iiwa14 chart, adopted from run iiwa14_ddp_r1 at step 620000 (2.540B
# samples). Path is relative to the repo root on the cluster; models/ is staged.
NEW_IIWA_CKPT = "models/iiwa14/iiwa14__ddp-r1__step620000.pkl"


def stage_CKPT(wall, targets, guesses, shards, corr_cost=CORR_COST, seed=1):
    """Adoption test for the retrained iiwa14 chart: ddp-r1 step 620000 against
    lemon-haze-7, on IDENTICAL cells, both tasks, both protocols.

    Both networks are re-measured in the same campaign rather than pairing the new one
    against the archived lemon-haze columns: those predate several changes to this repo,
    and a chart comparison is exactly the place where an unnoticed difference in the
    program would be indistinguishable from the effect under test.

    The claim to beat is the archived iiwa grasp deficit -- 229/480 native, 235/480
    paired -- so `mug` is the row that matters; `pose` is carried to show the change does
    not cost anything there. seed=1 and 60x8 match the headline grid.
    """
    items = []
    for name, flags in (("new", ["--checkpoint", NEW_IIWA_CKPT]), ("old", [])):
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                items += item("iiwa", f"sc_CKPT_{name}_{task}_{int(wall)}_{start}",
                              ["--task", task, "--config", "latent", "--start", start,
                               "--set", f"correction_cost_weight={corr_cost}"] + flags,
                              targets, guesses, ALL_ARMS["iiwa"], wall, shards, seed=seed)
    return items


# ---------------------------------------------------------------------------------
# The reduced-capacity chart ladder (cluster/ladder_runs.txt).
#
# Each rung is one trained chart; the benchmark question is whether a chart with less
# architectural headroom -- and so a lower worst-case gain -- wins more cells than the
# accuracy it gives up. Rung -> exported checkpoint path, relative to the repo root on
# the cluster (models/ is staged; a checkpoint's `.arch.json` sidecar travels with it and
# is what tells the program how many coupling blocks to build).
#
# `None` means "that robot's existing control": the iiwa's adopted ddp-r1 chart, and the
# Panda's upstream pretrained lp191_5.25m, which the benchmark loads when no --checkpoint
# is given. The Panda ALSO gets a self-trained 12x1024 rung (panda_n12), because without
# one every reduced Panda chart would be confounded with "our training recipe against
# Jeremy's".
LADDER_RUNGS = [
    ("iiwa", "ddpr1",    "models/iiwa14/iiwa14__ddp-r1__step620000.pkl"),   # control, 12x1024
    ("iiwa", "n8",       "models/iiwa14/iiwa14__n8__step620000.pkl"),
    ("iiwa", "n6",       "models/iiwa14/iiwa14__n6__step620000.pkl"),
    ("iiwa", "n4",       "models/iiwa14/iiwa14__n4__step620000.pkl"),
    ("iiwa", "n12w256",  "models/iiwa14/iiwa14__n12_w256__step620000.pkl"),
    ("panda", "upstream", None),                                            # control, downloaded
    ("panda", "n12",     "models/panda/panda__n12__step620000.pkl"),        # self-trained control
    ("panda", "n8",      "models/panda/panda__n8__step620000.pkl"),
    ("panda", "n6",      "models/panda/panda__n6__step620000.pkl"),
    ("panda", "n4",      "models/panda/panda__n4__step620000.pkl"),
    ("panda", "n12w256", "models/panda/panda__n12_w256__step620000.pkl"),
]

# learned + numerical only. The analytic arms never touch the flow, so a chart comparison
# cannot move them -- running them would burn compute to reproduce a constant. The
# numerical arm IS kept, on every rung, precisely because it must not move: if it does,
# the grid or the harness drifted rather than the chart.
LADDER_ARMS = "learned,numerical"


def stage_LADDER(wall, targets, guesses, shards, only=None, tag="LADDER", seed=1):
    """The depth/width ladder against its controls, on IDENTICAL cells.

    `only` is a comma-separated list of rung labels, so the stage can be generated for
    whichever rungs have finished training -- training is sequential and takes days, and
    waiting for all nine before measuring any would waste the cluster.

    Grid and seed match stage CKPT deliberately, so every rung is cell-comparable with
    BOTH the adopted ddp-r1 chart and the archived lemon-haze-7 column, and
    `scripts/collate.py --pair learned` can run exact McNemar across them.
    """
    ## Labels are NOT unique across robots -- both robots have n4/n6/n8/n12w256 rungs -- so a
    ## bare label selects the rung on EVERY robot. That silently pulled panda_n6 into an
    ## iiwa-only triage whose checkpoint would not exist for days, and those items would have
    ## failed against a missing file. Accept "robot:label" to disambiguate, and keep bare
    ## labels working for the (rare) case where selecting both robots is intended.
    wanted = set(only.split(",")) if only else None
    items = []
    for robot, label, ckpt in LADDER_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        flags = ["--checkpoint", ckpt] if ckpt else []
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                items += item(robot, f"sc_{tag}_{robot}_{label}_{task}_{int(wall)}_{start}",
                              ["--task", task, "--config", "latent", "--start", start,
                               "--set", f"correction_cost_weight={CORR_COST}"] + flags,
                              targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


# ---------------------------------------------------------------------------------
# The hardened grasp/pose problem.
#
# The grasp benchmark used to accept a target wherever a collision-free draw put the gripper,
# so targets sat in free air far more often than in clutter and the collision-avoidance half
# of the problem barely bound. `../codebase` hit the same weakness and hardened its Grasp
# Selection experiment on 2026-09-15; this is the same treatment, on the same shelves, at the
# same 0.10 m depth inset: no bin, no decorative mugs, and a target accepted only if it lands
# inside a shelf compartment (grasp targets additionally screened for the mug penetrating the
# scene).
#
# The grasp task is always shelf-contained -- a grasp target that is not in clutter is not a
# grasp-selection problem. The POSE task is fielded BOTH ways, because "does containment
# matter there?" is the open question: de-cluttering the scene without constraining targets
# makes pose strictly easier, and constraining them makes it harder, so both are measured and
# the verdict is read off the numbers rather than assumed.
#
# Nothing passes --max-target-rejections: the guard's default is already sized from the
# measured acceptance rates (scripts/probe_shelf_acceptance.py). At ../codebase's 5000 the
# iiwa pose row would trip somewhere in a 60-target grid 41% of the time.
HARD_POSE_PLACEMENTS = (("posein", "shelf"), ("posefree", "free"))
HARD_SHELF_INSET = 0.10


INSET_SWEEP = (0.0, 0.05, 0.10, 0.125)
## One rung per robot, and it is the best one, because the sweep asks whether the inset
## changes the STORY -- not which chart wins, which the ladder already answered.
INSET_RUNGS = {"iiwa": "n4", "panda": "n6"}


CAP_SWEEP = (90.0, 180.0, 360.0)


def stage_CAP(wall, targets, guesses, shards, only=None, tag="CAP", seed=1):
    """Is the iiwa's contained-grasp deficit a budget problem? Sweep the wall-clock cap.

    The diagnosis says yes. Every cell the iiwa `n4` learned arm loses to joint space on the
    contained grasp task is `fail_reason = "constraint"` with median `max_violation`
    0.009-0.037 -- centimetres off, not the >= 1e+03 that marks a runaway -- and the timeout
    counts track the lost counts closely (88 timeouts against 81 lost cells native, 75
    against 65 paired) at 398-434 median iterations and 34 ms/it. That is a solver running
    out of clock while still converging, and more clock is the direct test of it.

    `wall` is ignored; the sweep supplies its own caps. The 45 s column already exists under
    sc_HARD_iiwa_n4_mug_45_*, and **on the same grid**: the cap does not enter target
    sampling, so these are paired against it cell for cell. (The grasp task's wrist and
    fingertip containment points both resolve to `between_fingers`, so pinning fingertips as
    the default did not move this grid either.)

    Sharded 16-way rather than 8: at 360 s a shard of 60 cells that mostly times out would
    need 6 h and run_items.sh kills an item at 4 h. 30 cells caps the worst case at 3 h.
    """
    items = []
    for cap in CAP_SWEEP:
        for start in ("paired", "native"):
            items += item("iiwa",
                          f"sc_{tag}_iiwa_n4_mug_{int(cap)}_{start}",
                          ["--task", "mug", "--start", start, "--config", "latent",
                           "--set", f"correction_cost_weight={CORR_COST}",
                           "--scene", "hardened", "--target-placement", "shelf",
                           "--shelf-inset", str(HARD_SHELF_INSET),
                           "--checkpoint", "models/iiwa14/iiwa14__n4__step620000.pkl"],
                          targets, guesses, LADDER_ARMS, cap, shards, seed=seed)
    return items


def stage_POSE2(wall, targets, guesses, shards, only=None, tag="POSE2", seed=1):
    """The pose task re-measured after the gripper and containment-point fixes.

    Both of stage HARD's pose halves are void:

    * the PANDA pose scene changed outright -- it was `panda_jrl.urdf` with the stock Franka
      hand and is now the same `panda_no_hand` + finray the grasp task uses, so its collision
      geometry, its position count (9 -> 7) and its target frame all moved;
    * the CONTAINMENT POINT changed on both robots. `posein` keyed on each task's own target
      frame, which is the arm FLANGE on the iiwa (0.184 m behind the fingers) and the GRIPPER
      MOUNT on the Panda (0.100 m) -- 84 mm apart, so the two robots were not screened on the
      same point on the hand. Both now key on the gripper base, exactly 0.100 m behind
      `between_fingers` on each.

    Both placements, so the contained-vs-free verdict is re-established on the corrected
    scene rather than carried over from a table that no longer describes this program.
    """
    wanted = set(only.split(",")) if only else None
    items = []
    for robot, label, ckpt in LADDER_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        common = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                   "--scene", "hardened", "--placement-point", "wrist"]
                  + (["--checkpoint", ckpt] if ckpt else []))
        for token, placement, extra in (("posein", "shelf",
                                         ["--shelf-inset", str(HARD_SHELF_INSET)]),
                                        ("posefree", "free", [])):
            for start in ("paired", "native"):
                items += item(robot, f"sc_{tag}_{robot}_{label}_{token}_{int(wall)}_{start}",
                              ["--task", "pose", "--start", start,
                               "--target-placement", placement] + common + extra,
                              targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


def stage_FINGER(wall, targets, guesses, shards, only=None, tag="FINGER", seed=1):
    """Pose containment measured at the FINGERTIPS instead of at the target frame.

    The pose task's target is the frame the arms are given -- `iiwa_link_7` or `panda_hand`
    -- which is the WRIST. Requiring the wrist inside a compartment is a deeper and more
    awkward reach than requiring the hand inside it, and the two robots do not even agree on
    how much deeper: the iiwa's `between_fingers` is 0.184 m out from `iiwa_link_7` while the
    Panda's TCP is 0.1034 m from `panda_hand`, which already sits past the flange. (Two arms,
    three grippers: both GRASP scenes carry the same finray, but the Panda POSE scene is
    jrl's Panda with its own stock hand, because that is the model the flow was trained on.)

    Thomas, 2026-09-15: "Wrist containment > no pose containment, but tbd if fingertip
    containment is better." The wrist arm is stage HARD's `posein` columns, on the same grid
    shape and seed, so this stage is only the fingertip half.
    """
    wanted = set(only.split(",")) if only else None
    items = []
    for robot, label, ckpt in LADDER_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        common = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                   "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET),
                   "--target-placement", "shelf", "--placement-point", "fingertips"]
                  + (["--checkpoint", ckpt] if ckpt else []))
        for start in ("paired", "native"):
            items += item(robot, f"sc_{tag}_{robot}_{label}_posetip_{int(wall)}_{start}",
                          ["--task", "pose", "--start", start] + common,
                          targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


def stage_GRASPFREE(wall, targets, guesses, shards, only=None, tag="GRASPFREE", seed=1):
    """The grasp task WITHOUT containment, on the hardened scene -- the adopted default.

    Grasp containment is deferred (Thomas, 2026-09-15: its sign is opposite on the two robots
    and too many other knobs are in flight), so the default resolves to `free` for the grasp
    task. That configuration -- hardened scene, free grasp targets -- has never been measured:
    the archived columns are the LEGACY scene and stage HARD is the contained one. Without
    this the adopted default has no table.
    """
    wanted = set(only.split(",")) if only else None
    items = []
    for robot, label, ckpt in LADDER_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        common = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                   "--scene", "hardened", "--target-placement", "free"]
                  + (["--checkpoint", ckpt] if ckpt else []))
        for start in ("paired", "native"):
            items += item(robot, f"sc_{tag}_{robot}_{label}_mugfree_{int(wall)}_{start}",
                          ["--task", "mug", "--start", start] + common,
                          targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


## The adopted rungs: the chart each robot's ladder settled on. `n4` on the iiwa (its gain
## ceiling of 2.2e4 sits below the runaway band), `n6` on the Panda. Anything measuring a
## change OTHER than the chart should use these and only these.
ADOPTED_RUNGS = (("panda", "n6", "models/panda/panda__n6__step620000.pkl"),
                 ("iiwa", "n4", "models/iiwa14/iiwa14__n4__step620000.pkl"))

## The solver axis: three METHOD CLASSES, not three vendors. ipopt is interior point, snopt
## is SQP, nlopt is an augmented Lagrangian (LD_AUGLAG -- LD_SLSQP would be a second SQP
## column and no AL column). `ipopt` is included so the run carries its own baseline on its
## own grid rather than being pointed at an archived column measured on different code.
SOLVER_CLASSES = {"ipopt": "interior point", "snopt": "SQP", "nlopt": "augmented Lagrangian"}


def stage_SOLVER(wall, targets, guesses, shards, only=None, tag="SOLVER", seed=1,
                 solvers="snopt"):
    """The solver axis, on the adopted rungs and the adopted defaults.

    Every one of the ~395 archived runs is IPOPT, because `parse_log` matched only IPOPT's
    log format and a SNOPT run reported None for iterations, the evaluation counts and the
    exit -- so the axis existed as a flag and produced nothing readable. That is fixed; this
    is its first measurement.

    **Triage scale by decision (Thomas, 2026-09-16), shortest-first.** 15 targets x 4
    guesses is 60 cells, deliberately NOT the 480-cell grid the adopted columns are measured
    on: the question here is whether a solver reads sensibly on this problem at all, not
    where it ranks. Read a one- or two-cell difference as noise -- reproducibility at the cap
    is +/-1 cell.

    **Not cell-comparable with the archived IPOPT tables**, which are 480 cells at seed 1.
    The comparison this stage supports is between its own columns, which is why `ipopt` is
    generated alongside whatever else is asked for: the baseline has to come from the same
    grid and the same code.

    Adopted defaults throughout, so the solver is the only thing moving: hardened scene,
    grasp targets FREE and pose targets shelf-contained at the fingertips, both start
    protocols, learned against joint space, correction penalty 10, compiled flow Jacobian.

    Four logical runs per robot per solver (grasp x 2 starts, pose x 2 starts).
    """
    wanted = set(only.split(",")) if only else None
    chosen = [x.strip() for x in solvers.split(",") if x.strip()]
    ## `ipopt` always rides along: without a baseline on this grid there is nothing to read
    ## the new solvers against, and pointing at an archived column would silently compare
    ## across grids and across code versions.
    if "ipopt" not in chosen:
        chosen = ["ipopt"] + chosen
    for name in chosen:
        if name not in SOLVER_CLASSES:
            raise SystemExit(f"--solvers: unknown solver {name!r}; "
                             f"expected from {sorted(SOLVER_CLASSES)}")
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        ## (task, tag token, placement flags) -- the adopted default for each task.
        rows = (("mug", "mugfree", ["--target-placement", "free"]),
                ("pose", "posetip", ["--target-placement", "shelf",
                                     "--placement-point", "fingertips"]))
        for solver in chosen:
            for task, token, placement in rows:
                for start in ("paired", "native"):
                    ## The solver goes in the tag, not only the metadata. Two runs of one
                    ## grid under different solvers are different measurements, and without
                    ## it they resolve to the same summary.json and overwrite each other --
                    ## the trap --shard and --checkpoint were each fixed for.
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}_{solver}_{token}_{int(wall)}_{start}",
                                  ["--task", task, "--start", start,
                                   "--solver", solver] + placement + base,
                                  targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


## The three placements the parity suite fields. The first two are the ADOPTED defaults --
## `--target-placement auto` resolves to `free` for the grasp task and `shelf` for pose --
## and the third is the contained grasp task, which is not the default but which CLAUDE.md
## names as "the lever to revisit whenever another knob moves the picture, a solver change
## most of all": on the contained task joint space needs 970 median iterations against 48 on
## the free one, so a solver that changes how the baseline copes with a hard active set is
## exactly the thing that could move that verdict.
##
## `--placement-point` is deliberately absent from both grasp rows: the mug is welded at
## `between_fingers`, so the wrist and fingertip frames coincide there and the flag is a
## no-op. It is a 0.100 m step on the pose task only.
SOLVER2_ROWS = (("mug", "mugfree", ["--target-placement", "free"]),
                ("mug", "mugshelf", ["--target-placement", "shelf"]),
                ("pose", "posetip", ["--target-placement", "shelf",
                                     "--placement-point", "fingertips"]))


def stage_SOLVER2(wall, targets, guesses, shards, only=None, tag="SOLVER2", seed=1,
                  solvers="snopt", triage_solvers="nlopt",
                  triage_targets=15, triage_guesses=4, triage_shards=2):
    """The solver axis at campaign scale, on the adopted rungs and the adopted defaults.

    Stage SOLVER measured this axis once, at 60 cells, and found IPOPT ahead on all eight
    rows -- on BOTH arms, which is what says it is a property of the problem rather than of
    the learned formulation. This is the same question at the campaign's own 480 cells, plus
    the contained grasp rows that a solver change is the stated reason to re-try.

    Two grid sizes, which is why the cell count is in the tag. `solvers` run at full scale;
    `triage_solvers` run at 15 x 4 -- stage SOLVER's own grid, so those columns are
    cell-comparable with the IPOPT and SNOPT triage columns already on disk. NLopt is there
    because the axis stands for three METHOD CLASSES and the augmented Lagrangian has only
    ever run in local smoke tests, where it solved nothing; a cheap honest column beats an
    expensive one for a result that may well be empty.

    `ipopt` always rides along at full scale rather than being read off the archived
    sc_GRASPFREE / sc_FINGER columns. That costs half the compute and buys two things worth
    more: the comparison sits inside one code version and one machine state, and reproducing
    those archived columns is itself the check that this branch did not perturb IPOPT.
    """
    wanted = set(only.split(",")) if only else None
    chosen = [x.strip() for x in solvers.split(",") if x.strip()]
    if "ipopt" not in chosen:
        chosen = ["ipopt"] + chosen
    triage = [x.strip() for x in triage_solvers.split(",") if x.strip()] if triage_solvers else []
    for name in chosen + triage:
        if name not in SOLVER_CLASSES:
            raise SystemExit(f"--solvers: unknown solver {name!r}; "
                             f"expected from {sorted(SOLVER_CLASSES)}")
    overlap = set(chosen) & set(triage)
    if overlap:
        ## Two grid sizes under one solver name would differ only by the cell token, which
        ## is a comparison nobody asked for and an easy way to pair the wrong pair.
        raise SystemExit(f"--solvers and --triage-solvers overlap on {sorted(overlap)}; "
                         "a solver belongs to one scale or the other")
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        for solver, t, g, sh in ([(x, targets, guesses, shards) for x in chosen]
                                 + [(x, triage_targets, triage_guesses, triage_shards)
                                    for x in triage]):
            for task, token, placement in SOLVER2_ROWS:
                for start in ("paired", "native"):
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}_{solver}_{token}_{t * g}"
                                  f"_{int(wall)}_{start}",
                                  ["--task", task, "--start", start,
                                   "--solver", solver] + placement + base,
                                  t, g, LADDER_ARMS, wall, sh, seed=seed)
    return items


## The solver-settings sweep. ONE FACTOR AT A TIME against each solver's own defaults, so a
## row reads as "what this knob is worth", not as a joint optimum found by search.
##
## STEP REJECTION IS DELIBERATELY ABSENT, both solvers: ipopt_theta_max_fact,
## ipopt_watchdog_trigger, ipopt_max_soc, snopt_violation_limit and snopt_major_step_limit
## are a separate question (Thomas, 2026-09-16) and the selftest asserts none of them
## appears here. Do not add one to this table -- add it to that stage when it exists.
SWEEP_SNOPT = [
    ## snopt_major_step_limit was ADOPTED at 0.5 on 2026-09-19, so "set nothing" no
    ## longer means Drake's defaults. This column does, and now says so explicitly;
    ## without the =None a re-generation would silently measure the tuned value.
    ("default", ["snopt_major_step_limit=None"]),
    ## The flow Jacobian is ~84% of a solve, so evaluations per major IS the cost model.
    ("lstol0p1", ["snopt_linesearch_tolerance=0.1"]),
    ("lstol0p5", ["snopt_linesearch_tolerance=0.5"]),
    ("lstol0p99", ["snopt_linesearch_tolerance=0.99"]),
    ## ...and the largest available saving: a line search that never asks for a gradient.
    ("nonderivls", ["snopt_nonderivative_linesearch=True"]),
    ## The QP subproblem costs NO function evaluations, so more minors may be nearly free.
    ("minor50", ["snopt_minor_iterations_limit=50"]),
    ("minor2k", ["snopt_minor_iterations_limit=2000"]),
    ## The counterpart of ipopt_nlp_scaling_method, which measured inert. n4's Jacobian
    ## gain reaches ~2e4, so the problem is genuinely badly scaled.
    ("scale1", ["snopt_scale_option=1"]),
    ("scale2", ["snopt_scale_option=2"]),
    ## Feasibility stays two orders under the 1e-3 task gate at its loosest: rung 3 of the
    ## tolerance ladder says never let a gate sit at a bound the solver optimises against.
    ("majfeas1em08", ["snopt_major_feasibility_tol=1e-8"]),
    ("majfeas1em05", ["snopt_major_feasibility_tol=1e-5"]),
    ("majopt1em08", ["snopt_major_optimality_tol=1e-8"]),
    ("majopt1em04", ["snopt_major_optimality_tol=1e-4"]),
    ## SNOPT never resets its quasi-Newton approximation by default (99999999). The chart's
    ## gain varies by orders of magnitude across the domain, so a stale one is suspect.
    ("hessfreq20", ["snopt_hessian_frequency=20"]),
    ("hessfreq100", ["snopt_hessian_frequency=100"]),
    ## The paired start is infeasible by policy, and elastic mode is how SNOPT copes.
    ("elastic1e2", ["snopt_elastic_weight=100.0"]),
    ("elastic1e7", ["snopt_elastic_weight=1e7"]),
    ("crash0", ["snopt_crash_option=0"]),
    ## How far the first major is allowed to move from the start we hand it.
    ("prox0", ["snopt_proximal_point_method=0"]),
    ("prox2", ["snopt_proximal_point_method=2"]),
]

## IPOPT's first three entries are not tuning, they are a FAIRNESS question. Every archived
## run fields IPOPT with acceptable_tol=1e-3 and acceptable_iter=1, so it may stop after a
## SINGLE iteration meeting a relaxed test, while SNOPT converges to its own 1e-6 majors.
## That asymmetry was inherited from the original setup and never chosen; it is a live
## candidate explanation for IPOPT's clean sweep of the triage, and it has to be measured
## before the solver table is written up.
IPOPT_ACCEPTABLE_TIGHT = ["acceptable_tol=1e-6", "acceptable_constr_viol_tol=1e-6",
                             "acceptable_dual_inf_tol=1e-6", "acceptable_compl_inf_tol=1e-6",
                             "acceptable_iter=15"]
IPOPT_ACCEPTABLE_OFF = ["acceptable_tol=1e-12", "acceptable_constr_viol_tol=1e-12",
                        "acceptable_dual_inf_tol=1e-12", "acceptable_compl_inf_tol=1e-12",
                        "acceptable_iter=100000"]
## And the OTHER half of the same fairness question, which is the larger half. The
## `acceptable_*` family above is IPOPT's relaxed EARLY STOP; `tol`/`constr_viol_tol`/
## `dual_inf_tol`/`compl_inf_tol` are what it actually converges to, and this repo has
## never set them -- so IPOPT has been allowed 1e-4 constraint violation against SNOPT's
## 1e-6, and dual infeasibility 1 against SNOPT's 2e-6. `convsnopt` holds IPOPT to SNOPT's
## own numbers; `fair` does that AND disables the early stop, i.e. the most symmetric
## comparison the two solvers admit.
IPOPT_CONVERGENCE_AS_SNOPT = ["ipopt_tol=1e-6", "ipopt_constr_viol_tol=1e-6",
                              "ipopt_dual_inf_tol=1e-6", "ipopt_compl_inf_tol=1e-6"]
SWEEP_IPOPT = [
    ("default", []),
    ("acctight", IPOPT_ACCEPTABLE_TIGHT),
    ("accoff", IPOPT_ACCEPTABLE_OFF),
    ("acciter15", ["acceptable_iter=15"]),
    ("acciter5", ["acceptable_iter=5"]),
    ## IPOPT's TRUE defaults for the acceptable family, read out of its own
    ## `print_options_documentation` dump rather than from memory -- which matters, because
    ## they are not uniformly tighter or looser than what this repo fields. IPOPT defaults
    ## to acceptable_tol 1e-6 and acceptable_iter 15 (both TIGHTER than the fielded 1e-3
    ## and 1), but to acceptable_dual_inf_tol 1e10 and the two infeasibility tolerances
    ## 1e-2 (all three far LOOSER than the fielded 1e-4). So "IPOPT at its own defaults" is
    ## its own arm and cannot be inferred from the single-factor rows.
    ("ipoptdefault", ["acceptable_tol=1e-6", "acceptable_constr_viol_tol=1e-2",
                      "acceptable_dual_inf_tol=1e10", "acceptable_compl_inf_tol=1e-2",
                      "acceptable_iter=15"]),
    ## The loose infeasibility triple alone, holding acceptable_tol/iter as fielded.
    ("accviolloose", ["acceptable_constr_viol_tol=1e-2", "acceptable_dual_inf_tol=1e10",
                      "acceptable_compl_inf_tol=1e-2"]),
    ("convsnopt", IPOPT_CONVERGENCE_AS_SNOPT),
    ("fair", IPOPT_ACCEPTABLE_OFF + IPOPT_CONVERGENCE_AS_SNOPT),
    ("cvt1em06", ["ipopt_constr_viol_tol=1e-6"]),
    ("cvt1em08", ["ipopt_constr_viol_tol=1e-8"]),
    ("dualinf1em06", ["ipopt_dual_inf_tol=1e-6"]),
    ("tol1em06", ["ipopt_tol=1e-6"]),
    ("tol1em10", ["ipopt_tol=1e-10"]),
    ## Drake gives IPOPT no second derivatives, so the L-BFGS history IS the Hessian here.
    ("lmhist3", ["ipopt_limited_memory_max_history=3"]),
    ("lmhist12", ["ipopt_limited_memory_max_history=12"]),
    ("lmhist25", ["ipopt_limited_memory_max_history=25"]),
    ("lmsr1", ["ipopt_limited_memory_update_type=sr1"]),
    ## mu_init is read ONLY under mu_strategy=monotone -- alone it is echoed `used = no`.
    ("muinit1em03", ["ipopt_mu_strategy=monotone", "ipopt_mu_init=1e-3"]),
    ("muinit1", ["ipopt_mu_strategy=monotone", "ipopt_mu_init=1.0"]),
    ("muadaptive", ["ipopt_mu_strategy=adaptive"]),
    ("alphaybound", ["ipopt_alpha_for_y=bound-mult"]),
    ("recalcy", ["ipopt_recalc_y=yes"]),
    ("brf0", ["ipopt_bound_relax_factor=0.0"]),
    ("scalenone", ["ipopt_nlp_scaling_method=none"]),
    ("scalemax1e4", ["ipopt_nlp_scaling_max_gradient=1e4"]),
]

SWEEP_SETTINGS = {"snopt": SWEEP_SNOPT, "ipopt": SWEEP_IPOPT}

## The rows the sweep runs on: both robots, both ADOPTED-default tasks, `paired` only. That
## protocol is where the SNOPT deficit concentrated (15 / 20 / 9 / 25 cells of 60 lost) and
## it is the diagnostic one, since it hands every arm the same infeasible start. Whatever
## survives here gets confirmed on `native` and at 480 cells, which is a follow-up, not an
## assumption -- say so when reporting, because a knob screened on one protocol is not a
## knob measured on both.
SWEEP_ROWS = (("mug", "mugfree", ["--target-placement", "free"]),
              ("pose", "posetip", ["--target-placement", "shelf",
                                   "--placement-point", "fingertips"]))
## Everything the step-rejection question owns. The selftest refuses a sweep entry naming
## any of these, so the two questions cannot silently merge.
STEP_REJECTION_KNOBS = ("ipopt_theta_max_fact", "ipopt_watchdog_trigger", "ipopt_max_soc",
                        "snopt_violation_limit", "snopt_major_step_limit")


def FieldsAValue(knob):
    """True if a `NAME=VALUE` override FIELDS a setting, False if it restores a solver default.

    `NAME=None` emits nothing at all -- `_SnoptOptions`/`_IpoptOptions` skip a None -- so it is
    the exact opposite of fielding a setting, and every blacklist and disjointness check below
    has to stop counting it as one. This became necessary on 2026-09-19, when
    `snopt_major_step_limit` was adopted at 0.5: from that point an entry that sets nothing no
    longer means "Drake's SNOPT defaults", so the four historical SNOPT tables say `=None`
    explicitly, and they must not thereby look like they field a step-rejection knob.
    """
    return knob.split("=", 1)[1] != "None"


def stage_SWEEP(wall, targets, guesses, shards, only=None, tag="SWEEP", seed=1,
                solvers="ipopt,snopt"):
    """Solver settings, one factor at a time, on stage SOLVER's own 60-cell grid.

    The grid is deliberately the triage one (15 x 4, seed 1) rather than the campaign's 480
    cells: the question is directional, exactly as `stage_INSET` argued, and using the same
    grid makes every sweep cell pair against the default columns already measured for both
    solvers. Read a one-cell difference as noise -- reproducibility at the cap is +/-1 cell.

    Each solver keeps its own defaults as its baseline. The two columns are NOT swept
    against each other here; that is what stage SOLVER2 is for.
    """
    wanted = set(only.split(",")) if only else None
    chosen = [x.strip() for x in solvers.split(",") if x.strip()]
    for name in chosen:
        if name not in SWEEP_SETTINGS:
            raise SystemExit(f"--solvers: {name!r} has no settings table; "
                             f"expected from {sorted(SWEEP_SETTINGS)}")
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        for solver in chosen:
            for name, sets in SWEEP_SETTINGS[solver]:
                for knob in sets:
                    ## `NAME=None` is the one permitted form: it RESTORES the solver's own
                    ## default, which is the opposite of fielding a step-rejection setting.
                    ## It became necessary on 2026-09-19, when snopt_major_step_limit was
                    ## adopted at 0.5 -- after which an entry that sets nothing no longer
                    ## means "Drake's defaults" and a `default` column has to say so.
                    if (knob.split("=")[0] in STEP_REJECTION_KNOBS
                            and FieldsAValue(knob)):
                        raise SystemExit(
                            f"stage_SWEEP entry {name!r} names {knob.split('=')[0]!r}, "
                            "which belongs to the separate step-rejection question")
                for task, token, placement in SWEEP_ROWS:
                    args = (["--task", task, "--start", "paired", "--solver", solver]
                            + placement + base)
                    for knob in sets:
                        args += ["--set", knob]
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}_{solver}_{token}"
                                  f"_{int(wall)}_paired_{name}",
                                  args, targets, guesses, LADDER_ARMS, wall, shards,
                                  seed=seed)
    return items


## ======================= Step rejection: the last lever on the solver axis ==============
##
## `stage_SWEEP` deliberately EXCLUDES this family and raises if one of its knobs appears
## there; this stage owns them, and reuses the same tuple as a WHITELIST so the two
## questions cannot silently merge from either side.
##
## Step rejection is not the same lever as the gradient damping that was refuted. Damping
## altered the DERIVATIVES IPOPT was handed and broke the correspondence between the values
## it evaluates and the gradients it uses (310 cells better against 1,509 worse, line
## closed). Filter tuning leaves the program exactly as written and changes only which trial
## points are ACCEPTED. And per Thomas it is worth measuring whether or not anything runs
## away: "step rejection (and trust region ideas) can still help even if we're not running
## away. There's a reason people like trust region solvers."
##
## WHAT THE OUTCOME VARIABLE IS, since the historical one is gone. The 16-cell lead was
## measured on `ddp-r1` when the runaway was live. On the adopted rungs there are ZERO
## runaway cells, so runaway count measures nothing. The two live targets, from the 480-cell
## paired columns: the iiwa grasp rows fail ~100% at the wall clock (budget bound -- step
## control helps only by making iterations more productive), and the Panda pose row does NOT
## (75 failures, 5 at the clock, median violation 2.3e-01 -- stalling with budget left).
## Those are different mechanisms and are reported separately.

## The IPOPT table. The band is narrow BY MEASUREMENT, not by taste -- the old probe's raw
## results are still on disk under results/iiwa/benchmark/:
##
##   full_base       11/16   mean iters 115.818181...   median cost 7.611527805
##   full_theta1     14/16   +3 / -0 per cell           cost 6.754
##   full_theta10    11/16   0 / 0 -- BIT-IDENTICAL to full_base, to fourteen digits
##   x_pose_theta0p1 10/16   +1 / -2                    net negative
##
## So theta_max_fact >= 10 is a PROVEN NULL and must not be re-added. The mechanism for the
## low end is readable in IPOPT's source: IpFilterLSAcceptor.cpp:328 sets
## theta_max = theta_max_fact * max(1, reference_theta) ONCE, from the first iterate's
## violation (:259). Iterate-0 inf_pr on the adopted n4 rung is 1.15-2.91, i.e. O(1), so a
## factor below 1 puts the ceiling UNDER the start's own violation and every trial point is
## refused until IPOPT drops into feasibility restoration -- on the probe's lost cell, 100 of
## 176 iterations in restoration against 2 of 36 at the default. The legality floor is
## theta_min_fact (1e-4, asserted < theta_max_fact at :214); the USEFUL floor is 1.0. One
## sub-1 arm is kept as the mechanism exhibit, on the adopted rungs so the claim is measured
## here rather than inherited from ddp-r1.
##
## max_soc and watchdog_shortened_iter_trigger are CONDITIONALLY meaningful, which is why
## they appear crossed. Both echo `used = yes` and both measured equal to the default at the
## default 1e4 ceiling -- as they must: at that ceiling the filter essentially never rejects
## on violation, so the machinery that rescues violation-rejected steps has nothing to act
## on. `soc0` stays as the one unconditional control, where it also tests a throughput angle
## (each second-order correction costs a constraint evaluation, which here is a flow
## Jacobian).
STEP_IPOPT = [
    ("default", []),
    ## The live axis, [1, 10). theta1 is the only value with a positive measurement.
    ("theta1", ["ipopt_theta_max_fact=1.0"]),
    ("theta2", ["ipopt_theta_max_fact=2.0"]),
    ("theta3", ["ipopt_theta_max_fact=3.0"]),
    ("theta5", ["ipopt_theta_max_fact=5.0"]),
    ## Below 1: predicted to force restoration from iteration 1. The exhibit, not a candidate.
    ("theta0p3", ["ipopt_theta_max_fact=0.3"]),
    ## The two rescue mechanisms, crossed with a ceiling tight enough to give them something
    ## to rescue. Bare soc8/wdoff were measured equal to the default and are not repeated.
    ("theta1soc8", ["ipopt_theta_max_fact=1.0", "ipopt_max_soc=8"]),
    ("theta1wdoff", ["ipopt_theta_max_fact=1.0", "ipopt_watchdog_trigger=0"]),
    ## The one unconditional arm: no second-order corrections at all, which also makes every
    ## rejected iteration cheaper on a problem where a constraint evaluation is a flow pass.
    ("soc0", ["ipopt_max_soc=0"]),
]

## SNOPT's counterparts. Secondary by decision -- the axis is settled, IPOPT > SNOPT >>>
## NLopt at 480 cells on both arms -- but one hypothesis here is real rather than a fishing
## expedition. SNOPT's failures under `paired` are 36.2% INFO 41 `current point cannot be
## improved`, which IS a line search that cannot find an acceptable step, so a SMALLER
## `Major step limit` may let it succeed; mstep10 is the control in the other direction.
##
## `Violation limit` is swept in BOTH directions on purpose. SNOPT's dominant failure is INFO
## 13 `nonlinear infeasibilities minimized` at 50.7%, i.e. it gives up on feasibility.
## Tightening the violation limit pushes it into elastic mode earlier and plausibly makes
## INFO 13 MORE likely, so loosening is the untested direction that could actually help.
STEP_SNOPT = [
    ## snopt_major_step_limit was ADOPTED at 0.5 on 2026-09-19, so "set nothing" no
    ## longer means Drake's defaults. This column does, and now says so explicitly;
    ## without the =None a re-generation would silently measure the tuned value.
    ("default", ["snopt_major_step_limit=None"]),
    ("mstep0p1", ["snopt_major_step_limit=0.1"]),
    ("mstep0p5", ["snopt_major_step_limit=0.5"]),
    ("mstep10", ["snopt_major_step_limit=10.0"]),
    ("viol1", ["snopt_violation_limit=1.0"]),
    ("viol100", ["snopt_violation_limit=100.0"]),
    ("sstrict", ["snopt_major_step_limit=0.5", "snopt_violation_limit=1.0"]),
]

STEP_SETTINGS = {"ipopt": STEP_IPOPT, "snopt": STEP_SNOPT}

## All three placements, unlike stage SWEEP's two. `mugshelf` is where the last deficit in
## the project lives, and it is the row with the sharp bar: of its 74 paired failures at
## 45 s, the 180 s CAP column solves exactly 47 and the 360 s column adds nothing, so +47 ->
## 453/480 is the budget-recoverable ceiling and those 47 cells are individually named.
##
## The two `mugfree` rows are at 58/60 and 59/60, so they can only register HARM, not gain --
## kept deliberately (Thomas, 2026-09-16), because a step-rejection knob is a global solver
## setting and must be shown not to regress the adopted default grasp row. Selection is read
## off `mugshelf` + `posetip` only; say so when reporting rather than pooling all six rows
## into one selection number.
STEP_ROWS = SOLVER2_ROWS


def stage_STEP(wall, targets, guesses, shards, only=None, tag="STEP", seed=1,
               solvers="ipopt,snopt", settings=None, starts="paired"):
    """Step rejection and step-size limits, one factor at a time, against each solver's own
    defaults.

    Runs at TWO scales, which is why the cell count is in the tag exactly as `stage_SOLVER2`
    does it: a 60-cell screen on stage SOLVER's own grid (so the mugfree and posetip columns
    pair against the defaults already on disk), then a 480-cell confirmation of whatever
    survives. Those grids are DIFFERENT OBJECTS -- 15x4 hashes 0a6d3cba534e-mug, 60x8 free
    hashes fdca0bad64bc-mug and 60x8 shelf hashes fa692df81e7d-mug -- so the screen cannot
    address the 47-cell question and the confirmation must report fa692df81e7d-mug or that
    question is unanswerable.

    `settings` filters the tables by token, which is how the confirmation fields only the
    survivors without a code edit. `starts` is `paired` for the screen (the diagnostic
    protocol, which hands every arm the same infeasible start) and `paired,native` for the
    confirmation. Do NOT read a protocol conclusion off the 60-cell screen -- that has
    misled twice in this repo, and `theta_max_fact` makes it sharper than usual, because
    under `native` the start is near-feasible so `max(1, theta_0)` is 1 and the ceiling
    becomes theta_max_fact in ABSOLUTE units. A different regime, not a weaker version.

    Nothing here is adopted whatever the result (Thomas's call on this plan): a knob that
    wins is reported, not fielded, because changing a default breaks comparability with every
    archived run.
    """
    wanted = set(only.split(",")) if only else None
    chosen = [x.strip() for x in solvers.split(",") if x.strip()]
    for name in chosen:
        if name not in STEP_SETTINGS:
            raise SystemExit(f"--solvers: {name!r} has no step-rejection settings table; "
                             f"expected from {sorted(STEP_SETTINGS)}")
    want_starts = [x.strip() for x in starts.split(",") if x.strip()]
    for s in want_starts:
        if s not in ("paired", "native"):
            raise SystemExit(f"--starts: unknown protocol {s!r}; expected paired or native")
    keep = set(settings.split(",")) if settings else None
    ## The WHITELIST, and it is asserted over the TABLES rather than over the emitted args:
    ## every row carries `--set correction_cost_weight=...` from `base`, so an args-level
    ## check would fire on that and the stage could never generate at all. `stage_SWEEP` uses
    ## the same tuple as a blacklist, so between them the two questions are provably disjoint.
    for solver in chosen:
        for name, sets in STEP_SETTINGS[solver]:
            for knob in sets:
                if knob.split("=")[0] not in STEP_REJECTION_KNOBS:
                    raise SystemExit(
                        f"stage_STEP entry {name!r} names {knob.split('=')[0]!r}, which is "
                        "not a step-rejection knob -- this stage owns that family and "
                        "nothing else, or the sweep and the step question merge")
    if keep is not None:
        known = {n for s in chosen for n, _ in STEP_SETTINGS[s]}
        unknown = keep - known
        if unknown:
            raise SystemExit(f"--settings: no such token(s) {sorted(unknown)}; "
                             f"expected from {sorted(known)}")
        if "default" not in keep:
            ## Without the untouched baseline on the same grid and the same code there is
            ## nothing to read the surviving settings against.
            raise SystemExit("--settings must include 'default': a confirmation without its "
                             "own baseline column pairs against an archived one, i.e. across "
                             "code versions")
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        for solver in chosen:
            for name, sets in STEP_SETTINGS[solver]:
                if keep is not None and name not in keep:
                    continue
                for task, token, placement in STEP_ROWS:
                    for start in want_starts:
                        args = (["--task", task, "--start", start, "--solver", solver]
                                + placement + base)
                        for knob in sets:
                            args += ["--set", knob]
                        items += item(robot,
                                      f"sc_{tag}_{robot}_{label}_{solver}_{token}"
                                      f"_{targets * guesses}_{int(wall)}_{start}_{name}",
                                      args, targets, guesses, LADDER_ARMS, wall, shards,
                                      seed=seed)
    return items


## ================= SNOPT's own configuration, at campaign scale =========================
##
## WHY THIS STAGE EXISTS, and it is a FAIRNESS repair rather than tuning for its own sake.
## Every table in the campaign fields IPOPT with the acceptable-point early stop
## (acceptable_tol=1e-4, acceptable_iter=1) and SNOPT at bare Drake defaults. That early stop
## is worth 39 cells and a 9x speedup on stage SWEEP's grid -- turning it off drops IPOPT to
## 62/240 -- so the solver-class comparison currently reads "a tuned interior-point method
## against an untuned SQP". Thomas, 2026-09-17:
##
##     I'm okay with playing with solver settings on a per-solver basis, as long as it's not
##     per-problem.
##
## So SNOPT is entitled to its own configuration, chosen ONCE and applied to every row. The
## rule cuts the other way too: a setting that helps one robot and hurts the other is not
## adoptable by picking the robot it helps, which is why every candidate runs on all twelve
## rows and the decision rule below counts rows rather than pooling them.
##
## THE CANDIDATES are stage SWEEP's own top performers plus the one step-rejection knob that
## screened well, and then the combinations. None of the singles reached significance at 60
## cells (best was +12 of 240, p = 0.20), which is the entire reason this runs at 480: stage
## STEP's headline is a 60-cell lead that REVERSED at 480 with its mechanism column reversing
## alongside it. Treat every number below as a direction, not a result.
##
##   nonderivls   +12/240, never loses a row, and the only candidate with a mechanism tied to
##                the gradients: INFO 41 `current point cannot be improved` -- the documented
##                signature of inaccurate or badly scaled derivatives -- falls 41 -> 32.
##                Thomas's reading is that SNOPT's derivative-based line search does not cope
##                with the flow's Jacobian, and a line search that never asks for a gradient
##                is the direct test of it.
##   hessfreq20   +11/240, and its strongest single row is iiwa grasp 40 -> 48 (p = 0.077),
##                the row SNOPT is worst on. SNOPT never resets its quasi-Newton approximation
##                by default (99999999) and the chart's gain varies by orders of magnitude
##                across the domain, so a stale one is suspect. NOTE `Hessian updates` is
##                inert below 75 variables and these programs have 20-21; frequency is the
##                knob that bites.
##   lstol0p99    +11/240. The loose end of the line-search tolerance, i.e. accept sooner.
##   lstol0p1     +6/240 pooled but the best single setting on iiwa pose (25 -> 34).
##   majopt1em08  +11/240. Converge harder rather than sooner -- the opposite direction to
##                everything else here, kept precisely for that reason.
##   elastic1e2   +0/240 pooled but the best setting on Panda grasp (51 -> 53); the paired
##                start is infeasible by policy and elastic mode is how SNOPT copes with it.
##   crash0       +7/240. Cold-start the basis instead of crashing one.
##   mstep0p5     +15/360 on stage STEP's six rows (p = 0.18) and a DIFFERENT mechanism from
##                every other entry: INFO 41 stays at exactly 81 while INFO 13 `nonlinear
##                infeasibilities minimized` falls 51 -> 42. It helps SNOPT REACH feasibility
##                rather than escape a stalled line search. It is also the one candidate
##                measured as a robot TRADE (iiwa +17, Panda -2), which is exactly the thing
##                480 cells has to resolve before the per-solver rule can accept it.
##
## `snopt_major_step_limit` belongs to STEP_REJECTION_KNOBS, which stage_SWEEP blacklists and
## stage_STEP whitelists so those two questions can never merge. This stage has neither guard
## ON PURPOSE: step rejection as a QUESTION is closed (measured, refuted, nothing adopted), so
## the knob is available here as an ordinary SNOPT setting among others. Do not re-add a guard
## without also deciding what happens to the combination rows.
##
## THE COMBINATIONS are the part 60 cells could not address at all. nonderivls and mstep0p5
## cut different exit codes, so they are the one pair with a reason to be additive rather than
## redundant; hessfreq20 is crossed in because its gain sits on the row the other two are
## weakest on.
SNOPTTUNE_SNOPT = [
    ## snopt_major_step_limit was ADOPTED at 0.5 on 2026-09-19, so "set nothing" no
    ## longer means Drake's defaults. This column does, and now says so explicitly;
    ## without the =None a re-generation would silently measure the tuned value.
    ("default", ["snopt_major_step_limit=None"]),
    ("nonderivls", ["snopt_nonderivative_linesearch=True"]),
    ("hessfreq20", ["snopt_hessian_frequency=20"]),
    ("hessfreq100", ["snopt_hessian_frequency=100"]),
    ("lstol0p99", ["snopt_linesearch_tolerance=0.99"]),
    ("lstol0p1", ["snopt_linesearch_tolerance=0.1"]),
    ("majopt1em08", ["snopt_major_optimality_tol=1e-8"]),
    ("elastic1e2", ["snopt_elastic_weight=100.0"]),
    ("crash0", ["snopt_crash_option=0"]),
    ("mstep0p5", ["snopt_major_step_limit=0.5"]),
    ("ndlsmstep", ["snopt_nonderivative_linesearch=True",
                   "snopt_major_step_limit=0.5"]),
    ("ndlshess20", ["snopt_nonderivative_linesearch=True",
                    "snopt_hessian_frequency=20"]),
    ("ndlshess20mstep", ["snopt_nonderivative_linesearch=True",
                         "snopt_hessian_frequency=20",
                         "snopt_major_step_limit=0.5"]),
]

## Every entry must name a SNOPT option and nothing else: this stage decides SNOPT's column
## and an IPOPT knob here would silently make it a two-solver comparison.
SNOPTTUNE_SETTINGS = {"snopt": SNOPTTUNE_SNOPT}


def stage_SNOPTTUNE(wall, targets, guesses, shards, only=None, tag="SNOPTTUNE", seed=1,
                    settings=None, starts="paired,native"):
    """SNOPT's own best configuration, twelve rows at 480 cells, one setting for all of them.

    Rows are `SOLVER2_ROWS` x both protocols x both adopted rungs -- the same twelve rows the
    archived `sc_SOLVER2_*_snopt_*` columns cover, on the same grids, so every cell pairs
    against the fielded default. The fresh `default` column rides along anyway rather than
    being read off that archive: it absorbs any difference in code version or in node
    contention, and reproducing the archived counts is itself the check that nothing else
    moved. Compare WITHIN this run, not against the archive.

    THE DECISION RULE IS PRE-REGISTERED, because thirteen columns x twelve rows is 156
    McNemar tests and that many will manufacture a winner. A setting is adopted as SNOPT's
    configuration only if, on the LEARNED arm, it beats `default` on at least 9 of the 12
    rows, is significantly worse (p < 0.05) on none, and is significantly better on at least
    one. No pooling across experiments -- Thomas, 2026-09-17: "Do not pool experiments, that
    is useless." The same configuration must apply to every row; a per-experiment pick is
    exactly what the per-solver rule forbids.

    Expect this NOT to flip a verdict. The candidates are worth about +12 of 240 on the sweep
    grid, so ~+24 of 480; applied to the rows SNOPT loses (iiwa grasp native 348 v 457,
    paired 333 v 398) that is nowhere near parity. The point is a defensible SNOPT column,
    not a rescue.
    """
    wanted = set(only.split(",")) if only else None
    want_starts = [x.strip() for x in starts.split(",") if x.strip()]
    for s in want_starts:
        if s not in ("paired", "native"):
            raise SystemExit(f"--starts: unknown protocol {s!r}; expected paired or native")
    keep = set(settings.split(",")) if settings else None
    for name, sets in SNOPTTUNE_SNOPT:
        for knob in sets:
            if not knob.startswith("snopt_"):
                raise SystemExit(
                    f"stage_SNOPTTUNE entry {name!r} names {knob.split('=')[0]!r}, which is "
                    "not a SNOPT option -- this stage decides SNOPT's column alone")
    if keep is not None:
        known = {n for n, _ in SNOPTTUNE_SNOPT}
        unknown = keep - known
        if unknown:
            raise SystemExit(f"--settings: no such token(s) {sorted(unknown)}; "
                             f"expected from {sorted(known)}")
        if "default" not in keep:
            raise SystemExit("--settings must include 'default': without its own baseline "
                             "column a run pairs against an archived one, i.e. across code "
                             "versions and node contention")
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        for name, sets in SNOPTTUNE_SNOPT:
            if keep is not None and name not in keep:
                continue
            for task, token, placement in SOLVER2_ROWS:
                for start in want_starts:
                    args = (["--task", task, "--start", start, "--solver", "snopt"]
                            + placement + base)
                    for knob in sets:
                        args += ["--set", knob]
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}_snopt_{token}"
                                  f"_{targets * guesses}_{int(wall)}_{start}_{name}",
                                  args, targets, guesses, LADDER_ARMS, wall, shards,
                                  seed=seed)
    return items


## ---------------------------------------------------------------------------- SNOPTCOMBO --
##
## Stage SNOPTTUNE fielded thirteen SNOPT settings at 480 cells x 12 rows and found ONE
## survivor of the pre-registered bar, `Major step limit = 0.5` (11 of 12 rows better on the
## learned arm, 3 significantly better, 0 significantly worse). Three further factors were
## positive and failed the bar: `Elastic weight = 100` (8 better / 4 worse, 3 significant
## gains -- one row short of the >= 9 clause), `Hessian frequency = 20` (8/4) and
## `Major optimality tolerance = 1e-8` (8/3, no significant gain).
##
## But all four COMBINATIONS that stage tested contained `Nonderivative linesearch`, which
## turned out to be the harmful factor: 6 better / 6 worse alone, and it dragged down every
## combination it appeared in, including both that also carried the winner. So the survivor
## was never crossed with the three other positive factors. That is what this stage does.
##
## `Nonderivative linesearch` and `Crash option = 0` are therefore DELIBERATELY ABSENT -- both
## are refuted at 480 cells (6/6 and 5/7) and the former poisoned SNOPTTUNE's combinations.
## Do not re-add them; a cross with a refuted factor is a cross with a refuted factor.
##
## Expect this NOT to reach parity with IPOPT. Pooled over these twelve rows the learned arm
## scores 3981/5760 at SNOPT's defaults and 4128 with the step limit, against IPOPT's 5302, so
## the survivor closes ~12% of the gap and factors of similar size will not close the rest.
## The deliverable is a defensible SNOPT column, not a rescue.
SNOPTCOMBO_SNOPT = [
    ## snopt_major_step_limit was ADOPTED at 0.5 on 2026-09-19, so "set nothing" no
    ## longer means Drake's defaults. This column does, and now says so explicitly;
    ## without the =None a re-generation would silently measure the tuned value.
    ("default", ["snopt_major_step_limit=None"]),
    ## The two single factors ride along rather than being read off the SNOPTTUNE archive, so
    ## every comparison is WITHIN one run on the same cells; reproducing their archived counts
    ## is itself the check that nothing else moved.
    ("mstep0p5", ["snopt_major_step_limit=0.5"]),
    ("elastic1e2", ["snopt_elastic_weight=100.0"]),
    ("mstepelastic", ["snopt_major_step_limit=0.5", "snopt_elastic_weight=100.0"]),
    ("mstephess20", ["snopt_major_step_limit=0.5", "snopt_hessian_frequency=20"]),
    ("mstepmajopt", ["snopt_major_step_limit=0.5", "snopt_major_optimality_tol=1e-8"]),
    ("mstepelastichess", ["snopt_major_step_limit=0.5", "snopt_elastic_weight=100.0",
                          "snopt_hessian_frequency=20"]),
    ("mstepelastichessmajopt", ["snopt_major_step_limit=0.5", "snopt_elastic_weight=100.0",
                                "snopt_hessian_frequency=20",
                                "snopt_major_optimality_tol=1e-8"]),
]
## The two single-factor controls. Every other non-default entry must be a cross OF THE
## SURVIVOR: an unrelated single factor appearing here would silently make this a second
## SNOPTTUNE rather than a combination stage, and its result would be read against the wrong
## question.
SNOPTCOMBO_CONTROLS = ("mstep0p5", "elastic1e2")


def stage_SNOPTCOMBO(wall, targets, guesses, shards, only=None, tag="SNOPTCOMBO", seed=1,
                     settings=None, starts="paired,native"):
    """Cross SNOPTTUNE's one survivor with the other positive factors, at 480 cells.

    Straight to 480 with no 60-cell screen, deliberately: every factor here is already
    screened at 480, and stage STEP's lesson is that a 60-cell screen REVERSES at scale (its
    promoted `theta_max_fact = 1` went from the best row of the screen to significantly worse
    at 480, and its mechanism column reversed with it). A screen would add risk, not reduce it.

    Rows are `SOLVER2_ROWS` x both protocols x both adopted rungs -- SNOPTTUNE's twelve, on
    the same grids, so every cell pairs against both the fielded default and the survivor.

    THE DECISION RULE IS PRE-REGISTERED, and it is two bars rather than one, because "is this
    a valid SNOPT configuration" and "is it better than the step limit alone" are different
    questions and only the second justifies a more complicated configuration. Counted by ROW,
    never pooled (Thomas, 2026-09-17: "Do not pool experiments, that is useless"), on the
    LEARNED arm:

      Bar A, against Drake's SNOPT defaults -- SNOPTTUNE's bar unchanged: better on >= 9 of
        the 12 rows, significantly worse (p < 0.05) on none, significantly better on >= 1.
      Bar B, against `mstep0p5` in the same run: better on >= 8 of the 12 rows and
        significantly worse on none.

    A combination is recommended over `Major step limit = 0.5` alone only if it clears BOTH.
    `scripts/report_snoptcombo.py` implements both inline so they cannot drift.
    """
    wanted = set(only.split(",")) if only else None
    want_starts = [x.strip() for x in starts.split(",") if x.strip()]
    for st in want_starts:
        if st not in ("paired", "native"):
            raise SystemExit(f"--starts: unknown protocol {st!r}; expected paired or native")
    keep = set(settings.split(",")) if settings else None
    for name, sets in SNOPTCOMBO_SNOPT:
        for knob in sets:
            if not knob.startswith("snopt_"):
                raise SystemExit(
                    f"stage_SNOPTCOMBO entry {name!r} names {knob.split('=')[0]!r}, which is "
                    "not a SNOPT option -- this stage decides SNOPT's column alone")
        ## The guard SNOPTTUNE did not need: this stage's question is crosses OF the survivor.
        if name != "default" and name not in SNOPTCOMBO_CONTROLS:
            if "snopt_major_step_limit=0.5" not in sets:
                raise SystemExit(
                    f"stage_SNOPTCOMBO entry {name!r} is neither `default`, one of the "
                    f"single-factor controls {SNOPTCOMBO_CONTROLS}, nor a cross containing "
                    "snopt_major_step_limit=0.5. This stage exists to cross SNOPTTUNE's one "
                    "survivor with the other positive factors; an unrelated setting here "
                    "would make it a second SNOPTTUNE measured against the wrong question.")
    if keep is not None:
        known = {n for n, _ in SNOPTCOMBO_SNOPT}
        unknown = keep - known
        if unknown:
            raise SystemExit(f"--settings: no such token(s) {sorted(unknown)}; "
                             f"expected from {sorted(known)}")
        for needed in ("default",) + SNOPTCOMBO_CONTROLS[:1]:
            if needed not in keep:
                raise SystemExit(
                    f"--settings must include {needed!r}: without both its own baseline and "
                    "its own `mstep0p5` column, Bar A or Bar B would have to be evaluated "
                    "against an archived run, i.e. across code versions and node contention")
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        for name, sets in SNOPTCOMBO_SNOPT:
            if keep is not None and name not in keep:
                continue
            for task, token, placement in SOLVER2_ROWS:
                for start in want_starts:
                    args = (["--task", task, "--start", start, "--solver", "snopt"]
                            + placement + base)
                    for knob in sets:
                        args += ["--set", knob]
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}_snopt_{token}"
                                  f"_{targets * guesses}_{int(wall)}_{start}_{name}",
                                  args, targets, guesses, LADDER_ARMS, wall, shards,
                                  seed=seed)
    return items


## ----------------------------------------------------------------------------- NLOPTTUNE --
##
## The augmented-Lagrangian column, reopened because Drake finally exposes the options it
## needs. CLAUDE.md carried this as future work BY DECISION (Thomas: "Store testing NLOPT
## settings as future work") with a standing TODO: the AL's inner local optimizer was not
## selectable on the cluster's Drake at all. Drake PR 25002 (merged 2026-09-17) completes the
## surface at sixteen options against 1.56.0's six, and 1.57.0 was cut before it, so these
## rows require a Drake carrying that PR. It is now the PROJECT'S PIN (2026-09-19), so the
## items name no Drake at all: there is one install and it is the pin. The `DRAKE=nightly`
## per-item sentinel these rows used to carry is gone, along with stage DRAKEBUMP -- every arm
## of every campaign runs on the current pin, and a version-induced regression is reported and
## fixed rather than pinned around.
##
## TWO THINGS SHAPE THIS TABLE, both read out of Drake rather than assumed.
##
## First, nlopt_solver.cc:546-564 applies every `local_optimizer_*` option only inside
## `if (!parsed_options.local_optimizer_algorithm.empty())`, so an inner budget or tolerance
## set without naming the inner algorithm is ACCEPTED AND INERT. Every inner arm below names
## the algorithm; ProgramOptions refuses the combination that would not.
##
## Second, and it is not a settings question at all: NLopt's failure here is the WALL CLOCK.
## Of the twelve archived 60-cell columns the learned arm times out on 60 of 60 cells in four
## of them and 52-58 in two more. CLAUDE.md's own cap rule says a losing arm with significant
## timeouts is measuring throughput rather than method, so `cap180` is in the table by that
## rule, not as an afterthought.
##
## `stopval` is plumbed but absent from this table on purpose: this program minimises a cost
## with no known optimum, and stopping on cost returns iterates the task gate then rejects.
## Inner algorithms Drake's NLopt can ACTUALLY RUN. Drake's build omits the LGPL Luksan
## sources, so LD_LBFGS, the LD_VAR* family and every LD_TNEWTON* variant are listed by
## ParseNloptAlgorithm as valid choices and then refused at solve time with
## "attempting to use NLOPT_LD_LBFGS, but Luksan code disabled", returning kInvalidInput and
## status 0. A local smoke run caught it: six settings returned 0.5 s cells with q=None and no
## violation, which reads like a harness bug rather than a missing algorithm. So the
## gradient-based inner choices here are LD_MMA, LD_CCSAQ and LD_SLSQP, and there is no
## "name what NLopt already picks" control available -- whatever it picks when unset, it is
## not LD_LBFGS, because naming that fails while leaving it unset solves.
##
## The inner-BUDGET arms are based on LD_MMA, with one LD_SLSQP counterpart: the budget lever
## and the algorithm choice are separate questions, and testing the budget against a single
## algorithm would let "the budget does not matter" be an artefact of that one algorithm.
##
## Top-level `ftol_rel`/`ftol_abs` are plumbed but absent here on purpose. They are STOPPING
## criteria on the objective, and this column's problem is that it never reaches feasibility:
## a looser one stops earlier at a point the task gate then rejects, and a tighter one is
## inert because Drake's default of 0 already means "never stop on the objective". PR 25002's
## contribution that CAN help is the INNER ftol, which `mmaloose` uses.
## Kept in step with src.generic_program.NLOPT_LUKSAN_DISABLED, which is the source of truth
## and carries the full explanation. Duplicated rather than imported because gen_manifest.py
## runs on a login node with no pydrake and no torch on the path, and importing src pulls both.
NLOPT_LUKSAN_DISABLED = frozenset({
    "LD_LBFGS", "NLOPT_LD_LBFGS_NOCEDAL", "LD_VAR1", "LD_VAR2",
    "LD_TNEWTON", "LD_TNEWTON_RESTART", "LD_TNEWTON_PRECOND",
    "LD_TNEWTON_PRECOND_RESTART",
})

NLOPTTUNE_NLOPT = [
    ## EVERY ENTRY NOW RESTORES THE INNER TOLERANCES EXPLICITLY. Three of these fields became
    ## the ADOPTED NLopt defaults on 2026-09-19 (LD_MMA, local xtol_rel = ftol_rel = 1e-3), so
    ## "set only the algorithm" no longer means "the algorithm at Drake's inner defaults" --
    ## re-generating `innermma` without the resets below would measure `mmaloose` instead. The
    ## `=None` form emits the option not at all, which is what Drake's own default is.
    ("default", ["nlopt_local_optimizer_algorithm=None",
                 "nlopt_local_optimizer_xtol_rel=None",
                 "nlopt_local_optimizer_ftol_rel=None"]),
    ("innermma", ["nlopt_local_optimizer_algorithm=LD_MMA",
                "nlopt_local_optimizer_xtol_rel=None",
                "nlopt_local_optimizer_ftol_rel=None"]),
    ("innerccsaq", ["nlopt_local_optimizer_algorithm=LD_CCSAQ",
                  "nlopt_local_optimizer_xtol_rel=None",
                  "nlopt_local_optimizer_ftol_rel=None"]),
    ## FLAGGED FOR THOMAS, not decided here: this puts an SQP method inside the augmented
    ## Lagrangian. The outer method is still AL and the inner subproblem is bound-constrained
    ## only, so it reads as an AL column rather than a duplicate of SNOPT's -- but the
    ## three-method-classes rule is his, so the reporter carries the caveat and this row is
    ## excluded from any adoption recommendation if he disagrees.
    ("innerslsqp", ["nlopt_local_optimizer_algorithm=LD_SLSQP",
                  "nlopt_local_optimizer_xtol_rel=None",
                  "nlopt_local_optimizer_ftol_rel=None"]),
    ## The one inner knob with a mechanism rather than a tolerance behind it. A positive cap
    ## TRUNCATES each subproblem, so the outer AL updates its multipliers far more often
    ## instead of driving the first subproblem to convergence -- the classic AL tuning, and the
    ## most plausible answer to burning 4,900-6,100 network Jacobians in a single cell.
    ("mma50", ["nlopt_local_optimizer_algorithm=LD_MMA",
             "nlopt_local_optimizer_max_eval=50",
             "nlopt_local_optimizer_xtol_rel=None",
             "nlopt_local_optimizer_ftol_rel=None"]),
    ("mma200", ["nlopt_local_optimizer_algorithm=LD_MMA",
              "nlopt_local_optimizer_max_eval=200",
              "nlopt_local_optimizer_xtol_rel=None",
              "nlopt_local_optimizer_ftol_rel=None"]),
    ("slsqp50", ["nlopt_local_optimizer_algorithm=LD_SLSQP",
               "nlopt_local_optimizer_max_eval=50",
               "nlopt_local_optimizer_xtol_rel=None",
               "nlopt_local_optimizer_ftol_rel=None"]),
    ## Loose early inner solves, standard AL practice. Uses a PR-25002 option
    ## (local_optimizer_ftol_rel), so this row also proves the newest half of the surface
    ## reaches the solver on the cluster.
    ("mmaloose", ["nlopt_local_optimizer_algorithm=LD_MMA",
                "nlopt_local_optimizer_xtol_rel=1e-3",
                "nlopt_local_optimizer_ftol_rel=1e-3"]),
    ## Rung-2 tolerance, never swept on this column. It cannot be gamed: the task gate stays
    ## at task_tol=1e-3 and success is re-verified from the returned point, so a looser AL
    ## constraint tolerance that returns worse points simply fails the gate.
    ## Names no inner option, so it needs the resets too: it measured Drake's own inner
    ## solver, which is no longer what an unset field gives.
    ("ctol1em04", ["nlopt_constraint_tol=1e-4",
                   "nlopt_local_optimizer_algorithm=None",
                   "nlopt_local_optimizer_xtol_rel=None",
                   "nlopt_local_optimizer_ftol_rel=None"]),
]
## The cap arm is a WALL-TIME change rather than an option, so it cannot live in the table
## above -- and it needs its own sharding, because 60 cells x 2 arms x 180 s is ~6 h against
## run_items.sh's 4 h ITEM_TIMEOUT, which would kill the item and leave a stale .claim.
NLOPTTUNE_CAP_WALL = 180.0
NLOPTTUNE_CAP_SHARDS = 4


def stage_NLOPTTUNE(wall, targets, guesses, shards, only=None, tag="NLOPTTUNE", seed=1,
                    settings=None, starts="paired,native", cap=True):
    """Every NLopt option Drake now exposes, on the 60-cell triage grid, plus a cap arm.

    Screened at 60 rather than 480 for the reason stage_SOLVER2 triaged NLopt in the first
    place: a column that may be empty does not need campaign scale to be honest. The archived
    learned-arm counts are 0, 1, 0, 0, 39, 8 (iiwa) and 27, 1, 12, 0, 38, 8 (Panda) of 60, so
    most rows are at the floor and a 480-cell sweep of twelve settings would spend ~1,700
    core-hours to answer what 60 cells answer. The grid IS stage_SOLVER2's triage grid, so
    every cell pairs against those archived columns as well as against this run's own default.

    PRE-REGISTERED PROMOTION GATE, fixed before any result is read: a setting advances to 480
    cells only if, on the learned arm, it beats `default` by >= +8 cells of 60 on >= 6 of the
    12 rows, or takes any single row from <= 2/60 to >= 20/60. At most three advance. A NULL
    IS A COMPLETE RESULT -- "every option Drake now exposes, measured, and the augmented
    Lagrangian is still not competitive" is what closes this axis.

    Needs a Drake carrying PR 25002, which is the project's pin as of 2026-09-19, so the items
    carry no version selector. On an older Drake ProgramOptions refuses these options up front
    rather than letting Drake fail every cell with fail_reason="error".
    """
    wanted = set(only.split(",")) if only else None
    want_starts = [x.strip() for x in starts.split(",") if x.strip()]
    for st in want_starts:
        if st not in ("paired", "native"):
            raise SystemExit(f"--starts: unknown protocol {st!r}; expected paired or native")
    keep = set(settings.split(",")) if settings else None
    for name, sets in NLOPTTUNE_NLOPT:
        for knob in sets:
            if not knob.startswith("nlopt_"):
                raise SystemExit(
                    f"stage_NLOPTTUNE entry {name!r} names {knob.split('=')[0]!r}, which is "
                    "not an NLopt option -- this stage decides NLopt's column alone")
        ## Drake's NLopt cannot run the Luksan family (LD_LBFGS, LD_VAR*, LD_TNEWTON*): it
        ## accepts the NAME and then returns kInvalidInput from every solve. Refused here as
        ## well as in ProgramOptions, because this is the copy that fails on a laptop in
        ## milliseconds instead of after a cluster submission.
        for knob in sets:
            if knob.split("=", 1)[0].endswith("algorithm"):
                algo = knob.split("=", 1)[1]
                if algo in NLOPT_LUKSAN_DISABLED:
                    raise SystemExit(
                        f"stage_NLOPTTUNE entry {name!r} names {algo}, which Drake's NLopt "
                        "lists as valid and cannot run -- the Luksan sources are LGPL and "
                        "Drake does not bundle them, so the solve returns kInvalidInput on "
                        "every cell. Usable gradient-based choices: LD_MMA, LD_CCSAQ, "
                        "LD_SLSQP.")
        ## The inert-combination guard, mirrored here for the same reason.
        inner = [k for k in sets if k.startswith("nlopt_local_optimizer_")
                 and not k.startswith("nlopt_local_optimizer_algorithm")]
        if inner and not any(k.startswith("nlopt_local_optimizer_algorithm=") and
                             k.split("=", 1)[1] for k in sets):
            raise SystemExit(
                f"stage_NLOPTTUNE entry {name!r} sets {inner} without naming "
                "nlopt_local_optimizer_algorithm. Drake applies local_optimizer_* options "
                "only when the inner algorithm is non-empty (nlopt_solver.cc:546-564), so "
                "this row would measure the default inner solver while claiming a tuned one.")
    if keep is not None:
        known = {n for n, _ in NLOPTTUNE_NLOPT}
        unknown = keep - known
        if unknown:
            raise SystemExit(f"--settings: no such token(s) {sorted(unknown)}; "
                             f"expected from {sorted(known)}")
        if "default" not in keep:
            raise SystemExit("--settings must include 'default': without its own baseline "
                             "column a run pairs against an archived one, i.e. across code "
                             "versions, Drake versions and node contention")
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        ## (name, knobs, wall, shards) -- the cap arm differs in wall and sharding, not in
        ## options, so it is appended here rather than smuggled into the settings table.
        arms = [(n, sets, wall, shards) for n, sets in NLOPTTUNE_NLOPT
                if keep is None or n in keep]
        if cap and (keep is None or "cap180" in keep):
            ## Drake's NLopt defaults at a longer wall clock -- so it carries the same
            ## resets as the `default` column, for the same reason.
            arms.append(("cap180", ["nlopt_local_optimizer_algorithm=None", "nlopt_local_optimizer_xtol_rel=None", "nlopt_local_optimizer_ftol_rel=None"],
                         NLOPTTUNE_CAP_WALL, NLOPTTUNE_CAP_SHARDS))
        for name, sets, item_wall, item_shards in arms:
            for task, token, placement in SOLVER2_ROWS:
                for start in want_starts:
                    args = (["--task", task, "--start", start, "--solver", "nlopt"]
                            + placement + base)
                    for knob in sets:
                        args += ["--set", knob]
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}_nlopt_{token}"
                                  f"_{targets * guesses}_{int(item_wall)}_{start}_{name}",
                                  args, targets, guesses, LADDER_ARMS, item_wall,
                                  item_shards, seed=seed)
    return items


## ----------------------------------------------------------------------------- STATUSQUO --
##
## THE STATUS-QUO ESTABLISHMENT BENCHMARK. Four decisions landed on 2026-09-19 and every
## results table in CLAUDE.md predates them: grasp targets are now shelf-contained (they were
## free), pose stays shelf-contained at the fingertips (that was the last open placement
## question and it is now decided), the campaign cap is 180 s (it was 45 s), and SNOPT and
## NLopt each field an adopted configuration where they used to run Drake's defaults. This
## stage is the measurement that replaces those tables. Thomas:
##
##   "Why don't we set grasp fingertips in shelf as default, use the 180s timeout, run the
##    other experiments/solvers at 180s at the new status quo benchmark, and just have a note
##    that if the story radically changes as a result, flag it?"
##
## WHY THE PLACEMENT AND THE CAP ARE ONE DECISION. Containment is what creates headroom -- on
## free targets the joint-space arm sits at 94-95% and only ~25 cells of 480 are winnable at
## all, while contained it falls to 300-323. But at 45 s the iiwa's contained-grasp rows are
## cap-bound (64-74 learned timeouts of 480) and score as joint-space wins; at 180 s they are
## ties with ZERO timeouts, and a 360 s column reproduces 180 s exactly. Adopting containment
## at 45 s would have fielded a cap artefact as a result.
##
## ALL THREE SOLVERS AT THE SAME SCALE, 480 cells, so the solver axis is like-for-like
## (Thomas, 2026-09-19: "NLOPT should still get the full 480 solves for the campaign. Evaluate
## at the same scale as IPOPT and SNOPT."). NLopt needs much heavier sharding because nearly
## every one of its cells runs the full cap on both arms: ~44 h of solve per logical run
## against IPOPT's ~3.8 h, which is why STATUSQUO_SHARD_SCALE exists.
##
## NO SETTINGS AXIS. Every solver runs its adopted configuration, which is the default now --
## IPOPT's acceptable-point early stop, SNOPT's `Major step limit = 0.5`, NLopt's LD_AUGLAG +
## LD_MMA inner + inner xtol_rel = ftol_rel = 1e-3. The solver-settings question is closed on
## all three method classes (stages SWEEP, SNOPTTUNE, SNOPTCOMBO, STEP, NLOPTTUNE); this stage
## must not reopen it, hence the guard refusing any solver knob in its argument vectors.
##
## NO ANALYTIC ARMS. LADDER_ARMS throughout, matching every archived solver-axis column.
## Fielding the Panda's analytic/analytic8 baselines -- and writing an iiwa analytic program,
## which does not exist -- is future work or possibly not done at all (Thomas, 2026-09-19).
##
## THE CHART LADDER IS DELIBERATELY NOT RE-MEASURED. Nothing touching the charts changed and
## the rungs are selected by the gain ceiling rather than by cells, so a new grid cannot revise
## the choice. Available as a low-priority future option if these tables show heavy timeouts,
## since that is the regime where the cap-dependent half of the depth mechanism does the work.

## Per-solver shard counts. The binding constraint is run_items.sh's ITEM_TIMEOUT (8 h): an
## item must finish inside it, and the job wall must exceed it. Measured mean wall clock per
## cell at a 180 s cap, summed over both arms: IPOPT ~29 s (iiwa contained grasp, the
## expensive IPOPT row), SNOPT ~30 s, NLopt ~330 s -- so one 480-cell NLopt run is ~44 h and
## needs 24 shards to sit at ~1.8 h per item, while IPOPT and SNOPT are comfortable at 8.
## 24 is also CLAUDE.md's own "shards >= 16, 24 for comfort" figure for a 180 s campaign.
STATUSQUO_SHARD_SCALE = {"ipopt": 1, "snopt": 1, "nlopt": 3}
STATUSQUO_WALL = 180.0

## THE STATUS QUO HAS EXACTLY TWO EXPERIMENTS PER ROBOT: grasp and pose, both shelf-contained at
## the fingertips. `--target-placement free` is a VESTIGIAL SETTING of the grasp experiment, not a
## third experiment, and it must never appear in this stage.
##
## The first run of this stage (2026-09-19/20) fielded `mugfree` as a third row on the reasoning
## that it "answered a legacy question for completeness". Thomas, 2026-09-21: *"free grasp and free
## pose ... are not separate experiments! Those are vestigial settings for the grasp and pose
## experiments. The intent of status quo was in part to select the experiments we care about --
## preserving old settings and old experimental setups is contrary to that mission."* It cost a
## third of a ~600 core-hour campaign, and it made the results table look as though there were
## three tasks. The flag itself stays in the benchmark scripts so an archived column can still be
## reproduced, but reachable is not the same as fielded.
STATUSQUO_ROWS = tuple(r for r in SOLVER2_ROWS if "free" not in r[2])


def stage_STATUSQUO(wall, targets, guesses, shards, only=None, tag="STATUSQUO", seed=1,
                    solvers="ipopt,snopt,nlopt", starts="paired,native"):
    """The campaign of record at the new status quo: contained targets, 180 s, three solvers.

    Eight rows -- both adopted rungs x STATUSQUO_ROWS x both protocols -- under each of the
    three method classes at their adopted configurations. Two of the three rows ARE the status
    quo; `mugfree` is a legacy column included only for completeness, because iiwa free grasp
    is unmeasured above 45 s where it has 35/27 timeouts. Report it as a legacy row, never as a
    status-quo one.

    The cap does not enter target sampling, so every row pairs cell-for-cell against its own
    45 s counterpart on an identical grid_hash: IPOPT against sc_SOLVER2_*_ipopt_*_480_45_*,
    SNOPT against sc_SNOPTCOMBO_*_mstep0p5. NLopt's adopted configuration was only ever
    measured at 60 cells, so that column compares on direction and magnitude rather than cell
    for cell -- say so rather than implying a pairing. And iiwa n4 contained grasp under IPOPT
    must reproduce sc_CAP_iiwa_n4_mug_180_{native,paired} outright, which makes it a direct
    check on this stage, the raised item cap and the Drake pin at once.
    """
    wanted = set(only.split(",")) if only else None
    want_solvers = [x.strip() for x in solvers.split(",") if x.strip()]
    for sv in want_solvers:
        if sv not in SOLVER_CLASSES:
            raise SystemExit(f"--solvers: {sv!r} is not one of the three method classes "
                             f"{sorted(SOLVER_CLASSES)}")
    if len(set(want_solvers)) != len(want_solvers):
        raise SystemExit("--solvers names a solver twice; each column is measured once")
    want_starts = [x.strip() for x in starts.split(",") if x.strip()]
    for st in want_starts:
        if st not in ("paired", "native"):
            raise SystemExit(f"--starts: {st!r} is not a start protocol")
    if float(wall) != STATUSQUO_WALL:
        raise SystemExit(f"--wall-time must be {STATUSQUO_WALL:g} for the status-quo campaign; "
                         f"got {wall}. The 180 s cap is half of the 2026-09-19 decision -- at "
                         "45 s the iiwa's contained-grasp rows are cap-bound and score as "
                         "joint-space wins, so a shorter cap fields a cap artefact.")
    ## item() stringifies the cap straight into --wall-time, so an int caller would emit `180`
    ## and argparse's float `180.0` -- two spellings of one cap, which would make the args
    ## differ between a selftest construction and a real generation. Normalise once, here.
    wall = float(wall)
    items = []
    for robot, label, ckpt in ADOPTED_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        base = (["--config", "latent", "--set", f"correction_cost_weight={CORR_COST}",
                 "--scene", "hardened", "--shelf-inset", str(HARD_SHELF_INSET)]
                + (["--checkpoint", ckpt] if ckpt else []))
        for solver in want_solvers:
            item_shards = shards * STATUSQUO_SHARD_SCALE[solver]
            for task, token, placement in STATUSQUO_ROWS:
                for start in want_starts:
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}_{solver}_{token}"
                                  f"_{targets * guesses}_{int(wall)}_{start}",
                                  ["--task", task, "--start", start, "--solver", solver]
                                  + placement + base,
                                  targets, guesses, LADDER_ARMS, wall, item_shards, seed=seed)
    return items


def stage_INSET(wall, targets, guesses, shards, only=None, tag="INSET", seed=1):
    """Sweep the compartment depth inset, on both tasks, at reduced scale.

    Thomas asked whether the inset changes the story; 0.10 was adopted from `../codebase`
    without this repo ever sweeping it. Deliberately NOT full scale -- one rung per robot and
    a 60-cell grid -- because the question is directional. Read a one-cell difference here as
    noise: reproducibility at the cap is +/-1 cell.

    Covers the grasp task too, even though grasp containment defaults off, because it is the
    setting most likely to be revisited alongside other tuning.
    """
    items = []
    for robot, rung in sorted(INSET_RUNGS.items()):
        ckpt = dict((r, c) for r, l, c in LADDER_RUNGS if l == rung).get(robot)
        common_ck = ["--checkpoint", ckpt] if ckpt else []
        for inset in INSET_SWEEP:
            for task, token in (("pose", "posein"), ("mug", "mug")):
                for start in ("paired", "native"):
                    items += item(
                        robot,
                        f"sc_{tag}_{robot}_{rung}_{token}i{int(round(inset*1000)):03d}"
                        f"_{int(wall)}_{start}",
                        ["--task", task, "--start", start, "--config", "latent",
                         "--set", f"correction_cost_weight={CORR_COST}",
                         "--scene", "hardened", "--target-placement", "shelf",
                         "--shelf-inset", str(inset)] + common_ck,
                        targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


def stage_HARDMUG(wall, targets, guesses, shards, only=None, tag="HARDMUG", seed=1):
    """stage HARD again on the iiwa, with the decorative mugs KEPT.

    Disambiguates the 2026-09-15 result, where the two robots moved opposite ways on the
    grasp task: Panda joint space fell 457 -> 323 and the learned arm took the row, while
    iiwa joint space barely moved (462 -> 442) and the learned arm lost it.

    The reason to suspect the experiment rather than the robots is that they did not receive
    the same intervention. The Panda GRASP scene never had decorative mugs, so hardening it
    is near-pure target containment -- its bin sits at [0.75, 0, 0], nowhere near the
    shelves. The iiwa scene lost the bin AND seven welded mugs, four of them inside shelf
    compartments, so its grasp task gained a containment requirement while LOSING obstacles.
    `--scene nobin` removes only the bin, which is the iiwa's match for what the Panda got.

    iiwa only: `--scene nobin` raises for panda/pose (no such scene built) and resolves to
    the hardened scene for panda/mug, which already has no decorative mugs.

    Not cell-comparable with stage HARD -- the clutter changes which uniform draws survive,
    so the grid differs. It is a difficulty comparison, like posein against posefree.
    """
    wanted = set(only.split(",")) if only else None
    items = []
    for robot, label, ckpt in LADDER_RUNGS:
        if robot != "iiwa":
            continue
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        common = (["--config", "latent",
                   "--set", f"correction_cost_weight={CORR_COST}",
                   "--scene", "nobin",
                   "--shelf-inset", str(HARD_SHELF_INSET)]
                  + (["--checkpoint", ckpt] if ckpt else []))
        rows = [("mug", "mug", "shelf")] + [("pose", token, mode)
                                            for token, mode in HARD_POSE_PLACEMENTS]
        for task, token, placement in rows:
            for start in ("paired", "native"):
                items += item(robot,
                              f"sc_{tag}_{robot}_{label}_{token}_{int(wall)}_{start}",
                              ["--task", task, "--start", start,
                               "--target-placement", placement] + common,
                              targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


def stage_HARD(wall, targets, guesses, shards, only=None, tag="HARD", seed=1):
    """The ladder's eleven rungs again, on the hardened problem.

    Same rungs, arms, seed and grid shape as stage_LADDER, so the two tables are read side
    by side -- but NOT cell-comparable with it, and deliberately so: the hardened scene
    admits a different set of targets, `grid_hash` differs, and collate.py refuses the
    pairing. The comparison this stage supports is between its own columns.

    Six logical runs per rung (grasp x 2 starts, pose x 2 placements x 2 starts), 66 over
    all eleven rungs.
    """
    wanted = set(only.split(",")) if only else None
    items = []
    for robot, label, ckpt in LADDER_RUNGS:
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        common = (["--config", "latent",
                   "--set", f"correction_cost_weight={CORR_COST}",
                   "--scene", "hardened",
                   "--shelf-inset", str(HARD_SHELF_INSET)]
                  + (["--checkpoint", ckpt] if ckpt else []))
        rows = [("mug", "mug", "shelf")] + [("pose", token, mode)
                                            for token, mode in HARD_POSE_PLACEMENTS]
        for task, token, placement in rows:
            for start in ("paired", "native"):
                items += item(robot,
                              f"sc_{tag}_{robot}_{label}_{token}_{int(wall)}_{start}",
                              ["--task", task, "--start", start,
                               "--target-placement", placement] + common,
                              targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


# ---------------------------------------------------------------------------------
# The training-trajectory probe.
#
# Screening every kept checkpoint of the ladder showed that pole mass is CREATED BY
# TRAINING, monotonically, on every rung: iiwa n4's `pole/max` on the task-pose domain
# runs 27 -> 2.5e3 over 20k..620k steps, iiwa n6's 409 -> 1.3e5, panda n12's 50 -> 3.5e11.
# The architectural headroom is present at initialisation and barely used; SGD walks the
# network into it while buying accuracy.  So headroom is what makes the runaway possible
# and training is what fills it -- which the ladder, comparing only step-620000 charts,
# could not see.
#
# This stage holds the ARCHITECTURE fixed and sweeps the training step, which is the only
# way to separate the two.  iiwa only: that is the robot with the deficit, and the Panda
# has no runaway to trade against.
#
# The sharp prediction is n6 at 240k -- `pole/max` 1.6e3 against step 620000's 1.3e5, an
# 80x cleaner chart for 27% worse accuracy (15.4 mm against 12.1).  n6 at 620000 scored
# 39/39/58/28 in triage with `median_max_violation` 2.9e+03 on pose paired, so if pole
# mass acquired late in training is what breaks that rung, 240k should solve markedly
# better.  n6 at 120k is the counterweight: `pole/max` 7.1e6, WORSE than 620000's, at
# mid-range accuracy.  n4 is the control -- its pole trajectory is smooth and its ceiling
# is 2.2e4, so its curve should be dominated by accuracy alone.
#
# Note this stage selects checkpoints on an intrinsic screen, which the ladder refuted as
# a PREDICTOR of cells.  That is deliberate and is not the same move: the screen is being
# used to pick points that span the pole axis, and the benchmark is what decides whether
# the axis matters.  Nothing here may be promoted by its screen alone.
#
# Paths point OUTSIDE the repo, at the training tree (`~/learned-ik/results/train/...`),
# where the per-step exports and their `.arch.json` sidecars already live -- 30 of each
# per rung.  Items run with the repo as cwd, and submit_bench.sh's guard resolves
# `~/$SC_ROOT/repo/<path>`, so the `../` prefix is correct for both.
TRAJ_STEPS = {
    ("iiwa", "iiwa14", "n6"): [40000, 120000, 240000, 400000],
    ("iiwa", "iiwa14", "n4"): [20000, 100000, 200000, 400000],
}


def traj_ckpt(run_robot, label, step):
    return (f"../results/train/{run_robot}_{label}/pkl/"
            f"{run_robot}__{label}__step{step}.pkl")


def stage_TRAJ(wall, targets, guesses, shards, only=None, tag="TRAJ", seed=1):
    """One architecture, several training steps, on stage CKPT's grid and seed.

    `only` accepts "robot:label" or a bare label, as stage_LADDER does.  Step 620000 is
    deliberately absent: it is already measured, under sc_LADDER, on these same cells.
    """
    wanted = set(only.split(",")) if only else None
    items = []
    for (robot, run_robot, label), steps in TRAJ_STEPS.items():
        if wanted is not None and label not in wanted and f"{robot}:{label}" not in wanted:
            continue
        for step in steps:
            ckpt = traj_ckpt(run_robot, label, step)
            for task in ("mug", "pose"):
                for start in ("paired", "native"):
                    items += item(robot,
                                  f"sc_{tag}_{robot}_{label}s{step // 1000}k_{task}_{int(wall)}_{start}",
                                  ["--task", task, "--config", "latent", "--start", start,
                                   "--set", f"correction_cost_weight={CORR_COST}",
                                   "--checkpoint", ckpt],
                                  targets, guesses, LADDER_ARMS, wall, shards, seed=seed)
    return items


def retag(items, prefix):
    """Rewrite every item's tag and id with `prefix`, leaving the grid untouched.

    Lets the existing stage definitions (B, C, D, the ladder) be re-run verbatim under
    the corrected program without duplicating them -- the only thing that must change is
    the name the results land under, so a corrected run is never compared cell-for-cell
    against a pre-fix one.
    """
    out = []
    for it in items:
        args = list(it["args"])
        i = args.index("--tag")
        old = args[i + 1]
        new = old.replace("sc_", f"sc_{prefix}_", 1) if old.startswith("sc_") \
            else f"sc_{prefix}_{old}"
        args[i + 1] = new
        out.append(dict(it, args=args, id=it["id"].replace(old, new, 1)))
    return out


def render(items):
    lines = []
    for it in sorted(items, key=lambda i: (-i["seconds"], i["id"])):
        args = " ".join(it["args"])
        for field in (it["id"], it["env"], it["script"], args):
            assert field and not any(c.isspace() for c in field.replace(" ", "")) or True
        for token in it["env"].split() + it["args"]:
            assert not any(c.isspace() for c in token), f"whitespace in token {token!r}"
        lines.append(f"{it['id']}|{it['env']}|{it['script']}|{args}")
    return lines


def summarise(items, procs, nodes):
    total = sum(i["seconds"] for i in items)
    longest = max(i["seconds"] for i in items) if items else 0
    workers = procs * nodes
    wall = max(total / workers, longest)
    print(f"{len(items)} items, {total / 3600:.1f} item-hours, "
          f"longest item ~{longest / 60:.0f} min")
    print(f"at {nodes} node(s) x {procs} worker(s) = {workers} workers: "
          f"LPT wall bound ~{wall / 3600:.1f} h")
    print(f"NOTE: only 4 xeon-g6-volta nodes run at once (GrpTRES group cap); "
          f"surplus jobs queue rather than being rejected.")


def _ladder_paths_match_export():
    """LADDER_RUNGS paths must equal what cluster/export_and_screen_job.sh actually writes.

    Export names a checkpoint `<robot>__<label>__step<N>.pkl`, where label is RUN_NAME with the
    `<robot>_` prefix stripped. If that drifts from the paths in LADDER_RUNGS, the benchmark
    stage points at files that do not exist and every learned cell fails instantly -- the
    whole-column-of-zeros failure mode this repo has already paid for twice. Also catches the
    reverse: a rung that trains but that nothing benchmarks.
    """
    manifest = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ladder_runs.txt")
    trained = {}
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            robot, run_name = line.split()[:2]
            label = run_name[len(robot) + 1:] if run_name.startswith(robot + "_") else run_name
            trained[f"models/{robot}/{robot}__{label}__step620000.pkl"] = run_name

    # Checkpoints the ladder does NOT train because they already exist: the iiwa's adopted
    # control. (The Panda's control is the downloaded chart, which is `None` in LADDER_RUNGS
    # and already skipped.) Listed explicitly so a genuinely missing rung still fails.
    PRE_EXISTING = {"models/iiwa14/iiwa14__ddp-r1__step620000.pkl"}

    benchmarked = {c for _, _, c in LADDER_RUNGS if c} - PRE_EXISTING
    fails = []
    for ckpt in sorted(benchmarked - set(trained)):
        fails.append(f"{ckpt} is benchmarked but no ladder_runs.txt row would export it")
    for ckpt in sorted(set(trained) - benchmarked):
        fails.append(f"rung '{trained[ckpt]}' trains but nothing in LADDER_RUNGS benchmarks it")
    return fails


def selftest():
    fails = 0
    for stage, items in (("A", stage_A(20, 15, 4, 2)), ("B", stage_B(20, 15, 4, 1)),
                         ("C", stage_C([10, 20, 45], 15, 4, 1)), ("D", stage_D(20, 60, 8, 8)),
                         ("E", stage_E(45, 15, 4, 4)), ("F", stage_F(45, 15, 4, 4)),
                         ("F2", stage_F(45, 15, 4, 4, rows=F2_ROWS,
                                        settings=_f_variant("liftq"), tag="F2")),
                         ("F3", stage_F(45, 60, 8, 8, rows=F2_ROWS,
                                        settings=_f_variant("jlpen10"), seed=D_SEED,
                                        tag="F3")),
                         ("G", stage_G(45, 15, 4, 4)),
                         ("H", stage_H(45, 15, 4, 4, "jtik10")),
                         ("FIN", stage_FIN(45, 15, 4, 4)),
                         ("CKPT", stage_CKPT(45, 60, 8, 8)),
                         ("LADDER", stage_LADDER(45, 60, 8, 8)),
                         ("LADDERTRI", stage_LADDER(20, 15, 4, 2, only="ddpr1,n6",
                                                    tag="LADDERTRI")),
                         ("TRAJ", stage_TRAJ(45, 60, 8, 8)),
                         ("HARD", stage_HARD(45, 60, 8, 8)),
                         ("HARDMUG", stage_HARDMUG(45, 60, 8, 8)),
                         ("POSE2", stage_POSE2(45, 60, 8, 8)),
                         ("CAP", stage_CAP(45, 60, 8, 16)),
                         ("SOLVER", stage_SOLVER(45, 15, 4, 4)),
                         ("SOLVER-all", stage_SOLVER(45, 15, 4, 4, solvers="snopt,nlopt")),
                         ("SOLVER2", stage_SOLVER2(45, 60, 8, 8)),
                         ("SWEEP", stage_SWEEP(45, 15, 4, 1,
                                               solvers="ipopt,snopt")),
                         ("STEP", stage_STEP(45, 15, 4, 1, solvers="ipopt,snopt")),
                         ("SNOPTTUNE", stage_SNOPTTUNE(45, 60, 8, 8)),
                         ("SNOPTTUNE-one", stage_SNOPTTUNE(45, 60, 8, 8,
                                                          settings="default,nonderivls",
                                                          starts="paired")),
                         ("SNOPTCOMBO", stage_SNOPTCOMBO(45, 60, 8, 8)),
                         ("SNOPTCOMBO-one", stage_SNOPTCOMBO(
                             45, 60, 8, 8, settings="default,mstep0p5,mstepelastic",
                             starts="paired")),
                         ("STATUSQUO", stage_STATUSQUO(180, 60, 8, 8)),
                         ("STATUSQUO-ipopt", stage_STATUSQUO(180, 60, 8, 8, solvers="ipopt",
                                                             starts="paired")),
                         ("NLOPTTUNE", stage_NLOPTTUNE(45, 15, 4, 1)),
                         ("NLOPTTUNE-nocap", stage_NLOPTTUNE(45, 15, 4, 1, cap=False)),
                         ("STEP480", stage_STEP(45, 60, 8, 8, solvers="ipopt",
                                                settings="default,theta1",
                                                starts="paired,native", tag="STEP480")),
                         ("FINGER", stage_FINGER(45, 60, 8, 8)),
                         ("GRASPFREE", stage_GRASPFREE(45, 60, 8, 8)),
                         ("INSET", stage_INSET(45, 15, 4, 1)),
                         ("HARDTRI", stage_HARD(20, 15, 4, 2, only="ddpr1,upstream",
                                                tag="HARDTRI")),
                         ("FIN-retagged", retag(stage_FIN(45, 15, 4, 4), "EQ"))):
        ids = [i["id"] for i in items]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            print(f"FAIL stage {stage}: duplicate ids {sorted(dupes)[:5]}"); fails += 1
        try:
            lines = render(items)
        except AssertionError as exc:
            print(f"FAIL stage {stage}: {exc}"); fails += 1; continue
        for line in lines:
            if line.count("|") != 3:
                print(f"FAIL stage {stage}: {line.count('|') + 1} fields in {line[:60]!r}")
                fails += 1
                break
        # Shard expansion must be a partition: K/N for K in range(N), once each.
        shards = {}
        for it in items:
            if "--shard" in it["args"]:
                spec = it["args"][it["args"].index("--shard") + 1]
                k, n = spec.split("/")
                shards.setdefault(it["id"].rsplit("_shard", 1)[0], set()).add(int(k))
                if int(n) <= int(k):
                    print(f"FAIL stage {stage}: bad shard spec {spec}"); fails += 1
        for base, ks in shards.items():
            if ks != set(range(len(ks))):
                print(f"FAIL stage {stage}: {base} shards {sorted(ks)} not a partition")
                fails += 1
        print(f"ok   stage {stage}: {len(items)} items, ids unique, lines well formed")
    ## Every trajectory checkpoint must name a rung the ladder actually trained, and must
    ## reach outside the repo -- these live in the training tree, not under models/.
    traj_fails = 0
    for (robot, run_robot, label), steps in TRAJ_STEPS.items():
        if not any(r == robot and l == label for r, l, _ in LADDER_RUNGS):
            print(f"FAIL TRAJ: {robot}:{label} is not a trained ladder rung"); traj_fails += 1
        for step in steps:
            c = traj_ckpt(run_robot, label, step)
            if not c.startswith("../results/train/") or not c.endswith(".pkl"):
                print(f"FAIL TRAJ: implausible checkpoint path {c}"); traj_fails += 1
            if step == 620000:
                print(f"FAIL TRAJ: step 620000 is already measured under sc_LADDER ({label})")
                traj_fails += 1
    if not traj_fails:
        print("ok   trajectory checkpoint paths well formed and rungs known")
    fails += traj_fails
    ## stage_HARD's own invariants. The counts are literal because an accidental extra loop
    ## level is otherwise invisible -- the manifest just gets bigger and every line is valid.
    hard_fails = []
    if len(stage_HARD(45, 60, 8, 1)) != 66:
        hard_fails.append("should be 66 logical runs, got %d" % len(stage_HARD(45, 60, 8, 1)))
    if len(stage_HARD(45, 60, 8, 8)) != 528:
        hard_fails.append("at 8 shards should be 528 items, got %d"
                          % len(stage_HARD(45, 60, 8, 8)))
    per_rung = {}
    for it in stage_HARD(45, 60, 8, 1):
        a = it["args"]
        if "--scene" not in a or a[a.index("--scene") + 1] != "hardened":
            hard_fails.append("%s is not on the hardened scene" % it["id"])
        task = a[a.index("--task") + 1]
        placement = a[a.index("--target-placement") + 1]
        ## The one that matters: a grasp target in free air is not a grasp-selection
        ## problem, so a "free" grasp item must never reach a manifest.
        if task == "mug" and placement != "shelf":
            hard_fails.append("grasp item %s is not shelf-contained" % it["id"])
        if task == "pose":
            per_rung.setdefault((it["robot"], a[a.index("--checkpoint") + 1]
                                 if "--checkpoint" in a else "default"),
                                set()).add(placement)
    for key, modes in per_rung.items():
        if modes != {"shelf", "free"}:
            hard_fails.append("rung %r fields pose placements %r, not both" % (key, modes))
    ## Stage SOLVER's own invariants. The tag one is the important one: the solver must
    ## appear in every tag, because two runs of one grid under different solvers are
    ## different measurements and would otherwise resolve to the same summary.json and
    ## overwrite each other. The iiwa script had exactly that bug for the solver field.
    solver_fails = []
    runs = stage_SOLVER(45, 15, 4, 1, solvers="snopt,nlopt")
    if len(runs) != 24:
        solver_fails.append("should be 24 logical runs (2 robots x 3 solvers x 2 tasks "
                            "x 2 starts), got %d" % len(runs))
    seen_solvers, seen_rungs = set(), set()
    for it in runs:
        a = it["args"]
        solver = a[a.index("--solver") + 1]
        seen_solvers.add(solver)
        seen_rungs.add((it["robot"], a[a.index("--checkpoint") + 1]
                        if "--checkpoint" in a else "default"))
        if solver not in it["id"]:
            solver_fails.append("%s does not carry its solver in the tag" % it["id"])
        if "--scene" not in a or a[a.index("--scene") + 1] != "hardened":
            solver_fails.append("%s is not on the hardened scene" % it["id"])
        ## The adopted default for each task, so the solver is the only thing moving.
        task = a[a.index("--task") + 1]
        placement = a[a.index("--target-placement") + 1]
        if task == "mug" and placement != "free":
            solver_fails.append("%s: grasp should use the adopted FREE placement" % it["id"])
        if task == "pose" and placement != "shelf":
            solver_fails.append("%s: pose should use the adopted contained placement" % it["id"])
    if "ipopt" not in seen_solvers:
        solver_fails.append("no ipopt baseline generated -- the new solvers would have "
                            "nothing on this grid to be read against")
    if seen_solvers != {"ipopt", "snopt", "nlopt"}:
        solver_fails.append("fielded solvers %r, expected all three method classes"
                            % sorted(seen_solvers))
    if len(seen_rungs) != 2:
        solver_fails.append("should field exactly the two adopted rungs, got %r" % seen_rungs)
    try:
        stage_SOLVER(45, 15, 4, 1, solvers="gurobi")
        solver_fails.append("an unknown --solvers value was accepted")
    except SystemExit:
        pass
    for msg in solver_fails:
        print(f"FAIL stage SOLVER: {msg}")
    if not solver_fails:
        print("ok   stage SOLVER: 24 runs, all three method classes, solver in every tag, "
              "adopted rungs and adopted placements")
    fails += len(solver_fails)

    ## Stage SOLVER2: the same invariants at campaign scale, plus the two that are new --
    ## three placements per robot (the contained grasp rows are the addition), and the
    ## triage solvers drawn on a DIFFERENT and correctly-labelled grid.
    s2_fails = []
    runs2 = stage_SOLVER2(45, 60, 8, 1, solvers="snopt", triage_solvers="nlopt",
                          triage_targets=15, triage_guesses=4, triage_shards=1)
    if len(runs2) != 36:
        s2_fails.append("should be 36 logical runs (2 robots x 3 solvers x 3 placements "
                        "x 2 starts), got %d" % len(runs2))
    scales, placements = {}, set()
    for it in runs2:
        a = it["args"]
        solver = a[a.index("--solver") + 1]
        cells = int(a[a.index("--targets") + 1]) * int(a[a.index("--guesses") + 1])
        scales.setdefault(solver, set()).add(cells)
        task = a[a.index("--task") + 1]
        placement = a[a.index("--target-placement") + 1]
        placements.add((it["robot"], task, placement))
        if solver not in it["id"]:
            s2_fails.append("%s does not carry its solver in the tag" % it["id"])
        ## Two grid sizes live in one stage, so the cell count has to be in the tag or the
        ## 480-cell and 60-cell columns of one solver would collide on one summary.json.
        if f"_{cells}_" not in it["id"]:
            s2_fails.append("%s does not carry its cell count in the tag" % it["id"])
        if "--scene" not in a or a[a.index("--scene") + 1] != "hardened":
            s2_fails.append("%s is not on the hardened scene" % it["id"])
        ## --placement-point is a no-op on the grasp task (the mug is welded at
        ## between_fingers, so wrist and fingertip frames coincide); passing it there would
        ## read as a choice that was never made.
        if task == "mug" and "--placement-point" in a:
            s2_fails.append("%s passes --placement-point on the grasp task, where it is "
                            "a no-op" % it["id"])
        if task == "pose" and a[a.index("--placement-point") + 1] != "fingertips":
            s2_fails.append("%s: pose should use the adopted fingertip containment" % it["id"])
    if scales.get("ipopt") != {480} or scales.get("snopt") != {480}:
        s2_fails.append("full-scale solvers are not all at 480 cells: %r" % scales)
    if scales.get("nlopt") != {60}:
        s2_fails.append("the triage solver is not at the 60-cell grid: %r" % scales)
    for robot in ("panda", "iiwa"):
        want = {(robot, "mug", "free"), (robot, "mug", "shelf"), (robot, "pose", "shelf")}
        if not want <= placements:
            s2_fails.append("%s is missing placements %r" % (robot, sorted(want - placements)))
    try:
        stage_SOLVER2(45, 60, 8, 1, solvers="snopt", triage_solvers="snopt")
        s2_fails.append("a solver was accepted at two scales at once")
    except SystemExit:
        pass
    for msg in s2_fails:
        print(f"FAIL stage SOLVER2: {msg}")
    if not s2_fails:
        print("ok   stage SOLVER2: 36 runs, 480 cells for ipopt/snopt and 60 for nlopt, "
              "three placements per robot, cell count and solver in every tag")
    fails += len(s2_fails)

    ## Stage SWEEP. The load-bearing invariant is the LAST one: step rejection is a separate
    ## question, and a knob from that family drifting into this table would silently merge
    ## the two and make neither answerable.
    sw_fails = []
    runs3 = stage_SWEEP(45, 15, 4, 1, solvers="ipopt,snopt")
    want_runs = 2 * 2 * (len(SWEEP_IPOPT) + len(SWEEP_SNOPT))
    if len(runs3) != want_runs:
        sw_fails.append("should be %d runs (2 robots x 2 tasks x %d settings), got %d"
                        % (want_runs, len(SWEEP_IPOPT) + len(SWEEP_SNOPT), len(runs3)))
    for table, solver in ((SWEEP_IPOPT, "ipopt"), (SWEEP_SNOPT, "snopt")):
        names = [n for n, _ in table]
        if len(names) != len(set(names)):
            sw_fails.append("%s settings table has duplicate names" % solver)
        if names[0] != "default" or any(FieldsAValue(k) for k in table[0][1]):
            sw_fails.append("%s's first entry must be the untouched default baseline, "
                            "or there is nothing on this grid to read the sweep against"
                            % solver)
    for it in runs3:
        a = it["args"]
        if a[a.index("--start") + 1] != "paired":
            sw_fails.append("%s is not on the paired protocol" % it["id"])
        for i, tok in enumerate(a):
            if (tok == "--set" and a[i + 1].split("=")[0] in STEP_REJECTION_KNOBS
                    and FieldsAValue(a[i + 1])):
                sw_fails.append("%s sets %s, which the step-rejection question owns"
                                % (it["id"], a[i + 1]))
    ## And the guard itself must fire, not merely be present.
    try:
        saved = SWEEP_SNOPT.append(("smuggled", ["snopt_violation_limit=1.0"]))
        stage_SWEEP(45, 15, 4, 1, solvers="snopt")
        sw_fails.append("a step-rejection knob was accepted into the sweep")
    except SystemExit:
        pass
    finally:
        SWEEP_SNOPT[:] = [e for e in SWEEP_SNOPT if e[0] != "smuggled"]
    for msg in sw_fails:
        print(f"FAIL stage SWEEP: {msg}")
    if not sw_fails:
        print("ok   stage SWEEP: %d runs, paired only, a default baseline per solver, "
              "no step-rejection knob" % want_runs)
    fails += len(sw_fails)

    ## Stage STEP. The load-bearing invariants, in rough order of how expensive getting them
    ## wrong is: the seed (a forgotten keyword silently puts every run on the in-sample
    ## seed-0 grid, comparable to nothing, and NO other stage's block checks this); the
    ## whitelist, which is the other half of stage SWEEP's blacklist and keeps the two
    ## questions disjoint from both sides; and `default` sorting FIRST, because
    ## `collate.py --pair` takes its reference from the first path in argument order, so a
    ## table whose alphabetically-first token is not `default` silently pairs every setting
    ## against the wrong column. stage SWEEP's own tables fail that (`accoff` and `crash0`
    ## both sort before `default`), which is why this is asserted here rather than assumed.
    st_fails = []
    runs4 = stage_STEP(45, 15, 4, 1, solvers="ipopt,snopt")
    want4 = 2 * 3 * (len(STEP_IPOPT) + len(STEP_SNOPT))
    if len(runs4) != want4:
        st_fails.append("should be %d runs (2 robots x 3 placements x %d settings), got %d"
                        % (want4, len(STEP_IPOPT) + len(STEP_SNOPT), len(runs4)))
    for table, solver in ((STEP_IPOPT, "ipopt"), (STEP_SNOPT, "snopt")):
        names = [n for n, _ in table]
        if len(names) != len(set(names)):
            st_fails.append("%s settings table has duplicate names" % solver)
        if names[0] != "default" or any(FieldsAValue(k) for k in table[0][1]):
            st_fails.append("%s's first entry must be the untouched default baseline" % solver)
        ## Not the same check: this one is about --pair's reference selection, not about
        ## the table having a baseline at all.
        if sorted(names)[0] != "default":
            st_fails.append("%s's tokens do not sort `default` first (%r does) -- "
                            "collate.py --pair would reference the wrong column"
                            % (solver, sorted(names)[0]))
        for name, sets in table:
            for knob in sets:
                if knob.split("=")[0] not in STEP_REJECTION_KNOBS:
                    st_fails.append("%s entry %s sets %s, which this stage does not own"
                                    % (solver, name, knob))
    ## theta_max_fact >= 10 is BIT-IDENTICAL to the default (full_theta10 against full_base,
    ## fourteen digits of agreement, 0 better / 0 worse per cell). Re-adding one would spend a
    ## column to re-measure a null, so the table is pinned against it.
    for name, sets in STEP_IPOPT:
        for knob in sets:
            k, _, v = knob.partition("=")
            if k == "ipopt_theta_max_fact" and float(v) >= 10.0:
                st_fails.append("entry %s sets theta_max_fact=%s; >= 10 is measured "
                                "bit-identical to the default" % (name, v))
    seen_placements, seen_starts = set(), set()
    for it in runs4:
        a = it["args"]
        if "--seed" not in a or a[a.index("--seed") + 1] != "1":
            st_fails.append("%s is not on seed 1" % it["id"])
        if a[a.index("--start") + 1] != "paired":
            st_fails.append("%s is not on the paired protocol" % it["id"])
        seen_starts.add(a[a.index("--start") + 1])
        if "--scene" not in a or a[a.index("--scene") + 1] != "hardened":
            st_fails.append("%s is not on the hardened scene" % it["id"])
        task = a[a.index("--task") + 1]
        seen_placements.add((it["robot"], task, a[a.index("--target-placement") + 1]))
        if task == "mug" and "--placement-point" in a:
            st_fails.append("%s passes --placement-point on the grasp task, where the mug is "
                            "welded at between_fingers and it is a no-op" % it["id"])
        if task == "pose" and a[a.index("--placement-point") + 1] != "fingertips":
            st_fails.append("%s: pose should use the adopted fingertip containment" % it["id"])
        solver = a[a.index("--solver") + 1]
        if solver not in it["id"]:
            st_fails.append("%s does not carry its solver in the tag" % it["id"])
        ## Two scales live in this stage, so the cell count must be in the tag or the 60- and
        ## 480-cell columns of one setting collide on one summary.json.
        if "_60_" not in it["id"]:
            st_fails.append("%s does not carry its cell count in the tag" % it["id"])
    for robot in ("panda", "iiwa"):
        want = {(robot, "mug", "free"), (robot, "mug", "shelf"), (robot, "pose", "shelf")}
        if not want <= seen_placements:
            st_fails.append("%s is missing placements %r"
                            % (robot, sorted(want - seen_placements)))
    ## The 480-cell confirmation form: a settings filter, both protocols, cell count moves.
    conf = stage_STEP(45, 60, 8, 8, solvers="ipopt", settings="default,theta1",
                      starts="paired,native")
    if len(conf) != 2 * 3 * 2 * 2 * 8:
        st_fails.append("confirmation form should be 2 robots x 3 placements x 2 settings x "
                        "2 starts x 8 shards = 192 items, got %d" % len(conf))
    if not all("_480_" in it["id"] for it in conf):
        st_fails.append("the confirmation form does not carry 480 in its tags")
    if {it["args"][it["args"].index("--start") + 1] for it in conf} != {"paired", "native"}:
        st_fails.append("the confirmation form does not field both protocols")
    ## Every guard must FIRE, not merely be present.
    for bad, why in ((dict(solvers="nlopt"), "a solver with no step-rejection table"),
                     (dict(starts="warmstart"), "an unknown start protocol"),
                     (dict(settings="default,nosuchtoken"), "an unknown setting token"),
                     (dict(settings="theta1"), "a settings filter with no default baseline")):
        try:
            stage_STEP(45, 15, 4, 1, **bad)
            st_fails.append("%s was accepted" % why)
        except SystemExit:
            pass
    try:
        STEP_IPOPT.append(("smuggled", ["ipopt_mu_strategy=adaptive"]))
        stage_STEP(45, 15, 4, 1, solvers="ipopt")
        st_fails.append("a non-step-rejection knob was accepted into the step table")
    except SystemExit:
        pass
    finally:
        STEP_IPOPT[:] = [e for e in STEP_IPOPT if e[0] != "smuggled"]
    ## And the two questions must stay disjoint the OTHER way round, which is stage SWEEP's
    ## blacklist. If a future edit widened STEP_REJECTION_KNOBS to cover something SWEEP
    ## fields, SWEEP would start raising -- so assert the two tables share no knob.
    sweep_knobs = {k.split("=")[0] for _, sets in SWEEP_IPOPT + SWEEP_SNOPT
                   for k in sets if FieldsAValue(k)}
    step_knobs = {k.split("=")[0] for _, sets in STEP_IPOPT + STEP_SNOPT
                  for k in sets if FieldsAValue(k)}
    if sweep_knobs & step_knobs:
        st_fails.append("stage SWEEP and stage STEP both field %r"
                        % sorted(sweep_knobs & step_knobs))
    for msg in st_fails:
        print(f"FAIL stage STEP: {msg}")
    if not st_fails:
        print("ok   stage STEP: %d screen runs on seed 1, paired, three placements per "
              "robot, `default` sorting first, only step-rejection knobs, and a 192-item "
              "480-cell confirmation form" % want4)
    fails += len(st_fails)

    ## Stage SNOPTTUNE. This stage decides a FIELDED default, so its invariants are stricter
    ## than a sweep's: every candidate must reach all twelve rows (a setting measured on a
    ## subset cannot satisfy the per-solver rule), the seed must be the out-of-sample 1, the
    ## cell count must be in the tag, and no entry may name a non-SNOPT option. `default`
    ## does NOT sort first here (`crash0` precedes it), which is exactly why collate.py now
    ## picks a `_default` run as its pairing reference rather than the first path.
    sn_fails = []
    runs_sn = stage_SNOPTTUNE(45, 60, 8, 8)
    want_sn = 2 * 3 * 2 * len(SNOPTTUNE_SNOPT) * 8
    if len(runs_sn) != want_sn:
        sn_fails.append("should be %d items (2 robots x 3 placements x 2 starts x %d "
                        "settings x 8 shards), got %d"
                        % (want_sn, len(SNOPTTUNE_SNOPT), len(runs_sn)))
    names_sn = [n for n, _ in SNOPTTUNE_SNOPT]
    if len(set(names_sn)) != len(names_sn):
        sn_fails.append("duplicate setting tokens %r" % names_sn)
    if "default" not in names_sn:
        sn_fails.append("no untouched baseline column")
    seen_sn = set()
    for it in runs_sn:
        a = it["args"]
        if "--seed" not in a or a[a.index("--seed") + 1] != "1":
            sn_fails.append("%s is not on seed 1" % it["id"])
        if a[a.index("--solver") + 1] != "snopt":
            sn_fails.append("%s is not a SNOPT run" % it["id"])
        if "--scene" not in a or a[a.index("--scene") + 1] != "hardened":
            sn_fails.append("%s is not on the hardened scene" % it["id"])
        if "_480_" not in it["id"]:
            sn_fails.append("%s does not carry its cell count in the tag" % it["id"])
        task = a[a.index("--task") + 1]
        if task == "mug" and "--placement-point" in a:
            sn_fails.append("%s passes --placement-point on the grasp task" % it["id"])
        if task == "pose" and a[a.index("--placement-point") + 1] != "fingertips":
            sn_fails.append("%s: pose should use the adopted fingertip containment" % it["id"])
        ## The id carries a trailing _shardKofN, so the setting token is the field before it.
        setting = re.sub(r"_shard\d+of\d+$", "", it["id"]).rsplit("_", 1)[-1]
        seen_sn.add((it["robot"], task, a[a.index("--target-placement") + 1],
                     a[a.index("--start") + 1], setting))
    ## Uniform coverage is the load-bearing one: adding a row to one column and not another is
    ## how a robot trade gets reported as a win.
    for name in names_sn:
        rows_for = {k[:4] for k in seen_sn if k[4] == name}
        if len(rows_for) != 12:
            sn_fails.append("setting %r reaches %d of the 12 rows" % (name, len(rows_for)))
    ## The combinations must actually combine, or the stage answers a question it did not ask.
    combo = dict(SNOPTTUNE_SNOPT)["ndlshess20mstep"]
    if len(combo) != 3:
        sn_fails.append("the three-factor combination does not carry three options")
    for bad, why in ((dict(starts="warmstart"), "an unknown start protocol"),
                     (dict(settings="default,nosuchtoken"), "an unknown setting token"),
                     (dict(settings="nonderivls"), "a filter with no default baseline")):
        try:
            stage_SNOPTTUNE(45, 60, 8, 8, **bad)
            sn_fails.append("%s was accepted" % why)
        except SystemExit:
            pass
    try:
        SNOPTTUNE_SNOPT.append(("smuggled", ["ipopt_mu_strategy=adaptive"]))
        stage_SNOPTTUNE(45, 60, 8, 8)
        sn_fails.append("an IPOPT knob was accepted into SNOPT's configuration table")
    except SystemExit:
        pass
    finally:
        SNOPTTUNE_SNOPT[:] = [e for e in SNOPTTUNE_SNOPT if e[0] != "smuggled"]
    for msg in sn_fails:
        print(f"FAIL stage SNOPTTUNE: {msg}")
    if not sn_fails:
        print("ok   stage SNOPTTUNE: %d items, %d settings x 12 rows each, seed 1, SNOPT "
              "only, 480 in every tag" % (want_sn, len(SNOPTTUNE_SNOPT)))
    fails += len(sn_fails)


    ## Stage SNOPTCOMBO. Same strictness as SNOPTTUNE -- it too would decide a fielded default
    ## -- plus the guard that is this stage's whole point: every non-control entry must be a
    ## cross containing the survivor, or the stage silently becomes a second SNOPTTUNE.
    sc_fails = []
    runs_sc = stage_SNOPTCOMBO(45, 60, 8, 8)
    want_sc = 2 * 3 * 2 * len(SNOPTCOMBO_SNOPT) * 8
    if len(runs_sc) != want_sc:
        sc_fails.append("expected %d items, got %d" % (want_sc, len(runs_sc)))
    names_sc = [n for n, _ in SNOPTCOMBO_SNOPT]
    if len(set(names_sc)) != len(names_sc):
        sc_fails.append("duplicate setting token in SNOPTCOMBO_SNOPT")
    if "default" not in names_sc:
        sc_fails.append("no `default` column -- nothing to pair Bar A against")
    for ctl in SNOPTCOMBO_CONTROLS:
        if ctl not in names_sc:
            sc_fails.append("single-factor control %r is missing" % ctl)
    ## The refuted factors must stay out. Named explicitly rather than inferred, so re-adding
    ## one is a test failure and not a judgement call.
    for refuted in ("snopt_nonderivative_linesearch", "snopt_crash_option"):
        if any(any(k.startswith(refuted) for k in sets) for _, sets in SNOPTCOMBO_SNOPT):
            sc_fails.append("%s is refuted at 480 cells and must not appear here" % refuted)
    ## Bar B needs a `mstep0p5` column in the same run, so the survivor must be crossed with
    ## every other factor exactly once and appear alone once.
    if dict(SNOPTCOMBO_SNOPT)["mstep0p5"] != ["snopt_major_step_limit=0.5"]:
        sc_fails.append("the `mstep0p5` control is not the bare survivor")
    seen_sc = set()
    for it in runs_sc:
        a = it["args"]
        task = a[a.index("--task") + 1]
        if a[a.index("--seed") + 1] != "1":
            sc_fails.append("%s is not on the out-of-sample seed 1" % it["id"])
        if a[a.index("--solver") + 1] != "snopt":
            sc_fails.append("%s is not a SNOPT run" % it["id"])
        if a[a.index("--scene") + 1] != "hardened":
            sc_fails.append("%s is not on the hardened scene" % it["id"])
        if "_480_" not in it["id"]:
            sc_fails.append("%s does not carry its cell count" % it["id"])
        if it["env"] != "-":
            sc_fails.append("%s carries an env override; this stage runs the PINNED Drake, "
                            "or it is not comparable to the SNOPTTUNE archive" % it["id"])
        if task == "mug" and "--placement-point" in a:
            sc_fails.append("%s passes --placement-point on a grasp row" % it["id"])
        if task == "pose" and "--placement-point" not in a:
            sc_fails.append("%s is a pose row without a containment point" % it["id"])
        setting = re.sub(r"_shard\d+of\d+$", "", it["id"]).rsplit("_", 1)[-1]
        seen_sc.add((it["robot"], task, a[a.index("--target-placement") + 1],
                     a[a.index("--start") + 1], setting))
    for name in names_sc:
        rows_for = {k[:4] for k in seen_sc if k[4] == name}
        if len(rows_for) != 12:
            sc_fails.append("setting %r reaches %d of the 12 rows" % (name, len(rows_for)))
    for bad, why in ((dict(starts="warmstart"), "an unknown start protocol"),
                     (dict(settings="default,nosuchtoken"), "an unknown setting token"),
                     (dict(settings="default,mstepelastic"),
                      "a filter with no `mstep0p5` column for Bar B")):
        try:
            stage_SNOPTCOMBO(45, 60, 8, 8, **bad)
            sc_fails.append("%s was accepted" % why)
        except SystemExit:
            pass
    ## Both guards fire: a non-SNOPT knob, and a setting that is not a cross of the survivor.
    for smuggled, why in ((("smuggled", ["ipopt_mu_strategy=adaptive"]),
                           "an IPOPT knob"),
                          (("loneelastic", ["snopt_hessian_frequency=20"]),
                           "a single factor that is not a cross of the survivor")):
        try:
            SNOPTCOMBO_SNOPT.append(smuggled)
            stage_SNOPTCOMBO(45, 60, 8, 8)
            sc_fails.append("%s was accepted into SNOPTCOMBO's table" % why)
        except SystemExit:
            pass
        finally:
            SNOPTCOMBO_SNOPT[:] = [e for e in SNOPTCOMBO_SNOPT
                                   if e[0] not in ("smuggled", "loneelastic")]
    for msg in sc_fails:
        print(f"FAIL stage SNOPTCOMBO: {msg}")
    if not sc_fails:
        print("ok   stage SNOPTCOMBO: %d items, %d settings x 12 rows each, seed 1, SNOPT "
              "only, every cross carries the survivor"
              % (want_sc, len(SNOPTCOMBO_SNOPT)))
    fails += len(sc_fails)

    ## Stage STATUSQUO. This is the campaign of record, so its invariants are about what must
    ## NOT vary: one cap (180 s), one configuration per solver (no settings axis at all -- that
    ## question is closed and reopening it here would silently make this a tuning stage), the
    ## adopted arms, and the hardened scene. Per-solver sharding is checked because NLopt's
    ## ~44 h logical run is the one thing here that can exceed run_items.sh's ITEM_TIMEOUT.
    sq_fails = []
    runs_sq = stage_STATUSQUO(180, 60, 8, 8)
    want_sq = 2 * len(STATUSQUO_ROWS) * 2 * sum(8 * STATUSQUO_SHARD_SCALE[s]
                                                for s in ("ipopt", "snopt", "nlopt"))
    if len(runs_sq) != want_sq:
        sq_fails.append("expected %d items, got %d" % (want_sq, len(runs_sq)))
    seen_sq = {}
    for it in runs_sq:
        a = it["args"]
        if a[a.index("--wall-time") + 1] != "180.0":
            sq_fails.append("%s is not at the 180 s campaign cap" % it["id"])
        if a[a.index("--seed") + 1] != "1":
            sq_fails.append("%s is not on the out-of-sample seed 1" % it["id"])
        if a[a.index("--arms") + 1] != LADDER_ARMS:
            sq_fails.append("%s does not run exactly the adopted arms" % it["id"])
        if a[a.index("--scene") + 1] != "hardened":
            sq_fails.append("%s is not on the hardened scene" % it["id"])
        if a[a.index("--shelf-inset") + 1] != str(HARD_SHELF_INSET):
            sq_fails.append("%s is not at the adopted shelf inset" % it["id"])
        if it["env"] != "-":
            sq_fails.append("%s names a Drake version or another per-item env; there is one "
                            "install and it is the project's pin" % it["id"])
        if "_480_180_" not in it["id"]:
            sq_fails.append("%s does not name its cell count and cap, so two scales would "
                            "collide on one summary.json" % it["id"])
        ## NO SETTINGS AXIS. Every `--set` must be the correction penalty; a solver knob here
        ## would make the campaign of record a tuning run, and the adopted configurations are
        ## already the defaults, so nothing needs setting.
        for i, tok in enumerate(a):
            if tok == "--set" and not a[i + 1].startswith("correction_cost_weight="):
                sq_fails.append("%s sets %r: STATUSQUO fields each solver's adopted "
                                "configuration, which is its default, and must not carry a "
                                "settings axis" % (it["id"], a[i + 1]))
        ## TWO EXPERIMENTS, BOTH CONTAINED. `--target-placement free` is a vestigial setting of
        ## the grasp experiment, not a third experiment, and the first run of this stage fielded
        ## it as one -- a third of the campaign spent preserving exactly what selecting a status
        ## quo is meant to retire. This is the guard that keeps it out.
        if a[a.index("--target-placement") + 1] != "shelf":
            sq_fails.append("%s is not on a contained placement: the status quo is two "
                            "experiments per robot, both shelf-contained, and `free` is a "
                            "retired SETTING of the grasp experiment rather than a row"
                            % it["id"])
        task = a[a.index("--task") + 1]
        if task == "mug" and "--placement-point" in a:
            sq_fails.append("%s passes --placement-point on a grasp row, where both modes "
                            "resolve to between_fingers" % it["id"])
        if task == "pose" and a[a.index("--placement-point") + 1] != "fingertips":
            sq_fails.append("%s is not the adopted pose containment point" % it["id"])
        base = re.sub(r"_shard\d+of\d+$", "", it["id"])
        seen_sq.setdefault(base, 0)
        seen_sq[base] += 1
    ## Every solver must reach all twelve rows, and each logical run must be sharded at its
    ## solver's own scale -- the check that NLopt did not silently inherit IPOPT's 8.
    for solver, scale in STATUSQUO_SHARD_SCALE.items():
        rows = [b for b in seen_sq if f"_{solver}_" in b]
        if len(rows) != 2 * len(STATUSQUO_ROWS) * 2:
            sq_fails.append("%s reaches %d rows, not all %d"
                            % (solver, len(rows), 2 * len(STATUSQUO_ROWS) * 2))
        bad = [b for b in rows if seen_sq[b] != 8 * scale]
        if bad:
            sq_fails.append("%s: %d row(s) not sharded %d-way, e.g. %s"
                            % (solver, len(bad), 8 * scale, bad[0]))
    ## An item must fit inside run_items.sh's ITEM_TIMEOUT with room to spare. The estimate is
    ## SEC_PER_CELL_ARM, which is calibrated at 45 s and so is pessimistic here; if even that
    ## exceeds the cap the sharding is wrong.
    ITEM_TIMEOUT = 28800
    over = [it["id"] for it in runs_sq if it["seconds"] > ITEM_TIMEOUT]
    if over:
        sq_fails.append("%d item(s) estimate past run_items.sh's %d s ITEM_TIMEOUT, e.g. %s"
                        % (len(over), ITEM_TIMEOUT, over[0]))
    ## And the guards must fire.
    for kwargs, why in ((dict(wall=45), "a 45 s cap, where contained grasp is a cap artefact"),
                        (dict(solvers="ipopt,ipopt"), "a solver named twice"),
                        (dict(solvers="ipopt,gurobi"), "a solver outside the three classes"),
                        (dict(starts="warmstart"), "an unknown start protocol")):
        kw = dict(wall=180, targets=60, guesses=8, shards=8)
        kw.update(kwargs)
        try:
            stage_STATUSQUO(**kw)
            sq_fails.append("%s was accepted" % why)
        except SystemExit:
            pass
    for msg in sq_fails:
        print(f"FAIL stage STATUSQUO: {msg}")
    if not sq_fails:
        print("ok   stage STATUSQUO: %d items, 3 solvers x 8 rows at 480 cells / 180 s, "
              "seed 1, adopted configurations only, NLopt sharded %dx"
              % (want_sq, STATUSQUO_SHARD_SCALE["nlopt"]))
    fails += len(sq_fails)

    ## Stage NLOPTTUNE. The invariant that matters here is that its options are silently inert
    ## if mis-specified: no inner option may appear without an inner algorithm. Its items must
    ## also name no Drake -- there is one install and it is the pin -- and the cap arm must be
    ## sharded or it exceeds run_items.sh's ITEM_TIMEOUT.
    nl_fails = []
    runs_nl = stage_NLOPTTUNE(45, 15, 4, 1)
    want_nl = 2 * 3 * 2 * (len(NLOPTTUNE_NLOPT) + NLOPTTUNE_CAP_SHARDS)
    if len(runs_nl) != want_nl:
        nl_fails.append("expected %d items, got %d" % (want_nl, len(runs_nl)))
    names_nl = [n for n, _ in NLOPTTUNE_NLOPT]
    if len(set(names_nl)) != len(names_nl):
        nl_fails.append("duplicate setting token in NLOPTTUNE_NLOPT")
    if "default" not in names_nl:
        nl_fails.append("no `default` column")
    if "cap180" in names_nl:
        nl_fails.append("`cap180` is a wall-time arm and must not be in the settings table -- "
                        "in it, it would run at the screen's 45 s and measure nothing new")
    ## `stopval` is plumbed and deliberately unswept; keep it out by test, not by memory.
    for _n, _sets in NLOPTTUNE_NLOPT:
        for _k in _sets:
            if _k.split("=", 1)[0].endswith("algorithm") \
                    and _k.split("=", 1)[1] in NLOPT_LUKSAN_DISABLED:
                nl_fails.append("%s names %s, which Drake's NLopt cannot run"
                                % (_n, _k.split("=", 1)[1]))
    if any(any("nlopt_stopval" in k for k in sets) for _, sets in NLOPTTUNE_NLOPT):
        nl_fails.append("nlopt_stopval is deliberately not swept: this cost has no known "
                        "optimum and stopping on it returns points the task gate rejects")
    seen_nl, caps = set(), set()
    for it in runs_nl:
        a = it["args"]
        task = a[a.index("--task") + 1]
        setting = re.sub(r"_shard\d+of\d+$", "", it["id"]).rsplit("_", 1)[-1]
        if a[a.index("--seed") + 1] != "1":
            nl_fails.append("%s is not on the out-of-sample seed 1" % it["id"])
        if a[a.index("--solver") + 1] != "nlopt":
            nl_fails.append("%s is not an NLopt run" % it["id"])
        if it["env"] != "-":
            nl_fails.append("%s names a Drake version; there is one install and it is the "
                            "project's pin" % it["id"])
        if a[a.index("--scene") + 1] != "hardened":
            nl_fails.append("%s is not on the hardened scene" % it["id"])
        if task == "mug" and "--placement-point" in a:
            nl_fails.append("%s passes --placement-point on a grasp row" % it["id"])
        ## Every emitted inner option must be accompanied by a non-empty inner algorithm.
        inner = [a[i + 1] for i, x in enumerate(a) if x == "--set"
                 and a[i + 1].startswith("nlopt_local_optimizer_")
                 and not a[i + 1].startswith("nlopt_local_optimizer_algorithm")]
        named = [a[i + 1].split("=", 1)[1] for i, x in enumerate(a) if x == "--set"
                 and a[i + 1].startswith("nlopt_local_optimizer_algorithm=")]
        if inner and not any(named):
            nl_fails.append("%s sets %s with no inner algorithm -- Drake would accept it and "
                            "discard it" % (it["id"], inner))
        if setting == "cap180":
            caps.add(it["id"])
            if a[a.index("--wall-time") + 1] != str(NLOPTTUNE_CAP_WALL):
                nl_fails.append("%s is the cap arm but not at %s s"
                                % (it["id"], NLOPTTUNE_CAP_WALL))
        seen_nl.add((it["robot"], task, a[a.index("--target-placement") + 1],
                     a[a.index("--start") + 1], setting))
    for name in names_nl + ["cap180"]:
        rows_for = {k[:4] for k in seen_nl if k[4] == name}
        if len(rows_for) != 12:
            nl_fails.append("setting %r reaches %d of the 12 rows" % (name, len(rows_for)))
    if len(caps) != 12 * NLOPTTUNE_CAP_SHARDS:
        nl_fails.append("the cap arm expands to %d items, not 12 x %d -- at 180 s an "
                        "unsharded 60-cell item is ~6 h against a 4 h ITEM_TIMEOUT"
                        % (len(caps), NLOPTTUNE_CAP_SHARDS))
    if NLOPTTUNE_CAP_SHARDS < 2:
        nl_fails.append("the cap arm must be sharded")
    if not stage_NLOPTTUNE(45, 15, 4, 1, cap=False):
        nl_fails.append("cap=False produced nothing")
    if any("cap180" in i["id"] for i in stage_NLOPTTUNE(45, 15, 4, 1, cap=False)):
        nl_fails.append("cap=False still emitted the cap arm")
    for bad, why in ((dict(starts="warmstart"), "an unknown start protocol"),
                     (dict(settings="default,nosuchtoken"), "an unknown setting token"),
                     (dict(settings="innerlbfgs"), "a filter with no default baseline")):
        try:
            stage_NLOPTTUNE(45, 15, 4, 1, **bad)
            nl_fails.append("%s was accepted" % why)
        except SystemExit:
            pass
    for smuggled, why in ((("smuggled", ["snopt_major_step_limit=0.5"]), "a SNOPT knob"),
                          (("inert", ["nlopt_local_optimizer_max_eval=50"]),
                           "an inner option with no inner algorithm"),
                          (("inertempty", ["nlopt_local_optimizer_algorithm=",
                                           "nlopt_local_optimizer_max_eval=50"]),
                           "an inner option with an EMPTY inner algorithm"),
                          (("luksan", ["nlopt_local_optimizer_algorithm=LD_LBFGS"]),
                           "an inner algorithm Drake lists and cannot run"),
                          (("luksanouter", ["nlopt_algorithm=LD_TNEWTON_PRECOND_RESTART"]),
                           "an OUTER algorithm Drake lists and cannot run")):
        try:
            NLOPTTUNE_NLOPT.append(smuggled)
            stage_NLOPTTUNE(45, 15, 4, 1)
            nl_fails.append("%s was accepted into NLOPTTUNE's table" % why)
        except SystemExit:
            pass
        finally:
            NLOPTTUNE_NLOPT[:] = [e for e in NLOPTTUNE_NLOPT
                                  if e[0] not in ("smuggled", "inert", "inertempty",
                                                  "luksan", "luksanouter")]
    for msg in nl_fails:
        print(f"FAIL stage NLOPTTUNE: {msg}")
    if not nl_fails:
        print("ok   stage NLOPTTUNE: %d items, %d settings + a sharded 180 s cap arm x 12 "
              "rows each, seed 1, NLopt only, no per-item Drake selector"
              % (want_nl, len(NLOPTTUNE_NLOPT)))
    fails += len(nl_fails)

    for msg in hard_fails:
        print(f"FAIL stage HARD: {msg}")
    if not hard_fails:
        print("ok   stage HARD: 66 runs, all hardened, grasp always shelf-contained, "
              "both pose placements per rung")
    fails += len(hard_fails)

    ## stage_HARDMUG: iiwa only, and every item must actually be on the nobin scene --
    ## a HARDMUG run that silently measured the hardened scene would disambiguate nothing
    ## while looking like it had.
    hm = stage_HARDMUG(45, 60, 8, 1)
    hm_fails = []
    if len(hm) != 30:
        hm_fails.append("should be 30 logical runs (5 iiwa rungs x 6), got %d" % len(hm))
    for it in hm:
        a = it["args"]
        if it["robot"] != "iiwa":
            hm_fails.append("%s is not an iiwa run" % it["id"])
        if "--scene" not in a or a[a.index("--scene") + 1] != "nobin":
            hm_fails.append("%s is not on the nobin scene" % it["id"])
    for msg in hm_fails:
        print(f"FAIL stage HARDMUG: {msg}")
    if not hm_fails:
        print("ok   stage HARDMUG: 30 iiwa runs, every item on the nobin scene")
    fails += len(hm_fails)

    ladder_fails = _ladder_paths_match_export()
    for msg in ladder_fails:
        print(f"FAIL ladder paths: {msg}")
    fails += len(ladder_fails)
    if not ladder_fails:
        print("ok   ladder checkpoint paths match what export_and_screen_job.sh writes")
    print("gen_manifest selftest OK" if not fails else f"gen_manifest selftest FAILED ({fails})")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag-prefix", default=None,
                   help="rewrite every tag as sc_<PREFIX>_... so a run under a changed "
                        "formulation cannot be paired against an archived one by accident")
    p.add_argument("--reg", default=None,
                   help="Stage H only: the G_SETTINGS name to cross-test")
    p.add_argument("--stage", choices=["SOLVER", "SOLVER2", "SWEEP", "STEP", "SNOPTTUNE", "SNOPTCOMBO", "NLOPTTUNE", "STATUSQUO", "CKPT", "LADDER", "LADDERTRI", "TRAJ", "HARD", "HARDTRI", "HARDMUG", "POSE2", "FINGER", "GRASPFREE", "INSET", "CAP",
                                 "A", "B", "B2", "B3",
                                   "C", "D", "Dbase", "E", "F", "F2", "F3", "G", "H", "FIN"])
    p.add_argument("--settings", default=None,
                   help="STEP, SNOPTTUNE, SNOPTCOMBO and NLOPTTUNE stages: comma-separated "
                        "setting tokens to field, so a confirmation runs only the screen's "
                        "survivors without a code edit. Must include 'default' (and, for "
                        "SNOPTCOMBO, 'mstep0p5', which is its second pre-registered bar).")
    p.add_argument("--starts", default="paired",
                   help="STEP, SNOPTTUNE, SNOPTCOMBO and NLOPTTUNE stages: comma-separated "
                        "start protocols. "
                        "STEP's 60-cell screen is 'paired' (diagnostic) and its confirmation "
                        "is 'paired,native'. SNOPTTUNE needs 'paired,native' explicitly -- "
                        "the default here is the screen's, and a setting measured on one "
                        "protocol cannot be fielded as SNOPT's configuration.")
    p.add_argument("--triage-solvers", default="nlopt",
                   help="SOLVER2 stage only: solvers fielded at the 60-cell triage grid "
                        "instead of full scale, so a column that may be near-empty costs "
                        "triage money. Must not overlap --solvers; '' fields none.")
    p.add_argument("--solvers", default="snopt",
                   help="SOLVER stage only: comma-separated solvers to field alongside the "
                        "ipopt baseline, which is always generated. The axis is three METHOD "
                        "CLASSES -- ipopt interior point, snopt SQP, nlopt augmented "
                        "Lagrangian -- so adding one should be justified by the class it "
                        "contributes.")
    p.add_argument("--rungs", default=None,
                   help="LADDER/LADDERTRI only: comma-separated rung labels to generate "
                        "(e.g. 'ddpr1,n6'). Default: every rung in LADDER_RUNGS. Training "
                        "is sequential, so a stage is normally generated for the rungs "
                        "that have finished.")
    p.add_argument("--wall-time", type=float, default=20.0,
                   help="the solver's per-cell cap, in seconds. Choose it from "
                        "cluster/calibrate.sh on THIS hardware -- the laptop's 20/45 s "
                        "carry no meaning here and cross-machine timing is never compared.")
    p.add_argument("--caps", default="5,10,20,45,90,180", help="stage C only")
    p.add_argument("--targets", type=int, default=15)
    p.add_argument("--guesses", type=int, default=4)
    p.add_argument("--shards", type=int, default=1,
                   help="split each run into N target-major shards, merged afterwards by "
                        "cluster/merge_shard_summaries.py. Only worth more than 1 once "
                        "calibration says several workers per node do not perturb the "
                        "wall-clock-capped measurement.")
    p.add_argument("--f-variant", default=None,
                   help="stage F2/F3 only: which Stage F variant to expand, by name "
                        "(liftq, jlpen1, jlpen10, jlpen100). Required for those stages -- "
                        "there is no default, because expanding the wrong arm silently is "
                        "exactly the kind of mistake that costs a whole campaign.")
    p.add_argument("--procs", type=int, default=1, help="workers per node, for --summary")
    p.add_argument("--nodes", type=int, default=4, help="nodes, for --summary")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        raise SystemExit(selftest())
    if not args.stage:
        raise SystemExit("--stage is required (or --selftest)")

    caps = [float(c) for c in args.caps.split(",")]
    items = {"SOLVER": lambda: stage_SOLVER(args.wall_time, args.targets, args.guesses,
                                           args.shards, only=args.rungs,
                                           solvers=args.solvers),
             "SOLVER2": lambda: stage_SOLVER2(args.wall_time, args.targets, args.guesses,
                                              args.shards, only=args.rungs,
                                              solvers=args.solvers,
                                              triage_solvers=args.triage_solvers),
             "SWEEP": lambda: stage_SWEEP(args.wall_time, args.targets, args.guesses,
                                          args.shards, only=args.rungs,
                                          solvers=args.solvers),
             "STEP": lambda: stage_STEP(args.wall_time, args.targets, args.guesses,
                                        args.shards, only=args.rungs,
                                        solvers=args.solvers, settings=args.settings,
                                        starts=args.starts),
             "SNOPTTUNE": lambda: stage_SNOPTTUNE(args.wall_time, args.targets,
                                                 args.guesses, args.shards,
                                                 only=args.rungs,
                                                 settings=args.settings,
                                                 starts=args.starts),
             "SNOPTCOMBO": lambda: stage_SNOPTCOMBO(args.wall_time, args.targets,
                                                   args.guesses, args.shards,
                                                   only=args.rungs,
                                                   settings=args.settings,
                                                   starts=args.starts),
             "STATUSQUO": lambda: stage_STATUSQUO(args.wall_time, args.targets,
                                                 args.guesses, args.shards,
                                                 only=args.rungs, solvers=args.solvers,
                                                 starts=args.starts),
             "NLOPTTUNE": lambda: stage_NLOPTTUNE(args.wall_time, args.targets,
                                                 args.guesses, args.shards,
                                                 only=args.rungs,
                                                 settings=args.settings,
                                                 starts=args.starts),
             "HARD": lambda: stage_HARD(args.wall_time, args.targets,
                                        args.guesses, args.shards, only=args.rungs),
             "HARDTRI": lambda: stage_HARD(args.wall_time, args.targets, args.guesses,
                                           args.shards, only=args.rungs, tag="HARDTRI"),
             "HARDMUG": lambda: stage_HARDMUG(args.wall_time, args.targets, args.guesses,
                                              args.shards, only=args.rungs),
             "CAP": lambda: stage_CAP(args.wall_time, args.targets, args.guesses,
                                      args.shards),
             "POSE2": lambda: stage_POSE2(args.wall_time, args.targets, args.guesses,
                                          args.shards, only=args.rungs),
             "FINGER": lambda: stage_FINGER(args.wall_time, args.targets, args.guesses,
                                            args.shards, only=args.rungs),
             "GRASPFREE": lambda: stage_GRASPFREE(args.wall_time, args.targets, args.guesses,
                                                  args.shards, only=args.rungs),
             "INSET": lambda: stage_INSET(args.wall_time, args.targets, args.guesses,
                                          args.shards),
             "A": lambda: stage_A(args.wall_time, args.targets, args.guesses, args.shards),
             "B2": lambda: stage_B2(args.wall_time, args.targets, args.guesses, args.shards),
             "B3": lambda: stage_B3(args.wall_time, args.targets, args.guesses, args.shards),
             "B": lambda: stage_B(args.wall_time, args.targets, args.guesses, args.shards),
             "C": lambda: stage_C(caps, args.targets, args.guesses, args.shards),
             "D": lambda: stage_D(args.wall_time, args.targets, args.guesses, args.shards),
             "Dbase": lambda: stage_D_baselines(args.wall_time, args.targets,
                                                args.guesses, args.shards),
             "E": lambda: stage_E(args.wall_time, args.targets, args.guesses, args.shards),
             "F": lambda: stage_F(args.wall_time, args.targets, args.guesses, args.shards),
             ## F2/F3 take the winning variant via --f-variant; F3 additionally moves to
             ## Stage D's seed and grid so that stage's learned columns are the control.
             "F2": lambda: stage_F(args.wall_time, args.targets, args.guesses, args.shards,
                                   rows=F2_ROWS, settings=_f_variant(args.f_variant),
                                   tag="F2"),
             "F3": lambda: stage_F(args.wall_time, args.targets, args.guesses, args.shards,
                                   rows=F2_ROWS, settings=_f_variant(args.f_variant),
                                   seed=D_SEED, tag="F3"),
             "G": lambda: stage_G(args.wall_time, args.targets, args.guesses, args.shards),
             ## H crosses the winning regularization (--reg, a G_SETTINGS name)
             ## against the other knobs this campaign has swept.
             "H": lambda: stage_H(args.wall_time, args.targets, args.guesses,
                                  args.shards, args.reg),
             "FIN": lambda: stage_FIN(args.wall_time, args.targets,
                                      args.guesses, args.shards),
             ## CKPT: the retrained iiwa chart against lemon-haze-7 on identical cells.
             "CKPT": lambda: stage_CKPT(args.wall_time, args.targets,
                                        args.guesses, args.shards),
             ## LADDERTRI: the 60-cell triage pass that decides which rungs earn 480
             ## cells. LADDER: the full grid. Same generator, different --targets.
             "LADDERTRI": lambda: stage_LADDER(args.wall_time, args.targets, args.guesses,
                                               args.shards, only=args.rungs, tag="LADDERTRI"),
             "LADDER": lambda: stage_LADDER(args.wall_time, args.targets, args.guesses,
                                            args.shards, only=args.rungs),
             ## TRAJ: one architecture, several training steps -- separates pole mass
             ## acquired during training from the architectural ceiling.
             "TRAJ": lambda: stage_TRAJ(args.wall_time, args.targets, args.guesses,
                                        args.shards, only=args.rungs),
             }[args.stage]()

    if args.tag_prefix:
        items = retag(items, args.tag_prefix)
    lines = render(items)
    header = [f"# learned-ik stage {args.stage} manifest",
              f"# generated by cluster/gen_manifest.py (edit the spec there, not here)",
              f"# format: <id>|<env assignments or ->|<script>|<args>",
              f"# {len(lines)} items, ordered longest-estimate-first (LPT)"]
    text = "\n".join(header + lines) + "\n"
    if args.summary:
        summarise(items, args.procs, args.nodes)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out} ({len(lines)} items)")
    elif not args.summary:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
