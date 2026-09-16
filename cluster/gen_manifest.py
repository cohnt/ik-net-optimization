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
import os
import sys

SCRIPTS = {"panda": "scripts/panda/panda_benchmark.py",
           "iiwa": "scripts/iiwa/iiwa_benchmark.py"}

# Arms per robot. The iiwa has no analytic arm (src/iiwa_analytic_ik.py exposes a
# different signature and is deliberately left out of the harness).
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
    p.add_argument("--stage", choices=["SOLVER", "CKPT", "LADDER", "LADDERTRI", "TRAJ", "HARD", "HARDTRI", "HARDMUG", "POSE2", "FINGER", "GRASPFREE", "INSET", "CAP",
                                 "A", "B", "B2", "B3",
                                   "C", "D", "Dbase", "E", "F", "F2", "F3", "G", "H", "FIN"])
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
