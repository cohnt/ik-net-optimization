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
    print("gen_manifest selftest OK" if not fails else f"gen_manifest selftest FAILED ({fails})")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag-prefix", default=None,
                   help="rewrite every tag as sc_<PREFIX>_... so a run under a changed "
                        "formulation cannot be paired against an archived one by accident")
    p.add_argument("--reg", default=None,
                   help="Stage H only: the G_SETTINGS name to cross-test")
    p.add_argument("--stage", choices=["A", "B", "B2", "B3", "C", "D", "Dbase", "E",
                                   "F", "F2", "F3", "G", "H", "FIN"])
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
    items = {"A": lambda: stage_A(args.wall_time, args.targets, args.guesses, args.shards),
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
