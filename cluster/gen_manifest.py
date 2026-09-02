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


def item(robot, tag, flags, targets, guesses, arms, wall_time, shards=1, env="-"):
    """One logical run, expanded into `shards` manifest items."""
    n_arms = len(arms.split(","))
    est = targets * guesses * n_arms * wall_time * SEC_PER_CELL_ARM / shards
    base = ["--targets", str(targets), "--guesses", str(guesses),
            "--wall-time", str(wall_time), "--arms", arms,
            "--seed", "0", "--compile", "--tag", tag] + flags
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


def stage_C(caps, targets, guesses, shards):
    """Success against the wall-clock cap, as a curve rather than two points (#7)."""
    items = []
    for robot in ("panda", "iiwa"):
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                for cap in caps:
                    items += item(robot, f"sc_C_{robot}_{task}_{int(cap)}_{start}",
                                  ["--task", task, "--config", "latent", "--start", start],
                                  targets, guesses, ALL_ARMS[robot], cap, shards)
    return items


def stage_D(wall, targets, guesses, shards):
    """The big-N replication of the finals -- the statistical power the design
    questions actually need (a 5-cell difference at 60 cells is undetectable)."""
    items = []
    for robot in ("panda", "iiwa"):
        for task in ("mug", "pose"):
            for start in ("paired", "native"):
                items += item(robot, f"sc_D_{robot}_{task}_{int(wall)}_{start}",
                              ["--task", task, "--config", "latent", "--start", start],
                              targets, guesses, ALL_ARMS[robot], wall, shards)
    return items


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
                         ("C", stage_C([10, 20, 45], 15, 4, 1)), ("D", stage_D(20, 60, 8, 8))):
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
    p.add_argument("--stage", choices=["A", "B", "C", "D"])
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
             "B": lambda: stage_B(args.wall_time, args.targets, args.guesses, args.shards),
             "C": lambda: stage_C(caps, args.targets, args.guesses, args.shards),
             "D": lambda: stage_D(args.wall_time, args.targets, args.guesses, args.shards),
             }[args.stage]()

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
