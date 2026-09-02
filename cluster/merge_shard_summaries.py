#!/usr/bin/env python3
"""Pool the `--shard K/N` summaries of one run and recompute the summary over the grid.

============================ STANDING REMINDER ============================
If you (human or agent) are even REMOTELY unsure about a SuperCloud action
-- a command's side effects, a submission's size, a policy question -- STOP
and ask Thomas before running it.  Cluster interaction goes through the
committed scripts in cluster/, not ad-hoc ssh command strings.
Be cautious, be polite, run as few commands as possible.
===========================================================================

This is the analogue of ../codebase/cluster/merge_cluster_results.py, with one
material difference: that merger *stitches* per-shard arrays, and this one must
**recompute**. Every aggregate `src/benchmark.summarise` produces is shard-local --
`success_ci` bootstraps over whole targets, `solved_within_k` counts restarts within a
target, `_mcnemar` pairs cells, `_common_cells` intersects the arms' success sets, and
every mean and median is over the shard's records. Concatenating those numbers would be
meaningless; concatenating the *records* and calling `summarise` on them is exact, and
gives a merged summary indistinguishable from an unsharded run's.

Usage:
    python cluster/merge_shard_summaries.py results/panda/benchmark
    python cluster/merge_shard_summaries.py <staging-dir> --dry-run
    python cluster/merge_shard_summaries.py <staging-dir> --out results/panda/benchmark
"""
import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import benchmark as bm

SHARD_RE = re.compile(r"_shard(\d+)of(\d+)$")

# Metadata that must agree across the shards of one run. `cells` and `shard` are expected
# to differ (that is what a shard is), and `compile_seconds` is a per-process timing.
# Everything else describes the experiment, so a mismatch means these are not shards of
# one run and merging them would fabricate a table that was never measured.
MUST_MATCH = ("robot", "task", "solver", "config", "wall_time", "seed", "grid_hash",
              "compiled", "overrides", "start", "guess_filter", "n_targets", "n_guesses",
              "device", "torch_version", "host")


def find_shard_groups(root):
    """{base_tag: {shard_index: (path, payload)}} over every summary.json under `root`."""
    groups, unsharded = {}, []
    for dirpath, _, filenames in os.walk(root):
        if "summary.json" not in filenames:
            continue
        path = os.path.join(dirpath, "summary.json")
        tag = os.path.basename(dirpath)
        m = SHARD_RE.search(tag)
        with open(path) as f:
            payload = json.load(f)
        if not m:
            unsharded.append((tag, path, payload))
            continue
        index, count = int(m.group(1)), int(m.group(2))
        base = tag[: m.start()]
        groups.setdefault(base, {"count": count, "shards": {}})
        if groups[base]["count"] != count:
            raise SystemExit(f"{base}: shards disagree on N "
                             f"({groups[base]['count']} vs {count})")
        if index in groups[base]["shards"]:
            raise SystemExit(f"{base}: duplicate shard {index}")
        groups[base]["shards"][index] = (path, payload)
    return groups, unsharded


def validate(base, group):
    """Refuse anything that is not a complete set of shards of one identical run.

    Returns (records, metadata, arm_names, n_targets, n_guesses) or raises SystemExit.
    """
    count, shards = group["count"], group["shards"]
    missing = [i for i in range(count) if i not in shards]
    if missing:
        raise SystemExit(f"{base}: INCOMPLETE -- missing shards {missing} of {count}. "
                         f"Re-run those shards (the manifest is idempotent) and merge again.")

    metas = {i: p.get("metadata", {}) for i, (_, p) in shards.items()}
    ref = metas[0]
    for key in MUST_MATCH:
        # Tolerate absent-vs-present rather than demanding a frozen schema: a shard
        # produced before a metadata key existed should still merge, and the sibling
        # campaign lost a merge to exactly that strictness.
        seen = {json.dumps(m[key], sort_keys=True) for m in metas.values() if key in m}
        if len(seen) > 1:
            raise SystemExit(f"{base}: shards disagree on metadata[{key!r}]: {sorted(seen)}")

    # Order, not just membership: `summarise` emits `_mcnemar` keys and pair
    # directions in the order the arms are given, so re-sorting here would flip the
    # merged table relative to an unsharded run's. JSON preserves insertion order,
    # so the shards' own order is the run's order.
    arm_names = None
    for i, (_, payload) in sorted(shards.items()):
        names = tuple(k for k in payload["summary"] if not k.startswith("_"))
        if arm_names is None:
            arm_names = names
        elif names != arm_names:
            raise SystemExit(f"{base}: shard {i} has arms {names}, shard 0 has {arm_names}")

    n_targets = ref.get("n_targets") or payload["n_targets"]
    n_guesses = ref.get("n_guesses") or payload["n_guesses"]

    # Coverage: the union of the shards' cells must be exactly the grid, with no cell
    # solved twice. A duplicate would double-count in every aggregate.
    covered, duplicates = set(), set()
    for i, (_, payload) in sorted(shards.items()):
        cells = payload.get("metadata", {}).get("cells")
        if cells is None:
            raise SystemExit(f"{base}: shard {i} records no metadata['cells']; "
                             f"cannot verify coverage")
        for ti, gi in cells:
            if (ti, gi) in covered:
                duplicates.add((ti, gi))
            covered.add((ti, gi))
    if duplicates:
        raise SystemExit(f"{base}: {len(duplicates)} cell(s) appear in more than one "
                         f"shard, e.g. {sorted(duplicates)[:5]}")
    expected = {(t, g) for t in range(n_targets) for g in range(n_guesses)}
    if covered != expected:
        gap = sorted(expected - covered)
        raise SystemExit(f"{base}: INCOMPLETE -- {len(gap)} cell(s) unsolved, "
                         f"e.g. {gap[:8]}. Re-run the shards that own them.")

    records = {name: [] for name in arm_names}
    for _, (_, payload) in sorted(shards.items()):
        for name in arm_names:
            records[name].extend(payload["records"][name])
    for name in arm_names:
        records[name].sort(key=lambda r: (r["target"], r["guess"]))

    # `_common_cells` aligns the arms positionally, so they must agree cell for cell.
    keys = [[(r["target"], r["guess"]) for r in records[n]] for n in arm_names]
    if any(k != keys[0] for k in keys):
        raise SystemExit(f"{base}: arms do not cover identical cells after merging")

    merged_meta = dict(ref)
    merged_meta.pop("cells", None)
    merged_meta.pop("shard", None)
    merged_meta["merged_from_shards"] = [
        os.path.basename(os.path.dirname(shards[i][0])) for i in range(count)]
    merged_meta["compile_seconds"] = [metas[i].get("compile_seconds") for i in range(count)]
    return records, merged_meta, list(arm_names), n_targets, n_guesses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="directory to walk for shard summary.json files")
    p.add_argument("--out", default=None,
                   help="where merged runs are written (default: alongside the shards, "
                        "in a sibling directory named for the base tag)")
    p.add_argument("--force", action="store_true", help="overwrite an existing merged run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default="", help="substring filter on the base tag "
                                              "(default: everything -- opt in, never out)")
    args = p.parse_args()

    groups, unsharded = find_shard_groups(args.root)
    if not groups:
        print(f"no sharded runs under {args.root}")
    for base, group in sorted(groups.items()):
        if args.only and args.only not in base:
            continue
        records, meta, arm_names, n_targets, n_guesses = validate(base, group)
        arms = [SimpleNamespace(name=n) for n in arm_names]
        n_cells = len(records[arm_names[0]])
        print(f"{base}: {group['count']} shards, {n_cells} cells x {len(arm_names)} arms")
        if args.dry_run:
            continue
        parent = args.out or os.path.dirname(os.path.dirname(group["shards"][0][0]))
        out_dir = os.path.join(parent, base)
        out_path = os.path.join(out_dir, "summary.json")
        if os.path.exists(out_path) and not args.force:
            print(f"  exists, skipping (use --force): {out_path}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        bm._write_summary(records, arms, n_targets, n_guesses, out_path, meta, partial=False)
        print(f"  wrote {out_path}")

    for tag, path, _ in unsharded:
        print(f"unsharded, left alone: {tag}")


if __name__ == "__main__":
    main()
