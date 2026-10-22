#!/usr/bin/env python3
"""Fold stage OVR's re-run shards into the runs they replace, and re-merge those runs.

Stage OVR (cluster/gen_manifest.py, Thomas 2026-10-21) re-ran every SNOPT shard of the record --
REMEASURE (with its GVS rows), PAPER, ANALYTIC/ANALYTICT/ANALYTICC -- that held a cell past 1.5x
the clock, whole and verbatim under the overrun watcher, tagged `sc_OVR_<original minus sc_>`.
For each affected run this builds the shard set {re-run shard where one exists, else the
original shard}, validates it with the merger's own `validate` (complete grid, matching metadata),
and writes the merged run under the ORIGINAL tag in results/<robot>/benchmark/, so every reader
picks it up unchanged. Every merged copy it supersedes -- in results/ or in any staging directory
-- is MOVED, never deleted, to results/archive_preoverrun/<same relative path>. Shards are left in
place (no reader of merged runs reads them).

Every re-run shard is first checked against its original: grid hash and cell set equal, and every
cell the original ran past the threshold is `killed_overrun` in the re-run. The merged metadata
records which shards were re-run (`overrun_rerun_shards`).

    python scripts/apply_overrun_rerun.py            # dry run: what would be folded and moved
    python scripts/apply_overrun_rerun.py --apply
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from types import SimpleNamespace

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "cluster"))

from src import benchmark as bm                                     # noqa: E402
import merge_shard_summaries as merger                              # noqa: E402

ROOTS = ("results", "results/_cluster_staging/*/results")
ARCHIVE = "results/archive_preoverrun"
SHARD_RE = re.compile(r"^(.*)_shard(\d+)of(\d+)$")


def find(tag_glob):
    """{tag: [summary paths]} over every root (newest staging last; results/ last of all)."""
    out = {}
    for root in ("results/_cluster_staging/*/results", "results"):
        for f in sorted(glob.glob(f"{root}/*/benchmark/{tag_glob}/summary.json")):
            out.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    return out


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--apply", action="store_true")
    p.add_argument("--factor", type=float, default=1.5)
    args = p.parse_args()
    os.chdir(REPO)

    with open("cluster/overrun_rerun_items.txt") as fh:
        wanted = [line.strip().split("|", 1)[1] for line in fh
                  if line.strip() and not line.startswith("#")]
    reruns = find("sc_OVR_*_shard*")
    originals = find("sc_*_snopt_*_shard*")
    missing = [w for w in wanted if "sc_OVR_" + w[3:] not in reruns]
    if missing:
        print(f"{len(missing)} of {len(wanted)} re-run shard(s) not collected yet, e.g. {missing[:3]}")
        if args.apply:
            return 1

    by_base, bad = {}, 0
    for orig in wanted:
        new = "sc_OVR_" + orig[3:]
        if new not in reruns:
            continue
        o, n = load(originals[orig][-1]), load(reruns[new][-1])
        cap = float(o["metadata"].get("wall_time") or 180.0)
        killed = over = 0
        if o["metadata"].get("grid_hash") != n["metadata"].get("grid_hash") or \
                sorted(map(tuple, o["metadata"].get("cells") or [])) != \
                sorted(map(tuple, n["metadata"].get("cells") or [])):
            print(f"BAD {new}: grid or cells differ from {orig}")
            bad += 1
            continue
        for arm, recs in o["records"].items():
            N = {(r["target"], r["guess"]): r for r in n["records"][arm]}
            for r in recs:
                if (r.get("wall_time") or 0.0) > args.factor * cap:
                    over += 1
                    killed += N[(r["target"], r["guess"])].get("fail_reason") == "killed_overrun"
        if killed != over:
            print(f"BAD {new}: {killed} of the original's {over} overrun cell(s) came back killed")
            bad += 1
            continue
        base, k, count = SHARD_RE.match(orig).groups()
        by_base.setdefault(base, {"count": int(count), "rerun": {}})["rerun"][int(k)] = \
            (reruns[new][-1], n)
    if bad:
        print(f"{bad} re-run shard(s) failed their check; nothing applied")
        return 1

    for base, info in sorted(by_base.items()):
        shards = {}
        for k in range(info["count"]):
            if k in info["rerun"]:
                shards[k] = info["rerun"][k]
            else:
                tag = f"{base}_shard{k}of{info['count']}"
                if tag not in originals:
                    raise SystemExit(f"{base}: original shard {k} not found")
                shards[k] = (originals[tag][-1], load(originals[tag][-1]))
        records, meta, arm_names, n_targets, n_guesses = merger.validate(
            base, {"count": info["count"], "shards": shards})
        meta["overrun_rerun_shards"] = sorted(info["rerun"])
        robot_dir = os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(info["rerun"][min(info["rerun"])][0]))))
        target = os.path.join("results", robot_dir, "benchmark", base, "summary.json")
        superseded = [f for root in ROOTS
                      for f in glob.glob(f"{root}/*/benchmark/{base}/summary.json")]
        print(f"{base}: shards {sorted(info['rerun'])} re-run; {len(superseded)} merged "
              f"cop{'y' if len(superseded) == 1 else 'ies'} to archive -> {target}")
        if not args.apply:
            continue
        for f in superseded:
            dest = os.path.join(ARCHIVE, os.path.relpath(os.path.dirname(f), "results"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                raise SystemExit(f"{dest} already archived; refusing to overwrite it")
            shutil.move(os.path.dirname(f), dest)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        bm._write_summary(records, [SimpleNamespace(name=a) for a in arm_names], n_targets,
                          n_guesses, target, meta, partial=False)
    print(f"\n{len(by_base)} run(s) {'re-merged' if args.apply else 'would be re-merged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
