#!/usr/bin/env python3
"""Read stages MERGECHKGVS / MERGECHKREC: did the merged cluster tree reproduce the record?

Each sc_MERGECHK_* shard re-ran a committed stage-of-record item verbatim (cluster/gen_manifest.py,
`stage_MERGECHK`) after ~/learned-ik-gvs was folded into ~/learned-ik and the main venv gained the
JAX stack (2026-10-06). A cell that converges is exactly reproducible -- stage STEP's
same-configuration control, and stage GVS across protocols and against GVSJS -- so the test is
cell for cell against the original merged run:

  - grid_hash must match;
  - on every cell where NEITHER run hit the wall clock (`timed_out`), the verdict (`feasible`) and
    the iteration count must be IDENTICAL. Any difference there is a failure of the merge;
  - cells that hit the clock in either run are reported separately and are allowed to differ,
    since where a solve is stopped depends on node contention.

Shards are read directly (a partial shard set does not merge, by design). Originals are the merged
runs under results/ or results/_cluster_staging/*/results/.

    python scripts/check_mergechk.py [--root <checkout holding results/>]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def load_original(tag):
    for root in ("results", "results/_cluster_staging/*/results"):
        hits = sorted(glob.glob(f"{root}/*/benchmark/{tag}/summary.json"))
        if hits:
            with open(hits[-1]) as fh:
                return json.load(fh)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = p.parse_args()
    os.chdir(args.root)

    shards = {}
    for root in ("results", "results/_cluster_staging/*/results"):
        for f in glob.glob(f"{root}/*/benchmark/sc_MERGECHK_*_shard*of8/summary.json"):
            shards.setdefault(os.path.basename(os.path.dirname(f)), f)
    if not shards:
        print("no sc_MERGECHK_* shards found")
        return 1

    hdr = (f"{'shard':<66}{'arm':<10}{'cells':>6}{'same':>6}{'DIFF conv':>10}"
           f"{'diff clock':>11}{'wall new/old':>13}")
    print(hdr)
    print("-" * len(hdr))
    bad, total_conv, total_clock_diff = 0, 0, 0
    for tag in sorted(shards):
        with open(shards[tag]) as fh:
            new = json.load(fh)
        orig_tag = re.sub(r"_shard\d+of8$", "", tag.replace("sc_MERGECHK_", "sc_", 1))
        old = load_original(orig_tag)
        if old is None:
            print(f"{tag}: ORIGINAL {orig_tag} NOT FOUND")
            bad += 1
            continue
        if new["metadata"].get("grid_hash") != old["metadata"].get("grid_hash"):
            print(f"{tag}: GRID MISMATCH {new['metadata'].get('grid_hash')} vs "
                  f"{old['metadata'].get('grid_hash')}")
            bad += 1
            continue
        for arm, recs in new["records"].items():
            O = {(r["target"], r["guess"]): r for r in old["records"][arm]}
            same = conv_diff = clock_diff = 0
            ratios = []
            for r in recs:
                o = O.get((r["target"], r["guess"]))
                if o is None:
                    print(f"{tag} {arm}: cell {(r['target'], r['guess'])} absent from original")
                    bad += 1
                    continue
                identical = (bool(r["feasible"]) == bool(o["feasible"])
                             and r.get("iterations") == o.get("iterations"))
                clock = bool(r.get("timed_out")) or bool(o.get("timed_out"))
                if identical:
                    same += 1
                elif clock:
                    clock_diff += 1
                else:
                    conv_diff += 1
                    print(f"  DIFF {tag} {arm} cell {(r['target'], r['guess'])}: feasible "
                          f"{o['feasible']}->{r['feasible']}, iterations "
                          f"{o.get('iterations')}->{r.get('iterations')}, status "
                          f"{o.get('solver_status')}->{r.get('solver_status')}")
                if not clock and r.get("wall_time") and o.get("wall_time"):
                    ratios.append(r["wall_time"] / o["wall_time"])
            bad += conv_diff
            total_conv += same + conv_diff
            total_clock_diff += clock_diff
            label = tag.replace("sc_MERGECHK_", "")
            ratio = f"{np.median(ratios):.2f}" if ratios else "--"
            print(f"{label:<66}{arm:<10}{len(recs):>6}{same:>6}{conv_diff:>10}{clock_diff:>11}"
                  f"{ratio:>13}")
    print(f"\n{len(shards)} shards. Differences on cells that ran to the clock in either run: "
          f"{total_clock_diff} (allowed).")
    print("MERGE CHECK PASSED: every non-clock-bound cell reproduces exactly" if not bad
          else f"MERGE CHECK FAILED: {bad} problem(s) above")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
