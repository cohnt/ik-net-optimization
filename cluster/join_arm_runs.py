#!/usr/bin/env python3
"""Join stage GVSL's learned-only runs to stage GVSJS's joint-space runs, as `sc_GVS_` rows.

Stage GVS can be run as two stages on ONE grid: the go/no-go pre-check's joint-space cells
(GVSJS, measured before any chart existed) and, once the charts exist, the learned cells
alone (GVSL). This writes each pair as the two-arm summary stage GVS would have produced,
by concatenating the RECORDS and re-running `summarise` -- exactly as
`merge_shard_summaries.py` merges shards, and for the same reason: every aggregate
(`_mcnemar`, `_common_cells`, the bootstrap CI) is computed over records, so stitching
numbers would be meaningless.

What makes a pair joinable, checked rather than assumed:
  - the same `grid_hash` -- the same targets and the same guesses, cell for cell;
  - every other `MUST_MATCH` key of the shard merger EXCEPT `start` (joint space is
    bit-identical between the two protocols -- its native start IS the random configuration
    the paired protocol hands it -- so one GVSJS run feeds both protocol rows);
  - both arms covering the identical cell set.
The checkpoint differs by construction (GVSJS loads the untrained chart joint space never
evaluates) and is recorded rather than compared.

Fielded only because the pre-check showed a joint-space solve costs the same wall time in a
joint-space-only job as beside a learned arm (`scripts/report_gvs.py precheck`, check 3):
the time-matched column compares exactly these wall times.

    python cluster/join_arm_runs.py <results root> [--dry-run] [--force]
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from merge_shard_summaries import MUST_MATCH  # noqa: E402
from src import benchmark as bm  # noqa: E402

LEARNED, NUMERICAL, OUT = "sc_GVSL_", "sc_GVSJS_", "sc_GVS_"
ARMS = ("learned", "numerical")   # the order stage GVS fields them, so McNemar directions match


def parse(tag, prefix):
    """`<prefix><robot>_<label>_<solver>_<row>_<cells>_<cap>_<start>`, read from the right."""
    p = tag[len(prefix):].split("_")
    label, solver, row, cells, cap, start = p[-6:]
    return dict(robot="_".join(p[:-6]), label=label, solver=solver, row=row, cells=cells,
                cap=cap, start=start)


def join(lpath, npath):
    with open(lpath) as f:
        L = json.load(f)
    with open(npath) as f:
        N = json.load(f)
    lm, nm = L["metadata"], N["metadata"]
    for key in MUST_MATCH:
        if key == "start":
            continue
        if key in lm and key in nm and json.dumps(lm[key], sort_keys=True) != \
                json.dumps(nm[key], sort_keys=True):
            raise SystemExit(f"metadata[{key!r}] differs: {lm[key]!r} against {nm[key]!r}")
    if set(L["records"]) != {"learned"} or set(N["records"]) != {"numerical"}:
        raise SystemExit(f"expected one learned-only and one numerical-only run, got arms "
                         f"{sorted(L['records'])} and {sorted(N['records'])}")
    records = {"learned": sorted(L["records"]["learned"], key=lambda r: (r["target"], r["guess"])),
               "numerical": sorted(N["records"]["numerical"],
                                   key=lambda r: (r["target"], r["guess"]))}
    keys = [[(r["target"], r["guess"]) for r in records[a]] for a in ARMS]
    if keys[0] != keys[1]:
        raise SystemExit("the two runs do not cover identical cells")
    meta = dict(lm)
    meta["joined_from"] = {"learned": os.path.basename(os.path.dirname(lpath)),
                           "numerical": os.path.basename(os.path.dirname(npath))}
    meta["checkpoint_numerical_run"] = nm.get("checkpoint")
    meta["hosts"] = sorted(set(lm.get("hosts", [])) | set(nm.get("hosts", [])))
    meta["arms_measured_in_separate_jobs"] = True
    return records, meta, lm["n_targets"], lm["n_guesses"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", help="a results root holding <robot>/benchmark/<tag>/summary.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    fails = 0
    for lpath in sorted(glob.glob(os.path.join(args.root, "*/benchmark", LEARNED + "*",
                                               "summary.json"))):
        ltag = os.path.basename(os.path.dirname(lpath))
        if "_shard" in ltag:
            continue
        t = parse(ltag, LEARNED)
        ntag = (f"{NUMERICAL}{t['robot']}_js_{t['solver']}_{t['row']}_{t['cells']}_{t['cap']}"
                f"_paired")
        npath = os.path.join(os.path.dirname(os.path.dirname(lpath)), ntag, "summary.json")
        otag = OUT + ltag[len(LEARNED):]
        if not os.path.exists(npath):
            print(f"{ltag}: SKIPPED -- no merged {ntag}")
            fails += 1
            continue
        try:
            records, meta, n_targets, n_guesses = join(lpath, npath)
        except SystemExit as exc:
            print(f"{ltag}: REFUSED -- {exc}")
            fails += 1
            continue
        out_path = os.path.join(os.path.dirname(os.path.dirname(lpath)), otag, "summary.json")
        print(f"{otag}: {len(records['learned'])} cells, learned from {ltag}, joint space "
              f"from {ntag}")
        if args.dry_run:
            continue
        if os.path.exists(out_path) and not args.force:
            print(f"  exists, skipping (use --force): {out_path}")
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        bm._write_summary(records, [SimpleNamespace(name=a) for a in ARMS], n_targets,
                          n_guesses, out_path, meta, partial=False)
        print(f"  wrote {out_path}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
