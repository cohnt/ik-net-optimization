"""Read the REPRO_SVGD diagnostic: is the svgd swarm bit reproducible across independent processes?

Each condition (A..D, docs/svgd-solver.md "Diagnostic REPRO_SVGD") ran the same record shard in eight
processes with inductor's caches disabled. A cell's fingerprint is its outer step count, selected
particle, pre-polish worst violation and returned `q`, all bitwise; a condition is reproducible iff
its eight replicas give ONE fingerprint on every cell. Also compares each replica's cells against the
R31 shard the items were copied from.

    .venv/bin/python scripts/svgd/repro_check.py [--root results]
"""
import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

CONDITIONS = {"A": ("iiwa", "learned", "R31 learned, record flags"),
              "B": ("iiwa", "learned", "A + TORCHINDUCTOR_DETERMINISTIC=1"),
              "C": ("soft12", "numerical", "R31 joint space (soft PCS FK), record flags"),
              "D": ("soft12", "numerical", "C + TORCHINDUCTOR_DETERMINISTIC=1")}


def fingerprint(r):
    s = r.get("svgd") or {}
    q = json.dumps(r.get("q"), sort_keys=True)
    key = (s.get("iterations"), s.get("selected_index"), repr(s.get("polish_viol_before")), q)
    return hashlib.sha1(repr(key).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    args = ap.parse_args()
    for c, (robot, arm, what) in CONDITIONS.items():
        runs = sorted(glob.glob(os.path.join(args.root, robot, "benchmark", f"sc_REPRO_SVGD_{c}_r*_shard0of24")))
        if not runs:
            print(f"{c}: no runs under {args.root}/{robot}/benchmark")
            continue
        per_cell = defaultdict(list)
        hosts = []
        for run in runs:
            s = json.load(open(os.path.join(run, "summary.json")))
            hosts.append(s["metadata"].get("host"))
            for i, r in enumerate(s["records"][arm]):
                per_cell[i].append(fingerprint(r))
        n_cells = len(per_cell)
        distinct = [len(set(v)) for v in per_cell.values()]
        ## replica classes: replicas that agree on EVERY cell
        sig = Counter(tuple(per_cell[i][k] for i in range(n_cells)) for k in range(len(runs)))
        tag = re.sub(r"_r\d+_", "_", os.path.basename(runs[0]))
        r31_tag = tag.replace(f"sc_REPRO_SVGD_{c}_", "sc_SVGD_R31_")
        ## shard dirs are not promoted: R31's lives in a collection's staging dir
        r31 = sorted(glob.glob(os.path.join(args.root, robot, "benchmark", r31_tag)) +
                     glob.glob(os.path.join(args.root, "_cluster_staging", "*", "results", robot, "benchmark", r31_tag)))
        vs_r31 = ""
        if r31:
            ref = [fingerprint(r) for r in json.load(open(os.path.join(r31[0], "summary.json")))["records"][arm]]
            same = sum(all(per_cell[i][k] == ref[i] for i in range(n_cells)) for k in range(len(runs)))
            vs_r31 = f"; replicas identical to R31's shard 0 on every cell: {same} / {len(runs)}"
        print(f"{c} ({what}): {len(runs)} replicas x {n_cells} cells; replica classes {sorted(sig.values(), reverse=True)}; "
              f"cells with > 1 outcome {sum(d > 1 for d in distinct)} / {n_cells} (max {max(distinct)} outcomes)"
              f"{vs_r31}; hosts {Counter(hosts)}")


if __name__ == "__main__":
    main()
