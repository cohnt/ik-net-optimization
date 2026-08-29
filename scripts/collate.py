"""Collate benchmark summaries into the tables the write-up needs.

    python scripts/collate.py 'results/panda/benchmark/*/summary.json'
    python scripts/collate.py --pair learned 'results/panda/benchmark/ladder3_*/summary.json'

`--pair ARM` is for the ablation ladder and the knob sweeps, where the interesting question
is whether one *run* beats another rather than whether one arm beats another. It takes the
first summary as the reference and runs an exact McNemar against each of the others on the
same (target, guess) cells -- refusing any run whose `grid_hash` differs, since comparing
runs that were not measured on the same cells is what the paired grid exists to prevent.
"""
import glob
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.benchmark import mcnemar_exact


def load(path):
    with open(path) as f:
        return json.load(f)


def pair(arm, paths):
    runs = []
    for path in paths:
        data = load(path)
        if arm not in data["summary"]:
            continue
        runs.append((os.path.basename(os.path.dirname(path)), data))
    if len(runs) < 2:
        print(f"need at least two runs carrying an arm named {arm!r}")
        return
    ref_name, ref = runs[0]
    ref_cells = {(r["target"], r["guess"]): bool(r.get("feasible"))
                 for r in ref["records"][arm]}
    print(f"paired against {ref_name} "
          f"({sum(ref_cells.values())}/{len(ref_cells)}, grid {ref['metadata'].get('grid_hash')})\n")
    header = f"{'run':<28} {'success':>9} {'better':>7} {'worse':>7} {'p':>9}  notes"
    print(header)
    print("-" * len(header))
    for name, data in runs[1:]:
        cells = {(r["target"], r["guess"]): bool(r.get("feasible"))
                 for r in data["records"][arm]}
        note = ""
        if data["metadata"].get("grid_hash") != ref["metadata"].get("grid_hash"):
            note = "DIFFERENT GRID -- not comparable"
        shared = sorted(set(cells) & set(ref_cells))
        m = mcnemar_exact([cells[c] for c in shared], [ref_cells[c] for c in shared])
        print(f"{name:<28} {sum(cells.values()):>4}/{len(cells):<4} {m['a_only']:>7} "
              f"{m['b_only']:>7} {m['p']:>9.3g}  {note}")


def main(patterns):
    arm = None
    if patterns and patterns[0] == "--pair":
        arm, patterns = patterns[1], patterns[2:]
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if arm is not None:
        return pair(arm, paths)
    rows = []
    for path in paths:
        try:
            data = load(path)
        except Exception as exc:
            print(f"skipping {path}: {exc}")
            continue
        meta = data.get("metadata", {})
        name = os.path.basename(os.path.dirname(path))
        for arm, s in data["summary"].items():
            if arm.startswith("_"):
                continue
            lo, hi = s.get("success_ci", (float("nan"),) * 2)
            rows.append(dict(
                run=name, arm=arm, config=meta.get("config", ""),
                start=meta.get("start", ""), solver=meta.get("solver", ""),
                n=s["n"], ok=s["successes"], rate=s["success_rate"], lo=lo, hi=hi,
                solver_ok=s["solver_successes"], timeouts=s["timeouts"],
                iters=s["mean_iterations"], jac=s["mean_jacobian_evals"],
                wall=s["mean_wall_time"], wall_ok=s["mean_wall_time_success"],
                cost=s["median_cost"], reasons=s["fail_reasons"],
                start_err=s.get("median_start_q_error", float("nan")),
                qc=s.get("median_correction_inf", float("nan")),
                binding=s.get("correction_binding", float("nan"))))

    header = (f"{'run':<26} {'arm':<10} {'success':>9} {'rate':>6} {'95% CI':>14} "
              f"{'t/out':>6} {'iters':>7} {'jac':>7} {'wall':>7} {'cost':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run']:<26} {r['arm']:<10} {r['ok']:>4}/{r['n']:<4} {r['rate']:>6.2f} "
              f"[{r['lo']:.2f},{r['hi']:.2f}]".ljust(len(header) - 44)
              + f"{r['timeouts']:>6} {r['iters']:>7.0f} {r['jac']:>7.0f} "
                f"{r['wall']:>7.2f} {r['cost']:>7.2f}")
    print("\nstart fidelity and correction use")
    for r in rows:
        print(f"  {r['run']:<26} {r['arm']:<10} |q(start)-q_init| {r['start_err']:>8.4f}   "
              f"|q_c| {r['qc']:>7.4f}   on the box {r['binding']:>5.2f}")
    print("\nfailure modes")
    for r in rows:
        if r["reasons"]:
            print(f"  {r['run']:<26} {r['arm']:<10} {r['reasons']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/*/benchmark/*/summary.json"])
