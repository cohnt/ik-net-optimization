"""Collate benchmark summaries into the tables the write-up needs.

    python scripts/collate.py results/panda/benchmark/*/summary.json
"""
import glob
import json
import os
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
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
                cost=s["median_cost"], reasons=s["fail_reasons"]))

    header = (f"{'run':<26} {'arm':<10} {'success':>9} {'rate':>6} {'95% CI':>14} "
              f"{'t/out':>6} {'iters':>7} {'jac':>7} {'wall':>7} {'cost':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run']:<26} {r['arm']:<10} {r['ok']:>4}/{r['n']:<4} {r['rate']:>6.2f} "
              f"[{r['lo']:.2f},{r['hi']:.2f}]".ljust(len(header) - 44)
              + f"{r['timeouts']:>6} {r['iters']:>7.0f} {r['jac']:>7.0f} "
                f"{r['wall']:>7.2f} {r['cost']:>7.2f}")
    print("\nfailure modes")
    for r in rows:
        if r["reasons"]:
            print(f"  {r['run']:<26} {r['arm']:<10} {r['reasons']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/*/benchmark/*/summary.json"])
