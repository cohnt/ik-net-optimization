"""Per-cell comparison of the cVAE charts against the flow on the soft PCS arm, native start.

On the soft PCS arm the reported cost is 0.5 * w * |q|^2 (q_nominal is the straight rod), so a
cheaper solution is a less-strained rod. For each cVAE column and solver: the per-cell cost ratio
cVAE / flow on cells both solved (median, 95% bootstrap CI over whole targets, fraction cheaper),
the same against joint space, and the within-target spread of solutions across the 8 guesses
(median over targets of the mean distance of each guess's q from the target's mean q) -- small
spread means the chart returns nearly one configuration per target whatever the start. A soft PCS
record's `q` is the 231-entry PLANT vector (the sphere bodies' poses), not the 12 strains: distances
between two of them are zero iff the configurations agree, but their norms mean nothing.
Run from the repo root: python scripts/soft_cvae_cost_cells.py
"""
import sys
import numpy as np
sys.path.insert(0, "scripts")
from report_chart_ablation import load_columns  # noqa: E402

rng = np.random.default_rng(0)


def by_cell(recs):
    return {(r["target"], r["guess"]): r for r in recs}


def ratio_stats(A, B):
    keys = [k for k in A if k in B and A[k]["feasible"] and B[k]["feasible"]
            and A[k].get("cost") is not None and B[k].get("cost") is not None and B[k]["cost"] > 0]
    if len(keys) < 10:
        return "--"
    lr = {k: np.log(A[k]["cost"] / B[k]["cost"]) for k in keys}
    targets = sorted({k[0] for k in keys})
    per_t = {t: [lr[k] for k in keys if k[0] == t] for t in targets}
    boots = []
    for _ in range(2000):
        pick = rng.choice(targets, len(targets))
        boots.append(np.median(np.concatenate([per_t[t] for t in pick])))
    lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
    v = np.array(list(lr.values()))
    return f"{np.exp(np.median(v)):.2f} [{lo:.2f}, {hi:.2f}], cheaper {np.mean(v < 0):.0%} (n {len(v)})"


def spread(A):
    out = []
    for t in sorted({k[0] for k in A}):
        Q = [np.asarray(A[k]["q"]) for k in A if k[0] == t and A[k]["feasible"] and A[k]["q"] is not None]
        if len(Q) >= 4:
            Q = np.array(Q)
            out.append(np.mean(np.linalg.norm(Q - Q.mean(0), axis=1)))
    return np.median(out) if out else float("nan")


cols_all = load_columns()
for (solver, robot, row, start), cols in sorted(cols_all.items()):
    if robot != "soft12" or start != "native":
        continue
    F = by_cell(cols["flow"])
    J = by_cell(cols["JS"]) if cols.get("JS") else None
    print(f"\n== {solver} {row} native: flow within-target spread {spread(F):.3f}"
          + (f"; JS spread {spread(J):.3f}" if J else ""))
    for c, recs in cols.items():
        if c in ("flow", "JS") or recs is None:
            continue
        A = by_cell(recs)
        print(f"  {c:8s} spread {spread(A):.3f} | v flow {ratio_stats(A, F)}"
              + (f" | v JS {ratio_stats(A, J)}" if J else ""))
