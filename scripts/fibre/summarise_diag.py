"""Summarise fibre_diag.py output: <dir>/<robot>.s<seed>.json."""
import glob
import json
import sys

import numpy as np

D = sys.argv[1]
for r in ("soft12", "soft16", "gvs_pushrod9_o1", "gvs_pushrod9_o2", "iiwa14", "panda"):
    T = []
    for f in sorted(glob.glob(f"{D}/{r}.s*.json")):
        T += json.load(open(f))["targets"]
    if not T:
        continue
    g = np.array([t["greedy_box"] for t in T])
    c = np.array([t["plus_chord_box"] for t in T])
    nb = np.array([t["nobox"] for t in T])
    print(f"\n== {r}: {len(T)} targets")
    for name, x in (("greedy, box", g), ("+ chord, box", c), ("both, NO box", nb)):
        vals, cnt = np.unique(x, return_counts=True)
        hist = ", ".join(f"{v}:{k}" for v, k in zip(vals[:8], cnt[:8])) + (" ..." if len(vals) > 8 else "")
        print(f"  {name:13s} components median {np.median(x):.0f}, one component {(x == 1).sum()}/{len(x)}; "
              f"histogram {hist}")
    multi = g > 1
    print(f"  of {multi.sum()} multi-component targets: chord joins all {((c == 1) & multi).sum()}, "
          f"box removal joins all {((nb == 1) & multi).sum()}; total joins by chord {int((g - c).sum())}, "
          f"by box removal {int((c - nb).sum())}, left {int((nb - 1).sum())}")
    print(f"  coords at a limit per solution, median over targets {np.median([t['at_limit_coords_median'] for t in T]):.0f}")
    if "distinct_labels" in T[0]:
        lab = np.array([t["distinct_labels"] for t in T])
        print(f"  distinct GC labels median {np.median(lab):.0f}; no-box components minus labels: median "
              f"{np.median(nb - lab):.0f}, = 0 on {(nb == lab).sum()}/{len(T)}; impure (greedy/chord/nobox) "
              f"{sum(t['impure_greedy_box'] for t in T)}/{sum(t['impure_plus_chord_box'] for t in T)}/"
              f"{sum(t['impure_nobox'] for t in T)}")
    if "twist" in T[0]:
        # the NO-box components: does total torsion separate them, by about 2 pi?
        sep, gaps, spans = [], [], []
        for t in T:
            tw = np.array(t["twist"]); L = np.array(t["labels"]["nobox"])
            roots = sorted(set(L.tolist()))
            if len(roots) != 2:
                continue
            a, b = tw[L == roots[0]], tw[L == roots[1]]
            sep.append(a.max() < b.min() or b.max() < a.min())
            gaps.append(abs(np.median(a) - np.median(b)))
            spans.append(max(np.ptp(a), np.ptp(b)))
        if sep:
            print(f"  no-box 2-component targets {len(sep)}: total twist SEPARATES the two on {sum(sep)}/{len(sep)}; "
                  f"|median twist difference| median {np.median(gaps):.2f} rad (2 pi = 6.28), p10 "
                  f"{np.percentile(gaps, 10):.2f}, p90 {np.percentile(gaps, 90):.2f}; within-component twist span "
                  f"median {np.median(spans):.2f} rad")
        allt = np.concatenate([np.array(t["twist"]) for t in T])
        print(f"  total twist over all solutions: range {allt.min():.2f} .. {allt.max():.2f} rad (box allows +-6.80)")
