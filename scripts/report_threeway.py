"""Three-way comparison (python scripts/report_threeway.py), (learned L, joint space JS, analytic A) per experiment x solver, on
success, speed and cost; single start and multi-start (restarts over each target's 8 recorded
starts). Scratch analysis; numbers only."""
import sys, json, collections
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, ".")
from scripts.report_statusquo import load_record, load_svgd_record, by_cell, mcnemar, parse_tag
from scripts.report_analytic import load_analytic, TAG, record_key, index_record

CAP = 180.0
A_STAGE = {"ipopt": "", "snopt": "", "nlopt": "N", "svgd": "S"}   # default ladder, Python
P = 0.05


def wall(r):
    return min(r["wall_time"], CAP) if r.get("wall_time") is not None else CAP


def ok(r):
    return bool(r.get("feasible"))


def load():
    rec, _ = load_record(480, which="paper")
    rec = dict(rec, **load_svgd_record(rec, 480)[0])
    rec = index_record(rec)
    an = load_analytic(480)
    rows = []
    for tag, A in sorted(an.items()):
        m = TAG.match(tag)
        stage, ladder, kern = m.group(1), m.group(2), (m.group(4) or "")
        t = parse_tag(tag[:len(tag) - len(kern)])
        if ladder or A_STAGE[t["solver"]] != stage or (stage == "S" and kern != "_pin2kq"):
            continue
        _, R = rec[record_key(t)]
        arms = {"L": by_cell(R, "learned"), "JS": by_cell(R, "numerical"),
                "A": by_cell(A, "analytic")}
        if "analytic8" in A["records"]:
            arms["A8"] = by_cell(A, "analytic8")
        cpp = an.get(tag.replace("sc_ANALYTIC_", "sc_ANALYTICC_", 1)) if stage == "" else None
        rows.append(dict(t, arms=arms, cpp=by_cell(cpp, "analytic") if cpp else None))
    return rows


def pair_wall(X, Y):
    keys = sorted(set(X) & set(Y))
    x = np.array([wall(X[k]) for k in keys]); y = np.array([wall(Y[k]) for k in keys])
    d = x - y
    if np.all(d == 0):
        return 1.0
    return wilcoxon(x, y).pvalue


def pair_cost(X, Y, keys):
    x = np.array([X[k]["cost"] for k in keys]); y = np.array([Y[k]["cost"] for k in keys])
    if len(keys) < 10 or np.all(x == y):
        return None
    return wilcoxon(x, y).pvalue


def best(names, score, better, pval):
    """Names not significantly beaten by the top scorer (score: higher better if better=+1)."""
    order = sorted(names, key=lambda n: -better * score[n])
    top = order[0]
    win = [top] + [n for n in order[1:] if pval(top, n) >= P]
    return "=".join(win) if len(win) < len(names) else "all tie"


def single(row, names):
    a = row["arms"]
    succ = {n: sum(ok(r) for r in a[n].values()) for n in names}
    def p_succ(x, y):
        b = sum(1 for k in a[x] if ok(a[x][k]) and not ok(a[y][k]))
        c = sum(1 for k in a[x] if ok(a[y][k]) and not ok(a[x][k]))
        return mcnemar(b, c)
    mwall = {n: np.mean([wall(r) for r in a[n].values()]) for n in names}
    shared = [k for k in a["L"] if all(ok(a[n][k]) for n in names)]
    cost = {n: (np.median([a[n][k]["cost"] for k in shared]) if len(shared) >= 10 else np.nan)
            for n in names}
    def p_cost(x, y):
        p = pair_cost(a[x], a[y], shared); return 1.0 if p is None else p
    return dict(succ=succ, wall=mwall, cost=cost, nshared=len(shared),
                best_succ=best(names, succ, +1, p_succ),
                best_wall=best(names, mwall, -1, lambda x, y: pair_wall(a[x], a[y])),
                best_cost=(best(names, cost, -1, p_cost) if len(shared) >= 10 else "N/A"))


def multistart(row, names, budgets, n_perm=500, seed=0):
    """Per target: run its 8 recorded starts in random order. Success within budget B = first
    success's cumulative wall <= B. Time-to-first-success, and best (min) cost among successes
    within the budget. Averaged over random orders."""
    rng = np.random.default_rng(seed)
    a = row["arms"]
    out = {}
    targets = sorted({k[0] for k in a["L"]})
    for n in names:
        by_t = collections.defaultdict(list)
        for k, r in a[n].items():
            by_t[k[0]].append(r)
        succ_B = {B: [] for B in budgets}; ttfs = []; starved = {B: [] for B in budgets}
        bestcost = {B: {} for B in budgets}
        for t in targets:
            rs = by_t[t]
            w = np.array([wall(r) for r in rs]); s = np.array([ok(r) for r in rs])
            c = np.array([r["cost"] if ok(r) else np.inf for r in rs])
            perms = np.argsort(rng.random((n_perm, len(rs))), axis=1)
            cw = np.cumsum(w[perms], axis=1); sp = s[perms]
            first = np.where(sp.any(1), cw[np.arange(n_perm), sp.argmax(1)], np.inf)
            if s.any():
                ttfs.append(first.mean())
            for B in budgets:
                succ_B[B].append((np.isfinite(first) & (first <= B)).mean())
                starved[B].append(w.sum() < B)
                within = cw <= B
                cp = np.where(within, c[perms], np.inf).min(1)
                fin = cp[np.isfinite(cp)]
                bestcost[B][t] = fin.mean() if len(fin) == n_perm else np.nan
        out[n] = dict(succ={B: np.mean(v) for B, v in succ_B.items()},
                      starved={B: np.mean(v) for B, v in starved.items()},
                      ttfs=np.median(ttfs) if ttfs else np.nan, nsolved=len(ttfs),
                      bestcost=bestcost)
    return out


def report():
    BUD = [1.0, 10.0, 60.0, 180.0, np.inf]
    res = []
    for row in load():
        names = ["L", "JS", "A"]
        s = single(row, names); m = multistart(row, names, BUD)
        ts = [t for t in m["L"]["bestcost"][np.inf]
              if all(np.isfinite(m[n]["bestcost"][np.inf][t]) for n in names)]
        mc = ({n: np.median([m[n]["bestcost"][np.inf][t] for t in ts]) for n in names}, len(ts))
        res.append(dict(robot=row["robot"], row=row["row"], start=row["start"], solver=row["solver"],
                        single=s, multi=m, mcost=mc))
    order = {"ipopt": 0, "snopt": 1, "nlopt": 2, "svgd": 3}
    res.sort(key=lambda r: (order[r["solver"]], r["row"], r["robot"], r["start"]))
    f = lambda x: "--" if x is None or np.isnan(x) else f"{x:.2f}"
    N = ("L", "JS", "A")
    print("SINGLE START: success /480 | mean wall s (clamped 180) | median cost on cells all three solved;")
    print("best = top arm(s) not significantly beaten (McNemar / Wilcoxon, p < 0.05)")
    for r in res:
        s = r["single"]; e = f"{r['robot']} {'grasp' if r['row'] == 'mugshelf' else 'pose'} {r['start']}"
        print(f"{r['solver']:<6}{e:<22}{' / '.join(str(s['succ'][n]) for n in N):>16} {s['best_succ']:<8}"
              f"{' / '.join(f(s['wall'][n]) for n in N):>24} {s['best_wall']:<8}"
              f"{' / '.join(f(s['cost'][n]) for n in N):>20} [{s['nshared']}] {s['best_cost']}")
    print("\nMULTI-START over each target's 8 recorded starts: % targets solved within 10 s / 60 s / any;")
    print("median time to first success; best-of-8 cost on targets all three solved")
    for r in res:
        m = r["multi"]; e = f"{r['robot']} {'grasp' if r['row'] == 'mugshelf' else 'pose'} {r['start']}"
        g = lambda B: " / ".join(f"{100 * m[n]['succ'][B]:.0f}" for n in N)
        c, nc = r["mcost"]
        print(f"{r['solver']:<6}{e:<22}{g(10.0):>16}{g(60.0):>16}{g(np.inf):>16}"
              f"{' / '.join(f(m[n]['ttfs']) for n in N):>24}{' / '.join(f(c[n]) for n in N):>22} [{nc}]")


if __name__ == "__main__":
    report()
