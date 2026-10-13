"""The analytic + optimization arm beside the record: numbers and verdicts, no prose.

    python scripts/report_analytic.py            # every analytic run found under results/
    python scripts/report_analytic.py --ladder T # only the tight clip ladder

The analytic arm (`analytic` and `analytic8` on the Panda, `analytic` on the iiwa) runs as its
own stage (ANALYTIC: IPOPT and SNOPT; ANALYTICN: NLopt; a `T` suffix is the tight clip ladder),
each item the record's item with only the arms, the tag and the ladder changed. This reader
pairs every analytic run CELL FOR CELL with the record's run on the same row -- refusing unless
`grid_hash` and `scene_fingerprint` agree and the cells coincide -- and prints, per row:

  * L v JS first, the comparison of record (exact McNemar), then each analytic arm A;
  * success, and the cap check: clock stops AND iteration-budget stops (>= 24 flags the row);
  * median iterations over successes and ms per iteration (wall / iterations, median over
    successes; NLopt has no iterations, so ms per map evaluation there);
  * cost on cells BOTH arms solved (median A and the other's, >= 10 shared cells), A v L, A v JS;
  * mean wall over all cells, clamped at the clock;
  * paired starts scored `unrepresentable_start`, and those solved from the clip band;
  * exact McNemar A v L and A v JS.

The analytic arm is an UPPER-BOUND REFERENCE column (Thomas, 2026-10-12): the record's verdicts
are learned v joint space, and nothing here changes them. Thomas writes every word; this prints
numbers.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.report_statusquo import (BUDGET_CELLS, MIN_COST_CELLS, budget_bound,  # noqa: E402
                                      by_cell, clock_bound, load_record, load_svgd_record,
                                      mcnemar, mean,
                                      median, map_jacobians, num, parse_tag)

ARMS = {"panda": ("analytic", "analytic8"), "iiwa": ("analytic",)}
ROBOT_DIR = {"panda": "panda", "iiwa": "iiwa"}
LADDER_NAME = {"": "clip 1e-4 / reach 1e-3 (../codebase)", "T": "clip 1e-8 / reach 1.1e-4 (tight)"}
#: ANALYTIC (IPOPT/SNOPT), ANALYTICN (NLopt), ANALYTICS (AL-SVGD, `_pin2kx` / `_pin2kq` kernel
#: suffix); a `T` after the stage token is the tight clip ladder.
TAG = re.compile(r"^sc_ANALYTIC([NS]?)(T?)_(.+?)(_pin2k[xq])?$")


def load_analytic(cells=480):
    """tag -> summary for every merged analytic run (smokes and shards skipped)."""
    out = {}
    for root in ("results/_cluster_staging/*/results", "results"):
        for f in sorted(glob.glob(f"{root}/*/benchmark/sc_ANALYTIC*/summary.json")):
            tag = os.path.basename(os.path.dirname(f))
            if "_shard" in tag or tag.startswith("sc_ANALYTICSMOKE") or not TAG.match(tag):
                continue
            with open(f) as fh:
                s = json.load(fh)
            n = {len(v) for v in s["records"].values()}
            if cells is not None and n != {cells}:
                continue
            out[tag] = s
    return out


def record_key(t):
    return (t["robot"], t["row"], t["start"], t["solver"])


def index_record(runs):
    out = {}
    for tag, s in runs.items():
        out[record_key(parse_tag(tag))] = (tag, s)
    return out


def check_pairable(a_tag, A, r_tag, R):
    for field in ("grid_hash", "scene_fingerprint", "wall_time", "seed"):
        if A["metadata"].get(field) != R["metadata"].get(field) or A["metadata"].get(field) is None:
            raise SystemExit(f"REFUSING to pair {a_tag} with {r_tag}: {field} differs "
                             f"({A['metadata'].get(field)!r} v {R['metadata'].get(field)!r})")
    a_cells = {k for arm in A["records"] for k in by_cell(A, arm)}
    r_cells = set(by_cell(R, "learned")) | set(by_cell(R, "numerical"))
    if a_cells != r_cells:
        raise SystemExit(f"REFUSING to pair {a_tag} with {r_tag}: the cells differ")


def ms_per_it(r, solver):
    if not r.get("feasible") or not r.get("wall_time"):
        return None
    n = map_jacobians(r) if solver == "nlopt" else r.get("iterations")
    return 1e3 * r["wall_time"] / n if n else None


def describe(cells, solver, cap):
    rs = list(cells.values())
    ok = [r for r in rs if r.get("feasible")]
    return dict(
        succ=len(ok), n=len(rs),
        clock=sum(1 for r in rs if clock_bound(r)),
        budget=sum(1 for r in rs if budget_bound(r)),
        iters=median([r.get("iterations") for r in ok]),
        msit=median([ms_per_it(r, solver) for r in ok]),
        wall=mean([min(r["wall_time"], cap) for r in rs if r.get("wall_time") is not None]),
        unrep=sum(1 for r in rs if r.get("fail_reason") == "unrepresentable_start"),
        band=sum(1 for r in rs if r.get("start_in_clip_band")),
        relaxed_only=sum(1 for r in rs if r.get("feasible_relaxed") and not r.get("feasible")))


def versus(A, B, names=("A", "other")):
    """(A wins, B wins, p, verdict, cost A, cost B, shared) on the cells both cover; the
    verdict names the winning side by `names`, or 'tie'."""
    keys = sorted(set(A) & set(B))
    a = sum(1 for k in keys if A[k].get("feasible") and not B[k].get("feasible"))
    b = sum(1 for k in keys if B[k].get("feasible") and not A[k].get("feasible"))
    p = mcnemar(a, b)
    both = [k for k in keys if A[k].get("feasible") and B[k].get("feasible")]
    ca = median([A[k].get("cost") for k in both]) if len(both) >= MIN_COST_CELLS else None
    cb = median([B[k].get("cost") for k in both]) if len(both) >= MIN_COST_CELLS else None
    return a, b, p, ("tie" if p >= 0.05 else (names[0] if a > b else names[1])), ca, cb, len(both)


def rows(analytic, record, ladder_filter=None):
    """Yield one dict per (analytic run, analytic arm), paired with the record's run."""
    rec = index_record(record)
    for tag in sorted(analytic):
        m = TAG.match(tag)
        ladder = m.group(2)
        if ladder_filter is not None and ladder != ladder_filter:
            continue
        A = analytic[tag]
        kernel = (m.group(4) or "")[len("_pin2k"):]
        t = parse_tag(tag[:len(tag) - len(m.group(4) or "")])
        t["kernel"] = kernel
        hit = rec.get(record_key(t))
        if hit is None:
            print(f"  note  {tag}: no record run for {record_key(t)}; skipped")
            continue
        r_tag, R = hit
        check_pairable(tag, A, r_tag, R)
        cap = A["metadata"].get("wall_time") or float("inf")
        L, J = by_cell(R, "learned"), by_cell(R, "numerical")
        for arm in A["records"]:
            Ac = by_cell(A, arm)
            yield dict(t, tag=tag, record=r_tag, ladder=ladder, arm=arm, clock=cap,
                       L=describe(L, t["solver"], cap), J=describe(J, t["solver"], cap),
                       A=describe(Ac, t["solver"], cap),
                       LvJ=versus(L, J, ("L", "JS")), AvL=versus(Ac, L, ("A", "L")),
                       AvJ=versus(Ac, J, ("A", "JS")))


def flag(d):
    return "  BUDGET-BOUND" if d["budget"] >= BUDGET_CELLS else ""


def print_rows(rs):
    head = (f"{'robot':6}{'row':9}{'start':7}{'solver':7}{'arm':10}"
            f"{'L':>5}{'JS':>5}{'A':>5}{'L v JS':>14}{'A v L':>16}{'A v JS':>16}"
            f"{'it L/JS/A':>16}{'ms/it L/JS/A':>20}{'cost A:L':>16}{'cost A:JS':>16}"
            f"{'wall L/JS/A':>20}{'clk A':>6}{'bud A':>6}{'unrep':>6}{'band':>5}")
    print(head)
    for r in rs:
        L, J, A = r["L"], r["J"], r["A"]

        def vs(v):
            a, b, p, verdict, *_ = v
            return f"{a:>3}/{b:<3} {verdict:>5}{'' if verdict == 'tie' else '*'}"

        def cost(v):
            _, _, _, _, ca, cb, n = v
            return (f"{num(ca, 6, 2)}:{num(cb, 6, 2)}" if ca is not None
                    else f"{'N/A':>8} ({n})")
        solver = r['solver'] + (f"/{r['kernel']}" if r.get("kernel") else "")
        print(f"{r['robot']:6}{r['row']:9}{r['start']:7}{solver:7}{r['arm']:10}"
              f"{L['succ']:>5}{J['succ']:>5}{A['succ']:>5}{vs(r['LvJ']):>14}{vs(r['AvL']):>16}"
              f"{vs(r['AvJ']):>16}"
              f"{num(L['iters'], 5)}{num(J['iters'], 5)}{num(A['iters'], 6)}"
              f"{num(L['msit'], 7, 1)}{num(J['msit'], 6, 1)}{num(A['msit'], 7, 1)}"
              f"{cost(r['AvL']):>16}{cost(r['AvJ']):>16}"
              f"{num(L['wall'], 7, 2)}{num(J['wall'], 6, 2)}{num(A['wall'], 7, 2)}"
              f"{A['clock']:>6}{A['budget']:>6}{A['unrep']:>6}{A['band']:>5}{flag(A)}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", choices=["", "T"], default=None,
                   help="'' for ../codebase's clip ladder, 'T' for the tight one; default both")
    p.add_argument("--cells", type=int, default=480)
    args = p.parse_args(argv)
    record, _ = load_record(args.cells, which="paper")
    try:
        svgd, _ = load_svgd_record(record, args.cells)
        record = dict(record, **svgd)
    except SystemExit as e:
        print(f"  note  the AL-SVGD record rows are not loaded: {e}")
    analytic = load_analytic(args.cells)
    print(f"{len(analytic)} analytic runs, {len(record)} record runs (IPOPT / SNOPT at paper "
          f"conditions, NLopt at PROCS=8 MPS=1)")
    print("Verdicts are exact McNemar on the paired cells, naming the side that solved more where")
    print("p < 0.05 ('*'). Cost is the median on cells both arms solved (N/A under "
          f"{MIN_COST_CELLS} shared cells,")
    print("count in brackets). ms/it is per IPOPT/SNOPT iteration, per map evaluation under NLopt.")
    print(f"A row with >= {BUDGET_CELLS} analytic cells at an iteration budget carries no verdict.\n")
    all_rows = list(rows(analytic, record, args.ladder))
    for ladder in ("", "T"):
        rs = [r for r in all_rows if r["ladder"] == ladder]
        if not rs:
            continue
        print(f"=== {LADDER_NAME[ladder]} ===")
        print_rows(rs)
        relaxed = sum(r["A"]["relaxed_only"] for r in rs)
        print(f"  cells the analytic arm passes only at the relaxed gate (task_tol): {relaxed}\n")
    return all_rows


if __name__ == "__main__":
    main()
