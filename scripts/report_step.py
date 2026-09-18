#!/usr/bin/env python3
"""Read a stage-STEP screen or confirmation: success, iterations, cost, wall clock.

Why this exists rather than `collate.py`:

  - `collate.py --pair` prints success/better/worse/p but NOT iterations, cost or wall
    clock, and Thomas's standing rule is that a result is told in all four.
  - cost has to be compared on the cells BOTH SETTINGS solved. `summary["_common"]` is
    cells both ARMS solved within ONE run, so using it across settings is the same
    "median over each arm's own successes" trap one level up -- the easy cells are exactly
    the ones a weaker setting also solves.
  - a knob here is a SOLVER option, so it moves the joint-space arm too. Both arms are
    reported, and the learned-vs-joint-space McNemar is carried per setting, because
    "does the knob close the row" is the actual question and is already computed.

Usage:
    scripts/report_step.py 'results/*/benchmark/sc_STEP_iiwa_n4_ipopt_mugshelf_60_*'
    scripts/report_step.py --arm numerical '<glob>'
    scripts/report_step.py --ref accdefault '<glob>'          # any baseline, by name
    scripts/report_step.py --recover <baseline-summary> <candidate-summary> ...
    scripts/report_step.py --resto '<glob>'                   # IPOPT restoration fraction
"""
import glob
import json
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmark import mcnemar_exact  # noqa: E402


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "summary.json")
    with open(path) as f:
        return json.load(f)


def cells(data, arm):
    """(target, guess) -> record, for one arm."""
    return {(r["target"], r["guess"]): r for r in data["records"].get(arm, [])}


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def token(tag):
    """The setting token is the last underscore-separated field of the tag."""
    return tag.rsplit("_", 1)[-1]


def comparable(a, b):
    """The same guard collate.py uses, so this script cannot pair what that one refuses."""
    ka = ("grid_hash", "scene", "target_placement", "shelf_inset", "start", "solver",
          "checkpoint", "task")
    return [k for k in ka if a["metadata"].get(k) != b["metadata"].get(k)]


def report(paths, arm, ref=None):
    runs = []
    for p in paths:
        try:
            d = load(p)
        except (OSError, ValueError):
            continue
        if arm in d.get("summary", {}):
            runs.append((os.path.basename(os.path.dirname(os.path.join(p, "x"))), d))
    if len(runs) < 2:
        sys.exit(f"need at least two runs carrying an arm named {arm!r}; got {len(runs)}")
    ## The reference is chosen BY NAME, never by sort order or argument order. That is the
    ## whole lesson of the collate.py --pair trap: there, the reference is the first path in
    ## argument order, and stage SWEEP's tokens happen to put `accdefault`, `acciter5` and
    ## `crash0` ahead of `default`, so a bare glob silently paired every setting against the
    ## wrong column. `default` is the default reference because that is what stage STEP's
    ## tables are built around; `--ref` overrides it, which is what makes this script usable
    ## on the archived SWEEP rows too.
    runs.sort(key=lambda r: token(r[0]))
    ref = ref or "default"
    matches = [r for r in runs if token(r[0]) == ref]
    if not matches:
        sys.exit(f"no run with setting token {ref!r} among "
                 f"{sorted({token(n) for n, _ in runs})}. Pass --ref to name the baseline.")
    if len(matches) > 1:
        sys.exit(f"{len(matches)} runs carry the token {ref!r} -- the glob spans more than "
                 f"one row. Constrain it.")
    base_name, base = matches[0]
    ## Reference first, then the rest in token order, so the table reads as a sweep.
    runs = matches + [r for r in runs if r[0] != base_name]
    bc = cells(base, arm)

    print(f"arm: {arm}    reference: {base_name}")
    print(f"grid {base['metadata'].get('grid_hash')}  "
          f"{base['metadata'].get('solver')}  {base['metadata'].get('start')}  "
          f"{base['metadata'].get('target_placement')}\n")
    hdr = (f"{'setting':13s} {'ok/n':>8} {'vs def':>9} {'p':>9} {'t/out':>6} "
           f"{'iters':>7} {'ms/it':>6} {'wall':>7} {'cost(both)':>11} {'defcost':>8} "
           f"{'viol':>9}  notes")
    print(hdr)
    print("-" * len(hdr))
    for name, d in runs:
        s = d["summary"][arm]
        c = cells(d, arm)
        shared = sorted(set(c) & set(bc))
        note = ""
        diff = comparable(d, base)
        if diff:
            note = "NOT COMPARABLE (%s)" % ", ".join(diff)
        mine = [bool(c[k].get("feasible")) for k in shared]
        theirs = [bool(bc[k].get("feasible")) for k in shared]
        m = mcnemar_exact(mine, theirs)
        ## Cost on cells BOTH settings solved -- the whole point of this script.
        both = [k for k in shared
                if c[k].get("feasible") and bc[k].get("feasible")]
        cost_both = _median([c[k].get("cost") for k in both])
        cost_ref = _median([bc[k].get("cost") for k in both])
        ok = [r for r in c.values() if r.get("feasible")]
        it = _median([r.get("iterations") for r in ok])
        wall = _median([r.get("wall_time") for r in ok])
        msit = (1000.0 * wall / it) if (it and wall) else None
        def f(x, spec):
            return format(x, spec) if x is not None else "--"
        delta = "" if name == base_name else f"+{m['a_only']}/-{m['b_only']}"
        print(f"{token(name):13s} {s['successes']:>4}/{s['n']:<3} {delta:>9} "
              f"{'' if name == base_name else format(m['p'], '.3g'):>9} "
              f"{s['timeouts']:>6} {f(it, '.0f'):>7} {f(msit, '.0f'):>6} "
              f"{f(wall, '.2f'):>7} {f(cost_both, '.3f'):>11} {f(cost_ref, '.3f'):>8} "
              f"{f(s.get('median_max_violation'), '.2e'):>9}  {note}")

    ## The row's actual question, already computed inside each run.
    print(f"\nlearned vs joint space, per setting (from summary['_mcnemar']):")
    for name, d in runs:
        mc = (d["summary"].get("_mcnemar") or {}).get("learned vs numerical")
        if mc:
            print(f"  {token(name):13s} +{mc['a_only']:>3} / -{mc['b_only']:<3} "
                  f"p = {mc['p']:.3g}")

    ## Failure-mode shift is the mechanism claim for SNOPT: did a smaller step limit turn
    ## INFO 41 (`current point cannot be improved`, a line search that cannot find a step)
    ## into convergence? A success count alone cannot show that.
    print(f"\nsolver status histogram over FAILED cells (SNOPT INFO 13/41/34):")
    for name, d in runs:
        h = {}
        for r in cells(d, arm).values():
            if not r.get("feasible"):
                h[r.get("solver_status")] = h.get(r.get("solver_status"), 0) + 1
        print(f"  {token(name):13s} {dict(sorted(h.items(), key=lambda kv: -kv[1]))}")


def recover(baseline, candidates, arm):
    """How much of a NAMED budget-recoverable set a setting actually recovers.

    The iiwa contained-grasp row's 47 cells are the case this was written for: they are the
    cells the 45 s column fails and the 180 s column solves, so they are known to be
    solvable and known to be budget-bound. "Recovered K of 47" says far more than a success
    count, because it separates recovering stalled cells from resampling noise.
    """
    b = load(baseline)
    bc = cells(b, arm)
    fails = {k for k, r in bc.items() if not r.get("feasible")}
    print(f"baseline {os.path.basename(os.path.dirname(baseline))}: "
          f"{len(fails)} failed cells of {len(bc)}  "
          f"(grid {b['metadata'].get('grid_hash')})\n")
    for p in candidates:
        d = load(p)
        diff = comparable(d, b)
        c = cells(d, arm)
        rec = sum(1 for k in fails if k in c and c[k].get("feasible"))
        lost = sum(1 for k, r in bc.items()
                   if r.get("feasible") and k in c and not c[k].get("feasible"))
        print(f"  {os.path.basename(os.path.dirname(p)):58s} "
              f"recovered {rec:>3}/{len(fails)}  lost {lost:>3}"
              + ("   NOT COMPARABLE (%s)" % ", ".join(diff) if diff else ""))


def restoration(paths, arm="learned"):
    """Fraction of IPOPT iterations spent in the FEASIBILITY RESTORATION phase, per run.

    This is the mechanism column for the theta_max_fact axis, and it is free: IPOPT marks a
    restoration iteration with a trailing `r` on the iteration index in its own print file,
    and `src/benchmark.py` already archives one print file per cell into
    `solver_logs.tar.gz`.

    Why it is the right column. IpFilterLSAcceptor.cpp:328 fixes
    theta_max = theta_max_fact * max(1, theta(x_0)) once, from the first iterate. Below 1 the
    ceiling sits UNDER the start's own violation, so every trial point that fails to reduce
    violation immediately is refused and IPOPT falls into restoration -- which a success count
    cannot distinguish from any other way of failing.
    """
    it_re = re.compile(r"^\s*(\d+)(r?)\s")
    print(f"{'run':58s} {'cells':>6} {'tot it':>7} {'resto it':>9} {'resto %':>8} "
          f"{'>50% resto':>11}")
    for path in paths:
        tar = path if path.endswith(".tar.gz") else os.path.join(
            os.path.dirname(os.path.join(path, "x")), "solver_logs.tar.gz")
        if not os.path.exists(tar):
            continue
        tot, res, hi = [], [], 0
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if not m.isfile() or arm not in os.path.basename(m.name):
                    continue
                t = r = 0
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                for raw in fh.read().decode("utf-8", "ignore").splitlines():
                    mm = it_re.match(raw)
                    if mm:
                        t += 1
                        r += mm.group(2) == "r"
                if t:
                    tot.append(t)
                    res.append(r)
                    hi += (r / t) > 0.5
        if not tot:
            continue
        pct = [100.0 * r / t for r, t in zip(res, tot)]
        print(f"{os.path.basename(os.path.dirname(tar)):58s} {len(tot):>6} "
              f"{_median(tot):>7.0f} {_median(res):>9.0f} {_median(pct):>7.1f}% {hi:>11}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    arm, ref = "learned", None
    if "--arm" in argv:
        i = argv.index("--arm")
        arm = argv[i + 1]
        del argv[i:i + 2]
    if "--ref" in argv:
        i = argv.index("--ref")
        ref = argv[i + 1]
        del argv[i:i + 2]
    if argv and argv[0] == "--resto":
        paths = [q for p in argv[1:] for q in (sorted(glob.glob(p)) or [p])]
        if not paths:
            sys.exit("usage: --resto <run-dir-or-glob>...")
        restoration(paths, arm)
    elif argv and argv[0] == "--recover":
        paths = [q for p in argv[1:] for q in (sorted(glob.glob(p)) or [p])]
        if len(paths) < 2:
            sys.exit("usage: --recover <baseline> <candidate> [<candidate>...]")
        recover(paths[0], paths[1:], arm)
    else:
        paths = [q for p in argv for q in (sorted(glob.glob(p)) or [p])]
        if not paths:
            sys.exit(__doc__)
        report(paths, arm, ref)
