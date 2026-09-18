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
    # The reference is the first path in argument order, which makes it depend on
    # lexicographic luck whenever a glob expands a settings table: `accoff` and `crash0` both
    # sort before `default`, so a sweep glob would silently baseline every setting against a
    # non-default column. Prefer a run whose tag ends in `_default` when one is present, and
    # say which rule picked it.
    _pick = [i for i, (n, _) in enumerate(runs) if n.endswith("_default")]
    ref_name, ref = runs[_pick[0]] if _pick else runs[0]
    if _pick:
        runs = [runs[_pick[0]]] + [r for i, r in enumerate(runs) if i != _pick[0]]
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
        mine, theirs = data["metadata"].get("grid_hash"), ref["metadata"].get("grid_hash")
        if mine is None or theirs is None:
            note = "no grid_hash -- provenance unknown, not evidence of a shared grid"
        elif mine != theirs:
            note = "DIFFERENT GRID -- not comparable"
        else:
            # A matching grid_hash is necessary but not sufficient, in two separate ways.
            #
            # The hardened-scene axis changes which targets were admissible and which
            # obstacles exist, and two runs differing only there would otherwise print a
            # clean McNemar row.
            #
            # And the grid_hash does NOT depend on the start protocol, the solver or the
            # checkpoint -- it hashes the targets and guesses, nothing else. Measured:
            # sc_SOLVER2_iiwa_n4_ipopt_mugshelf_480_45_native, its _paired twin, the snopt
            # version of both, and sc_CAP_iiwa_n4_mug_180_paired all carry the SAME
            # fa692df81e7d-mug. Since a stage writes those columns into one
            # results/<robot>/benchmark/ directory with tags differing only in a middle
            # token, a glob like '..._mugshelf_480_45_*' sorts native before paired and
            # would pair a native column against a paired one with no warning at all. That
            # is the same class of collision that --shard, --checkpoint and the iiwa tag's
            # missing solver token were each fixed for; here it corrupts an ANALYSIS rather
            # than a filename, which is harder to notice afterwards.
            #
            # And `task` is in the list because the PANDA's grasp and pose grids collide
            # outright. The iiwa's script appends the task to its hash (see its own comment
            # there, added because its mug and pose grids hashed identically); the Panda's
            # never did, and measured on the archive
            # sc_SOLVER2_panda_n6_*_mugshelf_* and sc_SOLVER2_panda_n6_*_posetip_* both
            # carry grid_hash d7a4ef1609b9. So a Panda glob spanning tasks would pair a
            # GRASP column against a POSE one with no warning. Deliberately fixed here
            # rather than by suffixing the Panda hash: changing that hash would make every
            # new Panda run incomparable to every archived one, which is a far larger loss
            # than the trap, and `task` has always been in the metadata.
            differs = [k for k in ("scene", "target_placement", "shelf_inset",
                                   "start", "solver", "checkpoint", "task")
                       if data["metadata"].get(k) != ref["metadata"].get(k)]
            if differs:
                note = ("DIFFERENT SCENE/PLACEMENT/PROTOCOL (%s) -- not comparable"
                        % ", ".join(differs))
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
                icap=s.get("iteration_capped", 0),
                iters=s["mean_iterations"], jac=s["mean_jacobian_evals"],
                ## The cross-solver cost column. NLopt reports no iteration count at all
                ## (no log, and a details struct with one status field), so under it
                ## `iters` and `jac` are nan by construction and THIS is what the run is
                ## read by. Counted by the program itself, so it is comparable across
                ## solvers and across arms.
                mapjac=s.get("mean_map_jacobians", float("nan")),
                ## Seconds are this machine's; iterations are the formulation's. Reporting
                ## both is a standing rule (see CLAUDE.md, "Iterations alongside wall
                ## clock"), and ms/iter is what separates the two -- the learned arm's
                ## iteration costs ~30x the joint-space arm's, so equal iteration counts
                ## are not equal solves.
                ms_it=(1000.0 * s["mean_wall_time"] / s["mean_iterations"]
                       if s["mean_iterations"] else float("nan")),
                wall=s["mean_wall_time"], wall_ok=s["mean_wall_time_success"],
                cost=s["median_cost"], reasons=s["fail_reasons"],
                start_err=s.get("median_start_q_error", float("nan")),
                qc=s.get("median_correction_inf", float("nan")),
                binding=s.get("correction_binding", float("nan")),
                ok_relaxed=s.get("successes_relaxed"),
                gain=s.get("relaxation_gain"),
                maxviol=s.get("median_max_violation")))

    ## A solver that reports no iteration count must print as "--", never as nan or 0.
    ## NLopt genuinely has none, and a 0 in this column would read as "converged instantly"
    ## rather than "this solver does not tell us".
    def num(value, width, places):
        if value is None or value != value:
            return f"{'--':>{width}}"
        return f"{value:>{width}.{places}f}"

    header = (f"{'run':<26} {'arm':<10} {'success':>9} {'rate':>6} {'95% CI':>14} "
              f"{'t/out':>6} {'i/cap':>6} {'iters':>7} {'ms/it':>7} {'jac':>7} "
              f"{'mapjac':>7} {'wall':>7} {'cost':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run']:<26} {r['arm']:<10} {r['ok']:>4}/{r['n']:<4} {r['rate']:>6.2f} "
              f"[{r['lo']:.2f},{r['hi']:.2f}]".ljust(len(header) - 52)
              + f"{r['timeouts']:>6} {r['icap']:>6} {num(r['iters'], 7, 0)} "
                f"{num(r['ms_it'], 7, 1)} {num(r['jac'], 7, 0)} "
                f"{num(r['mapjac'], 7, 0)} "
                f"{num(r['wall'], 7, 2)} {num(r['cost'], 7, 2)}")
    ## Both success criteria, side by side. `strict` gates on the program's own
    ## constraint rows at ik_constraint_tol; `task-tol` relaxes that to the task gate's
    ## tolerance, which is the open question about what should count as a solve. Only
    ## printed when a run actually carries the relaxed scoring, so archived summaries
    ## still collate.
    if any(r["ok_relaxed"] is not None for r in rows):
        print("\nsuccess under both criteria (strict = program rows; task-tol = relaxed to the task gate)")
        for r in rows:
            if r["ok_relaxed"] is None:
                continue
            mv = "n/a" if r["maxviol"] is None else f"{r['maxviol']:.2e}"
            print(f"  {r['run']:<26} {r['arm']:<10} strict {r['ok']:>3}/{r['n']:<4} "
                  f"task-tol {r['ok_relaxed']:>3}/{r['n']:<4} "
                  f"gained {r['gain']:>3}   median max violation {mv}")
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
