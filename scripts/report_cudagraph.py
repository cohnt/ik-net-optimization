#!/usr/bin/env python3
"""Read stage CUDAGRAPH: the record's IPOPT rows with and without `flow_cuda_graph`.

Each row ran twice in one stage (`cluster/gen_manifest.py`, `stage_CUDAGRAPH`), same nodes,
same PROCS=8, same grid: sc_CGCTRL_STATUSQUO_<run> is the record's configuration re-run, and
sc_CUDAGRAPH_STATUSQUO_<run> adds the switch. Only the learned arm evaluates the network, so
the switch can move only that arm; the joint-space columns are the contention control.

Per row:

  - per-iteration cost of each arm in each variant (median over ALL cells of wall / iterations,
    so failures that ran to the clock count), and the learned arm's premium over joint space;
  - learned time to solution on cells BOTH variants solved, as a ratio, so the same cells are
    compared;
  - learned success gained / lost between the variants on identical cells (exact McNemar), and
    the learned-vs-joint-space verdict in each;
  - `same its`: on cells both variants solved, the fraction converging in exactly the same
    number of iterations. The replayed Jacobian is bit-identical to the compiled one, but the
    forward pass becomes compiled where it was eager (agreeing to ~1e-14), so trajectories may
    part; this says how often they did;
  - the control against the record (stage ITCAP for the grasp rows, STATUSQUO for pose), the
    acceptance check that the re-run reproduces what was reported.

    python scripts/report_cudagraph.py        # stage CUDAGRAPH, PROCS=8
    python scripts/report_cudagraph.py P2     # stage CUDAGRAPHP2, one process per V100
    python scripts/report_cudagraph.py MPS    # stage CUDAGRAPHMPS, PROCS=8 under MPS
"""
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from report_statusquo import arm_stats, by_cell, load, mcnemar, verdict  # noqa: E402


def ms_per_it(summary, arm):
    vals = [1e3 * r["wall_time"] / r["iterations"] for r in summary["records"][arm]
            if r.get("iterations") and r.get("wall_time")]
    return median(vals) if vals else None


def between_arms(s):
    A, B = by_cell(s, "learned"), by_cell(s, "numerical")
    L = sum(r["feasible"] for r in A.values())
    J = sum(r["feasible"] for r in B.values())
    lo = sum(1 for k in A if k in B and A[k]["feasible"] and not B[k]["feasible"])
    jo = sum(1 for k in A if k in B and B[k]["feasible"] and not A[k]["feasible"])
    p = mcnemar(lo, jo)
    return L, J, verdict(L, J, p), p


def gained_lost(old, new, arm):
    O, N = by_cell(old, arm), by_cell(new, arm)
    g = sum(1 for k in N if k in O and N[k]["feasible"] and not O[k]["feasible"])
    lost = sum(1 for k in N if k in O and O[k]["feasible"] and not N[k]["feasible"])
    return g, lost, mcnemar(g, lost)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    sfx = sys.argv[1] if len(sys.argv) > 1 else ""
    G_, C_ = f"sc_CUDAGRAPH{sfx}_STATUSQUO_", f"sc_CGCTRL{sfx}_STATUSQUO_"
    graph = load(G_, cells=480)
    ctrl = load(C_, cells=480)
    if not graph:
        print("no merged sc_CUDAGRAPH runs found")
        return 1
    f = lambda x, pr=1: "--" if x is None else "%.*f" % (pr, x)
    print(f"### Stage CUDAGRAPH{sfx}: IPOPT, 180 s, {dict(P2='PROCS=2', MPS='PROCS=8 + MPS').get(sfx, 'PROCS=8')}, control (`--compile`) against "
          "`--compile --set flow_cuda_graph=True`\n")
    print("| row | ms/it L ctrl -> graph (x) | ms/it J ctrl / graph | premium L/J ctrl -> graph "
          "| L time-to-solve, shared cells (x, n) | L success ctrl -> graph (gained / lost, p) "
          "| verdict ctrl -> graph | same its | L iters ctrl / graph | timeouts L ctrl / graph |")
    print("| " + " | ".join(["---"] * 10) + " |")
    for tag, g in sorted(graph.items()):
        run = tag[len(G_):]
        name = run.replace("_ipopt", "").replace("_480_180", "")
        c = ctrl.get(f"{C_}{run}")
        if c is None:
            print(f"| {name} | CONTROL NOT FOUND |" + " |" * 8)
            continue
        if c["metadata"].get("grid_hash") != g["metadata"].get("grid_hash"):
            print(f"| {name} | GRID MISMATCH |" + " |" * 8)
            continue
        Lc, Lg = ms_per_it(c, "learned"), ms_per_it(g, "learned")
        Jc, Jg = ms_per_it(c, "numerical"), ms_per_it(g, "numerical")
        C, G = by_cell(c, "learned"), by_cell(g, "learned")
        both = [k for k in C if k in G and C[k]["feasible"] and G[k]["feasible"]]
        speed = (median([C[k]["wall_time"] / G[k]["wall_time"] for k in both])
                 if both else None)
        same = (sum(C[k]["iterations"] == G[k]["iterations"] for k in both) / len(both)
                if both else None)
        gl = gained_lost(c, g, "learned")
        vc, vg = between_arms(c), between_arms(g)
        sc, sg = arm_stats(c, "learned", "numerical"), arm_stats(g, "learned", "numerical")
        print(f"| {name} | {f(Lc)} -> {f(Lg)} ({f(Lc / Lg if Lc and Lg else None, 2)}) "
              f"| {f(Jc)} / {f(Jg)} "
              f"| {f(Lc / Jc if Lc and Jc else None)} -> {f(Lg / Jg if Lg and Jg else None)} "
              f"| {f(speed, 2)} ({len(both)}) "
              f"| {vc[0]} -> {vg[0]} ({gl[0]} / {gl[1]}, p = {gl[2]:.2g}) "
              f"| {vc[2]} ({vc[0]} v {vc[1]}) -> {vg[2]} ({vg[0]} v {vg[1]}) "
              f"| {f(100 * same if same is not None else None, 0)}% "
              f"| {f(sc['iters'], 0)} / {f(sg['iters'], 0)} | {sc['to']} / {sg['to']} |")

    print("\n### Acceptance: the control against the record (same grid, same configuration)\n")
    print("| row | record L / J | control L / J | learned discordant (record-only / control-only) "
          "| joint discordant |")
    print("| --- | --- | --- | --- | --- |")
    for tag, c in sorted(ctrl.items()):
        run = tag[len(C_):]
        name = run.replace("_ipopt", "").replace("_480_180", "")
        rec = (load(f"sc_ITCAP1e6_STATUSQUO_{run}", cells=480).get(f"sc_ITCAP1e6_STATUSQUO_{run}")
               or load(f"sc_STATUSQUO_{run}", cells=480).get(f"sc_STATUSQUO_{run}"))
        if rec is None or rec["metadata"].get("grid_hash") != c["metadata"].get("grid_hash"):
            print(f"| {name} | RECORD NOT FOUND OR GRID MISMATCH | | | |")
            continue
        r, n = between_arms(rec), between_arms(c)
        dl, dj = gained_lost(rec, c, "learned"), gained_lost(rec, c, "numerical")
        print(f"| {name} | {r[0]} / {r[1]} | {n[0]} / {n[1]} | {dl[1]} / {dl[0]} | {dj[1]} / {dj[0]} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
