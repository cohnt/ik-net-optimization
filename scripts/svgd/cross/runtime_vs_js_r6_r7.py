import json, statistics as st, random, collections
B="results/panda/benchmark/"
def load(tag, arm):
    s=json.load(open(B+tag+"/summary.json")); recs=s["records"][arm]
    def key(r): return (r["target"],r["guess"]) if "target" in r else (r["target_index"],r["guess_index"])
    return {key(r):(bool(r.get("feasible")), min(r["wall_time"],180.0)) for r in recs}
random.seed(0)
def multistart(js, budgets, nperm=300):
    """per target: random order over its recorded starts, sequential restarts, stop at first success.
    returns (frac targets solved within per-cell budget, averaged over cells & perms; frac starved), and time-to-first-success medians."""
    tg=collections.defaultdict(list)
    for (t,g),v in js.items(): tg[t].append(v)
    hit=0; n=0; starved=0; ttfs=[]
    for t,cells in tg.items():
        for _ in range(nperm):
            order=cells[:]; random.shuffle(order); cum=0; solved_at=None
            for ok,w in order:
                cum+=w
                if ok: solved_at=cum; break
            ttfs.append(solved_at if solved_at is not None else float("inf"))
            for (tt,g),bud in budgets.items():
                if tt!=t: continue
                n+=1
                if solved_at is not None and solved_at<=bud: hit+=1
                elif solved_at is None and cum<=bud: starved+=1
    return hit/n, starved/n, ttfs
def fixed_budget(ttfs, Bs): return {b: sum(1 for x in ttfs if x<=b)/len(ttfs) for b in Bs}
rows=[("mugshelf","paired"),("mugshelf","native"),("posetip","paired"),("posetip","native")]
Bs=[1,2,5,10,30,180]
for task,start in rows:
    print(f"\n=== Panda {task} {start}")
    for label, svtag, iptag, sntag in [
        ("seed 1: svgd K1 (clearance row) vs R2 IPOPT / PAPER SNOPT", f"sc_SVGD_R6_panda_n6_svgd_{task}_480_180_{start}_K1", f"sc_SVGD_R2_panda_n6_ipopt_{task}_480_180_{start}", f"sc_PAPER_panda_n6_snopt_{task}_480_180_{start}"),
        ("seed 2: svgd kq (clearance row) vs R7 IPOPT twin", f"sc_SVGD_R7_panda_n6_svgd_{task}_480_180_{start}_kq", f"sc_SVGD_R7_panda_n6_ipopt_{task}_480_180_{start}", None)]:
        sv=load(svtag,"learned"); ip_js=load(iptag,"numerical"); ip_l=load(iptag,"learned")
        print(f"  {label}")
        def line(name,d):
            w=[v[1] for v in d.values()]; print(f"    {name:28s} succ {sum(v[0] for v in d.values()):3d}/480  wall mean {st.mean(w):6.2f} med {st.median(w):6.2f} p90 {sorted(w)[int(0.9*len(w))]:6.2f} max {max(w):6.1f}")
        line("svgd learned N=64",sv); line("IPOPT learned",ip_l); line("IPOPT joint space",ip_js)
        budgets={k:v[1] for k,v in sv.items()}
        f,sv_,tt=multistart(ip_js,budgets); fb=fixed_budget(tt,Bs)
        print(f"    IPOPT JS restarts within svgd's per-cell wall: {100*f:5.1f}% of cells (starved {100*sv_:.0f}%) | JS restarts solve a target within B s: "+" ".join(f"{b}s:{100*fb[b]:.0f}%" for b in Bs))
        svtt=sorted(v[1] for v in sv.values()); print(f"    svgd single solve within B s:                                                       "+" ".join(f"{b}s:{100*sum(1 for x,ok in [(v[1],v[0]) for v in sv.values()] if x<=b and ok)/480:.0f}%" for b in Bs))
        if sntag:
            sn_js=load(sntag,"numerical"); sn_l=load(sntag,"learned"); line("SNOPT learned",sn_l); line("SNOPT joint space",sn_js)
            f,sv_,tt=multistart(sn_js,budgets); fb=fixed_budget(tt,Bs)
            print(f"    SNOPT JS restarts within svgd's per-cell wall: {100*f:5.1f}% of cells (starved {100*sv_:.0f}%) | within B s: "+" ".join(f"{b}s:{100*fb[b]:.0f}%" for b in Bs))
