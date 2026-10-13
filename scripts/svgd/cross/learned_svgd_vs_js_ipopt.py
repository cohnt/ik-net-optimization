import json, statistics as st, random, collections, math
random.seed(0)
def load(path, arm):
    s=json.load(open(path+"/summary.json")); recs=s["records"][arm]
    def key(r): return (r["target"],r["guess"]) if "target" in r else (r["target_index"],r["guess_index"])
    return {key(r):(bool(r.get("feasible")), min(r["wall_time"],180.0), r.get("cost")) for r in recs}, s["metadata"]["grid_hash"]
def multistart(js, budgets, nperm=200):
    tg=collections.defaultdict(list)
    for (t,g),v in js.items(): tg[t].append(v[:2])
    hit=n=starved=0
    for t,cells in tg.items():
        for _ in range(nperm):
            order=cells[:]; random.shuffle(order); cum=0; solved_at=None
            for ok,w in order:
                cum+=w
                if ok: solved_at=cum; break
            for (tt,g),bud in budgets.items():
                if tt!=t: continue
                n+=1
                if solved_at is not None and solved_at<=bud: hit+=1
                elif solved_at is None and cum<=bud: starved+=1
    return hit/n, starved/n
ROBOTS=[("panda","panda","n6","sc_SVGD_R2_panda_n6_ipopt"),("iiwa","iiwa","n4","sc_PAPER_iiwa_n4_ipopt"),
        ("screw7_p050","screw_arm","n6","sc_PAPER_screw7_p050_n6_ipopt"),("soft12","soft12","n6","sc_PAPER_soft12_n6_ipopt")]
print(f"{'row':28s} {'L-svgd':>6s} {'JS-IPOPT':>8s} {'L+/JS+':>7s} {'p':>8s}  {'JS time-matched':>15s} {'starved':>7s}  {'cost L/JS (n)':>16s}  {'wall L/JS':>11s}")
for robot,rdir,rung,iptag in ROBOTS:
    for task in ("mugshelf","posetip"):
        for start in ("native","paired"):
            sv,h1=load(f"results/{rdir}/benchmark/sc_SVGD_R20_{robot}_{rung}_svgd_{task}_480_180_{start}_pin2","learned")
            js,h2=load(f"results/{rdir}/benchmark/{iptag}_{task}_480_180_{start}","numerical")
            assert h1==h2, (robot,task,start,h1,h2)
            keys=sorted(sv); a=sum(sv[k][0] for k in keys); b=sum(js[k][0] for k in keys)
            lp=sum(sv[k][0] and not js[k][0] for k in keys); jp=sum(js[k][0] and not sv[k][0] for k in keys)
            n=lp+jp; k=min(lp,jp); p=min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1.0
            tm,stv=multistart(js,{kk:sv[kk][1] for kk in keys})
            shared=[kk for kk in keys if sv[kk][0] and js[kk][0] and sv[kk][2] is not None and js[kk][2] is not None]
            cl=st.median([sv[kk][2] for kk in shared]); cj=st.median([js[kk][2] for kk in shared])
            wl=st.mean([sv[kk][1] for kk in keys]); wj=st.mean([js[kk][1] for kk in keys])
            print(f"{robot+' '+task+' '+start:28s} {a:6d} {b:8d} {lp:3d}/{jp:<3d} {p:8.1e}  {100*tm:14.1f}% {100*stv:6.1f}%  {cl:6.2f}/{cj:<6.2f}({len(shared):3d})  {wl:5.2f}/{wj:<5.2f}")
