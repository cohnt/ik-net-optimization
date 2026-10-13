import json, statistics as st, math, sys
def load(path, arm):
    s=json.load(open(path+"/summary.json")); recs=s["records"][arm]
    def key(r): return (r["target"],r["guess"]) if "target" in r else (r["target_index"],r["guess_index"])
    return {key(r):(bool(r.get("feasible")), min(r["wall_time"],180.0), r.get("cost"), r.get("svgd",{})) for r in recs}, s["metadata"]["grid_hash"]
def mcn(a,b,keys):
    ap=sum(a[k][0] and not b[k][0] for k in keys); bp=sum(b[k][0] and not a[k][0] for k in keys)
    n=ap+bp; k=min(ap,bp); p=min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1.0
    return ap,bp,p
ROBOTS=[("panda","panda","n6","sc_SVGD_R2_panda_n6_ipopt","R20"),("iiwa","iiwa","n4","sc_PAPER_iiwa_n4_ipopt","R21"),
        ("screw7_p050","screw_arm","n6","sc_PAPER_screw7_p050_n6_ipopt","R21"),("soft12","soft12","n6","sc_PAPER_soft12_n6_ipopt","R21")]
print(f"{'row':26s} {'JSsvgd':>6s} {'JSipopt':>7s} {'J+/I+':>7s} {'p':>8s} | {'Lsvgd':>5s} {'L+/J+':>7s} {'p':>8s} | {'cost JSsvgd/JSipopt/Lsvgd':>26s} | {'wall JSsvgd/JSipopt/Lsvgd':>26s} | {'clock':>5s} {'feas':>4s} {'steps':>5s}")
for robot,rdir,rung,iptag,rd in ROBOTS:
    for task in ("mugshelf","posetip"):
        for start in ("native","paired"):
            js,h1=load(f"results/{rdir}/benchmark/sc_SVGD_{rd}_{robot}_{rung}_svgd_{task}_480_180_{start}_pin2kq","numerical")
            lv,h2=load(f"results/{rdir}/benchmark/sc_SVGD_R20_{robot}_{rung}_svgd_{task}_480_180_{start}_pin2","learned")
            ip,h3=load(f"results/{rdir}/benchmark/{iptag}_{task}_480_180_{start}","numerical")
            assert h1==h2==h3,(robot,task,start)
            keys=sorted(js); a=sum(js[k][0] for k in keys); b=sum(ip[k][0] for k in keys); c=sum(lv[k][0] for k in keys)
            jp,ipp,p1=mcn(js,ip,keys); lp,jp2,p2=mcn(lv,js,keys)
            sh=[k for k in keys if js[k][0] and ip[k][0] and lv[k][0]]
            cj=st.median([js[k][2] for k in sh]); ci=st.median([ip[k][2] for k in sh]); cl=st.median([lv[k][2] for k in sh])
            wj=st.mean([js[k][1] for k in keys]); wi=st.mean([ip[k][1] for k in keys]); wl=st.mean([lv[k][1] for k in keys])
            clock=sum(bool(js[k][3].get("timed_out")) or js[k][3].get("stop")=="wall_clock" for k in keys)
            feas=st.median([js[k][3].get("n_feasible",0) for k in keys]); steps=st.median([js[k][3].get("iterations",0) for k in keys])
            print(f"{robot+' '+task+' '+start:26s} {a:6d} {b:7d} {jp:3d}/{ipp:<3d} {p1:8.1e} | {c:5d} {lp:3d}/{jp2:<3d} {p2:8.1e} | {cj:8.2f}/{ci:7.2f}/{cl:7.2f} | {wj:8.2f}/{wi:7.2f}/{wl:7.2f} | {clock:5d} {feas:4.0f} {steps:5.0f}")
