"""Runaway cells of the cVAE chart rows: |q|_inf, latent norm, per-binding violations and the
shared direction of the exploded q. Run from the repo root on merged summaries."""
import json, glob, sys
import numpy as np
rows = sorted(glob.glob("results/*/benchmark/sc_CVAE_*_480_180_native/summary.json"))
for p in rows:
    d = json.load(open(p)); r = d["records"]["learned"]
    name = p.split("/")[3].replace("sc_CVAE_", "").replace("_480_180_native", "")
    tr = d["metadata"].get("latent_trust_region_effective")
    fail = [c for c in r if not c["feasible"]]
    run = [c for c in fail if c["q"] is not None and np.max(np.abs(c["q"])) > 1000]
    print(f"\n== {name}: {len(fail)} failed, {len(run)} runaways (|q|_inf > 1000); trust radius {tr}")
    if not run:
        continue
    Q = np.array([c["q"] for c in run]); qi = np.max(np.abs(Q), 1)
    zn = np.array([c["z_norm"] for c in run])
    zn_ok = np.array([c["z_norm"] for c in r if c["feasible"]])
    print("  |q|_inf  p0/p50/p100: %.3g %.3g %.3g" % tuple(np.percentile(qi, [0, 50, 100])))
    print("  z_norm runaways p0/p50/p100: %.3g %.3g %.3g ; successes p50/p100: %s" % (
        *np.percentile(zn, [0, 50, 100]), np.round(np.percentile(zn_ok, [50, 100]), 2) if len(zn_ok) else "--"))
    print("  timed_out %d/%d; fail reasons %s" % (sum(c["timed_out"] for c in run), len(run),
          {k: sum(c["fail_reason"] == k for c in run) for k in set(c["fail_reason"] for c in run)}))
    v = run[0]["detail"]["violations"]
    print("  violation keys:", list(v) if isinstance(v, dict) else type(v))
    U = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    C = np.abs(U @ U.T)[np.triu_indices(len(U), 1)]
    print("  pairwise |cos|: p0/p10/p50: %.3f %.3f %.3f" % tuple(np.percentile(C, [0, 10, 50])) if len(C) else "")
    dom = np.argmax(np.abs(Q), 1)
    print("  dominant joint counts:", np.bincount(dom, minlength=Q.shape[1]).tolist())
    # principal direction share
    s = np.linalg.svd(U, compute_uv=False)
    print("  share of sum sq on top-1/top-2 directions: %.3f %.3f" % ((s[0]**2)/np.sum(s**2), (s[:2]**2).sum()/np.sum(s**2)))
    _, _, Vt = np.linalg.svd(U, full_matrices=False)
    print("  top direction:", np.round(Vt[0], 3).tolist())
