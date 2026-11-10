"""Merge <dir>/<robot>.s<seed>.json from fibre_probe.py and print one block per robot.

    python scripts/fibre/summarise_fibre.py <dir>
"""
import glob
import json
import sys

import numpy as np

ROBOTS = ("soft9", "soft12", "soft16", "gvs_pushrod9_o1", "gvs_pushrod9_o2", "panda", "iiwa14")
med = lambda xs: float(np.median(xs))


def lines_block(name, L):
    nl = sum(l["lines"] for l in L)
    jl = sum(l["lines_with_jump"] for l in L)
    print(f"{name} ({nl}): median step {med([l['median_step'] for l in L]):.4f}; jumpiness "
          f"(max/median step) median {med([l['jumpiness_median'] for l in L]):.2f}, p90 "
          f"{med([l['jumpiness_p90'] for l in L]):.1f}; lines with a jump (>10x median, >0.05) "
          f"{jl}/{nl} = {jl / max(nl, 1):.1%}; jumps/line {sum(l['jumps_total'] for l in L) / max(nl, 1):.2f}")
    print(f"  pose err: pos median {med([l['pos_err_median_mm'] for l in L]):.1f} mm, p99 "
          f"{med([l['pos_err_p99_mm'] for l in L]):.1f} mm; rot median "
          f"{med([l['rot_err_median_deg'] for l in L]):.2f} deg; decoded outside the box "
          f"{med([l['out_of_box_frac'] for l in L]):.0%}")
    if "branch_flips_total" in L[0]:
        fl = sum(l["branch_flips_total"] for l in L)
        print(f"  branch flips {fl} on {sum(l['lines_with_flip'] for l in L)}/{nl} lines; "
              f"flips that are jumps {sum(l['flips_at_jump'] for l in L)}/{fl}")


def main(D):
    for r in ROBOTS:
        T, checks = [], []
        for f in sorted(glob.glob(f"{D}/{r}.s*.json")):
            d = json.load(open(f))
            T += d["targets"]
            checks.append(d.get("jac_check_rel"))
        if not T:
            continue
        C = [t["conn"] for t in T if "conn" in t]
        comps = np.array([c["components"] for c in C])
        print(f"\n== {r}: {len(T)} targets; exact-vs-FD Jacobian max rel diff "
              f"{max(x for x in checks if x is not None):.1e}")
        if C:
            print(f"connectivity: solutions/target median {med([c['n_solutions'] for c in C]):.0f}; "
                  f"components median {med(comps):.0f} (p10 {np.percentile(comps, 10):.0f}, "
                  f"p90 {np.percentile(comps, 90):.0f}, max {comps.max()}); ONE component on "
                  f"{(comps == 1).sum()}/{len(C)}; largest component share median "
                  f"{med([c['component_sizes'][0] / c['n_solutions'] for c in C]):.2f}; "
                  f"flow solutions span median {med([c['flow_components'] for c in C]):.0f} components")
            if "labels" in C[0]:
                # pieces that contain at least one solution inside the joint / strain / force box
                touch, far = [], []
                for c in C:
                    U, lab = np.array(c["solutions"]), np.array(c["labels"])
                    inbox = (np.abs(U) <= 1 + 1e-9).all(1)
                    roots = set(lab.tolist())
                    touch.append(sum(inbox[lab == r].any() for r in roots))
                    far.append(float(np.median([np.abs(U[lab == r]).max(1).min() for r in roots
                                                if not inbox[lab == r].any()] or [np.nan])))
                touch = np.array(touch)
                print(f"  limits {'ON' if d.get('limits') else 'OFF'}; method {C[0].get('method')}"
                      + (f", loops closed {sum(c['loops_closed'] for c in C)}, open {sum(c['loops_open'] for c in C)}"
                         if C[0].get("method") == "trace" else "")
                      + f"; solutions outside the box median {med([c['outside_box_solutions'] for c in C]):.0f}")
                print(f"  pieces containing an in-box solution: median {med(touch):.0f}, ONE on "
                      f"{(touch == 1).sum()}/{len(C)}; pieces entirely outside the box: median "
                      f"{med(comps - touch):.0f}, their closest point max|u| median {np.nanmedian(far):.2f}")
            if "distinct_branches" in C[0]:
                print(f"  analytic branches/target median {med([c['distinct_branches'] for c in C]):.0f}; "
                      f"impure components (tracker merged two branches) {sum(c['impure_components'] for c in C)}"
                      f"/{comps.sum()}; branches split into >1 component {sum(c['branches_split'] for c in C)}"
                      f"/{sum(c['distinct_branches'] for c in C)}")
        if "latent" in T[0]:
            lines_block("latent lines, c fixed", [t["latent"] for t in T])
        if "pose_lines" in T[0]:
            lines_block("pose lines, z fixed", [t["pose_lines"] for t in T])


if __name__ == "__main__":
    main(sys.argv[1])
