"""What do the joint / strain limits do to an IK fibre? (fibre_probe.py now measures
connectivity WITHOUT limits by default; this tool compares the two.) Three stages per target.

`fibre_probe.py` counts components of {u in box : FK(u) = target} by joining sampled
solutions with a greedy fibre-following path (`track`). That count is an UPPER bound: the
path can stall where the fibre is connected. This script takes the same solutions and asks,
for every pair of components the greedy path left apart:

  1. chord, box on   -- a second joining method (the straight chord between the two
                        solutions, projected onto the fibre step by step). A join here
                        means the split was a greedy-path failure, not the fibre.
  2. both, box off   -- both methods with the joint/strain box removed. A join here (and
                        not in 1) means the box cut the fibre: a real split for the robot,
                        caused by its limits.
  3. what is left     -- components neither method joins even without the box.

Calibration: on the iiwa the three GC signs (sign q2, q4, q6) are separated by singular
configurations, so without the box the components should fall to the distinct GC labels
and no lower ("impure" counts merges across labels).

soft16 only: total torsion per solution (sum over segments of kappa_z * segment length),
to test whether components differ by multiples of 2 pi (torsion wrap-around).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import fibre_probe as fp  # noqa: E402
from fibre_probe import Robot, components, pinv, project, track  # noqa: E402


def chord(R, ua, ub, tgt, box=True, ds=0.02, iters=6):
    """Follow the chord ua -> ub, projecting each point onto the fibre (predictor along the
    chord, chord-Newton corrector with the box as an active set). Returns ok [P]."""
    P, n = ua.shape
    lim = 1.0 if box else np.inf
    K = int(max(20, np.ceil(np.linalg.norm(ub - ua, axis=1).max() / ds)))
    du = (ub - ua) / K
    u = ua.copy()
    ok = np.ones(P, bool)
    for _ in range(K):
        idx = np.where(ok)[0]
        if len(idx) == 0:
            break
        U, T = u[idx], tgt[idx]
        Up = np.clip(U + du[idx], -lim, lim)
        J = R.jac(Up, T)
        for _ in range(iters):
            e = R.err(Up, T)
            Jp, _ = pinv(J)
            step = -(Jp @ e[..., None])[..., 0]
            if box:
                act = (np.abs(Up) >= 1 - 1e-12) & (np.sign(step) * np.sign(Up) > 0)
                if act.any():
                    Jp, _ = pinv(J, act)
                    step = -(Jp @ e[..., None])[..., 0]
            Up = np.clip(Up + step, -lim, lim)
        e = np.abs(R.err(Up, T)).max(1)
        moved = np.linalg.norm(Up - U, axis=1)
        good = (e < 1e-8) & (moved < 3 * np.maximum(np.linalg.norm(du[idx], axis=1), 0.01))
        u[idx[good]] = Up[good]
        ok[idx[~good]] = False
    return ok & (np.linalg.norm(u - ub, axis=1) < 0.05)


def merge(R, sols, tgt7, labels, trackers, cross=3, rounds=4):
    """Union components of `labels` by trying `cross` closest pairs per component pair with
    each tracker, until no new join. Returns new labels and pairs tried."""
    parent = {int(l): int(l) for l in labels}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    D = np.linalg.norm(sols[:, None] - sols[None], axis=-1)
    tried = 0
    for _ in range(rounds):
        lab = np.array([find(int(l)) for l in labels])
        roots = sorted(set(lab.tolist()))
        if len(roots) == 1:
            break
        pairs = []
        for x in range(len(roots)):
            for y in range(x + 1, len(roots)):
                A = np.where(lab == roots[x])[0]
                B = np.where(lab == roots[y])[0]
                sub = D[np.ix_(A, B)]
                for f in np.argsort(sub, axis=None)[:cross]:
                    i, j = np.unravel_index(f, sub.shape)
                    pairs.append((int(A[i]), int(B[j])))
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        T = np.repeat(tgt7[None], len(pairs), 0)
        ok = np.zeros(len(pairs), bool)
        for tr in trackers:
            ok |= tr(R, sols[a], sols[b], T)
        tried += len(pairs)
        joined = False
        for (i, j), o in zip(pairs, ok):
            if o and find(int(labels[i])) != find(int(labels[j])):
                parent[find(int(labels[i]))] = find(int(labels[j]))
                joined = True
        if not joined:
            break
    return np.array([find(int(l)) for l in labels]), tried


def ncomp(lab):
    return int(len(set(np.asarray(lab).tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--nflow", type=int, default=32)
    ap.add_argument("--nrand", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    R = Robot(a.robot)
    n = R.ndof
    twist_idx = None
    if a.robot == "soft16":
        spec = R.jrl._spec
        seg_len = spec.total_length / spec.num_segments
        per = len(spec.strain_basis)
        k = 2  # (kx, ky, kz, sz): kz is the third strain of each segment
        twist_idx = [s * per + k for s in range(spec.num_segments)]
        twist_scale = spec.strain_limits[k] * seg_len
    res = {"robot": a.robot, "targets": []}
    for t in range(a.targets):
        t0 = time.time()
        u_star = rng.uniform(-1, 1, (1, n))
        tgt7 = R.fk(u_star)[0]
        uf = R.decode(rng.standard_normal((a.nflow, R.width)), np.repeat(tgt7[None], a.nflow, 0))
        ur = rng.uniform(-1, 1, (a.nrand, n))
        ## This tool studies what the BOX does, so solutions and the first two stages run
        ## with the probe's LIMITS on; the third stage switches them off.
        fp.LIMITS = True
        up, okp = project(R, np.clip(np.vstack([uf, ur]), -1, 1),
                          np.repeat(tgt7[None], a.nflow + a.nrand, 0))
        sols = np.vstack([up[okp], u_star])
        keep = []
        for i in range(len(sols)):
            if all(np.linalg.norm(sols[i] - sols[j]) > 1e-2 for j in keep):
                keep.append(i)
        sols = sols[keep]
        lab0, _ = components(R, sols, tgt7)  # greedy OR chord since the probe's revision
        lab1, tried1 = merge(R, sols, tgt7, lab0, [chord])
        fp.LIMITS = False  # track() reads it: no box, revolute joints wrapped
        nobox = [lambda R_, x, y, T: track(R_, x, y, T),
                 lambda R_, x, y, T: chord(R_, x, y, T, box=False)]
        lab2, tried2 = merge(R, sols, tgt7, lab1, nobox)
        rec = {"n_solutions": int(len(sols)), "greedy_box": ncomp(lab0), "plus_chord_box": ncomp(lab1),
               "nobox": ncomp(lab2), "pairs_chord": tried1, "pairs_nobox": tried2,
               "at_limit_coords_median": float(np.median((np.abs(sols) > 1 - 1e-6).sum(1))),
               "solutions": sols.round(6).tolist(), "labels": {"greedy_box": lab0.tolist(),
               "plus_chord_box": lab1.tolist(), "nobox": lab2.tolist()}}
        if R.labeler is not None:
            gl = [R.labeler(R.q(x)) for x in sols]
            rec["distinct_labels"] = len(set(gl))
            for key, lab in (("greedy_box", lab0), ("plus_chord_box", lab1), ("nobox", lab2)):
                rec[f"impure_{key}"] = int(sum(len({gl[i] for i in np.where(lab == r)[0]}) > 1
                                               for r in set(lab.tolist())))
        if twist_idx is not None:
            tw = sols[:, twist_idx].sum(1) * twist_scale
            rec["twist"] = tw.round(4).tolist()
            roots = sorted(set(lab1.tolist()))
            means = [float(np.mean(tw[lab1 == r])) for r in roots]
            spans = [float(np.ptp(tw[lab1 == r])) for r in roots]
            rec["twist_by_component_box"] = {"mean": means, "span": spans,
                                             "sizes": [int((lab1 == r).sum()) for r in roots]}
        rec["seconds"] = round(time.time() - t0, 1)
        res["targets"].append(rec)
        print(f"target {t}: " + json.dumps({k: v for k, v in rec.items()
                                            if k not in ("solutions", "labels", "twist")}), flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
