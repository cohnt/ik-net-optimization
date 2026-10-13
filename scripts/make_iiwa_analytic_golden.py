"""Regenerate tests/data/iiwa_analytic_codebase_golden.npz from ../codebase's iiwa map.

The iiwa closed form in src/iiwa_analytic_ik.py is a port of ../codebase's maintained
`Analytic_IK_7DoF`. The parity check runs against a COMMITTED golden file rather than a
live import, for the soft PCS arm's reason: a test that runs only where the sibling repo
sits beside this one is a check that silently stops running elsewhere.

Run FROM ../codebase (it imports that repo's `src`), writing nowhere inside it:

    cd ../codebase && PYTHONDONTWRITEBYTECODE=1 python ../learned-ik/scripts/make_iiwa_analytic_golden.py \
        ../learned-ik/tests/data/iiwa_analytic_codebase_golden.npz

Generated 2026-10-12 against ../codebase at ed58edff.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from src.iiwa_analytic_ik import Analytic_IK_7DoF, iiwa_limits_lower as lo, iiwa_limits_upper as hi  # noqa: E402
from pydrake.all import RigidTransform  # noqa: E402

B = Analytic_IK_7DoF()
rng = np.random.default_rng(20261012)
Q, X, G, P, QI, U = [], [], [], [], [], []
for _ in range(200):
    q = rng.uniform(lo, hi)
    T = B.FK(q)
    gc = B.GC(q)
    psi = float(B.psi(q))
    psi_eval = psi + rng.normal(0, 0.3)  # also off the recovered arm angle
    QI.append(np.asarray(B.IK(RigidTransform(T), gc, psi_eval), float))
    U.append(np.asarray(B.IK(RigidTransform(T), gc, psi_eval, return_unclipped_vals=True), float))
    Q.append(q)
    X.append(T)
    G.append(gc)
    P.append([psi, psi_eval])
np.savez(sys.argv[1], q=np.array(Q), X=np.array(X), gc=np.array(G), psi=np.array(P),
         q_ik=np.array(QI), u=np.array(U))
print("wrote", sys.argv[1])
