"""IK branch classifiers for the rigid 7-DoF arms, vectorised and numpy only.

A BRANCH is the discrete part of a 7-DoF arm's IK solution: at a fixed end-effector pose the
solution set is several disconnected arcs, and each arc is labelled by a few signs that the
redundancy parameter cannot change continuously. The analytic arms already name them:

  * Panda (He-Liu, `src/panda_analytic_ik.py:Analytic_IK_Panda.gc(q, branches=3)`):
    (B, C, A) -- B in {1, 2} the wrist (sign of O2O6 . x5), C in {1, 2} the shoulder
    (1 iff q2 > 0), A in {+1, -1} the elbow (+1 iff q4 < q3_add - pi).
  * iiwa (Faria S-R-S, `src/iiwa_analytic_ik.py:Analytic_IK_7DoF.GC`):
    (sign q2, sign q4, sign q6), each +-1 with 0 -> +1.

Why this exists (LOInK phase 2, Thomas 2026-10-24): a LOInK chart is one bijection of R^n, so
it carries one connected line per task value; on the rigid arms that never fits the several
arcs of the full solution set (`docs/notes-chart.md`, "Training outcome (2026-10-23)"). The
phase-2 charts are trained on ONE branch, where the solution set is one arc parameterised by
the redundancy. The branch is the NOMINAL configuration's (`src/nominal_configurations.py`),
COMPUTED here from the constant, never hard-coded: the matched training cost's minimiser lies
on it, and it is the branch the analytic arm seeds from the home pose.

Numpy only, deliberately: training jobs never import pydrake, and the analytic classes do.
So the Panda's DH table is a PINNED COPY of `Analytic_IK_Panda`'s constructor defaults, and
`tests/test_branches.py` asserts both the table's equality and the classifiers' row-by-row
agreement with the analytic classes.
"""

import numpy as np

from src.nominal_configurations import IIWA_Q_NOMINAL, PANDA_Q_NOMINAL

#: Pinned copy of `src/panda_analytic_ik.py`'s `panda_alpha`, `panda_a`, `panda_d` (the
#: `Analytic_IK_Panda` defaults); `tests/test_branches.py` asserts equality.
PANDA_DH_ALPHA = np.array([0., -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0., 0.])
PANDA_DH_A = np.array([0., 0., 0, 0.0825, -0.0825, 0, 0.088, 0., 0.])
PANDA_DH_D = np.array([0.333, 0., 0.316, 0., 0.384, 0., 0, 0.107, 0.1034])

#: Panda's elbow threshold offset: the two elbow relations partition at q4 = q3_add - pi.
PANDA_Q3_ADD = (np.arctan2(PANDA_DH_D[2], PANDA_DH_A[3])
                + np.arctan2(PANDA_DH_D[4], abs(PANDA_DH_A[4])))

#: The datasets the single-branch charts train on (built by
#: scripts/training/build_branch_dataset.py), keyed by jrl robot name.
BRANCH_DATASET = {"panda": "panda_branch", "iiwa14": "iiwa14_branch"}

_Q_NOMINAL = {"panda": PANDA_Q_NOMINAL, "iiwa14": IIWA_Q_NOMINAL}


def _as_batch(q):
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] < 7:
        raise ValueError(f"expected configurations of shape (N, 7) or (7,), got {q.shape}")
    return q[:, :7]


def _panda_dh(theta, i):
    """Batched `Analytic_IK_Panda.Ts[i](theta)`, the same (modified) DH convention: (N, 4, 4)."""
    al, a, d = PANDA_DH_ALPHA[i], PANDA_DH_A[i], PANDA_DH_D[i]
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(al), np.sin(al)
    T = np.zeros(theta.shape + (4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 3] = c, -s, a
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = s * ca, c * ca, -sa, -d * sa
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = s * sa, c * sa, ca, d * ca
    T[:, 3, 3] = 1.0
    return T


def PandaBranch(q):
    """(N, 3) int array (B, C, A), row by row equal to `Analytic_IK_Panda().gc(q, branches=3)`."""
    q = _as_batch(q)
    out = np.zeros((q.shape[0], 3), dtype=int)
    out[:, 1] = np.where(q[:, 1] > 0, 1, 2)
    T5 = _panda_dh(q[:, 0], 0)
    for i in range(1, 5):
        T5 = T5 @ _panda_dh(q[:, i], i)
    T6 = T5 @ _panda_dh(q[:, 5], 5)
    O2O6 = T6[:, :3, 3] - np.array([0.0, 0.0, PANDA_DH_D[0]])
    x5 = T5[:, :3, 0]
    out[:, 0] = np.where(np.einsum("ij,ij->i", O2O6, x5) < 0, 1, 2)
    out[:, 2] = np.where(q[:, 3] < PANDA_Q3_ADD - np.pi, 1, -1)
    return out


def IiwaBranch(q):
    """(N, 3) int array (sign q2, sign q4, sign q6), 0 -> +1, as `Analytic_IK_7DoF.GC`."""
    q = _as_batch(q)
    return np.where(q[:, [1, 3, 5]] >= 0, 1, -1).astype(int)


def Branch(robot_name, q):
    """The branch of each configuration, (N, 3), for a jrl robot name ("panda", "iiwa14")."""
    if robot_name == "panda":
        return PandaBranch(q)
    if robot_name == "iiwa14":
        return IiwaBranch(q)
    raise ValueError(f"no branch classifier for robot {robot_name!r} (only panda, iiwa14)")


def NominalBranch(robot_name):
    """The nominal configuration's branch, computed from `src/nominal_configurations.py`: (3,)."""
    if robot_name not in _Q_NOMINAL:
        raise ValueError(f"no nominal branch for robot {robot_name!r} (only panda, iiwa14)")
    return Branch(robot_name, _Q_NOMINAL[robot_name])[0]


def ResolveBranch(robot_name, branch):
    """`branch` as a (3,) int array: None or "nominal" -> `NominalBranch(robot_name)`."""
    if branch is None or (isinstance(branch, str) and branch == "nominal"):
        return NominalBranch(robot_name)
    branch = np.asarray(branch, dtype=int).reshape(-1)
    if branch.shape != (3,):
        raise ValueError(f"a branch is three integers, got {branch!r}")
    return branch


def OnBranch(robot_name, q, branch=None):
    """Boolean mask, (N,): which configurations lie on `branch` (default: the nominal one)."""
    b = ResolveBranch(robot_name, branch)
    return np.all(Branch(robot_name, q) == b[None, :], axis=1)


def BranchString(branch):
    """A compact, parseable rendering for info files and logs, e.g. "(1, 2, 1)"."""
    if branch is None:
        return "none"
    return "(" + ", ".join(str(int(v)) for v in np.asarray(branch).reshape(-1)) + ")"
