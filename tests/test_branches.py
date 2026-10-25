"""Guards for the numpy branch classifiers (`src/branches.py`, LOInK phase 2, 2026-10-24).

There is no test runner in this repo; run this by hand, CPU only, under the local memory scope
(seconds; it builds no program and loads no robot):

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=20G \
        -p MemorySwapMax=0 .venv/bin/python tests/test_branches.py

`src/branches.py` is a pydrake-free copy of the analytic arms' branch functions, because the
single-branch LOInK datasets and trainer run where pydrake is never imported. A copy drifts
silently, so this pins it to the originals:

1. The copied Panda DH table equals `Analytic_IK_Panda`'s constructor defaults.
2. `PandaBranch == Analytic_IK_Panda().gc(q, branches=3)` and
   `IiwaBranch == Analytic_IK_7DoF().GC(q)` row by row on 4000 seeded in-limit q.
3. `NominalBranch` equals the programs' own `RecoverBranch(Q_NOMINAL)` (the analytic arms'
   method, called on a stub carrying only the analytic solver -- constructing a program
   builds a diagram, which this check does not need), with `Q_NOMINAL` read off the program
   classes, and equals the values the plan recorded (Panda (1, 2, 1), iiwa (1, -1, 1)).
4. `OnBranch` accepts the nominal configuration, and its acceptance on uniform in-limit q is
   what the branch structure predicts (iiwa ~1/8; Panda ~1/2 x ~1/2 x ~0.9).
"""

import os
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO)

from src import branches  # noqa: E402
from src.iiwa_analytic_ik import (  # noqa: E402
    Analytic_IK_7DoF, iiwa_limits_lower, iiwa_limits_upper)
from src.panda_analytic_ik import (  # noqa: E402
    Analytic_IK_Panda, panda_a, panda_alpha, panda_d, panda_limits_lower, panda_limits_upper)

N = 4000


def _uniform(lower, upper, n, seed):
    rng = np.random.default_rng(seed)
    return lower + (upper - lower) * rng.random((n, len(lower)))


def test_dh_table_pinned():
    ik = Analytic_IK_Panda()
    for name, copy, orig, default in (("alpha", branches.PANDA_DH_ALPHA, ik.alpha, panda_alpha),
                                      ("a", branches.PANDA_DH_A, ik.a, panda_a),
                                      ("d", branches.PANDA_DH_D, ik.d, panda_d)):
        assert np.array_equal(copy, orig) and np.array_equal(copy, default), (
            f"src/branches.py's Panda DH {name} has drifted from Analytic_IK_Panda's default")
    q3_add = np.arctan2(ik.d[2], ik.a[3]) + np.arctan2(ik.d[4], abs(ik.a[4]))
    assert branches.PANDA_Q3_ADD == q3_add
    print("PASS the copied Panda DH table and q3_add equal Analytic_IK_Panda's defaults")


def test_panda_classifier():
    ik = Analytic_IK_Panda()
    q = _uniform(panda_limits_lower, panda_limits_upper, N, seed=0)
    got = branches.PandaBranch(q)
    want = np.array([ik.gc(qi, branches=3) for qi in q]).astype(int)
    bad = np.flatnonzero(np.any(got != want, axis=1))
    assert bad.size == 0, f"PandaBranch != gc(q, 3) on {bad.size}/{N} rows, e.g. {q[bad[0]]}"
    # Every label occurs, or the agreement would be vacuous for the missing one.
    assert {tuple(r) for r in got} == {(b, c, a) for b in (1, 2) for c in (1, 2) for a in (1, -1)}
    # A single (7,) configuration returns one row.
    assert branches.PandaBranch(q[0]).shape == (1, 3)
    print(f"PASS PandaBranch == Analytic_IK_Panda().gc(q, branches=3) on {N}/{N} rows "
          "(all 8 labels present)")


def test_iiwa_classifier():
    ik = Analytic_IK_7DoF()
    q = _uniform(iiwa_limits_lower, iiwa_limits_upper, N, seed=1)
    q[:7, [1, 3, 5]] = 0.0  # the 0 -> +1 convention
    got = branches.IiwaBranch(q)
    want = np.array([ik.GC(qi) for qi in q]).astype(int)
    bad = np.flatnonzero(np.any(got != want, axis=1))
    assert bad.size == 0, f"IiwaBranch != GC(q) on {bad.size}/{N} rows, e.g. {q[bad[0]]}"
    assert len({tuple(r) for r in got}) == 8
    print(f"PASS IiwaBranch == Analytic_IK_7DoF().GC(q) on {N}/{N} rows (all 8 labels, zeros -> +1)")


def test_nominal_branch():
    from src.iiwa_program import Iiwa14IKProgramAnalytic
    from src.panda_program import PandaIKProgramAnalytic

    panda_stub = SimpleNamespace(analytic_ik=Analytic_IK_Panda(),
                                 options=SimpleNamespace(analytic_branches=8))
    iiwa_stub = SimpleNamespace(analytic_ik=Analytic_IK_7DoF())
    want_panda = np.asarray(PandaIKProgramAnalytic.RecoverBranch(
        panda_stub, PandaIKProgramAnalytic.Q_NOMINAL)).astype(int)
    want_iiwa = np.asarray(Iiwa14IKProgramAnalytic.RecoverBranch(
        iiwa_stub, Iiwa14IKProgramAnalytic.Q_NOMINAL)).astype(int)
    got_panda, got_iiwa = branches.NominalBranch("panda"), branches.NominalBranch("iiwa14")
    assert np.array_equal(got_panda, want_panda), (got_panda, want_panda)
    assert np.array_equal(got_iiwa, want_iiwa), (got_iiwa, want_iiwa)
    assert tuple(got_panda) == (1, 2, 1) and tuple(got_iiwa) == (1, -1, 1)
    assert branches.BranchString(got_panda) == "(1, 2, 1)"
    assert branches.BranchString(None) == "none"
    assert np.array_equal(branches.ResolveBranch("panda", "nominal"), got_panda)
    print(f"PASS NominalBranch == RecoverBranch(Q_NOMINAL): panda {branches.BranchString(got_panda)},"
          f" iiwa14 {branches.BranchString(got_iiwa)}")


def test_on_branch():
    for robot, q_nom, lower, upper, lo, hi in (
            ("panda", branches.PANDA_Q_NOMINAL, panda_limits_lower, panda_limits_upper, 0.15, 0.30),
            ("iiwa14", branches.IIWA_Q_NOMINAL, iiwa_limits_lower, iiwa_limits_upper, 0.09, 0.16)):
        assert branches.OnBranch(robot, q_nom).tolist() == [True], robot
        q = _uniform(lower, upper, 20000, seed=2)
        frac = branches.OnBranch(robot, q).mean()
        assert lo < frac < hi, f"{robot}: OnBranch acceptance {frac:.3f} outside ({lo}, {hi})"
        # An explicit branch and the default agree.
        explicit = branches.OnBranch(robot, q, branches.NominalBranch(robot))
        assert np.array_equal(explicit, branches.OnBranch(robot, q))
        print(f"PASS OnBranch({robot}) accepts the nominal configuration; uniform acceptance "
              f"{frac:.3f}")
    try:
        branches.Branch("screw7_p050", np.zeros(7))
    except ValueError:
        print("PASS Branch refuses a robot with no classifier")
    else:
        raise AssertionError("Branch accepted a robot with no classifier")


if __name__ == "__main__":
    t0 = time.monotonic()
    test_dh_table_pinned()
    test_panda_classifier()
    test_iiwa_classifier()
    test_nominal_branch()
    test_on_branch()
    print(f"ALL PASS ({time.monotonic() - t0:.1f} s)")
