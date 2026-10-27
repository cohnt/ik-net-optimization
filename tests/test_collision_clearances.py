"""`ClearanceEvaluator` (one batched `CalcRobotClearances` call) against the program's own row.

RUNS ONLY WHERE THE BINDING EXISTS. `CollisionChecker.CalcRobotClearances` is in Drake master
(#25073, 6ad25878d5, 2026-10-12) and so in the default local pydrake; a pydrake that predates it
(the record's nightly 20260918) gets only the check that `svgd_collision_backend="clearances"` is
REFUSED at options construction with the reason, prints SKIP and exits 0.

    .venv/bin/python tests/test_collision_clearances.py

THE PROGRAM IS THE ORACLE, as in `tests/test_collision_backend.py`: every value and gradient is
compared with `program.collision_free_constraint_eval.Eval(InitializeAutoDiff(q))` times
`collision_row_scale`, on both Panda scenes (hardened pose; grasp with the target mug welded as
the benchmark welds it), over 64 uniform draws in the joint limits plus NaN rows. The two paths
are the same functional computed by different code -- Drake's AutoDiffXd penalty with
`CalcJacobianTranslationalVelocity` against a numpy replica of it with the checker's
`CalcJacobianPositionVector` -- so the gradient is compared to a tolerance relative to its
largest entry (1e-12; measured 1.7e-14 pose, 4.1e-14 grasp) and the measured difference is
printed. The VALUE came out bitwise equal on every row measured; it is held to 1e-12 absolute. A second spec with a 0.1 mm influence offset exercises the
"no pair within the influence distance" branch (the fielded 0.1 m leaves no Panda configuration
without one: the wrist links are always within it of each other).

Also pinned: results are BITWISE independent of the thread count (Max vs 1) and of the batch
(a configuration alone vs inside a batch of 64), and `need_grad=False` returns the same values.
Tiers (tests/_tiers.py): the default `fast` runs in ~8 s on one CPU core; no `slow` group.
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import _tiers                                                             # noqa: E402

from src.svgd.collision_backend import clearances_unavailable_reason   # noqa: E402

TOL_VALUE = 1e-12
TOL_GRAD_REL = 1e-12   # of the largest gradient entry: the two Jacobian paths differ in roundoff


def _skip_branch(reason):
    from src.generic_program import ProgramOptions
    try:
        ProgramOptions(svgd_collision_backend="clearances")
    except ValueError as e:
        assert "CalcRobotClearances" in str(e), str(e)
        print(f"PASS svgd_collision_backend='clearances' is refused here: {e}")
    else:
        raise AssertionError("clearances accepted by a pydrake without the binding")
    print(f"SKIP tests/test_collision_clearances.py: {reason}")


def _min_clearance(ev, Q):
    """Per row: the smallest reported distance (inf when no pair is within influence)."""
    out = np.full(Q.shape[0], np.inf)
    for i, r in enumerate(ev.checker.CalcRobotClearances(list(Q), ev.influence)):
        if r.size():
            out[i] = float(np.min(r.distances()))
    return out


def _compare(label, v, g, ref_v, ref_g):
    finite = np.isfinite(ref_v)
    assert np.array_equal(np.isnan(v), ~finite) and np.array_equal(np.isnan(g).any(1), ~finite)
    dv = float(np.max(np.abs(v[finite] - ref_v[finite])))
    dg = float(np.max(np.abs(g[finite] - ref_g[finite])))
    rel = float(np.max(np.abs(g[finite] - ref_g[finite]))
                / max(1e-300, np.max(np.abs(ref_g[finite]))))
    print(f"     {label}: max |dvalue| {dv:.2e} (|value| <= {np.max(np.abs(ref_v[finite])):.3g}), "
          f"max |dgrad| {dg:.2e} (|grad| <= {np.max(np.abs(ref_g[finite])):.3g}, rel {rel:.1e})")
    return dv, dg, rel


def test_parity_with_the_program():
    from tests.test_collision_backend import _draw, _program, _reference, _spec
    from src.svgd.collision_backend import ClearanceEvaluator, CollisionEvaluator
    rng = np.random.default_rng(2026)
    worst = [0.0, 0.0]
    for task in ("pose", "mug"):
        program = _program("panda", task)
        spec = _spec(program)
        Q = _draw(program, 64, rng, nan_rows=2)
        ev = ClearanceEvaluator.from_program(program, spec=spec)
        ref_v, ref_g = _reference(program, Q)
        finite = np.isfinite(ref_v)
        dmin = _min_clearance(ev, Q[finite])
        n_pen, n_none = int((dmin < 0).sum()), int(np.isinf(dmin).sum())
        print(f"     panda {task}: {ev.max_num_values} candidate pairs (= checker's "
              f"MaxNumDistances), robot {ev.robot_instance_names}, {ev.num_threads} threads; "
              f"{int(finite.sum())} finite rows, {n_pen} penetrating (min distance < 0), "
              f"{int((ref_v[finite] > spec.row_scale).sum())} above the row's bound, "
              f"{n_none} with no pair inside the influence distance")
        assert n_pen > 0, "the draw must contain penetrating configurations"
        v, g = ev.eval(Q)
        dv, dg, rel = _compare(f"panda {task} clearances vs program", v, g, ref_v, ref_g)
        worst = [max(worst[0], dv), max(worst[1], rel)]
        assert dv <= TOL_VALUE and rel <= TOL_GRAD_REL, (task, dv, dg, rel)
        v2, none = ev.eval(Q, need_grad=False)
        assert none is None and np.array_equal(v2, v, equal_nan=True)
        ## And against the autodiff evaluator itself (which equals the program bitwise).
        av, ag = CollisionEvaluator.from_program(program).eval(Q)
        assert np.array_equal(av, ref_v, equal_nan=True) and np.array_equal(ag, ref_g, equal_nan=True)

        ## A 0.1 mm influence offset: most configurations have NO pair inside it.
        from dataclasses import replace
        tiny = replace(spec, influence_distance_offset=1e-4)
        ev_t, ad_t = ClearanceEvaluator.from_spec(tiny), CollisionEvaluator.from_spec(tiny)
        rv, rg = ad_t.eval(Q)
        dmin_t = _min_clearance(ev_t, Q[finite])
        n_none_t = int(np.isinf(dmin_t).sum())
        print(f"     panda {task}, influence offset 1e-4: {n_none_t} of {int(finite.sum())} rows "
              f"with no pair inside the influence distance, {int((dmin_t < 0).sum())} penetrating")
        assert n_none_t > 0 and n_none_t < finite.sum()
        tv, tg = ev_t.eval(Q)
        dv, dg, rel = _compare(f"panda {task} (offset 1e-4) clearances vs autodiff", tv, tg, rv, rg)
        worst = [max(worst[0], dv), max(worst[1], rel)]
        assert dv <= TOL_VALUE and rel <= TOL_GRAD_REL, (task, dv, dg, rel)
        none_rows = np.flatnonzero(finite)[np.isinf(dmin_t)]
        assert np.array_equal(tv[none_rows], rv[none_rows]) and not tg[none_rows].any()
    print(f"PASS the clearances row equals the program's binding: worst |dvalue| {worst[0]:.2e}, "
          f"worst |dgrad| / max |grad| {worst[1]:.1e}, on both Panda scenes")


def test_threads_and_batch_do_not_change_the_result():
    from pydrake.common import Parallelism
    from tests.test_collision_backend import _draw, _program, _spec
    from src.svgd.collision_backend import ClearanceEvaluator
    rng = np.random.default_rng(7)
    for task in ("pose", "mug"):
        program = _program("panda", task)
        spec = _spec(program)
        Q = _draw(program, 64, rng)
        par = ClearanceEvaluator.from_program(program, spec=spec)
        one = ClearanceEvaluator.from_program(program, spec=spec, parallelism=Parallelism(1))
        assert one.num_threads == 1 and par.num_threads >= 1
        vp, gp = par.eval(Q)
        v1, g1 = one.eval(Q)
        assert np.array_equal(vp, v1) and np.array_equal(gp, g1), task
        for i in (0, 17, 63):
            vi, gi = par.eval(Q[i:i + 1])
            assert np.array_equal(vi[0], vp[i]) and np.array_equal(gi[0], gp[i]), (task, i)
        print(f"     panda {task}: {par.num_threads} threads == 1 thread, and single == batched, "
              f"bitwise")
    print("PASS the clearances row is independent of the thread count and of the batch")


def test_collision_row_through_autograd():
    import torch
    from tests.test_collision_backend import _draw, _program, _spec
    from src.svgd.collision_backend import ClearanceEvaluator, collision_row
    program = _program("panda", "mug")
    ev = ClearanceEvaluator.from_program(program, spec=_spec(program))
    Q = _draw(program, 8, np.random.default_rng(3))
    v, g = ev.eval(Q)
    q = torch.tensor(Q, dtype=torch.float32, requires_grad=True)
    y = collision_row(q, ev)
    w = torch.arange(1.0, 9.0)
    (w * y).sum().backward()
    assert torch.allclose(y.double(), torch.as_tensor(v), rtol=1e-6)
    assert torch.allclose(q.grad.double(), w[:, None].double() * torch.as_tensor(g), rtol=1e-5,
                          atol=1e-6)
    print("PASS collision_row runs the clearances evaluator through autograd")


GROUPS = {"fast": [test_parity_with_the_program, test_threads_and_batch_do_not_change_the_result,
                   test_collision_row_through_autograd]}


def main():
    reason = clearances_unavailable_reason()
    if reason is not None:
        _skip_branch(reason)
        sys.exit(0)
    import pydrake
    print(f"     pydrake: {os.path.dirname(pydrake.__file__)}")
    t0 = time.perf_counter()
    _tiers.run(GROUPS)
    print(f"ALL PASS ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
