"""`src/svgd/collision_backend.py` against the program's own collision binding.

THE PROGRAM IS THE ORACLE. Every number the pool returns is compared with
`program.collision_free_constraint_eval.Eval(InitializeAutoDiff(q))` scaled by
`collision_row_scale`, on the very scene the solver would run on -- the hardened pose scene and
a grasp scene with the target mug welded exactly as `GenerateDiagramWithMug` welds it. The
comparison is BITWISE (`np.array_equal`), not approximate: the pool runs the same Drake code on
a scene rebuilt from a picklable description, so any difference at all would mean the rebuilt
scene is not the solve scene. That the mug weld survives this is the point of carrying its
rotation matrix rather than a quaternion (see the module docstring).

The boolean C++-parallel checker is held to the relation that actually holds. Drake's penalty
is `SmoothOverMax(...)`, which over-approximates the maximum, so `min distance < bound` implies
`penalty > 1` exactly, while the converse fails in a band above the bound whose worst case is
`log(n_pairs)/alpha` in penalty units (~2 mm here; one of 16 draws in the smoke sat at
+1.16 mm with penalty 1.0024). The test asserts the implication, asserts agreement outside a
2.5 mm band, and REPORTS the band count rather than hiding it.

No pytest config in this repo, so these are plain `test_*` functions with a `__main__` driver.
Needs the Panda and iiwa `n6` charts under `models/` (the joint-space programs still load a
network, because the base class does), and spawns worker processes, so it runs as a script.
"""

import os
import pickle
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.autodiffutils import ExtractGradient, ExtractValue, InitializeAutoDiff  # noqa: E402

from src.flow_loading import LoadFlowSolver                                  # noqa: E402
from src.generic_program import ProgramOptions                               # noqa: E402
from src.iiwa_program import Iiwa14IKProgramNumerical, IiwaMugProgramNumerical  # noqa: E402
from src.panda_program import PandaIKProgramNumerical, PandaMugProgramNumerical  # noqa: E402
from src.svgd.collision_backend import (CollisionRow, DrakeCollisionPool,     # noqa: E402
                                        ParallelCollisionChecker, SceneSpec,
                                        collision_row, measure_pool)
from src.target_screening import SCENES, SceneFile                           # noqa: E402
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints, RepoDir  # noqa: E402

CHECKPOINTS = {
    "panda": "models/panda/panda__n6__step620000.pkl",
    "iiwa": "models/iiwa14/iiwa14__n6__step620000.pkl",
}
NUMERICAL = {
    ("panda", "pose"): PandaIKProgramNumerical, ("panda", "mug"): PandaMugProgramNumerical,
    ("iiwa", "pose"): Iiwa14IKProgramNumerical, ("iiwa", "mug"): IiwaMugProgramNumerical,
}
MUG_SEEDS = {"panda": 101, "iiwa": 202}
_SOLVERS = {}
_PROGRAMS = {}


def _solver(robot):
    if robot not in _SOLVERS:
        with HiddenPrints():
            _SOLVERS[robot] = LoadFlowSolver(
                "iiwa14" if robot == "iiwa" else robot,
                os.path.join(RepoDir(), CHECKPOINTS[robot]))
    return _SOLVERS[robot]


def _program(robot, task):
    """A joint-space program on the hardened scene; the grasp one with a welded target mug.

    The mug is placed the way the benchmark places it: at the grasp frame of a sampled
    configuration of a MUG-type sampler, so `program.target_mug.middle` is the very transform
    the weld was formed from.
    """
    key = (robot, task)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    yaml = SceneFile(robot, task, "hardened")
    options = ProgramOptions()
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=yaml)
        if task == "pose":
            program = NUMERICAL[key](diagram, options=options, model=_solver(robot))
            program.create_prog()
        else:
            sampler = NUMERICAL[key](diagram, options=options, model=_solver(robot))
            sampler.create_prog()
            rng = np.random.default_rng(MUG_SEEDS[robot])
            lower = sampler.plant.GetPositionLowerLimits()
            upper = sampler.plant.GetPositionUpperLimits()
            # Collision-free, as the benchmark's targets are, so the mug lands somewhere the
            # arm can actually be and the mug-visibility check below has something to find.
            while True:
                q = rng.uniform(lower, upper)
                if np.ravel(sampler.collision_free_constraint_eval.Eval(q))[0] < 1.0:
                    break
            diagram_with_mug, mug = GenerateDiagramWithMug(q, sampler, yaml, None)
            program = NUMERICAL[key](diagram_with_mug, options=options, model=_solver(robot))
            program.create_prog(target_mug=mug)
            program._test_mug_q = q
    program._test_yaml = yaml
    _PROGRAMS[key] = program
    return program


def _spec(program):
    return SceneSpec.from_program(program, yaml_path=program._test_yaml)


def _draw(program, n, rng, nan_rows=0):
    lower = program.plant.GetPositionLowerLimits()
    upper = program.plant.GetPositionUpperLimits()
    Q = rng.uniform(lower, upper, size=(n, lower.size))
    if nan_rows:
        Q = np.vstack([Q, np.full((nan_rows, lower.size), np.nan)])
    return Q


def _reference(program, Q):
    """The program's own scaled row and gradient, NaN where the configuration is."""
    scale = program.options.collision_row_scale
    nq = Q.shape[1]
    values = np.full(Q.shape[0], np.nan)
    grads = np.full(Q.shape, np.nan)
    for i, q in enumerate(Q):
        if not np.all(np.isfinite(q)):
            continue
        y = program.collision_free_constraint_eval.Eval(InitializeAutoDiff(q))
        values[i] = scale * np.ravel(ExtractValue(y))[0]
        g = np.ravel(ExtractGradient(y))
        grads[i] = scale * g if g.size == nq else 0.0
    return values, grads


def _min_distance(program, q, max_distance=0.3):
    """True minimum signed distance over the SAME candidate pairs the constraint sees."""
    program.plant.SetPositions(program.plant_context, q)
    query = program.plant.get_geometry_query_input_port().Eval(program.plant_context)
    pairs = query.ComputeSignedDistancePairwiseClosestPoints(max_distance)
    return min(p.distance for p in pairs) if pairs else max_distance


## ------------------------------------------------------------------------------------ ##

def test_scene_spec_round_trips_the_mug_exactly():
    for robot in ("panda", "iiwa"):
        pose = _program(robot, "pose")
        spec = _spec(pose)
        assert not spec.has_mug and spec.mug_pose() is None and spec.extra_directives() == []
        assert spec.row_scale == pose.options.collision_row_scale
        assert spec.collision_bound == pose.options.collision_bound
        assert spec.influence_distance_offset == pose.options.collision_influence_offset

        mug = _program(robot, "mug")
        spec = _spec(mug)
        assert spec.has_mug
        X = spec.mug_pose()
        assert np.array_equal(X.GetAsMatrix4(), mug.target_mug.middle.GetAsMatrix4()), (
            "the mug weld must be rebuilt bit-for-bit")
        back = pickle.loads(pickle.dumps(spec))
        assert back == spec, "SceneSpec must survive pickling unchanged (it is what a worker gets)"
        assert mug.plant.HasModelInstanceNamed(spec.mug_model_name)
        # The weld-readback fallback (no `target_mug` attribute) is exact to rounding, not
        # bitwise; say how far, so the docstring's claim is a measurement.
        saved = mug.target_mug
        del mug.target_mug
        try:
            fallback = SceneSpec.from_program(mug, yaml_path=mug._test_yaml)
        finally:
            mug.target_mug = saved
        err = np.abs(fallback.mug_pose().GetAsMatrix4() - X.GetAsMatrix4()).max()
        assert err < 1e-14, err
        print(f"     {robot}: weld read-back fallback within {err:.1e} of the exact pose")
    # No YAML anywhere -> a clear error, not a scene built from a guess. BuildEnv records
    # the directives file on the diagram (`diagram.scene_yaml`), so a program built the
    # normal way always has one; the refusal is exercised by clearing it. A scene built
    # without a file (BuildEnv(directives_file=None)) carries None and must refuse too.
    bare = _program("panda", "pose")
    assert SceneSpec.from_program(bare).yaml_path.endswith(".yaml")
    saved_yaml = bare.diagram.scene_yaml
    bare.diagram.scene_yaml = None
    try:
        SceneSpec.from_program(bare)
    except ValueError as e:
        assert "yaml_path" in str(e)
    else:
        raise AssertionError("from_program must refuse when no scene YAML is recorded")
    finally:
        bare.diagram.scene_yaml = saved_yaml
    print("PASS SceneSpec round-trips the mug exactly and refuses an unknown scene")


def test_pool_matches_the_program_bitwise():
    rng = np.random.default_rng(2026)
    for robot in ("panda", "iiwa"):
        for task in ("pose", "mug"):
            program = _program(robot, task)
            spec = _spec(program)
            Q = _draw(program, 64, rng, nan_rows=2)
            ref_v, ref_g = _reference(program, Q)
            scale = spec.row_scale
            finite = np.isfinite(ref_v)
            assert (ref_v[finite] > scale).any() and (ref_v[finite] <= scale).any(), (
                "the draw must contain both penetrating and clear configurations")
            for workers in (1, 4):
                with DrakeCollisionPool(spec, workers=workers) as pool:
                    v, g = pool.eval(Q)
                    v2, _ = pool.eval(Q, need_grad=False)
                assert v.shape == (Q.shape[0],) and g.shape == Q.shape
                dv = np.nanmax(np.abs(v - ref_v)) if finite.any() else 0.0
                dg = np.nanmax(np.abs(g - ref_g)) if finite.any() else 0.0
                assert np.array_equal(v, ref_v, equal_nan=True), (robot, task, workers, dv)
                assert np.array_equal(g, ref_g, equal_nan=True), (robot, task, workers, dg)
                assert np.array_equal(v2, ref_v, equal_nan=True), (robot, task, workers)
                assert np.all(np.isnan(v[~finite])) and np.all(np.isnan(g[~finite]))
                print(f"     {robot} {task:4s} K={workers}: {Q.shape[0]} rows, nq={Q.shape[1]}, "
                      f"max |dvalue| {dv:.1e}, max |dgrad| {dg:.1e}, "
                      f"{int((ref_v[finite] > scale).sum())} of {int(finite.sum())} penetrating, "
                      f"{int((~finite).sum())} NaN rows returned NaN")
    print("PASS pool value and gradient equal the program's binding bitwise, for K in {1, 4}")


def test_collision_row_autograd():
    import torch
    program = _program("panda", "pose")
    spec = _spec(program)
    rng = np.random.default_rng(7)
    Q = _draw(program, 12, rng)
    ref_v, ref_g = _reference(program, Q)
    with DrakeCollisionPool(spec, workers=4) as pool:
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            # float64: value and VJP exact.
            q = torch.tensor(Q, dtype=torch.float64, device=device, requires_grad=True)
            w = torch.arange(1, Q.shape[0] + 1, dtype=torch.float64, device=device)
            y = collision_row(q, pool)
            assert y.dtype == torch.float64 and y.device.type == device and y.shape == (Q.shape[0],)
            assert np.array_equal(y.detach().cpu().numpy(), ref_v)
            (y * w).sum().backward()
            expected = w.cpu().numpy()[:, None] * ref_g
            assert np.array_equal(q.grad.cpu().numpy(), expected), (
                "backward must be exactly grad_output[:, None] * stored gradient")
            # float32 in -> float32 out, on the input's device; Drake still ran in float64 AT
            # THE ROUNDED CONFIGURATION, so the reference is the pool at float32(Q), and the
            # only difference left is the cast of the result back to float32.
            Q32 = Q.astype(np.float32).astype(np.float64)
            ref32_v, ref32_g = pool.eval(Q32)
            q32 = torch.tensor(Q32, dtype=torch.float32, device=device, requires_grad=True)
            y32 = CollisionRow.apply(q32, pool)
            assert y32.dtype == torch.float32 and y32.device.type == device
            assert np.array_equal(y32.detach().cpu().numpy(), ref32_v.astype(np.float32))
            y32.sum().backward()
            assert q32.grad.dtype == torch.float32
            assert np.array_equal(q32.grad.cpu().numpy(), ref32_g.astype(np.float32))
            # A NaN particle propagates as NaN and never raises.
            qn = torch.tensor(Q[:2], dtype=torch.float64, device=device, requires_grad=True)
            with torch.no_grad():
                qn[1] = float("nan")
            yn = collision_row(qn, pool)
            yn.sum().backward()
            assert torch.isfinite(yn[0]) and torch.isnan(yn[1]) and torch.isnan(qn.grad[1]).all()
            print(f"     {device}: value exact, VJP exact, float32 preserved, NaN row propagates")
        # Finite differences: the autodiff gradient is Drake's own, so this is a sanity check
        # on the plumbing rather than on Drake. Mesh-pair distances are only piecewise smooth,
        # so the median is asserted and the max reported.
        h = 1e-6
        rel = []
        for i in range(Q.shape[0]):
            if not np.any(ref_g[i]):
                continue
            fd = np.zeros(Q.shape[1])
            for j in range(Q.shape[1]):
                e = np.zeros(Q.shape[1]); e[j] = h
                vp, _ = pool.eval(np.vstack([Q[i] + e, Q[i] - e]), need_grad=False)
                fd[j] = (vp[0] - vp[1]) / (2 * h)
            rel.append(np.linalg.norm(fd - ref_g[i]) / max(np.linalg.norm(ref_g[i]), 1e-12))
        rel = np.array(rel)
        assert rel.size >= 4, "need rows with a nonzero gradient"
        assert np.median(rel) < 1e-5, rel
        print(f"     central differences: median rel. error {np.median(rel):.1e}, "
              f"max {rel.max():.1e} over {rel.size} rows")
    print("PASS collision_row is an exact autograd wrapper of the pool")


def test_boolean_checker_agrees_with_the_row_outside_the_band():
    BAND = 2.5e-3   # metres above the bound where SmoothOverMax may still exceed 1
    rng = np.random.default_rng(11)
    for robot, task in (("panda", "pose"), ("panda", "mug"), ("iiwa", "pose"), ("iiwa", "mug")):
        program = _program(robot, task)
        spec = _spec(program)
        bound = spec.collision_bound
        checker = ParallelCollisionChecker(spec)
        assert checker.padding == bound
        assert tuple(sorted(checker.robot_instance_names)) == tuple(
            sorted(SCENES[(robot, task)].robot_instances)), (
            checker.robot_instance_names, SCENES[(robot, task)].robot_instances)
        checker0 = ParallelCollisionChecker(spec, padding=0.0)

        Q = _draw(program, 256, rng, nan_rows=1)
        ok = checker.collision_free(Q)
        ok0 = checker0.collision_free(Q)
        assert not ok[-1] and not ok0[-1], "a NaN row is not collision-free"
        Q = Q[:-1]; ok = ok[:-1]; ok0 = ok0[:-1]
        y = np.array([np.ravel(program.collision_free_constraint_eval.Eval(q))[0] for q in Q])
        dmin = np.array([_min_distance(program, q) for q in Q])

        # The exact relations.
        below = dmin < bound
        assert np.all(y[below] > 1.0), "min distance below the bound must put the penalty above 1"
        # The checker is a direct distance test against its padding; a mismatch with the
        # plant's own min distance would mean a different candidate set.
        tie = np.abs(dmin - bound) <= 1e-9
        assert np.array_equal(ok[~tie], ~below[~tie]), (
            int((ok[~tie] != ~below[~tie]).sum()), "checker vs true min distance")
        # Padding 0 differs from padding `bound` exactly on 0 <= d < bound.
        sliver = (dmin >= 0.0) & (dmin < bound)
        tie0 = tie | (np.abs(dmin) <= 1e-9)
        assert np.array_equal((ok0 != ok)[~tie0], sliver[~tie0]), (
            "padding 0 vs padding bound must differ exactly on the [0, bound) sliver")
        # Agreement with the ROW outside the smoothing band; the band is reported.
        outside = np.abs(dmin - bound) > BAND
        disagree = (y <= 1.0) != ok
        assert not np.any(disagree & outside), (
            int((disagree & outside).sum()), dmin[disagree & outside], y[disagree & outside])
        band_cases = int((~outside).sum())
        band_disagree = int((disagree & ~outside).sum())
        print(f"     {robot} {task:4s}: {Q.shape[0]} draws, {int(below.sum())} below the bound, "
              f"{int(sliver.sum())} in [0, bound), {band_cases} within +/-{BAND*1e3:.1f} mm of "
              f"the bound of which {band_disagree} disagree with the row (the SmoothOverMax "
              f"band); 0 disagreements outside it")

    # The mug scene's checker SEES the mug: a configuration the pose scene passes and the mug
    # scene rejects is rejected by the mug checker and passed by the pose checker.
    for robot in ("panda", "iiwa"):
        pose, mug = _program(robot, "pose"), _program(robot, "mug")
        spec_pose, spec_mug = _spec(pose), _spec(mug)
        # Around the (collision-free) grasp configuration the hand is at the mug, so small
        # perturbations swing the fingers into it while the arm stays clear of the shelves.
        q_star = mug._test_mug_q
        Q = np.vstack([q_star[None, :]] + [
            q_star + rng.normal(0.0, sigma, size=(128, q_star.size))
            for sigma in (0.02, 0.05, 0.1, 0.2)] + [_draw(pose, 256, rng)])
        with DrakeCollisionPool(spec_pose, workers=4) as pp, \
             DrakeCollisionPool(spec_mug, workers=4) as pm:
            v_pose, _ = pp.eval(Q, need_grad=False)
            v_mug, _ = pm.eval(Q, need_grad=False)
        only_mug = (v_mug > spec_mug.row_scale) & (v_pose <= spec_pose.row_scale)
        assert only_mug.any(), "no draw hits only the mug; the mug scene has no visible mug?"
        cp = ParallelCollisionChecker(spec_pose).collision_free(Q[only_mug])
        cm = ParallelCollisionChecker(spec_mug).collision_free(Q[only_mug])
        assert np.all(cp) and not np.any(cm), (cp, cm)
        print(f"     {robot}: {int(only_mug.sum())} of {Q.shape[0]} draws hit only the mug "
              f"(grasp configuration itself: mug row {v_mug[0]:.4f}, pose row {v_pose[0]:.4f}); "
              f"mug checker rejects all, pose checker passes all")
    print("PASS boolean checker agrees with the row outside the SmoothOverMax band and sees the mug")


def test_timing_table():
    spec = _spec(_program("panda", "pose"))
    t0 = time.perf_counter()
    rows = measure_pool(spec, Ns=(64, 256, 1024), workers=(8,), repeats=3,
                        out=lambda s: print("     " + s))
    assert len(rows) == 6
    print(f"PASS timing table completed in {time.perf_counter() - t0:.1f} s "
          f"({os.cpu_count()} CPUs on this machine; seconds are never compared across machines)")


def test_pool_lifecycle():
    spec = _spec(_program("panda", "pose"))
    Q = _draw(_program("panda", "pose"), 8, np.random.default_rng(3))
    pool = DrakeCollisionPool(spec, workers=2)
    assert pool._ctx.get_start_method() == "spawn"
    assert all(pool.alive())
    v, g = pool.eval(Q)
    assert np.all(np.isfinite(v))
    pool.close()
    assert pool.closed and not any(pool.alive())
    pool.close()   # idempotent
    try:
        pool.eval(Q)
    except RuntimeError as e:
        assert "closed" in str(e)
    else:
        raise AssertionError("eval after close must raise")
    with DrakeCollisionPool(spec, workers=2) as second:
        v2, g2 = second.eval(Q)
    assert np.array_equal(v, v2) and np.array_equal(g, g2)
    assert not any(second.alive())
    # Fewer rows than workers: the idle workers get nothing and the result is still complete.
    with DrakeCollisionPool(spec, workers=4) as wide:
        v3, g3 = wide.eval(Q[:2])
        assert np.array_equal(v3, v[:2]) and np.array_equal(g3, g[:2])
        v4, g4 = wide.eval(Q[:0])
        assert v4.shape == (0,) and g4.shape == (0, Q.shape[1])
    print("PASS pool lifecycle: spawn context, close terminates, second pool works, small batches")


if __name__ == "__main__":
    test_scene_spec_round_trips_the_mug_exactly()
    test_pool_matches_the_program_bitwise()
    test_collision_row_autograd()
    test_boolean_checker_agrees_with_the_row_outside_the_band()
    test_timing_table()
    test_pool_lifecycle()
    print("ALL PASS")
