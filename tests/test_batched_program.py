"""`src/svgd/batched_program.py` against the Drake programs it replays.

THE PROGRAM IS THE ORACLE. On all four rigid program classes x both arms -- Panda and iiwa,
pose and grasp, learned (`n6` charts) and joint space -- eight programs built the way the
benchmark builds them (hardened scenes, a grasp scene with the target mug welded by
`GenerateDiagramWithMug`), this file pins:

  1. every entry of `drake_rows` against `prog.EvalBinding(all_constraints, x)` at B = 1 and
     B = 32 (the collision row BITWISE against the same Drake functional), the extra
     bindings (z box, c box, trust region) against their own `EvalBinding`, `F` against the
     sum of every cost binding and `F_reported` against the sum without the learned-only
     regularizers, and `h` / `g` against a recomputation from the rows and the bounds;
  2. the autograd Jacobian of the rows against the program's own AutoDiffXd chain
     (`EvalAllConstraints(InitializeAutoDiff(x))`), the collision row's gradient bitwise
     against the in-process row's stored Drake gradient on the joint-space arm (and one ulp from the
     program's chain, which scales through pydrake's `float * AutoDiffXd`), and
     `jacobian_q` / `vjp_q` against autograd of `evaluate().q`;
  3. `init_from_q` against `SetStartFromQ` (c and z UNCLIPPED, q_c the clipped residual)
     and the round trip `evaluate(init_from_q(q)).q == q` to the flow's inverse floor;
  4. `to_drake_x` / `from_drake_x`, including through `benchmark.verify`;
  5. `project` returns the exact clip distance, and `regions=True` clips the c and z boxes;
  6. a timing table (printed only): evaluate with and without the collision row, N in
     {64, 256, 1024}, float32 and float64, with the flow / FK / collision split.

The flow's runaway population (CLAUDE.md, the gain ceiling) amplifies last-ulp differences
in the rpy -> quaternion conversion into 1e-9..1e-7 in q, so the tolerance assertions skip
particles with |q|_inf >= 100 (the sibling `tests/test_batched_flow.py` documents this) and
the figure over ALL particles is recorded; nothing may be NaN unless Drake's is.

No pytest config in this repo, so these are plain `test_*` functions with a `__main__`
driver. Needs both `n6` charts under `models/` and spawns collision workers, so run it as
a script: `.venv/bin/python tests/test_batched_program.py`.
"""

import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydrake.autodiffutils import InitializeAutoDiff                       # noqa: E402
from pydrake.all import AutoDiffXd                                          # noqa: E402

from src import benchmark                                                  # noqa: E402
from src.flow_loading import LoadFlowSolver                                 # noqa: E402
from src.generic_program import ProgramOptions                              # noqa: E402
from src.iiwa_program import (Iiwa14IKProgram, Iiwa14IKProgramNumerical,    # noqa: E402
                              IiwaMugProgram, IiwaMugProgramNumerical)
from src.panda_program import (PandaIKProgram, PandaIKProgramNumerical,     # noqa: E402
                               PandaMugProgram, PandaMugProgramNumerical)
from src.svgd.batched_program import (GENERIC_BINDING, REGULARIZER_COSTS,   # noqa: E402
                                      BatchedProgram, RowScaling)
from src.target_screening import SceneFile                                  # noqa: E402
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints, RepoDir  # noqa: E402

CHECKPOINTS = {
    "panda": "models/panda/panda__n6__step620000.pkl",
    "iiwa": "models/iiwa14/iiwa14__n6__step620000.pkl",
}
LEARNED = {
    ("panda", "pose"): PandaIKProgram, ("panda", "mug"): PandaMugProgram,
    ("iiwa", "pose"): Iiwa14IKProgram, ("iiwa", "mug"): IiwaMugProgram,
}
NUMERICAL = {
    ("panda", "pose"): PandaIKProgramNumerical, ("panda", "mug"): PandaMugProgramNumerical,
    ("iiwa", "pose"): Iiwa14IKProgramNumerical, ("iiwa", "mug"): IiwaMugProgramNumerical,
}
KEYS = [(r, t, a) for r in ("panda", "iiwa") for t in ("pose", "mug") for a in ("learned", "numerical")]
MUG_SEEDS = {"panda": 101, "iiwa": 202}
POSE_SEEDS = {"panda": 11, "iiwa": 22}
CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

_SOLVERS, _SCENES, _PROGRAMS, _BATCHED = {}, {}, {}, {}


## ------------------------------------------------------------------------------------ ##
##                                 program construction                                  ##
## ------------------------------------------------------------------------------------ ##

def _solver(robot):
    if robot not in _SOLVERS:
        with HiddenPrints():
            _SOLVERS[robot] = LoadFlowSolver(
                "iiwa14" if robot == "iiwa" else robot,
                os.path.join(RepoDir(), CHECKPOINTS[robot]))
    return _SOLVERS[robot]


def _options(robot, task, arm):
    """The benchmark's shape of options: learned 1e-4 centering, joint space 1.0; the
    `latent` config's trust region; the approved correction penalty. The iiwa programs
    additionally exercise the latent regularizer and, on the pose task, the `rpy_boxed`
    form, so every cost and bound kind the replay knows is covered by some program."""
    opts = ProgramOptions(visualize=False, joint_centering_cost=1e-4 if arm == "learned" else 1.0,
                          latent_trust_region=4.0, correction_cost_weight=10.0, mug_height=0.04,
                          ik_constraint_tol=(1e-4, 0.01))
    if robot == "iiwa":
        opts = replace(opts, latent_cost_weight=0.1)
        if task == "pose":
            opts = replace(opts, orientation_error_form="rpy_boxed")
    return opts


def _scene(robot, task):
    """`(diagram, target)` shared by both arms: a pose target (7-vector) at the scene frame
    of a random configuration, or a `Mug` welded at the grasp frame of a collision-free
    one, exactly as the benchmark places them."""
    key = (robot, task)
    if key in _SCENES:
        return _SCENES[key]
    yaml = SceneFile(robot, task, "hardened")
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=yaml)
        helper = NUMERICAL[key](diagram, options=_options(robot, task, "numerical"),
                                model=_solver(robot))
        if task == "pose":
            rng = np.random.default_rng(POSE_SEEDS[robot])
            q = rng.uniform(*helper.ConfigLimits())
            translation, wxyz = helper.fk(helper.ConfigToPlantQ(q))
            _SCENES[key] = (diagram, np.concatenate([translation, wxyz]))
        else:
            helper.create_prog()
            rng = np.random.default_rng(MUG_SEEDS[robot])
            lower, upper = helper.plant.GetPositionLowerLimits(), helper.plant.GetPositionUpperLimits()
            while True:
                q = rng.uniform(lower, upper)
                if np.ravel(helper.collision_free_constraint_eval.Eval(q))[0] < 1.0:
                    break
            diagram_mug, mug = GenerateDiagramWithMug(q, helper, yaml, None)
            _SCENES[key] = (diagram_mug, mug)
    return _SCENES[key]


def program(robot, task, arm):
    key = (robot, task, arm)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    diagram, target = _scene(robot, task)
    cls = (LEARNED if arm == "learned" else NUMERICAL)[(robot, task)]
    with HiddenPrints():
        p = cls(diagram, options=_options(robot, task, arm), model=_solver(robot))
        if task == "pose":
            p.create_prog(target)
        else:
            p.create_prog(target_mug=target)
    _PROGRAMS[key] = p
    return p


def batched(robot, task, arm, device=DEVICE, dtype=torch.float64):
    """The program's batched replay (its collision row runs in process, on the program's own
    constraint). One is cached at a time: every use in these tests is sequential."""
    key = (robot, task, arm, str(device), dtype)
    if key not in _BATCHED:
        close_all()
        _BATCHED[key] = BatchedProgram.from_program(program(robot, task, arm), dtype=dtype,
                                                    device=device)
    return _BATCHED[key]


def close_all():
    """Close every cached BatchedProgram and forget it (a later `batched` rebuilds)."""
    for bp in _BATCHED.values():
        bp.close()
    _BATCHED.clear()


def lumped_batch(bp, rng, B):
    """Random decision vectors: learned `c = native_c +- U(0.3)`, `z ~ N(0, I)`,
    `q_c ~ U(+-0.1)`; joint space `q ~ U(limits)`."""
    p = bp.program
    if bp.is_learned:
        c = bp.native_c().cpu().numpy()[None, :] + rng.uniform(-0.3, 0.3, size=(B, 6))
        z = rng.standard_normal((B, bp.width))
        qc = rng.uniform(-0.1, 0.1, size=(B, bp.ndof))
        return np.concatenate([c, z, qc], axis=1)
    lower, upper = p.ConfigLimits()
    return rng.uniform(np.asarray(lower)[:bp.ndof], np.asarray(upper)[:bp.ndof], size=(B, bp.ndof))


def drake_q(p, x):
    return np.asarray([float(v) for v in p.VarsToQ(np.asarray(x, dtype=float))])


def bindings_by_description(p):
    out = {}
    for b in p.prog.GetAllConstraints():
        out.setdefault(b.evaluator().get_description(), []).append(b)
    return out


def collision_free_q(p, rng, n):
    lower, upper = p.ConfigLimits()
    out = []
    while len(out) < n:
        q = rng.uniform(np.asarray(lower)[:p.num_arm_dof], np.asarray(upper)[:p.num_arm_dof])
        if np.ravel(p.collision_free_constraint_eval.Eval(p.ConfigToPlantQ(q)))[0] < 1.0:
            out.append(q)
    return np.stack(out)


## ------------------------------------------------------------------------------------ ##
##                                       the tests                                       ##
## ------------------------------------------------------------------------------------ ##

def _check_rows(bp, X_np, label):
    """Parity of one BatchedProgram against its program on a batch; returns a dict of the
    measured figures."""
    p = bp.program
    prog = p.prog
    B = X_np.shape[0]
    X = torch.tensor(X_np, dtype=bp.dtype, device=bp.device)
    scale = p.options.collision_row_scale
    nrows = bp.n_rows
    coll = [i for i, blk in enumerate(bp._blocks) if blk.kind == "collision"]
    assert len(coll) == 1
    coll_row = bp._blocks[coll[0]].start
    noncoll = [r for r in range(nrows) if r != coll_row]

    ref_rows = np.stack([np.asarray(prog.EvalBinding(p.all_constraints, bp.to_drake_x(X_np[i])),
                                    dtype=float).reshape(-1) for i in range(B)])
    assert ref_rows.shape == (B, nrows), (ref_rows.shape, nrows)
    q_ref = np.stack([drake_q(p, X_np[i]) for i in range(B)])
    ordinary = np.abs(q_ref).max(axis=1) < 100.0

    ev = bp.evaluate(X)
    rows = ev.drake_rows.detach().cpu().numpy()
    evs1 = [bp.evaluate(X[i:i + 1]) for i in range(B)]
    rows1 = np.concatenate([e.drake_rows.detach().cpu().numpy() for e in evs1])
    q_plant1 = np.concatenate([e.q_plant.detach().cpu().numpy() for e in evs1])
    assert rows.shape == ref_rows.shape
    assert not np.isnan(rows).any() and not np.isnan(ref_rows).any()

    d1 = np.abs(rows1 - ref_rows)[:, noncoll]
    dB = np.abs(rows - ref_rows)[:, noncoll]
    fig = dict(n_ordinary=int(ordinary.sum()), B=B,
               noncoll_B1=float(d1[ordinary].max()), noncoll_BN=float(dB[ordinary].max()),
               noncoll_B1_all=float(d1.max()), noncoll_BN_all=float(dB.max()),
               max_q_inf=float(np.abs(q_ref).max()))
    assert fig["noncoll_B1"] <= 1e-10, (label, fig)
    assert fig["noncoll_BN"] <= 1e-8, (label, fig)

    ## The collision row: the same Drake functional at the SAME plant vector is bitwise.
    ## On the joint-space arm that vector is Drake's own, so the row equals the binding's
    ## bitwise; on the learned arm q differs from Drake's by the rpy conversion's last
    ## ulp, so bitwise is asserted against the binding evaluated at the batched q_plant
    ## and the difference against the Drake row is recorded.
    ## (B = 1 and B = 32 are NOT compared with each other on the learned arm: the network's
    ## batched matmuls differ at the reduction-order level, so their q differ by ~1e-16 and
    ## the collision value with them -- each batch is checked against Drake at ITS q.)
    q_plant = ev.q_plant.detach().cpu().numpy()
    for Qp, got in ((q_plant, rows[:, coll_row]), (q_plant1, rows1[:, coll_row])):
        same_functional = np.array([scale * np.ravel(p.collision_free_constraint_eval.Eval(q))[0]
                                    for q in Qp])
        assert np.array_equal(got, same_functional), label
    if not bp.is_learned:
        assert np.array_equal(rows1[:, coll_row], rows[:, coll_row]), label
    fig["collision_B1_vs_BN"] = float(np.abs(rows1[:, coll_row] - rows[:, coll_row]).max())
    assert torch.equal(ev.collision_y * scale, ev.drake_rows[:, coll_row]) or \
        torch.allclose(ev.collision_y * scale, ev.drake_rows[:, coll_row], rtol=0, atol=1e-15)
    fig["collision_vs_drake_row"] = float(np.abs(rows[:, coll_row] - ref_rows[:, coll_row]).max())
    if not bp.is_learned:
        assert np.array_equal(rows[:, coll_row], ref_rows[:, coll_row]), label

    ## Extra bindings, each against its own EvalBinding.
    by_desc = bindings_by_description(p)
    extra_err = 0.0
    for key_desc, key in (("ZBoundingBoxConstraint", "z_box"), ("CBoxConstraint", "c_box"),
                          ("LatentTrustRegion", "trust")):
        if key_desc in by_desc:
            assert key in ev.extra_rows, (label, key)
            (binding,) = by_desc[key_desc]
            ref = np.stack([np.asarray(prog.EvalBinding(binding, bp.to_drake_x(X_np[i])),
                                       dtype=float).reshape(-1) for i in range(B)])
            got = ev.extra_rows[key].detach().cpu().numpy()
            assert got.shape == ref.shape, (label, key, got.shape, ref.shape)
            extra_err = max(extra_err, float(np.abs(got - ref).max()))
        else:
            assert key not in ev.extra_rows, (label, key)
    fig["extra_rows"] = extra_err
    assert extra_err <= 1e-12, (label, extra_err)

    ## Costs.
    assert REGULARIZER_COSTS == benchmark._REGULARIZER_COSTS
    F_ref = np.zeros(B)
    Frep_ref = np.zeros(B)
    for i in range(B):
        x_full = bp.to_drake_x(X_np[i])
        for binding in prog.GetAllCosts():
            v = float(np.asarray(prog.EvalBinding(binding, x_full)).sum())
            F_ref[i] += v
            if binding.evaluator().get_description() not in REGULARIZER_COSTS:
                Frep_ref[i] += v
    F = ev.F.detach().cpu().numpy()
    Frep = ev.F_reported.detach().cpu().numpy()
    relF = np.abs(F - F_ref) / np.maximum(1.0, np.abs(F_ref))
    relR = np.abs(Frep - Frep_ref) / np.maximum(1.0, np.abs(Frep_ref))
    fig["F_rel"] = float(relF[ordinary].max())
    fig["F_reported_rel"] = float(relR[ordinary].max())
    assert fig["F_rel"] <= 1e-10 and fig["F_reported_rel"] <= 1e-10, (label, fig)
    if bp.is_learned:
        assert (Frep_ref <= F_ref + 1e-12).all() and (F_ref - Frep_ref).max() > 0, (
            "the learned arm's regularizers must contribute to F and not to F_reported")
    else:
        assert np.allclose(F_ref, Frep_ref), "joint space carries no regularizer"

    ## h / g recomputed from the rows and the bounds (RowScaling is the default here).
    def value_of(spec):
        if spec.drake_binding == GENERIC_BINDING:
            return rows[:, spec.drake_row]
        key = {"ZBoundingBoxConstraint": "z_box", "CBoxConstraint": "c_box",
               "LatentTrustRegion": "trust"}[spec.drake_binding]
        return ev.extra_rows[key].detach().cpu().numpy()[:, spec.drake_row]
    h = ev.h.detach().cpu().numpy()
    g = ev.g.detach().cpu().numpy()
    assert h.shape == (B, len(bp.h_spec)) and g.shape == (B, len(bp.g_spec))
    for j, spec in enumerate(bp.h_spec):
        assert spec.kind == "eq" and spec.lb == spec.ub
        assert np.allclose(h[:, j], value_of(spec) - spec.lb, rtol=0, atol=1e-13), (label, spec)
    for j, spec in enumerate(bp.g_spec):
        v = value_of(spec)
        expect = spec.lb - v if spec.kind == "lo" else v - spec.ub
        assert spec.kind in ("lo", "hi")
        assert np.isfinite(spec.lb if spec.kind == "lo" else spec.ub)
        assert np.allclose(g[:, j], expect, rtol=0, atol=1e-13), (label, spec)
    los = [s.kind for s in bp.g_spec]
    assert los == sorted(los, key=lambda k: 0 if k == "lo" else 1), "all lo rows precede hi rows"
    ## The inventory, by binding.
    counts = {}
    for s in bp.h_spec + bp.g_spec:
        counts[(s.drake_binding, s.kind)] = counts.get((s.drake_binding, s.kind), 0) + 1
    fig["inventory"] = counts
    nd = bp.ndof
    if bp.is_mug:
        assert counts[(GENERIC_BINDING, "eq")] == 2                      # mug x, y
        assert counts[(GENERIC_BINDING, "lo")] == 1 + nd                 # mug z + joint limits
        assert counts[(GENERIC_BINDING, "hi")] == 1 + 1 + nd             # mug z + collision + jl
    elif p.options.orientation_error_form == "rpy":
        assert counts[(GENERIC_BINDING, "eq")] == 6
        assert counts[(GENERIC_BINDING, "lo")] == nd
        assert counts[(GENERIC_BINDING, "hi")] == 1 + nd
    else:
        assert counts[(GENERIC_BINDING, "eq")] == 3
        assert counts[(GENERIC_BINDING, "lo")] == 3 + nd
        assert counts[(GENERIC_BINDING, "hi")] == 3 + 1 + nd
    if bp.is_learned:
        w = bp.width
        assert counts[("ZBoundingBoxConstraint", "lo")] == w and counts[("ZBoundingBoxConstraint", "hi")] == w
        assert counts[("CBoxConstraint", "lo")] == 6 and counts[("CBoxConstraint", "hi")] == 6
        assert counts[("LatentTrustRegion", "hi")] == 1 and ("LatentTrustRegion", "lo") not in counts
    else:
        assert all(k[0] == GENERIC_BINDING for k in counts)
    return fig


def test_row_parity_all_programs():
    rng = np.random.default_rng(0)
    print("  row parity (max |batched - Drake| over ordinary particles, |q|_inf < 100):")
    for robot, task, arm in KEYS:
        bp = batched(robot, task, arm)
        X_np = lumped_batch(bp, rng, 32)
        label = f"{robot}/{task}/{arm}"
        fig = _check_rows(bp, X_np, label)
        print(f"    {label:22s} cuda: non-collision rows B=1 {fig['noncoll_B1']:.2e}, B=32 "
              f"{fig['noncoll_BN']:.2e} ({fig['n_ordinary']}/32 ordinary; over all: "
              f"{fig['noncoll_B1_all']:.2e} / {fig['noncoll_BN_all']:.2e}, max |q|_inf "
              f"{fig['max_q_inf']:.3g}); collision bitwise (vs Drake row "
              f"{fig['collision_vs_drake_row']:.1e}); extra rows {fig['extra_rows']:.1e}; "
              f"F rel {fig['F_rel']:.1e}, F_reported rel {fig['F_reported_rel']:.1e}")
    ## One program on the CPU (the joint-space arm needs no network, so the shared network
    ## never has to move).
    bp_cpu = batched("panda", "pose", "numerical", device="cpu")
    fig = _check_rows(bp_cpu, lumped_batch(bp_cpu, rng, 32), "panda/pose/numerical cpu")
    print(f"    {'panda/pose/numerical':22s} cpu : non-collision rows B=1 {fig['noncoll_B1']:.2e}, "
          f"B=32 {fig['noncoll_BN']:.2e}; collision bitwise")
    ## A NaN particle never raises and comes back NaN; its neighbours are untouched.
    bp = batched("panda", "pose", "learned")
    X_np = lumped_batch(bp, rng, 3)
    X_nan = X_np.copy(); X_nan[1] = np.nan
    ev = bp.evaluate(torch.tensor(X_nan, dtype=torch.float64, device=bp.device))
    ev_ok = bp.evaluate(torch.tensor(X_np, dtype=torch.float64, device=bp.device))
    assert torch.isnan(ev.drake_rows[1]).all() and torch.isnan(ev.F[1]) and torch.isnan(ev.g[1]).all()
    assert torch.equal(ev.drake_rows[[0, 2]], ev_ok.drake_rows[[0, 2]])
    ## need_collision=False leaves the collision entries NaN and collision_y None.
    ev2 = bp.evaluate(torch.tensor(X_np, dtype=torch.float64, device=bp.device), need_collision=False)
    coll = next(b.start for b in bp._blocks if b.kind == "collision")
    assert ev2.collision_y is None and torch.isnan(ev2.drake_rows[:, coll]).all()
    keep = [r for r in range(bp.n_rows) if r != coll]
    assert torch.equal(ev2.drake_rows[:, keep], ev_ok.drake_rows[:, keep])
    ## RowScaling multiplies h / g only.
    bp_s = BatchedProgram.from_program(bp.program, dtype=torch.float64, device=bp.device,
                                       collision=bp.collision, row_scaling=RowScaling(position=10.0, collision=3.0))
    ev_s = bp_s.evaluate(torch.tensor(X_np, dtype=torch.float64, device=bp.device))
    assert torch.equal(ev_s.drake_rows, ev_ok.drake_rows)
    for j, spec in enumerate(bp_s.h_spec):
        f = 10.0 if spec.group == "position" else 1.0
        assert torch.allclose(ev_s.h[:, j], f * ev_ok.h[:, j], rtol=0, atol=1e-13)
    for j, spec in enumerate(bp_s.g_spec):
        f = 3.0 if spec.group == "collision" else 1.0
        assert torch.allclose(ev_s.g[:, j], f * ev_ok.g[:, j], rtol=0, atol=1e-13)
    print("PASS row parity on 8 programs (+ CPU), NaN propagation, need_collision=False, RowScaling")


def test_gradients_match_autodiff():
    rng = np.random.default_rng(1)
    print("  gradients (relative to max(1, |J_Drake|_max)):")
    for robot, task, arm in KEYS:
        bp = batched(robot, task, arm)
        p = bp.program
        coll_row = next(b.start for b in bp._blocks if b.kind == "collision")
        noncoll = [r for r in range(bp.n_rows) if r != coll_row]
        worst, worst_coll, worst_jq, n_used = 0.0, 0.0, 0.0, 0
        for x_np in lumped_batch(bp, rng, 16):
            if n_used >= 4 or np.abs(drake_q(p, x_np)).max() >= 100.0:
                continue
            n_used += 1
            x = torch.tensor(x_np, dtype=torch.float64, device=bp.device)
            J = torch.autograd.functional.jacobian(
                lambda v: bp.evaluate(v[None]).drake_rows[0], x).cpu().numpy()
            rows_ad = p.EvalAllConstraints(InitializeAutoDiff(x_np).reshape(-1))
            J_ref = np.stack([
                r.derivatives() if isinstance(r, AutoDiffXd) and r.derivatives().size == bp.nvars
                else np.zeros(bp.nvars) for r in rows_ad])
            assert J.shape == J_ref.shape == (bp.n_rows, bp.nvars)
            denom = max(1.0, np.abs(J_ref[noncoll]).max())
            worst = max(worst, float(np.abs(J[noncoll] - J_ref[noncoll]).max() / denom))
            dc = np.abs(J[coll_row] - J_ref[coll_row]).max() / max(1.0, np.abs(J_ref[coll_row]).max())
            worst_coll = max(worst_coll, float(dc))
            if not bp.is_learned:
                ## Bitwise against the in-process row's stored gradient, which is Drake's own
                ## `ExtractGradient` scaled in numpy. The program's chain scales through
                ## pydrake's `float * AutoDiffXd` operator instead, and THAT product is one
                ## ulp off `scale * g` (measured 8.7e-19 on a 7.5e-3 entry), so against the
                ## AutoDiffXd chain the claim is 1e-15 relative, not bitwise.
                _, g_pool = bp.collision.eval(p.ConfigToPlantQ(x_np)[None])
                assert np.array_equal(J[coll_row], g_pool[0]), (robot, task, arm)
                assert dc <= 1e-15, (robot, task, arm, dc)
            ## jacobian_q / vjp_q against autograd of evaluate().q (the same graph).
            Jq = bp.jacobian_q(x[None])[0]
            Jq_ref = torch.autograd.functional.jacobian(
                lambda v: bp.evaluate(v[None], need_collision=False).q[0], x)
            worst_jq = max(worst_jq, float((Jq - Jq_ref).abs().max()))
            R = torch.tensor(rng.standard_normal((1, bp.ndof)), dtype=torch.float64, device=bp.device)
            vjp = bp.vjp_q(x[None], R)
            assert torch.allclose(vjp, (R[0] @ Jq)[None], rtol=0, atol=1e-12)
        assert n_used == 4, (robot, task, arm, n_used)
        tol = 1e-8 if bp.is_learned else 1e-12
        print(f"    {robot}/{task}/{arm:9s}: non-collision rows {worst:.2e}, collision row "
              f"{worst_coll:.2e}{' (bitwise vs the in-process row)' if not bp.is_learned else ''}, "
              f"jacobian_q vs autograd {worst_jq:.1e}")
        assert worst <= tol, (robot, task, arm, worst)
        assert worst_coll <= max(tol, 1e-15), (robot, task, arm, worst_coll)
        assert worst_jq <= 1e-12
    print("PASS gradients match the AutoDiffXd chain; collision gradient bitwise on joint space")


def test_init_from_q_matches_SetStartFromQ():
    rng = np.random.default_rng(2)
    for robot, task in LEARNED:
        bp = batched(robot, task, "learned")
        p = bp.program
        Q = collision_free_q(p, rng, 16)
        X_batched = bp.init_from_q(torch.tensor(Q, dtype=torch.float64, device=bp.device))
        refs, singles = [], []
        for q in Q:
            p.SetStartFromQ(q)
            refs.append(np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float))
            singles.append(bp.init_from_q(torch.tensor(q[None], dtype=torch.float64,
                                                       device=bp.device))[0].cpu().numpy())
        refs, singles = np.stack(refs), np.stack(singles)
        got = X_batched.cpu().numpy()
        w = bp.width
        dc = np.abs(got[:, :6] - refs[:, :6]).max()
        dz = np.abs(got[:, 6:6 + w] - refs[:, 6:6 + w]).max()
        dqc = np.abs(got[:, 6 + w:] - refs[:, 6 + w:]).max()
        d_single = np.abs(singles - got).max()
        zn = np.linalg.norm(refs[:, 6:6 + w], axis=1)
        q_back = bp.evaluate(X_batched).q.detach().cpu().numpy()
        rt = np.abs(q_back - Q).max()
        print(f"    {robot}/{task}: |dc| {dc:.1e}, |dz| {dz:.1e}, |dq_c| {dqc:.1e}, "
              f"batched vs single {d_single:.1e}; |q(start) - q_init| {rt:.1e}; |z| range "
              f"{zn.min():.2f}..{zn.max():.2f} (components past +-5 on "
              f"{int((np.abs(refs[:, 6:6 + w]) > 5).any(axis=1).sum())}/16 starts, unclipped)")
        assert dc <= 1e-9 and dz <= 1e-9 and dqc <= 1e-9, (robot, task, dc, dz, dqc)
        assert d_single <= 1e-8
        assert rt <= 1e-6, (robot, task, rt)
    ## Joint space: q itself, unprojected; `project` reproduces the program's clip distance.
    bp = batched("panda", "pose", "numerical")
    q = np.array([[0.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0]])
    X = bp.init_from_q(torch.tensor(q, dtype=torch.float64, device=bp.device))
    assert np.array_equal(X.cpu().numpy(), q)
    far = q + 20.0
    _, dist = bp.project(torch.tensor(far, dtype=torch.float64, device=bp.device))
    ref = bp.program.SetStartFromQ(far[0])
    assert abs(float(dist[0]) - ref) <= 1e-12, (float(dist[0]), ref)
    print("PASS init_from_q matches SetStartFromQ (c, z unclipped; q_c clipped residual)")


def test_to_drake_x_round_trips():
    rng = np.random.default_rng(3)
    for robot, task, arm in KEYS:
        bp = batched(robot, task, arm)
        p = bp.program
        x = lumped_batch(bp, rng, 1)[0]
        x_full = bp.to_drake_x(x)
        assert x_full.shape == (p.prog.num_vars(),)
        assert np.array_equal(bp.from_drake_x(x_full), x)
        mine = np.asarray(p.prog.EvalBinding(p.all_constraints, x_full), dtype=float).reshape(-1)
        theirs = np.asarray(p.EvalAllConstraints(x), dtype=float).reshape(-1)
        assert np.array_equal(mine, theirs)
        ## Through the harness's own scatter.
        verdict = benchmark.verify(p, None, lambda program, q: (True, {}), 1e-4, x_lumped=x)
        ev = bp.evaluate(torch.tensor(x[None], dtype=torch.float64, device=bp.device))
        q_mine = ev.q_plant[0].detach().cpu().numpy()
        err = np.abs(np.asarray(verdict.detail["q"]) - q_mine).max()
        if np.abs(q_mine).max() < 100.0:
            assert err <= 1e-10, (robot, task, arm, err)
        assert GENERIC_BINDING in verdict.detail["violations_all"]
    print("PASS to_drake_x / from_drake_x round-trip, including through benchmark.verify")


def test_project():
    rng = np.random.default_rng(4)
    for robot, task, arm in (("panda", "mug", "learned"), ("iiwa", "pose", "learned"),
                             ("iiwa", "mug", "numerical")):
        bp = batched(robot, task, arm)
        lo, hi = (t.cpu().numpy() for t in bp.bounds)
        rlo, rhi = (t.cpu().numpy() for t in bp.regions)
        if bp.is_learned:
            w = bp.width
            assert np.all(np.isinf(lo[:6 + w])) and np.all(np.isinf(hi[:6 + w]))
            assert np.allclose(lo[6 + w:], -bp.correction_bound) and np.allclose(hi[6 + w:], bp.correction_bound)
            assert np.allclose(rlo[6:6 + w], -5.0) and np.allclose(rhi[6:6 + w], 5.0)
            assert np.allclose(rlo[:6], bp.program.c_box[0]) and np.allclose(rhi[:6], bp.program.c_box[1])
            assert np.all(np.isinf(rlo[6 + w:])) and np.all(np.isinf(rhi[6 + w:]))
        else:
            lower, upper = bp.program.ConfigLimits()
            assert np.allclose(lo, lower[:bp.ndof]) and np.allclose(hi, upper[:bp.ndof])
            assert np.all(np.isinf(rlo)) and np.all(np.isinf(rhi))
        X_np = lumped_batch(bp, rng, 8)
        inside = np.clip(X_np, np.where(np.isfinite(lo), lo, -1e9), np.where(np.isfinite(hi), hi, 1e9))
        X = torch.tensor(inside, dtype=torch.float64, device=bp.device)
        Xp, d = bp.project(X)
        assert torch.equal(Xp, X) and torch.equal(d, torch.zeros_like(d))
        out = inside.copy()
        out[:, -1] = hi[-1] + 0.7            # beyond the last variable's bound
        out[:, 0] = X_np[:, 0] + 100.0       # c (learned) or q_0 (joint space)
        Xo = torch.tensor(out, dtype=torch.float64, device=bp.device)
        Xp, d = bp.project(Xo)
        expect_p = np.clip(out, lo, hi)
        assert np.array_equal(Xp.cpu().numpy(), expect_p)
        assert np.allclose(d.cpu().numpy(), np.linalg.norm(expect_p - out, axis=1), rtol=0, atol=1e-12)
        if bp.is_learned:
            assert np.allclose(d.cpu().numpy(), 0.7), "c is not a variable bound; only q_c clips"
            Xr, dr = bp.project(Xo, regions=True)
            expect_r = np.clip(out, np.maximum(lo, rlo), np.minimum(hi, rhi))
            assert np.array_equal(Xr.cpu().numpy(), expect_r)
            assert np.allclose(dr.cpu().numpy(), np.linalg.norm(expect_r - out, axis=1))
            assert (dr > d).all(), "regions=True must report the c clip too"
    print("PASS project: zero inside, exact distance outside, regions=True clips c and z")


def test_robot_hooks():
    """The hooks a robot whose configuration is not the plant's q plugs in behind. A
    program whose `ConfigToPlantQ` the default pad does not reproduce is refused BY NAME;
    hooks that are passed are used (here: the default pad written differently and the
    default provider, so the result must be bitwise the default's)."""
    bp = batched("panda", "pose", "numerical")
    p = bp.program
    saved = p.ConfigToPlantQ
    p.ConfigToPlantQ = lambda cfg: saved(cfg) + 1.0     # not the pad any more
    try:
        BatchedProgram.from_program(p, dtype=torch.float64, device=bp.device, collision=bp.collision)
    except NotImplementedError as e:
        assert type(p).__name__ in str(e) and "config_to_plant_q" in str(e)
    else:
        raise AssertionError("a robot the default map does not serve must be refused")
    finally:
        p.ConfigToPlantQ = saved

    def my_map(cfg):
        extra = p.num_pos - p.num_arm_dof
        return torch.cat([cfg, torch.full((cfg.shape[0], extra), 0.04, dtype=cfg.dtype,
                                          device=cfg.device)], dim=1) if extra else cfg
    hooked = BatchedProgram.from_program(p, dtype=torch.float64, device=bp.device, collision=bp.collision,
                                         body_pose_provider=bp.fk, config_to_plant_q=my_map)
    X = torch.tensor(lumped_batch(bp, np.random.default_rng(6), 8), dtype=torch.float64,
                     device=bp.device)
    a, b = bp.evaluate(X), hooked.evaluate(X)
    assert torch.equal(a.drake_rows, b.drake_rows) and torch.equal(a.F, b.F)
    print("PASS robot hooks: default map checked against ConfigToPlantQ; passed hooks are used")


def test_timing():
    """Printed only. Panda learned pose, CUDA, median of 10 after 3 warm-ups."""
    if not CUDA:
        print("  (no CUDA; timing skipped)")
        return
    rng = np.random.default_rng(5)
    p = program("panda", "pose", "learned")
    print(f"  evaluate timing, Panda learned pose, ms (median of 10; collision row in process, "
          f"{os.cpu_count()} CPUs): total [flow / fk / collision / rows+costs]")
    print(f"    {'N':>6} {'dtype':>8} {'with collision':>36} {'without':>30}")
    col = batched("panda", "pose", "learned").collision
    for dtype in (torch.float64, torch.float32):
        bp = BatchedProgram.from_program(p, dtype=dtype, device="cuda", collision=col, profile=True)
        for N in (64, 256, 1024):
            X = torch.tensor(lumped_batch(bp, rng, N), dtype=dtype, device="cuda")
            cells = []
            for need in (True, False):
                for _ in range(3):
                    bp.evaluate(X, need_collision=need)
                samples, splits = [], []
                for _ in range(10):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    ev = bp.evaluate(X, need_collision=need)
                    torch.cuda.synchronize()
                    samples.append(1e3 * (time.perf_counter() - t0))
                    splits.append([1e3 * ev.timing.get(k, 0.0) for k in ("flow", "fk", "collision", "rows_costs")])
                med = np.median(np.array(splits), axis=0)
                cells.append((float(np.median(samples)), med))
            (tw, sw), (to, so) = cells
            print(f"    {N:>6} {str(dtype).replace('torch.', ''):>8} "
                  f"{tw:8.2f} [{sw[0]:.2f} / {sw[1]:.2f} / {sw[2]:.2f} / {sw[3]:.2f}]"
                  f"   {to:8.2f} [{so[0]:.2f} / {so[1]:.2f} / - / {so[3]:.2f}]")


if __name__ == "__main__":
    try:
        test_row_parity_all_programs()
        test_gradients_match_autodiff()
        test_init_from_q_matches_SetStartFromQ()
        test_to_drake_x_round_trips()
        test_project()
        test_robot_hooks()
        test_timing()
    finally:
        close_all()
    print("ALL PASS")
