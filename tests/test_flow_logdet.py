"""The flow's log-det term (`flow_logdet_weight`, stage R27) on the Panda (n6), float64.

    .venv/bin/python tests/test_flow_logdet.py        (alone, under the memory scope)

`w * log|det dx/dz|` of the IKFlow map; with `latent_cost_weight = w` the pair is
`-w log p(x | c)` up to a constant. What is pinned:

  1. FrEIA's log-det (the forward pass's second return, `MakeFlowInference(with_logdet=True)`)
     IS `slogdet(jacrev(x wrt z))` up to a constant (the first scaling layer's log-det is
     stored in float32), on the Panda and -- where the map is 8 wide against 7 joints, the
     density of `(q, pad)` -- the iiwa;
  2. the Drake cost's AutoDiffXd gradient against central differences of its own value;
  3. the batched program's per-particle value and gradient (autograd, and the split step's
     extra jacrev row) against the Drake program at the same points, and its `F` /
     `F_reported` against the cost bindings with the term excluded from the reported one;
  4. the binding is absent at w = 0 and on the joint-space arm, and `ProgramOptions` refuses
     the term under `lift_q` and under `chart_error_scale`.
Tiers (tests/_tiers.py): the default `fast` runs in ~10 s on one CPU core; no `slow` group.
"""
import os
import sys
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
import _tiers                                                             # noqa: E402

from pydrake.autodiffutils import InitializeAutoDiff                   # noqa: E402

import test_batched_program as T                                       # noqa: E402
from src import benchmark                                              # noqa: E402
from src.flow_loading import LoadFlowSolver                            # noqa: E402
from src.generic_program import DEVICE, MakeFlowInference, ProgramOptions  # noqa: E402
from src.panda_program import PandaIKProgram, PandaIKProgramNumerical  # noqa: E402
from src.svgd import fused                                             # noqa: E402
from src.svgd.batched_program import REGULARIZER_COSTS, BatchedProgram  # noqa: E402
from src.svgd.solver import _Target                                    # noqa: E402
from src.utils import HiddenPrints, RepoDir                            # noqa: E402

W = 1.0
_PROGRAMS = {}


def panda_pose(w=W, arm="learned"):
    key = (w, arm)
    if key not in _PROGRAMS:
        diagram, target = T._scene("panda", "pose")
        opts = replace(T._options("panda", "pose", arm), latent_cost_weight=w,
                       flow_logdet_weight=w)
        cls = PandaIKProgram if arm == "learned" else PandaIKProgramNumerical
        with HiddenPrints():
            p = cls(diagram, options=opts, model=T._solver("panda"))
            p.create_prog(target)
        _PROGRAMS[key] = p
    return _PROGRAMS[key]


def binding_of(p, desc):
    found = [b for b in p.prog.GetAllCosts() if b.evaluator().get_description() == desc]
    return found[0] if found else None


def lumped_points(p, rng, B):
    """Decision vectors near the native start: `c` +- 0.1, `z ~ 0.7 N(0, I)`, `q_c` small."""
    bp_c = np.asarray(p.initial_guess, dtype=float)
    width = p.ik_solver.network_width
    c = bp_c[None, :] + rng.uniform(-0.1, 0.1, size=(B, 6))
    z = 0.7 * rng.standard_normal((B, width))
    qc = rng.uniform(-0.05, 0.05, size=(B, p.num_arm_dof))
    return np.concatenate([c, z, qc], axis=1)


def test_freia_logdet_is_slogdet():
    for robot, ckpt in (("panda", "models/panda/panda__n6__step620000.pkl"),
                        ("iiwa14", "models/iiwa14/iiwa14__n6__step620000.pkl")):
        with HiddenPrints():
            solver = LoadFlowSolver(robot, os.path.join(RepoDir(), ckpt))
        net = solver.nn_model.to(torch.float64).eval()
        width = solver.network_width
        ndof = 7
        g = torch.Generator(device="cpu").manual_seed(0)
        B = 4
        cpu = dict(dtype=torch.float64, device="cpu")   # jrl sets a CUDA default device
        xyz = (0.3 * torch.randn(B, 3, generator=g, **cpu)
               + torch.tensor([0.4, 0.0, 0.5], **cpu))
        quat = torch.randn(B, 4, generator=g, **cpu)
        quat = quat / quat.norm(dim=1, keepdim=True)
        z = torch.randn(B, width, generator=g, **cpu)
        dev = next(net.parameters()).device
        infer = MakeFlowInference(net, width, ndof, dev, with_logdet=True)
        plain = MakeFlowInference(net, width, ndof, dev)
        diffs = []
        for i in range(B):
            v = torch.cat([xyz[i], quat[i], z[i], torch.zeros(ndof, **cpu)]).to(dev)
            with torch.no_grad():
                y, _ = infer(v)
                q, _ = plain(v)
            cond = torch.cat([xyz[i], quat[i], torch.zeros(1, **cpu)])[None].to(dev)
            J = torch.func.jacrev(lambda zz: net(zz[None], c=cond, rev=True)[0][0])(z[i].to(dev))
            sl = float(torch.linalg.slogdet(J)[1])
            diffs.append(sl - float(y[ndof]))
            assert torch.equal(y[:ndof], q), (robot, "q rows of the with_logdet call differ")
        spread = max(diffs) - min(diffs)
        print(f"  {robot}: width {width}, slogdet(jacrev) - FrEIA logdet = {np.mean(diffs):+.3e} "
              f"(spread {spread:.2e} over {B} points)")
        assert spread < 1e-5, (robot, diffs)
        assert abs(np.mean(diffs)) < 1e-4, (robot, diffs)
    print("PASS FrEIA log-det == slogdet(jacrev) up to a constant (Panda; iiwa: the (q, pad) map)")


def test_drake_cost_gradient_matches_central_differences():
    p = panda_pose()
    b = binding_of(p, "FlowLogDetCost")
    assert b is not None
    ev = b.evaluator()
    rng = np.random.default_rng(3)
    worst = 0.0
    for x in lumped_points(p, rng, 3):
        ad = ev.Eval(InitializeAutoDiff(x))
        val = float(ad[0].value())
        grad = np.asarray(ad[0].derivatives(), dtype=float)
        assert abs(val - float(ev.Eval(x)[0])) < 1e-12, "AutoDiffXd value != double value"
        ## and the value is w * the FrEIA log-det of the flow at x
        nld = p.FlowNegLogDensity(x)
        z = x[6:6 + p.ik_solver.network_width]
        assert abs(val / W - (nld - 0.5 * z @ z)) < 1e-10, (val, nld)
        h = 1e-6
        fd = np.zeros_like(grad)
        for k in range(len(x)):
            e = np.zeros_like(x)
            e[k] = h
            fd[k] = (float(ev.Eval(x + e)[0]) - float(ev.Eval(x - e)[0])) / (2 * h)
        rel = float(np.abs(grad - fd).max() / max(1e-12, np.abs(grad).max()))
        worst = max(worst, rel)
        print(f"  value {val:+.6f}  |grad|_inf {np.abs(grad).max():.3e}  "
              f"|grad - FD|_inf / |grad|_inf = {rel:.2e}")
    assert worst < 1e-6, worst
    print("PASS Drake FlowLogDetCost: AutoDiffXd gradient == central differences")


def test_batched_matches_drake():
    p = panda_pose()
    bp = BatchedProgram.from_program(p, dtype=torch.float64, device=DEVICE)
    try:
        assert bp.has_flow_logdet and bp._w_logdet == W
        rng = np.random.default_rng(5)
        B = 6
        Xn = lumped_points(p, rng, B)
        X = torch.tensor(Xn, dtype=torch.float64, device=bp.device, requires_grad=True)
        ev = bp.evaluate(X, need_collision=True)
        ld = ev.extras["flow_logdet"]
        b = binding_of(p, "FlowLogDetCost")
        ref_v, ref_g = np.zeros(B), np.zeros((B, bp.nvars))
        F_ref, Frep_ref = np.zeros(B), np.zeros(B)
        for i in range(B):
            ad = b.evaluator().Eval(InitializeAutoDiff(Xn[i]))
            ref_v[i] = float(ad[0].value())
            ref_g[i] = np.asarray(ad[0].derivatives(), dtype=float)
            x_full = bp.to_drake_x(Xn[i])
            for bb in p.prog.GetAllCosts():
                v = float(np.asarray(p.prog.EvalBinding(bb, x_full)).sum())
                F_ref[i] += v
                if bb.evaluator().get_description() not in REGULARIZER_COSTS:
                    Frep_ref[i] += v
        got_v = (W * ld).detach().cpu().numpy()
        (gl,) = torch.autograd.grad((W * ld).sum(), X, retain_graph=True)
        got_g = gl.cpu().numpy()
        dv = float(np.abs(got_v - ref_v).max())
        dg = float(np.abs(got_g - ref_g).max() / np.abs(ref_g).max())
        dF = float(np.abs(ev.F.detach().cpu().numpy() - F_ref).max())
        dFr = float(np.abs(ev.F_reported.detach().cpu().numpy() - Frep_ref).max())
        print(f"  batched vs Drake: value {dv:.2e}, gradient (rel) {dg:.2e}, F {dF:.2e}, "
              f"F_reported {dFr:.2e} (log-det term {np.abs(ref_v).max():.3f} excluded)")
        assert dv < 1e-9 and dg < 1e-9 and dF < 1e-9 and dFr < 1e-9, (dv, dg, dF, dFr)
        assert "FlowLogDetCost" in REGULARIZER_COSTS
        assert REGULARIZER_COSTS == benchmark._REGULARIZER_COSTS

        ## The split step's stage 2: one extra jacrev row, the log-det's.
        tg = _Target(bp, 1e-4)
        Xd = X.detach()
        with torch.no_grad():
            cfg = bp._config(Xd)
        with torch.enable_grad():
            out2 = fused.stage2(tg, Xd, cfg)
        assert len(out2) == 3, "stage 2 must return (J_q, kin, (ld, J_ld)) with the term on"
        J_q, kin, (ld2, J_ld) = out2
        dJ = float((J_ld - gl / W).abs().max() / (gl / W).abs().max())
        dl = float((ld2 - ld.detach()).abs().max())
        ## J_q against the plain (term-off) jacrev: equal up to the reordering one extra row
        ## may cause, which is why the extra row is behind the flag.
        J_plain = torch.func.jacrev(lambda Xv: bp._config(Xv).sum(dim=0))(Xd).permute(1, 0, 2)
        dq = float((J_q - J_plain).abs().max() / J_plain.abs().max())
        print(f"  stage 2: J_ld vs autograd (rel) {dJ:.2e}, ld {dl:.2e}, "
              f"J_q vs the plain jacrev (rel) {dq:.2e}")
        assert dJ < 1e-10 and dl < 1e-12 and dq < 1e-12, (dJ, dl, dq)

        ## The full-objective gradient stage 3 forms (J_q^T dF/dcfg + dF/dX|direct + w J_ld)
        ## against autograd of the batched F.
        dFc, dFx = bp.cost_gradient_parts(Xd, cfg)
        gF = (J_q.transpose(1, 2) @ dFc.unsqueeze(2)).squeeze(2) + dFx + bp._w_logdet * J_ld
        (gFa,) = torch.autograd.grad(ev.F.sum(), X, retain_graph=True)
        dgF = float((gF - gFa).abs().max() / gFa.abs().max())
        ## and the eager path's closed-form branch (`_Target.cost_parts`)
        tev = tg.evaluate(Xd)
        c_cfg, c_x = tg.cost_parts(tev)
        gE = (J_q.transpose(1, 2) @ c_cfg.unsqueeze(2)).squeeze(2) + c_x
        dgE = float((gE - gFa).abs().max() / gFa.abs().max())
        print(f"  objective gradient: stage-3 assembly vs autograd (rel) {dgF:.2e}, "
              f"_Target.cost_parts vs autograd (rel) {dgE:.2e}")
        assert dgF < 1e-10 and dgE < 1e-10, (dgF, dgE)
    finally:
        bp.close()
    print("PASS batched program: value, gradient, F, F_reported and the split step's J_ld "
          "match the Drake program")


def test_absent_at_zero_and_on_joint_space():
    p0 = panda_pose(w=0.0)
    assert binding_of(p0, "FlowLogDetCost") is None
    assert binding_of(p0, "LatentRegularizerCost") is None
    pn = panda_pose(w=W, arm="numerical")
    assert binding_of(pn, "FlowLogDetCost") is None
    ## and the w = 0 program's flow call is the plain one (no wrapper, no stash)
    assert not p0._FlowLogDetOn() and getattr(p0, "_logdet_stash", None) is None
    print("PASS no FlowLogDetCost at w = 0 or on the joint-space arm")


def test_options_refused():
    for kw in (dict(flow_logdet_weight=1.0, lift_q=True),
               dict(flow_logdet_weight=1.0, chart_error_scale=0.01),
               dict(flow_logdet_weight=-1.0)):
        try:
            ProgramOptions(**kw)
        except ValueError as exc:
            print(f"  refused {kw}: {exc}")
            continue
        raise AssertionError(f"ProgramOptions({kw}) accepted")
    ProgramOptions(flow_logdet_weight=1.0)
    print("PASS ProgramOptions refuses the term under lift_q / chart_error_scale / w < 0")


GROUPS = {"fast": [test_options_refused, test_freia_logdet_is_slogdet,
                   test_drake_cost_gradient_matches_central_differences,
                   test_batched_matches_drake, test_absent_at_zero_and_on_joint_space]}


def main():
    _tiers.run(GROUPS)
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
