"""Guards for the LOInK chart family (`src/charts/loink.py`, "the chart family" part), its
export contract, its trainer and its place in the program.

There is no test runner in this repo; run this by hand, CPU only, under the memory scope:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 systemd-run --user --scope -p MemoryMax=20G \
        -p MemorySwapMax=0 .venv/bin/python tests/test_loink_chart.py

Every model here is UNTRAINED (every model fit is a cluster job). The network core is
`tests/test_loink_core.py` / `tests/test_loink_reference.py`; this file protects:

1. The task representation: `Pose7ToXyzRpy` is Drake's `RollPitchYaw` (1e-12 on 1000 random
   rotations) and `XyzRpyToPose7` its inverse on the principal domain.
2. The training costs: `m` = 0.5 |q - q_nom|^2 at the programs' own nominal, `x` = LOInK eq. 18
   (0 at the limits' midpoint, -log(eps) at a limit), `u` = None.
3. The chart takes the program's RAW `c = [xyz, rpy]` (`task_input = "xyzrpy"`): shapes,
   `encode == G_2` (task ignored), `invert` -> (raw y_hat, z), and `decode(invert(q)) == q`
   to 1e-9 on EVERY row for Panda (K = 1) and soft12 (K = 6), including rows whose y_hat lies
   off the rpy principal domain; `decode` is continuous across roll = pi. `jacrev` through
   `MakeFlowInference(..., task_dim=6)` matches central differences in c6 and z, on and off
   the domain.
4. The export contract: bare state dict + mandatory `"family": "loink"` sidecar, identical
   round trip; a sidecar contradicting the weights RAISES (units, depth, ndof, mu, the head's
   presence, K, task_dim, head_hidden), a missing one RAISES, another family / robot RAISES, a
   same-DoF robot RAISES on the joint scale; the activation is the documented hole.
5. `LoadChart` dispatches `"loink"`; `loink_loss` is finite for m, x, u; the trainer's
   `training_step` / validation step run and its nu guard clamps.
6. The program: `PandaIKProgram(checkpoint=<untrained loink>)` constructs, `SetStartFromQ`
   is exact through `invert` on 8 random starts, 3 of them off the rpy domain
   (`start_q_error < 1e-8`, `start_c_error` finite and recorded); `VarsToQ`'s AutoDiffXd
   derivative matches central differences and is continuous across roll = pi;
   `BatchedProgram.init_from_q` equals `SetStartFromQ` and the float32 `BatchedFlow` takes the
   raw c; `SetNativeStart` works, one IPOPT `Solve()` with a 5 s wall runs, and
   `flow_logdet_weight > 0` is refused.
"""

import json
import os
import sys
import tempfile
import time
from dataclasses import replace

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "training"))

from src.charts.loink import (  # noqa: E402
    BARRIER_EPS, ArchFromLoinkStateDict, LoadLoinkChart, LoinkChart, LoinkModel, NominalConfiguration,
    Pose7ToXyzRpy, RpyCutDistance, TrainingCost, XyzRpyToPose7, loink_loss)
from export_loink import BuildLoinkModel, write_untrained  # noqa: E402
from jrl.robots import get_robot  # noqa: E402
from src.flow_loading import SidecarPath  # noqa: E402

CPU = torch.device("cpu")
F64 = torch.float64


def _gen(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def _expect_raise(fn, exc, what):
    try:
        fn()
    except exc as e:
        print(f"PASS {what} raises: {str(e).splitlines()[0][:110]}")
        return
    raise AssertionError(f"{what} did not raise")


def _q_in_limits(robot, n, g):
    lo, hi = torch.tensor(robot.actuated_joints_limits, dtype=F64, device=CPU).T
    return lo + (hi - lo) * torch.rand((n, robot.ndof), generator=g, dtype=F64, device=CPU)


def _chart(tmp, robot_name, cost, activation, name=None):
    out = write_untrained(robot_name, cost, activation,
                          out_path=os.path.join(tmp, name or f"{robot_name}_{cost}{activation}.pkl"))[0]
    return out, LoadLoinkChart(robot_name, out)


## ---------------------------------------------------------------------- 1

def test_pose7_rpy_matches_drake():
    from pydrake.all import Quaternion, RollPitchYaw, RotationMatrix
    rng = np.random.default_rng(0)
    quat = rng.normal(size=(1000, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    ref = np.array([RollPitchYaw(RotationMatrix(Quaternion(qq))).vector() for qq in quat])
    pose7 = torch.tensor(np.c_[rng.normal(size=(1000, 3)), quat], dtype=F64, device=CPU)
    got = Pose7ToXyzRpy(pose7)
    err = np.abs(got[:, 3:].numpy() - ref).max()
    assert err < 1e-12, err
    assert torch.equal(got[:, :3], pose7[:, :3])
    ## The sign of the quaternion is irrelevant.
    flipped = torch.cat([pose7[:, :3], -pose7[:, 3:]], dim=1)
    assert (Pose7ToXyzRpy(flipped) - got).abs().max() < 1e-14
    ## Round trip on the principal domain, and the quaternion is Drake's (w >= 0).
    back = XyzRpyToPose7(got)
    assert (Pose7ToXyzRpy(back) - got).abs().max() < 1e-12
    assert (back[:, 3] >= 0).all()
    drake_q = np.array([RotationMatrix(RollPitchYaw(r)).ToQuaternion().wxyz() for r in ref])
    assert np.abs(back[:, 3:].numpy() - drake_q).max() < 1e-12
    d = RpyCutDistance(got[:, 3:])
    assert (d >= 0).all() and (d <= np.pi / 2).all()
    print(f"PASS Pose7ToXyzRpy == Drake RollPitchYaw on 1000 rotations (max err {err:.1e}); "
          "XyzRpyToPose7 round trip")


## ---------------------------------------------------------------------- 2

def test_training_costs(panda, soft12):
    from src.panda_program import PandaIKProgram
    from src.iiwa_program import Iiwa14IKProgram
    assert np.array_equal(NominalConfiguration("panda", 7), PandaIKProgram.Q_NOMINAL)
    assert np.array_equal(NominalConfiguration("iiwa14", 7), Iiwa14IKProgram.Q_NOMINAL)
    m = TrainingCost("panda", "m", robot=panda)
    q_nom = torch.tensor(PandaIKProgram.Q_NOMINAL, dtype=F64, device=CPU)[None]
    assert float(m(q_nom)) == 0.0
    step = q_nom.clone()
    step[0, 3] += 0.4
    step[0, 5] -= 0.3
    assert abs(float(m(step)) - 0.5 * (0.16 + 0.09)) < 1e-15
    x = TrainingCost("panda", "x", robot=panda)
    lo, hi = torch.tensor(panda.actuated_joints_limits, dtype=F64, device=CPU).T
    mid = (0.5 * (lo + hi))[None]
    assert float(x(mid)) == 0.0, "x is not 0 at the limits' midpoint"
    q = mid.clone()
    q[0, 2] = mid[0, 2] + 0.5 * 0.5 * (hi[2] - lo[2])          # q_hat = 0.5 on one joint
    assert abs(float(x(q)) + np.log(0.75)) < 1e-12
    q[0, 2] = hi[2]                                              # at the limit: the floor
    assert abs(float(x(q)) + np.log(BARRIER_EPS)) < 1e-12
    ## soft12: zeros nominal, limits +-1, so m = 0.5 |q|^2 and x(0) = 0.
    ms, xs = TrainingCost("soft12", "m", robot=soft12), TrainingCost("soft12", "x", robot=soft12)
    qs = 0.3 * torch.ones((1, 12), dtype=F64, device=CPU)
    assert abs(float(ms(qs)) - 0.5 * 12 * 0.09) < 1e-14
    assert float(xs(torch.zeros((1, 12), dtype=F64, device=CPU))) == 0.0
    assert abs(float(xs(qs)) + 12 * np.log(1 - 0.09)) < 1e-12
    assert TrainingCost("panda", "u", robot=panda) is None
    ## float32 on the way in, float32 out (constants follow q).
    assert m(q_nom.float()).dtype == torch.float32
    print("PASS TrainingCost: m at the programs' nominal, x = eq. 18 (0 at the midpoint, "
          "-log(eps) at a limit), u = None")


## ---------------------------------------------------------------------- 3

def test_chart_functions(tmp, robot_name, robot, cost, activation):
    _, chart = _chart(tmp, robot_name, cost, activation)
    model = chart.nn_model.double()
    k = robot.ndof - 6
    assert isinstance(chart, LoinkChart) and chart.family == "loink"
    assert not chart.has_density and chart.exact_inverse and chart.network_width == k
    assert chart.task_input == "xyzrpy"
    g = _gen(1)
    q = _q_in_limits(robot, 256, g)
    with torch.no_grad():
        c6, z = chart.invert(model, q)
        assert c6.shape == (256, 6) and z.shape == (256, k)
        y_hat, z2 = model.invert(q)
        assert torch.equal(c6, y_hat) and torch.equal(z, z2), "invert is not the raw G(q)"
        assert torch.equal(chart.encode(model, q, c6), z), "encode is not G_2"
        assert torch.equal(chart.encode(model, q, torch.zeros_like(c6)), z), \
            "encode depends on the task"
        q_back, logdet = chart.decode(model, z, c6)
        assert logdet is None and q_back.shape == (256, robot.ndof)
        off = RpyCutDistance(c6[:, 3:6]) < 0
    err = (q_back - q).abs().max().item()
    assert err < 1e-9, err
    ## Continuity across the rpy cut: roll = pi -+ eps decode to neighbouring q (a pose7
    ## route would jump to roll = -pi + eps there).
    eps = 1e-7
    c_lo, c_hi = c6[:8].clone(), c6[:8].clone()
    c_lo[:, 3], c_hi[:, 3] = np.pi - eps, np.pi + eps
    with torch.no_grad():
        jump = (chart.decode(model, z[:8], c_hi)[0] - chart.decode(model, z[:8], c_lo)[0]).abs().max().item()
        wrapped = c_hi.clone()
        wrapped[:, 3] = -np.pi + eps
        far = (chart.decode(model, z[:8], wrapped)[0] - chart.decode(model, z[:8], c_lo)[0]).abs().max().item()
    assert jump < 1e-4, jump
    print(f"PASS chart functions ({robot_name} {cost}/{activation}, K={k}): decode(invert(q)) == q "
          f"on all 256 rows ({err:.1e}; {int(off.sum())} with y_hat off the rpy domain); decode "
          f"continuous across roll = pi ({jump:.1e} over 2e-7, against {far:.2f} at the wrapped roll)")
    return chart


def test_jacrev_through_make_flow_inference(chart, robot):
    """The program's 6-D task path: `MakeFlowInference(task_dim=6)` on `[c6 | z | correction]`,
    at an on-domain c and at one past the roll cut."""
    from src.generic_program import MakeFlowInference
    model = chart.nn_model.double()
    k, n = chart.network_width, robot.ndof
    g = _gen(2)
    q = _q_in_limits(robot, 2, g)
    with torch.no_grad():
        c6, _ = chart.invert(model, q)
    f = MakeFlowInference(model, k, n, CPU, decode=chart.decode, task_dim=6)
    errs = []
    for c in (c6[0], torch.cat([c6[1, :3], torch.tensor([3.3, 0.2, -3.25], dtype=F64)])):
        z = torch.randn(k, generator=g, dtype=F64, device=CPU)
        corr = 0.01 * torch.randn(n, generator=g, dtype=F64, device=CPU)
        vars = torch.cat([c, z, corr])
        J, q0 = torch.func.jacrev(f, has_aux=True)(vars)
        assert J.shape == (n, 6 + k + n)
        with torch.no_grad():
            assert torch.equal(q0, model.decode(c[None], z[None])[0] + corr)
        h = 1e-6
        fd = torch.stack([(f(vars + h * e)[0] - f(vars - h * e)[0]) / (2 * h)
                          for e in torch.eye(vars.numel(), dtype=F64, device=CPU)], dim=1)
        errs.append(((J - fd).abs().max() / max(1.0, J.abs().max())).item())
        assert torch.equal(J[:, 6 + k:], torch.eye(n, dtype=F64)), "correction block is not identity"
    assert max(errs) < 1e-6, errs
    print(f"PASS jacrev through MakeFlowInference(task_dim=6) v central differences in c6 and z "
          f"({robot.name}, K={k}): rel err {errs[0]:.1e} on-domain, {errs[1]:.1e} past the cut")


## ---------------------------------------------------------------------- 4

def test_export_roundtrip_identical(tmp):
    out, arch, hp, state_dict = write_untrained("panda", "m", "tanh",
                                                out_path=os.path.join(tmp, "rt.pkl"))
    with open(SidecarPath(out)) as f:
        sidecar = json.load(f)
    assert sidecar["family"] == "loink" and sidecar["provenance"]["untrained"] is True
    rec = ArchFromLoinkStateDict(state_dict)
    assert rec == {"ndof": 7, "depth": 2, "units": [256] * 4, "mu": rec["mu"], "has_head": True,
                   "task_dim": 6, "head_hidden": 128, "dim_latent_space": 1}, rec
    assert abs(rec["mu"] - 0.1) < 1e-7
    chart = LoadLoinkChart("panda", out)
    assert chart.arch["activation"] == "tanh" and chart.arch["n_dys"] == 100
    loaded = chart.nn_model.double()
    original = BuildLoinkModel(arch, hp["scale"]).double().eval()
    original.load_state_dict(state_dict)
    g = _gen(3)
    q = _q_in_limits(get_robot("panda"), 32, g)
    z = torch.randn((32, 1), generator=g, dtype=F64, device=CPU)
    with torch.no_grad():
        c6 = original.forward_G(q)[0]
        assert torch.equal(chart.decode(loaded, z, c6)[0], original(z, c6)), \
            "reloaded decode is not identical"
    print("PASS export -> LoadLoinkChart round trip identical (panda m/tanh)")


def test_contradicting_sidecars_raise(tmp):
    out, _ = _chart(tmp, "panda", "m", "relu", "bad.pkl")
    sidecar = SidecarPath(out)
    with open(sidecar) as f:
        good = json.load(f)
    for field, value in (("units", [256, 256, 256]), ("units", [128] * 4), ("depth", 3),
                         ("ndof", 6), ("mu", 0.05), ("cost", "u"), ("dim_latent_space", 2),
                         ("task_dim", 5), ("head_hidden", 64), ("j_std", None)):
        with open(sidecar, "w") as f:
            json.dump(dict(good, **{field: value}), f)
        _expect_raise(lambda: LoadLoinkChart("panda", out), ValueError,
                      f"a sidecar declaring {field}={value}")
    with open(sidecar, "w") as f:
        json.dump(dict(good, family="cvae"), f)
    _expect_raise(lambda: LoadLoinkChart("panda", out), ValueError, "a non-loink family")
    with open(sidecar, "w") as f:
        json.dump(dict(good, robot_name="iiwa14"), f)
    _expect_raise(lambda: LoadLoinkChart("panda", out), ValueError, "another robot's sidecar")
    with open(sidecar, "w") as f:
        json.dump({k: v for k, v in good.items() if k != "n_dys"}, f)
    _expect_raise(lambda: LoadLoinkChart("panda", out), ValueError, "a sidecar without n_dys")
    ## The fields no check can catch (ArchFromLoinkStateDict): a tanh sidecar over relu
    ## weights, and the cost letter x over an m head, both LOAD -- the holes that make the
    ## sidecar mandatory.
    with open(sidecar, "w") as f:
        json.dump(dict(good, activation="tanh", cost="x"), f)
    ch = LoadLoinkChart("panda", out)
    assert ch.arch["activation"] == "tanh" and ch.arch["cost"] == "x"
    print("PASS activation and m-v-x are not recoverable from the weights (documented holes)")

    ## A cost-"u" chart (no head) whose sidecar claims a head.
    out_u, _ = _chart(tmp, "panda", "u", "relu", "bad_u.pkl")
    with open(SidecarPath(out_u)) as f:
        good_u = json.load(f)
    assert good_u["j_mean"] is None and good_u["j_std"] is None
    with open(SidecarPath(out_u), "w") as f:
        json.dump(dict(good_u, cost="m", j_mean=0.0, j_std=1.0), f)
    _expect_raise(lambda: LoadLoinkChart("panda", out_u), ValueError,
                  "a cost-m sidecar over head-less weights")
    os.remove(sidecar)
    _expect_raise(lambda: LoadLoinkChart("panda", out), FileNotFoundError, "a missing sidecar")


def test_same_dof_other_robot_raises(tmp):
    try:
        get_robot("screw7_p050")
    except Exception as e:
        print(f"SKIP same-DoF guard: screw7_p050 does not resolve ({e})")
        return
    out, _ = _chart(tmp, "panda", "m", "relu", "panda_for_screw.pkl")
    with open(SidecarPath(out)) as f:
        payload = json.load(f)
    payload.pop("robot_name")
    with open(SidecarPath(out), "w") as f:
        json.dump(payload, f)
    _expect_raise(lambda: LoadLoinkChart("screw7_p050", out), ValueError,
                  "a Panda chart loaded as screw7_p050")


## ---------------------------------------------------------------------- 5

def test_load_chart_dispatch(tmp):
    from src.charts.loading import FAMILIES, ChartFamily, LoadChart
    assert "loink" in FAMILIES
    out, _ = _chart(tmp, "panda", "u", "tanh", "dispatch.pkl")
    assert ChartFamily(out) == "loink"
    ch = LoadChart("panda", out)
    assert type(ch) is LoinkChart and ch.arch["family"] == "loink" and ch.invert is not None
    print("PASS LoadChart dispatches 'loink' to LoadLoinkChart")


def test_loss_and_trainer(panda):
    from train_loink import LoinkLitModel, NU_GUARD_REL, RECIPE_DEFAULTS
    g = _gen(4)
    q = _q_in_limits(panda, 128, g)
    pose = panda.forward_kinematics(q, out_device="cpu").double()
    y = Pose7ToXyzRpy(pose)
    for cost in ("m", "x", "u"):
        torch.manual_seed(0)
        with torch.device("cpu"):
            model = LoinkModel(7, 6, [max(abs(a), abs(b)) for a, b in panda.actuated_joints_limits],
                               cost=cost).double()
        fn = TrainingCost("panda", cost, robot=panda)
        j = None if fn is None else (fn(q) - fn(q).mean()) / fn(q).std()
        loss, parts = loink_loss(model, q, y, j)
        assert torch.isfinite(loss) and (("aux_mse" in parts) == (cost != "u"))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
    print("PASS loink_loss finite with finite gradients for m, x, u (u has no aux term)")

    for cost, act, recipe in (("m", "relu", "P"), ("u", "tanh", "A")):
        d = RECIPE_DEFAULTS[recipe]
        torch.manual_seed(0)
        lit = LoinkLitModel(
            robot_name="panda", ndof=7, units=[256] * 4, depth=2, mu=0.1, activation=act,
            cost=cost, head_hidden=128, n_dys=100, n_newton=2,
            scale=[max(abs(a), abs(b)) for a, b in panda.actuated_joints_limits],
            j_mean=None if cost == "u" else 0.6, j_std=None if cost == "u" else 0.3, lam=1.0,
            tau_penalty=1e-3, recipe=recipe, optimizer=d["optimizer"],
            learning_rate=d["learning_rate"], weight_decay=d["weight_decay"],
            scheduler=d["scheduler"], gamma=0.9795, step_lr_every=2441, plateau_factor=0.5,
            plateau_patience=500, plateau_ema=0.99, gradient_clip=d["gradient_clip_val"],
            log_every=1000, fk_eval_n=16, inv_eval_n=16, run_dir=tempfile.gettempdir())
        qf, posef = q.float(), pose.float()
        loss = lit.training_step((qf, posef), 0)
        assert torch.isfinite(loss) and torch.isfinite(lit.loss_ema)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in lit.model.parameters())
        ## The nu guard: a layer pushed below mu is clamped back and counted.
        with torch.no_grad():
            lit.model.G.mon[0].lognu.fill_(-5.0)
        lit.on_train_batch_end(None, None, 0)
        mon = lit.model.G.mon[0]
        assert torch.isclose(mon.lognu, torch.log(mon.mu * (1 + NU_GUARD_REL))).all()
        assert int(lit.nu_clamped_steps) == 1 and bool(lit.model.G.check_bounds())
        lit.validation_step((qf, posef), 0)
        val = lit.last_val_metrics
        for key in ("val/pos_rmse_mm", "val/rot_rmse_deg", "val/tau", "val/inv_residual_max",
                    "val/fk_pos_mm_median"):
            assert val[key] == val[key], f"NaN {key}"
        assert (cost == "u") == ("val/aux_mse" not in val)
        print(f"PASS trainer {cost}/{act}/{recipe}: training_step finite ({loss.item():.3f}), nu "
              f"guard clamps, validation step runs (inv residual {val['val/inv_residual_max']:.1e}, "
              f"fk median {val['val/fk_pos_mm_median']:.0f} mm)")


## ---------------------------------------------------------------------- 6

def test_program(tmp):
    from src.benchmark import start_diagnostics
    from src.generic_program import ProgramOptions
    from src.panda_program import PandaIKProgram
    from src.target_screening import SceneFile
    from src.utils import BuildEnv, HiddenPrints

    out, _ = _chart(tmp, "panda", "m", "relu", "program.pkl")
    opts = ProgramOptions(visualize=False, correction_cost_weight=10.0, max_wall_time=5.0,
                          which_solver="ipopt", latent_trust_region=2.5,
                          share_flow_evaluations=True, joint_centering_cost=1e-4)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=SceneFile("panda", "pose", "hardened"))
    _expect_raise(lambda: PandaIKProgram(diagram, options=replace(opts, flow_logdet_weight=1.0),
                                         checkpoint=out), ValueError,
                  "flow_logdet_weight > 0 with a loink chart")
    t = time.monotonic()
    with HiddenPrints():
        program = PandaIKProgram(diagram, options=opts, checkpoint=out)
    assert isinstance(program.ik_solver, LoinkChart)
    chart = program.ik_solver

    ## Several starts, at least three of whose chart task estimates y_hat lie OFF the rpy
    ## principal domain: the raw-c path must represent every one exactly.
    rng = np.random.default_rng(0)
    lower, upper = program.plant.GetPositionLowerLimits()[:7], program.plant.GetPositionUpperLimits()[:7]
    model = chart.nn_model
    on, off = [], []
    for _ in range(20000):
        q = rng.uniform(lower, upper)
        with torch.no_grad():
            c6, _ = chart.invert(model, torch.tensor(q[None], dtype=program.torch_dtype, device=CPU))
        (off if bool(RpyCutDistance(c6[:, 3:6])[0] < 0) else on).append(q)
        if len(on) >= 5 and len(off) >= 3:
            break
    qs = on[:5] + off[:3]
    assert len(qs) == 8, (len(on), len(off))
    refs, errs, cerrs = [], [], []

    def fresh():
        ## One program per cell, sharing the chart, as the benchmark builds them: a program's
        ## `create_prog` appends its constraints, so it is not reusable for a second cell.
        with HiddenPrints():
            return PandaIKProgram(diagram, options=opts, model=chart)

    for q_init in qs:
        program = fresh()
        program.plant.SetPositions(program.plant_context, program.PadQ(q_init))
        X = program.frame.CalcPoseInWorld(program.plant_context)
        target = np.concatenate([X.translation(), X.rotation().ToQuaternion().wxyz()])
        with HiddenPrints():
            program.create_prog(target)
            program.clip_distance = program.SetStartFromQ(q_init)
        diag = start_diagnostics(program, q_init)
        assert diag["start_q_error"] < 1e-8, diag
        assert diag["start_c_error"] is not None and np.isfinite(diag["start_c_error"]), diag
        c0 = program.prog.GetInitialGuess(program.c)
        c_chart, _ = program.InvertChart(q_init)
        assert np.array_equal(c0, c_chart), "c was not taken from the chart"
        refs.append(np.asarray(program.prog.GetInitialGuess(program.lumped_vars), dtype=float))
        errs.append(diag["start_q_error"])
        cerrs.append(diag["start_c_error"])
    print(f"PASS SetStartFromQ through invert on 8 starts (3 with y_hat off the rpy domain): "
          f"max start_q_error {max(errs):.1e}, start_c_error {min(cerrs):.2f}..{max(cerrs):.2f} "
          f"(constructed in {time.monotonic() - t:.1f} s)")

    ## The program's VarsToQ at the last (off-domain) start: AutoDiffXd derivative v central
    ## differences of the float path, and continuity across roll = pi.
    from pydrake.all import AutoDiffXd, ExtractGradient, InitializeAutoDiff
    x0 = refs[-1]
    q_ad = program.VarsToQ(InitializeAutoDiff(x0).reshape(-1))[:7]
    J_ad = ExtractGradient(np.array(q_ad))
    v_ad = np.array([v.value() for v in q_ad])
    assert np.array_equal(v_ad, np.asarray(program.VarsToQ(x0)[:7], dtype=float))
    h = 1e-6
    J_fd = np.stack([(program.VarsToQ(x0 + h * e)[:7] - program.VarsToQ(x0 - h * e)[:7]) / (2 * h)
                     for e in np.eye(x0.size)], axis=1)
    err_ad = np.abs(J_ad - J_fd).max() / max(1.0, np.abs(J_ad).max())
    assert err_ad < 1e-6, err_ad
    x_lo, x_hi = x0.copy(), x0.copy()
    x_lo[3], x_hi[3] = np.pi - 1e-7, np.pi + 1e-7
    jump = np.abs(program.VarsToQ(x_hi)[:7] - program.VarsToQ(x_lo)[:7]).max()
    assert jump < 1e-4, jump
    print(f"PASS VarsToQ through the raw c: AutoDiffXd v central differences rel err {err_ad:.1e}; "
          f"continuous across roll = pi ({jump:.1e} over 2e-7)")

    ## The batched SetStartFromQ (AL-SVGD's `init_from_q`) takes the raw c from the chart too.
    from src.svgd.batched_program import BatchedProgram
    Q = np.stack(qs)
    bp = BatchedProgram.from_program(program, dtype=torch.float64, device=CPU)
    try:
        Xb = bp.init_from_q(torch.tensor(Q, dtype=F64, device=CPU)).numpy()
        d = np.abs(Xb - np.stack(refs)).max()
        q_back = bp.evaluate(torch.tensor(Xb, dtype=F64, device=CPU)).q.detach().numpy()
        rt = np.abs(q_back - Q).max()
    finally:
        bp.close()
    assert d <= 1e-9 and rt <= 1e-8, (d, rt)
    ## AL-SVGD's particles are float32, on a private copy of the chart (`BatchedFlow`).
    from src.svgd.batched_flow import BatchedFlow
    bf32 = BatchedFlow.from_program(program, dtype=torch.float32, device=CPU)
    assert bf32.task_input == "xyzrpy" and bf32.task_dim == 6
    q32 = bf32.q_from_vars(torch.tensor(Xb, dtype=torch.float32, device=CPU))
    e32 = (q32.double() - torch.tensor(Q, dtype=F64)).abs().max().item()
    assert q32.dtype == torch.float32 and e32 < 1e-3, e32
    assert next(program.ik_solver.nn_model.parameters()).dtype == F64, "shared chart was cast"
    print(f"PASS BatchedProgram.init_from_q == SetStartFromQ on the 8 starts (|dX| {d:.1e}, "
          f"|q(start) - q_init| {rt:.1e}); float32 BatchedFlow decode {e32:.1e} off")

    program = fresh()
    with HiddenPrints():
        program.create_prog(target)
        program.SetNativeStart(q_init, np.random.default_rng(1))
    z0 = program.prog.GetInitialGuess(program.z)
    assert z0.shape == (1,) and np.isfinite(z0).all()
    t = time.monotonic()
    with HiddenPrints():
        result = program.Solve()
    wall = time.monotonic() - t
    x = result.GetSolution(program.lumped_vars)
    assert np.isfinite(x).all() and wall < 60, wall
    print(f"PASS native start + one IPOPT Solve() ({wall:.1f} s, 5 s wall): solver success "
          f"{result.is_success()}, |z| {np.linalg.norm(result.GetSolution(program.z)):.2f}")


if __name__ == "__main__":
    t0 = time.monotonic()
    panda = get_robot("panda")
    soft12 = get_robot("soft12")
    test_pose7_rpy_matches_drake()
    test_training_costs(panda, soft12)
    with tempfile.TemporaryDirectory() as tmp:
        for robot_name, robot, cost, act in (("panda", panda, "m", "relu"),
                                             ("panda", panda, "u", "tanh"),
                                             ("soft12", soft12, "x", "tanh"),
                                             ("soft12", soft12, "m", "relu")):
            chart = test_chart_functions(tmp, robot_name, robot, cost, act)
            test_jacrev_through_make_flow_inference(chart, robot)
        test_export_roundtrip_identical(tmp)
        test_contradicting_sidecars_raise(tmp)
        test_same_dof_other_robot_raises(tmp)
        test_load_chart_dispatch(tmp)
        test_loss_and_trainer(panda)
        test_program(tmp)
    print(f"ALL PASS ({time.monotonic() - t0:.1f} s)")
