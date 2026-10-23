"""The chart interface (`src/charts`): the flow adapter changes nothing, loading dispatches on
the sidecar's `family`, and a density option refuses a chart without a density.

    CUDA_VISIBLE_DEVICES= .venv/bin/python tests/test_chart_interface.py   (alone, under the memory scope)

What is pinned:

  1. `FlowChart.decode` / `encode` are BIT-IDENTICAL (`torch.equal`) to the direct FrEIA
     calls the program made before the interface -- `nn_model(z, c=[pose7, 0], rev=True)`
     and the zero-padded `rev=False` call -- values and log-det, on the Panda n6 and the
     iiwa n6 charts, in float64 and float32. So is `MakeFlowInference` with the chart's
     decode against its flow default, value and `jacrev`, with and without the log-det row.
     This is the unit-level half of the refactor's bit-identity; the benchmark-level half
     (per-cell records before and after, IPOPT and AL-SVGD) was run once at the refactor.
  2. `LoadChart` on a sidecar WITHOUT `family` gives a `FlowChart` (every pre-ablation
     sidecar), an explicit `"family": "ikflow"` likewise, an unknown family raises;
     `WrapChart` wraps a bare solver and passes a chart through; `"family": "loink"` gives a
     `LoinkChart`, the one family with an `invert` (the flow's is None).
  3. `flow_logdet_weight > 0` with a chart whose `has_density` is False is REFUSED at program
     construction (a dummy chart: the flow's own decode / encode with the density switched
     off), while the same chart at weight 0 constructs; `FlowNegLogDensity` refuses it too.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(HERE))

from src.charts.base import Chart                                       # noqa: E402
from src.charts.flow import FlowChart                                   # noqa: E402
from src.charts.loading import LoadChart, WrapChart                     # noqa: E402
from src.flow_loading import LoadFlowSolver, SidecarPath                # noqa: E402
from src.generic_program import DEVICE, MakeFlowInference, ProgramOptions  # noqa: E402
from src.utils import BuildEnv, HiddenPrints, RepoDir                   # noqa: E402

CHECKPOINTS = {"panda": "models/panda/panda__n6__step620000.pkl",
               "iiwa14": "models/iiwa14/iiwa14__n6__step620000.pkl"}
_CHARTS = {}


def chart(robot):
    if robot not in _CHARTS:
        with HiddenPrints():
            _CHARTS[robot] = LoadChart(robot, os.path.join(RepoDir(), CHECKPOINTS[robot]))
    return _CHARTS[robot]


def _inputs(width, ndof, dtype, dev, n=16, seed=0):
    ## Drawn on the CPU explicitly: importing jrl sets the process default device to CUDA
    ## whenever one is visible, and a CPU generator refuses a CUDA draw.
    g = torch.Generator(device="cpu").manual_seed(seed)
    cpu = dict(generator=g, dtype=torch.float64, device="cpu")
    z = torch.randn((n, width), **cpu)
    xyz = 0.3 * torch.randn((n, 3), **cpu) + torch.tensor([0.4, 0.0, 0.5], dtype=torch.float64, device="cpu")
    quat = torch.randn((n, 4), **cpu)
    quat = quat / quat.norm(dim=1, keepdim=True)
    q = 2.0 * torch.rand((n, ndof), **cpu) - 1.0
    pose7 = torch.cat([xyz, quat], dim=1)
    return (z.to(dtype=dtype, device=dev), pose7.to(dtype=dtype, device=dev),
            q.to(dtype=dtype, device=dev))


def test_flow_chart_bit_identical():
    for robot in CHECKPOINTS:
        ch = chart(robot)
        assert isinstance(ch, FlowChart) and ch.family == "ikflow"
        assert ch.has_density and ch.exact_inverse
        width, ndof = ch.network_width, ch.num_dof
        for dtype in (torch.float64, torch.float32):
            ## A private cast, so the shared module's dtype is left alone.
            m = copy.deepcopy(ch.nn_model).to(dtype).eval()
            dev = next(m.parameters()).device
            z, pose7, q = _inputs(width, ndof, dtype, dev)
            c = torch.cat([pose7, torch.zeros((pose7.shape[0], 1), dtype=dtype, device=dev)], dim=1)
            with torch.no_grad():
                ## decode against the direct reverse call
                ref_out, ref_ld = m(z, c=c, rev=True)
                got_q, got_ld = ch.decode(m, z, pose7)
                assert torch.equal(got_q, ref_out[:, :ndof]), (robot, dtype, "decode q")
                assert torch.equal(got_ld, ref_ld), (robot, dtype, "decode logdet")
                ## encode against the zero-padded forward call
                x = torch.zeros((q.shape[0], width), dtype=dtype, device=dev)
                x[:, :ndof] = q
                ref_z, _ = m(x, c=c, rev=False)
                got_z = ch.encode(m, q, pose7)
                assert torch.equal(got_z, ref_z), (robot, dtype, "encode")

            ## MakeFlowInference: the chart's decode against the flow default (what every
            ## caller without a chart -- the screens, the probes -- still gets), value and
            ## jacrev, plain and with the log-det row.
            vars = torch.cat([pose7[0], z[0], 0.01 * q[0]])
            for with_logdet in (False, True):
                a = MakeFlowInference(m, width, ndof, dev, with_logdet=with_logdet)
                b = MakeFlowInference(m, width, ndof, dev, with_logdet=with_logdet,
                                      decode=ch.decode)
                with torch.no_grad():
                    assert torch.equal(a(vars)[0], b(vars)[0]), (robot, dtype, with_logdet)
                ja, ya = torch.func.jacrev(a, has_aux=True)(vars)
                jb, yb = torch.func.jacrev(b, has_aux=True)(vars)
                assert torch.equal(ja, jb) and torch.equal(ya, yb), (robot, dtype, with_logdet)
            print(f"  [{robot} {str(dtype)[6:]}] decode / encode / logdet / MakeFlowInference "
                  "value + jacrev: bit-identical")
    print("PASS FlowChart is bit-identical to the direct flow calls")


def test_load_chart_family_dispatch():
    ckpt = os.path.join(RepoDir(), CHECKPOINTS["panda"])
    with open(SidecarPath(ckpt)) as f:
        sidecar = json.load(f)
    assert "family" not in sidecar, "the shipped Panda sidecar predates `family`"
    ch = chart("panda")
    assert type(ch) is FlowChart and ch.arch["family"] == "ikflow"
    assert ch.arch_source == SidecarPath(ckpt)

    tmp = tempfile.mkdtemp(prefix="chart_family_")
    try:
        def staged(family):
            stem = os.path.join(tmp, f"panda__{family}")
            pkl = stem + ".pkl"
            if not os.path.exists(pkl):
                os.symlink(ckpt, pkl)
            with open(stem + ".arch.json", "w") as f:
                json.dump(dict(sidecar, family=family), f)
            return pkl
        with HiddenPrints():
            explicit = LoadChart("panda", staged("ikflow"))
        assert type(explicit) is FlowChart
        assert torch.equal(next(explicit.nn_model.parameters()).cpu(),
                           next(ch.nn_model.parameters()).cpu())
        try:
            LoadChart("panda", staged("bogus"))
        except ValueError as exc:
            print(f"  refused: {exc}")
        else:
            raise AssertionError("an unknown chart family loaded")
    finally:
        shutil.rmtree(tmp)

    with HiddenPrints():
        bare = LoadFlowSolver("panda", ckpt)
    wrapped = WrapChart(bare)
    assert type(wrapped) is FlowChart and wrapped.solver is bare
    assert wrapped.nn_model is bare.nn_model and wrapped.network_width == bare.network_width
    assert WrapChart(wrapped) is wrapped
    print("PASS LoadChart: no family = ikflow, explicit ikflow loads, unknown raises; WrapChart")


def test_load_chart_loink_dispatch():
    """A `"family": "loink"` sidecar loads a `LoinkChart` (untrained, written to a temp dir);
    it carries `invert`, while the flow's `invert` is the base default None -- the switch
    `SetStartFromQ` reads, so the flow and cVAE paths stay as they were."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "training"))
    from export_loink import write_untrained
    from src.charts.loink import LoinkChart

    assert chart("panda").invert is None, "the flow chart must not carry an invert"
    tmp = tempfile.mkdtemp(prefix="chart_loink_")
    try:
        with HiddenPrints():
            pkl = write_untrained("panda", "m", "relu", out_path=os.path.join(tmp, "l.pkl"))[0]
            ch = LoadChart("panda", pkl)
        assert type(ch) is LoinkChart and ch.family == "loink" and ch.arch["family"] == "loink"
        assert ch.invert is not None and not ch.has_density and ch.exact_inverse
        assert ch.network_width == 1 and WrapChart(ch) is ch
    finally:
        shutil.rmtree(tmp)
    print("PASS LoadChart: family loink -> LoinkChart (with invert); the flow's invert is None")


class DensityFreeChart(Chart):
    """The flow with its density switched off: everything a program reads, nothing else."""
    family = "dummy"
    has_density = False
    exact_inverse = True

    def __init__(self, flow):
        self.nn_model, self.network_width, self.robot = (
            flow.nn_model, flow.network_width, flow.robot)
        self.arch, self.arch_source = {"family": self.family}, None
        self.decode, self.encode = flow.decode, flow.encode


def test_density_option_refused_without_density():
    from src.panda_program import PandaIKProgram
    from src.target_screening import SceneFile

    dummy = DensityFreeChart(chart("panda"))
    opts = ProgramOptions(visualize=False, correction_cost_weight=10.0)
    with HiddenPrints():
        diagram = BuildEnv(meshcat=None, directives_file=SceneFile("panda", "pose", "hardened"))
    try:
        with HiddenPrints():
            PandaIKProgram(diagram, options=replace(opts, flow_logdet_weight=1.0), model=dummy)
    except ValueError as exc:
        print(f"  refused: {exc}")
        assert "density" in str(exc)
    else:
        raise AssertionError("flow_logdet_weight > 0 accepted a chart without a density")
    with HiddenPrints():
        p = PandaIKProgram(diagram, options=opts, model=dummy)
    assert p.ik_solver is dummy
    try:
        p.FlowNegLogDensity(np.zeros(6 + dummy.network_width + 7))
    except ValueError:
        pass
    else:
        raise AssertionError("FlowNegLogDensity evaluated a chart without a density")
    print("PASS flow_logdet_weight > 0 refuses a chart without a density; w = 0 constructs")


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    test_flow_chart_bit_identical()
    test_load_chart_family_dispatch()
    test_load_chart_loink_dispatch()
    test_density_option_refused_without_density()
    print("ALL PASS")
