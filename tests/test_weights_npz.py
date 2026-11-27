"""The pickle-free chart weight format (`src/weights_io.py`, `.npz`) against the legacy `.pkl`.

    .venv/bin/python tests/test_weights_npz.py        (alone, under the memory scope)

Pinned:
  1. an `.npz` loads with NO pickle: `pickle.load`, `pickle.Unpickler` and `torch.load` are
     patched to raise while the flow loader (`LoadFlowSolver`) and the cVAE loader read one;
     an object array in an `.npz` is refused outright;
  2. a flow chart's `.npz` and `.pkl` give a bit-identical state dict AND bit-identical network
     outputs (CPU float64, fixed input), the same for a cVAE chart;
  3. the sidecar is found by stem for both extensions (`SidecarPath`).
Checks needing the gitignored weights skip cleanly when they are absent. The weights are looked
up in `$LEARNED_IK_MODELS_DIR` or `<repo>/models`, and the `.npz` is made by
`scripts/training/convert_pkl_to_npz.py`. Tiers (tests/_tiers.py): `fast` only, a few seconds.
"""
import os
import pickle
import sys
import tempfile
from unittest import mock

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)
import _tiers                                                             # noqa: E402

from src.flow_loading import LoadFlowSolver, SidecarPath              # noqa: E402
from src.utils import RepoDir                                          # noqa: E402
from src.weights_io import LoadStateDictNpz, SaveStateDictNpz          # noqa: E402

MODELS = os.environ.get("LEARNED_IK_MODELS_DIR") or os.path.join(RepoDir(), "models")
FLOW = ("panda", "panda/panda__n6__step620000")
CVAE = ("panda", "panda/panda__cvae7__step620000")


def _pair(rel):
    pkl, npz = (os.path.join(MODELS, rel + ext) for ext in (".pkl", ".npz"))
    return (pkl, npz) if os.path.exists(pkl) and os.path.exists(npz) else None


def _no_pickle():
    def refuse(*a, **k):
        raise AssertionError("pickle machinery reached while loading an .npz")
    return (mock.patch.object(pickle, "load", refuse), mock.patch.object(pickle, "loads", refuse),
            mock.patch.object(pickle, "Unpickler", refuse), mock.patch.object(torch, "load", refuse))


def _scale(patches):
    class _All:
        def __enter__(self):
            for p in patches:
                p.start()

        def __exit__(self, *exc):
            for p in patches:
                p.stop()
    return _All()


def test_npz_roundtrip_and_object_arrays_refused():
    sd = {"a.weight": torch.randn(3, 4, dtype=torch.float64), "b": torch.arange(5),
          "c": torch.tensor(2.5, dtype=torch.float32)}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.npz")
        SaveStateDictNpz(path, sd)
        try:
            SaveStateDictNpz(path, sd)
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwrote an existing .npz")
        with _scale(_no_pickle()):
            back = LoadStateDictNpz(path)
        assert list(back) == list(sd)
        for k in sd:
            assert back[k].dtype == sd[k].dtype and torch.equal(back[k], sd[k]), k
        bad = os.path.join(d, "obj.npz")
        np.savez(bad, w=np.array([{"x": 1}], dtype=object))
        try:
            LoadStateDictNpz(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("an object array was loaded")
    assert SidecarPath("m/x.pkl") == SidecarPath("m/x.npz") == "m/x.arch.json"
    print("PASS npz round trip, refuse-overwrite, object arrays refused, sidecar by stem")


def _outputs(solver):
    model = solver.nn_model.double().eval()
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4, solver.network_width, generator=g, dtype=torch.float64).to(dev)
    c = torch.tensor([[0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]] * 4, dtype=torch.float64,
                     device=dev)
    c[:, :3] += 0.05 * torch.randn(4, 3, generator=g, dtype=torch.float64).to(dev)
    with torch.no_grad():
        q, _ = model(z, c=c, rev=True)
    return q.cpu()


def test_flow_npz_matches_pkl_bit_for_bit(chart=FLOW):
    robot, rel = chart
    paths = _pair(rel)
    if paths is None:
        print(f"SKIP flow chart {rel}: weights absent")
        return
    pkl, npz = paths
    ref = LoadFlowSolver(robot, pkl)
    with _scale(_no_pickle()):
        new = LoadFlowSolver(robot, npz)
    assert ref.arch == new.arch and ref.arch_source == new.arch_source
    a, b = ref.nn_model.state_dict(), new.nn_model.state_dict()
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    qa, qb = _outputs(ref), _outputs(new)
    assert torch.equal(qa, qb), f"outputs differ by {(qa - qb).abs().max().item()}"
    print(f"PASS {rel}: {len(a)} tensors and the float64 forward pass are bit-identical")


def test_cvae_npz_matches_pkl_bit_for_bit(chart=CVAE):
    from src.charts.cvae import LoadCVAEChart
    robot, rel = chart
    paths = _pair(rel)
    if paths is None:
        print(f"SKIP cvae chart {rel}: weights absent")
        return
    pkl, npz = paths
    ref = LoadCVAEChart(robot, pkl)
    with _scale(_no_pickle()):
        new = LoadCVAEChart(robot, npz)
    a, b = ref.nn_model.state_dict(), new.nn_model.state_dict()
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    print(f"PASS {rel}: {len(a)} cVAE tensors bit-identical")


def test_upstream_panda_npz_matches_download_bit_for_bit():
    """The upstream chart: `LoadUpstreamPandaSolver` (the `.npz`, no pickle) against ikflow's own
    `get_ik_solver` (its downloaded pickle). Skips when either is absent -- never downloads."""
    from src.flow_loading import UPSTREAM_PANDA_NAME, UpstreamPandaCheckpoint, LoadUpstreamPandaSolver
    npz = UpstreamPandaCheckpoint()
    from ikflow.config import MODELS_DIR
    from ikflow.model_loading import MODEL_DESCRIPTIONS, get_ik_solver, model_filename
    cached = os.path.join(MODELS_DIR, model_filename(MODEL_DESCRIPTIONS[UPSTREAM_PANDA_NAME]["model_weights_url"]))
    if npz is None or not os.path.exists(cached):
        print("SKIP upstream panda chart: .npz or the ikflow cached pickle absent")
        return
    ref = get_ik_solver(UPSTREAM_PANDA_NAME)[0]
    with _scale(_no_pickle()):
        new = LoadUpstreamPandaSolver()
    a, b = ref.nn_model.state_dict(), new.nn_model.state_dict()
    assert list(a) == list(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    qa, qb = _outputs(ref), _outputs(new)
    assert torch.equal(qa, qb), f"outputs differ by {(qa - qb).abs().max().item()}"
    print(f"PASS upstream panda: {len(a)} tensors and the float64 forward pass are bit-identical")


GROUPS = {"fast": [test_npz_roundtrip_and_object_arrays_refused,
                   test_flow_npz_matches_pkl_bit_for_bit,
                   test_cvae_npz_matches_pkl_bit_for_bit,
                   test_upstream_panda_npz_matches_download_bit_for_bit]}


def main():
    _tiers.run(GROUPS)
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
