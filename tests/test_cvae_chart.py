"""Guards for the cVAE chart (`src/charts/cvae.py`), its export contract and its trainer.

There is no test runner in this repo; run this by hand, under the local memory scope:

    .venv/bin/python tests/test_cvae_chart.py

Every model here is UNTRAINED: every model fit is a cluster job, so the trainer is tested
through one `training_step` on a random batch, never a fit. What is protected:

1. The chart's two functions are what the program needs: batch-first shapes, float64 on a
   deep-copied dtype-cast module (how `batched_flow` evaluates private copies), `encode` is
   the encoder MEAN (the approved "encoder only" paired start), and `torch.func.jacrev`
   through `decode` matches central differences in both `z` and the pose -- the Jacobian the
   solvers step on.
2. The export contract: a bare state dict plus a mandatory `"family": "cvae"` sidecar that
   `LoadCVAEChart` cross-checks against the weights. A sidecar that contradicts the weights
   (K, depth, hidden width) RAISES; a missing sidecar RAISES (there is no legacy fallback:
   the activation cannot be recovered from shapes, the cVAE's `rnvp_clamp`); a chart for
   another robot of the same DoF RAISES on its joint scale.
3. The trainer's loss is finite and differentiable on a batch, its validation step (FK
   screen included) runs, and its batched dataloader keeps each sample with its own pose.
"""

import copy
import json
import os
import pickle
import sys
import tempfile
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "training"))

from src.charts.cvae import (  # noqa: E402
    ArchFromCVAEStateDict, CVAEModel, JointScale, LoadCVAEChart, SidecarPath, elbo_loss)
from export_cvae import write_untrained  # noqa: E402
from jrl.robots import get_robot  # noqa: E402

CPU = torch.device("cpu")


def _poses(n, g):
    pose = torch.randn(n, 7, generator=g, dtype=torch.float64, device=CPU)
    pose[:, 3:] = pose[:, 3:] / pose[:, 3:].norm(dim=1, keepdim=True)
    return pose


def _model(robot, k, seed=0):
    torch.manual_seed(seed)
    with torch.device("cpu"):
        return CVAEModel(robot.ndof, k, JointScale(robot)).double().eval()


def _write(tmp, robot_name, k, name):
    return write_untrained(robot_name, k, out_path=os.path.join(tmp, name))[0]


def test_shapes_float64_and_encode_is_mu(robot, k):
    from src.charts.cvae import CVAEChart
    model = _model(robot, k)
    chart = CVAEChart(model, robot, dict(model.arch))
    g = torch.Generator(device="cpu").manual_seed(1)
    pose = _poses(16, g)
    z = torch.randn(16, k, generator=g, dtype=torch.float64, device=CPU)
    q, logdet = chart.decode(model, z, pose)
    assert logdet is None and q.shape == (16, robot.ndof) and q.dtype == torch.float64
    mu = chart.encode(model, q, pose)
    assert mu.shape == (16, k)
    assert torch.equal(mu, model.encode_dist(q, pose)[0]), "encode is not the encoder mean"

    ## A deep-copied, dtype-cast private copy, as batched_flow makes: float32 in, float32
    ## out, and its values agree with the float64 original to float32 precision.
    copy32 = copy.deepcopy(model).float()
    q32, _ = chart.decode(copy32, z.float(), pose.float())
    assert q32.dtype == torch.float32
    assert (q32.double() - q).abs().max() < 1e-4
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PASS shapes / float64 / encode = mu ({robot.name}, K={k}, {n_params} parameters)")
    return n_params


def test_jacrev_matches_central_differences(robot, k):
    from src.charts.cvae import CVAEChart
    model = _model(robot, k)
    chart = CVAEChart(model, robot, dict(model.arch))
    g = torch.Generator(device="cpu").manual_seed(2)
    pose = _poses(1, g)[0]
    z = torch.randn(k, generator=g, dtype=torch.float64, device=CPU)

    def f(z1, p1):
        return chart.decode(model, z1[None], p1[None])[0][0]

    jz, jp = torch.func.jacrev(f, argnums=(0, 1))(z, pose)
    h = 1e-6
    fd_z = torch.stack([(f(z + h * e, pose) - f(z - h * e, pose)) / (2 * h)
                        for e in torch.eye(k, dtype=torch.float64, device=CPU)], dim=1)
    fd_p = torch.stack([(f(z, pose + h * e) - f(z, pose - h * e)) / (2 * h)
                        for e in torch.eye(7, dtype=torch.float64, device=CPU)], dim=1)
    err_z = (jz - fd_z).abs().max().item() / max(1.0, jz.abs().max().item())
    err_p = (jp - fd_p).abs().max().item() / max(1.0, jp.abs().max().item())
    assert err_z < 1e-7 and err_p < 1e-7, (err_z, err_p)
    print(f"PASS jacrev v central differences ({robot.name}, K={k}): "
          f"rel err dq/dz {err_z:.1e}, dq/dpose {err_p:.1e}")


def test_export_roundtrip_identical(tmp, robot_name, k):
    out, arch, hp, state_dict = write_untrained(robot_name, k,
                                                out_path=os.path.join(tmp, f"rt{k}.pkl"))
    with open(SidecarPath(out)) as f:
        sidecar = json.load(f)
    assert sidecar["family"] == "cvae" and sidecar["provenance"]["untrained"] is True
    assert ArchFromCVAEStateDict(state_dict) == {key: arch[key] for key in (
        "ndof", "dim_latent_space", "hidden", "depth", "pose_dim")}

    chart = LoadCVAEChart(robot_name, out)
    assert chart.family == "cvae" and not chart.has_density and not chart.exact_inverse
    assert chart.network_width == k and chart.arch["activation"] == "silu"
    loaded = chart.nn_model.double()
    dev = next(loaded.parameters()).device
    original = CVAEModel(arch["ndof"], k, hp["scale"]).to(dev).double().eval()
    original.load_state_dict(state_dict)
    g = torch.Generator(device="cpu").manual_seed(3)
    pose = _poses(64, g).to(dev)
    z = torch.randn(64, k, generator=g, dtype=torch.float64, device=CPU).to(dev)
    with torch.no_grad():
        assert torch.equal(chart.decode(loaded, z, pose)[0], original(z, pose)), \
            "reloaded chart's decode is not identical"
    print(f"PASS export -> LoadCVAEChart round trip identical ({robot_name}, K={k}, on {dev})")


def _expect_raise(fn, exc, what):
    try:
        fn()
    except exc as e:
        print(f"PASS {what} raises: {str(e).splitlines()[0][:110]}")
        return
    raise AssertionError(f"{what} did not raise")


def test_contradicting_sidecars_raise(tmp):
    out = _write(tmp, "panda", 7, "bad.pkl")
    sidecar = SidecarPath(out)
    with open(sidecar) as f:
        good = json.load(f)

    for field, value in (("dim_latent_space", 1), ("depth", 4), ("hidden", 256), ("ndof", 6)):
        with open(sidecar, "w") as f:
            json.dump(dict(good, **{field: value}), f)
        _expect_raise(lambda: LoadCVAEChart("panda", out), ValueError,
                      f"a sidecar declaring {field}={value}")

    with open(sidecar, "w") as f:
        json.dump(dict(good, family="ikflow"), f)
    _expect_raise(lambda: LoadCVAEChart("panda", out), ValueError, "a non-cvae family")

    with open(sidecar, "w") as f:
        json.dump(dict(good, robot_name="iiwa14"), f)
    _expect_raise(lambda: LoadCVAEChart("panda", out), ValueError, "another robot's sidecar")

    ## The activation is the one field the weights cannot contradict: a tanh sidecar over
    ## SiLU weights LOADS. That is the hole that makes the sidecar mandatory.
    with open(sidecar, "w") as f:
        json.dump(dict(good, activation="tanh"), f)
    assert LoadCVAEChart("panda", out).arch["activation"] == "tanh"
    print("PASS activation is not shape-recoverable (documented hole; sidecar mandatory)")

    os.remove(sidecar)
    _expect_raise(lambda: LoadCVAEChart("panda", out), FileNotFoundError, "a missing sidecar")


def test_same_dof_other_robot_raises(tmp):
    """A Panda chart against the 7-DoF screw arm: shapes agree, the joint scale does not."""
    try:
        get_robot("screw7_p050")
    except Exception as e:
        print(f"SKIP same-DoF guard: screw7_p050 does not resolve ({e})")
        return
    out = _write(tmp, "panda", 7, "panda_for_screw.pkl")
    with open(SidecarPath(out)) as f:
        payload = json.load(f)
    payload.pop("robot_name")      # so only the scale check stands between them
    with open(SidecarPath(out), "w") as f:
        json.dump(payload, f)
    _expect_raise(lambda: LoadCVAEChart("screw7_p050", out), ValueError,
                  "a Panda chart loaded as screw7_p050")


def test_training_step_is_finite(robot, k):
    from train_cvae import CVAELitModel
    torch.manual_seed(0)
    with torch.device("cpu"):
        lit = CVAELitModel(robot_name=robot.name, ndof=robot.ndof, dim_latent_space=k,
                           hidden=512, depth=6, activation="silu",
                           scale=JointScale(robot).tolist(), learning_rate=1.5e-4,
                           gamma=0.9795, step_lr_every=2441, weight_decay=1.8e-5,
                           gradient_clip=1.0, log_every=1000, fk_eval_n=16,
                           run_dir=tempfile.gettempdir())
        lo, hi = torch.tensor(robot.actuated_joints_limits, dtype=torch.float32).T
        q = lo + (hi - lo) * torch.rand(256, robot.ndof)
        pose = _poses(256, torch.Generator(device="cpu").manual_seed(4)).float()
        loss = lit.training_step((q, pose), 0)
        assert torch.isfinite(loss), loss
        loss.backward()
    grads = [p.grad for p in lit.parameters()]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    _, parts = elbo_loss(lit.model, q, pose)
    assert parts["kl_per_dim"].shape == (k,)

    ## The validation step, FK screen included, outside a trainer (nothing is logged).
    lit.validation_step((q, pose.float()), 0)
    val = lit.last_val_metrics
    assert val["val/active_dims"] <= k and all(f"val/kl_dim{i}" in val for i in range(k))
    assert all(val[key] == val[key] for key in ("val/l2_error", "val/angular_error",
                                                "val/fk_pos_mm_median")), "NaN FK metric"
    print(f"PASS training_step finite with finite gradients ({robot.name}, K={k}): "
          f"loss {loss.item():.3f}, recon_mse {parts['recon_mse'].item():.3f}, "
          f"kl {parts['kl'].item():.3f}")


def test_batched_dataloaders(robot_name="iiwa14"):
    """The trainer's dataset is the fork's, with batched indexing: same files, same shapes.

    Needs a local dataset; the small iiwa14 one exists on the laptop (Panda's is cluster-only).
    """
    from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING
    from ikflow.utils import get_dataset_directory
    from train_cvae import CVAEDataset
    if not os.path.isdir(get_dataset_directory(robot_name)):
        print(f"SKIP batched dataloaders: no local {robot_name} dataset")
        return
    data = CVAEDataset(robot_name, 4096, 1000, [DATASET_TAG_NON_SELF_COLLIDING])
    q, pose = next(iter(data.train_dataloader()))
    assert q.shape == (4096, data._samples_tr.shape[1]) and pose.shape == (4096, 7)
    assert q.device.type == "cpu"
    ## Each batch row must still be a (sample, its own pose) pair after batched indexing.
    rows = (data._samples_tr[:, None, :] == q[None, :8, :]).all(dim=2).nonzero()
    assert torch.equal(data._endpoints_tr[rows[:, 0]], pose[rows[:, 1]])
    q_val, pose_val = next(iter(data.val_dataloader()))
    assert q_val.shape[0] == 1000 and torch.equal(q_val, data._samples_te[:1000])
    print(f"PASS batched dataloaders ({robot_name}): train batch {tuple(q.shape)}, "
          f"val batch {tuple(q_val.shape)}, pairs intact")


if __name__ == "__main__":
    t0 = time.monotonic()
    panda = get_robot("panda")
    counts = {}
    for k in (7, 1):
        counts[("panda", k)] = test_shapes_float64_and_encode_is_mu(panda, k)
        test_jacrev_matches_central_differences(panda, k)
        test_training_step_is_finite(panda, k)

    ## soft12 only if it resolves cheaply: its construction is the robot's, not the chart's.
    t = time.monotonic()
    try:
        soft12 = get_robot("soft12")
    except Exception as e:
        soft12 = None
        print(f"SKIP soft12: does not resolve ({e})")
    if soft12 is not None:
        print(f"soft12 resolved in {time.monotonic() - t:.1f} s")
        for k in (12, 6):
            counts[("soft12", k)] = test_shapes_float64_and_encode_is_mu(soft12, k)
            test_jacrev_matches_central_differences(soft12, k)

    with tempfile.TemporaryDirectory() as tmp:
        for k in (7, 1):
            test_export_roundtrip_identical(tmp, "panda", k)
        test_contradicting_sidecars_raise(tmp)
        test_same_dof_other_robot_raises(tmp)
    test_batched_dataloaders()
    print("parameter counts:", {f"{r} K={k}": n for (r, k), n in counts.items()})
    print(f"ALL PASS ({time.monotonic() - t0:.1f} s)")
