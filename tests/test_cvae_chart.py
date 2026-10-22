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
4. The two decoder-variance variants (`decoder_variance`): "fixed" has NO `log_sigma_q`, so
   the field is recoverable from the state dict's keys and a sidecar contradicting it RAISES;
   a sidecar written before the field existed loads as "learned" (the committed untrained
   Panda charts included); the fixed loss is exactly sum-of-squares + KL; the learned loss is
   bit-identical to the first rung's formula.
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


def _write(tmp, robot_name, k, name, decoder_variance="learned"):
    return write_untrained(robot_name, k, out_path=os.path.join(tmp, name),
                           decoder_variance=decoder_variance)[0]


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
        "ndof", "dim_latent_space", "hidden", "depth", "pose_dim", "decoder_variance")}

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


def test_training_step_is_finite(robot, k, decoder_variance="learned"):
    from train_cvae import CVAELitModel
    torch.manual_seed(0)
    with torch.device("cpu"):
        lit = CVAELitModel(robot_name=robot.name, ndof=robot.ndof, dim_latent_space=k,
                           hidden=512, depth=6, activation="silu",
                           scale=JointScale(robot).tolist(), learning_rate=1.5e-4,
                           gamma=0.9795, step_lr_every=2441, weight_decay=1.8e-5,
                           gradient_clip=1.0, log_every=1000, fk_eval_n=16,
                           run_dir=tempfile.gettempdir(), decoder_variance=decoder_variance)
        assert lit.model.decoder_variance == decoder_variance
        assert hasattr(lit.model, "log_sigma_q") == (decoder_variance == "learned")
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
    print(f"PASS training_step finite with finite gradients ({robot.name}, K={k}, "
          f"{decoder_variance} variance): "
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
    ## The fork's IkfLitDataset `torch.load`s with no `map_location`, and the laptop's iiwa14
    ## dataset was saved from CUDA tensors, so it cannot load in a CPU-only run (local runs
    ## are CPU-only since the 2026-10-18 crash). The fork is not edited for a test.
    if not torch.cuda.is_available():
        print(f"SKIP batched dataloaders: the {robot_name} dataset holds CUDA tensors and "
              "the fork loads without map_location; CPU-only run")
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


def _reference_learned_loss(model, q, pose7):
    """The first rung's elbo_loss, verbatim (commit e4bb656): the learned path must not move."""
    x = q / model.scale
    mu, logvar = model.encode_dist(q, pose7)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    x_hat = model.decode_normalised(z, pose7)
    log_sigma = model.log_sigma_q
    sq = (x - x_hat) ** 2
    recon_nll = 0.5 * (sq * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma
                       + torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
                       ).sum(dim=-1)
    kl = (0.5 * (mu ** 2 + torch.exp(logvar) - 1.0 - logvar)).sum(dim=-1)
    return (recon_nll + kl).mean()


def test_decoder_variance_losses(robot, k):
    g = torch.Generator(device="cpu").manual_seed(5)
    lo, hi = torch.tensor(robot.actuated_joints_limits, dtype=torch.float64, device=CPU).T
    q = lo + (hi - lo) * torch.rand(128, robot.ndof, generator=g, dtype=torch.float64, device=CPU)
    pose = _poses(128, g)

    learned = _model(robot, k)
    with torch.no_grad():
        learned.log_sigma_q.copy_(torch.linspace(-0.7, 0.4, robot.ndof, dtype=torch.float64))
    torch.manual_seed(11)
    got, _ = elbo_loss(learned, q, pose)
    torch.manual_seed(11)
    ref = _reference_learned_loss(learned, q, pose)
    assert torch.equal(got, ref), (got, ref)

    torch.manual_seed(0)
    with torch.device("cpu"):
        fixed = CVAEModel(robot.ndof, k, JointScale(robot), decoder_variance="fixed").double()
    assert "log_sigma_q" not in fixed.state_dict() and not hasattr(fixed, "log_sigma_q")
    assert {n for n in learned.state_dict()} - {n for n in fixed.state_dict()} == {"log_sigma_q"}
    torch.manual_seed(12)
    loss, parts = elbo_loss(fixed, q, pose)
    torch.manual_seed(12)
    mu, logvar = fixed.encode_dist(q, pose)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    sq = ((q / fixed.scale - fixed.decode_normalised(z, pose)) ** 2).sum(dim=-1)
    kl = (0.5 * (mu ** 2 + logvar.exp() - 1.0 - logvar)).sum(dim=-1)
    assert torch.isfinite(loss) and torch.allclose(loss, (sq + kl).mean(), rtol=0, atol=1e-12)
    assert torch.allclose(parts["recon_nll"], sq.mean(), rtol=0, atol=1e-12)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fixed.parameters())
    print(f"PASS decoder-variance losses ({robot.name}, K={k}): learned bit-identical to the first "
          f"rung's formula; fixed = sum of squares + KL ({loss.item():.3f}), finite gradients")


def test_fixed_variant_sidecars(tmp):
    from src.charts.cvae import ReadCVAEArch
    out = _write(tmp, "panda", 7, "fixed.pkl", decoder_variance="fixed")
    with open(out, "rb") as f:
        state_dict = pickle.load(f)
    assert "log_sigma_q" not in state_dict
    assert ArchFromCVAEStateDict(state_dict)["decoder_variance"] == "fixed"
    sidecar = SidecarPath(out)
    with open(sidecar) as f:
        good = json.load(f)
    assert good["decoder_variance"] == "fixed"
    chart = LoadCVAEChart("panda", out)
    assert chart.arch["decoder_variance"] == "fixed"
    assert chart.nn_model.decoder_variance == "fixed"

    ## A fixed chart's sidecar claiming "learned", or carrying no field (which reads as
    ## learned): the weights have no log_sigma_q, so both RAISE.
    with open(sidecar, "w") as f:
        json.dump(dict(good, decoder_variance="learned"), f)
    _expect_raise(lambda: LoadCVAEChart("panda", out), ValueError,
                  "a learned-variance sidecar over fixed-variance weights")
    with open(sidecar, "w") as f:
        json.dump({k: v for k, v in good.items() if k != "decoder_variance"}, f)
    _expect_raise(lambda: LoadCVAEChart("panda", out), ValueError,
                  "a field-less (old, = learned) sidecar over fixed-variance weights")

    ## A learned chart's sidecar claiming "fixed": the weights carry log_sigma_q, so it RAISES;
    ## the same sidecar with no field at all (written before it existed) loads as learned.
    out_l = _write(tmp, "panda", 7, "learned.pkl")
    side_l = SidecarPath(out_l)
    with open(side_l) as f:
        good_l = json.load(f)
    assert good_l["decoder_variance"] == "learned"
    with open(side_l, "w") as f:
        json.dump(dict(good_l, decoder_variance="fixed"), f)
    _expect_raise(lambda: LoadCVAEChart("panda", out_l), ValueError,
                  "a fixed-variance sidecar over learned-variance weights")
    with open(side_l, "w") as f:
        json.dump({k: v for k, v in good_l.items() if k != "decoder_variance"}, f)
    assert ReadCVAEArch(out_l)[0]["decoder_variance"] == "learned"
    assert LoadCVAEChart("panda", out_l).arch["decoder_variance"] == "learned"
    print("PASS a sidecar without decoder_variance loads as learned")


def test_committed_untrained_panda_charts_load():
    """The untrained learned-variance Panda charts in models/panda, written before the field
    existed, still load (as learned) and decode finitely."""
    for k in (7, 1):
        path = os.path.join(REPO, "models", "panda", f"panda__cvae{k}untrained__step0.pkl")
        if not os.path.exists(path):
            print(f"SKIP {os.path.basename(path)}: not present (gitignored, local only)")
            continue
        with open(SidecarPath(path)) as f:
            assert "decoder_variance" not in json.load(f), "expected a pre-field sidecar"
        chart = LoadCVAEChart("panda", path)
        assert chart.arch["decoder_variance"] == "learned" and hasattr(chart.nn_model, "log_sigma_q")
        model = chart.nn_model.double()
        dev = next(model.parameters()).device
        g = torch.Generator(device="cpu").manual_seed(6)
        z = torch.randn(8, k, generator=g, dtype=torch.float64, device=CPU).to(dev)
        with torch.no_grad():
            q, _ = chart.decode(model, z, _poses(8, g).to(dev))
        assert torch.isfinite(q).all()
        print(f"PASS committed {os.path.basename(path)} loads as learned and decodes")


if __name__ == "__main__":
    t0 = time.monotonic()
    panda = get_robot("panda")
    counts = {}
    for k in (7, 1):
        counts[("panda", k)] = test_shapes_float64_and_encode_is_mu(panda, k)
        test_jacrev_matches_central_differences(panda, k)
        test_training_step_is_finite(panda, k)
        test_training_step_is_finite(panda, k, decoder_variance="fixed")
        test_decoder_variance_losses(panda, k)

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
        test_fixed_variant_sidecars(tmp)
    test_committed_untrained_panda_charts_load()
    test_batched_dataloaders()
    print("parameter counts:", {f"{r} K={k}": n for (r, k), n in counts.items()})
    print(f"ALL PASS ({time.monotonic() - t0:.1f} s)")
