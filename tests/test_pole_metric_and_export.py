"""Guards for the retraining campaign's measurement and export machinery.

There is no test runner in this repo; run this by hand:

    .venv/bin/python tests/test_pole_metric_and_export.py

Three things are protected, each of which failed silently at least once in design:

1. rpy_to_wxyz must match pydrake's RollPitchYaw -> Quaternion EXACTLY, sign included.
   Drake canonicalizes to w >= 0, and q vs -q are different network conditioning
   vectors -- the non-canonical half of quaternion space nearly doubles the measured
   pole fraction (measured: 6.35% vs 3.53% on lemon-haze-7). A sign-agnostic test
   passes while the metric reads ~2x high.

2. The pole sampler is frozen. The recorded baselines (iiwa14 lemon-haze-7: 3.34%,
   N=20000, seed 0) were measured with this exact RNG stream -- position, rpy, latent
   direction, latent radius, per sample, from default_rng(seed). Any edit that changes
   the stream silently invalidates every baseline comparison, so the first draws are
   pinned here by value.

3. The .ckpt -> .pkl export must produce exactly what IKFlowSolver.load_state_dict
   expects, and must refuse checkpoints whose hyperparameters differ from what
   src/iiwa_program.py hardcodes (the loader keeps upstream defaults for any field
   not in its dict, so a mismatched checkpoint loads cleanly and is silently wrong).
"""

import os
import pickle
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(REPO)

from ikflow.training.pole_callback import (  # noqa: E402
    rpy_to_wxyz,
    sample_conditioning_and_latents,
    pole_metrics,
)


def test_rpy_to_wxyz_matches_pydrake_exactly():
    from pydrake.math import RotationMatrix, RollPitchYaw

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(2000):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        ours = rpy_to_wxyz(rpy)
        drake = RotationMatrix(RollPitchYaw(rpy)).ToQuaternion().wxyz()
        worst = max(worst, float(np.abs(ours - drake).max()))  # STRICT: sign included
        assert ours[0] >= 0.0, f"non-canonical quaternion (w={ours[0]}) for rpy={rpy}"
    assert worst < 1e-12, f"rpy_to_wxyz drifted from pydrake: max err {worst:.3e}"
    print(f"PASS rpy_to_wxyz vs pydrake (strict), max err {worst:.2e}")


def test_sampler_stream_frozen():
    c, z = sample_conditioning_and_latents(2, 8, seed=0)
    # First two samples of the seed-0 stream, pinned by value. If this fails, the
    # sampler's RNG stream changed and the recorded baselines no longer apply.
    expected_c0 = np.array(
        [0.4684808437, -0.1151066431, 0.2704867620, 0.7928684496, 0.1912372612, 0.5204767835, -0.2527683711]
    )
    assert np.allclose(c[0, :7], expected_c0, atol=1e-9), f"sampler stream changed: c[0]={c[0, :7]}"
    assert c[0, 7] == 0.0 and c[1, 7] == 0.0, "softflow column must be zero"
    norms = np.linalg.norm(z, axis=1)
    assert np.all(norms <= 4.3 + 1e-12), "latent left the trust ball"
    assert abs(norms[0] - 3.1375184196) < 1e-9, f"latent stream changed: |z0|={norms[0]}"
    print("PASS sampler stream frozen (seed 0 first draws match pinned values)")


def test_export_roundtrip_and_hparam_guard():
    sys.path.append(os.path.join(REPO, "scripts", "training"))
    from export_ckpt_to_pkl import export, LOADER_ASSUMPTIONS

    from ikflow.model import IkflowModelParameters
    from ikflow.ikflow_solver import IKFlowSolver
    from jrl.robots import get_robot

    hyper = IkflowModelParameters()
    hyper.__dict__.update(
        {"nb_nodes": 12, "dim_latent_space": 8, "coeff_fn_config": 3,
         "coeff_fn_internal_size": 1024, "rnvp_clamp": 2.5, "robot_name": "iiwa14",
         # train_ddp.py sets this explicitly (the class default is 0.01, the training
         # default 0.001 -- the export guard rightly rejects the class default).
         "softflow_noise_scale": 0.001}
    )
    robot = get_robot("iiwa14")
    solver = IKFlowSolver(hyper, robot, compile_model=None)

    with tempfile.TemporaryDirectory() as tmp:
        # Fake a Lightning checkpoint from the randomly initialized model.
        ckpt_path = os.path.join(tmp, "fake.ckpt")
        state = {"nn_model." + k: v for k, v in solver.nn_model.state_dict().items()}
        torch.save(
            {"state_dict": state, "global_step": 123, "hyper_parameters": {"base_hparams": hyper}},
            ckpt_path,
        )
        pkl_path = os.path.join(tmp, "out.pkl")
        export(ckpt_path, pkl_path)

        # Round trip through the loader's own path.
        solver2 = IKFlowSolver(hyper, robot, compile_model=None)
        solver2.load_state_dict(pkl_path)
        with open(pkl_path, "rb") as f:
            sd = pickle.load(f)
        assert all(not k.startswith("nn_model.") for k in sd), "prefix not stripped"
        ref = {k: v for k, v in solver.nn_model.state_dict().items()}
        assert all(torch.equal(sd[k], ref[k].cpu()) for k in ref), "tensors changed in export"

        # The guard must fire on a mismatched hyperparameter.
        bad = IkflowModelParameters()
        bad.__dict__.update(hyper.__dict__)
        bad.nb_nodes = 6
        bad_ckpt = os.path.join(tmp, "bad.ckpt")
        torch.save(
            {"state_dict": state, "global_step": 1, "hyper_parameters": {"base_hparams": bad}}, bad_ckpt
        )
        try:
            export(bad_ckpt, os.path.join(tmp, "bad.pkl"))
        except AssertionError as e:
            assert "nb_nodes" in str(e)
        else:
            raise AssertionError("hparam guard did not fire on nb_nodes=6")
    assert set(LOADER_ASSUMPTIONS) >= {"nb_nodes", "dim_latent_space", "rnvp_clamp", "softflow_enabled"}
    print("PASS export roundtrip + hparam guard")


def test_pole_metrics_shape():
    # Tiny run on a random model: just that the machinery runs and returns sane keys.
    from ikflow.model import IkflowModelParameters
    from ikflow.ikflow_solver import IKFlowSolver
    from jrl.robots import get_robot

    hyper = IkflowModelParameters()
    hyper.__dict__.update(
        {"nb_nodes": 2, "dim_latent_space": 8, "coeff_fn_config": 1,
         "coeff_fn_internal_size": 32, "rnvp_clamp": 2.5, "robot_name": "iiwa14"}
    )
    solver = IKFlowSolver(hyper, get_robot("iiwa14"), compile_model=None)
    m = pole_metrics(solver.nn_model, width=8, ndof=7, n=64, chunk=32)
    assert 0.0 <= m["pole/frac_gt_1000"] <= 1.0 and m["pole/n"] == 64.0
    print("PASS pole_metrics machinery")


if __name__ == "__main__":
    test_rpy_to_wxyz_matches_pydrake_exactly()
    test_sampler_stream_frozen()
    test_export_roundtrip_and_hparam_guard()
    test_pole_metrics_shape()
    print("ALL PASS")
