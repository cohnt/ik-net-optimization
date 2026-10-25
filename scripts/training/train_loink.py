"""Train a LOInK chart (`src/charts/loink.py`) on one GPU.

The chart-loink campaign's ladder is 3 costs x 2 activations x 2 recipes per robot, one
architecture (`DEFAULT_ARCH`: depth 2, 4 x 256 FTN blocks, mu = 0.1, head 2 x 128):

  * `--cost m` fits the head to the program's own joint-centering cost `0.5 |q - q_nom|^2`;
    `x` to LOInK's eq. 18 joint-limit barrier; `u` has no head (lambda = 0). The target is
    standardised by its mean / std over the first 1M training samples (`j_mean`, `j_std`, in
    the hyperparameters and the sidecar), so `z = O(1)`.
  * `--activation relu|tanh`.
  * `--recipe P`, the project recipe, the flow's and the cVAE's (`train_cvae.py`): AdamW
    1.5e-4, weight decay 1.8e-5, StepLR x0.9795 every 2441 steps, gradient clipping by VALUE
    at 1, batch 4096, 620000 steps.
  * `--recipe A`, the paper's (LOInK Sec. V.B): Adam 1e-3, no weight decay, no clipping,
    batch 500, 150000 steps (3 epochs of the 25M datasets), ReduceLROnPlateau(factor 0.5,
    patience 500 steps). The paper says "reduce-on-plateau with patience 500"; the factor and
    the signal are ours, pre-registered: the signal is an EMA (0.99, a ~100-step horizon) of
    the training loss, handed to the scheduler EVERY step -- the same signal the replication
    trainer uses (`scripts/loink_replication/train_replication.py`), because a raw minibatch
    loss is noise-dominated and a validation pass every step is not affordable. The EMA is a
    buffer of this module, so a resumed run continues it.

Any recipe default (`--batch_size`, `--max_steps`, the optimiser's numbers) is overridden by
an explicit flag, wherever it appears on the command line (the cluster appends
`--batch_size` / `--max_steps` after a row's own arguments; argparse keeps the last).

One guard the parametrisation needs and the paper does not mention: a monotone layer is
(mu, nu) only while nu = exp(lognu) > mu (sqrt(2 gam) is NaN otherwise), and the distortion
penalty pulls nu DOWN toward the fixed mu. After every optimiser step each layer's `lognu` is
clamped to >= log(mu (1 + 1e-3)) -- as the replication trainer does -- and the number of
steps at which the clamp was active is logged (`tr/nu_clamped_steps`).

Cluster conventions are the cVAE trainer's: `--ckpt_path auto` resumes from
`<run_dir>/checkpoints/last.ckpt` (every checkpoint kept, `loink-checkpoint-step=<N>.ckpt`);
CSV metrics under `<run_dir>/metrics`, wandb OFFLINE; `<run_dir>/status.json` with
`"family": "loink"`; a THROUGHPUT line at the end. Validation every `--eval_every` steps on the
test split: task position / rotation RMSE of `y_hat = G_1(q)`, aux MSE, `nu`, `tau`, the
inversion residual `|G(decode(w)) - w|` at the model's `n_dys` (DYS only and with the Newton
polish) and the FK error of `decode(y, 0)` on `--fk_eval_n` samples.

Single-branch charts (LOInK phase 2, Thomas 2026-10-24): `--dataset_name <robot>_branch
--branch nominal` trains on a dataset rejection-sampled onto the nominal configuration's IK
branch (`src/branches.py`, `scripts/training/build_branch_dataset.py`), because one bijection
carries one connected solution arc per pose and the rigid arms' full solution sets are several.
Both land in the hyperparameters (`dataset_name`, `branch` as the signs or None) and from there
in the sidecar, record only. With `nominal` the first `J_STATS_SAMPLES` training rows are
checked to be on the branch, a guard against pairing a run with the wrong dataset.

IMPORT ORDER IS LOAD-BEARING, as in train_cvae.py: ikflow resolves DATASET_DIR from HOME at
import, and jrl sets torch's default device to CUDA at import (reset to CPU straight after).

    python scripts/training/train_loink.py --robot_name panda --run_dir <dir> \
        --cost m --activation relu --recipe P --ckpt_path auto
    python scripts/training/train_loink.py --robot_name panda --run_dir <dir> \
        --cost m --activation relu --recipe P --dataset_name panda_branch --branch nominal
"""

import argparse
import json
import os
import sys
import time

import torch
import jrl.config  # noqa: F401  (side effect: sets torch's default device)

torch.set_default_device("cpu")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "training"))

import src.register_robots  # noqa: E402,F401  -- before anything resolves a robot

import numpy as np  # noqa: E402
from jrl.math_utils import geodesic_distance_between_quaternions  # noqa: E402
from pytorch_lightning import LightningModule, seed_everything  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger, WandbLogger  # noqa: E402
from pytorch_lightning.plugins.environments import LightningEnvironment  # noqa: E402
from pytorch_lightning.trainer import Trainer  # noqa: E402

from ikflow import config as ikflow_config  # noqa: E402
from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING  # noqa: E402

from src.branches import BranchString, NominalBranch, OnBranch  # noqa: E402
from src.charts.cvae import JointScale  # noqa: E402
from src.charts.loink import (  # noqa: E402
    ACTIVATIONS, COSTS, DEFAULT_ARCH, FAMILY, TASK_DIM, LoinkModel, Pose7ToXyzRpy,
    TrainingCost, XyzRpyToPose7, dys_inverse, loink_loss)
from train_cvae import CVAEDataset, ResolveRobot  # noqa: E402

RECIPES = ("P", "A")

#: Per-recipe defaults; an explicit flag always wins (`ResolveRecipe`). Recipe A clips gradients at
#: 1.0 although the paper names no clipping: the replication (docs/loink-replication.md, Sec. VII)
#: hit a loss spike at ~step 5800 without it, after which the plateau scheduler drove the learning
#: rate to 2e-7. The paper's own recipe is otherwise unchanged.
RECIPE_DEFAULTS = {
    "P": {"optimizer": "adamw", "learning_rate": 1.5e-4, "weight_decay": 1.8e-5,
          "scheduler": "step", "gamma": 0.9795, "step_lr_every": 2441, "gradient_clip_val": 1.0,
          "batch_size": 4096, "max_steps": 620000, "eval_every": 20000,
          "checkpoint_every": 20000},
    "A": {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 0.0,
          "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 500,
          "plateau_ema": 0.99, "gradient_clip_val": 1.0, "batch_size": 500,
          "max_steps": 150000, "eval_every": 10000, "checkpoint_every": 10000},
}

#: Samples the cost standardisation is computed on (the first, in dataset order).
J_STATS_SAMPLES = 1_000_000

#: lognu >= log(mu (1 + NU_GUARD_REL)) after every step (module docstring).
NU_GUARD_REL = 1e-3


def CostStats(robot_name, cost, samples, robot):
    """`(j_mean, j_std)` of the training cost on the first `J_STATS_SAMPLES` rows, float64 on
    the CPU; `(None, None)` for cost "u"."""
    fn = TrainingCost(robot_name, cost, robot=robot)
    if fn is None:
        return None, None
    q = samples[:J_STATS_SAMPLES].to(device="cpu", dtype=torch.float64)
    j = torch.cat([fn(q[i:i + 100000]) for i in range(0, q.shape[0], 100000)])
    if not torch.isfinite(j).all():
        raise ValueError(f"non-finite training cost {cost!r} on the dataset")
    return float(j.mean()), float(j.std())


def CheckOnBranch(robot_name, samples, branch, dataset_name):
    """Raise unless the first `J_STATS_SAMPLES` training rows all lie on `branch` (2026-10-24:
    a single-branch run paired with an unrestricted dataset would train silently on every
    branch, which is the phase-1 chart again)."""
    q = samples[:J_STATS_SAMPLES].to(device="cpu", dtype=torch.float64).numpy()
    on = OnBranch(robot_name, q, branch)
    if not on.all():
        raise ValueError(
            f"--branch {BranchString(branch)}: {int((~on).sum())} of the first {q.shape[0]} "
            f"training rows of dataset {dataset_name!r} are off the branch -- is --dataset_name "
            f"the branch dataset (src.branches.BRANCH_DATASET)?")
    return q.shape[0]


class LoinkLitModel(LightningModule):
    def __init__(self, robot_name, ndof, units, depth, mu, activation, cost, head_hidden,
                 n_dys, n_newton, scale, j_mean, j_std, lam, tau_penalty, recipe, optimizer,
                 learning_rate, weight_decay, scheduler, gamma, step_lr_every, plateau_factor,
                 plateau_patience, plateau_ema, gradient_clip, log_every, fk_eval_n,
                 inv_eval_n, run_dir, dataset_name=None, branch=None):
        super().__init__()
        ## Everything the exporter needs is a constructor argument, so `save_hyperparameters`
        ## puts it in every checkpoint's `hyper_parameters`.
        self.save_hyperparameters()
        with torch.device("cpu"):
            self.model = LoinkModel(ndof, TASK_DIM, scale, units=tuple(units), depth=depth,
                                    mu=mu, activation=activation, cost=cost,
                                    head_hidden=head_hidden, n_dys=n_dys, n_newton=n_newton)
        ## Training-loop state that must survive a resume, as buffers (the exporter keeps
        ## only `model.*`): the plateau signal and the nu-guard counter.
        self.register_buffer("loss_ema", torch.tensor(float("nan")))
        self.register_buffer("nu_clamped_steps", torch.tensor(0, dtype=torch.long))
        self._cost_fn = None
        self._robot = None
        self._last_tr_loss = None

    @property
    def robot(self):
        if self._robot is None:
            self._robot = ResolveRobot(self.hparams.robot_name)
        return self._robot

    @property
    def cost_fn(self):
        if self._cost_fn is None:
            self._cost_fn = TrainingCost(self.hparams.robot_name, self.hparams.cost, robot=self.robot)
        return self._cost_fn

    def on_fit_start(self):
        _ = self.robot

    def configure_optimizers(self):
        hp = self.hparams
        params = self.model.parameters()
        if hp.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params, lr=hp.learning_rate, weight_decay=hp.weight_decay)
        else:
            optimizer = torch.optim.Adam(params, lr=hp.learning_rate, weight_decay=hp.weight_decay)
        if hp.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.step_lr_every,
                                                        gamma=hp.gamma)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=hp.plateau_factor, patience=hp.plateau_patience)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1,
                              "monitor": "tr_loss_ema", "strict": True}]

    def get_lr(self):
        return self.optimizers().optimizer.param_groups[0]["lr"]

    def safe_log_metrics(self, values):
        for logger in self.loggers:
            logger.log_metrics(values, step=self.global_step)

    def _targets(self, q, pose):
        y = Pose7ToXyzRpy(pose.to(q.dtype))
        if self.cost_fn is None:
            return y, None
        j = (self.cost_fn(q) - self.hparams.j_mean) / self.hparams.j_std
        return y, j.to(q.dtype)

    def training_step(self, batch, batch_idx):
        q, pose = batch
        y, j = self._targets(q, pose)
        loss, parts = loink_loss(self.model, q, y, j, lam=self.hparams.lam,
                                 tau_penalty=self.hparams.tau_penalty)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {self.global_step}: {loss.item()} "
                             f"({ {k: float(v) for k, v in parts.items()} })")
        ## The plateau signal, on the device (no sync here; the scheduler reads it).
        lv = loss.detach()
        beta = self.hparams.plateau_ema
        self.loss_ema.copy_(torch.where(torch.isnan(self.loss_ema), lv,
                                        beta * self.loss_ema + (1 - beta) * lv))
        if self.hparams.scheduler == "plateau":
            self.log("tr_loss_ema", self.loss_ema, on_step=True, on_epoch=False, logger=False,
                     prog_bar=False)
        if self.global_step % self.hparams.log_every == 0 and self.global_step > 0:
            self._last_tr_loss = float(loss.item())
            metrics = {"tr/loss": self._last_tr_loss, "tr/loss_ema": float(self.loss_ema),
                       "tr/task_mse": float(parts["task_mse"]), "tr/dist": float(parts["dist"]),
                       "tr/nu": float(parts["nu"]), "tr/tau": float(parts["tau"]),
                       "tr/learning_rate": self.get_lr(),
                       "tr/nu_clamped_steps": int(self.nu_clamped_steps)}
            if "aux_mse" in parts:
                metrics["tr/aux_mse"] = float(parts["aux_mse"])
            self.safe_log_metrics(metrics)
        return loss

    def on_after_backward(self):
        if self.hparams.gradient_clip:
            ## By VALUE, as the flow recipe does (IkfLitModel.on_after_backward).
            torch.nn.utils.clip_grad_value_(self.parameters(), self.hparams.gradient_clip)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        ## The nu > mu guard (module docstring), after the optimiser step; no host sync.
        with torch.no_grad():
            hit = torch.zeros((), dtype=torch.bool, device=self.loss_ema.device)
            for mon in self.model.G.mon:
                lo = torch.log(mon.mu * (1 + NU_GUARD_REL)).to(mon.lognu.dtype)
                hit = hit | (mon.lognu < lo).any()
                mon.lognu.copy_(torch.maximum(mon.lognu, lo))
            self.nu_clamped_steps.add_(hit.long())

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        q, pose = batch
        y, j = self._targets(q, pose)
        loss, parts = loink_loss(self.model, q, y, j, lam=self.hparams.lam,
                                 tau_penalty=self.hparams.tau_penalty)
        y_hat, _ = self.model.forward_G(q)
        pos = torch.linalg.norm(y_hat[:, :3] - pose[:, :3].to(y_hat.dtype), dim=1)
        ang = geodesic_distance_between_quaternions(
            XyzRpyToPose7(y_hat.double().cpu())[:, 3:7], pose[:, 3:7].double().cpu())
        metrics = {
            "val/loss": float(loss),
            "val/task_mse": float(parts["task_mse"]),
            "val/pos_rmse_mm": float(1e3 * torch.sqrt((pos.double() ** 2).mean())),
            "val/rot_rmse_deg": float(np.degrees(torch.sqrt((ang ** 2).mean()).item())),
            "val/nu": float(parts["nu"]),
            "val/tau": float(parts["tau"]),
        }
        if "aux_mse" in parts:
            metrics["val/aux_mse"] = float(parts["aux_mse"])
        metrics.update(self._decode_screen(q, pose, y))
        self.last_val_metrics = metrics

    def _decode_screen(self, q, pose, y):
        """`decode(y, 0)` on a small subset: the inversion residual `|G(x) - w|_inf` at the
        model's own `n_dys` (DYS alone, and the full decode with the Newton polish), and the
        FK error of the decoded configuration v the dataset pose (as train_cvae's `_fk_error`;
        the soft PCS arm's FK is float64 on the CPU, hence the subset)."""
        n_inv, n_fk = self.hparams.inv_eval_n, self.hparams.fk_eval_n
        m = self.model
        w = torch.cat([y[:n_inv], torch.zeros((min(n_inv, y.shape[0]), m.K), dtype=y.dtype,
                                              device=y.device)], dim=1)
        x_dys = dys_inverse(m.G, w, m.n_dys)
        q_dec = m.decode(w[:, :m.p], w[:, m.p:])
        res_dys = (m.G(x_dys) - w).abs().amax(dim=1)
        res_dec = (m.G(q_dec / m.scale) - w).abs().amax(dim=1)
        out = {"val/inv_residual_dys_max": float(res_dys.max()),
               "val/inv_residual_dys_median": float(res_dys.median()),
               "val/inv_residual_max": float(res_dec.max()),
               "val/inv_residual_median": float(res_dec.median())}

        qf = q_dec[:n_fk].double().cpu()
        realised = self.robot.forward_kinematics(qf, out_device="cpu").double()
        target = pose[:n_fk].double().cpu()
        p_err = torch.linalg.norm(realised[:, :3] - target[:, :3], dim=1).numpy()
        a_err = geodesic_distance_between_quaternions(realised[:, 3:7], target[:, 3:7]).numpy()
        out.update({"val/l2_error": float(p_err.mean()), "val/angular_error": float(a_err.mean()),
                    "val/fk_pos_mm_median": float(1e3 * np.median(p_err)),
                    "val/fk_pos_mm_p95": float(1e3 * np.percentile(p_err, 95)),
                    "val/fk_ang_deg_median": float(np.degrees(np.median(a_err))),
                    "val/fk_ang_deg_p95": float(np.degrees(np.percentile(a_err, 95))),
                    "val/q_absmax": float(qf.abs().max())})
        return out

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = getattr(self, "last_val_metrics", None)
        if metrics is None:
            return
        self.safe_log_metrics(metrics)
        self._write_status(metrics)

    def _write_status(self, val):
        """The PoleFractionCallback's heartbeat, key for key, as train_cvae.py writes it."""
        batch = self.trainer.datamodule._batch_size
        try:
            lr = self.get_lr()
        except Exception:
            lr = None
        status = {
            "global_step": self.trainer.global_step,
            "world_size": self.trainer.world_size,
            "batch_size_per_rank": batch,
            "samples_per_step": self.trainer.world_size * batch,
            "samples_seen": self.trainer.global_step * self.trainer.world_size * batch,
            "learning_rate": lr,
            "val_l2_error": val.get("val/l2_error"),
            "val_angular_error": val.get("val/angular_error"),
            "tr_loss": self._last_tr_loss,
            "pole": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            ## LOInK-only extras; the status reader ignores keys it does not know.
            "family": FAMILY,
            "val_pos_rmse_mm": val.get("val/pos_rmse_mm"),
            "val_rot_rmse_deg": val.get("val/rot_rmse_deg"),
            "val_aux_mse": val.get("val/aux_mse"),
            "val_tau": val.get("val/tau"),
            "val_inv_residual_max": val.get("val/inv_residual_max"),
            "nu_clamped_steps": int(self.nu_clamped_steps),
        }
        run_dir = self.hparams.run_dir
        os.makedirs(run_dir, exist_ok=True)
        tmp = os.path.join(run_dir, ".status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(status, f, indent=1)
        os.replace(tmp, os.path.join(run_dir, "status.json"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot_name", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--cost", required=True, choices=COSTS)
    p.add_argument("--activation", required=True, choices=ACTIVATIONS)
    p.add_argument("--recipe", required=True, choices=RECIPES)
    ## The architecture (one, no ladder) and the loss weights (pre-registered).
    p.add_argument("--units", type=int, nargs="+", default=list(DEFAULT_ARCH["units"]))
    p.add_argument("--depth", type=int, default=DEFAULT_ARCH["depth"])
    p.add_argument("--mu", type=float, default=DEFAULT_ARCH["mu"])
    p.add_argument("--head_hidden", type=int, default=DEFAULT_ARCH["head_hidden"])
    p.add_argument("--n_dys", type=int, default=DEFAULT_ARCH["n_dys"])
    p.add_argument("--n_newton", type=int, default=DEFAULT_ARCH["n_newton"])
    p.add_argument("--lam", type=float, default=0.01,
                   help="aux (cost) loss weight, eq. 16; the paper gives no value. 0.01 is what "
                        "reproduced Fig. 3 (docs/loink-replication.md): lambda = 1 underfits the "
                        "task (0.207 m v 0.021 m validation RMSE on the planar 3-DoF example)")
    p.add_argument("--tau_penalty", type=float, default=1e-3,
                   help="weight of the distortion penalty tau_penalty * log(tau)")
    ## Recipe-dependent: None means "the recipe's default" (RECIPE_DEFAULTS).
    for name, typ in (("batch_size", int), ("max_steps", int), ("learning_rate", float),
                      ("weight_decay", float), ("gamma", float), ("step_lr_every", int),
                      ("plateau_factor", float), ("plateau_patience", int),
                      ("plateau_ema", float), ("gradient_clip_val", float),
                      ("eval_every", int), ("checkpoint_every", int)):
        p.add_argument(f"--{name}", type=typ, default=None)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--val_set_size", type=int, default=20000)
    p.add_argument("--fk_eval_n", type=int, default=500)
    p.add_argument("--inv_eval_n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_name", default=None,
                   help="ikflow dataset directory name (default: --robot_name); the single-branch "
                        "charts train on <robot>_branch (2026-10-24)")
    p.add_argument("--branch", default="none", choices=("none", "nominal"),
                   help="'nominal': the dataset must be the nominal configuration's IK branch "
                        "(src/branches.py), checked on the first 1M rows and recorded in the "
                        "hyperparameters; 'none': unrestricted")
    p.add_argument("--ckpt_path", default="auto",
                   help="'auto' resumes from <run_dir>/checkpoints/last.ckpt when present; "
                        "'none' forces a fresh start; otherwise an explicit .ckpt")
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--run_description", default=None)
    args = ResolveRecipe(p.parse_args(argv))
    if args.dataset_name is None:
        args.dataset_name = args.robot_name
    return args


def ResolveRecipe(args):
    """Fill every recipe-dependent argument the command line left at None."""
    defaults = RECIPE_DEFAULTS[args.recipe]
    for key in ("batch_size", "max_steps", "learning_rate", "weight_decay", "gradient_clip_val",
                "eval_every", "checkpoint_every"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    args.optimizer, args.scheduler = defaults["optimizer"], defaults["scheduler"]
    for key, fallback in (("gamma", 0.9795), ("step_lr_every", 2441), ("plateau_factor", 0.5),
                          ("plateau_patience", 500), ("plateau_ema", 0.99)):
        if getattr(args, key) is None:
            setattr(args, key, defaults.get(key, fallback))
    return args


def main():
    args = parse_args()
    print(f"[train_loink] DATASET_DIR={ikflow_config.DATASET_DIR} (resolved from HOME at import)")
    for k, v in vars(args).items():
        print(f"  {k}={v}")
    seed_everything(args.seed, workers=True)

    robot = ResolveRobot(args.robot_name)
    if robot.ndof <= TASK_DIM:
        raise ValueError(f"{args.robot_name} has ndof {robot.ndof}: LOInK needs a redundant "
                         f"robot (ndof > {TASK_DIM})")
    scale = JointScale(robot)
    data_module = CVAEDataset(args.dataset_name, args.batch_size, args.val_set_size,
                              [DATASET_TAG_NON_SELF_COLLIDING])
    assert data_module._samples_tr.shape[1] == robot.ndof, "dataset is not this robot's"
    assert data_module._endpoints_tr.shape[1] == 7, "dataset poses are not [xyz, wxyz]"
    branch = None
    if args.branch == "nominal":
        branch = [int(v) for v in NominalBranch(robot.name)]
        n_checked = CheckOnBranch(robot.name, data_module._samples_tr, branch, args.dataset_name)
        print(f"[train_loink] branch {BranchString(branch)} (nominal): the first {n_checked} "
              f"training rows of {args.dataset_name!r} are all on it")
    j_mean, j_std = CostStats(args.robot_name, args.cost, data_module._samples_tr, robot)
    print(f"[train_loink] cost {args.cost}: j_mean {j_mean}, j_std {j_std} "
          f"(first {min(J_STATS_SAMPLES, data_module._samples_tr.shape[0])} training samples)")

    model = LoinkLitModel(
        robot_name=args.robot_name, ndof=robot.ndof, units=list(args.units), depth=args.depth,
        mu=args.mu, activation=args.activation, cost=args.cost, head_hidden=args.head_hidden,
        n_dys=args.n_dys, n_newton=args.n_newton, scale=scale.tolist(), j_mean=j_mean,
        j_std=j_std, lam=args.lam, tau_penalty=args.tau_penalty, recipe=args.recipe,
        optimizer=args.optimizer, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, scheduler=args.scheduler, gamma=args.gamma,
        step_lr_every=args.step_lr_every, plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience, plateau_ema=args.plateau_ema,
        gradient_clip=args.gradient_clip_val, log_every=args.log_every,
        fk_eval_n=args.fk_eval_n, inv_eval_n=args.inv_eval_n, run_dir=args.run_dir,
        dataset_name=args.dataset_name, branch=branch)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[train_loink] {args.robot_name}: ndof {robot.ndof}, K {robot.ndof - TASK_DIM}, "
          f"depth {args.depth} x units {args.units} {args.activation}, mu {args.mu}, cost "
          f"{args.cost}, recipe {args.recipe}, {n_params} parameters")

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    loggers = [CSVLogger(save_dir=args.run_dir, name="metrics")]
    if not args.disable_wandb:
        import wandb
        assert args.wandb_project and args.wandb_entity, \
            "set WANDB_PROJECT/WANDB_ENTITY (or --wandb_project/--wandb_entity), or --disable_wandb"
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_DIR", args.run_dir)
        cfg = dict(vars(args), family=FAMILY, robot=args.robot_name, world_size=1,
                   samples_per_step=args.batch_size, n_params=n_params, j_mean=j_mean,
                   j_std=j_std)
        data_module.add_dataset_hashes_to_cfg(cfg)
        wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                   notes=args.run_description, config=cfg)
        loggers.append(WandbLogger(save_dir=args.run_dir))

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=args.checkpoint_every,
        save_on_train_epoch_end=False,
        save_top_k=-1,
        save_last=True,
        filename="loink-checkpoint-{step}",
    )
    trainer = Trainer(
        logger=loggers,
        callbacks=[checkpoint_callback],
        val_check_interval=args.eval_every,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        devices=1,
        plugins=[LightningEnvironment()],
        log_every_n_steps=args.log_every,
        max_epochs=-1,
        max_steps=args.max_steps,
        num_sanity_val_steps=1,
        enable_progress_bar=not (os.getenv("IS_SLURM") is not None),
    )

    resume = args.ckpt_path
    if resume == "auto":
        last = os.path.join(ckpt_dir, "last.ckpt")
        resume = last if os.path.isfile(last) else None
    elif resume == "none":
        resume = None
    print(f"[train_loink] resume={resume}")

    t0 = time.monotonic()
    trainer.fit(model, data_module, ckpt_path=resume)
    elapsed = time.monotonic() - t0
    print(f"THROUGHPUT steps={trainer.global_step} elapsed_s={elapsed:.1f} "
          f"steps_per_s={trainer.global_step / elapsed:.3f} "
          f"samples_per_s={trainer.global_step * args.batch_size / elapsed:.0f} "
          f"world_size={trainer.world_size} batch_per_rank={args.batch_size}")


if __name__ == "__main__":
    main()
