"""Train the conditional-VAE chart (`src/charts/cvae.py`) on one GPU.

The chart ablation's cVAE rung, trained with the flow rungs' recipe so that the only thing
that differs between a flow chart and a cVAE chart is the model: AdamW, lr 1.5e-4, StepLR
x0.9795 every 2441 steps, weight decay 1.8e-5, gradient clipping by VALUE at 1 (the fork's
`IkfLitModel.on_after_backward`), 620000 steps, seed 0, the same jrl datasets through the
fork's own `IkfLitDataset`. The batch is 4096 on ONE GPU, which is the flows' 8 ranks x 512
global batch, so the same number of samples is seen per step and over the run.

Single GPU, no DDP: the model is ~2.7M parameters and a V100 is not the bottleneck, so the
cluster runs all eight (robot x rung) networks concurrently, one per GPU, instead.

Cluster conventions are the flow trainer's (`third_party/ikflow/scripts/train_ddp.py`), so
the tooling works the same:
  * `--ckpt_path auto` resumes from `<run_dir>/checkpoints/last.ckpt`; every checkpoint is
    kept (`save_top_k=-1`, see train_ddp.py for why);
  * CSV metrics under `<run_dir>/metrics`, wandb OFFLINE with `WANDB_DIR=<run_dir>`;
  * `<run_dir>/status.json`, the heartbeat `cluster/submit_ladder.sh --status` reads, with
    the PoleFractionCallback's keys (`pole` is null: the flow's pole screen is the flow's;
    the cVAE's runaway screen is `scripts/training/screen_cvae.py`, run after export);
  * a THROUGHPUT line at the end.

IMPORT ORDER IS LOAD-BEARING, twice over:
  1. ikflow resolves DATASET_DIR from `expanduser("~")` AT IMPORT. The cluster payload sets
     HOME to the persistent tree before Python starts (as `cluster/train_flow.sh` does);
     the resolved directory is printed first thing so a wrong one is visible in the log
     rather than as a missing-file error deep inside setup.
  2. jrl calls `torch.set_default_device(DEVICE)` at import, which poisons every
     default-device tensor creation in the data path. It is reset to CPU straight after,
     as train_ddp.py does, and Lightning moves the module.

    python scripts/training/train_cvae.py --robot_name panda --run_dir <dir> \
        --dim_latent_space 7 --ckpt_path auto
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

import src.register_robots  # noqa: E402,F401  -- before anything resolves a robot

import numpy as np  # noqa: E402
from jrl.math_utils import geodesic_distance_between_quaternions  # noqa: E402
from jrl.robots import get_robot  # noqa: E402
from pytorch_lightning import LightningModule, seed_everything  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger, WandbLogger  # noqa: E402
from pytorch_lightning.plugins.environments import LightningEnvironment  # noqa: E402
from pytorch_lightning.trainer import Trainer  # noqa: E402
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset  # noqa: E402

from ikflow import config as ikflow_config  # noqa: E402
from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING  # noqa: E402
from ikflow.training.lt_data import IkfLitDataset  # noqa: E402

from src.charts.cvae import ACTIVATIONS, CVAEModel, JointScale, elbo_loss  # noqa: E402

#: A latent dimension whose batch-mean KL to the prior is below this (nats) carries no
#: information: the encoder ignores it and the decoder learns to ignore it too. In the
#: flow-width rung, the dimensions beyond the redundancy are EXPECTED to collapse, and then
#: they are flat decision variables in the program -- worth knowing before benchmarking.
ACTIVE_KL_NATS = 0.01


def ResolveRobot(robot_name):
    """jrl's Robot.__init__ truncate-rewrites a cached URDF under $HOME/.cache/jrl on every
    construction, and two single-GPU runs of the same robot can share a node (the K = width
    and K = redundancy rungs). A run that reads while the other writes sees half-written XML
    (the flow campaign lost job 5549615 to this), so retry as train_ddp.py does."""
    for attempt in range(3):
        try:
            return get_robot(robot_name)
        except (AssertionError, ValueError):
            if attempt == 2:
                raise
            time.sleep(5.0)


class CVAEDataset(IkfLitDataset):
    """The fork's dataset module -- same files, same split, same loading -- with batched
    indexing.

    The fork's loader indexes a `TensorDataset` one row at a time and collates, which costs
    more than this MLP's whole step at batch 4096. A `BatchSampler` hands the dataset a
    list of indices, and `TensorDataset` answers that with ONE fancy-index per tensor. The
    sampling distribution (uniform without replacement per epoch, drop_last) is the same.
    """

    def train_dataloader(self):
        dataset = TensorDataset(self._samples_tr, self._endpoints_tr)
        sampler = BatchSampler(RandomSampler(dataset), batch_size=self._batch_size,
                               drop_last=True)
        return DataLoader(dataset, sampler=sampler, batch_size=None)

    def val_dataloader(self):
        n = self._val_set_size
        dataset = TensorDataset(self._samples_te[:n], self._endpoints_te[:n])
        return DataLoader(dataset, batch_size=n, shuffle=False)


class CVAELitModel(LightningModule):
    def __init__(self, robot_name, ndof, dim_latent_space, hidden, depth, activation, scale,
                 learning_rate, gamma, step_lr_every, weight_decay, gradient_clip,
                 log_every, fk_eval_n, run_dir):
        super().__init__()
        ## Everything the exporter needs to rebuild the arch is a constructor argument, so
        ## `save_hyperparameters` puts it in every checkpoint's `hyper_parameters`.
        self.save_hyperparameters()
        self.model = CVAEModel(ndof, dim_latent_space, scale, hidden=hidden, depth=depth,
                               activation=activation)
        self._robot = None
        self._last_tr_loss = None

    @property
    def robot(self):
        if self._robot is None:
            self._robot = ResolveRobot(self.hparams.robot_name)
        return self._robot

    def on_fit_start(self):
        _ = self.robot

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_lr_every, gamma=self.hparams.gamma)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_lr(self):
        return self.optimizers().optimizer.param_groups[0]["lr"]

    def safe_log_metrics(self, values):
        ## To EVERY logger (CSV and wandb): `self.logger` is only the first.
        for logger in self.loggers:
            logger.log_metrics(values, step=self.global_step)

    def training_step(self, batch, batch_idx):
        q, pose = batch
        loss, parts = elbo_loss(self.model, q, pose)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {self.global_step}: {loss.item()}")
        if self.global_step % self.hparams.log_every == 0 and self.global_step > 0:
            self._last_tr_loss = float(loss.item())
            self.safe_log_metrics({
                "tr/loss": self._last_tr_loss,
                "tr/recon_nll": float(parts["recon_nll"]),
                "tr/kl": float(parts["kl"]),
                "tr/recon_mse": float(parts["recon_mse"]),
                "tr/learning_rate": self.get_lr(),
                "tr/sigma_q_mean": float(self.model.log_sigma_q.exp().mean()),
            })
        return loss

    def on_after_backward(self):
        ## By VALUE at 1, as the flow recipe does (IkfLitModel.on_after_backward).
        torch.nn.utils.clip_grad_value_(self.parameters(), self.hparams.gradient_clip)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        q, pose = batch
        loss, parts = elbo_loss(self.model, q, pose)
        kl_per_dim = parts["kl_per_dim"].detach().cpu().numpy()
        metrics = {
            "val/loss": float(loss),
            "val/recon_nll": float(parts["recon_nll"]),
            "val/recon_mse": float(parts["recon_mse"]),
            "val/kl": float(parts["kl"]),
            "val/active_dims": int((kl_per_dim > ACTIVE_KL_NATS).sum()),
        }
        for i, v in enumerate(kl_per_dim):
            metrics[f"val/kl_dim{i}"] = float(v)
        metrics.update(self._fk_error(pose[:self.hparams.fk_eval_n]))
        self.last_val_metrics = metrics

    def _fk_error(self, pose):
        """FK error of `decode(pose, z ~ N(0, I))`: the chart a native start would see.

        Kept to a small subset: the soft PCS arm's FK is float64 on the CPU and dominates
        validation otherwise. Errors in metres / radians under the flow's status keys, and
        in mm / deg under the cVAE's own.
        """
        z = torch.randn(pose.shape[0], self.hparams.dim_latent_space, device=pose.device)
        q = self.model(z, pose).double().cpu()
        ## At the robot's own default dtype: jrl's rigid-arm FK is float32 inside whatever
        ## `dtype` says (float64 raises); this project's robots compute in float64.
        realised = self.robot.forward_kinematics(q, out_device="cpu").double()
        target = pose.double().cpu()
        pos = torch.linalg.norm(realised[:, :3] - target[:, :3], dim=1).numpy()
        ang = geodesic_distance_between_quaternions(realised[:, 3:7], target[:, 3:7]).numpy()
        return {"val/l2_error": float(pos.mean()), "val/angular_error": float(ang.mean()),
                "val/fk_pos_mm_median": float(1e3 * np.median(pos)),
                "val/fk_pos_mm_p95": float(1e3 * np.percentile(pos, 95)),
                "val/fk_ang_deg_median": float(np.degrees(np.median(ang))),
                "val/fk_ang_deg_p95": float(np.degrees(np.percentile(ang, 95))),
                "val/q_absmax": float(q.abs().max())}

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = getattr(self, "last_val_metrics", None)
        if metrics is None:
            return
        self.safe_log_metrics(metrics)
        self._write_status(metrics)

    def _write_status(self, val):
        """The PoleFractionCallback's heartbeat, key for key."""
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
            ## cVAE-only extras; the status reader ignores keys it does not know.
            "family": "cvae",
            "val_active_dims": val.get("val/active_dims"),
            "val_kl": val.get("val/kl"),
            "val_recon_mse": val.get("val/recon_mse"),
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
    p.add_argument("--dim_latent_space", type=int, required=True,
                   help="K: the flow's width (Panda 7, iiwa 8, screw 7, soft12 12) or the "
                        "redundancy (DOF - 6)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--activation", default="silu", choices=sorted(ACTIVATIONS))
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--learning_rate", type=float, default=1.5e-4)
    p.add_argument("--step_lr_every", type=int, default=2441)
    p.add_argument("--gamma", type=float, default=0.9795)
    p.add_argument("--weight_decay", type=float, default=1.8e-5)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=620000)
    p.add_argument("--eval_every", type=int, default=20000)
    p.add_argument("--checkpoint_every", type=int, default=20000)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--val_set_size", type=int, default=20000,
                   help="test-split rows for the ELBO metrics (one GPU batch)")
    p.add_argument("--fk_eval_n", type=int, default=500,
                   help="of those, how many get the FK-error screen (CPU FK for the soft arm)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_path", default="auto",
                   help="'auto' resumes from <run_dir>/checkpoints/last.ckpt when present; "
                        "'none' forces a fresh start; otherwise an explicit .ckpt")
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--run_description", default=None)
    return p.parse_args(argv)


def main():
    args = parse_args()
    print(f"[train_cvae] DATASET_DIR={ikflow_config.DATASET_DIR} (resolved from HOME at import)")
    for k, v in vars(args).items():
        print(f"  {k}={v}")
    seed_everything(args.seed, workers=True)

    robot = ResolveRobot(args.robot_name)
    scale = JointScale(robot)
    data_module = CVAEDataset(robot.name, args.batch_size, args.val_set_size,
                              [DATASET_TAG_NON_SELF_COLLIDING])
    assert data_module._samples_tr.shape[1] == robot.ndof, "dataset is not this robot's"
    assert data_module._endpoints_tr.shape[1] == 7, "dataset poses are not [xyz, wxyz]"

    model = CVAELitModel(
        robot_name=args.robot_name, ndof=robot.ndof, dim_latent_space=args.dim_latent_space,
        hidden=args.hidden, depth=args.depth, activation=args.activation,
        scale=scale.tolist(), learning_rate=args.learning_rate, gamma=args.gamma,
        step_lr_every=args.step_lr_every, weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip_val, log_every=args.log_every,
        fk_eval_n=args.fk_eval_n, run_dir=args.run_dir)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[train_cvae] {args.robot_name}: ndof {robot.ndof}, K {args.dim_latent_space}, "
          f"{args.depth} x {args.hidden} {args.activation}, {n_params} parameters")

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    loggers = [CSVLogger(save_dir=args.run_dir, name="metrics")]
    if not args.disable_wandb:
        import wandb
        assert args.wandb_project and args.wandb_entity, \
            "set WANDB_PROJECT/WANDB_ENTITY (or --wandb_project/--wandb_entity), or --disable_wandb"
        ## Offline unless the caller says otherwise: compute nodes have no internet and no
        ## credentials, and runs are synced from the laptop (cluster/train_flow.sh).
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_DIR", args.run_dir)
        cfg = dict(vars(args), family="cvae", robot=args.robot_name, world_size=1,
                   samples_per_step=args.batch_size, n_params=n_params)
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
        filename="cvae-checkpoint-{step}",
    )
    trainer = Trainer(
        logger=loggers,
        callbacks=[checkpoint_callback],
        val_check_interval=args.eval_every,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        devices=1,
        ## Pinned, as train_ddp.py pins it: inside a Slurm job Lightning would otherwise
        ## auto-select SLURMEnvironment and read the job's task layout.
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
    print(f"[train_cvae] resume={resume}")

    t0 = time.monotonic()
    trainer.fit(model, data_module, ckpt_path=resume)
    elapsed = time.monotonic() - t0
    ## As train_ddp.py: steps/s is diluted by the restored offset on a resumed fit.
    print(f"THROUGHPUT steps={trainer.global_step} elapsed_s={elapsed:.1f} "
          f"steps_per_s={trainer.global_step / elapsed:.3f} "
          f"samples_per_s={trainer.global_step * args.batch_size / elapsed:.0f} "
          f"world_size={trainer.world_size} batch_per_rank={args.batch_size}")


if __name__ == "__main__":
    main()
