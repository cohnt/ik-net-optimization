import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2.interface as wr

X = "trainer/global_step"
ENTITY, PROJECT = "cohnt-massachusetts-institute-of-technology", "ikflow"


def lp(title, y, log_y=False, **kw):
    return wr.LinePlot(title=title, x=X, y=y, log_y=log_y, title_x="optimizer step", **kw)


wksp = ws.Workspace(
    entity=ENTITY,
    project=PROJECT,
    name="iiwa14 retrain — curated",
    settings=ws.WorkspaceSettings(x_axis=X, smoothing_type="exponential", smoothing_weight=0),
    sections=[
        ws.Section(
            name="1. Poles — the reason for this retrain",
            is_open=True,
            panels=[
                lp("frac_gt_1000  (lemon-haze-7 = 0.0334, target <= 0.001)",
                   ["pole/frac_gt_1000"]),
                lp("pole/max  [log]  (lemon-haze-7 = 5.5e16)", ["pole/max"], log_y=True),
                lp("frac_gt_3  (outside joint limits)", ["pole/frac_gt_3"]),
                lp("pole p50 / p99  (bulk of the distribution)", ["pole/p50", "pole/p99"]),
            ],
        ),
        ws.Section(
            name="2. Chart accuracy — the other acceptance criterion",
            is_open=True,
            panels=[
                lp("val l2_error  [log]  (metres)", ["val/l2_error", "val_clamped/l2_error"], log_y=True),
                lp("val angular_error  [log]  (rad)",
                   ["val/angular_error", "val_clamped/angular_error"], log_y=True),
                lp("worst-case error over the val set  [log]",
                   ["val/l2_ave_max_error", "val/angular_ave_max_error"], log_y=True),
                lp("joint limits exceeded / self collisions",
                   ["val/joint_limits_exceeded", "val/self_collisions"]),
            ],
        ),
        ws.Section(
            name="3. Training health — check these if something looks wrong",
            is_open=True,
            panels=[
                lp("tr/loss  (negative log-likelihood; DOWN is good, no floor at 0)",
                   ["tr/loss"]),
                lp("loss_is_nan  — must stay flat at 0", ["tr/loss_is_nan"]),
                lp("learning rate  [log]  — must be a straight line",
                   ["tr/learning_rate"], log_y=True),
                lp("gradients  (grad_max is clipped at 1 by design)",
                   ["tr/grad_abs_ave", "tr/grad_max"]),
                lp("network output magnitude  [log]  — a spike here precedes a pole",
                   ["tr/output_abs_ave", "tr/output_max"], log_y=True),
                lp("throughput  (batches/s across all 8 ranks)", ["tr/batches_p_sec"]),
            ],
        ),
    ],
)

saved = wksp.save()
print("VIEW URL:", saved.url)
