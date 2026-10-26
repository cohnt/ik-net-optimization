"""The conditional-VAE chart: the network-architecture ablation's first rung.

Why this module exists
----------------------
The learned formulation is `q = chart(c, z) + correction`, and until the chart ablation the
chart was always an IKFlow normalizing flow. The ablation asks how much of the learned arm's
behaviour is the FLOW, so the simplest network that does IK while still representing
self-motion is swapped in: a standard conditional VAE (Sohn 2015; the baseline of Ardizzone
2019 / Kruse 2021; Kim & Perez ICRA 2021 is the robotics precedent). Its DECODER is the chart,
`q = D(c, z)`; everything else in the formulation -- eq. (6), the correction, the `c` box, the
latent box and trust region, frame calibration -- is untouched, so the only thing ablated is
the network.

The model
---------
    encoder  [q / scale (ndof), pose7] -> (mu, log sigma^2)      K each
    decoder  [pose7, z (K)]            -> q / scale              linear, UNBOUNDED output

`pose7` is the dataset's `[x, y, z, qw, qx, qy, qz]` -- no softflow column, which is a flow
training device. `scale` is the flow's own per-joint `max(|lo|, |hi|)` (ikflow's first
layer is that offset-free rescaling), registered as a buffer so a chart carries the scaling
it was trained at and `decode` returns q in robot units. The output is deliberately not
squashed to the joint limits: squashing would make the program's joint-limit rows vacuous,
which is a different formulation, and the flow is not squashed either.

The decoder variance comes in two variants, the architecture field `decoder_variance`:

  * "learned" (the first rung, and the default so its checkpoints and sidecars load
    unchanged): a Gaussian likelihood with a LEARNED per-joint output log-std `log_sigma_q`
    (a "calibrated decoder"), so the reconstruction-versus-KL balance is fitted rather than
    hand-picked. Trained, it modelled the Panda's joints 1-3 as noise (sigma ~0.55 in
    normalised units) instead of encoding them in z, and the screw arm collapsed (KL 0).
  * "fixed" (Thomas, 2026-10-19): the TEXTBOOK cVAE objective, sum over joints of the squared
    reconstruction error in normalised units + KL(q(z|q,c) || N(0, I)), beta = 1 -- a
    Gaussian likelihood with sigma^2 = 1/2, constants dropped. The model then has NO
    `log_sigma_q` parameter, so unlike the activation this field IS recoverable from the
    state dict's keys and is cross-checked on every load.

Decode and encode are identical in both; only the training objective differs.

The paired start is "encoder only" (Thomas, 2026-10-18): `encode` returns the encoder MEAN,
with no refinement; whatever `q - decode(c, mu)` is left over goes to the correction, and a
cell whose residual exceeds the correction box is `unrepresentable_start` under the existing
rule. Hence `exact_inverse = False`.

What the sidecar cross-check can and cannot catch
-------------------------------------------------
As for flows (`src/flow_loading.py`), a `.pkl` is a bare state dict and the architecture
lives in a mandatory `<name>.arch.json` sidecar. `ndof`, `dim_latent_space`, `hidden`,
`depth` and `pose_dim` are all recoverable from tensor shapes and are verified on every
load. **`activation` is not**: SiLU and tanh weights have identical shapes, so a mismatched
activation loads cleanly and is silently a different chart -- the cVAE's `rnvp_clamp`. That
is why the sidecar is mandatory here, with no legacy fallback at all (there are no
pre-sidecar cVAE checkpoints to be backwards-compatible with).
"""

import json
import os
import pickle
import re

import torch
from torch import nn

from src.charts.base import Chart
from src.flow_loading import SidecarPath, WriteArch

FAMILY = "cvae"
POSE_DIM = 7

#: Smooth activations only. The decoder sits INSIDE an NLP: IPOPT and SNOPT step on its
#: exact Jacobian (one `jacrev` per iterate), and AL-SVGD's Gauss-Newton blocks use it too.
#: A ReLU network is piecewise linear, so its Jacobian is piecewise CONSTANT and jumps
#: across every activation boundary -- a non-C1 map that breaks the solvers' local models
#: (line searches, quasi-Newton updates, convergence tests). SiLU is the fielded choice.
ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh}

#: The fields that define the chart's forward pass. Everything else in a sidecar
#: (training recipe, provenance) is carried for the record and not enforced.
ARCH_FIELDS = ("ndof", "dim_latent_space", "hidden", "depth", "activation", "pose_dim",
               "decoder_variance")

#: The output-variance variants (module docstring). A sidecar written before the field
#: existed has no `decoder_variance` and reads as "learned", the only variant there was.
DECODER_VARIANCES = ("learned", "fixed")
DEFAULT_DECODER_VARIANCE = "learned"


def JointScale(robot):
    """Per-joint `max(|lo|, |hi|)`: ikflow's offset-free first-layer normalisation.

    On the CPU explicitly: jrl sets torch's default device to CUDA at import."""
    return torch.tensor([max(abs(lo), abs(hi)) for lo, hi in robot.actuated_joints_limits],
                        dtype=torch.float32, device="cpu")


def _MLP(n_in, n_out, hidden, depth, activation):
    """`depth` hidden layers of width `hidden`, each Linear + activation, then a LINEAR
    output layer. As an `nn.Sequential` the Linears sit at even indices, which is what
    `ArchFromCVAEStateDict` reads back."""
    layers, width = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), ACTIVATIONS[activation]()]
        width = hidden
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


class CVAEModel(nn.Module):
    """A standard conditional VAE over joint configurations, conditioned on a pose.

    `forward(z, pose7)` IS the decoder, in robot units. That is deliberate: when the chart
    is `torch.compile`d (`compile_model`), the compiled wrapper compiles `forward` and
    forwards every other attribute to the uncompiled module, so the call the program makes
    on every iterate has to be `forward` -- exactly as the flow's chart call is the flow's
    own `forward(..., rev=True)`.
    """

    def __init__(self, ndof, dim_latent_space, scale, hidden=512, depth=6, activation="silu",
                 pose_dim=POSE_DIM, decoder_variance=DEFAULT_DECODER_VARIANCE):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation {activation!r} not in {sorted(ACTIVATIONS)}; the "
                             "chart must be smooth (see ACTIVATIONS)")
        if decoder_variance not in DECODER_VARIANCES:
            raise ValueError(f"decoder_variance {decoder_variance!r} not in {DECODER_VARIANCES}")
        self.decoder_variance = decoder_variance
        self.arch = {"ndof": int(ndof), "dim_latent_space": int(dim_latent_space),
                     "hidden": int(hidden), "depth": int(depth), "activation": activation,
                     "pose_dim": int(pose_dim), "decoder_variance": decoder_variance}
        self.ndof = int(ndof)
        self.dim_latent_space = int(dim_latent_space)
        scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if scale.shape != (self.ndof,):
            raise ValueError(f"scale has shape {tuple(scale.shape)}, expected ({self.ndof},)")
        self.register_buffer("scale", scale)
        self.encoder = _MLP(ndof + pose_dim, 2 * dim_latent_space, hidden, depth, activation)
        self.decoder = _MLP(pose_dim + dim_latent_space, ndof, hidden, depth, activation)
        ## Per-joint output log-std of the Gaussian likelihood, in NORMALISED units. Fitted
        ## jointly with the network (its optimum is the per-joint reconstruction RMS), which
        ## is what sets the reconstruction-versus-KL balance without a hand-picked weight.
        ## The "fixed" variant has no such parameter at all (sigma^2 = 1/2 is in the loss).
        if decoder_variance == "learned":
            self.log_sigma_q = nn.Parameter(torch.zeros(ndof))

    def encode_dist(self, q, pose7):
        """(mu, logvar) of the approximate posterior; `q` in robot units."""
        h = self.encoder(torch.cat([q / self.scale, pose7], dim=-1))
        return h[..., :self.dim_latent_space], h[..., self.dim_latent_space:]

    def encode_mean(self, q, pose7):
        return self.encode_dist(q, pose7)[0]

    def decode_normalised(self, z, pose7):
        return self.decoder(torch.cat([pose7, z], dim=-1))

    def forward(self, z, pose7):
        return self.decode_normalised(z, pose7) * self.scale


def elbo_loss(model, q, pose7):
    """Negative ELBO, averaged over the batch, and its parts.

    Reparameterised `z = mu + exp(logvar / 2) * eps`; KL(N(mu, sigma^2) || N(0, I)) summed
    over the K latent dimensions; the reconstruction term by `model.decoder_variance`:
    "learned", the Gaussian NLL of `q / scale` under the decoder mean and the learned
    per-joint `log_sigma_q`; "fixed", the textbook objective, the squared error of `q / scale`
    summed over joints (the Gaussian NLL at sigma^2 = 1/2 with its constants dropped), beta = 1.
    Returns `(loss, parts)` where `parts` holds `recon_nll` (the reconstruction term as it
    enters the loss), `kl`, `kl_per_dim` (a [K] vector, the batch mean -- a dimension whose
    KL stays near zero has collapsed to the prior and decodes nothing) and `recon_mse` (in
    normalised units, comparable across robots and variants).
    """
    x = q / model.scale
    mu, logvar = model.encode_dist(q, pose7)
    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    x_hat = model.decode_normalised(z, pose7)

    sq = (x - x_hat) ** 2
    if model.decoder_variance == "fixed":
        recon_nll = sq.sum(dim=-1)
    else:
        log_sigma = model.log_sigma_q
        recon_nll = 0.5 * (sq * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma
                           + torch.log(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device))
                           ).sum(dim=-1)
    kl_per = 0.5 * (mu ** 2 + torch.exp(logvar) - 1.0 - logvar)
    kl = kl_per.sum(dim=-1)
    loss = (recon_nll + kl).mean()
    return loss, {"recon_nll": recon_nll.mean(), "kl": kl.mean(),
                  "kl_per_dim": kl_per.mean(dim=0), "recon_mse": sq.mean()}


class CVAEChart(Chart):
    """The cVAE as a chart: `decode` is the decoder, `encode` the encoder mean.

    Both are pure torch on the module passed in -- never on `self.nn_model` -- so they work
    on the private dtype copies `batched_flow` makes and inside `torch.func.jacrev` and
    `torch.compile`. No data-dependent Python control flow.
    """

    family = FAMILY
    has_density = False
    exact_inverse = False

    def __init__(self, nn_model, robot, arch, arch_source=None):
        self.nn_model = nn_model
        self.robot = robot
        self.arch = dict(arch)
        self.arch_source = arch_source
        self.network_width = int(arch["dim_latent_space"])
        self.ndof = int(robot.ndof)

    def decode(self, model, z, pose7):
        return model(z, pose7), None

    def encode(self, model, q, pose7):
        return model.encode_mean(q, pose7)


# ---- sidecar, shape cross-check and loading -------------------------------------------


def WriteCVAEArch(checkpoint_pkl, arch, provenance=None):
    """Write `<stem>.arch.json` for a cVAE chart, with `"family": "cvae"`."""
    missing = [k for k in ARCH_FIELDS if k not in arch]
    if missing:
        raise ValueError(f"cVAE arch is missing {missing}")
    return WriteArch(checkpoint_pkl, dict(arch, family=FAMILY), provenance=provenance)


def _Linears(state_dict, prefix):
    idx = sorted(int(m.group(1)) for k in state_dict
                 if (m := re.fullmatch(rf"{prefix}\.(\d+)\.weight", k)))
    if not idx:
        raise ValueError(f"state dict has no '{prefix}.*.weight' keys -- not a cVAE chart")
    if idx != list(range(0, 2 * len(idx), 2)):
        raise ValueError(f"unexpected {prefix} layout: Linear indices {idx}")
    return [state_dict[f"{prefix}.{i}.weight"] for i in idx]


def ArchFromCVAEStateDict(state_dict):
    """Recover every structurally-visible field from a raw `CVAEModel` state dict.

    Returns `ndof`, `dim_latent_space`, `hidden`, `depth`, `pose_dim` and
    `decoder_variance` ("learned" iff a `log_sigma_q` tensor is present). `activation` is
    deliberately absent: it leaves no trace in the weights.
    """
    enc, dec = _Linears(state_dict, "encoder"), _Linears(state_dict, "decoder")
    ndof = int(dec[-1].shape[0])
    two_k = int(enc[-1].shape[0])
    if two_k % 2:
        raise ValueError(f"encoder output width {two_k} is odd; expected 2K (mu, logvar)")
    k = two_k // 2
    pose_dim = int(dec[0].shape[1]) - k
    hidden = int(enc[0].shape[0])
    depth = len(enc) - 1

    problems = []
    if len(dec) != len(enc):
        problems.append(f"encoder has {len(enc) - 1} hidden layers, decoder {len(dec) - 1}")
    if int(enc[0].shape[1]) != ndof + pose_dim:
        problems.append(f"encoder input {int(enc[0].shape[1])} != ndof {ndof} + pose {pose_dim}")
    for name, layers in (("encoder", enc), ("decoder", dec)):
        widths = [int(w.shape[0]) for w in layers[:-1]] + [int(w.shape[1]) for w in layers[1:]]
        if any(w != hidden for w in widths):
            problems.append(f"{name} hidden widths {widths} are not all {hidden}")
    decoder_variance = "learned" if "log_sigma_q" in state_dict else "fixed"
    for name in ("log_sigma_q", "scale") if decoder_variance == "learned" else ("scale",):
        if name not in state_dict:
            problems.append(f"no '{name}' tensor")
        elif tuple(state_dict[name].shape) != (ndof,):
            problems.append(f"'{name}' has shape {tuple(state_dict[name].shape)}, expected ({ndof},)")
    if problems:
        raise ValueError("not a well-formed cVAE state dict:\n  " + "\n  ".join(problems))
    return {"ndof": ndof, "dim_latent_space": k, "hidden": hidden, "depth": depth,
            "pose_dim": pose_dim, "decoder_variance": decoder_variance}


def ReadCVAEArch(checkpoint):
    """Read the MANDATORY sidecar of a cVAE checkpoint; raise if it is absent or not a cVAE's."""
    sidecar = SidecarPath(checkpoint)
    if not os.path.exists(sidecar):
        raise FileNotFoundError(
            f"no architecture sidecar at {sidecar}. A cVAE chart has no legacy fallback: "
            "the activation cannot be recovered from the weights, so loading without the "
            "sidecar could silently build a different chart. Export it with "
            "scripts/training/export_cvae.py.")
    with open(sidecar) as f:
        payload = json.load(f)
    if payload.get("family") != FAMILY:
        raise ValueError(f"{sidecar} declares family {payload.get('family')!r}, not {FAMILY!r}")
    ## Written before the field existed: the only variant there was (DECODER_VARIANCES).
    payload.setdefault("decoder_variance", DEFAULT_DECODER_VARIANCE)
    missing = [k for k in ARCH_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"{sidecar} is missing the cVAE arch fields {missing}")
    return payload, sidecar


def VerifyCVAEArch(declared, state_dict, source):
    """Raise if the sidecar disagrees with what the weights structurally imply."""
    actual = ArchFromCVAEStateDict(state_dict)
    declared = dict(declared)
    declared.setdefault("decoder_variance", DEFAULT_DECODER_VARIANCE)
    mismatches = [f"{k}: sidecar says {declared[k]!r}, weights imply {actual[k]!r}"
                  for k in actual if declared[k] != actual[k]]
    if mismatches:
        raise ValueError(
            f"architecture in {source} does not match the weights:\n  " + "\n  ".join(mismatches)
            + "\nLoading this checkpoint would silently produce a different chart.")


def LoadCVAEChart(robot_name, checkpoint, compile_model=None):
    """Build a `CVAEChart` from a bare state-dict `.pkl` and its mandatory sidecar.

    Every field recoverable from shapes is cross-checked and a contradiction RAISES. Two
    further guards, both cheap: the sidecar's `robot_name` (when present) must be the one
    asked for, and the chart's `scale` buffer must equal the robot's own joint scaling --
    the latter is what stops a Panda chart loading against the 7-DoF screw arm, which no
    shape check can tell apart.
    """
    ## Imported here rather than at module top so a trainer can import the model before its
    ## device preamble has neutralised jrl's `torch.set_default_device` side effect.
    from jrl.config import DEVICE
    from jrl.robots import get_robot

    payload, source = ReadCVAEArch(checkpoint)
    if payload.get("robot_name") not in (None, robot_name):
        raise ValueError(f"{source} was trained for {payload['robot_name']!r}, "
                         f"not {robot_name!r}")
    arch = {k: payload[k] for k in ARCH_FIELDS}

    with open(checkpoint, "rb") as f:
        state_dict = pickle.load(f)
    VerifyCVAEArch(arch, state_dict, source)

    robot = get_robot(robot_name)
    if int(robot.ndof) != arch["ndof"]:
        raise ValueError(f"{source}: chart has ndof {arch['ndof']}, robot {robot_name!r} "
                         f"has {robot.ndof}")
    expected_scale = JointScale(robot)
    if not torch.allclose(state_dict["scale"].float().cpu(), expected_scale, rtol=0, atol=1e-6):
        raise ValueError(f"{source}: the chart's joint scale {state_dict['scale'].tolist()} is "
                         f"not robot {robot_name!r}'s {expected_scale.tolist()} -- trained for "
                         "a different robot?")

    model = CVAEModel(arch["ndof"], arch["dim_latent_space"], expected_scale,
                      hidden=arch["hidden"], depth=arch["depth"],
                      activation=arch["activation"], pose_dim=arch["pose_dim"],
                      decoder_variance=arch["decoder_variance"])
    model.load_state_dict(state_dict, strict=True)
    ## The same device the flow lands on: IKFlowSolver builds its network under jrl's
    ## `torch.set_default_device(DEVICE)`, i.e. CUDA whenever it is available.
    model = model.to(DEVICE).eval()
    nn_model = torch.compile(model, **compile_model) if compile_model is not None else model
    return CVAEChart(nn_model, robot, dict(arch, family=FAMILY, robot_name=robot_name),
                     arch_source=source)
