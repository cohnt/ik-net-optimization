"""The two closed-form IK maps in batched, differentiable torch: the svgd twin of each robot's
pydrake `Solve`.

`BatchedAnalyticMap.from_program(program, dtype, device)` reads an analytic program
(`src/analytic_program.py:AnalyticProgramMixin`) and `map(X [N, 7]) -> (q [N, 7], u [N, 4])`
is, for the decision vector `X = [xyz, rpy, psi]`, EXACTLY what the program's
`VarsToConfigAndQ` computes at each `X[i]`: the pose `RigidTransform(RollPitchYaw(rpy), xyz)
@ program.pose_offset`, the closed form on the program's branch `program.gc` with the clip
`1 - analytic_clip_stepback`, the configuration and the four PRE-clip arguments the
`ReachabilityConstraint` rows hold. `tests/test_batched_analytic.py` pins values (<= 1e-12)
and the autograd Jacobian against the AutoDiffXd chain (<= 1e-9) on every branch of both
robots, in the clipped regime and across the Panda's q6 seam.

TRANSCRIBED, NOT RE-DERIVED. Each function below is the pydrake `Solve` line by line
(`src/panda_analytic_ik.py`, `src/iiwa_analytic_ik.py`), batched over N, with the equation
numbers kept, so a reader can hold the two side by side. Where pydrake branches on the
discrete configuration the twin multiplies by a SIGN TENSOR instead, so one traced structure
serves every branch (a compiled / graphed svgd step is cached per structure and rebinds its
constant tensors per cell -- `fused.copy_constants` -- so the branch, the pose offset, the
clip and the seam are all rebindable tensors here, never Python floats a stage would bake):

  iiwa (Faria et al.)  GC = (g2, g4, g6), each +-1, already multiplicative in `Solve`;
  Panda (He & Liu)     B:  q6 = pi/2 (1 + sB) - sB psi6 - phi6   (sB = +1 for B1: pi - psi6 -
                           phi6; -1 for B2: psi6 - phi6),
                       C:  q1 = atan2(sC y, sC x), q2 = sC acos(.)  (sC = +1 for C1, -1 C2),
                       A:  both triangle angles times sA, q4 = ... - pi (1 + sA)  (A = +1:
                           the historical "- 2 pi"; A = -1: no offset). A 2-element gc is
                           the 4-branch chart, A = +1.
  Each identity is exact in floating point too (`pi/2 * 2 == pi`, `0 - (-x) == x`), so the
  twin's VALUES match the branchy original to the last ulp of the shared operations.

THE NON-SMOOTH PIECES, mirrored rather than smoothed:
  * the clip `max(-c, min(c, u))` is `torch.clamp`: zero derivative where active, as the
    pydrake clip's bounds carry none;
  * the Panda's q6 wrap subtracts `2 pi floor((q6 - seam) / 2 pi)` computed on the DETACHED
    value -- a piecewise-constant shift chosen on the value only, as `_value(q[5])` is;
  * the Panda's `safe_norm` / `safe_divide` replace a norm below `eps` by `eps` and a
    denominator below `eps` by `+-eps` -- constants, so zero derivative there. Mirrored with
    the double-`where` trick: the guarded operation runs on a SAFE substitute where the guard
    fires, so the untaken branch never produces an inf / NaN that `where`'s backward would
    multiply by zero and still leak as NaN. Everything pydrake leaves unguarded (`p_26 /
    |p_26|`, `sqrt(x626^2 + y626^2)`, the unsafe `|O2P|` in the q2 argument) is left
    unguarded here too: the twin is the map, singularities included.

House rules (as `batched_program.py`): explicit dtype / device on every tensor built here
(`jrl.config` mutates torch's defaults at import); no Python loop over particles; nothing in
`__call__` creates a tensor from host data (`torch.tensor` would be a host-to-device copy a
CUDA-graph capture refuses) -- every constant is built once, at construction.
"""

import math

import numpy as np
import torch
from torch import Tensor

HALF_PI = 0.5 * math.pi
TWO_PI = 2.0 * math.pi
#: The Panda's `safe_norm` / `safe_divide` eps (src/panda_analytic_ik.py).
PANDA_SAFE_EPS = 1e-6


## ------------------------------------------------------------------------------------ ##
##                                     small helpers                                     ##
## ------------------------------------------------------------------------------------ ##

def rpy_to_R(rpy: Tensor) -> Tensor:
    """`[N, 3]` roll-pitch-yaw -> `[N, 3, 3]`, Drake's convention `R = Rz(y) Ry(p) Rx(r)`
    (`RollPitchYaw(rpy).ToRotationMatrix()`)."""
    cr, sr = torch.cos(rpy[:, 0]), torch.sin(rpy[:, 0])
    cp, sp = torch.cos(rpy[:, 1]), torch.sin(rpy[:, 1])
    cy, sy = torch.cos(rpy[:, 2]), torch.sin(rpy[:, 2])
    r0 = torch.stack([cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=1)
    r1 = torch.stack([cp * sy, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=1)
    r2 = torch.stack([-sp, cp * sr, cp * cr], dim=1)
    return torch.stack([r0, r1, r2], dim=1)


def _cross(a: Tensor, b: Tensor) -> Tensor:
    """Row-wise cross product of `[N, 3]` tensors (written out: compiles and vmaps cleanly)."""
    return torch.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], dim=1)


def _dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=1)


def _norm(v: Tensor) -> Tensor:
    """`np.linalg.norm` of each row, unguarded (`sqrt(v . v)`, as numpy computes it on an
    object array)."""
    return torch.sqrt((v * v).sum(dim=1))


def _skew(a: Tensor) -> Tensor:
    """`[a]x`, `[N, 3, 3]` (`iiwa_analytic_ik.cross_product_matrix`)."""
    z = torch.zeros_like(a[:, 0])
    return torch.stack([torch.stack([z, -a[:, 2], a[:, 1]], dim=1),
                        torch.stack([a[:, 2], z, -a[:, 0]], dim=1),
                        torch.stack([-a[:, 1], a[:, 0], z], dim=1)], dim=1)


def _bmv(A: Tensor, v: Tensor) -> Tensor:
    return (A @ v.unsqueeze(2)).squeeze(2)


def _safe_norm(v: Tensor, eps: float) -> Tensor:
    """The Panda's `safe_norm`: `eps` (a constant) where `|v| < eps`, else `|v|`. A NaN norm
    stays NaN (`NaN < eps` is False there too). Double-where: the square root runs on 1
    where the guard fires, so its backward never sees a zero."""
    sq = (v * v).sum(dim=1)
    small = torch.sqrt(sq.detach()) < eps
    n = torch.sqrt(torch.where(small, torch.ones_like(sq), sq))
    return torch.where(small, torch.full_like(n, eps), n)


def _safe_divide(num: Tensor, den: Tensor, eps: float) -> Tensor:
    """The Panda's `safe_divide`: a denominator with `|den| < eps` becomes the constant
    `+-eps` (sign of `den`, `+` at 0). The substituted denominator is never 0, so the plain
    division is safe on both branches."""
    small = den.detach().abs() < eps
    sub = torch.where(den.detach() >= 0, torch.full_like(den, eps), torch.full_like(den, -eps))
    return num / torch.where(small, sub, den)


## ------------------------------------------------------------------------------------ ##
##                                the iiwa: Faria et al.                                 ##
## ------------------------------------------------------------------------------------ ##

def _iiwa_rot(theta: Tensor, ca: float, sa: float) -> Tensor:
    """The rotation block of the iiwa's standard-DH transform (`Analytic_IK_7DoF.Ts`)."""
    c, s = torch.cos(theta), torch.sin(theta)
    z = torch.zeros_like(theta)
    return torch.stack([torch.stack([c, -s * ca, s * sa], dim=-1),
                        torch.stack([s, c * ca, -c * sa], dim=-1),
                        torch.stack([z, z + sa, z + ca], dim=-1)], dim=-2)


def iiwa_solve(R07: Tensor, p07: Tensor, psi: Tensor, g2, g4, g6, clip, P) -> tuple:
    """`Analytic_IK_7DoF.Solve` batched: `(q [N, 7], u [N, 4])` for DH frame 7 at
    `(R07 [N, 3, 3], p07 [N, 3])`, arm angle `psi [N]`, global configuration `(g2, g4, g6)`
    (+-1 tensors, scalar or `[N]`) and clip `clip` (`1 - clip_stepback`, a tensor). `P` the
    DH constants (`BatchedAnalyticMap._iiwa_params`)."""
    d_bs, d_se, d_ew, d_wf = P["d_bs"], P["d_se"], P["d_ew"], P["d_wf"]
    ca, sa = P["ca"], P["sa"]
    sin_psi, cos_psi = torch.sin(psi), torch.cos(psi)
    lo, hi = -clip, clip

    p26 = p07 - P["e_z"] * d_bs - R07[:, :, 2] * d_wf             # EQ (3)
    sq26 = _dot(p26, p26)
    norm26 = torch.sqrt(sq26)
    p26_hat = p26 / norm26.unsqueeze(1)

    u0 = (sq26 - d_se ** 2 - d_ew ** 2) / (2 * d_se * d_ew)        # EQ (4)
    th4v = g4 * torch.acos(torch.clamp(u0, lo, hi))
    th1v = torch.atan2(p26[:, 1], p26[:, 0])                       # EQ (5)
    u1 = (d_se ** 2 + sq26 - d_ew ** 2) / (2 * d_se * norm26)      # EQ (7)
    phi = torch.acos(torch.clamp(u1, lo, hi))
    th2v = torch.atan2(_norm(p26[:, :2]), p26[:, 2]) + g4 * phi

    R03v = (_iiwa_rot(th1v, ca[0], sa[0]) @ _iiwa_rot(th2v, ca[1], sa[1])
            @ _iiwa_rot(torch.zeros_like(th1v), ca[2], sa[2]))
    K = _skew(p26_hat)                                             # EQ (15)
    A_s = K @ R03v
    B_s = -(K @ K @ R03v)
    C_s = (p26_hat.unsqueeze(2) @ p26_hat.unsqueeze(1)) @ R03v

    def s(A, B, C, i, j):
        return A[:, i, j] * sin_psi + B[:, i, j] * cos_psi + C[:, i, j]

    q1 = torch.atan2(g2 * s(A_s, B_s, C_s, 1, 1), g2 * s(A_s, B_s, C_s, 0, 1))   # EQ (17)-(19)
    u2 = s(A_s, B_s, C_s, 2, 1)
    q2 = g2 * torch.acos(torch.clamp(u2, lo, hi))
    q3 = torch.atan2(-g2 * s(A_s, B_s, C_s, 2, 2), -g2 * s(A_s, B_s, C_s, 2, 0))

    R34T = _iiwa_rot(th4v, ca[3], sa[3]).transpose(1, 2)           # EQ (20)
    A_w = R34T @ A_s.transpose(1, 2) @ R07
    B_w = R34T @ B_s.transpose(1, 2) @ R07
    C_w = R34T @ C_s.transpose(1, 2) @ R07

    q5 = torch.atan2(g6 * s(A_w, B_w, C_w, 1, 2), g6 * s(A_w, B_w, C_w, 0, 2))   # EQ (22)-(24)
    u3 = s(A_w, B_w, C_w, 2, 2)
    q6 = g6 * torch.acos(torch.clamp(u3, lo, hi))
    q7 = torch.atan2(g6 * s(A_w, B_w, C_w, 2, 1), -g6 * s(A_w, B_w, C_w, 2, 0))

    q = torch.stack([q1, q2, q3, th4v, q5, q6, q7], dim=1)
    u = torch.stack([u0, u1, u2, u3], dim=1)
    return q, u


def _iiwa_T(theta: Tensor, ca: float, sa: float, d: float) -> Tensor:
    """The iiwa's full standard-DH transform `[N, 4, 4]`."""
    R = _iiwa_rot(theta, ca, sa)
    N = theta.shape[0]
    t = torch.zeros(N, 3, 1, dtype=theta.dtype, device=theta.device)
    t[:, 2, 0] = d
    top = torch.cat([R, t], dim=2)
    bottom = torch.zeros(N, 1, 4, dtype=theta.dtype, device=theta.device)
    bottom[:, 0, 3] = 1.0
    return torch.cat([top, bottom], dim=1)


def iiwa_psi(q: Tensor, alpha, d) -> Tensor:
    """`Analytic_IK_7DoF.psi` batched (no gradient needed: it initialises a start). The arm
    angle of each configuration `q [N, 7]`: the signed angle from the virtual arm's
    shoulder-elbow-wrist plane to the actual one. `alpha`, `d` the DH arrays (numpy)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    d_bs, d_se, d_ew, d_wf = float(d[0]), float(d[2]), float(d[4]), float(d[6])
    Ts = [_iiwa_T(q[:, k], float(ca[k]), float(sa[k]), float(d[k])) for k in range(7)]
    T07 = Ts[0]
    for T in Ts[1:]:
        T07 = T07 @ T
    p07, R07 = T07[:, :3, 3], T07[:, :3, :3]
    sgn = lambda v: torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
    G4 = sgn(q[:, 3])

    p02 = torch.zeros_like(p07)
    p02[:, 2] = d_bs
    p26 = p07 - p02 - R07[:, :, 2] * d_wf
    arg = (_dot(p26, p26) - d_se ** 2 - d_ew ** 2) / (2 * d_se * d_ew)
    th4v = G4 * torch.acos(torch.clamp(arg, -1.0, 1.0))
    R01 = Ts[0][:, :3, :3]
    cr = _cross(p26, R01[:, :, 2])
    th1v = torch.where(_dot(cr, cr) > 0, torch.atan2(p26[:, 1], p26[:, 0]), torch.zeros_like(arg))
    arg = (d_se ** 2 + _dot(p26, p26) - d_ew ** 2) / (2 * d_se * _norm(p26))
    phi = torch.acos(torch.clamp(arg, -1.0, 1.0))
    th2v = torch.atan2(_norm(p26[:, :2]), p26[:, 2]) + G4 * phi

    thv = [th1v, th2v, torch.zeros_like(th1v), th4v]
    Tv = [_iiwa_T(thv[k], float(ca[k]), float(sa[k]), float(d[k])) for k in range(4)]
    p02v = (Tv[0] @ Tv[1])[:, :3, 3]
    p04v = (Tv[0] @ Tv[1] @ Tv[2] @ Tv[3])[:, :3, 3]
    p04 = (Ts[0] @ Ts[1] @ Ts[2] @ Ts[3])[:, :3, 3]
    p06 = (Ts[0] @ Ts[1] @ Ts[2] @ Ts[3] @ Ts[4] @ Ts[5])[:, :3, 3]

    unit = lambda v: v / _norm(v).unsqueeze(1)
    v_sew_v = unit(_cross(unit(p04v - p02v), unit(p06 - p02v)))
    v_sew = unit(_cross(unit(p04 - p02), unit(p06 - p02)))
    sg = torch.sign(_dot(_cross(v_sew_v, v_sew), p26))
    return sg * torch.acos(torch.clamp(_dot(v_sew_v, v_sew), -1.0, 1.0))


## ------------------------------------------------------------------------------------ ##
##                                 the Panda: He & Liu                                   ##
## ------------------------------------------------------------------------------------ ##

def _panda_rot(theta: Tensor, ca: float, sa: float) -> Tensor:
    """The rotation block of the Panda's modified-DH transform (`Analytic_IK_Panda.Ts`)."""
    c, s = torch.cos(theta), torch.sin(theta)
    z = torch.zeros_like(theta)
    return torch.stack([torch.stack([c, -s, z], dim=-1),
                        torch.stack([s * ca, c * ca, z - sa], dim=-1),
                        torch.stack([s * sa, c * sa, z + ca], dim=-1)], dim=-2)


def panda_solve(R: Tensor, p: Tensor, psi: Tensor, sB, sC, sA, clip, seam, P) -> tuple:
    """`Analytic_IK_Panda.Solve` batched: `(q [N, 7], u [N, 4])` for the analytic frame at
    `(R [N, 3, 3], p [N, 3])`, `psi = q7 [N]`, branch signs `sB, sC, sA` (+-1 tensors,
    scalar or `[N]`; module docstring), clip `clip` and the q6 seam `seam` (tensors). `P`
    the DH constants (`BatchedAnalyticMap._panda_params`)."""
    d, a, ca, sa = P["d"], P["a"], P["ca"], P["sa"]
    lo, hi = -clip, clip
    eps = PANDA_SAFE_EPS

    O2O4, O4O6 = P["O2O4"], P["O4O6"]                              # eq 3, 4
    p7 = p - R[:, :, 2] * (d[7] + d[8])                            # eq 5
    ang = -psi + math.pi / 4
    x6 = R[:, :, 0] * torch.cos(ang).unsqueeze(1) + R[:, :, 1] * torch.sin(ang).unsqueeze(1)  # eq 6, 7
    p6 = p7 - a[6] * x6                                            # eq 8
    O2O6_vec = p6 - P["e_z"] * d[0]
    O2O6 = _norm(O2O6_vec)                                         # eq 10
    u0 = (O2O4 ** 2 + O4O6 ** 2 - O2O6 ** 2) / (2 * O2O4 * O4O6)
    O2O4O6 = sA * torch.acos(torch.clamp(u0, lo, hi))
    O2O4O3, q3_add = P["O2O4O3"], P["q3_add"]                      # eq 11, 12
    q4 = O2O4O6 + q3_add - math.pi * (1 + sA)

    u1 = (O2O6 ** 2 + O4O6 ** 2 - O2O4 ** 2) / (2 * O2O6 * O4O6)
    O2O6O4 = sA * torch.acos(torch.clamp(u1, lo, hi))             # eq 14
    O2O6H = O2O6O4 + P["atan_a4d4"]                               # eq 13
    y6 = -R[:, :, 2]
    z6 = _cross(x6, y6)                                           # eq 15
    R6 = torch.stack([x6, y6, z6], dim=2)
    x626, y626 = _dot(x6, O2O6_vec), _dot(y6, O2O6_vec)
    phi6 = torch.atan2(y626, x626)
    u2 = O2O6 * torch.cos(O2O6H) / torch.sqrt(x626 ** 2 + y626 ** 2)
    psi6 = torch.asin(torch.clamp(u2, lo, hi))                    # eq 21

    q6 = HALF_PI * (1 + sB) - sB * psi6 - phi6                    # eq 22, B1 / B2
    q6 = q6 - TWO_PI * torch.floor((q6.detach() - seam) / TWO_PI)

    O3O2O4 = HALF_PI - O2O4O3
    O4O2O6 = math.pi - O2O4O6 - O2O6O4
    PO2O6 = O3O2O4 + O4O2O6
    O2PO6 = O2O6O4 + O2O4O6 + O2O4O3 - O2O6H - HALF_PI           # eq 24
    PO6 = _safe_divide(O2O6 * torch.sin(PO2O6), torch.sin(O2PO6), eps)   # eq 25
    z65 = torch.stack([torch.sin(q6), torch.cos(q6), torch.zeros_like(q6)], dim=1)   # eq 16
    O2P_vec = O2O6_vec - _bmv(PO6.view(-1, 1, 1) * R6, z65)      # eq 26
    x2P, y2P, z2P = O2P_vec[:, 0], O2P_vec[:, 1], O2P_vec[:, 2]
    norm_O2P = _safe_norm(O2P_vec, eps)

    u3 = z2P / _norm(O2P_vec)
    q1 = torch.atan2(sC * y2P, sC * x2P)                          # C1 / C2
    q2 = sC * torch.acos(torch.clamp(u3, lo, hi))                 # eq 28, 29

    cr = _cross(O2P_vec, O2O6_vec)
    y3 = cr / _safe_norm(cr, eps).unsqueeze(1)
    z3 = O2P_vec / norm_O2P.unsqueeze(1)
    x3 = _cross(y3, z3)                                           # eq 30
    R2 = _panda_rot(q1, ca[0], sa[0]) @ _panda_rot(q2, ca[1], sa[1])   # eq 31
    x23 = _bmv(R2.transpose(1, 2), x3)                            # eq 32
    q3 = torch.atan2(x23[:, 2], x23[:, 0])                        # eq 33

    p4 = P["e_z"] * d[0] + d[2] * z3 + a[3] * x3                  # eq 35
    R56 = _panda_rot(q6, ca[5], sa[5])
    R5 = R6 @ R56.transpose(1, 2)
    z5 = R5[:, :, 2]
    HO4_vec = p4 - (p6 - d[4] * z5)                               # eq 34
    O5S = _bmv(R5.transpose(1, 2), HO4_vec)                       # eq 36
    q5 = -torch.atan2(O5S[:, 1], O5S[:, 0])                       # eq 37

    q = torch.stack([q1, q2, q3, q4, q5, q6, psi], dim=1)
    u = torch.stack([u0, u1, u2, u3], dim=1)
    return q, u


## ------------------------------------------------------------------------------------ ##
##                                 the map, from a program                               ##
## ------------------------------------------------------------------------------------ ##

class BatchedAnalyticMap:
    """`X [N, 7] -> (q [N, 7], u [N, 4])` for one analytic program (module docstring).

    Every constant the call reads is a TENSOR built here (the DH constants, the pose offset,
    the clip, the Panda's seam, the branch signs), so a compiled / graphed svgd step that
    reuses a template program's map is rebound to a new program's branch and offset by
    `fused.copy_constants` like any other per-cell constant. `kind` ("panda" | "iiwa") is a
    string, so it is part of the step's structure signature."""

    def __init__(self):
        raise TypeError("use BatchedAnalyticMap.from_program(program, dtype, device)")

    @staticmethod
    def from_program(program, dtype=torch.float64, device="cpu", gc=None):
        """`gc` overrides the program's branch (tests); default `program.gc`, the branch the
        start protocol set (`SetStartFromQ` / `SetNativeStart`) -- fixed per cell."""
        from src.iiwa_analytic_ik import Analytic_IK_7DoF
        from src.panda_analytic_ik import Analytic_IK_Panda, Q6_WRAP_SEAM
        self = BatchedAnalyticMap.__new__(BatchedAnalyticMap)
        self.dtype = dtype
        self.device = torch.device(device)
        kw = dict(dtype=dtype, device=self.device)
        ik = program.analytic_ik
        gc = np.asarray(program.gc if gc is None else gc, dtype=float).reshape(-1)
        if isinstance(ik, Analytic_IK_Panda):
            self.kind = "panda"
            self.P = self._panda_params(ik, kw)
            self.seam = torch.tensor(float(Q6_WRAP_SEAM), **kw)
        elif isinstance(ik, Analytic_IK_7DoF):
            self.kind = "iiwa"
            self.P = self._iiwa_params(ik, kw)
            self.seam = None
            self.alpha = np.asarray(ik.alpha, dtype=float).copy()
            self.d = np.asarray(ik.d, dtype=float).copy()
        else:
            raise NotImplementedError(
                f"{type(program).__name__}: no torch twin of {type(ik).__name__}'s closed form")
        self.signs = self.branch_signs(gc, kw)
        self.clip = torch.tensor(1.0 - float(program.options.analytic_clip_stepback), **kw)
        X = program.pose_offset
        self.R_off = torch.tensor(np.asarray(X.rotation().matrix(), dtype=float), **kw)
        self.p_off = torch.tensor(np.asarray(X.translation(), dtype=float), **kw)
        return self

    def branch_signs(self, gc, kw):
        """The program's `gc` as the three sign tensors the twin multiplies by."""
        gc = np.asarray(gc, dtype=float).reshape(-1)
        if self.kind == "iiwa":
            if gc.size != 3 or not np.all(np.abs(gc) == 1):
                raise ValueError(f"iiwa gc must be three +-1 entries, got {gc}")
            vals = [float(v) for v in gc]
        else:
            if gc.size not in (2, 3):
                raise ValueError(f"Panda gc must have 2 or 3 entries, got {gc}")
            A = float(gc[2]) if gc.size > 2 else 1.0
            if A not in (1.0, -1.0):
                raise ValueError(f"Panda elbow branch A must be +-1, got {A}")
            ## `Solve` takes B1 / C1 iff the entry == 1, the other branch otherwise.
            vals = [1.0 if gc[0] == 1 else -1.0, 1.0 if gc[1] == 1 else -1.0, A]
        return tuple(torch.tensor(v, **kw) for v in vals)

    @staticmethod
    def _iiwa_params(ik, kw):
        alpha = np.asarray(ik.alpha, dtype=float)
        d = np.asarray(ik.d, dtype=float)
        return dict(d_bs=float(d[0]), d_se=float(d[2]), d_ew=float(d[4]), d_wf=float(d[6]),
                    ca=[float(v) for v in np.cos(alpha)], sa=[float(v) for v in np.sin(alpha)],
                    e_z=torch.tensor([0.0, 0.0, 1.0], **kw))

    @staticmethod
    def _panda_params(ik, kw):
        d = [float(v) for v in ik.d]
        a = [float(v) for v in ik.a]
        alpha = np.asarray(ik.alpha, dtype=float)
        O2O4O3 = float(np.arctan2(d[2], a[3]))
        return dict(d=d, a=a, ca=[float(v) for v in np.cos(alpha)],
                    sa=[float(v) for v in np.sin(alpha)],
                    O2O4=math.sqrt(d[2] ** 2 + a[3] ** 2), O4O6=math.sqrt(d[4] ** 2 + a[4] ** 2),
                    O2O4O3=O2O4O3, q3_add=O2O4O3 + float(np.arctan2(d[4], abs(a[4]))),
                    atan_a4d4=float(np.arctan2(-a[4], d[4])),
                    e_z=torch.tensor([0.0, 0.0, 1.0], **kw))

    def pose(self, X: Tensor):
        """`(R, p)` of the map's own frame: `RigidTransform(RollPitchYaw(rpy), xyz) @ X_FA`."""
        R = rpy_to_R(X[:, 3:6])
        return R @ self.R_off, X[:, :3] + _bmv(R, self.p_off.unsqueeze(0).expand(X.shape[0], 3))

    def solve(self, R, p, psi, signs=None):
        """The closed form at `(R, p, psi)` on the map's branch (or `signs`, e.g. `[N]`
        tensors for a batch spanning branches)."""
        s1, s2, s3 = self.signs if signs is None else signs
        if self.kind == "panda":
            return panda_solve(R, p, psi, s1, s2, s3, self.clip, self.seam, self.P)
        return iiwa_solve(R, p, psi, s1, s2, s3, self.clip, self.P)

    def __call__(self, X: Tensor, signs=None):
        R, p = self.pose(X)
        return self.solve(R, p, X[:, 6], signs)

    def psi_of(self, q: Tensor) -> Tensor:
        """The redundancy parameter of each configuration `q [N, 7]`: q7 on the Panda, the
        arm angle on the iiwa (`RecoverPsi`), NaN -> 0 as `SetStartFromQ` does."""
        if self.kind == "panda":
            psi = q[:, 6].clone()
        else:
            psi = iiwa_psi(q, self.alpha, self.d)
        return torch.nan_to_num(psi, nan=0.0)
