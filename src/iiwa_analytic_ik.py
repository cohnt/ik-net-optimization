## Closed-form forward and inverse kinematics of the KUKA LBR iiwa 14 R820, an S-R-S arm.
##
## Follows Faria et al., "Position-based kinematics for 7-DoF serial manipulators with
## global configuration control, joint limit and singularity avoidance": the configuration
## is parameterised by the pose of DH frame 7, the ARM ANGLE `psi` (the angle of the
## shoulder-elbow-wrist plane against the plane of a virtual arm with theta_3 = 0) and the
## global configuration `GC = (sign q2, sign q4, sign q6)`, eight branches in all.
##
## Ported 2026-10-12 from ../codebase/src/iiwa_analytic_ik.py, the maintained copy (this
## repo's own was deleted unreached in 3874e48, and that deleted copy returned the clip
## arguments after computing only the first). Two deliberate changes: `Solve` returns the
## configuration AND the four pre-clip arccos arguments in one pass, so a reachability row
## never re-runs the map; and the gripper helpers with their hard-coded 0.135 / 0.185 tool
## offsets are gone -- the offset from DH frame 7 to whatever frame a program's variables
## denote is MEASURED against the plant (`src/analytic_program.py:MeasureAnalyticFrame`).
##
## Everything is written on `pydrake.math`, so one code path serves float and AutoDiffXd.

import numpy as np
import pydrake.math
from pydrake.all import AutoDiffXd

# NOTE: the LBR iiwa 14 R820, so slightly different from the report at
# https://zenodo.org/record/4063575. The last entry puts DH frame 7 at the origin of
# Drake's iiwa_link_7 (0.081 m past the wrist centre).
iiwa_alpha = np.array([-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0])
iiwa_d = np.array([0.36, 0, 0.42, 0, 0.4, 0, 0.126 - 0.045])

## Hand-typed limits. The plant's own differ by 1e-6; every program bounds against the
## plant (`ConfigLimits`), and these survive as the `legacy_robot_settings` control and for
## the older scripts that import them.
iiwa_limits_lower = np.array([
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-3.054326
])
iiwa_limits_upper = np.array([
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	3.054326
])

#: How many pre-clip arguments `Solve` reports: the elbow (EQ 4), the virtual shoulder
#: (EQ 7), q2 (EQ 18) and q6 (EQ 23) arccos inputs.
NUM_REACH_VALUES = 4


def cross_product_matrix(a):
    return np.cross(a, np.identity(a.shape[0]) * -1)


def scalar_clip(val, a, b):
    '''Clip with zero derivative where the clip is active (the bounds carry none).'''
    if type(val) == AutoDiffXd:
        a = AutoDiffXd(a, np.zeros(val.derivatives().shape))
        b = AutoDiffXd(b, np.zeros(val.derivatives().shape))
    return pydrake.math.max(a, pydrake.math.min(b, val))


def safe_arccos(val, a, b):
    return pydrake.math.arccos(scalar_clip(val, a, b))


def _is_autodiff(*arrays):
    return any(isinstance(x, AutoDiffXd) for arr in arrays for x in np.ravel(arr))


def ComposePose(R, p, pose_offset):
    '''`(R, p) @ pose_offset` on plain arrays, so a float offset composes with an
    AutoDiffXd pose without converting either.'''
    if pose_offset is None:
        return R, p
    R_off = pose_offset.rotation().matrix()
    p_off = pose_offset.translation()
    return R @ R_off, p + R @ p_off


class Analytic_IK_7DoF():
    '''Analytic forward and inverse kinematics of an S-R-S 7-DoF manipulator.'''

    def __init__(self, alpha=iiwa_alpha, d=iiwa_d, limits_lower=iiwa_limits_lower,
                 limits_upper=iiwa_limits_upper):
        # alpha and d are the DH parameters; every other one is zero.
        assert len(alpha) == len(d) == len(limits_lower) == len(limits_upper) == 7
        assert d[1] == d[3] == d[5] == 0
        self.alpha = alpha.copy()
        self.d = d.copy()
        self.d_bs, self.d_se, self.d_ew, self.d_wf = d[0], d[2], d[4], d[6]
        self.limits_lower = limits_lower.copy()
        self.limits_upper = limits_upper.copy()

        self.Ts = [
            lambda ti, ai=ai, di=di: np.array([
                [pydrake.math.cos(ti), -pydrake.math.sin(ti)*np.cos(ai), pydrake.math.sin(ti)*np.sin(ai), 0],
                [pydrake.math.sin(ti), pydrake.math.cos(ti)*np.cos(ai), -pydrake.math.cos(ti)*np.sin(ai), 0],
                [0, np.sin(ai), np.cos(ai), di],
                [0, 0, 0, 1]
            ])
            for ai, di in zip(self.alpha, self.d)
        ]

    def FK(self, thetas):
        '''X_B7, DH frame 7 in the base frame, as a 4x4 array.'''
        return np.linalg.multi_dot([T(t) for T, t in zip(self.Ts, thetas)])

    def GC(self, thetas):
        '''The global configuration (sign q2, sign q4, sign q6), each +-1.'''
        return np.array([1 if thetas[k] >= 0 else -1 for k in (1, 3, 5)])

    def BranchSet(self):
        '''All eight global configurations.'''
        return np.array([(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)])

    def psi(self, thetas):
        '''The arm angle of a configuration (float only): the signed angle from the virtual
        arm's shoulder-elbow-wrist plane to the actual one.'''
        T_07 = self.FK(thetas)
        p_07 = T_07[:-1, -1]
        R_07 = T_07[:-1, :-1]
        GC2, GC4, GC6 = self.GC(thetas)

        p_02 = np.array([0, 0, self.d_bs])
        p_67 = np.array([0, 0, self.d_wf])

        p_26 = p_07 - p_02 - (R_07 @ p_67)
        arccos_in = (np.dot(p_26, p_26) - self.d_se**2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
        theta_4v = GC4 * np.arccos(np.clip(arccos_in, -1, 1))

        R_01 = self.Ts[0](thetas[0])[0:3, 0:3]
        cross = np.cross(p_26, R_01[:, -1])
        theta_1v = np.arctan2(p_26[1], p_26[0]) if np.dot(cross, cross) > 0 else 0

        arccos_in = (self.d_se**2 + np.dot(p_26, p_26) - self.d_ew**2) / (2 * self.d_se * np.linalg.norm(p_26))
        phi = np.arccos(np.clip(arccos_in, -1, 1))
        theta_2v = np.arctan2(np.linalg.norm(p_26[:2]), p_26[2]) + (GC4 * phi)

        theta_vs = [theta_1v, theta_2v, 0, theta_4v]
        T_vs = [T(t) for T, t in zip(self.Ts[:4], theta_vs)]
        p_02_v = np.linalg.multi_dot(T_vs[0:2])[:-1, -1]
        p_04_v = np.linalg.multi_dot(T_vs[0:4])[:-1, -1]

        Ts = [T(t) for T, t in zip(self.Ts, thetas)]
        p_04 = np.linalg.multi_dot(Ts[0:4])[:-1, -1]
        p_06 = np.linalg.multi_dot(Ts[0:6])[:-1, -1]
        p_06_v = p_06

        def unit(v):
            return v / np.linalg.norm(v)
        v_sew_v_hat = unit(np.cross(unit(p_04_v - p_02_v), unit(p_06_v - p_02_v)))
        v_sew_hat = unit(np.cross(unit(p_04 - p_02), unit(p_06 - p_02)))

        sg_psi = np.sign(np.dot(np.cross(v_sew_v_hat, v_sew_hat), p_26))
        return float(sg_psi * np.arccos(np.clip(np.dot(v_sew_v_hat, v_sew_hat), -1, 1)))

    def Solve(self, R_07, p_07, GC, psi, clip_stepback=1e-6, pose_offset=None):
        '''`(q[7], u[4])` for the frame pose `(R_07, p_07) @ pose_offset`, one pass.

        `u` are the four arccos arguments BEFORE clipping; the configuration is computed
        from them clipped into `+-(1 - clip_stepback)`, with zero derivative where the clip
        is active, so a pose out of reach returns a finite but wrong configuration and only
        a constraint on `u` can tell.'''
        R_07, p_07 = ComposePose(np.asarray(R_07), np.asarray(p_07), pose_offset)
        clip = 1 - clip_stepback
        ad = _is_autodiff(R_07, p_07, [psi])
        dtype = AutoDiffXd if ad else float
        thetas = np.zeros(7, dtype=dtype)
        unclipped = np.zeros(NUM_REACH_VALUES, dtype=dtype)

        GC2, GC4, GC6 = GC
        sin_psi, cos_psi = pydrake.math.sin(psi), pydrake.math.cos(psi)

        p_02 = np.array([0, 0, self.d_bs])
        p_67 = np.array([0, 0, self.d_wf])
        p_26 = p_07 - p_02 - (R_07 @ p_67)  # EQ (3)
        norm_p_26 = np.linalg.norm(p_26)
        p_26_hat = p_26 / norm_p_26

        # EQ (4)
        unclipped[0] = (np.dot(p_26, p_26) - self.d_se**2 - self.d_ew**2) / (2 * self.d_se * self.d_ew)
        theta_4v = GC4 * safe_arccos(unclipped[0], -clip, clip)
        thetas[3] = theta_4v

        theta_1v = pydrake.math.atan2(p_26[1], p_26[0])  # EQ (5)

        # EQ (7)
        unclipped[1] = (self.d_se**2 + np.dot(p_26, p_26) - self.d_ew**2) / (2 * self.d_se * norm_p_26)
        phi = safe_arccos(unclipped[1], -clip, clip)
        theta_2v = pydrake.math.atan2(np.linalg.norm(p_26[:2]), p_26[2]) + (GC4 * phi)

        T_vs = [T(t) for T, t in zip(self.Ts[:4], [theta_1v, theta_2v, 0, theta_4v])]
        R_03_v = np.linalg.multi_dot(T_vs[0:3])[:-1, :-1]

        # EQ (15)
        cprod_p_26 = cross_product_matrix(p_26_hat)
        A_s = cprod_p_26 @ R_03_v
        B_s = -1 * cprod_p_26 @ cprod_p_26 @ R_03_v
        C_s = np.outer(p_26_hat, p_26_hat) @ R_03_v

        def s(M, i, j):
            return M[0][i, j] * sin_psi + M[1][i, j] * cos_psi + M[2][i, j]
        S = (A_s, B_s, C_s)

        # EQ (17)-(19)
        thetas[0] = pydrake.math.atan2(GC2 * s(S, 1, 1), GC2 * s(S, 0, 1))
        unclipped[2] = s(S, 2, 1)
        thetas[1] = GC2 * safe_arccos(unclipped[2], -clip, clip)
        thetas[2] = pydrake.math.atan2(-GC2 * s(S, 2, 2), -GC2 * s(S, 2, 0))

        # EQ (20)
        R_34 = T_vs[3][:-1, :-1]
        W = (R_34.T @ A_s.T @ R_07, R_34.T @ B_s.T @ R_07, R_34.T @ C_s.T @ R_07)

        # EQ (22)-(24)
        thetas[4] = pydrake.math.atan2(GC6 * s(W, 1, 2), GC6 * s(W, 0, 2))
        unclipped[3] = s(W, 2, 2)
        thetas[5] = GC6 * safe_arccos(unclipped[3], -clip, clip)
        thetas[6] = pydrake.math.atan2(GC6 * s(W, 2, 1), -GC6 * s(W, 2, 0))

        return thetas, unclipped

    def IK(self, rigid_transform, GC, psi, return_unclipped_vals=False, clip_stepback=1e-6):
        '''../codebase's signature, kept for parity: the configuration, or the four
        pre-clip arguments.'''
        X = rigid_transform.GetAsMatrix4()
        q, u = self.Solve(X[:3, :3], X[:3, 3], GC, psi, clip_stepback)
        return u if return_unclipped_vals else q
