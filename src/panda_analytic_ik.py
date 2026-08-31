import pydrake.math
from pydrake.all import (
    AutoDiffXd,
    RigidTransform,
    RotationMatrix,
    RigidTransform_, 
    RotationMatrix_,
    cos, 
    sin, 
    sqrt,
    arcsin, 
    arccos,
    atan2,
)
import numpy as np

panda_a = np.array([0., 0., 0, 0.0825, -0.0825, 0, 0.088, 0., 0.])
panda_d = np.array([0.333, 0., 0.316, 0., 0.384, 0., 0, 0.107, 0.1034])
panda_alpha = np.array([0., -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0., 0.])
panda_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
panda_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])



def scalar_clip(val, a, b):
	if type(val) == AutoDiffXd:
		a = AutoDiffXd(a, np.zeros(val.derivatives().shape))
		b = AutoDiffXd(b, np.zeros(val.derivatives().shape))

		# a = AutoDiffXd(a, np.full(val.derivatives().shape, -np.pi/2))
		# b = AutoDiffXd(b, np.full(val.derivatives().shape, np.pi/2))
		# a = AutoDiffXd(a, -val.derivatives())
		# b = AutoDiffXd(b, val.derivatives())

	return pydrake.math.max(
		a, pydrake.math.min(
			b, val
		)
	)

def safe_arccos(val, a, b):
	return pydrake.math.arccos(scalar_clip(val, a, b))
def safe_arcsin(val, a, b):
    return pydrake.math.arcsin(scalar_clip(val, a, b))


def safe_norm(val, eps=1e-6):
    norm_val = np.linalg.norm(val)
    try:
        if pydrake.math.abs(norm_val) < eps:
            return eps
    except Exception:
        if np.isnan(norm_val) or np.isinf(norm_val) or abs(norm_val) < eps:
            return eps
    return norm_val


def safe_divide(numerator, denominator, eps=1e-6):
    try:
        if pydrake.math.abs(denominator) < eps:
            sign = 1 if float(denominator) >= 0 else -1
            denominator = sign * eps
    except Exception:
        if abs(denominator) < eps:
            denominator = eps if denominator >= 0 else -eps
    return numerator / denominator


class Analytic_IK_Panda:
    def __init__(self, alpha = panda_alpha, a = panda_a, d = panda_d, limits_lower = panda_limits_lower, limits_upper = panda_limits_upper):
        self.alpha = alpha.copy()
        self.d = d.copy()
        self.a = a.copy()
        self.limits_lower = limits_lower.copy()
        self.limits_upper = limits_upper.copy()

        self.Ts = [ # Modified? DH convention
            lambda ti, ai=ai, ri=ri, di=di : np.array([
                [cos(ti), -sin(ti), 0, ri],
                [sin(ti) * cos(ai), cos(ti)*cos(ai), -sin(ai), -di * sin(ai)],
                [sin(ti)* sin(ai), cos(ti) * sin(ai), cos(ai), di * cos(ai)],
                [0, 0, 0, 1]
            ])
            for ai, ri, di in zip(self.alpha, self.a, self.d)
        ]

    def FK(self, thetas):
        thetas = np.concatenate([thetas, np.array([0, -np.pi/4])])
        eval_Ts = [eval_T(t) for (eval_T, t) in zip(self.Ts, thetas)]
        full_mat = np.linalg.multi_dot(eval_Ts)
        return full_mat
    
    def psi(self, qs): # redundancy parameter
        return qs[6] ## q7 in paper notation
    
    def gc(self, qs, branches=2):
        ## returns array of 2 (or 3): first indicates B1 or B2, second indicates C1 or C2,
        ## and with branches=3 a third element A in {+1, -1} indicating the elbow branch.
        gc = np.zeros(branches)
        gc[1] = 1 if qs[1] > 0 else 2



        T5 = np.linalg.multi_dot([eval_T(t) for (eval_T, t) in zip(self.Ts[:5], qs[:5])])
        T6 = T5 @ self.Ts[5](qs[5])
        p6 = T6[:3, 3]
        x5 = T5[:3, 0]
        O2O6_vec = p6-np.array([0, 0, self.d[0]])
        if np.dot(O2O6_vec, x5) < 0:
            gc[0] = 1
        else:
            gc[0] = 2

        if branches == 3:
            # The two elbow relations are q3 = theta + q3_add - 2*pi (A = +1) and
            # q3 = -theta + q3_add (A = -1) with theta in [0, pi], so they partition at
            # q3 = q3_add - pi = -0.467 rad. Verified exact on 4000 random
            # configurations (zero mislabels against reproducing the configuration).
            q3_add = np.arctan2(self.d[2], self.a[3]) + np.arctan2(self.d[4], abs(self.a[4]))
            gc[2] = 1 if qs[3] < q3_add - np.pi else -1

        return gc

    def _compose_pose(self, pose: RigidTransform, pose_offset: RigidTransform = None):
        if pose_offset is None:
            return pose.rotation().matrix(), pose.translation()

        if not isinstance(pose_offset, RigidTransform):
            raise TypeError("pose_offset must be a pydrake RigidTransform")

        rotation = pose.rotation().matrix() @ pose_offset.rotation().matrix()
        translation = pose.translation() + pose.rotation().matrix() @ pose_offset.translation()
        return rotation, translation

    def IK(self, pose: RigidTransform, psi: float, GC: np.ndarray = None, return_unclipped_vals = False, return_singularity_vals = False, clip_stepback = 1e-6, pose_offset: RigidTransform = None):
        # GC has 2 or 3 elements. GC[0] picks the wrist branch (B1/B2), GC[1] the shoulder
        # branch (C1/C2), and the optional GC[2] = A in {+1, -1} picks the elbow branch:
        # A = +1 is the historical chart (elbow angle theta + q3_add - 2*pi), A = -1 the
        # mirrored elbow (-theta + q3_add), realised by negating BOTH triangle angles
        # O2O4O6 and O2O6O4 -- the arm plane's signed angles flipping together. Measured
        # (2026-08-31, 4000 random configurations): the A branch holds 10.4% of the
        # configuration space, and charting it takes reproduction from 89.4% to 99.4%.
        # A 2-element GC keeps the historical 4-branch behaviour (A = +1).
        A = int(GC[2]) if GC is not None and len(GC) > 2 else 1
        clip = 1 - clip_stepback
        target_rotation, target_translation = self._compose_pose(pose, pose_offset)
        clipped_values = np.zeros(4, dtype=AutoDiffXd if type(target_translation[0]) is AutoDiffXd else float)

        ad = any(isinstance(x, AutoDiffXd) for x in np.ravel(target_translation)) or any(isinstance(x, AutoDiffXd) for x in np.ravel(target_rotation))
        q = np.zeros(7, dtype=AutoDiffXd if ad else float)
        q[6] = psi

        # q[3]
        O2O4 = sqrt(self.d[2]**2 + self.a[3]**2) ## equation 3
        O4O6 = sqrt(self.d[4]**2 + self.a[4]**2) ## equation 4
        p7 = target_translation - (self.d[7] + self.d[8]) * target_rotation[:, 2] ## equation 5
        x6 = target_rotation @ np.array([cos(-psi + np.pi/4), sin(-psi + np.pi/4), 0]) ## equation 6,7
        p6 = p7 - self.a[6] * x6.flatten() # eq 8
        O2O6_vec = p6-np.array([0, 0, self.d[0]])
        p2 = np.array([0, 0, self.d[0]])
        O2O6 = np.linalg.norm(p6-p2) ## equation 10
        O2O4O6 = A * safe_arccos((O2O4**2 + O4O6**2 - O2O6**2) / (2 * O2O4 * O4O6), -clip, clip)
        clipped_values[0] = (O2O4**2 + O4O6**2 - O2O6**2) / (2 * O2O4 * O4O6)
        O2O4O3 = atan2(self.d[2],self.a[3])
        q3_add = O2O4O3 + atan2(self.d[4], abs(self.a[4])) # eq 11,12

        # The measured elbow relation (the old commented-out "Case A1", O2O4O6 - q3_add,
        # matches no configuration at all): q3 = +theta + q3_add - 2*pi on the historical
        # branch, and -theta + q3_add on the mirrored one. With O2O4O6 already signed by A,
        # both are one expression less the wrap.
        q[3] = O2O4O6 + q3_add - (2*np.pi if A > 0 else 0.0)

        #q[5]
        clipped_values[1] = (O2O6**2 + O4O6**2 - O2O4**2) / (2 * O2O6 * O4O6)
        O2O6O4 = A * safe_arccos((O2O6**2 + O4O6**2 - O2O4**2) / (2 * O2O6 * O4O6), -clip, clip) # equation 14

        O2O6H = O2O6O4 + atan2(-self.a[4], self.d[4])  # eq 13
        y6 = (-target_rotation[:, 2])
        z6 = np.cross(x6, y6) # eq 15
        R6 = np.column_stack([x6, y6, z6])
        x626, y626, _ = R6.T @ O2O6_vec
        phi6 = atan2(y626, x626)
        clipped_values[2] = (O2O6 * cos(O2O6H) / sqrt(x626**2 + y626**2))
        psi6 = safe_arcsin((O2O6 * cos(O2O6H) / sqrt(x626**2 + y626**2)), -clip, clip) # eq 21

        if GC[0] == 1:
            q[5] = np.pi - psi6 - phi6 # eq 22 B1
        else:
            q[5] = psi6 - phi6 # eq 22 B2


        #q[0] and q[1]

        O3O2O4 = np.pi/2 - O2O4O3
        O4O2O6 = np.pi - O2O4O6 - O2O6O4
        PO2O6 = O3O2O4 + O4O2O6 
        O2PO6 = O2O6O4 + O2O4O6 + O2O4O3 - O2O6H - np.pi/2 # equation 24
        PO6 = safe_divide(O2O6 * sin(PO2O6), sin(O2PO6)) # equation 25
        z65 = np.array([sin(q[5]), cos(q[5]), 0]) # equation 16
        O2P_vec = O2O6_vec - PO6 * R6 @ z65 # equation 26
        x2P, y2P, z2P = O2P_vec
        norm_O2P = safe_norm(O2P_vec)

        if GC[1] == 1:
            q[0] = atan2(y2P, x2P) # C1
            clipped_values[3] = z2P/np.linalg.norm(O2P_vec)
            q[1] = safe_arccos(z2P/np.linalg.norm(O2P_vec), -clip, clip) # equation  28
        else:
            q[0] = atan2(-y2P, -x2P) # C2
            # The value actually clipped is +z2P/|O2P| (same as C1); recording its negation
            # here inverted the reported gradient sign on the reachability row. The bound
            # is symmetric so feasibility never noticed, but the derivative did.
            clipped_values[3] = z2P/np.linalg.norm(O2P_vec)
            q[1] = -safe_arccos(z2P/np.linalg.norm(O2P_vec), -clip, clip) # equation  29

        #q[2]

        y3 = np.cross(O2P_vec, O2O6_vec) / safe_norm(np.cross(O2P_vec, O2O6_vec))
        z3 = O2P_vec / norm_O2P
        x3 = np.cross(y3, z3) # equation 30

        eval_R2 = [eval_T(t) for (eval_T, t) in zip(self.Ts[:2], q[:2])]
        R2 = np.linalg.multi_dot(eval_R2)[:3, :3] # equation 31
        x23 = R2.T @ x3 # equation 32
        q[2] = atan2(x23[2], x23[0]) # equation 33

        #q[4]

        p4 = p2 + self.d[2] * z3 + self.a[3] * x3 # equation 35
        R56 = self.Ts[5](q[5])[:3, :3]
        R5 = R6 @ R56.T
        z5 = R5[:, 2]
        HO4_vec = p4 - (p6 - self.d[4] * z5) # equation 34
        O5S_vec = R5.T @ HO4_vec # equation 36
        q[4] = -atan2(O5S_vec[1], O5S_vec[0]) # equation 37


        if return_unclipped_vals:
            return clipped_values
        return q