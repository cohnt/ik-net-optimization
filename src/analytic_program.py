"""The analytic + optimization formulation, shared by the Panda and the iiwa14.

Decision variables: the pose `xyz_rpy` (6) of the frame the program's task acts on, and the
redundancy parameter `psi` (1). `q = closed-form IK(pose @ X_FA, psi, GC)` with the branch
`GC` fixed per solve -- recovered from the start configuration under the paired protocol,
drawn from the cell's generator under the native one, never searched. This is the
formulation of ../codebase (`IiwaIKProblemNewFormulation`) and of the TRO paper.

It is an UPPER-BOUND REFERENCE column. It is expected to beat both other arms -- that is the
earlier work's result -- and the paper's case is robots that have no closed form, so
learned v joint space stays the comparison of record. Baselines get bug fixes and standard
treatments, no novel research.

What lives here, so the two robots cannot drift apart:

* `MeasureAnalyticFrame` -- `X_FA`, the map's own frame expressed in the frame the variables
  denote. Measured against the plant at several configurations and checked constant, the
  `CalibrateFlowFrame` pattern; it replaces the Panda's hand-typed `X_L7_ANALYTIC` and the
  codebase iiwa's hard-coded 0.135 / 0.185 tool offsets. A base not welded at the world
  origin, or a joint convention that differs from the DH table, makes the offset vary and
  raises.
* The reachability rows. The map clips each arccos/arcsin argument into
  `+-(1 - analytic_clip_stepback)` (zero derivative where active) and these rows hold the
  unclipped arguments inside `+-(1 - analytic_reach_stepback)`, a strictly tighter band, so
  at any point the verifier calls feasible the clip is inactive and `q` is exactly the
  closed form (`CheckClipCannotEscapeGate`). The arguments come out of the SAME pass as `q`
  and are read through the `QAndPose` memo -- the map never runs twice per iterate.
* The pose-target row (pose task) and the mug rows (grasp task) as LINEAR constraints on the
  pose variables: general constraints, never variable bounds, so a paired start outside
  them is not projected (see CLAUDE.md, "a region an initial guess may violate").
"""
import numpy as np
from pydrake.all import (AutoDiffXd, MathematicalProgram, Quaternion, RigidTransform,
                         RollPitchYaw, RollPitchYaw_, RotationMatrix, RigidTransform_)


def CheckClipCannotEscapeGate(options, tol):
    '''Raise unless a point the verifier passes cannot have an active clip.

    A reachability row violated by up to `tol` (the verifier's slack) leaves an argument at
    `1 - reach + tol`; the clip is inactive there only if that is below `1 - clip`. Ported
    from ../codebase/src/common_optimization_utilities.py, which warned; this raises, and is
    called before the grid by every script that fields an analytic arm.'''
    clip, reach = options.analytic_clip_stepback, options.analytic_reach_stepback
    if not reach - clip >= tol:
        raise ValueError(
            f"analytic_reach_stepback - analytic_clip_stepback = {reach - clip:.3g} < the "
            f"verifier's slack {tol:.3g}: a reachability row violated within the slack "
            f"could sit on an active clip, where q is not the closed form")


def MeasureAnalyticFrame(plant, context, var_frame, analytic_fk, pad_q, lower, upper,
                         samples=4, tol=1e-9):
    '''`X_FA`: the analytic map's frame A expressed in the variables' frame F.

    `analytic_fk(q_arm) -> 4x4` is the map's own forward kinematics (base = world). Both
    frames are welded to the last link, so the offset is constant -- measured at `samples`
    configurations from a fixed-seed generator (it must not consume global draws) and
    checked, not assumed.'''
    rng = np.random.default_rng(0)
    offsets = []
    for _ in range(samples):
        q_arm = rng.uniform(lower, upper)
        plant.SetPositions(context, pad_q(q_arm))
        X_WF = var_frame.CalcPoseInWorld(context)
        offsets.append(X_WF.inverse() @ RigidTransform(np.asarray(analytic_fk(q_arm), float)))
    spread = max(np.linalg.norm(offsets[0].translation() - o.translation())
                 + abs((offsets[0].inverse() @ o).rotation().ToAngleAxis().angle())
                 for o in offsets[1:])
    if spread > tol:
        raise RuntimeError(
            f"the analytic frame's offset from {var_frame.name()} is not constant "
            f"(spread {spread:.3e} > {tol:.0e}): the robot base is not at the world origin "
            f"or the plant's joint convention differs from the map's DH table")
    return offsets[0], spread


class AnalyticProgramMixin:
    '''The analytic formulation, mixed in ahead of a robot's learned program class.

    Inheriting the learned base keeps `CalibrateFlowFrame` / `frame_for_flow` intact for
    code that reads them (the svgd batched program, the grasp `X_grasp_ee`). The benchmark
    hands every program the already-loaded flow, so the only cost is the calibration's
    four FK calls; no network is ever evaluated by this formulation.

    A robot provides: `ANALYTIC_IK_CLASS`, `PSI_BOUNDS`, `_AnalyticSolve(R, p, psi, gc,
    clip)`, `AnalyticFK(q_arm)`, `BranchSet()`, `RecoverBranch(q_arm)`, `RecoverPsi(q_arm)`.
    '''

    #: How many pre-clip arguments the map reports (both robots: four).
    NUM_REACH_VALUES = 4
    #: psi's variable bound, symmetric. A bound, not a general constraint: no start violates
    #: it. Never q7's limits on the Panda (psi IS q7 there, and an identical bound beside the
    #: joint-limit row is a duplicated active constraint).
    PSI_BOUNDS = 5.0
    #: The native start's psi draw.
    PSI_NATIVE_RANGE = (-np.pi, np.pi)
    #: The grasp task's xyz_rpy variable bound (no start violates it).
    GRASP_POSE_BOUNDS = 5.0
    #: The native grasp start before 2026-10-12, restored only under legacy_robot_settings
    #: (the Panda's `[mug xyz, 1, 0, 0]`); None where there is no legacy start.
    LEGACY_GRASP_START_RPY = None

    ## ----------------------------- construction ---------------------------- ##

    def _InitAnalytic(self):
        self.analytic_ik = self.ANALYTIC_IK_CLASS()
        self._analytic_offsets = {}
        self._analytic_aux = {}

    def AnalyticPoseOffset(self):
        '''`X_FA` for the current `self.frame`, measured once per frame.'''
        name = self.frame.name()
        if name not in self._analytic_offsets:
            lower, upper = (np.asarray(b, dtype=float)[:self.num_arm_dof]
                            for b in self.ConfigLimits())
            self._analytic_offsets[name] = MeasureAnalyticFrame(
                self.plant, self.plant_context, self.frame, self.AnalyticFK, self.PadQ,
                lower, upper)
        return self._analytic_offsets[name][0]

    def _BuildAnalyticProg(self, q_nominal, pose_offset, gc):
        self.prog = MathematicalProgram()
        self.xyz_rpy = self.prog.NewContinuousVariables(6, "xyz_rpy")
        self.psi = self.prog.NewContinuousVariables(1, "psi")
        self.lumped_vars = np.hstack([self.xyz_rpy, self.psi])
        self.q_nominal = self.ResolveQNominal(q_nominal)
        self.pose_offset_source = "measured" if pose_offset is None else "caller"
        self.pose_offset = self.AnalyticPoseOffset() if pose_offset is None else pose_offset
        ## The branch of the home pose, as ../codebase seeds its branch from the guess;
        ## either start protocol overwrites it. Never the global stream.
        self.gc = (np.asarray(gc) if gc is not None
                   else self.RecoverBranch(self.q_nominal))
        self._analytic_aux = {}

    def create_prog(self, target_pose=np.array([0., 0., 0., 1., 0., 0., 0.]), q_nominal=None,
                    pose_offset=None, gc=None):
        '''The pose task: `xyz_rpy` pinned to the target by a linear equality.'''
        self._BuildAnalyticProg(q_nominal, pose_offset, gc)
        self.target_pose = target_pose
        self.target_rpy = np.array([*target_pose[:3], *RotationMatrix(
            Quaternion(target_pose[3:])).ToRollPitchYaw().vector()])
        self.prog.SetInitialGuess(self.xyz_rpy, self.target_rpy)
        self.prog.SetInitialGuess(self.psi, [self.RecoverPsi(self.q_nominal)])
        self.add_constraints()
        self.add_costs()

    def create_grasp_prog(self, target_mug, q_nominal=None, pose_offset=None, gc=None):
        '''The grasp task: `xyz_rpy` is the `between_fingers` pose, on the mug axis.'''
        self._BuildAnalyticProg(q_nominal, pose_offset, gc)
        self.target_mug = target_mug
        self.prog.SetInitialGuess(self.xyz_rpy, self.AnalyticNativePose())
        self.prog.SetInitialGuess(self.psi, [self.RecoverPsi(self.q_nominal)])
        self.add_constraints()
        self.add_costs()

    def AnalyticNativePose(self):
        '''The pose variables' native start: the target's pose on the pose task; on the grasp
        task the pose of `between_fingers` at the configuration that GENERATED the mug --
        which is `mug.middle` itself, the analytic counterpart of the learned arm's
        `GraspCStart` (PI, 2026-10-08: anchoring on a known-valid grasp is accepted).'''
        if getattr(self, "target_mug", None) is None:
            return np.asarray(self.target_rpy, dtype=float)
        X = self.target_mug.middle
        if self.options.legacy_robot_settings and self.LEGACY_GRASP_START_RPY is not None:
            return np.array([*X.translation(), *self.LEGACY_GRASP_START_RPY])
        return np.concatenate([X.translation(), X.rotation().ToRollPitchYaw().vector()])

    ## ------------------------------ the map -------------------------------- ##

    def _Solve(self, vars, clip_stepback):
        xyz_rpy, psi = vars[:6], vars[6]
        t = AutoDiffXd if isinstance(xyz_rpy[0], AutoDiffXd) else float
        X = RigidTransform_[t](RollPitchYaw_[t](xyz_rpy[3:6]), xyz_rpy[:3])
        R = X.rotation().matrix()
        p = X.translation()
        R_off = self.pose_offset.rotation().matrix()
        return self._AnalyticSolve(R @ R_off, p + R @ self.pose_offset.translation(),
                                   psi, self.gc, clip_stepback)

    def VarsToConfigAndQ(self, vars):
        '''One closed-form pass: the plant positions, with the four pre-clip arguments
        stashed under the iterate's memo key for `ReachabilityValues`.'''
        q_arm, u = self._Solve(vars, self.options.analytic_clip_stepback)
        ad = isinstance(q_arm[0], AutoDiffXd)
        q = np.zeros(self.num_pos, dtype=AutoDiffXd if ad else float)
        q[:self.num_arm_dof] = q_arm
        q[self.num_arm_dof:] = 0.04  # fixed gripper joints, as PadQ
        aux = self._analytic_aux
        aux[self._FlowCacheKey(vars)] = u
        while len(aux) > 8:
            aux.pop(next(iter(aux)))
        return q, q

    def VarsToQ(self, vars, add_correction=False):
        return self.VarsToConfigAndQ(vars)[1]

    def ReachabilityValues(self, vars):
        '''The unclipped arguments at an iterate, from the pass `QAndPose` already made.'''
        self.QAndPose(vars)
        u = self._analytic_aux.get(self._FlowCacheKey(vars))
        if u is None:  # evicted between the two calls; recompute rather than guess
            self.VarsToConfigAndQ(vars)
            u = self._analytic_aux[self._FlowCacheKey(vars)]
        return u

    ## ---------------------------- constraints ------------------------------ ##

    def add_constraints(self):
        if self.options.collision_avoidance:
            self.CreateCollisionFreeConstraint()
        if self.options.joint_limits:
            self.CreateJointLimitsConstraint()
        self.ApplyConstraints()
        if getattr(self, "target_mug", None) is not None:
            self.MugAxisConstraint()
        else:
            self.PoseTargetConstraint()
        self.ReachabilityConstraint()
        self.BoundingBoxConstraint()

    def PoseTargetConstraint(self):
        '''A linear EQUALITY pinning the pose variables to the target, as ../codebase's new
        formulation pins them. Not a box (a `+-ik_constraint_tol` box once handed this arm
        +-0.01 rad of free orientation), and not a variable bound (IPOPT's bound_push would
        then project the paired start onto the target -- a median 2.7 rad head start).'''
        self.xyz_rpy_box = (self.target_rpy.copy(), self.target_rpy.copy())
        self.pose_target_constraint = self.prog.AddLinearEqualityConstraint(
            np.eye(6), self.target_rpy, self.xyz_rpy)
        self.pose_target_constraint.evaluator().set_description("PoseTargetConstraint")
        return self.pose_target_constraint

    def MugAxisConstraint(self):
        '''The grasp point on the mug's axis (x = y = 0 exactly) and within its height.

        Affine in the position variables, so written as the linear constraint it is
        (../codebase's MugConstraint form, minus the homogeneous row, which is identically 1
        with a zero gradient and costs LICQ a rank).'''
        M_inv = self.target_mug.middle.inverse()
        A = M_inv.rotation().matrix()
        b = M_inv.translation()
        h = self.options.mug_height
        lb = np.array([0.0, 0.0, -h]) - b
        ub = np.array([0.0, 0.0, h]) - b
        self.ik_constraint = self.prog.AddLinearConstraint(A, lb, ub, self.xyz_rpy[:3])
        self.ik_constraint.evaluator().set_description("IKMugConstraint")
        return self.ik_constraint

    def ReachabilityConstraint(self):
        bound = 1.0 - self.options.analytic_reach_stepback
        n = self.NUM_REACH_VALUES
        self.reachability_constraint = self.prog.AddConstraint(
            func=self.ReachabilityValues, lb=-bound * np.ones(n), ub=bound * np.ones(n),
            vars=self.lumped_vars, description="ReachabilityConstraint")
        return self.reachability_constraint

    def BoundingBoxConstraint(self):
        self.bounding_box_constraint_psi = self.prog.AddBoundingBoxConstraint(
            -self.PSI_BOUNDS, self.PSI_BOUNDS, self.psi)
        self.bounding_box_constraint_psi.evaluator().set_description("PsiBoundingBox")
        if getattr(self, "target_mug", None) is not None:
            b = self.GRASP_POSE_BOUNDS
            self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
                -b * np.ones(6), b * np.ones(6), self.xyz_rpy)
            self.bounding_box_constraint.evaluator().set_description("PoseBoundingBox")

    ## ------------------------------- starts -------------------------------- ##

    def SetStartFromQ(self, q_arm):
        '''The paired start: `q_arm` in this arm's own variables, by inversion -- its branch,
        the pose of `self.frame` (the frame the variables denote) and its psi -- set
        UNCLIPPED. Returns the distance the start sits outside the variables' regions; on
        the pose task the pose region is a general constraint, so that term is diagnostic
        only (nothing projects it), while psi's and the grasp pose's are bounds.'''
        q_arm = np.asarray(q_arm, dtype=float)[:self.num_arm_dof]
        self.gc = self.RecoverBranch(q_arm)
        self.plant.SetPositions(self.plant_context, self.PadQ(q_arm))
        pose = self.frame.CalcPoseInWorld(self.plant_context)
        xyz_rpy = np.concatenate([pose.translation(), pose.rotation().ToRollPitchYaw().vector()])
        psi = np.nan_to_num(self.RecoverPsi(q_arm))
        box = getattr(self, "xyz_rpy_box", None)
        dist = (float(np.linalg.norm(np.clip(xyz_rpy, *box) - xyz_rpy)) if box is not None
                else self._BoxDistance(self.xyz_rpy, xyz_rpy))
        self.prog.SetInitialGuess(self.xyz_rpy, xyz_rpy)
        return dist + self._SetClipped(self.psi, [psi])

    def SetNativeStart(self, q_init, rng):
        '''The formulation's own initialisation, sampled, never searched: the branch uniformly
        over the chart's set, then psi uniformly over `PSI_NATIVE_RANGE` (in that order, so
        a cell's draws reproduce), the pose at `AnalyticNativePose()`.'''
        branches = self.BranchSet()
        self.gc = branches[rng.integers(len(branches))]
        self.prog.SetInitialGuess(self.psi, [rng.uniform(*self.PSI_NATIVE_RANGE)])
        self.prog.SetInitialGuess(self.xyz_rpy, self.AnalyticNativePose())
        return 0.0

    def UnclippedStartError(self, q_init):
        '''How far the CLOSED FORM -- clip relaxed to the arccos domain itself -- sits from
        `q_init` at the current initial guess.

        The paired harness scores a start that cannot be represented as an immediate
        failure. A start inside the clip band IS represented by the variables (they are
        exactly `q_init`'s pose, psi and branch); only the clipped map returns a nearby
        configuration. That start is an infeasible start on the reachability rows -- a
        general constraint, which the solver walks off -- and ../codebase solves it as such.
        This number tells the two apart: ~1e-12 for a clip-band start, radians for one the
        branch genuinely cannot reproduce (the 4-branch Panda chart's elbow half).'''
        x = np.asarray(self.prog.GetInitialGuess(self.lumped_vars), dtype=float)
        q_arm, _ = self._Solve(x, 0.0)
        diff = np.asarray(q_arm, dtype=float) - np.asarray(q_init, dtype=float)[:self.num_arm_dof]
        return float(np.max(np.abs(np.angle(np.exp(1j * diff)))))

    def RobotSettings(self):
        out = super().RobotSettings()
        X = self.pose_offset
        out.update(
            formulation="analytic",
            gc=[float(v) for v in np.ravel(self.gc)],
            branch_set_size=int(len(self.BranchSet())),
            psi_bounds=[-float(self.PSI_BOUNDS), float(self.PSI_BOUNDS)],
            psi_native_range=[float(v) for v in self.PSI_NATIVE_RANGE],
            pose_offset_xyz_rpy=[*map(float, X.translation()),
                                 *map(float, X.rotation().ToRollPitchYaw().vector())],
            pose_offset_source=self.pose_offset_source,
            analytic_clip_stepback=float(self.options.analytic_clip_stepback),
            analytic_reach_stepback=float(self.options.analytic_reach_stepback))
        if getattr(self, "target_mug", None) is not None:
            out["analytic_grasp_start"] = [float(v) for v in self.AnalyticNativePose()]
        return out
