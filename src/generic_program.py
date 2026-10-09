import os
import time

import pydrake.math
import torch
from ikflow.config import DEVICE
import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field, fields as dataclass_fields
from functools import lru_cache, partial
from pydrake.all import (
    AutoDiffXd,
    IpoptSolver,
    NloptSolver,
    SnoptSolver,
    SolverOptions,
    CommonSolverOption,
    MinimumDistanceLowerBoundConstraint,
    Quaternion,
    RigidTransform,
    RigidTransform_,
    RotationMatrix,
    RotationMatrix_,
    RollPitchYaw,
    RollPitchYaw_,
    Quaternion_,
)


# ------------------------- the NLopt option surface, per machine -------------------------
#
# THREE NLopt option surfaces are live in this project at once. The cluster runs the official
# Drake 1.56.0 tarball, whose NloptSolver declares exactly six option names (`algorithm`,
# `constraint_tol`, `xtol_rel`, `xtol_abs`, `max_eval`, `max_time`). This workstation's source
# build declares eleven -- those six plus five `local_optimizer_*`. A 2026-09-18 nightly
# declares sixteen, adding `ftol_rel`, `ftol_abs`, `stopval` and two `local_optimizer_ftol_*`.
# So "it ran locally" proves nothing about the other two, in either direction.
#
# And the failure is NOT a crash. Drake accepts an unknown NLopt name at SetOption time and
# raises only from inside Solve --
#
#     RuntimeError: NLopt: the following solver option names were not recognized: ftol_rel
#
# -- which lands in benchmark.run_grid's per-cell `except Exception` and is recorded as
# fail_reason="error". One post-1.56 key emitted against the cluster's Drake therefore does
# not stop a run; it returns a FULL COLUMN of instant failures, which is exactly the shape
# `--set correction_cost_weight=10.0` on a numerical arm already cost this repo (0 of 480 at
# ~10 ms a cell). `_abort_on_dead_arm` catches it on the third cell, but that is a backstop,
# not a gate: the configuration has to be refused before the first cell.
#
# Detection is by ACCESSOR, never by a hardcoded string. Drake is the authority both on
# whether an option exists and on how it is spelled, and the static accessor answers both at
# once. The list is explicit rather than scraped from dir(NloptSolver) because `SolverName` is
# an INSTANCE method on SolverInterface and calling it unbound raises TypeError.
NLOPT_ACCESSORS = (
    # Drake 1.56.0's six -- the cluster's whole surface, emitted unconditionally.
    "AlgorithmName", "ConstraintToleranceName", "XRelativeToleranceName",
    "XAbsoluteToleranceName", "MaxEvalName", "MaxTimeName",
    # Everything after 1.56.0, emitted only when the matching field is set.
    "FRelativeToleranceName", "FAbsoluteToleranceName", "StopValName",
    "LocalOptimizerAlgorithmName", "LocalOptimizerXRelativeToleranceName",
    "LocalOptimizerXAbsoluteToleranceName", "LocalOptimizerFRelativeToleranceName",
    "LocalOptimizerFAbsoluteToleranceName", "LocalOptimizerMaxEvalName",
    "LocalOptimizerMaxTimeName",
)


@lru_cache(maxsize=1)
def NloptOptionSurface():
    """{accessor name: option string} for the NLopt options THIS Drake accepts.

    A property of the installed Drake rather than of a call site, so it is computed once. It
    reads class attributes only -- no solver is constructed and no program is touched.
    """
    return {name: getattr(NloptSolver, name)()
            for name in NLOPT_ACCESSORS if hasattr(NloptSolver, name)}


# Each post-1.56.0 option: the accessor naming it, the type Drake demands (it routes by type,
# so an int handed to a double-valued option reaches the wrong setter and raises -- the same
# trap the SNOPT casts exist for), and whether it is an INNER-solver option, meaning one Drake
# reads unconditionally but applies only inside
# `if (!parsed_options.local_optimizer_algorithm.empty())` (nlopt_solver.cc:546-564).
# `local_optimizer_algorithm` is the gate itself, so it is not marked inner.
NloptOption = namedtuple("NloptOption", "accessor cast inner")
NLOPT_POST_1_56_OPTIONS = {
    "nlopt_ftol_rel": NloptOption("FRelativeToleranceName", float, False),
    "nlopt_ftol_abs": NloptOption("FAbsoluteToleranceName", float, False),
    "nlopt_stopval": NloptOption("StopValName", float, False),
    "nlopt_local_optimizer_algorithm": NloptOption(
        "LocalOptimizerAlgorithmName", str, False),
    "nlopt_local_optimizer_xtol_rel": NloptOption(
        "LocalOptimizerXRelativeToleranceName", float, True),
    "nlopt_local_optimizer_xtol_abs": NloptOption(
        "LocalOptimizerXAbsoluteToleranceName", float, True),
    "nlopt_local_optimizer_ftol_rel": NloptOption(
        "LocalOptimizerFRelativeToleranceName", float, True),
    "nlopt_local_optimizer_ftol_abs": NloptOption(
        "LocalOptimizerFAbsoluteToleranceName", float, True),
    "nlopt_local_optimizer_max_eval": NloptOption("LocalOptimizerMaxEvalName", int, True),
    "nlopt_local_optimizer_max_time": NloptOption("LocalOptimizerMaxTimeName", float, True),
}
# Drake deliberately exposes no INNER stopval: AUGLAG and MLSL reset the inner stopval from
# the outer one just before running, so any value set there would be accepted and silently
# discarded (nlopt_solver.cc:557-563). Ten is the complete set; do not invent an eleventh.


## Algorithms Drake LISTS as valid and then cannot run. Drake's bundled NLopt is built
## WITHOUT the Luksan sources (they are LGPL), which compiles out L-BFGS, the variable-metric
## family and every truncated-Newton variant. `ParseNloptAlgorithm` still accepts the names --
## they appear in the "valid choices are:" message it prints for a typo -- and the failure
## arrives from inside the solve as
##
##     ERROR - attempting to use NLOPT_LD_LBFGS, but Luksan code disabled
##
## on stderr, with Drake returning SolutionResult.kInvalidInput and NloptSolverDetails.status
## 0. Measured 2026-09-18 on the 0.0.20260918 nightly with a three-variable bound- and
## nonlinear-constrained program: exactly these eight are refused, while LD_MMA, LD_CCSAQ,
## LD_SLSQP, LN_COBYLA and LN_BOBYQA all solve it. So the usable GRADIENT-BASED inner
## optimizers for an augmented Lagrangian here are LD_MMA, LD_CCSAQ and LD_SLSQP, and nothing
## else.
##
## Refused here rather than discovered on the cluster, because the symptom is a quiet one: a
## 0.5 s cell with q=None, max_violation=None and fail_reason unset, which reads like a harness
## bug rather than an unavailable algorithm.
NLOPT_LUKSAN_DISABLED = frozenset({
    "LD_LBFGS", "NLOPT_LD_LBFGS_NOCEDAL",
    "LD_VAR1", "LD_VAR2",
    "LD_TNEWTON", "LD_TNEWTON_RESTART", "LD_TNEWTON_PRECOND",
    "LD_TNEWTON_PRECOND_RESTART",
})


def CheckNloptAlgorithms(options):
    """Refuse an algorithm Drake's NLopt accepts by name and then cannot run.

    Checked for the OUTER algorithm as well as the inner one: `--set nlopt_algorithm=LD_LBFGS`
    fails in exactly the same way, and this column's whole value is that it is a third method
    class, so a silent kInvalidInput on every cell is the worst failure available here.
    """
    for field_name in ("nlopt_algorithm", "nlopt_local_optimizer_algorithm"):
        value = getattr(options, field_name, None)
        if value in NLOPT_LUKSAN_DISABLED:
            raise ValueError(
                f"NLopt: {field_name}={value!r} is compiled OUT of Drake's NLopt. Drake's "
                f"ParseNloptAlgorithm lists it as a valid choice, but the solve then prints "
                f"'attempting to use NLOPT_{value}, but Luksan code disabled' and returns "
                f"kInvalidInput with status 0 -- a 0.5 s cell with no q and no violation, on "
                f"every cell of the run. The Luksan sources are LGPL and Drake does not "
                f"bundle them, so all of {sorted(NLOPT_LUKSAN_DISABLED)} are unavailable. The "
                f"usable gradient-based choices are LD_MMA, LD_CCSAQ and LD_SLSQP.")


def CheckNloptOptions(options):
    """Validate the post-1.56.0 NLopt fields, returning the ones to emit.

    Two refusals, and their order is deliberate:

      1. An INNER option set while `nlopt_local_optimizer_algorithm` is unset. Drake reads it
         and then throws it away, so the cell measures the DEFAULT inner solver while the
         record claims a tuned one -- a swept column that reports a real-looking null. This
         is a property of the options, true on every Drake, so it is checked first. That also
         makes the refusal reproducible on the cluster's 1.56.0, where the availability check
         below would otherwise fire first and give a different (also true, less useful)
         reason.
      2. An option this Drake does not have, reported with the surface actually found, so the
         message says WHICH MACHINE is wrong rather than merely that something is.

    An empty string counts as unset for the inner algorithm, because that is precisely what
    Drake means by it (`local_optimizer_algorithm.empty()`).
    """
    ## Unconditional: the OUTER algorithm is a pre-1.56 option and can be wrong on its own.
    CheckNloptAlgorithms(options)
    requested = {name: getattr(options, name) for name in NLOPT_POST_1_56_OPTIONS
                 if getattr(options, name) is not None and getattr(options, name) != ""}
    ## The common path -- nothing post-1.56 set -- costs one comprehension and touches Drake
    ## not at all, so nothing about a default run changes.
    if not requested:
        return requested

    inner = sorted(n for n in requested if NLOPT_POST_1_56_OPTIONS[n].inner)
    if inner and not requested.get("nlopt_local_optimizer_algorithm"):
        raise ValueError(
            f"NLopt: {', '.join(inner)} set with nlopt_local_optimizer_algorithm unset. "
            f"Drake reads every local_optimizer_* option but applies them only inside "
            f"`if (!parsed_options.local_optimizer_algorithm.empty())` "
            f"(nlopt_solver.cc:546-564), so this combination is ACCEPTED AND INERT -- the "
            f"solve would run NLopt's own inner optimizer while the record claims a tuned "
            f"one. Name the inner algorithm (LD_MMA, LD_CCSAQ and LD_SLSQP are the "
            f"gradient-based choices Drake's build can actually run) or drop the inner "
            f"options.")

    ## Availability is checked only for a run that will actually EMIT these options. It used
    ## to be unconditional, which was right while every post-1.56.0 field defaulted to None:
    ## a non-None value was then always deliberate. Since 2026-09-19 three of them are the
    ## ADOPTED NLopt defaults, so an unconditional check would refuse `ProgramOptions()`
    ## itself -- and therefore every IPOPT and SNOPT cell -- on the cluster's pinned 1.56.0,
    ## for options those solvers never touch. The other two refusals above stay unconditional
    ## because they are logic errors on every Drake.
    if getattr(options, "which_solver", None) != "nlopt":
        return requested
    surface = NloptOptionSurface()
    missing = [(n, NLOPT_POST_1_56_OPTIONS[n].accessor) for n in sorted(requested)
               if NLOPT_POST_1_56_OPTIONS[n].accessor not in surface]
    if missing:
        raise ValueError(
            "NLopt: this Drake has no option for "
            + ", ".join(f"{n} (NloptSolver.{a}())" for n, a in missing)
            + f". The NLopt options it accepts are {sorted(surface.values())}. Drake raises "
            f"on a name it does not know, and benchmark.run_grid records each such cell as "
            f"fail_reason='error' rather than stopping -- so emitting it anyway returns a "
            f"full column of instant failures, not an error. The cluster runs Drake 1.56.0, "
            f"whose surface is the six algorithm, constraint_tol, xtol_rel, xtol_abs, "
            f"max_eval, max_time; the local_optimizer_* options and ftol_rel/ftol_abs/"
            f"stopval each arrived later, and in different releases.")
    return requested


## ------------------------- the svgd option surface -----------------------------------
##
## The enum-valued `svgd_*` fields and what each accepts. Checked in
## `ProgramOptions.__post_init__`, so a mistyped `--set svgd_method=...` raises at options
## construction -- before the first cell and outside run_grid's per-cell `except` -- rather
## than landing as a full column of instant failures, exactly as CheckNloptOptions does for
## NLopt. Like that check it is NOT gated on `which_solver == "svgd"`: a manifest that sets
## a nonsense svgd value on an IPOPT arm is a broken sweep and should say so.
SVGD_ENUM_FIELDS = {
    "svgd_method": ("al_svgd",),
    "svgd_dtype": ("float32", "float64"),
    "svgd_paired_init": ("jitter", "native"),
    "svgd_kernel": ("q", "none"),
    "svgd_warmup": ("none", "cem"),
}

## The svgd step size's default (`svgd_lr`): see `docs/svgd-solver.md`, "The step size".
SVGD_LR_DEFAULT = 1e-10


def CheckSvgdOptions(options):
    """Refuse an svgd enum value outside its choices, a non-positive particle count,
    temperature or step size, and the CUDA-graph switch without the compile switch."""
    for name, allowed in SVGD_ENUM_FIELDS.items():
        value = getattr(options, name)
        if value not in allowed:
            raise ValueError(f"{name}={value!r} is not one of {allowed}")
    if int(options.svgd_n) < 1:
        raise ValueError(f"svgd_n must be >= 1, got {options.svgd_n!r} (N = 1 is the "
                         f"degenerate single-particle control, not 0)")
    for name in ("svgd_temperature", "svgd_lr", "svgd_rho"):
        if not float(getattr(options, name)) > 0.0:
            raise ValueError(f"{name} must be > 0, got {getattr(options, name)!r}")
    if options.svgd_cuda_graph and not options.svgd_compile:
        raise ValueError("svgd_cuda_graph=True requires svgd_compile=True (the graphs replay "
                         "the compiled stages)")


@dataclass
class ProgramOptions:
    joint_centering_cost: float = field(default=0.0, metadata={"help": "Weight for joint centering cost"})
    collision_avoidance: bool = field(default=True, metadata={"help": "Add collision avoidance constraints"})
    # The shape of the collision row, exposed so it can be swept. The defaults are the
    # values that were hardcoded in CreateCollisionFreeConstraint, so nothing moves unless
    # they are set. `collision_influence_offset` is the distance at which a geometry pair
    # starts contributing to Drake's smooth penalty, i.e. it sets the gradient the solver
    # has to follow while it is still far from contact; `collision_row_scale` scales the
    # whole row (and its upper bound with it, since the binding's own threshold is 1).
    collision_bound: float = field(default=1e-3, metadata={"help": "MinimumDistanceLowerBoundConstraint 'bound' (metres)"})
    collision_influence_offset: float = field(default=1e-1, metadata={"help": "MinimumDistanceLowerBoundConstraint 'influence_distance_offset' (metres)"})
    collision_row_scale: float = field(default=0.1, metadata={"help": "Scaling applied to the collision constraint row and its upper bound"})
    joint_limits: bool = field(default=True, metadata={"help": "Enforce joint limits"})
    ik_constraint_tol: tuple = field(default=(1e-4, 0.01), metadata={"help": "Tolerance for IK constraints: tuple of (position tol, orientation tol in radians). The orientation entry is used only by orientation_error_form='rpy_boxed'; 'rpy' pins the residual to zero"})
    orientation_error_form: str = field(default="rpy", metadata={"help": "'rpy' pins the roll-pitch-yaw residual to zero, as ../codebase's pose constraint does; 'rpy_boxed' allows +-ori_tol on each row"})
    correction_cost_weight: float = field(default=0.0, metadata={"help": "Weight for correction cost to keep close to zero"})

    ## The two Stage F interventions against the flow's runaway regions. BOTH ARE STATED
    ## DEVIATIONS from the draft's eq. (6), authorised by Thomas as experiments while he
    ## said he dislikes them ("we should be able to rely on the constraint to handle it";
    ## lifting "effectively adds a nonlinear equality constraint whereas currently one just
    ## has a free variable"). They are diagnostics, not candidates to adopt silently: an
    ## arm running either must be labelled as such wherever it is reported, and neither may
    ## be fielded under the name of the paper's learned formulation.
    ##
    ## The pathology they address: the joint-limit row is imposed on `q = flow(c, z) + q_c`,
    ## and in the flow's worst-case-gain regions `dq/dz` is as large as `q` is (~1e13 for 12
    ## coupling blocks at rnvp_clamp 2.5), so a Newton step is *attracted* into them. Note
    ## the arm does control `q` there -- the limit row's gradient is pulled back through the
    ## network to `z` exactly, which is the point of differentiating through it -- so this is
    ## a magnitude problem, not a blindness problem.
    lift_q: bool = field(default=False, metadata={"help": "Stage F: add q as a bounded decision variable and impose the chart as an equality, so joint limits become variable bounds IPOPT satisfies at every iterate"})
    joint_limit_penalty_weight: float = field(default=0.0, metadata={"help": "Stage F: weight on the quadratic hinge penalty for joint-limit violation; zero (with zero gradient) inside the limits"})

    mug_height: float = field(default=0.035, metadata={"help": "Mug height for valid grasp poses"})

    ## Network evaluation ##
    # The flow is a float32 artifact, but evaluating it in float64 costs ~15% and
    # makes the map smooth well below 1e-4. In float32 the finite-difference error
    # against the analytic Jacobian blows up from 6e-3 (h=1e-4) to 1.8 (h=1e-6),
    # which is what makes SNOPT's derivative check fail and starves the line search.
    use_float64: bool = field(default=True, metadata={"help": "Evaluate the flow in float64 so the map is smooth at solver step sizes"})

    c_position_slack: float = field(default=0.25, metadata={"help": "Half-width of the box on the conditioning position, about the target"})

    # torch.compile on the jacrev: measured 17.98 -> 13.55 ms (1.33x) at batch 1 in float64,
    # one dynamo graph with no recompiles across iterates, agreeing with eager to 1e-14 on a
    # Jacobian of magnitude 12, for a 14.2 s one-off compile penalty. The Jacobian is 84% of
    # a learned solve, so inside a fixed wall-clock cap this is roughly 30% more iterations
    # -- which means it *moves the learned arm's success rate* and every arm of a reported
    # comparison has to be run with the same setting. Off by default so a solve costs no
    # compile; the benchmark scripts turn it on and warm it up before the grid.
    compile_flow_jacobian: bool = field(default=False, metadata={"help": "torch.compile the flow Jacobian once per process and share it between programs"})

    # CUDA-graph replay of the compiled flow, on BOTH paths a solve uses: the jacrev and the
    # plain forward pass, which IPOPT runs as often as the jacrev (Drake evaluates constraint
    # values in double and costs in AutoDiffXd, so each trial point costs one of each) and
    # which is otherwise eager, uncompiled and recording autograd for nothing. Replay drops
    # the CPU dispatch that bounds a batch-1 call. Measured on a V100, float64, against
    # --compile: jacrev 4.40 -> 2.37 ms (Panda n6), 3.14 -> 1.62 ms (iiwa n4); forward pass
    # 7.07 -> 0.70 and 4.79 -> 0.49 ms against the eager path it replaces -- about 3.7x on
    # the flow work of an IPOPT trial point and ~1.9x under SNOPT, which never takes the
    # value path. The replayed Jacobian is the compiled one's kernels. Requires
    # compile_flow_jacobian, and like it moves iterations per wall-clock cap, so every run
    # being compared must set it the same way. Capture is offline: WarmUpJacobian pays it.
    flow_cuda_graph: bool = field(default=False, metadata={"help": "Replay the compiled flow (jacrev and forward pass) as CUDA graphs; requires compile_flow_jacobian"})


    ## Evaluation sharing ##
    # Every Drake binding evaluates its own callback, so the joint-centering cost used to
    # run a *second* forward pass and a second jacrev through the flow at the same point
    # as the constraint binding. An IPOPT log of the mug problem records 1276 objective
    # evaluations against 1276 constraint evaluations and 455 objective gradients against
    # 490 constraint Jacobians -- i.e. about half the network work was redundant.
    # On by default: the memoised path returns bit-identical values *and* derivatives, so
    # this is a pure throughput win (measured 12/30 -> 22/30 on the Panda grasp once the
    # frame was also fixed). Turn it off only to reproduce a pre-overhaul measurement.
    share_flow_evaluations: bool = field(default=True, metadata={"help": "Memoise VarsToQ/fk so the cost and constraint bindings share one flow evaluation per point"})

    ## Conditioning of the learned program ##
    # The flow is conditioned on the pose of the frame it was *trained* on. In the finray
    # grasp scene the body called "panda_hand" is a different frame, 27 mm and 120 degrees
    # away, so looking it up by name conditions the network on a pose it never saw. Off by
    # default only so the ladder can measure the repair separately from the redesigns.
    calibrate_flow_frame: bool = field(default=True, metadata={"help": "Express the conditioning pose in the frame the flow was trained on rather than in whichever scene body shares its name"})
    # The latent prior is N(0, I), so |z| concentrates near sqrt(latent_dim); a +-5 box
    # per component lets the optimiser walk to |z| ~ 13, deep into the tail where the flow
    # has seen no training mass and its output stops meaning anything. A norm bound is the
    # learned analogue of the analytic framework's reachability constraint, which is the
    # gap the draft itself names ("outside the reachable set IKFlow's gradients explode").
    latent_trust_region: float = field(default=None, metadata={"help": "Bound on ||z||; None keeps the per-component box only"})
    latent_cost_weight: float = field(default=0.0, metadata={"help": "Weight on ||z||^2, keeping the latent in the flow's typical set"})
    correction_bound: float = field(default=0.1, metadata={"help": "Half-width of the box on the joint-space correction"})
    ## The trust-region RULE, as an alternative to a per-robot constant. Every robot but the
    ## Panda sizes its region as `round(sqrt(dim_latent) + 1.5, 2)` (iiwa 4.3 by hand, soft
    ## 4.96, screw 4.15, GVS 4.5, computed); the Panda's `--config latent` carries 4.0. With
    ## this on, the radius is computed from the LOADED chart's width inside the program and
    ## `latent_trust_region` is ignored, so one flag selects the rule for every robot at once
    ## -- the A/B the PI asked for (2026-10-08) rather than a per-script edit. What moves
    ## under it: the Panda 4.0 -> 4.15, and the iiwa 4.3 -> 4.33 (its hand constant was the
    ## rule rounded to one decimal); the soft, screw and GVS arms already carry the rule.
    latent_trust_region_rule: bool = field(default=False, metadata={"help": "Size the latent trust region as round(sqrt(dim_latent)+1.5, 2) from the loaded chart, ignoring latent_trust_region. Panda 4.0->4.15, iiwa 4.3->4.33; the other robots already use the rule"})

    ## ---- cross-robot settings, unified 2026-10-08 (PI's decision) -------------------
    ## An inventory ahead of re-measuring the grasp rows found five settings that differed
    ## between robots with no deliberate reason. Four are unified here behind ONE flag (the
    ## fifth, the trust region, is the A/B above):
    ##   1. the native grasp start for `c` is the flow-frame pose of the generating
    ##      configuration, `mug.middle @ X_grasp_ee` (the iiwa and screw arms seeded at the
    ##      mug centre with zero rpy);
    ##   2. the joint-space arm's variable bound is `ConfigLimits()` (the Panda used +-10 rad,
    ##      the iiwa a hand-typed table);
    ##   3. `q_nominal` is a NONSINGULAR HOME pose per robot (it was zeros everywhere, which
    ##      on the Panda is outside q4's range and singular, and singular on every S-R-S arm);
    ##   5. the grasp `c` box is centred on that same flow-frame pose, so every robot gets the
    ##      same +-c_position_slack rather than a margin that depends on its gripper standoff.
    ## `legacy_robot_settings=True` restores all four old behaviours together -- the CONTROL
    ## for the re-run, reachable from `--set`, recorded in `metadata["overrides"]` and in
    ## `metadata["robot_settings"]` (IKFlowProgram.RobotSettings).
    legacy_robot_settings: bool = field(default=False, metadata={"help": "Restore the pre-2026-10-08 per-robot settings together: Panda +-10 q bound and iiwa's hand-typed limits, zeros q_nominal, mug-centred grasp c start (iiwa, screw) and mug-centred grasp c box. The control arm for the unified re-run"})

    ## Solver behaviour ##
    ipopt_mu_strategy: str = field(default=None, metadata={"help": "IPOPT 'mu_strategy'; 'adaptive' often helps on badly scaled problems"})
    ## The learned arm's joint-limit row is evaluated on the flow's output, which has a
    ## worst-case gain of ~1e13 (12 coupling blocks at rnvp_clamp 2.5), and the runaway
    ## cells return configurations of 1e7-1e16 rad. IPOPT's default gradient-based scaling
    ## computes its factors from the gradients at the *starting* point and caps them at
    ## `nlp_scaling_max_gradient` (100), so a row that only becomes enormous later is
    ## scaled as though it were ordinary. These two make that reachable from `--set`.
    ipopt_nlp_scaling_method: str = field(default=None, metadata={"help": "IPOPT 'nlp_scaling_method': 'gradient-based' (default), 'none', 'equilibration-based'"})
    ipopt_nlp_scaling_max_gradient: float = field(default=None, metadata={"help": "IPOPT 'nlp_scaling_max_gradient' (default 100)"})
    ## Quasi-Newton and barrier knobs. Drake runs IPOPT with
    ## `hessian_approximation = limited-memory` -- confirmed in the user-options echo of an
    ## archived log -- because it supplies no second derivatives, so the L-BFGS history IS
    ## the Hessian here and its length is a live knob. `mu_init` is read ONLY under
    ## `mu_strategy = monotone`: set alone it is echoed `used = no`, and it becomes
    ## `used = yes` only when the strategy is passed explicitly alongside it. Measured.
    ##
    ## Note what is NOT reachable: `linear_solver`. Drake's IPOPT is built against SPRAL
    ## and offers only `spral` and `custom` -- `mumps` raises at SetOption -- so there is
    ## no linear-solver axis on this problem.
    ## IPOPT's REAL convergence test, as opposed to the `acceptable_*` family above --
    ## which is its RELAXED EARLY STOP and a different thing entirely. Nothing in this repo
    ## ever set these, so IPOPT has always converged at its own defaults, and two of them
    ## are far looser than the SQP column is held to:
    ##
    ##     constr_viol_tol  1e-4   vs SNOPT's Major feasibility tolerance  1e-6
    ##     dual_inf_tol     1      vs SNOPT's Major optimality tolerance   2e-6
    ##     tol              1e-8
    ##     compl_inf_tol    1e-4
    ##
    ## So a solver comparison that leaves these alone lets IPOPT stop at 100x the
    ## constraint violation and six orders more dual infeasibility. That is a property of
    ## the HARNESS, not of interior-point methods, and it has to be measured before the
    ## solver table means anything. Rung 2 of the tolerance ladder says each solver gets
    ## its own well-posed defaults -- but it does not say never check what that is worth.
    ipopt_tol: float = field(default=None, metadata={"help": "IPOPT 'tol' (default 1e-8): the overall convergence tolerance"})
    ipopt_constr_viol_tol: float = field(default=None, metadata={"help": "IPOPT 'constr_viol_tol' (default 1e-4): constraint violation at convergence. SNOPT's counterpart defaults to 1e-6"})
    ipopt_dual_inf_tol: float = field(default=None, metadata={"help": "IPOPT 'dual_inf_tol' (default 1): dual infeasibility at convergence. SNOPT's counterpart defaults to 2e-6"})
    ipopt_compl_inf_tol: float = field(default=None, metadata={"help": "IPOPT 'compl_inf_tol' (default 1e-4): complementarity at convergence"})
    ipopt_limited_memory_max_history: int = field(default=None, metadata={"help": "IPOPT 'limited_memory_max_history' (default 6): the L-BFGS history length, which is the whole Hessian approximation here"})
    ipopt_limited_memory_update_type: str = field(default=None, metadata={"help": "IPOPT 'limited_memory_update_type': 'bfgs' (default) or 'sr1'"})
    ipopt_mu_init: float = field(default=None, metadata={"help": "IPOPT 'mu_init' (default 0.1). Only read under mu_strategy=monotone -- pass ipopt_mu_strategy=monotone with it or it is silently unused"})
    ipopt_alpha_for_y: str = field(default=None, metadata={"help": "IPOPT 'alpha_for_y' (default 'primal'): how the dual step size is chosen"})
    ipopt_recalc_y: str = field(default=None, metadata={"help": "IPOPT 'recalc_y' (default 'no'): recompute the multipliers from a least-squares estimate"})
    ipopt_bound_relax_factor: float = field(default=None, metadata={"help": "IPOPT 'bound_relax_factor' (default 1e-8): how far bounds are relaxed before the solve"})
    ## STEP ACCEPTANCE, which is a different lever from everything already refuted. The
    ## damping strategies (`jacobian_max_norm` and friends) altered the DERIVATIVES the
    ## solver was handed, breaking the correspondence between the constraint values IPOPT
    ## evaluates and the gradients it uses -- which is why the most aggressive settings
    ## failed hardest. These three leave the program exactly as written and change only
    ## which trial points the filter will accept.
    ##
    ## Why the runaway is reachable at all: IPOPT holds VARIABLE BOUNDS at every iterate but
    ## general constraints only at convergence, and in the learned formulation `q` is not a
    ## decision variable, so the joint-limit rows are general constraints and an iterate may
    ## sit at |q| = 1e8. That is exactly why `lift_q` (q as a bounded variable) produced zero
    ## runaway cells -- at the cost of seven equality rows. `theta_max_fact` attacks the same
    ## mechanism without touching the formulation: IPOPT rejects any trial point whose
    ## constraint violation exceeds theta_max_fact * max(1, theta(x_0)).
    ipopt_theta_max_fact: float = field(default=None, metadata={"help": "IPOPT 'theta_max_fact' (default 1e4): trial points above theta_max_fact*max(1,theta(x0)) constraint violation are rejected outright"})
    ipopt_watchdog_trigger: int = field(default=None, metadata={"help": "IPOPT 'watchdog_shortened_iter_trigger' (default 10); 0 disables the watchdog, which otherwise RELAXES filter acceptance for a few iterations"})
    ipopt_max_soc: int = field(default=None, metadata={"help": "IPOPT 'max_soc' (default 4): second-order corrections, which exist to rescue steps the filter rejected for constraint violation"})
    max_iter: int = field(default=None, metadata={"help": "Iteration cap (IPOPT max_iter / SNOPT Major iterations limit / NLopt max_eval)"})

    ## SNOPT, the SQP arm of the solver axis. Every field is None = "leave SNOPT's own
    ## default alone", which is the rule for this axis: the three solvers stand for three
    ## METHOD CLASSES (interior point, SQP, augmented Lagrangian) and each must converge
    ## at its own well-posed tolerances. Transplanting IPOPT's numbers onto SNOPT is what
    ## the code used to do -- it fed `acceptable_tol` and `acceptable_constr_viol_tol`,
    ## IPOPT's RELAXED EARLY-STOP criteria, into SNOPT's Major tolerances, which are its
    ## actual convergence test. IPOPT converges to `tol` (1e-8); SNOPT was being asked for
    ## 1e-3. The shared, deliberately-looser task gate is what decides success.
    ##
    ## Drake routes these by Python type, so a float option passed as an int (or the
    ## reverse) reaches the wrong snSet* and raises. Hence the explicit casts below.
    snopt_major_feasibility_tol: float = field(default=None, metadata={"help": "SNOPT 'Major feasibility tolerance' (SNOPT default 1e-6)"})
    snopt_major_optimality_tol: float = field(default=None, metadata={"help": "SNOPT 'Major optimality tolerance' (SNOPT default 1e-6)"})
    snopt_minor_feasibility_tol: float = field(default=None, metadata={"help": "SNOPT 'Minor feasibility tolerance'"})
    snopt_minor_iterations_limit: int = field(default=None, metadata={"help": "SNOPT 'Minor iterations limit'"})
    snopt_iterations_limit: int = field(default=None, metadata={"help": "SNOPT 'Iterations limit': the TOTAL minor-iteration budget, INFO 31 when it binds. Distinct from 'Minor iterations limit' (per QP subproblem) and from max_iter (majors, INFO 32); both 31 and 32 count as hit_iteration_cap, so lifting the iteration budget means raising this too"})
    snopt_scale_option: int = field(default=None, metadata={"help": "SNOPT 'Scale option' (0 none, 1 linear, 2 all). The counterpart of ipopt_nlp_scaling_method, which measured inert"})
    snopt_verify_level: int = field(default=None, metadata={"help": "SNOPT 'Verify level'; -1 disables the derivative check. Our gradients are analytic, so a check costs evaluations for nothing"})
    snopt_linesearch_tolerance: float = field(default=None, metadata={"help": "SNOPT 'Linesearch tolerance' (default 0.9); smaller means a more accurate line search"})
    snopt_superbasics_limit: int = field(default=None, metadata={"help": "SNOPT 'Superbasics limit'; INFO 33 means this was too small"})
    ## Four more knobs, every default below read off SNOPT's OWN parameter echo rather
    ## than from documentation -- the echo is the only thing that proves an option landed.
    ## Two traps this turned up, both of the "accepted and inert" kind that `Timing Level`
    ## already cost this repo once:
    ##
    ##  * `Hessian updates` (default 99999999) is ignored in FULL-memory mode, which is
    ##    what SNOPT picks at these problem sizes (n = 20 or 21, well under its 75-variable
    ##    threshold). Setting it changes nothing. `Hessian frequency` is the one that bites,
    ##    and setting it moves BOTH numbers in the echo. So only the frequency is exposed.
    ##  * `Nonderivative linesearch` is a VALUELESS keyword: SNOPT switches on the keyword
    ##    appearing at all, so passing 0 turns it ON exactly as passing 1 does. It is
    ##    therefore a bool here, emitted only when True. It matters because the flow
    ##    Jacobian is ~84% of a solve, so a line search that needs only function values is
    ##    the largest single saving available on this problem. In the echo it appears
    ##    abbreviated as `Nonderiv.  linesearch`, not by its full name.
    snopt_hessian_frequency: int = field(default=None, metadata={"help": "SNOPT 'Hessian frequency' (default 99999999, i.e. never reset). 'Hessian updates' is inert in the full-memory mode this problem size selects"})
    snopt_elastic_weight: float = field(default=None, metadata={"help": "SNOPT 'Elastic weight' (default 1e5): the penalty on constraint violation in elastic mode, which is how SNOPT copes with an infeasible start"})
    snopt_crash_option: int = field(default=None, metadata={"help": "SNOPT 'Crash option' (default 3): how the initial basis is chosen"})
    snopt_proximal_point_method: int = field(default=None, metadata={"help": "SNOPT 'Proximal point method' (default 1): how far the first major moves from the given start"})
    snopt_nonderivative_linesearch: bool = field(default=False, metadata={"help": "SNOPT 'Nonderivative linesearch'. Valueless keyword -- emitted only when True, and passing 0 would turn it ON, not off"})
    ## The two SNOPT knobs that are genuine analogues of the repo's best open IPOPT lead.
    ## `Major step limit` bounds ||dx|| <= limit*(1 + ||x||) per major iteration, which is
    ## the trust region the learned formulation wants -- the runaway is ONE accepted
    ## catastrophic step out of a well-behaved trajectory. `Violation limit` is SNOPT's
    ## counterpart to ipopt_theta_max_fact, which gained 3 cells and lost none on a probe.
    ## ADOPTED 2026-09-19 by Thomas, and the ONE solver setting this project fields.
    ## Drake/SNOPT's own default is 2.0; 0.5 is the single survivor of stages SNOPTTUNE
    ## (13 settings) and SNOPTCOMBO (7 crosses of it), each at 480 cells x 12 rows. On the
    ## learned arm it is better on 11 of 12 rows, significantly on 3, significantly worse on
    ## none, and the joint-space arm rises on all 12 -- so it is a property of SNOPT on this
    ## problem, not of the chart. It moves NO learned-vs-joint-space verdict (learned +161
    ## cells of 5760, joint space +164; per-row margins sum to -3), so it is adopted for
    ## FAIRNESS, not for the comparison: IPOPT's column already runs a tuned configuration
    ## (the acceptable-point early stop) and SNOPT's ran bare defaults.
    ##
    ## Consequences for anything reading archived numbers:
    ##   * The SNOPT numbers of record are now SNOPTCOMBO's `mstep0p5` column, NOT the
    ##     `sc_SOLVER2_*_snopt_*` columns, which were measured at Drake's defaults.
    ##   * A stage whose column means "Drake's SNOPT defaults" must now say so explicitly
    ##     with `--set snopt_major_step_limit=None`; the four historical SNOPT tables in
    ##     cluster/gen_manifest.py were updated so re-generating them reproduces what ran.
    ## It is still a member of STEP_REJECTION_KNOBS, but stage STEP is not why it is here:
    ## that stage refuted the family at 480 cells and this value was adopted on the separate
    ## per-solver-tuning question.
    snopt_major_step_limit: float = field(default=0.5, metadata={"help": "SNOPT 'Major step limit' (Drake/SNOPT default 2.0; ADOPTED at 0.5): bounds ||dx|| <= limit*(1+||x||) per major iteration"})
    snopt_violation_limit: float = field(default=None, metadata={"help": "SNOPT 'Violation limit' (default 10): the largest constraint violation allowed beyond the initial point; SNOPT's theta ceiling"})
    ## Print verbosity. SNOPT's end-of-run summary block -- 'No. of major iterations' and
    ## the funobj/funcon call counts -- is the ONLY place its iteration count exists, so
    ## the major level must stay >= 1. The minor level stays 0 by default because a 45 s
    ## solve writes one log per cell and minor lines dominate the size.
    snopt_major_print_level: int = field(default=1, metadata={"help": "SNOPT 'Major print level'; >=1 is required for the summary block parse_log reads"})
    snopt_minor_print_level: int = field(default=0, metadata={"help": "SNOPT 'Minor print level'; 0 keeps the per-cell log small"})
    snopt_solution_print: bool = field(default=False, metadata={"help": "SNOPT 'Solution Yes': dump every row and column at the end. Off -- nothing parses it and it is a fifth of the file"})
    ## Used only when `max_iter` is None. IPOPT's max_iter default is 3000 and SNOPT's Major
    ## iterations limit default is 1000, so leaving both alone gives the two solvers
    ## different budgets under a wall-clock-capped comparison. Measured binding.
    snopt_major_iterations_default: int = field(default=3000, metadata={"help": "SNOPT 'Major iterations limit' when max_iter is unset; 3000 matches IPOPT's own default so the wall clock is what binds"})

    ## NLopt, the AUGMENTED LAGRANGIAN arm. `LD_SLSQP` would be the wrong default: it is an
    ## SQP method, so it would make this column a duplicate of SNOPT's rather than a third
    ## method class. `LD_AUGLAG` rather than `LD_AUGLAG_EQ` because the `_EQ` variant
    ## absorbs only EQUALITY constraints into the augmented Lagrangian and leaves
    ## inequalities to the inner solver, and this program carries both kinds (pose or
    ## mug-axis equalities alongside collision, joint-limit and trust-region inequalities).
    ##
    ## The four fields immediately below, plus `max_time`, are Drake 1.56.0's ENTIRE NLopt
    ## surface, which is what the cluster runs; they are safe to emit anywhere. Everything
    ## newer lives in the block after `nlopt_max_eval`, defaults to None, and is emitted only
    ## when set -- see NLOPT_POST_1_56_OPTIONS and CheckNloptOptions above for why that
    ## asymmetry exists and what it protects.
    nlopt_algorithm: str = field(default="LD_AUGLAG", metadata={"help": "NLopt 'algorithm'. An augmented-Lagrangian variant by design; LD_SLSQP would duplicate SNOPT's method class"})
    nlopt_constraint_tol: float = field(default=None, metadata={"help": "NLopt 'constraint_tol' (Drake default 1e-6)"})
    nlopt_xtol_rel: float = field(default=None, metadata={"help": "NLopt 'xtol_rel' (Drake default 1e-6)"})
    nlopt_xtol_abs: float = field(default=None, metadata={"help": "NLopt 'xtol_abs' (Drake default 1e-6)"})
    ## Drake DEFAULTS max_eval to 1000 -- this is a cap, not "unset", and it binds here:
    ## the learned arm runs hundreds of major iterations and several map evaluations each.
    ## Left alone, the NLopt column would silently measure a 1000-evaluation budget instead
    ## of the wall-clock cap every other column is measured under. 0 disables it.
    nlopt_max_eval: int = field(default=0, metadata={"help": "NLopt 'max_eval'; 0 disables. Drake's own default is 1000, which would bind and make this column an evaluation-budget measurement"})

    ## ---- the post-1.56.0 surface: opt-in, never emitted unless set ----------------------
    ## Every field below names an option absent from the cluster's Drake 1.56.0, so every one
    ## defaults to None and _NloptOptions emits nothing for an unset field. That is not
    ## tidiness: it is what keeps a default NLopt cell byte-identical to every archived NLopt
    ## result, and it is what keeps a cluster run alive. Setting one on a Drake that lacks it
    ## is refused at CONFIGURATION time, with the surface actually found -- not left to raise
    ## from inside Drake on every cell, where run_grid would swallow it.
    nlopt_ftol_rel: float = field(default=None, metadata={"help": "NLopt 'ftol_rel': relative cost-change stopping criterion (Drake default 0, disabled). Post-1.56.0"})
    nlopt_ftol_abs: float = field(default=None, metadata={"help": "NLopt 'ftol_abs': absolute cost-change stopping criterion (Drake default 0, disabled). Post-1.56.0"})
    ## Plumbed for completeness and DELIBERATELY NOT SWEPT. `stopval` stops the solve the
    ## moment the cost falls to a target, and this program minimises a cost whose optimum is
    ## unknown -- a weighted sum whose scale moves with the target, the formulation and the
    ## weights, so no threshold we could name means the same thing in two cells. Worse,
    ## stopping on cost returns an iterate never driven to feasibility, and feasibility is
    ## what the shared task gate scores: the cells it "won" would come back gate failures.
    nlopt_stopval: float = field(default=None, metadata={"help": "NLopt 'stopval': stop once cost <= this (Drake default -inf, never fires). Post-1.56.0. Plumbed but NOT swept -- this cost has no known optimum and stopping on it returns points that fail the task gate"})
    ## The AUGMENTED LAGRANGIAN's INNER SOLVER. AUGLAG solves nothing itself: it hands a
    ## sequence of bound-constrained subproblems to a second NLopt algorithm. Left unnamed,
    ## Drake never calls set_local_optimizer and NLopt picks its own -- a supported state, and
    ## the one every archived NLopt cell ran in. Naming it is the only knob on this column that
    ## changes the METHOD rather than a tolerance, which is what makes it worth the traps below.
    ##
    ## NOTE, measured rather than assumed: whatever NLopt picks when this is unset, it is NOT
    ## LD_LBFGS. Drake's NLopt is built without the LGPL Luksan sources, so LD_LBFGS and the
    ## whole variable-metric and truncated-Newton family are compiled out (see
    ## NLOPT_LUKSAN_DISABLED). An earlier comment here and in CLAUDE.md asserted LD_LBFGS was
    ## the default; it cannot be, because naming it FAILS while leaving this unset solves.
    ##
    ## TRAP ONE, and why the six fields after it are REFUSED rather than ignored when this one
    ## is unset: Drake reads every local_optimizer_* option unconditionally but applies them
    ## only inside `if (!parsed_options.local_optimizer_algorithm.empty())`. Set
    ## nlopt_local_optimizer_max_eval alone and Drake accepts it, never calls
    ## set_local_optimizer, and discards the number -- the "accepted and did nothing" failure
    ## that `Timing Level`, `Hessian updates` and `linear_solver=mumps` have each already cost
    ## this repo a measurement.
    ##
    ## TRAP TWO: naming the algorithm is NOT neutral even with every tolerance left alone.
    ## Drake then pushes its OWN inner defaults (xtol_rel 1e-6, xtol_abs 1e-6, ftol_rel 0,
    ## ftol_abs 0, max_eval 0, max_time 0) into the local optimizer, where before NLopt's own
    ## defaults applied. So "the same inner algorithm, said out loud" is a DIFFERENT
    ## configuration from leaving it unset, and a sweep wanting an unset baseline must field
    ## one explicitly rather than assume a named algorithm reproduces it.
    ## ADOPTED 2026-09-19 by Thomas: the NLopt column's fielded configuration is
    ## LD_AUGLAG with an LD_MMA inner optimizer truncated by loose inner tolerances. On the
    ## criterion that decides this column -- feasibility, cost secondary -- it is better on
    ## 5 of 12 rows, worse on 1 (Panda pose native, -5 cells one-directionally) and unchanged
    ## on 6 rows where nothing works at all, and it collapses the median residual (Panda
    ## contained grasp 3.7e-04 -> 6.8e-07) and the network work per cell (3532 -> 66).
    ## It FAILED stage NLOPTTUNE's pre-registered gate, which asked whether to spend 480-cell
    ## compute on it, NOT whether it is the best configuration; those are different questions
    ## and only the second is what adoption answers. Report the gate failure alongside it.
    ##
    ## REQUIRES a Drake carrying PR 25002: `local_optimizer_ftol_rel` is absent from 1.56.0
    ## (6 options) and from this workstation's source build (11), and present only on the
    ## nightly (16). An NLopt run on an older Drake now raises from CheckNloptOptions naming
    ## the missing option; IPOPT and SNOPT runs are unaffected, because the availability
    ## check fires only for `which_solver == "nlopt"`.
    nlopt_local_optimizer_algorithm: str = field(default="LD_MMA", metadata={"help": "NLopt 'local_optimizer_algorithm': the AUGLAG inner solver. Drake's build can run LD_MMA, LD_CCSAQ, LD_SLSQP; the LD_LBFGS/LD_VAR*/LD_TNEWTON* family is compiled out (Luksan, LGPL) and is refused. None or '' leaves NLopt's own choice, as every archived cell ran; required before any nlopt_local_optimizer_* below. Post-1.56.0"})
    nlopt_local_optimizer_xtol_rel: float = field(default=1e-3, metadata={"help": "NLopt 'local_optimizer_xtol_rel' (Drake's inner default 1e-6). Inert unless the inner algorithm is named -- refused, not ignored. Post-1.56.0"})
    nlopt_local_optimizer_xtol_abs: float = field(default=None, metadata={"help": "NLopt 'local_optimizer_xtol_abs' (Drake's inner default 1e-6). Inert unless the inner algorithm is named. Post-1.56.0"})
    nlopt_local_optimizer_ftol_rel: float = field(default=1e-3, metadata={"help": "NLopt 'local_optimizer_ftol_rel' (Drake's inner default 0, disabled). Inert unless the inner algorithm is named. Post-1.56.0, and absent from this workstation's source build too"})
    nlopt_local_optimizer_ftol_abs: float = field(default=None, metadata={"help": "NLopt 'local_optimizer_ftol_abs' (Drake's inner default 0, disabled). Inert unless the inner algorithm is named. Post-1.56.0, and absent from this workstation's source build too"})
    ## The one inner knob with a mechanism rather than a tolerance behind it: a positive cap
    ## TRUNCATES each subproblem, so the outer augmented Lagrangian updates its multipliers
    ## more often instead of solving the first subproblem to convergence. 0 means no cap and
    ## is a REAL value here, not "unset" -- which is why CheckNloptOptions tests `is None`
    ## rather than truthiness.
    nlopt_local_optimizer_max_eval: int = field(default=None, metadata={"help": "NLopt 'local_optimizer_max_eval' (Drake's inner default 0 = unlimited); a positive value truncates each subproblem so the outer AL updates multipliers more often. Inert unless the inner algorithm is named. Post-1.56.0"})
    nlopt_local_optimizer_max_time: float = field(default=None, metadata={"help": "NLopt 'local_optimizer_max_time' in seconds (Drake's inner default 0 = no cap). Inert unless the inner algorithm is named. Post-1.56.0"})


    ## Solver options ##
    which_solver: str = field(default="ipopt", metadata={"help": "Which IKFlow solver to use"})
    acceptable_tol: float = field(default=1e-4, metadata={"help": "Acceptable tolerance for solver convergence"})
    acceptable_dual_inf_tol: float = field(default=1e-4, metadata={"help": "Acceptable dual infeasibility tolerance for solver convergence"})
    acceptable_compl_inf_tol: float = field(default=1e-4, metadata={"help": "Acceptable complementary infeasibility tolerance for solver convergence"})
    acceptable_constr_viol_tol: float = field(default=1e-6, metadata={"help": "Acceptable constraint violation tolerance for solver convergence"})
    acceptable_iter: int = field(default=1, metadata={"help": "Acceptable number of iterations for solver convergence"})
    file_print_level: int = field(default=5, metadata={"help": "File print level for the solver"})
    file_print_name: str = field(default="ikflow_solver_log.txt", metadata={"help": "File name for solver log"})
    max_wall_time: float = field(default=60, metadata={"help": "Maximum wall time for the solver in seconds"})
    # Which discrete branch set the Panda analytic chart uses: 4 is the historical chart
    # (elbow branch pinned, the half far from the joint limits); 8 adds the mirrored elbow
    # branch, taking round-trip coverage from 89.4% to 99.4% of random configurations. The
    # default stays 4 so archived runs remain reproducible; the residual ~0.6% at 8 is a
    # measured property of this chart, left as future work (arXiv:2503.03992 may help).
    analytic_branches: int = 4
    # Restores the pre-2026-08-31 paired start, in which the conditioning pose was clipped
    # into its box before the solve and the correction started at zero -- leaving the
    # learned arm 1.2-3.3 rad from the shared q_init. The repaired default sets the guess
    # exactly (Drake accepts an infeasible initial guess; IPOPT projects bounds itself),
    # for reproducing archived runs only.
    legacy_paired_start: bool = False
    # Degrade the flow's chart by a deterministic smooth perturbation of this magnitude
    # (rad, per-joint sin features of [c; z]); 0 disables. Experimental knob for the
    # chart-error dose-response -- see MakeFlowInference.
    chart_error_scale: float = 0.0
    snopt_function_precision: float = field(default=None, metadata={"help": "SNOPT 'Function precision'. Leave None in float64; set ~1e-6 if evaluating the flow in float32"})

    ## Gradient regularization for the flow Jacobian ##
    # The flow's worst-case gain is ~1e13 (12 coupling blocks at rnvp_clamp 2.5), and the
    # runaway cells are attracted to these regions because the Jacobian there is as large
    # as the returned configuration. These knobs damp the Jacobian before the chain rule.
    jacobian_max_norm: float = field(default=None, metadata={"help": "Clip the Jacobian's Frobenius norm to this value; None disables"})
    jacobian_tikhonov_lambda: float = field(default=0.0, metadata={"help": "Tikhonov/LM damping: replace singular values s with s*λ/(s+λ), bounded by λ; 0 disables"})
    jacobian_svd_floor: float = field(default=0.0, metadata={"help": "Floor on singular values (pseudoinverse-style truncation); 0 disables"})

    ## The svgd solver ##
    ## A FOURTH METHOD CLASS on the solver axis: batch-parallel SVGD-style particle methods
    ## in torch, on the GPU (reopened by Thomas, 2026-10-08). Every field here is prefixed
    ## `svgd_` and is INERT unless `which_solver == "svgd"`: none is read by `_IpoptOptions`,
    ## `_SnoptOptions` or `_NloptOptions`, so a Drake column emits exactly what it always
    ## has (tests/test_solver_plumbing.py pins that). `use_float64` is untouched and governs
    ## the Drake path only; the swarm's dtype is `svgd_dtype`, and the Drake re-check is
    ## always float64. The enum fields are validated in `__post_init__` (CheckSvgdOptions) so a bad
    ## `--set` dies before the first cell. Everything actually handed to the solver is
    ## recorded in `emitted_solver_options["svgd"]` like the Drake solvers' options are.
    ##
    ## NO FORMULATION-SPECIFIC INFORMATION is encoded by any of these (Thomas, 2026-10-08).
    ## The solver's target is exactly the program as written, -L_rho/T: no latent-prior
    ## term, no assumption that the correction or any residual is zero-centred, no kernel or
    ## schedule that knows what one arm's variables mean. What IS allowed is
    ## formulation-agnostic: a kernel in configuration space (every arm has a q), row
    ## scaling by tolerance, coordinates normalised by each region's half-width (read from
    ## the program's own options and bounds), and the native start's own draw.
    ##
    ## PLAIN AUGMENTED-LAGRANGIAN SVGD, nothing else (Thomas, 2026-10-09): every field below
    ## names one part of the method in `docs/svgd-solver.md`; the removed pieces are listed
    ## there under "Removed, 2026-10-09".
    svgd_method: str = field(default="al_svgd", metadata={"help": "svgd: the method -- 'al_svgd', plain augmented-Lagrangian SVGD (docs/svgd-solver.md), the only one. Named in full in every table"})
    svgd_n: int = field(default=64, metadata={"help": "svgd: number of particles N -- a METHOD PARAMETER, reported everywhere; N = 1 is the degenerate single-particle control"})
    svgd_dtype: str = field(default="float32", metadata={"help": "svgd: the swarm's torch dtype, 'float32' (fielded) or 'float64' (a control). The Drake re-check is always float64"})
    svgd_seed: int = field(default=0, metadata={"help": "svgd: base seed for the particle draws (combined with the cell's initial guess)"})
    svgd_paired_init: str = field(default="jitter", metadata={"help": "svgd: how the swarm is drawn around the program's initial guess -- 'jitter' adds svgd_jitter * N(0, I) in the normalised coordinates (particle 0 is the guess, never clipped), 'native' draws the arm's own start distribution"})
    svgd_jitter: float = field(default=0.1, metadata={"help": "svgd: paired-jitter sigma, as a fraction of each variable's region half-width (the normalised coordinates)"})
    svgd_temperature: float = field(default=1.0, metadata={"help": "svgd: the temperature T of the target exp(-L_rho / T); fixed for the whole solve"})
    svgd_lr: float = field(default=SVGD_LR_DEFAULT, metadata={"help": "svgd: the step size; particle i steps svgd_lr / rho_i along its Stein direction in the normalised coordinates (plain gradient step, no momentum, no adaptation)"})
    svgd_kernel: str = field(default="q", metadata={"help": "svgd: 'q' (RBF kernel on the configuration, median bandwidth) or 'none' (no interaction: both kernel terms dropped, the batched-AL control)"})
    svgd_bandwidth_floor: float = field(default=0.05, metadata={"help": "svgd: floor on the median-heuristic kernel bandwidth"})
    svgd_constraint_inside_kernel: bool = field(default=False, metadata={"help": "svgd: False is the Tabor-Hermans form (each particle's own constraint gradient outside the kernel average); True the literal SVGD on exp(-L_rho / T), the whole AL gradient inside it -- an A/B"})
    svgd_rho: float = field(default=10.0, metadata={"help": "svgd: the penalty rho of the augmented Lagrangian -- ONE scalar, fixed for the whole solve, shared by every particle; also the dual-ascent step"})
    svgd_multiplier_max: float = field(default=1e4, metadata={"help": "svgd: box on the magnitude of every multiplier; the clipped entries are counted"})
    svgd_inner_iters: int = field(default=10, metadata={"help": "svgd: K -- every K steps the unconditional dual-ascent step on every particle, runaway resampling, the stop rule and the wall clock"})
    svgd_outer_iters: int = field(default=300, metadata={"help": "svgd: cap on outer checks (the 'iterations' column); `max_iter`, when set, also binds"})
    svgd_resample_q_max: float = field(default=10.0, metadata={"help": "svgd: |q|_inf (rad) above which a particle counts as runaway and is redrawn from the arm's native start at the next outer check (counted as n_resampled)"})
    svgd_recheck_topk: int = field(default=10, metadata={"help": "svgd: how many particles feasible on the batched rows, in objective order, get the exact Drake `EvalBinding` re-check; the first passer is returned"})
    svgd_time_reserve: float = field(default=0.1, metadata={"help": "svgd: fraction of `max_wall_time` held back from the swarm for the selection and the Drake re-check"})
    svgd_stop_patience: int = field(default=5, metadata={"help": "svgd: once a particle is feasible at the gate, stop (converged) after this many consecutive outer checks at which the best feasible particle's objective improved by less than svgd_stop_rel"})
    svgd_stop_rel: float = field(default=1e-3, metadata={"help": "svgd: the relative improvement of the best feasible objective that counts as progress for the stop rule"})
    svgd_warmup: str = field(default="none", metadata={"help": "svgd: 'none' or 'cem' -- a forward-only Cross-Entropy phase on the penalised merit before the gradient phase; a phase of the population method, named in full in every table and A/B tested"})
    svgd_warmup_iters: int = field(default=10, metadata={"help": "svgd: CEM warm-up iterations"})
    svgd_warmup_elite: float = field(default=0.1, metadata={"help": "svgd: CEM elite fraction"})
    svgd_collision_workers: int = field(default=None, metadata={"help": "svgd: processes in the exact-collision pool; None resolves inside the solver to os.cpu_count() // PROCS in a Slurm job (SLURM_JOB_ID set) and to min(os.cpu_count() // PROCS, 8) elsewhere. Every pool is admitted by the memory guard (src/svgd/collision_backend.py: workers x 0.75 GB <= half of MemAvailable, live workers <= SVGD_MAX_LIVE_WORKERS)"})
    svgd_compile: bool = field(default=False, metadata={"help": "svgd: torch.compile the split step's three stages (src/svgd/fused.py; compiled by WarmUpSvgdStep, outside any timed cell)"})
    svgd_cuda_graph: bool = field(default=False, metadata={"help": "svgd: replay the compiled stages as CUDA graphs, captured by WarmUpSvgdStep before the first timed cell (requires svgd_compile and CUDA)"})
    svgd_pool_overlap: bool = field(default=True, metadata={"help": "svgd: dispatch the collision batch to the pool BEFORE the flow Jacobian and the kinematics are launched, collecting it after, so Drake's IPC and compute run behind the GPU work; False collects at once (the measurement's control)"})

    vars_file: str = field(default=None, metadata={"help": "If provided, saves variable trajectories to this file"})
    visualize: bool = field(default=False, metadata={"help": "If true, visualizes the IK solving process in Meshcat"})

    def __post_init__(self):
        ## The NLopt surface is checked HERE, at options construction, and not only where the
        ## options are emitted. _NloptOptions runs inside Solve(), and Solve() runs inside
        ## run_grid's per-cell `except Exception` -- so a raise from there is caught, written
        ## into the record as fail_reason="error", and the sweep proceeds into a full column
        ## of instant failures. A raise from here happens in `ProgramOptions(...)` and in the
        ## `replace()` that apply_overrides performs for `--set`, both before the first cell
        ## and outside any try. It is free: options are constructed a handful of times per
        ## run, never per cell.
        ##
        ## Deliberately NOT gated on `which_solver == "nlopt"`. A manifest that sets an NLopt
        ## field on an IPOPT arm is a broken sweep and should say so rather than run.
        CheckNloptOptions(self)
        ## Same rule, same place, for the svgd enum fields.
        CheckSvgdOptions(self)
        ## Graphing the eager jacrev is not possible (it allocates its basis offsets with a
        ## host-to-device copy every call, illegal under capture), and graphing anything
        ## else would be a third configuration nobody measured.
        if self.flow_cuda_graph and not self.compile_flow_jacobian:
            raise ValueError("flow_cuda_graph requires compile_flow_jacobian (--compile)")



# ----------------------------- orientation error ------------------------------
#
# The orientation half of the IK pose constraint: three signed rows of the roll-pitch-yaw
# residual, the same shape as the position rows and the same quantity ../codebase's
# EEPoseConstraint imposes. ProgramOptions.orientation_error_form picks the bounds:
#
#   rpy          residual == 0        pinned, as ../codebase pins it  [default]
#   rpy_boxed    |residual| <= ori_tol
#
# Earlier revisions measured the mismatch as a single scalar angle, 2*arccos(|q.q_t|).
# That shape is what put a branch point at zero error -- taking a norm of a
# three-component error is exactly the operation that manufactures one -- and its eps
# clamp additionally handed back an AutoDiffXd with an empty derivative vector at
# convergence. Three signed rows have neither problem: the residual is smooth at the
# solution and its Jacobian is full rank there. See git history (0be5342) for the scalar
# forms and the measurements that retired them.
#
# Only the pose-target programs reach this (PandaIKProgram, Iiwa14IKProgram and their
# ...Numerical variants); the mug programs override CreateIKConstraint, and the analytic
# programs never add an IK constraint at all.


def orientation_error_rpy(orientation, target_rpy):
    '''Signed roll-pitch-yaw residual, rpy(R) - rpy(R_target), wrapped to (-pi, pi].

    The chart degenerates at pitch = +-pi/2 (gimbal lock) and each component wraps at
    +-pi. Wrapping the residual, rather than differencing raw angles, removes the 2*pi
    jump for targets sitting near a branch; it shifts by a constant, so derivatives are
    untouched. Gimbal lock is a property of the *target pose*, not of the error, so it
    does not degrade behaviour at convergence.

    `orientation` is the achieved orientation as a wxyz quaternion, plain or AutoDiffXd;
    `target_rpy` is the target's roll-pitch-yaw as plain floats, precomputed by the caller.
    '''
    ad = isinstance(orientation[0], AutoDiffXd)
    dtype = AutoDiffXd if ad else float
    achieved = RollPitchYaw_[dtype](RotationMatrix_[dtype](Quaternion_[dtype](orientation)))
    residual = achieved.vector() - np.asarray(target_rpy)
    # Wrap each component into (-pi, pi] by subtracting a constant multiple of 2*pi.
    for i in range(3):
        value = residual[i].value() if ad else float(residual[i])
        turns = np.round(value / (2.0 * np.pi))
        if turns != 0.0:
            residual[i] = residual[i] - 2.0 * np.pi * turns
    return residual


ORIENTATION_ERROR_FORMS = ("rpy", "rpy_boxed")

## ---------------------------- the flow evaluation ----------------------------- ##
#
# The network forward pass lives here as a *free function of the lumped variables* rather
# than as a method, for one reason: torch.compile guards on everything the compiled callable
# closes over, so a bound method would carry the program instance into the guards and each of
# the thirty programs in a benchmark grid would re-trigger dynamo. Closing over the network
# alone lets one graph be compiled once per process and reused by every program that shares
# it. (This is what the old "200 ms compilation penalty per program" comment was measuring:
# the penalty was per instance because the compiled thing was per instance.)


def MakeFlowInference(nn_model, width, num_arm_dof, device, chart_error_scale=0.0):
    """`vars -> (q, q)`, the shape `jacrev(..., has_aux=True)` wants.

    Returning q twice is what lets one reverse pass yield both dq/dvars and q, instead of
    evaluating the network again for the value. `vars` is [conditioning pose as xyz + wxyz,
    latent, correction]; the trailing zero on the conditioning row is the padding the flow
    was trained with.

    `chart_error_scale` adds a deterministic, smooth, seeded perturbation
    `eps * sin(W [c; z] + b)` to the network's output -- an experimental knob that
    degrades the chart's accuracy without touching anything else, so success can be
    measured against chart error with the scene, kinematics and solver held fixed (the
    dose-response experiment: the Panda's 3.8 mm chart pushed through the iiwa's
    16.6-64 mm regime). It is a function of the conditioning pose and latent only, so the
    correction's analytic identity block in the Jacobian is untouched.
    """
    if chart_error_scale:
        # Explicit device="cpu": ikflow installs a global default-device override that
        # would otherwise pair the CPU generator with a CUDA allocation and fail.
        gen = torch.Generator(device="cpu").manual_seed(0)
        W = torch.randn((7 + width, num_arm_dof), generator=gen,
                        dtype=torch.float64, device="cpu").to(device)
        b = torch.randn(num_arm_dof, generator=gen,
                        dtype=torch.float64, device="cpu").to(device)

    def flow_inference(vars):
        c, z, correction = vars[:7], vars[7:7 + width], vars[7 + width:]
        c_torch = torch.cat(
            [c.unsqueeze(0), torch.zeros((1, 1), dtype=vars.dtype, device=device)], dim=1)
        output, _ = nn_model(z.unsqueeze(0), c=c_torch, rev=True)
        q = output[0, :num_arm_dof] + correction
        if chart_error_scale:
            q = q + chart_error_scale * torch.sin(
                vars[:7 + width].to(W.dtype) @ W + b).to(q.dtype)
        return q, q
    return flow_inference


_COMPILED_JACOBIANS = {}

## ------------------------------ CUDA-graph replay ------------------------------ ##
#
# At batch 1 a compiled flow call is still ~290 kernel launches described one at a time from
# Python; a CUDA graph records them once and replays them in one launch. What it bakes in:
# the kernels, the parameters' and buffers' ADDRESSES, and the input/output buffers. So the
# network must not be re-cast or have its parameters reassigned after capture (`.to()` to the
# dtype it already has is a no-op, which is why ConfigureNetworkDtype stays safe), and dynamo's
# guards are evaluated at capture only -- replay checks nothing. Keyed like the compiled
# Jacobian, so one graph per process serves every program in a grid.

_FLOW_GRAPHS = {}
## Set by WarmUpJacobian once the graphs exist. A capture after that point would be paid
## inside a timed solve -- compile and capture are offline compute and must not be -- so it
## raises instead of quietly eating the cap.
_FLOW_GRAPHS_FROZEN = False


def FreezeFlowGraphs(frozen=True):
    global _FLOW_GRAPHS_FROZEN
    _FLOW_GRAPHS_FROZEN = frozen


class GraphedFlowCall:
    """`fn` replayed as a CUDA graph; same signature, returns a tuple of fresh tensors.

    Captured lazily, on the first call, at that call's input shape; any other shape, or a
    CPU device, falls through to `fn`. Outputs are cloned out of the graph's static buffers,
    because the next replay overwrites them.
    """

    def __init__(self, fn, grad_enabled):
        self.fn = fn
        self.grad_enabled = grad_enabled
        self.graph = None
        self.captures = 0

    def _capture(self, example):
        if _FLOW_GRAPHS_FROZEN:
            raise RuntimeError(
                "a flow CUDA graph would be captured inside a timed solve; WarmUpJacobian "
                "must capture every graph the grid uses (is a program holding a different "
                "network, dtype or input width?)")
        mode = torch.enable_grad if self.grad_enabled else torch.no_grad
        self.static_in = example.detach().clone()
        current = torch.cuda.current_stream(example.device)
        side = torch.cuda.Stream(device=example.device)
        side.wait_stream(current)
        with torch.cuda.stream(side), mode():       # warm up (and compile) off the capture
            for _ in range(3):
                self.fn(self.static_in)
        current.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), mode():
            out = self.fn(self.static_in)
        self.static_out = out if isinstance(out, tuple) else (out,)
        torch.cuda.synchronize(example.device)
        self.graph = graph
        self.captures += 1

    def __call__(self, x):
        if x.device.type != "cuda":
            return self.fn(x)
        if self.graph is None:
            self._capture(x)
        elif x.shape != self.static_in.shape or x.dtype != self.static_in.dtype:
            return self.fn(x)
        self.static_in.copy_(x)
        self.graph.replay()
        return tuple(t.clone() for t in self.static_out)


def _FlowKey(nn_model, width, num_arm_dof, device, chart_error_scale):
    return (id(nn_model), width, num_arm_dof, str(device),
            next(nn_model.parameters()).dtype, float(chart_error_scale))


def FlowJacobianGen(nn_model, width, num_arm_dof, device, compile_it, chart_error_scale=0.0,
                    cuda_graph=False):
    """`vars -> (dq/dvars, q)`, compiled once per (network, shape, dtype) if asked, and
    replayed as a CUDA graph on top of that if asked.

    Reverse mode is the right primitive at this shape -- 7 outputs against 21 inputs, of
    which 13 reach the network -- and the measurements behind that are in CLAUDE.md.
    """
    jacobian_gen = torch.func.jacrev(
        MakeFlowInference(nn_model, width, num_arm_dof, device, chart_error_scale),
        has_aux=True)
    if not compile_it:
        return jacobian_gen
    key = _FlowKey(nn_model, width, num_arm_dof, device, chart_error_scale)
    if key not in _COMPILED_JACOBIANS:
        _COMPILED_JACOBIANS[key] = torch.compile(jacobian_gen)
    if not cuda_graph:
        return _COMPILED_JACOBIANS[key]
    graph_key = key + ("jacobian",)
    if graph_key not in _FLOW_GRAPHS:
        _FLOW_GRAPHS[graph_key] = GraphedFlowCall(_COMPILED_JACOBIANS[key], grad_enabled=True)
    return _FLOW_GRAPHS[graph_key]


def FlowValueGen(nn_model, width, num_arm_dof, device, chart_error_scale=0.0):
    """`vars -> (q, q)` for the plain forward pass: compiled, under no_grad, CUDA-graphed.

    Used only under `flow_cuda_graph`; otherwise the forward pass stays the eager
    `MakeFlowInference`, so a run without the switch is exactly what it always was.
    """
    key = _FlowKey(nn_model, width, num_arm_dof, device, chart_error_scale) + ("value",)
    if key not in _FLOW_GRAPHS:
        infer = MakeFlowInference(nn_model, width, num_arm_dof, device, chart_error_scale)

        def forward(vars):
            with torch.no_grad():
                return infer(vars)[0]
        graphed = GraphedFlowCall(torch.compile(forward), grad_enabled=False)

        def value(vars):
            (q,) = graphed(vars)
            return q, q
        value.graphed = graphed
        _FLOW_GRAPHS[key] = value
    return _FLOW_GRAPHS[key]


def regularize_jacobian(jacobian_np, options):
    """Apply gradient regularization to the flow Jacobian before the chain rule.

    The flow's worst-case gain is ~1e13 (12 coupling blocks at rnvp_clamp 2.5), and the
    runaway cells are attracted to these regions because the Jacobian there is as large
    as the returned configuration. These knobs damp the Jacobian so the solver sees
    bounded gradients while the value `q` is unchanged.

    Three strategies, applied in order if enabled:
    - jacobian_max_norm: clip the Frobenius norm (isotropic)
    - jacobian_tikhonov_lambda: Tikhonov/LM damping on singular values (anisotropic)
    - jacobian_svd_floor: floor on singular values (pseudoinverse-style)
    """
    if options.jacobian_max_norm:
        norm = np.linalg.norm(jacobian_np)
        if norm > options.jacobian_max_norm:
            jacobian_np = jacobian_np * (options.jacobian_max_norm / norm)

    if options.jacobian_tikhonov_lambda > 0 or options.jacobian_svd_floor > 0:
        U, s, Vh = np.linalg.svd(jacobian_np, full_matrices=False)
        if options.jacobian_tikhonov_lambda > 0:
            # Tikhonov/LM: s_damped = s * λ / (s + λ)
            # As s → ∞, s_damped → λ (bounded); as s → 0, s_damped → 0 (no amplification)
            lam = options.jacobian_tikhonov_lambda
            s = s * lam / (s + lam)
        if options.jacobian_svd_floor > 0:
            s = np.maximum(s, options.jacobian_svd_floor)
        jacobian_np = U @ np.diag(s) @ Vh

    return jacobian_np


class IKFlowConstraints:
    def __init__(self, lb, ub, eval_func, description=""):
        self.lb = lb
        self.ub = ub
        self.eval_func = eval_func
        self.description = description
    def __len__(self):
        return len(self.lb)


class IKFlowProgram:
    def __init__(self, diagram, frame, solver, options=ProgramOptions()):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context)

        self.frame = frame
        self.autodiff_frame = self.autodiff_plant.GetBodyByName(frame.name()).body_frame()

        self.ik_solver = solver
        self.ik_solver.nn_model.eval()
        self.options = options

        self.constraints = []

    def add_constraints(self):
        self.CreateIKConstraint()
        if self.options.collision_avoidance:
            self.CreateCollisionFreeConstraint()
        ## Under `lift_q` the limits are the lifted variable's own bounding box, so the
        ## generic row would duplicate them -- and duplicating an active constraint is
        ## exactly the rank-deficiency this repo has been bitten by before.
        if self.options.joint_limits and not self._LiftingQ():
            self.CreateJointLimitsConstraint()
        if self._LiftingQ():
            self.CreateFlowConsistencyConstraint()
        self.ApplyConstraints()
        self.BoundingBoxConstraint()
        ## Added here rather than inside BoundingBoxConstraint because the mug programs
        ## override that method, and the last time a box lived in three places the pose
        ## arms were repaired and the grasp arms silently were not.
        if self._LiftingQ():
            self.LiftedQBoxConstraint()
        # Only the learned formulations have a latent; the joint-space and analytic arms
        # share this options object so that budgets and tolerances stay identical between
        # them, which means options that name learned-only variables must be guarded.
        if hasattr(self, "z") and self.LatentTrustRadius() is not None:
            self.LatentTrustRegion()

    def add_costs(self):
        if self.options.joint_centering_cost > 0.0:
            self.JointCenteringCost()
        ## `correction` is a learned-only decision variable, so this must be guarded the
        ## same way `latent_cost_weight` and `latent_trust_region` are -- the baseline
        ## programs share this options object. Ungated, a `--set correction_cost_weight`
        ## run raises AttributeError inside every numerical/analytic program's
        ## construction and scores that whole column 0 in about 10 ms per cell.
        if self.options.correction_cost_weight > 0.0 and hasattr(self, "correction"):
            self.CorrectionCost()
        if self.options.latent_cost_weight > 0.0 and hasattr(self, "z"):
            self.LatentCost()
        ## Guarded like the others: it reaches `q` through the flow, so it means nothing to
        ## an arm whose variables *are* the configuration (and whose limits are already
        ## bounds). Pointless rather than fatal there, but the guard keeps the baselines
        ## bit-identical, which is what the A/B test checks.
        if self.options.joint_limit_penalty_weight > 0.0 and hasattr(self, "z"):
            self.JointLimitPenaltyCost()

    def LatentCost(self):
        '''A quadratic pull towards the centre of the latent prior.

        Excluded from the reported objective by name, the way ../codebase excludes its
        barrier terms, so the learned column of a cost table still measures the same
        objective as the other formulations.'''
        width = self.ik_solver.network_width
        self.latent_cost = self.prog.AddQuadraticCost(
            Q=self.options.latent_cost_weight * np.eye(width),
            b=np.zeros(width), vars=self.z)
        self.latent_cost.evaluator().set_description("LatentRegularizerCost")

    def LatentTrustRegion(self):
        '''`||z||^2 <= r^2`, imposed as an inequality rather than a per-component box.

        The prior mass sits on a shell of radius sqrt(latent_dim), and a per-component box
        of +-5 admits norms far outside it. An inequality is deliberate: its gradient,
        `2 z`, does not vanish where the constraint is active, so it does not reproduce
        the degenerate active set that a norm-residual equality would create.'''
        radius = self.LatentTrustRadius()
        width = self.ik_solver.network_width
        self.latent_trust_constraint = self.prog.AddQuadraticConstraint(
            2.0 * np.eye(width), np.zeros(width), -np.inf, radius ** 2, self.z)
        self.latent_trust_constraint.evaluator().set_description("LatentTrustRegion")

    def LatentTrustRadius(self):
        '''The trust-region radius this program imposes, or None for no region.

        Under `latent_trust_region_rule` it is `round(sqrt(dim_latent) + 1.5, 2)` computed
        from the LOADED chart's width -- the one place that width is known for certain --
        so one flag sizes every robot's region by the same rule. Otherwise it is the
        per-robot constant in `latent_trust_region` (None keeps the +-5 box only).'''
        if getattr(self.options, "latent_trust_region_rule", False):
            return float(round(np.sqrt(self.ik_solver.network_width) + 1.5, 2))
        radius = self.options.latent_trust_region
        return None if radius is None else float(radius)

    ## ---------------- settings unified across robots (2026-10-08) ----------------- ##
    #
    # Each method below is the ONE place a formerly per-robot setting lives; a subclass
    # calls it and carries nothing but its robot-specific constant (a home pose, a legacy
    # bound). `legacy_robot_settings` restores the old behaviour INSIDE the same method, so
    # the control and the unified arm cannot drift apart by a copy being missed -- the
    # lesson of the latent box, repaired once in this file while three mug-program copies
    # kept the bug.

    #: Mug programs whose LEGACY native start seeded `c` at the mug centre with zero rpy
    #: (the iiwa and screw arms) set this True; the others always seeded the flow-frame
    #: grasp pose. Read only under `legacy_robot_settings`.
    LEGACY_GRASP_C_MUG_CENTRED = False

    def NominalConfiguration(self):
        '''The joint-centering target: a nonsingular home pose, in CONFIGURATION space.

        Zeros by default, which is right for the soft arms (the straight unstretched rod
        is their natural nominal) and wrong for every rigid S-R-S arm, where the straight
        configuration is singular -- and on the Panda outside q4's range. The rigid
        programs override this with a per-robot constant held in one place.'''
        return np.zeros(self.num_arm_dof)

    def ResolveQNominal(self, q_nominal):
        '''`q_nominal` as `create_prog` stores it: the caller's if given, else the robot's
        home pose, else -- under `legacy_robot_settings` -- zeros.'''
        if q_nominal is not None:
            return np.asarray(q_nominal, dtype=float)
        if self.options.legacy_robot_settings:
            return np.zeros(self.num_arm_dof)
        q = np.asarray(self.NominalConfiguration(), dtype=float)
        if q.shape != (self.num_arm_dof,):
            raise ValueError(f"{type(self).__name__}.NominalConfiguration() has shape "
                             f"{q.shape}; expected ({self.num_arm_dof},)")
        return q

    def QBoundingBoxConstraint(self, legacy_bounds=None):
        '''The joint-space arm's variable bound on `q`: the robot's own `ConfigLimits()`.

        A variable bound rather than a general constraint is right here, because a start
        never violates it -- `q_init` is drawn inside the limits. `legacy_bounds` is what
        this robot's program used before unification (the Panda's +-10 rad, the iiwa's
        hand-typed table), applied only under `legacy_robot_settings`.'''
        if self.options.legacy_robot_settings and legacy_bounds is not None:
            lower, upper = (np.asarray(b, dtype=float) for b in legacy_bounds)
        else:
            lower, upper = (np.asarray(b, dtype=float)[:self.num_arm_dof]
                            for b in self.ConfigLimits())
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            ## Drake accepts an infinite bounding box without comment; on the screw arm
            ## that is exactly the parser's silent +-inf, so refuse it here.
            raise RuntimeError(f"non-finite q bounds {lower} .. {upper}: the plant's "
                               f"limits were not repaired before the program was built")
        self.q_bounds = (lower, upper)
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(lower, upper, self.q)
        self.bounding_box_constraint.evaluator().set_description("QBoundingBoxConstraint")
        return self.bounding_box_constraint

    def GraspFlowPose(self):
        '''The flow-frame pose of the configuration that GENERATED the target mug.

        The mug is welded at the grasp frame of a sampled configuration, so
        `mug.middle @ X_grasp_ee` is exactly the conditioning pose that configuration has
        -- a pose known to admit a valid grasp. The PI accepts (2026-10-08) that the native
        start and the `c` box are both anchored on it.'''
        return self.target_mug.middle @ self.X_grasp_ee

    def GraspCStart(self):
        '''`c`'s native initial guess on the grasp task, as xyz + rpy.

        The flow-frame grasp pose, not the mug centre: `c` is the pose of the frame the
        flow was trained on, which sits `X_grasp_ee` (0.1-0.2 m) behind the grasp point,
        so seeding at the mug is what once put the Panda's `c` 120 degrees from where the
        network expects it.'''
        if self.options.legacy_robot_settings and self.LEGACY_GRASP_C_MUG_CENTRED:
            return np.array([*self.target_mug.middle.translation(), 0.0, 0.0, 0.0])
        X = self.GraspFlowPose()
        return np.concatenate([X.translation(), X.rotation().ToRollPitchYaw().vector()])

    def GraspCBoxConstraint(self):
        '''`c`'s region on the grasp task: +-c_position_slack about the flow-frame grasp
        pose's position, +-2*pi on each rpy (orientation free, no wraparound clipping).

        Centred on the FLOW frame's pose rather than on the mug: `c` is the flow frame,
        which sits a per-robot standoff (0.10-0.20 m, `X_grasp_ee`) behind the grasp
        point, so a mug-centred box left 0.05-0.15 m of per-axis margin depending on the
        gripper. Centred here, every robot gets the same +-0.25.

        A general linear constraint, deliberately NOT a bounding box, and the distinction
        is load-bearing: IPOPT's bound_push projects the initial guess into every variable
        box before evaluating anything, which silently destroyed the exact paired start
        (`c` teleported to the box face while the latent stayed tuned to the unprojected
        pose). A general constraint may start violated -- the violation lands in inf_pr --
        and the solver walks `c` in continuously while `z` and the correction adapt.'''
        slack = self.options.c_position_slack
        centre = (self.target_mug.middle.translation() if self.options.legacy_robot_settings
                  else self.GraspFlowPose().translation())
        self.c_box = (np.concatenate([centre - slack, -2 * np.pi * np.ones(3)]),
                      np.concatenate([centre + slack, 2 * np.pi * np.ones(3)]))
        self.c_box_constraint = self.prog.AddLinearConstraint(
            np.eye(6), self.c_box[0], self.c_box[1], self.c)
        self.c_box_constraint.evaluator().set_description("CBoxConstraint")
        return self.c_box_constraint

    def PoseCBoxConstraint(self):
        '''`c`'s region on the pose task: +-1 (metres on xyz, radians on rpy) about the
        target's flow-frame pose. A general linear constraint, for the reason given on
        `GraspCBoxConstraint`.'''
        self.c_box = (self.initial_guess - 1, self.initial_guess + 1)
        self.c_box_constraint = self.prog.AddLinearConstraint(
            np.eye(len(self.c)), self.c_box[0], self.c_box[1], self.c)
        self.c_box_constraint.evaluator().set_description("CBoxConstraint")
        return self.c_box_constraint

    def CorrectionBoxConstraint(self):
        '''The correction's +-correction_bound box. A genuine variable bound: a paired start
        sets the correction to the residual clipped into it, so no start violates it.'''
        bound = self.options.correction_bound
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            -bound * np.ones(self.num_arm_dof), bound * np.ones(self.num_arm_dof),
            self.correction)
        self.correction_bounding_box_constraint.evaluator().set_description(
            "CorrectionBoundingBoxConstraint")
        return self.correction_bounding_box_constraint

    def RobotSettings(self):
        '''What this program ACTUALLY uses of the unified settings, for a run's metadata.

        `metadata["overrides"]` records that `--set legacy_robot_settings=True` was asked
        for; this records what each program did with it -- the home pose, the q bounds,
        the grasp start and box centre, the trust radius -- so a run can be read off the
        run rather than reconstructed from the code that produced it.'''
        out = dict(legacy_robot_settings=bool(self.options.legacy_robot_settings),
                   latent_trust_region_rule=bool(
                       getattr(self.options, "latent_trust_region_rule", False)))
        if hasattr(self, "q_nominal"):
            out["q_nominal"] = [float(v) for v in np.asarray(self.q_nominal).ravel()]
        if hasattr(self, "z"):
            out["latent_trust_region"] = self.LatentTrustRadius()
        if hasattr(self, "q_bounds"):
            out["q_bounds"] = [[float(v) for v in b] for b in self.q_bounds]
        if hasattr(self, "c_box"):
            out["c_box_lower"] = [float(v) for v in self.c_box[0]]
            out["c_box_upper"] = [float(v) for v in self.c_box[1]]
        if getattr(self, "target_mug", None) is not None and hasattr(self, "c"):
            out["grasp_c_start"] = [float(v) for v in self.GraspCStart()]
            out["grasp_c_box_centre"] = ("mug" if self.options.legacy_robot_settings
                                         else "flow_pose")
        return out

    def FlowInference(self):
        """The forward pass, built once per program and shared with the compiled
        Jacobian so both paths run identical code. Eager, unless `flow_cuda_graph`
        replays the compiled pass instead."""
        fn = getattr(self, "_flow_inference", None)
        if fn is None:
            make = (FlowValueGen if getattr(self.options, "flow_cuda_graph", False)
                    else MakeFlowInference)
            fn = self._flow_inference = make(
                self.ik_solver.nn_model, self.ik_solver.network_width,
                self.num_arm_dof, DEVICE, self.options.chart_error_scale)
        return fn

    def MakeJacobianGen(self):
        return FlowJacobianGen(
            self.ik_solver.nn_model, self.ik_solver.network_width, self.num_arm_dof,
            DEVICE, self.options.compile_flow_jacobian, self.options.chart_error_scale,
            cuda_graph=getattr(self.options, "flow_cuda_graph", False))

    def WarmUpJacobian(self):
        """Pay torch.compile's one-off cost outside any timed solve.

        Returns the seconds spent. The benchmark scripts call this on the sampler program,
        which holds the same network every later program is handed, so the grid never sees
        the compile.
        """
        width = self.ik_solver.network_width
        vars = np.zeros(7 + width + self.num_arm_dof)
        vars[3] = 1.0                                  # a unit quaternion, w first
        tensor = torch.tensor(vars, dtype=self.torch_dtype, device=DEVICE)
        start = time.time()
        gen = self.MakeJacobianGen()
        gen(tensor)
        if getattr(self.options, "flow_cuda_graph", False):
            ## Both graphs, here and not in the first solve; then no more may be made.
            self.FlowInference()(tensor)
            FreezeFlowGraphs()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time() - start

    def WarmUpSvgdStep(self):
        """Pay the svgd solver's one-off compile cost outside any timed solve.

        Returns the seconds spent. The `WarmUpJacobian` pattern: the benchmark scripts call
        it on the sampler program when `which_solver == "svgd"`, so the first timed cell
        does not carry the compile.
        """
        from src.svgd.solver import SvgdSolver
        return SvgdSolver(self).warm_up()

    def PrepareSvgdSolve(self):
        """The svgd solver's per-cell infrastructure -- the batched program and this scene's
        collision pool with its workers' scenes built -- paid in a cell's SETUP, before its
        clock starts (`run_grid` calls it), as a Drake arm's scene is built in setup. No
        solver state survives into the solve except the pool. Returns the seconds spent."""
        from src.svgd.solver import SvgdSolver
        return SvgdSolver(self).prepare()

    def fk(self, q, matrix = False):
        frame, context = self.SetPositions(q)
        rigid_transform = frame.CalcPoseInWorld(context)
        if matrix:
            return rigid_transform.GetAsMatrix4()
        else:
            return rigid_transform.translation(), rigid_transform.rotation().ToQuaternion().wxyz()

    ## ---------------------- shared flow evaluation ------------------------- ##

    def _FlowCacheKey(self, vars):
        '''Key an iterate by its values *and* its derivative block.

        Drake hands the cost and the constraint the same `vars` at the same point, but an
        AutoDiffXd carries a gradient as well as a value, and the two callbacks can be
        called with different seed matrices. Keying on the value alone would silently
        return a Jacobian computed against the wrong seeds.
        '''
        if isinstance(vars[0], AutoDiffXd):
            values = np.array([v.value() for v in vars])
            derivatives = np.array([v.derivatives() for v in vars])
            return (True, values.tobytes(), derivatives.shape, derivatives.tobytes())
        return (False, np.asarray(vars, dtype=float).tobytes())

    ## The keys of `eval_counts`. Declared here so a consumer can rely on the shape even
    ## for a program that never solved.
    EVAL_COUNT_KEYS = ("map_forward", "map_jacobian", "memo_hits", "callback")

    def ResetEvalCounts(self):
        '''Zero the cross-solver evaluation counters. Called at the top of `Solve`.

        These exist because **no solver reports an iteration count through Drake**.
        `SnoptSolverDetails` carries `info`, `solve_time` and the multipliers but no
        iteration count, and `NloptSolverDetails` carries a single `status` field and
        nothing else -- NLopt writes no log of any kind and ignores `kPrintFileName`. So
        the only quantity available under *every* solver is one we count ourselves.

        `map_jacobian` is the number this project already says to quote when comparing
        formulations: for the learned arm it is one `jacrev` through the network each, the
        dominant cost of the solve. Counting it here rather than per solver also makes the
        arms comparable -- the numerical and analytic arms evaluate Drake kinematics
        through the same funnel -- and cross-validates IPOPT's own logged counts.

        Deliberately NOT an iteration count. A line search evaluates the map several times
        per accepted step, and each solver does so differently, so this measures work done
        rather than steps taken. Where a solver does report iterations (IPOPT and SNOPT,
        both via their print file) that is reported separately and is the number to compare
        across solvers.

        **Under the `svgd` solver one count is one BATCHED pass**, i.e. one evaluation of
        the map over all N particles at once, not one particle. The per-particle work is
        `N x passes`, recoverable because `n_particles` is recorded in the svgd details
        (`record["svgd"]`) beside the counts. The batched solver bumps these counters
        itself, through the same keys; `map_jacobian` is a batched Jacobian pass and
        `map_forward` a batched forward pass. Comparing a batched count against a Drake
        solver's count is comparing passes, not particles -- say so wherever both appear.
        '''
        self.eval_counts = {k: 0 for k in self.EVAL_COUNT_KEYS}

    def QAndPose(self, vars):
        '''`(q, pose)` for an iterate, evaluating the flow at most once per point.

        The forward pass and the `jacrev` are the dominant cost of the learned
        formulation -- everything else in a `VarsToQ` evaluation is about 0.01 ms -- so
        sharing them between the constraint binding and the cost binding is close to a
        factor of two on the whole solve.

        This is also the single funnel every arm's solve passes through -- the constraint
        block, the joint-limit row and the joint-centering cost all call it -- which is why
        the evaluation counters live here. See `ResetEvalCounts`.
        '''
        counts = getattr(self, "eval_counts", None)
        if counts is None:
            self.ResetEvalCounts()
            counts = self.eval_counts
        ## Under `lift_q` the configuration IS a decision variable, so there is nothing to
        ## evaluate: every task row and the joint-centering cost see the variable directly,
        ## and the network appears only in `CreateFlowConsistencyConstraint`. This makes the
        ## forward kinematics cheaper too (Drake trig on a plain variable rather than on a
        ## network output), and it is why lifting costs no extra flow evaluation.
        ## An AutoDiffXd iterate is the expensive one -- it is the `jacrev` through the
        ## network -- so the two are counted apart rather than lumped together.
        bucket = ("map_jacobian" if isinstance(vars[0], AutoDiffXd)
                  else "map_forward")
        if self._LiftingQ():
            counts[bucket] += 1
            q = self.LiftedQ(vars)
            self._last_config = (None, q)
            return q, self.fk(q)
        if not getattr(self.options, "share_flow_evaluations", False):
            counts[bucket] += 1
            cfg, q = self.VarsToConfigAndQ(vars)
            self._last_config = (self._FlowCacheKey(vars), cfg)
            return q, self.fk(q)
        cache = getattr(self, "_flow_cache", None)
        if cache is None:
            cache = self._flow_cache = {}
        key = self._FlowCacheKey(vars)
        hit = cache.get(key)
        if hit is None:
            counts[bucket] += 1
            cfg, q = self.VarsToConfigAndQ(vars)
            hit = (q, self.fk(q), cfg)
            cache[key] = hit
            while len(cache) > 4:
                cache.pop(next(iter(cache)))
        else:
            counts["memo_hits"] += 1
        return hit[0], hit[1]

    def VarsToConfigAndQ(self, vars):
        '''`(configuration, plant positions)` for an iterate.

        The two coincide on a robot whose configuration IS its plant position vector,
        which is both rigid arms -- `num_pos == num_arm_dof == 7` in every live scene --
        and the default returns the SAME OBJECT twice, so nothing downstream can tell the
        difference. They part company on a robot whose configuration parameterises
        something else: the soft PCS arm decides over 12 strains and drives 231 floating-body
        positions, where the plant vector carries quaternions and metres and its "limits"
        are +-inf.

        Subclasses that separate the two override THIS, and let `VarsToQ` delegate to it,
        so the expensive map runs once per iterate rather than once per consumer.
        '''
        q = self.VarsToQ(vars)
        return q, q

    def Config(self, vars):
        '''The configuration at an iterate, from the same memo `QAndPose` fills.

        A constraint row may call this freely: on the shared path it is a pure dictionary
        hit, so the joint-limit row costs no second network pass. That matters -- the rule
        this project learned the hard way is never to add a constraint that recomputes
        `VarsToQ` itself, which is how half the network work was once redundant.
        '''
        key = self._FlowCacheKey(vars)
        cache = getattr(self, "_flow_cache", None)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None and len(hit) > 2:
                return hit[2]
        last = getattr(self, "_last_config", None)
        if last is not None and last[0] is not None and last[0] == key:
            return last[1]
        return self.VarsToConfigAndQ(vars)[0]

    ## ------------------------- shared starting point ----------------------- ##

    def CalibrateFlowFrame(self, samples=4, tol=None):
        '''Measure the offset between the scene's end-effector frame and the frame the
        flow was actually trained on, and cache it.

        This is not a nicety. The flow is conditioned on the pose of a specific frame --
        jrl's `panda_hand`, at the standard Franka offset from `panda_link7`. The mug
        scene welds a finray gripper to `panda_link7` and that model *also* contains a
        body called `panda_hand`, but at translation [0, 0, 0.134] and rpy [90, 0, 45]
        rather than the Franka hand's [0, 0, 0.107] and [0, 0, -45]. Looking the frame up
        by name therefore returns a frame 27 mm and 120 degrees away from the one the
        network means, and every `c` handed to the flow in the grasp experiments was in
        that wrong frame. Measured symptom: inverting a random configuration at the
        scene's frame returns |z| = 67.6, against 2.23 -- essentially sqrt(7), the typical
        norm under the latent prior -- at the correct frame. The network was being asked
        about configurations it considers astronomically unlikely on every iterate.

        Both frames are rigidly welded to the same link, so the offset is a constant; it
        is measured at several configurations and checked rather than assumed.
        '''
        if not self.options.calibrate_flow_frame:
            self.X_ee_flow = RigidTransform()
            return self.X_ee_flow
        lower, upper = (np.asarray(bound)[:self.num_arm_dof]
                        for bound in self.ConfigLimits())
        # A local, fixed-seed generator, for two reasons. The offset is constant by
        # construction, but it is *measured*, so it carries ~1e-8 of numerical noise that
        # differs with the configurations it was measured at; drawing them from the global
        # stream gave every program a slightly different X_ee_flow, and the flow amplifies
        # 1e-8 in the conditioning pose to 1e-6 in q -- enough that two arms of the same
        # cell were not solving quite the same problem. It also stops this call from
        # consuming global draws, which used to shift the benchmark's target grid depending
        # on whether calibrate_flow_frame was on.
        rng = np.random.default_rng(0)
        offsets = []
        for _ in range(samples):
            q_arm = rng.uniform(lower, upper)
            self.plant.SetPositions(self.plant_context, self.ConfigToPlantQ(q_arm))
            X_scene = self.frame_for_flow.CalcPoseInWorld(self.plant_context)
            pose = self.ik_solver.robot.forward_kinematics(
                torch.tensor(q_arm[None, :], dtype=torch.float64, device=DEVICE))
            pose = pose.detach().cpu().numpy()[0]
            wxyz = pose[3:] / np.linalg.norm(pose[3:])
            X_flow = RigidTransform(Quaternion(wxyz), pose[:3])
            offsets.append(X_scene.inverse() @ X_flow)
        spread = max(np.linalg.norm(offsets[0].translation() - o.translation())
                     + abs((offsets[0].inverse() @ o).rotation().ToAngleAxis().angle())
                     for o in offsets[1:])
        ## `tol` was a dead parameter for the life of this method -- the bound below was
        ## hardcoded at 1e-6 and the signature's default was 1e-9. It is live now because a
        ## robot whose forward model is a LEARNED surrogate cannot pass a 1e-6 constancy
        ## check: the offset it measures is then the surrogate's own error, and the right
        ## response is to state the tolerance and RECORD the spread, not to switch the
        ## check off. Default unchanged at 1e-6, and no caller passes it.
        if tol is None:
            tol = getattr(self, "flow_frame_tol", 1e-6)
        self.flow_frame_spread = spread
        if spread > tol:
            raise RuntimeError(
                f"the flow frame offset is not constant across configurations "
                f"(spread {spread:.3e} > {tol:.3e}); the scene's joint convention does not "
                f"match the one the network was trained with")
        self.X_ee_flow = offsets[0]
        return self.X_ee_flow

    def FlowPoseInWorld(self, context=None):
        '''The pose the flow should be conditioned on, in the world frame.'''
        context = self.plant_context if context is None else context
        X = self.frame_for_flow.CalcPoseInWorld(context)
        offset = getattr(self, "X_ee_flow", None)
        return X if offset is None else X @ offset

    @property
    def frame_for_flow(self):
        return self.FlowFrame()

    def FlowFrame(self):
        '''The frame the flow was conditioned on during training.

        The mug programs move `self.frame` to `between_fingers` because that is where the
        grasp constraint acts, but the network still speaks in terms of the end-effector
        frame it was trained against, so `c` must always be expressed there.
        '''
        return getattr(self, "ee_frame", self.frame)

    def InvertFlow(self, q_arm, c):
        '''The latent that reproduces `q_arm` under conditioning pose `c`.

        IKFlow is a normalizing flow, so this is the network run forwards (`rev=False`)
        and is exact -- not an optimisation. It is what lets the learned formulation start
        from the *same* configuration as the joint-space one, the way ../codebase seeds
        its analytic formulation by recovering `psi` and `GC` from `q_initial`.
        '''
        dtype = self.torch_dtype
        pose7 = self.CToPose7(np.asarray(c, dtype=float))
        c_t = torch.tensor(np.concatenate([pose7, [0.0]])[None, :], dtype=dtype, device=DEVICE)
        x = np.zeros((1, self.ik_solver.network_width))
        x[0, :self.num_arm_dof] = np.asarray(q_arm, dtype=float)[:self.num_arm_dof]
        x_t = torch.tensor(x, dtype=dtype, device=DEVICE)
        with torch.no_grad():
            z, _ = self.ik_solver.nn_model(x_t, c=c_t, rev=False)
        return z.squeeze(0).detach().cpu().numpy().astype(float)

    def SetStartFromQ(self, q_arm):
        '''Start this program at the configuration `q_arm`, in its own variables.

        Returns how far the start had to be clipped to sit inside the variable bounds; a
        start outside the box is not the same start, and the amount matters when reading
        a paired comparison.
        '''
        q_arm = np.asarray(q_arm, dtype=float)[:self.num_arm_dof]
        self.plant.SetPositions(self.plant_context, self.ConfigToPlantQ(q_arm))
        pose = self.FlowPoseInWorld()
        c = np.concatenate([pose.translation(), pose.rotation().ToRollPitchYaw().vector()])
        # Invert at the *unclipped* conditioning pose, then clip. The temptation is to do
        # it the other way round so that q(start) is exactly q_arm, and that is wrong: a
        # random collision-free configuration is not a grasp of this mug, so its
        # conditioning pose sits outside the program's box, and asking the flow for the
        # latent that produces q_arm under a *projected* pose returns |z| ~ 1e7 -- the
        # flow correctly reporting that this configuration is astronomically unlikely
        # there. Measured: |z| goes from 1.6 to 6.2e7 between the two orders. Inverting
        # first keeps the latent inside the typical set and lets the box move only the
        # conditioning pose, which is the quantity the box is actually about.
        z = self.InvertFlow(q_arm, c)
        # The c region is a general linear constraint now, not a variable bound, so
        # _SetClipped/_BoxDistance no longer see it; measure and (for legacy) apply the
        # clip against the stored region instead.
        c_lo, c_hi = self.c_box
        z_lo, z_hi = self.z_box
        c_clip_distance = float(np.linalg.norm(np.clip(c, c_lo, c_hi) - c))
        z_clip_distance = float(np.linalg.norm(np.clip(z, z_lo, z_hi) - z))
        if self.options.legacy_paired_start:
            self.prog.SetInitialGuess(self.c, np.clip(c, c_lo, c_hi))
            self.prog.SetInitialGuess(self.z, np.clip(z, z_lo, z_hi))
            clipped = c_clip_distance + z_clip_distance
            self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
            return clipped
        # The exact paired start. Both the conditioning pose and the latent are set
        # *unclipped* -- a Drake initial guess need not satisfy a constraint -- so
        # flow(c, z) reproduces q_arm to the network's noise floor (measured ~1e-6 in
        # float32, tighter in float64), and the correction closes that residual. q(start)
        # is then q_arm to float precision, which is what "paired" claims; the pre-clipped
        # version started 1.2-3.3 rad away.
        #
        # What is returned is the distance from BOTH guesses to their own regions, summed.
        # It is a pure DIAGNOSTIC and nothing projects it: `CBoxConstraint` and
        # `LatentBoxConstraint` are general linear constraints, so IPOPT's `bound_push`
        # -- which acts only on variable bounds -- never sees them and the first evaluated
        # iterate is the guess as written. That is the whole point of those two regions not
        # being bounding boxes. So read a nonzero value here as "the start began outside the
        # region and the solver walked it in", the behaviour the latent region exists to
        # allow (measured: |z| ~ 7.9 at the start against a radius-4.96 region, |z| ~ 2.9 at
        # the solution), NOT as a displacement already applied to iterate 0. The correction's
        # +-0.1 box is still a variable bound, but a paired start always respects it, being
        # set to zero here.
        self.prog.SetInitialGuess(self.c, c)
        self.prog.SetInitialGuess(self.z, z)
        clipped = c_clip_distance + z_clip_distance
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        ## Against the CONFIGURATION, not the first `num_arm_dof` entries of a plant
        ## vector. The two are the same thing on both rigid arms, where the configuration
        ## IS the plant vector; on a robot whose plant carries floating-body quaternions
        ## the slice would be comparing strains against quaternion components.
        residual = q_arm - np.asarray(
            self.Config(self.prog.GetInitialGuess(self.lumped_vars)),
            dtype=float)[:self.num_arm_dof]
        bound = self.options.correction_bound
        residual = np.nan_to_num(residual, nan=0.0, posinf=bound, neginf=-bound)
        self.prog.SetInitialGuess(self.correction, np.clip(residual, -bound, bound))
        ## Under lifting the paired start is exact by construction: `q_lift` IS the
        ## configuration, so it is set to `q_arm` itself and `start_q_error` must read 0.
        ## `q_init` is drawn inside the joint limits, so this guess is inside the box too
        ## and IPOPT has nothing to project.
        if self._LiftingQ():
            self.prog.SetInitialGuess(self.q_lift, q_arm)
        return clipped

    def SetNativeStart(self, q_init, rng):
        """This formulation's *own* initialisation, as it would be run outside a comparison.

        The learned formulation's natural procedure is the flow's inference procedure:
        condition on the pose the task hands you and draw the latent from the prior the
        network was trained against, with no correction. `create_prog` has already set the
        conditioning pose from the target, so only the latent is drawn here.

        This is a *sample*, not a search. Nothing in it looks at the problem's constraints
        or its objective, and no candidate is scored or selected -- which is the line that
        separates a formulation's natural initialisation from solving part of the problem
        outside the solver.

        `q_init` is accepted so that formulations whose natural start *is* a configuration
        can use it; this one ignores it. Returns a clip distance, for symmetry with
        `SetStartFromQ`.
        """
        self.prog.SetInitialGuess(self.z, rng.standard_normal(self.ik_solver.network_width))
        self.prog.SetInitialGuess(self.correction, np.zeros(self.num_arm_dof))
        ## Under lifting, start the lifted variable *consistent with the draw* -- i.e. at
        ## the flow's own output -- so the equality begins satisfied and the native start
        ## is still one sample with nothing scored. Set unclipped, on the same reasoning as
        ## the conditioning pose: a Drake guess need not satisfy its bounds and IPOPT's own
        ## projection is the solver's first move. A draw that lands in a runaway region
        ## therefore shows up as a large projection rather than being hidden.
        if self._LiftingQ():
            q_flow = np.asarray(self.VarsToQ(
                self.prog.GetInitialGuess(self.lumped_vars)), dtype=float)
            q_flow = np.nan_to_num(q_flow[:self.num_arm_dof], nan=0.0)
            self.prog.SetInitialGuess(self.q_lift, q_flow)
            return self._BoxDistance(self.q_lift, q_flow)
        return 0.0

    def _VariableBounds(self, variables):
        '''The tightest bounding box the program imposes on `variables`.'''
        lower = np.full(len(variables), -np.inf)
        upper = np.full(len(variables), np.inf)
        names = {v.get_id(): i for i, v in enumerate(variables)}
        for binding in self.prog.bounding_box_constraints():
            evaluator = binding.evaluator()
            for row, var in enumerate(binding.variables()):
                index = names.get(var.get_id())
                if index is not None:
                    lower[index] = max(lower[index], evaluator.lower_bound()[row])
                    upper[index] = min(upper[index], evaluator.upper_bound()[row])
        return lower, upper

    def _SetClipped(self, variables, values):
        '''Set an initial guess, clipped into the program's bounding box on it.'''
        values = np.asarray(values, dtype=float)
        lower, upper = self._VariableBounds(variables)
        clipped = np.clip(values, lower, upper)
        self.prog.SetInitialGuess(variables, clipped)
        return float(np.linalg.norm(clipped - values))

    def _BoxDistance(self, variables, values):
        '''How far `values` sits outside the program's bounding box on `variables` --
        the projection distance IPOPT will apply at its first iterate when the guess is
        set unclipped.'''
        values = np.asarray(values, dtype=float)
        lower, upper = self._VariableBounds(variables)
        return float(np.linalg.norm(np.clip(values, lower, upper) - values))

    ## ------------------------- multi-start seeding ------------------------- ##

    @property
    def torch_dtype(self):
        return torch.float64 if self.options.use_float64 else torch.float32

    def ConfigureNetworkDtype(self):
        '''Cast the flow to the working dtype. Idempotent, so it is safe to call on a
        solver instance shared between programs.'''
        self.ik_solver.nn_model.to(self.torch_dtype)
        self.ik_solver.nn_model.eval()

    def ConfigLimits(self):
        '''`(lower, upper)` for the CONFIGURATION, which is what the joint-limit row bounds.

        The default is the plant's own position limits, which is exactly right for a robot
        whose configuration is its plant vector. It is exactly wrong for one whose plant
        vector is floating-body quaternions and metres -- there the limits are +-inf, and
        handing IPOPT 231 vacuous rows is not a constraint, it is noise.
        '''
        return (self.plant.GetPositionLowerLimits(), self.plant.GetPositionUpperLimits())

    def ConfigToPlantQ(self, cfg):
        '''Configuration -> a full plant position vector. Inverse of nothing; a map.'''
        return self.PadQ(cfg)

    def SampleConfiguration(self, rng):
        '''One uniform draw from the configuration box.

        Exists so a benchmark script can sample the ROBOT rather than the plant. Drawing
        uniformly over plant position limits is what the rigid arms' scripts do, and it is
        the same thing there; on a floating-body model those limits are +-inf and the draw
        is nan. Consumes exactly as many doubles as the configuration is wide.
        '''
        lower, upper = self.ConfigLimits()
        return rng.uniform(lower, upper)

    def PadQ(self, q_arm):
        '''Arm joint angles -> a full plant position vector.'''
        q = np.zeros(self.num_pos)
        q[:self.num_arm_dof] = q_arm
        q[self.num_arm_dof:] = 0.04  # fixed gripper joints
        return q

    @staticmethod
    def CToPose7(c):
        '''Conditioning variable (xyz + rpy) -> the xyz + wxyz the flow is conditioned on.'''
        return np.concatenate([c[:3], RotationMatrix(RollPitchYaw(c[3:6])).ToQuaternion().wxyz()])

    ## These are Robot Specific need to be implemented in each file ##
    def ik_inference(self, vars):
        pass
    def VarsToQ(self, vars):
        pass

    def SetPositions(self, q):
        if isinstance(q[0], AutoDiffXd):
            self.autodiff_plant.SetPositions(self.autodiff_context, q)
            return self.autodiff_frame, self.autodiff_context
        else:
            self.plant.SetPositions(self.plant_context, q)
            return self.frame, self.plant_context


    def EvalAllConstraints(self, vars):
        '''Parallelize as much of the VarsToQ as possible to shorten computation time'''
        q, pose = self.QAndPose(vars)  ## one flow evaluation for every constraint row
        total_length = sum(len(constraint) for constraint in self.constraints)
        result = np.full(total_length, q[0]) ## q datatype
        idx = 0
        for constraint in self.constraints:
            l = len(constraint)
            result[idx:idx + l] = constraint.eval_func(vars = vars, q = q, pose = pose)
            idx += l
        return result

    
    def ApplyConstraints(self):
        total_lb = np.hstack([constraint.lb for constraint in self.constraints])
        total_ub = np.hstack([constraint.ub for constraint in self.constraints])
        self.all_constraints = self.prog.AddConstraint(
            func=self.EvalAllConstraints,
            lb=total_lb,
            ub=total_ub,
            vars=self.lumped_vars
        )
        self.all_constraints.evaluator().set_description("AllIKFlowConstraints")


    def CreateIKConstraint(self):
        '''Six rows: the per-axis position error, then the roll-pitch-yaw residual.

        All six are an EQUALITY, lb == ub == 0: the end-effector is at the target or it
        is not, and that is what the task says. This matches ../codebase's
        EEPoseConstraint, which passes `lb=extract_xyzrpy(target), ub=extract_xyzrpy(target)`,
        and eaik-experiment's reachability row, whose bounds are `lb=[0], ub=[0]` under a
        comment reading "(no slack)".

        The position rows used to be a `+-ik_constraint_tol[0]` box, which is not a
        slightly looser equality but a different kind of constraint: an interior-point
        method parks ON the face of an inequality instead of driving the residual to
        zero. Measured over 480 Stage D cells, 67-97% of every arm's successful pose
        solves sat at exactly 1e-4, while the orientation rows of this same constraint --
        already an equality -- converged to ~1e-9. Same program, same solver, five orders
        of magnitude apart, purely from how the bound was written. The mug task's axis
        rows have always been `0 == 0` and the learned arm satisfies them to a median
        9.5e-09, so this is well within what the flow can do; the old note about not
        asking for a tolerance below the network's noise floor was refuted by its own
        grasp numbers.

        Numerical slack belongs in the solver (IPOPT's constr_viol_tol and the
        acceptable_* family, SNOPT's Major feasibility tolerance) and in the benchmark's
        post-hoc gate -- never here.

        `orientation_error_form="rpy_boxed"` remains as an explicit, opt-in ablation and
        is now the only way to obtain a boxed row; it has to be named to be had.
        '''
        _, ori_tol = self.options.ik_constraint_tol
        form = self.options.orientation_error_form
        if form not in ORIENTATION_ERROR_FORMS:
            raise ValueError(f"Unknown orientation_error_form {form!r}; expected one of "
                             f"{sorted(ORIENTATION_ERROR_FORMS)}")
        rpy_tol = 0.0 if form == "rpy" else ori_tol

        # The target's rpy is fixed for the life of the program, so compute it once
        # rather than per constraint evaluation.
        target_rpy = RollPitchYaw(RotationMatrix(Quaternion(self.target_pose[3:]))).vector()
        lb = np.array([0.0] * 3 + [-rpy_tol] * 3)
        ub = np.array([0.0] * 3 + [rpy_tol] * 3)

        def eval_func(vars, q, pose):
            position, orientation = pose
            pos_error = position - self.target_pose[:3]
            return np.concatenate([pos_error, orientation_error_rpy(orientation, target_rpy)])

        self.ik_constraint = IKFlowConstraints(lb, ub, eval_func, description="IKConstraint")
        self.constraints.append(self.ik_constraint)
        return self.ik_constraint

    def CreateCollisionFreeConstraint(self):
        scale = self.options.collision_row_scale
        self.collision_free_constraint_eval = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=self.options.collision_bound,
            influence_distance_offset=self.options.collision_influence_offset,
            plant_context=self.plant_context
        )
        def eval_func(vars = None, q = np.zeros(7), pose = None):
            return scale * self.collision_free_constraint_eval.Eval(q)
        lb = np.array([-np.inf])
        # The binding's raw value is "in collision" above 1, so the scaled bound is the
        # scale itself; keeping the two coupled means changing the scale reshapes the
        # gradient without moving the feasible set.
        ub = np.array([scale])
        self.collision_free_constraint = IKFlowConstraints(lb, ub, eval_func, description="CollisionFreeConstraint")
        self.constraints.append(self.collision_free_constraint)
        return self.collision_free_constraint
    
    def CreateJointLimitsConstraint(self):
        ## Bounds the CONFIGURATION, not the plant vector. On both rigid arms those are the
        ## same seven numbers against the same seven rows -- `ConfigLimits` defaults to the
        ## plant's limits and `Config` returns the very object `VarsToQ` produced -- so this
        ## is bit-identical there by construction rather than by measurement.
        lower_limits, upper_limits = self.ConfigLimits()
        def eval_func(vars = None, q = None, pose = None):
            return self.Config(vars)
        self.joint_limit_constraint = IKFlowConstraints(lower_limits, upper_limits, eval_func, description="JointLimitsConstraint")
        self.constraints.append(self.joint_limit_constraint)
        return self.joint_limit_constraint


    ## ------------------------- Stage F: acting on q ------------------------ ##
    ##
    ## Both of the following are STATED DEVIATIONS from the draft's eq. (6), run as
    ## diagnostics of the flow's runaway regions. See ProgramOptions.lift_q.

    def _LiftingQ(self):
        """True when this program actually carries a lifted configuration variable.

        Tests the attribute as well as the option because the baselines share one
        `ProgramOptions`, and a `--set lift_q=True` run must leave them untouched rather
        than raise inside their construction -- the failure mode that scored two whole
        columns zero before `_abort_on_dead_arm` existed.
        """
        return bool(getattr(self.options, "lift_q", False)) and hasattr(self, "q_lift")

    def LiftedQ(self, vars):
        """The lifted configuration, padded to the plant's position vector.

        `q_lift` is appended last to `lumped_vars`, so it is the final `num_arm_dof`
        entries of whatever Drake hands the callback.
        """
        n = self.num_arm_dof
        ad = isinstance(vars[0], AutoDiffXd)
        q = np.zeros(self.num_pos, dtype=AutoDiffXd if ad else float)
        q[n:] = [0.04] * (self.num_pos - n)      # fixed gripper joints, as in VarsToQ
        q[:n] = vars[-n:]
        return q

    def LiftedQBoxConstraint(self):
        """The joint limits, as the lifted variable's own bounding box.

        A bounding box is the point of the exercise, not an oversight of the rule that a
        region an initial guess may violate must be a general constraint: the paired start
        sets `q_lift = q_init`, which is drawn inside the limits, so this box is never
        violated at the guess. What it buys is that IPOPT keeps the iterate inside it, so
        a solve cannot return a configuration of 1e11 rad.
        """
        self.lifted_q_box = self.prog.AddBoundingBoxConstraint(
            self.plant.GetPositionLowerLimits()[:self.num_arm_dof],
            self.plant.GetPositionUpperLimits()[:self.num_arm_dof],
            self.q_lift)
        self.lifted_q_box.evaluator().set_description("LiftedQBox")

    def CreateFlowConsistencyConstraint(self):
        """`flow(c, z) + q_c - q_lift = 0`: the chart, as an explicit equality.

        This is the whole content of the lifting. The feasible set is unchanged -- it is
        the same problem written with the network's output named -- but the iterate path is
        not: the joint limits are now `q_lift`'s own bounding box, which IPOPT satisfies at
        every iterate, so a solve can no longer *return* a configuration of 1e11 rad.

        Thomas's objection, which this is run to test rather than to dismiss: the badly
        scaled Jacobian does not disappear, it moves out of an inequality row and into an
        equality row, and an interior-point method need not be any happier with it there.
        """
        n = self.num_arm_dof
        zeros = np.zeros(n)

        def eval_func(vars=None, q=None, pose=None):
            ## `q` is the LIFTED configuration here (QAndPose returns it under lifting), so
            ## the flow must be evaluated separately -- this is the one place it enters.
            return self.VarsToQ(vars)[:n] - vars[-n:]

        self.flow_consistency_constraint = IKFlowConstraints(
            zeros, zeros, eval_func, description="FlowConsistencyConstraint")
        self.constraints.append(self.flow_consistency_constraint)
        return self.flow_consistency_constraint

    def JointLimitPenaltyCost(self):
        self.joint_limit_penalty_cost = self.prog.AddCost(
            func=self.EvalJointLimitPenaltyCost, vars=self.lumped_vars)
        self.joint_limit_penalty_cost.evaluator().set_description("JointLimitPenaltyCost")

    def EvalJointLimitPenaltyCost(self, vars):
        """`w * sum(max(0, q - ub)^2 + max(0, lb - q)^2)`.

        A quadratic hinge, so it is C^1 and -- unlike the correction penalty, which is
        active everywhere -- **identically zero with zero gradient inside the limits**. A
        cell that already satisfies them must therefore return a bit-identical solution at
        every weight, which is the A/B test for this implementation.

        Goes through `QAndPose`, never a second `VarsToQ`: the flow's forward pass and
        `jacrev` are the dominant cost of the whole formulation, and a cost binding that
        recomputed them would roughly double the solve.
        """
        q, _ = self.QAndPose(vars)
        lower = self.plant.GetPositionLowerLimits()
        upper = self.plant.GetPositionUpperLimits()
        total = 0.0 * q[0]                       # keeps the AutoDiffXd type and its seeds
        for i in range(len(q)):
            over = q[i] - upper[i]
            if over > 0.0:
                total = total + over * over
            under = lower[i] - q[i]
            if under > 0.0:
                total = total + under * under
        return self.options.joint_limit_penalty_weight * total

    def LatentBoxConstraint(self):
        '''`-5 <= z <= 5`, as a general linear constraint rather than a variable bound.

        Defined once here and called by every program that owns a latent -- the pose
        programs through `BoundingBoxConstraint` below and the mug programs through their
        overrides -- because the three copies this replaces are how the bug below survived
        the first repair: the pose arms were fixed and the mug arms silently were not.

        The reason it is not a bounding box is the same one that applies to the
        conditioning pose, but the failure was worse, because we applied the projection
        ourselves rather than leaving it to IPOPT: `SetStartFromQ` clipped the inverted
        latent into this box before the solver ever ran. The flow is a bijection, so
        `flow(c, InvertFlow(q, c))` reproduces `q` exactly -- but only at the *unclipped*
        latent. On the iiwa pose task the inversion routinely returns components past +-5
        (measured |z| ~ 9.1), so the clip moved the start several radians, the +-0.1
        correction could not close the residual, and 49 of 60 paired cells were recorded
        as `unrepresentable_start`: an arm scored as unable to represent a configuration
        it represents exactly. The feasible set is unchanged; what changes is only the
        guess's freedom to start outside it.
        '''
        width = self.ik_solver.network_width
        self.z_box = (-5. * np.ones(width), 5. * np.ones(width))
        self.bounding_box_constraint = self.prog.AddLinearConstraint(
            np.eye(width), self.z_box[0], self.z_box[1], self.z)
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        return self.bounding_box_constraint

    def BoundingBoxConstraint(self):
        '''The learned arm's three regions: the latent box, the conditioning-pose box and
        the correction's bound.

        ONE method for every learned program, pose and grasp alike, dispatching on whether
        the program carries a target mug. The five mug programs used to override this with
        five copies of the grasp box, which is how the latent-box repair once fixed the
        pose arms and silently missed the grasp arms. The joint-space arms override it
        with `QBoundingBoxConstraint`; the analytic arms with their own regions.'''
        self.LatentBoxConstraint()
        if getattr(self, "target_mug", None) is not None:
            self.GraspCBoxConstraint()
        else:
            self.PoseCBoxConstraint()
        self.CorrectionBoxConstraint()
    

    
    def JointCenteringCost(self):
        self.joint_centering_cost = self.prog.AddCost(
            func = self.EvalJointCenteringCost,
            vars = self.lumped_vars
        )
        self.joint_centering_cost.evaluator().set_description("JointCenteringCost")
    
    def EvalJointCenteringCost(self, vars):
        # Shares the constraint binding's flow evaluation when share_flow_evaluations is
        # on; otherwise this is a second full forward pass / jacrev at the same point.
        ## Acts on the CONFIGURATION. Identical on the rigid arms, where the configuration
        ## is the plant vector; on the soft PCS arm this is a quadratic in strains, whose zero
        ## is the straight unstretched rod -- the elastic-energy analogue of joint
        ## centering, and the same shared objective either way.
        self.QAndPose(vars)
        n = self.num_arm_dof
        diff = self.Config(vars)[:n] - self.q_nominal[:n]
        return 0.5 * diff @ (self.options.joint_centering_cost * np.eye(n)) @ diff
    
    def CorrectionCost(self):
        self.correction_cost = self.prog.AddQuadraticCost(
            Q=self.options.correction_cost_weight * np.eye(self.num_arm_dof),
            b=np.zeros(self.num_arm_dof),
            vars=self.correction
        )
        self.correction_cost.evaluator().set_description("CorrectionCost")
    

    ## The solver axis. These are METHOD CLASSES, not vendors: IPOPT is an interior-point
    ## method, SNOPT is SQP, and NLopt is here as an augmented Lagrangian. A solver added
    ## here should be justified by the class it contributes. The fourth class: `svgd`,
    ## batch-parallel SVGD-style particle methods in torch (reopened by Thomas, 2026-10-08).
    ## It is not a Drake solver -- `Solve()` dispatches it to `_SolveSvgd` before the Drake
    ## option builders, and it returns a duck-typed `SvgdResult` (`src/svgd/result.py`).
    SOLVERS = ("ipopt", "snopt", "nlopt", "svgd")

    def _IpoptOptions(self):
        solver = IpoptSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_tol", self.options.acceptable_tol)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_constr_viol_tol", self.options.acceptable_constr_viol_tol)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_dual_inf_tol", self.options.acceptable_dual_inf_tol)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_compl_inf_tol", self.options.acceptable_compl_inf_tol)
        solver_options.SetOption(IpoptSolver().solver_id(), "file_print_level", self.options.file_print_level)
        solver_options.SetOption(IpoptSolver().solver_id(), "print_user_options", "yes")
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_iter", self.options.acceptable_iter)
        solver_options.SetOption(IpoptSolver().solver_id(), "max_wall_time", self.options.max_wall_time)
        if self.options.ipopt_mu_strategy is not None:
            solver_options.SetOption(IpoptSolver().solver_id(), "mu_strategy", self.options.ipopt_mu_strategy)
        if self.options.ipopt_nlp_scaling_method is not None:
            solver_options.SetOption(IpoptSolver().solver_id(), "nlp_scaling_method",
                                     self.options.ipopt_nlp_scaling_method)
        if self.options.ipopt_nlp_scaling_max_gradient is not None:
            solver_options.SetOption(IpoptSolver().solver_id(), "nlp_scaling_max_gradient",
                                     float(self.options.ipopt_nlp_scaling_max_gradient))
        if self.options.ipopt_theta_max_fact is not None:
            solver_options.SetOption(IpoptSolver().solver_id(), "theta_max_fact",
                                     float(self.options.ipopt_theta_max_fact))
        if self.options.ipopt_watchdog_trigger is not None:
            solver_options.SetOption(IpoptSolver().solver_id(), "watchdog_shortened_iter_trigger",
                                     int(self.options.ipopt_watchdog_trigger))
        if self.options.ipopt_max_soc is not None:
            solver_options.SetOption(IpoptSolver().solver_id(), "max_soc", int(self.options.ipopt_max_soc))
        ## The quasi-Newton and barrier knobs, added for the solver-settings sweep. Every
        ## one of these was confirmed to reach IPOPT with `used = yes` in the user-options
        ## echo before being exposed -- `print_user_options` above is what makes that
        ## checkable, and it is why `linear_solver` is absent (Drake's IPOPT rejects
        ## anything but `spral`/`custom`) and why `mu_init` carries the warning it does.
        for name, value, cast in (
                ("tol", self.options.ipopt_tol, float),
                ("constr_viol_tol", self.options.ipopt_constr_viol_tol, float),
                ("dual_inf_tol", self.options.ipopt_dual_inf_tol, float),
                ("compl_inf_tol", self.options.ipopt_compl_inf_tol, float),
                ("limited_memory_max_history", self.options.ipopt_limited_memory_max_history, int),
                ("limited_memory_update_type", self.options.ipopt_limited_memory_update_type, str),
                ("mu_init", self.options.ipopt_mu_init, float),
                ("alpha_for_y", self.options.ipopt_alpha_for_y, str),
                ("recalc_y", self.options.ipopt_recalc_y, str),
                ("bound_relax_factor", self.options.ipopt_bound_relax_factor, float)):
            if value is not None:
                solver_options.SetOption(IpoptSolver().solver_id(), name, cast(value))
        if self.options.max_iter is not None:
            solver_options.SetOption(IpoptSolver().solver_id(), "max_iter", int(self.options.max_iter))
        return solver, solver_options

    def _SnoptOptions(self):
        '''SNOPT, the SQP arm.

        Note what is NOT here: `acceptable_tol` and `acceptable_constr_viol_tol`. An earlier
        revision fed those into SNOPT's Major optimality and feasibility tolerances, but
        IPOPT's `acceptable_*` family is its RELAXED EARLY-STOP criterion, not what it
        converges to -- so SNOPT was being handed 1e-3 where IPOPT drives to 1e-8. Unset
        `snopt_*` fields simply are not passed, leaving SNOPT at its own 1e-6 defaults.
        '''
        solver = SnoptSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(SnoptSolver.id(), "Major print level",
                                 int(self.options.snopt_major_print_level))
        solver_options.SetOption(SnoptSolver.id(), "Minor print level",
                                 int(self.options.snopt_minor_print_level))
        ## "Timing level", NOT "Timing Level". Drake accepts either -- it only raises on a
        ## keyword SNOPT's table does not know at all -- but SNOPT's parser is case
        ## sensitive on the second word, so the capitalised form is silently INERT and
        ## writes no timing block. Measured both ways. A SNOPT option can be accepted and
        ## do nothing, so anything set here has to be confirmed in the print file.
        solver_options.SetOption(SnoptSolver.id(), "Timing level", 3)
        ## The solution dump (Sections 1 and 2, every row and column) is a fifth of the
        ## print file and nothing parses it. One log per cell over a 480-cell grid is
        ## exactly the many-small-files pattern this project already had to fix once.
        solver_options.SetOption(SnoptSolver.id(), "Solution",
                                 "Yes" if self.options.snopt_solution_print else "No")
        ## SNOPT types its options: "Time limit" is a double and the iteration limits are
        ## ints. The wrong Python type reaches the wrong snSet* and raises.
        solver_options.SetOption(SnoptSolver.id(), "Time Limit", float(self.options.max_wall_time))
        for name, value, cast in (
                ("Major feasibility tolerance", self.options.snopt_major_feasibility_tol, float),
                ("Major optimality tolerance", self.options.snopt_major_optimality_tol, float),
                ("Minor feasibility tolerance", self.options.snopt_minor_feasibility_tol, float),
                ("Minor iterations limit", self.options.snopt_minor_iterations_limit, int),
                ("Iterations limit", self.options.snopt_iterations_limit, int),
                ("Scale option", self.options.snopt_scale_option, int),
                ("Verify level", self.options.snopt_verify_level, int),
                ("Linesearch tolerance", self.options.snopt_linesearch_tolerance, float),
                ("Superbasics limit", self.options.snopt_superbasics_limit, int),
                ("Major step limit", self.options.snopt_major_step_limit, float),
                ("Violation limit", self.options.snopt_violation_limit, float),
                ("Function precision", self.options.snopt_function_precision, float),
                ("Hessian frequency", self.options.snopt_hessian_frequency, int),
                ("Elastic weight", self.options.snopt_elastic_weight, float),
                ("Crash option", self.options.snopt_crash_option, int),
                ("Proximal point method", self.options.snopt_proximal_point_method, int)):
            if value is not None:
                solver_options.SetOption(SnoptSolver.id(), name, cast(value))
        ## A VALUELESS keyword, so it cannot live in the table above: SNOPT switches to the
        ## gradient-free line search on the keyword being present at all, and `= 0` turns it
        ## ON exactly as `= 1` does. Emitting it only when True is the only way to express
        ## "off". Verified in the parameter echo, where it reads `Nonderiv.  linesearch`.
        if self.options.snopt_nonderivative_linesearch:
            solver_options.SetOption(SnoptSolver.id(), "Nonderivative linesearch", 1)
        ## The BUDGET, which is a different thing from the tolerances above and is NOT left
        ## at each solver's own default. The controlled variable of this comparison is the
        ## WALL CLOCK, so a solver quietly stopping at its own iteration default is being
        ## given a different budget rather than converging at its own tolerance -- the same
        ## class of unfairness as handing it someone else's tolerances.
        ##
        ## SNOPT's default Major iterations limit is 1000; IPOPT's max_iter default is 3000.
        ## Measured: an iiwa joint-space cell stopped at exactly 1000 majors inside a 20 s
        ## cap, so this binds in practice and is not hypothetical. Equalised at IPOPT's
        ## 3000, rather than raised out of the way entirely, so that IPOPT's own path stays
        ## byte-identical to every archived run and either solver capping is visible and
        ## equal (`hit_iteration_cap`). NLopt has no notion of an iteration, so its nearest
        ## analogue -- max_eval -- is disabled by default and the wall clock is all it has.
        solver_options.SetOption(SnoptSolver.id(), "Major iterations limit",
                                 int(self.options.max_iter) if self.options.max_iter is not None
                                 else int(self.options.snopt_major_iterations_default))
        return solver, solver_options

    def _NloptOptions(self):
        '''NLopt, the AUGMENTED LAGRANGIAN arm.

        Two things to know before editing this.

        **The six options emitted unconditionally are the cluster's whole NLopt surface.**
        Drake 1.56.0 declares exactly `algorithm`, `constraint_tol`, `xtol_rel`, `xtol_abs`,
        `max_eval` and `max_time`; this workstation's source build declares eleven and a
        nightly sixteen, and Drake RAISES on a name it does not know. So the six are emitted
        unconditionally and everything newer only when its `ProgramOptions` field is set,
        after `CheckNloptOptions` has confirmed this Drake carries it and that no
        inner-solver option was set without an inner algorithm to apply it to. A run setting
        none of the new fields emits exactly these keys with exactly these values, which is
        what keeps archived NLopt results comparable. Leaving the local optimizer unset
        remains a supported state -- Drake does not call `set_local_optimizer`, and NLopt
        supplies its own, which is the right shape because an augmented Lagrangian's inner
        problem is bound-constrained only. It is NOT LD_LBFGS: Drake's NLopt is built without
        the LGPL Luksan sources, so that family is compiled out (NLOPT_LUKSAN_DISABLED).

        **NLopt reports nothing.** No print file, no console output, and a details struct
        with a single `status` field -- no iteration count, no evaluation count, not even a
        solve time. `kPrintFileName` is accepted and silently ignored, which is why this is
        the one branch that does not set it. The evaluation counters on the program
        (`ResetEvalCounts`) exist because of this.
        '''
        solver = NloptSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(NloptSolver.id(), NloptSolver.AlgorithmName(),
                                 str(self.options.nlopt_algorithm))
        ## `max_time` is what makes the wall-clock cap bind. Without it a cell runs until
        ## the harness's own per-item timeout and takes the whole item with it.
        solver_options.SetOption(NloptSolver.id(), NloptSolver.MaxTimeName(),
                                 float(self.options.max_wall_time))
        ## Drake DEFAULTS max_eval to 1000. That is a cap, not "unset", and it binds here --
        ## left alone this column would measure an evaluation budget rather than the cap.
        max_eval = (int(self.options.max_iter) if self.options.max_iter is not None
                    else int(self.options.nlopt_max_eval))
        solver_options.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(), max_eval)
        for name, value in (
                (NloptSolver.ConstraintToleranceName(), self.options.nlopt_constraint_tol),
                (NloptSolver.XRelativeToleranceName(), self.options.nlopt_xtol_rel),
                (NloptSolver.XAbsoluteToleranceName(), self.options.nlopt_xtol_abs)):
            if value is not None:
                solver_options.SetOption(NloptSolver.id(), name, float(value))
        ## The post-1.56.0 surface. `CheckNloptOptions` returns only the fields actually set,
        ## so this loop is empty -- and this method's output byte-identical to what it produced
        ## before these fields existed -- unless a run asks for one. It is called again here
        ## rather than trusted from __post_init__ because options objects are mutated in place
        ## elsewhere (benchmark.py assigns `file_print_name` on a live one), which bypasses
        ## __post_init__ entirely; this is the last place a wrong name can be stopped before
        ## Drake sees it. The per-option cast is load-bearing: Drake routes NLopt options by
        ## Python type exactly as it does SNOPT's, so a mismatch reaches the wrong setter.
        surface = NloptOptionSurface()
        for name, value in CheckNloptOptions(self.options).items():
            spec = NLOPT_POST_1_56_OPTIONS[name]
            solver_options.SetOption(NloptSolver.id(), surface[spec.accessor],
                                     spec.cast(value))
        return solver, solver_options

    def RecordIterate(self, vars):
        '''Keep the newest iterate on the program, in memory, always.

        A solve that ends abnormally -- an exception, a harness kill, the (measured, once
        in 1740 cells) C++-level wedge inside a single IPOPT iteration -- can then still be
        verified from the point the solver actually had, instead of the point being
        discarded with the solve. The predecessor design raised SolveTimeout from the
        constraint callback, which both threw the iterate away and could not fire during
        the wedge (no Python ran for 102 minutes); when a wedge releases, IPOPT's own
        max_wall_time ends the solve at the next iteration boundary with the iterate
        intact, which needs no help from us.

        Called from `Solve()`'s visualization callback on every Drake solver, and by the
        svgd solver at every outer step. Also counts the callback and runs the Meshcat /
        `vars_file` visualization, so the two paths are the same code.
        '''
        self.last_iterate = np.array(vars, dtype=float)
        self.eval_counts["callback"] += 1
        visualization_callback(vars, diagram=self.diagram, diagram_context=self.diagram_context,
                               plant=self.plant, plant_context=self.plant_context,
                               vars_to_q=self.VarsToQ, vars_file=self.options.vars_file,
                               visualize=self.options.visualize)

    def _SolveSvgd(self):
        '''The fourth solver class. Not a Drake solver: nothing here touches SolverOptions.

        Records every `svgd_*` field as the emitted configuration -- the same role
        `emitted_solver_options` plays for the Drake solvers, so an adopted default is
        readable off the run -- resets the evaluation counters the batched solver bumps
        per pass, and hands the program to `SvgdSolver`. The import is lazy so the Drake
        columns never load torch-side solver code they do not use.
        '''
        self.ResetEvalCounts()
        self.emitted_solver_options = {"svgd": {
            f.name: getattr(self.options, f.name)
            for f in dataclass_fields(self.options) if f.name.startswith("svgd_")}}
        from src.svgd.solver import SvgdSolver
        return SvgdSolver(self).solve()

    def Solve(self):
        if os.path.exists(self.options.file_print_name):
            with open(self.options.file_print_name, "r+") as f:
                f.seek(0)
                f.truncate()

        which = self.options.which_solver
        ## An explicit check, because the predecessor was two bare `if`s: any value outside
        ## {ipopt, snopt} left `solver` unbound and died with UnboundLocalError several
        ## lines later. `--set which_solver=...` bypasses argparse's `choices`, so that was
        ## reachable from the command line.
        if which not in self.SOLVERS:
            raise ValueError(f"unknown which_solver {which!r}; expected one of {self.SOLVERS}")
        ## The svgd class is not a Drake solver: no SolverOptions, no print-file option, a
        ## duck-typed result. Everything from here down is the Drake path and stays as it was.
        if which == "svgd":
            return self._SolveSvgd()
        solver, solver_options = {"ipopt": self._IpoptOptions,
                                  "snopt": self._SnoptOptions,
                                  "nlopt": self._NloptOptions}[which]()

        ## THE OPTIONS ACTUALLY HANDED TO THE SOLVER, recorded so a run's configuration can be
        ## read off the run rather than reconstructed from the code that produced it. A run's
        ## metadata records `--set` overrides only, which is NOT the same thing: an ADOPTED
        ## DEFAULT reaches the solver without appearing anywhere in the record. Three of those
        ## exist now (`snopt_major_step_limit=0.5`, and NLopt's `LD_MMA` inner optimizer with
        ## its two loose inner tolerances), so without this a future reader cannot tell a run
        ## at today's defaults from one at Drake's.
        ##
        ## The failure this guards against is not hypothetical and not ours: the sibling
        ## `ik-tune` project lost a fourteen-rung verdict to it. Drake PR 25002 deleted a NaN
        ## sentinel by which an unset `local_optimizer_xtol_rel` INHERITED the outer
        ## `xtol_rel`; after the PR it is 1e-6 unconditionally. That is invisible at Drake's
        ## own outer default of 1e-6 and a factor of 100 for anyone who sets the outer value,
        ## and nothing in their run configuration recorded the effective inner tolerance, so
        ## every affected column became unpairable with anything measured after the bump.
        ## Recording what was EMITTED rather than what was SET is what makes that detectable.
        self.emitted_solver_options = {k: dict(v) for k, v in solver_options.options.items()}

        ## NLopt writes no log and ignores this key, so it is set only where a log exists.
        if which != "nlopt":
            solver_options.SetOption(CommonSolverOption.kPrintFileName, self.options.file_print_name)

        self.ResetEvalCounts()

        def record_iterate(vars):
            # Keep the newest iterate on the program, in memory, always -- see RecordIterate
            # for why (abnormal exits are verified from the point the solver had).
            self.RecordIterate(vars)

        self.prog.AddVisualizationCallback(record_iterate, self.lumped_vars)
        
        return solver.Solve(self.prog, solver_options=solver_options)


def visualization_callback(vars, diagram, diagram_context, plant, plant_context, vars_to_q, vars_file, visualize):
    if visualize or vars_file is not None:
        q = vars_to_q(vars)
        if visualize:
            plant.SetPositions(plant_context, q)
            diagram.ForcedPublish(diagram_context)
        if vars_file is not None:
            with open(vars_file, "a") as f:
                f.write(",".join([str(val) for val in vars]) + "\n")