"""Comparison of the orientation-error formulations in the IK constraint.

The orientation row of the IK constraint used to be 2*arccos(|q . q_target|), whose
derivative is infinite at zero error -- exactly where the solver converges -- and whose eps
clamp hands back an AutoDiffXd with an *empty* derivative vector, freezing the row's value
at 2.83e-4 inside theta < 2.8e-4 rad. This script measures whether that actually costs
anything, separating the two effects:

    legacy        2*arccos(clip(d))    today's behaviour: frozen value, empty derivative
    angle         theta                same metric, accurate value, finite derivative
    angle_scaled  theta/ori_tol        the angle metric, scaled to O(1) like `squared`
    squared       theta^2/ori_tol^2    no branch point: the derivative tends to -8

legacy -> angle answers "was the clamp the problem?"; angle -> squared answers "is the
infinite derivative the problem?"; angle_scaled is the control that separates the metric
from the row scaling, since `squared` changes both at once.

All four impose the same feasible set (theta <= ori_tol), so a success-rate difference is
attributable to the gradients or the conditioning rather than to a looser tolerance.

Both formulations that impose an IK constraint are run: the learned one (where the flow's
Jacobian composes with this row) and the numerical one (a clean control with an exact
Jacobian). The analytic formulation never adds an IK constraint -- its pose is a decision
variable -- so it cannot be affected and is not run.

Every arm sees identical targets. Within the learned formulation every arm also starts
from an identical initial guess: the legacy arm performs the multi-start seeding, and the
others are handed its choice, since the seeding ranks candidates by constraint
violation and would otherwise pick different starts for different row scalings.

Usage:  python panda_orientation_headtohead.py [num_tests] [max_wall_time]
"""
import os, re, sys, time, json
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaIKProgram, PandaIKProgramNumerical
from src.generic_program import ProgramOptions
from pydrake.all import StartMeshcat, Quaternion, RigidTransform
from tqdm import tqdm

num_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 50
max_wall_time = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
seed = 0

FORMS = ["legacy", "angle", "angle_scaled", "squared"]

base_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-4,
    max_wall_time=max_wall_time,
    which_solver='ipopt',
    acceptable_tol=1e-4,
    acceptable_constr_viol_tol=1e-4,
    # 1e-4 rather than panda_collision.py's 1e-6, which sits below the flow's noise floor.
    ik_constraint_tol=(1e-4, 0.01),
)
pos_tol, ori_tol = base_options.ik_constraint_tol

log_dir = os.path.join(RepoDir(), "results/panda/orientation")
os.makedirs(log_dir, exist_ok=True)

np.random.seed(seed)
meshcat = StartMeshcat()
yaml_file = os.path.join(RepoDir(), "models/panda/panda_collision.yaml")
with HiddenPrints():
    diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
    # Only used to sample targets and to share the loaded network.
    sampler = PandaIKProgram(diagram, options=replace(base_options, num_seed_samples=0))
    sampler.create_prog()
ik_solver = sampler.ik_solver


## ------------------------------- targets ------------------------------- ##
# Same generation procedure as panda_collision.py: collision-free, gripper below 0.8 m.
targets = []
for _ in tqdm(range(num_tests), desc="targets"):
    q = np.zeros(sampler.plant.num_positions())
    q[7:] = [0.04] * (sampler.plant.num_positions() - 7)
    while True:
        q[:7] = np.random.uniform(low=sampler.plant.GetPositionLowerLimits()[:-2],
                                  high=sampler.plant.GetPositionUpperLimits()[:-2])
        sampler.plant.SetPositions(sampler.plant_context, q)
        if sampler.collision_free_constraint.eval_func(q=q) < 0.1:
            if sampler.frame.CalcPoseInWorld(sampler.plant_context).translation()[2] < 0.8:
                break
    pose = sampler.frame.CalcPoseInWorld(sampler.plant_context)
    targets.append(np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()]))


## ------------------------------ log parsing ----------------------------- ##
_PATTERNS = {
    "iterations": r"Number of Iterations\.*:\s*(\d+)",
    "objective_evals": r"Number of objective function evaluations\s*=\s*(\d+)",
    "constraint_evals": r"Number of inequality constraint evaluations\s*=\s*(\d+)",
    "jacobian_evals": r"Number of inequality constraint Jacobian evaluations\s*=\s*(\d+)",
    "ipopt_seconds": r"Total seconds in IPOPT\s*=\s*([\d.]+)",
}


def parse_log(path):
    '''Iteration and evaluation counts from an IPOPT log. These are the hardware-
    independent cost measure; the Jacobian count is what the flow actually pays for.'''
    out = {k: None for k in _PATTERNS}
    out["exit"] = None
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return out
    for key, pattern in _PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            out[key] = float(m.group(1)) if key == "ipopt_seconds" else int(m.group(1))
    m = re.search(r"EXIT: (.*)", text)
    if m:
        out["exit"] = m.group(1).strip()
    return out


def pose_error(prog, result, target):
    '''True position and orientation error, measured independently of whichever
    orientation form the program was constraining.

    The position constraint is per-axis, so the per-axis maximum is what can be compared
    against pos_tol; the Euclidean norm is up to sqrt(3) times larger by construction.
    '''
    q = prog.VarsToQ(result.GetSolution(prog.lumped_vars))
    translation, wxyz = prog.fk(q)
    achieved = RigidTransform(Quaternion(wxyz), translation)
    desired = RigidTransform(Quaternion(target[3:]), target[:3])
    angle, distance = CalculateError(achieved, desired)
    axis_max = float(np.max(np.abs(np.asarray(translation) - np.asarray(target[:3]))))
    return axis_max, float(distance), float(angle)


## ------------------------------- the runs ------------------------------- ##
FORMULATIONS = {
    "learned": (PandaIKProgram, base_options),
    "numerical": (PandaIKProgramNumerical, replace(base_options, joint_centering_cost=1e0)),
}

results = {f"{formulation}/{form}": {"records": []}
           for formulation in FORMULATIONS for form in FORMS}

for formulation, (cls, options) in FORMULATIONS.items():
    for i in tqdm(range(num_tests), desc=formulation):
        target = targets[i]
        seeded_guess = None
        for form in FORMS:
            # The legacy arm seeds; the others are handed its guess, so the arms differ
            # only in the constraint formulation.
            reuse_guess = formulation == "learned" and seeded_guess is not None
            arm_options = replace(options, orientation_error_form=form,
                                  num_seed_samples=0 if reuse_guess else options.num_seed_samples)
            np.random.seed(seed + i)
            with HiddenPrints():
                prog = cls(diagram, options=arm_options, model=ik_solver)
                prog.create_prog(target)
                if formulation == "learned":
                    if reuse_guess:
                        prog.prog.SetInitialGuess(prog.c, seeded_guess[0])
                        prog.prog.SetInitialGuess(prog.z, seeded_guess[1])
                        prog.prog.SetInitialGuess(prog.correction, seeded_guess[2])
                    else:
                        seeded_guess = (prog.prog.GetInitialGuess(prog.c),
                                        prog.prog.GetInitialGuess(prog.z),
                                        prog.prog.GetInitialGuess(prog.correction))

            log_path = os.path.join(log_dir, f"{formulation}_{form}_{i}.txt")
            if os.path.exists(log_path):
                os.remove(log_path)
            prog.options.file_print_name = log_path

            start = time.time()
            with HiddenPrints():
                result = prog.Solve()
            elapsed = time.time() - start

            record = dict(target=i, wall_time=elapsed, success=bool(result.is_success()))
            record.update(parse_log(log_path))
            if result.is_success():
                record["cost"] = result.get_optimal_cost() / arm_options.joint_centering_cost
                with HiddenPrints():
                    (record["pos_error"], record["pos_dist"],
                     record["ori_error"]) = pose_error(prog, result, target)
            results[f"{formulation}/{form}"]["records"].append(record)


## ------------------------------ reporting ------------------------------- ##
def agg(records, key, successful_only=True):
    vals = [r[key] for r in records
            if r.get(key) is not None and (r["success"] or not successful_only)]
    return float(np.mean(vals)) if vals else float("nan")


for name, block in results.items():
    recs = block["records"]
    ok = [r for r in recs if r["success"]]
    block["summary"] = dict(
        n=len(recs), successes=len(ok),
        mean_wall_time=agg(recs, "wall_time"),
        mean_iterations=agg(recs, "iterations"),
        mean_jacobian_evals=agg(recs, "jacobian_evals"),
        mean_cost=agg(recs, "cost"),
        mean_pos_error=agg(recs, "pos_error"),
        max_pos_error=float(np.max([r["pos_error"] for r in ok])) if ok else float("nan"),
        mean_ori_error=agg(recs, "ori_error"),
        max_ori_error=float(np.max([r["ori_error"] for r in ok])) if ok else float("nan"),
    )

print()
header = (f"{'arm':<20} {'success':>9} {'iters':>8} {'jac evals':>10} {'wall (s)':>9} "
          f"{'cost':>9} {'max ori err':>12} {'max pos err':>12}")
print(header)
print("-" * len(header))
for name, block in results.items():
    s = block["summary"]
    print(f"{name:<20} {s['successes']:>4}/{s['n']:<4} {s['mean_iterations']:>8.1f} "
          f"{s['mean_jacobian_evals']:>10.1f} {s['mean_wall_time']:>9.2f} {s['mean_cost']:>9.3f} "
          f"{s['max_ori_error']:>12.2e} {s['max_pos_error']:>12.2e}")

# Cost and iteration counts are only comparable on targets every arm solved: each arm's
# success set is a different, self-selected subset of the problems.
for formulation in FORMULATIONS:
    arms = [f"{formulation}/{form}" for form in FORMS]
    common = [i for i in range(num_tests)
              if all(results[a]["records"][i]["success"] for a in arms)]
    results[f"_common_{formulation}"] = common
    print(f"\n{formulation}: the {len(common)}/{num_tests} targets all three forms solved")
    print(f"{'arm':<20} {'iters':>8} {'jac evals':>10} {'wall (s)':>9} {'cost':>9} {'ori err':>10}")
    print("-" * 70)
    for a in arms:
        recs = [results[a]["records"][i] for i in common]
        if not recs:
            continue
        def m(key):
            vals = [r[key] for r in recs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")
        print(f"{a:<20} {m('iterations'):>8.1f} {m('jacobian_evals'):>10.1f} "
              f"{m('wall_time'):>9.2f} {m('cost'):>9.3f} {m('ori_error'):>10.2e}")

# Sanity gate: a "success" whose true angular error exceeds the tolerance would mean the
# row scaling is wrong and the comparison is void. IPOPT's "acceptable level" exit permits
# a violation of acceptable_constr_viol_tol *beyond* each bound, so that slack is part of
# the gate: the point is to catch a formulation that quietly bought its success rate with a
# looser tolerance, not to re-derive the solver's own termination criteria.
slack = base_options.acceptable_constr_viol_tol
pos_gate = pos_tol + slack
# The squared row is normalised, so its slack is in units of ori_tol^2: theta <=
# ori_tol*sqrt(1+slack). The angle rows take the slack directly in radians.
ori_gate = {form: (ori_tol * np.sqrt(1.0 + slack) if form == "squared" else ori_tol + slack)
            for form in FORMS}
print(f"\ntolerance gate: position {pos_gate:.2e} m per axis, "
      f"orientation {ori_tol + slack:.4e} rad (angle forms) / "
      f"{ori_tol * np.sqrt(1.0 + slack):.4e} rad (squared)")
for name, block in results.items():
    if not isinstance(block, dict) or "records" not in block:
        continue
    form = name.split("/")[1]
    bad = [r for r in block["records"] if r["success"] and r.get("ori_error", 0) > ori_gate[form]]
    # pos_error is the per-axis maximum, which is what pos_tol bounds.
    bad_p = [r for r in block["records"] if r["success"] and r.get("pos_error", 0) > pos_gate]
    if bad or bad_p:
        print(f"WARNING {name}: {len(bad)} solves exceed the angular tolerance, "
              f"{len(bad_p)} exceed the position tolerance")
    else:
        print(f"{name}: every reported success is within tolerance")

with open(os.path.join(log_dir, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {os.path.join(log_dir, 'summary.json')}")
