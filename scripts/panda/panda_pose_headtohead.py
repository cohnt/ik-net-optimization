"""Head-to-head of the joint-space and learned formulations on the collision-free IK
experiment ("Arm on a Table"), on identical targets and an identical wall-clock budget.

    joint-space   decision variables are the joint angles q (the C-space formulation)
    learned       decision variables are the conditioning pose c, the latent z and a
                  correction q_c; q = IKFlow(c, z) + q_c

Both impose the same constraints through the same machinery: the IK pose constraint,
collision avoidance, joint limits, and a quadratic joint-centering cost. The pose
constraint is the current implementation -- three signed rows of the roll-pitch-yaw
residual, `rpy(FK(q)) - rpy(target)`, wrapped to (-pi, pi] and pinned to zero, matching
how ../codebase constrains the end-effector pose. Costs are reported normalised by each
formulation's joint-centering weight so they are comparable.

Usage:  python panda_pose_headtohead.py [num_tests] [max_wall_time]
"""
import os, re, sys, time, json
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaIKProgram, PandaIKProgramNumerical
from src.generic_program import ProgramOptions, orientation_error_rpy
from pydrake.all import StartMeshcat, Quaternion, RigidTransform, RollPitchYaw, RotationMatrix
from tqdm import tqdm

num_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 50
max_wall_time = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
seed = 0

base_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-4,
    max_wall_time=max_wall_time,
    which_solver='ipopt',
    acceptable_tol=1e-4,
    acceptable_constr_viol_tol=1e-4,
    # 1e-4 rather than panda_collision.py's 1e-6, which sits below the flow's noise floor.
    # The orientation entry is unused by the rpy form, which pins the residual to zero.
    ik_constraint_tol=(1e-4, 0.01),
)

log_dir = os.path.join(RepoDir(), "results/panda/pose_headtohead")
os.makedirs(log_dir, exist_ok=True)

np.random.seed(seed)
meshcat = StartMeshcat()
yaml_file = os.path.join(RepoDir(), "models/panda/panda_collision.yaml")
with HiddenPrints():
    diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
    # Only used to sample targets and to share the loaded network.
    sampler = PandaIKProgram(diagram, options=base_options)
    sampler.create_prog()
ik_solver = sampler.ik_solver


## ------------------------------- targets ------------------------------- ##
# Same generation procedure as panda_collision.py: collision-free, gripper below 0.8 m.
# Sampling a configuration and taking its FK guarantees every target is reachable.
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
    independent cost measure; for the learned formulation the constraint-Jacobian count is
    what the flow actually pays for, one jacrev through the network each.'''
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
    '''Achieved pose error, measured from the solution rather than from the solver's own
    constraint values. The position constraint is per-axis, so the per-axis maximum is
    what compares against the gate; the Euclidean norm is up to sqrt(3) times larger.'''
    q = prog.VarsToQ(result.GetSolution(prog.lumped_vars))
    translation, wxyz = prog.fk(q)
    achieved = RigidTransform(Quaternion(wxyz), translation)
    desired = RigidTransform(Quaternion(target[3:]), target[:3])
    angle, distance = CalculateError(achieved, desired)
    axis_max = float(np.max(np.abs(np.asarray(translation) - np.asarray(target[:3]))))
    residual = orientation_error_rpy(
        wxyz, RollPitchYaw(RotationMatrix(Quaternion(target[3:]))).vector())
    rpy_max = float(np.max(np.abs(np.asarray(residual, dtype=float))))
    return axis_max, float(distance), float(angle), rpy_max


## ------------------------------- the runs ------------------------------- ##
# The joint-centering weights differ, as they do in the other head-to-head scripts; costs
# are divided by the weight below so the two are on the same scale.
FORMULATIONS = {
    "learned": (PandaIKProgram, base_options),
    "joint-space": (PandaIKProgramNumerical, replace(base_options, joint_centering_cost=1e0)),
}

results = {name: {"records": []} for name in FORMULATIONS}

for name, (cls, options) in FORMULATIONS.items():
    for i in tqdm(range(num_tests), desc=name):
        target = targets[i]
        np.random.seed(seed + i)
        with HiddenPrints():
            prog = cls(diagram, options=options, model=ik_solver)
            prog.create_prog(target)

        log_path = os.path.join(log_dir, f"{name}_{i}.txt")
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
            record["cost"] = result.get_optimal_cost() / options.joint_centering_cost
            with HiddenPrints():
                (record["pos_error"], record["pos_dist"], record["ori_error"],
                 record["rpy_error"]) = pose_error(prog, result, target)
        results[name]["records"].append(record)


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
        median_cost=float(np.median([r["cost"] for r in ok])) if ok else float("nan"),
        mean_pos_error=agg(recs, "pos_error"),
        max_pos_error=float(np.max([r["pos_error"] for r in ok])) if ok else float("nan"),
        mean_ori_error=agg(recs, "ori_error"),
        max_rpy_error=float(np.max([r["rpy_error"] for r in ok])) if ok else float("nan"),
    )

print()
header = (f"{'formulation':<14} {'success':>9} {'iters':>8} {'jac evals':>10} {'wall (s)':>9} "
          f"{'mean cost':>10} {'med cost':>9} {'max rpy err':>12} {'max pos err':>12}")
print(header)
print("-" * len(header))
for name, block in results.items():
    s = block["summary"]
    print(f"{name:<14} {s['successes']:>4}/{s['n']:<4} {s['mean_iterations']:>8.1f} "
          f"{s['mean_jacobian_evals']:>10.1f} {s['mean_wall_time']:>9.2f} {s['mean_cost']:>10.3f} "
          f"{s['median_cost']:>9.3f} {s['max_rpy_error']:>12.2e} {s['max_pos_error']:>12.2e}")

# Cost is only comparable on targets both formulations solved: each one's success set is a
# different, self-selected subset of the problems.
common = [i for i in range(num_tests)
          if all(results[n]["records"][i]["success"] for n in FORMULATIONS)]
results["_common_targets"] = common
print(f"\non the {len(common)}/{num_tests} targets both formulations solved")
print(f"{'formulation':<14} {'iters':>8} {'jac evals':>10} {'wall (s)':>9} {'mean cost':>10} {'med cost':>9}")
print("-" * 64)
for name in FORMULATIONS:
    recs = [results[name]["records"][i] for i in common]
    def m(key):
        vals = [r[key] for r in recs if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")
    costs = [r["cost"] for r in recs]
    print(f"{name:<14} {m('iterations'):>8.1f} {m('jacobian_evals'):>10.1f} {m('wall_time'):>9.2f} "
          f"{np.mean(costs) if costs else float('nan'):>10.3f} "
          f"{np.median(costs) if costs else float('nan'):>9.3f}")

# Failure modes, since "timed out" and "declared infeasible" mean different things.
print()
for name, block in results.items():
    if not isinstance(block, dict) or "records" not in block:
        continue
    fails = [r for r in block["records"] if not r["success"]]
    modes = {}
    for r in fails:
        key = (r["exit"] or "no log")[:52]
        modes[key] = modes.get(key, 0) + 1
    print(f"{name}: {len(fails)} failures {modes if modes else ''}")

# Sanity gate: the pose constraint is an equality on all six rows, so a success may only
# miss it by the solver's own constraint-violation slack. This comment used to say
# "equality" while the code below conceded `pos_tol + slack` -- because the position rows
# were in fact a +-1e-4 box at the time. They are now `lb = ub = 0` (see
# IKFlowProgram.CreateIKConstraint), so the gate is the solver's slack on every row.
slack = base_options.acceptable_constr_viol_tol
print(f"\ntolerance gate: {slack:.2e} on every row (position, m per axis; rpy, rad)")
for name, block in results.items():
    if not isinstance(block, dict) or "records" not in block:
        continue
    bad = [r for r in block["records"] if r["success"] and r.get("rpy_error", 0) > slack]
    bad_p = [r for r in block["records"] if r["success"] and r.get("pos_error", 0) > slack]
    if bad or bad_p:
        print(f"WARNING {name}: {len(bad)} solves exceed the rpy residual gate, "
              f"{len(bad_p)} exceed the position gate")
    else:
        print(f"{name}: every reported success is within tolerance")

with open(os.path.join(log_dir, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nwrote {os.path.join(log_dir, 'summary.json')}")
