"""Head-to-head of the three mug formulations on identical targets and an identical
wall-clock budget.

The archived comparison was not budget-matched: the learned runs were capped at a
different max_wall_time than the numerical and analytic runs, and every learned failure
was a timeout rather than an infeasibility. This script gives all three the same targets,
the same cap and the same constraint tolerance, which is the only setup in which
"formulation A beats formulation B" is a meaningful statement.

Usage:  python panda_mug_headtohead.py [num_tests] [max_wall_time]
"""
import os, sys, time, json
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaMugProgram, PandaMugProgramNumerical, PandaMugProgramAnalytic
from src.generic_program import ProgramOptions
from pydrake.all import StartMeshcat, RigidTransform, RotationMatrix
from tqdm import tqdm

num_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 12
max_wall_time = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
seed = 0

# Same seed and generation procedure as panda_mug_ablation.py, so the targets match.
base_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-4,
    max_wall_time=max_wall_time,
    which_solver='ipopt',
    acceptable_tol=1e-3,
    acceptable_constr_viol_tol=1e-4,
    ik_constraint_tol=(1e-4, 0.01),
    mug_height=0.04,
)
analytic_offset = RigidTransform(
    RotationMatrix([[0, 0., 1.], [0, -1, 0.], [1., 0, 0.]]),
    np.array([-0.0236, -1.87933e-05, 0.0]))

np.random.seed(seed)
meshcat = StartMeshcat()
mug_meshcat = StartMeshcat()

yaml_file = os.path.join(RepoDir(), "models/panda/panda_finray_collision.yaml")
base_diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
program = PandaMugProgram(base_diagram, options=replace(base_options, num_seed_samples=0))
program.create_prog()
ik_solver = program.ik_solver

qs = np.zeros((num_tests, 7))
i = 0
while i < num_tests:
    q = np.random.uniform(program.plant.GetPositionLowerLimits(), program.plant.GetPositionUpperLimits())
    program.plant.SetPositions(program.plant_context, q)
    if program.collision_free_constraint_eval.Eval(q) < 1:
        qs[i] = q
        i += 1

mugs = [GenerateDiagramWithMug(qs[i], program, yaml_file, mug_meshcat) for i in range(num_tests)]

log_dir = os.path.join(RepoDir(), "results/panda/mug/headtohead")
os.makedirs(log_dir, exist_ok=True)

# The learned formulation gets every improvement; the other two are unchanged.
methods = {
    "learned  ": (PandaMugProgram, replace(base_options, use_float64=True, num_seed_samples=256), {}),
    "numerical": (PandaMugProgramNumerical, replace(base_options, joint_centering_cost=1e0), {}),
    "analytic ": (PandaMugProgramAnalytic, base_options, dict(pose_offset=analytic_offset)),
}

results = {}
for name, (cls, options, extra) in methods.items():
    successes, times, costs, setups = 0, [], [], []
    per_target_cost = [None] * num_tests   # None where the method failed
    per_target_time = [None] * num_tests
    np.random.seed(seed)
    for i in tqdm(range(num_tests), desc=name.strip()):
        diagram_with_mug, mug = mugs[i]
        t0 = time.time()
        with HiddenPrints():
            prog = cls(diagram_with_mug, options=options, model=ik_solver)
            prog.SetPositions(qs[i])
            prog.create_prog(target_mug=mug, **extra)
        setups.append(time.time() - t0)
        prog.options.file_print_name = os.path.join(log_dir, f"{name.strip()}_{i}.txt")
        start = time.time()
        result = prog.Solve()
        elapsed = time.time() - start
        if result.is_success():
            successes += 1
            times.append(elapsed)
            # Normalise by the weight so costs are comparable across methods.
            cost = result.get_optimal_cost() / options.joint_centering_cost
            costs.append(cost)
            per_target_cost[i] = cost
            per_target_time[i] = elapsed
    results[name] = dict(successes=successes, n=num_tests,
                         mean_solve=float(np.mean(times)) if times else float('nan'),
                         mean_setup=float(np.mean(setups)),
                         mean_cost=float(np.mean(costs)) if costs else float('nan'),
                         median_cost=float(np.median(costs)) if costs else float('nan'),
                         per_target_cost=per_target_cost,
                         per_target_time=per_target_time)

print()
print(f"{'method':<12} {'success':>9} {'solve (s)':>10} {'setup (s)':>10} {'mean cost':>10} {'med cost':>10}")
print("-" * 66)
for name, r in results.items():
    print(f"{name:<12} {r['successes']:>4}/{r['n']:<4} {r['mean_solve']:>10.2f} {r['mean_setup']:>10.2f} "
          f"{r['mean_cost']:>10.2f} {r['median_cost']:>10.2f}")

# Cost is only comparable on targets every method solved: each method's own success
# set is a different, self-selected subset of the problems.
common = [i for i in range(num_tests)
          if all(r["per_target_cost"][i] is not None for r in results.values())]
print(f"\ncost on the {len(common)}/{num_tests} targets every method solved:")
print(f"{'method':<12} {'mean cost':>10} {'med cost':>10} {'mean solve':>11}")
print("-" * 46)
for name, r in results.items():
    c = [r["per_target_cost"][i] for i in common]
    t = [r["per_target_time"][i] for i in common]
    print(f"{name:<12} {np.mean(c):>10.2f} {np.median(c):>10.2f} {np.mean(t):>11.2f}")
results["_common_targets"] = common

with open(os.path.join(log_dir, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)
