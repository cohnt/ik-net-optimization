"""Ablation of the learned mug formulation.

Runs the learned (IKFlow-in-the-loop) mug grasp on one fixed set of targets under a
sequence of configurations, so the contribution of each change can be read off
independently. Every configuration uses IPOPT, so the numbers are comparable with
each other (the archived results/panda/mug/learned logs were produced with SNOPT
and are *not* a valid baseline for these).

Two of the fixes are structural rather than option-controlled and are therefore
present in all four configurations:
  - the mug IK constraint no longer emits the homogeneous row (lb == ub == 1 with an
    identically zero gradient, which broke LICQ), and
  - the conditioning variable c is boxed near the mug instead of +-5 m.
"""
import os, sys, time, json
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaMugProgram
from src.generic_program import ProgramOptions
from pydrake.all import StartMeshcat
from tqdm import tqdm

####### Options #######
# Usage:  python panda_mug_ablation.py [config_index] [num_tests] [max_wall_time]
# With no config_index every configuration is run in sequence; pass one to run a
# single configuration (useful when each invocation has a wall-clock budget).
num_tests = 25
seed = 0
max_wall_time = 15.0

only_config = int(sys.argv[1]) if len(sys.argv) > 1 else None
if len(sys.argv) > 2:
    num_tests = int(sys.argv[2])
if len(sys.argv) > 3:
    max_wall_time = float(sys.argv[3])

base_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-4,
    max_wall_time=max_wall_time,
    which_solver='ipopt',
    acceptable_tol=1e-3,
    acceptable_constr_viol_tol=1e-4,
    mug_height=0.04,
)

# Cumulative: each row adds one change to the row above it.
configurations = [
    ("original            ", dict(use_float64=False, num_seed_samples=0,   ik_constraint_tol=(1e-6, 0.01))),
    ("+ float64           ", dict(use_float64=True,  num_seed_samples=0,   ik_constraint_tol=(1e-6, 0.01))),
    ("+ tolerance 1e-4    ", dict(use_float64=True,  num_seed_samples=0,   ik_constraint_tol=(1e-4, 0.01))),
    ("+ 256-sample seeding", dict(use_float64=True,  num_seed_samples=256, ik_constraint_tol=(1e-4, 0.01))),
]
if only_config is not None:
    configurations = [configurations[only_config]]
#######################

np.random.seed(seed)
meshcat = StartMeshcat()
mug_meshcat = StartMeshcat()

yaml_file = os.path.join(RepoDir(), "models/panda/panda_finray_collision.yaml")
base_diagram = BuildEnv(meshcat=meshcat, directives_file=yaml_file)
program = PandaMugProgram(base_diagram, options=replace(base_options, num_seed_samples=0))
program.create_prog()
ik_solver = program.ik_solver

# Sample collision-free configurations; the gripper pose of each becomes a mug to grasp,
# so every target is known to admit at least one valid grasp.
qs = np.zeros((num_tests, 7))
i = 0
while i < num_tests:
    q = np.random.uniform(program.plant.GetPositionLowerLimits(), program.plant.GetPositionUpperLimits())
    program.plant.SetPositions(program.plant_context, q)
    if program.collision_free_constraint_eval.Eval(q) < 1:
        qs[i] = q
        i += 1
print(f"Generated {num_tests} collision-free targets")

# Build every mug once so all configurations see identical problems.
mugs = []
for i in range(num_tests):
    diagram_with_mug, mug = GenerateDiagramWithMug(qs[i], program, yaml_file, mug_meshcat)
    mugs.append((diagram_with_mug, mug))

log_dir = os.path.join(RepoDir(), "results/panda/mug/ablation")
os.makedirs(log_dir, exist_ok=True)

results = {}
for name, overrides in configurations:
    options = replace(base_options, **overrides)
    successes, times, costs, setup_times = 0, [], [], []
    np.random.seed(seed)  # same latent draws across configurations

    for i in tqdm(range(num_tests), desc=name.strip()):
        diagram_with_mug, mug = mugs[i]
        t0 = time.time()
        with HiddenPrints():
            mug_program = PandaMugProgram(diagram_with_mug, options=options, model=ik_solver)
            mug_program.SetPositions(qs[i])
            mug_program.create_prog(target_mug=mug)
        setup_times.append(time.time() - t0)
        mug_program.options.file_print_name = os.path.join(log_dir, f"{name.strip().replace(' ', '_')}_{i}.txt")

        start = time.time()
        result = mug_program.Solve()
        elapsed = time.time() - start
        if result.is_success():
            successes += 1
            times.append(elapsed)
            costs.append(result.get_optimal_cost() / options.joint_centering_cost)

    results[name] = dict(
        successes=successes,
        n=num_tests,
        mean_solve=float(np.mean(times)) if times else float('nan'),
        mean_setup=float(np.mean(setup_times)),
        mean_cost=float(np.mean(costs)) if costs else float('nan'),
        median_cost=float(np.median(costs)) if costs else float('nan'),
    )

print()
print(f"{'configuration':<22} {'success':>9} {'solve (s)':>10} {'setup (s)':>10} {'mean cost':>10} {'med cost':>10}")
print("-" * 75)
for name, r in results.items():
    print(f"{name:<22} {r['successes']:>4}/{r['n']:<4} {r['mean_solve']:>10.2f} {r['mean_setup']:>10.2f} "
          f"{r['mean_cost']:>10.2f} {r['median_cost']:>10.2f}")

# Merge into the summary so single-configuration invocations accumulate.
out = os.path.join(log_dir, "summary.json")
merged = {}
if os.path.exists(out):
    with open(out) as f:
        merged = json.load(f)
merged.update(results)
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"\nwrote {out}")
