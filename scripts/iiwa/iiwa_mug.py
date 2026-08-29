import os, sys, time
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.iiwa_program import IiwaMugProgram
from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper
from src.generic_program import ProgramOptions
from pydrake.all import (
    StartMeshcat,
)
from tqdm import tqdm

####### Options #######
num_tests = 100
seed = 0
program_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-4,
    max_wall_time=50.0,
    which_solver='ipopt',
    acceptable_tol = 1e-3,
    acceptable_constr_viol_tol = 1e-4,
    # Above the flow's noise floor; see scripts/panda/panda_mug.py.
    ik_constraint_tol = (1e-4, 0.01),
    mug_height = 0.04,
    use_float64 = True,
)
#######################

np.random.seed(seed)

meshcat = StartMeshcat()
mug_meshcat = StartMeshcat()

yaml_file = os.path.join(RepoDir(), "models/iiwa14/iiwa14_collision.yaml")
base_diagram = BuildEnv(meshcat=meshcat, directives_file = yaml_file)
# Only used to sample targets and to share the loaded network, so skip the seeding work.
program = IiwaMugProgram(base_diagram, options=program_options)
program.create_prog()
ik_solver = program.ik_solver  # shared so the network is loaded once, not per target

results_dir = os.path.join(RepoDir(), "results/iiwa/mug/learned")
os.makedirs(results_dir, exist_ok=True)

# Sample collision-free configurations and use the gripper pose of each as a mug to grasp,
# so every target is known to admit at least one valid grasp.
start = time.time()
i = 0
qs = np.zeros((num_tests, 7))
targets = np.zeros((num_tests, 7))
while i < num_tests:
    q = np.random.uniform(iiwa_limits_lower, iiwa_limits_upper)
    q_full = program.PadQ(q)
    program.plant.SetPositions(program.plant_context, q_full)
    if program.collision_free_constraint_eval.Eval(q_full) < 1:
        pose = program.frame.CalcPoseInWorld(program.plant_context)
        targets[i] = np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()])
        qs[i] = q
        i += 1
print("Generated {} collision-free targets in {:.2f} seconds".format(num_tests, time.time() - start))

learn_successes = 0
learn_times = []
learn_costs = []

for i in tqdm(range(num_tests)):
    diagram_with_mug, mug = GenerateDiagramWithMug(program.PadQ(qs[i]), program, yaml_file, mug_meshcat)

    with HiddenPrints():
        mug_program = IiwaMugProgram(diagram_with_mug, options=program_options, model=ik_solver)
        mug_program.SetPositions(program.PadQ(qs[i]))
        mug_program.create_prog(target_mug=mug)
    mug_program.options.file_print_name = os.path.join(results_dir, f"collision_test_{i}.txt")
    start = time.time()
    result = mug_program.Solve()

    if not result.is_success():
        print("Failed IK for target {} in {:.2f} seconds".format(i, time.time() - start))
    else:
        print("Solved IK for target {} in {:.2f} seconds".format(i, time.time() - start))
        learn_times.append(time.time() - start)
        learn_costs.append(result.get_optimal_cost() / mug_program.options.joint_centering_cost)
        learn_successes += 1

print("Learned IK")
print("Solved {} / {} targets in {:.2f} seconds".format(learn_successes, num_tests, sum(learn_times)))
print("Average cost: {:.2f}".format(sum(learn_costs) / len(learn_costs) if learn_costs else 0))
print("Average time: {:.2f}".format(sum(learn_times) / len(learn_times) if learn_times else 0))
