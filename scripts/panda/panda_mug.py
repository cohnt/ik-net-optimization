import os, sys, time

from requests import options
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaMugProgram
from src.generic_program import ProgramOptions
from pydrake.all import (
    StartMeshcat,
)
from tqdm import tqdm

####### Options #######
num_tests = 100
num_initial_guesses = 10
program_options = ProgramOptions(
    visualize=True,
    joint_centering_cost=1e-4,
    max_wall_time=60.0,
    which_solver='snopt',
    acceptable_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-4,
    ik_constraint_tol = (1e-6, 0.01),
    mug_height = 0.04
)
#######################



meshcat = StartMeshcat()
mug_meshcat = StartMeshcat()

yaml_file = os.path.join(RepoDir(), "models/panda/panda_finray_collision.yaml")
base_diagram = BuildEnv(meshcat=meshcat, directives_file = yaml_file)
program = PandaMugProgram(base_diagram)
program.create_prog()


start = time.time()
i = 0
q = np.zeros(7)
qs = np.zeros((num_tests, 7))
targets = np.zeros((num_tests, 7))
while i < num_tests:
    q = np.random.uniform(program.plant.GetPositionLowerLimits(), program.plant.GetPositionUpperLimits())
    program.plant.SetPositions(program.plant_context, q)
    if program.collision_free_constraint_eval.Eval(q) < 1:
        pose = program.frame.CalcPoseInWorld(program.plant_context)
        targets[i] = np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()])
        qs[i] = q
        i += 1
print("Generated {} collision-free targets in {:.2f} seconds".format(num_tests, time.time() - start))

ik_solver = program.ik_solver

successes = 0
times = []
costs = []

for i in tqdm(range(num_tests)):
    diagram_with_mug, mug = GenerateDiagramWithMug(qs[i], program, yaml_file, mug_meshcat)
    with HiddenPrints():
        mug_program = PandaMugProgram(diagram_with_mug, options=program_options, model=ik_solver)
        mug_program.SetPositions(qs[i])
    # for j in range(num_initial_guesses):
    mug_program.create_prog(target_mug=mug)
    mug_program.options.file_print_name = RepoDir() + f"/results/panda/mug/learned/collision_test_{i}.txt"
    start = time.time()
    result = mug_program.Solve()
    if not result.is_success():
        print("Failed IK for target {} in {:.2f} seconds".format(i, time.time() - start))
    else:
        print("Solved IK for target {} in {:.2f} seconds".format(i, time.time() - start))
        print(result.get_optimal_cost() / mug_program.options.joint_centering_cost)
        times.append(time.time() - start)
        costs.append(result.get_optimal_cost() / mug_program.options.joint_centering_cost)
        successes += 1
    del mug_program

print("Solved {} / {} targets in {:.2f} seconds".format(successes, num_tests, sum(times)))
print("Average cost: {:.2f}".format(sum(costs) / len(costs) if costs else 0))
print("Average time: {:.2f}".format(sum(times) / len(times) if times else 0))



