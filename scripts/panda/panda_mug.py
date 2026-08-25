import os, sys, time
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaMugProgram, PandaMugProgramNumerical, PandaMugProgramAnalytic
from src.generic_program import ProgramOptions
from pydrake.all import (
    StartMeshcat,
)
from tqdm import tqdm

####### Options #######
num_tests = 100
visualize = False
seed = 0
program_options = ProgramOptions(
    visualize=visualize,
    joint_centering_cost=1e-4,
    max_wall_time=50.0,
    which_solver='ipopt',
    acceptable_tol = 1e-3,
    acceptable_constr_viol_tol = 1e-4,
    # 1e-6 sat below the flow's noise floor: in float32 the map is only smooth down
    # to ~1e-4, so the solver was chasing a tolerance it could not resolve. Even in
    # float64 there is no reason to centre the gripper to a micron.
    ik_constraint_tol = (1e-4, 0.01),
    mug_height = 0.04,
    use_float64 = True,
    num_seed_samples = 256,
)
# All three formulations share ik_constraint_tol so they solve the same feasible
# problem and the success rates are comparable.
numerical_options = ProgramOptions(
    visualize=visualize,
    joint_centering_cost=1e0,
    max_wall_time=60.0,
    which_solver='ipopt',
    acceptable_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-6,
    ik_constraint_tol = (1e-4, 0.01),
    mug_height = 0.04
)
analytic_options = ProgramOptions(
    visualize=visualize,
    joint_centering_cost=1e-4,
    max_wall_time=60.0,
    which_solver='ipopt',
    acceptable_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-4,
    ik_constraint_tol = (1e-4, 0.01),
    mug_height = 0.04
)

analytic_offset = RigidTransform(
        RotationMatrix([[0, 0., 1.], 
                        [0, -1, 0.],
                        [1., 0, 0.]]),
        np.array([-0.0236, -1.87933e-05, 0.0]),
    )
#######################



np.random.seed(seed)

meshcat = StartMeshcat()
mug_meshcat = StartMeshcat()

yaml_file = os.path.join(RepoDir(), "models/panda/panda_finray_collision.yaml")
base_diagram = BuildEnv(meshcat=meshcat, directives_file = yaml_file)
# Only used to sample targets and to share the loaded network, so skip the seeding work.
program = PandaMugProgram(base_diagram, options=replace(program_options, num_seed_samples=0))
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

learn_successes = 0
learn_times = []
learn_costs = []

numerical_successes = 0
numerical_times = []
numerical_costs = []

analytic_successes = 0
analytic_times = []
analytic_costs = []

for i in tqdm(range(num_tests)):
    diagram_with_mug, mug = GenerateDiagramWithMug(qs[i], program, yaml_file, mug_meshcat)


    # Learned IK
    with HiddenPrints():
        mug_program = PandaMugProgram(diagram_with_mug, options=program_options, model=ik_solver)
        mug_program.SetPositions(qs[i])
        mug_program.create_prog(target_mug=mug)
    mug_program.options.file_print_name = RepoDir() + f"/results/panda/mug/learned/collision_test_{i}.txt"
    start = time.time()
    result = mug_program.Solve()
    if not result.is_success():
        print("Failed IK for target {} in {:.2f} seconds".format(i, time.time() - start))
    else:
        print("Solved IK for target {} in {:.2f} seconds".format(i, time.time() - start))
        learn_times.append(time.time() - start)
        learn_costs.append(result.get_optimal_cost() / mug_program.options.joint_centering_cost)
        learn_successes += 1



    # Numerical IK
    with HiddenPrints():
        numerical_program = PandaMugProgramNumerical(diagram_with_mug, options=numerical_options, model=ik_solver)
    numerical_program.create_prog(target_mug=mug)
    numerical_program.options.file_print_name = RepoDir() + f"/results/panda/mug/numerical/collision_test_{i}.txt"
    start = time.time()
    result = numerical_program.Solve()

    if not result.is_success():
        print("Failed IK for target {} in {:.2f} seconds".format(i, time.time() - start))
    else:
        print("Solved IK for target {} in {:.2f} seconds".format(i, time.time() - start))
        numerical_times.append(time.time() - start)
        numerical_costs.append(result.get_optimal_cost() / numerical_program.options.joint_centering_cost)
        numerical_successes += 1


    # Analytic IK
    with HiddenPrints():
        analytic_program = PandaMugProgramAnalytic(diagram_with_mug, options=analytic_options, model=ik_solver)
    analytic_program.create_prog(target_mug=mug, pose_offset = analytic_offset)
    analytic_program.options.file_print_name = RepoDir() + f"/results/panda/mug/analytic/collision_test_{i}.txt"
    start = time.time()
    result = analytic_program.Solve()

    if not result.is_success():
        print("Failed IK for target {} in {:.2f} seconds".format(i, time.time() - start))
    else:
        print("Solved IK for target {} in {:.2f} seconds".format(i, time.time() - start))
        analytic_times.append(time.time() - start)
        analytic_costs.append(result.get_optimal_cost() / analytic_program.options.joint_centering_cost)
        analytic_successes += 1

print("Learned IK")
print("Solved {} / {} targets in {:.2f} seconds".format(learn_successes, num_tests, sum(learn_times)))
print("Average cost: {:.2f}".format(sum(learn_costs) / len(learn_costs) if learn_costs else 0))
print("Average time: {:.2f}".format(sum(learn_times) / len(learn_times) if learn_times else 0))


print("Numerical IK")
print("Solved {} / {} targets in {:.2f} seconds".format(numerical_successes, num_tests, sum(numerical_times)))
print("Average cost: {:.2f}".format(sum(numerical_costs) / len(numerical_costs) if numerical_costs else 0))
print("Average time: {:.2f}".format(sum(numerical_times) / len(numerical_times) if numerical_times else 0))

print("Analytic IK")
print("Solved {} / {} targets in {:.2f} seconds".format(analytic_successes, num_tests, sum(analytic_times)))
print("Average cost: {:.2f}".format(sum(analytic_costs) / len(analytic_costs) if analytic_costs else 0))
print("Average time: {:.2f}".format(sum(analytic_times) / len(analytic_times) if analytic_times else 0))


