import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaIKProgram, PandaIKProgramNumerical, PandaIKProgramAnalytic
from pydrake.all import (
    StartMeshcat,
    Quaternion,
    RigidTransform,
    RotationMatrix,
)
from tqdm import tqdm
from src.generic_program import ProgramOptions


###### OPTIONS ######

n = 100
program_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-4,
    max_wall_time=20.0,
    which_solver='ipopt',
    acceptable_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-4,
    ik_constraint_tol = (1e-6, 0.01),
)

numerical_program_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e0,
    max_wall_time=20.0,
    which_solver='ipopt',
    acceptable_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-4,
    ik_constraint_tol = (1e-6, 0.01),
)

analytic_program_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-2,
    max_wall_time=20.0,
    which_solver='ipopt',
    acceptable_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-4,
    ik_constraint_tol = (1e-6, 0.01),
)
######################




meshcat = StartMeshcat()
diagram = BuildEnv(meshcat=meshcat, directives_file = os.path.join(RepoDir(), "models/panda/panda_collision.yaml"))
program = PandaIKProgram(diagram, options=program_options)
program.create_prog()

numerical_program = PandaIKProgramNumerical(diagram, options=numerical_program_options)
analytic_program = PandaIKProgramAnalytic(diagram, options=analytic_program_options)

old_times = []
old_successes = 0
old_costs = []

analytic_times = []
analytic_successes = 0
analytic_costs = []

times = []
successes = 0
costs = []

for i in tqdm(range(n)):
    found = False
    q = np.zeros(program.plant.num_positions())
    q[7:] = [0.04] * (program.plant.num_positions() - 7)
    while not found:
        q[:7] = np.random.uniform(low=program.plant.GetPositionLowerLimits()[:-2], 
                                        high = program.plant.GetPositionUpperLimits()[:-2])
        program.plant.SetPositions(program.plant_context, q)
        if program.collision_free_constraint.eval_func(q = q) < 0.1:
            if program.frame.CalcPoseInWorld(program.plant_context).translation()[2] < 0.8:
                found = True
    program.diagram.ForcedPublish(program.diagram_context)
    pose = program.frame.CalcPoseInWorld(program.plant_context)


    ## Learned
    program.create_prog(np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()]))
    pose = program.target_pose
    pose = RigidTransform(Quaternion(pose[3], pose[4], pose[5], pose[6]), [pose[0], pose[1], pose[2]])
    DrawAxes(pose, meshcat)

    program.options.file_print_name = RepoDir() + f"/results/panda/learned/collision_test_{i}.txt"

    start = time.time()
    result = program.Solve()
    if result.is_success():
        times.append(time.time() - start)
        successes += 1
        if program.options.joint_centering_cost != 0:
            costs.append(result.get_optimal_cost() / program.options.joint_centering_cost)

    with open(program.options.file_print_name, 'a') as file:
        file.write(str(program.target_pose))


    ## Old
    numerical_program.create_prog(np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()]))
    numerical_program.options.file_print_name = RepoDir() + f"/results/panda/old/collision_test_{i}.txt"

    start = time.time()
    result = numerical_program.Solve()
    if result.is_success():
        old_times.append(time.time() - start)
        old_successes += 1
        if numerical_program.options.joint_centering_cost != 0:
            old_costs.append(result.get_optimal_cost() / numerical_program.options.joint_centering_cost)

    ## Analytic
    program.plant.SetPositions(program.plant_context, np.zeros(9))
    program.diagram.ForcedPublish(program.diagram_context)
    time.sleep(0.1)
    analytic_pose = np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()])
    analytic_offset = RigidTransform(
        RotationMatrix.Identity(),
        np.array([0.0, 0.0, 0.1034]),
    )
    analytic_program.create_prog(analytic_pose, pose_offset=analytic_offset)
    analytic_program.options.file_print_name = RepoDir() + f"/results/panda/analytic/collision_test_{i}.txt"

    start = time.time()
    result = analytic_program.Solve()
    if result.is_success():
        analytic_times.append(time.time() - start)
        analytic_successes += 1
        if analytic_program.options.joint_centering_cost != 0:
            analytic_costs.append(result.get_optimal_cost() / analytic_program.options.joint_centering_cost)

    

print("Success rate: ", successes / n)
print("Average time: ", np.mean(times))
print("Average cost: ", np.mean(costs))


print("Old Success rate: ", old_successes / n)
print("Old Average time: ", np.mean(old_times))
print("Old Average cost: ", np.mean(old_costs))

print("Analytic Success rate: ", analytic_successes / n)
print("Analytic Average time: ", np.mean(analytic_times))
print("Analytic Average cost: ", np.mean(analytic_costs))


