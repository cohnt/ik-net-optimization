import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaIKProgram
from pydrake.all import (
    StartMeshcat,
    Quaternion,
)
from tqdm import tqdm
from src.generic_program import ProgramOptions


###### OPTIONS ######

num_tests = 100
program_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-5,
    max_wall_time=20.0,
    which_solver='snopt',
    acceptable_tol = 1e-3,
    acceptable_constr_viol_tol = 1e-4,
    ik_constraint_tol = (0., 0.),
)

######################




meshcat = StartMeshcat()
diagram = BuildEnv(meshcat=meshcat, directives_file = os.path.join(RepoDir(), "models/panda/panda_collision.yaml"))
program = PandaIKProgram(diagram, options=program_options)
program.create_prog()

times = []
successes = 0
costs = []

for i in tqdm(range(100)):
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

    program.create_prog(np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()]))
    pose = program.target_pose
    pose = RigidTransform(Quaternion(pose[3], pose[4], pose[5], pose[6]), [pose[0], pose[1], pose[2]])
    DrawAxes(pose, meshcat)

    program.options.file_print_name = RepoDir() + f"/results/panda/collision_test_{i}.txt"

    start = time.time()
    result = program.Solve()
    if result.is_success():
        times.append(time.time() - start)
        successes += 1
        if program.options.joint_centering_cost != 0:
            costs.append(result.get_optimal_cost() / program.options.joint_centering_cost)

    with open(program.options.file_print_name, 'a') as file:
        file.write(str(program.target_pose))

    

print("Success rate: ", successes / 100)
print("Average time: ", np.mean(times))
print("Average cost: ", np.mean(costs))




