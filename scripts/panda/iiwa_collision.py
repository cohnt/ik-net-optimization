import os, sys
import time

from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.iiwa_program import Iiwa14IKProgram, iiwa_limits_lower, iiwa_limits_upper
from pydrake.all import (
    StartMeshcat,
    Quaternion,
    RigidTransform,
    MinimumDistanceLowerBoundConstraint,
    SceneGraphInspector,
)
from src.generic_program import ProgramOptions


program_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-5,
    max_wall_time=20.0,
    which_solver='snopt',
    acceptable_tol = 1e-3,
    acceptable_constr_viol_tol = 1e-4,

)



meshcat = StartMeshcat()
diagram = BuildEnv(meshcat=meshcat, directives_file = os.path.join(RepoDir(), "models/iiwa14/iiwa14_collision.yaml"))

program = Iiwa14IKProgram(diagram, options = program_options)
program.create_prog()

times = []
successes = 0
costs = []

for i in tqdm(range(100)):
    found = False
    while not found:
        q = np.random.uniform(low=iiwa_limits_lower, 
                                        high = iiwa_limits_upper)
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

    program.options.file_print_name = RepoDir() + f"/results/iiwa/collision_test_{i}.txt"

    start = time.time()
    result = program.Solve()
    if result.is_success():
        times.append(time.time() - start)
        successes += 1
        costs.append(result.get_optimal_cost() / program.options.joint_centering_cost)

    with open(program.options.file_print_name, 'a') as file:
        file.write(str(program.target_pose))

    

print("Success rate: ", successes / 100)
print("Average time: ", np.mean(times))
print("Average cost: ", np.mean(costs))