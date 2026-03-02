import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
from src.panda_program import PandaIKProgram
from pydrake.all import (
    StartMeshcat,
)
from tqdm import tqdm


###### OPTIONS ######

num_tests = 1000

######################




meshcat = StartMeshcat()
diagram = BuildEnv(meshcat=meshcat, directives_file = os.path.join(RepoDir(), "models/panda/panda_collision.yaml"))
program = PandaIKProgram(diagram)
program.create_prog()

start = time.time()
i = 0
q = np.zeros(9)
targets = np.zeros((num_tests, 7))
while i < num_tests:
    q[:7] = np.random.uniform(program.plant.GetPositionLowerLimits()[:-2], program.plant.GetPositionUpperLimits()[:-2])
    program.plant.SetPositions(program.plant_context, q)
    if program.collision_free_constraint_eval.Eval(q) < 1:
        pose = program.frame.CalcPoseInWorld(program.plant_context)
        targets[i] = np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()])
        i += 1
print("Generated {} collision-free targets in {:.2f} seconds".format(num_tests, time.time() - start))

for i in tqdm(range(num_tests)):
    target_pose = targets[i]
    program.create_prog(target_pose=target_pose)
    start = time.time()
    result = program.Solve()
    if not result.is_success():
        print("Failed to solve for target:", target_pose)
    else:
        print("Solved IK for target {} in {:.2f} seconds".format(i, time.time() - start))







