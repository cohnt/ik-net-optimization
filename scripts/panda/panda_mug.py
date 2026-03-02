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
options = ProgramOptions(ik_constraint_tol = (1e-3, 0))
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


for i in tqdm(range(num_tests)):
    diagram_with_mug, mug = GenerateDiagramWithMug(qs[i], program, yaml_file, mug_meshcat)
    with HiddenPrints():
        program = PandaMugProgram(diagram_with_mug, options=options)
        program.SetPositions(qs[i])
    for j in range(num_initial_guesses):
        program.create_prog(target_mug=mug)
        start = time.time()
        result = program.Solve()
        if not result.is_success():
            print("Failed IK for target {} in {:.2f} seconds".format(i, time.time() - start))
        else:
            print("Solved IK for target {} in {:.2f} seconds".format(i, time.time() - start))





