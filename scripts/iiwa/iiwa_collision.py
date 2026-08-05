import os, sys
import time

from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils import *
sys.path.append(os.path.abspath(os.path.join(RepoDir(), '../analytic-and-optimization-ik')))

from src.iiwa_program import Iiwa14IKProgram, iiwa_limits_lower, iiwa_limits_upper
from pydrake.all import (
    StartMeshcat,
    Quaternion,
    RigidTransform,
    MinimumDistanceLowerBoundConstraint,
    SceneGraphInspector,
)
from src.generic_program import ProgramOptions
from src.iiwa_experiments import IiwaIKProblemOldFormulation, IiwaIKProblemNewFormulation, IiwaProblemOptions, IiwaProblemOptionsNew



learn_options = ProgramOptions(
    visualize=False,
    joint_centering_cost=1e-5,
    max_wall_time=20.0,
    which_solver='ipopt',
    acceptable_tol = 1e-4,
    acceptable_constr_viol_tol = 1e-5,
)

opt_options = IiwaProblemOptions(
    solver = 'IPOPT',
    new_formulation_options=IiwaProblemOptionsNew(
        impose_reachability_barrier_costs=False,
        impose_reachability_barrier_constraint=False,
        # impose_singularity_barrier_costs=False,
        # impose_singularity_constraints=False,
    ),
    impose_joint_centering_cost=True,
    joint_centering_cost_multiplier=1e-3,
    acceptable_constr_viol_tol=1e-4,
    acceptable_tol=1e-3,
    acceptable_dual_inf_tol=1e-3,
    acceptable_compl_inf_tol=1e-3,
    acceptable_iter=1,
    max_wall_time=2.0,
    same_gc=False,
    joint_nominal=False,
    joint_limits=True,
    print_level=5,
)

logfile_path = "/results/iiwa"


meshcat = StartMeshcat()
diagram = BuildEnv(meshcat=meshcat, directives_file = os.path.join(RepoDir(), "models/iiwa14/iiwa14_collision.yaml"))

program = Iiwa14IKProgram(diagram, options = learn_options)
old_formulation = IiwaIKProblemOldFormulation(diagram)
new_formulation = IiwaIKProblemNewFormulation(diagram)
program.create_prog()

learned_times = []
old_times = []
new_times = []
learn_successes = 0
old_successes = 0
new_successes = 0

learned_costs = []
old_optimal_costs = []
new_optimal_costs = []

np.random.seed(50)

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


    ## Learned IK
    program.create_prog(np.array([*pose.translation(), *pose.rotation().ToQuaternion().wxyz()]))
    program.options.file_print_name = RepoDir() + f"{logfile_path}/learned/collision_test_{i}.txt"
    start = time.time()
    result = program.Solve()
    if result.is_success():
        learned_times.append(time.time() - start)
        learn_successes += 1
        learned_costs.append(result.get_optimal_cost() / program.options.joint_centering_cost)
    with open(program.options.file_print_name, 'a') as file:
        file.write(str(program.target_pose) + '\n')



    ## Old Numerical IK
    opt_options.target = old_formulation.ee_frame.CalcPoseInWorld(program.plant_context)


    if logfile_path:
        opt_options.print_file_name = RepoDir() + f"{logfile_path}/old/collision_test_{i}.txt"
    old_formulation.ApplyOptions(opt_options)
    start = time.time()
    old_result = old_formulation.Solve(visualize=False)
    end = time.time()

    if old_result.is_success():
        old_successes += 1
        old_optimal_costs.append(old_formulation.EvalCost(old_result))
    old_times.append(end - start)


    if logfile_path:
        opt_options.print_file_name = RepoDir() + f"{logfile_path}/new/collision_test_{i}.txt"
        new_formulation.ApplyOptions(options=opt_options)

    start = time.time()
    new_result = new_formulation.Solve(visualize=False)
    end = time.time()

    if new_result.is_success():
        new_successes += 1
        new_optimal_costs.append(new_formulation.EvalCost(new_result))
        new_times.append(end - start)






print("Learned IK")
print("Success rate: ", learn_successes / 100)
print("Average time: ", np.mean(learned_times))
print("Average cost: ", np.mean(learned_costs))

print("Old IK")
print("Success rate: ", old_successes / 100)
print("Average time: ", np.mean(old_times))
print("Average cost: ", np.mean(old_optimal_costs))

print("New IK")
print("Success rate: ", new_successes / 100)
print("Average time: ", np.mean(new_times))
print("Average cost: ", np.mean(new_optimal_costs))