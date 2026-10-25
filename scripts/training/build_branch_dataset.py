"""Build an ikflow-format dataset restricted to ONE IK branch (LOInK phase 2, 2026-10-24).

The single-branch LOInK charts (`cluster/chart_runs.txt`, rows `<robot>_loinkB...`) train on the
nominal configuration's branch only (`src/branches.py`): a LOInK chart is one bijection, so it
carries one connected solution arc per pose, and a 7-DoF arm's full solution set at a pose is
several. This writes the fork's exact dataset (the four float32 tensors and info.txt named by
`get_dataset_filepaths(dir, [DATASET_TAG_NON_SELF_COLLIDING])`, under
`get_dataset_directory(<dataset_name>)`), so `CVAEDataset(<dataset_name>, ...)` loads it unchanged.

It follows `third_party/ikflow/scripts/build_dataset.py:save_dataset_to_disk` -- seed np and
torch, the test set first and then the training set, the fork's `joint_limit_eps`, jrl's
uniform joint-angle draw (bare `np.random`, seeded here), jrl's Klampt self-collision filter
and Klampt FK, the same sanity checks and `print_tensor_stats` info file -- with two
differences, both deliberate:

  * Rejection onto the branch happens BEFORE the self-collision check and FK (vectorised
    `OnBranch` on each drawn chunk). The kept rows are exactly the uniform draw restricted
    to (branch AND non-self-colliding), as filtering after would give, but only the kept
    ~1/8 (iiwa) to ~1/4 (Panda) pay the ~14 us Klampt check. Of the same stream, so the
    dataset is a function of (robot, branch, sizes, seed).
  * Each drawn chunk is rounded to float32 FIRST, so the branch, the collision check and FK
    all see the values the tensors store. A row within float32 rounding of a branch
    boundary (B is a continuous dot product) would otherwise be saved off-branch and trip
    the trainer's on-branch guard.

Run it through the cluster job, never locally at full size (Thomas, 2026-10-24):

    DATASET_ROBOT=panda_branch LLsub ./cluster/build_dataset_job.sh -s 48 -q xeon-p8 -T 04:00:00 -J lik_dataset_branch

    python scripts/training/build_branch_dataset.py --robot_name panda --dataset_name panda_branch \
        --training_set_size 25000000 --test_set_size 15000 --seed 0
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

# Before anything resolves a robot, and before ikflow reads HOME for DATASET_DIR
# (`scripts/training/ikflow_entry.py` has the reason).
import src.register_robots  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING  # noqa: E402
from ikflow.utils import (  # noqa: E402
    assert_joint_angle_tensor_in_joint_limits,
    get_dataset_directory,
    get_dataset_filepaths,
    print_tensor_stats,
    safe_mkdir,
)
from jrl.robots import get_robot  # noqa: E402

from src.branches import BranchString, OnBranch, ResolveBranch  # noqa: E402

#: The fork's `build_dataset.py` value, np.deg2rad(0.25).
JOINT_LIMIT_EPS = 0.004363323129985824
CHUNK = 100_000
PROGRESS_EVERY = 1_000_000


class Counts:
    def __init__(self):
        # `collision_free` counts every on-branch row that passed the check, including the
        # tail of the last chunk that is not kept, so the fractions are unbiased by chunking.
        self.drawn = self.on_branch = self.collision_free = self.kept = 0


def SampleOnBranch(robot, n, branch, counts, label):
    """`n` (q, pose) rows, uniform in the padded joint box, on `branch`, not self-colliding."""
    samples = np.zeros((n, robot.ndof))
    poses = np.zeros((n, 7))
    filled, next_report, t0 = 0, PROGRESS_EVERY, time.monotonic()
    while filled < n:
        q = robot.sample_joint_angles(CHUNK, joint_limit_eps=JOINT_LIMIT_EPS)
        q = q.astype(np.float32).astype(np.float64)
        counts.drawn += q.shape[0]
        q = q[OnBranch(robot.name, q, branch)]
        counts.on_branch += q.shape[0]
        free = np.array([not robot.config_self_collides(qi) for qi in q], dtype=bool)
        counts.collision_free += int(free.sum())
        q = q[free][: n - filled]
        if q.shape[0]:
            samples[filled:filled + q.shape[0]] = q
            poses[filled:filled + q.shape[0]] = robot.forward_kinematics_klampt(q)
            filled += q.shape[0]
            counts.kept += q.shape[0]
        if filled >= next_report or filled == n:
            print(f"  {label}: {filled}/{n} rows, {time.monotonic() - t0:.0f} s", flush=True)
            next_report += PROGRESS_EVERY
    return samples, poses


def BuildBranchDataset(robot_name, dataset_name, training_set_size, test_set_size, seed,
                       branch="nominal"):
    robot = get_robot(robot_name)
    branch = ResolveBranch(robot.name, branch)
    tags = [DATASET_TAG_NON_SELF_COLLIDING]
    directory = get_dataset_directory(dataset_name)
    print(f"building {dataset_name}: robot {robot.name}, branch {BranchString(branch)}, "
          f"train {training_set_size}, test {test_set_size}, seed {seed} -> {directory}", flush=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    safe_mkdir(directory)

    counts = Counts()
    t0 = time.monotonic()
    samples_te, poses_te = SampleOnBranch(robot, test_set_size, branch, counts, "test")
    samples_tr, poses_tr = SampleOnBranch(robot, training_set_size, branch, counts, "train")

    samples_tr_path, poses_tr_path, samples_te_path, poses_te_path, info_path = (
        get_dataset_filepaths(directory, tags))
    samples_tr = torch.tensor(samples_tr, dtype=torch.float32)
    poses_tr = torch.tensor(poses_tr, dtype=torch.float32)
    samples_te = torch.tensor(samples_te, dtype=torch.float32)
    poses_te = torch.tensor(poses_te, dtype=torch.float32)

    for arr in (samples_tr, samples_te, poses_tr, poses_te):
        for i in range(arr.shape[1]):
            assert torch.std(arr[:, i]) > 0.001, f"Error: Column {i} in samples has zero stdev"
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples_tr, "samples_tr", 0.0)
    assert_joint_angle_tensor_in_joint_limits(robot.actuated_joints_limits, samples_te, "samples_te", 0.0)
    for name, arr in (("samples_tr", samples_tr), ("samples_te", samples_te)):
        on = OnBranch(robot.name, arr.double().numpy(), branch)
        assert on.all(), f"{name}: {int((~on).sum())} saved rows are off branch {BranchString(branch)}"

    branch_fraction = counts.on_branch / counts.drawn
    collision_free_fraction = counts.collision_free / max(counts.on_branch, 1)
    acceptance = counts.collision_free / counts.drawn
    with open(info_path, "w") as f:
        f.write("Dataset info\n")
        f.write(f"  robot:             {robot.name}\n")
        f.write(f"  dataset_name:      {dataset_name}\n")
        f.write(f"  dataset_directory: {directory}\n")
        f.write(f"  training_set_size: {training_set_size}\n")
        f.write(f"  test_set_size:     {test_set_size}\n")
        f.write(f"  seed:              {seed}\n")
        f.write(f"  branch:            {BranchString(branch)}  (src/branches.py, nominal configuration's)\n")
        f.write(f"  joint_limit_eps:   {JOINT_LIMIT_EPS}\n")
        f.write(f"  drawn:             {counts.drawn}\n")
        f.write(f"  on_branch:         {counts.on_branch}  (fraction {branch_fraction:.4f})\n")
        f.write(f"  collision_free:    {counts.collision_free}  (fraction of on-branch "
                f"{collision_free_fraction:.4f}; overall acceptance {acceptance:.4f})\n")
        f.write(f"  kept:              {counts.kept}\n")
        print_tensor_stats(samples_tr, writable=f, name="samples_tr")
        print_tensor_stats(poses_tr, writable=f, name="poses_tr")
        print_tensor_stats(samples_te, writable=f, name="samples_te")
        print_tensor_stats(poses_te, writable=f, name="poses_te")

    torch.save(samples_tr, samples_tr_path)
    torch.save(poses_tr, poses_tr_path)
    torch.save(samples_te, samples_te_path)
    torch.save(poses_te, poses_te_path)
    print(f"saved {dataset_name} in {time.monotonic() - t0:.1f} s: branch fraction "
          f"{branch_fraction:.4f}, overall acceptance {acceptance:.4f}", flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot_name", required=True, choices=["panda", "iiwa14"])
    parser.add_argument("--dataset_name", default=None,
                        help="dataset directory name (default: src.branches.BRANCH_DATASET[robot])")
    parser.add_argument("--training_set_size", type=int, default=25_000_000)
    parser.add_argument("--test_set_size", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from src.branches import BRANCH_DATASET
    BuildBranchDataset(args.robot_name, args.dataset_name or BRANCH_DATASET[args.robot_name],
                       args.training_set_size, args.test_set_size, args.seed)


if __name__ == "__main__":
    main()
