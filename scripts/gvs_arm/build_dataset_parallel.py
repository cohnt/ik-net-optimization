"""Build a GVS push-rod arm dataset with MANY PROCESSES, in ikflow's exact on-disk format.

WHY NOT ikflow's build_dataset.py. Its `save_dataset_to_disk` is one process calling
`robot.sample_joint_angles_and_poses` for the whole set. On the soft PCS arm that was a
closed-form batched map at 14 us/config and 25M took eleven minutes. Here every sample is a
Newton solve on SoRoMoX's rod, and the batched JAX solve does NOT parallelise across a
node's cores: measured 14.9 ms/sample on a 96-core xeon-p8 node with XLA's pool unpinned,
against 3.4 ms/sample on a laptop with two threads -- 8 h per 2M samples, 100 h per 25M.
`jax.pmap` over host devices is refused by lineax under optimistix's root-find, so the
parallelism has to be at the PROCESS level: N workers, each with its own single-threaded
JAX, each drawing its own share with its own seed, concatenated by the parent.

THE FILES ARE ikflow's. Names from `ikflow.utils.get_dataset_filepaths` with the
`non-self-colliding` tag, float32 tensors, the same per-column std and joint-limit sanity
checks, an `info.txt`; `train_flow.sh` and ikflow's `IkfLitModel` cannot tell the
difference, and `.DONE` is written by `cluster/build_dataset_job.sh` exactly as before.

REPRODUCIBILITY. Worker `i` seeds numpy with `(seed, split, i)` through
`np.random.seed`, which is what jrl's `sample_joint_angles` reads; the dataset is a function
of `(robot, sizes, seed, workers)` -- the worker count is part of the seed and is recorded
in `info.txt`.

    python scripts/gvs_arm/build_dataset_parallel.py --robot_name=gvs_pushrod9_o1 \\
        --training_set_size=25000000 --workers=96
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

## ONE THREAD PER WORKER, set before numpy/torch/jax load in this process or any spawned
## child (children inherit the environment). Without this every worker's OpenBLAS and
## torch OpenMP pools spin up one thread per core: 96 workers x 64 threads blew through the
## node's RLIMIT_NPROC of 4096 at startup on the first cluster run and killed the build.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "GVS_ARM_XLA_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

TEST_SET_SIZE = 15000          # ikflow's build_dataset.py
TAG = "non-self-colliding"     # ikflow.config.DATASET_TAG_NON_SELF_COLLIDING


def _worker(job):
    """One process: its own CPU slice, its own single-threaded JAX, its own seed and share."""
    robot_name, count, seed, split, index, cpus, batch = job
    ## PIN THE AFFINITY FIRST. XLA sizes its per-process Eigen and compiler pools from the
    ## CPUs the process may run on, not from XLA_FLAGS (jax 0.11 ignored the threads flag:
    ## 96 unpinned workers each grew ~420 threads, oversubscribed a 48-core node 10x, and
    ## the whole build sat idle on futexes for hours). A worker pinned to its own slice gets
    ## a pool of that size, and the node runs one JAX thread per CPU as intended.
    if cpus:
        os.sched_setaffinity(0, set(cpus))
    os.environ["GVS_ARM_XLA_THREADS"] = str(max(1, len(cpus) if cpus else 1))
    os.environ["GVS_ARM_SAMPLE_BATCH"] = str(batch)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    np.random.seed(int(np.random.SeedSequence([seed, split, index]).generate_state(1)[0]))
    import torch
    torch.set_num_threads(1)
    import src.register_robots  # noqa: F401
    from jrl.robots import get_robot
    robot = get_robot(robot_name)
    start = time.time()
    samples, poses = robot.sample_joint_angles_and_poses(count, only_non_self_colliding=True)
    return (index, samples.astype(np.float32), poses.astype(np.float32),
            int(robot.rejected_unconverged), time.time() - start)


def _cpu_slices(workers):
    """Partition this process's CPU set into `workers` contiguous slices."""
    cpus = sorted(os.sched_getaffinity(0))
    return [cpus[i::workers] for i in range(workers)] if cpus else [[] for _ in range(workers)]


def _draw(robot_name, total, seed, split, workers, pool, batch, label, timeout):
    shares = [total // workers + (1 if i < total % workers else 0) for i in range(workers)]
    slices = _cpu_slices(workers)
    jobs = [(robot_name, n, seed, split, i, slices[i], batch)
            for i, n in enumerate(shares) if n > 0]
    ## `imap_unordered` with a per-result timeout: a worker that hangs fails the job with a
    ## named worker instead of holding the node until the wall clock, and each finished
    ## worker prints a line, so progress is visible in the log.
    results, start = [], time.time()
    iterator = pool.imap_unordered(_worker, jobs)
    for k in range(len(jobs)):
        result = iterator.next(timeout=timeout)
        results.append(result)
        if k < 3 or (k + 1) % 8 == 0 or k == len(jobs) - 1:
            print(f"    {label}: worker {result[0]} done, {len(result[1]):,} samples in "
                  f"{result[4]:.0f} s ({result[4] / max(len(result[1]), 1) * 1e3:.1f} ms each); "
                  f"{k + 1}/{len(jobs)} workers finished at {time.time() - start:.0f} s",
                  flush=True)
    results.sort(key=lambda r: r[0])
    samples = np.concatenate([r[1] for r in results], axis=0)
    poses = np.concatenate([r[2] for r in results], axis=0)
    rejected = sum(r[3] for r in results)
    slowest = max(r[4] for r in results)
    return samples, poses, rejected, slowest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot_name", required=True)
    p.add_argument("--training_set_size", type=int, default=25_000_000)
    p.add_argument("--test_set_size", type=int, default=TEST_SET_SIZE)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("DATASET_WORKERS", 0)) or None,
                   help="default: half the CPUs this process may run on (one worker per "
                        "physical core when the cpuset lists both hyperthreads), or "
                        "$DATASET_WORKERS")
    p.add_argument("--batch", type=int, default=int(os.environ.get("GVS_ARM_SAMPLE_BATCH", 4096)),
                   help="samples per vmapped solve inside each worker")
    p.add_argument("--worker_timeout", type=float, default=3 * 3600,
                   help="seconds to wait for any single worker's result before failing")
    p.add_argument("--only_non_self_colliding", action="store_true", default=True,
                   help="always on; accepted so build_dataset_job.sh's command line is unchanged")
    args = p.parse_args()
    if not args.workers:
        args.workers = max(1, len(os.sched_getaffinity(0)) // 2)

    ## Each JAX worker still owns ~100 threads (XLA's per-core Eigen pool, idle, plus its
    ## compiler's), and a process limit that cannot hold `workers x 100` kills the build at
    ## startup rather than slowing it. Cap the worker count by the SOFT limit, so a job whose
    ## `ulimit -u` could not be raised runs slowly instead of dying; say so in the log.
    import resource
    soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY:
        allowed = max(1, (soft - 256) // 160)
        if allowed < args.workers:
            print(f"  process limit {soft} allows ~{allowed} JAX workers; using that instead of "
                  f"{args.workers} (raise `ulimit -u` to use every CPU)", flush=True)
            args.workers = allowed

    import torch
    from ikflow.utils import get_dataset_directory, get_dataset_filepaths, safe_mkdir
    from ikflow.utils import assert_joint_angle_tensor_in_joint_limits, print_tensor_stats

    directory = get_dataset_directory(args.robot_name)
    safe_mkdir(directory)
    paths = get_dataset_filepaths(directory, [TAG])
    samples_tr_path, poses_tr_path, samples_te_path, poses_te_path, info_path = paths
    print(f"{args.robot_name}: {args.training_set_size:,} train + {args.test_set_size:,} test "
          f"with {args.workers} workers (batch {args.batch}) over CPUs "
          f"{sorted(os.sched_getaffinity(0))[:4]}... -> {directory}", flush=True)

    ## `spawn`, not `fork`: the parent must never have initialised JAX, and a forked JAX is
    ## undefined behaviour anyway.
    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        t0 = time.time()
        samples_te, poses_te, rejected_te, _ = _draw(
            args.robot_name, args.test_set_size, args.seed, 1, args.workers, pool,
            args.batch, "test", args.worker_timeout)
        print(f"  test set: {len(samples_te):,} in {time.time() - t0:.0f} s "
              f"(rejected unconverged: {rejected_te})", flush=True)
        t1 = time.time()
        samples_tr, poses_tr, rejected_tr, slowest = _draw(
            args.robot_name, args.training_set_size, args.seed, 0, args.workers, pool,
            args.batch, "train", args.worker_timeout)
        elapsed = time.time() - t1
        print(f"  training set: {len(samples_tr):,} in {elapsed:.0f} s "
              f"({elapsed / max(len(samples_tr), 1) * 1e6:.1f} us/sample over the node; slowest "
              f"worker {slowest:.0f} s; rejected unconverged: {rejected_tr})", flush=True)

    samples_tr = torch.tensor(samples_tr, dtype=torch.float32)
    poses_tr = torch.tensor(poses_tr, dtype=torch.float32)
    samples_te = torch.tensor(samples_te, dtype=torch.float32)
    poses_te = torch.tensor(poses_te, dtype=torch.float32)
    for arr in (samples_tr, samples_te, poses_tr, poses_te):
        for i in range(arr.shape[1]):
            assert torch.std(arr[:, i]) > 0.001, f"column {i} has zero stdev"
    limits = [(-1.0, 1.0)] * samples_tr.shape[1]
    assert_joint_angle_tensor_in_joint_limits(limits, samples_tr, "samples_tr", 0.0)
    assert_joint_angle_tensor_in_joint_limits(limits, samples_te, "samples_te", 0.0)

    with open(info_path, "w") as f:
        f.write("Dataset info (scripts/gvs_arm/build_dataset_parallel.py)\n")
        f.write(f"  robot:             {args.robot_name}\n")
        f.write(f"  dataset_directory: {directory}\n")
        f.write(f"  training_set_size: {args.training_set_size}\n")
        f.write(f"  test_set_size:     {args.test_set_size}\n")
        f.write(f"  seed:              {args.seed}\n")
        f.write(f"  workers:           {args.workers}  (part of the seed)\n")
        f.write(f"  batch:             {args.batch}\n")
        f.write(f"  rejected_unconverged: train {rejected_tr}, test {rejected_te}\n")
        f.write(f"  seconds_training:  {elapsed:.0f}\n")
        print_tensor_stats(samples_tr, writable=f, name="samples_tr")
        print_tensor_stats(poses_tr, writable=f, name="poses_tr")
        print_tensor_stats(samples_te, writable=f, name="samples_te")
        print_tensor_stats(poses_te, writable=f, name="poses_te")

    torch.save(samples_tr, samples_tr_path)
    torch.save(poses_tr, poses_tr_path)
    torch.save(samples_te, samples_te_path)
    torch.save(poses_te, poses_te_path)
    print(f"wrote {directory}: {os.path.basename(samples_tr_path)} {tuple(samples_tr.shape)}, "
          f"{os.path.basename(poses_tr_path)} {tuple(poses_tr.shape)}, test {tuple(samples_te.shape)}")


if __name__ == "__main__":
    main()
