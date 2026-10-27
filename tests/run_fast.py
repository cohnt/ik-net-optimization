"""Every test file's `fast` tier (tests/_tiers.py): `python tests/run_fast.py`.

Each file runs in its own fresh process, so nothing one file changes (a program's options, a
patched function) can reach another, and its memory is freed when it ends. `--jobs` files run at
once (default 4, single-threaded each: four cores; Thomas, 2026-10-27, for this runner only --
the project has no other process pool), longest first by the last run's seconds. Files that
write the same path (CONFLICTS) never overlap.

Local rules, applied by default:
- re-executes itself under `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`
  when systemd-run exists (`--no-scope` to skip); the ONE scope covers every file's process;
- CPU only (`CUDA_VISIBLE_DEVICES=` unless already set);
- `--jobs` x `--threads` cores (affinity), `--threads` threads per file. The GVS arm's XLA
  threads stay at 1: the thread count is part of XLA's compile options, so changing it would
  miss the JAX compilation cache (`_tiers.jax_compilation_cache`).

Each file's output goes to `<log dir>/<file>.log`; one line per file here (seconds and peak
memory), and a failing file's log tail. Exit status 1 if any file failed.

    python tests/run_fast.py                     # all 36 files, four at a time
    python tests/run_fast.py --jobs 1            # one at a time
    python tests/run_fast.py svgd gvs            # files whose name contains any of these
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
LOGS = os.path.join(REPO, "results", "_tests_fast")
#: Files that write the same path, so never run at the same time: test_svgd_pin_and_gate solves
#: through test_svgd_solver's `run_cell`, whose solver log (results/_test_svgd_<task>_<arm>.log)
#: test_svgd_solver reads back.
CONFLICTS = [{"test_svgd_solver", "test_svgd_pin_and_gate"}]


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("only", nargs="*", help="run files whose name contains any of these")
    ap.add_argument("--jobs", type=int, default=4, help="files at once (default 4)")
    ap.add_argument("--threads", type=int, default=1, help="threads per file (default 1)")
    ap.add_argument("--no-scope", action="store_true", help="do not re-exec under systemd-run")
    ap.add_argument("--log-dir", default=None,
                    help="default results/_tests_fast/<timestamp> (results/ is gitignored)")
    return ap.parse_args()


def scoped(args):
    """Re-exec under the local memory scope, once."""
    if args.no_scope or os.environ.get("LEARNED_IK_TESTS_SCOPED") or not shutil.which("systemd-run"):
        return
    env = dict(os.environ, LEARNED_IK_TESTS_SCOPED="1")
    cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=20G", "-p", "MemorySwapMax=0",
           sys.executable, *sys.argv]
    sys.exit(subprocess.run(cmd, env=env).returncode)


def environment(threads, cores_wanted):
    """CPU only, `threads` threads per file, on `cores_wanted` cores (inherited by every file)."""
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS", "DRAKE_NUM_THREADS"):
        env[var] = str(threads)
    env["GVS_ARM_XLA_THREADS"] = "1"
    env.setdefault("TQDM_DISABLE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) > cores_wanted:
        os.sched_setaffinity(0, cores[:cores_wanted])
    return env


def conflicts(name, running):
    return any(name in group and running & (group - {name}) for group in CONFLICTS)


def main():
    args = parse()
    scoped(args)
    env = environment(args.threads, args.jobs * args.threads)
    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    if args.only:
        files = [f for f in files if any(o in os.path.basename(f) for o in args.only)]
    log_dir = args.log_dir or os.path.join(LOGS, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    ## Longest first, by the last run's seconds (a file never timed goes first).
    last_path = os.path.join(LOGS, "last_seconds.json")
    try:
        with open(last_path) as f:
            last = json.load(f)
    except (OSError, ValueError):
        last = {}
    name_of = lambda path: os.path.splitext(os.path.basename(path))[0]
    queue = sorted(files, key=lambda path: -last.get(name_of(path), float("inf")))
    print(f"{len(files)} files, fast tier, {args.jobs} at a time, {args.threads} thread(s) each, "
          f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r}; logs in {log_dir}", flush=True)

    running, failed, seconds, t_all = {}, [], {}, time.perf_counter()
    while queue or running:
        ## Start what fits: free slots, no conflict with a running file.
        for path in list(queue):
            if len(running) >= args.jobs:
                break
            name = name_of(path)
            if conflicts(name, {r[0] for r in running.values()}):
                continue
            queue.remove(path)
            log = os.path.join(log_dir, name + ".log")
            with open(log, "w") as out:
                proc = subprocess.Popen([sys.executable, path], cwd=REPO, env=env, stdout=out,
                                        stderr=subprocess.STDOUT)
            running[proc.pid] = (name, log, time.perf_counter())
        ## Reap one finished file; its own peak memory comes with it.
        pid, status, usage = os.wait4(-1, 0)
        if pid not in running:
            continue
        name, log, t = running.pop(pid)
        rc = os.waitstatus_to_exitcode(status)
        seconds[name] = time.perf_counter() - t
        print(f"  {'ok  ' if rc == 0 else 'FAIL'} {seconds[name]:6.1f} s  {usage.ru_maxrss / 2 ** 20:4.1f} GB  "
              f"{name}", flush=True)
        if rc != 0:
            failed.append(name)
            with open(log) as f:
                print("".join("        " + line for line in f.readlines()[-15:]), flush=True)
    with open(last_path, "w") as f:
        json.dump({**last, **seconds}, f, indent=1, sort_keys=True)
    print(f"{len(files) - len(failed)}/{len(files)} files passed in {time.perf_counter() - t_all:.0f} s"
          + (f"; FAILED: {' '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
