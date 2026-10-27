"""Every test file's `fast` tier (tests/_tiers.py), one file at a time: `python tests/run_fast.py`.

Each file runs in its own fresh process, in series -- no pool, so no two files hold memory at
once, and nothing one file changes (a program's options, a patched function) can reach the next.

Local rules, applied by default:
- re-executes itself under `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`
  when systemd-run exists (`--no-scope` to skip); the scope covers every file's process;
- CPU only (`CUDA_VISIBLE_DEVICES=` unless already set);
- `--threads` (default 4, Thomas 2026-10-27) threads on as many cores. The GVS arm's XLA
  threads stay at 1 whatever `--threads` is: the thread count is part of XLA's compile options,
  so changing it would miss the JAX compilation cache (`_tiers.jax_compilation_cache`).

Each file's output goes to `<log dir>/<file>.log`; one line per file here, and a failing file's
log tail. Exit status 1 if any file failed.

    python tests/run_fast.py                     # all 36 files
    python tests/run_fast.py svgd gvs            # files whose name contains any of these
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("only", nargs="*", help="run files whose name contains any of these")
    ap.add_argument("--threads", type=int, default=4)
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


def environment(threads):
    """CPU only, `threads` threads on as many cores (the affinity is inherited by every file)."""
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS", "DRAKE_NUM_THREADS"):
        env[var] = str(threads)
    env["GVS_ARM_XLA_THREADS"] = "1"
    env.setdefault("TQDM_DISABLE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    cores = sorted(os.sched_getaffinity(0))
    if len(cores) > threads:
        os.sched_setaffinity(0, cores[:threads])
    return env


def main():
    args = parse()
    scoped(args)
    env = environment(args.threads)
    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    if args.only:
        files = [f for f in files if any(o in os.path.basename(f) for o in args.only)]
    log_dir = args.log_dir or os.path.join(REPO, "results", "_tests_fast", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"{len(files)} files, fast tier, one at a time, {args.threads} threads, "
          f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r}; logs in {log_dir}", flush=True)

    failed, t_all = [], time.perf_counter()
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        log = os.path.join(log_dir, name + ".log")
        t = time.perf_counter()
        with open(log, "w") as out:
            rc = subprocess.run([sys.executable, path], cwd=REPO, env=env, stdout=out,
                                stderr=subprocess.STDOUT).returncode
        print(f"  {'ok  ' if rc == 0 else 'FAIL'} {time.perf_counter() - t:6.1f} s  {name}", flush=True)
        if rc != 0:
            failed.append(name)
            with open(log) as f:
                print("".join("        " + line for line in f.readlines()[-15:]), flush=True)
    print(f"{len(files) - len(failed)}/{len(files)} files passed in {time.perf_counter() - t_all:.0f} s"
          + (f"; FAILED: {' '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
