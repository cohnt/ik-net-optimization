"""A real benchmark killed mid-grid and resumed reproduces the uninterrupted run, cell for cell.

There is no test suite in this repo; run this by hand, alone, under the memory scope:

    python tests/test_resume_e2e.py slow                  # both solvers
    python tests/test_resume_e2e.py [ipopt|svgd ...]      # one of them

Tiers (tests/_tiers.py): every check here is end to end, so the default `fast` tier runs
nothing and says so -- the same contract is held cheaply by tests/test_cell_resume.py (run_grid's
per-cell persistence, resume and kill marker, on stub arms) and tests/test_run_items_watcher.py
(the watcher's kill and relaunch). `slow` (~20 min, needs the GPU: `--compile` and the CUDA
graphs are the record's flags) is this file; on SuperCloud it is part of cluster/tests_slow_job.sh.

For each solver, a 2 x 2 Panda pose grid (`scripts/panda/panda_benchmark.py`, the record's flags
at a 60 s clock) is run three times, one process at a time:

  ref    uninterrupted, with no watcher environment -- the reference;
  crash  under LEARNED_IK_RESUME / LEARNED_IK_WATCH_DIR, SIGKILLed after two recorded cells with
         no kill marker (a process death), then relaunched: the in-flight cell is re-run;
  kill   the same, but killed inside a solve after an iterate has streamed, with the watcher's
         kill marker written first (exactly what cluster/run_items.sh does): the relaunch records
         that cell as `killed_overrun` from its last iterate.

On every cell that timed out in neither run the records must be identical apart from timing
keys -- the svgd block included under svgd -- and so must the summaries' non-timing columns.
The killed cell must be a timed-out `killed_overrun` failure carrying a verified `recovered_*`
point. Requires ikflow and the GPU; ~20 min for both solvers.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)

import src.benchmark as bm                                              # noqa: E402

FAILURES = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


COMMON = ["--task", "pose", "--targets", "2", "--guesses", "2", "--wall-time", "60",
          "--seed", "1", "--start", "paired", "--target-placement", "shelf",
          "--placement-point", "fingertips", "--config", "latent", "--scene", "hardened",
          "--set", "correction_cost_weight=10.0",
          "--checkpoint", "models/panda/panda__n6__step620000.pkl",
          "--compile", "--set", "flow_cuda_graph=True"]
SOLVER_ARGS = {
    "ipopt": ["--solver", "ipopt", "--arms", "learned,numerical", "--set", "max_iter=1000000"],
    "svgd": ["--solver", "svgd", "--arms", "learned", "--set", "svgd_compile=True",
             "--set", "svgd_cuda_graph=True"],
}


def is_timing(key):
    return "time" in key or "seconds" in key


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if not is_timing(k)}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def command(solver, tag):
    return [sys.executable, "-u", "scripts/panda/panda_benchmark.py", *COMMON,
            *SOLVER_ARGS[solver], "--tag", tag]


def out_dir(tag):
    return os.path.join(REPO, "results", "panda", "benchmark", tag)


def n_records(tag):
    try:
        with open(os.path.join(out_dir(tag), "summary.json.cells.jsonl")) as fh:
            return sum(1 for line in fh if '"record"' in line)
    except OSError:
        return 0


def heartbeat(wd):
    try:
        with open(os.path.join(wd, "heartbeat")) as fh:
            return fh.read().split()
    except OSError:
        return None


def run_to_end(cmd, env, log):
    with open(log, "a") as fh:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc


def run_and_kill(cmd, env, log, tag, wd, with_marker):
    """Launch, wait for two recorded cells and then for a solve in flight (with a streamed
    iterate, for the marker case), SIGKILL it -- writing the watcher's marker first when asked --
    and return the killed (arm, ti, gi) or None if the run finished first."""
    with open(log, "a") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    killed = None
    while proc.poll() is None:
        time.sleep(0.02)
        if n_records(tag) < 2:
            continue
        hb = heartbeat(wd)
        if not hb or hb[4] != "solve":
            continue
        arm, ti, gi = hb[3], int(hb[1]), int(hb[2])
        if with_marker:
            it = bm.IterateFile.read(os.path.join(wd, "iterate.bin"), arm, ti, gi)
            if it is None or it[0] is None:
                continue
            with open(os.path.join(wd, "killed.tmp"), "w") as fh:
                fh.write(" ".join(hb) + f" {time.time():.6f}\n")
            os.replace(os.path.join(wd, "killed.tmp"), os.path.join(wd, "killed"))
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        killed = (arm, ti, gi)
    return killed


def load(tag):
    with open(os.path.join(out_dir(tag), "summary.json")) as fh:
        return json.load(fh)


def by_cell(summary):
    return {(arm, r["target"], r["guess"]): r for arm, recs in summary["records"].items()
            for r in recs}


def compare(ref, got, skip=(), label=""):
    a, b = by_cell(ref), by_cell(got)
    check(set(a) == set(b), f"{label}: the same (cell, arm) set")
    same = differ = clocked = 0
    for key in sorted(a):
        if key in skip or key not in b:
            continue
        if a[key].get("timed_out") or b[key].get("timed_out"):
            clocked += 1
            continue
        if strip(a[key]) == strip(b[key]):
            same += 1
        else:
            differ += 1
            diff = sorted(k for k in set(a[key]) | set(b[key])
                          if strip(a[key].get(k)) != strip(b[key].get(k)))
            print(f"     {label} {key} differs in {diff}")
    check(differ == 0 and same > 0 and clocked == 0,
          f"{label}: {same} identical, {differ} differing, {clocked} at the clock")
    return differ == 0


def run_solvers(solvers):
    if not solvers:
        print("fast tier: nothing to run -- this file is end to end (`slow`, GPU). The resume "
              "contract's cheap guards are tests/test_cell_resume.py and tests/test_run_items_watcher.py.")
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = tempfile.mkdtemp(prefix="resume_e2e_")
    tags = []
    for solver in solvers:
        print(f"== {solver}", flush=True)
        log = os.path.join(work, f"{solver}.log")
        base_env = {k: v for k, v in os.environ.items()
                    if k not in (bm.RESUME_ENV, bm.WATCH_ENV, bm.OVERRUN_ENV)}
        ref_tag, crash_tag, kill_tag = (f"_test_resume_e2e_{stamp}_{solver}_{w}"
                                        for w in ("ref", "crash", "kill"))
        tags += [ref_tag, crash_tag, kill_tag]

        rc = run_to_end(command(solver, ref_tag), base_env, log)
        check(rc == 0, f"{solver}: the reference run finishes (rc {rc})")
        if rc != 0:
            continue
        ref = load(ref_tag)

        for tag, marker in ((crash_tag, False), (kill_tag, True)):
            wd = os.path.join(work, f"watch_{tag}")
            env = dict(base_env, **{bm.RESUME_ENV: "1", bm.WATCH_ENV: wd,
                                    bm.OVERRUN_ENV: "1.5"})
            killed = run_and_kill(command(solver, tag), env, log, tag, wd, marker)
            check(killed is not None, f"{solver}/{tag[-5:]}: killed mid-grid in {killed}")
            rc = run_to_end(command(solver, tag), env, log)
            check(rc == 0, f"{solver}/{tag[-5:]}: the relaunch resumes and finishes (rc {rc})")
            if rc != 0 or killed is None:
                continue
            got = load(tag)
            check([e["records_loaded"] for e in got["metadata"].get("resume", [])] != [],
                  f"{solver}/{tag[-5:]}: the relaunch resumed from the JSONL "
                  f"({got['metadata'].get('resume')})")
            if not marker:
                compare(ref, got, label=f"{solver}/crash")
                check(strip(got["summary"]) == strip(ref["summary"]),
                      f"{solver}/crash: the summary equals the uninterrupted run's")
                continue
            compare(ref, got, skip=(killed,), label=f"{solver}/kill")
            rec = by_cell(got)[killed]
            check(rec["fail_reason"] == "killed_overrun" and rec["feasible"] is False
                  and rec["timed_out"] is True and rec["hit_iteration_cap"] is False,
                  f"{solver}/kill: {killed} is a timed-out killed_overrun failure")
            check(rec.get("recovered_q") is not None
                  and rec.get("recovered_fail_reason", "?") != "?"
                  and rec.get("recovered_max_violation") is not None,
                  f"{solver}/kill: it is verified from its last iterate "
                  f"(recovered_feasible={rec.get('recovered_feasible')}, "
                  f"max_violation={rec.get('recovered_max_violation')})")
            check((rec.get("eval_counts") or {}).get("callback", 0) > 0
                  and rec.get("svgd") is None and rec.get("start_q_error") is not None,
                  f"{solver}/kill: counts from the stream, start diagnostics from the marker")
            check(not os.path.exists(os.path.join(wd, "killed")), f"{solver}/kill: marker consumed")
        if solver == "svgd":
            svgd = [r.get("svgd") for r in ref["records"]["learned"]]
            check(all(isinstance(s, dict) and s.get("n_particles") for s in svgd),
                  "svgd: the reference records carry the svgd block (compared above)")

    if not FAILURES:
        for tag in tags:
            shutil.rmtree(out_dir(tag), ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"logs kept in {work}; results under results/panda/benchmark/_test_resume_e2e_*")
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


GROUPS = {"fast": [], "slow": ["ipopt", "svgd"], "ipopt": ["ipopt"], "svgd": ["svgd"]}


def main():
    ## The groups name solvers, not checks: the solvers share one tag stamp and one cleanup, so
    ## they run in one `run_solvers` rather than through `_tiers.run`.
    names = sys.argv[1:] or ["fast"]
    unknown = [n for n in names if n not in GROUPS and n != "all"]
    if unknown:
        sys.exit(f"unknown group(s) {unknown}; this file has {sorted(GROUPS)} and 'all'")
    solvers = [s for n in names for s in GROUPS["slow" if n == "all" else n]]
    return run_solvers(list(dict.fromkeys(solvers)))


if __name__ == "__main__":
    sys.exit(main())
