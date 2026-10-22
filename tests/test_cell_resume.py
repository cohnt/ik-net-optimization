"""Per-cell persistence and resume in `run_grid`, on stub arms (no Drake solve).

There is no test suite in this repo; run this by hand:

    python tests/test_cell_resume.py

`run_grid` appends every finished (cell, arm) to `<out_path>.cells.jsonl`, and under
`LEARNED_IK_RESUME=1` resumes from it (src/benchmark.py, the persistence section). This drives
`run_grid` with stub programs whose verdicts are a deterministic function of the cell, with
`verify`, `start_diagnostics` and `reported_cost` replaced, and checks that:

  - a run interrupted mid-grid -- including between the two arms of one cell -- and resumed
    produces the same records and summary as an uninterrupted run;
  - a resume against a JSONL of a different run refuses, and a torn final line is cut off;
  - a cell killed in its solve (a watcher kill marker, `LEARNED_IK_WATCH_DIR`) is recorded as
    `killed_overrun` from exactly the iterate streamed to `iterate.bin`, with `timed_out` set;
  - a kill marker for a cell that already has a record is discarded;
  - the heartbeat carries the solve's deadline at the overrun factor;
  - `IterateFile` rejects a corrupt buffer and another cell's iterate;
  - the iterate's count layout is `IKFlowProgram.EVAL_COUNT_KEYS`.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)

import src.benchmark as bm                                              # noqa: E402

FAILURES = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


class Crash(BaseException):
    """Not an Exception, so run_grid's per-cell handler cannot catch it: a process death."""


class StubResult:
    def __init__(self, ok):
        self.ok = ok

    def is_success(self):
        return self.ok

    def get_solution_result(self):
        return "kSolutionFound" if self.ok else "kIterationLimit"

    def get_solver_details(self):
        raise RuntimeError("no details")


class StubProgram:
    def __init__(self, arm, cell, plan):
        self.arm, self.cell, self.plan = arm, cell, plan
        self.options = SimpleNamespace(which_solver="ipopt", max_wall_time=7.0,
                                       file_print_name=os.devnull, correction_bound=0.1)
        self.eval_counts = {}
        self.last_iterate = None
        self.iterate_sink = None

    def Solve(self):
        ti, gi = self.cell
        x = np.array([ti, gi, 0.5 * ti + gi, np.float32(0.1) * (gi + 1)], dtype=np.float64)
        self.eval_counts = {"map_forward": 0, "map_jacobian": 3 * ti + gi, "memo_hits": 2,
                            "callback": 4}
        self.last_iterate = x
        if self.iterate_sink is not None:
            self.iterate_sink(x, self.eval_counts)
        hook = self.plan.get(("solve", self.arm, ti, gi))
        if hook is not None:
            hook(self)
        return StubResult((ti + gi + len(self.arm)) % 3 != 0)


def stub_verify(program, result, task_gate, tol, x_lumped=None, relaxed_tol=None):
    x = program.last_iterate if x_lumped is None else np.asarray(x_lumped)
    ti, gi = program.cell
    ok = (ti + gi + len(program.arm)) % 3 != 0 if x_lumped is None else bool(x[0] >= 1)
    detail = dict(q=np.asarray(x) * 2, max_violation=np.float64(1e-3 * (ti + 1)),
                  z_norm=np.float32(0.25 * gi), correction_inf=0.01, flow_neg_log_density=None)
    return SimpleNamespace(feasible=ok, fail_reason=None if ok else "constraint",
                           feasible_relaxed=ok, fail_reason_relaxed=None if ok else "constraint",
                           detail=detail)


bm.verify = stub_verify
bm.start_diagnostics = lambda program, q: {"start_q_error": np.float64(1e-7),
                                           "clip_distance": 0.0, "start_z_norm": 1.5}
bm.reported_cost = lambda program, result, weight: np.float64(program.cell[0] + 0.1) / weight

N_TARGETS, N_GUESSES = 3, 2
TARGETS = [None] * N_TARGETS
GUESSES = [[None] * N_GUESSES for _ in range(N_TARGETS)]


def make_arms(plan):
    def maker(name):
        def make(target, q_init, cell):
            hook = plan.get(("build", name) + tuple(cell))
            if hook is not None:
                hook()
            return StubProgram(name, tuple(cell), plan)
        return make
    return [bm.Arm("learned", maker("learned"), 1e-4), bm.Arm("numerical", maker("numerical"), 1.0)]


def run(out_dir, plan=None, metadata=None, env=None):
    env = env or {}
    saved = {k: os.environ.get(k) for k in (bm.RESUME_ENV, bm.WATCH_ENV, bm.OVERRUN_ENV)}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        return bm.run_grid(make_arms(plan or {}), TARGETS, GUESSES, task_gate=None,
                           log_dir=out_dir, out_path=os.path.join(out_dir, "summary.json"),
                           tol=1e-4, relaxed_tol=1e-3,
                           metadata=dict(metadata or {"grid_hash": "abc", "solver": "ipopt"}))
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def load(out_dir):
    with open(os.path.join(out_dir, "summary.json")) as fh:
        return json.load(fh)


TIMING = ("wall_time", "setup_time")


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items()
                if k not in TIMING and not k.startswith("mean_wall") and k != "mean_setup_time"}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def crash(*_):
    raise Crash()


def main():
    root = tempfile.mkdtemp(prefix="cell_resume_")

    ## The reference: one uninterrupted process.
    ref_dir = os.path.join(root, "ref")
    run(ref_dir)
    ref = load(ref_dir)
    n_lines = sum(1 for _ in open(os.path.join(ref_dir, "summary.json.cells.jsonl")))
    check(n_lines >= 1 + 2 * N_TARGETS * N_GUESSES,
          f"the JSONL holds a header and one line per (cell, arm) ({n_lines} lines)")

    ## Interrupted twice -- once before cell (1,0)'s learned arm, once between the two arms of
    ## cell (2,1) -- and resumed each time.
    dir_ = os.path.join(root, "resumed")
    for plan in ({("build", "learned", 1, 0): crash}, {("build", "numerical", 2, 1): crash}):
        try:
            run(dir_, plan, env={bm.RESUME_ENV: "1"})
            check(False, "the crash propagated")
        except Crash:
            pass
    keys = [json.loads(line) for line in open(os.path.join(dir_, "summary.json.cells.jsonl"))]
    check(any(k.get("arm") == "learned" and k["record"]["target"] == 2
              and k["record"]["guess"] == 1 for k in keys)
          and not any(k.get("arm") == "numerical" and k["record"]["target"] == 2
                      and k["record"]["guess"] == 1 for k in keys),
          "the second crash left a cell with its learned arm recorded and joint space not")
    run(dir_, env={bm.RESUME_ENV: "1"})
    got = load(dir_)
    check(strip(got["records"]) == strip(ref["records"]),
          "resumed records equal the uninterrupted run's, cell for cell")
    check(strip(got["summary"]) == strip(ref["summary"]),
          "resumed summary equals the uninterrupted run's")
    check([e["records_loaded"] for e in got["metadata"].get("resume", [])] == [4, 11],
          f"metadata['resume'] logs each relaunch: {got['metadata'].get('resume')}")

    ## Without LEARNED_IK_RESUME the JSONL is started afresh.
    run(dir_)
    n = sum(1 for _ in open(os.path.join(dir_, "summary.json.cells.jsonl")))
    check("resume" not in load(dir_)["metadata"] and n == n_lines,
          "without the resume variable the run starts afresh")

    ## A torn final line (a kill mid-write) is cut off, and the run resumes.
    with open(os.path.join(dir_, "summary.json.cells.jsonl"), "a") as fh:
        fh.write('{"arm": "learned", "rec')
    run(dir_, env={bm.RESUME_ENV: "1"})
    check(strip(load(dir_)["records"]) == strip(ref["records"]), "a torn final line is dropped")

    ## A different run refuses to resume.
    try:
        run(dir_, metadata={"grid_hash": "OTHER", "solver": "ipopt"}, env={bm.RESUME_ENV: "1"})
        check(False, "a header mismatch refuses")
    except SystemExit as exc:
        check("grid_hash" in str(exc), f"a header mismatch refuses: {exc}")

    ## The watcher path: the process dies inside cell (1,1)'s learned solve after streaming an
    ## iterate; the watcher's kill marker is the heartbeat line plus the kill time.
    kill_dir, watch_dir = os.path.join(root, "killed"), os.path.join(root, "watch")
    env = {bm.RESUME_ENV: "1", bm.WATCH_ENV: watch_dir, bm.OVERRUN_ENV: "1.5"}
    seen = {}

    def die_in_solve(program):
        with open(os.path.join(watch_dir, "heartbeat")) as fh:
            seen["heartbeat"] = fh.read().split()
        seen["t"] = time.time()
        with open(os.path.join(watch_dir, "killed"), "w") as fh:
            fh.write(" ".join(seen["heartbeat"]) + f" {time.time() + 2.0:.3f}\n")
        raise Crash()

    try:
        run(kill_dir, {("solve", "learned", 1, 1): die_in_solve}, env=env)
    except Crash:
        pass
    hb = seen.get("heartbeat", [])
    check(hb[1:5] == ["1", "1", "learned", "solve"]
          and abs(int(hb[5]) - (seen["t"] + 1.5 * 7.0)) < 2.0,
          f"the heartbeat names the solve and its deadline at 1.5 x the clock: {hb}")
    run(kill_dir, env=env)
    out = load(kill_dir)
    rec = next(r for r in out["records"]["learned"] if (r["target"], r["guess"]) == (1, 1))
    check(rec["fail_reason"] == "killed_overrun" and rec["feasible"] is False
          and rec["timed_out"] is True and rec["hit_iteration_cap"] is False,
          "the killed cell is a killed_overrun failure, timed out, not iteration-capped")
    check(rec["recovered_q"] == [2.0, 2.0, 3.0, 2 * float(np.float32(0.1)) * 2]
          and rec["recovered_feasible"] is True,
          f"it is verified from the streamed iterate: recovered_q={rec.get('recovered_q')}")
    check(rec["eval_counts"] == {"map_forward": 0, "map_jacobian": 4, "memo_hits": 2,
                                 "callback": 4}, f"its eval counts come from the stream")
    check(1.9 < rec["wall_time"] < 3.0, f"its wall clock runs to the kill: {rec['wall_time']:.2f}")
    check(rec["start_q_error"] == 1e-7 and "setup_time" in rec,
          "it carries the start marker's partial record")
    check(not os.path.exists(os.path.join(watch_dir, "killed")), "the kill marker is consumed")
    others = [r for arm in ("learned", "numerical") for r in out["records"][arm]
              if (arm, r["target"], r["guess"]) != ("learned", 1, 1)]
    ref_others = [r for arm in ("learned", "numerical") for r in ref["records"][arm]
                  if (arm, r["target"], r["guess"]) != ("learned", 1, 1)]
    check(strip(others) == strip(ref_others), "every other cell equals the uninterrupted run's")
    check(out["summary"]["learned"]["fail_reasons"].get("killed_overrun") == 1
          and out["summary"]["learned"]["timeouts"] == 1, "the summary counts it")

    ## A kill marker for a cell that already has a record is discarded.
    with open(os.path.join(watch_dir, "killed"), "w") as fh:
        fh.write(f"1 0 1 learned solve 0 {time.time():.3f}\n")
    run(kill_dir, env=env)
    check(strip(load(kill_dir)["records"]) == strip(out["records"])
          and not os.path.exists(os.path.join(watch_dir, "killed")),
          "a stale kill marker for a recorded cell is discarded")

    ## IterateFile.
    path = os.path.join(root, "iterate.bin")
    f = bm.IterateFile(path)
    f.sink("learned", 3, 4)(np.array([1.0, 2.0]), {"callback": 9})
    v, c = bm.IterateFile.read(path, "learned", 3, 4)
    check(list(v) == [1.0, 2.0] and c["callback"] == 9, "an iterate reads back")
    check(bm.IterateFile.read(path, "numerical", 3, 4) is None
          and bm.IterateFile.read(path, "learned", 3, 5) is None,
          "another cell's iterate is not attributed")
    f.reset("learned", 3, 4)
    check(bm.IterateFile.read(path, "learned", 3, 4)[0] is None, "a reset leaves no iterate")
    f.sink("learned", 3, 4)(np.array([1.0, 2.0]), {})
    with open(path, "r+b") as fh:
        fh.seek(bm.IterateFile._HEAD.size + 3)
        fh.write(b"\xff")
    check(bm.IterateFile.read(path, "learned", 3, 4) is None, "a corrupt buffer is rejected")
    f.close()

    from src.generic_program import IKFlowProgram
    check(tuple(bm._ITERATE_COUNT_KEYS) == tuple(IKFlowProgram.EVAL_COUNT_KEYS),
          "the iterate's count layout is IKFlowProgram.EVAL_COUNT_KEYS")

    shutil.rmtree(root, ignore_errors=True)
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
