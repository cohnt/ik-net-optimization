"""The overrun watcher in `cluster/run_items.sh`, driven end to end with a fake benchmark.

There is no test suite in this repo; run this by hand:

    python tests/test_run_items_watcher.py

`run_items.sh` gives every item a node-local watch directory and SIGKILLs the item's process
once its heartbeat (`pid ti gi arm phase deadline`, written by src/benchmark.py's `CellWatch`)
is in `solve` past the deadline, writing the `killed` marker first, then relaunches the item --
at most MAX_OVERRUNS times. The fake script plays run_grid's part: it consumes a marker it finds
and either overruns again or exits. Nothing here touches Drake, torch or the GPU; the script runs
with the real cluster script against a throwaway LEARNED_IK_ROOT, repo, state and TMPDIR.

Checked: an item that overruns twice is killed twice, relaunched twice and finishes `.done`, each
relaunch seeing the marker of the previous kill; an item that always overruns stops after
MAX_OVERRUNS relaunches with no `.done`; a heartbeat past its deadline but in `post` (the solve
returned) is never killed; an item that writes no heartbeat runs as before; every kill is in
kills.log.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


FAKE = textwrap.dedent('''
    import os, sys, time
    mode, kills = sys.argv[1], int(sys.argv[2])
    wd = os.environ["LEARNED_IK_WATCH_DIR"]
    assert os.environ.get("LEARNED_IK_RESUME") == "1"
    factor = os.environ["LEARNED_IK_OVERRUN_FACTOR"]
    count = os.path.join(wd, "launches")
    n = int(open(count).read()) + 1 if os.path.exists(count) else 1
    open(count, "w").write(str(n))
    marker = os.path.join(wd, "killed")
    with open(os.path.join(wd, "seen"), "a") as fh:
        fh.write(f"launch {n} factor {factor} marker "
                 f"{open(marker).read().split()[1:4] if os.path.exists(marker) else None}\\n")
    if os.path.exists(marker):
        os.remove(marker)                     # run_grid consumes it when it records the cell
    def beat(phase, deadline):
        with open(os.path.join(wd, "heartbeat.tmp"), "w") as fh:
            fh.write(f"{os.getpid()} {n} 3 learned {phase} {deadline}\\n")
        os.replace(os.path.join(wd, "heartbeat.tmp"), os.path.join(wd, "heartbeat"))
    if mode == "quiet":
        time.sleep(3)
        sys.exit(0)
    if mode == "post":
        beat("post", int(time.time()) - 5)
        time.sleep(4)
        sys.exit(0)
    if mode == "always" or n <= kills:
        beat("setup", 0)
        time.sleep(1)
        beat("solve", int(time.time()) - 1)   # already past its deadline
        time.sleep(60)
        sys.exit(3)                           # never reached: the watcher kills first
    sys.exit(0)
''')


def main():
    root = tempfile.mkdtemp(prefix="watcher_")
    repo, state, tmp = (os.path.join(root, d) for d in ("repo", "state", "tmp"))
    for d in (repo, state, tmp, os.path.join(root, "home", ".cache")):
        os.makedirs(d)
    with open(os.path.join(repo, "fake_bench.py"), "w") as fh:
        fh.write(FAKE)
    manifest = os.path.join(root, "manifest_WATCHTEST.txt")
    items = {"twice": "overrun 2", "always": "always 0", "post": "post 0", "quiet": "quiet 0"}
    with open(manifest, "w") as fh:
        for item, args in items.items():
            fh.write(f"{item}|-|fake_bench.py|{args}\n")
    env = dict(os.environ, MANIFEST=manifest, PROCS="1", LEARNED_IK_ROOT=root,
               TEST_REPO=repo, TEST_STATE_DIR=state, TEST_PYTHON=sys.executable,
               LEARNED_IK_RESULTS=os.path.join(root, "results"), TMPDIR=tmp,
               WATCH_POLL="1", MAX_OVERRUNS="2", ITEM_TIMEOUT="600", DEVICE="cpu")
    env.pop("SLURM_JOB_ID", None)
    proc = subprocess.run(["bash", os.path.join(REPO, "cluster", "run_items.sh")], env=env,
                          capture_output=True, text=True, timeout=600)
    sdir = os.path.join(state, "manifest_WATCHTEST")
    log = open(os.path.join(tmp, "worker_0.log")).read()

    def seen(item):
        path = os.path.join(tmp, f"watch.0.{item}", "seen")
        return open(path).read().splitlines() if os.path.exists(path) else None

    check(os.path.exists(os.path.join(sdir, "twice.done")), "the twice-overrunning item finishes")
    check(seen("twice") is None, "a finished item's watch directory is removed")
    check(log.count("twice KILLED") == 2 and log.count("relaunch 1 after") >= 1
          and log.count("relaunch 2 after") >= 1, "it was killed and relaunched twice")
    always = seen("always")
    check(not os.path.exists(os.path.join(sdir, "always.done")) and always is not None
          and len(always) == 3 and "giving up: more than MAX_OVERRUNS=2" in log,
          f"an item that always overruns stops after MAX_OVERRUNS relaunches: {always}")
    check(always is not None and always[0].endswith("marker None")
          and all("marker ['2', '3', 'learned']" in line or "marker ['1', '3', 'learned']" in line
                  for line in always[1:]),
          "each relaunch sees the kill marker of the cell killed before it")
    check(all("factor 1.5" in line for line in always or []),
          "the overrun factor reaches the process (default 1.5)")
    check(os.path.exists(os.path.join(sdir, "post.done")) and "post KILLED" not in log,
          "a heartbeat past its deadline in `post` is not killed")
    check(os.path.exists(os.path.join(sdir, "quiet.done")) and "quiet KILLED" not in log,
          "an item without a heartbeat runs as before")
    kills = [l for l in open(os.path.join(sdir, "kills.log")).read().splitlines() if " KILLED " in l]
    check(len(kills) == 2 + 3, f"every kill is in kills.log ({len(kills)} lines)")
    check(proc.returncode != 0, "the job reports the failed item")
    if FAILURES:
        print(proc.stdout[-2000:], proc.stderr[-2000:], log[-4000:], sep="\n----\n")
    shutil.rmtree(root, ignore_errors=True)
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
