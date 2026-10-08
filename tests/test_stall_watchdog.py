"""Assert that the per-cell stall dump cannot kill the process it is watching.

There is no test suite in this repo; run this by hand:

    python tests/test_stall_watchdog.py

`run_grid` dumps every thread's stack when a cell overruns `cell_timeout`. It used
`faulthandler.dump_traceback_later`, whose C watchdog thread walks the frame stacks WITHOUT
the GIL and segfaulted ~7% of the processes it fired on (stage SEGVREP, 2026-10-07; see
`StallWatchdog`). The laptop reproduction is a main thread calling a `torch.compile`d
`jacrev` under a 1 ms repeating dump: faulthandler crashes it in every run, within seconds.

So this runs that same workload in a subprocess under `StallWatchdog` firing every 1 ms and
requires it to exit cleanly with every dump complete, checks arm/repeat/cancel, and guards
against `dump_traceback_later` coming back anywhere in the tree.
"""
import io
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)

from src.benchmark import StallWatchdog                                 # noqa: E402

FAILURES = []


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILURES.append(what)


STRESS = r'''
import sys, time
sys.path.insert(0, sys.argv[1])
import torch
from torch.func import jacrev
from src.benchmark import StallWatchdog
torch.set_default_device("cpu")       # jrl.config does this at import: a TorchFunctionMode
W = torch.randn(21, 21, dtype=torch.float64) * 0.1
def g(v):
    h = v
    for _ in range(6):
        h = torch.tanh(h @ W) + h
    return h[:7]
J = torch.compile(jacrev(g))
x = torch.randn(21, dtype=torch.float64)
J(x)
out = open(sys.argv[2], "w")
wd = StallWatchdog(out)
wd.arm(0.001)
def nest(v, d):
    return J(v).sum() if d == 0 else nest(v, d - 1)
t0 = time.time()
while time.time() - t0 < float(sys.argv[3]):
    nest(x, 5)
wd.cancel()
out.close()
print("survived")
'''


def test_survives_compiled_workload(tmpdir, seconds=10.0):
    dump = os.path.join(tmpdir, "stalls.txt")
    proc = subprocess.run([sys.executable, "-c", STRESS, REPO, dump, str(seconds)],
                          capture_output=True, text=True, timeout=600)
    check(proc.returncode == 0 and "survived" in proc.stdout,
          f"compiled-jacrev workload under a 1 ms dump exits cleanly (rc {proc.returncode})")
    text = open(dump).read()
    dumps = text.count("Timeout (")
    check(dumps > 1000, f"the watchdog actually fired ({dumps} dumps)")
    ## Every main-thread section must reach the script's own outermost frame: a dump cut
    ## off mid-stack, or a frame read as `???`, is exactly what faulthandler produced.
    main = re.findall(r"\[MainThread\] \(most recent call first\):\n((?:  File .*\n)+)", text)
    complete = sum(1 for m in main if m.rstrip().endswith("in <module>"))
    check(len(main) == dumps and complete == dumps,
          f"every dump is complete ({complete} of {dumps} reach <module>)")
    check("???" not in text, "no garbage frames")


def test_arm_repeat_cancel():
    buf = io.StringIO()
    wd = StallWatchdog(buf)
    time.sleep(0.2)
    check(buf.getvalue() == "", "an unarmed watchdog writes nothing")
    wd.arm(0.05)
    time.sleep(0.28)
    n = buf.getvalue().count("Timeout (")
    check(4 <= n <= 6, f"armed at 50 ms it repeats while the cell runs ({n} dumps in 280 ms)")
    wd.cancel()
    time.sleep(0.2)
    check(buf.getvalue().count("Timeout (") == n, "cancel stops it")
    wd.arm(10.0)
    time.sleep(0.1)
    check(buf.getvalue().count("Timeout (") == n, "re-arming restarts the clock")
    wd.cancel()


def test_no_faulthandler_watchdog_in_tree():
    hits = []
    for top in ("src", "scripts", "cluster", "tests"):
        for root, _, files in os.walk(os.path.join(REPO, top)):
            for f in files:
                path = os.path.join(root, f)
                if not f.endswith((".py", ".sh")) or path == os.path.abspath(__file__):
                    continue
                for i, line in enumerate(open(path, errors="replace"), 1):
                    if re.search(r"dump_traceback_later\s*\(", line):
                        hits.append(f"{os.path.relpath(path, REPO)}:{i}")
    check(not hits, "no faulthandler.dump_traceback_later call in the tree" +
          (f": {hits}" if hits else ""))


if __name__ == "__main__":
    import tempfile
    test_arm_repeat_cancel()
    test_no_faulthandler_watchdog_in_tree()
    with tempfile.TemporaryDirectory() as tmp:
        test_survives_compiled_workload(tmp)
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1 if FAILURES else 0)
