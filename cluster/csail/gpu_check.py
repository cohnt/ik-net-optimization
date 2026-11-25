"""GPU plumbing check, run INSIDE the image (cluster/csail/check_job.sh, and the start of every job.sh).

Exits non-zero on any failure (the csail-cluster skill's trap 5: a broken GPU makes the job FAILED and writes no
success sentinel). Checks: torch sees a CUDA device, one real kernel launches and returns the right answer, the
device name matches EXPECT_GPU (a regex, optional), and -- with --drake -- pydrake imports from the Drake pin
(/opt/lik/DRAKE_PIN = the .def's DRAKE_VERSION and DRAKE_COMMIT: release 1.58.0) from the tarball, with IPOPT
available and CollisionChecker.CalcRobotClearances present (AL-SVGD's svgd_collision_backend=clearances).

    python gpu_check.py [--drake]
"""
import os
import re
import sys


def main():
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda is not available")
        return 3
    name = torch.cuda.get_device_name(0)
    print("device:", name, "capability", torch.cuda.get_device_capability(0), "count", torch.cuda.device_count())
    want = os.environ.get("EXPECT_GPU", "")
    if want and not re.search(want, name, re.I):
        print(f"FAIL: device '{name}' does not match EXPECT_GPU '{want}' (a campaign never mixes GPU types)")
        return 4
    x = torch.arange(1024, device="cuda", dtype=torch.float64)
    y = (x * x).sum().item()           # a real kernel launch, read back
    if y != float(sum(i * i for i in range(1024))):
        print("FAIL: kernel result is wrong:", y)
        return 5
    print("kernel launch ok")
    if "--drake" in sys.argv:
        import pydrake
        from pydrake.solvers import IpoptSolver
        from pydrake.planning import CollisionChecker
        pin = open("/opt/lik/DRAKE_PIN").read().split()        # "<version> <commit>", written at build from the .def
        got = open("/opt/lik/drake/share/doc/drake/VERSION.TXT").read().split()
        print("pydrake from", pydrake.__file__, "| VERSION.TXT", got, "| pin", pin)
        if not pydrake.__file__.startswith("/opt/lik/drake/"):
            print("FAIL: pydrake does not come from the pinned tarball")
            return 6
        if got != pin:
            print(f"FAIL: Drake is {got}, the pin is {pin}")
            return 6
        if not IpoptSolver().available():
            print("FAIL: IPOPT unavailable")
            return 7
        if not hasattr(CollisionChecker, "CalcRobotClearances"):
            print("FAIL: no CollisionChecker.CalcRobotClearances (svgd_collision_backend=clearances needs it)")
            return 8
        print("drake pin ok:", got, "| CalcRobotClearances present")
    print("GPU CHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
