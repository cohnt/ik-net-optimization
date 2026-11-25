"""GPU plumbing check, run INSIDE the Engaging image on a GPU node (cluster/engaging/plumb_job.sh).

Passes only if: torch sees a CUDA device and runs one kernel on it (so the cu126 wheel carries the card's
arch); pydrake imports FROM THE IMAGE's pin, VERSION.TXT names the pinned release and commit (argv[1],
argv[2], from lik.def's DRAKE_* variables), CollisionChecker has CalcRobotClearances (AL-SVGD's
svgd_collision_backend=clearances); IPOPT is available; ikflow resolves to the staged
repo's vendored fork. With --warm it also caches the Panda weights and builds every scene once (Drake's
drake_models cache is fetched on first use), under the HOME it is run with -- compute nodes have
internet; every later job runs from the warm cache.
"""
import os
import subprocess
import sys

expect_version, expect_commit = sys.argv[1], sys.argv[2]
warm = "--warm" in sys.argv
print("HOME", os.environ.get("HOME"), "expanduser", os.path.expanduser("~"))
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip())

import torch  # noqa: E402

assert torch.cuda.is_available(), "torch.cuda is not available"
dev = torch.device("cuda:0")
cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0), "capability", cap,
      "arch list", torch.cuda.get_arch_list())
a = torch.randn(512, 512, device=dev, dtype=torch.float64)
b = (a @ a).sum().item()           # one real kernel launch, float64 as the project runs
torch.cuda.synchronize()
assert b == b, "kernel returned NaN"
print("kernel ok:", b)

import pydrake  # noqa: E402
from pydrake.solvers import IpoptSolver, SnoptSolver  # noqa: E402

assert pydrake.__file__.startswith("/opt/lik/drake/"), f"pydrake is not the image's pin: {pydrake.__file__}"
version = open("/opt/lik/drake/share/doc/drake/VERSION.TXT").read().split()
print("pydrake", pydrake.__file__, "VERSION.TXT", version)
assert version[:2] == [expect_version, expect_commit], f"VERSION.TXT {version} != pin {expect_version} {expect_commit}"
from pydrake.planning import CollisionChecker  # noqa: E402
assert hasattr(CollisionChecker, "CalcRobotClearances"), "this Drake lacks CollisionChecker.CalcRobotClearances"
print("CollisionChecker.CalcRobotClearances present")
assert IpoptSolver().available(), "IPOPT unavailable"
print("IPOPT", IpoptSolver().available(), "SNOPT", SnoptSolver().available())

import ikflow  # noqa: E402
import jrl  # noqa: E402

repo = os.environ["LIK_REPO"]
assert os.path.realpath(ikflow.__file__).startswith(os.path.realpath(repo) + "/third_party/ikflow"), ikflow.__file__
print("ikflow", ikflow.__file__, "jrl", jrl.__file__)

if warm:
    from ikflow.model_loading import get_ik_solver
    get_ik_solver("panda__full__lp191_5.25m")
    print("panda weights cached")
    sys.path.insert(0, repo)
    from src.utils import BuildEnv, RepoDir
    for scene in ("models/panda/panda_finray_collision.yaml", "models/iiwa14/iiwa14_collision.yaml",
                  "models/panda/panda_finray_collision_hardened.yaml", "models/iiwa14/iiwa14_collision_nobin.yaml",
                  "models/iiwa14/iiwa14_collision_hardened.yaml"):
        BuildEnv(meshcat=None, directives_file=os.path.join(RepoDir(), scene))
        print("scene builds:", scene)
print("PLUMB OK")
