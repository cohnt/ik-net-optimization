"""Run a vendored-ikflow script with this project's robots registered first.

ikflow resolves its robot through `jrl.robots.get_robot`, which scans a registry the soft
arm is not in until `src.soft_arm.register` is imported. The fork's own scripts do not
import it -- and should not have to, since it is our robot and not theirs. This wrapper
imports the registration and then runs the fork's script unchanged, so the fork stays
generic and there is no third-party edit to carry.

IMPORT ORDER IS LOAD-BEARING, for the same reason `cluster/train_flow.sh` reassigns HOME
before anything imports ikflow: ikflow resolves DATASET_DIR from `expanduser("~")` AT
IMPORT. A path bug of exactly that shape once cost a rung its entire 620k-step run, so the
registration happens before the delegated script is even located.

    python scripts/training/ikflow_entry.py build_dataset --robot_name=soft12 ...
    python scripts/training/ikflow_entry.py train_ddp --robot_name=soft12 ...
"""

import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

import src.soft_arm.register  # noqa: E402,F401  -- before anything resolves a robot

FORK_SCRIPTS = os.path.join(REPO, "third_party", "ikflow", "scripts")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {os.path.basename(__file__)} <script> [args...]\n"
                         f"  scripts live in {FORK_SCRIPTS}")
    name = sys.argv[1]
    path = os.path.join(FORK_SCRIPTS, name if name.endswith(".py") else f"{name}.py")
    if not os.path.exists(path):
        available = sorted(f[:-3] for f in os.listdir(FORK_SCRIPTS) if f.endswith(".py"))
        raise SystemExit(f"no such ikflow script {name!r}; available: {available}")
    print(f"[ikflow_entry] registered {src.soft_arm.register.REGISTERED}, running {path}")
    sys.argv = [path] + sys.argv[2:]
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
