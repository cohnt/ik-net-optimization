"""Run a vendored-ikflow script with this project's robots registered first.

ikflow resolves its robot through `jrl.robots.get_robot`, which scans a registry a robot
this project defines is not in until its register module is imported. `src.register_robots`
imports every one of them -- the soft PCS arm's and the screw-joint arm's -- so a new robot
adds one entry there and touches nothing here. The fork's own scripts do not import it --
and should not have to, since these are our robots and not theirs. This wrapper imports the
registration and then runs the fork's script unchanged, so the fork stays generic and there
is no third-party edit to carry.

IMPORT ORDER IS LOAD-BEARING, for the same reason `cluster/train_flow.sh` reassigns HOME
before anything imports ikflow: ikflow resolves DATASET_DIR from `expanduser("~")` AT
IMPORT. A path bug of exactly that shape once cost a rung its entire 620k-step run, so the
registration happens before the delegated script is even located.

    python scripts/training/ikflow_entry.py build_dataset --robot_name=soft12 ...
    python scripts/training/ikflow_entry.py train_ddp     --robot_name=helix7_p050 ...
"""

import json
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO)

import src.register_robots  # noqa: E402,F401  -- before anything resolves a robot

FORK_SCRIPTS = os.path.join(REPO, "third_party", "ikflow", "scripts")


def _with_pole_domain(script, argv):
    """Supply the pole screen's domain for a training run, from THIS project's convention.

    The fork's screen is generic and its constants are iiwa-shaped: a box around
    [0.4, 0, 0.5], a radius-4.3 latent ball, and "runaway" at 1000 RADIANS. On a robot
    whose coordinates are normalized strain none of the three means anything, and worse,
    the sampler draws position and orientation independently -- so for an arm without
    torsion, whose tip orientation is not free given tip position, nearly every sample is
    unreachable. Measured on `soft12__n6__step620000`, that reads `pole/max` 2.4e8 where
    an in-distribution screen reads 5.44 at a stricter threshold: eight orders apart, and
    a statement about unreachable poses rather than about the chart.

    Fixing it belongs HERE rather than in the fork. `ScreenDomain` is this project's
    convention (threshold as the same multiple of the coordinate limit, latent ball as
    sqrt(width) + 1.5), and the fork should not carry it. So the fork grew two neutral
    flags and this appends them.

    Only for the soft rungs, and only if the caller has not set them: the rigid arms keep
    the defaults so their archived pole curves stay comparable, which is the whole reason
    the fork's defaults were left alone.

    The screw-joint arm keeps the defaults too, deliberately. Its orientation set is
    full-dimensional (scripts/probe_orientation_freedom.py), so the independent draw is
    reachable and the callback is in distribution as it stands -- and its whole chart ladder
    trained against those defaults, so retargeting it would split one ladder's pole curves
    across two domains.
    """
    if not script.startswith("train_ddp"):
        return argv
    if any(a.split("=")[0] in ("--pole_domain", "--pole_in_distribution") for a in argv):
        return argv
    robot = None
    for i, a in enumerate(argv):
        if a.startswith("--robot_name="):
            robot = a.split("=", 1)[1]
        elif a == "--robot_name" and i + 1 < len(argv):
            robot = argv[i + 1]
    sys.path.insert(0, os.path.join(REPO, "scripts", "training"))
    from pole_metric import ScreenDomain, SoftRungNames
    if robot not in SoftRungNames():
        return argv
    base, slack, radius, threshold = ScreenDomain(robot)
    domain = {"position_base": list(base), "position_slack": slack,
              "latent_radius": radius, "pole_threshold": threshold}
    print(f"[ikflow_entry] pole screen retargeted for {robot}: {domain}, in-distribution poses")
    return argv + [f"--pole_domain={json.dumps(domain)}", "--pole_in_distribution"]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {os.path.basename(__file__)} <script> [args...]\n"
                         f"  scripts live in {FORK_SCRIPTS}")
    name = sys.argv[1]
    path = os.path.join(FORK_SCRIPTS, name if name.endswith(".py") else f"{name}.py")
    if not os.path.exists(path):
        available = sorted(f[:-3] for f in os.listdir(FORK_SCRIPTS) if f.endswith(".py"))
        raise SystemExit(f"no such ikflow script {name!r}; available: {available}")
    argv = _with_pole_domain(name, sys.argv[2:])
    print(f"[ikflow_entry] registered {src.register_robots.REGISTERED}, running {path}")
    sys.argv = [path] + argv
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
