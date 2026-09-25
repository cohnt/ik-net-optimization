"""Register every robot this project defines with `jrl.robots`, once.

ikflow resolves a robot by name through `jrl.robots.get_robot`, which scans
`jrl.robots.ALL_CLCS`. A robot this project defines is not in that list until its register
module is imported, so this is the seam where that happens -- ONE list, so the next custom
robot adds one entry here and touches nothing else.

It is robot-generic on purpose. The alternative, hardcoding a particular robot's register
module at each of the three import sites, is how the same need was met before, and it means
every new robot has to find all three. The three are:

  * `src/<robot>_program.py`, because the program resolves a chart by robot name;
  * `src/flow_loading.py`, because `LoadFlowSolver` is the SINGLE funnel every by-name
    lookup passes through -- the programs, all three checkpoint screens, and the export
    round-trip. Registering only where the programs import it leaves `get_robot` raising
    inside the screens, which run at the END of a 620k-step training job;
  * `scripts/training/ikflow_entry.py`, before the vendored fork's script is even located.

IMPORT ORDER IS LOAD-BEARING at that third site: ikflow resolves `DATASET_DIR` from
`expanduser("~")` AT IMPORT, and a path bug of exactly that shape once cost a training rung
its entire run.
"""

#: Modules whose import registers robots. Each is expected to be idempotent and to expose
#: `REGISTERED`, a tuple of the names it added.
MODULES = ("src.helix_arm.register",)


def RegisterAll():
    """Import every register module and return the full tuple of names added."""
    import importlib

    names = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        names.extend(getattr(module, "REGISTERED", ()))
    return tuple(names)


def ProjectRobotNames():
    """Sorted names of the robots this project defines.

    For `--robot` choices in the checkpoint screens, so a new robot does not have to be
    spelled into three argument parsers.
    """
    return sorted(RegisterAll())


REGISTERED = RegisterAll()
