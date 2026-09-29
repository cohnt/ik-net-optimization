"""Register every robot this project defines with `jrl.robots`, once.

ikflow resolves a robot by name through `jrl.robots.get_robot`, which scans
`jrl.robots.ALL_CLCS`. A robot this project defines is not in that list until its register
module is imported, so this is the seam where that happens -- ONE list, so the next custom
robot adds one entry here and touches nothing else.

It is robot-generic on purpose. The alternative, hardcoding a particular robot's register
module at each of the three import sites, is how the soft PCS arm was wired, and it means
every new robot has to find all three:

  * `src/<robot>_program.py`, because the program resolves a chart by robot name;
  * `src/flow_loading.py`, because `LoadFlowSolver` is the SINGLE funnel every by-name
    lookup passes through -- the programs, all three checkpoint screens, and the export
    round-trip. Registering only where the programs import it leaves `get_robot` raising
    inside the screens, which run at the END of a 620k-step training job;
  * `scripts/training/ikflow_entry.py`, before the vendored fork's script is even located.

IMPORT ORDER IS LOAD-BEARING at that third site: ikflow resolves `DATASET_DIR` from
`expanduser("~")` AT IMPORT.

Registering a robot must stay CHEAP: `src.gvs_arm.register` imports the JAX model module but
builds no model (`GetModel` is lazy), so importing this from a Panda benchmark costs a JAX
import and nothing else.
"""

#: Modules whose import registers robots. Each is idempotent and exposes `REGISTERED`.
MODULES = ("src.soft_arm.register", "src.gvs_arm.register")


def RegisterAll():
    """Import every register module and return the full tuple of names added."""
    import importlib

    names = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        names.extend(getattr(module, "REGISTERED", ()))
    return tuple(names)


def ProjectRobotNames():
    """Sorted names of the robots this project defines -- for `--robot` choice lists."""
    return sorted(RegisterAll())


REGISTERED = RegisterAll()
