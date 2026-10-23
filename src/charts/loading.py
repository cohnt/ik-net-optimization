"""Load any chart from a checkpoint, dispatching on its sidecar's `family`.

The `.arch.json` sidecar every checkpoint carries (`src/flow_loading.py`) gains one key,
`family`. It is ABSENT from every sidecar written before the chart ablation, and absent
means "ikflow", so all of them keep loading exactly as they did -- through `LoadFlowSolver`,
whose behaviour is untouched (its `ReadArch` keeps only `ARCH_FIELDS`, so an explicit
`"family": "ikflow"` is dropped there and changes nothing). A sidecar-less checkpoint is a
flow too: that is `LoadFlowSolver`'s legacy fallback, and no other family predates the
sidecar.
"""

import json
import os

from src.charts.base import Chart
from src.charts.flow import FlowChart

DEFAULT_FAMILY = "ikflow"
FAMILIES = ("ikflow", "cvae", "loink")


def ChartFamily(checkpoint):
    """The `family` a checkpoint's sidecar declares ("ikflow" when absent, or no sidecar)."""
    from src.flow_loading import SidecarPath
    sidecar = SidecarPath(checkpoint)
    if not os.path.exists(sidecar):
        return DEFAULT_FAMILY
    with open(sidecar) as f:
        return json.load(f).get("family", DEFAULT_FAMILY)


def LoadChart(robot_name, checkpoint, fallback_arch=None, compile_model=None):
    """The chart stored at `checkpoint`, as a `Chart`.

    `fallback_arch` is the flow loader's (a sidecar-less legacy checkpoint). It is ignored
    for any other family: those sidecars are mandatory, so there is nothing for it to stand
    in for, and the programs pass their robot's flow fallback unconditionally.
    """
    family = ChartFamily(checkpoint)
    if family == "ikflow":
        from src.flow_loading import LoadFlowSolver
        return FlowChart(LoadFlowSolver(robot_name, checkpoint, fallback_arch=fallback_arch,
                                        compile_model=compile_model))
    if family == "cvae":
        try:
            from src.charts.cvae import LoadCVAEChart
        except ImportError as e:
            raise ImportError(f"{checkpoint} declares family 'cvae', but src.charts.cvae "
                              f"could not provide LoadCVAEChart: {e}") from e
        return LoadCVAEChart(robot_name, checkpoint, compile_model=compile_model)
    if family == "loink":
        try:
            from src.charts.loink import LoadLoinkChart
        except ImportError as e:
            raise ImportError(f"{checkpoint} declares family 'loink', but src.charts.loink "
                              f"could not provide LoadLoinkChart: {e}") from e
        return LoadLoinkChart(robot_name, checkpoint, compile_model=compile_model)
    raise ValueError(f"{checkpoint}: unknown chart family {family!r} in its sidecar "
                     f"(known: {', '.join(FAMILIES)})")


def WrapChart(obj):
    """`obj` as a `Chart`: itself if it already is one, else a bare `IKFlowSolver` (one
    passed in as `model=`, or the upstream Panda chart from `get_ik_solver`) in a
    `FlowChart`."""
    return obj if isinstance(obj, Chart) else FlowChart(obj)
