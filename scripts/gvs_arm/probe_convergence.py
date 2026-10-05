"""How accurate is the discretization? Tip-pose error against SoRoMoX's quadrature order.

The soft PCS arm's discretization was EXACT (a constant twist commutes with itself, so
splitting a segment changes nothing). This robot's is not: SoRoMoX integrates the
variable-strain rod with a Magnus expansion on a Gauss-Legendre grid of `num_gauss_points`
per segment, and the same quadrature carries the stiffness and the rod-force integrals, so
that one number sets the fidelity of the whole forward model. The spec fixes it; this
measures what it buys, in float64, against a dense reference (`--reference` points), over
uniform draws of the rod forces on every rung.

The number to read is the tip error at the FIELDED order relative to the task tolerance
(1e-3 m). Reported, and recorded in docs/gvs-arm.md; not tuned to a success count.

    GVS_ARM_XLA_THREADS=4 .venv/bin/python scripts/gvs_arm/probe_convergence.py
"""

import argparse
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.gvs_arm.model import GvsArmModel  # noqa: E402
from src.gvs_arm.params import RUNGS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rungs", default=",".join(sorted(RUNGS)))
    p.add_argument("--orders", default="5,7,9,12,16")
    p.add_argument("--reference", type=int, default=40)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for name in args.rungs.split(","):
        spec = RUNGS[name]
        cfgs = rng.uniform(-1.0, 1.0, size=(args.draws, spec.ninputs))
        reference = GvsArmModel(dataclasses.replace(spec, name=f"{name}_ref",
                                                    num_gauss_points=args.reference))
        tips_ref, ok = reference.TipPoseBatch(cfgs)
        assert ok.all()
        print(f"\n{name} (fielded num_gauss_points = {spec.num_gauss_points}), "
              f"reference {args.reference} points, {args.draws} draws")
        print(f"{'points':>7}  {'median mm':>10}  {'p99 mm':>8}  {'max mm':>8}  {'max deg':>8}")
        for order in (int(x) for x in args.orders.split(",")):
            model = GvsArmModel(dataclasses.replace(spec, name=f"{name}_g{order}",
                                                    num_gauss_points=order))
            tips, ok = model.TipPoseBatch(cfgs)
            assert ok.all()
            err_mm = np.linalg.norm(tips[:, :3] - tips_ref[:, :3], axis=1) * 1000
            dots = np.abs((tips[:, 3:] * tips_ref[:, 3:]).sum(axis=1)).clip(0, 1)
            err_deg = np.degrees(2 * np.arccos(dots))
            flag = "  <- fielded" if order == spec.num_gauss_points else ""
            print(f"{order:>7}  {np.median(err_mm):>10.4f}  {np.percentile(err_mm, 99):>8.4f}  "
                  f"{err_mm.max():>8.4f}  {err_deg.max():>8.4f}{flag}")


if __name__ == "__main__":
    main()
