"""Write an UNTRAINED chart for a rung, so the whole benchmark pipeline can be exercised.

The GVS push-rod arm has no chart until its dataset is built and trained on the cluster,
and the driver refuses to run without `--checkpoint` (a missing chart is the
whole-column-of-zeros failure mode). A smoke run of the pipeline -- the sampler, the shelf
targets, both arms, `verify()`, the summary -- must not wait for 620k training steps, so
this writes a randomly initialised chart in the export contract's own format: a bare
`nn_model` state-dict pickle plus its `.arch.json` sidecar, named `<rung>__untrained__step0.pkl`
so it can never be mistaken for a trained one. It says nothing about solve quality.

    .venv/bin/python scripts/gvs_arm/make_untrained_chart.py --rung gvs_pushrod9_o1
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch  # noqa: E402
from ikflow.ikflow_solver import IKFlowSolver  # noqa: E402
from ikflow.model import IkflowModelParameters  # noqa: E402
from jrl.robots import get_robot  # noqa: E402

import src.register_robots  # noqa: E402,F401
from src.flow_loading import LEGACY_ARCH_BY_ROBOT, LoadFlowSolver, WriteArch  # noqa: E402
from src.gvs_arm.params import PRIMARY, RUNGS  # noqa: E402
from src.utils import RepoDir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rung", choices=sorted(RUNGS), default=PRIMARY)
    p.add_argument("--nb_nodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arch = dict(LEGACY_ARCH_BY_ROBOT[args.rung], nb_nodes=args.nb_nodes)
    parameters = IkflowModelParameters()
    parameters.__dict__.update(arch)
    torch.manual_seed(args.seed)
    solver = IKFlowSolver(parameters, get_robot(args.rung))

    out = os.path.join(RepoDir(), "models", args.rung, f"{args.rung}__untrained__step0.pkl")
    with open(out, "wb") as handle:
        pickle.dump({k: v.detach().cpu() for k, v in solver.nn_model.state_dict().items()},
                    handle)
    WriteArch(out, dict(parameters.__dict__),
              provenance={"untrained": True, "seed": args.seed,
                          "purpose": "pipeline smoke test; says nothing about solve quality"})
    ## Round-trip through the single funnel, so the sidecar is cross-checked against the
    ## weights exactly as a trained export would be.
    LoadFlowSolver(args.rung, out)
    print(f"wrote {os.path.relpath(out, RepoDir())} (+ .arch.json), nb_nodes={args.nb_nodes}, "
          f"dim_latent_space={arch['dim_latent_space']}")


if __name__ == "__main__":
    main()
