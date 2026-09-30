"""Dump SoRoMoX's forward kinematics to a committed golden file.

SoRoMoX is this robot's DEFINITION: the arm is its spatial Piecewise Constant Strain model,
and `src/soft_arm/kinematics.py` is our torch reimplementation of the same map -- necessary
because the solver needs it differentiable in torch and driving a Drake plant, and SoRoMoX is
JAX. "The analytic model from the soft robot repo" is only an honest description of our map if
that equivalence is CHECKED, which is what this file exists for.

WHY A GOLDEN FILE RATHER THAN A DIRECT COMPARISON. A test that imports SoRoMoX runs only where
SoRoMoX is installed -- which is one disposable venv on one laptop -- and this project's own
lesson is that a check which silently stops running is indistinguishable from one that passes.
The golden file is committed, so the equivalence test runs everywhere with no JAX at all; a
separate opt-in test regenerates it and asserts it has not moved.

JAX DEFAULTS TO FLOAT32. A 1e-9 agreement claim against a float32 oracle is a fiction, so
`jax_enable_x64` is set before anything else and recorded in the file's metadata.

Run with the disposable oracle venv, NOT the project venv:
    .venv-soromox/bin/python scripts/soft_arm/generate_fk_golden.py
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from soromox.systems.components.links import LinkSpec  # noqa: E402
from soromox.systems.pcs.pcs import PCS  # noqa: E402
from soromox.systems.pcs.structures import PCSStructure  # noqa: E402

from src.soft_arm.params import RUNGS  # noqa: E402  -- pure dataclasses, no torch

#: Elastic parameters. They do not enter the KINEMATICS at all -- PCS forward kinematics is
#: the exponential of the strain twist and knows nothing about stiffness -- but `LinkSpec`
#: requires them, so they are fixed here and recorded, rather than left to look meaningful.
_DENSITY = 1000.0
_YOUNG_MODULUS = 1e5
_POISSON_RATIO = 0.45
_SAMPLES = 256
_SEED = 0


def BuildModel(spec):
    """The SoRoMoX model of one rung, from the same spec our torch map reads."""
    selector = np.zeros(6, dtype=bool)
    for index in spec.strain_basis:
        selector[index] = True
    ## The selector is the FULL 6 * num_segments vector, not a per-segment pattern that
    ## SoRoMoX broadcasts -- it raises otherwise, which is how this was established.
    selector = jnp.asarray(np.tile(selector, spec.num_segments))
    ## Reference strain: an unstretched rod advances one unit of arc length per unit of arc
    ## length, so the axial entry is 1 and `sigma_z` is the DEVIATION from it. Same
    ## convention as `kinematics.cfg_to_twists`.
    reference = np.zeros(6)
    reference[5] = 1.0
    links = [LinkSpec.circular(length=spec.segment_length, radius=spec.backbone_radius,
                               density=_DENSITY, reference_strain=jnp.asarray(reference),
                               young_modulus=_YOUNG_MODULUS, poisson_ratio=_POISSON_RATIO)
             for _ in range(spec.num_segments)]
    return PCS.from_links(links, structure=PCSStructure(strain_selector=selector))


def main():
    ## Deliberately NOT `src.utils.RepoDir`: that module imports pydrake, and the oracle venv
    ## has no Drake -- keeping them apart is the point of it being a separate venv.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    out = os.path.join(repo, "tests", "data", "soft_arm_fk_golden.npz")
    payload = {"jax_enable_x64": np.array(jax.config.jax_enable_x64),
               "soromox_version": np.array(getattr(__import__("soromox"), "__version__", "?")),
               "samples": np.array(_SAMPLES), "seed": np.array(_SEED),
               "density": np.array(_DENSITY), "young_modulus": np.array(_YOUNG_MODULUS),
               "poisson_ratio": np.array(_POISSON_RATIO)}

    generator = np.random.default_rng(_SEED)
    for name, spec in sorted(RUNGS.items()):
        robot = BuildModel(spec)
        ## Normalized configurations in [-1, 1], the coordinates every formulation decides
        ## over, including the corners where the arc is tightest.
        cfg = generator.uniform(-1.0, 1.0, size=(_SAMPLES, spec.ndof))
        cfg[0] = 0.0                                  # the straight arm
        cfg[1] = 1.0                                  # a corner
        cfg[2] = -1.0                                 # the opposite corner
        limits = np.asarray(spec.limits_per_dof)
        tips = np.stack([np.asarray(robot.forward_kinematics_tips(jnp.asarray(c * limits)))
                         for c in cfg])
        payload[f"{name}/cfg"] = cfg
        payload[f"{name}/tips"] = tips               # (samples, num_segments, 4, 4)
        print(f"{name}: {tips.shape} segment-tip transforms over {_SAMPLES} configurations")

    np.savez_compressed(out, **payload)
    print(f"\nwrote {os.path.relpath(out, repo)}")


if __name__ == "__main__":
    main()
