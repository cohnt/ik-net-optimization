"""End-to-end checks of `src/svgd/solver.py` on the Panda pose task (n6 chart).

FIRST WAVE: only the end-to-end cell on Panda learned pose and on Panda joint-space pose is
exercised, each from a PAIRED start (`SetStartFromQ` at a random collision-free q, the target
the scene frame of another random q, as the benchmark samples), under `al_svgd` with the
configurations named in CONFIGS. Each result is scored by `benchmark.verify` with the
benchmark's pose gate (copied from `scripts/panda/panda_benchmark.py`) at `tol = 1e-4`, and
the per-configuration numbers are PRINTED: that is the first real signal of the project, so
they are reported plainly whether or not they are good.

Asserted: no exception; `last_iterate` set; `wall_time <= cap + 1`; `details.map_evals`
agrees with `program.eval_counts["map_forward"]`; `drake_feasible == verify's feasible`.
NOT YET WRITTEN (next wave): the grasp cells, native init, determinism, the wall-clock stop
on N = 256, the NaN-injection resample check (`SvgdSolver(program, particles_override=...)`
exists for it) and the subprocess run of `panda_benchmark.py --solver svgd`.

Run as a script: `.venv/bin/python tests/test_svgd_solver.py` (spawns collision workers, so
it needs the `__main__` guard).
"""

import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix   # noqa: E402
from pydrake.common.eigen_geometry import Quaternion                    # noqa: E402

from src import benchmark as bm                                         # noqa: E402
from src.generic_program import orientation_error_rpy                   # noqa: E402
from src.svgd.result import SvgdResult                                  # noqa: E402
from src.svgd.solver import SvgdSolver                                  # noqa: E402
from src.utils import HiddenPrints                                      # noqa: E402
import test_batched_program as T                                        # noqa: E402

WALL = 20.0
CONFIGS = [
    ("al_svgd N=1 kernel=none", dict(svgd_method="al_svgd", svgd_n=1, svgd_kernel="none")),
    ("al_svgd N=64 kernel=none", dict(svgd_method="al_svgd", svgd_n=64, svgd_kernel="none")),
    ("al_svgd N=64 kernel=q", dict(svgd_method="al_svgd", svgd_n=64, svgd_kernel="q")),
    ("tsvgd N=64 kernel=q", dict(svgd_method="tsvgd", svgd_n=64, svgd_kernel="q")),
]
FAILURES = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"\n          {detail}"))
    if not cond:
        FAILURES.append(name)


def pose_gate(task_tol=1e-3, ori_tol=0.01):
    def task_gate(program, q):
        translation, wxyz = program.fk(q)
        target = program.target_pose
        axis_max = float(np.max(np.abs(np.asarray(translation) - target[:3])))
        target_rpy = RollPitchYaw(RotationMatrix(Quaternion(target[3:]))).vector()
        rpy_max = float(np.max(np.abs(np.asarray(orientation_error_rpy(wxyz, target_rpy), dtype=float))))
        return axis_max <= task_tol and rpy_max <= ori_tol, dict(pos_error=axis_max, rpy_error=rpy_max)
    return task_gate


def run_cell(arm, label, overrides, wall=WALL, seed=7):
    p = T.program("panda", "pose", arm)
    opts = replace(p.options, which_solver="svgd", max_wall_time=wall, acceptable_constr_viol_tol=1e-4,
                   file_print_name=os.path.join(T.RepoDir(), "results", f"_test_svgd_{arm}.log"),
                   **overrides)
    os.makedirs(os.path.dirname(opts.file_print_name), exist_ok=True)
    p.options = opts
    rng = np.random.default_rng(seed)
    q_init = T.collision_free_q(p, rng, 1)[0]
    with HiddenPrints():
        p.SetStartFromQ(q_init)
    p.ResetEvalCounts()
    t0 = time.time()
    with HiddenPrints():
        result = p.Solve()
    wall_time = time.time() - t0
    d = result.get_solver_details()
    verdict = bm.verify(p, result, pose_gate(), 1e-4, relaxed_tol=1e-3)
    cost = bm.reported_cost(p, result, p.options.joint_centering_cost)
    print(f"  [{arm}] {label}: feasible={verdict.feasible} fail={verdict.fail_reason!r} "
          f"max_viol={verdict.detail.get('max_violation')} cost={cost:.4g} "
          f"outer={d.iterations} steps={d.inner_steps} map_evals={d.map_evals} "
          f"n_feasible={d.n_feasible} resampled={d.n_resampled} status={d.status_name!r} "
          f"wall={wall_time:.1f}s collision={d.collision_seconds:.1f}s phases={ {k: round(v, 2) for k, v in d.phase_times.items()} }")
    if d.inner_steps:
        print(f"      ms/step = {1e3 * d.phase_times.get('swarm', 0.0) / d.inner_steps:.1f}")
    check(f"[{arm}] {label}: returns an SvgdResult", isinstance(result, SvgdResult))
    check(f"[{arm}] {label}: last_iterate set", getattr(p, "last_iterate", None) is not None)
    check(f"[{arm}] {label}: wall_time <= cap + 1", wall_time <= wall + 1.0, f"{wall_time}")
    check(f"[{arm}] {label}: map_evals agrees with eval_counts",
          ## `verify` and the Drake re-check evaluate single points through QAndPose, so the
          ## program's counter is >= the batched count; the delta recorded by the solver is exact.
          d.map_evals == d.extras["eval_counts_delta"]["map_forward"]
          and d.map_evals <= p.eval_counts["map_forward"], str(p.eval_counts))
    check(f"[{arm}] {label}: drake_feasible == verify feasible",
          bool(d.drake_feasible) == bool(verdict.feasible), f"{d.drake_feasible} vs {verdict}")
    return verdict.feasible


def test_panda_pose_learned_paired():
    print("\n--- Panda learned pose, paired start ---")
    any_ok = False
    for label, ov in CONFIGS:
        any_ok |= bool(run_cell("learned", label, ov))
    check("learned pose: at least one configuration is verify-feasible (go/no-go)", any_ok)


def test_panda_pose_numerical_paired():
    print("\n--- Panda joint-space pose, paired start ---")
    any_ok = False
    for label, ov in CONFIGS[1:3]:
        any_ok |= bool(run_cell("numerical", label, ov))
    check("numerical pose: at least one configuration is verify-feasible (go/no-go)", any_ok)


def main():
    which = sys.argv[1:] or ["learned", "numerical"]
    if "learned" in which:
        test_panda_pose_learned_paired()
    if "numerical" in which:
        test_panda_pose_numerical_paired()
    T.close_all()
    print(f"\n{len(FAILURES)} failed")
    for f in FAILURES:
        print("  FAILED:", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
