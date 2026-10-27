"""The fielded pin of 2026-10-12 and the gate-edge rule, end to end on the Panda (n6).

    .venv/bin/python tests/test_svgd_pin_and_gate.py            (the checks; ~3 min on the GPU)
    Tiers (tests/_tiers.py): the default `fast` runs (a) as pinned, (b) at the target's own
    configuration on an N=16 swarm, and (c), in ~N_FAST s on one CPU core; `slow` (~N_SLOW s,
    CPU ok) holds (b) from the generic start at N=64.
    .venv/bin/python tests/test_svgd_pin_and_gate.py capture    (rewrite the reference: DON'T,
                                                                 unless the old pin's behaviour
                                                                 is meant to change)

Thomas's rulings (docs/svgd-solver.md, "RULINGS 2026-10-12" and "Gate-edge rule"): the defaults
moved to c x z + rho = 1e5 + the H / T metric + the native-draw paired swarm, and a stall stop is
accepted only once the float64 Drake re-check passes -- otherwise the swarm continues at a
tightened internal feasibility level. On 2026-10-14 the temperature options (`svgd_temperature`,
`svgd_metric_temperature`) were removed (stage SVGD_R22 measured T inert): the step is the T = 1
one, at which the H / T metric was H bit for bit, so neither pin names a metric form any more.
What is pinned here:

  (a) THE OLD PIN, SET EXPLICITLY, REPRODUCES THE PRE-CHANGE SOLVER BIT FOR BIT. Three cells
      (Panda pose learned and pose joint space under the default stop rule, grasp learned at a
      10-check step cap; paired start, N = 64, eager step) solved at `svgd_kernel=q, svgd_rho=1000,
      svgd_paired_init=jitter` (and, when captured, `svgd_metric_temperature=off`, T = 1 -- the
      step that is now the only one) must return the point, the outer-step count, the best-merit
      history, the selected particle and the Drake violation recorded in
      `tests/data/svgd_oldpin_reference.npz`, which was captured on the tree BEFORE the change
      (svgd d200121). So moving the defaults does not alter explicit-option behaviour, and the
      gate rule is a no-op when the first re-check passes (each pose reference cell stops
      converged on its first re-check: `n_recheck_failures == 0`, `feas_level == 1.0`). The arrays are float32
      GPU arithmetic, so the comparison is bitwise on the GPU they were captured on and is SKIPPED,
      loudly, on any other device.
  (b) THE GATE-EDGE RULE FIRES AND RECOVERS: with the first re-check forced to fail (every
      candidate's Drake violation read as 1.0005e-4, just outside the 1e-4 gate), the swarm does
      not stop; it runs more outer steps than the uninjected solve, records one failure, stops at
      feasibility level 0.9, and returns a Drake-feasible point.
  (c) Under the native-draw paired swarm (the new default) particle 0 is EXACTLY the shared
      guess. (The old (c), metric `on` == `off` at T = 1 on the fused step, went with the option.)
"""
import os
import sys
from dataclasses import replace
from functools import partial

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(HERE))
sys.path.append(HERE)

import test_svgd_solver as S                                          # noqa: E402
import _tiers                                                         # noqa: E402
from src.generic_program import ProgramOptions                        # noqa: E402
from src.svgd.solver import SvgdSolver                                # noqa: E402

REF = os.path.join(HERE, "data", "svgd_oldpin_reference.npz")
#: The pin every svgd round through R19 ran (kernel q, metric off and init jitter through the
#: then-defaults; rho 1000 named by the manifests).
OLD_PIN = S.LEGACY_PIN
#: The pin fielded 2026-10-12 (it also named `svgd_metric_temperature=on`, removed 2026-10-14).
NEW_PIN = dict(svgd_kernel="cz", svgd_rho=100000.0, svgd_paired_init="native")
#: Every other svgd field a cell depends on, named so that options left on a cached test program
#: by an earlier test (run_cell accumulates them) cannot leak into these solves.
#: The fielded stall patience is 3 since 2026-10-16; the reference was recorded at 5, named here.
REST = dict(svgd_n=64, svgd_outer_iters=300, svgd_stop_patience=5, svgd_stop_rel=1e-3,
            svgd_inner_iters=10, svgd_recheck_topk=10, svgd_lr=0.3, svgd_gn_lm=10.0,
            svgd_resample="native", svgd_compile=False,
            svgd_cuda_graph=False, svgd_dtype="float32", svgd_seed=0, svgd_warmup="none",
            svgd_constraint_inside_kernel=False, svgd_metric="gn", svgd_multiplier_max=1e4,
            svgd_dual_lr=None, svgd_jitter=0.1, svgd_collision_backend="autodiff",
            svgd_time_reserve=0.1, svgd_polish="none", max_iter=None)
#: (task, arm, extra overrides). The grasp cell runs to the clock under the stop rule on the eager
#: step, which no reference can pin, so it is held to a 10-check step cap instead; the two pose
#: cells stop on the stall rule (and so pass through the gate-edge re-check).
CELLS = (("pose", "learned", {}), ("mug", "learned", dict(svgd_outer_iters=10)),
         ("pose", "numerical", {}))
FIELDS = ("x", "iterations", "inner_steps", "best_history", "selected_index",
          "drake_max_violation", "n_feasible", "status")


def _solve(task, arm, pin, wall=60.0, q_init=None, **extra):
    p = S.program_for(task, arm)
    _, d, result = S.run_cell(p, task, f"{task}/{arm}", {**REST, **pin, **extra}, wall=wall,
                              seed=7, quiet=True, q_init=q_init)
    out = dict(x=np.asarray(result.get_x_val(), dtype=float), iterations=int(d.iterations),
               inner_steps=int(d.inner_steps),
               best_history=np.asarray(d.extras["best_history"], dtype=float),
               selected_index=int(d.selected_index),
               drake_max_violation=float(d.extras["drake_max_violation"]),
               n_feasible=int(d.n_feasible), status=str(d.status_name))
    return out, d


def _device():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def capture():
    """Write the reference (run on the PRE-change tree)."""
    arrays = {"device": np.array(_device())}
    for task, arm, extra in CELLS:
        out, d = _solve(task, arm, OLD_PIN, **extra)
        print(f"  {task}/{arm}: {out['status']} stop {d.stop_reason} outer {out['iterations']} "
              f"selected {out['selected_index']} drake {out['drake_max_violation']:.3g}")
        assert d.stop_reason == ("step_cap" if extra else "converged"), (task, arm, d.stop_reason)
        for k in FIELDS:
            arrays[f"{task}_{arm}_{k}"] = np.asarray(out[k])
    os.makedirs(os.path.dirname(REF), exist_ok=True)
    np.savez(REF, **arrays)
    print(f"wrote {REF}")


def test_old_pin_reproduces_the_pre_change_solver():
    print("\n--- (a) the old pin, set explicitly, against the pre-change reference ---")
    ref = np.load(REF)
    if str(ref["device"]) != _device():
        print(f"  SKIP: reference captured on {ref['device']}, this is {_device()}")
        return
    for task, arm, extra in CELLS:
        out, d = _solve(task, arm, OLD_PIN, **extra)
        bad = [k for k in FIELDS if not np.array_equal(np.asarray(out[k]), ref[f"{task}_{arm}_{k}"])]
        S.check(f"old pin {task}/{arm}: {', '.join(FIELDS)} bit for bit", not bad,
                f"differs: {bad}; outer {out['iterations']} vs {ref[f'{task}_{arm}_iterations']}")
        S.check(f"old pin {task}/{arm}: the gate rule did not fire (one re-check, passed)",
                d.n_recheck_failures == 0 and d.feas_level == 1.0
                and d.stop_reason == ("step_cap" if extra else "converged"),
                f"{d.n_recheck_failures} {d.feas_level} {d.stop_reason}")


def test_gate_edge_rule_continues_after_a_failed_recheck(at_target=False):
    ## The fast tier starts both solves at the configuration the target was generated from, on
    ## an N=16 swarm (`test_svgd_solver.AT_TARGET`): the rule is what is checked, not the reach.
    ## (the stall patience stays REST's 5).
    S.pose_program("learned")                                   # draws the target configuration
    start = (dict(S.AT_TARGET, svgd_stop_patience=REST["svgd_stop_patience"],
                  q_init=S._POSE["q_target"]) if at_target else {})
    print(f"\n--- (b) the gate-edge rule: the first float64 re-check forced to fail"
          f"{' (at the target configuration, N=16)' if at_target else ''} ---")
    base, d0 = _solve("pose", "learned", NEW_PIN, **start)
    S.check("gate: the uninjected solve stops converged on its first re-check",
            d0.stop_reason == "converged" and d0.n_recheck_failures == 0 and d0.feas_level == 1.0
            and d0.drake_feasible, f"{d0.stop_reason} {d0.n_recheck_failures} {d0.feas_level}")
    orig = SvgdSolver._drake_recheck
    calls = []

    def injected(self, *args, **kw):
        calls.append(1)
        if len(calls) > 1:
            return orig(self, *args, **kw)
        true_viol = self._drake_max_violation
        self._drake_max_violation = lambda x: 1.0005e-4      # just outside the 1e-4 gate
        try:
            return orig(self, *args, **kw)
        finally:
            self._drake_max_violation = true_viol
    SvgdSolver._drake_recheck = injected
    try:
        out, d = _solve("pose", "learned", NEW_PIN, **start)
    finally:
        SvgdSolver._drake_recheck = orig
    print(f"    uninjected: outer {base['iterations']}; injected: outer {out['iterations']}, "
          f"re-checks {len(calls)}, failures {d.n_recheck_failures}, level {d.feas_level}, "
          f"stop {d.stop_reason}, drake {d.drake_feasible} ({out['drake_max_violation']:.3g})")
    S.check("gate: the swarm continued past the failed re-check",
            out["iterations"] > base["iterations"], f"{out['iterations']} vs {base['iterations']}")
    S.check("gate: one failure recorded, level tightened to 0.9",
            d.n_recheck_failures == 1 and d.feas_level == 0.9, f"{d.n_recheck_failures} {d.feas_level}")
    S.check("gate: the returned cell is Drake-feasible, stopped converged",
            d.drake_feasible and d.solver_feasible and d.stop_reason == "converged"
            and out["drake_max_violation"] <= 1e-4, f"{d.drake_feasible} {d.stop_reason}")
    S.check("gate: two re-checks in all (the failed one and the passing one)", len(calls) == 2,
            str(len(calls)))


def test_native_paired_swarm_keeps_particle_0_at_the_guess():
    print("\n--- (c) the native-draw paired swarm: particle 0 is the shared guess ---")
    for arm in ("learned", "numerical"):
        p = S.program_for("pose", arm)
        p.options = replace(p.options, which_solver="svgd", **REST,
                            **(NEW_PIN if arm == "learned" else dict(NEW_PIN, svgd_kernel="q")))
        sol = SvgdSolver(p)
        sol._build()
        x0 = np.asarray(p.prog.GetInitialGuess(p.lumped_vars), dtype=float)
        X, clip, _ = sol._init_particles(x0)
        X0 = X[0].detach().cpu().double().numpy()
        S.check(f"native paired init ({arm}): particle 0 is the guess exactly, unclipped",
                np.array_equal(X0, x0.astype(np.float32).astype(float)) and float(clip[0]) == 0.0,
                f"{np.abs(X0 - x0).max()}")
    S.check("defaults: the fielded pin of 2026-10-12",
            all(getattr(ProgramOptions(), k) == v for k, v in NEW_PIN.items())
            and ProgramOptions().svgd_n == 64,
            str({k: getattr(ProgramOptions(), k) for k in NEW_PIN}))


GROUPS = {
    ## (a) skips loudly off the capture GPU, and on it is the pin: unchanged in both tiers.
    "fast": [test_old_pin_reproduces_the_pre_change_solver,
             partial(test_gate_edge_rule_continues_after_a_failed_recheck, at_target=True),
             test_native_paired_swarm_keeps_particle_0_at_the_guess],
    "slow": [test_gate_edge_rule_continues_after_a_failed_recheck],
}


def main():
    if sys.argv[1:] == ["capture"]:
        capture()
        return 0
    _tiers.run(GROUPS, after=S.release)
    print(f"\n{len(S.FAILURES)} failed")
    for f in S.FAILURES:
        print("  FAILED:", f)
    return 1 if S.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
