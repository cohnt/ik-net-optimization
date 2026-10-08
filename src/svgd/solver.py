"""The `svgd` solver's entry point: `SvgdSolver(program).solve() -> SvgdResult`.

# TODO(svgd): algorithm lands in the next wave. This file is a SKELETON that fixes the
# interface the harness calls (`__init__(program)`, `warm_up() -> seconds`,
# `solve() -> SvgdResult`) and every side effect the harness depends on -- the ones it
# cannot see from the result object:
#
#   * `program.last_iterate` is set, so an abnormal exit can still be verified from the
#     point the solver had (Thomas: no raising or killing from inside an evaluation);
#   * `program.eval_counts` is bumped through `program.QAndPose`, the one funnel every
#     arm's work is counted at (under svgd one count is one BATCHED pass -- see
#     `IKFlowProgram.ResetEvalCounts`);
#   * the visualization callback fires the way `record_iterate` fires it in `Solve()`,
#     via `program.RecordIterate`, so `--set visualize=True` / `vars_file` keep working;
#   * a text log in the `svgd` format goes to `program.options.file_print_name`
#     (`src/svgd/result.write_log`), which `src/benchmark.parse_log` reads.
#
# The skeleton evaluates the initial guess ONCE (counted as `iterations = 1`, one outer
# step consisting of one forward pass) and returns it with status "not implemented",
# `success=False`. The one-step count is deliberate: `tests/test_solver_plumbing.py`
# requires `parse_log` to recover `iterations > 0` for every log-writing solver, and the
# skeleton honestly performed one evaluation of the start.
#
# Nothing here may encode formulation-specific information (no latent prior, no
# assumption that any residual is zero-centred): the target is the program as written.
"""
import time

import numpy as np

from src.svgd.result import (STATUS_NOT_IMPLEMENTED, SvgdResult, SvgdSolverDetails,
                             status_name, write_log)


class SvgdSolver:
    def __init__(self, program):
        self.program = program
        self.options = program.options

    def warm_up(self):
        """Pay any compile cost outside the timed grid. Returns seconds spent.

        # TODO(svgd): compile the fused batched step here (the `WarmUpJacobian` pattern).
        """
        return 0.0

    def solve(self):
        program = self.program
        opts = self.options
        t0 = time.time()
        x0 = np.asarray(program.prog.GetInitialGuess(program.lumped_vars), dtype=float)

        ## One outer step: evaluate the start through the counted funnel and record it.
        program.QAndPose(x0)                     # bumps eval_counts["map_forward"]
        program.RecordIterate(x0)                # last_iterate, eval_counts["callback"], viz

        x_full = np.zeros(program.prog.num_vars())
        x_full[program.prog.FindDecisionVariableIndices(program.lumped_vars)] = x0

        details = SvgdSolverDetails(
            status=STATUS_NOT_IMPLEMENTED,
            status_name=status_name(STATUS_NOT_IMPLEMENTED),
            method=opts.svgd_method,
            n_particles=int(opts.svgd_n),
            dtype=opts.svgd_dtype,
            iterations=1,
            inner_steps=0,
            map_evals=1,
            n_feasible=0,
            n_resampled=0,
            selected_index=0,
            phase_times={"init": time.time() - t0},
            timed_out=False,
            hit_iteration_cap=False,
            solver_feasible=False,
            drake_feasible=False,
            solve_seconds=time.time() - t0,
            collision_seconds=0.0,
        )
        write_log(opts.file_print_name, details)
        return SvgdResult(program, x_full, details, success=False)
