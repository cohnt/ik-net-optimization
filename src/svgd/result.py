"""What the `svgd` solver hands back to the harness, and the log format it writes.

The benchmark reads a Drake `MathematicalProgramResult` through exactly seven calls --
`is_success`, `get_x_val`, `GetSolution`, `EvalBinding`, `get_optimal_cost`,
`get_solution_result`, `get_solver_details` (`src/benchmark.py`: `verify`, `reported_cost`,
`solver_diagnostics`, `run_grid`). `SvgdResult` duck-types those seven and nothing else.

**A hand-built Drake `MathematicalProgramResult` ABORTS THE INTERPRETER**: `set_x_val` is
guarded by a C++ assert against a decision-variable index map the Python side cannot set,
so `MathematicalProgramResult()` + `set_x_val(...)` dies with no Python traceback. Never
construct one; return this object.

Status codes are module constants, decoded by name in `SvgdSolverDetails.status_name` and
mapped onto Drake's own solver-neutral `SolutionResult` by `SvgdResult.get_solution_result`
so a record can be read without knowing which solver ran (the `solver_diagnostics`
contract). `timed_out` and `hit_iteration_cap` are carried as booleans on the details,
decoded numerically rather than from log text -- the same rule that made SNOPT's INFO 34
and NLopt's status 6 readable.

The text log is written by `write_log` and parsed by `src/benchmark.py`'s
`_LOG_PATTERNS["svgd"]` / `_EXIT_PATTERNS["svgd"]`; its line formats live HERE, in
`LOG_LINES`, so the writer and the parser share one source.
"""
from dataclasses import dataclass, field

import numpy as np
from pydrake.solvers import SolutionResult
from pydrake.symbolic import Variable

## ------------------------------- status codes --------------------------------------
STATUS_CONVERGED = 0          # a particle passed the exact Drake re-check
STATUS_WALL_CLOCK = 1         # `max_wall_time` reached between outer steps
STATUS_STEP_CAP = 2           # `svgd_outer_iters` (or `max_iter`) reached
STATUS_INFEASIBLE = 3         # ran to a stop; no particle passed the re-check
STATUS_NAN = 4                # every particle non-finite; the start is returned
STATUS_NOT_IMPLEMENTED = 5    # the skeleton: returns the initial guess untouched

STATUS_NAMES = {
    STATUS_CONVERGED: "converged",
    STATUS_WALL_CLOCK: "wall-clock limit",
    STATUS_STEP_CAP: "step cap",
    STATUS_INFEASIBLE: "infeasible",
    STATUS_NAN: "nan",
    STATUS_NOT_IMPLEMENTED: "not implemented",
}

## Which Drake verdict each status reads as. `kSolverSpecificError` is what Drake itself
## reports for NLopt's "returned nothing" cells, so a NaN swarm lands in the same bucket
## `_recover_from_last_iterate` already knows how to score.
_SOLUTION_RESULT = {
    STATUS_CONVERGED: SolutionResult.kSolutionFound,
    STATUS_WALL_CLOCK: SolutionResult.kIterationLimit,
    STATUS_STEP_CAP: SolutionResult.kIterationLimit,
    STATUS_INFEASIBLE: SolutionResult.kInfeasibleConstraints,
    STATUS_NAN: SolutionResult.kSolverSpecificError,
    STATUS_NOT_IMPLEMENTED: SolutionResult.kSolverSpecificError,
}


def status_name(status):
    return STATUS_NAMES.get(int(status), f"unknown status {status}")


@dataclass
class SvgdSolverDetails:
    """Everything the solver knows about its own run, in the shape `solver_diagnostics`
    reads. Field names are the contract with `src/benchmark.py`; add, do not rename."""
    status: int
    status_name: str
    method: str
    n_particles: int
    dtype: str
    iterations: int = 0            # OUTER steps taken -- the cross-solver "iterations" column
    inner_steps: int = 0           # total inner (per-outer) gradient steps
    map_evals: int = 0             # batched passes through the map (N particles per pass)
    n_feasible: int = 0            # particles feasible on the batched rows at stop
    n_resampled: int = 0           # particles re-drawn for |q|_inf > cap or non-finite rows
    selected_index: int = -1       # which particle was returned; -1 if none
    phase_times: dict = field(default_factory=dict)
    timed_out: bool = False
    hit_iteration_cap: bool = False
    solver_feasible: bool = False  # the returned particle passed the solver's own rows
    drake_feasible: bool = False   # ... and the exact `prog.EvalBinding` re-check
    solve_seconds: float = 0.0
    collision_seconds: float = 0.0 # host time blocked in the collision backend (the exactness premium)
    stop_reason: str = ""          # why the SWARM stopped: converged | wall_clock | step_cap
    n_dual_updates: int = 0        # dual-ascent steps taken (each on every particle)
    lam_inf_median: float = None   # |lam_i|_inf over the particles at stop: median ...
    lam_inf_max: float = None      # ... and largest
    mu_inf_median: float = None    # |mu_i|_inf over the particles at stop: median ...
    mu_inf_max: float = None       # ... and largest
    bound_clip: float = 0.0        # total clamp distance onto the true bounds (normalised y)
    n_multiplier_clipped: int = 0  # multiplier entries the +-svgd_multiplier_max clip bound
    feasible_q_spread: float = None  # median pairwise |q_a - q_b| among feasible particles at stop
    extras: dict = field(default_factory=dict)


class SvgdResult:
    """Duck-types the seven `MathematicalProgramResult` calls the harness makes.

    `x_full` is the FULL decision-variable vector, `prog.num_vars()` long, in Drake's own
    variable order -- exactly what `get_x_val()` returns on a Drake result, which is what
    `verify` scatters from and `EvalBinding` indexes into.
    """

    def __init__(self, program, x_full, details, success):
        self.program = program
        self.prog = program.prog
        self._x = np.array(x_full, dtype=float).reshape(-1)
        if self._x.shape[0] != self.prog.num_vars():
            raise ValueError(f"SvgdResult: x_full has {self._x.shape[0]} entries, the program "
                             f"has {self.prog.num_vars()} decision variables")
        self._details = details
        self._success = bool(success)

    def is_success(self):
        return self._success

    def get_x_val(self):
        return self._x.copy()

    def GetSolution(self, vars):
        """A single `Variable` -> float; a list or ndarray of Variables -> ndarray."""
        if isinstance(vars, Variable):
            return float(self._x[self.prog.FindDecisionVariableIndex(vars)])
        idx = self.prog.FindDecisionVariableIndices(np.atleast_1d(np.asarray(vars)))
        return self._x[np.asarray(idx, dtype=int)].copy()

    def EvalBinding(self, binding):
        return self.prog.EvalBinding(binding, self._x)

    def get_optimal_cost(self):
        return float(sum(float(np.asarray(self.EvalBinding(b)).sum())
                         for b in self.prog.GetAllCosts()))

    def get_solution_result(self):
        return _SOLUTION_RESULT.get(int(self._details.status),
                                    SolutionResult.kSolverSpecificError)

    def get_solver_details(self):
        return self._details


## ------------------------------------ the log ----------------------------------------
## One source for both sides: `write_log` formats with these, and `src/benchmark.py`'s
## `_LOG_PATTERNS["svgd"]` must match them. Keep "SVGD steps:" and "SVGD seconds =" stable.
LOG_LINES = (
    "SVGD method: {method}",
    "SVGD particles: {n_particles}",
    "SVGD dtype: {dtype}",
    "SVGD steps: {iterations}",
    "SVGD inner steps: {inner_steps}",
    "SVGD map evals: {map_evals}",
    "SVGD feasible particles: {n_feasible}",
    "SVGD resampled particles: {n_resampled}",
    "SVGD selected index: {selected_index}",
    "SVGD dual updates: {n_dual_updates}",
    "SVGD |lam|_inf at stop: median {lam_inf_median} max {lam_inf_max}",
    "SVGD |mu|_inf at stop: median {mu_inf_median} max {mu_inf_max}",
    "SVGD seconds = {solve_seconds:.6f}",
    "SVGD collision seconds = {collision_seconds:.6f}",
    "EXIT: {status_name}",
)


def write_log(path, details):
    """Write the per-cell text log the harness parses. Overwrites `path`.

    Not IPOPT-formatted, deliberately: `parse_log` is told the solver and uses this
    format's own patterns, so an IPOPT-shaped line here would be read as IPOPT's quantity.
    """
    if not path:
        return
    fields = {
        "method": details.method, "n_particles": details.n_particles,
        "dtype": details.dtype, "iterations": details.iterations,
        "inner_steps": details.inner_steps, "map_evals": details.map_evals,
        "n_feasible": details.n_feasible, "n_resampled": details.n_resampled,
        "selected_index": details.selected_index,
        "n_dual_updates": details.n_dual_updates,
        "lam_inf_median": details.lam_inf_median, "lam_inf_max": details.lam_inf_max,
        "mu_inf_median": details.mu_inf_median, "mu_inf_max": details.mu_inf_max,
        "solve_seconds": float(details.solve_seconds),
        "collision_seconds": float(details.collision_seconds),
        "status_name": details.status_name,
    }
    with open(path, "w") as f:
        for line in LOG_LINES:
            f.write(line.format(**fields) + "\n")
        for k, v in sorted(details.phase_times.items()):
            f.write(f"SVGD phase {k} = {float(v):.6f}\n")
        if getattr(details, "stop_reason", ""):
            f.write(f"SVGD stop reason: {details.stop_reason}\n")
        ## The per-check trace, one line per column (`_Target.TRACE_COLUMNS`): violation,
        ## feasibility, the multiplier magnitudes, clips, resampling.
        ## Informational -- the harness parses none of these lines.
        for name, values in (details.extras or {}).get("trace", {}).items():
            f.write(f"SVGD trace {name} = " + " ".join(f"{float(v):.4g}" for v in values) + "\n")
