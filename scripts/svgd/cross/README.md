# Cross-solver readers for the svgd record (2026-10-12)

Throwaway-grade scripts, kept because their numbers are in `docs/svgd-solver.md` ("R20", "R21") and
CLAUDE.md. Run from the repo root (or a worktree with `results` symlinked) with `.venv/bin/python`;
each reads the promoted `results/<robot>/benchmark/sc_SVGD_R20_* / sc_SVGD_R21_*` runs and the IPOPT
twins (`sc_SVGD_R2_*` Panda, `sc_PAPER_*` the rest) and asserts equal `grid_hash` before pairing.

- `learned_svgd_vs_js_ipopt.py`: learned under svgd (pin2) against joint space under IPOPT, same
  cells: success, discordant cells, exact McNemar, time-matched joint space (restarts from the
  target's recorded starts inside svgd's wall on that cell, random order, 200 permutations, and the
  starved fraction), cost on shared cells, mean wall.
- `js_svgd_vs_js_ipopt_and_learned.py`: joint space under svgd (pin2kq; R20 Panda, R21 the rest)
  against joint space under IPOPT and against learned under svgd, same cells, plus clock stops,
  feasible particles and outer steps.
- `runtime_vs_js_r6_r7.py`: the earlier (R6 K1 / R7 kq) version of the multistart comparison on the
  Panda, kept for the record of how the 55-65% / ~90% figures in CLAUDE.md were produced.
