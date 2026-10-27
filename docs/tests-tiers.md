# The tests' two tiers (branch `tests-cheap`, 2026-10-27)

Every file in `tests/` runs as `python tests/<file>.py [group ...]` through `tests/_tiers.py`:

- **`fast`** (the default): invariants, wiring, pinned defaults, record fields, gradient checks
  on small inputs. About two minutes or less per file on ONE CPU core.
- **`slow`** (opt-in): the record-scale forms, meaning solves to the stop rule or the clock from
  a generic start, full sample counts, every rung, and the compiled and CUDA-graphed paths. It
  runs on the GPU or on SuperCloud (`cluster/tests_slow_job.sh`, every file's `all`).
- **`all`**: both tiers.

A fast form is the SAME function with a keyword that shrinks inputs (swarm size, outer steps,
rung, draws, an iteration cap, a start at the target's own configuration). Every original form
is kept unchanged in `slow`. Nothing in `src/` changed, and no tolerance moved. Each check prints
`[  12.3 s] name`, so a tier can be re-measured from its own output.

**Local rules** (CLAUDE.md, "Local compute"): one file at a time, in the foreground, under
`systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`, `taskset -c 0`, thread counts
1, `CUDA_VISIBLE_DEVICES=` unless the GPU is approved.

## Seconds

Measured 2026-10-27 on the workstation, one file at a time, on one CPU core (load 2.0-3.3 on
every row of the paired columns). "main" and "default" are back to back, one after the other on
the same machine; "slow" was measured the same day under the same rules (load <= 4.1). GPU
seconds (the RTX 3080 Ti laptop GPU, approved for this) are their own column and never mixed
with CPU seconds.

| file | main default (s) | default = `fast` (s) | `slow`, CPU (s) | `slow`, GPU (s) |
| --- | ---: | ---: | ---: | ---: |
| svgd_solver | 221 (rc 1) | 76 | 276 (rc 1) | 181 (rc 1) |
| gvs_arm_model | 204 | 32 (cold cache 115) | 168 | |
| solver_plumbing | 114 | 39 | 76 | |
| svgd_soft_fastpath | 87 | 22 | 82 | 138 |
| svgd_analytic | 73 | 13 | 72 | 174 |
| svgd_pin_and_gate | 63 | 32 | 70 | 44 (`all`, (a) bitwise) |
| gvs_arm_program | 60 | 19 (cold cache 51) | no slow | |
| resume_e2e | 42 (rc 1, crash) | 0 (pointer) | needs GPU | 353 |
| robot_settings_unified | 41 | 39 | no slow | |
| batched_flow | 41 | 20 | 33 | |
| constraint_bounds | 35 | 34 | no slow | |
| svgd_result | 32 | 24 | 11 | |
| batched_program | 28 (rc 1) | 25 (rc 1) | no slow | fast 64, passes |
| cvae_chart | 27 | 25 | no slow | |
| 22 other files | 3-24 each | unchanged | no slow | flow_cuda_graph fast 26, soft_arm_program fast 6 (passes) |
| **all 36 files** | **1261** | **579** | | |

The `slow` CPU seconds are each file's slow group alone. `svgd_solver`'s rc 1 in slow is the
original `test_polish`, which fails on every device (see below). The cold-cache figures are the
first run on a machine. The GVS files fill a JAX persistent compilation cache at
`~/.cache/learned-ik/jax` (`_tiers.jax_compilation_cache()`, ~8 MB); every later run reads it.

## Coverage moved (nothing deleted)

Every shrunk check is the same function with a keyword: the fast tier calls it with the
keyword, the slow tier without. No check was deleted.

| file: check | what it guards | `fast` now | full form now |
| --- | --- | --- | --- |
| svgd_solver: grasp_cells | record contract on Panda mug, 2 arms x 2 inits | same 4 cells, N=16, 2 outer steps | `slow` (N=64, to stop rule / 20 s) |
| svgd_solver: wall_clock_stop | clock stop, iterate kept, overrun <= step granularity | N=16 | `slow` (N=64) |
| svgd_solver: stop_select_objective | selection = argmin of the chosen objective | from q*, native swarm, N=16 | `slow` (generic start, N=32) |
| svgd_solver: polish | polish record, feasibility | from q*, native swarm, N=16, fielded patience | `slow` (original) |
| svgd_solver: benchmark_subprocess | `panda_benchmark.py --solver svgd` record | N=8, 2 outer steps | `slow` (20 s clock) |
| svgd_solver: sweep | pose configs x arms go/no-go | none | `slow`, `sweep` |
| svgd_analytic: cells_eager_both_kernels, native_start | record fields, split step, counts | N=16, 2 outer steps | `slow` (20 s clock) |
| svgd_analytic: compiled_and_graphed | compiled step + template reuse | **none** (CPU inductor compile) | `slow` |
| svgd_soft_fastpath: split vs autograd step | split step = eager autograd within the 1-ulp floor | 5 steps (floor taken at the same step) | `slow` (30 steps), `quick` |
| svgd_soft_fastpath: compiled_and_graphed | compiled / graphed soft step | none (needed CUDA anyway) | `slow` |
| svgd_pin_and_gate: (b) gate-edge rule | a failed re-check continues and tightens | from q*, N=16, same four asserts | `slow` (generic start, N=64) |
| svgd_pin_and_gate: (a) old pin, (c) | the pin, bitwise | unchanged (skips off its capture GPU) | - |
| svgd_result: duck_type | harness contract on an svgd result | N=8, 2 outer steps | `slow` (5 s clock, N=64) |
| gvs_arm_model: rest_and_axial, drake_matches_model, spheres | per-rung physics, SDF = model, sphere superset | primary rung o1 (spheres 8 + 5 corners) | `slow` (3 rungs, 30 + 5) |
| gvs_arm_model: taper trap | taper keeps higher strain orders live | o1 vs its uniform twin | `slow` (o1 and o2) |
| gvs_arm_model: equilibrium uniqueness | Newton converges, unique | 200 draws, 5 restarts | `slow` (2000, 50) |
| solver_plumbing: each solver solves and reports (x4) | log parsing, counts, diagnostics per solver | `max_iter=5` (IPOPT / SNOPT majors / NLopt max_eval); svgd N=8, 2 outer steps | `slow` (20 s clock) |
| batched_flow: float32 private copy | float32 copy cached, bounded vs float64 | 32 particles | `slow` (256: iiwa runaway particles reliably drawn only here) |
| resume_e2e: everything | a killed and resumed benchmark = uninterrupted run | none; points at test_cell_resume + test_run_items_watcher | `slow` (GPU), `cluster/tests_slow_job.sh` |

Fixture sharing without a coverage change: `solver_plumbing`'s target is memoised (it was rebuilt
~12 times), `svgd_result`'s too, and `gvs_arm_model`'s per-rung Drake diagram is cached.

Mutation spot-checks (`src/` mutated, the fast form run, `src/` restored): stopping the re-check
failure count fails pin_and_gate's fast (b); reading SNOPT's exit from the SNMEMA pass fails
solver_plumbing's capped SNOPT cell; misrecording `polish_viol_after` fails svgd_solver's fast
polish.

## The failures on main

Four files fail on `main` on one CPU core. Two are CPU v GPU, one is a regression on main, and
one is a test bug. Nothing was loosened. The fixes below are proposals for Thomas.

1. **`test_batched_program`, iiwa/mug learned VJP, `atol=1e-12` (l.549): a tolerance that was
   never portable to CPU.**
   - On the failing particle the error is 1.42e-12 on CPU and 8.5e-14 on the GPU, against
     `|R| |J_q|` = 130, i.e. 1.1e-14 relative on CPU.
   - `vjp_q` (one backward of `(q R).sum()`) and `R @ jacobian_q` (ndof backwards, then a
     matmul) are two summation orders of the same product, and the CPU kernels round
     differently.
   - `src/svgd/batched_program.py` is unchanged since the assert was written (bb11ccc) apart
     from two unrelated commits. The file passes on the GPU.
   - Proposal: `atol = 1e-14 * max(1, (|R| @ |J_q|).max())`, the scale-relative form the rest of
     that test already uses.
2. **`test_soft_arm_program`, "learned/IK: gradient off by 3.03e-3": not a derivative error.**
   - The untrained flow (LeakyReLU subnets) has a kink 5e-7 from the test point along `c_z`.
   - One-sided differences: forward -0.168624 = the analytic derivative; backward -0.1747 at
     h = 1e-6, -0.1686 at h <= 3e-7.
   - The h = 1e-6 central difference straddles the kink. The error is flat for h in 1e-3..1e-6
     and vanishes at h <= 1e-7.
   - On the GPU the start point differs by ~1e-7 and the file passes (3.4e-10 relative).
   - Proposal: assert agreement with one of the two one-sided differences, or move the point
     off the kink. Not a loosening: the tolerance stays 1e-6.
3. **`test_svgd_solver`, two causes.**
   - (a) **CPU v GPU seconds.** `run_cell`'s `wall_time <= cap + 1` on the 3 s clock cell: an
     N=64 outer step is 2.7-3.1 s on one CPU core. It failed on 1 of 4 CPU runs (5.5 s, when a
     second outer step started before the deadline), and the other three stopped after one step
     at 3.0-3.3 s. A fixed 1 s margin is not portable to a step longer than the margin; the
     test's own step-scaled bounds pass on every run. The fast tier runs the cell at N=16.
     On `main` the default run's rc 1 is otherwise (b).
   - (b) **A test bug on main, on every device.** `test_polish` asserts "converges", but
     `run_cell` accumulates overrides on the shared pose program, and the earlier
     `test_wall_clock_stop` and `test_stop_reasons_and_log` leave `svgd_stop_patience = 10**6`.
     No stall stop is then possible, and the solve runs to its 30 s clock.
   - Evidence: in the default order it fails on the GPU too, stopping on `wall_clock`. Run
     alone (`... polish`) on the GPU it passes in 8 s, stopping `converged`.
   - The fast form names the fielded patience. Proposal for the original: the same one-word
     fix (`svgd_stop_patience=ProgramOptions().svgd_stop_patience` in its overrides).
4. **`test_report_svgd`: a regression on main (656238d, the R31B manifest).**
   - `report_svgd.stage_catalogue()` refuses: "soft12 pol runs different settings or arms in
     SVGD_R31 and SVGD_R31B".
   - R31B re-runs R31's `pol` column with the same settings and arms, and the catalogue keys
     variants by (robot, column), so the same column in two stages trips the
     manifest-equality assert.
   - This is a reporting-script bug, not a test problem: `scripts/report_svgd.py` refuses
     the current manifests. Left for Thomas (svgd-polish's area).
