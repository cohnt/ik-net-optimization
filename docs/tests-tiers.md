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
| svgd_solver | 221 (rc 1) | 76 | 277 | 101 |
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
| batched_program | 28 (rc 1, aborts at check 2) | 50 (runs to the end) | no slow | fast 64 |
| cvae_chart | 27 | 25 | no slow | |
| 21 other files | 3-24 each | unchanged | no slow | flow_cuda_graph fast 26, soft_arm_program fast 6 (passes) |
| report_svgd | 2 (rc 1, aborts) | 96 (runs to the end) | no slow | |
| **all 36 files** | **1261** | **579** as measured; ~700 now that batched_program and report_svgd run to the end | | |

The `slow` CPU seconds are each file's slow group alone. `svgd_solver`'s slow figures are after
the `test_polish` fix (below). The cold-cache figures are the
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

## The failures on main, and their fixes (Thomas: "Fix all 4 tests", 2026-10-27)

Four files failed on `main` on one CPU core. All four pass now, on one CPU core and on the GPU.
Two were CPU v GPU, one was a test bug plus a seconds margin, and one was a regression in a
reporting script.

1. **`test_batched_program`, iiwa/mug learned VJP, `atol=1e-12`: never portable to CPU.**
   - `vjp_q` (one backward of `(q R).sum()`) and `R @ jacobian_q` (ndof backwards, then a
     matmul) are two summation orders of one product.
   - Measured over every particle, relative to `max(|R| @ |J_q|)`: <= 1.15e-14 on one CPU core,
     <= 1.1e-15 on the GPU. The worst CPU particle was 1.42e-12 absolute (iiwa grasp, scale 130).
   - **Fix:** `atol = 1e-13 * max(1, max(|R| @ |J_q|))`, 9x over the worst CPU particle.
2. **`test_soft_arm_program`, "learned/IK: gradient off by 3.03e-3": not a derivative error.**
   - The untrained flow (LeakyReLU subnets) has a kink 5e-7 from the CPU start along `c_z`.
     The h = 1e-6 central difference straddles it.
   - The one-sided differences at 1e-8 match the derivative to <= 5e-7. On the GPU the start
     moves ~1e-7 and misses the kink.
   - **Fix:** where the central difference disagrees on a column, the column passes only if the
     derivative equals a one-sided difference at step / 100 to the same 1e-6. Each such column is
     printed as a note.
   - Mutation: scaling the learned chain rule by 1.001 still fails it (7.96e-3).
3. **`test_svgd_solver`, two causes.**
   - (a) **`run_cell`'s `wall_time <= cap + 1`.** The clock is read between outer steps, and an
     N=64 outer step is 2.7-3.1 s on one CPU core. The bound failed on 1 of 4 CPU runs (5.5 s,
     when a second step started before the deadline). **Fix:** `cap + max(1 s, 1.5 outer
     steps)`, the granularity `test_wall_clock_stop` already bounds the swarm by.
   - (b) **`test_polish` failed on every device in the default order.** `run_cell` accumulates
     overrides on the shared pose program, and the wall-clock and stop-reason tests leave
     `svgd_stop_patience = 10**6`, so no stall stop was possible. Run alone, it passed on the GPU
     in 8 s. **Fix:** the test names the fielded patience.
   - After both fixes the slow tier passes on the GPU (101 s) and on one CPU core (277 s), with
     both polish solves stopping `converged`.
4. **`test_report_svgd`: a regression from the R31B manifest commit (656238d).**
   - `report_svgd.stage_catalogue()` refused "soft12 pol runs different settings or arms in
     SVGD_R31 and SVGD_R31B". R31B re-runs R31's `pol` / `polkq` columns with exactly R31's
     tokens, so the column appeared in two stages. Before the assert existed the report would
     have picked between them by dict order.
   - **Fix:** `scripts/report_svgd.py` declares `STAGE_REMEASURES = {"SVGD_R31B": "SVGD_R31"}`.
     On every (robot, row, start, column) a re-measure runs, its run replaces the earlier
     stage's, provided the settings and arms are identical; the column reads the later manifest.
     Any other column built by two stages is still refused.
   - Result: soft12 `pol` and the screw / soft12 `polkq` read R31B; the Panda and iiwa read R31.
   - The test's expectations name R31B on that scope, and check one run per re-measured row
     and column.
