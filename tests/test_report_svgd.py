"""Guards for `scripts/report_svgd.py`, on synthetic summaries -- no solver, no GPU.

    .venv/bin/python tests/test_report_svgd.py        (or under pytest)

The fixtures are written by `src.benchmark.summarise` itself, so their schema is the harness's,
not a copy of it. What is pinned:

  - a grid_hash mismatch REFUSES the pairing (nonzero exit), never prints a McNemar row;
  - McNemar directions: learned-only / joint-space-only within a variant, and svgd-only /
    IPOPT-only per arm against the twin;
  - cost is the median over cells BOTH arms solved, and N/A under 10 such cells;
  - solver_feasible vs drake_feasible, and drake_feasible vs verify(), print as BUG lines;
  - the two population metrics (feasible particles, their q-spread) are medians over cells;
  - the A/B section pairs a column against the primary al64 per arm;
  - a missing tag is reported and not fatal.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts", "svgd")):
    if _p not in sys.path:
        sys.path.append(_p)

import report_svgd as R                                             # noqa: E402
import smoke as M                                                   # noqa: E402
from src import benchmark as bm                                     # noqa: E402

T, G = 4, 4                       # a 16-cell grid
N_CELLS, CAP = T * G, 20.0
ROW = ("panda", "mugshelf", "paired")
GRID = "abc123def456"
CKPT = "models/panda/panda__n6__step620000.pkl"


def _meta(grid=GRID, solver="svgd"):
    return dict(task="mug", solver=solver, config="latent", wall_time=CAP, seed=1, grid_hash=grid,
                robot="panda", scene="panda_finray_collision_hardened.yaml", scene_mode="hardened",
                target_placement="shelf", shelf_inset=0.1, start="paired", checkpoint=CKPT,
                n_targets=T, n_guesses=G, svgd_warmup_seconds={"learned": 12.0, "numerical": 3.0})


def _rec(i, ok, cost, svgd=True, solver_ok=None, drake_ok=None):
    r = dict(target=i // G, guess=i % G, feasible=ok, fail_reason="" if ok else "constraint",
             wall_time=2.0, iterations=40, timed_out=False, hit_iteration_cap=False,
             max_violation=1e-9 if ok else 1e-2, solver_success=ok)
    if ok:
        r["cost"] = cost
    if svgd:
        r["svgd"] = dict(method="al_svgd", n_particles=64, iterations=40, n_feasible=8 if ok else 0,
                         n_resampled=2, selected_index=0 if ok else 3,
                         solver_feasible=ok if solver_ok is None else solver_ok,
                         drake_feasible=ok if drake_ok is None else drake_ok,
                         phase_times={"swarm": 1.0}, collision_seconds=0.5,
                         feasible_q_spread=(0.1 * (i % 3 + 1)) if ok else None,
                         n_dual_updates=4, lam_inf_median=2.0, lam_inf_max=50.0 if ok else 1e4,
                         mu_inf_median=0.0, mu_inf_max=3.0, n_multiplier_clipped=0 if ok else 2,
                         stop_reason="converged" if ok else "wall_clock")
    return r


def _write(root, column, learned, numerical, meta):
    tag = M.make_tag(*ROW[:2], ROW[2], column, N_CELLS, CAP)
    d = os.path.join(root, "panda", "benchmark", tag)
    os.makedirs(d, exist_ok=True)
    records = {"learned": learned, "numerical": numerical}
    arms = [bm.Arm("learned", None), bm.Arm("numerical", None)]
    payload = dict(metadata=meta, n_targets=T, n_guesses=G,
                   summary=bm.summarise(records, arms, T, G), records=records)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(payload, f, default=bm._json_default)
    return tag


def _twin(root, grid=GRID):
    ## IPOPT twin: learned solves cells 0..9, joint space 0..4 -> 5 mutual cells (cost N/A).
    learned = [_rec(i, i < 10, 1.5, svgd=False) for i in range(N_CELLS)]
    numerical = [_rec(i, i < 5, 2.5, svgd=False) for i in range(N_CELLS)]
    return _write(root, M.IPOPT, learned, numerical, _meta(grid, "ipopt"))


def _run(root, extra=()):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rep = R.main(["--root", root, "--rows", ",".join(ROW), "--cells", str(N_CELLS),
                      "--cap", str(int(CAP))] + list(extra))
    return rep, out.getvalue()


def test_mcnemar_directions_cost_and_bugs():
    with tempfile.TemporaryDirectory() as root:
        _twin(root)
        ## svgd al64: learned solves 0..13 (cost 1 on 0..9, 1000 on its own 10..13), joint
        ## space solves 0..9 and 14 (cost 2 on 0..9, 1000 on 14). Mutual = 0..9, ten cells.
        learned = [_rec(i, i < 14, 1.0 if i < 10 else 1000.0) for i in range(N_CELLS)]
        numerical = [_rec(i, i < 10 or i == 14, 2.0 if i < 10 else 1000.0) for i in range(N_CELLS)]
        ## Two seeded inconsistencies on learned cell 15 (a failure): the solver's rows said
        ## feasible and Drake did not; on joint-space cell 15, Drake said feasible, verify did not.
        learned[15] = _rec(15, False, None, solver_ok=True, drake_ok=False)
        numerical[15] = _rec(15, False, None, solver_ok=True, drake_ok=True)
        _write(root, "al64", learned, numerical, _meta())
        ## al64none: learned solves 0..7 only -> 6 cells only al64 solved, 0 only al64none.
        _write(root, "al64none", [_rec(i, i < 8, 1.0) for i in range(N_CELLS)],
               [_rec(i, i < 10, 2.0) for i in range(N_CELLS)], _meta())
        rep, text = _run(root, ["--columns", "al64,al64none,al64cem"])

        b = rep.blocks[ROW + ("al64",)]
        assert (b["lj"]["a_only"], b["lj"]["b_only"]) == (4, 1), b["lj"]
        assert "L-only 4 / JS-only 1" in text
        ## Per arm against the IPOPT twin: (svgd-only, IPOPT-only).
        assert (b["vs_twin"]["learned"]["a_only"], b["vs_twin"]["learned"]["b_only"]) == (4, 0)
        assert (b["vs_twin"]["numerical"]["a_only"], b["vs_twin"]["numerical"]["b_only"]) == (6, 0)
        ## Against the record-less root the record column is absent, not invented.
        assert b["vs_rec"] == {}
        ## Cost on mutual cells only: the 1000s on single-arm cells must not move the medians.
        assert b["L"]["n_both"] == 10 and b["L"]["cost"] == 1.0 and b["J"]["cost"] == 2.0
        ## The twin shares 5 solved cells: cost N/A, printed as such.
        tb = rep.blocks[ROW + (M.IPOPT,)]
        assert tb["L"]["n_both"] == 5 and tb["L"]["cost"] is None
        assert "N/A" in text
        ## Both inconsistencies are BUG lines, and the exit status says so.
        assert any("solver_feasible != drake_feasible" in m for m in rep.bugs), rep.bugs
        assert any("drake_feasible != verify()" in m for m in rep.bugs), rep.bugs
        assert "!!!!!! BUG" in text and rep.status == 3
        ## The population metrics: medians over the cells that recorded them.
        assert b["L"]["n_feasible"] == 8.0 and abs(b["L"]["q_spread"] - 0.2) < 1e-12, b["L"]
        assert b["L"]["lam_max"] == 1e4 and b["L"]["mclip"] == 4 and "q-spread among them" in text
        assert "dual updates (median)" in text and "multiplier clips 4" in text
        ## A/B against the primary: al64none has 6 learned cells only al64 solved.
        assert "A/B against the primary al64" in text
        line = next(l for l in text.splitlines() if "al64none" in l and "learned" in l
                    and "grasp" in l)
        f = line.split()
        assert f[f.index("learned") + 1:f.index("learned") + 5] == ["14", "8", "0", "6"], line
        ## The CEM column was asked for and is not on disk: reported, not fatal.
        cem_tag = M.make_tag(*ROW[:2], ROW[2], "al64cem", N_CELLS, CAP)
        assert cem_tag in rep.missing and cem_tag in text


def test_grid_hash_refusal():
    with tempfile.TemporaryDirectory() as root:
        _twin(root)
        learned = [_rec(i, i < 12, 1.0) for i in range(N_CELLS)]
        numerical = [_rec(i, i < 12, 2.0) for i in range(N_CELLS)]
        _write(root, "al64none", learned, numerical, _meta(grid="ffffffffffff"))
        rep, text = _run(root, ["--columns", "al64none"])
        b = rep.blocks[ROW + ("al64none",)]
        assert b["vs_twin"] == {}, "a mismatched grid must not be paired"
        assert any("grid_hash ffffffffffff != " + GRID in m for m in rep.refused), rep.refused
        assert "REFUSED" in text and rep.status == 2 and not rep.bugs


def test_metadata_mismatch_refused_even_on_equal_hash():
    ## The Panda's grasp and pose grids hash identically, so task must be checked separately.
    with tempfile.TemporaryDirectory() as root:
        _twin(root)
        meta = dict(_meta(), task="pose")
        _write(root, "al256", [_rec(i, True, 1.0) for i in range(N_CELLS)],
               [_rec(i, True, 1.0) for i in range(N_CELLS)], meta)
        rep, _ = _run(root, ["--columns", "al256"])
        assert any("task" in m for m in rep.refused) and rep.status == 2


def test_everything_missing_is_not_fatal():
    with tempfile.TemporaryDirectory() as root:
        rep, text = _run(root)
        ## Every column, the twin included, is on the missing list; nothing raised.
        assert len(rep.missing) == len(M.ALL_COLUMNS), rep.missing
        assert "missing runs: %d" % len(M.ALL_COLUMNS) in text
        assert rep.status == 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
