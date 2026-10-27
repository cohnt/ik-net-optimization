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

And the stage mode (`--stage SVGD`, the cluster's `sc_SVGD_*` family):

  - every tag cluster/gen_manifest.py builds parses back to its item, for every variant it names;
  - the LEAD pairing is learned-under-svgd against learned-under-IPOPT (the R2 twin), the joint-space
    ablation its own pairing, and the A/B and variant-vs-variant tables pair against `kq`;
  - a missing run prints as MISSING with its tag and is not fatal;
  - a grid_hash mismatch, and a run whose overrides are not its manifest item's, REFUSE (exit 2);
  - the rounds run the learned arm only (gen_manifest's `--arms learned`): such a run pairs its
    LEAD, prints its ablation as `not run`, takes cost against the twin's learned arm, and a run
    whose arms are not its manifest item's is REFUSED;
  - the ROBOT dimension: the iiwa round R12 is its own block after the Panda's, its LEAD paired
    against stage PAPER's iiwa IPOPT run (an `sc_PAPER_` tag) with the same provenance check as
    R2, its ablation `not run`; with nothing on disk it is MISSING and not fatal;
  - the SEED dimension: R7 (grid seed 2) is the column `kq@seed2` beside `kq` in the LEAD table,
    paired against its own twin `ipopt@seed2` (SVGD_R7IPOPT) and never against R2's; a run at
    the wrong seed is REFUSED as not the run its tag names, and a twin on another grid is refused;
  - R13's kernel variants carry exactly their one override and R14's exactly their declared
    factors; the screw arm's R15 and the soft PCS arm's R16 are their own blocks, each paired
    against stage PAPER's IPOPT run on that robot, and `--rows` accepts those robots;
  - R17 / R17S carry the Panda's `kcz` / `kczrho1e5` settings to the iiwa and the screw arm as
    columns of those robots' blocks, and R18's three-factor stacks are Panda columns, the
    `initnative` one on the paired rows only;
  - R19 (the candidate `kczrho1e5` on R7's seed-2 grids) is the Panda column `kczrho1e5@seed2`,
    R14's settings exactly, paired with R7IPOPT's twin `ipopt@seed2` and never with a seed-1 run,
    and `--rows panda --variants kq,kczrho1e5` resolves it as `kq@seed2` resolves; R17P carries
    `kcz` / `kczrho1e5` to the soft PCS arm;
  - R20 (the record rows at the fielded pin, SVGD_PINNED_V2) is ONE stage across four robots: `pin2`
    is a column of every robot's block with the same settings, each paired with that robot's twin,
    and `pin2kq` (the joint-space ablation, `--arms numerical`) a Panda column whose LEAD is `not run`
    and whose ABLATION pairs against the twin's joint-space arm.
The stage fixtures keep gen_manifest's real tags but are 16-cell runs: the catalogue's cell count
is patched to the fixture's, which is the one thing the 480-cell tag cannot carry.
Tiers (tests/_tiers.py): the default `fast` runs in ~95 s on one CPU core; no `slow` group.
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts"), os.path.join(REPO, "scripts", "svgd"),
           os.path.join(REPO, "cluster")):
    if _p not in sys.path:
        sys.path.append(_p)

import gen_manifest as GM                                          # noqa: E402
import report_svgd as R                                             # noqa: E402
import smoke as M                                                   # noqa: E402
from src import benchmark as bm                                     # noqa: E402
import _tiers                                                       # noqa: E402

## `stage_catalogue` is a pure function of gen_manifest's code (nothing on disk), and every stage
## check reads it, directly or through `R.main` -- ~5 s per build. Built once per process; each
## caller gets a deep copy, so a check that edits what it is handed cannot leak into the next.
_CATALOGUES, _build_catalogue = {}, R.stage_catalogue


def _catalogue(family="SVGD"):
    if family not in _CATALOGUES:
        _CATALOGUES[family] = _build_catalogue(family)
    return copy.deepcopy(_CATALOGUES[family])


R.stage_catalogue = _catalogue

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


def test_multiplier_clips_by_group_line():
    """Records carrying `multiplier_clips_by_group` print one summed line per arm, zeros
    kept; an arm whose records predate the key prints no such line."""
    with tempfile.TemporaryDirectory() as root:
        _twin(root)
        learned = [_rec(i, i < 10, 1.0) for i in range(N_CELLS)]
        for i, r in enumerate(learned):
            r["svgd"]["multiplier_clips_by_group"] = {"pose_pos": 0, "trust": i % 2, "c_box_hi": 1}
        numerical = [_rec(i, i < 10, 2.0) for i in range(N_CELLS)]
        _write(root, "al64", learned, numerical, _meta())
        _, text = _run(root, ["--columns", "al64"])
        lines = [l for l in text.splitlines() if "multiplier clips by row group" in l]
        assert lines == [f"    {'':<17} multiplier clips by row group  pose_pos 0  trust "
                         f"{N_CELLS // 2}  c_box_hi {N_CELLS}"], lines


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


## ------------------------------------------------------------------ stage mode --
STAGE_ROW = ("panda", "mugshelf", "paired")
STAGE_FP = "64dbd94dc2c2f4007d89fc3bfe3f8777e0be272f"


def _stage_tags(robot="panda"):
    """(runs, `robot`'s columns, (robot, row, start, column) -> tag)."""
    runs, variants = R.stage_catalogue()
    by = {(t["robot"], t["row"], t["start"], t["column"]): tag for tag, t in runs.items()}
    return runs, variants[robot], by


CKPTS = {robot: ck for (robot, _), (ck, _) in GM.SVGD_RUNGS.items()}


def _stage_meta(info, grid=GRID, overrides=None, seed=None):
    return dict(task=M.TASK_ROWS[info["row"]][0], solver=info["solver"], config="latent",
                wall_time=info["cap"], seed=info["seed"] if seed is None else seed,
                grid_hash=grid, robot=info["robot"],
                scene=f"{info['robot']}_collision_hardened.yaml", scene_mode="hardened",
                scene_fingerprint=STAGE_FP, target_placement="shelf", shelf_inset=0.1,
                start=info["start"], checkpoint=CKPTS[info["robot"]], n_targets=T, n_guesses=G,
                overrides=info["sets"] if overrides is None else overrides)


def _stage_write(root, variant, learned, numerical, grid=GRID, overrides=None, row=None,
                 seed=None):
    """`numerical=None` writes a learned-only run, as a round's `--arms learned` item does, and
    `learned=None` a joint-space-only one, as R20's `pin2kq` (`--arms numerical`) does.
    `variant` is the report's column (`kq`, `kq@seed2`, `ipopt@seed2`); `row` defaults to
    STAGE_ROW; `seed` overrides the metadata's (default: the manifest item's)."""
    row = STAGE_ROW if row is None else row
    runs, _, by = _stage_tags(row[0])
    tag = by[row + (variant,)]
    d = os.path.join(root, row[0], "benchmark", tag)
    os.makedirs(d, exist_ok=True)
    records = {a: r for a, r in (("learned", learned), ("numerical", numerical)) if r is not None}
    arms = [bm.Arm(a, None) for a in records]
    payload = dict(metadata=_stage_meta(runs[tag], grid, overrides, seed), n_targets=T,
                   n_guesses=G, summary=bm.summarise(records, arms, T, G), records=records)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(payload, f, default=bm._json_default)
    return tag


def _stage_run(root, extra=(), row=None):
    real = R.stage_catalogue

    def small(family="SVGD"):
        runs, variants = real(family)
        return {k: dict(v, cells=N_CELLS) for k, v in runs.items()}, variants

    out = io.StringIO()
    R.stage_catalogue = small
    try:
        with contextlib.redirect_stdout(out):
            rep = R.main(["--stage", "SVGD", "--root", root, "--rows",
                          ",".join(STAGE_ROW if row is None else row)] + list(extra))
    finally:
        R.stage_catalogue = real
    return rep, out.getvalue()


def _stage_twin(root, grid=GRID, column=R.STAGE_TWIN, row=None, overrides=None):
    ## IPOPT twin: learned solves 0..9, joint space 0..9.
    return _stage_write(root, column, [_rec(i, i < 10, 1.5, svgd=False) for i in range(N_CELLS)],
                        [_rec(i, i < 10, 2.5, svgd=False) for i in range(N_CELLS)], grid,
                        overrides=overrides, row=row)


def test_stage_tag_parsing_every_variant():
    runs, variants, _ = _stage_tags()
    ## The Panda's seed-1 columns: the twin, R1's and every variant a round runs on the Panda at
    ## seed 1 (a round's one context, or -- R20 -- the variant's own scope among several).
    P1 = ("panda", "n6", 1)
    panda1 = [r for r in GM.SVGD_ROUNDS if any(P1 in GM.svgd_variant_contexts(r, v)
                                               for v in GM.SVGD_ROUNDS[r])]
    named = ({R.STAGE_TWIN} | set(GM.SVGD_VARIANTS)
             | {v for r in panda1 for v in GM.SVGD_ROUNDS[r] if P1 in GM.svgd_variant_contexts(r, v)})
    assert {c for c in variants if not R.seed_suffix(c)} == named, (sorted(variants), sorted(named))
    assert next(iter(variants)) == R.STAGE_TWIN and list(variants)[1] == R.stage_primary() == "kq"
    seen = set()
    for stage in GM.SVGD_STAGES:
        if stage in GM.SVGD_SMOKES:
            continue
        for it in GM.stage_SVGD(stage):
            a = it["args"]
            tag = a[a.index("--tag") + 1]
            t = R.parse_stage_tag(tag)
            assert t["solver"] == a[a.index("--solver") + 1] and t["start"] == a[a.index("--start") + 1]
            assert M.TASK_ROWS[t["row"]][0] == a[a.index("--task") + 1]
            ## Round trip: the fields rebuild the tag exactly.
            rebuilt = "_".join(["sc", "SVGD", t["round"], t["robot"], t["rung"], t["solver"], t["row"],
                                str(t["cells"]), f"{t['cap']:g}", t["start"]]
                               + ([] if t["variant"] == R.STAGE_TWIN else [t["variant"]]))
            assert rebuilt == tag, (rebuilt, tag)
            if tag not in runs:
                ## Superseded: a declared re-measure (STAGE_REMEASURES) ran this row and column.
                later = [st for st, old in R.STAGE_REMEASURES.items() if old == stage]
                assert later and tag.replace(f"sc_{stage}_", f"sc_{later[0]}_") in runs, tag
                continue
            assert runs[tag]["manifest"] == stage
            if (t["robot"], t["rung"], runs[tag]["seed"]) == ("panda", "n6", 1):
                seen.add((t["variant"], t["row"], t["start"]))
    ## Every variant on the rows its manifest gives it (both starts, or a round variant's own
    ## `starts` filter -- `initnative` runs the paired rows only), and its settings are its manifest's.
    expected = set()
    for v in named:
        starts = M.STARTS
        for r in panda1:
            if v in GM.SVGD_ROUNDS[r]:
                starts = tuple(GM.SVGD_ROUNDS[r][v][2].split(","))
        expected |= {(v, r, s) for r in M.TASK_ROWS for s in starts}
    assert seen == expected, (seen ^ expected)
    assert variants["knone"]["sets"]["svgd_kernel"] == "none"
    assert variants["rho1e4"]["sets"]["svgd_rho"] == 10000 and variants["kq"]["sets"]["svgd_rho"] == 1000
    ## R11: the resample mode, the one change; the primary runs the default (native) unset.
    assert variants["drop"]["sets"]["svgd_resample"] == "drop" and variants["drop"]["round"] == "R11"
    assert variants["keep"]["sets"]["svgd_resample"] == "keep" and "svgd_resample" not in variants["kq"]["sets"]
    assert "max_iter" in variants[R.STAGE_TWIN]["sets"] and "max_iter" not in variants["kq"]["sets"]
    ## The arms are the manifest's: the twin and R1 run both, the rounds what SVGD_ROUND_ARMS says
    ## -- but for a variant that names its own (R20's joint-space ablation, `pin2kq`).
    assert variants[R.STAGE_TWIN]["arms"] == variants["kq"]["arms"] == R.ARMS
    for r in panda1:
        for v in GM.SVGD_ROUNDS[r]:
            if P1 not in GM.svgd_variant_contexts(r, v):
                continue                  # not a Panda seed-1 column (R30's `pin4s2`: seed 2)
            want = tuple(GM.svgd_variant_arms(r, v).split(","))
            assert variants[v]["arms"] == want, (v, variants[v]["arms"])
            assert want == (("numerical",) if (r, v) in (("SVGD_R20", "pin2kq"),
                                                          ("SVGD_R22", "clipoffjs"),
                                                          ("SVGD_R30", "pin4kq"),
                                                          ("SVGD_R31", "polkq"))
                            else tuple(GM.SVGD_ROUND_ARMS.split(","))), (r, v, want)
    ## Not stage tags: the smoke's, the cluster smoke's, the record's, a twin with a variant token.
    for bad in (M.make_tag("panda", "mugshelf", "paired", "al64", 10, 20),
                "sc_SVGDSMOKE_panda_n6_svgd_posetip_4_60_paired_kq",
                "sc_REMEASURE_panda_n6_ipopt_mugshelf_480_180_paired",
                "sc_SVGD_R2_panda_n6_ipopt_mugshelf_480_180_paired_kq",
                "sc_SVGD_R1_panda_n6_svgd_mugshelf_480_180_paired"):
        try:
            R.parse_stage_tag(bad)
        except ValueError:
            continue
        raise AssertionError(f"parsed a non-stage tag: {bad}")


def test_stage_lead_pairing_ab_and_missing():
    with tempfile.TemporaryDirectory() as root:
        _stage_twin(root)
        ## kq: learned solves 0..13 (svgd-only 10..13), joint space 0..4 and 15 (svgd-only 15,
        ## IPOPT-only 5..9).
        _stage_write(root, "kq", [_rec(i, i < 14, 1.0) for i in range(N_CELLS)],
                     [_rec(i, i < 5 or i == 15, 2.0) for i in range(N_CELLS)])
        ## n1 (a round: learned only): learned solves 0..7 only -> 6 cells only kq solved.
        _stage_write(root, "n1", [_rec(i, i < 8, 1.0) for i in range(N_CELLS)], None)
        rep, text = _stage_run(root)
        key = STAGE_ROW
        b = rep.blocks[key + ("kq",)]
        ## The LEAD: learned under svgd against learned under IPOPT, (svgd-only, IPOPT-only).
        assert (b["vs_twin"]["learned"]["a_only"], b["vs_twin"]["learned"]["b_only"]) == (4, 0)
        assert (b["vs_twin"]["numerical"]["a_only"], b["vs_twin"]["numerical"]["b_only"]) == (1, 5)
        lead = text[text.index("LEAD -- learned under svgd"):text.index("ABLATION -- joint space")]
        kq = next(l for l in lead.splitlines() if l.strip().startswith("kq"))
        assert kq.split()[2:7] == ["14", "10", "4", "0", "0.125"], kq
        assert rep.verdicts[(key, "kq", "learned")] == "tie"
        abl = text[text.index("ABLATION -- joint space"):]
        kqa = next(l for l in abl.splitlines() if l.strip().startswith("kq"))
        assert kqa.split()[2:6] == ["6", "10", "1", "5"], kqa
        ## learned-vs-joint-space under svgd is printed and labelled as no headline.
        assert "learned vs joint space under svgd (printed, NOT a headline): 14 v 6" in text
        assert "steps = svgd OUTER steps, NOT comparable to IPOPT majors" in text
        ## Missing runs print with their tags and are not fatal.
        _, _, by = _stage_tags()
        knone = by[key + ("knone",)]
        assert knone in rep.missing and f"MISSING: {knone}" in text
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        ## A/B against the primary kq: n1 has 0 cells only it solved, 6 only kq solved.
        ab = text[text.index("A/B against the primary kq"):text.index("VARIANTS ACROSS ROUNDS")]
        line = next(l for l in ab.splitlines() if " n1 " in l and "learned" in l)
        f = line.split()
        assert f[f.index("learned") + 1:f.index("learned") + 5] == ["14", "8", "0", "6"], line
        ## The overview: the twin's count, then kq's with its verdict letter and W/T/L tally.
        ov = text[text.index("VARIANTS ACROSS ROUNDS"):text.index("VARIANT vs VARIANT")]
        row = next(l for l in ov.splitlines() if l.strip().startswith("kq"))
        assert "14 T" in row and row.rstrip().endswith("0/1/0"), row
        ## Variant vs variant: kq against n1 on the learned arm, +6/-0.
        pw = text[text.index("VARIANT vs VARIANT"):]
        assert "+6/-0" in pw, pw
        ## Mean wall is clamped at the cap in stage mode.
        tw = R.load(root, by[key + (R.STAGE_TWIN,)])
        tw["records"]["learned"][0]["wall_time"] = 1000.0
        st = R.arm_stats(tw, "learned", "numerical", clamp_wall=True)
        assert abs(st["wall"] - (2.0 * 15 + 180.0) / 16) < 1e-12, st["wall"]


def test_stage_learned_only_round():
    with tempfile.TemporaryDirectory() as root:
        _stage_twin(root)                          # learned 0..9 (cost 1.5), joint space 0..9
        _stage_write(root, "kq", [_rec(i, i < 14, 1.0) for i in range(N_CELLS)],
                     [_rec(i, i < 5 or i == 15, 2.0) for i in range(N_CELLS)])
        ## n1, learned arm only: solves 0..11 (cost 3.0) -> svgd-only 10, 11 against the twin, and
        ## 10 cells it and the twin's learned arm both solved (0..9) for the cost column.
        n1 = _stage_write(root, "n1", [_rec(i, i < 12, 3.0) for i in range(N_CELLS)], None)
        rep, text = _stage_run(root)
        key = STAGE_ROW
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        assert n1 not in rep.missing
        ## The LEAD pairs n1's learned arm with the twin's.
        b = rep.blocks[key + ("n1",)]
        assert set(b["vs_twin"]) == {"learned"} and b["J"] is None and b["arms"] == ("learned",)
        lead = text[text.index("LEAD -- learned under svgd"):text.index("ABLATION -- joint space")]
        line = next(l for l in lead.splitlines() if l.strip().startswith("n1"))
        assert line.split()[2:6] == ["12", "10", "2", "0"], line
        assert rep.verdicts[(key, "n1", "learned")] == "tie"
        assert (key, "n1", "numerical") not in rep.verdicts
        ## The ABLATION says not run -- not MISSING, not a row of zeros.
        abl = text[text.index("ABLATION -- joint space"):text.index("[ipopt] round")]
        line = next(l for l in abl.splitlines() if l.strip().startswith("n1"))
        assert line.split()[2:] == ["not", "run", "(learned", "arm", "only)"], line
        ## Its block: the learned row only, no learned-vs-joint-space line, cost against the twin.
        blk = text[text.index(f"[n1] round"):text.index("A/B against the primary")]
        assert "joint space arm not run (learned arm only" in blk, blk
        assert "learned vs joint space" not in blk and "joint space  " not in blk, blk
        assert "cost/tw = median reported cost on the n cells this arm AND the same arm of the " \
               "IPOPT twin both solved" in blk
        row = next(l for l in blk.splitlines() if l.strip().startswith("learned "))
        assert row.split()[:7] == ["learned", "12/16", "+2/-0", "p=0.5", "1.0e-09", "3.000", "10"], row
        assert b["L"]["cost_vs_twin"] == 3.0 and b["L"]["n_vs_twin"] == 10
        ## The kq block keeps both arms and its learned-vs-joint-space line.
        assert "learned vs joint space under svgd (printed, NOT a headline): 14 v 6" in text
        ## A/B: n1's joint-space line is `not run`; its learned line pairs (+0 / -2 against kq).
        ab = text[text.index("A/B against the primary kq"):text.index("VARIANTS ACROSS ROUNDS")]
        assert any(" n1 " in l and "joint space" in l and "not run (learned arm only)" in l
                   for l in ab.splitlines()), ab
        line = next(l for l in ab.splitlines() if " n1 " in l and "learned" in l)
        f = line.split()
        assert f[f.index("learned") + 1:f.index("learned") + 5] == ["14", "12", "0", "2"], line
        ## The overview: n1's learned cell carries its verdict, its joint-space cell is `--`.
        ov = text[text.index("VARIANTS ACROSS ROUNDS"):text.index("VARIANT vs VARIANT")]
        lt = ov[ov.index("learned under svgd"):ov.index("joint space under svgd")]
        jt = ov[ov.index("joint space under svgd"):]
        assert "12 T" in next(l for l in lt.splitlines() if l.strip().startswith("n1"))
        assert next(l for l in jt.splitlines() if l.strip().startswith("n1")).split()[2] == "--"
        ## Pairwise: the learned matrix pairs kq against n1; the joint-space one skips n1.
        pw = text[text.index("VARIANT vs VARIANT"):]
        assert "+2/-0" in pw[:pw.index("joint space under svgd")]
        js = pw[pw.index("joint space under svgd"):]
        assert "1 variant(s) landed -- nothing to pair" in js.splitlines()[0], js


def test_stage_arms_not_the_manifests_are_refused():
    with tempfile.TemporaryDirectory() as root:
        _stage_twin(root)
        ## A round run carrying a joint-space arm its manifest item never ran ...
        n1 = _stage_write(root, "n1", [_rec(i, True, 1.0) for i in range(N_CELLS)],
                          [_rec(i, True, 1.0) for i in range(N_CELLS)])
        ## ... and an R1 run missing the joint-space arm its item did run.
        kq = _stage_write(root, "kq", [_rec(i, True, 1.0) for i in range(N_CELLS)], None)
        rep, _ = _stage_run(root)
        assert any(n1 in m and "carries arm(s) ['numerical']" in m for m in rep.refused), rep.refused
        assert any(kq in m and "joint space has 0 of" in m for m in rep.refused), rep.refused
        assert STAGE_ROW + ("n1",) not in rep.blocks and STAGE_ROW + ("kq",) not in rep.blocks
        assert rep.status == 2


def test_stage_grid_hash_refusal():
    with tempfile.TemporaryDirectory() as root:
        _stage_twin(root)
        _stage_write(root, "kq", [_rec(i, i < 12, 1.0) for i in range(N_CELLS)],
                     [_rec(i, i < 12, 2.0) for i in range(N_CELLS)], grid="ffffffffffff")
        rep, text = _stage_run(root)
        assert rep.blocks[STAGE_ROW + ("kq",)]["vs_twin"] == {}, "a mismatched grid must not pair"
        assert any("grid_hash ffffffffffff != " + GRID in m for m in rep.refused), rep.refused
        assert "REFUSED" in text and rep.status == 2 and not rep.bugs


def test_stage_run_not_what_its_tag_names_is_refused():
    with tempfile.TemporaryDirectory() as root:
        _stage_twin(root)
        _, variants, by = _stage_tags()
        ## The kq tag holding a run made with knone's settings.
        tag = _stage_write(root, "kq", [_rec(i, True, 1.0) for i in range(N_CELLS)],
                           [_rec(i, True, 1.0) for i in range(N_CELLS)],
                           overrides=variants["knone"]["sets"])
        rep, text = _stage_run(root)
        assert any(tag in m and "not the run its tag names" in m for m in rep.refused), rep.refused
        assert STAGE_ROW + ("kq",) not in rep.blocks and rep.status == 2


## ------------------------------------------------------- robot and seed dimensions --
IIWA_ROW = ("iiwa", "mugshelf", "paired")
GRID2 = "fedcba987654"


def test_stage_catalogue_robots_and_seeds():
    runs, _ = R.stage_catalogue()
    _, panda, by = _stage_tags("panda")
    _, iiwa, _ = _stage_tags("iiwa")
    ## The iiwa: its twin (stage PAPER's IPOPT run, the record's rows at paper conditions), the
    ## primary, then R17's kernel-space variants, learned arm only, on the n4 chart at seed 1.
    assert list(iiwa) == [R.STAGE_TWIN, "kq", "kcz", "kczrho1e5", "pin2", "pin2kq",
                          "clipoff", "drop2", "T10v2", "T0p1", "ident", "lm1", "lm100"] + LATER_ROUNDS, list(iiwa)
    assert iiwa[R.STAGE_TWIN]["manifest"] == "PAPER" and iiwa["kq"]["manifest"] == "SVGD_R12"
    ## R21: the joint-space ablation at the pin on the three non-Panda record contexts.
    assert iiwa["pin2kq"]["manifest"] == "SVGD_R21" and iiwa["pin2kq"]["arms"] == ("numerical",)
    assert iiwa["kcz"]["manifest"] == iiwa["kczrho1e5"]["manifest"] == "SVGD_R17"
    assert iiwa["kq"]["arms"] == ("learned",) and iiwa[R.STAGE_TWIN]["arms"] == R.ARMS
    ## R12 is the primary's settings exactly, plus the clearance collision row (an execution
    ## setting: the same functional, batched; gen_manifest's SVGD_CLEARANCE_STAGES).
    CLR = {"svgd_collision_backend": "clearances"}
    assert iiwa["kq"]["sets"] == {**panda["kq"]["sets"], **CLR}, "R12 is the primary + the clearance row"
    twin = by[IIWA_ROW + (R.STAGE_TWIN,)]
    assert twin == "sc_PAPER_iiwa_n4_ipopt_mugshelf_480_180_paired", twin
    assert R.parse_stage_tag(twin)["round"] == "PAPER" and runs[twin]["seed"] == 1
    assert runs[twin]["sets"] == {"correction_cost_weight": 10.0, "flow_cuda_graph": True,
                                  "max_iter": 1000000}
    assert by[IIWA_ROW + ("kq",)] == "sc_SVGD_R12_iiwa_n4_svgd_mugshelf_480_180_paired_kq"
    ## The Panda's second grid: R7's twin and primary, BESIDE the seed-1 pair (columns 3 and 4).
    assert list(panda)[:4] == [R.STAGE_TWIN, "kq", "ipopt@seed2", "kq@seed2"], list(panda)[:4]
    assert panda["ipopt@seed2"]["manifest"] == "SVGD_R7IPOPT" and panda["kq@seed2"]["manifest"] == "SVGD_R7"
    assert panda["ipopt@seed2"]["seed"] == panda["kq@seed2"]["seed"] == 2
    assert panda["ipopt@seed2"]["sets"] == panda[R.STAGE_TWIN]["sets"]
    assert panda["kq@seed2"]["sets"] == {**panda["kq"]["sets"], **CLR} and \
        panda["kq@seed2"]["arms"] == ("learned",)
    ## The ipopt twins never carry it; R11's variants do, and the report names it in their diff.
    assert "svgd_collision_backend" not in panda["ipopt@seed2"]["sets"]
    assert "svgd_collision_backend='clearances'" in R.setting_diff(panda, "drop", "kq")
    assert R.twin_of("kq@seed2") == "ipopt@seed2" and R.twin_of("kq") == R.STAGE_TWIN
    assert by[STAGE_ROW + ("kq@seed2",)] == "sc_SVGD_R7_panda_n6_svgd_mugshelf_480_180_paired_kq"
    assert by[STAGE_ROW + ("ipopt@seed2",)] == "sc_SVGD_R7_panda_n6_ipopt_mugshelf_480_180_paired"
    ## Not stage tags: PAPER is IPOPT only; an unknown family; a PAPER tag with a variant.
    for bad in ("sc_PAPER_iiwa_n4_svgd_mugshelf_480_180_paired_kq",
                "sc_PAPER_iiwa_n4_ipopt_mugshelf_480_180_paired_kq",
                "sc_PAPERX_iiwa_n4_ipopt_mugshelf_480_180_paired"):
        try:
            R.parse_stage_tag(bad)
        except ValueError:
            continue
        raise AssertionError(f"parsed a non-stage tag: {bad}")


def test_stage_iiwa_block_after_the_panda():
    with tempfile.TemporaryDirectory() as root:
        ## Nothing on disk: every iiwa run is MISSING, not fatal, no refusal, no bug.
        rep, text = _stage_run(root, row=("mugshelf", "paired"))
        _, _, by = _stage_tags("iiwa")
        assert by[IIWA_ROW + ("kq",)] in rep.missing and by[IIWA_ROW + (R.STAGE_TWIN,)] in rep.missing
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        assert text.index("robot panda (rung n6)") < text.index("robot iiwa (rung n4)")
        iiwa = text[text.index("robot iiwa (rung n4)"):]
        assert "IPOPT twin (PAPER): sc_PAPER_iiwa_n4_ipopt_mugshelf_480_180_paired" in iiwa
        abl = iiwa[iiwa.index("ABLATION -- joint space"):iiwa.index("[ipopt] round")]
        assert next(l for l in abl.splitlines() if l.strip().startswith("kq")).split()[2:] == \
            ["not", "run", "(learned", "arm", "only)"]
    with tempfile.TemporaryDirectory() as root:
        ## The twin (stage PAPER's run) and R12's learned-only kq: the LEAD pairs them.
        _stage_twin(root, row=IIWA_ROW)
        _stage_write(root, "kq", [_rec(i, i < 13, 1.0) for i in range(N_CELLS)], None, row=IIWA_ROW)
        rep, text = _stage_run(root, row=IIWA_ROW)
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        b = rep.blocks[IIWA_ROW + ("kq",)]
        assert (b["vs_twin"]["learned"]["a_only"], b["vs_twin"]["learned"]["b_only"]) == (3, 0)
        assert set(b["vs_twin"]) == {"learned"} and b["J"] is None
        lead = text[text.index("LEAD -- learned under svgd"):text.index("ABLATION -- joint space")]
        assert next(l for l in lead.splitlines() if l.strip().startswith("kq")).split()[1:6] == \
            ["R12", "13", "10", "3", "0"]
        assert rep.verdicts[(IIWA_ROW, "kq", "learned")] == "tie"
        ## R17's columns have not landed: the A/B against kq prints them as missing, not fatal.
        assert "missing (kcz)" in text and "missing (kczrho1e5)" in text
    with tempfile.TemporaryDirectory() as root:
        ## A PAPER run whose overrides are not the PAPER item's (its budget unlifted) is REFUSED,
        ## exactly as an R2 twin would be -- so R12 has nothing to pair with.
        sets = dict(R.stage_catalogue()[0]["sc_PAPER_iiwa_n4_ipopt_mugshelf_480_180_paired"]["sets"])
        del sets["max_iter"]
        tw = _stage_twin(root, row=IIWA_ROW, overrides=sets)
        _stage_write(root, "kq", [_rec(i, i < 13, 1.0) for i in range(N_CELLS)], None, row=IIWA_ROW)
        rep, text = _stage_run(root, row=IIWA_ROW)
        assert any(tw in m and "not the run its tag names" in m for m in rep.refused), rep.refused
        assert rep.blocks[IIWA_ROW + ("kq",)]["vs_twin"] == {} and rep.status == 2
        assert "not paired: the twin is REFUSED" in text


def test_stage_seed2_pairs_its_own_twin_only():
    with tempfile.TemporaryDirectory() as root:
        ## Seed 1: R2's twin and R1's kq on GRID. Seed 2: R7's twin and kq on GRID2.
        _stage_twin(root)
        _stage_write(root, "kq", [_rec(i, i < 14, 1.0) for i in range(N_CELLS)],
                     [_rec(i, i < 10, 2.0) for i in range(N_CELLS)])
        _stage_twin(root, grid=GRID2, column="ipopt@seed2")
        _stage_write(root, "kq@seed2", [_rec(i, i < 12, 1.0) for i in range(N_CELLS)], None,
                     grid=GRID2)
        rep, text = _stage_run(root)
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        assert rep.blocks[STAGE_ROW + ("kq@seed2",)]["vs_twin"]["learned"]["a_only"] == 2
        assert rep.verdicts[(STAGE_ROW, "kq@seed2", "learned")] == "tie"
        ## Beside kq in the LEAD table, its own round, paired against the seed-2 twin.
        lead = text[text.index("LEAD -- learned under svgd"):text.index("ABLATION -- joint space")]
        lines = [l.split() for l in lead.splitlines() if l.strip().startswith("kq")]
        assert [l[0] for l in lines[:2]] == ["kq", "kq@seed2"], lines
        assert lines[1][1:6] == ["R7", "12", "10", "2", "0"], lines[1]
        assert "IPOPT twin, grid seed 2 (SVGD_R7IPOPT): sc_SVGD_R7_panda_n6_ipopt_mugshelf_480_180_paired" in text
        ## The A/B against the seed-1 primary does not pair it; the overview prints it beside kq.
        ab = text[text.index("A/B against the primary kq"):text.index("VARIANTS ACROSS ROUNDS")]
        assert "kq@seed2: not paired with the primary -- grid seed 2" in ab
        assert not any("kq@seed2" in l and "learned" in l.split() for l in ab.splitlines())
        ov = text[text.index("VARIANTS ACROSS ROUNDS"):text.index("VARIANT vs VARIANT")]
        rows = [l.split()[0] for l in ov[:ov.index("joint space under svgd")].splitlines()
                if l.strip().startswith(("kq", "ipopt"))]
        assert rows[:4] == ["ipopt", "kq", "ipopt@seed2", "kq@seed2"], rows
    with tempfile.TemporaryDirectory() as root:
        ## R7's run carrying the seed-1 grid (R2's hash) is refused against the seed-2 twin, and
        ## an R7 run made at seed 1 is not the run its tag names.
        _stage_twin(root, grid=GRID2, column="ipopt@seed2")
        _stage_write(root, "kq@seed2", [_rec(i, i < 12, 1.0) for i in range(N_CELLS)], None,
                     grid=GRID)
        rep, _ = _stage_run(root)
        assert any("grid_hash " + GRID + " != " + GRID2 in m for m in rep.refused), rep.refused
        assert rep.blocks[STAGE_ROW + ("kq@seed2",)]["vs_twin"] == {} and rep.status == 2
    with tempfile.TemporaryDirectory() as root:
        _stage_twin(root, grid=GRID2, column="ipopt@seed2")
        tag = _stage_write(root, "kq@seed2", [_rec(i, i < 12, 1.0) for i in range(N_CELLS)], None,
                           grid=GRID2, seed=1)
        rep, _ = _stage_run(root)
        assert any(tag in m and "seed 1 != the manifest's 2" in m for m in rep.refused), rep.refused
        assert STAGE_ROW + ("kq@seed2",) not in rep.blocks and rep.status == 2

## The tuning push's rounds after R23 (R24 / R26 / R27 / R28), in catalogue order, on every record robot.
LATER_ROUNDS = ['N16', 'N32', 'rho1e6', 'pat2', 'pat3', 'alpha0p1', 'alpha10', 'alpha0', 'floor1em3', 'floor0p2', 'kz3', 'lp0p1', 'lp1', 'lp10', 'lp1notr', 'ld1', 'fd0p1', 'fd1', 'qc0', 'qc1', 'qc100', 'pin4', 'so', 'lp0p1so', 'lp1so', 'pin4kq', 'pin4s2@seed2', 'pol', 'polkq']
LATER_ROUND_OF = {**{v: "SVGD_R24" for v in LATER_ROUNDS[:11]}, **{v: "SVGD_R26" for v in LATER_ROUNDS[11:15]},
                  **{v: "SVGD_R27" for v in LATER_ROUNDS[15:18]}, **{v: "SVGD_R28" for v in LATER_ROUNDS[18:21]},
                  **{v: "SVGD_R29" for v in LATER_ROUNDS[21:25]}, **{v: "SVGD_R30" for v in LATER_ROUNDS[25:27]},
                  **{v: "SVGD_R31" for v in LATER_ROUNDS[27:]}}


## ------------------------------------------- R13 / R14 and the screw / soft PCS arms --
OTHER_ROBOTS = {"screw7_p050": ("n6", "SVGD_R15"), "soft12": ("n6", "SVGD_R16")}
## The rounds after the primary on each of those robots: R17S's / R17P's kernel-space variants.
OTHER_ROBOT_ROUNDS = {"screw7_p050": {"kcz": "SVGD_R17S", "kczrho1e5": "SVGD_R17S",
                                      "pin2": "SVGD_R20", "pin2kq": "SVGD_R21",
                                      "clipoff": "SVGD_R22", "drop2": "SVGD_R22",
                                      "T10v2": "SVGD_R22", "T0p1": "SVGD_R22",
                                      "ident": "SVGD_R23", "lm1": "SVGD_R23", "lm100": "SVGD_R23",
                                      **LATER_ROUND_OF, "polkq": "SVGD_R31B"},
                      "soft12": {"kcz": "SVGD_R17P", "kczrho1e5": "SVGD_R17P", "pin2": "SVGD_R20",
                                 "pin2kq": "SVGD_R21",
                                 "clipoff": "SVGD_R22", "drop2": "SVGD_R22",
                                 "T10v2": "SVGD_R22", "T0p1": "SVGD_R22",
                                 "ident": "SVGD_R23", "lm1": "SVGD_R23", "lm100": "SVGD_R23",
                                 **LATER_ROUND_OF, "pol": "SVGD_R31B", "polkq": "SVGD_R31B"}}
## R31B re-measures R31's `pol` on soft12 and `polkq` on the screw and soft12 contexts with exactly
## R31's tokens (report_svgd.STAGE_REMEASURES): there its runs replace R31's and the column reads R31B.


def test_stage_kernel_and_two_factor_rounds():
    _, panda, _ = _stage_tags("panda")
    kq = panda["kq"]["sets"]
    CLR = {"svgd_collision_backend": "clearances"}
    ## R13: one override of svgd_kernel each (kq's own kernel is the default q, unset).
    for v, kernel in (("kz", "z"), ("kcz", "cz"), ("kqz", "qz"), ("kx", "x")):
        assert panda[v]["round"] == "R13" and panda[v]["sets"] == {**kq, **CLR, "svgd_kernel": kernel}
    ## R14: exactly the declared factors; rho 1e5 replaces the pinned 1000 in place.
    want = {"kzHonT0p1": dict(svgd_kernel="z", svgd_metric_temperature="on", svgd_temperature=0.1),
            "kzrho1e5": dict(svgd_kernel="z", svgd_rho=100000),
            "kzn128": dict(svgd_kernel="z", svgd_n=128),
            "kczrho1e5": dict(svgd_kernel="cz", svgd_rho=100000)}
    for v, over in want.items():
        assert panda[v]["round"] == "R14" and panda[v]["sets"] == {**kq, **CLR, **over}, v
    assert "svgd_metric_temperature='on'" in R.setting_diff(panda, "kzHonT0p1", "kq")
    assert "svgd_rho=100000 (primary 1000)" in R.setting_diff(panda, "kzrho1e5", "kq")
    ## The pin moved on 2026-10-12 (defaults now cz / H/T / native init), but every round here is a v1
    ## stage: what it did not name, it ran at the THEN-defaults (gen_manifest's
    ## SVGD_HISTORICAL_DEFAULTS), and the diff must say so -- `kq` is kernel q, metric off, init jitter.
    assert {k: panda["kq"]["settings"][k] for k in ("svgd_kernel", "svgd_metric_temperature",
                                                    "svgd_paired_init")} == \
        {"svgd_kernel": "q", "svgd_metric_temperature": "off", "svgd_paired_init": "jitter"}
    assert "svgd_kernel" not in panda["kq"]["sets"], "the manifests never named kernel q"
    assert "svgd_kernel='cz' (primary 'q')" in R.setting_diff(panda, "kcz", "kq")
    assert "svgd_metric_temperature='on' (primary 'off')" in R.setting_diff(panda, "kzHonT0p1", "kq")
    assert "svgd_paired_init='native' (primary 'jitter')" in \
        R.setting_diff(panda, "kczrho1e5initnative", "kq")
    assert "svgd_kernel" not in R.setting_diff(panda, "kzrho1e5", "kz"), "both z: no kernel diff"
    assert all("settings" in v for v in panda.values())


def test_stage_screw_and_soft_blocks():
    runs, variants = R.stage_catalogue()
    _, panda, _ = _stage_tags("panda")
    CLR = {"svgd_collision_backend": "clearances"}
    ## Robot blocks in manifest order: the Panda, the iiwa, then the screw and soft PCS arms.
    assert list(variants)[:2] == ["panda", "iiwa"] and set(OTHER_ROBOTS) <= set(variants)
    for robot, (rung, stage) in OTHER_ROBOTS.items():
        _, cols, by = _stage_tags(robot)
        assert list(cols) == [R.STAGE_TWIN, "kq"] + list(OTHER_ROBOT_ROUNDS[robot]), (robot, list(cols))
        assert cols[R.STAGE_TWIN]["manifest"] == "PAPER" and cols["kq"]["manifest"] == stage
        for v, st in OTHER_ROBOT_ROUNDS[robot].items():
            arms = ("numerical",) if v in ("pin2kq", "pin4kq", "polkq") else ("learned",)    # R21 / R30 / R31: the joint-space ablation
            assert cols[v]["manifest"] == st and cols[v]["arms"] == arms, (robot, v)
        assert cols["kq"]["arms"] == ("learned",) and cols[R.STAGE_TWIN]["arms"] == R.ARMS
        assert cols["kq"]["sets"] == {**panda["kq"]["sets"], **CLR}, robot
        row = (robot, "mugshelf", "paired")
        twin = by[row + (R.STAGE_TWIN,)]
        assert twin == f"sc_PAPER_{robot}_{rung}_ipopt_mugshelf_480_180_paired", twin
        assert runs[twin]["sets"] == {"correction_cost_weight": 10.0, "flow_cuda_graph": True,
                                      "max_iter": 1000000}
        assert by[row + ("kq",)] == f"sc_{stage}_{robot}_{rung}_svgd_mugshelf_480_180_paired_kq"
        for v in ("pol", "polkq"):
            st = OTHER_ROBOT_ROUNDS[robot][v]
            assert by[row + (v,)] == f"sc_{st}_{robot}_{rung}_svgd_mugshelf_480_180_paired_{v}", by[row + (v,)]
            assert sum(1 for t in runs.values() if (t["robot"], t["row"], t["start"], t["column"])
                       == row + (v,)) == 1, (robot, v, "one run per re-measured row and column")
        with tempfile.TemporaryDirectory() as root:
            ## Nothing on disk: MISSING, not fatal; `--rows` takes the robot smoke.py does not know.
            rep, text = _stage_run(root, row=row)
            assert by[row + ("kq",)] in rep.missing and twin in rep.missing
            assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
            assert f"robot {robot} (rung {rung})" in text and "robot panda" not in text
        with tempfile.TemporaryDirectory() as root:
            ## The PAPER twin and the learned-only kq: the LEAD pairs them.
            _stage_twin(root, row=row)
            _stage_write(root, "kq", [_rec(i, i < 13, 1.0) for i in range(N_CELLS)], None, row=row)
            rep, text = _stage_run(root, row=row)
            assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
            b = rep.blocks[row + ("kq",)]
            assert (b["vs_twin"]["learned"]["a_only"], b["vs_twin"]["learned"]["b_only"]) == (3, 0)
            assert f"IPOPT twin (PAPER): {twin}" in text


def test_stage_kernel_transfer_and_three_factor_rounds():
    """R17 (iiwa) / R17S (screw arm): R13's `kcz` and R14's `kczrho1e5` carried to each robot with
    exactly the Panda's settings; R18 (Panda): the declared three-factor stacks, `initnative` on the
    paired rows only; every new tag resolves through parse_stage_tag and `--variants`."""
    runs, variants = R.stage_catalogue()
    _, panda, _ = _stage_tags("panda")
    for robot, rung, rnd in (("iiwa", "n4", "R17"), ("screw7_p050", "n6", "R17S"),
                             ("soft12", "n6", "R17P")):
        _, cols, by = _stage_tags(robot)
        for v in ("kcz", "kczrho1e5"):
            assert cols[v]["round"] == rnd and cols[v]["sets"] == panda[v]["sets"], (robot, v)
            assert cols[v]["arms"] == ("learned",)
            for row in (("mugshelf", "paired"), ("posetip", "native")):
                tag = by[(robot,) + row + (v,)]
                assert tag == f"sc_SVGD_{rnd}_{robot}_{rung}_svgd_{row[0]}_480_180_{row[1]}_{v}", tag
                t = R.parse_stage_tag(tag)
                assert (t["round"], t["robot"], t["rung"], t["variant"]) == (rnd, robot, rung, v)
    kczrho = panda["kczrho1e5"]["sets"]
    assert panda["kczrho1e5n128"]["round"] == panda["kczrho1e5initnative"]["round"] == "R18"
    assert panda["kczrho1e5n128"]["sets"] == {**kczrho, "svgd_n": 128}
    assert panda["kczrho1e5initnative"]["sets"] == {**kczrho, "svgd_paired_init": "native"}
    starts = {v: {t["start"] for t in runs.values() if t["robot"] == "panda" and t["column"] == v}
              for v in ("kczrho1e5n128", "kczrho1e5initnative")}
    assert starts == {"kczrho1e5n128": {"paired", "native"}, "kczrho1e5initnative": {"paired"}}, starts
    assert "svgd_n=128 (primary" not in R.setting_diff(panda, "kczrho1e5n128", "kq") and \
        "svgd_n=128" in R.setting_diff(panda, "kczrho1e5n128", "kq")
    ## `--rows <robot> --variants kq,kcz,kczrho1e5` resolves on every robot: nothing on disk, so
    ## every run is MISSING (not fatal), and the robot's block carries the three columns.
    for robot, rnd in (("iiwa", "R17"), ("screw7_p050", "R17S"), ("soft12", "R17P")):
        row = (robot, "mugshelf", "paired")
        _, _, by = _stage_tags(robot)
        with tempfile.TemporaryDirectory() as root:
            rep, text = _stage_run(root, extra=["--variants", "kq,kcz,kczrho1e5"], row=row)
            assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
            for v in ("kq", "kcz", "kczrho1e5"):
                assert by[row + (v,)] in rep.missing, (robot, v)
            assert f"robot {robot}" in text and "robot panda" not in text


def test_stage_seed2_replication_of_the_candidate():
    """R19: the candidate `kczrho1e5` on R7's seed-2 grids. It is the Panda column
    `kczrho1e5@seed2` -- R14's settings exactly, its own round and manifest, learned arm only --
    and pairs with R7IPOPT's twin `ipopt@seed2`, never with R2's seed-1 twin or any seed-1 run.
    `--rows panda --variants kq,kczrho1e5` resolves it exactly as it resolves `kq@seed2`."""
    runs, _ = R.stage_catalogue()
    _, panda, by = _stage_tags("panda")
    col = "kczrho1e5@seed2"
    assert (panda[col]["round"], panda[col]["manifest"], panda[col]["seed"]) == ("R19", "SVGD_R19", 2)
    assert panda[col]["sets"] == panda["kczrho1e5"]["sets"] and panda[col]["arms"] == ("learned",)
    assert panda["kczrho1e5"]["round"] == "R14" and panda["kczrho1e5"]["seed"] == 1
    assert R.twin_of(col) == R.twin_of("kq@seed2") == "ipopt@seed2"
    assert by[STAGE_ROW + (col,)] == "sc_SVGD_R19_panda_n6_svgd_mugshelf_480_180_paired_kczrho1e5"
    assert {(t["row"], t["start"]) for t in runs.values() if t["column"] == col} == \
        {(r, s) for r in M.TASK_ROWS for s in M.STARTS}
    ## The seed-2 columns sit after the seed-1 pair and R7's pair, R19's after R7's primary.
    order = list(panda)
    assert order[:4] == [R.STAGE_TWIN, "kq", "ipopt@seed2", "kq@seed2"] and col in order[4:], order
    with tempfile.TemporaryDirectory() as root:
        ## Nothing on disk: both seeds' twin, primary and candidate are MISSING, nothing else of
        ## the Panda's prints, and the variant header names the seed-2 twin as its only pairing.
        rep, text = _stage_run(root, extra=["--variants", "kq,kczrho1e5"], row=("panda",))
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        want = {R.STAGE_TWIN, "kq", "kczrho1e5", "ipopt@seed2", "kq@seed2", col}
        got = {runs[t]["column"] for t in rep.missing if runs[t]["robot"] == "panda"}
        assert got == want, got ^ want
        assert len([t for t in rep.missing if runs[t]["robot"] == "panda"]) == 4 * len(want)
        assert f"grid seed 2 -- pairs only with ipopt@seed2, never with seed-1 runs" in text
        hdr = next(l for l in text.splitlines() if l.strip().startswith(col))
        assert hdr.split()[:3] == [col, "R19", "SVGD_R19"], hdr
    with tempfile.TemporaryDirectory() as root:
        ## Both twins and R14's / R7's / R19's runs on disk: R19 pairs with the seed-2 twin only
        ## (2 cells only it solved there), R14's with the seed-1 twin; the A/B against the seed-1
        ## primary skips R19, and the seed-2 variant-vs-variant table pairs R19 with R7's kq.
        _stage_twin(root)
        _stage_twin(root, grid=GRID2, column="ipopt@seed2")
        _stage_write(root, "kq", [_rec(i, i < 14, 1.0) for i in range(N_CELLS)],
                     [_rec(i, i < 10, 2.0) for i in range(N_CELLS)])
        _stage_write(root, "kczrho1e5", [_rec(i, i < 11, 1.0) for i in range(N_CELLS)], None)
        _stage_write(root, "kq@seed2", [_rec(i, i < 12, 1.0) for i in range(N_CELLS)], None,
                     grid=GRID2)
        _stage_write(root, col, [_rec(i, i < 12, 0.8) for i in range(N_CELLS)], None, grid=GRID2)
        rep, text = _stage_run(root, extra=["--variants", "kq,kczrho1e5"])
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        assert rep.blocks[STAGE_ROW + (col,)]["vs_twin"]["learned"]["a_only"] == 2
        assert rep.blocks[STAGE_ROW + ("kczrho1e5",)]["vs_twin"]["learned"]["a_only"] == 1
        lead = text[text.index("LEAD -- learned under svgd"):text.index("ABLATION -- joint space")]
        line = next(l.split() for l in lead.splitlines() if l.strip().startswith(col))
        assert line[1:6] == ["R19", "12", "10", "2", "0"], line
        ab = text[text.index("A/B against the primary kq"):text.index("VARIANTS ACROSS ROUNDS")]
        assert f"{col}: not paired with the primary -- grid seed 2" in ab
        pw = text[text.index("VARIANT vs VARIANT"):]
        g2 = pw[pw.index("(grid seed2)"):]
        hdr, first = g2.splitlines()[1].split(), g2.splitlines()[2].split()
        assert hdr == ["A", "vs", "B", col] and first[:2] == ["kq@seed2", "+0/-0"], g2[:300]
    with tempfile.TemporaryDirectory() as root:
        ## R19's run carrying the seed-1 grid (R2's hash) is refused against the seed-2 twin, and
        ## one made at seed 1 is not the run its tag names -- it never pairs with seed-1 runs.
        _stage_twin(root)
        _stage_twin(root, grid=GRID2, column="ipopt@seed2")
        _stage_write(root, col, [_rec(i, i < 12, 1.0) for i in range(N_CELLS)], None, grid=GRID)
        rep, _ = _stage_run(root)
        assert any("grid_hash " + GRID + " != " + GRID2 in m for m in rep.refused), rep.refused
        assert rep.blocks[STAGE_ROW + (col,)]["vs_twin"] == {} and rep.status == 2
    with tempfile.TemporaryDirectory() as root:
        _stage_twin(root)
        tag = _stage_write(root, col, [_rec(i, i < 12, 1.0) for i in range(N_CELLS)], None,
                           seed=1)
        rep, _ = _stage_run(root)
        assert any(tag in m and "seed 1 != the manifest's 2" in m for m in rep.refused), rep.refused
        assert STAGE_ROW + (col,) not in rep.blocks and rep.status == 2


def test_stage_r20_record_rows_at_the_pin():
    """R20, the record rows at the fielded pin (SVGD_PINNED_V2), ONE stage across four robots:
    `pin2` (the pin, zero overrides, learned arm) is a column of every robot's block with the same
    settings on each, paired with that robot's twin (R2 for the Panda, stage PAPER for the others);
    `pin2kq` (the joint-space ablation, kernel q, `--arms numerical`, its own items) only the
    Panda's, its LEAD `not run` and its ABLATION paired against the twin's joint-space arm. Both are
    v2 settings named in full, so the diff against a v1 `kq` names every moved one."""
    runs, variants = R.stage_catalogue()
    PIN = dict(svgd_kernel="cz", svgd_rho=100000, svgd_metric_temperature="on",
               svgd_paired_init="native", svgd_gn_lm=10, svgd_lr=0.3)
    EXEC = dict(correction_cost_weight=10.0, flow_cuda_graph=True, svgd_compile=True,
                svgd_cuda_graph=True, svgd_collision_backend="clearances", svgd_outer_iters=1000000)
    robots = {"panda": ("n6", "SVGD_R2"), "iiwa": ("n4", "PAPER"), "screw7_p050": ("n6", "PAPER"),
              "soft12": ("n6", "PAPER")}
    for robot, (rung, twin_stage) in robots.items():
        cols = variants[robot]
        assert cols["pin2"]["round"] == "R20" and cols["pin2"]["manifest"] == "SVGD_R20", robot
        assert cols["pin2"]["sets"] == {**EXEC, **PIN} and cols["pin2"]["arms"] == ("learned",), robot
        ## A v2 stage names every default that moved by 2026-10-12, but ran the stall patience at the
        ## then-default 5 (moved to 3 on 2026-10-16, named from SVGD_PINNED_V4 on).
        assert cols["pin2"]["settings"] == {**cols["pin2"]["sets"], "svgd_stop_patience": "5"}, \
            "a v2 stage relies on no default but the patience"
        ## pin2kq: R20 on the Panda, R21 on the other three (same token, pin and overrides).
        assert cols["pin2kq"]["round"] == ("R20" if robot == "panda" else "R21"), (robot, cols["pin2kq"])
        assert cols["pin2kq"]["arms"] == ("numerical",) and cols["pin2kq"]["sets"] == {**EXEC, **PIN, "svgd_kernel": "q"}
        for row in M.TASK_ROWS:
            for start in M.STARTS:
                tag = f"sc_SVGD_R20_{robot}_{rung}_svgd_{row}_480_180_{start}_pin2"
                assert runs[tag]["column"] == "pin2" and runs[tag]["seed"] == 1, tag
                tw = [t for t, i in runs.items() if i["variant"] == R.STAGE_TWIN and i["seed"] == 1
                      and (i["robot"], i["row"], i["start"]) == (robot, row, start)]
                assert len(tw) == 1 and runs[tw[0]]["manifest"] == twin_stage, (tag, tw)
    panda = variants["panda"]
    assert panda["pin2kq"]["sets"] == {**EXEC, **PIN, "svgd_kernel": "q"}
    assert panda["pin2kq"]["arms"] == ("numerical",) and panda["pin2kq"]["round"] == "R20"
    assert {(t["row"], t["start"]) for t in runs.values() if t["column"] == "pin2kq"} == \
        {(r, s) for r in M.TASK_ROWS for s in M.STARTS}
    ## The diff against the v1 primary `kq` (R1: kernel q, metric off, init jitter unnamed, rho 1e3,
    ## the autodiff row) names the four moved settings and the clearance row; pin2kq's has no kernel.
    d = R.setting_diff(panda, "pin2", "kq")
    for s in ("svgd_kernel='cz' (primary 'q')", "svgd_rho=100000 (primary 1000)",
              "svgd_metric_temperature='on' (primary 'off')",
              "svgd_paired_init='native' (primary 'jitter')", "svgd_collision_backend='clearances'"):
        assert s in d, (s, d)
    assert "svgd_kernel" not in R.setting_diff(panda, "pin2kq", "kq")
    ## Nothing on disk: `--rows <robot> --variants pin2,pin2kq` resolves on every robot, every R20
    ## run MISSING and not fatal.
    for robot in robots:
        row = (robot, "mugshelf", "native")
        with tempfile.TemporaryDirectory() as root:
            rep, text = _stage_run(root, extra=["--variants", "pin2,pin2kq"], row=row)
            assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
            r21 = "R20" if robot == "panda" else "R21"
            want = {f"sc_SVGD_R20_{robot}_{robots[robot][0]}_svgd_mugshelf_480_180_native_pin2",
                    f"sc_SVGD_{r21}_{robot}_{robots[robot][0]}_svgd_mugshelf_480_180_native_pin2kq"}
            assert want <= set(rep.missing), (robot, rep.missing)
            assert f"robot {robot}" in text
    ## On disk: the Panda twin, R20's pin2 (learned) and pin2kq (joint space) on one row.
    with tempfile.TemporaryDirectory() as root:
        row = ("panda", "mugshelf", "native")
        _stage_twin(root, row=row)                 # learned 0..9, joint space 0..9
        _stage_write(root, "pin2", [_rec(i, i < 13, 1.0) for i in range(N_CELLS)], None, row=row)
        tag = _stage_write(root, "pin2kq", None, [_rec(i, i < 8 or i == 12, 2.0)
                                                  for i in range(N_CELLS)], row=row)
        rep, text = _stage_run(root, extra=["--variants", "pin2,pin2kq"], row=row)
        assert rep.status == 0 and not rep.refused and not rep.bugs, (rep.refused, rep.bugs)
        assert tag not in rep.missing
        b = rep.blocks[row + ("pin2",)]
        assert (b["vs_twin"]["learned"]["a_only"], b["vs_twin"]["learned"]["b_only"]) == (3, 0)
        k = rep.blocks[row + ("pin2kq",)]
        assert set(k["vs_twin"]) == {"numerical"} and k["L"] is None
        assert (k["vs_twin"]["numerical"]["a_only"], k["vs_twin"]["numerical"]["b_only"]) == (1, 2)
        lead = text[text.index("LEAD -- learned under svgd"):text.index("ABLATION -- joint space")]
        line = next(l for l in lead.splitlines() if l.strip().startswith("pin2kq"))
        assert "not run (joint space arm only)" in line, line
        abl = text[text.index("ABLATION -- joint space"):text.index("[ipopt] round")]
        line = next(l for l in abl.splitlines() if l.strip().startswith("pin2kq"))
        assert line.split()[1:6] == ["R20", "9", "10", "1", "2"], line


## The test_* functions in name order (what `sorted(globals())` gave the old driver).
GROUPS = {"fast": [test_everything_missing_is_not_fatal, test_grid_hash_refusal,
                   test_mcnemar_directions_cost_and_bugs,
                   test_metadata_mismatch_refused_even_on_equal_hash,
                   test_multiplier_clips_by_group_line,
                   test_stage_arms_not_the_manifests_are_refused,
                   test_stage_catalogue_robots_and_seeds,
                   test_stage_grid_hash_refusal,
                   test_stage_iiwa_block_after_the_panda,
                   test_stage_kernel_and_two_factor_rounds,
                   test_stage_kernel_transfer_and_three_factor_rounds,
                   test_stage_lead_pairing_ab_and_missing,
                   test_stage_learned_only_round,
                   test_stage_r20_record_rows_at_the_pin,
                   test_stage_run_not_what_its_tag_names_is_refused,
                   test_stage_screw_and_soft_blocks,
                   test_stage_seed2_pairs_its_own_twin_only,
                   test_stage_seed2_replication_of_the_candidate,
                   test_stage_tag_parsing_every_variant]}


def main():
    _tiers.run(GROUPS)
    print(f"{len(GROUPS['fast'])} passed")


if __name__ == "__main__":
    main()
