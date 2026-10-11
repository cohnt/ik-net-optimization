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
    whose arms are not its manifest item's is REFUSED.
The stage fixtures keep gen_manifest's real tags but are 16-cell runs: the catalogue's cell count
is patched to the fixture's, which is the one thing the 480-cell tag cannot carry.
"""
import contextlib
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


## ------------------------------------------------------------------ stage mode --
STAGE_ROW = ("panda", "mugshelf", "paired")
STAGE_FP = "64dbd94dc2c2f4007d89fc3bfe3f8777e0be272f"


def _stage_tags():
    runs, variants = R.stage_catalogue()
    by = {(t["robot"], t["row"], t["start"], t["variant"]): tag for tag, t in runs.items()}
    return runs, variants, by


def _stage_meta(info, grid=GRID, overrides=None):
    return dict(task=M.TASK_ROWS[info["row"]][0], solver=info["solver"], config="latent",
                wall_time=info["cap"], seed=1, grid_hash=grid, robot=info["robot"],
                scene="panda_finray_collision_hardened.yaml", scene_mode="hardened",
                scene_fingerprint=STAGE_FP, target_placement="shelf", shelf_inset=0.1,
                start=info["start"], checkpoint=CKPT, n_targets=T, n_guesses=G,
                overrides=info["sets"] if overrides is None else overrides)


def _stage_write(root, variant, learned, numerical, grid=GRID, overrides=None):
    """`numerical=None` writes a learned-only run, as a round's `--arms learned` item does."""
    runs, _, by = _stage_tags()
    tag = by[STAGE_ROW + (variant,)]
    d = os.path.join(root, "panda", "benchmark", tag)
    os.makedirs(d, exist_ok=True)
    records = {"learned": learned} if numerical is None else {"learned": learned,
                                                              "numerical": numerical}
    arms = [bm.Arm(a, None) for a in records]
    payload = dict(metadata=_stage_meta(runs[tag], grid, overrides), n_targets=T, n_guesses=G,
                   summary=bm.summarise(records, arms, T, G), records=records)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(payload, f, default=bm._json_default)
    return tag


def _stage_run(root, extra=()):
    real = R.stage_catalogue

    def small(family="SVGD"):
        runs, variants = real(family)
        return {k: dict(v, cells=N_CELLS) for k, v in runs.items()}, variants

    out = io.StringIO()
    R.stage_catalogue = small
    try:
        with contextlib.redirect_stdout(out):
            rep = R.main(["--stage", "SVGD", "--root", root, "--rows", ",".join(STAGE_ROW)]
                         + list(extra))
    finally:
        R.stage_catalogue = real
    return rep, out.getvalue()


def _stage_twin(root, grid=GRID):
    ## IPOPT twin: learned solves 0..9, joint space 0..9.
    return _stage_write(root, R.STAGE_TWIN, [_rec(i, i < 10, 1.5, svgd=False) for i in range(N_CELLS)],
                        [_rec(i, i < 10, 2.5, svgd=False) for i in range(N_CELLS)], grid)


def test_stage_tag_parsing_every_variant():
    runs, variants, _ = _stage_tags()
    named = ({R.STAGE_TWIN} | set(GM.SVGD_VARIANTS)
             | {v for rnd in GM.SVGD_ROUNDS.values() for v in rnd})
    assert set(variants) == named, (sorted(variants), sorted(named))
    assert next(iter(variants)) == R.STAGE_TWIN and list(variants)[1] == R.stage_primary() == "kq"
    seen = set()
    for stage in GM.SVGD_STAGES:
        if stage in R.STAGE_SKIP:
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
            assert runs[tag]["manifest"] == stage
            seen.add((t["variant"], t["row"], t["start"]))
    ## Every variant on the rows its manifest gives it (both starts, or a round variant's own
    ## `starts` filter -- `initnative` runs the paired rows only), and its settings are its manifest's.
    expected = set()
    for v in named:
        starts = M.STARTS
        for rnd in GM.SVGD_ROUNDS.values():
            if v in rnd:
                starts = tuple(rnd[v][2].split(","))
        expected |= {(v, r, s) for r in M.TASK_ROWS for s in starts}
    assert seen == expected, (seen ^ expected)
    assert variants["knone"]["sets"]["svgd_kernel"] == "none"
    assert variants["rho1e4"]["sets"]["svgd_rho"] == 10000 and variants["kq"]["sets"]["svgd_rho"] == 1000
    assert "max_iter" in variants[R.STAGE_TWIN]["sets"] and "max_iter" not in variants["kq"]["sets"]
    ## The arms are the manifest's: the twin and R1 run both, the rounds what SVGD_ROUND_ARMS says.
    assert variants[R.STAGE_TWIN]["arms"] == variants["kq"]["arms"] == R.ARMS
    round_arms = tuple(GM.SVGD_ROUND_ARMS.split(","))
    for rnd in GM.SVGD_ROUNDS.values():
        for v in rnd:
            assert variants[v]["arms"] == round_arms, (v, variants[v]["arms"])
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


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
