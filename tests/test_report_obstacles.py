"""Guards for `scripts/report_obstacles.py` and for stages OBST / PILLAR's manifests.

    .venv/bin/python tests/test_report_obstacles.py        (fast tier, no solver, no Drake)

Synthetic summary.json fixtures on a 6 x 4 = 24-cell grid (plain JSON, written to a temp root):

  - the reporter's tag patterns resolve the runs of every rung and are exactly the tags the
    manifests carry;
  - a grid_hash / scene_fingerprint mismatch inside a rung is REFUSED (exit 2); within OBST a grid
    that differs across rungs, or a fingerprint repeated across rungs, is refused; PILLAR grids
    may differ across rungs and are never paired;
  - McNemar and cost / tie arithmetic on a hand-checked example, and the time-matched column;
  - a missing run prints `missing` and is not fatal;
  - >= 24 cells at a budget gives `no verdict` (and 23 does not);
  - cluster/manifest_stageOBST.txt and ..._stagePILLAR.txt: every item is its record item (PAPER /
    SVGD_R2 IPOPT, R29 `pin4`, R30 `pin4kq`, native) with ONLY the obstacle arguments (OBST) or the
    `--scene` value (PILLAR) and the tag / id prefix changed.
"""
import contextlib
import io
import json
import math
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.append(_p)

import report_obstacles as RO                                       # noqa: E402
import _tiers                                                       # noqa: E402

T, G = 6, 4
N = T * G
CHECKS, FAILURES = [0], []


def check(name, ok, detail=""):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(name)
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else f"  [{detail}]"))


def _recs(solved, cost=1.0, timed_out=0, wall=2.0, startcol=None):
    out = []
    for i in range(N):
        ok = i in solved
        r = dict(target=i // G, guess=i % G, feasible=ok, wall_time=wall,
                 timed_out=i < timed_out, hit_iteration_cap=False,
                 start_collision_value=(startcol if startcol is not None else 1.0))
        if ok:
            r["cost"] = cost
        out.append(r)
    return out


def _write(root, tag, arms, grid="g1", fp="f1", obstacles=None):
    d = os.path.join(root, "results", "x", "benchmark", tag)
    os.makedirs(d, exist_ok=True)
    meta = dict(grid_hash=grid, scene_fingerprint=fp)
    if obstacles is not None:
        meta["obstacles"] = obstacles
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(dict(metadata=meta, records=arms), f)


def _row(root, prefix, robot="iiwa", rung="n4", task="mugshelf", grid="g1", fp="f1",
         p=range(0, 18), b=range(0, 10), a=range(6, 22), base=range(0, 4), p_kw=None, ob=None):
    k = dict(grid=grid, fp=fp)
    _write(root, RO.tag_of(prefix, robot, rung, task, "svgd"),
           {"learned": _recs(set(p), **(p_kw or {}))}, obstacles=ob, **k)
    _write(root, RO.tag_of(prefix, robot, rung, task, "svgdkq"), {"numerical": _recs(set(b), cost=3.0)}, **k)
    _write(root, RO.tag_of(prefix, robot, rung, task, "ipopt"),
           {"learned": _recs(set(a), cost=2.0), "numerical": _recs(set(base), cost=4.0)}, **k)


def _main(root, *extra):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = RO.main(["--root", root, "--n-perm", "50", *extra])
    return rc, out.getvalue(), err.getvalue()


def checks_reporter():
    with tempfile.TemporaryDirectory() as root:
        _row(root, "sc_OBST8_", fp="f8",
             ob=dict(family="per_target", n=8, achieved=[8] * 5 + [7], short_targets=[5], cap_hits=[],
                     certificates=[dict(witness_obstacle_clearance=0.05),
                                   dict(witness_obstacle_clearance=0.031), None]))
        _row(root, "sc_OBST0_", fp="f0", ob=dict(family="none"))
        data = RO.build(root, ["OBST"], 50, 0)
        an = data["OBST"][("iiwa", "mugshelf")]["8"]["an"]
        check("tags resolve: the four combinations of a rung are found",
              [an["succ"][n] for n in RO.ORDER] == [18, 10, 16, 4], an["succ"])
        check("a row with no run prints as missing data, not a crash",
              data["OBST"][("iiwa", "posetip")]["8"]["an"]["succ"] == {})
        # P-only {0..5} = 6, A-only {18..21} = 4 -> n=10, k=4, p = 2*(1+10+45+120+210)/1024
        pa = an["pairs"]["proposed v A"]
        want = 2 * (1 + 10 + 45 + 120 + 210) / 1024
        check("McNemar: +6 / -4, exact two-sided p", (pa["ap"], pa["bp"]) == (6, 4)
              and abs(pa["p"] - want) < 1e-12 and pa["verdict"] == "tie", pa)
        # B solves 0..9 inside P's 0..17: P-only 8, B-only 0 -> p = 2/256
        pb = an["pairs"]["proposed v B"]
        check("McNemar: 8 / 0 is a proposed win at p = 2/256",
              (pb["ap"], pb["bp"]) == (8, 0) and abs(pb["p"] - 2 / 256) < 1e-12
              and pb["verdict"] == "proposed", pb)
        pl = an["pairs"]["L+IPOPT v JS+IPOPT"]
        check("learned + IPOPT v JS + IPOPT names its own winner",
              pl["verdict"] == "learned" and (pl["ap"], pl["bp"]) == (16, 4), pl)
        cp, ratio, n = an["cost"]["L+IPOPT"]
        check("cost: median on the 12 shared cells, ratio corner / proposed = 2, outside the tie band",
              (cp, ratio, n) == (1.0, 2.0, 12) and RO.band(ratio) == "proposed cheaper", (cp, ratio, n))
        check("cost: JS+SVGD shares 10 cells with the proposed method (enough), ratio 3",
              an["cost"]["JS+SVGD"] == (1.0, 3.0, 10), an["cost"]["JS+SVGD"])
        check("cost: JS+IPOPT shares 4 cells (< 10) and is N/A", math.isnan(an["cost"]["JS+IPOPT"][1])
              and an["cost"]["JS+IPOPT"][2] == 4, an["cost"]["JS+IPOPT"])
        check("tie band is [0.94, 1.10]", RO.band(0.94) == "tie" and RO.band(1.10) == "tie"
              and RO.band(0.93) == "corner cheaper" and RO.band(1.11) == "proposed cheaper")
        check("wall is the mean per cell clamped at the 180 s cap",
              an["wall"]["L+SVGD"] == 2.0 and RO.wall([dict(wall_time=999.0)]) == 180.0)
        check("time-matched is computed for the IPOPT corners and lies in [0, 1]",
              set(an["tm"]) == {"L+IPOPT", "JS+IPOPT"} and all(0 <= v["J_match"] <= 1 for v in an["tm"].values()))
        check("start-collision share counts start_collision_value > 1 (here 0 of 24)",
              an["startcol"]["L+SVGD"] == (0, 24), an["startcol"])
        d = data["OBST"][("iiwa", "mugshelf")]["8"]["diag"]
        check("obstacle diagnostics: achieved median / min, short targets, certificate minimum",
              d["median"] == 8 and d["min"] == 7 and d["short"] == 1 and d["cert_min"] == 0.031
              and d["n_cert"] == 2, d)
        rc, out, err = _main(root, "--family", "OBST")
        check("the report prints and exits 0 with missing runs", rc == 0 and "missing" in out, (rc, err))
        check("layout: learned and joint space adjacent per solver",
              "| L+SVGD | JS+SVGD | L+IPOPT | JS+IPOPT |" in out)

    with tempfile.TemporaryDirectory() as root:
        rc, out, _ = _main(root)
        check("an empty root prints every row as missing and exits 0", rc == 0 and out.count("missing") > 50)

    with tempfile.TemporaryDirectory() as root:
        _row(root, "sc_OBST8_", fp="f8", p_kw=dict(timed_out=23))
        _row(root, "sc_OBST16_", fp="f16", p_kw=dict(timed_out=24))
        data = RO.build(root, ["OBST"], 50, 0)
        a8 = data["OBST"][("iiwa", "mugshelf")]["8"]["an"]
        a16 = data["OBST"][("iiwa", "mugshelf")]["16"]["an"]
        check("23 cells at a budget keeps the verdict", not a8["flag"]["L+SVGD"]
              and not a8["pairs"]["proposed v A"]["no_verdict"])
        check("24 cells at a budget: no verdict for every comparison the run enters",
              a16["flag"]["L+SVGD"]
              and all(c["no_verdict"] for k, c in a16["pairs"].items() if k.startswith("proposed"))
              and not a16["pairs"]["L+IPOPT v JS+IPOPT"]["no_verdict"])
        _, out, _ = _main(root, "--family", "OBST")
        check("the flag prints", "NO VERDICT" in out and "no verdict" in out)

    ## AL-SVGD's own budget stops count too (svgd.stop_reason wall_clock / step_cap), as in
    ## report_svgd_2x2.clock_stops; a converged stop does not.
    recs = _recs(set(range(N)))
    for i, r in enumerate(recs):
        r["svgd"] = {"stop_reason": ("wall_clock" if i < 12 else "step_cap" if i < 24
                                     else "stall")}
    check("svgd clock / step-cap stops count as cells at a budget",
          RO.cells_at_budget(recs) == 24, RO.cells_at_budget(recs))

    def refused(label, build, fam="OBST"):
        with tempfile.TemporaryDirectory() as root:
            build(root)
            rc, out, err = _main(root, "--family", fam)
            check(label, rc == 2 and err.startswith("refused") and "|" not in out, (rc, err[:120]))

    def mismatch_grid(root):
        _row(root, "sc_OBST8_", fp="f8")
        _write(root, RO.tag_of("sc_OBST8_", "iiwa", "n4", "mugshelf", "svgdkq"),
               {"numerical": _recs(set(range(10)))}, grid="OTHER", fp="f8")
    refused("a grid_hash mismatch inside a rung is refused", mismatch_grid)

    def mismatch_fp(root):
        _row(root, "sc_OBST8_", fp="f8")
        _write(root, RO.tag_of("sc_OBST8_", "iiwa", "n4", "mugshelf", "ipopt"),
               {"learned": _recs(set()), "numerical": _recs(set())}, fp="OTHER")
    refused("a scene_fingerprint mismatch inside a rung is refused", mismatch_fp)

    def cells(root):
        _row(root, "sc_OBST8_", fp="f8")
        _write(root, RO.tag_of("sc_OBST8_", "iiwa", "n4", "mugshelf", "svgdkq"),
               {"numerical": _recs(set())[:20]}, fp="f8")
    refused("runs covering different cells are refused", cells)

    def obst_grid(root):
        _row(root, "sc_OBST8_", fp="f8")
        _row(root, "sc_OBST16_", fp="f16", grid="g2")
    refused("OBST: a grid that differs across rungs is refused", obst_grid)

    def obst_fp(root):
        _row(root, "sc_OBST8_", fp="same")
        _row(root, "sc_OBST16_", fp="same")
    refused("OBST: equal fingerprints across rungs are refused", obst_fp)

    def nofp(root):
        _row(root, "sc_OBST8_", fp=None)
    refused("a run that records no scene_fingerprint is refused", nofp)

    with tempfile.TemporaryDirectory() as root:
        _row(root, "sc_PILLAR040_", grid="gA", fp="fA", p=range(0, 24))
        _row(root, "sc_PILLAR080_", grid="gB", fp="fB", p=range(0, 5))
        rc, out, err = _main(root, "--family", "PILLAR")
        r = RO.build(root, ["PILLAR"], 50, 0)["PILLAR"][("iiwa", "mugshelf")]
        check("PILLAR: grids may differ across rungs; each rung is read on its own run",
              rc == 0 and r["040"]["an"]["succ"]["L+SVGD"] == 24 and r["080"]["an"]["succ"]["L+SVGD"] == 5)
        p40, p80 = (r[k]["an"]["pairs"]["proposed v A"] for k in ("040", "080"))
        # A solves 6..21 on both rungs: 040 P-only = 0..5 + 22,23, 080 P-only = 0..4, A-only = 16
        check("cross-rung pairing never happens: each rung's McNemar uses only its own runs",
              (p40["ap"], p40["bp"]) == (8, 0) and (p80["ap"], p80["bp"]) == (5, 16), (p40, p80))
        check("PILLAR rung 0 is its own tag family (sc_PILLAR000_)",
              RO.tag_of("sc_PILLAR000_", "panda", "n6", "mugshelf", "ipopt")
              == "sc_PILLAR000_panda_n6_ipopt_mugshelf_480_180_native")


def _parse(path):
    items = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            i, env, script, args = line.rstrip("\n").split("|", 3)
            items[i] = (env, script, args.split(" "))
    return items


def checks_manifests():
    cl = os.path.join(REPO, "cluster")
    rec = {}
    for stage in ("PAPER", "SVGD_R2", "SVGD_R29", "SVGD_R30"):
        for i, v in _parse(os.path.join(cl, f"manifest_stage{stage}.txt")).items():
            rec[i] = (stage, v)
    for family in ("OBST", "PILLAR"):
        items = _parse(os.path.join(cl, f"manifest_stage{family}.txt"))
        bad, tags = [], {}
        for i, (env, script, args) in items.items():
            tag = args[args.index("--tag") + 1]
            m = re.match(r"sc_(OBST(\d+)|PILLAR(\d+))_(.*)$", tag)
            if not m or not i.startswith(f"sc_{m.group(1)}_"):
                bad.append(("tag", i))
                continue
            tail = i[len(f"sc_{m.group(1)}_"):]
            cands = [rec[f"sc_{s}_{tail}"] for s in ("PAPER", "SVGD_R2", "SVGD_R29", "SVGD_R30")
                     if f"sc_{s}_{tail}" in rec]
            if len(cands) != 1:
                bad.append(("no unique record item", i, len(cands)))
                continue
            stage, (renv, rscript, rargs) = cands[0]
            a = list(args)
            if family == "OBST":
                n = int(m.group(2))
                if n:
                    if a[-4:] != ["--obstacles", "per_target", "--obstacle-count", str(n)]:
                        bad.append(("obstacle args", i))
                        continue
                    a = a[:-4]
            else:
                # r120 alone raises the sampler's abort threshold (Tommy, 2026-11-26): it only stops sampling,
                # accepted draws are unchanged (gen_manifest.PILLAR_R120_MAX_REJECTIONS)
                cap = ["--max-target-rejections", "500000"]
                if (a[-2:] == cap) != (m.group(3) == "120"):
                    bad.append(("max-target-rejections only and exactly on r120", i))
                    continue
                if m.group(3) == "120":
                    a = a[:-2]
                scene = {"000": "hardened", "040": "pillars_r040", "080": "pillars_r080",
                         "120": "pillars_r120"}[m.group(3)]
                j = a.index("--scene")
                if a[j + 1] != scene:
                    bad.append(("scene", i))
                    continue
                a[j + 1] = "hardened"
            a[a.index("--tag") + 1] = rargs[rargs.index("--tag") + 1]
            if (env, script, a) != (renv, rscript, rargs):
                bad.append(("differs from the record item", i))
            rtag = rargs[rargs.index("--tag") + 1]
            if stage == "SVGD_R29" and not rtag.endswith("_native_pin4"):
                bad.append(("R29 not pin4", i))
            if stage == "SVGD_R30" and not rtag.endswith("_native_pin4kq"):
                bad.append(("R30 not pin4kq", i))
            tags[tag] = m.group(1)
        check(f"{family}: every item is its record item with only the obstacle args / scene and tag changed",
              not bad and bool(items), bad[:3])
        want = 120 if family == "OBST" else 96
        check(f"{family}: {want} logical runs", len(tags) == want, len(tags))
        prefixes = [p for _, p in RO.FAMILIES[family]]
        expect = {RO.tag_of(p, r, g, t, k) for p in prefixes for r, g in RO.ROBOTS
                  for t, _ in RO.TASKS for k in ("svgd", "svgdkq", "ipopt")}
        check(f"{family}: the reporter's tag patterns are exactly the manifest's tags",
              expect == set(tags), sorted(expect ^ set(tags))[:3])
        check(f"{family}: only IPOPT and AL-SVGD, native start, no pin4s2",
              all(a[a.index("--solver") + 1] in ("ipopt", "svgd") and a[a.index("--start") + 1] == "native"
                  and "pin4s2" not in a[a.index("--tag") + 1] for _, _, a in items.values()))


GROUPS = {"fast": [checks_reporter, checks_manifests]}


def main():
    _tiers.run(GROUPS)
    print(f"\n{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
