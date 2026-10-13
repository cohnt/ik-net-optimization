"""scripts/report_analytic.py on synthetic summaries: it pairs, counts and refuses correctly.

    python tests/test_report_analytic.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import report_analytic as ra  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          {detail}")
        FAILURES.append(name)


def recs(feasible, **extra):
    return [dict(target=i // 2, guess=i % 2, feasible=f, feasible_relaxed=f, wall_time=1.0 + i,
                 iterations=10 + i, cost=float(i + 1), **extra) for i, f in enumerate(feasible)]


def write(root, tag, records, grid="g0-pose", fp="f" * 40):
    d = os.path.join(root, "results/panda/benchmark", tag)
    os.makedirs(d, exist_ok=True)
    meta = dict(grid_hash=grid, scene_fingerprint=fp, wall_time=180.0, seed=1, solver="ipopt",
                hosts=["c-1-2-3"] if "ANALYTIC" in tag else ["d-7-8-2", "d-9-4-1"])
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(dict(metadata=meta, records=records, summary={}), f)


def main():
    cells = 12
    tail = "panda_n6_ipopt_posetip_12_180_paired"
    with tempfile.TemporaryDirectory() as tmp:
        here = os.getcwd()
        os.chdir(tmp)
        try:
            L = [True] * 6 + [False] * 6
            J = [True] * 9 + [False] * 3
            A = [True] * 12
            write(tmp, f"sc_PAPER_{tail}", dict(learned=recs(L), numerical=recs(J)))
            write(tmp, f"sc_ANALYTIC_{tail}", dict(analytic=recs(A), analytic8=recs(A)))
            write(tmp, f"sc_ANALYTICSMOKE_{tail}", dict(analytic=recs(A[:4])))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rows = ra.main(["--cells", str(cells)])
            check("both analytic arms paired against the record run", len(rows) == 2,
                  out.getvalue())
            r = rows[0]
            check("L v JS counted from the record (0 learned-only, 3 joint-only)",
                  r["LvJ"][:2] == (0, 3), r["LvJ"])
            check("A v L: 6 analytic-only cells, 0 learned-only", r["AvL"][:2] == (6, 0), r["AvL"])
            check("A v JS: 3 analytic-only cells", r["AvJ"][:2] == (3, 0), r["AvJ"])
            check("cost needs >= 10 shared cells (A v L shares 6: N/A)", r["AvL"][4] is None)
            check("the smoke run is not read", all("SMOKE" not in x["tag"] for x in rows))
            check("node classes recorded (analytic on c, record on d)", r["nodes"] == ("c", "d"),
                  r["nodes"])
            check("the cross-machine note is printed",
                  "node class c, the record on d" in out.getvalue(), out.getvalue()[-400:])

            write(tmp, f"sc_ANALYTIC_{tail}", dict(analytic=recs(A)), grid="other-pose")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ra.main(["--cells", str(cells)])
                check("a grid_hash mismatch is refused", False, "paired anyway")
            except SystemExit as e:
                check("a grid_hash mismatch is refused", "grid_hash" in str(e), str(e))
            write(tmp, f"sc_ANALYTIC_{tail}", dict(analytic=recs(A)), fp="0" * 40)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ra.main(["--cells", str(cells)])
                check("a scene_fingerprint mismatch is refused", False, "paired anyway")
            except SystemExit as e:
                check("a scene_fingerprint mismatch is refused", "scene_fingerprint" in str(e), str(e))
        finally:
            os.chdir(here)
    m = ra.TAG.match("sc_ANALYTICST_iiwa_n4_svgd_mugshelf_480_180_native_pin2kq")
    check("an svgd analytic tag parses: stage S, tight ladder, kernel q",
          m is not None and m.group(1) == "S" and m.group(2) == "T" and m.group(4) == "_pin2kq"
          and ra.parse_tag(m.group(0)[:-len(m.group(4))])["start"] == "native", m and m.groups())
    print(f"\n{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
