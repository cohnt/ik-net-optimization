"""The svgd smoke: every variant of plain AL-SVGD on a FIXED SUBSET of the record's own cells.

    .venv/bin/python scripts/svgd/smoke.py --dry-run                 # print every command
    .venv/bin/python scripts/svgd/smoke.py --only-missing            # run what is not on disk
    .venv/bin/python scripts/svgd/smoke.py --rows panda,posetip --columns al64,ipopt
    .venv/bin/python scripts/report_svgd.py                          # read it

The matrix and the go/no-go rules are pre-registered in `docs/svgd-solver.md`, written before
any smoke number was read; this script is the matrix, and `scripts/report_svgd.py` imports it
from here so the two cannot disagree about what a tag means.

**Why a subset of the record's grid.** Every run uses the record's grid flags exactly
(`--targets 60 --guesses 8 --seed 1 --scene hardened --shelf-inset 0.1 --config latent
--set correction_cost_weight=10.0 --compile`, the stage-STATUSQUO placement per row) and
restricts it with `--cells`. `grid_hash` hashes the whole seeded grid, not the cells run, so a
smoke summary carries the record's hash and pairs against it cell for cell. The cells are two
guesses of every 13th target, so the subset spans the grid rather than its first target.

**Why an IPOPT twin per row.** The record ran at 180 s on SuperCloud V100s; seconds are never
compared across machines. The twin is the same row on the same cells on THIS machine, at the
smoke's cap, `--solver ipopt --set flow_cuda_graph=True` (the CUDA-graph switch the svgd
solver's graphed mode is the counterpart of), so learned-under-svgd has a same-machine,
same-cap, same-chart IPOPT column to pair against.

**The iiwa rung is n6, not the record's n4**: n4 is not on this laptop. The tag says so
(`n6-not-record-n4`), and the reporter pairs the iiwa learned arm against its IPOPT twin only,
never against the record's n4 learned column; the joint-space arm, which never evaluates the
chart, still pairs against the record. On the iiwa GRASP rows neither arm pairs against the
record: its wsg-gripper grasp rows predate the mug-handle fix (`452d784`, branch
`mug-handle-yaw`) and are void, so those rows are read against the twin alone -- and should be
run only once that fix is merged into this branch.

Tags name every setting in full and parallel the record's
`sc_<STAGE>_<robot>_<rung>_<solver>_<row>_<cells>_<cap>_<start>`:

    smoke_SVGD_<robot>_<rung>_<variant>_<row>_<cells>_<cap>_<start>
    variant = svgd-al_svgd-n<N>-kernel_<kernel>-constraint_inside_kernel_<bool>
              -warmup_<warmup>-init_<paired_init>-<dtype>-<mode>
            | ipopt-flow_cuda_graph

`init_<paired_init>` is in every svgd tag although `svgd_paired_init` is read only under the
paired protocol, so one variant has one token on both protocols. Columns that fix their own
`svgd_paired_init` (`al64native`) carry it whatever `--paired-init` says.

**Memory.** Every child is launched under `systemd-run --user --scope -p MemoryMax=<cap>
-p MemorySwapMax=0` where `systemd-run` exists (`--memory-max`, default 24G; `--no-memory-cap`
disables), so a runaway kills the run and not the user's session. Not part of the tag: an
execution setting.

Writes only under `results/` (gitignored): each run's `summary.json` from the benchmark script,
and its stdout beside it as `driver.log`. Touches no tracked file. Runs sequentially by default
because the GPU is shared; `--jobs K` runs K at once with `PROCS=K` in each child's environment.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(REPO)

## ------------------------------------------------------------------- the matrix --
PREFIX = "smoke_SVGD"
RECORD_CELLS = "0:0,0:1,13:0,13:1,26:0,26:1,39:0,39:1,52:0,52:1"

## The record's grid, verbatim from stage STATUSQUO (`cluster/gen_manifest.py`): `item()`'s
## base flags plus that stage's `base`. `--config latent` matters: the Panda script defaults
## to `baseline`, the iiwa's to `latent`.
GRID_FLAGS = ["--targets", "60", "--guesses", "8", "--seed", "1", "--compile",
              "--config", "latent", "--set", "correction_cost_weight=10.0",
              "--scene", "hardened", "--shelf-inset", "0.1"]
ARMS = "learned,numerical"

ROBOTS = {
    "panda": dict(script="scripts/panda/panda_benchmark.py", rung="n6",
                  checkpoint="models/panda/panda__n6__step620000.pkl",
                  record_rung="n6", results="results/panda/benchmark"),
    ## n4 is the record's rung and is not on this laptop. The rung token says so, so no reader
    ## takes an iiwa smoke column for the record's chart.
    "iiwa": dict(script="scripts/iiwa/iiwa_benchmark.py", rung="n6-not-record-n4",
                 checkpoint="models/iiwa14/iiwa14__n6__step620000.pkl",
                 record_rung="n4", results="results/iiwa/benchmark",
                 ## The wsg finray's mug handle sat 18 mm inside a finger until `452d784`
                 ## (branch mug-handle-yaw); every wsg-gripper GRASP row of the record predates
                 ## the fix and is void once it is merged. The Panda's gripper never had it.
                 record_grasp_void=True),
}

## The record's two experiments, by its own row tokens (`STATUSQUO_ROWS`).
TASK_ROWS = {
    "mugshelf": ("mug", ["--target-placement", "shelf"]),
    "posetip": ("pose", ["--target-placement", "shelf", "--placement-point", "fingertips"]),
}
ROW_NAME = {"mugshelf": "grasp contained", "posetip": "pose contained (tip)"}
STARTS = ("paired", "native")

## The svgd columns: id -> the settings it FIXES (everything else at its ProgramOptions default).
## `al64` is the primary (N = 64, RBF kernel on q, the Tabor-Hermans form, no warm-up); every
## other column differs from it in exactly one setting, so each is an A/B against `al64`:
## N = 1 (single particle), kernel none (no interaction: batched AL), N = 256, the literal
## form (the whole AL gradient inside the kernel average), the native swarm init, and the CEM
## warm-up.
PRIMARY = "al64"
_BASE = dict(svgd_n=64, svgd_kernel="q", svgd_constraint_inside_kernel=False, svgd_warmup="none")
SVGD_BASES = {
    "al1": dict(_BASE, svgd_n=1),
    "al64none": dict(_BASE, svgd_kernel="none"),
    "al64": dict(_BASE),
    "al256": dict(_BASE, svgd_n=256),
    "al64lit": dict(_BASE, svgd_constraint_inside_kernel=True),
    "al64native": dict(_BASE, svgd_paired_init="native"),
    "al64cem": dict(_BASE, svgd_warmup="cem"),
}
IPOPT = "ipopt"
SVGD_COLUMNS = tuple(SVGD_BASES)
ALL_COLUMNS = (IPOPT,) + SVGD_COLUMNS

## The svgd step modes, as `ProgramOptions` fields.
MODES = {"eager": {"svgd_compile": False, "svgd_cuda_graph": False},
         "compiled": {"svgd_compile": True, "svgd_cuda_graph": False},
         "graphed": {"svgd_compile": True, "svgd_cuda_graph": True}}


def column_settings(column, paired_init="jitter"):
    """column id -> the full svgd settings it runs with (empty for the IPOPT twin):
    `svgd_method`, the column's own settings, and `svgd_paired_init` (the column's where it
    fixes one, else `paired_init`)."""
    if column == IPOPT:
        return {}
    if column not in SVGD_BASES:
        raise SystemExit(f"unknown column {column!r}; columns are {', '.join(ALL_COLUMNS)}")
    return dict(dict(svgd_method="al_svgd", svgd_paired_init=paired_init), **SVGD_BASES[column])


def variant_token(column, paired_init="jitter", dtype="float32", mode="graphed"):
    if column == IPOPT:
        return "ipopt-flow_cuda_graph"
    s = column_settings(column, paired_init)
    return (f"svgd-{s['svgd_method']}-n{s['svgd_n']}-kernel_{s['svgd_kernel']}"
            f"-constraint_inside_kernel_{s['svgd_constraint_inside_kernel']}"
            f"-warmup_{s['svgd_warmup']}-init_{s['svgd_paired_init']}-{dtype}-{mode}")


def cap_token(cap):
    return f"{float(cap):g}".replace(".", "p")


def make_tag(robot, row, start, column, n_cells, cap, prefix=PREFIX, **variant_kw):
    return "_".join([prefix, robot, ROBOTS[robot]["rung"], variant_token(column, **variant_kw),
                     row, str(n_cells), cap_token(cap), start])


def record_tags(robot, row, start, stage="STATUSQUO"):
    """The record's IPOPT column for this row, lifted-budget re-run first (the record is
    REPORTED at the lifted budget where stage ITCAP re-ran it, `report_statusquo.py`)."""
    rr = ROBOTS[robot]["record_rung"]
    base = f"{stage}_{robot}_{rr}_ipopt_{row}_480_180_{start}"
    return [f"sc_ITCAP1e6_{base}", f"sc_{base}"]


def rows_matrix(robots=None, rows=None, starts=None):
    out = []
    for robot in ROBOTS:
        for row in TASK_ROWS:
            for start in STARTS:
                if robots and robot not in robots:
                    continue
                if rows and row not in rows:
                    continue
                if starts and start not in starts:
                    continue
                out.append((robot, row, start))
    return out


def parse_row_filter(spec):
    """`--rows panda,posetip,paired` -> (robots, rows, starts); within a dimension the tokens
    OR, across dimensions they AND. `mug`/`pose` are accepted for the row tokens."""
    if not spec:
        return None, None, None
    robots, rows, starts = set(), set(), set()
    alias = {"mug": "mugshelf", "grasp": "mugshelf", "pose": "posetip"}
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        tok = alias.get(tok, tok)
        if tok in ROBOTS:
            robots.add(tok)
        elif tok in TASK_ROWS:
            rows.add(tok)
        elif tok in STARTS:
            starts.add(tok)
        else:
            raise SystemExit(f"--rows: {tok!r} is not a robot ({', '.join(ROBOTS)}), a row "
                             f"({', '.join(TASK_ROWS)}) or a start ({', '.join(STARTS)})")
    return robots or None, rows or None, starts or None


def parse_columns(spec):
    if not spec:
        return list(ALL_COLUMNS)
    out = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        column_settings(tok)                         # validates
        if tok not in out:
            out.append(tok)
    return out


def memory_wrapper(memory_max):
    """The `systemd-run` scope prefix that caps a child's memory (no swap), or [] where
    `systemd-run` is unavailable or `memory_max` is None."""
    if not memory_max or shutil.which("systemd-run") is None:
        return []
    return ["systemd-run", "--user", "--scope", "-p", f"MemoryMax={memory_max}",
            "-p", "MemorySwapMax=0", "--quiet"]


def command(robot, row, start, column, cells, cap, paired_init, dtype, mode, prefix=PREFIX,
            memory_max="24G"):
    spec = ROBOTS[robot]
    task, placement = TASK_ROWS[row]
    n_cells = len(cells.split(","))
    tag = make_tag(robot, row, start, column, n_cells, cap, prefix,
                   paired_init=paired_init, dtype=dtype, mode=mode)
    args = memory_wrapper(memory_max) + [sys.executable, os.path.join(REPO, spec["script"]),
            "--task", task, "--start", start, "--arms", ARMS, "--wall-time", f"{float(cap):g}",
            "--cells", cells, "--tag", tag, "--checkpoint", spec["checkpoint"]]
    args += GRID_FLAGS + placement
    if column == IPOPT:
        args += ["--solver", "ipopt", "--set", "flow_cuda_graph=True"]
    else:
        args += ["--solver", "svgd"]
        settings = dict(column_settings(column, paired_init), svgd_dtype=dtype,
                        **MODES[mode])
        for k, v in settings.items():
            args += ["--set", f"{k}={v}"]
    return tag, args


def summary_path(robot, tag):
    return os.path.join(REPO, ROBOTS[robot]["results"], tag, "summary.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default=None,
                   help="filter tokens: robots (panda, iiwa), rows (mugshelf/mug, posetip/pose), "
                        "starts (paired, native); OR within a dimension, AND across")
    p.add_argument("--columns", default=None,
                   help=f"column ids ({', '.join(ALL_COLUMNS)}); default all")
    p.add_argument("--wall-time", type=float, default=20.0, help="the per-cell cap, seconds")
    p.add_argument("--cells", default=RECORD_CELLS, help="TI:GI list; default the record subset")
    p.add_argument("--paired-init", choices=("jitter", "native"), default="jitter",
                   help="svgd_paired_init for every svgd column (named in the tag)")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                   help="svgd_dtype for every svgd column (named in the tag)")
    p.add_argument("--mode", choices=tuple(MODES), default="graphed",
                   help="svgd step mode: eager, compiled (svgd_compile) or graphed "
                        "(svgd_compile + svgd_cuda_graph); named in the tag")
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--memory-max", default="24G",
                   help="systemd-run MemoryMax per child (MemorySwapMax=0); default 24G")
    p.add_argument("--no-memory-cap", action="store_true", help="launch children uncapped")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    p.add_argument("--only-missing", action="store_true",
                   help="skip every tag whose summary.json already exists")
    p.add_argument("--jobs", type=int, default=1,
                   help="runs at once (default 1: the GPU is shared); children get PROCS=K")
    p.add_argument("--item-timeout", type=float, default=None,
                   help="OS-level bound per run, seconds (default: generous, from the cap)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    robots, rows, starts = parse_row_filter(args.rows)
    columns = parse_columns(args.columns)
    matrix = rows_matrix(robots, rows, starts)
    n_cells = len(args.cells.split(","))
    plan = []
    for robot, row, start in matrix:              # rows outer: a partial smoke has whole rows
        for column in columns:
            tag, cmd = command(robot, row, start, column, args.cells, args.wall_time,
                               args.paired_init, args.dtype, args.mode, args.prefix,
                               None if args.no_memory_cap else args.memory_max)
            plan.append((robot, tag, cmd))
    skipped = [t for r, t, _ in plan if args.only_missing and os.path.exists(summary_path(r, t))]
    todo = [(r, t, c) for r, t, c in plan if t not in skipped]
    ## Process startup and the flow / svgd compile are outside the cells' clock; the item bound
    ## only has to catch a wedged process, never a slow solve.
    timeout = args.item_timeout or 2.0 * (n_cells * 2 * (args.wall_time + 30.0)) + 900.0
    print(f"svgd smoke: {len(plan)} runs ({len(matrix)} rows x {len(columns)} columns), "
          f"{n_cells} cells x 2 arms each at {args.wall_time:g} s; {len(skipped)} on disk, "
          f"{len(todo)} to run; worst case {len(todo) * n_cells * 2 * args.wall_time / 3600:.1f} h "
          f"of solve clock", flush=True)
    wrap = memory_wrapper(None if args.no_memory_cap else args.memory_max)
    print(f"memory cap per child: {'MemoryMax=' + args.memory_max + ', MemorySwapMax=0 (systemd-run)' if wrap else 'NONE'}",
          flush=True)
    if args.dry_run:
        for _, tag, cmd in todo:
            print(shlex.join(cmd))
        return 0

    env = dict(os.environ, PROCS=str(max(1, args.jobs)))

    def run(item):
        robot, tag, cmd = item
        out_dir = os.path.dirname(summary_path(robot, tag))
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        with open(os.path.join(out_dir, "driver.log"), "w") as log:
            log.write(shlex.join(cmd) + "\n\n")
            log.flush()
            try:
                rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        ok = rc == 0 and os.path.exists(summary_path(robot, tag))
        print(f"[{'ok' if ok else 'FAIL'}] {tag}  rc={rc}  {time.time() - t0:.0f} s", flush=True)
        return tag, ok

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(run, todo))
    failed = [t for t, ok in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} runs wrote a summary.")
    for t in failed:
        print(f"  FAILED: {t}  (see its driver.log)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
