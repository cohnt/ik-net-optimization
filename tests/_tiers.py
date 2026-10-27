"""The two tiers every file in `tests/` runs in: `python tests/<file>.py [group ...]`.

  fast  (the default) the IMPLEMENTATION tier: invariants, wiring, bit-identity of a default to
        its pinned value, schema and record fields, gradient checks on small inputs. Each file's
        fast tier runs in about two minutes or less on ONE CPU core, a few seconds per check.
  slow  the PARITY / PERFORMANCE tier, opt-in by name: reproductions of record behaviour,
        end-to-end grids, convergence from generic starts, anything that asserts seconds. It runs
        on the GPU or on SuperCloud (`cluster/tests_slow_job.sh`), never as a routine local check.
  all   both, fast first (fast alone in a file with no slow group).

A file may name further groups (a single robot, a sweep, a profile); a check listed in two
requested groups runs once. Every check is timed and printed as `[   12.3 s] name`, so a file's
tiers can be re-measured from its own output. Failure reporting stays each file's own (an
`assert` aborts; a `check()` list is returned by the file's `main`).
"""
import sys
import time


def label(check):
    """A check's printed name: the function's, or a partial's with its arguments."""
    func = getattr(check, "func", check)
    args = ", ".join([*map(repr, getattr(check, "args", ())),
                      *(f"{k}={v!r}" for k, v in (getattr(check, "keywords", None) or {}).items())])
    return f"{func.__name__}({args})" if args else func.__name__


def run(groups, argv=None, default=("fast",), after=None):
    """Run the checks of the requested groups in order. `groups` maps a group name to a list of
    zero-argument callables (bind arguments with `functools.partial`); `after` runs after every
    check, pass or fail (e.g. to release cached replays)."""
    names = list(sys.argv[1:] if argv is None else argv) or list(default)
    if "all" in names:
        i = names.index("all")
        names[i:i + 1] = [n for n in ("fast", "slow") if n in groups]
    unknown = [n for n in names if n not in groups]
    if unknown:
        sys.exit(f"unknown group(s) {unknown}; this file has {sorted(groups)} and 'all'")
    seen, t_all = set(), time.perf_counter()
    for name in names:
        print(f"\n=== group {name} ===", flush=True)
        for check in groups[name]:
            if check in seen:
                continue
            seen.add(check)
            t = time.perf_counter()
            try:
                check()
            finally:
                if after is not None:
                    after()
                print(f"[{time.perf_counter() - t:7.1f} s] {label(check)}", flush=True)
    print(f"[{time.perf_counter() - t_all:7.1f} s] total, groups {' '.join(names)}", flush=True)


def jax_compilation_cache():
    """Point JAX's persistent compilation cache at this project's own cache directory. Call it
    before anything imports jax. The GVS arm's model is a set of XLA programs whose compiles are
    most of a cold run on one core (~190 s cold, ~45 s warm for `test_gvs_arm_model`); the cache
    is keyed on the program, the compile options and the jax/jaxlib version, so a hit is the
    same executable. Set `JAX_COMPILATION_CACHE_DIR` yourself to move it, or to "" to disable."""
    import os
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "learned-ik", "jax"))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")

