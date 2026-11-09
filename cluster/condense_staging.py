#!/usr/bin/env python3
"""Promote what is unique in results/_cluster_staging into results/, then trash the rest.

The rule (Thomas, 2026-11-09, adopted from the analytic-and-optimization-ik project): staging is
TEMPORARY, not an archive. Once a collection's shards are merged and the merged runs are in
results/, its staging tree goes to the desktop trash (`gio trash`, never `rm`), and a critic
checks before the trash is emptied. Nothing on the cluster is touched. Nothing is trashed whose
data is not verified, on disk, to exist elsewhere under results/; anything else stays in staging
and is listed.

Logs and shard metadata, for every run directory about to leave staging. Every file in a run directory other than its summary.json -- in
practice the run's `solver_logs.tar.gz` -- is MOVED to the log store
`results/_solver_logs/<robot>/<tag>/<file>` (`<stamp>.<file>` when a different file of that name
is already stored; left in place when an identical one is). Logs are never trashed here; deleting
them is `cluster/prune_logs.sh`'s decision. `scripts/report_step.py --resto` reads them there.
Every shard's summary.json minus its `records` (its own metadata -- host, cells, platform, resume
events, timings -- and its shard-local summary block; ~5 KB) is WRITTEN to
`results/_shard_stubs/<robot>/<tag>/summary_stub.json` (same naming), because a merged run keeps
the records verbatim but only shard 0's metadata. So a trashed shard leaves nothing behind that
exists only in the trash: its records are in a merged run, its logs and metadata in the stores.

Units. A run directory `<stamp>/results/<robot>/benchmark/<tag>/` is one unit; every other file
(or symlink) is its own unit. A unit's destination is the same path with `<stamp>/` dropped
(`<stamp>/results/X` -> `results/X`; `<stamp>/calib/X` -> `results/_cluster_calib/X`). Each unit
is exactly one of:

  redundant    a shard (`*_shardKofN`) every one of whose records, arm by arm, is byte-for-byte
               a record of a summary.json already under results/ for the same robot (its merged
               run, or a merged run archived since, e.g. results/archive_preoverrun/) -- checked
               by content hash, record by record, never by name; a run whose destination holds
               an identical summary.json; a file whose destination holds an identical file, or
               one it is a byte prefix of (launch logs and wandb files only grow).
  promote      nothing at the destination: copy it there (a run: its summary.json; a promoted
               unmerged shard goes to results/_unmerged_shards/<robot>/benchmark/<tag>/, out
               of the benchmark tree every reader globs). When several
               collections staged one destination, the NEWEST collection's copy is promoted --
               the copy every report script picked while they still read staging -- and older
               copies are judged against it.
  conflict     the destination holds something different. Every reader preferred results/, so
               the staged copy is the superseded one; kept unless --archive-conflicts copies it
               to results/archive_staging/<stamp>/<same path> (then it is redundant).
  unmerged     a shard whose records are not all under results/: an in-flight stage, or a shard
               set never merged (smokes, MERGECHK). Kept, so a later merge can find it, unless
               --promote-unmerged copies it to results/ like any other run.
  keep         a run directory with no summary.json (a crashed run's summary.json.partial), or
               with anything in it but regular files. Kept and listed.

Trashing. Units verified redundant against what is ON DISK after promotion (the plan is
recomputed from scratch) are moved, paths intact, into one holding directory
`results/_cluster_staging/_condensed-<now>/`, empty directories left behind are removed, and the
holding directory goes to the trash in one `gio trash` -- so a restore is one action and the
trash never holds more than one entry per run of this script.

    .venv/bin/python cluster/condense_staging.py                    # dry run: the plan, no changes
    .venv/bin/python cluster/condense_staging.py --apply            # what collect_results.sh runs
    .venv/bin/python cluster/condense_staging.py --apply --promote-unmerged --archive-conflicts
"""
import argparse
import filecmp
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = os.path.join("results", "_cluster_staging")
ARCHIVE = os.path.join("results", "archive_staging")
LOGSTORE = os.path.join("results", "_solver_logs")
STUBSTORE = os.path.join("results", "_shard_stubs")
UNMERGED = os.path.join("results", "_unmerged_shards")
STUB = "summary_stub.json"
STAMP_RE = re.compile(r"^\d{8}-\d{6}$")
SHARD_RE = re.compile(r"_shard\d+of\d+$")
SUMMARY = "summary.json"


def collections():
    return sorted(d for d in os.listdir(STAGING)
                  if STAMP_RE.match(d) and os.path.isdir(os.path.join(STAGING, d))
                  and not os.path.islink(os.path.join(STAGING, d)))


def destination(rel):
    """`<stamp>/results/X` -> `results/X`; `<stamp>/<top>/X` -> `results/_cluster_<top>/X`."""
    parts = rel.split(os.sep)
    if parts[1] == "results":
        return os.path.join("results", *parts[2:])
    return os.path.join("results", f"_cluster_{parts[1]}", *parts[2:])


def is_run_dir(path):
    parts = os.path.relpath(path, STAGING).split(os.sep)
    return len(parts) == 5 and parts[1] == "results" and parts[3] == "benchmark"


def units(stamp):
    """Yield (kind, staged path) for every unit of one collection: run directories, and every
    other non-directory entry (regular files, symlinks -- including symlinks to directories,
    which os.walk lists under dirnames and does not follow -- and anything else)."""
    for dirpath, dirnames, filenames in os.walk(os.path.join(STAGING, stamp)):
        if is_run_dir(dirpath):
            dirnames[:] = []
            yield "run", dirpath
            continue
        for d in [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            yield "file", os.path.join(dirpath, d)
        for f in filenames:
            yield "file", os.path.join(dirpath, f)


def robot_tag(path):
    """(robot, tag) of a staged run directory or a `.../<robot>/benchmark/<tag>/summary.json`."""
    parts = path.rstrip(os.sep).split(os.sep)
    if parts[-1] == SUMMARY:
        parts = parts[:-1]
    i = len(parts) - 1 - parts[::-1].index("benchmark")
    return parts[i - 1], parts[i + 1]


def regular(path):
    return stat.S_ISREG(os.lstat(path).st_mode)


def same_file(src, dst):
    """Byte-identical regular files, or symlinks (wandb's, often dangling) to the same target."""
    if os.path.islink(src):
        return os.path.islink(dst) and os.readlink(src) == os.readlink(dst)
    return (regular(src) and os.path.lexists(dst) and regular(dst)
            and filecmp.cmp(src, dst, shallow=False))


def is_prefix(src, ref):
    if not (regular(src) and os.path.lexists(ref) and regular(ref)):
        return False
    n = os.path.getsize(src)
    if n > os.path.getsize(ref):
        return False
    with open(src, "rb") as a, open(ref, "rb") as b:
        while n > 0:
            k = min(n, 1 << 20)
            if a.read(k) != b.read(k):
                return False
            n -= k
    return True


def run_ok(path):
    """A run directory this script can reason about: a summary.json, and only regular files."""
    return (os.path.lexists(os.path.join(path, SUMMARY))
            and all(regular(os.path.join(path, f)) for f in os.listdir(path)))


# ------------------------------------------------------------------------------ the log store
def log_moves(path):
    """[(staged log, store path, already stored identically)] for one run directory."""
    robot, tag = robot_tag(path)
    stamp = os.path.relpath(path, STAGING).split(os.sep)[0]
    out = []
    for f in sorted(os.listdir(path)):
        if f == SUMMARY:
            continue
        src = os.path.join(path, f)
        first = os.path.join(LOGSTORE, robot, tag, f)
        second = os.path.join(LOGSTORE, robot, tag, f"{stamp}.{f}")
        if same_file(src, first) or same_file(src, second):
            out.append((src, first if same_file(src, first) else second, True))
        else:
            out.append((src, second if os.path.lexists(first) else first, False))
    return out


def logs_preserved(path):
    """Every file of the run directory but its summary.json has an identical copy in the store."""
    return all(stored for _, _, stored in log_moves(path))


def stub_bytes(path):
    with open(os.path.join(path, SUMMARY)) as fh:
        s = json.load(fh)
    s["records_omitted"] = {arm: len(r) for arm, r in (s.pop("records", None) or {}).items()}
    return json.dumps(s, indent=1).encode()          # key order kept (the arms' order matters)


def stub_slot(path, data):
    """(store path, already stored identically) for one shard's stub."""
    robot, tag = robot_tag(path)
    stamp = os.path.relpath(path, STAGING).split(os.sep)[0]
    first = os.path.join(STUBSTORE, robot, tag, STUB)
    second = os.path.join(STUBSTORE, robot, tag, f"{stamp}.{STUB}")
    for p in (first, second):
        if os.path.lexists(p) and regular(p):
            with open(p, "rb") as fh:
                if fh.read() == data:
                    return p, True
    return (second if os.path.lexists(first) else first), False


def stub_preserved(path):
    return stub_slot(path, stub_bytes(path))[1]


def write_atomic(dst, data):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = os.path.join(os.path.dirname(dst), ".condense-tmp-" + os.path.basename(dst))
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.lexists(dst):
        os.unlink(tmp)
        raise SystemExit(f"refusing to overwrite {dst}")
    os.rename(tmp, dst)


def stash_stubs(runs, apply):
    n = nbytes = 0
    for path in runs:
        if not SHARD_RE.search(path):
            continue
        data = stub_bytes(path)
        dst, stored = stub_slot(path, data)
        if stored:
            continue
        n += 1
        nbytes += len(data)
        if apply:
            write_atomic(dst, data)
    return n, nbytes


def stash_logs(runs, apply):
    moved, nbytes = 0, 0
    for path in runs:
        for src, dst, stored in log_moves(path):
            if stored:
                continue
            if os.path.lexists(dst):
                raise SystemExit(f"refusing to overwrite {dst}")
            moved += 1
            nbytes += os.lstat(src).st_size
            if apply:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
    return moved, nbytes


# ------------------------------------------------------------------------- the record index
def record_keys(path, robot):
    """{(robot, arm, record hash)} of one summary.json, or None if it holds no records."""
    with open(path) as fh:
        s = json.load(fh)
    keys = set()
    for arm, recs in (s.get("records") or {}).items():
        for r in recs:
            keys.add((robot, arm, hashlib.sha1(
                json.dumps(r, sort_keys=True).encode()).digest()[:16]))
    return keys or None


class Index:
    """Record hashes of every summary.json under results/ outside staging, cached by mtime."""

    def __init__(self):
        self.keys, self.seen = set(), {}

    def refresh(self, extra=()):
        todo = []
        for dirpath, dirnames, filenames in os.walk("results"):
            if dirpath == "results":
                dirnames[:] = [d for d in dirnames if d != "_cluster_staging"]
            if SUMMARY in filenames and "benchmark" in dirpath.split(os.sep):
                p = os.path.join(dirpath, SUMMARY)
                todo.append((p, robot_tag(p)[0]))
        for p, robot in todo + list(extra):
            st = os.stat(p)
            sig = (st.st_mtime_ns, st.st_size)
            if self.seen.get(p) != sig:
                self.seen[p] = sig
                self.keys |= record_keys(p, robot) or set()

    def contains(self, path, robot):
        keys = record_keys(path, robot)
        return keys is not None and keys <= self.keys


# ----------------------------------------------------------------------------------- the plan
def contained(src, ref, kind):
    if kind == "run":
        return same_file(os.path.join(src, SUMMARY), os.path.join(ref, SUMMARY))
    return same_file(src, ref) or is_prefix(src, ref)


def exists_at(dst, kind):
    return os.path.lexists(os.path.join(dst, SUMMARY) if kind == "run" else dst)


def unit_size(path, kind):
    if kind == "run":
        return sum(os.lstat(os.path.join(path, f)).st_size for f in os.listdir(path))
    return os.lstat(path).st_size


def plan(index, args, anticipate=False):
    """{unit path: [verdict, destination, kind, size]} over every collection.

    `anticipate` counts the staged files this plan is about to promote (and, with
    --archive-conflicts, archive) as if already under results/ -- needed by the FIRST plan, or a
    shard whose merged run is still in staging looks unmerged and --promote-unmerged copies it
    too (it did, 2026-11-09: 1,407 duplicate shards). The second plan, the one that decides what
    is trashed, never anticipates and runs on a fresh Index: disk only."""
    out, by_dest, pending = {}, {}, []
    for stamp in collections():
        for kind, path in units(stamp):
            dest = destination(os.path.relpath(path, STAGING))
            row = [None, dest, kind, unit_size(path, kind)]
            out[path] = row
            if kind == "run" and not run_ok(path):
                row[0] = "keep"
            elif kind == "run" and SHARD_RE.search(path):
                row[0] = "shard?"
            else:
                by_dest.setdefault(dest, []).append(path)

    for dest, paths in by_dest.items():
        kind, newest = out[paths[0]][2], paths[-1]   # collections() and units() are sorted
        if exists_at(dest, kind):
            for p in paths:
                out[p][0] = "redundant" if contained(p, dest, kind) else "conflict"
        else:
            out[newest][0] = "promote"
            if kind == "run":
                pending.append((os.path.join(newest, SUMMARY), robot_tag(newest)[0]))
            for p in paths[:-1]:
                out[p][0] = "redundant" if contained(p, newest, kind) else "conflict"

    if anticipate and args.archive_conflicts:
        pending += [(os.path.join(p, SUMMARY), robot_tag(p)[0])
                    for p, r in out.items() if r[0] == "conflict" and r[2] == "run"]
    index.refresh(pending if anticipate else ())
    for p, row in out.items():
        if row[0] != "shard?":
            continue
        dest, kind = row[1], row[2]
        if index.contains(os.path.join(p, SUMMARY), robot_tag(p)[0]):
            row[0] = "redundant"
        elif exists_at(dest, kind):
            row[0] = "redundant" if contained(p, dest, kind) else "conflict"
        else:
            row[0] = "unmerged"
    if args.promote_unmerged:
        ## to results/_unmerged_shards/, newest collection's copy first, as for whole units
        by_dest = {}
        for p, row in out.items():
            if row[0] == "unmerged":
                row[1] = os.path.join(UNMERGED, os.path.relpath(row[1], "results"))
                by_dest.setdefault(row[1], []).append(p)
        for dest, paths in by_dest.items():
            newest = paths[-1]
            out[newest][0] = "conflict" if exists_at(dest, "run") else "promote"
            for p in paths[:-1]:
                out[p][0] = "redundant" if contained(p, newest, "run") else "conflict"
    return out


def copy_unit(src, dst, kind):
    """Copy to a temporary name beside `dst`, then rename: an interrupted copy never leaves a
    truncated file at `dst` (which a later run would take for a conflict)."""
    if kind == "run":
        src, dst = os.path.join(src, SUMMARY), os.path.join(dst, SUMMARY)
    if os.path.lexists(dst):
        raise SystemExit(f"refusing to overwrite {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = os.path.join(os.path.dirname(dst), ".condense-tmp-" + os.path.basename(dst))
    if os.path.lexists(tmp):
        os.unlink(tmp)
    shutil.copy2(src, tmp, follow_symlinks=False)
    if not os.path.islink(tmp):
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
    os.rename(tmp, dst)


def archived(path, kind):
    target = os.path.join(ARCHIVE, os.path.relpath(path, STAGING))
    return target, exists_at(target, kind) and contained(path, target, kind)


def tally(p, label):
    rows = {}
    for v, _, _, size in p.values():
        n, b = rows.get(v, (0, 0))
        rows[v] = (n + 1, b + size)
    print(f"{label}:")
    for v in ("redundant", "promote", "conflict", "unmerged", "keep"):
        n, b = rows.get(v, (0, 0))
        print(f"  {v:<10} {n:7d} units  {b / 2**20:9.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="stash logs, promote, then trash")
    ap.add_argument("--promote-unmerged", action="store_true",
                    help="copy shards that were never merged into results/ (closed stages only: "
                         "an in-flight stage's shards must stay in staging for the merger)")
    ap.add_argument("--archive-conflicts", action="store_true",
                    help=f"copy staged units that differ from results/ to {ARCHIVE}/<stamp>/")
    ap.add_argument("--list", action="store_true", help="also list every promoted unit")
    args = ap.parse_args()
    os.chdir(REPO)
    if not os.path.isdir(STAGING):
        print("no staging directory")
        return 0

    p = plan(Index(), args, anticipate=True)
    tally(p, "plan")
    for path, (v, dest, kind, size) in sorted(p.items()):
        if v in ("conflict", "unmerged", "keep") or (args.list and v == "promote"):
            print(f"  {v:<9} {os.path.relpath(path, STAGING)}")
    if not args.apply:
        ## Logs and stubs move only for run directories that leave staging (in-flight shards keep
        ## theirs); in the dry run that is every redundant or promoted run, archived conflicts too.
        leaving = sorted(k for k, r in p.items() if r[2] == "run" and (
            r[0] in ("redundant", "promote") or (r[0] == "conflict" and args.archive_conflicts)))
        n, b = stash_logs(leaving, apply=False)
        print(f"logs to move to {LOGSTORE}/: {n} files, {b / 2**20:.1f} MB")
        n, b = stash_stubs(leaving, apply=False)
        print(f"shard stubs to write to {STUBSTORE}/: {n} files, {b / 2**20:.1f} MB")
        print("DRY RUN: nothing moved, copied or trashed")
        return 0

    for path, (v, dest, kind, size) in sorted(p.items()):
        if v == "promote":
            copy_unit(path, dest, kind)
        elif v == "conflict" and args.archive_conflicts:
            target, done = archived(path, kind)
            if not done and exists_at(target, kind):
                print(f"  kept, archive copy differs: {os.path.relpath(path, STAGING)}")
            elif not done:
                copy_unit(path, target, kind)

    ## Re-plan from scratch, from disk: only what is now verifiably redundant leaves staging.
    p = plan(Index(), args)                      # fresh index: no anticipated keys survive
    if args.archive_conflicts:
        for path, row in p.items():
            if row[0] == "conflict" and archived(path, row[2])[1]:
                row[0] = "redundant"
    tally(p, "after promotion, verified on disk")
    leaving = sorted(k for k, r in p.items() if r[0] == "redundant" and r[2] == "run")
    n, b = stash_logs(leaving, apply=True)
    print(f"logs moved to {LOGSTORE}/: {n} files, {b / 2**20:.1f} MB")
    n, b = stash_stubs(leaving, apply=True)
    print(f"shard stubs written to {STUBSTORE}/: {n} files, {b / 2**20:.1f} MB")
    doomed = [k for k, r in p.items() if r[0] == "redundant"
              and (r[2] != "run" or (logs_preserved(k)
                                     and (not SHARD_RE.search(k) or stub_preserved(k))))]
    held_back = sum(1 for r in p.values() if r[0] == "redundant") - len(doomed)
    if held_back:
        print(f"WARNING: {held_back} redundant unit(s) kept: logs or stub not verified in the store")
    if not doomed:
        print("nothing to trash")
        return 0
    hold = os.path.join(STAGING, "_condensed-" + time.strftime("%Y%m%d-%H%M%S"))
    for k in doomed:
        target = os.path.join(hold, os.path.relpath(k, STAGING))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.rename(k, target)
    for stamp in collections():                   # the directories trashed units left behind
        subprocess.run(["find", os.path.join(STAGING, stamp), "-depth", "-type", "d",
                        "-empty", "-delete"], check=True)
    subprocess.run(["gio", "trash", "--", hold], check=True)
    print(f"trashed {len(doomed)} unit(s), {sum(p[k][3] for k in doomed) / 2**30:.2f} GB, as "
          f"one entry ({os.path.basename(hold)}); empty the trash only after a check. "
          f"Left in staging: {len(collections())} collection(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
