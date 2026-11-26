"""Geometry-only probe of the obstacle campaign's scenes: no solve, no compile.

For each record row (four robots x grasp/pose) and each scene configuration -- rung 0, the OBST
rungs (`--obstacles per_target --obstacle-count N`) and the pillar scenes (`--scene
pillars_rXXX`) -- run the row's own benchmark script with `--prepare-only` on the record's grid
arguments (60 targets x 8 guesses, seed 1) and collect its `prepare.json`.  What it measures is
a property of the SCENE, not of any method, which is why it may be run before any result:

* OBST: achieved obstacles per target, cap hits, rejection reasons, the witness certificates
  (constraint value at q*, robot-obstacle clearance);
* PILLAR: whether target sampling stays viable on the new scene (acceptance; a guard trip
  shows up as a failed run) and the grid it draws;
* both: the share of guesses that START in collision on their target's solve scene
  (`guess_collision_values` > 1; guesses are drawn on the base scene by design for OBST).

Runs ONE process at a time, in the foreground, with whatever wrapper the caller uses (the local
rules: `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`, one core, CPU only).

    python scripts/probe_obstacles.py                  # every row x every configuration
    python scripts/probe_obstacles.py --rows iiwa:pose --configs obst8,pillars_r040
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT)

## row -> (script, robot args); the record's charts (the grid does not depend on the chart, so
## a stand-in checkpoint of the right robot reproduces the hash when the record's is absent).
ROWS = {
    "panda": ("panda/panda_benchmark.py", ["--checkpoint", "models/panda/panda__n6__step620000.pkl"]),
    "iiwa": ("iiwa/iiwa_benchmark.py", ["--checkpoint", "models/iiwa14/iiwa14__n4__step620000.pkl"]),
    "screw7_p050": ("screw_arm/screw_arm_benchmark.py",
                    ["--robot", "screw7_p050", "--checkpoint",
                     "models/screw7_p050/screw7_p050__n6__step620000.pkl"]),
    "soft12": ("soft_arm/soft_arm_benchmark.py",
               ["--rung", "soft12", "--checkpoint", "models/soft12/soft12__n6__step620000.pkl"]),
}
GRID = ["--targets", "60", "--guesses", "8", "--seed", "1", "--target-placement", "shelf",
        "--shelf-inset", "0.1", "--prepare-only"]
CONFIGS = {
    "rung0": ["--scene", "hardened", "--obstacles", "none"],
    "obst8": ["--scene", "hardened", "--obstacles", "per_target", "--obstacle-count", "8"],
    "obst16": ["--scene", "hardened", "--obstacles", "per_target", "--obstacle-count", "16"],
    "obst32": ["--scene", "hardened", "--obstacles", "per_target", "--obstacle-count", "32"],
    "obst64": ["--scene", "hardened", "--obstacles", "per_target", "--obstacle-count", "64"],
    "pillars_r040": ["--scene", "pillars_r040", "--obstacles", "none"],
    "pillars_r080": ["--scene", "pillars_r080", "--obstacles", "none"],
    "pillars_r120": ["--scene", "pillars_r120", "--obstacles", "none",
                     "--max-target-rejections", "500000"],   # Thomas, 2026-11-26
}


def _main_checkout():
    out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--git-common-dir"],
                         capture_output=True, text=True).stdout.strip()
    return os.path.dirname(os.path.abspath(os.path.join(ROOT, out))) if out else ROOT


def _ckpt(rel):
    for base in (ROOT, _main_checkout()):
        for cand in (rel, rel.replace("iiwa14__n4__", "iiwa14__n6__")):
            if os.path.exists(os.path.join(base, cand)):
                return os.path.join(base, cand)
    raise SystemExit("checkpoint missing: %s" % rel)


def run_one(robot, task, config, out_dir):
    script, rargs = ROWS[robot]
    rargs = list(rargs)
    rargs[rargs.index("--checkpoint") + 1] = _ckpt(rargs[rargs.index("--checkpoint") + 1])
    tag = "probe_obst_%s_%s_%s" % (robot, task, config)
    cmd = ([sys.executable, os.path.join(ROOT, "scripts", script), "--task", task, "--tag", tag]
           + rargs + GRID + CONFIGS[config])
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    seconds = time.time() - t0
    entry = dict(robot=robot, task=task, config=config, seconds=seconds, returncode=r.returncode)
    if r.returncode != 0:
        entry["error"] = "\n".join(r.stderr.splitlines()[-6:])
        return entry
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("wrote ")][-1]
    with open(line[len("wrote "):]) as fh:
        report = json.load(fh)
    entry.update(summarise(report))
    with open(os.path.join(out_dir, tag + ".json"), "w") as fh:
        json.dump(report, fh)
    return entry


def summarise(report):
    out = dict(grid_hash=report.get("grid_hash"),
               scene_fingerprint=(report.get("scene_fingerprint") or "")[:12])
    ts = report.get("target_stats") or {}
    if ts:
        out["target_accept_rate"] = ts.get("accept_rate")
        out["target_draws"] = ts.get("drawn")
    g = report.get("guess_collision_values")
    if g:
        vals = np.array([v for row in g if row for v in row])
        out["guesses_in_collision"] = int(np.sum(vals > 1.0))
        out["guesses"] = int(vals.size)
    obst = report.get("obstacles") or {}
    if obst.get("family") == "per_target":
        ach = np.asarray(obst["achieved"])
        out["achieved_min"] = int(ach.min())
        out["achieved_median"] = float(np.median(ach))
        out["short_targets"] = len(obst.get("short_targets", []))
        out["cap_hits"] = len(obst.get("cap_hits", []))
        certs = obst.get("certificates") or []
        certs = [c for c in certs if c]
        if certs:
            out["cert_ok"] = int(sum(bool(c.get("ok")) for c in certs))
            out["cert_value_max"] = float(max(c["constraint_value"] for c in certs))
            out["clearance_min"] = float(min(c["witness_obstacle_clearance"] for c in certs))
        cands = np.asarray(obst.get("candidates", []), dtype=float)
        if cands.size:
            out["candidates_median"] = float(np.median(cands))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", default=",".join("%s:%s" % (r, t) for r in ROWS
                                               for t in ("mug", "pose")))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "_probe_obstacles"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    entries = json.load(open(summary_path)) if os.path.exists(summary_path) else []
    done = {(e["robot"], e["task"], e["config"]) for e in entries if e.get("returncode") == 0}
    for row in args.rows.split(","):
        robot, task = row.split(":")
        for config in args.configs.split(","):
            if (robot, task, config) in done:
                continue
            e = run_one(robot, task, config, args.out)
            print(json.dumps(e), flush=True)
            entries = [x for x in entries
                       if (x["robot"], x["task"], x["config"]) != (robot, task, config)]
            entries.append(e)
            with open(summary_path, "w") as fh:
                json.dump(entries, fh, indent=1)


if __name__ == "__main__":
    main()
