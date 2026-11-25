"""The obstacle campaign wired into the four record benchmark scripts.

`--prepare-only` runs a script's whole front end (grid, obstacle sets, certificates, solve
diagrams, guesses' collision values) and exits without compiling or solving; these tests run
it as a subprocess, the way a cluster item would.

fast  iiwa pose and mug at 2 targets x 2 guesses: rung 0 (`--obstacles none`) and rungs 8 / 16
      share one grid_hash; rung 0's fingerprint is the plain `scene_fingerprint`, the rungs'
      differ from it and each other; rung 8's centres are the head of rung 16's; every
      certificate holds; a pillar scene redraws the targets and records itself; bad flag
      combinations are refused; the start's collision value is a per-cell record field.
slow  ALL EIGHT record rows (panda / iiwa / screw7_p050 / soft12 x mug / pose) at the record's
      grid (60 x 8, seed 1, the manifests' scene arguments) reproduce the record's grid_hash and
      scene_fingerprint at rung 0 -- the bit-identity of rung 0 to the record -- and one
      end-to-end solve on an obstacle scene completes and records the metadata.

Checkpoints are gitignored; a worktree falls back to the main checkout's (read-only). The grid
does not depend on the chart, so a stand-in checkpoint of the right robot reproduces the hash.
"""

import json
import os
import shutil
import subprocess
import sys
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import _tiers                                                             # noqa: E402

from src.benchmark import MUG_MODEL_FILE, scene_fingerprint               # noqa: E402
from src.obstacles import per_target as pt                                # noqa: E402
from src.target_screening import SceneFile                                # noqa: E402
from src.utils import RepoDir                                             # noqa: E402

ROOT = RepoDir()
GOLDEN = json.load(open(os.path.join(ROOT, "tests", "data", "record_scene_fingerprints.json")))

## row -> (script, robot-specific args, record grid_hash, SCENES key)
RECORD = {
    ("panda", "mug"): ("panda/panda_benchmark.py",
                       ["--checkpoint", "models/panda/panda__n6__step620000.pkl"],
                       "734d0f70e7bb", "panda"),
    ("panda", "pose"): ("panda/panda_benchmark.py",
                        ["--checkpoint", "models/panda/panda__n6__step620000.pkl"],
                        "f872dcf34dd1", "panda"),
    ("iiwa", "mug"): ("iiwa/iiwa_benchmark.py",
                      ["--checkpoint", "models/iiwa14/iiwa14__n4__step620000.pkl"],
                      "b8a7b81e4972-mug", "iiwa"),
    ("iiwa", "pose"): ("iiwa/iiwa_benchmark.py",
                       ["--checkpoint", "models/iiwa14/iiwa14__n4__step620000.pkl"],
                       "4361e3d807f1-pose", "iiwa"),
    ("screw7_p050", "mug"): ("screw_arm/screw_arm_benchmark.py",
                             ["--robot", "screw7_p050", "--checkpoint",
                              "models/screw7_p050/screw7_p050__n6__step620000.pkl"],
                             "519d384f5f8f-mug", "screw7_p050"),
    ("screw7_p050", "pose"): ("screw_arm/screw_arm_benchmark.py",
                              ["--robot", "screw7_p050", "--checkpoint",
                               "models/screw7_p050/screw7_p050__n6__step620000.pkl"],
                              "4918db68403b-pose", "screw7_p050"),
    ("soft12", "mug"): ("soft_arm/soft_arm_benchmark.py",
                        ["--rung", "soft12", "--checkpoint",
                         "models/soft12/soft12__n6__step620000.pkl"],
                        "c5f821104d29-mug", "soft12"),
    ("soft12", "pose"): ("soft_arm/soft_arm_benchmark.py",
                         ["--rung", "soft12", "--checkpoint",
                          "models/soft12/soft12__n6__step620000.pkl"],
                         "5fe0e6d06944-pose", "soft12"),
}
## The record's scene arguments (cluster/manifest_stagePAPER.txt), minus solver and arms.
RECORD_SCENE_ARGS = ["--targets", "60", "--guesses", "8", "--seed", "1",
                     "--target-placement", "shelf", "--scene", "hardened",
                     "--shelf-inset", "0.1"]


def _main_checkout():
    out = subprocess.run(["git", "-C", ROOT, "rev-parse", "--git-common-dir"],
                         capture_output=True, text=True).stdout.strip()
    return os.path.dirname(os.path.abspath(os.path.join(ROOT, out))) if out else ROOT


def _ckpt(rel):
    """The checkpoint path as a script should get it: here if present, else the main checkout's.
    The iiwa record chart (n4) is not always on a laptop; `n6` has the same grid."""
    for base in (ROOT, _main_checkout()):
        for cand in (rel, rel.replace("iiwa14__n4__", "iiwa14__n6__")):
            if os.path.exists(os.path.join(base, cand)):
                return os.path.join(base, cand)
    raise SystemExit("SKIP-needed checkpoint missing: %s" % rel)


def _run(robot, task, extra, scene_args=None, expect_ok=True):
    script, rargs, _, _ = RECORD[(robot, task)]
    rargs = list(rargs)
    rargs[rargs.index("--checkpoint") + 1] = _ckpt(rargs[rargs.index("--checkpoint") + 1])
    cmd = [sys.executable, os.path.join(ROOT, "scripts", script), "--task", task] + rargs \
        + (scene_args if scene_args is not None else []) + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not expect_ok:
        return r
    assert r.returncode == 0, "\n".join(r.stderr.splitlines()[-15:])
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("wrote ")][-1]
    path = line[len("wrote "):]
    with open(path) as fh:
        report = json.load(fh)
    shutil.rmtree(os.path.dirname(path), ignore_errors=True)
    return report


def _plain_fp(robot, task, scene="hardened"):
    return scene_fingerprint(SceneFile(robot, task, scene),
                             extra_models=[MUG_MODEL_FILE] if task == "mug" else [])


SMALL = ["--targets", "2", "--guesses", "2", "--seed", "1", "--prepare-only"]


def test_prepare_only_rungs_pose():
    r0 = _run("iiwa", "pose", SMALL + ["--obstacles", "none"])
    r8 = _run("iiwa", "pose", SMALL + ["--obstacles", "per_target", "--obstacle-count", "8"])
    r16 = _run("iiwa", "pose", SMALL + ["--obstacles", "per_target", "--obstacle-count", "16"])
    assert r0["grid_hash"] == r8["grid_hash"] == r16["grid_hash"]
    assert r0["scene_fingerprint"] == _plain_fp("iiwa", "pose")
    assert len({r0["scene_fingerprint"], r8["scene_fingerprint"], r16["scene_fingerprint"]}) == 3
    assert r0["obstacles"] == {"family": "none"}
    o8, o16 = r8["obstacles"], r16["obstacles"]
    assert (o8["n"], o16["n"]) == (8, 16) and o8["rule"] == pt.RULE_VERSION
    assert o8["achieved"] == [8, 8] and o16["achieved"] == [16, 16] and not o16["short_targets"]
    for t in range(2):
        assert len(o8["centres"][t]) == 8 and len(o16["centres"][t]) == 16
        assert o8["centres"][t] == o16["centres"][t][:8]       # prefix-nested rungs
    assert all(c["ok"] for c in o8["certificates"] + o16["certificates"])
    assert len(r8["guess_collision_values"]) == 2 and len(r8["guess_collision_values"][0]) == 2


def test_prepare_only_rungs_mug():
    r0 = _run("iiwa", "mug", SMALL + ["--obstacles", "none"])
    r8 = _run("iiwa", "mug", SMALL + ["--obstacles", "per_target", "--obstacle-count", "8"])
    assert r0["grid_hash"] == r8["grid_hash"]
    assert r0["scene_fingerprint"] == _plain_fp("iiwa", "mug")
    assert r8["scene_fingerprint"] != r0["scene_fingerprint"]
    assert all(c["ok"] for c in r8["obstacles"]["certificates"])
    assert r8["obstacles"]["achieved"] == [8, 8]


def test_pillar_scene_prepare_only():
    h = _run("iiwa", "pose", SMALL)
    p = _run("iiwa", "pose", SMALL + ["--scene", "pillars_r080"])
    assert p["scene_fingerprint"] == _plain_fp("iiwa", "pose", "pillars_r080")
    assert p["grid_hash"] != h["grid_hash"]            # targets are redrawn on the new scene
    assert p["obstacles"] == {"family": "none"}


def test_flag_validation():
    bad = [["--obstacles", "none", "--obstacle-count", "8"],
           ["--obstacles", "per_target", "--obstacle-count", "0"]]
    for extra in bad:
        r = _run("iiwa", "pose", SMALL + extra, expect_ok=False)
        assert r.returncode != 0 and "obstacle" in r.stderr, (extra, r.stderr[-300:])


def test_start_collision_value_is_a_cell_field():
    """start_diagnostics records the program's own collision value at the start, via the same
    plant-vector path QAndPose uses; no other field changes."""
    import numpy as np
    from src import benchmark as bm
    from src.generic_program import ProgramOptions
    from src.iiwa_program import Iiwa14IKProgramNumerical
    from src.utils import BuildEnv, HiddenPrints
    with HiddenPrints():
        d = BuildEnv(meshcat=None, directives_file=SceneFile("iiwa", "pose"))
        prog = Iiwa14IKProgramNumerical(d, options=ProgramOptions(visualize=False))
        q = np.array([0.1, 0.4, -0.2, -1.2, 0.3, 0.8, 0.0])
        prog.create_prog(np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]))
        prog.clip_distance = prog.SetStartFromQ(q)
    out = bm.start_diagnostics(prog, q)
    assert set(out) >= {"clip_distance", "start_q_error", "start_collision_value"}
    expect = float(np.asarray(prog.collision_free_constraint_eval.Eval(q)).flatten()[0])
    assert abs(out["start_collision_value"] - expect) < 1e-9
    assert out["start_q_error"] < 1e-6


## --------------------------------------------------------------------------- slow ----

def test_rung0_is_the_record(robot, task):
    """Rung 0 at the record's grid reproduces the record's grid_hash AND scene_fingerprint."""
    _, _, grid_hash, key = RECORD[(robot, task)]
    r = _run(robot, task, ["--prepare-only", "--obstacles", "none"], RECORD_SCENE_ARGS)
    assert r["grid_hash"] == grid_hash, (robot, task, r["grid_hash"], grid_hash)
    assert r["scene_fingerprint"] == GOLDEN["%s/%s" % (key, task)], (robot, task)
    print("   %s/%s: grid_hash %s  scene_fingerprint %s  OK"
          % (robot, task, r["grid_hash"], r["scene_fingerprint"][:12]))


def test_end_to_end_obstacle_solve():
    """iiwa pose, one cell, IPOPT joint space, 16 obstacles: completes, carries the obstacle
    metadata, and the collision gate ran on the obstacle scene."""
    script, rargs, _, _ = RECORD[("iiwa", "pose")]
    rargs = list(rargs)
    rargs[rargs.index("--checkpoint") + 1] = _ckpt(rargs[rargs.index("--checkpoint") + 1])
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", script), "--task", "pose"] + rargs
        + ["--targets", "1", "--guesses", "1", "--seed", "1", "--arms", "numerical",
           "--solver", "ipopt", "--wall-time", "20", "--obstacles", "per_target",
           "--obstacle-count", "16"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, "\n".join(r.stderr.splitlines()[-15:])
    path = [ln for ln in r.stdout.splitlines() if ln.startswith("wrote ")][-1][len("wrote "):]
    s = json.load(open(path))
    try:
        ob = s["metadata"]["obstacles"]
        assert ob["family"] == "per_target" and ob["n"] == 16 and len(ob["centres"][0]) == 16
        rec = s["records"]["numerical"][0]
        assert "start_collision_value" in rec
        det = rec["detail"]
        print("   record:", dict(feasible=rec["feasible"], fail_reason=rec["fail_reason"],
                                 min_distance=det.get("min_distance"),
                                 pair=det.get("min_distance_pair"),
                                 collision_value=det.get("collision_value"),
                                 start_collision_value=rec["start_collision_value"]))
        assert det.get("min_distance_pair") is not None
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


GROUPS = {
    "fast": [test_prepare_only_rungs_pose, test_prepare_only_rungs_mug,
             test_pillar_scene_prepare_only, test_flag_validation,
             test_start_collision_value_is_a_cell_field],
    "slow": [partial(test_rung0_is_the_record, r, t) for (r, t) in RECORD]
            + [test_end_to_end_obstacle_solve],
    "e2e": [test_end_to_end_obstacle_solve],
}


def main():
    _tiers.run(GROUPS)
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
