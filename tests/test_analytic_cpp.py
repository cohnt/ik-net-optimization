"""The C++ backend of the analytic maps (`analytic_backend="cpp"`) against the Python reference.

Run by hand, one file at a time (builds `cpp/build/liblearnedik_analytic.so` first if it is
missing):

    python tests/test_analytic_cpp.py

What it pins:
  * VALUES: C++ against the Python map through the mixin's composition (RPY -> R, the pose
    offset, then `Solve`), on 2000 draws per chart -- Panda 4-branch, Panda 8-branch, iiwa --
    at both clip ladders: configurations uniform in the joint limits mapped to pose, psi and
    branch, a third left exact, a third nudged and a third pushed far enough that clips go
    ACTIVE. Against the AutoDiffXd path's values, which the C++ transliterates op for op,
    <= 1e-12 (measured: 0, bit for bit). Against the float path, no further than the
    reference is from ITSELF: numpy runs float matmuls through BLAS and object ones left to
    right, and near an arcsin seam at the tight clip the two Python paths differ by up to
    3.9e-12 (Panda 8, measured), so a flat 1e-12 there would test numpy, not the port.
  * JACOBIANS: the C++ d[q; u]/dx against Python AutoDiffXd (identity seed) <= 1e-10, on the
    same draws, on draws within 1e-9..1e-3 rad of the Panda's q6 wrap seam, and with the
    rows a clipped argument feeds required EXACTLY zero on both sides.
  * PROGRAMS: the four analytic programs (Panda and iiwa, pose and grasp) built once per
    backend; every binding evaluated by `prog.EvalBinding`, float and AutoDiffXd, at random
    points and branches, agrees <= 1e-11; so does `UnclippedStartError` (clip 0).
  * REFUSALS: an unknown backend name, a missing library (at construction, not at the first
    solve), and a branch outside the chart's set all raise.
  * Prints us per call, python against cpp, float and AutoDiffXd -- indicative only.
Tiers (tests/_tiers.py): the default `fast` runs in ~14 s on one CPU core; no `slow` group.
"""
import os
import subprocess
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _tiers                                                             # noqa: E402

from pydrake.all import (AutoDiffXd, ExtractGradient, ExtractValue,             # noqa: E402
                         InitializeAutoDiff, RigidTransform, RigidTransform_, RollPitchYaw,
                         RollPitchYaw_)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import src.analytic_cpp as analytic_cpp                                          # noqa: E402
if not os.path.isfile(analytic_cpp.LIBRARY_PATH):
    subprocess.run(["bash", os.path.join(REPO, "cpp", "build.sh")], check=True)

from src.analytic_cpp import AnalyticCppMap                                      # noqa: E402
from src.generic_program import ProgramOptions                                   # noqa: E402
from src.iiwa_analytic_ik import (Analytic_IK_7DoF, iiwa_limits_lower,           # noqa: E402
                                  iiwa_limits_upper)
from src.panda_analytic_ik import (Analytic_IK_Panda, Q6_WRAP_SEAM,              # noqa: E402
                                   panda_limits_lower, panda_limits_upper)
from src.utils import BuildEnv, GenerateDiagramWithMug, HiddenPrints             # noqa: E402

FAILURES = []
CHECKS = [0]

IIWA = Analytic_IK_7DoF()
PANDA = Analytic_IK_Panda()

#: The two clip ladders fielded (Thomas, 2026-10-12: measure both).
CLIPS = (1e-4, 1e-8)
N_DRAWS = 2000


def check(name, condition, detail=""):
    CHECKS[0] += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}\n          {detail}")
        FAILURES.append(name)


## ---------------- the Python reference, exactly as the mixin's _Solve runs it ---------------- ##

def python_solve(robot, x, offset, gc, clip):
    t = AutoDiffXd if isinstance(x[0], AutoDiffXd) else float
    X = RigidTransform_[t](RollPitchYaw_[t](x[3:6]), x[:3])
    R, p = X.rotation().matrix(), X.translation()
    R_off = offset.rotation().matrix()
    R, p = R @ R_off, p + R @ offset.translation()
    if robot == "panda":
        return PANDA.Solve(R, p, x[6], gc, clip_stepback=clip)
    return IIWA.Solve(R, p, gc, x[6], clip_stepback=clip)


class Chart:
    def __init__(self, name, robot, lo, hi, fk, branch, psi_of):
        self.name, self.robot, self.lo, self.hi = name, robot, lo, hi
        self.fk, self.branch, self.psi_of = fk, branch, psi_of


CHARTS = (
    Chart("panda4", "panda", panda_limits_lower, panda_limits_upper, PANDA.FK,
          lambda q: PANDA.gc(q, 2), lambda q: q[6]),
    Chart("panda8", "panda", panda_limits_lower, panda_limits_upper, PANDA.FK,
          lambda q: PANDA.gc(q, 3), lambda q: q[6]),
    Chart("iiwa8", "iiwa", iiwa_limits_lower, iiwa_limits_upper, IIWA.FK,
          IIWA.GC, IIWA.psi),
)

#: Rows of J = d[q; u]/dx (q0..q6, u0..u3) that a clipped argument zeroes OUTRIGHT: the joint
#: is +-arccos/arcsin(clip) (+ a constant), nothing else. {u index: q index}.
CLIP_ROWS = {"panda": {0: 3, 3: 1}, "iiwa": {0: 3, 2: 1, 3: 5}}


def vars_of(q, chart, offset):
    '''The decision variables whose composed pose is FK(q): x = pose(FK(q) @ offset^-1), psi.'''
    X = RigidTransform(chart.fk(q)) @ offset.inverse()
    return np.concatenate([X.translation(), X.rotation().ToRollPitchYaw().vector(),
                           [chart.psi_of(q)]])


def jacobian_python(robot, x, offset, gc, clip):
    q, u = python_solve(robot, InitializeAutoDiff(x)[:, 0], offset, gc, clip)
    y = np.concatenate([q, u])
    return ExtractValue(y).ravel(), ExtractGradient(y)


def jacobian_cpp(m, x, offset, gc, clip):
    q, u = m(InitializeAutoDiff(x)[:, 0], offset, gc, clip)
    y = np.concatenate([q, u])
    return ExtractValue(y).ravel(), ExtractGradient(y)


def test_map_parity():
    print(f"map parity: {N_DRAWS} draws per chart, both clip ladders, through the composition")
    rng = np.random.default_rng(11)
    for chart in CHARTS:
        m = AnalyticCppMap(chart.robot, PANDA if chart.robot == "panda" else IIWA)
        offset = RigidTransform(RollPitchYaw(*rng.uniform(-np.pi, np.pi, 3)),
                                rng.uniform(-0.15, 0.15, 3))
        dv_float = dv_ad = dv_self = dv_excess = dj = dj_rel = 0.0
        n_active = n_rows = 0
        rows_zero = calls_agree = True
        for k in range(N_DRAWS):
            clip = CLIPS[k % 2]
            q = rng.uniform(chart.lo, chart.hi)
            gc = chart.branch(q)
            x = vars_of(q, chart, offset)
            kind = k % 3
            if kind == 1:
                x = x + rng.normal(0, 1e-3, 7)
            elif kind == 2:
                x = x + rng.normal(0, [0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 1.0])
            q_py, u_py = python_solve(chart.robot, x, offset, gc, clip)
            q_c, u_c = m(x, offset, gc, clip)
            f_py, f_c = np.r_[q_py, u_py], np.r_[q_c, u_c]
            v_py, J_py = jacobian_python(chart.robot, x, offset, gc, clip)
            v_c, J_c = jacobian_cpp(m, x, offset, gc, clip)
            calls_agree &= np.array_equal(f_c, v_c)
            dv_ad = max(dv_ad, np.abs(v_py - f_c).max(), np.abs(v_py - v_c).max())
            ## The Python float path runs numpy's BLAS matmuls, its AutoDiffXd path object
            ## arithmetic, so the reference disagrees with ITSELF by a few ulps, amplified
            ## near an arccos/arcsin seam. The C++ follows the AutoDiffXd path; it may be as
            ## far from the float path as the reference is from itself, and no further.
            self_spread = np.abs(f_py - v_py).max()
            d = np.abs(f_py - f_c).max()
            dv_float, dv_self = max(dv_float, d), max(dv_self, self_spread)
            dv_excess = max(dv_excess, d - max(1e-12, self_spread))
            dj = max(dj, np.abs(J_py - J_c).max())
            dj_rel = max(dj_rel, np.abs(J_py - J_c).max() / max(1.0, np.abs(J_py).max()))
            active = np.abs(u_c) > 1 - clip
            n_active += bool(active.any())
            for ui, qi in CLIP_ROWS[chart.robot].items():
                if active[ui]:
                    n_rows += 1
                    rows_zero &= (not J_py[qi].any()) and (not J_c[qi].any())
        print(f"    {chart.name}: max |dvalue| against the AutoDiffXd path {dv_ad:.2e}, "
              f"against the float path {dv_float:.2e} (the reference's own float-v-AD spread "
              f"{dv_self:.2e}); max |dJ| {dj:.2e} (relative {dj_rel:.2e}); "
              f"{n_active} draws with an active clip")
        check(f"{chart.name}: q, u agree with the Python AutoDiffXd path's values <= 1e-12 "
              f"(both C++ calls)", dv_ad <= 1e-12, dv_ad)
        check(f"{chart.name}: the C++ value-only and Jacobian calls agree bit for bit",
              calls_agree)
        check(f"{chart.name}: q, u agree with the Python float path <= max(1e-12, the "
              f"reference's own float-v-AD spread)", dv_excess <= 0.0, dv_excess)
        check(f"{chart.name}: J agrees with Python AutoDiffXd <= 1e-10", dj <= 1e-10, dj)
        check(f"{chart.name}: >= 10% of draws have an active clip", n_active >= N_DRAWS // 10,
              n_active)
        check(f"{chart.name}: every clipped joint's row is exactly zero on both sides "
              f"({n_rows} rows)", rows_zero and n_rows > 0, n_rows)


def test_q6_seam():
    print("Panda: draws within 1e-9..1e-3 rad of the q6 wrap seam")
    rng = np.random.default_rng(12)
    m = AnalyticCppMap("panda", PANDA)
    offset = RigidTransform()
    dv = dv_excess = dj = 0.0
    sides = set()
    for k in range(400):
        q = rng.uniform(panda_limits_lower, panda_limits_upper)
        delta = 10 ** rng.uniform(-9, -3) * (1 if k % 2 else -1)
        q[5] = Q6_WRAP_SEAM + 2 * np.pi * (k % 4 >= 2) + delta  # both representatives
        gc = PANDA.gc(q, 3)
        x = vars_of(q, CHARTS[1], offset)
        f_py = np.r_[python_solve("panda", x, offset, gc, 1e-8)]
        f_c = np.r_[m(x, offset, gc, 1e-8)]
        sides.add(bool(f_c[5] > Q6_WRAP_SEAM + np.pi))
        v_py, J_py = jacobian_python("panda", x, offset, gc, 1e-8)
        _, J_c = jacobian_cpp(m, x, offset, gc, 1e-8)
        dv = max(dv, np.abs(v_py - f_c).max())
        dv_excess = max(dv_excess, np.abs(f_py - f_c).max()
                        - max(1e-12, np.abs(f_py - v_py).max()))
        dj = max(dj, np.abs(J_py - J_c).max())
    print(f"    max |dvalue| against the AutoDiffXd path {dv:.2e}, max |dJ| {dj:.2e}")
    check("seam draws: values agree with the AutoDiffXd path <= 1e-12 (the same "
          "representative)", dv <= 1e-12, dv)
    check("seam draws: no further from the float path than the reference is from itself",
          dv_excess <= 0.0, dv_excess)
    check("seam draws: J agrees <= 1e-10", dj <= 1e-10, dj)
    check("seam draws land on both sides of the window", len(sides) == 2, sides)


def test_refusals():
    print("refusals")
    try:
        ProgramOptions(analytic_backend="fortran")
        check("an unknown analytic_backend is refused", False, "constructed")
    except ValueError:
        check("an unknown analytic_backend is refused", True)
    check("the default backend is the Python reference",
          ProgramOptions().analytic_backend == "python")
    missing = os.path.join(REPO, "cpp", "build", "does_not_exist.so")
    try:
        AnalyticCppMap("panda", library_path=missing)
        check("a missing library raises", False, "loaded")
    except FileNotFoundError as exc:
        check("a missing library raises, naming the build script", "cpp/build.sh" in str(exc),
              exc)
    ## A library built from other sources (the cluster keeps cpp/build/ across restaging) is
    ## refused by its compiled-in source hash: copy the real library to a fresh path (so the
    ## per-path cache does not answer) and make the sources the loader hashes differ.
    import shutil
    import tempfile
    import src.analytic_cpp as acpp
    with tempfile.TemporaryDirectory() as tmp:
        lib_copy = os.path.join(tmp, "stale.so")
        shutil.copy(acpp.LIBRARY_PATH, lib_copy)
        src_copy = os.path.join(tmp, "analytic_ik")
        shutil.copytree(acpp.SOURCE_DIR, src_copy)
        with open(os.path.join(src_copy, "ad7.h"), "a") as f:
            f.write("\n// edited after the build\n")
        real = acpp.SOURCE_DIR
        acpp.SourceHash.__defaults__ = (src_copy,)
        try:
            acpp.LoadLibrary(lib_copy)
            check("a library built from other sources is refused", False, "loaded")
        except RuntimeError as exc:
            check("a library built from other sources is refused, naming the build script",
                  "cpp/build.sh" in str(exc), exc)
        finally:
            acpp.SourceHash.__defaults__ = (real,)
    check("the built library's hash is the current sources'",
          acpp.LoadLibrary().learnedik_analytic_src_hash().decode() == acpp.SourceHash())
    m = AnalyticCppMap("panda")
    x = np.array([0.4, 0.1, 0.5, np.pi, 0.0, 0.0, 0.2])
    for gc in ((3, 1), (1, 1, 0), (1,)):
        try:
            m(x, RigidTransform(), gc, 1e-4)
            check(f"panda branch {gc} is refused", False, "accepted")
        except ValueError:
            check(f"panda branch {gc} is refused", True)
    try:
        AnalyticCppMap("iiwa")(x, RigidTransform(), (1, 2, 1), 1e-4)
        check("iiwa branch (1, 2, 1) is refused", False, "accepted")
    except ValueError:
        check("iiwa branch (1, 2, 1) is refused", True)


## --------------------------------- programs ---------------------------------- ##

def build_programs():
    '''The four analytic programs, each once per backend, on the scenes and checkpoints
    tests/test_constraint_bounds.py uses. -> {name: (python_prog, cpp_prog)}.'''
    from src.panda_program import (PandaIKProgram, PandaIKProgramAnalytic, PandaMugProgram,
                                   PandaMugProgramAnalytic)
    from src.iiwa_program import (Iiwa14IKProgram, Iiwa14IKProgramAnalytic, IiwaMugProgram,
                                  IiwaMugProgramAnalytic)
    rng = np.random.default_rng(0)
    out = {}

    def options(backend):
        return ProgramOptions(collision_avoidance=True, joint_limits=True, use_float64=True,
                              joint_centering_cost=1.0, analytic_branches=8,
                              analytic_backend=backend)

    for robot, scene, sampler_cls, mug_sampler_cls, pose_cls, mug_cls in (
            ("panda", "models/panda/panda_finray_collision_hardened.yaml", PandaIKProgram,
             PandaMugProgram, PandaIKProgramAnalytic, PandaMugProgramAnalytic),
            ("iiwa", "models/iiwa14/iiwa14_collision_hardened.yaml", Iiwa14IKProgram,
             IiwaMugProgram, Iiwa14IKProgramAnalytic, IiwaMugProgramAnalytic)):
        scene = os.path.join(REPO, scene)
        with HiddenPrints():
            diagram = BuildEnv(meshcat=None, directives_file=scene)
            sampler = sampler_cls(diagram, options=options("python"))
            sampler.create_prog()
            model = sampler.ik_solver
            q_t = rng.uniform(sampler.plant.GetPositionLowerLimits(),
                              sampler.plant.GetPositionUpperLimits())
            target = np.concatenate(sampler.fk(q_t))
            mug_sampler = mug_sampler_cls(diagram, options=options("python"), model=model)
            mug_sampler.create_prog()
            q_m = rng.uniform(mug_sampler.plant.GetPositionLowerLimits(),
                              mug_sampler.plant.GetPositionUpperLimits())
            diagram_with_mug, mug = GenerateDiagramWithMug(q_m, mug_sampler, scene, None)
            for task, cls, dg, arg in (("pose", pose_cls, diagram, dict(target_pose=target)),
                                       ("grasp", mug_cls, diagram_with_mug,
                                        dict(target_mug=mug))):
                pair = []
                for backend in ("python", "cpp"):
                    p = cls(dg, options=options(backend), model=model)
                    p.create_prog(**arg)
                    pair.append(p)
                out[f"{robot} {task}"] = pair
    return out


def eval_all(p, x):
    vals = []
    for b in list(p.prog.GetAllConstraints()) + list(p.prog.GetAllCosts()):
        vals.append((b.evaluator().get_description(), np.atleast_1d(p.prog.EvalBinding(b, x))))
    return vals


def test_program_parity(programs):
    print("program parity: every binding, float and AutoDiffXd, python v cpp backend")
    rng = np.random.default_rng(13)
    for name, (pp, pc) in programs.items():
        check(f"{name}: RobotSettings records the backend",
              pp.RobotSettings()["analytic_backend"] == "python"
              and pc.RobotSettings()["analytic_backend"] == "cpp")
        check(f"{name}: both backends measured the same pose offset",
              pp.pose_offset.IsExactlyEqualTo(pc.pose_offset))
        branches = pp.BranchSet()
        worst_f = worst_v = worst_g = 0.0
        n_bindings = 0
        same_rows = True
        for k in range(6):
            gc = branches[rng.integers(len(branches))]
            pp.gc, pc.gc = gc.copy(), gc.copy()
            x0 = np.asarray(pp.prog.GetInitialGuess(pp.lumped_vars), dtype=float)
            x = x0 + rng.normal(0, [0.05] * 6 + [0.5]) * (k > 0)
            x_ad = InitializeAutoDiff(x)[:, 0]
            fp, fc = eval_all(pp, x), eval_all(pc, x)
            ap, ac = eval_all(pp, x_ad), eval_all(pc, x_ad)
            n_bindings = len(fp)
            for (dp, vp), (dc, vc) in zip(fp, fc):
                same_rows &= dp == dc and vp.shape == vc.shape
                worst_f = max(worst_f, float(np.abs(vp - vc).max()))
            for (dp, vp), (dc, vc) in zip(ap, ac):
                same_rows &= dp == dc and vp.shape == vc.shape
                worst_v = max(worst_v, float(np.abs(ExtractValue(vp) - ExtractValue(vc)).max()))
                worst_g = max(worst_g, float(np.abs(ExtractGradient(vp)
                                                    - ExtractGradient(vc)).max()))
        print(f"    {name}: {n_bindings} bindings, max |d| float {worst_f:.2e}, AutoDiffXd "
              f"value {worst_v:.2e}, gradient {worst_g:.2e}")
        check(f"{name}: the same bindings in the same order", same_rows)
        check(f"{name}: every binding agrees <= 1e-11 (float, AD value, AD gradient)",
              max(worst_f, worst_v, worst_g) <= 1e-11, (worst_f, worst_v, worst_g))
        ## The paired harness's closed-form check runs the map at clip 0.
        lo, hi = (np.asarray(b, float)[:7] for b in pp.ConfigLimits())
        d_err = 0.0
        for _ in range(5):
            q = rng.uniform(lo, hi)
            pp.SetStartFromQ(q)
            pc.SetStartFromQ(q)
            d_err = max(d_err, abs(pp.UnclippedStartError(q) - pc.UnclippedStartError(q)))
        check(f"{name}: UnclippedStartError (clip 0) agrees <= 1e-11", d_err <= 1e-11, d_err)


def test_missing_library_at_construction(programs):
    print("a cpp program without its library fails at CONSTRUCTION")
    _, pc = programs["panda pose"]
    saved = analytic_cpp.LIBRARY_PATH
    analytic_cpp.LIBRARY_PATH = os.path.join(REPO, "cpp", "build", "does_not_exist.so")
    try:
        with HiddenPrints():
            type(pc)(pc.diagram, options=pc.options, model=pc.ik_solver)
        check("construction raises", False, "constructed")
    except FileNotFoundError:
        check("construction raises", True)
    finally:
        analytic_cpp.LIBRARY_PATH = saved


def test_timing(programs):
    print("us per map call through _Solve, python v cpp (one core, niced; INDICATIVE ONLY)")
    for name in ("panda pose", "iiwa pose"):
        pp, pc = programs[name]
        x = np.asarray(pp.prog.GetInitialGuess(pp.lumped_vars), dtype=float)
        x_ad = InitializeAutoDiff(x)[:, 0]
        clip = pp.options.analytic_clip_stepback
        row = []
        for p in (pp, pc):
            for arg, n in ((x, 300), (x_ad, 100)):
                p._Solve(arg, clip)
                t = time.perf_counter()
                for _ in range(n):
                    p._Solve(arg, clip)
                row.append((time.perf_counter() - t) / n * 1e6)
        print(f"    {name}: python {row[0]:.0f} us float, {row[1]:.0f} us AutoDiffXd (7 "
              f"derivatives); cpp {row[2]:.1f} us float, {row[3]:.1f} us AutoDiffXd")


PROGRAMS = []                  # built once by `build_programs_once`, read by the three program checks


def build_programs_once():
    PROGRAMS.append(build_programs())


def program_parity():
    test_program_parity(PROGRAMS[0])


def missing_library_at_construction():
    test_missing_library_at_construction(PROGRAMS[0])


def timing():
    test_timing(PROGRAMS[0])


GROUPS = {"fast": [test_map_parity, test_q6_seam, test_refusals, build_programs_once,
                   program_parity, missing_library_at_construction, timing]}


if __name__ == "__main__":
    _tiers.run(GROUPS)
    print(f"\n{CHECKS[0] - len(FAILURES)}/{CHECKS[0]} checks passed")
    if FAILURES:
        print("FAILED:", *FAILURES, sep="\n  ")
        sys.exit(1)
