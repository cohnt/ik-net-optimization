"""The C++ backend of the analytic formulation's closed-form maps (`analytic_backend="cpp"`).

`cpp/build.sh` builds `cpp/build/liblearnedik_analytic.so`, a plain C ABI loaded here with
ctypes (no pybind11 in the venv, and none wanted in the cluster's). The library does what
`AnalyticProgramMixin._Solve` does -- RPY -> R, the measured offset, the robot's map -- on
the seven decision variables directly, and on request returns `J = d[q; u]/dx` (11 x 7).
An AutoDiffXd call then chain-rules `J @ G` into the caller's seed matrix `G`, the same
dual-path pattern `VarsToQ` uses for the flow.

The Python maps (src/panda_analytic_ik.py, src/iiwa_analytic_ik.py) are the SPEC; the C++
transliterates them operation for operation, including which operands are floats and
Drake's lazy derivative coefficient (cpp/analytic_ik/ad7.h), so values and derivatives agree
with the AutoDiffXd path bit for bit, as measured (tests/test_analytic_cpp.py). The Python
FLOAT path differs from its own AutoDiffXd path by a few ulps (numpy's BLAS matmuls), so the
C++ value call matches the float path only as closely as the reference matches itself.

A missing or stale library RAISES; there is no silent fall-back to Python, because a run
that asked for one backend and measured the other would record the wrong one.
"""
import ctypes
import os

import numpy as np
from pydrake.all import AutoDiffXd, ExtractGradient, InitializeAutoDiff, RigidTransform

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY_PATH = os.path.join(REPO, "cpp", "build", "liblearnedik_analytic.so")
#: Must equal `learnedik_analytic_abi_version()` in cpp/analytic_ik/analytic_c_api.cc.
ABI_VERSION = 1
ROBOTS = ("panda", "iiwa")
NUM_Q, NUM_U, NUM_VARS = 7, 4, 7

_STATUS = {1: "a branch index outside the chart's set (or the wrong number of them)",
           2: "clip_stepback outside [0, 1)"}
_LIBRARIES = {}


def LoadLibrary(path=None):
    '''The shared library, loaded once per path; raises with the build command if absent.'''
    path = LIBRARY_PATH if path is None else path
    lib = _LIBRARIES.get(path)
    if lib is not None:
        return lib
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"analytic_backend='cpp' needs {path}, which does not exist. Build it with "
            f"`bash cpp/build.sh` (g++ only) on the machine that will run it.")
    lib = ctypes.CDLL(path)
    lib.learnedik_analytic_abi_version.restype = ctypes.c_int
    lib.learnedik_analytic_abi_version.argtypes = []
    version = lib.learnedik_analytic_abi_version()
    if version != ABI_VERSION:
        raise RuntimeError(f"{path} has ABI version {version}, this wrapper expects "
                           f"{ABI_VERSION}: rebuild it with `bash cpp/build.sh`")
    p = ctypes.c_void_p
    for name in ("learnedik_panda_analytic", "learnedik_iiwa_analytic"):
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int
        fn.argtypes = [p, p, p, p, ctypes.c_int, ctypes.c_double, ctypes.c_int, p, p, p]
    _LIBRARIES[path] = lib
    return lib


def _CheckParameters(robot, analytic_ik):
    '''The parameters the C++ hard-codes, checked against the Python map a program holds,
    so a map built with other DH values can never be routed to the C++ silently.'''
    if robot == "panda":
        from src import panda_analytic_ik as m
        fields = dict(a=m.panda_a, d=m.panda_d, alpha=m.panda_alpha)
        ok = (all(np.array_equal(getattr(analytic_ik, k), v) for k, v in fields.items())
              and np.array_equal(analytic_ik.limits_lower[5], m.panda_limits_lower[5])
              and np.array_equal(analytic_ik.limits_upper[5], m.panda_limits_upper[5]))
    else:
        from src import iiwa_analytic_ik as m
        fields = dict(d=m.iiwa_d, alpha=m.iiwa_alpha)
        ok = all(np.array_equal(getattr(analytic_ik, k), v) for k, v in fields.items())
    if not ok:
        raise ValueError(f"the C++ {robot} map hard-codes the default DH parameters; this "
                         f"program's {type(analytic_ik).__name__} uses others")


class AnalyticCppMap:
    '''One robot's C++ map with its own call buffers (not thread-safe, like the program
    that owns it). `__call__(vars, pose_offset, gc, clip)` returns `(q[7], u[4])` as floats
    for float `vars` and as AutoDiffXd for AutoDiffXd `vars`.'''

    def __init__(self, robot, analytic_ik=None, library_path=None):
        if robot not in ROBOTS:
            raise ValueError(f"no C++ analytic map for {robot!r}; have {ROBOTS}")
        if analytic_ik is not None:
            _CheckParameters(robot, analytic_ik)
        self.robot = robot
        lib = LoadLibrary(library_path)
        self._fn = getattr(lib, f"learnedik_{robot}_analytic")
        self._x = np.zeros(NUM_VARS)
        self._off = np.zeros(12)  # R_off row-major, then p_off
        self._gc = np.zeros(3, dtype=np.int32)
        self._q = np.zeros(NUM_Q)
        self._u = np.zeros(NUM_U)
        self._J = np.zeros((NUM_Q + NUM_U, NUM_VARS))
        self._ptr = [a.ctypes.data for a in (self._x, self._off, self._gc, self._q, self._u,
                                             self._J)]
        self._offset_key = None
        self._gc_key = None

    def _SetOffset(self, pose_offset):
        ## Packed once per offset object: a program's `pose_offset` is assigned, never
        ## mutated, so identity is the right key (and the per-call RigidTransform reads
        ## would otherwise be most of the call).
        if pose_offset is self._offset_key:
            return
        if not isinstance(pose_offset, RigidTransform):
            raise TypeError("pose_offset must be a pydrake RigidTransform")
        self._off[:9] = pose_offset.rotation().matrix().ravel()
        self._off[9:] = pose_offset.translation()
        self._offset_key = pose_offset

    def _SetGc(self, gc):
        ## Keyed on the VALUES (a branch array may be reassigned or edited in place), and
        ## validated only when they change.
        key = tuple(np.ravel(gc).tolist())
        if key == self._gc_key:
            return len(key)
        if len(key) > 3 or any(v != round(v) for v in key):
            raise ValueError(f"branch {key} is not a tuple of at most 3 integers")
        self._gc[:len(key)] = key
        self._gc_key = key
        return len(key)

    def _Call(self, n_gc, clip_stepback, want_jacobian):
        x, off, gc, q, u, J = self._ptr
        status = self._fn(x, off, off + 72, gc, n_gc, clip_stepback, want_jacobian, q, u, J)
        if status != 0:
            raise ValueError(f"C++ {self.robot} analytic map refused the call: "
                             f"{_STATUS.get(status, f'status {status}')} "
                             f"(gc={self._gc[:n_gc]}, clip_stepback={clip_stepback})")

    def __call__(self, vars, pose_offset, gc, clip_stepback):
        self._SetOffset(pose_offset)
        n_gc = self._SetGc(gc)
        if not isinstance(vars[0], AutoDiffXd):
            self._x[:] = vars
            self._Call(n_gc, float(clip_stepback), 0)
            return self._q.copy(), self._u.copy()
        vars = np.asarray(vars)
        self._x[:] = [v.value() for v in vars]
        G = ExtractGradient(vars)
        self._Call(n_gc, float(clip_stepback), 1)
        y = InitializeAutoDiff(np.concatenate([self._q, self._u]), self._J @ G)[:, 0]
        return y[:NUM_Q], y[NUM_Q:]


def SolveCpp(robot, vars, pose_offset, gc, clip_stepback):
    '''One-off convenience: a fresh `AnalyticCppMap(robot)` per call. Programs hold one.'''
    return AnalyticCppMap(robot)(vars, pose_offset, gc, clip_stepback)
