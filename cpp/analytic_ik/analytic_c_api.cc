/// @file
/// The plain C ABI `src/analytic_cpp.py` loads with ctypes -- no pybind11, no Drake, no
/// Eigen, so the shared library builds with g++ alone on the laptop and the cluster.
///
/// Both entry points take the analytic formulation's seven decision variables
/// `x = [x, y, z, roll, pitch, yaw, psi]` and do everything the Python mixin's `_Solve`
/// does: RPY -> R (Drake's convention), the measured offset `(R @ R_off, p + R @ p_off)`,
/// then the robot's map. So `J = d[q; u]/dx` is with respect to the decision variables
/// themselves, and the caller chain-rules it into whatever seed matrix it holds.
///
///   x[7], R_off[9] (row-major), p_off[3], gc[n_gc], clip_stepback, want_jacobian
///   -> q[7], u[4], and J[11 x 7] row-major (rows q0..q6, u0..u3) if want_jacobian.
///
/// Returns 0, or a nonzero status (below) without writing any output.

#include "ad7.h"
#include "iiwa_analytic_ik.h"
#include "linalg3.h"
#include "panda_analytic_ik.h"

namespace {

using learned_ik::Ad7;
using learned_ik::kNumVars;
using learned_ik::Mat3;
using learned_ik::Vec3;

constexpr int kNumQ = 7;
constexpr int kNumU = 4;

enum Status : int {
  kOk = 0,
  kBadBranch = 1,      ///< a gc entry outside the chart's set, or the wrong gc length
  kBadClip = 2,        ///< clip_stepback outside [0, 1)
};

/// Runs `solve(R, p, psi, q, u)` on doubles or on Ad7 and writes the outputs.
template <typename Solver>
int Run(const double* x, const double* R_off, const double* p_off, int want_jacobian,
        double* q_out, double* u_out, double* J_out, Solver solve) {
  if (!want_jacobian) {
    double xv[kNumVars];
    for (int k = 0; k < kNumVars; ++k) xv[k] = x[k];
    Mat3<double> R;
    Vec3<double> p;
    learned_ik::ComposeVarsPose(xv, R_off, p_off, &R, &p);
    double q[kNumQ], u[kNumU];
    solve(R, p, xv[6], q, u);
    for (int i = 0; i < kNumQ; ++i) q_out[i] = q[i];
    for (int i = 0; i < kNumU; ++i) u_out[i] = u[i];
    return kOk;
  }
  Ad7 xa[kNumVars];
  for (int k = 0; k < kNumVars; ++k) xa[k] = Ad7::Variable(x[k], k);
  Mat3<Ad7> R;
  Vec3<Ad7> p;
  learned_ik::ComposeVarsPose(xa, R_off, p_off, &R, &p);
  Ad7 q[kNumQ], u[kNumU];
  solve(R, p, xa[6], q, u);
  for (int i = 0; i < kNumQ; ++i) q_out[i] = q[i].v;
  for (int i = 0; i < kNumU; ++i) u_out[i] = u[i].v;
  for (int i = 0; i < kNumQ + kNumU; ++i) {
    const Ad7& y = (i < kNumQ) ? q[i] : u[i - kNumQ];
    for (int k = 0; k < kNumVars; ++k) J_out[i * kNumVars + k] = y.ad ? y.Partial(k) : 0.0;
  }
  return kOk;
}

bool ClipOk(double clip_stepback) { return clip_stepback >= 0.0 && clip_stepback < 1.0; }

}  // namespace

extern "C" {

/// Bumped whenever a signature or the meaning of an output changes; the loader refuses a
/// library built from another version rather than calling it with the wrong arguments.
int learnedik_analytic_abi_version() { return 1; }

/// The Panda (He & Liu). gc = (B, C) on the 4-branch chart or (B, C, A) on 8, with
/// B, C in {1, 2} and A in {+1, -1}.
int learnedik_panda_analytic(const double* x, const double* R_off, const double* p_off,
                             const int* gc, int n_gc, double clip_stepback,
                             int want_jacobian, double* q, double* u, double* J) {
  if (n_gc != 2 && n_gc != 3) return kBadBranch;
  const int B = gc[0], C = gc[1], A = (n_gc == 3) ? gc[2] : 1;
  if ((B != 1 && B != 2) || (C != 1 && C != 2) || (A != 1 && A != -1)) return kBadBranch;
  if (!ClipOk(clip_stepback)) return kBadClip;
  return Run(x, R_off, p_off, want_jacobian, q, u, J,
             [&](const auto& R, const auto& p, const auto& psi, auto* q_, auto* u_) {
               learned_ik::panda::Solve(R, p, psi, B, C, A, clip_stepback, q_, u_);
             });
}

/// The iiwa14 (Faria S-R-S). gc = (sign q2, sign q4, sign q6), each +-1.
int learnedik_iiwa_analytic(const double* x, const double* R_off, const double* p_off,
                            const int* gc, int n_gc, double clip_stepback, int want_jacobian,
                            double* q, double* u, double* J) {
  if (n_gc != 3) return kBadBranch;
  for (int i = 0; i < 3; ++i)
    if (gc[i] != 1 && gc[i] != -1) return kBadBranch;
  if (!ClipOk(clip_stepback)) return kBadClip;
  return Run(x, R_off, p_off, want_jacobian, q, u, J,
             [&](const auto& R, const auto& p, const auto& psi, auto* q_, auto* u_) {
               learned_ik::iiwa::Solve(R, p, gc[0], gc[1], gc[2], psi, clip_stepback, q_,
                                       u_);
             });
}

}  // extern "C"
