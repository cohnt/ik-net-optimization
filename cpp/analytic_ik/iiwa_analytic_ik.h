#pragma once

/// @file
/// The iiwa14's closed-form IK (Faria et al.'s S-R-S map, psi = the arm angle, GC = the
/// signs of q2, q4, q6), a transliteration of `Analytic_IK_7DoF.Solve` in
/// src/iiwa_analytic_ik.py, which is the spec. ../codebase/cpp/analytic_ik/srs_arm_ik.h
/// ports the same map on Drake's AutoDiffXd; this one follows OUR Python's statement
/// order and float/AD mixing instead (see ad7.h).

#include <cmath>

#include "ad7.h"
#include "linalg3.h"

namespace learned_ik {
namespace iiwa {

constexpr double kPi = 3.141592653589793;  // np.pi
constexpr double alpha[7] = {-kPi / 2, kPi / 2, kPi / 2, -kPi / 2, -kPi / 2, kPi / 2, 0};
constexpr double d[7] = {0.36, 0, 0.42, 0, 0.4, 0, 0.126 - 0.045};
constexpr double d_bs = d[0], d_se = d[2], d_ew = d[4], d_wf = d[6];

/// `self.Ts[i](t)`: the standard-DH link transform as the Python's 4x4 object array.
template <typename T>
Mat4<T> Dh(int i, const T& t) {
  using std::cos;
  using std::sin;
  const double ai = alpha[i], di = d[i];
  Mat4<T> X;
  X[0] = {cos(t), -sin(t) * T(std::cos(ai)), sin(t) * T(std::sin(ai)), T(0.0)};
  X[1] = {sin(t), cos(t) * T(std::cos(ai)), -cos(t) * T(std::sin(ai)), T(0.0)};
  X[2] = {T(0.0), T(std::sin(ai)), T(std::cos(ai)), T(di)};
  X[3] = {T(0.0), T(0.0), T(0.0), T(1.0)};
  return X;
}

/// `cross_product_matrix(a)` = `np.cross(a, -I)`: row i is `a x (-e_i)`, the products
/// with the float entries of -I kept (they are what numpy computes).
template <typename T>
Mat3<T> CrossProductMatrix(const Vec3<T>& a) {
  Mat3<T> K;
  for (int i = 0; i < 3; ++i) {
    Vec3<T> b{T(-0.0), T(-0.0), T(-0.0)};
    b[i] = T(-1.0);
    K[i] = Cross(a, b);
  }
  return K;
}

/// `s(M, i, j)` of EQ (15)/(20): `M0[i, j] sin psi + M1[i, j] cos psi + M2[i, j]`.
template <typename T>
T S(const Mat3<T>* M, int i, int j, const T& sin_psi, const T& cos_psi) {
  return M[0][i][j] * sin_psi + M[1][i][j] * cos_psi + M[2][i][j];
}

/// `Solve(R_07, p_07, GC, psi, clip_stepback)` -> q[7], u[4]; GC entries are +-1.
template <typename T>
void Solve(const Mat3<T>& R_07, const Vec3<T>& p_07, int GC2_, int GC4_, int GC6_,
           const T& psi, double clip_stepback, T* thetas, T* unclipped) {
  using std::acos;
  using std::atan2;
  using std::cos;
  using std::sin;
  using std::sqrt;
  const double clip = 1 - clip_stepback;
  const T GC2 = T(static_cast<double>(GC2_)), GC4 = T(static_cast<double>(GC4_)),
          GC6 = T(static_cast<double>(GC6_));
  const T mGC2 = T(static_cast<double>(-GC2_)), mGC6 = T(static_cast<double>(-GC6_));
  const T sin_psi = sin(psi), cos_psi = cos(psi);

  const Vec3<T> p_67{T(0.0), T(0.0), T(d_wf)};
  const Vec3<T> R_p67 = MatVec(R_07, p_67);
  Vec3<T> p_26;  // EQ (3)
  const double p_02[3] = {0, 0, d_bs};
  for (int i = 0; i < 3; ++i) p_26[i] = p_07[i] - T(p_02[i]) - R_p67[i];
  const T norm_p_26 = Norm(p_26);
  Vec3<T> p_26_hat;
  for (int i = 0; i < 3; ++i) p_26_hat[i] = p_26[i] / norm_p_26;

  // EQ (4)
  unclipped[0] = (Dot(p_26, p_26) - T(Pow2(d_se)) - T(Pow2(d_ew))) / T(2 * d_se * d_ew);
  const T theta_4v = GC4 * acos(Clip(unclipped[0], -clip, clip));
  thetas[3] = theta_4v;

  const T theta_1v = atan2(p_26[1], p_26[0]);  // EQ (5)

  // EQ (7)
  unclipped[1] = (T(Pow2(d_se)) + Dot(p_26, p_26) - T(Pow2(d_ew))) /
                 (T(2 * d_se) * norm_p_26);
  const T phi = acos(Clip(unclipped[1], -clip, clip));
  const T norm_xy = sqrt(p_26[0] * p_26[0] + p_26[1] * p_26[1]);
  const T theta_2v = atan2(norm_xy, p_26[2]) + GC4 * phi;

  // R_03_v = multi_dot(T_vs[0:3]), which numpy associates as A @ (B @ C) for 4x4s.
  const Mat4<T> T0 = Dh(0, theta_1v), T1 = Dh(1, theta_2v), T2 = Dh(2, T(0.0));
  const Mat3<T> R_03_v = Rot(MatMul<T, 4>(T0, MatMul<T, 4>(T1, T2)));

  // EQ (15)
  const Mat3<T> K = CrossProductMatrix(p_26_hat);
  Mat3<T> mK, outer;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      mK[i][j] = T(-1.0) * K[i][j];
      outer[i][j] = p_26_hat[i] * p_26_hat[j];
    }
  const Mat3<T> Sm[3] = {MatMul<T, 3>(K, R_03_v),
                         MatMul<T, 3>(MatMul<T, 3>(mK, K), R_03_v),
                         MatMul<T, 3>(outer, R_03_v)};

  // EQ (17)-(19)
  thetas[0] = atan2(GC2 * S(Sm, 1, 1, sin_psi, cos_psi), GC2 * S(Sm, 0, 1, sin_psi, cos_psi));
  unclipped[2] = S(Sm, 2, 1, sin_psi, cos_psi);
  thetas[1] = GC2 * acos(Clip(unclipped[2], -clip, clip));
  thetas[2] = atan2(mGC2 * S(Sm, 2, 2, sin_psi, cos_psi), mGC2 * S(Sm, 2, 0, sin_psi, cos_psi));

  // EQ (20): W_k = R_34.T @ M_k.T @ R_07, left to right.
  const Mat3<T> R_34T = Transpose<T, 3>(Rot(Dh(3, theta_4v)));
  Mat3<T> W[3];
  for (int k = 0; k < 3; ++k)
    W[k] = MatMul<T, 3>(MatMul<T, 3>(R_34T, Transpose<T, 3>(Sm[k])), R_07);

  // EQ (22)-(24)
  thetas[4] = atan2(GC6 * S(W, 1, 2, sin_psi, cos_psi), GC6 * S(W, 0, 2, sin_psi, cos_psi));
  unclipped[3] = S(W, 2, 2, sin_psi, cos_psi);
  thetas[5] = GC6 * acos(Clip(unclipped[3], -clip, clip));
  thetas[6] = atan2(GC6 * S(W, 2, 1, sin_psi, cos_psi), mGC6 * S(W, 2, 0, sin_psi, cos_psi));
}

}  // namespace iiwa
}  // namespace learned_ik
