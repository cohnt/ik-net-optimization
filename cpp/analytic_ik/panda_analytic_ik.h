#pragma once

/// @file
/// The Panda's closed-form IK (He & Liu, psi = q7), a transliteration of
/// `Analytic_IK_Panda.Solve` in src/panda_analytic_ik.py -- which is the SPEC: where this
/// file and pullback-metrics' C++ differ (four watched arguments, not five; the elbow
/// branch A; q6 wrapped into its window), the Python wins. Statement order, association
/// and which operands are floats follow the Python line by line; see ad7.h for why that
/// matters.

#include <cmath>

#include "ad7.h"
#include "linalg3.h"

namespace learned_ik {
namespace panda {

constexpr double kPi = 3.141592653589793;  // np.pi
constexpr double a[9] = {0., 0., 0, 0.0825, -0.0825, 0, 0.088, 0., 0.};
constexpr double d[9] = {0.333, 0., 0.316, 0., 0.384, 0., 0, 0.107, 0.1034};
constexpr double alpha[9] = {0., -kPi / 2, kPi / 2, kPi / 2, -kPi / 2, kPi / 2, kPi / 2,
                             0., 0.};
constexpr double kLimitLower5 = -0.0175;
constexpr double kLimitUpper5 = 3.7525;
constexpr double kSafeEps = 1e-6;

/// Q6_WRAP_SEAM, evaluated as the Python evaluates it.
inline double Q6WrapSeam() {
  return 0.5 * (kLimitUpper5 + kLimitLower5 + 2 * kPi) - 2 * kPi;
}

/// `self.Ts[i](t)`: the modified-DH link transform, as the Python's 4x4 object array (its
/// float and int entries stay floats).
template <typename T>
Mat4<T> Dh(int i, const T& t) {
  using std::cos;
  using std::sin;
  const double ai = alpha[i], ri = a[i], di = d[i];
  Mat4<T> X;
  X[0] = {cos(t), -sin(t), T(0.0), T(ri)};
  X[1] = {sin(t) * T(std::cos(ai)), cos(t) * T(std::cos(ai)), T(-std::sin(ai)),
          T(-di * std::sin(ai))};
  X[2] = {sin(t) * T(std::sin(ai)), cos(t) * T(std::sin(ai)), T(std::cos(ai)),
          T(di * std::cos(ai))};
  X[3] = {T(0.0), T(0.0), T(0.0), T(1.0)};
  return X;
}

/// `safe_norm`: eps (a float) when the norm is below it.
template <typename T>
T SafeNorm(const Vec3<T>& v) {
  const T n = Norm(v);
  if (std::abs(Value(n)) < kSafeEps) return T(kSafeEps);
  return n;
}

/// `safe_divide`: the denominator replaced by +-eps (a float) when below it.
template <typename T>
T SafeDivide(const T& num, const T& den) {
  if (std::abs(Value(den)) < kSafeEps) {
    const double sign = Value(den) >= 0 ? 1 : -1;
    return num / T(sign * kSafeEps);
  }
  return num / den;
}

/// `Solve(R, p, psi, GC, clip_stepback)` -> q[7], u[4]. `B`, `C` in {1, 2}, `A` in
/// {+1, -1} (A = +1 on the 4-branch chart).
template <typename T>
void Solve(const Mat3<T>& R, const Vec3<T>& p, const T& psi, int B, int C, int A,
           double clip_stepback, T* q, T* u) {
  using std::acos;
  using std::asin;
  using std::atan2;
  using std::cos;
  using std::sin;
  using std::sqrt;
  const double clip = 1 - clip_stepback;
  q[6] = psi;

  // q[3]
  const double O2O4 = std::sqrt(Pow2(d[2]) + Pow2(a[3]));
  const double O4O6 = std::sqrt(Pow2(d[4]) + Pow2(a[4]));
  Vec3<T> p7;
  for (int i = 0; i < 3; ++i) p7[i] = p[i] - T(d[7] + d[8]) * R[i][2];
  const T ang = -psi + T(kPi / 4);
  const Vec3<T> x6 = MatVec(R, Vec3<T>{cos(ang), sin(ang), T(0.0)});
  Vec3<T> p6, O2O6_vec, p6_p2;
  for (int i = 0; i < 3; ++i) p6[i] = p7[i] - T(a[6]) * x6[i];
  const double p2[3] = {0, 0, d[0]};
  for (int i = 0; i < 3; ++i) O2O6_vec[i] = p6[i] - T(p2[i]);
  for (int i = 0; i < 3; ++i) p6_p2[i] = p6[i] - T(p2[i]);
  const T O2O6 = Norm(p6_p2);
  u[0] = (T(Pow2(O2O4) + Pow2(O4O6)) - Pow2(O2O6)) / T(2 * O2O4 * O4O6);
  const T O2O4O6 = T(static_cast<double>(A)) * acos(Clip(u[0], -clip, clip));
  const double O2O4O3 = std::atan2(d[2], a[3]);
  const double q3_add = O2O4O3 + std::atan2(d[4], std::abs(a[4]));
  q[3] = O2O4O6 + T(q3_add) - T(A > 0 ? 2 * kPi : 0.0);

  // q[5]
  u[1] = (Pow2(O2O6) + T(Pow2(O4O6)) - T(Pow2(O2O4))) / (T(2.0) * O2O6 * T(O4O6));
  const T O2O6O4 = T(static_cast<double>(A)) * acos(Clip(u[1], -clip, clip));
  const T O2O6H = O2O6O4 + T(std::atan2(-a[4], d[4]));
  const Vec3<T> y6{-R[0][2], -R[1][2], -R[2][2]};
  const Vec3<T> z6 = Cross(x6, y6);
  Mat3<T> R6;
  for (int i = 0; i < 3; ++i) R6[i] = {x6[i], y6[i], z6[i]};
  const Vec3<T> v626 = MatTVec(R6, O2O6_vec);
  const T &x626 = v626[0], &y626 = v626[1];
  const T phi6 = atan2(y626, x626);
  u[2] = O2O6 * cos(O2O6H) / sqrt(Pow2(x626) + Pow2(y626));
  const T psi6 = asin(Clip(u[2], -clip, clip));
  T q5 = (B == 1) ? T(kPi) - psi6 - phi6 : psi6 - phi6;
  // Wrapped by a multiple of 2pi chosen on the VALUE only.
  q5 = q5 - T(2 * kPi * std::floor((Value(q5) - Q6WrapSeam()) / (2 * kPi)));
  q[5] = q5;

  // q[0] and q[1]
  const double O3O2O4 = kPi / 2 - O2O4O3;
  const T O4O2O6 = T(kPi) - O2O4O6 - O2O6O4;
  const T PO2O6 = T(O3O2O4) + O4O2O6;
  const T O2PO6 = O2O6O4 + O2O4O6 + T(O2O4O3) - O2O6H - T(kPi / 2);
  const T PO6 = SafeDivide(O2O6 * sin(PO2O6), sin(O2PO6));
  const Vec3<T> z65{sin(q5), cos(q5), T(0.0)};
  Mat3<T> PO6_R6;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) PO6_R6[i][j] = PO6 * R6[i][j];
  const Vec3<T> PO6_R6_z65 = MatVec(PO6_R6, z65);
  Vec3<T> O2P_vec;
  for (int i = 0; i < 3; ++i) O2P_vec[i] = O2O6_vec[i] - PO6_R6_z65[i];
  const T &x2P = O2P_vec[0], &y2P = O2P_vec[1], &z2P = O2P_vec[2];
  const T norm_O2P = SafeNorm(O2P_vec);

  u[3] = z2P / Norm(O2P_vec);
  if (C == 1) {
    q[0] = atan2(y2P, x2P);
    q[1] = acos(Clip(u[3], -clip, clip));
  } else {
    q[0] = atan2(-y2P, -x2P);
    q[1] = -acos(Clip(u[3], -clip, clip));
  }

  // q[2]
  const Vec3<T> c_ = Cross(O2P_vec, O2O6_vec);
  const T c_norm = SafeNorm(Cross(O2P_vec, O2O6_vec));
  Vec3<T> y3, z3;
  for (int i = 0; i < 3; ++i) y3[i] = c_[i] / c_norm;
  for (int i = 0; i < 3; ++i) z3[i] = O2P_vec[i] / norm_O2P;
  const Vec3<T> x3 = Cross(y3, z3);
  const Mat3<T> R2 = Rot(MatMul<T, 4>(Dh(0, q[0]), Dh(1, q[1])));
  const Vec3<T> x23 = MatTVec(R2, x3);
  q[2] = atan2(x23[2], x23[0]);

  // q[4]
  Vec3<T> p4;
  for (int i = 0; i < 3; ++i) p4[i] = T(p2[i]) + T(d[2]) * z3[i] + T(a[3]) * x3[i];
  const Mat3<T> R56 = Rot(Dh(5, q5));
  const Mat3<T> R5 = MatMul<T, 3>(R6, Transpose<T, 3>(R56));
  Vec3<T> HO4;
  for (int i = 0; i < 3; ++i) HO4[i] = p4[i] - (p6[i] - T(d[4]) * R5[i][2]);
  const Vec3<T> O5S = MatTVec(R5, HO4);
  q[4] = -atan2(O5S[1], O5S[0]);
}

}  // namespace panda
}  // namespace learned_ik
