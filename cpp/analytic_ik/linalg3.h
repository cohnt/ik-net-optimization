#pragma once

/// @file
/// The few array operations the maps use, each in the order numpy evaluates it on an
/// object array: a matrix product's entry is `p_0 + p_1 + ...` from the FIRST product
/// (never `0 + p_0`), `np.cross` is `a1*b2 - a2*b1, ...`, `np.linalg.norm` is
/// `sqrt(x.dot(x))`. Generic in the scalar, so `double` and `Ad7` share them.

#include <array>
#include <cmath>

namespace learned_ik {

template <typename T> using Vec3 = std::array<T, 3>;
template <typename T, int N> using Mat = std::array<std::array<T, N>, N>;
template <typename T> using Mat3 = Mat<T, 3>;
template <typename T> using Mat4 = Mat<T, 4>;

/// A @ B for square object arrays.
template <typename T, int N>
Mat<T, N> MatMul(const Mat<T, N>& A, const Mat<T, N>& B) {
  Mat<T, N> C;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      T sum = A[i][0] * B[0][j];
      for (int k = 1; k < N; ++k) sum = sum + A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  return C;
}

template <typename T, int N>
Mat<T, N> Transpose(const Mat<T, N>& A) {
  Mat<T, N> B;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) B[i][j] = A[j][i];
  return B;
}

/// A @ v.
template <typename T>
Vec3<T> MatVec(const Mat3<T>& A, const Vec3<T>& v) {
  Vec3<T> r;
  for (int i = 0; i < 3; ++i) r[i] = A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2];
  return r;
}

/// A.T @ v.
template <typename T>
Vec3<T> MatTVec(const Mat3<T>& A, const Vec3<T>& v) {
  Vec3<T> r;
  for (int i = 0; i < 3; ++i) r[i] = A[0][i] * v[0] + A[1][i] * v[1] + A[2][i] * v[2];
  return r;
}

/// The top-left 3x3 block of a 4x4.
template <typename T>
Mat3<T> Rot(const Mat4<T>& X) {
  Mat3<T> R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i][j] = X[i][j];
  return R;
}

template <typename T>
Vec3<T> Cross(const Vec3<T>& a, const Vec3<T>& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

template <typename T>
T Dot(const Vec3<T>& a, const Vec3<T>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
T Norm(const Vec3<T>& a) {
  using std::sqrt;
  return sqrt(Dot(a, a));
}

/// Drake's RotationMatrix(RollPitchYaw), term for term (math/rotation_matrix.cc), the
/// convention `R = Rz(yaw) Ry(pitch) Rx(roll)`.
template <typename T>
Mat3<T> RotationFromRpy(const T& r, const T& p, const T& y) {
  using std::cos;
  using std::sin;
  const T c0 = cos(r), c1 = cos(p), c2 = cos(y);
  const T s0 = sin(r), s1 = sin(p), s2 = sin(y);
  const T c2_s1 = c2 * s1, s2_s1 = s2 * s1;
  Mat3<T> R;
  R[0] = {c2 * c1, c2_s1 * s0 - s2 * c0, c2_s1 * c0 + s2 * s0};
  R[1] = {s2 * c1, s2_s1 * s0 + c2 * c0, s2_s1 * c0 - c2 * s0};
  R[2] = {-s1, c1 * s0, c1 * c0};
  return R;
}

/// The mixin's `_Solve` ahead of the map: the decision variables' pose composed with the
/// measured offset, `(R @ R_off, p + R @ p_off)`. `R_off` is row-major.
template <typename T>
void ComposeVarsPose(const T* x, const double* R_off, const double* p_off, Mat3<T>* R_out,
                     Vec3<T>* p_out) {
  const Mat3<T> R = RotationFromRpy(x[3], x[4], x[5]);
  Mat3<T> Ro;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) Ro[i][j] = T(R_off[3 * i + j]);
  *R_out = MatMul<T, 3>(R, Ro);
  const Vec3<T> po{T(p_off[0]), T(p_off[1]), T(p_off[2])};
  const Vec3<T> Rp = MatVec(R, po);
  for (int i = 0; i < 3; ++i) (*p_out)[i] = x[i] + Rp[i];
}

}  // namespace learned_ik
