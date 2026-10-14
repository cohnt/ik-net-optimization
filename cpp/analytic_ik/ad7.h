#pragma once

/// @file
/// `Ad7`: a forward-mode dual number with exactly seven partials -- one per decision
/// variable of the analytic formulation (xyz, rpy, psi) -- written to reproduce, operation
/// for operation, what the Python maps compute when handed `AutoDiffXd`.
///
/// Why not a plain dual. The Python reference runs Drake's `AutoDiffXd` through numpy
/// object arrays, and two things there decide the last bits of every derivative:
///
///  * Drake stores partials LAZILY, as `coeff * storage` (common/ad/internal/partials.cc):
///    a multiply or divide by a scalar touches only `coeff`, and a sum folds it into the
///    storage. A plain dual would round `d * a / b` where Drake rounds `d * (a / b)`.
///  * Python mixes plain floats (and ints) with AutoDiffXd: `float * AD` scales, `AD + AD`
///    folds, `AD * 0` keeps an AD whose coefficient is zero. Which operands were AD is
///    therefore part of the arithmetic, so `Ad7` carries an `ad` flag and a constant is a
///    float exactly as Python's is.
///
/// With both mirrored, and the maps transliterated in the Python's own order (numpy object
/// matmul sums left to right starting from the first product), values AND derivatives match
/// the AutoDiffXd path bit for bit (tests/test_analytic_cpp.py, 6000 draws, max diff 0).
/// Values are what matter most: near an arccos seam a one-ulp difference in an argument is
/// amplified by 1/sqrt(1 - u^2), into the value and the derivative alike, whereas a
/// derivative's own rounding is not amplified.
///
/// The same maps are instantiated on `double` for the value-only call; a double run takes
/// the same value operations, so the two calls agree bit for bit with each other.

#include <cmath>

namespace learned_ik {

constexpr int kNumVars = 7;

struct Ad7 {
  double v = 0.0;
  bool ad = false;  ///< false: a Python float -- no partials at all (Drake's "empty")
  double c = 1.0;   ///< Drake's lazy coefficient
  double s[kNumVars] = {0, 0, 0, 0, 0, 0, 0};

  Ad7() = default;
  /// A constant: a Python float, never an AutoDiffXd.
  Ad7(double value) : v(value) {}  // NOLINT(runtime/explicit): floats mix freely, as in Python

  /// The k-th decision variable, as InitializeAutoDiff makes it (unit partial).
  static Ad7 Variable(double value, int k) {
    Ad7 x(value);
    x.ad = true;
    x.s[k] = 1.0;
    return x;
  }
  /// An AutoDiffXd with all-zero (NOT empty) partials: `scalar_clip`'s promoted bounds.
  static Ad7 ZeroPartials(double value) {
    Ad7 x(value);
    x.ad = true;
    return x;
  }
  double Partial(int k) const { return c * s[k]; }
};

/// ----------------------- Drake's Partials, verbatim ----------------------------- ///

inline void PartialsMul(Ad7* x, double factor) {
  if (std::isfinite(factor)) {
    x->c *= factor;
  } else {
    const double total = x->c * factor;
    for (double& d : x->s)
      if ((d < 0) || (d > 0)) d *= total;
    x->c = 1.0;
  }
}

inline void PartialsDiv(Ad7* x, double factor) {
  if (!std::isnan(factor) && factor != 0.0) {
    x->c /= factor;
  } else {
    const double total = x->c / factor;
    for (double& d : x->s)
      if ((d < 0) || (d > 0)) d *= total;
    x->c = 1.0;
  }
}

/// `x += scale * other` on the partials; an empty side behaves as zero, as in Drake.
inline void PartialsAddScaled(Ad7* x, double scale, const Ad7& other) {
  if (!other.ad) return;
  if (!x->ad) {
    for (int i = 0; i < kNumVars; ++i) x->s[i] = other.s[i];
    x->c = other.c;
    x->ad = true;
    PartialsMul(x, scale);
    return;
  }
  if (std::isfinite(scale)) {
    const double k = scale * other.c;
    for (int i = 0; i < kNumVars; ++i) x->s[i] = x->c * x->s[i] + k * other.s[i];
  } else {
    for (int i = 0; i < kNumVars; ++i) {
      const double mine = x->c * x->s[i];
      const double theirs = other.c * other.s[i];
      x->s[i] = ((theirs < 0) || (theirs > 0)) ? mine + theirs * scale : mine;
    }
  }
  x->c = 1.0;
}

/// ------------------------------ arithmetic --------------------------------------- ///
/// Each mirrors the Drake operator Python reaches: AD op AD, AD op double, double op AD.

inline Ad7 operator+(Ad7 a, const Ad7& b) {
  if (a.ad && b.ad) {
    PartialsAddScaled(&a, 1.0, b);
    a.v += b.v;
    return a;
  }
  if (a.ad) { a.v += b.v; return a; }
  Ad7 r = b;  // double + AD: b += a
  r.v += a.v;
  return r;
}

inline Ad7 operator-(Ad7 x) {
  if (x.ad) PartialsMul(&x, -1);
  x.v *= -1;
  return x;
}

inline Ad7 operator-(Ad7 a, const Ad7& b) {
  if (a.ad && b.ad) {
    PartialsAddScaled(&a, -1, b);
    a.v -= b.v;
    return a;
  }
  if (a.ad || !b.ad) { a.v -= b.v; return a; }
  Ad7 r = b;  // double - AD: b *= -1; b += a
  PartialsMul(&r, -1);
  r.v *= -1;
  r.v += a.v;
  return r;
}

inline Ad7 operator*(Ad7 a, const Ad7& b) {
  if (a.ad && b.ad) {
    PartialsMul(&a, b.v);
    PartialsAddScaled(&a, a.v, b);
    a.v *= b.v;
    return a;
  }
  if (a.ad) { PartialsMul(&a, b.v); a.v *= b.v; return a; }
  if (!b.ad) { a.v *= b.v; return a; }
  Ad7 r = b;  // double * AD: b *= a
  PartialsMul(&r, a.v);
  r.v *= a.v;
  return r;
}

inline Ad7 operator/(Ad7 a, const Ad7& b) {
  if (a.ad && b.ad) {
    PartialsMul(&a, b.v);
    PartialsAddScaled(&a, -a.v, b);
    PartialsDiv(&a, b.v * b.v);
    a.v /= b.v;
    return a;
  }
  if (a.ad) { PartialsDiv(&a, b.v); a.v /= b.v; return a; }
  if (!b.ad) { a.v /= b.v; return a; }
  // double / AD: an empty AutoDiff divided in place by b.
  Ad7 r(a.v);
  PartialsAddScaled(&r, -a.v, b);
  PartialsDiv(&r, b.v * b.v);
  r.v = a.v / b.v;
  return r;
}

/// ------------------------------ functions ---------------------------------------- ///

inline Ad7 sin(Ad7 x) {
  const double nv = std::sin(x.v);
  if (x.ad) PartialsMul(&x, std::cos(x.v));
  x.v = nv;
  return x;
}

inline Ad7 cos(Ad7 x) {
  const double nv = std::cos(x.v);
  if (x.ad) PartialsMul(&x, -std::sin(x.v));
  x.v = nv;
  return x;
}

inline Ad7 asin(Ad7 x) {
  const double nv = std::asin(x.v);
  if (x.ad) PartialsDiv(&x, std::sqrt(1 - x.v * x.v));
  x.v = nv;
  return x;
}

inline Ad7 acos(Ad7 x) {
  const double nv = std::acos(x.v);
  if (x.ad) PartialsDiv(&x, -std::sqrt(1 - x.v * x.v));
  x.v = nv;
  return x;
}

inline Ad7 sqrt(Ad7 x) {
  const double nv = std::sqrt(x.v);
  if (x.ad) PartialsDiv(&x, 2 * nv);
  x.v = nv;
  return x;
}

/// pydrake's module `atan2(y, x)` is bound for AD x AD only, so a float argument arrives
/// as an EMPTY AutoDiffXd (pybind's implicit conversion), not through Drake's mixed
/// overloads. The maps never mix; this keeps the behaviour if one ever does.
inline Ad7 atan2(Ad7 a, const Ad7& b) {
  const double nv = std::atan2(a.v, b.v);
  if (!a.ad && !b.ad) { a.v = nv; return a; }
  const double norm = a.v * a.v + b.v * b.v;
  PartialsMul(&a, b.v);
  PartialsAddScaled(&a, -a.v, b);
  PartialsDiv(&a, norm);
  a.v = nv;
  return a;
}

/// libm's `pow(b, 2)`, really called. g++ folds a literal `pow(b, 2.0)` into `b * b`, which
/// need not equal libm's result bit for bit -- measured: one Panda draw in 200 differed by
/// an ulp in an arccos argument until this was added. Python's `x**2` (Drake's
/// `pow(AD, 2.0)`, CPython's float pow) calls libm, so this does too.
inline double LibmPow2(double b) {
  volatile double two = 2.0;
  return std::pow(b, two);
}

/// `x**2`: pydrake's `__pow__(double)` is Drake's `pow(AD, AD{2.0})`, whose exponent has
/// empty partials, so only the base's scale `2 * pow(b, 1)` applies.
inline Ad7 Pow2(Ad7 x) {
  const double b = x.v;
  const double p = LibmPow2(b);
  if (x.ad) {
    double db_scale = 2.0 * std::pow(b, 2.0 - 1);
    if (!std::isfinite(b) || !std::isfinite(p)) db_scale = std::nan("");
    PartialsMul(&x, db_scale);
  }
  x.v = p;
  return x;
}

inline double Pow2(double x) { return LibmPow2(x); }

inline double Value(const Ad7& x) { return x.v; }
inline double Value(double x) { return x; }

/// `scalar_clip(val, -clip, clip)`: `max(a, min(b, val))` with the bounds promoted to
/// AutoDiffXd with zero partials, so an active clip has zero derivative. Ties go to the
/// first argument (the bound), as Drake's AD x AD min/max do.
inline Ad7 Clip(const Ad7& val, double lo, double hi) {
  if (!val.ad) {
    const double m = (val.v < hi) ? val.v : hi;  // std::min(hi, val)
    return Ad7((lo < m) ? m : lo);               // std::max(lo, m)
  }
  const Ad7 a = Ad7::ZeroPartials(lo), b = Ad7::ZeroPartials(hi);
  const Ad7& m = (b.v == val.v) ? b : ((val.v < b.v) ? val : b);
  return (a.v == m.v) ? a : ((a.v < m.v) ? m : a);
}

inline double Clip(double val, double lo, double hi) {
  const double m = (val < hi) ? val : hi;
  return (lo < m) ? m : lo;
}

}  // namespace learned_ik
