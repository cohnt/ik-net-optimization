#!/usr/bin/env bash
## Builds cpp/build/liblearnedik_analytic.so, the closed-form IK maps behind
## `ProgramOptions.analytic_backend="cpp"` (loaded by src/analytic_cpp.py with ctypes).
##
## One g++ command and no dependencies, so it runs unchanged on the laptop and the cluster.
## Deliberately no -ffast-math (it would break the parity with the Python reference, and
## IEEE semantics are what the clip's tie rules and the NaN paths rely on) and no
## -march=native (the .so must run on whichever Xeon a job lands on). Build ON the machine
## that runs it, or on one with the same glibc: the maps call libm's sin/cos/acos/atan2,
## and the parity tests assume the same libm as the Python process.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$here/build"
"${CXX:-g++}" -O2 -std=c++17 -fPIC -shared -Wall -Wextra \
    -o "$here/build/liblearnedik_analytic.so" "$here/analytic_ik/analytic_c_api.cc"
echo "built $here/build/liblearnedik_analytic.so"
