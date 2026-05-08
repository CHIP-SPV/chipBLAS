// Shared test helpers — checked HIP / hipBLAS macros, deterministic data
// generators, and a generic "compare device output to host reference"
// routine.
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_TEST_COMMON_HH
#define CHIPBLAS_TEST_COMMON_HH

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_HIP(expr) do { \
    hipError_t _e = (expr); \
    if (_e != hipSuccess) { \
        std::fprintf(stderr, "HIP error %d at %s:%d: %s\n", (int)_e, \
                     __FILE__, __LINE__, hipGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_BLAS(expr) do { \
    hipblasStatus_t _s = (expr); \
    if (_s != HIPBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "hipBLAS error %d at %s:%d\n", (int)_s, \
                     __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

namespace chipblas_test {

// CTest shards: argv[1] selects a single case; no extra args runs all cases in
// the executable.
inline bool case_filter_active(int argc, char** argv) {
    return argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0';
}
inline bool should_run_case(int argc, char** argv, const char* slug) {
    if (!case_filter_active(argc, argv))
        return true;
    return std::strcmp(argv[1], slug) == 0;
}

// Deterministic [-1, 1)-ish filler keyed on (i, salt) so multiple buffers
// in the same test get distinct content.
inline float fillF(int i, int salt) {
    // Unsigned: signed int overflow in (i * 1103515245 + ...) was undefined and
    // could trap under -O3 before any HIP call (empty stderr, SIGABRT in fill).
    unsigned v = (static_cast<unsigned>(i) * 1103515245u
                    + static_cast<unsigned>(salt) * 12345u)
                   & 0xffffu;
    return (static_cast<float>(v) / 32768.0f) - 1.0f;
}
inline double fillD(int i, int salt) {
    return static_cast<double>(fillF(i, salt));
}

// Real-valued comparison.
template <class T>
bool closeReal(const std::vector<T>& a, const std::vector<T>& b, T tol) {
    if (a.size() != b.size()) return false;
    T maxErr = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T e = std::fabs(a[i] - b[i]);
        if (e > maxErr) maxErr = e;
    }
    if (maxErr > tol) {
        std::fprintf(stderr, "  mismatch: maxErr=%g tol=%g\n",
                     (double)maxErr, (double)tol);
        return false;
    }
    return true;
}

// Complex comparison (interleaved {x, y}).
template <class C, class T>
bool closeComplex(const std::vector<C>& a, const std::vector<C>& b, T tol) {
    if (a.size() != b.size()) return false;
    T maxErr = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T er = std::fabs(a[i].x - b[i].x);
        T ei = std::fabs(a[i].y - b[i].y);
        if (er > maxErr) maxErr = er;
        if (ei > maxErr) maxErr = ei;
    }
    if (maxErr > tol) {
        std::fprintf(stderr, "  mismatch: maxErr=%g tol=%g\n",
                     (double)maxErr, (double)tol);
        return false;
    }
    return true;
}

inline void report(const char* name, bool ok) {
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", name);
}

} // namespace chipblas_test

#endif // CHIPBLAS_TEST_COMMON_HH
