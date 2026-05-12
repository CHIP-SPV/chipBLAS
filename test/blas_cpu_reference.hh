// Naive column-major BLAS reference kernels for CPU-vs-device checks (same
// layout as CLBlast / hipBLAS column-major storage).
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_TEST_BLAS_CPU_REFERENCE_HH
#define CHIPBLAS_TEST_BLAS_CPU_REFERENCE_HH

#include <hipblas/hipblas.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace chipblas_cpu_ref {

inline std::complex<float> cload(const hipblasComplex* x, int inc, int i) {
    int o = i * inc;
    return {x[o].x, x[o].y};
}
inline void cstore(hipblasComplex* x, int inc, int i, std::complex<float> v) {
    int o = i * inc;
    x[o].x = v.real();
    x[o].y = v.imag();
}
inline std::complex<double> zload(const hipblasDoubleComplex* x, int inc,
                                    int i) {
    int o = i * inc;
    return {x[o].x, x[o].y};
}
inline void zstore(hipblasDoubleComplex* x, int inc, int i,
                   std::complex<double> v) {
    int o = i * inc;
    x[o].x = v.real();
    x[o].y = v.imag();
}

inline float halfToFloat(hipblasHalf h) {
    uint16_t u = static_cast<uint16_t>(h);
    uint32_t sign = (u >> 15) & 1u;
    uint32_t exp = (u >> 10) & 0x1fu;
    uint32_t mant = u & 0x3ffu;
    if (exp == 0) {
        if (mant == 0)
            return sign ? -0.0f : 0.0f;
        return std::ldexp((sign ? -1.f : 1.f) * (mant / 1024.f), -14);
    }
    if (exp == 31)
        return std::numeric_limits<float>::infinity() * (sign ? -1.f : 1.f);
    uint32_t f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    float out;
    std::memcpy(&out, &f, sizeof(f));
    return out;
}

inline hipblasHalf floatToHalf(float x) {
    uint32_t f;
    std::memcpy(&f, &x, sizeof(f));
    uint32_t sign = (f >> 31) & 1u;
    uint32_t exp = (f >> 23) & 0xffu;
    uint32_t mant = f & 0x7fffffu;
    if (exp == 0 && mant == 0)
        return static_cast<hipblasHalf>(sign << 15);
    int e = static_cast<int>(exp) - 127 + 15;
    uint32_t m = mant | 0x800000u;
    if (e <= 0) {
        uint32_t shift = 1u - static_cast<uint32_t>(e);
        m >>= shift;
        e = 0;
    } else if (e >= 31) {
        return static_cast<hipblasHalf>((sign << 15) | (31u << 10));
    }
    return static_cast<hipblasHalf>(
        (sign << 15) | (static_cast<uint32_t>(e) << 10) | ((m >> 13) & 0x3ffu));
}

template <class T>
T dotRef(int n, const T* x, int incx, const T* y, int incy) {
    T s{};
    for (int i = 0; i < n; ++i)
        s += x[i * incx] * y[i * incy];
    return s;
}

template <class T>
T nrm2Ref(int n, const T* x, int incx) {
    T s{};
    for (int i = 0; i < n; ++i) {
        T xi = x[i * incx];
        s += xi * xi;
    }
    return std::sqrt(s);
}

template <class T>
T asumRef(int n, const T* x, int incx) {
    T s{};
    for (int i = 0; i < n; ++i)
        s += std::fabs(x[i * incx]);
    return s;
}

template <class T>
void swapRef(int n, T* x, int incx, T* y, int incy) {
    for (int i = 0; i < n; ++i)
        std::swap(x[i * incx], y[i * incy]);
}

template <class T>
void copyRef(int n, const T* x, int incx, T* y, int incy) {
    for (int i = 0; i < n; ++i)
        y[i * incy] = x[i * incx];
}

template <class T>
void axpyRef(int n, T alpha, const T* x, int incx, T* y, int incy) {
    for (int i = 0; i < n; ++i)
        y[i * incy] += alpha * x[i * incx];
}

template <class T>
void scalRef(int n, T alpha, T* x, int incx) {
    for (int i = 0; i < n; ++i)
        x[i * incx] *= alpha;
}

template <class T>
void rotRef(int n, T* x, int incx, T* y, int incy, T c, T s) {
    for (int i = 0; i < n; ++i) {
        T xi = x[i * incx];
        T yi = y[i * incy];
        x[i * incx] = c * xi + s * yi;
        y[i * incy] = c * yi - s * xi;
    }
}

inline void srotgRef(float& a, float& b, float& c, float& s) {
    float roe = (std::fabs(a) > std::fabs(b)) ? a : b;
    float scal = std::fabs(a) + std::fabs(b);
    if (scal == 0.f) {
        c = 1.f;
        s = 0.f;
        a = 0.f;
        b = 0.f;
        return;
    }
    float r = scal * std::sqrt((a / scal) * (a / scal) + (b / scal) * (b / scal));
    r = (roe < 0.f) ? -r : r;
    c = a / r;
    s = b / r;
    float z = s;
    if (std::fabs(a) > std::fabs(b))
        z = 1.f / c;
    a = r;
    b = z;
}

inline void drotgRef(double& a, double& b, double& c, double& s) {
    double roe = (std::fabs(a) > std::fabs(b)) ? a : b;
    double scal = std::fabs(a) + std::fabs(b);
    if (scal == 0.) {
        c = 1.;
        s = 0.;
        a = 0.;
        b = 0.;
        return;
    }
    double r = scal * std::sqrt((a / scal) * (a / scal) + (b / scal) * (b / scal));
    r = (roe < 0.) ? -r : r;
    c = a / r;
    s = b / r;
    double z = s;
    if (std::fabs(a) > std::fabs(b))
        z = 1. / c;
    a = r;
    b = z;
}

template <class T>
int iamaxRef1Based(int n, const T* x, int incx) {
    if (n <= 0) return 0;
    int best = 0;
    T bestv = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        T v = std::fabs(x[i * incx]);
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best + 1;
}

template <class T>
int iaminRef1Based(int n, const T* x, int incx) {
    if (n <= 0) return 0;
    int best = 0;
    T bestv = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        T v = std::fabs(x[i * incx]);
        if (v < bestv) {
            bestv = v;
            best = i;
        }
    }
    return best + 1;
}

inline float cabs1(const hipblasComplex& z) {
    return std::fabs(z.x) + std::fabs(z.y);
}
inline double zabs1(const hipblasDoubleComplex& z) {
    return std::fabs(z.x) + std::fabs(z.y);
}

inline int icamaxRef1Based(int n, const hipblasComplex* x, int incx) {
    if (n <= 0) return 0;
    int best = 0;
    float bestv = cabs1(x[0]);
    for (int i = 1; i < n; ++i) {
        int o = i * incx;
        float v = cabs1(x[o]);
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best + 1;
}

inline int icaminRef1Based(int n, const hipblasComplex* x, int incx) {
    if (n <= 0) return 0;
    int best = 0;
    float bestv = cabs1(x[0]);
    for (int i = 1; i < n; ++i) {
        int o = i * incx;
        float v = cabs1(x[o]);
        if (v < bestv) {
            bestv = v;
            best = i;
        }
    }
    return best + 1;
}

inline int izamaxRef1Based(int n, const hipblasDoubleComplex* x, int incx) {
    if (n <= 0) return 0;
    int best = 0;
    double bestv = zabs1(x[0]);
    for (int i = 1; i < n; ++i) {
        int o = i * incx;
        double v = zabs1(x[o]);
        if (v > bestv) {
            bestv = v;
            best = i;
        }
    }
    return best + 1;
}

inline int izaminRef1Based(int n, const hipblasDoubleComplex* x, int incx) {
    if (n <= 0) return 0;
    int best = 0;
    double bestv = zabs1(x[0]);
    for (int i = 1; i < n; ++i) {
        int o = i * incx;
        double v = zabs1(x[o]);
        if (v < bestv) {
            bestv = v;
            best = i;
        }
    }
    return best + 1;
}

inline std::complex<float> cuDotRef(int n, const hipblasComplex* x, int incx,
                                      const hipblasComplex* y, int incy) {
    std::complex<float> s{};
    for (int i = 0; i < n; ++i)
        s += cload(x, incx, i) * cload(y, incy, i);
    return s;
}

inline std::complex<float> ccDotRef(int n, const hipblasComplex* x, int incx,
                                    const hipblasComplex* y, int incy) {
    std::complex<float> s{};
    for (int i = 0; i < n; ++i)
        s += cload(x, incx, i) * std::conj(cload(y, incy, i));
    return s;
}

inline std::complex<double> zuDotRef(int n, const hipblasDoubleComplex* x,
                                     int incx, const hipblasDoubleComplex* y,
                                     int incy) {
    std::complex<double> s{};
    for (int i = 0; i < n; ++i)
        s += zload(x, incx, i) * zload(y, incy, i);
    return s;
}

inline std::complex<double> zzDotRef(int n, const hipblasDoubleComplex* x,
                                     int incx, const hipblasDoubleComplex* y,
                                     int incy) {
    std::complex<double> s{};
    for (int i = 0; i < n; ++i)
        s += zload(x, incx, i) * std::conj(zload(y, incy, i));
    return s;
}

inline float scnrm2Ref(int n, const hipblasComplex* x, int incx) {
    float s = 0;
    for (int i = 0; i < n; ++i) {
        auto z = cload(x, incx, i);
        s += std::norm(z);
    }
    return std::sqrt(s);
}

inline double dznrm2Ref(int n, const hipblasDoubleComplex* x, int incx) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        auto z = zload(x, incx, i);
        s += std::norm(z);
    }
    return std::sqrt(s);
}

inline float scasumRef(int n, const hipblasComplex* x, int incx) {
    float s = 0;
    for (int i = 0; i < n; ++i) {
        auto z = cload(x, incx, i);
        s += std::fabs(z.real()) + std::fabs(z.imag());
    }
    return s;
}

inline double dzasumRef(int n, const hipblasDoubleComplex* x, int incx) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        auto z = zload(x, incx, i);
        s += std::fabs(z.real()) + std::fabs(z.imag());
    }
    return s;
}

template <class T>
T symmElemLower(const T* A, int lda, int i, int j) {
    if (i >= j)
        return A[(size_t)j * lda + i];
    return A[(size_t)i * lda + j];
}

template <class T>
void symmLeftLowerRef(int m, int n, T alpha, const T* A, int lda, const T* B,
                      int ldb, T beta, T* C, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T acc{};
            for (int k = 0; k < m; ++k)
                acc += symmElemLower(A, lda, i, k) * B[(size_t)j * ldb + k];
            C[(size_t)j * ldc + i] =
                alpha * acc + beta * C[(size_t)j * ldc + i];
        }
    }
}

inline std::complex<float> hermLower(const hipblasComplex* A, int lda, int i,
                                     int k) {
    if (i >= k) {
        auto v = A[(size_t)k * lda + i];
        return {v.x, v.y};
    }
    auto v = A[(size_t)i * lda + k];
    return {v.x, -v.y};
}

inline std::complex<double> zhermLower(const hipblasDoubleComplex* A, int lda,
                                       int i, int k) {
    if (i >= k) {
        auto v = A[(size_t)k * lda + i];
        return {v.x, v.y};
    }
    auto v = A[(size_t)i * lda + k];
    return {v.x, -v.y};
}

inline void chemmLeftLowerRef(int m, int n, std::complex<float> alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* B, int ldb,
                             std::complex<float> beta, hipblasComplex* C,
                             int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<float> acc{};
            for (int k = 0; k < m; ++k) {
                const hipblasComplex& bk = B[(size_t)j * ldb + k];
                acc += hermLower(A, lda, i, k)
                       * std::complex<float>{bk.x, bk.y};
            }
            hipblasComplex& cij = C[(size_t)j * ldc + i];
            std::complex<float> c0{cij.x, cij.y};
            std::complex<float> r = alpha * acc + beta * c0;
            cij.x = r.real();
            cij.y = r.imag();
        }
    }
}

inline void zhemmLeftLowerRef(int m, int n, std::complex<double> alpha,
                              const hipblasDoubleComplex* A, int lda,
                              const hipblasDoubleComplex* B, int ldb,
                              std::complex<double> beta,
                              hipblasDoubleComplex* C, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<double> acc{};
            for (int k = 0; k < m; ++k) {
                const hipblasDoubleComplex& bk = B[(size_t)j * ldb + k];
                acc += zhermLower(A, lda, i, k)
                       * std::complex<double>{bk.x, bk.y};
            }
            hipblasDoubleComplex& cij = C[(size_t)j * ldc + i];
            std::complex<double> c0{cij.x, cij.y};
            std::complex<double> r = alpha * acc + beta * c0;
            cij.x = r.real();
            cij.y = r.imag();
        }
    }
}

template <class T>
void syrkLowerNRef(int n, int k, T alpha, const T* A, int lda, T beta, T* C,
                   int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            T acc{};
            for (int p = 0; p < k; ++p)
                acc += A[(size_t)p * lda + i] * A[(size_t)p * lda + j];
            C[(size_t)j * ldc + i] =
                alpha * acc + beta * C[(size_t)j * ldc + i];
        }
    }
}

inline void csyrkLowerNRef(int n, int k, hipblasComplex alpha,
                           const hipblasComplex* A, int lda, hipblasComplex beta,
                           hipblasComplex* C, int ldc) {
    using cxf = std::complex<float>;
    cxf a{alpha.x, alpha.y};
    cxf b{beta.x, beta.y};
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            cxf acc{};
            for (int p = 0; p < k; ++p) {
                cxf ai{A[(size_t)p * lda + i].x, A[(size_t)p * lda + i].y};
                cxf aj{A[(size_t)p * lda + j].x, A[(size_t)p * lda + j].y};
                acc += ai * aj;
            }
            cxf c0{C[(size_t)j * ldc + i].x, C[(size_t)j * ldc + i].y};
            cxf r = a * acc + b * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

inline void zsyrkLowerNRef(int n, int k, hipblasDoubleComplex alpha,
                           const hipblasDoubleComplex* A, int lda,
                           hipblasDoubleComplex beta,
                           hipblasDoubleComplex* C, int ldc) {
    using cx = std::complex<double>;
    cx a{alpha.x, alpha.y};
    cx b{beta.x, beta.y};
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            cx acc{};
            for (int p = 0; p < k; ++p) {
                cx ai{A[(size_t)p * lda + i].x, A[(size_t)p * lda + i].y};
                cx aj{A[(size_t)p * lda + j].x, A[(size_t)p * lda + j].y};
                acc += ai * aj;
            }
            cx c0{C[(size_t)j * ldc + i].x, C[(size_t)j * ldc + i].y};
            cx r = a * acc + b * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

inline void cherkLowerNRef(int n, int k, float alpha, const hipblasComplex* A,
                           int lda, float beta, hipblasComplex* C, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            std::complex<float> acc{};
            for (int p = 0; p < k; ++p) {
                std::complex<float> ai{A[(size_t)p * lda + i].x,
                                         A[(size_t)p * lda + i].y};
                std::complex<float> aj{A[(size_t)p * lda + j].x,
                                         A[(size_t)p * lda + j].y};
                acc += ai * std::conj(aj);
            }
            std::complex<float> c0{C[(size_t)j * ldc + i].x,
                                     C[(size_t)j * ldc + i].y};
            std::complex<float> r = alpha * acc + beta * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

inline void zherkLowerNRef(int n, int k, double alpha,
                             const hipblasDoubleComplex* A, int lda,
                             double beta, hipblasDoubleComplex* C, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            std::complex<double> acc{};
            for (int p = 0; p < k; ++p) {
                std::complex<double> ai{A[(size_t)p * lda + i].x,
                                          A[(size_t)p * lda + i].y};
                std::complex<double> aj{A[(size_t)p * lda + j].x,
                                          A[(size_t)p * lda + j].y};
                acc += ai * std::conj(aj);
            }
            std::complex<double> c0{C[(size_t)j * ldc + i].x,
                                      C[(size_t)j * ldc + i].y};
            std::complex<double> r = alpha * acc + beta * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

template <class T>
void syr2kLowerNRef(int n, int k, T alpha, const T* A, int lda, const T* B,
                    int ldb, T beta, T* C, int ldc) {
    // C += alpha * A * B^T + alpha * B * A^T  (symmetric; stored lower of C)
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            T acc{};
            for (int p = 0; p < k; ++p)
                acc += A[(size_t)p * lda + i] * B[(size_t)p * ldb + j]
                     + B[(size_t)p * lda + i] * A[(size_t)p * ldb + j];
            C[(size_t)j * ldc + i] =
                alpha * acc + beta * C[(size_t)j * ldc + i];
        }
    }
}

inline void csyr2kLowerNRef(int n, int k, hipblasComplex alpha,
                            const hipblasComplex* A, int lda,
                            const hipblasComplex* B, int ldb, hipblasComplex beta,
                            hipblasComplex* C, int ldc) {
    using cx = std::complex<float>;
    cx a{alpha.x, alpha.y};
    cx b{beta.x, beta.y};
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            cx acc{};
            for (int p = 0; p < k; ++p) {
                cx Ai{A[(size_t)p * lda + i].x, A[(size_t)p * lda + i].y};
                cx Aj{A[(size_t)p * lda + j].x, A[(size_t)p * lda + j].y};
                cx Bi{B[(size_t)p * ldb + i].x, B[(size_t)p * ldb + i].y};
                cx Bj{B[(size_t)p * ldb + j].x, B[(size_t)p * ldb + j].y};
                acc += Ai * Bj + Bi * Aj;
            }
            cx c0{C[(size_t)j * ldc + i].x, C[(size_t)j * ldc + i].y};
            cx r = a * acc + b * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

inline void zsyr2kLowerNRef(int n, int k, hipblasDoubleComplex alpha,
                            const hipblasDoubleComplex* A, int lda,
                            const hipblasDoubleComplex* B, int ldb,
                            hipblasDoubleComplex beta,
                            hipblasDoubleComplex* C, int ldc) {
    using cx = std::complex<double>;
    cx a{alpha.x, alpha.y};
    cx b{beta.x, beta.y};
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            cx acc{};
            for (int p = 0; p < k; ++p) {
                cx Ai{A[(size_t)p * lda + i].x, A[(size_t)p * lda + i].y};
                cx Aj{A[(size_t)p * lda + j].x, A[(size_t)p * lda + j].y};
                cx Bi{B[(size_t)p * ldb + i].x, B[(size_t)p * ldb + i].y};
                cx Bj{B[(size_t)p * ldb + j].x, B[(size_t)p * ldb + j].y};
                acc += Ai * Bj + Bi * Aj;
            }
            cx c0{C[(size_t)j * ldc + i].x, C[(size_t)j * ldc + i].y};
            cx r = a * acc + b * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

inline void cher2kLowerNRef(int n, int k, std::complex<float> alpha,
                            const hipblasComplex* A, int lda,
                            const hipblasComplex* B, int ldb, float beta,
                            hipblasComplex* C, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            std::complex<float> acc{};
            for (int p = 0; p < k; ++p) {
                std::complex<float> Ai{A[(size_t)p * lda + i].x,
                                         A[(size_t)p * lda + i].y};
                std::complex<float> Aj{A[(size_t)p * lda + j].x,
                                         A[(size_t)p * lda + j].y};
                std::complex<float> Bi{B[(size_t)p * ldb + i].x,
                                         B[(size_t)p * ldb + i].y};
                std::complex<float> Bj{B[(size_t)p * ldb + j].x,
                                         B[(size_t)p * ldb + j].y};
                acc += alpha * Ai * std::conj(Bj)
                     + std::conj(alpha) * Bi * std::conj(Aj);
            }
            std::complex<float> c0{C[(size_t)j * ldc + i].x,
                                     C[(size_t)j * ldc + i].y};
            std::complex<float> r = acc + beta * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

inline void zher2kLowerNRef(int n, int k, std::complex<double> alpha,
                            const hipblasDoubleComplex* A, int lda,
                            const hipblasDoubleComplex* B, int ldb,
                            double beta, hipblasDoubleComplex* C, int ldc) {
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            std::complex<double> acc{};
            for (int p = 0; p < k; ++p) {
                std::complex<double> Ai{A[(size_t)p * lda + i].x,
                                         A[(size_t)p * lda + i].y};
                std::complex<double> Aj{A[(size_t)p * lda + j].x,
                                         A[(size_t)p * lda + j].y};
                std::complex<double> Bi{B[(size_t)p * ldb + i].x,
                                         B[(size_t)p * ldb + i].y};
                std::complex<double> Bj{B[(size_t)p * ldb + j].x,
                                         B[(size_t)p * ldb + j].y};
                acc += alpha * Ai * std::conj(Bj)
                     + std::conj(alpha) * Bi * std::conj(Aj);
            }
            std::complex<double> c0{C[(size_t)j * ldc + i].x,
                                      C[(size_t)j * ldc + i].y};
            std::complex<double> r = acc + beta * c0;
            C[(size_t)j * ldc + i].x = r.real();
            C[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

template <class T>
void gerRef(int m, int n, T alpha, const T* x, int incx, const T* y, int incy,
            T* A, int lda) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A[(size_t)j * lda + i] +=
                alpha * x[i * incx] * y[j * incy];
        }
    }
}

inline void cgeruRef(int m, int n, std::complex<float> alpha,
                     const hipblasComplex* x, int incx, const hipblasComplex* y,
                     int incy, hipblasComplex* A, int lda) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<float> xi = cload(x, incx, i);
            std::complex<float> yj = cload(y, incy, j);
            std::complex<float> aij{A[(size_t)j * lda + i].x,
                                      A[(size_t)j * lda + i].y};
            aij += alpha * xi * yj;
            A[(size_t)j * lda + i].x = aij.real();
            A[(size_t)j * lda + i].y = aij.imag();
        }
    }
}

inline void cgercRef(int m, int n, std::complex<float> alpha,
                     const hipblasComplex* x, int incx, const hipblasComplex* y,
                     int incy, hipblasComplex* A, int lda) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<float> xi = cload(x, incx, i);
            std::complex<float> yj = cload(y, incy, j);
            std::complex<float> aij{A[(size_t)j * lda + i].x,
                                      A[(size_t)j * lda + i].y};
            aij += alpha * xi * std::conj(yj);
            A[(size_t)j * lda + i].x = aij.real();
            A[(size_t)j * lda + i].y = aij.imag();
        }
    }
}

inline void zgeruRef(int m, int n, std::complex<double> alpha,
                     const hipblasDoubleComplex* x, int incx,
                     const hipblasDoubleComplex* y, int incy,
                     hipblasDoubleComplex* A, int lda) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<double> xi = zload(x, incx, i);
            std::complex<double> yj = zload(y, incy, j);
            std::complex<double> aij{A[(size_t)j * lda + i].x,
                                      A[(size_t)j * lda + i].y};
            aij += alpha * xi * yj;
            A[(size_t)j * lda + i].x = aij.real();
            A[(size_t)j * lda + i].y = aij.imag();
        }
    }
}

inline void zgercRef(int m, int n, std::complex<double> alpha,
                     const hipblasDoubleComplex* x, int incx,
                     const hipblasDoubleComplex* y, int incy,
                     hipblasDoubleComplex* A, int lda) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<double> xi = zload(x, incx, i);
            std::complex<double> yj = zload(y, incy, j);
            std::complex<double> aij{A[(size_t)j * lda + i].x,
                                      A[(size_t)j * lda + i].y};
            aij += alpha * xi * std::conj(yj);
            A[(size_t)j * lda + i].x = aij.real();
            A[(size_t)j * lda + i].y = aij.imag();
        }
    }
}

template <class T>
T triElemLower(const T* A, int lda, int i, int j) {
    if (i >= j)
        return A[(size_t)j * lda + i];
    return T{};
}

template <class T>
void trmvLowerNonUnitRef(int n, const T* A, int lda, T* x, int incx) {
    std::vector<T> xcp(n);
    for (int i = 0; i < n; ++i)
        xcp[i] = x[i * incx];
    for (int i = 0; i < n; ++i) {
        T acc = T{};
        for (int j = 0; j <= i; ++j)
            acc += triElemLower(A, lda, i, j) * xcp[j];
        x[i * incx] = acc;
    }
}

template <class T>
void trsvLowerNonUnitRef(int n, const T* A, int lda, T* x, int incx) {
    for (int i = 0; i < n; ++i) {
        T rhs = x[i * incx];
        for (int j = 0; j < i; ++j)
            rhs -= triElemLower(A, lda, i, j) * x[j * incx];
        T diag = triElemLower(A, lda, i, i);
        x[i * incx] = rhs / diag;
    }
}

inline void ctrmvLowerNonUnitRef(int n, const hipblasComplex* A, int lda,
                                 hipblasComplex* x, int incx) {
    std::vector<std::complex<float>> xcp(n);
    for (int i = 0; i < n; ++i)
        xcp[i] = cload(x, incx, i);
    for (int i = 0; i < n; ++i) {
        std::complex<float> acc{};
        for (int j = 0; j <= i; ++j) {
            std::complex<float> aij = (i >= j)
                ? std::complex<float>{A[(size_t)j * lda + i].x,
                                     A[(size_t)j * lda + i].y}
                : std::complex<float>{};
            acc += aij * xcp[j];
        }
        cstore(x, incx, i, acc);
    }
}

inline void ztrmvLowerNonUnitRef(int n, const hipblasDoubleComplex* A,
                                 int lda, hipblasDoubleComplex* x, int incx) {
    std::vector<std::complex<double>> xcp(n);
    for (int i = 0; i < n; ++i)
        xcp[i] = zload(x, incx, i);
    for (int i = 0; i < n; ++i) {
        std::complex<double> acc{};
        for (int j = 0; j <= i; ++j) {
            std::complex<double> aij = (i >= j)
                ? std::complex<double>{A[(size_t)j * lda + i].x,
                                        A[(size_t)j * lda + i].y}
                : std::complex<double>{};
            acc += aij * xcp[j];
        }
        zstore(x, incx, i, acc);
    }
}

inline void ctrsvLowerNonUnitRef(int n, const hipblasComplex* A, int lda,
                                 hipblasComplex* x, int incx) {
    for (int i = 0; i < n; ++i) {
        std::complex<float> rhs = cload(x, incx, i);
        for (int j = 0; j < i; ++j) {
            std::complex<float> aij{A[(size_t)j * lda + i].x,
                                     A[(size_t)j * lda + i].y};
            rhs -= aij * cload(x, incx, j);
        }
        std::complex<float> diag{A[(size_t)i * lda + i].x,
                                   A[(size_t)i * lda + i].y};
        cstore(x, incx, i, rhs / diag);
    }
}

inline void ztrsvLowerNonUnitRef(int n, const hipblasDoubleComplex* A,
                                 int lda, hipblasDoubleComplex* x, int incx) {
    for (int i = 0; i < n; ++i) {
        std::complex<double> rhs = zload(x, incx, i);
        for (int j = 0; j < i; ++j) {
            std::complex<double> aij{A[(size_t)j * lda + i].x,
                                       A[(size_t)j * lda + i].y};
            rhs -= aij * zload(x, incx, j);
        }
        std::complex<double> diag{A[(size_t)i * lda + i].x,
                                    A[(size_t)i * lda + i].y};
        zstore(x, incx, i, rhs / diag);
    }
}

template <class T>
void trmmLeftLowerNonUnitRef(int m, int n, T alpha, const T* A, int lda,
                             T* B, int ldb) {
    std::vector<T> bcol(m);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i)
            bcol[i] = B[(size_t)j * ldb + i];
        for (int i = 0; i < m; ++i) {
            T acc{};
            for (int k = 0; k <= i; ++k)
                acc += triElemLower(A, lda, i, k) * bcol[k];
            B[(size_t)j * ldb + i] = alpha * acc;
        }
    }
}

inline void ctrmmLeftLowerNonUnitRef(int m, int n, std::complex<float> alpha,
                                     const hipblasComplex* A, int lda,
                                     hipblasComplex* B, int ldb) {
    std::vector<std::complex<float>> bcol(m);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const hipblasComplex& v = B[(size_t)j * ldb + i];
            bcol[i] = {v.x, v.y};
        }
        for (int i = 0; i < m; ++i) {
            std::complex<float> acc{};
            for (int k = 0; k <= i; ++k) {
                std::complex<float> aik = (i >= k)
                    ? std::complex<float>{A[(size_t)k * lda + i].x,
                                          A[(size_t)k * lda + i].y}
                    : std::complex<float>{};
                acc += aik * bcol[k];
            }
            std::complex<float> r = alpha * acc;
            B[(size_t)j * ldb + i].x = r.real();
            B[(size_t)j * ldb + i].y = r.imag();
        }
    }
}

inline void ztrmmLeftLowerNonUnitRef(int m, int n, std::complex<double> alpha,
                                     const hipblasDoubleComplex* A, int lda,
                                     hipblasDoubleComplex* B, int ldb) {
    std::vector<std::complex<double>> bcol(m);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const hipblasDoubleComplex& v = B[(size_t)j * ldb + i];
            bcol[i] = {v.x, v.y};
        }
        for (int i = 0; i < m; ++i) {
            std::complex<double> acc{};
            for (int k = 0; k <= i; ++k) {
                std::complex<double> aik = (i >= k)
                    ? std::complex<double>{A[(size_t)k * lda + i].x,
                                           A[(size_t)k * lda + i].y}
                    : std::complex<double>{};
                acc += aik * bcol[k];
            }
            std::complex<double> r = alpha * acc;
            B[(size_t)j * ldb + i].x = r.real();
            B[(size_t)j * ldb + i].y = r.imag();
        }
    }
}

template <class T>
void trsmLeftLowerNonUnitRef(int m, int n, T alpha, const T* A, int lda,
                             T* B, int ldb) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i)
            B[(size_t)j * ldb + i] *= alpha;
        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < i; ++k) {
                T aik = triElemLower(A, lda, i, k);
                B[(size_t)j * ldb + i] -= aik * B[(size_t)j * ldb + k];
            }
            T diag = triElemLower(A, lda, i, i);
            B[(size_t)j * ldb + i] /= diag;
        }
    }
}

inline void ctrsmLeftLowerNonUnitRef(int m, int n, std::complex<float> alpha,
                                     const hipblasComplex* A, int lda,
                                     hipblasComplex* B, int ldb) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<float> bi{B[(size_t)j * ldb + i].x,
                                     B[(size_t)j * ldb + i].y};
            bi *= alpha;
            B[(size_t)j * ldb + i].x = bi.real();
            B[(size_t)j * ldb + i].y = bi.imag();
        }
        for (int i = 0; i < m; ++i) {
            std::complex<float> acc{B[(size_t)j * ldb + i].x,
                                    B[(size_t)j * ldb + i].y};
            for (int k = 0; k < i; ++k) {
                std::complex<float> aik{A[(size_t)k * lda + i].x,
                                         A[(size_t)k * lda + i].y};
                std::complex<float> bk{B[(size_t)j * ldb + k].x,
                                       B[(size_t)j * ldb + k].y};
                acc -= aik * bk;
            }
            std::complex<float> diag{A[(size_t)i * lda + i].x,
                                     A[(size_t)i * lda + i].y};
            acc /= diag;
            B[(size_t)j * ldb + i].x = acc.real();
            B[(size_t)j * ldb + i].y = acc.imag();
        }
    }
}

inline void ztrsmLeftLowerNonUnitRef(int m, int n, std::complex<double> alpha,
                                     const hipblasDoubleComplex* A, int lda,
                                     hipblasDoubleComplex* B, int ldb) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            std::complex<double> bi{B[(size_t)j * ldb + i].x,
                                      B[(size_t)j * ldb + i].y};
            bi *= alpha;
            B[(size_t)j * ldb + i].x = bi.real();
            B[(size_t)j * ldb + i].y = bi.imag();
        }
        for (int i = 0; i < m; ++i) {
            std::complex<double> acc{B[(size_t)j * ldb + i].x,
                                     B[(size_t)j * ldb + i].y};
            for (int k = 0; k < i; ++k) {
                std::complex<double> aik{A[(size_t)k * lda + i].x,
                                          A[(size_t)k * lda + i].y};
                std::complex<double> bk{B[(size_t)j * ldb + k].x,
                                        B[(size_t)j * ldb + k].y};
                acc -= aik * bk;
            }
            std::complex<double> diag{A[(size_t)i * lda + i].x,
                                      A[(size_t)i * lda + i].y};
            acc /= diag;
            B[(size_t)j * ldb + i].x = acc.real();
            B[(size_t)j * ldb + i].y = acc.imag();
        }
    }
}

} // namespace chipblas_cpu_ref

#endif
