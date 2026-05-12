// Host-side BLAS reference kernels for chipBLAS conformance tests (same math
// as test_l1 / test_l2 / test_l3).
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_TEST_BLAS_REFERENCE_HH
#define CHIPBLAS_TEST_BLAS_REFERENCE_HH

#include <cstddef>
#include <complex>

#include <hipblas/hipblas.h>

namespace chipblas_test {

template <class T>
void axpyHost(int n, T alpha, const T* x, int incx, T* y, int incy) {
    for (int i = 0; i < n; ++i)
        y[i * incy] += alpha * x[i * incx];
}

template <class T>
void scalHost(int n, T alpha, T* x, int incx) {
    for (int i = 0; i < n; ++i)
        x[i * incx] *= alpha;
}

template <class T>
void gemvHost(hipblasOperation_t op, int m, int n, T alpha,
              const T* A, int lda, const T* x, int incx,
              T beta, T* y, int incy) {
    int yLen = (op == HIPBLAS_OP_N) ? m : n;
    int xLen = (op == HIPBLAS_OP_N) ? n : m;
    for (int i = 0; i < yLen; ++i)
        y[i * incy] *= beta;
    for (int j = 0; j < xLen; ++j) {
        T xj = x[j * incx];
        for (int i = 0; i < yLen; ++i) {
            T a;
            if (op == HIPBLAS_OP_N)
                a = A[(size_t)j * lda + i];
            else
                a = A[(size_t)i * lda + j];
            y[i * incy] += alpha * a * xj;
        }
    }
}

template <class T>
void gemmHostReal(hipblasOperation_t opA, hipblasOperation_t opB,
                  int m, int n, int k, T alpha,
                  const T* A, int lda, const T* B, int ldb,
                  T beta, T* C, int ldc) {
    auto a = [&](int i, int p) {
        if (opA == HIPBLAS_OP_N)
            return A[(size_t)p * lda + i];
        return A[(size_t)i * lda + p];
    };
    auto b = [&](int p, int j) {
        if (opB == HIPBLAS_OP_N)
            return B[(size_t)j * ldb + p];
        return B[(size_t)p * ldb + j];
    };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T acc = T{};
            for (int p = 0; p < k; ++p)
                acc += a(i, p) * b(p, j);
            C[(size_t)j * ldc + i]
                = alpha * acc + beta * C[(size_t)j * ldc + i];
        }
    }
}

template <class S, class C>
void gemmHostComplex(hipblasOperation_t opA, hipblasOperation_t opB,
                     int m, int n, int k, C alpha,
                     const C* A, int lda, const C* B, int ldb,
                     C beta, C* Cmat, int ldc) {
    using cstd = std::complex<S>;
    auto load = [&](const C* M, int lda_, int row, int col,
                    hipblasOperation_t op) -> cstd {
        if (op == HIPBLAS_OP_N) {
            const C& v = M[(size_t)col * lda_ + row];
            return {v.x, v.y};
        }
        const C& v = M[(size_t)row * lda_ + col];
        cstd r{v.x, v.y};
        if (op == HIPBLAS_OP_C)
            r = std::conj(r);
        return r;
    };
    cstd alpha_c{alpha.x, alpha.y};
    cstd beta_c{beta.x, beta.y};
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cstd acc{0, 0};
            for (int p = 0; p < k; ++p)
                acc += load(A, lda, i, p, opA) * load(B, ldb, p, j, opB);
            cstd c0{Cmat[(size_t)j * ldc + i].x, Cmat[(size_t)j * ldc + i].y};
            cstd r = alpha_c * acc + beta_c * c0;
            Cmat[(size_t)j * ldc + i].x = r.real();
            Cmat[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

template <class T>
size_t vecStorage(int n, int inc) {
    int absInc = inc < 0 ? -inc : inc;
    if (n <= 0 || absInc < 1)
        return 0;
    return (size_t)(n - 1) * (size_t)absInc + 1;
}

} // namespace chipblas_test

#endif
