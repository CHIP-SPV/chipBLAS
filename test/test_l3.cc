// L3 BLAS correctness vs. CPU reference: Sgemm, Dgemm, Cgemm, Zgemm,
// each across the four (op_a, op_b) ∈ {N,T} × {N,T} combinations. Cgemm
// and Zgemm additionally exercise the conjugate-transpose path.
// Pass argv[1] for a single CTest shard (see test/CMakeLists.txt).
//
// SPDX-License-Identifier: MIT

#include "test_common.hh"

#include <complex>

using namespace chipblas_test;

namespace {

// Real-typed column-major GEMM reference.
template <class T>
void gemmHostReal(hipblasOperation_t opA, hipblasOperation_t opB,
                  int m, int n, int k, T alpha,
                  const T* A, int lda, const T* B, int ldb,
                  T beta, T* C, int ldc) {
    auto a = [&](int i, int p) {
        if (opA == HIPBLAS_OP_N) return A[(size_t)p * lda + i];
        return                          A[(size_t)i * lda + p];
    };
    auto b = [&](int p, int j) {
        if (opB == HIPBLAS_OP_N) return B[(size_t)j * ldb + p];
        return                          B[(size_t)p * ldb + j];
    };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T acc = T{};
            for (int p = 0; p < k; ++p) acc += a(i, p) * b(p, j);
            C[(size_t)j * ldc + i] = alpha * acc + beta * C[(size_t)j * ldc + i];
        }
    }
}

// Complex-typed column-major GEMM reference (handles N/T/C ops).
template <class S, class C>  // S = scalar (float/double), C = complex struct
void gemmHostComplex(hipblasOperation_t opA, hipblasOperation_t opB,
                     int m, int n, int k, C alpha,
                     const C* A, int lda, const C* B, int ldb,
                     C beta, C* Cmat, int ldc) {
    using cstd = std::complex<S>;
    auto load = [&](const C* M, int lda_, int row, int col, hipblasOperation_t op) -> cstd {
        if (op == HIPBLAS_OP_N) {
            const C& v = M[(size_t)col * lda_ + row];
            return {v.x, v.y};
        }
        const C& v = M[(size_t)row * lda_ + col];
        cstd r{v.x, v.y};
        if (op == HIPBLAS_OP_C) r = std::conj(r);
        return r;
    };
    cstd alpha_c{alpha.x, alpha.y};
    cstd beta_c {beta.x,  beta.y};
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cstd acc{0, 0};
            for (int p = 0; p < k; ++p) acc += load(A, lda, i, p, opA)
                                              * load(B, ldb, p, j, opB);
            cstd c0{Cmat[(size_t)j * ldc + i].x, Cmat[(size_t)j * ldc + i].y};
            cstd r = alpha_c * acc + beta_c * c0;
            Cmat[(size_t)j * ldc + i].x = r.real();
            Cmat[(size_t)j * ldc + i].y = r.imag();
        }
    }
}

bool runSgemm(hipblasOperation_t opA, hipblasOperation_t opB,
              int m, int n, int k) {
    int lda = (opA == HIPBLAS_OP_N) ? m : k;
    int ldb = (opB == HIPBLAS_OP_N) ? k : n;
    int ldc = m;
    int aCols = (opA == HIPBLAS_OP_N) ? k : m;
    int bCols = (opB == HIPBLAS_OP_N) ? n : k;
    size_t aN = (size_t)lda * aCols, bN = (size_t)ldb * bCols, cN = (size_t)ldc * n;
    float alpha = 1.25f, beta = 0.5f;
    std::vector<float> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i) A[i] = fillF((int)i, 31);
    for (size_t i = 0; i < bN; ++i) B[i] = fillF((int)i, 32);
    for (size_t i = 0; i < cN; ++i) C[i] = fillF((int)i, 33);
    C_ref = C;
    gemmHostReal<float>(opA, opB, m, n, k, alpha, A.data(), lda,
                        B.data(), ldb, beta, C_ref.data(), ldc);
    float *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, aN * sizeof(float)));
    CHECK_HIP(hipMalloc(&dB, bN * sizeof(float)));
    CHECK_HIP(hipMalloc(&dC, cN * sizeof(float)));
    CHECK_HIP(hipMemcpy(dA, A.data(), aN * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, B.data(), bN * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, C.data(), cN * sizeof(float), hipMemcpyHostToDevice));
    hipblasHandle_t h; CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasSgemm(h, opA, opB, m, n, k, &alpha,
                            dA, lda, dB, ldb, &beta, dC, ldc));
    CHECK_BLAS(hipblasDestroy(h));
    std::vector<float> C_out(cN);
    CHECK_HIP(hipMemcpy(C_out.data(), dC, cN * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
    return closeReal<float>(C_out, C_ref, 5e-4f);
}

bool runDgemm(hipblasOperation_t opA, hipblasOperation_t opB,
              int m, int n, int k) {
    int lda = (opA == HIPBLAS_OP_N) ? m : k;
    int ldb = (opB == HIPBLAS_OP_N) ? k : n;
    int ldc = m;
    int aCols = (opA == HIPBLAS_OP_N) ? k : m;
    int bCols = (opB == HIPBLAS_OP_N) ? n : k;
    size_t aN = (size_t)lda * aCols, bN = (size_t)ldb * bCols, cN = (size_t)ldc * n;
    double alpha = 0.875, beta = -0.125;
    std::vector<double> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i) A[i] = fillD((int)i, 41);
    for (size_t i = 0; i < bN; ++i) B[i] = fillD((int)i, 42);
    for (size_t i = 0; i < cN; ++i) C[i] = fillD((int)i, 43);
    C_ref = C;
    gemmHostReal<double>(opA, opB, m, n, k, alpha, A.data(), lda,
                         B.data(), ldb, beta, C_ref.data(), ldc);
    double *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, aN * sizeof(double)));
    CHECK_HIP(hipMalloc(&dB, bN * sizeof(double)));
    CHECK_HIP(hipMalloc(&dC, cN * sizeof(double)));
    CHECK_HIP(hipMemcpy(dA, A.data(), aN * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, B.data(), bN * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, C.data(), cN * sizeof(double), hipMemcpyHostToDevice));
    hipblasHandle_t h; CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasDgemm(h, opA, opB, m, n, k, &alpha,
                            dA, lda, dB, ldb, &beta, dC, ldc));
    CHECK_BLAS(hipblasDestroy(h));
    std::vector<double> C_out(cN);
    CHECK_HIP(hipMemcpy(C_out.data(), dC, cN * sizeof(double), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
    return closeReal<double>(C_out, C_ref, 1e-10);
}

bool runCgemm(hipblasOperation_t opA, hipblasOperation_t opB,
              int m, int n, int k) {
    int lda = (opA == HIPBLAS_OP_N) ? m : k;
    int ldb = (opB == HIPBLAS_OP_N) ? k : n;
    int ldc = m;
    int aCols = (opA == HIPBLAS_OP_N) ? k : m;
    int bCols = (opB == HIPBLAS_OP_N) ? n : k;
    size_t aN = (size_t)lda * aCols, bN = (size_t)ldb * bCols, cN = (size_t)ldc * n;
    hipblasComplex alpha = {1.0f, 0.5f}, beta = {0.25f, -0.125f};
    std::vector<hipblasComplex> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i) { A[i].x = fillF((int)i*2,   51); A[i].y = fillF((int)i*2+1, 51); }
    for (size_t i = 0; i < bN; ++i) { B[i].x = fillF((int)i*2,   52); B[i].y = fillF((int)i*2+1, 52); }
    for (size_t i = 0; i < cN; ++i) { C[i].x = fillF((int)i*2,   53); C[i].y = fillF((int)i*2+1, 53); }
    C_ref = C;
    gemmHostComplex<float, hipblasComplex>(opA, opB, m, n, k, alpha,
        A.data(), lda, B.data(), ldb, beta, C_ref.data(), ldc);
    hipblasComplex *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, aN * sizeof(hipblasComplex)));
    CHECK_HIP(hipMalloc(&dB, bN * sizeof(hipblasComplex)));
    CHECK_HIP(hipMalloc(&dC, cN * sizeof(hipblasComplex)));
    CHECK_HIP(hipMemcpy(dA, A.data(), aN * sizeof(hipblasComplex), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, B.data(), bN * sizeof(hipblasComplex), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, C.data(), cN * sizeof(hipblasComplex), hipMemcpyHostToDevice));
    hipblasHandle_t h; CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasCgemm(h, opA, opB, m, n, k, &alpha,
                            dA, lda, dB, ldb, &beta, dC, ldc));
    CHECK_BLAS(hipblasDestroy(h));
    std::vector<hipblasComplex> C_out(cN);
    CHECK_HIP(hipMemcpy(C_out.data(), dC, cN * sizeof(hipblasComplex), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
    return closeComplex<hipblasComplex, float>(C_out, C_ref, 5e-4f);
}

bool runZgemm(hipblasOperation_t opA, hipblasOperation_t opB,
              int m, int n, int k) {
    int lda = (opA == HIPBLAS_OP_N) ? m : k;
    int ldb = (opB == HIPBLAS_OP_N) ? k : n;
    int ldc = m;
    int aCols = (opA == HIPBLAS_OP_N) ? k : m;
    int bCols = (opB == HIPBLAS_OP_N) ? n : k;
    size_t aN = (size_t)lda * aCols, bN = (size_t)ldb * bCols, cN = (size_t)ldc * n;
    hipblasDoubleComplex alpha = {0.75, -0.25}, beta = {-0.5, 0.125};
    std::vector<hipblasDoubleComplex> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i) { A[i].x = fillD((int)i*2,   61); A[i].y = fillD((int)i*2+1, 61); }
    for (size_t i = 0; i < bN; ++i) { B[i].x = fillD((int)i*2,   62); B[i].y = fillD((int)i*2+1, 62); }
    for (size_t i = 0; i < cN; ++i) { C[i].x = fillD((int)i*2,   63); C[i].y = fillD((int)i*2+1, 63); }
    C_ref = C;
    gemmHostComplex<double, hipblasDoubleComplex>(opA, opB, m, n, k, alpha,
        A.data(), lda, B.data(), ldb, beta, C_ref.data(), ldc);
    hipblasDoubleComplex *dA, *dB, *dC;
    CHECK_HIP(hipMalloc(&dA, aN * sizeof(hipblasDoubleComplex)));
    CHECK_HIP(hipMalloc(&dB, bN * sizeof(hipblasDoubleComplex)));
    CHECK_HIP(hipMalloc(&dC, cN * sizeof(hipblasDoubleComplex)));
    CHECK_HIP(hipMemcpy(dA, A.data(), aN * sizeof(hipblasDoubleComplex), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, B.data(), bN * sizeof(hipblasDoubleComplex), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, C.data(), cN * sizeof(hipblasDoubleComplex), hipMemcpyHostToDevice));
    hipblasHandle_t h; CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasZgemm(h, opA, opB, m, n, k, &alpha,
                            dA, lda, dB, ldb, &beta, dC, ldc));
    CHECK_BLAS(hipblasDestroy(h));
    std::vector<hipblasDoubleComplex> C_out(cN);
    CHECK_HIP(hipMemcpy(C_out.data(), dC, cN * sizeof(hipblasDoubleComplex), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
    return closeComplex<hipblasDoubleComplex, double>(C_out, C_ref, 1e-10);
}

} // namespace

int main(int argc, char** argv) {
    bool ok = true;
    const int M = 32, N = 24, K = 16;
#define RUN(slug, name, fn)                                                        \
    if (should_run_case(argc, argv, slug)) {                                       \
        bool _p = (fn)();                                                           \
        report(name, _p);                                                           \
        ok &= _p;                                                                   \
        if (case_filter_active(argc, argv))                                         \
            return ok ? 0 : 1;                                                       \
    }                                                                               \
    do {                                                                            \
    } while (0)

    RUN("l3:sgemm-NN", "Sgemm NN 32x24x16",
        ([M, N, K]() {
            return runSgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:sgemm-NT", "Sgemm NT 32x24x16",
        ([M, N, K]() {
            return runSgemm(HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K);
        }));
    RUN("l3:sgemm-TN", "Sgemm TN 32x24x16",
        ([M, N, K]() {
            return runSgemm(HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:sgemm-TT", "Sgemm TT 32x24x16",
        ([M, N, K]() {
            return runSgemm(HIPBLAS_OP_T, HIPBLAS_OP_T, M, N, K);
        }));
#if defined(CHIPBLAS_HAS_FP64)
    RUN("l3:dgemm-NN", "Dgemm NN 32x24x16",
        ([M, N, K]() {
            return runDgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:dgemm-NT", "Dgemm NT 32x24x16",
        ([M, N, K]() {
            return runDgemm(HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K);
        }));
    RUN("l3:dgemm-TN", "Dgemm TN 32x24x16",
        ([M, N, K]() {
            return runDgemm(HIPBLAS_OP_T, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:dgemm-TT", "Dgemm TT 32x24x16",
        ([M, N, K]() {
            return runDgemm(HIPBLAS_OP_T, HIPBLAS_OP_T, M, N, K);
        }));
#endif
    RUN("l3:cgemm-NN", "Cgemm NN 32x24x16",
        ([M, N, K]() {
            return runCgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:cgemm-CN", "Cgemm CN 32x24x16",
        ([M, N, K]() {
            return runCgemm(HIPBLAS_OP_C, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:cgemm-NC", "Cgemm NC 32x24x16",
        ([M, N, K]() {
            return runCgemm(HIPBLAS_OP_N, HIPBLAS_OP_C, M, N, K);
        }));
    RUN("l3:cgemm-CC", "Cgemm CC 32x24x16",
        ([M, N, K]() {
            return runCgemm(HIPBLAS_OP_C, HIPBLAS_OP_C, M, N, K);
        }));
#if defined(CHIPBLAS_HAS_FP64)
    RUN("l3:zgemm-NN", "Zgemm NN 32x24x16",
        ([M, N, K]() {
            return runZgemm(HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:zgemm-CN", "Zgemm CN 32x24x16",
        ([M, N, K]() {
            return runZgemm(HIPBLAS_OP_C, HIPBLAS_OP_N, M, N, K);
        }));
    RUN("l3:zgemm-NC", "Zgemm NC 32x24x16",
        ([M, N, K]() {
            return runZgemm(HIPBLAS_OP_N, HIPBLAS_OP_C, M, N, K);
        }));
    RUN("l3:zgemm-CC", "Zgemm CC 32x24x16",
        ([M, N, K]() {
            return runZgemm(HIPBLAS_OP_C, HIPBLAS_OP_C, M, N, K);
        }));
#endif
#undef RUN

    if (case_filter_active(argc, argv)) {
        std::fprintf(stderr, "unknown l3 case \"%s\"\n", argv[1]);
        return 2;
    }
    return ok ? 0 : 1;
}
