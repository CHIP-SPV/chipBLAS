// Numerical conformance vs host reference (same scenarios as the old gtest
// driver). Plain return codes — run under CTest only (no GoogleTest).
//
// SPDX-License-Identifier: MIT

#include "blas_reference.hh"
#include "test_common.hh"

#include <cstdio>
#include <vector>

using chipblas_test::axpyHost;
using chipblas_test::gemmHostComplex;
using chipblas_test::gemmHostReal;
using chipblas_test::gemvHost;
using chipblas_test::scalHost;
using chipblas_test::vecStorage;
using chipblas_test::closeComplex;
using chipblas_test::closeReal;
using chipblas_test::fillD;
using chipblas_test::fillF;

#define REQ(cond, fmt)                                                         \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: " fmt "\n", __FILE__, __LINE__); \
            return false;                                                      \
        }                                                                      \
    } while (0)

static bool test_lifecycle_create_destroy() {
    hipblasHandle_t h = nullptr;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "hipblasCreate");
    REQ(h != nullptr, "handle null");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "hipblasDestroy");
    return true;
}

static bool test_lifecycle_destroy_null() {
    REQ(hipblasDestroy(nullptr) == HIPBLAS_STATUS_HANDLE_IS_NULLPTR,
        "destroy null");
    return true;
}

static bool test_lifecycle_stream_roundtrip() {
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    hipStream_t s = nullptr;
    REQ(hipStreamCreate(&s) == hipSuccess, "hipStreamCreate");
    REQ(hipblasSetStream(h, s) == HIPBLAS_STATUS_SUCCESS, "setstream");
    hipStream_t got = (hipStream_t)0;
    REQ(hipblasGetStream(h, &got) == HIPBLAS_STATUS_SUCCESS, "getstream");
    REQ(got == s, "stream mismatch");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy handle");
    REQ(hipStreamDestroy(s) == hipSuccess, "destroy stream");
    return true;
}

static bool test_lifecycle_pointer_mode() {
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    hipblasPointerMode_t m = HIPBLAS_POINTER_MODE_DEVICE;
    REQ(hipblasGetPointerMode(h, &m) == HIPBLAS_STATUS_SUCCESS, "get ptrmode");
    REQ(m == HIPBLAS_POINTER_MODE_HOST, "default ptrmode");
    REQ(hipblasSetPointerMode(h, HIPBLAS_POINTER_MODE_DEVICE)
            == HIPBLAS_STATUS_SUCCESS,
        "set ptrmode");
    REQ(hipblasGetPointerMode(h, &m) == HIPBLAS_STATUS_SUCCESS, "get ptrmode2");
    REQ(m == HIPBLAS_POINTER_MODE_DEVICE, "ptrmode device");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    return true;
}

static bool test_lifecycle_version() {
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    int v = 0;
    REQ(hipblasGetVersion(h, &v) == HIPBLAS_STATUS_SUCCESS, "version");
    REQ(v > 0, "version <= 0");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    return true;
}

static bool test_saxpy_inc1() {
    int n = 128;
    int incx = 1, incy = 1;
    float alpha = 1.75f;
    size_t nx = vecStorage<float>(n, incx);
    size_t ny = vecStorage<float>(n, incy);
    std::vector<float> x(nx), y(ny), y_ref;
    for (size_t i = 0; i < nx; ++i)
        x[i] = fillF((int)i, 1);
    for (size_t i = 0; i < ny; ++i)
        y[i] = fillF((int)i, 2);
    y_ref = y;
    axpyHost<float>(n, alpha, x.data(), incx, y_ref.data(), incy);

    float *dX, *dY;
    REQ(hipMalloc(&dX, nx * sizeof(float)) == hipSuccess, "malloc dX");
    REQ(hipMalloc(&dY, ny * sizeof(float)) == hipSuccess, "malloc dY");
    REQ(hipMemcpy(dX, x.data(), nx * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d X");
    REQ(hipMemcpy(dY, y.data(), ny * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d Y");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasSaxpy(h, n, &alpha, dX, incx, dY, incy)
            == HIPBLAS_STATUS_SUCCESS,
        "saxpy");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");

    std::vector<float> y_out(ny);
    REQ(hipMemcpy(y_out.data(), dY, ny * sizeof(float), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dX);
    hipFree(dY);
    REQ(closeReal<float>(y_out, y_ref, 1e-5f), "saxpy mismatch");
    return true;
}

static bool test_saxpy_strided() {
    int n = 64;
    int incx = 2, incy = 3;
    float alpha = -0.5f;
    size_t nx = vecStorage<float>(n, incx);
    size_t ny = vecStorage<float>(n, incy);
    std::vector<float> x(nx), y(ny), y_ref;
    for (size_t i = 0; i < nx; ++i)
        x[i] = fillF((int)i, 3);
    for (size_t i = 0; i < ny; ++i)
        y[i] = fillF((int)i, 4);
    y_ref = y;
    axpyHost<float>(n, alpha, x.data(), incx, y_ref.data(), incy);

    float *dX, *dY;
    REQ(hipMalloc(&dX, nx * sizeof(float)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dY, ny * sizeof(float)) == hipSuccess, "malloc");
    REQ(hipMemcpy(dX, x.data(), nx * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dY, y.data(), ny * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasSaxpy(h, n, &alpha, dX, incx, dY, incy)
            == HIPBLAS_STATUS_SUCCESS,
        "saxpy");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<float> y_out(ny);
    REQ(hipMemcpy(y_out.data(), dY, ny * sizeof(float), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dX);
    hipFree(dY);
    REQ(closeReal<float>(y_out, y_ref, 1e-5f), "saxpy strided mismatch");
    return true;
}

#if defined(CHIPBLAS_HAS_FP64)
static bool test_daxpy() {
    int n = 96;
    double alpha = -2.5;
    size_t nx = vecStorage<double>(n, 1);
    size_t ny = vecStorage<double>(n, 1);
    std::vector<double> x(nx), y(ny), y_ref;
    for (size_t i = 0; i < nx; ++i)
        x[i] = fillD((int)i, 3);
    for (size_t i = 0; i < ny; ++i)
        y[i] = fillD((int)i, 4);
    y_ref = y;
    axpyHost<double>(n, alpha, x.data(), 1, y_ref.data(), 1);

    double *dX, *dY;
    REQ(hipMalloc(&dX, nx * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dY, ny * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMemcpy(dX, x.data(), nx * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dY, y.data(), ny * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasDaxpy(h, n, &alpha, dX, 1, dY, 1) == HIPBLAS_STATUS_SUCCESS,
        "daxpy");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<double> y_out(ny);
    REQ(hipMemcpy(y_out.data(), dY, ny * sizeof(double), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dX);
    hipFree(dY);
    REQ(closeReal<double>(y_out, y_ref, 1e-12), "daxpy mismatch");
    return true;
}
#endif

static bool test_sscal() {
    int n = 100;
    float alpha = 2.25f;
    size_t nx = vecStorage<float>(n, 1);
    std::vector<float> x(nx), ref;
    for (size_t i = 0; i < nx; ++i)
        x[i] = fillF((int)i, 5);
    ref = x;
    scalHost<float>(n, alpha, ref.data(), 1);

    float* dX;
    REQ(hipMalloc(&dX, nx * sizeof(float)) == hipSuccess, "malloc");
    REQ(hipMemcpy(dX, x.data(), nx * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasSscal(h, n, &alpha, dX, 1) == HIPBLAS_STATUS_SUCCESS, "sscal");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<float> out(nx);
    REQ(hipMemcpy(out.data(), dX, nx * sizeof(float), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dX);
    REQ(closeReal<float>(out, ref, 1e-5f), "sscal mismatch");
    return true;
}

#if defined(CHIPBLAS_HAS_FP64)
static bool test_dscal() {
    int n = 88;
    double alpha = 0.5;
    size_t nx = vecStorage<double>(n, 1);
    std::vector<double> x(nx), ref;
    for (size_t i = 0; i < nx; ++i)
        x[i] = fillD((int)i, 6);
    ref = x;
    scalHost<double>(n, alpha, ref.data(), 1);

    double* dX;
    REQ(hipMalloc(&dX, nx * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMemcpy(dX, x.data(), nx * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasDscal(h, n, &alpha, dX, 1) == HIPBLAS_STATUS_SUCCESS, "dscal");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<double> out(nx);
    REQ(hipMemcpy(out.data(), dX, nx * sizeof(double), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dX);
    REQ(closeReal<double>(out, ref, 1e-12), "dscal mismatch");
    return true;
}
#endif

static bool test_sgemv_n() {
    hipblasOperation_t op = HIPBLAS_OP_N;
    int m = 64, n = 48, incx = 1, incy = 1;
    int lda = m;
    float alpha = 1.5f, beta = -0.25f;
    int yLen = m, xLen = n;
    size_t aN = (size_t)lda * n;
    size_t xN = vecStorage<float>(xLen, incx);
    size_t yN = vecStorage<float>(yLen, incy);
    std::vector<float> A(aN), x(xN), y(yN), y_ref;
    for (size_t i = 0; i < aN; ++i)
        A[i] = fillF((int)i, 11);
    for (size_t i = 0; i < xN; ++i)
        x[i] = fillF((int)i, 12);
    for (size_t i = 0; i < yN; ++i)
        y[i] = fillF((int)i, 13);
    y_ref = y;
    gemvHost<float>(op, m, n, alpha, A.data(), lda, x.data(), incx, beta,
                    y_ref.data(), incy);

    float *dA, *dX, *dY;
    REQ(hipMalloc(&dA, aN * sizeof(float)) == hipSuccess, "malloc A");
    REQ(hipMalloc(&dX, xN * sizeof(float)) == hipSuccess, "malloc X");
    REQ(hipMalloc(&dY, yN * sizeof(float)) == hipSuccess, "malloc Y");
    REQ(hipMemcpy(dA, A.data(), aN * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d A");
    REQ(hipMemcpy(dX, x.data(), xN * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d X");
    REQ(hipMemcpy(dY, y.data(), yN * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d Y");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasSgemv(h, op, m, n, &alpha, dA, lda, dX, incx, &beta, dY,
                     incy)
            == HIPBLAS_STATUS_SUCCESS,
        "sgemv");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<float> y_out(yN);
    REQ(hipMemcpy(y_out.data(), dY, yN * sizeof(float), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dA);
    hipFree(dX);
    hipFree(dY);
    REQ(closeReal<float>(y_out, y_ref, 5e-4f), "sgemv mismatch");
    return true;
}

#if defined(CHIPBLAS_HAS_FP64)
static bool test_dgemv_t() {
    hipblasOperation_t op = HIPBLAS_OP_T;
    int m = 64, n = 48, incx = 1, incy = 1;
    int lda = m;
    double alpha = 0.875, beta = 0.125;
    int xLen = m, yLen = n;
    size_t aN = (size_t)lda * n;
    size_t xN = vecStorage<double>(xLen, incx);
    size_t yN = vecStorage<double>(yLen, incy);
    std::vector<double> A(aN), x(xN), y(yN), y_ref;
    for (size_t i = 0; i < aN; ++i)
        A[i] = fillD((int)i, 21);
    for (size_t i = 0; i < xN; ++i)
        x[i] = fillD((int)i, 22);
    for (size_t i = 0; i < yN; ++i)
        y[i] = fillD((int)i, 23);
    y_ref = y;
    gemvHost<double>(op, m, n, alpha, A.data(), lda, x.data(), incx, beta,
                     y_ref.data(), incy);

    double *dA, *dX, *dY;
    REQ(hipMalloc(&dA, aN * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dX, xN * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dY, yN * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMemcpy(dA, A.data(), aN * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dX, x.data(), xN * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dY, y.data(), yN * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasDgemv(h, op, m, n, &alpha, dA, lda, dX, incx, &beta, dY,
                     incy)
            == HIPBLAS_STATUS_SUCCESS,
        "dgemv");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<double> y_out(yN);
    REQ(hipMemcpy(y_out.data(), dY, yN * sizeof(double), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dA);
    hipFree(dX);
    hipFree(dY);
    REQ(closeReal<double>(y_out, y_ref, 1e-11), "dgemv mismatch");
    return true;
}
#endif

static bool runSgemmCase(hipblasOperation_t opA, hipblasOperation_t opB, int m,
                         int n, int k) {
    int lda = (opA == HIPBLAS_OP_N) ? m : k;
    int ldb = (opB == HIPBLAS_OP_N) ? k : n;
    int ldc = m;
    int aCols = (opA == HIPBLAS_OP_N) ? k : m;
    int bCols = (opB == HIPBLAS_OP_N) ? n : k;
    size_t aN = (size_t)lda * aCols, bN = (size_t)ldb * bCols, cN = (size_t)ldc * n;
    float alpha = 1.25f, beta = 0.5f;
    std::vector<float> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i)
        A[i] = fillF((int)i, 31);
    for (size_t i = 0; i < bN; ++i)
        B[i] = fillF((int)i, 32);
    for (size_t i = 0; i < cN; ++i)
        C[i] = fillF((int)i, 33);
    C_ref = C;
    gemmHostReal<float>(opA, opB, m, n, k, alpha, A.data(), lda, B.data(), ldb,
                        beta, C_ref.data(), ldc);

    float *dA, *dB, *dC;
    REQ(hipMalloc(&dA, aN * sizeof(float)) == hipSuccess, "malloc A");
    REQ(hipMalloc(&dB, bN * sizeof(float)) == hipSuccess, "malloc B");
    REQ(hipMalloc(&dC, cN * sizeof(float)) == hipSuccess, "malloc C");
    REQ(hipMemcpy(dA, A.data(), aN * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d A");
    REQ(hipMemcpy(dB, B.data(), bN * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d B");
    REQ(hipMemcpy(dC, C.data(), cN * sizeof(float), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d C");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasSgemm(h, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC,
                     ldc)
            == HIPBLAS_STATUS_SUCCESS,
        "sgemm");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<float> C_out(cN);
    REQ(hipMemcpy(C_out.data(), dC, cN * sizeof(float), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    REQ(closeReal<float>(C_out, C_ref, 5e-4f), "sgemm mismatch");
    return true;
}

static bool test_sgemm_nn() {
    return runSgemmCase(HIPBLAS_OP_N, HIPBLAS_OP_N, 32, 24, 16);
}
static bool test_sgemm_nt() {
    return runSgemmCase(HIPBLAS_OP_N, HIPBLAS_OP_T, 32, 24, 16);
}
static bool test_sgemm_tn() {
    return runSgemmCase(HIPBLAS_OP_T, HIPBLAS_OP_N, 32, 24, 16);
}
static bool test_sgemm_tt() {
    return runSgemmCase(HIPBLAS_OP_T, HIPBLAS_OP_T, 32, 24, 16);
}

#if defined(CHIPBLAS_HAS_FP64)
static bool test_dgemm_nn() {
    hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_N;
    int m = 32, n = 24, k = 16;
    int lda = m, ldb = k, ldc = m;
    size_t aN = (size_t)lda * k, bN = (size_t)ldb * n, cN = (size_t)ldc * n;
    double alpha = 0.875, beta = -0.125;
    std::vector<double> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i)
        A[i] = fillD((int)i, 41);
    for (size_t i = 0; i < bN; ++i)
        B[i] = fillD((int)i, 42);
    for (size_t i = 0; i < cN; ++i)
        C[i] = fillD((int)i, 43);
    C_ref = C;
    gemmHostReal<double>(opA, opB, m, n, k, alpha, A.data(), lda, B.data(),
                         ldb, beta, C_ref.data(), ldc);

    double *dA, *dB, *dC;
    REQ(hipMalloc(&dA, aN * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dB, bN * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dC, cN * sizeof(double)) == hipSuccess, "malloc");
    REQ(hipMemcpy(dA, A.data(), aN * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dB, B.data(), bN * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dC, C.data(), cN * sizeof(double), hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasDgemm(h, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta,
                     dC, ldc)
            == HIPBLAS_STATUS_SUCCESS,
        "dgemm");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<double> C_out(cN);
    REQ(hipMemcpy(C_out.data(), dC, cN * sizeof(double), hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    REQ(closeReal<double>(C_out, C_ref, 1e-10), "dgemm mismatch");
    return true;
}
#endif

static bool test_cgemm_nn() {
    hipblasOperation_t opA = HIPBLAS_OP_N, opB = HIPBLAS_OP_N;
    int m = 32, n = 24, k = 16;
    int lda = m, ldb = k, ldc = m;
    int aCols = k, bCols = n;
    size_t aN = (size_t)lda * aCols, bN = (size_t)ldb * bCols, cN = (size_t)ldc * n;
    hipblasComplex alpha = {1.0f, 0.5f}, beta = {0.25f, -0.125f};
    std::vector<hipblasComplex> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i) {
        A[i].x = fillF((int)i * 2, 51);
        A[i].y = fillF((int)i * 2 + 1, 51);
    }
    for (size_t i = 0; i < bN; ++i) {
        B[i].x = fillF((int)i * 2, 52);
        B[i].y = fillF((int)i * 2 + 1, 52);
    }
    for (size_t i = 0; i < cN; ++i) {
        C[i].x = fillF((int)i * 2, 53);
        C[i].y = fillF((int)i * 2 + 1, 53);
    }
    C_ref = C;
    gemmHostComplex<float, hipblasComplex>(opA, opB, m, n, k, alpha, A.data(),
                                            lda, B.data(), ldb, beta, C_ref.data(),
                                            ldc);

    hipblasComplex *dA, *dB, *dC;
    REQ(hipMalloc(&dA, aN * sizeof(hipblasComplex)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dB, bN * sizeof(hipblasComplex)) == hipSuccess, "malloc");
    REQ(hipMalloc(&dC, cN * sizeof(hipblasComplex)) == hipSuccess, "malloc");
    REQ(hipMemcpy(dA, A.data(), aN * sizeof(hipblasComplex),
                  hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dB, B.data(), bN * sizeof(hipblasComplex),
                  hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dC, C.data(), cN * sizeof(hipblasComplex),
                  hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasCgemm(h, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC,
                     ldc)
            == HIPBLAS_STATUS_SUCCESS,
        "cgemm");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<hipblasComplex> C_out(cN);
    REQ(hipMemcpy(C_out.data(), dC, cN * sizeof(hipblasComplex),
                  hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    REQ((closeComplex<hipblasComplex, float>(C_out, C_ref, 5e-4f)),
        "cgemm mismatch");
    return true;
}

#if defined(CHIPBLAS_HAS_FP64)
static bool test_zgemm_cc() {
    hipblasOperation_t opA = HIPBLAS_OP_C, opB = HIPBLAS_OP_C;
    int m = 32, n = 24, k = 16;
    int lda = (opA == HIPBLAS_OP_N) ? m : k;
    int ldb = (opB == HIPBLAS_OP_N) ? k : n;
    int ldc = m;
    int aCols = (opA == HIPBLAS_OP_N) ? k : m;
    int bCols = (opB == HIPBLAS_OP_N) ? n : k;
    size_t aN = (size_t)lda * aCols, bN = (size_t)ldb * bCols, cN = (size_t)ldc * n;
    hipblasDoubleComplex alpha = {0.75, -0.25}, beta = {-0.5, 0.125};
    std::vector<hipblasDoubleComplex> A(aN), B(bN), C(cN), C_ref;
    for (size_t i = 0; i < aN; ++i) {
        A[i].x = fillD((int)i * 2, 61);
        A[i].y = fillD((int)i * 2 + 1, 61);
    }
    for (size_t i = 0; i < bN; ++i) {
        B[i].x = fillD((int)i * 2, 62);
        B[i].y = fillD((int)i * 2 + 1, 62);
    }
    for (size_t i = 0; i < cN; ++i) {
        C[i].x = fillD((int)i * 2, 63);
        C[i].y = fillD((int)i * 2 + 1, 63);
    }
    C_ref = C;
    gemmHostComplex<double, hipblasDoubleComplex>(
        opA, opB, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta,
        C_ref.data(), ldc);

    hipblasDoubleComplex *dA, *dB, *dC;
    REQ(hipMalloc(&dA, aN * sizeof(hipblasDoubleComplex)) == hipSuccess,
        "malloc");
    REQ(hipMalloc(&dB, bN * sizeof(hipblasDoubleComplex)) == hipSuccess,
        "malloc");
    REQ(hipMalloc(&dC, cN * sizeof(hipblasDoubleComplex)) == hipSuccess,
        "malloc");
    REQ(hipMemcpy(dA, A.data(), aN * sizeof(hipblasDoubleComplex),
                  hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dB, B.data(), bN * sizeof(hipblasDoubleComplex),
                  hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    REQ(hipMemcpy(dC, C.data(), cN * sizeof(hipblasDoubleComplex),
                  hipMemcpyHostToDevice)
            == hipSuccess,
        "h2d");
    hipblasHandle_t h;
    REQ(hipblasCreate(&h) == HIPBLAS_STATUS_SUCCESS, "create");
    REQ(hipblasZgemm(h, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta,
                     dC, ldc)
            == HIPBLAS_STATUS_SUCCESS,
        "zgemm");
    REQ(hipblasDestroy(h) == HIPBLAS_STATUS_SUCCESS, "destroy");
    std::vector<hipblasDoubleComplex> C_out(cN);
    REQ(hipMemcpy(C_out.data(), dC, cN * sizeof(hipblasDoubleComplex),
                  hipMemcpyDeviceToHost)
            == hipSuccess,
        "d2h");
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    REQ((closeComplex<hipblasDoubleComplex, double>(C_out, C_ref, 1e-10)),
        "zgemm mismatch");
    return true;
}
#endif

int main(int argc, char** argv) {
    bool ok = true;
    using chipblas_test::report;
#define RUN(slug, name, fn)                                                        \
    if (chipblas_test::should_run_case(argc, argv, slug)) {                      \
        bool _p = (fn)();                                                          \
        report(name, _p);                                                          \
        ok &= _p;                                                                  \
        if (chipblas_test::case_filter_active(argc, argv))                         \
            return ok ? 0 : 1;                                                     \
    }                                                                              \
    do {                                                                           \
    } while (0)

    RUN("conformance:lifecycle-create-destroy",
        "lifecycle create/destroy",
        test_lifecycle_create_destroy);
    RUN("conformance:lifecycle-destroy-null", "lifecycle destroy null",
        test_lifecycle_destroy_null);
    RUN("conformance:lifecycle-stream", "lifecycle stream roundtrip",
        test_lifecycle_stream_roundtrip);
    RUN("conformance:lifecycle-pointer-mode", "lifecycle pointer mode",
        test_lifecycle_pointer_mode);
    RUN("conformance:lifecycle-version", "lifecycle version", test_lifecycle_version);
    RUN("conformance:saxpy-inc1", "saxpy inc1", test_saxpy_inc1);
    RUN("conformance:saxpy-strided", "saxpy strided", test_saxpy_strided);
#if defined(CHIPBLAS_HAS_FP64)
    RUN("conformance:daxpy", "daxpy", test_daxpy);
#endif
    RUN("conformance:sscal", "sscal", test_sscal);
#if defined(CHIPBLAS_HAS_FP64)
    RUN("conformance:dscal", "dscal", test_dscal);
#endif
    RUN("conformance:sgemv-N", "sgemv N", test_sgemv_n);
#if defined(CHIPBLAS_HAS_FP64)
    RUN("conformance:dgemv-T", "dgemv T", test_dgemv_t);
#endif
    RUN("conformance:sgemm-NN", "sgemm NN", test_sgemm_nn);
    RUN("conformance:sgemm-NT", "sgemm NT", test_sgemm_nt);
    RUN("conformance:sgemm-TN", "sgemm TN", test_sgemm_tn);
    RUN("conformance:sgemm-TT", "sgemm TT", test_sgemm_tt);
#if defined(CHIPBLAS_HAS_FP64)
    RUN("conformance:dgemm-NN", "dgemm NN", test_dgemm_nn);
#endif
    RUN("conformance:cgemm-NN", "cgemm NN", test_cgemm_nn);
#if defined(CHIPBLAS_HAS_FP64)
    RUN("conformance:zgemm-CC", "zgemm CC", test_zgemm_cc);
#endif
#undef RUN
    if (chipblas_test::case_filter_active(argc, argv)) {
        std::fprintf(stderr, "unknown conformance case \"%s\"\n", argv[1]);
        return 2;
    }
    return ok ? 0 : 1;
}
