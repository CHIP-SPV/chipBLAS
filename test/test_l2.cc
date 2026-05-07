// L2 BLAS correctness vs. CPU reference: Sgemv, Dgemv, with both
// no-transpose and transpose ops, plus a non-unit increment case.
//
// SPDX-License-Identifier: MIT

#include "test_common.hh"

using namespace chipblas_test;

namespace {

// y ← alpha*op(A)*x + beta*y, column-major. lda is the leading dimension
// of the storage (rows in column-major).
template <class T>
void gemvHost(hipblasOperation_t op, int m, int n, T alpha,
              const T* A, int lda, const T* x, int incx,
              T beta, T* y, int incy) {
    int yLen = (op == HIPBLAS_OP_N) ? m : n;
    int xLen = (op == HIPBLAS_OP_N) ? n : m;
    for (int i = 0; i < yLen; ++i) y[i * incy] *= beta;
    for (int j = 0; j < xLen; ++j) {
        T xj = x[j * incx];
        for (int i = 0; i < yLen; ++i) {
            T a;
            if (op == HIPBLAS_OP_N) a = A[(size_t)j * lda + i];
            else                    a = A[(size_t)i * lda + j];
            y[i * incy] += alpha * a * xj;
        }
    }
}

template <class T>
size_t storage(int n, int inc) { return (size_t)(n - 1) * (size_t)inc + 1; }

bool runSgemv(hipblasOperation_t op, int m, int n, int incx, int incy) {
    int lda = m;
    float alpha = 1.5f, beta = -0.25f;
    int yLen = (op == HIPBLAS_OP_N) ? m : n;
    int xLen = (op == HIPBLAS_OP_N) ? n : m;
    size_t aN = (size_t)lda * n;
    size_t xN = storage<float>(xLen, incx);
    size_t yN = storage<float>(yLen, incy);
    std::vector<float> A(aN), x(xN), y(yN), y_ref;
    for (size_t i = 0; i < aN; ++i) A[i] = fillF((int)i, 11);
    for (size_t i = 0; i < xN; ++i) x[i] = fillF((int)i, 12);
    for (size_t i = 0; i < yN; ++i) y[i] = fillF((int)i, 13);
    y_ref = y;
    gemvHost<float>(op, m, n, alpha, A.data(), lda,
                    x.data(), incx, beta, y_ref.data(), incy);

    float *dA, *dX, *dY;
    CHECK_HIP(hipMalloc(&dA, aN * sizeof(float)));
    CHECK_HIP(hipMalloc(&dX, xN * sizeof(float)));
    CHECK_HIP(hipMalloc(&dY, yN * sizeof(float)));
    CHECK_HIP(hipMemcpy(dA, A.data(), aN * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dX, x.data(), xN * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dY, y.data(), yN * sizeof(float),
                        hipMemcpyHostToDevice));

    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasSgemv(h, op, m, n, &alpha,
                            dA, lda, dX, incx, &beta, dY, incy));
    CHECK_BLAS(hipblasDestroy(h));

    std::vector<float> y_out(yN);
    CHECK_HIP(hipMemcpy(y_out.data(), dY, yN * sizeof(float),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
    return closeReal<float>(y_out, y_ref, 5e-4f);
}

bool runDgemv(hipblasOperation_t op, int m, int n, int incx, int incy) {
    int lda = m;
    double alpha = 0.875, beta = 0.125;
    int yLen = (op == HIPBLAS_OP_N) ? m : n;
    int xLen = (op == HIPBLAS_OP_N) ? n : m;
    size_t aN = (size_t)lda * n;
    size_t xN = storage<double>(xLen, incx);
    size_t yN = storage<double>(yLen, incy);
    std::vector<double> A(aN), x(xN), y(yN), y_ref;
    for (size_t i = 0; i < aN; ++i) A[i] = fillD((int)i, 21);
    for (size_t i = 0; i < xN; ++i) x[i] = fillD((int)i, 22);
    for (size_t i = 0; i < yN; ++i) y[i] = fillD((int)i, 23);
    y_ref = y;
    gemvHost<double>(op, m, n, alpha, A.data(), lda,
                     x.data(), incx, beta, y_ref.data(), incy);

    double *dA, *dX, *dY;
    CHECK_HIP(hipMalloc(&dA, aN * sizeof(double)));
    CHECK_HIP(hipMalloc(&dX, xN * sizeof(double)));
    CHECK_HIP(hipMalloc(&dY, yN * sizeof(double)));
    CHECK_HIP(hipMemcpy(dA, A.data(), aN * sizeof(double),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dX, x.data(), xN * sizeof(double),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dY, y.data(), yN * sizeof(double),
                        hipMemcpyHostToDevice));

    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasDgemv(h, op, m, n, &alpha,
                            dA, lda, dX, incx, &beta, dY, incy));
    CHECK_BLAS(hipblasDestroy(h));

    std::vector<double> y_out(yN);
    CHECK_HIP(hipMemcpy(y_out.data(), dY, yN * sizeof(double),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
    return closeReal<double>(y_out, y_ref, 1e-11);
}

} // namespace

int main() {
    bool ok = true, a;
    a = runSgemv(HIPBLAS_OP_N, 64, 48, 1, 1); report("Sgemv N 64x48 inc=1,1", a); ok &= a;
    a = runSgemv(HIPBLAS_OP_T, 64, 48, 1, 1); report("Sgemv T 64x48 inc=1,1", a); ok &= a;
    a = runSgemv(HIPBLAS_OP_N, 33, 27, 2, 3); report("Sgemv N 33x27 inc=2,3", a); ok &= a;
    a = runSgemv(HIPBLAS_OP_T, 33, 27, 2, 3); report("Sgemv T 33x27 inc=2,3", a); ok &= a;
    a = runDgemv(HIPBLAS_OP_N, 64, 48, 1, 1); report("Dgemv N 64x48 inc=1,1", a); ok &= a;
    a = runDgemv(HIPBLAS_OP_T, 64, 48, 1, 1); report("Dgemv T 64x48 inc=1,1", a); ok &= a;
    a = runDgemv(HIPBLAS_OP_T, 33, 27, 2, 1); report("Dgemv T 33x27 inc=2,1", a); ok &= a;
    return ok ? 0 : 1;
}
