// chipBLAS public header — hipBLAS C API surface implemented via CLBlast
// on chipStar (OpenCL SVM bridge). See src/hipblas_*.cc for coverage details.
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_HIPBLAS_H
#define CHIPBLAS_HIPBLAS_H

#include <hip/hip_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HIPBLAS_STATUS_SUCCESS           = 0,
    HIPBLAS_STATUS_NOT_INITIALIZED   = 1,
    HIPBLAS_STATUS_ALLOC_FAILED      = 2,
    HIPBLAS_STATUS_INVALID_VALUE     = 3,
    HIPBLAS_STATUS_MAPPING_ERROR     = 4,
    HIPBLAS_STATUS_EXECUTION_FAILED  = 5,
    HIPBLAS_STATUS_INTERNAL_ERROR    = 6,
    HIPBLAS_STATUS_NOT_SUPPORTED     = 7,
    HIPBLAS_STATUS_ARCH_MISMATCH     = 8,
    HIPBLAS_STATUS_HANDLE_IS_NULLPTR = 9,
    HIPBLAS_STATUS_INVALID_ENUM      = 10,
    HIPBLAS_STATUS_UNKNOWN           = 11
} hipblasStatus_t;

typedef enum {
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    HIPBLAS_OP_C = 113
} hipblasOperation_t;

typedef enum {
    HIPBLAS_FILL_MODE_UPPER = 121,
    HIPBLAS_FILL_MODE_LOWER = 122,
    HIPBLAS_FILL_MODE_FULL  = 123
} hipblasFillMode_t;

typedef enum {
    HIPBLAS_DIAG_NON_UNIT = 131,
    HIPBLAS_DIAG_UNIT     = 132
} hipblasDiagType_t;

typedef enum {
    HIPBLAS_SIDE_LEFT  = 141,
    HIPBLAS_SIDE_RIGHT = 142,
    HIPBLAS_SIDE_BOTH  = 143
} hipblasSideMode_t;

typedef enum {
    HIPBLAS_POINTER_MODE_HOST   = 0,
    HIPBLAS_POINTER_MODE_DEVICE = 1
} hipblasPointerMode_t;

typedef struct { float  x, y; } hipblasComplex;
typedef struct { double x, y; } hipblasDoubleComplex;
typedef uint16_t hipblasHalf;

typedef struct hipblasContext* hipblasHandle_t;

// --- Lifecycle ---------------------------------------------------------------
hipblasStatus_t hipblasCreate(hipblasHandle_t* handle);
hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t stream);
hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* stream);
hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle,
                                      hipblasPointerMode_t mode);
hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle,
                                      hipblasPointerMode_t* mode);
hipblasStatus_t hipblasGetVersion(hipblasHandle_t handle, int* version);

// --- Level 1 ----------------------------------------------------------------
hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy);
hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double* alpha,
                             const double* x, int incx, double* y, int incy);
hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x, int incx,
                             hipblasComplex* y, int incy);
hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x, int incx,
                             hipblasDoubleComplex* y, int incy);
hipblasStatus_t hipblasHaxpy(hipblasHandle_t handle, int n,
                             const hipblasHalf* alpha, const hipblasHalf* x,
                             int incx, hipblasHalf* y, int incy);

hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha,
                             float* x, int incx);
hipblasStatus_t hipblasDscal(hipblasHandle_t handle, int n, const double* alpha,
                             double* x, int incx);
hipblasStatus_t hipblasCscal(hipblasHandle_t handle, int n,
                             const hipblasComplex* alpha, hipblasComplex* x,
                             int incx);
hipblasStatus_t hipblasZscal(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* alpha,
                             hipblasDoubleComplex* x, int incx);
hipblasStatus_t hipblasHscal(hipblasHandle_t handle, int n,
                             const hipblasHalf* alpha, hipblasHalf* x, int incx);
hipblasStatus_t hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha,
                              hipblasComplex* x, int incx);
hipblasStatus_t hipblasZdscal(hipblasHandle_t handle, int n, const double* alpha,
                              hipblasDoubleComplex* x, int incx);

hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx,
                             float* y, int incy);
hipblasStatus_t hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx,
                             double* y, int incy);
hipblasStatus_t hipblasCswap(hipblasHandle_t handle, int n, hipblasComplex* x,
                             int incx, hipblasComplex* y, int incy);
hipblasStatus_t hipblasZswap(hipblasHandle_t handle, int n,
                             hipblasDoubleComplex* x, int incx,
                             hipblasDoubleComplex* y, int incy);
hipblasStatus_t hipblasHswap(hipblasHandle_t handle, int n, hipblasHalf* x,
                             int incx, hipblasHalf* y, int incy);

hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float* x,
                             int incx, float* y, int incy);
hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double* x,
                             int incx, double* y, int incy);
hipblasStatus_t hipblasCcopy(hipblasHandle_t handle, int n,
                             const hipblasComplex* x, int incx,
                             hipblasComplex* y, int incy);
hipblasStatus_t hipblasZcopy(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* x, int incx,
                             hipblasDoubleComplex* y, int incy);
hipblasStatus_t hipblasHcopy(hipblasHandle_t handle, int n, const hipblasHalf* x,
                             int incx, hipblasHalf* y, int incy);

hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, const float* x,
                             int incx, const float* y, int incy, float* result);
hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n, const double* x,
                             int incx, const double* y, int incy, double* result);
hipblasStatus_t hipblasCdotu(hipblasHandle_t handle, int n,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* y, int incy,
                             hipblasComplex* result);
hipblasStatus_t hipblasCdotc(hipblasHandle_t handle, int n,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* y, int incy,
                             hipblasComplex* result);
hipblasStatus_t hipblasZdotu(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* y, int incy,
                             hipblasDoubleComplex* result);
hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* y, int incy,
                             hipblasDoubleComplex* result);

hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x,
                             int incx, float* result);
hipblasStatus_t hipblasDnrm2(hipblasHandle_t handle, int n, const double* x,
                             int incx, double* result);
hipblasStatus_t hipblasScnrm2(hipblasHandle_t handle, int n,
                              const hipblasComplex* x, int incx, float* result);
hipblasStatus_t hipblasDznrm2(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              double* result);

hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x,
                             int incx, float* result);
hipblasStatus_t hipblasDasum(hipblasHandle_t handle, int n, const double* x,
                             int incx, double* result);
hipblasStatus_t hipblasScasum(hipblasHandle_t handle, int n,
                              const hipblasComplex* x, int incx, float* result);
hipblasStatus_t hipblasDzasum(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              double* result);

hipblasStatus_t hipblasSrot(hipblasHandle_t handle, int n, float* x, int incx,
                            float* y, int incy, const float* c, const float* s);
hipblasStatus_t hipblasDrot(hipblasHandle_t handle, int n, double* x, int incx,
                            double* y, int incy, const double* c,
                            const double* s);
hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b,
                             float* c, float* s);
hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b,
                             double* c, double* s);

hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x,
                              int incx, int* result);
hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x,
                              int incx, int* result);
hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n,
                              const hipblasComplex* x, int incx, int* result);
hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              int* result);
hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x,
                              int incx, int* result);
hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x,
                              int incx, int* result);
hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n,
                                const hipblasComplex* x, int incx, int* result);
hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              int* result);

// --- Level 2 ----------------------------------------------------------------
hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n, const float* alpha, const float* A,
                             int lda, const float* x, int incx,
                             const float* beta, float* y, int incy);
hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n, const double* alpha, const double* A,
                             int lda, const double* x, int incx,
                             const double* beta, double* y, int incy);
hipblasStatus_t hipblasCgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n, const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* beta, hipblasComplex* y,
                             int incy);
hipblasStatus_t hipblasZgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n, const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* y, int incy);
hipblasStatus_t hipblasHgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n, const hipblasHalf* alpha,
                             const hipblasHalf* A, int lda, const hipblasHalf* x,
                             int incx, const hipblasHalf* beta, hipblasHalf* y,
                             int incy);

hipblasStatus_t hipblasSger(hipblasHandle_t handle, int m, int n,
                            const float* alpha, const float* x, int incx,
                            const float* y, int incy, float* A, int lda);
hipblasStatus_t hipblasDger(hipblasHandle_t handle, int m, int n,
                            const double* alpha, const double* x, int incx,
                            const double* y, int incy, double* A, int lda);
hipblasStatus_t hipblasCgeru(hipblasHandle_t handle, int m, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* y, int incy,
                             hipblasComplex* A, int lda);
hipblasStatus_t hipblasCgerc(hipblasHandle_t handle, int m, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* y, int incy,
                             hipblasComplex* A, int lda);
hipblasStatus_t hipblasZgeru(hipblasHandle_t handle, int m, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* y, int incy,
                             hipblasDoubleComplex* A, int lda);
hipblasStatus_t hipblasZgerc(hipblasHandle_t handle, int m, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* y, int incy,
                             hipblasDoubleComplex* A, int lda);

hipblasStatus_t hipblasStrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const float* A, int lda, float* x,
                             int incx);
hipblasStatus_t hipblasDtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const double* A, int lda, double* x,
                             int incx);
hipblasStatus_t hipblasCtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const hipblasComplex* A, int lda,
                             hipblasComplex* x, int incx);
hipblasStatus_t hipblasZtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const hipblasDoubleComplex* A, int lda,
                             hipblasDoubleComplex* x, int incx);

hipblasStatus_t hipblasStrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const float* A, int lda, float* x,
                             int incx);
hipblasStatus_t hipblasDtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const double* A, int lda, double* x,
                             int incx);
hipblasStatus_t hipblasCtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const hipblasComplex* A, int lda,
                             hipblasComplex* x, int incx);
hipblasStatus_t hipblasZtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t transA, hipblasDiagType_t diag,
                             int n, const hipblasDoubleComplex* A, int lda,
                             hipblasDoubleComplex* x, int incx);

// --- Level 3 ----------------------------------------------------------------
hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB, int m, int n, int k,
                             const float* alpha, const float* A, int lda,
                             const float* B, int ldb, const float* beta,
                             float* C, int ldc);
hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB, int m, int n, int k,
                             const double* alpha, const double* A, int lda,
                             const double* B, int ldb, const double* beta,
                             double* C, int ldc);
hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB, int m, int n, int k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* B, int ldb,
                             const hipblasComplex* beta, hipblasComplex* C,
                             int ldc);
hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB, int m, int n, int k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* B, int ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* C, int ldc);
hipblasStatus_t hipblasHgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB, int m, int n, int k,
                             const hipblasHalf* alpha, const hipblasHalf* A,
                             int lda, const hipblasHalf* B, int ldb,
                             const hipblasHalf* beta, hipblasHalf* C, int ldc);

hipblasStatus_t hipblasSsymm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, int m, int n,
                             const float* alpha, const float* A, int lda,
                             const float* B, int ldb, const float* beta,
                             float* C, int ldc);
hipblasStatus_t hipblasDsymm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, int m, int n,
                             const double* alpha, const double* A, int lda,
                             const double* B, int ldb, const double* beta,
                             double* C, int ldc);
hipblasStatus_t hipblasCsymm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, int m, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* B, int ldb,
                             const hipblasComplex* beta, hipblasComplex* C,
                             int ldc);
hipblasStatus_t hipblasZsymm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, int m, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* B, int ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* C, int ldc);
hipblasStatus_t hipblasHsymm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, int m, int n,
                             const hipblasHalf* alpha, const hipblasHalf* A,
                             int lda, const hipblasHalf* B, int ldb,
                             const hipblasHalf* beta, hipblasHalf* C, int ldc);

hipblasStatus_t hipblasChemm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, int m, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* B, int ldb,
                             const hipblasComplex* beta, hipblasComplex* C,
                             int ldc);
hipblasStatus_t hipblasZhemm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, int m, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* B, int ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* C, int ldc);

hipblasStatus_t hipblasSsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t trans, int n, int k,
                             const float* alpha, const float* A, int lda,
                             const float* beta, float* C, int ldc);
hipblasStatus_t hipblasDsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t trans, int n, int k,
                             const double* alpha, const double* A, int lda,
                             const double* beta, double* C, int ldc);
hipblasStatus_t hipblasCsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t trans, int n, int k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* beta, hipblasComplex* C,
                             int ldc);
hipblasStatus_t hipblasZsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t trans, int n, int k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* C, int ldc);
hipblasStatus_t hipblasHsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t trans, int n, int k,
                             const hipblasHalf* alpha, const hipblasHalf* A,
                             int lda, const hipblasHalf* beta, hipblasHalf* C,
                             int ldc);

hipblasStatus_t hipblasCherk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t trans, int n, int k,
                             const float* alpha, const hipblasComplex* A,
                             int lda, const float* beta, hipblasComplex* C,
                             int ldc);
hipblasStatus_t hipblasZherk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                             hipblasOperation_t trans, int n, int k,
                             const double* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const double* beta, hipblasDoubleComplex* C,
                             int ldc);

hipblasStatus_t hipblasSsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, int n, int k,
                              const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta,
                              float* C, int ldc);
hipblasStatus_t hipblasDsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, int n, int k,
                              const double* alpha, const double* A, int lda,
                              const double* B, int ldb, const double* beta,
                              double* C, int ldc);
hipblasStatus_t hipblasCsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, int n, int k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A, int lda,
                              const hipblasComplex* B, int ldb,
                              const hipblasComplex* beta, hipblasComplex* C,
                              int ldc);
hipblasStatus_t hipblasZsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, int n, int k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A, int lda,
                              const hipblasDoubleComplex* B, int ldb,
                              const hipblasDoubleComplex* beta,
                              hipblasDoubleComplex* C, int ldc);
hipblasStatus_t hipblasHsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, int n, int k,
                              const hipblasHalf* alpha, const hipblasHalf* A,
                              int lda, const hipblasHalf* B, int ldb,
                              const hipblasHalf* beta, hipblasHalf* C, int ldc);

hipblasStatus_t hipblasCher2k(hipblasHandle_t handle, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, int n, int k,
                              const hipblasComplex* alpha,
                              const hipblasComplex* A, int lda,
                              const hipblasComplex* B, int ldb,
                              const float* beta, hipblasComplex* C, int ldc);
hipblasStatus_t hipblasZher2k(hipblasHandle_t handle, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, int n, int k,
                              const hipblasDoubleComplex* alpha,
                              const hipblasDoubleComplex* A, int lda,
                              const hipblasDoubleComplex* B, int ldb,
                              const double* beta, hipblasDoubleComplex* C,
                              int ldc);

hipblasStatus_t hipblasStrmm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const float* alpha, const float* A, int lda,
                             float* B, int ldb);
hipblasStatus_t hipblasDtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const double* alpha, const double* A, int lda,
                             double* B, int ldb);
hipblasStatus_t hipblasCtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             hipblasComplex* B, int ldb);
hipblasStatus_t hipblasZtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             hipblasDoubleComplex* B, int ldb);
hipblasStatus_t hipblasHtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const hipblasHalf* alpha, const hipblasHalf* A,
                             int lda, hipblasHalf* B, int ldb);

hipblasStatus_t hipblasStrsm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const float* alpha, const float* A, int lda,
                             float* B, int ldb);
hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const double* alpha, const double* A, int lda,
                             double* B, int ldb);
hipblasStatus_t hipblasCtrsm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             hipblasComplex* B, int ldb);
hipblasStatus_t hipblasZtrsm(hipblasHandle_t handle, hipblasSideMode_t side,
                             hipblasFillMode_t uplo, hipblasOperation_t transA,
                             hipblasDiagType_t diag, int m, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             hipblasDoubleComplex* B, int ldb);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CHIPBLAS_HIPBLAS_H
