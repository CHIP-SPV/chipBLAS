// chipBLAS public header — a subset of the ROCm hipBLAS C API, sufficient
// to drive CLBlast for the routines chipBLAS implements today.
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_HIPBLAS_H
#define CHIPBLAS_HIPBLAS_H

#include <hip/hip_runtime_api.h>

#include <stddef.h>

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

// --- Level 1: AXPY (y = alpha*x + y) -----------------------------------------
hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n,
                             const float* alpha,
                             const float* x, int incx,
                             float* y, int incy);
hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n,
                             const double* alpha,
                             const double* x, int incx,
                             double* y, int incy);

// --- Level 1: SCAL (x = alpha*x) ---------------------------------------------
hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n,
                             const float* alpha,
                             float* x, int incx);
hipblasStatus_t hipblasDscal(hipblasHandle_t handle, int n,
                             const double* alpha,
                             double* x, int incx);

// --- Level 2: GEMV (y = alpha*op(A)*x + beta*y) ------------------------------
hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n,
                             const float* alpha,
                             const float* A, int lda,
                             const float* x, int incx,
                             const float* beta,
                             float* y, int incy);
hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n,
                             const double* alpha,
                             const double* A, int lda,
                             const double* x, int incx,
                             const double* beta,
                             double* y, int incy);

// --- Level 3: GEMM (C = alpha*op(A)*op(B) + beta*C) --------------------------
hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const float* alpha,
                             const float* A, int lda,
                             const float* B, int ldb,
                             const float* beta,
                             float* C, int ldc);
hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const double* alpha,
                             const double* A, int lda,
                             const double* B, int ldb,
                             const double* beta,
                             double* C, int ldc);
hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* B, int ldb,
                             const hipblasComplex* beta,
                             hipblasComplex* C, int ldc);
hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* B, int ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* C, int ldc);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CHIPBLAS_HIPBLAS_H
