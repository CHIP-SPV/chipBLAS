// chipBLAS — Level 3 BLAS routines (GEMM and friends).
//
// hipBLAS uses column-major storage (matching cuBLAS). CLBlast supports
// column-major directly via CLBlastLayoutColMajor, so we map straight
// across; no transpose dance needed.
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"
#include "hipblas_clblast_common.hh"
#include "hipblas_matmul_bridge.hh"

#include <hip/hip_runtime.h>

#include <cstring>

#include <cstddef>

using hipblas_clblast::mapTranspose;
using hipblas_mm::gemmAByteCount;
using hipblas_mm::gemmBByteCount;
using hipblas_mm::gemmCByteCount;
using hipblas_mm::gemmRun;

extern "C" {

hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const float* alpha,
                             const float* A, int lda,
                             const float* B, int ldb,
                             const float* beta,
                             float* C, int ldc) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return gemmRun(handle, transA, transB,
        gemmAByteCount(transA, m, k, lda, sizeof(float)),
        gemmBByteCount(transB, k, n, ldb, sizeof(float)),
        gemmCByteCount(m, n, ldc, sizeof(float)),
        A, B, C,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& B_,
            chipblas::StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastSgemm(
                CLBlastLayoutColMajor,
                mapTranspose(transA), mapTranspose(transB),
                (size_t)m, (size_t)n, (size_t)k,
                *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta,
                C_.mem, C_.offset / E, (size_t)ldc,
                q, nullptr);
        });
}

hipblasStatus_t hipblasDgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const double* alpha,
                             const double* A, int lda,
                             const double* B, int ldb,
                             const double* beta,
                             double* C, int ldc) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return gemmRun(handle, transA, transB,
        gemmAByteCount(transA, m, k, lda, sizeof(double)),
        gemmBByteCount(transB, k, n, ldb, sizeof(double)),
        gemmCByteCount(m, n, ldc, sizeof(double)),
        A, B, C,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& B_,
            chipblas::StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDgemm(
                CLBlastLayoutColMajor,
                mapTranspose(transA), mapTranspose(transB),
                (size_t)m, (size_t)n, (size_t)k,
                *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta,
                C_.mem, C_.offset / E, (size_t)ldc,
                q, nullptr);
        });
}

hipblasStatus_t hipblasCgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* B, int ldb,
                             const hipblasComplex* beta,
                             hipblasComplex* C, int ldc) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    cl_float2 b = {{beta->x,  beta->y }};
    return gemmRun(handle, transA, transB,
        gemmAByteCount(transA, m, k, lda, sizeof(hipblasComplex)),
        gemmBByteCount(transB, k, n, ldb, sizeof(hipblasComplex)),
        gemmCByteCount(m, n, ldc, sizeof(hipblasComplex)),
        A, B, C,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& B_,
            chipblas::StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCgemm(
                CLBlastLayoutColMajor,
                mapTranspose(transA), mapTranspose(transB),
                (size_t)m, (size_t)n, (size_t)k,
                a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b,
                C_.mem, C_.offset / E, (size_t)ldc,
                q, nullptr);
        });
}

hipblasStatus_t hipblasZgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* B, int ldb,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* C, int ldc) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    cl_double2 b = {{beta->x,  beta->y }};
    return gemmRun(handle, transA, transB,
        gemmAByteCount(transA, m, k, lda, sizeof(hipblasDoubleComplex)),
        gemmBByteCount(transB, k, n, ldb, sizeof(hipblasDoubleComplex)),
        gemmCByteCount(m, n, ldc, sizeof(hipblasDoubleComplex)),
        A, B, C,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& B_,
            chipblas::StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZgemm(
                CLBlastLayoutColMajor,
                mapTranspose(transA), mapTranspose(transB),
                (size_t)m, (size_t)n, (size_t)k,
                a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b,
                C_.mem, C_.offset / E, (size_t)ldc,
                q, nullptr);
        });
}

hipblasStatus_t hipblasHgemm(hipblasHandle_t handle,
                             hipblasOperation_t transA,
                             hipblasOperation_t transB,
                             int m, int n, int k,
                             const hipblasHalf* alpha,
                             const hipblasHalf* A, int lda,
                             const hipblasHalf* B, int ldb,
                             const hipblasHalf* beta,
                             hipblasHalf* C, int ldc) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_half ah {};
    cl_half bh {};
    std::memcpy(&ah, alpha, sizeof(ah));
    std::memcpy(&bh, beta, sizeof(bh));
    return gemmRun(handle, transA, transB,
        gemmAByteCount(transA, m, k, lda, sizeof(hipblasHalf)),
        gemmBByteCount(transB, k, n, ldb, sizeof(hipblasHalf)),
        gemmCByteCount(m, n, ldc, sizeof(hipblasHalf)),
        A, B, C,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& B_,
            chipblas::StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            return CLBlastHgemm(
                CLBlastLayoutColMajor,
                mapTranspose(transA), mapTranspose(transB),
                (size_t)m, (size_t)n, (size_t)k,
                ah,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                bh,
                C_.mem, C_.offset / E, (size_t)ldc,
                q, nullptr);
        });
}

} // extern "C"
