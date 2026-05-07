// chipBLAS — Level 3 BLAS routines (GEMM and friends).
//
// hipBLAS uses column-major storage (matching cuBLAS). CLBlast supports
// column-major directly via CLBlastLayoutColMajor, so we map straight
// across; no transpose dance needed.
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"

#include <hip/hip_runtime.h>

#include <cstddef>

using chipblas::BufDir;
using chipblas::Handle;
using chipblas::StagedBuffer;

namespace {

CLBlastTranspose mapTranspose(hipblasOperation_t op) {
    switch (op) {
    case HIPBLAS_OP_N: return CLBlastTransposeNo;
    case HIPBLAS_OP_T: return CLBlastTransposeYes;
    case HIPBLAS_OP_C: return CLBlastTransposeConjugate;
    }
    return CLBlastTransposeNo;
}

// Column-major: op(A) is m×k logically; physically A occupies lda rows by
// (k if op_a==N else m) columns, so the byte footprint is lda * cols.
size_t gemmAByteCount(hipblasOperation_t op, int m, int k, int lda,
                      size_t elemBytes) {
    int cols = (op == HIPBLAS_OP_N) ? k : m;
    return static_cast<size_t>(lda) * static_cast<size_t>(cols) * elemBytes;
}
size_t gemmBByteCount(hipblasOperation_t op, int k, int n, int ldb,
                      size_t elemBytes) {
    int cols = (op == HIPBLAS_OP_N) ? n : k;
    return static_cast<size_t>(ldb) * static_cast<size_t>(cols) * elemBytes;
}
size_t gemmCByteCount(int /*m*/, int n, int ldc, size_t elemBytes) {
    return static_cast<size_t>(ldc) * static_cast<size_t>(n) * elemBytes;
}

// Common bulk: validate, stage, dispatch via a typed callable, write back.
// `Dispatch` is invoked as: int dispatch(cl_mem A, cl_mem B, cl_mem C,
//                                        cl_command_queue* q);
template <class Dispatch>
hipblasStatus_t gemmRun(hipblasHandle_t handle,
                        hipblasOperation_t /*transA*/,
                        hipblasOperation_t /*transB*/,
                        size_t aBytes, size_t bBytes, size_t cBytes,
                        const void* A, const void* B, void* C,
                        Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!A || !B || !C) return HIPBLAS_STATUS_INVALID_VALUE;

    StagedBuffer sa, sb, sc;
    auto rc = chipblas::bridgeStage(*h, const_cast<void*>(A), aBytes,
                                    BufDir::IN, &sa);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, const_cast<void*>(B), bBytes,
                               BufDir::IN, &sb);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        return rc;
    }
    rc = chipblas::bridgeStage(*h, C, cBytes, BufDir::INOUT, &sc);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        chipblas::bridgeWriteBack(*h, sb);
        return rc;
    }

    cl_command_queue queue = h->queue;
    int clb = dispatch(sa, sb, sc, &queue);

    // Inputs: just release. Output: read back to HIP and release.
    chipblas::bridgeWriteBack(*h, sa);
    chipblas::bridgeWriteBack(*h, sb);
    auto wb = chipblas::bridgeWriteBack(*h, sc);
    auto translated = chipblas::translate(clb);
    return (translated != HIPBLAS_STATUS_SUCCESS) ? translated : wb;
}

} // namespace

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

} // extern "C"
