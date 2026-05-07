// chipBLAS — Level 2 BLAS routines (matrix-vector).
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"

#include <hip/hip_runtime.h>

#include <cstdlib>

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

// Footprint for a vector with `len` logical elements and stride `inc`. We
// model a contiguous range from offset 0 to (len-1)*|inc|+1 elements; v0
// only supports inc > 0.
size_t vecBytes(int len, int inc, size_t elemBytes) {
    int absInc = inc < 0 ? -inc : inc;
    if (len <= 0 || absInc < 1) return 0;
    return (static_cast<size_t>(len - 1) * absInc + 1) * elemBytes;
}

template <class Dispatch>
hipblasStatus_t gemvRun(hipblasHandle_t handle, hipblasOperation_t trans,
                        int m, int n, int incx, int incy,
                        size_t aBytes, size_t elemBytes,
                        const void* A, const void* x, void* y,
                        Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0 || incy <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!A || !x || !y) return HIPBLAS_STATUS_INVALID_VALUE;

    int xLen = (trans == HIPBLAS_OP_N) ? n : m;
    int yLen = (trans == HIPBLAS_OP_N) ? m : n;

    StagedBuffer sa, sx, sy;
    auto rc = chipblas::bridgeStage(*h, const_cast<void*>(A), aBytes,
                                    BufDir::IN, &sa);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, const_cast<void*>(x),
                               vecBytes(xLen, incx, elemBytes),
                               BufDir::IN, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        return rc;
    }
    rc = chipblas::bridgeStage(*h, y, vecBytes(yLen, incy, elemBytes),
                               BufDir::INOUT, &sy);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        chipblas::bridgeWriteBack(*h, sx);
        return rc;
    }

    cl_command_queue queue = h->queue;
    int clb = dispatch(sa, sx, sy, &queue);

    chipblas::bridgeWriteBack(*h, sa);
    chipblas::bridgeWriteBack(*h, sx);
    auto wb = chipblas::bridgeWriteBack(*h, sy);
    auto translated = chipblas::translate(clb);
    return (translated != HIPBLAS_STATUS_SUCCESS) ? translated : wb;
}

} // namespace

extern "C" {

hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n,
                             const float* alpha,
                             const float* A, int lda,
                             const float* x, int incx,
                             const float* beta,
                             float* y, int incy) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n)
                  * sizeof(float);
    return gemvRun(handle, trans, m, n, incx, incy, aBytes, sizeof(float),
        A, x, y,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& X_,
            chipblas::StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastSgemv(
                CLBlastLayoutColMajor, mapTranspose(trans),
                (size_t)m, (size_t)n,
                *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx,
                *beta,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n,
                             const double* alpha,
                             const double* A, int lda,
                             const double* x, int incx,
                             const double* beta,
                             double* y, int incy) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n)
                  * sizeof(double);
    return gemvRun(handle, trans, m, n, incx, incy, aBytes, sizeof(double),
        A, x, y,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& X_,
            chipblas::StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDgemv(
                CLBlastLayoutColMajor, mapTranspose(trans),
                (size_t)m, (size_t)n,
                *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx,
                *beta,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

} // extern "C"
