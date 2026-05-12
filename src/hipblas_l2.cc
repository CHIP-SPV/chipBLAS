// chipBLAS — Level 2 BLAS routines (matrix-vector).
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"
#include "hipblas_clblast_common.hh"

#include <hip/hip_runtime.h>

#include <cstring>
#include <cstdlib>

using chipblas::BufDir;
using chipblas::Handle;
using chipblas::StagedBuffer;
using hipblas_clblast::mapTranspose;

namespace {

// Footprint for a vector with `len` logical elements and stride `inc`. We
// model a contiguous range from offset 0 to (len-1)*|inc|+1 elements; v0
// only supports inc > 0.
size_t vecBytes(int len, int inc, size_t elemBytes) {
    return hipblas_clblast::vecBytesElem(len, inc, elemBytes);
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

hipblasStatus_t hipblasCgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* A, int lda,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* beta,
                             hipblasComplex* y, int incy) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n)
                  * sizeof(hipblasComplex);
    return gemvRun(handle, trans, m, n, incx, incy, aBytes,
                   sizeof(hipblasComplex),
        A, x, y,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& X_,
            chipblas::StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            cl_float2 a = {{alpha->x, alpha->y}};
            cl_float2 b = {{beta->x,  beta->y}};
            return CLBlastCgemv(
                CLBlastLayoutColMajor, mapTranspose(trans),
                (size_t)m, (size_t)n,
                a,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx,
                b,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasZgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* A, int lda,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* beta,
                             hipblasDoubleComplex* y, int incy) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n)
                  * sizeof(hipblasDoubleComplex);
    return gemvRun(handle, trans, m, n, incx, incy, aBytes,
                   sizeof(hipblasDoubleComplex),
        A, x, y,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& X_,
            chipblas::StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            cl_double2 a = {{alpha->x, alpha->y}};
            cl_double2 b = {{beta->x,  beta->y}};
            return CLBlastZgemv(
                CLBlastLayoutColMajor, mapTranspose(trans),
                (size_t)m, (size_t)n,
                a,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx,
                b,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasHgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                             int m, int n,
                             const hipblasHalf* alpha,
                             const hipblasHalf* A, int lda,
                             const hipblasHalf* x, int incx,
                             const hipblasHalf* beta,
                             hipblasHalf* y, int incy) {
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n)
                  * sizeof(hipblasHalf);
    return gemvRun(handle, trans, m, n, incx, incy, aBytes,
                   sizeof(hipblasHalf),
        A, x, y,
        [&](chipblas::StagedBuffer& A_, chipblas::StagedBuffer& X_,
            chipblas::StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            cl_half ah {};
            cl_half bh {};
            std::memcpy(&ah, alpha, sizeof(ah));
            std::memcpy(&bh, beta, sizeof(bh));
            return CLBlastHgemv(
                CLBlastLayoutColMajor, mapTranspose(trans),
                (size_t)m, (size_t)n,
                ah,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx,
                bh,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

} // extern "C"
