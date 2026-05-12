// chipBLAS — Level 1 BLAS routines (vector-vector / vector-scalar).
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"

#include <hip/hip_runtime.h>

#include <cstdlib>

using chipblas::BufDir;
using chipblas::Handle;
using chipblas::StagedBuffer;

namespace {

size_t vecBytes(int n, int inc, size_t elemBytes) {
    int absInc = inc < 0 ? -inc : inc;
    if (n <= 0 || absInc < 1) return 0;
    return (static_cast<size_t>(n - 1) * absInc + 1) * elemBytes;
}

template <class Dispatch>
hipblasStatus_t axpyRun(hipblasHandle_t handle, int n, int incx, int incy,
                        size_t elemBytes,
                        const void* x, void* y,
                        Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0 || incy <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!x || !y) return HIPBLAS_STATUS_INVALID_VALUE;

    StagedBuffer sx, sy;
    auto rc = chipblas::bridgeStage(*h, const_cast<void*>(x),
                                    vecBytes(n, incx, elemBytes),
                                    BufDir::IN, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, y, vecBytes(n, incy, elemBytes),
                               BufDir::INOUT, &sy);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sx);
        return rc;
    }

    cl_command_queue queue = h->queue;
    int clb = dispatch(sx, sy, &queue);

    chipblas::bridgeWriteBack(*h, sx);
    auto wb = chipblas::bridgeWriteBack(*h, sy);
    auto translated = chipblas::translate(clb);
    return (translated != HIPBLAS_STATUS_SUCCESS) ? translated : wb;
}

template <class Dispatch>
hipblasStatus_t scalRun(hipblasHandle_t handle, int n, int incx,
                        size_t elemBytes, void* x,
                        Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!x) return HIPBLAS_STATUS_INVALID_VALUE;

    StagedBuffer sx;
    auto rc = chipblas::bridgeStage(*h, x, vecBytes(n, incx, elemBytes),
                                    BufDir::INOUT, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;

    cl_command_queue queue = h->queue;
    int clb = dispatch(sx, &queue);

    auto wb = chipblas::bridgeWriteBack(*h, sx);
    auto translated = chipblas::translate(clb);
    return (translated != HIPBLAS_STATUS_SUCCESS) ? translated : wb;
}

} // namespace

extern "C" {

hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n,
                             const float* alpha,
                             const float* x, int incx,
                             float* y, int incy) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return axpyRun(handle, n, incx, incy, sizeof(float), x, y,
        [&](chipblas::StagedBuffer& X_, chipblas::StagedBuffer& Y_,
            cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastSaxpy((size_t)n, *alpha,
                                X_.mem, X_.offset / E, (size_t)incx,
                                Y_.mem, Y_.offset / E, (size_t)incy,
                                q, nullptr);
        });
}

hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n,
                             const double* alpha,
                             const double* x, int incx,
                             double* y, int incy) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return axpyRun(handle, n, incx, incy, sizeof(double), x, y,
        [&](chipblas::StagedBuffer& X_, chipblas::StagedBuffer& Y_,
            cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDaxpy((size_t)n, *alpha,
                                X_.mem, X_.offset / E, (size_t)incx,
                                Y_.mem, Y_.offset / E, (size_t)incy,
                                q, nullptr);
        });
}

hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n,
                             const float* alpha,
                             float* x, int incx) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return scalRun(handle, n, incx, sizeof(float), x,
        [&](chipblas::StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastSscal((size_t)n, *alpha,
                                X_.mem, X_.offset / E, (size_t)incx,
                                q, nullptr);
        });
}

hipblasStatus_t hipblasDscal(hipblasHandle_t handle, int n,
                             const double* alpha,
                             double* x, int incx) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return scalRun(handle, n, incx, sizeof(double), x,
        [&](chipblas::StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDscal((size_t)n, *alpha,
                                X_.mem, X_.offset / E, (size_t)incx,
                                q, nullptr);
        });
}

} // extern "C"
