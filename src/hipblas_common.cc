// chipBLAS — handle lifecycle and stream/pointer-mode plumbing.
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"

#include <new>

using chipblas::Handle;

extern "C" {

hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
    if (!handle) return HIPBLAS_STATUS_INVALID_VALUE;
    // Force chipStar initialization before we call hipGetBackendNativeHandles.
    // hipGetBackendNativeHandles holds ApiMtx while calling CHIPInitialize();
    // if chipStar initializes back-calling HIP APIs, it re-enters ApiMtx and
    // deadlocks. hipInit with the mutex NOT held lets the init complete first.
    hipInit(0);
    auto* h = new (std::nothrow) Handle();
    if (!h) return HIPBLAS_STATUS_ALLOC_FAILED;
    // Bind to the default (null) HIP stream until the user overrides it.
    auto status = chipblas::bridgeBindStream(*h);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        delete h;
        return status;
    }
    chipblas::applyTuningOverrides(*h);
    *handle = reinterpret_cast<hipblasHandle_t>(h);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    delete reinterpret_cast<Handle*>(handle);
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t stream) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    auto* h = reinterpret_cast<Handle*>(handle);
    hipStream_t previous = h->stream;
    h->stream = stream;
    // Re-bind: a different stream may sit on a different cl_command_queue.
    auto st = chipblas::bridgeBindStream(*h);
    if (st != HIPBLAS_STATUS_SUCCESS) {
        h->stream = previous;
        (void)chipblas::bridgeBindStream(*h);
    }
    return st;
}

hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* stream) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!stream) return HIPBLAS_STATUS_INVALID_VALUE;
    *stream = reinterpret_cast<Handle*>(handle)->stream;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle,
                                      hipblasPointerMode_t mode) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (mode != HIPBLAS_POINTER_MODE_HOST &&
        mode != HIPBLAS_POINTER_MODE_DEVICE) {
        return HIPBLAS_STATUS_INVALID_ENUM;
    }
    reinterpret_cast<Handle*>(handle)->pointerMode = mode;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle,
                                      hipblasPointerMode_t* mode) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!mode)   return HIPBLAS_STATUS_INVALID_VALUE;
    *mode = reinterpret_cast<Handle*>(handle)->pointerMode;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasGetVersion(hipblasHandle_t handle, int* version) {
    (void)handle;
    if (!version) return HIPBLAS_STATUS_INVALID_VALUE;
    *version = 100; // chipBLAS 0.1.0 → encoded as MAJOR*100 + MINOR*10 + PATCH
    return HIPBLAS_STATUS_SUCCESS;
}

} // extern "C"
