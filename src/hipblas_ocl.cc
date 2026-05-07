// chipBLAS — OpenCL bridge between chipStar HIP streams and CLBlast.
//
// chipStar exposes the underlying cl_context / cl_command_queue backing
// each HIP stream via hipGetBackendNativeHandles(). We fish those out so
// CLBlast can submit its own kernels into the same context.
//
// HIP device pointers on chipStar's SVM allocation path are valid virtual
// addresses in canonical user-space. We wrap them as cl_mem via
// clCreateBuffer(CL_MEM_USE_HOST_PTR) — no host copy, CLBlast operates
// directly on the SVM-backed device memory.
//
// Requirement: chipStar must use an SVM allocation strategy
// (CHIP_OCL_USE_ALLOC_STRATEGY=svm or coarsegrain). Intel USM device
// pointers appear at non-canonical addresses and are rejected.
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"

#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <cstdio>
#include <cstring>

namespace chipblas {

namespace {

// chipStar's native-handle vector is 5 entries on the OpenCL backend:
//   [0] (uintptr_t) "opencl"  — backend name string literal
//   [1] cl_platform_id
//   [2] cl_device_id
//   [3] cl_context
//   [4] cl_command_queue
constexpr int kHandleCount = 5;

const char* readBackendTag(uintptr_t tag) {
    auto* s = reinterpret_cast<const char*>(tag);
    if (!s) return "unknown";
    if (std::strcmp(s, "opencl") == 0) return "opencl";
    if (std::strcmp(s, "level0") == 0) return "level0";
    return "unknown";
}

} // namespace

hipblasStatus_t bridgeBindStream(Handle& h) {
    int numHandles = 0;
    int rc = hipGetBackendNativeHandles(
        reinterpret_cast<uintptr_t>(h.stream), nullptr, &numHandles);
    if (rc != 0 || numHandles < 1 || numHandles > kHandleCount) {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    uintptr_t handles[kHandleCount] = {};
    rc = hipGetBackendNativeHandles(
        reinterpret_cast<uintptr_t>(h.stream), handles, nullptr);
    if (rc != 0) {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    h.backendName = readBackendTag(handles[0]);
    if (std::strcmp(h.backendName, "opencl") != 0) {
        h.isOpenCL = false;
        return HIPBLAS_STATUS_SUCCESS;
    }

    if (numHandles < kHandleCount) {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    h.platform = reinterpret_cast<cl_platform_id>(handles[1]);
    h.device   = reinterpret_cast<cl_device_id>(handles[2]);
    h.context  = reinterpret_cast<cl_context>(handles[3]);
    h.queue    = reinterpret_cast<cl_command_queue>(handles[4]);
    h.isOpenCL = true;
    return HIPBLAS_STATUS_SUCCESS;
}

namespace {

// Canonical user-space on Linux x86-64: [0, 0x00007fffffffffff].
// Intel USM device-only pointers appear above this range; wrapping them
// with USE_HOST_PTR silently aliases wrong memory.
constexpr uintptr_t kCanonicalMax = 0x00007fffffffffffULL;

} // namespace

hipblasStatus_t bridgeStage(Handle& h, void* hipPtr, size_t bytes,
                            BufDir dir, StagedBuffer* out) {
    if (!h.isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!out) return HIPBLAS_STATUS_INVALID_VALUE;
    if (bytes == 0) return HIPBLAS_STATUS_INVALID_VALUE;

    out->hipPtr = hipPtr;
    out->bytes  = bytes;
    out->dir    = dir;

    if (reinterpret_cast<uintptr_t>(hipPtr) > kCanonicalMax) {
        std::fprintf(stderr,
            "chipBLAS: SVM wrap failed — pointer %p is not in canonical "
            "user-space (USM device pointer?). Use "
            "CHIP_OCL_USE_ALLOC_STRATEGY=svm.\n", hipPtr);
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    cl_int clerr = CL_SUCCESS;
    out->mem = clCreateBuffer(h.context,
                              CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                              bytes, hipPtr, &clerr);
    if (clerr != CL_SUCCESS || !out->mem) {
        out->mem = nullptr;
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }
    out->staged  = true;
    out->svmWrap = true;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t bridgeWriteBack(Handle& h, StagedBuffer& buf) {
    // SVM wrap: CLBlast wrote directly into the SVM-backed memory.
    // Just release the cl_mem view — no copy needed.
    if (buf.mem) {
        clReleaseMemObject(buf.mem);
        buf.mem = nullptr;
    }
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t translate(int clblastStatus) {
    if (clblastStatus == 0 /* CLBlastSuccess */) return HIPBLAS_STATUS_SUCCESS;
    if (clblastStatus < 0) return HIPBLAS_STATUS_EXECUTION_FAILED;
    return HIPBLAS_STATUS_INTERNAL_ERROR;
}

} // namespace chipblas
