// chipBLAS — OpenCL bridge between chipStar HIP streams and CLBlast.
//
// chipStar exposes the underlying cl_context / cl_command_queue backing
// each HIP stream via hipGetBackendNativeHandles(). We fish those out so
// CLBlast can submit its own kernels into the same context.
//
// HIP device pointers on chipStar's SVM allocation path are wrapped as
// cl_mem via clCreateBuffer(CL_MEM_USE_HOST_PTR) — no host copy, CLBlast
// operates directly on the SVM-backed device memory.
//
// Requirement: chipStar must hand us host-addressable pointers
// (CHIP_OCL_USE_ALLOC_STRATEGY=svm). On Intel platforms that expose
// cl_intel_unified_shared_memory we use clGetMemAllocInfoINTEL to reject
// device-only USM pointers — wrapping those would silently alias the
// wrong memory because the OpenCL runtime does not validate the host_ptr.
// Platforms without the extension (e.g. Mali, PoCL) cannot produce USM
// device-only pointers, so the per-call query is skipped after the first
// negative resolution.
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"

#include <hip/hip_runtime.h>
#include <hip/hip_interop.h>

#include <cstdio>
#include <cstring>
#include <mutex>

namespace chipblas {

namespace {

// chipStar's native-handle vector is 5 entries on the OpenCL backend:
//   [0] (uintptr_t) "opencl"  — backend name string literal
//   [1] cl_platform_id
//   [2] cl_device_id
//   [3] cl_context
//   [4] cl_command_queue
constexpr int kHandleCount = 5;

// cl_intel_unified_shared_memory enums (kept local; the extension header
// isn't always installed alongside CL/cl.h).
constexpr cl_uint kClMemAllocTypeIntel = 0x419A;
constexpr cl_uint kClMemTypeUnknown    = 0x4196;
constexpr cl_uint kClMemTypeDevice     = 0x4198;

using GetMemAllocInfoFn = cl_int (CL_API_CALL*)(
    cl_context, const void*, cl_uint, size_t, void*, size_t*);

// chipStar binds one OpenCL platform per process — resolve the USM query
// once. Null `fn` means the platform doesn't expose the extension, so no
// pointer on it can be USM device-only and per-call validation is skipped.
struct UsmProbe {
    GetMemAllocInfoFn fn = nullptr;
};

const UsmProbe& probeUsm(cl_platform_id platform) {
    static UsmProbe       probe;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        probe.fn = reinterpret_cast<GetMemAllocInfoFn>(
            clGetExtensionFunctionAddressForPlatform(
                platform, "clGetMemAllocInfoINTEL"));
    });
    return probe;
}

bool isUsmDeviceOnly(const Handle& h, const void* ptr) {
    const UsmProbe& probe = probeUsm(h.platform);
    if (!probe.fn) return false;

    cl_uint allocType = kClMemTypeUnknown;
    if (probe.fn(h.context, ptr, kClMemAllocTypeIntel,
                 sizeof(allocType), &allocType, nullptr) != CL_SUCCESS) {
        return false;
    }
    return allocType == kClMemTypeDevice;
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

    auto* backendStr = reinterpret_cast<const char*>(handles[0]);
    if (!backendStr || std::strcmp(backendStr, "opencl") != 0) {
        // chipBLAS is OpenCL-only; never leave stale CL_* from an earlier binding
        // (e.g. hipblasSetStream from OpenCL queue to Level Zero queue).
        h.platform = nullptr;
        h.device   = nullptr;
        h.context  = nullptr;
        h.queue    = nullptr;
        h.isOpenCL = false;
        if (backendStr && std::strcmp(backendStr, "level0") == 0)
            h.backendName = "level0";
        else
            h.backendName = "unknown";
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    h.backendName = "opencl";

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

hipblasStatus_t bridgeStage(Handle& h, void* hipPtr, size_t bytes,
                            BufDir dir, StagedBuffer* out) {
    if (!h.isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!out) return HIPBLAS_STATUS_INVALID_VALUE;
    if (bytes == 0) return HIPBLAS_STATUS_INVALID_VALUE;

    out->hipPtr = hipPtr;
    out->bytes  = bytes;
    out->dir    = dir;

    if (isUsmDeviceOnly(h, hipPtr)) {
        std::fprintf(stderr,
            "chipBLAS: cannot wrap pointer %p — it is a USM device-only "
            "allocation. Use CHIP_OCL_USE_ALLOC_STRATEGY=svm so chipStar "
            "returns host-addressable SVM pointers.\n", hipPtr);
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
