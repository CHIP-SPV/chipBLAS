// chipBLAS internal handle and bridge declarations.
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_INTERNAL_HH
#define CHIPBLAS_INTERNAL_HH

#include "hipblas/hipblas.h"
#include "chipblas/chipblas_ext.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <clblast_c.h>

#include <cstddef>

namespace chipblas {

// One per hipblasHandle_t. Owns a CLBlast-side cl_command_queue (which we
// pull out of chipStar via hipGetBackendNativeHandles) and the staging
// machinery used to bridge HIP device pointers ↔ cl_mem.
struct Handle {
    hipStream_t          stream      = nullptr;
    hipblasPointerMode_t pointerMode = HIPBLAS_POINTER_MODE_HOST;

    // Native OpenCL handles borrowed from chipStar. We do not retain or
    // release these — chipStar owns their lifetime.
    cl_platform_id    platform = nullptr;
    cl_device_id      device   = nullptr;
    cl_context        context  = nullptr;
    cl_command_queue  queue    = nullptr;
    bool              isOpenCL = false;  // false ⇒ stream is on Level Zero

    // Cached "backend" string for chipblasBackend(). One of:
    // "opencl", "level0", "unknown". String literal — no ownership.
    const char* backendName = "unknown";
};

// --- OpenCL bridge -----------------------------------------------------------
// Pull cl_context / cl_command_queue / cl_device_id out of the HIP stream
// the handle is bound to and stash them on the handle.
hipblasStatus_t bridgeBindStream(Handle& h);

// Wrap a HIP SVM pointer as a cl_mem (USE_HOST_PTR). No host copy.
// Requires chipStar to use an SVM allocation strategy
// (CHIP_OCL_USE_ALLOC_STRATEGY=svm). Fails with NOT_SUPPORTED for
// non-canonical addresses (Intel USM device pointers).
enum class BufDir { IN, OUT, INOUT };  // kept for call-site compatibility

struct StagedBuffer {
    cl_mem  mem      = nullptr;
    size_t  offset   = 0;
    // Internal flags — not used by callers.
    bool    staged   = false;
    bool    svmWrap  = false;
    void*   hipPtr   = nullptr;
    size_t  bytes    = 0;
    BufDir  dir      = BufDir::IN;
};

hipblasStatus_t bridgeStage(Handle& h, void* hipPtr, size_t bytes,
                            BufDir dir, StagedBuffer* out);

// Release the SVM-wrapped cl_mem. No data movement needed.
hipblasStatus_t bridgeWriteBack(Handle& h, StagedBuffer& buf);

// Translate CLBlast's status code to a hipBLAS status.
hipblasStatus_t translate(int clblastStatus);

// Load tuning override JSONs from $CHIPBLAS_TUNING_DIR (one file per
// kernel family, as emitted by CLBlast's tuner binaries) and feed them
// into clblast::OverrideParameters for the bound device. No-op when the
// env var is unset or the directory is missing. Errors are non-fatal —
// they get logged on stderr; the call never fails the handle creation.
void applyTuningOverrides(Handle& h);

} // namespace chipblas

#endif // CHIPBLAS_INTERNAL_HH
