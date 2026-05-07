// chipBLAS-specific extension API. Useful for inspecting how the wrapper
// is communicating with CLBlast (for diagnostics, tests, and tooling).
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_CHIPBLAS_EXT_H
#define CHIPBLAS_CHIPBLAS_EXT_H

#include "hipblas/hipblas.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the chipBLAS version as MAJOR*100 + MINOR*10 + PATCH.
int chipblasVersion(void);

// Backend reported by the live HIP stream the handle is bound to. The
// returned string is a compile-time constant — do not free.
//   "opencl"      — chipStar OpenCL backend (CLBlast can run directly)
//   "level0"      — Level Zero backend (CLBlast cannot drive this; calls
//                   that need GPU work will return HIPBLAS_STATUS_NOT_SUPPORTED)
//   "unknown"     — backend not recognized
const char* chipblasBackend(hipblasHandle_t handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CHIPBLAS_CHIPBLAS_EXT_H
