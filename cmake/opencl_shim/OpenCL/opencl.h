// Minimal shim used on macOS to redirect CLBlast's
// `#include <OpenCL/opencl.h>` (the Apple-framework-style spelling it
// emits when __APPLE__ is defined) to the Khronos-style `<CL/opencl.h>`.
//
// Without this, the build picks up either Apple's framework header
// (which uses Apple-only deprecation macros that the rest of the
// toolchain doesn't define) or a pocl include stub that pulls in
// pocl-specific extension headers that may not be on the include path.
//
// This shim is added to the include search path before any system
// headers, only on Apple, by the top-level CMakeLists.
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_OPENCL_OPENCL_H_SHIM
#define CHIPBLAS_OPENCL_OPENCL_H_SHIM

#include <CL/opencl.h>

#endif // CHIPBLAS_OPENCL_OPENCL_H_SHIM
