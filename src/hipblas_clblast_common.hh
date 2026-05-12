// Shared CLBlast ⇄ hipBLAS mapping helpers (layout, uplo, diag, side).
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_HIPBLAS_CLBLAST_COMMON_HH
#define CHIPBLAS_HIPBLAS_CLBLAST_COMMON_HH

#include "chipblas_internal.hh"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hipblas_clblast {

inline CLBlastTranspose mapTranspose(hipblasOperation_t op) {
    switch (op) {
    case HIPBLAS_OP_N: return CLBlastTransposeNo;
    case HIPBLAS_OP_T: return CLBlastTransposeYes;
    case HIPBLAS_OP_C: return CLBlastTransposeConjugate;
    default: return CLBlastTransposeNo;
    }
}

inline CLBlastTriangle mapTriangle(hipblasFillMode_t u) {
    switch (u) {
    case HIPBLAS_FILL_MODE_UPPER: return CLBlastTriangleUpper;
    case HIPBLAS_FILL_MODE_LOWER: return CLBlastTriangleLower;
    case HIPBLAS_FILL_MODE_FULL:
    default:
        return CLBlastTriangleUpper; // callers must reject FULL
    }
}

inline CLBlastDiagonal mapDiag(hipblasDiagType_t d) {
    return (d == HIPBLAS_DIAG_UNIT) ? CLBlastDiagonalUnit
                                    : CLBlastDiagonalNonUnit;
}

inline CLBlastSide mapSide(hipblasSideMode_t s) {
    switch (s) {
    case HIPBLAS_SIDE_LEFT:  return CLBlastSideLeft;
    case HIPBLAS_SIDE_RIGHT: return CLBlastSideRight;
    case HIPBLAS_SIDE_BOTH:
    default:
        return CLBlastSideLeft; // callers must reject BOTH
    }
}

inline chipblas::Handle* asHandle(hipblasHandle_t h) {
    return reinterpret_cast<chipblas::Handle*>(h);
}

// Stride in elements; length >= 1 for n>0.
inline size_t vecBytesElem(int n, int inc, size_t elem) {
    int a = inc < 0 ? -inc : inc;
    if (n <= 0 || a < 1) return 0;
    return (static_cast<size_t>(n - 1) * static_cast<size_t>(a) + 1) * elem;
}

inline bool okHandle_stream(chipblas::Handle* h) {
    return h && h->isOpenCL;
}

} // namespace hipblas_clblast

#endif
