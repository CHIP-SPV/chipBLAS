// Shared three-matrix staging for CLBlast GEMM-style kernels (also used by
// SYMM, SYRK, SYR2K). Extracted so hipblas_l3.cc and hipblas_extras.cc stay in
// sync.
//
// SPDX-License-Identifier: MIT

#ifndef CHIPBLAS_HIPBLAS_MATMUL_BRIDGE_HH
#define CHIPBLAS_HIPBLAS_MATMUL_BRIDGE_HH

#include "chipblas_internal.hh"
#include "hipblas_clblast_common.hh"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace hipblas_mm {

using chipblas::BufDir;
using chipblas::Handle;
using chipblas::StagedBuffer;

// Column-major: op(A) is m×k logically; physically A occupies lda rows by
// (k if op_a==N else m) columns.
inline size_t gemmAByteCount(hipblasOperation_t op, int m, int k, int lda,
                             size_t elemBytes) {
    int cols = (op == HIPBLAS_OP_N) ? k : m;
    return static_cast<size_t>(lda) * static_cast<size_t>(cols) * elemBytes;
}
inline size_t gemmBByteCount(hipblasOperation_t op, int k, int n, int ldb,
                             size_t elemBytes) {
    int cols = (op == HIPBLAS_OP_N) ? n : k;
    return static_cast<size_t>(ldb) * static_cast<size_t>(cols) * elemBytes;
}
inline size_t gemmCByteCount(int /*m*/, int n, int ldc, size_t elemBytes) {
    return static_cast<size_t>(ldc) * static_cast<size_t>(n) * elemBytes;
}

template <class Dispatch>
hipblasStatus_t gemmRun(hipblasHandle_t handle,
                        hipblasOperation_t /*transA*/,
                        hipblasOperation_t /*transB*/,
                        size_t aBytes, size_t bBytes, size_t cBytes,
                        const void* A, const void* B, void* C,
                        Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!A || !B || !C) return HIPBLAS_STATUS_INVALID_VALUE;

    StagedBuffer sa, sb, sc;
    auto rc = chipblas::bridgeStage(*h, const_cast<void*>(A), aBytes,
                                    BufDir::IN, &sa);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, const_cast<void*>(B), bBytes,
                               BufDir::IN, &sb);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        return rc;
    }
    rc = chipblas::bridgeStage(*h, C, cBytes, BufDir::INOUT, &sc);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        chipblas::bridgeWriteBack(*h, sb);
        return rc;
    }

    cl_command_queue queue = h->queue;
    int clb = dispatch(sa, sb, sc, &queue);

    chipblas::bridgeWriteBack(*h, sa);
    chipblas::bridgeWriteBack(*h, sb);
    auto wb = chipblas::bridgeWriteBack(*h, sc);
    auto translated = chipblas::translate(clb);
    return (translated != HIPBLAS_STATUS_SUCCESS) ? translated : wb;
}

// Two-buffer: first typically IN (const), second INOUT (e.g. SYRK A,C; TRMM
// A,B; TRMV A,x).
template <class Dispatch>
hipblasStatus_t buf2Run(hipblasHandle_t handle, size_t aBytes, size_t bBytes,
                        const void* A, void* B, Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!A || !B) return HIPBLAS_STATUS_INVALID_VALUE;

    StagedBuffer sa, sb;
    auto rc = chipblas::bridgeStage(*h, const_cast<void*>(A), aBytes,
                                    BufDir::IN, &sa);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, B, bBytes, BufDir::INOUT, &sb);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        return rc;
    }

    cl_command_queue queue = h->queue;
    int clb = dispatch(sa, sb, &queue);

    chipblas::bridgeWriteBack(*h, sa);
    auto wb = chipblas::bridgeWriteBack(*h, sb);
    auto translated = chipblas::translate(clb);
    return (translated != HIPBLAS_STATUS_SUCCESS) ? translated : wb;
}

inline size_t symmDim(hipblasSideMode_t side, int m, int n) {
    return static_cast<size_t>((side == HIPBLAS_SIDE_LEFT) ? m : n);
}

inline size_t symmABytes(hipblasSideMode_t side, int m, int n, int lda,
                         size_t elemBytes) {
    size_t d = symmDim(side, m, n);
    return static_cast<size_t>(lda) * d * elemBytes;
}

inline size_t symmBcBytes(int m, int n, int ld, size_t elemBytes) {
    return static_cast<size_t>(ld) * static_cast<size_t>(n) * elemBytes;
}

inline size_t syrkABytes(hipblasOperation_t trans, int n, int k, int lda,
                         size_t elemBytes) {
    int cols = (trans == HIPBLAS_OP_N) ? k : n;
    return static_cast<size_t>(lda) * static_cast<size_t>(cols) * elemBytes;
}

inline size_t syrkCBytes(int n, int ldc, size_t elemBytes) {
    return static_cast<size_t>(ldc) * static_cast<size_t>(n) * elemBytes;
}

inline size_t trmmTriDim(hipblasSideMode_t side, int m, int n) {
    return static_cast<size_t>((side == HIPBLAS_SIDE_LEFT) ? m : n);
}

inline size_t trmmABytes(hipblasSideMode_t side, int m, int n, int lda,
                         size_t elemBytes) {
    size_t d = trmmTriDim(side, m, n);
    return static_cast<size_t>(lda) * d * elemBytes;
}

inline size_t trmmBBytes(int m, int n, int ldb, size_t elemBytes) {
    return static_cast<size_t>(ldb) * static_cast<size_t>(n) * elemBytes;
}

} // namespace hipblas_mm

#endif
