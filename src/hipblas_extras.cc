
// chipBLAS — extended BLAS (CLBlast ⇒ hipBLAS) for routines outside the
// minimal l1/l2/l3 cores.
//
// SPDX-License-Identifier: MIT

#include "chipblas_internal.hh"
#include "hipblas_clblast_common.hh"
#include "hipblas_matmul_bridge.hh"

#include <CL/cl.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <utility>

using chipblas::BufDir;
using chipblas::Handle;
using chipblas::StagedBuffer;
using hipblas_clblast::mapDiag;
using hipblas_clblast::mapSide;
using hipblas_clblast::mapTriangle;
using hipblas_clblast::mapTranspose;
using hipblas_mm::buf2Run;
using hipblas_mm::gemmRun;
using hipblas_mm::symmABytes;
using hipblas_mm::symmBcBytes;
using hipblas_mm::syrkABytes;
using hipblas_mm::syrkCBytes;
using hipblas_mm::trmmABytes;
using hipblas_mm::trmmBBytes;

namespace {

size_t vecBytes(int n, int inc, size_t elemBytes) {
    return hipblas_clblast::vecBytesElem(n, inc, elemBytes);
}

inline bool rejectGeom(int n) { return n <= 0; }

inline hipblasStatus_t rejectFullBoth(hipblasFillMode_t uplo,
                                      hipblasSideMode_t side) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (side == HIPBLAS_SIDE_BOTH) return HIPBLAS_STATUS_INVALID_VALUE;
    return HIPBLAS_STATUS_SUCCESS;
}

template <class Dispatch>
hipblasStatus_t vec2Run(hipblasHandle_t handle, int n, int incx, int incy,
                        size_t elemBytes, void* x, void* y, Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0 || incy <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!x || !y) return HIPBLAS_STATUS_INVALID_VALUE;
    if (rejectGeom(n)) return HIPBLAS_STATUS_SUCCESS;

    StagedBuffer sx, sy;
    auto rc = chipblas::bridgeStage(*h, x, vecBytes(n, incx, elemBytes),
                                    BufDir::INOUT, &sx);
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
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

template <class Dispatch>
hipblasStatus_t copyRun(hipblasHandle_t handle, int n, int incx, int incy,
                        size_t elemBytes, const void* x, void* y,
                        Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0 || incy <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!x || !y) return HIPBLAS_STATUS_INVALID_VALUE;
    if (rejectGeom(n)) return HIPBLAS_STATUS_SUCCESS;

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
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

template <class T, class Dispatch>
hipblasStatus_t scalLike(hipblasHandle_t handle, int n, int incx,
                         void* x, Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!x) return HIPBLAS_STATUS_INVALID_VALUE;
    if (rejectGeom(n)) return HIPBLAS_STATUS_SUCCESS;
    StagedBuffer sx;
    auto rc = chipblas::bridgeStage(*h, x, vecBytes(n, incx, sizeof(T)),
                                    BufDir::INOUT, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    cl_command_queue queue = h->queue;
    int clb = dispatch(sx, &queue);
    auto wb = chipblas::bridgeWriteBack(*h, sx);
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

template <class Dispatch>
hipblasStatus_t axpyLike(hipblasHandle_t handle, int n, int incx, int incy,
                         size_t elemBytes, const void* x, void* y,
                         Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0 || incy <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!x || !y) return HIPBLAS_STATUS_INVALID_VALUE;
    if (rejectGeom(n)) return HIPBLAS_STATUS_SUCCESS;
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
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

template <class ResElem, class VecElem, class Dispatch>
hipblasStatus_t unaryReduceRun(hipblasHandle_t handle, int n, int incx,
                               void* result, const void* x,
                               Dispatch&& dispatch) {
    constexpr size_t R = sizeof(ResElem);
    constexpr size_t V = sizeof(VecElem);
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!result || !x) return HIPBLAS_STATUS_INVALID_VALUE;
    if (incx <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (rejectGeom(n)) {
        std::memset(result, 0, R);
        return HIPBLAS_STATUS_SUCCESS;
    }
    StagedBuffer sr, sx;
    auto rc = chipblas::bridgeStage(*h, result, R, BufDir::INOUT, &sr);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, const_cast<void*>(x),
                              vecBytes(n, incx, V), BufDir::IN, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sr);
        return rc;
    }
    cl_command_queue queue = h->queue;
    int clb = dispatch(sr, sx, &queue);
    chipblas::bridgeWriteBack(*h, sx);
    auto wb = chipblas::bridgeWriteBack(*h, sr);
    auto tr = chipblas::translate(clb);
    (void)wb;
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : HIPBLAS_STATUS_SUCCESS;
}

template <class Dispatch>
hipblasStatus_t dot2Run(hipblasHandle_t handle, int n, int incx, int incy,
                        size_t resElem, size_t vecElem, void* result,
                        const void* x, const void* y, Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!result || !x || !y) return HIPBLAS_STATUS_INVALID_VALUE;
    if (incx <= 0 || incy <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (rejectGeom(n)) {
        std::memset(result, 0, resElem);
        return HIPBLAS_STATUS_SUCCESS;
    }
    StagedBuffer sr, sx, sy;
    auto rc = chipblas::bridgeStage(*h, result, resElem, BufDir::INOUT, &sr);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, const_cast<void*>(x),
                              vecBytes(n, incx, vecElem), BufDir::IN, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sr);
        return rc;
    }
    rc = chipblas::bridgeStage(*h, const_cast<void*>(y),
                              vecBytes(n, incy, vecElem), BufDir::IN, &sy);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sr);
        chipblas::bridgeWriteBack(*h, sx);
        return rc;
    }
    cl_command_queue queue = h->queue;
    int clb = dispatch(sr, sx, sy, &queue);
    chipblas::bridgeWriteBack(*h, sx);
    chipblas::bridgeWriteBack(*h, sy);
    auto wb = chipblas::bridgeWriteBack(*h, sr);
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

template <class Dispatch>
hipblasStatus_t iamaxRun(hipblasHandle_t handle, int n, int incx,
                         const void* x, size_t vecElem, int* result,
                         Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!result || !x) return HIPBLAS_STATUS_INVALID_VALUE;
    if (incx <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (rejectGeom(n)) {
        *result = 0;
        return HIPBLAS_STATUS_SUCCESS;
    }
    StagedBuffer sr, sx;
    auto rc =
        chipblas::bridgeStage(*h, result, sizeof(int), BufDir::INOUT, &sr);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, const_cast<void*>(x),
                               vecBytes(n, incx, vecElem), BufDir::IN, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sr);
        return rc;
    }
    cl_command_queue queue = h->queue;
    int clb = dispatch(sr, sx, &queue);
    chipblas::bridgeWriteBack(*h, sx);
    auto tr = chipblas::translate(clb);
    if (tr != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sr);
        return tr;
    }
    chipblas::bridgeWriteBack(*h, sr);
    uint32_t raw = 0;
    std::memcpy(&raw, result, sizeof(raw));
    *result = static_cast<int>(raw) + 1;
    return HIPBLAS_STATUS_SUCCESS;
}

template <class Dispatch>
hipblasStatus_t ger3Run(hipblasHandle_t handle, int m, int n, int incx,
                        int incy, size_t aBytes, size_t vxBytes, size_t vyBytes,
                        const void* x, const void* y, void* A, Dispatch&& d) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0 || incy <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (rejectGeom(m) || rejectGeom(n)) return HIPBLAS_STATUS_SUCCESS;
    if (!A || !x || !y) return HIPBLAS_STATUS_INVALID_VALUE;
    StagedBuffer sx, sy, sa;
    auto rc = chipblas::bridgeStage(*h, const_cast<void*>(x), vxBytes,
                                    BufDir::IN, &sx);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, const_cast<void*>(y), vyBytes, BufDir::IN,
                               &sy);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sx);
        return rc;
    }
    rc = chipblas::bridgeStage(*h, A, aBytes, BufDir::INOUT, &sa);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sx);
        chipblas::bridgeWriteBack(*h, sy);
        return rc;
    }
    cl_command_queue queue = h->queue;
    int clb = d(sa, sx, sy, &queue);
    chipblas::bridgeWriteBack(*h, sx);
    chipblas::bridgeWriteBack(*h, sy);
    auto wb = chipblas::bridgeWriteBack(*h, sa);
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

template <class Dispatch>
hipblasStatus_t trmvLike(hipblasHandle_t handle, hipblasFillMode_t uplo,
                         int n, int lda, int incx, const void* A, void* x,
                         size_t elemBytes, Dispatch&& dispatch) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (incx <= 0) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    if (!A || !x) return HIPBLAS_STATUS_INVALID_VALUE;
    if (rejectGeom(n)) return HIPBLAS_STATUS_SUCCESS;
    size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n)
                  * elemBytes;
    return buf2Run(handle, aBytes, vecBytes(n, incx, elemBytes), A, x,
                   std::forward<Dispatch>(dispatch));
}

} // namespace

extern "C" {


hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n,
                                  float* x, int incx, float* y, int incy) {
    return vec2Run(handle, n, incx, incy, sizeof(float), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(float);
            return CLBlastSswap((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasDswap(hipblasHandle_t handle, int n,
                                  double* x, int incx, double* y, int incy) {
    return vec2Run(handle, n, incx, incy, sizeof(double), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(double);
            return CLBlastDswap((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasCswap(hipblasHandle_t handle, int n,
                                  hipblasComplex* x, int incx, hipblasComplex* y, int incy) {
    return vec2Run(handle, n, incx, incy, sizeof(hipblasComplex), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(hipblasComplex);
            return CLBlastCswap((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasZswap(hipblasHandle_t handle, int n,
                                  hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy) {
    return vec2Run(handle, n, incx, incy, sizeof(hipblasDoubleComplex), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(hipblasDoubleComplex);
            return CLBlastZswap((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasHswap(hipblasHandle_t handle, int n,
                                  hipblasHalf* x, int incx, hipblasHalf* y, int incy) {
    return vec2Run(handle, n, incx, incy, sizeof(hipblasHalf), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(hipblasHalf);
            return CLBlastHswap((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n,
                                  const float* x, int incx, float* y, int incy) {
    return copyRun(handle, n, incx, incy, sizeof(float), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(float);
            return CLBlastScopy((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n,
                                  const double* x, int incx, double* y, int incy) {
    return copyRun(handle, n, incx, incy, sizeof(double), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(double);
            return CLBlastDcopy((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasCcopy(hipblasHandle_t handle, int n,
                                  const hipblasComplex* x, int incx, hipblasComplex* y, int incy) {
    return copyRun(handle, n, incx, incy, sizeof(hipblasComplex), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(hipblasComplex);
            return CLBlastCcopy((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasZcopy(hipblasHandle_t handle, int n,
                                  const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy) {
    return copyRun(handle, n, incx, incy, sizeof(hipblasDoubleComplex), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(hipblasDoubleComplex);
            return CLBlastZcopy((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasHcopy(hipblasHandle_t handle, int n,
                                  const hipblasHalf* x, int incx, hipblasHalf* y, int incy) {
    return copyRun(handle, n, incx, incy, sizeof(hipblasHalf), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t Esz = sizeof(hipblasHalf);
            return CLBlastHcopy((size_t)n,
                X_.mem, X_.offset / Esz, (size_t)incx,
                Y_.mem, Y_.offset / Esz, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle, int n,
                             const hipblasComplex* alpha,
                             const hipblasComplex* x, int incx,
                             hipblasComplex* y, int incy) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    return axpyLike(handle, n, incx, incy, sizeof(hipblasComplex), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCaxpy((size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* alpha,
                             const hipblasDoubleComplex* x, int incx,
                             hipblasDoubleComplex* y, int incy) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    return axpyLike(handle, n, incx, incy, sizeof(hipblasDoubleComplex), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZaxpy((size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasHaxpy(hipblasHandle_t handle, int n,
                             const hipblasHalf* alpha,
                             const hipblasHalf* x, int incx,
                             hipblasHalf* y, int incy) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_half ah {};
    std::memcpy(&ah, alpha, sizeof(ah));
    return axpyLike(handle, n, incx, incy, sizeof(hipblasHalf), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            return CLBlastHaxpy((size_t)n, ah,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasCscal(hipblasHandle_t handle, int n,
                             const hipblasComplex* alpha,
                             hipblasComplex* x, int incx) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    return scalLike<hipblasComplex>(handle, n, incx, x,
        [&](StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCscal((size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasZscal(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* alpha,
                             hipblasDoubleComplex* x, int incx) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    return scalLike<hipblasDoubleComplex>(handle, n, incx, x,
        [&](StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZscal((size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasHscal(hipblasHandle_t handle, int n,
                             const hipblasHalf* alpha,
                             hipblasHalf* x, int incx) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_half ah {};
    std::memcpy(&ah, alpha, sizeof(ah));
    return scalLike<hipblasHalf>(handle, n, incx, x,
        [&](StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            return CLBlastHscal((size_t)n, ah,
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasCsscal(hipblasHandle_t handle, int n,
                              const float* alpha,
                              hipblasComplex* x, int incx) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    hipblasComplex ca = { *alpha, 0.0f };
    return hipblasCscal(handle, n, &ca, x, incx);
}

hipblasStatus_t hipblasZdscal(hipblasHandle_t handle, int n,
                              const double* alpha,
                              hipblasDoubleComplex* x, int incx) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    hipblasDoubleComplex ca = { *alpha, 0.0 };
    return hipblasZscal(handle, n, &ca, x, incx);
}

#include "hipblas_extras_impl.inc"

} // extern "C"
