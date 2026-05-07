// Lifecycle smoke: create/destroy, set/get stream, set/get pointer mode,
// version queries, chipblasBackend extension. No CPU reference — these
// are state-machine checks, not numerical.
//
// SPDX-License-Identifier: MIT

#include "test_common.hh"

#include <chipblas/chipblas_ext.h>

#include <cstring>

using namespace chipblas_test;

int main() {
    bool allOk = true;

    // Create / destroy.
    {
        hipblasHandle_t h = nullptr;
        CHECK_BLAS(hipblasCreate(&h));
        if (!h) { report("create-nonnull", false); return 1; }
        CHECK_BLAS(hipblasDestroy(h));
        report("create-destroy", true);
    }

    // null-handle returns HIPBLAS_STATUS_HANDLE_IS_NULLPTR on Destroy.
    {
        bool ok = (hipblasDestroy(nullptr) == HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
        report("destroy-null-rejected", ok);
        allOk &= ok;
    }

    // set/get stream round-trip.
    {
        hipblasHandle_t h;
        CHECK_BLAS(hipblasCreate(&h));
        hipStream_t s = nullptr;
        CHECK_HIP(hipStreamCreate(&s));
        CHECK_BLAS(hipblasSetStream(h, s));
        hipStream_t got = (hipStream_t)0xdeadbeef;
        CHECK_BLAS(hipblasGetStream(h, &got));
        bool ok = (got == s);
        report("setstream-getstream", ok);
        allOk &= ok;
        CHECK_BLAS(hipblasDestroy(h));
        CHECK_HIP(hipStreamDestroy(s));
    }

    // set/get pointer mode.
    {
        hipblasHandle_t h;
        CHECK_BLAS(hipblasCreate(&h));
        hipblasPointerMode_t m = HIPBLAS_POINTER_MODE_DEVICE;
        CHECK_BLAS(hipblasGetPointerMode(h, &m));
        bool ok = (m == HIPBLAS_POINTER_MODE_HOST);
        CHECK_BLAS(hipblasSetPointerMode(h, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_BLAS(hipblasGetPointerMode(h, &m));
        ok &= (m == HIPBLAS_POINTER_MODE_DEVICE);
        report("pointer-mode-roundtrip", ok);
        allOk &= ok;
        CHECK_BLAS(hipblasDestroy(h));
    }

    // SetPointerMode rejects bogus enum.
    {
        hipblasHandle_t h;
        CHECK_BLAS(hipblasCreate(&h));
        bool ok = (hipblasSetPointerMode(h, (hipblasPointerMode_t)42)
                   == HIPBLAS_STATUS_INVALID_ENUM);
        report("pointer-mode-rejects-bad-enum", ok);
        allOk &= ok;
        CHECK_BLAS(hipblasDestroy(h));
    }

    // Version + chipblasBackend.
    {
        hipblasHandle_t h;
        CHECK_BLAS(hipblasCreate(&h));
        int v = 0;
        CHECK_BLAS(hipblasGetVersion(h, &v));
        bool ok = (v > 0) && (chipblasVersion() == v);
        const char* b = chipblasBackend(h);
        ok &= (b != nullptr);
        // We need the OpenCL backend for any of the BLAS tests to work;
        // if we're not on it, the rest of the suite will skip — flag it
        // here as a warning, not a failure.
        if (std::strcmp(b, "opencl") != 0) {
            std::printf("  note: backend is '%s' (BLAS tests will skip)\n", b);
        }
        report("version-and-backend", ok);
        allOk &= ok;
        CHECK_BLAS(hipblasDestroy(h));
    }

    return allOk ? 0 : 1;
}
