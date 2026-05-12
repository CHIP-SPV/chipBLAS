// Lifecycle smoke: create/destroy, stream, pointer mode, version.
// Pass argv[1] to run a single CTest shard (see test/CMakeLists.txt).
//
// SPDX-License-Identifier: MIT

#include "test_common.hh"

using namespace chipblas_test;

namespace {

bool tc_create_destroy() {
    hipblasHandle_t h = nullptr;
    CHECK_BLAS(hipblasCreate(&h));
    if (!h)
        return false;
    CHECK_BLAS(hipblasDestroy(h));
    return true;
}

bool tc_destroy_null() {
    return hipblasDestroy(nullptr) == HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
}

bool tc_stream() {
    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    hipStream_t s = nullptr;
    CHECK_HIP(hipStreamCreate(&s));
    CHECK_BLAS(hipblasSetStream(h, s));
    hipStream_t got = (hipStream_t)0xdeadbeef;
    CHECK_BLAS(hipblasGetStream(h, &got));
    bool ok = (got == s);
    CHECK_BLAS(hipblasDestroy(h));
    CHECK_HIP(hipStreamDestroy(s));
    return ok;
}

bool tc_pointer_roundtrip() {
    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    hipblasPointerMode_t m = HIPBLAS_POINTER_MODE_DEVICE;
    CHECK_BLAS(hipblasGetPointerMode(h, &m));
    bool ok = (m == HIPBLAS_POINTER_MODE_HOST);
    CHECK_BLAS(hipblasSetPointerMode(h, HIPBLAS_POINTER_MODE_DEVICE));
    CHECK_BLAS(hipblasGetPointerMode(h, &m));
    ok &= (m == HIPBLAS_POINTER_MODE_DEVICE);
    CHECK_BLAS(hipblasDestroy(h));
    return ok;
}

bool tc_pointer_bad_enum() {
    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    bool ok = (hipblasSetPointerMode(h, (hipblasPointerMode_t)42)
                   == HIPBLAS_STATUS_INVALID_ENUM);
    CHECK_BLAS(hipblasDestroy(h));
    return ok;
}

bool tc_version() {
    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    int v = 0;
    CHECK_BLAS(hipblasGetVersion(h, &v));
    bool ok = (v > 0);
    CHECK_BLAS(hipblasDestroy(h));
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    bool ok = true;
#define RUN(slug, disp, fn)                                                        \
    if (should_run_case(argc, argv, slug)) {                                       \
        bool _p = (fn)();                                                          \
        report(disp, _p);                                                          \
        ok &= _p;                                                                  \
        if (case_filter_active(argc, argv))                                        \
            return ok ? 0 : 1;                                                     \
    }                                                                              \
    do {                                                                           \
    } while (0)

    RUN("lifecycle:create-destroy", "create / destroy", tc_create_destroy);
    RUN("lifecycle:destroy-null", "destroy null rejected", tc_destroy_null);
    RUN("lifecycle:setstream-getstream", "set/get stream", tc_stream);
    RUN("lifecycle:pointer-mode-roundtrip", "pointer-mode round-trip",
        tc_pointer_roundtrip);
    RUN("lifecycle:pointer-mode-bad-enum", "pointer-mode rejects bad enum",
        tc_pointer_bad_enum);
    RUN("lifecycle:version", "version", tc_version);
#undef RUN

    if (case_filter_active(argc, argv)) {
        std::fprintf(stderr, "unknown lifecycle case \"%s\"\n", argv[1]);
        return 2;
    }
    return ok ? 0 : 1;
}
