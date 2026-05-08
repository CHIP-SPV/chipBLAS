// L1 BLAS correctness vs. CPU reference: Saxpy, Daxpy, Sscal, Dscal.
// Each routine is exercised with both unit stride and a non-unit stride to
// catch offset-handling bugs in the bridge. Pass argv[1] for a single CTest
// shard slug (see test/CMakeLists.txt).
//
// SPDX-License-Identifier: MIT

#include "test_common.hh"

using namespace chipblas_test;

namespace {

// y_ref ← alpha*x + y_ref, both with strides.
template <class T>
void axpyHost(int n, T alpha, const T* x, int incx, T* y, int incy) {
    for (int i = 0; i < n; ++i) y[i * incy] += alpha * x[i * incx];
}

// x ← alpha*x with stride.
template <class T>
void scalHost(int n, T alpha, T* x, int incx) {
    for (int i = 0; i < n; ++i) x[i * incx] *= alpha;
}

template <class T>
size_t storage(int n, int inc) { return (size_t)(n - 1) * (size_t)inc + 1; }

bool runSaxpy(int n, int incx, int incy) {
    float alpha = 1.75f;
    size_t nx = storage<float>(n, incx);
    size_t ny = storage<float>(n, incy);
    std::vector<float> x(nx), y(ny), y_ref;
    for (size_t i = 0; i < nx; ++i) x[i] = fillF((int)i, 1);
    for (size_t i = 0; i < ny; ++i) y[i] = fillF((int)i, 2);
    y_ref = y;
    axpyHost<float>(n, alpha, x.data(), incx, y_ref.data(), incy);

    float *dX, *dY;
    CHECK_HIP(hipMalloc(&dX, nx * sizeof(float)));
    CHECK_HIP(hipMalloc(&dY, ny * sizeof(float)));
    CHECK_HIP(hipMemcpy(dX, x.data(), nx * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dY, y.data(), ny * sizeof(float),
                        hipMemcpyHostToDevice));

    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasSaxpy(h, n, &alpha, dX, incx, dY, incy));
    CHECK_BLAS(hipblasDestroy(h));

    std::vector<float> y_out(ny);
    CHECK_HIP(hipMemcpy(y_out.data(), dY, ny * sizeof(float),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
    return closeReal<float>(y_out, y_ref, 1e-5f);
}

bool runDaxpy(int n, int incx, int incy) {
    double alpha = -2.5;
    size_t nx = storage<double>(n, incx);
    size_t ny = storage<double>(n, incy);
    std::vector<double> x(nx), y(ny), y_ref;
    for (size_t i = 0; i < nx; ++i) x[i] = fillD((int)i, 3);
    for (size_t i = 0; i < ny; ++i) y[i] = fillD((int)i, 4);
    y_ref = y;
    axpyHost<double>(n, alpha, x.data(), incx, y_ref.data(), incy);

    double *dX, *dY;
    CHECK_HIP(hipMalloc(&dX, nx * sizeof(double)));
    CHECK_HIP(hipMalloc(&dY, ny * sizeof(double)));
    CHECK_HIP(hipMemcpy(dX, x.data(), nx * sizeof(double),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dY, y.data(), ny * sizeof(double),
                        hipMemcpyHostToDevice));

    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasDaxpy(h, n, &alpha, dX, incx, dY, incy));
    CHECK_BLAS(hipblasDestroy(h));

    std::vector<double> y_out(ny);
    CHECK_HIP(hipMemcpy(y_out.data(), dY, ny * sizeof(double),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dX)); CHECK_HIP(hipFree(dY));
    return closeReal<double>(y_out, y_ref, 1e-12);
}

bool runSscal(int n, int incx) {
    float alpha = 0.625f;
    size_t nx = storage<float>(n, incx);
    std::vector<float> x(nx), x_ref;
    for (size_t i = 0; i < nx; ++i) x[i] = fillF((int)i, 5);
    x_ref = x;
    scalHost<float>(n, alpha, x_ref.data(), incx);

    float *dX;
    CHECK_HIP(hipMalloc(&dX, nx * sizeof(float)));
    CHECK_HIP(hipMemcpy(dX, x.data(), nx * sizeof(float),
                        hipMemcpyHostToDevice));

    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasSscal(h, n, &alpha, dX, incx));
    CHECK_BLAS(hipblasDestroy(h));

    std::vector<float> x_out(nx);
    CHECK_HIP(hipMemcpy(x_out.data(), dX, nx * sizeof(float),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dX));
    return closeReal<float>(x_out, x_ref, 1e-6f);
}

bool runDscal(int n, int incx) {
    double alpha = -3.125;
    size_t nx = storage<double>(n, incx);
    std::vector<double> x(nx), x_ref;
    for (size_t i = 0; i < nx; ++i) x[i] = fillD((int)i, 6);
    x_ref = x;
    scalHost<double>(n, alpha, x_ref.data(), incx);

    double *dX;
    CHECK_HIP(hipMalloc(&dX, nx * sizeof(double)));
    CHECK_HIP(hipMemcpy(dX, x.data(), nx * sizeof(double),
                        hipMemcpyHostToDevice));

    hipblasHandle_t h;
    CHECK_BLAS(hipblasCreate(&h));
    CHECK_BLAS(hipblasDscal(h, n, &alpha, dX, incx));
    CHECK_BLAS(hipblasDestroy(h));

    std::vector<double> x_out(nx);
    CHECK_HIP(hipMemcpy(x_out.data(), dX, nx * sizeof(double),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dX));
    return closeReal<double>(x_out, x_ref, 1e-13);
}

} // namespace

int main(int argc, char** argv) {
    bool ok = true;
#define RUN(slug, name, fn)                                                        \
    if (should_run_case(argc, argv, slug)) {                                       \
        bool _p = (fn)();                                                           \
        report(name, _p);                                                           \
        ok &= _p;                                                                   \
        if (case_filter_active(argc, argv))                                         \
            return ok ? 0 : 1;                                                      \
    }                                                                               \
    do {                                                                            \
    } while (0)

    RUN("l1:saxpy-inc1", "Saxpy n=1024 inc=1,1",
        ([]() { return runSaxpy(1024, 1, 1); }));
    RUN("l1:saxpy-strided", "Saxpy n=513  inc=2,3",
        ([]() { return runSaxpy(513, 2, 3); }));
#if defined(CHIPBLAS_HAS_FP64)
    RUN("l1:daxpy-inc1", "Daxpy n=1024 inc=1,1",
        ([]() { return runDaxpy(1024, 1, 1); }));
    RUN("l1:daxpy-strided", "Daxpy n=257  inc=4,1",
        ([]() { return runDaxpy(257, 4, 1); }));
#endif
    RUN("l1:sscal-inc1", "Sscal n=2048 inc=1",
        ([]() { return runSscal(2048, 1); }));
    RUN("l1:sscal-strided", "Sscal n=331  inc=5",
        ([]() { return runSscal(331, 5); }));
#if defined(CHIPBLAS_HAS_FP64)
    RUN("l1:dscal-inc1", "Dscal n=2048 inc=1",
        ([]() { return runDscal(2048, 1); }));
    RUN("l1:dscal-strided", "Dscal n=331  inc=3",
        ([]() { return runDscal(331, 3); }));
#endif
#undef RUN

    if (case_filter_active(argc, argv)) {
        std::fprintf(stderr, "unknown l1 case \"%s\"\n", argv[1]);
        return 2;
    }
    return ok ? 0 : 1;
}
