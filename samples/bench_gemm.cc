// SGEMM/DGEMM micro-benchmark. Run identical args under chipBLAS or
// H4I-hipBLAS by linking against the matching libhipblas.so. Reports
// median GFLOPS over `iters` repetitions per size, after a warmup.
//
// Build (chipBLAS):
//   c++ -std=c++17 -O3 -I<chipStar>/include -I../include \
//       bench_gemm.cc <chipBLAS-build>/libhipblas.so \
//       <chipStar>/lib/libCHIP.so -o bench_chipblas
// Build (H4I-hipBLAS):
//   c++ -std=c++17 -O3 -I<chipStar>/include -I<H4I-hipBLAS>/include \
//       bench_gemm.cc <H4I-hipBLAS>/lib/libhipblas.so \
//       <chipStar>/lib/libCHIP.so -o bench_h4i
//
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHK_HIP(e) do { auto _e = (e); if (_e != hipSuccess) { \
    std::fprintf(stderr, "HIP %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
    std::exit(1); }} while(0)
#define CHK_BL(e)  do { auto _s = (e); if (_s != HIPBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "hipBLAS %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
    std::exit(1); }} while(0)

static double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

template <class T>
double run_gemm(hipblasHandle_t h, int M, int N, int K, int iters,
                hipblasStatus_t (*api)(hipblasHandle_t, hipblasOperation_t,
                                       hipblasOperation_t, int, int, int,
                                       const T*, const T*, int, const T*, int,
                                       const T*, T*, int)) {
    size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
    std::vector<T> A(aN), B(bN), C(cN);
    for (size_t i = 0; i < aN; ++i) A[i] = static_cast<T>((i * 7 % 97) * 0.01);
    for (size_t i = 0; i < bN; ++i) B[i] = static_cast<T>((i * 11 % 89) * 0.01);
    for (size_t i = 0; i < cN; ++i) C[i] = 0;

    T *dA, *dB, *dC;
    CHK_HIP(hipMalloc(&dA, aN * sizeof(T)));
    CHK_HIP(hipMalloc(&dB, bN * sizeof(T)));
    CHK_HIP(hipMalloc(&dC, cN * sizeof(T)));
    CHK_HIP(hipMemcpy(dA, A.data(), aN * sizeof(T), hipMemcpyHostToDevice));
    CHK_HIP(hipMemcpy(dB, B.data(), bN * sizeof(T), hipMemcpyHostToDevice));
    CHK_HIP(hipMemcpy(dC, C.data(), cN * sizeof(T), hipMemcpyHostToDevice));

    T alpha = (T)1.0, beta = (T)0.0;

    // Warm up — the first call may JIT kernels.
    CHK_BL(api(h, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &alpha,
               dA, M, dB, K, &beta, dC, M));
    CHK_HIP(hipDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CHK_HIP(hipDeviceSynchronize());
        double t0 = now_s();
        CHK_BL(api(h, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &alpha,
                   dA, M, dB, K, &beta, dC, M));
        CHK_HIP(hipDeviceSynchronize());
        samples.push_back(now_s() - t0);
    }
    std::sort(samples.begin(), samples.end());
    double median = samples[samples.size() / 2];

    CHK_HIP(hipFree(dA)); CHK_HIP(hipFree(dB)); CHK_HIP(hipFree(dC));

    // GFLOPS = 2*M*N*K / time / 1e9.
    return (2.0 * M * N * K) / median / 1e9;
}

int main(int argc, char** argv) {
    int iters = 10;
    bool dp = false;  // double precision?
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--dp") dp = true;
        else if (a == "--sizes" && i + 1 < argc) {
            sizes.clear();
            for (char* tok = std::strtok(argv[++i], ","); tok;
                 tok = std::strtok(nullptr, ",")) {
                sizes.push_back(std::atoi(tok));
            }
        } else {
            std::fprintf(stderr, "Usage: %s [--iters N] [--dp] [--sizes M1,M2,...]\n",
                         argv[0]);
            return 1;
        }
    }

    hipblasHandle_t h;
    CHK_BL(hipblasCreate(&h));

    std::printf("# %s GEMM, iters=%d (median timing)\n",
                dp ? "double" : "single", iters);
    std::printf("# %8s %8s\n", "size", "GFLOPS");
    for (int n : sizes) {
        double gf = dp
            ? run_gemm<double>(h, n, n, n, iters, &hipblasDgemm)
            : run_gemm<float >(h, n, n, n, iters, &hipblasSgemm);
        std::printf("  %8d %8.2f\n", n, gf);
        std::fflush(stdout);
    }

    CHK_BL(hipblasDestroy(h));
    return 0;
}
