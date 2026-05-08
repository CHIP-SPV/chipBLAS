# chipBLAS

A portable implementation of the [hipBLAS](https://github.com/ROCm/hipBLAS)
API on top of [chipStar](https://github.com/CHIP-SPV/chipStar) and
[CLBlast](https://github.com/CNugteren/CLBlast).

chipBLAS lets HIP applications using `hipblas.h` run on any device
chipStar's OpenCL backend supports — Intel GPUs, PoCL CPU, AMD via Mesa
Clover, etc. — without rocBLAS or cuBLAS.

## How it works

chipStar exposes the underlying OpenCL handles backing each HIP stream via
`hipGetBackendNativeHandles()`. chipBLAS pulls those handles out, hands
them to CLBlast, and stages user data through `cl_mem` buffers (CLBlast's
required input format). The hipBLAS calls are thin shims over the matching
`CLBlastSgemm` / `CLBlastSgemv` / etc. entry points.

```
  user app  ──hipBLAS API──►  libhipblas.so (chipBLAS)
                                  │
                                  ├── handle: cl_context, cl_command_queue
                                  │           ◄── hipGetBackendNativeHandles()
                                  └── exec:   stage cl_mem ↔ HIP ptr
                                              └── CLBlast{S,D,C,Z}{gemm,...}
```

## Status

Early. The handle and stream plumbing work; the BLAS subset shipping today
is:

- L1: `hipblasSaxpy`, `hipblasDaxpy`, `hipblasSscal`, `hipblasDscal`
- L2: `hipblasSgemv`, `hipblasDgemv`
- L3: `hipblasSgemm`, `hipblasDgemm`, `hipblasCgemm`, `hipblasZgemm`

Backends:
- OpenCL — supported. chipStar's hipMalloc returns device-address pointers
  that are not directly usable as `cl_mem`, so the v0 bridge stages user
  data through host bounce buffers (one `clCreateBuffer` +
  `clEnqueueWriteBuffer` per input, one `clEnqueueReadBuffer` per output).
  Functional but not free — a direct `cl_mem` interop path requires a new
  hook in chipStar (`hipGetClMemFromDevPtr` or similar) and is on the
  roadmap.
- Level Zero — not supported. CLBlast is OpenCL-only; on an L0 chipStar
  build, exec entry points return `HIPBLAS_STATUS_NOT_SUPPORTED`. Build
  chipStar against OpenCL or use `H4I-hipBLAS`.

Pointer mode: HOST only. `hipblasSetPointerMode(handle, DEVICE)` is
recorded but ignored; alpha/beta are dereferenced on the host.

## Requirements

- chipStar, installed and built against OpenCL (provides `CHIPTargets.cmake`)
- OpenCL ICD loader + headers
- CMake 3.20+
- A C++17 compiler

## Build

```bash
git clone --recursive https://github.com/<you>/chipBLAS
cd chipBLAS
# If you forgot --recursive:
git submodule add https://github.com/CNugteren/CLBlast.git third_party/CLBlast
cmake -S . -B build \
    -DCMAKE_PREFIX_PATH=/path/to/chipStar/install
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Reference checks (CTest)

Numerical tests compare GPU results to a **host reference** in `test/blas_reference.hh` (same formulas as the BLAS routines). Run everything with:

```bash
ctest --test-dir build --output-on-failure
```

The `conformance` executable (`test_conformance`) bundles the broader L1/L2/L3 scenario set in one process (exit `0` only if all pass).

To use a system-installed CLBlast instead of the vendored submodule:

```bash
cmake -S . -B build -DCHIPBLAS_USE_VENDORED_CLBLAST=OFF \
    -DCMAKE_PREFIX_PATH="/path/to/chipStar/install;/path/to/clblast"
```

### macOS

CMake's default `find_package(OpenCL)` selects Apple's `OpenCL.framework`,
which is incompatible with the ICD loader chipStar links against — the
two stacks segfault when a queue from one is passed to the other. The
build sets `CMAKE_FIND_FRAMEWORK=NEVER` automatically; you also need to
point CMake at chipStar's bundled CL headers and the loader you want
CLBlast to link against:

```bash
cmake -S . -B build \
    -DCMAKE_PREFIX_PATH=/path/to/chipStar/install \
    -DOPENCL_INCLUDE_DIRS=/path/to/chipStar/install/include \
    -DOPENCL_LIBRARIES=/path/to/ocl-icd-loader/lib/libOpenCL.dylib \
    -DOpenCL_INCLUDE_DIR=/path/to/chipStar/install/include \
    -DOpenCL_LIBRARY=/path/to/ocl-icd-loader/lib/libOpenCL.dylib
```

Both upper- and lower-case variants of `OPENCL_*` are needed because
CLBlast's bundled `FindOpenCL.cmake` uses the older spelling while CMake's
own module uses the newer one.

PoCL / chipStar testing on macOS may need canonical SVM device pointers for
the bridge (see `src/hipblas_ocl.cc`): set
`CHIP_OCL_USE_ALLOC_STRATEGY=svm` when running tests (the `test-opencl-macos`
CI job sets this for the self-hosted runner). CLBlast may return
`kNoHalfPrecision` (-2045) for `hipblasHalf*` kernels on CPU/PoCL stacks; that
CI job also sets `CHIPBLAS_SKIP_HALF_API_SURFACE` so `api_surface` can complete.

```cpp
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

float *dA, *dB, *dC;
hipMalloc(&dA, M*K*sizeof(float));
hipMalloc(&dB, K*N*sizeof(float));
hipMalloc(&dC, M*N*sizeof(float));
// ... copy inputs in ...

hipblasHandle_t h;
hipblasCreate(&h);

float alpha = 1.0f, beta = 0.0f;
hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N,
             M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);

hipblasDestroy(h);
```

## License

MIT. CLBlast is bundled as a submodule under its own Apache-2.0 license.
