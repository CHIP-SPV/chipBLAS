cmake_minimum_required(VERSION 3.20)
project(chipBLAS
    VERSION 0.1.0
    DESCRIPTION "Portable hipBLAS on chipStar + CLBlast"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CHIPBLAS_BUILD_TESTS "Build chipBLAS tests" ON)
option(CHIPBLAS_USE_VENDORED_CLBLAST
       "Build CLBlast from third_party/CLBlast (off = use system CLBlast)" ON)

# On macOS, prevent any find_package/find_library from selecting Apple's
# OpenCL.framework. Linking CLBlast against the framework while the
# loader-from-chipStar is active at runtime hands queues across two
# different OpenCL stacks and crashes in dynamic_cast on the first
# clRetain*. Must be set before any find_package(OpenCL).
if(APPLE)
    set(CMAKE_FIND_FRAMEWORK NEVER)
endif()

# chipStar's CHIPTargets.cmake transitively requires Threads::Threads.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- chipStar -----------------------------------------------------------------
# chipStar's install ships CHIPTargets.cmake (no Config wrapper); locate it
# via CHIP_DIR or by searching CMAKE_PREFIX_PATH.
find_path(CHIP_TARGETS_DIR CHIPTargets.cmake
    HINTS
        ${CHIP_DIR}
        ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES cmake/CHIP)
if(NOT CHIP_TARGETS_DIR)
    message(FATAL_ERROR
        "Could not locate chipStar's CHIPTargets.cmake. "
        "Pass -DCHIP_DIR=<chipStar-install>/cmake/CHIP "
        "or -DCMAKE_PREFIX_PATH=<chipStar-install>.")
endif()
include(${CHIP_TARGETS_DIR}/CHIPTargets.cmake)
message(STATUS "chipBLAS: using chipStar from ${CHIP_TARGETS_DIR}")

# --- Level Zero guard ----------------------------------------------------
# chipBLAS uses CLBlast on the OpenCL backend exclusively. Intel GPUs with
# Level Zero should use H4I-HipBLAS (Intel MKL-based), which is the
# vendor-optimized hipBLAS implementation for Level Zero targets. Refuse
# to configure if Level Zero is detected so users don't end up with two
# libhipblas.so installs racing for the same prefix.
find_path(_LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h
    HINTS /usr/local/include /usr/include)
find_library(_LEVEL_ZERO_LIBRARY ze_loader
    HINTS /usr/local/lib /usr/lib)
if(_LEVEL_ZERO_INCLUDE_DIR OR _LEVEL_ZERO_LIBRARY)
    message(FATAL_ERROR
        "Level Zero was found on this system but chipBLAS targets the "
        "OpenCL/CLBlast stack only. For Intel GPUs with Level Zero, use "
        "H4I-HipBLAS (https://github.com/CHIP-SPV/H4I-HipBLAS) instead, "
        "which is backed by Intel MKL and is the recommended hipBLAS "
        "implementation for Level Zero targets.")
endif()

# --- OpenCL -------------------------------------------------------------------
# CLBlast is OpenCL-only, so we always need OpenCL headers + ICD loader.
find_package(OpenCL REQUIRED)

# On macOS, CLBlast's clblast.h does `#include <OpenCL/opencl.h>` (the
# framework-style spelling). Apple's framework header uses Apple-only
# deprecation macros that don't survive a non-Apple OpenCL stack. Force
# that include to the Khronos <CL/opencl.h> via a one-line shim, by
# putting our shim dir first on every target's include path.
if(APPLE)
    include_directories(BEFORE
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/opencl_shim
        ${OpenCL_INCLUDE_DIRS})
endif()

# --- CLBlast ------------------------------------------------------------------
if(CHIPBLAS_USE_VENDORED_CLBLAST)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CLBlast/CMakeLists.txt)
        message(FATAL_ERROR
            "third_party/CLBlast is empty. Run:\n"
            "  git submodule add https://github.com/CNugteren/CLBlast.git "
            "third_party/CLBlast\n"
            "or pass -DCHIPBLAS_USE_VENDORED_CLBLAST=OFF to use the system "
            "package.")
    endif()
    # Suppress CLBlast's own tests/samples — we only want the library.
    set(TESTS    OFF CACHE BOOL "" FORCE)
    set(CLIENTS  OFF CACHE BOOL "" FORCE)
    set(SAMPLES  OFF CACHE BOOL "" FORCE)
    set(TUNERS   OFF CACHE BOOL "" FORCE)
    set(NETLIB   OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/CLBlast EXCLUDE_FROM_ALL)
    set(CLBLAST_TARGET clblast)
else()
    find_package(CLBlast REQUIRED)
    set(CLBLAST_TARGET clblast)
endif()

# --- Library ------------------------------------------------------------------
set(_CHIPBLAS_SRC
    src/hipblas_common.cc
    src/hipblas_ocl.cc
    src/hipblas_tuning.cc
    src/hipblas_l1.cc
    src/hipblas_l2.cc
    src/hipblas_l3.cc
    src/hipblas_extras.cc
)

add_library(hipblas SHARED ${_CHIPBLAS_SRC})
target_include_directories(hipblas
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(hipblas
    PUBLIC __HIP_PLATFORM_SPIRV__)
target_link_libraries(hipblas
    PUBLIC  CHIP::CHIP
    PRIVATE OpenCL::OpenCL ${CLBLAST_TARGET})

set_target_properties(hipblas PROPERTIES
    OUTPUT_NAME hipblas
    SOVERSION   ${PROJECT_VERSION_MAJOR}
    VERSION     ${PROJECT_VERSION})

# --- Install ------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS hipblas
    EXPORT chipBLASTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/hipblas include/chipblas
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# --- Tests --------------------------------------------------------------------
if(CHIPBLAS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
