# On devices without fp64 (e.g. Mali-G52 with no cl_khr_fp64), set
# CHIPBLAS_HAS_FP64=OFF to exclude double-precision test cases that would
# abort with CLBlast kNotImplemented (-2044).
option(CHIPBLAS_HAS_FP64 "Device supports double-precision; run fp64 tests" ON)

foreach(t IN ITEMS lifecycle l1 l2 l3 conformance api_surface)
    add_executable(test_${t} test_${t}.cc)
    target_link_libraries(test_${t} PRIVATE hipblas)
    target_include_directories(test_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(CHIPBLAS_HAS_FP64)
        target_compile_definitions(test_${t} PRIVATE CHIPBLAS_HAS_FP64)
    endif()
endforeach()

# -----------------------------------------------------------------------------
# One CTest per shard: executable argv[1] selects a single case inside the binary.
function(chipblas_shard_tests exe_target case_list)
    foreach(c IN LISTS case_list)
        add_test(NAME ${c} COMMAND ${exe_target} ${c})
    endforeach()
endfunction()

set(chipblas_lifecycle_cases
    lifecycle:create-destroy
    lifecycle:destroy-null
    lifecycle:setstream-getstream
    lifecycle:pointer-mode-roundtrip
    lifecycle:pointer-mode-bad-enum
    lifecycle:version
)
chipblas_shard_tests(test_lifecycle "${chipblas_lifecycle_cases}")

set(chipblas_l1_cases
    l1:saxpy-inc1
    l1:saxpy-strided
    l1:sscal-inc1
    l1:sscal-strided
)
if(CHIPBLAS_HAS_FP64)
    list(APPEND chipblas_l1_cases
        l1:daxpy-inc1
        l1:daxpy-strided
        l1:dscal-inc1
        l1:dscal-strided
    )
endif()
chipblas_shard_tests(test_l1 "${chipblas_l1_cases}")

set(chipblas_l2_cases
    l2:sgemv-N-unit
    l2:sgemv-T-unit
    l2:sgemv-N-strided
    l2:sgemv-T-strided
)
if(CHIPBLAS_HAS_FP64)
    list(APPEND chipblas_l2_cases
        l2:dgemv-N-unit
        l2:dgemv-T-unit
        l2:dgemv-T-strided
    )
endif()
chipblas_shard_tests(test_l2 "${chipblas_l2_cases}")

set(chipblas_l3_cases
    l3:sgemm-NN
    l3:sgemm-NT
    l3:sgemm-TN
    l3:sgemm-TT
    l3:cgemm-NN
    l3:cgemm-CN
    l3:cgemm-NC
    l3:cgemm-CC
)
if(CHIPBLAS_HAS_FP64)
    list(APPEND chipblas_l3_cases
        l3:dgemm-NN
        l3:dgemm-NT
        l3:dgemm-TN
        l3:dgemm-TT
        l3:zgemm-NN
        l3:zgemm-CN
        l3:zgemm-NC
        l3:zgemm-CC
    )
endif()
chipblas_shard_tests(test_l3 "${chipblas_l3_cases}")

set(chipblas_conformance_cases
    conformance:lifecycle-create-destroy
    conformance:lifecycle-destroy-null
    conformance:lifecycle-stream
    conformance:lifecycle-pointer-mode
    conformance:lifecycle-version
    conformance:saxpy-inc1
    conformance:saxpy-strided
    conformance:sscal
    conformance:sgemv-N
    conformance:sgemm-NN
    conformance:sgemm-NT
    conformance:sgemm-TN
    conformance:sgemm-TT
    conformance:cgemm-NN
)
if(CHIPBLAS_HAS_FP64)
    list(APPEND chipblas_conformance_cases
        conformance:daxpy
        conformance:dscal
        conformance:dgemv-T
        conformance:dgemm-NN
        conformance:zgemm-CC
    )
endif()
chipblas_shard_tests(test_conformance "${chipblas_conformance_cases}")

add_test(NAME api_surface COMMAND test_api_surface)
