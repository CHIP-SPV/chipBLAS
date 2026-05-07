# On devices without fp64 (e.g. Mali-G52 with no cl_khr_fp64), set
# CHIPBLAS_HAS_FP64=OFF to exclude double-precision test cases that would
# abort with CLBlast kNotImplemented (-2044).
option(CHIPBLAS_HAS_FP64 "Device supports double-precision; run fp64 tests" ON)

foreach(t IN ITEMS lifecycle l1 l2 l3)
    add_executable(test_${t} test_${t}.cc)
    target_link_libraries(test_${t} PRIVATE hipblas)
    target_include_directories(test_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(CHIPBLAS_HAS_FP64)
        target_compile_definitions(test_${t} PRIVATE CHIPBLAS_HAS_FP64)
    endif()
    add_test(NAME ${t} COMMAND test_${t})
endforeach()
