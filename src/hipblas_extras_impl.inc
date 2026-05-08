
// Generated: CLBlast-backed Level 1–3 extras (included from hipblas_extras.cc).

hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n,
                            const float* x, int incx, const float* y, int incy,
                            float* result) {
    return dot2Run(handle, n, incx, incy, sizeof(float), sizeof(float),
                   result, x, y,
        [&](StagedBuffer& R_, StagedBuffer& X_, StagedBuffer& Y_,
            cl_command_queue* q) {
            return CLBlastSdot((size_t)n, R_.mem, R_.offset / sizeof(float),
                X_.mem, X_.offset / sizeof(float), (size_t)incx,
                Y_.mem, Y_.offset / sizeof(float), (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n,
                            const double* x, int incx, const double* y, int incy,
                            double* result) {
    return dot2Run(handle, n, incx, incy, sizeof(double), sizeof(double),
                   result, x, y,
        [&](StagedBuffer& R_, StagedBuffer& X_, StagedBuffer& Y_,
            cl_command_queue* q) {
            return CLBlastDdot((size_t)n, R_.mem, R_.offset / sizeof(double),
                X_.mem, X_.offset / sizeof(double), (size_t)incx,
                Y_.mem, Y_.offset / sizeof(double), (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasCdotu(hipblasHandle_t handle, int n,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* y, int incy,
                             hipblasComplex* result) {
    return dot2Run(handle, n, incx, incy, sizeof(hipblasComplex),
                   sizeof(hipblasComplex), result, x, y,
        [&](StagedBuffer& R_, StagedBuffer& X_, StagedBuffer& Y_,
            cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCdotu((size_t)n, R_.mem, R_.offset / E,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasCdotc(hipblasHandle_t handle, int n,
                             const hipblasComplex* x, int incx,
                             const hipblasComplex* y, int incy,
                             hipblasComplex* result) {
    return dot2Run(handle, n, incx, incy, sizeof(hipblasComplex),
                   sizeof(hipblasComplex), result, x, y,
        [&](StagedBuffer& R_, StagedBuffer& X_, StagedBuffer& Y_,
            cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCdotc((size_t)n, R_.mem, R_.offset / E,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasZdotu(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* y, int incy,
                             hipblasDoubleComplex* result) {
    return dot2Run(handle, n, incx, incy, sizeof(hipblasDoubleComplex),
                   sizeof(hipblasDoubleComplex), result, x, y,
        [&](StagedBuffer& R_, StagedBuffer& X_, StagedBuffer& Y_,
            cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZdotu((size_t)n, R_.mem, R_.offset / E,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n,
                             const hipblasDoubleComplex* x, int incx,
                             const hipblasDoubleComplex* y, int incy,
                             hipblasDoubleComplex* result) {
    return dot2Run(handle, n, incx, incy, sizeof(hipblasDoubleComplex),
                   sizeof(hipblasDoubleComplex), result, x, y,
        [&](StagedBuffer& R_, StagedBuffer& X_, StagedBuffer& Y_,
            cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZdotc((size_t)n, R_.mem, R_.offset / E,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                q, nullptr);
        });
}

hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n,
                            const float* x, int incx, float* result) {
    return unaryReduceRun<float, float>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastSnrm2((size_t)n, R_.mem, R_.offset / sizeof(float),
                X_.mem, X_.offset / sizeof(float), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasDnrm2(hipblasHandle_t handle, int n,
                            const double* x, int incx, double* result) {
    return unaryReduceRun<double, double>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastDnrm2((size_t)n, R_.mem, R_.offset / sizeof(double),
                X_.mem, X_.offset / sizeof(double), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n,
                            const float* x, int incx, float* result) {
    return unaryReduceRun<float, float>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastSasum((size_t)n, R_.mem, R_.offset / sizeof(float),
                X_.mem, X_.offset / sizeof(float), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasDasum(hipblasHandle_t handle, int n,
                            const double* x, int incx, double* result) {
    return unaryReduceRun<double, double>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastDasum((size_t)n, R_.mem, R_.offset / sizeof(double),
                X_.mem, X_.offset / sizeof(double), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasScnrm2(hipblasHandle_t handle, int n,
                              const hipblasComplex* x, int incx, float* result) {
    return unaryReduceRun<float, hipblasComplex>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t Ve = sizeof(hipblasComplex);
            return CLBlastScnrm2((size_t)n, R_.mem, R_.offset / sizeof(float),
                X_.mem, X_.offset / Ve, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasDznrm2(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              double* result) {
    return unaryReduceRun<double, hipblasDoubleComplex>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t Ve = sizeof(hipblasDoubleComplex);
            return CLBlastDznrm2((size_t)n, R_.mem, R_.offset / sizeof(double),
                X_.mem, X_.offset / Ve, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasScasum(hipblasHandle_t handle, int n,
                              const hipblasComplex* x, int incx, float* result) {
    return unaryReduceRun<float, hipblasComplex>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t Ve = sizeof(hipblasComplex);
            return CLBlastScasum((size_t)n, R_.mem, R_.offset / sizeof(float),
                X_.mem, X_.offset / Ve, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasDzasum(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              double* result) {
    return unaryReduceRun<double, hipblasDoubleComplex>(handle, n, incx, result, x,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t Ve = sizeof(hipblasDoubleComplex);
            return CLBlastDzasum((size_t)n, R_.mem, R_.offset / sizeof(double),
                X_.mem, X_.offset / Ve, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasSrot(hipblasHandle_t handle, int n, float* x, int incx,
                            float* y, int incy, const float* c, const float* s) {
    if (!c || !s) return HIPBLAS_STATUS_INVALID_VALUE;
    return vec2Run(handle, n, incx, incy, sizeof(float), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            return CLBlastSrot((size_t)n,
                X_.mem, X_.offset / sizeof(float), (size_t)incx,
                Y_.mem, Y_.offset / sizeof(float), (size_t)incy,
                *c, *s, q, nullptr);
        });
}

hipblasStatus_t hipblasDrot(hipblasHandle_t handle, int n, double* x, int incx,
                            double* y, int incy, const double* c, const double* s) {
    if (!c || !s) return HIPBLAS_STATUS_INVALID_VALUE;
    return vec2Run(handle, n, incx, incy, sizeof(double), x, y,
        [&](StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            return CLBlastDrot((size_t)n,
                X_.mem, X_.offset / sizeof(double), (size_t)incx,
                Y_.mem, Y_.offset / sizeof(double), (size_t)incy,
                *c, *s, q, nullptr);
        });
}

hipblasStatus_t hipblasSrotg(hipblasHandle_t handle,
                             float* a, float* b, float* c, float* s) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!a || !b || !c || !s) return HIPBLAS_STATUS_INVALID_VALUE;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    StagedBuffer sa, sb, sc, ss;
    auto rc = chipblas::bridgeStage(*h, a, sizeof(float), BufDir::INOUT, &sa);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, b, sizeof(float), BufDir::INOUT, &sb);
    if (rc != HIPBLAS_STATUS_SUCCESS) { chipblas::bridgeWriteBack(*h, sa); return rc; }
    rc = chipblas::bridgeStage(*h, c, sizeof(float), BufDir::INOUT, &sc);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        chipblas::bridgeWriteBack(*h, sb);
        return rc;
    }
    rc = chipblas::bridgeStage(*h, s, sizeof(float), BufDir::INOUT, &ss);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        chipblas::bridgeWriteBack(*h, sb);
        chipblas::bridgeWriteBack(*h, sc);
        return rc;
    }
    cl_command_queue queue = h->queue;
    int clb = CLBlastSrotg(sa.mem, sa.offset / sizeof(float),
        sb.mem, sb.offset / sizeof(float),
        sc.mem, sc.offset / sizeof(float),
        ss.mem, ss.offset / sizeof(float),
        &queue, nullptr);
    chipblas::bridgeWriteBack(*h, sa);
    chipblas::bridgeWriteBack(*h, sb);
    chipblas::bridgeWriteBack(*h, sc);
    auto wb = chipblas::bridgeWriteBack(*h, ss);
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

hipblasStatus_t hipblasDrotg(hipblasHandle_t handle,
                             double* a, double* b, double* c, double* s) {
    if (!handle) return HIPBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!a || !b || !c || !s) return HIPBLAS_STATUS_INVALID_VALUE;
    auto* h = reinterpret_cast<Handle*>(handle);
    if (!h->isOpenCL) return HIPBLAS_STATUS_NOT_SUPPORTED;
    StagedBuffer sa, sb, sc, ss;
    auto rc = chipblas::bridgeStage(*h, a, sizeof(double), BufDir::INOUT, &sa);
    if (rc != HIPBLAS_STATUS_SUCCESS) return rc;
    rc = chipblas::bridgeStage(*h, b, sizeof(double), BufDir::INOUT, &sb);
    if (rc != HIPBLAS_STATUS_SUCCESS) { chipblas::bridgeWriteBack(*h, sa); return rc; }
    rc = chipblas::bridgeStage(*h, c, sizeof(double), BufDir::INOUT, &sc);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        chipblas::bridgeWriteBack(*h, sb);
        return rc;
    }
    rc = chipblas::bridgeStage(*h, s, sizeof(double), BufDir::INOUT, &ss);
    if (rc != HIPBLAS_STATUS_SUCCESS) {
        chipblas::bridgeWriteBack(*h, sa);
        chipblas::bridgeWriteBack(*h, sb);
        chipblas::bridgeWriteBack(*h, sc);
        return rc;
    }
    cl_command_queue queue = h->queue;
    int clb = CLBlastDrotg(sa.mem, sa.offset / sizeof(double),
        sb.mem, sb.offset / sizeof(double),
        sc.mem, sc.offset / sizeof(double),
        ss.mem, ss.offset / sizeof(double),
        &queue, nullptr);
    chipblas::bridgeWriteBack(*h, sa);
    chipblas::bridgeWriteBack(*h, sb);
    chipblas::bridgeWriteBack(*h, sc);
    auto wb = chipblas::bridgeWriteBack(*h, ss);
    auto tr = chipblas::translate(clb);
    return (tr != HIPBLAS_STATUS_SUCCESS) ? tr : wb;
}

hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x,
                              int incx, int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(float), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastiSamax((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / sizeof(float), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x,
                              int incx, int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(double), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastiDamax((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / sizeof(double), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n,
                              const hipblasComplex* x, int incx, int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(hipblasComplex), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastiCamax((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(hipblasDoubleComplex), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastiZamax((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x,
                              int incx, int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(float), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastiSamin((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / sizeof(float), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x,
                              int incx, int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(double), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastiDamin((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / sizeof(double), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n,
                              const hipblasComplex* x, int incx, int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(hipblasComplex), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastiCamin((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n,
                              const hipblasDoubleComplex* x, int incx,
                              int* result) {
    return iamaxRun(handle, n, incx, x, sizeof(hipblasDoubleComplex), result,
        [&](StagedBuffer& R_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastiZamin((size_t)n,
                R_.mem, R_.offset / sizeof(uint32_t),
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasSsymm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, int m, int n, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return gemmRun(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        symmABytes(side, m, n, lda, sizeof(float)),
        symmBcBytes(m, n, ldb, sizeof(float)),
        symmBcBytes(m, n, ldc, sizeof(float)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastSsymm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                (size_t)m, (size_t)n, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasDsymm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, int m, int n, const double* alpha,
    const double* A, int lda, const double* B, int ldb,
    const double* beta, double* C, int ldc) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return gemmRun(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        symmABytes(side, m, n, lda, sizeof(double)),
        symmBcBytes(m, n, ldb, sizeof(double)),
        symmBcBytes(m, n, ldc, sizeof(double)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDsymm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                (size_t)m, (size_t)n, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasCsymm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, int m, int n, const hipblasComplex* alpha,
    const hipblasComplex* A, int lda, const hipblasComplex* B, int ldb,
    const hipblasComplex* beta, hipblasComplex* C, int ldc) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    cl_float2 b = {{beta->x, beta->y}};
    return gemmRun(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        symmABytes(side, m, n, lda, sizeof(hipblasComplex)),
        symmBcBytes(m, n, ldb, sizeof(hipblasComplex)),
        symmBcBytes(m, n, ldc, sizeof(hipblasComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCsymm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasZsymm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, int m, int n, const hipblasDoubleComplex* alpha,
    const hipblasDoubleComplex* A, int lda, const hipblasDoubleComplex* B, int ldb,
    const hipblasDoubleComplex* beta, hipblasDoubleComplex* C, int ldc) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    cl_double2 b = {{beta->x, beta->y}};
    return gemmRun(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        symmABytes(side, m, n, lda, sizeof(hipblasDoubleComplex)),
        symmBcBytes(m, n, ldb, sizeof(hipblasDoubleComplex)),
        symmBcBytes(m, n, ldc, sizeof(hipblasDoubleComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZsymm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasHsymm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, int m, int n, const hipblasHalf* alpha,
    const hipblasHalf* A, int lda, const hipblasHalf* B, int ldb,
    const hipblasHalf* beta, hipblasHalf* C, int ldc) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_half ah{}, bh{};
    std::memcpy(&ah, alpha, sizeof(ah));
    std::memcpy(&bh, beta, sizeof(bh));
    return gemmRun(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        symmABytes(side, m, n, lda, sizeof(hipblasHalf)),
        symmBcBytes(m, n, ldb, sizeof(hipblasHalf)),
        symmBcBytes(m, n, ldc, sizeof(hipblasHalf)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            return CLBlastHsymm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                (size_t)m, (size_t)n, ah,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                bh, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasChemm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, int m, int n, const hipblasComplex* alpha,
    const hipblasComplex* A, int lda, const hipblasComplex* B, int ldb,
    const hipblasComplex* beta, hipblasComplex* C, int ldc) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    cl_float2 b = {{beta->x, beta->y}};
    return gemmRun(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        symmABytes(side, m, n, lda, sizeof(hipblasComplex)),
        symmBcBytes(m, n, ldb, sizeof(hipblasComplex)),
        symmBcBytes(m, n, ldc, sizeof(hipblasComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastChemm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasZhemm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, int m, int n, const hipblasDoubleComplex* alpha,
    const hipblasDoubleComplex* A, int lda, const hipblasDoubleComplex* B, int ldb,
    const hipblasDoubleComplex* beta, hipblasDoubleComplex* C, int ldc) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    cl_double2 b = {{beta->x, beta->y}};
    return gemmRun(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
        symmABytes(side, m, n, lda, sizeof(hipblasDoubleComplex)),
        symmBcBytes(m, n, ldb, sizeof(hipblasDoubleComplex)),
        symmBcBytes(m, n, ldc, sizeof(hipblasDoubleComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZhemm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasSsyrk(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        syrkABytes(trans, n, k, lda, sizeof(float)),
        syrkCBytes(n, ldc, sizeof(float)),
        A, C,
        [&](StagedBuffer& A_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastSsyrk(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                 *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasDsyrk(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        syrkABytes(trans, n, k, lda, sizeof(double)),
        syrkCBytes(n, ldc, sizeof(double)),
        A, C,
        [&](StagedBuffer& A_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDsyrk(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                 *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasCsyrk(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasComplex* alpha, const hipblasComplex* A, int lda, const hipblasComplex* beta, hipblasComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    cl_float2 b = {{beta->x, beta->y}};
    return buf2Run(handle,
        syrkABytes(trans, n, k, lda, sizeof(hipblasComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasComplex)),
        A, C,
        [&](StagedBuffer& A_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCsyrk(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, a,
                A_.mem, A_.offset / E, (size_t)lda,
                 b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasZsyrk(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* A, int lda, const hipblasDoubleComplex* beta, hipblasDoubleComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    cl_double2 b = {{beta->x, beta->y}};
    return buf2Run(handle,
        syrkABytes(trans, n, k, lda, sizeof(hipblasDoubleComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasDoubleComplex)),
        A, C,
        [&](StagedBuffer& A_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZsyrk(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, a,
                A_.mem, A_.offset / E, (size_t)lda,
                 b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasHsyrk(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasHalf* alpha, const hipblasHalf* A, int lda,
    const hipblasHalf* beta, hipblasHalf* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_half ah{}, bh{};
    std::memcpy(&ah, alpha, sizeof(ah));
    std::memcpy(&bh, beta, sizeof(bh));
    return buf2Run(handle,
        syrkABytes(trans, n, k, lda, sizeof(hipblasHalf)),
        syrkCBytes(n, ldc, sizeof(hipblasHalf)),
        A, C,
        [&](StagedBuffer& A_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            return CLBlastHsyrk(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, ah,
                A_.mem, A_.offset / E, (size_t)lda,
                 bh, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasCherk(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const float* alpha, const hipblasComplex* A, int lda,
    const float* beta, hipblasComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        syrkABytes(trans, n, k, lda, sizeof(hipblasComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasComplex)),
        A, C,
        [&](StagedBuffer& A_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCherk(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasZherk(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const double* alpha, const hipblasDoubleComplex* A, int lda,
    const double* beta, hipblasDoubleComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        syrkABytes(trans, n, k, lda, sizeof(hipblasDoubleComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasDoubleComplex)),
        A, C,
        [&](StagedBuffer& A_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZherk(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasSsyr2k(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return gemmRun(handle, trans, trans,
        syrkABytes(trans, n, k, lda, sizeof(float)),
        syrkABytes(trans, n, k, ldb, sizeof(float)),
        syrkCBytes(n, ldc, sizeof(float)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastSsyr2k(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasDsyr2k(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const double* alpha, const double* A, int lda, const double* B, int ldb,
    const double* beta, double* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    return gemmRun(handle, trans, trans,
        syrkABytes(trans, n, k, lda, sizeof(double)),
        syrkABytes(trans, n, k, ldb, sizeof(double)),
        syrkCBytes(n, ldc, sizeof(double)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDsyr2k(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasCsyr2k(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasComplex* alpha, const hipblasComplex* A, int lda, const hipblasComplex* B, int ldb,
    const hipblasComplex* beta, hipblasComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    cl_float2 b = {{beta->x, beta->y}};
    return gemmRun(handle, trans, trans,
        syrkABytes(trans, n, k, lda, sizeof(hipblasComplex)),
        syrkABytes(trans, n, k, ldb, sizeof(hipblasComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCsyr2k(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasZsyr2k(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* A, int lda, const hipblasDoubleComplex* B, int ldb,
    const hipblasDoubleComplex* beta, hipblasDoubleComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    cl_double2 b = {{beta->x, beta->y}};
    return gemmRun(handle, trans, trans,
        syrkABytes(trans, n, k, lda, sizeof(hipblasDoubleComplex)),
        syrkABytes(trans, n, k, ldb, sizeof(hipblasDoubleComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasDoubleComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZsyr2k(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                b, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasHsyr2k(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasHalf* alpha, const hipblasHalf* A, int lda,
    const hipblasHalf* B, int ldb, const hipblasHalf* beta,
    hipblasHalf* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_half ah{}, bh{};
    std::memcpy(&ah, alpha, sizeof(ah));
    std::memcpy(&bh, beta, sizeof(bh));
    return gemmRun(handle, trans, trans,
        syrkABytes(trans, n, k, lda, sizeof(hipblasHalf)),
        syrkABytes(trans, n, k, ldb, sizeof(hipblasHalf)),
        syrkCBytes(n, ldc, sizeof(hipblasHalf)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            return CLBlastHsyr2k(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, ah,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                bh, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasCher2k(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasComplex* alpha, const hipblasComplex* A, int lda,
    const hipblasComplex* B, int ldb, const float* beta,
    hipblasComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta || !B) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    return gemmRun(handle, trans, trans,
        syrkABytes(trans, n, k, lda, sizeof(hipblasComplex)),
        syrkABytes(trans, n, k, ldb, sizeof(hipblasComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCher2k(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasZher2k(hipblasHandle_t handle,
    hipblasFillMode_t uplo, hipblasOperation_t trans, int n, int k,
    const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* A, int lda,
    const hipblasDoubleComplex* B, int ldb, const double* beta,
    hipblasDoubleComplex* C, int ldc) {
    if (uplo == HIPBLAS_FILL_MODE_FULL) return HIPBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta || !B) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    return gemmRun(handle, trans, trans,
        syrkABytes(trans, n, k, lda, sizeof(hipblasDoubleComplex)),
        syrkABytes(trans, n, k, ldb, sizeof(hipblasDoubleComplex)),
        syrkCBytes(n, ldc, sizeof(hipblasDoubleComplex)),
        A, B, C,
        [&](StagedBuffer& A_, StagedBuffer& B_, StagedBuffer& C_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZher2k(CLBlastLayoutColMajor, mapTriangle(uplo), mapTranspose(trans),
                (size_t)n, (size_t)k, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb,
                *beta, C_.mem, C_.offset / E, (size_t)ldc, q, nullptr);
        });
}

hipblasStatus_t hipblasStrmm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const float* alpha, const float* A, int lda,
    float* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(float)),
        trmmBBytes(m, n, ldb, sizeof(float)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastStrmm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasStrsm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const float* alpha, const float* A, int lda,
    float* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(float)),
        trmmBBytes(m, n, ldb, sizeof(float)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(float);
            return CLBlastStrsm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasDtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const double* alpha, const double* A, int lda,
    double* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(double)),
        trmmBBytes(m, n, ldb, sizeof(double)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDtrmm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const double* alpha, const double* A, int lda,
    double* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(double)),
        trmmBBytes(m, n, ldb, sizeof(double)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(double);
            return CLBlastDtrsm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, *alpha,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasCtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const hipblasComplex* alpha, const hipblasComplex* A, int lda,
    hipblasComplex* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(hipblasComplex)),
        trmmBBytes(m, n, ldb, sizeof(hipblasComplex)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCtrmm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasCtrsm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const hipblasComplex* alpha, const hipblasComplex* A, int lda,
    hipblasComplex* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(hipblasComplex)),
        trmmBBytes(m, n, ldb, sizeof(hipblasComplex)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCtrsm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasZtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* A, int lda,
    hipblasDoubleComplex* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(hipblasDoubleComplex)),
        trmmBBytes(m, n, ldb, sizeof(hipblasDoubleComplex)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZtrmm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasZtrsm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* A, int lda,
    hipblasDoubleComplex* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(hipblasDoubleComplex)),
        trmmBBytes(m, n, ldb, sizeof(hipblasDoubleComplex)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZtrsm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, a,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

hipblasStatus_t hipblasHtrmm(hipblasHandle_t handle, hipblasSideMode_t side,
    hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag,
    int m, int n, const hipblasHalf* alpha, const hipblasHalf* A, int lda,
    hipblasHalf* B, int ldb) {
    auto ge = rejectFullBoth(uplo, side);
    if (ge != HIPBLAS_STATUS_SUCCESS) return ge;
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_half ah{};
    std::memcpy(&ah, alpha, sizeof(ah));
    return buf2Run(handle,
        trmmABytes(side, m, n, lda, sizeof(hipblasHalf)),
        trmmBBytes(m, n, ldb, sizeof(hipblasHalf)),
        A, B,
        [&](StagedBuffer& A_, StagedBuffer& B_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasHalf);
            return CLBlastHtrmm(CLBlastLayoutColMajor, mapSide(side), mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)m, (size_t)n, ah,
                A_.mem, A_.offset / E, (size_t)lda,
                B_.mem, B_.offset / E, (size_t)ldb, q, nullptr);
        });
}

// --- GER / TRMV / TRSV -----------------------------------------------------

hipblasStatus_t hipblasSger(hipblasHandle_t handle, int m, int n,
    const float* alpha, const float* x, int incx, const float* y, int incy,
    float* A, int lda) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    size_t aB = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);
    return ger3Run(handle, m, n, incx, incy,
        aB, vecBytes(m, incx, sizeof(float)), vecBytes(n, incy, sizeof(float)),
        x, y, A,
        [&](StagedBuffer& A_, StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            return CLBlastSger(CLBlastLayoutColMajor, (size_t)m, (size_t)n,
                *alpha, X_.mem, X_.offset / sizeof(float), (size_t)incx,
                Y_.mem, Y_.offset / sizeof(float), (size_t)incy,
                A_.mem, A_.offset / sizeof(float), (size_t)lda, q, nullptr);
        });
}

hipblasStatus_t hipblasDger(hipblasHandle_t handle, int m, int n,
    const double* alpha, const double* x, int incx, const double* y, int incy,
    double* A, int lda) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    size_t aB = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(double);
    return ger3Run(handle, m, n, incx, incy,
        aB, vecBytes(m, incx, sizeof(double)), vecBytes(n, incy, sizeof(double)),
        x, y, A,
        [&](StagedBuffer& A_, StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            return CLBlastDger(CLBlastLayoutColMajor, (size_t)m, (size_t)n,
                *alpha, X_.mem, X_.offset / sizeof(double), (size_t)incx,
                Y_.mem, Y_.offset / sizeof(double), (size_t)incy,
                A_.mem, A_.offset / sizeof(double), (size_t)lda, q, nullptr);
        });
}

hipblasStatus_t hipblasCgeru(hipblasHandle_t handle, int m, int n,
    const hipblasComplex* alpha, const hipblasComplex* x, int incx,
    const hipblasComplex* y, int incy, hipblasComplex* A, int lda) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    size_t aB = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(hipblasComplex);
    return ger3Run(handle, m, n, incx, incy,
        aB, vecBytes(m, incx, sizeof(hipblasComplex)),
        vecBytes(n, incy, sizeof(hipblasComplex)),
        x, y, A,
        [&](StagedBuffer& A_, StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCgeru(CLBlastLayoutColMajor, (size_t)m, (size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                A_.mem, A_.offset / E, (size_t)lda, q, nullptr);
        });
}

hipblasStatus_t hipblasCgerc(hipblasHandle_t handle, int m, int n,
    const hipblasComplex* alpha, const hipblasComplex* x, int incx,
    const hipblasComplex* y, int incy, hipblasComplex* A, int lda) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_float2 a = {{alpha->x, alpha->y}};
    size_t aB = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(hipblasComplex);
    return ger3Run(handle, m, n, incx, incy,
        aB, vecBytes(m, incx, sizeof(hipblasComplex)),
        vecBytes(n, incy, sizeof(hipblasComplex)),
        x, y, A,
        [&](StagedBuffer& A_, StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCgerc(CLBlastLayoutColMajor, (size_t)m, (size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                A_.mem, A_.offset / E, (size_t)lda, q, nullptr);
        });
}

hipblasStatus_t hipblasZgeru(hipblasHandle_t handle, int m, int n,
    const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx,
    const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* A, int lda) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    size_t aB = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(hipblasDoubleComplex);
    return ger3Run(handle, m, n, incx, incy,
        aB, vecBytes(m, incx, sizeof(hipblasDoubleComplex)),
        vecBytes(n, incy, sizeof(hipblasDoubleComplex)),
        x, y, A,
        [&](StagedBuffer& A_, StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZgeru(CLBlastLayoutColMajor, (size_t)m, (size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                A_.mem, A_.offset / E, (size_t)lda, q, nullptr);
        });
}

hipblasStatus_t hipblasZgerc(hipblasHandle_t handle, int m, int n,
    const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx,
    const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* A, int lda) {
    if (!alpha) return HIPBLAS_STATUS_INVALID_VALUE;
    cl_double2 a = {{alpha->x, alpha->y}};
    size_t aB = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(hipblasDoubleComplex);
    return ger3Run(handle, m, n, incx, incy,
        aB, vecBytes(m, incx, sizeof(hipblasDoubleComplex)),
        vecBytes(n, incy, sizeof(hipblasDoubleComplex)),
        x, y, A,
        [&](StagedBuffer& A_, StagedBuffer& X_, StagedBuffer& Y_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZgerc(CLBlastLayoutColMajor, (size_t)m, (size_t)n, a,
                X_.mem, X_.offset / E, (size_t)incx,
                Y_.mem, Y_.offset / E, (size_t)incy,
                A_.mem, A_.offset / E, (size_t)lda, q, nullptr);
        });
}

hipblasStatus_t hipblasStrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n, const float* A,
    int lda, float* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(float),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastStrmv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / sizeof(float), (size_t)lda,
                X_.mem, X_.offset / sizeof(float), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasDtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n, const double* A,
    int lda, double* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(double),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastDtrmv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / sizeof(double), (size_t)lda,
                X_.mem, X_.offset / sizeof(double), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasCtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n,
    const hipblasComplex* A, int lda, hipblasComplex* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(hipblasComplex),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCtrmv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasZtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n,
    const hipblasDoubleComplex* A, int lda, hipblasDoubleComplex* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(hipblasDoubleComplex),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZtrmv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasStrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n, const float* A,
    int lda, float* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(float),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastStrsv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / sizeof(float), (size_t)lda,
                X_.mem, X_.offset / sizeof(float), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasDtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n, const double* A,
    int lda, double* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(double),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            return CLBlastDtrsv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / sizeof(double), (size_t)lda,
                X_.mem, X_.offset / sizeof(double), (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasCtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n,
    const hipblasComplex* A, int lda, hipblasComplex* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(hipblasComplex),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasComplex);
            return CLBlastCtrsv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}

hipblasStatus_t hipblasZtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo,
    hipblasOperation_t transA, hipblasDiagType_t diag, int n,
    const hipblasDoubleComplex* A, int lda, hipblasDoubleComplex* x, int incx) {
    return trmvLike(handle, uplo, n, lda, incx, A, x, sizeof(hipblasDoubleComplex),
        [&](StagedBuffer& A_, StagedBuffer& X_, cl_command_queue* q) {
            constexpr size_t E = sizeof(hipblasDoubleComplex);
            return CLBlastZtrsv(CLBlastLayoutColMajor, mapTriangle(uplo),
                mapTranspose(transA), mapDiag(diag), (size_t)n,
                A_.mem, A_.offset / E, (size_t)lda,
                X_.mem, X_.offset / E, (size_t)incx, q, nullptr);
        });
}
