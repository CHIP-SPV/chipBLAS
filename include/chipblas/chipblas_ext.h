/*
 * chipBLAS extensions / install marker.
 *
 * Installed under include/chipblas/ alongside the standard
 * include/hipblas/hipblas.h. Its presence is a sentinel that distinguishes
 * a chipBLAS install (OpenCL/CLBlast backed) from an H4I-HipBLAS install
 * (Intel MKL backed), since both ship lib/libhipblas.so into the same
 * unified prefix.
 */
#ifndef CHIPBLAS_EXT_H
#define CHIPBLAS_EXT_H

#define CHIPBLAS_VERSION_MAJOR 0
#define CHIPBLAS_VERSION_MINOR 1

#endif /* CHIPBLAS_EXT_H */
