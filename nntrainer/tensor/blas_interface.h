// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_interface.h
 * @date   28 Aug 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is dummy header for blas support
 *
 */

#ifndef __BLAS_INTERFACE_H_
#define __BLAS_INTERFACE_H_
#ifdef __cplusplus

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#else
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };

enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};

#endif

#ifdef USE_CUBLAS
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

#include <tensor_dim.h>

namespace nntrainer {

void sscal(const unsigned int N, const float alpha, void *X, const int incX,
           ml::train::TensorDim::DataType d_type);

void sscal(const unsigned int N, const float alpha, float *X, const int incX);

void sscal(const unsigned int N, const float alpha, __fp16 *X, const int incX);

float snrm2(const int N, const float *X, const int incX);

void scopy(const unsigned int N, const void *X, const int incX, void *Y,
           const int incY, ml::train::TensorDim::DataType d_type);

void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int intY);

void scopy(const unsigned int N, const __fp16 *X, const int incX, __fp16 *Y,
           const int intY);

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY);

void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY);

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc);

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY);

unsigned int isamax(const unsigned int N, const float *X, const int incX);

} /* namespace nntrainer */
#endif /* __cplusplus */
#endif /* __BLAS_INTERFACE_H__ */
