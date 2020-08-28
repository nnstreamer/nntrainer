// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	blas_interface.h
 * @date	28 Aug 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is dummy header for blas support
 *
 */

#ifndef __BLAS_INTERFACE_H_
#define __BLAS_INTERFACE_H_
#ifdef __cplusplus

#include <assert.h>
#include <iostream>
#include <stdio.h>

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

namespace nntrainer {

/* TODO : need to scopy, sscal, snrm2 */
void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY);

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc);

} /* namespace nntrainer */
#endif /* __cplusplus */
#endif /* __BLAS_INTERFACE_H__ */
