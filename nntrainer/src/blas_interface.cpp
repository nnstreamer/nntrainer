// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	blas_interface.cpp
 * @date	28 Aug 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is dummy header for blas support
 *
 */

#include <blas_interface.h>

namespace nntrainer {

#ifndef USE_BLAS
static void saxpy_raw(const unsigned int N, const float alpha, const float *X,
                      const int incX, float *Y, const int incY) {
  if (incX <= 0 or incY <= 0)
    throw std::invalid_argument(
      "Error: negative inc not supported without cblas");
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + X[i * incX] * alpha;
}

static void sgemm_raw(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                      CBLAS_TRANSPOSE TransB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *B, const unsigned int ldb, const float beta,
                      float *C, const unsigned int ldc) {
  if (TransA != CblasNoTrans or TransB != CblasNoTrans)
    throw std::invalid_argument("Error: Transpose of A or B is not supported");

  for (unsigned int i = 0; i < M; ++i) {
    for (unsigned int j = 0; j < N; ++j) {
      C[i * N + j] = beta * C[i * N + j];
      for (unsigned int k = 0; k < K; ++k) {
        C[i * N + j] += alpha * A[i * K + k] * B[k * N + j];
      }
    }
  }
}
#endif

void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY) {
#ifdef USE_BLAS
  cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  saxpy_raw(N, alpha, X, incX, Y, incY);
#endif
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {

#ifdef USE_BLAS
  cblas_sgemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
#else
  sgemm_raw(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
            ldc);
#endif
}

} // namespace nntrainer
