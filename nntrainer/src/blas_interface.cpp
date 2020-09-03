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

#define sgemv_loop(ci, cj, cM, cN)           \
  do {                                       \
    double y0;                               \
    unsigned int i, j;                       \
    for (ci = 0; ci != cM; ci++) {           \
      y0 = Y[ci * incy] * beta;              \
      for (cj = 0; cj != cN; cj++)           \
        y0 += A[i + j * lda] * X[cj * incx]; \
      Y[ci * incy] = y0;                     \
    }                                        \
  } while (0);

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

static void sgemv_raw(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *X, const int incX, const float beta,
                      float *Y, const int incY) {

  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);
  if (TransA == CblasTrans) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

static void scopy_raw(const unsigned int N, const float *X, const int incX,
                      float *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = X[i * incx];
}

static void sscal_raw(const unsigned int N, const float alpha, float *X,
                      const int incX) {
  unsigned int incx = abs(incX);

  for (unsigned int i = 0; i < N; ++i)
    X[i * incx] = alpha * X[i * incx];
}

static float snrm2_raw(const unsigned int N, const float *X, const int incX) {
  unsigned int incx = abs(incX);
  float sum = 0.0f;
  float tmp;
#pragma omp parallel for private(tmp) reduction(+ : sum)
  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incx];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

static void sgemm_raw(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                      CBLAS_TRANSPOSE TransB, const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *B, const unsigned int ldb, const float beta,
                      float *C, const unsigned int ldc) {

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double c = 0.0;
      float c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        float a, b;
        a = ((TransA == CblasTrans) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == CblasTrans) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0)
        C[m * ldc + n] += beta * c_old;
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

void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int incY) {
#ifdef USE_BLAS
  cblas_scopy(N, X, incX, Y, incY);
#else
  scopy_raw(N, X, incX, Y, incY);
#endif
}

void sscal(const int N, const float alpha, float *X, const int incX) {
#ifdef USE_BLAS
  cblas_sscal(N, alpha, X, incX);
#else
  sscal_raw(N, alpha, X, incX);
#endif
}

float snrm2(const int N, const float *X, const int incX) {
#ifdef USE_BLAS
  return cblas_snrm2(N, X, incX);
#else
  return snrm2_raw(N, X, incX);
#endif
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY) {
#ifdef USE_BLAS
  return cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                     incY);
#else
  return sgemv_raw(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
#endif
}

} // namespace nntrainer
