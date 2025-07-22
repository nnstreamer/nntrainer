// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   x86_compute_backend_fp16.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for x86
 *
 */

#include <assert.h>
#include <avx2_impl.h>
#include <fallback_internal.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <x86_compute_backend.h>

#define ROW_MAJOR 0
#define COL_MAJOR 1

namespace nntrainer {

void sscal(const unsigned int N, const float alpha, _FP16 *X,
           const unsigned int incX) {
  __fallback_sscal(N, alpha, X, incX);
}

_FP16 snrm2(const unsigned int N, const _FP16 *X, const unsigned int incX) {
  assert(incX > 0);
  _FP16 sum = __fallback_snrm2(N, X, incX);
  return sum;
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::avx2::vcvt_f32_f16(N, X, Y);

  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::avx2::vcvt_f16_f32(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    __fallback_scopy_int4_to_float16(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
}

void scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
}

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  _FP16 ret = 0;
  return __fallback_sdot(N, X, incX, Y, incY);
}

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const unsigned int incX, _FP16 *Y, const unsigned int incY) {
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc) {
  float *A_ = new float[M * K];
  float *B_ = new float[N * K];
  float *C_ = new float[M * N];

  scopy(M * K, A, 1, A_, 1);
  scopy(N * K, B, 1, B_, 1);
  scopy(M * N, C, 1, C_, 1);

  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A_, lda, B_, ldb,
                beta, C_, ldc);
  scopy(M * N, C_, 1, C, 1);

  delete[] A_;
  delete[] B_;
  delete[] C_;
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const unsigned int incX,
           const float beta, _FP16 *Y, const unsigned int incY) {
  unsigned int lenX = (TransA) ? 1 + (M - 1) * (incX) : 1 + (N - 1) * (incX);
  unsigned int lenY = (TransA) ? 1 + (N - 1) * (incY) : 1 + (M - 1) * (incY);

  float *A_ = new float[M * N];
  float *X_ = new float[lenX];
  float *Y_ = new float[lenY];

  scopy(M * N, A, 1, A_, 1);
  scopy(lenX, X, 1, X_, 1);
  scopy(lenY, Y, 1, Y_, 1);

  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A_, lda, X_, incX, beta, Y_,
                incY);

  scopy(lenY, Y_, 1, Y, 1);

  delete[] A_;
  delete[] X_;
  delete[] Y_;
}

void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const float *A, const unsigned int lda,
            const _FP16 *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc) {
  float *B_ = new float[N * K];
  scopy(N * K, B, 1, B_, 1);

  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B_, ldb,
                beta, C, ldc);

  delete[] B_;
}

void shgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
            const unsigned int N, const float alpha, const float *A,
            const unsigned int lda, const _FP16 *X, const unsigned int incX,
            const float beta, float *Y, const unsigned int incY) {
  unsigned int lenX = (TransA) ? 1 + (M - 1) * (incX) : 1 + (N - 1) * (incX);
  unsigned int lenY = (TransA) ? 1 + (N - 1) * (incY) : 1 + (M - 1) * (incY);

  float *X_ = new float[lenX];

  scopy(lenX, X, 1, X_, 1);

  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X_, incX, beta, Y,
                incY);

  delete[] X_;
}

void hsgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const _FP16 *A, const unsigned int lda,
            const float *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc) {
  float *A_ = new float[M * K];

  scopy(M * K, A, 1, A_, 1);

  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A_, lda, B, ldb,
                beta, C, ldc);

  delete[] A_;
}

void hsgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
            const unsigned int N, const float alpha, const _FP16 *A,
            const unsigned int lda, const float *X, const unsigned int incX,
            const float beta, float *Y, const unsigned int incY) {
  unsigned int lenX = (TransA) ? 1 + (M - 1) * (incX) : 1 + (N - 1) * (incX);
  unsigned int lenY = (TransA) ? 1 + (N - 1) * (incY) : 1 + (M - 1) * (incY);

  float *A_ = new float[M * N];

  scopy(M * N, A, 1, A_, 1);

  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A_, lda, X, incX, beta, Y,
                incY);

  delete[] A_;
}

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

unsigned int isamax(const unsigned int N, const _FP16 *X,
                    const unsigned int incX) {
  unsigned int max_idx = 0;
  max_idx = __fallback_isamax(N, X, incX);
  return max_idx;
}

void inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  __fallback_inv_sqrt_inplace(N, X);
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const _FP16 *src, unsigned int ld_src, _FP16 *dst,
                      unsigned int ld_dst) {
  __fallback_transpose_matrix(M, N, src, ld_src, dst, ld_dst);
}

bool is_valid(const unsigned int N, const _FP16 *input) {
  return nntrainer::avx2::is_valid(N, input);
}

void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_) {
  __fallback_compute_rotary_embedding_value(dim, half_, w, in, out, cos_, sin_);
}

void swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z) {
  __fallback_swiglu(N, X, Y, Z);
}

_FP16 max_val(const unsigned int N, _FP16 *X) { return __fallback_max(N, X); }

void softmax(const unsigned int N, _FP16 *X, _FP16 *Y) {
  __fallback_softmax(N, X, Y);
}

} /* namespace nntrainer */
