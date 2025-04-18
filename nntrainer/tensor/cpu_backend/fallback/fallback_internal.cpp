// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback_internal.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback computation functions (raw implementation)
 *
 */

#include <algorithm>
#include <assert.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fallback_internal.h>
#include <stdexcept>
#include <tensor_dim.h>

#define sgemv_loop(ci, cj, cM, cN)                                             \
  do {                                                                         \
    float y0;                                                                  \
    unsigned int i, j;                                                         \
    for (ci = 0; ci != cM; ci++) {                                             \
      y0 = Y[ci * incY] * beta;                                                \
      for (cj = 0; cj != cN; cj++)                                             \
        y0 += A[i + j * lda] * X[cj * incX];                                   \
      Y[ci * incY] = y0;                                                       \
    }                                                                          \
  } while (0);
namespace nntrainer {

void __fallback_sscal(const unsigned int N, const float alpha, float *X,
                      const unsigned int incX) {
  assert(incX > 0);
  for (unsigned int i = 0; i < N; ++i)
    X[i * incX] = alpha * X[i * incX];
}

float __fallback_snrm2(const unsigned int N, const float *X,
                       const unsigned int incX) {
  assert(incX > 0);
  float sum = 0.0f;
  float tmp;

  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incX];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

void __fallback_copy_s16_fp32(const unsigned int N, const int16_t *X,
                              float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_u16_fp32(const unsigned int N, const uint16_t *X,
                              float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_u32(const unsigned int N, const float *X,
                              uint32_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_u16(const unsigned int N, const float *X,
                              uint16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_s16(const unsigned int N, const float *X,
                              int16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_scopy(const unsigned int N, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}

void __fallback_scopy(const unsigned int N, const uint8_t *X,
                      const unsigned int incX, uint8_t *Y,
                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy(const unsigned int N, const int8_t *X,
                      const unsigned int incX, int8_t *Y,
                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

/// @todo function with the same internal representation should be merged.
void __fallback_scopy_uint8_to_float32(const unsigned int N, const uint8_t *X,
                                       const unsigned int incX, float *Y,
                                       const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

float __fallback_sdot(const unsigned int N, const float *X,
                      const unsigned int incX, const float *Y,
                      const unsigned int incY) {
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
}

void __fallback_saxpy(const unsigned int N, const float alpha, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + X[i * incX] * alpha;
}

void __fallback_sgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const float *A,
                      const unsigned int lda, const float *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc) {
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double c = 0.0;
      float c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        float a, b;
        a = ((TransA == true) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == true) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0)
        C[m * ldc + n] += beta * c_old;
    }
  }
}

void __fallback_sgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY) {

  if (TransA == true) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

unsigned int __fallback_isamax(const unsigned int N, const float *X,
                               const unsigned int incX) {
  unsigned int max_idx = 0;
  float max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    float cur_val = abs(X[n]);
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }

  return max_idx;
}

void __fallback_sine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]);
    ++i;
  }
}

void __fallback_cosine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]);
    ++i;
  }
}

void __fallback_inv_sqrt_inplace(const unsigned int N, float *X) {
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = 1 / std::sqrt(static_cast<float>(X[i]));
  }
}

void __fallback_ele_mul(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X * alpha * *Y + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_add(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X + alpha * *Y + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_sub(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X - alpha * *Y + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_div(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X / (alpha * *Y) + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_transpose_matrix(const unsigned int M, const unsigned int N,
                                 const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

bool __fallback_isValid(const unsigned int N, const float *X) {
  for (size_t i = 0; i < N; ++i) {
    if (*X != *X || *X == std::numeric_limits<float>::infinity())
      return false;
    ++X;
  }

  return true;
}

void __fallback_calc_trigonometric_vals_dup(unsigned int N_half, float *angle,
                                            float *cos_, float *sin_,
                                            unsigned int alpha) {
  throw std::runtime_error(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
}

void __fallback_swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  unsigned int i = 0;
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(-Y[i]))) * Z[i];
    ++i;
  }
}

float __fallback_max(const unsigned int N, float *X) {
  std::vector<float> v(X, X + N);
  return *std::max_element(v.begin(), v.end());
}

void __fallback_softmax(const unsigned int N, float *X, float *Y) {
  unsigned int i = 0;
  float sum = 0.f;
  float max_x = __fallback_max(N, X);
  while (i < N) {
    sum += std::exp(X[i] - max_x);
    ++i;
  }
  i = 0;
  while (i < N) {
    Y[i] = std::exp(X[i] - max_x) / sum;
    ++i;
  }
}

void __gemm_q4_K(const unsigned int M, const unsigned int N,
                 const unsigned int K, const float *A, const unsigned int lda,
                 const void *B, const unsigned int ldb, float *C,
                 const unsigned int ldc) {
  throw std::runtime_error("NYI : __gemm_q4_K");
}

size_t __quantize_q4_K(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights){
  throw std::runtime_error("NYI : __quantize_q4_K");
  return 1;
}

void __dequantize_row_q4_K(const void * x_raw, float * y, int64_t k){
  throw std::runtime_error("NYI : __dequantize_row_q4_K");
}
void __dequantize_row_q8_K(const void * x, float * y, int64_t k){
  throw std::runtime_error("NYI : __dequantize_row_q8_K");
}

void __repack_q4_K_to_q4_K_8(void* W, void* repacked_W, size_t data_size, const unsigned int M, const unsigned int N){
  throw std::runtime_error("NYI : __repack_q4_K_to_q4_K_8");
}

} // namespace nntrainer
