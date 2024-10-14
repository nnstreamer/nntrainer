// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback_internal.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Single-precision computation functions based on NEON
 *
 */

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <fallback_internal.h>
#include <tensor_dim.h>

#define sgemv_loop(ci, cj, cM, cN)           \
  do {                                       \
    float y0;                                \
    unsigned int i, j;                       \
    for (ci = 0; ci != cM; ci++) {           \
      y0 = Y[ci * incY] * beta;              \
      for (cj = 0; cj != cN; cj++)           \
        y0 += A[i + j * lda] * X[cj * incX]; \
      Y[ci * incY] = y0;                     \
    }                                        \
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

void __fallback_scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void __fallback_scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
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
} // namespace nntrainer
