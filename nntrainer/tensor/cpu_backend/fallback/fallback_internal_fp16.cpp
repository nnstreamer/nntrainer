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

#define hgemv_loop(ci, cj, cM, cN)                                      \
  do {                                                                  \
    float y0;                                                           \
    unsigned int i, j;                                                  \
    for (ci = 0; ci != cM; ci++) {                                      \
      y0 = static_cast<float>(Y[ci * incY] * static_cast<_FP16>(beta)); \
      for (cj = 0; cj != cN; cj++)                                      \
        y0 += static_cast<float>(A[i + j * lda] * X[cj * incX]);        \
      Y[ci * incY] = static_cast<_FP16>(y0);                            \
    }                                                                   \
  } while (0);

#define hgemm_loop()                                          \
  do {                                                        \
    for (unsigned int m = 0; m < M; ++m) {                    \
      for (unsigned int n = 0; n < N; ++n) {                  \
        float c = 0;                                          \
        _FP16 c_old = C[m * ldc + n];                         \
        for (unsigned int k = 0; k < K; ++k) {                \
          _FP16 a, b;                                         \
          a = ((TransA) ? A[k * lda + m] : A[m * lda + k]);   \
          b = ((TransB) ? B[n * ldb + k] : B[k * ldb + n]);   \
          c += static_cast<float>(a * b);                     \
        }                                                     \
        C[m * ldc + n] = static_cast<_FP16>(alpha * c);       \
        if (beta != 0.0)                                      \
          C[m * ldc + n] += static_cast<_FP16>(beta) * c_old; \
      }                                                       \
    }                                                         \
  } while (0);

#define haxpy_loop()                                                       \
  do {                                                                     \
    unsigned int i;                                                        \
    for (i = 0; i < N; ++i)                                                \
      Y[i * incY] = Y[i * incY] + static_cast<_FP16>(alpha) * X[i * incX]; \
  } while (0);

namespace nntrainer {

void __fallback_sscal(const unsigned int N, const float alpha, _FP16 *X,
                      const unsigned int incX) {
  for (unsigned int i = 0; i < N; ++i)
    X[i * incX] = static_cast<_FP16>(alpha) * X[i * incX];
}

_FP16 __fallback_snrm2(const unsigned int N, const _FP16 *X,
                       const unsigned int incX) {
  float sum = 0;
  float tmp;
  for (unsigned int i = 0; i < N; i++) {
    tmp = static_cast<float>(X[i * incX]);
    sum += tmp * tmp;
  }
  return static_cast<_FP16>(sqrt(sum));
}

void __fallback_scopy(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}

void __fallback_scopy(const unsigned int N, const float *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<_FP16>(X[i * incX]);
}

void __fallback_scopy(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<float>(X[i * incX]);
}

void __fallback_scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void __fallback_scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx] = X[idx];
  }
}

_FP16 __fallback_sdot(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, const _FP16 *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += static_cast<float>(X[i * incX]) * static_cast<float>(Y[i * incY]);
  }
  return static_cast<_FP16>(ret);
}

void __fallback_saxpy(const unsigned int N, const float alpha, const _FP16 *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY) {
  haxpy_loop();
}

void __fallback_sgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const _FP16 *A,
                      const unsigned int lda, const _FP16 *B,
                      const unsigned int ldb, const float beta, _FP16 *C,
                      const unsigned int ldc) {
  hgemm_loop();
}

void __fallback_sgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const _FP16 *A, const unsigned int lda,
                      const _FP16 *X, const unsigned int incX, const float beta,
                      _FP16 *Y, const unsigned int incY) {

  if (TransA == true) {
    hgemv_loop(i, j, N, M);
  } else {
    hgemv_loop(j, i, M, N);
  }
}

void __fallback_ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X * static_cast<_FP16>(alpha) * *Y + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X + static_cast<_FP16>(alpha) * *Y + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X - static_cast<_FP16>(alpha) * *Y + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X / (static_cast<_FP16>(alpha) * *Y) + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

unsigned int __fallback_isamax(const unsigned int N, const _FP16 *X,
                               const unsigned int incX) {
  unsigned int max_idx = 0;
  _FP16 max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    _FP16 cur_val = (X[n] >= 0) ? X[n] : -1 * X[n];
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }
  return max_idx;
}

void __fallback_inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = static_cast<_FP16>(1 / std::sqrt(static_cast<float>(X[i])));
  }
}

void __fallback_transpose_matrix(const unsigned int M, const unsigned int N,
                                 const _FP16 *src, unsigned int ld_src,
                                 _FP16 *dst, unsigned int ld_dst) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

} // namespace nntrainer
