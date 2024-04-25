// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_interface.cpp
 * @date   28 Aug 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is dummy header for blas support
 *
 */

#include <blas_interface.h>
#include <nntrainer_error.h>

#if (defined USE__FP16 && defined USE_NEON)
#include <blas_neon.h>
#endif

#if USE_AVX
#include <blas_avx.h>
#endif

#include <cmath>

#define sgemv_loop(ci, cj, cM, cN)           \
  do {                                       \
    float y0;                                \
    unsigned int i, j;                       \
    for (ci = 0; ci != cM; ci++) {           \
      y0 = Y[ci * incy] * beta;              \
      for (cj = 0; cj != cN; cj++)           \
        y0 += A[i + j * lda] * X[cj * incx]; \
      Y[ci * incy] = y0;                     \
    }                                        \
  } while (0);

#define hgemv_loop(ci, cj, cM, cN)                                      \
  do {                                                                  \
    float y0;                                                           \
    unsigned int i, j;                                                  \
    for (ci = 0; ci != cM; ci++) {                                      \
      y0 = static_cast<float>(Y[ci * incy] * static_cast<_FP16>(beta)); \
      for (cj = 0; cj != cN; cj++)                                      \
        y0 += static_cast<float>(A[i + j * lda] * X[cj * incx]);        \
      Y[ci * incy] = static_cast<_FP16>(y0);                            \
    }                                                                   \
  } while (0);

#define haxpy_loop()                                                       \
  do {                                                                     \
    unsigned int i;                                                        \
    for (i = 0; i < N; ++i)                                                \
      Y[i * incY] = Y[i * incY] + static_cast<_FP16>(alpha) * X[i * incX]; \
  } while (0);

#define hgemm_loop()                                                      \
  do {                                                                    \
    for (unsigned int m = 0; m < M; ++m) {                                \
      for (unsigned int n = 0; n < N; ++n) {                              \
        float c = 0;                                                      \
        _FP16 c_old = C[m * ldc + n];                                     \
        for (unsigned int k = 0; k < K; ++k) {                            \
          _FP16 a, b;                                                     \
          a = ((TransA == CblasTrans) ? A[k * lda + m] : A[m * lda + k]); \
          b = ((TransB == CblasTrans) ? B[n * ldb + k] : B[k * ldb + n]); \
          c += static_cast<float>(a * b);                                 \
        }                                                                 \
        C[m * ldc + n] = static_cast<_FP16>(alpha * c);                   \
        if (beta != 0.0)                                                  \
          C[m * ldc + n] += static_cast<_FP16>(beta) * c_old;             \
      }                                                                   \
    }                                                                     \
  } while (0);

namespace nntrainer {

#ifdef ENABLE_FP16
static void saxpy_FP16(const unsigned int N, const float alpha, const _FP16 *X,
                       const int incX, _FP16 *Y, const int incY) {
  if (incX < 0 or incY < 0)
    throw std::invalid_argument(
      "Error: negative inc not supported without cblas");

#if (defined USE__FP16 && USE_NEON)
  // USE__FP16 is defined when platform is android
  if (incX == 1 && incY == 1) {
    nntrainer::neon::haxpy(N, alpha, X, Y);
  } else {
    haxpy_loop();
  }
#else
  haxpy_loop();
#endif
}

static void sgemv_FP16(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                       const unsigned int M, const unsigned int N,
                       const float alpha, const _FP16 *A,
                       const unsigned int lda, const _FP16 *X, const int incX,
                       const float beta, _FP16 *Y, const int incY) {
#if (defined USE__FP16 && USE_NEON)
  if (TransA == CblasTrans) {
    nntrainer::neon::hgemv_transpose(A, X, Y, M, N, alpha, beta);
  } else {
    nntrainer::neon::hgemv(A, X, Y, M, N, alpha, beta);
  }
#else
  unsigned int lenX =
    (TransA == CblasTrans) ? 1 + (M - 1) * abs(incX) : 1 + (N - 1) * abs(incX);
  unsigned int lenY =
    (TransA == CblasTrans) ? 1 + (N - 1) * abs(incY) : 1 + (M - 1) * abs(incY);

  float *A_ = new float[M * N];
  float *X_ = new float[lenX];
  float *Y_ = new float[lenY];

  scopy(M * N, A, 1, A_, 1);
  scopy(lenX, X, 1, X_, 1);
  scopy(lenY, Y, 1, Y_, 1);

  sgemv(order, TransA, M, N, alpha, A_, lda, X_, incX, beta, Y_, incY);

  scopy(lenY, Y_, 1, Y, 1);

  delete[] A_;
  delete[] X_;
  delete[] Y_;
#endif
}

static _FP16 sdot_FP16(const unsigned int N, const _FP16 *X,
                       const unsigned int incX, const _FP16 *Y,
                       const unsigned int incY) {

  if (incX < 0 or incY < 0)
    throw std::invalid_argument("Error: negative inc not supported");

  _FP16 ret = 0;

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1 && incY == 1) {
    ret = nntrainer::neon::hdot(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i) {
      ret += X[i * incX] * Y[i * incY];
    }
  }
#else
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
#endif
  return ret;
}

static void scopy_FP16(const unsigned int N, const _FP16 *X, const int incX,
                       _FP16 *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1 && incY == 1) {
    nntrainer::neon::hcopy(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incy] = X[i * incx];
  }
#else
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = X[i * incx];
#endif
}

static void copy_float32_to_float16(const unsigned int N, const float *X,
                                    const int incX, _FP16 *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp32_to_fp16(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incy] = X[i * incx];
  }
#elif USE_AVX
  if (incX == 1 && incY == 1) {
    nntrainer::avx::vcvt_f32_f16(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incy] = static_cast<_FP16>(X[i * incx]);
  }
#else
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = static_cast<_FP16>(X[i * incx]);
#endif
}

static void copy_float16_to_float32(const unsigned int N, const _FP16 *X,
                                    const int incX, float *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp16_to_fp32(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incy] = X[i * incx];
  }
#elif USE_AVX
  if (incX == 1 && incY == 1) {
    nntrainer::avx::vcvt_f16_f32(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incy] = static_cast<float>(X[i * incx]);
  }
#else
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = static_cast<float>(X[i * incx]);
#endif
}

static void copy_int4_to_fp16(const unsigned int N, const uint8_t *X,
                              const int incX, _FP16 *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int4_to_fp16(N, X, Y);
  } else {
    throw std::invalid_argument(
      "Error: incX == 1 && incY == 1 is supported only");
  }
#else
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
#endif
}

static void copy_int8_to_fp16(const unsigned int N, const uint8_t *X,
                              const int incX, _FP16 *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp16(N, X, Y);
  } else {
    throw std::invalid_argument(
      "Error: incX == 1 && incY == 1 is supported only");
  }
#else
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx] = X[idx];
  }
#endif
}

void sscal(const unsigned int N, const float alpha, _FP16 *X, const int incX) {
  unsigned int incx = abs(incX);

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1) {
    nntrainer::neon::hscal(N, X, alpha);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      X[i * incx] = static_cast<_FP16>(alpha) * X[i * incx];
  }
#else
  for (unsigned int i = 0; i < N; ++i)
    X[i * incx] = static_cast<_FP16>(alpha) * X[i * incx];
#endif
}

static _FP16 snrm2_FP16(const unsigned int N, const _FP16 *X, const int incX) {
  unsigned int incx = abs(incX);
  _FP16 sum;
  _FP16 tmp;
#if (defined USE__FP16 && USE_NEON)
  if (incX == 1) {
    sum = nntrainer::neon::hnrm2(N, X);
  } else {
    float sum32 = 0;
    for (unsigned int i = 0; i < N; i++) {
      tmp = X[i * incx];
      sum32 += tmp * tmp;
    }
    sum = static_cast<_FP16>(sqrt(sum32));
  }
#else
  float sum32 = 0;
  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incx];
    sum32 += tmp * tmp;
  }
  sum = static_cast<_FP16>(sqrt(sum32));
#endif
  return sum;
}

static void sgemm_FP16(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA,
                       CBLAS_TRANSPOSE TransB, const unsigned int M,
                       const unsigned int N, const unsigned int K,
                       const float alpha, const _FP16 *A,
                       const unsigned int lda, const _FP16 *B,
                       const unsigned int ldb, const float beta, _FP16 *C,
                       const unsigned int ldc) {

#if (defined USE__FP16 && USE_NEON)
  nntrainer::neon::hgemm(A, B, C, M, N, K, alpha, beta, TransA == CblasTrans,
                         TransB == CblasTrans);
#else
  float *A_ = new float[M * K];
  float *B_ = new float[N * K];
  float *C_ = new float[M * N];

  scopy(M * K, A, 1, A_, 1);
  scopy(N * K, B, 1, B_, 1);
  scopy(M * N, C, 1, C_, 1);
  sgemm(order, TransA, TransB, M, N, K, alpha, A_, lda, B_, ldb, beta, C_, ldc);
  scopy(M * N, C_, 1, C, 1);

  delete[] A_;
  delete[] B_;
  delete[] C_;
#endif
}

static unsigned int isamax_FP16(const unsigned int N, const _FP16 *X,
                                const int incX) {
  unsigned int max_idx = 0;

#if (defined USE__FP16 && USE_NEON)
  if (incX == 1 && N >= 8) {
    max_idx = nntrainer::neon::isamax(N, X);
  } else {
    _FP16 max_val = X[0];
    for (unsigned int n = 1; n < N; n += incX) {
      _FP16 cur_val = (X[n] >= 0) ? X[n] : -1 * X[n];
      if (cur_val > max_val) {
        max_val = cur_val;
        max_idx = n;
      }
    }
  }
#else
  _FP16 max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    _FP16 cur_val = (X[n] >= 0) ? X[n] : -1 * X[n];
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }
#endif

  return max_idx;
}

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const int incX, _FP16 *Y, const int incY) {
  saxpy_FP16(N, alpha, X, incX, Y, incY);
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc) {
  sgemm_FP16(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
             ldc);
}

void scopy(const unsigned int N, const _FP16 *X, const int incX, _FP16 *Y,
           const int incY) {
  scopy_FP16(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const float *X, const int incX, _FP16 *Y,
           const int incY) {
  copy_float32_to_float16(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const _FP16 *X, const int incX, float *Y,
           const int incY) {
  copy_float16_to_float32(N, X, incX, Y, incY);
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY) {
  copy_int4_to_fp16(N, X, incX, Y, incY);
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY) {
  copy_int8_to_fp16(N, X, incX, Y, incY);
}

static void ele_mul_fallback(const unsigned int N, const _FP16 *X,
                             const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X * static_cast<_FP16>(alpha) * *Y + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

static void ele_add_fallback(const unsigned int N, const _FP16 *X,
                             const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X + static_cast<_FP16>(alpha) * *Y + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

static void ele_sub_fallback(const unsigned int N, const _FP16 *X,
                             const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X - static_cast<_FP16>(alpha) * *Y + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

static void ele_div_fallback(const unsigned int N, const _FP16 *X,
                             const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X / (static_cast<_FP16>(alpha) * *Y) + static_cast<_FP16>(beta) * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#if (defined USE__FP16 && USE_NEON)
    nntrainer::neon::ele_mul(N, X, Y, Z, alpha, beta);
#else
    ele_mul_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_mul_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#if (defined USE__FP16 && USE_NEON)
    nntrainer::neon::ele_add(N, X, Y, Z, alpha, beta);
#else
    ele_add_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_add_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#if (defined USE__FP16 && USE_NEON)
    nntrainer::neon::ele_sub(N, X, Y, Z, alpha, beta);
#else
    ele_sub_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_sub_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#if (defined USE__FP16 && USE_NEON)
    nntrainer::neon::ele_div(N, X, Y, Z, alpha, beta);
#else
    ele_div_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_div_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

_FP16 snrm2(const int N, const _FP16 *X, const int incX) {
  return snrm2_FP16(N, X, incX);
}

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY) {
  return sdot_FP16(N, X, incX, Y, incY);
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const int incX,
           const float beta, _FP16 *Y, const int incY) {
  sgemv_FP16(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

unsigned int isamax(const unsigned int N, const _FP16 *X, const int incX) {
  /// @todo isamax_FP16 for BLAS_NUM_THREADS
  return isamax_FP16(N, X, incX);
}

void inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
#ifdef USE_NEON
  nntrainer::neon::inv_sqrt_inplace(N, X);
#else
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = static_cast<_FP16>(1 / std::sqrt(static_cast<float>(X[i])));
  }
#endif
}
#endif

#ifndef USE_BLAS
static void saxpy_raw(const unsigned int N, const float alpha, const float *X,
                      const int incX, float *Y, const int incY) {
  if (incX < 0 or incY < 0)
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

static float sdot_raw(const unsigned int N, const float *X,
                      const unsigned int incX, const float *Y,
                      const unsigned int incY) {
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
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

static unsigned int isamax_raw(const unsigned int N, const float *X,
                               const int incX) {

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

#endif

void sscal(const unsigned int N, const float alpha, void *X, const int incX,
           ml::train::TensorDim::DataType d_type) {

  if (d_type == ml::train::TensorDim::DataType::FP32) {

#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
    openblas_set_num_threads(BLAS_NUM_THREADS);
#endif // BLAS_NUM_THREADS
    cblas_sscal(N, alpha, (float *)X, incX);
#else  // USE_BLAS else
    sscal_raw(N, alpha, (float *)X, incX);
#endif //  USE_BLAS
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    sscal(N, alpha, (_FP16 *)X, incX);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void sscal(const unsigned int N, const float alpha, float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sscal(N, alpha, X, incX);
#else
  sscal_raw(N, alpha, X, incX);
#endif
}

void saxpy(const unsigned int N, const float alpha, const void *X,
           const int incX, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type) {
  if (d_type == ml::train::TensorDim::DataType::FP32) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
    openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
    cblas_saxpy(N, alpha, static_cast<const float *>(X), incX,
                static_cast<float *>(Y), incY);
#else
    saxpy_raw(N, alpha, static_cast<const float *>(X), incX,
              static_cast<float *>(Y), incY);
#endif
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    saxpy_FP16(N, alpha, static_cast<const _FP16 *>(X), incX,
               static_cast<_FP16 *>(Y), incY);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  saxpy_raw(N, alpha, X, incX, Y, incY);
#endif
}

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const void *A, const unsigned int lda,
           const void *B, const unsigned int ldb, const float beta, void *C,
           const unsigned int ldc, ml::train::TensorDim::DataType d_type) {

  if (d_type == ml::train::TensorDim::DataType::FP32) {
#ifdef USE_CUBLAS
    int devID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devID);
    float *d_A, *d_B, *d_C;

    unsigned int size_A = M * K * sizeof(float);
    unsigned int size_B = K * N * sizeof(float);
    unsigned int size_C = M * N * sizeof(float);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C, size_C);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasOperation_t transA =
      (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB =
      (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(handle, transA, transB, N, M, K, &alpha, d_B, N, d_A, K, &beta,
                d_C, N);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

#elif defined USE_BLAS

#ifdef BLAS_NUM_THREADS
    openblas_set_num_threads(BLAS_NUM_THREADS);
#endif

    cblas_sgemm(
      order, TransA, TransB, M, N, K, alpha, static_cast<const float *>(A), lda,
      static_cast<const float *>(B), ldb, beta, static_cast<float *>(C), ldc);
#else
    sgemm_raw(order, TransA, TransB, M, N, K, alpha,
              static_cast<const float *>(A), lda, static_cast<const float *>(B),
              ldb, beta, static_cast<float *>(C), ldc);
#endif

  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    sgemm_FP16(
      order, TransA, TransB, M, N, K, alpha, static_cast<const _FP16 *>(A), lda,
      static_cast<const _FP16 *>(B), ldb, beta, static_cast<_FP16 *>(C), ldc);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
} // namespace nntrainer

void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {

#ifdef USE_CUBLAS
  int devID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  float *d_A, *d_B, *d_C;

  unsigned int size_A = M * K * sizeof(float);
  unsigned int size_B = K * N * sizeof(float);
  unsigned int size_C = M * N * sizeof(float);

  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_C, size_C);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasOperation_t transA = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(handle, transA, transB, N, M, K, &alpha, d_B, N, d_A, K, &beta,
              d_C, N);

  cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
#elif defined USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_sgemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
#else
  sgemm_raw(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
            ldc);
#endif
}

void scopy(const unsigned int N, const void *X, const int incX, void *Y,
           const int incY, ml::train::TensorDim::DataType d_type) {

  if (d_type == ml::train::TensorDim::DataType::FP32) {

#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
    openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
    cblas_scopy(N, (float *)X, incX, (float *)Y, incY);
#else
    scopy_raw(N, (float *)X, incX, (float *)Y, incY);
#endif

  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    scopy_FP16(N, (_FP16 *)X, incX, (_FP16 *)Y, incY);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  cblas_scopy(N, X, incX, Y, incY);
#else
  scopy_raw(N, X, incX, Y, incY);
#endif
}

void scopy(const unsigned int N, const uint8_t *X, const int incX, uint8_t *Y,
           const int intY) {
#ifdef USE_NEON
  nntrainer::neon::copy_int8_or_int4(N, X, Y);
#else
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx] = X[idx];
  }
#endif
}

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int incY) {
#ifdef USE_NEON
  nntrainer::neon::copy_int4_to_fp32(N, X, Y);
#else
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
#endif
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int incY) {
#ifdef USE_NEON
  nntrainer::neon::copy_int8_to_fp32(N, X, Y);
#else
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx] = X[idx];
  }
#endif
}

float snrm2(const int N, const float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_snrm2(N, X, incX);
#else
  return snrm2_raw(N, X, incX);
#endif
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sdot(N, X, incX, Y, incY);
#else
  return sdot_raw(N, X, incX, Y, incY);
#endif
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const void *A,
           const unsigned int lda, const void *X, const int incX,
           const float beta, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type) {
  if (d_type == ml::train::TensorDim::DataType::FP32) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
    openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
    return cblas_sgemv(
      order, TransA, M, N, alpha, static_cast<const float *>(A), lda,
      static_cast<const float *>(X), incX, beta, static_cast<float *>(Y), incY);
#else

    return sgemv_raw(order, TransA, M, N, alpha, static_cast<const float *>(A),
                     lda, static_cast<const float *>(X), incX, beta,
                     static_cast<float *>(Y), incY);
#endif
  } else if (d_type == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    return sgemv_FP16(order, TransA, M, N, alpha, static_cast<const _FP16 *>(A),
                      lda, static_cast<const _FP16 *>(X), incX, beta,
                      static_cast<_FP16 *>(Y), incY);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                     incY);
#else
  return sgemv_raw(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
#endif
}

unsigned int isamax(const unsigned int N, const float *X, const int incX) {
#ifdef USE_BLAS
#ifdef BLAS_NUM_THREADS
  openblas_set_num_threads(BLAS_NUM_THREADS);
#endif
  return cblas_isamax(N, X, incX);
#else
  return isamax_raw(N, X, incX);
#endif
}

void sine(const unsigned int N, float *X, float *Y, float alpha) {
#ifdef USE_NEON
  nntrainer::neon::sine(N, X, Y, alpha);
#else
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]);
    ++i;
  }
#endif
}

void cosine(const unsigned int N, float *X, float *Y, float alpha) {
#ifdef USE_NEON
  nntrainer::neon::cosine(N, X, Y, alpha);
#else
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]);
    ++i;
  }
#endif
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
#ifdef USE_NEON
  nntrainer::neon::inv_sqrt_inplace(N, X);
#else
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = 1 / std::sqrt(static_cast<float>(X[i]));
  }
#endif
}
static void ele_mul_fallback(const unsigned int N, const float *X,
                             const float *Y, float *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X * alpha * *Y + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

static void ele_add_fallback(const unsigned int N, const float *X,
                             const float *Y, float *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X + alpha * *Y + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

static void ele_sub_fallback(const unsigned int N, const float *X,
                             const float *Y, float *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X - alpha * *Y + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

static void ele_div_fallback(const unsigned int N, const float *X,
                             const float *Y, float *Z, float alpha, float beta,
                             unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X / (alpha * *Y) + beta * *Z;
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#ifdef USE_NEON
    nntrainer::neon::ele_mul(N, X, Y, Z, alpha, beta);
#else
    ele_mul_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_mul_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#ifdef USE_NEON
    nntrainer::neon::ele_add(N, X, Y, Z, alpha, beta);
#else
    ele_add_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_add_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#ifdef USE_NEON
    nntrainer::neon::ele_sub(N, X, Y, Z, alpha, beta);
#else
    ele_sub_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_sub_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
#ifdef USE_NEON
    nntrainer::neon::ele_div(N, X, Y, Z, alpha, beta);
#else
    ele_div_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
#endif
  } else
    ele_div_fallback(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

} // namespace nntrainer
