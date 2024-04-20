
#include <assert.h>
#include <cstdint>
#include <cmath>
#include <fallback_internal.h>
#include <tensor_dim.h>

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

#define haxpy_loop()                                                       \
  do {                                                                     \
    unsigned int i;                                                        \
    for (i = 0; i < N; ++i)                                                \
      Y[i * incY] = Y[i * incY] + static_cast<_FP16>(alpha) * X[i * incX]; \
  } while (0);

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
namespace nntrainer {
#ifdef ENABLE_FP16
void __fallback_sscal(const unsigned int N, const float alpha, _FP16 *X,
                      const int incX) {
  for (unsigned int i = 0; i < N; ++i)
    X[i * incx] = static_cast<_FP16>(alpha) * X[i * incx];
}

_FP16 __fallback_snrm2(const unsigned int N, const _FP16 *X, const int incX) {
  unsigned int incx = abs(incX);
  float sum = 0;
  float tmp;
  for (unsigned int i = 0; i < N; i++) {
    tmp = static_cast<float>(X[i * incx]);
    sum += tmp * tmp;
  }
  return static_cast<_FP16>(sqrt(sum));
}

void __fallback_scopy(const unsigned int N, const _FP16 *X, const int incX,
                      _FP16 *Y, const int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incy] = X[i * incx];
}

void __fallback_scopy(const unsigned int N, const float *X, const int incX,
                      _FP16 *Y, const int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<_FP16>(X[i * incX]);
}

void __fallback_scopy(const unsigned int N, const _FP16 *X, const int incX,
                      float *Y, const int incY) {
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = static_cast<float>(X[i * incX]);
}

void __fallback_scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                      const int incX, _FP16 *Y,
                                      const int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void __fallback_scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                                      const int incX, _FP16 *Y,
                                      const int incY) {
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
                      const int incX, _FP16 *Y, const int incY) {
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
                      const _FP16 *X, const int incX, const float beta,
                      _FP16 *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);

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
                               const int incX) {
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
#endif
void __fallback_sscal(const unsigned int N, const float alpha, float *X,
                      const int incX) {
  assert(incX > 0);
  for (unsigned int i = 0; i < N; ++i)
    X[i * incX] = alpha * X[i * incX];
}

float __fallback_snrm2(const unsigned int N, const float *X, const int incX) {
  assert(incX > 0);
  float sum = 0.0f;
  float tmp;

  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incX];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

void __fallback_scopy(const unsigned int N, const float *X, const int incX,
                      float *Y, const int intY) {
  assert(incX > 0 && intY > 0);
  for (unsigned int i = 0; i < N; ++i)
    Y[i * intY] = X[i * incX];
}

void __fallback_scopy(const unsigned int N, const uint8_t *X, const int incX,
                      uint8_t *Y, const int intY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * intY];
  }
}

void __fallback_scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                      const int incX, float *Y,
                                      const int intY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void __fallback_scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                                      const int incX, float *Y,
                                      const int intY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * intY];
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
                      const int incX, float *Y, const int incY) {
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
                      const float *X, const int incX, const float beta,
                      float *Y, const int incY) {
  unsigned int incy = abs(incY);
  unsigned int incx = abs(incX);
  if (TransA == true) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

unsigned int __fallback_isamax(const unsigned int N, const float *X,
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
}
