#include <neon_single.h>
#if (defined USE__FP16)
#include <neon_half.h>
#endif

namespace nntrainer {

#ifdef ENABLE_FP16
void sscal(const unsigned int N, const float alpha, _FP16 *X, const int incX);

_FP16 snrm2(const unsigned int N, const _FP16 *X, const int incX);

void scopy(const unsigned int N, const _FP16 *X, const int incX, _FP16 *Y,
           const int incY);

void scopy(const unsigned int N, const float *X, const int incX, _FP16 *Y,
           const int incY);

void scopy(const unsigned int N, const _FP16 *X, const int incX, float *Y,
           const int incY);

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY);

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY);

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY);

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const int incX, _FP16 *Y, const int incY);

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc);

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const int incX,
           const float beta, _FP16 *Y, const int incY);

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

unsigned int isamax(const unsigned int N, const _FP16 *X, const int incX);

void inv_sqrt_inplace(const unsigned int N, _FP16 *X);
#endif
void scopy(const unsigned int N, const uint8_t *X, const int incX, uint8_t *Y,
           const int intY) {
  if (incX == 1 && intY == 1) {
    nntrainer::neon::copy_int8_or_int4(N, X, Y);
  } else {
    for (unsigned int idx = 0; idx < N; idx++) {
      Y[idx*incX] = X[idx*intY];
    }
  }
}

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int intY) {
  nntrainer::neon::copy_int4_to_fp32(N, X, Y);
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int intY) {
  nntrainer::neon::copy_int8_to_fp32(N, X, Y);
}

void sine(const unsigned int N, float *X, float *Y, float alpha) {
  nntrainer::neon::sine(N, X, Y, alpha);
}

void cosine(const unsigned int N, float *X, float *Y, float alpha) {
  nntrainer::neon::cosine(N, X, Y, alpha);
}

void inv_sqrt_inplace(const unsigned int N, float *X);

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);
} /* namespace nntrainer */