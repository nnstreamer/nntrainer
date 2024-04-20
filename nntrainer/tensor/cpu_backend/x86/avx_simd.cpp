#include <avx_simd.h>

namespace nntrainer {
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
  sgemm(TransA, TransB, M, N, K, alpha, A_, lda, B_, ldb, beta, C_, ldc);
  scopy(M * N, C_, 1, C, 1);

  delete[] A_;
  delete[] B_;
  delete[] C_;
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const int incX,
           const float beta, _FP16 *Y, const int incY) {
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

  sgemv(TransA, M, N, alpha, A_, lda, X_, incX, beta, Y_, incY);

  scopy(lenY, Y_, 1, Y, 1);

  delete[] A_;
  delete[] X_;
  delete[] Y_;
}

void scopy(const unsigned int N, const float *X, const int incX, _FP16 *Y,
           const int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::avx::vcvt_f32_f16(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incy] = static_cast<_FP16>(X[i * incx]);
  }
}

void scopy(const unsigned int N, const _FP16 *X, const int incX, float *Y,
           const int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::avx::vcvt_f16_f32(N, X, Y);
  } else {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incy] = static_cast<float>(X[i * incx]);
  }
}
} // namespace nntrainer