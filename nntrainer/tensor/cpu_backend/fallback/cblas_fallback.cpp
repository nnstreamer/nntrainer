#include <assert.h>
#include <cblas_fallback.h>
#include <cmath>

namespace nntrainer {
void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY) {
  __fallback_saxpy(N, alpha, X, incX, Y, incY);
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY) {
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                   incY);
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
  __fallback_sdot(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void sscal(const unsigned int N, const float alpha, float *X, const int incX) {
  __fallback_sscal(N, alpha, X, incX);
}

float snrm2(const unsigned int N, const float *X, const int incX) {
  __fallback_snrm2(N, X, incX);
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
}

unsigned int isamax(const unsigned int N, const float *X, const int incX) {
  __fallback_isamax(N, X, incX);
}
} // namespace nntrainer
