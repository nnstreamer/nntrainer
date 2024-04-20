#include <tmp_fallback.h>

namespace nntrainer {
#ifdef ENABLE_FP16

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc) {
  __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                   ldb, beta, C, ldc);
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const int incX,
           const float beta, _FP16 *Y, const int incY) {
  __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                   incY);
}

void scopy(const unsigned int N, const float *X, const int incX, _FP16 *Y,
           const int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const _FP16 *X, const int incX, float *Y,
           const int incY) {
  __fallback_scopy(N, X, incX, Y, incY);
}
#endif
} // namespace nntrainer
