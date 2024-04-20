
#ifndef __CBLAS_INTERFACE_H__
#define __CBLAS_INTERFACE_H__
#ifdef __cplusplus

// #include <cblas.h>

namespace nntrainer {
void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY);

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY);

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY);

void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int incY);

void sscal(const unsigned int N, const float alpha, float *X, const int incX);

float snrm2(const unsigned int N, const float *X, const int incX);

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc);

unsigned int isamax(const unsigned int N, const float *X, const int incX);
} // namespace nntrainer

#endif
#endif
