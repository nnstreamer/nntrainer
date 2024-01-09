// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_interface.h
 * @date   28 Aug 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is dummy header for blas support
 *
 */

#ifndef __BLAS_INTERFACE_H_
#define __BLAS_INTERFACE_H_
#ifdef __cplusplus

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#else
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };

enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
};

#endif

#ifdef USE_CUBLAS
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

#include <cstdint>
#include <tensor_dim.h>
namespace nntrainer {

#ifdef ENABLE_FP16
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] alpha float number
 */
void sscal(const unsigned int N, const float alpha, _FP16 *X, const int incX);

/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
_FP16 snrm2(const int N, const _FP16 *X, const int incX);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy(const unsigned int N, const _FP16 *X, const int incX, _FP16 *Y,
           const int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy(const unsigned int N, const float *X, const int incX, _FP16 *Y,
           const int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y float * for Vector Y
 */
void scopy(const unsigned int N, const _FP16 *X, const int incX, float *Y,
           const int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const int incX, _FP16 *Y, const int incY);

/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY);

/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const int incX, _FP16 *Y, const int incY);

/**
 * @brief     sgemm computation : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] A __fp16 * for Matrix A
 * @param[in] B __fp16 * for Matrix B
 * @param[in] C __fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc);
/**
 * @brief     sgemv computation : Y = alpha*A*X + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const int incX,
           const float beta, _FP16 *Y, const int incY);
/**
 * @brief     elementwise vector multiplication : Z = X ⊙ Y
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 */
void ewvm(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z);

/**
 * @brief     elementwise vector addition : Z = X + Y
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 */
void ewva(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z);

/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
unsigned int isamax(const unsigned int N, const _FP16 *X, const int incX);

/**
 * @brief squared root transformation inplace : X = sqrt(X)
 *
 * @param N size of X
 * @param X __fp16 * for Vector X
 */
void inv_sqrt_inplace(const unsigned int N, _FP16 *X);
#endif
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X void * for Vector X
 * @param[in] alpha float number
 */
void sscal(const unsigned int N, const float alpha, void *X, const int incX,
           ml::train::TensorDim::DataType d_type);
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] alpha float number
 */
void sscal(const unsigned int N, const float alpha, float *X, const int incX);
/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
float snrm2(const int N, const float *X, const int incX);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X void * for Vector X
 * @param[in] Y void * for Vector Y
 */
void scopy(const unsigned int N, const void *X, const int incX, void *Y,
           const int incY, ml::train::TensorDim::DataType d_type);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
void scopy(const unsigned int N, const float *X, const int incX, float *Y,
           const int intY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void scopy(const unsigned int N, const uint8_t *X, const int incX, uint8_t *Y,
           const int intY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int intY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const int incX, float *Y, const int intY);

/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY);
/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X void * for Vector X
 * @param[in] Y void * for Vector Y
 */
void saxpy(const unsigned int N, const float alpha, const void *X,
           const int incX, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type);
/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
void saxpy(const unsigned int N, const float alpha, const float *X,
           const int incX, float *Y, const int incY);
/**
 * @brief     sgemm computation  : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] A void * for Matrix A
 * @param[in] B void * for Matrix B
 * @param[in] C void * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const void *A, const unsigned int lda,
           const void *B, const unsigned int ldb, const float beta, void *C,
           const unsigned int ldc, ml::train::TensorDim::DataType d_type);
/**
 * @brief     sgemm computation  : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] A float * for Matrix A
 * @param[in] B float * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc);
/**
 * @brief     sgemv computation  : Y = alpha*A*X + beta*Y
 * @param[in] A void * for Matrix A
 * @param[in] X void * for Vector X
 * @param[in] Y void * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const void *A,
           const unsigned int lda, const void *X, const int incX,
           const float beta, void *Y, const int incY,
           ml::train::TensorDim::DataType d_type);
/**
 * @brief     sgemv computation  : Y = alpha*A*X + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const int incX,
           const float beta, float *Y, const int incY);
/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
unsigned int isamax(const unsigned int N, const float *X, const int incX);

/**
 * @brief     sine transformation with neon: Y = sin(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void sine_transformation(const unsigned int N, float *X, float *Y,
                         float alpha = 1.0);

/**
 * @brief     cosine transformation with neon: Y = cos(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void cosine_transformation(const unsigned int N, float *X, float *Y,
                           float alpha = 1.0);

/**
 * @brief inversed squared root transformation inplace : X = 1 / sqrt(X)
 *
 * @param N size of X
 * @param X float * for Vector X
 */
void inv_sqrt_inplace(const unsigned int N, float *X);
} /* namespace nntrainer */
#endif /* __cplusplus */
#endif /* __BLAS_INTERFACE_H__ */
