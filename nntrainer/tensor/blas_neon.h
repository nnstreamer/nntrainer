/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_neon.h
 * @date   4 Aug 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is header for blas neon implementation
 *
 */

#ifndef __BLAS_NEON_H_
#define __BLAS_NEON_H_
#ifdef __cplusplus

#include <arm_neon.h>
#include <cmath>

namespace nntrainer::neon {

/**
 * @brief     sgemv computation with neon : Y = alpha*A*X + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_neon(const float *A, const float *X, float *Y, uint32_t rows,
                uint32_t cols, const float alpha, const float beta);

/**
 * @brief     transposed sgemv computation with neon
 *            Y = alpha*transpose(A)*X
 * + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_transpose_neon(const float *A, const float *X, float *Y,
                          uint32_t rows, uint32_t cols, float alpha,
                          float beta);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void scopy_neon_int4_to_fp32(const unsigned int N, const uint8_t *X, float *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void scopy_neon_int8_to_fp32(const unsigned int N, const uint8_t *X, float *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void scopy_neon_int8_or_int4(const unsigned int N, const uint8_t *X,
                             uint8_t *Y);
/**
 * @brief     sine transformation with neon: Y = sin(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void sine_transformation_neon(const unsigned int N, float *X, float *Y,
                              float alpha = 1.0);

/**
 * @brief     cosine transformation with neon: Y = cos(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void cosine_transformation_neon(const unsigned int N, float *X, float *Y,
                                float alpha = 1.0);

#ifdef ENABLE_FP16
/**
 * @brief     sgemv computation with neon : Y = alpha*A*X + beta*Y
 * @param[in] A __fp16 * for Matrix A
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t rows,
                     uint32_t cols, float alpha, float beta);

/**
 * @brief     elementwise vector multiplication with neon : Z = X ⊙ Y
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 */
void elementwise_vector_multiplication_neon_fp16(const unsigned N,
                                                 const __fp16 *X,
                                                 const __fp16 *Y, __fp16 *Z);
/**
 * @brief     elementwise vector addition with neon : Z = X + Y
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 */
void elementwise_vector_addition_neon_fp16(const unsigned N, const __fp16 *X,
                                           const __fp16 *Y, __fp16 *Z);

/**
 * @brief     transposed sgemv computation with neon
 *            Y = alpha*transpose(A)*X
 * + beta*Y
 * @param[in] A __fp16 * for Matrix A
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] rows number of A's row
 * @param[in] cols number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_transpose_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y,
                               uint32_t rows, uint32_t cols, float alpha,
                               float beta);

/**
 * @brief     saxpy computation with neon: Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void saxpy_neon_fp16(const unsigned int N, const float alpha, const __fp16 *X,
                     __fp16 *Y);

/**
 * @brief     sdot computation with neon: sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
__fp16 sdot_neon_fp16(const unsigned int N, const __fp16 *X, const __fp16 *Y);

/**
 * @brief     snrm2 computation with neon: Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
__fp16 snrm2_neon_fp16(const unsigned int N, const __fp16 *X);

/**
 * @brief     sscal computation with neon: X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] alpha float number
 */
void sscal_neon_fp16(const unsigned int N, __fp16 *X, const float alpha);

/**
 * @brief     convert uint32x4_t to float32x4_t with neon with bitwise
 * optimization
 * @param[in] u32 element to convert
 */
float32x4_t vcvtq_f32_u32_bitwise(uint32x4_t u32);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy_neon_fp16(const unsigned int N, const __fp16 *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void scopy_neon_int4_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void scopy_neon_int8_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void scopy_neon_fp32_to_fp16(const unsigned int N, const float *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y float * for Vector Y
 */
void scopy_neon_fp16_to_fp32(const unsigned int N, const __fp16 *X, float *Y);

/**
 * @brief     isamax function with neon: index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
unsigned int isamax_neon_fp16(const unsigned int N, const __fp16 *X);

/**
 * @brief     sgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
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
void sgemm_neon_fp16(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                     uint32_t N, uint32_t K, float alpha, float beta,
                     bool TransA, bool TransB);
/**
 * @brief     sgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
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
void sgemm_neon_fp16_noTrans(const __fp16 *A, const __fp16 *B, float *C,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             float beta);
/**
 * @brief     sgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
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
void sgemm_neon_fp16_transA(const __fp16 *A, const __fp16 *B, float *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            float beta);
/**
 * @brief     sgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
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
void sgemm_neon_fp16_transB(const __fp16 *A, const __fp16 *B, float *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            float beta);
/**
 * @brief     sgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
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
void sgemm_neon_fp16_transAB(const __fp16 *A, const __fp16 *B, float *C,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             float beta, uint32_t idx);
#endif

} // namespace nntrainer::neon

#endif /* __cplusplus */
#endif /* __BLAS_NEON_H__ */
