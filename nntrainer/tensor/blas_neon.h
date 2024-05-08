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
#include <neon_mathfun.h>

namespace nntrainer::neon {

/**
 * @brief     sgemv computation with neon : Y = alpha*A*X + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] M number of A's row
 * @param[in] N number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv(const float *A, const float *X, float *Y, uint32_t M, uint32_t N,
           const float alpha, const float beta);

/**
 * @brief     transposed sgemv computation with neon
 *            Y = alpha*transpose(A)*X
 * + beta*Y
 * @param[in] A float * for Matrix A
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] M number of A's row
 * @param[in] N number of A's columns
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void sgemv_transpose(const float *A, const float *X, float *Y, uint32_t M,
                     uint32_t N, float alpha, float beta);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void copy_int4_to_fp32(const unsigned int N, const uint8_t *X, float *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void copy_int8_to_fp32(const unsigned int N, const uint8_t *X, float *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void copy_int8_or_int4(const unsigned int N, const uint8_t *X, uint8_t *Y);
/**
 * @brief     sine with neon: Y = sin(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void sine(const unsigned int N, float *X, float *Y, float alpha = 1.f);

/**
 * @brief     cosine with neon: Y = cos(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
void cosine(const unsigned int N, float *X, float *Y, float alpha = 1.f);

/**
 * @brief inversed squared root transformation with neon : X = 1 / sqrt(X)
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 */
void inv_sqrt_inplace(const unsigned int N, float *X);

/**
 * @brief     elementwise vector multiplication : Z = X ⊙ alpha * Y + beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha = 1.f, float beta = 0.f);

/**
 * @brief     elementwise vector addition : Z = X + alpha * Y + beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha = 1.f, float beta = 0.f);
/**
 * @brief     elementwise vector subtraction with neon : Z = X - alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha = 1.f, float beta = 0.f);

/**
 * @brief     elementwise vector division with neon : Z = X / (alpha * Y) + beta
 * * Z
 * @note ZeroDivisionError is not guaranteed in this function
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha = 1.f, float beta = 0.f);

/**
 * @brief     check if the X has NaN value
 * @note it compare !(x==x)
 * @param[in] N  length of the vector
 * @param[in] input float * for Vector X
 * @param[out] true if it has NaN
 */
bool hasNaN(const size_t N, const float *input);

#ifdef ENABLE_FP16
/**
 * @brief     hgemv computation with neon : Y = alpha*A*X + beta*Y
 * @param[in] A __fp16 * for Matrix A
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] M number of A's row
 * @param[in] N number of A's column
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemv(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t M, uint32_t N,
           float alpha, float beta);

/**
 * @brief     elementwise vector multiplication with neon : Z = X ⊙ alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_mul(const unsigned N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha = 1.f, float beta = 0.f);
/**
 * @brief     elementwise vector addition with neon : Z = X + alpha * Y + beta *
 * Z
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_add(const unsigned N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha = 1.f, float beta = 0.f);

/**
 * @brief     elementwise vector subtraction with neon : Z = X - alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_sub(const unsigned N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha = 1.f, float beta = 0.f);

/**
 * @brief     elementwise vector division with neon : Z = X / (alpha * Y) + beta
 * * Z
 * @note ZeroDivisionError is not guaranteed in this function
 * @param[in] N  length of the vector
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] Z __fp16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 */
void ele_div(const unsigned N, const __fp16 *X, const __fp16 *Y, __fp16 *Z,
             float alpha = 1.f, float beta = 0.f);

/**
 * @brief     transposed hgemv computation with neon
 *            Y = alpha*transpose(A)*X
 * + beta*Y
 * @param[in] A __fp16 * for Matrix A
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 * @param[in] M number of A's row
 * @param[in] N number of A's column
 * @param[in] alpha float number
 * @param[in] beta float number
 */
void hgemv_transpose(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t M,
                     uint32_t N, float alpha, float beta);

/**
 * @brief     haxpy computation with neon: Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void haxpy(const unsigned int N, const float alpha, const __fp16 *X, __fp16 *Y);

/**
 * @brief     hdot computation with neon: sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
__fp16 hdot(const unsigned int N, const __fp16 *X, const __fp16 *Y);

/**
 * @brief     hnrm2 computation with neon: Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
__fp16 hnrm2(const unsigned int N, const __fp16 *X);

/**
 * @brief     hscal computation with neon: X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] alpha float number
 */
void hscal(const unsigned int N, __fp16 *X, const float alpha);

/**
 * @brief     convert uint32x4_t to float32x4_t with neon with bitwise
 * optimization
 * @param[in] u32 element to convert
 */
float32x4_t vcvtq_f32_u32_bitwise(uint32x4_t u32);

/**
 * @brief     hcopy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void hcopy(const unsigned int N, const __fp16 *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void copy_int4_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void copy_int8_to_fp16(const unsigned int N, const uint8_t *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void copy_fp32_to_fp16(const unsigned int N, const float *X, __fp16 *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y float * for Vector Y
 */
void copy_fp16_to_fp32(const unsigned int N, const __fp16 *X, float *Y);

/**
 * @brief     isamax function with neon: index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
unsigned int isamax(const unsigned int N, const __fp16 *X);

/**
 * @brief     hgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
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
void custom_hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                  uint32_t N, uint32_t K, float alpha, float beta, bool TransA,
                  bool TransB);

/**
 * @brief squared root transformation with neon : X = sqrt(X)
 *
 * @param N number of elements in X
 * @param X __fp16 * for Vector X
 */
void inv_sqrt_inplace(const unsigned int N, __fp16 *X);

/**
 * @brief     check if the X has NaN value
 * @note it compare !(x==x)
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[out] true if it has NaN
 */
bool hasNaN(const size_t N, const __fp16 *X);
#endif

} // namespace nntrainer::neon

#endif /* __cplusplus */
#endif /* __BLAS_NEON_H__ */
