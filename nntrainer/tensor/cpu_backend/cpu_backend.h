// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   cpu_backend.h
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Computational backend for CPU considering architecture dependency
 *
 */

#ifndef __CPU_BACKEND_H__
#define __CPU_BACKEND_H__
#ifdef __cplusplus
#if defined(__aarch64__) || defined(__ARM_ARCH_7A__) ||                        \
  defined(__ANDROID__) || defined(__arm__)
#include <arm_compute_backend.h>
#elif defined(__x86_64__) || defined(__i586__) || defined(_M_X64) ||           \
  defined(_M_IX86)
#include <x86_compute_backend.h>
#else
#include <fallback.h>
#endif

#include <cstdint>
#include <tensor_dim.h>

#ifdef ENABLE_FP16
/**
 * @brief Accelerating function for rotary embedding layer forwarding
 *
 * @param dim unit length of simd computation
 * @param half_ criterion for rotational direction of embedding
 * @param w current w value from b, c, h, w
 * @param in _FP16* input
 * @param out _FP16* output
 * @param cos_ precomputed cos_ for corresponding rotational indices
 * @param sin_ precomputed sin_ for corresponding rotational indices
 */
extern void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                           unsigned int w, _FP16 *in,
                                           _FP16 *out, float *cos_,
                                           float *sin_);
/**
 * @brief swiglu function : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X _FP16 * for Vector X
 * @param Y _FP16 * for Vector Y
 * @param Z _FP16 * for Vector Z
 */
extern void swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z);

/**
 * @brief returns maximum value of the vector X
 *
 * @param N number of elements in X
 * @param X _FP16 * for Vector X
 * @return _FP16 maximum value of vector X
 */
extern _FP16 max_val(const unsigned int N, _FP16 *X);

/**
 * @brief soft max function with neon y_i = exp(x_i) / sum( exp(x_i) )
 * Note that half-precision softmax function needs to be computed with
 * single-precision
 *
 * @param N number of elements in X
 * @param X _FP16 * for Vector X
 * @param Y  _FP16 * for Vector Y
 */
extern void softmax(const unsigned int N, _FP16 *X, _FP16 *Y);

/**
 * @brief     check if X array has NaN or inf
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[out] bool false if not valid else true
 */
extern bool is_valid(const unsigned int N, const _FP16 *X);

/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] alpha float number
 */
extern void sscal(const unsigned int N, const float alpha, _FP16 *X,
                  const unsigned int incX);

/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
extern _FP16 snrm2(const unsigned int N, const _FP16 *X,
                   const unsigned int incX);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
extern void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
extern void scopy(const unsigned int N, const float *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  float *Y, const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
extern void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
extern void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
extern void scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);
/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
extern _FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  const _FP16 *Y, const unsigned int incY);

/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
extern void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
                  const unsigned int incX, _FP16 *Y, const unsigned int incY);

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
extern void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
                  const unsigned int M, const unsigned int N,
                  const unsigned int K, const float alpha, const _FP16 *A,
                  const unsigned int lda, const _FP16 *B,
                  const unsigned int ldb, const float beta, _FP16 *C,
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
extern void sgemv(const unsigned int TStorageOrder, bool TransA,
                  const unsigned int M, const unsigned int N, const float alpha,
                  const _FP16 *A, const unsigned int lda, const _FP16 *X,
                  const unsigned int incX, const float beta, _FP16 *Y,
                  const unsigned int incY);
/**
 * @brief     elementwise vector multiplication : Z = X ⊙ alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     elementwise vector addition : Z = X + alpha * Y + beta *
 * Z
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);
/**
 * @brief     elementwise vector subtraction with neon : Z = X - alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     elementwise vector division with neon : Z = X / (alpha * Y) + beta
 * * Z
 * @note ZeroDivisionError is not guaranteed in this function
 * @param[in] N  length of the vector
 * @param[in] X _FP16 * for Vector X
 * @param[in] Y _FP16 * for Vector Y
 * @param[in] Z _FP16 * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
extern unsigned int isamax(const unsigned int N, const _FP16 *X,
                           const unsigned int incX);

/**
 * @brief squared root transformation inplace : X = sqrt(X)
 *
 * @param N size of X
 * @param X __fp16 * for Vector X
 */
extern void inv_sqrt_inplace(const unsigned int N, _FP16 *X);

/**
 * @brief Matrix transpose / 2D Tensor transpose
 *
 * @param M row length of input matrix
 * @param N col length of input matrix
 * @param src src data of input matrix
 * @param ld_src data offset of input matrix
 * @param dst destination of output matrix
 * @param ld_dst data offset of output matrix
 */
extern void transpose_matrix(const unsigned int M, const unsigned int N,
                             const _FP16 *src, unsigned int ld_src, _FP16 *dst,
                             unsigned int ld_dst);
#endif
/**
 * @brief Get half-sized angles, transform them into each cos, sin, and scopy in
 * the same vector : cos_ = cos(freq).extend(cos(freq)), sin_ =
 * sin(freq).extend(sin_(req))
 *
 * @param N_half : size of angle
 * @param freqs float* for Vector angle
 * @param cos_ float* for cos_
 * @param sin_ float* for sin_
 * @param alpha scaling factor
 */
extern void calc_trigonometric_vals_dup(unsigned int N_half, float *angle,
                                        float *cos_, float *sin_,
                                        unsigned int alpha = 1.0);
/**
 * @brief swiglu function with neon : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y float * for Vector Y
 * @param Z float * for Vector Z
 */
extern void swiglu(const unsigned int N, float *X, float *Y, float *Z);

/**
 * @brief returns maximum value of the vector X
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @return float maximum value of vector X
 */
extern float max_val(const unsigned int N, float *X);

/**
 * @brief softmax function y_i = exp(x_i) / sum( exp(x_i) )
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y  float * for Vector Y
 */
extern void softmax(const unsigned int N, float *X, float *Y);

/**
 * @brief Matrix transpose / 2D Tensor transpose
 *
 * @param M row length of input matrix
 * @param N col length of input matrix
 * @param src src data of input matrix
 * @param ld_src data offset of input matrix
 * @param dst destination of output matrix
 * @param ld_dst data offset of output matrix
 */
extern void transpose_matrix(const unsigned int M, const unsigned int N,
                             const float *src, unsigned int ld_src, float *dst,
                             unsigned int ld_dst);
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] alpha float number
 */
extern void sscal(const unsigned int N, const float alpha, float *X,
                  const unsigned int incX);
/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
extern float snrm2(const unsigned int N, const float *X,
                   const unsigned int incX);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void scopy(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
extern void scopy(const unsigned int N, const uint8_t *X,
                  const unsigned int incX, uint8_t *Y, const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int8_t * for Vector X
 * @param[in] Y int8_t * for Vector Y
 */
extern void scopy(const unsigned int N, const int8_t *X,
                  const unsigned int incX, int8_t *Y, const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                  const unsigned int incX, float *Y,
                                  const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int16_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint16_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int16_t * for Vector X
 * @param[in] Y int16_t * for Vector Y
 */
extern void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                                  const unsigned int incX, float *Y,
                                  const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                                  const unsigned int incX, float *Y,
                                  const unsigned int incY);
/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern float sdot(const unsigned int N, const float *X, const unsigned int incX,
                  const float *Y, const unsigned int incY);

/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
extern void saxpy(const unsigned int N, const float alpha, const float *X,
                  const unsigned int incX, float *Y, const unsigned int incY);
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
extern void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
                  const unsigned int M, const unsigned int N,
                  const unsigned int K, const float alpha, const float *A,
                  const unsigned int lda, const float *B,
                  const unsigned int ldb, const float beta, float *C,
                  const unsigned int ldc);
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
extern void sgemv(const unsigned int TStorageOrder, bool TransA,
                  const unsigned int M, const unsigned int N, const float alpha,
                  const float *A, const unsigned int lda, const float *X,
                  const unsigned int incX, const float beta, float *Y,
                  const unsigned int incY);
/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
extern unsigned int isamax(const unsigned int N, const float *X,
                           const unsigned int incX);

/**
 * @brief     sine with neon: Y = sin(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
extern void sine(const unsigned int N, float *X, float *Y, float alpha = 1.f);

/**
 * @brief     cosine with neon: Y = cos(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
extern void cosine(const unsigned int N, float *X, float *Y, float alpha = 1.f);

/**
 * @brief inversed squared root transformation inplace : X = 1 / sqrt(X)
 *
 * @param N size of X
 * @param X float * for Vector X
 */
extern void inv_sqrt_inplace(const unsigned int N, float *X);
/**
 * @brief     elementwise vector multiplication : Z = X ⊙ alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_mul(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

/**
 * @brief     elementwise vector addition : Z = X + alpha * Y + beta *
 * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_add(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);
/**
 * @brief     elementwise vector subtraction with neon : Z = X - alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);

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
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
extern void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
                    float alpha = 1.f, float beta = 0.f,
                    unsigned int i_stride = 1, unsigned int o_stride = 1);
/**
 * @brief     check if X array has NaN or inf
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[out] bool false if not valid else true
 */
extern bool is_valid(const unsigned int N, const float *X);

/**
 * @brief q4_K GEMM : A (M,K) * W.T (N,K) = O (M,N)
 *
 * @param M Original row size of output
 * @param N Original col size of output
 * @param K Hidden size
 * @param A Input activation to be online-runtime quantized to q8_K_MxN format
 * @param lda Leading dimension of A
 * @param B (void*) (block_q4_K*) for Offline-quantized transposed weight
 * @param ldb Leading dimenstion of B
 * @param C float* output
 * @param ldc Leading dimension of C
 */
extern void gemm_q4_K(const unsigned int M, const unsigned int N,
                      const unsigned int K, const float *A,
                      const unsigned int lda, const void *B,
                      const unsigned int ldb, float *C, const unsigned int ldc);
#endif
#endif
