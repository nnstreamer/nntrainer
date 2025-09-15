// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback_internal.h
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback interface
 *
 */

#ifndef __FALLBACK_INTERNAL_H__
#define __FALLBACK_INTERNAL_H__
#ifdef __cplusplus

#include <cstdint>
#include <limits>
#include <tensor_dim.h>

namespace nntrainer {

#ifdef ENABLE_FP16
/**
 * @brief F32 * F16 = F32 GEMM
 *
 * @param TStorageOrder Row major / Col major
 * @param TransA boolean flag for transpose A
 * @param TransB boolean flag for transpose B
 * @param M Row of C
 * @param N Col of C
 * @param K Shared dimension
 * @param alpha float scale alpha
 * @param A lhs matrix
 * @param lda leading dimension of A
 * @param B rhs matrix
 * @param ldb leading dimension of B
 * @param beta float scale beta
 * @param C output matrix
 * @param ldc leading dimension of C
 */
void __fallback_shgemm(const unsigned int TStorageOrder, bool TransA,
                       bool TransB, const unsigned int M, const unsigned int N,
                       const unsigned int K, const float alpha, const float *A,
                       const unsigned int lda, const _FP16 *B,
                       const unsigned int ldb, const float beta, float *C,
                       const unsigned int ldc);
/**
 * @brief F32 * F16 = F32 GEMV
 *
 * @param TStorageOrder Row major / Col major
 * @param TransA boolean flag for transpose A
 * @param TransB boolean flag for transpose B
 * @param M Row of C
 * @param N Col of C
 * @param K Shared dimension
 * @param alpha float scale alpha
 * @param A lhs matrix
 * @param lda leading dimension of A
 * @param B rhs matrix
 * @param ldb leading dimension of B
 * @param beta float scale beta
 * @param C output matrix
 * @param ldc leading dimension of C
 */
void __fallback_shgemv(const unsigned int TStorageOrder, bool TransA,
                       const unsigned int M, const unsigned int N,
                       const float alpha, const float *A,
                       const unsigned int lda, const _FP16 *X,
                       const unsigned int incX, const float beta, float *Y,
                       const unsigned int incY);
/**
 * @brief F16 * F32 = F32 GEMM
 *
 * @param TStorageOrder Row major / Col major
 * @param TransA boolean flag for transpose A
 * @param TransB boolean flag for transpose B
 * @param M Row of C
 * @param N Col of C
 * @param K Shared dimension
 * @param alpha float scale alpha
 * @param A lhs matrix
 * @param lda leading dimension of A
 * @param B rhs matrix
 * @param ldb leading dimension of B
 * @param beta float scale beta
 * @param C output matrix
 * @param ldc leading dimension of C
 */
void __fallback_hsgemm(const unsigned int TStorageOrder, bool TransA,
                       bool TransB, const unsigned int M, const unsigned int N,
                       const unsigned int K, const float alpha, const _FP16 *A,
                       const unsigned int lda, const float *B,
                       const unsigned int ldb, const float beta, float *C,
                       const unsigned int ldc);
/**
 * @brief F16 * F32 = F32 GEMV
 *
 * @param TStorageOrder Row major / Col major
 * @param TransA boolean flag for transpose A
 * @param TransB boolean flag for transpose B
 * @param M Row of C
 * @param N Col of C
 * @param K Shared dimension
 * @param alpha float scale alpha
 * @param A lhs matrix
 * @param lda leading dimension of A
 * @param B rhs matrix
 * @param ldb leading dimension of B
 * @param beta float scale beta
 * @param C output matrix
 * @param ldc leading dimension of C
 */
void __fallback_hsgemv(const unsigned int TStorageOrder, bool TransA,
                       const unsigned int M, const unsigned int N,
                       const float alpha, const _FP16 *A,
                       const unsigned int lda, const float *X,
                       const unsigned int incX, const float beta, float *Y,
                       const unsigned int incY);
/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src float* src to be quantized
 * @param dst void* dst to store quantized data
 * @param k number of elements in src
 */
void __fallback_quantize_row_q8_0(const _FP16 *__restrict src,
                                  void *__restrict dst, int64_t k);

/**
 * @brief Quantize _FP16 to q8_0 Quantization format
 *
 * @param src input src to be quantized
 * @param dst output destination for quantized data
 * @param nrow number of row
 * @param n_per_row number of elements per row
 * @param quant_weights additional information for quantization. Currently in
 * no use.
 * @return size_t total size of quantized data
 */
size_t __fallback_quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights);
/**
 * @brief q8_0 to _FP16 dequantize
 *
 * @param x_raw input src to be dequantized
 * @param y output destination for dequantized data
 * @param k data length
 */
void __fallback_dequantize_row_q8_0(const void *x_raw, _FP16 *y, int64_t k);
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] alpha float number
 */
void __fallback_sscal(const unsigned int N, const float alpha, _FP16 *X,
                      const unsigned int incX);

/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
_FP16 __fallback_snrm2(const unsigned int N, const _FP16 *X,
                       const unsigned int incX);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void __fallback_scopy(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void __fallback_scopy(const unsigned int N, const float *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_scopy(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void __fallback_scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void __fallback_scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void __fallback_scopy_int8_to_float16(const unsigned int N, const int8_t *X,
                                      const unsigned int incX, _FP16 *Y,
                                      const unsigned int incY);
/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
_FP16 __fallback_sdot(const unsigned int N, const _FP16 *X,
                      const unsigned int incX, const _FP16 *Y,
                      const unsigned int incY);

/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X __fp16 * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void __fallback_saxpy(const unsigned int N, const float alpha, const _FP16 *X,
                      const unsigned int incX, _FP16 *Y,
                      const unsigned int incY);

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
void __fallback_sgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
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
void __fallback_sgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const _FP16 *A, const unsigned int lda,
                      const _FP16 *X, const unsigned int incX, const float beta,
                      _FP16 *Y, const unsigned int incY);
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
void __fallback_ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);

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
void __fallback_ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);
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
void __fallback_ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);

/**
 * @brief     elementwise vector division with neon : Z = X / (alpha * Y) +
 * beta
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
void __fallback_ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y,
                        _FP16 *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);

/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X __fp16 * for Vector X
 */
unsigned int __fallback_isamax(const unsigned int N, const _FP16 *X,
                               const unsigned int incX);

/**
 * @brief squared root transformation inplace : X = sqrt(X)
 *
 * @param N size of X
 * @param X __fp16 * for Vector X
 */
void __fallback_inv_sqrt_inplace(const unsigned int N, _FP16 *X);

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
void __fallback_transpose_matrix(const unsigned int M, const unsigned int N,
                                 const _FP16 *src, unsigned int ld_src,
                                 _FP16 *dst, unsigned int ld_dst);
/**
 * @brief     check if X array has NaN or inf
 * @param[in] N  length of the vector
 * @param[in] X float/fp16 * for Vector X
 * @param[out] bool false if not valide else true
 */
bool __fallback_isValid(const unsigned int N, const _FP16 *X);

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
void __fallback_compute_rotary_embedding_value(unsigned int dim,
                                               unsigned int half_,
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
void __fallback_swiglu(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z);

/**
 * @brief returns maximum value of the vector X
 *
 * @param N number of elements in X
 * @param X _FP16 * for Vector X
 * @return _FP16 maximum value of vector X
 */
_FP16 __fallback_max(const unsigned int N, _FP16 *X);

/**
 * @brief soft max function with neon y_i = exp(x_i) / sum( exp(x_i) )
 * Note that half-precision softmax function needs to be computed with
 * single-precision
 *
 * @param N number of elements in X
 * @param X _FP16 * for Vector X
 * @param Y  _FP16 * for Vector Y
 */
void __fallback_softmax(const unsigned int N, _FP16 *X, _FP16 *Y);
#endif

/**
 * @copydoc unpack_q4_0x8_transpose16 in cpu_backend.h
 */
void __fallback_unpack_q4_0x8_transpose16(const void *src,
                                          uint16_t *__restrict d_out,
                                          uint16_t *__restrict qs_out, int N,
                                          int K, int CT = 1);

/**
 * @brief Get half-sized angles, transform them into each cos, sin, and scopy
 * in the same vector : cos_ = cos(freq).extend(cos(freq)), sin_ =
 * sin(freq).extend(sin_(req))
 *
 * @param N_half : size of angle
 * @param freqs float* for Vector angle
 * @param cos_ float* for cos_
 * @param sin_ float* for sin_
 * @param from from starting index for angle calculation
 * @param attention_scaling scaling factor to apply to cos and sin values
 */
template <typename T = float>
void __fallback_calc_trigonometric_vals_dup(unsigned int N_half, T *angle,
                                            T *cos_, T *sin_,
                                            unsigned int from = 0,
                                            float attention_scaling = 1.0f);
/**
 * @brief swiglu function with neon : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y float * for Vector Y
 * @param Z float * for Vector Z
 */
void __fallback_swiglu(const unsigned int N, float *X, float *Y, float *Z);

/**
 * @brief swiglu function with alpha : X = (Y / (1 + exp(- alpha * Y))) * Z
 * @param N number of elements in X
 * @param X float* for Vector X
 * @param Y float* for Vector Y
 * @param Z float* for Vector Z
 * @param alpha float
 */
void __fallback_swiglu(const unsigned int N, float *X, float *Y, float *Z,
                       float alpha);

/**
 * @brief returns maximum value of the vector X
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @return float maximum value of vector X
 */
float __fallback_max(const unsigned int N, float *X);

/**
 * @brief softmax function y_i = exp(x_i) / sum( exp(x_i) )
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y  float * for Vector Y
 */
void __fallback_softmax(const unsigned int N, float *X, float *Y);

/**
 * @brief     check if X array has NaN or inf
 * @param[in] N  length of the vector
 * @param[in] X float/fp16 * for Vector X
 * @param[out] bool false if not valide else true
 */
bool __fallback_isValid(const unsigned int N, const float *X);

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
void __fallback_transpose_matrix(const unsigned int M, const unsigned int N,
                                 const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst);
/**
 * @brief     sscal computation : X = alpha * X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] alpha float number
 */
void __fallback_sscal(const unsigned int N, const float alpha, float *X,
                      const unsigned int incX);
/**
 * @brief     snrm2 computation : Euclidean norm
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
float __fallback_snrm2(const unsigned int N, const float *X,
                       const unsigned int incX);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int16_t * for Vector X
 * @param[in] Y int16_t * for Vector Y
 */
void __fallback_copy_s16(const unsigned int N, const int16_t *X, int16_t *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint16_t * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
void __fallback_copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int16_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint16_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_copy_u16_fp32(const unsigned int N, const uint16_t *X,
                              float *Y);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint32_t * for Vector Y
 */
void __fallback_copy_fp32_u32(const unsigned int N, const float *X,
                              uint32_t *Y);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
void __fallback_copy_fp32_u16(const unsigned int N, const float *X,
                              uint16_t *Y);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void __fallback_copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int16_t * for Vector Y
 */
void __fallback_copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y int8_t * for Vector Y
 */
void __fallback_copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_scopy(const unsigned int N, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void __fallback_scopy(const unsigned int N, const uint8_t *X,
                      const unsigned int incX, uint8_t *Y,
                      const unsigned int incY);
/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int8_t * for Vector X
 * @param[in] Y int8_t * for Vector Y
 */
void __fallback_scopy(const unsigned int N, const int8_t *X,
                      const unsigned int incX, int8_t *Y,
                      const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_scopy_uint8_to_float32(const unsigned int N, const uint8_t *X,
                                       const unsigned int incX, float *Y,
                                       const unsigned int incY);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY);
/**
 * @brief     sdot computation : sum of all X * Y
 * @param[in] N number of elements in Y
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
float __fallback_sdot(const unsigned int N, const float *X,
                      const unsigned int incX, const float *Y,
                      const unsigned int incY);

/**
 * @brief     saxpy computation : Y = alpha*X + Y
 * @param[in] N number of elements in Y
 * @param[in] alpha float number
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 */
void __fallback_saxpy(const unsigned int N, const float alpha, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY);
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
void __fallback_sgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
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
void __fallback_sgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY);
/**
 * @brief     isamax function : index of first maxima
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 */
unsigned int __fallback_isamax(const unsigned int N, const float *X,
                               const unsigned int incX);

/**
 * @brief     sine with neon: Y = sin(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
template <typename T = float>
void __fallback_sine(const unsigned int N, T *X, T *Y, float alpha = 1.F,
                     float beta = 1.F);

/**
 * @brief     cosine with neon: Y = cos(alpha * X)
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] alpha float * for scaling angle (radian)
 */
template <typename T = float>
void __fallback_cosine(const unsigned int N, T *X, T *Y, float alpha = 1.F,
                       float beta = 1.F);

/**
 * @brief inversed squared root transformation inplace : X  / sqrt(X)
 *
 * @param N size of X
 * @param X float * for Vector X
 */
void __fallback_inv_sqrt_inplace(const unsigned int N, float *X);
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
void __fallback_ele_mul(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);

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
void __fallback_ele_add(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);
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
void __fallback_ele_sub(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);

/**
 * @brief     elementwise vector division with neon : Z = X / (alpha * Y) +
 * beta
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
void __fallback_ele_div(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride);

/**
 * @brief q4_0 GEMM : A (M,K) * W.T (N,K) = O (M,N)
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
template <typename T = float>
void __fallback_gemm_q4_0(const unsigned int M, const unsigned int N,
                          const unsigned int K, const T *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, T *C, const unsigned int ldc);

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
void __fallback_gemm_q4_K(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc);
/**
 * @brief q6_K GEMM : A (M,K) * W.T (N,K) = O (M,N)
 *
 * @param M Original row size of output
 * @param N Original col size of output
 * @param K Hidden size
 * @param A Input activation to be online-runtime quantized to q8_K_MxN format
 * @param lda Leading dimension of A
 * @param B (void*) (block_q4_K*) for Offline-quantized transposed weight
 * @param ldb Leading dimenstion of B
 * @param C T* output
 * @param ldc Leading dimension of C
 */
template <typename T = float>
void __fallback_gemm_q6_K(const unsigned int M, const unsigned int N,
                          const unsigned int K, const T *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, T *C, const unsigned int ldc);
/**
 * @brief (1xK)*(Kx1) dot product for q6_K and q8_K vectors
 *
 * @param K Length of vectors
 * @param v_q6_K lhs vector - data stored in Q6_K format
 * @param v_q8_K rhs vector - data stored in Q8_K format
 * @return float Result of performing dot operation on v_q6_K and v_q8_K
 */
float __fallback_dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                               const void *v_q8_K);

/**
 * @brief (1xK)*(Kx1) dot product for q6_K and f32 vectors
 *
 * @param K Length of vectors
 * @param v_q6_K lhs vector - data stored in Q6_K format
 * @param f rhs vector - data stored in f32 format
 * @return float float Result of performing dot operation on v_q6_K and f
 */
float __fallback_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f);

/**
 * @brief quantize_q4_0 function
 *
 * @param src float* to quantize
 * @param dst q4_0* to store quantized data
 * @param nrow number of rows in src
 * @param n_per_row number of elements in each row of src
 * @param quant_weights unused for now -> imatrix
 * @return size_t size of total quantized data in bytes
 */
size_t __fallback_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights);
/**
 * @brief quantize_q4_K function
 *
 * @param src float* to quantize
 * @param dst q4_K* to store quantized data
 * @param nrow number of rows in src
 * @param n_per_row number of elements in each row of src
 * @param quant_weights unused for now -> imatrix
 * @return size_t size of total quantized data in bytes
 */
size_t __fallback_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights);
/**
 * @brief quantize_q6_K function
 *
 * @param src float* to quantize
 * @param dst q6_K* to store quantized data
 * @param nrow number of rows in src
 * @param n_per_row number of elements in each row of src
 * @param quant_weights unused for now -> imatrix
 * @return size_t size of total quantized data in bytes
 */
size_t __fallback_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights);

/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src float* src to be quantized
 * @param dst void* dst to store quantized data
 * @param k number of elements in src
 */
void __fallback_quantize_row_q6_K(const float *src, void *dst, int64_t k);

/**
 * @brief Quantize float to q6_K Quantization format
 *
 * @param src float* src to be quantized
 * @param dst void* dst to store quantized data
 * @param k number of elements in src
 */
template <typename T = float>
void __fallback_quantize_row_q8_K(const T *src, void *dst, int64_t k);

/**
 * @brief dequantize row of q4_K data to float
 *
 * @param x input to be dequantized from q4_K to float
 * @param y dequantized data output
 * @param k number of elements in x
 */
void __fallback_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k);

/**
 * @brief dequantize row of q4_0 data to float
 *
 * @param x input to be dequantized from q4_0 to float
 * @param y dequantized data output
 * @param k number of elements in x
 */
void __fallback_dequantize_row_q4_0(const void *x_raw, float *y, int64_t k);

/**
 * @brief dequantize row of q6_K data to float
 *
 * @param x input to be dequantized from q6_K to float
 * @param y dequantized data output
 * @param k number of elements in x
 */
void __fallback_dequantize_row_q6_K(const void *x, float *y, int64_t k);

/**
 *
 * @brief dequantize row of q8_K data to float
 *
 * @param x input to be dequantized from q8_K to float
 * @param y dequantized data output
 * @param k number of elements in x
 */
template <typename T = float>
void __fallback_dequantize_row_q8_K(const void *x, T *y, int64_t k);

/**
 * @brief repack q40 to q40x8
 *
 * @param W input q40
 * @param repacked_W output q40x8
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __fallback_repack_q4_0_to_q4_0_4(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N);

/**
 * @brief repack q40 to q40x8
 *
 * @param W input q40
 * @param repacked_W output q40x8
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __fallback_repack_q4_0_to_q4_0_8(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N);

/**
 * @brief repack q4K to q4Kx8
 *
 * @param W input q4K
 * @param repacked_W output q4Kx8
 * @param data_size total weight size
 * @param M number of rows
 * @param N number of columns
 */
void __fallback_repack_q4_K_to_q4_K_8(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N);

/**
 * @brief Multihead softmax, exp(x_i) / sum(exp(x_i)), inplace version
 * @param[in/out] qk_out float* input/output values
 * @param[in] start_row start row number
 * @param[in] end_row end row number
 * @param[in] num_heads heads number
 */
void __fallback_softmax_row_inplace(float *qk_out, size_t start_row,
                                    size_t end_row, size_t num_heads);

/**
 * @brief Multihead softmax, exp(x_i) / sum(exp(x_i))
 * @param[in/out] qk_out float* input/output values
 * @param[in] start_row start row number
 * @param[in] end_row end row number
 * @param[in] num_heads heads number
 */
void __fallback_softmax_row(float *qk_out, size_t start_row, size_t end_row,
                            size_t num_heads);

/**
 * @brief Compute vcache for one row transposed
 * @param[in] row_num row number
 * @param[in] in float* input vector
 * @param[in] vcache uint16_t* input vector
 * @param[out] output float* output vector
 * @param[in] num_cache_head number head of cache
 * @param[in] gqa_size size of group
 * @param[in] head_dim head dimension
 * @param[in] local_window_size windows size for local attention
 */
void __fallback_compute_fp16vcache_fp32_transposed(
  int row_num, const float *in, const uint16_t *vcache, float *output,
  int num_cache_head, int gqa_size, int head_dim,
  size_t local_window_size = UINT_MAX);

/**
 * @brief Compute kcaches
 * @tparam BType type of B vector element
 * @param[in] in float* input vector
 * @param[in] kcache BType* input vector with keys cache
 * @param[out] output float* output float vector
 * @param[in] num_rows number of row
 * @param[in] num_cache_head number head of cache
 * @param[in] head_dim head dimension
 * @param[in] gqa_size size of group
 * @param[in] tile_size size of tile
 * @param[in] local_window_size windows size for local attention
 */
template <typename BType>
void __fallback_compute_kcaches(const float *in, const BType *kcache,
                                float *output, int num_rows, int num_cache_head,
                                int head_dim, int gqa_size, int tile_size,
                                size_t local_window_size = UINT_MAX);

/**
 * @brief Compute rotary embedding value
 * @param[in] width current w value from b, c, h, w
 * @param[in] dim unit length of simd computation
 * @param[in] half_ criterion for rotational direction of embedding
 * @param[in/out] inout float* uesed also as output when expected output
 * float* values
 * @param[out] output void* output values, used when expected output __fp16*
 * values
 * @param[in] cos_ float* input con values
 * @param[in] sin_ float* input sin values
 * @param[in] only_convert_to_fp16 equal true if method is used only for
 * conversion
 */
void __fallback_compute_rotary_emb_value(unsigned int width, unsigned int dim,
                                         unsigned int half_, float *inout,
                                         void *output, const float *cos_,
                                         const float *sin_,
                                         bool only_convert_to_fp16);
/**
 * @brief rms normalization computation w.r.t. width in H*W matrix input
 *
 * @param X input
 * @param Y output
 * @param H height of input matrix
 * @param W width of input matrix
 * @param epsilon epsilon of root mean squared dividing scale
 */
void __fallback_rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                                  float *__restrict Y, size_t H,
                                                  size_t W, float epsilon);
/**
 * @brief rms normalization computation w.r.t. width in H*W matrix input
 *
 * @param X input
 * @param Y output
 * @param H height of input matrix
 * @param W width of input matrix
 * @param epsilon epsilon of root mean squared dividing scale
 */
template <typename T = float>
void __fallback_rms_norm_wrt_width_fp16_intrinsic(const T *__restrict X,
                                                  T *__restrict Y, size_t H,
                                                  size_t W, float epsilon);
/**
 * @brief fallback for clamping function.
 *
 * @tparam T Type of input data
 * @param input input vector
 * @param output output vector
 * @param length length of IO
 * @param lower_bound ditto
 * @param upper_bound ditto
 */
template <typename T = float>
void __fallback_clamp(const T *input, T *output, size_t length,
                      T lower_bound = std::numeric_limits<T>::lowest(),
                      T upper_bound = std::numeric_limits<T>::max());
/**
 * @brief qs4cx quantization of (n*k) matrix. Typically a weight quantization,
 * and generally regard the weight is already transposed, and quantize it as it
 * is. qs4cx refers to quantized symmetric 4-bit quantization of channelwise x
 * groups.
 *
 * @param n N length of the matrix
 * @param k K length of the matrix
 * @param rhs_native_mtx_f32 matrix data before quantization to load
 * @param rhs_native_mtx_qs4cx matrix data after quantization to stroe
 * @param rhs_scales_f32 matrix quant scale after quantization to stroe
 * @param transB
 */
void __fallback_nntr_quant_qs4cx_f32(size_t n, size_t k,
                                     void *rhs_native_mtx_f32,
                                     void *rhs_native_mtx_qs4cx,
                                     void *rhs_scales_f32, bool transB = true);
/**
 * @brief GEMM of qai8dxp runtime-quantized activation and offline qs4cx
 * quantized weight
 *
 * @tparam T dataType of input activation and output matrices
 * @param m M length of the matrix
 * @param n N length of the matrix
 * @param k K length of the matrix
 * @param lhs_native_mtx activation (not quantized)
 * @param rhs_native_mtx_qs4cx offline quantized weight
 * @param rhs_scales scale factor vector of quantized weight
 * @param dst_mtx dst matrix
 * @param lower_bound lower bound to clamp
 * @param upper_bound upper bound to clamp
 * @param transB Choose weight data to be transposed or not. Default value
 * regards the weight to be transpoed.
 */
template <typename T = float>
void __fallback_nntr_gemm_qai8dxp_qsi4cxp(
  size_t m, size_t n, size_t k, void *lhs_native_mtx,
  void *rhs_native_mtx_qs4cx, void *rhs_scales, T *dst_mtx, bool transB = true,
  T lower_bound = std::numeric_limits<T>::lowest(),
  T upper_bound = std::numeric_limits<T>::max());
} // namespace nntrainer
#endif
#endif
