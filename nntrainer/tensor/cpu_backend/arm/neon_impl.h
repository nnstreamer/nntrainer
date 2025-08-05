// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file neon_impl.h
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Single-precision computation functions based on NEON
 *
 */

#ifndef __NEON_IMPL_H_
#define __NEON_IMPL_H_
#ifdef __cplusplus

#include <arm_neon.h>
#include <cmath>
#include <neon_mathfun.h>
#include <tensor_dim.h>

namespace nntrainer::neon {
#ifdef ENABLE_FP16
/**
 * @brief Accelerating function for rotary embedding layer forwarding
 *
 * @param dim unit length of simd computation
 * @param half_ criterion for rotational direction of embedding
 * @param w current w value from b, c, h, w
 * @param in __fp16* input
 * @param out __fp16* output
 * @param cos_ precomputed cos_ for corresponding rotational indices
 * @param sin_ precomputed sin_ for corresponding rotational indices
 */
void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, __fp16 *in, __fp16 *out,
                                    float *cos_, float *sin_);
/**
 * @brief swiglu function with neon : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X __fp16 * for Vector X
 * @param Y __fp16 * for Vector Y
 * @param Z __fp16 * for Vector Z
 */
void swiglu(const unsigned int N, __fp16 *X, __fp16 *Y, __fp16 *Z);

/**
 * @brief returns maximum value of the vector X
 *
 * @param N number of elements in X
 * @param X __fp16 * for Vector X
 * @return __fp16 maximum value of vector X
 */
__fp16 max_val(const unsigned int N, __fp16 *X);

/**
 * @brief soft max function with neon y_i = exp(x_i) / sum( exp(x_i) )
 * Note that half-precision softmax function needs to be computed with
 * single-precision
 *
 * @param N number of elements in X
 * @param X __fp16 * for Vector X
 * @param Y  __fp16 * for Vector Y
 */
void softmax(const unsigned int N, __fp16 *X, __fp16 *Y);

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
 * @param[in] X int8_t * for Vector X
 * @param[in] Y __fp16 * for Vector Y
 */
void copy_int8_to_fp16(const unsigned int N, const int8_t *X, __fp16 *Y);

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
 * @brief     check if the X is valid: Check NaN or Inf
 * @note it compare (x!=x || x == inf)
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[out] false if it has NaN or Inf
 */
bool is_valid(const unsigned int N, const __fp16 *X);

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
void transpose_matrix(const unsigned int M, const unsigned int N,
                      const __fp16 *src, unsigned int ld_src, __fp16 *dst,
                      unsigned int ld_dst);
#endif
/**
 * @brief Elementwise multiplication with neon : Z = X ⊙ Y
 *
 * @param lhs int8_t * input
 * @param rhs int8_t * input
 * @param res int8_t * output
 * @param data_len length of input data
 * @param lhs_scale float scale of lhs data
 * @param rhs_scale float scale of rhs data
 * @param res_scale resultant scale factor
 * @param scale_len length of scale factor
 */
void ele_qmul(int8_t *lhs, int8_t *rhs, int8_t *res, unsigned int data_len,
              const float *lhs_scale, const float *rhs_scale,
              const float *res_scale, unsigned int scale_len);
/**
 * @brief Get half-sized angles, transform them into each cos, sin, and scopy in
 * the same vector : cos_ = cos(freq).extend(cos(freq)), sin_ =
 * sin(freq).extend(sin_(req))
 *
 * @param N_half : size of angle
 * @param angle float* for Vector (radian) angle
 * @param cos_ float* for cos_
 * @param sin_ float* for sin_
 * @param alpha scaling factor
 */
void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int alpha = 1.0);

/**
 * @brief swiglu function with neon : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y float * for Vector Y
 * @param Z float * for Vector Z
 */
void swiglu(const unsigned int N, float *X, float *Y, float *Z);

/**
 * @brief returns maximum value of the vector X
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @return float maximum value of vector X
 */
float max_val(const unsigned int N, float *X);

/**
 * @brief soft max function with neon y_i = exp(x_i) / sum( exp(x_i) )
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y  float * for Vector Y
 */
void softmax(const unsigned int N, float *X, float *Y);

/**
 * @brief exponential inplace function
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 */
void exp_i(const unsigned int N, float *X);
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
void copy_int8_to_fp32(const unsigned int N, const uint8_t *X, float *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X int8_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void copy_int8_to_fp32(const unsigned int N, const int8_t *X, float *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint8_t * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void copy_int8_or_int4(const unsigned int N, const uint8_t *X, uint8_t *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X int8_t * for Vector X
 * @param[in] Y int8_t * for Vector Y
 */
void copy_s8(const unsigned int N, const int8_t *X, int8_t *Y);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X int16_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint16_t * for Vector X
 * @param[in] Y float * for Vector Y
 */
void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X int16_t * for Vector X
 * @param[in] Y int16_t * for Vector Y
 */
void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y);

/**
 * @brief     copy function : Y = X
 * @param[in] N number of elements in X
 * @param[in] X uint16_t * for Vector X
 * @param[in] Y uint16_t * for Vector Y
 */
void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y);

/**
 * @brief     check if the X has NaN value or Inf
 * @note it compare (x!=x || x == inf)
 * @param[in] N  length of the vector
 * @param[in] input float * for Vector X
 * @param[out] false if it has NaN or Inf
 */
bool is_valid(const unsigned int N, const float *input);

/**
 * @brief cblas_scopy occasionally emits SIGSEGV, so implement a custom version.
 *
 * @param N length of the vector
 * @param X float * for Vector X (input)
 * @param Y float * for Vector Y (output)
 */
void custom_scopy(const unsigned int N, const float *X, const int incX,
                  float *Y, const int incY);

/**
 * @brief     copy function with neon: Y = X
 * @param[in] N number of elements in X
 * @param[in] X float * for Vector X
 * @param[in] Y uint8_t * for Vector Y
 */
void copy_int4_to_fp32(const unsigned int N, const uint8_t *X, float *Y);

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
 * @brief Matrix transpose / 2D Tensor transpose
 *
 * @param M row length of input matrix
 * @param N col length of input matrix
 * @param src src data of input matrix
 * @param ld_src data offset of input matrix
 * @param dst destination of output matrix
 * @param ld_dst data offset of output matrix
 */
void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst);

/**
 * @brief
 * @param qk_out float * input/output values
 * @param start_row start row number
 * @param end_row end row number
 * @param num_heads heads number
 */
void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads);

/**
 * @brief
 * @param qk_out float * input/output values
 * @param start_row start row number
 * @param end_row end row number
 * @param num_heads heads number
 */
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads);

#if !defined(ARMV7)
/**
 * @brief
 * @param iter
 * @param in
 * @param vcache
 * @param output
 * @param seq
 * @param num_cache_head
 * @param gqa_size
 * @param head_dim
 * @param process_all
 */
void compute_fp16vcache_fp32_transposed(int iter, const float *in,
                                        const __fp16 *vcache, float *output,
                                        int seq, int num_cache_head,
                                        int gqa_size, int head_dim,
                                        bool process_all);

/**
 * @brief
 * @tparam BType
 * @param A
 * @param B
 * @param output
 * @param num_rows
 * @param N
 * @param chunk_size
 * @param group_size
 * @param tile_size
 */
template <typename BType>
void compute_kcaches(const float *A, const BType *B, float *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size);

/**
 * @brief
 * @param width
 * @param dim
 * @param half_
 * @param inout
 * @param output
 * @param cos_
 * @param sin_
 * @param only_convert_to_fp16
 */
void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16);
#endif
} // namespace nntrainer::neon

#endif /* __cplusplus */
#endif /* __NEON_SINGLE_H__ */
