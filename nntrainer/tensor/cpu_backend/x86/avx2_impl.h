// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl.h
 * @date   20 Feb 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a header for AVX implementation
 *
 */

#ifndef __AVX2_IMPL_H_
#define __AVX2_IMPL_H_
#ifdef __cplusplus

#include <cstdint>
#include <stddef.h>

namespace nntrainer::avx2 {

#ifdef ENABLE_FP16
/**
 * @brief Converts half-precision floating point values to single-precision
 * floating point values.
 *
 * @param[in]  N number of elements in input vector
 * @param[in]  input vector containing 16-bit floating point values
 * @param[out] output vector containing single-precision floating point values.
 */
void vcvt_f16_f32(unsigned int N, const _Float16 *input, float *output);

/**
 * @brief  Converts single-precision floating point values to half-precision
 * floating point values.
 *
 * @param[in]  N number of elements in input vector
 * @param[in]  input vector containing single-precision floating point values
 * @param[out] output vector containing 16-bit floating point values
 */
void vcvt_f32_f16(unsigned int N, const float *input, _Float16 *output);

/**
 * @brief     check if the X has NaN value
 * @note it compare (x!=x || x == inf)
 * @param[in] N  length of the vector
 * @param[in] X half-precision * for Vector X
 * @param[out] false if it has NaN or inf
 */
bool is_valid(const unsigned int N, const _Float16 *X);
#endif

/**
 * @brief     check if the X has NaN value
 * @note it compare (x!=x || x == inf)
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[out] false if it has NaN or inf
 */
bool is_valid(const unsigned int N, const float *X);

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
 * @brief swiglu function with AVX : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y float * for Vector Y
 * @param Z float * for Vector Z
 */
void swiglu(const unsigned int N, float *X, const float *Y, const float *Z);

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
void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha = 1.f, float beta = 0.f, unsigned int i_stride = 1,
             unsigned int o_stride = 1);

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
void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

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
                                        const uint16_t *vcache, float *output,
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

} // namespace nntrainer::avx2

#endif /* __cplusplus */
#endif /* __BLAS_AVX_H_ */
