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

void ele_add(const unsigned int N, const float *X, const float *Y,
    float *Z, float alpha, float beta,
    unsigned int i_stride, unsigned int o_stride);

} // namespace nntrainer::avx2

#endif /* __cplusplus */
#endif /* __BLAS_AVX_H_ */
