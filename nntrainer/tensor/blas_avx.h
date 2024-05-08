// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   blas_avx.h
 * @date   20 Feb 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a header for AVX implementation
 *
 */

#ifndef __BLAS_AVX2_H_
#define __BLAS_AVX2_H_
#ifdef __cplusplus

#include <cmath>
#include <immintrin.h>

namespace nntrainer::avx {

#ifdef ENABLE_FP16
/**
 * @brief Converts half-precision floating point values to single-precision
 * floating point values.
 *
 * @param[in]  N number of elements in input vector
 * @param[in]  input vector containing 16-bit floating point values
 * @param[out] output vector containing single-precision floating point values.
 */
void vcvt_f16_f32(size_t N, const void *input, float *output);

/**
 * @brief  Converts single-precision floating point values to half-precision
 * floating point values.
 *
 * @param[in]  N number of elements in input vector
 * @param[in]  input vector containing single-precision floating point values
 * @param[out] output vector containing 16-bit floating point values
 */
void vcvt_f32_f16(size_t N, const float *input, void *output);

/**
 * @brief     check if the X has NaN value
 * @note it compare !(x==x)
 * @param[in] N  length of the vector
 * @param[in] X half-precision * for Vector X
 * @param[out] true if it has NaN
 */
bool hasNaN(const size_t N, const _Float16 *X);
#endif

/**
 * @brief     check if the X has NaN value
 * @note it compare !(x==x)
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[out] true if it has NaN
 */
bool hasNaN(const size_t N, const float *X);

} // namespace nntrainer::avx

#endif /* __cplusplus */
#endif /* __BLAS_AVX_H_ */
