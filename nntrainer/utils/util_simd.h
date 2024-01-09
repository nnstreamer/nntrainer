// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd.h
 * @date	09 Jan 2024
 * @brief	This is a collection of simd util functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __UTIL_SIMD_H__
#define __UTIL_SIMD_H__

#ifdef __cplusplus

#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <util_simd_neon.h>

namespace nntrainer {

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
void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int alpha = 1.0);

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
void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, _FP16 *in, _FP16 *out,
                                    float *cos_, float *sin_);
#endif

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __UTIL_SIMD_H__ */
