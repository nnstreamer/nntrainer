// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd_neon.h
 * @date	09 Jan 2024
 * @brief	This is a collection of simd util neon functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __UTIL_SIMD_NEON_H__
#define __UTIL_SIMD_NEON_H__

#ifdef __cplusplus

#define VL_FP32 4
#define VL_FP16 8
namespace nntrainer::neon {

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
float max(const unsigned int N, float *X);

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
__fp16 max(const unsigned int N, __fp16 *X);

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
#endif

} // namespace nntrainer::neon

#endif /* __cplusplus */
#endif /* __UTIL_SIMD_NEON_H__ */
