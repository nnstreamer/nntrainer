// SPDX-License-Identifier: Apache-2.0
/**
 * @file	util_simd_neon.cpp
 * @date	09 Jan 2024
 * @brief	This is a collection of simd util neon functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <blas_neon.h>
#include <util_simd_neon.h>

namespace nntrainer::neon {

void calc_trigonometric_vals_dup_neon(unsigned int N_half, float *angle,
                                             float *cos_, float *sin_,
                                             unsigned int from) {
  cosine_transformation_neon(N_half, angle, cos_, from);
  sine_transformation_neon(N_half, angle, sin_, from);

  unsigned int N = 2 * N_half;
  unsigned int i = N_half;
  unsigned int i_half = 0;

  for (; (N - i >= 4) && (N_half - i_half >= 4);
       i += 4, i_half += 4) {
    vst1q_f32(&cos_[i], vld1q_f32(&cos_[i_half]));
    vst1q_f32(&sin_[i], vld1q_f32(&sin_[i_half]));
  }
  while (i < N || i_half < N_half) {
    cos_[i] = cos_[i_half];
    sin_[i] = sin_[i_half];
    ++i;
    ++i_half;
  }
}

#ifdef ENABLE_FP16
void compute_rotary_embedding_value_neon(unsigned int dim, unsigned int half_,
                                         unsigned int w, __fp16 *in,
                                         __fp16 *out, float *cos_,
                                         float *sin_) {
  unsigned int k = 0;
  while (k < dim) {
    unsigned int span = w + k;

    if (k < half_) { // upper half
      if (half_ - k >= 8) {
        float16x8_t values0_7 = vld1q_f16(&in[span]);
        float16x8_t transformed_values0_7 =
          vmulq_n_f16(vld1q_f16(&in[span + half_]), -1);
        float32x4_t cos0_3 = vld1q_f32(&cos_[k]);
        float32x4_t cos4_7 = vld1q_f32(&cos_[k + 4]);
        float32x4_t sin0_3 = vld1q_f32(&sin_[k]);
        float32x4_t sin4_7 = vld1q_f32(&sin_[k + 4]);

        float32x4_t values0_3 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_low_f16(values0_7)), cos0_3),
          vmulq_f32(vcvt_f32_f16(vget_low_f16(transformed_values0_7)), sin0_3));
        float32x4_t values4_7 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_high_f16(values0_7)), cos4_7),
          vmulq_f32(vcvt_f32_f16(vget_high_f16(transformed_values0_7)),
                    sin4_7));

        vst1q_f16(&out[span], vcombine_f16(vcvt_f16_f32(values0_3),
                                           vcvt_f16_f32(values4_7)));

        k += 8;
      } else {
        float value = in[span];
        float transformed_value = -1 * in[span + half_];

        value = (value * cos_[k]) + (transformed_value * sin_[k]);

        out[span] = value;

        ++k;
      }
    } else { // lower half : k >= half_
      if (dim - k >= 8) {
        float16x8_t values0_7 = vld1q_f16(&in[span]);
        float16x8_t transformed_values0_7 = vld1q_f16(&in[span - half_]);
        float32x4_t cos0_3 = vld1q_f32(&cos_[k]);
        float32x4_t cos4_7 = vld1q_f32(&cos_[k + 4]);
        float32x4_t sin0_3 = vld1q_f32(&sin_[k]);
        float32x4_t sin4_7 = vld1q_f32(&sin_[k + 4]);

        float32x4_t values0_3 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_low_f16(values0_7)), cos0_3),
          vmulq_f32(vcvt_f32_f16(vget_low_f16(transformed_values0_7)), sin0_3));
        float32x4_t values4_7 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_high_f16(values0_7)), cos4_7),
          vmulq_f32(vcvt_f32_f16(vget_high_f16(transformed_values0_7)),
                    sin4_7));

        vst1q_f16(&out[span], vcombine_f16(vcvt_f16_f32(values0_3),
                                           vcvt_f16_f32(values4_7)));

        k += 8;
      } else {
        float value = in[span];
        float transformed_value = in[span - half_];

        value = (value * cos_[k]) + (transformed_value * sin_[k]);

        out[span] = value;

        ++k;
      }
    }
  }
}
#endif

} // namespace nntrainer::neon
