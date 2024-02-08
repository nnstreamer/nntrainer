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

void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int from) {
  cosine(N_half, angle, cos_, from);
  sine(N_half, angle, sin_, from);

  unsigned int N = 2 * N_half;
  unsigned int i = N_half;
  unsigned int i_half = 0;

  for (; (N - i >= VL_FP32) && (N_half - i_half >= VL_FP32);
       i += VL_FP32, i_half += VL_FP32) {
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

void swish(const unsigned int N, float *X, float *Y, float *Z) {
  unsigned int i = 0;
  for (; N - i >= VL_FP32; i += VL_FP32) {
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    float32x4_t z0_3 = vld1q_f32(&Z[i]);
    float32x4_t y0_3_minus = vmulq_n_f32(y0_3, -1);
    float32x4_t exp0_3 = exp_ps(y0_3_minus);

    exp0_3 = vaddq_f32(exp0_3, vmovq_n_f32(1.f));
    exp0_3 = vdivq_f32(y0_3, exp0_3);
    exp0_3 = vmulq_f32(exp0_3, z0_3);

    vst1q_f32(&X[i], exp0_3);
  }
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(static_cast<float>(-Y[i])))) * Z[i];
    ++i;
  }
}

#ifdef ENABLE_FP16
void compute_rotary_embedding_value(unsigned int dim, unsigned int half_,
                                    unsigned int w, __fp16 *in, __fp16 *out,
                                    float *cos_, float *sin_) {
  unsigned int k = 0;
  while (k < dim) {
    unsigned int span = w + k;

    if (k < half_) { // upper half
      if (half_ - k >= VL_FP16) {
        float16x8_t values0_7 = vld1q_f16(&in[span]);
        float16x8_t transformed_values0_7 =
          vmulq_n_f16(vld1q_f16(&in[span + half_]), -1);
        float32x4_t cos0_3 = vld1q_f32(&cos_[k]);
        float32x4_t cos4_7 = vld1q_f32(&cos_[k + VL_FP32]);
        float32x4_t sin0_3 = vld1q_f32(&sin_[k]);
        float32x4_t sin4_7 = vld1q_f32(&sin_[k + VL_FP32]);

        float32x4_t values0_3 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_low_f16(values0_7)), cos0_3),
          vmulq_f32(vcvt_f32_f16(vget_low_f16(transformed_values0_7)), sin0_3));
        float32x4_t values4_7 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_high_f16(values0_7)), cos4_7),
          vmulq_f32(vcvt_f32_f16(vget_high_f16(transformed_values0_7)),
                    sin4_7));

        vst1q_f16(&out[span], vcombine_f16(vcvt_f16_f32(values0_3),
                                           vcvt_f16_f32(values4_7)));

        k += VL_FP16;
      } else {
        float value = in[span];
        float transformed_value = -1 * in[span + half_];

        value = (value * cos_[k]) + (transformed_value * sin_[k]);

        out[span] = value;

        ++k;
      }
    } else { // lower half : k >= half_
      if (dim - k >= VL_FP16) {
        float16x8_t values0_7 = vld1q_f16(&in[span]);
        float16x8_t transformed_values0_7 = vld1q_f16(&in[span - half_]);
        float32x4_t cos0_3 = vld1q_f32(&cos_[k]);
        float32x4_t cos4_7 = vld1q_f32(&cos_[k + VL_FP32]);
        float32x4_t sin0_3 = vld1q_f32(&sin_[k]);
        float32x4_t sin4_7 = vld1q_f32(&sin_[k + VL_FP32]);

        float32x4_t values0_3 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_low_f16(values0_7)), cos0_3),
          vmulq_f32(vcvt_f32_f16(vget_low_f16(transformed_values0_7)), sin0_3));
        float32x4_t values4_7 = vaddq_f32(
          vmulq_f32(vcvt_f32_f16(vget_high_f16(values0_7)), cos4_7),
          vmulq_f32(vcvt_f32_f16(vget_high_f16(transformed_values0_7)),
                    sin4_7));

        vst1q_f16(&out[span], vcombine_f16(vcvt_f16_f32(values0_3),
                                           vcvt_f16_f32(values4_7)));

        k += VL_FP16;
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

void swish(const unsigned int N, __fp16 *X, __fp16 *Y, __fp16 *Z) {
  unsigned int i = 0;
  for (; N - i >= VL_FP16; i += VL_FP16) {
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    float16x8_t z0_7 = vld1q_f16(&Z[i]);
    float16x8_t y0_7_minus = vmulq_n_f16(y0_7, -1);

    float32x4_t exp0_3 = exp_ps(vcvt_f32_f16(vget_low_f16(y0_7_minus)));
    float32x4_t exp4_7 = exp_ps(vcvt_f32_f16(vget_high_f16(y0_7_minus)));

    float16x8_t exp0_7 =
      vcombine_f16(vcvt_f16_f32(exp0_3), vcvt_f16_f32(exp4_7));
    exp0_7 = vaddq_f16(exp0_7, vmovq_n_f16(1.f));
    exp0_7 = vdivq_f16(y0_7, exp0_7);
    exp0_7 = vmulq_f16(exp0_7, z0_7);

    vst1q_f16(&X[i], exp0_7);
  }
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(static_cast<float>(-Y[i])))) * Z[i];
    ++i;
  }
}
#endif

} // namespace nntrainer::neon
