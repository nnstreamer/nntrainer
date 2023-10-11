// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_neon.cpp
 * @date   4 Aug 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Source for blas neon implementation
 *
 */

#include <blas_neon.h>
#include <nntrainer_error.h>
namespace nntrainer::neon {

void sgemv_neon(const float *A, const float *X, float *Y, uint32_t rows,
                uint32_t cols, float alpha, float beta) {
  const float *__restrict x;

  for (unsigned int i = 0; i < rows; ++i) {
    Y[i] = Y[i] * beta;
  }

  float32x4_t v_alpha = vmovq_n_f32(alpha);

  if (cols % 16 == 0) {
    for (unsigned i = 0; i < cols; i += 16) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);
      float32x4_t x8_11 = vld1q_f32(&X[i + 8]);
      float32x4_t x12_15 = vld1q_f32(&X[i + 12]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
        x8_11 = vmulq_f32(x8_11, v_alpha);
        x12_15 = vmulq_f32(x12_15, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);
        y0 = vmlaq_f32(y0, wvec8_11, x8_11);
        y0 = vmlaq_f32(y0, wvec12_15, x12_15);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }

  } else if (cols % 8 == 0) {
    for (unsigned i = 0; i < cols; i += 8) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  } else if (cols % 4 == 0) {
    for (unsigned i = 0; i < cols; i += 4) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  }
}

void sgemv_transpose_neon(const float *A, const float *X, float *Y,
                          uint32_t rows, uint32_t cols, float alpha,
                          float beta) {
  const float *__restrict x;

  const float32x4_t v_beta = vdupq_n_f32(beta);
  const float32x4_t v_alpha = vdupq_n_f32(alpha);

  if (cols % 16 == 0) {
    unsigned int n = cols / 16;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);
    unsigned int step;
    for (unsigned int i = 0; i < cols / 16; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 16) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);
        float32x4_t y8_11 = vld1q_f32(&y[8]);
        float32x4_t y12_15 = vld1q_f32(&y[12]);
        step = j / 16;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          y8_11 = vmulq_f32(y8_11, v_beta);
          y12_15 = vmulq_f32(y12_15, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        y8_11 = vmlaq_f32(y8_11, wvec8_11, x);
        y12_15 = vmlaq_f32(y12_15, wvec12_15, x);

        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
        vst1q_f32(&y[8], y8_11);
        vst1q_f32(&y[12], y12_15);
      }
    }
    free(initialized);
    return;
  } else if (cols % 8 == 0) {
    unsigned int n = cols / 8;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);
    unsigned int step;
    for (unsigned int i = 0; i < cols / 8; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 8) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);

        step = j / 8;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
      }
    }
    free(initialized);
    return;
  } else if (cols % 4 == 0) {
    unsigned int n = cols / 4;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);

    unsigned int step;
    for (unsigned int i = 0; i < cols / 4; ++i) {
      initialized[i] = false;
    }
    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 4) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        step = j / 4;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        vst1q_f32(&y[0], y0_3);
      }
    }
    free(initialized);
  }

  return;
}

#ifdef ENABLE_FP16

void sgemv_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t rows,
                     uint32_t cols, float alpha, float beta) {
  const __fp16 *__restrict x;
  const float32x4_t v_beta_32 = vmovq_n_f32(beta);
  float Y32[rows];

  unsigned int idx = 0;

  for (; rows - idx >= 8; idx += 8) {
    float32x4_t y0_3 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));
    y0_3 = vmulq_f32(y0_3, v_beta_32);
    y4_7 = vmulq_f32(y4_7, v_beta_32);

    vst1q_f32(&Y32[idx], y0_3);
    vst1q_f32(&Y32[idx + 4], y4_7);
  }
  for (; rows - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_f32(y0_3_32, v_beta_32);

    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; idx < rows; ++idx) {
    Y32[idx] = beta * Y[idx];
  }

  idx = 0;
  for (; cols - idx >= 64; idx += 64) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);
    float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
    float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
    float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

    float16x8_t x32_39 = vld1q_f16(&X[idx + 32]);
    float16x8_t x40_47 = vld1q_f16(&X[idx + 40]);
    float16x8_t x48_55 = vld1q_f16(&X[idx + 48]);
    float16x8_t x56_63 = vld1q_f16(&X[idx + 56]);

    if (alpha != 1.0) {
      x0_7 = vmulq_n_f16(x0_7, alpha);
      x8_15 = vmulq_n_f16(x8_15, alpha);
      x16_23 = vmulq_n_f16(x16_23, alpha);
      x24_31 = vmulq_n_f16(x24_31, alpha);
      x32_39 = vmulq_n_f16(x32_39, alpha);
      x40_47 = vmulq_n_f16(x40_47, alpha);
      x48_55 = vmulq_n_f16(x48_55, alpha);
      x56_63 = vmulq_n_f16(x56_63, alpha);
    }

    const __fp16 *__restrict w;

    float yVal_low;
    float yVal_high;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float16x8_t wvec0_7 = vld1q_f16(&w[0]);
      float16x8_t wvec8_15 = vld1q_f16(&w[8]);
      float16x8_t wvec16_23 = vld1q_f16(&w[16]);
      float16x8_t wvec24_31 = vld1q_f16(&w[24]);

      float16x8_t wvec32_39 = vld1q_f16(&w[32]);
      float16x8_t wvec40_47 = vld1q_f16(&w[40]);
      float16x8_t wvec48_55 = vld1q_f16(&w[48]);
      float16x8_t wvec56_63 = vld1q_f16(&w[56]);

      float16x8_t y = vmulq_f16(wvec0_7, x0_7);
      y = vfmaq_f16(y, wvec8_15, x8_15);
      y = vfmaq_f16(y, wvec16_23, x16_23);
      y = vfmaq_f16(y, wvec24_31, x24_31);

      y = vfmaq_f16(y, wvec32_39, x32_39);
      y = vfmaq_f16(y, wvec40_47, x40_47);
      y = vfmaq_f16(y, wvec48_55, x48_55);
      y = vfmaq_f16(y, wvec56_63, x56_63);

      yVal_low = vaddvq_f32(vcvt_f32_f16(vget_low_f16(y)));
      yVal_high = vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));

      Y32[j] += yVal_low + yVal_high;
    }
  }
  for (; cols - idx >= 32; idx += 32) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);
    float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
    float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
    float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

    if (alpha != 1.0) {
      x0_7 = vmulq_n_f16(x0_7, alpha);
      x8_15 = vmulq_n_f16(x8_15, alpha);
      x16_23 = vmulq_n_f16(x16_23, alpha);
      x24_31 = vmulq_n_f16(x24_31, alpha);
    }

    const __fp16 *__restrict w;

    float yVal_low;
    float yVal_high;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float16x8_t wvec0_7 = vld1q_f16(&w[0]);
      float16x8_t wvec8_15 = vld1q_f16(&w[8]);
      float16x8_t wvec16_23 = vld1q_f16(&w[16]);
      float16x8_t wvec24_31 = vld1q_f16(&w[24]);

      float16x8_t y = vmulq_f16(wvec0_7, x0_7);
      y = vfmaq_f16(y, wvec8_15, x8_15);
      y = vfmaq_f16(y, wvec16_23, x16_23);
      y = vfmaq_f16(y, wvec24_31, x24_31);

      yVal_low = vaddvq_f32(vcvt_f32_f16(vget_low_f16(y)));
      yVal_high = vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));

      Y32[j] += yVal_low + yVal_high;
    }
  }
  for (; cols - idx >= 16; idx += 16) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);
    float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);

    if (alpha != 1.0) {
      x0_7 = vmulq_n_f16(x0_7, alpha);
      x8_15 = vmulq_n_f16(x8_15, alpha);
    }

    const __fp16 *__restrict w;
    float yVal_low;
    float yVal_high;
    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float16x8_t wvec0_7 = vld1q_f16(&w[0]);
      float16x8_t wvec8_15 = vld1q_f16(&w[8]);

      float16x8_t y = vmulq_f16(wvec0_7, x0_7);
      y = vfmaq_f16(y, wvec8_15, x8_15);

      yVal_low = vaddvq_f32(vcvt_f32_f16(vget_low_f16(y)));
      yVal_high = vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));

      Y32[j] += yVal_low + yVal_high;
    }
  }
  for (; cols - idx >= 8; idx += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);

    if (alpha != 1.0) {
      x0_7 = vmulq_n_f16(x0_7, alpha);
    }

    const __fp16 *__restrict w;

    float yVal_low;
    float yVal_high;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float16x8_t wvec0_7 = vld1q_f16(&w[0]);
      float16x8_t y = vmulq_f16(wvec0_7, x0_7);

      yVal_low = vaddvq_f32(vcvt_f32_f16(vget_low_f16(y)));
      yVal_high = vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));

      Y32[j] += yVal_low + yVal_high;
    }
  }
  for (; cols - idx >= 4; idx += 4) {
    float16x4_t x0_3 = vld1_f16(&X[idx]);

    if (alpha != 1.0) {
      x0_3 = vmul_n_f16(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float16x4_t wvec0_3 = (vld1_f16(&w[0]));
      float16x4_t y0 = vmul_f16(wvec0_3, x0_3);

      Y32[j] += vaddvq_f32(vcvt_f32_f16(y0));
    }
  }

  // now, cols - idx is under 4 : 0 1 2 3 = cols - idx
  if (cols != idx) {
    float16x4_t x0_3 = vld1_f16(&X[idx]);
    for (int j = cols - idx; j < 4; ++j) {
      x0_3[j] = 0;
    }

    if (alpha != 1.0) {
      x0_3 = vmul_n_f16(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    __fp16 yVal;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float16x4_t wvec0_3 = vld1_f16(&w[0]);

      for (int k = cols - idx; k < 4; ++k) {
        wvec0_3[k] = 0;
      }

      float16x4_t y0 = vmul_f16(wvec0_3, x0_3);

      for (int k = 0; k < cols - idx; ++k) {
        Y32[j] += y0[k];
      }
    }
  }

  scopy_neon_fp32_to_fp16(rows, Y32, Y);
  return;
}

void sgemv_transpose_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y,
                               uint32_t rows, uint32_t cols, float alpha,
                               float beta) {
  float Y32[cols];
  const float32x4_t v_beta_32 = vmovq_n_f32(beta);

  unsigned int idx = 0;

  for (; cols - idx >= 8; idx += 8) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7_32 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));

    y0_3_32 = vmulq_f32(y0_3_32, v_beta_32);
    y4_7_32 = vmulq_f32(y4_7_32, v_beta_32);

    vst1q_f32(&Y32[idx], y0_3_32);
    vst1q_f32(&Y32[idx + 4], y4_7_32);
  }
  for (; cols - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_f32(y0_3_32, v_beta_32);
    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; idx < cols; ++idx) {
    Y32[idx] = beta * Y[idx];
  }
  if (rows % 16 == 0) {
    for (unsigned int i = 0; i < rows; i += 16) {
      float x = alpha * static_cast<float>(X[i]);
      float x2 = alpha * static_cast<float>(X[i + 1]);
      float x3 = alpha * static_cast<float>(X[i + 2]);
      float x4 = alpha * static_cast<float>(X[i + 3]);
      float x5 = alpha * static_cast<float>(X[i + 4]);
      float x6 = alpha * static_cast<float>(X[i + 5]);
      float x7 = alpha * static_cast<float>(X[i + 6]);
      float x8 = alpha * static_cast<float>(X[i + 7]);
      float x9 = alpha * static_cast<float>(X[i + 8]);
      float x10 = alpha * static_cast<float>(X[i + 9]);
      float x11 = alpha * static_cast<float>(X[i + 10]);
      float x12 = alpha * static_cast<float>(X[i + 11]);
      float x13 = alpha * static_cast<float>(X[i + 12]);
      float x14 = alpha * static_cast<float>(X[i + 13]);
      float x15 = alpha * static_cast<float>(X[i + 14]);
      float x16 = alpha * static_cast<float>(X[i + 15]);

      idx = 0;

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        const __fp16 *__restrict w;
        const __fp16 *__restrict w2;
        const __fp16 *__restrict w3;
        const __fp16 *__restrict w4;
        const __fp16 *__restrict w5;
        const __fp16 *__restrict w6;
        const __fp16 *__restrict w7;
        const __fp16 *__restrict w8;
        const __fp16 *__restrict w9;
        const __fp16 *__restrict w10;
        const __fp16 *__restrict w11;
        const __fp16 *__restrict w12;
        const __fp16 *__restrict w13;
        const __fp16 *__restrict w14;
        const __fp16 *__restrict w15;
        const __fp16 *__restrict w16;

        w = &A[i * cols + idx];
        w2 = &A[(i + 1) * cols + idx];
        w3 = &A[(i + 2) * cols + idx];
        w4 = &A[(i + 3) * cols + idx];
        w5 = &A[(i + 4) * cols + idx];
        w6 = &A[(i + 5) * cols + idx];
        w7 = &A[(i + 6) * cols + idx];
        w8 = &A[(i + 7) * cols + idx];
        w9 = &A[(i + 8) * cols + idx];
        w10 = &A[(i + 9) * cols + idx];
        w11 = &A[(i + 10) * cols + idx];
        w12 = &A[(i + 11) * cols + idx];
        w13 = &A[(i + 12) * cols + idx];
        w14 = &A[(i + 13) * cols + idx];
        w15 = &A[(i + 14) * cols + idx];
        w16 = &A[(i + 15) * cols + idx];

        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w[0])), x);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w2[0])), x2);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w3[0])), x3);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w4[0])), x4);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w5[0])), x5);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w6[0])), x6);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w7[0])), x7);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w8[0])), x8);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w9[0])), x9);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w10[0])), x10);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w11[0])), x11);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w12[0])), x12);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w13[0])), x13);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w14[0])), x14);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w15[0])), x15);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w16[0])), x16);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols != idx) {
        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t wvec0_3_32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));
        float32x4_t w3vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx]));
        float32x4_t w4vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx]));
        float32x4_t w5vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 4) * cols + idx]));
        float32x4_t w6vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 5) * cols + idx]));
        float32x4_t w7vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 6) * cols + idx]));
        float32x4_t w8vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 7) * cols + idx]));
        float32x4_t w9vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 8) * cols + idx]));
        float32x4_t w10vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 9) * cols + idx]));
        float32x4_t w11vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 10) * cols + idx]));
        float32x4_t w12vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 11) * cols + idx]));
        float32x4_t w13vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 12) * cols + idx]));
        float32x4_t w14vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 13) * cols + idx]));
        float32x4_t w15vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 14) * cols + idx]));
        float32x4_t w16vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 15) * cols + idx]));

        for (int j = cols - idx; j < 4; ++j) {
          y0_3[j] = 0;
          wvec0_3_32[j] = 0;
          w2vec0_3_32[j] = 0;
          w3vec0_3_32[j] = 0;
          w4vec0_3_32[j] = 0;
          w5vec0_3_32[j] = 0;
          w6vec0_3_32[j] = 0;
          w7vec0_3_32[j] = 0;
          w8vec0_3_32[j] = 0;
          w9vec0_3_32[j] = 0;
          w10vec0_3_32[j] = 0;
          w11vec0_3_32[j] = 0;
          w12vec0_3_32[j] = 0;
          w13vec0_3_32[j] = 0;
          w14vec0_3_32[j] = 0;
          w15vec0_3_32[j] = 0;
          w16vec0_3_32[j] = 0;
        }

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_32, x2);
        y0_3 = vfmaq_n_f32(y0_3, w3vec0_3_32, x3);
        y0_3 = vfmaq_n_f32(y0_3, w4vec0_3_32, x4);
        y0_3 = vfmaq_n_f32(y0_3, w5vec0_3_32, x5);
        y0_3 = vfmaq_n_f32(y0_3, w6vec0_3_32, x6);
        y0_3 = vfmaq_n_f32(y0_3, w7vec0_3_32, x7);
        y0_3 = vfmaq_n_f32(y0_3, w8vec0_3_32, x8);

        y0_3 = vfmaq_n_f32(y0_3, w9vec0_3_32, x9);
        y0_3 = vfmaq_n_f32(y0_3, w10vec0_3_32, x10);
        y0_3 = vfmaq_n_f32(y0_3, w11vec0_3_32, x11);
        y0_3 = vfmaq_n_f32(y0_3, w12vec0_3_32, x12);
        y0_3 = vfmaq_n_f32(y0_3, w13vec0_3_32, x13);
        y0_3 = vfmaq_n_f32(y0_3, w14vec0_3_32, x14);
        y0_3 = vfmaq_n_f32(y0_3, w15vec0_3_32, x15);
        y0_3 = vfmaq_n_f32(y0_3, w16vec0_3_32, x16);

        vst1q_f32(&Y32[idx], y0_3);
      }
    }
  } else if (rows % 8 == 0) {
    for (unsigned int i = 0; i < rows; i += 8) {
      float x = alpha * static_cast<float>(X[i]);
      float x2 = alpha * static_cast<float>(X[i + 1]);
      float x3 = alpha * static_cast<float>(X[i + 2]);
      float x4 = alpha * static_cast<float>(X[i + 3]);
      float x5 = alpha * static_cast<float>(X[i + 4]);
      float x6 = alpha * static_cast<float>(X[i + 5]);
      float x7 = alpha * static_cast<float>(X[i + 6]);
      float x8 = alpha * static_cast<float>(X[i + 7]);

      idx = 0;

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        const __fp16 *__restrict w;
        const __fp16 *__restrict w2;
        const __fp16 *__restrict w3;
        const __fp16 *__restrict w4;
        const __fp16 *__restrict w5;
        const __fp16 *__restrict w6;
        const __fp16 *__restrict w7;
        const __fp16 *__restrict w8;

        w = &A[i * cols + idx];
        w2 = &A[(i + 1) * cols + idx];
        w3 = &A[(i + 2) * cols + idx];
        w4 = &A[(i + 3) * cols + idx];
        w5 = &A[(i + 4) * cols + idx];
        w6 = &A[(i + 5) * cols + idx];
        w7 = &A[(i + 6) * cols + idx];
        w8 = &A[(i + 7) * cols + idx];

        float32x4_t w0_3 = vcvt_f32_f16(vld1_f16(&w[0]));
        float32x4_t w4_7 = vcvt_f32_f16(vld1_f16(&w2[0]));
        float32x4_t w8_11 = vcvt_f32_f16(vld1_f16(&w3[0]));
        float32x4_t w12_15 = vcvt_f32_f16(vld1_f16(&w4[0]));
        float32x4_t w16_19 = vcvt_f32_f16(vld1_f16(&w5[0]));
        float32x4_t w20_23 = vcvt_f32_f16(vld1_f16(&w6[0]));
        float32x4_t w24_27 = vcvt_f32_f16(vld1_f16(&w7[0]));
        float32x4_t w28_31 = vcvt_f32_f16(vld1_f16(&w8[0]));

        y0_3 = vfmaq_n_f32(y0_3, w0_3, x);
        y0_3 = vfmaq_n_f32(y0_3, w4_7, x2);
        y0_3 = vfmaq_n_f32(y0_3, w8_11, x3);
        y0_3 = vfmaq_n_f32(y0_3, w12_15, x4);
        y0_3 = vfmaq_n_f32(y0_3, w16_19, x5);
        y0_3 = vfmaq_n_f32(y0_3, w20_23, x6);
        y0_3 = vfmaq_n_f32(y0_3, w24_27, x7);
        y0_3 = vfmaq_n_f32(y0_3, w28_31, x8);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols != idx) {
        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t wvec0_3_32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));
        float32x4_t w3vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx]));
        float32x4_t w4vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx]));
        float32x4_t w5vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 4) * cols + idx]));
        float32x4_t w6vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 5) * cols + idx]));
        float32x4_t w7vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 6) * cols + idx]));
        float32x4_t w8vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 7) * cols + idx]));

        for (int j = cols - idx; j < 4; ++j) {
          y0_3[j] = 0;
          wvec0_3_32[j] = 0;
          w2vec0_3_32[j] = 0;
          w3vec0_3_32[j] = 0;
          w4vec0_3_32[j] = 0;
          w5vec0_3_32[j] = 0;
          w6vec0_3_32[j] = 0;
          w7vec0_3_32[j] = 0;
          w8vec0_3_32[j] = 0;
        }

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_32, x2);
        y0_3 = vfmaq_n_f32(y0_3, w3vec0_3_32, x3);
        y0_3 = vfmaq_n_f32(y0_3, w4vec0_3_32, x4);
        y0_3 = vfmaq_n_f32(y0_3, w5vec0_3_32, x5);
        y0_3 = vfmaq_n_f32(y0_3, w6vec0_3_32, x6);
        y0_3 = vfmaq_n_f32(y0_3, w7vec0_3_32, x7);
        y0_3 = vfmaq_n_f32(y0_3, w8vec0_3_32, x8);

        vst1q_f32(&Y32[idx], y0_3);
      }
    }
  } else if (rows % 4 == 0) {
    for (unsigned int i = 0; i < rows; i += 4) {
      float x = alpha * static_cast<float>(X[i]);
      float x2 = alpha * static_cast<float>(X[i + 1]);
      float x3 = alpha * static_cast<float>(X[i + 2]);
      float x4 = alpha * static_cast<float>(X[i + 3]);

      idx = 0;
      for (; cols - idx >= 16; idx += 16) {
        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t y4_7 = vld1q_f32(&Y32[idx + 4]);
        float32x4_t y8_11 = vld1q_f32(&Y32[idx + 8]);
        float32x4_t y12_15 = vld1q_f32(&Y32[idx + 12]);

        const __fp16 *__restrict w;
        const __fp16 *__restrict w2;
        const __fp16 *__restrict w3;
        const __fp16 *__restrict w4;

        w = &A[i * cols + idx];
        w2 = &A[(i + 1) * cols + idx];
        w3 = &A[(i + 2) * cols + idx];
        w4 = &A[(i + 3) * cols + idx];

        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w[0])), x);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w2[0])), x2);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w3[0])), x3);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&w4[0])), x4);

        y4_7 = vfmaq_n_f32(y4_7, vcvt_f32_f16(vld1_f16(&w[4])), x);
        y4_7 = vfmaq_n_f32(y4_7, vcvt_f32_f16(vld1_f16(&w2[4])), x2);
        y4_7 = vfmaq_n_f32(y4_7, vcvt_f32_f16(vld1_f16(&w3[4])), x3);
        y4_7 = vfmaq_n_f32(y4_7, vcvt_f32_f16(vld1_f16(&w4[4])), x4);

        y8_11 = vfmaq_n_f32(y8_11, vcvt_f32_f16(vld1_f16(&w[8])), x);
        y8_11 = vfmaq_n_f32(y8_11, vcvt_f32_f16(vld1_f16(&w2[8])), x2);
        y8_11 = vfmaq_n_f32(y8_11, vcvt_f32_f16(vld1_f16(&w3[8])), x3);
        y8_11 = vfmaq_n_f32(y8_11, vcvt_f32_f16(vld1_f16(&w4[8])), x4);

        y12_15 = vfmaq_n_f32(y12_15, vcvt_f32_f16(vld1_f16(&w[12])), x);
        y12_15 = vfmaq_n_f32(y12_15, vcvt_f32_f16(vld1_f16(&w2[12])), x2);
        y12_15 = vfmaq_n_f32(y12_15, vcvt_f32_f16(vld1_f16(&w3[12])), x3);
        y12_15 = vfmaq_n_f32(y12_15, vcvt_f32_f16(vld1_f16(&w4[12])), x4);

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
        vst1q_f32(&Y32[idx + 8], y8_11);
        vst1q_f32(&Y32[idx + 12], y12_15);
      }

      for (; cols - idx >= 8; idx += 8) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t y4_7 = vld1q_f32(&Y32[idx + 4]);

        float32x4_t wvec0_3_f32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));
        float32x4_t w3vec0_3_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx]));
        float32x4_t w4vec0_3_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx]));

        float32x4_t wvec3_7_f32 =
          vcvt_f32_f16(vld1_f16(&A[i * cols + idx + 4]));
        float32x4_t w2vec3_7_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx + 4]));
        float32x4_t w3vec3_7_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx + 4]));
        float32x4_t w4vec3_7_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx + 4]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_f32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_f32, x2);
        y0_3 = vfmaq_n_f32(y0_3, w3vec0_3_f32, x3);
        y0_3 = vfmaq_n_f32(y0_3, w4vec0_3_f32, x4);

        y4_7 = vfmaq_n_f32(y4_7, wvec3_7_f32, x);
        y4_7 = vfmaq_n_f32(y4_7, w2vec3_7_f32, x2);
        y4_7 = vfmaq_n_f32(y4_7, w3vec3_7_f32, x3);
        y4_7 = vfmaq_n_f32(y4_7, w4vec3_7_f32, x4);

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
      }

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        float32x4_t wvec0_3_32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));
        float32x4_t w3vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx]));
        float32x4_t w4vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_32, x2);
        y0_3 = vfmaq_n_f32(y0_3, w3vec0_3_32, x3);
        y0_3 = vfmaq_n_f32(y0_3, w4vec0_3_32, x4);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols != idx) {
        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t wvec0_3_32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));
        float32x4_t w3vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx]));
        float32x4_t w4vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx]));

        for (int j = cols - idx; j < 4; ++j) {
          y0_3[j] = 0;
          wvec0_3_32[j] = 0;
          w2vec0_3_32[j] = 0;
          w3vec0_3_32[j] = 0;
          w4vec0_3_32[j] = 0;
        }

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_32, x2);
        y0_3 = vfmaq_n_f32(y0_3, w3vec0_3_32, x3);
        y0_3 = vfmaq_n_f32(y0_3, w4vec0_3_32, x4);

        vst1q_f32(&Y32[idx], y0_3);
      }
    }

  } else if (rows % 2 == 0) {
    for (unsigned int i = 0; i < rows; i += 2) {
      float x = alpha * static_cast<float>(X[i]);
      float x2 = alpha * static_cast<float>(X[i + 1]);

      idx = 0;

      for (; cols - idx >= 8; idx += 8) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t y4_7 = vld1q_f32(&Y32[idx + 4]);

        float32x4_t wvec0_3_f32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));
        float32x4_t wvec4_7_f32 =
          vcvt_f32_f16(vld1_f16(&A[i * cols + idx + 4]));
        float32x4_t w2vec4_7_f32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx + 4]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_f32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_f32, x2);
        y4_7 = vfmaq_n_f32(y4_7, wvec4_7_f32, x);
        y4_7 = vfmaq_n_f32(y4_7, w2vec4_7_f32, x2);

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
      }

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        float32x4_t wvec0_3_32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_32, x2);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols != idx) {
        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t wvec0_3_32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t w2vec0_3_32 =
          vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx]));

        for (int j = cols - idx; j < 4; ++j) {
          y0_3[j] = 0;
          wvec0_3_32[j] = 0;
          w2vec0_3_32[j] = 0;
        }

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);
        y0_3 = vfmaq_n_f32(y0_3, w2vec0_3_32, x2);

        vst1q_f32(&Y32[idx], y0_3);
      }
    }

  } else {
    for (unsigned int i = 0; i < rows; ++i) {
      float x = alpha * static_cast<float>(X[i]);
      idx = 0;

      for (; cols - idx >= 32; idx += 32) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t y4_7 = vld1q_f32(&Y32[idx + 4]);
        float32x4_t y8_11 = vld1q_f32(&Y32[idx + 8]);
        float32x4_t y12_15 = vld1q_f32(&Y32[idx + 12]);
        float32x4_t y16_19 = vld1q_f32(&Y32[idx + 16]);
        float32x4_t y20_23 = vld1q_f32(&Y32[idx + 20]);
        float32x4_t y24_27 = vld1q_f32(&Y32[idx + 24]);
        float32x4_t y28_31 = vld1q_f32(&Y32[idx + 28]);

        const __fp16 *__restrict w;

        w = &A[i * cols + idx];

        float32x4_t wvec0_3_f32 = vcvt_f32_f16(vld1_f16(&w[0]));
        float32x4_t wvec4_7_f32 = vcvt_f32_f16(vld1_f16(&w[4]));

        float32x4_t wvec8_11_f32 = vcvt_f32_f16(vld1_f16(&w[8]));
        float32x4_t wvec12_15_f32 = vcvt_f32_f16(vld1_f16(&w[12]));

        float32x4_t wvec16_19_f32 = vcvt_f32_f16(vld1_f16(&w[16]));
        float32x4_t wvec20_23_f32 = vcvt_f32_f16(vld1_f16(&w[20]));

        float32x4_t wvec24_27_f32 = vcvt_f32_f16(vld1_f16(&w[24]));
        float32x4_t wvec28_31_f32 = vcvt_f32_f16(vld1_f16(&w[28]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_f32, x);
        y4_7 = vfmaq_n_f32(y4_7, wvec4_7_f32, x);

        y8_11 = vfmaq_n_f32(y8_11, wvec8_11_f32, x);
        y12_15 = vfmaq_n_f32(y12_15, wvec12_15_f32, x);

        y16_19 = vfmaq_n_f32(y16_19, wvec16_19_f32, x);
        y20_23 = vfmaq_n_f32(y20_23, wvec20_23_f32, x);

        y24_27 = vfmaq_n_f32(y24_27, wvec24_27_f32, x);
        y28_31 = vfmaq_n_f32(y28_31, wvec28_31_f32, x);

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
        vst1q_f32(&Y32[idx + 8], y8_11);
        vst1q_f32(&Y32[idx + 12], y12_15);
        vst1q_f32(&Y32[idx + 16], y16_19);
        vst1q_f32(&Y32[idx + 20], y20_23);
        vst1q_f32(&Y32[idx + 24], y24_27);
        vst1q_f32(&Y32[idx + 28], y28_31);
      }

      for (; cols - idx >= 16; idx += 16) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t y4_7 = vld1q_f32(&Y32[idx + 4]);
        float32x4_t y8_11 = vld1q_f32(&Y32[idx + 8]);
        float32x4_t y12_15 = vld1q_f32(&Y32[idx + 12]);

        const __fp16 *__restrict w;

        w = &A[i * cols + idx];

        float32x4_t wvec0_3_f32 = vcvt_f32_f16(vld1_f16(&w[0]));
        float32x4_t wvec4_7_f32 = vcvt_f32_f16(vld1_f16(&w[4]));

        float32x4_t wvec8_11_f32 = vcvt_f32_f16(vld1_f16(&w[8]));
        float32x4_t wvec12_15_f32 = vcvt_f32_f16(vld1_f16(&w[12]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_f32, x);
        y4_7 = vfmaq_n_f32(y4_7, wvec4_7_f32, x);

        y8_11 = vfmaq_n_f32(y8_11, wvec8_11_f32, x);
        y12_15 = vfmaq_n_f32(y12_15, wvec12_15_f32, x);

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
        vst1q_f32(&Y32[idx + 8], y8_11);
        vst1q_f32(&Y32[idx + 12], y12_15);
      }

      for (; cols - idx >= 8; idx += 8) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        float32x4_t y4_7 = vld1q_f32(&Y32[idx + 4]);

        float32x4_t wvec0_3_f32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));
        float32x4_t wvec3_7_f32 =
          vcvt_f32_f16(vld1_f16(&A[i * cols + idx + 4]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_f32, x);
        y4_7 = vfmaq_n_f32(y4_7, wvec3_7_f32, x);

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
      }

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        float32x4_t wvec0_3_32 = vcvt_f32_f16(vld1_f16(&A[i * cols + idx]));

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols != idx) {
        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        float16x4_t wvec0_3 = vld1_f16(&A[i * cols + idx]);

        for (int j = cols - idx; j < 4; ++j) {
          y0_3[j] = 0;
          wvec0_3[j] = 0;
        }

        float32x4_t wvec0_3_32 = vcvt_f32_f16(wvec0_3);

        y0_3 = vfmaq_n_f32(y0_3, wvec0_3_32, x);

        vst1q_f32(&Y32[idx], y0_3);
      }
    }
  }
  scopy_neon_fp32_to_fp16(cols, Y32, Y);
  return;
}

void saxpy_neon_fp16(const unsigned int N, const float alpha, const __fp16 *X,
                     __fp16 *Y) {

  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x8_t mulacc = vfmaq_f16(y, v_alphaX8, x);
    vst1q_f16(&Y[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float16x4_t y = vld1_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x4_t mulacc = vfma_f16(y, v_alphaX4, x);
    vst1_f16(&Y[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = Y[idx] + alpha * X[idx];
}

__fp16 sdot_neon_fp16(const unsigned int N, const __fp16 *X, const __fp16 *Y) {

  float16x8_t accX8 = vmovq_n_f16(0);
  float16x4_t accX4 = vmov_n_f16(0);

  unsigned int idx = 0;
  __fp16 ret = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    // x*y + accX8 -> accX8
    accX8 = vfmaq_f16(accX8, x, y);
  }

  // check at least one batch of 8 is processed
  if (N - 8 >= 0) {
    __fp16 result[8];
    vst1q_f16(result, accX8);
    for (unsigned int i = 0; i < 8; i++)
      ret += result[i];
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float16x4_t y = vld1_f16(&Y[idx]);

    // x*y + accX4 -> accX4
    accX4 = vfma_f16(accX4, x, y);
  }

  // check at least one batch of 4 is processed
  if (N % 8 >= 4) {
    __fp16 result[4];
    vst1_f16(result, accX4);
    ret += result[0] + result[1] + result[2] + result[3];
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    ret += X[idx] * Y[idx];

  return ret;
}

__fp16 snrm2_neon_fp16(const unsigned int N, const __fp16 *X) {

  float16x8_t accX8 = vmovq_n_f16(0);
  float16x4_t accX4 = vmov_n_f16(0);

  unsigned int idx = 0;
  __fp16 ret = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);

    // x*x + accX8 -> accX8
    accX8 = vfmaq_f16(accX8, x, x);
  }

  // check at least one batch of 8 is processed
  if (N - 8 >= 0) {
    __fp16 result[8];
    vst1q_f16(result, accX8);
    for (unsigned int i = 0; i < 8; i++)
      ret += result[i];
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);

    // x*x + accX4 -> accX4
    accX4 = vfma_f16(accX4, x, x);
  }

  // check at least one batch of 4 is processed
  if (N % 8 >= 4) {
    __fp16 result[4];
    vst1_f16(result, accX4);
    ret += result[0] + result[1] + result[2] + result[3];
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    ret += X[idx] * X[idx];

  return ret;
}

void sscal_neon_fp16(const unsigned int N, __fp16 *X, const float alpha) {
  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);

    // alpha*X -> X
    float16x8_t mulacc = vmulq_f16(v_alphaX8, x);
    vst1q_f16(&X[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);

    // alpha*X -> X
    float16x4_t mulacc = vmul_f16(v_alphaX4, x);
    vst1_f16(&X[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    X[idx] = alpha * X[idx];
}

void scopy_neon_fp16(const unsigned int N, const __fp16 *X, __fp16 *Y) {

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t batch = vld1q_f16(&X[idx]);
    vst1q_f16(&Y[idx], batch);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t batch = vld1_f16(&X[idx]);
    vst1_f16(&Y[idx], batch);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = X[idx];
}

void scopy_neon_int4_to_fp16(const unsigned int N, const uint8_t *X,
                             __fp16 *Y) {

  unsigned int idx = 0;

  // keep in mind that : len(X) = N, and len(Y) = 2*N

  // processing batch of 16

  float16x8_t y0, y1, y2, y3;
  float16x4_t yh0, yh1;

  uint8_t low0, low1, high0, high1;

  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);

    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    for (int i = 0; i < 8; ++i) {
      low0 = low[i] >> 4;
      low1 = low[i] & 0x0f;

      high0 = high[i] >> 4;
      high1 = high[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }

      if (i < 4) {
        y2[2 * i] = high0;
        y2[2 * i + 1] = high1;
      } else {
        y3[2 * (i - 4)] = high0;
        y3[2 * (i - 4) + 1] = high1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
    vst1q_f16(&Y[2 * idx + 16], y2);
    vst1q_f16(&Y[2 * idx + 24], y3);
  }

  // processing remaining batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    int8x8_t batch = vld1_u8(&X[idx]);

    for (int i = 0; i < 8; ++i) {
      low0 = batch[i] >> 4;
      low1 = batch[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
  }

  // pocessing remaining values
  for (; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void scopy_neon_fp16_to_fp32(const unsigned int N, const __fp16 *X, float *Y) {
  int idx = 0;

  for (; N - idx >= 8; idx += 8) {
    float32x4_t y1 = vcvt_f32_f16(vld1_f16(&X[idx]));
    float32x4_t y2 = vcvt_f32_f16(vld1_f16(&X[idx + 4]));

    vst1q_f32(&Y[idx], y1);
    vst1q_f32(&Y[idx + 4], y2);
  }

  for (; N - idx >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float32x4_t y1 = vcvt_f32_f16(x);
    vst1q_f32(&Y[idx], y1);
  }

  for (; idx < N; ++idx) {
    Y[idx] = static_cast<float>(X[idx]);
  }
}

void scopy_neon_fp32_to_fp16(const unsigned int N, const float *X, __fp16 *Y) {
  int idx = 0;

  for (; N - idx >= 8; idx += 8) {
    float32x4_t x1 = vld1q_f32(&X[idx]);
    float32x4_t x2 = vld1q_f32(&X[idx + 4]);

    float16x8_t y1 = vcombine_f16(vcvt_f16_f32(x1), vcvt_f16_f32(x2));

    vst1q_f16(&Y[idx], y1);
  }

  for (; N - idx >= 4; idx += 4) {
    float32x4_t x1 = vld1q_f32(&X[idx]);

    float16x4_t y1 = vcvt_f16_f32(x1);

    vst1_f16(&Y[idx], y1);
  }

  for (; idx < N; ++idx) {
    Y[idx] = static_cast<__fp16>(X[idx]);
  }
}

unsigned int isamax_neon_fp16(const unsigned int N, const __fp16 *X) {

  unsigned int retIdx;
  __fp16 maxNum;

  uint16_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16x8_t stride = vmovq_n_u16(8);
  float16x8_t batch = vld1q_f16(&X[0]);
  uint16x8_t curr_index = vld1q_u16(indices);
  uint16x8_t max_index = curr_index;

  unsigned int idx = 8;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t values = vld1q_f16(&X[idx]);
    curr_index = vaddq_u16(curr_index, stride);

    // comparison
    uint16x8_t mask = vcgtq_f16(batch, values);

    // blend values and indices based on the mask
    batch = vbslq_f16(mask, batch, values);
    max_index = vbslq_u16(mask, max_index, curr_index);
  }

  // storing indices and max values
  __fp16 maxVal[8];
  vst1q_f16(maxVal, batch);
  vst1q_u16(indices, max_index);

  // getting the index of the maxima
  maxNum = maxVal[0];
  retIdx = max_index[0];
  for (int i = 1; i < 8; i++) {
    if (maxVal[i] > maxNum) {
      maxNum = maxVal[i];
      retIdx = max_index[i];
    }
  }

  // processing remaining values
  for (; idx < N; idx++) {
    if (X[idx] > maxNum) {
      maxNum = X[idx];
      retIdx = idx;
    }
  }

  return retIdx;
}

void sgemm_neon_fp16(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                     uint32_t N, uint32_t K, float alpha, float beta,
                     bool TransA, bool TransB) {

  // dynamic creation to avoid reaching stack limit(causes segmentation fault)
  float *C32 = (float *)malloc(M * N * sizeof(float));

  // performing beta*C
  unsigned int idx = 0;
  unsigned int size = M * N;
  for (; idx < (size - idx) >= 8; idx += 8) {
    float16x8_t c = vmulq_n_f16(vld1q_f16(&C[idx]), static_cast<__fp16>(beta));

    vst1q_f32(&C32[idx], vcvt_f32_f16(vget_low_f16(c)));
    vst1q_f32(&C32[idx + 4], vcvt_f32_f16(vget_high_f16(c)));
  }
  // remaining 4
  for (; idx < (size - idx) >= 4; idx += 4) {
    float16x4_t c = vmul_n_f16(vld1_f16(&C[idx]), static_cast<__fp16>(beta));

    vst1q_f32(&C32[idx], vcvt_f32_f16(c));
  }

  // remaining values if dimensions not a multiple of 8
  for (; idx < size; idx++) {
    C32[idx] = C[idx] * beta;
  }

  if (!TransA && TransB) {
    sgemm_neon_fp16_transB(A, B, C32, M, N, K, alpha, beta);
  } else if (TransA && !TransB) {
    sgemm_neon_fp16_transA(A, B, C32, M, N, K, alpha, beta);
  } else if (!TransA && !TransB) {
    sgemm_neon_fp16_noTrans(A, B, C32, M, N, K, alpha, beta);
  } else { // TransA && TransB
    sgemm_neon_fp16_transAB(A, B, C32, M, N, K, alpha, beta, idx);
  }

  scopy_neon_fp32_to_fp16(M * N, C32, C);
  free(C32);
}

void sgemm_neon_fp16_noTrans(const __fp16 *A, const __fp16 *B, float *C,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             float beta) {

  unsigned int k = 0, n = 0;
  __fp16 a[16];
  for (; (K - k) >= 16; k += 16) {
    for (unsigned int m = 0; m < M; m++) {
      // calculating A * alpha;
      vst1q_f16(a, vmulq_n_f16(vld1q_f16(&A[m * K + k]), alpha));
      vst1q_f16(&a[8], vmulq_n_f16(vld1q_f16(&A[m * K + k + 8]), alpha));

      for (n = 0; (N - n) >= 8; n += 8) {

        // fp16 multiplications and partial accumulations
        float16x8_t b0_7_0 = vmulq_n_f16(vld1q_f16(&B[k * N + n]), a[0]);
        b0_7_0 = vfmaq_n_f16(b0_7_0, vld1q_f16(&B[(k + 1) * N + n]), a[1]);
        b0_7_0 = vfmaq_n_f16(b0_7_0, vld1q_f16(&B[(k + 2) * N + n]), a[2]);
        b0_7_0 = vfmaq_n_f16(b0_7_0, vld1q_f16(&B[(k + 3) * N + n]), a[3]);
        float16x8_t b0_7_4 = vmulq_n_f16(vld1q_f16(&B[(k + 4) * N + n]), a[4]);
        b0_7_4 = vfmaq_n_f16(b0_7_4, vld1q_f16(&B[(k + 5) * N + n]), a[5]);
        b0_7_4 = vfmaq_n_f16(b0_7_4, vld1q_f16(&B[(k + 6) * N + n]), a[6]);
        b0_7_4 = vfmaq_n_f16(b0_7_4, vld1q_f16(&B[(k + 7) * N + n]), a[7]);
        float16x8_t b0_7_8 = vmulq_n_f16(vld1q_f16(&B[(k + 8) * N + n]), a[8]);
        b0_7_8 = vfmaq_n_f16(b0_7_8, vld1q_f16(&B[(k + 9) * N + n]), a[9]);
        b0_7_8 = vfmaq_n_f16(b0_7_8, vld1q_f16(&B[(k + 10) * N + n]), a[10]);
        b0_7_8 = vfmaq_n_f16(b0_7_8, vld1q_f16(&B[(k + 11) * N + n]), a[11]);
        float16x8_t b0_7_12 =
          vmulq_n_f16(vld1q_f16(&B[(k + 12) * N + n]), a[12]);
        b0_7_12 = vfmaq_n_f16(b0_7_12, vld1q_f16(&B[(k + 13) * N + n]), a[13]);
        b0_7_12 = vfmaq_n_f16(b0_7_12, vld1q_f16(&B[(k + 14) * N + n]), a[14]);
        b0_7_12 = vfmaq_n_f16(b0_7_12, vld1q_f16(&B[(k + 15) * N + n]), a[15]);

        float32x4_t c0_7_low_32 = vaddq_f32(vld1q_f32(&C[m * N + n]),
                                            vcvt_f32_f16(vget_low_f16(b0_7_0)));
        float32x4_t c0_7_high_32 = vaddq_f32(
          vld1q_f32(&C[m * N + n + 4]), vcvt_f32_f16(vget_high_f16(b0_7_0)));

        // fp32 partial accumulations
        c0_7_low_32 =
          vaddq_f32(c0_7_low_32, vcvt_f32_f16(vget_low_f16(b0_7_4)));
        c0_7_high_32 =
          vaddq_f32(c0_7_high_32, vcvt_f32_f16(vget_high_f16(b0_7_4)));
        c0_7_low_32 =
          vaddq_f32(c0_7_low_32, vcvt_f32_f16(vget_low_f16(b0_7_8)));
        c0_7_high_32 =
          vaddq_f32(c0_7_high_32, vcvt_f32_f16(vget_high_f16(b0_7_8)));
        c0_7_low_32 =
          vaddq_f32(c0_7_low_32, vcvt_f32_f16(vget_low_f16(b0_7_12)));
        c0_7_high_32 =
          vaddq_f32(c0_7_high_32, vcvt_f32_f16(vget_high_f16(b0_7_12)));

        vst1q_f32(&C[m * N + n], c0_7_low_32);
        vst1q_f32(&C[m * N + n + 4], c0_7_high_32);
      }
    }
  }

  for (; (K - k) >= 8; k += 8) {
    for (unsigned int m = 0; m < M; m++) {
      // calculating A * alpha;
      vst1q_f16(a, vmulq_n_f16(vld1q_f16(&A[m * K + k]), alpha));

      for (n = 0; (N - n) >= 8; n += 8) {

        // fp16 multiplications and partial accumulations
        float16x8_t b0_7_0 = vmulq_n_f16(vld1q_f16(&B[k * N + n]), a[0]);
        b0_7_0 = vfmaq_n_f16(b0_7_0, vld1q_f16(&B[(k + 1) * N + n]), a[1]);
        b0_7_0 = vfmaq_n_f16(b0_7_0, vld1q_f16(&B[(k + 2) * N + n]), a[2]);
        b0_7_0 = vfmaq_n_f16(b0_7_0, vld1q_f16(&B[(k + 3) * N + n]), a[3]);
        float16x8_t b0_7_4 = vmulq_n_f16(vld1q_f16(&B[(k + 4) * N + n]), a[4]);
        b0_7_4 = vfmaq_n_f16(b0_7_4, vld1q_f16(&B[(k + 5) * N + n]), a[5]);
        b0_7_4 = vfmaq_n_f16(b0_7_4, vld1q_f16(&B[(k + 6) * N + n]), a[6]);
        b0_7_4 = vfmaq_n_f16(b0_7_4, vld1q_f16(&B[(k + 7) * N + n]), a[7]);

        float32x4_t c0_7_low_32 = vaddq_f32(vld1q_f32(&C[m * N + n]),
                                            vcvt_f32_f16(vget_low_f16(b0_7_0)));
        float32x4_t c0_7_high_32 = vaddq_f32(
          vld1q_f32(&C[m * N + n + 4]), vcvt_f32_f16(vget_high_f16(b0_7_0)));

        // fp32 partial accumulations
        c0_7_low_32 =
          vaddq_f32(c0_7_low_32, vcvt_f32_f16(vget_low_f16(b0_7_4)));
        c0_7_high_32 =
          vaddq_f32(c0_7_high_32, vcvt_f32_f16(vget_high_f16(b0_7_4)));

        vst1q_f32(&C[m * N + n], c0_7_low_32);
        vst1q_f32(&C[m * N + n + 4], c0_7_high_32);
      }
    }
  }

  for (; (K - k) >= 4; k += 4) {
    for (unsigned int m = 0; m < M; m++) {
      // calculating A * alpha;
      vst1_f16(a, vmul_n_f16(vld1_f16(&A[m * K + k]), alpha));

      for (n = 0; (N - n) >= 8; n += 8) {

        float16x8_t b0_7_0 = vmulq_n_f16(vld1q_f16(&B[k * N + n]), a[0]);
        b0_7_0 = vfmaq_n_f16(b0_7_0, vld1q_f16(&B[(k + 1) * N + n]), a[1]);
        float16x8_t b0_7_2 = vmulq_n_f16(vld1q_f16(&B[(k + 2) * N + n]), a[2]);
        b0_7_2 = vfmaq_n_f16(b0_7_2, vld1q_f16(&B[(k + 3) * N + n]), a[3]);

        float32x4_t c0_7_low_32 = vaddq_f32(vld1q_f32(&C[m * N + n]),
                                            vcvt_f32_f16(vget_low_f16(b0_7_0)));
        float32x4_t c0_7_high_32 = vaddq_f32(
          vld1q_f32(&C[m * N + n + 4]), vcvt_f32_f16(vget_high_f16(b0_7_0)));

        c0_7_low_32 =
          vaddq_f32(c0_7_low_32, vcvt_f32_f16(vget_low_f16(b0_7_2)));
        c0_7_high_32 =
          vaddq_f32(c0_7_high_32, vcvt_f32_f16(vget_high_f16(b0_7_2)));

        vst1q_f32(&C[m * N + n], c0_7_low_32);
        vst1q_f32(&C[m * N + n + 4], c0_7_high_32);
      }
    }
  }

  // remaining K values
  for (; k < K; k++) {
    for (unsigned int m = 0; m < M; m++) {
      __fp16 a0 = alpha * A[m * K + k];

      for (n = 0; (N - n) >= 8; n += 8) {

        float16x8_t b0_7 = vmulq_n_f16(vld1q_f16(&B[k * N + n]), a0);

        float32x4_t c0_7_low_32 =
          vaddq_f32(vld1q_f32(&C[m * N + n]), vcvt_f32_f16(vget_low_f16(b0_7)));

        float32x4_t c0_7_high_32 = vaddq_f32(vld1q_f32(&C[m * N + n + 4]),
                                             vcvt_f32_f16(vget_high_f16(b0_7)));

        vst1q_f32(&C[m * N + n], c0_7_low_32);
        vst1q_f32(&C[m * N + n + 4], c0_7_high_32);
      }
    }
  }

  // remaining N values
  if (n < N) {
    __fp16 valsB[8];
    float valsC[8];
    for (k = 0; k < K; k++) {
      for (unsigned int m = 0; m < M; m++) {
        __fp16 a = alpha * A[m * K + k];
        for (unsigned int idx = n; idx < N; idx++) {
          valsB[idx - n] = B[k * N + idx];

          // load previously calculated C
          valsC[idx - n] = C[m * N + idx];
        }

        float16x8_t b = vmulq_n_f16(vld1q_f16(valsB), a);

        float32x4_t c0_7_low_32 =
          vaddq_f32(vld1q_f32(valsC), vcvt_f32_f16(vget_low_f16(b)));

        float32x4_t c0_7_high_32 =
          vaddq_f32(vld1q_f32(valsC + 4), vcvt_f32_f16(vget_high_f16(b)));

        vst1q_f32(valsC, c0_7_low_32);
        vst1q_f32(valsC + 4, c0_7_high_32);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }
}

void sgemm_neon_fp16_transA(const __fp16 *A, const __fp16 *B, float *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            float beta) {
  __fp16 valsB[8];
  float valsC[8];
  for (unsigned int k = 0; k < K; k++) {
    for (unsigned int m = 0; m < M; m++) {
      __fp16 a = alpha * A[k * M + m];
      unsigned int n = 0;
      for (; (N - n) >= 8; n += 8) {
        // fp16 multiplication
        float16x8_t b = vmulq_n_f16(vld1q_f16(&B[k * N + n]), a);

        // fp32 additions
        float32x4_t c0_7_low_32 =
          vaddq_f32(vld1q_f32(&C[m * N + n]), vcvt_f32_f16(vget_low_f16(b)));
        float32x4_t c0_7_high_32 = vaddq_f32(vld1q_f32(&C[m * N + n + 4]),
                                             vcvt_f32_f16(vget_high_f16(b)));

        vst1q_f32(&C[m * N + n], c0_7_low_32);
        vst1q_f32(&C[m * N + n + 4], c0_7_high_32);
      }

      // remaining N values
      if (n < N) {
        for (unsigned int idx = n; idx < N; idx++) {
          valsB[idx - n] = B[k * N + idx];

          // load previously calculated C
          valsC[idx - n] = C[m * N + idx];
        }

        float16x8_t b = vmulq_n_f16(vld1q_f16(valsB), a);

        float32x4_t c0_7_low_32 =
          vaddq_f32(vld1q_f32(valsC), vcvt_f32_f16(vget_low_f16(b)));
        float32x4_t c0_7_high_32 =
          vaddq_f32(vld1q_f32(valsC + 4), vcvt_f32_f16(vget_high_f16(b)));

        vst1q_f32(valsC, c0_7_low_32);
        vst1q_f32(valsC + 4, c0_7_high_32);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }
}

void sgemm_neon_fp16_transB(const __fp16 *A, const __fp16 *B, float *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            float beta) {
  if (K % 16 == 0) {
    for (unsigned int m = 0; m < M; m++) {
      for (unsigned int n = 0; n < N; n++) {
        float32x4_t sum = vmovq_n_f32(0.0f);
        unsigned int k = 0;
        for (; (K - k) >= 16; k += 16) {
          // fp16 multiplication
          float16x8_t ab =
            vmulq_f16(vld1q_f16(&A[m * K + k]), vld1q_f16(&B[n * K + k]));
          float16x8_t ab8_15 = vmulq_f16(vld1q_f16(&A[m * K + k + 8]),
                                         vld1q_f16(&B[n * K + k + 8]));

          // fp16 partial accumulation
          ab = vaddq_f16(ab, ab8_15);

          // fp32 partial accumulation
          sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(ab)));
          sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(ab)));
        }

        sum = vmulq_n_f32(sum, alpha);

        C[m * N + n] += vaddvq_f32(sum);
      }
    }
  } else {
    __fp16 valsB[8];
    __fp16 valsA[8];
    for (unsigned int m = 0; m < M; m++) {
      for (unsigned int n = 0; n < N; n++) {
        float32x4_t sum = vmovq_n_f32(0.0f);
        unsigned int k = 0;
        for (; (K - k) >= 8; k += 8) {
          float16x8_t ab =
            vmulq_f16(vld1q_f16(&A[m * K + k]), vld1q_f16(&B[n * K + k]));

          sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(ab)));
          sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(ab)));
        }

        // remaining K values
        if (k < K) {
          unsigned int idx;
          for (idx = k; idx < K; idx++) {
            valsA[idx - k] = A[m * K + idx];
            valsB[idx - k] = B[n * K + idx];
          }
          // to cover entire 128 bits (reset unused bits)
          while (idx < (k + 8)) {
            valsA[idx - k] = 0;
            valsB[idx - k] = 0;
            idx++;
          }
          // updating sum
          float16x8_t ab = vmulq_f16(vld1q_f16(valsA), vld1q_f16(valsB));

          sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(ab)));
          sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(ab)));
        }

        sum = vmulq_n_f32(sum, alpha);

        C[m * N + n] += vaddvq_f32(sum);
      }
    }
  }
}

void sgemm_neon_fp16_transAB(const __fp16 *A, const __fp16 *B, float *C,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             float beta, uint32_t idx) {
  float vals[8];
  __fp16 vals_fp16[8];
  for (unsigned int n = 0; n < N; n++) {
    for (unsigned int k = 0; k < K; k++) {

      __fp16 b = alpha * B[n * K + k];
      unsigned int m = 0;
      for (; (M - m) >= 8; m += 8) {
        // fp16 multiplication
        float16x8_t a = vmulq_n_f16(vld1q_f16(&A[k * M + m]), b);

        vst1q_f32(vals, vcvt_f32_f16(vget_low_f16(a)));
        vst1q_f32(vals + 4, vcvt_f32_f16(vget_high_f16(a)));

        // calculations for all M values (fp32 additions)
        for (unsigned int idx = m; idx < m + 8; idx++)
          C[idx * N + n] += vals[idx - m];
      }

      // remaining when M is not a multiple of 8
      if (m < M) {
        for (idx = m; idx < M; idx++) {
          vals_fp16[idx - m] = A[k * M + idx];
        }

        float16x8_t a = vmulq_n_f16(vld1q_f16(vals_fp16), b);

        vst1q_f32(vals, vcvt_f32_f16(vget_low_f16(a)));
        vst1q_f32(vals + 4, vcvt_f32_f16(vget_high_f16(a)));

        // calculations for all remaining M values
        for (idx = m; idx < M; idx++)
          C[idx * N + n] += vals[idx - m];
      }
    }
  }
}

void elementwise_vector_multiplication_neon_fp16(const unsigned int N,
                                                 const __fp16 *X,
                                                 const __fp16 *Y, __fp16 *Z) {
  int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    float16x8_t z0_7 = vmulq_f16(x0_7, y0_7);

    vst1q_f16(&Z[i], z0_7);
  }
  while (i < N) {
    Z[i] = X[i] * Y[i];
    ++i;
  }
}

void elementwise_vector_addition_neon_fp16(const unsigned int N,
                                           const __fp16 *X, const __fp16 *Y,
                                           __fp16 *Z) {
  int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    float16x8_t z0_7 = vaddq_f16(x0_7, y0_7);

    vst1q_f16(&Z[i], z0_7);
  }
  while (i < N) {
    Z[i] = X[i] * Y[i];
    ++i;
  }
}

#endif
} // namespace nntrainer::neon
