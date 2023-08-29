// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_neon.cpp
 * @date   4 Aug 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
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

  float16x8_t v_beta = vmovq_n_f16(beta);

  for (unsigned int i = 0; i < rows; i += 8) {
    float16x8_t y = vld1q_f16(&Y[i]);
    y = vmulq_f16(v_beta, y);
    vst1q_f16(&Y[i], y);
  }

  float16x8_t v_alpha = vmovq_n_f16(alpha);

  if (cols % 32 == 0) {
    for (unsigned i = 0; i < cols; i += 32) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);
      float16x8_t x8_15 = vld1q_f16(&X[i + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[i + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[i + 24]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
        x8_15 = vmulq_f16(x8_15, v_alpha);
        x16_23 = vmulq_f16(x16_23, v_alpha);
        x24_31 = vmulq_f16(x24_31, v_alpha);
      }

      float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;

      const __fp16 *__restrict w;

      float16x8_t y0;
      __fp16 r[4];

      float16x4_t y0_high;
      float16x4_t y0_low;
      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f16(0);

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);
        wvec16_23 = vld1q_f16(&w[16]);
        wvec24_31 = vld1q_f16(&w[24]);

        y0 = vfmaq_f16(y0, wvec0_7, x0_7);
        y0 = vfmaq_f16(y0, wvec8_15, x8_15);
        y0 = vfmaq_f16(y0, wvec16_23, x16_23);
        y0 = vfmaq_f16(y0, wvec24_31, x24_31);

        y0_high = vget_high_f16(y0);
        y0_low = vget_low_f16(y0);

        y0_low = vadd_f16(y0_high, y0_low);
        vst1_f16(r, y0_low);

        Y[j] += r[0] + r[1] + r[2] + r[3];
      }
    }

  } else if (cols % 16 == 0) {

    for (unsigned i = 0; i < cols; i += 16) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);
      float16x8_t x8_15 = vld1q_f16(&X[i + 8]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
        x8_15 = vmulq_f16(x8_15, v_alpha);
      }

      float16x8_t wvec0_7, wvec8_15;

      const __fp16 *__restrict w;

      float16x8_t y0;
      __fp16 r[4];

      float16x4_t y0_high;
      float16x4_t y0_low;
      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f16(0);

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);

        y0 = vfmaq_f16(y0, wvec0_7, x0_7);
        y0 = vfmaq_f16(y0, wvec8_15, x8_15);

        y0_high = vget_high_f16(y0);
        y0_low = vget_low_f16(y0);

        y0_low = vadd_f16(y0_high, y0_low);
        vst1_f16(r, y0_low);

        Y[j] += r[0] + r[1] + r[2] + r[3];
      }
    }
  } else if (cols % 8 == 0) {
    for (unsigned i = 0; i < cols; i += 8) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
      }

      float16x8_t wvec0_7;

      float16x8_t y0;
      __fp16 r[4];

      float16x4_t y0_high;
      float16x4_t y0_low;
      for (unsigned int j = 0; j < rows; ++j) {

        wvec0_7 = vld1q_f16(&A[j * cols + i]);

        y0 = vmulq_f16(wvec0_7, x0_7);

        y0_high = vget_high_f16(y0);
        y0_low = vget_low_f16(y0);

        y0_low = vadd_f16(y0_high, y0_low);
        vst1_f16(r, y0_low);

        Y[j] += r[0] + r[1] + r[2] + r[3];
      }
    }
  }
}

void sgemv_transpose_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y,
                               uint32_t rows, uint32_t cols, float alpha,
                               float beta) {

  const float16x8_t v_beta = vmovq_n_f16(beta);
  const float16x8_t v_alpha = vmovq_n_f16(alpha);

  if (cols % 32 == 0) {

    for (unsigned int j = 0; j < cols; j += 8) {
      float16x8_t y0_7 = vld1q_f16(&Y[j]);
      y0_7 = vmulq_f16(y0_7, v_beta);
      vst1q_f16(&Y[j], y0_7);
    }

    for (unsigned int i = 0; i < rows; ++i) {
      __fp16 x = alpha * X[i];

      for (unsigned int j = 0; j < cols; j += 32) {
        __fp16 *__restrict y = &Y[j];

        float16x8_t y0_7 = vld1q_f16(&y[0]);
        float16x8_t y8_15 = vld1q_f16(&y[8]);
        float16x8_t y16_23 = vld1q_f16(&y[16]);
        float16x8_t y24_31 = vld1q_f16(&y[24]);

        float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;
        const __fp16 *__restrict w;

        w = &A[i * cols + j];

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);
        wvec16_23 = vld1q_f16(&w[16]);
        wvec24_31 = vld1q_f16(&w[24]);

        y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);
        y8_15 = vfmaq_n_f16(y8_15, wvec8_15, x);
        y16_23 = vfmaq_n_f16(y16_23, wvec16_23, x);
        y24_31 = vfmaq_n_f16(y24_31, wvec24_31, x);

        vst1q_f16(&y[0], y0_7);
        vst1q_f16(&y[8], y8_15);
        vst1q_f16(&y[16], y16_23);
        vst1q_f16(&y[24], y24_31);
      }
    }
    return;
  } else if (cols % 16 == 0) {

    for (unsigned int j = 0; j < cols; j += 8) {
      float16x8_t y0_7 = vld1q_f16(&Y[j]);
      y0_7 = vmulq_f16(y0_7, v_beta);
      vst1q_f16(&Y[j], y0_7);
    }

    for (unsigned int i = 0; i < rows; ++i) {
      __fp16 x = alpha * X[i];

      for (unsigned int j = 0; j < cols; j += 16) {
        __fp16 *__restrict y = &Y[j];

        float16x8_t y0_7 = vld1q_f16(&y[0]);
        float16x8_t y8_15 = vld1q_f16(&y[8]);

        float16x8_t wvec0_7, wvec8_15;
        const __fp16 *__restrict w;

        w = &A[i * cols + j];

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);

        y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);
        y8_15 = vfmaq_n_f16(y8_15, wvec8_15, x);

        vst1q_f16(&y[0], y0_7);
        vst1q_f16(&y[8], y8_15);
      }
    }
    return;
  } else if (cols % 8 == 0) {

    for (unsigned int j = 0; j < cols; j += 8) {
      float16x8_t y0_7 = vld1q_f16(&Y[j]);
      y0_7 = vmulq_f16(y0_7, v_beta);
      vst1q_f16(&Y[j], y0_7);
    }

    for (unsigned int i = 0; i < rows; ++i) {

      __fp16 x = alpha * X[i];

      for (unsigned int j = 0; j < cols; j += 8) {

        float16x8_t y0_7 = vld1q_f16(&Y[j]);
        float16x8_t wvec0_7 = vld1q_f16(&A[i * cols + j]);

        y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);

        vst1q_f16(&Y[j], y0_7);
      }
    }
    return;
  }
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
#endif

} // namespace nntrainer::neon
