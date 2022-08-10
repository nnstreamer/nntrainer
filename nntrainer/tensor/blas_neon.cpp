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
    bool initialized[cols / 16];
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
    return;
  } else if (cols % 8 == 0) {
    bool initialized[cols / 8];
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
    return;
  } else if (cols % 4 == 0) {
    bool initialized[cols / 4];
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
  }
  return;
}

} // namespace nntrainer::neon
