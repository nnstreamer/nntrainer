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
#include <memory>
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

void scopy_neon_int4_to_fp32(const unsigned int N, const uint8_t *X, float *Y) {

  unsigned int idx = 0;

  // keep in mind that : len(X) = N, and len(Y) = 2*N

  // processing batch of 16
  float32x4_t y0, y1, y2, y3;
  float32x4_t y4, y5, y6, y7;

  uint8_t low0, low1, high0, high1;

  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);

    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);
    unsigned int i = 0;
    for (; i < 8; ++i) {
      low0 = low[i] >> 4;
      low1 = low[i] & 0x0f;

      high0 = high[i] >> 4;
      high1 = high[i] & 0x0f;

      // 0 ~ 8
      if (i < 2) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else if (i < 4) {
        y1[2 * (i - 2)] = low0;
        y1[2 * (i - 2) + 1] = low1;
      } else if (i < 6) {
        y2[2 * (i - 4)] = low0;
        y2[2 * (i - 4) + 1] = low1;
      } else {
        y3[2 * (i - 6)] = low0;
        y3[2 * (i - 6) + 1] = low1;
      }

      // 8 ~ 16
      if (i < 2) {
        y4[2 * i] = high0;
        y4[2 * i + 1] = high1;
      } else if (i < 4) {
        y5[2 * (i - 2)] = high0;
        y5[2 * (i - 2) + 1] = high1;
      } else if (i < 6) {
        y6[2 * (i - 4)] = high0;
        y6[2 * (i - 4) + 1] = high1;
      } else {
        y7[2 * (i - 6)] = high0;
        y7[2 * (i - 6) + 1] = high1;
      }
    }
    vst1q_f32(&Y[2 * idx], y0);
    vst1q_f32(&Y[2 * idx + 4], y1);
    vst1q_f32(&Y[2 * idx + 8], y2);
    vst1q_f32(&Y[2 * idx + 12], y3);
    vst1q_f32(&Y[2 * idx + 16], y4);
    vst1q_f32(&Y[2 * idx + 20], y5);
    vst1q_f32(&Y[2 * idx + 24], y6);
    vst1q_f32(&Y[2 * idx + 28], y7);
  }

  // processing remaining batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    unsigned int i = 0;
    for (; i < 8; ++i) {
      low0 = batch[i] >> 4;
      low1 = batch[i] & 0x0f;

      if (i < 2) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else if (i < 4) {
        y1[2 * (i - 2)] = low0;
        y1[2 * (i - 2) + 1] = low1;
      } else if (i < 6) {
        y2[2 * (i - 4)] = low0;
        y2[2 * (i - 4) + 1] = low1;
      } else {
        y3[2 * (i - 6)] = low0;
        y3[2 * (i - 6) + 1] = low1;
      }
    }

    vst1q_f32(&Y[2 * idx], y0);
    vst1q_f32(&Y[2 * idx + 4], y1);
    vst1q_f32(&Y[2 * idx + 8], y2);
    vst1q_f32(&Y[2 * idx + 12], y3);
  }

  // pocessing remaining values
  for (; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

void scopy_neon_int8_or_int4(const unsigned int N, const uint8_t *X,
                             uint8_t *Y) {
  ///@note int8 Tensor and int4 Tensor share the same memory offset
  unsigned int idx = 0;
  for (; N - idx >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    vst1q_u8(&Y[idx], batch);
  }
  for (; N - idx >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);
    vst1_u8(&Y[idx], batch);
  }
  for (; N - idx >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void sine_transformation_neon(const unsigned int N, float *X, float *Y,
                              float alpha) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    if (alpha != 1.0)
      x0_3 = vmulq_n_f32(x0_3, alpha);
    float32x4_t sinx0_3 = sin_ps(x0_3);
    vst1q_f32(&Y[i], sinx0_3);
  }
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]);
    ++i;
  }
}

void cosine_transformation_neon(const unsigned int N, float *X, float *Y,
                                float alpha) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    if (alpha != 1.0)
      x0_3 = vmulq_n_f32(x0_3, alpha);
    float32x4_t cosx0_3 = cos_ps(x0_3);
    vst1q_f32(&Y[i], cosx0_3);
  }
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]);
    ++i;
  }
}

void inv_sqrt_inplace_neon(const unsigned int N, float *X) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_7 = vld1q_f32(&X[i]);
    float32x4_t x0_7_sqrt = vsqrtq_f32(x0_7);
    float32x4_t ones = vmovq_n_f32(1);
    float32x4_t x0_7_sqrt_div = vdivq_f32(ones, x0_7_sqrt);
    vst1q_f32(&X[i], x0_7_sqrt_div);
  }
  while (i < N) {
    X[i] = (1 / std::sqrt(static_cast<float>(X[i])));
    ++i;
  }
}

#ifdef ENABLE_FP16

void sgemv_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t rows,
                     uint32_t cols, float alpha, float beta) {
  const int batch = 0;
  const __fp16 *__restrict x;
  float *Y32 = new float[rows];

  unsigned int idx = 0;

  for (; rows - idx >= 8; idx += 8) {
    float32x4_t y0_3 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));
    y0_3 = vmulq_n_f32(y0_3, beta);
    y4_7 = vmulq_n_f32(y4_7, beta);

    vst1q_f32(&Y32[idx], y0_3);
    vst1q_f32(&Y32[idx + 4], y4_7);
  }
  for (; rows - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_n_f32(y0_3_32, beta);

    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; idx < rows; ++idx) {
    Y32[idx] = beta * Y[idx];
  }

  idx = 0;
  if (cols / 120 >= batch) {
    for (; cols - idx >= 120; idx += 120) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[idx + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[idx + 24]);

      float16x8_t x32_39 = vld1q_f16(&X[idx + 32]);
      float16x8_t x40_47 = vld1q_f16(&X[idx + 40]);
      float16x8_t x48_55 = vld1q_f16(&X[idx + 48]);
      float16x8_t x56_63 = vld1q_f16(&X[idx + 56]);

      float16x8_t x64_71 = vld1q_f16(&X[idx + 64]);
      float16x8_t x72_79 = vld1q_f16(&X[idx + 72]);
      float16x8_t x80_87 = vld1q_f16(&X[idx + 80]);

      float16x8_t x88_95 = vld1q_f16(&X[idx + 88]);
      float16x8_t x96_103 = vld1q_f16(&X[idx + 96]);
      float16x8_t x104_111 = vld1q_f16(&X[idx + 104]);
      float16x8_t x112_120 = vld1q_f16(&X[idx + 112]);

      if (alpha != 1.0) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
        x16_23 = vmulq_n_f16(x16_23, alpha);
        x24_31 = vmulq_n_f16(x24_31, alpha);
        x32_39 = vmulq_n_f16(x32_39, alpha);
        x40_47 = vmulq_n_f16(x40_47, alpha);
        x48_55 = vmulq_n_f16(x48_55, alpha);
        x56_63 = vmulq_n_f16(x56_63, alpha);

        x64_71 = vmulq_n_f16(x64_71, alpha);
        x72_79 = vmulq_n_f16(x72_79, alpha);
        x80_87 = vmulq_n_f16(x80_87, alpha);
        x88_95 = vmulq_n_f16(x88_95, alpha);
        x96_103 = vmulq_n_f16(x96_103, alpha);
        x104_111 = vmulq_n_f16(x104_111, alpha);
        x112_120 = vmulq_n_f16(x112_120, alpha);
      }

      const __fp16 *__restrict w;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        y = vfmaq_f16(y, vld1q_f16(&w[32]), x32_39);
        y = vfmaq_f16(y, vld1q_f16(&w[40]), x40_47);
        y = vfmaq_f16(y, vld1q_f16(&w[48]), x48_55);
        y = vfmaq_f16(y, vld1q_f16(&w[56]), x56_63);

        y = vfmaq_f16(y, vld1q_f16(&w[64]), x64_71);
        y = vfmaq_f16(y, vld1q_f16(&w[72]), x72_79);
        y = vfmaq_f16(y, vld1q_f16(&w[80]), x80_87);

        y = vfmaq_f16(y, vld1q_f16(&w[88]), x88_95);
        y = vfmaq_f16(y, vld1q_f16(&w[96]), x96_103);
        y = vfmaq_f16(y, vld1q_f16(&w[104]), x104_111);
        y = vfmaq_f16(y, vld1q_f16(&w[112]), x112_120);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (cols / 64 >= batch) {
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

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        y = vfmaq_f16(y, vld1q_f16(&w[32]), x32_39);
        y = vfmaq_f16(y, vld1q_f16(&w[40]), x40_47);
        y = vfmaq_f16(y, vld1q_f16(&w[48]), x48_55);
        y = vfmaq_f16(y, vld1q_f16(&w[56]), x56_63);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (cols / 32 >= batch) {
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

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);
        y = vfmaq_f16(y, vld1q_f16(&w[16]), x16_23);
        y = vfmaq_f16(y, vld1q_f16(&w[24]), x24_31);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  if (cols / 16 >= batch) {
    for (; cols - idx >= 16; idx += 16) {
      float16x8_t x0_7 = vld1q_f16(&X[idx]);
      float16x8_t x8_15 = vld1q_f16(&X[idx + 8]);

      if (alpha != 1.0) {
        x0_7 = vmulq_n_f16(x0_7, alpha);
        x8_15 = vmulq_n_f16(x8_15, alpha);
      }

      const __fp16 *__restrict w;
      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + idx];
        float16x8_t y = vmulq_f16(vld1q_f16(&w[0]), x0_7);
        y = vfmaq_f16(y, vld1q_f16(&w[8]), x8_15);

        Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                  vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
      }
    }
  }
  for (; cols - idx >= 8; idx += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[idx]);

    if (alpha != 1.0) {
      x0_7 = vmulq_n_f16(x0_7, alpha);
    }

    const __fp16 *__restrict w;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float16x8_t wvec0_7 = vld1q_f16(&w[0]);
      float16x8_t y = vmulq_f16(wvec0_7, x0_7);

      Y32[j] += vaddvq_f32(vcvt_f32_f16(vget_low_f16(y))) +
                vaddvq_f32(vcvt_f32_f16(vget_high_f16(y)));
    }
  }
  for (; cols - idx >= 4; idx += 4) {
    float32x4_t x0_3 = vcvt_f32_f16(vld1_f16(&X[idx]));

    if (alpha != 1.0) {
      x0_3 = vmulq_n_f32(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float32x4_t wvec0_3 = vcvt_f32_f16(vld1_f16(&w[0]));
      float32x4_t y0 = vmulq_f32(wvec0_3, x0_3);

      Y32[j] += vaddvq_f32(y0);
    }
  }

  // now, cols - idx is under 4 : 0 1 2 3 = cols - idx
  if (cols != idx) {
    float32x4_t x0_3 = vcvt_f32_f16(vld1_f16(&X[idx]));
    for (int j = cols - idx; j < 4; ++j) {
      x0_3[j] = 0;
    }

    if (alpha != 1.0) {
      x0_3 = vmulq_n_f32(x0_3, alpha);
    }

    const __fp16 *__restrict w;

    __fp16 yVal;

    for (unsigned int j = 0; j < rows; ++j) {
      w = &A[j * cols + idx];
      float32x4_t wvec0_3 = vcvt_f32_f16(vld1_f16(&w[0]));

      for (int k = cols - idx; k < 4; ++k) {
        wvec0_3[k] = 0;
      }

      float32x4_t y0 = vmulq_f32(wvec0_3, x0_3);

      for (unsigned int k = 0; k < cols - idx; ++k) {
        Y32[j] += y0[k];
      }
    }
  }

  scopy_neon_fp32_to_fp16(rows, Y32, Y);
  delete[] Y32;
  return;
}

void sgemv_transpose_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y,
                               uint32_t rows, uint32_t cols, float alpha,
                               float beta) {
  float *Y32 = new float[cols];
  const int batch = 20;
  unsigned int idx = 0;
  for (; cols - idx >= 8; idx += 8) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    float32x4_t y4_7_32 = vcvt_f32_f16(vld1_f16(&Y[idx + 4]));

    y0_3_32 = vmulq_n_f32(y0_3_32, beta);
    y4_7_32 = vmulq_n_f32(y4_7_32, beta);

    vst1q_f32(&Y32[idx], y0_3_32);
    vst1q_f32(&Y32[idx + 4], y4_7_32);
  }
  for (; cols - idx >= 4; idx += 4) {
    float32x4_t y0_3_32 = vcvt_f32_f16(vld1_f16(&Y[idx]));
    y0_3_32 = vmulq_n_f32(y0_3_32, beta);
    vst1q_f32(&Y32[idx], y0_3_32);
  }
  for (; cols - idx >= 1; idx += 1) {
    Y32[idx] = beta * Y[idx];
  }
  if (rows % 16 == 0 && rows / 16 >= batch && cols % 4 == 0) {
    for (unsigned int i = 0; i < rows; i += 16) {
      __fp16 x = alpha * (X[i]);
      __fp16 x2 = alpha * (X[i + 1]);
      __fp16 x3 = alpha * (X[i + 2]);
      __fp16 x4 = alpha * (X[i + 3]);
      __fp16 x5 = alpha * (X[i + 4]);
      __fp16 x6 = alpha * (X[i + 5]);
      __fp16 x7 = alpha * (X[i + 6]);
      __fp16 x8 = alpha * (X[i + 7]);
      __fp16 x9 = alpha * (X[i + 8]);
      __fp16 x10 = alpha * (X[i + 9]);
      __fp16 x11 = alpha * (X[i + 10]);
      __fp16 x12 = alpha * (X[i + 11]);
      __fp16 x13 = alpha * (X[i + 12]);
      __fp16 x14 = alpha * (X[i + 13]);
      __fp16 x15 = alpha * (X[i + 14]);
      __fp16 x16 = alpha * (X[i + 15]);

      idx = 0;
      for (; cols - idx >= 8; idx += 8) {
        float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * cols + idx]), x);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * cols + idx]), x2);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 2) * cols + idx]), x3);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 3) * cols + idx]), x4);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 4) * cols + idx]), x5);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 5) * cols + idx]), x6);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 6) * cols + idx]), x7);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 7) * cols + idx]), x8);

        float16x8_t w2vec0_7_f16 =
          vmulq_n_f16(vld1q_f16(&A[(i + 8) * cols + idx]), x9);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 9) * cols + idx]), x10);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 10) * cols + idx]), x11);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 11) * cols + idx]), x12);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 12) * cols + idx]), x13);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 13) * cols + idx]), x14);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 14) * cols + idx]), x15);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 15) * cols + idx]), x16);

        float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                     vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
        y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
        float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                     vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
        y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
      }

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&A[i * cols + idx])), x);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx])), x2);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx])), x3);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx])), x4);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 4) * cols + idx])), x5);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 5) * cols + idx])), x6);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 6) * cols + idx])), x7);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 7) * cols + idx])), x8);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 8) * cols + idx])), x9);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 9) * cols + idx])), x10);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 10) * cols + idx])), x11);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 11) * cols + idx])), x12);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 12) * cols + idx])), x13);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 13) * cols + idx])), x14);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 14) * cols + idx])), x15);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 15) * cols + idx])), x16);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols - idx >= 1) {
        float y0_3_0[4];

        float v0[4], v1[4], v2[4], v3[4];
        float v4[4], v5[4], v6[4], v7[4];
        float v8[4], v9[4], v10[4], v11[4];
        float v12[4], v13[4], v14[4], v15[4];

        unsigned int k = 0;
        for (; k < cols - idx; ++k) {

          y0_3_0[k] = Y32[idx + k];

          v0[k] = A[i * cols + idx + k];
          v1[k] = A[(i + 1) * cols + idx + k];
          v2[k] = A[(i + 2) * cols + idx + k];
          v3[k] = A[(i + 3) * cols + idx + k];
          v4[k] = A[(i + 4) * cols + idx + k];
          v5[k] = A[(i + 5) * cols + idx + k];
          v6[k] = A[(i + 6) * cols + idx + k];
          v7[k] = A[(i + 7) * cols + idx + k];
          v8[k] = A[(i + 8) * cols + idx + k];
          v9[k] = A[(i + 9) * cols + idx + k];
          v10[k] = A[(i + 10) * cols + idx + k];
          v11[k] = A[(i + 11) * cols + idx + k];
          v12[k] = A[(i + 12) * cols + idx + k];
          v13[k] = A[(i + 13) * cols + idx + k];
          v15[k] = A[(i + 15) * cols + idx + k];
        }
        for (; k < 4; ++k) {
          y0_3_0[k] = 0;

          v0[k] = v1[k] = v2[k] = v3[k] = 0;
          v4[k] = v5[k] = v6[k] = v7[k] = 0;
          v8[k] = v9[k] = v10[k] = v11[k] = 0;
          v12[k] = v13[k] = v14[k] = v15[k] = 0;
        }

        float32x4_t y0_3 = vld1q_f32(y0_3_0);
        float32x4_t y0_3_tmp = vmovq_n_f32(0);

        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v0), x);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v1), x2);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v2), x3);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v3), x4);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v4), x5);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v5), x6);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v6), x7);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v7), x8);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v8), x9);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v9), x10);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v10), x11);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v11), x12);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v12), x13);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v13), x14);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v14), x15);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v15), x16);

        for (unsigned int k = 0; k < cols - idx; ++k) {
          Y32[idx + k] = y0_3[k];
        }
      }
    }
  } else if (rows % 8 == 0 && rows / 8 >= batch) {
    for (unsigned int i = 0; i < rows; i += 8) {
      __fp16 x = alpha * X[i];
      __fp16 x2 = alpha * X[i + 1];
      __fp16 x3 = alpha * X[i + 2];
      __fp16 x4 = alpha * X[i + 3];
      __fp16 x5 = alpha * X[i + 4];
      __fp16 x6 = alpha * X[i + 5];
      __fp16 x7 = alpha * X[i + 6];
      __fp16 x8 = alpha * X[i + 7];

      idx = 0;
      for (; cols - idx >= 8; idx += 8) {
        float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * cols + idx]), x);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * cols + idx]), x2);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 2) * cols + idx]), x3);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 3) * cols + idx]), x4);

        float16x8_t w2vec0_7_f16 =
          vmulq_n_f16(vld1q_f16(&A[(i + 4) * cols + idx]), x5);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 5) * cols + idx]), x6);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 6) * cols + idx]), x7);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 7) * cols + idx]), x8);

        float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                     vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
        y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
        float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                     vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
        y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
      }

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&A[i * cols + idx])), x);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx])), x2);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx])), x3);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx])), x4);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 4) * cols + idx])), x5);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 5) * cols + idx])), x6);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 6) * cols + idx])), x7);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 7) * cols + idx])), x8);
        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols - idx >= 1) {
        float y0_3_0[4];

        float v0[4], v1[4], v2[4], v3[4];
        float v4[4], v5[4], v6[4], v7[4];

        unsigned int k = 0;
        for (; k < cols - idx; ++k) {
          y0_3_0[k] = Y32[idx + k];

          v0[k] = A[i * cols + idx + k];
          v1[k] = A[(i + 1) * cols + idx + k];
          v2[k] = A[(i + 2) * cols + idx + k];
          v3[k] = A[(i + 3) * cols + idx + k];
          v4[k] = A[(i + 4) * cols + idx + k];
          v5[k] = A[(i + 5) * cols + idx + k];
          v6[k] = A[(i + 6) * cols + idx + k];
          v7[k] = A[(i + 7) * cols + idx + k];
        }
        for (; k < 4; ++k) {
          y0_3_0[k] = 0;
          v0[k] = v1[k] = v2[k] = v3[k] = 0;
          v4[k] = v5[k] = v6[k] = v7[k] = 0;
        }

        float32x4_t y0_3 = vld1q_f32(y0_3_0);

        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v0), x);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v1), x2);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v2), x3);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v3), x4);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v4), x5);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v5), x6);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v6), x7);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v7), x8);

        for (unsigned int k = 0; k < cols - idx; ++k) {
          Y32[idx + k] = y0_3[k];
        }
      }
    }
  } else if (rows % 4 == 0 && rows / 4 >= batch) {
    for (unsigned int i = 0; i < rows; i += 4) {
      __fp16 x = alpha * (X[i]);
      __fp16 x2 = alpha * (X[i + 1]);
      __fp16 x3 = alpha * (X[i + 2]);
      __fp16 x4 = alpha * (X[i + 3]);

      idx = 0;
      for (; cols - idx >= 8; idx += 8) {
        float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * cols + idx]), x);
        wvec0_7_f16 =
          vfmaq_n_f16(wvec0_7_f16, vld1q_f16(&A[(i + 1) * cols + idx]), x2);
        float16x8_t w2vec0_7_f16 =
          vmulq_n_f16(vld1q_f16(&A[(i + 2) * cols + idx]), x3);
        w2vec0_7_f16 =
          vfmaq_n_f16(w2vec0_7_f16, vld1q_f16(&A[(i + 3) * cols + idx]), x4);

        float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                     vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
        y0_3 = vaddq_f32(y0_3, vcvt_f32_f16(vget_low_f16(w2vec0_7_f16)));
        float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                     vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));
        y4_7 = vaddq_f32(y4_7, vcvt_f32_f16(vget_high_f16(w2vec0_7_f16)));

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
      }

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);

        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&A[i * cols + idx])), x);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 1) * cols + idx])), x2);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 2) * cols + idx])), x3);
        y0_3 = vfmaq_n_f32(
          y0_3, vcvt_f32_f16(vld1_f16(&A[(i + 3) * cols + idx])), x4);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols - idx >= 1) {
        float y0_3_0[4];

        float v0[4], v1[4], v2[4], v3[4];
        unsigned int k = 0;
        for (; k < cols - idx; ++k) {
          y0_3_0[k] = Y32[idx + k];

          v0[k] = A[i * cols + idx + k];
          v1[k] = A[(i + 1) * cols + idx + k];
          v2[k] = A[(i + 2) * cols + idx + k];
          v3[k] = A[(i + 3) * cols + idx + k];
        }
        for (; k < 4; ++k) {
          y0_3_0[k] = 0;
          v0[k] = v1[k] = v2[k] = v3[k] = 0;
        }

        float32x4_t y0_3 = vld1q_f32(y0_3_0);

        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v0), x);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v1), x2);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v2), x3);
        y0_3 = vfmaq_n_f32(y0_3, vld1q_f32(v3), x4);

        for (unsigned int k = 0; k < cols - idx; ++k) {
          Y32[idx + k] = y0_3[k];
        }
      }
    }
  } else {
    for (unsigned int i = 0; i < rows; ++i) {
      __fp16 x = alpha * (X[i]);
      idx = 0;
      for (; cols - idx >= 8; idx += 8) {

        float16x8_t wvec0_7_f16 = vmulq_n_f16(vld1q_f16(&A[i * cols + idx]), x);
        float32x4_t y0_3 = vaddq_f32(vld1q_f32(&Y32[idx]),
                                     vcvt_f32_f16(vget_low_f16(wvec0_7_f16)));
        float32x4_t y4_7 = vaddq_f32(vld1q_f32(&Y32[idx + 4]),
                                     vcvt_f32_f16(vget_high_f16(wvec0_7_f16)));

        vst1q_f32(&Y32[idx], y0_3);
        vst1q_f32(&Y32[idx + 4], y4_7);
      }

      for (; cols - idx >= 4; idx += 4) {

        float32x4_t y0_3 = vld1q_f32(&Y32[idx]);
        y0_3 = vfmaq_n_f32(y0_3, vcvt_f32_f16(vld1_f16(&A[i * cols + idx])), x);

        vst1q_f32(&Y32[idx], y0_3);
      }

      if (cols != idx) {
        float y0_3[4];
        float wvec0_3[4];
        for (unsigned int j = 0; j < cols - idx; ++j) {
          y0_3[j] = Y32[idx + j];
          wvec0_3[j] = A[i * cols + idx + j];
        }
        for (int j = cols - idx; j < 4; ++j) {
          y0_3[j] = 0;
          wvec0_3[j] = 0;
        }

        float32x4_t y0_3_32 = vld1q_f32(y0_3);
        y0_3_32 = vfmaq_n_f32(y0_3_32, vld1q_f32(wvec0_3), x);

        for (unsigned int j = 0; j < cols - idx; ++j) {
          Y32[idx + j] = y0_3_32[j];
        }
      }
    }
  }
  scopy_neon_fp32_to_fp16(cols, Y32, Y);
  delete[] Y32;
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

float32x4_t vcvtq_f32_u32_bitwise(uint32x4_t u32) {
  constexpr uint32_t offsetValue = 0x4b000000;
  const uint32x4_t offsetInt = vdupq_n_u32(offsetValue);
  return vsubq_f32(vreinterpretq_f32_u32(vorrq_u32(u32, offsetInt)),
                   vreinterpretq_f32_u32(offsetInt));
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
    uint8x8_t batch = vld1_u8(&X[idx]);

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

void scopy_neon_int8_to_fp16(const unsigned int N, const uint8_t *X,
                             __fp16 *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    // convert to u16
    uint16x8_t batch_low_u16 = vmovl_u8(low);
    uint16x8_t batch_high_u16 = vmovl_u8(high);

    // todo : experiment with vcvt_f32_u32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f16(&Y[idx], vcvtq_f16_u16(batch_low_u16));
    vst1q_f16(&Y[idx + 8], vcvtq_f16_u16(batch_high_u16));
  }
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    // convert to u16
    uint16x8_t batch_u16 = vmovl_u8(batch);
    vst1q_f16(&Y[idx], vcvtq_f16_u16(batch_u16));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void scopy_neon_int8_to_fp32(const unsigned int N, const uint8_t *X, float *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);
    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    // convert to u16
    uint16x8_t batch_low_u16 = vmovl_u8(low);
    uint16x8_t batch_high_u16 = vmovl_u8(high);

    // convert to u32
    uint32x4_t batch_low_u32_low = vmovl_u16(vget_low_u16(batch_low_u16));
    uint32x4_t batch_low_u32_high = vmovl_u16(vget_high_u16(batch_low_u16));
    uint32x4_t batch_high_u32_low = vmovl_u16(vget_low_u16(batch_high_u16));
    uint32x4_t batch_high_u32_high = vmovl_u16(vget_high_u16(batch_high_u16));

    // todo : experiment with vcvt_f32_u32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f32(&Y[idx], vcvtq_f32_u32(batch_low_u32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_u32(batch_low_u32_high));
    vst1q_f32(&Y[idx + 8], vcvtq_f32_u32(batch_high_u32_low));
    vst1q_f32(&Y[idx + 12], vcvtq_f32_u32(batch_high_u32_high));
  }
  for (; (N - idx) >= 8; idx += 8) {
    uint8x8_t batch = vld1_u8(&X[idx]);

    // convert to u16
    uint16x8_t batch_u16 = vmovl_u8(batch);

    // convert to u32
    uint32x4_t batch_u32_low = vmovl_u16(vget_low_u16(batch_u16));
    uint32x4_t batch_u32_high = vmovl_u16(vget_high_u16(batch_u16));

    vst1q_f32(&Y[idx], vcvtq_f32_u32(batch_u32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_u32(batch_u32_high));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void scopy_neon_fp16_to_fp32(const unsigned int N, const __fp16 *X, float *Y) {
  unsigned int idx = 0;

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
  unsigned int idx = 0;

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
  for (; idx < (size - idx) && (size - idx) >= 8; idx += 8) {
    float16x8_t c = vmulq_n_f16(vld1q_f16(&C[idx]), static_cast<__fp16>(beta));

    vst1q_f32(&C32[idx], vcvt_f32_f16(vget_low_f16(c)));
    vst1q_f32(&C32[idx + 4], vcvt_f32_f16(vget_high_f16(c)));
  }
  // remaining 4
  for (; idx < (size - idx) && (size - idx) >= 4; idx += 4) {
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
  __fp16 valsB[8];
  __fp16 valsA[8];
  int m = 0;
  for (; M - m >= 1; m++) {
    int n = 0;
    __fp16 valsB_2[8];
    __fp16 valsB_3[8];
    __fp16 valsB_4[8];
    for (; N - n >= 4; n += 4) {
      float32x4_t sum = vmovq_n_f32(0.0f);
      float32x4_t sum2 = vmovq_n_f32(0.0f);
      float32x4_t sum3 = vmovq_n_f32(0.0f);
      float32x4_t sum4 = vmovq_n_f32(0.0f);
      unsigned int k = 0;
      for (; (K - k) >= 16; k += 16) {
        float16x8_t a0_7 = vld1q_f16(&A[m * K + k]);
        float16x8_t a8_15 = vld1q_f16(&A[m * K + k + 8]);
        // fp16 multiplication
        float16x8_t ab = vmulq_f16(a0_7, vld1q_f16(&B[n * K + k]));
        float16x8_t ab2 = vmulq_f16(a0_7, vld1q_f16(&B[(n + 1) * K + k]));
        float16x8_t ab3 = vmulq_f16(a0_7, vld1q_f16(&B[(n + 2) * K + k]));
        float16x8_t ab4 = vmulq_f16(a0_7, vld1q_f16(&B[(n + 3) * K + k]));
        float16x8_t ab8_15 = vmulq_f16(a8_15, vld1q_f16(&B[n * K + k + 8]));
        float16x8_t ab2_8_15 =
          vmulq_f16(a8_15, vld1q_f16(&B[(n + 1) * K + k + 8]));
        float16x8_t ab3_8_15 =
          vmulq_f16(a8_15, vld1q_f16(&B[(n + 2) * K + k + 8]));
        float16x8_t ab4_8_15 =
          vmulq_f16(a8_15, vld1q_f16(&B[(n + 3) * K + k + 8]));

        // fp16 partial accumulation
        ab = vaddq_f16(ab, ab8_15);
        ab2 = vaddq_f16(ab2, ab2_8_15);
        ab3 = vaddq_f16(ab3, ab3_8_15);
        ab4 = vaddq_f16(ab4, ab4_8_15);
        // fp32 partial accumulation
        sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(ab)));
        sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(ab)));

        sum2 = vaddq_f32(sum2, vcvt_f32_f16(vget_low_f16(ab2)));
        sum2 = vaddq_f32(sum2, vcvt_f32_f16(vget_high_f16(ab2)));

        sum3 = vaddq_f32(sum3, vcvt_f32_f16(vget_low_f16(ab3)));
        sum3 = vaddq_f32(sum3, vcvt_f32_f16(vget_high_f16(ab3)));

        sum4 = vaddq_f32(sum4, vcvt_f32_f16(vget_low_f16(ab4)));
        sum4 = vaddq_f32(sum4, vcvt_f32_f16(vget_high_f16(ab4)));
      }
      for (; (K - k) >= 8; k += 8) {
        float16x8_t a0_7 = vld1q_f16(&A[m * K + k]);

        float16x8_t ab = vmulq_f16(a0_7, vld1q_f16(&B[n * K + k]));
        float16x8_t ab2 = vmulq_f16(a0_7, vld1q_f16(&B[(n + 1) * K + k]));
        float16x8_t ab3 = vmulq_f16(a0_7, vld1q_f16(&B[(n + 2) * K + k]));
        float16x8_t ab4 = vmulq_f16(a0_7, vld1q_f16(&B[(n + 3) * K + k]));

        sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(ab)));
        sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(ab)));

        sum2 = vaddq_f32(sum2, vcvt_f32_f16(vget_low_f16(ab2)));
        sum2 = vaddq_f32(sum2, vcvt_f32_f16(vget_high_f16(ab2)));

        sum3 = vaddq_f32(sum3, vcvt_f32_f16(vget_low_f16(ab3)));
        sum3 = vaddq_f32(sum3, vcvt_f32_f16(vget_high_f16(ab3)));

        sum4 = vaddq_f32(sum4, vcvt_f32_f16(vget_low_f16(ab4)));
        sum4 = vaddq_f32(sum4, vcvt_f32_f16(vget_high_f16(ab4)));
      }

      // remaining K values
      if (k < K) {
        unsigned int idx;
        for (idx = k; idx < K; idx++) {
          valsA[idx - k] = A[m * K + idx];
          valsB[idx - k] = B[n * K + idx];
          valsB_2[idx - k] = B[(n + 1) * K + idx];
          valsB_3[idx - k] = B[(n + 2) * K + idx];
          valsB_4[idx - k] = B[(n + 3) * K + idx];
        }
        // to cover entire 128 bits (reset unused bits)
        while (idx < (k + 8)) {
          valsA[idx - k] = 0;
          valsB[idx - k] = 0;
          valsB_2[idx - k] = 0;
          valsB_3[idx - k] = 0;
          valsB_4[idx - k] = 0;
          idx++;
        }
        // updating sum
        float16x8_t ab = vmulq_f16(vld1q_f16(valsA), vld1q_f16(valsB));
        float16x8_t ab2 = vmulq_f16(vld1q_f16(valsA), vld1q_f16(valsB_2));
        float16x8_t ab3 = vmulq_f16(vld1q_f16(valsA), vld1q_f16(valsB_3));
        float16x8_t ab4 = vmulq_f16(vld1q_f16(valsA), vld1q_f16(valsB_4));

        sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(ab)));
        sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(ab)));
        sum2 = vaddq_f32(sum2, vcvt_f32_f16(vget_low_f16(ab2)));
        sum2 = vaddq_f32(sum2, vcvt_f32_f16(vget_high_f16(ab2)));
        sum3 = vaddq_f32(sum3, vcvt_f32_f16(vget_low_f16(ab3)));
        sum3 = vaddq_f32(sum3, vcvt_f32_f16(vget_high_f16(ab3)));
        sum4 = vaddq_f32(sum4, vcvt_f32_f16(vget_low_f16(ab4)));
        sum4 = vaddq_f32(sum4, vcvt_f32_f16(vget_high_f16(ab4)));
      }

      if (alpha != 1) {
        sum = vmulq_n_f32(sum, alpha);
        sum2 = vmulq_n_f32(sum2, alpha);
        sum3 = vmulq_n_f32(sum3, alpha);
        sum4 = vmulq_n_f32(sum4, alpha);
      }

      C[m * N + n] += vaddvq_f32(sum);
      C[m * N + n + 1] += vaddvq_f32(sum2);
      C[m * N + n + 2] += vaddvq_f32(sum3);
      C[m * N + n + 3] += vaddvq_f32(sum4);
    }
    for (; N - n >= 1; n++) {
      float32x4_t sum = vmovq_n_f32(0.0f);
      unsigned int k = 0;
      for (; (K - k) >= 16; k += 16) {
        // fp16 multiplication
        float16x8_t ab =
          vmulq_f16(vld1q_f16(&A[m * K + k]), vld1q_f16(&B[n * K + k]));
        float16x8_t ab8_15 =
          vmulq_f16(vld1q_f16(&A[m * K + k + 8]), vld1q_f16(&B[n * K + k + 8]));
        // fp16 partial accumulation
        ab = vaddq_f16(ab, ab8_15);
        // fp32 partial accumulation
        sum = vaddq_f32(sum, vcvt_f32_f16(vget_low_f16(ab)));
        sum = vaddq_f32(sum, vcvt_f32_f16(vget_high_f16(ab)));
      }
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
  unsigned int i = 0;
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
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    float16x8_t z0_7 = vaddq_f16(x0_7, y0_7);

    vst1q_f16(&Z[i], z0_7);
  }
  while (i < N) {
    Z[i] = X[i] + Y[i];
    ++i;
  }
}

void inv_sqrt_inplace_neon(const unsigned int N, __fp16 *X) {
  unsigned int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t x0_7_sqrt = vsqrtq_f16(x0_7);
    float16x8_t ones = vmovq_n_f16(1);
    float16x8_t x0_7_sqrt_div = vdivq_f16(ones, x0_7_sqrt);
    vst1q_f16(&X[i], x0_7_sqrt_div);
  }
  while (i < N) {
    X[i] = (1 / std::sqrt(static_cast<float>(X[i])));
    ++i;
  }
}

#endif
} // namespace nntrainer::neon
