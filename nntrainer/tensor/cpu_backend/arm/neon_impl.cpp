// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file neon_impl.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Single-precision computation functions based on NEON
 *
 */

#include <climits>
#include <fp16.h>
#include <matrix_transpose_neon.h>
#include <memory>
#include <neon_impl.h>
#include <neon_setting.h>
#include <nntrainer_error.h>
#ifdef ARMV7
#include <armv7_neon.h>
#endif
#include <fallback_internal.h>
#include <util_func.h>

namespace nntrainer::neon {
static inline void __ele_qmul_kernel(int8_t *lhs, int8_t *rhs, int8_t *res,
                                     unsigned int data_len,
                                     const float lhs_scale,
                                     const float rhs_scale,
                                     const float res_scale) {
  float32x4_t multiplier = vdupq_n_f32(lhs_scale * rhs_scale / res_scale);
  int8x16_t int8_max = vdupq_n_s8(127);
  int8x16_t int8_min = vdupq_n_s8(-128);
  unsigned int N16 = (data_len >> 4) << 4;
  for (unsigned int n = 0; n < N16; n += 16) {
    int16x8_t lhs0_7 = vmovl_s8(vld1_s8(lhs));
    lhs += 8;
    int16x8_t lhs8_15 = vmovl_s8(vld1_s8(lhs));
    lhs += 8;
    int16x8_t rhs0_7 = vmovl_s8(vld1_s8(rhs));
    rhs += 8;
    int16x8_t rhs8_15 = vmovl_s8(vld1_s8(rhs));
    rhs += 8;

    int32x4_t res0_3 = vmull_s16(vget_low_s16(lhs0_7), vget_low_s16(rhs0_7));
    int32x4_t res4_7 = vmull_s16(vget_high_s16(lhs0_7), vget_high_s16(rhs0_7));
    int32x4_t res8_11 = vmull_s16(vget_low_s16(lhs8_15), vget_low_s16(rhs8_15));
    int32x4_t res12_15 =
      vmull_s16(vget_high_s16(lhs8_15), vget_high_s16(rhs8_15));

    float32x4_t res_f32_0 = vcvtq_f32_s32(res0_3);
    float32x4_t res_f32_1 = vcvtq_f32_s32(res4_7);
    float32x4_t res_f32_2 = vcvtq_f32_s32(res8_11);
    float32x4_t res_f32_3 = vcvtq_f32_s32(res12_15);

    res_f32_0 = vmulq_f32(res_f32_0, multiplier);
    res_f32_1 = vmulq_f32(res_f32_1, multiplier);
    res_f32_2 = vmulq_f32(res_f32_2, multiplier);
    res_f32_3 = vmulq_f32(res_f32_3, multiplier);

    /// @note: currently we use vcvtnq_s32_f32 instead of vcvtq_s32_f32
    res0_3 = vcvtnq_s32_f32(res_f32_0);
    res4_7 = vcvtnq_s32_f32(res_f32_1);
    res8_11 = vcvtnq_s32_f32(res_f32_2);
    res12_15 = vcvtnq_s32_f32(res_f32_3);

    int8x16_t output = vcombine_s8(
      vqmovn_s16(vcombine_s16(vqmovn_s32(res0_3), vqmovn_s32(res4_7))),
      vqmovn_s16(vcombine_s16(vqmovn_s32(res8_11), vqmovn_s32(res12_15))));

    output = vmaxq_s8(output, int8_min);
    output = vminq_s8(output, int8_max);

    vst1q_s8(res, output);
    res += 16;
  }
  for (unsigned int n = N16; n < data_len; ++n) {
    res[n] = std::max(
      -128, std::min(127, (int)std::lround(lhs[n] * rhs[n] * lhs_scale *
                                           rhs_scale / res_scale)));
  }
}

static inline void
__ele_qmul_kernel(int8_t *lhs, int8_t *rhs, int8_t *res, unsigned int data_len,
                  const float *lhs_scale, const float *rhs_scale,
                  const float *res_scale, unsigned int scale_len) {
  std::invalid_argument(
    "Error : __ele_qmul_kernel for vector quantization parameter is NYI.");
}

void ele_qmul(int8_t *lhs, int8_t *rhs, int8_t *res, unsigned int data_len,
              const float *lhs_scale, const float *rhs_scale,
              const float *res_scale, unsigned int scale_len) {
  if (scale_len == 1) {
    return __ele_qmul_kernel(lhs, rhs, res, data_len, lhs_scale[0],
                             rhs_scale[0], res_scale[0]);
  } else {
    return __ele_qmul_kernel(lhs, rhs, res, data_len, lhs_scale, rhs_scale,
                             res_scale, scale_len);
  }
}

bool is_valid(const unsigned int N, const float *X) {
  size_t i = 0;
  float inf_s = std::numeric_limits<float>::infinity();
  float32x4_t inf = vdupq_n_f32(inf_s);
  uint32x4_t zero = vdupq_n_u32(0);

  for (; N - i >= 4; i += 4) {
    float32x4_t vec = vld1q_f32(&X[i]);
    uint32x4_t vcmp = vceqq_f32(vec, vec);

    vcmp = vceqq_u32(vcmp, zero);

    if (vaddvq_u32(vcmp))
      return false;

    vcmp = vceqq_f32(vec, inf);

    if (vaddvq_u32(vcmp))
      return false;
  }

  while (i < N) {
    if (!isFloatValid(X[i])) {
      return false;
    }
    ++i;
  }

  return true;
}

void sgemv(const float *A, const float *X, float *Y, uint32_t rows,
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

      if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
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

      if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
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

      if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
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

void sgemv_transpose(const float *A, const float *X, float *Y, uint32_t rows,
                     uint32_t cols, float alpha, float beta) {
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

void copy_int4_to_fp32(const unsigned int N, const uint8_t *X, float *Y) {

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

void copy_int8_or_int4(const unsigned int N, const uint8_t *X, uint8_t *Y) {
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

void sine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO)
      x0_3 = vmulq_n_f32(x0_3, alpha);
    float32x4_t sinx0_3 = sin_ps(x0_3);
    vst1q_f32(&Y[i], sinx0_3);
  }
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]);
    ++i;
  }
}

void cosine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO)
      x0_3 = vmulq_n_f32(x0_3, alpha);
    float32x4_t cosx0_3 = cos_ps(x0_3);
    vst1q_f32(&Y[i], cosx0_3);
  }
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]);
    ++i;
  }
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
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

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vmulq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = alpha * X[i] * Y[i] + beta * Z[i];
    else
      Z[i] = alpha * X[i] * Y[i];
    ++i;
  }
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vaddq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] + alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] + alpha * Y[i];
    ++i;
  }
}

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vsubq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] - alpha * Y[i] + beta * Z[i];
    else
      Z[i] = X[i] - alpha * Y[i];
    ++i;
  }
}

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta) {
  unsigned int i = 0;
  float32x4_t alpha_vec = vdupq_n_f32(alpha);
  float32x4_t beta_vec = vdupq_n_f32(beta);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    float32x4_t y0_3 = vld1q_f32(&Y[i]);
    if (std::fpclassify(alpha - 1.F) != FP_ZERO) {
      y0_3 = vmulq_f32(y0_3, alpha_vec);
    }
    float32x4_t xy0_3 = vdivq_f32(x0_3, y0_3);
    if (std::abs(beta) > __FLT_MIN__) {
      float32x4_t z0_3 = vmulq_f32(vld1q_f32(&Z[i]), beta_vec);
      vst1q_f32(&Z[i], vaddq_f32(z0_3, xy0_3));
    } else
      vst1q_f32(&Z[i], xy0_3);
  }
  while (i < N) {
    if (std::abs(beta) > __FLT_MIN__)
      Z[i] = X[i] / (alpha * Y[i]) + beta * Z[i];
    else
      Z[i] = X[i] / (alpha * Y[i]);
    ++i;
  }
}

static inline void __scopy_kernel(const float *X, float *Y) {
  vst1q_f32(Y, vld1q_f32(X));
}

void custom_scopy(const unsigned int N, const float *X, const int incX,
                  float *Y, const int incY) {
  unsigned int N4 = (N >> 2) << 2;
  for (unsigned int i = 0; i < N4; i += 4) {
#ifdef __aarch64__
    __asm__ __volatile__("ld1 {v0.4s}, [%1]\n\t"
                         "st1 {v0.4s}, [%0]\n\t"
                         :
                         : "r"(&Y[i]), "r"(&X[i])
                         : "v0", "memory");
#else
    __scopy_kernel(X + i, Y + i);
#endif
  }
  for (unsigned int i = N4; i < N; ++i) {
    Y[i] = X[i];
  }
}

void copy_int8_to_fp32(const unsigned int N, const uint8_t *X, float *Y) {
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

void copy_int8_to_fp32(const unsigned int N, const int8_t *X, float *Y) {
  unsigned int idx = 0;
  for (; (N - idx) >= 16; idx += 16) {
    int8x16_t batch = vld1q_s8(&X[idx]);
    int8x8_t low = vget_low_s8(batch);
    int8x8_t high = vget_high_s8(batch);

    // convert to s16
    int16x8_t batch_low_s16 = vmovl_s8(low);
    int16x8_t batch_high_s16 = vmovl_s8(high);

    // convert to s32
    int32x4_t batch_low_s32_low = vmovl_s16(vget_low_s16(batch_low_s16));
    int32x4_t batch_low_s32_high = vmovl_s16(vget_high_s16(batch_low_s16));
    int32x4_t batch_high_s32_low = vmovl_s16(vget_low_s16(batch_high_s16));
    int32x4_t batch_high_s32_high = vmovl_s16(vget_high_s16(batch_high_s16));

    // todo : experiment with vcvt_f32_s32_ bitwise operation w.r.t.
    // time/accuracy
    vst1q_f32(&Y[idx], vcvtq_f32_s32(batch_low_s32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_s32(batch_low_s32_high));
    vst1q_f32(&Y[idx + 8], vcvtq_f32_s32(batch_high_s32_low));
    vst1q_f32(&Y[idx + 12], vcvtq_f32_s32(batch_high_s32_high));
  }
  for (; (N - idx) >= 8; idx += 8) {
    int8x8_t batch = vld1_s8(&X[idx]);

    // convert to s16
    int16x8_t batch_s16 = vmovl_s8(batch);

    // convert to s32
    int32x4_t batch_s32_low = vmovl_s16(vget_low_s16(batch_s16));
    int32x4_t batch_s32_high = vmovl_s16(vget_high_s16(batch_s16));

    vst1q_f32(&Y[idx], vcvtq_f32_s32(batch_s32_low));
    vst1q_f32(&Y[idx + 4], vcvtq_f32_s32(batch_s32_high));
  }
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y) {
  /// @todo implement int16_t to fp32
  unsigned int idx = 0;
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y) {
  /// @todo implement int16_t to fp32
  unsigned int idx = 0;
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  /// @todo implement int16_t to int16_t
  unsigned int idx = 0;
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  /// @todo implement uint16_t to uint16_t
  unsigned int idx = 0;
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void copy_s8(const unsigned int N, const int8_t *X, int8_t *Y) {
  /// @todo implement int8_t to int8_t
  unsigned int idx = 0;
  for (; (N - idx) >= 1; ++idx) {
    Y[idx] = X[idx];
  }
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  transpose_neon<float>(M, N, src, ld_src, dst, ld_dst);
}

void calc_trigonometric_vals_dup(unsigned int N_half, float *angle, float *cos_,
                                 float *sin_, unsigned int from) {
  cosine(N_half, angle, cos_, from);
  sine(N_half, angle, sin_, from);

  unsigned int N = 2 * N_half;
  unsigned int i = N_half;
  unsigned int i_half = 0;

  for (; (N - i >= 4) && (N_half - i_half >= 4); i += 4, i_half += 4) {
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

void swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
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

float max_val(const unsigned int N, float *X) {
  unsigned int i = 0;
  float ret = X[i];
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    ret = std::fmax(ret, vmaxvq_f32(x0_3));
  }
  while (i < N) {
    ret = std::fmax(ret, X[i]);
    ++i;
  }
  return ret;
}

void softmax(const unsigned int N, float *X, float *Y) {
  unsigned int i = 0;
  float sum = 0.f;
  float max_x = max_val(N, X);
  float32x4_t max_x_v = vmovq_n_f32(max_x);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    x0_3 = vsubq_f32(x0_3, max_x_v);
    float32x4_t exp0_3 = exp_ps(x0_3);
    sum += vaddvq_f32(exp0_3);
  }
  while (i < N) {
    sum += std::exp(X[i] - max_x);
    ++i;
  }

  i = 0;
  float32x4_t sum_vec = vmovq_n_f32(sum);
  for (; N - i >= 4; i += 4) {
    float32x4_t x0_3 = vld1q_f32(&X[i]);
    x0_3 = vsubq_f32(x0_3, max_x_v);
    float32x4_t exp0_3 = exp_ps(x0_3);
    float32x4_t softmax0_3 = vdivq_f32(exp0_3, sum_vec);
    vst1q_f32(&Y[i], softmax0_3);
  }
  while (i < N) {
    Y[i] = std::exp(X[i] - max_x) / sum;
    ++i;
  }
}

void exp_i(const unsigned int N, float *X) {
  unsigned int i = 0;
  for (; N - i >= 4; i += 4) {
    vst1q_f32(&X[i], exp_ps(vld1q_f32(&X[i])));
  }
  while (i < N) {
    X[i] = std::exp(X[i]);
    ++i;
  }
}

void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads) {
  size_t row_range = end_row - start_row;
  const size_t full_blocks = (num_heads / 4) * 4;
  // const size_t remainder = num_heads % 4;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];

  // 1. max
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r)
      max_val = std::max(max_val, qk_out[r * num_heads + c]);
    max_vals[c] = max_val;
  }

  // 2. inplace exp + sum
  for (size_t c = 0; c < full_blocks; c += 4) {
    float32x4_t maxv = vld1q_f32(&max_vals[c]);
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (size_t r = 0; r < row_range; ++r) {
      float *ptr = &qk_out[(start_row + r) * num_heads + c];
      float32x4_t val = vld1q_f32(ptr);
      float32x4_t e = exp_ps(vsubq_f32(val, maxv));
      vst1q_f32(ptr, e); // overwrite qk_out
      sum = vaddq_f32(sum, e);
    }
    vst1q_f32(&sum_vals[c], sum);
  }

  for (size_t c = full_blocks; c < num_heads; ++c) {
    float sum = 0.0f;
    float maxv = max_vals[c];
    for (size_t r = 0; r < row_range; ++r) {
      float &a = qk_out[(start_row + r) * num_heads + c];
      a = std::exp(a - maxv); // overwrite qk_out
      sum += a;
    }
    sum_vals[c] = sum;
  }
  // 3. softmax = exp / sum (inplace)
  for (size_t r = 0; r < row_range; ++r) {
    for (size_t c = 0; c < full_blocks; c += 4) {
      float *ptr = &qk_out[(start_row + r) * num_heads + c];
      float32x4_t val = vld1q_f32(ptr); // already exp(x - max)
      float32x4_t sumv = vld1q_f32(&sum_vals[c]);
      float32x4_t soft = vdivq_f32(val, sumv);
      vst1q_f32(ptr, soft);
    }
    for (size_t c = full_blocks; c < num_heads; ++c) {
      qk_out[(start_row + r) * num_heads + c] /= sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads) {
  const size_t full_block = (num_heads / 4) * 4;

  float *max_vals = new float[num_heads];
  float *sum_vals = new float[num_heads];

  // 1. Find Max along with col
  for (size_t c = 0; c < num_heads; ++c) {
    float max_val = -INFINITY;
    for (size_t r = start_row; r < end_row; ++r) {
      max_val = std::max(max_val, qk_out[r * num_heads + c]);
    }
    max_vals[c] = max_val;
  }

  // 2. Compute sum along with col (exp vectorized)
  for (size_t c = 0; c < full_block; c += 4) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (size_t r = start_row; r < end_row; ++r) {
      float32x4_t val = vld1q_f32(&qk_out[r * num_heads + c]);
      float32x4_t maxv = vld1q_f32(&max_vals[c]);
      float32x4_t sub = vsubq_f32(val, maxv);
      float32x4_t e = exp_ps(sub);
      sum = vaddq_f32(sum, e);
    }
    vst1q_f32(&sum_vals[c], sum);
  }

  for (size_t c = full_block; c < num_heads; ++c) {
    float sum = 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      sum += std::exp(qk_out[r * num_heads + c] - max_vals[c]);
    }
    sum_vals[c] = sum;
  }

  // 3. apply softmax
  for (size_t r = start_row; r < end_row; ++r) {
    for (size_t c = 0; c < full_block; c += 4) {
      float32x4_t val = vld1q_f32(&qk_out[r * num_heads + c]);
      float32x4_t maxv = vld1q_f32(&max_vals[c]);
      float32x4_t sub = vsubq_f32(val, maxv);
      float32x4_t e = exp_ps(sub);
      float32x4_t sumv = vld1q_f32(&sum_vals[c]);
      float32x4_t softmax = vdivq_f32(e, sumv);
      vst1q_f32(&qk_out[r * num_heads + c], softmax);
    }
    for (size_t c = full_block; c < num_heads; ++c) {
      qk_out[r * num_heads + c] =
        std::exp(qk_out[r * num_heads + c] - max_vals[c]) / sum_vals[c];
    }
  }

  delete[] max_vals;
  delete[] sum_vals;
}

static inline float convert_scalar_f16_to_f32(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

void compute_fp16vcache_fp32_transposed(int iter, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int seq, int num_cache_head,
                                        int gqa_size, int head_dim,
                                        bool process_all) {
#ifdef ARMV7
  __fallback_compute_fp16vcache_fp32_transposed(iter, in, vcache, output, seq,
                                                num_cache_head, gqa_size,
                                                head_dim, process_all);
#else
  std::vector<float> tmp_fp32(head_dim);
  int a_row_start =
    process_all ? ((iter * (iter + 1)) / 2) * num_cache_head * gqa_size : 0;
  int out_offset = process_all ? iter : 0;

  for (int n = 0; n < num_cache_head; ++n) {
    int num_blocks = head_dim / 4;
    int rem = head_dim % 4;

    std::vector<float32x4_t> sumVec(num_blocks * gqa_size, vdupq_n_f32(0.0f));
    std::vector<float> sumRem(gqa_size * rem, 0.0f);

    for (int j = 0; j <= iter; ++j) {
      if (j + 1 < seq) {
        const uint16_t *next_vptr =
          vcache + ((j + 1) * num_cache_head + n) * head_dim;
        //_mm_prefetch(reinterpret_cast<const char *>(next_vptr), _MM_HINT_T0);
        __builtin_prefetch(reinterpret_cast<const char *>(next_vptr), 0,
                           3); // READ, L1 load
      }

      const uint16_t *vptr = vcache + (j * num_cache_head + n) * head_dim;

      int d0 = 0;
      for (; d0 + 4 <= head_dim; d0 += 4) {
        float16x4_t half =
          vld1_f16(reinterpret_cast<const float16_t *>(vptr + d0));
        float32x4_t f32 = vcvt_f32_f16(half);
        vst1q_f32(&tmp_fp32[d0], f32);
      }
      for (; d0 < head_dim; ++d0) {
        tmp_fp32[d0] = convert_scalar_f16_to_f32(vptr[d0]);
      }

      for (int h = 0; h < gqa_size; ++h) {
        // float a_val = in[a_row_start + (j * gqa_size + h) * num_cache_head +
        // n];
        float a_val =
          in[a_row_start + (j * gqa_size * num_cache_head + n * gqa_size + h)];
        float32x4_t inVec = vdupq_n_f32(a_val);

        for (int b = 0; b < num_blocks; ++b) {
          float32x4_t bVec = vld1q_f32(&tmp_fp32[b * 4]);
          sumVec[h * num_blocks + b] =
            vfmaq_f32(sumVec[h * num_blocks + b], inVec, bVec);
        }

        float *remPtr = &sumRem.data()[h * rem];
        int base = num_blocks * 4;
        for (int r = 0; r < rem; ++r) {
          remPtr[r] += a_val * tmp_fp32[base + r];
        }
      }
    }

    for (int h = 0; h < gqa_size; ++h) {
      for (int b = 0; b < num_blocks; ++b) {
        int out_base =
          ((out_offset * num_cache_head + n) * gqa_size + h) * head_dim + b * 4;
        vst1q_f32(&output[out_base], sumVec[h * num_blocks + b]);
      }

      float *remPtr = &sumRem.data()[h * rem];
      // float *remPtr = &sumRem[h * rem];
      int base = num_blocks * 4;
      for (int r = 0; r < rem; ++r) {
        int out_idx =
          ((out_offset * num_cache_head + n) * gqa_size + h) * head_dim + base +
          r;
        output[out_idx] = remPtr[r];
      }
    }
  }
#endif
}

#if !defined(ARMV7)
static inline float32x4_t load_fp16_4_NEON(const uint16_t *src) {
  float16x4_t in = vld1_f16(reinterpret_cast<const float16_t *>(src));
  return vcvt_f32_f16(in);
}

static inline void load_fp16_4_to_chunk_NEON(const uint16_t *b_src,
                                             float *temp_row, int chunk_size) {
  int i = 0;
  for (; i + 4 <= chunk_size; i += 4) {
    float32x4_t f32 = load_fp16_4_NEON((const uint16_t *)(b_src + i));
    vst1q_f32(temp_row + i, f32);
  }
  for (; i < chunk_size; ++i) {
    temp_row[i] = convert_scalar_f16_to_f32(b_src[i]);
  }
}
#endif

template <>
void compute_kcaches(const float *A, const uint16_t *B, float *output,
                     int num_rows, int N, int chunk_size, int group_size,
                     int tile_size) {
#ifdef ARMV7
  __fallback_compute_kcaches<uint16_t>(A, B, output, num_rows, N, chunk_size,
                                       group_size, tile_size);
#else
  using BType = uint16_t;
  int row_stride = N * chunk_size;
  const int group_stride = group_size * chunk_size;
  const int tile_count = (num_rows + tile_size - 1) / tile_size;

  // FP32 Cache Buffer
  thread_local std::vector<float> temp_tile_buf(tile_size * chunk_size);

  for (int n = 0; n < N; ++n) {
    for (int t = 0; t < tile_count; ++t) {
      int row_tile_start = t * tile_size;
      int tile_rows = std::min(tile_size, num_rows - row_tile_start);

      // FP16 to FP32 Conversion : preprocessing (row unit)
      if constexpr (!std::is_same<BType, float>::value) {
        for (int row = 0; row < tile_rows; ++row) {
          const BType *b_src =
            B + (row_tile_start + row) * row_stride + n * chunk_size;
          float *dst = temp_tile_buf.data() + row * chunk_size;
          load_fp16_4_to_chunk_NEON(b_src, dst, chunk_size);
        }
      }

      for (int g = 0; g < group_size; ++g) {
        const float *a_ptr = A + n * group_stride + g * chunk_size;
        for (int row = 0; row < tile_rows; ++row) {
          const float *b_row;
          if constexpr (std::is_same<BType, float>::value) {
            b_row = reinterpret_cast<const float *>(
              B + (row_tile_start + row) * row_stride + n * chunk_size);
          } else {
            b_row = temp_tile_buf.data() + row * chunk_size;
          }

          float sum = 0.0f;
          int i = 0;
          float32x4_t acc = vdupq_n_f32(0.0f);
          for (; i + 4 <= chunk_size; i += 4) {
            float32x4_t va = vld1q_f32(a_ptr + i);
            float32x4_t vb = vld1q_f32(b_row + i);
            acc = vfmaq_f32(acc, va, vb);
          }

          acc = vpaddq_f32(acc, acc);
          acc = vpaddq_f32(acc, acc);
          sum += vgetq_lane_f32(acc, 0);

          for (; i < chunk_size; ++i)
            sum += a_ptr[i] * b_row[i];

          output[(row_tile_start + row) * N * group_size + n * group_size + g] =
            sum / sqrt((float)chunk_size);
        }
      }
    }
  }
#endif
}

#define COMPUTE_FP32_TO_FP16(x) vcvt_f16_f32(vdupq_n_f32(x))[0]

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
#ifdef ARMV7
  __fallback_compute_rotary_emb_value(width, dim, half_, inout, output, cos_,
                                      sin_, only_convert_to_fp16);
#else
  enum class OutputType { FP16, FP32 };

  OutputType out_type = OutputType::FP32;
  if (output != nullptr)
    out_type = OutputType::FP16;

  for (unsigned int w = 0; w < width; w += dim) {
    unsigned int k = 0;
    for (; k + 3 < half_; k += 4) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      float32x4_t a = vld1q_f32(&inout[i0]);
      float32x4_t b = vld1q_f32(&inout[i1]);

      if (only_convert_to_fp16) {
        if (out_type == OutputType::FP16) {
          float16x4_t a_fp16 = vcvt_f16_f32(a);
          float16x4_t b_fp16 = vcvt_f16_f32(b);

          vst1_f16(static_cast<__fp16 *>(output) + i0, a_fp16);
          vst1_f16(static_cast<__fp16 *>(output) + i1, b_fp16);
        }

      } else {
        float32x4_t cos_v = vld1q_f32(&cos_[k]);
        float32x4_t sin_v = vld1q_f32(&sin_[k]);

        float32x4_t out0 = vsubq_f32(vmulq_f32(a, cos_v), vmulq_f32(b, sin_v));
        float32x4_t out1 = vaddq_f32(vmulq_f32(a, sin_v), vmulq_f32(b, cos_v));

        if (out_type == OutputType::FP16) {
          float16x4_t out0_fp16 = vcvt_f16_f32(out0);
          float16x4_t out1_fp16 = vcvt_f16_f32(out1);

          vst1_f16(static_cast<__fp16 *>(output) + i0, out0_fp16);
          vst1_f16(static_cast<__fp16 *>(output) + i1, out1_fp16);
        } else if (out_type == OutputType::FP32) {
          vst1q_f32(&inout[i0], out0);
          vst1q_f32(&inout[i1], out1);
        }
      }
    }

    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      // assert(i1 < width && "Scalar i1 overflow!");
      float a = inout[i0];
      float b = inout[i1];

      if (only_convert_to_fp16) {
        static_cast<__fp16 *>(output)[i0] = COMPUTE_FP32_TO_FP16(a);
        static_cast<__fp16 *>(output)[i1] = COMPUTE_FP32_TO_FP16(b);

      } else {

        float c = cos_[k];
        float s = sin_[k];

        float out0 = a * c - b * s;
        float out1 = a * s + b * c;

        if (out_type == OutputType::FP16) {
          static_cast<__fp16 *>(output)[i0] = COMPUTE_FP32_TO_FP16(out0);
          static_cast<__fp16 *>(output)[i1] = COMPUTE_FP32_TO_FP16(out1);
        } else if (out_type == OutputType::FP32) {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
  }
#endif
}

} // namespace nntrainer::neon
