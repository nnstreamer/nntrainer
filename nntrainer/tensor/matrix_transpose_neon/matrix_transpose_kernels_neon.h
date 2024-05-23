// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   matrix_transpose_kernels_neon.h
 * @date   09 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  These are internal util functions for transposing matrix with NEON
 *
 */

#include <arm_neon.h>
#include <cassert>
#include <cstdint>
#include <mask_neon.h>

#define TRANSPOSE_FP16_4x4(row0, row1, row2, row3)                           \
  do {                                                                       \
    float16x4x2_t row01 = vtrn_f16(row0, row1);                              \
    float16x4x2_t row23 = vtrn_f16(row2, row3);                              \
    row0 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(row01.val[0])),    \
                                vget_low_f32(vcvt_f32_f16(row23.val[0]))));  \
    row1 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(row01.val[1])),    \
                                vget_low_f32(vcvt_f32_f16(row23.val[1]))));  \
    row2 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_high_f32(vcvt_f32_f16(row01.val[0])),   \
                                vget_high_f32(vcvt_f32_f16(row23.val[0])))); \
    row3 =                                                                   \
      vcvt_f16_f32(vcombine_f32(vget_high_f32(vcvt_f32_f16(row01.val[1])),   \
                                vget_high_f32(vcvt_f32_f16(row23.val[1])))); \
  } while (0)
/**
 * @brief 4x4 sized kernel for matrix transpose in NEON
 *
 * @param src __fp16* source data
 * @param ld_src col length of src
 * @param dst __fp16* destination data
 * @param ld_dst col length of dst
 */
static inline void transpose_kernel_4x4_neon(const __fp16 *src,
                                             unsigned int ld_src, __fp16 *dst,
                                             unsigned int ld_dst) {
  float16x4_t a = vld1_f16(&src[0 * ld_src]);
  float16x4_t b = vld1_f16(&src[1 * ld_src]);
  float16x4_t c = vld1_f16(&src[2 * ld_src]);
  float16x4_t d = vld1_f16(&src[3 * ld_src]);

  TRANSPOSE_FP16_4x4(a, b, c, d);

  vst1_f16(&dst[0 * ld_dst], a);
  vst1_f16(&dst[1 * ld_dst], b);
  vst1_f16(&dst[2 * ld_dst], c);
  vst1_f16(&dst[3 * ld_dst], d);
}

/**
 * @brief general case mxn sized matrix transpose kernel with 128 bit SIMD
 * register
 *
 * @tparam M leftover size for row direction
 * @param N leftover size for col direction
 * @param src __fp16* source data
 * @param ld_src col length of src
 * @param dst __fp16* destination data
 * @param ld_dst col length of dst
 */
template <unsigned int M>
static void transpose_kernel_mxn_neon_128(unsigned int N, const __fp16 *src,
                                          unsigned int ld_src, __fp16 *dst,
                                          unsigned int ld_dst) {

  uint16x4_t bitmask_v8 =
    vld1_u16(reinterpret_cast<const uint16_t *>(masks[N]));
  float16x4_t input[4];
  float16x4_t ZEROS = vmov_n_f16(0.F);

  unsigned i;
  for (i = 0; i < M; ++i) {
    input[i] = vbsl_f16(bitmask_v8, vld1_f16(&src[i * ld_src]), ZEROS);
  }
  for (; i < 4; ++i) {
    input[i] = vmov_n_f16(0.F);
  }

  float16x4_t temp[4];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = vzip1_f16(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = vzip2_f16(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 4; ++i) {
    temp[i] = vmov_n_f16(0.F);
  }

  bitmask_v8 = vld1_u16(reinterpret_cast<const uint16_t *>(masks[M]));
  for (i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      input[i] =
        vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(temp[i / 2])),
                                  vget_low_f32(vcvt_f32_f16(temp[2 + i / 2]))));
    } else {
      input[i] = vcvt_f16_f32(
        vcombine_f32(vget_high_f32(vcvt_f32_f16(temp[i / 2])),
                     vget_high_f32(vcvt_f32_f16(temp[2 + i / 2]))));
    }
    vst1_f16(&dst[i * ld_dst],
             vbsl_f16(bitmask_v8, input[i], vld1_f16(&dst[i * ld_dst])));
  }
}
/**
 * @brief 8x8 sized kernel for matrix transpose in NEON
 *
 * @param src __fp16* source data
 * @param ld_src col length of src
 * @param dst __fp16* destination data
 * @param ld_dst col length of dst
 */
static inline void transpose_kernel_8x8_neon(const __fp16 *src,
                                             unsigned int ld_src, __fp16 *dst,
                                             unsigned int ld_dst) {
  float16x8_t a = vld1q_f16(&src[0 * ld_src]);
  float16x8_t b = vld1q_f16(&src[1 * ld_src]);
  float16x8_t c = vld1q_f16(&src[2 * ld_src]);
  float16x8_t d = vld1q_f16(&src[3 * ld_src]);
  float16x8_t e = vld1q_f16(&src[4 * ld_src]);
  float16x8_t f = vld1q_f16(&src[5 * ld_src]);
  float16x8_t g = vld1q_f16(&src[6 * ld_src]);
  float16x8_t h = vld1q_f16(&src[7 * ld_src]);

  float16x8_t ab0145, ab2367, cd0145, cd2367, ef0145, ef2367, gh0145, gh2367;
  float16x8_t abcd04, abcd15, efgh04, efgh15, abcd26, abcd37, efgh26, efgh37;

  ab0145 = vcombine_f16(vzip1_f16(vget_low_f16(a), vget_low_f16(b)),
                        vzip1_f16(vget_high_f16(a), vget_high_f16(b)));
  ab2367 = vcombine_f16(vzip2_f16(vget_low_f16(a), vget_low_f16(b)),
                        vzip2_f16(vget_high_f16(a), vget_high_f16(b)));
  cd0145 = vcombine_f16(vzip1_f16(vget_low_f16(c), vget_low_f16(d)),
                        vzip1_f16(vget_high_f16(c), vget_high_f16(d)));
  cd2367 = vcombine_f16(vzip2_f16(vget_low_f16(c), vget_low_f16(d)),
                        vzip2_f16(vget_high_f16(c), vget_high_f16(d)));
  ef0145 = vcombine_f16(vzip1_f16(vget_low_f16(e), vget_low_f16(f)),
                        vzip1_f16(vget_high_f16(e), vget_high_f16(f)));
  ef2367 = vcombine_f16(vzip2_f16(vget_low_f16(e), vget_low_f16(f)),
                        vzip2_f16(vget_high_f16(e), vget_high_f16(f)));
  gh0145 = vcombine_f16(vzip1_f16(vget_low_f16(g), vget_low_f16(h)),
                        vzip1_f16(vget_high_f16(g), vget_high_f16(h)));
  gh2367 = vcombine_f16(vzip2_f16(vget_low_f16(g), vget_low_f16(h)),
                        vzip2_f16(vget_high_f16(g), vget_high_f16(h)));

  uint16x8_t shuffle_mask =
    vld1q_u16(reinterpret_cast<const uint16_t *>(shuffle_masks));
  abcd04 = vbslq_f16(shuffle_mask, ab0145, vextq_f16(cd0145, cd0145, 6));
  abcd15 = vbslq_f16(shuffle_mask, vextq_f16(ab0145, ab0145, 2), cd0145);

  efgh04 = vbslq_f16(shuffle_mask, ef0145, vextq_f16(gh0145, gh0145, 6));
  efgh15 = vbslq_f16(shuffle_mask, vextq_f16(ef0145, ef0145, 2), gh0145);

  abcd26 = vbslq_f16(shuffle_mask, ab2367, vextq_f16(cd2367, cd2367, 6));
  abcd37 = vbslq_f16(shuffle_mask, vextq_f16(ab2367, ab2367, 2), cd2367);

  efgh26 = vbslq_f16(shuffle_mask, ef2367, vextq_f16(gh2367, gh2367, 6));
  efgh37 = vbslq_f16(shuffle_mask, vextq_f16(ef2367, ef2367, 2), gh2367);

  a = vcombine_f16(vget_low_f16(abcd04), vget_low_f16(efgh04));
  b = vcombine_f16(vget_low_f16(abcd15), vget_low_f16(efgh15));
  c = vcombine_f16(vget_low_f16(abcd26), vget_low_f16(efgh26));
  d = vcombine_f16(vget_low_f16(abcd37), vget_low_f16(efgh37));
  e = vcombine_f16(vget_high_f16(abcd04), vget_high_f16(efgh04));
  f = vcombine_f16(vget_high_f16(abcd15), vget_high_f16(efgh15));
  g = vcombine_f16(vget_high_f16(abcd26), vget_high_f16(efgh26));
  h = vcombine_f16(vget_high_f16(abcd37), vget_high_f16(efgh37));

  vst1q_f16(&dst[0 * ld_dst], a);
  vst1q_f16(&dst[1 * ld_dst], b);
  vst1q_f16(&dst[2 * ld_dst], c);
  vst1q_f16(&dst[3 * ld_dst], d);
  vst1q_f16(&dst[4 * ld_dst], e);
  vst1q_f16(&dst[5 * ld_dst], f);
  vst1q_f16(&dst[6 * ld_dst], g);
  vst1q_f16(&dst[7 * ld_dst], h);
}
/**
 * @brief general case mxn sized matrix transpose kernel with 256 bit SIMD
 * register
 *
 * @tparam M leftover size for row direction
 * @param N leftover size for col direction
 * @param src __fp16* source data
 * @param ld_src col length of src
 * @param dst __fp16* destination data
 * @param ld_dst col length of dst
 */
template <unsigned int M>
static void transpose_kernel_mxn_neon_256(unsigned int N, const __fp16 *src,
                                          unsigned int ld_src, __fp16 *dst,
                                          unsigned int ld_dst) {
  float16x8_t ZEROS = vmovq_n_f16(0.F);
  uint16x8_t bitmask_v8 =
    vld1q_u16(reinterpret_cast<const uint16_t *>(neon_16bit_masks[N]));
  float16x8_t input[8];
  unsigned i;
  for (i = 0; i < M; ++i) {
    input[i] = vbslq_f16(bitmask_v8, vld1q_f16(&src[i * ld_src]), ZEROS);
  }
  for (; i < 8; ++i) {
    input[i] = ZEROS;
  }
  float16x8_t temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = vcombine_f16(
      vzip1_f16(vget_low_f16(input[2 * i]), vget_low_f16(input[2 * i + 1])),
      vzip1_f16(vget_high_f16(input[2 * i]), vget_high_f16(input[2 * i + 1])));
    temp[2 * i + 1] = vcombine_f16(
      vzip2_f16(vget_low_f16(input[2 * i]), vget_low_f16(input[2 * i + 1])),
      vzip2_f16(vget_high_f16(input[2 * i]), vget_high_f16(input[2 * i + 1])));
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = ZEROS;
  }

  uint16x8_t shuffle_mask =
    vld1q_u16(reinterpret_cast<const uint16_t *>(shuffle_masks));
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = vbslq_f16(shuffle_mask, temp[4 * i],
                             vextq_f16(temp[4 * i + 2], temp[4 * i + 2], 6));
    input[4 * i + 1] = vbslq_f16(
      shuffle_mask, vextq_f16(temp[4 * i], temp[4 * i], 2), temp[4 * i + 2]);
    input[4 * i + 2] =
      vbslq_f16(shuffle_mask, temp[4 * i + 1],
                vextq_f16(temp[4 * i + 3], temp[4 * i + 3], 6));
    input[4 * i + 3] =
      vbslq_f16(shuffle_mask, vextq_f16(temp[4 * i + 1], temp[4 * i + 1], 2),
                temp[4 * i + 3]);
  }
  bitmask_v8 =
    vld1q_u16(reinterpret_cast<const uint16_t *>(neon_16bit_masks[M]));
  for (i = 0; i < N; ++i) {
    if (i < 4) {
      temp[i] =
        vcombine_f16(vget_low_f16(input[i]), vget_low_f16(input[4 + i]));
    } else {
      temp[i] =
        vcombine_f16(vget_high_f16(input[i - 4]), vget_high_f16(input[i]));
    }
    vst1q_f16(&dst[i * ld_dst],
              vbslq_f16(bitmask_v8, temp[i], vld1q_f16(&dst[i * ld_dst])));
  }
}
