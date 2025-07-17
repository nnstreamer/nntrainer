// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_4x8.cpp
 * @date   03 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x8 kernel
 *
 */

#include <arm_neon.h>
#include <assert.h>
#include <hgemm_kernel.h>
#include <hgemm_util.h>
#include <stdlib.h>
#ifdef ARMV7
#include <armv7_neon.h>
#endif

#define INIT_KERNEL_4X8()  \
  do {                     \
    v0 = vdupq_n_f16(0.F); \
    v3 = vdupq_n_f16(0.F); \
    v6 = vdupq_n_f16(0.F); \
    v9 = vdupq_n_f16(0.F); \
  } while (0)

#define KERNEL_4x8_ACC_N4(N)                \
  do {                                      \
    for (int i = 0; i < N; i += 4) {        \
      dv0 = vld1_f16(a + 4 * i);            \
      v24 = vld1q_f16(b + 8 * i);           \
      v0 = vfmaq_lane_f16(v0, v24, dv0, 0); \
      v3 = vfmaq_lane_f16(v3, v24, dv0, 1); \
      v6 = vfmaq_lane_f16(v6, v24, dv0, 2); \
      v9 = vfmaq_lane_f16(v9, v24, dv0, 3); \
      dv1 = vld1_f16(a + 4 * i + 4);        \
      v25 = vld1q_f16(b + 8 * i + 8);       \
      v0 = vfmaq_lane_f16(v0, v25, dv1, 0); \
      v3 = vfmaq_lane_f16(v3, v25, dv1, 1); \
      v6 = vfmaq_lane_f16(v6, v25, dv1, 2); \
      v9 = vfmaq_lane_f16(v9, v25, dv1, 3); \
      dv2 = vld1_f16(a + 4 * i + 8);        \
      v26 = vld1q_f16(b + 8 * i + 16);      \
      v0 = vfmaq_lane_f16(v0, v26, dv2, 0); \
      v3 = vfmaq_lane_f16(v3, v26, dv2, 1); \
      v6 = vfmaq_lane_f16(v6, v26, dv2, 2); \
      v9 = vfmaq_lane_f16(v9, v26, dv2, 3); \
      dv3 = vld1_f16(a + 4 * i + 12);       \
      v27 = vld1q_f16(b + 8 * i + 24);      \
      v0 = vfmaq_lane_f16(v0, v27, dv3, 0); \
      v3 = vfmaq_lane_f16(v3, v27, dv3, 1); \
      v6 = vfmaq_lane_f16(v6, v27, dv3, 2); \
      v9 = vfmaq_lane_f16(v9, v27, dv3, 3); \
    }                                       \
    l += N;                                 \
    __builtin_prefetch(b + 8 * N, 0, 3);    \
    __builtin_prefetch(a + 4 * N, 0, 3);    \
    b += 8 * N;                             \
    a += 4 * N;                             \
  } while (0)

#define KERNEL_4x8_ACC1()                 \
  do {                                    \
    dv0 = vld1_f16(a);                    \
    v24 = vld1q_f16(b);                   \
    v0 = vfmaq_lane_f16(v0, v24, dv0, 0); \
    v3 = vfmaq_lane_f16(v3, v24, dv0, 1); \
    v6 = vfmaq_lane_f16(v6, v24, dv0, 2); \
    v9 = vfmaq_lane_f16(v9, v24, dv0, 3); \
    l += 1;                               \
    __builtin_prefetch(b + 8, 0, 3);      \
    __builtin_prefetch(a + 4, 0, 3);      \
    b += 8 * 1;                           \
    a += 4 * 1;                           \
  } while (0)

#define SAVE_KERNEL_4X8_F16_F32()                                             \
  do {                                                                        \
    vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0))));    \
    vst1q_f32(c + ldc,                                                        \
              vaddq_f32(vld1q_f32(c + ldc), vcvt_f32_f16(vget_low_f16(v3)))); \
    vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),                  \
                                     vcvt_f32_f16(vget_low_f16(v6))));        \
    vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),                  \
                                     vcvt_f32_f16(vget_low_f16(v9))));        \
                                                                              \
    vst1q_f32(c + 4,                                                          \
              vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v0))));  \
    vst1q_f32(c + 4 + ldc, vaddq_f32(vld1q_f32(c + 4 + ldc),                  \
                                     vcvt_f32_f16(vget_high_f16(v3))));       \
    vst1q_f32(c + 4 + 2 * ldc, vaddq_f32(vld1q_f32(c + 4 + 2 * ldc),          \
                                         vcvt_f32_f16(vget_high_f16(v6))));   \
    vst1q_f32(c + 4 + 3 * ldc, vaddq_f32(vld1q_f32(c + 4 + 3 * ldc),          \
                                         vcvt_f32_f16(vget_high_f16(v9))));   \
  } while (0)

template <>
void hgemm_kernel_4x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 8 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int K8 = (K >> 3) << 3;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 4) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0, v3, v6, v9;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x4_t dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7;
      INIT_KERNEL_4X8();
      l = 0;
      for (; l < K;) {
        KERNEL_4x8_ACC1();
      }
      vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0));
      vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v3));
      vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v6));
      vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v9));
      c += 8;
      a -= 4 * K;
    }
    sc += ldc * 4;
    c = sc;
    a += 4 * K;
    b = sb;
  }
}

template <>
void hgemm_kernel_4x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 8 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int K4 = get_prev_mltpl_of_2p_n(K, 2);
  unsigned int K8 = get_prev_mltpl_of_2p_n(K, 3);
  unsigned int K16 = get_prev_mltpl_of_2p_n(K, 4);
  unsigned int i, j, l;
  for (i = 0; i < M; i += 4) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0, v3, v6, v9;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x4_t dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7;
      l = 0;
      for (; l < K16;) {
        INIT_KERNEL_4X8();
        KERNEL_4x8_ACC_N4(16);
        SAVE_KERNEL_4X8_F16_F32();
      }
      for (; l < K8;) {
        INIT_KERNEL_4X8();
        KERNEL_4x8_ACC_N4(8);
        SAVE_KERNEL_4X8_F16_F32();
      }
      for (; l < K4;) {
        INIT_KERNEL_4X8();
        KERNEL_4x8_ACC_N4(4);
        SAVE_KERNEL_4X8_F16_F32();
      }
      for (; l < K;) {
        INIT_KERNEL_4X8();
        KERNEL_4x8_ACC1();
        SAVE_KERNEL_4X8_F16_F32();
      }
      c += 8;
      a -= 4 * K;
    }
    sc += ldc * 4;
    c = sc;
    a += 4 * K;
    b = sb;
  }
}
