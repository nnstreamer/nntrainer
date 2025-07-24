// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x8.cpp
 * @date   01 April 2024
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

#define INIT_KERNEL_8x8()   \
  do {                      \
    v24 = vdupq_n_f16(0.F); \
    v25 = vdupq_n_f16(0.F); \
    v26 = vdupq_n_f16(0.F); \
    v27 = vdupq_n_f16(0.F); \
    v28 = vdupq_n_f16(0.F); \
    v29 = vdupq_n_f16(0.F); \
    v30 = vdupq_n_f16(0.F); \
    v31 = vdupq_n_f16(0.F); \
  } while (0)

#define KERNEL_8x8_ACC_N4(N)                   \
  do {                                         \
    for (int i = 0; i < N; i += 4) {           \
      va0 = vld1q_f16(a + 8 * i);              \
      v16 = vld1q_f16(b + 8 * i);              \
      v24 = vfmaq_laneq_f16(v24, v16, va0, 0); \
      v25 = vfmaq_laneq_f16(v25, v16, va0, 1); \
      v26 = vfmaq_laneq_f16(v26, v16, va0, 2); \
      v27 = vfmaq_laneq_f16(v27, v16, va0, 3); \
      v28 = vfmaq_laneq_f16(v28, v16, va0, 4); \
      v29 = vfmaq_laneq_f16(v29, v16, va0, 5); \
      v30 = vfmaq_laneq_f16(v30, v16, va0, 6); \
      v31 = vfmaq_laneq_f16(v31, v16, va0, 7); \
      va1 = vld1q_f16(a + 8 * i + 8);          \
      v17 = vld1q_f16(b + 8 * i + 8);          \
      v24 = vfmaq_laneq_f16(v24, v17, va1, 0); \
      v25 = vfmaq_laneq_f16(v25, v17, va1, 1); \
      v26 = vfmaq_laneq_f16(v26, v17, va1, 2); \
      v27 = vfmaq_laneq_f16(v27, v17, va1, 3); \
      v28 = vfmaq_laneq_f16(v28, v17, va1, 4); \
      v29 = vfmaq_laneq_f16(v29, v17, va1, 5); \
      v30 = vfmaq_laneq_f16(v30, v17, va1, 6); \
      v31 = vfmaq_laneq_f16(v31, v17, va1, 7); \
      va2 = vld1q_f16(a + 8 * i + 16);         \
      v18 = vld1q_f16(b + 8 * i + 16);         \
      v24 = vfmaq_laneq_f16(v24, v18, va2, 0); \
      v25 = vfmaq_laneq_f16(v25, v18, va2, 1); \
      v26 = vfmaq_laneq_f16(v26, v18, va2, 2); \
      v27 = vfmaq_laneq_f16(v27, v18, va2, 3); \
      v28 = vfmaq_laneq_f16(v28, v18, va2, 4); \
      v29 = vfmaq_laneq_f16(v29, v18, va2, 5); \
      v30 = vfmaq_laneq_f16(v30, v18, va2, 6); \
      v31 = vfmaq_laneq_f16(v31, v18, va2, 7); \
      va3 = vld1q_f16(a + 8 * i + 24);         \
      v19 = vld1q_f16(b + 8 * i + 24);         \
      v24 = vfmaq_laneq_f16(v24, v19, va3, 0); \
      v25 = vfmaq_laneq_f16(v25, v19, va3, 1); \
      v26 = vfmaq_laneq_f16(v26, v19, va3, 2); \
      v27 = vfmaq_laneq_f16(v27, v19, va3, 3); \
      v28 = vfmaq_laneq_f16(v28, v19, va3, 4); \
      v29 = vfmaq_laneq_f16(v29, v19, va3, 5); \
      v30 = vfmaq_laneq_f16(v30, v19, va3, 6); \
      v31 = vfmaq_laneq_f16(v31, v19, va3, 7); \
    }                                          \
    __builtin_prefetch(b + 8 * N, 0, 3);       \
    __builtin_prefetch(a + 8 * N, 0, 3);       \
    l += N;                                    \
    b += 8 * N;                                \
    a += 8 * N;                                \
  } while (0)

#define KERNEL_8x8_ACC1()                    \
  do {                                       \
    va0 = vld1q_f16(a);                      \
    v16 = vld1q_f16(b);                      \
    v24 = vfmaq_laneq_f16(v24, v16, va0, 0); \
    v25 = vfmaq_laneq_f16(v25, v16, va0, 1); \
    v26 = vfmaq_laneq_f16(v26, v16, va0, 2); \
    v27 = vfmaq_laneq_f16(v27, v16, va0, 3); \
    v28 = vfmaq_laneq_f16(v28, v16, va0, 4); \
    v29 = vfmaq_laneq_f16(v29, v16, va0, 5); \
    v30 = vfmaq_laneq_f16(v30, v16, va0, 6); \
    v31 = vfmaq_laneq_f16(v31, v16, va0, 7); \
    __builtin_prefetch(b + 8, 0, 3);         \
    __builtin_prefetch(a + 8, 0, 3);         \
    l += 1;                                  \
    b += 8 * 1;                              \
    a += 8 * 1;                              \
  } while (0)

#define SAVE_KERNEL_8X8_F16_f32()                                              \
  do {                                                                         \
    vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v24))));    \
    vst1q_f32(c + 4,                                                           \
              vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v24))));  \
                                                                               \
    vst1q_f32(c + ldc,                                                         \
              vaddq_f32(vld1q_f32(c + ldc), vcvt_f32_f16(vget_low_f16(v25)))); \
    vst1q_f32(c + 4 + ldc, vaddq_f32(vld1q_f32(c + 4 + ldc),                   \
                                     vcvt_f32_f16(vget_high_f16(v25))));       \
                                                                               \
    vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v26))));        \
    vst1q_f32(c + 4 + 2 * ldc, vaddq_f32(vld1q_f32(c + 4 + 2 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v26))));   \
                                                                               \
    vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v27))));        \
    vst1q_f32(c + 4 + 3 * ldc, vaddq_f32(vld1q_f32(c + 4 + 3 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v27))));   \
                                                                               \
    vst1q_f32(c + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v28))));        \
    vst1q_f32(c + 4 + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 + 4 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v28))));   \
                                                                               \
    vst1q_f32(c + 5 * ldc, vaddq_f32(vld1q_f32(c + 5 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v29))));        \
    vst1q_f32(c + 4 + 5 * ldc, vaddq_f32(vld1q_f32(c + 4 + 5 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v29))));   \
                                                                               \
    vst1q_f32(c + 6 * ldc, vaddq_f32(vld1q_f32(c + 6 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v30))));        \
    vst1q_f32(c + 4 + 6 * ldc, vaddq_f32(vld1q_f32(c + 4 + 6 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v30))));   \
                                                                               \
    vst1q_f32(c + 7 * ldc, vaddq_f32(vld1q_f32(c + 7 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v31))));        \
    vst1q_f32(c + 4 + 7 * ldc, vaddq_f32(vld1q_f32(c + 4 + 7 * ldc),           \
                                         vcvt_f32_f16(vget_high_f16(v31))));   \
  } while (0)

template <>
void hgemm_kernel_8x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 8 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x8_t v16, v17, v18, v19, v20, v21, v22, v23;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x8_t va0, va1, va2, va3, va4, va5, va6, va7;
      INIT_KERNEL_8x8();
      l = 0;
      for (; l < K;) {
        KERNEL_8x8_ACC1();
      }
      vst1q_f16(c, vaddq_f16(vld1q_f16(c), v24));
      vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v25));
      vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v26));
      vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v27));
      vst1q_f16(c + 4 * ldc, vaddq_f16(vld1q_f16(c + 4 * ldc), v28));
      vst1q_f16(c + 5 * ldc, vaddq_f16(vld1q_f16(c + 5 * ldc), v29));
      vst1q_f16(c + 6 * ldc, vaddq_f16(vld1q_f16(c + 6 * ldc), v30));
      vst1q_f16(c + 7 * ldc, vaddq_f16(vld1q_f16(c + 7 * ldc), v31));
      c += 8;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}

template <>
void hgemm_kernel_8x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  unsigned int K4 = get_prev_mltpl_of_2p_n(K, 2);
  unsigned int K8 = get_prev_mltpl_of_2p_n(K, 3);
  unsigned int K16 = get_prev_mltpl_of_2p_n(K, 4);
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x8_t v16, v17, v18, v19, v20, v21, v22, v23;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x8_t va0, va1, va2, va3, va4, va5, va6, va7;
      l = 0;
      for (; l < K16;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC_N4(16);
        SAVE_KERNEL_8X8_F16_f32();
      }
      for (; l < K8;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC_N4(8);
        SAVE_KERNEL_8X8_F16_f32();
      }
      for (; l < K4;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC_N4(4);
        SAVE_KERNEL_8X8_F16_f32();
      }
      for (; l < K;) {
        INIT_KERNEL_8x8();
        KERNEL_8x8_ACC1();
        SAVE_KERNEL_8X8_F16_f32();
      }
      c += 8;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}
