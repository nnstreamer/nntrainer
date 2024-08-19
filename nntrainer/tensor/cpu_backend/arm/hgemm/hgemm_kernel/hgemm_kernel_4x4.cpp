// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_4x4.cpp
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 4x4 kernel
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

#define INIT_KERNEL_4x4()  \
  do {                     \
    v24 = vdup_n_f16(0.F); \
    v25 = vdup_n_f16(0.F); \
    v26 = vdup_n_f16(0.F); \
    v27 = vdup_n_f16(0.F); \
  } while (0)

// 1. Partial sum 256 digits
#define KERNEL_4x4_ACC16()                 \
  do {                                     \
    dv0 = vld1_f16(a);                     \
    vb0 = vld1_f16(b);                     \
    v24 = vfma_lane_f16(v24, vb0, dv0, 0); \
    v25 = vfma_lane_f16(v25, vb0, dv0, 1); \
    v26 = vfma_lane_f16(v26, vb0, dv0, 2); \
    v27 = vfma_lane_f16(v27, vb0, dv0, 3); \
    dv1 = vld1_f16(a + 4);                 \
    vb1 = vld1_f16(b + 4);                 \
    v24 = vfma_lane_f16(v24, vb1, dv1, 0); \
    v25 = vfma_lane_f16(v25, vb1, dv1, 1); \
    v26 = vfma_lane_f16(v26, vb1, dv1, 2); \
    v27 = vfma_lane_f16(v27, vb1, dv1, 3); \
    dv2 = vld1_f16(a + 4 * 2);             \
    vb2 = vld1_f16(b + 4 * 2);             \
    v24 = vfma_lane_f16(v24, vb2, dv2, 0); \
    v25 = vfma_lane_f16(v25, vb2, dv2, 1); \
    v26 = vfma_lane_f16(v26, vb2, dv2, 2); \
    v27 = vfma_lane_f16(v27, vb2, dv2, 3); \
    dv3 = vld1_f16(a + 4 * 3);             \
    vb3 = vld1_f16(b + 4 * 3);             \
    v24 = vfma_lane_f16(v24, vb3, dv3, 0); \
    v25 = vfma_lane_f16(v25, vb3, dv3, 1); \
    v26 = vfma_lane_f16(v26, vb3, dv3, 2); \
    v27 = vfma_lane_f16(v27, vb3, dv3, 3); \
    dv4 = vld1_f16(a + 4 * 4);             \
    vb4 = vld1_f16(b + 4 * 4);             \
    v24 = vfma_lane_f16(v24, vb4, dv4, 0); \
    v25 = vfma_lane_f16(v25, vb4, dv4, 1); \
    v26 = vfma_lane_f16(v26, vb4, dv4, 2); \
    v27 = vfma_lane_f16(v27, vb4, dv4, 3); \
    dv5 = vld1_f16(a + 4 * 5);             \
    vb5 = vld1_f16(b + 4 * 5);             \
    v24 = vfma_lane_f16(v24, vb5, dv5, 0); \
    v25 = vfma_lane_f16(v25, vb5, dv5, 1); \
    v26 = vfma_lane_f16(v26, vb5, dv5, 2); \
    v27 = vfma_lane_f16(v27, vb5, dv5, 3); \
    dv6 = vld1_f16(a + 4 * 6);             \
    vb6 = vld1_f16(b + 4 * 6);             \
    v24 = vfma_lane_f16(v24, vb6, dv6, 0); \
    v25 = vfma_lane_f16(v25, vb6, dv6, 1); \
    v26 = vfma_lane_f16(v26, vb6, dv6, 2); \
    v27 = vfma_lane_f16(v27, vb6, dv6, 3); \
    dv7 = vld1_f16(a + 4 * 7);             \
    vb7 = vld1_f16(b + 4 * 7);             \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 8);             \
    vb7 = vld1_f16(b + 4 * 8);             \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 9);             \
    vb7 = vld1_f16(b + 4 * 9);             \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 10);            \
    vb7 = vld1_f16(b + 4 * 10);            \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 11);            \
    vb7 = vld1_f16(b + 4 * 11);            \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 12);            \
    vb7 = vld1_f16(b + 4 * 12);            \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 13);            \
    vb7 = vld1_f16(b + 4 * 13);            \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 14);            \
    vb7 = vld1_f16(b + 4 * 14);            \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    dv7 = vld1_f16(a + 4 * 15);            \
    vb7 = vld1_f16(b + 4 * 15);            \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    l += 16;                               \
    __builtin_prefetch(b + 64, 0, 3);      \
    __builtin_prefetch(a + 64, 0, 3);      \
    b += 4 * 16;                           \
    a += 4 * 16;                           \
  } while (0)

// 2. Partial sum 128 digits
#define KERNEL_4x4_ACC8()                  \
  do {                                     \
    dv0 = vld1_f16(a);                     \
    vb0 = vld1_f16(b);                     \
    v24 = vfma_lane_f16(v24, vb0, dv0, 0); \
    v25 = vfma_lane_f16(v25, vb0, dv0, 1); \
    v26 = vfma_lane_f16(v26, vb0, dv0, 2); \
    v27 = vfma_lane_f16(v27, vb0, dv0, 3); \
    dv1 = vld1_f16(a + 4);                 \
    vb1 = vld1_f16(b + 4);                 \
    v24 = vfma_lane_f16(v24, vb1, dv1, 0); \
    v25 = vfma_lane_f16(v25, vb1, dv1, 1); \
    v26 = vfma_lane_f16(v26, vb1, dv1, 2); \
    v27 = vfma_lane_f16(v27, vb1, dv1, 3); \
    dv2 = vld1_f16(a + 8);                 \
    vb2 = vld1_f16(b + 8);                 \
    v24 = vfma_lane_f16(v24, vb2, dv2, 0); \
    v25 = vfma_lane_f16(v25, vb2, dv2, 1); \
    v26 = vfma_lane_f16(v26, vb2, dv2, 2); \
    v27 = vfma_lane_f16(v27, vb2, dv2, 3); \
    dv3 = vld1_f16(a + 12);                \
    vb3 = vld1_f16(b + 12);                \
    v24 = vfma_lane_f16(v24, vb3, dv3, 0); \
    v25 = vfma_lane_f16(v25, vb3, dv3, 1); \
    v26 = vfma_lane_f16(v26, vb3, dv3, 2); \
    v27 = vfma_lane_f16(v27, vb3, dv3, 3); \
    dv4 = vld1_f16(a + 16);                \
    vb4 = vld1_f16(b + 16);                \
    v24 = vfma_lane_f16(v24, vb4, dv4, 0); \
    v25 = vfma_lane_f16(v25, vb4, dv4, 1); \
    v26 = vfma_lane_f16(v26, vb4, dv4, 2); \
    v27 = vfma_lane_f16(v27, vb4, dv4, 3); \
    dv5 = vld1_f16(a + 20);                \
    vb5 = vld1_f16(b + 20);                \
    v24 = vfma_lane_f16(v24, vb5, dv5, 0); \
    v25 = vfma_lane_f16(v25, vb5, dv5, 1); \
    v26 = vfma_lane_f16(v26, vb5, dv5, 2); \
    v27 = vfma_lane_f16(v27, vb5, dv5, 3); \
    dv6 = vld1_f16(a + 24);                \
    vb6 = vld1_f16(b + 24);                \
    v24 = vfma_lane_f16(v24, vb6, dv6, 0); \
    v25 = vfma_lane_f16(v25, vb6, dv6, 1); \
    v26 = vfma_lane_f16(v26, vb6, dv6, 2); \
    v27 = vfma_lane_f16(v27, vb6, dv6, 3); \
    dv7 = vld1_f16(a + 28);                \
    vb7 = vld1_f16(b + 28);                \
    v24 = vfma_lane_f16(v24, vb7, dv7, 0); \
    v25 = vfma_lane_f16(v25, vb7, dv7, 1); \
    v26 = vfma_lane_f16(v26, vb7, dv7, 2); \
    v27 = vfma_lane_f16(v27, vb7, dv7, 3); \
    l += 8;                                \
    __builtin_prefetch(b + 32, 0, 3);      \
    __builtin_prefetch(a + 32, 0, 3);      \
    b += 4 * 8;                            \
    a += 4 * 8;                            \
  } while (0)

// 3. Partial sum 16 digits
#define KERNEL_4x4_ACC1()                  \
  do {                                     \
    dv0 = vld1_f16(a);                     \
    vb0 = vld1_f16(b);                     \
    v24 = vfma_lane_f16(v24, vb0, dv0, 0); \
    v25 = vfma_lane_f16(v25, vb0, dv0, 1); \
    v26 = vfma_lane_f16(v26, vb0, dv0, 2); \
    v27 = vfma_lane_f16(v27, vb0, dv0, 3); \
    l += 1;                                \
    __builtin_prefetch(b + 4, 0, 3);       \
    __builtin_prefetch(a + 4, 0, 3);       \
    b += 4 * 1;                            \
    a += 4 * 1;                            \
  } while (0)

#define SAVE_KERNEL_4X4_F16_F32()                                         \
  do {                                                                    \
    vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(v24)));             \
    vst1q_f32(c + ldc, vaddq_f32(vld1q_f32(c + ldc), vcvt_f32_f16(v25))); \
    vst1q_f32(c + 2 * ldc,                                                \
              vaddq_f32(vld1q_f32(c + 2 * ldc), vcvt_f32_f16(v26)));      \
    vst1q_f32(c + 3 * ldc,                                                \
              vaddq_f32(vld1q_f32(c + 3 * ldc), vcvt_f32_f16(v27)));      \
  } while (0)

template <>
void hgemm_kernel_4x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 4 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 4) {
    for (j = 0; j < N; j += 4) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x4_t v24;
      float16x4_t v25;
      float16x4_t v26;
      float16x4_t v27;
      INIT_KERNEL_4x4();

      for (l = 0; l < K; l += 4) {
        float16x4_t v0 = vld1_f16(b);
        float16x4_t v16 = vld1_f16(a);

        v24 = vfma_lane_f16(v24, v0, v16, 0);
        v25 = vfma_lane_f16(v25, v0, v16, 1);
        v26 = vfma_lane_f16(v26, v0, v16, 2);
        v27 = vfma_lane_f16(v27, v0, v16, 3);

        float16x4_t v1 = vld1_f16(b + 4);
        float16x4_t v17 = vld1_f16(a + 4);

        v24 = vfma_lane_f16(v24, v1, v17, 0);
        v25 = vfma_lane_f16(v25, v1, v17, 1);
        v26 = vfma_lane_f16(v26, v1, v17, 2);
        v27 = vfma_lane_f16(v27, v1, v17, 3);

        float16x4_t v2 = vld1_f16(b + 8);
        float16x4_t v18 = vld1_f16(a + 8);

        v24 = vfma_lane_f16(v24, v2, v18, 0);
        v25 = vfma_lane_f16(v25, v2, v18, 1);
        v26 = vfma_lane_f16(v26, v2, v18, 2);
        v27 = vfma_lane_f16(v27, v2, v18, 3);

        float16x4_t v3 = vld1_f16(b + 12);
        float16x4_t v19 = vld1_f16(a + 12);

        v24 = vfma_lane_f16(v24, v3, v19, 0);
        v25 = vfma_lane_f16(v25, v3, v19, 1);
        v26 = vfma_lane_f16(v26, v3, v19, 2);
        v27 = vfma_lane_f16(v27, v3, v19, 3);

        __builtin_prefetch(b + 16, 0, 3);
        __builtin_prefetch(a + 16, 0, 3);

        b += 16;
        a += 16;
      }

      v24 = vadd_f16(vld1_f16(c), v24);
      v25 = vadd_f16(vld1_f16(c + ldc), v25);
      v26 = vadd_f16(vld1_f16(c + 2 * ldc), v26);
      v27 = vadd_f16(vld1_f16(c + 3 * ldc), v27);

      vst1_f16(c, v24);
      vst1_f16(c + ldc, v25);
      vst1_f16(c + 2 * ldc, v26);
      vst1_f16(c + 3 * ldc, v27);

      c += 4;
      a -= 4 * K;
    }
    sc += ldc * 4;
    c = sc;
    a += 4 * K;
    b = sb;
  }
}

template <>
void hgemm_kernel_4x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 4 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  unsigned int K8 = get_prev_mltpl_of_2p_n(K, 3);
  unsigned int K16 = get_prev_mltpl_of_2p_n(K, 4);
  for (i = 0; i < M; i += 4) {
    for (j = 0; j < N; j += 4) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x4_t v24, v25, v26, v27;
      float16x4_t dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7;
      float16x4_t vb0, vb1, vb2, vb3, vb4, vb5, vb6, vb7;
      l = 0;
      for (; l < K16;) {
        INIT_KERNEL_4x4();
        KERNEL_4x4_ACC16();
        SAVE_KERNEL_4X4_F16_F32();
      }
      for (; l < K8;) {
        INIT_KERNEL_4x4();
        KERNEL_4x4_ACC8();
        SAVE_KERNEL_4X4_F16_F32();
      }
      for (; l < K;) {
        INIT_KERNEL_4x4();
        KERNEL_4x4_ACC1();
        SAVE_KERNEL_4X4_F16_F32();
      }

      c += 4;
      a -= 4 * K;
    }
    sc += ldc * 4;
    c = sc;
    a += 4 * K;
    b = sb;
  }
}
