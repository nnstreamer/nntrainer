// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_4x8.h
 * @date   03 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x8 kernel
 *
 */

#include <hgemm_common.h>
#include <stdlib.h>

// 1. Partial sum 256 digits : worst accuracy, best latency
#define KERNEL_4x8_ACC8()               \
  v0 = vdupq_n_f16(0.F);                \
  v3 = vdupq_n_f16(0.F);                \
  v6 = vdupq_n_f16(0.F);                \
  v9 = vdupq_n_f16(0.F);                \
  dv0 = vld1_f16(a);                    \
  v24 = vld1q_f16(b);                   \
  v0 = vfmaq_lane_f16(v0, v24, dv0, 0); \
  v3 = vfmaq_lane_f16(v3, v24, dv0, 1); \
  v6 = vfmaq_lane_f16(v6, v24, dv0, 2); \
  v9 = vfmaq_lane_f16(v9, v24, dv0, 3); \
  dv1 = vld1_f16(a + 4);                \
  v25 = vld1q_f16(b + 8);               \
  v0 = vfmaq_lane_f16(v0, v25, dv1, 0); \
  v3 = vfmaq_lane_f16(v3, v25, dv1, 1); \
  v6 = vfmaq_lane_f16(v6, v25, dv1, 2); \
  v9 = vfmaq_lane_f16(v9, v25, dv1, 3); \
  dv2 = vld1_f16(a + 8);                \
  v26 = vld1q_f16(b + 16);              \
  v0 = vfmaq_lane_f16(v0, v26, dv2, 0); \
  v3 = vfmaq_lane_f16(v3, v26, dv2, 1); \
  v6 = vfmaq_lane_f16(v6, v26, dv2, 2); \
  v9 = vfmaq_lane_f16(v9, v26, dv2, 3); \
  dv3 = vld1_f16(a + 12);               \
  v27 = vld1q_f16(b + 24);              \
  v0 = vfmaq_lane_f16(v0, v27, dv3, 0); \
  v3 = vfmaq_lane_f16(v3, v27, dv3, 1); \
  v6 = vfmaq_lane_f16(v6, v27, dv3, 2); \
  v9 = vfmaq_lane_f16(v9, v27, dv3, 3); \
  dv4 = vld1_f16(a + 16);               \
  v28 = vld1q_f16(b + 32);              \
  v0 = vfmaq_lane_f16(v0, v28, dv4, 0); \
  v3 = vfmaq_lane_f16(v3, v28, dv4, 1); \
  v6 = vfmaq_lane_f16(v6, v28, dv4, 2); \
  v9 = vfmaq_lane_f16(v9, v28, dv4, 3); \
  dv5 = vld1_f16(a + 20);               \
  v29 = vld1q_f16(b + 40);              \
  v0 = vfmaq_lane_f16(v0, v29, dv5, 0); \
  v3 = vfmaq_lane_f16(v3, v29, dv5, 1); \
  v6 = vfmaq_lane_f16(v6, v29, dv5, 2); \
  v9 = vfmaq_lane_f16(v9, v29, dv5, 3); \
  dv6 = vld1_f16(a + 24);               \
  v30 = vld1q_f16(b + 48);              \
  v0 = vfmaq_lane_f16(v0, v30, dv6, 0); \
  v3 = vfmaq_lane_f16(v3, v30, dv6, 1); \
  v6 = vfmaq_lane_f16(v6, v30, dv6, 2); \
  v9 = vfmaq_lane_f16(v9, v30, dv6, 3); \
  dv7 = vld1_f16(a + 28);               \
  v31 = vld1q_f16(b + 56);              \
  v0 = vfmaq_lane_f16(v0, v31, dv7, 0); \
  v3 = vfmaq_lane_f16(v3, v31, dv7, 1); \
  v6 = vfmaq_lane_f16(v6, v31, dv7, 2); \
  v9 = vfmaq_lane_f16(v9, v31, dv7, 3); \
  l += 8;                               \
  b += 8 * 8;                           \
  a += 4 * 8;

// 2. Partial sum 128 digits : medium accuracy, medium latency
#define KERNEL_4x8_ACC4()               \
  v0 = vdupq_n_f16(0.F);                \
  v3 = vdupq_n_f16(0.F);                \
  v6 = vdupq_n_f16(0.F);                \
  v9 = vdupq_n_f16(0.F);                \
  dv0 = vld1_f16(a);                    \
  v24 = vld1q_f16(b);                   \
  v0 = vfmaq_lane_f16(v0, v24, dv0, 0); \
  v3 = vfmaq_lane_f16(v3, v24, dv0, 1); \
  v6 = vfmaq_lane_f16(v6, v24, dv0, 2); \
  v9 = vfmaq_lane_f16(v9, v24, dv0, 3); \
  dv1 = vld1_f16(a + 4);                \
  v25 = vld1q_f16(b + 8);               \
  v0 = vfmaq_lane_f16(v0, v25, dv1, 0); \
  v3 = vfmaq_lane_f16(v3, v25, dv1, 1); \
  v6 = vfmaq_lane_f16(v6, v25, dv1, 2); \
  v9 = vfmaq_lane_f16(v9, v25, dv1, 3); \
  dv2 = vld1_f16(a + 8);                \
  v26 = vld1q_f16(b + 16);              \
  v0 = vfmaq_lane_f16(v0, v26, dv2, 0); \
  v3 = vfmaq_lane_f16(v3, v26, dv2, 1); \
  v6 = vfmaq_lane_f16(v6, v26, dv2, 2); \
  v9 = vfmaq_lane_f16(v9, v26, dv2, 3); \
  dv3 = vld1_f16(a + 12);               \
  v27 = vld1q_f16(b + 24);              \
  v0 = vfmaq_lane_f16(v0, v27, dv3, 0); \
  v3 = vfmaq_lane_f16(v3, v27, dv3, 1); \
  v6 = vfmaq_lane_f16(v6, v27, dv3, 2); \
  v9 = vfmaq_lane_f16(v9, v27, dv3, 3); \
  l += 4;                               \
  b += 8 * 4;                           \
  a += 4 * 4;

// 3. Partial sum 32 digits : Best accuracy, worst latency
#define KERNEL_4x8_ACC1()               \
  v0 = vdupq_n_f16(0.F);                \
  v3 = vdupq_n_f16(0.F);                \
  v6 = vdupq_n_f16(0.F);                \
  v9 = vdupq_n_f16(0.F);                \
  dv0 = vld1_f16(a);                    \
  v24 = vld1q_f16(b);                   \
  v0 = vfmaq_lane_f16(v0, v24, dv0, 0); \
  v3 = vfmaq_lane_f16(v3, v24, dv0, 1); \
  v6 = vfmaq_lane_f16(v6, v24, dv0, 2); \
  v9 = vfmaq_lane_f16(v9, v24, dv0, 3); \
  l += 1;                               \
  b += 8 * 1;                           \
  a += 4 * 1;

/**
 * @brief hgemm 4x8 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_4x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 8 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int k8 = (K >> 3) << 3;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 4) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0, v3, v6, v9;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x4_t dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7;
      l = 0;
      for (; l < k8;) {
        KERNEL_4x8_ACC8();

        vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0));
        vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v3));
        vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v6));
        vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v9));
      }
      for (; l < K;) {
        KERNEL_4x8_ACC1();

        vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0));
        vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v3));
        vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v6));
        vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v9));
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

/**
 * @brief hgemm 4x8 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_4x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 8 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int k8 = (K >> 3) << 3;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 4) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0, v3, v6, v9;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x4_t dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7;
      l = 0;
      for (; l < k8;) {
        KERNEL_4x8_ACC8();

        vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0))));
        vst1q_f32(c + ldc, vaddq_f32(vld1q_f32(c + ldc),
                                     vcvt_f32_f16(vget_low_f16(v3))));
        vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v6))));
        vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v9))));

        vst1q_f32(c + 4,
                  vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v0))));
        vst1q_f32(c + 4 + ldc, vaddq_f32(vld1q_f32(c + 4 + ldc),
                                         vcvt_f32_f16(vget_high_f16(v3))));
        vst1q_f32(c + 4 + 2 * ldc, vaddq_f32(vld1q_f32(c + 4 + 2 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v6))));
        vst1q_f32(c + 4 + 3 * ldc, vaddq_f32(vld1q_f32(c + 4 + 3 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v9))));
      }
      for (; l < K;) {
        KERNEL_4x8_ACC1();

        vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0))));
        vst1q_f32(c + ldc, vaddq_f32(vld1q_f32(c + ldc),
                                     vcvt_f32_f16(vget_low_f16(v3))));
        vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v6))));
        vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v9))));

        vst1q_f32(c + 4,
                  vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v0))));
        vst1q_f32(c + 4 + ldc, vaddq_f32(vld1q_f32(c + 4 + ldc),
                                         vcvt_f32_f16(vget_high_f16(v3))));
        vst1q_f32(c + 4 + 2 * ldc, vaddq_f32(vld1q_f32(c + 4 + 2 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v6))));
        vst1q_f32(c + 4 + 3 * ldc, vaddq_f32(vld1q_f32(c + 4 + 3 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v9))));
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
