// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   hgemm_kernel_1x8.h
 * @date   05 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 1x8 kernel
 *
 */

#include <hgemm_common.h>
#include <stdlib.h>

// 1. Partial sum 64 digits : worst accuracy, best latency
#define KERNEL_1x8_ACC8()           \
  do {                              \
    v0 = vdupq_n_f16(0.F);          \
    dv0 = *a;                       \
    v24 = vld1q_f16(b);             \
    v0 = vfmaq_n_f16(v0, v24, dv0); \
    dv1 = *(a + 1);                 \
    v25 = vld1q_f16(b + 8);         \
    v0 = vfmaq_n_f16(v0, v25, dv1); \
    dv2 = *(a + 2);                 \
    v26 = vld1q_f16(b + 16);        \
    v0 = vfmaq_n_f16(v0, v26, dv2); \
    dv3 = *(a + 3);                 \
    v27 = vld1q_f16(b + 24);        \
    v0 = vfmaq_n_f16(v0, v27, dv3); \
    dv4 = *(a + 4);                 \
    v28 = vld1q_f16(b + 32);        \
    v0 = vfmaq_n_f16(v0, v28, dv4); \
    dv5 = *(a + 5);                 \
    v29 = vld1q_f16(b + 40);        \
    v0 = vfmaq_n_f16(v0, v29, dv5); \
    dv6 = *(a + 6);                 \
    v30 = vld1q_f16(b + 48);        \
    v0 = vfmaq_n_f16(v0, v30, dv6); \
    dv7 = *(a + 7);                 \
    v31 = vld1q_f16(b + 56);        \
    v0 = vfmaq_n_f16(v0, v31, dv7); \
    l += 8;                         \
    b += 8 * 8;                     \
    a += 8;                         \
  } while (0)

// 2. Partial sum 32 digits : medium accuracy, medium latency
#define KERNEL_1x8_ACC4()           \
  do {                              \
    v0 = vdupq_n_f16(0.F);          \
    dv0 = *a;                       \
    v24 = vld1q_f16(b);             \
    v0 = vfmaq_n_f16(v0, v24, dv0); \
    dv1 = *(a + 1);                 \
    v25 = vld1q_f16(b + 8);         \
    v0 = vfmaq_n_f16(v0, v25, dv1); \
    dv2 = *(a + 2);                 \
    v26 = vld1q_f16(b + 16);        \
    v0 = vfmaq_n_f16(v0, v26, dv2); \
    dv3 = *(a + 3);                 \
    v27 = vld1q_f16(b + 24);        \
    v0 = vfmaq_n_f16(v0, v27, dv3); \
    l += 4;                         \
    b += 8 * 4;                     \
    a += 4;                         \
  } while (0)

// 3. Partial sum 8 digits : Best accuracy, worst latency
#define KERNEL_1x8_ACC1()           \
  do {                              \
    v0 = vdupq_n_f16(0.F);          \
    dv0 = *(a);                     \
    v24 = vld1q_f16(b);             \
    v0 = vfmaq_n_f16(v0, v24, dv0); \
    l += 1;                         \
    b += 8 * 1;                     \
    a++;                            \
  } while (0)

/**
 * @brief hgemm 1x8 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_1x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(N % 8 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int k8 = (K >> 3) << 3;
  unsigned int i, j, l;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16_t dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7;
      l = 0;
      for (; l < k8;) {
        KERNEL_1x8_ACC8();

        vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0));
      }
      for (; l < K;) {
        KERNEL_1x8_ACC1();

        vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0));
      }
      c += 8;
      a -= K;
    }
    sc += ldc;
    c = sc;
    a += K;
    b = sb;
  }
}

/**
 * @brief hgemm 1x8 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_1x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(N % 8 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int k8 = (K >> 3) << 3;
  unsigned int i, j, l;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j += 8) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16_t dv0, dv1, dv2, dv3, dv4, dv5, dv6, dv7;
      l = 0;
      for (; l < k8;) {
        KERNEL_1x8_ACC8();

        vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0))));

        vst1q_f32(c + 4,
                  vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v0))));
      }
      for (; l < K;) {
        KERNEL_1x8_ACC1();

        vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0))));

        vst1q_f32(c + 4,
                  vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v0))));
      }
      c += 8;
      a -= K;
    }
    sc += ldc;
    c = sc;
    a += K;
    b = sb;
  }
}
