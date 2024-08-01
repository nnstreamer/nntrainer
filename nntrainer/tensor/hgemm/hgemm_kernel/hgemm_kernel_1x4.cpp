// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   hgemm_kernel_1x4.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 1x4 kernel
 *
 */

#include <arm_neon.h>
#include <assert.h>
#include <hgemm_kernel.h>
#include <stdlib.h>

template <>
void hgemm_kernel_1x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(N % 4 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j += 4) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      for (l = 0; l < K; l += 4) {
        float16x4_t v24 = {0.F};
        float16x4_t v0 = vld1_f16(b);
        float16_t v16 = *a;

        v24 = vfma_n_f16(v24, v0, v16);

        float16x4_t v1 = vld1_f16(b + 4);
        float16_t v17 = *(a + 1);

        v24 = vfma_n_f16(v24, v1, v17);

        float16x4_t v2 = vld1_f16(b + 8);
        float16_t v18 = *(a + 2);

        v24 = vfma_n_f16(v24, v2, v18);

        float16x4_t v3 = vld1_f16(b + 12);
        float16_t v19 = *(a + 3);

        v24 = vfma_n_f16(v24, v3, v19);

        __builtin_prefetch(b + 16, 0, 3);
        __builtin_prefetch(a + 4, 0, 3);

        b += 16;
        a += 4;

        v24 = vadd_f16(vld1_f16(c), v24);

        vst1_f16(c, v24);
      }
      c += 4;
      a -= K;
    }
    sc += ldc;
    c = sc;
    a += K;
    b = sb;
  }
}

template <>
void hgemm_kernel_1x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(N % 4 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j += 4) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      for (l = 0; l < K; l += 4) {
        float16x4_t v24 = {0.F};
        float16x4_t v0 = vld1_f16(b);
        float16_t v16 = *a;

        v24 = vfma_n_f16(v24, v0, v16);

        float16x4_t v1 = vld1_f16(b + 4);
        float16_t v17 = *(a + 1);

        v24 = vfma_n_f16(v24, v1, v17);

        float16x4_t v2 = vld1_f16(b + 8);
        float16_t v18 = *(a + 2);

        v24 = vfma_n_f16(v24, v2, v18);

        float16x4_t v3 = vld1_f16(b + 12);
        float16_t v19 = *(a + 3);

        v24 = vfma_n_f16(v24, v3, v19);

        __builtin_prefetch(b + 16, 0, 3);
        __builtin_prefetch(a + 4, 0, 3);

        b += 16;
        a += 4;

        vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(v24)));
      }
      c += 4;
      a -= K;
    }
    sc += ldc;
    c = sc;
    a += K;
    b = sb;
  }
}
