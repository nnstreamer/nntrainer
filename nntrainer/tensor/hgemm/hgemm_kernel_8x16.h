// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x16.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x16 kernel
 *
 */

#include <hgemm_common.h>
#include <stdlib.h>

// NOPE! THIS is not the right way to implement 8x16 kernel
// note that A is right, but B would be a little bit different
// 1. Partial sum 256 digits : worst accuracy, best latency
#define KERNEL_8x16_ACC8()                 \
  v0 = vdupq_n_f16(0.F);                   \
  v3 = vdupq_n_f16(0.F);                   \
  v6 = vdupq_n_f16(0.F);                   \
  v9 = vdupq_n_f16(0.F);                   \
  v12 = vdupq_n_f16(0.F);                  \
  v15 = vdupq_n_f16(0.F);                  \
  v18 = vdupq_n_f16(0.F);                  \
  v21 = vdupq_n_f16(0.F);                  \
  va0 = vld1q_f16(a);                      \
  v24 = vld1q_f16(b);                      \
  v0 = vfmaq_laneq_f16(v0, v24, va0, 0);   \
  v3 = vfmaq_laneq_f16(v3, v24, va0, 1);   \
  v6 = vfmaq_laneq_f16(v6, v24, va0, 2);   \
  v9 = vfmaq_laneq_f16(v9, v24, va0, 3);   \
  v12 = vfmaq_laneq_f16(v12, v24, va0, 4); \
  v15 = vfmaq_laneq_f16(v15, v24, va0, 5); \
  v18 = vfmaq_laneq_f16(v18, v24, va0, 6); \
  v21 = vfmaq_laneq_f16(v21, v24, va0, 7); \
  va1 = vld1q_f16(a + 4);                  \
  v25 = vld1q_f16(b + 8);                  \
  v0 = vfmaq_laneq_f16(v0, v25, va1, 0);   \
  v3 = vfmaq_laneq_f16(v3, v25, va1, 1);   \
  v6 = vfmaq_laneq_f16(v6, v25, va1, 2);   \
  v9 = vfmaq_laneq_f16(v9, v25, va1, 3);   \
  v12 = vfmaq_laneq_f16(v12, v25, va1, 4); \
  v15 = vfmaq_laneq_f16(v15, v25, va1, 5); \
  v18 = vfmaq_laneq_f16(v18, v25, va1, 6); \
  v21 = vfmaq_laneq_f16(v21, v25, va1, 7); \
  va2 = vld1q_f16(a + 8);                  \
  v26 = vld1q_f16(b + 16);                 \
  v0 = vfmaq_laneq_f16(v0, v26, va2, 0);   \
  v3 = vfmaq_laneq_f16(v3, v26, va2, 1);   \
  v6 = vfmaq_laneq_f16(v6, v26, va2, 2);   \
  v9 = vfmaq_laneq_f16(v9, v26, va2, 3);   \
  v12 = vfmaq_laneq_f16(v12, v26, va2, 4); \
  v15 = vfmaq_laneq_f16(v15, v26, va2, 5); \
  v18 = vfmaq_laneq_f16(v18, v26, va2, 6); \
  v21 = vfmaq_laneq_f16(v21, v26, va2, 7); \
  va3 = vld1q_f16(a + 12);                 \
  v27 = vld1q_f16(b + 24);                 \
  v0 = vfmaq_laneq_f16(v0, v27, va3, 0);   \
  v3 = vfmaq_laneq_f16(v3, v27, va3, 1);   \
  v6 = vfmaq_laneq_f16(v6, v27, va3, 2);   \
  v9 = vfmaq_laneq_f16(v9, v27, va3, 3);   \
  v12 = vfmaq_laneq_f16(v12, v27, va3, 4); \
  v15 = vfmaq_laneq_f16(v15, v27, va3, 5); \
  v18 = vfmaq_laneq_f16(v18, v27, va3, 6); \
  v21 = vfmaq_laneq_f16(v21, v27, va3, 7); \
  va4 = vld1q_f16(a + 16);                 \
  v28 = vld1q_f16(b + 32);                 \
  v0 = vfmaq_laneq_f16(v0, v28, va4, 0);   \
  v3 = vfmaq_laneq_f16(v3, v28, va4, 1);   \
  v6 = vfmaq_laneq_f16(v6, v28, va4, 2);   \
  v9 = vfmaq_laneq_f16(v9, v28, va4, 3);   \
  v12 = vfmaq_laneq_f16(v12, v28, va4, 4); \
  v15 = vfmaq_laneq_f16(v15, v28, va4, 5); \
  v18 = vfmaq_laneq_f16(v18, v28, va4, 6); \
  v21 = vfmaq_laneq_f16(v21, v28, va4, 7); \
  va5 = vld1q_f16(a + 20);                 \
  v29 = vld1q_f16(b + 40);                 \
  v0 = vfmaq_laneq_f16(v0, v29, va5, 0);   \
  v3 = vfmaq_laneq_f16(v3, v29, va5, 1);   \
  v6 = vfmaq_laneq_f16(v6, v29, va5, 2);   \
  v9 = vfmaq_laneq_f16(v9, v29, va5, 3);   \
  v12 = vfmaq_laneq_f16(v12, v29, va5, 4); \
  v15 = vfmaq_laneq_f16(v15, v29, va5, 5); \
  v18 = vfmaq_laneq_f16(v18, v29, va5, 6); \
  v21 = vfmaq_laneq_f16(v21, v29, va5, 7); \
  va6 = vld1q_f16(a + 24);                 \
  v30 = vld1q_f16(b + 48);                 \
  v0 = vfmaq_laneq_f16(v0, v30, va6, 0);   \
  v3 = vfmaq_laneq_f16(v3, v30, va6, 1);   \
  v6 = vfmaq_laneq_f16(v6, v30, va6, 2);   \
  v9 = vfmaq_laneq_f16(v9, v30, va6, 3);   \
  v12 = vfmaq_laneq_f16(v12, v30, va6, 4); \
  v15 = vfmaq_laneq_f16(v15, v30, va6, 5); \
  v18 = vfmaq_laneq_f16(v18, v30, va6, 6); \
  v21 = vfmaq_laneq_f16(v21, v30, va6, 7); \
  va7 = vld1q_f16(a + 28);                 \
  v31 = vld1q_f16(b + 56);                 \
  v0 = vfmaq_laneq_f16(v0, v31, va7, 0);   \
  v3 = vfmaq_laneq_f16(v3, v31, va7, 1);   \
  v6 = vfmaq_laneq_f16(v6, v31, va7, 2);   \
  v9 = vfmaq_laneq_f16(v9, v31, va7, 3);   \
  v12 = vfmaq_laneq_f16(v12, v31, va7, 4); \
  v15 = vfmaq_laneq_f16(v15, v31, va7, 5); \
  v18 = vfmaq_laneq_f16(v18, v31, va7, 6); \
  v21 = vfmaq_laneq_f16(v21, v31, va7, 7); \
  l += 8;                                  \
  b += 16 * 8;                             \
  a += 8 * 8;

/**
 * @brief hgemm 8x16 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_8x16(unsigned int m, unsigned int n, unsigned int k,
                       __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(m > 0 && n > 0 && k > 0);
  assert(m % 4 == 0 && n % 8 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int k8 = (k >> 3) << 3;
  unsigned int i, j, l;
  for (i = 0; i < m; i += 8) {
    for (j = 0; j < n; j += 16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      float16x8_t v0, v3, v6, v9;
      float16x8_t v12, v15, v18, v21;
      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x8_t va0, va1, va2, va3, va4, va5, va6, va7;
      l = 0;
      for (; l < k8;) {
        KERNEL_8x16_ACC8();

        vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0));
        vst1q_f16(c + 8, vaddq_f16(vld1q_f16(c + 8), v3));

        vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v6));
        vst1q_f16(c + 8 + ldc, vaddq_f16(vld1q_f16(c + 8 + ldc), v9));

        vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v12));
        vst1q_f16(c + 8 + 2 * ldc, vaddq_f16(vld1q_f16(c + 8 + 2 * ldc), v15));

        vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v18));
        vst1q_f16(c + 8 + 3 * ldc, vaddq_f16(vld1q_f16(c + 8 + 3 * ldc), v21));
      }
      //   for (; l < k;) {
      //     KERNEL_8x16_ACC1();

      //     vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0));
      //     vst1q_f16(c + 8, vaddq_f16(vld1q_f16(c + 8), v3));

      //     vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v6));
      //     vst1q_f16(c + 8 + ldc, vaddq_f16(vld1q_f16(c + 8 + ldc), v9));

      //     vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v12));
      //     vst1q_f16(c + 8 + 2 * ldc, vaddq_f16(vld1q_f16(c + 8 + 2 * ldc),
      //     v15));

      //     vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v18));
      //     vst1q_f16(c + 8 + 3 * ldc, vaddq_f16(vld1q_f16(c + 8 + 3 * ldc),
      //     v21));
      //   }
      c += 16;
      a -= 8 * k;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * k;
    b = sb;
  }
}
