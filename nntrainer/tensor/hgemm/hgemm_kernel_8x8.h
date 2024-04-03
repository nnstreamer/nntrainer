// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x8.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x8 kernel
 *
 */

#include <hgemm_common.h>
#include <stdlib.h>

/**
 * @brief hgemm 8x8 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_8x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += VL_FP16) {
    for (j = 0; j < N; j += VL_FP16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x8_t v24 = {0};
      float16x8_t v25 = {0};
      float16x8_t v26 = {0};
      float16x8_t v27 = {0};
      float16x8_t v28 = {0};
      float16x8_t v29 = {0};
      float16x8_t v30 = {0};
      float16x8_t v31 = {0};

      for (l = 0; l < K; l += VL_FP16) {
        float16x8_t v0 = vld1q_f16(b);
        float16x8_t v16 = vld1q_f16(a);

        v24 = vfmaq_laneq_f16(v24, v0, v16, 0);
        v25 = vfmaq_laneq_f16(v25, v0, v16, 1);
        v26 = vfmaq_laneq_f16(v26, v0, v16, 2);
        v27 = vfmaq_laneq_f16(v27, v0, v16, 3);
        v28 = vfmaq_laneq_f16(v28, v0, v16, 4);
        v29 = vfmaq_laneq_f16(v29, v0, v16, 5);
        v30 = vfmaq_laneq_f16(v30, v0, v16, 6);
        v31 = vfmaq_laneq_f16(v31, v0, v16, 7);

        float16x8_t v1 = vld1q_f16(b + 8);
        float16x8_t v17 = vld1q_f16(a + 8);

        v24 = vfmaq_laneq_f16(v24, v1, v17, 0);
        v25 = vfmaq_laneq_f16(v25, v1, v17, 1);
        v26 = vfmaq_laneq_f16(v26, v1, v17, 2);
        v27 = vfmaq_laneq_f16(v27, v1, v17, 3);
        v28 = vfmaq_laneq_f16(v28, v1, v17, 4);
        v29 = vfmaq_laneq_f16(v29, v1, v17, 5);
        v30 = vfmaq_laneq_f16(v30, v1, v17, 6);
        v31 = vfmaq_laneq_f16(v31, v1, v17, 7);

        float16x8_t v2 = vld1q_f16(b + 16);
        float16x8_t v18 = vld1q_f16(a + 16);

        v24 = vfmaq_laneq_f16(v24, v2, v18, 0);
        v25 = vfmaq_laneq_f16(v25, v2, v18, 1);
        v26 = vfmaq_laneq_f16(v26, v2, v18, 2);
        v27 = vfmaq_laneq_f16(v27, v2, v18, 3);
        v28 = vfmaq_laneq_f16(v28, v2, v18, 4);
        v29 = vfmaq_laneq_f16(v29, v2, v18, 5);
        v30 = vfmaq_laneq_f16(v30, v2, v18, 6);
        v31 = vfmaq_laneq_f16(v31, v2, v18, 7);

        float16x8_t v3 = vld1q_f16(b + 24);
        float16x8_t v19 = vld1q_f16(a + 24);

        v24 = vfmaq_laneq_f16(v24, v3, v19, 0);
        v25 = vfmaq_laneq_f16(v25, v3, v19, 1);
        v26 = vfmaq_laneq_f16(v26, v3, v19, 2);
        v27 = vfmaq_laneq_f16(v27, v3, v19, 3);
        v28 = vfmaq_laneq_f16(v28, v3, v19, 4);
        v29 = vfmaq_laneq_f16(v29, v3, v19, 5);
        v30 = vfmaq_laneq_f16(v30, v3, v19, 6);
        v31 = vfmaq_laneq_f16(v31, v3, v19, 7);

        float16x8_t v4 = vld1q_f16(b + 32);
        float16x8_t v20 = vld1q_f16(a + 32);

        v24 = vfmaq_laneq_f16(v24, v4, v20, 0);
        v25 = vfmaq_laneq_f16(v25, v4, v20, 1);
        v26 = vfmaq_laneq_f16(v26, v4, v20, 2);
        v27 = vfmaq_laneq_f16(v27, v4, v20, 3);
        v28 = vfmaq_laneq_f16(v28, v4, v20, 4);
        v29 = vfmaq_laneq_f16(v29, v4, v20, 5);
        v30 = vfmaq_laneq_f16(v30, v4, v20, 6);
        v31 = vfmaq_laneq_f16(v31, v4, v20, 7);

        float16x8_t v5 = vld1q_f16(b + 40);
        float16x8_t v21 = vld1q_f16(a + 40);

        v24 = vfmaq_laneq_f16(v24, v5, v21, 0);
        v25 = vfmaq_laneq_f16(v25, v5, v21, 1);
        v26 = vfmaq_laneq_f16(v26, v5, v21, 2);
        v27 = vfmaq_laneq_f16(v27, v5, v21, 3);
        v28 = vfmaq_laneq_f16(v28, v5, v21, 4);
        v29 = vfmaq_laneq_f16(v29, v5, v21, 5);
        v30 = vfmaq_laneq_f16(v30, v5, v21, 6);
        v31 = vfmaq_laneq_f16(v31, v5, v21, 7);

        float16x8_t v6 = vld1q_f16(b + 48);
        float16x8_t v22 = vld1q_f16(a + 48);

        v24 = vfmaq_laneq_f16(v24, v6, v22, 0);
        v25 = vfmaq_laneq_f16(v25, v6, v22, 1);
        v26 = vfmaq_laneq_f16(v26, v6, v22, 2);
        v27 = vfmaq_laneq_f16(v27, v6, v22, 3);
        v28 = vfmaq_laneq_f16(v28, v6, v22, 4);
        v29 = vfmaq_laneq_f16(v29, v6, v22, 5);
        v30 = vfmaq_laneq_f16(v30, v6, v22, 6);
        v31 = vfmaq_laneq_f16(v31, v6, v22, 7);

        float16x8_t v7 = vld1q_f16(b + 56);
        float16x8_t v23 = vld1q_f16(a + 56);

        v24 = vfmaq_laneq_f16(v24, v7, v23, 0);
        v25 = vfmaq_laneq_f16(v25, v7, v23, 1);
        v26 = vfmaq_laneq_f16(v26, v7, v23, 2);
        v27 = vfmaq_laneq_f16(v27, v7, v23, 3);
        v28 = vfmaq_laneq_f16(v28, v7, v23, 4);
        v29 = vfmaq_laneq_f16(v29, v7, v23, 5);
        v30 = vfmaq_laneq_f16(v30, v7, v23, 6);
        v31 = vfmaq_laneq_f16(v31, v7, v23, 7);

        __builtin_prefetch(b + 64, 0, 3);
        __builtin_prefetch(a + 64, 0, 3);

        b += 64;
        a += 64;
      }

      v24 = vaddq_f16(vld1q_f16(c), v24);
      v25 = vaddq_f16(vld1q_f16(c + ldc), v25);
      v26 = vaddq_f16(vld1q_f16(c + 2 * ldc), v26);
      v27 = vaddq_f16(vld1q_f16(c + 3 * ldc), v27);
      v28 = vaddq_f16(vld1q_f16(c + 4 * ldc), v28);
      v29 = vaddq_f16(vld1q_f16(c + 5 * ldc), v29);
      v30 = vaddq_f16(vld1q_f16(c + 6 * ldc), v30);
      v31 = vaddq_f16(vld1q_f16(c + 7 * ldc), v31);

      vst1q_f16(c, v24);
      vst1q_f16(c + ldc, v25);
      vst1q_f16(c + 2 * ldc, v26);
      vst1q_f16(c + 3 * ldc, v27);
      //
      vst1q_f16(c + 4 * ldc, v28);
      vst1q_f16(c + 5 * ldc, v29);
      vst1q_f16(c + 6 * ldc, v30);
      vst1q_f16(c + 7 * ldc, v31);

      c += 8;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}

/**
 * @brief hgemm 8x8 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_8x8(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 8 == 0 && K % 8 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += VL_FP16) {
    for (j = 0; j < N; j += VL_FP16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      for (l = 0; l < K; l += VL_FP16) {
        float16x8_t v24 = {0};
        float16x8_t v25 = {0};
        float16x8_t v26 = {0};
        float16x8_t v27 = {0};
        float16x8_t v28 = {0};
        float16x8_t v29 = {0};
        float16x8_t v30 = {0};
        float16x8_t v31 = {0};

        float16x8_t v0 = vld1q_f16(b);
        float16x8_t v16 = vld1q_f16(a);

        v24 = vfmaq_laneq_f16(v24, v0, v16, 0);
        v25 = vfmaq_laneq_f16(v25, v0, v16, 1);
        v26 = vfmaq_laneq_f16(v26, v0, v16, 2);
        v27 = vfmaq_laneq_f16(v27, v0, v16, 3);
        v28 = vfmaq_laneq_f16(v28, v0, v16, 4);
        v29 = vfmaq_laneq_f16(v29, v0, v16, 5);
        v30 = vfmaq_laneq_f16(v30, v0, v16, 6);
        v31 = vfmaq_laneq_f16(v31, v0, v16, 7);

        float16x8_t v1 = vld1q_f16(b + 8);
        float16x8_t v17 = vld1q_f16(a + 8);

        v24 = vfmaq_laneq_f16(v24, v1, v17, 0);
        v25 = vfmaq_laneq_f16(v25, v1, v17, 1);
        v26 = vfmaq_laneq_f16(v26, v1, v17, 2);
        v27 = vfmaq_laneq_f16(v27, v1, v17, 3);
        v28 = vfmaq_laneq_f16(v28, v1, v17, 4);
        v29 = vfmaq_laneq_f16(v29, v1, v17, 5);
        v30 = vfmaq_laneq_f16(v30, v1, v17, 6);
        v31 = vfmaq_laneq_f16(v31, v1, v17, 7);

        float16x8_t v2 = vld1q_f16(b + 16);
        float16x8_t v18 = vld1q_f16(a + 16);

        v24 = vfmaq_laneq_f16(v24, v2, v18, 0);
        v25 = vfmaq_laneq_f16(v25, v2, v18, 1);
        v26 = vfmaq_laneq_f16(v26, v2, v18, 2);
        v27 = vfmaq_laneq_f16(v27, v2, v18, 3);
        v28 = vfmaq_laneq_f16(v28, v2, v18, 4);
        v29 = vfmaq_laneq_f16(v29, v2, v18, 5);
        v30 = vfmaq_laneq_f16(v30, v2, v18, 6);
        v31 = vfmaq_laneq_f16(v31, v2, v18, 7);

        float16x8_t v3 = vld1q_f16(b + 24);
        float16x8_t v19 = vld1q_f16(a + 24);

        v24 = vfmaq_laneq_f16(v24, v3, v19, 0);
        v25 = vfmaq_laneq_f16(v25, v3, v19, 1);
        v26 = vfmaq_laneq_f16(v26, v3, v19, 2);
        v27 = vfmaq_laneq_f16(v27, v3, v19, 3);
        v28 = vfmaq_laneq_f16(v28, v3, v19, 4);
        v29 = vfmaq_laneq_f16(v29, v3, v19, 5);
        v30 = vfmaq_laneq_f16(v30, v3, v19, 6);
        v31 = vfmaq_laneq_f16(v31, v3, v19, 7);

        float16x8_t v4 = vld1q_f16(b + 32);
        float16x8_t v20 = vld1q_f16(a + 32);

        v24 = vfmaq_laneq_f16(v24, v4, v20, 0);
        v25 = vfmaq_laneq_f16(v25, v4, v20, 1);
        v26 = vfmaq_laneq_f16(v26, v4, v20, 2);
        v27 = vfmaq_laneq_f16(v27, v4, v20, 3);
        v28 = vfmaq_laneq_f16(v28, v4, v20, 4);
        v29 = vfmaq_laneq_f16(v29, v4, v20, 5);
        v30 = vfmaq_laneq_f16(v30, v4, v20, 6);
        v31 = vfmaq_laneq_f16(v31, v4, v20, 7);

        float16x8_t v5 = vld1q_f16(b + 40);
        float16x8_t v21 = vld1q_f16(a + 40);

        v24 = vfmaq_laneq_f16(v24, v5, v21, 0);
        v25 = vfmaq_laneq_f16(v25, v5, v21, 1);
        v26 = vfmaq_laneq_f16(v26, v5, v21, 2);
        v27 = vfmaq_laneq_f16(v27, v5, v21, 3);
        v28 = vfmaq_laneq_f16(v28, v5, v21, 4);
        v29 = vfmaq_laneq_f16(v29, v5, v21, 5);
        v30 = vfmaq_laneq_f16(v30, v5, v21, 6);
        v31 = vfmaq_laneq_f16(v31, v5, v21, 7);

        float16x8_t v6 = vld1q_f16(b + 48);
        float16x8_t v22 = vld1q_f16(a + 48);

        v24 = vfmaq_laneq_f16(v24, v6, v22, 0);
        v25 = vfmaq_laneq_f16(v25, v6, v22, 1);
        v26 = vfmaq_laneq_f16(v26, v6, v22, 2);
        v27 = vfmaq_laneq_f16(v27, v6, v22, 3);
        v28 = vfmaq_laneq_f16(v28, v6, v22, 4);
        v29 = vfmaq_laneq_f16(v29, v6, v22, 5);
        v30 = vfmaq_laneq_f16(v30, v6, v22, 6);
        v31 = vfmaq_laneq_f16(v31, v6, v22, 7);

        float16x8_t v7 = vld1q_f16(b + 56);
        float16x8_t v23 = vld1q_f16(a + 56);

        v24 = vfmaq_laneq_f16(v24, v7, v23, 0);
        v25 = vfmaq_laneq_f16(v25, v7, v23, 1);
        v26 = vfmaq_laneq_f16(v26, v7, v23, 2);
        v27 = vfmaq_laneq_f16(v27, v7, v23, 3);
        v28 = vfmaq_laneq_f16(v28, v7, v23, 4);
        v29 = vfmaq_laneq_f16(v29, v7, v23, 5);
        v30 = vfmaq_laneq_f16(v30, v7, v23, 6);
        v31 = vfmaq_laneq_f16(v31, v7, v23, 7);

        vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v24))));
        vst1q_f32(
          c + 4, vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v24))));

        vst1q_f32(c + ldc, vaddq_f32(vld1q_f32(c + ldc),
                                     vcvt_f32_f16(vget_low_f16(v25))));
        vst1q_f32(c + 4 + ldc, vaddq_f32(vld1q_f32(c + 4 + ldc),
                                         vcvt_f32_f16(vget_high_f16(v25))));

        vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v26))));
        vst1q_f32(c + 4 + 2 * ldc, vaddq_f32(vld1q_f32(c + 4 + 2 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v26))));

        vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v27))));
        vst1q_f32(c + 4 + 3 * ldc, vaddq_f32(vld1q_f32(c + 4 + 3 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v27))));

        vst1q_f32(c + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v28))));
        vst1q_f32(c + 4 + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 + 4 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v28))));

        vst1q_f32(c + 5 * ldc, vaddq_f32(vld1q_f32(c + 5 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v29))));
        vst1q_f32(c + 4 + 5 * ldc, vaddq_f32(vld1q_f32(c + 4 + 5 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v29))));

        vst1q_f32(c + 6 * ldc, vaddq_f32(vld1q_f32(c + 6 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v30))));
        vst1q_f32(c + 4 + 6 * ldc, vaddq_f32(vld1q_f32(c + 4 + 6 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v30))));

        vst1q_f32(c + 7 * ldc, vaddq_f32(vld1q_f32(c + 7 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v31))));
        vst1q_f32(c + 4 + 7 * ldc, vaddq_f32(vld1q_f32(c + 4 + 7 * ldc),
                                             vcvt_f32_f16(vget_high_f16(v31))));

        __builtin_prefetch(b + 64, 0, 3);
        __builtin_prefetch(a + 64, 0, 3);

        b += 64;
        a += 64;
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
