// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_4x4.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 4x4 kernel
 *
 */

#include <hgemm_common.h>
#include <stdlib.h>

/**
 * @brief hgemm 4x4 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading dimension of matrix C
 */
void hgemm_kernel_4x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 4 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += VL_FP16_HALF) {
    for (j = 0; j < N; j += VL_FP16_HALF) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x4_t v24 = {0};
      float16x4_t v25 = {0};
      float16x4_t v26 = {0};
      float16x4_t v27 = {0};

      for (l = 0; l < K; l += VL_FP16_HALF) {
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

/**
 * @brief hgemm 4x4 kernel sc = sa * sb
 *
 * @param m length of the row of matrix A
 * @param n length of the col of matrix B
 * @param k length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading dimension of matrix C
 */
void hgemm_kernel_4x4(unsigned int M, unsigned int N, unsigned int K,
                      __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 4 == 0 && N % 4 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += VL_FP16_HALF) {
    for (j = 0; j < N; j += VL_FP16_HALF) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);

      float16x4_t v24 = {0};
      float16x4_t v25 = {0};
      float16x4_t v26 = {0};
      float16x4_t v27 = {0};

      for (l = 0; l < K; l += VL_FP16_HALF) {
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

        vst1_f32(c, vadd_f32(vld1_f32(c), vcvt_f32_f16(v24)));
        vst1_f32(c + ldc, vadd_f32(vld1_f32(c + ldc), vcvt_f32_f16(v25)));
        vst1_f32(c + 2 * ldc, vadd_f32(vld1_f32(c + 2 * ldc), vcvt_f32_f16(v26)));
        vst1_f32(c + 3 * ldc,  vadd_f32(vld1_f32(c + 3 * ldc), vcvt_f32_f16(v27)));
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

