// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_pack.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is for half-precision packing for kernel-based GEMM
 */

#include <assert.h>
#include <hgemm_common.h>

/**
 * @brief packing function of input matrix A
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param lda leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_A4(unsigned int M, unsigned int K, const __fp16 *src,
                unsigned int lda, const __fp16 *dst) {

  assert(K != 0 && M != 0 && K % 4 == 0 && M % 4 == 0);
  unsigned int i, j;

  __fp16 *a_off, *a_off1, *a_off2, *a_off3, *a_off4;
  __fp16 *b_off;
  __fp16 c1, c2, c3, c4;
  __fp16 c5, c6, c7, c8;
  __fp16 c9, c10, c11, c12;
  __fp16 c13, c14, c15, c16;

  a_off = (__fp16 *)src;
  b_off = (__fp16 *)dst;

  j = (M >> 2);
  do {
    a_off1 = a_off;
    a_off2 = a_off1 + lda;
    a_off3 = a_off2 + lda;
    a_off4 = a_off3 + lda;
    a_off += 4 * lda;

    i = (K >> 2);
    do {
      c1 = *(a_off1 + 0);
      c2 = *(a_off1 + 1);
      c3 = *(a_off1 + 2);
      c4 = *(a_off1 + 3);

      c5 = *(a_off2 + 0);
      c6 = *(a_off2 + 1);
      c7 = *(a_off2 + 2);
      c8 = *(a_off2 + 3);

      c9 = *(a_off3 + 0);
      c10 = *(a_off3 + 1);
      c11 = *(a_off3 + 2);
      c12 = *(a_off3 + 3);

      c13 = *(a_off4 + 0);
      c14 = *(a_off4 + 1);
      c15 = *(a_off4 + 2);
      c16 = *(a_off4 + 3);

      *(b_off + 0) = c1;
      *(b_off + 1) = c5;
      *(b_off + 2) = c9;
      *(b_off + 3) = c13;

      *(b_off + 4) = c2;
      *(b_off + 5) = c6;
      *(b_off + 6) = c10;
      *(b_off + 7) = c14;

      *(b_off + 8) = c3;
      *(b_off + 9) = c7;
      *(b_off + 10) = c11;
      *(b_off + 11) = c15;

      *(b_off + 12) = c4;
      *(b_off + 13) = c8;
      *(b_off + 14) = c12;
      *(b_off + 15) = c16;

      a_off1 += 4;
      a_off2 += 4;
      a_off3 += 4;
      a_off4 += 4;

      b_off += 16;
      i--;
    } while (i > 0);
    j--;
  } while (j > 0);
}

/**
 * @brief packing function of input matrix A
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param lda leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_A8(unsigned int M, unsigned int K, const __fp16 *src,
                unsigned int lda, const __fp16 *dst) {

  assert(K != 0 && M != 0 && K % 8 == 0 && M % 8 == 0);

  uint16x4_t msk = {0xFFFF, 0xFFFF, 0x0000, 0x0000};
  uint16x4_t inv_msk = {0x0000, 0x0000, 0xFFFF, 0xFFFF};

  const __fp16 *a_off = (__fp16 *)src;
  __fp16 *b_off = (__fp16 *)dst;

  for (unsigned int i = 0; i < M; i += 8) {
    const __fp16 *a_off1 = a_off;
    const __fp16 *a_off2 = a_off1 + lda;
    const __fp16 *a_off3 = a_off2 + lda;
    const __fp16 *a_off4 = a_off3 + lda;
    const __fp16 *a_off5 = a_off4 + lda;
    const __fp16 *a_off6 = a_off5 + lda;
    const __fp16 *a_off7 = a_off6 + lda;
    const __fp16 *a_off8 = a_off7 + lda;
    a_off += 8 * lda;

    for (unsigned int j = 0; j < K; j += 8) {
      float16x8_t _v0 = vld1q_f16(a_off1);
      float16x8_t _v1 = vld1q_f16(a_off2);
      float16x8_t _v2 = vld1q_f16(a_off3);
      float16x8_t _v3 = vld1q_f16(a_off4);

      float16x8_t _v4 = vld1q_f16(a_off5);
      float16x8_t _v5 = vld1q_f16(a_off6);
      float16x8_t _v6 = vld1q_f16(a_off7);
      float16x8_t _v7 = vld1q_f16(a_off8);

      a_off1 += 8;
      a_off2 += 8;
      a_off3 += 8;
      a_off4 += 8;
      a_off5 += 8;
      a_off6 += 8;
      a_off7 += 8;
      a_off8 += 8;

      float16x8x2_t _vv0 = vtrnq_f16(_v0, _v1);
      float16x8x2_t _vv1 = vtrnq_f16(_v2, _v3);
      float16x8x2_t _vv2 = vtrnq_f16(_v4, _v5);
      float16x8x2_t _vv3 = vtrnq_f16(_v6, _v7);

      float16x8_t _v8 =
        vcombine_f16(vget_low_f16(_vv0.val[0]), vget_low_f16(_vv1.val[0]));
      float16x8_t _v9 =
        vcombine_f16(vget_low_f16(_vv0.val[1]), vget_low_f16(_vv1.val[1]));
      float16x8_t _v10 =
        vcombine_f16(vget_high_f16(_vv0.val[0]), vget_high_f16(_vv1.val[0]));
      float16x8_t _v11 =
        vcombine_f16(vget_high_f16(_vv0.val[1]), vget_high_f16(_vv1.val[1]));

      float16x8_t _v12 =
        vcombine_f16(vget_low_f16(_vv2.val[0]), vget_low_f16(_vv3.val[0]));
      float16x8_t _v13 =
        vcombine_f16(vget_low_f16(_vv2.val[1]), vget_low_f16(_vv3.val[1]));
      float16x8_t _v14 =
        vcombine_f16(vget_high_f16(_vv2.val[0]), vget_high_f16(_vv3.val[0]));
      float16x8_t _v15 =
        vcombine_f16(vget_high_f16(_vv2.val[1]), vget_high_f16(_vv3.val[1]));

      // pack-in-pack
      float16x4_t tmp_low_v8 = vget_low_f16(_v8);
      float16x4_t tmp_high_v8 = vget_high_f16(_v8);
      float16x4_t mid_v8 = vext_f16(tmp_low_v8, tmp_high_v8, 2);

      float16x4_t tmp_low_v9 = vget_low_f16(_v9);
      float16x4_t tmp_high_v9 = vget_high_f16(_v9);
      float16x4_t mid_v9 = vext_f16(tmp_low_v9, tmp_high_v9, 2);

      float16x4_t tmp_low_v10 = vget_low_f16(_v10);
      float16x4_t tmp_high_v10 = vget_high_f16(_v10);
      float16x4_t mid_v10 = vext_f16(tmp_low_v10, tmp_high_v10, 2);

      float16x4_t tmp_low_v11 = vget_low_f16(_v11);
      float16x4_t tmp_high_v11 = vget_high_f16(_v11);
      float16x4_t mid_v11 = vext_f16(tmp_low_v11, tmp_high_v11, 2);

      float16x4_t tmp_low_v12 = vget_low_f16(_v12);
      float16x4_t tmp_high_v12 = vget_high_f16(_v12);
      float16x4_t mid_v12 = vext_f16(tmp_low_v12, tmp_high_v12, 2);

      float16x4_t tmp_low_v13 = vget_low_f16(_v13);
      float16x4_t tmp_high_v13 = vget_high_f16(_v13);
      float16x4_t mid_v13 = vext_f16(tmp_low_v13, tmp_high_v13, 2);

      float16x4_t tmp_low_v14 = vget_low_f16(_v14);
      float16x4_t tmp_high_v14 = vget_high_f16(_v14);
      float16x4_t mid_v14 = vext_f16(tmp_low_v14, tmp_high_v14, 2);

      float16x4_t tmp_low_v15 = vget_low_f16(_v15);
      float16x4_t tmp_high_v15 = vget_high_f16(_v15);
      float16x4_t mid_v15 = vext_f16(tmp_low_v15, tmp_high_v15, 2);

      _v8 = vcombine_f16(vbsl_f16(msk, tmp_low_v8, mid_v8),
                         vbsl_f16(msk, tmp_low_v12, mid_v12));
      _v12 = vcombine_f16(vbsl_f16(msk, tmp_low_v9, mid_v9),
                          vbsl_f16(msk, tmp_low_v13, mid_v13));
      _v9 = vcombine_f16(vbsl_f16(inv_msk, tmp_high_v8, mid_v8),
                         vbsl_f16(inv_msk, tmp_high_v12, mid_v12));
      _v13 = vcombine_f16(vbsl_f16(inv_msk, tmp_high_v9, mid_v9),
                          vbsl_f16(inv_msk, tmp_high_v13, mid_v13));
      _v10 = vcombine_f16(vbsl_f16(msk, tmp_low_v10, mid_v10),
                          vbsl_f16(msk, tmp_low_v14, mid_v14));
      _v14 = vcombine_f16(vbsl_f16(msk, tmp_low_v11, mid_v11),
                          vbsl_f16(msk, tmp_low_v15, mid_v15));
      _v11 = vcombine_f16(vbsl_f16(inv_msk, tmp_high_v10, mid_v10),
                          vbsl_f16(inv_msk, tmp_high_v14, mid_v14));
      _v15 = vcombine_f16(vbsl_f16(inv_msk, tmp_high_v11, mid_v11),
                          vbsl_f16(inv_msk, tmp_high_v15, mid_v15));

      vst1q_f16(b_off + 0, _v8);
      vst1q_f16(b_off + 8, _v12);
      vst1q_f16(b_off + 16, _v9);
      vst1q_f16(b_off + 24, _v13);
      vst1q_f16(b_off + 32, _v10);
      vst1q_f16(b_off + 40, _v14);
      vst1q_f16(b_off + 48, _v11);
      vst1q_f16(b_off + 56, _v15);
      b_off += 64;
    }
  }
}

/**
 * @brief packing function of input matrix B
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_B4(unsigned int K, unsigned int N, const __fp16 *src,
                unsigned int ldb, const __fp16 *dst) {
  assert(K != 0 && N != 0 && K % 4 == 0 && N % 4 == 0);
  unsigned int i, j;

  __fp16 *a_off, *a_off1, *a_off2, *a_off3, *a_off4;
  __fp16 *b_off, *b_off1;
  __fp16 c1, c2, c3, c4;
  __fp16 c5, c6, c7, c8;
  __fp16 c9, c10, c11, c12;
  __fp16 c13, c14, c15, c16;
  a_off = (__fp16 *)src;
  b_off = (__fp16 *)dst;

  j = (K >> 2);
  do {
    a_off1 = a_off;
    a_off2 = a_off1 + ldb;
    a_off3 = a_off2 + ldb;
    a_off4 = a_off3 + ldb;
    a_off += 4 * ldb;

    b_off1 = b_off;
    b_off += 16;

    i = (N >> 2);
    do {
      c1 = *(a_off1 + 0);
      c2 = *(a_off1 + 1);
      c3 = *(a_off1 + 2);
      c4 = *(a_off1 + 3);

      c5 = *(a_off2 + 0);
      c6 = *(a_off2 + 1);
      c7 = *(a_off2 + 2);
      c8 = *(a_off2 + 3);

      c9 = *(a_off3 + 0);
      c10 = *(a_off3 + 1);
      c11 = *(a_off3 + 2);
      c12 = *(a_off3 + 3);

      c13 = *(a_off4 + 0);
      c14 = *(a_off4 + 1);
      c15 = *(a_off4 + 2);
      c16 = *(a_off4 + 3);

      a_off1 += 4;
      a_off2 += 4;
      a_off3 += 4;
      a_off4 += 4;

      *(b_off1 + 0) = c1;
      *(b_off1 + 1) = c2;
      *(b_off1 + 2) = c3;
      *(b_off1 + 3) = c4;

      *(b_off1 + 4) = c5;
      *(b_off1 + 5) = c6;
      *(b_off1 + 6) = c7;
      *(b_off1 + 7) = c8;

      *(b_off1 + 8) = c9;
      *(b_off1 + 9) = c10;
      *(b_off1 + 10) = c11;
      *(b_off1 + 11) = c12;

      *(b_off1 + 12) = c13;
      *(b_off1 + 13) = c14;
      *(b_off1 + 14) = c15;
      *(b_off1 + 15) = c16;

      b_off1 += K * 4;
      i--;
    } while (i > 0);
    j--;
  } while (j > 0);
}

/**
 * @brief packing function of input matrix B
 *
 * @param M length of the row of the matrix
 * @param K length of the col of the matrix
 * @param src input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param dst output of packed data of the matrix
 */
void packing_B8(unsigned int K, unsigned int N, const __fp16 *src,
                unsigned int ldb, const __fp16 *dst) {
  assert(K != 0 && N != 0 && N % 8 == 0);

  for (int i = 0; i < K; i++) {
    const __fp16 *a_off = src + i * ldb;
    __fp16 *b_off = (__fp16 *)dst + i * 8;
    for (int j = 0; j < N; j += 8) {
      float16x8_t v = vld1q_f16(a_off);
      a_off += 8;

      vst1q_f16(b_off, v);
      b_off += 8 * K;
    }
  }
}
