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
 * @param m length of the row of the matrix
 * @param k length of the col of the matrix
 * @param from input of original source of the matrix
 * @param lda leading dimension of the matrix
 * @param to output of packed data of the matrix
 */
void packA_4(unsigned int m, unsigned int k, const __fp16 *from,
             unsigned int lda, const __fp16 *to) {

  assert(k != 0 && m != 0 && k % 4 == 0 && m % 4 == 0);
  unsigned int i, j;

  __fp16 *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
  __fp16 *b_offset;
  __fp16 ctemp1, ctemp2, ctemp3, ctemp4;
  __fp16 ctemp5, ctemp6, ctemp7, ctemp8;
  __fp16 ctemp9, ctemp10, ctemp11, ctemp12;
  __fp16 ctemp13, ctemp14, ctemp15, ctemp16;

  a_offset = (__fp16 *)from;
  b_offset = (__fp16 *)to;

  j = (m >> 2);
  do {
    a_offset1 = a_offset;
    a_offset2 = a_offset1 + lda;
    a_offset3 = a_offset2 + lda;
    a_offset4 = a_offset3 + lda;
    a_offset += 4 * lda;

    i = (k >> 2);
    do {
      ctemp1 = *(a_offset1 + 0);
      ctemp2 = *(a_offset1 + 1);
      ctemp3 = *(a_offset1 + 2);
      ctemp4 = *(a_offset1 + 3);

      ctemp5 = *(a_offset2 + 0);
      ctemp6 = *(a_offset2 + 1);
      ctemp7 = *(a_offset2 + 2);
      ctemp8 = *(a_offset2 + 3);

      ctemp9 = *(a_offset3 + 0);
      ctemp10 = *(a_offset3 + 1);
      ctemp11 = *(a_offset3 + 2);
      ctemp12 = *(a_offset3 + 3);

      ctemp13 = *(a_offset4 + 0);
      ctemp14 = *(a_offset4 + 1);
      ctemp15 = *(a_offset4 + 2);
      ctemp16 = *(a_offset4 + 3);

      *(b_offset + 0) = ctemp1;
      *(b_offset + 1) = ctemp5;
      *(b_offset + 2) = ctemp9;
      *(b_offset + 3) = ctemp13;

      *(b_offset + 4) = ctemp2;
      *(b_offset + 5) = ctemp6;
      *(b_offset + 6) = ctemp10;
      *(b_offset + 7) = ctemp14;

      *(b_offset + 8) = ctemp3;
      *(b_offset + 9) = ctemp7;
      *(b_offset + 10) = ctemp11;
      *(b_offset + 11) = ctemp15;

      *(b_offset + 12) = ctemp4;
      *(b_offset + 13) = ctemp8;
      *(b_offset + 14) = ctemp12;
      *(b_offset + 15) = ctemp16;

      a_offset1 += 4;
      a_offset2 += 4;
      a_offset3 += 4;
      a_offset4 += 4;

      b_offset += 16;
      i--;
    } while (i > 0);
    j--;
  } while (j > 0);
}

/**
 * @brief packing function of input matrix A
 * 
 * @param m length of the row of the matrix
 * @param k length of the col of the matrix
 * @param from input of original source of the matrix
 * @param lda leading dimension of the matrix
 * @param to output of packed data of the matrix
 */
void packA_8(unsigned int m, unsigned int k, const __fp16 *from,
             unsigned int lda, const __fp16 *to) {

  assert(k != 0 && m != 0 && k % 8 == 0 && m % 8 == 0);

  uint16x4_t msk = {0xFFFF, 0xFFFF, 0x0000, 0x0000};
  uint16x4_t inv_msk = {0x0000, 0x0000, 0xFFFF, 0xFFFF};

  const __fp16 *a_offset = (__fp16 *)from;
  __fp16 *b_offset = (__fp16 *)to;

  for (unsigned int i = 0; i < m; i += 8) {
    const __fp16 *a_offset1 = a_offset;
    const __fp16 *a_offset2 = a_offset1 + lda;
    const __fp16 *a_offset3 = a_offset2 + lda;
    const __fp16 *a_offset4 = a_offset3 + lda;
    const __fp16 *a_offset5 = a_offset4 + lda;
    const __fp16 *a_offset6 = a_offset5 + lda;
    const __fp16 *a_offset7 = a_offset6 + lda;
    const __fp16 *a_offset8 = a_offset7 + lda;
    a_offset += 8 * lda;

    for (unsigned int j = 0; j < k; j += 8) {
      float16x8_t _v0 = vld1q_f16(a_offset1);
      float16x8_t _v1 = vld1q_f16(a_offset2);
      float16x8_t _v2 = vld1q_f16(a_offset3);
      float16x8_t _v3 = vld1q_f16(a_offset4);

      float16x8_t _v4 = vld1q_f16(a_offset5);
      float16x8_t _v5 = vld1q_f16(a_offset6);
      float16x8_t _v6 = vld1q_f16(a_offset7);
      float16x8_t _v7 = vld1q_f16(a_offset8);

      a_offset1 += 8;
      a_offset2 += 8;
      a_offset3 += 8;
      a_offset4 += 8;
      a_offset5 += 8;
      a_offset6 += 8;
      a_offset7 += 8;
      a_offset8 += 8;

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

      vst1q_f16(b_offset + 0, _v8);
      vst1q_f16(b_offset + 8, _v12);
      vst1q_f16(b_offset + 16, _v9);
      vst1q_f16(b_offset + 24, _v13);
      vst1q_f16(b_offset + 32, _v10);
      vst1q_f16(b_offset + 40, _v14);
      vst1q_f16(b_offset + 48, _v11);
      vst1q_f16(b_offset + 56, _v15);
      b_offset += 64;
    }
  }
}

/**
 * @brief packing function of input matrix B
 * 
 * @param m length of the row of the matrix
 * @param k length of the col of the matrix
 * @param from input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param to output of packed data of the matrix
 */
void packB_4(unsigned int k, unsigned int n, const __fp16 *from,
             unsigned int ldb, const __fp16 *to) {
  assert(k != 0 && n != 0 && k % 4 == 0 && n % 4 == 0);
  unsigned int i, j;

  __fp16 *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
  __fp16 *b_offset, *b_offset1;
  __fp16 ctemp1, ctemp2, ctemp3, ctemp4;
  __fp16 ctemp5, ctemp6, ctemp7, ctemp8;
  __fp16 ctemp9, ctemp10, ctemp11, ctemp12;
  __fp16 ctemp13, ctemp14, ctemp15, ctemp16;
  a_offset = (__fp16 *)from;
  b_offset = (__fp16 *)to;

  j = (k >> 2);
  do {
    a_offset1 = a_offset;
    a_offset2 = a_offset1 + ldb;
    a_offset3 = a_offset2 + ldb;
    a_offset4 = a_offset3 + ldb;
    a_offset += 4 * ldb;

    b_offset1 = b_offset;
    b_offset += 16;

    i = (n >> 2);
    do {
      ctemp1 = *(a_offset1 + 0);
      ctemp2 = *(a_offset1 + 1);
      ctemp3 = *(a_offset1 + 2);
      ctemp4 = *(a_offset1 + 3);

      ctemp5 = *(a_offset2 + 0);
      ctemp6 = *(a_offset2 + 1);
      ctemp7 = *(a_offset2 + 2);
      ctemp8 = *(a_offset2 + 3);

      ctemp9 = *(a_offset3 + 0);
      ctemp10 = *(a_offset3 + 1);
      ctemp11 = *(a_offset3 + 2);
      ctemp12 = *(a_offset3 + 3);

      ctemp13 = *(a_offset4 + 0);
      ctemp14 = *(a_offset4 + 1);
      ctemp15 = *(a_offset4 + 2);
      ctemp16 = *(a_offset4 + 3);

      a_offset1 += 4;
      a_offset2 += 4;
      a_offset3 += 4;
      a_offset4 += 4;

      *(b_offset1 + 0) = ctemp1;
      *(b_offset1 + 1) = ctemp2;
      *(b_offset1 + 2) = ctemp3;
      *(b_offset1 + 3) = ctemp4;

      *(b_offset1 + 4) = ctemp5;
      *(b_offset1 + 5) = ctemp6;
      *(b_offset1 + 6) = ctemp7;
      *(b_offset1 + 7) = ctemp8;

      *(b_offset1 + 8) = ctemp9;
      *(b_offset1 + 9) = ctemp10;
      *(b_offset1 + 10) = ctemp11;
      *(b_offset1 + 11) = ctemp12;

      *(b_offset1 + 12) = ctemp13;
      *(b_offset1 + 13) = ctemp14;
      *(b_offset1 + 14) = ctemp15;
      *(b_offset1 + 15) = ctemp16;

      b_offset1 += k * 4;
      i--;
    } while (i > 0);
    j--;
  } while (j > 0);
}

/**
 * @brief packing function of input matrix B
 * 
 * @param m length of the row of the matrix
 * @param k length of the col of the matrix
 * @param from input of original source of the matrix
 * @param ldb leading dimension of the matrix
 * @param to output of packed data of the matrix
 */
void packB_8(unsigned int k, unsigned int n, const __fp16 *from,
             unsigned int ldb, const __fp16 *to) {
  assert(k != 0 && n != 0 && n % 8 == 0);

  for (int i = 0; i < k; i++) {
    const __fp16 *a_offset1 = from + i * ldb;
    __fp16 *b_offset = (__fp16 *)to + i * 8;
    for (int j = 0; j < n; j += 8) {
      float16x8_t _v0 = vld1q_f16(a_offset1);
      a_offset1 += 8;

      vst1q_f16(b_offset, _v0);
      b_offset += 8 * k;
    }
  }
}
