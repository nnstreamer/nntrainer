// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ggml_cpu_impl.cpp
 * @date	03 April 2025
 * @brief	This is ggml cpu implemenation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "nntr_ggml_essential.h"
#include <assert.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <chrono>
using std::chrono::nanoseconds; // or microseconds
using std::chrono::microseconds; // or microseconds
using std::chrono::milliseconds; // or microseconds
using std::chrono::seconds; // or microseconds
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;

#ifdef __cplusplus
// restrict not standard in C++
#if defined(__GNUC__)
#define GGML_RESTRICT __restrict__
#elif defined(__clang__)
#define GGML_RESTRICT __restrict
#elif defined(_MSC_VER)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT
#endif
#else
#if defined(_MSC_VER) && (__STDC_VERSION__ < 201112L)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT restrict
#endif
#endif

#define GGML_UNUSED(x) (void)(x)
#define UNUSED GGML_UNUSED

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

struct block_q4_Kx8 {
  ggml_half d[8];     // super-block scale for quantized scales
  ggml_half dmin[8];  // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};
static_assert(sizeof(block_q4_Kx8) ==
                sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4,
              "wrong q4_K block size/padding");

struct block_q8_Kx4 {
  float d[4];              // delta
  int8_t qs[QK_K * 4];     // quants
  int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};
static_assert(sizeof(block_q8_Kx4) ==
                sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t),
              "wrong q8_K block size/padding");

static inline int nearest_int(float fval) {
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

/*
 RUNTIME ACTIVATION QUANTIZATION
 */
static void ggml_quantize_mat_q8_K_4x8(const float *GGML_RESTRICT x,
                                       void *GGML_RESTRICT vy, int64_t k) {
  assert(QK_K == 256);
  // assert(k % QK_K == 0);
  // const int nb = k / QK_K;
  const int nb = (k + QK_K - 1) / QK_K;

  block_q8_Kx4 *GGML_RESTRICT y = (block_q8_Kx4 *)vy;

#if defined(__AVX2__)
  float iscale[4];
  __m256 srcv[4][32];
  __m256 iscale_vec[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 256);
      __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 8);
      __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 16);
      __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 24);

      // Compute max(abs(e)) for the block
      const __m256 signBit = _mm256_set1_ps(-0.0f);
      __m256 abs0 = _mm256_andnot_ps(signBit, v0);
      __m256 abs1 = _mm256_andnot_ps(signBit, v1);
      __m256 abs2 = _mm256_andnot_ps(signBit, v2);
      __m256 abs3 = _mm256_andnot_ps(signBit, v3);

      __m256 maxAbs = _mm256_max_ps(abs0, abs1);
      maxAbs = _mm256_max_ps(maxAbs, abs2);
      maxAbs = _mm256_max_ps(maxAbs, abs3);

      __m256 mask0 = _mm256_cmp_ps(maxAbs, v0, _CMP_EQ_OQ);
      __m256 mask1 = _mm256_cmp_ps(maxAbs, v1, _CMP_EQ_OQ);
      __m256 mask2 = _mm256_cmp_ps(maxAbs, v2, _CMP_EQ_OQ);
      __m256 mask3 = _mm256_cmp_ps(maxAbs, v3, _CMP_EQ_OQ);

      __m256 maskAbs =
        _mm256_or_ps(_mm256_or_ps(mask0, mask1), _mm256_or_ps(mask2, mask3));

      srcv[row_iter][0] = v0;
      srcv[row_iter][1] = v1;
      srcv[row_iter][2] = v2;
      srcv[row_iter][3] = v3;

      for (int sb = 1; sb < 8; sb++) {
        // Temporarily stores absolute quant values
        __m256 tempAbs = maxAbs;

        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32);
        __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 8);
        __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 16);
        __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 24);

        // Compute max(abs(e)) for the block
        __m256 abs0 = _mm256_andnot_ps(signBit, v0);
        __m256 abs1 = _mm256_andnot_ps(signBit, v1);
        __m256 abs2 = _mm256_andnot_ps(signBit, v2);
        __m256 abs3 = _mm256_andnot_ps(signBit, v3);

        maxAbs = _mm256_max_ps(maxAbs, abs0);
        maxAbs = _mm256_max_ps(maxAbs, abs1);
        maxAbs = _mm256_max_ps(maxAbs, abs2);
        maxAbs = _mm256_max_ps(maxAbs, abs3);

        __m256 mask_prev = _mm256_cmp_ps(tempAbs, maxAbs, _CMP_EQ_OQ);
        maskAbs = _mm256_and_ps(maskAbs, mask_prev);

        mask0 = _mm256_cmp_ps(maxAbs, v0, _CMP_EQ_OQ);
        mask1 = _mm256_cmp_ps(maxAbs, v1, _CMP_EQ_OQ);
        mask2 = _mm256_cmp_ps(maxAbs, v2, _CMP_EQ_OQ);
        mask3 = _mm256_cmp_ps(maxAbs, v3, _CMP_EQ_OQ);

        __m256 mask_curr =
          _mm256_or_ps(_mm256_or_ps(mask0, mask1), _mm256_or_ps(mask2, mask3));
        maskAbs = _mm256_or_ps(maskAbs, mask_curr);

        srcv[row_iter][sb * 4] = v0;
        srcv[row_iter][sb * 4 + 1] = v1;
        srcv[row_iter][sb * 4 + 2] = v2;
        srcv[row_iter][sb * 4 + 3] = v3;
      }

      __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                               _mm256_castps256_ps128(maxAbs));
      max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
      max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
      const float maxScalar = _mm_cvtss_f32(max4);

      __m256 maxScalarVec = _mm256_set1_ps(maxScalar);

      __m256 mask_next = _mm256_cmp_ps(maxScalarVec, maxAbs, _CMP_EQ_OQ);
      __m256 finalMask = _mm256_and_ps(maskAbs, mask_next);

      const int mask = _mm256_movemask_ps(finalMask);
      iscale[row_iter] = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;

      if (mask) {
        iscale[row_iter] = (maxScalar != 0.0f) ? -127.f / maxScalar : 0.0f;
      }

      y[i].d[row_iter] = maxScalar ? 1 / iscale[row_iter] : 0;
      iscale_vec[row_iter] = _mm256_set1_ps(iscale[row_iter]);
    }

    __m256i quants_interleaved[32];
    for (int j = 0; j < 32; j++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
      __m256 v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
      __m256 v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
      __m256 v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i i0 = _mm256_cvtps_epi32(v0);
      __m256i i1 = _mm256_cvtps_epi32(v1);
      __m256i i2 = _mm256_cvtps_epi32(v2);
      __m256i i3 = _mm256_cvtps_epi32(v3);

      // Convert int32 to int16
      i0 = _mm256_packs_epi32(i0, i1);
      i2 = _mm256_packs_epi32(i2, i3);
      // Convert int16 to int8
      i0 = _mm256_packs_epi16(i0, i2);

      //  Permute and store the quantized weights in the required order after
      //  the pack instruction
      const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      i0 = _mm256_permutevar8x32_epi32(i0, perm);

      _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
      quants_interleaved[j] = i0;
    }

    // Masks to shuffle the quants of corresonding sub blocks for rearraning
    // quants for vectorized bsums computation
    __m256i shuffle_mask_sb2 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15));
    shuffle_mask_sb2 =
      _mm256_permute2f128_si256(shuffle_mask_sb2, shuffle_mask_sb2, 0);
    __m256i shuffle_mask_sb3 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15));
    shuffle_mask_sb3 =
      _mm256_permute2f128_si256(shuffle_mask_sb3, shuffle_mask_sb3, 0);
    __m256i shuffle_mask_sb4 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9));
    shuffle_mask_sb4 =
      _mm256_permute2f128_si256(shuffle_mask_sb4, shuffle_mask_sb4, 0);

    for (int k = 0; k < 4; k++) {
      // Quants from four different sub blocks are taken
      __m256i q0 = quants_interleaved[k * 8 + 0];
      __m256i q1 = quants_interleaved[k * 8 + 1];
      __m256i q2 = quants_interleaved[k * 8 + 2];
      __m256i q3 = quants_interleaved[k * 8 + 3];
      __m256i q4 = quants_interleaved[k * 8 + 4];
      __m256i q5 = quants_interleaved[k * 8 + 5];
      __m256i q6 = quants_interleaved[k * 8 + 6];
      __m256i q7 = quants_interleaved[k * 8 + 7];

      // The below code block has the first half of different sub blocks
      // shuffled and blended so as to process 2 values from each sub block at a
      // time
      __m256i sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
      __m256i sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
      __m256i sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
      sb_h1_interleaved =
        _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
      __m256i sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
      sb_h1_interleaved =
        _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

      __m256i one = _mm256_set1_epi8(1);
      __m256i bsums_r1 = _mm256_maddubs_epi16(one, sb_h1_interleaved);

      for (int l = 0; l < 3; l++) {
        // Quants value shifted to process next two values from each sub block
        q0 = _mm256_srli_epi64(q0, 16);
        q2 = _mm256_srli_epi64(q2, 16);
        q4 = _mm256_srli_epi64(q4, 16);
        q6 = _mm256_srli_epi64(q6, 16);

        sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
        sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
        sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
        sb_h1_interleaved =
          _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
        sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
        sb_h1_interleaved =
          _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

        bsums_r1 = _mm256_add_epi16(
          bsums_r1, _mm256_maddubs_epi16(one, sb_h1_interleaved));
      }

      // The below code block has the second half of different sub blocks
      // shuffled and blended so as to process 2 values from each sub block at a
      // time
      __m256i sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
      __m256i sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
      __m256i sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
      sb_h2_interleaved =
        _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
      __m256i sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
      sb_h2_interleaved =
        _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

      __m256i bsums_r2 = _mm256_maddubs_epi16(one, sb_h2_interleaved);

      for (int l = 0; l < 3; l++) {
        // Quants value shifted to process next two values from each sub block
        q1 = _mm256_srli_epi64(q1, 16);
        q3 = _mm256_srli_epi64(q3, 16);
        q5 = _mm256_srli_epi64(q5, 16);
        q7 = _mm256_srli_epi64(q7, 16);

        sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
        sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
        sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
        sb_h2_interleaved =
          _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
        sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
        sb_h2_interleaved =
          _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

        bsums_r2 = _mm256_add_epi16(
          bsums_r2, _mm256_maddubs_epi16(one, sb_h2_interleaved));
      }

      // Overall bsums in interleaved fashion computed by adding results of both
      // halves
      __m256i bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
      _mm256_storeu_si256((__m256i *)(y[i].bsums + 16 * k), bsums_r);
    }
  }

#else

  // scalar
  const int blck_size_interleave = 8;
  float srcv[4][QK_K];
  float iscale[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max
      float max = 0;

      for (int j = 0; j < QK_K; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
        // Update the maximum value of the corresponding super block
        if (amax < fabsf(srcv[row_iter][j])) {
          amax = fabsf(srcv[row_iter][j]);
          max = srcv[row_iter][j];
        }
      }

      iscale[row_iter] = amax ? -127.f / max : 0;

      y[i].d[row_iter] = amax ? 1 / iscale[row_iter] : 0;
    }

    for (int j = 0; j < QK_K / 4; j++) {
      y[i].bsums[j] = 0;
    }

    // Quants values are interleaved in sequence of eight bytes from
    // corresponding super blocks Bsums values are interleaved in sequence of
    // four bsums from each super block taken for interleaving i.e first four
    // bsums from the first super block, followed by first four bsums from
    // second super block and so on
    for (int j = 0; j < QK_K * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);
      int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

      float x0 = srcv[src_id][src_offset] * iscale[src_id];
      y[i].qs[j] = nearest_int(x0);
      y[i].bsums[index] += y[i].qs[j];
    }
  }
#endif
}

template <int64_t INTER_SIZE, ggml_type PARAM_TYPE>
void ggml_quantize_mat_t(const float *GGML_RESTRICT x, void *GGML_RESTRICT vy,
                         int64_t nrow, int64_t n_per_row);

template <>
void ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(const float *GGML_RESTRICT x,
                                            void *GGML_RESTRICT vy,
                                            int64_t nrow, int64_t n_per_row) {
  assert(nrow == 4);
  UNUSED(nrow);
  ggml_quantize_mat_q8_K_4x8(x, vy, n_per_row);
}

static inline void quantize_row_q8_K_ref(const float *GGML_RESTRICT x,
                                         block_q8_K *GGML_RESTRICT y,
                                         int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {

    float max = 0;
    float amax = 0;
    for (int j = 0; j < QK_K; ++j) {
      float ax = fabsf(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (!amax) {
      y[i].d = 0;
      memset(y[i].qs, 0, QK_K);
      x += QK_K;
      continue;
    }
    // const float iscale = -128.f/max;
    //  We need this change for IQ2_XXS, else the AVX implementation becomes
    //  very awkward
    const float iscale = -127.f / max;
    for (int j = 0; j < QK_K; ++j) {
      int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = MIN(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) {
        sum += y[i].qs[j * 16 + ii];
      }
      y[i].bsums[j] = sum;
    }
    y[i].d = 1 / iscale;
    x += QK_K;
  }
}

void quantize_row_q8_K(const float *GGML_RESTRICT x, void *GGML_RESTRICT y,
                       int64_t k) {
  quantize_row_q8_K_ref(x, (block_q8_K *)y, k);
}
/*
 RUNTIME ACTIVATION QUANTIZATION
 */

/*
GEMM GEMV KERNEL
 */
static void ggml_gemv_q4_K_8x8_q8_K(int n, float *GGML_RESTRICT s, size_t bs,
                                    const void *GGML_RESTRICT vx,
                                    const void *GGML_RESTRICT vy, int nr,
                                    int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  UNUSED(s);
  UNUSED(bs);
  UNUSED(vx);
  UNUSED(vy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(ncols_interleaved);
  UNUSED(blocklen);

#if defined(__AVX2__)
  // Lookup table to convert signed nibbles to signed bytes
  __m256i signextendlut = _mm256_castsi128_si256(
    _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
  signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
  // Shuffle masks to rearrange delta and scale values to multiply with
  // appropriate scales
  __m128i deltamask =
    _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
  __m128i scalemask =
    _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
  // Permute mask used for easier vector processing at later stages
  __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  // Mask to extract nibbles from bytes
  const __m256i m4b = _mm256_set1_epi8(0x0F);

  int64_t b_nb = n / QK_K;

  const block_q4_Kx8 *b_ptr_start = (const block_q4_Kx8 *)vx;
  const block_q8_K *a_ptr_start = (const block_q8_K *)vy;

  // Process Q8_K blocks one by one
  for (int64_t y = 0; y < nr; y++) {

    // Pointers to LHS blocks of block_q8_K format
    const block_q8_K *a_ptr = a_ptr_start + (y * nb);

    // Take group of eight interleaved block_q4_K structures at each pass of the
    // loop and perform dot product operation
    for (int64_t x = 0; x < nc / 8; x++) {

      // Pointers to RHS blocks
      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_row = _mm256_setzero_ps();
      __m256 acc_min_rows = _mm256_setzero_ps();

      for (int64_t b = 0; b < nb; b++) {

        // Load and convert to FP32 scale from block_q8_K
        const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

        // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
        // col_scale_f32 rearranged so as to multiply with appropriate quants
        const __m256 col_scale_f32 =
          GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        __m256i iacc_b = _mm256_setzero_si256();
        __m256i iacc_min_b = _mm256_setzero_si256();

        const __m256i q8sums =
          _mm256_loadu_si256((const __m256i *)(a_ptr[b].bsums));
        __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
          _mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
        q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

        // Processes two sub blocks from each Q4_K in each iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_K for two sub blocks quantized values
          // interleaved with each other in chunks of eight - B0,B1 ....B6,B7
          const __m256i rhs_raw_vec_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_vec_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_vec_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_vec_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_vec_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_vec_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_vec_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_vec_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // 4-bit -> 8-bit
          // Values of the first sub block of eight block_q4_K structures for
          // the sb loop
          const __m256i rhs_vec_0123_00 =
            _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
          const __m256i rhs_vec_4567_00 =
            _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
          const __m256i rhs_vec_0123_01 =
            _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
          const __m256i rhs_vec_4567_01 =
            _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
          const __m256i rhs_vec_0123_02 =
            _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
          const __m256i rhs_vec_4567_02 =
            _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
          const __m256i rhs_vec_0123_03 =
            _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
          const __m256i rhs_vec_4567_03 =
            _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

          // Values of the second sub block of eight block_q4_K structures when
          // sb = 1
          const __m256i rhs_vec_0123_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
          const __m256i rhs_vec_4567_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
          const __m256i rhs_vec_0123_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
          const __m256i rhs_vec_4567_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
          const __m256i rhs_vec_0123_12 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
          const __m256i rhs_vec_4567_12 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
          const __m256i rhs_vec_0123_13 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
          const __m256i rhs_vec_4567_13 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q8_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          __m128i scales_rearrange_0 =
            _mm_shuffle_epi8(mins_and_scales_0, scalemask);
          __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

          // Scales of second sub block in the sb loop
          __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          __m128i scales_rearrange_1 =
            _mm_shuffle_epi8(mins_and_scales_1, scalemask);
          __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          // Load the two sub block values corresponding to sb in block_q8_K in
          // batches of 16 bytes and replicate the same across 256 bit vector
          __m256i lhs_vec_00 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
          __m256i lhs_vec_01 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
          __m256i lhs_vec_10 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
          __m256i lhs_vec_11 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

          lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
          lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
          lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
          lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

          // Dot product done within 32 bit lanes and accumulated in the same
          // vector First done for first sub block and thenn for second sub
          // block in each sb B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3)
          // B3(0-3) B7(0-3) with A0(0-3) B0(4-7) B4(4-7) B1(4-7) B5(4-7)
          // B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
          // ...........................................................................
          // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31)
          // B3(28-31) B7(28-31) with A0(28-31)

          __m256i iacc_0 = _mm256_setzero_si256();
          __m256i iacc_1 = _mm256_setzero_si256();

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_00,
                        _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_00, 0)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177),
                                 rhs_vec_4567_00, 170),
              _mm256_shuffle_epi32(lhs_vec_00, 85)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_01,
                        _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_00, 170)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177),
                                 rhs_vec_4567_01, 170),
              _mm256_shuffle_epi32(lhs_vec_00, 255)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_02,
                        _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_01, 0)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177),
                                 rhs_vec_4567_02, 170),
              _mm256_shuffle_epi32(lhs_vec_01, 85)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_03,
                        _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_01, 170)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177),
                                 rhs_vec_4567_03, 170),
              _mm256_shuffle_epi32(lhs_vec_01, 255)));

          iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_10,
                        _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_10, 0)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177),
                                 rhs_vec_4567_10, 170),
              _mm256_shuffle_epi32(lhs_vec_10, 85)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_11,
                        _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_10, 170)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177),
                                 rhs_vec_4567_11, 170),
              _mm256_shuffle_epi32(lhs_vec_10, 255)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_12,
                        _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_11, 0)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177),
                                 rhs_vec_4567_12, 170),
              _mm256_shuffle_epi32(lhs_vec_11, 85)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_13,
                        _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_11, 170)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177),
                                 rhs_vec_4567_13, 170),
              _mm256_shuffle_epi32(lhs_vec_11, 255)));

          iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

          // Accumulate the iacc value for one sb
          __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

          // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K
          // across the vector Multiply-Add with corresponding mins of Q4_Kx8
          // with bsums
          __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
          __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
          q8s = _mm256_bsrli_epi128(q8s, 4);

          // Accumulate for the complete block
          iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
          iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }

        // Multiply-Add with scale values for the complete super block
        acc_row =
          _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b),
                          _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
        acc_min_rows = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(iacc_min_b),
          _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
      }

      // Accumulated output values permuted so as to be stored in appropriate
      // order post accumulation
      acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
      _mm256_storeu_ps(s + (y * nr + x * 8),
                       _mm256_sub_ps(acc_row, acc_min_rows));
    }
  }

#else

  float sumf[8];
  float sum_minf[8];
  uint32_t utmp[32];
  int sumi1;
  int sumi2;
  int sumi;

  const block_q8_K *a_ptr = (const block_q8_K *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_Kx8 *b_ptr = (const block_q4_Kx8 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++) {
      sumf[j] = 0.0;
      sum_minf[j] = 0.0;
    }
    for (int l = 0; l < nb; l++) {
      for (int sb = 0; sb < 8; sb++) {
        memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
        utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) |
                           (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
        const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
        utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) |
                           (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
        utmp[sb * 4 + 2] = uaux_0;
        utmp[sb * 4 + 0] &= kmask1;
      }
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        uint8_t *scales_0 = (uint8_t *)utmp + (k / 4) * 32;
        uint8_t *scales_1 = (uint8_t *)utmp + (k / 4) * 32 + 16;
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi1 = 0;
          sumi2 = 0;
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] &
                       0xF);
            const int v1 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] >>
                       4);
            sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
            sumi2 =
              (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
            sumi1 = sumi1 * scales_0[j];
            sumi2 = sumi2 * scales_1[j];
            sumi += sumi1 + sumi2;
          }
          sumf[j] += sumi * nntr_fp16_to_fp32(b_ptr[l].d[j]) * a_ptr[l].d;
        }
      }
      for (int sb = 0; sb < 8; sb++) {
        uint8_t *mins = (uint8_t *)utmp + 8 + sb * 16;
        for (int j = 0; j < ncols_interleaved; j++) {
          sum_minf[j] += mins[j] *
                         (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                         nntr_fp16_to_fp32(b_ptr[l].dmin[j]) * a_ptr[l].d;
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++) {
      s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
    }
  }
#endif
}

static void ggml_gemm_q4_K_8x8_q8_K(int n, float *GGML_RESTRICT s, size_t bs,
                                    const void *GGML_RESTRICT vx,
                                    const void *GGML_RESTRICT vy, int nr,
                                    int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  UNUSED(s);
  UNUSED(bs);
  UNUSED(vx);
  UNUSED(vy);
  UNUSED(nr);
  UNUSED(nc);
  UNUSED(nb);
  UNUSED(ncols_interleaved);
  UNUSED(blocklen);

#if defined(__AVX2__)
  const block_q4_Kx8 *b_ptr_start = (const block_q4_Kx8 *)vx;
  const block_q8_Kx4 *a_ptr_start = (const block_q8_Kx4 *)vy;
  int64_t b_nb = n / QK_K;
  int64_t y = 0;

  // Mask to mask out nibbles from packed bytes
  const __m256i m4b = _mm256_set1_epi8(0x0F);
  // Permute mask used for easier vector processing at later stages
  __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);

  int anr = nr - nr % 16;
  ; // Used to align nr with boundary of 16
  // Take group of four block_q8_Kx4 structures at each pass of the loop and
  // perform dot product operation
  for (; y < anr / 4; y += 4) {

    const block_q8_Kx4 *a_ptrs[4];

    a_ptrs[0] = a_ptr_start + (y * nb);
    for (int i = 0; i < 3; ++i) {
      a_ptrs[i + 1] = a_ptrs[i] + nb;
    }

    // Take group of eight block_q4_kx8 structures at each pass of the loop and
    // perform dot product operation
    for (int64_t x = 0; x < nc / 8; x++) {

      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_rows[i] = _mm256_setzero_ps();
      }

      __m256 acc_min_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_min_rows[i] = _mm256_setzero_ps();
      }

      // For super block
      for (int64_t b = 0; b < nb; b++) {

        // Scale values - Load the eight scale values of block_q4_kx8
        const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

        // dmin values - Load the eight dmin values of block_q4_kx8
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_K for two sub blocks quantized values
          // interleaved with each other in chunks of eight bytes - B0,B1
          // ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_00 = _mm256_and_si256(
            rhs_raw_mat_0145_0, m4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i rhs_mat_2367_00 = _mm256_and_si256(
            rhs_raw_mat_2367_0, m4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i rhs_mat_0145_01 = _mm256_and_si256(
            rhs_raw_mat_0145_1, m4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i rhs_mat_2367_01 = _mm256_and_si256(
            rhs_raw_mat_2367_1, m4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i rhs_mat_0145_02 = _mm256_and_si256(
            rhs_raw_mat_0145_2,
            m4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i rhs_mat_2367_02 = _mm256_and_si256(
            rhs_raw_mat_2367_2,
            m4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i rhs_mat_0145_03 = _mm256_and_si256(
            rhs_raw_mat_0145_3,
            m4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i rhs_mat_2367_03 = _mm256_and_si256(
            rhs_raw_mat_2367_3,
            m4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i rhs_mat_2367_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i rhs_mat_0145_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i rhs_mat_2367_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i rhs_mat_0145_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_2, 4),
            m4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i rhs_mat_2367_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_2, 4),
            m4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i rhs_mat_0145_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_3, 4),
            m4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i rhs_mat_2367_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_3, 4),
            m4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3)
                                   // B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3)
                                   // B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_01, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                   // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_01, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                   // B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19)
                  // B05(16-19) B04(16-19) B05(16-19)
          const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19)
                  // B07(16-19) B06(16-19) B07(16-19)

          const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27)
                  // B05(24-27) B04(24-27) B05(24-27)
          const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27)
                  // B07(24-27) B06(24-27) B07(24-27)

          const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3)
                                   // B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3)
                                   // B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_11, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                   // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_11, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                   // B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19)
                  // B15(16-19) B14(16-19) B15(16-19)
          const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19)
                  // B17(16-19) B16(16-19) B17(16-19)

          const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27)
                  // B15(24-27) B14(24-27) B15(24-27)
          const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27)
                  // B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7)
                                   // B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7)
                                   // B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_01,
            221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15)
                  // B05(12-15) B04(12-15) B05(12-15)
          const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_01,
            221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15)
                  // B07(12-15) B06(12-15) B07(12-15)

          const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23)
                  // B05(20-23) B04(20-23) B05(20-23)
          const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23)
                  // B07(20-23) B06(20-23) B07(20-23)

          const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31)
                  // B05(28-31) B04(28-31) B05(28-31)
          const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31)
                  // B07(28-31) B06(28-31) B07(28-31)

          const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7)
                                   // B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7)
                                   // B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_11,
            221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15)
                  // B15(12-15) B14(12-15) B15(12-15)
          const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_11,
            221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15)
                  // B17(12-15) B16(12-15) B17(12-15)

          const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23)
                  // B15(20-23) B14(20-23) B15(20-23)
          const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23)
                  // B17(20-23) B16(20-23) B17(20-23)

          const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31)
                  // B15(28-31) B14(28-31) B15(28-31)
          const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31)
                  // B17(28-31) B16(28-31) B17(28-31)

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          const __m256i scales_0 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          const __m256i scales_1 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
          const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

          const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
          const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

          for (int rp = 0; rp < 4; rp++) {

            // Load the four block_q8_k quantized values interleaved with each
            // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
            // bit vectors and repeated into a 256 bit vector
            __m256i lhs_mat_0123_00 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 256 * sb)));
            __m256i lhs_mat_01_00 =
              _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
            __m256i lhs_mat_23_00 =
              _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
            __m256i lhs_mat_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 32 + 256 * sb)));
            __m256i lhs_mat_01_01 =
              _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
            __m256i lhs_mat_23_01 =
              _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
            __m256i lhs_mat_0123_02 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 64 + 256 * sb)));
            __m256i lhs_mat_01_02 =
              _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
            __m256i lhs_mat_23_02 =
              _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
            __m256i lhs_mat_0123_03 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 96 + 256 * sb)));
            __m256i lhs_mat_01_03 =
              _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
            __m256i lhs_mat_23_03 =
              _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
            __m256i lhs_mat_0123_10 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 128 + 256 * sb)));
            __m256i lhs_mat_01_10 =
              _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
            __m256i lhs_mat_23_10 =
              _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
            __m256i lhs_mat_0123_11 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 160 + 256 * sb)));
            __m256i lhs_mat_01_11 =
              _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
            __m256i lhs_mat_23_11 =
              _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
            __m256i lhs_mat_0123_12 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 192 + 256 * sb)));
            __m256i lhs_mat_01_12 =
              _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
            __m256i lhs_mat_23_12 =
              _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
            __m256i lhs_mat_0123_13 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 224 + 256 * sb)));
            __m256i lhs_mat_01_13 =
              _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
            __m256i lhs_mat_23_13 =
              _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

            // Bsums are loaded - four bsums are loaded (for two sub blocks) for
            // the different Q8_K blocks
            __m256i lhs_bsums_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].bsums + 16 * sb)));
            __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(
              _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                             _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
            lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(
              lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

            // Shuffle pattern one - left side input
            const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_00, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                                   // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_00, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3)
                                   // A02(0-3) A03(0-3) A02(0-3) A03(0-3)

            const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_01, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_01, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11)
                                   // A02(8-11) A03(8-11) A02(8-11) A03(8-11)

            const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_02,
              160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19)
                    // A00(16-19) A01(16-19) A01(16-19)
            const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_02,
              160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19)
                    // A03(16-19) A02(16-19) A03(16-19)

            const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_03,
              160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27)
                    // A00(24-27) A01(24-27) A01(24-27)
            const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_03,
              160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27)
                    // A03(24-27) A02(24-27) A03(24-27)

            const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_10, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                                   // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_10, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3)
                                   // A12(0-3) A13(0-3) A12(0-3) A13(0-3)

            const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_11, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_11, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11)
                                   // A12(8-11) A13(8-11) A12(8-11) A13(8-11)

            const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_12,
              160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19)
                    // A10(16-19) A11(16-19) A11(16-19)
            const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_12,
              160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19)
                    // A13(16-19) A12(16-19) A13(16-19)

            const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_13,
              160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27)
                    // A10(24-27) A11(24-27) A11(24-27)
            const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_13,
              160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27)
                    // A13(24-27) A12(24-27) A13(24-27)

            // Shuffle pattern two- left side input
            const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_00, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                                   // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_00, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7)
                                   // A02(4-7) A03(4-7) A02(4-7) A03(4-7)

            const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_01,
              245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15)
                    // A00(12-15) A01(12-15) A01(12-15)
            const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_01,
              245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15)
                    // A03(12-15) A02(12-15) A03(12-15)

            const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_02,
              245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23)
                    // A00(20-23) A01(20-23) A01(20-23)
            const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_02,
              245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23)
                    // A03(20-23) A02(20-23) A03(20-23)

            const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_03,
              245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31)
                    // A00(28-31) A01(28-31) A01(28-31)
            const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_03,
              245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31)
                    // A03(28-31) A02(28-31) A03(28-31)

            const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_10, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                                   // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_10, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7)
                                   // A12(4-7) A13(4-7) A12(4-7) A13(4-7)

            const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_11,
              245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15)
                    // A10(12-15) A11(12-15) A11(12-15)
            const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_11,
              245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15)
                    // A13(12-15) A12(12-15) A13(12-15)

            const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_12,
              245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23)
                    // A10(20-23) A11(20-23) A11(20-23)
            const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_12,
              245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23)
                    // A13(20-23) A12(20-23) A13(20-23)

            const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_13,
              245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31)
                    // A10(28-31) A11(28-31) A11(28-31)
            const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_13,
              245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31)
                    // A13(28-31) A12(28-31) A13(28-31)

            // The values arranged in shuffle patterns are operated with dot
            // product operation within 32 bit lane i.e corresponding bytes and
            // multiplied and added into 32 bit integers within 32 bit lane
            __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
            __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
            __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
            __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
            __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
            __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
            __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
            __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

            __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
            __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
            __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
            __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
            __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
            __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
            __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
            __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            __m256i iacc_mat_00_0 =
              _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
            __m256i iacc_mat_01_0 =
              _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
            __m256i iacc_mat_10_0 =
              _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
            __m256i iacc_mat_11_0 =
              _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

            __m256i iacc_mat_00_1 =
              _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
            __m256i iacc_mat_01_1 =
              _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
            __m256i iacc_mat_10_1 =
              _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
            __m256i iacc_mat_11_1 =
              _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
            iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
            iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
            iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

            iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
            iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
            iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
            iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

            // Straighten out to make 4 row vectors (4 for each sub block which
            // are accumulated together in the next step)
            __m256i iacc_row_0_0 = _mm256_blend_epi32(
              iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
            __m256i iacc_row_1_0 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
            __m256i iacc_row_2_0 = _mm256_blend_epi32(
              iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
            __m256i iacc_row_3_0 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
            __m256i iacc_row_0_1 = _mm256_blend_epi32(
              iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
            __m256i iacc_row_1_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
            __m256i iacc_row_2_1 = _mm256_blend_epi32(
              iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
            __m256i iacc_row_3_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

            __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
            __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
            __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
            __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
            // across lanes
            const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
            const __m256 row_scale_f32 = _mm256_set_m128(
              row_scale_f32_sse, row_scale_f32_sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d,
                                                     // loadMask);

            // Multiply with appropiate scales and accumulate (for both d and
            // dmin) below
            acc_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_0),
              _mm256_mul_ps(col_scale_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_rows[rp * 4]);
            acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_1),
              _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(
                                             row_scale_f32, row_scale_f32, 85)),
              acc_rows[rp * 4 + 1]);
            acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_2),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
              acc_rows[rp * 4 + 2]);
            acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_3),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
              acc_rows[rp * 4 + 3]);

            __m256i iacc_row_min_0 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
            __m256i iacc_row_min_1 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
            __m256i iacc_row_min_2 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
            __m256i iacc_row_min_3 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

            acc_min_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_0),
              _mm256_mul_ps(col_dmin_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_min_rows[rp * 4]);
            acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_1),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32,
                                                            row_scale_f32, 85)),
              acc_min_rows[rp * 4 + 1]);
            acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_2),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 170)),
              acc_min_rows[rp * 4 + 2]);
            acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_3),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 255)),
              acc_min_rows[rp * 4 + 3]);
          }
        }
      }
      for (int i = 0; i < 16; i++) {
        _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }
  for (; y < nr / 4; y++) {

    const block_q8_Kx4 *a_ptr = a_ptr_start + (y * nb);

    for (int64_t x = 0; x < nc / 8; x++) {

      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_rows[i] = _mm256_setzero_ps();
      }

      __m256 acc_min_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_min_rows[i] = _mm256_setzero_ps();
      }

      for (int64_t b = 0; b < nb; b++) {

        // Scale values - Load the eight scale values of block_q4_Kx8
        const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

        // dmin values - Load the eight dmin values of block_q4_Kx8
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_k for two sub blocks quantized values
          // interleaved with each other in chunks of eight bytes - B0,B1
          // ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_00 = _mm256_and_si256(
            rhs_raw_mat_0145_0, m4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i rhs_mat_2367_00 = _mm256_and_si256(
            rhs_raw_mat_2367_0, m4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i rhs_mat_0145_01 = _mm256_and_si256(
            rhs_raw_mat_0145_1, m4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i rhs_mat_2367_01 = _mm256_and_si256(
            rhs_raw_mat_2367_1, m4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i rhs_mat_0145_02 = _mm256_and_si256(
            rhs_raw_mat_0145_2,
            m4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i rhs_mat_2367_02 = _mm256_and_si256(
            rhs_raw_mat_2367_2,
            m4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i rhs_mat_0145_03 = _mm256_and_si256(
            rhs_raw_mat_0145_3,
            m4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i rhs_mat_2367_03 = _mm256_and_si256(
            rhs_raw_mat_2367_3,
            m4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i rhs_mat_2367_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i rhs_mat_0145_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i rhs_mat_2367_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i rhs_mat_0145_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_2, 4),
            m4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i rhs_mat_2367_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_2, 4),
            m4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i rhs_mat_0145_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_3, 4),
            m4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i rhs_mat_2367_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_3, 4),
            m4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3)
                                   // B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3)
                                   // B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_01, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                   // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_01, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                   // B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19)
                  // B05(16-19) B04(16-19) B05(16-19)
          const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19)
                  // B07(16-19) B06(16-19) B07(16-19)

          const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27)
                  // B05(24-27) B04(24-27) B05(24-27)
          const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27)
                  // B07(24-27) B06(24-27) B07(24-27)

          const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3)
                                   // B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3)
                                   // B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_11, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                   // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_11, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                   // B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19)
                  // B15(16-19) B14(16-19) B15(16-19)
          const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19)
                  // B17(16-19) B16(16-19) B17(16-19)

          const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27)
                  // B15(24-27) B14(24-27) B15(24-27)
          const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27)
                  // B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7)
                                   // B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7)
                                   // B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_01,
            221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15)
                  // B05(12-15) B04(12-15) B05(12-15)
          const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_01,
            221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15)
                  // B07(12-15) B06(12-15) B07(12-15)

          const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23)
                  // B05(20-23) B04(20-23) B05(20-23)
          const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23)
                  // B07(20-23) B06(20-23) B07(20-23)

          const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31)
                  // B05(28-31) B04(28-31) B05(28-31)
          const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31)
                  // B07(28-31) B06(28-31) B07(28-31)

          const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7)
                                   // B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7)
                                   // B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_11,
            221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15)
                  // B15(12-15) B14(12-15) B15(12-15)
          const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_11,
            221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15)
                  // B17(12-15) B16(12-15) B17(12-15)

          const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23)
                  // B15(20-23) B14(20-23) B15(20-23)
          const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23)
                  // B17(20-23) B16(20-23) B17(20-23)

          const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31)
                  // B15(28-31) B14(28-31) B15(28-31)
          const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31)
                  // B17(28-31) B16(28-31) B17(28-31)

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures when sb = 1
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          const __m256i scales_0 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          const __m256i scales_1 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
          const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

          const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
          const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

          // Load the four block_q8_k quantized values interleaved with each
          // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
          // bit vectors and repeated into a 256 bit vector
          __m256i lhs_mat_0123_00 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 256 * sb)));
          __m256i lhs_mat_01_00 =
            _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
          __m256i lhs_mat_23_00 =
            _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
          __m256i lhs_mat_0123_01 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 32 + 256 * sb)));
          __m256i lhs_mat_01_01 =
            _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
          __m256i lhs_mat_23_01 =
            _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
          __m256i lhs_mat_0123_02 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 64 + 256 * sb)));
          __m256i lhs_mat_01_02 =
            _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
          __m256i lhs_mat_23_02 =
            _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
          __m256i lhs_mat_0123_03 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 96 + 256 * sb)));
          __m256i lhs_mat_01_03 =
            _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
          __m256i lhs_mat_23_03 =
            _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
          __m256i lhs_mat_0123_10 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 128 + 256 * sb)));
          __m256i lhs_mat_01_10 =
            _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
          __m256i lhs_mat_23_10 =
            _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
          __m256i lhs_mat_0123_11 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 160 + 256 * sb)));
          __m256i lhs_mat_01_11 =
            _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
          __m256i lhs_mat_23_11 =
            _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
          __m256i lhs_mat_0123_12 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 192 + 256 * sb)));
          __m256i lhs_mat_01_12 =
            _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
          __m256i lhs_mat_23_12 =
            _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
          __m256i lhs_mat_0123_13 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 224 + 256 * sb)));
          __m256i lhs_mat_01_13 =
            _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
          __m256i lhs_mat_23_13 =
            _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

          // Bsums are loaded - four bsums are loaded (for two sub blocks) for
          // the different Q8_K blocks
          __m256i lhs_bsums_0123_01 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].bsums + 16 * sb)));
          __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(
            _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                           _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
          lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(
            lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

          // Shuffle pattern one - left side input
          const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_00, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                 // A00(0-3) A01(0-3) A01(0-3)
          const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_00, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3)
                                 // A03(0-3) A02(0-3) A03(0-3)

          const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_01, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                 // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_01, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11)
                                 // A02(8-11) A03(8-11) A02(8-11) A03(8-11)

          const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_02, 160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                 // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
          const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_02, 160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19)
                                 // A02(16-19) A03(16-19) A02(16-19) A03(16-19)

          const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_03, 160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                 // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
          const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_03, 160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27)
                                 // A02(24-27) A03(24-27) A02(24-27) A03(24-27)

          const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_10, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                 // A10(0-3) A11(0-3) A11(0-3)
          const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_10, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3)
                                 // A13(0-3) A12(0-3) A13(0-3)

          const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_11, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                 // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_11, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11)
                                 // A12(8-11) A13(8-11) A12(8-11) A13(8-11)

          const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_12, 160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                 // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
          const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_12, 160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19)
                                 // A12(16-19) A13(16-19) A12(16-19) A13(16-19)

          const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_13, 160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                 // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
          const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_13, 160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27)
                                 // A12(24-27) A13(24-27) A12(24-27) A13(24-27)

          // Shuffle pattern two- left side input
          const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_00, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                 // A00(4-7) A01(4-7) A01(4-7)
          const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_00, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7)
                                 // A03(4-7) A02(4-7) A03(4-7)

          const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_01, 245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                 // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_01, 245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15)
                                 // A02(12-15) A03(12-15) A02(12-15) A03(12-15)

          const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_02, 245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                 // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
          const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_02, 245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23)
                                 // A02(20-23) A03(20-23) A02(20-23) A03(20-23)

          const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_03, 245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                 // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
          const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_03, 245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31)
                                 // A02(28-31) A03(28-31) A02(28-31) A03(28-31)

          const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_10, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                 // A10(4-7) A11(4-7) A11(4-7)
          const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_10, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7)
                                 // A13(4-7) A12(4-7) A13(4-7)

          const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_11, 245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                 // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_11, 245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15)
                                 // A12(12-15) A13(12-15) A12(12-15) A13(12-15)

          const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_12, 245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                 // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
          const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_12, 245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23)
                                 // A12(20-23) A13(20-23) A12(20-23) A13(20-23)

          const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_13, 245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                 // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
          const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_13, 245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31)
                                 // A12(28-31) A13(28-31) A12(28-31) A13(28-31)

          // The values arranged in shuffle patterns are operated with dot
          // product operation within 32 bit lane i.e corresponding bytes and
          // multiplied and added into 32 bit integers within 32 bit lane
          __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
          __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
          __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
          __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
          __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
          __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
          __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
          __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

          __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
          __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
          __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
          __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
          __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
          __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
          __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
          __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          __m256i iacc_mat_00_0 =
            _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
          __m256i iacc_mat_01_0 =
            _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
          __m256i iacc_mat_10_0 =
            _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
          __m256i iacc_mat_11_0 =
            _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

          __m256i iacc_mat_00_1 =
            _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
          __m256i iacc_mat_01_1 =
            _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
          __m256i iacc_mat_10_1 =
            _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
          __m256i iacc_mat_11_1 =
            _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
          iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
          iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
          iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

          iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
          iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
          iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
          iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

          // Straighten out to make 4 row vectors (4 for each sub block which
          // are accumulated together in the next step)
          __m256i iacc_row_0_0 = _mm256_blend_epi32(
            iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
          __m256i iacc_row_1_0 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
          __m256i iacc_row_2_0 = _mm256_blend_epi32(
            iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
          __m256i iacc_row_3_0 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
          __m256i iacc_row_0_1 = _mm256_blend_epi32(
            iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
          __m256i iacc_row_1_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
          __m256i iacc_row_2_1 = _mm256_blend_epi32(
            iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
          __m256i iacc_row_3_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

          __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
          __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
          __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
          __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
          // across lanes
          const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
          const __m256 row_scale_f32 =
            _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d,
                                                                   // loadMask);

          // Multiply with appropiate scales and accumulate (for both d and
          // dmin) below
          acc_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_0),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_rows[0]);
          acc_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_1),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_rows[1]);
          acc_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_2),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_rows[2]);
          acc_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_3),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_rows[3]);

          __m256i iacc_row_min_0 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
          __m256i iacc_row_min_1 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
          __m256i iacc_row_min_2 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
          __m256i iacc_row_min_3 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

          acc_min_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_0),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_min_rows[0]);
          acc_min_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_1),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_min_rows[1]);
          acc_min_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_2),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_min_rows[2]);
          acc_min_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_3),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_min_rows[3]);
        }
      }

      // Store the accumulated values
      for (int i = 0; i < 4; i++) {
        _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }

#else

  float sumf[4][8];
  float sum_minf[4][8];
  uint32_t utmp[32];
  int sumi1;
  int sumi2;
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_Kx4 *a_ptr = (const block_q8_Kx4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_Kx8 *b_ptr = (const block_q4_Kx8 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumf[m][j] = 0.0;
          sum_minf[m][j] = 0.0;
        }
      }
      for (int l = 0; l < nb; l++) {
        for (int sb = 0; sb < 8; sb++) {
          memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
          utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) |
                             (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
          utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) |
                             (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
          utmp[sb * 4 + 2] = uaux_0;
          utmp[sb * 4 + 0] &= kmask1;
        }
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          uint8_t *scales_0 = (uint8_t *)utmp + (k / 4) * 32;
          uint8_t *scales_1 = (uint8_t *)utmp + (k / 4) * 32 + 16;
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi1 = 0;
              sumi2 = 0;
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] &
                           0xF);
                const int v1 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] >>
                           4);
                sumi1 =
                  (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen +
                                    m * blocklen + i]);
                sumi2 =
                  (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen +
                                    m * blocklen + i + 128]);
                sumi1 = sumi1 * scales_0[j];
                sumi2 = sumi2 * scales_1[j];
                sumi += sumi1 + sumi2;
              }
              sumf[m][j] +=
                sumi * nntr_fp16_to_fp32(b_ptr[l].d[j]) * a_ptr[l].d[m];
            }
          }
        }
        for (int sb = 0; sb < 8; sb++) {
          uint8_t *mins = (uint8_t *)utmp + 8 + sb * 16;
          for (int m = 0; m < 4; m++) {
            const int16_t *bsums =
              a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
            for (int j = 0; j < ncols_interleaved; j++) {
              sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                nntr_fp16_to_fp32(b_ptr[l].dmin[j]) *
                                a_ptr[l].d[m];
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] =
            sumf[m][j] - sum_minf[m][j];
        }
      }
    }
  }
#endif
}
/*
GEMM GEMV KERNEL
 */

// /*
// GEMM PREPROCESSING : WEIGHT BLOCKING FUNCTION
//  */
static block_q4_Kx8 make_block_q4_Kx8(block_q4_K *in,
                                      unsigned int blck_size_interleave) {
  // static block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, unsigned int
  // blck_size_interleave) {
  block_q4_Kx8 out;
  // Delta(scale) and dmin values of the eight Q4_K structures are copied onto
  // the output interleaved structure
  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
  }

  for (int i = 0; i < 8; i++) {
    out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
  }

  const int end = QK_K * 4 / blck_size_interleave;

  // Interleave Q4_K quants by taking 8 bytes at a time
  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  // The below logic is designed so as to unpack and rearrange scales and mins
  // values in Q4_K Currently the Q4_K structure has 8 scales and 8 mins packed
  // in 12 bytes ( 6 bits for each value) The output Q4_Kx8 structure has 96
  // bytes Every 12 byte is packed such that it contains scales and mins for
  // corresponding sub blocks from Q4_K structure For eg - First 12 bytes
  // contains 8 scales and 8 mins - each of first sub block from different Q4_K
  // structures
  uint8_t s[8], m[8];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = in[j].scales[i] & 63;
      m[j] = in[j].scales[i + 4] & 63;
    }

    out.scales[i * 12] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
      m[j] =
        ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
    }

    out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  return out;
}

static int repack_q4_K_to_q4_K_8_bl(void *t, int interleave_block,
                                    const void *GGML_RESTRICT data,
                                    size_t data_size, int64_t M, int64_t N) {
  assert(interleave_block == 8);
  constexpr int nrows_interleaved = 8;

  block_q4_Kx8 *dst = (block_q4_Kx8 *)t;
  const block_q4_K *src = (const block_q4_K *)data;
  block_q4_K dst_tmp[8];
  /// @todo ggml_nrow raw implemenation
  // int nrow = ggml_nrows(t);
  int nrow = M * 1 * 1;
  // int nblocks = t->ne[0] / QK_K;
  ///@todo check if this is valid way
  int nblocks = (N + QK_K - 1) / QK_K;
  // int nblocks = N / QK_K;

  assert(data_size == nrow * nblocks * sizeof(block_q4_K));

  if (M % nrows_interleaved != 0 || N % 8 != 0) {
    return -1;
  }

  for (int b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (int i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst++ = make_block_q4_Kx8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;

  GGML_UNUSED(data_size);
}

// // repack
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int _repack(void *, const void *, size_t, int64_t, int64_t);

template <>
int _repack<block_q4_K, 8, 8>(void *t, const void *data, size_t data_size,
                              int64_t M, int64_t N) {
  return repack_q4_K_to_q4_K_8_bl(t, 8, data, data_size, M, N);
}
// /*
// GEMM PREPROCESSING : BLOCKING FUNCTION
//  */

/*
GEMM/GEMV KERNEL FUNCTION INTERFACE
 */
// gemv
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS,
          ggml_type PARAM_TYPE>
void gemv(int, float *, size_t, const void *, const void *, int, int);

template <>
void gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float *s, size_t bs,
                                            const void *vx, const void *vy,
                                            int nr, int nc) {
  ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

// gemm
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS,
          ggml_type PARAM_TYPE>
void gemm(int, float *, size_t, const void *, const void *, int, int);

template <>
void gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float *s, size_t bs,
                                            const void *vx, const void *vy,
                                            int nr, int nc) {
  ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}
/*
GEMM/GEMV KERNEL FUNCTION INTERFACE
 */
void print_q8_k_block(void* block){
    block_q8_K* b = (block_q8_K*) block;
    printf("d : %f\n", b->d);
    printf("qs 0-3 : ");
    for (int i = 0; i < 4; i++) {
        printf("%d ", b->qs[i]);
    }
    printf("\n");
    printf("bsums 0-3 : ");
    for (int i = 0; i < 4; i++) {
        printf("%d ", b->bsums[i]);
    }
    printf("\n");
}

void print_q8_kx4_block_1(void* block, int64_t processed){
  if (processed == 0) return print_q8_k_block(block);
  
    block_q8_Kx4* b = (block_q8_Kx4*) block;
    printf("d : %f\n", b->d[0]);
    printf("qs 0-3 : ");
    for (int i = 0; i < 4; i++) {
        printf("%d ", b->qs[i]);
    }
    printf("\n");
    printf("bsums 0-3 : ");
    for (int i = 0; i < 4; i++) {
        printf("%d ", b->bsums[i]);
    }
    printf("\n");
}
void print_5_floats(float* src){
    for (int i = 0; i < 5; ++i){
        printf("%f ", src[i]);
    }
    printf("\n");
}
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS,
          ggml_type PARAM_TYPE>
class nntr_gemm_ggml_traits {
public:
  bool compute_forward(const unsigned int M, const unsigned int N,
                       const unsigned int K, const float *A,
                       const unsigned int lda, const void *B,
                       const unsigned int ldb, float *C,
                       const unsigned int ldc) {
    // bool compute_forward(struct ggml_compute_params * params, struct
    // ggml_tensor * op) {
    /// @todo Add mul_mat_id opeartion
    forward_mul_mat(M, N, K, A, lda, B, ldb, C, ldc);
    return true;
  }

  void forward_mul_mat(const unsigned int M, const unsigned int N,
                       const unsigned int K, const float *A,
                       const unsigned int lda, const void *B,
                       const unsigned int ldb, float *C,
                       const unsigned int ldc) {
    /// @todo raw implementation of GGML_TENSOR_BINRAY_OP_LOCALS
    // GGML_TENSOR_BINARY_OP_LOCALS
    /*
        src0 : q4_K Weight (N,K)
        src1 : float Activation (M,K)
        dst : float Output (M,N)

        int64_t ne0 : output row length
        int64_t ne1 : output column length
        int64_t ne2 : 1
        int64_t ne3 : 1
        int64_t ne01 = ne0
        int64_t ne11 = ne1
        int64_t ne12 = ne2
        int64_t ne13 = ne3

        size_t nb0 : size of float
        size_t nb1 : nb0 * ne0
        size_t nb2 : nb1 * ne1
        size_t nb3 : nb2 * ne2

        size_t nb01 : src0->nb[1]
        size_t nb11 : src1->nb[1]
        size_t nbw1 : solved
     */

    /// @todo Enable multithreading
    // const int ith = params->ith;
    // const int nth = params->nth;
    const int n_threads = 1; // DO NOT FIX!
    const int ith = 0;
    const int nth = 1;

    int64_t ne0 = N;
    int64_t ne1 = M;
    int64_t ne2 = 1;
    int64_t ne3 = 1;

    int64_t ne01 = ne0;
    int64_t ne11 = ne1;
    int64_t ne12 = ne2;
    int64_t ne13 = ne3;

    ///@todo Check if this is correct
    int64_t ne10 = K; // ne0
    int64_t ne00 = K; // ne1

    assert(ne0 == ne01);
    assert(ne1 == ne11);
    assert(ne2 == ne12);
    assert(ne3 == ne13);

    size_t nb0 = sizeof(float); // sizeof(typeof(A))
    size_t nb1 = nb0 * ne0;
    size_t nb2 = nb1 * ne1;
    size_t nb3 = nb2 * ne2;

    int64_t weight_block_size =sizeof(block_q4_K);

    ///@todo Check if this is correct
    size_t nb11 = nb1;
    size_t nb01 = weight_block_size;

    // dst cannot be transposed or permuted
    assert(nb0 == sizeof(float));
    assert(nb0 <= nb1);
    assert(nb1 <= nb2);
    assert(nb2 <= nb3);

    // assert(src1->type == GGML_TYPE_F32);

    // GGML_ASSERT(ggml_n_dims(op->src[0]) == 2);
    // GGML_ASSERT(ggml_n_dims(op->src[1]) == 2);

    // Allocate wdata : note that there is no re-allocation per Op during
    // original llama.cpp. wdata is repeatedly reused. This should be optimized
    // afterwards.

    /*MY DEFINITION*/
    int64_t nb00, nb10;
    ne00 = K, ne01 = N; // weight block params
    ne10 = K, ne11 = M, ne12 = 1, ne13 = 1; // activation block params
    ne0 = N, ne1 = M, ne2 = 1, ne3 = 1; // output block params
    
    nb00 = weight_block_size; // ggml_type_size(type);
    nb01 = nb00 * (ne00 / /*QK_K*/ 256 );
    
    nb10 = sizeof(float);
    nb11 = nb10 * (ne10 / 1);

    nb0 = sizeof(float);
    nb1 = nb0 * (ne0 / 1);
    nb2 = nb1 * ne1;
    /*MY DEFINITION ENDS*/
    size_t work_size = 0;

    size_t cur = 0;
    ///@todo Automatically choose online quantization type based on Weight
    ///quantization type : refer to type_traits_cpu[YOUR_TYPE].vec_dot_type
    const enum ggml_type vec_dot_type = GGML_TYPE_Q8_K;

    // For Q4_K quantization Weight, it is essential for activation to be
    // quantized online.
    if (GGML_TYPE_F32 != vec_dot_type) {
      ///@todo Generalize ggml_row_size
      // Refer to ggml-cpu.c
      // cur = ggml_row_size(vec_dot_type, (ne10 * ne11 * ne12 * ne13));
      // QK_K, sizeof(block_q8_K) = blocksize, typesize
      size_t nnee = (ne10 * ne11 * ne12 * ne13);
      cur =  (sizeof(block_q8_K)  * nnee) / QK_K;
    }
    work_size = MAX(work_size, cur);
    if (work_size > 0) {
      work_size += CACHE_LINE_SIZE * (n_threads);
    }
    // char *       wdata = static_cast<char *>(params->wdata);
    char *wdata = (new char [work_size]);

    ///@todo Generalize ggml_row_size
    // const size_t nbw1 = ggml_row_size(PARAM_TYPE, ne10);
    const size_t nbw1 = (sizeof(block_q8_K) * ne10) / QK_K;

    assert(work_size >= nbw1 * ne11);

    /// @todo Generailize to get type traits considering template parameter
    /// PARAM_TYPE
    // const ggml_from_float_t from_float =
    // ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

    print_5_floats((float *)A);
    // auto t1 = high_resolution_clock::now();
    int64_t i11_processed = 0;
    for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
      ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(
        (float *)((char *)A + i11 * nb11), (void *)(wdata + i11 * nbw1), 4,
        ne10);
    }
    // auto t2 = high_resolution_clock::now();
    // auto dt = duration_cast<nanoseconds>(t2 - t1);
    // std::cout << "ggml_quantize_mat_t : " << dt.count()
    //         << " ns " << std::endl;

    i11_processed = ne11 - ne11 % 4;
    // t1 = high_resolution_clock::now();
    for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
      ///@todo Generailize to get type traits considering template parameter
      ///PARAM_TYPE
      // from_float((float *) ((char *) src1->data + i11 * nb11), (void *)
      // (wdata + i11 * nbw1), ne10);
      quantize_row_q8_K((float *)((char *)A + i11 * nb11),
                        (void *)(wdata + i11 * nbw1), ne10);
    }
    // t2 = high_resolution_clock::now();
    // dt = duration_cast<nanoseconds>(t2 - t1);
    // std::cout << "quantize_row_q8_K : " << dt.count()
    //         << " ns " << std::endl;

    print_q8_kx4_block_1(wdata, i11_processed);

    /// @todo Enable multithreading
    // ggml_barrier(params->threadpool);

    const void *src1_wdata = (void *)wdata;
    ///@todo Generalize ggml_row_size
    const size_t src1_col_stride = (sizeof(block_q8_K) * ne10) / QK_K;
    int64_t src0_start = (ith * ne01) / nth; // = 0
    int64_t src0_end = ((ith + 1) * ne01) / nth; // ne01 = N
    src0_start = (src0_start % NB_COLS)
                   ? src0_start + NB_COLS - (src0_start % NB_COLS)
                   : src0_start;
    src0_end = (src0_end % NB_COLS) ? src0_end + NB_COLS - (src0_end % NB_COLS)
                                    : src0_end;
    if (src0_start >= src0_end) {
      return;
    }

    // If there are more than three rows in src1, use gemm; otherwise, use gemv.
    // t1 = high_resolution_clock::now();
    if (ne11 > 3) {
      gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
        ne00, (float *)((char *)C) + src0_start, ne01,
        (const char *)B + src0_start * nb01, (const char *)src1_wdata,
        ne11 - ne11 % 4, src0_end - src0_start);
    }
    for (int iter = ne11 - ne11 % 4; iter < ne11; iter++) {
      gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
        ne00, (float *)((char *)C + (iter * nb1)) + src0_start, ne01,
        (const char *)B + src0_start * nb01,
        (const char *)src1_wdata + (src1_col_stride * iter), 1,
        src0_end - src0_start);
    }
    // t2 = high_resolution_clock::now();
    // dt = duration_cast<nanoseconds>(t2 - t1);
    // std::cout << "compute kernel : " << dt.count()
    //         << " ns " << std::endl;

    delete[] wdata;
        printf("forward_mul_mat INFORMATION\n");
        printf("ne00: %ld, ne01: %ld, ne02: %d, ne03: %d\n", ne00, ne01, 1, 1);
        printf("nb00: %ld, nb01: %ld\n", nb00, nb01);
        // printf("nb00: %ld, nb01: %ld, nb02: %ld, nb03: %ld\n", nb00, nb01, nb02, nb03);

        printf("ne10: %ld, ne11: %ld, ne12: %ld, ne13: %ld\n", ne10, ne11, ne12, ne13);
        printf("nb10: %ld, nb11: %ld\n", nb10, nb11);
        // printf("nb10: %ld, nb11: %ld, nb12: %ld, nb13: %ld\n", nb10, nb11, nb12, nb13);
        
        printf("ne0: %ld, ne1: %ld, ne2: %ld, ne3: %ld\n", ne0, ne1, ne2, ne3);
        printf("nb0: %ld, nb1: %ld, nb2: %ld, nb3: %ld\n", nb0, nb1, nb2, nb3);

        printf("ith: %d, nth: %d\n", ith, nth);
        printf("nbw1: %ld, nb11: %ld\n", nbw1, nb11);
  }

  ///@note repack is not called during GEMM runtime. It should be called weight
  ///is being loaded.
  int repack(void *t, const void *data, size_t data_size, int64_t M,
             int64_t N) {
    return _repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size, M, N);
  }
};

// instance for Q4
static nntr_gemm_ggml_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

static nntr_gemm_ggml_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> *
ggml_get_optimal_repack_type(unsigned int row_size,
                             ggml_type wType = GGML_TYPE_Q4_K,
                             ggml_type aQType = GGML_TYPE_Q8_K) {
  /// @todo Add more optimal repack types and check arch-dep once more
  if (row_size % 8 == 0) {
    return &q4_K_8x8_q8_K;
  }
  return nullptr;
}

void ggml_q4_K_8x8_q8_K_GEMM(const unsigned int M, const unsigned int N,
                             const unsigned int K, const float *A,
                             const unsigned int lda, const void *B,
                             const unsigned int ldb, float *C,
                             const unsigned int ldc) {
  auto gemm_fn = ggml_get_optimal_repack_type(N);
  if (gemm_fn) {
    gemm_fn->compute_forward(M, N, K, A, lda, B, ldb, C, ldc);
  }
}

void ggml_repack_q4_K_to_q8_K(void *W, void *repacked_W, size_t data_size,
                              const unsigned int M, const unsigned int N) {
  auto gemm_fn = ggml_get_optimal_repack_type(N);
  if (gemm_fn) {
    gemm_fn->repack(W, repacked_W, data_size, M, N);
  }
}

static float make_qkx3_quants(int n, int nmax, const float *x,
                              const float *weights, uint8_t *L, float *the_min,
                              uint8_t *Laux, float rmin, float rdelta,
                              int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights ? weights[0] : x[0] * x[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights ? weights[i] : x[i] * x[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) {
    min = 0;
  }
  if (max <= min) {
    memset(L, 0, n);
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights ? weights[i] : x[i] * x[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      Laux[i] = l;
      float w = weights ? weights[i] : x[i] * x[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static float make_qkx2_quants(int n, int nmax, const float *x,
                              const float *weights, uint8_t *L, float *the_min,
                              uint8_t *Laux, float rmin, float rdelta,
                              int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0)
    min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i)
      L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      Laux[i] = l;
      float w = weights[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                    uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

static float make_qp_quants(int n, int nmax, const float *x, uint8_t *L,
                            const float *quant_weights) {
  float max = 0;
  for (int i = 0; i < n; ++i) {
    max = MAX(max, x[i]);
  }
  if (!max) { // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = nmax / max;
  for (int i = 0; i < n; ++i) {
    L[i] = nearest_int(iscale * x[i]);
  }
  float scale = 1 / iscale;
  float best_mse = 0;
  for (int i = 0; i < n; ++i) {
    float diff = x[i] - scale * L[i];
    float w = quant_weights[i];
    best_mse += w * diff * diff;
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0)
      continue;
    float iscale_is = (0.1f * is + nmax) / max;
    float scale_is = 1 / iscale_is;
    float mse = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale_is * x[i]);
      l = MIN(nmax, l);
      float diff = x[i] - scale_is * l;
      float w = quant_weights[i];
      mse += w * diff * diff;
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MIN(nmax, l);
    L[i] = l;
    float w = quant_weights[i];
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      float w = quant_weights[i];
      float slx = sumlx - w * x[i] * L[i];
      float sl2 = suml2 - w * L[i] * L[i];
      if (slx > 0 && sl2 > 0) {
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = MIN(nmax, new_l);
        if (new_l != L[i]) {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (slx * slx * suml2 > sumlx * sumlx * sl2) {
            L[i] = new_l;
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed) {
      break;
    }
  }
  return sumlx / suml2;
}

static void quantize_row_q4_K_impl(const float *x, block_q4_K *y,
                                   int64_t n_per_row,
                                   const float *quant_weights) {
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  uint8_t Ls[QK_K / 32];
  uint8_t Lm[QK_K / 32];
  float weights[32];
  float sw[QK_K / 32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l)
      sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; ++j) {
      if (quant_weights) {
        const float *qw = quant_weights + QK_K * i + 32 * j;
        for (int l = 0; l < 32; ++l)
          weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
      } else {
        for (int l = 0; l < 32; ++l)
          weights[l] = av_x + fabsf(x[32 * j + l]);
      }
      float sumw = 0;
      for (int l = 0; l < 32; ++l)
        sumw += weights[l];
      sw[j] = sumw;
      scales[j] = make_qkx3_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                   &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = Ls[j];
      uint8_t lm = Lm[j];
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = nntr_fp32_to_fp16(d_block);
    y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin =
      nntr_fp32_to_fp16(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d =
        nntr_fp16_to_fp32(y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) * sc;
      if (!d)
        continue;
      const float dm =
        nntr_fp16_to_fp32(y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }
    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l)
        q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

static void quantize_row_q4_K_ref(const float *x, block_q4_K *y, int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  float weights[32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {
    float max_scale =
      0; // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      // scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9,
      // 0.5f);
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l)
        sum_x2 += x[32 * j + l] * x[32 * j + l];
      float av_x = sqrtf(sum_x2 / 32);
      for (int l = 0; l < 32; ++l)
        weights[l] = av_x + fabsf(x[32 * j + l]);
      scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                   &mins[j], Laux, -1.f, 0.1f, 20, false);
      float scale = scales[j];
      if (scale > max_scale) {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min) {
        max_min = min;
      }
    }

    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d =
      nntr_fp32_to_fp16(max_scale / 63.f);
    y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin =
      nntr_fp32_to_fp16(max_min / 63.f);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d =
        nntr_fp16_to_fp32(y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d) * sc;
      if (!d)
        continue;
      const float dm =
        nntr_fp16_to_fp32(y[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l)
        q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

size_t ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                          int64_t n_per_row, const float *quant_weights) {
  size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
  if (!quant_weights) {
    quantize_row_q4_K_ref(src, (block_q4_K *)dst, (int64_t)nrow * n_per_row);
  } else {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
      quantize_row_q4_K_impl(src, (block_q4_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

void ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  block_q4_K *x = (block_q4_K *)x_raw;
  const int nb = k / 256;
  for (int i = 0; i < nb; i++) {
    const uint8_t *q = x[i].qs;

    const float d =
      nntr_fp16_to_fp32(x[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
    const float min =
      nntr_fp16_to_fp32(x[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);

    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < 256; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l)
        *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l)
        *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

void ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  // block_q8_K
  block_q8_K *x_casted = (block_q8_K *)x;
  const int64_t nb = k / 256;
  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < 256; ++j) {
      *y++ = x_casted[i].d * x_casted[i].qs[j];
    }
  }
}
