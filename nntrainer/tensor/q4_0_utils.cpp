// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_utils.cpp
 * @date	15 October 2025
 * @brief	This is Q4_0Utils class for utils for Q4_0 quantization format.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Maciej Nalewaj <m.nalewaj@samsung.com>
 * @bug		No known bugs
 */

#include <cassert>
#include <cmath>

#include "cpu_backend.h"
#include "fp16.h"
#include "int4_utils.h"
#include "nntrainer_error.h"
#include "q4_0_utils.h"
#include "util_func.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace nntrainer {

void Q4_0Utils::unpackOneBlockQ4_0x8(const block_q4_0x8 *in, block_q4_0 *dst) {
  unsigned int blck_size_interleave = 8;

  for (int i = 0; i < 8; i++) {
    dst[i].d = in->d[i];
  }

  const int end = QK4_0 * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int dst_id = i % 8;
    int dst_offset = (i / 8) * blck_size_interleave;
    int src_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in->qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&dst[dst_id].qs[dst_offset], &elems, sizeof(uint64_t));
  }
}

void Q4_0Utils::unpackBlocksQ4_0x8(const block_q4_0x8 *__restrict src,
                                   size_t data_size, size_t nrow, size_t K,
                                   block_q4_0 *__restrict dst) {
  int interleave_block = 8;

  const block_q4_0x8 *src_ = src;
  block_q4_0 *dst_ = (block_q4_0 *)dst;
  block_q4_0 dst_tmp[8];
  int nblocks = K / QK4_0;

  assert(data_size == (nrow / 8) * nblocks * sizeof(block_q4_0x8));

  for (size_t b = 0; b < nrow; b += interleave_block) {
    for (int64_t x = 0; x < nblocks; x++) {
      unpackOneBlockQ4_0x8(src_++, dst_tmp);

      for (size_t i = 0; i < interleave_block; i++) {
        dst_[x + i * nblocks] = dst_tmp[i];
      }
    }
    dst_ += interleave_block * nblocks;
  }
}

void Q4_0Utils::dequantizeQ4_0x8(const void *q4_weight_repacked, int N, int K,
                                 float *dequantized_weights) {
  assert(K % QK4_0 == 0);
  assert(N % 8 == 0);
  size_t data_size = (K / QK4_0) * (N / 8) * sizeof(block_q4_0x8);
  std::vector<uint8_t> q4_weight_out(data_size);
  unpackBlocksQ4_0x8((block_q4_0x8 *)q4_weight_repacked, data_size, N, K,
                     (block_q4_0 *)q4_weight_out.data());

  nntrainer::dequantize_row_q4_0((const void *)q4_weight_out.data(),
                                 dequantized_weights, K * N);
}

void Q4_0Utils::transformQ4_0Block(const uint8_t *int4_weight, uint16_t scale,
                                   block_q4_0 *block) {
  block->d = scale;

  // Input:  | 0, 1 | 2, 3 | 4, 5 | ... |14,15 |16,17 | ... |28,29 |30,31 |
  // Input:  | A, B | A, B | A, B | ... | A, B | C, D | ... | C, D | C, D |
  //
  // Output: | 0,16 | 1,17 | 2,18 | 3,19 | ...          ... |14,30 |15,31 |
  // Output: | A, C | B, D | A, C | B, D | ...          ... | A, C | B, D |

#ifdef __AVX2__
  // Load 16 bytes of input data
  __m128i input = _mm_loadu_si128((const __m128i *)int4_weight);

  // Create masks for extracting low and high nibbles
  const __m128i low_nibble_mask = _mm_set1_epi8(0x0F);
  const __m128i high_nibble_mask = _mm_set1_epi8(0xF0);
  const __m128i xor_mask = _mm_set1_epi8(0x88);

  // Extract low nibbles from first 8 bytes
  __m128i A = _mm_and_si128(input, low_nibble_mask);

  // Extract high nibbles from first 8 bytes and shift right
  __m128i B = _mm_and_si128(input, high_nibble_mask);
  B = _mm_srli_epi16(B, 4);

  // Extract low nibbles from second 8 bytes
  __m128i input_shifted = _mm_bsrli_si128(input, 8);
  __m128i C = _mm_and_si128(input_shifted, low_nibble_mask);

  // Extract high nibbles from second 8 bytes and shift right
  __m128i D = _mm_and_si128(input_shifted, high_nibble_mask);
  D = _mm_srli_epi16(D, 4);

  // Interleave low nibbles: v0 from first8, v2 from second8
  __m128i AC = _mm_or_si128(A, _mm_slli_epi16(C, 4));

  // Interleave high nibbles: v1 from first8, v3 from second8
  __m128i BD = _mm_or_si128(B, _mm_slli_epi16(D, 4));

  // Pack the results: interleave low and high bytes
  __m128i result = _mm_unpacklo_epi8(AC, BD);

  // Store the 16 bytes result
  _mm_storeu_si128((__m128i *)block->qs, result);
#else
  // Scalar version for non-AVX2 systems
  for (int i = 0; i < 8; i++) {
    char v0 = int4_weight[i] & 0xF;
    char v1 = (int4_weight[i] >> 4) & 0xF;
    char v2 = int4_weight[8 + i] & 0xF;
    char v3 = (int4_weight[8 + i] >> 4) & 0xF;
    block->qs[2 * i] = (v0 | (v2 << 4));
    block->qs[2 * i + 1] = (v1 | (v3 << 4));
  }
#endif
}

inline static block_q4_0x8 nntr_make_block_q4_0x8(block_q4_0 *in) {
  block_q4_0x8 out;
  constexpr size_t IN_CNT = 8;
  constexpr size_t HALF_SIZE = 8;

  for (int i = 0; i < IN_CNT; ++i) {
    out.d[i] = in[i].d;
  }

  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out.qs[i * HALF_SIZE], &in[i].qs[0], HALF_SIZE);
  }
  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out.qs[IN_CNT * HALF_SIZE + i * HALF_SIZE], &in[i].qs[8],
           HALF_SIZE);
  }
  return out;
}

void Q4_0Utils::transformQ4_0x8FromInt4(size_t N, size_t K,
                                        const uint8_t *osv32_weights,
                                        const uint16_t *osv32_scales,
                                        size_t scale_group_size,
                                        void *dst_q4_0x8) {

  NNTR_THROW_IF((!(scale_group_size == 32 || scale_group_size == 64 ||
                   scale_group_size == 128)),
                std::invalid_argument)
    << "Scale group size must be 32/64/128";
  NNTR_THROW_IF(K % QK4_0 != 0, std::invalid_argument)
    << "K size must be divisable by QK4_0 (32)";
  NNTR_THROW_IF(N % 8 != 0, std::invalid_argument)
    << "N size must be divisable by 8";

  static constexpr const size_t ROW_BLOCK_SIZE = 32;
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  constexpr size_t nrows_interleaved = 8;
  uint8_t int4_weight[16];
  uint16_t scale;
  block_q4_0 dst_tmp[8];
  block_q4_0x8 *dst_ = (block_q4_0x8 *)dst_q4_0x8;

  // --- Layout ---
  const size_t rows_count_pad = align(N, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(K, ROW_BLOCK_SIZE);
  const size_t column_blocks_count =
    columns_count_pad / COLUMN_BLOCK_SIZE; // COLUMN_BLOCK_SIZE == 2
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;

  for (size_t row_id = 0; row_id < N; row_id += nrows_interleaved) {
    const size_t row_block_id = row_id / ROW_BLOCK_SIZE;
    size_t i_in_block = row_id % ROW_BLOCK_SIZE;
    for (int64_t column_idx = 0; column_idx < K; column_idx += QK4_0) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        int row_idx = row_id + i;
        // Address the bytes for this row
        const size_t row_block_base =
          row_block_id * bytes_per_row_block_span + i_in_block + i;
        int index0 = row_block_base + (column_idx / 2) * ROW_BLOCK_SIZE;

        for (size_t column_block_id = 0; column_block_id < 16;
             ++column_block_id) {
          int4_weight[column_block_id] =
            osv32_weights[index0 + column_block_id * ROW_BLOCK_SIZE];
        }
        scale = osv32_scales[row_idx +
                             (column_idx / scale_group_size) * rows_count_pad];

        transformQ4_0Block(int4_weight, scale, &dst_tmp[i]);
      }
      *dst_++ = nntr_make_block_q4_0x8(dst_tmp);
    }
  }
}

void Q4_0Utils::printBlockQ4_0(const block_q4_0 *block) {
  printf("Q4_0: ");
  for (int i = 0; i < 16; i++) {
    printf("%i %i ", block->qs[i] & 0x0F, (block->qs[i] >> 4) & 0x0F);
  }
  printf("| scale:%f\n", compute_fp16_to_fp32(block->d));
}

} // namespace nntrainer
