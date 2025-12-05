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

namespace nntrainer {

void Q4_0Utils::unpackOneBlockQ4_0x4(const block_q4_0x4 *in, block_q4_0 *dst) {
  unsigned int blck_size_interleave = 8;

  for (int i = 0; i < 4; i++) {
    dst[i].d = in->d[i];
  }

  const int end = QK4_0 * 2 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int dst_id = i % 4;
    int dst_offset = (i / 4) * blck_size_interleave;
    int src_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in->qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&dst[dst_id].qs[dst_offset], &elems, sizeof(uint64_t));
  }
}

void Q4_0Utils::unpackBlocksQ4_0x4(const block_q4_0x4 *__restrict src,
                                   size_t data_size, size_t nrow, size_t K,
                                   block_q4_0 *__restrict dst) {
  int interleave_block = 4;

  const block_q4_0x4 *src_ = src;
  block_q4_0 *dst_ = (block_q4_0 *)dst;
  block_q4_0 dst_tmp[4];
  int nblocks = K / QK4_0;

  assert(data_size == (nrow / 4) * nblocks * sizeof(block_q4_0x4));

  for (size_t b = 0; b < nrow; b += interleave_block) {
    for (int64_t x = 0; x < nblocks; x++) {
      unpackOneBlockQ4_0x4(src_++, dst_tmp);

      for (size_t i = 0; i < interleave_block; i++) {
        dst_[x + i * nblocks] = dst_tmp[i];
      }
    }
    dst_ += interleave_block * nblocks;
  }
}

void Q4_0Utils::dequantizeQ4_0x4(const void *q4_weight_repacked, int N, int K,
                                 float *dequantized_weights) {
  assert(K % QK4_0 == 0);
  assert(N % 4 == 0);
  size_t data_size = (K / QK4_0) * (N / 4) * sizeof(block_q4_0x4);
  std::vector<uint8_t> q4_weight_out(data_size);
  unpackBlocksQ4_0x4((block_q4_0x4 *)q4_weight_repacked, data_size, N, K,
                     (block_q4_0 *)q4_weight_out.data());

  nntrainer::dequantize_row_q4_0((const void *)q4_weight_out.data(),
                                 dequantized_weights, K * N);
}

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

void Q4_0Utils::transformQ4_0x_FromInt4(size_t N, size_t K,
                                        const uint8_t *osv32_weights,
                                        const uint16_t *osv32_scales,
                                        size_t scale_group_size,
                                        int q4_0x_block_size, void *dst_q4_0x) {
  nntrainer::transform_q4_0x_from_int4(N, K, osv32_weights, osv32_scales,
                                       scale_group_size, dst_q4_0x);
}

void Q4_0Utils::printBlockQ4_0(const block_q4_0 *block) {
  printf("Q4_0: ");
  for (int i = 0; i < 16; i++) {
    printf("%i %i ", block->qs[i] & 0x0F, (block->qs[i] >> 4) & 0x0F);
  }
  printf("| scale:%f\n", compute_fp16_to_fp32(block->d));
}

} // namespace nntrainer
