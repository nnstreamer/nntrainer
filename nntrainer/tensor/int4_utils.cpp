#include "int4_utils.h"

#include <cassert>
#include <cmath>

#include "cpu_backend.h"
#include "fp16.h"
#include "nntrainer_error.h"

namespace nntrainer {

float Int4Utils::computeScaleForGroup(const float *group_weights,
                                      const size_t group_size) {
  auto max_absolute_weight = 0.0f;

  for (size_t i = 0; i < group_size; ++i) {
    auto weight = group_weights[i];

    NNTR_THROW_IF(!std::isfinite(weight), std::invalid_argument)
      << "Weight is not finite value";

    const auto absolute_weight = std::abs(weight);

    if (absolute_weight > max_absolute_weight) {
      max_absolute_weight = absolute_weight;
    }
  }

  auto group_scale =
    (max_absolute_weight == 0.0f) ? 1.0f : (max_absolute_weight / 7.0f);

  NNTR_THROW_IF(!std::isfinite(group_scale), std::invalid_argument)
    << "Scale is not finite value";

  return group_scale;
}

void Int4Utils::computeScales(const float *weights, const size_t rows_count,
                              const size_t columns_count,
                              const size_t group_size,
                              std::vector<float> &scales) {
  NNTR_THROW_IF(columns_count % group_size, std::invalid_argument)
    << "Columns size not divisible by group size";

  const auto groups_per_row = columns_count / group_size;
  scales.resize(rows_count * groups_per_row, 1.0f);

  for (size_t row_id = 0; row_id < rows_count; ++row_id) {
    const auto *weights_row = weights + (row_id * columns_count);

    for (size_t group_id = 0; group_id < groups_per_row; ++group_id) {
      const auto *weights_group = weights_row + (group_id * group_size);
      scales[(group_id * rows_count) + row_id] =
        computeScaleForGroup(weights_group, group_size);
    }
  }
}

void Int4Utils::quantize(const float *weights, const size_t rows_count,
                         const size_t columns_count, const size_t group_size,
                         std::vector<uint8_t> &out_weights,
                         std::vector<uint16_t> &out_scales) {
  NNTR_THROW_IF(!weights, std::invalid_argument) << "Weight cannot be null";

  NNTR_THROW_IF((rows_count <= 0), std::invalid_argument)
    << "Rows count needs to be greater than 0";

  NNTR_THROW_IF((columns_count <= 0), std::invalid_argument)
    << "Columns count needs to be greater than 0";

  NNTR_THROW_IF((!(group_size == 32 || group_size == 64 || group_size == 128)),
                std::invalid_argument)
    << "Group size must be 32/64/128";

  std::vector<float> scales_fp32;
  computeScales(weights, rows_count, columns_count, group_size, scales_fp32);

  out_scales.resize(scales_fp32.size());
  for (size_t scale_id = 0; scale_id < scales_fp32.size(); ++scale_id) {
    out_scales[scale_id] = compute_fp32_to_fp16(scales_fp32[scale_id]);
  }

  out_weights.resize(rows_count * columns_count);

  for (size_t row_id = 0; row_id < rows_count; ++row_id) {
    for (size_t column_id = 0; column_id < columns_count; ++column_id) {

      const float weight = weights[(row_id * columns_count) + column_id];
      const float scale =
        scales_fp32[row_id + ((column_id / group_size) * rows_count)];

      out_weights[(row_id * columns_count) + column_id] =
        quantizeToInt4(weight, scale);
    }
  }
}

uint8_t Int4Utils::pack(const float *weights, const float *scales,
                        const size_t row_id, const size_t column_id,
                        const size_t groups_per_row, const size_t group_size,
                        const size_t rows_count, const size_t columns_count) {
  {
    const float scale =
      scales[row_id + ((column_id / group_size) * rows_count)];
    const float weight = weights[(row_id * columns_count) + column_id];
    return quantizeToInt4(weight, scale);
  }
}

void Int4Utils::quantizeAndRepack(const float *weights, const size_t rows_count,
                                  const size_t columns_count,
                                  const size_t group_size,
                                  std::vector<uint8_t> &out_weights,
                                  std::vector<uint16_t> &out_scales) {
  NNTR_THROW_IF(!weights, std::invalid_argument) << "Weight cannot be null";

  NNTR_THROW_IF((rows_count <= 0), std::invalid_argument)
    << "Rows count needs to be greater than 0";

  NNTR_THROW_IF((columns_count <= 0), std::invalid_argument)
    << "Columns count needs to be greater than 0";

  NNTR_THROW_IF((!(group_size == 32 || group_size == 64 || group_size == 128)),
                std::invalid_argument)
    << "Group size must be 32/64/128";

  std::vector<float> scales_fp32;
  computeScales(weights, rows_count, columns_count, group_size, scales_fp32);

  out_scales.resize(scales_fp32.size());
  for (size_t scale_id = 0; scale_id < scales_fp32.size(); ++scale_id) {
    out_scales[scale_id] = compute_fp32_to_fp16(scales_fp32[scale_id]);
  }

  NNTR_THROW_IF(rows_count % ROW_BLOCK_SIZE, std::invalid_argument)
    << "Rows size not divisible by row block size";

  NNTR_THROW_IF(columns_count % COLUMN_BLOCK_SIZE, std::invalid_argument)
    << "Columns size not divisible by column block size";

  // Prepare output buffer in OS_IS_YX_OSV32_ISV2 layout
  const auto groups_per_row = columns_count / group_size;
  const auto row_blocks_count = rows_count / ROW_BLOCK_SIZE;
  const auto column_blocks_count = columns_count / COLUMN_BLOCK_SIZE;
  out_weights.resize((rows_count * columns_count) / 2, 0);

  size_t out_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        const auto column_id_absolute_lo =
          (column_block_id * COLUMN_BLOCK_SIZE);
        const auto column_id_absolute_hi = column_id_absolute_lo + 1;

        lo = pack(weights, scales_fp32.data(), row_id_absolute,
                  column_id_absolute_lo, groups_per_row, group_size, rows_count,
                  columns_count);

        hi = pack(weights, scales_fp32.data(), row_id_absolute,
                  column_id_absolute_hi, groups_per_row, group_size, rows_count,
                  columns_count);

        out_weights[out_idx++] = uint8_t((hi << 4) | lo);
      }
    }
  }
}

uint8_t Int4Utils::quantizeToInt4(const float weight, const float scale) {
  auto div = std::nearbyintf(weight / scale);

  if (std::isnan(div)) {
    div = 0.0f;
  }

  div = std::clamp(div, -8.0f, 7.0f);
  int quantized = (int)div;
  return uint8_t(quantized & 0xF);
}

int Int4Utils::convertInt4ToInt(const uint8_t int4_value) {
  static int lookup[] = {0,  1,  2,  3,  4,  5,  6,  7,
                         -8, -7, -6, -5, -4, -3, -2, -1};

  return lookup[int4_value];
}

void Int4Utils::dequantizePacked(const std::vector<uint8_t> &weights,
                                 const std::vector<uint16_t> &scales,
                                 const size_t rows_count,
                                 const size_t columns_count,
                                 const size_t group_size,
                                 std::vector<float> &dequantized_weights) {
  const auto groups_per_row = columns_count / group_size;
  const auto row_blocks_count = rows_count / ROW_BLOCK_SIZE;
  const auto column_blocks_count = columns_count / COLUMN_BLOCK_SIZE;

  dequantized_weights.resize(rows_count * columns_count);

  size_t out_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        const auto column_id_absolute_lo =
          (column_block_id * COLUMN_BLOCK_SIZE);
        const auto column_id_absolute_hi = column_id_absolute_lo + 1;

        const auto scale_lo =
          scales[row_id_absolute +
                 ((column_id_absolute_lo / group_size) * rows_count)];

        const auto scale_hi =
          scales[row_id_absolute +
                 ((column_id_absolute_hi / group_size) * rows_count)];

        const auto weight = weights[out_idx++];
        const auto weight_lo = weight & 0xF;
        const auto weight_hi = (weight >> 4) & 0xF;

        dequantized_weights[(row_id_absolute * columns_count) +
                            column_id_absolute_lo] =
          Int4Utils::convertInt4ToInt(weight_lo) *
          nntrainer::compute_fp16_to_fp32(scale_lo);
        dequantized_weights[(row_id_absolute * columns_count) +
                            column_id_absolute_hi] =
          Int4Utils::convertInt4ToInt(weight_hi) *
          nntrainer::compute_fp16_to_fp32(scale_hi);
      }
    }
  }
}

void Int4Utils::nntr_depack_block_q4_0x8(const block_q4_0x8 *in,
                                         block_q4_0 *dst,
                                         unsigned int blck_size_interleave) {
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

void Int4Utils::nntr_depack_q4_0_8_bl_to_q4_0(void *__restrict dst,
                                              const void *__restrict data,
                                              size_t data_size, size_t nrow,
                                              size_t k) {
  int interleave_block = 8;

  const block_q4_0x8 *src_ = (const block_q4_0x8 *)data;
  block_q4_0 *dst_ = (block_q4_0 *)dst;
  block_q4_0 dst_tmp[8];
  int nblocks = k / QK4_0;

  assert(data_size == (nrow / 8) * nblocks * sizeof(block_q4_0x8));

  for (size_t b = 0; b < nrow; b += interleave_block) {
    for (int64_t x = 0; x < nblocks; x++) {
      nntr_depack_block_q4_0x8(src_++, dst_tmp, interleave_block);

      for (size_t i = 0; i < interleave_block; i++) {
        dst_[x + i * nblocks] = dst_tmp[i];
      }
    }
    dst_ += interleave_block * nblocks;
  }
}

void Int4Utils::dequantize_q4_0(void *q4_weight_repack, float *weight_fp32_out,
                                int N, int K) {
  std::vector<float> q4_weight_out(N * K);
  nntr_depack_q4_0_8_bl_to_q4_0(q4_weight_out.data(), q4_weight_repack,
                                (K / QK4_0) * (N / 8) * sizeof(block_q4_0x8), N,
                                K);

  nntrainer::dequantize_row_q4_0((const void *)q4_weight_out.data(),
                                 weight_fp32_out, K * N);
}

} // namespace nntrainer
