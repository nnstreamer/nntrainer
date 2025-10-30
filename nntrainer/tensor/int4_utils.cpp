// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_utils.cpp
 * @date	15 October 2025
 * @brief	This is Int4Utils class for utils for INT4 quantization format.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Grzegorz Kisala <gkisala@gmail.com>
 * @bug		No known bugs
 */

#include "int4_utils.h"

#include <cassert>
#include <cmath>

#include "cpu_backend.h"
#include "fp16.h"
#include "nntrainer_error.h"
#include "util_func.h"

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
  // NNTR_THROW_IF(columns_count % group_size, std::invalid_argument)
  //   << "Columns size not divisible by group size";
  NNTR_THROW_IF(columns_count % 4, std::invalid_argument)
    << "Columns size not divisible by 4";

  const auto full_groups_per_row = columns_count / group_size;
  const auto last_group_size = columns_count % group_size;
  const auto padded_groups_per_row = ceilDiv(columns_count, group_size);
  const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  scales.resize(rows_count_pad * padded_groups_per_row, 1.0f);

  for (size_t row_id = 0; row_id < rows_count; ++row_id) {
    const auto *weights_row = weights + (row_id * columns_count);

    for (size_t group_id = 0; group_id < full_groups_per_row; ++group_id) {
      const auto *weights_group = weights_row + (group_id * group_size);
      scales[(group_id * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, group_size);
    }

    // Compute scale for the last padded group
    if (last_group_size > 0) {
      const auto *weights_group =
        weights_row + (full_groups_per_row * group_size);
      scales[(full_groups_per_row * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, last_group_size);
    }
  }
}

uint8_t Int4Utils::pack(const float *weights, const float *scales,
                        const size_t row_id, const size_t column_id,
                        const size_t groups_per_row, const size_t group_size,
                        const size_t rows_count, const size_t columns_count) {
  {
    const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
    const float scale =
      scales[row_id + ((column_id / group_size) * rows_count_pad)];
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

  NNTR_THROW_IF(columns_count % COLUMN_BLOCK_SIZE, std::invalid_argument)
    << "Columns size not divisible by column block size";

  // Prepare output buffer in OS_IS_YX_OSV32_ISV2 layout
  const auto groups_per_row = ceilDiv(columns_count, group_size);
  const auto row_blocks_count = ceilDiv(rows_count, ROW_BLOCK_SIZE);
  const auto columns_count_pad = align(columns_count, group_size);
  const auto column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE);
  const auto rows_count_pad = row_blocks_count * ROW_BLOCK_SIZE;

  out_weights.resize((rows_count_pad * columns_count_pad) / 2, 0);

  size_t out_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        if (row_id_absolute < rows_count) {
          const auto column_id_absolute_lo =
            (column_block_id * COLUMN_BLOCK_SIZE);
          if (column_id_absolute_lo < columns_count) {
            lo = pack(weights, scales_fp32.data(), row_id_absolute,
                      column_id_absolute_lo, groups_per_row, group_size,
                      rows_count, columns_count);

            const auto column_id_absolute_hi = column_id_absolute_lo + 1;
            if (column_id_absolute_hi < columns_count) {
              hi = pack(weights, scales_fp32.data(), row_id_absolute,
                        column_id_absolute_hi, groups_per_row, group_size,
                        rows_count, columns_count);
            }
          }
        }

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
  const auto groups_per_row = ceilDiv(columns_count, group_size);
  const auto rows_count_pad = align(rows_count, group_size);
  const auto row_blocks_count = ceilDiv(rows_count, ROW_BLOCK_SIZE);
  const auto columns_count_pad = align(columns_count, group_size);
  const auto column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE);

  dequantized_weights.resize(rows_count * columns_count);

  size_t weights_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        if (row_id_absolute < rows_count) {
          const auto column_id_absolute_lo =
            (column_block_id * COLUMN_BLOCK_SIZE);
          if (column_id_absolute_lo < columns_count) {
            const auto column_id_absolute_hi = column_id_absolute_lo + 1;

            const auto scale_lo =
              scales[row_id_absolute +
                     ((column_id_absolute_lo / group_size) * rows_count_pad)];

            const auto scale_hi =
              scales[row_id_absolute +
                     ((column_id_absolute_hi / group_size) * rows_count_pad)];

            const auto weight = weights[weights_idx];
            const auto weight_lo = weight & 0xF;
            const auto weight_hi = (weight >> 4) & 0xF;

            dequantized_weights[(row_id_absolute * columns_count) +
                                column_id_absolute_lo] =
              Int4Utils::convertInt4ToInt(weight_lo) *
              nntrainer::compute_fp16_to_fp32(scale_lo);

            if (column_id_absolute_hi < columns_count) {
              dequantized_weights[(row_id_absolute * columns_count) +
                                  column_id_absolute_hi] =
                Int4Utils::convertInt4ToInt(weight_hi) *
                nntrainer::compute_fp16_to_fp32(scale_hi);
            }
          }
        }
        weights_idx++;
      }
    }
  }
}

} // namespace nntrainer
