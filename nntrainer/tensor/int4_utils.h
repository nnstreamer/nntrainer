#ifndef __NNTRAINER_INT4_UTILS_H__
#define __NNTRAINER_INT4_UTILS_H__

#include <algorithm>
#include <cstdint>
#include <vector>

// TODO those structures should be defined in one place
#define QK4_0 32

struct block_q4_0x8 {
  uint16_t d[8];   // 16B
  uint8_t qs[128]; // 16 x u64
};

struct block_q4_0 {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
};

namespace nntrainer {

class Int4Utils {
public:
  static constexpr const size_t ROW_BLOCK_SIZE = 32;
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  static float computeScaleForGroup(const float *group_weights,
                                    const size_t group_size);

  static void computeScales(const float *weights, const size_t rows_count,
                            const size_t columns_count, const size_t group_size,
                            std::vector<float> &scales);

  static void quantize(const float *weights, const size_t rows_count,
                       const size_t columns_count, const size_t group_size,
                       std::vector<uint8_t> &out_weights,
                       std::vector<uint16_t> &out_scales);

  static uint8_t pack(const float *weights, const float *scales,
                      const size_t row_id, const size_t column_id,
                      const size_t groups_per_row, const size_t group_size,
                      const size_t rows_count, const size_t columns_count);

  static void quantizeAndRepack(const float *weights, const size_t rows_count,
                                const size_t columns_count,
                                const size_t group_size,
                                std::vector<uint8_t> &out_weights,
                                std::vector<uint16_t> &out_scales);

  static uint8_t quantizeToInt4(const float weight, const float scale);

  static int convertInt4ToInt(const uint8_t int4_value);

  static void dequantizePacked(const uint8_t *weights, const uint16_t *scales,
                               const size_t rows_count,
                               const size_t columns_count,
                               const size_t group_size,
                               std::vector<float> &dequantized_weights);

  // TODO begin
  // temporary put Q4_0 depack here in order to be able to run it both in tests
  // and llm inference
  static void nntr_depack_block_q4_0x8(const block_q4_0x8 *in, block_q4_0 *dst,
                                       unsigned int blck_size_interleave);

  static void nntr_depack_q4_0_8_bl_to_q4_0(void *__restrict dst,
                                            const void *__restrict data,
                                            size_t data_size, size_t nrow,
                                            size_t k);

  static void dequantize_q4_0(void *q4_weight_repack, float *weight_fp32_out,
                              int N, int K);
  // TODO end
};

} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__
