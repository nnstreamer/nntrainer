#ifndef __NNTRAINER_INT4_UTILS_H__
#define __NNTRAINER_INT4_UTILS_H__

#include <algorithm>
#include <vector>

namespace nntrainer {

class Int4Utils {
public:
  static constexpr const size_t ROW_BLOCK_SIZE = 32;
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  static int clampToInt4(const int value) { return std::clamp(value, -8, 7); }

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
};

} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__
