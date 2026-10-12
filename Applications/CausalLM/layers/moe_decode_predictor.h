// SPDX-License-Identifier: Apache-2.0
/** @file moe_decode_predictor.h
 * @brief CPU prediction weights from the inference router's QS4CX values.
 */
#ifndef NNTR_MOE_DECODE_PREDICTOR_H
#define NNTR_MOE_DECODE_PREDICTOR_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace causallm {
/** QS4CX stores output-major rows, even K in the low nibble, as q + 8.
 * Return [hidden, experts] so the prediction loop vectorizes over experts.
 * This is a prediction-only copy; actual routing continues on the backend.
 */
inline std::vector<float> decodePredictionWeights(const uint8_t *packed,
                                                  const float *scales,
                                                  unsigned hidden,
                                                  unsigned experts) {
  if (!packed || !scales || !hidden || !experts)
    throw std::invalid_argument("Invalid decode prediction weights");
  std::vector<float> weights(static_cast<size_t>(hidden) * experts);
  const size_t stride = (static_cast<size_t>(hidden) + 1) / 2;
  for (unsigned e = 0; e < experts; ++e)
    for (unsigned h = 0; h < hidden; ++h) {
      const uint8_t byte = packed[e * stride + h / 2];
      const int q = static_cast<int>((byte >> (4 * (h & 1))) & 15) - 8;
      weights[static_cast<size_t>(h) * experts + e] = q * scales[e];
    }
  return weights;
}
} // namespace causallm
#endif
