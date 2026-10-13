// SPDX-License-Identifier: Apache-2.0
/** @file moe_decode_predictor_check.cpp
 * @brief Verify predictor layout, signed nibbles, scales and channel order.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "moe_decode_predictor.h"
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
  const uint8_t packed[] = {0x10, 0x87, 0x0f, 0xfe, 0xa9, 0x0b};
  const float scales[] = {0.5f, 2.0f};
  const float expected[] = {-4, 12, -3.5f, 14, -0.5f, 2, 0, 4, 3.5f, 6};
  const auto small = causallm::decodePredictionWeights(packed, scales, 5, 2);
  for (size_t i = 0; i < small.size(); ++i)
    assert(small[i] == expected[i]);

  constexpr unsigned K = 2048, E = 64;
  std::vector<uint8_t> bytes(E * K / 2);
  std::vector<float> scale(E), input(K);
  for (unsigned e = 0; e < E; ++e) {
    scale[e] = (e + 1) / 1024.0f;
    for (unsigned h = 0; h < K; h += 2)
      bytes[e * (K / 2) + h / 2] =
        ((h * 7 + e * 3) & 15) | (((h * 5 + e * 11 + 1) & 15) << 4);
  }
  for (unsigned h = 0; h < K; ++h)
    input[h] = (static_cast<int>(h % 17) - 8) / 16.0f;
  const auto weights =
    causallm::decodePredictionWeights(bytes.data(), scale.data(), K, E);
  for (unsigned e = 0; e < E; ++e) {
    float dot = 0, reference = 0;
    for (unsigned h = 0; h < K; ++h) {
      dot += input[h] * weights[h * E + e];
      const int nibble =
        (h & 1) ? (((h - 1) * 5 + e * 11 + 1) % 16) : ((h * 7 + e * 3) % 16);
      reference += input[h] * (nibble - 8) * scale[e];
    }
    assert(dot == reference);
  }
  bool rejected = false;
  try {
    causallm::decodePredictionWeights(nullptr, scales, 5, 2);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
  std::puts("MOE Q4 DECODE PREDICTOR OK");
}
