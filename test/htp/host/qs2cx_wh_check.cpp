// SPDX-License-Identifier: Apache-2.0
/**
 * @file qs2cx_wh_check.cpp
 * @brief Check compact expert expansion against an independent tile encoder.
 * @bug No known bugs except for NYI items
 * @author NNTrainer contributors
 */
#include <htp_wh_layout.h>

#include <array>
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
  const std::array<std::array<int8_t, 4>, 3> palettes = {
    {{{-8, -3, 0, 7}}, {{-2, -1, 0, 1}}, {{-8, -8, 7, 7}}}};
  size_t checked = 0;
  for (uint32_t K : {32u, 64u, 96u}) {
    for (uint32_t N : {32u, 64u, 160u}) {
      for (const auto &palette : palettes) {
        const size_t bytes = nntrainer::whBytes(K, N);
        std::vector<uint8_t> compact(bytes / 2, 0);
        std::vector<int8_t> expected(static_cast<size_t>(K) * N);
        // The model format's coordinate definition, independent of the
        // expansion's 128-byte plane walk and nibble-pair lookup table.
        for (uint32_t k = 0; k < K; ++k) {
          for (uint32_t n = 0; n < N; ++n) {
            const uint32_t row = k % 32, col = n % 32;
            const uint32_t slot =
              (row / 8) * 256 + col * 8 + (row % 4) * 2 + (row / 4) % 2;
            const uint32_t old_byte = slot / 2;
            const size_t tile = static_cast<size_t>(k / 32) * (N / 32) + n / 32;
            const size_t byte =
              tile * 256 + (old_byte / 256) * 128 + old_byte % 128;
            const unsigned shift = 2 * (slot % 2) + ((old_byte / 128) % 2) * 4;
            const unsigned index = (k * 13 + n * 7 + k / 3 + n / 5) % 4;
            compact[byte] |= static_cast<uint8_t>(index << shift);
            expected[static_cast<size_t>(k) * N + n] = palette[index];
          }
        }
        std::vector<uint8_t> output(bytes + 2, 0xa5);
        assert(nntrainer::wh2Expand(compact.data(), palette.data(), K, N,
                                    output.data() + 1));
        assert(output.front() == 0xa5 && output.back() == 0xa5);
        std::vector<int8_t> actual(expected.size());
        nntrainer::whUnpack(output.data() + 1, K, N, actual.data());
        assert(actual == expected);
        checked += actual.size();
      }
    }
  }
  uint8_t packed[256]{}, output[512];
  std::fill(std::begin(output), std::end(output), 0xa5);
  int8_t bad[4] = {-9, -1, 0, 1};
  assert(!nntrainer::wh2Expand(packed, bad, 32, 32, output));
  bad[0] = -8;
  bad[3] = 8;
  assert(!nntrainer::wh2Expand(packed, bad, 32, 32, output));
  bad[3] = 7;
  assert(!nntrainer::wh2Expand(packed, bad, 31, 32, output));
  assert(!nntrainer::wh2Expand(packed, bad, 32, 0, output));
  for (uint8_t byte : output)
    assert(byte == 0xa5);
  std::printf("QS2CX WH: %zu weights exact; guards and invalid inputs OK\n",
              checked);
}
