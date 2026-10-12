// SPDX-License-Identifier: Apache-2.0
/** @file htp_qs2_reader.h
 * @brief Load compact QS2 codes and validated metadata into a private slot.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HTP_QS2_READER_H__
#define __NNTRAINER_HTP_QS2_READER_H__

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace nntrainer {

/** Timings are per-reader elapsed time, not additive model wall time. */
struct HtpQs2ReadStats {
  uint64_t read_us = 0, copy_us = 0, metadata_us = 0, copied_bytes = 0;
};

/**
 * @brief Read an immutable QS2 record using a complete positional-read
 * callback.
 * @param read Returns zero, an errno, or -1 for a short file, like preadAll.
 * @note The caller owns dst exclusively until successful registration. A failed
 *       read can leave partial bytes there; the slot must not be published.
 *       No read is made from the uncached destination.
 */
template <typename Read>
int htpReadQs2Compact(Read read, uint64_t off, uint32_t K, uint32_t N,
                      uint8_t *dst, bool direct, std::vector<uint8_t> &scratch,
                      HtpQs2ReadStats *stats = nullptr) {
  const uint64_t codes64 = static_cast<uint64_t>(K) * N / 4u;
  const uint64_t tail64 = 4u + 8u * static_cast<uint64_t>(N);
  if (!dst || !K || !N || K % 32u || N % 32u ||
      tail64 > std::numeric_limits<size_t>::max() ||
      codes64 > std::numeric_limits<size_t>::max() - tail64 ||
      off > std::numeric_limits<uint64_t>::max() - codes64 - tail64)
    return EINVAL;
  const size_t codes = static_cast<size_t>(codes64);
  const size_t tail = static_cast<size_t>(tail64);
  using Clock = std::chrono::steady_clock;
  auto now = [&] { return stats ? Clock::now() : Clock::time_point{}; };
  auto us = [](Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
        .count());
  };
  scratch.resize(direct ? tail : codes + tail);
  auto t0 = now();
  int rc = read(scratch.data(), scratch.size(), direct ? off + codes : off);
  auto t1 = now();
  if (stats)
    stats->read_us += us(t0, t1);
  if (rc)
    return rc;
  const uint8_t *metadata = scratch.data() + (direct ? 0u : codes);
  const int8_t *palette = reinterpret_cast<const int8_t *>(metadata);
  for (unsigned i = 0; i < 4; ++i)
    if (palette[i] < -8 || palette[i] > 7)
      return EINVAL;
  std::vector<int32_t> sums(N);
  for (uint32_t i = 0; i < N; ++i) {
    float value;
    std::memcpy(&value, metadata + 4u + sizeof(float) * (N + i), sizeof(value));
    if (!std::isfinite(value) || std::fabs(value) > 8.0 * K)
      return EINVAL;
    sums[i] = static_cast<int32_t>(value);
  }
  auto t2 = now();
  if (stats)
    stats->metadata_us += us(t1, t2);
  if (direct) {
    rc = read(dst, codes, off);
  } else {
    std::memcpy(dst, scratch.data(), codes);
  }
  auto t3 = now();
  if (stats) {
    if (direct)
      stats->read_us += us(t2, t3);
    else {
      stats->copy_us += us(t2, t3);
      stats->copied_bytes += codes;
    }
  }
  if (rc)
    return rc;
  std::memcpy(dst + codes, metadata + 4u, sizeof(float) * N);
  std::memcpy(dst + codes + sizeof(float) * N, sums.data(),
              sizeof(int32_t) * N);
  std::memcpy(dst + codes + 8u * N, palette, 4u);
  if (stats)
    stats->metadata_us += us(t3, now());
  return 0;
}
} // namespace nntrainer
#endif
