// SPDX-License-Identifier: Apache-2.0
/** @file qs2_inplace_reader_check.cpp
 * @brief Exact compact heap reads, metadata aliases, failures and bounds.
 */
#include "htp_qs2_reader.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <utility>

#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
static std::atomic<size_t> physical_reads{0};
static ssize_t countedPread(int fd, void *dst, size_t bytes, off_t off) {
  ++physical_reads;
  return ::pread(fd, dst, bytes, off);
}
#define pread countedPread
#include "weight_direct_file.h"
#undef pread
#endif

using namespace nntrainer;
static std::atomic<size_t> checks{0};
static void require(bool ok) {
  const size_t index = ++checks;
  if (!ok) {
    std::fprintf(stderr, "QS2 in-place assertion %zu failed\n", index);
    std::abort();
  }
}

struct Record {
  uint32_t K, N;
  size_t codes;
  std::vector<uint8_t> raw;
  Record(uint32_t k, uint32_t n) :
    K(k), N(n), codes(size_t(k) * n / 4), raw(codes + 4 + size_t(8) * n) {
    for (size_t i = 0; i < codes; ++i)
      raw[i] = static_cast<uint8_t>(i * 37u + i / 127u + n);
    const int8_t palette[] = {-8, -8, 0, 7};
    std::memcpy(raw.data() + codes, palette, sizeof(palette));
    const uint32_t scale_bits[] = {0,          0x80000000, 0x7f800000,
                                   0xff800000, 0x7fc12345, 0x7f800001,
                                   1,          0x3f800000};
    const float sums[] = {-8.0f * k, 8.0f * k, -0.0f,  0.0f,
                          -3.875f,   3.875f,   -0.25f, 0.25f};
    for (uint32_t i = 0; i < N; ++i) {
      std::memcpy(raw.data() + codes + 4 + sizeof(float) * i,
                  &scale_bits[i % 8], sizeof(float));
      std::memcpy(raw.data() + codes + 4 + sizeof(float) * (size_t(N) + i),
                  &sums[i % 8], sizeof(float));
    }
  }
  int read(void *dst, size_t bytes, uint64_t off) const {
    if (off > raw.size() || bytes > raw.size() - off)
      return -1;
    std::memcpy(dst, raw.data() + off, bytes);
    return 0;
  }
};

static void validCase(uint32_t K, uint32_t N, size_t shift) {
  const Record record(K, N);
  const size_t bytes = record.raw.size(), prefix = 128 + shift;
  std::vector<uint8_t> expected(bytes, 0), scratch;
  auto read = [&](void *dst, size_t n, uint64_t at) {
    return record.read(dst, n, at);
  };
  require(htpReadQs2Compact(read, 0, K, N, expected.data(), false, scratch) ==
          0);
#ifdef NNTR_QS2_BASELINE
  std::vector<uint8_t> baseline(bytes, 0), old_scratch;
  require(nntrainer_qs2_baseline::htpReadQs2Compact(
            read, 0, K, N, baseline.data(), false, old_scratch) == 0);
  require(expected == baseline);
#endif
  for (bool direct : {false, true}) {
    std::vector<uint8_t> legacy(bytes, 0);
    require(htpReadQs2Compact(read, 0, K, N, legacy.data(), direct, scratch) ==
            0);
    require(legacy == expected);
  }
  for (bool profile : {false, true}) {
    std::vector<uint8_t> actual(prefix + bytes + 128, 0xa5);
    size_t calls = 0, read_bytes = 0;
    HtpQs2ReadStats stats;
    stats.copy_us = 19;
    stats.copied_bytes = 23;
    require(htpReadQs2CompactInPlace(
              [&](void *dst, size_t n, uint64_t at) {
                ++calls;
                read_bytes += n;
                require(dst == actual.data() + prefix);
                return read(dst, n, at);
              },
              0, K, N, actual.data() + prefix, bytes + 31,
              profile ? &stats : nullptr) == 0);
    require(calls == 1 && read_bytes == bytes);
    require(stats.copy_us == 19 && stats.copied_bytes == 23);
    require(
      std::equal(expected.begin(), expected.end(), actual.begin() + prefix));
    require(std::all_of(actual.begin(), actual.begin() + prefix,
                        [](uint8_t b) { return b == 0xa5; }));
    require(std::all_of(actual.begin() + prefix + bytes, actual.end(),
                        [](uint8_t b) { return b == 0xa5; }));
    // Scales are arbitrary bit patterns, including NaNs and signed zero.
    require(std::memcmp(actual.data() + prefix + record.codes,
                        record.raw.data() + record.codes + 4,
                        sizeof(float) * N) == 0);
    const int32_t converted[] = {
      -int32_t(8 * K), int32_t(8 * K), 0, 0, -3, 3, 0, 0};
    for (uint32_t i = 0; i < N; ++i) {
      int32_t value;
      std::memcpy(&value,
                  actual.data() + prefix + record.codes +
                    sizeof(int32_t) * (size_t(N) + i),
                  sizeof(value));
      require(value == converted[i % 8]);
    }
  }
}

static void invalidMetadata() {
  const Record valid(32, 64);
  auto run = [&](Record bad) {
    std::vector<uint8_t> actual(bad.raw.size() + 256, 0xa5);
    std::vector<uint8_t> output(bad.raw.size(), 0xa5), scratch;
    auto read = [&](void *dst, size_t n, uint64_t off) {
      return bad.read(dst, n, off);
    };
    for (bool direct : {false, true}) {
      require(htpReadQs2Compact(read, 0, bad.K, bad.N, output.data(), direct,
                                scratch) == EINVAL);
      require(std::all_of(output.begin(), output.end(),
                          [](uint8_t b) { return b == 0xa5; }));
    }
#ifdef NNTR_QS2_BASELINE
    require(nntrainer_qs2_baseline::htpReadQs2Compact(
              read, 0, bad.K, bad.N, output.data(), false, scratch) == EINVAL);
#endif
    HtpQs2ReadStats stats;
    require(htpReadQs2CompactInPlace(read, 0, bad.K, bad.N, actual.data() + 128,
                                     bad.raw.size(), &stats) == EINVAL);
    // Failed validation leaves the complete raw record untransformed.
    require(std::equal(bad.raw.begin(), bad.raw.end(), actual.begin() + 128));
    require(std::all_of(actual.begin(), actual.begin() + 128,
                        [](uint8_t b) { return b == 0xa5; }));
    require(std::all_of(actual.end() - 128, actual.end(),
                        [](uint8_t b) { return b == 0xa5; }));
    require(stats.copy_us == 0 && stats.copied_bytes == 0);
  };
  for (unsigned at = 0; at < 4; ++at)
    for (int8_t value : {int8_t(-9), int8_t(8)}) {
      Record bad = valid;
      std::memcpy(bad.raw.data() + bad.codes + at, &value, 1);
      run(std::move(bad));
    }
  for (uint32_t at : {0u, valid.N / 2, valid.N - 1})
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::nextafter(8.0f * valid.K, INFINITY),
                        std::nextafter(-8.0f * valid.K, -INFINITY)}) {
      Record bad = valid;
      std::memcpy(bad.raw.data() + bad.codes + 4 +
                    sizeof(float) * (size_t(bad.N) + at),
                  &value, sizeof(value));
      run(std::move(bad));
    }
}

static void boundsAndFailures() {
  Record record(32, 64);
  const size_t bytes = record.raw.size();
  std::vector<uint8_t> output(bytes + 256, 0xa5);
  auto *dst = output.data() + 128;
  size_t calls = 0;
  auto unexpected = [&](void *, size_t, uint64_t) {
    ++calls;
    return 0;
  };
  for (size_t capacity : {size_t(0), bytes - 1})
    require(htpReadQs2CompactInPlace(unexpected, 0, 32, 64, dst, capacity) ==
            EINVAL);
  require(htpReadQs2CompactInPlace(unexpected, 0, 32, 64, nullptr, bytes) ==
          EINVAL);
  require(htpReadQs2CompactInPlace(unexpected, UINT64_MAX, 32, 64, dst,
                                   bytes) == EINVAL);
  require(htpReadQs2CompactInPlace(
            unexpected, 0, 32, 64,
            reinterpret_cast<uint8_t *>(UINTPTR_MAX - bytes + 1),
            bytes) == EINVAL);
  for (auto shape : {std::pair<uint32_t, uint32_t>{0, 64},
                     {32, 0},
                     {31, 64},
                     {32, 63},
                     {UINT32_MAX - 31, 64},
                     {32, UINT32_MAX - 31}})
    require(htpReadQs2CompactInPlace(unexpected, 0, shape.first, shape.second,
                                     dst, bytes) == EINVAL);
  require(calls == 0);
  require(std::all_of(output.begin(), output.end(),
                      [](uint8_t b) { return b == 0xa5; }));
  for (int error : {-1, EIO, EBADF})
    for (size_t partial : {size_t(0), size_t(1), record.codes - 1, record.codes,
                           record.codes + 4, bytes - 1}) {
      std::fill(output.begin(), output.end(), 0xa5);
      HtpQs2ReadStats stats;
      require(htpReadQs2CompactInPlace(
                [&](void *out, size_t n, uint64_t off) {
                  require(n == bytes && off == 0);
                  std::memcpy(out, record.raw.data(), partial);
                  return error;
                },
                0, 32, 64, dst, bytes, &stats) == error);
      require(std::equal(record.raw.begin(), record.raw.begin() + partial,
                         output.begin() + 128));
      require(std::all_of(output.begin(), output.begin() + 128,
                          [](uint8_t b) { return b == 0xa5; }));
      require(std::all_of(output.begin() + 128 + partial, output.end(),
                          [](uint8_t b) { return b == 0xa5; }));
      require(stats.metadata_us == 0 && stats.copy_us == 0 &&
              stats.copied_bytes == 0);
    }
}

#if defined(__linux__) || defined(__ANDROID__)
static void directReads() {
  Record gu(2048, 3072), dn(1536, 2048);
  const char *directory = std::getenv("TMPDIR");
  if (!directory || !*directory) {
#if defined(__ANDROID__)
    directory = "/data/local/tmp";
#else
    directory = "/tmp";
#endif
  }
  const std::string name = std::string(directory) + "/nntr-qs2-inplace-XXXXXX";
  std::vector<char> path(name.begin(), name.end());
  path.push_back('\0');
  const int fd = mkstemp(path.data());
  require(fd >= 0);
  require(unlink(path.data()) == 0);
  const uint64_t off_gu = 4093, off_dn = off_gu + gu.raw.size() + 13;
  require(pwrite(fd, gu.raw.data(), gu.raw.size(), off_gu) ==
          ssize_t(gu.raw.size()));
  require(pwrite(fd, dn.raw.data(), dn.raw.size(), off_dn) ==
          ssize_t(dn.raw.size()));
  WeightDirectFile file(fd);
  for (bool inplace : {false, true}) {
    physical_reads = 0;
    for (auto item :
         {std::make_pair(&gu, off_gu), std::make_pair(&dn, off_dn)}) {
      const auto &record = *item.first;
      std::vector<uint8_t> actual(record.raw.size()), expected(actual.size()),
        scratch;
      auto direct = [&](void *out, size_t n, uint64_t off) {
        return file.read(out, n, off);
      };
      auto memory = [&](void *out, size_t n, uint64_t off) {
        return record.read(out, n, off);
      };
      require(htpReadQs2Compact(memory, 0, record.K, record.N, expected.data(),
                                false, scratch) == 0);
      const int rc =
        inplace
          ? htpReadQs2CompactInPlace(direct, item.second, record.K, record.N,
                                     actual.data(), actual.size())
          : htpReadQs2Compact(direct, item.second, record.K, record.N,
                              actual.data(), false, scratch);
      require(rc == 0 && actual == expected);
    }
    require(physical_reads == 3); // GU two chunks, DN one chunk.
  }
  require(close(fd) == 0);
}
#endif

int main() {
  for (auto shape : {std::pair<uint32_t, uint32_t>{32, 32},
                     {64, 32},
                     {96, 64},
                     {2048, 3072},
                     {1536, 2048},
                     {8192, 32},
                     {32, 16384}})
    for (size_t shift : {0u, 1u, 3u, 7u, 31u, 63u, 127u})
      validCase(shape.first, shape.second, shift);
  invalidMetadata();
  boundsAndFailures();
  std::vector<std::thread> readers;
  for (unsigned i = 0; i < 7; ++i)
    readers.emplace_back([i] { validCase(96, 64, i); });
  for (auto &thread : readers)
    thread.join();
#if defined(__linux__) || defined(__ANDROID__)
  directReads();
#endif
  std::printf("PASS QS2 in-place: %zu assertions, exact bytes/scales/sums, "
              "metadata rejection, guards, partial errors, concurrent reads, "
              "GU2+DN1 O_DIRECT reads\n",
              checks.load());
}
