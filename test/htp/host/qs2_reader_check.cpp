// SPDX-License-Identifier: Apache-2.0
/** @file qs2_reader_check.cpp
 * @brief QS2 slot bytes, failed reads, metadata rejection and concurrent loads.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "htp_qs2_mapped_file.h"
#include "htp_qs2_reader.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <thread>
#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif

using namespace nntrainer;

static void check(uint32_t K, uint32_t N) {
  const size_t codes = static_cast<size_t>(K) * N / 4;
  const size_t bytes = codes + 4 + 8u * N, offset = 37, guard = 128;
  std::vector<uint8_t> file(offset + bytes, 0), expected(bytes);
  for (size_t i = 0; i < codes; ++i)
    expected[i] = file[offset + i] = static_cast<uint8_t>(i * 13u + i / 251u);
  const int8_t palette[] = {-8, -3, 2, 7};
  std::memcpy(file.data() + offset + codes, palette, 4);
  std::memcpy(expected.data() + codes + 8u * N, palette, 4);
  for (uint32_t i = 0; i < N; ++i) {
    const float scale = (i + 1) * 0.03125f;
    const int32_t sum =
      static_cast<int32_t>(i % (8u * K)) - 4 * static_cast<int32_t>(K);
    const float stored_sum = static_cast<float>(sum);
    std::memcpy(file.data() + offset + codes + 4 + 4u * i, &scale, 4);
    std::memcpy(file.data() + offset + codes + 4 + 4u * (N + i), &stored_sum,
                4);
    std::memcpy(expected.data() + codes + 4u * i, &scale, 4);
    std::memcpy(expected.data() + codes + 4u * (N + i), &sum, 4);
  }
  auto run = [&](bool direct, bool failures) {
    std::vector<uint8_t> dst(bytes + 2 * guard, 0xa5), scratch;
    size_t read_bytes = 0;
    auto read = [&](void *out, size_t len, uint64_t at) {
      assert(at <= file.size() && len <= file.size() - at);
      std::memcpy(out, file.data() + at, len);
      read_bytes += len;
      return 0;
    };
    for (unsigned reuse = 0; reuse < 3; ++reuse) {
      std::fill(dst.begin(), dst.end(), 0xa5);
      read_bytes = 0;
      HtpQs2ReadStats stats;
      assert(htpReadQs2Compact(read, offset, K, N, dst.data() + guard, direct,
                               scratch, &stats) == 0);
      assert(read_bytes == bytes);
      assert(stats.copied_bytes == (direct ? 0 : codes));
      assert(scratch.size() == (direct ? bytes - codes : bytes));
      assert(std::equal(expected.begin(), expected.end(), dst.begin() + guard));
      assert(std::all_of(dst.begin(), dst.begin() + guard,
                         [](uint8_t c) { return c == 0xa5; }));
      assert(std::all_of(dst.end() - guard, dst.end(),
                         [](uint8_t c) { return c == 0xa5; }));
    }
    if (!failures)
      return;
    for (int fail : {-1, EIO}) {
      assert(htpReadQs2Compact([&](void *, size_t, uint64_t) { return fail; },
                               offset, K, N, dst.data() + guard, direct,
                               scratch) == fail);
      if (direct) {
        auto fail_codes = [&](void *out, size_t len, uint64_t at) {
          return at == offset ? fail : read(out, len, at);
        };
        assert(htpReadQs2Compact(fail_codes, offset, K, N, dst.data() + guard,
                                 true, scratch) == fail);
      }
    }
  };
  run(false, true);
  run(true, true);
  std::vector<std::thread> readers;
  for (unsigned i = 0; i < 4; ++i)
    readers.emplace_back(run, i % 2 != 0, false);
  for (auto &thread : readers)
    thread.join();

  std::vector<uint8_t> dst(bytes, 0xa5), scratch;
  auto read = [&](void *out, size_t len, uint64_t at) {
    std::memcpy(out, file.data() + at, len);
    return 0;
  };
  for (bool direct : {false, true}) {
    const auto original = file;
    for (int8_t invalid : {int8_t(-9), int8_t(8)}) {
      file[offset + codes] = static_cast<uint8_t>(invalid);
      assert(htpReadQs2Compact(read, offset, K, N, dst.data(), direct,
                               scratch) == EINVAL);
    }
    file = original;
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          static_cast<float>(8u * K + 1u)}) {
      std::memcpy(file.data() + offset + codes + 4 + 4u * N, &invalid, 4);
      assert(htpReadQs2Compact(read, offset, K, N, dst.data(), direct,
                               scratch) == EINVAL);
    }
    file = original;
    assert(htpReadQs2Compact(read, UINT64_MAX, K, N, dst.data(), direct,
                             scratch) == EINVAL);
    assert(htpReadQs2Compact(read, offset, 0, N, dst.data(), direct, scratch) ==
           EINVAL);
  }
}

#if defined(__linux__) || defined(__ANDROID__)
static void checkMappedFiles() {
  std::FILE *first = std::tmpfile(), *second = std::tmpfile();
  assert(first && second);
  std::vector<uint8_t> a(8193), b(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<uint8_t>(i * 17u);
    b[i] = static_cast<uint8_t>(i * 31u + 1u);
  }
  assert(std::fwrite(a.data(), 1, a.size(), first) == a.size());
  assert(std::fwrite(b.data(), 1, b.size(), second) == b.size());
  assert(!std::fflush(first) && !std::fflush(second));
  HtpQs2MappedFiles files;
  int fd = dup(fileno(first));
  assert(fd >= 0);
  auto compare = [&](const std::vector<uint8_t> &source) {
    std::vector<uint8_t> actual(source.size());
    assert(files.read(fd, actual.data(), actual.size(), 0) == 0);
    assert(actual == source);
    assert(files.read(fd, actual.data(), 29, 4093) == 0);
    assert(
      std::equal(actual.begin(), actual.begin() + 29, source.begin() + 4093));
    assert(files.read(fd, actual.data(), actual.size(), 1) == -1);
    assert(files.read(fd, actual.data(), 1, UINT64_MAX) == -1);
  };
  std::vector<std::thread> readers;
  for (unsigned i = 0; i < 4; ++i)
    readers.emplace_back(compare, std::cref(a));
  for (auto &thread : readers)
    thread.join();
  assert(close(fd) == 0);
  assert(dup2(fileno(second), fd) == fd);
  compare(b); // The reused descriptor must not read the first file's mapping.
  assert(close(fd) == 0);
  uint8_t value;
  assert(files.read(fd, &value, 1, 0) == EBADF);
  std::fclose(first);
  std::fclose(second);
}
#endif

int main() {
  check(32, 32);
  check(2048, 3072);
  check(1536, 2048);
#if defined(__linux__) || defined(__ANDROID__)
  checkMappedFiles();
#endif
  std::cout
    << "QS2 reader: bytes, guards, failures, concurrent reuse and mmap PASS\n";
}
