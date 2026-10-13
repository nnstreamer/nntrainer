// SPDX-License-Identifier: Apache-2.0
#ifndef __NNTRAINER_EXPERT_HOST_STAGING_FIXTURE_H__
#define __NNTRAINER_EXPERT_HOST_STAGING_FIXTURE_H__
#if defined(__linux__) || defined(__ANDROID__)
#include "htp_expert_host_staging.h"
#include "htp_qs2_reader.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <deque>
#include <future>
#include <thread>
#include <type_traits>

struct Desc {
  const void *key_gu, *key_dn;
  int fd;
  size_t off_gu, off_dn;
  unsigned K, inter, N_out, weight_bits = 2;
  bool tl2_packed = false;
};
using Bank = nntrainer::HtpExpertHostStaging<Desc>;
static_assert(!std::is_copy_constructible<Bank::Lease>::value,
              "lease owns one count");
static_assert(!std::is_move_constructible<Bank::Lease>::value,
              "lease address is stable");
static size_t checks;
static inline void require(bool condition) {
  ++checks;
  if (!condition)
    throw std::runtime_error("host staging assertion " +
                             std::to_string(checks));
}
static inline int readAll(int fd, void *dst, size_t n, uint64_t off) {
  auto *p = static_cast<uint8_t *>(dst);
  while (n) {
    const ssize_t got = pread(fd, p, n, off);
    if (got < 0) {
      if (errno == EINTR)
        continue;
      return errno;
    }
    if (!got)
      return -1;
    p += got;
    off += got;
    n -= got;
  }
  return 0;
}
static inline int readRecord(Bank::Record &record) {
  auto read = [&](void *dst, size_t n, uint64_t off) {
    return readAll(record.fd(), dst, n, off);
  };
  std::vector<uint8_t> scratch;
  auto &d = record.d;
  int rc = nntrainer::htpReadQs2Compact(read, d.off_gu, d.K, 2 * d.inter,
                                        record.payload.get(), true, scratch);
  if (!rc)
    rc = nntrainer::htpReadQs2Compact(read, d.off_dn, d.inter, d.N_out,
                                      record.payload.get() + record.gu_bytes,
                                      true, scratch);
  return rc;
}
struct Fixture {
  int fd = -1;
  std::vector<Desc> ds;
  std::vector<std::vector<uint8_t>> expected;
  std::vector<int> keys;
  explicit Fixture(size_t n) : keys(2 * n) {
    char path[] = "/tmp/nntr-host-stage-XXXXXX";
    fd = mkstemp(path);
    require(fd >= 0);
    unlink(path);
    size_t offset = 17;
    for (size_t i = 0; i < n; ++i) {
      Desc d{&keys[2 * i], &keys[2 * i + 1], fd, offset, 0, 32, 32, 32};
      std::vector<uint8_t> payload;
      for (unsigned half = 0; half < 2; ++half) {
        const unsigned k = half ? d.inter : d.K;
        const unsigned cols = half ? d.N_out : 2 * d.inter;
        const size_t codes = size_t(k) * cols / 4;
        std::vector<uint8_t> file(codes + 4 + 8 * cols);
        for (size_t b = 0; b < codes; ++b)
          file[b] = static_cast<uint8_t>(i * 31 + b * 13 + half);
        const int8_t palette[] = {-2, 0, 1, 3};
        std::memcpy(file.data() + codes, palette, 4);
        for (unsigned c = 0; c < cols; ++c) {
          const float scale = (c + 1) / 17.0f;
          const float sum = float(int(c % 31) - 15);
          std::memcpy(file.data() + codes + 4 + 4 * c, &scale, 4);
          std::memcpy(file.data() + codes + 4 + 4 * (cols + c), &sum, 4);
        }
        require(pwrite(fd, file.data(), file.size(), offset) ==
                ssize_t(file.size()));
        if (half)
          d.off_dn = offset;
        const size_t old = payload.size();
        payload.resize(old + file.size());
        std::vector<uint8_t> scratch;
        require(!nntrainer::htpReadQs2Compact(
          [&](void *dst, size_t len, uint64_t off) {
            return readAll(fd, dst, len, off);
          },
          offset, k, cols, payload.data() + old, false, scratch));
        offset += file.size() + 13;
      }
      ds.push_back(d);
      expected.push_back(std::move(payload));
    }
  }
  ~Fixture() {
    if (fd >= 0)
      close(fd);
  }
};
static inline void fill(const std::shared_ptr<Bank> &bank, unsigned n = 7) {
  std::vector<std::thread> readers;
  for (unsigned i = 0; i < n; ++i)
    readers.emplace_back([bank] {
      while (bank->readOne(readRecord)) {
      }
    });
  bank->waitReads();
  for (auto &reader : readers)
    reader.join();
}
template <typename F> static void rejected(F f) {
  bool threw = false;
  try {
    f();
  } catch (const std::exception &) {
    threw = true;
  }
  require(threw);
}

#endif
#endif
