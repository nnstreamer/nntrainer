// SPDX-License-Identifier: Apache-2.0
/** @file decode_prefetch_check.cpp
 * @brief Device check for selective completion and reserved-slot recycling.
 * Link against the matching libnntrainer and run with one prefetch reader to
 * exercise queued cancellation as well as reads already in flight.
 */
#include <cassert>
#include <chrono>
#include <compute_ops.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <fcntl.h>
#include <unistd.h>

static void appendWeight(std::vector<uint8_t> &file, unsigned k, unsigned n) {
  file.resize(file.size() + static_cast<size_t>(k) * n / 4, 0);
  file.insert(file.end(), {0, 1, 2, 3});
  for (unsigned part = 0; part < 2; ++part)
    for (unsigned i = 0; i < n; ++i) {
      const float value = part ? 0.0f : 1.0f;
      const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
      file.insert(file.end(), bytes, bytes + sizeof(value));
    }
}

int main() {
  constexpr unsigned K = 2048, I = 1536, N = 2048;
  std::vector<uint8_t> data;
  appendWeight(data, K, 2 * I);
  const size_t down = data.size();
  appendWeight(data, I, N);
  char path[96];
  std::snprintf(path, sizeof(path), "prefetch-check-%ld.bin", (long)getpid());
  const int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
  assert(fd >= 0);
  size_t written = 0;
  while (written < data.size()) {
    const auto n = write(fd, data.data() + written, data.size() - written);
    assert(n > 0);
    written += static_cast<size_t>(n);
  }
  nntrainer::init_backend();
  auto *ops = nntrainer::get_htp_ops();
  ops->reserve_qs4cx_wh_expert_slots(4);
  int keys[8] = {};
  std::vector<nntrainer::ComputeOps::ExpertFileDesc> batch;
  for (unsigned e = 0; e < 4; ++e)
    batch.push_back({keys + 2 * e, keys + 2 * e + 1, fd, 0, down, K, I, N, 2});
  size_t pending_total = 0, cancelled_total = 0;
  for (unsigned iteration = 0; iteration < 80; ++iteration) {
    assert(ops->prefetch_qs4cx_wh_experts_decode(batch));
    std::vector<const void *> needed;
    if (iteration % 4)
      needed.push_back(batch[iteration % 4].key_gu);
    if (iteration % 4 == 2)
      needed.push_back(batch[0].key_gu);
    auto done = ops->prefetch_qs4cx_wh_experts_select(needed);
    pending_total += done.pending;
    cancelled_total += done.cancelled;
    for (const auto *key : needed) {
      bool found = false;
      for (const auto *registered : done.keys)
        found |= registered == key;
      assert(found);
    }
    for (const auto *key : done.keys)
      assert(ops->release_qs4cx_wh_expert(key));
    const auto start = std::chrono::steady_clock::now();
    while (done.pending) {
      const size_t freed = ops->prefetch_qs4cx_wh_experts_reap();
      assert(freed <= done.pending);
      done.pending -= freed;
      assert(std::chrono::steady_clock::now() - start <
             std::chrono::seconds(5));
      if (done.pending)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(ops->prefetch_qs4cx_wh_experts_reap() == 0);
  }
  // A required short read must fail, and return its slot for a valid retry.
  auto invalid = batch[0];
  invalid.off_gu = data.size();
  assert(ops->prefetch_qs4cx_wh_experts_decode({invalid}));
  bool failed = false;
  try {
    ops->prefetch_qs4cx_wh_experts_select({invalid.key_gu});
  } catch (const std::runtime_error &) {
    failed = true;
  }
  assert(failed);
  assert(ops->prefetch_qs4cx_wh_experts_decode(batch));
  for (const auto *key : ops->prefetch_qs4cx_wh_experts_end())
    assert(ops->release_qs4cx_wh_expert(key));
  close(fd);
  assert(unlink(path) == 0);
  std::printf("DECODE PREFETCH PASS: rounds=80 pending=%zu cancelled=%zu\n",
              pending_total, cancelled_total);
}
#else
int main() { return 0; }
#endif
