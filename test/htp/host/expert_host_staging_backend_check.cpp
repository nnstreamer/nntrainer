// SPDX-License-Identifier: Apache-2.0
#include "expert_host_staging_fixture.h"
#include <cstdio>
#if defined(__linux__) || defined(__ANDROID__)
#include <unordered_map>
using namespace nntrainer;
using remote_handle64 = uint64_t;
struct HtpBackend {
  static HtpBackend &global() {
    static HtpBackend x;
    return x;
  }
  uint64_t handle() { return 0; }
};
struct HtpProfile {
  static HtpProfile &global() {
    static HtpProfile x;
    return x;
  }
  unsigned level() { return 0; }
  static uint64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  }
  void addPrefetchRead(uint64_t) {}
};
static unsigned cpuBit() { return 1; }
class Harness {
public:
  using ExpertFileDesc = Desc;
  using HostExpertBank = HtpExpertHostStaging<Desc>;
  struct ArenaEntry {
    unsigned chunk = 0, off = 0, K = 0, N = 0;
  };
  struct ExpertSlot {
    unsigned chunk = 0, off = 0;
  };
  struct Buffer {
    std::vector<uint8_t> storage;
    explicit Buffer(size_t n) : storage(n, 0xcd) {}
    uint8_t *data() { return storage.data(); }
  };
  struct Arena {
    std::unique_ptr<Buffer> buf;
  };
#include "expert_host_staging_staged_type.inc"
  struct PrefetchBatch {
    std::vector<StagedExpert> experts;
    bool decode = false;
    size_t done = 0;
    uint64_t done_mask = 0;
  };
  struct Jobs : std::deque<std::pair<PrefetchBatch *, size_t>> {
    int failure = -1;
    void emplace_back(PrefetchBatch *b, size_t i) {
      if (failure == 0)
        throw std::bad_alloc();
      if (failure > 0)
        --failure;
      std::deque<std::pair<PrefetchBatch *, size_t>>::emplace_back(b, i);
    }
  } prefetch_jobs_;
  std::vector<Arena> arena_chunks_;
  std::vector<ExpertSlot> free_expert_slots_;
  std::unordered_map<const void *, int> experts_;
  size_t expert_slots_wanted_ = 160, expert_slots_placed_ = 160;
  std::mutex handle_mutex_, prefetch_mutex_, host_scope_mutex_;
  std::condition_variable prefetch_cv_, prefetch_done_cv_;
  bool prefetch_stop_ = false;
  std::vector<std::thread> prefetch_readers_;
  std::deque<std::unique_ptr<PrefetchBatch>> prefetch_batches_,
    prefetch_retired_;
  std::shared_ptr<HostExpertBank> host_expert_bank_;
  std::atomic<unsigned> prefetch_reader_cpus_{0};
  uint64_t host_expert_epoch_ = 0;
  std::atomic<uint64_t> host_expert_wait_us_{0};
  std::atomic<unsigned> slots_taken{0}, reads{0};
  bool compact = true, direct = true, prepared = false, tl2 = false;
  bool compactExperts() { return compact; }
  bool weightDirectIO() { return direct; }
  const char *preparedQs2Dir() { return prepared ? "x" : nullptr; }
  const char *preparedTl2File() { return nullptr; }
  int ternaryMode() { return tl2 ? 1 : 0; }
  std::atomic<unsigned> prepared_tl2_kind_{0};
  void startPrefetchReaders() {
    if (!prefetch_readers_.empty())
      return;
    for (int i = 0; i < 7; ++i)
      prefetch_readers_.emplace_back([this] { prefetchReaderLoop(); });
  }
  ExpertSlot takeExpertSlot(remote_handle64, const Desc &) {
    require(!free_expert_slots_.empty());
    auto slot = free_expert_slots_.back();
    free_expert_slots_.pop_back();
    ++slots_taken;
    return slot;
  }
  static unsigned expertStride(unsigned k, unsigned n, unsigned bits) {
    return ((uint64_t(k) * n * bits / 8 + 8 * n + 4 + 4095) & ~uint64_t(4095));
  }
  int readWeight(int fd, uint64_t off, unsigned k, unsigned n, uint8_t *dst,
                 ArenaEntry &e, bool, unsigned = 2, bool = false, bool = false,
                 unsigned = 0) {
    ++reads;
    std::vector<uint8_t> scratch;
    int rc = htpReadQs2Compact(
      [&](void *p, size_t len, uint64_t at) { return readAll(fd, p, len, at); },
      off, k, n, dst, true, scratch);
    if (!rc) {
      e.K = k;
      e.N = n;
    }
    return rc;
  }
  void releaseFinishedSlots(size_t n) {
    require(arena_chunks_.empty());
    arena_chunks_.push_back({std::make_unique<Buffer>(8192 * n)});
    for (size_t i = 0; i < n; ++i)
      free_expert_slots_.push_back({0, unsigned(i * 8192)});
  }
  ~Harness() {
    if (auto b = hostExpertBank())
      end_qs2_host_prefetch(b->token());
    {
      std::lock_guard<std::mutex> l(prefetch_mutex_);
      prefetch_stop_ = true;
    }
    prefetch_cv_.notify_all();
    for (auto &t : prefetch_readers_)
      t.join();
  }

#include "expert_host_staging_backend_methods.inc"
};
int main() {
  Fixture f(160);
  {
    Harness h;
    require(h.begin_qs2_host_prefetch(f.ds, &f, 0) == 0);
    require(h.begin_qs2_host_prefetch(f.ds, &f, 161) == 0);
    require(h.reads == 0 && h.slots_taken == 0 && h.prefetch_readers_.empty());
    for (int mode = 0; mode < 6; ++mode) {
#if !NNTR_TEST_HOST_STAGE_HAS_TL2
      if (mode == 3)
        continue;
#endif
      auto ds = f.ds;
      if (mode == 0)
        h.compact = false;
      if (mode == 1)
        h.direct = false;
      if (mode == 2)
        h.prepared = true;
      if (mode == 3)
        h.tl2 = true;
      if (mode == 4)
        ds[0].weight_bits = 4;
      if (mode == 5)
        h.expert_slots_placed_ = 159;
      require(h.begin_qs2_host_prefetch(ds, &f, 64) == 0);
      h.compact = h.direct = true;
      h.prepared = h.tl2 = false;
      h.expert_slots_placed_ = 160;
    }
    auto token = h.begin_qs2_host_prefetch(f.ds, &f, 64);
    require(token != 0);
    require(h.begin_qs2_host_prefetch(f.ds, &f.keys[0], 64) == 0);
    h.end_qs2_host_prefetch(token + 1);
    require(h.hostExpertBank() != nullptr);
    h.waitHostExpertReads();
    require(h.reads == 128 && h.slots_taken == 0);
    h.end_qs2_host_prefetch(token);
    require(h.hostExpertBank() == nullptr);
    auto newer = h.begin_qs2_host_prefetch(f.ds, &f.keys[0], 1);
    require(newer != 0 && newer != token);
    h.end_qs2_host_prefetch(token);
    require(h.hostExpertBank() != nullptr);
    h.end_qs2_host_prefetch(newer);
  }
  // Real stage/read/worker methods reuse each host payload once and keep
  // padding.
  {
    Harness h;
    auto token = h.begin_qs2_host_prefetch(f.ds, &f, 4);
    h.waitHostExpertReads();
    require(h.reads == 8 && h.slots_taken == 0);
    std::vector<Desc> four(f.ds.begin(), f.ds.begin() + 4);
    h.releaseFinishedSlots(4);
    require(h.beginExpertPrefetch(four, false));
    {
      std::unique_lock<std::mutex> lock(h.prefetch_mutex_);
      auto &b = *h.prefetch_batches_.front();
      h.prefetch_done_cv_.wait(lock,
                               [&] { return b.done == b.experts.size(); });
      for (size_t i = 0; i < 4; ++i) {
        auto &e = b.experts[i];
        require(e.rc == 0 && !e.host && e.gu.K == 32 && e.gu.N == 64 &&
                e.dn.K == 32 && e.dn.N == 32);
        require(std::equal(f.expected[i].begin(), f.expected[i].begin() + 1028,
                           e.base));
        require(std::equal(f.expected[i].begin() + 1028, f.expected[i].end(),
                           e.base + 4096));
        require(e.base[1028] == 0xcd && e.base[4095] == 0xcd &&
                e.base[4096 + 516] == 0xcd);
      }
      h.prefetch_batches_.pop_front();
    }
    require(h.reads == 8 && h.slots_taken == 4);
    h.end_qs2_host_prefetch(token);
  }
  // Inject every actual beginExpertPrefetch queue failure: all slots and leases
  // return, no jobs survive, and end() must never hang during stack unwinding.
  for (int failure = 0; failure < 4; ++failure) {
    Harness h;
    auto token = h.begin_qs2_host_prefetch(f.ds, &f, 4);
    h.waitHostExpertReads();
    h.releaseFinishedSlots(4);
    h.prefetch_jobs_.failure = failure;
    std::vector<Desc> four(f.ds.begin(), f.ds.begin() + 4);
    rejected([&] { h.beginExpertPrefetch(four, false); });
    require(h.prefetch_jobs_.empty() && h.prefetch_batches_.empty() &&
            h.free_expert_slots_.size() == 4);
    h.end_qs2_host_prefetch(token);
  }
  std::printf("PASS extracted backend: %zu assertions, disabled/stale scope, "
              "actual worker copies and enqueue rollback\n",
              checks);
  return 0;
}

#else
int main() {
  std::puts("SKIP: host staging reads require Linux or Android");
  return 0;
}
#endif
