// SPDX-License-Identifier: Apache-2.0
/** @file htp_expert_host_staging.h
 * @brief Bounded ordinary-memory expert reads before DSP slots become free.
 */
#ifndef __NNTRAINER_HTP_EXPERT_HOST_STAGING_H__
#define __NNTRAINER_HTP_EXPERT_HOST_STAGING_H__

#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nntrainer {

/** Publish a batch atomically while the caller excludes queue consumers.
 * On an allocation failure the caller regains the batch and can return slots
 * and release host-copy leases outside the queue lock. */
template <typename Batch, typename Batches, typename Jobs>
void htpPublishExpertBatch(std::unique_ptr<Batch> &batch, Batches &batches,
                           Jobs &jobs) {
  Batch *ptr = batch.get();
  batches.push_back(std::move(batch));
  try {
    for (size_t i = 0; i < ptr->experts.size(); ++i)
      jobs.emplace_back(ptr, i);
  } catch (...) {
    while (!jobs.empty() && jobs.back().first == ptr)
      jobs.pop_back();
    batch = std::move(batches.back());
    batches.pop_back();
    throw;
  }
}

/** No DSP objects or slots belong to this bank. Readers own each payload until
 * completion, and a one-use lease retains it until the normal slot copy ends.
 * The caller queues readOne on its existing reader pool, then waitReads before
 * queuing any jobs that consume leases. Cancellation never interrupts pread.
 */
template <typename Descriptor>
class HtpExpertHostStaging
  : public std::enable_shared_from_this<HtpExpertHostStaging<Descriptor>> {
  // Some backend branches carry an optional alternate on-disk encoding flag.
  // Keep this helper usable by the public compact-QS2 descriptor as well.
  template <typename T, typename = void>
  struct HasPackedFlag : std::false_type {};
  template <typename T>
  struct HasPackedFlag<
    T, std::void_t<decltype(std::declval<const T &>().tl2_packed)>>
    : std::true_type {};
  static bool packed(const Descriptor &d) {
    if constexpr (HasPackedFlag<Descriptor>::value)
      return d.tl2_packed;
    return false;
  }
  struct Source {
    int fd = -1;
    uint64_t size = 0;
    explicit Source(int original) {
#if defined(__linux__) || defined(__ANDROID__)
      struct stat before {
      }, after{};
      if (original < 0 || fstat(original, &before) || before.st_size < 0 ||
          !S_ISREG(before.st_mode))
        throw std::invalid_argument("invalid host expert source");
      fd = fcntl(original, F_DUPFD_CLOEXEC, 0);
      if (fd < 0)
        throw std::runtime_error("cannot retain host expert source");
      if (fstat(fd, &after) || before.st_dev != after.st_dev ||
          before.st_ino != after.st_ino || before.st_size != after.st_size) {
        close(fd);
        fd = -1;
        throw std::runtime_error("host expert source changed");
      }
      size = static_cast<uint64_t>(after.st_size);
#else
      (void)original;
      throw std::invalid_argument("host expert reads are unavailable");
#endif
    }
    Source(const Source &) = delete;
    Source &operator=(const Source &) = delete;
    ~Source() {
#if defined(__linux__) || defined(__ANDROID__)
      if (fd >= 0)
        close(fd);
#endif
    }
  };

  class LeaseKey {
    LeaseKey() = default;
    friend class HtpExpertHostStaging;
  };

public:
  struct Record {
    Descriptor d;
    std::shared_ptr<Source> source;
    size_t gu_bytes, dn_bytes;
    std::unique_ptr<uint8_t[]> payload;
    int rc = 0;
    bool taken = false;
    int fd() const { return source->fd; }
  };
  struct Stats {
    size_t count = 0, read = 0, taken = 0, cancelled = 0, failed = 0;
    size_t bytes = 0;
  };
  struct Lease {
    std::shared_ptr<HtpExpertHostStaging> bank;
    size_t index;
    Lease(LeaseKey, std::shared_ptr<HtpExpertHostStaging> owner, size_t at) :
      bank(std::move(owner)), index(at) {}
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    Lease(Lease &&) = delete;
    Lease &operator=(Lease &&) = delete;
    ~Lease() {
      std::unique_ptr<uint8_t[]> payload;
      {
        std::lock_guard<std::mutex> lock(bank->mutex_);
        payload = std::move(bank->records_[index].payload);
      }
      // The final shared lease follows every GU/DN copy. Free its ordinary
      // payload while later graph work runs, without holding the bank lock.
      // Keep the lease counted until free completes so cancellation joins it.
      payload.reset();
      {
        std::lock_guard<std::mutex> lock(bank->mutex_);
        --bank->leases_;
      }
      bank->cv_.notify_all();
    }
    const Record &record() const { return bank->records_[index]; }
    int copyPart(bool down, uint8_t *destination, size_t capacity) const {
      const auto &r = record();
      if (r.rc)
        return r.rc;
      const size_t count = down ? r.dn_bytes : r.gu_bytes;
      if (!destination || capacity < count)
        return EINVAL;
      std::memcpy(destination, r.payload.get() + (down ? r.gu_bytes : 0),
                  count);
      return 0;
    }
  };

  HtpExpertHostStaging(const std::vector<Descriptor> &descriptors,
                       uint64_t token, const void *owner) :
    token_(token), owner_(owner) {
    if (!token || !owner || descriptors.empty() || descriptors.size() > 160)
      throw std::invalid_argument("invalid host expert bank");
    std::map<int, std::shared_ptr<Source>> sources;
    records_.reserve(descriptors.size());
    for (const auto &d : descriptors) {
      if (!d.key_gu || !d.key_dn || d.key_gu == d.key_dn ||
          d.weight_bits != 2 || packed(d) || !d.K || !d.inter || !d.N_out ||
          d.K > 8192 || d.inter > 8192 || d.N_out > 8192 || d.K % 32 ||
          d.inter % 32 || d.N_out % 32)
        throw std::invalid_argument("unsupported host expert descriptor");
      for (const auto &old : records_)
        if (d.key_gu == old.d.key_gu || d.key_dn == old.d.key_dn ||
            d.key_gu == old.d.key_dn || d.key_dn == old.d.key_gu)
          throw std::invalid_argument("duplicate host expert descriptor");
      const size_t gu = bytes(d.K, 2u * d.inter);
      const size_t dn = bytes(d.inter, d.N_out);
      if (gu + dn > (size_t(384) << 20) - stats_.bytes)
        throw std::invalid_argument("host expert bank exceeds byte limit");
      auto &source = sources[d.fd];
      if (!source)
        source = std::make_shared<Source>(d.fd);
      if (d.off_gu > source->size || gu > source->size - d.off_gu ||
          d.off_dn > source->size || dn > source->size - d.off_dn)
        throw std::invalid_argument("host expert record exceeds source");
      records_.push_back({d, source, gu, dn, {}, 0, false});
      stats_.bytes += gu + dn;
    }
    stats_.count = records_.size();
  }

  uint64_t token() const { return token_; }
  const void *owner() const { return owner_; }
  bool hasWork() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !closing_ && next_ < records_.size();
  }

  /** A failed allocation/read stays failed and is reported by the eventual
   * consumer; it never exposes partial bytes as a valid registered weight. */
  template <typename Reader> bool readOne(Reader reader) {
    size_t at;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closing_ || next_ == records_.size())
        return false;
      at = next_++;
      ++active_;
    }
    Record &record = records_[at];
    try {
      record.payload.reset(new uint8_t[record.gu_bytes + record.dn_bytes]);
      record.rc = reader(record);
    } catch (const std::bad_alloc &) {
      record.rc = ENOMEM;
    } catch (...) {
      record.rc = EIO;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --active_;
      ++done_;
      ++stats_.read;
      stats_.failed += record.rc != 0;
    }
    cv_.notify_all();
    return true;
  }

  void waitReads() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return done_ == records_.size(); });
  }

  std::shared_ptr<Lease> take(const Descriptor &d) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closing_ || done_ != records_.size())
      return {};
    for (size_t i = 0; i < records_.size(); ++i) {
      auto &r = records_[i];
      if (r.taken || !same(r.d, d))
        continue;
      auto lease =
        std::make_shared<Lease>(LeaseKey{}, this->shared_from_this(), i);
      r.taken = true;
      ++leases_;
      ++stats_.taken;
      return lease;
    }
    return {};
  }

  /** Stop unstarted reads and join running reads/copies. The owner must keep
   * the ordinary reader pool alive while leases are outstanding. */
  void cancelAndWait() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    closing_ = true;
    const size_t cancelled = records_.size() - next_;
    stats_.cancelled += cancelled;
    done_ += cancelled;
    next_ = records_.size();
    cv_.notify_all();
    cv_.wait(lock, [&] { return active_ == 0 && leases_ == 0; });
    // A reader may still hold a shared_ptr while returning to its idle loop.
    // Release large payloads and file leases here, inside the measured scope.
    for (auto &record : records_) {
      record.payload.reset();
      record.source.reset();
    }
  }
  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

private:
  static size_t bytes(uint32_t k, uint32_t n) {
    return static_cast<size_t>(uint64_t(k) * n / 4u + 8u * n + 4u);
  }
  static bool same(const Descriptor &a, const Descriptor &b) {
    return a.key_gu == b.key_gu && a.key_dn == b.key_dn && a.fd == b.fd &&
           a.off_gu == b.off_gu && a.off_dn == b.off_dn && a.K == b.K &&
           a.inter == b.inter && a.N_out == b.N_out &&
           a.weight_bits == b.weight_bits && packed(a) == packed(b);
  }
  uint64_t token_;
  const void *owner_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<Record> records_;
  size_t next_ = 0, active_ = 0, done_ = 0, leases_ = 0;
  bool closing_ = false;
  Stats stats_;
};
} // namespace nntrainer
#endif
