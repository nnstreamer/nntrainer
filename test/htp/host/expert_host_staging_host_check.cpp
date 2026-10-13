// SPDX-License-Identifier: Apache-2.0
#include "expert_host_staging_fixture.h"
#include <cstdio>
#if defined(__linux__) || defined(__ANDROID__)
int main() {
  Fixture fixture(160);
  for (size_t n :
       {size_t(1), size_t(64), size_t(96), size_t(128), size_t(160)}) {
    std::vector<Desc> ds(fixture.ds.begin(), fixture.ds.begin() + n);
    auto bank = std::make_shared<Bank>(ds, 7, &fixture);
    require(bank->token() == 7 && bank->owner() == &fixture);
    require(!bank->take(ds[0]));
    fill(bank);
    Desc absent = ds[0];
    absent.off_gu += 1;
    require(!bank->take(absent));
    for (size_t i = 0; i < n; ++i) {
      auto lease = bank->take(ds[i]);
      require(bool(lease) && !bank->take(ds[i]));
      const auto &record = lease->record();
      std::vector<uint8_t> slot(8192, 0xa5);
      require(!lease->copyPart(false, slot.data(), record.gu_bytes));
      require(!lease->copyPart(true, slot.data() + 4096, record.dn_bytes));
      require(std::equal(slot.begin(), slot.begin() + record.gu_bytes,
                         fixture.expected[i].begin()));
      require(std::equal(slot.begin() + 4096,
                         slot.begin() + 4096 + record.dn_bytes,
                         fixture.expected[i].begin() + record.gu_bytes));
      require(std::all_of(slot.begin() + record.gu_bytes, slot.begin() + 4096,
                          [](uint8_t x) { return x == 0xa5; }));
      require(lease->copyPart(false, slot.data(), record.gu_bytes - 1) ==
              EINVAL);
    }
    bank->cancelAndWait();
    const auto stats = bank->stats();
    require(stats.count == n && stats.read == n && stats.taken == n &&
            stats.failed == 0 && stats.cancelled == 0);
  }
  // Mainline descriptor has no optional packed-encoding member.
  {
    struct PlainDesc {
      const void *key_gu, *key_dn;
      int fd;
      size_t off_gu, off_dn;
      unsigned K, inter, N_out, weight_bits;
    };
    const auto &d = fixture.ds[0];
    PlainDesc plain{d.key_gu, d.key_dn, d.fd,    d.off_gu,     d.off_dn,
                    d.K,      d.inter,  d.N_out, d.weight_bits};
    using PlainBank = nntrainer::HtpExpertHostStaging<PlainDesc>;
    auto bank =
      std::make_shared<PlainBank>(std::vector<PlainDesc>{plain}, 6, &fixture);
    bank->readOne([](PlainBank::Record &record) {
      std::vector<uint8_t> scratch;
      auto read = [&](void *dst, size_t count, uint64_t at) {
        return readAll(record.fd(), dst, count, at);
      };
      const auto &d = record.d;
      int rc = nntrainer::htpReadQs2Compact(
        read, d.off_gu, d.K, 2 * d.inter, record.payload.get(), true, scratch);
      return rc ? rc
                : nntrainer::htpReadQs2Compact(
                    read, d.off_dn, d.inter, d.N_out,
                    record.payload.get() + record.gu_bytes, true, scratch);
    });
    auto lease = bank->take(plain);
    require(lease && lease->record().rc == 0);
    require(std::equal(fixture.expected[0].begin(), fixture.expected[0].end(),
                       lease->record().payload.get()));
    lease.reset();
    bank->cancelAndWait();
  }
  // No read can begin after cancellation; this also covers an unused scope.
  {
    auto bank = std::make_shared<Bank>(fixture.ds, 8, &fixture);
    bank->cancelAndWait();
    require(!bank->readOne(readRecord));
    require(bank->stats().read == 0 && bank->stats().cancelled == 160);
    require(!bank->take(fixture.ds[0]));
  }
  // Cancellation blocks until an in-progress read releases the file/payload.
  {
    auto bank = std::make_shared<Bank>(fixture.ds, 9, &fixture);
    std::promise<void> started, release;
    auto gate = release.get_future().share();
    std::thread reader([&] {
      bank->readOne([&](Bank::Record &r) {
        started.set_value();
        gate.wait();
        return readRecord(r);
      });
    });
    started.get_future().wait();
    auto done = std::async(std::launch::async, [&] { bank->cancelAndWait(); });
    require(done.wait_for(std::chrono::milliseconds(20)) ==
            std::future_status::timeout);
    release.set_value();
    reader.join();
    done.get();
    require(bank->stats().read == 1 && bank->stats().cancelled == 159);
  }
  // A promotion lease keeps the bank alive until all GU/DN copies finish.
  {
    std::vector<Desc> one{fixture.ds[0]};
    auto bank = std::make_shared<Bank>(one, 10, &fixture);
    fill(bank, 1);
    auto lease = bank->take(one[0]);
    const auto *retained_record = &lease->record();
    const int retained_fd = retained_record->fd();
    auto done = std::async(std::launch::async, [&] { bank->cancelAndWait(); });
    require(done.wait_for(std::chrono::milliseconds(20)) ==
            std::future_status::timeout);
    lease.reset();
    done.get();
    require(!retained_record->payload && !retained_record->source);
    require(fcntl(retained_fd, F_GETFD) == -1 && errno == EBADF);
    require(!bank->take(one[0]));
  }
  // Partial reads, malformed metadata, allocation exceptions never publish.
  for (int mode = 0; mode < 4; ++mode) {
    std::vector<Desc> one{fixture.ds[0]};
    auto bank = std::make_shared<Bank>(one, 20 + mode, &fixture);
    bank->readOne([&](Bank::Record &r) -> int {
      if (mode == 0) {
        r.payload[0] = 7;
        return EIO;
      }
      if (mode == 1)
        throw std::bad_alloc();
      if (mode == 2)
        throw std::runtime_error("injected read failure");
      std::vector<uint8_t> scratch;
      return nntrainer::htpReadQs2Compact(
        [](void *dst, size_t n, uint64_t) {
          std::memset(dst, 127, n);
          return 0;
        },
        0, 32, 64, r.payload.get(), true, scratch);
    });
    auto lease = bank->take(one[0]);
    std::vector<uint8_t> untouched(2048, 0x7e);
    require(lease->copyPart(false, untouched.data(), untouched.size()) != 0);
    require(std::all_of(untouched.begin(), untouched.end(),
                        [](uint8_t b) { return b == 0x7e; }));
    lease.reset();
    bank->cancelAndWait();
    require(bank->stats().failed == 1);
  }
  // Own the source fd: closing/reusing the borrowed integer cannot change
  // bytes.
  {
    Fixture own(1);
    auto bank = std::make_shared<Bank>(own.ds, 30, &own);
    const int original = own.fd;
    close(own.fd);
    own.fd = -1;
    const int replacement = open("/dev/zero", O_RDONLY);
    require(replacement >= 0);
    if (replacement != original)
      require(dup2(replacement, original) == original);
    fill(bank, 1);
    auto lease = bank->take(own.ds[0]);
    require(std::equal(own.expected[0].begin(), own.expected[0].end(),
                       lease->record().payload.get()));
    lease.reset();
    bank->cancelAndWait();
    close(replacement);
    if (replacement != original)
      close(original);
  }
  // Input rejection is complete before any background jobs can be published.
  rejected([&] { Bank b({}, 1, &fixture); });
  rejected([&] { Bank b(fixture.ds, 0, &fixture); });
  rejected([&] { Bank b(fixture.ds, 1, nullptr); });
  for (unsigned mode = 0; mode < 7; ++mode) {
    auto ds = std::vector<Desc>{fixture.ds[0]};
    if (mode == 0)
      ds[0].weight_bits = 4;
    if (mode == 1)
      ds[0].tl2_packed = true;
    if (mode == 2)
      ds[0].K = 31;
    if (mode == 3)
      ds[0].off_gu = size_t(-1);
    if (mode == 4)
      ds[0].fd = -1;
    if (mode == 5)
      ds.push_back(ds[0]);
    if (mode == 6)
      ds.resize(161, ds[0]);
    rejected([&] { Bank b(ds, 1, &fixture); });
  }
  // Actual queue publication helper rolls back every partially queued job.
  // The batch retains its leases so caller-side slot cleanup can unwind safely.
  {
    struct Batch {
      std::vector<std::shared_ptr<Bank::Lease>> experts;
    };
    struct Jobs : std::deque<std::pair<Batch *, size_t>> {
      int remaining;
      explicit Jobs(int n) : remaining(n) {}
      void emplace_back(Batch *b, size_t at) {
        if (remaining-- == 0)
          throw std::bad_alloc();
        std::deque<std::pair<Batch *, size_t>>::emplace_back(b, at);
      }
    };
    for (int failure = 0; failure < 4; ++failure) {
      std::vector<Desc> ds(fixture.ds.begin(), fixture.ds.begin() + 4);
      auto bank = std::make_shared<Bank>(ds, 35 + failure, &fixture);
      fill(bank);
      auto batch = std::make_unique<Batch>();
      for (auto d : ds)
        batch->experts.push_back(bank->take(d));
      Batch existing;
      std::deque<std::unique_ptr<Batch>> batches;
      Jobs jobs(failure);
      jobs.push_back({&existing, 99});
      rejected([&] { nntrainer::htpPublishExpertBatch(batch, batches, jobs); });
      require(batch && batches.empty() && jobs.size() == 1 &&
              jobs.front().first == &existing);
      batch.reset();
      bank->cancelAndWait();
    }
  }
  // Target payload accounting is exact, and aggregate capacity is bounded.
  {
    Fixture sparse(1);
    require(ftruncate(sparse.fd, 128u << 20) == 0);
    std::vector<Desc> target;
    for (unsigned i = 0; i < 160; ++i) {
      auto d = fixture.ds[i];
      d.fd = sparse.fd;
      d.K = 2048;
      d.inter = 1536;
      d.N_out = 2048;
      d.off_gu = 0;
      d.off_dn = 1597444;
      target.push_back(d);
    }
    auto bank = std::make_shared<Bank>(target, 39, &sparse);
    require(bank->stats().bytes == 384042240u);
    bank->cancelAndWait();
    for (auto &d : target) {
      d.K = 8192;
      d.inter = 8192;
      d.N_out = 8192;
      d.off_dn = 64u << 20;
    }
    target.resize(8);
    rejected([&] { Bank invalid(target, 40, &sparse); });
  }
  // Fresh owner/token scopes neither consume nor retain a previous payload.
  for (unsigned iteration = 0; iteration < 12; ++iteration) {
    auto ds = std::vector<Desc>{fixture.ds[iteration]};
    auto bank =
      std::make_shared<Bank>(ds, 40 + iteration, &fixture.keys[iteration]);
    fill(bank, 1);
    require(bank->token() == 40 + iteration &&
            bank->owner() == &fixture.keys[iteration]);
    auto lease = bank->take(ds[0]);
    require(bool(lease));
    lease.reset();
    bank->cancelAndWait();
  }
  std::printf("PASS host staging: %zu assertions, actual QS2 validator, "
              "readers=7, bounds=160\n",
              checks);
  return 0;
}

#else
int main() {
  std::puts("SKIP: host staging reads require Linux or Android");
  return 0;
}
#endif
