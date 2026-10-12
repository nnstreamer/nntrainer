// SPDX-License-Identifier: Apache-2.0
/** @file expert_update_check.cpp
 * @brief Device regression: reused QS2 slots change codes, palette and scales.
 * Compare the emitted FP32 file across expert-update modes 0, 1, 2 and 3.
 */
#include <cassert>
#include <chrono>
#include <cmath>
#include <compute_ops.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#include <fcntl.h>
#include <unistd.h>

static void appendWeight(std::vector<uint8_t> &file, unsigned k, unsigned n,
                         unsigned seed) {
  const unsigned code = seed % 4;
  file.resize(file.size() + static_cast<size_t>(k) * n / 4,
              static_cast<uint8_t>(code * 0x55u));
  const int8_t palette[4] = {1, -2, static_cast<int8_t>(seed % 3 + 2), 3};
  file.insert(file.end(), reinterpret_cast<const uint8_t *>(palette),
              reinterpret_cast<const uint8_t *>(palette) + 4);
  for (unsigned part = 0; part < 2; ++part)
    for (unsigned i = 0; i < n; ++i) {
      const float value = part ? static_cast<float>((int)k * palette[code])
                               : .002f * (seed + 1) * (1.f + (i % 7) * .1f);
      const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
      file.insert(file.end(), bytes, bytes + sizeof(value));
    }
}

int main(int argc, char **argv) {
  assert(argc == 2);
  constexpr unsigned K = 128, I = 128, N = 128, E = 8;
  std::vector<uint8_t> data;
  uint64_t gu_off[E], dn_off[E];
  for (unsigned e = 0; e < E; ++e) {
    gu_off[e] = data.size();
    appendWeight(data, K, 2 * I, e);
    dn_off[e] = data.size();
    appendWeight(data, I, N, e + 3);
  }
  char path[80];
  std::snprintf(path, sizeof(path), "expert-update-%ld.bin", (long)getpid());
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
  assert(ops);
  ops->reserve_qs4cx_wh_expert_slots(4);
  int keys[2 * E] = {};
  std::vector<nntrainer::ComputeOps::ExpertFileDesc> desc;
  for (unsigned e = 0; e < E; ++e)
    desc.push_back(
      {keys + 2 * e, keys + 2 * e + 1, fd, gu_off[e], dn_off[e], K, I, N, 2});
  std::vector<unsigned> previous;
  std::vector<float> input(K), output(N);
  for (unsigned i = 0; i < K; ++i)
    input[i] = (static_cast<int>(i % 7) - 2) * .05f;
  FILE *result = std::fopen(argv[1], "wb");
  assert(result);
  size_t discarded = 0;
  for (unsigned round = 0; round < 80; ++round) {
    for (unsigned e : previous)
      assert(ops->release_qs4cx_wh_expert(desc[e].key_gu));
    previous.clear();
    for (unsigned i = 0; i < 4; ++i)
      previous.push_back((round * 3 + i) % E);
    // Validate a failed file read without poisoning the next valid update.
    if (round == 20) {
      auto invalid = desc[previous[0]];
      invalid.off_gu = data.size();
      bool failed = false;
      try {
        ops->register_qs4cx_wh_expert_files({invalid});
      } catch (const std::runtime_error &) {
        failed = true;
      }
      assert(failed);
    }
    if (round % 2) {
      const unsigned wrong = (previous[0] + 5) % E;
      assert(ops->prefetch_qs4cx_wh_experts_decode(
        {desc[previous[0]], desc[wrong]}));
      auto done =
        ops->prefetch_qs4cx_wh_experts_select({desc[previous[0]].key_gu});
      for (const void *key : done.keys)
        if (key != desc[previous[0]].key_gu) {
          assert(ops->release_qs4cx_wh_expert(key));
          ++discarded;
        }
      const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (done.pending) {
        const size_t freed = ops->prefetch_qs4cx_wh_experts_reap();
        assert(freed <= done.pending);
        done.pending -= freed;
        assert(std::chrono::steady_clock::now() < deadline);
        if (done.pending)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    std::vector<nntrainer::ComputeOps::ExpertFileDesc> demand;
    for (unsigned e : previous)
      demand.push_back(desc[e]);
    assert(ops->register_qs4cx_wh_expert_files(demand));
    std::vector<void *> gu, dn;
    for (unsigned e : previous) {
      gu.push_back(keys + 2 * e);
      dn.push_back(keys + 2 * e + 1);
    }
    ops->gemm_qs4cx_moe_layer_fp32(
      gu, std::vector<float *>(4, nullptr), dn,
      std::vector<float *>(4, nullptr), {0, 0, 0, 0}, {1, 1, 1, 1},
      {.1f, .2f, .3f, .4f}, input.data(), output.data(), 1, K, I, N, true);
    for (float v : output)
      assert(std::isfinite(v));
    assert(std::fwrite(output.data(), sizeof(float), N, result) == N);
  }
  for (unsigned e : previous)
    assert(ops->release_qs4cx_wh_expert(desc[e].key_gu));
  // Bootstrap eight physical slots, then combine four ready predictions and
  // four demand misses, exercising the maximum batch of sixteen weight tails.
  assert(ops->register_qs4cx_wh_expert_files(desc));
  for (const auto &d : desc)
    assert(ops->release_qs4cx_wh_expert(d.key_gu));
  assert(ops->prefetch_qs4cx_wh_experts_decode(
    {desc[0], desc[1], desc[2], desc[3]}));
  auto all = ops->prefetch_qs4cx_wh_experts_select(
    {desc[0].key_gu, desc[1].key_gu, desc[2].key_gu, desc[3].key_gu});
  assert(all.keys.size() == 4 && !all.pending);
  assert(
    ops->register_qs4cx_wh_expert_files({desc[4], desc[5], desc[6], desc[7]}));
  std::vector<void *> gu, dn;
  for (unsigned e = 4; e < E; ++e) {
    gu.push_back(keys + 2 * e);
    dn.push_back(keys + 2 * e + 1);
  }
  ops->gemm_qs4cx_moe_layer_fp32(gu, std::vector<float *>(4, nullptr), dn,
                                 std::vector<float *>(4, nullptr), {0, 0, 0, 0},
                                 {1, 1, 1, 1}, {.1f, .2f, .3f, .4f},
                                 input.data(), output.data(), 1, K, I, N, true);
  assert(std::fwrite(output.data(), sizeof(float), N, result) == N);
  assert(std::fclose(result) == 0);
  for (const auto &d : desc)
    assert(ops->release_qs4cx_wh_expert(d.key_gu));
  close(fd);
  assert(unlink(path) == 0);
  std::printf("EXPERT UPDATE PASS: rounds=80 batch8=1 discarded=%zu\n",
              discarded);
}
#else
int main() { return 0; }
#endif
